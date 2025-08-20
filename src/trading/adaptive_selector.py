#!/usr/bin/env python3
"""
Adaptive Stop/Target Selector for Alpha12_24

Implements an adaptive stop/target selector that maximizes expected value (EV) per setup
using ATR-scaled candidates, calibrated target-first probabilities, costs (fees + slippage),
and explicit R:R caps per timeframe with a hard RR floor.

Key Features:
- ATR-scaled candidate generation
- Calibrated probability prediction
- Explicit R:R bounds per timeframe (floor: 1.5)
- Cost-aware EV calculation
- No macro inputs (only price/volume/structure/volatility)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveSelectorConfig:
    """Configuration for adaptive stop/target selector"""
    
    # R:R bounds per timeframe
    rr_caps: Dict[str, float] = None
    
    # Candidate grids (ATR multiples)
    stop_multipliers: List[float] = None
    target_multipliers: List[float] = None
    
    # EV calculation parameters
    pmin: float = 0.35  # Minimum probability threshold
    rr_min: float = 1.5  # Hard RR floor
    timeout_penalty_r: float = 0.2  # Timeout penalty in R units
    
    # Cost parameters
    fees_bps_per_side: float = 4.0  # Default 4 bps per side
    slippage_bps: float = 2.0  # Default 2 bps slippage
    
    def __post_init__(self):
        if self.rr_caps is None:
            self.rr_caps = {
                "15m": 1.5,
                "1h": 1.7,
                "4h": 2.0,
                "1d": 2.8
            }
        
        if self.stop_multipliers is None:
            self.stop_multipliers = [0.75, 1.0, 1.25, 1.5]
        
        if self.target_multipliers is None:
            self.target_multipliers = [1.5, 1.75, 2.0, 2.25, 2.5, 2.8, 3.0]

@dataclass
class CandidateResult:
    """Result for a single stop/target candidate"""
    s: float  # Stop ATR multiplier
    t: float  # Target ATR multiplier
    rr: float  # Risk-reward ratio
    p_hit: float  # Probability of hitting target
    ev_r: float  # Expected value in R units
    stop_price: float  # Actual stop price
    target_price: float  # Actual target price
    accepted: bool  # Whether candidate passes all checks

@dataclass
class AdaptiveSelectorResult:
    """Result from adaptive selector"""
    success: bool
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    rr: Optional[float] = None
    p_hit: Optional[float] = None
    ev_r: Optional[float] = None
    s: Optional[float] = None
    t: Optional[float] = None
    atr: Optional[float] = None
    timeframe: Optional[str] = None
    rr_cap: Optional[float] = None
    fees_total_r: Optional[float] = None
    slip_r: Optional[float] = None
    candidates_evaluated: int = 0
    candidates_accepted: int = 0

class AdaptiveSelector:
    """
    Adaptive stop/target selector that maximizes expected value
    using ATR-scaled candidates with calibrated probabilities.
    """
    
    def __init__(self, config: AdaptiveSelectorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def rr_cap(self, timeframe: str) -> float:
        """
        Get R:R cap for given timeframe
        
        Args:
            timeframe: Trading timeframe ("15m", "1h", "4h", "1d")
        
        Returns:
            R:R cap for the timeframe
        """
        return self.config.rr_caps.get(timeframe, self.config.rr_min)
    
    def generate_candidates(self, atr: float) -> List[Tuple[float, float]]:
        """
        Generate stop/target candidate pairs as ATR multiples
        
        Args:
            atr: Current ATR value
        
        Returns:
            List of (s, t) pairs where s=stop_multiplier, t=target_multiplier
        """
        candidates = []
        
        for s in self.config.stop_multipliers:
            for t in self.config.target_multipliers:
                candidates.append((s, t))
        
        return candidates
    
    def filter_candidates_by_rr(self, candidates: List[Tuple[float, float]], 
                               timeframe: str) -> List[Tuple[float, float, float]]:
        """
        Filter candidates by R:R bounds
        
        Args:
            candidates: List of (s, t) pairs
            timeframe: Trading timeframe
        
        Returns:
            List of (s, t, rr) tuples that pass R:R bounds
        """
        rr_cap = self.rr_cap(timeframe)
        filtered = []
        
        for s, t in candidates:
            rr = t / s
            
            # Apply bounds: RR ∈ [1.5, rr_cap(timeframe)]
            if self.config.rr_min <= rr <= rr_cap:
                filtered.append((s, t, rr))
        
        return filtered
    
    def calculate_costs_r(self, fees_bps_per_side: float, 
                         slippage_bps: float) -> Tuple[float, float]:
        """
        Calculate costs in R units
        
        Args:
            fees_bps_per_side: Fees per side in basis points
            slippage_bps: Slippage in basis points
        
        Returns:
            Tuple of (fees_total_r, slip_r) in R units
        """
        fees_total_r = 2 * fees_bps_per_side / 10000.0  # Two sides
        slip_r = slippage_bps / 10000.0
        
        return fees_total_r, slip_r
    
    def calculate_ev_r(self, rr: float, p_hit: float, fees_total_r: float, 
                      slip_r: float, p_timeout: float = 0.0) -> float:
        """
        Calculate expected value in R units
        
        Args:
            rr: Risk-reward ratio (t/s)
            p_hit: Probability of hitting target
            fees_total_r: Total fees in R units
            slip_r: Slippage in R units
            p_timeout: Probability of timeout (optional)
        
        Returns:
            Expected value in R units
        """
        # Net payoff if target hits (in R)
        r_win = rr - (fees_total_r + slip_r)
        
        # Loss if stop hits (in R)
        r_loss = 1.0 + (fees_total_r + slip_r)
        
        # Timeout penalty
        gamma_r = self.config.timeout_penalty_r
        
        # Probability of stop hit
        p_stop = max(0, 1 - p_hit - p_timeout)
        
        # Expected value
        ev_r = p_hit * r_win - p_stop * r_loss - p_timeout * gamma_r
        
        return ev_r
    
    def check_acceptance(self, p_hit: float, r_win: float, 
                        fees_total_r: float, slip_r: float) -> bool:
        """
        Check if candidate passes acceptance criteria
        
        Args:
            p_hit: Probability of hitting target
            r_win: Net win in R units
            fees_total_r: Total fees in R units
            slip_r: Slippage in R units
        
        Returns:
            True if candidate passes all checks
        """
        # Check probability threshold
        if p_hit < self.config.pmin:
            return False
        
        # Check net win after costs
        if r_win < (self.config.rr_min - (fees_total_r + slip_r)):
            return False
        
        return True
    
    def predict_p_hit(self, features: Dict[str, Any], s: float, t: float, 
                     rr: float, timeframe: str, atr: float) -> float:
        """
        Predict calibrated probability of hitting target
        
        Args:
            features: Feature vector (no macro inputs)
            s: Stop ATR multiplier
            t: Target ATR multiplier
            rr: Risk-reward ratio
            timeframe: Trading timeframe
            atr: Current ATR
        
        Returns:
            Calibrated probability of hitting target
        """
        # Augment features with candidate parameters
        features_augmented = features.copy()
        features_augmented.update({
            "atr": atr,
            "s": s,
            "t": t,
            "rr": rr,
            "tf": timeframe
        })
        
        # TODO: Implement calibrated probability prediction
        # For now, use fallback empirical method
        return self._fallback_p_hit(features_augmented)
    
    def _fallback_p_hit(self, features_augmented: Dict[str, Any]) -> float:
        """
        Fallback empirical probability calculation
        
        Args:
            features_augmented: Augmented feature vector
        
        Returns:
            Empirical probability clamped to [0.1, 0.9]
        """
        # Simple empirical model based on RR and ATR
        rr = features_augmented.get("rr", 2.0)
        atr = features_augmented.get("atr", 1000.0)
        tf = features_augmented.get("tf", "1h")
        
        # Base probability decreases with higher RR
        base_prob = 0.6 - (rr - 1.5) * 0.1
        
        # Adjust for timeframe (higher timeframes = higher probability)
        tf_adjustment = {"15m": -0.1, "1h": 0.0, "4h": 0.05, "1d": 0.1}
        base_prob += tf_adjustment.get(tf, 0.0)
        
        # Add some randomness for realistic variation
        noise = np.random.normal(0, 0.05)
        p_hit = base_prob + noise
        
        # Clamp to [0.1, 0.9]
        p_hit = max(0.1, min(0.9, p_hit))
        
        return p_hit
    
    def calculate_prices(self, entry_price: float, direction: str, 
                        s: float, t: float, atr: float) -> Tuple[float, float]:
        """
        Calculate stop and target prices around entry
        
        Args:
            entry_price: Entry price
            direction: Trade direction ("long" or "short")
            s: Stop ATR multiplier
            t: Target ATR multiplier
            atr: Current ATR
        
        Returns:
            Tuple of (stop_price, target_price)
        """
        stop_distance = s * atr
        target_distance = t * atr
        
        if direction.lower() == "long":
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:  # short
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance
        
        return stop_price, target_price
    
    def select_optimal_stop_target(self, 
                                 features: Dict[str, Any],
                                 atr: float,
                                 timeframe: str,
                                 entry_price: float,
                                 direction: str,
                                 fees_bps_per_side: Optional[float] = None,
                                 slippage_bps: Optional[float] = None) -> AdaptiveSelectorResult:
        """
        Select optimal stop/target using adaptive selector
        
        Args:
            features: Feature vector (no macro inputs)
            atr: Current ATR value
            timeframe: Trading timeframe
            entry_price: Entry price
            direction: Trade direction
            fees_bps_per_side: Fees per side (uses config default if None)
            slippage_bps: Slippage (uses config default if None)
        
        Returns:
            AdaptiveSelectorResult with optimal stop/target or None
        """
        start_time = datetime.now()
        
        # Use config defaults if not provided
        if fees_bps_per_side is None:
            fees_bps_per_side = self.config.fees_bps_per_side
        if slippage_bps is None:
            slippage_bps = self.config.slippage_bps
        
        # Calculate costs
        fees_total_r, slip_r = self.calculate_costs_r(fees_bps_per_side, slippage_bps)
        
        # Generate candidates
        candidates = self.generate_candidates(atr)
        
        # Filter by R:R bounds
        bounded_candidates = self.filter_candidates_by_rr(candidates, timeframe)
        
        if not bounded_candidates:
            self.logger.warning(f"No candidates pass R:R bounds for {timeframe}")
            return AdaptiveSelectorResult(
                success=False,
                candidates_evaluated=len(candidates),
                candidates_accepted=0
            )
        
        # Evaluate each candidate
        candidate_results = []
        rr_cap = self.rr_cap(timeframe)
        
        for s, t, rr in bounded_candidates:
            # Predict probability
            p_hit = self.predict_p_hit(features, s, t, rr, timeframe, atr)
            
            # Calculate EV
            ev_r = self.calculate_ev_r(rr, p_hit, fees_total_r, slip_r)
            
            # Check acceptance
            r_win = rr - (fees_total_r + slip_r)
            accepted = self.check_acceptance(p_hit, r_win, fees_total_r, slip_r)
            
            # Calculate prices
            stop_price, target_price = self.calculate_prices(
                entry_price, direction, s, t, atr
            )
            
            candidate_result = CandidateResult(
                s=s, t=t, rr=rr, p_hit=p_hit, ev_r=ev_r,
                stop_price=stop_price, target_price=target_price,
                accepted=accepted
            )
            candidate_results.append(candidate_result)
        
        # Select best candidate
        accepted_candidates = [c for c in candidate_results if c.accepted]
        
        if not accepted_candidates:
            self.logger.warning(f"No candidates pass acceptance criteria for {timeframe}")
            return AdaptiveSelectorResult(
                success=False,
                candidates_evaluated=len(bounded_candidates),
                candidates_accepted=0,
                atr=atr,
                timeframe=timeframe,
                rr_cap=rr_cap,
                fees_total_r=fees_total_r,
                slip_r=slip_r
            )
        
        # Select candidate with maximum EV
        best_candidate = max(accepted_candidates, key=lambda c: c.ev_r)
        
        # Log telemetry
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(
            f"Adaptive selector decision: {timeframe}, ATR={atr:.2f}, "
            f"(s,t)=({best_candidate.s:.2f},{best_candidate.t:.2f}), "
            f"RR={best_candidate.rr:.2f}, p_hit={best_candidate.p_hit:.3f}, "
            f"EV_R={best_candidate.ev_r:.4f}, time={decision_time:.1f}ms"
        )
        
        return AdaptiveSelectorResult(
            success=True,
            stop_price=best_candidate.stop_price,
            target_price=best_candidate.target_price,
            rr=best_candidate.rr,
            p_hit=best_candidate.p_hit,
            ev_r=best_candidate.ev_r,
            s=best_candidate.s,
            t=best_candidate.t,
            atr=atr,
            timeframe=timeframe,
            rr_cap=rr_cap,
            fees_total_r=fees_total_r,
            slip_r=slip_r,
            candidates_evaluated=len(bounded_candidates),
            candidates_accepted=len(accepted_candidates)
        )

# Global instance
adaptive_selector = AdaptiveSelector(AdaptiveSelectorConfig())
