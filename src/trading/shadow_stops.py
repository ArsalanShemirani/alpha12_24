#!/usr/bin/env python3
"""
Shadow Dynamic Stops - Phase 1

Computes ATR/volatility-aware dynamic stop candidates for telemetry purposes only.
Does NOT modify any live trading decisions - purely for logging and future A/B testing.

Key Features:
- Timeframe-specific ATR-based stop calculation
- Volatility Z-score adjustment for high-vol periods
- Percentage-of-price fallback caps
- Rich telemetry logging with setup tracking
- Zero impact on live trading logic
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration (can be overridden by environment variables)
DEFAULT_CONFIG = {
    # Base ATR multipliers per timeframe
    "STOP_BASE_ATR_15M": 0.90,
    "STOP_BASE_ATR_1H": 1.20,
    "STOP_BASE_ATR_4H": 1.60,
    "STOP_BASE_ATR_1D": 2.00,
    
    # Min/Max ATR multiplier ranges per timeframe
    "STOP_MIN_ATR_15M": 0.60,
    "STOP_MAX_ATR_15M": 1.20,
    "STOP_MIN_ATR_1H": 0.90,
    "STOP_MAX_ATR_1H": 1.60,
    "STOP_MIN_ATR_4H": 1.20,
    "STOP_MAX_ATR_4H": 2.00,
    "STOP_MIN_ATR_1D": 1.60,
    "STOP_MAX_ATR_1D": 2.40,
    
    # Percentage-of-price caps per timeframe
    "STOP_PCT_CAP_15M": 0.0035,  # 0.35%
    "STOP_PCT_CAP_1H": 0.0060,   # 0.60%
    "STOP_PCT_CAP_4H": 0.0100,   # 1.00%
    "STOP_PCT_CAP_1D": 0.0150,   # 1.50%
    
    # Volatility relaxer parameters
    "VOL_Z_RELAX_THRESHOLD": 2.0,
    "VOL_Z_RELAX_FACTOR": 1.15,
    
    # Feature flag
    "SHADOW_DYNAMIC_STOP_LOGGING": 1,
}

@dataclass
class ShadowStopConfig:
    """Configuration for shadow dynamic stop computation"""
    
    # Base ATR multipliers
    base_atr_15m: float = DEFAULT_CONFIG["STOP_BASE_ATR_15M"]
    base_atr_1h: float = DEFAULT_CONFIG["STOP_BASE_ATR_1H"]
    base_atr_4h: float = DEFAULT_CONFIG["STOP_BASE_ATR_4H"]
    base_atr_1d: float = DEFAULT_CONFIG["STOP_BASE_ATR_1D"]
    
    # ATR multiplier ranges
    min_atr_15m: float = DEFAULT_CONFIG["STOP_MIN_ATR_15M"]
    max_atr_15m: float = DEFAULT_CONFIG["STOP_MAX_ATR_15M"]
    min_atr_1h: float = DEFAULT_CONFIG["STOP_MIN_ATR_1H"]
    max_atr_1h: float = DEFAULT_CONFIG["STOP_MAX_ATR_1H"]
    min_atr_4h: float = DEFAULT_CONFIG["STOP_MIN_ATR_4H"]
    max_atr_4h: float = DEFAULT_CONFIG["STOP_MAX_ATR_4H"]
    min_atr_1d: float = DEFAULT_CONFIG["STOP_MIN_ATR_1D"]
    max_atr_1d: float = DEFAULT_CONFIG["STOP_MAX_ATR_1D"]
    
    # Percentage caps
    pct_cap_15m: float = DEFAULT_CONFIG["STOP_PCT_CAP_15M"]
    pct_cap_1h: float = DEFAULT_CONFIG["STOP_PCT_CAP_1H"]
    pct_cap_4h: float = DEFAULT_CONFIG["STOP_PCT_CAP_4H"]
    pct_cap_1d: float = DEFAULT_CONFIG["STOP_PCT_CAP_1D"]
    
    # Volatility relaxer
    vol_z_threshold: float = DEFAULT_CONFIG["VOL_Z_RELAX_THRESHOLD"]
    vol_z_factor: float = DEFAULT_CONFIG["VOL_Z_RELAX_FACTOR"]
    
    # Feature flag
    enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'ShadowStopConfig':
        """Create configuration from environment variables with defaults"""
        kwargs = {}
        
        # Map environment variable names to attribute names
        env_mapping = {
            "STOP_BASE_ATR_15M": "base_atr_15m",
            "STOP_BASE_ATR_1H": "base_atr_1h", 
            "STOP_BASE_ATR_4H": "base_atr_4h",
            "STOP_BASE_ATR_1D": "base_atr_1d",
            "STOP_MIN_ATR_15M": "min_atr_15m",
            "STOP_MAX_ATR_15M": "max_atr_15m",
            "STOP_MIN_ATR_1H": "min_atr_1h",
            "STOP_MAX_ATR_1H": "max_atr_1h",
            "STOP_MIN_ATR_4H": "min_atr_4h",
            "STOP_MAX_ATR_4H": "max_atr_4h",
            "STOP_MIN_ATR_1D": "min_atr_1d",
            "STOP_MAX_ATR_1D": "max_atr_1d",
            "STOP_PCT_CAP_15M": "pct_cap_15m",
            "STOP_PCT_CAP_1H": "pct_cap_1h",
            "STOP_PCT_CAP_4H": "pct_cap_4h",
            "STOP_PCT_CAP_1D": "pct_cap_1d",
            "VOL_Z_RELAX_THRESHOLD": "vol_z_threshold",
            "VOL_Z_RELAX_FACTOR": "vol_z_factor",
        }
        
        for env_var, attr_name in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    kwargs[attr_name] = float(env_value)
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {env_value}, using default")
        
        # Check feature flag
        enabled = os.getenv("SHADOW_DYNAMIC_STOP_LOGGING", "1") == "1"
        kwargs["enabled"] = enabled
        
        return cls(**kwargs)

@dataclass
class ShadowStopResult:
    """Result from shadow dynamic stop computation"""
    setup_id: str
    tf: str
    entry_price: float
    atr14: Optional[float]
    median_atr14_20d: Optional[float]
    vol_z: Optional[float]
    dynamic_stop_candidate_price: Optional[float]
    dynamic_stop_candidate_R: Optional[float]
    applied_stop_price: float
    applied_stop_distance: float
    applied_stop_R: float
    stop_cap_used: Optional[str]
    rr_planned: float
    rr_realized: Optional[float]
    p_hit: Optional[float]
    conf: Optional[float]
    outcome: str
    mfe_R: Optional[float]
    mae_R: Optional[float]
    shadow_valid: bool
    shadow_notes: str
    ts: str

class ShadowStopComputer:
    """
    Computes dynamic stop candidates for shadow logging.
    
    This class performs ATR-based dynamic stop calculations without affecting
    any live trading decisions. Results are logged for future A/B testing.
    """
    
    def __init__(self, config: ShadowStopConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup logging directory
        self.log_dir = Path("runs/shadow_stops")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file for telemetry
        self.telemetry_file = self.log_dir / "shadow_dynamic_stops.csv"
        self._ensure_telemetry_header()
    
    def _ensure_telemetry_header(self):
        """Ensure the telemetry CSV file has the proper header"""
        if not self.telemetry_file.exists():
            fieldnames = [
                'setup_id', 'tf', 'entry_price', 'atr14', 'median_atr14_20d', 
                'vol_z', 'dynamic_stop_candidate_price', 'dynamic_stop_candidate_R',
                'applied_stop_price', 'applied_stop_distance', 'applied_stop_R',
                'stop_cap_used', 'rr_planned', 'rr_realized', 'p_hit', 'conf',
                'outcome', 'mfe_R', 'mae_R', 'shadow_valid', 'shadow_notes', 'ts'
            ]
            
            with open(self.telemetry_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
    
    def get_timeframe_config(self, tf: str) -> Tuple[float, float, float, float]:
        """
        Get timeframe-specific configuration values
        
        Returns:
            Tuple of (base_atr, min_atr, max_atr, pct_cap)
        """
        tf_configs = {
            "15m": (self.config.base_atr_15m, self.config.min_atr_15m, 
                   self.config.max_atr_15m, self.config.pct_cap_15m),
            "1h": (self.config.base_atr_1h, self.config.min_atr_1h,
                   self.config.max_atr_1h, self.config.pct_cap_1h),
            "4h": (self.config.base_atr_4h, self.config.min_atr_4h,
                   self.config.max_atr_4h, self.config.pct_cap_4h),
            "1d": (self.config.base_atr_1d, self.config.min_atr_1d,
                   self.config.max_atr_1d, self.config.pct_cap_1d),
        }
        
        return tf_configs.get(tf, tf_configs["1h"])  # Default to 1h if unknown
    
    def compute_atr_metrics(self, data: pd.DataFrame, tf: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute ATR14 and 20-day median ATR for volatility Z-score
        
        Args:
            data: OHLCV dataframe with sufficient history
            tf: Timeframe string
            
        Returns:
            Tuple of (atr14, median_atr14_20d, vol_z)
        """
        try:
            if len(data) < 14:
                return None, None, None
            
            # Calculate ATR14
            high = data['high']
            low = data['low'] 
            close = data['close']
            prev_close = close.shift(1)
            
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            
            atr14 = tr.rolling(window=14, min_periods=14).mean()
            
            if atr14.dropna().empty:
                return None, None, None
                
            current_atr = float(atr14.iloc[-1])
            
            # Calculate 20-day median ATR for volatility Z-score
            # Need at least 20 + 14 bars for robust median
            if len(data) < 34:
                return current_atr, None, None
                
            # Get last 20 ATR values (excluding current)
            recent_atr = atr14.dropna().iloc[-21:-1]  # Last 20, excluding current
            
            if len(recent_atr) < 20:
                return current_atr, None, None
                
            median_atr = float(recent_atr.median())
            
            # Calculate volatility Z-score
            vol_z = current_atr / median_atr if median_atr > 0 else None
            
            return current_atr, median_atr, vol_z
            
        except Exception as e:
            self.logger.warning(f"Error computing ATR metrics: {e}")
            return None, None, None
    
    def compute_dynamic_stop_candidate(self, 
                                     entry_price: float,
                                     tf: str,
                                     atr14: Optional[float],
                                     vol_z: Optional[float]) -> Tuple[Optional[float], str]:
        """
        Compute dynamic stop candidate price
        
        Args:
            entry_price: Entry price for the setup
            tf: Timeframe string
            atr14: Current ATR14 value
            vol_z: Volatility Z-score
            
        Returns:
            Tuple of (dynamic_stop_candidate, notes)
        """
        if atr14 is None:
            return None, "ATR14 unavailable"
        
        base_atr, min_atr, max_atr, pct_cap = self.get_timeframe_config(tf)
        
        # Step 1: Compute base stop distance
        s_base = base_atr * atr14
        
        # Step 2: Clamp within timeframe band
        s_clamped = np.clip(s_base, min_atr * atr14, max_atr * atr14)
        
        # Step 3: Apply percentage cap
        pct_cap_distance = pct_cap * entry_price
        s_pct_capped = min(s_clamped, pct_cap_distance)
        
        # Step 4: Apply high-vol relaxer if needed
        s_final = s_pct_capped
        relaxer_applied = False
        
        if vol_z is not None and vol_z >= self.config.vol_z_threshold:
            s_final *= self.config.vol_z_factor
            relaxer_applied = True
        
        # Create notes about what was applied
        notes_parts = []
        if s_clamped != s_base:
            notes_parts.append("ATR_clamped")
        if s_pct_capped != s_clamped:
            notes_parts.append("pct_capped")
        if relaxer_applied:
            notes_parts.append(f"vol_relaxed({vol_z:.2f})")
        
        notes = ",".join(notes_parts) if notes_parts else "baseline"
        
        return s_final, notes
    
    def compute_shadow_stop(self,
                          setup_id: str,
                          tf: str,
                          entry_price: float,
                          applied_stop_price: float,
                          rr_planned: float,
                          data: Optional[pd.DataFrame] = None,
                          p_hit: Optional[float] = None,
                          conf: Optional[float] = None,
                          outcome: str = "pending") -> ShadowStopResult:
        """
        Compute shadow dynamic stop candidate and create telemetry record
        
        Args:
            setup_id: Unique setup identifier
            tf: Timeframe string
            entry_price: Entry price for the setup
            applied_stop_price: Stop price chosen by current system
            rr_planned: Planned risk-reward ratio
            data: OHLCV dataframe for ATR computation (optional)
            p_hit: Model probability/confidence (optional)
            conf: Model confidence (optional) 
            outcome: Current outcome status
            
        Returns:
            ShadowStopResult with all computed values
        """
        if not self.config.enabled:
            # Return minimal result if shadow logging is disabled
            return ShadowStopResult(
                setup_id=setup_id,
                tf=tf,
                entry_price=entry_price,
                atr14=None,
                median_atr14_20d=None,
                vol_z=None,
                dynamic_stop_candidate_price=None,
                dynamic_stop_candidate_R=None,
                applied_stop_price=applied_stop_price,
                applied_stop_distance=abs(entry_price - applied_stop_price),
                applied_stop_R=1.0,
                stop_cap_used=None,
                rr_planned=rr_planned,
                rr_realized=None,
                p_hit=p_hit,
                conf=conf,
                outcome=outcome,
                mfe_R=None,
                mae_R=None,
                shadow_valid=False,
                shadow_notes="disabled",
                ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        # Compute ATR metrics if data is available
        atr14, median_atr14_20d, vol_z = None, None, None
        shadow_valid = True
        shadow_notes = []
        
        if data is not None:
            atr14, median_atr14_20d, vol_z = self.compute_atr_metrics(data, tf)
            if atr14 is None:
                shadow_valid = False
                shadow_notes.append("insufficient_data_for_ATR")
        else:
            shadow_valid = False
            shadow_notes.append("no_price_data")
        
        # Compute dynamic stop candidate
        dynamic_stop_candidate = None
        dynamic_stop_candidate_R = None
        
        if atr14 is not None:
            candidate_distance, computation_notes = self.compute_dynamic_stop_candidate(
                entry_price, tf, atr14, vol_z
            )
            
            if candidate_distance is not None:
                dynamic_stop_candidate = candidate_distance
                
                # Compute R units relative to applied stop
                applied_stop_distance = abs(entry_price - applied_stop_price)
                if applied_stop_distance > 1e-9:
                    dynamic_stop_candidate_R = dynamic_stop_candidate / applied_stop_distance
                
                shadow_notes.append(computation_notes)
        
        # Applied stop metrics
        applied_stop_distance = abs(entry_price - applied_stop_price)
        applied_stop_R = 1.0  # By definition
        
        # Combine notes
        final_notes = ";".join(shadow_notes) if shadow_notes else "ok"
        
        # Create result
        result = ShadowStopResult(
            setup_id=setup_id,
            tf=tf,
            entry_price=entry_price,
            atr14=atr14,
            median_atr14_20d=median_atr14_20d,
            vol_z=vol_z,
            dynamic_stop_candidate_price=dynamic_stop_candidate,
            dynamic_stop_candidate_R=dynamic_stop_candidate_R,
            applied_stop_price=applied_stop_price,
            applied_stop_distance=applied_stop_distance,
            applied_stop_R=applied_stop_R,
            stop_cap_used=None,  # TODO: Extract from existing system if available
            rr_planned=rr_planned,
            rr_realized=None,  # Will be filled on trade completion
            p_hit=p_hit,
            conf=conf,
            outcome=outcome,
            mfe_R=None,  # TODO: Compute if price data available
            mae_R=None,  # TODO: Compute if price data available  
            shadow_valid=shadow_valid,
            shadow_notes=final_notes,
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return result
    
    def log_shadow_result(self, result: ShadowStopResult):
        """
        Log shadow stop result to telemetry CSV
        
        Args:
            result: ShadowStopResult to log
        """
        if not self.config.enabled:
            return
            
        try:
            with open(self.telemetry_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                row = [
                    result.setup_id,
                    result.tf,
                    result.entry_price,
                    result.atr14,
                    result.median_atr14_20d,
                    result.vol_z,
                    result.dynamic_stop_candidate_price,
                    result.dynamic_stop_candidate_R,
                    result.applied_stop_price,
                    result.applied_stop_distance,
                    result.applied_stop_R,
                    result.stop_cap_used,
                    result.rr_planned,
                    result.rr_realized,
                    result.p_hit,
                    result.conf,
                    result.outcome,
                    result.mfe_R,
                    result.mae_R,
                    result.shadow_valid,
                    result.shadow_notes,
                    result.ts
                ]
                
                writer.writerow(row)
                
        except Exception as e:
            self.logger.error(f"Failed to log shadow result: {e}")

# Global instance for easy access
_shadow_config = None
_shadow_computer = None

def get_shadow_computer() -> ShadowStopComputer:
    """Get or create the global shadow stop computer instance"""
    global _shadow_config, _shadow_computer
    
    if _shadow_computer is None:
        _shadow_config = ShadowStopConfig.from_env()
        _shadow_computer = ShadowStopComputer(_shadow_config)
    
    return _shadow_computer

def compute_and_log_shadow_stop(setup_id: str,
                               tf: str, 
                               entry_price: float,
                               applied_stop_price: float,
                               rr_planned: float,
                               data: Optional[pd.DataFrame] = None,
                               p_hit: Optional[float] = None,
                               conf: Optional[float] = None,
                               outcome: str = "pending") -> ShadowStopResult:
    """
    Convenience function to compute and log shadow stop in one call
    
    Args:
        setup_id: Unique setup identifier
        tf: Timeframe string
        entry_price: Entry price for the setup
        applied_stop_price: Stop price chosen by current system
        rr_planned: Planned risk-reward ratio
        data: OHLCV dataframe for ATR computation (optional)
        p_hit: Model probability/confidence (optional)
        conf: Model confidence (optional)
        outcome: Current outcome status
        
    Returns:
        ShadowStopResult with all computed values
    """
    computer = get_shadow_computer()
    result = computer.compute_shadow_stop(
        setup_id=setup_id,
        tf=tf,
        entry_price=entry_price,
        applied_stop_price=applied_stop_price,
        rr_planned=rr_planned,
        data=data,
        p_hit=p_hit,
        conf=conf,
        outcome=outcome
    )
    
    computer.log_shadow_result(result)
    return result

