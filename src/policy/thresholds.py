#!/usr/bin/env python3
"""
Threshold policy for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ThresholdPolicy:
    """Threshold policy configuration"""
    prob_long: float = 0.60
    prob_short: float = 0.40
    min_rr: float = 1.8
    confidence_threshold: float = 0.65
    volatility_threshold: float = 0.02


class ThresholdManager:
    """Threshold manager for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
        self.policy = ThresholdPolicy(
            prob_long=config.prob_long,
            prob_short=config.prob_short,
            min_rr=config.min_rr,
            confidence_threshold=config.get('signal.confidence_threshold', 0.65),
            volatility_threshold=config.get('signal.volatility_threshold', 0.02)
        )
    
    def calculate_signal_strength(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate signal strength from probabilities
        
        Args:
            probabilities: Model probabilities [prob_down, prob_up]
        
        Returns:
            Signal strength array
        """
        prob_up = probabilities[:, 1]
        
        # Calculate signal strength based on distance from neutral (0.5)
        signal_strength = np.abs(prob_up - 0.5) * 2  # Scale to [0, 1]
        
        return signal_strength
    
    def determine_signal(self, probabilities: np.ndarray, 
                        volatility: Optional[float] = None) -> Tuple[str, float, Dict]:
        """
        Determine trading signal based on probabilities and thresholds
        
        Args:
            probabilities: Model probabilities [prob_down, prob_up]
            volatility: Current volatility (optional)
        
        Returns:
            Tuple of (signal, confidence, metadata)
        """
        prob_up = probabilities[:, 1]
        prob_down = probabilities[:, 0]
        
        # Calculate signal strength
        signal_strength = self.calculate_signal_strength(probabilities)
        
        # Determine signal
        if prob_up > self.policy.prob_long:
            signal = "LONG"
            confidence = prob_up
        elif prob_down > self.policy.prob_short:
            signal = "SHORT"
            confidence = prob_down
        else:
            signal = "HOLD"
            confidence = max(prob_up, prob_down)
        
        # Check confidence threshold
        if confidence < self.policy.confidence_threshold:
            signal = "HOLD"
        
        # Check volatility threshold if provided
        if volatility is not None and volatility < self.policy.volatility_threshold:
            signal = "HOLD"
        
        metadata = {
            'prob_up': prob_up,
            'prob_down': prob_down,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'volatility': volatility
        }
        
        return signal, confidence, metadata
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                                  take_profit: float) -> float:
        """
        Calculate risk/reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Risk/reward ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return float('inf')
        
        return reward / risk
    
    def validate_signal(self, signal: str, confidence: float, 
                       risk_reward: float, metadata: Dict) -> bool:
        """
        Validate trading signal
        
        Args:
            signal: Trading signal
            confidence: Signal confidence
            risk_reward: Risk/reward ratio
            metadata: Signal metadata
        
        Returns:
            True if signal is valid
        """
        # Must be a valid signal
        if signal not in ["LONG", "SHORT", "HOLD"]:
            return False
        
        # Must meet confidence threshold
        if confidence < self.policy.confidence_threshold:
            return False
        
        # Must meet minimum risk/reward ratio (for non-HOLD signals)
        if signal != "HOLD" and risk_reward < self.policy.min_rr:
            return False
        
        # Must have sufficient signal strength
        signal_strength = metadata.get('signal_strength', 0)
        if signal_strength < 0.1:  # Minimum signal strength
            return False
        
        return True
    
    def get_signal_metadata(self, signal: str, confidence: float, 
                           metadata: Dict) -> Dict:
        """
        Get formatted signal metadata
        
        Args:
            signal: Trading signal
            confidence: Signal confidence
            metadata: Raw metadata
        
        Returns:
            Formatted metadata
        """
        formatted_metadata = {
            'signal': signal,
            'confidence': confidence,
            'prob_up': metadata.get('prob_up', 0),
            'prob_down': metadata.get('prob_down', 0),
            'signal_strength': metadata.get('signal_strength', 0),
            'volatility': metadata.get('volatility', 0),
            'timestamp': pd.Timestamp.now()
        }
        
        return formatted_metadata
    
    def adjust_thresholds(self, market_conditions: Dict) -> None:
        """
        Dynamically adjust thresholds based on market conditions
        
        Args:
            market_conditions: Dictionary with market condition indicators
        """
        # Adjust based on volatility
        volatility = market_conditions.get('volatility', 0)
        if volatility > 0.05:  # High volatility
            self.policy.confidence_threshold = 0.70
            self.policy.prob_long = 0.65
            self.policy.prob_short = 0.35
        elif volatility < 0.01:  # Low volatility
            self.policy.confidence_threshold = 0.60
            self.policy.prob_long = 0.55
            self.policy.prob_short = 0.45
        else:  # Normal volatility
            self.policy.confidence_threshold = 0.65
            self.policy.prob_long = 0.60
            self.policy.prob_short = 0.40
        
        # Adjust based on trend strength
        trend_strength = market_conditions.get('trend_strength', 0)
        if abs(trend_strength) > 0.05:  # Strong trend
            self.policy.min_rr = 1.5  # Lower RR requirement in strong trends
        else:
            self.policy.min_rr = 1.8  # Higher RR requirement in sideways markets
    
    def get_policy_summary(self) -> Dict:
        """
        Get current policy summary
        
        Returns:
            Dictionary with policy summary
        """
        return {
            'prob_long': self.policy.prob_long,
            'prob_short': self.policy.prob_short,
            'min_rr': self.policy.min_rr,
            'confidence_threshold': self.policy.confidence_threshold,
            'volatility_threshold': self.policy.volatility_threshold
        }
