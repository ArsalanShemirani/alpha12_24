#!/usr/bin/env python3
"""
Leverage management for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class LeverageConfig:
    """Leverage configuration"""
    max_leverage: float = 10.0
    base_leverage: float = 1.0
    volatility_scaling: bool = True
    regime_scaling: bool = True
    max_position_size: float = 0.2  # Max 20% of portfolio per position


class LeverageManager:
    """Leverage manager for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
        self.leverage_config = LeverageConfig()
        self.current_leverage = 1.0
        self.leverage_history = []
    
    def calculate_volatility_leverage(self, volatility: float, 
                                    base_volatility: float = 0.02) -> float:
        """
        Calculate leverage based on volatility
        
        Args:
            volatility: Current volatility
            base_volatility: Base volatility for scaling
        
        Returns:
            Leverage multiplier
        """
        if not self.leverage_config.volatility_scaling:
            return 1.0
        
        # Inverse relationship: higher volatility = lower leverage
        volatility_ratio = base_volatility / max(volatility, 0.001)
        
        # Cap the volatility scaling
        volatility_ratio = min(volatility_ratio, 2.0)
        volatility_ratio = max(volatility_ratio, 0.5)
        
        return volatility_ratio
    
    def calculate_regime_leverage(self, regime_implications: Optional[Dict] = None) -> float:
        """
        Calculate leverage based on market regime
        
        Args:
            regime_implications: Market regime implications
        
        Returns:
            Leverage multiplier
        """
        if not self.leverage_config.regime_scaling or not regime_implications:
            return 1.0
        
        # Get position sizing from regime
        position_sizing = regime_implications.get('position_sizing', 1.0)
        
        # Convert position sizing to leverage multiplier
        leverage_multiplier = position_sizing
        
        # Cap the regime scaling
        leverage_multiplier = min(leverage_multiplier, 1.5)
        leverage_multiplier = max(leverage_multiplier, 0.5)
        
        return leverage_multiplier
    
    def calculate_confidence_leverage(self, confidence: float) -> float:
        """
        Calculate leverage based on signal confidence
        
        Args:
            confidence: Signal confidence (0-1)
        
        Returns:
            Leverage multiplier
        """
        # Higher confidence = higher leverage
        confidence_multiplier = 0.5 + (confidence - 0.5) * 2
        
        # Cap the confidence scaling
        confidence_multiplier = min(confidence_multiplier, 1.5)
        confidence_multiplier = max(confidence_multiplier, 0.5)
        
        return confidence_multiplier
    
    def calculate_risk_reward_leverage(self, risk_reward: float, 
                                     target_rr: float = 1.8) -> float:
        """
        Calculate leverage based on risk/reward ratio
        
        Args:
            risk_reward: Actual risk/reward ratio
            target_rr: Target risk/reward ratio
        
        Returns:
            Leverage multiplier
        """
        # Better risk/reward = higher leverage
        rr_ratio = risk_reward / target_rr
        
        # Cap the RR scaling
        rr_multiplier = min(rr_ratio, 1.5)
        rr_multiplier = max(rr_multiplier, 0.5)
        
        return rr_multiplier
    
    def calculate_optimal_leverage(self, volatility: float, confidence: float,
                                 risk_reward: float, regime_implications: Optional[Dict] = None,
                                 portfolio_value: float = 10000) -> float:
        """
        Calculate optimal leverage for a trade
        
        Args:
            volatility: Current volatility
            confidence: Signal confidence
            risk_reward: Risk/reward ratio
            regime_implications: Market regime implications
            portfolio_value: Portfolio value
        
        Returns:
            Optimal leverage
        """
        # Calculate individual leverage components
        vol_leverage = self.calculate_volatility_leverage(volatility)
        regime_leverage = self.calculate_regime_leverage(regime_implications)
        confidence_leverage = self.calculate_confidence_leverage(confidence)
        rr_leverage = self.calculate_risk_reward_leverage(risk_reward)
        
        # Combine leverage components (geometric mean for stability)
        combined_leverage = np.power(
            vol_leverage * regime_leverage * confidence_leverage * rr_leverage, 
            0.25
        )
        
        # Apply base leverage
        optimal_leverage = self.leverage_config.base_leverage * combined_leverage
        
        # Cap by maximum leverage
        optimal_leverage = min(optimal_leverage, self.leverage_config.max_leverage)
        
        # Ensure minimum leverage
        optimal_leverage = max(optimal_leverage, 0.1)
        
        return optimal_leverage
    
    def adjust_position_size_for_leverage(self, base_position_size: float,
                                        leverage: float, portfolio_value: float) -> float:
        """
        Adjust position size for leverage
        
        Args:
            base_position_size: Base position size
            leverage: Leverage multiplier
            portfolio_value: Portfolio value
        
        Returns:
            Adjusted position size
        """
        # Calculate leveraged position size
        leveraged_size = base_position_size * leverage
        
        # Check maximum position size constraint
        max_position_value = portfolio_value * self.leverage_config.max_position_size
        
        # Cap by maximum position size
        if leveraged_size > max_position_value:
            leveraged_size = max_position_value
        
        return leveraged_size
    
    def calculate_margin_requirement(self, position_size: float, leverage: float,
                                   price: float) -> float:
        """
        Calculate margin requirement
        
        Args:
            position_size: Position size
            leverage: Leverage multiplier
            price: Asset price
        
        Returns:
            Margin requirement
        """
        position_value = position_size * price
        margin_requirement = position_value / leverage
        
        return margin_requirement
    
    def check_margin_sufficiency(self, margin_requirement: float,
                               available_margin: float) -> Tuple[bool, float]:
        """
        Check if margin is sufficient
        
        Args:
            margin_requirement: Required margin
            available_margin: Available margin
        
        Returns:
            Tuple of (is_sufficient, margin_ratio)
        """
        margin_ratio = available_margin / margin_requirement if margin_requirement > 0 else float('inf')
        is_sufficient = margin_ratio >= 1.0
        
        return is_sufficient, margin_ratio
    
    def calculate_leverage_metrics(self, position_size: float, leverage: float,
                                 entry_price: float, current_price: float) -> Dict:
        """
        Calculate leverage-related metrics
        
        Args:
            position_size: Position size
            leverage: Leverage multiplier
            entry_price: Entry price
            current_price: Current price
        
        Returns:
            Dictionary with leverage metrics
        """
        position_value = position_size * entry_price
        margin_requirement = position_value / leverage
        
        # Calculate P&L
        price_change = (current_price - entry_price) / entry_price
        pnl = position_value * price_change
        
        # Calculate P&L as percentage of margin
        pnl_margin_ratio = pnl / margin_requirement if margin_requirement > 0 else 0
        
        # Calculate liquidation price (simplified)
        liquidation_price = entry_price * (1 - 1/leverage)
        
        return {
            'position_value': position_value,
            'margin_requirement': margin_requirement,
            'leverage': leverage,
            'pnl': pnl,
            'pnl_margin_ratio': pnl_margin_ratio,
            'liquidation_price': liquidation_price,
            'price_change': price_change
        }
    
    def update_leverage_history(self, leverage: float, timestamp: datetime = None) -> None:
        """
        Update leverage history
        
        Args:
            leverage: Leverage value
            timestamp: Timestamp (optional)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.leverage_history.append({
            'leverage': leverage,
            'timestamp': timestamp
        })
        
        # Keep only last 1000 entries
        if len(self.leverage_history) > 1000:
            self.leverage_history = self.leverage_history[-1000:]
    
    def get_leverage_summary(self) -> Dict:
        """
        Get leverage summary
        
        Returns:
            Dictionary with leverage summary
        """
        if not self.leverage_history:
            return {
                'current_leverage': self.current_leverage,
                'avg_leverage': 1.0,
                'max_leverage': 1.0,
                'min_leverage': 1.0
            }
        
        leverages = [entry['leverage'] for entry in self.leverage_history]
        
        return {
            'current_leverage': self.current_leverage,
            'avg_leverage': np.mean(leverages),
            'max_leverage': np.max(leverages),
            'min_leverage': np.min(leverages),
            'leverage_std': np.std(leverages),
            'recent_leverages': leverages[-10:]  # Last 10 leverage values
        }
    
    def get_leverage_config(self) -> Dict:
        """
        Get leverage configuration
        
        Returns:
            Dictionary with leverage configuration
        """
        return {
            'max_leverage': self.leverage_config.max_leverage,
            'base_leverage': self.leverage_config.base_leverage,
            'volatility_scaling': self.leverage_config.volatility_scaling,
            'regime_scaling': self.leverage_config.regime_scaling,
            'max_position_size': self.leverage_config.max_position_size
        }
