#!/usr/bin/env python3
"""
Trading planner for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TradePlan:
    """Trade plan structure"""
    signal: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward: float
    confidence: float
    timestamp: datetime
    metadata: Dict


class TradingPlanner:
    """Trading planner for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
        self.portfolio_value = 10000  # Default portfolio value
        self.risk_per_trade = config.risk_per_trade
        self.stop_min_frac = config.stop_min_frac
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              portfolio_value: Optional[float] = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            portfolio_value: Portfolio value (optional)
        
        Returns:
            Position size in base currency
        """
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
        
        # Calculate risk per trade in currency
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Calculate stop loss distance
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        # Calculate position size
        position_size = risk_amount / stop_distance
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, signal: str, 
                          volatility: float, atr: Optional[float] = None) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            signal: Trading signal ('LONG' or 'SHORT')
            volatility: Current volatility
            atr: Average True Range (optional)
        
        Returns:
            Stop loss price
        """
        # Use ATR if available, otherwise use volatility
        if atr is not None:
            stop_distance = atr * 2  # 2x ATR
        else:
            stop_distance = entry_price * volatility * 2  # 2x volatility
        
        # Ensure minimum stop distance
        min_stop_distance = entry_price * self.stop_min_frac
        stop_distance = max(stop_distance, min_stop_distance)
        
        if signal == "LONG":
            stop_loss = entry_price - stop_distance
        elif signal == "SHORT":
            stop_loss = entry_price + stop_distance
        else:
            stop_loss = entry_price
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            signal: str, risk_reward: float) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            signal: Trading signal ('LONG' or 'SHORT')
            risk_reward: Target risk/reward ratio
        
        Returns:
            Take profit price
        """
        # Calculate risk distance
        risk_distance = abs(entry_price - stop_loss)
        
        # Calculate reward distance
        reward_distance = risk_distance * risk_reward
        
        if signal == "LONG":
            take_profit = entry_price + reward_distance
        elif signal == "SHORT":
            take_profit = entry_price - reward_distance
        else:
            take_profit = entry_price
        
        return take_profit
    
    def create_trade_plan(self, signal: str, entry_price: float, 
                         confidence: float, volatility: float,
                         atr: Optional[float] = None,
                         portfolio_value: Optional[float] = None,
                         regime_implications: Optional[Dict] = None) -> TradePlan:
        """
        Create complete trade plan
        
        Args:
            signal: Trading signal
            entry_price: Entry price
            confidence: Signal confidence
            volatility: Current volatility
            atr: Average True Range (optional)
            portfolio_value: Portfolio value (optional)
            regime_implications: Market regime implications (optional)
        
        Returns:
            TradePlan object
        """
        if signal == "HOLD":
            return None
        
        # Apply regime adjustments
        position_multiplier = 1.0
        stop_multiplier = 1.0
        take_profit_multiplier = 1.0
        
        if regime_implications:
            position_multiplier = regime_implications.get('position_sizing', 1.0)
            stop_multiplier = regime_implications.get('stop_loss_multiplier', 1.0)
            take_profit_multiplier = regime_implications.get('take_profit_multiplier', 1.0)
        
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(entry_price, signal, volatility, atr)
        stop_loss = entry_price + (stop_loss - entry_price) * stop_multiplier
        
        # Calculate take profit with target risk/reward
        target_rr = self.config.min_rr * take_profit_multiplier
        take_profit = self.calculate_take_profit(entry_price, stop_loss, signal, target_rr)
        
        # Calculate position size
        base_position_size = self.calculate_position_size(entry_price, stop_loss, portfolio_value)
        position_size = base_position_size * position_multiplier
        
        # Calculate actual risk/reward ratio
        risk_reward = self.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        
        # Create metadata
        metadata = {
            'volatility': volatility,
            'atr': atr,
            'regime_implications': regime_implications,
            'position_multiplier': position_multiplier,
            'stop_multiplier': stop_multiplier,
            'take_profit_multiplier': take_profit_multiplier
        }
        
        return TradePlan(
            signal=signal,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_reward=risk_reward,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    def calculate_risk_reward_ratio(self, entry_price: float, 
                                  stop_loss: float, take_profit: float) -> float:
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
    
    def validate_trade_plan(self, trade_plan: TradePlan) -> Tuple[bool, str]:
        """
        Validate trade plan
        
        Args:
            trade_plan: TradePlan object
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if trade_plan is None:
            return False, "No trade plan"
        
        # Check signal
        if trade_plan.signal not in ["LONG", "SHORT"]:
            return False, f"Invalid signal: {trade_plan.signal}"
        
        # Check prices
        if trade_plan.entry_price <= 0:
            return False, "Invalid entry price"
        
        if trade_plan.signal == "LONG":
            if trade_plan.stop_loss >= trade_plan.entry_price:
                return False, "Stop loss must be below entry for LONG"
            if trade_plan.take_profit <= trade_plan.entry_price:
                return False, "Take profit must be above entry for LONG"
        else:  # SHORT
            if trade_plan.stop_loss <= trade_plan.entry_price:
                return False, "Stop loss must be above entry for SHORT"
            if trade_plan.take_profit >= trade_plan.entry_price:
                return False, "Take profit must be below entry for SHORT"
        
        # Check risk/reward ratio
        if trade_plan.risk_reward < self.config.min_rr:
            return False, f"Risk/reward ratio {trade_plan.risk_reward:.2f} below minimum {self.config.min_rr}"
        
        # Check position size
        if trade_plan.position_size <= 0:
            return False, "Invalid position size"
        
        # Check confidence
        if trade_plan.confidence < 0.5:
            return False, f"Low confidence: {trade_plan.confidence:.2f}"
        
        return True, "Valid trade plan"
    
    def get_trade_summary(self, trade_plan: TradePlan) -> Dict:
        """
        Get trade plan summary
        
        Args:
            trade_plan: TradePlan object
        
        Returns:
            Dictionary with trade summary
        """
        if trade_plan is None:
            return {}
        
        return {
            'signal': trade_plan.signal,
            'entry_price': trade_plan.entry_price,
            'stop_loss': trade_plan.stop_loss,
            'take_profit': trade_plan.take_profit,
            'position_size': trade_plan.position_size,
            'risk_reward': trade_plan.risk_reward,
            'confidence': trade_plan.confidence,
            'timestamp': trade_plan.timestamp,
            'risk_amount': abs(trade_plan.entry_price - trade_plan.stop_loss) * trade_plan.position_size,
            'potential_profit': abs(trade_plan.take_profit - trade_plan.entry_price) * trade_plan.position_size
        }
    
    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update portfolio value
        
        Args:
            new_value: New portfolio value
        """
        self.portfolio_value = new_value
    
    def get_planner_summary(self) -> Dict:
        """
        Get planner summary
        
        Returns:
            Dictionary with planner summary
        """
        return {
            'portfolio_value': self.portfolio_value,
            'risk_per_trade': self.risk_per_trade,
            'stop_min_frac': self.stop_min_frac,
            'min_rr': self.config.min_rr
        }
