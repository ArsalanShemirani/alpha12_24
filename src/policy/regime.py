#!/usr/bin/env python3
"""
Regime policy for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeState:
    """Market regime state"""
    regime: MarketRegime
    confidence: float
    duration: int
    strength: float
    metadata: Dict


class RegimeDetector:
    """Market regime detector for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = []
        self.min_regime_duration = 24  # Minimum hours for regime change
    
    def detect_trend_regime(self, prices: pd.Series, window: int = 50) -> Dict:
        """
        Detect trend regime using moving averages
        
        Args:
            prices: Price series
            window: Window for trend calculation
        
        Returns:
            Dictionary with trend regime information
        """
        if len(prices) < window:
            return {'regime': MarketRegime.SIDEWAYS, 'confidence': 0.5, 'strength': 0.0}
        
        # Calculate moving averages
        sma_short = prices.rolling(window=window//2).mean()
        sma_long = prices.rolling(window=window).mean()
        
        # Calculate trend strength
        trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        
        # Calculate trend consistency
        recent_trend = prices.tail(window//2).pct_change().mean()
        
        # Determine regime
        if trend_strength > 0.02 and recent_trend > 0:
            regime = MarketRegime.TRENDING_UP
            confidence = min(abs(trend_strength) * 10, 0.95)
            strength = abs(trend_strength)
        elif trend_strength < -0.02 and recent_trend < 0:
            regime = MarketRegime.TRENDING_DOWN
            confidence = min(abs(trend_strength) * 10, 0.95)
            strength = abs(trend_strength)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5
            strength = 0.0
        
        return {
            'regime': regime,
            'confidence': confidence,
            'strength': strength,
            'trend_strength': trend_strength,
            'recent_trend': recent_trend
        }
    
    def detect_volatility_regime(self, returns: pd.Series, window: int = 24) -> Dict:
        """
        Detect volatility regime
        
        Args:
            returns: Return series
            window: Window for volatility calculation
        
        Returns:
            Dictionary with volatility regime information
        """
        if len(returns) < window:
            return {'regime': MarketRegime.SIDEWAYS, 'confidence': 0.5, 'strength': 0.0}
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=window).std()
        current_vol = volatility.iloc[-1]
        
        # Calculate volatility percentile
        vol_percentile = (volatility < current_vol).mean()
        
        # Determine regime
        if vol_percentile > 0.8:  # High volatility
            regime = MarketRegime.VOLATILE
            confidence = vol_percentile
            strength = current_vol
        elif vol_percentile < 0.2:  # Low volatility
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 1 - vol_percentile
            strength = 1 / current_vol if current_vol > 0 else 1
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5
            strength = 0.0
        
        return {
            'regime': regime,
            'confidence': confidence,
            'strength': strength,
            'volatility': current_vol,
            'vol_percentile': vol_percentile
        }
    
    def detect_momentum_regime(self, prices: pd.Series, window: int = 20) -> Dict:
        """
        Detect momentum regime using RSI and MACD
        
        Args:
            prices: Price series
            window: Window for momentum calculation
        
        Returns:
            Dictionary with momentum regime information
        """
        if len(prices) < window:
            return {'regime': MarketRegime.SIDEWAYS, 'confidence': 0.5, 'strength': 0.0}
        
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        current_rsi = rsi.iloc[-1]
        current_macd = macd.iloc[-1]
        current_macd_signal = macd_signal.iloc[-1]
        
        # Determine momentum regime
        if current_rsi > 70 and current_macd > current_macd_signal:
            regime = MarketRegime.TRENDING_UP
            confidence = min((current_rsi - 50) / 50, 0.95)
            strength = current_rsi / 100
        elif current_rsi < 30 and current_macd < current_macd_signal:
            regime = MarketRegime.TRENDING_DOWN
            confidence = min((50 - current_rsi) / 50, 0.95)
            strength = (100 - current_rsi) / 100
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5
            strength = 0.0
        
        return {
            'regime': regime,
            'confidence': confidence,
            'strength': strength,
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_macd_signal
        }
    
    def detect_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            RegimeState object
        """
        prices = df['close']
        returns = prices.pct_change().dropna()
        
        # Detect different regime types
        trend_regime = self.detect_trend_regime(prices)
        volatility_regime = self.detect_volatility_regime(returns)
        momentum_regime = self.detect_momentum_regime(prices)
        
        # Combine regime signals
        regime_scores = {
            MarketRegime.TRENDING_UP: 0,
            MarketRegime.TRENDING_DOWN: 0,
            MarketRegime.SIDEWAYS: 0,
            MarketRegime.VOLATILE: 0,
            MarketRegime.LOW_VOLATILITY: 0
        }
        
        # Weight the different regime signals
        if trend_regime['regime'] == MarketRegime.TRENDING_UP:
            regime_scores[MarketRegime.TRENDING_UP] += trend_regime['confidence'] * 0.4
        elif trend_regime['regime'] == MarketRegime.TRENDING_DOWN:
            regime_scores[MarketRegime.TRENDING_DOWN] += trend_regime['confidence'] * 0.4
        else:
            regime_scores[MarketRegime.SIDEWAYS] += trend_regime['confidence'] * 0.3
        
        if volatility_regime['regime'] == MarketRegime.VOLATILE:
            regime_scores[MarketRegime.VOLATILE] += volatility_regime['confidence'] * 0.3
        elif volatility_regime['regime'] == MarketRegime.LOW_VOLATILITY:
            regime_scores[MarketRegime.LOW_VOLATILITY] += volatility_regime['confidence'] * 0.3
        else:
            regime_scores[MarketRegime.SIDEWAYS] += volatility_regime['confidence'] * 0.2
        
        if momentum_regime['regime'] == MarketRegime.TRENDING_UP:
            regime_scores[MarketRegime.TRENDING_UP] += momentum_regime['confidence'] * 0.3
        elif momentum_regime['regime'] == MarketRegime.TRENDING_DOWN:
            regime_scores[MarketRegime.TRENDING_DOWN] += momentum_regime['confidence'] * 0.3
        else:
            regime_scores[MarketRegime.SIDEWAYS] += momentum_regime['confidence'] * 0.2
        
        # Determine dominant regime
        dominant_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[dominant_regime]
        
        # Calculate regime strength
        if dominant_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            strength = max(trend_regime['strength'], momentum_regime['strength'])
        elif dominant_regime == MarketRegime.VOLATILE:
            strength = volatility_regime['strength']
        elif dominant_regime == MarketRegime.LOW_VOLATILITY:
            strength = volatility_regime['strength']
        else:
            strength = 0.0
        
        # Check if regime has changed
        duration = 1
        if self.regime_history and self.regime_history[-1].regime == dominant_regime:
            duration = self.regime_history[-1].duration + 1
        
        # Create regime state
        regime_state = RegimeState(
            regime=dominant_regime,
            confidence=confidence,
            duration=duration,
            strength=strength,
            metadata={
                'trend_regime': trend_regime,
                'volatility_regime': volatility_regime,
                'momentum_regime': momentum_regime,
                'regime_scores': regime_scores
            }
        )
        
        # Update regime history
        if not self.regime_history or self.regime_history[-1].regime != dominant_regime:
            self.regime_history.append(regime_state)
        
        self.current_regime = dominant_regime
        
        return regime_state
    
    def get_regime_implications(self, regime_state: RegimeState) -> Dict:
        """
        Get trading implications for current regime
        
        Args:
            regime_state: Current regime state
        
        Returns:
            Dictionary with trading implications
        """
        implications = {
            'position_sizing': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'max_positions': 3,
            'preferred_strategies': []
        }
        
        if regime_state.regime == MarketRegime.TRENDING_UP:
            implications.update({
                'position_sizing': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.2,
                'max_positions': 4,
                'preferred_strategies': ['trend_following', 'momentum']
            })
        elif regime_state.regime == MarketRegime.TRENDING_DOWN:
            implications.update({
                'position_sizing': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.2,
                'max_positions': 4,
                'preferred_strategies': ['trend_following', 'momentum']
            })
        elif regime_state.regime == MarketRegime.VOLATILE:
            implications.update({
                'position_sizing': 0.7,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 0.8,
                'max_positions': 2,
                'preferred_strategies': ['mean_reversion', 'volatility']
            })
        elif regime_state.regime == MarketRegime.LOW_VOLATILITY:
            implications.update({
                'position_sizing': 0.8,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.0,
                'max_positions': 2,
                'preferred_strategies': ['range_trading', 'mean_reversion']
            })
        else:  # SIDEWAYS
            implications.update({
                'position_sizing': 0.9,
                'stop_loss_multiplier': 1.1,
                'take_profit_multiplier': 1.0,
                'max_positions': 3,
                'preferred_strategies': ['range_trading', 'mean_reversion']
            })
        
        return implications
    
    def get_regime_summary(self) -> Dict:
        """
        Get current regime summary
        
        Returns:
            Dictionary with regime summary
        """
        if not self.regime_history:
            return {'current_regime': 'unknown', 'confidence': 0.0}
        
        current = self.regime_history[-1]
        
        return {
            'current_regime': current.regime.value,
            'confidence': current.confidence,
            'duration': current.duration,
            'strength': current.strength,
            'regime_history': [r.regime.value for r in self.regime_history[-5:]]
        }
