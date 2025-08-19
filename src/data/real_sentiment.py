#!/usr/bin/env python3
"""
Real sentiment data provider using CFGI (Crypto Fear & Greed Index) API
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RealSentimentProvider:
    """Real sentiment data provider using CFGI API"""
    
    def __init__(self):
        self.base_url = "https://api.alternative.me/fng/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'alpha12_24/real_sentiment (contact: ops@alpha12_24.local)'
        })
    
    def _make_request(self, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All retries failed for CFGI API")
                    return None
    
    def get_current_sentiment(self) -> Optional[Dict]:
        """
        Get current sentiment data
        
        Returns:
            Dict with current sentiment data or None if failed
        """
        try:
            params = {
                "limit": 1,
                "format": "json"
            }
            
            data = self._make_request(params)
            if not data or "data" not in data or not data["data"]:
                return None
            
            latest = data["data"][0]
            timestamp = int(latest["timestamp"])
            date = datetime.fromtimestamp(timestamp)
            value = int(latest["value"])
            
            # Calculate normalized sentiment score (-1 to 1)
            sentiment_score = (value - 50) / 50
            
            return {
                "value": value,
                "classification": latest["value_classification"],
                "timestamp": timestamp,
                "date": date,
                "sentiment_score": sentiment_score,
                "time_until_update": latest.get("time_until_update", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting current sentiment: {e}")
            return None
    
    def get_historical_sentiment(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical sentiment data
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame with historical sentiment data or None if failed
        """
        try:
            params = {
                "limit": days,
                "format": "json"
            }
            
            data = self._make_request(params)
            if not data or "data" not in data:
                return None
            
            # Convert to DataFrame
            df_data = []
            for point in data["data"]:
                timestamp = int(point["timestamp"])
                date = datetime.fromtimestamp(timestamp)
                value = int(point["value"])
                sentiment_score = (value - 50) / 50
                
                df_data.append({
                    "date": date,
                    "timestamp": timestamp,
                    "value": value,
                    "classification": point["value_classification"],
                    "sentiment_score": sentiment_score
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values("date").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical sentiment: {e}")
            return None
    
    def get_sentiment_features(self, days: int = 30) -> Dict:
        """
        Get sentiment features for model training
        
        Args:
            days: Number of days to fetch
            
        Returns:
            Dict with sentiment features
        """
        try:
            df = self.get_historical_sentiment(days)
            if df is None or df.empty:
                return self._get_fallback_sentiment()
            
            # Calculate sentiment features
            current = df.iloc[-1]
            
            features = {
                "fear_greed": current["value"],
                "fear_greed_normalized": current["sentiment_score"],
                "fear_greed_classification": current["classification"],
                
                # Historical features
                "fear_greed_7d_avg": df["value"].tail(7).mean(),
                "fear_greed_7d_std": df["value"].tail(7).std(),
                "fear_greed_30d_avg": df["value"].mean(),
                "fear_greed_30d_std": df["value"].std(),
                
                # Change features
                "fear_greed_1d_change": df["value"].iloc[-1] - df["value"].iloc[-2] if len(df) > 1 else 0,
                "fear_greed_7d_change": df["value"].iloc[-1] - df["value"].iloc[-8] if len(df) > 7 else 0,
                
                # Sentiment score features
                "sentiment_score": current["sentiment_score"],
                "sentiment_score_7d_avg": df["sentiment_score"].tail(7).mean(),
                "sentiment_score_30d_avg": df["sentiment_score"].mean(),
                
                # Regime features
                "risk_on_regime": 1 if current["value"] > 60 else 0,
                "risk_off_regime": 1 if current["value"] < 40 else 0,
                "extreme_greed": 1 if current["value"] > 75 else 0,
                "extreme_fear": 1 if current["value"] < 25 else 0,
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting sentiment features: {e}")
            return self._get_fallback_sentiment()
    
    def _get_fallback_sentiment(self) -> Dict:
        """Get fallback sentiment data when API fails"""
        logger.warning("Using fallback sentiment data")
        
        return {
            "fear_greed": 50,  # Neutral
            "fear_greed_normalized": 0.0,
            "fear_greed_classification": "Neutral",
            "fear_greed_7d_avg": 50,
            "fear_greed_7d_std": 5,
            "fear_greed_30d_avg": 50,
            "fear_greed_30d_std": 10,
            "fear_greed_1d_change": 0,
            "fear_greed_7d_change": 0,
            "sentiment_score": 0.0,
            "sentiment_score_7d_avg": 0.0,
            "sentiment_score_30d_avg": 0.0,
            "risk_on_regime": 0,
            "risk_off_regime": 0,
            "extreme_greed": 0,
            "extreme_fear": 0,
        }
    
    def get_sentiment_weight(self, sentiment_score: float) -> float:
        """
        Calculate sentiment weight for setup generation
        
        Args:
            sentiment_score: Normalized sentiment score (-1 to 1)
            
        Returns:
            Weight multiplier (0.5 to 1.5)
        """
        # Sentiment weight calculation:
        # - Extreme fear (sentiment < -0.5): Higher weight for long setups (1.2-1.5)
        # - Fear (sentiment -0.5 to -0.1): Slight weight for long setups (1.0-1.2)
        # - Neutral (sentiment -0.1 to 0.1): Normal weight (0.9-1.1)
        # - Greed (sentiment 0.1 to 0.5): Slight weight for short setups (0.8-1.0)
        # - Extreme greed (sentiment > 0.5): Higher weight for short setups (0.5-0.8)
        
        if sentiment_score < -0.5:  # Extreme fear
            return 1.2 + (abs(sentiment_score) - 0.5) * 0.6  # 1.2 to 1.5
        elif sentiment_score < -0.1:  # Fear
            return 1.0 + abs(sentiment_score) * 0.4  # 1.0 to 1.2
        elif sentiment_score < 0.1:  # Neutral
            return 0.9 + (sentiment_score + 0.1) * 1.0  # 0.9 to 1.1
        elif sentiment_score < 0.5:  # Greed
            return 1.0 - sentiment_score * 0.4  # 0.8 to 1.0
        else:  # Extreme greed
            return 0.8 - (sentiment_score - 0.5) * 0.6  # 0.5 to 0.8

    def get_direction_specific_sentiment_weight(self, sentiment_score: float, direction: str) -> float:
        """
        Calculate direction-specific sentiment weight for setup generation
        
        Args:
            sentiment_score: Normalized sentiment score (-1 to 1)
            direction: 'long' or 'short'
            
        Returns:
            Weight multiplier (0.5 to 1.5)
        """
        direction = direction.lower()
        
        # Convert sentiment_score back to CFGI value for easier mapping
        cfgi_value = int(50 + (sentiment_score * 50))
        
        # Contrarian approach: 
        # - Fear favors LONG setups (buy when others are fearful)
        # - Greed favors SHORT setups (sell when others are greedy)
        
        if direction == 'long':
            # For LONG setups:
            # - Fear (low CFGI) → Higher confidence
            # - Greed (high CFGI) → Lower confidence
            if cfgi_value <= 10:  # Extreme fear
                return 1.35  # +35% confidence
            elif cfgi_value <= 20:  # Fear
                return 1.25  # +25% confidence
            elif cfgi_value <= 30:  # Fear
                return 1.15  # +15% confidence
            elif cfgi_value <= 40:  # Fear
                return 1.10  # +10% confidence
            elif cfgi_value <= 50:  # Neutral
                return 1.00  # No change
            elif cfgi_value <= 60:  # Greed
                return 0.95  # -5% confidence
            elif cfgi_value <= 70:  # Greed
                return 0.92  # -8% confidence
            elif cfgi_value <= 80:  # Greed
                return 0.84  # -16% confidence
            elif cfgi_value <= 90:  # Extreme greed
                return 0.75  # -25% confidence
            else:  # Extreme greed
                return 0.65  # -35% confidence
                
        elif direction == 'short':
            # For SHORT setups:
            # - Fear (low CFGI) → Lower confidence
            # - Greed (high CFGI) → Higher confidence
            if cfgi_value <= 10:  # Extreme fear
                return 0.65  # -35% confidence
            elif cfgi_value <= 20:  # Fear
                return 0.75  # -25% confidence
            elif cfgi_value <= 30:  # Fear
                return 0.85  # -15% confidence
            elif cfgi_value <= 40:  # Fear
                return 0.90  # -10% confidence
            elif cfgi_value <= 50:  # Neutral
                return 1.00  # No change
            elif cfgi_value <= 60:  # Greed
                return 1.05  # +5% confidence
            elif cfgi_value <= 70:  # Greed
                return 1.08  # +8% confidence
            elif cfgi_value <= 80:  # Greed
                return 1.16  # +16% confidence
            elif cfgi_value <= 90:  # Extreme greed
                return 1.25  # +25% confidence
            else:  # Extreme greed
                return 1.35  # +35% confidence
        else:
            # Fallback to direction-agnostic weight
            return self.get_sentiment_weight(sentiment_score)

# Global instance
_sentiment_provider = None

def get_sentiment_provider() -> RealSentimentProvider:
    """Get global sentiment provider instance"""
    global _sentiment_provider
    if _sentiment_provider is None:
        _sentiment_provider = RealSentimentProvider()
    return _sentiment_provider

def get_current_sentiment() -> Optional[Dict]:
    """Get current sentiment data"""
    provider = get_sentiment_provider()
    return provider.get_current_sentiment()

def get_sentiment_features(days: int = 30) -> Dict:
    """Get sentiment features for model training"""
    provider = get_sentiment_provider()
    return provider.get_sentiment_features(days)

def get_sentiment_weight(sentiment_score: float) -> float:
    """Get sentiment weight for setup generation"""
    provider = get_sentiment_provider()
    return provider.get_sentiment_weight(sentiment_score)

def get_direction_specific_sentiment_weight(sentiment_score: float, direction: str) -> float:
    """Get direction-specific sentiment weight for setup generation"""
    provider = get_sentiment_provider()
    return provider.get_direction_specific_sentiment_weight(sentiment_score, direction)
