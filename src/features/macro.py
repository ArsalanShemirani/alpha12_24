#!/usr/bin/env python3
"""
Macro features for alpha12_24
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time


class MacroFeatures:
    """Macroeconomic features for alpha12_24"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'alpha12_24/1.0'
        })
    
    def get_fear_greed_index(self) -> int:
        """Get Fear & Greed Index"""
        try:
            # Use our new sentiment provider
            from src.data.real_sentiment import get_current_sentiment
            sentiment_data = get_current_sentiment()
            
            if sentiment_data:
                return int(sentiment_data['value'])
            else:
                # Fallback to direct API call
                url = "https://api.alternative.me/fng/"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                return int(data['data'][0]['value'])
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return 50  # Neutral
    
    def get_dxy_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get DXY (US Dollar Index) data
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with DXY data
        """
        try:
            # Using Alpha Vantage API (free tier)
            # Note: In production, you'd want to use a paid API or different source
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'UUP',  # ProShares UltraShort Euro ETF as proxy
                'apikey': 'demo',  # Replace with actual API key
                'outputsize': 'compact'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                # Return synthetic data if API fails
                return self._generate_synthetic_dxy(days)
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convert types
            for col in ['1. open', '2. high', '3. low', '4. close', '5. volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return df.tail(days)
            
        except Exception as e:
            print(f"Error fetching DXY data: {e}")
            return self._generate_synthetic_dxy(days)
    
    def get_vix_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get VIX (Volatility Index) data
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with VIX data
        """
        try:
            # Using Alpha Vantage API for VIX
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'VIX',
                'apikey': 'demo',  # Replace with actual API key
                'outputsize': 'compact'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                # Return synthetic data if API fails
                return self._generate_synthetic_vix(days)
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convert types
            for col in ['1. open', '2. high', '3. low', '4. close', '5. volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return df.tail(days)
            
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return self._generate_synthetic_vix(days)
    
    def get_gold_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get Gold price data
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with Gold data
        """
        try:
            # Using Alpha Vantage API for Gold
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'GLD',  # SPDR Gold Trust as proxy
                'apikey': 'demo',  # Replace with actual API key
                'outputsize': 'compact'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                # Return synthetic data if API fails
                return self._generate_synthetic_gold(days)
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convert types
            for col in ['1. open', '2. high', '3. low', '4. close', '5. volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return df.tail(days)
            
        except Exception as e:
            print(f"Error fetching Gold data: {e}")
            return self._generate_synthetic_gold(days)
    
    def get_treasury_yields(self) -> Dict:
        """
        Get Treasury yields (synthetic data for now)
        
        Returns:
            Dictionary with Treasury yields
        """
        try:
            # Synthetic Treasury yields data
            yields = {
                'yield_2y': np.random.normal(4.5, 0.5),
                'yield_5y': np.random.normal(4.2, 0.4),
                'yield_10y': np.random.normal(4.0, 0.3),
                'yield_30y': np.random.normal(4.3, 0.2),
                'yield_curve_10y_2y': np.random.normal(-0.5, 0.2),
                'yield_curve_30y_10y': np.random.normal(0.3, 0.1)
            }
            
            return yields
            
        except Exception as e:
            print(f"Error fetching Treasury yields: {e}")
            return {}
    
    def get_inflation_data(self) -> Dict:
        """
        Get inflation data (synthetic for now)
        
        Returns:
            Dictionary with inflation data
        """
        try:
            # Synthetic inflation data
            inflation = {
                'cpi_yoy': np.random.normal(3.2, 0.5),
                'core_cpi_yoy': np.random.normal(3.8, 0.3),
                'pce_yoy': np.random.normal(2.8, 0.4),
                'core_pce_yoy': np.random.normal(3.2, 0.3),
                'inflation_expectations_5y': np.random.normal(2.5, 0.2),
                'inflation_expectations_10y': np.random.normal(2.3, 0.2)
            }
            
            return inflation
            
        except Exception as e:
            print(f"Error fetching inflation data: {e}")
            return {}
    
    def get_fed_funds_rate(self) -> float:
        """
        Get Federal Funds Rate (synthetic for now)
        
        Returns:
            Federal Funds Rate
        """
        try:
            # Synthetic Fed Funds Rate
            return np.random.normal(5.25, 0.1)
            
        except Exception as e:
            print(f"Error fetching Fed Funds Rate: {e}")
            return 5.25
    
    def calculate_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate macro features and add to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with macro features added
        """
        out = df.copy()
        
        # Get macro data
        fear_greed = self.get_fear_greed_index()
        dxy_data = self.get_dxy_data(30)
        vix_data = self.get_vix_data(30)
        gold_data = self.get_gold_data(30)
        treasury_yields = self.get_treasury_yields()
        inflation_data = self.get_inflation_data()
        fed_rate = self.get_fed_funds_rate()
        
        # Fear & Greed features
        out['fear_greed'] = fear_greed
        out['fear_greed_normalized'] = (fear_greed - 50) / 50
        
        # DXY features (if available)
        if not dxy_data.empty:
            out['dxy_close'] = dxy_data['close'].iloc[-1]
            out['dxy_change'] = dxy_data['close'].pct_change().iloc[-1]
            out['dxy_volatility'] = dxy_data['close'].pct_change().rolling(20).std().iloc[-1]
        else:
            out['dxy_close'] = 100.0
            out['dxy_change'] = 0.0
            out['dxy_volatility'] = 0.01
        
        # VIX features (if available)
        if not vix_data.empty:
            out['vix_close'] = vix_data['close'].iloc[-1]
            out['vix_change'] = vix_data['close'].pct_change().iloc[-1]
            out['vix_volatility'] = vix_data['close'].pct_change().rolling(20).std().iloc[-1]
        else:
            out['vix_close'] = 20.0
            out['vix_change'] = 0.0
            out['vix_volatility'] = 0.05
        
        # Gold features (if available)
        if not gold_data.empty:
            out['gold_close'] = gold_data['close'].iloc[-1]
            out['gold_change'] = gold_data['close'].pct_change().iloc[-1]
            out['gold_volatility'] = gold_data['close'].pct_change().rolling(20).std().iloc[-1]
        else:
            out['gold_close'] = 2000.0
            out['gold_change'] = 0.0
            out['gold_volatility'] = 0.02
        
        # Treasury yield features
        for key, value in treasury_yields.items():
            out[f'treasury_{key}'] = value
        
        # Inflation features
        for key, value in inflation_data.items():
            out[f'inflation_{key}'] = value
        
        # Fed rate features
        out['fed_funds_rate'] = fed_rate
        out['real_rate'] = fed_rate - inflation_data.get('core_pce_yoy', 3.0)
        
        # Macro regime features
        out['risk_on_regime'] = np.where(
            (out['vix_close'] < 20) & (out['fear_greed'] > 60), 1, 0
        )
        out['risk_off_regime'] = np.where(
            (out['vix_close'] > 30) & (out['fear_greed'] < 40), 1, 0
        )
        
        # Dollar strength features
        out['dollar_strength'] = np.where(out['dxy_change'] > 0.01, 1, 
                                        np.where(out['dxy_change'] < -0.01, -1, 0))
        
        # Safe haven demand
        out['safe_haven_demand'] = np.where(
            (out['gold_change'] > 0.01) & (out['vix_close'] > 25), 1, 0
        )
        
        return out
    
    def _generate_synthetic_dxy(self, days: int) -> pd.DataFrame:
        """Generate synthetic DXY data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic DXY data
        base_price = 100.0
        returns = np.random.normal(0, 0.01, days)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, days)),
            'high': prices * (1 + abs(np.random.normal(0, 0.002, days))),
            'low': prices * (1 - abs(np.random.normal(0, 0.002, days))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, days) * 1000
        }, index=dates)
        
        return df
    
    def _generate_synthetic_vix(self, days: int) -> pd.DataFrame:
        """Generate synthetic VIX data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic VIX data
        base_price = 20.0
        returns = np.random.normal(0, 0.05, days)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, days)),
            'high': prices * (1 + abs(np.random.normal(0, 0.02, days))),
            'low': prices * (1 - abs(np.random.normal(0, 0.02, days))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, days) * 1000
        }, index=dates)
        
        return df
    
    def _generate_synthetic_gold(self, days: int) -> pd.DataFrame:
        """Generate synthetic Gold data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic Gold data
        base_price = 2000.0
        returns = np.random.normal(0, 0.015, days)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, days)),
            'high': prices * (1 + abs(np.random.normal(0, 0.005, days))),
            'low': prices * (1 - abs(np.random.normal(0, 0.005, days))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, days) * 1000
        }, index=dates)
        
        return df
