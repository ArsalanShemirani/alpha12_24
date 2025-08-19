#!/usr/bin/env python3
"""
Real ETF flows data provider (basic implementation)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RealETFFlowsProvider:
    """Real ETF flows data provider"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'alpha12_24/real_etf_flows'
        })
        
        # ETF symbols to track
        self.etf_symbols = {
            "GBTC": "Grayscale Bitcoin Trust",
            "BITO": "ProShares Bitcoin Strategy ETF",
            "BITI": "ProShares Short Bitcoin Strategy ETF"
        }
    
    def get_etf_price_data(self, symbol: str = "GBTC", days: int = 30) -> Optional[pd.DataFrame]:
        """Get ETF price data as proxy for flows"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "range": f"{days}d",
                "interval": "1d",
                "includePrePost": "false"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
                return None
            
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]
            
            df_data = []
            for i, ts in enumerate(timestamps):
                date = datetime.fromtimestamp(ts)
                df_data.append({
                    "date": date,
                    "close": quotes["close"][i] if quotes["close"][i] else None,
                    "volume": quotes["volume"][i] if quotes["volume"][i] else None
                })
            
            df = pd.DataFrame(df_data)
            df = df.dropna().reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching ETF data for {symbol}: {e}")
            return None
    
    def get_etf_flow_features(self, days: int = 30) -> Dict:
        """Get ETF flow features for model training"""
        try:
            features = {
                "etf_flow_sentiment": 0.0,
                "etf_flow_classification": "Neutral",
                "etf_flow_valid_count": 0,
                "etf_flow_total_count": len(self.etf_symbols),
            }
            
            # Try to get GBTC data as primary indicator
            gbtc_df = self.get_etf_price_data("GBTC", days)
            if gbtc_df is not None and not gbtc_df.empty:
                current = gbtc_df.iloc[-1]
                if len(gbtc_df) > 1:
                    price_change = (current["close"] - gbtc_df.iloc[-2]["close"]) / gbtc_df.iloc[-2]["close"]
                    features["etf_flow_sentiment"] = np.clip(price_change * 10, -1, 1)
                    features["etf_flow_classification"] = "Inflow" if price_change > 0 else "Outflow"
                    features["etf_flow_valid_count"] = 1
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting ETF flow features: {e}")
            return {
                "etf_flow_sentiment": 0.0,
                "etf_flow_classification": "Neutral",
                "etf_flow_valid_count": 0,
                "etf_flow_total_count": len(self.etf_symbols),
            }

# Global instance
_etf_flows_provider = None

def get_etf_flows_provider() -> RealETFFlowsProvider:
    """Get global ETF flows provider instance"""
    global _etf_flows_provider
    if _etf_flows_provider is None:
        _etf_flows_provider = RealETFFlowsProvider()
    return _etf_flows_provider

def get_etf_flow_features(days: int = 30) -> Dict:
    """Get ETF flow features for model training"""
    provider = get_etf_flows_provider()
    return provider.get_etf_flow_features(days)
