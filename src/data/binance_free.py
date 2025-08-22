from __future__ import annotations
import logging
import time
from typing import Optional, Dict, Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class BinanceFreeProvider:
    """Free Binance spot public REST provider.

    - Real OHLCV (klines) only
    - No synthetic derivatives data
    - Spot-only endpoints

    Returns a unified frame:
    ['open','high','low','close','volume']
    """

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "alpha12_24/1.0 (binance-spot)"})

    # ---------------- HTTP ----------------
    def _make_request(self, endpoint: str, params: Dict[str, Any], max_retries: int = 4) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/{endpoint}"
                r = self.session.get(url, params=params, timeout=25)
                r.raise_for_status()
                try:
                    return r.json()
                except ValueError:
                    logger.warning(f"Non-JSON response from {endpoint}")
                    return None
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}) for {endpoint}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        logger.error(f"All retries failed for {endpoint}")
        return None

    # ---------------- Core fetchers ----------------
    def get_klines(self, symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 300) -> pd.DataFrame:
        """Fetch OHLCV data from Binance spot API"""
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        data = self._make_request("klines", params)
        
        if not data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        rows = []
        for k in data:
            row = {
                "timestamp": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    # ---------------- Assembly ----------------
    def assemble(self, symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 300) -> pd.DataFrame:
        """Assemble spot-only data - no synthetic derivatives"""
        logger.info(f"Assembling Binance spot data {symbol} @ {interval}")
        
        df = self.get_klines(symbol, interval, limit)
        
        if df.empty:
            logger.error("Binance klines empty; returning empty frame")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        # Return only the columns we actually have - no synthetic data
        return df[["open", "high", "low", "close", "volume"]]


# ---- Module-level helper for composite loader ----
def assemble(symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 300) -> pd.DataFrame:
    """Spot-only data assembly - no synthetic derivatives"""
    return BinanceFreeProvider().assemble(symbol=symbol, interval=interval, limit=limit)

def get_latest_price(symbol: str = "BTCUSDT") -> Optional[float]:
    """Get the latest price for a symbol from Binance spot API"""
    try:
        provider = BinanceFreeProvider()
        df = provider.get_klines(symbol, interval="1m", limit=1)
        if not df.empty:
            return float(df["close"].iloc[-1])
        return None
    except Exception as e:
        logger.error(f"Failed to get latest price for {symbol}: {e}")
        return None