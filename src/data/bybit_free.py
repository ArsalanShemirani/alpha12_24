"""
Bybit free data provider (public v5). NO synthetic derivatives.

Exposes:
- assemble(symbol: str="BTCUSDT", interval: str="5m", limit: int=300) -> pd.DataFrame
  Returns a DataFrame with UTC DatetimeIndex and columns:
  ['open','high','low','close','volume']
"""

from __future__ import annotations

import time
from typing import Dict, Any

import pandas as pd
import requests

__all__ = ["assemble", "BybitFreeProvider"]

BYBIT_HOSTS = [
    "https://api.bybit.com",
    "https://api.bytick.com",  # fallback CDN
]

# Map our intervals to Bybit v5 kline intervals (minutes as strings)
_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

class BybitFreeProvider:
    def __init__(self, timeout: float = 12.0):
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": "alpha12_24/bybit_free (contact: ops@alpha12_24.local)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        self.timeout = timeout

    def _get_json(self, path: str, params: Dict[str, Any], tries: int = 6, backoff: float = 0.5) -> Dict[str, Any]:
        last_err = None
        for attempt in range(1, tries + 1):
            for base in BYBIT_HOSTS:
                url = f"{base}{path}"
                try:
                    r = self.sess.get(url, params=params, timeout=self.timeout)
                    # Retry on transient server pressure
                    if r.status_code in (429, 502, 503, 504):
                        last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
                        time.sleep(backoff * attempt)
                        continue
                    r.raise_for_status()
                    js = r.json()
                    # Bybit v5 wraps status in retCode/retMsg
                    if js.get("retCode") != 0:
                        last_err = RuntimeError(f"Bybit retCode={js.get('retCode')} retMsg={js.get('retMsg')}")
                        time.sleep(backoff * attempt)
                        continue
                    return js
                except Exception as e:
                    last_err = e
                    time.sleep(backoff * attempt)
                    continue
        raise RuntimeError(f"Bybit GET failed for {path} params={params}: {last_err}")

    @staticmethod
    def _map_interval(interval: str) -> str:
        iv = _INTERVAL_MAP.get(interval)
        if iv is None:
            raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(_INTERVAL_MAP.keys())}")
        return iv

    def get_ohlcv(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        iv = self._map_interval(interval)
        params = dict(category="linear", symbol=symbol, interval=iv, limit=min(int(limit), 1000))
        js = self._get_json("/v5/market/kline", params)
        data = js.get("result", {}).get("list", []) or []
        if not data:
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)

        # Per docs, 'list' rows: [startTime, open, high, low, close, volume, turnover]
        rows = []
        for row in data:
            ts_ms = int(row[0])
            o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
            v = float(row[5])
            rows.append((pd.to_datetime(ts_ms, unit="ms", utc=True), o, h, l, c, v))
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
        df = df.sort_index()
        return df

    def assemble(self, symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 300) -> pd.DataFrame:
        """
        Only returns real OHLCV from Bybit public API. NO funding/OI/liquidations/basis.
        """
        df = self.get_ohlcv(symbol, interval, limit)
        # Ensure exact columns and UTC index
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        return df[["open","high","low","close","volume"]]

def assemble(symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 300) -> pd.DataFrame:
    """
    Convenience function returning Bybit OHLCV as a clean DataFrame.
    """
    return BybitFreeProvider().assemble(symbol=symbol, interval=interval, limit=limit)
