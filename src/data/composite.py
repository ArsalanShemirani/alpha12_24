"""
Bybit free data provider (public v5). NO synthetic derivatives.

Exposes:
- assemble(symbol: str="BTCUSDT", interval: str="5m", limit: int=300) -> pd.DataFrame
  Returns a DataFrame with UTC DatetimeIndex and columns:
  ['open','high','low','close','volume']
"""


import os
import time
import requests
import pandas as pd
from typing import Dict, Any, List

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
}

class BybitFreeProvider:
    def __init__(self, timeout: float = 10.0):
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
                    if js.get("retCode") != 0:
                        # Some errors still return 200; surface the message
                        last_err = RuntimeError(f"Bybit retCode={js.get('retCode')} retMsg={js.get('retMsg')}")
                        time.sleep(backoff * attempt)
                        continue
                    return js
                except Exception as e:
                    last_err = e
                    time.sleep(backoff * attempt)
                    continue
        raise RuntimeError(f"Bybit GET failed for {path} params={params}: {last_err}")

    def get_ohlcv(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        iv = _INTERVAL_MAP.get(interval)
        if iv is None:
            raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(_INTERVAL_MAP.keys())}")
        # We use linear (USDT perp) category for crypto pairs like BTCUSDT/ETHUSDT
        params = dict(category="linear", symbol=symbol, interval=iv, limit=min(int(limit), 1000))
        js = self._get_json("/v5/market/kline", params)
        data = js.get("result", {}).get("list", []) or []
        if not data:
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)

        # Per docs, 'list' is list of rows: [startTime, open, high, low, close, volume, turnover]
        # Convert and sort ascending by time
        rows = []
        for row in data:
            # row fields are strings
            ts_ms = int(row[0])
            o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
            v = float(row[5])
            rows.append((pd.to_datetime(ts_ms, unit="ms", utc=True), o, h, l, c, v))
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
        df = df.sort_index()
        return df

    def assemble(self, symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 300) -> pd.DataFrame:
        """
        Only returns real OHLCV from Bybit public API. NO funding/OI/liquidations/basis dummies.
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

"""
Composite data loader (NO synthetic derivatives).

This module selects between Binance Spot OHLCV (preferred) and Bybit OHLCV (fallback),
and exposes a single assemble() entrypoint compatible with the rest of the app.

Exports:
- assemble(symbol: str="BTCUSDT", interval: str="5m", limit: int=300, source: str="composite") -> pd.DataFrame
  Returns a UTC-indexed DataFrame with columns: ['open','high','low','close','volume']

- assemble_spot_plus_bybit(...) -> pd.DataFrame
  Same as assemble(source="composite"). Kept for backward compatibility.

Notes:
- No synthetic funding/OI/liquidations/basis are created here.
- If the preferred provider returns an empty frame, we automatically fall back to the other.
"""


from typing import Literal
import pandas as pd

from src.data.binance_free import assemble as binance_assemble
from src.data.bybit_free import assemble as bybit_assemble

__all__ = ["assemble", "assemble_spot_plus_bybit"]

def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df

def assemble_spot_plus_bybit(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 300,
) -> pd.DataFrame:
    """
    Preferred: Binance Spot OHLCV. Fallback: Bybit OHLCV.
    """
    # Try Binance first
    try:
        b = _ensure_utc(binance_assemble(symbol=symbol, interval=interval, limit=limit))
    except Exception:
        b = pd.DataFrame()

    if b is not None and not b.empty:
        # Keep only canonical OHLCV columns
        cols = [c for c in ["open","high","low","close","volume"] if c in b.columns]
        return b[cols]

    # Fallback to Bybit
    try:
        y = _ensure_utc(bybit_assemble(symbol=symbol, interval=interval, limit=limit))
    except Exception:
        y = pd.DataFrame()

    if y is not None and not y.empty:
        cols = [c for c in ["open","high","low","close","volume"] if c in y.columns]
        return y[cols]

    # Both empty → return empty canonical frame
    return _ensure_utc(pd.DataFrame())

def assemble(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 300,
    source: Literal["composite","binance","bybit"] = "composite",
) -> pd.DataFrame:
    """
    General entrypoint used by the app.

    - source="binance": force Binance Spot OHLCV
    - source="bybit": force Bybit OHLCV
    - source="composite": prefer Binance, fallback to Bybit
    """
    if source == "binance":
        try:
            df = _ensure_utc(binance_assemble(symbol=symbol, interval=interval, limit=limit))
            cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
            return df[cols]
        except Exception:
            return _ensure_utc(pd.DataFrame())

    if source == "bybit":
        try:
            df = _ensure_utc(bybit_assemble(symbol=symbol, interval=interval, limit=limit))
            cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
            return df[cols]
        except Exception:
            return _ensure_utc(pd.DataFrame())

    # default: composite
    return assemble_spot_plus_bybit(symbol=symbol, interval=interval, limit=limit)