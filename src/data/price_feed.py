from __future__ import annotations
import os, time, requests, pandas as pd
from typing import Optional, Dict

BINANCE_FAPI_DISABLE = os.getenv("BINANCE_FAPI_DISABLE", "1") == "1"

# Simple session with retries
_session = requests.Session()
_session.headers.update({"User-Agent":"alpha12_24-tracker/1.0"})

def _get(url, params=None, timeout=8):
    for _ in range(4):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(0.8)
    raise RuntimeError(f"GET failed: {url}")

def get_latest_candle(symbol: str, interval: str) -> Dict:
    """
    Returns dict: {timestamp, open, high, low, close, volume}
    Prefers futures; falls back to spot.
    """
    base_list = [
        "https://fapi.binance.com",
        "https://fapi1.binance.com",
        "https://fapi2.binance.com",
    ] if not BINANCE_FAPI_DISABLE else []

    # Futures first (if enabled)
    if not BINANCE_FAPI_DISABLE:
        for base in base_list:
            try:
                js = _get(f"{base}/fapi/v1/klines", params={"symbol":symbol, "interval":interval, "limit":1})
                k = js[-1]
                return {
                    "timestamp": pd.to_datetime(k[0], unit="ms", utc=True).tz_convert("Asia/Kuala_Lumpur"),
                    "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                    "close": float(k[4]), "volume": float(k[5]),
                }
            except Exception:
                continue

    # Spot fallback
    js = _get("https://api.binance.com/api/v3/klines", params={"symbol":symbol, "interval":interval, "limit":1})
    k = js[-1]
    return {
        "timestamp": pd.to_datetime(k[0], unit="ms", utc=True).tz_convert("Asia/Kuala_Lumpur"),
        "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
        "close": float(k[4]), "volume": float(k[5]),
    }

def get_window(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Pull a small window for labeling between start..end (inclusive).
    Uses spot klines; enough for exit checks (H/L first-touch).
    """
    # compute limit naive: ceil((end-start)/bar) + buffer
    per = {"5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}.get(interval, 5)
    minutes = max(1, int((end.tz_convert("UTC") - start.tz_convert("UTC")).total_seconds() / 60))
    limit = min(1500, minutes//per + 10)

    js = _get("https://api.binance.com/api/v3/klines", params={"symbol":symbol, "interval":interval, "limit":limit})
    rows = []
    for k in js:
        ts = pd.to_datetime(k[0], unit="ms", utc=True).tz_convert("Asia/Kuala_Lumpur")
        if ts < start or ts > end:
            continue
        rows.append({
            "timestamp": ts, "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])
        })
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df
