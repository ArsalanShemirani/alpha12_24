from __future__ import annotations
import pandas as pd

MY_TZ = "Asia/Kuala_Lumpur"

def now():
    return pd.Timestamp.utcnow().tz_convert(MY_TZ)

def to_my_tz(ts):
    ts = pd.to_datetime(ts)
    if getattr(ts, "tz", None) is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MY_TZ)

def bars_to_minutes(interval: str) -> int:
    return {"5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}.get(interval, 5)
