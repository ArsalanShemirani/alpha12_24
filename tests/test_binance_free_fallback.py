import json
import os
import responses
import pandas as pd
from src.data import binance_free as bf

def _mk_klines(n=3, start=1_000_000_000_000):
    out = []
    for i in range(n):
        ts = start + i*60_000
        open_ = 100 + i
        high_ = open_ + 1
        low_  = open_ - 1
        close_= open_ + 0.5
        vol   = 10.0 + i
        out.append([ts, str(open_), str(high_), str(low_), str(close_), str(vol),
                    ts+60_000, "0", 0, "0", "0", "0"])
    return out

@responses.activate
def test_assemble_spot_fallback_when_futures_unreachable(monkeypatch):
    # Force futures endpoints to "bad" responses:
    for base in ("https://fapi1.binance.com", "https://fapi2.binance.com", "https://fapi.binancefuture.com"):
        responses.add(
            responses.GET,
            f"{base}/fapi/v1/klines",
            body="<!DOCTYPE html>not json",
            status=200,
            content_type="text/html"
        )
    # Spot klines works:
    responses.add(
        responses.GET,
        "https://api.binance.com/api/v3/klines",
        json=_mk_klines(5),
        status=200
    )
    # Disable futures extras (funding/oi/liqs/markprice) to keep the test simple
    monkeypatch.setenv("BINANCE_FAPI_DISABLE", "1")

    df = bf.assemble("BTCUSDT", "5m", 5)
    assert isinstance(df, pd.DataFrame)
    # Ensure required columns exist (spot-only schema)
    for c in ("open","high","low","close","volume"):
        assert c in df.columns
    assert len(df) == 5