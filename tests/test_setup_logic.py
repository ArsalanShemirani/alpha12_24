import numpy as np
import pandas as pd
from src.dashboard import app as appmod

def test_build_setup_and_rr_and_expiry():
    now = pd.Timestamp("2025-08-15 00:00:00", tz="UTC")
    # price 100, ATR 2, RR 2.0, entry offset 0.5 ATR, stop 1 ATR
    s_long = appmod._build_setup(
        direction="long", price=100.0, atr=2.0, rr=2.0,
        k_entry=0.5, k_stop=1.0, valid_bars=6,
        now_ts=now, bar_interval="1h", entry_buffer_bps=5.0
    )
    assert s_long["entry"] < 100.0  # pullback
    assert s_long["target"] > s_long["entry"]
    # 6 bars * 60m on 1h interval
    assert (pd.to_datetime(s_long["expires_at"]) - now).total_seconds() == 6*3600

    s_short = appmod._build_setup(
        direction="short", price=100.0, atr=2.0, rr=1.8,
        k_entry=0.5, k_stop=1.0, valid_bars=12,
        now_ts=now, bar_interval="5m", entry_buffer_bps=0.0
    )
    assert s_short["entry"] > 100.0  # pullback up for short
    assert s_short["target"] < s_short["entry"]

def test_trigger_rules_touch_and_close_through():
    row_long = {"direction":"long", "entry": 99.0}
    bar_ok_touch = {"low": 98.9, "high":100.5, "close": 98.95}
    bar_no_touch = {"low": 99.1, "high":100.5, "close": 99.05}

    assert appmod._check_trigger(row_long, bar_ok_touch, trigger_rule="touch", buffer_bps=5.0) is True
    assert appmod._check_trigger(row_long, bar_no_touch, trigger_rule="touch", buffer_bps=5.0) is False

    # For close-through, must first touch and then close back above entry by buffer
    assert appmod._check_trigger(row_long, {"low":98.9,"high":100.5,"close":99.01}, trigger_rule="close-through", buffer_bps=10.0) is False
    assert appmod._check_trigger(row_long, {"low":98.9,"high":100.5,"close":99.20}, trigger_rule="close-through", buffer_bps=10.0) is True

    row_short = {"direction":"short","entry":101.0}
    # touch on high, then close back below by buffer
    assert appmod._check_trigger(row_short, {"low":99.5,"high":101.1,"close":100.50}, trigger_rule="close-through", buffer_bps=10.0) is True