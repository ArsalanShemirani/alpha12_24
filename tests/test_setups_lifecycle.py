import pandas as pd
import sys
import os

# Add the src directory to the path to import the functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def _build_setup(direction: str, price: float, atr: float, rr: float,
                 k_entry: float, k_stop: float, valid_bars: int,
                 now_ts, bar_interval: str, entry_buffer_bps: float) -> dict | None:
    # Base entry from ATR offset
    if direction == "long":
        base_entry = price - k_entry * atr
    elif direction == "short":
        base_entry = price + k_entry * atr
    else:
        return None
    # Anti stop-hunt entry adjustment: n bps in the direction of intended fill
    # For long, make entry slightly deeper (lower); for short, slightly higher.
    entry = base_entry * (1.0 - entry_buffer_bps/10000.0) if direction == "long" else base_entry * (1.0 + entry_buffer_bps/10000.0)
    stop  = entry - k_stop * atr if direction == "long" else entry + k_stop * atr
    target = entry + rr * (entry - stop) if direction == "long" else entry - rr * (stop - entry)
    per_bar_min = {"5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}.get(bar_interval, 5)
    expires_at = pd.to_datetime(now_ts) + pd.Timedelta(minutes=valid_bars * per_bar_min)
    return {
        "entry": float(entry), "stop": float(stop), "target": float(target),
        "rr": float(rr), "expires_at": expires_at
    }

def _check_trigger(setup_row: dict, latest_bar: dict, trigger_rule: str = "touch", buffer_bps: float = 5.0) -> bool:
    """
    Return True if setup should trigger on the latest bar.

    For pullback-style limit entries we use:
      - LONG (entry below current): trigger on **low <= entry**
      - SHORT (entry above current): trigger on **high >= entry**

    If trigger_rule == 'close-through', require an additional confirmation close:
      - LONG: close >= entry * (1 + buffer)
      - SHORT: close <= entry * (1 - buffer)
    """
    entry = float(setup_row["entry"])
    close = float(latest_bar["close"])

    if setup_row["direction"] == "long":
        touched = float(latest_bar["low"]) <= entry
        if trigger_rule == "touch":
            return touched
        # confirm: close back above entry by buffer to avoid knife-catch
        return touched and close >= entry * (1.0 + buffer_bps/10000.0)

    if setup_row["direction"] == "short":
        touched = float(latest_bar["high"]) >= entry
        if trigger_rule == "touch":
            return touched
        # confirm: close back below entry by buffer to avoid wick fills
        return touched and close <= entry * (1.0 - buffer_bps/10000.0)

    return False

def test_build_setup_and_expiry():
    now = pd.Timestamp("2025-08-15 12:00:00", tz="Asia/Kuala_Lumpur")
    s_long = _build_setup("long", price=100.0, atr=2.0, rr=1.8, k_entry=0.5, k_stop=1.0,
                          valid_bars=24, now_ts=now, bar_interval="1h", entry_buffer_bps=5.0)
    assert s_long["entry"] < 100.0 and s_long["stop"] < s_long["entry"] < s_long["target"]
    assert s_long["rr"] == 1.8
    assert s_long["expires_at"] > now

def test_trigger_touch_and_close_through():
    setup_long = {"direction":"long","entry":100.0}
    bar_touch = {"low":99.9, "high":101.0, "close":99.95}
    bar_confirm = {"low":99.9, "high":101.0, "close":100.2}
    assert _check_trigger(setup_long, bar_touch, "touch", 5.0) is True
    assert _check_trigger(setup_long, bar_touch, "close-through", 5.0) is False
    assert _check_trigger(setup_long, bar_confirm, "close-through", 5.0) is True

    setup_short = {"direction":"short","entry":100.0}
    bar_touch_s = {"low":98.0, "high":100.1, "close":100.05}
    bar_confirm_s = {"low":98.0, "high":100.2, "close":99.7}
    assert _check_trigger(setup_short, bar_touch_s, "touch", 5.0) is True
    assert _check_trigger(setup_short, bar_touch_s, "close-through", 5.0) is False
    assert _check_trigger(setup_short, bar_confirm_s, "close-through", 5.0) is True
