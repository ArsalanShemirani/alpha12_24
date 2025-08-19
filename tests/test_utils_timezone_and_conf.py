import pandas as pd
from src.dashboard import app as appmod

def test_my_tz_conversion_series():
    idx = pd.date_range("2025-08-15 00:00:00", periods=3, freq="H", tz="UTC")
    out = appmod._to_my_tz_index(idx)
    assert str(out.tz) == "Asia/Kuala_Lumpur"
    #  UTC midnight = 08:00 Malaysia
    assert out[0].hour == 8

def test_my_tz_conversion_scalar():
    ts = "2025-08-15T00:00:00Z"
    out = appmod._to_my_tz_ts(ts)
    assert str(out.tz) == "Asia/Kuala_Lumpur"
    assert out.hour == 8  # +8h

def test_confidence_badge_thresholds():
    assert appmod.confidence_badge(0.70) == "HIGH"
    assert appmod.confidence_badge(0.58) == "MEDIUM"
    assert appmod.confidence_badge(0.40) == "LOW"

def test_tg_escape_md2():
    s = "*_[]()~`>#+-=|{}.! Test"
    esc = appmod.tg_escape_md2(s)
    # All specials must be escaped with backslashes
    assert esc != s
    assert "\\" in esc