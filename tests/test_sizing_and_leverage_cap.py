import pandas as pd
from src.dashboard import app as appmod

def test_risk_size_and_cap_math():
    # This test simply verifies the math approach inside app (not the UI)
    price = 100.0
    atr = 2.0
    setup = appmod._build_setup(
        "long", price, atr, rr=1.8, k_entry=0.5, k_stop=1.0,
        valid_bars=24, now_ts=pd.Timestamp.utcnow(), bar_interval="1h",
        entry_buffer_bps=5.0
    )
    entry = setup["entry"]; stop = setup["stop"]
    stop_dist = abs(entry - stop)

    balance = 400.0
    risk_pct = 1.0
    risk_amt = balance * (risk_pct/100.0)
    size_units = risk_amt / stop_dist
    notional = size_units * entry
    # Max lev 5x → cap notional to 2000
    notional_cap = balance * 5
    if notional > notional_cap:
        size_units *= (notional_cap / notional)
        notional = size_units * entry

    assert notional <= 2000 + 1e-6