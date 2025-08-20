#!/usr/bin/env python3
"""
Test script to create a setup with the new position sizing logic
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
sys.path.append('src')

from src.daemon.autosignal import _size_position, _build_setup, _get_ui_override_config
from src.dashboard.app import _append_setup_row

def test_new_setup_creation():
    """Test creating a setup with the new position sizing logic"""
    
    # Get UI configuration
    ui_config = _get_ui_override_config()
    print("UI Config:", ui_config)
    
    # Test parameters
    direction = "short"
    price = 113500.0
    atr = 500.0  # Simulated ATR
    rr = 1.8
    interval = "1h"
    
    # Build setup with new logic
    setup = _build_setup(
        direction=direction, 
        price=price, 
        atr=atr, 
        rr=rr,
        k_entry=0.5,
        k_stop=1.0,
        valid_bars=24,
        now_ts=datetime.now(),
        bar_interval=interval,
        entry_buffer_bps=5.0
    )
    
    print(f"\nSetup Levels:")
    print(f"Entry: ${setup['entry']:,.2f}")
    print(f"Stop: ${setup['stop']:,.2f}")
    print(f"Target: ${setup['target']:,.2f}")
    print(f"RR: {setup['rr']:.2f}")
    
    # Calculate position sizing with new logic
    balance = ui_config.get("acct_balance", 400.0)
    max_lev = ui_config.get("max_leverage", 10)
    risk_pct = ui_config.get("risk_per_trade_pct", 1.0)
    nominal_position_pct = ui_config.get("nominal_position_pct", 25.0)
    
    size_units, notional, lev = _size_position(
        setup["entry"], 
        setup["stop"], 
        balance, 
        max_lev, 
        risk_pct, 
        interval, 
        nominal_position_pct
    )
    
    print(f"\nPosition Sizing Results:")
    print(f"Balance: ${balance:,.2f}")
    print(f"Max Leverage: {max_lev}x")
    print(f"Risk per Trade: {risk_pct}%")
    print(f"Nominal Position: {nominal_position_pct}%")
    print(f"Size Units: {size_units:.6f}")
    print(f"Notional: ${notional:,.2f}")
    print(f"Leverage: {lev:.2f}x")
    
    # Create a test setup row
    setup_id = f"TEST-NEW-{int(datetime.now().timestamp())}"
    setup_row = {
        "id": setup_id,
        "asset": "BTCUSDT",
        "interval": interval,
        "direction": direction,
        "entry": setup["entry"],
        "stop": setup["stop"],
        "target": setup["target"],
        "rr": setup["rr"],
        "size_units": size_units,
        "notional_usd": notional,
        "leverage": lev,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
        "status": "pending",
        "confidence": 0.75,
        "trigger_rule": "touch",
        "entry_buffer_bps": 5.0,
        "origin": "test"
    }
    
    # Append to CSV
    _append_setup_row(setup_row)
    print(f"\n✅ Test setup created with ID: {setup_id}")
    print(f"Check runs/setups.csv for the new setup!")

if __name__ == "__main__":
    test_new_setup_creation()
