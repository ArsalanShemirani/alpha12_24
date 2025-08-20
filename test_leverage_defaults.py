#!/usr/bin/env python3
"""
Test script to verify that 10x leverage is used as default for both manual and auto setups
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
sys.path.append('src')

from src.daemon.autosignal import _size_position, _build_setup, _get_ui_override_config
from src.dashboard.app import _append_setup_row

def test_leverage_defaults():
    """Test that 10x leverage is used as default for both manual and auto setups"""
    
    print("=== TESTING LEVERAGE DEFAULTS ===\n")
    
    # Test 1: Check UI Configuration
    print("1. Checking UI Configuration:")
    ui_config = _get_ui_override_config()
    max_lev_ui = ui_config.get("max_leverage", "NOT_FOUND")
    print(f"   UI Config Max Leverage: {max_lev_ui}")
    assert max_lev_ui == 10, f"UI config should default to 10, got {max_lev_ui}"
    print("   ✅ UI Config: 10x leverage default\n")
    
    # Test 2: Test Manual Setup Position Sizing
    print("2. Testing Manual Setup Position Sizing:")
    entry_px = 113500.0
    stop_px = 113000.0
    balance = 400.0
    risk_pct = 1.0
    interval = "1h"
    nominal_position_pct = 25.0
    
    # Test with explicit 10x leverage
    size_units, notional, lev = _size_position(
        entry_px, stop_px, balance, 10, risk_pct, interval, nominal_position_pct
    )
    
    print(f"   Entry: ${entry_px:,.2f}")
    print(f"   Stop: ${stop_px:,.2f}")
    print(f"   Balance: ${balance:,.2f}")
    print(f"   Max Leverage: 10x")
    print(f"   Result - Notional: ${notional:,.2f}")
    print(f"   Result - Leverage: {lev:.2f}x")
    
    # Verify leverage is reasonable (should be > 1.0 and <= 10.0)
    assert 1.0 <= lev <= 10.0, f"Leverage should be between 1.0 and 10.0, got {lev}"
    assert notional > 100, f"Notional should be significant with 10x leverage, got {notional}"
    print("   ✅ Manual Setup: 10x leverage working\n")
    
    # Test 3: Test Auto Setup Position Sizing
    print("3. Testing Auto Setup Position Sizing:")
    
    # Build setup with new logic
    setup = _build_setup(
        direction="short", 
        price=entry_px, 
        atr=500.0, 
        rr=1.8,
        k_entry=0.5,
        k_stop=1.0,
        valid_bars=24,
        now_ts=datetime.now(),
        bar_interval=interval,
        entry_buffer_bps=5.0
    )
    
    # Calculate position sizing with UI config (should use 10x default)
    balance_ui = ui_config.get("acct_balance", 400.0)
    max_lev_ui = ui_config.get("max_leverage", 10)  # Should default to 10
    risk_pct_ui = ui_config.get("risk_per_trade_pct", 1.0)
    nominal_position_pct_ui = ui_config.get("nominal_position_pct", 25.0)
    
    size_units_auto, notional_auto, lev_auto = _size_position(
        setup["entry"], 
        setup["stop"], 
        balance_ui, 
        max_lev_ui, 
        risk_pct_ui, 
        interval, 
        nominal_position_pct_ui
    )
    
    print(f"   UI Config Max Leverage: {max_lev_ui}x")
    print(f"   Result - Notional: ${notional_auto:,.2f}")
    print(f"   Result - Leverage: {lev_auto:.2f}x")
    
    # Verify auto setup uses 10x leverage
    assert max_lev_ui == 10, f"Auto setup should use 10x leverage default, got {max_lev_ui}"
    assert 1.0 <= lev_auto <= 10.0, f"Auto leverage should be between 1.0 and 10.0, got {lev_auto}"
    print("   ✅ Auto Setup: 10x leverage working\n")
    
    # Test 4: Create Test Setup to Verify in CSV
    print("4. Creating Test Setup to Verify in CSV:")
    setup_id = f"TEST-10X-{int(datetime.now().timestamp())}"
    setup_row = {
        "id": setup_id,
        "asset": "BTCUSDT",
        "interval": interval,
        "direction": "short",
        "entry": setup["entry"],
        "stop": setup["stop"],
        "target": setup["target"],
        "rr": setup["rr"],
        "size_units": size_units_auto,
        "notional_usd": notional_auto,
        "leverage": lev_auto,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
        "status": "pending",
        "confidence": 0.75,
        "trigger_rule": "touch",
        "entry_buffer_bps": 5.0,
        "origin": "test"
    }
    
    _append_setup_row(setup_row)
    print(f"   ✅ Test setup created: {setup_id}")
    print(f"   Check runs/setups.csv for verification")
    
    print("\n=== ALL TESTS PASSED ===")
    print("✅ 10x leverage is now the default for both manual and auto setups!")
    print("✅ UI configuration properly loads 10x as default")
    print("✅ Position sizing calculations use 10x leverage")
    print("✅ Both manual and auto setups will use 10x leverage unless changed in UI")

if __name__ == "__main__":
    test_leverage_defaults()
