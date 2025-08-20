#!/usr/bin/env python3
"""
Test script to verify the new position sizing logic
"""

import sys
import os
sys.path.append('src')

from src.daemon.autosignal import _size_position, _get_timeframe_stop_multiplier

def test_position_sizing():
    """Test the new position sizing logic"""
    
    # Test parameters
    entry_px = 113500.0
    stop_px = 113000.0  # 1h baseline
    balance = 400.0
    max_lev = 10
    risk_pct = 1.0
    interval = '1h'
    nominal_position_pct = 25.0

    # Calculate position sizing
    size_units, notional, lev = _size_position(entry_px, stop_px, balance, max_lev, risk_pct, interval, nominal_position_pct)

    print(f'=== 1h TIMEFRAME TEST ===')
    print(f'Entry: ${entry_px:,.2f}')
    print(f'Stop: ${stop_px:,.2f}')
    print(f'Stop Distance: ${abs(entry_px - stop_px):,.2f}')
    print(f'Balance: ${balance:,.2f}')
    print(f'Max Leverage: {max_lev}x')
    print(f'Risk per Trade: {risk_pct}%')
    print(f'Nominal Position: {nominal_position_pct}%')
    print(f'--- Results ---')
    print(f'Size Units: {size_units:.6f}')
    print(f'Notional: ${notional:,.2f}')
    print(f'Leverage: {lev:.2f}x')
    print(f'Risk Amount: ${balance * risk_pct / 100:.2f}')
    print(f'Nominal Balance: ${balance * max_lev:,.2f}')
    print(f'Target Position: ${balance * max_lev * nominal_position_pct / 100:,.2f}')
    
    # Test 4h timeframe
    print(f'\n=== 4h TIMEFRAME TEST ===')
    interval_4h = '4h'
    stop_multiplier = _get_timeframe_stop_multiplier(interval_4h)
    stop_px_4h = entry_px + (stop_px - entry_px) * stop_multiplier
    
    size_units_4h, notional_4h, lev_4h = _size_position(entry_px, stop_px_4h, balance, max_lev, risk_pct, interval_4h, nominal_position_pct)
    
    print(f'Entry: ${entry_px:,.2f}')
    print(f'Stop: ${stop_px_4h:,.2f}')
    print(f'Stop Distance: ${abs(entry_px - stop_px_4h):,.2f}')
    print(f'Stop Multiplier: {stop_multiplier}x')
    print(f'--- Results ---')
    print(f'Size Units: {size_units_4h:.6f}')
    print(f'Notional: ${notional_4h:,.2f}')
    print(f'Leverage: {lev_4h:.2f}x')

if __name__ == "__main__":
    test_position_sizing()
