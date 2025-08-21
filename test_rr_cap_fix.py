#!/usr/bin/env python3
"""
Test script to verify R:R cap is being applied correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

def test_rr_cap_fix():
    """Test that R:R cap is being applied correctly"""
    print("=" * 80)
    print("R:R CAP FIX TEST")
    print("=" * 80)
    
    try:
        from src.daemon.autosignal import build_autosetup_levels
        
        # Test parameters
        direction = "short"
        last_price = 4420.0
        atr = 44.0  # 1% of price
        rr = 3.0  # High R:R that should be capped
        interval = "1d"  # Test 1d specifically
        
        print(f"Test parameters:")
        print(f"  Direction: {direction}")
        print(f"  Last Price: {last_price}")
        print(f"  ATR: {atr}")
        print(f"  Original R:R: {rr}")
        print(f"  Timeframe: {interval}")
        print(f"  Expected R:R cap: 2.8")
        
        # Test the function
        print(f"\nCalling build_autosetup_levels...")
        setup_levels = build_autosetup_levels(direction, last_price, atr, rr, interval)
        
        print(f"\nResults:")
        print(f"  Entry: {setup_levels['entry']:.2f}")
        print(f"  Stop: {setup_levels['stop']:.2f}")
        print(f"  Target: {setup_levels['target']:.2f}")
        print(f"  R:R: {setup_levels['rr']:.2f}")
        
        # Calculate actual R:R from prices
        entry = setup_levels['entry']
        stop = setup_levels['stop']
        target = setup_levels['target']
        
        if direction == "short":
            stop_distance = stop - entry
            target_distance = entry - target
        else:
            stop_distance = entry - stop
            target_distance = target - entry
        
        actual_rr = target_distance / stop_distance if stop_distance > 0 else 0
        
        print(f"\nCalculations:")
        print(f"  Stop Distance: {stop_distance:.2f}")
        print(f"  Target Distance: {target_distance:.2f}")
        print(f"  Actual R:R: {actual_rr:.2f}")
        
        # Verify the cap is applied
        expected_cap = 2.8
        if actual_rr <= expected_cap + 0.01:  # Allow for floating point precision
            print(f"✅ R:R cap applied correctly: {actual_rr:.2f} <= {expected_cap}")
        else:
            print(f"❌ R:R cap NOT applied: {actual_rr:.2f} > {expected_cap}")
        
        # Test with features=None to force fallback
        print(f"\n" + "=" * 80)
        print("TESTING WITH FEATURES=NONE (FORCE FALLBACK)")
        print("=" * 80)
        
        setup_levels_fallback = build_autosetup_levels(direction, last_price, atr, rr, interval, features=None)
        
        print(f"\nFallback Results:")
        print(f"  Entry: {setup_levels_fallback['entry']:.2f}")
        print(f"  Stop: {setup_levels_fallback['stop']:.2f}")
        print(f"  Target: {setup_levels_fallback['target']:.2f}")
        print(f"  R:R: {setup_levels_fallback['rr']:.2f}")
        
        # Calculate actual R:R from prices
        entry_fb = setup_levels_fallback['entry']
        stop_fb = setup_levels_fallback['stop']
        target_fb = setup_levels_fallback['target']
        
        if direction == "short":
            stop_distance_fb = stop_fb - entry_fb
            target_distance_fb = entry_fb - target_fb
        else:
            stop_distance_fb = entry_fb - stop_fb
            target_distance_fb = target_fb - entry_fb
        
        actual_rr_fb = target_distance_fb / stop_distance_fb if stop_distance_fb > 0 else 0
        
        print(f"\nFallback Calculations:")
        print(f"  Stop Distance: {stop_distance_fb:.2f}")
        print(f"  Target Distance: {target_distance_fb:.2f}")
        print(f"  Actual R:R: {actual_rr_fb:.2f}")
        
        if actual_rr_fb <= expected_cap + 0.01:  # Allow for floating point precision
            print(f"✅ Fallback R:R cap applied correctly: {actual_rr_fb:.2f} <= {expected_cap}")
        else:
            print(f"❌ Fallback R:R cap NOT applied: {actual_rr_fb:.2f} > {expected_cap}")
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the test"""
    print("🔍 R:R CAP FIX TEST")
    
    test_rr_cap_fix()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("This test will:")
    print("1. Test that R:R cap is applied correctly for 1d timeframe")
    print("2. Test with features=None to force fallback path")
    print("3. Verify that high R:R values are properly capped")

if __name__ == "__main__":
    main()
