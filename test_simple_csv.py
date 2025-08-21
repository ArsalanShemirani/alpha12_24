#!/usr/bin/env python3
"""
Simple test to isolate CSV corruption issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import csv
import pandas as pd

def test_simple_csv():
    """Test simple CSV writing"""
    print("=" * 80)
    print("SIMPLE CSV TEST")
    print("=" * 80)
    
    # Test file
    test_file = "test_setup.csv"
    
    # Simple setup row
    setup_row = {
        "id": "TEST-ETHUSDT-4h-20250821_183000",
        "asset": "ETHUSDT",
        "interval": "4h",
        "direction": "short",
        "entry": 4406.90,
        "stop": 4473.00,
        "target": 4124.05,
        "rr": 2.0,
        "size_units": 0.151278,
        "notional_usd": 666.67,
        "leverage": 10.0,
        "created_at": "2025-08-21T18:30:00+08:00",
        "expires_at": "2025-08-25T18:30:00+08:00",
        "triggered_at": "",
        "status": "pending",
        "confidence": 0.81,
        "trigger_rule": "touch",
        "entry_buffer_bps": 5.0,
        "origin": "manual",
    }
    
    # SETUP_FIELDS from dashboard
    SETUP_FIELDS = [
        "id","asset","interval","direction","entry","stop","target","rr",
        "size_units","notional_usd","leverage",
        "created_at","expires_at","triggered_at","status","confidence","trigger_rule","entry_buffer_bps",
        "origin"
    ]
    
    print(f"Test setup:")
    print(f"  ID: {setup_row['id']}")
    print(f"  Entry: {setup_row['entry']}")
    print(f"  Status: {setup_row['status']}")
    
    # Test 1: Write with QUOTE_MINIMAL
    print(f"\nTest 1: Write with QUOTE_MINIMAL")
    try:
        with open(test_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SETUP_FIELDS, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            w.writerow(setup_row)
        
        # Try to read it
        df = pd.read_csv(test_file)
        print(f"✅ QUOTE_MINIMAL: CSV can be read, {len(df)} rows")
    except Exception as e:
        print(f"❌ QUOTE_MINIMAL: Error reading CSV: {e}")
    
    # Test 2: Write with QUOTE_ALL
    print(f"\nTest 2: Write with QUOTE_ALL")
    try:
        with open(test_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SETUP_FIELDS, extrasaction="ignore", quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerow(setup_row)
        
        # Try to read it
        df = pd.read_csv(test_file)
        print(f"✅ QUOTE_ALL: CSV can be read, {len(df)} rows")
    except Exception as e:
        print(f"❌ QUOTE_ALL: Error reading CSV: {e}")
    
    # Test 3: Check the actual file content
    print(f"\nTest 3: Check file content")
    try:
        with open(test_file, 'r') as f:
            content = f.read()
        
        print(f"File content:")
        print(content)
        
        # Check for newlines in the data
        lines = content.split('\n')
        print(f"Number of lines: {len(lines)}")
        
        for i, line in enumerate(lines):
            if line.strip():
                fields = line.split(',')
                print(f"Line {i+1}: {len(fields)} fields")
                if len(fields) != 18:
                    print(f"  WARNING: Line {i+1} has {len(fields)} fields instead of 18")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

def main():
    """Run the test"""
    print("🔍 SIMPLE CSV TEST")
    
    test_simple_csv()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("This test will:")
    print("1. Test simple CSV writing with different quoting options")
    print("2. Check if the issue is with the data or the writing method")
    print("3. Identify the root cause of CSV corruption")

if __name__ == "__main__":
    main()
