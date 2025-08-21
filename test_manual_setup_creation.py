#!/usr/bin/env python3
"""
Test script to verify manual setup creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

def test_manual_setup_creation():
    """Test manual setup creation process"""
    print("=" * 80)
    print("MANUAL SETUP CREATION TEST")
    print("=" * 80)
    
    try:
        # Test the setup creation process step by step
        from src.dashboard.app import _append_setup_row, _mk_setup_id, SETUP_FIELDS
        
        # Create a test setup row
        setup_id = _mk_setup_id("ETHUSDT", "4h")
        created_at = pd.Timestamp.now(tz="UTC").tz_convert("Asia/Kuala_Lumpur")
        expires_at = created_at + pd.Timedelta(hours=96)
        
        setup_row = {
            "id": setup_id,
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
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "triggered_at": "",
            "status": "pending",
            "confidence": 0.81,
            "trigger_rule": "touch",
            "entry_buffer_bps": 5.0,
            "origin": "manual",
        }
        
        print(f"Test setup:")
        print(f"  ID: {setup_id}")
        print(f"  Entry: {setup_row['entry']}")
        print(f"  Created: {created_at}")
        print(f"  Status: {setup_row['status']}")
        print(f"  Origin: {setup_row['origin']}")
        
        # Verify all required fields are present
        missing_fields = [field for field in SETUP_FIELDS if field not in setup_row]
        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
            return
        
        print(f"✅ All required fields present")
        
        # Try to save
        print(f"\nSaving setup...")
        try:
            _append_setup_row(setup_row)
            print(f"✅ Setup saved successfully!")
        except Exception as e:
            print(f"❌ Error saving setup: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Check if it was actually saved
        print(f"\nVerifying save...")
        setups_path = "runs/setups.csv"
        if os.path.exists(setups_path):
            try:
                df = pd.read_csv(setups_path)
                matching_setups = df[df['id'] == setup_id]
                if len(matching_setups) > 0:
                    print(f"✅ Setup found in CSV!")
                    row = matching_setups.iloc[0]
                    print(f"  Entry: {row.get('entry', 'N/A')}")
                    print(f"  Status: {row.get('status', 'N/A')}")
                    print(f"  Origin: {row.get('origin', 'N/A')}")
                else:
                    print(f"❌ Setup not found in CSV!")
                    
                    # Check recent setups
                    recent = df.sort_values('created_at', ascending=False).head(3)
                    print(f"\nRecent setups:")
                    for idx, row in recent.iterrows():
                        print(f"  {row.get('created_at', 'N/A')} - {row.get('id', 'N/A')} - {row.get('asset', 'N/A')} - {row.get('status', 'N/A')}")
                    
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
        else:
            print(f"❌ CSV file not found!")
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()

def check_csv_file():
    """Check the current state of the CSV file"""
    print(f"\n" + "=" * 80)
    print("CSV FILE CHECK")
    print("=" * 80)
    
    setups_path = "runs/setups.csv"
    if not os.path.exists(setups_path):
        print(f"❌ CSV file not found: {setups_path}")
        return
    
    try:
        df = pd.read_csv(setups_path)
        print(f"✅ CSV file loaded: {len(df)} rows")
        
        # Check recent setups
        if 'created_at' in df.columns:
            df['created_at_ts'] = pd.to_datetime(df['created_at'], errors='coerce')
            recent = df.sort_values('created_at_ts', ascending=False).head(5)
            
            print(f"\nRecent setups:")
            for idx, row in recent.iterrows():
                print(f"  {row.get('created_at', 'N/A')} - {row.get('id', 'N/A')} - {row.get('asset', 'N/A')} - {row.get('status', 'N/A')}")
        
        # Check for setups with entry around 4406
        similar_setups = df[abs(df['entry'] - 4406.90) <= 5]
        if len(similar_setups) > 0:
            print(f"\nSetups with entry around 4406.90:")
            for idx, row in similar_setups.iterrows():
                print(f"  {row.get('id', 'N/A')} - Entry: {row.get('entry', 'N/A')} - Status: {row.get('status', 'N/A')}")
        else:
            print(f"\nNo setups found with entry around 4406.90")
    
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

def main():
    """Run the test"""
    print("🔍 MANUAL SETUP CREATION TEST")
    
    # Test setup creation
    test_manual_setup_creation()
    
    # Check CSV file
    check_csv_file()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("This test will:")
    print("1. Test manual setup creation step by step")
    print("2. Verify the setup was saved")
    print("3. Check the CSV file state")

if __name__ == "__main__":
    main()
