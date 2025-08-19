#!/usr/bin/env python3
"""
Test script for setup lifecycle tracking and Telegram alerts
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.daemon.tracker import _load_setups_df, _save_setups_df, _tg_send, _SETUP_FIELDS

def create_test_setup():
    """Create a test setup for lifecycle testing"""
    print("Creating test setup...")
    
    # Create a test setup
    test_setup = {
        "id": f"test_{int(time.time())}",
        "asset": "BTCUSDT",
        "interval": "5m",
        "direction": "long",
        "entry": 50000.0,
        "stop": 49500.0,
        "target": 51000.0,
        "rr": 2.0,
        "size_units": 0.001,
        "notional_usd": 50.0,
        "leverage": 1.0,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
        "status": "pending",
        "confidence": 0.75,
        "trigger_rule": "touch",
        "entry_buffer_bps": 5.0,
        "origin": "manual"
    }
    
    # Load existing setups
    try:
        df = _load_setups_df()
    except Exception:
        df = pd.DataFrame(columns=_SETUP_FIELDS)
    
    # Add test setup
    df = pd.concat([df, pd.DataFrame([test_setup])], ignore_index=True)
    _save_setups_df(df)
    
    print(f"✅ Test setup created: {test_setup['id']}")
    return test_setup['id']

def test_telegram_alerts():
    """Test Telegram alerts"""
    print("\nTesting Telegram alerts...")
    
    # Test different alert types
    alerts = [
        "🧪 Test alert from script\\nThis confirms Telegram alerts are working!",
        "🎯 Setup TRIGGERED\\nBTCUSDT 5m LONG\\nEntry: 50000.00 → 50010.00\\nStop: 49500.00 | Target: 51000.00",
        "✅ Setup TARGET\\nBTCUSDT 5m LONG\\nEntry: 50000.00 → Exit: 51000.00\\nPnL: 2.00%",
        "❌ Setup STOP\\nBTCUSDT 5m LONG\\nEntry: 50000.00 → Exit: 49500.00\\nPnL: -1.00%",
        "⏰ Setup TIMEOUT\\nBTCUSDT 5m LONG\\nEntry: 50000.00 → Exit: 50005.00\\nPnL: 0.01%"
    ]
    
    for i, alert in enumerate(alerts, 1):
        print(f"Sending alert {i}/5...")
        _tg_send(alert)
        time.sleep(2)  # Wait between alerts
    
    print("✅ Telegram alerts test completed")

def check_setup_status():
    """Check current setup status"""
    print("\nChecking setup status...")
    
    try:
        df = _load_setups_df()
        if df.empty:
            print("No setups found")
            return
        
        print(f"Total setups: {len(df)}")
        status_counts = df["status"].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
            
        # Show recent setups
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            recent = df.sort_values("created_at", ascending=False).head(3)
            print("\nRecent setups:")
            for _, row in recent.iterrows():
                print(f"  {row['id']}: {row['status']} - {row['asset']} {row['direction']}")
                
    except Exception as e:
        print(f"Error checking setup status: {e}")

def main():
    print("🧪 Setup Lifecycle Test Script")
    print("=" * 50)
    
    # Check current status
    check_setup_status()
    
    # Create test setup
    setup_id = create_test_setup()
    
    # Test Telegram alerts
    test_telegram_alerts()
    
    # Check status again
    check_setup_status()
    
    print("\n✅ Test completed!")
    print(f"Test setup ID: {setup_id}")
    print("Check the dashboard to see the setup lifecycle in action!")

if __name__ == "__main__":
    main()
