#!/usr/bin/env python3
"""
Setup Data Validator and Repair Tool
Validates and fixes corrupted setup data in setups.csv
"""

import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Define paths
SETUPS_FILE = "runs/setups.csv"
MY_TZ = "Asia/Kuala_Lumpur"

def validate_setup_data():
    """Validate all setup data and report issues"""
    if not os.path.exists(SETUPS_FILE):
        print(f"❌ Setups file not found: {SETUPS_FILE}")
        return False
    
    try:
        df = pd.read_csv(SETUPS_FILE)
        print(f"📊 Loaded {len(df)} setups from {SETUPS_FILE}")
    except Exception as e:
        print(f"❌ Failed to load setups file: {e}")
        return False
    
    issues = []
    corrupted_count = 0
    
    for idx, row in df.iterrows():
        setup_id = row.get('id', f'ROW-{idx}')
        row_issues = []
        
        # Check required fields
        if pd.isna(row.get('id')) or str(row.get('id', '')).strip() == '':
            row_issues.append("Missing setup ID")
        if pd.isna(row.get('unique_id')) or str(row.get('unique_id', '')).strip() == '':
            row_issues.append("Missing unique ID")
        if pd.isna(row.get('created_at')) or str(row.get('created_at', '')).strip() == '':
            row_issues.append("Missing created_at timestamp")
        if pd.isna(row.get('status')) or str(row.get('status', '')).strip() == '':
            row_issues.append("Missing status")
        
        # Check price fields
        if pd.isna(row.get('entry')) or float(row.get('entry', 0)) <= 0:
            row_issues.append(f"Invalid entry price: {row.get('entry')}")
        if pd.isna(row.get('stop')) or float(row.get('stop', 0)) <= 0:
            row_issues.append(f"Invalid stop price: {row.get('stop')}")
        if pd.isna(row.get('target')) or float(row.get('target', 0)) <= 0:
            row_issues.append(f"Invalid target price: {row.get('target')}")
        
        # Check status consistency
        status = str(row.get('status', '')).strip()
        if status == 'triggered' and (pd.isna(row.get('triggered_at')) or str(row.get('triggered_at', '')).strip() == ''):
            row_issues.append("Triggered setup missing triggered_at timestamp")
        
        if row_issues:
            issues.append((idx, setup_id, row_issues))
            corrupted_count += 1
    
    if issues:
        print(f"\n🚨 Found {corrupted_count} corrupted setups:")
        for idx, setup_id, row_issues in issues:
            print(f"  {setup_id} (row {idx}): {', '.join(row_issues)}")
        return False
    else:
        print("✅ All setups are valid!")
        return True

def fix_corrupted_setups():
    """Fix corrupted setups automatically"""
    if not os.path.exists(SETUPS_FILE):
        print(f"❌ Setups file not found: {SETUPS_FILE}")
        return False
    
    # Create backup
    backup_file = f"{SETUPS_FILE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        import shutil
        shutil.copy2(SETUPS_FILE, backup_file)
        print(f"📋 Created backup: {backup_file}")
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False
    
    try:
        df = pd.read_csv(SETUPS_FILE)
        print(f"📊 Loaded {len(df)} setups")
    except Exception as e:
        print(f"❌ Failed to load setups file: {e}")
        return False
    
    fixed_count = 0
    cancelled_count = 0
    
    for idx, row in df.iterrows():
        setup_id = row.get('id', f'ROW-{idx}')
        fixes_applied = []
        
        # Fix missing unique ID
        if pd.isna(row.get('unique_id')) or str(row.get('unique_id', '')).strip() == '':
            try:
                asset = row.get('asset', 'UNKNOWN')
                interval = row.get('interval', '1h')
                direction = row.get('direction', 'long')
                unique_id = f"{asset}-{interval}-{direction.upper()}-{datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).strftime('%Y%m%d-%H%M')}"
                df.loc[idx, 'unique_id'] = unique_id
                fixes_applied.append(f"Generated unique_id: {unique_id}")
            except Exception as e:
                print(f"❌ Failed to generate unique_id for {setup_id}: {e}")
        
        # Fix missing created_at
        if pd.isna(row.get('created_at')) or str(row.get('created_at', '')).strip() == '':
            try:
                current_time = datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).isoformat()
                df.loc[idx, 'created_at'] = current_time
                fixes_applied.append(f"Set created_at: {current_time}")
            except Exception as e:
                print(f"❌ Failed to set created_at for {setup_id}: {e}")
        
        # Fix missing status
        if pd.isna(row.get('status')) or str(row.get('status', '')).strip() == '':
            df.loc[idx, 'status'] = 'pending'
            fixes_applied.append("Set status: pending")
        
        # Fix triggered setups with missing triggered_at
        status = str(row.get('status', '')).strip()
        if status == 'triggered' and (pd.isna(row.get('triggered_at')) or str(row.get('triggered_at', '')).strip() == ''):
            df.loc[idx, 'status'] = 'pending'
            df.loc[idx, 'triggered_at'] = ''
            fixes_applied.append("Reset to pending (missing triggered_at)")
        
        # Cancel setups with invalid prices
        if (pd.isna(row.get('entry')) or float(row.get('entry', 0)) <= 0 or
            pd.isna(row.get('stop')) or float(row.get('stop', 0)) <= 0 or
            pd.isna(row.get('target')) or float(row.get('target', 0)) <= 0):
            df.loc[idx, 'status'] = 'cancelled'
            fixes_applied.append("Cancelled due to invalid prices")
            cancelled_count += 1
        
        if fixes_applied:
            print(f"🔧 Fixed {setup_id}: {', '.join(fixes_applied)}")
            fixed_count += 1
    
    if fixed_count > 0:
        try:
            df.to_csv(SETUPS_FILE, index=False)
            print(f"\n✅ Fixed {fixed_count} setups, cancelled {cancelled_count} corrupted setups")
            print(f"💾 Saved fixed data to {SETUPS_FILE}")
            return True
        except Exception as e:
            print(f"❌ Failed to save fixed data: {e}")
            return False
    else:
        print("✅ No fixes needed!")
        return True

def main():
    """Main validation and repair function"""
    print("🔍 Setup Data Validator and Repair Tool")
    print("=" * 50)
    
    # First validate
    print("\n1. Validating setup data...")
    is_valid = validate_setup_data()
    
    if not is_valid:
        print("\n2. Attempting to fix corrupted setups...")
        fix_success = fix_corrupted_setups()
        
        if fix_success:
            print("\n3. Re-validating after fixes...")
            is_valid = validate_setup_data()
            
            if is_valid:
                print("\n🎉 All issues resolved!")
            else:
                print("\n⚠️  Some issues remain after fixes")
        else:
            print("\n❌ Failed to fix corrupted setups")
    else:
        print("\n🎉 No issues found!")

if __name__ == "__main__":
    main()
