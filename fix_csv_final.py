#!/usr/bin/env python3
"""
Script to fix CSV file and prevent future corruption
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import csv
from datetime import datetime

def fix_csv_final():
    """Fix the CSV file and prevent future corruption"""
    print("=" * 80)
    print("FINAL CSV FIX")
    print("=" * 80)
    
    setups_path = "runs/setups.csv"
    backup_path = "runs/setups_backup_final.csv"
    
    if not os.path.exists(setups_path):
        print(f"❌ CSV file not found: {setups_path}")
        return
    
    print(f"✅ Found CSV file: {setups_path}")
    
    # Create backup
    try:
        import shutil
        shutil.copy2(setups_path, backup_path)
        print(f"✅ Created backup: {backup_path}")
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return
    
    # Read the entire file as text and fix line breaks
    print(f"\nReading and fixing CSV file...")
    
    try:
        with open(setups_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"File size: {len(content)} characters")
        
        # Split into lines and fix line breaks in data
        lines = content.split('\n')
        print(f"Total lines: {len(lines)}")
        
        # Find the header line
        header_line = None
        for i, line in enumerate(lines):
            if line.startswith('id,asset,interval'):
                header_line = i
                break
        
        if header_line is None:
            print(f"❌ Could not find header line")
            return
        
        print(f"Header found at line {header_line + 1}")
        header = lines[header_line].split(',')
        expected_fields = len(header)
        print(f"Expected fields: {expected_fields}")
        print(f"Header: {header}")
        
        # Process data lines and fix line breaks
        valid_rows = [header]
        corrupted_rows = []
        
        # Start from the line after header
        i = header_line + 1
        while i < len(lines):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                i += 1
                continue
            
            # Check if this line has the right number of fields
            fields = line.split(',')
            
            if len(fields) == expected_fields:
                # This line is complete
                valid_rows.append(fields)
                i += 1
            elif len(fields) < expected_fields:
                # This line is incomplete, try to merge with next lines
                print(f"Line {i+1}: Incomplete ({len(fields)} fields), merging...")
                
                merged_line = line
                j = i + 1
                while j < len(lines) and len(merged_line.split(',')) < expected_fields:
                    next_line = lines[j].strip()
                    if next_line:
                        merged_line += next_line
                    j += 1
                
                merged_fields = merged_line.split(',')
                if len(merged_fields) == expected_fields:
                    print(f"  ✅ Successfully merged into complete row")
                    valid_rows.append(merged_fields)
                else:
                    print(f"  ❌ Could not merge into complete row ({len(merged_fields)} fields)")
                    corrupted_rows.append((i+1, merged_fields))
                
                i = j
            else:
                # This line has too many fields
                print(f"Line {i+1}: Too many fields ({len(fields)}), skipping...")
                corrupted_rows.append((i+1, fields))
                i += 1
        
        print(f"\nResults:")
        print(f"  Valid rows: {len(valid_rows) - 1}")  # Exclude header
        print(f"  Corrupted rows: {len(corrupted_rows)}")
        
        # Write fixed CSV with proper escaping
        print(f"\nWriting fixed CSV with proper escaping...")
        with open(setups_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields to prevent issues
            for row in valid_rows:
                # Clean any newlines or special characters in the data
                cleaned_row = []
                for field in row:
                    if isinstance(field, str):
                        # Remove newlines and carriage returns
                        cleaned_field = field.replace('\n', ' ').replace('\r', ' ').strip()
                        cleaned_row.append(cleaned_field)
                    else:
                        cleaned_row.append(field)
                writer.writerow(cleaned_row)
        
        print(f"✅ Fixed CSV written successfully!")
        
        # Verify the fix
        print(f"\nVerifying fix...")
        try:
            df = pd.read_csv(setups_path)
            print(f"✅ CSV file can be read: {len(df)} rows")
            
            # Show recent setups
            if 'created_at' in df.columns:
                df['created_at_ts'] = pd.to_datetime(df['created_at'], errors='coerce')
                recent = df.sort_values('created_at_ts', ascending=False).head(5)
                
                print(f"\nRecent setups:")
                for idx, row in recent.iterrows():
                    print(f"  {row.get('created_at', 'N/A')} - {row.get('id', 'N/A')} - {row.get('asset', 'N/A')} - {row.get('status', 'N/A')}")
                
        except Exception as e:
            print(f"❌ CSV still corrupted: {e}")
    
    except Exception as e:
        print(f"❌ Error fixing CSV: {e}")
        import traceback
        traceback.print_exc()

def update_csv_writing():
    """Update CSV writing to prevent future corruption"""
    print(f"\n" + "=" * 80)
    print("UPDATING CSV WRITING")
    print("=" * 80)
    
    # Update the _append_setup_row function to use QUOTE_ALL
    dashboard_path = "src/dashboard/app.py"
    autosignal_path = "src/daemon/autosignal.py"
    
    print(f"Updating {dashboard_path}...")
    try:
        with open(dashboard_path, 'r') as f:
            content = f.read()
        
        # Replace QUOTE_MINIMAL with QUOTE_ALL
        content = content.replace('quoting=csv.QUOTE_MINIMAL', 'quoting=csv.QUOTE_ALL')
        
        with open(dashboard_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {dashboard_path}")
    except Exception as e:
        print(f"❌ Error updating {dashboard_path}: {e}")
    
    print(f"Updating {autosignal_path}...")
    try:
        with open(autosignal_path, 'r') as f:
            content = f.read()
        
        # Replace QUOTE_MINIMAL with QUOTE_ALL
        content = content.replace('quoting=csv.QUOTE_MINIMAL', 'quoting=csv.QUOTE_ALL')
        
        with open(autosignal_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {autosignal_path}")
    except Exception as e:
        print(f"❌ Error updating {autosignal_path}: {e}")

def main():
    """Run the CSV fix"""
    print("🔧 FINAL CSV FIX")
    
    # Fix the CSV corruption
    fix_csv_final()
    
    # Update CSV writing to prevent future corruption
    update_csv_writing()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("This script will:")
    print("1. Fix the corrupted CSV file")
    print("2. Update CSV writing to prevent future corruption")
    print("3. Ensure manual setups are saved correctly")

if __name__ == "__main__":
    main()
