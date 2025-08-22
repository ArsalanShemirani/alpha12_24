# Setup Validation System

## Overview

The Setup Validation System prevents and automatically fixes corrupted setup data that could cause issues like:
- Setups being incorrectly marked as "triggered" without proper entry price validation
- Missing required fields (unique_id, created_at, etc.)
- Invalid price data
- Inconsistent status values

## Components

### 1. Real-time Validation

#### Autosignal Validation (`src/daemon/autosignal.py`)
- **Location**: Lines 1070-1100
- **Trigger**: Before saving any new setup
- **Checks**:
  - Missing setup ID
  - Missing unique ID
  - Missing created_at timestamp
  - Invalid status (must be "pending")
  - Invalid price data (entry, stop, target > 0)
- **Action**: Prevents saving and sends Telegram alert if validation fails

#### Dashboard Validation (`src/dashboard/app.py`)
- **Location**: Lines 1700-1730
- **Trigger**: Before saving manual setups
- **Checks**: Same as autosignal validation
- **Action**: Shows error message and prevents setup creation

#### Tracker Validation (`src/daemon/tracker.py`)
- **Location**: Lines 590-650
- **Trigger**: Before processing any setups
- **Checks**:
  - All required fields present
  - Status consistency (triggered setups must have triggered_at)
  - Valid price data
- **Action**: Automatically fixes common issues or marks as cancelled

### 2. Standalone Validator Tool

#### Setup Validator (`src/utils/setup_validator.py`)
- **Purpose**: Comprehensive validation and repair tool
- **Features**:
  - Validates all setups in `runs/setups.csv`
  - Creates automatic backups before repairs
  - Fixes missing unique IDs
  - Fixes missing timestamps
  - Resets corrupted triggered setups to pending
  - Cancels setups with invalid prices
- **Usage**: `python3 src/utils/setup_validator.py`

#### Validation Script (`scripts/validate_setups.sh`)
- **Purpose**: Cron job wrapper for automated validation
- **Schedule**: Every 30 minutes via crontab
- **Logging**: Outputs to `runs/setup_validation.log`

## Validation Rules

### Required Fields
1. **id**: Setup identifier (non-empty)
2. **unique_id**: Human-readable unique identifier (non-empty)
3. **created_at**: ISO timestamp with timezone (non-empty)
4. **status**: Must be one of: "pending", "triggered", "executed", "completed", "cancelled", "expired"
5. **entry**: Positive number > 0
6. **stop**: Positive number > 0
7. **target**: Positive number > 0

### Status Consistency Rules
1. **triggered** status requires non-empty `triggered_at` timestamp
2. **pending** status should have empty `triggered_at`
3. **completed** status should have non-empty `triggered_at`

### Price Validation Rules
1. All prices must be positive numbers
2. Entry price must be between stop and target (for valid RR)
3. Stop and target must be on opposite sides of entry

## Automatic Fixes

### Missing Unique ID
- **Issue**: `unique_id` field is empty
- **Fix**: Generate new unique ID using format: `{asset}-{interval}-{direction}-{timestamp}`
- **Example**: `ETHUSDT-4h-SHORT-20250822-1316`

### Missing Created At
- **Issue**: `created_at` field is empty
- **Fix**: Set to current timestamp in Malaysia timezone
- **Format**: ISO format with timezone

### Missing Status
- **Issue**: `status` field is empty
- **Fix**: Set to "pending" (default state)

### Triggered Without Timestamp
- **Issue**: Status is "triggered" but `triggered_at` is empty
- **Fix**: Reset to "pending" status and clear `triggered_at`

### Invalid Prices
- **Issue**: Entry, stop, or target price is invalid (≤ 0 or NaN)
- **Fix**: Mark setup as "cancelled"

## Monitoring and Alerts

### Telegram Alerts
- **Validation Failures**: Sent when setup creation fails validation
- **Save Failures**: Sent when setup cannot be saved to CSV
- **Format**: Includes setup data and specific error details

### Logging
- **Real-time**: All validation events logged to console
- **Cron Job**: Validation results logged to `runs/setup_validation.log`
- **Backup**: Automatic backups created before repairs

## Cron Job Setup

### Current Schedule
```bash
*/30 * * * * /home/ubuntu/alpha12_24/scripts/validate_setups.sh >> /home/ubuntu/alpha12_24/runs/setup_validation.log 2>&1
```

### Manual Execution
```bash
# Run validation manually
./scripts/validate_setups.sh

# Run validator tool directly
python3 src/utils/setup_validator.py
```

## Prevention Measures

### 1. Input Validation
- All setup creation points validate data before saving
- Prevents corrupted data from entering the system

### 2. Real-time Monitoring
- Tracker validates data before processing
- Automatically fixes common issues

### 3. Regular Audits
- Cron job runs every 30 minutes
- Comprehensive validation of all setups

### 4. Automatic Backups
- Backup created before any repairs
- Timestamped backup files for recovery

## Troubleshooting

### Common Issues

#### "Missing unique ID" Errors
- **Cause**: Old setups created before unique_id system
- **Solution**: Validator automatically generates new IDs

#### "Invalid status" Errors
- **Cause**: CSV corruption or manual editing
- **Solution**: Reset to appropriate status based on other fields

#### "Missing triggered_at" for Triggered Setups
- **Cause**: Data corruption during trigger process
- **Solution**: Reset to pending status

### Manual Recovery
```bash
# Check validation log
tail -f runs/setup_validation.log

# Run manual validation
python3 src/utils/setup_validator.py

# Check backup files
ls -la runs/setups.csv.backup.*

# Restore from backup if needed
cp runs/setups.csv.backup.20250822_131643 runs/setups.csv
```

## Benefits

1. **Prevents False Triggers**: Ensures setups only trigger when entry price is actually reached
2. **Data Integrity**: Maintains consistent and valid setup data
3. **Automatic Recovery**: Self-healing system that fixes common issues
4. **Monitoring**: Continuous validation with alerts for critical issues
5. **Backup Safety**: Automatic backups prevent data loss during repairs

## Future Enhancements

1. **Enhanced Price Validation**: Check for logical price relationships
2. **Performance Metrics**: Track validation success rates
3. **Advanced Alerts**: More sophisticated notification system
4. **Database Migration**: Move from CSV to proper database for better data integrity
