#!/bin/bash
# Setup Data Validation Cron Job
# Run this script regularly to validate and fix corrupted setup data

# Set up environment
cd /home/ubuntu/alpha12_24
source venv/bin/activate

# Set Python path
export PYTHONPATH=/home/ubuntu/alpha12_24

# Run validation
echo "=== Setup Validation Cron Job - $(date) ==="
python3 src/utils/setup_validator.py

# Log the result
if [ $? -eq 0 ]; then
    echo "✅ Setup validation completed successfully"
else
    echo "❌ Setup validation failed"
    # Send alert if validation fails
    # You can add Telegram alert here if needed
fi

echo "=== Validation Complete - $(date) ==="
