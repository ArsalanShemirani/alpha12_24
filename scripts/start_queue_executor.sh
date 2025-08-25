#!/bin/bash
set -e

# Change to the project directory
cd /home/ubuntu/alpha12_24

# Source the virtual environment
source venv/bin/activate

# Set environment variables
export RUNS_DIR="runs"
export SETUPS_CSV="runs/setups.csv"
export CMD_QUEUE_DIR="runs/command_queue"
export TELEGRAM_AUDIT="runs/telegram_audit.csv"

# Run the queue executor
exec python3 scripts/queue_executor_stub.py
