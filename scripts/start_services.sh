#!/bin/bash
set -e

# Change to the project directory
cd /home/ubuntu/alpha12_24

# Source the virtual environment
source venv/bin/activate

# Set environment variables
export TG_BOT_TOKEN="8119234697:AAE7dGn707CEXZo0hyzSHpzCkQtIklEhDkE"
export TG_CHAT_ID="-4873132631"
export TG_ALLOWLIST="-4873132631,123456789"
export RUNS_DIR="runs"
export SETUPS_CSV="runs/setups.csv"
export CMD_QUEUE_DIR="runs/command_queue"
export TELEGRAM_AUDIT="runs/telegram_audit.csv"

# Ensure directories exist
mkdir -p "$CMD_QUEUE_DIR"
mkdir -p "$(dirname "$TELEGRAM_AUDIT")"

echo "Starting Alpha12 services..."

# Start Telegram bot
echo "Starting Telegram bot..."
nohup python3 scripts/tg_bot.py > runs/telegram_bot.log 2>&1 &
TG_BOT_PID=$!
echo "Telegram bot started with PID: $TG_BOT_PID"

# Start queue executor
echo "Starting queue executor..."
nohup python3 scripts/queue_executor_stub.py > runs/queue_executor.log 2>&1 &
QUEUE_EXEC_PID=$!
echo "Queue executor started with PID: $QUEUE_EXEC_PID"

# Save PIDs for later use
echo $TG_BOT_PID > runs/telegram_bot.pid
echo $QUEUE_EXEC_PID > runs/queue_executor.pid

echo "All services started successfully!"
echo "Telegram bot PID: $TG_BOT_PID"
echo "Queue executor PID: $QUEUE_EXEC_PID"
echo "Logs: runs/telegram_bot.log, runs/queue_executor.log"
