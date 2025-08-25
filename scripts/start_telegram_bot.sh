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

# Run the bot
exec python3 scripts/tg_bot.py
