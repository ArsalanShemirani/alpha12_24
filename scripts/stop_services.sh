#!/bin/bash

# Change to the project directory
cd /home/ubuntu/alpha12_24

echo "Stopping Alpha12 services..."

# Stop Telegram bot
if [ -f runs/telegram_bot.pid ]; then
    TG_BOT_PID=$(cat runs/telegram_bot.pid)
    if ps -p $TG_BOT_PID > /dev/null 2>&1; then
        echo "Stopping Telegram bot (PID: $TG_BOT_PID)..."
        kill $TG_BOT_PID
        rm -f runs/telegram_bot.pid
    else
        echo "Telegram bot process not found"
        rm -f runs/telegram_bot.pid
    fi
else
    echo "Telegram bot PID file not found"
fi

# Stop queue executor
if [ -f runs/queue_executor.pid ]; then
    QUEUE_EXEC_PID=$(cat runs/queue_executor.pid)
    if ps -p $QUEUE_EXEC_PID > /dev/null 2>&1; then
        echo "Stopping queue executor (PID: $QUEUE_EXEC_PID)..."
        kill $QUEUE_EXEC_PID
        rm -f runs/queue_executor.pid
    else
        echo "Queue executor process not found"
        rm -f runs/queue_executor.pid
    fi
else
    echo "Queue executor PID file not found"
fi

echo "All services stopped!"
