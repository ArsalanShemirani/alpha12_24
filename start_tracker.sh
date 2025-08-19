#!/bin/bash

# Alpha12_24 Tracker Daemon Startup Script
# This script starts the 24/7 trade lifecycle tracker

echo "🚀 Starting Alpha12_24 Tracker Daemon..."

# Set environment variables
export PYTHONPATH=$(pwd)
export BINANCE_FAPI_DISABLE=1
export ALPHA12_SYMBOL=${ALPHA12_SYMBOL:-"BTCUSDT"}
export ALPHA12_INTERVAL=${ALPHA12_INTERVAL:-"5m"}
export ALPHA12_SLEEP=${ALPHA12_SLEEP:-"15"}

echo "📊 Configuration:"
echo "  Symbol: $ALPHA12_SYMBOL"
echo "  Interval: $ALPHA12_INTERVAL"
echo "  Sleep: ${ALPHA12_SLEEP}s"
echo "  Log: /tmp/alpha_tracker.log"

# Start the tracker in background
nohup python -m src.daemon.tracker > /tmp/alpha_tracker.log 2>&1 &

# Get the process ID
TRACKER_PID=$!
echo "✅ Tracker started with PID: $TRACKER_PID"
echo "📝 Log file: /tmp/alpha_tracker.log"
echo "🛑 To stop: kill $TRACKER_PID"

# Save PID to file for easy management
echo $TRACKER_PID > /tmp/alpha_tracker.pid
echo "💾 PID saved to /tmp/alpha_tracker.pid"

echo ""
echo "🔍 Monitor the tracker:"
echo "  tail -f /tmp/alpha_tracker.log"
echo ""
echo "📊 Check status in Streamlit dashboard"
echo "  streamlit run src/dashboard/app.py"
