#!/bin/bash

# Alpha12_24 Tracker Daemon Stop Script

echo "🛑 Stopping Alpha12_24 Tracker Daemon..."

# Check if PID file exists
if [ -f /tmp/alpha_tracker.pid ]; then
    TRACKER_PID=$(cat /tmp/alpha_tracker.pid)
    
    # Check if process is still running
    if ps -p $TRACKER_PID > /dev/null; then
        echo "📊 Stopping tracker with PID: $TRACKER_PID"
        kill $TRACKER_PID
        
        # Wait a moment and check if it stopped
        sleep 2
        if ps -p $TRACKER_PID > /dev/null; then
            echo "⚠️  Process still running, force killing..."
            kill -9 $TRACKER_PID
        fi
        
        echo "✅ Tracker stopped"
    else
        echo "ℹ️  Tracker process not running (PID: $TRACKER_PID)"
    fi
    
    # Clean up PID file
    rm -f /tmp/alpha_tracker.pid
else
    echo "ℹ️  No PID file found, checking for running tracker processes..."
    
    # Find and kill any running tracker processes
    PIDS=$(pgrep -f "python -m src.daemon.tracker")
    if [ -n "$PIDS" ]; then
        echo "📊 Found tracker processes: $PIDS"
        echo $PIDS | xargs kill
        echo "✅ Tracker processes stopped"
    else
        echo "ℹ️  No tracker processes found"
    fi
fi

echo ""
echo "📝 Log file remains at: /tmp/alpha_tracker.log"
echo "🔍 Check logs: tail -20 /tmp/alpha_tracker.log"
