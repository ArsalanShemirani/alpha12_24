#!/bin/bash

# Alpha12 Background Analysis Startup Script
# This script sets up the background analysis daemon to run on 15m data
# while keeping autosignal on 1h intervals

set -e

echo "🚀 Alpha12 Background Analysis Setup"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "src/daemon/background_analysis.py" ]; then
    echo "❌ Error: Please run this script from the alpha12_24 directory"
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env..."
    source load_env.sh
else
    echo "❌ Error: .env file not found"
    exit 1
fi

# Add background analysis configuration to .env if not present
if ! grep -q "ALPHA12_TRAINING_INTERVAL" .env; then
    echo "📝 Adding background analysis configuration to .env..."
    cat >> .env << EOF

# --- Background Analysis Configuration
ALPHA12_TRAINING_INTERVAL=15m
ALPHA12_ANALYSIS_SLEEP=300
ALPHA12_ANALYSIS_DAYS=120
EOF
    echo "✅ Added background analysis configuration"
fi

# Test the background analysis script
echo "🧪 Testing background analysis script..."
PYTHONPATH=$(pwd) python -c "
from src.daemon.background_analysis import analyze_asset
import os
assets = os.getenv('ALPHA12_SYMBOL', 'BTCUSDT').split(',')
print(f'Testing analysis for {assets[0]} 15m...')
result = analyze_asset(assets[0], '15m', days=30)
if result:
    print(f'✅ Analysis successful: {result[\"accuracy\"]:.3f} accuracy, {result[\"samples\"]} samples')
else:
    print('❌ Analysis failed')
"

# Show current configuration
echo ""
echo "📊 Current Configuration:"
echo "  Autosignal Interval: ${ALPHA12_INTERVAL:-1h}"
echo "  Training Interval: ${ALPHA12_TRAINING_INTERVAL:-15m}"
echo "  Analysis Sleep: ${ALPHA12_ANALYSIS_SLEEP:-300} seconds"
echo "  Assets: ${ALPHA12_SYMBOL:-BTCUSDT}"
echo ""

# Show how to run the services
echo "🎯 Service Management:"
echo ""
echo "1. Start background analysis daemon:"
echo "   PYTHONPATH=\$(pwd) python src/daemon/background_analysis.py"
echo ""
echo "2. Start autosignal daemon (1h intervals):"
echo "   PYTHONPATH=\$(pwd) python src/daemon/autosignal.py"
echo ""
echo "3. Start tracker daemon:"
echo "   PYTHONPATH=\$(pwd) python src/daemon/tracker.py"
echo ""
echo "4. For systemd deployment:"
echo "   sudo cp ops/alpha12-background-analysis.service /etc/systemd/system/"
echo "   sudo systemctl enable alpha12-background-analysis"
echo "   sudo systemctl start alpha12-background-analysis"
echo ""

echo "✅ Background analysis setup complete!"
echo ""
echo "💡 The system will now:"
echo "   • Generate autosignals on 1h+ intervals (avoiding noise)"
echo "   • Run continuous analysis on 15m data for model improvement"
echo "   • Save analysis results to runs/background_analysis.csv"
echo "   • Retrain models from live logs when available"
