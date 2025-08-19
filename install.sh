#!/bin/bash

# Alpha12_24 Installation Script
echo "🚀 Installing Alpha12_24 Trading System..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt

# Ask if user wants to install development dependencies
read -p "🤔 Install development dependencies for testing? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Ask if user wants to install optional dependencies
echo ""
echo "🔧 Optional Dependencies:"
echo "1. XGBoost (enhanced ML performance)"
echo "2. TA-Lib (advanced technical indicators)"
echo "3. Statsmodels (trendline plotting)"
echo "4. All optional dependencies"
echo "5. Skip optional dependencies"

read -p "Choose option (1-5): " -n 1 -r
echo

case $REPLY in
    1)
        echo "📊 Installing XGBoost..."
        pip install xgboost
        ;;
    2)
        echo "📈 Installing TA-Lib..."
        echo "Note: TA-Lib may require system dependencies"
        pip install TA-Lib
        ;;
    3)
        echo "📉 Installing Statsmodels..."
        pip install statsmodels
        ;;
    4)
        echo "📦 Installing all optional dependencies..."
        pip install xgboost TA-Lib statsmodels
        ;;
    5)
        echo "⏭️ Skipping optional dependencies"
        ;;
    *)
        echo "⏭️ Skipping optional dependencies"
        ;;
esac

# Set up environment variables
echo "🔧 Setting up environment variables..."
export PYTHONPATH=$(pwd)
export BINANCE_FAPI_DISABLE=1

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p runs
mkdir -p artifacts
mkdir -p logs

# Test installation
echo "🧪 Testing installation..."
python -c "from src.core.config import config; print('✅ Configuration loaded successfully')"
python -c "from src.features.engine import FeatureEngine; print('✅ Feature engine imported successfully')"
python -c "from src.models.train import ModelTrainer; print('✅ Model trainer imported successfully')"

echo ""
echo "🎉 Alpha12_24 installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Configure your settings in config.yaml"
echo "2. Start the dashboard: streamlit run src/dashboard/app.py"
echo "3. Run tests: make test"
echo ""
echo "📚 Documentation: README.md"
echo "🐛 Issues: Check the logs directory for any errors"
