@echo off
REM Alpha12_24 Installation Script for Windows

echo 🚀 Installing Alpha12_24 Trading System...

REM Check if Python 3.11+ is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install core dependencies
echo 📚 Installing core dependencies...
pip install -r requirements.txt

REM Ask if user wants to install development dependencies
set /p dev_deps="🤔 Install development dependencies for testing? (y/n): "
if /i "%dev_deps%"=="y" (
    echo 🧪 Installing development dependencies...
    pip install -r requirements-dev.txt
)

REM Ask if user wants to install optional dependencies
echo.
echo 🔧 Optional Dependencies:
echo 1. XGBoost (enhanced ML performance)
echo 2. TA-Lib (advanced technical indicators)
echo 3. Statsmodels (trendline plotting)
echo 4. All optional dependencies
echo 5. Skip optional dependencies
echo.

set /p opt_choice="Choose option (1-5): "

if "%opt_choice%"=="1" (
    echo 📊 Installing XGBoost...
    pip install xgboost
) else if "%opt_choice%"=="2" (
    echo 📈 Installing TA-Lib...
    echo Note: TA-Lib may require system dependencies
    pip install TA-Lib
) else if "%opt_choice%"=="3" (
    echo 📉 Installing Statsmodels...
    pip install statsmodels
) else if "%opt_choice%"=="4" (
    echo 📦 Installing all optional dependencies...
    pip install xgboost TA-Lib statsmodels
) else (
    echo ⏭️ Skipping optional dependencies
)

REM Set up environment variables
echo 🔧 Setting up environment variables...
set PYTHONPATH=%cd%
set BINANCE_FAPI_DISABLE=1

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "runs" mkdir runs
if not exist "artifacts" mkdir artifacts
if not exist "logs" mkdir logs

REM Test installation
echo 🧪 Testing installation...
python -c "from src.core.config import config; print('✅ Configuration loaded successfully')"
python -c "from src.features.engine import FeatureEngine; print('✅ Feature engine imported successfully')"
python -c "from src.models.train import ModelTrainer; print('✅ Model trainer imported successfully')"

echo.
echo 🎉 Alpha12_24 installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Configure your settings in config.yaml
echo 2. Start the dashboard: streamlit run src/dashboard/app.py
echo 3. Run tests: make test
echo.
echo 📚 Documentation: README.md
echo 🐛 Issues: Check the logs directory for any errors
echo.
pause
