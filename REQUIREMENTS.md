# 📋 Alpha12_24 Requirements Documentation

## 🎯 Overview

This document provides a comprehensive guide to all dependencies and requirements for the Alpha12_24 cryptocurrency trading system.

## 🐍 Python Requirements

- **Python Version**: 3.11 or higher
- **Virtual Environment**: Recommended (`.venv`)

## 📦 Core Dependencies

### Data Science & Machine Learning
- **pandas** (≥2.3.0): Data manipulation and analysis
- **numpy** (≥2.3.0): Numerical computing
- **scikit-learn** (≥1.7.0): Machine learning algorithms
- **scipy** (≥1.16.0): Scientific computing

### Web Framework & Dashboard
- **streamlit** (≥1.48.0): Web application framework
- **plotly** (≥6.3.0): Interactive plotting

### HTTP & API Integration
- **requests** (≥2.32.0): HTTP library for API calls
- **httpx** (≥0.28.0): Modern HTTP client

### Configuration & Data Formats
- **PyYAML** (≥6.0.0): YAML configuration files
- **python-dateutil** (≥2.9.0): Date utilities

### Timezone & Date Handling
- **pytz** (≥2025.0): Timezone support
- **tzdata** (≥2025.0): Timezone database

### Data Processing & Utilities
- **joblib** (≥1.5.0): Parallel computing
- **tqdm** (≥4.67.0): Progress bars

### Resilience & Retry Logic
- **tenacity** (≥9.1.0): Retry logic for API calls

## 🧪 Development Dependencies

### Testing Framework
- **pytest** (≥8.0.0): Testing framework
- **pytest-cov** (≥4.0.0): Coverage reporting
- **pytest-mock** (≥3.10.0): Mocking utilities

### HTTP Testing
- **responses** (≥0.24.0): Mock HTTP responses

### Code Quality
- **black** (≥23.0.0): Code formatting
- **flake8** (≥6.0.0): Linting
- **isort** (≥5.12.0): Import sorting

### Type Checking
- **mypy** (≥1.0.0): Static type checking

## 🔧 Optional Dependencies

### Enhanced Machine Learning
- **xgboost** (≥2.0.0): Gradient boosting (optional)
  - Provides enhanced ML performance
  - Falls back to scikit-learn if not available

### Advanced Technical Analysis
- **TA-Lib** (≥0.4.0): Technical analysis library (optional)
  - Requires system-level dependencies
  - Falls back to simplified indicators if not available

### Statistical Analysis
- **statsmodels** (≥0.14.0): Statistical modeling (optional)
  - Used for trendline plotting in plotly
  - Falls back to basic scatter plots if not available

## 🚀 Installation Methods

### 1. Quick Installation (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd alpha12_24

# Run installation script
./install.sh  # Linux/macOS
install.bat   # Windows
```

### 2. Manual Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install optional dependencies (optional)
pip install xgboost TA-Lib statsmodels
```

### 3. System Dependencies for TA-Lib

#### macOS
```bash
brew install ta-lib
pip install TA-Lib
```

#### Ubuntu/Debian
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

#### Windows
```bash
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.24-cp311-cp311-win_amd64.whl
```

## 🔍 Dependency Verification

### Test Installation
```bash
# Test core functionality
python -c "from src.core.config import config; print('✅ Config loaded')"
python -c "from src.features.engine import FeatureEngine; print('✅ Features loaded')"
python -c "from src.models.train import ModelTrainer; print('✅ Models loaded')"

# Run tests
make test
```

### Check Optional Dependencies
```python
# Check XGBoost
try:
    import xgboost
    print("✅ XGBoost available")
except ImportError:
    print("⚠️ XGBoost not available")

# Check TA-Lib
try:
    import talib
    print("✅ TA-Lib available")
except ImportError:
    print("⚠️ TA-Lib not available")

# Check Statsmodels
try:
    import statsmodels
    print("✅ Statsmodels available")
except ImportError:
    print("⚠️ Statsmodels not available")
```

## 📊 Dependency Categories

### Essential (Required)
- Core data science libraries
- Web framework
- HTTP clients
- Configuration management

### Recommended (Development)
- Testing framework
- Code quality tools
- Type checking

### Optional (Enhanced Features)
- XGBoost for better ML performance
- TA-Lib for advanced technical indicators
- Statsmodels for statistical analysis

## 🔧 Environment Variables

```bash
# Required
export PYTHONPATH=$(pwd)
export BINANCE_FAPI_DISABLE=1

# Optional
export TG_BOT_TOKEN="your_telegram_bot_token"
export TG_CHAT_ID="your_telegram_chat_id"
```

## 🐛 Troubleshooting

### Common Issues

1. **TA-Lib Installation Fails**
   - Install system dependencies first
   - Use pre-compiled wheels for Windows
   - System falls back to simplified indicators

2. **XGBoost Installation Fails**
   - System falls back to scikit-learn models
   - No functionality loss, just reduced performance

3. **Streamlit Not Found**
   - Ensure virtual environment is activated
   - Reinstall: `pip install streamlit`

4. **Import Errors**
   - Check PYTHONPATH is set correctly
   - Ensure all dependencies are installed
   - Verify Python version (3.11+)

### Performance Optimization

- **XGBoost**: 20-30% better ML performance
- **TA-Lib**: Faster technical indicators
- **Statsmodels**: Enhanced plotting capabilities

## 📈 Dependency Impact

| Dependency | Impact | Fallback |
|------------|--------|----------|
| XGBoost | High ML performance | scikit-learn |
| TA-Lib | Advanced indicators | Simplified indicators |
| Statsmodels | Trendline plots | Basic scatter plots |
| pytest | Testing | Manual testing |
| black/flake8 | Code quality | None |

## 🔄 Updates

### Update Dependencies
```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Update specific dependency
pip install --upgrade pandas

# Generate new lock file
pip freeze > requirements.lock.txt
```

### Check for Updates
```bash
# Check outdated packages
pip list --outdated

# Update with pip-review
pip install pip-review
pip-review --auto
```

## 📚 Additional Resources

- [Python Package Index](https://pypi.org/)
- [TA-Lib Documentation](https://mrjbq7.github.io/ta-lib/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [pytest Documentation](https://docs.pytest.org/)
