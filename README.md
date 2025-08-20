# 🚀 Alpha12_24 - Advanced Algorithmic Trading System

A comprehensive algorithmic trading platform with machine learning-powered signals, real-time risk management, and automated trade execution.

## 📊 System Overview

**Alpha12_24** is a sophisticated trading system that combines:
- **Machine Learning Models** for signal generation
- **Real-time Risk Management** with dynamic position sizing
- **Automated Trade Execution** with anti-stop-hunt features
- **Comprehensive Dashboard** for monitoring and control
- **Telegram Integration** for alerts and notifications

## 🏗️ Architecture

### Core Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| **Dashboard** | 3,816 | Streamlit web interface for setup creation and monitoring |
| **Daemon** | 2,036 | Background tracker for automated trade execution |
| **Data** | 2,297 | Data collection, processing, and feature engineering |
| **Models** | 925 | Machine learning models for signal generation |
| **Trading** | 1,070 | Trading logic, position sizing, and risk management |
| **Evaluation** | 1,434 | Backtesting, performance analysis, and optimization |
| **Features** | 850 | Technical indicators and feature engineering |
| **Policy** | 574 | Trading policies and rules |
| **Backtest** | 459 | Backtesting framework |
| **Core** | 352 | Configuration and utilities |

**Total Codebase**: 18,218 lines of Python code across 139 files

## 🎯 Key Features

### 1. **Intelligent Signal Generation**
- **Multi-Model Ensemble**: Random Forest, XGBoost, and composite models
- **Feature Engineering**: 50+ technical indicators and market features
- **Confidence Scoring**: Probability-based signal confidence
- **Timeframe Optimization**: Different models for 15m, 1h, 4h, 1d intervals

### 2. **Advanced Risk Management**
- **Dynamic Position Sizing**: Risk-based and nominal-based calculations
- **Timeframe-Specific Risk**: 
  - 15m: 0.5% risk (tighter stops)
  - 1h: 1.0% risk (baseline)
  - 4h: 1.5% risk (wider stops)
  - 1d: 2.0% risk (widest stops)
- **Consistent Dollar Risk**: 2.5% of account balance per trade
- **Leverage Management**: Configurable up to 10x with safety caps

### 3. **Anti Stop-Hunt Technology**
- **Pullback Entries**: Entry below current price for longs, above for shorts
- **Buffer Adjustments**: Shift entries deeper to avoid wick fills
- **Close-Through Confirmation**: Optional bar close confirmation
- **First-Touch Resolution**: Conservative exit logic

### 4. **Real-Time Trade Execution**
- **Automated Triggering**: 15-second polling for setup activation
- **Exit Monitoring**: Continuous tracking of stop/target hits
- **Expiration Management**: Automatic timeout handling
- **Performance Logging**: Comprehensive trade history

### 5. **Comprehensive Dashboard**
- **Live Price Charts**: Interactive candlestick charts with indicators
- **Signal Analysis**: Real-time signal generation and confidence scoring
- **Setup Management**: Create, monitor, and manage trade setups
- **Performance Tracking**: Win rate, profit factor, drawdown analysis
- **Account Management**: Balance, leverage, and position tracking

### 6. **Telegram Integration**
- **Real-Time Alerts**: Setup triggers, exits, and expirations
- **Performance Updates**: Daily/weekly performance summaries
- **Error Notifications**: System status and error alerts

## 🔧 Technical Specifications

### Risk Management System
```python
# Timeframe-Specific Risk Percentages
risk_percentages = {
    "15m": 0.5,  # $2,000 notional → $10 risk
    "1h": 1.0,   # $1,000 notional → $10 risk  
    "4h": 1.5,   # $666 notional → $10 risk
    "1d": 2.0    # $500 notional → $10 risk
}

# Position Sizing Formula
Position Size = Dollar Risk / (Risk Percentage × Entry Price)
```

### Signal Generation Pipeline
1. **Data Collection**: Real-time price feeds from multiple sources
2. **Feature Engineering**: Technical indicators, market microstructure
3. **Model Prediction**: Ensemble of ML models with confidence scoring
4. **Signal Filtering**: Confidence thresholds and market conditions
5. **Setup Creation**: Entry, stop, target calculation with risk management

### Trade Lifecycle
1. **Setup Creation**: Dashboard generates trade setups with risk parameters
2. **Pending Status**: Setup waits for trigger conditions
3. **Trigger Detection**: Price reaches entry level with anti-stop-hunt logic
4. **Active Monitoring**: Continuous tracking of stop/target levels
5. **Exit Resolution**: First-touch policy for conservative exits
6. **Performance Logging**: Complete trade history with analytics

## 📈 Performance Metrics

### Key Performance Indicators
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Sum of gains / Sum of losses
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Risk-Adjusted Returns**: Sharpe ratio and Calmar ratio
- **Average Trade Duration**: Time from trigger to exit

### Risk Metrics
- **Per-Trade Risk**: Consistent 2.5% of account balance
- **Position Sizing**: Dynamic based on volatility and timeframe
- **Leverage Utilization**: Optimal use of available leverage
- **Correlation Management**: Diversification across timeframes

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- Pandas, NumPy, Scikit-learn
- Telegram Bot (optional)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd alpha12_24

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run dashboard
streamlit run src/dashboard/app.py

# Run tracker (in separate terminal)
python src/daemon/tracker.py
```

### Configuration
```python
# Key Configuration Parameters
ACCOUNT_BALANCE = 400.0          # Account balance in USD
MAX_LEVERAGE = 10                # Maximum leverage allowed
RISK_PER_TRADE_PCT = 2.5         # Risk per trade as % of balance
NOMINAL_POSITION_PCT = 25.0      # Position size as % of nominal balance
CONFIDENCE_THRESHOLD = 0.6       # Minimum confidence for signals
```

## 📊 Data Files

### Core Data Storage
- `runs/setups.csv`: Trade setups with risk parameters
- `runs/trade_history.csv`: Complete trade execution history
- `runs/features_at_signal.parquet`: Feature vectors for ML training
- `runs/daemon_heartbeat.txt`: Tracker status monitoring

### Configuration Files
- `.env`: Environment variables and API keys
- `src/core/config.py`: System configuration
- `src/core/ui_config.py`: Dashboard UI settings

## 🔄 System Workflow

### 1. Signal Generation
```
Market Data → Feature Engineering → ML Models → Signal Filtering → Setup Creation
```

### 2. Risk Management
```
Setup → Position Sizing → Risk Calculation → Leverage Check → Execution
```

### 3. Trade Execution
```
Setup Trigger → Entry Execution → Stop/Target Monitoring → Exit Resolution → Logging
```

## 🛡️ Safety Features

### Risk Controls
- **Maximum Leverage**: Hard cap on leverage utilization
- **Position Limits**: Maximum position size constraints
- **Daily Limits**: Maximum trades per day
- **Drawdown Protection**: Automatic shutdown on excessive losses

### Error Handling
- **Graceful Degradation**: System continues with reduced functionality
- **Error Logging**: Comprehensive error tracking and reporting
- **Recovery Mechanisms**: Automatic restart and recovery procedures
- **Data Validation**: Input validation and sanitization

## 🔮 Future Enhancements

### Planned Features
- **Trailing Stop-Loss**: Dynamic stop adjustment based on price movement
- **Portfolio Management**: Multi-asset correlation and diversification
- **Advanced Analytics**: Machine learning for parameter optimization
- **Mobile App**: Native mobile application for monitoring
- **API Integration**: REST API for external system integration

### Performance Optimizations
- **Real-Time Processing**: Sub-second signal generation
- **Scalability**: Support for multiple exchanges and assets
- **Backtesting Engine**: Advanced historical performance analysis
- **Risk Modeling**: Monte Carlo simulation and stress testing

## 📞 Support

For technical support, feature requests, or bug reports:
- **Documentation**: Comprehensive inline code documentation
- **Logging**: Detailed system logs for troubleshooting
- **Monitoring**: Real-time system status and performance metrics

---

**Alpha12_24** - Advanced Algorithmic Trading System  
*Built for professional traders and quantitative analysts*
