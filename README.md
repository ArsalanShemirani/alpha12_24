# 🚀 Alpha12_24 - Advanced Cryptocurrency Trading System

Alpha12_24 is a comprehensive cryptocurrency trading system that combines machine learning, technical analysis, market microstructure, and advanced risk management to generate profitable trading signals.

## 🎯 Features

### 🤖 Machine Learning Models
- **XGBoost**: Gradient boosting for high-performance predictions
- **Random Forest**: Ensemble learning for robust signal generation
- **Logistic Regression**: Interpretable baseline model
- **Model Calibration**: Probability calibration for better signal quality

### 📊 Technical Analysis
- **RSI, MACD, Bollinger Bands**: Classic technical indicators
- **Advanced Indicators**: ATR, ADX, CCI, Williams %R
- **Multi-timeframe Analysis**: 5m to 1d intervals
- **Market Microstructure**: Order flow, volatility, regime detection

### 🌍 Market Data
- **Multi-exchange Support**: Binance, Bybit, Deribit
- **Real-time Data**: Live market feeds with fallback mechanisms
- **ETF Flows**: Institutional flow tracking
- **Macro Indicators**: Fear & Greed, DXY, VIX, Gold correlations

### 🛡️ Risk Management
- **Dynamic Position Sizing**: Based on volatility and confidence
- **Leverage Optimization**: Adaptive leverage based on market conditions
- **Stop-loss/Take-profit**: Automated risk management
- **Portfolio Protection**: Maximum drawdown controls

### 📈 Backtesting & Evaluation
- **Walk-forward Backtesting**: Robust out-of-sample testing
- **Signal Scoring**: Quality assessment of trading signals
- **Hyperparameter Optimization**: Automated parameter tuning
- **Performance Analytics**: Comprehensive metrics and visualizations

## 🏗️ Architecture

```
alpha12_24/
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── src/
│   ├── core/
│   │   └── config.py        # Configuration management
│   ├── data/
│   │   ├── binance_free.py  # Binance data provider
│   │   ├── bybit_free.py    # Bybit data provider
│   │   ├── deribit_free.py  # Deribit data provider
│   │   └── etf_flows.py     # ETF flow data
│   ├── features/
│   │   ├── engine.py        # Feature engineering
│   │   └── macro.py         # Macro features
│   ├── models/
│   │   ├── train.py         # Model training
│   │   └── calibrate.py     # Model calibration
│   ├── policy/
│   │   ├── thresholds.py    # Signal thresholds
│   │   └── regime.py        # Market regime detection
│   ├── trading/
│   │   ├── planner.py       # Trade planning
│   │   ├── leverage.py      # Leverage management
│   │   └── logger.py        # Trading logger
│   ├── backtest/
│   │   └── runner.py        # Backtesting engine
│   ├── eval/
│   │   ├── score_signals.py # Signal scoring
│   │   └── optimizer.py     # Hyperparameter optimization
│   └── dashboard/
│       └── app.py           # Streamlit dashboard
├── artifacts/               # Model artifacts and logs
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd alpha12_24

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Optional: Install development dependencies for testing
pip install -r requirements-dev.txt
```

#### Optional Dependencies

For enhanced functionality, you can install optional dependencies:

```bash
# XGBoost (for enhanced ML performance)
pip install xgboost

# TA-Lib (for advanced technical indicators)
# Note: TA-Lib requires system-level dependencies
# macOS: brew install ta-lib
# Ubuntu: sudo apt-get install ta-lib
pip install TA-Lib

# Statsmodels (for trendline plotting)
pip install statsmodels
```

### 2. Configuration

Edit `config.yaml` to customize your settings:

```yaml
project:
  assets: ["BTCUSDT"]
  horizons_hours: [12, 24]
  bar_interval: "5m"

model:
  learner: "xgb"
  calibrate: true
  train_days: 90
  test_days: 14
  embargo_hours: 24

signal:
  prob_long: 0.60
  prob_short: 0.40
  min_rr: 1.8

risk:
  risk_per_trade: 0.01
  stop_min_frac: 0.003
```

### 3. Run Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/app.py
```

### 4. Run Backtest

```python
from src.backtest.runner import BacktestRunner
from src.data.binance_free import BinanceFreeProvider
from src.core.config import config

# Initialize components
provider = BinanceFreeProvider()
runner = BacktestRunner(config)

# Fetch data
data = provider.get_historical_data("BTCUSDT", "1h", 90)

# Run backtest
result = runner.run_backtest(data, "BTCUSDT")
print(f"Total Return: {result.performance_metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.performance_metrics['sharpe_ratio']:.2f}")
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **Real-time Data Visualization**: Interactive price charts with technical indicators
- **Signal Analysis**: Signal distribution and confidence analysis
- **Model Performance**: Cross-validation metrics and feature importance
- **Feature Analysis**: Correlation heatmaps and feature statistics
- **Configuration Management**: Easy parameter adjustment

## 🔧 Advanced Usage

### Hyperparameter Optimization

```python
from src.eval.optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(config)

# Optimize model parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15]
}

result = optimizer.optimize_model_hyperparameters(
    data, param_grid, metric='sharpe_ratio', n_trials=20
)
```

### Signal Scoring

```python
from src.eval.score_signals import SignalScorer

scorer = SignalScorer(config)

# Score signals
for signal in signals:
    score = scorer.score_signal(
        signal['signal'], signal['confidence'],
        signal['prob_up'], signal['prob_down'],
        current_price, future_price
    )
```

### Custom Feature Engineering

```python
from src.features.engine import FeatureEngine

engine = FeatureEngine()

# Add custom features
def add_custom_features(df):
    df['custom_indicator'] = df['close'].rolling(20).mean() / df['close']
    return df

# Build feature matrix
feature_df, feature_cols = engine.build_feature_matrix(data, [12, 24])
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total return, average return, return volatility
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Trading Metrics**: Win rate, profit factor, average win/loss
- **Model Metrics**: Cross-validation accuracy, precision, recall, F1-score
- **Signal Quality**: Signal consistency, confidence stability, diversity

## 🛡️ Risk Management

### Position Sizing
- Dynamic sizing based on volatility and confidence
- Maximum position size limits
- Portfolio-level risk controls

### Leverage Management
- Adaptive leverage based on market conditions
- Volatility-based scaling
- Margin requirement monitoring

### Stop-loss/Take-profit
- Automated risk management
- Dynamic stop-loss based on ATR
- Risk/reward ratio optimization

## 🔄 Walk-forward Backtesting

The system uses walk-forward backtesting to ensure robust performance:

1. **Training Window**: 90 days of historical data
2. **Testing Window**: 14 days of out-of-sample testing
3. **Embargo Period**: 24 hours between train/test to prevent data leakage
4. **Rolling Windows**: Continuous retraining and testing

## 📊 Data Sources

### Free APIs
- **Binance**: OHLCV data, funding rates, open interest
- **Bybit**: Market data and order book information
- **Deribit**: Options data and volatility metrics
- **Alternative.me**: Fear & Greed Index
- **CoinGecko**: Market cap and volume data

### Synthetic Data
- ETF flows (synthetic for now)
- Macro indicators (synthetic for now)
- Social sentiment (synthetic for now)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended to provide financial advice. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

## 🆘 Support

For questions, issues, or contributions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Join our community discussions

---

**🚀 Happy Trading! 🚀**
