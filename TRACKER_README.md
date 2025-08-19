# 🚀 Alpha12_24 24/7 Trade Lifecycle Tracker

## Overview

The Alpha12_24 Tracker is a robust 24/7 daemon that monitors trading setups from creation to completion, providing real-time trade lifecycle management with comprehensive logging and performance tracking.

## 🏗️ Architecture

### Core Components

1. **`src/runtime/clock.py`** - Timezone handling and utility functions
2. **`src/data/price_feed.py`** - Real-time price data fetching from Binance
3. **`src/daemon/tracker.py`** - Main tracker daemon with lifecycle management
4. **Streamlit Integration** - Real-time status monitoring in dashboard

### Data Flow

```
Setup Creation (Streamlit) → Pending Orders → Trigger Detection → Exit Monitoring → Trade Logging
```

## 🚀 Quick Start

### 1. Start the Tracker

```bash
# Basic start (BTCUSDT, 5m interval)
./start_tracker.sh

# Custom configuration
ALPHA12_SYMBOL=ETHUSDT ALPHA12_INTERVAL=15m ./start_tracker.sh

# Manual start
export PYTHONPATH=$(pwd)
nohup python -m src.daemon.tracker > /tmp/alpha_tracker.log 2>&1 &
```

### 2. Monitor Status

```bash
# Check logs
tail -f /tmp/alpha_tracker.log

# Check process
ps aux | grep tracker

# Check heartbeat
cat runs/daemon_heartbeat.txt
```

### 3. Stop the Tracker

```bash
./stop_tracker.sh

# Manual stop
kill $(cat /tmp/alpha_tracker.pid)
```

## 📊 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPHA12_SYMBOL` | `BTCUSDT` | Trading symbol |
| `ALPHA12_INTERVAL` | `5m` | Candle interval |
| `ALPHA12_SLEEP` | `15` | Polling interval (seconds) |
| `BINANCE_FAPI_DISABLE` | `1` | Disable futures API |

### Multiple Trackers

Run multiple trackers for different symbols/intervals:

```bash
# Terminal 1: BTC 5m
ALPHA12_SYMBOL=BTCUSDT ALPHA12_INTERVAL=5m ./start_tracker.sh

# Terminal 2: ETH 15m  
ALPHA12_SYMBOL=ETHUSDT ALPHA12_INTERVAL=15m ./start_tracker.sh

# Terminal 3: Monitor both
tail -f /tmp/alpha_tracker.log
```

## 🔄 Trade Lifecycle

### 1. Setup Creation
- **Source**: Streamlit dashboard generates setups
- **Storage**: `runs/setups.csv`
- **Status**: `pending`

### 2. Trigger Detection
- **Polling**: Every 15 seconds
- **Logic**: Anti stop-hunt with optional close-through confirmation
- **Status**: `pending` → `triggered`

### 3. Exit Monitoring
- **Method**: First-touch policy (conservative)
- **Outcomes**: `target`, `stop`, `timeout`
- **Logging**: `runs/trade_history.csv`

### 4. Performance Tracking
- **Metrics**: P&L %, win rate, RR achieved
- **Features**: Snapshot at signal time for ML training
- **Alerts**: Optional webhook notifications

## 📈 Data Files

### `runs/setups.csv`
```csv
id,asset,interval,direction,entry,stop,target,rr,created_at,expires_at,status,confidence,trigger_rule,entry_buffer_bps
```

### `runs/trade_history.csv`
```csv
setup_id,asset,interval,direction,created_at,trigger_ts,entry,stop,target,exit_ts,exit_price,outcome,pnl_pct,rr_planned,confidence
```

### `runs/features_at_signal.parquet`
- Feature vectors at signal generation time
- Used for ML model retraining
- Compressed with zstd

### `runs/daemon_heartbeat.txt`
- Last tracker heartbeat timestamp
- Used by Streamlit for status display

## 🎯 Anti Stop-Hunt Features

### Entry Logic
- **Pullback Entries**: Entry below current price for longs, above for shorts
- **Buffer Adjustment**: Shift entry deeper to avoid wick fills
- **Close-Through**: Optional confirmation on bar close

### Trigger Rules
```python
# Touch: Immediate trigger on level touch
if direction == "long":
    return bar["low"] <= entry
else:
    return bar["high"] >= entry

# Close-Through: Require close beyond entry
if direction == "long":
    return touched and bar["close"] >= entry * (1 + buffer_bps/10000.0)
```

## 📊 Streamlit Integration

### Daemon Status Panel
- **Location**: Sidebar under setup controls
- **Features**: Heartbeat display, start instructions
- **Status**: Green (running) / Red (stopped)

### Setup Monitoring Tab
- **Real-time Updates**: Auto-refresh every minute
- **Status Tracking**: Pending, Triggered, Expired, Cancelled
- **Auto-cancel Logic**: Trend flip detection, confidence decay

## 🔧 Advanced Features

### Cron Job Alternative
```bash
# Add to crontab for system-level persistence
* * * * * cd /path/to/alpha12_24 && /usr/bin/env PYTHONPATH=$(pwd) python -m src.daemon.tracker >> /tmp/alpha_tracker.log 2>&1
```

### Fee Integration
```python
# Add to _calc_pnl_pct in tracker.py
def _calc_pnl_pct(direction: str, entry: float, exit_px: float, fee_bps: float = 7.5) -> float:
    # Apply taker fees
    entry_with_fee = entry * (1 + fee_bps/10000.0)
    exit_with_fee = exit_px * (1 - fee_bps/10000.0)
    # ... rest of calculation
```

### Multiple Exchange Support
```python
# Extend price_feed.py with additional exchanges
def get_latest_candle_bybit(symbol: str, interval: str) -> Dict:
    # Bybit implementation
    pass

def get_latest_candle_deribit(symbol: str, interval: str) -> Dict:
    # Deribit implementation  
    pass
```

## 🚨 Troubleshooting

### Common Issues

1. **Tracker not starting**
   ```bash
   # Check Python path
   echo $PYTHONPATH
   
   # Check dependencies
   pip install -r requirements.txt
   ```

2. **No heartbeat detected**
   ```bash
   # Check if process is running
   ps aux | grep tracker
   
   # Check logs
   tail -20 /tmp/alpha_tracker.log
   ```

3. **API rate limits**
   ```bash
   # Increase sleep interval
   export ALPHA12_SLEEP=30
   ./start_tracker.sh
   ```

### Log Analysis
```bash
# Monitor real-time
tail -f /tmp/alpha_tracker.log

# Search for errors
grep -i error /tmp/alpha_tracker.log

# Check performance
grep "tracker error" /tmp/alpha_tracker.log
```

## 📊 Performance Metrics

### Key Indicators
- **Uptime**: Tracker availability
- **Response Time**: Setup → Trigger latency
- **Accuracy**: Predicted vs actual outcomes
- **Win Rate**: Percentage of profitable trades

### Monitoring Commands
```bash
# Check recent trades
tail -20 runs/trade_history.csv

# Calculate win rate
awk -F',' 'NR>1 && $12=="target" {wins++} END {print "Win rate:", wins/(NR-1)*100 "%"}' runs/trade_history.csv

# Monitor active setups
grep "pending\|triggered" runs/setups.csv
```

## 🔮 Future Enhancements

1. **WebSocket Integration**: Real-time price feeds
2. **Database Backend**: PostgreSQL for scalability
3. **REST API**: External monitoring interface
4. **Alert System**: Email/SMS notifications
5. **Backtesting Integration**: Historical performance analysis
6. **Risk Management**: Position sizing, correlation limits

## 📝 License

This tracker is part of the Alpha12_24 trading system. Use at your own risk for educational and research purposes only.
