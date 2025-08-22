# Alpha12 Background Analysis Setup

## Overview

This setup provides a dual-interval approach for optimal trading system performance:

- **Autosignal**: Generates trading setups on 4h+ intervals (avoiding noise)
- **Background Analysis**: Runs continuous analysis on 15m data for model improvement

## Architecture

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│   Autosignal    │    │ Background Analysis │    │     Tracker     │
│   (1h+ only)    │    │   (15m training)    │    │   (24/7 loop)   │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    runs/setups.csv                         │
   │                    runs/background_analysis.csv             │
   │                    runs/trade_history.csv                  │
   └─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# --- Background Analysis Configuration
ALPHA12_TRAINING_INTERVAL=15m
ALPHA12_ANALYSIS_SLEEP=300
ALPHA12_ANALYSIS_DAYS=120
```

### Current Settings

- **Autosignal Interval**: 4h (generates setups)
- **Training Interval**: 15m (model analysis)
- **Analysis Sleep**: 300 seconds (5 minutes)
- **History Days**: 120 days
- **Assets**: BTCUSDT

## Files Created

### 1. Background Analysis Daemon
- **File**: `src/daemon/background_analysis.py`
- **Purpose**: Continuous analysis on 15m data
- **Features**:
  - Model training and evaluation
  - Feature importance tracking
  - Cross-validation metrics
  - Live logs retraining

### 2. Systemd Service
- **File**: `ops/alpha12-background-analysis.service`
- **Purpose**: Production deployment
- **Features**:
  - Auto-restart on failure
  - Memory limits (400MB)
  - Logging to `/var/log/alpha12/`
  - Security hardening

### 3. Startup Script
- **File**: `start_background_analysis.sh`
- **Purpose**: Easy setup and testing
- **Features**:
  - Environment configuration
  - Service testing
  - Deployment instructions

## Usage

### Quick Start

1. **Setup and Test**:
   ```bash
   ./start_background_analysis.sh
   ```

2. **Run Background Analysis**:
   ```bash
   PYTHONPATH=$(pwd) python src/daemon/background_analysis.py
   ```

3. **Run Autosignal** (separate terminal):
   ```bash
   PYTHONPATH=$(pwd) python src/daemon/autosignal.py
   ```

4. **Run Tracker** (separate terminal):
   ```bash
   PYTHONPATH=$(pwd) python src/daemon/tracker.py
   ```

### Production Deployment

1. **Copy service file**:
   ```bash
   sudo cp ops/alpha12-background-analysis.service /etc/systemd/system/
   ```

2. **Enable and start**:
   ```bash
   sudo systemctl enable alpha12-background-analysis
   sudo systemctl start alpha12-background-analysis
   ```

3. **Check status**:
   ```bash
   sudo systemctl status alpha12-background-analysis
   ```

## Analysis Results

### Output Files

1. **`runs/background_analysis.csv`**
   - Continuous analysis results
   - Model performance metrics
   - Feature importance tracking
   - Timestamp tracking

2. **`runs/background_analysis_heartbeat.txt`**
   - Daemon health monitoring
   - Last run timestamp

### Sample Analysis Result

```csv
asset,interval,timestamp,samples,features,accuracy,win_rate,cv_accuracy,cv_precision,cv_recall,cv_f1,top_features,model_type,calibrated
BTCUSDT,15m,2025-08-18T03:39:53.291825+00:00,1000,44,0.862,0.49,0.5048192771084337,0.483994653994654,0.4510574460517739,0.41326083050232204,[],rf,True
```

### Metrics Explained

- **accuracy**: Model accuracy on training data
- **win_rate**: Percentage of winning trades in dataset
- **cv_accuracy**: Cross-validation accuracy
- **cv_precision**: Cross-validation precision
- **cv_recall**: Cross-validation recall
- **cv_f1**: Cross-validation F1 score
- **samples**: Number of training samples
- **features**: Number of features used
- **calibrated**: Whether model is probability-calibrated

## Benefits

### 1. Noise Reduction
- Autosignal only generates setups on 4h+ intervals
- Avoids false signals from 5m/15m noise
- Higher quality setups with better win rates

### 2. Model Improvement
- Continuous training on 15m data
- More frequent model updates
- Better feature engineering insights
- Improved prediction accuracy

### 3. Efficiency
- Separate concerns: signal generation vs model training
- Scalable architecture
- Resource optimization
- Better monitoring and debugging

### 4. Data Utilization
- Uses high-frequency data for training
- Maintains low-frequency signals
- Optimal balance of speed and accuracy

## Monitoring

### Health Checks

1. **Heartbeat Monitoring**:
   ```bash
   cat runs/background_analysis_heartbeat.txt
   ```

2. **Analysis Results**:
   ```bash
   tail -f runs/background_analysis.csv
   ```

3. **Systemd Logs**:
   ```bash
   sudo journalctl -u alpha12-background-analysis -f
   ```

### Performance Metrics

- **Analysis Frequency**: Every 5 minutes
- **Data Points**: 1000 samples per analysis
- **Features**: 44 technical indicators
- **Model Type**: Random Forest (calibrated)
- **Memory Usage**: <400MB

## Troubleshooting

### Common Issues

1. **No Analysis Results**:
   - Check internet connection
   - Verify API access
   - Check log files

2. **High Memory Usage**:
   - Reduce `ALPHA12_ANALYSIS_DAYS`
   - Increase `ALPHA12_ANALYSIS_SLEEP`
   - Monitor with `htop`

3. **Service Not Starting**:
   - Check systemd logs
   - Verify file permissions
   - Check Python environment

### Log Locations

- **Application Logs**: `runs/background_analysis_heartbeat.txt`
- **Systemd Logs**: `/var/log/alpha12/background-analysis.*.log`
- **Analysis Results**: `runs/background_analysis.csv`

## Future Enhancements

1. **Multi-Asset Support**: Extend to multiple cryptocurrencies
2. **Advanced Features**: Add sentiment analysis, on-chain data
3. **Model Ensembles**: Combine multiple model types
4. **Real-time Alerts**: Notify on model performance changes
5. **Dashboard Integration**: Web interface for monitoring

## Conclusion

This setup provides the optimal balance between signal quality and model improvement:

- ✅ **High-quality signals** from 1h+ intervals
- ✅ **Continuous learning** from 15m data
- ✅ **Scalable architecture** for production
- ✅ **Comprehensive monitoring** and logging
- ✅ **Easy deployment** and maintenance

The system is now ready for production deployment with robust monitoring and continuous improvement capabilities.
