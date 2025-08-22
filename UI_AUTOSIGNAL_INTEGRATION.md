# UI-Autosignal Integration

## Overview

The UI-autosignal integration allows the autosignal daemon to respect user settings from the dashboard while maintaining the 4h timeframe restriction for optimal signal quality.

## Key Features

### ✅ **Hardcoded 1h Timeframe**
- Autosignal always uses 4h interval regardless of UI setting
- Prevents noise from lower timeframes (5m, 15m)
- Ensures high-quality signal generation

### ✅ **UI Settings Override**
- All UI thresholds and gates are respected by autosignal
- Real-time configuration updates
- Fallback to environment variables when UI config unavailable

### ✅ **Comprehensive Parameter Support**
- Confidence thresholds (`min_conf_arm`)
- Gate settings (`gate_ob`, `gate_regime`, `gate_rr25`)
- Risk parameters (`k_entry`, `k_stop`, `valid_bars`)
- Account settings (`acct_balance`, `max_leverage`, `risk_per_trade_pct`)
- Order book settings (`ob_edge_delta`, `rr25_thresh`)

## Architecture

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   UI Config File    │    │   Autosignal    │
│   (UI Settings) │───▶│   (runs/ui_config.  │───▶│   (1h Only)     │
│                 │    │   json)             │    │                 │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    Environment Variables                    │
   │                    (Fallback Configuration)                 │
   └─────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. UI Configuration Manager (`src/core/ui_config.py`)

**Key Functions:**
- `save_ui_config()`: Saves UI settings to JSON file
- `load_ui_config()`: Loads UI settings from file
- `get_autosignal_config()`: Returns autosignal-specific configuration
- `get_ui_setting()`: Gets setting with environment fallback

**Configuration File:** `runs/ui_config.json`
```json
{
  "timestamp": "2025-08-18T03:58:12.494549",
  "source": "ui_dashboard",
  "settings": {
    "min_conf_arm": 0.45,
    "gate_ob": false,
    "gate_regime": false,
    "gate_rr25": false,
    "max_setups_per_day": 10,
    "k_entry": 0.20,
    "k_stop": 0.80,
    "valid_bars": 48,
    "entry_buffer_bps": 10.0,
    "trigger_rule": "touch",
    "acct_balance": 1000.0,
    "max_leverage": 10,
    "risk_per_trade_pct": 2.0,
    "ob_edge_delta": 0.15,
    "rr25_thresh": 0.05
  }
}
```

### 2. Dashboard Integration (`src/dashboard/app.py`)

**Automatic Saving:**
- UI settings are automatically saved when changed
- No manual intervention required
- Real-time configuration updates

**Integration Point:**
```python
# Save UI settings to session state
ui_settings = dict(
    max_setups_per_day=max_setups_per_day,
    gate_regime=gate_regime, gate_rr25=gate_rr25, gate_ob=gate_ob,
    rr25_thresh=rr25_thresh,
    ob_edge_delta=ob_edge_delta, ob_signed_thr=ob_signed_thr,
    min_conf_arm=min_conf_arm
)
st.session_state.update(ui_settings)

# Save UI settings to file for autosignal to use
try:
    from src.core.ui_config import save_ui_config
    save_ui_config(ui_settings)
except Exception as e:
    st.warning(f"Could not save UI config for autosignal: {e}")
```

### 3. Autosignal Integration (`src/daemon/autosignal.py`)

**Configuration Loading:**
```python
def _get_ui_override_config():
    """Get UI configuration overrides for autosignal."""
    try:
        from src.core.ui_config import get_autosignal_config
        return get_autosignal_config()
    except Exception as e:
        print(f"[autosignal] UI config error: {e}")
        return {}
```

**Interval Override:**
```python
# Always use 4h for autosignal regardless of input interval
autosignal_interval = "4h"
if interval != autosignal_interval:
          print(f"[autosignal] Overriding interval {interval} → {autosignal_interval} (autosignal always uses 4h)")
```

**UI Settings Application:**
```python
# Get UI configuration overrides
min_conf_arm = ui_config.get("min_conf_arm", MIN_CONF_ARM)
gate_ob = ui_config.get("gate_ob", GATE_OB)
ob_edge_delta = ui_config.get("ob_edge_delta", OB_EDGE_DELTA)
# ... etc for all settings
```

## Supported Settings

### Confidence & Gates
- `min_conf_arm`: Minimum confidence to arm (0.50-0.90)
- `gate_ob`: Order book gate enabled/disabled
- `gate_regime`: Regime detection gate enabled/disabled
- `gate_rr25`: RR25 gate enabled/disabled
- `ob_edge_delta`: Order book edge delta (0.0-1.0)
- `rr25_thresh`: RR25 threshold

### Risk & Sizing
- `k_entry`: Entry distance multiplier (0.10+)
- `k_stop`: Stop distance multiplier (0.50+)
- `valid_bars`: Setup validity period (24+)
- `entry_buffer_bps`: Entry buffer in basis points
- `trigger_rule`: Trigger rule ("touch" or "close-through")

### Account Settings
- `acct_balance`: Account balance in USD
- `max_leverage`: Maximum leverage allowed
- `risk_per_trade_pct`: Risk per trade percentage
- `max_setups_per_day`: Daily setup limit

## Usage Examples

### 1. Lower Confidence Threshold
**UI Setting:** `min_conf_arm = 0.45`
**Result:** Autosignal generates more setups with lower confidence

### 2. Disable Order Book Gate
**UI Setting:** `gate_ob = false`
**Result:** Autosignal ignores order book imbalances

### 3. Increase Risk Tolerance
**UI Setting:** `risk_per_trade_pct = 2.0`
**Result:** Larger position sizes for each setup

### 4. Tighter Entry Levels
**UI Setting:** `k_entry = 0.20`
**Result:** Entries closer to current price (easier to hit)

## Fallback Strategy

### Priority Order:
1. **UI Configuration** (highest priority)
2. **Environment Variables** (fallback)
3. **Hardcoded Defaults** (lowest priority)

### Environment Variable Examples:
```bash
MIN_CONF_ARM=0.60
GATE_OB=1
GATE_REGIME=1
MAX_SETUPS_PER_DAY=2
K_ENTRY_ATR=0.25
ACCOUNT_BALANCE_USD=400
MAX_LEVERAGE=10
```

## Monitoring

### Configuration Status
```bash
# Check UI config file
cat runs/ui_config.json

# Check autosignal logs
tail -f /var/log/alpha12/autosignal.out.log

# Verify interval override
grep "Overriding interval" /var/log/alpha12/autosignal.out.log
```

### Configuration Validation
```python
from src.core.ui_config import get_autosignal_config

config = get_autosignal_config()
print(f"Interval: {config['interval']}")  # Should always be "1h"
print(f"Min Confidence: {config['min_conf_arm']}")
print(f"OB Gate: {config['gate_ob']}")
```

## Benefits

### 1. **User Control**
- Full control over autosignal behavior through UI
- Real-time parameter adjustments
- No need to restart services

### 2. **Signal Quality**
- Maintains 1h timeframe for optimal quality
- Prevents noise from lower timeframes
- Consistent with manual setup generation

### 3. **Flexibility**
- Easy parameter experimentation
- Quick adaptation to market conditions
- Comprehensive setting coverage

### 4. **Reliability**
- Fallback to environment variables
- Graceful error handling
- No service interruption

## Troubleshooting

### Common Issues

**1. UI Settings Not Applied**
```bash
# Check if config file exists
ls -la runs/ui_config.json

# Check file permissions
chmod 644 runs/ui_config.json

# Restart autosignal service
sudo systemctl restart alpha12-autosignal.timer
```

**2. Configuration Not Loading**
```bash
# Check autosignal logs
sudo journalctl -u alpha12-autosignal.timer -f

# Verify import path
python -c "from src.core.ui_config import get_autosignal_config; print('OK')"
```

**3. Settings Reverted**
```bash
# Check environment variables
env | grep -E "(MIN_CONF|GATE_|MAX_|K_)"

# Verify UI config timestamp
cat runs/ui_config.json | jq '.timestamp'
```

## Future Enhancements

### Planned Features:
1. **Configuration Validation**: Validate UI settings before saving
2. **Configuration History**: Track setting changes over time
3. **Configuration Templates**: Predefined setting profiles
4. **Real-time Monitoring**: Dashboard showing current autosignal config
5. **Configuration Backup**: Automatic backup of UI settings

### Potential Improvements:
1. **Webhook Integration**: Notify external systems of config changes
2. **A/B Testing**: Compare different configuration sets
3. **Configuration Analytics**: Track performance by configuration
4. **Multi-user Support**: User-specific configuration profiles

## Conclusion

The UI-autosignal integration provides a powerful and flexible way to control autosignal behavior while maintaining the quality benefits of the 4h timeframe restriction. Users can now fine-tune all aspects of the autosignal system through the familiar dashboard interface, with real-time updates and comprehensive fallback mechanisms ensuring reliable operation.
