# R:R Invariant Logging System

## Overview

The R:R Invariant Logging System tracks planned vs realized risk:reward ratios across auto and manual setups without changing any trading behavior. This system provides comprehensive logging to reconcile R:R consistency and detect distortions.

## Features

### 1. Decision-Time Logging
When a setup is created (auto or manual), the system logs:
- **R_used**: ATR14 or structure-based R unit used for risk math
- **s_planned**: Stop distance in R units (e.g., 1.0)
- **t_planned**: Target distance in R units (e.g., 1.8)
- **rr_planned**: Planned R:R ratio (t_planned / s_planned)
- **entry_planned**: Reference price used at decision time
- **RR Floor Guardrail**: Clamps rr_planned to minimum 1.5 if violated

### 2. Trigger/Fill-Time Logging
When an order triggers/fills, the system logs:
- **entry_fill**: Actual fill price (mandatory)
- **entry_shift_R**: Entry shift in R units (signed)
- **stop_final_from_fill**: Recalculated stop from fill using planned R-space
- **tp_final_from_fill**: Recalculated target from fill using planned R-space
- **rr_realized_from_fill**: R:R from planned R-space (should equal rr_planned)
- **rr_realized_from_prices**: R:R from actual live prices
- **rr_distortion**: Percentage deviation from planned R:R

### 3. Telemetry Flags
- **warn_rr_floor_violation**: Set when rr_planned < 1.5
- **warn_missing_fill**: Set when entry_fill is missing at trigger time

## Implementation

### Core Module: `src/utils/rr_invariants.py`

#### `compute_rr_invariants()`
Centralized function that computes all invariant fields:
```python
def compute_rr_invariants(
    direction: str,
    entry_planned: float,
    entry_fill: Optional[float],
    R_used: float,
    s_planned: float,
    t_planned: float,
    live_entry: float,
    live_stop: float,
    live_tp: float,
    setup_id: str,
    tf: str
) -> Dict[str, Union[float, str, bool, None]]
```

#### `log_rr_invariants()`
Persists invariants to sidecar CSV file:
```python
def log_rr_invariants(invariants: Dict, sidecar_file: str = "runs/rr_invariants.csv") -> bool
```

### Integration Points

#### 1. Autosignal (`src/daemon/autosignal.py`)
- **Decision Time**: Logs invariants after setup creation, before saving to CSV
- **Location**: After `_append_setup_row()` call
- **R_used**: Uses `_estimate_atr(df)` from OHLCV data

#### 2. Tracker (`src/daemon/tracker.py`)
- **Trigger Time**: Updates invariants with fill data when setup triggers
- **Location**: After trigger detection, before Telegram alert
- **entry_fill**: Uses `bar['close']` as fill price

#### 3. Dashboard (`src/dashboard/app.py`)
- **Decision Time**: Logs invariants for manual setups
- **Location**: After setup creation, before Telegram alert
- **R_used**: Uses `_estimate_atr(data)` from OHLCV data

## Data Schema

### Sidecar File: `runs/rr_invariants.csv`

| Field | Type | Description |
|-------|------|-------------|
| setup_id | str | Unique setup identifier |
| tf | str | Timeframe (e.g., "1h", "4h") |
| direction | str | "long" or "short" |
| R_used | float | R unit used for risk math |
| s_planned | float | Stop distance in R units |
| t_planned | float | Target distance in R units |
| rr_planned | float | Planned R:R ratio |
| rr_planned_logged | float | Logged R:R (clamped to 1.5 if needed) |
| warn_rr_floor_violation | bool | RR floor violation flag |
| entry_planned | float | Reference price at decision time |
| entry_fill | float | Actual fill price (NaN if not filled) |
| entry_shift_R | float | Entry shift in R units |
| stop_final_from_fill | float | Recalculated stop from fill |
| tp_final_from_fill | float | Recalculated target from fill |
| rr_realized_from_fill | float | R:R from planned R-space |
| rr_realized_from_prices | float | R:R from actual prices |
| rr_distortion | float | Percentage deviation from planned |
| warn_missing_fill | bool | Missing fill price flag |
| invariant_ts | str | Timestamp of invariant computation |

## Configuration

### Environment Variables
- **RR_INVARIANT_LOGGING**: Enable/disable logging (default: "1" = enabled)
- **Sidecar File**: Configurable via `sidecar_file` parameter (default: "runs/rr_invariants.csv")

### Feature Gate
The system is feature-gated and gracefully degrades if the module is unavailable:
```python
try:
    from src.utils.rr_invariants import compute_rr_invariants, log_rr_invariants
    RR_INVARIANT_AVAILABLE = True
except ImportError:
    RR_INVARIANT_AVAILABLE = False
    print("[component] R:R invariant logging not available")
```

## Mathematical Formulas

### R:R Calculations
- **Planned R:R**: `rr_planned = t_planned / s_planned`
- **Realized from Fill**: `rr_realized_from_fill = t_planned / s_planned` (should equal rr_planned)
- **Realized from Prices**: `rr_realized_from_prices = |target - entry| / |entry - stop|`
- **Distortion**: `rr_distortion = |rr_realized_from_prices - rr_planned| / max(rr_planned, 1e-9)`

### Entry Shift
- **Long**: `entry_shift_R = (entry_fill - entry_planned) / R_used`
- **Short**: `entry_shift_R = -(entry_fill - entry_planned) / R_used` (sign flipped)

### Final Levels from Fill
- **Long**:
  - `stop_final_from_fill = entry_fill - s_planned * R_used`
  - `tp_final_from_fill = entry_fill + t_planned * R_used`
- **Short**:
  - `stop_final_from_fill = entry_fill + s_planned * R_used`
  - `tp_final_from_fill = entry_fill - t_planned * R_used`

## Usage Examples

### Verification Harness Integration
The verification harness can read the sidecar file to analyze R:R consistency:
```python
import pandas as pd

# Load invariants
invariants_df = pd.read_csv("runs/rr_invariants.csv")

# Analyze distortions
high_distortion = invariants_df[invariants_df['rr_distortion'] > 0.05]
print(f"High distortion setups: {len(high_distortion)}")

# Check RR floor violations
violations = invariants_df[invariants_df['warn_rr_floor_violation'] == True]
print(f"RR floor violations: {len(violations)}")
```

### Real-Time Monitoring
Monitor R:R consistency in real-time:
```python
from src.utils.rr_invariants import get_rr_invariants

# Get invariants for specific setup
invariants = get_rr_invariants("AUTO-BTCUSDT-4h-LONG-20250823-1200")
if invariants:
    distortion = invariants.get('rr_distortion', 0)
    if distortion > 0.05:  # 5% threshold
        print(f"High R:R distortion detected: {distortion:.2%}")
```

## Benefits

1. **Non-Intrusive**: No changes to trading behavior or decision logic
2. **Comprehensive**: Tracks both planned and realized R:R metrics
3. **Reconcilable**: Enables verification of R:R consistency across setups
4. **Robust**: Graceful degradation and error handling
5. **Centralized**: Single source of truth for R:R invariant computation
6. **Extensible**: Easy to add new metrics or modify existing ones

## Testing

The system includes comprehensive test cases:
- ✅ Long/short setups with various R:R ratios
- ✅ RR floor violation detection
- ✅ Missing fill price handling
- ✅ Entry shift calculations
- ✅ CSV persistence and retrieval

## Future Enhancements

1. **Real-time Alerts**: Telegram notifications for high distortions
2. **Dashboard Integration**: R:R consistency metrics in web interface
3. **Historical Analysis**: Trend analysis of R:R distortions over time
4. **Automated Corrections**: Suggest adjustments for consistent R:R violations
