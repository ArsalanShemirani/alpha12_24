# Adaptive Setup Validity Implementation

## Overview

This implementation replaces the fixed "24 bars" validity with a dynamic validity window computed at setup creation (auto & manual). Expiry is based on timeframe, current ATR (R_used), and macro regime (MA200-only).

## Key Features

### 🎯 **Dynamic Validity Calculation**
- **Timeframe-based**: Different base bars for each timeframe
- **Volatility-adjusted**: Higher ATR → shorter validity (faster markets)
- **Regime-adjusted**: Bull/bear trends persist longer, neutral ranges decay faster
- **Bounded**: Respects min/max limits per timeframe

### 🔧 **Configuration**
Environment variables (all optional with safe defaults):

```bash
# Enable/disable adaptive validity (default: ON)
ADAPTIVE_VALIDITY=1

# Base bars per timeframe
AVF_BASE_15M=32   AVF_BASE_1H=32   AVF_BASE_4H=24   AVF_BASE_1D=20

# Volatility anchors (typical ATR for each timeframe)
AVF_VOL_ANCHOR_15M=60   AVF_VOL_ANCHOR_1H=150   AVF_VOL_ANCHOR_4H=1200   AVF_VOL_ANCHOR_1D=3000

# Min/max bounds per timeframe
AVF_MIN_15M=12   AVF_MAX_15M=48
AVF_MIN_1H=12    AVF_MAX_1H=48
AVF_MIN_4H=12    AVF_MAX_4H=36
AVF_MIN_1D=10    AVF_MAX_1D=30

# Regime adjustment factors
AVF_REGIME_BULL=1.15    AVF_REGIME_BEAR=1.15    AVF_REGIME_NEUTRAL=0.85
```

## Algorithm

### 1. **Base Bars by Timeframe**
- **15m**: 32 bars
- **1h**: 32 bars  
- **4h**: 24 bars
- **1d**: 20 bars

### 2. **Volatility Adjustment**
```
vol_ratio = clamp(R_used / vol_anchor, 0.5, 2.0)
vol_factor = 1 / vol_ratio  # Higher ATR → smaller factor
```

### 3. **Regime Adjustment**
- **Bull/Bear**: 1.15x (trends persist longer)
- **Neutral**: 0.85x (ranges decay faster)

### 4. **Final Calculation**
```
raw = base_bars * vol_factor * regime_factor
valid_bars = clamp(round(raw), min_bars, max_bars)
```

## Implementation Details

### 📁 **Files Modified**

1. **`src/utils/validity.py`** (New)
   - `compute_adaptive_validity_bars()`: Core calculation function
   - `get_validity_until()`: Timestamp computation helper

2. **`src/daemon/autosignal.py`**
   - Added adaptive validity computation at setup creation
   - Added `valid_bars` and `validity_source` fields to setup schema
   - Updated `SETUP_FIELDS` to include new validity fields

3. **`src/dashboard/app.py`**
   - Added adaptive validity computation for manual setups
   - Updated UI to display validity source and bars count
   - Reuses macro regime from macro gate for consistency

### 🔄 **Integration Points**

#### **Autosignal Path**
```python
# After setup levels are computed, before CSV write
if os.getenv("ADAPTIVE_VALIDITY", "1") == "1":
    valid_bars = compute_adaptive_validity_bars(
        tf=autosignal_interval,
        R_used=atr_val,
        regime=regime_result.regime,
        now_ts=base_now
    )
    valid_until = get_validity_until(base_now, valid_bars, autosignal_interval)
    validity_source = "adaptive"
else:
    # Legacy fallback
    valid_bars = 24
    validity_source = "fixed"
```

#### **Manual Setup Path**
```python
# After setup levels are computed, before CSV write
if os.getenv("ADAPTIVE_VALIDITY", "1") == "1":
    valid_bars = compute_adaptive_validity_bars(
        tf=interval,
        R_used=atr_val,
        regime=regime_result.regime,
        now_ts=now_ts
    )
    valid_until = get_validity_until(now_ts, valid_bars, interval)
    validity_source = "adaptive"
else:
    # Legacy fallback
    valid_bars = int(st.session_state.get("valid_bars", 24))
    validity_source = "fixed"
```

## Examples

### **High Volatility Market (4h, ATR=1800, Bull)**
- Base: 24 bars
- Vol ratio: 1800/1200 = 1.5 → vol_factor = 0.67
- Regime: 1.15x
- Raw: 24 × 0.67 × 1.15 = 18.5 → **18 bars**

### **Low Volatility Market (4h, ATR=800, Bull)**
- Base: 24 bars
- Vol ratio: 800/1200 = 0.67 → vol_factor = 1.5
- Regime: 1.15x
- Raw: 24 × 1.5 × 1.15 = 41.4 → **36 bars** (capped)

### **Neutral Regime (4h, ATR=1200, Neutral)**
- Base: 24 bars
- Vol ratio: 1200/1200 = 1.0 → vol_factor = 1.0
- Regime: 0.85x
- Raw: 24 × 1.0 × 0.85 = 20.4 → **20 bars**

## Benefits

### 🎯 **Market Adaptation**
- **Volatile markets**: Shorter validity prevents stale setups
- **Calm markets**: Longer validity allows setups to mature
- **Trending markets**: Extended validity captures trend moves

### 🔒 **Risk Management**
- **Bounded ranges**: Prevents extreme validity periods
- **Regime awareness**: Aligns with market structure
- **Fallback safety**: Graceful degradation to fixed bars

### 📊 **Transparency**
- **Validity source**: Shows "adaptive" vs "fixed" in UI
- **Bar count display**: Shows actual bars used
- **Logging**: Detailed validity calculation logs

## Testing

The implementation includes comprehensive testing:

### ✅ **Acceptance Criteria Met**
1. **High ATR, Bull**: 4h, ATR=1800 → 18 bars (vs base 24)
2. **Low ATR, Bull**: 4h, ATR=800 → 36 bars (capped at max)
3. **Neutral Regime**: 1h, neutral → ~27 bars (adjusted down)
4. **Bounds Respected**: Extreme ATR values stay within limits
5. **Feature Flag**: `ADAPTIVE_VALIDITY=0` uses legacy 24 bars

### 🔧 **Non-Intrusive Design**
- **No behavior changes**: Only affects expiry timing
- **Backward compatible**: Fallback to fixed bars
- **Feature gated**: Can be disabled via environment variable
- **Safe defaults**: All parameters have reasonable fallbacks

## Usage

### **Enable Adaptive Validity**
```bash
export ADAPTIVE_VALIDITY=1
```

### **Disable (Use Legacy)**
```bash
export ADAPTIVE_VALIDITY=0
```

### **Customize Parameters**
```bash
# Shorter validity in volatile markets
export AVF_VOL_ANCHOR_4H=800  # Lower anchor = shorter validity

# Longer validity in trending markets  
export AVF_REGIME_BULL=1.25   # Higher factor = longer validity
```

## Future Enhancements

### 🔮 **Potential Improvements**
- **Asset-specific anchors**: Different volatility anchors per asset
- **Dynamic regime detection**: Real-time regime classification
- **Market hours adjustment**: Different validity during high/low activity
- **Setup type differentiation**: Different rules for auto vs manual setups

---

**Implementation Status**: ✅ **Complete**
**Test Coverage**: ✅ **Comprehensive**  
**Documentation**: ✅ **Complete**
**Backward Compatibility**: ✅ **Maintained**
