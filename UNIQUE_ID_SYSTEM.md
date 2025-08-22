# 🆔 **UNIQUE ID SYSTEM**

## 📋 **OVERVIEW**

The unique ID system provides user-friendly, consistent identifiers for all setups (both manual and auto) to make them easier to reference, track, and manage.

---

## 🎯 **UNIQUE ID FORMAT**

### **Format Structure**
```
{ORIGIN}-{ASSET}-{TIMEFRAME}-{DIRECTION}-{TIMESTAMP}
```

### **Examples**
- **Auto Setup**: `AUTO-ETHUSDT-4h-SHORT-20250822-1201`
- **Manual Setup**: `MANUAL-BTCUSDT-1h-LONG-20250822-1430`

### **Components**
- **ORIGIN**: `AUTO` (autosignal) or `MANUAL` (dashboard)
- **ASSET**: Trading pair (e.g., `ETHUSDT`, `BTCUSDT`)
- **TIMEFRAME**: Chart interval (e.g., `1h`, `4h`, `1d`)
- **DIRECTION**: Trade direction (`LONG` or `SHORT`)
- **TIMESTAMP**: Creation time (`YYYYMMDD-HHMM`)

---

## 🔧 **IMPLEMENTATION**

### **1. Database Schema**
Added `unique_id` field to all setup tables:
```python
SETUP_FIELDS = [
    "id", "unique_id", "asset", "interval", "direction", "entry", "stop", "target", "rr",
    "size_units", "notional_usd", "leverage",
    "created_at", "expires_at", "triggered_at", "status", "confidence", "trigger_rule", "entry_buffer_bps",
    "origin"
]
```

### **2. Generation Function**
```python
def _generate_unique_id(asset: str, interval: str, direction: str, origin: str = "auto") -> str:
    """
    Generate a unique, user-friendly ID for setups.
    
    Format: {ORIGIN}-{ASSET}-{TIMEFRAME}-{DIRECTION}-{TIMESTAMP}
    Example: AUTO-ETHUSDT-4h-SHORT-20250822-1201
    """
    timestamp = _now_utc().strftime('%Y%m%d-%H%M')
    prefix = "AUTO" if origin == "auto" else "MANUAL"
    return f"{prefix}-{asset}-{interval}-{direction.upper()}-{timestamp}"
```

### **3. Integration Points**
- **Autosignal**: Generates unique IDs for auto setups
- **Dashboard**: Generates unique IDs for manual setups
- **Tracker**: Handles unique IDs in trade processing
- **UI**: Displays unique IDs in all setup tables

---

## 📊 **DASHBOARD DISPLAY**

### **Setup Monitor Sections**
All setup sections now display the `unique_id` instead of the internal `id`:

1. **🟠 EXECUTED SETUPS**
   - Shows: `AUTO-ETHUSDT-4h-SHORT-20250822-1201`
   - Action: Cancel setup

2. **🟡 PENDING SETUPS**
   - Shows: `MANUAL-BTCUSDT-1h-LONG-20250822-1430`
   - Action: Cancel setup

3. **🔵 TRIGGERED SETUPS**
   - Shows: `AUTO-ETHUSDT-1h-SHORT-20250822-1100`
   - Action: Manual exit options

4. **✅ COMPLETED SETUPS**
   - Shows: `MANUAL-ETHUSDT-4h-LONG-20250821-1800`
   - Action: View trade history

### **Setup Selection**
Users can now easily identify and select setups using the unique ID:
```
AUTO-ETHUSDT-4h-SHORT-20250822-1201 - ETHUSDT SHORT (Entry: 4319.56)
```

---

## 🔄 **WORKFLOW INTEGRATION**

### **Setup Creation**
1. **Auto Setup**: Autosignal generates unique ID automatically
2. **Manual Setup**: Dashboard generates unique ID when user creates setup

### **Setup Management**
1. **Cancel Setup**: Uses unique ID to identify and cancel
2. **Manual Exit**: Uses unique ID to identify and exit
3. **Trade Recording**: Uses unique ID in trade history

### **Notifications**
All Telegram notifications now include the unique ID:
```
❌ Setup CANCELLED
ETHUSDT 4h SHORT
Setup ID: AUTO-ETHUSDT-4h-SHORT-20250822-1201
```

---

## 🎯 **BENEFITS**

### **1. User-Friendly References**
- Easy to read and understand
- Consistent format across all setups
- No confusing internal IDs

### **2. Better Tracking**
- Clear origin identification (AUTO vs MANUAL)
- Time-based sorting and filtering
- Asset and timeframe identification

### **3. Improved Communication**
- Easy to reference in discussions
- Clear setup identification in notifications
- Consistent across all systems

### **4. Enhanced Debugging**
- Quick setup identification
- Clear setup lineage tracking
- Easy correlation with logs

---

## 📝 **USAGE EXAMPLES**

### **Dashboard Reference**
```
Setup: AUTO-ETHUSDT-4h-SHORT-20250822-1201
Status: Executed
Entry: $4,319.56
Target: $4,202.93
Stop: $4,384.35
```

### **Telegram Notification**
```
🎯 Manual Exit
ETHUSDT 4h SHORT
Entry: 4319.56 → Exit: 4250.00
PnL: +1.61%
Setup ID: AUTO-ETHUSDT-4h-SHORT-20250822-1201
```

### **Trade History**
```
setup_id: AUTO-ETHUSDT-4h-SHORT-20250822-1201
asset: ETHUSDT
interval: 4h
direction: short
outcome: manual_exit
pnl_pct: 1.61
```

---

## 🔧 **MIGRATION**

### **Existing Setups**
- Existing setups will have empty `unique_id` field
- New setups will automatically get unique IDs
- Dashboard gracefully handles missing unique IDs

### **Backward Compatibility**
- Internal `id` field still exists for system operations
- `unique_id` is the primary display identifier
- All existing functionality preserved

---

## 🚀 **FUTURE ENHANCEMENTS**

### **Potential Features**
1. **Setup Search**: Search by unique ID
2. **Setup Filtering**: Filter by origin, asset, timeframe
3. **Setup Grouping**: Group related setups
4. **Setup Analytics**: Performance tracking by unique ID

### **API Integration**
- Unique IDs for API endpoints
- Setup reference in webhooks
- External system integration
