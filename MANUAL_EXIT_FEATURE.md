# 🎯 **MANUAL EXIT FEATURE**

## 📋 **OVERVIEW**

The Manual Exit feature allows users to cancel/exit triggered setups through the Dashboard. When a user manually exits a setup, the system marks it as completed and updates PnL and other values same as setups that are concluded by stop, take profit, or timeout.

---

## ✨ **KEY FEATURES**

### **1. Real-Time Price Display**
- Shows current market price for the selected setup
- Displays entry price and calculated PnL
- Real-time PnL calculation based on current market conditions

### **2. Multiple Exit Options**
- **🟢 Exit at Current Price**: Exit at the current market price
- **🔴 Exit at Stop Loss**: Exit at the planned stop loss level
- **🟡 Exit at Entry**: Exit at the original entry price (break-even)

### **3. Complete Trade Recording**
- Updates `setups.csv` with new status
- Records trade in `trade_history.csv` with proper PnL calculation
- Includes all trade metadata (fees, timestamps, etc.)

### **4. Notifications**
- Telegram notifications for manual exits
- Success messages in the dashboard
- Proper logging of all actions

---

## 🎛️ **HOW TO USE**

### **Step 1: Access the Feature**
1. Open the Dashboard
2. Navigate to the "📊 Setups Monitor & Lifecycle Tracking" section
3. Look for the "🔵 TRIGGERED SETUPS" section

### **Step 2: Select a Setup**
1. In the "🎯 Manual Exit Controls" section
2. Use the dropdown to select the setup you want to exit
3. The system will display:
   - Entry Price
   - Current Market Price
   - Calculated PnL

### **Step 3: Choose Exit Method**
1. **🟢 Exit at Current Price**: Best for taking current market conditions
2. **🔴 Exit at Stop Loss**: Exit at planned stop level
3. **🟡 Exit at Entry**: Break-even exit

### **Step 4: Confirm Exit**
1. Click the desired exit button
2. System will process the exit
3. Success message will appear
4. Setup will move to "Completed" section

---

## 📊 **SYSTEM INTEGRATION**

### **Status Updates**
- **Before**: `triggered` (active trade)
- **After**: `manual_exit` (completed trade)

### **Trade History Recording**
The system records the manual exit with the same structure as automatic exits:

```csv
setup_id,asset,interval,direction,created_at,trigger_ts,entry,stop,target,exit_ts,exit_price,outcome,pnl_pct,pnl_pct_net,rr_planned,confidence,size_units,notional_usd,leverage,fees_bps_per_side,price_at_trigger,trigger_rule,entry_buffer_bps
```

### **PnL Calculation**
- **Long positions**: `(exit_price - entry_price) / entry_price * 100`
- **Short positions**: `(entry_price - exit_price) / entry_price * 100`
- **Net PnL**: Includes trading fees (default 4 bps per side)

### **Performance Metrics**
Manual exits are included in all performance calculations:
- Win rate calculations
- Profit factor analysis
- Equity curve updates
- Model training data

---

## 🔧 **TECHNICAL IMPLEMENTATION**

### **Core Functions**

#### **`_manual_exit_setup()`**
```python
def _manual_exit_setup(setup_id: str, outcome: str, exit_price: float, setup_row: pd.Series):
    """
    Manually exit a triggered setup and record the trade.
    """
    # 1. Calculate PnL
    # 2. Update setups.csv
    # 3. Record in trade_history.csv
    # 4. Send notifications
    # 5. Show success message
```

#### **`_append_trade_row()`**
```python
def _append_trade_row(trade_row: dict):
    """
    Append a trade row to trade_history.csv
    """
    # 1. Calculate net PnL with fees
    # 2. Format timestamps
    # 3. Ensure proper column structure
    # 4. Append to CSV
```

#### **`get_latest_price()`**
```python
def get_latest_price(symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Get the latest price for a symbol from Binance spot API
    """
    # Fetch 1-minute candle and return close price
```

### **Data Flow**
```
User Action → UI Selection → Price Fetch → PnL Calculation → 
Status Update → Trade Recording → Notification → Success Message
```

---

## 📈 **PERFORMANCE IMPACT**

### **Training Data**
- Manual exits are included in model training data
- Contributes to win rate and profit factor calculations
- Helps improve model accuracy over time

### **Risk Management**
- Allows users to exit trades based on market conditions
- Provides flexibility beyond automated stop/target levels
- Helps manage risk in volatile market conditions

### **System Reliability**
- Graceful error handling for price fetching failures
- Fallback to entry price if market data unavailable
- Proper transaction logging and audit trail

---

## 🚨 **SAFETY FEATURES**

### **Validation**
- Setup must be in "triggered" status
- Valid setup ID verification
- Price validation and bounds checking

### **Error Handling**
- Network failures during price fetching
- File system errors during updates
- Invalid data format handling

### **Audit Trail**
- All manual exits are logged
- Timestamps recorded for all actions
- Trade history maintains complete record

---

## 📱 **NOTIFICATIONS**

### **Telegram Alerts**
Format: `Setup MANUAL EXIT {asset} {interval}\nEntry: {entry} → Exit: {exit}\nPnL: {pnl}%\nTime: {timestamp} MY`

### **Dashboard Messages**
- Success: `🎯 Setup {id} exited successfully! PnL: {pnl}%`
- Error: `Failed to exit setup: {error}`

---

## 🔄 **WORKFLOW INTEGRATION**

### **Before Manual Exit**
1. Setup is in "triggered" status
2. Trade is active and being monitored
3. User can see setup in triggered section

### **During Manual Exit**
1. User selects setup from dropdown
2. System fetches current market price
3. User chooses exit method
4. System calculates PnL

### **After Manual Exit**
1. Setup status changes to "manual_exit"
2. Trade recorded in history
3. Setup moves to completed section
4. Performance metrics updated
5. Notifications sent

---

## 🎯 **USE CASES**

### **1. Market Condition Changes**
- Exit early if market conditions deteriorate
- Take profits before target if conditions improve
- Adjust position based on news or events

### **2. Risk Management**
- Exit at break-even to preserve capital
- Exit at stop loss to limit losses
- Exit at current price to lock in gains

### **3. Portfolio Management**
- Rebalance portfolio by exiting positions
- Free up capital for new opportunities
- Adjust position sizes based on market conditions

---

## 📊 **MONITORING & ANALYTICS**

### **Dashboard Metrics**
- Manual exit count in status summary
- PnL from manual exits included in totals
- Performance comparison with automated exits

### **Trade History**
- All manual exits recorded with "manual_exit" outcome
- Full audit trail of user actions
- Integration with existing analytics

### **Performance Analysis**
- Manual vs automated exit performance
- User decision quality analysis
- Risk-adjusted return calculations

---

## 🔮 **FUTURE ENHANCEMENTS**

### **Planned Features**
- **Bulk Exit**: Exit multiple setups at once
- **Conditional Exits**: Set exit conditions based on indicators
- **Exit History**: Detailed history of manual exits
- **Performance Analytics**: Advanced analysis of manual exit performance

### **Integration Opportunities**
- **Risk Management**: Integration with position sizing
- **Portfolio Management**: Multi-asset exit strategies
- **Machine Learning**: Learn from user exit patterns

---

## ✅ **CONCLUSION**

The Manual Exit feature provides users with complete control over their triggered setups while maintaining full system integration. All manual exits are properly recorded, calculated, and included in performance metrics, ensuring a seamless trading experience with comprehensive risk management capabilities.
