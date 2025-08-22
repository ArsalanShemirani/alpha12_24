# 🔄 **SETUP WORKFLOW GUIDE**

## 📋 **OVERVIEW**

This guide explains the complete setup lifecycle from creation to completion, including the new "executed" status and manual cancel/exit functionality.

---

## 🎯 **SETUP STATUS WORKFLOW**

### **1. 🟠 EXECUTED** 
- **Definition**: User has decided to execute the setup (considering it for execution)
- **Action**: Setup is ready and waiting for entry price to be hit
- **Dashboard Location**: "EXECUTED SETUPS" section
- **User Actions Available**: 
  - ❌ **Cancel Setup**: Cancel before entry price is hit
  - 📊 View setup details and current market price

### **2. 🟡 PENDING**
- **Definition**: Setup is waiting for entry price to be reached
- **Action**: Will trigger when price touches the entry level
- **Dashboard Location**: "PENDING SETUPS" section
- **User Actions Available**:
  - ❌ **Cancel Setup**: Cancel before entry price is hit

### **3. 🔵 TRIGGERED**
- **Definition**: Entry price has been hit, trade is now active
- **Action**: Trade is running and waiting for target/stop
- **Dashboard Location**: "TRIGGERED SETUPS" section (Active Trades)
- **User Actions Available**:
  - 🟢 **Exit at Current Price**: Exit at current market price
  - 🔴 **Exit at Stop Loss**: Exit at planned stop loss level
  - 🟡 **Exit at Entry**: Break-even exit

### **4. ✅ COMPLETED**
- **Definition**: Trade has finished with one of these outcomes:
  - 🟢 **Target**: Take profit level reached
  - 🔴 **Stop**: Stop loss level reached
  - 🟣 **Timeout**: Setup expired without triggering
  - 🎯 **Manual Exit**: User manually exited the trade
  - ⚫ **Cancelled**: User cancelled before entry
- **Dashboard Location**: "COMPLETED SETUPS" section
- **Action**: Trade history recorded with PnL calculation

---

## 🎮 **USER CONTROLS BY STATUS**

### **🟠 EXECUTED SETUPS**
```
┌─────────────────────────────────────┐
│ 🟠 EXECUTED SETUPS - Ready for Entry │
├─────────────────────────────────────┤
│ • View setup details                │
│ • See current market price          │
│ • ❌ Cancel Setup                   │
└─────────────────────────────────────┘
```

### **🟡 PENDING SETUPS**
```
┌─────────────────────────────────────┐
│ 🟡 PENDING SETUPS - Waiting for Entry│
├─────────────────────────────────────┤
│ • View setup details                │
│ • ❌ Cancel Setup                   │
└─────────────────────────────────────┘
```

### **🔵 TRIGGERED SETUPS (Active Trades)**
```
┌─────────────────────────────────────┐
│ 🔵 TRIGGERED SETUPS - Active Trades │
├─────────────────────────────────────┤
│ • View trade details                │
│ • See real-time PnL                 │
│ • 🟢 Exit at Current Price          │
│ • 🔴 Exit at Stop Loss              │
│ • 🟡 Exit at Entry (Break-even)     │
└─────────────────────────────────────┘
```

---

## 📊 **DASHBOARD DISPLAY**

### **Status Summary**
```
┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  Total  │ 🟠 Exec  │ 🟡 Pend  │ 🔵 Trig  │ 🟢 Target│ 🔴 Stop  │ 🟣 Time  │ 🎯 Manual│ ⚫ Cancel│
├─────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│   15    │    2     │    3     │    1     │    4     │    2     │    1     │    1     │    1     │
└─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### **Setup Lifecycle Sections**
1. **🟠 EXECUTED SETUPS** - Ready for Entry
2. **🟡 PENDING SETUPS** - Waiting for Entry  
3. **🔵 TRIGGERED SETUPS** - Active Trades
4. **✅ COMPLETED SETUPS** - Finished Trades

---

## 🔧 **MANUAL CANCEL/EXIT LOGIC**

### **For EXECUTED/PENDING Setups**
- **Action**: Cancel the setup
- **Result**: Status changes to "cancelled"
- **PnL**: No PnL calculation (setup never triggered)
- **Notification**: Telegram alert sent

### **For TRIGGERED Setups**
- **Action**: Manual exit with current price
- **Result**: Status changes to "manual_exit"
- **PnL**: Calculated based on exit price vs entry
- **Trade History**: Recorded in trade_history.csv
- **Notification**: Telegram alert sent

---

## 📈 **PNL CALCULATION**

### **Manual Exit PnL Formula**
```
For LONG trades:
PnL % = (Exit Price - Entry Price) / Entry Price × 100

For SHORT trades:
PnL % = (Entry Price - Exit Price) / Entry Price × 100
```

### **Example**
```
Setup: ETHUSDT SHORT
Entry: $4,319.56
Exit: $4,250.00 (manual exit)

PnL % = (4319.56 - 4250.00) / 4319.56 × 100 = +1.61%
```

---

## 🔔 **NOTIFICATIONS**

### **Setup Cancelled**
```
❌ Setup CANCELLED
ETHUSDT 4h SHORT
Setup ID: AUTO-ETHUSDT-4h-20250822_041007
```

### **Manual Exit**
```
🎯 Manual Exit
ETHUSDT 4h SHORT
Exit Price: $4,250.00
PnL: +1.61%
Setup ID: AUTO-ETHUSDT-4h-20250822_041007
```

---

## ⚡ **QUICK REFERENCE**

| Status | Meaning | User Actions | Location |
|--------|---------|--------------|----------|
| 🟠 Executed | Ready for entry | Cancel | Executed section |
| 🟡 Pending | Waiting for entry | Cancel | Pending section |
| 🔵 Triggered | Active trade | Exit options | Active trades |
| ✅ Completed | Finished trade | View history | Completed section |

---

## 🚀 **BEST PRACTICES**

1. **Review executed setups** before they trigger
2. **Monitor active trades** for manual exit opportunities
3. **Use break-even exits** to minimize losses
4. **Check current prices** before manual actions
5. **Review completed trades** for performance analysis
