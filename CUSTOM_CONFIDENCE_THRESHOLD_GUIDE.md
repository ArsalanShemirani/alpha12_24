# 🎯 **CUSTOM CONFIDENCE THRESHOLD GUIDE**

## 📋 **OVERVIEW**

The adaptive confidence gate system allows you to set custom confidence thresholds for each timeframe. This guide explains all the ways you can customize these thresholds.

---

## 🎛️ **METHOD 1: UI DASHBOARD (EASIEST)**

### **Step-by-Step Instructions**

1. **Open the Dashboard**
   - Navigate to the trading dashboard
   - Look for the sidebar configuration section

2. **Find the Adaptive Confidence Section**
   - Look for "🎯 Adaptive Confidence Thresholds"
   - You'll see current thresholds for all timeframes

3. **Open Custom Configuration**
   - Click on "⚙️ Custom Threshold Configuration"
   - Expand "Set Custom Confidence Thresholds"

4. **Set Your Values**
   - **15m Custom Threshold**: Set for 15-minute timeframes
   - **1h Custom Threshold**: Set for 1-hour timeframes  
   - **4h Custom Threshold**: Set for 4-hour timeframes
   - **1d Custom Threshold**: Set for daily timeframes

5. **Apply Changes**
   - Click "Apply Custom Thresholds" button
   - Refresh the page to see updated values

6. **Reset if Needed**
   - Click "Reset to Defaults" to return to base thresholds

### **UI Features**
- **Range**: 0.50 to 0.95 (safe limits)
- **Precision**: 0.01 increments
- **Session-based**: Changes apply to current session
- **Visual Feedback**: Success messages confirm changes

---

## 🔧 **METHOD 2: ENVIRONMENT VARIABLES (PERMANENT)**

### **Set Environment Variables**

```bash
# Set custom thresholds (permanent)
export MIN_CONF_15M=0.75  # 15m threshold
export MIN_CONF_1H=0.66   # 1h threshold
export MIN_CONF_4H=0.60   # 4h threshold
export MIN_CONF_1D=0.58   # 1d threshold

# Restart the application to apply changes
```

### **Add to Your Shell Profile**

For permanent settings, add to your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export MIN_CONF_15M=0.75' >> ~/.bashrc
echo 'export MIN_CONF_1H=0.66' >> ~/.bashrc
echo 'export MIN_CONF_4H=0.60' >> ~/.bashrc
echo 'export MIN_CONF_1D=0.58' >> ~/.bashrc

# Reload profile
source ~/.bashrc
```

---

## 📊 **UNDERSTANDING THE THRESHOLDS**

### **Default Base Thresholds**
- **15m**: 0.72 (highest - noisy timeframe)
- **1h**: 0.69 (baseline - moderate stability)
- **4h**: 0.64 (lower - stable timeframe)
- **1d**: 0.62 (lowest - most stable)

### **When to Adjust**

#### **Higher Thresholds (More Restrictive)**
- **Use when**: You want fewer, higher-quality setups
- **Example**: Set 1h to 0.75 for stricter filtering
- **Effect**: Fewer setups generated, higher confidence required

#### **Lower Thresholds (More Permissive)**
- **Use when**: You want more trading opportunities
- **Example**: Set 1h to 0.65 for more setups
- **Effect**: More setups generated, lower confidence accepted

### **Timeframe-Specific Considerations**

#### **15m Timeframe**
- **Characteristics**: Noisy, frequent signals
- **Typical Range**: 0.70-0.80
- **Recommendation**: Keep higher threshold (0.72+)

#### **1h Timeframe**
- **Characteristics**: Balanced, moderate stability
- **Typical Range**: 0.65-0.75
- **Recommendation**: Good baseline (0.69)

#### **4h Timeframe**
- **Characteristics**: Stable, fewer signals
- **Typical Range**: 0.60-0.70
- **Recommendation**: Can be lower (0.64)

#### **1d Timeframe**
- **Characteristics**: Most stable, least frequent
- **Typical Range**: 0.55-0.65
- **Recommendation**: Can be lowest (0.62)

---

## 🎯 **PRACTICAL EXAMPLES**

### **Example 1: Conservative Trader**
```bash
# Higher thresholds for quality over quantity
export MIN_CONF_15M=0.78  # Very strict 15m
export MIN_CONF_1H=0.72   # Stricter 1h
export MIN_CONF_4H=0.68   # Moderate 4h
export MIN_CONF_1D=0.65   # Moderate 1d
```

### **Example 2: Active Trader**
```bash
# Lower thresholds for more opportunities
export MIN_CONF_15M=0.70  # More 15m setups
export MIN_CONF_1H=0.65   # More 1h setups
export MIN_CONF_4H=0.60   # More 4h setups
export MIN_CONF_1D=0.58   # More 1d setups
```

### **Example 3: Timeframe-Specific Focus**
```bash
# Focus on 1h and 4h timeframes
export MIN_CONF_15M=0.75  # Restrict 15m (noisy)
export MIN_CONF_1H=0.65   # Allow more 1h setups
export MIN_CONF_4H=0.60   # Allow more 4h setups
export MIN_CONF_1D=0.65   # Moderate 1d
```

---

## 🔍 **MONITORING YOUR CHANGES**

### **Check Current Thresholds**
1. **Dashboard**: Look at the "🎯 Adaptive Confidence Thresholds" section
2. **Override Indicators**: Shows "Override: 0.75" when custom values are active
3. **Base Indicators**: Shows "Base" when using default values

### **Verify Changes**
```python
# Check programmatically
from src.trading.adaptive_confidence_gate import adaptive_confidence_gate

thresholds = adaptive_confidence_gate.get_all_thresholds()
for tf, info in thresholds.items():
    print(f"{tf}: {info['effective']:.3f} (override: {info['user_override']})")
```

### **Monitor Setup Generation**
- **Dashboard**: Check setup generation logs
- **Confidence Gate**: Look for "Confidence gate:" messages
- **Setup Count**: Monitor number of setups generated

---

## ⚠️ **SAFETY FEATURES**

### **Automatic Clamping**
- **Minimum**: 0.50 (prevents overly permissive thresholds)
- **Maximum**: 0.95 (prevents overly restrictive thresholds)
- **Warning**: System logs when values are clamped

### **Validation**
- **Range Check**: All values validated before use
- **Type Check**: Only numeric values accepted
- **Fallback**: Invalid values ignored, base thresholds used

### **Error Handling**
- **Graceful Degradation**: System continues working with defaults
- **Warning Messages**: Clear feedback when issues occur
- **Logging**: All changes and issues logged for audit

---

## 🔄 **CHANGING THRESHOLDS**

### **Temporary Changes (UI)**
1. Use the dashboard UI
2. Set new values
3. Click "Apply Custom Thresholds"
4. Refresh page to see changes

### **Permanent Changes (Environment)**
1. Set environment variables
2. Restart the application
3. Changes persist across sessions

### **Reverting Changes**
1. **UI Method**: Click "Reset to Defaults"
2. **Environment Method**: Remove environment variables
3. **Manual Reset**: Delete environment variables and restart

---

## 📈 **PERFORMANCE IMPACT**

### **Threshold Effects**
- **Higher Thresholds**: Fewer setups, potentially higher quality
- **Lower Thresholds**: More setups, potentially lower quality
- **Optimal Range**: 0.60-0.75 for most timeframes

### **Monitoring Performance**
- **Win Rate**: Track setup success rate
- **Setup Frequency**: Monitor number of setups generated
- **Confidence Distribution**: Analyze confidence levels of setups

---

## 🎯 **BEST PRACTICES**

### **1. Start Conservative**
- Begin with default thresholds
- Monitor performance for 1-2 weeks
- Adjust gradually based on results

### **2. Test Changes**
- Make small adjustments (0.02-0.05 increments)
- Monitor for at least 1 week before further changes
- Keep track of performance metrics

### **3. Consider Market Conditions**
- **Volatile Markets**: May need higher thresholds
- **Calm Markets**: May allow lower thresholds
- **Trending vs Ranging**: Different thresholds may work better

### **4. Timeframe Alignment**
- **Higher Timeframes**: Generally allow lower thresholds
- **Lower Timeframes**: Generally need higher thresholds
- **Your Trading Style**: Align with your risk tolerance

---

## 🆘 **TROUBLESHOOTING**

### **Common Issues**

#### **Changes Not Applied**
- **Check**: Refresh the dashboard page
- **Verify**: Look for override indicators
- **Restart**: Restart the application if using environment variables

#### **Invalid Values**
- **Range**: Ensure values are between 0.50 and 0.95
- **Format**: Use decimal format (0.75, not 75%)
- **Type**: Ensure values are numeric

#### **System Errors**
- **Logs**: Check application logs for errors
- **Fallback**: System will use base thresholds if errors occur
- **Reset**: Use "Reset to Defaults" if needed

### **Getting Help**
- **Documentation**: Check this guide
- **Logs**: Review application logs
- **Support**: Contact system administrator

---

## 🎉 **SUMMARY**

The adaptive confidence gate system provides flexible, user-friendly ways to customize confidence thresholds:

1. **UI Dashboard**: Easy, interactive configuration
2. **Environment Variables**: Permanent, scriptable configuration
3. **Safety Features**: Automatic validation and clamping
4. **Monitoring**: Clear feedback and logging
5. **Flexibility**: Per-timeframe customization

**Start with the defaults, monitor performance, and adjust gradually based on your trading results!** 🚀
