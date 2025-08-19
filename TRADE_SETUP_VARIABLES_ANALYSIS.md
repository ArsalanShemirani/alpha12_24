# Trade Setup Variables Analysis

## Overview

The Alpha12_24 system considers a comprehensive set of variables across multiple categories when generating trade setups. This analysis covers all technical, sentiment, on-chain, and market microstructure factors that influence setup generation.

## 📊 **3. Does the system consider sentiment in generating trade setups?**

**✅ YES** - The system considers multiple types of sentiment:

### **Sentiment Variables Included:**

1. **Technical Sentiment Indicators:**
   - `rsi_sentiment`: RSI-based sentiment (-1, 0, 1)
   - `macd_sentiment`: MACD signal crossover sentiment
   - `bb_sentiment`: Bollinger Band position sentiment

2. **ETF Flow Sentiment:**
   - `sentiment_etf_flows`: ETF flow sentiment from external data
   - `etf_flow_sentiment`: Aggregated ETF flow sentiment

3. **Market Sentiment:**
   - `fear_greed`: Fear & Greed Index
   - `risk_on_regime`: Risk-on market regime
   - `risk_off_regime`: Risk-off market regime

4. **Options Sentiment (RR25):**
   - `rr25`: Risk Reversal 25-delta (options sentiment)
   - RR25 gate for directional bias

---

## 📈 **Complete List of Variables Considered in Setup Generation**

### **🔧 TECHNICAL INDICATORS (41+ features)**

#### **Price-Based Features:**
- `returns`: Price returns
- `log_returns`: Logarithmic returns
- `high_low_ratio`: High/Low ratio
- `close_open_ratio`: Close/Open ratio

#### **Moving Averages:**
- `sma_10`: Simple Moving Average (10 periods)
- `sma_20`: Simple Moving Average (20 periods)
- `sma_50`: Simple Moving Average (50 periods)
- `ema_12`: Exponential Moving Average (12 periods)
- `ema_26`: Exponential Moving Average (26 periods)

#### **Price Momentum:**
- `price_momentum_1h`: 1-hour price momentum
- `price_momentum_4h`: 4-hour price momentum
- `price_momentum_24h`: 24-hour price momentum

#### **Volatility Features:**
- `volatility_1h`: 1-hour volatility
- `volatility_4h`: 4-hour volatility
- `volatility_24h`: 24-hour volatility
- `volatility_7d`: 7-day volatility

#### **Volume Features:**
- `volume_sma`: Volume Simple Moving Average
- `volume_ratio`: Current volume vs average
- `volume_momentum`: Volume momentum

#### **Advanced Technical Indicators (TA-Lib):**
- `rsi_14`, `rsi_21`: Relative Strength Index
- `macd`, `macd_signal`, `macd_hist`: MACD components
- `bb_upper`, `bb_middle`, `bb_lower`: Bollinger Bands
- `bb_width`, `bb_position`: Bollinger Band metrics
- `stoch_k`, `stoch_d`: Stochastic Oscillator
- `atr`, `atr_ratio`: Average True Range
- `adx`: Average Directional Index
- `cci`: Commodity Channel Index
- `williams_r`: Williams %R

#### **Trend Features:**
- `trend_strength`: Trend strength vs SMA20
- `trend_strength_50`: Trend strength vs SMA50

#### **Support/Resistance:**
- `support_20`: 20-period support level
- `resistance_20`: 20-period resistance level
- `price_vs_support`: Price vs support ratio
- `price_vs_resistance`: Price vs resistance ratio

---

### **🎯 MARKET MICROSTRUCTURE FEATURES**

#### **Order Flow Proxies:**
- `body_size`: Candle body size
- `upper_shadow`: Upper shadow length
- `lower_shadow`: Lower shadow length

#### **Volume-Price Relationship:**
- `volume_price_trend`: Volume-weighted price trend
- `volume_weighted_price`: Volume-weighted average price

#### **Market Efficiency:**
- `hurst_exponent`: Hurst exponent for market efficiency

#### **Price Acceleration:**
- `price_acceleration`: Price momentum acceleration
- `volume_acceleration`: Volume momentum acceleration

#### **Market Regime Features:**
- `trend_regime`: Trend regime classification
- `volatility_regime`: Volatility regime classification

---

### **📊 MACRO TREND FEATURES (Higher Timeframe)**

#### **4-Hour Timeframe:**
- `htf4h_ema_fast`: 4H fast EMA
- `htf4h_ema_slow`: 4H slow EMA
- `htf4h_slope`: 4H EMA slope
- `htf4h_strength`: 4H trend strength

#### **1-Day Timeframe:**
- `htf1d_ema_fast`: 1D fast EMA
- `htf1d_ema_slow`: 1D slow EMA
- `htf1d_slope`: 1D EMA slope
- `htf1d_strength`: 1D trend strength

#### **Macro Agreement:**
- `macro_agree_trend`: Trend agreement between timeframes
- `macro_agree_slope`: Slope agreement between timeframes

---

### **😊 SENTIMENT FEATURES**

#### **Technical Sentiment:**
- `rsi_sentiment`: RSI-based sentiment (-1, 0, 1)
- `macd_sentiment`: MACD signal crossover sentiment
- `bb_sentiment`: Bollinger Band position sentiment

#### **ETF Flow Sentiment:**
- `sentiment_etf_flows`: ETF flow sentiment
- `etf_flow_sentiment`: Aggregated ETF flow sentiment

#### **Market Sentiment:**
- `fear_greed`: Fear & Greed Index
- `risk_on_regime`: Risk-on market regime
- `risk_off_regime`: Risk-off market regime

---

### **🔗 ON-CHAIN FEATURES**

#### **Blockchain Data:**
- `onchain_*`: Various on-chain metrics (when available)
- Network activity indicators
- Transaction volume metrics
- Wallet behavior patterns

---

### **📈 MACROECONOMIC FEATURES**

#### **VIX (Volatility Index):**
- `vix_close`: VIX closing price
- `vix_change`: VIX price change
- `vix_volatility`: VIX volatility

#### **DXY (Dollar Index):**
- `dxy_change`: DXY price change
- `dxy_volatility`: DXY volatility

#### **Gold:**
- `gold_close`: Gold closing price
- `gold_change`: Gold price change
- `gold_volatility`: Gold volatility

#### **Treasury Yields:**
- `treasury_*`: Various treasury yield metrics

#### **Inflation Data:**
- `inflation_*`: Various inflation metrics

#### **Federal Reserve:**
- `fed_funds_rate`: Federal funds rate
- `real_rate`: Real interest rate

#### **Macro Regimes:**
- `risk_on_regime`: Risk-on market conditions
- `risk_off_regime`: Risk-off market conditions
- `dollar_strength`: Dollar strength indicator
- `safe_haven_demand`: Safe haven demand

---

### **📊 ORDER BOOK FEATURES (Live)**

#### **Order Book Imbalance:**
- `ob_imb_top20`: Order book imbalance (top 20 levels)
- `ob_spread_w`: Weighted spread
- `ob_bidv_top20`: Bid volume (top 20)
- `ob_askv_top20`: Ask volume (top 20)

---

### **🎯 OPTIONS SENTIMENT (RR25)**

#### **Risk Reversal:**
- `rr25`: 25-delta risk reversal (options sentiment)
- RR25 threshold for directional bias

---

### **⚙️ GATING VARIABLES**

#### **Confidence Gates:**
- `min_conf_arm`: Minimum confidence to arm setup
- `confidence_threshold`: Model confidence threshold

#### **Regime Gates:**
- `gate_regime`: Enable/disable regime gating
- `regime_override_conf`: Confidence override for regime gate

#### **Order Book Gates:**
- `gate_ob`: Enable/disable order book gating
- `ob_edge_delta`: Order book edge delta threshold
- `ob_signed_thr`: Order book signed threshold
- `ob_neutral_conf`: Neutral confidence for OB gate

#### **RR25 Gates:**
- `gate_rr25`: Enable/disable RR25 gating
- `rr25_thresh`: RR25 threshold

#### **Frequency Gates:**
- `max_setups_per_day`: Maximum setups per day

---

### **💰 RISK & SIZING VARIABLES**

#### **Position Sizing:**
- `acct_balance`: Account balance
- `max_leverage`: Maximum leverage
- `risk_per_trade_pct`: Risk per trade percentage

#### **Setup Geometry:**
- `k_entry`: Entry distance multiplier
- `k_stop`: Stop distance multiplier
- `valid_bars`: Setup validity period
- `entry_buffer_bps`: Entry buffer in basis points
- `trigger_rule`: Trigger rule ("touch" or "close-through")

#### **Risk Management:**
- `min_rr`: Minimum risk/reward ratio
- `stop_min_frac`: Minimum stop fraction
- `taker_bps_per_side`: Taker fees per side

---

### **🎯 TARGET VARIABLES**

#### **Directional Targets:**
- `target_12h`: 12-hour directional target
- `target_24h`: 24-hour directional target

#### **Volatility-Adjusted Targets:**
- `target_12h_vol_adj`: Volatility-adjusted 12h target
- `target_24h_vol_adj`: Volatility-adjusted 24h target

#### **Trend-Following Targets:**
- `target_12h_trend`: Trend-following 12h target
- `target_24h_trend`: Trend-following 24h target

---

## 🔄 **Setup Generation Process**

### **1. Feature Engineering Pipeline:**
```
OHLCV Data → Technical Features → Microstructure Features → 
Macro Features → Sentiment Features → On-chain Features → 
Target Variables → Feature Hardening → Model Training
```

### **2. Setup Generation Flow:**
```
1. Model Prediction → Confidence Score
2. Direction Selection → Long/Short/Flat
3. Gate Validation → Regime/OB/RR25 gates
4. Setup Building → Entry/Stop/Target levels
5. Position Sizing → Risk-based sizing
6. Setup Storage → CSV persistence
7. Telegram Alert → Notification
```

### **3. Gating Hierarchy:**
```
Confidence Gate → Regime Gate → RR25 Gate → OB Gate → Setup Creation
```

---

## 📊 **Variable Categories Summary**

| Category | Count | Examples |
|----------|-------|----------|
| **Technical Indicators** | 41+ | RSI, MACD, Bollinger Bands, ATR |
| **Market Microstructure** | 8+ | Order flow, volume-price, efficiency |
| **Macro Trends** | 8+ | Higher timeframe EMAs, slopes |
| **Sentiment** | 6+ | Technical sentiment, ETF flows, RR25 |
| **On-chain** | Variable | Network metrics, wallet behavior |
| **Macroeconomic** | 15+ | VIX, DXY, Gold, Treasuries, Fed rates |
| **Order Book** | 4 | Imbalance, spread, volumes |
| **Gating Variables** | 10+ | Confidence, regime, OB, RR25 thresholds |
| **Risk & Sizing** | 10+ | Balance, leverage, risk, geometry |
| **Target Variables** | 6+ | Directional, volatility-adjusted, trend |

**Total Variables: 100+**

---

## 🎯 **Key Insights**

### **✅ Sentiment Integration:**
- **Technical Sentiment**: RSI, MACD, Bollinger Band sentiment
- **Market Sentiment**: Fear & Greed, risk regimes
- **Options Sentiment**: RR25 risk reversal
- **Flow Sentiment**: ETF flows and order book imbalance

### **✅ Comprehensive Coverage:**
- **Technical**: 41+ traditional indicators
- **Microstructure**: Order flow and market efficiency
- **Macro**: Higher timeframe trends and economic data
- **Sentiment**: Multiple sentiment sources
- **On-chain**: Blockchain metrics (when available)

### **✅ Multi-Timeframe Analysis:**
- **1H**: Primary analysis timeframe
- **4H**: Macro trend alignment
- **1D**: Long-term trend context

### **✅ Risk Management:**
- **Gating**: Multiple validation gates
- **Sizing**: Risk-based position sizing
- **Geometry**: ATR-based entry/exit levels

The system provides a comprehensive, multi-factor approach to trade setup generation, incorporating technical, sentiment, macroeconomic, and market microstructure variables for robust signal generation.
