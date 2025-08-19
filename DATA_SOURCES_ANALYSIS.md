# Data Sources Analysis: Real vs Synthetic Data

## Overview

The Alpha12_24 system uses a mix of **real-time data** and **synthetic/dummy data** depending on data availability and API access. This analysis categorizes all data sources by their authenticity.

---

## ✅ **REAL DATA SOURCES (Live & Up-to-Date)**

### **🔧 Technical Indicators (100% Real)**
- **Source**: Calculated from real OHLCV data
- **Status**: ✅ **100% Real**
- **Details**: All technical indicators (RSI, MACD, Bollinger Bands, etc.) are calculated from real price data

### **📊 Price Data (100% Real)**
- **Source**: Binance Spot API, Bybit API
- **Status**: ✅ **100% Real**
- **Endpoints**:
  - `https://api.binance.com/api/v3/klines`
  - `https://api.bybit.com/v5/market/kline`
- **Data**: Real OHLCV (Open, High, Low, Close, Volume)

### **📈 Order Book Data (100% Real)**
- **Source**: Binance Spot API
- **Status**: ✅ **100% Real**
- **Endpoint**: `https://api.binance.com/api/v3/depth`
- **Features**:
  - `ob_imb_top20`: Real order book imbalance
  - `ob_spread_w`: Real weighted spread
  - `ob_bidv_top20`: Real bid volume
  - `ob_askv_top20`: Real ask volume

### **🎯 Options Data - RR25 (100% Real)**
- **Source**: Deribit API
- **Status**: ✅ **100% Real**
- **Endpoint**: `https://www.deribit.com/api/v2`
- **Features**:
  - Real 25-delta risk reversal calculations
  - Real implied volatility data
  - Real options chain data

### **📊 Market Microstructure (100% Real)**
- **Source**: Calculated from real OHLCV
- **Status**: ✅ **100% Real**
- **Features**:
  - Real order flow proxies
  - Real volume-price relationships
  - Real market efficiency metrics

---

## ⚠️ **SYNTHETIC/DUMMY DATA SOURCES**

### **📈 Macroeconomic Data (100% Synthetic)**
- **Source**: Generated with `np.random.normal()`
- **Status**: ❌ **100% Synthetic**
- **Features**:
  - `vix_close`, `vix_change`, `vix_volatility`: Synthetic VIX data
  - `dxy_close`, `dxy_change`, `dxy_volatility`: Synthetic DXY data
  - `gold_close`, `gold_change`, `gold_volatility`: Synthetic Gold data
  - `treasury_*`: Synthetic Treasury yields
  - `inflation_*`: Synthetic inflation data
  - `fed_funds_rate`: Synthetic Fed rate

### **😊 ETF Flow Data (100% Synthetic)**
- **Source**: Generated with random patterns
- **Status**: ❌ **100% Synthetic**
- **Features**:
  - `sentiment_etf_flows`: Synthetic ETF flows
  - `etf_flow_sentiment`: Synthetic flow sentiment
  - ETF premium/discount data

### **🔗 On-Chain Data (Variable)**
- **Source**: Placeholder implementation
- **Status**: ⚠️ **Mostly Synthetic**
- **Features**: `onchain_*` variables (when available, but currently synthetic)

### **📊 Fear & Greed Index (Synthetic)**
- **Source**: Generated with `np.random.normal()`
- **Status**: ❌ **100% Synthetic**
- **Features**: `fear_greed`, `fear_greed_normalized`

---

## 📊 **Detailed Breakdown by Category**

### **✅ REAL DATA (Live & Accurate)**

| Category | Data Source | Real/Synthetic | API Endpoint |
|----------|-------------|----------------|--------------|
| **Price Data** | Binance Spot | ✅ Real | `api.binance.com/api/v3/klines` |
| **Price Data** | Bybit | ✅ Real | `api.bybit.com/v5/market/kline` |
| **Order Book** | Binance Spot | ✅ Real | `api.binance.com/api/v3/depth` |
| **Options (RR25)** | Deribit | ✅ Real | `deribit.com/api/v2` |
| **Technical Indicators** | Calculated | ✅ Real | From real OHLCV |
| **Market Microstructure** | Calculated | ✅ Real | From real OHLCV |

### **❌ SYNTHETIC DATA (Generated)**

| Category | Data Source | Real/Synthetic | Generation Method |
|----------|-------------|----------------|-------------------|
| **VIX Data** | Generated | ❌ Synthetic | `np.random.normal(20.0, 0.05)` |
| **DXY Data** | Generated | ❌ Synthetic | `np.random.normal(100.0, 0.01)` |
| **Gold Data** | Generated | ❌ Synthetic | `np.random.normal(2000.0, 0.015)` |
| **Treasury Yields** | Generated | ❌ Synthetic | `np.random.normal()` for each yield |
| **Inflation Data** | Generated | ❌ Synthetic | `np.random.normal()` for each metric |
| **Fed Rate** | Generated | ❌ Synthetic | `np.random.normal(5.25, 0.1)` |
| **ETF Flows** | Generated | ❌ Synthetic | Random flow patterns |
| **Fear & Greed** | Generated | ❌ Synthetic | `np.random.normal(50, 15)` |

---

## 🔍 **Code Evidence**

### **Real Data Examples:**

```python
# Real Binance API call
def get_klines(self, symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 300):
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    data = self._make_request("klines", params)  # Real API call
    # Process real data...

# Real Order Book API call
def fetch_depth(symbol: str = "BTCUSDT", limit: int = 100):
    r = _sess.get("https://api.binance.com/api/v3/depth",  # Real API
                  params={"symbol": symbol, "limit": limit})
    # Process real order book data...

# Real Deribit RR25 API call
def get_index_price(currency="BTC"):
    idx = _get("/public/get_index_price", {"index_name": name})  # Real API
    return float(idx.get("index_price", float("nan")))
```

### **Synthetic Data Examples:**

```python
# Synthetic VIX data
def get_vix_data(self, days: int = 30):
    # Synthetic VIX data
    base_price = 20.0
    returns = np.random.normal(0, 0.05, days)  # Random generation
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

# Synthetic Treasury yields
def get_treasury_yields(self):
    yields = {
        'yield_2y': np.random.normal(4.5, 0.5),    # Random generation
        'yield_5y': np.random.normal(4.2, 0.4),    # Random generation
        'yield_10y': np.random.normal(4.0, 0.3),   # Random generation
        # ... more random yields
    }

# Synthetic Fed rate
def get_fed_funds_rate(self):
    return np.random.normal(5.25, 0.1)  # Random generation
```

---

## 📈 **Impact on Trading Decisions**

### **✅ High-Impact Real Data:**
- **Price Data**: Directly affects all calculations
- **Order Book**: Critical for entry/exit decisions
- **Options (RR25)**: Important sentiment indicator
- **Technical Indicators**: Core trading signals

### **⚠️ Medium-Impact Synthetic Data:**
- **Macroeconomic**: Used for regime detection but synthetic
- **ETF Flows**: Sentiment indicator but synthetic
- **Fear & Greed**: Market sentiment but synthetic

### **🔧 Calculated Real Data:**
- **Technical Indicators**: Real calculations from real prices
- **Market Microstructure**: Real calculations from real data

---

## 🎯 **Summary**

### **Real Data (Live & Accurate):**
- **Price Data**: 100% real from Binance/Bybit APIs
- **Order Book**: 100% real from Binance API
- **Options (RR25)**: 100% real from Deribit API
- **Technical Indicators**: 100% real (calculated from real prices)
- **Market Microstructure**: 100% real (calculated from real data)

### **Synthetic Data (Generated):**
- **Macroeconomic**: 100% synthetic (VIX, DXY, Gold, Treasuries, Inflation, Fed Rate)
- **ETF Flows**: 100% synthetic
- **Fear & Greed**: 100% synthetic
- **On-Chain**: Mostly synthetic (placeholder implementation)

### **Key Insight:**
The system uses **real data for core trading decisions** (price, order book, options) but **synthetic data for macroeconomic context**. The most critical trading signals come from real data sources, while synthetic data provides additional context that could be enhanced with real APIs in the future.

### **Recommendation:**
To improve accuracy, consider integrating real APIs for:
1. **VIX data** (CBOE API)
2. **DXY data** (Federal Reserve API)
3. **Gold data** (Kitco or similar)
4. **Treasury yields** (Federal Reserve API)
5. **ETF flows** (ETF provider APIs)
6. **Fear & Greed** (Alternative.me API)
