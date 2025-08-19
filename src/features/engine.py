#!/usr/bin/env python3
"""
Feature engineering engine for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Technical analysis imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using simplified technical indicators.")


class FeatureEngine:
    """Feature engineering engine for alpha12_24"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = None
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis features
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical features added
        """
        out = df.copy()
        
        # Price-based features
        out['returns'] = out['close'].pct_change()
        out['log_returns'] = np.log(out['close'] / out['close'].shift(1))
        out['high_low_ratio'] = out['high'] / out['low']
        out['close_open_ratio'] = out['close'] / out['open']
        
        # Moving averages (use shorter periods for smaller datasets)
        out['sma_10'] = out['close'].rolling(10, min_periods=5).mean()
        out['sma_20'] = out['close'].rolling(20, min_periods=10).mean()
        out['sma_50'] = out['close'].rolling(50, min_periods=25).mean()
        out['ema_12'] = out['close'].ewm(span=12).mean()
        out['ema_26'] = out['close'].ewm(span=26).mean()
        
        # Price momentum
        out['price_momentum_1h'] = out['close'].pct_change(1)
        out['price_momentum_4h'] = out['close'].pct_change(4)
        out['price_momentum_24h'] = out['close'].pct_change(24)
        
        # Volatility features
        out['volatility_1h'] = out['returns'].rolling(1, min_periods=1).std()
        out['volatility_4h'] = out['returns'].rolling(4, min_periods=2).std()
        out['volatility_24h'] = out['returns'].rolling(24, min_periods=12).std()
        out['volatility_7d'] = out['returns'].rolling(168, min_periods=84).std()
        
        # Volume features
        out['volume_sma'] = out['volume'].rolling(20, min_periods=10).mean()
        out['volume_ratio'] = out['volume'] / out['volume_sma']
        out['volume_momentum'] = out['volume'].pct_change()
        
        if TALIB_AVAILABLE:
            # RSI
            out['rsi_14'] = talib.RSI(out['close'], timeperiod=14)
            out['rsi_21'] = talib.RSI(out['close'], timeperiod=21)
            
            # MACD
            out['macd'], out['macd_signal'], out['macd_hist'] = talib.MACD(
                out['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Bollinger Bands
            out['bb_upper'], out['bb_middle'], out['bb_lower'] = talib.BBANDS(
                out['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / out['bb_middle']
            out['bb_position'] = (out['close'] - out['bb_lower']) / (out['bb_upper'] - out['bb_lower'])
            
            # Stochastic
            out['stoch_k'], out['stoch_d'] = talib.STOCH(
                out['high'], out['low'], out['close'], 
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            
            # ATR
            out['atr'] = talib.ATR(out['high'], out['low'], out['close'], timeperiod=14)
            out['atr_ratio'] = out['atr'] / out['close']
            
            # ADX
            out['adx'] = talib.ADX(out['high'], out['low'], out['close'], timeperiod=14)
            
            # CCI
            out['cci'] = talib.CCI(out['high'], out['low'], out['close'], timeperiod=14)
            
            # Williams %R
            out['williams_r'] = talib.WILLR(out['high'], out['low'], out['close'], timeperiod=14)
        else:
            # Simplified technical indicators without TA-Lib
            out['rsi_14'] = self._calculate_rsi(out['close'], 14)
            out['macd'] = out['ema_12'] - out['ema_26']
            out['macd_signal'] = out['macd'].ewm(span=9).mean()
            out['macd_hist'] = out['macd'] - out['macd_signal']
            
            # Simplified Bollinger Bands
            out['bb_middle'] = out['sma_20']
            bb_std = out['close'].rolling(20, min_periods=10).std()
            out['bb_upper'] = out['bb_middle'] + (2 * bb_std)
            out['bb_lower'] = out['bb_middle'] - (2 * bb_std)
            out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / out['bb_middle']
            out['bb_position'] = (out['close'] - out['bb_lower']) / (out['bb_upper'] - out['bb_lower'])
        
        # Trend features
        out['trend_strength'] = (out['close'] - out['sma_20']) / out['sma_20']
        out['trend_strength_50'] = (out['close'] - out['sma_50']) / out['sma_50']
        
        # Support/Resistance
        out['support_20'] = out['low'].rolling(20, min_periods=10).min()
        out['resistance_20'] = out['high'].rolling(20, min_periods=10).max()
        out['price_vs_support'] = (out['close'] - out['support_20']) / out['support_20']
        out['price_vs_resistance'] = (out['close'] - out['resistance_20']) / out['resistance_20']
        
        return out
    
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with microstructure features added
        """
        out = df.copy()
        
        # Order flow proxies
        out['body_size'] = abs(out['close'] - out['open']) / out['open']
        out['upper_shadow'] = (out['high'] - np.maximum(out['open'], out['close'])) / out['open']
        out['lower_shadow'] = (np.minimum(out['open'], out['close']) - out['low']) / out['open']
        
        # Volume-price relationship
        out['volume_price_trend'] = (out['volume'] * out['price_momentum_1h']).rolling(5, min_periods=3).sum()
        out['volume_weighted_price'] = (out['volume'] * out['close']).rolling(5, min_periods=3).sum() / out['volume'].rolling(5, min_periods=3).sum()
        
        # Market efficiency
        out['hurst_exponent'] = self._calculate_hurst_exponent(out['close'])
        
        # Price acceleration
        out['price_acceleration'] = out['price_momentum_1h'].diff()
        out['volume_acceleration'] = out['volume_momentum'].diff()
        
        # Market regime features
        out['trend_regime'] = np.where(out['trend_strength'] > 0.02, 1, 
                                     np.where(out['trend_strength'] < -0.02, -1, 0))
        out['volatility_regime'] = np.where(out['volatility_24h'] > out['volatility_24h'].rolling(20, min_periods=10).mean(), 1, 0)
        
        return out

    def add_macro_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add higher-timeframe (HTF) macro trend features derived only from OHLCV.
        - Resamples to 4H and 1D to compute EMA slopes and alignment with LTF.
        - Creates normalized slope features and macro agreement flags.
        """
        if df is None or df.empty:
            return df
        out = df.copy()

        # Ensure DatetimeIndex with freq where possible
        if not isinstance(out.index, (pd.DatetimeIndex,)):
            return out
        if out.index.tz is None:
            out = out.tz_localize("UTC")

        # Helper to build HTF features then align back
        def _make_htf(fe, rule: str, prefix: str) -> pd.DataFrame:
            r = fe.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
            # EMAs and slopes
            ema_fast = r["close"].ewm(span=12).mean()
            ema_slow = r["close"].ewm(span=48).mean()
            slope = ema_fast.pct_change().rolling(5, min_periods=3).mean()  # smoother slope
            strength = (ema_fast - ema_slow) / (r["close"].rolling(20, min_periods=10).mean())
            df_htf = pd.DataFrame({
                f"{prefix}_ema_fast": ema_fast,
                f"{prefix}_ema_slow": ema_slow,
                f"{prefix}_slope": slope,
                f"{prefix}_strength": strength,
            })
            return df_htf

        try:
            h4 = _make_htf(out, "4H", "htf4h")
            d1 = _make_htf(out, "1D", "htf1d")
            # Align back to original index (forward-fill to current bar)
            out = out.join(h4.reindex(out.index, method="ffill"))
            out = out.join(d1.reindex(out.index, method="ffill"))
            # Macro agreement flags
            out["macro_agree_trend"] = np.sign(out.get("trend_strength", 0.0)) * np.sign(out.get("htf4h_strength", 0.0))
            out["macro_agree_trend"] = out["macro_agree_trend"].fillna(0.0)
            out["macro_agree_slope"] = np.sign(out.get("price_momentum_1h", 0.0)) * np.sign(out.get("htf4h_slope", 0.0))
            out["macro_agree_slope"] = out["macro_agree_slope"].fillna(0.0)
        except Exception:
            # If resample fails (e.g., too few rows), return original
            return out

        return out
    
    def add_sentiment_features(self, df: pd.DataFrame, sentiment_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add sentiment features (only technical indicators, no synthetic data)
        
        Args:
            df: DataFrame with OHLCV data
            sentiment_data: Optional sentiment data dictionary
        
        Returns:
            DataFrame with sentiment features added
        """
        out = df.copy()
        
        # Technical sentiment indicators (only if base indicators exist)
        if 'rsi_14' in df.columns:
            out['rsi_sentiment'] = np.where(out['rsi_14'] > 70, -1, 
                                      np.where(out['rsi_14'] < 30, 1, 0))
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            out['macd_sentiment'] = np.where(out['macd'] > out['macd_signal'], 1, -1)
        
        if 'bb_position' in df.columns:
            out['bb_sentiment'] = np.where(out['bb_position'] > 0.8, -1, 
                                     np.where(out['bb_position'] < 0.2, 1, 0))
        
        # Sentiment data (if provided)
        if sentiment_data:
            for key, value in sentiment_data.items():
                out[f'sentiment_{key}'] = value
        
        return out
    
    def add_onchain_features(self, df: pd.DataFrame, onchain_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add on-chain features (only if real data provided, no synthetic data)
        
        Args:
            df: DataFrame with OHLCV data
            onchain_data: Optional on-chain data dictionary
        
        Returns:
            DataFrame with on-chain features added
        """
        out = df.copy()
        
        # On-chain data (if provided)
        if onchain_data:
            for key, value in onchain_data.items():
                out[f'onchain_{key}'] = value
        
        return out
    
    def create_targets(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """
        Create target variables for different horizons
        
        Args:
            df: DataFrame with OHLCV data
            horizons: List of prediction horizons in hours
        
        Returns:
            DataFrame with target variables added
        """
        out = df.copy()
        
        for horizon in horizons:
            # Directional targets
            out[f'target_{horizon}h'] = np.where(
                out['close'].shift(-horizon) > out['close'], 1, 0
            )
            
            # Volatility-adjusted targets
            volatility = out['close'].rolling(24, min_periods=12).std()
            out[f'target_{horizon}h_vol_adj'] = np.where(
                out['close'].shift(-horizon) > out['close'] * (1 + 0.5 * volatility), 1, 0
            )
            
            # Trend-following targets
            sma_20 = out['close'].rolling(20, min_periods=10).mean()
            out[f'target_{horizon}h_trend'] = np.where(
                out['close'].shift(-horizon) > sma_20, 1, 0
            )
        
        return out
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI without TA-Lib"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> pd.Series:
        """Calculate Hurst exponent for market efficiency"""
        def hurst(ts):
            try:
                lags = range(2, min(max_lag, len(ts)//2))
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                reg = np.polyfit(np.log(lags), np.log(tau), 1)
                return reg[0]
            except:
                return 0.5
        
        return series.rolling(50, min_periods=25).apply(hurst, raw=True)
    
    def harden_features(self, df: pd.DataFrame, cols: List[str], drop_threshold: float = 0.50, live_prefixes: Tuple[str, ...] = ("ob_", "rr25_", "deribit_", "iv_", "skew_rr")) -> Tuple[pd.DataFrame, List[str]]:
        """
        Harden features by:
          1) Excluding live-only cols from training set (kept in df for gating),
          2) Dropping chronically sparse features (> drop_threshold NaN),
          3) Dropping near-constant features (low variance),
          4) Pruning high collinearity (|corr| > 0.95),
          5) Light ffill/bfill on survivors to close merge gaps.
        """
        if not cols or df is None or df.empty:
            return df, cols

        # 1) Exclude live-only columns from train set
        train_cols = [c for c in cols if not any(c.startswith(p) for p in live_prefixes)]

        if not train_cols:
            return df, []

        X = df[train_cols]

        # 2) Drop sparse features
        na_frac = X.isna().mean()
        keep = [c for c in train_cols if na_frac.get(c, 0.0) <= drop_threshold]

        # 3) Drop near-constant features (variance below tiny threshold)
        if keep:
            var = X[keep].var(numeric_only=True).fillna(0.0)
            keep = [c for c in keep if var.get(c, 0.0) > 1e-10]

        # 4) Prune collinearity
        if len(keep) >= 2:
            Xk = X[keep].dropna()
            if len(Xk) >= 50:
                corr = Xk.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = set()
                for c in upper.columns:
                    if c in to_drop:
                        continue
                    highly = [r for r in upper.index if (upper.loc[r, c] > 0.95 if not np.isnan(upper.loc[r, c]) else False)]
                    to_drop.update(highly)
                keep = [c for c in keep if c not in to_drop]

        # 5) Fill small gaps
        if keep:
            df[keep] = df[keep].ffill().bfill()

        return df, keep
    
    def build_feature_matrix(self, df: pd.DataFrame, horizons: List[int], 
                           symbol: str = "BTCUSDT",
                           sentiment_data: Optional[Dict] = None,
                           onchain_data: Optional[Dict] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build complete feature matrix
        
        Args:
            df: DataFrame with OHLCV data
            horizons: List of prediction horizons
            sentiment_data: Optional sentiment data
            onchain_data: Optional on-chain data
        
        Returns:
            Tuple of (feature_matrix, target_series)
        """
        # Add all features
        df = self.add_technical_features(df)
        df = self.add_market_microstructure_features(df)
        df = self.add_macro_trend_features(df)
        df = self.add_sentiment_features(df, sentiment_data)
        df = self.add_onchain_features(df, onchain_data)
        df = self.create_targets(df, horizons)
        
        # Get feature columns (exclude targets)
        feature_cols = [col for col in df.columns if not col.startswith('target')]

        # --- Feature hardening: drop sparse features and fill gaps ---
        df, feature_cols = self.harden_features(
            df, feature_cols, drop_threshold=0.50,
            live_prefixes=("ob_", "rr25_", "deribit_", "iv_", "skew_rr")
        )

        # Remove rows with NaN values in training features, allow live-only cols to be NaN
        initial_rows = len(df)
        # Only enforce non-NaN on the *training* feature columns; allow live-only cols to be NaN
        if feature_cols:
            df_clean = df.dropna(subset=feature_cols)
        else:
            df_clean = df.copy()

        if len(df_clean) == 0:
            print(f"Warning: All {initial_rows} rows had NaN in training features. Attempting to fill NaN values...")
            if feature_cols:
                df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
                df_clean = df.dropna(subset=feature_cols)
            if len(df_clean) == 0:
                print("Error: Still no valid data after filling NaN values for training features")
                return pd.DataFrame(), []
        else:
            df = df_clean

        # Feature health: fraction of non-NaN across train features on each row
        if feature_cols:
            df["feat_health"] = (1.0 - df[feature_cols].isna().mean(axis=1)).astype(float)
        else:
            df["feat_health"] = 0.0
        self.feature_names = feature_cols

        # --- Optional order-book features (free, accurate) ---
        try:
            from src.data.orderbook_free import ob_features
            feats = ob_features(symbol=symbol, top=20)
            if feats:
                # Ensure columns exist
                for k in feats.keys():
                    if k not in df.columns:
                        df[k] = np.nan
                # Update the latest row with live snapshot, then back/forward fill so history is usable
                for k, v in feats.items():
                    df.loc[df.index[-1], k] = float(v)
                # Propagate the latest snapshot backward to avoid near-all-NaN columns;
                # still excluded from training features by harden_features' live_prefixes.
                for k in feats.keys():
                    df[k] = df[k].bfill().ffill()
            # Do NOT extend feature_cols with live-only OB columns; they are kept in df for gating, not for training.
        except Exception:
            # If OB not available, leave df unchanged; training set already excludes live-only features.
            pass

        return df, feature_cols
