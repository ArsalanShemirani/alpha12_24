#!/usr/bin/env python3
"""
Alpha12_24 autosignal daemon:
- Runs ad-hoc (triggered by systemd timer) to generate ~2 setups/day (configurable)
- Trains a model quickly per asset on latest data, produces a signal + confidence
- Applies hard gates (regime, RR25, OB imbalance edge-delta)
- Builds a pending limit-style setup (entry/stop/target via ATR)
- Sizes by risk, capped by leverage; writes to runs/setups.csv with origin="auto"
- Optional Telegram alert via env TG_BOT_TOKEN / TG_CHAT_ID
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Alpha12 imports
from src.core.config import config
from src.features.engine import FeatureEngine
from src.models.train import ModelTrainer

# --- Model loading imports
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd

# Check if XGBoost is available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import adaptive confidence gate
try:
    from src.trading.adaptive_confidence_gate import adaptive_confidence_gate
    ADAPTIVE_CONFIDENCE_AVAILABLE = True
except ImportError:
    ADAPTIVE_CONFIDENCE_AVAILABLE = False
    print("[autosignal] Adaptive confidence gate not available, using fallback")

# Import R:R invariant logging
try:
    from src.utils.rr_invariants import compute_rr_invariants, rr_invariant_writer, getenv_bool
    RR_INVARIANT_AVAILABLE = True
except ImportError:
    RR_INVARIANT_AVAILABLE = False
    print("[autosignal] R:R invariant logging not available")

MODEL_DIR = getattr(config, 'model_dir', 'artifacts')

def _load_latest_predictor(asset: str, interval: str):
    root = Path(MODEL_DIR)
    if not root.exists():
        return None, None
    # newest first
    for d in sorted([p for p in root.iterdir() if p.is_dir()], reverse=True):
        try:
            meta_p = d / "meta.json"
            pred_p = d / "model.joblib"
            if not meta_p.exists() or not pred_p.exists():
                continue
            meta = json.loads(meta_p.read_text())
            if meta.get("asset") == asset and meta.get("interval") == interval:
                predictor = joblib.load(pred_p)
                return predictor, meta
        except Exception:
            continue
    return None, None

# Data loaders
def _resolve_loader(source_choice: str):
    if source_choice.startswith("Composite"):
        from src.data.composite import assemble_spot_plus_bybit as load_df
    elif source_choice.startswith("Binance"):
        from src.data.binance_free import assemble as load_df
    else:
        from src.data.binance_free import assemble as load_df
    return load_df

# Order book (signed imbalance in [-1, +1])
from src.data.orderbook_free import ob_features

# --- Defaults / environment
MY_TZ = "Asia/Kuala_Lumpur"

RUNS_DIR = getattr(config, "runs_dir", "runs")
MODEL_DIR = getattr(config, "model_dir", "artifacts")

ASSETS = os.getenv("ALPHA12_ASSETS", "BTCUSDT,ETHUSDT").split(",")
ASSETS = [a.strip() for a in ASSETS if a.strip()]

INTERVAL = os.getenv("ALPHA12_AUTOSIGNAL_INTERVAL", getattr(config, "bar_interval", "1h"))

SOURCE_CHOICE = os.getenv("ALPHA12_SOURCE", "Composite (Binance Spot + Bybit derivs)")

MAX_SETUPS_PER_DAY = int(os.getenv("MAX_SETUPS_PER_DAY", "2"))  # 0 = unlimited

# Risk/sizing defaults (can be env-overridden)
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE_USD", "400"))
MAX_LEVERAGE     = int(os.getenv("MAX_LEVERAGE", "10"))
RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE_PCT", str(getattr(config, "risk_per_trade", 1.0))))  # %
MIN_RR           = float(os.getenv("MIN_RR", str(getattr(config, "min_rr", 1.8))))
TAKER_BPS_SIDE   = float(os.getenv("TAKER_BPS_PER_SIDE", str(getattr(config, "taker_bps_per_side", 5))))

# Setup geometry defaults
K_ENTRY          = float(os.getenv("K_ENTRY_ATR", "0.25"))   # entry offset in ATR (changed from 0.5 to 0.25 for autosignals)
K_STOP           = float(os.getenv("K_STOP_ATR", "1.0"))    # stop distance in ATR
VALID_BARS       = int(os.getenv("VALID_BARS", "24"))
ENTRY_BUFFER_BPS = float(os.getenv("ENTRY_BUFFER_BPS", "5.0"))
TRIGGER_RULE     = os.getenv("TRIGGER_RULE", "close-through")  # 'touch' | 'close-through'

# Model selection (XGBoost preferred if available, but UI can override)
PREFERRED_MODEL = os.getenv("ALPHA12_PREFERRED_MODEL", "xgb")  # 'auto', 'xgb', 'rf', 'logistic'
if PREFERRED_MODEL == "auto":
    if XGBOOST_AVAILABLE:
        PREFERRED_MODEL = "xgb"
    else:
        PREFERRED_MODEL = "rf"

# UI Configuration integration
def _get_ui_override_config():
    """Get UI configuration overrides for autosignal."""
    try:
        from src.core.ui_config import get_autosignal_config
        return get_autosignal_config()
    except Exception as e:
        print(f"[autosignal] UI config error: {e}")
        return {}

def _get_active_model():
    """Get the active model type, considering UI overrides."""
    # Check for UI override first
    ui_config = _get_ui_override_config()
    ui_model = ui_config.get("model_type")
    
    if ui_model and ui_model in ["xgb", "rf", "logistic"]:
        # Validate XGBoost availability if UI wants to use it
        if ui_model == "xgb" and not XGBOOST_AVAILABLE:
            print(f"[autosignal] UI requested XGBoost but not available, falling back to rf")
            return "rf"
        print(f"[autosignal] Using UI-overridden model: {ui_model}")
        return ui_model
    
    # Fall back to environment/default
    return PREFERRED_MODEL

# Gates
GATE_REGIME      = bool(int(os.getenv("GATE_REGIME", "1")))
GATE_RR25        = bool(int(os.getenv("GATE_RR25",   "1")))
GATE_OB          = bool(int(os.getenv("GATE_OB",     "1")))
RR25_THRESH      = float(os.getenv("RR25_THRESH", "0.00"))  # absolute threshold
OB_EDGE_DELTA    = float(os.getenv("OB_EDGE_DELTA", "0.20"))  # 0..1, delta from edges 0 or 1

# Max Pain weighting (optional)
MAXPAIN_ENABLE_WEIGHT = bool(int(os.getenv("MAXPAIN_ENABLE_WEIGHT", "1")))
MAXPAIN_DIST_REF_PCT  = float(os.getenv("MAXPAIN_DIST_REF_PCT", "5.0"))  # % distance at which full weight applies
MAXPAIN_WEIGHT_MAX    = float(os.getenv("MAXPAIN_WEIGHT_MAX", "0.10"))   # max +/- impact on confidence (e.g. 0.10 = 10%)
MAXPAIN_USE_ENHANCED  = bool(int(os.getenv("MAXPAIN_USE_ENHANCED", "1")))  # use enhanced multi-factor analysis

# Optional overrides
REGIME_OVERRIDE_CONF = float(os.getenv("REGIME_OVERRIDE_CONF", "0.0"))  # >= this conf bypasses regime gate
OB_NEUTRAL_CONF     = float(os.getenv("OB_NEUTRAL_CONF", "0.0"))      # >= this conf treats OB undecided as neutral

# Arm confidence threshold
MIN_CONF_ARM     = float(os.getenv("MIN_CONF_ARM", "0.60"))

# Telegram
TG_BOT_TOKEN     = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID       = os.getenv("TG_CHAT_ID", "")


# Autosignal defaults (more fill-friendly on calm hours)
# Can be overridden with environment variables or UI settings.
K_ENTRY_DEFAULT = float(os.getenv("AUTO_K_ENTRY_ATR", "0.30"))  # was 0.50
K_STOP_DEFAULT  = float(os.getenv("AUTO_K_STOP_ATR", "1.00"))
VALID_BARS_MIN  = int(os.getenv("AUTO_VALID_BARS_MIN", "24"))
TRIGGER_RULE_DEFAULT = os.getenv("AUTO_TRIGGER_RULE", "touch")      # "touch" for autosignal

# UI Configuration integration
def _get_ui_override_config():
    """Get UI configuration overrides for autosignal."""
    try:
        from src.core.ui_config import get_autosignal_config
        return get_autosignal_config()
    except Exception as e:
        print(f"[autosignal] UI config error: {e}")
        return {}


# --------- Time helpers (robust across pandas versions)
def _now_utc() -> pd.Timestamp:
    ts = pd.Timestamp.utcnow()
    if getattr(ts, "tz", None) is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def _now_local() -> pd.Timestamp:
    ts = pd.Timestamp.utcnow()
    if getattr(ts, "tz", None) is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MY_TZ)

def _to_my_tz_index(idx) -> pd.DatetimeIndex:
    try:
        idx = pd.to_datetime(idx, utc=True, errors="coerce")
        return idx.tz_convert(MY_TZ)
    except Exception:
        return pd.to_datetime(idx, errors="coerce", utc=True)

# --------- CSV utils (canonical schema)
SETUP_FIELDS = [
    "id","unique_id","asset","interval","direction","entry","stop","target","rr",
    "size_units","notional_usd","leverage",
    "created_at","expires_at","valid_bars","validity_source","triggered_at","status","confidence","trigger_rule","entry_buffer_bps",
    "origin"
]

def _generate_unique_id(asset: str, interval: str, direction: str, origin: str = "auto") -> str:
    """
    Generate a unique, user-friendly ID for setups.
    
    Format: {ORIGIN}-{ASSET}-{TIMEFRAME}-{DIRECTION}-{TIMESTAMP}
    Example: AUTO-ETHUSDT-4h-SHORT-20250822-1201
    """
    timestamp = _now_utc().strftime('%Y%m%d-%H%M')
    prefix = "AUTO" if origin == "auto" else "MANUAL"
    return f"{prefix}-{asset}-{interval}-{direction.upper()}-{timestamp}"

def _setups_csv_path() -> str:
    return os.path.join(RUNS_DIR, "setups.csv")

def _load_setups_df() -> pd.DataFrame:
    p = _setups_csv_path()
    if not os.path.exists(p):
        return pd.DataFrame(columns=SETUP_FIELDS)
    try:
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
    except Exception:
        import csv
        rows = []
        with open(p, "r", newline="") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
        df = pd.DataFrame(rows)
    for c in SETUP_FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SETUP_FIELDS].copy()
    for c in ("created_at","expires_at","triggered_at"):
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            try:
                df[c] = ts.dt.tz_convert(MY_TZ)
            except Exception:
                df[c] = ts
    try:
        df.to_csv(p, index=False)
    except Exception:
        pass
    return df

def _append_setup_row(row: dict):
    import csv
    p = _setups_csv_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    write_header = not os.path.exists(p)
    safe_row = {k: row.get(k, "") if row.get(k) is not None else "" for k in SETUP_FIELDS}
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SETUP_FIELDS, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        if write_header:
            w.writeheader()
        w.writerow(safe_row)


# --------- Telegram helpers (MarkdownV2 safe)
_TG_ESCAPE_RE = re.compile(r'([_*.\[\]()~`>#+\-=|{}!.])')

def _tg_escape_md2(text: str) -> str:
    if text is None:
        return ""
    return _TG_ESCAPE_RE.sub(r'\\\1', str(text))

def _send_telegram(text: str, timeout: int = 8) -> bool:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return False
    try:
        import requests
        payload = {"chat_id": TG_CHAT_ID, "text": _tg_escape_md2(text), "parse_mode": "MarkdownV2"}
        r = requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", json=payload, timeout=timeout)
        r.raise_for_status()
        return True
    except Exception:
        return False


# --------- Misc helpers
def _bars_per_day(interval: str) -> int:
    return {"5m":288, "15m":96, "1h":24, "4h":6, "1d":1}.get(interval, 96)

def _calc_limit(interval: str, days: int) -> int:
    base = _bars_per_day(interval) * days
    hard_cap = 1500 if interval in ("5m","15m") else 2500
    return min(base, hard_cap)

def _estimate_atr(df: pd.DataFrame, n: int = 14) -> float:
    if "atr" in df.columns and not df["atr"].dropna().empty:
        return float(df["atr"].dropna().iloc[-1])
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = (high - low).abs()
    tr = np.maximum(tr, (high - prev_close).abs())
    tr = np.maximum(tr, (low - prev_close).abs())
    atr = pd.Series(tr).rolling(n, min_periods=n).mean()
    val = atr.dropna()
    return float(val.iloc[-1]) if not val.empty else float((close.iloc[-1] * 0.002))  # ~20bps fallback


# --------- Gates
def _regime_gate(df: pd.DataFrame, direction: str) -> Optional[str]:
    """Return reason string if blocked, else None."""
    try:
        from src.trading.macro import get_macro_regime, is_direction_allowed
        
        # Get macro regime using MA200-only filter
        regime_result = get_macro_regime(df)
        
        # Check if direction is allowed
        is_allowed, reason = is_direction_allowed(regime_result, direction, allow_override=False)
        
        if not is_allowed:
            return f"macro gate: {regime_result.regime} vs {direction} ({regime_result.reason})"
        
        return None
    except Exception as e:
        print(f"[autosignal] Macro regime gate error: {e}")
        return None  # Allow if regime calculation fails

def _rr25_avg_for(asset_symbol: str) -> Optional[float]:
    """Read cached deribit rr25 JSON from runs/deribit_rr25_latest_{BTC,ETH}.json and average last 2."""
    runs = Path(RUNS_DIR)
    cur = "BTC" if "BTC" in asset_symbol else ("ETH" if "ETH" in asset_symbol else "BTC")
    p = runs / f"deribit_rr25_latest_{cur}.json"
    try:
        if not p.exists():
            return None
        rows = json.loads(p.read_text()).get("rows", [])
        if len(rows) == 0:
            return None
        vals = [float(r.get("rr25", float("nan"))) for r in rows[-2:]]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else None
    except Exception:
        return None

def _rr25_gate(direction: str, rr25: Optional[float], thr: float) -> Optional[str]:
    """Return reason string if blocked, else None."""
    if rr25 is None:
        return None  # treat as neutral
    if direction == "long" and not (rr25 >= +thr):
        return f"RR25 gate: rr={rr25:.4f} < +{thr:.4f}"
    if direction == "short" and not (rr25 <= -thr):
        return f"RR25 gate: rr={rr25:.4f} > -{thr:.4f}"
    return None

def _ob_gate(value: float, edge_delta: float) -> Optional[str]:
    """
    Edge-gated imbalance using SIGNED input in [-1, +1].

    T = 1 - 2*edge_delta (edge_delta is distance from edges 0 or 1 in raw space)
    - If signed ≥ +T ⇒ LONG
    - If signed ≤ -T ⇒ SHORT
    - Else ⇒ None

    Example: edge_delta=0.2 ⇒ T=0.6 ⇒ LONG if signed≥+0.6 (raw≥0.8), SHORT if signed≤-0.6 (raw≤0.2)
    """
    try:
        s = float(value)
    except Exception:
        return None
    if not np.isfinite(s):
        return None
    thr = float(max(0.0, min(1.0, 1.0 - 2.0*edge_delta)))
    if s >= +thr:
        return "long"
    if s <= -thr:
        return "short"
    return None


# --------- Core autosignal logic
def _train_and_score(asset: str, interval: str, days: int = 120) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
    """Load data, build features, train quick model, return prob_up for last row and feature DataFrame."""
    load_df = _resolve_loader(SOURCE_CHOICE)
    limit = _calc_limit(interval, days)
    df = load_df(asset, interval, limit)
    if df is None or df.empty:
        return None, None
    try:
        df.index = _to_my_tz_index(df.index)
    except Exception:
        pass

    fe = FeatureEngine()
    F, cols = fe.build_feature_matrix(df, config.horizons_hours, symbol=asset)
    if F is None or F.empty or not cols:
        return None, None

    tgt = f"target_{config.horizons_hours[0]}h"
    if tgt not in F.columns:
        return None, None

    X_all = F[cols]
    y_all = F[tgt]
    valid = ~(X_all.isna().any(axis=1) | y_all.isna())
    X_all, y_all = X_all[valid], y_all[valid]

    min_rows = 300 if interval in ("5m", "15m") else 150
    if len(X_all) < min_rows:
        return None, None

    # Try load latest predictor first
    predictor, meta = _load_latest_predictor(asset, interval)

    # NEW: Try to load timeframe-specific model first (if performance monitoring created one)
    if predictor is None:
        try:
            from src.daemon.performance_monitor import performance_monitor
            timeframe_model = performance_monitor.load_timeframe_model(asset, interval)
            if timeframe_model is not None:
                predictor = timeframe_model
                print(f"[autosignal] Using timeframe-specific model for {asset} {interval}")
        except Exception as e:
            print(f"[autosignal] No timeframe-specific model available for {asset} {interval}: {e}")

    if predictor is None:
        # Fallback: quick train
        trainer = ModelTrainer(config)
        # Allow enabling calibration via env (default OFF for speed on server)
        #   AUTOSIGNAL_CALIBRATE=1 to enable; AUTOSIGNAL_CALIB_CV to control folds (default 3)
        _CALIB_ENV = os.getenv("AUTOSIGNAL_CALIBRATE", "0") not in ("0", "false", "False", "")
        _CALIB_CV = int(os.getenv("AUTOSIGNAL_CALIB_CV", "3"))
        model = trainer.train_model(X_all, y_all, model_type=_get_active_model(), calibrate=_CALIB_ENV, calib_cv=_CALIB_CV)
        predictor = getattr(model, 'model', model)
    else:
        # (Optional) verify minimal feature intersection; if bad, quick-train fallback
        needed = list(X_all.columns)  # or your feature list
        try:
            # Test a dry predict on 1 row to ensure compatibility
            _ = predictor.predict_proba(X_all.iloc[[0]])
        except Exception:
            trainer = ModelTrainer(config)
            # Allow enabling calibration via env (default OFF for speed on server)
            #   AUTOSIGNAL_CALIBRATE=1 to enable; AUTOSIGNAL_CALIB_CV to control folds (default 3)
            _CALIB_ENV = os.getenv("AUTOSIGNAL_CALIBRATE", "0") not in ("0", "false", "False", "")
            _CALIB_CV = int(os.getenv("AUTOSIGNAL_CALIB_CV", "3"))
            model = trainer.train_model(X_all, y_all, model_type=_get_active_model(), calibrate=_CALIB_ENV, calib_cv=_CALIB_CV)
            predictor = getattr(model, 'model', model)

    # --- Robust probability extraction (trainer API or raw estimator) ---
    try:
        # Preferred path if your ModelTrainer implements predict_proba
        proba = predictor.predict_proba(X_all)  # shape (n, 2) or (n,)
    except Exception:
        # Fallback to the wrapped/base estimator
        base = getattr(predictor, "model", predictor)
        if hasattr(base, "predict_proba"):
            proba = base.predict_proba(X_all)
        else:
            # Last-resort heuristic: map binary predictions to pseudo-probabilities
            # (better than crashing; still allows gates/sizing to run)
            yhat = base.predict(X_all) if hasattr(base, "predict") else np.zeros(len(X_all))
            # Turn labels into [p_down, p_up]
            p_up = yhat.astype(float)
            proba = np.column_stack([1.0 - p_up, p_up])

    # Normalize to 2-column format
    if proba.ndim == 1:
        # treat as P(up)
        prob_up = float(proba[-1])
    else:
        # assume [:,1] is P(up)
        prob_up = float(proba[-1, 1])

    return prob_up, df


def _choose_direction(prob_up: float) -> Tuple[str, float]:
    """Return ('long'|'short'|'flat', confidence)."""
    if prob_up >= 0.5:
        return "long", prob_up
    else:
        return "short", 1.0 - prob_up


def _build_setup(direction: str, price: float, atr: float, rr: float,
                 k_entry: float, k_stop: float, valid_bars: int,
                 now_ts, bar_interval: str, entry_buffer_bps: float) -> dict:
    # Apply timeframe-specific stop loss multiplier
    stop_multiplier = _get_timeframe_stop_multiplier(bar_interval)
    adjusted_k_stop = k_stop * stop_multiplier
    
    if direction == "long":
        base_entry = price - k_entry * atr
        entry = base_entry * (1.0 - entry_buffer_bps/10000.0)
        stop  = entry - adjusted_k_stop * atr
        target = entry + rr * (entry - stop)
    else:
        base_entry = price + k_entry * atr
        entry = base_entry * (1.0 + entry_buffer_bps/10000.0)
        stop  = entry + adjusted_k_stop * atr
        target = entry - rr * (stop - entry)
    per_bar_min = {"5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}.get(bar_interval, 60)
    base_now = pd.to_datetime(now_ts, utc=True)
    try:
        base_now = base_now.tz_convert(MY_TZ)
    except Exception:
        pass
    expires_at = base_now + pd.Timedelta(minutes=valid_bars * per_bar_min)
    return {
        "entry": float(entry), "stop": float(stop), "target": float(target),
        "rr": float(rr), "expires_at": expires_at
    }


def build_autosetup_levels(direction: str, last_price: float, atr: float, rr: float, interval: str = "1h", features: dict = None, k_entry: float = None) -> dict:
    """
    Compute entry/stop/target levels for autosignal setups.
    Uses timeframe-specific stop loss percentages of entry price.
    Uses adaptive selector for take profit calculation if features available.
    """
    # Use provided k_entry or fall back to default
    if k_entry is None:
        k_entry = K_ENTRY_DEFAULT
    
    # Get timeframe-specific stop loss percentage of entry price
    stop_loss_pct = {
        "15m": 0.5,   # 0.5% of entry price
        "1h": 1.0,    # 1.0% of entry price
        "4h": 1.5,    # 1.5% of entry price
        "1d": 2.0     # 2.0% of entry price
    }
    
    stop_pct = stop_loss_pct.get(interval, 1.0)
    
    if direction == "long":
        entry = last_price - k_entry * atr
        stop = entry * (1.0 - stop_pct / 100.0)  # Stop loss as percentage of entry
    else:
        entry = last_price + k_entry * atr
        stop = entry * (1.0 + stop_pct / 100.0)  # Stop loss as percentage of entry
    
    # Try to use adaptive selector for take profit if features are available
    if features is not None:
        try:
            from src.trading.adaptive_selector import adaptive_selector
            
            # Prepare features (no macro inputs)
            clean_features = {k: v for k, v in features.items() 
                            if not k.startswith(('vix_', 'dxy_', 'gold_', 'treasury_', 'inflation_', 'fed_'))}
            
            # Select optimal stop/target using adaptive selector
            result = adaptive_selector.select_optimal_stop_target(
                features=clean_features,
                atr=atr,
                timeframe=interval,
                entry_price=entry,
                direction=direction
            )
            
            if result.success:
                # Use adaptive selector result, but always apply R:R cap
                rr_cap = {"15m": 1.5, "1h": 1.7, "4h": 2.0, "1d": 2.8}.get(interval, 1.5)
                original_rr = result.rr
                capped_rr = min(original_rr, rr_cap)
                
                # Always recalculate target to ensure exact R:R cap compliance
                target = entry + capped_rr * (entry - stop) if direction == "long" else entry - capped_rr * (stop - entry)
                
                if original_rr > rr_cap:
                    print(f"[autosignal] {interval}: Adaptive selector R:R capped from {original_rr:.2f} to {capped_rr:.2f}")
                else:
                    print(f"[autosignal] {interval}: Adaptive selector R:R within cap: {original_rr:.2f} <= {rr_cap}")
                
                rr = capped_rr
                print(f"[autosignal] {interval}: Using adaptive selector - RR: {rr:.2f}, p_hit: {result.p_hit:.3f}, EV_R: {result.ev_r:.4f}")
            else:
                # Fall back to fixed R:R calculation with timeframe cap
                rr_cap = {"15m": 1.5, "1h": 1.7, "4h": 2.0, "1d": 2.8}.get(interval, 1.5)
                capped_rr = min(rr, rr_cap)
                target = entry + capped_rr * (entry - stop) if direction == "long" else entry - capped_rr * (stop - entry)
                print(f"[autosignal] {interval}: Adaptive selector failed, using capped RR: {capped_rr:.2f} (original: {rr:.2f})")
                
        except Exception as e:
            # Fall back to fixed R:R calculation with timeframe cap
            rr_cap = {"15m": 1.5, "1h": 1.7, "4h": 2.0, "1d": 2.8}.get(interval, 1.5)
            capped_rr = min(rr, rr_cap)
            target = entry + capped_rr * (entry - stop) if direction == "long" else entry - capped_rr * (stop - entry)
            print(f"[autosignal] {interval}: Adaptive selector error: {e}, using capped RR: {capped_rr:.2f} (original: {rr:.2f})")
    else:
        # Use fixed R:R calculation when no features available with timeframe cap
        rr_cap = {"15m": 1.5, "1h": 1.7, "4h": 2.0, "1d": 2.8}.get(interval, 1.5)
        capped_rr = min(rr, rr_cap)
        target = entry + capped_rr * (entry - stop) if direction == "long" else entry - capped_rr * (stop - entry)
    
    # Update R:R to reflect the actual value used (for fallback cases only)
    if features is None or not result.success:
        # For fallback cases, calculate the actual R:R used
        if direction == "long":
            actual_rr = (target - entry) / (entry - stop) if entry > stop else rr
        else:
            actual_rr = (entry - target) / (stop - entry) if stop > entry else rr
        rr = actual_rr
    
    # Expiry in bars; caller will translate to minutes based on interval
    return {"entry": entry, "stop": stop, "target": target, "rr": rr, "valid_bars": VALID_BARS_MIN}


def _get_timeframe_stop_multiplier(interval: str) -> float:
    """
    Get stop loss multiplier based on timeframe.
    
    Higher timeframes have wider stops due to increased volatility:
    - 15m: 0.5x (tighter stops, more frequent trades)
    - 1h: 1.0x (baseline)
    - 4h: 1.5x (wider stops, fewer trades)
    - 1d: 2.0x (widest stops, least frequent trades)
    """
    multipliers = {
        "15m": 0.5,
        "1h": 1.0,
        "4h": 1.5,
        "1d": 2.0
    }
    return multipliers.get(interval, 1.0)

def _get_timeframe_notional(interval: str) -> float:
    """
    Get timeframe-specific notional position size.
    
    Returns the target notional position size for each timeframe:
    - 15m: $2,000 notional
    - 1h: $1,000 notional  
    - 4h: $667 notional
    - 1d: $500 notional
    """
    timeframe_notional = {
        "15m": 2000.0,
        "1h": 1000.0,
        "4h": 667.0,
        "1d": 500.0
    }
    return timeframe_notional.get(interval, 1000.0)  # Default to 1h if unknown

def _get_timeframe_risk_percentage(interval: str) -> float:
    """
    Get timeframe-specific risk percentage.
    
    Returns the risk percentage for each timeframe:
    - 15m: 0.5%
    - 1h: 1.0%
    - 4h: 1.5%
    - 1d: 2.0%
    """
    timeframe_risk = {
        "15m": 0.5,
        "1h": 1.0,
        "4h": 1.5,
        "1d": 2.0
    }
    return timeframe_risk.get(interval, 1.0)  # Default to 1h if unknown

def _size_position(entry_px: float, stop_px: float, balance: float, max_lev: int, risk_pct: float, 
                  interval: str = "1h", nominal_position_pct: float = 25.0) -> Tuple[float, float, float]:
    """
    Return (size_units, notional_usd, suggested_leverage).
    
    Implements timeframe-specific position sizing with consistent dollar risk:
    - All timeframes: Dollar risk = 2.5% of user-set balance
    - Position size calculated to achieve exactly 2.5% of balance risk
    - Notional amounts vary based on position size and entry price
    - Always 10x display leverage
    
    This ensures all trades lose maximum 2.5% of actual balance.
    """
    # Calculate stop distance
    stop_dist = abs(entry_px - stop_px)
    if stop_dist <= 0:
        return 0.0, 0.0, float(max_lev)
    
    # Calculate position size to achieve exactly 2.5% of balance risk
    target_dollar_risk = balance * 0.025  # Always 2.5% of user-set balance
    size_units = target_dollar_risk / stop_dist
    
    # Calculate actual notional
    notional = size_units * entry_px
    
    # Calculate actual dollar risk for verification
    actual_dollar_risk = size_units * stop_dist
    actual_risk_pct = (actual_dollar_risk / balance) * 100.0
    
    # Get timeframe-specific target notional for reference
    target_notional = _get_timeframe_notional(interval)
    timeframe_risk_pct = _get_timeframe_risk_percentage(interval)
    
    print(f"[autosignal] {interval} position sizing:")
    print(f"  Balance: ${balance:,.2f}")
    print(f"  Target dollar risk: ${target_dollar_risk:.2f} (2.5% of balance)")
    print(f"  Target notional: ${target_notional:,.0f}")
    print(f"  Actual notional: ${notional:,.0f}")
    print(f"  Target risk: {timeframe_risk_pct}%")
    print(f"  Actual risk: {actual_risk_pct:.2f}%")
    print(f"  Dollar risk: ${actual_dollar_risk:.2f}")
    print(f"  Display leverage: {max_lev}x")
    
    # Always return maximum leverage for display (10x)
    suggested_leverage = float(max_lev)
    
    return float(size_units), float(notional), suggested_leverage


def _daily_cap_reached(asset: str, interval: str, cap: int) -> bool:
    if cap <= 0:
        return False
    df_hist = _load_setups_df()
    if df_hist.empty:
        return False
    df = df_hist.copy()
    ts = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    try:
        df["created_at"] = ts.dt.tz_convert(MY_TZ)
    except Exception:
        df["created_at"] = ts
    now_l = _now_local()
    since = now_l - pd.Timedelta(days=1)
    mask = (
        (df["asset"] == asset) &
        (df["interval"] == interval) &
        (df["created_at"] >= since) &
        (df["status"].isin(["pending","triggered","target","stop","timeout","cancelled"])) &
        (df.get("origin", "auto") == "auto")
    )
    return bool(mask.sum() >= cap)


def autosignal_once(assets: List[str], interval: str, days: int = 120) -> None:
    """Generate at most one auto setup per run: pick best (highest confidence) candidate that passes all gates."""
    
    # Get UI configuration overrides
    ui_config = _get_ui_override_config()
    
    # Always use 4h for autosignal regardless of input interval
    autosignal_interval = "4h"
    if interval != autosignal_interval:
        print(f"[autosignal] Overriding interval {interval} → {autosignal_interval} (autosignal always uses 4h)")
    
    # Validate interval - only allow 1h+ timeframes to avoid noise
    allowed_intervals = ["1h", "4h", "1d"]
    if autosignal_interval not in allowed_intervals:
        print(f"[autosignal] Skip {autosignal_interval}: only {', '.join(allowed_intervals)} intervals allowed to avoid noise")
        return
    
    candidates = []

    # Build candidates per asset
    for sym in assets:
        prob_up, df = _train_and_score(sym, autosignal_interval, days=days)
        if prob_up is None or df is None or df.empty:
            print(f"[autosignal] Skip {sym}-{autosignal_interval}: no data/proba")
            continue
        direction, confidence = _choose_direction(prob_up)
        last_price = float(df["close"].iloc[-1])

        # OB gate (SIGNED imbalance) - use UI config if available
        ob_dir = None
        gate_ob = ui_config.get("gate_ob", GATE_OB)
        ob_edge_delta = ui_config.get("ob_edge_delta", OB_EDGE_DELTA)
        
        if gate_ob:
            try:
                ob = ob_features(sym, top=20)
                s_imb = float(ob.get("ob_imb_top20", float("nan")))  # signed −1..+1
                ob_dir = _ob_gate(s_imb, edge_delta=ob_edge_delta)
            except Exception as e:
                print(f"[autosignal] OB fetch error for {sym}: {e}")
                ob_dir = None

        candidates.append({
            "asset": sym,
            "interval": autosignal_interval,
            "prob_up": prob_up,
            "direction": direction,
            "confidence": confidence,
            "price": last_price,
            "df": df,
            "ob_dir": ob_dir
        })

    if not candidates:
        print(f"[autosignal] No candidates built at {_now_utc().isoformat()}")
        return

    # Filter candidates with detailed skip logs
    passing = []
    # Lazily construct Deribit provider for max pain
    maxpain_provider = None
    for c in candidates:
        sym = c["asset"]
        iv = c["interval"]
        conf = float(c["confidence"])
        dirn = c["direction"]

        # Get UI configuration overrides
        min_conf_arm = ui_config.get("min_conf_arm", MIN_CONF_ARM)
        gate_ob = ui_config.get("gate_ob", GATE_OB)
        ob_neutral_conf = ui_config.get("ob_neutral_conf", OB_NEUTRAL_CONF)
        gate_regime = ui_config.get("gate_regime", GATE_REGIME)
        regime_override_conf = ui_config.get("regime_override_conf", REGIME_OVERRIDE_CONF)
        gate_rr25 = ui_config.get("gate_rr25", GATE_RR25)
        rr25_thresh = ui_config.get("rr25_thresh", RR25_THRESH)
        max_setups_per_day = ui_config.get("max_setups_per_day", MAX_SETUPS_PER_DAY)

        # Adaptive confidence gate (replaces old min_conf_arm check)
        if ADAPTIVE_CONFIDENCE_AVAILABLE:
            try:
                confidence_result = adaptive_confidence_gate.evaluate_confidence(iv, conf)
                if not confidence_result.passed:
                    print(f"[autosignal] Skip {sym}-{iv}: adaptive confidence gate failed - confidence {conf:.3f} < threshold {confidence_result.effective_threshold:.3f}")
                    continue
                print(f"[autosignal] {sym}-{iv}: adaptive confidence gate passed - confidence {conf:.3f} >= threshold {confidence_result.effective_threshold:.3f}")
            except Exception as e:
                print(f"[autosignal] {sym}-{iv}: adaptive confidence gate error {e}, falling back to min_conf_arm")
                if conf < min_conf_arm:
                    print(f"[autosignal] Skip {sym}-{iv}: confidence {conf:.3f} < {min_conf_arm:.3f}")
                    continue
        else:
            # Fallback to old confidence check
            if conf < min_conf_arm:
                print(f"[autosignal] Skip {sym}-{iv}: confidence {conf:.3f} < {min_conf_arm:.3f}")
                continue

        # Sentiment weighting (direction-specific)
        sentiment_weight = 1.0  # Default weight
        try:
            from src.data.real_sentiment import get_current_sentiment, get_direction_specific_sentiment_weight
            sentiment_data = get_current_sentiment()
            if sentiment_data:
                sentiment_score = sentiment_data['sentiment_score']
                sentiment_weight = get_direction_specific_sentiment_weight(sentiment_score, dirn)
                
                # Apply sentiment weight to confidence
                adjusted_conf = conf * sentiment_weight
                print(f"[autosignal] {sym}-{iv}: {dirn.upper()} sentiment {sentiment_score:.3f} → weight {sentiment_weight:.2f} → adjusted conf {adjusted_conf:.3f}")
                
                # Use adjusted confidence for further checks
                conf = adjusted_conf
            else:
                print(f"[autosignal] {sym}-{iv}: sentiment data unavailable, using base confidence {conf:.3f}")
        except Exception as e:
            print(f"[autosignal] {sym}-{iv}: sentiment error {e}, using base confidence {conf:.3f}")

        # Enhanced Max Pain weighting (multi-factor analysis)
        maxpain_info = {}
        try:
            # Allow UI overrides
            maxpain_enable_weight = ui_config.get("maxpain_enable_weight", MAXPAIN_ENABLE_WEIGHT)
            maxpain_use_enhanced = ui_config.get("maxpain_use_enhanced", True)  # New: use enhanced calculation

            if maxpain_enable_weight:
                # Build provider once
                if maxpain_provider is None:
                    try:
                        from src.data.deribit_free import DeribitFreeProvider
                        maxpain_provider = DeribitFreeProvider()
                    except Exception as ie:
                        print(f"[autosignal] {sym}-{iv}: max pain provider init error: {ie}")
                
                if maxpain_provider is not None:
                    cur = "BTC" if "BTC" in sym else ("ETH" if "ETH" in sym else "BTC")
                    
                    if maxpain_use_enhanced:
                        # Use enhanced multi-factor analysis
                        enhanced_result = maxpain_provider.calculate_enhanced_max_pain_weight(cur, dirn)
                        if enhanced_result and enhanced_result.get("weight"):
                            maxpain_weight = enhanced_result["weight"]
                            adjusted_conf = conf * maxpain_weight
                            
                            # Enhanced logging with detailed analysis
                            factors = enhanced_result.get("factors", {})
                            print(f"[autosignal] {sym}-{iv}: Enhanced MaxPain {cur}")
                            print(f"  K*={enhanced_result.get('max_pain_strike', 0):.0f} dist={enhanced_result.get('distance_pct', 0):.2f}%")
                            print(f"  toward={enhanced_result.get('toward_direction', 'N/A').upper()} align={enhanced_result.get('direction_alignment', 0):.1f}")
                            print(f"  OI_conc={enhanced_result.get('oi_concentration', 0):.3f} PCR={enhanced_result.get('put_call_ratio', 0):.2f}")
                            print(f"  gamma={enhanced_result.get('gamma_exposure', 0):.3f} expiry_w={enhanced_result.get('avg_expiry_weight', 0):.2f}")
                            print(f"  weight={maxpain_weight:.3f} → adjusted conf {adjusted_conf:.3f}")
                            
                            conf = adjusted_conf
                            maxpain_info = {
                                "max_pain_currency": cur,
                                "max_pain_strike": enhanced_result.get("max_pain_strike"),
                                "max_pain_distance_pct": enhanced_result.get("distance_pct"),
                                "max_pain_toward": enhanced_result.get("toward_direction"),
                                "max_pain_weight": maxpain_weight,
                                "max_pain_oi_concentration": enhanced_result.get("oi_concentration"),
                                "max_pain_put_call_ratio": enhanced_result.get("put_call_ratio"),
                                "max_pain_gamma_exposure": enhanced_result.get("gamma_exposure"),
                                "max_pain_expiry_weight": enhanced_result.get("avg_expiry_weight"),
                                "max_pain_market_structure": enhanced_result.get("market_structure_score"),
                                "max_pain_enhanced": True,
                            }
                    else:
                        # Fallback to basic calculation
                        mp = maxpain_provider.calculate_max_pain(cur)
                        if mp and isinstance(mp, dict) and mp.get('max_pain_strike'):
                            S = float(mp.get('underlying_price', float('nan')))
                            Kp = float(mp.get('max_pain_strike', float('nan')))
                            if np.isfinite(S) and S > 0 and np.isfinite(Kp) and Kp > 0:
                                dist_pct = abs(S - Kp) / S * 100.0
                                toward_dir = "long" if S < Kp else "short"
                                # Directional weight: reward alignment toward max pain, penalize against
                                align = 1.0 if dirn == toward_dir else -1.0
                                scale = min(max(dist_pct / max(1e-6, float(maxpain_dist_ref_pct)), 0.0), 1.0)
                                maxpain_weight = 1.0 + align * (float(maxpain_weight_max) * scale)
                                # Clamp to sensible bounds
                                maxpain_weight = float(min(1.15, max(0.85, maxpain_weight)))
                                adjusted_conf = conf * maxpain_weight
                                print(f"[autosignal] {sym}-{iv}: Basic MaxPain {cur} K*={Kp:.0f} dist={dist_pct:.2f}% toward={toward_dir.upper()} → weight {maxpain_weight:.2f} → adjusted conf {adjusted_conf:.3f}")
                                conf = adjusted_conf
                                maxpain_info = {
                                    "max_pain_currency": cur,
                                    "max_pain_strike": Kp,
                                    "max_pain_distance_pct": dist_pct,
                                    "max_pain_toward": toward_dir,
                                    "max_pain_weight": maxpain_weight,
                                    "max_pain_enhanced": False,
                                }
        except Exception as e:
            print(f"[autosignal] {sym}-{iv}: max pain weighting error {e}")

        # OB direction consistency (must support direction)
        if gate_ob:
            if c["ob_dir"] is None:
                if conf >= ob_neutral_conf and ob_neutral_conf > 0.0:
                    print(f"[autosignal] OB undecided but conf {conf:.3f} >= {ob_neutral_conf:.3f}, treating as neutral")
                else:
                    print(f"[autosignal] Skip {sym}-{iv}: OB gate undecided (no edge hit)")
                    continue
            if c["ob_dir"] is not None and c["ob_dir"] != dirn:
                print(f"[autosignal] Skip {sym}-{iv}: OB gate wants {c['ob_dir']} vs model {dirn}")
                continue

        # Regime gate
        if gate_regime:
            reason = _regime_gate(c["df"], dirn)
            if reason:
                if conf >= regime_override_conf and regime_override_conf > 0.0:
                    print(f"[autosignal] Regime gate would block ({reason}), but conf {conf:.3f} >= {regime_override_conf:.3f}, overriding")
                else:
                    print(f"[autosignal] Skip {sym}-{iv}: {reason}")
                    continue

        # RR25 gate
        if gate_rr25:
            rr = _rr25_avg_for(sym)
            reason = _rr25_gate(dirn, rr, rr25_thresh)
            if reason:
                print(f"[autosignal] Skip {sym}-{iv}: {reason}")
                continue

        # Daily cap
        if _daily_cap_reached(sym, iv, max_setups_per_day):
            print(f"[autosignal] Skip {sym}-{iv}: daily cap {max_setups_per_day} reached")
            continue

        # Carry adjusted confidence and diagnostics forward for selection and logging
        try:
            c["confidence"] = float(conf)
            if maxpain_info:
                c["_maxpain_info"] = maxpain_info
            c["_sentiment_weight"] = float(sentiment_weight)
        except Exception:
            pass
        passing.append(c)

    if not passing:
        print(f"[autosignal] No candidates passed gates at {_now_utc().isoformat()}")
        print(f"[autosignal] Tried {len(candidates)} candidates; "
              f"max prob_up={max(c['prob_up'] for c in candidates):.3f}, "
              f"max conf={max(c['confidence'] for c in candidates):.3f}")
        return

    # Choose the highest confidence among those that passed
    best = max(passing, key=lambda x: x["confidence"])
    asset = best["asset"]
    direction = best["direction"]
    confidence = float(best["confidence"])
    maxpain_info = best.get("_maxpain_info", {})
    df = best["df"]
    price = float(best["price"])

        # Build setup levels with UI configuration overrides
    atr_val = _estimate_atr(df)
    
    # Get UI configuration for setup building
    k_entry = ui_config.get("k_entry", K_ENTRY_DEFAULT)
    k_stop = ui_config.get("k_stop", K_STOP_DEFAULT)
    valid_bars = ui_config.get("valid_bars", VALID_BARS_MIN)
    entry_buffer_bps = ui_config.get("entry_buffer_bps", ENTRY_BUFFER_BPS)
    min_rr = ui_config.get("min_rr", MIN_RR)
    
    # Apply bounds checking
    k_entry = max(k_entry, 0.10)
    valid_bars = max(valid_bars, VALID_BARS_MIN)
    
    # Get features for adaptive selector if available
    features = None
    try:
        from src.features.engine import FeatureEngine
        feature_engine = FeatureEngine()
        feature_df, feature_cols = feature_engine.build_feature_matrix(df, horizons_hours=24)
        if not feature_df.empty:
            features = feature_df.iloc[-1].to_dict()
    except Exception as e:
        print(f"[autosignal] Could not build features for adaptive selector: {e}")
    
    setup = build_autosetup_levels(
        direction=direction, last_price=price, atr=atr_val, rr=min_rr,
        interval=autosignal_interval, features=features, k_entry=k_entry
    )
    
    # Add adaptive expiry time
    base_now = pd.to_datetime(_now_local(), utc=True)
    try:
        base_now = base_now.tz_convert(MY_TZ)
    except Exception:
        pass
    
    # Compute adaptive validity based on timeframe, ATR, and macro regime
    if os.getenv("ADAPTIVE_VALIDITY", "1") == "1":
        try:
            from src.utils.validity import compute_adaptive_validity_bars, get_validity_until
            from src.trading.macro import get_macro_regime
            
            # Get macro regime for validity calculation
            regime_result = get_macro_regime(df)
            regime = regime_result.regime
            
            # Compute adaptive validity bars
            valid_bars = compute_adaptive_validity_bars(
                tf=autosignal_interval,
                R_used=atr_val,  # ATR used for this setup
                regime=regime,
                now_ts=base_now
            )
            
            # Compute valid_until timestamp
            valid_until = get_validity_until(base_now, valid_bars, autosignal_interval)
            
            validity_source = "adaptive"
            
            print(f"[autosignal] Adaptive validity: {valid_bars} bars (regime: {regime}, ATR: {atr_val:.2f})")
            
        except Exception as e:
            print(f"[autosignal] Adaptive validity failed: {e}, using fallback")
            # Fallback to legacy fixed bars
            valid_bars = ui_config.get("valid_bars", VALID_BARS_MIN)
            valid_bars = max(valid_bars, VALID_BARS_MIN)
            per_bar_min = {"5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}.get(autosignal_interval, 60)
            valid_until = base_now + pd.Timedelta(minutes=valid_bars * per_bar_min)
            validity_source = "fixed"
    else:
        # Legacy fixed bars fallback
        valid_bars = ui_config.get("valid_bars", VALID_BARS_MIN)
        valid_bars = max(valid_bars, VALID_BARS_MIN)
        per_bar_min = {"5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}.get(autosignal_interval, 60)
        valid_until = base_now + pd.Timedelta(minutes=valid_bars * per_bar_min)
        validity_source = "fixed"
    
    setup["expires_at"] = valid_until

    # Sizing with UI configuration overrides
    balance = ui_config.get("acct_balance", float(os.getenv("ACCOUNT_BALANCE_USD", "400")))
    max_lev = ui_config.get("max_leverage", int(os.getenv("MAX_LEVERAGE", "10")))
    risk_pct = ui_config.get("risk_per_trade_pct", float(os.getenv("RISK_PER_TRADE_PCT", "1.0")))
    
    nominal_position_pct = ui_config.get("nominal_position_pct", 25.0)
    size_units, notional, lev = _size_position(setup["entry"], setup["stop"], balance, max_lev, risk_pct, 
                                              autosignal_interval, nominal_position_pct)

    # Get sentiment data for setup
    sentiment_info = {}
    try:
        from src.data.real_sentiment import get_current_sentiment
        sentiment_data = get_current_sentiment()
        if sentiment_data:
            sentiment_info = {
                "sentiment_value": sentiment_data['value'],
                "sentiment_classification": sentiment_data['classification'],
                "sentiment_score": sentiment_data['sentiment_score'],
                "sentiment_weight": sentiment_weight
            }
    except Exception as e:
        print(f"[autosignal] Failed to get sentiment data: {e}")
    
    # Get macro regime information for logging
    macro_regime_info = {}
    try:
        from src.trading.macro import get_macro_regime
        regime_result = get_macro_regime(df)
        macro_regime_info = {
            "macro_regime_at_creation": regime_result.regime,
            "macro_regime_ma200": regime_result.ma200,
            "macro_regime_price_vs_ma200_pct": regime_result.price_vs_ma200_pct,
            "macro_regime_reason": regime_result.reason,
        }
    except Exception as e:
        print(f"[autosignal] Failed to get macro regime info: {e}")
        macro_regime_info = {
            "macro_regime_at_creation": "unknown",
            "macro_regime_ma200": None,
            "macro_regime_price_vs_ma200_pct": None,
            "macro_regime_reason": f"Error: {e}",
        }
    
    # Append to CSV
    row_id = f"AUTO-{asset}-{autosignal_interval}-{_now_utc().strftime('%Y%m%d_%H%M%S')}"
    unique_id = _generate_unique_id(asset, autosignal_interval, direction, "auto")
    row = {
        "id": row_id,
        "unique_id": unique_id,
        "asset": asset,
        "interval": autosignal_interval,
        "direction": direction,
        "entry": setup["entry"],
        "stop": setup["stop"],
        "target": setup["target"],
        "rr": setup["rr"],
        "size_units": size_units,
        "notional_usd": notional,
        "leverage": lev,
        "created_at": _now_local().isoformat() if _now_local() else pd.Timestamp.now(tz='Asia/Kuala_Lumpur').isoformat(),
        "expires_at": setup["expires_at"].isoformat(),
        "valid_bars": valid_bars,
        "validity_source": validity_source,
        "triggered_at": "",  # Will be set when setup is triggered
        "status": "pending",
        "confidence": confidence,
        # autosignal prefers touch; dashboard may use close-through
        "trigger_rule": ui_config.get("trigger_rule", TRIGGER_RULE_DEFAULT),
        "entry_buffer_bps": entry_buffer_bps,
        "origin": "auto",
        # Sentiment data
        "sentiment_value": sentiment_info.get("sentiment_value", 50),
        "sentiment_classification": sentiment_info.get("sentiment_classification", "Neutral"),
        "sentiment_score": sentiment_info.get("sentiment_score", 0.0),
        "sentiment_weight": sentiment_info.get("sentiment_weight", 1.0),
        # Max pain data
        "max_pain_currency": maxpain_info.get("max_pain_currency"),
        "max_pain_strike": maxpain_info.get("max_pain_strike"),
        "max_pain_distance_pct": maxpain_info.get("max_pain_distance_pct"),
        "max_pain_toward": maxpain_info.get("max_pain_toward"),
        "max_pain_weight": maxpain_info.get("max_pain_weight", 1.0),
        # Macro regime data
        "macro_regime_at_creation": macro_regime_info["macro_regime_at_creation"],
        "macro_regime_ma200": macro_regime_info["macro_regime_ma200"],
        "macro_regime_price_vs_ma200_pct": macro_regime_info["macro_regime_price_vs_ma200_pct"],
        "macro_regime_reason": macro_regime_info["macro_regime_reason"],
    }
    
    # Validate setup data before saving
    validation_errors = []
    if not row.get('id') or row['id'].strip() == '':
        validation_errors.append("Missing setup ID")
    if not row.get('unique_id') or row['unique_id'].strip() == '':
        validation_errors.append("Missing unique ID")
    if not row.get('created_at') or row['created_at'].strip() == '':
        validation_errors.append("Missing created_at timestamp")
    if not row.get('status') or row['status'] != 'pending':
        validation_errors.append(f"Invalid status: {row.get('status')} (should be 'pending')")
    if not row.get('entry') or float(row['entry']) <= 0:
        validation_errors.append(f"Invalid entry price: {row.get('entry')}")
    if not row.get('stop') or float(row['stop']) <= 0:
        validation_errors.append(f"Invalid stop price: {row.get('stop')}")
    if not row.get('target') or float(row['target']) <= 0:
        validation_errors.append(f"Invalid target price: {row.get('target')}")
    
    if validation_errors:
        error_msg = f"[autosignal] VALIDATION FAILED for {row_id}: " + "; ".join(validation_errors)
        print(error_msg)
        # Send alert about validation failure
        _send_telegram(f"🚨 SETUP VALIDATION FAILED\n{error_msg}\nSetup data: {row}")
        return False
    
    # Save setup with validation
    try:
        _append_setup_row(row)
        print(f"[autosignal] ✅ VALIDATED and appended {row_id}: {direction} {asset} {autosignal_interval} "
              f"entry={row['entry']:.2f} stop={row['stop']:.2f} target={row['target']:.2f} "
              f"rr={row['rr']:.2f} conf={row['confidence']:.3f} created_at={row['created_at']}")
    except Exception as e:
        error_msg = f"[autosignal] FAILED to save setup {row_id}: {e}"
        print(error_msg)
        _send_telegram(f"🚨 SETUP SAVE FAILED\n{error_msg}\nSetup data: {row}")
        return False

    # R:R invariant logging (decision time)
    if getenv_bool("RR_INVARIANT_LOGGING", True):
        try:
            # Get ATR as R_used
            R_used = _estimate_atr(df)
            
            # Calculate s_planned and t_planned from the setup (final R-multipliers post caps/optimizers)
            entry_price = float(row["entry"])
            stop_price = float(row["stop"])
            target_price = float(row["target"])
            
            if direction == "long":
                s_planned = abs(entry_price - stop_price) / R_used
                t_planned = abs(target_price - entry_price) / R_used
            else:  # short
                s_planned = abs(stop_price - entry_price) / R_used
                t_planned = abs(entry_price - target_price) / R_used
            
            # Compute and log invariants at decision time (entry_fill=None)
            inv = compute_rr_invariants(
                direction=direction,
                R_used=R_used,
                s_planned=s_planned,
                t_planned=t_planned,
                entry_planned=entry_price,
                entry_fill=None,                 # no fill yet
                live_entry=None, live_stop=None, live_tp=None,
                setup_id=unique_id, tf=autosignal_interval,
            )
            rr_invariant_writer.append(inv)     # sidecar write
            
            print(f"[autosignal] R:R invariants logged for {unique_id}: "
                  f"s_planned={s_planned:.2f}, t_planned={t_planned:.2f}, "
                  f"rr_planned={inv.get('rr_planned', 'N/A'):.2f}")
        except Exception as e:
            print(f"[autosignal] R:R invariant logging error: {e}")

    # Shadow stop logging (Phase-1, no behavior change)
    try:
        from src.trading.shadow_stops import compute_and_log_shadow_stop
        shadow_result = compute_and_log_shadow_stop(
            setup_id=row_id,
            tf=autosignal_interval,
            entry_price=float(row["entry"]),
            applied_stop_price=float(row["stop"]),
            rr_planned=float(row["rr"]),
            data=df,  # OHLCV data for ATR computation
            p_hit=None,  # Could be extracted from adaptive selector if available
            conf=confidence,
            outcome="pending"
        )
        print(f"[autosignal] shadow stop logged: valid={shadow_result.shadow_valid}, "
              f"dynamic_stop_R={shadow_result.dynamic_stop_candidate_R:.3f if shadow_result.dynamic_stop_candidate_R else 'N/A'}")
    except Exception as e:
        print(f"[autosignal] shadow stop logging error: {e}")

    # Telegram alert (optional)
    sentiment_text = ""
    if sentiment_info:
        sentiment_text = f"\nSentiment: {sentiment_info['sentiment_value']} ({sentiment_info['sentiment_classification']})\nWeight: {sentiment_info['sentiment_weight']:.2f}x"
    maxpain_text = ""
    if maxpain_info:
        try:
            mp_strike = float(maxpain_info.get('max_pain_strike', float('nan')))
            mp_dist = float(maxpain_info.get('max_pain_distance_pct', float('nan')))
            mp_toward = str(maxpain_info.get('max_pain_toward', '')).upper()
            mp_weight = float(maxpain_info.get('max_pain_weight', 1.0))
            if np.isfinite(mp_strike) and np.isfinite(mp_dist):
                maxpain_text = f"\nMaxPain: {mp_strike:.0f} ({mp_dist:.2f}%) toward {mp_toward}\nWeight: {mp_weight:.2f}x"
        except Exception:
            pass
    
    _send_telegram(
        text=(
            f"Auto setup {asset} {autosignal_interval} ({direction.upper()})\n"
            f"Setup ID: {unique_id}\n"
            f"Entry: {row['entry']:.2f}\n"
            f"Stop: {row['stop']:.2f}\n"
            f"Target: {row['target']:.2f}\n"
            f"RR: {row['rr']:.2f}\n"
            f"Confidence: {row['confidence']:.0%}{sentiment_text}{maxpain_text}\n"
            f"Size: {row['size_units']:.6f}  Notional: ${row['notional_usd']:.2f}  Lev: {row['leverage']:.1f}x\n"
            f"Created at: {row['created_at']}\n"
            f"Valid until: {row['expires_at']}"
        )
    )


def should_run_autosignal():
    """Check if we should run autosignal based on 4h candle timing."""
    now_utc = pd.Timestamp.now(tz='UTC')
    candle_hours = [0, 4, 8, 12, 16, 20]
    
    # Check if we're within 5 minutes of a 4h candle close
    for hour in candle_hours:
        candle_close = now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)
        if now_utc.hour == hour and now_utc.minute < 5:
            return True, f"4h candle close at {hour:02d}:00 UTC"
    
    return False, f"Not at 4h candle close (current: {now_utc.strftime('%H:%M UTC')})"


def main():
    # Check if we should run based on 4h candle timing
    should_run, reason = should_run_autosignal()
    if not should_run:
        print(f"[autosignal] ⏰ Skipping - {reason}")
        return
    
    print(f"[autosignal] 🎯 Running at {reason}")
    
    # Default lookback days for autosignal training
    days = int(os.getenv("ALPHA12_AUTOSIGNAL_LOOKBACK_DAYS", "120"))
    try:
        autosignal_once(ASSETS, INTERVAL, days=days)
    except Exception as e:
        # Best effort: don't crash the timer permanently
        msg = f"[autosignal] error: {e}"
        print(msg)


if __name__ == "__main__":
    main()
