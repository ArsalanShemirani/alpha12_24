#!/usr/bin/env python3
"""
Streamlit dashboard for alpha12_24
"""

# Load environment variables FIRST, before any other imports
import os
from pathlib import Path

def load_env_vars():
    """Load environment variables from .env file if it exists"""
    try:
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            # Remove inline comments (everything after #)
                            if '#' in value:
                                value = value.split('#')[0].strip()
                            # Expand $(pwd) if present
                            if '$(pwd)' in value:
                                import subprocess
                                pwd_result = subprocess.run(['pwd'], capture_output=True, text=True)
                                if pwd_result.returncode == 0:
                                    value = value.replace('$(pwd)', pwd_result.stdout.strip())
                            os.environ[key.strip()] = value.strip()
    except Exception:
        pass

# Load environment variables at startup
load_env_vars()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import warnings
import random
import math
warnings.filterwarnings('ignore')

# --- Default Telegram credentials (hardcoded) ---
DEFAULT_TG_BOT_TOKEN = "8119234697:AAE7dGn707CEXZo0hyzSHpzCkQtIklEhDkE"
DEFAULT_TG_CHAT_ID = "-4873132631"

# --- Autorefresh import ---
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None



# --- Local timezone: Malaysia ---
MY_TZ = "Asia/Kuala_Lumpur"

def _to_my_tz_index(idx):
    try:
        idx = pd.to_datetime(idx)
        if getattr(idx, 'tz', None) is None:
            idx = idx.tz_localize('UTC')
        return idx.tz_convert(MY_TZ)
    except Exception:
        return idx

def _to_my_tz_ts(ts):
    try:
        ts = pd.to_datetime(ts)
        if getattr(ts, 'tz', None) is None:
            ts = ts.tz_localize('UTC')
        return ts.tz_convert(MY_TZ)
    except Exception:
        return pd.to_datetime(ts, errors='coerce')

# --- Prompt 4: extra imports ---
import json, time
from pathlib import Path
from typing import Union, Optional
import joblib
import requests
import re

# --- Backtest helpers ---
def _ensure_two_col_proba(p):
    """Return ndarray shape (n,2) as [P0, P1] regardless of input shape."""
    import numpy as _np
    p = _np.asarray(p)
    if p.ndim == 1:
        p1 = _np.clip(p, 0.0, 1.0)
        return _np.stack([1.0 - p1, p1], axis=1)
    if p.ndim == 2 and p.shape[1] == 1:
        p1 = _np.clip(p[:, 0], 0.0, 1.0)
        return _np.stack([1.0 - p1, p1], axis=1)
    if p.ndim == 2 and p.shape[1] >= 2:
        # assume col 1 is class 1 if binary; else take max as class 1
        if p.shape[1] == 2:
            return _np.clip(p, 0.0, 1.0)
        m = p.max(axis=1)
        return _np.stack([1.0 - m, m], axis=1)
    # degenerate
    return _np.zeros((len(p) if hasattr(p, "__len__") else 1, 2), dtype=float)

def _series_aligned(df_like, idx, col, fallback=0.0):
    """Return a Series aligned to idx, filled with fallback."""
    import pandas as _pd, numpy as _np
    if isinstance(df_like, _pd.DataFrame) and col in df_like.columns:
        s = _pd.to_numeric(df_like.loc[idx, col], errors="coerce")
        return s.fillna(fallback)
    if isinstance(df_like, _pd.Series):
        s = _pd.to_numeric(df_like.reindex(idx), errors="coerce")
        return s.fillna(fallback)
    return _pd.Series([fallback] * len(idx), index=idx, dtype=float)

# Import alpha12_24 modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.config import config
from src.features.engine import FeatureEngine
from src.features.macro import MacroFeatures
from src.models.train import ModelTrainer
from src.policy.thresholds import ThresholdManager
from src.policy.regime import RegimeDetector
from src.trading.planner import TradingPlanner
from src.trading.leverage import LeverageManager
from src.trading.logger import TradingLogger
from src.backtest.runner import BacktestRunner
from src.eval.score_signals import SignalScorer
from src.eval.optimizer import HyperparameterOptimizer
from src.eval.live_metrics import load_trade_history, compute_metrics, calibration_bins


# --- helpers for bar limits and safe defaults ---
def _bars_per_day(interval: str) -> int:
    mapping = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
    return mapping.get(interval, 288)

def _calc_limit(interval: str, days: int) -> int:
    # Give higher cap for slower intervals so training has enough samples
    base = _bars_per_day(interval) * days
    hard_cap = 1500 if interval in ("5m", "15m") else 2500  # was 1000 before
    return min(base, hard_cap)

EMPTY_MODEL_SUMMARY = {
    "model_type": "n/a",
    "n_features": 0,
    "cv_accuracy_mean": 0.0,
    "cv_precision_mean": 0.0,
    "cv_recall_mean": 0.0,
    "cv_f1_mean": 0.0,
    "top_features": [],
}

# --- Setups CSV canonical schema (keep stable) ---
SETUP_FIELDS = [
    "id","asset","interval","direction","entry","stop","target","rr",
    "size_units","notional_usd","leverage",
    "created_at","expires_at","status","confidence","trigger_rule","entry_buffer_bps",
    "origin"
]

# --- Prompt 4 inline utilities (persistence, logging, confidence, alerts) ---
def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_artifacts_inline(model, meta: dict, model_dir: str = "artifacts") -> str:
    d = _ensure_dir(model_dir) / _now_tag()
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib")
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return str(d)

def load_latest_inline(model_dir: str = "artifacts"):
    root = Path(model_dir)
    if not root.exists():
        return None, None
    candidates = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    for d in candidates:
        try:
            m = joblib.load(d / "model.joblib")
            meta = json.loads((d / "meta.json").read_text())
            return m, meta
        except Exception:
            continue
    return None, None

def load_latest_with_feature_check(model_dir: str = "artifacts", feature_cols: list = None):
    """Load latest model with feature compatibility check"""
    m_loaded, meta = load_latest_inline(model_dir)
    if m_loaded is not None and feature_cols is not None:
        want = meta.get("feature_cols", [])
        have = list(feature_cols)
        if want and want != have:
            st.error("Feature mismatch vs saved model. Retrain or align features.")
            return None, meta
    return m_loaded, meta

def append_csv_row(path: str, row: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    import csv
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def confidence_badge(prob: float) -> str:
    try:
        p = float(prob)
    except Exception:
        return "LOW"
    if p >= 0.66:
        return "HIGH"
    if p >= 0.55:
        return "MEDIUM"
    return "LOW"

def post_webhook_inline(url: str, payload: dict, timeout: int = 8) -> bool:
    if not url:
        return False
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True
    except Exception:
        return False


# Telegram helper (MarkdownV2-safe)
def tg_escape_md2(text: str) -> str:
    """Escape text for Telegram MarkdownV2."""
    if text is None:
        return ""
    # Escape all special chars per Telegram MarkdownV2
    return re.sub(r'([_*.\[\]()~`>#+\-=|{}!.])', r'\\\\\\1', str(text))


def send_telegram(bot_token: str, chat_id: str, text: str, timeout=8):
    if not bot_token or not chat_id:
        try:
            import streamlit as st
            st.session_state['tg_last_error'] = 'Missing bot token or chat id'
        except Exception:
            pass
        return False
    try:
        safe_text = tg_escape_md2(text)
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": safe_text, "parse_mode": "MarkdownV2"},
            timeout=timeout,
        )
        r.raise_for_status()
        try:
            import streamlit as st
            st.session_state['tg_last_error'] = ''
        except Exception:
            pass
        return True
    except Exception as e:
        # Try to capture Telegram error description for the UI
        desc = None
        try:
            desc = r.json().get('description')  # type: ignore[name-defined]
        except Exception:
            desc = str(e)
        try:
            import streamlit as st
            st.session_state['tg_last_error'] = desc
        except Exception:
            pass
        return False


# Telegram diagnostics
def tg_get_me(bot_token: str, timeout=8):
    if not bot_token:
        return False, {"error": "Empty token"}
    try:
        r = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=timeout)
        j = r.json()
        return bool(j.get("ok")), j
    except Exception as e:
        return False, {"error": str(e)}


def tg_get_chat(bot_token: str, chat_id: str, timeout=8):
    if not bot_token or not chat_id:
        return False, {"error": "Missing token or chat_id"}
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getChat",
            params={"chat_id": chat_id},
            timeout=timeout,
        )
        j = r.json()
        return bool(j.get("ok")), j
    except Exception as e:
        return False, {"error": str(e)}


def resolve_loader(source_choice: str):
    if source_choice.startswith("Composite"):
        from src.data.composite import assemble_spot_plus_bybit as load_df
    elif source_choice.startswith("Binance"):
        from src.data.binance_free import assemble as load_df
    else:
        from src.data.binance_free import assemble as load_df  # Default to spot
    return load_df

# --- Setup → Trigger helpers (pending orders with anti stop-hunt) ---

def _setup_now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def _setups_csv_path():
    return os.path.join(getattr(config, 'runs_dir', 'runs'), 'setups.csv')

def _append_setup_row(row: dict):
    import csv
    p = _setups_csv_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    write_header = not os.path.exists(p)
    # keep only known fields and in canonical order
    safe_row = {k: row.get(k, "") for k in SETUP_FIELDS}
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=SETUP_FIELDS,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL
        )
        if write_header:
            w.writeheader()
        w.writerow(safe_row)

def _load_setups_df():
    p = _setups_csv_path()
    if not os.path.exists(p):
        return pd.DataFrame(columns=SETUP_FIELDS)
    try:
        # Be tolerant to historical rows with mismatched columns
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
    except Exception:
        # Fallback: read line-by-line into DictReader to salvage what we can
        import csv
        rows = []
        with open(p, "r", newline="") as f:
            r = csv.DictReader(f)
            for rrow in r:
                rows.append(rrow)
        df = pd.DataFrame(rows)
    # Normalize columns to canonical schema
    for c in SETUP_FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    # drop any unexpected columns to avoid downstream issues
    df = df[SETUP_FIELDS].copy()
    # Parse datetimes (ingest as UTC, display/operate in Malaysia time)
    for c in ("created_at", "expires_at"):
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            try:
                df[c] = ts.dt.tz_convert(MY_TZ)
            except Exception:
                df[c] = ts
    # (Optional) Write back a repaired CSV so future reads are clean
    try:
        df.to_csv(p, index=False)
    except Exception:
        pass
    return df

def _save_setups_df(df: pd.DataFrame):
    p = _setups_csv_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)

def _mk_setup_id(asset, interval):
    return f"{asset}-{interval}-{_setup_now_tag()}"

def _estimate_atr(df: pd.DataFrame, n: int = 14) -> float:
    # Use existing ATR if present, else quick estimator
    if "atr" in df.columns and not df["atr"].dropna().empty:
        return float(df["atr"].dropna().iloc[-1])
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = (high - low).abs()
    tr = np.maximum(tr, (high - prev_close).abs())
    tr = np.maximum(tr, (low - prev_close).abs())
    atr = pd.Series(tr).rolling(n, min_periods=n).mean()
    val = atr.dropna()
    return float(val.iloc[-1]) if not val.empty else float((close.iloc[-1] * 0.002))  # fallback ~20bps of price

def _build_setup(direction: str, price: float, atr: float, rr: float,
                 k_entry: float, k_stop: float, valid_bars: int,
                 now_ts, bar_interval: str, entry_buffer_bps: float) -> Optional[dict]:
    # Base entry from ATR offset
    if direction == "long":
        base_entry = price - k_entry * atr
    elif direction == "short":
        base_entry = price + k_entry * atr
    else:
        return None
    # Anti stop-hunt entry adjustment: n bps in the direction of intended fill
    # For long, make entry slightly deeper (lower); for short, slightly higher.
    entry = base_entry * (1.0 - entry_buffer_bps/10000.0) if direction == "long" else base_entry * (1.0 + entry_buffer_bps/10000.0)
    stop  = entry - k_stop * atr if direction == "long" else entry + k_stop * atr
    target = entry + rr * (entry - stop) if direction == "long" else entry - rr * (stop - entry)
    per_bar_min = {"5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}.get(bar_interval, 5)
    expires_at = pd.to_datetime(now_ts) + pd.Timedelta(minutes=valid_bars * per_bar_min)
    return {
        "entry": float(entry), "stop": float(stop), "target": float(target),
        "rr": float(rr), "expires_at": expires_at
    }

def _check_trigger(setup_row: dict, latest_bar: dict, trigger_rule: str = "touch", buffer_bps: float = 5.0) -> bool:
    """
    Return True if setup should trigger on the latest bar.

    For pullback-style limit entries we use:
      - LONG (entry below current): trigger on **low <= entry**
      - SHORT (entry above current): trigger on **high >= entry**

    If trigger_rule == 'close-through', require an additional confirmation close:
      - LONG: close >= entry * (1 + buffer)
      - SHORT: close <= entry * (1 - buffer)
    """
    entry = float(setup_row["entry"])
    close = float(latest_bar["close"])

    if setup_row["direction"] == "long":
        touched = float(latest_bar["low"]) <= entry
        if trigger_rule == "touch":
            return touched
        # confirm: close back above entry by buffer to avoid knife-catch
        return touched and close >= entry * (1.0 + buffer_bps/10000.0)

    if setup_row["direction"] == "short":
        touched = float(latest_bar["high"]) >= entry
        if trigger_rule == "touch":
            return touched
        # confirm: close back below entry by buffer to avoid wick fills
        return touched and close <= entry * (1.0 - buffer_bps/10000.0)

    return False


def main():
    """Main dashboard application"""
    # Set determinism
    random.seed(42)
    np.random.seed(42)

    # --- Auth gate ---
    try:
        from src.dashboard.auth import login_gate, render_logout_sidebar
        if not login_gate():
            return
    except Exception as _auth_e:
        # If auth module fails, default to open (but surface a warning)
        st.warning(f"Auth module error: {_auth_e}. Dashboard left unprotected for this session.")

    st.set_page_config(
        page_title="Alpha12_24 Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 Alpha12_24 Trading Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Data source selection
        source_choice = st.selectbox(
            "Data Source",
            ["Composite (Binance Spot + Bybit derivs)", "Binance only", "Bybit only"],
            index=0,
        )
        st.session_state["source_choice"] = source_choice

        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto refresh every 1 min", value=True)
        if auto_refresh and st_autorefresh:
            st_autorefresh(interval=60_000, key="alpha_autorefresh")

        # Asset selection
        asset = st.selectbox(
            "Select Asset",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"],
            index=0
        )

        # Time interval
        interval = st.selectbox(
            "Time Interval",
            ["5m", "15m", "1h", "4h", "1d"],
            index=2
        )

        # Data period
        days = st.slider("Data Period (Days)", 30, 365, 90)

        # Model selection
        # Check for XGBoost availability
        try:
            from src.models.train import XGBOOST_AVAILABLE
            available_models = ["rf", "logistic"]
            if XGBOOST_AVAILABLE:
                available_models.append("xgb")
        except Exception:
            available_models = ["rf", "logistic"]
            
        model_type = st.selectbox(
            "Model Type",
            available_models,
            index=0
        )
        # --- Prompt 4: calibration & alerts sidebar controls ---
        # Model-level probability calibration (handled inside ModelTrainer.train_model)
        calibrate_probs = st.checkbox("Calibrate probabilities (model-level)", value=True,
                                      help="Wraps the trained estimator with CalibratedClassifierCV. Avoids double calibration.")
        alerts_enabled = st.toggle("Enable alerts", value=True)
        webhook_url = st.text_input("Webhook URL (optional)", value="")
        st.session_state["calibrate_probs"] = calibrate_probs
        st.session_state["alerts_enabled"] = alerts_enabled
        st.session_state["webhook_url"] = webhook_url

        # Setup → Trigger parameters
        st.markdown("---")
        st.caption("Setup → Trigger (limit-style) parameters")
        k_entry = st.slider("Entry offset (ATR)", 0.1, 2.0, 0.5, 0.1)
        k_stop  = st.slider("Stop distance (ATR)", 0.5, 3.0, 1.0, 0.1)
        valid_bars = st.slider("Setup validity (bars)", 6, 288, 24, 1)
        entry_buffer_bps = st.number_input("Entry anti stop-hunt buffer (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, help="Shift entry slightly deeper (long) or higher (short) to avoid wick fills")
        confirm_on_close = st.toggle("Confirm on close (anti stop-hunt)", value=True, help="Require bar close beyond entry by buffer before triggering")
        auto_arm = st.toggle("Auto-arm & monitor setups", value=True)
        # --- Insert auto-cancel controls after setup controls ---
        cancel_on_flip = st.toggle("Auto-cancel on trend flip", value=True, help="Cancel pending setups if the latest model signal flips direction")
        min_conf_keep = st.slider("Min confidence to keep setup", 0.00, 0.90, 0.55, 0.01, help="Cancel pending setups if confidence drops below this")
        st.session_state.update(dict(
            k_entry=k_entry, k_stop=k_stop, valid_bars=valid_bars,
            entry_buffer_bps=entry_buffer_bps, confirm_on_close=confirm_on_close,
            auto_arm=auto_arm,
            cancel_on_flip=cancel_on_flip,
            min_conf_keep=min_conf_keep,
        ))

        # --- Account & Leverage controls ---
        st.markdown("---")
        st.caption("Account & Leverage")
        acct_balance = st.number_input("Account balance (USD)", min_value=50.0, max_value=1_000_000.0, value=400.0, step=50.0)
        max_leverage = st.slider("Max leverage", 1, 10, 10)
        st.session_state.update(dict(acct_balance=acct_balance, max_leverage=max_leverage))

        # --- Daemon status panel ---
        from pathlib import Path as _Path
        hb_path = _Path(getattr(config, 'runs_dir', 'runs')) / "daemon_heartbeat.txt"
        st.markdown("---")
        st.caption("🟢 Live Tracker (daemon)")
        if hb_path.exists():
            try:
                hb = pd.to_datetime(hb_path.read_text())
                st.success(f"Tracker heartbeat: {hb.tz_localize('UTC').tz_convert('Asia/Kuala_Lumpur')}")
            except Exception:
                st.success("Tracker heartbeat: detected")
        else:
            st.warning("Tracker not detected. Start it with:  \n`PYTHONPATH=$(pwd) nohup python -m src.daemon.tracker >/tmp/alpha_tracker.log 2>&1 &`")

        # Hard Gating & Frequency
        st.markdown("---")
        st.caption("Hard Gating & Frequency")
        # Get default from environment variable
        default_max_setups = int(os.getenv("MAX_SETUPS_PER_DAY", "0"))
        max_setups_per_day = st.slider("Max setups per 24h", 0, 10, default_max_setups)
        gate_regime = st.toggle("Gate by Macro Regime (MA50 vs MA200)", value=True)
        gate_rr25   = st.toggle("Gate by Deribit RR25 (nearest 2 expiries)", value=True)
        gate_ob     = st.toggle("Gate by OB Imbalance (top20)", value=True)
        rr25_thresh = st.number_input("RR25 abs threshold", min_value=0.0, max_value=0.10, value=0.00, step=0.01, help="Use 0.00 for zero-cross; 0.02 for stronger bias")
        ob_edge_pct = st.slider(
            "OB Imbalance Δ from edge (%)",
            min_value=0, max_value=50, value=20, step=1,
            help="Delta from the edges 0 or 1. Example: 20% ⇒ SHORT if raw ≤ 0.20, LONG if raw ≥ 0.80. Internally requires |signed| ≥ 1 − 2·Δ, where Δ = (percent/100)."
        )
        ob_edge_delta = ob_edge_pct / 100.0
        # Convert edge-delta (distance from 0 or 1) to a signed threshold (−1..+1)
        # Pass condition: |signed_imbalance| ≥ (1 − 2*edge_delta)
        ob_signed_thr = float(max(0.0, min(1.0, 1.0 - 2.0 * ob_edge_delta)))
        min_conf_arm  = st.number_input("Min confidence to arm", min_value=0.50, max_value=0.90, value=0.58, step=0.01)
        # Save UI settings to session state
        ui_settings = dict(
            max_setups_per_day=max_setups_per_day,
            gate_regime=gate_regime, gate_rr25=gate_rr25, gate_ob=gate_ob,
            rr25_thresh=rr25_thresh,
            ob_edge_delta=ob_edge_delta, ob_signed_thr=ob_signed_thr,
            min_conf_arm=min_conf_arm
        )
        st.session_state.update(ui_settings)
        
        # Save UI settings to file for autosignal to use
        try:
            from src.core.ui_config import save_ui_config
            save_ui_config(ui_settings)
        except Exception as e:
            st.warning(f"Could not save UI config for autosignal: {e}")

        # Live OB imbalance snapshot + hint
        try:
            from src.data.orderbook_free import ob_features as _obf
            _ob = _obf(symbol=asset, top=20)
            if _ob and "ob_imb_top20" in _ob:
                _s = float(_ob["ob_imb_top20"])  # signed −1..+1
                _r = 0.5 + 0.5 * _s              # raw 0..1
                _d = abs(_r - 0.5)               # delta from neutral
                _hint = "BUY-side pressure (bias LONG)" if _s >= 0 else "SELL-side pressure (bias SHORT)"
                _edge = float(st.session_state.get("ob_edge_delta", 0.20))
                _need = float(st.session_state.get("ob_signed_thr", 1.0 - 2.0 * _edge))
                _edge_pct = int(round(_edge * 100))
                st.caption(
                    f"OB imbalance (top20): signed **{_s:+.3f}** • raw **{_r:.3f}** • Δ from neutral **{_d:.3f}** • {_hint}.  "
                    f"Edge Δ=**{_edge_pct}%** ⇒ pass if raw ≤ **{_edge:.2f}** (SHORT) or ≥ **{1-_edge:.2f}** (LONG). "
                    f"Gate needs |signed| ≥ **{_need:.3f}**."
                )
            else:
                st.caption("OB imbalance unavailable (neutral).")
        except Exception:
            st.caption("OB imbalance check failed (neutral).")

        # Telegram Alerts
        st.caption("Telegram Alerts")
        tg_bot = st.text_input("TG Bot Token", value=DEFAULT_TG_BOT_TOKEN)
        tg_chat = st.text_input("TG Chat ID", value=DEFAULT_TG_CHAT_ID)
        st.session_state.update(dict(tg_bot=tg_bot, tg_chat=tg_chat))
        # Surface Telegram send errors (if any)
        if st.session_state.get('tg_last_error'):
            st.warning(f"Telegram last error: {st.session_state['tg_last_error']}")

        # Run analysis button
        run_analysis = st.button("🔄 Run Analysis", type="primary")

        # Auth controls
        try:
            render_logout_sidebar()
        except Exception:
            pass

    # Main content
    if run_analysis:
        with st.spinner("🔄 Running analysis..."):
            run_dashboard_analysis(asset, interval, days, model_type)
    else:
        # Show tabs immediately with placeholders so the UI is fully visible before analysis
        display_analysis_results(
            data=pd.DataFrame(),
            feature_df=pd.DataFrame(),
            signals=[],
            model_summary=EMPTY_MODEL_SUMMARY,
            asset=asset,
            interval=interval,
            source_choice=st.session_state.get("source_choice", "Composite (Binance Spot + Bybit derivs)")
        )


def show_welcome_page():
    """Show welcome page with instructions"""
    st.header("📊 Welcome to Alpha12_24")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 What is Alpha12_24?")
        st.write("""
        Alpha12_24 is an advanced cryptocurrency trading system that combines:
        - **Machine Learning Models**: Random Forest, Logistic Regression
        - **Technical Analysis**: RSI, MACD, Bollinger Bands, and more
        - **Market Microstructure**: Order flow, volatility, regime detection
        - **Risk Management**: Dynamic position sizing, leverage optimization
        - **Real-time Data**: Spot OHLCV + optional order-book features; Deribit RR25 (BTC/ETH) via public API. No synthetic derivatives; if an endpoint is unavailable, features are skipped.
        """)

    with col2:
        st.subheader("🚀 Key Features")
        st.write("""
        - **Multi-timeframe Analysis**: 5m to 1d intervals
        - **Walk-forward Backtesting**: Robust out-of-sample testing
        - **Regime Detection**: Automatic market state identification
        - **Signal Scoring**: Quality assessment of trading signals
        - **Hyperparameter Optimization**: Automated parameter tuning
        - **Real-time Monitoring**: Live signal generation and tracking
        - **Spot-only Data**: No synthetic derivatives, options & OI excluded until paid API
        """)

    st.subheader("📋 How to Use")
    st.write("""
    1. **Configure Parameters**: Select data source, asset, interval, and model in the sidebar
    2. **Run Analysis**: Click the "Run Analysis" button to start
    3. **Review Results**: Explore performance metrics, signals, and visualizations
    4. **Backtest**: Use the backtest tab to validate strategies
    5. **Optimize**: Use the optimization tools to improve performance
    """)


def run_dashboard_analysis(asset: str, interval: str, days: int, model_type: str):
    # Pull sidebar selections from session (added in main) or default to Composite
    source_choice = st.session_state.get("source_choice", "Composite (Binance Spot + Bybit derivs)")

    # Initialize components
    feature_engine = FeatureEngine()
    macro_features = MacroFeatures()
    model_trainer = ModelTrainer(config)
    threshold_manager = ThresholdManager(config)

    # Placeholders we will fill progressively
    data = pd.DataFrame()
    feature_df = pd.DataFrame()
    feature_cols = []
    model = None
    model_summary = EMPTY_MODEL_SUMMARY.copy()
    signals = []
    predictor = None

    # 1) Data
    st.subheader("📊 Data Loading")
    try:
        _load_df = resolve_loader(source_choice)
        limit = _calc_limit(interval, days)
        data = _load_df(asset, interval, limit)
        if data.empty:
            st.error("❌ Failed to fetch data (empty). Check your source/interval/days.")
        else:
            # Convert incoming timestamps (usually UTC) to Malaysia local time
            try:
                data.index = _to_my_tz_index(data.index)
            except Exception:
                pass
            st.success(f"✅ Loaded {len(data)} data points for {asset}")
            st.info(f"Data source: **{source_choice}** (Spot OHLCV only) • Timezone: **{MY_TZ}**")
    except Exception as e:
        st.error(f"❌ Data load error: {e}")

    # 2) Features
    st.subheader("🔧 Feature Engineering")
    try:
        if data.empty:
            st.warning("No data → skipping feature build.")
        else:
            feature_df, feature_cols = feature_engine.build_feature_matrix(data, config.horizons_hours, symbol=asset)
            # --- Feature hardening already done in engine, just report stats ---
            if feature_df is not None and len(feature_cols) > 0:
                try:
                    # Report usable rows after engine hardening
                    usable_rows = int((~feature_df[feature_cols].isna().any(axis=1)).sum())
                    st.caption(f"Usable rows after hardening: {usable_rows} (of {len(feature_df)})")
                except Exception as _hard_e:
                    st.warning(f"Feature stats skipped: {_hard_e}")
            feature_df = macro_features.calculate_macro_features(feature_df)
            st.success(f"✅ Built {len(feature_cols)} features")

            # --- Optional Deribit RR25 snapshots (read-only) ---
            try:
                from pathlib import Path
                runs = Path(getattr(config, 'runs_dir', 'runs'))
                for cur in ("BTC","ETH"):
                    p = runs / f"deribit_rr25_latest_{cur}.json"
                    if p.exists():
                        import json
                        rr = json.loads(p.read_text()).get("rows", [])
                        if rr:
                            last = rr[-1]
                            st.info(f"Deribit {cur} RR25: {last.get('rr25'):.4f}  (C25={last.get('iv_call25'):.4f}, P25={last.get('iv_put25'):.4f})  @ {last.get('updated_at')}")
            except Exception:
                pass
    except Exception as e:
        st.error(f"❌ Feature build error: {e}")
        feature_df, feature_cols = pd.DataFrame(), []

    # 3) Model training
    st.subheader("🤖 Model Training")
    try:
        if feature_df.empty or not feature_cols:
            st.error("Insufficient data for training (no features).")
        else:
            target_col = f'target_{config.horizons_hours[0]}h'
            if target_col not in feature_df.columns:
                st.error(f"❌ Target column {target_col} not found")
            else:
                X = feature_df[feature_cols]
                y = feature_df[target_col]
                valid_idx = ~(X.isna().any(axis=1) | y.isna())
                X, y = X[valid_idx], y[valid_idx]

                # Require fewer rows during early setup; raise later once data flows 24/7
                min_rows = 300 if interval in ("5m","15m") else 150
                st.caption(f"Training candidates: {len(X)} rows • Features: {len(feature_cols)}")
                if len(X) < min_rows:
                    st.error(f"Insufficient data for training (got {len(X)}, need ≥ {min_rows}). "
                             f"Try increasing 'Data Period (Days)' or using a faster interval.")
                else:
                    model = model_trainer.train_model(
                        X, y, model_type,
                        calibrate=bool(st.session_state.get("calibrate_probs", True)),
                        calib_cv=3,
                    )
                    # Predictor is always the trained model (possibly calibrated)
                    predictor = getattr(model, 'model', model)

                    # Show save/load controls
                    with st.container():
                        colA, colB = st.columns(2)
                        with colA:
                            if st.button("💾 Save model", use_container_width=True):
                                meta = {
                                    "asset": asset, 
                                    "interval": interval, 
                                    "model_type": getattr(model, 'model_type', model_type),
                                    "feature_cols": feature_cols
                                }
                                path = save_artifacts_inline(predictor, meta, model_dir=getattr(config, 'model_dir', 'artifacts'))
                                st.success(f"Saved to {path}")
                        with colB:
                            if st.button("📥 Load latest", use_container_width=True):
                                m_loaded, meta = load_latest_inline(getattr(config, 'model_dir', 'artifacts'))
                                if m_loaded is None:
                                    st.error("No saved model found.")
                                else:
                                    predictor = m_loaded
                                    st.success(f"Loaded model: {meta}")
                    model_summary = model_trainer.get_model_summary(model)
                    st.success(f"✅ Model trained: {model.model_type}")
                    st.json(model_summary)
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")

    # 4) Signals (best-effort)
    st.subheader("🔮 Signal Generation")
    try:
        if (model is None) or feature_df.empty or not feature_cols:
            st.warning("Skipped: model or features unavailable.")
            # Extra diagnostics: how many clean recent rows exist?
            try:
                if not feature_df.empty and feature_cols:
                    recent_probe = feature_df.tail(200)[feature_cols]
                    clean = int((~recent_probe.isna().any(axis=1)).sum())
                    st.caption(f"Recent clean rows available for inference: {clean}")
            except Exception:
                pass
            st.write(f"Debug: model={model is not None}, feature_df.empty={feature_df.empty}, feature_cols={len(feature_cols) if feature_cols else 0}")
        else:
            recent_data = feature_df.tail(50).copy()
            X_recent = recent_data[feature_cols]
            # Drop any row with NaNs to prevent shape errors
            mask = ~X_recent.isna().any(axis=1)
            X_recent = X_recent[mask]
            recent_data = recent_data.loc[X_recent.index]

            if len(X_recent) == 0:
                st.warning("No clean rows available for prediction.")
            else:
                st.write(f"Debug: Generating signals for {len(X_recent)} clean rows")
                # Use the model-level (possibly calibrated) predictor
                if predictor is not None and hasattr(predictor, 'predict_proba'):
                    probabilities = predictor.predict_proba(X_recent)
                else:
                    predictions, probabilities = model_trainer.predict(model, X_recent)
                signals = []
                for i, (idx, row) in enumerate(recent_data.iterrows()):
                    prob_up = float(probabilities[i, 1]) if probabilities.ndim == 2 else float(probabilities[i])
                    prob_down = 1.0 - prob_up
                    vol = float(row.get('volatility_24h', 0.02))
                    sig, conf, meta = threshold_manager.determine_signal(probabilities[i:i+1], vol)
                    # Convert signal to lowercase and map HOLD to flat
                    signal_lower = sig.lower() if sig != "HOLD" else "flat"
                    signals.append({
                        'timestamp': idx, 'signal': signal_lower, 'confidence': conf,
                        'prob_up': prob_up, 'prob_down': prob_down,
                        'price': float(row.get('close', np.nan)),
                        'volatility': vol
                    })
                st.success(f"✅ Generated {len(signals)} signals")
                # Cache the most recent signal for monitoring logic (auto-cancel on flip / conf drop)
                try:
                    last_sig = signals[-1]
                    st.session_state["last_signal_direction"] = last_sig.get("signal", "flat")
                    st.session_state["last_signal_confidence"] = float(last_sig.get("confidence", 0.0))
                    st.session_state["last_signal_ts"] = str(last_sig.get("timestamp", ""))
                except Exception:
                    pass
                # --- Prompt 4: logging and optional alert ---
                try:
                    if signals:
                        last = signals[-1]
                        last_ts_local = _to_my_tz_ts(last.get("timestamp"))
                        row = {
                            "ts": str(last_ts_local),
                            "asset": asset,
                            "interval": interval,
                            "signal": last.get("signal"),
                            "confidence": float(last.get("confidence", 0.0)),
                            "prob_up": float(last.get("prob_up", 0.0)),
                            "prob_down": float(last.get("prob_down", 0.0)),
                            "price": float(last.get("price", float('nan'))),
                        }
                        runs_dir = getattr(config, 'runs_dir', 'runs')
                        append_csv_row(os.path.join(runs_dir, 'signals.csv'), row)

                        # Feature snapshot for training-from-logs
                        try:
                            # snapshot feature row matching last_ts (if present)
                            if 'feature_df' in locals() and not feature_df.empty:
                                idx = pd.to_datetime(last_ts_local)
                                if idx in feature_df.index:
                                    feats_row = feature_df.loc[[idx]]
                                else:
                                    # nearest previous index to avoid lookahead
                                    prev_idx = feature_df.index[feature_df.index <= idx]
                                    feats_row = feature_df.loc[[prev_idx.max()]] if len(prev_idx) else None
                                if feats_row is not None:
                                    # Write to a single CSV in append mode (robust across environments)
                                    fs_csv = os.path.join(runs_dir, "features_at_signal.csv")
                                    row_df = feats_row.copy()
                                    row_df["asset"] = asset
                                    row_df["interval"] = interval
                                    row_df["ts"] = str(last_ts_local)
                                    # Ensure consistent column order
                                    cols = [c for c in row_df.columns if c not in ("asset","interval","ts")] + ["asset","interval","ts"]
                                    row_df = row_df[cols]
                                    # Append safely (header only if file doesn't exist)
                                    write_header = not os.path.exists(fs_csv)
                                    row_df.to_csv(fs_csv, mode="a", header=write_header, index=False)
                        except Exception as _e:
                            st.info(f"Feature snapshot skipped: {_e}")

                        # Optional webhook
                        if st.session_state.get('alerts_enabled') and st.session_state.get('webhook_url'):
                            conf_badge = confidence_badge(max(row['prob_up'], row['prob_down']))
                            payload = {
                                "title": f"Alpha12_24 {asset} {interval} signal",
                                "signal": row['signal'],
                                "confidence": conf_badge,
                                "prob_up": row['prob_up'],
                                "prob_down": row['prob_down'],
                                "price": row['price'],
                                "ts": row['ts'],
                            }
                            ok = post_webhook_inline(st.session_state['webhook_url'], payload)
                            st.info("Alert posted" if ok else "Alert failed")
                except Exception as _log_e:
                    st.warning(f"Logging/alert skipped: {_log_e}")
    except Exception as e:
        st.error(f"❌ Signal generation failed: {e}")
        signals = []

    # 4.5) Setup creation (pending order from latest non-flat signal)
    st.subheader("🧷 Setup (Pending Order)")
    try:
        if signals:
            latest_sig = signals[-1]
            direction = latest_sig.get("signal", "flat")
            if direction != "flat" and not data.empty:
                last_price = float(latest_sig.get("price", data["close"].iloc[-1]))
                atr_val = _estimate_atr(data)
                rr = float(getattr(config, "min_rr", 1.8))
                now_ts = _to_my_tz_ts(latest_sig.get("timestamp", data.index[-1]))
                setup_levels = _build_setup(
                    direction=direction, price=last_price, atr=atr_val, rr=rr,
                    k_entry=float(st.session_state.get("k_entry", 0.5)),
                    k_stop=float(st.session_state.get("k_stop", 1.0)),
                    valid_bars=int(st.session_state.get("valid_bars", 24)),
                    now_ts=now_ts, bar_interval=interval,
                    entry_buffer_bps=float(st.session_state.get("entry_buffer_bps", 5.0)),
                )
                if setup_levels:
                    # --- Position sizing (risk-based with leverage cap) ---
                    try:
                        balance = float(st.session_state.get("acct_balance", 400.0))
                        max_lev = int(st.session_state.get("max_leverage", 10))
                        risk_pct = float(getattr(config, "risk_per_trade", 1.0))
                        risk_frac = max(risk_pct / 100.0, 0.0)
                        entry_px = float(setup_levels["entry"])
                        stop_px  = float(setup_levels["stop"])
                        stop_dist = abs(entry_px - stop_px)
                        # base sizing by risk
                        risk_amt = balance * risk_frac
                        size_units = 0.0 if stop_dist <= 0 else (risk_amt / stop_dist)
                        notional = size_units * entry_px
                        # cap by leverage
                        notional_cap = balance * max_lev
                        if notional_cap > 0 and notional > notional_cap:
                            scale = notional_cap / (notional + 1e-9)
                            size_units *= scale
                            notional = size_units * entry_px
                        suggested_leverage = 1.0 if balance <= 0 else min(max_lev, max(notional / balance, 1.0))
                    except Exception:
                        size_units = 0.0
                        notional = 0.0
                        suggested_leverage = float(st.session_state.get("max_leverage", 10))
                    # --- Hard gates & frequency cap ---
                    # 5.1 Min confidence to arm
                    if float(latest_sig.get("confidence", 0.0)) < float(st.session_state.get("min_conf_arm", 0.60)):
                        st.info("Setup blocked: confidence below arm threshold.")
                        return

                    # 5.2 Macro regime gate (MA50/MA200)
                    if st.session_state.get("gate_regime", True):
                        close = data["close"]
                        ma_f = close.rolling(50, min_periods=50).mean()
                        ma_s = close.rolling(200, min_periods=200).mean()
                        if ma_f.dropna().empty or ma_s.dropna().empty:
                            st.info("Setup blocked: not enough data to compute regime.")
                            return
                        regime = "bull" if ma_f.iloc[-1] > ma_s.iloc[-1] else "bear"
                        if (regime == "bull" and direction != "long") or (regime == "bear" and direction != "short"):
                            st.info(f"Setup blocked by regime gate ({regime}).")
                            return

                    # 5.3 RR25 gate (nearest 2 expiries average; BTC for BTCUSDT, ETH for ETHUSDT)
                    def _rr25_avg_for(asset_symbol):
                        from pathlib import Path
                        runs = Path(getattr(config,'runs_dir','runs'))
                        cur = "BTC" if "BTC" in asset_symbol else ("ETH" if "ETH" in asset_symbol else "BTC")
                        p = runs / f"deribit_rr25_latest_{cur}.json"
                        try:
                            if not p.exists(): return None
                            import json
                            rows = json.loads(p.read_text()).get("rows", [])
                            if len(rows) == 0: return None
                            # take last 2 expiries if available
                            vals = [float(r.get("rr25", float("nan"))) for r in rows[-2:]]
                            vals = [v for v in vals if not np.isnan(v)]
                            return float(np.mean(vals)) if vals else None
                        except Exception:
                            return None

                    if st.session_state.get("gate_rr25", True):
                        rr = _rr25_avg_for(asset)
                        thr = float(st.session_state.get("rr25_thresh", 0.00))
                        if rr is None:
                            st.caption("RR25 not available → treating as neutral (no block).")
                        else:
                            if direction == "long" and not (rr >= +thr):
                                st.info(f"Setup blocked by RR25 gate (rr={rr:.4f} < +{thr:.4f}).")
                                return
                            if direction == "short" and not (rr <= -thr):
                                st.info(f"Setup blocked by RR25 gate (rr={rr:.4f} > -{thr:.4f}).")
                                return

                    # 5.4 Order-book imbalance gate
                    if st.session_state.get("gate_ob", True):
                        try:
                            from src.data.orderbook_free import ob_features
                            ob = ob_features(symbol=asset, top=20)
                            if not ob:
                                st.info("Setup blocked: OB features unavailable.")
                                return
                            s_imb = float(ob.get("ob_imb_top20", float("nan")))  # signed −1..+1
                            if np.isnan(s_imb):
                                st.caption("OB imbalance not available → treating as neutral (no block).")
                            else:
                                # Symmetric distance threshold: |signed| ≥ ob_signed_thr
                                thr_s = float(st.session_state.get("ob_signed_thr", 1.0 - 2.0 * float(st.session_state.get("ob_edge_delta", 0.20))))
                                if abs(s_imb) < thr_s:
                                    st.info(
                                        f"Setup blocked by OB gate: need |signed imbalance| ≥ {thr_s:.3f}. "
                                        f"Got {s_imb:+.3f} (raw {(0.5+0.5*s_imb):.3f}, Δ {abs(0.5+0.5*s_imb-0.5):.3f})."
                                    )
                                    st.caption(f"OB debug → spread_w (bps)≈{ob.get('ob_spread_w', float('nan')):,.6f}, bidV_top20={ob.get('ob_bidv_top20')}, askV_top20={ob.get('ob_askv_top20')}")
                                    return
                                # Directional consistency: require sign to match signal
                                if direction == "long" and s_imb < 0:
                                    st.info(f"Setup blocked by OB direction: LONG requires positive signed imbalance; got {s_imb:+.3f}.")
                                    return
                                if direction == "short" and s_imb > 0:
                                    st.info(f"Setup blocked by OB direction: SHORT requires negative signed imbalance; got {s_imb:+.3f}.")
                                    return
                        except Exception:
                            st.info("Setup blocked: OB gate error.")
                            return

                    # 5.5 Frequency cap (≤ N setups per 24h for same asset/interval)
                    # NOTE: Dashboard-created setups are tagged origin="manual" and are NOT capped.
                    # The daemon applies caps for origin="auto".
                    try:
                        origin = "manual"
                        if origin != "manual":
                            runs_dir = getattr(config, 'runs_dir', 'runs')
                            setups_path = os.path.join(runs_dir, "setups.csv")
                            df_hist = pd.read_csv(setups_path) if os.path.exists(setups_path) else pd.DataFrame(columns=["created_at","asset","interval","status","origin"])
                            if not df_hist.empty:
                                ts_hist = pd.to_datetime(df_hist["created_at"], errors="coerce", utc=True)
                                try:
                                    df_hist["created_at"] = ts_hist.dt.tz_convert(MY_TZ)
                                except Exception:
                                    df_hist["created_at"] = ts_hist
                                now_local = pd.Timestamp.now(tz="UTC").tz_convert(MY_TZ)
                                since = now_local - pd.Timedelta(days=1)
                                mask = (
                                    (df_hist["asset"]==asset) &
                                    (df_hist["interval"]==interval) &
                                    (df_hist["created_at"]>=since) &
                                    (df_hist["status"].isin(["pending","triggered","target","stop","timeout","cancelled"])) &
                                    (df_hist.get("origin","auto")=="auto")
                                )
                                n = int(mask.sum())
                                cap = int(st.session_state.get("max_setups_per_day", int(os.getenv("MAX_SETUPS_PER_DAY", "0"))))
                                if cap > 0 and n >= cap:
                                    st.info(f"Daily setup cap reached ({n}/{cap}) for auto setups.")
                                    return
                    except Exception:
                        pass

                    # --- end hard gates & cap ---

                    setup_id = _mk_setup_id(asset, interval)
                    trigger_rule = "touch"  # Same as autosignal
                    setup_row = {
                        "id": setup_id,
                        "asset": asset,
                        "interval": interval,
                        "direction": direction,
                        "entry": setup_levels["entry"],
                        "stop": setup_levels["stop"],
                        "target": setup_levels["target"],
                        "rr": setup_levels["rr"],
                        "size_units": float(size_units),
                        "notional_usd": float(notional),
                        "leverage": float(suggested_leverage),
                        "created_at": _to_my_tz_ts(now_ts),
                        "expires_at": setup_levels["expires_at"],
                        "status": "pending",
                        "confidence": float(latest_sig.get("confidence", 0.0)),
                        "trigger_rule": trigger_rule,
                        "entry_buffer_bps": float(st.session_state.get("entry_buffer_bps", 5.0)),
                        "origin": "manual",
                    }
                    _append_setup_row(setup_row)
                    st.success(f"Setup created and saved (ID: {setup_id}).")

                    # Telegram alert (with visible status)
                    if st.session_state.get("tg_bot") and st.session_state.get("tg_chat"):
                        msg = (
                            f"Manual setup {asset} {interval} ({direction.upper()})\n"
                            f"Entry: {setup_row['entry']:.2f}\n"
                            f"Stop: {setup_row['stop']:.2f}\n"
                            f"Target: {setup_row['target']:.2f}\n"
                            f"RR: {setup_row['rr']:.2f}\n"
                            f"Confidence: {float(setup_row['confidence']):.0%}\n"
                            f"Size: {setup_row['size_units']:.6f}  Notional: ${setup_row['notional_usd']:.2f}  Lev: {setup_row['leverage']:.1f}x\n"
                            f"Valid until: {setup_row['expires_at']}"
                        )
                        ok = send_telegram(st.session_state["tg_bot"], st.session_state["tg_chat"], msg)
                        if ok:
                            st.info("Telegram alert sent for new setup.")
                        else:
                            st.warning("Telegram alert failed to send. Verify token/chat and that you've started the bot.")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Direction", direction.upper())
                    c2.metric("Entry", f"{setup_row['entry']:.2f}")
                    c3.metric("Stop", f"{setup_row['stop']:.2f}")
                    c4.metric("Target", f"{setup_row['target']:.2f}")
                    # New sizing metrics row
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Size (units)", f"{float(size_units):,.6f}")
                    d2.metric("Notional (USD)", f"${float(notional):,.2f}")
                    d3.metric("Leverage", f"{float(suggested_leverage):.1f}x")

                    # (Optional) Show balance & leverage context
                    try:
                        bal = float(st.session_state.get("acct_balance", 400.0))
                        mlev = int(st.session_state.get("max_leverage", 10))
                        st.caption(f"Sizing context → Balance: ${bal:,.2f}  •  Max Leverage: {mlev}x")
                    except Exception:
                        pass

                    # Confidence + badge row
                    cbadge = confidence_badge(float(setup_row["confidence"]))
                    cA, cB, cC, cD = st.columns(4)
                    cA.metric("Confidence", f"{float(setup_row['confidence']):.0%}")
                    cB.metric("Conf. Badge", cbadge)
                    cC.metric("RR", f"{float(setup_row['rr']):.2f}")
                    cD.metric("Validity (bars)", f"{int(st.session_state.get('valid_bars', 24))}")

                    st.caption(
                        f"Valid until: {setup_row['expires_at']} • Trigger: {trigger_rule} • Anti-hunt buffer: {setup_row['entry_buffer_bps']} bps"
                    )
                else:
                    st.info("No setup created (invalid direction).")
            else:
                st.info("No actionable signal or no data.")
        else:
            st.info("No signals → no setup created.")
    except Exception as e:
        st.warning(f"Setup creation skipped: {e}")

    # 5) Always render the analysis tabs (with placeholders if needed)
    display_analysis_results(data, feature_df, signals, model_summary, asset, interval, source_choice)


def display_analysis_results(data, feature_df, signals, model_summary, asset, interval, source_choice):
    """Display comprehensive analysis results"""

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📊 Price Chart",
        "🎯 Signals",
        "🤖 Model Performance",
        "📋 Features",
        "⚙️ Settings",
        "🧪 Backtest",
        "📈 Performance",
        "🎚️ Calibration",
        "💼 Account",
        "📒 Live Metrics",
        "📌 Setups",
    ])

    # Price Chart Tab
    with tab1:
        display_price_chart(data, asset, interval)

    # Signals Tab
    with tab2:
        display_signals_analysis(signals, config)

    # Model Performance Tab
    with tab3:
        display_model_performance(model_summary)

    # Features Tab
    with tab4:
        display_feature_analysis(feature_df)

    # Settings Tab
    with tab5:
        display_settings(
            config,
            feature_df.columns.tolist() if hasattr(feature_df, 'columns') else [],
            model_summary.get('model_type','rf'),
            asset,
            interval,
        )

    # Backtest Tab
    with tab6:
        display_backtest_interface(asset, interval, source_choice, model_summary, config)

    # Performance / Win-rate history
    with tab7:
        display_performance_tab(asset, interval)

    # Confidence calibration (separate view)
    with tab8:
        display_calibration_tab()

    # Balance & leverage tracking
    with tab9:
        display_account_tab()

    # Live Metrics Tab
    with tab10:
        display_live_metrics()

    # Setups Tab
    with tab11:
        display_setups_monitor(asset, interval, data)


def display_price_chart(data, asset, interval):
    """Display price chart with technical indicators"""
    if data is None or data.empty:
        st.warning("No price data loaded yet.")
        return

    st.subheader(f"📊 {asset} Price Chart ({interval})")
    st.caption(f"All timestamps displayed in **{MY_TZ}**")

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='OHLC'
    )])

    fig.update_layout(
        title=f'{asset} Price Action',
        yaxis_title='Price (USD)',
        xaxis_title='Time',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True, key=_chart_key("price_chart"))

    # Volume chart
    if 'volume' in data.columns:
        fig_vol = px.bar(data, x=data.index, y='volume', title=f'{asset} Volume')
        st.plotly_chart(fig_vol, use_container_width=True, key=_chart_key("price_vol"))



def display_signals_analysis(signals, config):
    """Display signals analysis with trade plan"""
    st.subheader("🎯 Trading Signals")

    if not signals:
        st.warning("No signals generated")
        return

    signals_df = pd.DataFrame(signals)
    # --- Prompt 4: Latest signal banner ---
    if not signals_df.empty:
        latest_row = signals_df.tail(1).iloc[0]
        conf_badge = confidence_badge(max(latest_row.get('prob_up',0.0), latest_row.get('prob_down',0.0)))
        st.markdown(f"**Latest:** `{latest_row['signal'].upper()}` • Conf: `{conf_badge}` • Price: `{float(latest_row.get('price', float('nan'))):.2f}` • P(up) `{float(latest_row.get('prob_up',0.0)):.2%}`")

    # Signal distribution
    col1, col2, col3 = st.columns(3)

    with col1:
        signal_counts = signals_df['signal'].value_counts()
        st.metric("Total Signals", len(signals_df))
        st.metric("Long Signals", signal_counts.get('long', 0))

    with col2:
        st.metric("Short Signals", signal_counts.get('short', 0))
        st.metric("Flat Signals", signal_counts.get('flat', 0))

    with col3:
        avg_confidence = signals_df['confidence'].mean()
        # Convert numpy array to float if needed
        avg_confidence_val = float(avg_confidence[0]) if hasattr(avg_confidence, '__iter__') else float(avg_confidence)
        st.metric("Avg Confidence", f"{avg_confidence_val:.1%}")

    # Signal timeline
    if 'timestamp' in signals_df.columns:
        signals_df = signals_df.set_index('timestamp')
        fig = px.line(signals_df, x=signals_df.index, y='confidence',
                      title='Signal Confidence Over Time',
                      color='signal')
        st.plotly_chart(fig, use_container_width=True)

    # Recent signals table
    st.subheader("Recent Signals")
    recent_signals = signals_df.tail(10)[['signal', 'confidence', 'prob_up', 'price']]
    st.dataframe(recent_signals)

    # Trade Plan (Live)
    st.subheader("📑 Trade Plan (Live)")
    if not signals_df.empty:
        latest = signals_df.tail(1).iloc[0]
        direction = latest['signal']

        if direction == "flat":
            st.info("No trade — confidence below threshold")
        else:
            entry = float(latest['price'])
            # Risk settings
            risk_pct = float(getattr(config, "risk_per_trade", 1.0))
            risk_perc = risk_pct / 100.0
            balance = float(st.session_state.get("acct_balance", 400.0))
            max_lev = int(st.session_state.get("max_leverage", 10))

            # Compute stop/target using configured RR
            stop_frac = float(getattr(config, "stop_min_frac", 0.005))
            min_rr = float(getattr(config, "min_rr", 1.8))
            fee_bps = float(getattr(config, "taker_bps_per_side", 5))

            stop = entry * (1 - stop_frac) if direction == "long" else entry * (1 + stop_frac)
            tgt = entry * (1 + stop_frac * min_rr) if direction == "long" else entry * (1 - stop_frac * min_rr)

            # Risk-based sizing first
            risk_amt = balance * risk_perc
            per_unit_loss = abs(entry - stop)
            size_units = max(risk_amt / (per_unit_loss + 1e-9), 0.0)
            notional = size_units * entry
            notional_cap = balance * max_lev

            # Cap by leverage limit
            if notional > notional_cap and notional_cap > 0:
                scale = notional_cap / (notional + 1e-9)
                size_units *= scale
                notional = size_units * entry

            # Suggested leverage = min(max_lev, notional / balance)
            suggested_leverage = min(max_lev, max(notional / (balance + 1e-9), 1.0))

            rr = min_rr
            conf = latest['confidence']

            colA, colB, colC, colD = st.columns(4)
            colA.metric("Direction", direction.upper())
            colB.metric("Entry", f"{entry:,.2f}")
            colC.metric("Stop", f"{stop:,.2f}")
            colD.metric("Target", f"{tgt:,.2f}")

            colA, colB, colC, colD = st.columns(4)
            colA.metric("Size (units)", f"{float(size_units):,.6f}")
            colB.metric("Notional", f"${notional:,.2f}")
            colC.metric("Leverage", f"{float(suggested_leverage):.1f}x")
            colD.metric("Est. RR", f"{float(rr):.2f}")

            # Confidence
            st.metric("Confidence", f"{float(conf):.0%}")


def display_model_performance(model_summary):
    """Display model performance metrics"""
    st.subheader("🤖 Model Performance")

    if not model_summary:
        model_summary = EMPTY_MODEL_SUMMARY

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", model_summary.get('model_type', 'Unknown'))
        st.metric("Accuracy", f"{model_summary.get('cv_accuracy_mean', 0):.2%}")

    with col2:
        st.metric("Precision", f"{model_summary.get('cv_precision_mean', 0):.2%}")
        st.metric("Recall", f"{model_summary.get('cv_recall_mean', 0):.2%}")

    with col3:
        st.metric("F1 Score", f"{model_summary.get('cv_f1_mean', 0):.2%}")
        st.metric("Features", model_summary.get('n_features', 0))


def display_feature_analysis(feature_df):
    """Display feature analysis"""
    st.subheader("📋 Feature Analysis")

    if feature_df.empty:
        st.warning("No feature data available")
        return

    # Feature importance (if available)
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    st.write(f"Total features: {len(numeric_cols)}")

    # Correlation matrix
    if len(numeric_cols) > 1:
        corr_matrix = feature_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)


def display_settings(config, feature_cols, model_type, asset, interval):
    """Display current configuration settings"""
    st.subheader("⚙️ Configuration Settings")

    # Display config as JSON
    config_dict = {
        'assets': config.assets,
        'horizons_hours': config.horizons_hours,
        'bar_interval': config.bar_interval,
        'learner': config.learner,
        'train_days': config.train_days,
        'test_days': config.test_days,
        'prob_long': config.prob_long,
        'prob_short': config.prob_short,
        'min_rr': config.min_rr,
        'risk_per_trade': config.risk_per_trade,
        'stop_min_frac': config.stop_min_frac,
        'taker_bps_per_side': config.taker_bps_per_side
    }

    st.json(config_dict)

    # Learn from Logs
    st.subheader("Learn from Logs")
    st.caption("Train on real outcomes from the 24/7 tracker (signals + features + trade_history). Time-based split, metrics, and optional calibrated probabilities.")
    if st.button("Label, Split & Retrain from Live Logs"):
        from src.eval.from_logs import load_joined
        dj = load_joined(getattr(config,'runs_dir','runs'))
        if dj.empty:
            st.warning("No joined live data available yet. Let the daemon run and execute trades (target/stop/timeout) first.")
        else:
            # Build label: win if realized pnl_pct > 0 (you can tighten later)
            if "pnl_pct" not in dj.columns:
                st.error("Joined logs missing 'pnl_pct'. Wait for completed trades.")
                return
            dj["y"] = (dj["pnl_pct"] > 0).astype(int)

            # Feature set: prefer actual engineered feature columns that exist in live snapshot
            # Keep intersection with current `feature_cols` to avoid leakage of target/meta cols
            feat_cols_live = [c for c in dj.columns if (c in feature_cols) or c.startswith("feat_")]
            # Remove obvious non-features if they slipped in
            drop_like = {"pnl_pct","outcome","prob_up","prob_down","confidence","signal","price","asset","interval","ts","exit_ts"}
            feat_cols_live = [c for c in feat_cols_live if c not in drop_like]

            # Clean
            dj = dj.dropna(subset=feat_cols_live+["y"]).copy()
            dj = dj.sort_values("ts") if "ts" in dj.columns else dj
            n_total = len(dj)
            st.write(f"Samples available: **{n_total}** • Feature columns: **{len(feat_cols_live)}**")
            
            # Guard for too few joined samples
            if n_total < 200:
                st.warning("Need ≥200 rows for reliable training. Keep the system running to accumulate more completed trades.")
                return

            # Time-based split: 70/30 (no shuffling)
            split = int(n_total * 0.7)
            train_df, test_df = dj.iloc[:split], dj.iloc[split:]
            X_tr, y_tr = train_df[feat_cols_live], train_df["y"]
            X_te, y_te = test_df[feat_cols_live], test_df["y"]

            # Train
            from src.models.train import ModelTrainer
            tr = ModelTrainer(config)
            model_live = tr.train_model(X_tr, y_tr, model_type)

            # Predict probabilities, with optional calibration
            try:
                from sklearn.calibration import CalibratedClassifierCV
                base = getattr(model_live, 'model', model_live)
                name = type(base).__name__.lower()
                method = 'isotonic' if any(k in name for k in ['forest','tree','boost','xgb']) else 'sigmoid'
                calib = CalibratedClassifierCV(base, method=method, cv=3)
                calib.fit(X_tr, y_tr)
                proba_te = calib.predict_proba(X_te)[:,1]
                used_predictor = 'Calibrated'
            except Exception:
                proba_te = tr.predict_proba(model_live, X_te)[:,1]
                used_predictor = 'Raw'

            # Thresholding using current ThresholdManager (vol-aware if available)
            tm = ThresholdManager(config)
            vol_te = test_df.get('volatility_24h', pd.Series(0.02, index=test_df.index))
            preds, confs = [], []
            for i, ts in enumerate(test_df.index):
                p = proba_te[i]
                sig, conf, _ = tm.determine_signal(np.array([[1-p, p]]), float(vol_te.loc[ts] if ts in vol_te.index else 0.02))
                preds.append(1 if sig == 'long' else (0 if sig == 'short' else (1 if p>=0.5 else 0)))
                confs.append(conf)
            preds = np.array(preds)

            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            acc = accuracy_score(y_te, preds)
            prec = precision_score(y_te, preds, zero_division=0)
            rec = recall_score(y_te, preds, zero_division=0)
            f1 = f1_score(y_te, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_te, proba_te)
            except Exception:
                auc = float('nan')
            cm = confusion_matrix(y_te, preds)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Acc", f"{acc*100:.1f}%")
            c2.metric("Precision", f"{prec*100:.1f}%")
            c3.metric("Recall", f"{rec*100:.1f}%")
            c4.metric("ROC AUC", f"{auc:.3f}")
            st.write("Confusion Matrix:")
            st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

            # Simple expected value assuming RR from config and fee model
            rr = float(getattr(config, "min_rr", 1.8))
            winrate = float((preds == 1).mean() * (y_te[preds==1].mean() if (preds==1).any() else 0))  # crude proxy
            st.caption(f"Predictor: **{used_predictor}** • Assumed RR={rr:.2f}")

            # Save the calibrated predictor, not the raw model
            predictor = calib if 'calib' in locals() else getattr(model_live,'model',model_live)
            meta = {
                "asset": asset, "interval": interval, "from": "live_logs",
                "samples": n_total, "features": len(feat_cols_live),
                "model_type": bt_model if 'bt_model' in locals() else model_type,
                "feature_cols": feat_cols_live,
                "calibrated": hasattr(predictor, "predict_proba") and predictor is not getattr(model_live,'model',model_live),
                "trained_at": str(pd.Timestamp.utcnow().tz_localize('UTC').tz_convert(MY_TZ)),
            }
            path = save_artifacts_inline(predictor, meta, model_dir=getattr(config,'model_dir','artifacts'))
            st.success(f"Model retrained from live logs and saved: {path}")

            # Preview rows
            with st.expander("Preview joined dataset (tail)"):
                st.dataframe(dj.tail(20))

            # Export joined set if user wants offline analysis
            if st.button("Export joined dataset (CSV)"):
                outp = Path(getattr(config,'runs_dir','runs'))/"joined_logs_export.csv"
                dj.to_csv(outp, index=False)
                st.success(f"Exported: {outp}")


def display_backtest_interface(asset, interval, source_choice, model_summary, config):
    """Display backtest interface"""
    st.subheader("🧪 Walk-Forward Backtest")

    # Backtest parameters
    lb_days = st.number_input("Lookback Days", min_value=30, max_value=365, value=90, step=5)
    bt_interval = st.selectbox("Interval", ["5m","15m","1h","4h","1d"],
                              index=["5m","15m","1h","4h","1d"].index(getattr(config, "bar_interval", "1h")))
    # Check for XGBoost availability for backtest
    try:
        from src.models.train import XGBOOST_AVAILABLE
        bt_available_models = ["rf", "logistic"]
        if XGBOOST_AVAILABLE:
            bt_available_models.append("xgb")
    except Exception:
        bt_available_models = ["rf", "logistic"]
        
    # Determine default index
    default_model = model_summary.get('model_type', 'rf')
    if default_model in bt_available_models:
        default_index = bt_available_models.index(default_model)
    else:
        default_index = 0
        
    bt_model = st.selectbox("Model", bt_available_models, index=default_index)
    taker_bps = st.number_input("Taker BPS per side", min_value=0, max_value=50,
                               value=int(getattr(config, "taker_bps_per_side", 5)))
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.01, max_value=5.0,
                              value=float(getattr(config, "risk_per_trade", 1.0)), step=0.01, key="backtest_risk_pct")
    min_rr = st.number_input("Min RR", min_value=1.0, max_value=5.0,
                            value=float(getattr(config, "min_rr", 1.8)), step=0.1)

    run_bt = st.button("Run Backtest", type="primary")

    if run_bt:
        try:
            with st.spinner("Running walk-forward backtest..."):
                # Re-fetch with chosen interval/lookback
                _load_df = resolve_loader(source_choice)
                limit = _calc_limit(bt_interval, lb_days)
                df_bt = _load_df(asset, bt_interval, limit)

                if df_bt.empty:
                    st.error("Failed to fetch backtest data")
                    return
                # Convert to Malaysia local time
                try:
                    df_bt.index = _to_my_tz_index(df_bt.index)
                except Exception:
                    pass

                # Build features
                fe = FeatureEngine()
                feats_bt, feat_cols_bt = fe.build_feature_matrix(df_bt, config.horizons_hours)

                # Drop NaNs
                target_col = f'target_{config.horizons_hours[0]}h'
                feats_bt = feats_bt.dropna(subset=feat_cols_bt + [target_col], how="any")

                if len(feats_bt) < 100:
                    st.error("Insufficient data for backtesting")
                    return

                # Simple WF split
                n = len(feats_bt)
                train_end = int(n*0.7)
                X_tr = feats_bt.iloc[:train_end][feat_cols_bt]
                y_tr = feats_bt.iloc[:train_end][target_col]
                X_te = feats_bt.iloc[train_end:][feat_cols_bt]
                y_te = feats_bt.iloc[train_end:][target_col]

                # Train model
                trainer = ModelTrainer(config)
                model_bt = trainer.train_model(X_tr, y_tr, bt_model)
                # --- Robust probability normalization for backtest ---
                import numpy as _np  # safe (idempotent) import

                # Close series aligned to test set index
                if "close" in feats_bt.columns:
                    close_te = feats_bt.iloc[train_end:]["close"]
                    # ensure same index as X_te for alignment
                    close_te = close_te.loc[X_te.index]
                else:
                    # fall back to raw OHLC close aligned by index
                    try:
                        close_te = df_bt["close"].loc[X_te.index]
                    except Exception:
                        close_te = pd.Series(index=X_te.index, dtype=float)

                # Probability matrix (n,2) where col1 = P(class=1)
                probs_raw = trainer.predict_proba(model_bt, X_te)
                probs2 = _ensure_two_col_proba(probs_raw)     # (n,2)
                p1_series = pd.Series(probs2[:, 1], index=X_te.index)

                # Close aligned to test index (fallback NaN)
                if "close" in feats_bt.columns:
                    close_te = feats_bt["close"].reindex(X_te.index)
                else:
                    close_te = df_bt["close"].reindex(X_te.index)

                # Volatility aligned (fallback 0.02)
                vol_te = _series_aligned(feats_bt, X_te.index, "volatility_24h", fallback=0.02)

                # Thresholding controls
                tm = ThresholdManager(config)
                rr_target = float(min_rr)
                min_conf = float(st.session_state.get("min_conf_arm", 0.58))
                prefer = "auto"

                def _as_row_proba(p_up: float) -> _np.ndarray:
                    """Return a single (1,2) row for ThresholdManager.determine_signal."""
                    pu = float(_np.clip(p_up, 0.0, 1.0))
                    return _np.array([[1.0 - pu, pu]], dtype=float)

                # Build backtest results using aligned indices
                rows = []
                for i, (ts, price) in enumerate(zip(X_te.index, close_te)):
                    p = probs2[i].reshape(1, -1) if probs2.ndim == 2 else np.array([[probs2[i]]])
                    side, conf, thr = tm.determine_signal(
                        p,
                        rr_target=rr_target,
                        min_conf=min_conf
                    )

                    rows.append({
                        "ts": ts,
                        "price": float(price) if pd.notna(price) else float("nan"),
                        "win_prob": float(p[0, 1]) if p.shape[1] > 1 else float(p[0, 0]),
                        "signal": side,
                        "confidence": float(conf),
                        "rr": rr_target,
                    })

                res_df = pd.DataFrame(rows).set_index("ts").sort_index()

                # Calculate trade PnL (simplified, symmetric for long/short)
                res_df["trade_pnl"] = _np.where(
                    res_df["signal"] == "long",
                    res_df["rr"] * res_df["win_prob"] - (1 - res_df["win_prob"]),
                    _np.where(
                        res_df["signal"] == "short",
                        res_df["rr"] * res_df["win_prob"] - (1 - res_df["win_prob"]),
                        0.0,
                    ),
                )

                # Equity curve from constant risk %
                risk_frac = float(risk_pct) / 100.0
                res_df["equity"] = (1 + res_df["trade_pnl"] * risk_frac).cumprod() * 1000.0

                # Metrics
                win_mask = res_df["trade_pnl"] > 0
                winrate = float(win_mask.mean()) if len(res_df) else 0.0
                avg_rr = float(res_df["rr"].replace([_np.inf, -_np.inf], _np.nan).dropna().mean()) if len(res_df) else 0.0
                trades = int((res_df["signal"] != "flat").sum())
                equity = res_df["equity"]
                max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
                ret_pct = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100) if len(equity) else 0.0
                pf = float(res_df.loc[win_mask, "trade_pnl"].sum() / abs(res_df.loc[~win_mask, "trade_pnl"].sum())) if (~win_mask).any() else float("inf")
                sharpe = float((res_df["trade_pnl"].mean() / (res_df["trade_pnl"].std() + 1e-9)) * _np.sqrt(252 * 24 * 60 / 5)) if res_df["trade_pnl"].std() > 0 else 0.0

                # Display metrics
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Winrate", f"{winrate*100:.1f}%")
                    st.metric("Avg RR", f"{avg_rr:.2f}")
                    st.metric("Trades", f"{trades}")

                with c2:
                    st.metric("Profit Factor", f"{pf:.2f}")
                    st.metric("Max Drawdown", f"{max_dd*100:.1f}%")
                    st.metric("Net P&L", f"{ret_pct:.1f}%")

                # Equity curve
                st.subheader("Equity Curve")
                st.line_chart(res_df["equity"].rename("Equity (USD)"))

                # Recent trades
                st.subheader("Recent Trades")
                st.dataframe(res_df.tail(10))

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.exception(e)



# ---------- NEW HELPERS: performance / calibration / account ----------
def _chart_key(name: str) -> str:
    """
    Generate a unique Streamlit element key per render to avoid
    StreamlitDuplicateElementId for plotly charts that share specs.
    """
    try:
        import streamlit as st  # avoid top-level circulars
        seq = st.session_state.get("_chart_seq", 0) + 1
        st.session_state["_chart_seq"] = seq
        return f"{name}_{seq}"
    except Exception:
        import time, random
        return f"{name}_{int(time.time()*1e3)}_{random.randint(0, 9999)}"

def _load_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=parse_dates, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

# --- Inserted: display_live_metrics ---
# --- Inserted: display_live_metrics ---
def display_live_metrics():
    """Live metrics dashboard: recent signals, direction mix, and heartbeat."""
    st.subheader("📒 Live Metrics")

    runs_dir = getattr(config, 'runs_dir', 'runs')
    sig_path = os.path.join(runs_dir, "signals.csv")
    th_path  = os.path.join(runs_dir, "trade_history.csv")
    hb_path  = os.path.join(runs_dir, "daemon_heartbeat.txt")

    # Heartbeat
    hb_msg = "Tracker not detected."
    if os.path.exists(hb_path):
        try:
            hb = pd.to_datetime(Path(hb_path).read_text())
            hb_local = hb.tz_localize("UTC").tz_convert(MY_TZ)
            hb_msg = f"Tracker heartbeat: {hb_local}"
            st.success(hb_msg)
        except Exception:
            st.success("Tracker heartbeat: detected")
    else:
        st.warning("Tracker not detected. Start it with:  \n`PYTHONPATH=$(pwd) nohup python -m src.daemon.tracker >/tmp/alpha_tracker.log 2&gt;&amp;1 &amp;`")

    # Signals
    df_sig = _load_csv_safe(sig_path, parse_dates=["ts"])
    if df_sig.empty:
        st.info("No signals recorded yet.")
        return

    # Clean and sort
    ts_sig = pd.to_datetime(df_sig["ts"], errors="coerce", utc=True)
    try:
        df_sig["ts"] = ts_sig.dt.tz_convert(MY_TZ)
    except Exception:
        df_sig["ts"] = ts_sig
    df_sig = df_sig.dropna(subset=["ts"]).sort_values("ts")
    df_sig = df_sig.tail(500)  # last 500 rows for speed

    # Direction mix (last 100)
    lastN = st.slider("Window (signals)", 50, 500, 200, 25, key="lm_n")
    df_last = df_sig.tail(lastN)

    c1, c2, c3 = st.columns(3)
    total = len(df_last)
    long_n = int((df_last["signal"] == "long").sum()) if "signal" in df_last.columns else 0
    short_n = int((df_last["signal"] == "short").sum()) if "signal" in df_last.columns else 0
    flat_n = int((df_last["signal"] == "flat").sum()) if "signal" in df_last.columns else 0
    c1.metric("Signals (window)", f"{total}")
    c2.metric("Long / Short", f"{long_n} / {short_n}")
    c3.metric("Flat", f"{flat_n}")

    # Confidence timeline
    if "confidence" in df_sig.columns:
        try:
            fig = px.line(df_sig, x="ts", y="confidence", color="signal", title="Signal confidence over time")
            st.plotly_chart(fig, use_container_width=True, key=_chart_key("lm_conf"))
        except Exception:
            st.line_chart(df_sig.set_index("ts")[["confidence"]])

    # Probability histogram (if present)
    if {"prob_up", "prob_down"}.issubset(df_sig.columns):
        try:
            fig = px.histogram(df_last, x="prob_up", nbins=20, title="Distribution of P(up) in window")
            st.plotly_chart(fig, use_container_width=True, key=_chart_key("lm_pup"))
        except Exception:
            st.bar_chart(df_last[["prob_up"]])

    # Recent signals table
    with st.expander("Recent signals (tail)"):
        cols_show = [c for c in ["ts","asset","interval","signal","confidence","prob_up","prob_down","price"] if c in df_sig.columns]
        st.dataframe(df_sig.tail(25)[cols_show])

    # Trades-per-day quick view (if trade history exists)
    df_th = _load_csv_safe(th_path, parse_dates=["ts","exit_ts"])
    if not df_th.empty:
        try:
            df_cnt = df_th.copy()
            # Parse "ts" as UTC then convert to MY_TZ
            ts = pd.to_datetime(df_cnt["ts"], errors="coerce", utc=True)
            try:
                df_cnt["ts"] = ts.dt.tz_convert(MY_TZ)
            except Exception:
                df_cnt["ts"] = ts
            # Parse "exit_ts" as UTC then convert to MY_TZ
            ts_exit = pd.to_datetime(df_cnt["exit_ts"], errors="coerce", utc=True)
            try:
                df_cnt["exit_ts"] = ts_exit.dt.tz_convert(MY_TZ)
            except Exception:
                df_cnt["exit_ts"] = ts_exit
            df_cnt = df_cnt.dropna(subset=["ts"]).set_index("ts").sort_index()
            daily = df_cnt.groupby(pd.Grouper(freq="D")).size()
            st.bar_chart(daily.rename("trades_per_day"))
        except Exception:
            pass

# --- Setups monitor (pending/triggered lifecycle) ---
def display_setups_monitor(asset: str, interval: str, data: pd.DataFrame):
    """Show comprehensive setups lifecycle with Telegram alerts status.
    Uses setups.csv persisted via _append_setup_row / _save_setups_df.
    """
    st.subheader("📊 Setups Monitor & Lifecycle Tracking")

    # Load saved setups (tolerant reader already normalizes columns)
    try:
        setups = _load_setups_df()
    except Exception as e:
        st.warning(f"Failed to load setups.csv: {e}")
        return

    if setups.empty:
        st.info("No setups yet. Run analysis to generate one.")
        return

    # Normalize status/origin
    if "status" in setups.columns:
        setups["status"] = setups["status"].astype(str).str.strip().str.lower()
    if "origin" in setups.columns:
        setups["origin"] = setups["origin"].fillna("manual").astype(str).str.strip().str.lower()
    # Convert time columns
    for c in ("created_at","expires_at","triggered_at"):
        if c in setups.columns:
            ts = pd.to_datetime(setups[c], errors="coerce", utc=True)
        try:
            setups[c] = ts.dt.tz_convert(MY_TZ)
        except Exception:
            setups[c] = ts

    # Filter by current asset/interval (UI context)
    df_view = setups.copy()
    if "asset" in df_view.columns:
        df_view = df_view[df_view["asset"].astype(str) == str(asset)]
    if "interval" in df_view.columns:
        df_view = df_view[df_view["interval"].astype(str) == str(interval)]

    if df_view.empty:
        st.info("No setups for this asset/interval yet.")
        return

    # ---------- summary counts ----------
    total = len(df_view)
    statuses = ["pending", "triggered", "target", "stop", "timeout", "cancelled"]
    counts = {s: int((df_view["status"] == s).sum()) for s in statuses}

    # Display status summary with colors
    st.markdown("### 📈 Setup Status Summary")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    col1.metric("Total", total, delta=None)
    col2.metric("🟡 Pending", counts['pending'], delta=None)
    col3.metric("🔵 Triggered", counts['triggered'], delta=None)
    col4.metric("🟢 Target", counts['target'], delta=None)
    col5.metric("🔴 Stop", counts['stop'], delta=None)
    col6.metric("🟣 Timeout", counts['timeout'], delta=None)
    col7.metric("⚫ Cancelled", counts['cancelled'], delta=None)

    st.markdown("---")

    # Lifecycle sections
    st.markdown("### 🔄 Setup Lifecycle")

    # 1. PENDING SETUPS (waiting for entry)
    pending = df_view[df_view["status"] == "pending"].copy()
    if not pending.empty:
        st.markdown("#### 🟡 **PENDING SETUPS** - Waiting for Entry Price")
        st.caption("These setups are waiting for the entry price to be reached. They will trigger when price touches the entry level.")
        
        pending_cols = ["id", "asset", "interval", "direction", "entry", "stop", "target", "rr", "confidence", "created_at", "expires_at", "origin"]
        pending_cols = [col for col in pending_cols if col in pending.columns]
        
        if "created_at" in pending.columns:
            pending = pending.sort_values("created_at", ascending=False)
        
        st.dataframe(pending[pending_cols], use_container_width=True, hide_index=True)
        st.caption(f"📊 {len(pending)} pending setups")
        
        # Cancel setup functionality
        if st.button("❌ Cancel Selected Setup", key="cancel_pending"):
            if len(pending) > 0:
                # Get the most recent pending setup
                latest_pending = pending.iloc[0]
                setup_id = latest_pending["id"]
                
                try:
                    # Load and update setups
                    setups = _load_setups_df()
                    mask = setups["id"] == setup_id
                    if mask.any():
                        setups.loc[mask, "status"] = "cancelled"
                        _save_setups_df(setups)
                        
                        # Send Telegram alert
                        from src.daemon.tracker import _tg_send
                        _tg_send(f"❌ Setup CANCELLED\\n{latest_pending['asset']} {latest_pending['interval']} {latest_pending['direction'].upper()}\\nSetup ID: {setup_id}")
                        
                        st.success(f"✅ Setup {setup_id} cancelled successfully!")
                        st.rerun()
                    else:
                        st.error("Setup not found!")
                except Exception as e:
                    st.error(f"Failed to cancel setup: {e}")
            else:
                st.warning("No pending setups to cancel")

    # 2. TRIGGERED SETUPS (active trades)
    triggered = df_view[df_view["status"] == "triggered"].copy()
    if not triggered.empty:
        st.markdown("#### 🔵 **TRIGGERED SETUPS** - Active Trades")
        st.caption("These setups have been triggered and are now active trades waiting for target/stop.")
        
        triggered_cols = ["id", "asset", "interval", "direction", "entry", "stop", "target", "rr", "confidence", "triggered_at", "expires_at", "origin"]
        triggered_cols = [col for col in triggered_cols if col in triggered.columns]
        
        if "triggered_at" in triggered.columns:
            triggered = triggered.sort_values("triggered_at", ascending=False)
        
        st.dataframe(triggered[triggered_cols], use_container_width=True, hide_index=True)
        st.caption(f"📊 {len(triggered)} active trades")

    # 3. COMPLETED SETUPS (target/stop/timeout)
    completed = df_view[df_view["status"].isin(["target", "stop", "timeout"])].copy()
    if not completed.empty:
        st.markdown("#### ✅ **COMPLETED SETUPS** - Finished Trades")
        st.caption("These setups have completed with target hit, stop loss, or timeout.")
        
        # Load trade history for PnL data
        runs_dir = getattr(config, 'runs_dir', 'runs')
        th_path = os.path.join(runs_dir, "trade_history.csv")
        if os.path.exists(th_path):
            try:
                trade_history = pd.read_csv(th_path)
                if not trade_history.empty:
                    # Merge with trade history for PnL
                    completed = completed.merge(
                        trade_history[["setup_id", "outcome", "pnl_pct", "exit_ts"]], 
                        left_on="id", right_on="setup_id", how="left"
                    )
            except Exception:
                pass
        
        completed_cols = ["id", "asset", "interval", "direction", "entry", "stop", "target", "rr", "outcome", "pnl_pct", "created_at", "origin"]
        completed_cols = [col for col in completed_cols if col in completed.columns]
        
        if "created_at" in completed.columns:
            completed = completed.sort_values("created_at", ascending=False)
        
        st.dataframe(completed[completed_cols], use_container_width=True, hide_index=True)
        st.caption(f"📊 {len(completed)} completed trades")

    # 4. CANCELLED SETUPS
    cancelled = df_view[df_view["status"] == "cancelled"].copy()
    if not cancelled.empty:
        st.markdown("#### ⚫ **CANCELLED SETUPS**")
        st.caption("These setups were cancelled before execution.")
        
        cancelled_cols = ["id", "asset", "interval", "direction", "entry", "stop", "target", "rr", "confidence", "created_at", "origin"]
        cancelled_cols = [col for col in cancelled_cols if col in cancelled.columns]
        
        if "created_at" in cancelled.columns:
            cancelled = cancelled.sort_values("created_at", ascending=False)
        
        st.dataframe(cancelled[cancelled_cols], use_container_width=True, hide_index=True)
        st.caption(f"📊 {len(cancelled)} cancelled setups")

    # Telegram alerts status
    st.markdown("---")
    st.markdown("### 📱 Telegram Alerts Status")
    
    tg_token = os.getenv("TG_BOT_TOKEN") or os.getenv("TG_BOT")
    tg_chat = os.getenv("TG_CHAT_ID") or os.getenv("TG_CHAT")
    
    if tg_token and tg_chat:
        st.success("✅ Telegram alerts are configured and active")
        st.caption("You will receive alerts for: Setup creation, triggering, target hits, stop losses, timeouts, and cancellations")
    else:
        st.warning("⚠️ Telegram alerts not configured")
        st.caption("Set TG_BOT_TOKEN and TG_CHAT_ID environment variables to receive alerts")
    
    # Auto-refresh reminder
    st.markdown("---")
    st.caption("💡 **Tip**: Enable auto-refresh in the sidebar to keep this monitor live and see real-time status changes!")
    
    # Test Telegram alerts
    if st.button("🔔 Test Telegram Alert", help="Send a test message to verify Telegram alerts are working"):
        try:
            from src.daemon.tracker import _tg_send
            _tg_send("Test alert from dashboard\\nThis confirms Telegram alerts are working correctly!")
            st.success("✅ Test alert sent successfully!")
        except Exception as e:
            st.error(f"❌ Failed to send test alert: {e}")

def _status_badge_html(status: str) -> str:
    s = (status or "").strip().lower()
    color = "#6c757d"  # default gray
    label = s.upper() if s else "N/A"
    if s == "pending":
        color = "#f59f00"  # amber
    elif s == "triggered":
        color = "#228be6"  # blue
    elif s == "target":
        color = "#2fb344"  # green
    elif s == "stop":
        color = "#e03131"  # red
    elif s == "timeout":
        color = "#ae3ec9"  # purple
    elif s == "cancelled":
        color = "#495057"  # dark gray
    return f"<span style='background:{color};color:#fff;padding:2px 6px;border-radius:6px;font-size:12px;font-weight:600'>{label}</span>"

def _render_setups_table(df: pd.DataFrame) -> str:
    """Render a compact HTML table so we can mix rich badges with values."""
    if df.empty:
        return "<em>No rows</em>"

    css = """
    <style>
      table.alpha-setups { border-collapse: collapse; width: 100%; }
      .alpha-setups th, .alpha-setups td { border: 1px solid #e9ecef; padding: 6px 8px; font-size: 12px; }
      .alpha-setups th { background: #f8f9fa; text-align: left; }
      .alpha-setups tr:nth-child(even) { background: #fcfcfd; }
      .alpha-setups td { vertical-align: middle; }
      .alpha-setups .num { text-align: right; font-variant-numeric: tabular-nums; }
      .alpha-setups .dt  { white-space: nowrap; }
      .alpha-setups .id  { font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
    </style>
    """

    cols = list(df.columns)
    thead = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"

    rows_html = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r.get(c, "")
            cls = ""
            if c in {"entry","stop","target","rr","confidence","size_units","notional_usd","leverage","entry_buffer_bps"}:
                cls = "num"
                try:
                    if isinstance(v, (int,float,np.floating)) and not pd.isna(v):
                        if c in {"entry","stop","target","notional_usd"}:
                            v = f"{float(v):,.2f}"
                        elif c in {"rr","leverage"}:
                            v = f"{float(v):.2f}"
                        elif c == "confidence":
                            v = f"{float(v):.0%}"
                        else:
                            v = f"{float(v):,.6f}"
                except Exception:
                    pass
            elif c in {"created_at","expires_at"}:
                cls = "dt"
                try:
                    if pd.notna(v):
                        v = pd.to_datetime(v).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            elif c == "id":
                cls = "id"
            tds.append(f"<td class='{cls}'>{v}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    tbody = "".join(rows_html)
    return css + f"<table class='alpha-setups'><thead>{thead}</thead><tbody>{tbody}</tbody></table>"

def display_performance_tab(asset: str, interval: str):
    st.subheader("📈 Performance / Win-rate History")
    runs_dir = getattr(config, 'runs_dir', 'runs')
    th_path = os.path.join(runs_dir, "trade_history.csv")
    sig_path = os.path.join(runs_dir, "signals.csv")

    df_th = _load_csv_safe(th_path, parse_dates=["ts","exit_ts"])
    df_sig = _load_csv_safe(sig_path, parse_dates=["ts"])

    if df_th.empty and df_sig.empty:
        st.info("No trades or signals recorded yet. Keep the tracker running.")
        return

    # Completed trades only
    if not df_th.empty:
        df = df_th.copy()
        win = df["outcome"].isin(["target"])
        loss = df["outcome"].isin(["stop","timeout"])
        df = df[(win | loss)].copy()
        if df.empty:
            st.info("No completed trades yet.")
        else:
            df = df.sort_values("exit_ts" if "exit_ts" in df.columns else "ts")
            df["is_win"] = win[win | loss].astype(int)
            df["cum_trades"] = range(1, len(df)+1)
            df["cum_winrate"] = df["is_win"].expanding().mean()
            N = st.slider("Rolling window (trades)", 10, 200, 50, 5)
            df["roll_winrate"] = df["is_win"].rolling(N, min_periods=max(5, N//5)).mean()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{len(df)}")
            c2.metric("Winrate (cum)", f"{df['cum_winrate'].iloc[-1]*100:.1f}%")
            if "pnl_pct" in df.columns:
                gains = df.loc[df["pnl_pct"]>0, "pnl_pct"].sum()
                losses = -df.loc[df["pnl_pct"]<0, "pnl_pct"].sum()
                pf = float(gains / losses) if losses > 0 else float("inf")
                c3.metric("Profit Factor", f"{pf:.2f}")
            else:
                c3.metric("Profit Factor", "n/a")
            if "exit_ts" in df.columns and "ts" in df.columns:
                hold = (pd.to_datetime(df["exit_ts"]) - pd.to_datetime(df["ts"]))\
                    .dt.total_seconds()/3600
                c4.metric("Avg Hold (h)", f"{hold.mean():.2f}")
            else:
                c4.metric("Avg Hold (h)", "n/a")

            # Charts
            st.markdown("**Cumulative vs Rolling Win-rate**")
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                x = pd.to_datetime(df.get("exit_ts", df.get("ts")))
                fig.add_trace(go.Scatter(x=x, y=df["cum_winrate"], name="Cumulative WR"))
                fig.add_trace(go.Scatter(x=x, y=df["roll_winrate"], name=f"Rolling WR ({N})"))
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True, key=_chart_key("perf_tab"))
            except Exception:
                st.line_chart(df.set_index(pd.to_datetime(df.get("exit_ts", df.get("ts"))))[["cum_winrate","roll_winrate"]])

    st.markdown("---")
    st.subheader("Signal Volume (by direction)")
    if not df_sig.empty:
        df = df_sig.copy()
        if "ts" in df.columns:
            # Parse "ts" as UTC then convert to MY_TZ
            ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
            try:
                df["ts"] = ts.dt.tz_convert(MY_TZ)
            except Exception:
                df["ts"] = ts
            df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
            daily = df.groupby([pd.Grouper(freq="D"), "signal"]).size().unstack(fill_value=0)
            st.bar_chart(daily)
        else:
            st.dataframe(df.tail(20))
    else:
        st.caption("No signals.csv found yet.")

def display_calibration_tab():
    st.subheader("🎚️ Confidence Calibration")
    runs_dir = getattr(config, 'runs_dir', 'runs')
    th_path = os.path.join(runs_dir, "trade_history.csv")
    df = load_trade_history(th_path)
    if df.empty or not any(df["outcome"].isin(["target","stop","timeout"])):
        st.info("No completed trades yet. Keep the tracker running.")
        return
    g = calibration_bins(df)
    if g.empty:
        st.info("Not enough samples to compute calibration.")
        return
    st.dataframe(g[["conf_mid","empirical_winrate","count"]])
    try:
        import plotly.express as px
        fig = px.scatter(
            g, x="conf_mid", y="empirical_winrate", size="count", trendline="ols",
            labels={"conf_mid":"Predicted confidence","empirical_winrate":"Observed winrate"}
        )
    except Exception:
        import plotly.express as px
        fig = px.scatter(
            g, x="conf_mid", y="empirical_winrate", size="count",
            labels={"conf_mid":"Predicted confidence","empirical_winrate":"Observed winrate"}
        )
    st.plotly_chart(fig, use_container_width=True, key=_chart_key("calib_tab"))

def display_account_tab():
    st.subheader("💼 Balance & Leverage Tracking")
    runs_dir = getattr(config, 'runs_dir', 'runs')
    th_path = os.path.join(runs_dir, "trade_history.csv")
    su_path = os.path.join(runs_dir, "setups.csv")

    df_th = _load_csv_safe(th_path, parse_dates=["ts","exit_ts"])
    df_su = _load_csv_safe(su_path, parse_dates=["created_at","expires_at"])\
        .replace({"": np.nan})

    bal = float(st.session_state.get("acct_balance", 400.0))
                max_lev = int(st.session_state.get("max_leverage", 10))
    col1, col2, col3 = st.columns(3)
    col1.metric("Configured Balance", f"${bal:,.2f}")
    col2.metric("Max Leverage", f"{max_lev}x")
    if not df_su.empty and "leverage" in df_su.columns:
        avg_lev = float(pd.to_numeric(df_su["leverage"], errors="coerce").dropna().tail(200).mean())
        col3.metric("Avg Suggested Lev", f"{avg_lev:.1f}x")
    else:
        col3.metric("Avg Suggested Lev", "n/a")

    st.markdown("---")
    # --- Equity curve from completed trades, if available ---
    if not df_th.empty:
        df = df_th.copy()
        # Keep only completed outcomes if outcome column exists
        if "outcome" in df.columns:
            mask_done = df["outcome"].isin(["target", "stop", "timeout", "cancelled"])
            df = df[mask_done].copy()
        # Sort by exit_ts when available, otherwise by ts
        if "exit_ts" in df.columns:
            # Parse "exit_ts" as UTC then convert to MY_TZ
            ts_exit = pd.to_datetime(df["exit_ts"], errors="coerce", utc=True)
            try:
                df["exit_ts"] = ts_exit.dt.tz_convert(MY_TZ)
            except Exception:
                df["exit_ts"] = ts_exit
            df = df.sort_values("exit_ts")
            ts_fallback = pd.to_datetime(df.get("ts", pd.NaT), errors="coerce", utc=True)
            try:
                ts_fallback = ts_fallback.tz_convert(MY_TZ)
            except Exception:
                pass
            x_idx = df["exit_ts"].fillna(ts_fallback)
        else:
            df["ts"] = pd.to_datetime(df.get("ts"), errors="coerce", utc=True)
            try:
                df["ts"] = df["ts"].dt.tz_convert(MY_TZ)
            except Exception:
                pass
            df = df.sort_values("ts")
            x_idx = df["ts"]

        # Determine per-trade return in pct (net of fees if available)
        ret_col = None
        for c in ["pnl_pct_net", "pnl_pct"]:
            if c in df.columns:
                ret_col = c
                break

        if ret_col is None:
            # Fallback: build rough PnL from RR plan and outcome
            rr = float(getattr(config, "min_rr", 1.8))
            fee_bps = float(getattr(config, "taker_bps_per_side", 5))
            fee_frac = (fee_bps / 1e4) * 2.0  # taker in/out
            # Assume unit risk = 1, win = +rr, loss/timeout = -1
            df["_raw_ret"] = np.where(
                df.get("outcome", "").eq("target"), rr,
                np.where(df.get("outcome", "").isin(["stop", "timeout"]), -1.0, 0.0)
            )
            df["_ret_frac"] = df["_raw_ret"] - fee_frac
            ret_series = df["_ret_frac"]
        else:
            ret_series = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0) / 100.0

        # Equity curve using configured balance as starting capital
        eq = float(bal) * (1.0 + ret_series).cumprod()
        eq.index = x_idx.values

        st.subheader("Equity Curve")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity (USD)"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, key=_chart_key("acct_equity"))
        except Exception:
            st.line_chart(eq.rename("Equity (USD)"), use_container_width=True)

        # Recent completed trades table
        with st.expander("Recent completed trades"):
            cols_show = [c for c in ["ts", "exit_ts", "asset", "interval", "direction", "outcome", "pnl_pct_net", "pnl_pct", "rr_planned", "confidence"] if c in df.columns]
            st.dataframe(df[cols_show].tail(20))
    else:
        st.info("No trade history yet.")

    st.markdown("---")
    st.subheader("Recent Setups")
    if not df_su.empty:
        # Normalize and sort
        df_su = df_su.copy()
        df_su["created_at"] = pd.to_datetime(df_su.get("created_at"), errors="coerce", utc=True)
        try:
            df_su["created_at"] = df_su["created_at"].dt.tz_convert(MY_TZ)
        except Exception:
            pass
        df_su = df_su.sort_values("created_at")
        cols = [c for c in ["created_at", "asset", "interval", "direction", "entry", "stop", "target", "rr", "size_units", "notional_usd", "leverage", "status", "confidence", "expires_at"] if c in df_su.columns]
        st.dataframe(df_su[cols].tail(20))
    else:
        st.caption("No setups recorded yet. Create one from Signals → Setup.")

# Ensure module guard at end of file
if __name__ == "__main__":
    main()
