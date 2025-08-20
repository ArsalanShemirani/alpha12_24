from __future__ import annotations
import os
import time
import json
import signal
from pathlib import Path
from typing import Dict, Any, Optional

# --- Runtime debug banner: verify which file is executing and pandas version ---
try:
    import pandas as _pd_ver
    print(f"[tracker] using file: {__file__}")
    print(f"[tracker] pandas version: {_pd_ver.__version__}")
except Exception:
    pass

# Load environment variables from .env file if it exists
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
                            os.environ[key.strip()] = value.strip()
    except Exception:
        pass

# Load environment variables at startup
load_env_vars()

import pandas as pd
import numpy as np
import requests

from src.core.config import config
from src.runtime.clock import now, to_my_tz, bars_to_minutes
from src.data.price_feed import get_latest_candle, get_window

# --- Tracker robustness & gating patch ---
# - Fix heartbeat tz error (avoid tz_localize on tz-aware timestamps)
# - Ensure all CSV writes include a 'ts' column
# - Tag tracker-created setups as origin='auto'
# - Enforce daily cap only for origin='auto' (dashboard/manual stays uncapped)

# Local timezone (Malaysia) used across the project
MY_TZ = "Asia/Kuala_Lumpur"

# --- Exit resolution helper ---
def _resolve_exit(direction, entry, stop, target, bar):
    hi, lo = float(bar["high"]), float(bar["low"])
    # wick order first: if both hit, which touched first?
    if direction == "long":
        hit_stop  = lo <= stop
        hit_tgt   = hi >= target
    else:
        hit_stop  = hi >= stop
        hit_tgt   = lo <= target
    if hit_stop and hit_tgt:
        # prefer first touch by proximity to open
        o = float(bar.get("open", (hi+lo)/2))
        # estimate which is closer to open => likely touched first
        ds = abs(o - stop)
        dt = abs(o - target)
        return "target" if dt < ds else "stop"
    if hit_tgt:  return "target"
    if hit_stop: return "stop"
    return None

def _now_utc() -> pd.Timestamp:
    """UTC-aware current timestamp."""
    return pd.Timestamp.now(tz="UTC")

def _now_local() -> pd.Timestamp:
    """Local (MY_TZ) current timestamp."""
    return _now_utc().tz_convert(MY_TZ)

def _iso_utc() -> str:
    return _now_utc().isoformat()

def _iso_local() -> str:
    """Return current time in Malaysian timezone as ISO string."""
    return _now_local().isoformat()

def _runs_dir() -> Path:
    try:
        from src.core.config import config
        rd = getattr(config, "runs_dir", "runs")
    except Exception:
        rd = "runs"
    p = Path(rd)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """Append a dict row to CSV ensuring stable header and 'ts' present."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if "ts" not in row:
        row["ts"] = _iso_local()  # Use Malaysian time for all timestamps
    write_header = not path.exists()
    # Freeze field order for stability
    fieldnames = list(row.keys())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

 # --- Telegram alerts (MarkdownV2 safe) ---
_DEF_TZ = "Asia/Kuala_Lumpur"

# --- Signal handling for graceful shutdown ---
_STOP = False
def _sig_stop(signum, frame):
    global _STOP
    _STOP = True
    try:
        print(f"[tracker] received signal {signum}, stopping...")
    except Exception:
        pass

# --- Robust bar normalization for candle payloads ---
def _normalize_bar(raw) -> Optional[Dict]:
    """Coerce a latest-candle payload into a dict with keys:
    {"ts": pd.Timestamp(tz-aware in MY), "open","high","low","close"}.
    Accepts dicts with different ts field names or numeric epochs in s/ms.
    Returns None if mandatory fields are missing.
    """
    if raw is None:
        return None
    # If it's a pandas Series or object with attributes
    try:
        if hasattr(raw, "to_dict"):
            bar = dict(raw.to_dict())
        else:
            bar = dict(raw)
    except Exception:
        bar = raw if isinstance(raw, dict) else None
    if not isinstance(bar, dict):
        return None

    # Find timestamp-like field
    ts_val = None
    for key in ("ts", "timestamp", "open_time", "close_time", "t", "T", "time"):
        if key in bar and bar[key] is not None and not (isinstance(bar[key], float) and np.isnan(bar[key])):
            ts_val = bar[key]
            break

    ts = None
    if ts_val is not None:
        # Numeric epoch (s or ms)
        try:
            if isinstance(ts_val, (int, float)) and np.isfinite(ts_val):
                # Heuristic: ms if > 10^11
                unit = "ms" if float(ts_val) > 1e11 else "s"
                ts = pd.to_datetime(ts_val, unit=unit, utc=True)
            else:
                ts = pd.to_datetime(ts_val, utc=True)
        except Exception:
            ts = None

    if ts is None or pd.isna(ts):
        # As a last resort, use now() to avoid hard crash, but signal caller to skip if desired
        return None

    try:
        ts = ts.tz_convert(_DEF_TZ)
    except Exception:
        # If ts was naive
        try:
            ts = ts.tz_localize("UTC").tz_convert(_DEF_TZ)
        except Exception:
            return None

    # Ensure OHLC
    def _fget(k, alt=None):
        if k in bar: return bar[k]
        if alt and alt in bar: return bar[alt]
        return None

    o = _fget("open"); h = _fget("high"); l = _fget("low"); c = _fget("close")
    # Common alternates (by some providers)
    if o is None: o = _fget("o")
    if h is None: h = _fget("h")
    if l is None: l = _fget("l")
    if c is None: c = _fget("c")

    try:
        o = float(o); h = float(h); l = float(l); c = float(c)
    except Exception:
        return None

    return {"ts": ts, "open": o, "high": h, "low": l, "close": c}

RUNS_DIR = Path(getattr(config, "runs_dir", "runs"))
SETUPS_CSV = RUNS_DIR / "setups.csv"
TRADES_CSV = RUNS_DIR / "trade_history.csv"
HEARTBEAT  = RUNS_DIR / "daemon_heartbeat.txt"

# Cache of completed setup IDs to avoid duplicate trade rows
_COMPLETED_IDS: set[str] = set()

def _load_completed_ids() -> None:
    """Populate _COMPLETED_IDS from existing trade_history, if present."""
    try:
        if TRADES_CSV.exists():
            df_ids = pd.read_csv(TRADES_CSV, usecols=["setup_id","outcome"], engine="python", on_bad_lines="skip")
            done = df_ids[df_ids["outcome"].isin(["target","stop","timeout"])]["setup_id"].astype(str).unique().tolist()
            _COMPLETED_IDS.update(done)
    except Exception:
        pass

# Canonical setups schema (dashboard writes these) - OLD VERSION
# _SETUP_FIELDS = [
#     "id","asset","interval","direction","entry","stop","target","rr",
#     "size_units","notional_usd","leverage",
#     "created_at","expires_at","status","confidence","trigger_rule","entry_buffer_bps",
#     "origin","trigger_ts","trigger_price",
# ]

def _ensure_setup_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Add any missing columns with NaNs and reorder
    # Use the new _SETUP_FIELDS from the patch
    setup_fields = [
        "id","asset","interval","direction","entry","stop","target","rr",
        "size_units","notional_usd","leverage",
        "created_at","expires_at","status","confidence","trigger_rule","entry_buffer_bps",
        "origin"
    ]
    for col in setup_fields:
        if col not in df.columns:
            df[col] = pd.NA
    # Keep only known fields to avoid parse explosions
    df = df[setup_fields]
    return df


def _tg_escape_md2(text: str) -> str:
    """
    Escape MarkdownV2 special chars with a single backslash.
    """
    import re
    # Telegram MarkdownV2 special chars
    pattern = r'([_*\[\]()~`>#+\-=|{}.!])'
    return re.sub(pattern, r'\\\1', str(text))


def _tg_send(msg: str, timeout: int = 6) -> None:
    token = os.getenv("TG_BOT_TOKEN") or os.getenv("TG_BOT") or ""
    chat  = os.getenv("TG_CHAT_ID")  or os.getenv("TG_CHAT")  or ""
    if not token or not chat:
        return
    try:
        safe = _tg_escape_md2(msg)
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": safe, "parse_mode": "MarkdownV2"},
            timeout=timeout,
        )
        r.raise_for_status()
    except Exception:
        pass


def _read_setups() -> pd.DataFrame:
    # Use the new _load_setups_df function from the patch
    return _load_setups_df()

def _write_setups(df: pd.DataFrame):
    # Use the new _save_setups_df function from the patch
    _save_setups_df(df)

def _append_trade_row(row: Dict):
    """
    Append a trade row safely:
    - Avoid duplicate completion rows per setup_id
    - Enforce a stable column schema across the CSV (auto-upgrade if new columns appear)
    - Compute pnl_pct_net after fees if config.fees_bps_per_side is set
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # De-dup completed rows for same setup
    sid = str(row.get("setup_id", ""))
    outc = str(row.get("outcome", ""))
    if sid and outc in {"target","stop","timeout"}:
        if sid in _COMPLETED_IDS:
            return  # already recorded

    # Fees & net PnL
    try:
        fees_bps_side = float(getattr(config, "fees_bps_per_side", 4.0))  # default 4 bps per side
    except Exception:
        fees_bps_side = 4.0
    # Convert any Timestamp to ISO in MY tz & type-normalize
    norm: Dict[str, object] = {}
    for k, v in row.items():
        if isinstance(v, (pd.Timestamp,)):
            vv = v
            if vv.tzinfo is None:
                vv = vv.tz_localize(_DEF_TZ)
            norm[k] = vv.isoformat()
        else:
            norm[k] = v

    # Ensure required derived fields
    pnl_pct = float(norm.get("pnl_pct", 0.0) or 0.0)
    # two sides taker by default
    pnl_pct_net = pnl_pct - (2.0 * fees_bps_side) / 100.0
    norm.setdefault("fees_bps_per_side", fees_bps_side)
    norm.setdefault("pnl_pct_net", pnl_pct_net)

    # Try to keep a stable header order
    core_order = [
        "setup_id","asset","interval","direction",
        "created_at","trigger_ts","entry","stop","target",
        "exit_ts","exit_price","outcome","pnl_pct","pnl_pct_net",
        "rr_planned","confidence","size_units","notional_usd","leverage",
        "fees_bps_per_side","price_at_trigger","trigger_rule","entry_buffer_bps"
    ]

    # If file exists, upgrade schema if needed
    if TRADES_CSV.exists():
        try:
            old_hdr = pd.read_csv(TRADES_CSV, nrows=0, engine="python", on_bad_lines="skip").columns.tolist()
        except Exception:
            old_hdr = []
        new_cols = list(dict.fromkeys(core_order + list(norm.keys())))
        # If headers differ, rewrite file with unified schema
        if set(new_cols) != set(old_hdr):
            try:
                old_df = pd.read_csv(TRADES_CSV, engine="python", on_bad_lines="skip")
            except Exception:
                old_df = pd.DataFrame(columns=old_hdr)
            # Align
            for c in new_cols:
                if c not in old_df.columns:
                   old_df[c] = pd.NA
            old_df = old_df[new_cols]
            tmp_full = TRADES_CSV.with_suffix(".full.tmp")
            old_df.to_csv(tmp_full, index=False)
            tmp_full.replace(TRADES_CSV)
            header = new_cols
        else:
            header = old_hdr
    else:
        header = core_order

    # Build single-row DF in final column order
    # Ensure all header cols exist in this row
    for c in header:
        norm.setdefault(c, pd.NA)
    df_row = pd.DataFrame([{c: norm.get(c, pd.NA) for c in header}])

    # Append
    write_header = not TRADES_CSV.exists()
    mode = "a"
    if write_header:
        df_row.to_csv(TRADES_CSV, mode=mode, index=False, header=True)
    else:
        df_row.to_csv(TRADES_CSV, mode=mode, index=False, header=False)

    # Update dedup cache
    if sid and outc in {"target","stop","timeout"}:
        _COMPLETED_IDS.add(sid)

def _hb():
    """
    Write a heartbeat timestamp in Malaysian time ISO format.
    """
    runs = _runs_dir()
    ts_iso = _iso_local()  # Use Malaysian time for heartbeat
    (runs / "daemon_heartbeat.txt").write_text(ts_iso)

# ---------------------------------------------------------------------------
# Any signal/setup/trade logging helpers should use the safe appender above
# so logs are consistent for training-from-logs.
# ---------------------------------------------------------------------------

# Canonical setups schema (aligned with dashboard)
_SETUP_FIELDS = [
    "id","asset","interval","direction","entry","stop","target","rr",
    "size_units","notional_usd","leverage",
    "created_at","expires_at","status","confidence","trigger_rule","entry_buffer_bps",
    "origin"
]

def _load_setups_df() -> pd.DataFrame:
    p = _runs_dir() / "setups.csv"
    if not p.exists():
        return pd.DataFrame(columns=_SETUP_FIELDS)
    try:
        df = pd.read_csv(p, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(columns=_SETUP_FIELDS)
    # Normalize columns
    for c in _SETUP_FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[_SETUP_FIELDS].copy()
    for c in ("created_at","expires_at"):
        if c in df.columns:
            # Parse as UTC to avoid mixed-tz FutureWarning, then convert to local
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            try:
                df[c] = ts.dt.tz_convert(_DEF_TZ)
            except Exception:
                # If conversion fails, keep as UTC to preserve tz-awareness
                df[c] = ts
    return df

def _save_setups_df(df: pd.DataFrame) -> None:
    p = _runs_dir() / "setups.csv"
    # retain only canonical schema in the file for cleanliness
    for c in _SETUP_FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    df[_SETUP_FIELDS].to_csv(p, index=False)

def _auto_cap_ok(asset: str, interval: str) -> bool:
    """
    Enforce daily cap for AUTO-origin setups only.
    Env var MAX_SETUPS_PER_DAY (int, default 2). 0 disables cap.
    """
    try:
        cap = int(os.environ.get("MAX_SETUPS_PER_DAY", "2"))
    except Exception:
        cap = 2
    if cap <= 0:
        return True
    df = _load_setups_df()
    if df.empty:
        return True
    now_local = _now_local()
    since = now_local - pd.Timedelta(days=1)
    mask = (
        (df["asset"] == asset) &
        (df["interval"] == interval) &
        (df["origin"].fillna("auto") == "auto") &
        (
            pd.to_datetime(df["created_at"], errors="coerce", utc=True)
              .dt.tz_convert(MY_TZ)
              >= since
        ) &
        (df["status"].isin(["pending","triggered","target","stop","timeout","cancelled"]))
    )
    n = int(mask.sum())
    return n < cap

def _validate_setup_entry(asset: str, interval: str, entry: float, direction: str) -> bool:
    """
    Validate that entry price is reasonable.
    Returns True if valid, False if entry should be rejected.
    """
    try:
        from src.data.price_feed import get_latest_candle
        raw_bar = get_latest_candle(asset, interval)
        if raw_bar is None:
            return True  # Skip validation if can't get price data
        
        current_price = float(raw_bar.get("close", 0))
        if current_price <= 0:
            return True  # Skip validation if invalid price
        
        price_diff_pct = abs(entry - current_price) / current_price * 100.0
        
        # Reject if entry is more than 50% away from current price (unrealistic)
        if price_diff_pct > 50.0:
            print(f"[setup_validation] Rejected {asset} {interval} {direction}: entry {entry:.2f} too far from current {current_price:.2f} ({price_diff_pct:.3f}%)")
            return False
        
        return True
    except Exception as e:
        print(f"[setup_validation] Error validating setup: {e}")
        return True  # Skip validation on error

def append_setup_auto(row: Dict[str, Any]) -> bool:
    """
    Append an AUTO setup row after applying daily cap.
    Ensures 'origin' is set to 'auto' and required fields exist.
    Returns True if appended, False if blocked by cap.
    """
    if not _auto_cap_ok(str(row.get("asset","")), str(row.get("interval",""))):
        return False
    
    # Validate entry price
    asset = str(row.get("asset", ""))
    interval = str(row.get("interval", ""))
    entry = float(row.get("entry", 0))
    direction = str(row.get("direction", ""))
    
    if not _validate_setup_entry(asset, interval, entry, direction):
        return False
    
    safe = dict(row)
    safe["origin"] = "auto"
    # Timestamps
    if "created_at" not in safe or pd.isna(safe.get("created_at")):
        safe["created_at"] = _now_local().isoformat()
    # Persist through setups.csv, preserving schema
    df = _load_setups_df()
    # Align to schema; missing fields get NaN
    for c in _SETUP_FIELDS:
        if c not in safe:
            safe[c] = np.nan
    # Append and save
    df = pd.concat([df, pd.DataFrame([safe])], ignore_index=True)
    _save_setups_df(df)
    return True

# ---- Signal/trade logging helpers (ensure 'ts') ----------------------------
def log_signal(row: Dict[str, Any]) -> None:
    p = _runs_dir() / "signals.csv"
    _append_csv_row(p, row)

def log_trade_history(row: Dict[str, Any]) -> None:
    p = _runs_dir() / "trade_history.csv"
    _append_csv_row(p, row)

def _trigger_touch(rule: str, buf_bps: float, direction: str, entry: float, bar: Dict) -> bool:
    low, high, close = float(bar["low"]), float(bar["high"]), float(bar["close"])
    
    if direction == "long":
        touched = low <= entry
        if rule == "touch":
            return touched
        return touched and (close >= entry * (1 + buf_bps/10000.0))
    if direction == "short":
        touched = high >= entry
        if rule == "touch":
            return touched
        return touched and (close <= entry * (1 - buf_bps/10000.0))
    return False


def _exit_hit(direction: str, stop: float, target: float, bar: Dict) -> Optional[str]:
    high, low = float(bar["high"]), float(bar["low"])
    if direction == "long":
        hit_stop = low <= stop
        hit_tgt  = high >= target
        if hit_stop and hit_tgt:
            return "stop"  # conservative first-touch
        if hit_tgt:  return "target"
        if hit_stop: return "stop"
        return None
    if direction == "short":
        hit_stop = high >= stop
        hit_tgt  = low  <= target
        if hit_stop and hit_tgt:
            return "stop"
        if hit_tgt:  return "target"
        if hit_stop: return "stop"
        return None
    return None

def _calc_pnl_pct(direction: str, entry: float, exit_px: float) -> float:
    if direction == "long":
        return (exit_px - entry) / max(entry, 1e-9) * 100.0
    if direction == "short":
        return (entry - exit_px) / max(entry, 1e-9) * 100.0
    return 0.0

def track_loop(symbol_default="BTCUSDT", interval_default="5m", sleep_seconds=15):
    print("alpha12_24 tracker started.")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    global _STOP
    while True and not _STOP:
        try:
            _hb()
            df = _read_setups()
            if df.empty:
                time.sleep(max(3, int(sleep_seconds)))
                continue

            watch = df[df["status"].isin(["pending","triggered","executed"])].copy()
            if watch.empty:
                time.sleep(max(3, int(sleep_seconds)))
                continue

            # group to avoid duplicate fetches
            for (asset, iv), group in watch.groupby(["asset","interval"]):
                raw_bar = get_latest_candle(asset, iv)
                bar = _normalize_bar(raw_bar)
                if not bar:
                    # couldn't parse the provider response for this symbol/interval; skip this group iteration
                    continue
                bar_ts = bar["ts"]

                for idx in group.index:
                    row = df.loc[idx]
                    status = str(row["status"]) if pd.notna(row["status"]) else "pending"

                    # expiry check for pending
                    if status == "pending":
                        exp = pd.to_datetime(row["expires_at"], errors="coerce", utc=True)
                        try:
                            exp = exp.tz_convert(_DEF_TZ)
                        except Exception:
                            pass
                        if bar_ts > exp:
                            df.loc[idx, "status"] = "expired"
                            _append_trade_row({
                                "setup_id": row["id"],
                                "asset": asset, "interval": iv, "direction": row["direction"],
                                "created_at": row["created_at"],
                                "exit_ts": bar_ts, "exit_price": float(bar["close"]),
                                "entry": float(row.get("entry", float("nan"))),
                                "stop": float(row.get("stop", float("nan"))),
                                "target": float(row.get("target", float("nan"))),
                                "outcome": "timeout",
                                "pnl_pct": 0.0,
                                "rr_planned": float(row.get("rr", 0.0)),
                                "confidence": float(row.get("confidence", 0.0)),
                                "size_units": row.get("size_units", pd.NA),
                                "notional_usd": row.get("notional_usd", pd.NA),
                                "leverage": row.get("leverage", pd.NA),
                                "trigger_rule": row.get("trigger_rule", pd.NA),
                                "entry_buffer_bps": row.get("entry_buffer_bps", pd.NA),
                            })
                            # Convert to Malaysian time for Telegram alert
                            bar_ts_my = bar_ts.tz_convert(MY_TZ) if bar_ts.tz is not None else bar_ts.tz_localize('UTC').tz_convert(MY_TZ)
                            _tg_send(f"Setup EXPIRED {asset} {iv}\\nEntry: {float(row.get('entry', 0)):.2f}\\nExpired at: {bar_ts_my.strftime('%Y-%m-%d %H:%M:%S')} MY")
                            continue

                        rule = str(row.get("trigger_rule", "touch"))
                        buf  = float(row.get("entry_buffer_bps", 5.0))
                        
                        # Check if setup was created before the current candle started
                        # This prevents triggering on price movements that happened before setup creation
                        setup_created_at = pd.to_datetime(row["created_at"], errors="coerce", utc=True)
                        try:
                            setup_created_at = setup_created_at.tz_convert(_DEF_TZ)
                        except Exception:
                            pass
                        
                        # Only check trigger if setup was created before the current candle started
                        # This prevents false triggers from historical price movements
                        if pd.notna(setup_created_at) and pd.notna(bar_ts):
                            # Calculate candle duration based on interval
                            interval_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(iv, 5)
                            candle_start = bar_ts - pd.Timedelta(minutes=interval_minutes)
                            
                            # Skip trigger check if setup was created during this candle
                            # This prevents triggering on price movements that happened before setup creation
                            if setup_created_at >= candle_start:
                                continue
                        
                        if _trigger_touch(rule, buf, str(row["direction"]), float(row["entry"]), bar):
                            df.loc[idx, "status"] = "triggered"
                            df.loc[idx, "trigger_ts"] = bar_ts
                            df.loc[idx, "trigger_price"] = float(bar["close"])  # conservative
                            if pd.isna(row.get("trigger_price")) and "trigger_price" in df.columns:
                                df.loc[idx, "price_at_trigger"] = float(bar["close"])  # alias field for logs
                            # Convert to Malaysian time for Telegram alert
                            bar_ts_my = bar_ts.tz_convert(MY_TZ) if bar_ts.tz is not None else bar_ts.tz_localize('UTC').tz_convert(MY_TZ)
                            # Check if this is an executed setup
                            execution_type = ""
                            if row.get('status') == 'executed':
                                execution_type = " (ACTIVE EXECUTED)"
                            
                            _tg_send(f"Setup TRIGGERED{execution_type} {asset} {iv} ({row['direction'].upper()})\\nEntry: {row['entry']:.2f} → Triggered @ {float(bar['close']):.2f}\\nStop: {row['stop']:.2f} | Target: {row['target']:.2f}\\nTime: {bar_ts_my.strftime('%Y-%m-%d %H:%M:%S')} MY")
                            continue

                    # Check for trigger on executed setups that haven't been triggered yet
                    if status == "executed":
                        rule = str(row.get("trigger_rule", "touch"))
                        buf = float(row.get("entry_buffer_bps", 5.0))
                        
                        if _trigger_touch(rule, buf, str(row["direction"]), float(row["entry"]), bar):
                            df.loc[idx, "status"] = "triggered"
                            df.loc[idx, "trigger_ts"] = bar_ts
                            df.loc[idx, "trigger_price"] = float(bar["close"])  # conservative
                            if pd.isna(row.get("trigger_price")) and "trigger_price" in df.columns:
                                df.loc[idx, "price_at_trigger"] = float(bar["close"])  # alias field for logs
                            # Convert to Malaysian time for Telegram alert
                            bar_ts_my = bar_ts.tz_convert(MY_TZ) if bar_ts.tz is not None else bar_ts.tz_localize('UTC').tz_convert(MY_TZ)
                            # Check if this is an executed setup
                            execution_type = " (ACTIVE EXECUTED)"
                            
                            _tg_send(f"Setup TRIGGERED{execution_type} {asset} {iv} ({row['direction'].upper()})\\nEntry: {row['entry']:.2f} → Triggered @ {float(bar['close']):.2f}\\nStop: {row['stop']:.2f} | Target: {row['target']:.2f}\\nTime: {bar_ts_my.strftime('%Y-%m-%d %H:%M:%S')} MY")
                            continue
                    
                    if status == "triggered" or status == "executed":
                        trig_ts = pd.to_datetime(row.get("trigger_ts"), errors="coerce", utc=True)
                        if pd.isna(trig_ts):
                            trig_ts = bar_ts
                        else:
                            try:
                                trig_ts = trig_ts.tz_convert(_DEF_TZ)
                            except Exception:
                                pass
                        exp = pd.to_datetime(row["expires_at"], errors="coerce", utc=True)
                        try:
                            exp = exp.tz_convert(_DEF_TZ)
                        except Exception:
                            pass
                        end_ts = min(exp, bar_ts)

                        try:
                            win = get_window(asset, iv, trig_ts, end_ts)
                        except Exception:
                            win = None

                        # Check all bars since trigger for exit detection
                        outcome = None
                        exit_px, exit_ts = None, None
                        
                        if win is not None and not win.empty:
                            # Check each bar in the window for exit hits
                            for _, win_bar in win.iterrows():
                                win_bar_dict = win_bar.to_dict()
                                win_bar_dict["ts"] = win_bar.name  # Ensure timestamp is available
                                win_outcome = _resolve_exit(
                                    str(row["direction"]),
                                    float(row["entry"]),
                                    float(row["stop"]),
                                    float(row["target"]),
                                    win_bar_dict
                                )
                                if win_outcome is not None:
                                    outcome = win_outcome
                                    exit_ts = win_bar.name
                                    exit_px = float(row["target"]) if outcome == "target" else float(row["stop"])
                                    break
                        
                        # If no exit found in window, check current bar
                        if outcome is None:
                            outcome = _resolve_exit(
                                str(row["direction"]),
                                float(row["entry"]),
                                float(row["stop"]),
                                float(row["target"]),
                                bar
                            )
                            if outcome is not None:
                                exit_ts = bar_ts
                                exit_px = float(row["target"]) if outcome == "target" else float(row["stop"])

                        # timeout if reached expiry with no hit
                        if outcome is None and end_ts >= exp:
                            outcome = "timeout"
                            exit_ts = end_ts
                            if win is not None and not win.empty and "close" in win.columns:
                                exit_px = float(win["close"].iloc[-1])
                            else:
                                exit_px = float(bar.get("close", float("nan")))

                        if outcome is not None:
                            entry_px = float(row.get("entry", float("nan")))
                            pnl_pct = _calc_pnl_pct(str(row["direction"]), entry_px, float(exit_px)) if np.isfinite(entry_px) else 0.0
                            _append_trade_row({
                                "setup_id": row["id"],
                                "asset": asset, "interval": iv, "direction": row["direction"],
                                "created_at": row["created_at"], "trigger_ts": trig_ts,
                                "entry": entry_px, "stop": float(row["stop"]), "target": float(row["target"]),
                                "exit_ts": exit_ts, "exit_price": float(exit_px), "outcome": outcome,
                                "pnl_pct": float(pnl_pct), "rr_planned": float(row.get("rr", 0.0)),
                                "confidence": float(row.get("confidence", 0.0)),
                                "size_units": row.get("size_units", pd.NA),
                                "notional_usd": row.get("notional_usd", pd.NA),
                                "leverage": row.get("leverage", pd.NA),
                                "price_at_trigger": float(row.get("trigger_price", float("nan"))),
                                "trigger_rule": row.get("trigger_rule", pd.NA),
                                "entry_buffer_bps": row.get("entry_buffer_bps", pd.NA),
                            })
                            df.loc[idx, "status"] = outcome  # target/stop/timeout
                            outcome_emoji = {"target": "✅", "stop": "❌", "timeout": "⏰"}
                            emoji = outcome_emoji.get(outcome, "📊")
                            # Convert to Malaysian time for Telegram alert
                            exit_ts_my = exit_ts.tz_convert(MY_TZ) if exit_ts.tz is not None else exit_ts.tz_localize('UTC').tz_convert(MY_TZ)
                            # Check if this was an executed setup
                            execution_type = ""
                            if row.get('status') == 'executed':
                                execution_type = " (EXECUTED)"
                            
                            _tg_send(f"Setup {outcome.upper()}{execution_type} {asset} {iv}\\nEntry: {entry_px:.2f} → Exit: {float(exit_px):.2f}\\nPnL: {float(pnl_pct):.2f}%\\nTime: {exit_ts_my.strftime('%Y-%m-%d %H:%M:%S')} MY")

            # Save updated setups with proper schema
            _save_setups_df(df)

            # Sleep inside try so Ctrl+C is caught here
            time.sleep(max(3, int(sleep_seconds)))

        except KeyboardInterrupt:
            print("[tracker] stopping by user (SIGINT)")
            break
        except Exception as e:
            print("tracker error:", e)
            # brief backoff on error to avoid hot-looping
            try:
                time.sleep(max(3, int(sleep_seconds)))
            except Exception:
                pass


if __name__ == "__main__":
    # env overrides
    SYM = os.getenv("ALPHA12_SYMBOL", "BTCUSDT")
    IV  = os.getenv("ALPHA12_INTERVAL", "5m")
    SLP = int(os.getenv("ALPHA12_SLEEP", "15"))

    # One-shot heartbeat write so ops can confirm process ownership quickly
    _hb()
    print(f"[tracker] heartbeat written to {HEARTBEAT}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _sig_stop)
    try:
        signal.signal(signal.SIGTERM, _sig_stop)
    except Exception:
        # Some environments (e.g., Windows) may not have SIGTERM
        pass

    _load_completed_ids()
    print(f"[tracker] loaded {len(_COMPLETED_IDS)} completed setup IDs")

    # Start loop
    try:
        track_loop(SYM, IV, SLP)
    finally:
        try:
            _hb()
            print("[tracker] exited cleanly")
        except Exception:
            pass
