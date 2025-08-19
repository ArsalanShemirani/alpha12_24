from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MY_TZ = "Asia/Kuala_Lumpur"


# ---- merge_asof preparation helper ----

def _asof_ready(df: pd.DataFrame, time_col: str, by_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    # ensure time col exists
    if time_col not in d.columns:
        d[time_col] = pd.NaT
    # force UTC-aware datetime
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
    # ensure by keys
    for c in by_cols:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].astype(str)
    # drop NaT
    d = d.dropna(subset=[time_col])
    # stable sort by by_cols + time_col
    d = d.sort_values(by_cols + [time_col], kind="mergesort").reset_index(drop=True)
    # extra guard: enforce monotonic time within groups
    try:
        bad = (
            d.groupby(by_cols, sort=False)[time_col]
             .apply(lambda s: (s.diff().dt.total_seconds().fillna(0) < 0).any())
        )
        if bool(getattr(bad, "any", lambda: False)()):
            d = d.sort_values(by_cols + [time_col], kind="mergesort").reset_index(drop=True)
    except Exception:
        # best effort; sorting above usually suffices
        pass
    return d


def _asof_join_grouped(
    left: pd.DataFrame,
    right: pd.DataFrame,
    time_col_left: str,
    time_col_right: Optional[str] = None,
    by_cols: List[str] = ["asset", "interval"],
    direction: str = "backward",
    tolerance: Optional[pd.Timedelta] = None,
    suffixes: Tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    """Robust group-wise asof join. Ensures per-group monotonic sort and joins group-by-group.
    If a by-group is missing on the right, it returns left rows for that group with NaNs for right columns.
    """
    if left is None or left.empty:
        return pd.DataFrame()

    t_right = time_col_right or time_col_left

    # Prepare both frames
    L = _asof_ready(left, time_col_left, by_cols)
    R = _asof_ready(right if right is not None else pd.DataFrame(), t_right, by_cols)

    out_parts = []

    # iterate over left groups; align to right groups
    for gvals, lgrp in L.groupby(by_cols, sort=False):
        if not isinstance(gvals, tuple):
            gvals = (gvals,)
        # slice right group
        mask = np.ones(len(R), dtype=bool)
        for c, v in zip(by_cols, gvals):
            mask &= (R[c] == str(v))
        rgrp = R.loc[mask]
        if rgrp.empty:
            out_parts.append(lgrp.copy())
            continue
        # ensure sorted (safety)
        lgrp = lgrp.sort_values(time_col_left, kind="mergesort").reset_index(drop=True)
        rgrp = rgrp.sort_values(t_right, kind="mergesort").reset_index(drop=True)

        merged = pd.merge_asof(
            lgrp,
            rgrp,
            left_on=time_col_left,
            right_on=t_right,
            direction=direction,
            tolerance=tolerance,
            suffixes=suffixes,
        )
        out_parts.append(merged)

    if not out_parts:
        return pd.DataFrame()

    out = pd.concat(out_parts, ignore_index=True)
    # final stable sort/dedup
    out = out.sort_values(by_cols + [time_col_left], kind="mergesort").reset_index(drop=True)
    return out


# ---------------------- IO HELPERS ----------------------

def _runs_dir(runs_dir: Optional[str]) -> Path:
    return Path(runs_dir or "runs")


def _read_csv(
    p: Path,
    parse_time_cols: Optional[List[str]] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(
            p,
            engine="python",
            on_bad_lines="skip",
            usecols=usecols,
        )
    except Exception:
        df = pd.read_csv(p)  # last resort

    if parse_time_cols:
        for c in parse_time_cols:
            if c in df.columns:
                # Parse to UTC then convert to Malaysia time; avoid mixed-tz fallback
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                try:
                    s = s.dt.tz_convert(MY_TZ)
                except Exception:
                    # keep as UTC tz-aware if conversion fails
                    pass
                df[c] = s
    return df


def _read_parquet_or_csv(
    pq: Path,
    csv: Path,
    parse_time_cols: Optional[List[str]] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = _read_csv(csv, parse_time_cols=None, usecols=usecols)
    else:
        return pd.DataFrame()

    if parse_time_cols:
        for c in parse_time_cols:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                try:
                    s = s.dt.tz_convert(MY_TZ)
                except Exception:
                    pass
                df[c] = s
    return df


# ---------------------- CORE JOIN LOGIC ----------------------

def _ensure_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _dedup_latest(df: pd.DataFrame, key_cols: List[str], ts_col: str) -> pd.DataFrame:
    """Keep only the latest row per key using ts_col for ordering."""
    if df.empty:
        return df
    df = df.sort_values(ts_col).drop_duplicates(subset=key_cols, keep="last")
    return df


def load_joined(runs_dir: Optional[str] = "runs") -> pd.DataFrame:
    """
    Join logs with robust, timezone-safe logic:
      - signals.csv                 one row per model-issued signal
      - features_at_signal.(pq/csv) feature snapshot at signal time
      - trade_history.csv           trigger/target/stop/timeout outcomes

    Returns a dataframe suitable for ML: features + labels (pnl/outcome/confidence).
    """
    rd = _runs_dir(runs_dir)

    # --- Load pieces ---
    signals = _read_csv(
        rd / "signals.csv",
        parse_time_cols=["ts"],
    )  # ts is signal time

    feats = _read_parquet_or_csv(
        rd / "features_at_signal.parquet",
        rd / "features_at_signal.csv",
        parse_time_cols=["ts"],
    )  # ts matches signals (nearest-left join)

    trades = _read_csv(
        rd / "trade_history.csv",
        parse_time_cols=["created_at", "trigger_ts", "exit_ts"],
    )

    if signals.empty or feats.empty:
        return pd.DataFrame()

    # Normalize join keys
    for df in (signals, feats):
        for col in ("asset", "interval"):
            if col not in df.columns:
                df[col] = ""

    # Ensure types for ID linkage if present
    for df in (signals, feats, trades):
        _ensure_str(df, ["setup_id"]) if not df.empty else None

    # De-duplicate signals by (setup_id) or (asset, interval, ts)
    if "setup_id" in signals.columns and signals["setup_id"].notna().any():
        signals = _dedup_latest(signals, ["setup_id"], "ts")
    else:
        signals = _dedup_latest(signals, ["asset", "interval", "ts"], "ts")

    # --- As-of join features to signals (backward within 2h) ---
    # Normalize/clean join keys and time cols for merge_asof
    for df_ in (signals, feats):
        if "ts" not in df_.columns:
            df_["ts"] = pd.NaT
        # force UTC-aware datetime then convert to MY time for consistency
        df_["ts"] = pd.to_datetime(df_["ts"], errors="coerce", utc=True)
        try:
            df_["ts"] = df_["ts"].dt.tz_convert(MY_TZ)
        except Exception:
            pass
        # ensure by-keys exist and are strings
        for col in ("asset", "interval"):
            if col not in df_.columns:
                df_[col] = ""
            df_[col] = df_[col].astype(str)
        # drop invalid timestamps (required by merge_asof)
        df_.dropna(subset=["ts"], inplace=True)

    # stable-sort by by-keys + ts as required by merge_asof
    _left = _asof_ready(signals, "ts", ["asset", "interval"])
    _right = _asof_ready(feats, "ts", ["asset", "interval"])

    sf = _asof_join_grouped(
        left=_left,
        right=_right,
        time_col_left="ts",
        time_col_right="ts",
        by_cols=["asset", "interval"],
        direction="backward",
        tolerance=pd.Timedelta("2h"),
        suffixes=("", "_feat"),
    )

    # Outcome attachment
    if trades.empty:
        return sf

    # Best: exact setup_id linkage if present
    if "setup_id" in trades.columns and "setup_id" in sf.columns:
        out = sf.merge(
            trades[
                [
                    "setup_id",
                    "outcome",
                    "pnl_pct",
                    "pnl_pct_net",
                    "exit_ts",
                    "entry",
                    "stop",
                    "target",
                    "rr_planned",
                ]
            ],
            on="setup_id",
            how="left",
        )
    else:
        # Fallback: nearest-forward exit within 48h in same asset/interval
        sf2 = sf.copy()
        if "ts" not in sf2.columns:
            sf2["ts"] = pd.NaT
        sf2["ts"] = pd.to_datetime(sf2["ts"], errors="coerce", utc=True)
        try:
            sf2["ts"] = sf2["ts"].dt.tz_convert(MY_TZ)
        except Exception:
            pass
        for col in ("asset", "interval"):
            if col not in sf2.columns:
                sf2[col] = ""
            sf2[col] = sf2[col].astype(str)
        sf2.dropna(subset=["ts"], inplace=True)

        tr = trades.copy()
        if "exit_ts" not in tr.columns:
            tr["exit_ts"] = pd.NaT
        tr["exit_ts"] = pd.to_datetime(tr["exit_ts"], errors="coerce", utc=True)
        try:
            tr["exit_ts"] = tr["exit_ts"].dt.tz_convert(MY_TZ)
        except Exception:
            pass
        for col in ("asset", "interval"):
            if col not in tr.columns:
                tr[col] = ""
            tr[col] = tr[col].astype(str)
        tr.dropna(subset=["exit_ts"], inplace=True)

        left_sorted = _asof_ready(sf2, "ts", ["asset", "interval"])
        right_sorted = _asof_ready(tr, "exit_ts", ["asset", "interval"])

        # Build right-hand columns defensively (some are optional in older logs)
        base_cols = [
            "asset", "interval", "exit_ts", "outcome", "pnl_pct", "pnl_pct_net",
            "entry", "stop", "target", "rr_planned",
        ]
        if "setup_id" in tr.columns:
            base_cols.append("setup_id")
        keep_cols = [c for c in base_cols if c in right_sorted.columns]

        out = _asof_join_grouped(
            left=left_sorted,
            right=right_sorted[keep_cols],
            time_col_left="ts",
            time_col_right="exit_ts",
            by_cols=["asset", "interval"],
            direction="forward",
            tolerance=pd.Timedelta("48h"),
            suffixes=("", "_trade"),
        )

    # --- Post-process labels ---
    # Binary label y: 1 for target (win), 0 for stop (loss), NaN for others
    if "outcome" in out.columns:
        out["y"] = np.where(out["outcome"].eq("target"), 1,
                      np.where(out["outcome"].eq("stop"), 0, np.nan))

    # Confidence passthrough if missing: join on (asset, interval, ts) from original signals
    if "confidence" not in out.columns and "confidence" in signals.columns:
        # Ensure key types match and ts is UTC-aware
        for df_ in (out, signals):
            for col in ("asset", "interval"):
                if col not in df_.columns:
                    df_[col] = ""
                df_[col] = df_[col].astype(str)
            if "ts" not in df_.columns:
                df_["ts"] = pd.NaT
            df_["ts"] = pd.to_datetime(df_["ts"], errors="coerce", utc=True)
            try:
                df_["ts"] = df_["ts"].dt.tz_convert(MY_TZ)
            except Exception:
                pass
        key = ["asset", "interval", "ts"]
        conf_map = (
            signals[key + ["confidence"]]
            .dropna(subset=["ts"])  # required for merge keys
            .drop_duplicates(subset=key, keep="last")
        )
        out = out.merge(conf_map, on=key, how="left")

    # Sort by time, drop obvious duplicates again
    out = out.sort_values(["asset", "interval", "ts"], kind="mergesort").drop_duplicates(
        subset=[c for c in ["asset", "interval", "ts", "setup_id"] if c in out.columns], keep="last"
    )

    return out


# ---------------------- TRAINING-FRIENDLY WRAPPER ----------------------

def build_training_frame(
    runs_dir: Optional[str] = "runs",
    drop_na_label: bool = True,
    feature_prefixes: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Returns (X, y, feature_columns) built from `load_joined` output.
    - Selects only numeric features (and those with selected prefixes if provided).
    - Drops rows with NaN labels if drop_na_label=True.
    """
    df = load_joined(runs_dir=runs_dir)
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), []

    # Label
    y = df["y"] if "y" in df.columns else pd.Series(dtype=float)
    if drop_na_label and not y.empty:
        mask = y.notna()
        df = df.loc[mask].copy()
        y = y.loc[mask].astype(float)

    # Choose features: numeric cols except admin/meta
    meta_cols = {
        "y", "asset", "interval", "ts", "setup_id", "outcome",
        "exit_ts", "created_at", "trigger_ts", "direction",
        "entry", "stop", "target", "rr_planned", "pnl_pct", "pnl_pct_net",
        "price_at_trigger", "confidence"
    }

    num_df = df.select_dtypes(include=[np.number]).copy()

    if feature_prefixes:
        keep = [c for c in num_df.columns if any(c.startswith(p) for p in feature_prefixes)]
        X = num_df[keep].copy()
    else:
        X = num_df.drop(columns=[c for c in num_df.columns if c in meta_cols and c in num_df.columns], errors="ignore").copy()

    feats = list(X.columns)
    return X, y, feats
