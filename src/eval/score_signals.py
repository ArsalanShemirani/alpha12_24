
#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional import: used by score_from_logs
try:
    from .from_logs import load_joined
except Exception:
    load_joined = None  # will be checked at call-site

MY_TZ = "Asia/Kuala_Lumpur"

# ---------------------------------------------------------------------------
# Robust merge helpers (signals + feats) and (signals+feats + trades)
# ---------------------------------------------------------------------------

def prepare_merge(signals: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """As-of join features to signals (backward within 2h), leak-safe.
    Requires columns: ['asset','interval','ts'] after normalization.
    """
    if signals is None or feats is None:
        return pd.DataFrame()

    # Ensure required columns exist and are correctly typed
    for df_ in (signals, feats):
        if "ts" not in df_.columns:
            df_["ts"] = pd.NaT
        # Drop NaT because merge_asof requires sorted, valid keys
        df_.dropna(subset=["ts"], inplace=True)
        # Ensure join-by columns exist
        for col in ("asset", "interval"):
            if col not in df_.columns:
                df_[col] = ""
            else:
                df_[col] = df_[col].astype(str)

    # Sort strictly by keys + time using a stable algorithm
    s_left = signals.sort_values(["asset", "interval", "ts"], kind="mergesort").reset_index(drop=True)
    s_right = feats.sort_values(["asset", "interval", "ts"], kind="mergesort").reset_index(drop=True)

    sf = pd.merge_asof(
        s_left,
        s_right,
        on="ts",
        by=["asset", "interval"],
        direction="backward",
        tolerance=pd.Timedelta("2h"),
        suffixes=("", "_feat"),
    )
    return sf


def merge_signals_with_trades(signals: pd.DataFrame, feats: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Merge signals with features and trade outcomes.
    Falls back to nearest-forward match by (asset, interval) within 48h.
    """
    sf = prepare_merge(signals, feats)
    if sf.empty or trades is None or trades.empty:
        return sf

    # Clean time columns for fallback asof
    sf2 = sf.copy()
    if "ts" not in sf2.columns:
        sf2["ts"] = pd.NaT
    sf2.dropna(subset=["ts"], inplace=True)
    for col in ("asset", "interval"):
        if col not in sf2.columns:
            sf2[col] = ""
        else:
            sf2[col] = sf2[col].astype(str)

    tr = trades.copy()
    if "exit_ts" not in tr.columns:
        tr["exit_ts"] = pd.NaT
    tr.dropna(subset=["exit_ts"], inplace=True)
    for col in ("asset", "interval"):
        if col not in tr.columns:
            tr[col] = ""
        else:
            tr[col] = tr[col].astype(str)

    # Sort both sides by by-keys + time for merge_asof
    left_sorted = sf2.sort_values(["asset", "interval", "ts"], kind="mergesort").reset_index(drop=True)
    right_sorted = tr.sort_values(["asset", "interval", "exit_ts"], kind="mergesort").reset_index(drop=True)

    out = pd.merge_asof(
        left_sorted,
        right_sorted[[
            "asset", "interval", "exit_ts", "outcome", "pnl_pct", "pnl_pct_net",
            "entry", "stop", "target", "rr_planned", "setup_id"
        ] if "setup_id" in tr.columns else [
            "asset", "interval", "exit_ts", "outcome", "pnl_pct", "pnl_pct_net",
            "entry", "stop", "target", "rr_planned"
        ]],
        left_on="ts",
        right_on="exit_ts",
        by=["asset", "interval"],
        direction="forward",
        tolerance=pd.Timedelta("48h"),
    )

    out = out.sort_values(["asset", "interval", "ts"], kind="mergesort").drop_duplicates(
        subset=[c for c in ["asset", "interval", "ts", "setup_id"] if c in out.columns], keep="last"
    )
    return out

# ---------------------------------------------------------------------------
# Lightweight online scorer (compatibility with earlier UI flows)
# ---------------------------------------------------------------------------

@dataclass
class SignalScore:
    timestamp: datetime
    signal: str
    confidence: float
    prob_up: float
    prob_down: float
    actual_return: float
    predicted_return: float
    score: float
    metadata: Dict


class SignalScorer:
    def __init__(self, config=None):
        self.config = config
        self.scores: List[SignalScore] = []

    def score_signal(
        self,
        signal: str,
        confidence: float,
        prob_up: float,
        prob_down: float,
        current_price: float,
        future_price: float,
        horizon_hours: int = 24,
        metadata: Optional[Dict] = None,
    ) -> SignalScore:
        # realized return
        try:
            actual_return = (float(future_price) - float(current_price)) / max(float(current_price), 1e-9)
        except Exception:
            actual_return = 0.0

        # simple expected return heuristic (directional)
        if (signal or "").upper() == "LONG":
            predicted_return = float(prob_up) * 0.02 - float(prob_down) * 0.01
        elif (signal or "").upper() == "SHORT":
            predicted_return = float(prob_down) * 0.02 - float(prob_up) * 0.01
        else:
            predicted_return = 0.0

        score = self._score_formula(signal, float(confidence), float(actual_return), float(predicted_return))

        ss = SignalScore(
            timestamp=datetime.now(),
            signal=str(signal),
            confidence=float(confidence),
            prob_up=float(prob_up),
            prob_down=float(prob_down),
            actual_return=float(actual_return),
            predicted_return=float(predicted_return),
            score=float(score),
            metadata=metadata or {},
        )
        self.scores.append(ss)
        return ss

    @staticmethod
    def _score_formula(signal: str, confidence: float, actual_return: float, predicted_return: float) -> float:
        base = float(confidence)
        # direction bonus
        sig = (signal or "").upper()
        if sig == "LONG" and actual_return > 0:
            dir_bonus = 0.2
        elif sig == "SHORT" and actual_return < 0:
            dir_bonus = 0.2
        elif sig == "HOLD" and abs(actual_return) < 0.01:
            dir_bonus = 0.1
        else:
            dir_bonus = -0.1
        # magnitude bonus (soft)
        mag_err = abs(actual_return - predicted_return)
        mag_bonus = max(0.0, 0.1 - mag_err)
        x = base + dir_bonus + mag_bonus
        return float(max(0.0, min(1.0, x)))

# ---------------------------------------------------------------------------
# Log-based evaluation and calibration
# ---------------------------------------------------------------------------

def _safe_prob_series(df: pd.DataFrame) -> pd.Series:
    """Choose a probability column for up-move if available; otherwise
    fallback to normalized confidence in [0,1]."""
    if "prob_up" in df.columns and df["prob_up"].notna().any():
        p = df["prob_up"].astype(float).clip(0.0, 1.0)
    else:
        if "confidence" in df.columns and df["confidence"].notna().any():
            c = df["confidence"].astype(float)
            if c.max() > 1.0:
                c = (c / 100.0).clip(0.0, 1.0)
            p = c
        else:
            p = pd.Series(np.nan, index=df.index)
    return p


def compute_wr_pf_expectancy(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n": 0.0, "wr": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
        "pf": np.nan, "expectancy": np.nan, "brier": np.nan,
    }
    if df is None or df.empty:
        return out

    d = df.copy()
    if "y" not in d.columns:
        if "outcome" in d.columns:
            y = np.where(d["outcome"].eq("target"), 1, np.where(d["outcome"].eq("stop"), 0, np.nan))
            d["y"] = y
        else:
            return out

    # brier score if we have probabilities
    p = _safe_prob_series(d)
    if p.notna().any():
        try:
            mask = d["y"].notna() & p.notna()
            if mask.any():
                y01 = d.loc[mask, "y"].astype(float)
                psc = p.loc[mask].astype(float).clip(0, 1)
                out["brier"] = float(np.mean((psc - y01) ** 2))
        except Exception:
            pass

    # choose realized pnl
    r = None
    if "pnl_pct_net" in d.columns and d["pnl_pct_net"].notna().any():
        r = d["pnl_pct_net"].astype(float)
    elif "pnl_pct" in d.columns and d["pnl_pct"].notna().any():
        r = d["pnl_pct"].astype(float)

    yy = d["y"].dropna()
    out["n"] = float(len(yy))
    if len(yy) == 0:
        return out
    out["wr"] = float(np.mean(yy))

    if r is not None and r.notna().any():
        rr = r.loc[yy.index]
        wins = rr[yy == 1]
        losses = rr[yy == 0]
        if len(wins) > 0:
            out["avg_win"] = float(wins.mean())
        if len(losses) > 0:
            out["avg_loss"] = float(losses.mean())
        gross_win = float(wins[wins > 0].sum()) if len(wins) else 0.0
        gross_loss = float(np.abs(losses[losses < 0].sum())) if len(losses) else 0.0
        out["pf"] = (gross_win / gross_loss) if gross_loss > 1e-12 else np.nan
        out["expectancy"] = float(rr.mean())

    return out


def reliability_curve(df: pd.DataFrame, prob_col: str = "prob_up", n_bins: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["bin", "p_mean", "y_rate", "count"]).assign(bin=pd.Categorical([]))

    d = df.copy()
    if prob_col not in d.columns or d[prob_col].isna().all():
        d[prob_col] = _safe_prob_series(d)

    mask = d["y"].notna() & d[prob_col].notna()
    if not mask.any():
        return pd.DataFrame(columns=["bin", "p_mean", "y_rate", "count"]).assign(bin=pd.Categorical([]))

    x = d.loc[mask, prob_col].clip(0, 1)
    y = d.loc[mask, "y"].astype(float)
    bins = pd.cut(x, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grp = pd.DataFrame({"p": x, "y": y, "bin": bins}).groupby("bin", observed=False)
    out = grp.agg(p_mean=("p", "mean"), y_rate=("y", "mean"), count=("y", "size")).reset_index()
    return out


def confusion_like(df: pd.DataFrame, prob_col: str = "prob_up", thr: float = 0.5) -> Dict[str, int]:
    out = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    if df is None or df.empty:
        return out
    d = df.copy()
    if "y" not in d.columns or d["y"].isna().all():
        return out

    p = d[prob_col] if prob_col in d.columns else _safe_prob_series(d)
    p = p.fillna(0.5)
    # Handle NaN values in y before converting to int
    y = d["y"].astype(float).fillna(0.0).round().astype(int)

    pred = (p >= float(thr)).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    TN = int(((pred == 0) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

# ---------------------------------------------------------------------------
# High-level scoring entry point used by the dashboard
# ---------------------------------------------------------------------------

def score_from_logs(runs_dir: str = "runs") -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load joined logs and compute:
      - core metrics (wr, pf, expectancy, brier)
      - calibration table (10 bins)
      - compact per-trade table (ts, asset, interval, outcome, y, prob_up, pnl_pct_net)
      - confusion-like counts at 0.5 threshold
    Returns: (metrics_dict, calib_df, trades_small_df, confusion)
    """
    if load_joined is None:
        return ({}, pd.DataFrame(), pd.DataFrame(), {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

    df = load_joined(runs_dir=runs_dir)
    if df is None or df.empty:
        return ({}, pd.DataFrame(), pd.DataFrame(), {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

    metrics = compute_wr_pf_expectancy(df)
    calib = reliability_curve(df, prob_col="prob_up", n_bins=10)

    small_cols = [c for c in [
        "ts", "asset", "interval", "setup_id", "outcome", "y", "prob_up",
        "confidence", "pnl_pct_net", "pnl_pct", "rr_planned"
    ] if c in df.columns]
    small = df[small_cols].copy().sort_values("ts").reset_index(drop=True)

    conf = confusion_like(df, prob_col="prob_up", thr=0.5)
    return metrics, calib, small, conf


if __name__ == "__main__":
    m, calib, small, conf = score_from_logs("runs")
    print("Core metrics:", m)
    print("Confusion-like:", conf)
    print("Calibration (head):\n", calib.head())
    print("Sample trades (tail):\n", small.tail(10))
