from __future__ import annotations
import pandas as pd
import numpy as np

# Malaysia timezone constant
MY_TZ = "Asia/Kuala_Lumpur"

def _safe_pct(x): 
    try: return float(x)
    except: return np.nan

def load_trade_history(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["setup_id","asset","interval","direction","created_at","trigger_ts","entry","stop","target","exit_ts","exit_price","outcome","pnl_pct","rr_planned","confidence"])
    # parse times (ingest UTC then convert to Malaysia time)
    for c in ("created_at","trigger_ts","exit_ts"):
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            try:
                df[c] = ts.dt.tz_convert(MY_TZ)
            except Exception:
                df[c] = ts
    # numeric
    for c in ("pnl_pct","rr_planned","confidence","entry","exit_price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_metrics(df: pd.DataFrame, start=None, end=None) -> dict:
    # normalize filter bounds to Malaysia time
    if start is not None:
        start = pd.to_datetime(start, errors="coerce", utc=True)
        try:
            start = start.tz_convert(MY_TZ)
        except Exception:
            pass
    if end is not None:
        end = pd.to_datetime(end, errors="coerce", utc=True)
        try:
            end = end.tz_convert(MY_TZ)
        except Exception:
            pass
    if df.empty: 
        return {"trades":0, "winrate":0.0, "pf":0.0, "max_dd":0.0, "ret_pct":0.0}
    if start is not None: df = df[df["exit_ts"] >= start]
    if end   is not None: df = df[df["exit_ts"] <= end]
    # keep only completed outcomes
    df = df[df["outcome"].isin(["target","stop","timeout"])].copy()
    if df.empty:
        return {"trades":0, "winrate":0.0, "pf":0.0, "max_dd":0.0, "ret_pct":0.0}
    # per-trade return model: use pnl_pct (already %). Convert to decimal return per trade accounting for 1% risk per trade by default later in UI if needed.
    wins = df["pnl_pct"] > 0
    winrate = float(wins.mean())
    # profit factor: sum of gains / abs(sum of losses)
    gains = df.loc[wins,"pnl_pct"].sum()
    losses = df.loc[~wins,"pnl_pct"].sum()
    pf = float(gains / abs(losses)) if losses < 0 else float("inf")
    # equity (assume risk scaling handled elsewhere; treat pnl_pct as per-trade %)
    equity = (1.0 + df["pnl_pct"].fillna(0)/100.0).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    ret_pct = float((equity.iloc[-1] - 1.0) * 100.0)
    return {"trades": int(len(df)), "winrate": winrate, "pf": pf, "max_dd": max_dd, "ret_pct": ret_pct, "equity": equity, "df": df}

def calibration_bins(df: pd.DataFrame, bins=10):
    # reliability by confidence vs actual success (target beats stop/timeout for long/short)
    d = df[df["confidence"].notna() & df["outcome"].notna()].copy()
    if d.empty: 
        return pd.DataFrame(columns=["bin","conf_mid","empirical_winrate","count"])
    # define success as positive pnl
    d["success"] = d["pnl_pct"] > 0
    d["bin"] = pd.qcut(d["confidence"].clip(0,1), q=min(bins, d["confidence"].nunique()), duplicates="drop")
    g = d.groupby("bin", observed=True).agg(empirical_winrate=("success","mean"),
                                            conf_mid=("confidence","mean"),
                                            count=("success","size")).reset_index()
    return g
