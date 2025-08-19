#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.eval.from_logs import load_joined

MY_TZ = "Asia/Kuala_Lumpur"

def _closed_metrics(df: pd.DataFrame) -> dict:
    # Use only CLOSED trades (target/stop)
    closed = df[df["outcome"].isin(["target", "stop"])].copy()
    if closed.empty:
        return {
            "n": 0.0, "wr": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
            "pf": np.nan, "expectancy": np.nan, "brier": np.nan
        }

    y = (closed["outcome"] == "target").astype(int)
    p = closed["prob_up"].clip(0,1) if "prob_up" in closed else pd.Series([0.5]*len(closed), index=closed.index)

    wr = float(y.mean())
    avg_win = float(closed.loc[y==1, "pnl_pct"].mean()) if "pnl_pct" in closed else np.nan
    avg_loss = float(closed.loc[y==0, "pnl_pct"].mean()) if "pnl_pct" in closed else np.nan
    gw = float(closed.loc[y==1, "pnl_pct"].sum()) if "pnl_pct" in closed else np.nan
    gl = float((-closed.loc[y==0, "pnl_pct"]).sum()) if "pnl_pct" in closed else np.nan
    pf = (gw / gl) if (pd.notna(gw) and pd.notna(gl) and gl > 0) else np.nan
    expectancy = (pf * abs(avg_loss) - abs(avg_loss)) if (pd.notna(pf) and pd.notna(avg_loss)) else np.nan
    brier = float(((p - y)**2).mean()) if len(p)==len(y) and len(y)>0 else np.nan

    return {
        "n": float(len(closed)),
        "wr": wr, "avg_win": avg_win, "avg_loss": avg_loss,
        "pf": pf, "expectancy": expectancy, "brier": brier,
    }

def _now_local_iso() -> str:
    return pd.Timestamp.now(tz="UTC").tz_convert(MY_TZ).isoformat()

def run_once(runs_dir: str = "runs", charts: bool = False) -> dict:
    df = load_joined(runs_dir=runs_dir)
    metrics = _closed_metrics(df)

    out_dir = Path(runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_p = out_dir / "metrics_history.csv"

    row = {
        "ts": _now_local_iso(),
        **{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k,v in metrics.items()}
    }
    # append
    hdr = (not hist_p.exists())
    pd.DataFrame([row]).to_csv(hist_p, mode="a", index=False, header=hdr)

    if charts:
        try:
            import plotly.express as px
            hist = pd.read_csv(hist_p)
            for c in ["wr","pf","expectancy","brier","n"]:
                if c in hist.columns:
                    fig = px.line(hist, x="ts", y=c, title=c.upper())
                    fig.write_image(str(out_dir / f"metrics_{c}.png"))
        except Exception as e:
            print(f"[warn] chart export failed: {e}", file=sys.stderr)

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="runs directory")
    ap.add_argument("--charts", action="store_true", help="export png charts")
    args = ap.parse_args()
    m = run_once(args.runs, charts=args.charts)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()