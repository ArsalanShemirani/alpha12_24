import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.eval.from_logs import load_joined
from src.models.train import ModelTrainer
from src.core.config import config

def test_learn_from_logs_end_to_end(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    # Make config use our temp runs dir
    monkeypatch.setenv("ALPHA_RUNS_DIR_FOR_TESTS", str(runs))
    
    # Create 120 rows to meet minimum training requirement (100 for logistic regression)
    n_rows = 120
    
    # Write features_at_signal (2 features + close)
    feats = pd.DataFrame({
        "close": np.linspace(100, 120, n_rows),
        "feat_rsi": np.linspace(30, 70, n_rows),
        "feat_vol": np.linspace(0.01, 0.05, n_rows),
        "asset": ["BTCUSDT"]*n_rows,
        "interval": ["1h"]*n_rows,
        "ts": pd.date_range("2025-08-10", periods=n_rows, freq="h").astype(str),
    })
    feats.to_csv(runs / "features_at_signal.csv", index=False)

    # signals.csv (we need timestamps aligned)
    sigs = pd.DataFrame({
        "ts": feats["ts"],
        "asset": ["BTCUSDT"]*n_rows,
        "interval": ["1h"]*n_rows,
        "signal": ["long","short"]*(n_rows//2),
        "confidence": np.clip(np.linspace(0.55, 0.75, n_rows), 0, 1),
        "prob_up": np.linspace(0.51, 0.8, n_rows),
        "prob_down": 1 - np.linspace(0.51, 0.8, n_rows),
        "price": feats["close"],
    })
    sigs.to_csv(runs / "signals.csv", index=False)

    # trade_history with outcomes mapped to signals (win ~60%)
    out = np.array(["target","stop"]*(n_rows//2))[:n_rows]  # Alternate target/stop
    th = pd.DataFrame({
        "id": [f"id{i}" for i in range(n_rows)],
        "asset": ["BTCUSDT"]*n_rows,
        "interval": ["1h"]*n_rows,
        "created_at": feats["ts"],
        "exit_ts": feats["ts"],  # Add missing exit_ts column
        "entry": feats["close"],
        "stop": feats["close"]*0.99,
        "target": feats["close"]*1.02,
        "outcome": out,
        "pnl_pct": np.where(out=="target", 0.02, -0.01),
        "confidence": sigs["confidence"],
    })
    th.to_csv(runs / "trade_history.csv", index=False)

    dj = load_joined(str(runs))
    assert not dj.empty
    # Label: win if pnl_pct > 0
    dj["y"] = (dj["pnl_pct"] > 0).astype(int)

    feat_cols = [c for c in dj.columns if c.startswith("feat_")] + ["close"]
    dj = dj.dropna(subset=feat_cols + ["y"])

    # Basic train test (no calibration necessity here)
    tr = ModelTrainer(config)
    model = tr.train_model(dj[feat_cols], dj["y"], model_type="logistic")
    preds, probs = tr.predict(model, dj[feat_cols])
    assert probs.shape[0] == len(dj)