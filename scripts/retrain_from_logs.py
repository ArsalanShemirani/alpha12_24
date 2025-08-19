#!/usr/bin/env python3
"""
Production-grade model retraining from live logs
"""
import os, json, pandas as pd
from pathlib import Path
from datetime import datetime

from src.core.config import config
from src.eval.from_logs import load_joined
from src.models.train import ModelTrainer
from sklearn.calibration import CalibratedClassifierCV
from src.dashboard.app import save_artifacts_inline

RUNS = getattr(config,'runs_dir','runs')
MODEL_DIR = getattr(config,'model_dir','artifacts')

def main():
    # Load joined logs
    dj = load_joined(RUNS)
    
    # Reject if missing pnl_pct or too few rows
    if dj.empty or 'pnl_pct' not in dj.columns:
        print("❌ Not enough joined logs to retrain (missing pnl_pct or empty data)")
        return
    
    if len(dj) < 200:
        print(f"❌ Too few samples ({len(dj)} < 200). Need more completed trades.")
        return

    # Build target: win if realized pnl_pct > 0
    dj['y'] = (dj['pnl_pct'] > 0).astype(int)

    # Features: intersection of current approved features with live snapshot columns
    drop = {"pnl_pct","outcome","prob_up","prob_down","confidence","signal","price","asset","interval","ts","exit_ts","y"}
    feat_cols = [c for c in dj.columns if (c.startswith('feat_') or c in getattr(config,'feature_whitelist',[])) and c not in drop]

    # Clean and sort by time
    dj = dj.dropna(subset=feat_cols + ['y']).sort_values('ts')
    
    if len(dj) < 200:
        print(f"❌ Too few samples after cleaning ({len(dj)} < 200)")
        return

    # Time-based split (no shuffling)
    split = int(len(dj)*0.7)
    tr, te = dj.iloc[:split].copy(), dj.iloc[split:].copy()
    Xtr, ytr = tr[feat_cols], tr['y']
    Xte, yte = te[feat_cols], te['y']

    print(f"📊 Training on {len(tr)} samples, testing on {len(te)} samples")
    print(f"🎯 Features: {len(feat_cols)}")

    # Train base model
    trainer = ModelTrainer(config)
    model = trainer.train_model(Xtr, ytr, 'rf')
    base = getattr(model,'model',model)
    
    # Calibrate probabilities
    try:
        name = type(base).__name__.lower()
        method = 'isotonic' if any(k in name for k in ['forest','tree','boost','xgb']) else 'sigmoid'
        print(f"🔧 Calibrating with {method} method")
        cal = CalibratedClassifierCV(base, method=method, cv=3)
        cal.fit(Xtr, ytr)
        predictor = cal
        calibrated = True
    except Exception as e:
        print(f"⚠️  Calibration failed: {e}, using raw model")
        predictor = base
        calibrated = False

    # Rich metadata
    meta = {
        "asset": "BTCUSDT",  # Default, could be made configurable
        "interval": "1h",    # Default, could be made configurable
        "model_type": "rf",
        "samples": int(len(dj)),
        "features": int(len(feat_cols)),
        "from": "live_logs",
        "calibrated": calibrated,
        "trained_at": str(pd.Timestamp.utcnow().tz_localize('UTC').tz_convert("Asia/Kuala_Lumpur"))
    }
    
    # Save artifacts
    out = save_artifacts_inline(predictor, meta, model_dir=MODEL_DIR)
    print(f"✅ Model saved: {out}")
    print(f"📋 Metadata: {json.dumps(meta, indent=2)}")

if __name__ == "__main__":
    main()