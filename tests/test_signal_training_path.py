import pandas as pd
import numpy as np
from src.features.engine import FeatureEngine
from src.models.train import ModelTrainer
from src.core.config import config

def make_df(n=600, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 30000 + rng.normal(0, 20, n).cumsum()
    high = close + rng.uniform(0.5, 10, n)
    low  = close - rng.uniform(0.5, 10, n)
    vol  = rng.uniform(1, 50, n)
    return pd.DataFrame({"open":close, "high":high, "low":low, "close":close, "volume":vol}, index=idx)

def test_training_gate_and_predict_proba():
    df = make_df()
    fe = FeatureEngine()
    feats, cols = fe.build_feature_matrix(df, horizons=[24], symbol="BTCUSDT")

    # Harden
    na_frac = feats[cols].isna().mean()
    cols = [c for c in cols if na_frac.get(c, 0) <= 0.50]
    X = feats[cols].ffill().bfill()
    y = feats["target_24h"]
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]

    # Gate
    assert len(X) >= 300, "Fast interval should have at least 300 rows after hardening"

    tr = ModelTrainer(config)
    model = tr.train_model(X, y, model_type="rf")
    preds, proba = tr.predict(model, X.iloc[-10:])
    assert proba.shape[0] == 10, "Should output probabilities for 10 rows"
    assert proba.shape[1] == 2, "Binary probs (down, up)"
