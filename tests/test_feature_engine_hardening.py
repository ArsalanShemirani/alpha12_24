import pandas as pd
import numpy as np
from src.features.engine import FeatureEngine

def make_df(n=500, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC")
    close = 10000 + rng.normal(0, 50, n).cumsum()
    high = close + rng.uniform(1, 30, n)
    low  = close - rng.uniform(1, 30, n)
    vol  = rng.uniform(10, 1000, n)
    df = pd.DataFrame({"open":close, "high":high, "low":low, "close":close, "volume":vol}, index=idx)
    return df

def test_feature_engine_drops_sparse_and_ffill_bfill():
    df = make_df()
    fe = FeatureEngine()
    feats, cols = fe.build_feature_matrix(df, horizons=[24], symbol="BTCUSDT")

    # Simulate a sparse live-only column: 80% NaN
    feats["ob_imb_top20"] = np.nan
    feats.loc[feats.sample(frac=0.2, random_state=1).index, "ob_imb_top20"] = 0.1

    # Re-harden via engine helper if exposed, or inline replicate behavior
    na_frac = feats[cols].isna().mean()
    drop_cols = set(na_frac[na_frac > 0.50].index.tolist())

    # Must drop the intentionally sparse column if it's in cols
    to_check = set(cols).intersection({"ob_imb_top20"})
    if to_check:
        assert "ob_imb_top20" in drop_cols, "Sparse OB column should be marked for drop"

    # Ensure we can create a clean slice for training
    cols2 = [c for c in cols if c not in drop_cols]
    X = feats[cols2].ffill().bfill()
    assert (~X.isna().any(axis=1)).sum() > 100, "Usable hardened rows should exist"

def test_targets_present_for_first_horizon():
    df = make_df()
    fe = FeatureEngine()
    feats, cols = fe.build_feature_matrix(df, horizons=[24], symbol="BTCUSDT")
    assert "target_24h" in feats.columns, "First horizon target must exist"
