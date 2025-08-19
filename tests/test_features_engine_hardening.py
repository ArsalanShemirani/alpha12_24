import numpy as np
import pandas as pd
import importlib
from src.features.engine import FeatureEngine
from src.core.config import config

def test_build_feature_matrix_basic(sample_ohlcv_hourly, monkeypatch):
    fe = FeatureEngine()

    # Inject a sparse column to ensure hardening drops it
    df = sample_ohlcv_hourly.copy()
    df["very_sparse"] = np.nan
    df.loc[df.index[-1], "very_sparse"] = 1.0

    F, cols = fe.build_feature_matrix(df, config.horizons_hours, symbol="BTCUSDT")
    # Ensure features frame returned and at least 'close' preserved in F
    assert isinstance(F, pd.DataFrame)
    assert "close" in F.columns
    # If 'very_sparse' was proposed as feature, it must be dropped (our app hardening drops >50% NaN)
    assert "very_sparse" not in cols