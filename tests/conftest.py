import os
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

@pytest.fixture(autouse=True)
def _isolate_tmp_runs(tmp_path, monkeypatch):
    # Put all runtime artifacts into an isolated temp dir
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    # emulate config.runs_dir without importing config here
    monkeypatch.setenv("ALPHA_RUNS_DIR_FOR_TESTS", str(runs))
    yield
    # cleanup handled by tmp_path

@pytest.fixture
def sample_ohlcv_hourly():
    """Simple hourly OHLCV frame (UTC indexed)."""
    idx = pd.date_range("2025-08-14 00:00:00+00:00", periods=48, freq="H")
    base = 100.0
    df = pd.DataFrame({
        "open":  base + (pd.Series(range(len(idx))) * 0.2).values,
        "high":  base + (pd.Series(range(len(idx))) * 0.3).values + 1.0,
        "low":   base + (pd.Series(range(len(idx))) * 0.1).values - 1.0,
        "close": base + (pd.Series(range(len(idx))) * 0.2).values + 0.1,
        "volume": 1_000.0
    }, index=idx)
    df.index.name = "timestamp"
    return df