import pandas as pd
from pathlib import Path
from src.data import deribit_rr25 as rr

def test_write_latest_uses_snapshot_stub(tmp_path, monkeypatch):
    # Use a stubbed snapshot_rr25 -> small DataFrame
    def _fake_snapshot(currency="BTC"):
        now = pd.Timestamp("2025-08-15 00:00:00", tz="UTC")
        df = pd.DataFrame([{
            "currency": currency,
            "expiry": "2025-12-26",
            "rr25": 0.0123,
            "iv_call25": 0.60,
            "iv_put25": 0.58,
            "updated_at": now.isoformat(),
        }])
        return df

    monkeypatch.setattr(rr, "snapshot_rr25", _fake_snapshot)
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    # monkeypatch destination path inside module
    monkeypatch.setenv("RUNS_DIR", str(runs))

    # Call writer
    rr.write_latest(currency="BTC", runs_dir=str(runs))
    out = runs / "deribit_rr25_latest_BTC.json"
    assert out.exists()
    j = out.read_text()
    assert '"rr25": 0.0123' in j