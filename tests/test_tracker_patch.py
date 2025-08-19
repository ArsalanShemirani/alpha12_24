# tests/test_tracker_patch.py
import os
import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Import the functions under test
import src.daemon.tracker as tracker


@pytest.fixture()
def tmp_runs(monkeypatch, tmp_path: Path):
    """
    Redirect all tracker paths to a temp runs dir and clean state.
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    # Redirect globals created at import-time
    monkeypatch.setattr(tracker, "RUNS_DIR", runs, raising=True)
    monkeypatch.setattr(tracker, "SETUPS_CSV", runs / "setups.csv", raising=True)
    monkeypatch.setattr(tracker, "TRADES_CSV", runs / "trade_history.csv", raising=True)
    monkeypatch.setattr(tracker, "HEARTBEAT", runs / "daemon_heartbeat.txt", raising=True)

    # Redirect helper that heartbeat uses
    monkeypatch.setattr(tracker, "_runs_dir", lambda: runs, raising=True)

    # Make sure no env leakage from TG/limits
    monkeypatch.delenv("TG_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TG_CHAT_ID", raising=False)
    monkeypatch.delenv("MAX_SETUPS_PER_DAY", raising=False)

    return runs


def test_heartbeat_writes_iso_utc(tmp_runs):
    tracker._hb()
    hb = (tmp_runs / "daemon_heartbeat.txt")
    assert hb.exists()
    s = hb.read_text().strip()
    # Parse and confirm tz-aware UTC
    ts = pd.to_datetime(s, utc=True, errors="raise")
    assert ts.tz is not None
    # After parsing with utc=True every naive is coerced to UTC;
    # ensure it represents "now-ish"
    assert (pd.Timestamp.now(tz="UTC") - ts) < pd.Timedelta(minutes=5)


def test_append_csv_row_adds_ts(tmp_runs):
    # Use signals.csv append (through wrapper) to exercise _append_csv_row
    tracker.log_signal({"asset": "BTCUSDT", "interval": "5m", "note": "unit-test"})
    p = tmp_runs / "signals.csv"
    df = pd.read_csv(p)
    assert "ts" in df.columns
    assert len(df) == 1
    assert df.loc[0, "asset"] == "BTCUSDT"


def test_save_and_load_setups_schema(tmp_runs):
    # Save a minimal frame; loader should add missing canonical columns
    df = pd.DataFrame([{
        "id": "S1", "asset": "BTCUSDT", "interval": "5m",
        "direction": "long", "entry": 100.0, "stop": 95.0, "target": 110.0,
        "created_at": pd.Timestamp("2025-08-16T00:00:00Z")
    }])
    tracker._save_setups_df(df)
    out = tracker._load_setups_df()
    # Must have canonical columns
    for c in tracker._SETUP_FIELDS:
        assert c in out.columns
    # Datetime fields parsed
    assert pd.api.types.is_datetime64_any_dtype(out["created_at"])
    assert pd.isna(out.loc[0, "expires_at"]) or isinstance(out.loc[0, "expires_at"], pd.Timestamp)


def test_append_setup_auto_with_daily_cap(tmp_runs, monkeypatch):
    # Cap = 2 per 24h for AUTO origin
    monkeypatch.setenv("MAX_SETUPS_PER_DAY", "2")

    # Pre-seed two recent 'auto' setups within 24h
    now_local = tracker._now_local()
    base = pd.DataFrame([
        {
            "id": "A1", "asset": "BTCUSDT", "interval": "5m",
            "direction": "long", "entry": 100, "stop": 95, "target": 110,
            "created_at": now_local.isoformat(), "expires_at": (now_local + pd.Timedelta(hours=6)).isoformat(),
            "status": "pending", "origin": "auto",
        },
        {
            "id": "A2", "asset": "BTCUSDT", "interval": "5m",
            "direction": "short", "entry": 100, "stop": 105, "target": 90,
            "created_at": (now_local - pd.Timedelta(hours=1)).isoformat(),
            "expires_at": (now_local + pd.Timedelta(hours=6)).isoformat(),
            "status": "pending", "origin": "auto",
        },
    ])
    tracker._save_setups_df(base)

        # 3rd auto should be blocked by cap
    ok = tracker.append_setup_auto({
        "id": "A3", "asset": "BTCUSDT", "interval": "5m",
        "direction": "long", "entry": 117000, "stop": 116500, "target": 118000,
        "status": "pending"
    })
    assert ok is False

    # Disabling cap (0) should allow
    monkeypatch.setenv("MAX_SETUPS_PER_DAY", "0")
    ok2 = tracker.append_setup_auto({
        "id": "A4", "asset": "BTCUSDT", "interval": "5m",
        "direction": "long", "entry": 117000, "stop": 116500, "target": 118000,
        "status": "pending"
    })
    assert ok2 is True

    df = tracker._load_setups_df()
    # A4 should be present and tagged as auto
    assert (df["id"] == "A4").any()
    assert df.loc[df["id"] == "A4", "origin"].iloc[0] == "auto"


def test_trade_row_pnl_net_and_dedup(tmp_runs):
    # Single completion (target) should be appended once; repeats are ignored
    row = {
        "setup_id": "S123",
        "asset": "BTCUSDT", "interval": "15m", "direction": "long",
        "created_at": tracker._now_local(),
        "trigger_ts": tracker._now_local(),
        "entry": 100.0, "stop": 95.0, "target": 110.0,
        "exit_ts": tracker._now_local(), "exit_price": 110.0,
        "outcome": "target",
        "pnl_pct": 10.0,
        "rr_planned": 2.0,
        "confidence": 0.7,
        "size_units": 0.05, "notional_usd": 100.0, "leverage": 3,
        "price_at_trigger": 100.0, "trigger_rule": "touch", "entry_buffer_bps": 5.0,
    }
    # First append
    tracker._append_trade_row(row)
    # Duplicate should be ignored
    tracker._append_trade_row(row)

    df = pd.read_csv(tmp_runs / "trade_history.csv")
    assert len(df) == 1  # deduped by setup_id+outcome
    # pnl_pct_net computed: pnl_pct - 2 * fees_bps_per_side / 100
    fees_bps_side = float(getattr(tracker.config, "fees_bps_per_side", 4.0))
    expected_net = 10.0 - (2.0 * fees_bps_side) / 100.0
    assert pytest.approx(df.loc[0, "pnl_pct_net"], rel=1e-6) == expected_net
    # schema must include rr_planned, confidence
    for c in ["rr_planned", "confidence", "price_at_trigger", "trigger_rule", "entry_buffer_bps"]:
        assert c in df.columns