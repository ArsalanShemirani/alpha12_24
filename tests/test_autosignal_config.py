import os
import pytest
import numpy as np

from src.daemon import autosignal

def test_build_autosetup_levels_defaults(monkeypatch):
    # Force default environment
    monkeypatch.delenv("AUTO_K_ENTRY_ATR", raising=False)
    monkeypatch.delenv("AUTO_VALID_BARS", raising=False)
    monkeypatch.delenv("AUTO_TRIGGER_RULE", raising=False)

    # Reload module to reset constants
    import importlib
    importlib.reload(autosignal)

    last_price = 100.0
    atr = 10.0
    rr = 1.8

    levels = autosignal.build_autosetup_levels("long", last_price, atr, rr)

    # Entry should be ~0.25 ATR away (default)
    assert abs(levels["entry"] - last_price) <= 0.3 * atr
    # Valid bars should be 24
    assert levels["valid_bars"] == 24
    # RR preserved
    assert pytest.approx(levels["rr"], rel=1e-6) == rr

def test_build_autosetup_levels_env_override(monkeypatch):
    # Override env variables
    monkeypatch.setenv("AUTO_K_ENTRY_ATR", "0.4")
    monkeypatch.setenv("AUTO_VALID_BARS_MIN", "12")
    monkeypatch.setenv("AUTO_TRIGGER_RULE", "close-through")

    # Reload module to apply new env
    import importlib
    importlib.reload(autosignal)

    last_price = 200.0
    atr = 20.0
    rr = 2.0

    levels = autosignal.build_autosetup_levels("short", last_price, atr, rr)

    # Entry should respect 0.4 ATR distance
    expected_entry = last_price + 0.4 * atr
    assert abs(levels["entry"] - expected_entry) < 1e-6

    # Valid bars should reflect override
    assert levels["valid_bars"] == 12