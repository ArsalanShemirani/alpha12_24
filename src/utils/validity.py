"""
Adaptive setup validity computation.

Computes dynamic validity windows based on timeframe, current ATR (R_used), 
and macro regime instead of using fixed "24 bars" validity.
"""

import os
import math
from typing import Union
import pandas as pd


def compute_adaptive_validity_bars(
    tf: str, 
    R_used: float, 
    regime: str, 
    now_ts: Union[pd.Timestamp, str]
) -> int:
    """
    Compute adaptive validity bars based on timeframe, ATR, and macro regime.
    
    Args:
        tf: Timeframe ("15m", "1h", "4h", "1d")
        R_used: ATR/R unit at decision time
        regime: Macro regime ("bull", "bear", "neutral")
        now_ts: Current timestamp (for reference)
    
    Returns:
        Number of bars for validity window
    """
    
    # Check if adaptive validity is enabled
    if os.getenv("ADAPTIVE_VALIDITY", "1") != "1":
        # Return legacy fixed bars
        base = {
            "15m": 32.0,
            "1h":  32.0,
            "4h":  24.0,
            "1d":  20.0,
        }.get(tf.lower(), 24.0)
        return int(base)
    
    def envf(key: str, default: float) -> float:
        """Get environment variable as float with fallback."""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    # Normalize timeframe
    tfu = tf.lower()
    if tfu not in ("15m", "1h", "4h", "1d"):
        tfu = "4h"  # fallback

    # Base bars per timeframe
    base = {
        "15m": envf("AVF_BASE_15M", 32.0),
        "1h":  envf("AVF_BASE_1H",  32.0),
        "4h":  envf("AVF_BASE_4H",  24.0),
        "1d":  envf("AVF_BASE_1D",  20.0),
    }[tfu]

    # Volatility anchors (typical ATR for each timeframe)
    vol_anchor = {
        "15m": envf("AVF_VOL_ANCHOR_15M", 60.0),
        "1h":  envf("AVF_VOL_ANCHOR_1H",  150.0),
        "4h":  envf("AVF_VOL_ANCHOR_4H",  1200.0),
        "1d":  envf("AVF_VOL_ANCHOR_1D",  3000.0),
    }[tfu]

    # Min/max bounds per timeframe
    min_b = {
        "15m": int(envf("AVF_MIN_15M", 12)),
        "1h":  int(envf("AVF_MIN_1H",  12)),
        "4h":  int(envf("AVF_MIN_4H",  12)),
        "1d":  int(envf("AVF_MIN_1D",  10)),
    }[tfu]

    max_b = {
        "15m": int(envf("AVF_MAX_15M", 48)),
        "1h":  int(envf("AVF_MAX_1H",  48)),
        "4h":  int(envf("AVF_MAX_4H",  36)),
        "1d":  int(envf("AVF_MAX_1D",  30)),
    }[tfu]

    # Volatility adjustment (faster markets → shorter validity)
    try:
        vol_ratio = max(1e-9, float(R_used) / float(vol_anchor))
    except (ValueError, TypeError, ZeroDivisionError):
        vol_ratio = 1.0

    # Clamp vol_ratio between 0.5 and 2.0
    vol_ratio = max(0.5, min(2.0, vol_ratio))
    vol_factor = 1.0 / vol_ratio  # higher ATR → smaller factor

    # Regime adjustment (trends persist, ranges decay)
    reg = (regime or "neutral").lower()
    if reg == "bull":
        reg_factor = envf("AVF_REGIME_BULL", 1.15)
    elif reg == "bear":
        reg_factor = envf("AVF_REGIME_BEAR", 1.15)
    else:  # neutral
        reg_factor = envf("AVF_REGIME_NEUTRAL", 0.85)

    # Final validity calculation
    raw = base * vol_factor * reg_factor
    valid_bars = int(max(min_b, min(max_b, round(raw))))
    
    return valid_bars


def get_validity_until(created_at: Union[pd.Timestamp, str], valid_bars: int, tf: str) -> pd.Timestamp:
    """
    Compute valid_until timestamp from created_at, valid_bars, and timeframe.
    
    Args:
        created_at: Setup creation timestamp
        valid_bars: Number of bars for validity
        tf: Timeframe ("15m", "1h", "4h", "1d")
    
    Returns:
        Valid until timestamp
    """
    # Convert to pandas timestamp if string
    if isinstance(created_at, str):
        created_at = pd.to_datetime(created_at)
    
    # Timeframe duration in minutes
    tf_minutes = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf.lower()]
    
    # Calculate valid_until
    valid_until = created_at + pd.Timedelta(minutes=tf_minutes * valid_bars)
    
    # Remove timezone info if present (consistent with existing system)
    if getattr(valid_until, "tzinfo", None) is not None:
        valid_until = valid_until.tz_localize(None)
    
    return valid_until
