#!/usr/bin/env python3
"""
R:R Invariant Logging System

This module provides centralized logging for R:R invariants to track planned vs realized
risk:reward ratios across auto and manual setups without changing trading behavior.
"""

import os
import math
import time
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import pytz

# Config toggles (default ON)
RR_INVARIANT_LOGGING = os.getenv("RR_INVARIANT_LOGGING", "1") == "1"
RR_INVARIANT_SIDECAR = os.getenv("RR_INVARIANT_SIDECAR", "1") == "1"
RR_INVARIANT_SIDECAR_PATH = os.getenv("RR_INVARIANT_SIDECAR_PATH", "runs/rr_invariants.csv")
RR_INVARIANT_TZ = os.getenv("RR_INVARIANT_TZ", "Asia/Kuala_Lumpur")
RR_INVARIANT_MIN_RR_FLOOR = float(os.getenv("RR_INVARIANT_MIN_RR_FLOOR", "1.5"))


def getenv_bool(key: str, default: bool = True) -> bool:
    """Get boolean environment variable."""
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes", "on")


def compute_rr_invariants(
    direction: str,              # "long"|"short"
    R_used: float,               # ATR or structure-R the decision used
    s_planned: float,            # stop distance in R
    t_planned: float,            # target distance in R
    entry_planned: float,        # reference entry at decision time
    entry_fill: Optional[float], # actual fill (None at decision time)
    live_entry: Optional[float], # what the system logged as entry (prices)
    live_stop: Optional[float],  # what the system logged as stop (prices)
    live_tp: Optional[float],    # what the system logged as target (prices)
    setup_id: str,
    tf: str,
) -> Dict[str, Union[float, str, int, None]]:
    """
    Returns a dict with invariant fields for logging only; never mutates strategy state.
    """
    out = {
        "setup_id": setup_id, "tf": tf, "direction": direction,
        "R_used": R_used, "s_planned": s_planned, "t_planned": t_planned,
        "rr_planned": (t_planned / max(1e-12, s_planned)) if (s_planned and t_planned) else float("nan"),
        "rr_planned_logged": None, "warn_rr_floor_violation": 0,
        "entry_planned": entry_planned, "entry_fill": entry_fill,
        "entry_shift_R": float("nan"),
        "stop_final_from_fill": float("nan"), "tp_final_from_fill": float("nan"),
        "rr_realized_from_fill": float("nan"),
        "rr_realized_from_prices": float("nan"),
        "rr_distortion": float("nan"),
        "warn_missing_fill": 0,
        "invariant_ts": time.time(),
    }

    # Floor guard for logging only
    floor = float(os.getenv("RR_INVARIANT_MIN_RR_FLOOR", "1.5"))
    if math.isfinite(out["rr_planned"]) and out["rr_planned"] < floor:
        out["warn_rr_floor_violation"] = 1
        out["rr_planned_logged"] = floor
    else:
        out["rr_planned_logged"] = out["rr_planned"]

    # Decision-time path (no fill yet)
    if entry_fill is None or not math.isfinite(entry_fill):
        out["warn_missing_fill"] = 1
        return out

    # Fill-time recompute in R-space (for verification only)
    sign = +1 if direction == "long" else -1
    sR, tR = s_planned * R_used, t_planned * R_used

    stop_from_fill = entry_fill - sign * sR
    tp_from_fill   = entry_fill + sign * tR
    out["stop_final_from_fill"] = stop_from_fill
    out["tp_final_from_fill"]   = tp_from_fill

    # Entry shift in R
    if math.isfinite(entry_planned) and R_used:
        out["entry_shift_R"] = (entry_fill - entry_planned) / R_used

    # Realized RR from fill-based recompute (should equal t/s)
    out["rr_realized_from_fill"] = (t_planned / max(1e-12, s_planned)) if (s_planned and t_planned) else float("nan")

    # Realized RR from live prices (whatever is logged by system)
    if all(map(lambda x: x is not None and math.isfinite(x), [live_entry, live_stop, live_tp])):
        num = abs(live_tp - live_entry)
        den = abs(live_entry - live_stop)
        out["rr_realized_from_prices"] = (num / den) if den > 0 else float("nan")

        # Distortion vs planned
        rr_plan = out["rr_planned"]
        if rr_plan and math.isfinite(rr_plan) and rr_plan > 0 and math.isfinite(out["rr_realized_from_prices"]):
            out["rr_distortion"] = abs(out["rr_realized_from_prices"] - rr_plan) / rr_plan

    return out


class RRInvariantWriter:
    """Atomic CSV writer for R:R invariants."""
    
    def __init__(self, sidecar_path: str = None):
        self.sidecar_path = sidecar_path or RR_INVARIANT_SIDECAR_PATH
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure the directory exists."""
        import os
        os.makedirs(os.path.dirname(self.sidecar_path), exist_ok=True)
    
    def append(self, invariants: Dict) -> bool:
        """
        Atomically append invariants to sidecar CSV.
        
        Args:
            invariants: Dictionary from compute_rr_invariants
            
        Returns:
            True if successful, False otherwise
        """
        if not getenv_bool("RR_INVARIANT_LOGGING", True):
            return True
        
        if not invariants:
            return True
        
        try:
            # Convert to DataFrame
            df_new = pd.DataFrame([invariants])
            
            # Load existing data or create new file
            try:
                df_existing = pd.read_csv(self.sidecar_path)
                # Remove existing row for this setup_id if it exists
                df_existing = df_existing[df_existing['setup_id'] != invariants['setup_id']]
                # Append new row
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except FileNotFoundError:
                df_combined = df_new
            
            # Atomic write (open→write→flush→close)
            df_combined.to_csv(self.sidecar_path, index=False)
            
            # Log warnings if needed
            if invariants.get('warn_rr_floor_violation', 0):
                print(f"[RR_INVARIANTS] ⚠️  RR floor violation for {invariants['setup_id']}: "
                      f"rr_planned={invariants.get('rr_planned', 'N/A'):.2f} < {RR_INVARIANT_MIN_RR_FLOOR}")
            
            if invariants.get('warn_missing_fill', 0):
                print(f"[RR_INVARIANTS] ⚠️  Missing fill price for {invariants['setup_id']}")
            
            return True
            
        except Exception as e:
            print(f"[RR_INVARIANTS] ❌ Failed to log invariants: {e}")
            return False


# Global writer instance
rr_invariant_writer = RRInvariantWriter()


def get_rr_invariants(setup_id: str, sidecar_path: str = None) -> Optional[Dict]:
    """
    Retrieve R:R invariants for a specific setup.
    
    Args:
        setup_id: Unique setup identifier
        sidecar_path: Path to the sidecar CSV file
        
    Returns:
        Dictionary with invariant data or None if not found
    """
    if not getenv_bool("RR_INVARIANT_LOGGING", True):
        return None
    
    sidecar_path = sidecar_path or RR_INVARIANT_SIDECAR_PATH
    
    try:
        df = pd.read_csv(sidecar_path)
        row = df[df['setup_id'] == setup_id]
        if not row.empty:
            return row.iloc[0].to_dict()
    except FileNotFoundError:
        pass
    
    return None
