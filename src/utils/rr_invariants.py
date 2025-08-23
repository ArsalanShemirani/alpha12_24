#!/usr/bin/env python3
"""
R:R Invariant Logging System

This module provides centralized logging for R:R invariants to track planned vs realized
risk:reward ratios across auto and manual setups without changing trading behavior.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Union
import pytz

# Feature gate
RR_INVARIANT_LOGGING = os.getenv("RR_INVARIANT_LOGGING", "1") == "1"

# Timezone for timestamps
MY_TZ = pytz.timezone('Asia/Kuala_Lumpur')


def compute_rr_invariants(
    direction: str,
    entry_planned: float,
    entry_fill: Optional[float],
    R_used: float,
    s_planned: float,
    t_planned: float,
    live_entry: float,
    live_stop: float,
    live_tp: float,
    setup_id: str,
    tf: str
) -> Dict[str, Union[float, str, bool, None]]:
    """
    Compute R:R invariants for logging purposes.
    
    Args:
        direction: 'long' or 'short'
        entry_planned: Reference price used at decision time
        entry_fill: Actual fill price (None if not yet filled)
        R_used: R unit used for risk math (e.g., ATR14)
        s_planned: Stop distance in R units
        t_planned: Target distance in R units
        live_entry: Current live entry price
        live_stop: Current live stop price
        live_tp: Current live target price
        setup_id: Unique setup identifier
        tf: Timeframe
        
    Returns:
        Dictionary with all invariant fields
    """
    if not RR_INVARIANT_LOGGING:
        return {}
    
    # Compute planned R:R
    rr_planned = t_planned / s_planned if s_planned > 0 else np.nan
    
    # Apply RR floor guardrail (logging only)
    rr_planned_logged = max(rr_planned, 1.5) if not np.isnan(rr_planned) else 1.5
    warn_rr_floor_violation = rr_planned < 1.5 if not np.isnan(rr_planned) else False
    
    # Initialize fill-dependent fields
    entry_shift_R = np.nan
    stop_final_from_fill = np.nan
    tp_final_from_fill = np.nan
    rr_realized_from_fill = np.nan
    rr_realized_from_prices = np.nan
    rr_distortion = np.nan
    warn_missing_fill = entry_fill is None
    
    # Compute fill-dependent invariants if we have a fill price
    if entry_fill is not None and not np.isnan(entry_fill) and R_used > 0:
        # Entry shift in R units
        entry_shift_R = (entry_fill - entry_planned) / R_used
        if direction.lower() == 'short':
            entry_shift_R = -entry_shift_R  # Flip sign for shorts
        
        # Recompute final levels from fill using planned R-space
        if direction.lower() == 'long':
            stop_final_from_fill = entry_fill - s_planned * R_used
            tp_final_from_fill = entry_fill + t_planned * R_used
        else:  # short
            stop_final_from_fill = entry_fill + s_planned * R_used
            tp_final_from_fill = entry_fill - t_planned * R_used
        
        # Compute realized R:R metrics
        rr_realized_from_fill = t_planned / s_planned if s_planned > 0 else np.nan
        
        # Compute R:R from actual live prices
        if abs(live_entry - live_stop) > 0:
            rr_realized_from_prices = abs(live_tp - live_entry) / abs(live_entry - live_stop)
        else:
            rr_realized_from_prices = np.nan
        
        # Compute distortion
        if not np.isnan(rr_realized_from_prices) and not np.isnan(rr_planned) and rr_planned > 0:
            rr_distortion = abs(rr_realized_from_prices - rr_planned) / max(rr_planned, 1e-9)
        else:
            rr_distortion = np.nan
    
    return {
        'setup_id': setup_id,
        'tf': tf,
        'direction': direction,
        'R_used': R_used,
        's_planned': s_planned,
        't_planned': t_planned,
        'rr_planned': rr_planned,
        'rr_planned_logged': rr_planned_logged,
        'warn_rr_floor_violation': warn_rr_floor_violation,
        'entry_planned': entry_planned,
        'entry_fill': entry_fill,
        'entry_shift_R': entry_shift_R,
        'stop_final_from_fill': stop_final_from_fill,
        'tp_final_from_fill': tp_final_from_fill,
        'rr_realized_from_fill': rr_realized_from_fill,
        'rr_realized_from_prices': rr_realized_from_prices,
        'rr_distortion': rr_distortion,
        'warn_missing_fill': warn_missing_fill,
        'invariant_ts': datetime.now(MY_TZ).isoformat()
    }


def log_rr_invariants(invariants: Dict, sidecar_file: str = "runs/rr_invariants.csv") -> bool:
    """
    Log R:R invariants to a sidecar CSV file.
    
    Args:
        invariants: Dictionary from compute_rr_invariants
        sidecar_file: Path to the sidecar CSV file
        
    Returns:
        True if successful, False otherwise
    """
    if not RR_INVARIANT_LOGGING or not invariants:
        return True
    
    try:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(sidecar_file), exist_ok=True)
        
        # Convert to DataFrame
        df_new = pd.DataFrame([invariants])
        
        # Load existing data or create new file
        try:
            df_existing = pd.read_csv(sidecar_file)
            # Remove existing row for this setup_id if it exists
            df_existing = df_existing[df_existing['setup_id'] != invariants['setup_id']]
            # Append new row
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_combined = df_new
        
        # Save to CSV
        df_combined.to_csv(sidecar_file, index=False)
        
        # Log warning if RR floor was violated
        if invariants.get('warn_rr_floor_violation', False):
            print(f"[RR_INVARIANTS] ⚠️  RR floor violation for {invariants['setup_id']}: "
                  f"rr_planned={invariants.get('rr_planned', 'N/A'):.2f} < 1.5")
        
        # Log warning if fill price is missing
        if invariants.get('warn_missing_fill', False):
            print(f"[RR_INVARIANTS] ⚠️  Missing fill price for {invariants['setup_id']}")
        
        return True
        
    except Exception as e:
        print(f"[RR_INVARIANTS] ❌ Failed to log invariants: {e}")
        return False


def get_rr_invariants(setup_id: str, sidecar_file: str = "runs/rr_invariants.csv") -> Optional[Dict]:
    """
    Retrieve R:R invariants for a specific setup.
    
    Args:
        setup_id: Unique setup identifier
        sidecar_file: Path to the sidecar CSV file
        
    Returns:
        Dictionary with invariant data or None if not found
    """
    if not RR_INVARIANT_LOGGING:
        return None
    
    try:
        df = pd.read_csv(sidecar_file)
        row = df[df['setup_id'] == setup_id]
        if not row.empty:
            return row.iloc[0].to_dict()
    except FileNotFoundError:
        pass
    
    return None
