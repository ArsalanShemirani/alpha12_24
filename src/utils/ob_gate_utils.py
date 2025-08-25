"""
Orderbook gate utilities for consistent delta normalization and logging.
"""

import os
import logging
from typing import Dict, Any, Optional

def normalize_ob_delta(ui_delta: float, spread_w_bps: float, depth_topN: int = 20) -> float:
    """
    Normalize UI delta to internal delta used by the gate.
    
    Args:
        ui_delta: User interface delta (0.0-1.0, distance from edges)
        spread_w_bps: Weighted spread in basis points
        depth_topN: Depth window (top N levels)
    
    Returns:
        Normalized delta for internal gate calculation
    """
    # For now, return the UI delta directly since the current implementation
    # doesn't have additional normalization beyond the basic edge calculation
    # This function can be extended later if more complex normalization is needed
    return ui_delta

def compute_ob_gate_metrics(
    raw_imbalance: float,
    ui_delta: float,
    spread_w_bps: float,
    depth_topN: int = 20,
    bidV_topN: float = 0.0,
    askV_topN: float = 0.0
) -> Dict[str, Any]:
    """
    Compute all OB gate metrics for logging and preview.
    
    Args:
        raw_imbalance: Raw signed imbalance (-1 to +1)
        ui_delta: User interface delta (0.0-1.0)
        spread_w_bps: Weighted spread in basis points
        depth_topN: Depth window
        bidV_topN: Total bid volume in top N levels
        askV_topN: Total ask volume in top N levels
    
    Returns:
        Dictionary with all computed metrics
    """
    delta_norm = normalize_ob_delta(ui_delta, spread_w_bps, depth_topN)
    threshold = max(0.0, min(1.0, 1.0 - 2.0 * delta_norm))
    net_imbalance = raw_imbalance  # No additional adjustment in current implementation
    
    return {
        "ui_delta": ui_delta,
        "delta_norm": delta_norm,
        "raw_imbalance": raw_imbalance,
        "net_imbalance": net_imbalance,
        "threshold": threshold,
        "depth_topN": depth_topN,
        "spread_w_bps": spread_w_bps,
        "bidV_topN": bidV_topN,
        "askV_topN": askV_topN
    }

def log_ob_gate_metrics(
    asset: str,
    tf: str,
    metrics: Dict[str, Any],
    log_level: str = "INFO"
) -> None:
    """
    Log OB gate metrics in a consistent format.
    
    Args:
        asset: Asset symbol
        tf: Timeframe
        metrics: Metrics dictionary from compute_ob_gate_metrics
        log_level: Logging level (INFO, ERROR, etc.)
    """
    # Check if logging should be suppressed
    env_log_level = os.getenv("OB_GATE_LOG_LEVEL", "INFO").upper()
    if env_log_level == "ERROR" and log_level == "INFO":
        return
    
    logger = logging.getLogger(__name__)
    
    msg = (
        f"[ob_gate] sym={asset} tf={tf} "
        f"ui_delta={metrics['ui_delta']:.1%} "
        f"delta_norm={metrics['delta_norm']:.3f} "
        f"raw={metrics['raw_imbalance']:.3f} "
        f"net={metrics['net_imbalance']:.3f} "
        f"thr={metrics['threshold']:.3f} "
        f"topN={metrics['depth_topN']} "
        f"spread_w_bps={metrics['spread_w_bps']:.6f} "
        f"bidV={metrics['bidV_topN']:.6f} "
        f"askV={metrics['askV_topN']:.6f}"
    )
    
    if log_level == "INFO":
        logger.info(msg)
    elif log_level == "ERROR":
        logger.error(msg)
    elif log_level == "WARNING":
        logger.warning(msg)
    else:
        logger.debug(msg)

def format_ob_gate_block_message(
    signed_imbalance: float,
    raw_imbalance: float,
    delta_norm: float,
    ui_delta: float,
    threshold: float
) -> str:
    """
    Format a consistent OB gate block message.
    
    Args:
        signed_imbalance: Signed imbalance that failed the gate
        raw_imbalance: Raw imbalance value
        delta_norm: Normalized delta used
        ui_delta: UI delta value
        threshold: Threshold that wasn't met
    
    Returns:
        Formatted block message
    """
    return (
        f"Setup blocked by OB gate: need |signed imbalance| ≥ {threshold:.3f}. "
        f"Got {signed_imbalance:.3f} (raw {raw_imbalance:.3f}, Δ_norm {delta_norm:.3f}, ui_delta {ui_delta:.1%})."
    )
