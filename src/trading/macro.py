"""
Macro regime detection using MA200-only filter.
Simplified and unified macro trend filter across the system.
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass

RegimeType = Literal["bull", "bear", "neutral", "unknown"]

@dataclass
class MacroRegime:
    """Macro regime result with metadata."""
    regime: RegimeType
    ma200: Optional[float] = None
    current_price: Optional[float] = None
    price_vs_ma200_pct: Optional[float] = None
    reason: Optional[str] = None

def get_macro_regime(data: pd.DataFrame, asof_ts: Optional[pd.Timestamp] = None) -> MacroRegime:
    """
    Determine macro regime using MA200-only filter.
    
    Args:
        data: OHLCV DataFrame with datetime index
        asof_ts: Timestamp to evaluate regime at (defaults to latest data)
    
    Returns:
        MacroRegime object with regime classification and metadata
    """
    if data is None or data.empty:
        return MacroRegime(
            regime="unknown",
            reason="No data provided"
        )
    
    # Use latest data if no specific timestamp provided
    if asof_ts is None:
        asof_ts = data.index[-1]
    
    # Get data up to the evaluation timestamp
    mask = data.index <= asof_ts
    if not mask.any():
        return MacroRegime(
            regime="unknown",
            reason=f"No data available at {asof_ts}"
        )
    
    eval_data = data[mask].copy()
    
    # Compute MA200
    try:
        ma200 = eval_data["close"].rolling(window=200, min_periods=200).mean()
        if ma200.isna().all():
            return MacroRegime(
                regime="unknown",
                reason="Not enough data for MA200 (need 200+ bars)"
            )
        
        # Get the latest MA200 value
        latest_ma200 = ma200.dropna().iloc[-1]
        latest_price = eval_data["close"].iloc[-1]
        
        # Calculate percentage difference
        price_vs_ma200_pct = ((latest_price - latest_ma200) / latest_ma200) * 100
        
        # Determine regime based on rules
        if abs(price_vs_ma200_pct) <= 0.2:  # Within ±0.2%
            regime = "neutral"
            reason = f"Price {price_vs_ma200_pct:+.2f}% vs MA200 (within ±0.2% neutral zone)"
        elif price_vs_ma200_pct > 0.2:  # Price > MA200 + 0.2%
            regime = "bull"
            reason = f"Price {price_vs_ma200_pct:+.2f}% above MA200 (bull regime)"
        else:  # Price < MA200 - 0.2%
            regime = "bear"
            reason = f"Price {price_vs_ma200_pct:+.2f}% below MA200 (bear regime)"
        
        return MacroRegime(
            regime=regime,
            ma200=latest_ma200,
            current_price=latest_price,
            price_vs_ma200_pct=price_vs_ma200_pct,
            reason=reason
        )
        
    except Exception as e:
        return MacroRegime(
            regime="unknown",
            reason=f"Error computing MA200: {e}"
        )

def is_direction_allowed(regime: MacroRegime, direction: str, allow_override: bool = False) -> tuple[bool, str]:
    """
    Check if a trading direction is allowed given the macro regime.
    
    Args:
        regime: Macro regime result
        direction: Trading direction ("long" or "short")
        allow_override: Whether manual override is allowed
    
    Returns:
        Tuple of (is_allowed, reason)
    """
    if regime.regime == "unknown":
        return True, "Regime unknown - allowing all directions"
    
    if regime.regime == "neutral":
        return True, "Neutral regime - allowing all directions"
    
    # Check for conflicts
    if regime.regime == "bull" and direction == "short":
        if allow_override:
            return True, f"Counter-trend short allowed by override (bull regime)"
        else:
            return False, f"Counter-trend short blocked (bull regime)"
    
    if regime.regime == "bear" and direction == "long":
        if allow_override:
            return True, f"Counter-trend long allowed by override (bear regime)"
        else:
            return False, f"Counter-trend long blocked (bear regime)"
    
    # Direction aligns with regime
    return True, f"{direction.capitalize()} aligns with {regime.regime} regime"

def get_manual_gate_reason(regime: MacroRegime, direction: str) -> str:
    """
    Generate a human-readable reason for manual macro gate decisions.
    
    Args:
        regime: Macro regime result
        direction: Trading direction
    
    Returns:
        Reason string for logging
    """
    if regime.regime == "unknown":
        return "regime_unknown"
    
    if regime.regime == "neutral":
        return "regime_neutral"
    
    # Counter-trend scenarios
    if regime.regime == "bull" and direction == "short":
        return "bull_vs_short"
    
    if regime.regime == "bear" and direction == "long":
        return "bear_vs_long"
    
    # Aligned scenarios
    if regime.regime == "bull" and direction == "long":
        return "bull_vs_long"
    
    if regime.regime == "bear" and direction == "short":
        return "bear_vs_short"
    
    return "unknown"
