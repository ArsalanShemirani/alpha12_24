"""
Probabilistic HTF Support/Resistance (S/R) confluence module.

Provides zone detection, scoring, and probabilistic adjustments for p_hit and target shaping.
Feature-flagged and fail-safe with minimal runtime overhead.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pytz

# Environment configuration with safe defaults
SRZ_ENABLED = int(os.getenv("SRZ_ENABLED", "1"))
SRZ_HTF_LIST = os.getenv("SRZ_HTF_LIST", "4h,1d").split(",")
SRZ_BAND_K_4H = float(os.getenv("SRZ_BAND_K_4H", "0.6"))
SRZ_BAND_K_1D = float(os.getenv("SRZ_BAND_K_1D", "0.5"))
SRZ_TP_FRONTRUN = int(os.getenv("SRZ_TP_FRONTRUN", "0"))
SRZ_TP_FRONTRUN_ATR = float(os.getenv("SRZ_TP_FRONTRUN_ATR", "0.25"))
SRZ_DELTA_MAX = float(os.getenv("SRZ_DELTA_MAX", "0.06"))
SRZ_W1 = float(os.getenv("SRZ_W1", "0.05"))
SRZ_W2 = float(os.getenv("SRZ_W2", "0.03"))
SRZ_W3 = float(os.getenv("SRZ_W3", "0.04"))
SRZ_ZONE_NEAR_R = float(os.getenv("SRZ_ZONE_NEAR_R", "0.8"))
SRZ_TRIM_THRESH = float(os.getenv("SRZ_TRIM_THRESH", "0.6"))

logger = logging.getLogger(__name__)

# Global cache for zone data
_zone_cache = {}
_cache_timestamps = {}

def _get_band_k(tf: str) -> float:
    """Get band width multiplier for timeframe."""
    if tf == "4h":
        return SRZ_BAND_K_4H
    elif tf == "1d":
        return SRZ_BAND_K_1D
    else:
        return 0.5  # default

def _fetch_htf_data(symbol: str, tf: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """
    Fetch HTF data for zone detection.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        from src.data.binance_data import get_klines
        
        # Convert timeframe to minutes
        tf_minutes = {"4h": 240, "1d": 1440}.get(tf, 240)
        
        # Get data
        data = get_klines(symbol, tf_minutes, limit=limit)
        if data is None or data.empty:
            return None
            
        return data
        
    except Exception as e:
        logger.warning(f"Failed to fetch HTF data for {symbol} {tf}: {e}")
        return None

def _find_pivot_points(df: pd.DataFrame, window: int = 5, min_move: float = 0.01) -> Tuple[List[float], List[float]]:
    """
    Find pivot highs and lows using rolling window.
    
    Args:
        df: DataFrame with high, low columns
        window: Rolling window size
        min_move: Minimum price move to consider as pivot
        
    Returns:
        Tuple of (pivot_highs, pivot_lows) as price levels
    """
    if len(df) < window * 2:
        return [], []
    
    highs = []
    lows = []
    
    for i in range(window, len(df) - window):
        # Check for pivot high
        if all(df['high'].iloc[i] > df['high'].iloc[i-window:i]) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+1:i+window+1]):
            highs.append(df['high'].iloc[i])
            
        # Check for pivot low
        if all(df['low'].iloc[i] < df['low'].iloc[i-window:i]) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+1:i+window+1]):
            lows.append(df['low'].iloc[i])
    
    # Filter by minimum move
    if highs:
        mean_high = np.mean(highs)
        highs = [h for h in highs if abs(h - mean_high) / mean_high > min_move]
    
    if lows:
        mean_low = np.mean(lows)
        lows = [l for l in lows if abs(l - mean_low) / mean_low > min_move]
    
    return highs, lows

def _find_hvn_lvn(df: pd.DataFrame, k_percent: float = 0.1) -> Tuple[List[float], List[float]]:
    """
    Find High Volume Nodes (HVN) and Low Volume Nodes (LVN) using price*volume histogram.
    
    Args:
        df: DataFrame with close, volume columns
        k_percent: Percentage for top/bottom nodes
        
    Returns:
        Tuple of (hvn_levels, lvn_levels)
    """
    if len(df) < 20:
        return [], []
    
    # Calculate price*volume
    pv = df['close'] * df['volume']
    
    # Create histogram
    hist, bins = np.histogram(pv, bins=20)
    
    # Find top and bottom k% levels
    sorted_indices = np.argsort(hist)
    k_count = max(1, int(len(hist) * k_percent))
    
    hvn_indices = sorted_indices[-k_count:]
    lvn_indices = sorted_indices[:k_count]
    
    hvn_levels = [bins[i] for i in hvn_indices if hist[i] > 0]
    lvn_levels = [bins[i] for i in lvn_indices if hist[i] > 0]
    
    return hvn_levels, lvn_levels

def _calculate_anchored_vwap(df: pd.DataFrame, anchor_periods: List[str] = ["monthly", "quarterly"]) -> List[float]:
    """
    Calculate anchored VWAP levels.
    
    Args:
        df: DataFrame with timestamp, close, volume columns
        anchor_periods: List of anchor periods
        
    Returns:
        List of VWAP levels
    """
    vwap_levels = []
    
    try:
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        df_copy.set_index('timestamp', inplace=True)
        
        for period in anchor_periods:
            if period == "monthly":
                # Monthly VWAP
                monthly_groups = df_copy.groupby(df_copy.index.to_period('M'))
                for _, group in monthly_groups:
                    if len(group) > 5:  # Minimum bars
                        vwap = (group['close'] * group['volume']).sum() / group['volume'].sum()
                        vwap_levels.append(vwap)
                        
            elif period == "quarterly":
                # Quarterly VWAP
                quarterly_groups = df_copy.groupby(df_copy.index.to_period('Q'))
                for _, group in quarterly_groups:
                    if len(group) > 10:  # Minimum bars
                        vwap = (group['close'] * group['volume']).sum() / group['volume'].sum()
                        vwap_levels.append(vwap)
    
    except Exception as e:
        logger.warning(f"Failed to calculate anchored VWAP: {e}")
    
    return vwap_levels

def _merge_nearby_levels(levels: List[float], atr: float, merge_threshold: float = 0.5) -> List[Tuple[float, float]]:
    """
    Merge nearby levels into bands.
    
    Args:
        levels: List of price levels
        atr: Average True Range for band width
        merge_threshold: Threshold for merging (in ATR units)
        
    Returns:
        List of (center, width) tuples
    """
    if not levels:
        return []
    
    # Sort levels
    sorted_levels = sorted(levels)
    bands = []
    
    current_band = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        if abs(level - current_band[-1]) <= merge_threshold * atr:
            current_band.append(level)
        else:
            # Create band from current group
            center = np.mean(current_band)
            width = (max(current_band) - min(current_band)) / 2
            bands.append((center, width))
            current_band = [level]
    
    # Add final band
    if current_band:
        center = np.mean(current_band)
        width = (max(current_band) - min(current_band)) / 2
        bands.append((center, width))
    
    return bands

def _calculate_zone_strength(band_center: float, band_width: float, df: pd.DataFrame, 
                           recent_touches: int = 0, rejection_quality: float = 0.0) -> float:
    """
    Calculate zone strength score [0, 1].
    
    Args:
        band_center: Zone center price
        band_width: Zone width
        df: Recent price data
        recent_touches: Number of recent touches
        rejection_quality: Quality of rejections (0-1)
        
    Returns:
        Strength score [0, 1]
    """
    if len(df) < 20:
        return 0.0
    
    # Calculate ATR for normalization
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.rolling(window=14).mean().iloc[-1]
    
    if pd.isna(atr) or atr == 0:
        atr = 0.01
    
    # Base strength from band width (narrower = stronger)
    width_score = max(0, 1 - (band_width / atr))
    
    # Recent touches bonus
    touch_score = min(0.3, recent_touches * 0.1)
    
    # Rejection quality bonus
    rejection_score = rejection_quality * 0.2
    
    # Recency penalty (older zones are weaker)
    recency_penalty = 0.0  # Could be enhanced with actual touch timestamps
    
    strength = width_score + touch_score + rejection_score - recency_penalty
    return max(0.0, min(1.0, strength))

def _detect_zones(symbol: str, tf: str) -> Dict[str, Any]:
    """
    Detect support and resistance zones for a symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        tf: Timeframe (4h, 1d)
        
    Returns:
        Dictionary with support and resistance zones
    """
    # Fetch data
    df = _fetch_htf_data(symbol, tf)
    if df is None or len(df) < 50:
        return {"support": [], "resistance": []}
    
    # Calculate ATR for band width
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.rolling(window=14).mean().iloc[-1]
    
    if pd.isna(atr) or atr == 0:
        atr = 0.01
    
    band_k = _get_band_k(tf)
    band_width = band_k * atr
    
    # Find pivot points
    pivot_highs, pivot_lows = _find_pivot_points(df, window=5, min_move=0.01)
    
    # Find HVN/LVN
    hvn_levels, lvn_levels = _find_hvn_lvn(df, k_percent=0.1)
    
    # Calculate anchored VWAP
    vwap_levels = _calculate_anchored_vwap(df)
    
    # Merge all levels
    all_highs = pivot_highs + hvn_levels
    all_lows = pivot_lows + lvn_levels
    
    # Merge nearby levels into bands
    resistance_bands = _merge_nearby_levels(all_highs, atr, merge_threshold=0.5)
    support_bands = _merge_nearby_levels(all_lows, atr, merge_threshold=0.5)
    
    # Calculate strength for each band
    resistance_zones = []
    for center, width in resistance_bands:
        strength = _calculate_zone_strength(center, width, df)
        resistance_zones.append({
            "center": center,
            "width": width,
            "strength": strength,
            "upper_edge": center + width,
            "lower_edge": center - width
        })
    
    support_zones = []
    for center, width in support_bands:
        strength = _calculate_zone_strength(center, width, df)
        support_zones.append({
            "center": center,
            "width": width,
            "strength": strength,
            "upper_edge": center + width,
            "lower_edge": center - width
        })
    
    return {
        "support": support_zones,
        "resistance": resistance_zones,
        "atr": atr,
        "timestamp": datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
    }

def nearest_zone(symbol: str, tf_active: str, price: float, now_ts) -> Dict[str, Any]:
    """
    Find the nearest relevant zone for a given price and trade direction.
    
    Args:
        symbol: Trading symbol
        tf_active: Active timeframe
        price: Current price
        now_ts: Current timestamp
        
    Returns:
        Dictionary with zone information
    """
    if not SRZ_ENABLED:
        return {
            "zone_type": "none",
            "band_edge": None,
            "zone_dist_R": None,
            "zone_score": None,
            "confluence_score": None,
            "meta": {}
        }
    
    # Check cache
    cache_key = f"{symbol}_{tf_active}"
    if cache_key in _zone_cache:
        cached_data = _zone_cache[cache_key]
        cache_time = _cache_timestamps.get(cache_key)
        
        # Refresh cache if HTF bar has closed
        if cache_time and (now_ts - cache_time).total_seconds() < 3600:  # 1 hour cache
            zones = cached_data
        else:
            zones = _detect_zones(symbol, tf_active)
            _zone_cache[cache_key] = zones
            _cache_timestamps[cache_key] = now_ts
    else:
        zones = _detect_zones(symbol, tf_active)
        _zone_cache[cache_key] = zones
        _cache_timestamps[cache_key] = now_ts
    
    if not zones or ("support" not in zones and "resistance" not in zones):
        return {
            "zone_type": "none",
            "band_edge": None,
            "zone_dist_R": None,
            "zone_score": None,
            "confluence_score": None,
            "meta": {}
        }
    
    # Find nearest zones
    nearest_support = None
    nearest_resistance = None
    
    for zone in zones.get("support", []):
        if price > zone["upper_edge"]:  # Price above support zone
            dist = price - zone["upper_edge"]
            if nearest_support is None or dist < nearest_support["dist"]:
                nearest_support = {
                    "zone": zone,
                    "dist": dist,
                    "edge": zone["upper_edge"]
                }
    
    for zone in zones.get("resistance", []):
        if price < zone["lower_edge"]:  # Price below resistance zone
            dist = zone["lower_edge"] - price
            if nearest_resistance is None or dist < nearest_resistance["dist"]:
                nearest_resistance = {
                    "zone": zone,
                    "dist": dist,
                    "edge": zone["lower_edge"]
                }
    
    # Determine zone type and calculate metrics
    atr = zones.get("atr", 0.01)
    
    if nearest_support and nearest_resistance:
        # Choose closer zone
        if nearest_support["dist"] < nearest_resistance["dist"]:
            zone_info = nearest_support
            zone_type = "support"
        else:
            zone_info = nearest_resistance
            zone_type = "resistance"
    elif nearest_support:
        zone_info = nearest_support
        zone_type = "support"
    elif nearest_resistance:
        zone_info = nearest_resistance
        zone_type = "resistance"
    else:
        return {
            "zone_type": "none",
            "band_edge": None,
            "zone_dist_R": None,
            "zone_score": None,
            "confluence_score": None,
            "meta": {}
        }
    
    # Calculate metrics
    zone_dist_R = zone_info["dist"] / atr if atr > 0 else None
    zone_score = zone_info["zone"]["strength"]
    
    # Simple confluence score (can be enhanced)
    confluence_score = min(1.0, zone_score * 0.8 + 0.2)
    
    return {
        "zone_type": zone_type,
        "band_edge": zone_info["edge"],
        "zone_dist_R": zone_dist_R,
        "zone_score": zone_score,
        "confluence_score": confluence_score,
        "meta": {
            "zone_center": zone_info["zone"]["center"],
            "zone_width": zone_info["zone"]["width"],
            "atr": atr
        }
    }

def confirm_choch_bos(entry_tf: str, price: float) -> bool:
    """
    Confirm Change of Character (ChoCh) or Break of Structure (BoS).
    
    Args:
        entry_tf: Entry timeframe
        price: Current price
        
    Returns:
        True if ChoCh/BoS confirmed, False otherwise
    """
    # Placeholder implementation - can be enhanced with actual logic
    # For now, return False to avoid false positives
    return False

def confirm_ob_flip(entry_tf: str, window_s: int = 60) -> bool:
    """
    Confirm orderbook flip.
    
    Args:
        entry_tf: Entry timeframe
        window_s: Window in seconds
        
    Returns:
        True if OB flip confirmed, False otherwise
    """
    # Placeholder implementation - can be enhanced with actual logic
    # For now, return False to avoid false positives
    return False

def calculate_zone_adjustment(zone_info: Dict[str, Any], direction: str, 
                            p_hit_base: float, atr: float) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate probabilistic adjustment to p_hit based on zone information.
    
    Args:
        zone_info: Zone information from nearest_zone()
        direction: Trade direction ("long" or "short")
        p_hit_base: Base p_hit value
        atr: ATR for normalization
        
    Returns:
        Tuple of (adjusted_p_hit, adjustment_info)
    """
    if zone_info["zone_type"] == "none":
        return p_hit_base, {"delta_p": 0.0, "reason": "no_zone"}
    
    zone_type = zone_info["zone_type"]
    zone_dist_R = zone_info["zone_dist_R"]
    zone_score = zone_info["zone_score"]
    confluence_score = zone_info["confluence_score"]
    
    if zone_dist_R is None or zone_score is None:
        return p_hit_base, {"delta_p": 0.0, "reason": "missing_data"}
    
    # Check if zone is near enough to consider
    if zone_dist_R > SRZ_ZONE_NEAR_R:
        return p_hit_base, {"delta_p": 0.0, "reason": "zone_too_far"}
    
    delta_p = 0.0
    reason = "no_adjustment"
    
    # Trading into opposing zone (negative adjustment)
    if (direction == "long" and zone_type == "resistance") or \
       (direction == "short" and zone_type == "support"):
        
        # Calculate negative adjustment
        distance_factor = max(0, SRZ_ZONE_NEAR_R - zone_dist_R)
        delta_p = -min(SRZ_DELTA_MAX, 
                      (SRZ_W1 * zone_score + SRZ_W2 * distance_factor) * 
                      (1 - confluence_score * 0.5))
        reason = "opposing_zone"
    
    # Counter-trend from strong zone with confirmation (positive adjustment)
    elif (direction == "long" and zone_type == "support") or \
         (direction == "short" and zone_type == "resistance"):
        
        # Check for reversal confirmation
        rev_confirm = confirm_choch_bos("4h", 0.0) or confirm_ob_flip("4h", 60)
        
        if rev_confirm and zone_score >= SRZ_TRIM_THRESH:
            distance_factor = max(0, SRZ_ZONE_NEAR_R - zone_dist_R)
            delta_p = min(SRZ_DELTA_MAX, SRZ_W3 * zone_score * distance_factor)
            reason = "counter_trend_confirmed"
    
    # Apply adjustment
    p_hit_final = max(0.05, min(0.95, p_hit_base + delta_p))
    
    adjustment_info = {
        "delta_p": delta_p,
        "reason": reason,
        "zone_type": zone_type,
        "zone_dist_R": zone_dist_R,
        "zone_score": zone_score,
        "confluence_score": confluence_score
    }
    
    return p_hit_final, adjustment_info

def adjust_target_for_zone(zone_info: Dict[str, Any], direction: str, 
                          tp_raw: float, atr: float) -> Tuple[float, Dict[str, Any]]:
    """
    Adjust target price to front-run zone edge if enabled.
    
    Args:
        zone_info: Zone information from nearest_zone()
        direction: Trade direction ("long" or "short")
        tp_raw: Raw target price
        atr: ATR for normalization
        
    Returns:
        Tuple of (adjusted_tp, adjustment_info)
    """
    if not SRZ_TP_FRONTRUN:
        return tp_raw, {"tp_adjusted": False, "reason": "feature_disabled"}
    
    if zone_info["zone_type"] == "none":
        return tp_raw, {"tp_adjusted": False, "reason": "no_zone"}
    
    zone_type = zone_info["zone_type"]
    zone_dist_R = zone_info["zone_dist_R"]
    zone_score = zone_info["zone_score"]
    band_edge = zone_info["band_edge"]
    
    if zone_dist_R is None or zone_score is None or band_edge is None:
        return tp_raw, {"tp_adjusted": False, "reason": "missing_data"}
    
    # Check conditions for TP adjustment
    if zone_score < SRZ_TRIM_THRESH or zone_dist_R > SRZ_ZONE_NEAR_R:
        return tp_raw, {"tp_adjusted": False, "reason": "conditions_not_met"}
    
    # Check if trading with trend into opposing zone
    if (direction == "long" and zone_type == "resistance") or \
       (direction == "short" and zone_type == "support"):
        
        # Trim target to front-run the zone edge
        if direction == "long":
            tp_adj = min(tp_raw, band_edge - SRZ_TP_FRONTRUN_ATR * atr)
        else:  # short
            tp_adj = max(tp_raw, band_edge + SRZ_TP_FRONTRUN_ATR * atr)
        
        adjustment_info = {
            "tp_adjusted": True,
            "reason": "front_run_zone",
            "tp_raw": tp_raw,
            "tp_adj": tp_adj,
            "band_edge": band_edge,
            "trim_distance": SRZ_TP_FRONTRUN_ATR * atr
        }
        
        return tp_adj, adjustment_info
    
    return tp_raw, {"tp_adjusted": False, "reason": "not_opposing_zone"}
