"""
Setup Resolver - Handles overlapping setups by weight priority with recency as tie-breaker
"""

import os
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SetupInfo:
    """Setup information for resolution"""
    id: str
    weight: float
    created_at: pd.Timestamp
    status: str
    symbol: str
    timeframe: str
    direction: str
    valid_until: pd.Timestamp

@dataclass
class ResolutionResult:
    """Result of setup resolution"""
    winner_id: str
    loser_ids: List[str]
    winner_weight: float
    loser_weights: List[float]
    eps_used: float
    resolution_type: str  # "superseded" or "canceled"

def load_setups_for_resolution() -> pd.DataFrame:
    """Load setups data for resolution"""
    try:
        setups_file = os.path.join('runs', 'setups.csv')
        if os.path.exists(setups_file):
            df = pd.read_csv(setups_file)
            
            # Parse timestamps
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            if 'valid_until' in df.columns:
                df['valid_until'] = pd.to_datetime(df['valid_until'], errors='coerce')
            
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading setups for resolution: {e}")
        return pd.DataFrame()

def get_resolution_config() -> Dict:
    """Get resolution configuration from environment"""
    return {
        'enabled': os.getenv('SETUP_RESOLVE_ENABLED', '0') == '1',
        'eps_weight': float(os.getenv('SETUP_RESOLVE_EPS_WEIGHT', '0.05')),
        'statuses': os.getenv('SETUP_RESOLVE_STATUSES', 'pending,executed').split(','),
        'cancel_active': os.getenv('SETUP_RESOLVE_CANCEL_ACTIVE', '1') == '1',
        'log_enabled': os.getenv('SETUP_RESOLVE_LOG', '1') == '1'
    }

def resolve_overlapping_setups(
    symbol: str, 
    timeframe: str, 
    direction: str,
    current_time: Optional[pd.Timestamp] = None
) -> Optional[ResolutionResult]:
    """
    Resolve overlapping setups for a given symbol/timeframe/direction group.
    
    Returns:
        ResolutionResult if there are conflicts to resolve, None if no conflicts
    """
    config = get_resolution_config()
    if not config['enabled']:
        return None
    
    if current_time is None:
        current_time = pd.Timestamp.now(tz='Asia/Kuala_Lumpur')
    
    # Load setups data
    df = load_setups_for_resolution()
    if df.empty:
        return None
    
    # Filter for overlapping setups
    mask = (
        (df['asset'] == symbol) &
        (df['interval'] == timeframe) &
        (df['direction'] == direction) &
        (df['status'].isin(config['statuses'])) &
        (df['valid_until'] >= current_time)
    )
    
    overlapping = df[mask].copy()
    
    if len(overlapping) <= 1:
        return None  # No conflicts
    
    # Convert to SetupInfo objects
    setups = []
    for _, row in overlapping.iterrows():
        try:
            setup = SetupInfo(
                id=str(row['id']),
                weight=float(row.get('weight', 0.0)),
                created_at=row['created_at'],
                status=str(row['status']),
                symbol=str(row['asset']),
                timeframe=str(row['interval']),
                direction=str(row['direction']),
                valid_until=row['valid_until']
            )
            setups.append(setup)
        except Exception as e:
            logger.warning(f"Error processing setup {row.get('id', 'unknown')}: {e}")
            continue
    
    if len(setups) <= 1:
        return None
    
    # First, group setups by weight similarity (within epsilon)
    eps = config['eps_weight']
    weight_groups = []
    current_group = [setups[0]]
    
    for setup in setups[1:]:
        if abs(setup.weight - current_group[0].weight) <= eps:
            # Within epsilon - add to current group
            current_group.append(setup)
        else:
            # Outside epsilon - start new group
            weight_groups.append(current_group)
            current_group = [setup]
    
    weight_groups.append(current_group)
    
    # For each weight group, sort by recency (newer first)
    for i, group in enumerate(weight_groups):
        weight_groups[i] = sorted(group, key=lambda x: x.created_at, reverse=True)
    
    # Flatten back to list
    setups = [setup for group in weight_groups for setup in group]
    
    winner = setups[0]
    losers = setups[1:]
    
    # Check for conflicts within the same weight group
    significant_losers = []
    winner_group = None
    
    # Find which group the winner belongs to
    for group in weight_groups:
        if winner in group:
            winner_group = group
            break
    
    # All other setups in the same weight group are significant losers
    if winner_group:
        significant_losers = [setup for setup in winner_group if setup != winner]
    
    if not significant_losers:
        return None  # No significant conflicts
    
    # Determine resolution type
    resolution_type = "superseded"
    if config['cancel_active']:
        for loser in significant_losers:
            if loser.status == "executed":
                resolution_type = "canceled"
                break
    
    return ResolutionResult(
        winner_id=winner.id,
        loser_ids=[s.id for s in significant_losers],
        winner_weight=winner.weight,
        loser_weights=[s.weight for s in significant_losers],
        eps_used=eps,
        resolution_type=resolution_type
    )

def apply_resolution_result(result: ResolutionResult) -> bool:
    """
    Apply resolution result by updating setup statuses in CSV.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        setups_file = os.path.join('runs', 'setups.csv')
        if not os.path.exists(setups_file):
            logger.error("Setups file not found")
            return False
        
        df = pd.read_csv(setups_file)
        
        # Update loser statuses
        for loser_id in result.loser_ids:
            mask = df['id'] == loser_id
            if mask.any():
                if result.resolution_type == "superseded":
                    df.loc[mask, 'status'] = 'superseded'
                    df.loc[mask, 'superseded_by'] = result.winner_id
                else:  # canceled
                    df.loc[mask, 'status'] = 'canceled_by_resolver'
                    df.loc[mask, 'canceled_by'] = result.winner_id
        
        # Save updated data
        df.to_csv(setups_file, index=False)
        
        # Log resolution
        config = get_resolution_config()
        if config['log_enabled']:
            loser_info = ", ".join([f"{lid}(w={lw:.2f})" for lid, lw in zip(result.loser_ids, result.loser_weights)])
            logger.info(f"[resolve] {result.winner_id}(w={result.winner_weight:.2f}) {result.resolution_type} {loser_info}, eps={result.eps_used}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error applying resolution result: {e}")
        return False

def log_resolution_alert(result: ResolutionResult, symbol: str, timeframe: str, direction: str):
    """Log resolution alert for Telegram/notifications"""
    config = get_resolution_config()
    if not config['log_enabled']:
        return
    
    if result.resolution_type == "superseded":
        loser_info = ", ".join([f"{lid}(w={lw:.2f})" for lid, lw in zip(result.loser_ids, result.loser_weights)])
        message = f"[resolve] {symbol} {timeframe} {direction}: superseded {loser_info} by {result.winner_id}(w={result.winner_weight:.2f})"
    else:  # canceled
        active_losers = []
        for lid, lw in zip(result.loser_ids, result.loser_weights):
            active_losers.append(f"{lid}(active, w={lw:.2f})")
        active_info = ", ".join(active_losers)
        message = f"[resolve] {symbol} {timeframe} {direction}: canceled {active_info} replaced by {result.winner_id}(pending, w={result.winner_weight:.2f})"
    
    logger.info(message)
    print(message)  # Also print for immediate visibility

def should_skip_execution(setup_id: str) -> bool:
    """
    Check if a setup should be skipped during execution.
    
    Returns:
        True if setup should be skipped (superseded or canceled)
    """
    try:
        setups_file = os.path.join('runs', 'setups.csv')
        if not os.path.exists(setups_file):
            return False
        
        df = pd.read_csv(setups_file)
        mask = df['id'] == setup_id
        
        if not mask.any():
            return False
        
        status = df.loc[mask, 'status'].iloc[0]
        return status in ['superseded', 'canceled_by_resolver']
        
    except Exception as e:
        logger.error(f"Error checking execution skip for {setup_id}: {e}")
        return False
