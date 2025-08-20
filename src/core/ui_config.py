#!/usr/bin/env python3
"""
UI Configuration Manager
Handles saving and loading UI settings for autosignal to use.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Default configuration file path
CONFIG_FILE = Path("runs/ui_config.json")

def save_ui_config(config: Dict[str, Any]) -> bool:
    """
    Save UI configuration to file for autosignal to read.
    
    Args:
        config: Dictionary of UI settings
        
    Returns:
        bool: True if saved successfully
    """
    try:
        # Ensure directory exists
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        config_with_meta = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ui_dashboard",
            "settings": config
        }
        
        # Save to file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_with_meta, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving UI config: {e}")
        return False

def load_ui_config() -> Optional[Dict[str, Any]]:
    """
    Load UI configuration from file.
    
    Returns:
        Dict with UI settings or None if not found/error
    """
    try:
        if not CONFIG_FILE.exists():
            return None
            
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            
        return data.get("settings", {})
    except Exception as e:
        print(f"Error loading UI config: {e}")
        return None

def get_ui_setting(key: str, default: Any = None) -> Any:
    """
    Get a specific UI setting with fallback to environment variable.
    
    Args:
        key: Setting key to retrieve
        default: Default value if not found
        
    Returns:
        Setting value or default
    """
    config = load_ui_config()
    if config and key in config:
        return config[key]
    
    # Fallback to environment variable
    env_key = key.upper()
    if env_key in os.environ:
        value = os.environ[env_key]
        # Try to convert to appropriate type
        try:
            if isinstance(default, bool):
                return value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(default, int):
                return int(value)
            elif isinstance(default, float):
                return float(value)
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    return default

def get_autosignal_config() -> Dict[str, Any]:
    """
    Get configuration specifically for autosignal with UI overrides.
    
    Returns:
        Dict with autosignal configuration
    """
    # Load UI config
    ui_config = load_ui_config() or {}
    
    # Build autosignal config with UI overrides
    config = {
        # Always use 1h for autosignal (hardcoded)
        "interval": "1h",
        
        # Model settings (UI can override)
        "model_type": get_ui_setting("model_type", os.getenv("ALPHA12_PREFERRED_MODEL", "xgb")),
        "calibrate_probs": get_ui_setting("calibrate_probs", True),
        "alerts_enabled": get_ui_setting("alerts_enabled", True),
        
        # UI settings with environment fallbacks
        "min_conf_arm": get_ui_setting("min_conf_arm", float(os.getenv("MIN_CONF_ARM", "0.60"))),
        
        # Adaptive confidence gate settings
        "adaptive_confidence_enabled": get_ui_setting("adaptive_confidence_enabled", True),
        "gate_regime": get_ui_setting("gate_regime", bool(int(os.getenv("GATE_REGIME", "1")))),
        "gate_rr25": get_ui_setting("gate_rr25", bool(int(os.getenv("GATE_RR25", "1")))),
        "gate_ob": get_ui_setting("gate_ob", bool(int(os.getenv("GATE_OB", "1")))),
        "rr25_thresh": get_ui_setting("rr25_thresh", float(os.getenv("RR25_THRESH", "0.00"))),
        "ob_edge_delta": get_ui_setting("ob_edge_delta", float(os.getenv("OB_EDGE_DELTA", "0.20"))),
        "ob_signed_thr": get_ui_setting("ob_signed_thr", None),  # Calculated from ob_edge_delta
        "max_setups_per_day": get_ui_setting("max_setups_per_day", int(os.getenv("MAX_SETUPS_PER_DAY", "2"))),
        
        # Risk/sizing settings
        "k_entry": get_ui_setting("k_entry", float(os.getenv("K_ENTRY_ATR", "0.25"))),
        "k_stop": get_ui_setting("k_stop", float(os.getenv("K_STOP_ATR", "1.0"))),
        "valid_bars": get_ui_setting("valid_bars", int(os.getenv("VALID_BARS", "24"))),
        "entry_buffer_bps": get_ui_setting("entry_buffer_bps", float(os.getenv("ENTRY_BUFFER_BPS", "5.0"))),
        "trigger_rule": get_ui_setting("trigger_rule", os.getenv("TRIGGER_RULE", "touch")),
        
        # Account settings
        "acct_balance": get_ui_setting("acct_balance", float(os.getenv("ACCOUNT_BALANCE_USD", "400"))),
        "max_leverage": get_ui_setting("max_leverage", int(os.getenv("MAX_LEVERAGE", "10"))),
        "risk_per_trade_pct": get_ui_setting("risk_per_trade_pct", float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))),
        "nominal_position_pct": get_ui_setting("nominal_position_pct", float(os.getenv("NOMINAL_POSITION_PCT", "25.0"))),
        
        # Telegram settings
        "tg_bot": get_ui_setting("tg_bot", os.getenv("TG_BOT_TOKEN", "")),
        "tg_chat": get_ui_setting("tg_chat", os.getenv("TG_CHAT_ID", "")),
    }
    
    # Calculate ob_signed_thr if not set
    if config["ob_signed_thr"] is None:
        config["ob_signed_thr"] = 1.0 - 2.0 * config["ob_edge_delta"]
    
    return config

def is_ui_config_recent(hours: int = 24) -> bool:
    """
    Check if UI config was updated recently.
    
    Args:
        hours: Number of hours to consider "recent"
        
    Returns:
        bool: True if config is recent
    """
    try:
        if not CONFIG_FILE.exists():
            return False
            
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            
        timestamp_str = data.get("timestamp", "")
        if not timestamp_str:
            return False
            
        timestamp = pd.to_datetime(timestamp_str, utc=True)
        now = pd.Timestamp.utcnow()
        
        return (now - timestamp).total_seconds() < (hours * 3600)
        
    except Exception:
        return False
