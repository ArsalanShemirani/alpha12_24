#!/usr/bin/env python3
"""
Adaptive Minimum Confidence Gate for Alpha12_24

Implements an adaptive minimum confidence threshold used by the entry gate with:
- Base thresholds per timeframe (reduced by ~8% from prior proposal)
- User override support via environment variables/config
- Validation and clamping of override values
- Comprehensive logging and telemetry
"""

import os
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceGateResult:
    """Result from confidence gate evaluation"""
    passed: bool
    timeframe: str
    base_min_conf: float
    user_override: Optional[float]
    effective_min_conf: float
    model_confidence: float
    clamped: bool = False
    warning_message: Optional[str] = None

class AdaptiveConfidenceGate:
    """
    Adaptive minimum confidence gate with per-timeframe thresholds
    and user override support.
    """
    
    def __init__(self):
        # Base thresholds per timeframe (reduced by ~8% from prior proposal)
        self.base_thresholds = {
            "15m": 0.72,
            "1h": 0.69,
            "4h": 0.64,
            "1d": 0.62
        }
        
        # Safe range for user overrides
        self.min_override = 0.5
        self.max_override = 0.95
        
        # Cache for user overrides
        self._user_overrides = None
        self._load_user_overrides()
    
    def _load_user_overrides(self) -> None:
        """Load user overrides from environment variables"""
        self._user_overrides = {}
        
        # Load from environment variables
        for timeframe in self.base_thresholds.keys():
            env_var = f"MIN_CONF_{timeframe.upper().replace('M', 'M').replace('H', 'H').replace('D', 'D')}"
            override_value = os.getenv(env_var)
            
            if override_value is not None:
                try:
                    override_float = float(override_value)
                    self._user_overrides[timeframe] = override_float
                    logger.info(f"Loaded user override for {timeframe}: {override_float}")
                except ValueError:
                    logger.warning(f"Invalid override value for {timeframe}: {override_value} (not a valid float)")
        
        # Also try to load from config if available
        try:
            from src.core.config import config
            for timeframe in self.base_thresholds.keys():
                config_key = f"min_conf_{timeframe.lower().replace('m', 'm').replace('h', 'h').replace('d', 'd')}"
                if hasattr(config, config_key):
                    override_value = getattr(config, config_key)
                    if override_value is not None:
                        try:
                            override_float = float(override_value)
                            self._user_overrides[timeframe] = override_float
                            logger.info(f"Loaded config override for {timeframe}: {override_float}")
                        except ValueError:
                            logger.warning(f"Invalid config override for {timeframe}: {override_value}")
        except ImportError:
            pass  # Config not available
    
    def _clamp_override(self, value: float, timeframe: str) -> Tuple[float, bool, Optional[str]]:
        """
        Clamp override value to safe range
        
        Args:
            value: Override value to clamp
            timeframe: Timeframe for logging
            
        Returns:
            Tuple of (clamped_value, was_clamped, warning_message)
        """
        if value < self.min_override:
            clamped_value = self.min_override
            warning = f"User override for {timeframe} ({value}) below minimum ({self.min_override}), clamped to {clamped_value}"
            return clamped_value, True, warning
        elif value > self.max_override:
            clamped_value = self.max_override
            warning = f"User override for {timeframe} ({value}) above maximum ({self.max_override}), clamped to {clamped_value}"
            return clamped_value, True, warning
        else:
            return value, False, None
    
    def get_effective_threshold(self, timeframe: str) -> Tuple[float, Optional[float], bool, Optional[str]]:
        """
        Get effective minimum confidence threshold for timeframe
        
        Args:
            timeframe: Trading timeframe ("15m", "1h", "4h", "1d")
            
        Returns:
            Tuple of (effective_threshold, user_override, was_clamped, warning_message)
        """
        # Get base threshold
        base_threshold = self.base_thresholds.get(timeframe)
        if base_threshold is None:
            logger.warning(f"Unknown timeframe: {timeframe}, using 1h base threshold")
            base_threshold = self.base_thresholds["1h"]
        
        # Check for user override
        user_override = self._user_overrides.get(timeframe)
        
        if user_override is not None:
            # Clamp override to safe range
            clamped_value, was_clamped, warning = self._clamp_override(user_override, timeframe)
            
            if was_clamped:
                logger.warning(warning)
            
            effective_threshold = clamped_value
        else:
            effective_threshold = base_threshold
            was_clamped = False
            warning = None
        
        return effective_threshold, user_override, was_clamped, warning
    
    def evaluate_confidence(self, timeframe: str, model_confidence: float) -> ConfidenceGateResult:
        """
        Evaluate if model confidence passes the adaptive threshold
        
        Args:
            timeframe: Trading timeframe
            model_confidence: Calibrated model confidence (0.0 to 1.0)
            
        Returns:
            ConfidenceGateResult with evaluation details
        """
        # Get effective threshold
        effective_threshold, user_override, was_clamped, warning = self.get_effective_threshold(timeframe)
        
        # Get base threshold for logging
        base_threshold = self.base_thresholds.get(timeframe, self.base_thresholds["1h"])
        
        # Evaluate gate
        passed = model_confidence >= effective_threshold
        
        # Create result
        result = ConfidenceGateResult(
            passed=passed,
            timeframe=timeframe,
            base_min_conf=base_threshold,
            user_override=user_override,
            effective_min_conf=effective_threshold,
            model_confidence=model_confidence,
            clamped=was_clamped,
            warning_message=warning
        )
        
        # Log telemetry
        self._log_telemetry(result)
        
        return result
    
    def _log_telemetry(self, result: ConfidenceGateResult) -> None:
        """Log telemetry for confidence gate evaluation"""
        override_info = f"override={result.user_override:.3f}" if result.user_override is not None else "override=None"
        clamped_info = " (CLAMPED)" if result.clamped else ""
        
        logger.info(
            f"Confidence gate: {result.timeframe} | "
            f"base={result.base_min_conf:.3f} | "
            f"{override_info} | "
            f"effective={result.effective_min_conf:.3f}{clamped_info} | "
            f"confidence={result.model_confidence:.3f} | "
            f"passed={result.passed}"
        )
        
        if result.warning_message:
            logger.warning(result.warning_message)
    
    def get_all_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Get all thresholds for all timeframes
        
        Returns:
            Dictionary with threshold information for each timeframe
        """
        thresholds = {}
        
        for timeframe in self.base_thresholds.keys():
            effective_threshold, user_override, was_clamped, warning = self.get_effective_threshold(timeframe)
            
            thresholds[timeframe] = {
                "base": self.base_thresholds[timeframe],
                "user_override": user_override,
                "effective": effective_threshold,
                "clamped": was_clamped
            }
        
        return thresholds
    
    def reset_overrides(self) -> None:
        """Reset user overrides (for testing)"""
        self._user_overrides = {}
        logger.info("User overrides reset")
    
    def reload_overrides(self) -> None:
        """Reload user overrides from environment variables"""
        self._load_user_overrides()
        logger.info("User overrides reloaded")

# Global instance
adaptive_confidence_gate = AdaptiveConfidenceGate()
