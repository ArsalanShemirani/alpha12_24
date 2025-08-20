#!/usr/bin/env python3
"""
Performance Monitoring and Auto-Recalibration System
Monitors trading performance per timeframe and triggers recalibration when needed.
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Import our modules
from src.core.config import Config
from src.models.train import ModelTrainer
from src.features.engine import FeatureEngine
from src.data import load_binance_spot
from src.eval.from_logs import load_joined

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
config = Config()

# Performance thresholds per timeframe
PERFORMANCE_THRESHOLDS = {
    "5m": {"win_rate": 0.45, "profit_factor": 1.0, "min_trades": 50},
    "15m": {"win_rate": 0.48, "profit_factor": 1.1, "min_trades": 40},
    "1h": {"win_rate": 0.52, "profit_factor": 1.2, "min_trades": 30},
    "4h": {"win_rate": 0.55, "profit_factor": 1.3, "min_trades": 20},
    "1d": {"win_rate": 0.58, "profit_factor": 1.4, "min_trades": 15}
}

# Recalibration cooldown (hours) per timeframe
RECALIBRATION_COOLDOWN = {
    "5m": 2,    # 2 hours
    "15m": 4,   # 4 hours
    "1h": 8,    # 8 hours
    "4h": 24,   # 24 hours
    "1d": 72    # 72 hours
}

class PerformanceMonitor:
    """Performance monitoring and auto-recalibration system"""
    
    def __init__(self):
        self.config = config
        self.runs_dir = Path(getattr(config, 'runs_dir', 'runs'))
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking files
        self.performance_file = self.runs_dir / "performance_metrics.json"
        self.recalibration_log = self.runs_dir / "recalibration_log.json"
        self.timeframe_models_dir = self.runs_dir / "timeframe_models"
        self.timeframe_models_dir.mkdir(exist_ok=True)
        
        # Load existing performance data
        self.performance_data = self._load_performance_data()
        self.recalibration_history = self._load_recalibration_history()
    
    def _load_performance_data(self) -> Dict:
        """Load existing performance metrics"""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance data: {e}")
        return {}
    
    def _save_performance_data(self):
        """Save performance metrics"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _load_recalibration_history(self) -> Dict:
        """Load recalibration history"""
        if self.recalibration_log.exists():
            try:
                with open(self.recalibration_log, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading recalibration history: {e}")
        return {}
    
    def _save_recalibration_history(self):
        """Save recalibration history"""
        try:
            with open(self.recalibration_log, 'w') as f:
                json.dump(self.recalibration_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving recalibration history: {e}")
    
    def calculate_timeframe_performance(self, asset: str, interval: str, days: int = 30) -> Optional[Dict]:
        """
        Calculate performance metrics for a specific timeframe
        
        Args:
            asset: Asset symbol
            interval: Timeframe interval
            days: Number of days to analyze
        
        Returns:
            Performance metrics dictionary
        """
        try:
            # Load trade history
            trade_history_path = self.runs_dir / "trade_history.csv"
            if not trade_history_path.exists():
                logger.warning(f"No trade history found for {asset} {interval}")
                return None
            
            trades_df = pd.read_csv(trade_history_path)
            if trades_df.empty:
                logger.warning(f"Empty trade history for {asset} {interval}")
                return None
            
            # Filter by asset and interval
            timeframe_trades = trades_df[
                (trades_df['asset'] == asset) & 
                (trades_df['interval'] == interval)
            ].copy()
            
            if timeframe_trades.empty:
                logger.warning(f"No trades found for {asset} {interval}")
                return None
            
            # Filter by date range
            timeframe_trades['exit_ts'] = pd.to_datetime(timeframe_trades['exit_ts'])
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = timeframe_trades[timeframe_trades['exit_ts'] >= cutoff_date]
            
            if len(recent_trades) < PERFORMANCE_THRESHOLDS[interval]['min_trades']:
                logger.info(f"Insufficient trades for {asset} {interval}: {len(recent_trades)} < {PERFORMANCE_THRESHOLDS[interval]['min_trades']}")
                return None
            
            # Calculate metrics
            completed_trades = recent_trades[recent_trades['outcome'].isin(['target', 'stop', 'timeout'])]
            
            if completed_trades.empty:
                return None
            
            # Win rate
            wins = completed_trades['outcome'] == 'target'
            win_rate = wins.mean()
            
            # Profit factor
            gains = completed_trades.loc[wins, 'pnl_pct'].sum()
            losses = abs(completed_trades.loc[~wins, 'pnl_pct'].sum())
            profit_factor = gains / losses if losses > 0 else float('inf')
            
            # Average P&L
            avg_pnl = completed_trades['pnl_pct'].mean()
            
            # Sharpe ratio (simplified)
            returns = completed_trades['pnl_pct'] / 100.0
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            metrics = {
                'asset': asset,
                'interval': interval,
                'period_days': days,
                'total_trades': len(completed_trades),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'avg_pnl': float(avg_pnl),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'calculated_at': datetime.now().isoformat()
            }
            
            # Store metrics
            key = f"{asset}_{interval}"
            self.performance_data[key] = metrics
            self._save_performance_data()
            
            logger.info(f"Performance calculated for {asset} {interval}: "
                       f"WR={win_rate:.3f}, PF={profit_factor:.2f}, "
                       f"Trades={len(completed_trades)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance for {asset} {interval}: {e}")
            return None
    
    def check_performance_degradation(self, asset: str, interval: str) -> bool:
        """
        Check if performance has degraded below thresholds
        
        Args:
            asset: Asset symbol
            interval: Timeframe interval
        
        Returns:
            True if performance has degraded
        """
        try:
            key = f"{asset}_{interval}"
            if key not in self.performance_data:
                logger.info(f"No performance data for {asset} {interval}")
                return False
            
            metrics = self.performance_data[key]
            thresholds = PERFORMANCE_THRESHOLDS[interval]
            
            # Check if performance is below thresholds
            win_rate_degraded = metrics['win_rate'] < thresholds['win_rate']
            profit_factor_degraded = metrics['profit_factor'] < thresholds['profit_factor']
            
            if win_rate_degraded or profit_factor_degraded:
                logger.warning(f"Performance degraded for {asset} {interval}: "
                             f"WR={metrics['win_rate']:.3f} < {thresholds['win_rate']:.3f} or "
                             f"PF={metrics['profit_factor']:.2f} < {thresholds['profit_factor']:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance degradation for {asset} {interval}: {e}")
            return False
    
    def can_recalibrate(self, asset: str, interval: str) -> bool:
        """
        Check if recalibration is allowed (cooldown period)
        
        Args:
            asset: Asset symbol
            interval: Timeframe interval
        
        Returns:
            True if recalibration is allowed
        """
        try:
            key = f"{asset}_{interval}"
            if key not in self.recalibration_history:
                return True
            
            last_recalibration = self.recalibration_history[key].get('last_recalibration')
            if not last_recalibration:
                return True
            
            last_time = datetime.fromisoformat(last_recalibration)
            cooldown_hours = RECALIBRATION_COOLDOWN[interval]
            cooldown_delta = timedelta(hours=cooldown_hours)
            
            if datetime.now() - last_time < cooldown_delta:
                remaining = cooldown_delta - (datetime.now() - last_time)
                logger.info(f"Recalibration cooldown for {asset} {interval}: "
                           f"{remaining.total_seconds()/3600:.1f} hours remaining")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking recalibration cooldown for {asset} {interval}: {e}")
            return False
    
    def trigger_timeframe_recalibration(self, asset: str, interval: str) -> bool:
        """
        Trigger recalibration for a specific timeframe
        
        Args:
            asset: Asset symbol
            interval: Timeframe interval
        
        Returns:
            True if recalibration was successful
        """
        try:
            logger.info(f"Triggering recalibration for {asset} {interval}")
            
            # Check cooldown
            if not self.can_recalibrate(asset, interval):
                return False
            
            # Load data for the specific timeframe
            days_history = min(120, int(RECALIBRATION_COOLDOWN[interval] * 24 * 2))  # 2x cooldown period
            limit = min(days_history * 1440, 1000)
            
            df = load_binance_spot(asset, interval, limit=limit)
            if df is None or df.empty:
                logger.error(f"No data available for {asset} {interval} recalibration")
                return False
            
            # Build features
            fe = FeatureEngine()
            feature_df, feature_cols = fe.build_feature_matrix(df, config.horizons_hours, symbol=asset)
            
            if feature_df.empty or not feature_cols:
                logger.error(f"No features built for {asset} {interval} recalibration")
                return False
            
            # Train model
            target_col = f'target_{config.horizons_hours[0]}h'
            if target_col not in feature_df.columns:
                logger.error(f"Target column {target_col} not found for {asset} {interval}")
                return False
            
            X = feature_df[feature_cols]
            y = feature_df[target_col]
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X, y = X[valid_idx], y[valid_idx]
            
            if len(X) < 100:
                logger.error(f"Insufficient data for {asset} {interval} recalibration: {len(X)} rows")
                return False
            
            # Train model with calibration
            trainer = ModelTrainer(config)
            model = trainer.train_model(
                X, y, model_type="rf",
                calibrate=True,
                calib_cv=3
            )
            
            # Save timeframe-specific model
            model_dir = self.timeframe_models_dir / f"{asset}_{interval}"
            model_dir.mkdir(exist_ok=True)
            
            # Save model and metadata
            import joblib
            model_path = model_dir / "model.joblib"
            meta_path = model_dir / "meta.json"
            
            joblib.dump(model.model, model_path)
            
            meta = {
                "asset": asset,
                "interval": interval,
                "model_type": "rf",
                "calibrated": True,
                "samples": len(X),
                "features": len(feature_cols),
                "trained_at": datetime.now().isoformat(),
                "performance_trigger": True,
                "win_rate": self.performance_data.get(f"{asset}_{interval}", {}).get('win_rate', 0),
                "profit_factor": self.performance_data.get(f"{asset}_{interval}", {}).get('profit_factor', 0)
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Update recalibration history
            key = f"{asset}_{interval}"
            self.recalibration_history[key] = {
                "last_recalibration": datetime.now().isoformat(),
                "trigger_reason": "performance_degradation",
                "model_path": str(model_path),
                "meta_path": str(meta_path)
            }
            self._save_recalibration_history()
            
            logger.info(f"Successfully recalibrated model for {asset} {interval}")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering recalibration for {asset} {interval}: {e}")
            return False
    
    def monitor_all_timeframes(self, assets: List[str], intervals: List[str]) -> Dict:
        """
        Monitor performance across all timeframes and trigger recalibration if needed
        
        Args:
            assets: List of assets to monitor
            intervals: List of intervals to monitor
        
        Returns:
            Dictionary of monitoring results
        """
        results = {}
        
        for asset in assets:
            for interval in intervals:
                try:
                    # Calculate performance
                    metrics = self.calculate_timeframe_performance(asset, interval)
                    if not metrics:
                        continue
                    
                    # Check for degradation
                    degraded = self.check_performance_degradation(asset, interval)
                    
                    # Trigger recalibration if needed
                    recalibrated = False
                    if degraded:
                        recalibrated = self.trigger_timeframe_recalibration(asset, interval)
                    
                    results[f"{asset}_{interval}"] = {
                        "metrics": metrics,
                        "degraded": degraded,
                        "recalibrated": recalibrated
                    }
                    
                except Exception as e:
                    logger.error(f"Error monitoring {asset} {interval}: {e}")
                    results[f"{asset}_{interval}"] = {
                        "error": str(e)
                    }
        
        return results
    
    def get_timeframe_model_path(self, asset: str, interval: str) -> Optional[str]:
        """
        Get path to timeframe-specific model if it exists
        
        Args:
            asset: Asset symbol
            interval: Timeframe interval
        
        Returns:
            Path to model file or None
        """
        model_dir = self.timeframe_models_dir / f"{asset}_{interval}"
        model_path = model_dir / "model.joblib"
        
        if model_path.exists():
            return str(model_path)
        return None
    
    def load_timeframe_model(self, asset: str, interval: str):
        """
        Load timeframe-specific model
        
        Args:
            asset: Asset symbol
            interval: Timeframe interval
        
        Returns:
            Loaded model or None
        """
        try:
            model_path = self.get_timeframe_model_path(asset, interval)
            if not model_path:
                return None
            
            import joblib
            model = joblib.load(model_path)
            
            logger.info(f"Loaded timeframe-specific model for {asset} {interval}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading timeframe model for {asset} {interval}: {e}")
            return None

# Global instance
performance_monitor = PerformanceMonitor()
