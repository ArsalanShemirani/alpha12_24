#!/usr/bin/env python3
"""
Background Analysis Daemon
Runs continuous analysis on multiple timeframes for model training and improvement,
while autosignal continues to generate setups on 4h+ intervals.
"""

import os
import time
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from src.core.config import Config
from src.models.train import ModelTrainer
from src.features.engine import FeatureEngine
from src.data import load_binance_spot
from src.eval.from_logs import load_joined
from src.daemon.performance_monitor import performance_monitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
config = Config()

# Enhanced configuration for multi-timeframe analysis
TRAINING_INTERVALS = os.getenv("ALPHA12_TRAINING_INTERVALS", "5m,15m,1h,4h,1d").split(",")
ANALYSIS_SLEEP = int(os.getenv("ALPHA12_ANALYSIS_SLEEP", "300"))  # 5 minutes
ASSETS = os.getenv("ALPHA12_SYMBOL", "BTCUSDT").split(",")
DAYS_HISTORY = int(os.getenv("ALPHA12_ANALYSIS_DAYS", "120"))

# Performance monitoring frequency (every N analysis cycles)
PERFORMANCE_MONITORING_FREQUENCY = int(os.getenv("ALPHA12_PERF_MONITOR_FREQ", "12"))  # Every hour

def _now_utc() -> pd.Timestamp:
    """Get current UTC timestamp."""
    ts = pd.Timestamp.utcnow()
    if getattr(ts, "tz", None) is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def _heartbeat():
    """Write heartbeat for monitoring."""
    try:
        runs_dir = Path(getattr(config, 'runs_dir', 'runs'))
        runs_dir.mkdir(parents=True, exist_ok=True)
        hb_file = runs_dir / "background_analysis_heartbeat.txt"
        hb_file.write_text(_now_utc().isoformat())
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")

def analyze_asset_timeframe(asset: str, interval: str, days: int = 120) -> Optional[dict]:
    """
    Analyze a single asset/timeframe combination for model training and improvement.
    Returns analysis results without generating setups.
    """
    try:
        logger.info(f"Analyzing {asset} {interval} for training data...")
        
        # Load data
        # Convert days to approximate limit (assuming 1440 minutes per day for 1m data)
        limit = min(days * 1440, 1000)  # Cap at 1000 to avoid too much data
        df = load_binance_spot(asset, interval, limit=limit)
        if df is None or df.empty:
            logger.warning(f"No data for {asset} {interval}")
            return None
            
        # Build features
        fe = FeatureEngine()
        feature_df, feature_cols = fe.build_feature_matrix(df, config.horizons_hours, symbol=asset)
        
        if feature_df.empty or not feature_cols:
            logger.warning(f"No features built for {asset} {interval}")
            return None
            
        # Train model for analysis (not for setup generation)
        target_col = f'target_{config.horizons_hours[0]}h'
        if target_col not in feature_df.columns:
            logger.warning(f"Target column {target_col} not found for {asset} {interval}")
            return None
            
        X = feature_df[feature_cols]
        y = feature_df[target_col]
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[valid_idx], y[valid_idx]
        
        # Adjust minimum rows based on timeframe (higher timeframes need fewer samples)
        min_rows_map = {"5m": 300, "15m": 200, "1h": 150, "4h": 100, "1d": 80}
        min_rows = min_rows_map.get(interval, 150)
        
        if len(X) < min_rows:
            logger.warning(f"Insufficient data for {asset} {interval}: {len(X)} < {min_rows}")
            return None
            
        # Train model
        trainer = ModelTrainer(config)
        model = trainer.train_model(
            X, y, model_type="rf",
            calibrate=True,  # Always calibrate for analysis
            calib_cv=3
        )
        
        # Get predictions for analysis
        yhat, proba = trainer.predict(model, X)
        
        # Calculate metrics
        accuracy = (yhat == y).mean()
        win_rate = y.mean()  # Assuming 1 = win, 0 = loss
        
        # Get model summary
        summary = trainer.get_model_summary(model)
        
        # Store feature snapshot for potential future training (non-executed setups)
        _store_feature_snapshot(asset, interval, feature_df, feature_cols, y)
        
        analysis = {
            "asset": asset,
            "interval": interval,
            "timestamp": _now_utc().isoformat(),
            "samples": len(X),
            "features": len(feature_cols),
            "accuracy": accuracy,
            "win_rate": win_rate,
            "cv_accuracy": np.mean(summary.get("cv_scores", {}).get("accuracy", [0])),
            "cv_precision": np.mean(summary.get("cv_scores", {}).get("precision", [0])),
            "cv_recall": np.mean(summary.get("cv_scores", {}).get("recall", [0])),
            "cv_f1": np.mean(summary.get("cv_scores", {}).get("f1", [0])),
            "top_features": summary.get("top_features", [])[:5],
            "model_type": summary.get("model_type"),
            "calibrated": "calib" in str(type(model.model))
        }
        
        logger.info(f"Analysis complete for {asset} {interval}: "
                   f"accuracy={accuracy:.3f}, win_rate={win_rate:.3f}, "
                   f"samples={len(X)}, features={len(feature_cols)}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Analysis error for {asset} {interval}: {e}")
        return None

def _store_feature_snapshot(asset: str, interval: str, feature_df: pd.DataFrame, 
                           feature_cols: List[str], y: pd.Series):
    """
    Store feature snapshot for non-executed setups (all timeframes)
    This provides training data even for timeframes that don't generate setups
    """
    try:
        runs_dir = Path(getattr(config, 'runs_dir', 'runs'))
        snapshot_dir = runs_dir / "feature_snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        
        # Create snapshot with latest data
        snapshot = {
            "asset": asset,
            "interval": interval,
            "timestamp": _now_utc().isoformat(),
            "features": feature_df[feature_cols].iloc[-1].to_dict(),
            "target": int(y.iloc[-1]) if len(y) > 0 else None,
            "feature_cols": feature_cols,
            "setup_generated": False,  # This is background analysis, not setup generation
            "source": "background_analysis"
        }
        
        # Save to timeframe-specific file
        snapshot_file = snapshot_dir / f"{asset}_{interval}_snapshots.jsonl"
        
        # Append to existing file or create new
        with open(snapshot_file, 'a') as f:
            import json
            f.write(json.dumps(snapshot) + '\n')
        
        logger.debug(f"Stored feature snapshot for {asset} {interval}")
        
    except Exception as e:
        logger.error(f"Error storing feature snapshot for {asset} {interval}: {e}")

def save_analysis_results(results: List[dict]):
    """Save analysis results to CSV for tracking."""
    try:
        if not results:
            return
            
        runs_dir = Path(getattr(config, 'runs_dir', 'runs'))
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_file = runs_dir / "background_analysis.csv"
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Append to existing file or create new
        if analysis_file.exists():
            existing = pd.read_csv(analysis_file)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(analysis_file, index=False)
        logger.info(f"Saved {len(results)} analysis results to {analysis_file}")
        
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")

def retrain_from_live_logs():
    """Retrain models from live trading logs if available."""
    try:
        logger.info("Checking for live logs to retrain models...")
        
        # Load joined logs
        joined = load_joined("runs")
        if joined is None or len(joined) < 50:
            logger.info("Insufficient live logs for retraining")
            return
            
        logger.info(f"Found {len(joined)} live log entries for retraining")
        
        # This would integrate with your existing retrain_from_logs.py logic
        # For now, just log the availability
        logger.info("Live logs available for model retraining")
        
    except Exception as e:
        logger.error(f"Error in live logs retraining: {e}")

def run_performance_monitoring():
    """Run performance monitoring across all timeframes"""
    try:
        logger.info("Running performance monitoring across all timeframes...")
        
        # Monitor all assets and timeframes
        results = performance_monitor.monitor_all_timeframes(ASSETS, TRAINING_INTERVALS)
        
        # Log results
        for key, result in results.items():
            if "error" in result:
                logger.error(f"Performance monitoring error for {key}: {result['error']}")
            else:
                metrics = result.get("metrics", {})
                degraded = result.get("degraded", False)
                recalibrated = result.get("recalibrated", False)
                
                if degraded:
                    logger.warning(f"Performance degraded for {key}: "
                                 f"WR={metrics.get('win_rate', 0):.3f}, "
                                 f"PF={metrics.get('profit_factor', 0):.2f}")
                    
                    if recalibrated:
                        logger.info(f"Successfully recalibrated model for {key}")
                    else:
                        logger.warning(f"Failed to recalibrate model for {key}")
                else:
                    logger.info(f"Performance OK for {key}: "
                              f"WR={metrics.get('win_rate', 0):.3f}, "
                              f"PF={metrics.get('profit_factor', 0):.2f}")
        
        logger.info("Performance monitoring complete")
        
    except Exception as e:
        logger.error(f"Error in performance monitoring: {e}")

def background_analysis_loop():
    """Main background analysis loop."""
    logger.info(f"Starting enhanced background analysis daemon")
    logger.info(f"Training intervals: {TRAINING_INTERVALS}")
    logger.info(f"Analysis sleep: {ANALYSIS_SLEEP} seconds")
    logger.info(f"Assets: {ASSETS}")
    logger.info(f"History days: {DAYS_HISTORY}")
    logger.info(f"Performance monitoring frequency: every {PERFORMANCE_MONITORING_FREQUENCY} cycles")
    
    cycle_count = 0
    
    while True:
        try:
            _heartbeat()
            cycle_count += 1
            
            # Analyze each asset/timeframe combination
            results = []
            for asset in ASSETS:
                for interval in TRAINING_INTERVALS:
                    analysis = analyze_asset_timeframe(asset, interval, DAYS_HISTORY)
                    if analysis:
                        results.append(analysis)
            
            # Save results
            if results:
                save_analysis_results(results)
            
            # Run performance monitoring every N cycles
            if cycle_count % PERFORMANCE_MONITORING_FREQUENCY == 0:
                run_performance_monitoring()
            
            # Check for live logs retraining (less frequent)
            if cycle_count % (PERFORMANCE_MONITORING_FREQUENCY * 2) == 0:  # Every 2 hours
                retrain_from_live_logs()
            
            logger.info(f"Background analysis cycle {cycle_count} complete. "
                       f"Analyzed {len(results)} asset/timeframe combinations. "
                       f"Sleeping {ANALYSIS_SLEEP} seconds...")
            time.sleep(ANALYSIS_SLEEP)
            
        except KeyboardInterrupt:
            logger.info("Background analysis daemon stopped by user")
            break
        except Exception as e:
            logger.error(f"Background analysis error: {e}")
            time.sleep(ANALYSIS_SLEEP)

if __name__ == "__main__":
    background_analysis_loop()
