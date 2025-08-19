#!/usr/bin/env python3
"""
Background Analysis Daemon
Runs continuous analysis on 15m data for model training and improvement,
while autosignal continues to generate setups on 1h+ intervals.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
config = Config()
TRAINING_INTERVAL = os.getenv("ALPHA12_TRAINING_INTERVAL", "15m")
ANALYSIS_SLEEP = int(os.getenv("ALPHA12_ANALYSIS_SLEEP", "300"))  # 5 minutes
ASSETS = os.getenv("ALPHA12_SYMBOL", "BTCUSDT").split(",")
DAYS_HISTORY = int(os.getenv("ALPHA12_ANALYSIS_DAYS", "120"))

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

def analyze_asset(asset: str, interval: str = "15m", days: int = 120) -> Optional[dict]:
    """
    Analyze a single asset for model training and improvement.
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
        
        if len(X) < 100:  # Lower threshold for analysis
            logger.warning(f"Insufficient data for {asset} {interval}: {len(X)} rows")
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
        win_rate = y.mean()
        
        # Get feature importance
        summary = trainer.get_model_summary(model)
        
        analysis = {
            "asset": asset,
            "interval": interval,
            "timestamp": _now_utc().isoformat(),
            "samples": len(X),
            "features": len(feature_cols),
            "accuracy": float(accuracy),
            "win_rate": float(win_rate),
            "cv_accuracy": summary.get("cv_accuracy_mean"),
            "cv_precision": summary.get("cv_precision_mean"),
            "cv_recall": summary.get("cv_recall_mean"),
            "cv_f1": summary.get("cv_f1_mean"),
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

def background_analysis_loop():
    """Main background analysis loop."""
    logger.info(f"Starting background analysis daemon")
    logger.info(f"Training interval: {TRAINING_INTERVAL}")
    logger.info(f"Analysis sleep: {ANALYSIS_SLEEP} seconds")
    logger.info(f"Assets: {ASSETS}")
    logger.info(f"History days: {DAYS_HISTORY}")
    
    while True:
        try:
            _heartbeat()
            
            # Analyze each asset
            results = []
            for asset in ASSETS:
                analysis = analyze_asset(asset, TRAINING_INTERVAL, DAYS_HISTORY)
                if analysis:
                    results.append(analysis)
            
            # Save results
            if results:
                save_analysis_results(results)
            
            # Check for live logs retraining (less frequent)
            if int(time.time()) % (ANALYSIS_SLEEP * 12) == 0:  # Every hour
                retrain_from_live_logs()
            
            logger.info(f"Background analysis cycle complete. Sleeping {ANALYSIS_SLEEP} seconds...")
            time.sleep(ANALYSIS_SLEEP)
            
        except KeyboardInterrupt:
            logger.info("Background analysis daemon stopped by user")
            break
        except Exception as e:
            logger.error(f"Background analysis error: {e}")
            time.sleep(ANALYSIS_SLEEP)

if __name__ == "__main__":
    background_analysis_loop()
