#!/usr/bin/env python3
"""
Model calibration for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CalibratedModel:
    """Container for calibrated model"""
    original_model: Any
    calibrated_model: Any
    scaler: Any
    feature_names: List[str]
    model_type: str
    calibration_method: str
    calibration_scores: Dict[str, float]


class ModelCalibrator:
    """Model calibrator for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
    
    def calibrate_isotonic(self, model: Any, X: pd.DataFrame, y: pd.Series) -> CalibratedModel:
        """
        Calibrate model using isotonic regression
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
        
        Returns:
            CalibratedModel object
        """
        # Time series split for calibration
        tscv = TimeSeriesSplit(n_splits=min(3, len(y) // 30))
        
        # Use CalibratedClassifierCV with isotonic regression
        calibrated_model = CalibratedClassifierCV(
            model,
            cv=tscv,
            method='isotonic',
            n_jobs=1
        )
        
        # Fit calibrated model
        calibrated_model.fit(X, y)
        
        # Calculate calibration scores
        y_pred_proba = calibrated_model.predict_proba(X)[:, 1]
        calibration_scores = {
            'brier_score': brier_score_loss(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba)
        }
        
        return CalibratedModel(
            original_model=model,
            calibrated_model=calibrated_model,
            scaler=getattr(model, 'scaler', None),
            feature_names=getattr(model, 'feature_names', list(X.columns)),
            model_type=getattr(model, 'model_type', 'unknown'),
            calibration_method='isotonic',
            calibration_scores=calibration_scores
        )
    
    def calibrate_sigmoid(self, model: Any, X: pd.DataFrame, y: pd.Series) -> CalibratedModel:
        """
        Calibrate model using sigmoid (Platt scaling)
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
        
        Returns:
            CalibratedModel object
        """
        # Time series split for calibration
        tscv = TimeSeriesSplit(n_splits=min(3, len(y) // 30))
        
        # Use CalibratedClassifierCV with sigmoid
        calibrated_model = CalibratedClassifierCV(
            model,
            cv=tscv,
            method='sigmoid',
            n_jobs=1
        )
        
        # Fit calibrated model
        calibrated_model.fit(X, y)
        
        # Calculate calibration scores
        y_pred_proba = calibrated_model.predict_proba(X)[:, 1]
        calibration_scores = {
            'brier_score': brier_score_loss(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba)
        }
        
        return CalibratedModel(
            original_model=model,
            calibrated_model=calibrated_model,
            scaler=getattr(model, 'scaler', None),
            feature_names=getattr(model, 'feature_names', list(X.columns)),
            model_type=getattr(model, 'model_type', 'unknown'),
            calibration_method='sigmoid',
            calibration_scores=calibration_scores
        )
    
    def calibrate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'isotonic') -> CalibratedModel:
        """
        Calibrate model using specified method
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            method: Calibration method ('isotonic' or 'sigmoid')
        
        Returns:
            CalibratedModel object
        """
        if method == 'isotonic':
            return self.calibrate_isotonic(model, X, y)
        elif method == 'sigmoid':
            return self.calibrate_sigmoid(model, X, y)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def predict_calibrated(self, calibrated_model: CalibratedModel, 
                          X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using calibrated model
        
        Args:
            calibrated_model: CalibratedModel object
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, calibrated_probabilities)
        """
        # Scale features if scaler is available
        if calibrated_model.scaler:
            X_scaled = calibrated_model.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get calibrated probabilities
        calibrated_proba = calibrated_model.calibrated_model.predict_proba(X_scaled)
        
        # Get predictions
        predictions = calibrated_model.calibrated_model.predict(X_scaled)
        
        return predictions, calibrated_proba
    
    def evaluate_calibration(self, calibrated_model: CalibratedModel, 
                           X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate calibration quality
        
        Args:
            calibrated_model: CalibratedModel object
            X: Feature matrix
            y: Target variable
        
        Returns:
            Dictionary with calibration metrics
        """
        # Get calibrated probabilities
        _, calibrated_proba = self.predict_calibrated(calibrated_model, X)
        y_pred_proba = calibrated_proba[:, 1]
        
        # Calculate metrics
        metrics = {
            'brier_score': brier_score_loss(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba),
            'calibration_error': self._calculate_calibration_error(y, y_pred_proba)
        }
        
        return metrics
    
    def _calculate_calibration_error(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """
        Calculate calibration error
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            Calibration error
        """
        # Bin predictions
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Calculate mean prediction and mean actual
                mean_pred = np.mean(y_pred_proba[in_bin])
                mean_actual = np.mean(y_true[in_bin])
                
                # Add to calibration error
                calibration_error += np.abs(mean_pred - mean_actual) * np.sum(in_bin)
        
        return calibration_error / len(y_true)
    
    def get_calibration_summary(self, calibrated_model: CalibratedModel) -> Dict:
        """
        Get summary of calibrated model
        
        Args:
            calibrated_model: CalibratedModel object
        
        Returns:
            Dictionary with calibration summary
        """
        summary = {
            'model_type': calibrated_model.model_type,
            'calibration_method': calibrated_model.calibration_method,
            'n_features': len(calibrated_model.feature_names),
            'brier_score': calibrated_model.calibration_scores['brier_score'],
            'log_loss': calibrated_model.calibration_scores['log_loss']
        }
        
        return summary
