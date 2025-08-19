#!/usr/bin/env python3
"""
Model training for alpha12_24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"Warning: XGBoost not available ({e}). Using scikit-learn models only.")


@dataclass
class TrainedModel:
    """Container for trained model and metadata"""
    model: Any
    scaler: StandardScaler
    feature_names: List[str]
    model_type: str
    cv_scores: Dict[str, List[float]]
    feature_importance: Optional[Dict[str, float]] = None
    calibrator: Optional[CalibratedClassifierCV] = None
    decision_threshold: Optional[float] = None


class ModelTrainer:
    """Model trainer for alpha12_24"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}

    # --- helper: pick a sensible calibration method for the base model
    def _pick_calib_method(self, base_model) -> str:
        name = type(base_model).__name__.lower()
        # tree/ensemble/boosting often benefit from isotonic; linear models from sigmoid
        return 'isotonic' if any(k in name for k in ['forest','tree','boost','xgb']) else 'sigmoid'

    def make_pipeline(self, model_name: str):
        if model_name == "logistic":
            # Balanced classes + scaling
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
            pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                             ("clf", base)])
            return pipe, "logistic"
        elif model_name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            base = RandomForestClassifier(
                n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
            )
            return base, "rf"
        elif model_name == "xgb" and XGBOOST_AVAILABLE:
            # XGBoost with optimized parameters for trading
            base = xgb.XGBClassifier(
                n_estimators=1000,          # Increased for better performance
                max_depth=8,                # Slightly deeper for complex patterns
                learning_rate=0.05,         # Lower for better generalization
                subsample=0.85,             # Slightly higher for stability
                colsample_bytree=0.85,      # Slightly higher for stability
                colsample_bylevel=0.9,      # New: feature sampling per level
                reg_alpha=0.1,              # New: L1 regularization
                reg_lambda=1.0,             # New: L2 regularization
                min_child_weight=3,         # New: minimum sum of instance weight
                gamma=0.1,                  # New: minimum loss reduction for split
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=1.0,       # Will be adjusted based on class balance
                tree_method='hist',         # Fastest method
                enable_categorical=False,   # Disable for numerical features
                verbosity=0                 # Suppress warnings
            )
            return base, "xgb"
        else:
            raise ValueError(f"Unknown model {model_name}")

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "rf",
        *,
        calibrate: bool = False,
        calib_method: str | None = None,
        calib_cv: int = 3,
    ):
        if model_type == "rf":
            mdl = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                class_weight='balanced',
            )
        elif model_type == "logistic":
            mdl = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
                class_weight='balanced',
            )
        elif model_type == "xgb" and XGBOOST_AVAILABLE:
            # Calculate class balance for scale_pos_weight
            class_counts = y.value_counts()
            if len(class_counts) == 2:
                scale_pos_weight = class_counts[0] / class_counts[1]  # minority / majority
            else:
                scale_pos_weight = 1.0
            
            mdl = xgb.XGBClassifier(
                n_estimators=1000,          # Increased for better performance
                max_depth=8,                # Slightly deeper for complex patterns
                learning_rate=0.05,         # Lower for better generalization
                subsample=0.85,             # Slightly higher for stability
                colsample_bytree=0.85,      # Slightly higher for stability
                colsample_bylevel=0.9,      # New: feature sampling per level
                reg_alpha=0.1,              # New: L1 regularization
                reg_lambda=1.0,             # New: L2 regularization
                min_child_weight=3,         # New: minimum sum of instance weight
                gamma=0.1,                  # New: minimum loss reduction for split
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',         # Fastest method
                enable_categorical=False,   # Disable for numerical features
                verbosity=0                 # Suppress warnings
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        scaler = StandardScaler()
        # Cross-val on scaled data
        cv_scores = self._compute_cv(mdl, scaler, X, y)

        X_scaled = scaler.fit_transform(X)
        mdl.fit(X_scaled, y)

        if calibrate:
            try:
                method = calib_method or self._pick_calib_method(mdl)
                calib = CalibratedClassifierCV(mdl, method=method, cv=calib_cv)
                calib.fit(X_scaled, y)
                mdl = calib
            except Exception:
                pass

        # Get feature importance for XGBoost
        feature_importance = None
        if model_type == "xgb" and hasattr(mdl, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, mdl.feature_importances_))
        elif model_type == "rf" and hasattr(mdl, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, mdl.feature_importances_))

        return TrainedModel(
            model=mdl,
            scaler=scaler,              # keep scaler so predict/predict_proba scale correctly
            feature_names=list(X.columns) if hasattr(X, 'columns') else [],
            model_type=model_type,
            cv_scores=cv_scores,        # <-- not empty anymore
            feature_importance=feature_importance,
            calibrator=None,
            decision_threshold=None
        )

    def predict(self, trained: TrainedModel, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (pred_label, proba_matrix) with shape (n, 2).
        Applies trained.scaler if present.
        """
        model = getattr(trained, "model", trained)
        X_in = X
        try:
            if getattr(trained, "scaler", None) is not None:
                X_in = trained.scaler.transform(X)
        except Exception:
            # Fallback to raw X if transform fails
            X_in = X

        # Labels
        try:
            yhat = model.predict(X_in)
        except Exception:
            yhat = np.zeros(len(X), dtype=int)

        # Probabilities
        proba = self.predict_proba(trained, X)  # calls the wrapper-aware version below
        return yhat, proba

    def _check_min_rows(self, y: pd.Series, min_rows: int = 200) -> None:
        if y is None or len(y) < min_rows:
            raise ValueError(f"Insufficient rows for training (got {len(y)}, need ≥ {min_rows}).")

    def _compute_cv(self, model, scaler, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        tscv = TimeSeriesSplit(n_splits=max(2, min(5, len(y) // 20)))
        cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                continue
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        return cv_scores

    def _maybe_calibrate(self, base_model, X_scaled: np.ndarray, y: pd.Series, do_calibrate: bool) -> Optional[CalibratedClassifierCV]:
        if not do_calibrate:
            return None
        try:
            calib = CalibratedClassifierCV(base_estimator=base_model, cv=3, method="sigmoid")
            calib.fit(X_scaled, y)
            return calib
        except Exception:
            return None
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """
        Train XGBoost model with optimized parameters for trading
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            TrainedModel object
        """
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not available. Please install it first.")
        
        # Check class diversity/min rows
        if len(y.unique()) < 2:
            raise ValueError("Target variable must have at least 2 classes")
        self._check_min_rows(y, min_rows=100)

        # Calculate class balance for scale_pos_weight
        class_counts = y.value_counts()
        if len(class_counts) == 2:
            scale_pos_weight = class_counts[0] / class_counts[1]  # minority / majority
        else:
            scale_pos_weight = 1.0

        # Create XGBoost model with trading-optimized parameters
        mdl = xgb.XGBClassifier(
            n_estimators=1000,          # Increased for better performance
            max_depth=8,                # Slightly deeper for complex patterns
            learning_rate=0.05,         # Lower for better generalization
            subsample=0.85,             # Slightly higher for stability
            colsample_bytree=0.85,      # Slightly higher for stability
            colsample_bylevel=0.9,      # New: feature sampling per level
            reg_alpha=0.1,              # New: L1 regularization
            reg_lambda=1.0,             # New: L2 regularization
            min_child_weight=3,         # New: minimum sum of instance weight
            gamma=0.1,                  # New: minimum loss reduction for split
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',         # Fastest method
            enable_categorical=False,   # Disable for numerical features
            verbosity=0                 # Suppress warnings
        )

        scaler = StandardScaler()
        # Cross-val on scaled data
        cv_scores = self._compute_cv(mdl, scaler, X, y)

        X_scaled = scaler.fit_transform(X)
        mdl.fit(X_scaled, y)

        # Get feature importance
        feature_importance = dict(zip(X.columns, mdl.feature_importances_))

        return TrainedModel(
            model=mdl,
            scaler=scaler,
            feature_names=list(X.columns),
            model_type="xgb",
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            calibrator=None,
            decision_threshold=None
        )

    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """
        Train logistic regression model
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            TrainedModel object
        """
        # Check class diversity/min rows
        if len(y.unique()) < 2:
            raise ValueError("Target variable must have at least 2 classes")
        self._check_min_rows(y, min_rows=100)

        scaler = StandardScaler()
        base = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

        # CV scores
        cv_scores = self._compute_cv(base, scaler, X, y)

        # Final fit on all data
        X_scaled = scaler.fit_transform(X)
        base.fit(X_scaled, y)

        calibrator = self._maybe_calibrate(base, X_scaled, y, do_calibrate=True)

        return TrainedModel(
            model=base,
            scaler=scaler,
            feature_names=list(X.columns),
            model_type='logistic_regression',
            cv_scores=cv_scores,
            calibrator=calibrator
        )
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """
        Train random forest model
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            TrainedModel object
        """
        if len(y.unique()) < 2:
            raise ValueError("Target variable must have at least 2 classes")
        self._check_min_rows(y, min_rows=150)

        scaler = StandardScaler()
        base = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )

        cv_scores = self._compute_cv(base, scaler, X, y)

        X_scaled = scaler.fit_transform(X)
        base.fit(X_scaled, y)

        feature_importance = dict(zip(X.columns, getattr(base, "feature_importances_", np.zeros(len(X.columns)))))

        calibrator = self._maybe_calibrate(base, X_scaled, y, do_calibrate=True)

        return TrainedModel(
            model=base,
            scaler=scaler,
            feature_names=list(X.columns),
            model_type='random_forest',
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            calibrator=calibrator
        )
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """
        Train XGBoost model
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            TrainedModel object
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        # Check for class diversity
        if len(y.unique()) < 2:
            raise ValueError("Target variable must have at least 2 classes")
        self._check_min_rows(y, min_rows=200)
        
        # Initialize scaler and model
        scaler = StandardScaler()
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=1
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=max(2, min(5, len(y) // 20)))
        cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Check class diversity in fold
            if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                continue
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        
        # Train final model on all data
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        calibrator = self._maybe_calibrate(model, X_scaled, y, do_calibrate=True)
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        return TrainedModel(
            model=model,
            scaler=scaler,
            feature_names=list(X.columns),
            model_type='xgboost',
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            calibrator=calibrator
        )
    
    def train_model_legacy(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'xgb') -> TrainedModel:
        """
        Legacy train model method (kept for backward compatibility)
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model ('logistic', 'rf', 'xgb', 'auto')
        
        Returns:
            TrainedModel object
        """
        if model_type == 'logistic':
            return self.train_logistic_regression(X, y)
        elif model_type == 'rf':
            return self.train_random_forest(X, y)
        elif model_type == 'xgb':
            return self.train_xgboost(X, y)
        elif model_type == 'auto':
            return self.train_model_auto(X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_legacy(self, model: TrainedModel, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy predict method (kept for backward compatibility)
        Make predictions using trained model. Returns (predictions, probabilities)
        where probabilities is an (n,2) array [:,1] = P(up).
        """
        # Drop rows with all-NaN features defensively
        Xc = X.copy()
        if isinstance(Xc, pd.DataFrame):
            Xc = Xc.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        X_scaled = model.scaler.transform(Xc)

        if model.calibrator is not None:
            try:
                proba = model.calibrator.predict_proba(X_scaled)
                preds = (proba[:, 1] >= (model.decision_threshold or 0.5)).astype(int)
                return preds, proba
            except Exception:
                pass

        if hasattr(model.model, 'predict_proba'):
            probabilities = model.model.predict_proba(X_scaled)
            predictions = (probabilities[:, 1] >= (model.decision_threshold or 0.5)).astype(int)
            return predictions, probabilities
        else:
            predictions = model.model.predict(X_scaled)
            probabilities = np.zeros((len(predictions), 2), dtype=float)
            probabilities[:, 1] = predictions.astype(float)
            probabilities[:, 0] = 1.0 - probabilities[:, 1]
            return predictions, probabilities

    def predict_proba(self, model_or_trained: Any, X) -> np.ndarray:
        """
        Robust probability output with scaler support:
        - Accepts either a TrainedModel or a raw estimator.
        - Applies trained.scaler if present.
        - If estimator has predict_proba → use it.
        - Else if it has decision_function → logistic squash.
        - Else fallback to predict() and clip to [0,1].
        - Always returns shape (n, 2) with rows summing to 1.
        """
        import numpy as _np

        # Unpack TrainedModel if needed and transform X if scaler present
        mdl = model_or_trained
        X_in = X
        if hasattr(model_or_trained, "model") or isinstance(model_or_trained, TrainedModel):
            trained = model_or_trained
            mdl = getattr(trained, "model", trained)
            try:
                if getattr(trained, "scaler", None) is not None:
                    X_in = trained.scaler.transform(X)
            except Exception:
                X_in = X

        n = len(X)

        # 1) native predict_proba
        if hasattr(mdl, "predict_proba"):
            try:
                P = _np.asarray(mdl.predict_proba(X_in))
                if P.ndim == 1:
                    P = _np.vstack([1 - P, P]).T
                elif P.ndim == 2 and P.shape[1] == 2:
                    pass
                else:
                    if hasattr(mdl, "classes_") and 1 in getattr(mdl, "classes_", []):
                        idx = list(mdl.classes_).index(1)
                        p1 = P[:, idx]
                    else:
                        p1 = P.max(axis=1)
                    P = _np.vstack([1 - p1, p1]).T
                P = _np.clip(P, 0.0, 1.0)
                row_sum = P.sum(axis=1, keepdims=True) + 1e-12
                return P / row_sum
            except Exception:
                pass

        # 2) decision_function
        if hasattr(mdl, "decision_function"):
            try:
                s = _np.asarray(mdl.decision_function(X_in), dtype=float)
                p1 = 1.0 / (1.0 + _np.exp(-s))
                if p1.ndim > 1:
                    p1 = p1.squeeze()
                p1 = _np.clip(p1, 0.0, 1.0)
                return _np.vstack([1 - p1, p1]).T
            except Exception:
                pass

        # 3) regressor-like fallback
        if hasattr(mdl, "predict"):
            try:
                p1 = _np.asarray(mdl.predict(X_in), dtype=float)
                if p1.ndim > 1:
                    p1 = p1.squeeze()
                p1 = _np.clip(p1, 0.0, 1.0)
                return _np.vstack([1 - p1, p1]).T
            except Exception:
                pass

        # 4) last resort
        return _np.vstack([_np.ones(n), _np.zeros(n)]).T

    def choose_threshold_by_rr(self, y_true: Any, p_up: Any, rr: float = 1.8) -> float:
        """
        Choose a probability threshold that maximizes expected value per trade for a given RR.
        Robust to length/index mismatches:
          - If both are pandas Series with overlapping index → align on common index.
          - Otherwise → align by trailing overlap (use last min(len(y), len(p)) samples).
        Returns a threshold in [0,1]. Falls back to 0.5 if insufficient data.
        """
        # Convert/align
        try:
            import pandas as pd
            is_pandas = isinstance(y_true, pd.Series) and isinstance(p_up, pd.Series)
        except Exception:
            is_pandas = False

        if is_pandas:
            idx = y_true.index.intersection(p_up.index)
            if len(idx) == 0:
                return 0.5
            y = y_true.loc[idx].astype(int)
            p = p_up.loc[idx].astype(float)
        else:
            y_arr = np.asarray(y_true).astype(int)
            p_arr = np.asarray(p_up).astype(float)
            n = min(len(y_arr), len(p_arr))
            if n <= 10:  # too few to be meaningful
                return 0.5
            y = y_arr[-n:]
            p = p_arr[-n:]

        # Drop NaNs synchronously
        mask_valid = ~np.isnan(p)
        if mask_valid.sum() <= 10:
            return 0.5
        y = np.asarray(y)[mask_valid]
        p = np.asarray(p)[mask_valid]

        # Grid search thresholds
        grid = np.linspace(0.3, 0.7, 41)  # focus on central band
        best_t, best_ev = 0.5, -1e9
        for t in grid:
            decide = p >= t
            if not np.any(decide):
                continue
            win = (y[decide] == 1).astype(float)
            ev = (win * rr + (1.0 - win) * -1.0).mean()
            if np.isfinite(ev) and ev > best_ev:
                best_ev, best_t = ev, t

        return float(best_t)

    def train_model_auto(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """
        Train both LR and RF, pick the one with better mean CV-F1 (ties → RF).
        """
        lr = self.train_logistic_regression(X, y)
        rf = self.train_random_forest(X, y)
        f1_lr = np.nanmean(lr.cv_scores.get("f1", [])) if lr.cv_scores.get("f1") else -1
        f1_rf = np.nanmean(rf.cv_scores.get("f1", [])) if rf.cv_scores.get("f1") else -1
        return rf if f1_rf >= f1_lr else lr

    def get_model_summary(self, model: TrainedModel) -> Dict:
        """
        Get summary statistics for trained model.
        Defensive against missing/empty cv_scores.
        """
        cv = model.cv_scores or {}
        acc = cv.get('accuracy', [])
        prec = cv.get('precision', [])
        rec = cv.get('recall', [])
        f1s = cv.get('f1', [])
        summary = {
            'model_type': model.model_type,
            'n_features': len(model.feature_names or []),
            'cv_accuracy_mean': float(np.nanmean(acc)) if len(acc) else None,
            'cv_accuracy_std': float(np.nanstd(acc)) if len(acc) else None,
            'cv_precision_mean': float(np.nanmean(prec)) if len(prec) else None,
            'cv_recall_mean': float(np.nanmean(rec)) if len(rec) else None,
            'cv_f1_mean': float(np.nanmean(f1s)) if len(f1s) else None,
        }
        if model.feature_importance:
            sorted_features = sorted(
                model.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            summary['top_features'] = sorted_features
        return summary
