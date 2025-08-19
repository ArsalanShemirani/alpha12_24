#!/usr/bin/env python3
"""
Test XGBoost integration on Ubuntu
"""
import pandas as pd
import numpy as np
from src.models.train import ModelTrainer, XGBOOST_AVAILABLE
from src.core.config import config

print("🧪 Testing XGBoost Integration on Ubuntu")
print("=" * 50)

# Check availability
print(f"✅ XGBOOST_AVAILABLE: {XGBOOST_AVAILABLE}")

if not XGBOOST_AVAILABLE:
    print("❌ XGBoost not available - check installation")
    exit(1)

# Create test data
np.random.seed(42)
X = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
y = pd.Series(np.random.randint(0, 2, 100))

print(f"✅ Test data created: {X.shape}")

# Test XGBoost training
try:
    trainer = ModelTrainer(config)
    model = trainer.train_model(X, y, model_type='xgb', calibrate=True)
    
    print(f"✅ XGBoost model trained successfully!")
    print(f"   Model type: {model.model_type}")
    print(f"   CV scores: {model.cv_scores}")
    
    if model.feature_importance:
        print(f"   Feature importance: Available")
    
    # Test predictions
    yhat, proba = trainer.predict(model, X)
    print(f"✅ Predictions successful: {yhat.shape}, {proba.shape}")
    
    print("\n🎉 All XGBoost tests PASSED!")
    
except Exception as e:
    print(f"❌ XGBoost test failed: {e}")
    import traceback
    traceback.print_exc()
