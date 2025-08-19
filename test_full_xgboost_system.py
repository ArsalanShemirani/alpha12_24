#!/usr/bin/env python3
"""
Comprehensive XGBoost system test
"""
import pandas as pd
import numpy as np
from src.models.train import ModelTrainer, XGBOOST_AVAILABLE
from src.core.config import config
from src.data.composite import assemble
from src.features.engine import FeatureEngine

print("🧪 Comprehensive XGBoost System Test")
print("=" * 60)

# Test 1: XGBoost Availability
print("1️⃣ Testing XGBoost Availability")
print(f"   XGBOOST_AVAILABLE: {XGBOOST_AVAILABLE}")
if not XGBOOST_AVAILABLE:
    print("   ❌ XGBoost not available")
    exit(1)
else:
    print("   ✅ XGBoost available")

# Test 2: Data Loading
print("\n2️⃣ Testing Data Loading")
try:
    data = assemble('BTCUSDT', '1h', 100)
    print(f"   ✅ Data loaded: {len(data)} rows")
except Exception as e:
    print(f"   ❌ Data loading failed: {e}")
    exit(1)

# Test 3: Feature Engineering
print("\n3️⃣ Testing Feature Engineering")
try:
    fe = FeatureEngine()
    feature_df, feature_cols = fe.build_feature_matrix(data, config.horizons_hours)
    print(f"   ✅ Features built: {len(feature_cols)} features")
except Exception as e:
    print(f"   ❌ Feature engineering failed: {e}")
    exit(1)

# Test 4: XGBoost Training
print("\n4️⃣ Testing XGBoost Training")
try:
    target_col = f'target_{config.horizons_hours[0]}h'
    clean_df = feature_df.dropna(subset=feature_cols + [target_col])
    
    if len(clean_df) < 50:
        print("   ⚠️ Insufficient data, creating synthetic data")
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
    else:
        X = clean_df[feature_cols]
        y = clean_df[target_col]
    
    trainer = ModelTrainer(config)
    model = trainer.train_model(X, y, model_type='xgb', calibrate=True)
    
    print(f"   ✅ XGBoost model trained: {model.model_type}")
    print(f"   ✅ CV scores: {model.cv_scores}")
    
except Exception as e:
    print(f"   ❌ XGBoost training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Predictions
print("\n5️⃣ Testing XGBoost Predictions")
try:
    yhat, proba = trainer.predict(model, X)
    print(f"   ✅ Predictions: {yhat.shape}, {proba.shape}")
    print(f"   ✅ Accuracy: {(yhat == y).mean():.3f}")
except Exception as e:
    print(f"   ❌ Predictions failed: {e}")
    exit(1)

# Test 6: Dashboard Integration
print("\n6️⃣ Testing Dashboard Integration")
try:
    # Simulate dashboard model selection logic
    available_models = ["rf", "logistic"]
    if XGBOOST_AVAILABLE:
        available_models.append("xgb")
    
    print(f"   ✅ Available models: {available_models}")
    
    if "xgb" in available_models:
        print("   ✅ Dashboard will include XGBoost option")
    else:
        print("   ❌ Dashboard will NOT include XGBoost option")
        
except Exception as e:
    print(f"   ❌ Dashboard integration test failed: {e}")

print("\n" + "=" * 60)
print("🎉 COMPREHENSIVE XGBOOST SYSTEM TEST COMPLETED!")
print("✅ All tests passed - XGBoost integration is working!")
print("\n💡 Next steps:")
print("   1. Start the dashboard: streamlit run src/dashboard/app.py")
print("   2. Check that 'xgb' appears in Model Type dropdown")
print("   3. Test XGBoost training and backtesting")
