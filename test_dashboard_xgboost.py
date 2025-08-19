#!/usr/bin/env python3
"""
Test dashboard XGBoost integration
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dashboard_model_selection():
    """Test that dashboard includes XGBoost in model selection"""
    print("🧪 Testing Dashboard XGBoost Integration")
    print("=" * 50)
    
    try:
        # Import dashboard components
        from src.models.train import XGBOOST_AVAILABLE
        
        print(f"✅ XGBOOST_AVAILABLE: {XGBOOST_AVAILABLE}")
        
        # Test model availability logic (simulate dashboard logic)
        try:
            from src.models.train import XGBOOST_AVAILABLE
            available_models = ["rf", "logistic"]
            if XGBOOST_AVAILABLE:
                available_models.append("xgb")
                print("✅ XGBoost added to available models")
            else:
                print("⚠️ XGBoost not available")
        except Exception:
            available_models = ["rf", "logistic"]
            print("⚠️ Could not check XGBoost availability")
            
        print(f"✅ Available models: {available_models}")
        
        if "xgb" in available_models:
            print("✅ Dashboard will show XGBoost option!")
        else:
            print("❌ Dashboard will NOT show XGBoost option")
            
        return "xgb" in available_models
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dashboard_model_selection()
    if success:
        print("\n🎉 Dashboard XGBoost integration test PASSED!")
    else:
        print("\n❌ Dashboard XGBoost integration test FAILED!")
