#!/usr/bin/env python3
"""
Final verification script for real data integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.real_sentiment import get_current_sentiment, get_sentiment_features, get_sentiment_weight

def main():
    """Verify real data integration is working"""
    print("🔍 Verifying Real Data Integration...")
    print("=" * 50)
    
    # Test sentiment data
    print("\n📊 Sentiment Data (CFGI API):")
    current = get_current_sentiment()
    if current:
        print(f"  ✅ Working: {current['value']} ({current['classification']})")
        weight = get_sentiment_weight(current['sentiment_score'])
        print(f"  Weight: {weight:.2f}")
    else:
        print("  ❌ Failed")
        return False
    
    # Test sentiment features
    print("\n📊 Sentiment Features:")
    features = get_sentiment_features(days=30)
    if features:
        print(f"  ✅ Working: {features['fear_greed']} ({features['fear_greed_classification']})")
    else:
        print("  ❌ Failed")
        return False
    
    print("\n✅ Real data integration is working correctly!")
    print("🎯 System is ready for production with real-time sentiment data.")
    print("📝 Note: ETF flows removed from real-time decisions (too stale)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
