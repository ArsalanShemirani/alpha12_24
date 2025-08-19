#!/usr/bin/env python3
"""
Test direction-specific sentiment weighting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.real_sentiment import get_direction_specific_sentiment_weight

def test_direction_specific_sentiment():
    """Test direction-specific sentiment weighting"""
    
    print("🎯 DIRECTION-SPECIFIC SENTIMENT WEIGHTING (UPDATED)")
    print("=" * 70)
    
    # Test different CFGI values based on user requirements
    test_cases = [
        (10, "Extreme Fear"),
        (20, "Fear"), 
        (30, "Fear"),
        (40, "Fear"),
        (50, "Neutral"),
        (60, "Greed"),
        (70, "Greed"),
        (80, "Greed"),
        (90, "Extreme Greed"),
        (95, "Extreme Greed")
    ]
    
    print(f"{'CFGI':<4} {'Classification':<15} {'LONG Weight':<12} {'SHORT Weight':<12} {'LONG Impact':<12} {'SHORT Impact':<12}")
    print("-" * 80)
    
    for cfgi_value, classification in test_cases:
        # Calculate sentiment score (-1 to +1)
        sentiment_score = (cfgi_value - 50) / 50
        
        # Get direction-specific weights
        long_weight = get_direction_specific_sentiment_weight(sentiment_score, "long")
        short_weight = get_direction_specific_sentiment_weight(sentiment_score, "short")
        
        # Calculate impact on confidence
        base_confidence = 0.75  # 75%
        long_confidence = base_confidence * long_weight
        short_confidence = base_confidence * short_weight
        
        long_impact = f"{((long_weight - 1.0) * 100):+.1f}%"
        short_impact = f"{((short_weight - 1.0) * 100):+.1f}%"
        
        print(f"{cfgi_value:<4} {classification:<15} {long_weight:<12.3f} {short_weight:<12.3f} {long_impact:<12} {short_impact:<12}")
    
    print("\n" + "=" * 70)
    print("📊 INTERPRETATION:")
    print("• LONG Impact: How sentiment affects LONG setup confidence")
    print("• SHORT Impact: How sentiment affects SHORT setup confidence")
    print("• Positive % = Confidence INCREASED")
    print("• Negative % = Confidence DECREASED")
    print("\n🎯 CONTRARIAN LOGIC:")
    print("• Fear → Favors LONG setups (buy when others are fearful)")
    print("• Greed → Favors SHORT setups (sell when others are greedy)")
    
    # Test specific examples from user requirements
    print(f"\n🔍 USER REQUIREMENTS VERIFICATION:")
    print(f"CFGI 70: LONG {(get_direction_specific_sentiment_weight((70-50)/50, 'long') - 1.0) * 100:+.1f}% | SHORT {(get_direction_specific_sentiment_weight((70-50)/50, 'short') - 1.0) * 100:+.1f}%")
    print(f"CFGI 80: LONG {(get_direction_specific_sentiment_weight((80-50)/50, 'long') - 1.0) * 100:+.1f}% | SHORT {(get_direction_specific_sentiment_weight((80-50)/50, 'short') - 1.0) * 100:+.1f}%")
    print(f"CFGI 90: LONG {(get_direction_specific_sentiment_weight((90-50)/50, 'long') - 1.0) * 100:+.1f}% | SHORT {(get_direction_specific_sentiment_weight((90-50)/50, 'short') - 1.0) * 100:+.1f}%")
    print(f"CFGI 10: LONG {(get_direction_specific_sentiment_weight((10-50)/50, 'long') - 1.0) * 100:+.1f}% | SHORT {(get_direction_specific_sentiment_weight((10-50)/50, 'short') - 1.0) * 100:+.1f}%")

if __name__ == "__main__":
    test_direction_specific_sentiment()
