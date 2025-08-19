#!/usr/bin/env python3
"""
Test script for data providers in alpha12_24
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.binance_free import BinanceFreeProvider
from src.data.bybit_free import BybitFreeProvider
from src.data.deribit_free import DeribitFreeProvider
from src.data.etf_flows import ETFFlowsProvider
import pandas as pd

def test_binance_provider():
    """Test Binance data provider"""
    print("Testing Binance Provider...")
    
    provider = BinanceFreeProvider()
    
    # Test assemble function
    df = provider.assemble(symbol="BTCUSDT", interval="5m", limit=100)
    
    print(f"Binance data shape: {df.shape}")
    print(f"Binance columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    print()
    
    return df

def test_bybit_provider():
    """Test Bybit data provider"""
    print("Testing Bybit Provider...")
    
    provider = BybitFreeProvider()
    
    # Test assemble function
    df = provider.assemble(symbol="BTCUSDT", interval="5m", limit=100)
    
    print(f"Bybit data shape: {df.shape}")
    print(f"Bybit columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    print()
    
    return df

def test_deribit_provider():
    """Test Deribit data provider"""
    print("Testing Deribit Provider...")
    
    provider = DeribitFreeProvider()
    
    # Test hourly_iv_rr function
    df = provider.hourly_iv_rr(currency="BTC")
    
    print(f"Deribit IV/RR data shape: {df.shape}")
    print(f"Deribit columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    print()
    
    return df

def test_etf_flows_provider():
    """Test ETF flows provider"""
    print("Testing ETF Flows Provider...")
    
    provider = ETFFlowsProvider()
    
    # Test synthetic data generation
    grayscale = provider.get_grayscale_flows(days=7)
    spot = provider.get_spot_etf_flows(days=7)
    futures = provider.get_futures_etf_flows(days=7)
    aggregated = provider.get_aggregated_flows(days=7)
    sentiment = provider.get_flow_sentiment(days=7)
    
    print(f"Grayscale flows shape: {grayscale.shape}")
    print(f"Spot ETF flows shape: {spot.shape}")
    print(f"Futures ETF flows shape: {futures.shape}")
    print(f"Aggregated flows shape: {aggregated.shape}")
    print(f"Flow sentiment shape: {sentiment.shape}")
    print()
    
    print("Sample aggregated flows:")
    print(aggregated.head())
    print()
    
    return aggregated

def test_csv_loading():
    """Test CSV loading functionality"""
    print("Testing CSV Loading...")
    
    provider = ETFFlowsProvider()
    
    # Create a sample CSV URL (this will fail but tests the error handling)
    try:
        df = provider.load_csv_url("https://example.com/nonexistent.csv")
        print(f"CSV loading result: {df.shape}")
    except Exception as e:
        print(f"CSV loading error (expected): {e}")
    
    print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Data Providers for alpha12_24")
    print("=" * 60)
    print()
    
    try:
        # Test each provider
        binance_df = test_binance_provider()
        bybit_df = test_bybit_provider()
        deribit_df = test_deribit_provider()
        etf_df = test_etf_flows_provider()
        test_csv_loading()
        
        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
        # Summary
        print("\nSummary:")
        print(f"- Binance: {len(binance_df)} rows with {len(binance_df.columns)} columns")
        print(f"- Bybit: {len(bybit_df)} rows with {len(bybit_df.columns)} columns")
        print(f"- Deribit: {len(deribit_df)} rows with {len(deribit_df.columns)} columns")
        print(f"- ETF Flows: {len(etf_df)} rows with {len(etf_df.columns)} columns")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
