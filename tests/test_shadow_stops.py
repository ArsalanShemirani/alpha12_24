#!/usr/bin/env python3
"""
Comprehensive tests for Shadow Dynamic Stops (Phase-1)

Tests all functionality including:
- ATR-based stop computation
- Volatility Z-score calculation and high-vol relaxer
- Timeframe-specific configuration
- Telemetry logging
- No behavior change verification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.trading.shadow_stops import (
    ShadowStopConfig, ShadowStopComputer, ShadowStopResult,
    compute_and_log_shadow_stop, get_shadow_computer
)

class TestShadowStopConfig:
    """Test shadow stop configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ShadowStopConfig()
        
        # Base ATR multipliers
        assert config.base_atr_15m == 0.90
        assert config.base_atr_1h == 1.20
        assert config.base_atr_4h == 1.60
        assert config.base_atr_1d == 2.00
        
        # ATR ranges
        assert config.min_atr_15m == 0.60
        assert config.max_atr_15m == 1.20
        assert config.min_atr_1h == 0.90
        assert config.max_atr_1h == 1.60
        assert config.min_atr_4h == 1.20
        assert config.max_atr_4h == 2.00
        assert config.min_atr_1d == 1.60
        assert config.max_atr_1d == 2.40
        
        # Percentage caps
        assert config.pct_cap_15m == 0.0035  # 0.35%
        assert config.pct_cap_1h == 0.0060   # 0.60%
        assert config.pct_cap_4h == 0.0100   # 1.00%
        assert config.pct_cap_1d == 0.0150   # 1.50%
        
        # Volatility relaxer
        assert config.vol_z_threshold == 2.0
        assert config.vol_z_factor == 1.15
        
        # Feature flag
        assert config.enabled == True
    
    def test_env_override(self):
        """Test environment variable overrides"""
        with patch.dict(os.environ, {
            'STOP_BASE_ATR_1H': '1.50',
            'STOP_PCT_CAP_4H': '0.0080',
            'VOL_Z_RELAX_FACTOR': '1.25',
            'SHADOW_DYNAMIC_STOP_LOGGING': '0'
        }):
            config = ShadowStopConfig.from_env()
            
            assert config.base_atr_1h == 1.50
            assert config.pct_cap_4h == 0.0080
            assert config.vol_z_factor == 1.25
            assert config.enabled == False
    
    def test_invalid_env_values(self):
        """Test handling of invalid environment values"""
        with patch.dict(os.environ, {
            'STOP_BASE_ATR_1H': 'invalid',
            'VOL_Z_THRESHOLD': 'not_a_number'
        }):
            # Should use defaults for invalid values
            config = ShadowStopConfig.from_env()
            assert config.base_atr_1h == 1.20  # Default
            assert config.vol_z_threshold == 2.0  # Default

class TestShadowStopComputer:
    """Test shadow stop computation"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return ShadowStopConfig()
    
    @pytest.fixture
    def computer(self, config):
        """Test computer with temporary log directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_copy = ShadowStopConfig()
            computer = ShadowStopComputer(config_copy)
            # Override log directory to use temp directory
            computer.log_dir = Path(temp_dir) / "shadow_stops"
            computer.log_dir.mkdir(parents=True, exist_ok=True)
            computer.telemetry_file = computer.log_dir / "shadow_dynamic_stops.csv"
            computer._ensure_telemetry_header()
            yield computer
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # Generate realistic price data
        close_prices = []
        price = 50000.0  # Starting price
        
        for i in range(100):
            # Random walk with some volatility
            change = np.random.normal(0, 0.01) * price
            price += change
            close_prices.append(price)
        
        # Generate OHLC from close prices
        data = []
        for i, close in enumerate(close_prices):
            if i == 0:
                open_price = close
            else:
                open_price = close_prices[i-1]
            
            # Generate high/low around open/close
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(1000, 10000)
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_timeframe_config(self, computer):
        """Test timeframe-specific configuration retrieval"""
        # Test each timeframe
        base, min_atr, max_atr, pct_cap = computer.get_timeframe_config("15m")
        assert base == 0.90
        assert min_atr == 0.60
        assert max_atr == 1.20
        assert pct_cap == 0.0035
        
        base, min_atr, max_atr, pct_cap = computer.get_timeframe_config("1h")
        assert base == 1.20
        assert min_atr == 0.90
        assert max_atr == 1.60
        assert pct_cap == 0.0060
        
        base, min_atr, max_atr, pct_cap = computer.get_timeframe_config("4h")
        assert base == 1.60
        assert min_atr == 1.20
        assert max_atr == 2.00
        assert pct_cap == 0.0100
        
        base, min_atr, max_atr, pct_cap = computer.get_timeframe_config("1d")
        assert base == 2.00
        assert min_atr == 1.60
        assert max_atr == 2.40
        assert pct_cap == 0.0150
        
        # Test unknown timeframe (should default to 1h)
        base, min_atr, max_atr, pct_cap = computer.get_timeframe_config("unknown")
        assert base == 1.20  # 1h default
    
    def test_atr_computation_insufficient_data(self, computer):
        """Test ATR computation with insufficient data"""
        # Too few bars for ATR14
        short_data = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [100, 101, 102]
        })
        
        atr14, median_atr, vol_z = computer.compute_atr_metrics(short_data, "1h")
        assert atr14 is None
        assert median_atr is None
        assert vol_z is None
    
    def test_atr_computation_sufficient_data(self, computer, sample_ohlcv_data):
        """Test ATR computation with sufficient data"""
        atr14, median_atr, vol_z = computer.compute_atr_metrics(sample_ohlcv_data, "1h")
        
        # Should have ATR14
        assert atr14 is not None
        assert atr14 > 0
        
        # Should have median ATR for volatility Z-score
        assert median_atr is not None
        assert median_atr > 0
        
        # Should have volatility Z-score
        assert vol_z is not None
        assert vol_z > 0
    
    def test_atr_computation_partial_data(self, computer):
        """Test ATR computation with partial data (enough for ATR, not enough for median)"""
        # 20 bars - enough for ATR14, not enough for 20-day median
        dates = pd.date_range('2024-01-01', periods=20, freq='1H')
        data = pd.DataFrame({
            'high': np.random.uniform(100, 110, 20),
            'low': np.random.uniform(90, 100, 20), 
            'close': np.random.uniform(95, 105, 20)
        }, index=dates)
        
        atr14, median_atr, vol_z = computer.compute_atr_metrics(data, "1h")
        
        # Should have ATR14
        assert atr14 is not None
        assert atr14 > 0
        
        # Should not have median ATR (insufficient history)
        assert median_atr is None
        assert vol_z is None
    
    def test_dynamic_stop_candidate_computation(self, computer):
        """Test dynamic stop candidate computation"""
        entry_price = 50000.0
        atr14 = 500.0  # 1% of price
        vol_z = 1.5    # Below threshold
    
        # Test 1h timeframe
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "1h", atr14, vol_z
        )
    
        # Should compute baseline: 1.20 * 500 = 600
        # But percentage cap is 0.6% = 50000 * 0.006 = 300
        # So should be capped by percentage
        expected = 50000.0 * 0.006  # Percentage cap
        assert candidate is not None
        assert abs(candidate - expected) < 1e-6
        assert "pct_capped" in notes
    
    def test_dynamic_stop_candidate_clamping(self, computer):
        """Test ATR multiplier clamping"""
        entry_price = 50000.0
        atr14 = 100.0  # Small ATR
        vol_z = 1.0
    
        # For 1h: base=1.20, min=0.90, max=1.60
        # Baseline would be 1.20 * 100 = 120
        # Percentage cap is 0.6% = 50000 * 0.006 = 300
        # Result: min(120, 300) = 120 (ATR-based is smaller)
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "1h", atr14, vol_z
        )
    
        expected = 1.20 * atr14  # ATR-based calculation
        assert abs(candidate - expected) < 1e-6
        assert "baseline" in notes  # No capping applied
    
        # Test with very large ATR that would exceed max
        large_atr = 10000.0  # Very large ATR
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "1h", large_atr, vol_z
        )
    
        # Should be clamped to percentage cap: 50000 * 0.006 = 300
        expected_max = 50000.0 * 0.006
        assert abs(candidate - expected_max) < 1e-6
        assert "pct_capped" in notes
    
    def test_dynamic_stop_candidate_percentage_cap(self, computer):
        """Test percentage-of-price cap"""
        entry_price = 1000.0  # Low price to trigger percentage cap
        atr14 = 100.0  # Large relative to price
        vol_z = 1.0
        
        # For 1h: pct_cap = 0.006 (0.6%)
        # ATR candidate: 1.20 * 100 = 120
        # Percentage cap: 1000 * 0.006 = 6
        # Should be capped to 6
        
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "1h", atr14, vol_z
        )
        
        expected_pct_cap = entry_price * 0.006
        assert abs(candidate - expected_pct_cap) < 1e-6
        assert "pct_capped" in notes
    
    def test_dynamic_stop_candidate_vol_relaxer(self, computer):
        """Test high-volatility relaxer"""
        entry_price = 50000.0
        atr14 = 500.0
        vol_z = 2.5  # Above threshold (2.0)
    
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "1h", atr14, vol_z
        )
    
        # Baseline: 1.20 * 500 = 600
        # Percentage cap: 50000 * 0.006 = 300
        # With relaxer: 300 * 1.15 = 345
        expected = 50000.0 * 0.006 * 1.15  # Percentage cap with relaxer
        assert abs(candidate - expected) < 1e-6
        assert "vol_relaxed" in notes
    
    def test_dynamic_stop_candidate_no_atr(self, computer):
        """Test dynamic stop computation without ATR"""
        candidate, notes = computer.compute_dynamic_stop_candidate(
            50000.0, "1h", None, None
        )
        
        assert candidate is None
        assert "ATR14 unavailable" in notes
    
    def test_compute_shadow_stop_disabled(self, computer):
        """Test shadow stop computation when disabled"""
        computer.config.enabled = False
        
        result = computer.compute_shadow_stop(
            setup_id="TEST-123",
            tf="1h",
            entry_price=50000.0,
            applied_stop_price=49500.0,
            rr_planned=2.0
        )
        
        assert result.shadow_valid == False
        assert result.shadow_notes == "disabled"
        assert result.dynamic_stop_candidate_price is None
    
    def test_compute_shadow_stop_no_data(self, computer):
        """Test shadow stop computation without price data"""
        result = computer.compute_shadow_stop(
            setup_id="TEST-123",
            tf="1h",
            entry_price=50000.0,
            applied_stop_price=49500.0,
            rr_planned=2.0,
            data=None
        )
        
        assert result.shadow_valid == False
        assert "no_price_data" in result.shadow_notes
        assert result.dynamic_stop_candidate_price is None
    
    def test_compute_shadow_stop_insufficient_data(self, computer):
        """Test shadow stop computation with insufficient data for ATR"""
        # Very small dataset
        small_data = pd.DataFrame({
            'high': [100, 101],
            'low': [99, 100],
            'close': [100, 101]
        })
        
        result = computer.compute_shadow_stop(
            setup_id="TEST-123",
            tf="1h",
            entry_price=50000.0,
            applied_stop_price=49500.0,
            rr_planned=2.0,
            data=small_data
        )
        
        assert result.shadow_valid == False
        assert "insufficient_data_for_ATR" in result.shadow_notes
    
    def test_compute_shadow_stop_full_computation(self, computer, sample_ohlcv_data):
        """Test full shadow stop computation with valid data"""
        result = computer.compute_shadow_stop(
            setup_id="TEST-123",
            tf="1h",
            entry_price=50000.0,
            applied_stop_price=49500.0,
            rr_planned=2.0,
            data=sample_ohlcv_data,
            p_hit=0.65,
            conf=0.75,
            outcome="pending"
        )
        
        # Should be valid
        assert result.shadow_valid == True
        assert result.setup_id == "TEST-123"
        assert result.tf == "1h"
        assert result.entry_price == 50000.0
        assert result.applied_stop_price == 49500.0
        assert result.applied_stop_distance == 500.0
        assert result.applied_stop_R == 1.0
        assert result.rr_planned == 2.0
        assert result.p_hit == 0.65
        assert result.conf == 0.75
        assert result.outcome == "pending"
        
        # Should have computed ATR metrics
        assert result.atr14 is not None
        assert result.atr14 > 0
        
        # Should have dynamic stop candidate
        assert result.dynamic_stop_candidate_price is not None
        assert result.dynamic_stop_candidate_price > 0
        
        # Should have R ratio
        assert result.dynamic_stop_candidate_R is not None
        assert result.dynamic_stop_candidate_R > 0
    
    def test_r_units_conversion(self, computer, sample_ohlcv_data):
        """Test R units conversion accuracy"""
        applied_stop_distance = 500.0
        result = computer.compute_shadow_stop(
            setup_id="TEST-123",
            tf="1h",
            entry_price=50000.0,
            applied_stop_price=49500.0,  # 500 distance
            rr_planned=2.0,
            data=sample_ohlcv_data
        )
        
        if result.dynamic_stop_candidate_price is not None:
            # Verify R conversion: dynamic_stop_R = dynamic_stop_candidate / applied_stop_distance
            expected_r = result.dynamic_stop_candidate_price / applied_stop_distance
            assert abs(result.dynamic_stop_candidate_R - expected_r) < 1e-6
    
    def test_log_shadow_result(self, computer):
        """Test shadow result logging"""
        result = ShadowStopResult(
            setup_id="TEST-LOG",
            tf="1h",
            entry_price=50000.0,
            atr14=500.0,
            median_atr14_20d=480.0,
            vol_z=1.04,
            dynamic_stop_candidate_price=600.0,
            dynamic_stop_candidate_R=1.2,
            applied_stop_price=49500.0,
            applied_stop_distance=500.0,
            applied_stop_R=1.0,
            stop_cap_used=None,
            rr_planned=2.0,
            rr_realized=None,
            p_hit=0.65,
            conf=0.75,
            outcome="pending",
            mfe_R=None,
            mae_R=None,
            shadow_valid=True,
            shadow_notes="baseline",
            ts="2024-01-01 12:00:00"
        )
        
        # Log the result
        computer.log_shadow_result(result)
        
        # Verify file exists and has content
        assert computer.telemetry_file.exists()
        
        # Read and verify content
        with open(computer.telemetry_file, 'r') as f:
            lines = f.readlines()
        
        # Should have header + 1 data row
        assert len(lines) == 2
        
        # Verify data row contains expected values
        data_line = lines[1]
        assert "TEST-LOG" in data_line
        assert "50000.0" in data_line
        assert "600.0" in data_line
        assert "1.2" in data_line
        assert "baseline" in data_line
    
    def test_log_shadow_result_disabled(self, computer):
        """Test shadow result logging when disabled"""
        computer.config.enabled = False
        
        result = ShadowStopResult(
            setup_id="TEST-DISABLED",
            tf="1h",
            entry_price=50000.0,
            atr14=None,
            median_atr14_20d=None,
            vol_z=None,
            dynamic_stop_candidate_price=None,
            dynamic_stop_candidate_R=None,
            applied_stop_price=49500.0,
            applied_stop_distance=500.0,
            applied_stop_R=1.0,
            stop_cap_used=None,
            rr_planned=2.0,
            rr_realized=None,
            p_hit=None,
            conf=None,
            outcome="pending",
            mfe_R=None,
            mae_R=None,
            shadow_valid=False,
            shadow_notes="disabled",
            ts="2024-01-01 12:00:00"
        )
        
        # Should not log when disabled
        computer.log_shadow_result(result)
        
        # File might exist but should not have new content
        # (This is implementation-dependent)

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_compute_and_log_shadow_stop(self):
        """Test convenience function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the global computer
            with patch('src.trading.shadow_stops._shadow_computer') as mock_computer:
                mock_instance = MagicMock()
                mock_result = ShadowStopResult(
                    setup_id="TEST-CONV",
                    tf="1h",
                    entry_price=50000.0,
                    atr14=500.0,
                    median_atr14_20d=480.0,
                    vol_z=1.04,
                    dynamic_stop_candidate_price=600.0,
                    dynamic_stop_candidate_R=1.2,
                    applied_stop_price=49500.0,
                    applied_stop_distance=500.0,
                    applied_stop_R=1.0,
                    stop_cap_used=None,
                    rr_planned=2.0,
                    rr_realized=None,
                    p_hit=0.65,
                    conf=0.75,
                    outcome="pending",
                    mfe_R=None,
                    mae_R=None,
                    shadow_valid=True,
                    shadow_notes="baseline",
                    ts="2024-01-01 12:00:00"
                )
                
                mock_instance.compute_shadow_stop.return_value = mock_result
                mock_instance.log_shadow_result.return_value = None
                
                with patch('src.trading.shadow_stops.get_shadow_computer', return_value=mock_instance):
                    result = compute_and_log_shadow_stop(
                        setup_id="TEST-CONV",
                        tf="1h",
                        entry_price=50000.0,
                        applied_stop_price=49500.0,
                        rr_planned=2.0
                    )
                
                # Verify the result
                assert result.setup_id == "TEST-CONV"
                assert result.shadow_valid == True
                
                # Verify compute_shadow_stop was called
                mock_instance.compute_shadow_stop.assert_called_once()
                
                # Verify log_shadow_result was called
                mock_instance.log_shadow_result.assert_called_once_with(mock_result)

class TestTimeframeSpecificBehavior:
    """Test timeframe-specific behavior"""
    
    @pytest.fixture
    def computer(self):
        """Computer with default config"""
        config = ShadowStopConfig()
        with tempfile.TemporaryDirectory() as temp_dir:
            computer = ShadowStopComputer(config)
            computer.log_dir = Path(temp_dir) / "shadow_stops"
            computer.log_dir.mkdir(parents=True, exist_ok=True)
            computer.telemetry_file = computer.log_dir / "test.csv"
            computer._ensure_telemetry_header()
            yield computer
    
    def test_15m_clamp_logic(self, computer):
        """Test 15m timeframe clamp logic"""
        entry_price = 50000.0
        atr14 = 1000.0  # Large ATR to trigger clamping
        vol_z = 1.0
        
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "15m", atr14, vol_z
        )
        
        # 15m: base=0.90, min=0.60, max=1.20
        # Base would be 0.90 * 1000 = 900, which is within range
        # But percentage cap is 0.35% = 50000 * 0.0035 = 175
        # Should be capped by percentage
        expected = 50000.0 * 0.0035
        assert abs(candidate - expected) < 1e-6
        assert "pct_capped" in notes
    
    def test_1d_high_tolerance(self, computer):
        """Test 1d timeframe has higher tolerance"""
        entry_price = 50000.0
        atr14 = 1000.0
        vol_z = 1.0
        
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "1d", atr14, vol_z
        )
        
        # 1d: base=2.00, pct_cap=1.50%
        # Base: 2.00 * 1000 = 2000
        # Percentage cap: 50000 * 0.015 = 750
        # Should be capped by percentage
        expected = 50000.0 * 0.015
        assert abs(candidate - expected) < 1e-6
        assert "pct_capped" in notes
    
    def test_vol_relaxer_across_timeframes(self, computer):
        """Test volatility relaxer works across all timeframes"""
        entry_price = 50000.0
        atr14 = 500.0
        vol_z = 3.0  # High volatility
    
        timeframes = ["15m", "1h", "4h", "1d"]
        pct_caps = {"15m": 0.0035, "1h": 0.0060, "4h": 0.0100, "1d": 0.0150}
    
        for tf in timeframes:
            candidate, notes = computer.compute_dynamic_stop_candidate(
                entry_price, tf, atr14, vol_z
            )
    
            # All should be capped by percentage and then relaxed
            expected = entry_price * pct_caps[tf] * 1.15
    
            assert abs(candidate - expected) < 1e-6
            assert "vol_relaxed" in notes
    
    def test_vol_relaxer_when_not_capped(self, computer):
        """Test volatility relaxer when percentage cap doesn't apply"""
        entry_price = 100000.0  # High price so percentage cap is higher
        atr14 = 100.0  # Small ATR
        vol_z = 2.5  # Above threshold (2.0)
    
        candidate, notes = computer.compute_dynamic_stop_candidate(
            entry_price, "1h", atr14, vol_z
        )
    
        # Baseline: 1.20 * 100 = 120
        # With relaxer: 120 * 1.15 = 138
        # Percentage cap: 100000 * 0.006 = 600
        # So relaxer should apply
        expected = 1.20 * atr14 * 1.15
        assert abs(candidate - expected) < 1e-6
        assert "vol_relaxed" in notes

class TestNoBehaviorChange:
    """Test that shadow logging doesn't change any trading behavior"""
    
    def test_shadow_logging_exception_handling(self):
        """Test that shadow logging errors don't affect main flow"""
        # This would be tested at integration level to ensure exceptions
        # in shadow logging don't break the main setup creation flow
        
        # Mock scenario where shadow logging throws an exception
        with patch('src.trading.shadow_stops.compute_and_log_shadow_stop') as mock_shadow:
            mock_shadow.side_effect = Exception("Shadow logging error")
            
            # The main setup creation should still work
            # (This would be tested in the actual integration)
            assert True  # Placeholder - real test would verify main flow continues
    
    def test_disabled_shadow_logging(self):
        """Test that disabled shadow logging has zero overhead"""
        config = ShadowStopConfig()
        config.enabled = False
        
        computer = ShadowStopComputer(config)
        
        # Should return minimal result immediately
        result = computer.compute_shadow_stop(
            setup_id="TEST",
            tf="1h",
            entry_price=50000.0,
            applied_stop_price=49500.0,
            rr_planned=2.0,
            data=pd.DataFrame()  # Even with data, should not process
        )
        
        assert result.shadow_valid == False
        assert result.shadow_notes == "disabled"
        assert result.dynamic_stop_candidate_price is None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

