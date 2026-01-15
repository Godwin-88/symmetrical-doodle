"""Property-based tests for data validation and error handling."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta
from typing import List

from intelligence_layer.feature_extraction import (
    FeatureExtractor,
    FeatureConfig,
    FeatureValidator,
    WindowSize,
    create_market_window_features,
)
from intelligence_layer.models import MarketData


# Strategies for generating test data
@st.composite
def valid_ohlcv_data(draw):
    """Generate valid OHLCV data."""
    n_points = draw(st.integers(min_value=10, max_value=100))
    base_price = draw(st.floats(min_value=0.1, max_value=1000.0))
    
    data = []
    current_price = base_price
    
    for i in range(n_points):
        # Generate price movement
        price_change = draw(st.floats(min_value=-0.05, max_value=0.05))
        current_price = max(0.01, current_price * (1 + price_change))
        
        # Generate OHLC ensuring valid relationships
        high_offset = draw(st.floats(min_value=0.0, max_value=0.02))
        low_offset = draw(st.floats(min_value=0.0, max_value=0.02))
        
        high = current_price * (1 + high_offset)
        low = current_price * (1 - low_offset)
        
        # Ensure open and close are within high/low range
        open_price = draw(st.floats(min_value=low, max_value=high))
        close_price = draw(st.floats(min_value=low, max_value=high))
        volume = draw(st.floats(min_value=100.0, max_value=10000.0))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume,
        })
    
    return pd.DataFrame(data)


@st.composite
def invalid_ohlcv_data(draw):
    """Generate invalid OHLCV data with various issues."""
    n_points = draw(st.integers(min_value=5, max_value=20))
    issue_type = draw(st.sampled_from([
        'negative_values',
        'invalid_ohlc_relationships',
        'missing_columns',
        'extreme_values'
    ]))
    
    if issue_type == 'missing_columns':
        # Missing required columns
        return pd.DataFrame({
            'open': [1.0, 1.1, 1.2],
            'high': [1.05, 1.15, 1.25],
            # Missing low, close, volume
        })
    
    elif issue_type == 'negative_values':
        # Include negative values
        data = []
        for i in range(n_points):
            data.append({
                'open': draw(st.floats(min_value=-10.0, max_value=10.0)),
                'high': draw(st.floats(min_value=-10.0, max_value=10.0)),
                'low': draw(st.floats(min_value=-10.0, max_value=10.0)),
                'close': draw(st.floats(min_value=-10.0, max_value=10.0)),
                'volume': draw(st.floats(min_value=-1000.0, max_value=1000.0)),
            })
        return pd.DataFrame(data)
    
    elif issue_type == 'invalid_ohlc_relationships':
        # High < Low or other invalid relationships
        data = []
        for i in range(n_points):
            high = draw(st.floats(min_value=0.1, max_value=10.0))
            low = draw(st.floats(min_value=high + 0.1, max_value=high + 5.0))  # Low > High
            open_price = draw(st.floats(min_value=0.1, max_value=10.0))
            close_price = draw(st.floats(min_value=0.1, max_value=10.0))
            volume = draw(st.floats(min_value=100.0, max_value=1000.0))
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
            })
        return pd.DataFrame(data)
    
    else:  # extreme_values
        # Extreme price movements
        data = []
        for i in range(n_points):
            base_price = draw(st.floats(min_value=0.1, max_value=10.0))
            extreme_multiplier = draw(st.floats(min_value=10.0, max_value=100.0))
            
            data.append({
                'open': base_price,
                'high': base_price * extreme_multiplier,  # Extreme movement
                'low': base_price / extreme_multiplier,
                'close': base_price * draw(st.floats(min_value=0.1, max_value=10.0)),
                'volume': draw(st.floats(min_value=100.0, max_value=1000.0)),
            })
        return pd.DataFrame(data)


@st.composite
def market_data_list(draw):
    """Generate list of MarketData objects."""
    n_points = draw(st.integers(min_value=50, max_value=200))
    asset_id = draw(st.text(min_size=3, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
    
    data = []
    base_time = datetime.now() - timedelta(hours=n_points)
    base_price = draw(st.floats(min_value=0.1, max_value=1000.0))
    current_price = base_price
    
    for i in range(n_points):
        # Generate realistic price movement
        price_change = draw(st.floats(min_value=-0.02, max_value=0.02))
        current_price = max(0.01, current_price * (1 + price_change))
        
        # Generate OHLC with valid relationships
        high_offset = draw(st.floats(min_value=0.0, max_value=0.01))
        low_offset = draw(st.floats(min_value=0.0, max_value=0.01))
        
        high = current_price * (1 + high_offset)
        low = current_price * (1 - low_offset)
        
        open_price = draw(st.floats(min_value=low, max_value=high))
        close_price = draw(st.floats(min_value=low, max_value=high))
        volume = draw(st.floats(min_value=100.0, max_value=10000.0))
        
        data.append(MarketData(
            timestamp=base_time + timedelta(hours=i),
            asset_id=asset_id,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume,
        ))
    
    return data


class TestDataValidationProperties:
    """Property-based tests for data validation and error handling.
    
    **Validates: Requirements 9.3, 9.5**
    """
    
    @given(valid_ohlcv_data())
    @settings(max_examples=50, deadline=5000)
    def test_property_valid_data_passes_validation(self, ohlcv_data):
        """
        Property: For any valid OHLCV data, validation should pass.
        
        **Feature: algorithmic-trading-system, Property 22: Data Validation and Error Handling**
        **Validates: Requirements 9.3, 9.5**
        """
        is_valid, errors = FeatureValidator.validate_ohlcv_data(ohlcv_data)
        
        # Valid data should pass validation
        assert is_valid, f"Valid data failed validation with errors: {errors}"
        assert len(errors) == 0
    
    @given(invalid_ohlcv_data())
    @settings(max_examples=50, deadline=5000)
    def test_property_invalid_data_fails_validation(self, invalid_data):
        """
        Property: For any invalid OHLCV data, validation should fail with appropriate errors.
        
        **Feature: algorithmic-trading-system, Property 22: Data Validation and Error Handling**
        **Validates: Requirements 9.3, 9.5**
        """
        is_valid, errors = FeatureValidator.validate_ohlcv_data(invalid_data)
        
        # Invalid data should fail validation
        assert not is_valid, "Invalid data passed validation when it should have failed"
        assert len(errors) > 0, "No error messages provided for invalid data"
        
        # Error messages should be descriptive
        for error in errors:
            assert isinstance(error, str)
            assert len(error) > 0
    
    @given(market_data_list())
    @settings(max_examples=30, deadline=10000)
    def test_property_feature_extraction_handles_valid_data(self, market_data):
        """
        Property: For any valid market data, feature extraction should complete without errors.
        
        **Feature: algorithmic-trading-system, Property 22: Data Validation and Error Handling**
        **Validates: Requirements 9.3, 9.5**
        """
        assume(len(market_data) >= 64)  # Ensure we have enough data for largest window
        
        config = FeatureConfig(
            window_sizes=[WindowSize.SMALL],  # Use smallest window for faster testing
            normalize=True
        )
        
        try:
            result = create_market_window_features(market_data, config)
            
            # Should complete successfully
            assert 'features' in result
            assert 'feature_names' in result
            assert 'data_points' in result
            assert result['data_points'] == len(market_data)
            
            # Features should be valid
            features = result['features']
            assert len(features) > 0
            
            for window_key, feature_array in features.items():
                assert isinstance(feature_array, np.ndarray)
                assert feature_array.shape[0] > 0  # Should have some sequences
                assert feature_array.shape[1] > 0  # Should have some features
                
                # No NaN or infinite values in normalized features
                assert not np.isnan(feature_array).any(), f"NaN values found in {window_key}"
                assert not np.isinf(feature_array).any(), f"Infinite values found in {window_key}"
                
        except Exception as e:
            pytest.fail(f"Feature extraction failed on valid data: {str(e)}")
    
    @given(st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=10, max_size=100))
    @settings(max_examples=50, deadline=3000)
    def test_property_feature_validation_rejects_invalid_features(self, invalid_features):
        """
        Property: For any feature array containing NaN or infinite values, validation should fail.
        
        **Feature: algorithmic-trading-system, Property 22: Data Validation and Error Handling**
        **Validates: Requirements 9.3, 9.5**
        """
        # Convert to numpy array
        feature_array = np.array(invalid_features).reshape(-1, 1)
        
        # Only test if we actually have invalid values
        has_nan = np.isnan(feature_array).any()
        has_inf = np.isinf(feature_array).any()
        assume(has_nan or has_inf)
        
        is_valid, errors = FeatureValidator.validate_features(feature_array)
        
        # Should fail validation
        assert not is_valid, "Invalid features passed validation"
        assert len(errors) > 0, "No error messages for invalid features"
        
        # Should have appropriate error messages
        if has_nan:
            assert any("NaN" in error for error in errors), "Missing NaN error message"
        if has_inf:
            assert any("infinite" in error for error in errors), "Missing infinite error message"
    
    @given(st.integers(min_value=1, max_value=5), st.integers(min_value=10, max_value=50))
    @settings(max_examples=30, deadline=5000)
    def test_property_empty_data_handled_gracefully(self, n_features, n_attempts):
        """
        Property: For any empty or insufficient data, the system should handle it gracefully.
        
        **Feature: algorithmic-trading-system, Property 22: Data Validation and Error Handling**
        **Validates: Requirements 9.3, 9.5**
        """
        config = FeatureConfig(window_sizes=[WindowSize.SMALL])
        
        # Test with empty data
        try:
            create_market_window_features([], config)
            pytest.fail("Should have raised ValueError for empty data")
        except ValueError as e:
            assert "Empty market data" in str(e)
        except Exception as e:
            pytest.fail(f"Wrong exception type for empty data: {type(e).__name__}: {str(e)}")
        
        # Test with insufficient data (less than window size)
        insufficient_data = []
        base_time = datetime.now()
        
        for i in range(min(10, WindowSize.SMALL.value - 1)):  # Less than window size
            insufficient_data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                asset_id="TEST",
                open=1.0,
                high=1.01,
                low=0.99,
                close=1.0,
                volume=1000.0,
            ))
        
        # Should handle gracefully (may return empty features or raise appropriate error)
        try:
            result = create_market_window_features(insufficient_data, config)
            # If it succeeds, should have valid structure
            assert 'features' in result
            assert 'data_points' in result
        except ValueError as e:
            # Acceptable to raise ValueError for insufficient data
            assert len(str(e)) > 0
        except Exception as e:
            pytest.fail(f"Unexpected exception for insufficient data: {type(e).__name__}: {str(e)}")
    
    @given(
        st.integers(min_value=100, max_value=500),
        st.sampled_from([WindowSize.SMALL, WindowSize.MEDIUM, WindowSize.LARGE])
    )
    @settings(max_examples=20, deadline=10000)
    def test_property_feature_extraction_consistency(self, n_points, window_size):
        """
        Property: For any valid dataset, feature extraction should be deterministic and consistent.
        
        **Feature: algorithmic-trading-system, Property 22: Data Validation and Error Handling**
        **Validates: Requirements 9.3, 9.5**
        """
        # Generate consistent test data
        np.random.seed(42)  # Fixed seed for reproducibility
        
        market_data = []
        base_time = datetime.now()
        base_price = 100.0
        
        for i in range(n_points):
            price_change = np.random.normal(0, 0.01)
            base_price = max(0.01, base_price * (1 + price_change))
            
            high = base_price * (1 + abs(np.random.normal(0, 0.005)))
            low = base_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = np.random.uniform(low, high)
            close_price = np.random.uniform(low, high)
            volume = np.random.uniform(1000, 10000)
            
            market_data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                asset_id="TESTPAIR",
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
            ))
        
        config = FeatureConfig(
            window_sizes=[window_size],
            normalize=True
        )
        
        # Extract features twice
        result1 = create_market_window_features(market_data, config)
        result2 = create_market_window_features(market_data, config)
        
        # Results should be identical
        assert result1['data_points'] == result2['data_points']
        
        window_key = f"window_{window_size.value}"
        features1 = result1['features'][window_key]
        features2 = result2['features'][window_key]
        
        # Features should be numerically identical
        np.testing.assert_array_equal(features1, features2, 
                                    err_msg="Feature extraction is not deterministic")
        
        # Features should have expected properties
        assert features1.shape == features2.shape
        assert features1.shape[0] > 0  # Should have sequences
        assert features1.shape[1] > 0  # Should have features
        
        # Normalized features should be reasonable
        assert np.abs(features1).max() <= 10  # Should be clipped
        assert not np.isnan(features1).any()
        assert not np.isinf(features1).any()


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])