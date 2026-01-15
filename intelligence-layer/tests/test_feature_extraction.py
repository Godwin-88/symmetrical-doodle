"""Tests for feature extraction pipeline."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from intelligence_layer.feature_extraction import (
    FeatureExtractor,
    FeatureConfig,
    WindowSize,
    FeatureValidator,
    TechnicalIndicators,
    VolatilityFeatures,
    LiquidityFeatures,
    create_market_window_features,
)
from intelligence_layer.models import MarketData


def create_sample_market_data(n_points: int = 100, asset_id: str = "EURUSD") -> List[MarketData]:
    """Create sample market data for testing."""
    data = []
    base_price = 1.1000
    base_time = datetime.now() - timedelta(hours=n_points)
    
    for i in range(n_points):
        # Simple random walk with some volatility
        price_change = np.random.normal(0, 0.001)
        base_price *= (1 + price_change)
        
        # Generate OHLC around the base price ensuring valid relationships
        high_offset = abs(np.random.normal(0, 0.0005))
        low_offset = abs(np.random.normal(0, 0.0005))
        
        high = base_price * (1 + high_offset)
        low = base_price * (1 - low_offset)
        
        # Ensure open and close are within high/low range
        open_price = np.random.uniform(low, high)
        close_price = np.random.uniform(low, high)
        volume = np.random.uniform(1000, 10000)
        
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


class TestFeatureValidator:
    """Test feature validation functionality."""
    
    def test_validate_valid_ohlcv_data(self):
        """Test validation of valid OHLCV data."""
        data = pd.DataFrame({
            'open': [1.0, 1.1, 1.2],
            'high': [1.05, 1.15, 1.25],
            'low': [0.95, 1.05, 1.15],
            'close': [1.02, 1.12, 1.22],
            'volume': [1000, 1100, 1200],
        })
        
        is_valid, errors = FeatureValidator.validate_ohlcv_data(data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_invalid_ohlcv_data(self):
        """Test validation of invalid OHLCV data."""
        # Missing columns
        data = pd.DataFrame({
            'open': [1.0, 1.1],
            'high': [1.05, 1.15],
            # Missing low, close, volume
        })
        
        is_valid, errors = FeatureValidator.validate_ohlcv_data(data)
        assert not is_valid
        assert "Missing required columns" in errors[0]
    
    def test_validate_invalid_ohlc_relationships(self):
        """Test validation of invalid OHLC relationships."""
        data = pd.DataFrame({
            'open': [1.0, 1.1],
            'high': [0.95, 1.05],  # High < Open (invalid)
            'low': [0.90, 1.00],
            'close': [1.02, 1.08],
            'volume': [1000, 1100],
        })
        
        is_valid, errors = FeatureValidator.validate_ohlcv_data(data)
        assert not is_valid
        assert "Invalid OHLC relationships" in errors[0]
    
    def test_validate_features_with_nan(self):
        """Test feature validation with NaN values."""
        features = np.array([[1.0, 2.0], [np.nan, 3.0]])
        
        is_valid, errors = FeatureValidator.validate_features(features)
        assert not is_valid
        assert "NaN values" in errors[0]
    
    def test_validate_features_with_inf(self):
        """Test feature validation with infinite values."""
        features = np.array([[1.0, 2.0], [np.inf, 3.0]])
        
        is_valid, errors = FeatureValidator.validate_features(features)
        assert not is_valid
        assert "infinite values" in errors[0]


class TestTechnicalIndicators:
    """Test technical indicator calculations."""
    
    def test_sma(self):
        """Test Simple Moving Average calculation."""
        data = pd.Series([1, 2, 3, 4, 5])
        sma = TechnicalIndicators.sma(data, window=3)
        
        # First value should be 1 (min_periods=1)
        assert sma.iloc[0] == 1.0
        # Third value should be (1+2+3)/3 = 2
        assert sma.iloc[2] == 2.0
        # Last value should be (3+4+5)/3 = 4
        assert sma.iloc[4] == 4.0
    
    def test_rsi(self):
        """Test RSI calculation."""
        # Create data with clear trend
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2)  # Uptrend
        rsi = TechnicalIndicators.rsi(data, window=14)
        
        # RSI should be between 0 and 100 (excluding NaN values)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
        
        # For strong uptrend, RSI should be high
        assert valid_rsi.iloc[-1] > 50
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data, window=5)
        
        # Upper should be greater than middle, middle greater than lower (excluding NaN)
        valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
        valid_upper = upper[valid_indices]
        valid_middle = middle[valid_indices]
        valid_lower = lower[valid_indices]
        
        assert (valid_upper >= valid_middle).all()
        assert (valid_middle >= valid_lower).all()


class TestVolatilityFeatures:
    """Test volatility feature calculations."""
    
    def test_realized_volatility(self):
        """Test realized volatility calculation."""
        # Create returns with known volatility
        returns = pd.Series(np.random.normal(0, 0.02, 100))  # 2% daily vol
        vol = VolatilityFeatures.realized_volatility(returns, window=20)
        
        # Volatility should be positive (excluding NaN values)
        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all()
        
        # Should be roughly around expected level (allowing for randomness)
        assert valid_vol.iloc[-1] > 0.1  # Annualized vol should be > 10%
    
    def test_garman_klass_volatility(self):
        """Test Garman-Klass volatility estimator."""
        ohlc = pd.DataFrame({
            'open': [1.0, 1.1, 1.2, 1.1, 1.0],
            'high': [1.05, 1.15, 1.25, 1.15, 1.05],
            'low': [0.95, 1.05, 1.15, 1.05, 0.95],
            'close': [1.02, 1.12, 1.18, 1.08, 1.02],
        })
        
        gk_vol = VolatilityFeatures.garman_klass_volatility(ohlc, window=3)
        
        # Volatility should be positive (excluding NaN values)
        valid_vol = gk_vol.dropna()
        assert (valid_vol >= 0).all()


class TestLiquidityFeatures:
    """Test liquidity feature calculations."""
    
    def test_volume_profile(self):
        """Test volume profile calculation."""
        ohlcv = pd.DataFrame({
            'close': [1.0, 1.1, 1.2, 1.1, 1.0],
            'volume': [1000, 1100, 1200, 1100, 1000],
        })
        
        vp = LiquidityFeatures.volume_profile(ohlcv, window=3)
        
        # Volume profile should be non-negative (excluding NaN values)
        valid_vp = vp.dropna()
        assert (valid_vp >= 0).all()
    
    def test_bid_ask_spread_proxy(self):
        """Test bid-ask spread proxy calculation."""
        ohlc = pd.DataFrame({
            'high': [1.05, 1.15, 1.25],
            'low': [0.95, 1.05, 1.15],
            'close': [1.0, 1.1, 1.2],
        })
        
        spread = LiquidityFeatures.bid_ask_spread_proxy(ohlc, window=2)
        
        # Spread should be positive (excluding NaN values)
        valid_spread = spread.dropna()
        assert (valid_spread >= 0).all()


class TestFeatureExtractor:
    """Test main feature extraction functionality."""
    
    def test_extract_canonical_features_basic(self):
        """Test basic feature extraction."""
        market_data = create_sample_market_data(50)
        config = FeatureConfig(window_sizes=[WindowSize.SMALL])
        
        extractor = FeatureExtractor(config)
        features = extractor.extract_canonical_features(market_data)
        
        assert "window_32" in features
        assert features["window_32"].shape[0] == 50  # Same number of rows as input
        assert features["window_32"].shape[1] > 0   # Should have features
    
    def test_extract_canonical_features_multiple_windows(self):
        """Test feature extraction with multiple window sizes."""
        market_data = create_sample_market_data(100)
        config = FeatureConfig(window_sizes=[WindowSize.SMALL, WindowSize.MEDIUM])
        
        extractor = FeatureExtractor(config)
        features = extractor.extract_canonical_features(market_data)
        
        assert "window_32" in features
        assert "window_64" in features
        assert features["window_32"].shape[0] == 100
        assert features["window_64"].shape[0] == 100
    
    def test_extract_canonical_features_with_normalization(self):
        """Test feature extraction with normalization."""
        market_data = create_sample_market_data(50)
        config = FeatureConfig(
            window_sizes=[WindowSize.SMALL],
            normalize=True
        )
        
        extractor = FeatureExtractor(config)
        features = extractor.extract_canonical_features(market_data)
        
        # Features should be roughly normalized (most values between -3 and 3)
        feature_array = features["window_32"]
        assert np.abs(feature_array).max() <= 5  # Clipped at 5
    
    def test_extract_canonical_features_empty_data(self):
        """Test feature extraction with empty data."""
        config = FeatureConfig(window_sizes=[WindowSize.SMALL])
        extractor = FeatureExtractor(config)
        
        with pytest.raises(ValueError, match="Empty market data"):
            extractor.extract_canonical_features([])
    
    def test_feature_names(self):
        """Test feature name generation."""
        config = FeatureConfig(
            include_technical=True,
            include_volatility=True,
            include_liquidity=True
        )
        extractor = FeatureExtractor(config)
        
        names = extractor.get_feature_names(32)
        
        # Should have all feature types
        assert any("returns" in name for name in names)
        assert any("sma" in name for name in names)
        assert any("vol" in name for name in names)
        assert any("volume" in name for name in names)
    
    def test_feature_extraction_with_selective_features(self):
        """Test feature extraction with selective feature types."""
        market_data = create_sample_market_data(50)
        
        # Only technical features
        config = FeatureConfig(
            window_sizes=[WindowSize.SMALL],
            include_technical=True,
            include_volatility=False,
            include_liquidity=False
        )
        
        extractor = FeatureExtractor(config)
        features = extractor.extract_canonical_features(market_data)
        
        # Should have fewer features than full extraction
        feature_count = features["window_32"].shape[1]
        assert feature_count > 0
        
        # Feature names should only include technical
        names = extractor.get_feature_names(32)
        assert any("sma" in name for name in names)
        assert not any("vol" in name for name in names)  # No volatility features


class TestCreateMarketWindowFeatures:
    """Test convenience function for feature creation."""
    
    def test_create_market_window_features(self):
        """Test the convenience function."""
        market_data = create_sample_market_data(50)
        config = FeatureConfig(window_sizes=[WindowSize.SMALL])
        
        result = create_market_window_features(market_data, config)
        
        assert "features" in result
        assert "feature_names" in result
        assert "config" in result
        assert "data_points" in result
        assert "extraction_timestamp" in result
        
        assert result["data_points"] == 50
        assert "window_32" in result["features"]
        assert "window_32" in result["feature_names"]


# Property-based tests would go here if using hypothesis
# For now, we'll stick to unit tests as requested