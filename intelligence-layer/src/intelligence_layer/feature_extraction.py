"""Feature extraction pipeline for market data."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import logging

from .models import MarketData

logger = logging.getLogger(__name__)


class WindowSize(Enum):
    """Supported rolling window sizes."""
    SMALL = 32
    MEDIUM = 64
    LARGE = 128


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    window_sizes: List[WindowSize] = None
    include_technical: bool = True
    include_volatility: bool = True
    include_liquidity: bool = True
    include_correlation: bool = True
    normalize: bool = True
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [WindowSize.SMALL, WindowSize.MEDIUM, WindowSize.LARGE]


class FeatureValidator:
    """Validates feature quality and data integrity."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data quality.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if errors:
            return False, errors
        
        # Check for negative values
        if (data[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            errors.append("Found negative values in OHLCV data")
        
        # Check for zero/invalid price data
        price_cols = ['open', 'high', 'low', 'close']
        # Check if any row has all zero prices (invalid individual bars)
        zero_price_rows = (data[price_cols] == 0).all(axis=1)
        if zero_price_rows.any():
            num_zero_rows = zero_price_rows.sum()
            errors.append(f"Found {num_zero_rows} rows with all-zero prices - invalid market data")
        
        # Check for any individual zero prices (also invalid)
        zero_prices = (data[price_cols] == 0).any().any()
        if zero_prices:
            for col in price_cols:
                zero_count = (data[col] == 0).sum()
                if zero_count > 0:
                    errors.append(f"Found {zero_count} zero values in {col} - invalid price data")
        
        # Check for zero volume across all records
        if (data['volume'] == 0).all():
            errors.append("All volume data is zero - invalid market data")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        if invalid_ohlc.any():
            errors.append("Invalid OHLC relationships found")
        
        # Check for excessive gaps
        returns = data['close'].pct_change().dropna()
        if len(returns) > 0:  # Only check if we have returns data
            extreme_returns = np.abs(returns) > 0.5  # 50% moves
            if extreme_returns.any():
                errors.append(f"Found {extreme_returns.sum()} extreme price movements (>50%)")
        
        # Check for missing data
        missing_data = data.isnull().any().any()
        if missing_data:
            errors.append("Found missing values in data")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_features(features: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate extracted features.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for NaN or infinite values
        if np.isnan(features).any():
            errors.append("Features contain NaN values")
        
        if np.isinf(features).any():
            errors.append("Features contain infinite values")
        
        # Check feature range (should be normalized)
        if np.abs(features).max() > 10:
            errors.append("Features appear unnormalized (values > 10)")
        
        return len(errors) == 0, errors


class TechnicalIndicators:
    """Technical indicator calculations."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line


class VolatilityFeatures:
    """Volatility-based feature calculations."""
    
    @staticmethod
    def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
        """Realized volatility."""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def garman_klass_volatility(ohlc: pd.DataFrame, window: int) -> pd.Series:
        """Garman-Klass volatility estimator."""
        log_hl = np.log(ohlc['high'] / ohlc['low'])
        log_co = np.log(ohlc['close'] / ohlc['open'])
        
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return gk.rolling(window=window).mean().apply(np.sqrt) * np.sqrt(252)
    
    @staticmethod
    def parkinson_volatility(ohlc: pd.DataFrame, window: int) -> pd.Series:
        """Parkinson volatility estimator."""
        log_hl = np.log(ohlc['high'] / ohlc['low'])
        park = log_hl**2 / (4 * np.log(2))
        return park.rolling(window=window).mean().apply(np.sqrt) * np.sqrt(252)


class LiquidityFeatures:
    """Liquidity-based feature calculations."""
    
    @staticmethod
    def volume_profile(ohlcv: pd.DataFrame, window: int) -> pd.Series:
        """Volume-weighted average price deviation."""
        vwap = (ohlcv['close'] * ohlcv['volume']).rolling(window=window).sum() / ohlcv['volume'].rolling(window=window).sum()
        return np.abs(ohlcv['close'] - vwap) / ohlcv['close']
    
    @staticmethod
    def bid_ask_spread_proxy(ohlc: pd.DataFrame, window: int) -> pd.Series:
        """Proxy for bid-ask spread using high-low range."""
        spread_proxy = (ohlc['high'] - ohlc['low']) / ohlc['close']
        return spread_proxy.rolling(window=window).mean()
    
    @staticmethod
    def volume_rate_of_change(volume: pd.Series, window: int) -> pd.Series:
        """Volume rate of change."""
        return volume.pct_change(periods=window)


class FeatureExtractor:
    """Main feature extraction pipeline."""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.validator = FeatureValidator()
        self.technical = TechnicalIndicators()
        self.volatility = VolatilityFeatures()
        self.liquidity = LiquidityFeatures()
    
    def extract_canonical_features(self, market_data: List[MarketData]) -> Dict[str, np.ndarray]:
        """
        Extract canonical features from OHLCV data.
        
        Args:
            market_data: List of MarketData objects
            
        Returns:
            Dictionary mapping window sizes to feature arrays
        """
        if not market_data:
            raise ValueError("Empty market data provided")
        
        # Convert to DataFrame
        df = self._market_data_to_dataframe(market_data)
        
        # Validate input data
        is_valid, errors = self.validator.validate_ohlcv_data(df)
        if not is_valid:
            raise ValueError(f"Invalid OHLCV data: {'; '.join(errors)}")
        
        # Extract features for each window size
        features_by_window = {}
        
        for window_size in self.config.window_sizes:
            window = window_size.value
            features = self._extract_features_for_window(df, window)
            
            # Validate extracted features
            is_valid, errors = self.validator.validate_features(features)
            if not is_valid:
                logger.warning(f"Feature validation failed for window {window}: {'; '.join(errors)}")
            
            features_by_window[f"window_{window}"] = features
        
        return features_by_window
    
    def _market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert MarketData objects to DataFrame."""
        data = []
        for md in market_data:
            data.append({
                'timestamp': md.timestamp,
                'asset_id': md.asset_id,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume,
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def _extract_features_for_window(self, df: pd.DataFrame, window: int) -> np.ndarray:
        """Extract features for a specific window size."""
        features = []
        
        # Basic price features
        features.extend(self._extract_price_features(df, window))
        
        # Technical indicators
        if self.config.include_technical:
            features.extend(self._extract_technical_features(df, window))
        
        # Volatility features
        if self.config.include_volatility:
            features.extend(self._extract_volatility_features(df, window))
        
        # Liquidity features
        if self.config.include_liquidity:
            features.extend(self._extract_liquidity_features(df, window))
        
        # Stack features and handle missing values
        feature_array = np.column_stack(features)
        
        # Forward fill missing values
        feature_df = pd.DataFrame(feature_array)
        feature_df = feature_df.ffill().fillna(0)
        feature_array = feature_df.values
        
        # Normalize if requested
        if self.config.normalize:
            feature_array = self._normalize_features(feature_array)
        
        return feature_array
    
    def _extract_price_features(self, df: pd.DataFrame, window: int) -> List[np.ndarray]:
        """Extract basic price-based features."""
        features = []
        
        # Returns
        returns = df['close'].pct_change()
        features.append(returns.values)
        
        # Log returns
        log_returns = np.log(df['close'] / df['close'].shift(1))
        features.append(log_returns.values)
        
        # Price momentum
        momentum = df['close'] / df['close'].shift(window) - 1
        features.append(momentum.values)
        
        # High-low range
        hl_range = (df['high'] - df['low']) / df['close']
        features.append(hl_range.values)
        
        # Open-close gap
        oc_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features.append(oc_gap.values)
        
        return features
    
    def _extract_technical_features(self, df: pd.DataFrame, window: int) -> List[np.ndarray]:
        """Extract technical indicator features."""
        features = []
        
        # Moving averages
        sma = self.technical.sma(df['close'], window)
        ema = self.technical.ema(df['close'], window)
        
        # Price relative to moving averages
        features.append((df['close'] / sma - 1).values)
        features.append((df['close'] / ema - 1).values)
        
        # RSI
        rsi = self.technical.rsi(df['close'], min(window, 14))
        features.append((rsi / 100).values)  # Normalize to [0,1]
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.technical.bollinger_bands(df['close'], min(window, 20))
        bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features.append(bb_position.values)
        
        # MACD
        macd_line, signal_line = self.technical.macd(df['close'])
        macd_histogram = macd_line - signal_line
        features.append((macd_histogram / df['close']).values)  # Normalize by price
        
        return features
    
    def _extract_volatility_features(self, df: pd.DataFrame, window: int) -> List[np.ndarray]:
        """Extract volatility-based features."""
        features = []
        
        returns = df['close'].pct_change()
        
        # Realized volatility
        realized_vol = self.volatility.realized_volatility(returns, window)
        features.append(realized_vol.values)
        
        # Garman-Klass volatility
        gk_vol = self.volatility.garman_klass_volatility(df, window)
        features.append(gk_vol.values)
        
        # Parkinson volatility
        park_vol = self.volatility.parkinson_volatility(df, window)
        features.append(park_vol.values)
        
        # Volatility of volatility
        vol_of_vol = realized_vol.rolling(window=window).std()
        features.append(vol_of_vol.values)
        
        return features
    
    def _extract_liquidity_features(self, df: pd.DataFrame, window: int) -> List[np.ndarray]:
        """Extract liquidity-based features."""
        features = []
        
        # Volume features
        volume_ma = df['volume'].rolling(window=window).mean()
        volume_ratio = df['volume'] / volume_ma
        features.append(volume_ratio.values)
        
        # Volume profile
        volume_profile = self.liquidity.volume_profile(df, window)
        features.append(volume_profile.values)
        
        # Bid-ask spread proxy
        spread_proxy = self.liquidity.bid_ask_spread_proxy(df, window)
        features.append(spread_proxy.values)
        
        # Volume rate of change
        volume_roc = self.liquidity.volume_rate_of_change(df['volume'], window)
        features.append(volume_roc.values)
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using rolling z-score."""
        # Use robust normalization (median and MAD)
        median = np.nanmedian(features, axis=0)
        mad = np.nanmedian(np.abs(features - median), axis=0)
        
        # Avoid division by zero
        mad = np.where(mad == 0, 1, mad)
        
        normalized = (features - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
        
        # Clip extreme values
        normalized = np.clip(normalized, -5, 5)
        
        return normalized
    
    def get_feature_names(self, window: int) -> List[str]:
        """Get feature names for a given window size."""
        names = []
        
        # Price features
        names.extend(['returns', 'log_returns', f'momentum_{window}', 'hl_range', 'oc_gap'])
        
        # Technical features
        if self.config.include_technical:
            names.extend([
                f'price_sma_{window}', f'price_ema_{window}', 'rsi_norm',
                'bb_position', 'macd_histogram_norm'
            ])
        
        # Volatility features
        if self.config.include_volatility:
            names.extend([
                f'realized_vol_{window}', f'gk_vol_{window}',
                f'park_vol_{window}', f'vol_of_vol_{window}'
            ])
        
        # Liquidity features
        if self.config.include_liquidity:
            names.extend([
                f'volume_ratio_{window}', f'volume_profile_{window}',
                f'spread_proxy_{window}', f'volume_roc_{window}'
            ])
        
        return names


def create_market_window_features(
    market_data: List[MarketData],
    config: FeatureConfig = None
) -> Dict[str, Any]:
    """
    Convenience function to create market window features.
    
    Args:
        market_data: List of MarketData objects
        config: Feature extraction configuration
        
    Returns:
        Dictionary with extracted features and metadata
    """
    extractor = FeatureExtractor(config)
    
    try:
        features_by_window = extractor.extract_canonical_features(market_data)
        
        # Get feature names for documentation
        feature_names = {}
        for window_size in config.window_sizes if config else [WindowSize.MEDIUM]:
            window = window_size.value
            feature_names[f"window_{window}"] = extractor.get_feature_names(window)
        
        return {
            'features': features_by_window,
            'feature_names': feature_names,
            'config': config,
            'data_points': len(market_data),
            'extraction_timestamp': datetime.now(timezone.utc),
        }
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise