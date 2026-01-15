"""Regime detection and inference pipeline."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .models import MarketData, RegimeResponse
from .config import Config

logger = logging.getLogger(__name__)


class VolatilityLevel(Enum):
    """Market volatility levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendState(Enum):
    """Market trend states."""
    TRENDING = "trending"
    RANGING = "ranging"


class LiquidityState(Enum):
    """Market liquidity states."""
    NORMAL = "normal"
    STRESSED = "stressed"


@dataclass
class RegimeDefinition:
    """Market regime definition."""
    regime_id: str
    volatility_level: VolatilityLevel
    trend_state: TrendState
    liquidity_state: LiquidityState
    description: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {
            "regime_id": self.regime_id,
            "volatility_level": self.volatility_level.value,
            "trend_state": self.trend_state.value,
            "liquidity_state": self.liquidity_state.value,
            "description": self.description
        }


@dataclass
class RegimeTransition:
    """Regime transition information."""
    from_regime: str
    to_regime: str
    probability: float
    avg_duration: float
    confidence: float


class RegimeStabilizer:
    """Stabilization logic to prevent regime flickering."""
    
    def __init__(self, min_duration: int = 5, hysteresis_threshold: float = 0.1):
        """
        Initialize regime stabilizer.
        
        Args:
            min_duration: Minimum duration in periods before regime change
            hysteresis_threshold: Threshold for regime change confidence
        """
        self.min_duration = min_duration
        self.hysteresis_threshold = hysteresis_threshold
        self.current_regime: Optional[str] = None
        self.regime_start_time: Optional[datetime] = None
        self.regime_duration = 0
        
    def stabilize_regime(
        self, 
        predicted_regime: str, 
        confidence: float, 
        timestamp: datetime
    ) -> Tuple[str, bool]:
        """
        Apply stabilization logic to regime prediction.
        
        Args:
            predicted_regime: Raw regime prediction
            confidence: Confidence in prediction
            timestamp: Current timestamp
            
        Returns:
            Tuple of (stabilized_regime, regime_changed)
        """
        regime_changed = False
        
        # Initialize if first prediction
        if self.current_regime is None:
            self.current_regime = predicted_regime
            self.regime_start_time = timestamp
            self.regime_duration = 0
            regime_changed = True
            logger.info(f"Initial regime set to {predicted_regime}")
            return self.current_regime, regime_changed
        
        # Check if regime should change
        if predicted_regime != self.current_regime:
            # Require minimum confidence for regime change
            if confidence > (0.5 + self.hysteresis_threshold):
                # Require minimum duration before allowing change
                if self.regime_duration >= self.min_duration:
                    old_regime = self.current_regime
                    self.current_regime = predicted_regime
                    self.regime_start_time = timestamp
                    self.regime_duration = 0
                    regime_changed = True
                    logger.info(f"Regime changed from {old_regime} to {predicted_regime}")
                else:
                    logger.debug(f"Regime change blocked: duration {self.regime_duration} < {self.min_duration}")
            else:
                logger.debug(f"Regime change blocked: confidence {confidence} too low")
        else:
            # Same regime, increment duration
            self.regime_duration += 1
            
        return self.current_regime, regime_changed


class HMMRegimeDetector:
    """HMM-based regime detection with stabilization."""
    
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        """
        Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of hidden regimes
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_definitions: Dict[int, RegimeDefinition] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        
    def _extract_features(self, market_data: List[MarketData]) -> np.ndarray:
        """
        Extract features for regime detection.
        
        Args:
            market_data: List of market data points
            
        Returns:
            Feature matrix
        """
        df = pd.DataFrame([
            {
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            }
            for md in market_data
        ])
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # High-low spread as liquidity proxy
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Trend indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['trend'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Select features for regime detection
        features = [
            'returns', 'volatility', 'volume_ratio', 
            'hl_spread', 'trend'
        ]
        
        feature_matrix = df[features].dropna().values
        return feature_matrix
    
    def _define_regimes(self, states: np.ndarray, features: np.ndarray) -> None:
        """
        Define regime characteristics based on HMM states.
        
        Args:
            states: HMM state sequence
            features: Feature matrix used for training
        """
        self.regime_definitions = {}
        
        for state in range(self.n_regimes):
            state_mask = states == state
            if not np.any(state_mask):
                continue
                
            state_features = features[state_mask]
            
            # Analyze volatility (feature index 1)
            vol_mean = np.mean(state_features[:, 1])
            if vol_mean < 0.01:
                vol_level = VolatilityLevel.LOW
            elif vol_mean < 0.03:
                vol_level = VolatilityLevel.MEDIUM
            else:
                vol_level = VolatilityLevel.HIGH
            
            # Analyze trend (feature index 4)
            trend_mean = np.abs(np.mean(state_features[:, 4]))
            trend_state = TrendState.TRENDING if trend_mean > 0.02 else TrendState.RANGING
            
            # Analyze liquidity (feature index 3 - HL spread)
            liquidity_mean = np.mean(state_features[:, 3])
            liquidity_state = LiquidityState.STRESSED if liquidity_mean > 0.02 else LiquidityState.NORMAL
            
            # Create regime definition
            regime_id = f"regime_{state}"
            description = f"{vol_level.value.title()} volatility, {trend_state.value} market, {liquidity_state.value} liquidity"
            
            self.regime_definitions[state] = RegimeDefinition(
                regime_id=regime_id,
                volatility_level=vol_level,
                trend_state=trend_state,
                liquidity_state=liquidity_state,
                description=description
            )
            
            logger.info(f"Defined {regime_id}: {description}")
    
    def fit(self, market_data: List[MarketData]) -> None:
        """
        Fit HMM model to market data.
        
        Args:
            market_data: Historical market data for training
        """
        logger.info(f"Fitting HMM regime detector with {len(market_data)} data points")
        
        # Extract features
        features = self._extract_features(market_data)
        
        if len(features) < self.n_regimes * 10:
            raise ValueError(f"Insufficient data: need at least {self.n_regimes * 10} points")
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit HMM model
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_iter=100
        )
        
        self.model.fit(features_scaled)
        self.transition_matrix = self.model.transmat_
        
        # Get state sequence for regime definition
        states = self.model.predict(features_scaled)
        
        # Define regime characteristics
        self._define_regimes(states, features)
        
        self.is_fitted = True
        logger.info("HMM regime detector fitted successfully")
    
    def predict_regime(
        self, 
        recent_data: List[MarketData], 
        return_probabilities: bool = True
    ) -> Tuple[str, Dict[str, float], float]:
        """
        Predict current market regime.
        
        Args:
            recent_data: Recent market data for prediction
            return_probabilities: Whether to return regime probabilities
            
        Returns:
            Tuple of (regime_id, probabilities, confidence)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features from recent data
        features = self._extract_features(recent_data)
        
        if len(features) == 0:
            raise ValueError("No valid features extracted from recent data")
        
        # Use only the most recent observation
        recent_features = features[-1:].reshape(1, -1)
        features_scaled = self.scaler.transform(recent_features)
        
        # Get regime probabilities
        log_probs = self.model.predict_proba(features_scaled)
        probs = np.exp(log_probs[0])
        
        # Get most likely regime
        predicted_state = np.argmax(probs)
        confidence = float(probs[predicted_state])
        
        # Get regime ID
        if predicted_state not in self.regime_definitions:
            regime_id = f"regime_{predicted_state}"
        else:
            regime_id = self.regime_definitions[predicted_state].regime_id
        
        # Prepare probability dictionary
        regime_probs = {}
        if return_probabilities:
            for state, prob in enumerate(probs):
                state_regime_id = (
                    self.regime_definitions[state].regime_id 
                    if state in self.regime_definitions 
                    else f"regime_{state}"
                )
                regime_probs[state_regime_id] = float(prob)
        
        return regime_id, regime_probs, confidence
    
    def get_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Get regime transition probabilities.
        
        Returns:
            Dictionary of transition probabilities
        """
        if not self.is_fitted or self.transition_matrix is None:
            return {}
        
        transitions = {}
        for from_state in range(self.n_regimes):
            from_regime_id = (
                self.regime_definitions[from_state].regime_id
                if from_state in self.regime_definitions
                else f"regime_{from_state}"
            )
            
            transitions[from_regime_id] = {}
            for to_state in range(self.n_regimes):
                to_regime_id = (
                    self.regime_definitions[to_state].regime_id
                    if to_state in self.regime_definitions
                    else f"regime_{to_state}"
                )
                
                transitions[from_regime_id][to_regime_id] = float(
                    self.transition_matrix[from_state, to_state]
                )
        
        return transitions


class RegimeInferencePipeline:
    """Complete regime inference pipeline with stabilization."""
    
    def __init__(self, config: Config):
        """
        Initialize regime inference pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.detector = HMMRegimeDetector()
        self.stabilizer = RegimeStabilizer()
        self.is_trained = False
        
    def train(self, historical_data: List[MarketData]) -> None:
        """
        Train the regime detection model.
        
        Args:
            historical_data: Historical market data for training
        """
        logger.info("Training regime inference pipeline")
        self.detector.fit(historical_data)
        self.is_trained = True
        logger.info("Regime inference pipeline trained successfully")
    
    def infer_regime(
        self, 
        recent_data: List[MarketData], 
        timestamp: Optional[datetime] = None
    ) -> RegimeResponse:
        """
        Infer current market regime with stabilization.
        
        Args:
            recent_data: Recent market data for inference
            timestamp: Current timestamp (defaults to now)
            
        Returns:
            Regime inference response
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before inference")
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Get raw regime prediction
        raw_regime, regime_probs, raw_confidence = self.detector.predict_regime(recent_data)
        
        # Apply stabilization
        stable_regime, regime_changed = self.stabilizer.stabilize_regime(
            raw_regime, raw_confidence, timestamp
        )
        
        # Get transition probabilities
        transition_probs = self.detector.get_transition_probabilities()
        current_transitions = transition_probs.get(stable_regime, {})
        
        # Calculate regime entropy for uncertainty measure
        prob_values = list(regime_probs.values())
        regime_entropy = -sum(p * np.log(p + 1e-10) for p in prob_values if p > 0)
        
        # Adjust confidence based on stabilization
        final_confidence = raw_confidence
        if not regime_changed and stable_regime != raw_regime:
            # Lower confidence if stabilizer overrode the prediction
            final_confidence *= 0.8
        
        return RegimeResponse(
            regime_probabilities=regime_probs,
            transition_likelihoods=current_transitions,
            regime_entropy=float(regime_entropy),
            confidence=final_confidence,
            timestamp=timestamp
        )
    
    def get_regime_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current regime definitions.
        
        Returns:
            Dictionary of regime definitions
        """
        if not self.is_trained:
            return {}
        
        return {
            regime_def.regime_id: regime_def.to_dict()
            for regime_def in self.detector.regime_definitions.values()
        }