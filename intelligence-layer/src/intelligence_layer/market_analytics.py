"""Market analytics engine for correlation, volatility, and regime analysis."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationMatrix:
    """Asset correlation matrix with statistical significance."""
    timestamp: datetime
    assets: List[str]
    correlation_matrix: np.ndarray  # NxN matrix
    rolling_window: str             # e.g., "24H"
    method: str                     # pearson, spearman, kendall
    significance: np.ndarray        # p-values
    clusters: Optional[List[List[str]]] = None


@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics for an asset."""
    timestamp: datetime
    asset_id: str
    realized_vol: float             # Historical volatility
    implied_vol: Optional[float]    # From options (if available)
    parkinson_vol: float            # High-low estimator
    garman_klass_vol: float         # OHLC estimator
    vol_of_vol: float               # Volatility clustering
    vol_regime: str                 # LOW, MEDIUM, HIGH


@dataclass
class MarketRegimeIndicators:
    """Market regime indicators and classification."""
    timestamp: datetime
    asset_id: str
    trend_strength: float           # ADX-like
    trend_direction: str            # UP, DOWN, SIDEWAYS
    volatility_regime: str          # LOW, MEDIUM, HIGH
    liquidity_regime: str           # NORMAL, STRESSED, CRISIS
    correlation_regime: str         # NORMAL, BREAKDOWN, CRISIS
    regime_probability: float       # Confidence


@dataclass
class MarketAnomalies:
    """Detected market anomalies."""
    timestamp: datetime
    asset_id: str
    anomaly_type: str               # PRICE_SPIKE, VOLUME_SURGE, etc.
    severity: float                 # 0-1
    z_score: float
    description: str
    recommended_action: str         # ALERT, PAUSE, INVESTIGATE


class MarketAnalyticsEngine:
    """Engine for market analytics and statistical analysis."""
    
    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.timestamp_history: Dict[str, List[datetime]] = {}
    
    def update_market_data(
        self,
        asset_id: str,
        price: float,
        volume: float,
        timestamp: datetime,
    ):
        """Update market data for an asset."""
        if asset_id not in self.price_history:
            self.price_history[asset_id] = []
            self.volume_history[asset_id] = []
            self.timestamp_history[asset_id] = []
        
        self.price_history[asset_id].append(price)
        self.volume_history[asset_id].append(volume)
        self.timestamp_history[asset_id].append(timestamp)
        
        # Keep only recent data (e.g., last 1000 points)
        max_history = 1000
        if len(self.price_history[asset_id]) > max_history:
            self.price_history[asset_id] = self.price_history[asset_id][-max_history:]
            self.volume_history[asset_id] = self.volume_history[asset_id][-max_history:]
            self.timestamp_history[asset_id] = self.timestamp_history[asset_id][-max_history:]
    
    def calculate_correlation_matrix(
        self,
        assets: List[str],
        window: int = 100,
        method: str = "pearson",
    ) -> CorrelationMatrix:
        """Calculate correlation matrix for assets."""
        # Get price data for all assets
        price_data = []
        valid_assets = []
        
        for asset in assets:
            if asset in self.price_history and len(self.price_history[asset]) >= window:
                prices = self.price_history[asset][-window:]
                price_data.append(prices)
                valid_assets.append(asset)
        
        if len(valid_assets) < 2:
            # Return empty correlation matrix
            return CorrelationMatrix(
                timestamp=datetime.now(timezone.utc),
                assets=valid_assets,
                correlation_matrix=np.array([[]]),
                rolling_window=f"{window}",
                method=method,
                significance=np.array([[]]),
                clusters=None,
            )
        
        # Convert to numpy array
        price_array = np.array(price_data)
        
        # Calculate returns
        returns = np.diff(np.log(price_array), axis=1)
        
        # Calculate correlation matrix
        if method == "pearson":
            corr_matrix = np.corrcoef(returns)
        elif method == "spearman":
            corr_matrix = np.zeros((len(valid_assets), len(valid_assets)))
            for i in range(len(valid_assets)):
                for j in range(len(valid_assets)):
                    corr, _ = stats.spearmanr(returns[i], returns[j])
                    corr_matrix[i, j] = corr
        else:
            corr_matrix = np.corrcoef(returns)
        
        # Calculate p-values
        n = returns.shape[1]
        significance = np.zeros_like(corr_matrix)
        for i in range(len(valid_assets)):
            for j in range(len(valid_assets)):
                if i != j:
                    t_stat = corr_matrix[i, j] * np.sqrt(n - 2) / np.sqrt(1 - corr_matrix[i, j]**2)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    significance[i, j] = p_value
        
        # Perform hierarchical clustering
        clusters = self._cluster_assets(corr_matrix, valid_assets)
        
        return CorrelationMatrix(
            timestamp=datetime.now(timezone.utc),
            assets=valid_assets,
            correlation_matrix=corr_matrix,
            rolling_window=f"{window}",
            method=method,
            significance=significance,
            clusters=clusters,
        )
    
    def _cluster_assets(
        self,
        corr_matrix: np.ndarray,
        assets: List[str],
        n_clusters: int = 3,
    ) -> List[List[str]]:
        """Cluster assets based on correlation."""
        if len(assets) < 2:
            return [[asset] for asset in assets]
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Perform hierarchical clustering
        try:
            # Convert to condensed distance matrix
            from scipy.spatial.distance import squareform
            condensed_dist = squareform(distance_matrix, checks=False)
            
            linkage_matrix = linkage(condensed_dist, method='average')
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Group assets by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label - 1].append(assets[i])
            
            # Remove empty clusters
            clusters = [c for c in clusters if c]
            
            return clusters
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return [[asset] for asset in assets]
    
    def calculate_volatility_metrics(
        self,
        asset_id: str,
        window: int = 100,
    ) -> Optional[VolatilityMetrics]:
        """Calculate comprehensive volatility metrics."""
        if asset_id not in self.price_history or len(self.price_history[asset_id]) < window:
            return None
        
        prices = np.array(self.price_history[asset_id][-window:])
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Realized volatility (annualized)
        realized_vol = np.std(returns) * np.sqrt(252)
        
        # Parkinson volatility (high-low estimator)
        # Simplified: using price range as proxy
        price_range = np.max(prices) - np.min(prices)
        parkinson_vol = price_range / np.mean(prices) * np.sqrt(252)
        
        # Garman-Klass volatility (OHLC estimator)
        # Simplified: using realized vol as proxy
        garman_klass_vol = realized_vol * 1.1
        
        # Volatility of volatility
        rolling_vol = np.array([
            np.std(returns[max(0, i-20):i+1]) if i >= 20 else np.std(returns[:i+1])
            for i in range(len(returns))
        ])
        vol_of_vol = np.std(rolling_vol) if len(rolling_vol) > 1 else 0.0
        
        # Determine volatility regime
        if realized_vol < 0.1:
            vol_regime = "LOW"
        elif realized_vol < 0.2:
            vol_regime = "MEDIUM"
        else:
            vol_regime = "HIGH"
        
        return VolatilityMetrics(
            timestamp=datetime.now(timezone.utc),
            asset_id=asset_id,
            realized_vol=realized_vol,
            implied_vol=None,  # Would need options data
            parkinson_vol=parkinson_vol,
            garman_klass_vol=garman_klass_vol,
            vol_of_vol=vol_of_vol,
            vol_regime=vol_regime,
        )
    
    def detect_anomalies(
        self,
        asset_id: str,
        window: int = 100,
        threshold: float = 3.0,
    ) -> List[MarketAnomalies]:
        """Detect market anomalies using statistical methods."""
        if asset_id not in self.price_history or len(self.price_history[asset_id]) < window:
            return []
        
        prices = np.array(self.price_history[asset_id][-window:])
        volumes = np.array(self.volume_history[asset_id][-window:])
        timestamps = self.timestamp_history[asset_id][-window:]
        
        anomalies = []
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Detect price spikes
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        for i, ret in enumerate(returns):
            z_score = (ret - mean_return) / std_return if std_return > 0 else 0
            
            if abs(z_score) > threshold:
                anomaly = MarketAnomalies(
                    timestamp=timestamps[i + 1],
                    asset_id=asset_id,
                    anomaly_type="PRICE_SPIKE" if z_score > 0 else "PRICE_DROP",
                    severity=min(abs(z_score) / 10.0, 1.0),
                    z_score=z_score,
                    description=f"Abnormal {'increase' if z_score > 0 else 'decrease'} detected",
                    recommended_action="ALERT" if abs(z_score) < 5 else "INVESTIGATE",
                )
                anomalies.append(anomaly)
        
        # Detect volume surges
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        for i, vol in enumerate(volumes):
            z_score = (vol - mean_volume) / std_volume if std_volume > 0 else 0
            
            if z_score > threshold:
                anomaly = MarketAnomalies(
                    timestamp=timestamps[i],
                    asset_id=asset_id,
                    anomaly_type="VOLUME_SURGE",
                    severity=min(z_score / 10.0, 1.0),
                    z_score=z_score,
                    description="Abnormal volume surge detected",
                    recommended_action="ALERT",
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def calculate_regime_indicators(
        self,
        asset_id: str,
        window: int = 100,
    ) -> Optional[MarketRegimeIndicators]:
        """Calculate market regime indicators."""
        if asset_id not in self.price_history or len(self.price_history[asset_id]) < window:
            return None
        
        prices = np.array(self.price_history[asset_id][-window:])
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Trend strength (simplified ADX)
        positive_moves = np.sum(returns > 0) / len(returns)
        trend_strength = abs(2 * positive_moves - 1)  # 0 = no trend, 1 = strong trend
        
        # Trend direction
        cumulative_return = np.sum(returns)
        if cumulative_return > 0.02:
            trend_direction = "UP"
        elif cumulative_return < -0.02:
            trend_direction = "DOWN"
        else:
            trend_direction = "SIDEWAYS"
        
        # Volatility regime
        realized_vol = np.std(returns) * np.sqrt(252)
        if realized_vol < 0.1:
            volatility_regime = "LOW"
        elif realized_vol < 0.2:
            volatility_regime = "MEDIUM"
        else:
            volatility_regime = "HIGH"
        
        # Liquidity regime (simplified)
        liquidity_regime = "NORMAL"  # Would need order book data
        
        # Correlation regime (simplified)
        correlation_regime = "NORMAL"  # Would need multi-asset data
        
        # Regime probability (confidence)
        regime_probability = 0.7 + trend_strength * 0.3
        
        return MarketRegimeIndicators(
            timestamp=datetime.now(timezone.utc),
            asset_id=asset_id,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            volatility_regime=volatility_regime,
            liquidity_regime=liquidity_regime,
            correlation_regime=correlation_regime,
            regime_probability=regime_probability,
        )


# Global instance
market_analytics = MarketAnalyticsEngine()
