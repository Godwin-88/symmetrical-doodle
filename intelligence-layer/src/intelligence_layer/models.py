"""Data models for intelligence layer."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field


class SimilarityMatch(BaseModel):
    """Represents a similarity match for embeddings."""
    embedding_id: UUID
    similarity_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphFeatureSnapshot(BaseModel):
    """Graph structural features from Neo4j GDS."""
    timestamp: datetime
    asset_cluster_id: Optional[str] = None
    cluster_density: Optional[float] = None
    centrality_score: Optional[float] = None
    systemic_risk_proxy: Optional[float] = None
    features: Dict[str, float] = Field(default_factory=dict)


class IntelligenceState(BaseModel):
    """Composite intelligence state for RL and strategy orchestration."""
    
    # Embedding context
    embedding_similarity_context: List[SimilarityMatch] = Field(default_factory=list)
    
    # Regime information
    current_regime_label: Optional[str] = None
    regime_transition_probabilities: Dict[str, float] = Field(default_factory=dict)
    regime_confidence: Optional[float] = None
    
    # Graph features
    graph_structural_features: Optional[GraphFeatureSnapshot] = None
    
    # Confidence and uncertainty
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    timestamp: datetime
    version: str = Field(default="1.0")


class MarketWindowFeatures(BaseModel):
    """Input features for embedding inference."""
    asset_id: str
    window_data: List[Dict[str, float]]  # OHLCV + engineered features
    timestamp: datetime
    horizon: str = Field(default="1h")  # 1h, 4h, 1d


class EmbeddingResponse(BaseModel):
    """Response from embedding inference endpoint."""
    embedding_id: UUID
    embedding_vector: Optional[List[float]] = None  # Optional for privacy
    similarity_context: List[SimilarityMatch]
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class RegimeResponse(BaseModel):
    """Response from regime inference endpoint."""
    regime_probabilities: Dict[str, float]
    transition_likelihoods: Dict[str, float]
    regime_entropy: float
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class GraphFeaturesResponse(BaseModel):
    """Response from graph features endpoint."""
    cluster_membership: Optional[str] = None
    centrality_metrics: Dict[str, float] = Field(default_factory=dict)
    systemic_risk_proxies: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime


class RLStateResponse(BaseModel):
    """Response from RL state assembly endpoint."""
    composite_state: IntelligenceState
    state_components: Dict[str, Any] = Field(default_factory=dict)
    assembly_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class MarketData(BaseModel):
    """Market data point."""
    timestamp: datetime
    asset_id: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Dict[str, Any] = Field(default_factory=dict)