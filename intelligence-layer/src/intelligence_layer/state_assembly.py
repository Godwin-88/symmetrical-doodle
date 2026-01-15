"""Composite intelligence state assembly from multiple data sources."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import numpy as np
from uuid import UUID, uuid4
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase
import httpx

from .models import (
    IntelligenceState, 
    SimilarityMatch, 
    GraphFeatureSnapshot,
    MarketData,
    RLStateResponse
)
from .config import Config
from .regime_detection import RegimeInferencePipeline
from .graph_analytics import MarketGraphAnalytics

logger = logging.getLogger(__name__)


class PgVectorClient:
    """Client for pgvector operations."""
    
    def __init__(self, config: Config):
        """
        Initialize pgvector client.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.connection = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to PostgreSQL with pgvector."""
        try:
            self.connection = psycopg2.connect(
                self.config.database.postgres_url,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to PostgreSQL with pgvector")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def close(self) -> None:
        """Close PostgreSQL connection."""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed")
    
    def get_recent_market_embeddings(
        self, 
        asset_id: str, 
        limit: int = 10,
        horizon: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        Get recent market state embeddings for an asset.
        
        Args:
            asset_id: Asset identifier
            limit: Maximum number of embeddings to retrieve
            horizon: Time horizon filter
            
        Returns:
            List of embedding records
        """
        try:
            with self.connection.cursor() as cursor:
                query = """
                SELECT id, timestamp, asset_id, regime_id, embedding, 
                       volatility, liquidity, horizon, source_model, metadata
                FROM market_state_embeddings
                WHERE asset_id = %s AND horizon = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """
                
                cursor.execute(query, (asset_id, horizon, limit))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Failed to get market embeddings for {asset_id}: {e}")
            return []
    
    def find_similar_embeddings(
        self, 
        embedding_vector: List[float], 
        asset_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[SimilarityMatch]:
        """
        Find similar embeddings using cosine similarity.
        
        Args:
            embedding_vector: Query embedding vector
            asset_id: Optional asset filter
            limit: Maximum number of matches
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similarity matches
        """
        try:
            with self.connection.cursor() as cursor:
                # Convert embedding to string format for pgvector
                embedding_str = f"[{','.join(map(str, embedding_vector))}]"
                
                if asset_id:
                    query = """
                    SELECT id, timestamp, asset_id, 
                           1 - (embedding <=> %s::vector) AS similarity,
                           metadata
                    FROM market_state_embeddings
                    WHERE asset_id = %s
                      AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                    params = (embedding_str, asset_id, embedding_str, 
                             similarity_threshold, embedding_str, limit)
                else:
                    query = """
                    SELECT id, timestamp, asset_id,
                           1 - (embedding <=> %s::vector) AS similarity,
                           metadata
                    FROM market_state_embeddings
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                    params = (embedding_str, embedding_str, 
                             similarity_threshold, embedding_str, limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                matches = []
                for row in results:
                    matches.append(SimilarityMatch(
                        embedding_id=UUID(str(row['id'])),
                        similarity_score=float(row['similarity']),
                        timestamp=row['timestamp'],
                        metadata=row['metadata'] or {}
                    ))
                
                return matches
                
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def get_strategy_state_embeddings(
        self, 
        strategy_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent strategy state embeddings.
        
        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of embeddings
            
        Returns:
            List of strategy embedding records
        """
        try:
            with self.connection.cursor() as cursor:
                query = """
                SELECT id, timestamp, strategy_id, embedding,
                       pnl_state, drawdown, exposure, metadata
                FROM strategy_state_embeddings
                WHERE strategy_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """
                
                cursor.execute(query, (strategy_id, limit))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Failed to get strategy embeddings for {strategy_id}: {e}")
            return []


class RustCoreClient:
    """Client for communicating with Rust execution core."""
    
    def __init__(self, config: Config, base_url: str = "http://localhost:8001"):
        """
        Initialize Rust core client.
        
        Args:
            config: Configuration object
            base_url: Base URL for Rust core API
        """
        self.config = config
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
    
    async def get_portfolio_state(self) -> Dict[str, float]:
        """
        Get current portfolio state from Rust core.
        
        Returns:
            Dictionary with portfolio metrics
        """
        try:
            response = await self.client.get(f"{self.base_url}/portfolio/state")
            response.raise_for_status()
            
            data = response.json()
            return {
                "net_exposure": data.get("net_exposure", 0.0),
                "gross_exposure": data.get("gross_exposure", 0.0),
                "drawdown": data.get("drawdown", 0.0),
                "volatility_target_utilization": data.get("volatility_target_utilization", 0.0),
                "unrealized_pnl": data.get("unrealized_pnl", 0.0),
                "realized_pnl": data.get("realized_pnl", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio state from Rust core: {e}")
            return {
                "net_exposure": 0.0,
                "gross_exposure": 0.0,
                "drawdown": 0.0,
                "volatility_target_utilization": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            }
    
    async def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics from Rust core.
        
        Returns:
            Dictionary with risk metrics
        """
        try:
            response = await self.client.get(f"{self.base_url}/risk/metrics")
            response.raise_for_status()
            
            data = response.json()
            return {
                "var_1d": data.get("var_1d", 0.0),
                "var_5d": data.get("var_5d", 0.0),
                "max_drawdown": data.get("max_drawdown", 0.0),
                "sharpe_ratio": data.get("sharpe_ratio", 0.0),
                "volatility": data.get("volatility", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics from Rust core: {e}")
            return {
                "var_1d": 0.0,
                "var_5d": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0
            }


class StateValidator:
    """Validator for composite intelligence states."""
    
    @staticmethod
    def validate_embedding_context(
        similarity_context: List[SimilarityMatch]
    ) -> Tuple[bool, List[str]]:
        """
        Validate embedding similarity context.
        
        Args:
            similarity_context: List of similarity matches
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not similarity_context:
            errors.append("Empty similarity context")
            return False, errors
        
        for i, match in enumerate(similarity_context):
            if match.similarity_score < 0.0 or match.similarity_score > 1.0:
                errors.append(f"Invalid similarity score at index {i}: {match.similarity_score}")
            
            if match.timestamp > datetime.now(timezone.utc):
                errors.append(f"Future timestamp at index {i}: {match.timestamp}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_regime_probabilities(
        regime_probs: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Validate regime probabilities.
        
        Args:
            regime_probs: Dictionary of regime probabilities
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not regime_probs:
            errors.append("Empty regime probabilities")
            return False, errors
        
        total_prob = sum(regime_probs.values())
        if abs(total_prob - 1.0) > 0.01:  # Allow small floating point errors
            errors.append(f"Regime probabilities don't sum to 1.0: {total_prob}")
        
        for regime, prob in regime_probs.items():
            if prob < 0.0 or prob > 1.0:
                errors.append(f"Invalid probability for regime {regime}: {prob}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_graph_features(
        graph_features: Optional[GraphFeatureSnapshot]
    ) -> Tuple[bool, List[str]]:
        """
        Validate graph features.
        
        Args:
            graph_features: Graph feature snapshot
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if graph_features is None:
            errors.append("Missing graph features")
            return False, errors
        
        if graph_features.timestamp > datetime.now(timezone.utc):
            errors.append(f"Future timestamp in graph features: {graph_features.timestamp}")
        
        # Validate centrality scores are non-negative
        if graph_features.centrality_score is not None and graph_features.centrality_score < 0:
            errors.append(f"Negative centrality score: {graph_features.centrality_score}")
        
        if graph_features.systemic_risk_proxy is not None and graph_features.systemic_risk_proxy < 0:
            errors.append(f"Negative systemic risk proxy: {graph_features.systemic_risk_proxy}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_confidence_scores(
        confidence_scores: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Validate confidence scores.
        
        Args:
            confidence_scores: Dictionary of confidence scores
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for component, score in confidence_scores.items():
            if score < 0.0 or score > 1.0:
                errors.append(f"Invalid confidence score for {component}: {score}")
        
        return len(errors) == 0, errors


class CompositeStateAssembler:
    """Assembles composite intelligence states from multiple data sources."""
    
    def __init__(self, config: Config):
        """
        Initialize composite state assembler.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.pgvector_client = PgVectorClient(config)
        self.graph_analytics = MarketGraphAnalytics(config)
        self.rust_client = RustCoreClient(config)
        self.regime_pipeline: Optional[RegimeInferencePipeline] = None
        self.validator = StateValidator()
    
    def set_regime_pipeline(self, pipeline: RegimeInferencePipeline) -> None:
        """
        Set the regime inference pipeline.
        
        Args:
            pipeline: Trained regime inference pipeline
        """
        self.regime_pipeline = pipeline
    
    async def assemble_intelligence_state(
        self, 
        asset_id: str,
        recent_market_data: List[MarketData],
        timestamp: Optional[datetime] = None
    ) -> IntelligenceState:
        """
        Assemble composite intelligence state for an asset.
        
        Args:
            asset_id: Asset identifier
            recent_market_data: Recent market data for regime inference
            timestamp: Current timestamp (defaults to now)
            
        Returns:
            Composite intelligence state
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        logger.info(f"Assembling intelligence state for {asset_id}")
        
        # Get recent embeddings from pgvector
        recent_embeddings = self.pgvector_client.get_recent_market_embeddings(
            asset_id, limit=5
        )
        
        # Build similarity context
        similarity_context = []
        if recent_embeddings:
            # Use most recent embedding to find similar states
            latest_embedding = recent_embeddings[0]
            if latest_embedding['embedding']:
                similarity_context = self.pgvector_client.find_similar_embeddings(
                    latest_embedding['embedding'], asset_id, limit=5
                )
        
        # Get regime information
        current_regime_label = None
        regime_transition_probabilities = {}
        regime_confidence = 0.0
        
        if self.regime_pipeline and recent_market_data:
            try:
                regime_response = self.regime_pipeline.infer_regime(
                    recent_market_data, timestamp
                )
                
                # Find most likely regime
                if regime_response.regime_probabilities:
                    current_regime_label = max(
                        regime_response.regime_probabilities.items(),
                        key=lambda x: x[1]
                    )[0]
                
                regime_transition_probabilities = regime_response.transition_likelihoods
                regime_confidence = regime_response.confidence
                
            except Exception as e:
                logger.error(f"Failed to infer regime: {e}")
        
        # Get graph features
        graph_features = self.graph_analytics.get_graph_features_for_asset(asset_id)
        
        # Get portfolio and risk metrics from Rust core
        portfolio_state = await self.rust_client.get_portfolio_state()
        risk_metrics = await self.rust_client.get_risk_metrics()
        
        # Calculate confidence scores
        confidence_scores = {
            "embedding_similarity": (
                np.mean([m.similarity_score for m in similarity_context])
                if similarity_context else 0.0
            ),
            "regime_inference": regime_confidence,
            "graph_features": (
                1.0 if graph_features.centrality_score is not None else 0.5
            ),
            "portfolio_data": (
                1.0 if portfolio_state.get("net_exposure", 0) != 0 else 0.5
            )
        }
        
        # Create composite state
        intelligence_state = IntelligenceState(
            embedding_similarity_context=similarity_context,
            current_regime_label=current_regime_label,
            regime_transition_probabilities=regime_transition_probabilities,
            regime_confidence=regime_confidence,
            graph_structural_features=graph_features,
            confidence_scores=confidence_scores,
            timestamp=timestamp
        )
        
        # Validate state
        validation_errors = self._validate_state(intelligence_state)
        if validation_errors:
            logger.warning(f"State validation warnings: {validation_errors}")
        
        logger.info(f"Successfully assembled intelligence state for {asset_id}")
        return intelligence_state
    
    def _validate_state(self, state: IntelligenceState) -> List[str]:
        """
        Validate assembled intelligence state.
        
        Args:
            state: Intelligence state to validate
            
        Returns:
            List of validation error messages
        """
        all_errors = []
        
        # Validate embedding context
        if state.embedding_similarity_context:
            valid, errors = self.validator.validate_embedding_context(
                state.embedding_similarity_context
            )
            if not valid:
                all_errors.extend([f"Embedding context: {e}" for e in errors])
        
        # Validate regime probabilities
        if state.regime_transition_probabilities:
            valid, errors = self.validator.validate_regime_probabilities(
                state.regime_transition_probabilities
            )
            if not valid:
                all_errors.extend([f"Regime probabilities: {e}" for e in errors])
        
        # Validate graph features
        valid, errors = self.validator.validate_graph_features(
            state.graph_structural_features
        )
        if not valid:
            all_errors.extend([f"Graph features: {e}" for e in errors])
        
        # Validate confidence scores
        valid, errors = self.validator.validate_confidence_scores(
            state.confidence_scores
        )
        if not valid:
            all_errors.extend([f"Confidence scores: {e}" for e in errors])
        
        return all_errors
    
    async def assemble_rl_state(
        self, 
        asset_id: str,
        recent_market_data: List[MarketData],
        strategy_ids: List[str],
        timestamp: Optional[datetime] = None
    ) -> RLStateResponse:
        """
        Assemble complete RL state for strategy orchestration.
        
        Args:
            asset_id: Primary asset identifier
            recent_market_data: Recent market data
            strategy_ids: List of strategy identifiers
            timestamp: Current timestamp
            
        Returns:
            Complete RL state response
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        logger.info(f"Assembling RL state for {asset_id} with {len(strategy_ids)} strategies")
        
        # Get base intelligence state
        intelligence_state = await self.assemble_intelligence_state(
            asset_id, recent_market_data, timestamp
        )
        
        # Get strategy embeddings
        strategy_embeddings = {}
        for strategy_id in strategy_ids:
            embeddings = self.pgvector_client.get_strategy_state_embeddings(
                strategy_id, limit=1
            )
            if embeddings:
                strategy_embeddings[strategy_id] = embeddings[0]
        
        # Get portfolio and risk state
        portfolio_state = await self.rust_client.get_portfolio_state()
        risk_metrics = await self.rust_client.get_risk_metrics()
        
        # Assemble state components
        state_components = {
            "market_embeddings": {
                "current_embedding": (
                    intelligence_state.embedding_similarity_context[0].embedding_id
                    if intelligence_state.embedding_similarity_context else None
                ),
                "similarity_context": [
                    {
                        "embedding_id": str(match.embedding_id),
                        "similarity": match.similarity_score,
                        "timestamp": match.timestamp.isoformat()
                    }
                    for match in intelligence_state.embedding_similarity_context
                ]
            },
            "regime_state": {
                "current_regime": intelligence_state.current_regime_label,
                "regime_probabilities": intelligence_state.regime_transition_probabilities,
                "regime_confidence": intelligence_state.regime_confidence
            },
            "graph_features": {
                "cluster_id": intelligence_state.graph_structural_features.asset_cluster_id,
                "centrality": intelligence_state.graph_structural_features.centrality_score,
                "systemic_risk": intelligence_state.graph_structural_features.systemic_risk_proxy
            },
            "portfolio_state": portfolio_state,
            "risk_metrics": risk_metrics,
            "strategy_embeddings": strategy_embeddings
        }
        
        # Assembly metadata
        assembly_metadata = {
            "assembly_timestamp": timestamp.isoformat(),
            "asset_id": asset_id,
            "strategy_count": len(strategy_ids),
            "data_sources": {
                "pgvector": len(intelligence_state.embedding_similarity_context) > 0,
                "neo4j": intelligence_state.current_regime_label is not None,
                "rust_core": portfolio_state.get("net_exposure", 0) != 0
            },
            "validation_status": "passed" if not self._validate_state(intelligence_state) else "warnings"
        }
        
        return RLStateResponse(
            composite_state=intelligence_state,
            state_components=state_components,
            assembly_metadata=assembly_metadata,
            timestamp=timestamp
        )
    
    async def close(self) -> None:
        """Close all client connections."""
        self.pgvector_client.close()
        self.graph_analytics.close()
        await self.rust_client.close()
        logger.info("Composite state assembler closed")