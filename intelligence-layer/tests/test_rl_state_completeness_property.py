"""Property-based tests for composite RL state completeness."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4, UUID
import asyncio

from intelligence_layer.state_assembly import (
    CompositeStateAssembler,
    PgVectorClient,
    RustCoreClient,
    StateValidator
)
from intelligence_layer.models import (
    IntelligenceState,
    SimilarityMatch,
    GraphFeatureSnapshot,
    MarketData,
    RLStateResponse
)
from intelligence_layer.config import Config
from intelligence_layer.regime_detection import RegimeInferencePipeline
from intelligence_layer.graph_analytics import MarketGraphAnalytics


# Strategies for generating test data
@st.composite
def asset_id_strategy(draw):
    """Generate valid asset IDs."""
    return draw(st.text(min_size=3, max_size=12, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))


@st.composite
def strategy_ids_strategy(draw):
    """Generate list of strategy IDs."""
    n_strategies = draw(st.integers(min_value=1, max_value=5))
    return [
        draw(st.text(min_size=5, max_size=15, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
        for _ in range(n_strategies)
    ]


@st.composite
def market_data_strategy(draw):
    """Generate realistic market data."""
    n_points = draw(st.integers(min_value=50, max_value=150))
    asset_id = draw(asset_id_strategy())
    
    data = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=n_points)
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


@st.composite
def similarity_matches_strategy(draw):
    """Generate similarity matches."""
    n_matches = draw(st.integers(min_value=1, max_value=10))
    matches = []
    
    for _ in range(n_matches):
        matches.append(SimilarityMatch(
            embedding_id=uuid4(),
            similarity_score=draw(st.floats(min_value=0.0, max_value=1.0)),
            timestamp=datetime.now(timezone.utc) - timedelta(
                hours=draw(st.integers(min_value=1, max_value=168))
            ),
            metadata=draw(st.dictionaries(
                st.text(min_size=1, max_size=10),
                st.one_of(st.text(), st.floats(), st.integers()),
                min_size=0, max_size=3
            ))
        ))
    
    return matches


@st.composite
def regime_probabilities_strategy(draw):
    """Generate valid regime probabilities that sum to 1.0."""
    n_regimes = draw(st.integers(min_value=2, max_value=5))
    regime_names = [f"regime_{i}" for i in range(n_regimes)]
    
    # Generate random probabilities and normalize
    raw_probs = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(n_regimes)]
    total = sum(raw_probs)
    normalized_probs = [p / total for p in raw_probs]
    
    return dict(zip(regime_names, normalized_probs))


@st.composite
def graph_features_strategy(draw):
    """Generate graph feature snapshots."""
    return GraphFeatureSnapshot(
        asset_cluster_id=str(draw(st.integers(min_value=0, max_value=10))),
        centrality_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        systemic_risk_proxy=draw(st.floats(min_value=0.0, max_value=1.0)),
        timestamp=datetime.now(timezone.utc) - timedelta(
            minutes=draw(st.integers(min_value=1, max_value=60))
        )
    )


@st.composite
def portfolio_state_strategy(draw):
    """Generate portfolio state data."""
    return {
        "net_exposure": draw(st.floats(min_value=-1.0, max_value=1.0)),
        "gross_exposure": draw(st.floats(min_value=0.0, max_value=2.0)),
        "drawdown": draw(st.floats(min_value=0.0, max_value=0.5)),
        "volatility_target_utilization": draw(st.floats(min_value=0.0, max_value=1.5)),
        "unrealized_pnl": draw(st.floats(min_value=-10000.0, max_value=10000.0)),
        "realized_pnl": draw(st.floats(min_value=-10000.0, max_value=10000.0))
    }


@st.composite
def risk_metrics_strategy(draw):
    """Generate risk metrics data."""
    return {
        "var_1d": draw(st.floats(min_value=0.0, max_value=1000.0)),
        "var_5d": draw(st.floats(min_value=0.0, max_value=5000.0)),
        "max_drawdown": draw(st.floats(min_value=0.0, max_value=0.8)),
        "sharpe_ratio": draw(st.floats(min_value=-2.0, max_value=3.0)),
        "volatility": draw(st.floats(min_value=0.0, max_value=1.0))
    }


class TestRLStateCompletenessProperties:
    """Property-based tests for composite RL state completeness.
    
    **Validates: Requirements 2.12, 11.15, 12.7-12.10**
    """
    
    @given(
        asset_id_strategy(),
        market_data_strategy(),
        strategy_ids_strategy(),
        similarity_matches_strategy(),
        regime_probabilities_strategy(),
        graph_features_strategy(),
        portfolio_state_strategy(),
        risk_metrics_strategy()
    )
    @settings(max_examples=5, deadline=10000)
    def test_property_rl_state_contains_all_required_components(
        self, asset_id, market_data, strategy_ids, similarity_matches, 
        regime_probs, graph_features, portfolio_state, risk_metrics
    ):
        """
        Property: For any RL state construction, the state should contain embeddings from pgvector,
        regime labels from Neo4j, transition probabilities from Neo4j, strategy performance context
        from Neo4j, and risk metrics from Rust components.
        
        **Feature: algorithmic-trading-system, Property 6: Composite RL State Completeness**
        **Validates: Requirements 2.12, 11.15, 12.7-12.10**
        """
        # Mock configuration
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.postgres_url = "mock://postgres"
        
        # Mock regime pipeline
        regime_pipeline = Mock(spec=RegimeInferencePipeline)
        regime_response = Mock()
        regime_response.regime_probabilities = regime_probs
        regime_response.transition_likelihoods = regime_probs  # Use same for simplicity
        regime_response.confidence = 0.8
        regime_pipeline.infer_regime.return_value = regime_response
        
        # Mock strategy embeddings
        strategy_embeddings = {}
        for strategy_id in strategy_ids:
            strategy_embeddings[strategy_id] = {
                'id': str(uuid4()),
                'timestamp': datetime.now(timezone.utc),
                'strategy_id': strategy_id,
                'embedding': [0.1] * 128,
                'pnl_state': 0.05,
                'drawdown': 0.02,
                'exposure': 0.3,
                'metadata': {'version': '1.0'}
            }
        
        with patch('intelligence_layer.state_assembly.PgVectorClient') as mock_pgvector, \
             patch('intelligence_layer.state_assembly.MarketGraphAnalytics') as mock_graph, \
             patch('intelligence_layer.state_assembly.RustCoreClient') as mock_rust:
            
            # Setup PgVector mock
            mock_pgvector_instance = Mock()
            mock_pgvector.return_value = mock_pgvector_instance
            
            # Mock recent embeddings
            mock_pgvector_instance.get_recent_market_embeddings.return_value = [
                {
                    'id': str(uuid4()),
                    'timestamp': datetime.now(timezone.utc),
                    'asset_id': asset_id,
                    'regime_id': 'regime_0',
                    'embedding': [0.1] * 128,
                    'volatility': 0.02,
                    'liquidity': 0.8,
                    'horizon': '1h',
                    'source_model': 'tcn_v1',
                    'metadata': {}
                }
            ]
            
            # Mock similarity search
            mock_pgvector_instance.find_similar_embeddings.return_value = similarity_matches
            
            # Mock strategy embeddings
            def mock_get_strategy_embeddings(strategy_id, limit=5):
                if strategy_id in strategy_embeddings:
                    return [strategy_embeddings[strategy_id]]
                return []
            
            mock_pgvector_instance.get_strategy_state_embeddings.side_effect = mock_get_strategy_embeddings
            
            # Setup Graph Analytics mock
            mock_graph_instance = Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.get_graph_features_for_asset.return_value = graph_features
            mock_graph_instance.close = Mock()
            
            # Setup Rust Client mock
            mock_rust_instance = Mock()
            mock_rust.return_value = mock_rust_instance
            mock_rust_instance.get_portfolio_state = AsyncMock(return_value=portfolio_state)
            mock_rust_instance.get_risk_metrics = AsyncMock(return_value=risk_metrics)
            mock_rust_instance.close = AsyncMock()
            
            # Create assembler and set regime pipeline
            assembler = CompositeStateAssembler(config)
            assembler.set_regime_pipeline(regime_pipeline)
            
            # Run the test
            async def run_test():
                rl_state_response = await assembler.assemble_rl_state(
                    asset_id=asset_id,
                    recent_market_data=market_data,
                    strategy_ids=strategy_ids
                )
                
                # **Requirement 2.12**: Intelligence Layer SHALL construct composite RL states
                assert isinstance(rl_state_response, RLStateResponse)
                assert rl_state_response.composite_state is not None
                assert isinstance(rl_state_response.composite_state, IntelligenceState)
                
                # **Requirement 11.15**: Composite RL states combining current embeddings from pgvector
                state_components = rl_state_response.state_components
                assert 'market_embeddings' in state_components
                market_embeddings = state_components['market_embeddings']
                assert 'current_embedding' in market_embeddings
                assert 'similarity_context' in market_embeddings
                assert len(market_embeddings['similarity_context']) > 0
                
                # **Requirement 12.7**: Include 128-dimensional market embeddings from pgvector
                # Verify embedding context is present
                composite_state = rl_state_response.composite_state
                assert composite_state.embedding_similarity_context is not None
                assert len(composite_state.embedding_similarity_context) > 0
                
                # **Requirement 12.8**: Include regime IDs, transition probabilities, and regime entropy from Neo4j
                assert 'regime_state' in state_components
                regime_state = state_components['regime_state']
                assert 'current_regime' in regime_state
                assert 'regime_probabilities' in regime_state
                assert 'regime_confidence' in regime_state
                
                # Verify regime information in composite state
                assert composite_state.current_regime_label is not None
                assert composite_state.regime_transition_probabilities is not None
                assert len(composite_state.regime_transition_probabilities) > 0
                assert composite_state.regime_confidence > 0
                
                # **Requirement 12.9**: Include asset cluster IDs, centrality scores, and systemic risk proxies from Neo4j GDS
                assert 'graph_features' in state_components
                graph_state = state_components['graph_features']
                assert 'cluster_id' in graph_state
                assert 'centrality' in graph_state
                assert 'systemic_risk' in graph_state
                
                # Verify graph features in composite state
                assert composite_state.graph_structural_features is not None
                assert composite_state.graph_structural_features.asset_cluster_id is not None
                assert composite_state.graph_structural_features.centrality_score is not None
                assert composite_state.graph_structural_features.systemic_risk_proxy is not None
                
                # **Requirement 12.10**: Include net exposure, gross exposure, drawdown, and volatility utilization from Rust core
                assert 'portfolio_state' in state_components
                portfolio = state_components['portfolio_state']
                assert 'net_exposure' in portfolio
                assert 'gross_exposure' in portfolio
                assert 'drawdown' in portfolio
                assert 'volatility_target_utilization' in portfolio
                
                assert 'risk_metrics' in state_components
                risk = state_components['risk_metrics']
                assert 'var_1d' in risk
                assert 'max_drawdown' in risk
                assert 'volatility' in risk
                
                # Verify strategy embeddings are included
                assert 'strategy_embeddings' in state_components
                strategy_emb = state_components['strategy_embeddings']
                for strategy_id in strategy_ids:
                    assert strategy_id in strategy_emb
                
                # Verify assembly metadata
                metadata = rl_state_response.assembly_metadata
                assert 'assembly_timestamp' in metadata
                assert 'asset_id' in metadata
                assert metadata['asset_id'] == asset_id
                assert 'strategy_count' in metadata
                assert metadata['strategy_count'] == len(strategy_ids)
                
                # Verify data sources are tracked
                assert 'data_sources' in metadata
                data_sources = metadata['data_sources']
                assert 'pgvector' in data_sources
                assert 'neo4j' in data_sources
                assert 'rust_core' in data_sources
                
                # Verify confidence scores are present
                confidence_scores = composite_state.confidence_scores
                assert 'embedding_similarity' in confidence_scores
                assert 'regime_inference' in confidence_scores
                assert 'graph_features' in confidence_scores
                assert 'portfolio_data' in confidence_scores
                
                # All confidence scores should be between 0 and 1
                for component, score in confidence_scores.items():
                    assert 0.0 <= score <= 1.0, f"Invalid confidence score for {component}: {score}"
                
                await assembler.close()
            
            # Run the async test
            asyncio.run(run_test())
    
    @given(
        asset_id_strategy(),
        market_data_strategy(),
        strategy_ids_strategy()
    )
    @settings(max_examples=3, deadline=8000)
    def test_property_rl_state_handles_missing_data_gracefully(
        self, asset_id, market_data, strategy_ids
    ):
        """
        Property: For any RL state construction with missing data sources,
        the system should handle it gracefully and provide appropriate defaults.
        
        **Feature: algorithmic-trading-system, Property 6: Composite RL State Completeness**
        **Validates: Requirements 2.12, 11.15, 12.7-12.10**
        """
        # Mock configuration
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.postgres_url = "mock://postgres"
        
        with patch('intelligence_layer.state_assembly.PgVectorClient') as mock_pgvector, \
             patch('intelligence_layer.state_assembly.MarketGraphAnalytics') as mock_graph, \
             patch('intelligence_layer.state_assembly.RustCoreClient') as mock_rust:
            
            # Setup mocks to return empty/default data
            mock_pgvector_instance = Mock()
            mock_pgvector.return_value = mock_pgvector_instance
            mock_pgvector_instance.get_recent_market_embeddings.return_value = []
            mock_pgvector_instance.find_similar_embeddings.return_value = []
            mock_pgvector_instance.get_strategy_state_embeddings.return_value = []
            
            mock_graph_instance = Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.get_graph_features_for_asset.return_value = GraphFeatureSnapshot(
                asset_cluster_id="0",
                centrality_score=None,  # Missing data
                systemic_risk_proxy=None,  # Missing data
                timestamp=datetime.now(timezone.utc)
            )
            mock_graph_instance.close = Mock()
            
            mock_rust_instance = Mock()
            mock_rust.return_value = mock_rust_instance
            # Return default/empty portfolio state
            mock_rust_instance.get_portfolio_state = AsyncMock(return_value={
                "net_exposure": 0.0,
                "gross_exposure": 0.0,
                "drawdown": 0.0,
                "volatility_target_utilization": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            })
            mock_rust_instance.get_risk_metrics = AsyncMock(return_value={
                "var_1d": 0.0,
                "var_5d": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0
            })
            mock_rust_instance.close = AsyncMock()
            
            # Create assembler without regime pipeline (missing data)
            assembler = CompositeStateAssembler(config)
            
            async def run_test():
                rl_state_response = await assembler.assemble_rl_state(
                    asset_id=asset_id,
                    recent_market_data=market_data,
                    strategy_ids=strategy_ids
                )
                
                # Should still create a valid response structure
                assert isinstance(rl_state_response, RLStateResponse)
                assert rl_state_response.composite_state is not None
                
                # Should handle missing embeddings gracefully
                composite_state = rl_state_response.composite_state
                assert composite_state.embedding_similarity_context == []
                
                # Should handle missing regime data gracefully
                assert composite_state.current_regime_label is None
                assert composite_state.regime_transition_probabilities == {}
                assert composite_state.regime_confidence == 0.0
                
                # Should still have graph features structure
                assert composite_state.graph_structural_features is not None
                
                # Should have confidence scores reflecting missing data
                confidence_scores = composite_state.confidence_scores
                assert confidence_scores['embedding_similarity'] == 0.0  # No embeddings
                assert confidence_scores['regime_inference'] == 0.0  # No regime pipeline
                
                # Assembly metadata should reflect missing data sources
                metadata = rl_state_response.assembly_metadata
                data_sources = metadata['data_sources']
                assert data_sources['pgvector'] is False  # No embeddings
                assert data_sources['neo4j'] is False  # No regime data
                assert data_sources['rust_core'] is False  # Default portfolio state
                
                await assembler.close()
            
            asyncio.run(run_test())
    
    @given(
        asset_id_strategy(),
        market_data_strategy(),
        strategy_ids_strategy(),
        similarity_matches_strategy()
    )
    @settings(max_examples=3, deadline=8000)
    def test_property_rl_state_validation_consistency(
        self, asset_id, market_data, strategy_ids, similarity_matches
    ):
        """
        Property: For any RL state construction, validation should be consistent
        and provide meaningful feedback about data quality.
        
        **Feature: algorithmic-trading-system, Property 6: Composite RL State Completeness**
        **Validates: Requirements 2.12, 11.15, 12.7-12.10**
        """
        # Mock configuration
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.postgres_url = "mock://postgres"
        
        # Create validator
        validator = StateValidator()
        
        # Test embedding context validation
        valid, errors = validator.validate_embedding_context(similarity_matches)
        if similarity_matches:
            assert isinstance(valid, bool)
            assert isinstance(errors, list)
            
            if not valid:
                assert len(errors) > 0
                for error in errors:
                    assert isinstance(error, str)
                    assert len(error) > 0
        
        # Test regime probabilities validation
        regime_probs = {"regime_1": 0.6, "regime_2": 0.4}
        valid, errors = validator.validate_regime_probabilities(regime_probs)
        assert valid is True
        assert len(errors) == 0
        
        # Test invalid regime probabilities
        invalid_regime_probs = {"regime_1": 0.8, "regime_2": 0.4}  # Sum > 1.0
        valid, errors = validator.validate_regime_probabilities(invalid_regime_probs)
        assert valid is False
        assert len(errors) > 0
        
        # Test graph features validation
        valid_graph_features = GraphFeatureSnapshot(
            asset_cluster_id="1",
            centrality_score=0.5,
            systemic_risk_proxy=0.3,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        valid, errors = validator.validate_graph_features(valid_graph_features)
        assert valid is True
        assert len(errors) == 0
        
        # Test confidence scores validation
        confidence_scores = {
            "embedding_similarity": 0.8,
            "regime_inference": 0.7,
            "graph_features": 0.9,
            "portfolio_data": 0.6
        }
        valid, errors = validator.validate_confidence_scores(confidence_scores)
        assert valid is True
        assert len(errors) == 0
        
        # Test invalid confidence scores
        invalid_confidence = {
            "embedding_similarity": 1.5,  # > 1.0
            "regime_inference": -0.1,     # < 0.0
        }
        valid, errors = validator.validate_confidence_scores(invalid_confidence)
        assert valid is False
        assert len(errors) == 2  # Two invalid scores
    
    @given(
        asset_id_strategy(),
        market_data_strategy(),
        strategy_ids_strategy()
    )
    @settings(max_examples=3, deadline=8000)
    def test_property_rl_state_timestamp_consistency(
        self, asset_id, market_data, strategy_ids
    ):
        """
        Property: For any RL state construction, timestamps should be consistent
        and not contain future dates.
        
        **Feature: algorithmic-trading-system, Property 6: Composite RL State Completeness**
        **Validates: Requirements 2.12, 11.15, 12.7-12.10**
        """
        # Mock configuration
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.postgres_url = "mock://postgres"
        
        current_time = datetime.now(timezone.utc)
        
        with patch('intelligence_layer.state_assembly.PgVectorClient') as mock_pgvector, \
             patch('intelligence_layer.state_assembly.MarketGraphAnalytics') as mock_graph, \
             patch('intelligence_layer.state_assembly.RustCoreClient') as mock_rust:
            
            # Setup mocks with consistent timestamps
            mock_pgvector_instance = Mock()
            mock_pgvector.return_value = mock_pgvector_instance
            mock_pgvector_instance.get_recent_market_embeddings.return_value = []
            mock_pgvector_instance.find_similar_embeddings.return_value = []
            mock_pgvector_instance.get_strategy_state_embeddings.return_value = []
            
            mock_graph_instance = Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.get_graph_features_for_asset.return_value = GraphFeatureSnapshot(
                asset_cluster_id="1",
                centrality_score=0.5,
                systemic_risk_proxy=0.3,
                timestamp=current_time - timedelta(minutes=5)  # Past timestamp
            )
            mock_graph_instance.close = Mock()
            
            mock_rust_instance = Mock()
            mock_rust.return_value = mock_rust_instance
            mock_rust_instance.get_portfolio_state = AsyncMock(return_value={
                "net_exposure": 0.1,
                "gross_exposure": 0.2,
                "drawdown": 0.05,
                "volatility_target_utilization": 0.8,
                "unrealized_pnl": 100.0,
                "realized_pnl": 50.0
            })
            mock_rust_instance.get_risk_metrics = AsyncMock(return_value={
                "var_1d": 100.0,
                "var_5d": 300.0,
                "max_drawdown": 0.1,
                "sharpe_ratio": 1.2,
                "volatility": 0.15
            })
            mock_rust_instance.close = AsyncMock()
            
            assembler = CompositeStateAssembler(config)
            
            async def run_test():
                rl_state_response = await assembler.assemble_rl_state(
                    asset_id=asset_id,
                    recent_market_data=market_data,
                    strategy_ids=strategy_ids,
                    timestamp=current_time
                )
                
                # Verify main timestamp
                assert rl_state_response.timestamp <= current_time
                assert rl_state_response.composite_state.timestamp <= current_time
                
                # Verify assembly metadata timestamp
                metadata = rl_state_response.assembly_metadata
                assembly_time_str = metadata['assembly_timestamp']
                assembly_time = datetime.fromisoformat(assembly_time_str.replace('Z', '+00:00'))
                assert assembly_time <= current_time + timedelta(seconds=1)  # Allow small processing time
                
                # Verify graph features timestamp is not in future
                graph_features = rl_state_response.composite_state.graph_structural_features
                if graph_features and graph_features.timestamp:
                    assert graph_features.timestamp <= current_time
                
                await assembler.close()
            
            asyncio.run(run_test())


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])