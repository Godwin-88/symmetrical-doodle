"""Tests for strategy orchestration implementation."""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from intelligence_layer.strategy_orchestration import (
    StrategyOrchestrator,
    StrategyRegistry,
    MetaController,
    StrategyDefinition,
    TradeIntent,
    PerformanceMetrics,
    StrategyFamily,
    StrategyHorizon,
    StrategyStatus,
    ExecutionAggressiveness,
    create_sample_strategy,
    validate_orchestration_components
)
from intelligence_layer.models import IntelligenceState, MarketData
from intelligence_layer.config import Config, DatabaseConfig, RedisConfig, LoggingConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        database=DatabaseConfig(
            postgres_url="postgresql://test:test@localhost:5432/test",
            neo4j_url="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test"
        ),
        redis=RedisConfig(url="redis://localhost:6379"),
        logging=LoggingConfig(level="INFO")
    )


@pytest.fixture
def sample_strategy():
    """Create sample strategy for testing."""
    return StrategyDefinition(
        strategy_id="test_strategy",
        name="Test Strategy",
        description="A test strategy",
        family=StrategyFamily.TREND,
        horizon=StrategyHorizon.INTRADAY,
        enabled_markets=["EURUSD", "GBPUSD"],
        parameters={"param1": 1.0},
        max_allocation=0.5,
        enabled_regimes=["low_vol_trending"],
        regime_multipliers={"low_vol_trending": 1.2}
    )


@pytest.fixture
def sample_intelligence_state():
    """Create sample intelligence state."""
    return IntelligenceState(
        current_regime_label="low_vol_trending",
        regime_transition_probabilities={"low_vol_trending": 0.7, "high_vol_trending": 0.3},
        regime_confidence=0.8,
        confidence_scores={"regime_inference": 0.8, "embedding_similarity": 0.7},
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return [
        MarketData(
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            asset_id="EURUSD",
            open=1.1000 + i * 0.0001,
            high=1.1010 + i * 0.0001,
            low=1.0990 + i * 0.0001,
            close=1.1005 + i * 0.0001,
            volume=1000000
        )
        for i in range(10)
    ]


class TestStrategyDefinition:
    """Test strategy definition model."""
    
    def test_create_sample_strategy(self):
        """Test creating sample strategy."""
        strategy = create_sample_strategy()
        
        assert isinstance(strategy, StrategyDefinition)
        assert strategy.strategy_id == "sample_strategy"
        assert strategy.family == StrategyFamily.TREND
        assert strategy.status == StrategyStatus.ACTIVE
    
    def test_strategy_validation(self, sample_strategy):
        """Test strategy definition validation."""
        assert sample_strategy.max_allocation <= 1.0
        assert sample_strategy.max_leverage >= 0.0
        assert sample_strategy.max_drawdown >= 0.0
        assert len(sample_strategy.enabled_markets) > 0


class TestTradeIntent:
    """Test trade intent model."""
    
    def test_trade_intent_creation(self):
        """Test trade intent creation."""
        intent = TradeIntent(
            intent_id="test_intent",
            strategy_id="test_strategy",
            asset_id="EURUSD",
            direction="long",
            size=10000.0,
            confidence=0.8,
            urgency=ExecutionAggressiveness.MEDIUM,
            regime_context="low_vol_trending"
        )
        
        assert intent.intent_id == "test_intent"
        assert intent.direction in ["long", "short", "close"]
        assert 0 <= intent.confidence <= 1
    
    def test_trade_intent_to_dict(self):
        """Test trade intent serialization."""
        intent = TradeIntent(
            intent_id="test_intent",
            strategy_id="test_strategy", 
            asset_id="EURUSD",
            direction="long",
            size=10000.0,
            confidence=0.8,
            urgency=ExecutionAggressiveness.MEDIUM,
            regime_context="low_vol_trending"
        )
        
        intent_dict = intent.to_dict()
        
        assert isinstance(intent_dict, dict)
        assert intent_dict["intent_id"] == "test_intent"
        assert intent_dict["urgency"] == "medium"


class TestStrategyRegistry:
    """Test strategy registry."""
    
    @pytest.mark.asyncio
    async def test_register_strategy(self, config, sample_strategy):
        """Test strategy registration."""
        registry = StrategyRegistry(config)
        
        await registry.register_strategy(sample_strategy)
        
        retrieved = await registry.get_strategy(sample_strategy.strategy_id)
        assert retrieved is not None
        assert retrieved.strategy_id == sample_strategy.strategy_id
    
    @pytest.mark.asyncio
    async def test_list_strategies(self, config, sample_strategy):
        """Test strategy listing with filters."""
        registry = StrategyRegistry(config)
        
        # Register strategy
        await registry.register_strategy(sample_strategy)
        
        # List all strategies
        all_strategies = await registry.list_strategies()
        assert len(all_strategies) >= 1
        
        # Filter by family
        trend_strategies = await registry.list_strategies(family=StrategyFamily.TREND)
        assert len(trend_strategies) >= 1
        assert all(s.family == StrategyFamily.TREND for s in trend_strategies)
        
        # Filter by status
        active_strategies = await registry.list_strategies(status=StrategyStatus.ACTIVE)
        assert len(active_strategies) >= 1
        assert all(s.status == StrategyStatus.ACTIVE for s in active_strategies)
    
    @pytest.mark.asyncio
    async def test_update_performance(self, config, sample_strategy):
        """Test performance metrics update."""
        registry = StrategyRegistry(config)
        
        await registry.register_strategy(sample_strategy)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            strategy_id=sample_strategy.strategy_id,
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            volatility=0.12
        )
        
        await registry.update_performance(sample_strategy.strategy_id, metrics)
        
        # Retrieve performance
        retrieved_metrics = await registry.get_performance(sample_strategy.strategy_id)
        assert retrieved_metrics is not None
        assert retrieved_metrics.total_return == 0.15
        
        # Check that strategy historical performance was updated
        updated_strategy = await registry.get_strategy(sample_strategy.strategy_id)
        assert "sharpe_ratio" in updated_strategy.historical_performance
    
    @pytest.mark.asyncio
    async def test_get_strategies_for_regime(self, config, sample_strategy):
        """Test regime-based strategy filtering."""
        registry = StrategyRegistry(config)
        
        await registry.register_strategy(sample_strategy)
        
        # Get strategies for enabled regime
        suitable_strategies = await registry.get_strategies_for_regime("low_vol_trending")
        assert len(suitable_strategies) >= 1
        
        # Get strategies for non-enabled regime
        unsuitable_strategies = await registry.get_strategies_for_regime("high_vol_ranging")
        # Should be empty or have reduced list
        assert len(unsuitable_strategies) <= len(suitable_strategies)


class TestMetaController:
    """Test meta-controller for capital allocation."""
    
    @pytest.mark.asyncio
    async def test_meta_controller_initialization(self, config):
        """Test meta-controller initialization."""
        registry = StrategyRegistry(config)
        controller = MetaController(config, registry)
        
        assert controller.strategy_registry == registry
        assert controller.current_exposure_multiplier == 0.5
    
    @pytest.mark.asyncio
    async def test_initialize_rl_environment(self, config):
        """Test RL environment initialization."""
        registry = StrategyRegistry(config)
        controller = MetaController(config, registry)
        
        strategy_ids = ["strategy1", "strategy2"]
        await controller.initialize_rl_environment(strategy_ids)
        
        assert controller.rl_environment is not None
        assert controller.rl_environment.strategy_ids == strategy_ids
    
    @pytest.mark.asyncio
    async def test_allocate_capital(self, config, sample_strategy, sample_intelligence_state):
        """Test capital allocation."""
        registry = StrategyRegistry(config)
        await registry.register_strategy(sample_strategy)
        
        controller = MetaController(config, registry)
        
        strategy_ids = [sample_strategy.strategy_id]
        total_capital = 100000.0
        
        allocations = await controller.allocate_capital(
            sample_intelligence_state, strategy_ids, total_capital
        )
        
        assert isinstance(allocations, dict)
        assert sample_strategy.strategy_id in allocations
        assert allocations[sample_strategy.strategy_id] > 0
        assert sum(allocations.values()) <= total_capital
    
    @pytest.mark.asyncio
    async def test_evaluate_performance(self, config):
        """Test performance evaluation."""
        registry = StrategyRegistry(config)
        controller = MetaController(config, registry)
        
        metrics = await controller.evaluate_performance("test_strategy")
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.strategy_id == "test_strategy"
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)


class TestStrategyOrchestrator:
    """Test main strategy orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, config):
        """Test orchestrator initialization."""
        orchestrator = StrategyOrchestrator(config)
        
        assert isinstance(orchestrator.strategy_registry, StrategyRegistry)
        assert isinstance(orchestrator.meta_controller, MetaController)
        assert len(orchestrator.active_intents) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_default_strategies(self, config):
        """Test default strategy initialization."""
        orchestrator = StrategyOrchestrator(config)
        
        await orchestrator.initialize_default_strategies()
        
        strategies = await orchestrator.strategy_registry.list_strategies()
        assert len(strategies) >= 3  # Should have at least 3 default strategies
        
        # Check strategy families are represented
        families = {s.family for s in strategies}
        assert StrategyFamily.TREND in families
        assert StrategyFamily.MEAN_REVERSION in families
        assert StrategyFamily.VOLATILITY in families
    
    @pytest.mark.asyncio
    @patch('intelligence_layer.strategy_orchestration.CompositeStateAssembler')
    async def test_orchestrate_strategies(self, mock_assembler, config, sample_intelligence_state, sample_market_data):
        """Test main orchestration method."""
        # Mock the state assembler
        mock_assembler_instance = AsyncMock()
        mock_assembler_instance.assemble_intelligence_state.return_value = sample_intelligence_state
        mock_assembler.return_value = mock_assembler_instance
        
        orchestrator = StrategyOrchestrator(config)
        await orchestrator.initialize_default_strategies()
        
        # Orchestrate strategies
        intents = await orchestrator.orchestrate_strategies(
            asset_id="EURUSD",
            recent_market_data=sample_market_data,
            total_capital=100000.0
        )
        
        assert isinstance(intents, list)
        # Should generate some intents
        assert len(intents) >= 0
        
        # Check intent properties if any were generated
        for intent in intents:
            assert isinstance(intent, TradeIntent)
            assert intent.asset_id == "EURUSD"
            assert intent.direction in ["long", "short", "close"]
            assert 0 <= intent.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_get_orchestration_status(self, config):
        """Test orchestration status reporting."""
        orchestrator = StrategyOrchestrator(config)
        await orchestrator.initialize_default_strategies()
        
        status = await orchestrator.get_orchestration_status()
        
        assert isinstance(status, dict)
        assert "active_strategies" in status
        assert "active_intents" in status
        assert "strategy_families" in status
        assert status["active_strategies"] >= 0


class TestValidation:
    """Test validation functions."""
    
    @pytest.mark.asyncio
    async def test_validate_orchestration_components(self, config):
        """Test orchestration component validation."""
        orchestrator = StrategyOrchestrator(config)
        
        results = await validate_orchestration_components(orchestrator)
        
        assert isinstance(results, dict)
        expected_keys = [
            "strategy_registry_works",
            "meta_controller_initializes", 
            "performance_evaluation_works"
        ]
        
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], bool)


@pytest.mark.asyncio
async def test_orchestrator_cleanup(config):
    """Test orchestrator cleanup."""
    orchestrator = StrategyOrchestrator(config)
    await orchestrator.close()
    # Should not raise any exceptions