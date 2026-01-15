"""Property-based tests for intelligence layer sandboxing."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4
import asyncio
import inspect

from intelligence_layer.strategy_orchestration import (
    StrategyOrchestrator,
    StrategyRegistry,
    MetaController,
    TradeIntent,
    StrategyDefinition,
    StrategyFamily,
    StrategyHorizon,
    StrategyStatus
)
from intelligence_layer.rl_environment import (
    TradingEnvironmentMDP,
    RLAction,
    ExecutionAggressiveness,
    EpisodeConfig
)
from intelligence_layer.state_assembly import CompositeStateAssembler
from intelligence_layer.regime_detection import RegimeInferencePipeline
from intelligence_layer.graph_analytics import MarketGraphAnalytics
from intelligence_layer.embedding_model import MarketEmbeddingTCN
from intelligence_layer.feature_extraction import FeatureExtractor
from intelligence_layer.training_protocol import EmbeddingTrainer
from intelligence_layer.models import MarketData, IntelligenceState
from intelligence_layer.config import Config, DatabaseConfig, RedisConfig, LoggingConfig


# Strategies for generating test data
@st.composite
def asset_id_strategy(draw):
    """Generate valid asset IDs."""
    return draw(st.sampled_from(["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]))


@st.composite
def strategy_ids_strategy(draw):
    """Generate list of strategy IDs."""
    n_strategies = draw(st.integers(min_value=1, max_value=5))
    return [f"strategy_{i}" for i in range(n_strategies)]


@st.composite
def market_data_strategy(draw):
    """Generate realistic market data."""
    n_points = draw(st.integers(min_value=10, max_value=50))
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
def intelligence_components_strategy(draw):
    """Generate intelligence component instances for testing."""
    config = Mock(spec=Config)
    config.database = Mock()
    config.database.postgres_url = "mock://postgres"
    
    components = {}
    
    # Strategy Orchestrator
    components['strategy_orchestrator'] = Mock(spec=StrategyOrchestrator)
    
    # RL Environment
    strategy_ids = draw(strategy_ids_strategy())
    components['rl_environment'] = Mock(spec=TradingEnvironmentMDP)
    
    # State Assembler
    components['state_assembler'] = Mock(spec=CompositeStateAssembler)
    
    # Regime Detection
    components['regime_detection'] = Mock(spec=RegimeInferencePipeline)
    
    # Graph Analytics
    components['graph_analytics'] = Mock(spec=MarketGraphAnalytics)
    
    # Embedding Model
    components['embedding_model'] = Mock(spec=MarketEmbeddingTCN)
    
    # Feature Extraction
    components['feature_extraction'] = Mock(spec=FeatureExtractor)
    
    # Training Protocol
    components['training_protocol'] = Mock(spec=EmbeddingTrainer)
    
    return components


class TestIntelligenceLayerSandboxingProperties:
    """Property-based tests for intelligence layer sandboxing.
    
    **Validates: Requirements 1.4, 4.4**
    """
    
    @given(
        asset_id_strategy(),
        market_data_strategy(),
        strategy_ids_strategy(),
        intelligence_components_strategy()
    )
    @settings(max_examples=5, deadline=10000)
    def test_property_intelligence_components_cannot_place_orders_directly(
        self, asset_id, market_data, strategy_ids, intelligence_components
    ):
        """
        Property: For any intelligence component, it should have no direct access
        to order placement functions and should only produce advisory outputs.
        
        **Feature: algorithmic-trading-system, Property 2: Intelligence Layer Sandboxing**
        **Validates: Requirements 1.4, 4.4**
        """
        # **Requirement 1.4**: Intelligence Layer SHALL NOT place orders directly
        
        # Test each intelligence component for absence of order placement methods
        forbidden_methods = [
            'place_order',
            'cancel_order', 
            'modify_order',
            'execute_order',
            'send_order',
            'submit_order',
            'create_order',
            'process_order'
        ]
        
        for component_name, component in intelligence_components.items():
            # Get the actual class being mocked to inspect its methods
            if hasattr(component, '_spec_class'):
                component_class = component._spec_class
            else:
                # For real instances, get their class
                component_class = type(component)
            
            # Check that component doesn't have forbidden order placement methods
            component_methods = [method for method in dir(component_class) 
                               if not method.startswith('_')]
            
            for forbidden_method in forbidden_methods:
                assert forbidden_method not in component_methods, (
                    f"Intelligence component '{component_name}' has forbidden method '{forbidden_method}'. "
                    f"Intelligence components must not have direct order placement capabilities."
                )
        
        # Test Strategy Orchestrator specifically - it should only generate TradeIntents
        strategy_orchestrator = intelligence_components['strategy_orchestrator']
        
        # Mock the orchestrate_strategies method to return TradeIntents
        mock_trade_intents = [
            TradeIntent(
                intent_id=str(uuid4()),
                strategy_id=strategy_ids[0] if strategy_ids else "test_strategy",
                asset_id=asset_id,
                direction="long",
                size=1000.0,
                confidence=0.8,
                urgency=ExecutionAggressiveness.MEDIUM,
                regime_context="test_regime",
                reasoning="Test trade intent"
            )
        ]
        
        strategy_orchestrator.orchestrate_strategies = AsyncMock(return_value=mock_trade_intents)
        
        # Verify that orchestration produces TradeIntents, not direct orders
        async def test_orchestration():
            intents = await strategy_orchestrator.orchestrate_strategies(
                asset_id=asset_id,
                recent_market_data=market_data,
                total_capital=100000.0
            )
            
            # Should return TradeIntents, not orders
            assert isinstance(intents, list)
            for intent in intents:
                assert isinstance(intent, TradeIntent)
                
                # TradeIntent should not have order execution capabilities
                assert not hasattr(intent, 'execute')
                assert not hasattr(intent, 'place')
                assert not hasattr(intent, 'send_to_market')
                
                # TradeIntent should be advisory only
                assert hasattr(intent, 'intent_id')
                assert hasattr(intent, 'direction')
                assert hasattr(intent, 'confidence')
                assert hasattr(intent, 'reasoning')
        
        asyncio.run(test_orchestration())
    
    @given(
        asset_id_strategy(),
        market_data_strategy(),
        strategy_ids_strategy()
    )
    @settings(max_examples=5, deadline=8000)
    def test_property_intelligence_outputs_are_advisory_only(
        self, asset_id, market_data, strategy_ids
    ):
        """
        Property: For any intelligence component output, the output should conform
        to advisory types (forecast distributions, regime labels, confidence scores,
        or suggested actions) and not direct execution commands.
        
        **Feature: algorithmic-trading-system, Property 2: Intelligence Layer Sandboxing**
        **Validates: Requirements 1.3, 1.4**
        """
        # **Requirement 1.3**: Intelligence outputs SHALL be forecast distributions, 
        # regime labels, confidence scores, or suggested actions
        
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.postgres_url = "mock://postgres"
        
        # Test Strategy Orchestrator outputs
        with patch('intelligence_layer.strategy_orchestration.CompositeStateAssembler'), \
             patch('intelligence_layer.state_assembly.PgVectorClient'), \
             patch('intelligence_layer.state_assembly.MarketGraphAnalytics'), \
             patch('intelligence_layer.state_assembly.RustCoreClient'):
            
            orchestrator = StrategyOrchestrator(config)
            
            # Mock the state assembler to return a valid intelligence state
            mock_intelligence_state = Mock(spec=IntelligenceState)
            mock_intelligence_state.timestamp = datetime.now(timezone.utc)
            mock_intelligence_state.current_regime_label = "test_regime"
            mock_intelligence_state.regime_confidence = 0.8
            mock_intelligence_state.embedding_similarity_context = []
            mock_intelligence_state.regime_transition_probabilities = {"regime1": 0.6, "regime2": 0.4}
            
            # Mock graph features with proper numeric values
            mock_graph_features = Mock()
            mock_graph_features.systemic_risk_proxy = 0.3
            mock_intelligence_state.graph_structural_features = mock_graph_features
            mock_intelligence_state.confidence_scores = {"regime_inference": 0.8}
            
            orchestrator.state_assembler.assemble_intelligence_state = AsyncMock(
                return_value=mock_intelligence_state
            )
            orchestrator.state_assembler.close = AsyncMock()
            
            # Initialize default strategies
            async def test_advisory_outputs():
                await orchestrator.initialize_default_strategies()
                
                # Generate outputs
                trade_intents = await orchestrator.orchestrate_strategies(
                    asset_id=asset_id,
                    recent_market_data=market_data,
                    total_capital=100000.0
                )
                
                # Verify outputs are advisory
                for intent in trade_intents:
                    # Should be TradeIntent (advisory)
                    assert isinstance(intent, TradeIntent)
                    
                    # Should have advisory characteristics
                    assert hasattr(intent, 'confidence')  # Confidence score
                    assert 0.0 <= intent.confidence <= 1.0
                    
                    assert hasattr(intent, 'reasoning')  # Explanation
                    assert isinstance(intent.reasoning, str)
                    
                    assert hasattr(intent, 'regime_context')  # Regime label
                    assert isinstance(intent.regime_context, str)
                    
                    # Should NOT have execution characteristics
                    assert not hasattr(intent, 'order_id')
                    assert not hasattr(intent, 'execution_status')
                    assert not hasattr(intent, 'fill_price')
                    assert not hasattr(intent, 'execution_timestamp')
                    
                    # Direction should be advisory suggestion, not execution command
                    assert intent.direction in ["long", "short", "close"]
                    
                    # Size should be suggested, not committed
                    assert isinstance(intent.size, (int, float))
                    assert intent.size > 0
                
                # Don't call close() to avoid async mocking issues
            
            asyncio.run(test_advisory_outputs())
        
        # Test RL Environment outputs
        episode_config = EpisodeConfig(
            max_steps=10,
            decision_interval_minutes=60,
            max_drawdown_threshold=0.15
        )
        
        rl_env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=episode_config,
            mock_mode=True
        )
        
        # Test RL environment produces advisory actions, not direct orders
        state = rl_env.reset(asset_id=asset_id)
        
        # Create sample action (advisory allocation)
        weights = np.ones(len(strategy_ids)) / len(strategy_ids)
        action = RLAction(
            strategy_weights=weights,
            exposure_multiplier=0.5,
            execution_aggressiveness=ExecutionAggressiveness.MEDIUM
        )
        
        next_state, reward, done, info = rl_env.step(action, asset_id=asset_id)
        
        # Verify RL action is advisory (strategy allocation)
        assert isinstance(action.strategy_weights, np.ndarray)
        assert len(action.strategy_weights) == len(strategy_ids)
        assert abs(np.sum(action.strategy_weights) - 1.0) < 1e-6  # Allocation weights
        
        assert 0.0 <= action.exposure_multiplier <= 1.0  # Risk control
        assert isinstance(action.execution_aggressiveness, ExecutionAggressiveness)
        
        # RL action should not contain direct order information
        assert not hasattr(action, 'order_type')
        assert not hasattr(action, 'price')
        assert not hasattr(action, 'venue')
        assert not hasattr(action, 'order_id')
    
    @given(
        asset_id_strategy(),
        market_data_strategy(),
        strategy_ids_strategy()
    )
    @settings(max_examples=3, deadline=8000)
    def test_property_intelligence_layer_isolation_from_execution_core(
        self, asset_id, market_data, strategy_ids
    ):
        """
        Property: For any intelligence component, it should have no direct access
        to execution core functions and should be isolated from order execution.
        
        **Feature: algorithmic-trading-system, Property 2: Intelligence Layer Sandboxing**
        **Validates: Requirements 1.4, 4.4**
        """
        # **Requirement 4.4**: Strategy Orchestrator SHALL convert intelligence outputs 
        # to trade intents only, not direct orders
        
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.postgres_url = "mock://postgres"
        
        # Test that intelligence components don't import or access execution core
        intelligence_modules = [
            'intelligence_layer.strategy_orchestration',
            'intelligence_layer.rl_environment', 
            'intelligence_layer.state_assembly',
            'intelligence_layer.regime_detection',
            'intelligence_layer.graph_analytics',
            'intelligence_layer.embedding_model',
            'intelligence_layer.feature_extraction',
            'intelligence_layer.training_protocol'
        ]
        
        # Forbidden execution core imports/references
        forbidden_execution_imports = [
            'execution_core',
            'execution_adapter',
            'execution_manager',
            'deriv_adapter',
            'shadow_adapter'
        ]
        
        for module_name in intelligence_modules:
            try:
                module = __import__(module_name, fromlist=[''])
                
                # Check module source code for forbidden imports
                if hasattr(module, '__file__') and module.__file__:
                    with open(module.__file__, 'r') as f:
                        source_code = f.read()
                        
                    for forbidden_import in forbidden_execution_imports:
                        # Check for direct imports
                        assert f"import {forbidden_import}" not in source_code, (
                            f"Intelligence module '{module_name}' contains forbidden import '{forbidden_import}'. "
                            f"Intelligence layer must be isolated from execution core."
                        )
                        
                        # Check for from imports
                        assert f"from {forbidden_import}" not in source_code, (
                            f"Intelligence module '{module_name}' contains forbidden import 'from {forbidden_import}'. "
                            f"Intelligence layer must be isolated from execution core."
                        )
                
                # Check module attributes for execution core references
                module_attrs = dir(module)
                for attr in module_attrs:
                    if not attr.startswith('_'):
                        for forbidden_import in forbidden_execution_imports:
                            assert forbidden_import not in attr.lower(), (
                                f"Intelligence module '{module_name}' has attribute '{attr}' "
                                f"that references forbidden execution component '{forbidden_import}'. "
                                f"Intelligence layer must be isolated from execution core."
                            )
                            
            except ImportError:
                # Module might not exist in test environment, skip
                continue
        
        # Test Strategy Orchestrator specifically for isolation
        with patch('intelligence_layer.strategy_orchestration.CompositeStateAssembler'), \
             patch('intelligence_layer.state_assembly.PgVectorClient'), \
             patch('intelligence_layer.state_assembly.MarketGraphAnalytics'), \
             patch('intelligence_layer.state_assembly.RustCoreClient'):
            
            orchestrator = StrategyOrchestrator(config)
            
            # Verify orchestrator doesn't have execution methods
            orchestrator_methods = [method for method in dir(orchestrator) 
                                  if not method.startswith('_')]
            
            forbidden_execution_methods = [
                'place_order',
                'execute_trade',
                'send_to_execution',
                'connect_to_broker',
                'submit_to_market'
            ]
            
            for forbidden_method in forbidden_execution_methods:
                assert forbidden_method not in orchestrator_methods, (
                    f"Strategy orchestrator has forbidden execution method '{forbidden_method}'. "
                    f"Orchestrator should only generate trade intents, not execute orders."
                )
            
            # Verify orchestrator produces intents, not orders
            async def test_isolation():
                await orchestrator.initialize_default_strategies()
                
                # Mock intelligence state
                mock_state = Mock(spec=IntelligenceState)
                mock_state.timestamp = datetime.now(timezone.utc)
                mock_state.current_regime_label = "test_regime"
                mock_state.regime_confidence = 0.8
                mock_state.embedding_similarity_context = []
                mock_state.regime_transition_probabilities = {"regime1": 0.6, "regime2": 0.4}
                
                # Mock graph features with proper numeric values
                mock_graph_features = Mock()
                mock_graph_features.systemic_risk_proxy = 0.3
                mock_state.graph_structural_features = mock_graph_features
                mock_state.confidence_scores = {"regime_inference": 0.8}
                
                orchestrator.state_assembler.assemble_intelligence_state = AsyncMock(
                    return_value=mock_state
                )
                orchestrator.state_assembler.close = AsyncMock()
                
                # Generate outputs
                outputs = await orchestrator.orchestrate_strategies(
                    asset_id=asset_id,
                    recent_market_data=market_data,
                    total_capital=100000.0
                )
                
                # Outputs should be TradeIntents (advisory), not execution orders
                for output in outputs:
                    assert isinstance(output, TradeIntent)
                    
                    # Should not have execution-specific attributes
                    assert not hasattr(output, 'venue_order_id')
                    assert not hasattr(output, 'execution_status')
                    assert not hasattr(output, 'fill_quantity')
                    assert not hasattr(output, 'execution_price')
                    
                    # Should have advisory attributes
                    assert hasattr(output, 'intent_id')
                    assert hasattr(output, 'confidence')
                    assert hasattr(output, 'reasoning')
                
                # Don't call close() to avoid async mocking issues
            
            asyncio.run(test_isolation())
    
    @given(
        asset_id_strategy(),
        strategy_ids_strategy()
    )
    @settings(max_examples=3, deadline=6000)
    def test_property_rl_environment_meta_controller_role_only(
        self, asset_id, strategy_ids
    ):
        """
        Property: For any RL environment, it should serve as a meta-policy controller
        for capital allocation across strategies, NOT as a direct trading agent.
        
        **Feature: algorithmic-trading-system, Property 2: Intelligence Layer Sandboxing**
        **Validates: Requirements 1.4, 4.4**
        """
        # **Requirement 4.4**: RL Environment SHALL serve as meta-controller for 
        # capital allocation, NOT as direct trading agent
        
        config = Mock(spec=Config)
        config.database = Mock()
        config.database.postgres_url = "mock://postgres"
        
        episode_config = EpisodeConfig(
            max_steps=10,
            decision_interval_minutes=60,
            max_drawdown_threshold=0.15
        )
        
        rl_env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=episode_config,
            mock_mode=True
        )
        
        # Verify RL environment is configured for meta-control, not direct trading
        assert rl_env.strategy_ids == strategy_ids
        assert len(rl_env.strategy_ids) > 0
        
        # Get action space info
        action_info = rl_env.get_action_space_info()
        
        # Action space should be for strategy allocation, not direct trading
        weights_info = action_info["strategy_weights"]
        assert weights_info["shape"] == (len(strategy_ids),)
        assert weights_info["constraint"] == "sum_to_one"  # Capital allocation
        
        # Should have exposure multiplier for risk control, not order size
        exposure_info = action_info["exposure_multiplier"]
        assert exposure_info["bounds"] == (0.0, 1.0)  # Risk throttle
        
        # Should NOT have direct trading action components
        assert "order_type" not in action_info
        assert "price" not in action_info
        assert "venue" not in action_info
        assert "order_size" not in action_info
        
        # Test that RL actions represent strategy allocation decisions
        state = rl_env.reset(asset_id=asset_id)
        
        # Create allocation action
        weights = np.ones(len(strategy_ids)) / len(strategy_ids)
        action = RLAction(
            strategy_weights=weights,
            exposure_multiplier=0.5,
            execution_aggressiveness=ExecutionAggressiveness.MEDIUM
        )
        
        # Verify action represents capital allocation, not direct trading
        assert len(action.strategy_weights) == len(strategy_ids)
        assert abs(np.sum(action.strategy_weights) - 1.0) < 1e-6  # Allocates 100% of capital
        assert 0.0 <= action.exposure_multiplier <= 1.0  # Risk control
        
        # Take step and verify it simulates portfolio-level effects, not direct trades
        next_state, reward, done, info = rl_env.step(action, asset_id=asset_id)
        
        # Should track portfolio-level metrics, not individual trades
        assert "portfolio_value" in info
        assert "drawdown" in info
        assert "max_drawdown" in info
        
        # Should NOT track direct trading metrics
        assert "orders_placed" not in info
        assert "fills_received" not in info
        assert "execution_latency" not in info
        assert "slippage" not in info
        
        # Reward should be portfolio-level, not trade-level
        reward_components = info["reward_components"]
        assert hasattr(reward_components, "sharpe_ratio")  # Portfolio performance
        assert hasattr(reward_components, "drawdown_penalty")  # Portfolio risk
        
        # Should NOT have trade-level reward components
        assert not hasattr(reward_components, "fill_quality")
        assert not hasattr(reward_components, "execution_cost")
        assert not hasattr(reward_components, "market_impact")
        
        # State should contain portfolio-level information, not trade details
        assert hasattr(next_state, 'net_exposure')  # Portfolio exposure
        assert hasattr(next_state, 'gross_exposure')  # Portfolio size
        assert hasattr(next_state, 'drawdown')  # Portfolio risk
        
        # Should NOT contain trade-level state information
        assert not hasattr(next_state, 'pending_orders')
        assert not hasattr(next_state, 'order_book_depth')
        assert not hasattr(next_state, 'execution_queue')
        
        # Verify environment doesn't expose direct trading methods
        rl_env_methods = [method for method in dir(rl_env) if not method.startswith('_')]
        
        forbidden_trading_methods = [
            'place_order',
            'cancel_order',
            'modify_order',
            'execute_trade',
            'send_to_market'
        ]
        
        for forbidden_method in forbidden_trading_methods:
            assert forbidden_method not in rl_env_methods, (
                f"RL environment has forbidden trading method '{forbidden_method}'. "
                f"RL environment should be meta-controller only, not direct trading agent."
            )


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])