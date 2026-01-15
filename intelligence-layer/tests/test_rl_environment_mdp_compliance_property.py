"""Property-based tests for RL environment MDP compliance."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4
import asyncio

from intelligence_layer.rl_environment import (
    TradingEnvironmentMDP,
    RLAction,
    RLState,
    ExecutionAggressiveness,
    EpisodeConfig,
    RewardComponents,
    create_sample_action
)
from intelligence_layer.config import Config, DatabaseConfig, RedisConfig, LoggingConfig


# Strategies for generating test data
@st.composite
def strategy_ids_strategy(draw):
    """Generate list of strategy IDs."""
    n_strategies = draw(st.integers(min_value=1, max_value=5))
    return [f"strategy_{i}" for i in range(n_strategies)]


@st.composite
def episode_config_strategy(draw):
    """Generate episode configurations."""
    return EpisodeConfig(
        max_steps=draw(st.integers(min_value=5, max_value=50)),
        decision_interval_minutes=draw(st.integers(min_value=15, max_value=240)),
        max_drawdown_threshold=draw(st.floats(min_value=0.05, max_value=0.3)),
        regime_stability_threshold=draw(st.floats(min_value=0.3, max_value=0.9)),
        data_start_date=datetime.now(timezone.utc) - timedelta(days=30),
        data_end_date=datetime.now(timezone.utc)
    )


@st.composite
def rl_action_strategy(draw, n_strategies):
    """Generate valid RL actions."""
    # Generate strategy weights that sum to 1.0
    raw_weights = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(n_strategies)]
    total = sum(raw_weights)
    normalized_weights = np.array([w / total for w in raw_weights])
    
    return RLAction(
        strategy_weights=normalized_weights,
        exposure_multiplier=draw(st.floats(min_value=0.0, max_value=1.0)),
        execution_aggressiveness=draw(st.sampled_from(list(ExecutionAggressiveness)))
    )


@st.composite
def asset_id_strategy(draw):
    """Generate valid asset IDs."""
    return draw(st.sampled_from(["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]))


class TestRLEnvironmentMDPComplianceProperties:
    """Property-based tests for RL environment MDP compliance.
    
    **Validates: Requirements 12.1-12.6**
    """
    
    @given(
        strategy_ids_strategy(),
        episode_config_strategy(),
        asset_id_strategy()
    )
    @settings(max_examples=10, deadline=15000)
    def test_property_episodic_mdp_structure(self, strategy_ids, episode_config, asset_id):
        """
        Property: For any RL environment episode, the environment should implement
        an episodic, partially observable MDP with strategy decision intervals.
        
        **Feature: algorithmic-trading-system, Property 7: RL Environment MDP Compliance**
        **Validates: Requirements 12.1**
        """
        # Mock configuration
        config = Config(
            database=DatabaseConfig(
                postgres_url="postgresql://test:test@localhost:5432/test",
                neo4j_url="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test"
            ),
            redis=RedisConfig(url="redis://localhost:6379"),
            logging=LoggingConfig(level="INFO")
        )
        
        # Create environment
        env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=episode_config,
            mock_mode=True
        )
        
        # **Requirement 12.1**: Episodic MDP structure
        
        # Test episode initialization
        initial_state = env.reset(asset_id=asset_id)
        
        # Verify episodic structure
        assert env.episode_id is not None
        assert env.current_step == 0
        assert not env.done
        assert env.episode_start_time is not None
        assert env.current_timestamp is not None
        
        # Verify partially observable MDP properties
        assert isinstance(initial_state, RLState)
        assert initial_state.timestamp is not None
        
        # Test strategy decision intervals
        initial_timestamp = env.current_timestamp
        
        # Take a step
        action = create_sample_action(len(strategy_ids))
        next_state, reward, done, info = env.step(action, asset_id=asset_id)
        
        # Verify time advancement matches decision interval
        expected_next_time = initial_timestamp + timedelta(minutes=episode_config.decision_interval_minutes)
        assert env.current_timestamp == expected_next_time
        
        # Verify MDP step structure
        assert isinstance(next_state, RLState)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Verify episode tracking
        assert env.current_step == 1
        assert len(env.episode_history) == 1
        
        # Test episode termination
        if done:
            # If episode terminated, verify termination reason is valid
            assert info["termination_reason"] in [
                "max_steps", "max_drawdown", "regime_instability", "data_exhaustion"
            ]
        
        # Test episode can be reset
        new_initial_state = env.reset(asset_id=asset_id)
        assert env.current_step == 0
        assert not env.done
        assert env.episode_id != initial_state.timestamp  # New episode ID
    
    @given(
        strategy_ids_strategy(),
        episode_config_strategy(),
        asset_id_strategy()
    )
    @settings(max_examples=8, deadline=12000)
    def test_property_composite_state_vectors(self, strategy_ids, episode_config, asset_id):
        """
        Property: For any RL environment state, the state should combine latent market embeddings,
        discrete regime representations, graph structural features, portfolio state, and confidence metrics.
        
        **Feature: algorithmic-trading-system, Property 7: RL Environment MDP Compliance**
        **Validates: Requirements 12.2**
        """
        config = Config(
            database=DatabaseConfig(
                postgres_url="postgresql://test:test@localhost:5432/test",
                neo4j_url="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test"
            ),
            redis=RedisConfig(url="redis://localhost:6379"),
            logging=LoggingConfig(level="INFO")
        )
        
        env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=episode_config,
            mock_mode=True
        )
        
        # Reset environment and get initial state
        state = env.reset(asset_id=asset_id)
        
        # **Requirement 12.2**: Composite state vectors
        
        # Verify latent market embeddings (128-dimensional)
        assert hasattr(state, 'market_embedding')
        assert isinstance(state.market_embedding, np.ndarray)
        assert state.market_embedding.shape == (128,)
        assert np.isfinite(state.market_embedding).all()
        
        # Verify discrete regime representations
        assert hasattr(state, 'regime_id')
        assert hasattr(state, 'regime_probabilities')
        assert hasattr(state, 'regime_entropy')
        
        assert isinstance(state.regime_id, str)
        assert isinstance(state.regime_probabilities, dict)
        assert isinstance(state.regime_entropy, (int, float))
        
        # Regime probabilities should sum to approximately 1.0
        if state.regime_probabilities:
            prob_sum = sum(state.regime_probabilities.values())
            assert abs(prob_sum - 1.0) < 1e-6
            
            # All probabilities should be non-negative
            for prob in state.regime_probabilities.values():
                assert prob >= 0.0
        
        # Verify graph structural features
        assert hasattr(state, 'asset_cluster_id')
        assert hasattr(state, 'cluster_density')
        assert hasattr(state, 'centrality_score')
        assert hasattr(state, 'systemic_risk_proxy')
        
        # Graph features can be None (missing data) or valid floats
        if state.cluster_density is not None:
            assert isinstance(state.cluster_density, (int, float))
            assert 0.0 <= state.cluster_density <= 1.0
        
        if state.centrality_score is not None:
            assert isinstance(state.centrality_score, (int, float))
            assert 0.0 <= state.centrality_score <= 1.0
        
        if state.systemic_risk_proxy is not None:
            assert isinstance(state.systemic_risk_proxy, (int, float))
            assert 0.0 <= state.systemic_risk_proxy <= 1.0
        
        # Verify portfolio state
        assert hasattr(state, 'net_exposure')
        assert hasattr(state, 'gross_exposure')
        assert hasattr(state, 'drawdown')
        assert hasattr(state, 'volatility_target_utilization')
        
        assert isinstance(state.net_exposure, (int, float))
        assert isinstance(state.gross_exposure, (int, float))
        assert isinstance(state.drawdown, (int, float))
        assert isinstance(state.volatility_target_utilization, (int, float))
        
        # Portfolio constraints
        assert state.drawdown >= 0.0
        assert state.gross_exposure >= 0.0
        assert abs(state.net_exposure) <= state.gross_exposure
        
        # Verify confidence metrics
        assert hasattr(state, 'regime_confidence')
        assert hasattr(state, 'embedding_similarity_score')
        assert hasattr(state, 'forecast_dispersion')
        
        assert isinstance(state.regime_confidence, (int, float))
        assert isinstance(state.embedding_similarity_score, (int, float))
        assert isinstance(state.forecast_dispersion, (int, float))
        
        # Confidence scores should be in [0, 1]
        assert 0.0 <= state.regime_confidence <= 1.0
        assert 0.0 <= state.embedding_similarity_score <= 1.0
        assert 0.0 <= state.forecast_dispersion <= 1.0
        
        # Test state vector conversion
        state_vector = state.to_vector()
        assert isinstance(state_vector, np.ndarray)
        assert state_vector.dtype == np.float32
        assert len(state_vector) > 128  # At least embedding size plus other features
        assert np.isfinite(state_vector).all()
    
    @given(
        strategy_ids_strategy(),
        episode_config_strategy(),
        asset_id_strategy()
    )
    @settings(max_examples=8, deadline=12000)
    def test_property_continuous_bounded_action_space(self, strategy_ids, episode_config, asset_id):
        """
        Property: For any RL environment action, the action space should be continuous
        and bounded for strategy weights, exposure multipliers, and execution aggressiveness.
        
        **Feature: algorithmic-trading-system, Property 7: RL Environment MDP Compliance**
        **Validates: Requirements 12.3**
        """
        config = Config(
            database=DatabaseConfig(
                postgres_url="postgresql://test:test@localhost:5432/test",
                neo4j_url="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test"
            ),
            redis=RedisConfig(url="redis://localhost:6379"),
            logging=LoggingConfig(level="INFO")
        )
        
        env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=episode_config,
            mock_mode=True
        )
        
        # **Requirement 12.3**: Continuous bounded action spaces
        
        # Get action space information
        action_info = env.get_action_space_info()
        
        # Verify strategy weights action space
        weights_info = action_info["strategy_weights"]
        assert weights_info["type"] == "continuous"
        assert weights_info["shape"] == (len(strategy_ids),)
        assert weights_info["bounds"] == (0.0, 1.0)
        assert weights_info["constraint"] == "sum_to_one"
        
        # Verify exposure multiplier action space
        exposure_info = action_info["exposure_multiplier"]
        assert exposure_info["type"] == "continuous"
        assert exposure_info["shape"] == (1,)
        assert exposure_info["bounds"] == (0.0, 1.0)
        
        # Verify execution aggressiveness action space
        exec_info = action_info["execution_aggressiveness"]
        assert exec_info["type"] == "discrete"
        assert set(exec_info["choices"]) == {"low", "medium", "high"}
        
        # Test action validation with various valid actions
        env.reset(asset_id=asset_id)
        
        # Test valid action
        valid_action = create_sample_action(len(strategy_ids))
        env._validate_action(valid_action)  # Should not raise
        
        # Test boundary conditions
        # All weight on first strategy
        boundary_action1 = RLAction(
            strategy_weights=np.array([1.0] + [0.0] * (len(strategy_ids) - 1)),
            exposure_multiplier=0.0,  # Minimum exposure
            execution_aggressiveness=ExecutionAggressiveness.LOW
        )
        env._validate_action(boundary_action1)  # Should not raise
        
        # Maximum exposure
        boundary_action2 = RLAction(
            strategy_weights=np.array([1.0 / len(strategy_ids)] * len(strategy_ids)),
            exposure_multiplier=1.0,  # Maximum exposure
            execution_aggressiveness=ExecutionAggressiveness.HIGH
        )
        env._validate_action(boundary_action2)  # Should not raise
        
        # Test invalid actions
        # Weights don't sum to 1 (clearly outside tolerance)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            invalid_action1 = RLAction(
                strategy_weights=np.array([0.3] * len(strategy_ids)),  # Sum = 0.3 * n != 1.0
                exposure_multiplier=0.5,
                execution_aggressiveness=ExecutionAggressiveness.MEDIUM
            )
            env._validate_action(invalid_action1)
        
        # Negative weights
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            invalid_action2 = RLAction(
                strategy_weights=np.array([-0.1] + [1.1] + [0.0] * (len(strategy_ids) - 2)),
                exposure_multiplier=0.5,
                execution_aggressiveness=ExecutionAggressiveness.MEDIUM
            )
            env._validate_action(invalid_action2)
        
        # Invalid exposure multiplier
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            invalid_action3 = RLAction(
                strategy_weights=np.array([1.0 / len(strategy_ids)] * len(strategy_ids)),
                exposure_multiplier=1.5,  # > 1.0
                execution_aggressiveness=ExecutionAggressiveness.MEDIUM
            )
            env._validate_action(invalid_action3)
    
    @given(
        strategy_ids_strategy(),
        episode_config_strategy(),
        asset_id_strategy()
    )
    @settings(max_examples=6, deadline=10000)
    def test_property_risk_adjusted_reward_function(self, strategy_ids, episode_config, asset_id):
        """
        Property: For any RL environment step, the reward function should incorporate
        Sharpe ratio, drawdown penalties, turnover costs, and regime violation penalties.
        
        **Feature: algorithmic-trading-system, Property 7: RL Environment MDP Compliance**
        **Validates: Requirements 12.4**
        """
        config = Config(
            database=DatabaseConfig(
                postgres_url="postgresql://test:test@localhost:5432/test",
                neo4j_url="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test"
            ),
            redis=RedisConfig(url="redis://localhost:6379"),
            logging=LoggingConfig(level="INFO")
        )
        
        env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=episode_config,
            mock_mode=True
        )
        
        # **Requirement 12.4**: Risk-adjusted reward functions
        
        env.reset(asset_id=asset_id)
        
        # Take several steps to build up return history for Sharpe calculation
        actions = []
        rewards = []
        reward_components_list = []
        
        for i in range(min(5, episode_config.max_steps)):
            action = create_sample_action(len(strategy_ids))
            actions.append(action)
            
            next_state, reward, done, info = env.step(action, asset_id=asset_id)
            rewards.append(reward)
            
            # Verify reward components are present in info
            assert "reward_components" in info
            reward_components = info["reward_components"]
            reward_components_list.append(reward_components)
            
            # Verify reward_components is a RewardComponents object
            assert isinstance(reward_components, RewardComponents)
            
            # Verify all required reward components are present
            assert hasattr(reward_components, "sharpe_ratio")
            assert hasattr(reward_components, "drawdown_penalty")
            assert hasattr(reward_components, "turnover_cost")
            assert hasattr(reward_components, "regime_violation_penalty")
            
            # Verify component types
            assert isinstance(reward_components.sharpe_ratio, (int, float))
            assert isinstance(reward_components.drawdown_penalty, (int, float))
            assert isinstance(reward_components.turnover_cost, (int, float))
            assert isinstance(reward_components.regime_violation_penalty, (int, float))
            
            # Verify reward is numeric
            assert isinstance(reward, (int, float))
            assert np.isfinite(reward)
            
            if done:
                break
        
        # Verify reward components behave as expected
        if len(reward_components_list) > 1:
            # Drawdown penalty should be non-positive (penalty)
            for components in reward_components_list:
                assert components.drawdown_penalty <= 0.0
            
            # Turnover cost should be non-positive (cost)
            for i, components in enumerate(reward_components_list):
                if i > 0:  # First step has no previous action
                    assert components.turnover_cost <= 0.0
            
            # Regime violation penalty should be non-positive (penalty)
            for components in reward_components_list:
                assert components.regime_violation_penalty <= 0.0
        
        # Test reward calculation directly
        if len(env.episode_returns) >= 2:
            # Create mock state with low regime confidence to trigger violation penalty
            mock_state = RLState(
                market_embedding=np.random.normal(0, 1, 128),
                regime_id="test_regime",
                regime_probabilities={"regime1": 0.6, "regime2": 0.4},
                regime_entropy=0.8,
                asset_cluster_id="cluster1",
                cluster_density=0.7,
                centrality_score=0.5,
                systemic_risk_proxy=0.3,
                net_exposure=0.1,
                gross_exposure=0.2,
                drawdown=0.05,
                volatility_target_utilization=0.6,
                regime_confidence=0.3,  # Low confidence to trigger penalty
                embedding_similarity_score=0.75,
                forecast_dispersion=0.2,
                timestamp=datetime.now(timezone.utc)
            )
            
            action = create_sample_action(len(strategy_ids))
            action.exposure_multiplier = 0.8  # High exposure during uncertain regime
            
            reward_components = env._calculate_reward(action, mock_state)
            
            # Verify regime violation penalty is triggered
            assert reward_components.regime_violation_penalty < 0.0
            
            # Verify total reward is sum of components
            expected_total = (
                env.reward_params["alpha"] * reward_components.sharpe_ratio +
                env.reward_params["beta"] * reward_components.drawdown_penalty +
                env.reward_params["gamma"] * reward_components.turnover_cost +
                env.reward_params["delta"] * reward_components.regime_violation_penalty
            )
            assert abs(reward_components.total_reward - expected_total) < 1e-6
    
    @given(
        strategy_ids_strategy(),
        episode_config_strategy(),
        asset_id_strategy()
    )
    @settings(max_examples=6, deadline=10000)
    def test_property_episode_termination_conditions(self, strategy_ids, episode_config, asset_id):
        """
        Property: For any RL environment episode, termination should occur based on
        drawdown breaches, regime shifts, or data exhaustion.
        
        **Feature: algorithmic-trading-system, Property 7: RL Environment MDP Compliance**
        **Validates: Requirements 12.5**
        """
        config = Config(
            database=DatabaseConfig(
                postgres_url="postgresql://test:test@localhost:5432/test",
                neo4j_url="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test"
            ),
            redis=RedisConfig(url="redis://localhost:6379"),
            logging=LoggingConfig(level="INFO")
        )
        
        # **Requirement 12.5**: Episode termination conditions
        
        # Test max steps termination
        short_episode_config = EpisodeConfig(
            max_steps=3,
            decision_interval_minutes=60,
            max_drawdown_threshold=0.5,  # High threshold to avoid drawdown termination
            regime_stability_threshold=0.1,  # Low threshold to avoid regime termination
        )
        
        env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=short_episode_config,
            mock_mode=True
        )
        
        env.reset(asset_id=asset_id)
        action = create_sample_action(len(strategy_ids))
        
        # Take steps until termination
        step_count = 0
        done = False
        termination_reason = None
        
        while not done and step_count < 10:  # Safety limit
            _, _, done, info = env.step(action, asset_id=asset_id)
            step_count += 1
            
            if done:
                termination_reason = info.get("termination_reason")
                break
        
        # Should terminate due to max steps
        assert done
        assert termination_reason == "max_steps"
        assert step_count == short_episode_config.max_steps
        
        # Test drawdown termination
        drawdown_episode_config = EpisodeConfig(
            max_steps=100,  # High limit
            decision_interval_minutes=60,
            max_drawdown_threshold=0.01,  # Very low threshold
            regime_stability_threshold=0.1,  # Low threshold
        )
        
        env2 = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=drawdown_episode_config,
            mock_mode=True
        )
        
        env2.reset(asset_id=asset_id)
        
        # Force high drawdown by manipulating portfolio history
        env2.portfolio_value_history = [100000.0, 99500.0, 99000.0]  # 1% drawdown
        env2.max_drawdown = 0.01
        env2.episode_drawdown = 0.01
        
        # Check termination
        termination_check = env2._check_termination(env2.current_state)
        assert termination_check  # Should terminate due to drawdown
        
        termination_reason = env2._get_termination_reason(env2.current_state)
        assert termination_reason == "max_drawdown"
        
        # Test regime instability termination
        regime_episode_config = EpisodeConfig(
            max_steps=100,
            decision_interval_minutes=60,
            max_drawdown_threshold=0.5,  # High threshold
            regime_stability_threshold=0.9,  # High threshold
        )
        
        env3 = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=regime_episode_config,
            mock_mode=True
        )
        
        env3.reset(asset_id=asset_id)
        env3.current_step = 15  # Past warmup period
        
        # Create state with low regime confidence
        low_confidence_state = RLState(
            market_embedding=np.random.normal(0, 1, 128),
            regime_id="unstable_regime",
            regime_probabilities={"regime1": 0.5, "regime2": 0.5},
            regime_entropy=1.0,  # High entropy
            asset_cluster_id="cluster1",
            cluster_density=0.7,
            centrality_score=0.5,
            systemic_risk_proxy=0.3,
            net_exposure=0.1,
            gross_exposure=0.2,
            drawdown=0.05,
            volatility_target_utilization=0.6,
            regime_confidence=0.5,  # Below threshold
            embedding_similarity_score=0.75,
            forecast_dispersion=0.2,
            timestamp=datetime.now(timezone.utc)
        )
        
        termination_check = env3._check_termination(low_confidence_state)
        assert termination_check  # Should terminate due to regime instability
        
        termination_reason = env3._get_termination_reason(low_confidence_state)
        assert termination_reason == "regime_instability"
        
        # Test data exhaustion termination
        data_episode_config = EpisodeConfig(
            max_steps=100,
            decision_interval_minutes=60,
            max_drawdown_threshold=0.5,
            regime_stability_threshold=0.1,
            data_start_date=datetime.now(timezone.utc) - timedelta(hours=2),
            data_end_date=datetime.now(timezone.utc) - timedelta(hours=1)  # Past end date
        )
        
        env4 = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=data_episode_config,
            mock_mode=True
        )
        
        env4.reset(asset_id=asset_id)
        env4.current_timestamp = datetime.now(timezone.utc)  # Past end date
        
        termination_check = env4._check_termination(env4.current_state)
        assert termination_check  # Should terminate due to data exhaustion
        
        termination_reason = env4._get_termination_reason(env4.current_state)
        assert termination_reason == "data_exhaustion"
    
    @given(
        strategy_ids_strategy(),
        episode_config_strategy(),
        asset_id_strategy()
    )
    @settings(max_examples=6, deadline=10000)
    def test_property_meta_policy_controller_role(self, strategy_ids, episode_config, asset_id):
        """
        Property: For any RL environment, it should serve as a meta-policy controller
        for capital allocation across strategies, NOT as a direct trading agent.
        
        **Feature: algorithmic-trading-system, Property 7: RL Environment MDP Compliance**
        **Validates: Requirements 12.6**
        """
        config = Config(
            database=DatabaseConfig(
                postgres_url="postgresql://test:test@localhost:5432/test",
                neo4j_url="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="test"
            ),
            redis=RedisConfig(url="redis://localhost:6379"),
            logging=LoggingConfig(level="INFO")
        )
        
        env = TradingEnvironmentMDP(
            config=config,
            strategy_ids=strategy_ids,
            episode_config=episode_config,
            mock_mode=True
        )
        
        # **Requirement 12.6**: Meta-policy controller role
        
        # Verify environment is configured for strategy allocation
        assert env.strategy_ids == strategy_ids
        assert len(env.strategy_ids) > 0
        
        # Get action space info
        action_info = env.get_action_space_info()
        
        # Verify action space is designed for strategy allocation
        weights_info = action_info["strategy_weights"]
        assert weights_info["shape"] == (len(strategy_ids),)
        assert weights_info["constraint"] == "sum_to_one"  # Capital allocation constraint
        
        # Verify exposure multiplier for risk control
        exposure_info = action_info["exposure_multiplier"]
        assert exposure_info["bounds"] == (0.0, 1.0)  # Risk throttle
        
        # Test that actions represent strategy allocation decisions
        env.reset(asset_id=asset_id)
        action = create_sample_action(len(strategy_ids))
        
        # Verify action represents capital allocation
        assert len(action.strategy_weights) == len(strategy_ids)
        assert abs(np.sum(action.strategy_weights) - 1.0) < 1e-6  # Allocates 100% of capital
        assert 0.0 <= action.exposure_multiplier <= 1.0  # Risk control
        
        # Take step and verify it simulates portfolio-level effects
        next_state, reward, done, info = env.step(action, asset_id=asset_id)
        
        # Verify step info contains portfolio-level metrics
        assert "portfolio_value" in info
        assert "drawdown" in info
        assert "max_drawdown" in info
        
        # Verify reward components are portfolio-level
        reward_components = info["reward_components"]
        assert isinstance(reward_components, RewardComponents)
        assert hasattr(reward_components, "sharpe_ratio")  # Portfolio performance
        assert hasattr(reward_components, "drawdown_penalty")  # Portfolio risk
        assert hasattr(reward_components, "turnover_cost")  # Portfolio turnover
        
        # Verify state contains portfolio-level information
        assert hasattr(next_state, 'net_exposure')  # Portfolio exposure
        assert hasattr(next_state, 'gross_exposure')  # Portfolio size
        assert hasattr(next_state, 'drawdown')  # Portfolio risk
        assert hasattr(next_state, 'volatility_target_utilization')  # Risk budget
        
        # Verify environment tracks portfolio value history
        assert len(env.portfolio_value_history) > 1
        assert all(isinstance(val, (int, float)) for val in env.portfolio_value_history)
        
        # Verify episode summary contains portfolio-level metrics
        summary = env.get_episode_summary()
        assert "final_portfolio_value" in summary
        assert "total_return" in summary
        assert "max_drawdown" in summary
        assert "sharpe_ratio" in summary
        assert "volatility" in summary
        
        # Verify no direct trading capabilities are exposed
        # Environment should not have methods for direct order placement
        assert not hasattr(env, 'place_order')
        assert not hasattr(env, 'cancel_order')
        assert not hasattr(env, 'modify_order')
        
        # Verify simulation-based portfolio updates
        initial_portfolio_value = env.portfolio_value_history[0]
        final_portfolio_value = env.portfolio_value_history[-1]
        
        # Portfolio value should change based on simulated returns
        if len(env.episode_returns) > 0:
            expected_final_value = initial_portfolio_value
            for ret in env.episode_returns:
                expected_final_value *= (1 + ret)
            
            # Allow for small numerical differences
            assert abs(final_portfolio_value - expected_final_value) < 1e-6


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])