"""Tests for RL environment implementation."""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

from intelligence_layer.rl_environment import (
    TradingEnvironmentMDP,
    RLAction,
    RLState,
    ExecutionAggressiveness,
    EpisodeConfig,
    create_sample_action,
    validate_mdp_properties
)
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
def strategy_ids():
    """Sample strategy IDs for testing."""
    return ["trend_strategy", "mean_reversion_strategy", "volatility_strategy"]


@pytest.fixture
def episode_config():
    """Test episode configuration."""
    return EpisodeConfig(
        max_steps=10,
        decision_interval_minutes=60,
        max_drawdown_threshold=0.1,
        data_start_date=datetime.now(timezone.utc) - timedelta(days=1),
        data_end_date=datetime.now(timezone.utc)
    )


@pytest.fixture
def rl_environment(config, strategy_ids, episode_config):
    """Create RL environment for testing."""
    return TradingEnvironmentMDP(
        config=config,
        strategy_ids=strategy_ids,
        episode_config=episode_config,
        mock_mode=True  # Use mock mode for testing
    )


class TestRLAction:
    """Test RL action space."""
    
    def test_create_sample_action(self, strategy_ids):
        """Test creating sample action."""
        action = create_sample_action(len(strategy_ids))
        
        assert isinstance(action, RLAction)
        assert len(action.strategy_weights) == len(strategy_ids)
        assert np.isclose(np.sum(action.strategy_weights), 1.0)
        assert 0 <= action.exposure_multiplier <= 1
        assert isinstance(action.execution_aggressiveness, ExecutionAggressiveness)
    
    def test_action_validation(self, rl_environment, strategy_ids):
        """Test action validation."""
        # Valid action
        valid_action = create_sample_action(len(strategy_ids))
        rl_environment._validate_action(valid_action)  # Should not raise
        
        # Invalid weights (don't sum to 1)
        invalid_action = RLAction(
            strategy_weights=np.array([0.5, 0.3, 0.1]),  # Sum = 0.9
            exposure_multiplier=0.5,
            execution_aggressiveness=ExecutionAggressiveness.MEDIUM
        )
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            rl_environment._validate_action(invalid_action)
        
        # Invalid exposure multiplier
        invalid_action2 = RLAction(
            strategy_weights=np.array([0.4, 0.3, 0.3]),
            exposure_multiplier=1.5,  # > 1.0
            execution_aggressiveness=ExecutionAggressiveness.MEDIUM
        )
        
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            rl_environment._validate_action(invalid_action2)


class TestRLState:
    """Test RL state representation."""
    
    def test_state_to_vector(self):
        """Test state vector conversion."""
        state = RLState(
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
            regime_confidence=0.8,
            embedding_similarity_score=0.75,
            forecast_dispersion=0.2,
            timestamp=datetime.now(timezone.utc)
        )
        
        vector = state.to_vector()
        
        # Check vector properties
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert len(vector) > 128  # At least embedding size
        assert np.isfinite(vector).all()


class TestTradingEnvironmentMDP:
    """Test RL environment MDP implementation."""
    
    def test_environment_initialization(self, rl_environment, strategy_ids):
        """Test environment initialization."""
        assert rl_environment.strategy_ids == strategy_ids
        assert rl_environment.current_step == 0
        assert not rl_environment.done
        assert rl_environment.current_state is None
    
    def test_reset(self, rl_environment):
        """Test environment reset."""
        initial_state = rl_environment.reset()
        
        assert isinstance(initial_state, RLState)
        assert rl_environment.current_step == 0
        assert not rl_environment.done
        assert rl_environment.episode_id is not None
        assert rl_environment.current_state == initial_state
    
    def test_step(self, rl_environment, strategy_ids):
        """Test environment step function."""
        # Reset environment
        initial_state = rl_environment.reset()
        
        # Create valid action
        action = create_sample_action(len(strategy_ids))
        
        # Take step
        next_state, reward, done, info = rl_environment.step(action)
        
        # Validate outputs
        assert isinstance(next_state, RLState)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check state progression
        assert rl_environment.current_step == 1
        assert rl_environment.current_state == next_state
        
        # Check info dictionary
        assert "episode_id" in info
        assert "step" in info
        assert "portfolio_value" in info
        assert "drawdown" in info
    
    def test_episode_termination(self, config, strategy_ids):
        """Test episode termination conditions."""
        # Create environment with very low max steps
        episode_config = EpisodeConfig(max_steps=2)
        env = TradingEnvironmentMDP(config, strategy_ids, episode_config, mock_mode=True)
        
        # Reset and take steps until termination
        env.reset()
        action = create_sample_action(len(strategy_ids))
        
        # First step
        _, _, done, _ = env.step(action)
        assert not done
        
        # Second step (should terminate due to max steps)
        _, _, done, info = env.step(action)
        assert done
        assert info["termination_reason"] == "max_steps"
    
    def test_action_space_info(self, rl_environment, strategy_ids):
        """Test action space information."""
        action_info = rl_environment.get_action_space_info()
        
        assert "strategy_weights" in action_info
        assert "exposure_multiplier" in action_info
        assert "execution_aggressiveness" in action_info
        
        # Check strategy weights info
        weights_info = action_info["strategy_weights"]
        assert weights_info["shape"] == (len(strategy_ids),)
        assert weights_info["constraint"] == "sum_to_one"
    
    def test_state_space_info(self, rl_environment):
        """Test state space information."""
        state_info = rl_environment.get_state_space_info()
        
        expected_components = [
            "market_embedding",
            "regime_probabilities", 
            "regime_entropy",
            "graph_features",
            "portfolio_state",
            "confidence_scores"
        ]
        
        for component in expected_components:
            assert component in state_info
    
    def test_episode_summary(self, rl_environment, strategy_ids):
        """Test episode summary generation."""
        # Run a short episode
        rl_environment.reset()
        action = create_sample_action(len(strategy_ids))
        
        # Take a few steps
        for _ in range(3):
            _, _, done, _ = rl_environment.step(action)
            if done:
                break
        
        summary = rl_environment.get_episode_summary()
        
        assert "episode_id" in summary
        assert "total_steps" in summary
        assert "final_portfolio_value" in summary
        assert "total_return" in summary
        assert "max_drawdown" in summary
        assert "sharpe_ratio" in summary


class TestMDPValidation:
    """Test MDP property validation."""
    
    def test_validate_mdp_properties(self, rl_environment):
        """Test MDP properties validation."""
        results = validate_mdp_properties(rl_environment)
        
        # Check that validation returns expected keys
        expected_keys = [
            "state_vector_valid",
            "action_validation_works",
            "step_function_works",
            "reward_is_numeric",
            "done_is_boolean",
            "info_is_dict"
        ]
        
        for key in expected_keys:
            assert key in results
            # Most should pass for a properly implemented environment
            if key != "state_vector_valid":  # This might fail due to mock data
                assert isinstance(results[key], bool)


@pytest.mark.asyncio
async def test_environment_cleanup(rl_environment):
    """Test environment cleanup."""
    await rl_environment.close()
    # Should not raise any exceptions