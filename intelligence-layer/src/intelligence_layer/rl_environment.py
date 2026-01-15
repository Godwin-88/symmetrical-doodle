"""Reinforcement Learning Environment for Strategy Orchestration.

This module implements a formal MDP environment for meta-policy control
of strategy allocation and risk management. The environment serves as a
meta-controller for capital allocation across strategies, NOT as a direct
trading agent.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from uuid import uuid4

from .models import IntelligenceState, MarketData
from .state_assembly import CompositeStateAssembler
from .config import Config

logger = logging.getLogger(__name__)


class ExecutionAggressiveness(Enum):
    """Execution aggressiveness levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RLAction:
    """RL action space definition."""
    strategy_weights: np.ndarray  # Strategy weights (sum to 1)
    exposure_multiplier: float    # Global risk throttle [0, 1]
    execution_aggressiveness: ExecutionAggressiveness


@dataclass
class RLState:
    """Composite RL state representation."""
    # Latent market context (128-dim embedding)
    market_embedding: np.ndarray
    
    # Regime state
    regime_id: str
    regime_probabilities: Dict[str, float]
    regime_entropy: float
    
    # Graph structural features
    asset_cluster_id: Optional[str]
    cluster_density: Optional[float]
    centrality_score: Optional[float]
    systemic_risk_proxy: Optional[float]
    
    # Portfolio & risk state
    net_exposure: float
    gross_exposure: float
    drawdown: float
    volatility_target_utilization: float
    
    # Confidence & uncertainty
    regime_confidence: float
    embedding_similarity_score: float
    forecast_dispersion: float
    
    # Metadata
    timestamp: datetime
    
    def to_vector(self) -> np.ndarray:
        """Convert state to vector representation for RL algorithms."""
        # Market embedding (128 dims)
        state_vector = list(self.market_embedding)
        
        # Regime probabilities (assume max 5 regimes)
        regime_probs = [0.0] * 5
        for i, (regime, prob) in enumerate(self.regime_probabilities.items()):
            if i < 5:
                regime_probs[i] = prob
        state_vector.extend(regime_probs)
        
        # Regime entropy
        state_vector.append(self.regime_entropy)
        
        # Graph features
        state_vector.extend([
            self.cluster_density or 0.0,
            self.centrality_score or 0.0,
            self.systemic_risk_proxy or 0.0
        ])
        
        # Portfolio state
        state_vector.extend([
            self.net_exposure,
            self.gross_exposure,
            self.drawdown,
            self.volatility_target_utilization
        ])
        
        # Confidence scores
        state_vector.extend([
            self.regime_confidence,
            self.embedding_similarity_score,
            self.forecast_dispersion
        ])
        
        return np.array(state_vector, dtype=np.float32)


@dataclass
class EpisodeConfig:
    """Configuration for RL episodes."""
    max_steps: int = 100
    decision_interval_minutes: int = 60  # Strategy decision interval
    max_drawdown_threshold: float = 0.15  # 15% max drawdown
    regime_stability_threshold: float = 0.8  # Regime confidence threshold
    data_start_date: datetime = None
    data_end_date: datetime = None


@dataclass
class RewardComponents:
    """Components of the reward function."""
    sharpe_ratio: float
    drawdown_penalty: float
    turnover_cost: float
    regime_violation_penalty: float
    total_reward: float


class TradingEnvironmentMDP:
    """
    Episodic MDP environment for strategy orchestration.
    
    This environment implements a formal MDP for meta-policy control
    of capital allocation across strategies. It serves as a research
    environment for academic evaluation of strategy allocation and
    risk management.
    """
    
    def __init__(
        self, 
        config: Config,
        strategy_ids: List[str],
        episode_config: Optional[EpisodeConfig] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the RL environment.
        
        Args:
            config: System configuration
            strategy_ids: List of available strategy identifiers
            episode_config: Episode configuration parameters
            mock_mode: If True, use mock state assembler for testing
        """
        self.config = config
        self.strategy_ids = strategy_ids
        self.episode_config = episode_config or EpisodeConfig()
        self.mock_mode = mock_mode
        
        # State assembler for composite states
        if mock_mode:
            self.state_assembler = None
        else:
            self.state_assembler = CompositeStateAssembler(config)
        
        # Episode state
        self.current_step = 0
        self.episode_id = None
        self.episode_start_time = None
        self.current_timestamp = None
        self.done = False
        
        # State tracking
        self.current_state: Optional[RLState] = None
        self.previous_action: Optional[RLAction] = None
        self.episode_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.episode_returns: List[float] = []
        self.episode_drawdown = 0.0
        self.max_drawdown = 0.0
        self.portfolio_value_history: List[float] = []
        
        # Reward function parameters
        self.reward_params = {
            "alpha": 1.0,  # Sharpe ratio weight
            "beta": 2.0,   # Drawdown penalty weight
            "gamma": 0.1,  # Turnover cost weight
            "delta": 1.5   # Regime violation penalty weight
        }
        
        logger.info(f"Initialized RL environment with {len(strategy_ids)} strategies")
    
    def reset(
        self, 
        asset_id: str = "EURUSD",
        start_timestamp: Optional[datetime] = None
    ) -> RLState:
        """
        Reset the environment for a new episode.
        
        Args:
            asset_id: Primary asset for the episode
            start_timestamp: Episode start time (defaults to config)
            
        Returns:
            Initial state of the episode
        """
        self.episode_id = str(uuid4())
        self.current_step = 0
        self.done = False
        
        # Set episode timing
        if start_timestamp:
            self.episode_start_time = start_timestamp
        elif self.episode_config.data_start_date:
            self.episode_start_time = self.episode_config.data_start_date
        else:
            self.episode_start_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        self.current_timestamp = self.episode_start_time
        
        # Reset tracking
        self.episode_history = []
        self.episode_returns = []
        self.episode_drawdown = 0.0
        self.max_drawdown = 0.0
        self.portfolio_value_history = [100000.0]  # Start with $100k
        self.previous_action = None
        
        # Get initial state
        self.current_state = self._get_current_state(asset_id)
        
        logger.info(f"Reset episode {self.episode_id} starting at {self.episode_start_time}")
        return self.current_state
    
    def step(
        self, 
        action: RLAction,
        asset_id: str = "EURUSD"
    ) -> Tuple[RLState, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            asset_id: Primary asset identifier
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode is already done. Call reset() to start new episode.")
        
        # Validate action
        self._validate_action(action)
        
        # Advance time
        self.current_step += 1
        self.current_timestamp += timedelta(
            minutes=self.episode_config.decision_interval_minutes
        )
        
        # Execute action (simulate portfolio changes)
        portfolio_return = self._simulate_portfolio_step(action, asset_id)
        
        # Update portfolio tracking
        new_portfolio_value = self.portfolio_value_history[-1] * (1 + portfolio_return)
        self.portfolio_value_history.append(new_portfolio_value)
        self.episode_returns.append(portfolio_return)
        
        # Update drawdown
        peak_value = max(self.portfolio_value_history)
        current_drawdown = (peak_value - new_portfolio_value) / peak_value
        self.episode_drawdown = current_drawdown
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Get next state
        next_state = self._get_current_state(asset_id)
        
        # Calculate reward
        reward_components = self._calculate_reward(action, next_state)
        reward = reward_components.total_reward
        
        # Check termination conditions
        self.done = self._check_termination(next_state)
        
        # Update state
        self.current_state = next_state
        self.previous_action = action
        
        # Record step in history
        step_info = {
            "step": self.current_step,
            "timestamp": self.current_timestamp.isoformat(),
            "action": {
                "strategy_weights": action.strategy_weights.tolist(),
                "exposure_multiplier": action.exposure_multiplier,
                "execution_aggressiveness": action.execution_aggressiveness.value
            },
            "reward_components": {
                "sharpe_ratio": reward_components.sharpe_ratio,
                "drawdown_penalty": reward_components.drawdown_penalty,
                "turnover_cost": reward_components.turnover_cost,
                "regime_violation_penalty": reward_components.regime_violation_penalty
            },
            "portfolio_value": new_portfolio_value,
            "drawdown": current_drawdown,
            "regime": next_state.regime_id
        }
        self.episode_history.append(step_info)
        
        # Info dictionary
        info = {
            "episode_id": self.episode_id,
            "step": self.current_step,
            "portfolio_value": new_portfolio_value,
            "drawdown": current_drawdown,
            "max_drawdown": self.max_drawdown,
            "regime": next_state.regime_id,
            "regime_confidence": next_state.regime_confidence,
            "reward_components": reward_components,
            "termination_reason": self._get_termination_reason(next_state) if self.done else None
        }
        
        logger.debug(f"Step {self.current_step}: reward={reward:.4f}, done={self.done}")
        return next_state, reward, self.done, info
    
    def _validate_action(self, action: RLAction) -> None:
        """Validate action constraints."""
        # Check strategy weights sum to 1
        weight_sum = np.sum(action.strategy_weights)
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"Strategy weights must sum to 1.0, got {weight_sum}")
        
        # Check weight bounds
        if np.any(action.strategy_weights < 0) or np.any(action.strategy_weights > 1):
            raise ValueError("Strategy weights must be in [0, 1]")
        
        # Check exposure multiplier bounds
        if action.exposure_multiplier < 0 or action.exposure_multiplier > 1:
            raise ValueError("Exposure multiplier must be in [0, 1]")
        
        # Check strategy count matches
        if len(action.strategy_weights) != len(self.strategy_ids):
            raise ValueError(
                f"Action has {len(action.strategy_weights)} weights but "
                f"{len(self.strategy_ids)} strategies"
            )
    
    def _get_current_state(self, asset_id: str) -> RLState:
        """Get current RL state from intelligence layer."""
        # For simulation, we'll create a mock state
        # In production, this would call the state assembler
        
        # Mock market embedding (128-dim)
        market_embedding = np.random.normal(0, 1, 128).astype(np.float32)
        
        # Mock regime state
        regime_probs = {
            "low_vol_trending": 0.3,
            "high_vol_trending": 0.2,
            "low_vol_ranging": 0.3,
            "high_vol_ranging": 0.2
        }
        regime_entropy = -sum(p * np.log(p) for p in regime_probs.values() if p > 0)
        
        return RLState(
            market_embedding=market_embedding,
            regime_id="low_vol_trending",
            regime_probabilities=regime_probs,
            regime_entropy=regime_entropy,
            asset_cluster_id="cluster_1",
            cluster_density=0.7,
            centrality_score=0.5,
            systemic_risk_proxy=0.3,
            net_exposure=0.0,
            gross_exposure=0.0,
            drawdown=self.episode_drawdown,
            volatility_target_utilization=0.5,
            regime_confidence=0.8,
            embedding_similarity_score=0.75,
            forecast_dispersion=0.2,
            timestamp=self.current_timestamp
        )
    
    def _simulate_portfolio_step(self, action: RLAction, asset_id: str) -> float:
        """
        Simulate portfolio performance for one step.
        
        Args:
            action: Action taken
            asset_id: Primary asset
            
        Returns:
            Portfolio return for this step
        """
        # Mock return simulation based on action
        # In production, this would interface with the execution core
        
        # Base return (random walk)
        base_return = np.random.normal(0.0001, 0.02)  # ~0.01% mean, 2% vol
        
        # Adjust for exposure
        portfolio_return = base_return * action.exposure_multiplier
        
        # Add strategy-specific effects
        strategy_effect = 0.0
        for i, weight in enumerate(action.strategy_weights):
            # Mock strategy returns
            strategy_return = np.random.normal(0.0002, 0.015)  # Slightly positive expected return
            strategy_effect += weight * strategy_return
        
        portfolio_return += strategy_effect * action.exposure_multiplier
        
        # Execution cost based on aggressiveness
        execution_cost = {
            ExecutionAggressiveness.LOW: 0.0001,
            ExecutionAggressiveness.MEDIUM: 0.0002,
            ExecutionAggressiveness.HIGH: 0.0005
        }[action.execution_aggressiveness]
        
        portfolio_return -= execution_cost
        
        return portfolio_return
    
    def _calculate_reward(self, action: RLAction, state: RLState) -> RewardComponents:
        """
        Calculate risk-adjusted reward function.
        
        Args:
            action: Action taken
            state: Current state
            
        Returns:
            Reward components breakdown
        """
        # Sharpe ratio component (rolling)
        if len(self.episode_returns) >= 10:
            returns_array = np.array(self.episode_returns[-10:])
            sharpe_ratio = (
                np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0.0
        
        # Drawdown penalty
        drawdown_penalty = -self.episode_drawdown ** 2
        
        # Turnover cost
        turnover_cost = 0.0
        if self.previous_action is not None:
            weight_changes = np.abs(action.strategy_weights - self.previous_action.strategy_weights)
            turnover_cost = -np.sum(weight_changes) * 0.001  # 0.1% cost per unit turnover
        
        # Regime violation penalty
        regime_violation_penalty = 0.0
        if state.regime_confidence < 0.5:
            # Penalize high exposure during uncertain regimes
            regime_violation_penalty = -action.exposure_multiplier * (0.5 - state.regime_confidence)
        
        # Total reward
        total_reward = (
            self.reward_params["alpha"] * sharpe_ratio +
            self.reward_params["beta"] * drawdown_penalty +
            self.reward_params["gamma"] * turnover_cost +
            self.reward_params["delta"] * regime_violation_penalty
        )
        
        return RewardComponents(
            sharpe_ratio=sharpe_ratio,
            drawdown_penalty=drawdown_penalty,
            turnover_cost=turnover_cost,
            regime_violation_penalty=regime_violation_penalty,
            total_reward=total_reward
        )
    
    def _check_termination(self, state: RLState) -> bool:
        """Check if episode should terminate."""
        # Max steps reached
        if self.current_step >= self.episode_config.max_steps:
            return True
        
        # Max drawdown breached
        if self.max_drawdown >= self.episode_config.max_drawdown_threshold:
            return True
        
        # Regime shift with low confidence
        if (state.regime_confidence < self.episode_config.regime_stability_threshold and
            self.current_step > 10):  # Allow some warmup
            return True
        
        # Data exhaustion
        if (self.episode_config.data_end_date and 
            self.current_timestamp >= self.episode_config.data_end_date):
            return True
        
        return False
    
    def _get_termination_reason(self, state: RLState) -> str:
        """Get reason for episode termination."""
        if self.current_step >= self.episode_config.max_steps:
            return "max_steps"
        elif self.max_drawdown >= self.episode_config.max_drawdown_threshold:
            return "max_drawdown"
        elif state.regime_confidence < self.episode_config.regime_stability_threshold:
            return "regime_instability"
        elif (self.episode_config.data_end_date and 
              self.current_timestamp >= self.episode_config.data_end_date):
            return "data_exhaustion"
        else:
            return "unknown"
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        return {
            "strategy_weights": {
                "type": "continuous",
                "shape": (len(self.strategy_ids),),
                "bounds": (0.0, 1.0),
                "constraint": "sum_to_one"
            },
            "exposure_multiplier": {
                "type": "continuous",
                "shape": (1,),
                "bounds": (0.0, 1.0)
            },
            "execution_aggressiveness": {
                "type": "discrete",
                "choices": [e.value for e in ExecutionAggressiveness]
            }
        }
    
    def get_state_space_info(self) -> Dict[str, Any]:
        """Get information about the state space."""
        return {
            "market_embedding": {
                "type": "continuous",
                "shape": (128,),
                "description": "Latent market state representation"
            },
            "regime_probabilities": {
                "type": "continuous", 
                "shape": (5,),
                "bounds": (0.0, 1.0),
                "description": "Regime probability distribution"
            },
            "regime_entropy": {
                "type": "continuous",
                "shape": (1,),
                "description": "Regime uncertainty measure"
            },
            "graph_features": {
                "type": "continuous",
                "shape": (3,),
                "description": "Graph structural features"
            },
            "portfolio_state": {
                "type": "continuous",
                "shape": (4,),
                "description": "Portfolio and risk metrics"
            },
            "confidence_scores": {
                "type": "continuous",
                "shape": (3,),
                "bounds": (0.0, 1.0),
                "description": "Confidence and uncertainty measures"
            }
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the completed episode."""
        if not self.done:
            logger.warning("Episode not complete, summary may be incomplete")
        
        returns_array = np.array(self.episode_returns)
        
        return {
            "episode_id": self.episode_id,
            "total_steps": self.current_step,
            "final_portfolio_value": self.portfolio_value_history[-1],
            "total_return": (self.portfolio_value_history[-1] / self.portfolio_value_history[0]) - 1,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": (
                np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
                if len(returns_array) > 1 else 0.0
            ),
            "volatility": np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0.0,
            "episode_duration": self.current_timestamp - self.episode_start_time,
            "termination_reason": self._get_termination_reason(self.current_state) if self.done else None
        }
    
    async def close(self) -> None:
        """Close environment and cleanup resources."""
        if self.state_assembler:
            await self.state_assembler.close()
        logger.info("RL environment closed")


def create_sample_action(strategy_count: int) -> RLAction:
    """Create a sample action for testing."""
    # Equal weight allocation
    weights = np.ones(strategy_count) / strategy_count
    
    return RLAction(
        strategy_weights=weights,
        exposure_multiplier=0.5,
        execution_aggressiveness=ExecutionAggressiveness.MEDIUM
    )


def validate_mdp_properties(env: TradingEnvironmentMDP) -> Dict[str, bool]:
    """
    Validate that the environment satisfies MDP properties.
    
    Args:
        env: Environment to validate
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    # Test state space consistency
    try:
        state = env.reset()
        state_vector = state.to_vector()
        results["state_vector_valid"] = len(state_vector) > 0 and np.isfinite(state_vector).all()
    except Exception as e:
        logger.error(f"State validation failed: {e}")
        results["state_vector_valid"] = False
    
    # Test action space validation
    try:
        action = create_sample_action(len(env.strategy_ids))
        env._validate_action(action)
        results["action_validation_works"] = True
    except Exception as e:
        logger.error(f"Action validation failed: {e}")
        results["action_validation_works"] = False
    
    # Test step function
    try:
        state = env.reset()
        action = create_sample_action(len(env.strategy_ids))
        next_state, reward, done, info = env.step(action)
        results["step_function_works"] = True
        results["reward_is_numeric"] = isinstance(reward, (int, float))
        results["done_is_boolean"] = isinstance(done, bool)
        results["info_is_dict"] = isinstance(info, dict)
    except Exception as e:
        logger.error(f"Step function failed: {e}")
        results["step_function_works"] = False
        results["reward_is_numeric"] = False
        results["done_is_boolean"] = False
        results["info_is_dict"] = False
    
    return results