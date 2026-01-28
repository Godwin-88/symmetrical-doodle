"""
Tests for Feature Flag Service.

This module tests the comprehensive feature flag service with A/B testing capabilities.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from nautilus_integration.services.feature_flag_service import (
    FeatureFlagService,
    FeatureFlagConfig,
    FeatureFlagStatus,
    RolloutStrategy,
    UserGroup,
    FeatureFlagEvaluationContext,
    FeatureFlagResult,
    ABTestConfig
)


@pytest.fixture
async def feature_flag_service():
    """Create a feature flag service for testing."""
    service = FeatureFlagService(
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379",
        cache_ttl=60,
        enable_audit_logging=True
    )
    
    # Mock Redis client
    service.redis_client = AsyncMock()
    
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.fixture
def sample_flag_config():
    """Create a sample feature flag configuration."""
    return FeatureFlagConfig(
        name="test_nautilus_integration",
        description="Test NautilusTrader integration feature",
        status=FeatureFlagStatus.TESTING,
        rollout_strategy=RolloutStrategy.PERCENTAGE,
        rollout_percentage=25.0,
        target_groups=[UserGroup.DEVELOPERS, UserGroup.BETA_TESTERS],
        config_data={
            "enable_backtesting": True,
            "enable_live_trading": False,
            "max_strategies": 10
        }
    )


@pytest.fixture
def sample_evaluation_context():
    """Create a sample evaluation context."""
    return FeatureFlagEvaluationContext(
        user_id="test_user_123",
        user_groups=[UserGroup.DEVELOPERS],
        session_id="session_456",
        metadata={"environment": "testing"}
    )


class TestFeatureFlagService:
    """Test cases for FeatureFlagService."""
    
    async def test_create_feature_flag(self, feature_flag_service, sample_flag_config):
        """Test creating a feature flag."""
        # Create flag
        created_flag = await feature_flag_service.create_feature_flag(
            sample_flag_config,
            user_id="admin_user"
        )
        
        # Verify flag was created
        assert created_flag.name == sample_flag_config.name
        assert created_flag.description == sample_flag_config.description
        assert created_flag.status == sample_flag_config.status
        assert created_flag.rollout_percentage == sample_flag_config.rollout_percentage
        assert created_flag.created_by == "admin_user"
        
        # Verify flag is in cache
        assert sample_flag_config.name in feature_flag_service._flag_cache
    
    async def test_create_duplicate_flag_fails(self, feature_flag_service, sample_flag_config):
        """Test that creating a duplicate flag fails."""
        # Create first flag
        await feature_flag_service.create_feature_flag(sample_flag_config)
        
        # Try to create duplicate
        with pytest.raises(ValueError, match="already exists"):
            await feature_flag_service.create_feature_flag(sample_flag_config)
    
    async def test_update_feature_flag(self, feature_flag_service, sample_flag_config):
        """Test updating a feature flag."""
        # Create flag
        await feature_flag_service.create_feature_flag(sample_flag_config)
        
        # Update flag
        updates = {
            "rollout_percentage": 50.0,
            "status": FeatureFlagStatus.ENABLED,
            "config_data": {"enable_live_trading": True}
        }
        
        updated_flag = await feature_flag_service.update_feature_flag(
            sample_flag_config.name,
            updates,
            user_id="admin_user"
        )
        
        # Verify updates
        assert updated_flag.rollout_percentage == 50.0
        assert updated_flag.status == FeatureFlagStatus.ENABLED
        assert updated_flag.config_data["enable_live_trading"] is True
        assert updated_flag.updated_by == "admin_user"
    
    async def test_delete_feature_flag(self, feature_flag_service, sample_flag_config):
        """Test deleting a feature flag."""
        # Create flag
        await feature_flag_service.create_feature_flag(sample_flag_config)
        
        # Verify flag exists
        assert sample_flag_config.name in feature_flag_service._flag_cache
        
        # Delete flag
        result = await feature_flag_service.delete_feature_flag(
            sample_flag_config.name,
            user_id="admin_user"
        )
        
        # Verify deletion
        assert result is True
        assert sample_flag_config.name not in feature_flag_service._flag_cache
    
    async def test_evaluate_disabled_flag(self, feature_flag_service, sample_evaluation_context):
        """Test evaluating a disabled flag."""
        # Create disabled flag
        flag_config = FeatureFlagConfig(
            name="disabled_flag",
            status=FeatureFlagStatus.DISABLED
        )
        await feature_flag_service.create_feature_flag(flag_config)
        
        # Evaluate flag
        result = await feature_flag_service.evaluate_flag("disabled_flag", sample_evaluation_context)
        
        # Verify result
        assert result.enabled is False
        assert result.reason == "Flag disabled"
        assert result.flag_name == "disabled_flag"
    
    async def test_evaluate_enabled_flag(self, feature_flag_service, sample_evaluation_context):
        """Test evaluating an enabled flag."""
        # Create enabled flag
        flag_config = FeatureFlagConfig(
            name="enabled_flag",
            status=FeatureFlagStatus.ENABLED,
            config_data={"feature_value": "test_value"}
        )
        await feature_flag_service.create_feature_flag(flag_config)
        
        # Evaluate flag
        result = await feature_flag_service.evaluate_flag("enabled_flag", sample_evaluation_context)
        
        # Verify result
        assert result.enabled is True
        assert result.reason == "Flag enabled"
        assert result.config["feature_value"] == "test_value"
    
    async def test_evaluate_percentage_rollout(self, feature_flag_service):
        """Test percentage rollout evaluation."""
        # Create flag with 50% rollout
        flag_config = FeatureFlagConfig(
            name="percentage_flag",
            status=FeatureFlagStatus.TESTING,
            rollout_strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=50.0
        )
        await feature_flag_service.create_feature_flag(flag_config)
        
        # Test multiple users to verify percentage distribution
        enabled_count = 0
        total_tests = 100
        
        for i in range(total_tests):
            context = FeatureFlagEvaluationContext(user_id=f"user_{i}")
            result = await feature_flag_service.evaluate_flag("percentage_flag", context)
            if result.enabled:
                enabled_count += 1
        
        # Should be approximately 50% (allow some variance)
        assert 40 <= enabled_count <= 60
    
    async def test_evaluate_user_group_targeting(self, feature_flag_service):
        """Test user group targeting."""
        # Create flag targeting developers
        flag_config = FeatureFlagConfig(
            name="group_flag",
            status=FeatureFlagStatus.TESTING,
            rollout_strategy=RolloutStrategy.USER_GROUP,
            target_groups=[UserGroup.DEVELOPERS]
        )
        await feature_flag_service.create_feature_flag(flag_config)
        
        # Test developer user
        dev_context = FeatureFlagEvaluationContext(
            user_id="dev_user",
            user_groups=[UserGroup.DEVELOPERS]
        )
        result = await feature_flag_service.evaluate_flag("group_flag", dev_context)
        assert result.enabled is True
        assert "User group match" in result.reason
        
        # Test non-developer user
        user_context = FeatureFlagEvaluationContext(
            user_id="regular_user",
            user_groups=[UserGroup.EXTERNAL]
        )
        result = await feature_flag_service.evaluate_flag("group_flag", user_context)
        assert result.enabled is False
        assert "No matching user groups" in result.reason
    
    async def test_evaluate_whitelist_targeting(self, feature_flag_service):
        """Test whitelist targeting."""
        # Create flag with whitelist
        flag_config = FeatureFlagConfig(
            name="whitelist_flag",
            status=FeatureFlagStatus.TESTING,
            rollout_strategy=RolloutStrategy.WHITELIST,
            whitelist_users=["user_1", "user_2", "user_3"]
        )
        await feature_flag_service.create_feature_flag(flag_config)
        
        # Test whitelisted user
        whitelist_context = FeatureFlagEvaluationContext(user_id="user_1")
        result = await feature_flag_service.evaluate_flag("whitelist_flag", whitelist_context)
        assert result.enabled is True
        assert result.reason == "User in whitelist"
        
        # Test non-whitelisted user
        other_context = FeatureFlagEvaluationContext(user_id="user_4")
        result = await feature_flag_service.evaluate_flag("whitelist_flag", other_context)
        assert result.enabled is False
        assert result.reason == "User not in whitelist"
    
    async def test_evaluate_nonexistent_flag(self, feature_flag_service, sample_evaluation_context):
        """Test evaluating a non-existent flag."""
        result = await feature_flag_service.evaluate_flag("nonexistent_flag", sample_evaluation_context)
        
        assert result.enabled is False
        assert result.reason == "Flag not found"
        assert result.flag_name == "nonexistent_flag"
    
    async def test_create_ab_test(self, feature_flag_service, sample_flag_config):
        """Test creating an A/B test."""
        # Create feature flag first
        flag = await feature_flag_service.create_feature_flag(sample_flag_config)
        
        # Create A/B test
        ab_test_config = ABTestConfig(
            feature_flag_id=flag.id,
            name="test_ab_experiment",
            description="Test A/B experiment for NautilusTrader",
            variant_a_config={"algorithm": "basic"},
            variant_b_config={"algorithm": "advanced"},
            traffic_split=0.3  # 30% get variant B
        )
        
        created_test = await feature_flag_service.create_ab_test(
            ab_test_config,
            user_id="researcher"
        )
        
        # Verify test was created
        assert created_test.name == ab_test_config.name
        assert created_test.feature_flag_id == flag.id
        assert created_test.traffic_split == 0.3
        assert created_test.created_by == "researcher"
    
    async def test_evaluate_ab_test_flag(self, feature_flag_service):
        """Test evaluating a flag with A/B test."""
        # Create flag with A/B test strategy
        flag_config = FeatureFlagConfig(
            name="ab_test_flag",
            status=FeatureFlagStatus.TESTING,
            rollout_strategy=RolloutStrategy.AB_TEST
        )
        flag = await feature_flag_service.create_feature_flag(flag_config)
        
        # Create A/B test
        ab_test_config = ABTestConfig(
            feature_flag_id=flag.id,
            name="algorithm_test",
            variant_a_config={"algorithm": "basic", "threshold": 0.5},
            variant_b_config={"algorithm": "advanced", "threshold": 0.7},
            traffic_split=0.5,
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow() + timedelta(days=30)
        )
        await feature_flag_service.create_ab_test(ab_test_config)
        
        # Test multiple users to verify variant distribution
        variant_a_count = 0
        variant_b_count = 0
        total_tests = 100
        
        for i in range(total_tests):
            context = FeatureFlagEvaluationContext(user_id=f"user_{i}")
            result = await feature_flag_service.evaluate_flag("ab_test_flag", context)
            
            assert result.enabled is True
            assert result.variant in ["A", "B"]
            
            if result.variant == "A":
                variant_a_count += 1
                assert result.config["algorithm"] == "basic"
                assert result.config["threshold"] == 0.5
            else:
                variant_b_count += 1
                assert result.config["algorithm"] == "advanced"
                assert result.config["threshold"] == 0.7
        
        # Should be approximately 50/50 split (allow some variance)
        assert 40 <= variant_a_count <= 60
        assert 40 <= variant_b_count <= 60
    
    async def test_get_all_flags(self, feature_flag_service):
        """Test getting all feature flags."""
        # Create multiple flags
        flags_to_create = [
            FeatureFlagConfig(name="flag_1", description="First flag"),
            FeatureFlagConfig(name="flag_2", description="Second flag"),
            FeatureFlagConfig(name="flag_3", description="Third flag")
        ]
        
        for flag_config in flags_to_create:
            await feature_flag_service.create_feature_flag(flag_config)
        
        # Get all flags
        all_flags = await feature_flag_service.get_all_flags()
        
        # Verify all flags are returned
        assert len(all_flags) == 3
        flag_names = {flag.name for flag in all_flags}
        assert flag_names == {"flag_1", "flag_2", "flag_3"}
    
    async def test_update_callback_registration(self, feature_flag_service, sample_flag_config):
        """Test registering and receiving update callbacks."""
        callback_calls = []
        
        async def test_callback(flag_name, config):
            callback_calls.append((flag_name, config))
        
        # Register callback
        await feature_flag_service.register_update_callback(test_callback)
        
        # Create flag (should trigger callback)
        await feature_flag_service.create_feature_flag(sample_flag_config)
        
        # Update flag (should trigger callback)
        await feature_flag_service.update_feature_flag(
            sample_flag_config.name,
            {"rollout_percentage": 75.0}
        )
        
        # Delete flag (should trigger callback)
        await feature_flag_service.delete_feature_flag(sample_flag_config.name)
        
        # Verify callbacks were called
        assert len(callback_calls) == 3
        
        # Check create callback
        create_call = callback_calls[0]
        assert create_call[0] == sample_flag_config.name
        assert create_call[1] is not None
        
        # Check update callback
        update_call = callback_calls[1]
        assert update_call[0] == sample_flag_config.name
        assert update_call[1].rollout_percentage == 75.0
        
        # Check delete callback
        delete_call = callback_calls[2]
        assert delete_call[0] == sample_flag_config.name
        assert delete_call[1] is None
    
    async def test_flag_validation(self, feature_flag_service):
        """Test feature flag validation."""
        # Test invalid rollout percentage
        with pytest.raises(ValueError, match="between 0 and 100"):
            FeatureFlagConfig(
                name="invalid_flag",
                rollout_percentage=150.0
            )
        
        # Test invalid traffic split for A/B test
        with pytest.raises(ValueError, match="between 0 and 1"):
            ABTestConfig(
                feature_flag_id="test_id",
                name="invalid_test",
                traffic_split=1.5
            )
    
    async def test_error_handling_in_evaluation(self, feature_flag_service):
        """Test error handling during flag evaluation."""
        # Mock database error
        with patch.object(feature_flag_service, '_get_flag_config', side_effect=Exception("Database error")):
            context = FeatureFlagEvaluationContext(user_id="test_user")
            result = await feature_flag_service.evaluate_flag("test_flag", context)
            
            assert result.enabled is False
            assert "Evaluation error" in result.reason
    
    async def test_cache_functionality(self, feature_flag_service, sample_flag_config):
        """Test caching functionality."""
        # Create flag
        await feature_flag_service.create_feature_flag(sample_flag_config)
        
        # Verify flag is in cache
        assert sample_flag_config.name in feature_flag_service._flag_cache
        
        # Mock Redis calls to verify caching
        feature_flag_service.redis_client.setex.assert_called()
        
        # Evaluate flag (should use cache)
        context = FeatureFlagEvaluationContext(user_id="test_user")
        result = await feature_flag_service.evaluate_flag(sample_flag_config.name, context)
        
        # Should get result from cache
        assert result.flag_name == sample_flag_config.name
    
    @pytest.mark.asyncio
    async def test_background_tasks(self, feature_flag_service):
        """Test background tasks are started and stopped properly."""
        # Verify background tasks are running
        assert len(feature_flag_service._background_tasks) > 0
        
        # Shutdown should stop all tasks
        await feature_flag_service.shutdown()
        
        # Verify all tasks are cancelled
        for task in feature_flag_service._background_tasks:
            assert task.cancelled() or task.done()


class TestFeatureFlagIntegration:
    """Integration tests for feature flag service."""
    
    async def test_nautilus_integration_flag_scenario(self, feature_flag_service):
        """Test a realistic NautilusTrader integration scenario."""
        # Create NautilusTrader integration flag
        nautilus_flag = FeatureFlagConfig(
            name="nautilus_trader_integration",
            description="Enable NautilusTrader integration features",
            status=FeatureFlagStatus.TESTING,
            rollout_strategy=RolloutStrategy.USER_GROUP,
            target_groups=[UserGroup.DEVELOPERS, UserGroup.BETA_TESTERS],
            config_data={
                "enable_backtesting": True,
                "enable_live_trading": False,
                "max_concurrent_strategies": 5,
                "supported_venues": ["binance", "coinbase"],
                "risk_management_enabled": True
            }
        )
        
        await feature_flag_service.create_feature_flag(nautilus_flag, user_id="system")
        
        # Test developer access
        dev_context = FeatureFlagEvaluationContext(
            user_id="dev_123",
            user_groups=[UserGroup.DEVELOPERS],
            metadata={"environment": "development"}
        )
        
        result = await feature_flag_service.evaluate_flag("nautilus_trader_integration", dev_context)
        
        assert result.enabled is True
        assert result.config["enable_backtesting"] is True
        assert result.config["enable_live_trading"] is False
        assert result.config["max_concurrent_strategies"] == 5
        assert "binance" in result.config["supported_venues"]
        
        # Test production user (should be disabled)
        prod_context = FeatureFlagEvaluationContext(
            user_id="prod_user",
            user_groups=[UserGroup.EXTERNAL],
            metadata={"environment": "production"}
        )
        
        result = await feature_flag_service.evaluate_flag("nautilus_trader_integration", prod_context)
        assert result.enabled is False
    
    async def test_gradual_rollout_scenario(self, feature_flag_service):
        """Test gradual rollout scenario for NautilusTrader features."""
        # Create flag for gradual rollout
        rollout_flag = FeatureFlagConfig(
            name="nautilus_live_trading",
            description="Enable live trading with NautilusTrader",
            status=FeatureFlagStatus.ROLLOUT,
            rollout_strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=10.0,  # Start with 10%
            config_data={
                "enable_live_trading": True,
                "max_position_size": 1000,
                "supported_assets": ["BTC", "ETH"]
            }
        )
        
        await feature_flag_service.create_feature_flag(rollout_flag, user_id="admin")
        
        # Simulate gradual rollout increase
        rollout_percentages = [10.0, 25.0, 50.0, 100.0]
        
        for percentage in rollout_percentages:
            # Update rollout percentage
            await feature_flag_service.update_feature_flag(
                "nautilus_live_trading",
                {"rollout_percentage": percentage},
                user_id="admin"
            )
            
            # Test with multiple users
            enabled_count = 0
            total_users = 100
            
            for i in range(total_users):
                context = FeatureFlagEvaluationContext(user_id=f"trader_{i}")
                result = await feature_flag_service.evaluate_flag("nautilus_live_trading", context)
                if result.enabled:
                    enabled_count += 1
            
            # Verify rollout percentage (allow some variance)
            expected_min = max(0, percentage - 10)
            expected_max = min(100, percentage + 10)
            assert expected_min <= enabled_count <= expected_max
    
    async def test_ab_test_strategy_comparison(self, feature_flag_service):
        """Test A/B testing for strategy algorithm comparison."""
        # Create flag for strategy A/B test
        strategy_flag = FeatureFlagConfig(
            name="strategy_algorithm_test",
            description="A/B test different strategy algorithms",
            status=FeatureFlagStatus.TESTING,
            rollout_strategy=RolloutStrategy.AB_TEST
        )
        
        flag = await feature_flag_service.create_feature_flag(strategy_flag, user_id="researcher")
        
        # Create A/B test for strategy algorithms
        ab_test = ABTestConfig(
            feature_flag_id=flag.id,
            name="momentum_vs_mean_reversion",
            description="Compare momentum vs mean reversion strategies",
            variant_a_config={
                "strategy_type": "momentum",
                "lookback_period": 20,
                "threshold": 0.02,
                "risk_factor": 0.1
            },
            variant_b_config={
                "strategy_type": "mean_reversion",
                "lookback_period": 50,
                "threshold": 0.05,
                "risk_factor": 0.15
            },
            traffic_split=0.4,  # 40% get mean reversion, 60% get momentum
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=30)
        )
        
        await feature_flag_service.create_ab_test(ab_test, user_id="researcher")
        
        # Test variant distribution
        momentum_count = 0
        mean_reversion_count = 0
        
        for i in range(100):
            context = FeatureFlagEvaluationContext(
                user_id=f"strategy_user_{i}",
                metadata={"strategy_id": f"strat_{i}"}
            )
            
            result = await feature_flag_service.evaluate_flag("strategy_algorithm_test", context)
            
            assert result.enabled is True
            assert result.variant in ["A", "B"]
            
            if result.variant == "A":
                momentum_count += 1
                assert result.config["strategy_type"] == "momentum"
                assert result.config["lookback_period"] == 20
            else:
                mean_reversion_count += 1
                assert result.config["strategy_type"] == "mean_reversion"
                assert result.config["lookback_period"] == 50
        
        # Verify traffic split (60% A, 40% B with some variance)
        assert 50 <= momentum_count <= 70
        assert 30 <= mean_reversion_count <= 50