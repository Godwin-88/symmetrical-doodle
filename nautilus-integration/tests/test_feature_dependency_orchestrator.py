"""
Tests for Feature Flag and Dependency Management Orchestrator.

This module tests the orchestrator that coordinates feature flags and dependency management.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.services.feature_dependency_orchestrator import FeatureDependencyOrchestrator
from nautilus_integration.services.feature_flag_service import (
    FeatureFlagConfig,
    FeatureFlagStatus,
    RolloutStrategy,
    UserGroup,
    FeatureFlagResult,
    ABTestConfig
)
from nautilus_integration.services.dependency_manager import (
    DependencyInfo,
    DependencyType,
    DependencyStatus,
    DependencyHealthReport
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = MagicMock(spec=NautilusConfig)
    
    # Feature flags config
    config.feature_flags.enabled = True
    config.feature_flags.database_url = "sqlite+aiosqlite:///:memory:"
    config.feature_flags.redis_url = "redis://localhost:6379"
    config.feature_flags.config_file_path = "./test_feature_flags.json"
    config.feature_flags.cache_ttl = 60
    config.feature_flags.enable_audit_logging = True
    
    # Dependency management config
    config.dependency_management.enabled = True
    config.dependency_management.database_url = "sqlite+aiosqlite:///:memory:"
    config.dependency_management.environments = ["development", "testing", "production"]
    config.dependency_management.check_interval = 60
    config.dependency_management.vulnerability_sources = ["https://test-source.com"]
    
    return config


@pytest.fixture
async def orchestrator(mock_config):
    """Create an orchestrator for testing."""
    orchestrator = FeatureDependencyOrchestrator(mock_config)
    
    # Mock the services
    orchestrator.feature_flag_service = AsyncMock()
    orchestrator.dependency_manager = AsyncMock()
    
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()


@pytest.fixture
def sample_flag_result():
    """Create a sample feature flag result."""
    return FeatureFlagResult(
        flag_name="test_flag",
        enabled=True,
        config={"feature_value": "test"},
        reason="Flag enabled",
        evaluation_context=MagicMock()
    )


@pytest.fixture
def sample_health_report():
    """Create a sample dependency health report."""
    return DependencyHealthReport(
        environment="development",
        total_dependencies=10,
        healthy_count=8,
        warning_count=2,
        critical_count=0,
        unknown_count=0,
        deprecated_count=0,
        vulnerabilities_count=0,
        last_updated=datetime.utcnow(),
        dependencies=[
            DependencyInfo(
                name="nautilus_trader",
                type=DependencyType.PYTHON,
                current_version="1.190.0",
                status=DependencyStatus.WARNING
            )
        ]
    )


class TestFeatureDependencyOrchestrator:
    """Test cases for FeatureDependencyOrchestrator."""
    
    async def test_initialization_with_both_services_enabled(self, mock_config):
        """Test initialization with both services enabled."""
        orchestrator = FeatureDependencyOrchestrator(mock_config)
        
        with patch('nautilus_integration.services.feature_dependency_orchestrator.FeatureFlagService') as mock_ff_service, \
             patch('nautilus_integration.services.feature_dependency_orchestrator.DependencyManager') as mock_dep_manager:
            
            mock_ff_instance = AsyncMock()
            mock_dep_instance = AsyncMock()
            mock_ff_service.return_value = mock_ff_instance
            mock_dep_manager.return_value = mock_dep_instance
            
            await orchestrator.initialize()
            
            # Verify services were created and initialized
            mock_ff_service.assert_called_once()
            mock_dep_manager.assert_called_once()
            mock_ff_instance.initialize.assert_called_once()
            mock_dep_instance.initialize.assert_called_once()
            
            await orchestrator.shutdown()
    
    async def test_initialization_with_feature_flags_disabled(self, mock_config):
        """Test initialization with feature flags disabled."""
        mock_config.feature_flags.enabled = False
        
        orchestrator = FeatureDependencyOrchestrator(mock_config)
        
        with patch('nautilus_integration.services.feature_dependency_orchestrator.DependencyManager') as mock_dep_manager:
            mock_dep_instance = AsyncMock()
            mock_dep_manager.return_value = mock_dep_instance
            
            await orchestrator.initialize()
            
            # Verify only dependency manager was created
            assert orchestrator.feature_flag_service is None
            assert orchestrator.dependency_manager is not None
            
            await orchestrator.shutdown()
    
    async def test_evaluate_feature_flag_basic(self, orchestrator, sample_flag_result):
        """Test basic feature flag evaluation."""
        orchestrator.feature_flag_service.evaluate_flag.return_value = sample_flag_result
        
        result = await orchestrator.evaluate_feature_flag(
            "test_flag",
            user_id="test_user",
            user_groups=[UserGroup.DEVELOPERS]
        )
        
        assert result.flag_name == "test_flag"
        assert result.enabled is True
        assert result.config["feature_value"] == "test"
        
        # Verify service was called with correct context
        orchestrator.feature_flag_service.evaluate_flag.assert_called_once()
        call_args = orchestrator.feature_flag_service.evaluate_flag.call_args
        assert call_args[0][0] == "test_flag"  # flag_name
        assert call_args[0][1].user_id == "test_user"  # context
    
    async def test_evaluate_feature_flag_with_dependency_health_check(self, orchestrator):
        """Test feature flag evaluation with dependency health requirement."""
        # Mock flag that requires healthy dependencies
        flag_config = FeatureFlagConfig(
            name="dependency_sensitive_flag",
            config_data={"require_healthy_dependencies": True}
        )
        
        orchestrator.feature_flag_service.get_all_flags.return_value = [flag_config]
        
        # Mock unhealthy dependencies
        with patch.object(orchestrator, '_get_dependency_health_status', return_value="critical"):
            result = await orchestrator.evaluate_feature_flag("dependency_sensitive_flag")
        
        assert result.enabled is False
        assert "Dependencies not healthy" in result.reason
    
    async def test_evaluate_feature_flag_service_disabled(self, orchestrator):
        """Test feature flag evaluation when service is disabled."""
        orchestrator.feature_flag_service = None
        
        result = await orchestrator.evaluate_feature_flag("test_flag")
        
        assert result.enabled is False
        assert result.reason == "Feature flag service not enabled"
    
    async def test_create_feature_flag(self, orchestrator):
        """Test creating a feature flag."""
        created_flag = FeatureFlagConfig(
            name="new_flag",
            description="Test flag",
            rollout_percentage=25.0
        )
        
        orchestrator.feature_flag_service.create_feature_flag.return_value = created_flag
        
        result = await orchestrator.create_feature_flag(
            name="new_flag",
            description="Test flag",
            rollout_percentage=25.0,
            user_id="admin",
            require_healthy_dependencies=True
        )
        
        assert result.name == "new_flag"
        assert result.description == "Test flag"
        assert result.rollout_percentage == 25.0
        
        # Verify service was called
        orchestrator.feature_flag_service.create_feature_flag.assert_called_once()
        call_args = orchestrator.feature_flag_service.create_feature_flag.call_args
        flag_config = call_args[0][0]
        assert flag_config.config_data["require_healthy_dependencies"] is True
    
    async def test_create_ab_test(self, orchestrator):
        """Test creating an A/B test."""
        # Mock existing flag
        existing_flag = FeatureFlagConfig(id="flag_123", name="test_flag")
        orchestrator.feature_flag_service.get_all_flags.return_value = [existing_flag]
        
        # Mock created A/B test
        created_test = ABTestConfig(
            feature_flag_id="flag_123",
            name="test_experiment",
            traffic_split=0.5
        )
        orchestrator.feature_flag_service.create_ab_test.return_value = created_test
        
        result = await orchestrator.create_ab_test(
            feature_flag_name="test_flag",
            test_name="test_experiment",
            traffic_split=0.5,
            user_id="researcher"
        )
        
        assert result.name == "test_experiment"
        assert result.feature_flag_id == "flag_123"
        assert result.traffic_split == 0.5
    
    async def test_get_dependency_health_report(self, orchestrator, sample_health_report):
        """Test getting dependency health report."""
        orchestrator.dependency_manager.scan_dependencies.return_value = sample_health_report
        
        report = await orchestrator.get_dependency_health_report("development")
        
        assert report.environment == "development"
        assert report.total_dependencies == 10
        assert report.healthy_count == 8
        assert report.warning_count == 2
        
        orchestrator.dependency_manager.scan_dependencies.assert_called_once_with("development")
    
    async def test_create_system_snapshot(self, orchestrator):
        """Test creating a comprehensive system snapshot."""
        # Mock feature flags
        mock_flags = [
            FeatureFlagConfig(name="flag1", description="First flag"),
            FeatureFlagConfig(name="flag2", description="Second flag")
        ]
        orchestrator.feature_flag_service.get_all_flags.return_value = mock_flags
        
        # Mock dependency snapshot
        mock_dep_snapshot = MagicMock()
        mock_dep_snapshot.model_dump.return_value = {"dependencies": ["dep1", "dep2"]}
        orchestrator.dependency_manager.create_snapshot.return_value = mock_dep_snapshot
        
        with patch('os.makedirs'), patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            snapshot = await orchestrator.create_system_snapshot(
                name="test_snapshot",
                environment="development",
                user_id="admin",
                notes="Test snapshot"
            )
        
        assert snapshot["name"] == "test_snapshot"
        assert snapshot["environment"] == "development"
        assert snapshot["created_by"] == "admin"
        assert "feature_flags" in snapshot["components"]
        assert "dependencies" in snapshot["components"]
        
        # Verify feature flags were included
        assert len(snapshot["components"]["feature_flags"]) == 2
        
        # Verify dependency snapshot was created
        orchestrator.dependency_manager.create_snapshot.assert_called_once()
    
    async def test_rollback_system_dry_run(self, orchestrator):
        """Test system rollback in dry run mode."""
        # Mock snapshot data
        snapshot_data = {
            "name": "test_snapshot",
            "environment": "development",
            "components": {
                "feature_flags": [
                    {"name": "flag1", "status": "enabled"},
                    {"name": "flag2", "status": "disabled"}
                ],
                "dependencies": {"dependencies": ["dep1", "dep2"]}
            }
        }
        
        with patch.object(orchestrator, '_load_snapshot_data', return_value=snapshot_data):
            # Mock dependency rollback
            dep_rollback_result = {
                "changes": [{"action": "update", "dependency": "dep1"}],
                "success_count": 1,
                "failed_count": 0
            }
            orchestrator.dependency_manager.rollback_to_snapshot.return_value = dep_rollback_result
            
            # Mock feature flag rollback
            with patch.object(orchestrator, '_rollback_feature_flags') as mock_ff_rollback:
                ff_rollback_result = {
                    "changes": [{"action": "update", "flag_name": "flag1"}],
                    "success_count": 1,
                    "failed_count": 0
                }
                mock_ff_rollback.return_value = ff_rollback_result
                
                result = await orchestrator.rollback_system(
                    "test_snapshot",
                    "development",
                    dry_run=True
                )
        
        assert result["snapshot_name"] == "test_snapshot"
        assert result["dry_run"] is True
        assert "dependencies" in result["components"]
        assert "feature_flags" in result["components"]
        assert result["success"] is True
    
    async def test_get_system_health_status(self, orchestrator, sample_health_report):
        """Test getting comprehensive system health status."""
        # Mock feature flag service health
        orchestrator.feature_flag_service.get_all_flags.return_value = [
            FeatureFlagConfig(name="flag1"),
            FeatureFlagConfig(name="flag2")
        ]
        
        # Mock dependency manager health
        orchestrator.dependency_manager.scan_dependencies.return_value = sample_health_report
        
        health_status = await orchestrator.get_system_health_status()
        
        assert health_status["overall_status"] == "warning"  # Due to warning dependencies
        assert "feature_flags" in health_status["components"]
        assert "dependencies_development" in health_status["components"]
        
        # Verify feature flags component
        ff_component = health_status["components"]["feature_flags"]
        assert ff_component["status"] == "healthy"
        assert ff_component["flag_count"] == 2
        
        # Verify dependencies component
        dep_component = health_status["components"]["dependencies_development"]
        assert dep_component["status"] == "warning"
        assert dep_component["total_dependencies"] == 10
        assert dep_component["warning_count"] == 2
    
    async def test_feature_flag_update_callback(self, orchestrator):
        """Test feature flag update callback handling."""
        # Mock flag that affects dependencies
        flag_config = FeatureFlagConfig(
            name="dependency_affecting_flag",
            config_data={"affects_dependencies": True}
        )
        
        with patch.object(orchestrator, 'get_system_health_status') as mock_health_check:
            await orchestrator._on_feature_flag_update("dependency_affecting_flag", flag_config)
            
            # Should trigger health check
            mock_health_check.assert_called_once()
    
    async def test_dependency_alert_callback(self, orchestrator):
        """Test dependency alert callback handling."""
        # Mock critical dependency report
        critical_report = DependencyHealthReport(
            environment="production",
            total_dependencies=5,
            healthy_count=3,
            warning_count=1,
            critical_count=1,
            unknown_count=0,
            deprecated_count=0,
            vulnerabilities_count=0,
            last_updated=datetime.utcnow(),
            dependencies=[]
        )
        
        # Mock flag that requires healthy dependencies
        flag_config = FeatureFlagConfig(
            name="critical_flag",
            config_data={"require_healthy_dependencies": True}
        )
        orchestrator.feature_flag_service.get_all_flags.return_value = [flag_config]
        
        alerts = ["Critical dependencies found: 1"]
        
        # Should handle alert without raising exception
        await orchestrator._on_dependency_alert("production", alerts, critical_report)
        
        # Verify flag service was queried
        orchestrator.feature_flag_service.get_all_flags.assert_called_once()
    
    async def test_flag_requires_healthy_dependencies_check(self, orchestrator):
        """Test checking if a flag requires healthy dependencies."""
        # Mock flag with dependency requirement
        flag_config = FeatureFlagConfig(
            name="dependency_flag",
            config_data={"require_healthy_dependencies": True}
        )
        orchestrator.feature_flag_service.get_all_flags.return_value = [flag_config]
        
        result = await orchestrator._flag_requires_healthy_dependencies("dependency_flag")
        assert result is True
        
        # Test flag without dependency requirement
        flag_config.config_data = {}
        result = await orchestrator._flag_requires_healthy_dependencies("dependency_flag")
        assert result is False
        
        # Test non-existent flag
        orchestrator.feature_flag_service.get_all_flags.return_value = []
        result = await orchestrator._flag_requires_healthy_dependencies("nonexistent_flag")
        assert result is False
    
    async def test_get_dependency_health_status(self, orchestrator):
        """Test getting overall dependency health status."""
        # Mock healthy environment
        healthy_report = DependencyHealthReport(
            environment="development",
            total_dependencies=5,
            healthy_count=5,
            warning_count=0,
            critical_count=0,
            unknown_count=0,
            deprecated_count=0,
            vulnerabilities_count=0,
            last_updated=datetime.utcnow(),
            dependencies=[]
        )
        
        # Mock warning environment
        warning_report = DependencyHealthReport(
            environment="testing",
            total_dependencies=5,
            healthy_count=4,
            warning_count=1,
            critical_count=0,
            unknown_count=0,
            deprecated_count=0,
            vulnerabilities_count=0,
            last_updated=datetime.utcnow(),
            dependencies=[]
        )
        
        orchestrator.dependency_manager.scan_dependencies.side_effect = [
            healthy_report,
            warning_report
        ]
        
        status = await orchestrator._get_dependency_health_status()
        assert status == "warning"  # Should return worst status
    
    async def test_error_handling_in_evaluation(self, orchestrator):
        """Test error handling during feature flag evaluation."""
        orchestrator.feature_flag_service.evaluate_flag.side_effect = Exception("Service error")
        
        result = await orchestrator.evaluate_feature_flag("test_flag")
        
        assert result.enabled is False
        assert "Evaluation error" in result.reason
    
    async def test_background_tasks_lifecycle(self, orchestrator):
        """Test background tasks are started and stopped properly."""
        # Verify background tasks are running
        assert len(orchestrator._background_tasks) > 0
        
        # Shutdown should stop all tasks
        await orchestrator.shutdown()
        
        # Verify all tasks are cancelled
        for task in orchestrator._background_tasks:
            assert task.cancelled() or task.done()


class TestOrchestrationIntegration:
    """Integration tests for the orchestrator."""
    
    async def test_nautilus_trader_feature_rollout_scenario(self, orchestrator):
        """Test a realistic NautilusTrader feature rollout scenario."""
        # Create NautilusTrader integration flag
        nautilus_flag = FeatureFlagConfig(
            name="nautilus_trader_integration",
            description="Enable NautilusTrader integration",
            status=FeatureFlagStatus.TESTING,
            rollout_strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=25.0,
            config_data={
                "require_healthy_dependencies": True,
                "enable_backtesting": True,
                "enable_live_trading": False
            }
        )
        
        orchestrator.feature_flag_service.create_feature_flag.return_value = nautilus_flag
        orchestrator.feature_flag_service.get_all_flags.return_value = [nautilus_flag]
        
        # Create flag
        created_flag = await orchestrator.create_feature_flag(
            name="nautilus_trader_integration",
            description="Enable NautilusTrader integration",
            rollout_percentage=25.0,
            require_healthy_dependencies=True,
            user_id="admin"
        )
        
        assert created_flag.name == "nautilus_trader_integration"
        assert created_flag.config_data["require_healthy_dependencies"] is True
        
        # Mock healthy dependencies
        with patch.object(orchestrator, '_get_dependency_health_status', return_value="healthy"):
            # Mock flag evaluation
            flag_result = FeatureFlagResult(
                flag_name="nautilus_trader_integration",
                enabled=True,
                config=nautilus_flag.config_data,
                reason="Percentage rollout",
                evaluation_context=MagicMock()
            )
            orchestrator.feature_flag_service.evaluate_flag.return_value = flag_result
            
            result = await orchestrator.evaluate_feature_flag(
                "nautilus_trader_integration",
                user_id="trader_123"
            )
        
        assert result.enabled is True
        assert result.config["enable_backtesting"] is True
        assert result.config["enable_live_trading"] is False
    
    async def test_dependency_driven_feature_disabling(self, orchestrator):
        """Test feature disabling based on dependency health."""
        # Create flag that requires healthy dependencies
        dependency_flag = FeatureFlagConfig(
            name="critical_trading_feature",
            config_data={"require_healthy_dependencies": True}
        )
        
        orchestrator.feature_flag_service.get_all_flags.return_value = [dependency_flag]
        
        # Mock critical dependency issues
        with patch.object(orchestrator, '_get_dependency_health_status', return_value="critical"):
            result = await orchestrator.evaluate_feature_flag("critical_trading_feature")
        
        assert result.enabled is False
        assert "Dependencies not healthy: critical" in result.reason
    
    async def test_coordinated_rollback_scenario(self, orchestrator):
        """Test coordinated rollback of both feature flags and dependencies."""
        # Mock snapshot with both components
        snapshot_data = {
            "name": "stable_release",
            "environment": "production",
            "components": {
                "feature_flags": [
                    {
                        "name": "nautilus_integration",
                        "status": "disabled",
                        "rollout_percentage": 0.0
                    }
                ],
                "dependencies": {
                    "dependencies": [
                        {
                            "name": "nautilus_trader",
                            "current_version": "1.190.0",
                            "type": "python"
                        }
                    ]
                }
            }
        }
        
        with patch.object(orchestrator, '_load_snapshot_data', return_value=snapshot_data):
            # Mock successful dependency rollback
            dep_result = {
                "changes": [{"action": "update", "dependency": "nautilus_trader"}],
                "success_count": 1,
                "failed_count": 0,
                "completed": True
            }
            orchestrator.dependency_manager.rollback_to_snapshot.return_value = dep_result
            
            # Mock successful feature flag rollback
            with patch.object(orchestrator, '_rollback_feature_flags') as mock_ff_rollback:
                ff_result = {
                    "changes": [{"action": "update", "flag_name": "nautilus_integration"}],
                    "success_count": 1,
                    "failed_count": 0
                }
                mock_ff_rollback.return_value = ff_result
                
                result = await orchestrator.rollback_system(
                    "stable_release",
                    "production",
                    user_id="admin"
                )
        
        assert result["success"] is True
        assert result["components"]["dependencies"]["completed"] is True
        assert result["components"]["feature_flags"]["success_count"] == 1
    
    async def test_health_monitoring_integration(self, orchestrator):
        """Test integrated health monitoring across both systems."""
        # Mock feature flag service health
        orchestrator.feature_flag_service.get_all_flags.return_value = [
            FeatureFlagConfig(name="flag1"),
            FeatureFlagConfig(name="flag2"),
            FeatureFlagConfig(name="flag3")
        ]
        
        # Mock dependency health for multiple environments
        healthy_report = DependencyHealthReport(
            environment="development",
            total_dependencies=10,
            healthy_count=10,
            warning_count=0,
            critical_count=0,
            unknown_count=0,
            deprecated_count=0,
            vulnerabilities_count=0,
            last_updated=datetime.utcnow(),
            dependencies=[]
        )
        
        warning_report = DependencyHealthReport(
            environment="production",
            total_dependencies=15,
            healthy_count=12,
            warning_count=3,
            critical_count=0,
            unknown_count=0,
            deprecated_count=0,
            vulnerabilities_count=0,
            last_updated=datetime.utcnow(),
            dependencies=[]
        )
        
        orchestrator.dependency_manager.scan_dependencies.side_effect = [
            healthy_report,
            MagicMock(),  # testing environment
            warning_report  # production environment
        ]
        
        health_status = await orchestrator.get_system_health_status()
        
        # Overall status should be warning due to production dependencies
        assert health_status["overall_status"] == "warning"
        
        # Verify all components are included
        assert "feature_flags" in health_status["components"]
        assert "dependencies_development" in health_status["components"]
        assert "dependencies_production" in health_status["components"]
        
        # Verify feature flags are healthy
        ff_status = health_status["components"]["feature_flags"]
        assert ff_status["status"] == "healthy"
        assert ff_status["flag_count"] == 3
        
        # Verify production dependencies show warning
        prod_deps = health_status["components"]["dependencies_production"]
        assert prod_deps["status"] == "warning"
        assert prod_deps["warning_count"] == 3