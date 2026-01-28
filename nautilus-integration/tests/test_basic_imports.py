"""
Basic import tests to verify the services can be imported correctly.
"""

import pytest


def test_feature_flag_service_import():
    """Test that FeatureFlagService can be imported."""
    from nautilus_integration.services.feature_flag_service import (
        FeatureFlagService,
        FeatureFlagConfig,
        FeatureFlagStatus,
        RolloutStrategy,
        UserGroup
    )
    
    # Verify classes are available
    assert FeatureFlagService is not None
    assert FeatureFlagConfig is not None
    assert FeatureFlagStatus is not None
    assert RolloutStrategy is not None
    assert UserGroup is not None


def test_dependency_manager_import():
    """Test that DependencyManager can be imported."""
    from nautilus_integration.services.dependency_manager import (
        DependencyManager,
        DependencyInfo,
        DependencyType,
        DependencyStatus
    )
    
    # Verify classes are available
    assert DependencyManager is not None
    assert DependencyInfo is not None
    assert DependencyType is not None
    assert DependencyStatus is not None


def test_orchestrator_import():
    """Test that FeatureDependencyOrchestrator can be imported."""
    from nautilus_integration.services.feature_dependency_orchestrator import (
        FeatureDependencyOrchestrator
    )
    
    # Verify class is available
    assert FeatureDependencyOrchestrator is not None


def test_error_handling_import():
    """Test that error handling classes can be imported."""
    from nautilus_integration.core.error_handling_simple import (
        ErrorRecoveryManager,
        CircuitBreaker,
        CircuitBreakerConfig
    )
    
    # Verify classes are available
    assert ErrorRecoveryManager is not None
    assert CircuitBreaker is not None
    assert CircuitBreakerConfig is not None


def test_config_import():
    """Test that configuration classes can be imported."""
    from nautilus_integration.core.config import (
        NautilusConfig,
        FeatureFlagConfig,
        DependencyManagementConfig
    )
    
    # Verify classes are available
    assert NautilusConfig is not None
    assert FeatureFlagConfig is not None
    assert DependencyManagementConfig is not None


def test_feature_flag_config_creation():
    """Test creating a basic feature flag configuration."""
    from nautilus_integration.services.feature_flag_service import (
        FeatureFlagConfig,
        FeatureFlagStatus,
        RolloutStrategy,
        UserGroup
    )
    
    config = FeatureFlagConfig(
        name="test_flag",
        description="Test flag",
        status=FeatureFlagStatus.TESTING,
        rollout_strategy=RolloutStrategy.PERCENTAGE,
        rollout_percentage=25.0,
        target_groups=[UserGroup.DEVELOPERS]
    )
    
    assert config.name == "test_flag"
    assert config.description == "Test flag"
    assert config.status == FeatureFlagStatus.TESTING
    assert config.rollout_percentage == 25.0
    assert UserGroup.DEVELOPERS in config.target_groups


def test_dependency_info_creation():
    """Test creating a basic dependency info object."""
    from nautilus_integration.services.dependency_manager import (
        DependencyInfo,
        DependencyType,
        DependencyStatus
    )
    
    dep = DependencyInfo(
        name="nautilus_trader",
        type=DependencyType.PYTHON,
        current_version="1.190.0",
        latest_version="1.195.0",
        status=DependencyStatus.WARNING
    )
    
    assert dep.name == "nautilus_trader"
    assert dep.type == DependencyType.PYTHON
    assert dep.current_version == "1.190.0"
    assert dep.latest_version == "1.195.0"
    assert dep.status == DependencyStatus.WARNING


def test_circuit_breaker_creation():
    """Test creating a circuit breaker."""
    from nautilus_integration.core.error_handling_simple import (
        CircuitBreaker,
        CircuitBreakerConfig
    )
    
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30
    )
    
    breaker = CircuitBreaker(config, "test_component")
    
    assert breaker.config.failure_threshold == 3
    assert breaker.config.recovery_timeout == 30
    assert breaker.component == "test_component"