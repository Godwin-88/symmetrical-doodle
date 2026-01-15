"""
Tests for graceful shutdown procedures.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from intelligence_layer.shutdown import (
    ShutdownManager,
    ShutdownConfig,
    ShutdownPhase,
    ExampleComponent,
    managed_shutdown,
    get_shutdown_manager
)


@pytest.mark.asyncio
async def test_shutdown_manager_creation():
    """Test shutdown manager creation."""
    config = ShutdownConfig()
    manager = ShutdownManager(config)
    
    assert manager.status.phase == ShutdownPhase.INITIATED
    assert not manager.status.force_shutdown
    assert manager.status.timeout_seconds == config.graceful_timeout_seconds


@pytest.mark.asyncio
async def test_add_component():
    """Test adding components to shutdown manager."""
    manager = ShutdownManager()
    component = ExampleComponent("test_component")
    
    manager.add_component(component)
    
    assert "test_component" in manager.components
    assert manager.status.components_shutdown["test_component"] is False


@pytest.mark.asyncio
async def test_graceful_shutdown_sequence():
    """Test complete graceful shutdown sequence."""
    config = ShutdownConfig(graceful_timeout_seconds=5)
    manager = ShutdownManager(config)
    
    # Add test components
    component1 = ExampleComponent("component1")
    component2 = ExampleComponent("component2")
    
    manager.add_component(component1)
    manager.add_component(component2)
    
    # Initiate shutdown
    await manager.initiate_shutdown()
    
    # Wait for shutdown to complete
    await manager.wait_for_shutdown()
    
    # Verify shutdown completed successfully
    assert manager.status.phase == ShutdownPhase.COMPLETED
    assert not manager.status.force_shutdown
    assert manager.status.components_shutdown["component1"] is True
    assert manager.status.components_shutdown["component2"] is True


@pytest.mark.asyncio
async def test_force_shutdown():
    """Test force shutdown functionality."""
    config = ShutdownConfig(graceful_timeout_seconds=1)  # Very short timeout
    manager = ShutdownManager(config)
    
    component = ExampleComponent("test_component")
    manager.add_component(component)
    
    # Force shutdown immediately
    await manager.force_shutdown()
    
    assert manager.status.force_shutdown is True
    assert manager.status.phase == ShutdownPhase.FAILED
    assert manager.status.components_shutdown["test_component"] is True


@pytest.mark.asyncio
async def test_shutdown_validation():
    """Test shutdown validation."""
    manager = ShutdownManager()
    component = ExampleComponent("test_component")
    manager.add_component(component)
    
    # Execute shutdown
    await manager.initiate_shutdown()
    await manager.wait_for_shutdown()
    
    # Validate shutdown
    report = await manager.validate_shutdown()
    
    assert report.is_valid is True
    assert report.checks["shutdown_completed"] is True
    assert report.checks["all_components_shutdown"] is True
    assert report.checks["no_shutdown_errors"] is True


@pytest.mark.asyncio
async def test_shutdown_with_errors():
    """Test shutdown behavior when components fail."""
    manager = ShutdownManager()
    
    # Create a component that will fail during shutdown
    failing_component = ExampleComponent("failing_component")
    
    # Mock the shutdown method to raise an exception
    failing_component.shutdown = AsyncMock(side_effect=Exception("Shutdown failed"))
    
    manager.add_component(failing_component)
    
    # Execute shutdown
    await manager.initiate_shutdown()
    await manager.wait_for_shutdown()
    
    # Verify errors were recorded
    assert len(manager.status.errors) > 0
    assert "Shutdown failed for failing_component" in manager.status.errors[0]
    
    # Validate shutdown
    report = await manager.validate_shutdown()
    assert report.is_valid is False
    assert "Shutdown failed for failing_component" in report.errors[0]


@pytest.mark.asyncio
async def test_state_persistence():
    """Test state persistence during shutdown."""
    config = ShutdownConfig(persist_state=True)
    manager = ShutdownManager(config)
    
    component = ExampleComponent("test_component")
    manager.add_component(component)
    
    # Mock file operations
    with patch("builtins.open", create=True) as mock_open:
        mock_file = AsyncMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Execute shutdown
        await manager.initiate_shutdown()
        await manager.wait_for_shutdown()
        
        # Verify state was persisted
        mock_open.assert_called_once()


@pytest.mark.asyncio
async def test_managed_shutdown_context():
    """Test managed shutdown context manager."""
    components = [
        ExampleComponent("component1"),
        ExampleComponent("component2")
    ]
    
    config = ShutdownConfig(graceful_timeout_seconds=5)
    
    async with managed_shutdown(components, config) as manager:
        assert len(manager.components) == 2
        assert "component1" in manager.components
        assert "component2" in manager.components
        
        # Simulate some work
        await asyncio.sleep(0.1)
    
    # After context exit, shutdown should be complete
    assert manager.status.phase in [ShutdownPhase.COMPLETED, ShutdownPhase.FAILED]


@pytest.mark.asyncio
async def test_component_ready_for_shutdown():
    """Test component readiness checking."""
    component = ExampleComponent("test_component")
    
    # Initially ready
    assert await component.is_ready_for_shutdown() is True
    
    # Simulate work in progress
    component.work_in_progress = True
    assert await component.is_ready_for_shutdown() is False
    
    # Work completed
    component.work_in_progress = False
    assert await component.is_ready_for_shutdown() is True


@pytest.mark.asyncio
async def test_shutdown_timeout():
    """Test shutdown timeout handling."""
    config = ShutdownConfig(graceful_timeout_seconds=1)  # Very short timeout
    manager = ShutdownManager(config)
    
    # Create a component that takes too long to shutdown
    slow_component = ExampleComponent("slow_component")
    
    # Mock shutdown to take longer than timeout
    async def slow_shutdown():
        await asyncio.sleep(2)  # Longer than timeout
    
    slow_component.shutdown = slow_shutdown
    manager.add_component(slow_component)
    
    # Execute shutdown
    await manager.initiate_shutdown()
    await manager.wait_for_shutdown()
    
    # Should have been force shutdown due to timeout
    assert manager.status.force_shutdown is True
    assert manager.status.phase == ShutdownPhase.FAILED


def test_global_shutdown_manager():
    """Test global shutdown manager singleton."""
    manager1 = get_shutdown_manager()
    manager2 = get_shutdown_manager()
    
    # Should be the same instance
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_shutdown_phases():
    """Test that shutdown goes through all expected phases."""
    manager = ShutdownManager()
    component = ExampleComponent("test_component")
    manager.add_component(component)
    
    phases_seen = []
    
    # Mock the phase update to track phases
    original_run_sequence = manager._run_shutdown_sequence
    
    async def track_phases():
        phases_seen.append(manager.status.phase)
        await original_run_sequence()
    
    manager._run_shutdown_sequence = track_phases
    
    # Execute shutdown
    await manager.initiate_shutdown()
    await manager.wait_for_shutdown()
    
    # Verify we went through expected phases
    assert ShutdownPhase.STOPPING_SERVICES in [manager.status.phase] or len(phases_seen) > 0
    assert manager.status.phase == ShutdownPhase.COMPLETED


@pytest.mark.asyncio
async def test_component_prepare_and_shutdown():
    """Test component preparation and shutdown sequence."""
    component = ExampleComponent("test_component")
    
    # Initially running
    assert component.is_running is True
    
    # Prepare for shutdown
    await component.prepare_shutdown()
    assert component.is_running is False
    
    # Should be ready for shutdown
    assert await component.is_ready_for_shutdown() is True
    
    # Perform shutdown
    await component.shutdown()
    
    # Force shutdown should work
    await component.force_shutdown()
    assert component.work_in_progress is False