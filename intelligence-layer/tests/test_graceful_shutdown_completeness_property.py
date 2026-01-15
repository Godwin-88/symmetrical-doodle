"""
Property-based tests for graceful shutdown completeness validation.

This module validates Property 24: Graceful Shutdown Completeness
- All components complete shutdown without data loss or corruption
- State persistence works correctly across different scenarios
- Resource cleanup is complete and verifiable
- Shutdown validation reports accurate results
- Emergency shutdown procedures work under all conditions

Requirements validated: 10.5
"""

import asyncio
import json
import tempfile
import os
import random
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from intelligence_layer.shutdown import (
    ShutdownManager,
    ShutdownConfig,
    ShutdownPhase,
    ShutdownComponent,
    ShutdownStatus,
    ShutdownValidationReport,
    ExampleComponent
)


@dataclass
class MockComponentState:
    """State tracking for mock components during testing."""
    name: str
    is_running: bool = True
    work_in_progress: bool = False
    shutdown_duration: float = 0.1
    failure_probability: float = 0.0
    data_to_persist: Dict = None
    resources_allocated: Set[str] = None
    
    def __post_init__(self):
        if self.data_to_persist is None:
            self.data_to_persist = {}
        if self.resources_allocated is None:
            self.resources_allocated = set()


class PropertyTestComponent:
    """Test component with configurable behavior for property testing."""
    
    def __init__(self, state: MockComponentState):
        self.state = state
        self._prepare_called = False
        self._shutdown_called = False
        self._force_shutdown_called = False
        self._data_persisted = False
        self._resources_cleaned = False
    
    def component_name(self) -> str:
        return self.state.name
    
    async def prepare_shutdown(self) -> None:
        """Prepare for shutdown with configurable behavior."""
        self._prepare_called = True
        
        # Simulate failure during preparation
        if random.random() < self.state.failure_probability:
            raise Exception(f"Prepare shutdown failed for {self.state.name}")
        
        self.state.is_running = False
        await asyncio.sleep(0.01)  # Simulate preparation work
    
    async def shutdown(self) -> None:
        """Perform shutdown with configurable duration and failure."""
        self._shutdown_called = True
        
        # Simulate shutdown work
        await asyncio.sleep(self.state.shutdown_duration)
        
        # Simulate failure during shutdown
        if random.random() < self.state.failure_probability:
            raise Exception(f"Shutdown failed for {self.state.name}")
        
        # Persist data
        if self.state.data_to_persist:
            self._data_persisted = True
        
        # Clean up resources
        if self.state.resources_allocated:
            self._resources_cleaned = True
            self.state.resources_allocated.clear()
        
        self.state.work_in_progress = False
    
    async def force_shutdown(self) -> None:
        """Force shutdown - should always succeed."""
        self._force_shutdown_called = True
        self.state.is_running = False
        self.state.work_in_progress = False
        
        # Force cleanup without persistence
        self.state.resources_allocated.clear()
    
    async def is_ready_for_shutdown(self) -> bool:
        """Check if component is ready for shutdown."""
        return not self.state.work_in_progress
    
    def verify_shutdown_completeness(self) -> Dict[str, bool]:
        """Verify that shutdown was completed properly."""
        return {
            'prepare_called': self._prepare_called,
            'shutdown_called': self._shutdown_called or self._force_shutdown_called,
            'data_persisted': self._data_persisted or self._force_shutdown_called,
            'resources_cleaned': self._resources_cleaned or len(self.state.resources_allocated) == 0,
            'not_running': not self.state.is_running,
            'no_work_in_progress': not self.state.work_in_progress
        }


class ShutdownCompletenessStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for graceful shutdown completeness.
    
    This validates that shutdown procedures complete successfully
    under various conditions and component configurations.
    """
    
    def __init__(self):
        super().__init__()
        self.shutdown_manager: Optional[ShutdownManager] = None
        self.components: Dict[str, PropertyTestComponent] = {}
        self.shutdown_config: Optional[ShutdownConfig] = None
        self.shutdown_initiated = False
        self.temp_files: List[str] = []
    
    @initialize()
    def setup_shutdown_manager(self):
        """Initialize shutdown manager with random configuration."""
        self.shutdown_config = ShutdownConfig(
            graceful_timeout_seconds=10,
            state_persistence_timeout_seconds=5,
            persist_state=True,
            cleanup_temp_files=True,
            close_db_connections=True
        )
        
        self.shutdown_manager = ShutdownManager(self.shutdown_config)
        self.components = {}
        self.shutdown_initiated = False
    
    @rule(
        component_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        shutdown_duration=st.floats(min_value=0.01, max_value=2.0),
        failure_probability=st.floats(min_value=0.0, max_value=0.3),
        has_data=st.booleans(),
        has_resources=st.booleans()
    )
    def add_component(self, component_name: str, shutdown_duration: float, 
                     failure_probability: float, has_data: bool, has_resources: bool):
        """Add a component with specific characteristics."""
        assume(not self.shutdown_initiated)
        assume(component_name not in self.components)
        assume(len(self.components) < 10)  # Limit component count
        
        # Create component state
        state = MockComponentState(
            name=component_name,
            shutdown_duration=shutdown_duration,
            failure_probability=failure_probability,
            data_to_persist={'key': 'value'} if has_data else {},
            resources_allocated={'resource1', 'resource2'} if has_resources else set()
        )
        
        # Create and add component
        component = PropertyTestComponent(state)
        self.components[component_name] = component
        self.shutdown_manager.add_component(component)
    
    @rule()
    async def initiate_graceful_shutdown(self):
        """Initiate graceful shutdown and verify completeness."""
        assume(not self.shutdown_initiated)
        assume(len(self.components) > 0)
        
        self.shutdown_initiated = True
        
        # Create temporary files to test cleanup
        if self.shutdown_config.cleanup_temp_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(b"test data")
            temp_file.close()
            self.temp_files.append(temp_file.name)
        
        # Execute shutdown
        await self.shutdown_manager.initiate_shutdown()
        await self.shutdown_manager.wait_for_shutdown()
        
        # Verify shutdown completeness
        await self._verify_shutdown_completeness()
    
    @rule()
    async def initiate_force_shutdown(self):
        """Initiate force shutdown and verify completeness."""
        assume(not self.shutdown_initiated)
        assume(len(self.components) > 0)
        
        self.shutdown_initiated = True
        
        # Execute force shutdown
        await self.shutdown_manager.force_shutdown()
        
        # Verify force shutdown completeness
        await self._verify_force_shutdown_completeness()
    
    async def _verify_shutdown_completeness(self):
        """Verify that graceful shutdown completed properly."""
        status = self.shutdown_manager.get_status()
        
        # Verify shutdown reached completion or handled failures appropriately
        assert status.phase in [ShutdownPhase.COMPLETED, ShutdownPhase.FAILED], (
            f"Shutdown should reach completion or failure, got {status.phase}"
        )
        
        # If shutdown completed successfully, verify all components
        if status.phase == ShutdownPhase.COMPLETED:
            self._verify_all_components_shutdown_properly()
        
        # Verify validation report accuracy
        validation_report = await self.shutdown_manager.validate_shutdown()
        self._verify_validation_report_accuracy(validation_report, status)
        
        # Verify state persistence if enabled
        if self.shutdown_config.persist_state:
            self._verify_state_persistence()
    
    async def _verify_force_shutdown_completeness(self):
        """Verify that force shutdown completed properly."""
        status = self.shutdown_manager.get_status()
        
        # Force shutdown should always reach failed state
        assert status.phase == ShutdownPhase.FAILED, (
            f"Force shutdown should reach failed state, got {status.phase}"
        )
        assert status.force_shutdown, "Force shutdown flag should be set"
        
        # All components should be marked as shutdown
        for component_name in self.components.keys():
            assert status.components_shutdown.get(component_name, False), (
                f"Component {component_name} should be marked as shutdown after force shutdown"
            )
    
    def _verify_all_components_shutdown_properly(self):
        """Verify that all components completed shutdown properly."""
        for component_name, component in self.components.items():
            completeness = component.verify_shutdown_completeness()
            
            # All aspects of shutdown should be complete
            for aspect, is_complete in completeness.items():
                assert is_complete, (
                    f"Component {component_name} failed to complete {aspect} during shutdown"
                )
    
    def _verify_validation_report_accuracy(self, report: ShutdownValidationReport, 
                                         status: ShutdownStatus):
        """Verify that validation report accurately reflects shutdown state."""
        # Report validity should match actual shutdown success
        expected_validity = (
            status.phase == ShutdownPhase.COMPLETED and 
            not status.force_shutdown and
            len(status.errors) == 0
        )
        
        assert report.is_valid == expected_validity, (
            f"Validation report validity mismatch: expected {expected_validity}, got {report.is_valid}"
        )
        
        # Check individual validation aspects
        if 'shutdown_completed' in report.checks:
            expected_completion = status.phase == ShutdownPhase.COMPLETED and not status.force_shutdown
            assert report.checks['shutdown_completed'] == expected_completion
        
        if 'all_components_shutdown' in report.checks:
            expected_all_shutdown = all(status.components_shutdown.values())
            assert report.checks['all_components_shutdown'] == expected_all_shutdown
        
        if 'no_shutdown_errors' in report.checks:
            expected_no_errors = len(status.errors) == 0
            assert report.checks['no_shutdown_errors'] == expected_no_errors
    
    def _verify_state_persistence(self):
        """Verify that state was properly persisted."""
        try:
            # Check if state file exists and is valid
            with open("intelligence_layer_state.json", 'r') as f:
                state_data = json.load(f)
            
            # Verify required fields are present
            required_fields = ['timestamp', 'components', 'shutdown_status']
            for field in required_fields:
                assert field in state_data, f"Required field {field} missing from persisted state"
            
            # Verify component list matches
            expected_components = list(self.components.keys())
            assert set(state_data['components']) == set(expected_components), (
                "Persisted component list doesn't match actual components"
            )
            
        except FileNotFoundError:
            pytest.fail("State persistence enabled but no state file found")
        except json.JSONDecodeError:
            pytest.fail("State file exists but contains invalid JSON")
    
    @invariant()
    def shutdown_manager_state_is_consistent(self):
        """Verify shutdown manager state remains consistent."""
        if self.shutdown_manager is None:
            return
        
        status = self.shutdown_manager.get_status()
        
        # Phase should be valid
        assert isinstance(status.phase, ShutdownPhase)
        
        # Timestamps should be logical
        if status.completed_at is not None:
            assert status.completed_at >= status.started_at
        
        # Component shutdown tracking should match registered components
        for component_name in self.components.keys():
            assert component_name in status.components_shutdown
    
    def teardown(self):
        """Clean up test artifacts."""
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass
        
        # Clean up state file
        try:
            os.unlink("intelligence_layer_state.json")
        except FileNotFoundError:
            pass


@given(
    component_count=st.integers(min_value=1, max_value=8),
    timeout_seconds=st.integers(min_value=5, max_value=20),
    persist_state=st.booleans(),
    component_failure_rate=st.floats(min_value=0.0, max_value=0.2)
)
@settings(max_examples=30, deadline=10000)
def test_shutdown_completeness_under_various_conditions(
    component_count: int, timeout_seconds: int, persist_state: bool, component_failure_rate: float
):
    """
    Test that shutdown completes properly under various system conditions.
    
    Property: Shutdown procedures should complete without data loss or corruption
    regardless of system configuration and component behavior.
    """
    async def run_test():
        config = ShutdownConfig(
            graceful_timeout_seconds=timeout_seconds,
            persist_state=persist_state,
            cleanup_temp_files=True,
            close_db_connections=True
        )
        
        manager = ShutdownManager(config)
        components = []
        
        # Create components with varying characteristics
        for i in range(component_count):
            # Use fixed values instead of .example()
            shutdown_duration = 0.1 + (i * 0.1)  # Vary duration based on index
            
            state = MockComponentState(
                name=f"component_{i}",
                shutdown_duration=shutdown_duration,
                failure_probability=component_failure_rate,
                data_to_persist={'data': f'component_{i}_data'},
                resources_allocated={f'resource_{i}_1', f'resource_{i}_2'}
            )
            
            component = PropertyTestComponent(state)
            components.append(component)
            manager.add_component(component)
        
        # Execute shutdown
        await manager.initiate_shutdown()
        await manager.wait_for_shutdown()
        
        # Verify completeness
        status = manager.get_status()
        validation_report = await manager.validate_shutdown()
        
        # Property: Shutdown should reach a definitive state
        assert status.phase in [ShutdownPhase.COMPLETED, ShutdownPhase.FAILED], (
            f"Shutdown must reach completion or failure, got {status.phase}"
        )
        
        # Property: All components should be accounted for
        for component in components:
            component_name = component.component_name()
            assert component_name in status.components_shutdown, (
                f"Component {component_name} not tracked in shutdown status"
            )
        
        # Property: Validation report should accurately reflect actual state
        expected_validity = (
            status.phase == ShutdownPhase.COMPLETED and 
            not status.force_shutdown and
            len(status.errors) == 0
        )
        assert validation_report.is_valid == expected_validity, (
            "Validation report validity doesn't match actual shutdown state"
        )
        
        # Property: If shutdown completed successfully, all components should be properly shutdown
        if status.phase == ShutdownPhase.COMPLETED:
            for component in components:
                completeness = component.verify_shutdown_completeness()
                assert all(completeness.values()), (
                    f"Component {component.component_name()} not properly shutdown: {completeness}"
                )
        
        # Property: State persistence should work if enabled
        if persist_state and status.phase == ShutdownPhase.COMPLETED:
            try:
                with open("intelligence_layer_state.json", 'r') as f:
                    state_data = json.load(f)
                assert 'timestamp' in state_data, "Persisted state missing timestamp"
                assert 'components' in state_data, "Persisted state missing components"
            except FileNotFoundError:
                pytest.fail("State persistence enabled but no state file created")
            finally:
                # Cleanup
                try:
                    os.unlink("intelligence_layer_state.json")
                except FileNotFoundError:
                    pass
    
    # Run the async test
    asyncio.run(run_test())


@given(
    force_shutdown_delay=st.floats(min_value=0.0, max_value=2.0),
    component_count=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=20, deadline=5000)
def test_force_shutdown_always_completes(force_shutdown_delay: float, component_count: int):
    """
    Test that force shutdown always completes regardless of component state.
    
    Property: Force shutdown should always complete and leave system in safe state.
    """
    async def run_test():
        manager = ShutdownManager()
        components = []
        
        # Create components that might be slow or stuck
        for i in range(component_count):
            state = MockComponentState(
                name=f"component_{i}",
                shutdown_duration=10.0,  # Very long shutdown time
                work_in_progress=True,   # Stuck with work
                resources_allocated={f'resource_{i}'}
            )
            
            component = PropertyTestComponent(state)
            components.append(component)
            manager.add_component(component)
        
        # Wait a bit then force shutdown
        if force_shutdown_delay > 0:
            await asyncio.sleep(force_shutdown_delay)
        
        await manager.force_shutdown()
        
        # Verify force shutdown completed
        status = manager.get_status()
        
        # Property: Force shutdown should always reach failed state
        assert status.phase == ShutdownPhase.FAILED, (
            f"Force shutdown should reach failed state, got {status.phase}"
        )
        assert status.force_shutdown, "Force shutdown flag should be set"
        
        # Property: All components should be marked as shutdown
        for component in components:
            component_name = component.component_name()
            assert status.components_shutdown.get(component_name, False), (
                f"Component {component_name} should be marked as shutdown"
            )
            
            # Property: Force shutdown should clean up resources
            assert len(component.state.resources_allocated) == 0, (
                f"Component {component_name} should have cleaned up resources"
            )
    
    # Run the async test
    asyncio.run(run_test())


@given(
    data_sizes=st.lists(
        st.integers(min_value=1, max_value=1000),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=15, deadline=3000)
def test_data_persistence_integrity(data_sizes: List[int]):
    """
    Test that data persistence maintains integrity during shutdown.
    
    Property: Data should be persisted completely and accurately during shutdown.
    """
    async def run_test():
        config = ShutdownConfig(persist_state=True)
        manager = ShutdownManager(config)
        
        # Create test data of various sizes
        test_data = {}
        for i, size in enumerate(data_sizes):
            test_data[f'component_{i}'] = 'x' * size
        
        # Mock the state saving to capture what gets persisted
        persisted_data = {}
        
        async def mock_save_state():
            nonlocal persisted_data
            persisted_data = {
                "timestamp": datetime.now().isoformat(),
                "components": list(test_data.keys()),
                "test_data": test_data,
                "shutdown_status": manager.get_status().__dict__
            }
            
            # Simulate actual file writing
            with open("intelligence_layer_state.json", 'w') as f:
                json.dump(persisted_data, f, default=str)
        
        # Patch the save method
        with patch.object(manager, '_save_state', side_effect=mock_save_state):
            await manager.initiate_shutdown()
            await manager.wait_for_shutdown()
        
        # Verify data integrity
        assert persisted_data is not None, "No data was persisted"
        assert 'test_data' in persisted_data, "Test data not found in persisted state"
        
        # Property: All data should be persisted accurately
        for component_name, original_data in test_data.items():
            assert component_name in persisted_data['test_data'], (
                f"Component {component_name} data not persisted"
            )
            assert persisted_data['test_data'][component_name] == original_data, (
                f"Data corruption detected for component {component_name}"
            )
        
        # Cleanup
        try:
            os.unlink("intelligence_layer_state.json")
        except FileNotFoundError:
            pass
    
    # Run the async test
    asyncio.run(run_test())


# Test runner for the stateful machine
TestShutdownCompleteness = ShutdownCompletenessStateMachine.TestCase


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])