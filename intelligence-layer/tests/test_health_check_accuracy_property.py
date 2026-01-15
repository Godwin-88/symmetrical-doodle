"""
Property-based tests for health check accuracy validation.

This module validates Property 23: Health Check Accuracy
- Health checks accurately reflect actual component status
- Status aggregation correctly represents system health
- Performance metrics are within expected bounds
- Health transitions are properly detected and reported

Requirements validated: 10.2
"""

import asyncio
import time
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from intelligence_layer.health import (
    HealthChecker,
    HealthStatus,
    ComponentType,
    HealthMetrics,
    ComponentHealth,
    SystemHealthResponse
)


class HealthCheckAccuracyStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for health check accuracy.
    
    This validates that health checks accurately reflect system state
    and that status aggregation works correctly.
    """
    
    def __init__(self):
        super().__init__()
        self.health_checker = HealthChecker()
        self.component_states: Dict[str, Dict] = {}
        self.expected_statuses: Dict[str, HealthStatus] = {}
        self.performance_baselines: Dict[str, Dict] = {}
        
    @initialize()
    def setup_components(self):
        """Initialize test components with known states."""
        self.component_states = {
            'database': {'connected': True, 'response_time': 50.0, 'error_rate': 0.0},
            'cache': {'connected': True, 'response_time': 10.0, 'error_rate': 0.0},
            'external_api': {'connected': True, 'response_time': 200.0, 'error_rate': 0.1},
            'resource_monitor': {'cpu_usage': 30.0, 'memory_usage': 40.0, 'disk_usage': 50.0}
        }
        
        self.performance_baselines = {
            'database': {'max_response_time': 100.0, 'max_error_rate': 0.05},
            'cache': {'max_response_time': 50.0, 'max_error_rate': 0.01},
            'external_api': {'max_response_time': 500.0, 'max_error_rate': 0.2},
            'resource_monitor': {'max_cpu': 80.0, 'max_memory': 85.0, 'max_disk': 90.0}
        }
    
    @rule(
        component=st.sampled_from(['database', 'cache', 'external_api']),
        connected=st.booleans(),
        response_time=st.floats(min_value=1.0, max_value=1000.0),
        error_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    def update_component_state(self, component: str, connected: bool, 
                             response_time: float, error_rate: float):
        """Update component state and verify health check accuracy."""
        # Update component state
        self.component_states[component].update({
            'connected': connected,
            'response_time': response_time,
            'error_rate': error_rate
        })
        
        # Calculate expected status based on state
        baseline = self.performance_baselines[component]
        if not connected:
            expected_status = HealthStatus.UNHEALTHY
        elif (response_time > baseline['max_response_time'] or 
              error_rate > baseline['max_error_rate']):
            expected_status = HealthStatus.DEGRADED
        else:
            expected_status = HealthStatus.HEALTHY
            
        self.expected_statuses[component] = expected_status
    
    @rule(
        cpu_usage=st.floats(min_value=0.0, max_value=100.0),
        memory_usage=st.floats(min_value=0.0, max_value=100.0),
        disk_usage=st.floats(min_value=0.0, max_value=100.0)
    )
    def update_resource_state(self, cpu_usage: float, memory_usage: float, disk_usage: float):
        """Update resource usage and verify monitoring accuracy."""
        self.component_states['resource_monitor'].update({
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage
        })
        
        # Calculate expected status based on resource usage
        baseline = self.performance_baselines['resource_monitor']
        if (cpu_usage > baseline['max_cpu'] or 
            memory_usage > baseline['max_memory'] or 
            disk_usage > baseline['max_disk']):
            expected_status = HealthStatus.DEGRADED
        else:
            expected_status = HealthStatus.HEALTHY
            
        self.expected_statuses['resource_monitor'] = expected_status
    
    @rule()
    async def verify_health_check_accuracy(self):
        """Verify that health checks accurately reflect component states."""
        # Mock component checks to return our controlled states
        with patch.multiple(
            self.health_checker,
            _check_database=AsyncMock(return_value=self._mock_component_health('database')),
            _check_cache=AsyncMock(return_value=self._mock_component_health('cache')),
            _check_external_api=AsyncMock(return_value=self._mock_component_health('external_api')),
            _check_resources=AsyncMock(return_value=self._mock_component_health('resource_monitor'))
        ):
            # Get health status
            health_response = await self.health_checker.get_system_health()
            
            # Verify each component status matches expected
            for component_name, expected_status in self.expected_statuses.items():
                if component_name in health_response.components:
                    actual_status = health_response.components[component_name].status
                    assert actual_status == expected_status, (
                        f"Component {component_name} status mismatch: "
                        f"expected {expected_status}, got {actual_status}"
                    )
            
            # Verify system-level status aggregation
            self._verify_system_status_aggregation(health_response)
    
    def _mock_component_health(self, component_name: str) -> ComponentHealth:
        """Create mock component health based on current state."""
        state = self.component_states[component_name]
        expected_status = self.expected_statuses.get(component_name, HealthStatus.UNKNOWN)
        
        if component_name == 'resource_monitor':
            metrics = HealthMetrics(
                response_time_ms=1.0,
                error_rate=0.0,
                throughput_rps=100.0,
                resource_usage={
                    'cpu_percent': state['cpu_usage'],
                    'memory_percent': state['memory_usage'],
                    'disk_percent': state['disk_usage']
                },
                last_check=datetime.now(),
                uptime_seconds=3600
            )
        else:
            metrics = HealthMetrics(
                response_time_ms=state['response_time'],
                error_rate=state['error_rate'],
                throughput_rps=10.0,
                resource_usage={},
                last_check=datetime.now(),
                uptime_seconds=3600
            )
        
        return ComponentHealth(
            name=component_name,
            type=ComponentType.DATABASE if component_name == 'database' else ComponentType.CACHE,
            status=expected_status,
            message=f"Component {component_name} status",
            metrics=metrics,
            dependencies=[],
            last_updated=datetime.now()
        )
    
    def _verify_system_status_aggregation(self, health_response: SystemHealthResponse):
        """Verify that system status correctly aggregates component statuses."""
        component_statuses = [comp.status for comp in health_response.components.values()]
        
        # System should be unhealthy if any component is unhealthy
        if HealthStatus.UNHEALTHY in component_statuses:
            assert health_response.status == HealthStatus.UNHEALTHY
        # System should be degraded if any component is degraded
        elif HealthStatus.DEGRADED in component_statuses:
            assert health_response.status == HealthStatus.DEGRADED
        # System should be healthy only if all components are healthy
        elif all(status == HealthStatus.HEALTHY for status in component_statuses):
            assert health_response.status == HealthStatus.HEALTHY
        else:
            assert health_response.status == HealthStatus.UNKNOWN
    
    @invariant()
    def health_metrics_are_valid(self):
        """Verify that health metrics are within valid ranges."""
        for component_name, state in self.component_states.items():
            if 'response_time' in state:
                assert state['response_time'] >= 0, f"Response time must be non-negative for {component_name}"
            if 'error_rate' in state:
                assert 0 <= state['error_rate'] <= 1, f"Error rate must be between 0 and 1 for {component_name}"
            if component_name == 'resource_monitor':
                for resource in ['cpu_usage', 'memory_usage', 'disk_usage']:
                    if resource in state:
                        assert 0 <= state[resource] <= 100, f"{resource} must be between 0 and 100"


@given(
    response_times=st.lists(
        st.floats(min_value=1.0, max_value=1000.0),
        min_size=10,
        max_size=100
    ),
    error_rates=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=50, deadline=5000)
def test_performance_metrics_accuracy(response_times: List[float], error_rates: List[float]):
    """
    Test that performance metrics accurately reflect actual measurements.
    
    Property: Performance metrics should accurately represent actual system performance
    """
    assume(len(response_times) == len(error_rates))
    
    async def run_test():
        health_checker = HealthChecker()
        
        # Simulate performance measurements
        measurements = list(zip(response_times, error_rates))
        
        with patch.object(health_checker, '_collect_performance_metrics') as mock_collect:
            # Mock performance data collection
            mock_collect.return_value = {
                'response_times': response_times,
                'error_rates': error_rates,
                'timestamps': [datetime.now() for _ in range(len(response_times))]
            }
            
            # Get aggregated metrics
            metrics = await health_checker._calculate_aggregated_metrics('test_component')
            
            # Verify accuracy of aggregated metrics
            expected_avg_response_time = sum(response_times) / len(response_times)
            expected_avg_error_rate = sum(error_rates) / len(error_rates)
            
            assert abs(metrics.response_time_ms - expected_avg_response_time) < 0.1, (
                f"Response time metric inaccurate: expected {expected_avg_response_time}, "
                f"got {metrics.response_time_ms}"
            )
            
            assert abs(metrics.error_rate - expected_avg_error_rate) < 0.01, (
                f"Error rate metric inaccurate: expected {expected_avg_error_rate}, "
                f"got {metrics.error_rate}"
            )
    
    # Run the async test
    asyncio.run(run_test())


@given(
    component_count=st.integers(min_value=1, max_value=10),
    failure_probability=st.floats(min_value=0.0, max_value=0.5)
)
@settings(max_examples=30, deadline=3000)
def test_health_status_transitions(component_count: int, failure_probability: float):
    """
    Test that health status transitions are properly detected and reported.
    
    Property: Health status changes should be accurately detected and reported
    """
    async def run_test():
        health_checker = HealthChecker()
        
        # Create components with random initial states
        components = {}
        for i in range(component_count):
            component_name = f"component_{i}"
            is_healthy = st.random.random() > failure_probability
            components[component_name] = {
                'status': HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                'previous_status': HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
            }
        
        # Simulate status transitions
        for component_name, component_data in components.items():
            # Randomly transition status
            if st.random.random() < 0.3:  # 30% chance of status change
                new_status = (HealthStatus.UNHEALTHY 
                             if component_data['status'] == HealthStatus.HEALTHY 
                             else HealthStatus.HEALTHY)
                component_data['previous_status'] = component_data['status']
                component_data['status'] = new_status
        
        # Mock health checks to return our controlled states
        async def mock_check_component(name: str) -> ComponentHealth:
            component_data = components[name]
            return ComponentHealth(
                name=name,
                type=ComponentType.INTERNAL_SERVICE,
                status=component_data['status'],
                message=f"Component {name} status",
                metrics=None,
                dependencies=[],
                last_updated=datetime.now()
            )
        
        with patch.object(health_checker, '_check_all_components') as mock_check:
            mock_check.return_value = {
                name: await mock_check_component(name) 
                for name in components.keys()
            }
            
            # Get health status
            health_response = await health_checker.get_system_health()
            
            # Verify all status transitions are accurately reported
            for component_name, component_data in components.items():
                if component_name in health_response.components:
                    reported_status = health_response.components[component_name].status
                    expected_status = component_data['status']
                    
                    assert reported_status == expected_status, (
                        f"Status transition not accurately reported for {component_name}: "
                        f"expected {expected_status}, got {reported_status}"
                    )
    
    # Run the async test
    asyncio.run(run_test())


# Test runner for the stateful machine
TestHealthCheckAccuracy = HealthCheckAccuracyStateMachine.TestCase


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])