"""
Tests for comprehensive error handling and monitoring infrastructure.

This module tests the error handling framework and monitoring system
following the patterns established in the knowledge-ingestion system.

Requirements: 20.1, 20.2, 20.3, 20.4, 22.1, 22.2, 22.4, 22.7
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import shutil

from nautilus_integration.core.error_handling import (
    ErrorRecoveryManager,
    GracefulDegradationManager,
    CircuitBreaker,
    RateLimiter,
    ErrorContext,
    RetryConfig,
    CircuitBreakerConfig,
    GracefulDegradationConfig,
    NautilusErrorType,
    ErrorSeverity,
    CheckpointData,
    CircuitBreakerOpenError,
    DataValidator,
    NetworkHandler,
    classify_error,
    determine_error_severity,
    calculate_backoff_delay,
    retry_with_backoff,
    error_handling_context,
    setup_error_handling,
    setup_enhanced_error_handling,
    get_global_data_validator,
    get_global_network_handler,
    validate_and_process_market_data,
    validate_and_process_strategy_config,
    execute_network_request
)

from nautilus_integration.core.monitoring import (
    NautilusMonitor,
    MetricsCollector,
    LatencyTracker,
    HealthChecker,
    AlertManager,
    SystemMonitor,
    PerformanceMetric,
    HealthCheck,
    Alert,
    HealthStatus,
    MetricType,
    AlertSeverity,
    track_latency,
    record_latency,
    increment_counter,
    set_gauge,
    create_alert,
    setup_monitoring
)

from nautilus_integration.core.config import MonitoringConfig


class TestErrorClassification:
    """Test error classification functionality"""
    
    def test_classify_network_errors(self):
        """Test classification of network-related errors"""
        connection_error = ConnectionError("Connection failed")
        timeout_error = TimeoutError("Request timed out")
        
        assert classify_error(connection_error) == NautilusErrorType.NETWORK_ERROR
        assert classify_error(timeout_error) == NautilusErrorType.NETWORK_ERROR
    
    def test_classify_authentication_errors(self):
        """Test classification of authentication errors"""
        auth_error = PermissionError("Unauthorized access")
        
        assert classify_error(auth_error) == NautilusErrorType.AUTHENTICATION_ERROR
    
    def test_classify_component_specific_errors(self):
        """Test classification based on component context"""
        generic_error = ValueError("Invalid value")
        
        assert classify_error(generic_error, "nautilus_engine") == NautilusErrorType.NAUTILUS_ENGINE_ERROR
        assert classify_error(generic_error, "strategy_translation") == NautilusErrorType.STRATEGY_TRANSLATION_ERROR
        assert classify_error(generic_error, "signal_routing") == NautilusErrorType.SIGNAL_ROUTING_ERROR
        assert classify_error(generic_error, "f6_integration") == NautilusErrorType.F6_INTEGRATION_ERROR
        assert classify_error(generic_error, "f8_risk") == NautilusErrorType.F8_RISK_ERROR
    
    def test_determine_error_severity(self):
        """Test error severity determination"""
        # Critical errors
        assert determine_error_severity(
            NautilusErrorType.RISK_LIMIT_BREACH, "f8_risk"
        ) == ErrorSeverity.CRITICAL
        
        # High severity errors
        assert determine_error_severity(
            NautilusErrorType.NAUTILUS_ENGINE_ERROR, "nautilus_engine"
        ) == ErrorSeverity.HIGH
        
        # Medium severity errors
        assert determine_error_severity(
            NautilusErrorType.SIGNAL_ROUTING_ERROR, "signal_routing"
        ) == ErrorSeverity.MEDIUM
        
        # Context-based severity adjustment
        assert determine_error_severity(
            NautilusErrorType.NETWORK_ERROR, "execution", 
            context={'is_live_trading': True}
        ) == ErrorSeverity.HIGH


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_basic_functionality(self):
        """Test basic rate limiter operations"""
        limiter = RateLimiter(rate=2.0, burst=3, component="test")
        
        # Should allow burst requests
        assert limiter.acquire(1) == True
        assert limiter.acquire(1) == True
        assert limiter.acquire(1) == True
        
        # Should deny additional requests
        assert limiter.acquire(1) == False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_token(self):
        """Test waiting for tokens"""
        limiter = RateLimiter(rate=10.0, burst=1, component="test")
        
        # Consume the burst token
        assert limiter.acquire(1) == True
        
        # Wait for token should complete quickly with high rate
        start_time = time.time()
        await limiter.wait_for_token(1)
        elapsed = time.time() - start_time
        
        # Should complete in less than 0.2 seconds with rate=10.0
        assert elapsed < 0.2


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation"""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
        breaker = CircuitBreaker(config, "test_component")
        
        # Normal operation should work
        result = await breaker.call(lambda: "success")
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        breaker = CircuitBreaker(config, "test_component")
        
        # Cause failures to open circuit breaker
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(lambda: "should_fail")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        breaker = CircuitBreaker(config, "test_component")
        
        # Cause failures to open circuit breaker
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should allow one attempt in half-open state
        result = await breaker.call(lambda: "recovered")
        assert result == "recovered"


class TestErrorRecoveryManager:
    """Test error recovery manager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.recovery_manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_save_and_load(self):
        """Test checkpoint save and load functionality"""
        operation_id = "test_operation"
        checkpoint_data = CheckpointData(
            operation_id=operation_id,
            timestamp=datetime.now(),
            state={"key": "value"},
            completed_items=["item1", "item2"],
            failed_items=["item3"],
            strategy_states={"strategy1": {"param": "value"}},
            position_states={"position1": {"size": 100}}
        )
        
        # Save checkpoint
        self.recovery_manager.save_checkpoint(operation_id, checkpoint_data)
        
        # Load checkpoint
        loaded_data = self.recovery_manager.load_checkpoint(operation_id)
        
        assert loaded_data is not None
        assert loaded_data.operation_id == operation_id
        assert loaded_data.state == {"key": "value"}
        assert loaded_data.completed_items == ["item1", "item2"]
        assert loaded_data.failed_items == ["item3"]
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup"""
        operation_id = "test_cleanup"
        checkpoint_data = CheckpointData(
            operation_id=operation_id,
            timestamp=datetime.now(),
            state={},
            completed_items=[],
            failed_items=[]
        )
        
        # Save and cleanup checkpoint
        self.recovery_manager.save_checkpoint(operation_id, checkpoint_data)
        self.recovery_manager.cleanup_checkpoint(operation_id)
        
        # Should not be able to load after cleanup
        loaded_data = self.recovery_manager.load_checkpoint(operation_id)
        assert loaded_data is None


class TestGracefulDegradationManager:
    """Test graceful degradation manager functionality"""
    
    @pytest.mark.asyncio
    async def test_component_failure_handling(self):
        """Test component failure handling"""
        config = GracefulDegradationConfig(
            fallback_to_f7_engine=True,
            disable_live_trading=True,
            notify_operators=True
        )
        manager = GracefulDegradationManager(config)
        
        # Simulate component failure
        error = RuntimeError("Component failed")
        await manager.handle_component_failure("nautilus_engine", error)
        
        # Check that component is marked as degraded
        assert manager.is_component_degraded("nautilus_engine")
        
        degraded_components = manager.get_degraded_components()
        assert "nautilus_engine" in degraded_components


class TestRetryMechanism:
    """Test retry mechanism functionality"""
    
    def test_backoff_delay_calculation(self):
        """Test backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False
        )
        
        # Test exponential backoff
        delay1 = calculate_backoff_delay(0, config)
        delay2 = calculate_backoff_delay(1, config)
        delay3 = calculate_backoff_delay(2, config)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test successful retry operation"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        error_context = ErrorContext(
            error_id="test",
            error_type=NautilusErrorType.NETWORK_ERROR,
            severity=ErrorSeverity.MEDIUM,
            component="test",
            operation="test_op",
            timestamp=datetime.now(),
            attempt_count=0,
            max_attempts=3,
            error_message=""
        )
        
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await retry_with_backoff(test_func, config, error_context)
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_failure(self):
        """Test retry operation that ultimately fails"""
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        error_context = ErrorContext(
            error_id="test",
            error_type=NautilusErrorType.NETWORK_ERROR,
            severity=ErrorSeverity.MEDIUM,
            component="test",
            operation="test_op",
            timestamp=datetime.now(),
            attempt_count=0,
            max_attempts=2,
            error_message=""
        )
        
        async def failing_func():
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError):
            await retry_with_backoff(failing_func, config, error_context)


class TestErrorHandlingContext:
    """Test error handling context manager"""
    
    @pytest.mark.asyncio
    async def test_error_handling_context_success(self):
        """Test successful operation with error handling context"""
        async with error_handling_context(
            operation="test_operation",
            component="test_component"
        ) as context:
            assert "recovery_manager" in context
            assert "degradation_manager" in context
            assert "error_context" in context
            assert "correlation_id" in context
    
    @pytest.mark.asyncio
    async def test_error_handling_context_failure(self):
        """Test failed operation with error handling context"""
        with pytest.raises(ValueError):
            async with error_handling_context(
                operation="test_operation",
                component="test_component"
            ) as context:
                raise ValueError("Test error")


class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector(max_history=100)
    
    def test_counter_metrics(self):
        """Test counter metric operations"""
        self.collector.increment_counter("test_counter", "test_component", 5.0)
        self.collector.increment_counter("test_counter", "test_component", 3.0)
        
        value = self.collector.get_counter("test_counter", "test_component")
        assert value == 8.0
    
    def test_gauge_metrics(self):
        """Test gauge metric operations"""
        self.collector.set_gauge("test_gauge", "test_component", 42.0)
        
        value = self.collector.get_gauge("test_gauge", "test_component")
        assert value == 42.0
    
    def test_histogram_metrics(self):
        """Test histogram metric operations"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.collector.record_histogram("test_histogram", "test_component", value)
        
        stats = self.collector.get_histogram_stats("test_histogram", "test_component")
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
    
    def test_recent_metrics(self):
        """Test recent metrics retrieval"""
        # Record some metrics
        self.collector.increment_counter("recent_counter", "test_component", 1.0)
        self.collector.set_gauge("recent_gauge", "test_component", 10.0)
        
        recent_metrics = self.collector.get_recent_metrics("test_component", minutes=1)
        assert len(recent_metrics) == 2


class TestLatencyTracker:
    """Test latency tracking functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector()
        self.tracker = LatencyTracker(self.collector)
    
    def test_operation_tracking(self):
        """Test operation latency tracking"""
        operation_id = "test_op_123"
        component = "test_component"
        operation_name = "test_operation"
        
        # Start tracking
        self.tracker.start_operation(operation_id, component, operation_name)
        
        # Simulate some work
        time.sleep(0.01)
        
        # End tracking
        latency_ns = self.tracker.end_operation(operation_id, component, operation_name)
        
        assert latency_ns > 0
        assert latency_ns >= 10_000_000  # At least 10ms in nanoseconds
    
    def test_latency_context_manager(self):
        """Test latency tracking context manager"""
        with track_latency("test_operation", "test_component") as operation_id:
            assert operation_id is not None
            time.sleep(0.01)
        
        # Check that metric was recorded
        stats = self.collector.get_histogram_stats("test_operation_latency_ms", "test_component")
        assert stats["count"] == 1
        assert stats["min"] > 0


class TestHealthChecker:
    """Test health checking functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector()
        self.checker = HealthChecker(self.collector)
    
    @pytest.mark.asyncio
    async def test_health_check_registration_and_execution(self):
        """Test health check registration and execution"""
        async def test_health_check():
            return {
                'status': 'healthy',
                'message': 'Component is running',
                'details': {'uptime': '1h'}
            }
        
        self.checker.register_health_check("test_component", test_health_check)
        
        result = await self.checker.check_component_health("test_component")
        
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Component is running"
        assert result.details == {'uptime': '1h'}
        assert result.response_time_ns is not None
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure handling"""
        async def failing_health_check():
            raise RuntimeError("Health check failed")
        
        self.checker.register_health_check("failing_component", failing_health_check)
        
        result = await self.checker.check_component_health("failing_component")
        
        assert result.component == "failing_component"
        assert result.status == HealthStatus.CRITICAL
        assert "Health check failed with exception" in result.message
    
    @pytest.mark.asyncio
    async def test_overall_health_status(self):
        """Test overall health status calculation"""
        # Register healthy component
        self.checker.register_health_check("healthy_component", lambda: True)
        
        # Register unhealthy component
        self.checker.register_health_check("unhealthy_component", lambda: False)
        
        # Check all components
        await self.checker.check_all_components()
        
        # Overall health should be critical due to one unhealthy component
        overall_health = self.checker.get_overall_health()
        assert overall_health == HealthStatus.CRITICAL


class TestAlertManager:
    """Test alert management functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector()
        self.manager = AlertManager(self.collector)
    
    @pytest.mark.asyncio
    async def test_alert_creation_and_resolution(self):
        """Test alert creation and resolution"""
        # Create alert
        alert = await self.manager.create_alert(
            AlertSeverity.WARNING,
            "test_component",
            "Test Alert",
            "This is a test alert"
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.component == "test_component"
        assert alert.title == "Test Alert"
        assert not alert.resolved
        
        # Check active alerts
        active_alerts = self.manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == alert.alert_id
        
        # Resolve alert
        resolved = await self.manager.resolve_alert(alert.alert_id, "Issue resolved")
        assert resolved == True
        
        # Check no active alerts
        active_alerts = self.manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    def test_notification_handler_registration(self):
        """Test notification handler registration"""
        notifications_received = []
        
        def test_handler(alert):
            notifications_received.append(alert)
        
        self.manager.register_notification_handler(test_handler)
        
        # Notification handler should be registered
        assert len(self.manager.notification_handlers) == 1


class TestSystemMonitor:
    """Test system monitoring functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = MetricsCollector()
        self.monitor = SystemMonitor(self.collector)
    
    @pytest.mark.asyncio
    async def test_system_monitoring_start_stop(self):
        """Test system monitoring start and stop"""
        # Start monitoring
        await self.monitor.start_monitoring(interval_seconds=0.1)
        
        # Wait a bit for metrics collection
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await self.monitor.stop_monitoring()
        
        # Check that system metrics were collected
        cpu_metric = self.collector.get_gauge("cpu_percent", "system")
        assert cpu_metric is not None
        assert cpu_metric >= 0.0


class TestNautilusMonitor:
    """Test main monitoring system"""
    
    def setup_method(self):
        """Setup test environment"""
        config = MonitoringConfig(
            system_monitor_interval=1,
            health_check_interval=1.0
        )
        self.monitor = NautilusMonitor(config)
    
    @pytest.mark.asyncio
    async def test_monitor_start_stop(self):
        """Test monitor start and stop"""
        await self.monitor.start()
        
        # Wait a bit for initialization
        await asyncio.sleep(0.1)
        
        await self.monitor.stop()
    
    @pytest.mark.asyncio
    async def test_system_status_retrieval(self):
        """Test system status retrieval"""
        status = await self.monitor.get_system_status()
        
        assert "overall_health" in status
        assert "health_checks" in status
        assert "active_alerts" in status
        assert "recent_metrics" in status
        assert "timestamp" in status


class TestIntegration:
    """Test integration between error handling and monitoring"""
    
    @pytest.mark.asyncio
    async def test_error_handling_with_monitoring(self):
        """Test error handling integration with monitoring"""
        # Setup monitoring
        monitor = setup_monitoring()
        
        # Setup error handling
        setup_error_handling()
        
        # Test operation with both systems
        async with error_handling_context(
            operation="integration_test",
            component="test_component"
        ) as context:
            # Record some metrics
            increment_counter("test_operations", "test_component")
            set_gauge("test_value", "test_component", 42.0)
            
            with track_latency("test_operation", "test_component"):
                await asyncio.sleep(0.01)
        
        # Verify metrics were recorded
        counter_value = monitor.metrics_collector.get_counter("test_operations", "test_component")
        assert counter_value == 1.0
        
        gauge_value = monitor.metrics_collector.get_gauge("test_value", "test_component")
        assert gauge_value == 42.0
        
        latency_stats = monitor.metrics_collector.get_histogram_stats(
            "test_operation_latency_ms", "test_component"
        )
        assert latency_stats["count"] == 1


if __name__ == "__main__":
    pytest.main([__file__])


class TestDataValidator:
    """Test data validation functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.recovery_manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir)
        self.validator = DataValidator(self.recovery_manager)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_market_data_validation_success(self):
        """Test successful market data validation"""
        # Register validation rule
        def validate_test_data(data):
            return isinstance(data, dict) and 'price' in data
        
        self.validator.register_validation_rule('test_data', validate_test_data)
        
        # Test valid data
        valid_data = {'price': 100.0, 'volume': 1000}
        result = self.validator.validate_market_data(valid_data, 'test_data')
        
        assert result == True
        
        # Check metrics
        metrics = self.validator.get_validation_metrics()
        assert metrics['test_data']['total_validated'] == 1
        assert metrics['test_data']['validation_errors'] == 0
    
    def test_market_data_validation_failure(self):
        """Test market data validation failure"""
        # Register validation rule
        def validate_test_data(data):
            return isinstance(data, dict) and 'price' in data
        
        self.validator.register_validation_rule('test_data', validate_test_data)
        
        # Test invalid data
        invalid_data = {'volume': 1000}  # Missing 'price'
        result = self.validator.validate_market_data(invalid_data, 'test_data')
        
        assert result == False
        
        # Check metrics
        metrics = self.validator.get_validation_metrics()
        assert metrics['test_data']['total_validated'] == 1
        assert metrics['test_data']['validation_errors'] == 1
    
    def test_strategy_configuration_validation(self):
        """Test strategy configuration validation"""
        # Test valid configuration
        valid_config = {
            'strategy_id': 'test_strategy',
            'strategy_type': 'momentum',
            'parameters': {'lookback': 20, 'threshold': 0.02}
        }
        
        result = self.validator.validate_strategy_configuration(valid_config)
        assert result == True
        
        # Test invalid configuration (missing required field)
        invalid_config = {
            'strategy_id': 'test_strategy',
            'parameters': {'lookback': 20}
            # Missing 'strategy_type'
        }
        
        result = self.validator.validate_strategy_configuration(invalid_config)
        assert result == False
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted data that causes exceptions"""
        # Register validation rule that will cause exception
        def problematic_validation(data):
            return data['nonexistent_key']  # Will raise KeyError
        
        self.validator.register_validation_rule('problematic_data', problematic_validation)
        
        # Test with data that will cause exception
        test_data = {'some_key': 'some_value'}
        result = self.validator.validate_market_data(test_data, 'problematic_data')
        
        assert result == False
        
        # Check that corrupted data was counted
        metrics = self.validator.get_validation_metrics()
        assert metrics['problematic_data']['corrupted_skipped'] == 1


class TestNetworkHandler:
    """Test network handling functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.recovery_manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir)
        self.handler = NetworkHandler(self.recovery_manager)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_venue_configuration(self):
        """Test venue configuration registration"""
        venue_config = {
            'timeout': 10.0,
            'max_retries': 5,
            'retry_delay': 0.5
        }
        
        self.handler.register_venue_config('test_venue', venue_config)
        
        assert 'test_venue' in self.handler.venue_configs
        assert self.handler.venue_configs['test_venue']['timeout'] == 10.0
        assert self.handler.venue_configs['test_venue']['max_retries'] == 5
    
    @pytest.mark.asyncio
    async def test_successful_network_operation(self):
        """Test successful network operation"""
        # Register venue
        self.handler.register_venue_config('test_venue', {'max_retries': 2})
        
        # Mock successful operation
        async def mock_operation():
            return "success"
        
        result = await self.handler.execute_network_operation(
            'test_venue', 'test_operation', mock_operation
        )
        
        assert result == "success"
        
        # Check metrics
        metrics = self.handler.get_network_metrics()
        assert metrics['test_venue']['total_requests'] == 1
        assert metrics['test_venue']['successful_requests'] == 1
        assert metrics['test_venue']['failed_requests'] == 0
    
    @pytest.mark.asyncio
    async def test_network_operation_with_retries(self):
        """Test network operation that succeeds after retries"""
        # Register venue
        self.handler.register_venue_config('test_venue', {
            'max_retries': 3,
            'retry_delay': 0.1
        })
        
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await self.handler.execute_network_operation(
            'test_venue', 'test_operation', mock_operation
        )
        
        assert result == "success"
        assert call_count == 2
        
        # Check metrics
        metrics = self.handler.get_network_metrics()
        assert metrics['test_venue']['successful_requests'] == 1
    
    @pytest.mark.asyncio
    async def test_network_operation_failure(self):
        """Test network operation that fails after all retries"""
        # Register venue
        self.handler.register_venue_config('test_venue', {
            'max_retries': 2,
            'retry_delay': 0.1
        })
        
        async def failing_operation():
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError):
            await self.handler.execute_network_operation(
                'test_venue', 'test_operation', failing_operation
            )
        
        # Check that operation was added to dead letter queue
        dead_letters = self.recovery_manager.get_dead_letter_queue('network_test_venue')
        assert len(dead_letters) == 1
        assert dead_letters[0]['operation'] == 'test_operation'
    
    def test_venue_health_assessment(self):
        """Test venue health assessment"""
        # Initially no metrics
        health = self.handler.get_venue_health('unknown_venue')
        assert health['status'] == 'unknown'
        
        # Add some metrics
        venue = 'test_venue'
        self.handler.network_metrics[venue] = {
            'total_requests': 100,
            'successful_requests': 95,
            'failed_requests': 5,
            'retry_attempts': 10,
            'circuit_breaker_trips': 0
        }
        
        health = self.handler.get_venue_health(venue)
        assert health['status'] == 'healthy'  # 95% success rate
        assert health['success_rate'] == 0.95


class TestEnhancedErrorRecoveryManager:
    """Test enhanced error recovery manager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.recovery_manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_component_recovery_strategy_registration(self):
        """Test component recovery strategy registration"""
        def test_recovery_strategy(error, context):
            return True
        
        self.recovery_manager.register_component_recovery_strategy(
            'test_component', test_recovery_strategy
        )
        
        assert 'test_component' in self.recovery_manager.component_recovery_strategies
    
    @pytest.mark.asyncio
    async def test_component_recovery_execution(self):
        """Test component recovery strategy execution"""
        recovery_called = False
        
        async def test_recovery_strategy(error, context):
            nonlocal recovery_called
            recovery_called = True
            return True
        
        self.recovery_manager.register_component_recovery_strategy(
            'test_component', test_recovery_strategy
        )
        
        error = RuntimeError("Test error")
        result = await self.recovery_manager.execute_component_recovery(
            'test_component', error, {'test': 'context'}
        )
        
        assert result == True
        assert recovery_called == True
    
    def test_dead_letter_queue_operations(self):
        """Test dead letter queue operations"""
        component = 'test_component'
        operation = 'test_operation'
        data = {'key': 'value'}
        error = RuntimeError("Test error")
        
        # Add to dead letter queue
        self.recovery_manager.add_to_dead_letter_queue(component, operation, data, error)
        
        # Check queue contents
        queue = self.recovery_manager.get_dead_letter_queue(component)
        assert len(queue) == 1
        assert queue[0]['operation'] == operation
        assert queue[0]['error_type'] == 'RuntimeError'
        
        # Clear queue
        cleared_count = self.recovery_manager.clear_dead_letter_queue(component)
        assert cleared_count == 1
        
        # Check queue is empty
        queue = self.recovery_manager.get_dead_letter_queue(component)
        assert len(queue) == 0


class TestEnhancedIntegration:
    """Test enhanced integration functionality"""
    
    @pytest.mark.asyncio
    async def test_enhanced_error_handling_setup(self):
        """Test enhanced error handling setup"""
        setup_enhanced_error_handling()
        
        # Test that all components are available
        validator = get_global_data_validator()
        assert validator is not None
        
        handler = get_global_network_handler()
        assert handler is not None
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions for enhanced error handling"""
        setup_enhanced_error_handling()
        
        # Test market data validation
        valid_data = {
            'open': 100.0, 'high': 105.0, 'low': 95.0, 
            'close': 102.0, 'volume': 1000, 'timestamp': '2023-01-01T00:00:00Z'
        }
        
        result = await validate_and_process_market_data(valid_data, 'ohlcv_data')
        assert result == True
        
        # Test strategy configuration validation
        valid_config = {
            'strategy_id': 'test_strategy',
            'strategy_type': 'momentum',
            'parameters': {'lookback': 20}
        }
        
        result = await validate_and_process_strategy_config(valid_config)
        assert result == True
    
    @pytest.mark.asyncio
    async def test_network_request_convenience_function(self):
        """Test network request convenience function"""
        setup_enhanced_error_handling()
        
        # Register venue configuration
        handler = get_global_network_handler()
        handler.register_venue_config('test_venue', {'max_retries': 1})
        
        # Test successful request
        async def mock_request():
            return "response"
        
        result = await execute_network_request(
            'test_venue', 'test_request', mock_request
        )
        
        assert result == "response"


class TestErrorHandlingRequirements:
    """Test compliance with specific error handling requirements"""
    
    @pytest.mark.asyncio
    async def test_requirement_20_1_exponential_backoff(self):
        """Test Requirement 20.1: Exponential backoff and retry mechanisms"""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False
        )
        
        # Test exponential backoff calculation
        delay1 = calculate_backoff_delay(0, config)
        delay2 = calculate_backoff_delay(1, config)
        delay3 = calculate_backoff_delay(2, config)
        
        assert delay1 == 0.1
        assert delay2 == 0.2
        assert delay3 == 0.4
    
    @pytest.mark.asyncio
    async def test_requirement_20_2_network_retry_with_circuit_breaker(self):
        """Test Requirement 20.2: Network retry with circuit breaker patterns"""
        setup_enhanced_error_handling()
        handler = get_global_network_handler()
        
        # Register venue with circuit breaker config
        handler.register_venue_config('test_venue', {
            'circuit_breaker_threshold': 2,
            'circuit_breaker_timeout': 1,
            'max_retries': 1
        })
        
        # Test that circuit breaker opens after failures
        async def failing_operation():
            raise ConnectionError("Network failure")
        
        # First two failures should open circuit breaker
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await handler.execute_network_operation(
                    'test_venue', 'test_op', failing_operation
                )
        
        # Third attempt should fail with circuit breaker open
        with pytest.raises(CircuitBreakerOpenError):
            await handler.execute_network_operation(
                'test_venue', 'test_op', failing_operation
            )
    
    @pytest.mark.asyncio
    async def test_requirement_20_3_corrupted_data_handling(self):
        """Test Requirement 20.3: Skip corrupted inputs with detailed logging"""
        setup_enhanced_error_handling()
        validator = get_global_data_validator()
        
        # Test corrupted market data handling
        corrupted_data = None  # This will cause validation to fail
        
        result = validator.validate_market_data(corrupted_data, 'ohlcv_data')
        assert result == False
        
        # Check that corrupted data was logged and skipped
        metrics = validator.get_validation_metrics()
        assert metrics['ohlcv_data']['corrupted_skipped'] >= 1
    
    @pytest.mark.asyncio
    async def test_requirement_20_4_checkpoint_resumption(self):
        """Test Requirement 20.4: Resume operations from checkpoints"""
        temp_dir = Path(tempfile.mkdtemp())
        recovery_manager = ErrorRecoveryManager(checkpoint_dir=temp_dir)
        
        try:
            # Create checkpoint
            operation_id = "test_resumption"
            checkpoint_data = CheckpointData(
                operation_id=operation_id,
                timestamp=datetime.now(),
                state={'progress': 50},
                completed_items=['item1', 'item2'],
                failed_items=['item3']
            )
            
            recovery_manager.save_checkpoint(operation_id, checkpoint_data)
            
            # Load checkpoint and verify resumption capability
            loaded_data = recovery_manager.load_checkpoint(operation_id)
            
            assert loaded_data is not None
            assert loaded_data.state['progress'] == 50
            assert len(loaded_data.completed_items) == 2
            assert len(loaded_data.failed_items) == 1
            
        finally:
            shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_requirement_20_5_structured_logging_with_correlation_ids(self):
        """Test Requirement 20.5: Structured logs with correlation IDs"""
        from nautilus_integration.core.logging import with_correlation_id, get_correlation_id
        
        # Test correlation ID tracking
        with with_correlation_id() as correlation_id:
            assert get_correlation_id() == correlation_id
            
            # Test that error context includes correlation ID
            error_context = ErrorContext(
                error_id="test",
                error_type=NautilusErrorType.NETWORK_ERROR,
                severity=ErrorSeverity.MEDIUM,
                component="test",
                operation="test_op",
                timestamp=datetime.now(),
                attempt_count=1,
                max_attempts=3,
                error_message="test error",
                correlation_id=correlation_id
            )
            
            assert error_context.correlation_id == correlation_id
    
    @pytest.mark.asyncio
    async def test_requirement_20_6_graceful_degradation_strategy_translation(self):
        """Test Requirement 20.6: Graceful degradation for strategy translation"""
        config = GracefulDegradationConfig(
            fallback_to_f7_engine=True,
            notify_operators=True
        )
        
        manager = GracefulDegradationManager(config)
        
        # Simulate strategy translation failure
        error = RuntimeError("Strategy translation failed")
        await manager.handle_component_failure("strategy_translation", error)
        
        # Verify component is marked as degraded
        assert manager.is_component_degraded("strategy_translation")
    
    @pytest.mark.asyncio
    async def test_requirement_20_7_signal_routing_dead_letter_queue(self):
        """Test Requirement 20.7: Signal routing with dead letter queues"""
        setup_enhanced_error_handling()
        recovery_manager = get_global_recovery_manager()
        
        # Simulate signal routing failure
        component = "signal_routing"
        operation = "deliver_signal"
        data = {'signal_id': 'test_signal', 'strategy_id': 'test_strategy'}
        error = ConnectionError("Signal delivery failed")
        
        recovery_manager.add_to_dead_letter_queue(component, operation, data, error)
        
        # Verify dead letter queue contains the failed operation
        dead_letters = recovery_manager.get_dead_letter_queue(component)
        assert len(dead_letters) == 1
        assert dead_letters[0]['operation'] == operation
        assert dead_letters[0]['data']['signal_id'] == 'test_signal'
    
    @pytest.mark.asyncio
    async def test_requirement_20_8_live_trading_halt_on_failure(self):
        """Test Requirement 20.8: Halt live trading on adapter failure"""
        config = GracefulDegradationConfig(
            disable_live_trading=True,
            preserve_existing_positions=True,
            notify_operators=True
        )
        
        manager = GracefulDegradationManager(config)
        
        # Simulate live trading adapter failure
        error = RuntimeError("Execution adapter failed")
        await manager.handle_component_failure("live_trading", error)
        
        # Verify live trading component is degraded
        assert manager.is_component_degraded("live_trading")
        
        degraded_components = manager.get_degraded_components()
        assert "live_trading" in degraded_components