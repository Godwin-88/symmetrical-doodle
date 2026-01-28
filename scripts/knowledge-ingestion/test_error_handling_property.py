"""
Property-Based Tests for Error Handling System

**Property 5: Graceful Error Handling**
**Validates: Requirements 1.5, 2.5, 10.1, 10.2, 10.3, 10.4, 10.5**

This module tests the comprehensive error handling system including:
- Rate limiting with exponential backoff
- Circuit breaker patterns for network failures
- Corrupted file handling with detailed error logging
- Partial failure recovery with checkpoint resumption
"""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import random

from services.error_handling import (
    ErrorHandler, RateLimiter, CircuitBreaker, CheckpointManager,
    ErrorType, CircuitState, RetryConfig, CircuitBreakerConfig, CheckpointData,
    with_retry, resumable_operation
)

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic_functionality(self):
        """Test basic rate limiting behavior"""
        limiter = RateLimiter(rate=2.0, burst=2)  # 2 tokens per second, burst of 2
        
        # Should be able to acquire initial burst
        assert await limiter.acquire(2) == True
        
        # Should not be able to acquire more immediately
        assert await limiter.acquire(1) == False
        
        # Wait for token replenishment
        await asyncio.sleep(0.6)  # Should get ~1 token
        assert await limiter.acquire(1) == True

    @given(
        rate=st.floats(min_value=0.1, max_value=10.0),
        burst=st.integers(min_value=1, max_value=10),
        requests=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_rate_limiter_property(self, rate, burst, requests):
        """Property: Rate limiter should respect configured limits"""
        limiter = RateLimiter(rate=rate, burst=burst)
        
        # Should be able to acquire up to burst tokens initially
        acquired = 0
        for _ in range(burst):
            if await limiter.acquire(1):
                acquired += 1
        
        assert acquired <= burst
        
        # After burst, should need to wait for token replenishment
        immediate_acquire = await limiter.acquire(1)
        if acquired == burst:
            assert immediate_acquire == False

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        breaker = CircuitBreaker(config)
        
        # Initially closed
        assert breaker.state == CircuitState.CLOSED
        
        # Simulate failures
        failing_func = AsyncMock(side_effect=Exception("Test error"))
        
        # First failure
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        assert breaker.state == CircuitState.CLOSED
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        assert breaker.state == CircuitState.OPEN
        
        # Should reject calls when open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await breaker.call(failing_func)

    @given(
        failure_threshold=st.integers(min_value=1, max_value=5),
        failures=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=3000)
    @pytest.mark.asyncio
    async def test_circuit_breaker_threshold_property(self, failure_threshold, failures):
        """Property: Circuit breaker should open after threshold failures"""
        config = CircuitBreakerConfig(failure_threshold=failure_threshold)
        breaker = CircuitBreaker(config)
        
        failing_func = AsyncMock(side_effect=Exception("Test error"))
        
        for i in range(min(failures, failure_threshold)):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
            
            if i + 1 < failure_threshold:
                assert breaker.state == CircuitState.CLOSED
            else:
                assert breaker.state == CircuitState.OPEN

class TestCheckpointManager:
    """Test checkpoint management functionality"""
    
    def test_checkpoint_save_load_cycle(self):
        """Test saving and loading checkpoints"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))
            
            # Create test data
            checkpoint_data = CheckpointData(
                operation_id="test_op",
                timestamp=time.time(),
                state={"progress": 50, "current_file": "test.pdf"},
                completed_items=["item1", "item2"],
                failed_items=["item3"],
                metadata={"test": "value"}
            )
            
            # Save checkpoint
            manager.save_checkpoint("test_op", checkpoint_data)
            
            # Load checkpoint
            loaded_data = manager.load_checkpoint("test_op")
            
            assert loaded_data is not None
            assert loaded_data.operation_id == "test_op"
            assert loaded_data.state == {"progress": 50, "current_file": "test.pdf"}
            assert loaded_data.completed_items == ["item1", "item2"]
            assert loaded_data.failed_items == ["item3"]

    @given(
        operation_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        state_items=st.dictionaries(
            st.text(min_size=1, max_size=20), 
            st.one_of(st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
            min_size=0, max_size=5
        ),
        completed_items=st.lists(st.text(min_size=1, max_size=20), max_size=10),
        failed_items=st.lists(st.text(min_size=1, max_size=20), max_size=10)
    )
    @settings(max_examples=20, deadline=3000)
    def test_checkpoint_persistence_property(self, operation_id, state_items, completed_items, failed_items):
        """Property: Checkpoint data should persist correctly across save/load cycles"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))
            
            checkpoint_data = CheckpointData(
                operation_id=operation_id,
                timestamp=time.time(),
                state=state_items,
                completed_items=completed_items,
                failed_items=failed_items
            )
            
            # Save and load
            manager.save_checkpoint(operation_id, checkpoint_data)
            loaded_data = manager.load_checkpoint(operation_id)
            
            assert loaded_data is not None
            assert loaded_data.operation_id == operation_id
            assert loaded_data.state == state_items
            assert loaded_data.completed_items == completed_items
            assert loaded_data.failed_items == failed_items

class TestErrorHandler:
    """Test comprehensive error handling system"""
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test retry logic with exponential backoff"""
        handler = ErrorHandler()
        call_count = 0
        
        async def failing_then_succeeding_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        config = RetryConfig(max_attempts=5, base_delay=0.1, jitter=False)
        
        start_time = time.time()
        result = await handler.with_retry(
            failing_then_succeeding_func, 
            "test_operation", 
            config
        )
        end_time = time.time()
        
        assert result == "success"
        assert call_count == 3
        # Should have waited for backoff (0.1 + 0.2 = 0.3 seconds minimum)
        assert end_time - start_time >= 0.25

    @pytest.mark.asyncio
    async def test_resumable_operation_context(self):
        """Test resumable operation with checkpoint support"""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(checkpoint_dir=Path(temp_dir))
            
            # First attempt - simulate failure
            operation_id = "test_resumable_op"
            initial_state = {"total_items": 10, "processed": 0}
            
            try:
                async with handler.resumable_operation(operation_id, initial_state) as ctx:
                    ctx['state']['processed'] = 5
                    ctx['completed_items'].extend(['item1', 'item2', 'item3'])
                    # Simulate failure
                    raise Exception("Simulated failure")
            except Exception:
                pass
            
            # Second attempt - should resume from checkpoint
            async with handler.resumable_operation(operation_id) as ctx:
                assert ctx['state']['processed'] == 5
                assert ctx['completed_items'] == ['item1', 'item2', 'item3']
                # Complete the operation
                ctx['state']['processed'] = 10

    @given(
        max_attempts=st.integers(min_value=1, max_value=5),
        failure_count=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.asyncio
    async def test_retry_attempts_property(self, max_attempts, failure_count):
        """Property: Retry should respect max_attempts configuration"""
        handler = ErrorHandler()
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count <= failure_count:
                raise Exception(f"Failure {call_count}")
            return "success"
        
        config = RetryConfig(max_attempts=max_attempts, base_delay=0.01, jitter=False)
        
        if failure_count < max_attempts:
            # Should succeed
            result = await handler.with_retry(failing_func, "test_op", config)
            assert result == "success"
            assert call_count == failure_count + 1
        else:
            # Should fail after max_attempts
            with pytest.raises(Exception):
                await handler.with_retry(failing_func, "test_op", config)
            assert call_count == max_attempts

class ErrorHandlingStateMachine(RuleBasedStateMachine):
    """Stateful testing for error handling system"""
    
    def __init__(self):
        super().__init__()
        self.handler = ErrorHandler()
        self.operations = {}
        self.call_counts = {}
    
    @initialize()
    def setup(self):
        """Initialize the state machine"""
        self.operations = {}
        self.call_counts = {}
    
    @rule(
        operation_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        should_fail=st.booleans(),
        max_attempts=st.integers(min_value=1, max_value=3)
    )
    def add_operation(self, operation_name, should_fail, max_attempts):
        """Add a new operation to test"""
        assume(operation_name not in self.operations)
        
        self.operations[operation_name] = {
            'should_fail': should_fail,
            'max_attempts': max_attempts,
            'call_count': 0
        }
        self.call_counts[operation_name] = 0
    
    @rule(operation_name=st.sampled_from([]))
    async def execute_operation(self, operation_name):
        """Execute an operation with error handling"""
        if operation_name not in self.operations:
            return
        
        op_config = self.operations[operation_name]
        
        async def test_func():
            self.call_counts[operation_name] += 1
            if op_config['should_fail']:
                raise Exception(f"Simulated failure for {operation_name}")
            return f"success_{operation_name}"
        
        config = RetryConfig(max_attempts=op_config['max_attempts'], base_delay=0.01)
        
        if op_config['should_fail']:
            with pytest.raises(Exception):
                await self.handler.with_retry(test_func, operation_name, config)
        else:
            result = await self.handler.with_retry(test_func, operation_name, config)
            assert result == f"success_{operation_name}"
    
    @invariant()
    def call_count_invariant(self):
        """Invariant: Call counts should not exceed max attempts for failing operations"""
        for op_name, op_config in self.operations.items():
            if op_name in self.call_counts:
                if op_config['should_fail']:
                    assert self.call_counts[op_name] <= op_config['max_attempts']
                else:
                    assert self.call_counts[op_name] <= 1  # Should succeed on first try

# Integration tests
class TestErrorHandlingIntegration:
    """Integration tests for error handling with other components"""
    
    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test error classification for different error types"""
        handler = ErrorHandler()
        
        # Test different error types
        test_cases = [
            (Exception("Rate limit exceeded"), ErrorType.API_RATE_LIMIT),
            (Exception("Network connection failed"), ErrorType.NETWORK_ERROR),
            (Exception("File is corrupted"), ErrorType.FILE_CORRUPTION),
            (Exception("Authentication failed"), ErrorType.AUTHENTICATION_ERROR),
            (Exception("Database storage error"), ErrorType.STORAGE_ERROR),
            (Exception("Failed to parse document"), ErrorType.PARSING_ERROR),
            (Exception("Embedding model error"), ErrorType.EMBEDDING_ERROR),
            (Exception("Unknown issue"), ErrorType.UNKNOWN_ERROR),
        ]
        
        for error, expected_type in test_cases:
            classified_type = handler._classify_error(error)
            assert classified_type == expected_type
    
    @pytest.mark.asyncio
    async def test_error_statistics(self):
        """Test error statistics collection"""
        handler = ErrorHandler()
        
        # Generate some errors
        async def failing_func():
            raise Exception("Rate limit exceeded")
        
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        
        # Execute multiple failing operations
        for _ in range(3):
            with pytest.raises(Exception):
                await handler.with_retry(failing_func, "test_stats", config)
        
        stats = handler.get_error_statistics()
        assert stats['total_errors'] >= 3
        assert 'api_rate_limit' in stats['error_counts']

if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short"])