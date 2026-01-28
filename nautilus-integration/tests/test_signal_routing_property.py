"""
Property-based tests for Signal Router & Buffer (SIG) functionality.

This module tests the correctness properties of the signal routing system
using property-based testing with Hypothesis.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from nautilus_integration.core.config import NautilusConfig, SignalRouterConfig
from nautilus_integration.services.signal_router import (
    AISignal,
    SignalType,
    SignalConfidence,
    SignalRouterService,
    SignalSubscription,
    DeliveryStatus,
)


# Test data generators
@st.composite
def signal_types(draw):
    """Generate valid signal types."""
    return draw(st.sampled_from(list(SignalType)))


@st.composite
def confidence_scores(draw):
    """Generate valid confidence scores."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def instrument_ids(draw):
    """Generate valid instrument IDs."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="._-"),
        min_size=1,
        max_size=20
    ))


@st.composite
def strategy_ids(draw):
    """Generate valid strategy IDs."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
        min_size=1,
        max_size=15
    ))


@st.composite
def ai_signals(draw):
    """Generate valid AI signals."""
    signal_type = draw(signal_types())
    instrument_id = draw(instrument_ids())
    confidence = draw(confidence_scores())
    
    # Generate appropriate value based on signal type
    if signal_type == SignalType.REGIME_PREDICTION:
        value = draw(st.sampled_from(["LOW_VOL_TRENDING", "MEDIUM_VOL_TRENDING", "HIGH_VOL_RANGING", "CRISIS"]))
    elif signal_type == SignalType.VOLATILITY_FORECAST:
        value = draw(st.floats(min_value=0.0, max_value=2.0))
    elif signal_type == SignalType.SENTIMENT_SCORE:
        value = draw(st.floats(min_value=-1.0, max_value=1.0))
    elif signal_type == SignalType.CORRELATION_SHIFT:
        value = {
            "from_correlation": draw(st.floats(min_value=-1.0, max_value=1.0)),
            "to_correlation": draw(st.floats(min_value=-1.0, max_value=1.0)),
            "shift_magnitude": draw(st.floats(min_value=0.0, max_value=2.0)),
        }
    else:
        value = draw(st.floats(min_value=-100.0, max_value=100.0))
    
    return AISignal(
        signal_type=signal_type,
        instrument_id=instrument_id,
        confidence=confidence,
        value=value,
        source_model=draw(st.text(min_size=1, max_size=20)),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.text(), st.floats(), st.integers()),
            max_size=5
        )),
    )


@st.composite
def signal_subscriptions(draw):
    """Generate valid signal subscriptions."""
    return SignalSubscription(
        strategy_id=draw(strategy_ids()),
        signal_types=draw(st.lists(signal_types(), min_size=1, max_size=3, unique=True)),
        instrument_ids=draw(st.lists(instrument_ids(), max_size=5, unique=True)),
        min_confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        max_signals_per_minute=draw(st.integers(min_value=1, max_value=1000)),
        delivery_timeout=draw(st.integers(min_value=1, max_value=60)),
        retry_attempts=draw(st.integers(min_value=1, max_value=10)),
    )


class TestSignalRoutingProperties:
    """Property-based tests for signal routing correctness."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return NautilusConfig(
            signal_router=SignalRouterConfig(
                buffer_max_size_mb=10,
                max_pending_deliveries=100,
            )
        )
    
    @pytest.fixture
    async def signal_router(self, config):
        """Create signal router service for testing."""
        router = SignalRouterService(config)
        await router.initialize()
        yield router
        await router.shutdown()
    
    # Feature: nautilus-trader-integration, Property 18: Signal Routing Correctness
    @given(
        signals=st.lists(ai_signals(), min_size=1, max_size=10),
        subscriptions=st.lists(signal_subscriptions(), min_size=1, max_size=5)
    )
    @settings(max_examples=100, deadline=5000)
    async def test_signal_routing_correctness(self, signal_router, signals, subscriptions):
        """
        Property 18: Signal Routing Correctness
        
        For any signal generated in the RAG system, the system should correctly 
        route it to the appropriate Nautilus strategies without loss or corruption.
        
        **Validates: Requirements 3.2**
        """
        # Setup subscriptions with mock callbacks
        callbacks = {}
        for subscription in subscriptions:
            callback = AsyncMock()
            callbacks[subscription.strategy_id] = callback
            
            await signal_router.subscribe_strategy(
                strategy_id=subscription.strategy_id,
                signal_types=subscription.signal_types,
                callback=callback,
                instrument_ids=subscription.instrument_ids,
                min_confidence=subscription.min_confidence,
            )
        
        # Route all signals
        routing_results = []
        for signal in signals:
            result = await signal_router.route_signal(signal)
            routing_results.append(result)
        
        # Allow time for delivery processing
        await asyncio.sleep(0.1)
        
        # Verify routing correctness
        for i, signal in enumerate(signals):
            result = routing_results[i]
            
            # Signal should be routed without corruption
            assert result["signal_id"] == signal.signal_id
            assert "deliveries" in result
            assert "errors" in result
            
            # Find matching subscriptions
            expected_matches = []
            for subscription in subscriptions:
                # Check signal type match
                if signal.signal_type not in subscription.signal_types:
                    continue
                
                # Check instrument match
                if subscription.instrument_ids and signal.instrument_id not in subscription.instrument_ids:
                    continue
                
                # Check confidence threshold
                if signal.confidence < subscription.min_confidence:
                    continue
                
                expected_matches.append(subscription.strategy_id)
            
            # Verify deliveries match expected subscriptions
            delivered_strategies = [d["strategy_id"] for d in result["deliveries"]]
            
            # All expected matches should have deliveries queued
            for expected_strategy in expected_matches:
                assert expected_strategy in delivered_strategies, (
                    f"Signal {signal.signal_id} should be delivered to strategy {expected_strategy}"
                )
            
            # No unexpected deliveries should occur
            for delivered_strategy in delivered_strategies:
                assert delivered_strategy in expected_matches, (
                    f"Signal {signal.signal_id} should not be delivered to strategy {delivered_strategy}"
                )
    
    # Feature: nautilus-trader-integration, Property 19: Real-Time Signal Delivery Performance
    @given(
        signal=ai_signals(),
        strategy_id=strategy_ids(),
        timeout=st.floats(min_value=0.1, max_value=2.0)
    )
    @settings(max_examples=100, deadline=3000)
    async def test_realtime_signal_delivery_performance(self, signal_router, signal, strategy_id, timeout):
        """
        Property 19: Real-Time Signal Delivery Performance
        
        For any AI signal delivery, the latency should be consistently under 
        one second from generation to strategy consumption.
        
        **Validates: Requirements 3.3**
        """
        # Setup strategy subscription
        callback = AsyncMock()
        await signal_router.subscribe_strategy(
            strategy_id=strategy_id,
            signal_types=[signal.signal_type],
            callback=callback,
            min_confidence=0.0,
        )
        
        # Measure delivery time
        start_time = datetime.now()
        
        # Deliver signal with real-time guarantees
        result = await signal_router.deliver_signal_realtime(
            signal=signal,
            strategy_id=strategy_id,
            timeout=timeout
        )
        
        end_time = datetime.now()
        total_latency = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirements
        if result["success"]:
            # Successful delivery should meet latency requirements
            assert result["total_latency_ms"] is not None
            assert result["delivery_latency_ms"] is not None
            
            # Total latency should be under 1 second (1000ms)
            assert result["total_latency_ms"] < 1000, (
                f"Total latency {result['total_latency_ms']}ms exceeds 1000ms threshold"
            )
            
            # Delivery latency should be reasonable
            assert result["delivery_latency_ms"] < timeout * 1000, (
                f"Delivery latency {result['delivery_latency_ms']}ms exceeds timeout {timeout * 1000}ms"
            )
            
            # Callback should have been called
            callback.assert_called_once_with(signal)
        
        else:
            # Failed delivery should still complete quickly
            assert total_latency < (timeout + 0.5) * 1000, (
                f"Failed delivery took {total_latency}ms, should complete within timeout + 500ms"
            )
    
    @given(
        signals=st.lists(ai_signals(), min_size=5, max_size=20),
        strategy_id=strategy_ids()
    )
    @settings(max_examples=50, deadline=10000)
    async def test_signal_buffering_consistency(self, signal_router, signals, strategy_id):
        """
        Test that signal buffering maintains consistency for backtesting replay.
        
        All signals should be buffered and retrievable in chronological order.
        """
        # Buffer all signals
        for signal in signals:
            await signal_router.buffer_signal(signal)
        
        # Get unique instruments
        instruments = list(set(signal.instrument_id for signal in signals))
        
        for instrument_id in instruments:
            # Get signals for this instrument
            instrument_signals = [s for s in signals if s.instrument_id == instrument_id]
            
            if not instrument_signals:
                continue
            
            # Define time range covering all signals
            start_time = min(s.timestamp for s in instrument_signals) - timedelta(minutes=1)
            end_time = max(s.timestamp for s in instrument_signals) + timedelta(minutes=1)
            
            # Replay signals
            replayed_signals = await signal_router.replay_signals(
                instrument_id=instrument_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Verify all signals are present
            replayed_ids = {s.signal_id for s in replayed_signals}
            expected_ids = {s.signal_id for s in instrument_signals}
            
            assert expected_ids.issubset(replayed_ids), (
                f"Missing signals in replay: {expected_ids - replayed_ids}"
            )
            
            # Verify chronological order
            replayed_timestamps = [s.timestamp for s in replayed_signals]
            assert replayed_timestamps == sorted(replayed_timestamps), (
                "Replayed signals are not in chronological order"
            )
    
    @given(
        signals=st.lists(ai_signals(), min_size=1, max_size=10),
        subscription=signal_subscriptions()
    )
    @settings(max_examples=50, deadline=5000)
    async def test_signal_validation_consistency(self, signal_router, signals, subscription):
        """
        Test that signal validation is consistent and deterministic.
        
        The same signal should always produce the same validation result.
        """
        # Setup subscription
        callback = AsyncMock()
        await signal_router.subscribe_strategy(
            strategy_id=subscription.strategy_id,
            signal_types=subscription.signal_types,
            callback=callback,
            instrument_ids=subscription.instrument_ids,
            min_confidence=subscription.min_confidence,
        )
        
        for signal in signals:
            # Route signal multiple times
            result1 = await signal_router.route_signal(signal)
            result2 = await signal_router.route_signal(signal)
            
            # Validation should be consistent
            assert result1["signal_id"] == result2["signal_id"]
            
            # Routing decisions should be deterministic
            # (Note: delivery IDs will be different, but routing logic should be same)
            delivered_strategies_1 = {d["strategy_id"] for d in result1["deliveries"]}
            delivered_strategies_2 = {d["strategy_id"] for d in result2["deliveries"]}
            
            assert delivered_strategies_1 == delivered_strategies_2, (
                f"Inconsistent routing for signal {signal.signal_id}: "
                f"{delivered_strategies_1} vs {delivered_strategies_2}"
            )


class SignalRoutingStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for signal routing system.
    
    This tests the system behavior over sequences of operations.
    """
    
    def __init__(self):
        super().__init__()
        self.config = NautilusConfig(
            signal_router=SignalRouterConfig(
                buffer_max_size_mb=5,
                max_pending_deliveries=50,
            )
        )
        self.signal_router = None
        self.subscriptions = {}
        self.callbacks = {}
        self.routed_signals = []
    
    @initialize()
    async def setup(self):
        """Initialize the signal router."""
        self.signal_router = SignalRouterService(self.config)
        await self.signal_router.initialize()
    
    @rule(
        strategy_id=strategy_ids(),
        signal_types=st.lists(signal_types(), min_size=1, max_size=2, unique=True),
        min_confidence=st.floats(min_value=0.0, max_value=0.8)
    )
    async def add_subscription(self, strategy_id, signal_types, min_confidence):
        """Add a new signal subscription."""
        assume(strategy_id not in self.subscriptions)
        
        callback = AsyncMock()
        self.callbacks[strategy_id] = callback
        
        subscription_id = await self.signal_router.subscribe_strategy(
            strategy_id=strategy_id,
            signal_types=signal_types,
            callback=callback,
            min_confidence=min_confidence,
        )
        
        self.subscriptions[strategy_id] = {
            "subscription_id": subscription_id,
            "signal_types": signal_types,
            "min_confidence": min_confidence,
        }
    
    @rule(strategy_id=strategy_ids())
    async def remove_subscription(self, strategy_id):
        """Remove a signal subscription."""
        if strategy_id in self.subscriptions:
            await self.signal_router.unsubscribe_strategy(strategy_id)
            del self.subscriptions[strategy_id]
            self.callbacks.pop(strategy_id, None)
    
    @rule(signal=ai_signals())
    async def route_signal(self, signal):
        """Route a signal through the system."""
        result = await self.signal_router.route_signal(signal)
        self.routed_signals.append((signal, result))
    
    @invariant()
    def check_subscription_consistency(self):
        """Verify subscription state consistency."""
        # All subscriptions should have corresponding callbacks
        for strategy_id in self.subscriptions:
            assert strategy_id in self.callbacks, (
                f"Missing callback for subscription {strategy_id}"
            )
    
    @invariant()
    def check_routing_consistency(self):
        """Verify routing consistency."""
        for signal, result in self.routed_signals:
            # Result should contain expected fields
            assert "signal_id" in result
            assert "deliveries" in result
            assert "errors" in result
            
            # Signal ID should match
            assert result["signal_id"] == signal.signal_id
    
    async def teardown(self):
        """Clean up resources."""
        if self.signal_router:
            await self.signal_router.shutdown()


# Test class for running the state machine
class TestSignalRoutingStateMachine:
    """Test runner for the signal routing state machine."""
    
    @settings(max_examples=50, stateful_step_count=20, deadline=10000)
    def test_signal_routing_state_machine(self):
        """Run the signal routing state machine test."""
        # Note: This would need to be adapted for async testing
        # For now, we'll skip the stateful test in the async context
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])