"""
Performance and load integration tests for NautilusTrader integration.

This module tests system performance under various load conditions and validates
performance requirements for high-frequency trading scenarios.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import concurrent.futures

import pytest
import pandas as pd
import numpy as np
import psutil

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.services.integration_service import BacktestConfig
from nautilus_integration.services.integration_service import NautilusIntegrationService
from nautilus_integration.services.signal_router import SignalRouterService
from nautilus_integration.services.data_catalog_adapter import DataCatalogAdapter


class TestPerformanceAndLoadIntegration:
    """Performance and load integration tests."""
    
    @pytest.fixture
    def config(self):
        """Create performance test configuration."""
        return NautilusConfig(
            environment="testing",  # Use valid environment value
            log_level="WARNING",  # Reduce logging overhead
            performance_monitoring_enabled=True,
        )
    
    @pytest.fixture
    async def integration_service(self, config):
        """Create integration service for performance testing."""
        service = NautilusIntegrationService(config)
        await service.initialize()
        yield service
        await service.shutdown()
    
    @pytest.fixture
    async def signal_router(self, config):
        """Create signal router for performance testing."""
        router = SignalRouterService(config)
        await router.initialize()
        yield router
        await router.shutdown()
    
    @pytest.fixture
    def high_frequency_market_data(self):
        """Generate high-frequency market data for performance testing."""
        # Generate 1 week of 1-second bars (604,800 bars)
        start_time = datetime.now() - timedelta(days=7)
        timestamps = pd.date_range(start=start_time, periods=604800, freq='1S')
        
        np.random.seed(42)
        base_price = 1.1000
        
        data = []
        current_price = base_price
        
        for i, timestamp in enumerate(timestamps):
            # High-frequency price movements
            change = np.random.normal(0, 0.0001)
            current_price *= (1 + change)
            
            # Generate tick-level OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.00005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.00005)))
            open_price = current_price + np.random.normal(0, 0.00002)
            close_price = current_price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': timestamp,
                'instrument_id': 'EUR/USD',
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close_price, 5),
                'volume': round(volume, 2),
                'bar_type': '1-SECOND-MID',
            })
            
            # Add progress tracking for large dataset generation
            if i % 100000 == 0:
                print(f"Generated {i} bars...")
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def high_volume_strategies(self):
        """Generate multiple strategies for load testing."""
        strategies = []
        
        strategy_families = ["TREND_FOLLOWING", "MEAN_REVERSION", "MOMENTUM_ROTATION", "VOLATILITY_BREAKOUT"]
        instruments = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF"]
        
        for i in range(20):  # 20 strategies for load testing
            family = strategy_families[i % len(strategy_families)]
            instrument = instruments[i % len(instruments)]
            
            strategy = {
                "strategy_id": f"load_test_strategy_{i:02d}",
                "strategy_name": f"Load Test {family} {instrument}",
                "family": family,
                "version": "1.0.0",
                "parameters": {
                    "period": 10 + (i % 20),
                    "threshold": 0.001 + (i * 0.0001),
                    "risk_per_trade": 0.01 + (i * 0.001),
                },
                "instruments": [instrument],
                "risk_constraints": {
                    "max_position_size": 10000.0 + (i * 1000),
                    "max_daily_loss": -1000.0 - (i * 100),
                    "max_leverage": 2.0 + (i * 0.1),
                },
                "ai_signal_subscriptions": ["regime_prediction", "volatility_forecast"],
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            
            strategies.append(strategy)
        
        return strategies
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_high_frequency_data_processing_performance(
        self, 
        integration_service, 
        high_frequency_market_data
    ):
        """
        Test high-frequency data processing performance.
        
        Validates system can handle tick-level data processing within latency requirements.
        """
        # Performance requirements
        MAX_PROCESSING_TIME_PER_BAR_MS = 0.1  # 100 microseconds per bar
        MAX_MEMORY_USAGE_MB = 2000  # 2GB memory limit
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process data in chunks to simulate real-time processing
        chunk_size = 10000  # Process 10k bars at a time
        total_bars = len(high_frequency_market_data)
        processing_times = []
        
        for i in range(0, total_bars, chunk_size):
            chunk = high_frequency_market_data.iloc[i:i+chunk_size]
            
            start_time = time.perf_counter()
            
            # Mock data processing through integration service
            with patch.object(integration_service, '_process_market_data_chunk') as mock_process:
                mock_process.return_value = {
                    "processed_bars": len(chunk),
                    "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                    "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                }
                
                result = await integration_service.process_high_frequency_data(chunk)
            
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            processing_times.append(processing_time_ms)
            
            # Check per-bar processing time
            per_bar_time_ms = processing_time_ms / len(chunk)
            assert per_bar_time_ms < MAX_PROCESSING_TIME_PER_BAR_MS, (
                f"Per-bar processing time {per_bar_time_ms:.4f}ms exceeds limit {MAX_PROCESSING_TIME_PER_BAR_MS}ms"
            )
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            assert current_memory < MAX_MEMORY_USAGE_MB, (
                f"Memory usage {current_memory:.1f}MB exceeds limit {MAX_MEMORY_USAGE_MB}MB"
            )
            
            # Progress reporting
            if i % (chunk_size * 10) == 0:
                progress = (i / total_bars) * 100
                print(f"Processed {progress:.1f}% - Avg time per bar: {per_bar_time_ms:.4f}ms")
        
        # Analyze overall performance
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        p95_processing_time = np.percentile(processing_times, 95)
        
        print(f"Performance Summary:")
        print(f"  Total bars processed: {total_bars:,}")
        print(f"  Average chunk processing time: {avg_processing_time:.2f}ms")
        print(f"  Maximum chunk processing time: {max_processing_time:.2f}ms")
        print(f"  95th percentile processing time: {p95_processing_time:.2f}ms")
        print(f"  Memory usage increase: {current_memory - initial_memory:.1f}MB")
        
        # Performance assertions
        assert avg_processing_time < chunk_size * MAX_PROCESSING_TIME_PER_BAR_MS, (
            "Average processing time exceeds requirements"
        )
        assert p95_processing_time < chunk_size * MAX_PROCESSING_TIME_PER_BAR_MS * 2, (
            "95th percentile processing time exceeds tolerance"
        )
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_concurrent_strategy_execution_performance(
        self, 
        integration_service, 
        high_volume_strategies, 
        high_frequency_market_data
    ):
        """
        Test concurrent strategy execution performance.
        
        Validates system can handle multiple strategies executing simultaneously.
        """
        # Performance requirements
        MAX_STRATEGY_LATENCY_MS = 10  # 10ms per strategy execution
        MAX_CONCURRENT_STRATEGIES = 50
        MIN_THROUGHPUT_STRATEGIES_PER_SEC = 100
        
        # Limit data size for concurrent testing
        test_data = high_frequency_market_data.iloc[:10000]  # 10k bars
        
        # Test with increasing concurrency levels
        concurrency_levels = [1, 5, 10, 20, len(high_volume_strategies)]
        performance_results = []
        
        for concurrency in concurrency_levels:
            strategies_subset = high_volume_strategies[:concurrency]
            
            start_time = time.perf_counter()
            
            # Execute strategies concurrently
            tasks = []
            for strategy in strategies_subset:
                task = self._execute_strategy_with_data(
                    integration_service, 
                    strategy, 
                    test_data
                )
                tasks.append(task)
            
            # Wait for all strategies to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Analyze results
            successful_executions = [r for r in results if not isinstance(r, Exception)]
            failed_executions = [r for r in results if isinstance(r, Exception)]
            
            throughput = len(successful_executions) / total_time
            avg_latency_ms = (total_time / len(successful_executions)) * 1000 if successful_executions else float('inf')
            
            performance_result = {
                "concurrency": concurrency,
                "total_time_s": total_time,
                "successful_executions": len(successful_executions),
                "failed_executions": len(failed_executions),
                "throughput_strategies_per_sec": throughput,
                "avg_latency_ms": avg_latency_ms,
            }
            
            performance_results.append(performance_result)
            
            print(f"Concurrency {concurrency}: {throughput:.1f} strategies/sec, {avg_latency_ms:.2f}ms avg latency")
            
            # Performance assertions
            assert len(failed_executions) == 0, (
                f"Strategy executions failed at concurrency {concurrency}: {failed_executions}"
            )
            
            if concurrency <= MAX_CONCURRENT_STRATEGIES:
                assert avg_latency_ms < MAX_STRATEGY_LATENCY_MS, (
                    f"Average latency {avg_latency_ms:.2f}ms exceeds limit {MAX_STRATEGY_LATENCY_MS}ms "
                    f"at concurrency {concurrency}"
                )
                
                assert throughput >= MIN_THROUGHPUT_STRATEGIES_PER_SEC, (
                    f"Throughput {throughput:.1f} strategies/sec below minimum {MIN_THROUGHPUT_STRATEGIES_PER_SEC} "
                    f"at concurrency {concurrency}"
                )
        
        # Analyze scalability
        print("\nScalability Analysis:")
        for result in performance_results:
            print(f"  Concurrency {result['concurrency']:2d}: "
                  f"{result['throughput_strategies_per_sec']:6.1f} strategies/sec, "
                  f"{result['avg_latency_ms']:6.2f}ms latency")
    
    @pytest.mark.performance
    async def test_signal_routing_performance_under_load(
        self, 
        signal_router
    ):
        """
        Test signal routing performance under high load.
        
        Validates signal delivery latency and throughput requirements.
        """
        # Performance requirements
        MAX_SIGNAL_LATENCY_MS = 1  # 1ms signal delivery latency
        MIN_SIGNAL_THROUGHPUT_PER_SEC = 10000  # 10k signals per second
        MAX_SIGNAL_LOSS_RATE = 0.001  # 0.1% maximum signal loss
        
        # Generate high-volume signal load
        num_signals = 50000
        signals = []
        
        for i in range(num_signals):
            signal = {
                "signal_id": f"perf_test_signal_{i:06d}",
                "signal_type": "REGIME_PREDICTION" if i % 2 == 0 else "VOLATILITY_FORECAST",
                "instrument_id": ["EUR/USD", "GBP/USD", "USD/JPY"][i % 3],
                "timestamp": datetime.now() + timedelta(microseconds=i),
                "confidence": 0.5 + (i % 50) / 100,
                "value": f"signal_value_{i}",
                "source_model": "performance_test_model",
                "metadata": {"test_id": i},
            }
            signals.append(signal)
        
        # Set up signal subscriptions
        strategy_ids = [f"perf_strategy_{i}" for i in range(10)]
        callbacks = {}
        
        for strategy_id in strategy_ids:
            callback = AsyncMock()
            callbacks[strategy_id] = callback
            
            await signal_router.subscribe_strategy(
                strategy_id=strategy_id,
                signal_types=["REGIME_PREDICTION", "VOLATILITY_FORECAST"],
                callback=callback,
                min_confidence=0.0,
            )
        
        # Measure signal routing performance
        start_time = time.perf_counter()
        routing_results = []
        
        # Route signals in batches to simulate realistic load
        batch_size = 1000
        for i in range(0, num_signals, batch_size):
            batch = signals[i:i+batch_size]
            batch_start = time.perf_counter()
            
            # Route batch concurrently
            batch_tasks = [signal_router.route_signal(signal) for signal in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            
            routing_results.extend(batch_results)
            
            # Check batch performance
            batch_throughput = len(batch) / batch_time
            batch_latency_ms = (batch_time / len(batch)) * 1000
            
            print(f"Batch {i//batch_size + 1}: {batch_throughput:.0f} signals/sec, {batch_latency_ms:.3f}ms avg latency")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze routing results
        successful_routes = [r for r in routing_results if not isinstance(r, Exception)]
        failed_routes = [r for r in routing_results if isinstance(r, Exception)]
        
        total_throughput = len(successful_routes) / total_time
        avg_latency_ms = (total_time / len(successful_routes)) * 1000 if successful_routes else float('inf')
        signal_loss_rate = len(failed_routes) / num_signals
        
        print(f"\nSignal Routing Performance Summary:")
        print(f"  Total signals: {num_signals:,}")
        print(f"  Successful routes: {len(successful_routes):,}")
        print(f"  Failed routes: {len(failed_routes):,}")
        print(f"  Total throughput: {total_throughput:.0f} signals/sec")
        print(f"  Average latency: {avg_latency_ms:.3f}ms")
        print(f"  Signal loss rate: {signal_loss_rate:.4f} ({signal_loss_rate*100:.2f}%)")
        
        # Performance assertions
        assert total_throughput >= MIN_SIGNAL_THROUGHPUT_PER_SEC, (
            f"Signal throughput {total_throughput:.0f} signals/sec below minimum {MIN_SIGNAL_THROUGHPUT_PER_SEC}"
        )
        
        assert avg_latency_ms < MAX_SIGNAL_LATENCY_MS, (
            f"Average signal latency {avg_latency_ms:.3f}ms exceeds limit {MAX_SIGNAL_LATENCY_MS}ms"
        )
        
        assert signal_loss_rate < MAX_SIGNAL_LOSS_RATE, (
            f"Signal loss rate {signal_loss_rate:.4f} exceeds limit {MAX_SIGNAL_LOSS_RATE}"
        )
        
        # Verify signal delivery to subscribers
        total_deliveries = 0
        for strategy_id, callback in callbacks.items():
            delivery_count = callback.call_count
            total_deliveries += delivery_count
            print(f"  Strategy {strategy_id}: {delivery_count} signals delivered")
        
        # Each signal should be delivered to all matching subscribers
        expected_deliveries = len(successful_routes) * len(strategy_ids)
        delivery_success_rate = total_deliveries / expected_deliveries if expected_deliveries > 0 else 0
        
        print(f"  Total deliveries: {total_deliveries:,}")
        print(f"  Expected deliveries: {expected_deliveries:,}")
        print(f"  Delivery success rate: {delivery_success_rate:.4f} ({delivery_success_rate*100:.2f}%)")
        
        assert delivery_success_rate >= (1 - MAX_SIGNAL_LOSS_RATE), (
            f"Delivery success rate {delivery_success_rate:.4f} below threshold"
        )
    
    @pytest.mark.performance
    async def test_memory_usage_and_garbage_collection(
        self, 
        integration_service
    ):
        """
        Test memory usage patterns and garbage collection efficiency.
        
        Validates system memory management under sustained load.
        """
        import gc
        
        # Memory requirements
        MAX_MEMORY_GROWTH_MB = 500  # 500MB maximum memory growth
        MAX_GC_PAUSE_MS = 10  # 10ms maximum GC pause
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate sustained workload
        workload_iterations = 100
        memory_measurements = []
        gc_pause_times = []
        
        for iteration in range(workload_iterations):
            # Create temporary data structures
            temp_data = pd.DataFrame({
                'timestamp': pd.date_range(start=datetime.now(), periods=1000, freq='1S'),
                'price': np.random.rand(1000) * 100,
                'volume': np.random.rand(1000) * 1000,
            })
            
            # Simulate processing
            with patch.object(integration_service, '_process_temporary_data') as mock_process:
                mock_process.return_value = {"processed": True}
                await integration_service.process_market_data_batch(temp_data)
            
            # Measure memory before GC
            pre_gc_memory = process.memory_info().rss / 1024 / 1024
            
            # Force garbage collection and measure pause time
            gc_start = time.perf_counter()
            gc.collect()
            gc_end = time.perf_counter()
            
            gc_pause_ms = (gc_end - gc_start) * 1000
            gc_pause_times.append(gc_pause_ms)
            
            # Measure memory after GC
            post_gc_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append({
                "iteration": iteration,
                "pre_gc_memory_mb": pre_gc_memory,
                "post_gc_memory_mb": post_gc_memory,
                "gc_pause_ms": gc_pause_ms,
            })
            
            # Check GC pause time
            assert gc_pause_ms < MAX_GC_PAUSE_MS, (
                f"GC pause time {gc_pause_ms:.2f}ms exceeds limit {MAX_GC_PAUSE_MS}ms at iteration {iteration}"
            )
            
            # Progress reporting
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Memory {post_gc_memory:.1f}MB, GC pause {gc_pause_ms:.2f}ms")
        
        # Analyze memory usage patterns
        final_memory = memory_measurements[-1]["post_gc_memory_mb"]
        memory_growth = final_memory - initial_memory
        
        avg_gc_pause = np.mean(gc_pause_times)
        max_gc_pause = np.max(gc_pause_times)
        p95_gc_pause = np.percentile(gc_pause_times, 95)
        
        print(f"\nMemory Management Summary:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Average GC pause: {avg_gc_pause:.2f}ms")
        print(f"  Maximum GC pause: {max_gc_pause:.2f}ms")
        print(f"  95th percentile GC pause: {p95_gc_pause:.2f}ms")
        
        # Memory assertions
        assert memory_growth < MAX_MEMORY_GROWTH_MB, (
            f"Memory growth {memory_growth:.1f}MB exceeds limit {MAX_MEMORY_GROWTH_MB}MB"
        )
        
        assert max_gc_pause < MAX_GC_PAUSE_MS, (
            f"Maximum GC pause {max_gc_pause:.2f}ms exceeds limit {MAX_GC_PAUSE_MS}ms"
        )
        
        assert p95_gc_pause < MAX_GC_PAUSE_MS, (
            f"95th percentile GC pause {p95_gc_pause:.2f}ms exceeds limit {MAX_GC_PAUSE_MS}ms"
        )
    
    async def _execute_strategy_with_data(
        self, 
        integration_service, 
        strategy_config, 
        market_data
    ):
        """Helper method to execute a strategy with market data."""
        # Mock strategy execution
        with patch.object(integration_service, '_execute_single_strategy') as mock_execute:
            mock_execute.return_value = {
                "strategy_id": strategy_config["strategy_id"],
                "execution_time_ms": np.random.uniform(1, 5),  # 1-5ms execution time
                "bars_processed": len(market_data),
                "trades_generated": np.random.randint(0, 10),
                "success": True,
            }
            
            result = await integration_service.execute_strategy_backtest(
                strategy_config=strategy_config,
                market_data=market_data
            )
            
            return result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])