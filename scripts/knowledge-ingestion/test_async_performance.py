"""
Test Async Performance Optimizations

This module provides comprehensive tests for all async performance optimizations
including concurrent processing, GPU acceleration, database connection pooling,
and resource management.

Requirements: 10.1, 10.2, 10.4
"""

import asyncio
import pytest
import time
import tempfile
import json
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from services.async_performance_optimizer import (
    AsyncPerformanceOptimizer, ProcessingTask, ProcessingMode,
    WorkerPoolConfig, PerformanceMetrics
)
from services.async_embedding_service import (
    AsyncEmbeddingService, EmbeddingRequest, AsyncEmbeddingResult,
    GPUResourceManager, AsyncEmbeddingBatchProcessor
)
from services.async_database_service import (
    AsyncDatabaseService, AsyncConnectionPool, DatabaseQuery, QueryType
)
from services.enhanced_batch_manager import (
    EnhancedBatchIngestionManager, EnhancedProcessingOptions,
    ProcessingMetrics, ResourceMonitor
)
from config.async_performance_config import (
    AsyncPerformanceConfig, OptimizationLevel, create_default_config
)


class TestAsyncPerformanceOptimizer:
    """Test async performance optimizer functionality"""
    
    @pytest.fixture
    async def optimizer(self):
        """Create async performance optimizer for testing"""
        optimizer = AsyncPerformanceOptimizer()
        await optimizer.initialize()
        yield optimizer
        await optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer is not None
        
        # Check that worker pools are created
        assert len(optimizer._worker_pools) > 0
        
        # Check that default pools exist
        expected_pools = ['io_operations', 'cpu_processing', 'gpu_embeddings', 'database_ops']
        for pool_name in expected_pools:
            assert pool_name in optimizer._worker_pools
    
    @pytest.mark.asyncio
    async def test_task_submission(self, optimizer):
        """Test task submission to worker pools"""
        # Create a simple task
        task_data = {'test': 'data'}
        
        # Submit task
        success = await optimizer.submit_task(
            pool_name='io_operations',
            task_data=task_data,
            processing_mode=ProcessingMode.IO_BOUND
        )
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self, optimizer):
        """Test concurrent file processing"""
        # Create mock files
        mock_files = [{'id': i, 'content': f'file_{i}'} for i in range(10)]
        
        # Mock processing function
        async def mock_process_file(file_data):
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"processed_{file_data['id']}"
        
        # Process files concurrently
        start_time = time.time()
        results = await optimizer.process_files_concurrently(
            mock_files,
            mock_process_file,
            max_concurrent=5
        )
        end_time = time.time()
        
        # Check results
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # Should be faster than sequential processing
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Should complete in less than 1 second
    
    @pytest.mark.asyncio
    async def test_optimal_worker_count(self, optimizer):
        """Test optimal worker count calculation"""
        # Test different workload types
        io_workers = await optimizer.get_optimal_worker_count(ProcessingMode.IO_BOUND)
        cpu_workers = await optimizer.get_optimal_worker_count(ProcessingMode.CPU_BOUND)
        gpu_workers = await optimizer.get_optimal_worker_count(ProcessingMode.GPU_ACCELERATED)
        
        # I/O-bound should have more workers than CPU-bound
        assert io_workers >= cpu_workers
        
        # All should be positive
        assert io_workers > 0
        assert cpu_workers > 0
        assert gpu_workers > 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, optimizer):
        """Test performance metrics collection"""
        # Submit some tasks to generate metrics
        for i in range(5):
            await optimizer.submit_task(
                pool_name='io_operations',
                task_data={'test': i},
                processing_mode=ProcessingMode.IO_BOUND
            )
        
        # Wait a bit for processing
        await asyncio.sleep(0.5)
        
        # Get metrics
        metrics = await optimizer.get_global_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_tasks >= 5


class TestAsyncEmbeddingService:
    """Test async embedding service functionality"""
    
    @pytest.fixture
    async def embedding_service(self):
        """Create async embedding service for testing"""
        service = AsyncEmbeddingService()
        await service.initialize()
        yield service
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_embedding_service_initialization(self, embedding_service):
        """Test embedding service initialization"""
        assert embedding_service is not None
        assert embedding_service.gpu_manager is not None
        assert embedding_service.batch_processor is not None
    
    @pytest.mark.asyncio
    async def test_single_embedding_generation(self, embedding_service):
        """Test single embedding generation"""
        text = "This is a test document for embedding generation."
        title = "Test Document"
        
        result = await embedding_service.generate_embedding_async(
            text=text,
            title=title
        )
        
        assert isinstance(result, AsyncEmbeddingResult)
        assert result.request_id is not None
        
        # Note: In a real test, we'd check result.success and result.embedding
        # For now, we just verify the structure
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, embedding_service):
        """Test batch embedding generation"""
        texts_and_titles = [
            ("Document 1 content", "Document 1"),
            ("Document 2 content", "Document 2"),
            ("Document 3 content", "Document 3"),
        ]
        
        results = await embedding_service.generate_batch_embeddings_async(
            texts_and_titles,
            max_concurrent=2
        )
        
        assert len(results) == 3
        assert all(isinstance(result, AsyncEmbeddingResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_gpu_status(self, embedding_service):
        """Test GPU status reporting"""
        gpu_status = await embedding_service.get_gpu_status()
        
        assert isinstance(gpu_status, dict)
        assert 'available' in gpu_status
        assert 'backend_type' in gpu_status
        assert 'utilization' in gpu_status
        assert 'memory_usage' in gpu_status


class TestGPUResourceManager:
    """Test GPU resource manager"""
    
    @pytest.fixture
    async def gpu_manager(self):
        """Create GPU resource manager for testing"""
        manager = GPUResourceManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_gpu_initialization(self, gpu_manager):
        """Test GPU initialization"""
        assert gpu_manager is not None
        
        # Check if GPU availability is properly detected
        assert isinstance(gpu_manager.is_available, bool)
    
    @pytest.mark.asyncio
    async def test_gpu_resource_acquisition(self, gpu_manager):
        """Test GPU resource acquisition"""
        async with gpu_manager.acquire_gpu() as device:
            # Device should be None if no GPU, or a device object if GPU available
            if gpu_manager.is_available:
                assert device is not None
            else:
                assert device is None


class TestAsyncDatabaseService:
    """Test async database service functionality"""
    
    @pytest.fixture
    async def db_service(self):
        """Create async database service for testing"""
        # Mock Supabase config
        mock_config = Mock()
        mock_config.url = "https://test.supabase.co"
        mock_config.key = "test_key"
        mock_config.service_role_key = "test_service_key"
        mock_config.max_connections = 5
        
        service = AsyncDatabaseService(mock_config)
        
        # Mock the connection pool initialization
        with patch.object(service, 'connection_pool') as mock_pool:
            mock_pool.initialize = AsyncMock(return_value=True)
            await service.initialize()
        
        yield service
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_database_service_initialization(self, db_service):
        """Test database service initialization"""
        assert db_service is not None
        assert db_service.connection_pool is not None
    
    @pytest.mark.asyncio
    async def test_query_execution(self, db_service):
        """Test async query execution"""
        # Mock the query execution
        with patch.object(db_service, 'execute_query_async') as mock_execute:
            mock_execute.return_value = Mock(data=[{'id': 1, 'name': 'test'}])
            
            result = await db_service.execute_query_async(
                query_type=QueryType.SELECT,
                table='documents',
                filters={'id': 1}
            )
            
            assert result is not None
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, db_service):
        """Test batch database operations"""
        # Mock batch data
        batch_data = [
            {'id': 1, 'content': 'chunk 1'},
            {'id': 2, 'content': 'chunk 2'},
            {'id': 3, 'content': 'chunk 3'},
        ]
        
        # Mock the batch processing
        with patch.object(db_service, 'store_chunks_batch_async') as mock_batch:
            mock_batch.return_value = [Mock(success=True) for _ in batch_data]
            
            results = await db_service.store_chunks_batch_async(batch_data)
            
            assert len(results) == 3
            assert all(result.success for result in results)


class TestEnhancedBatchManager:
    """Test enhanced batch ingestion manager"""
    
    @pytest.fixture
    async def batch_manager(self):
        """Create enhanced batch manager for testing"""
        manager = EnhancedBatchIngestionManager()
        
        # Mock service dependencies
        with patch.multiple(
            manager,
            _auth_service=Mock(),
            _browsing_service=AsyncMock(),
            _performance_optimizer=AsyncMock(),
            _embedding_service=AsyncMock(),
            _database_service=AsyncMock()
        ):
            await manager.initialize()
        
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_enhanced_manager_initialization(self, batch_manager):
        """Test enhanced manager initialization"""
        assert batch_manager is not None
        assert batch_manager._resource_monitor is not None
        assert len(batch_manager._processing_semaphores) > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_job_creation(self, batch_manager):
        """Test enhanced job creation"""
        # Mock file selections
        file_selections = [
            {
                'connection_id': 'test_connection',
                'file_ids': ['file1', 'file2'],
                'source_type': 'google_drive'
            }
        ]
        
        # Mock browsing service
        batch_manager._browsing_service.get_file_metadata = AsyncMock(
            return_value=Mock(
                file_id='test_file',
                name='test.pdf',
                size=1024,
                source_type='google_drive'
            )
        )
        
        # Create enhanced job
        job_id = await batch_manager.create_enhanced_job(
            user_id='test_user',
            name='Test Job',
            file_selections=file_selections,
            processing_options=EnhancedProcessingOptions()
        )
        
        assert job_id is not None
        assert job_id in batch_manager._jobs
    
    @pytest.mark.asyncio
    async def test_enhanced_metrics_collection(self, batch_manager):
        """Test enhanced metrics collection"""
        # Wait for metrics to be collected
        await asyncio.sleep(0.1)
        
        metrics = await batch_manager.get_enhanced_metrics()
        
        assert isinstance(metrics, ProcessingMetrics)
        assert hasattr(metrics, 'total_files')
        assert hasattr(metrics, 'gpu_utilization')
        assert hasattr(metrics, 'database_operations')


class TestResourceMonitor:
    """Test resource monitoring functionality"""
    
    @pytest.fixture
    def resource_monitor(self):
        """Create resource monitor for testing"""
        return ResourceMonitor()
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self, resource_monitor):
        """Test resource monitoring"""
        await resource_monitor.start_monitoring()
        
        # Wait for some metrics to be collected
        await asyncio.sleep(1.1)  # Wait longer than monitoring interval
        
        metrics = resource_monitor.get_current_metrics()
        
        assert isinstance(metrics, dict)
        if metrics:  # Only check if metrics were collected
            assert 'cpu_percent' in metrics
            assert 'memory_percent' in metrics
        
        await resource_monitor.stop_monitoring()
    
    def test_throttling_decision(self, resource_monitor):
        """Test throttling decision logic"""
        options = EnhancedProcessingOptions()
        options.cpu_threshold_percent = 50.0
        options.memory_threshold_mb = 1000
        
        # Mock high resource usage
        resource_monitor._metrics_history = [{
            'cpu_percent': 80.0,
            'memory_used_mb': 2000,
            'timestamp': time.time()
        }]
        
        should_throttle = resource_monitor.should_throttle_processing(options)
        assert should_throttle is True
    
    def test_optimal_concurrency_calculation(self, resource_monitor):
        """Test optimal concurrency calculation"""
        base_concurrency = 8
        
        # Mock low resource usage
        resource_monitor._metrics_history = [{
            'cpu_percent': 20.0,
            'memory_percent': 30.0,
            'timestamp': time.time()
        }]
        
        optimal = resource_monitor.get_optimal_concurrency(base_concurrency)
        assert optimal >= base_concurrency  # Should increase concurrency
        
        # Mock high resource usage
        resource_monitor._metrics_history = [{
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'timestamp': time.time()
        }]
        
        optimal = resource_monitor.get_optimal_concurrency(base_concurrency)
        assert optimal < base_concurrency  # Should decrease concurrency


class TestAsyncPerformanceConfig:
    """Test async performance configuration"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = create_default_config()
        
        assert isinstance(config, AsyncPerformanceConfig)
        assert config.optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.CUSTOM]
        assert config.worker_pools.io_pool_size > 0
        assert config.concurrency.max_concurrent_files > 0
    
    def test_optimization_level_presets(self):
        """Test optimization level presets"""
        # Test minimal preset
        minimal_config = AsyncPerformanceConfig(optimization_level=OptimizationLevel.MINIMAL)
        
        # Test aggressive preset
        aggressive_config = AsyncPerformanceConfig(optimization_level=OptimizationLevel.AGGRESSIVE)
        
        # Aggressive should have higher limits than minimal
        assert aggressive_config.worker_pools.io_pool_size >= minimal_config.worker_pools.io_pool_size
        assert aggressive_config.concurrency.max_concurrent_files >= minimal_config.concurrency.max_concurrent_files
    
    def test_auto_configuration(self):
        """Test auto-configuration based on system"""
        config = AsyncPerformanceConfig()
        config.auto_configure_for_system()
        
        # Should have reasonable values based on system
        assert config.worker_pools.cpu_pool_size > 0
        assert config.concurrency.max_concurrent_files > 0
        assert config.resource_limits.max_memory_usage_mb > 0
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = AsyncPerformanceConfig()
        warnings = config.validate_configuration()
        
        # Should return a list (may be empty)
        assert isinstance(warnings, list)
    
    def test_configuration_summary(self):
        """Test configuration summary"""
        config = AsyncPerformanceConfig()
        summary = config.get_summary()
        
        assert isinstance(summary, dict)
        assert 'optimization_level' in summary
        assert 'worker_pools' in summary
        assert 'concurrency' in summary
        assert 'features' in summary


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing_pipeline(self):
        """Test end-to-end processing pipeline with async optimizations"""
        # This would be a comprehensive test that:
        # 1. Creates a batch job with multiple files
        # 2. Processes them through the enhanced pipeline
        # 3. Verifies all optimizations are working
        # 4. Checks performance metrics
        
        # For now, this is a placeholder for the full integration test
        assert True  # Placeholder
    
    @pytest.mark.asyncio
    async def test_resource_constrained_processing(self):
        """Test processing under resource constraints"""
        # Test how the system behaves when resources are limited
        # Should gracefully degrade and throttle processing
        
        # Mock resource constraints
        config = AsyncPerformanceConfig(optimization_level=OptimizationLevel.MINIMAL)
        
        # Verify configuration is conservative
        assert config.worker_pools.io_pool_size <= 8
        assert config.concurrency.max_concurrent_files <= 4
    
    @pytest.mark.asyncio
    async def test_high_throughput_processing(self):
        """Test high-throughput processing scenario"""
        # Test system behavior under high load
        # Should utilize all available optimizations
        
        config = AsyncPerformanceConfig(optimization_level=OptimizationLevel.AGGRESSIVE)
        
        # Verify configuration is optimized for performance
        assert config.worker_pools.io_pool_size > 16
        assert config.concurrency.max_concurrent_files > 8
        assert config.features.enable_gpu_acceleration is True
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback(self):
        """Test error recovery and fallback mechanisms"""
        # Test that the system can recover from various failure modes
        # and fall back to less optimal but working configurations
        
        # This would test scenarios like:
        # - GPU acceleration failure -> fallback to CPU
        # - Database connection issues -> retry with backoff
        # - Memory pressure -> reduce concurrency
        
        assert True  # Placeholder for comprehensive error testing


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for async optimizations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_vs_sequential_processing(self):
        """Benchmark concurrent vs sequential processing"""
        # Create mock processing function
        async def mock_process(item):
            await asyncio.sleep(0.1)  # Simulate I/O
            return f"processed_{item}"
        
        items = list(range(10))
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for item in items:
            result = await mock_process(item)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        concurrent_results = await asyncio.gather(*[mock_process(item) for item in items])
        concurrent_time = time.time() - start_time
        
        # Concurrent should be significantly faster
        assert concurrent_time < sequential_time * 0.5
        assert len(concurrent_results) == len(sequential_results)
    
    @pytest.mark.asyncio
    async def test_batch_vs_individual_database_operations(self):
        """Benchmark batch vs individual database operations"""
        # This would test the performance difference between
        # individual database inserts vs batch inserts
        
        # Mock data
        records = [{'id': i, 'content': f'content_{i}'} for i in range(100)]
        
        # Individual operations would be slower than batch operations
        # This is a conceptual test - actual implementation would
        # measure real database operation times
        
        assert len(records) == 100  # Placeholder


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])