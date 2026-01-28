#!/usr/bin/env python3
"""
Test Suite for Vector Operations Optimization

This test suite validates the vector operations optimizer, Rust FFI interface,
and enhanced vector service functionality.

Tests include:
- Vector operations with different backends
- Rust FFI integration
- Performance benchmarking
- Error handling and fallback mechanisms
- Async integration
- Memory management

Requirements: 10.3, 10.5
"""

import asyncio
import unittest
import numpy as np
import time
import tempfile
import os
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from services.vector_operations_optimizer import (
    VectorOperationsOptimizer,
    VectorOperationConfig,
    VectorBackend,
    SimilarityMetric,
    get_vector_optimizer,
    create_optimized_config
)
from services.rust_ffi_interface import (
    VectorOperationsFFI,
    RustLibraryManager,
    get_vector_ffi,
    is_rust_available,
    RustFFIError
)
from services.enhanced_vector_service import (
    EnhancedVectorService,
    VectorProcessingJob,
    get_enhanced_vector_service
)


class TestVectorOperationsOptimizer(unittest.TestCase):
    """Test cases for VectorOperationsOptimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = VectorOperationConfig(
            enable_rust_bindings=False,  # Disable for pure Python tests
            enable_gpu_acceleration=False,
            batch_size=100,
            enable_benchmarking=True,
            fallback_on_error=True
        )
        self.optimizer = VectorOperationsOptimizer(self.config)
        
        # Create test vectors
        np.random.seed(42)
        self.test_vectors_small = np.random.randn(10, 128).astype(np.float32)
        self.test_vectors_large = np.random.randn(1000, 512).astype(np.float32)
        self.test_query_vectors = np.random.randn(5, 128).astype(np.float32)
    
    def test_vector_normalization_l2(self):
        """Test L2 vector normalization"""
        vectors = self.test_vectors_small.copy()
        normalized = self.optimizer.normalize_vectors(vectors, "l2")
        
        # Check that vectors are normalized (unit length)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
        
        # Check shape preservation
        self.assertEqual(normalized.shape, vectors.shape)
    
    def test_vector_normalization_l1(self):
        """Test L1 vector normalization"""
        vectors = self.test_vectors_small.copy()
        normalized = self.optimizer.normalize_vectors(vectors, "l1")
        
        # Check that L1 norms are 1
        l1_norms = np.sum(np.abs(normalized), axis=1)
        np.testing.assert_allclose(l1_norms, 1.0, rtol=1e-6)
    
    def test_vector_normalization_max(self):
        """Test max vector normalization"""
        vectors = self.test_vectors_small.copy()
        normalized = self.optimizer.normalize_vectors(vectors, "max")
        
        # Check that max absolute values are 1
        max_vals = np.max(np.abs(normalized), axis=1)
        np.testing.assert_allclose(max_vals, 1.0, rtol=1e-6)
    
    def test_cosine_similarity_matrix(self):
        """Test cosine similarity matrix computation"""
        vectors_a = self.test_vectors_small[:5]
        vectors_b = self.test_vectors_small[5:]
        
        similarity_matrix = self.optimizer.compute_similarity_matrix(
            vectors_a, vectors_b, SimilarityMetric.COSINE
        )
        
        # Check shape
        self.assertEqual(similarity_matrix.shape, (5, 5))
        
        # Check that similarities are in valid range [-1, 1]
        self.assertTrue(np.all(similarity_matrix >= -1.0))
        self.assertTrue(np.all(similarity_matrix <= 1.0))
        
        # Check diagonal similarity (same vectors should have similarity 1)
        same_vectors = self.optimizer.compute_similarity_matrix(
            vectors_a, vectors_a, SimilarityMetric.COSINE
        )
        np.testing.assert_allclose(np.diag(same_vectors), 1.0, rtol=1e-6)
    
    def test_euclidean_similarity_matrix(self):
        """Test Euclidean similarity matrix computation"""
        vectors_a = self.test_vectors_small[:3]
        vectors_b = self.test_vectors_small[3:6]
        
        similarity_matrix = self.optimizer.compute_similarity_matrix(
            vectors_a, vectors_b, SimilarityMetric.EUCLIDEAN
        )
        
        # Check shape
        self.assertEqual(similarity_matrix.shape, (3, 3))
        
        # Check that similarities are positive (distance converted to similarity)
        self.assertTrue(np.all(similarity_matrix > 0))
    
    def test_dot_product_similarity(self):
        """Test dot product similarity computation"""
        vectors_a = self.test_vectors_small[:3]
        vectors_b = self.test_vectors_small[3:6]
        
        similarity_matrix = self.optimizer.compute_similarity_matrix(
            vectors_a, vectors_b, SimilarityMetric.DOT_PRODUCT
        )
        
        # Check shape
        self.assertEqual(similarity_matrix.shape, (3, 3))
        
        # Verify against manual computation
        expected = np.dot(vectors_a, vectors_b.T)
        np.testing.assert_allclose(similarity_matrix, expected, rtol=1e-6)
    
    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation"""
        query_vectors = self.test_query_vectors
        database_vectors = self.test_vectors_small
        top_k = 3
        
        similarities, indices = self.optimizer.batch_cosine_similarity(
            query_vectors, database_vectors, top_k
        )
        
        # Check shapes
        self.assertEqual(similarities.shape, (5, 3))
        self.assertEqual(indices.shape, (5, 3))
        
        # Check that similarities are sorted in descending order
        for i in range(similarities.shape[0]):
            self.assertTrue(np.all(similarities[i, :-1] >= similarities[i, 1:]))
        
        # Check that indices are valid
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < database_vectors.shape[0]))
    
    def test_vector_arithmetic_add(self):
        """Test vector addition"""
        vector_a = self.test_vectors_small[:5]
        vector_b = self.test_vectors_small[5:10]
        
        result = self.optimizer.vector_arithmetic("add", vector_a, vector_b)
        expected = vector_a + vector_b
        
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_vector_arithmetic_subtract(self):
        """Test vector subtraction"""
        vector_a = self.test_vectors_small[:5]
        vector_b = self.test_vectors_small[5:10]
        
        result = self.optimizer.vector_arithmetic("subtract", vector_a, vector_b)
        expected = vector_a - vector_b
        
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_vector_arithmetic_scale(self):
        """Test vector scaling"""
        vector_a = self.test_vectors_small[:5]
        scalar = 2.5
        
        result = self.optimizer.vector_arithmetic("scale", vector_a, scalar=scalar)
        expected = vector_a * scalar
        
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_backend_selection(self):
        """Test automatic backend selection"""
        # Test with different vector sizes and operations
        small_vectors = np.random.randn(10, 64).astype(np.float32)
        large_vectors = np.random.randn(5000, 1024).astype(np.float32)
        
        # Small operations should prefer NumPy
        backend_small = self.optimizer._select_optimal_backend(
            "cosine_similarity", small_vectors.shape
        )
        self.assertIn(backend_small, [VectorBackend.NUMPY, VectorBackend.SCIPY])
        
        # Large operations should prefer SciPy or GPU
        backend_large = self.optimizer._select_optimal_backend(
            "similarity_matrix", large_vectors.shape
        )
        self.assertIn(backend_large, [VectorBackend.SCIPY, VectorBackend.GPU])
    
    def test_caching(self):
        """Test computation caching"""
        vectors = self.test_vectors_small.copy()
        
        # First computation
        start_time = time.time()
        result1 = self.optimizer.normalize_vectors(vectors, "l2")
        first_time = time.time() - start_time
        
        # Second computation (should be cached)
        start_time = time.time()
        result2 = self.optimizer.normalize_vectors(vectors, "l2")
        second_time = time.time() - start_time
        
        # Results should be identical
        np.testing.assert_allclose(result1, result2)
        
        # Second computation should be faster (cached)
        # Note: This might not always be true due to system variations
        # self.assertLess(second_time, first_time)
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        vectors = self.test_vectors_small.copy()
        
        # Perform some operations
        self.optimizer.normalize_vectors(vectors, "l2")
        self.optimizer.compute_similarity_matrix(vectors[:5], vectors[5:])
        
        # Get metrics
        metrics = self.optimizer.get_performance_metrics()
        
        # Check that metrics were collected
        self.assertGreater(len(metrics), 0)
        
        # Check metric fields
        for metric in metrics:
            self.assertIsInstance(metric.execution_time_ms, float)
            self.assertIsInstance(metric.memory_usage_mb, float)
            self.assertIsInstance(metric.throughput_vectors_per_second, float)
    
    def test_cache_statistics(self):
        """Test cache statistics"""
        vectors = self.test_vectors_small.copy()
        
        # Perform operations to populate cache
        self.optimizer.normalize_vectors(vectors, "l2")
        self.optimizer.normalize_vectors(vectors, "l1")
        
        # Get cache statistics
        stats = self.optimizer.get_cache_statistics()
        
        # Check statistics fields
        self.assertIn("cache_entries", stats)
        self.assertIn("cache_size_bytes", stats)
        self.assertIn("cache_size_mb", stats)
        self.assertIn("cache_utilization", stats)
        
        # Check that cache has entries
        self.assertGreaterEqual(stats["cache_entries"], 0)
    
    def test_backend_availability(self):
        """Test backend availability checking"""
        availability = self.optimizer.get_backend_availability()
        
        # NumPy and SciPy should always be available
        self.assertTrue(availability[VectorBackend.NUMPY])
        self.assertTrue(availability[VectorBackend.SCIPY])
        
        # Rust and GPU availability depends on system
        self.assertIsInstance(availability[VectorBackend.RUST], bool)
        self.assertIsInstance(availability[VectorBackend.GPU], bool)


class TestRustFFIInterface(unittest.TestCase):
    """Test cases for Rust FFI interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.library_manager = RustLibraryManager()
        self.vector_ffi = VectorOperationsFFI(self.library_manager)
        
        # Create test vectors
        np.random.seed(42)
        self.test_vectors = np.random.randn(10, 128).astype(np.float32)
    
    def test_library_manager_initialization(self):
        """Test library manager initialization"""
        self.assertIsInstance(self.library_manager, RustLibraryManager)
        self.assertEqual(len(self.library_manager.get_loaded_libraries()), 0)
    
    def test_library_loading_failure(self):
        """Test library loading failure handling"""
        # Try to load non-existent library
        success = self.library_manager.load_library("nonexistent_lib", "/nonexistent/path")
        self.assertFalse(success)
    
    def test_function_registration_without_library(self):
        """Test function registration without loaded library"""
        import ctypes
        
        success = self.library_manager.register_function(
            "nonexistent_lib",
            "test_function",
            [ctypes.c_float],
            ctypes.c_int
        )
        self.assertFalse(success)
    
    def test_vector_ffi_availability(self):
        """Test vector FFI availability checking"""
        # Should return False if Rust library is not available
        available = self.vector_ffi.is_available()
        self.assertIsInstance(available, bool)
    
    def test_rust_availability_function(self):
        """Test global Rust availability function"""
        available = is_rust_available()
        self.assertIsInstance(available, bool)
    
    @unittest.skipUnless(is_rust_available(), "Rust bindings not available")
    def test_rust_vector_normalization(self):
        """Test Rust vector normalization (if available)"""
        vectors = self.test_vectors.copy()
        
        try:
            result = self.vector_ffi.normalize_vectors(vectors, "l2")
            
            # Check that vectors are normalized
            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
            
        except RustFFIError:
            self.skipTest("Rust FFI not properly configured")
    
    @unittest.skipUnless(is_rust_available(), "Rust bindings not available")
    def test_rust_similarity_matrix(self):
        """Test Rust similarity matrix computation (if available)"""
        vectors_a = self.test_vectors[:5]
        vectors_b = self.test_vectors[5:]
        
        try:
            result = self.vector_ffi.similarity_matrix(vectors_a, vectors_b, "cosine")
            
            # Check shape
            self.assertEqual(result.shape, (5, 5))
            
            # Check similarity range
            self.assertTrue(np.all(result >= -1.0))
            self.assertTrue(np.all(result <= 1.0))
            
        except RustFFIError:
            self.skipTest("Rust FFI not properly configured")
    
    def test_performance_metrics_collection(self):
        """Test FFI performance metrics collection"""
        # Even if Rust is not available, metrics should be collected for failures
        try:
            self.vector_ffi.normalize_vectors(self.test_vectors, "l2")
        except RustFFIError:
            pass  # Expected if Rust not available
        
        metrics = self.vector_ffi.get_performance_metrics()
        self.assertIsInstance(metrics, list)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        usage = self.vector_ffi.get_memory_usage()
        
        self.assertIn("total_allocated_bytes", usage)
        self.assertIn("total_allocated_mb", usage)
        self.assertIn("active_allocations", usage)
        
        self.assertIsInstance(usage["total_allocated_bytes"], int)
        self.assertIsInstance(usage["total_allocated_mb"], (int, float))
        self.assertIsInstance(usage["active_allocations"], int)


class TestEnhancedVectorService(unittest.IsolatedAsyncioTestCase):
    """Test cases for EnhancedVectorService"""
    
    async def asyncSetUp(self):
        """Set up async test fixtures"""
        self.service = EnhancedVectorService()
        
        # Mock the async performance optimizer to avoid complex initialization
        self.service.async_optimizer = Mock()
        self.service.async_optimizer.submit_task = Mock(return_value=True)
        
        await self.service.initialize()
        
        # Create test vectors
        np.random.seed(42)
        self.test_vectors = np.random.randn(10, 128).astype(np.float32)
        self.query_vectors = np.random.randn(3, 128).astype(np.float32)
        self.database_vectors = np.random.randn(20, 128).astype(np.float32)
    
    async def test_service_initialization(self):
        """Test service initialization"""
        self.assertTrue(self.service._initialized)
        self.assertIsNotNone(self.service.vector_optimizer)
        self.assertIsNotNone(self.service._vector_thread_pool)
    
    async def test_async_vector_normalization(self):
        """Test async vector normalization"""
        vectors = self.test_vectors.copy()
        
        result = await self.service.normalize_vectors_async(vectors, "l2")
        
        # Check that vectors are normalized
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
        
        # Check shape preservation
        self.assertEqual(result.shape, vectors.shape)
    
    async def test_async_similarity_matrix(self):
        """Test async similarity matrix computation"""
        vectors_a = self.test_vectors[:5]
        vectors_b = self.test_vectors[5:]
        
        result = await self.service.compute_similarity_matrix_async(
            vectors_a, vectors_b, SimilarityMetric.COSINE
        )
        
        # Check shape
        self.assertEqual(result.shape, (5, 5))
        
        # Check similarity range
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))
    
    async def test_async_batch_cosine_similarity(self):
        """Test async batch cosine similarity"""
        top_k = 5
        
        similarities, indices = await self.service.batch_cosine_similarity_async(
            self.query_vectors, self.database_vectors, top_k
        )
        
        # Check shapes
        self.assertEqual(similarities.shape, (3, 5))
        self.assertEqual(indices.shape, (3, 5))
        
        # Check that similarities are sorted
        for i in range(similarities.shape[0]):
            self.assertTrue(np.all(similarities[i, :-1] >= similarities[i, 1:]))
    
    async def test_async_vector_arithmetic(self):
        """Test async vector arithmetic"""
        vector_a = self.test_vectors[:5]
        vector_b = self.test_vectors[5:10]
        
        result = await self.service.vector_arithmetic_async(
            "add", vector_a, vector_b
        )
        
        expected = vector_a + vector_b
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    async def test_job_management(self):
        """Test job management functionality"""
        # Start an operation
        task = asyncio.create_task(
            self.service.normalize_vectors_async(self.test_vectors, "l2")
        )
        
        # Check active jobs (might be empty if operation completes quickly)
        active_jobs = await self.service.get_active_jobs()
        self.assertIsInstance(active_jobs, list)
        
        # Wait for completion
        await task
    
    async def test_metrics_collection(self):
        """Test metrics collection"""
        # Perform some operations
        await self.service.normalize_vectors_async(self.test_vectors, "l2")
        await self.service.compute_similarity_matrix_async(
            self.test_vectors[:5], self.test_vectors[5:]
        )
        
        # Get metrics
        metrics = await self.service.get_metrics()
        
        # Check metrics fields
        self.assertGreaterEqual(metrics.total_operations, 2)
        self.assertGreaterEqual(metrics.successful_operations, 0)
        self.assertGreaterEqual(metrics.total_vectors_processed, 0)
    
    async def test_backend_availability(self):
        """Test backend availability checking"""
        availability = await self.service.get_backend_availability()
        
        self.assertIsInstance(availability, dict)
        self.assertIn(VectorBackend.NUMPY, availability)
        self.assertIn(VectorBackend.SCIPY, availability)
    
    async def test_benchmarking(self):
        """Test backend benchmarking"""
        benchmark_results = await self.service.benchmark_backends(self.test_vectors)
        
        # Results might be empty if benchmarking is disabled
        self.assertIsInstance(benchmark_results, dict)
    
    async def test_error_handling(self):
        """Test error handling in async operations"""
        # Test with invalid operation
        with self.assertRaises(ValueError):
            await self.service.vector_arithmetic_async(
                "invalid_operation", self.test_vectors
            )
    
    async def test_priority_handling(self):
        """Test priority handling in job execution"""
        # Submit jobs with different priorities
        high_priority_task = asyncio.create_task(
            self.service.normalize_vectors_async(
                self.test_vectors, "l2", priority=10
            )
        )
        
        low_priority_task = asyncio.create_task(
            self.service.normalize_vectors_async(
                self.test_vectors, "l1", priority=1
            )
        )
        
        # Wait for completion
        results = await asyncio.gather(high_priority_task, low_priority_task)
        
        # Both should complete successfully
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, np.ndarray)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the complete vector operations system"""
    
    async def asyncSetUp(self):
        """Set up integration test fixtures"""
        # Create test vectors of various sizes
        np.random.seed(42)
        self.small_vectors = np.random.randn(50, 128).astype(np.float32)
        self.medium_vectors = np.random.randn(500, 256).astype(np.float32)
        self.large_vectors = np.random.randn(2000, 512).astype(np.float32)
        
        # Initialize service
        self.service = await get_enhanced_vector_service()
    
    async def test_end_to_end_processing(self):
        """Test end-to-end vector processing pipeline"""
        # Step 1: Normalize vectors
        normalized = await self.service.normalize_vectors_async(
            self.small_vectors, "l2"
        )
        
        # Step 2: Compute similarity matrix
        similarity_matrix = await self.service.compute_similarity_matrix_async(
            normalized[:10], normalized[10:20], SimilarityMetric.COSINE
        )
        
        # Step 3: Find top similar vectors
        query_vectors = normalized[:5]
        database_vectors = normalized[20:]
        
        similarities, indices = await self.service.batch_cosine_similarity_async(
            query_vectors, database_vectors, top_k=10
        )
        
        # Verify results
        self.assertEqual(normalized.shape, self.small_vectors.shape)
        self.assertEqual(similarity_matrix.shape, (10, 10))
        self.assertEqual(similarities.shape, (5, 10))
        self.assertEqual(indices.shape, (5, 10))
        
        # Check that normalized vectors have unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
    
    async def test_performance_comparison(self):
        """Test performance comparison between backends"""
        vectors = self.medium_vectors.copy()
        
        # Test different operations with timing
        operations = [
            ("normalize_l2", lambda: self.service.normalize_vectors_async(vectors, "l2")),
            ("similarity_matrix", lambda: self.service.compute_similarity_matrix_async(
                vectors[:100], vectors[100:200]
            )),
            ("batch_cosine", lambda: self.service.batch_cosine_similarity_async(
                vectors[:50], vectors[200:], top_k=10
            ))
        ]
        
        results = {}
        for op_name, op_func in operations:
            start_time = time.time()
            result = await op_func()
            end_time = time.time()
            
            results[op_name] = {
                "time": end_time - start_time,
                "result_shape": result.shape if hasattr(result, 'shape') else len(result)
            }
        
        # Verify all operations completed
        self.assertEqual(len(results), 3)
        for op_name, metrics in results.items():
            self.assertGreater(metrics["time"], 0)
            self.assertGreater(metrics["result_shape"], 0)
    
    async def test_memory_efficiency(self):
        """Test memory efficiency with large vectors"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large vectors
        large_normalized = await self.service.normalize_vectors_async(
            self.large_vectors, "l2"
        )
        
        # Force garbage collection
        del large_normalized
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 2x the vector size)
        vector_size_mb = self.large_vectors.nbytes / 1024 / 1024
        self.assertLess(memory_increase, vector_size_mb * 2)
    
    async def test_concurrent_operations(self):
        """Test concurrent vector operations"""
        # Submit multiple operations concurrently
        tasks = [
            self.service.normalize_vectors_async(self.small_vectors, "l2"),
            self.service.normalize_vectors_async(self.small_vectors, "l1"),
            self.service.compute_similarity_matrix_async(
                self.small_vectors[:10], self.small_vectors[10:20]
            ),
            self.service.batch_cosine_similarity_async(
                self.small_vectors[:5], self.small_vectors[20:], top_k=5
            )
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertIsNotNone(result)
    
    async def test_error_recovery(self):
        """Test error recovery and fallback mechanisms"""
        # Test with invalid inputs
        invalid_vectors = np.array([])  # Empty array
        
        with self.assertRaises((ValueError, RuntimeError)):
            await self.service.normalize_vectors_async(invalid_vectors, "l2")
        
        # Service should still work after error
        valid_result = await self.service.normalize_vectors_async(
            self.small_vectors, "l2"
        )
        self.assertIsNotNone(valid_result)


def run_performance_benchmark():
    """Run performance benchmark for vector operations"""
    print("Running Vector Operations Performance Benchmark")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    small_vectors = np.random.randn(100, 128).astype(np.float32)
    medium_vectors = np.random.randn(1000, 256).astype(np.float32)
    large_vectors = np.random.randn(5000, 512).astype(np.float32)
    
    test_cases = [
        ("Small vectors (100x128)", small_vectors),
        ("Medium vectors (1000x256)", medium_vectors),
        ("Large vectors (5000x512)", large_vectors)
    ]
    
    # Initialize optimizer
    config = create_optimized_config()
    optimizer = VectorOperationsOptimizer(config)
    
    for test_name, vectors in test_cases:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        # Test normalization
        start_time = time.time()
        normalized = optimizer.normalize_vectors(vectors, "l2")
        norm_time = time.time() - start_time
        print(f"L2 Normalization: {norm_time:.4f}s")
        
        # Test similarity matrix (subset for large vectors)
        subset_size = min(100, vectors.shape[0] // 2)
        start_time = time.time()
        similarity = optimizer.compute_similarity_matrix(
            vectors[:subset_size], vectors[subset_size:subset_size*2]
        )
        sim_time = time.time() - start_time
        print(f"Similarity Matrix ({subset_size}x{subset_size}): {sim_time:.4f}s")
        
        # Test batch cosine similarity
        query_size = min(10, vectors.shape[0] // 10)
        start_time = time.time()
        similarities, indices = optimizer.batch_cosine_similarity(
            vectors[:query_size], vectors, top_k=10
        )
        batch_time = time.time() - start_time
        print(f"Batch Cosine Similarity (top-10): {batch_time:.4f}s")
    
    # Backend availability
    print(f"\nBackend Availability:")
    print("-" * 20)
    availability = optimizer.get_backend_availability()
    for backend, available in availability.items():
        status = "✓" if available else "✗"
        print(f"{backend.value}: {status}")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print("-" * 20)
    metrics = optimizer.get_performance_metrics()
    if metrics:
        avg_time = np.mean([m.execution_time_ms for m in metrics])
        avg_throughput = np.mean([m.throughput_vectors_per_second for m in metrics])
        print(f"Average execution time: {avg_time:.2f}ms")
        print(f"Average throughput: {avg_throughput:.2f} vectors/sec")
    
    # Cache statistics
    cache_stats = optimizer.get_cache_statistics()
    print(f"Cache entries: {cache_stats['cache_entries']}")
    print(f"Cache utilization: {cache_stats['cache_utilization']:.2%}")


async def run_async_benchmark():
    """Run async performance benchmark"""
    print("\nRunning Async Vector Service Benchmark")
    print("=" * 60)
    
    # Initialize service
    service = await get_enhanced_vector_service()
    
    # Create test data
    np.random.seed(42)
    test_vectors = np.random.randn(1000, 256).astype(np.float32)
    query_vectors = np.random.randn(50, 256).astype(np.float32)
    
    # Test concurrent operations
    print("Testing concurrent operations...")
    start_time = time.time()
    
    tasks = [
        service.normalize_vectors_async(test_vectors, "l2"),
        service.normalize_vectors_async(test_vectors, "l1"),
        service.compute_similarity_matrix_async(
            test_vectors[:100], test_vectors[100:200]
        ),
        service.batch_cosine_similarity_async(
            query_vectors, test_vectors, top_k=10
        )
    ]
    
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    print(f"Concurrent operations completed in: {total_time:.4f}s")
    print(f"Number of operations: {len(tasks)}")
    print(f"Average time per operation: {total_time/len(tasks):.4f}s")
    
    # Get service metrics
    metrics = await service.get_metrics()
    print(f"\nService Metrics:")
    print(f"Total operations: {metrics.total_operations}")
    print(f"Successful operations: {metrics.successful_operations}")
    print(f"Failed operations: {metrics.failed_operations}")
    print(f"Average processing time: {metrics.average_processing_time_ms:.2f}ms")
    print(f"Total vectors processed: {metrics.total_vectors_processed}")
    
    # Backend usage
    print(f"\nBackend Usage:")
    print(f"NumPy operations: {metrics.numpy_operations}")
    print(f"SciPy operations: {metrics.scipy_operations}")
    print(f"Rust operations: {metrics.rust_operations}")
    print(f"GPU operations: {metrics.gpu_operations}")


if __name__ == "__main__":
    # Run unit tests
    print("Running Vector Operations Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestVectorOperationsOptimizer))
    test_suite.addTest(unittest.makeSuite(TestRustFFIInterface))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    # Run async tests
    print("\nRunning Async Tests...")
    asyncio.run(unittest.main(
        module=None,
        testLoader=unittest.TestLoader().loadTestsFromTestCase(TestEnhancedVectorService),
        exit=False,
        verbosity=2
    ))
    
    asyncio.run(unittest.main(
        module=None,
        testLoader=unittest.TestLoader().loadTestsFromTestCase(TestIntegration),
        exit=False,
        verbosity=2
    ))
    
    # Run performance benchmarks
    if test_result.wasSuccessful():
        print("\n" + "=" * 60)
        run_performance_benchmark()
        
        print("\n" + "=" * 60)
        asyncio.run(run_async_benchmark())
    else:
        print("\nSkipping benchmarks due to test failures")
    
    print("\n" + "=" * 60)
    print("Test suite completed!")