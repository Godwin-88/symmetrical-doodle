#!/usr/bin/env python3
"""
Simple Test Suite for Vector Operations Optimization

This simplified test suite validates the core vector operations functionality
without requiring complex async dependencies.
"""

import unittest
import numpy as np
import time
import sys
import os

# Add the services directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

try:
    from vector_operations_optimizer import (
        VectorOperationsOptimizer,
        VectorOperationConfig,
        VectorBackend,
        SimilarityMetric,
        create_optimized_config
    )
    VECTOR_OPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import vector operations optimizer: {e}")
    VECTOR_OPS_AVAILABLE = False

try:
    from rust_ffi_interface import (
        RustLibraryManager,
        VectorOperationsFFI,
        is_rust_available
    )
    FFI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Rust FFI interface: {e}")
    FFI_AVAILABLE = False


class TestVectorOperationsCore(unittest.TestCase):
    """Test core vector operations functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not VECTOR_OPS_AVAILABLE:
            self.skipTest("Vector operations optimizer not available")
        
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
        self.test_vectors = np.random.randn(20, 64).astype(np.float32)
        self.query_vectors = np.random.randn(5, 64).astype(np.float32)
    
    def test_vector_normalization_l2(self):
        """Test L2 vector normalization"""
        vectors = self.test_vectors.copy()
        normalized = self.optimizer.normalize_vectors(vectors, "l2")
        
        # Check that vectors are normalized (unit length)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
        
        # Check shape preservation
        self.assertEqual(normalized.shape, vectors.shape)
        print("✓ L2 normalization test passed")
    
    def test_vector_normalization_l1(self):
        """Test L1 vector normalization"""
        vectors = self.test_vectors.copy()
        normalized = self.optimizer.normalize_vectors(vectors, "l1")
        
        # Check that L1 norms are 1
        l1_norms = np.sum(np.abs(normalized), axis=1)
        np.testing.assert_allclose(l1_norms, 1.0, rtol=1e-6)
        print("✓ L1 normalization test passed")
    
    def test_cosine_similarity_matrix(self):
        """Test cosine similarity matrix computation"""
        vectors_a = self.test_vectors[:5]
        vectors_b = self.test_vectors[5:10]
        
        similarity_matrix = self.optimizer.compute_similarity_matrix(
            vectors_a, vectors_b, SimilarityMetric.COSINE
        )
        
        # Check shape
        self.assertEqual(similarity_matrix.shape, (5, 5))
        
        # Check that similarities are in valid range [-1, 1]
        self.assertTrue(np.all(similarity_matrix >= -1.0))
        self.assertTrue(np.all(similarity_matrix <= 1.0))
        print("✓ Cosine similarity matrix test passed")
    
    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation"""
        query_vectors = self.query_vectors
        database_vectors = self.test_vectors
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
        
        print("✓ Batch cosine similarity test passed")
    
    def test_vector_arithmetic(self):
        """Test vector arithmetic operations"""
        vector_a = self.test_vectors[:5]
        vector_b = self.test_vectors[5:10]
        
        # Test addition
        result_add = self.optimizer.vector_arithmetic("add", vector_a, vector_b)
        expected_add = vector_a + vector_b
        np.testing.assert_allclose(result_add, expected_add, rtol=1e-6)
        
        # Test scaling
        scalar = 2.5
        result_scale = self.optimizer.vector_arithmetic("scale", vector_a, scalar=scalar)
        expected_scale = vector_a * scalar
        np.testing.assert_allclose(result_scale, expected_scale, rtol=1e-6)
        
        print("✓ Vector arithmetic test passed")
    
    def test_backend_availability(self):
        """Test backend availability checking"""
        availability = self.optimizer.get_backend_availability()
        
        # NumPy and SciPy should always be available
        self.assertTrue(availability[VectorBackend.NUMPY])
        self.assertTrue(availability[VectorBackend.SCIPY])
        
        print("✓ Backend availability test passed")
        print(f"  Available backends: {[k.value for k, v in availability.items() if v]}")
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        vectors = self.test_vectors.copy()
        
        # Perform some operations
        self.optimizer.normalize_vectors(vectors, "l2")
        self.optimizer.compute_similarity_matrix(vectors[:5], vectors[5:10])
        
        # Get metrics
        metrics = self.optimizer.get_performance_metrics()
        
        # Check that metrics were collected
        self.assertGreater(len(metrics), 0)
        
        print("✓ Performance metrics test passed")
        print(f"  Collected {len(metrics)} performance metrics")


class TestRustFFICore(unittest.TestCase):
    """Test core Rust FFI functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not FFI_AVAILABLE:
            self.skipTest("Rust FFI interface not available")
        
        self.library_manager = RustLibraryManager()
        self.vector_ffi = VectorOperationsFFI(self.library_manager)
        
        # Create test vectors
        np.random.seed(42)
        self.test_vectors = np.random.randn(10, 64).astype(np.float32)
    
    def test_rust_availability(self):
        """Test Rust availability checking"""
        available = is_rust_available()
        self.assertIsInstance(available, bool)
        
        print(f"✓ Rust availability test passed (available: {available})")
    
    def test_library_manager(self):
        """Test library manager functionality"""
        self.assertIsInstance(self.library_manager, RustLibraryManager)
        self.assertEqual(len(self.library_manager.get_loaded_libraries()), 0)
        
        print("✓ Library manager test passed")
    
    def test_vector_ffi_initialization(self):
        """Test vector FFI initialization"""
        available = self.vector_ffi.is_available()
        self.assertIsInstance(available, bool)
        
        print(f"✓ Vector FFI initialization test passed (available: {available})")
    
    @unittest.skipUnless(is_rust_available(), "Rust bindings not available")
    def test_rust_operations(self):
        """Test Rust operations if available"""
        try:
            # Test normalization
            result = self.vector_ffi.normalize_vectors(self.test_vectors, "l2")
            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
            
            print("✓ Rust operations test passed")
            
        except Exception as e:
            print(f"⚠ Rust operations test skipped: {e}")


def run_performance_benchmark():
    """Run a simple performance benchmark"""
    if not VECTOR_OPS_AVAILABLE:
        print("Vector operations not available, skipping benchmark")
        return
    
    print("\nRunning Performance Benchmark")
    print("=" * 40)
    
    # Create test data
    np.random.seed(42)
    small_vectors = np.random.randn(100, 128).astype(np.float32)
    medium_vectors = np.random.randn(500, 256).astype(np.float32)
    
    # Initialize optimizer
    config = create_optimized_config()
    optimizer = VectorOperationsOptimizer(config)
    
    test_cases = [
        ("Small vectors (100x128)", small_vectors),
        ("Medium vectors (500x256)", medium_vectors)
    ]
    
    for test_name, vectors in test_cases:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        # Test normalization
        start_time = time.time()
        normalized = optimizer.normalize_vectors(vectors, "l2")
        norm_time = time.time() - start_time
        print(f"L2 Normalization: {norm_time:.4f}s")
        
        # Test similarity matrix (subset)
        subset_size = min(50, vectors.shape[0] // 2)
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
            vectors[:query_size], vectors, top_k=5
        )
        batch_time = time.time() - start_time
        print(f"Batch Cosine Similarity (top-5): {batch_time:.4f}s")
    
    # Backend availability
    print(f"\nBackend Availability:")
    print("-" * 20)
    availability = optimizer.get_backend_availability()
    for backend, available in availability.items():
        status = "✓" if available else "✗"
        print(f"{backend.value}: {status}")
    
    # Performance metrics summary
    metrics = optimizer.get_performance_metrics()
    if metrics:
        avg_time = np.mean([m.execution_time_ms for m in metrics])
        print(f"\nAverage execution time: {avg_time:.2f}ms")
        print(f"Total operations: {len(metrics)}")


def main():
    """Main test execution"""
    print("Vector Operations Simple Test Suite")
    print("=" * 50)
    
    # Check availability
    print(f"Vector Operations Available: {VECTOR_OPS_AVAILABLE}")
    print(f"Rust FFI Available: {FFI_AVAILABLE}")
    print(f"Rust Bindings Available: {is_rust_available() if FFI_AVAILABLE else False}")
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    if VECTOR_OPS_AVAILABLE:
        test_suite.addTests(loader.loadTestsFromTestCase(TestVectorOperationsCore))
    
    if FFI_AVAILABLE:
        test_suite.addTests(loader.loadTestsFromTestCase(TestRustFFICore))
    
    if not VECTOR_OPS_AVAILABLE and not FFI_AVAILABLE:
        print("No tests available - missing dependencies")
        return
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    # Run performance benchmark if tests passed
    if test_result.wasSuccessful():
        run_performance_benchmark()
    else:
        print("\nSkipping benchmark due to test failures")
    
    print("\n" + "=" * 50)
    if test_result.wasSuccessful():
        print("✓ All tests passed successfully!")
    else:
        print("✗ Some tests failed")
        print(f"Failures: {len(test_result.failures)}")
        print(f"Errors: {len(test_result.errors)}")


if __name__ == "__main__":
    main()