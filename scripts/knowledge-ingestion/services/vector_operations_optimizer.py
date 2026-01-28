"""
Vector Operations Optimization Service

This module provides optimized vector operations for embedding processing using
NumPy/SciPy optimizations and optional Rust bindings for performance-critical
mathematical computations.

Features:
- NumPy/SciPy optimized vector operations
- Optional Rust bindings for mathematical computations
- Python-Rust FFI for seamless data exchange
- Performance benchmarking and automatic fallback
- Similarity calculations and vector operations
- Memory-efficient batch processing
- GPU acceleration support (when available)

Requirements: 10.3, 10.5
"""

import os
import sys
import time
import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import scipy.spatial.distance as distance
import scipy.sparse as sparse
from scipy.optimize import minimize_scalar
from scipy.linalg import norm
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from contextlib import contextmanager

from core.config import get_settings
from core.logging import get_logger


class VectorBackend(Enum):
    """Available vector computation backends"""
    NUMPY = "numpy"
    SCIPY = "scipy"
    RUST = "rust"
    GPU = "gpu"
    AUTO = "auto"


class SimilarityMetric(Enum):
    """Supported similarity metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"
    JACCARD = "jaccard"
    HAMMING = "hamming"


@dataclass
class VectorOperationConfig:
    """Configuration for vector operations"""
    backend: VectorBackend = VectorBackend.AUTO
    enable_rust_bindings: bool = True
    enable_gpu_acceleration: bool = True
    batch_size: int = 1000
    num_threads: int = mp.cpu_count()
    memory_limit_mb: int = 2048
    use_sparse_matrices: bool = False
    precision: str = "float32"  # float16, float32, float64
    enable_benchmarking: bool = True
    fallback_on_error: bool = True
    cache_computations: bool = True
    cache_size_mb: int = 512


@dataclass
class PerformanceMetrics:
    """Performance metrics for vector operations"""
    operation_name: str
    backend_used: VectorBackend
    execution_time_ms: float
    memory_usage_mb: float
    throughput_vectors_per_second: float
    batch_size: int
    vector_dimension: int
    num_vectors: int
    cache_hit_rate: float = 0.0
    gpu_utilization: Optional[float] = None
    rust_binding_overhead_ms: float = 0.0


class RustBindingError(Exception):
    """Exception raised when Rust bindings fail"""
    pass


class VectorOperationsOptimizer:
    """
    Main vector operations optimizer that provides high-performance
    vector computations with multiple backend support.
    """
    
    def __init__(self, config: Optional[VectorOperationConfig] = None):
        self.config = config or VectorOperationConfig()
        self.logger = get_logger(__name__)
        
        # Backend availability
        self._rust_available = False
        self._gpu_available = False
        self._rust_module = None
        
        # Performance tracking
        self._performance_metrics: List[PerformanceMetrics] = []
        self._metrics_lock = threading.Lock()
        
        # Computation cache
        self._computation_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._cache_size_bytes = 0
        
        # Thread pool for parallel operations
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        
        # Initialize backends
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize available computation backends"""
        try:
            # Check Rust bindings availability
            if self.config.enable_rust_bindings:
                self._rust_available = self._check_rust_bindings()
                if self._rust_available:
                    self.logger.info("Rust bindings available for vector operations")
                else:
                    self.logger.info("Rust bindings not available, using Python implementations")
            
            # Check GPU availability
            if self.config.enable_gpu_acceleration:
                self._gpu_available = self._check_gpu_availability()
                if self._gpu_available:
                    self.logger.info("GPU acceleration available for vector operations")
                else:
                    self.logger.info("GPU acceleration not available")
            
            # Initialize thread pool
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.config.num_threads,
                thread_name_prefix="vector_ops"
            )
            
            self.logger.info(f"Vector operations optimizer initialized with {self.config.num_threads} threads")
            
        except Exception as e:
            self.logger.error(f"Error initializing vector operations backends: {e}")
            if not self.config.fallback_on_error:
                raise
    
    def _check_rust_bindings(self) -> bool:
        """Check if Rust bindings are available"""
        try:
            # Try to import the Rust module
            # This would be a compiled Rust extension module
            # For now, we'll simulate this check
            rust_module_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "rust_bindings", 
                "vector_ops.so"  # or .pyd on Windows
            )
            
            if os.path.exists(rust_module_path):
                # Try to load the module
                import importlib.util
                spec = importlib.util.spec_from_file_location("vector_ops", rust_module_path)
                if spec and spec.loader:
                    self._rust_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self._rust_module)
                    return True
            
            # Alternative: try to import from installed package
            try:
                import vector_ops_rust
                self._rust_module = vector_ops_rust
                return True
            except ImportError:
                pass
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Rust bindings check failed: {e}")
            return False
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            # Check for CuPy (CUDA)
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    return True
            except ImportError:
                pass
            
            # Check for PyTorch with CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    return True
            except ImportError:
                pass
            
            # Check for TensorFlow with GPU
            try:
                import tensorflow as tf
                if len(tf.config.list_physical_devices('GPU')) > 0:
                    return True
            except ImportError:
                pass
            
            return False
            
        except Exception as e:
            self.logger.debug(f"GPU availability check failed: {e}")
            return False
    
    def _select_optimal_backend(
        self, 
        operation: str, 
        vector_shape: Tuple[int, ...],
        data_type: str = "float32"
    ) -> VectorBackend:
        """Select optimal backend for a given operation"""
        if self.config.backend != VectorBackend.AUTO:
            return self.config.backend
        
        num_vectors, vector_dim = vector_shape if len(vector_shape) == 2 else (1, vector_shape[0])
        
        # Decision logic based on operation characteristics
        if operation in ["similarity_matrix", "batch_cosine_similarity"] and num_vectors > 10000:
            # Large batch operations benefit from GPU
            if self._gpu_available:
                return VectorBackend.GPU
            elif self._rust_available and vector_dim > 512:
                return VectorBackend.RUST
            else:
                return VectorBackend.SCIPY
        
        elif operation in ["vector_normalize", "vector_add", "vector_subtract"] and vector_dim > 1024:
            # High-dimensional operations benefit from Rust
            if self._rust_available:
                return VectorBackend.RUST
            else:
                return VectorBackend.NUMPY
        
        elif operation in ["cosine_similarity", "euclidean_distance"] and num_vectors < 100:
            # Small operations can use NumPy efficiently
            return VectorBackend.NUMPY
        
        else:
            # Default to SciPy for general operations
            return VectorBackend.SCIPY
    
    @contextmanager
    def _performance_tracking(self, operation_name: str, backend: VectorBackend, **kwargs):
        """Context manager for tracking operation performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = max(0, end_memory - start_memory)
            
            # Calculate throughput if vector count is provided
            num_vectors = kwargs.get('num_vectors', 1)
            throughput = num_vectors / max(0.001, end_time - start_time)
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                backend_used=backend,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                throughput_vectors_per_second=throughput,
                batch_size=kwargs.get('batch_size', 1),
                vector_dimension=kwargs.get('vector_dimension', 0),
                num_vectors=num_vectors
            )
            
            with self._metrics_lock:
                self._performance_metrics.append(metrics)
                # Keep only recent metrics
                if len(self._performance_metrics) > 1000:
                    self._performance_metrics = self._performance_metrics[-500:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key for operation"""
        import hashlib
        
        # Create a string representation of the operation and parameters
        key_data = f"{operation}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_result(self, key: str, result: Any):
        """Cache computation result"""
        if not self.config.cache_computations:
            return
        
        try:
            # Estimate result size
            result_size = sys.getsizeof(result)
            if hasattr(result, 'nbytes'):  # NumPy arrays
                result_size = result.nbytes
            
            with self._cache_lock:
                # Check cache size limit
                max_cache_bytes = self.config.cache_size_mb * 1024 * 1024
                
                if self._cache_size_bytes + result_size > max_cache_bytes:
                    # Clear some cache entries
                    self._clear_cache_entries(result_size)
                
                self._computation_cache[key] = result
                self._cache_size_bytes += result_size
                
        except Exception as e:
            self.logger.debug(f"Failed to cache result: {e}")
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached computation result"""
        if not self.config.cache_computations:
            return None
        
        with self._cache_lock:
            return self._computation_cache.get(key)
    
    def _clear_cache_entries(self, needed_bytes: int):
        """Clear cache entries to make space"""
        # Simple LRU-like clearing - remove oldest entries
        # In a production system, you'd want a proper LRU implementation
        entries_to_remove = max(1, len(self._computation_cache) // 4)
        
        keys_to_remove = list(self._computation_cache.keys())[:entries_to_remove]
        for key in keys_to_remove:
            result = self._computation_cache.pop(key, None)
            if result is not None and hasattr(result, 'nbytes'):
                self._cache_size_bytes -= result.nbytes
    
    # Core Vector Operations
    
    def normalize_vectors(
        self, 
        vectors: np.ndarray, 
        norm_type: str = "l2",
        backend: Optional[VectorBackend] = None
    ) -> np.ndarray:
        """
        Normalize vectors using specified norm.
        
        Args:
            vectors: Input vectors (2D array: [num_vectors, vector_dim])
            norm_type: Type of normalization ('l1', 'l2', 'max')
            backend: Specific backend to use
            
        Returns:
            Normalized vectors
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        backend = backend or self._select_optimal_backend(
            "vector_normalize", 
            vectors.shape,
            str(vectors.dtype)
        )
        
        cache_key = self._get_cache_key("normalize", vectors.tobytes(), norm_type, backend.value)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with self._performance_tracking(
            "normalize_vectors", 
            backend,
            num_vectors=vectors.shape[0],
            vector_dimension=vectors.shape[1]
        ):
            if backend == VectorBackend.RUST and self._rust_available:
                result = self._normalize_vectors_rust(vectors, norm_type)
            elif backend == VectorBackend.GPU and self._gpu_available:
                result = self._normalize_vectors_gpu(vectors, norm_type)
            elif backend == VectorBackend.SCIPY:
                result = self._normalize_vectors_scipy(vectors, norm_type)
            else:  # NumPy fallback
                result = self._normalize_vectors_numpy(vectors, norm_type)
        
        self._cache_result(cache_key, result)
        return result
    
    def _normalize_vectors_numpy(self, vectors: np.ndarray, norm_type: str) -> np.ndarray:
        """NumPy implementation of vector normalization"""
        if norm_type == "l2":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            return vectors / norms
        elif norm_type == "l1":
            norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return vectors / norms
        elif norm_type == "max":
            norms = np.max(np.abs(vectors), axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return vectors / norms
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
    
    def _normalize_vectors_scipy(self, vectors: np.ndarray, norm_type: str) -> np.ndarray:
        """SciPy implementation of vector normalization"""
        if norm_type == "l2":
            return vectors / norm(vectors, axis=1, keepdims=True)
        else:
            # Fall back to NumPy for other norms
            return self._normalize_vectors_numpy(vectors, norm_type)
    
    def _normalize_vectors_rust(self, vectors: np.ndarray, norm_type: str) -> np.ndarray:
        """Rust implementation of vector normalization"""
        if not self._rust_module:
            raise RustBindingError("Rust module not available")
        
        try:
            # Convert to appropriate format for Rust
            vectors_f32 = vectors.astype(np.float32)
            
            # Call Rust function
            if hasattr(self._rust_module, 'normalize_vectors'):
                result = self._rust_module.normalize_vectors(vectors_f32, norm_type)
                return result.astype(vectors.dtype)
            else:
                raise RustBindingError("normalize_vectors function not found in Rust module")
                
        except Exception as e:
            if self.config.fallback_on_error:
                self.logger.warning(f"Rust normalization failed, falling back to NumPy: {e}")
                return self._normalize_vectors_numpy(vectors, norm_type)
            else:
                raise RustBindingError(f"Rust normalization failed: {e}")
    
    def _normalize_vectors_gpu(self, vectors: np.ndarray, norm_type: str) -> np.ndarray:
        """GPU implementation of vector normalization"""
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_vectors = cp.asarray(vectors)
            
            if norm_type == "l2":
                norms = cp.linalg.norm(gpu_vectors, axis=1, keepdims=True)
                norms = cp.where(norms == 0, 1, norms)
                result = gpu_vectors / norms
            elif norm_type == "l1":
                norms = cp.sum(cp.abs(gpu_vectors), axis=1, keepdims=True)
                norms = cp.where(norms == 0, 1, norms)
                result = gpu_vectors / norms
            elif norm_type == "max":
                norms = cp.max(cp.abs(gpu_vectors), axis=1, keepdims=True)
                norms = cp.where(norms == 0, 1, norms)
                result = gpu_vectors / norms
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
            
            # Transfer back to CPU
            return cp.asnumpy(result)
            
        except ImportError:
            if self.config.fallback_on_error:
                self.logger.warning("CuPy not available, falling back to NumPy")
                return self._normalize_vectors_numpy(vectors, norm_type)
            else:
                raise
        except Exception as e:
            if self.config.fallback_on_error:
                self.logger.warning(f"GPU normalization failed, falling back to NumPy: {e}")
                return self._normalize_vectors_numpy(vectors, norm_type)
            else:
                raise
    
    def compute_similarity_matrix(
        self,
        vectors_a: np.ndarray,
        vectors_b: Optional[np.ndarray] = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        backend: Optional[VectorBackend] = None
    ) -> np.ndarray:
        """
        Compute similarity matrix between two sets of vectors.
        
        Args:
            vectors_a: First set of vectors [num_vectors_a, vector_dim]
            vectors_b: Second set of vectors [num_vectors_b, vector_dim] (optional)
            metric: Similarity metric to use
            backend: Specific backend to use
            
        Returns:
            Similarity matrix [num_vectors_a, num_vectors_b]
        """
        if vectors_a.ndim == 1:
            vectors_a = vectors_a.reshape(1, -1)
        
        if vectors_b is None:
            vectors_b = vectors_a
        elif vectors_b.ndim == 1:
            vectors_b = vectors_b.reshape(1, -1)
        
        backend = backend or self._select_optimal_backend(
            "similarity_matrix",
            (vectors_a.shape[0] + vectors_b.shape[0], vectors_a.shape[1])
        )
        
        cache_key = self._get_cache_key(
            "similarity_matrix", 
            vectors_a.tobytes(), 
            vectors_b.tobytes(), 
            metric.value, 
            backend.value
        )
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with self._performance_tracking(
            "similarity_matrix",
            backend,
            num_vectors=vectors_a.shape[0] * vectors_b.shape[0],
            vector_dimension=vectors_a.shape[1]
        ):
            if backend == VectorBackend.RUST and self._rust_available:
                result = self._similarity_matrix_rust(vectors_a, vectors_b, metric)
            elif backend == VectorBackend.GPU and self._gpu_available:
                result = self._similarity_matrix_gpu(vectors_a, vectors_b, metric)
            elif backend == VectorBackend.SCIPY:
                result = self._similarity_matrix_scipy(vectors_a, vectors_b, metric)
            else:  # NumPy fallback
                result = self._similarity_matrix_numpy(vectors_a, vectors_b, metric)
        
        self._cache_result(cache_key, result)
        return result
    
    def _similarity_matrix_numpy(
        self, 
        vectors_a: np.ndarray, 
        vectors_b: np.ndarray, 
        metric: SimilarityMetric
    ) -> np.ndarray:
        """NumPy implementation of similarity matrix computation"""
        if metric == SimilarityMetric.COSINE:
            # Normalize vectors
            norm_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)
            
            norm_a = np.where(norm_a == 0, 1, norm_a)
            norm_b = np.where(norm_b == 0, 1, norm_b)
            
            normalized_a = vectors_a / norm_a
            normalized_b = vectors_b / norm_b
            
            return np.dot(normalized_a, normalized_b.T)
            
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return np.dot(vectors_a, vectors_b.T)
            
        elif metric == SimilarityMetric.EUCLIDEAN:
            # Compute pairwise Euclidean distances
            distances = np.sqrt(
                np.sum((vectors_a[:, np.newaxis, :] - vectors_b[np.newaxis, :, :]) ** 2, axis=2)
            )
            # Convert to similarity (higher is more similar)
            return 1.0 / (1.0 + distances)
            
        elif metric == SimilarityMetric.MANHATTAN:
            # Compute pairwise Manhattan distances
            distances = np.sum(
                np.abs(vectors_a[:, np.newaxis, :] - vectors_b[np.newaxis, :, :]), axis=2
            )
            return 1.0 / (1.0 + distances)
            
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def _similarity_matrix_scipy(
        self, 
        vectors_a: np.ndarray, 
        vectors_b: np.ndarray, 
        metric: SimilarityMetric
    ) -> np.ndarray:
        """SciPy implementation of similarity matrix computation"""
        if metric == SimilarityMetric.COSINE:
            # Use SciPy's optimized cosine distance
            distances = distance.cdist(vectors_a, vectors_b, 'cosine')
            return 1.0 - distances  # Convert distance to similarity
            
        elif metric == SimilarityMetric.EUCLIDEAN:
            distances = distance.cdist(vectors_a, vectors_b, 'euclidean')
            return 1.0 / (1.0 + distances)
            
        elif metric == SimilarityMetric.MANHATTAN:
            distances = distance.cdist(vectors_a, vectors_b, 'cityblock')
            return 1.0 / (1.0 + distances)
            
        else:
            # Fall back to NumPy for unsupported metrics
            return self._similarity_matrix_numpy(vectors_a, vectors_b, metric)
    
    def _similarity_matrix_rust(
        self, 
        vectors_a: np.ndarray, 
        vectors_b: np.ndarray, 
        metric: SimilarityMetric
    ) -> np.ndarray:
        """Rust implementation of similarity matrix computation"""
        if not self._rust_module:
            raise RustBindingError("Rust module not available")
        
        try:
            # Convert to appropriate format for Rust
            vectors_a_f32 = vectors_a.astype(np.float32)
            vectors_b_f32 = vectors_b.astype(np.float32)
            
            # Call Rust function
            if hasattr(self._rust_module, 'similarity_matrix'):
                result = self._rust_module.similarity_matrix(
                    vectors_a_f32, 
                    vectors_b_f32, 
                    metric.value
                )
                return result.astype(vectors_a.dtype)
            else:
                raise RustBindingError("similarity_matrix function not found in Rust module")
                
        except Exception as e:
            if self.config.fallback_on_error:
                self.logger.warning(f"Rust similarity matrix failed, falling back to SciPy: {e}")
                return self._similarity_matrix_scipy(vectors_a, vectors_b, metric)
            else:
                raise RustBindingError(f"Rust similarity matrix failed: {e}")
    
    def _similarity_matrix_gpu(
        self, 
        vectors_a: np.ndarray, 
        vectors_b: np.ndarray, 
        metric: SimilarityMetric
    ) -> np.ndarray:
        """GPU implementation of similarity matrix computation"""
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_vectors_a = cp.asarray(vectors_a)
            gpu_vectors_b = cp.asarray(vectors_b)
            
            if metric == SimilarityMetric.COSINE:
                # Normalize vectors
                norm_a = cp.linalg.norm(gpu_vectors_a, axis=1, keepdims=True)
                norm_b = cp.linalg.norm(gpu_vectors_b, axis=1, keepdims=True)
                
                norm_a = cp.where(norm_a == 0, 1, norm_a)
                norm_b = cp.where(norm_b == 0, 1, norm_b)
                
                normalized_a = gpu_vectors_a / norm_a
                normalized_b = gpu_vectors_b / norm_b
                
                result = cp.dot(normalized_a, normalized_b.T)
                
            elif metric == SimilarityMetric.DOT_PRODUCT:
                result = cp.dot(gpu_vectors_a, gpu_vectors_b.T)
                
            elif metric == SimilarityMetric.EUCLIDEAN:
                # Compute pairwise Euclidean distances
                diff = gpu_vectors_a[:, cp.newaxis, :] - gpu_vectors_b[cp.newaxis, :, :]
                distances = cp.sqrt(cp.sum(diff ** 2, axis=2))
                result = 1.0 / (1.0 + distances)
                
            else:
                raise ValueError(f"GPU backend doesn't support metric: {metric}")
            
            # Transfer back to CPU
            return cp.asnumpy(result)
            
        except ImportError:
            if self.config.fallback_on_error:
                self.logger.warning("CuPy not available, falling back to SciPy")
                return self._similarity_matrix_scipy(vectors_a, vectors_b, metric)
            else:
                raise
        except Exception as e:
            if self.config.fallback_on_error:
                self.logger.warning(f"GPU similarity matrix failed, falling back to SciPy: {e}")
                return self._similarity_matrix_scipy(vectors_a, vectors_b, metric)
            else:
                raise
    
    def batch_cosine_similarity(
        self,
        query_vectors: np.ndarray,
        database_vectors: np.ndarray,
        top_k: Optional[int] = None,
        backend: Optional[VectorBackend] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between query vectors and database vectors.
        
        Args:
            query_vectors: Query vectors [num_queries, vector_dim]
            database_vectors: Database vectors [num_database, vector_dim]
            top_k: Return only top-k most similar vectors per query
            backend: Specific backend to use
            
        Returns:
            Tuple of (similarities, indices) arrays
        """
        backend = backend or self._select_optimal_backend(
            "batch_cosine_similarity",
            (query_vectors.shape[0] + database_vectors.shape[0], query_vectors.shape[1])
        )
        
        with self._performance_tracking(
            "batch_cosine_similarity",
            backend,
            num_vectors=query_vectors.shape[0] * database_vectors.shape[0],
            vector_dimension=query_vectors.shape[1]
        ):
            # Compute similarity matrix
            similarity_matrix = self.compute_similarity_matrix(
                query_vectors, 
                database_vectors, 
                SimilarityMetric.COSINE,
                backend
            )
            
            if top_k is None:
                # Return all similarities and indices
                indices = np.arange(database_vectors.shape[0])[np.newaxis, :].repeat(
                    query_vectors.shape[0], axis=0
                )
                return similarity_matrix, indices
            else:
                # Return top-k similarities and indices
                top_k = min(top_k, database_vectors.shape[0])
                
                # Get top-k indices for each query
                top_indices = np.argpartition(similarity_matrix, -top_k, axis=1)[:, -top_k:]
                
                # Sort the top-k indices by similarity (descending)
                sorted_indices = np.argsort(
                    np.take_along_axis(similarity_matrix, top_indices, axis=1),
                    axis=1
                )[:, ::-1]
                
                final_indices = np.take_along_axis(top_indices, sorted_indices, axis=1)
                final_similarities = np.take_along_axis(similarity_matrix, final_indices, axis=1)
                
                return final_similarities, final_indices
    
    def vector_arithmetic(
        self,
        operation: str,
        vector_a: np.ndarray,
        vector_b: Optional[np.ndarray] = None,
        scalar: Optional[float] = None,
        backend: Optional[VectorBackend] = None
    ) -> np.ndarray:
        """
        Perform vector arithmetic operations.
        
        Args:
            operation: Operation type ('add', 'subtract', 'multiply', 'divide', 'scale')
            vector_a: First vector or vector array
            vector_b: Second vector (for binary operations)
            scalar: Scalar value (for scaling operations)
            backend: Specific backend to use
            
        Returns:
            Result vector(s)
        """
        backend = backend or self._select_optimal_backend(
            f"vector_{operation}",
            vector_a.shape
        )
        
        with self._performance_tracking(
            f"vector_{operation}",
            backend,
            num_vectors=vector_a.shape[0] if vector_a.ndim > 1 else 1,
            vector_dimension=vector_a.shape[-1]
        ):
            if backend == VectorBackend.RUST and self._rust_available:
                return self._vector_arithmetic_rust(operation, vector_a, vector_b, scalar)
            elif backend == VectorBackend.GPU and self._gpu_available:
                return self._vector_arithmetic_gpu(operation, vector_a, vector_b, scalar)
            else:  # NumPy (default for arithmetic)
                return self._vector_arithmetic_numpy(operation, vector_a, vector_b, scalar)
    
    def _vector_arithmetic_numpy(
        self,
        operation: str,
        vector_a: np.ndarray,
        vector_b: Optional[np.ndarray],
        scalar: Optional[float]
    ) -> np.ndarray:
        """NumPy implementation of vector arithmetic"""
        if operation == "add":
            if vector_b is not None:
                return vector_a + vector_b
            elif scalar is not None:
                return vector_a + scalar
            else:
                raise ValueError("Add operation requires vector_b or scalar")
                
        elif operation == "subtract":
            if vector_b is not None:
                return vector_a - vector_b
            elif scalar is not None:
                return vector_a - scalar
            else:
                raise ValueError("Subtract operation requires vector_b or scalar")
                
        elif operation == "multiply":
            if vector_b is not None:
                return vector_a * vector_b
            elif scalar is not None:
                return vector_a * scalar
            else:
                raise ValueError("Multiply operation requires vector_b or scalar")
                
        elif operation == "divide":
            if vector_b is not None:
                return vector_a / np.where(vector_b == 0, 1e-8, vector_b)
            elif scalar is not None:
                return vector_a / max(scalar, 1e-8)
            else:
                raise ValueError("Divide operation requires vector_b or scalar")
                
        elif operation == "scale":
            if scalar is not None:
                return vector_a * scalar
            else:
                raise ValueError("Scale operation requires scalar")
                
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vector_arithmetic_rust(
        self,
        operation: str,
        vector_a: np.ndarray,
        vector_b: Optional[np.ndarray],
        scalar: Optional[float]
    ) -> np.ndarray:
        """Rust implementation of vector arithmetic"""
        if not self._rust_module:
            raise RustBindingError("Rust module not available")
        
        try:
            # Convert to appropriate format for Rust
            vector_a_f32 = vector_a.astype(np.float32)
            vector_b_f32 = vector_b.astype(np.float32) if vector_b is not None else None
            
            # Call Rust function
            if hasattr(self._rust_module, 'vector_arithmetic'):
                result = self._rust_module.vector_arithmetic(
                    operation,
                    vector_a_f32,
                    vector_b_f32,
                    scalar
                )
                return result.astype(vector_a.dtype)
            else:
                raise RustBindingError("vector_arithmetic function not found in Rust module")
                
        except Exception as e:
            if self.config.fallback_on_error:
                self.logger.warning(f"Rust vector arithmetic failed, falling back to NumPy: {e}")
                return self._vector_arithmetic_numpy(operation, vector_a, vector_b, scalar)
            else:
                raise RustBindingError(f"Rust vector arithmetic failed: {e}")
    
    def _vector_arithmetic_gpu(
        self,
        operation: str,
        vector_a: np.ndarray,
        vector_b: Optional[np.ndarray],
        scalar: Optional[float]
    ) -> np.ndarray:
        """GPU implementation of vector arithmetic"""
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_vector_a = cp.asarray(vector_a)
            gpu_vector_b = cp.asarray(vector_b) if vector_b is not None else None
            
            if operation == "add":
                if gpu_vector_b is not None:
                    result = gpu_vector_a + gpu_vector_b
                elif scalar is not None:
                    result = gpu_vector_a + scalar
                else:
                    raise ValueError("Add operation requires vector_b or scalar")
                    
            elif operation == "subtract":
                if gpu_vector_b is not None:
                    result = gpu_vector_a - gpu_vector_b
                elif scalar is not None:
                    result = gpu_vector_a - scalar
                else:
                    raise ValueError("Subtract operation requires vector_b or scalar")
                    
            elif operation == "multiply":
                if gpu_vector_b is not None:
                    result = gpu_vector_a * gpu_vector_b
                elif scalar is not None:
                    result = gpu_vector_a * scalar
                else:
                    raise ValueError("Multiply operation requires vector_b or scalar")
                    
            elif operation == "divide":
                if gpu_vector_b is not None:
                    result = gpu_vector_a / cp.where(gpu_vector_b == 0, 1e-8, gpu_vector_b)
                elif scalar is not None:
                    result = gpu_vector_a / max(scalar, 1e-8)
                else:
                    raise ValueError("Divide operation requires vector_b or scalar")
                    
            elif operation == "scale":
                if scalar is not None:
                    result = gpu_vector_a * scalar
                else:
                    raise ValueError("Scale operation requires scalar")
                    
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Transfer back to CPU
            return cp.asnumpy(result)
            
        except ImportError:
            if self.config.fallback_on_error:
                self.logger.warning("CuPy not available, falling back to NumPy")
                return self._vector_arithmetic_numpy(operation, vector_a, vector_b, scalar)
            else:
                raise
        except Exception as e:
            if self.config.fallback_on_error:
                self.logger.warning(f"GPU vector arithmetic failed, falling back to NumPy: {e}")
                return self._vector_arithmetic_numpy(operation, vector_a, vector_b, scalar)
            else:
                raise
    
    # Performance and Utility Methods
    
    def benchmark_backends(
        self,
        test_vectors: np.ndarray,
        operations: Optional[List[str]] = None
    ) -> Dict[str, Dict[VectorBackend, PerformanceMetrics]]:
        """
        Benchmark different backends for various operations.
        
        Args:
            test_vectors: Test vectors for benchmarking
            operations: List of operations to benchmark
            
        Returns:
            Benchmark results by operation and backend
        """
        if not self.config.enable_benchmarking:
            return {}
        
        operations = operations or [
            "normalize_vectors",
            "similarity_matrix", 
            "batch_cosine_similarity",
            "vector_add"
        ]
        
        results = {}
        
        for operation in operations:
            results[operation] = {}
            
            # Test available backends
            backends_to_test = [VectorBackend.NUMPY, VectorBackend.SCIPY]
            
            if self._rust_available:
                backends_to_test.append(VectorBackend.RUST)
            
            if self._gpu_available:
                backends_to_test.append(VectorBackend.GPU)
            
            for backend in backends_to_test:
                try:
                    # Clear metrics before test
                    with self._metrics_lock:
                        self._performance_metrics.clear()
                    
                    # Run operation
                    if operation == "normalize_vectors":
                        self.normalize_vectors(test_vectors, backend=backend)
                    elif operation == "similarity_matrix":
                        self.compute_similarity_matrix(
                            test_vectors[:100], 
                            test_vectors[:100], 
                            backend=backend
                        )
                    elif operation == "batch_cosine_similarity":
                        self.batch_cosine_similarity(
                            test_vectors[:50], 
                            test_vectors, 
                            top_k=10,
                            backend=backend
                        )
                    elif operation == "vector_add":
                        self.vector_arithmetic(
                            "add", 
                            test_vectors, 
                            test_vectors, 
                            backend=backend
                        )
                    
                    # Get metrics
                    with self._metrics_lock:
                        if self._performance_metrics:
                            results[operation][backend] = self._performance_metrics[-1]
                
                except Exception as e:
                    self.logger.warning(f"Benchmark failed for {operation} with {backend}: {e}")
        
        return results
    
    def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get all collected performance metrics"""
        with self._metrics_lock:
            return self._performance_metrics.copy()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        with self._cache_lock:
            return {
                "cache_entries": len(self._computation_cache),
                "cache_size_bytes": self._cache_size_bytes,
                "cache_size_mb": self._cache_size_bytes / (1024 * 1024),
                "cache_limit_mb": self.config.cache_size_mb,
                "cache_utilization": self._cache_size_bytes / (self.config.cache_size_mb * 1024 * 1024)
            }
    
    def clear_cache(self):
        """Clear computation cache"""
        with self._cache_lock:
            self._computation_cache.clear()
            self._cache_size_bytes = 0
    
    def get_backend_availability(self) -> Dict[VectorBackend, bool]:
        """Get availability status of all backends"""
        return {
            VectorBackend.NUMPY: True,  # Always available
            VectorBackend.SCIPY: True,  # Always available
            VectorBackend.RUST: self._rust_available,
            VectorBackend.GPU: self._gpu_available
        }
    
    def shutdown(self):
        """Shutdown the optimizer and clean up resources"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        self.clear_cache()
        
        self.logger.info("Vector operations optimizer shutdown complete")


# Global service instance
_vector_optimizer: Optional[VectorOperationsOptimizer] = None


def get_vector_optimizer(config: Optional[VectorOperationConfig] = None) -> VectorOperationsOptimizer:
    """Get or create global vector operations optimizer instance"""
    global _vector_optimizer
    
    if _vector_optimizer is None:
        _vector_optimizer = VectorOperationsOptimizer(config)
    
    return _vector_optimizer


def create_optimized_config() -> VectorOperationConfig:
    """Create optimized configuration based on system capabilities"""
    config = VectorOperationConfig()
    
    # Auto-configure based on system
    import psutil
    
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Adjust settings based on system resources
    if memory_gb >= 16:
        config.memory_limit_mb = 4096
        config.cache_size_mb = 1024
        config.batch_size = 2000
    elif memory_gb >= 8:
        config.memory_limit_mb = 2048
        config.cache_size_mb = 512
        config.batch_size = 1000
    else:
        config.memory_limit_mb = 1024
        config.cache_size_mb = 256
        config.batch_size = 500
    
    config.num_threads = min(cpu_count, 8)
    
    return config