"""
Enhanced Vector Service with Async Performance Integration

This module integrates the vector operations optimizer with the existing
async performance infrastructure, providing optimized vector processing
for the knowledge ingestion system.

Features:
- Integration with async performance optimizer
- Automatic backend selection (NumPy/SciPy/Rust/GPU)
- Batch processing with intelligent scheduling
- Performance monitoring and metrics
- Graceful fallback mechanisms
- Memory-efficient operations

Requirements: 10.3, 10.5
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from core.config import get_settings
from core.logging import get_logger
from .vector_operations_optimizer import (
    VectorOperationsOptimizer, 
    VectorOperationConfig, 
    VectorBackend, 
    SimilarityMetric,
    get_vector_optimizer,
    create_optimized_config
)
from .rust_ffi_interface import (
    VectorOperationsFFI,
    get_vector_ffi,
    is_rust_available,
    RustFFIError
)
from .async_performance_optimizer import (
    AsyncPerformanceOptimizer,
    ProcessingTask,
    ProcessingMode,
    get_performance_optimizer
)
from config.async_performance_config import get_async_performance_config


@dataclass
class VectorProcessingJob:
    """Vector processing job for async execution"""
    job_id: str
    operation: str
    vectors: np.ndarray
    additional_data: Optional[Dict[str, Any]] = None
    priority: int = 0
    backend_preference: Optional[VectorBackend] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class VectorServiceMetrics:
    """Metrics for the enhanced vector service"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    rust_operations: int = 0
    gpu_operations: int = 0
    numpy_operations: int = 0
    scipy_operations: int = 0
    average_processing_time_ms: float = 0.0
    total_vectors_processed: int = 0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EnhancedVectorService:
    """
    Enhanced vector service that integrates vector operations optimization
    with async performance infrastructure.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Configuration
        self.async_config = get_async_performance_config()
        self.vector_config = self._create_vector_config()
        
        # Core components
        self.vector_optimizer: Optional[VectorOperationsOptimizer] = None
        self.vector_ffi: Optional[VectorOperationsFFI] = None
        self.async_optimizer: Optional[AsyncPerformanceOptimizer] = None
        
        # Job management
        self._active_jobs: Dict[str, VectorProcessingJob] = {}
        self._job_counter = 0
        self._job_lock = threading.Lock()
        
        # Metrics
        self._metrics = VectorServiceMetrics()
        self._metrics_lock = threading.Lock()
        
        # Thread pool for CPU-bound vector operations
        self._vector_thread_pool: Optional[ThreadPoolExecutor] = None
        
        # Initialization flag
        self._initialized = False
    
    def _create_vector_config(self) -> VectorOperationConfig:
        """Create vector operation configuration from async config"""
        config = create_optimized_config()
        
        # Apply settings from async config
        if hasattr(self.async_config.features, 'enable_rust_bindings'):
            config.enable_rust_bindings = self.async_config.features.enable_rust_bindings
        
        if hasattr(self.async_config.features, 'enable_gpu_acceleration'):
            config.enable_gpu_acceleration = self.async_config.features.enable_gpu_acceleration
        
        if hasattr(self.async_config.features, 'vector_batch_size'):
            config.batch_size = self.async_config.features.vector_batch_size
        
        if hasattr(self.async_config.features, 'vector_cache_size_mb'):
            config.cache_size_mb = self.async_config.features.vector_cache_size_mb
        
        # Set thread count from async config
        config.num_threads = min(
            self.async_config.worker_pools.cpu_pool_size,
            config.num_threads
        )
        
        # Memory limits
        config.memory_limit_mb = min(
            self.async_config.resource_limits.max_memory_usage_mb // 4,  # Use 1/4 of total for vectors
            config.memory_limit_mb
        )
        
        return config
    
    async def initialize(self) -> bool:
        """Initialize the enhanced vector service"""
        try:
            self.logger.info("Initializing enhanced vector service")
            
            # Initialize vector optimizer
            self.vector_optimizer = get_vector_optimizer(self.vector_config)
            
            # Initialize Rust FFI if available
            if self.vector_config.enable_rust_bindings:
                self.vector_ffi = get_vector_ffi()
                if self.vector_ffi.is_available():
                    self.logger.info("Rust FFI initialized successfully")
                else:
                    self.logger.info("Rust FFI not available, using Python implementations")
            
            # Initialize async performance optimizer
            self.async_optimizer = await get_performance_optimizer()
            
            # Initialize thread pool for vector operations
            self._vector_thread_pool = ThreadPoolExecutor(
                max_workers=self.vector_config.num_threads,
                thread_name_prefix="vector_service"
            )
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            self._initialized = True
            self.logger.info("Enhanced vector service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced vector service: {e}")
            return False
    
    async def normalize_vectors_async(
        self,
        vectors: np.ndarray,
        norm_type: str = "l2",
        backend: Optional[VectorBackend] = None,
        priority: int = 0
    ) -> np.ndarray:
        """
        Asynchronously normalize vectors with optimal backend selection.
        
        Args:
            vectors: Input vectors to normalize
            norm_type: Type of normalization ("l1", "l2", "max")
            backend: Specific backend to use (optional)
            priority: Job priority (higher = more priority)
            
        Returns:
            Normalized vectors
        """
        if not self._initialized:
            raise RuntimeError("Enhanced vector service not initialized")
        
        # Create processing job
        job = VectorProcessingJob(
            job_id=f"normalize_{self._get_next_job_id()}",
            operation="normalize_vectors",
            vectors=vectors,
            additional_data={"norm_type": norm_type},
            priority=priority,
            backend_preference=backend
        )
        
        return await self._execute_vector_job(job)
    
    async def compute_similarity_matrix_async(
        self,
        vectors_a: np.ndarray,
        vectors_b: Optional[np.ndarray] = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        backend: Optional[VectorBackend] = None,
        priority: int = 0
    ) -> np.ndarray:
        """
        Asynchronously compute similarity matrix with optimal backend selection.
        
        Args:
            vectors_a: First set of vectors
            vectors_b: Second set of vectors (optional)
            metric: Similarity metric to use
            backend: Specific backend to use (optional)
            priority: Job priority
            
        Returns:
            Similarity matrix
        """
        if not self._initialized:
            raise RuntimeError("Enhanced vector service not initialized")
        
        job = VectorProcessingJob(
            job_id=f"similarity_{self._get_next_job_id()}",
            operation="similarity_matrix",
            vectors=vectors_a,
            additional_data={
                "vectors_b": vectors_b,
                "metric": metric
            },
            priority=priority,
            backend_preference=backend
        )
        
        return await self._execute_vector_job(job)
    
    async def batch_cosine_similarity_async(
        self,
        query_vectors: np.ndarray,
        database_vectors: np.ndarray,
        top_k: Optional[int] = None,
        backend: Optional[VectorBackend] = None,
        priority: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asynchronously compute batch cosine similarity with optimal backend selection.
        
        Args:
            query_vectors: Query vectors
            database_vectors: Database vectors
            top_k: Number of top results to return
            backend: Specific backend to use (optional)
            priority: Job priority
            
        Returns:
            Tuple of (similarities, indices)
        """
        if not self._initialized:
            raise RuntimeError("Enhanced vector service not initialized")
        
        job = VectorProcessingJob(
            job_id=f"batch_cosine_{self._get_next_job_id()}",
            operation="batch_cosine_similarity",
            vectors=query_vectors,
            additional_data={
                "database_vectors": database_vectors,
                "top_k": top_k
            },
            priority=priority,
            backend_preference=backend
        )
        
        return await self._execute_vector_job(job)
    
    async def vector_arithmetic_async(
        self,
        operation: str,
        vector_a: np.ndarray,
        vector_b: Optional[np.ndarray] = None,
        scalar: Optional[float] = None,
        backend: Optional[VectorBackend] = None,
        priority: int = 0
    ) -> np.ndarray:
        """
        Asynchronously perform vector arithmetic with optimal backend selection.
        
        Args:
            operation: Arithmetic operation ("add", "subtract", "multiply", "divide", "scale")
            vector_a: First vector
            vector_b: Second vector (optional)
            scalar: Scalar value (optional)
            backend: Specific backend to use (optional)
            priority: Job priority
            
        Returns:
            Result vectors
        """
        if not self._initialized:
            raise RuntimeError("Enhanced vector service not initialized")
        
        job = VectorProcessingJob(
            job_id=f"arithmetic_{self._get_next_job_id()}",
            operation="vector_arithmetic",
            vectors=vector_a,
            additional_data={
                "arithmetic_operation": operation,
                "vector_b": vector_b,
                "scalar": scalar
            },
            priority=priority,
            backend_preference=backend
        )
        
        return await self._execute_vector_job(job)
    
    async def _execute_vector_job(self, job: VectorProcessingJob) -> Any:
        """Execute a vector processing job asynchronously"""
        try:
            # Register job
            with self._job_lock:
                self._active_jobs[job.job_id] = job
            
            job.started_at = datetime.now(timezone.utc)
            
            # Submit to async performance optimizer
            if self.async_optimizer:
                # Create processing task for async optimizer
                task_data = self._create_processing_task_data(job)
                
                # Submit to appropriate worker pool
                success = await self.async_optimizer.submit_task(
                    pool_name="cpu_processing",  # Vector operations are CPU-bound
                    task_data=task_data,
                    processing_mode=ProcessingMode.CPU_BOUND,
                    callback=None,
                    priority=job.priority
                )
                
                if not success:
                    raise RuntimeError("Failed to submit job to async optimizer")
                
                # Wait for completion (in a real implementation, this would be event-driven)
                result = await self._wait_for_job_completion(job)
            else:
                # Execute directly in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._vector_thread_pool,
                    self._execute_vector_operation,
                    job
                )
            
            job.completed_at = datetime.now(timezone.utc)
            job.result = result
            
            # Update metrics
            await self._update_metrics(job, success=True)
            
            return result
            
        except Exception as e:
            job.completed_at = datetime.now(timezone.utc)
            job.error = str(e)
            
            # Update metrics
            await self._update_metrics(job, success=False)
            
            self.logger.error(f"Vector job {job.job_id} failed: {e}")
            raise
        finally:
            # Clean up job
            with self._job_lock:
                self._active_jobs.pop(job.job_id, None)
    
    def _create_processing_task_data(self, job: VectorProcessingJob) -> Any:
        """Create task data for async performance optimizer"""
        # This would be a custom data structure that the async optimizer can handle
        # For now, we'll use a simple wrapper
        class VectorTaskData:
            def __init__(self, job: VectorProcessingJob):
                self.job = job
                self.operation_type = 'vector_operation'
                self.use_multiprocessing = False  # Vector ops are better with threads
        
        return VectorTaskData(job)
    
    async def _wait_for_job_completion(self, job: VectorProcessingJob) -> Any:
        """Wait for job completion (simplified implementation)"""
        # In a real implementation, this would use proper async coordination
        # For now, we'll execute directly
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._vector_thread_pool,
            self._execute_vector_operation,
            job
        )
    
    def _execute_vector_operation(self, job: VectorProcessingJob) -> Any:
        """Execute vector operation synchronously"""
        try:
            if job.operation == "normalize_vectors":
                return self._execute_normalize_vectors(job)
            elif job.operation == "similarity_matrix":
                return self._execute_similarity_matrix(job)
            elif job.operation == "batch_cosine_similarity":
                return self._execute_batch_cosine_similarity(job)
            elif job.operation == "vector_arithmetic":
                return self._execute_vector_arithmetic(job)
            else:
                raise ValueError(f"Unknown operation: {job.operation}")
                
        except Exception as e:
            self.logger.error(f"Error executing vector operation {job.operation}: {e}")
            raise
    
    def _execute_normalize_vectors(self, job: VectorProcessingJob) -> np.ndarray:
        """Execute vector normalization"""
        vectors = job.vectors
        norm_type = job.additional_data.get("norm_type", "l2")
        backend = job.backend_preference
        
        # Try Rust FFI first if available and preferred
        if (self.vector_ffi and self.vector_ffi.is_available() and 
            self.vector_config.enable_rust_bindings and
            (backend is None or backend == VectorBackend.RUST)):
            try:
                result = self.vector_ffi.normalize_vectors(vectors, norm_type)
                self._record_backend_usage(VectorBackend.RUST)
                return result
            except RustFFIError as e:
                if not self.vector_config.fallback_on_error:
                    raise
                self.logger.warning(f"Rust normalization failed, falling back: {e}")
        
        # Use vector optimizer
        result = self.vector_optimizer.normalize_vectors(vectors, norm_type, backend)
        self._record_backend_usage(backend or VectorBackend.AUTO)
        return result
    
    def _execute_similarity_matrix(self, job: VectorProcessingJob) -> np.ndarray:
        """Execute similarity matrix computation"""
        vectors_a = job.vectors
        vectors_b = job.additional_data.get("vectors_b")
        metric = job.additional_data.get("metric", SimilarityMetric.COSINE)
        backend = job.backend_preference
        
        # Try Rust FFI first if available
        if (self.vector_ffi and self.vector_ffi.is_available() and 
            self.vector_config.enable_rust_bindings and
            (backend is None or backend == VectorBackend.RUST) and
            metric == SimilarityMetric.COSINE):  # Rust FFI currently supports cosine
            try:
                if vectors_b is None:
                    vectors_b = vectors_a
                result = self.vector_ffi.similarity_matrix(vectors_a, vectors_b, metric.value)
                self._record_backend_usage(VectorBackend.RUST)
                return result
            except RustFFIError as e:
                if not self.vector_config.fallback_on_error:
                    raise
                self.logger.warning(f"Rust similarity matrix failed, falling back: {e}")
        
        # Use vector optimizer
        result = self.vector_optimizer.compute_similarity_matrix(vectors_a, vectors_b, metric, backend)
        self._record_backend_usage(backend or VectorBackend.AUTO)
        return result
    
    def _execute_batch_cosine_similarity(self, job: VectorProcessingJob) -> Tuple[np.ndarray, np.ndarray]:
        """Execute batch cosine similarity"""
        query_vectors = job.vectors
        database_vectors = job.additional_data.get("database_vectors")
        top_k = job.additional_data.get("top_k")
        backend = job.backend_preference
        
        # Try Rust FFI first if available
        if (self.vector_ffi and self.vector_ffi.is_available() and 
            self.vector_config.enable_rust_bindings and
            (backend is None or backend == VectorBackend.RUST) and
            top_k is not None):
            try:
                result = self.vector_ffi.batch_cosine_similarity(query_vectors, database_vectors, top_k)
                self._record_backend_usage(VectorBackend.RUST)
                return result
            except RustFFIError as e:
                if not self.vector_config.fallback_on_error:
                    raise
                self.logger.warning(f"Rust batch cosine similarity failed, falling back: {e}")
        
        # Use vector optimizer
        result = self.vector_optimizer.batch_cosine_similarity(query_vectors, database_vectors, top_k, backend)
        self._record_backend_usage(backend or VectorBackend.AUTO)
        return result
    
    def _execute_vector_arithmetic(self, job: VectorProcessingJob) -> np.ndarray:
        """Execute vector arithmetic"""
        vector_a = job.vectors
        operation = job.additional_data.get("arithmetic_operation")
        vector_b = job.additional_data.get("vector_b")
        scalar = job.additional_data.get("scalar")
        backend = job.backend_preference
        
        # Try Rust FFI first if available
        if (self.vector_ffi and self.vector_ffi.is_available() and 
            self.vector_config.enable_rust_bindings and
            (backend is None or backend == VectorBackend.RUST)):
            try:
                result = self.vector_ffi.vector_arithmetic(operation, vector_a, vector_b, scalar)
                self._record_backend_usage(VectorBackend.RUST)
                return result
            except RustFFIError as e:
                if not self.vector_config.fallback_on_error:
                    raise
                self.logger.warning(f"Rust vector arithmetic failed, falling back: {e}")
        
        # Use vector optimizer
        result = self.vector_optimizer.vector_arithmetic(operation, vector_a, vector_b, scalar, backend)
        self._record_backend_usage(backend or VectorBackend.AUTO)
        return result
    
    def _record_backend_usage(self, backend: VectorBackend):
        """Record which backend was used for metrics"""
        with self._metrics_lock:
            if backend == VectorBackend.RUST:
                self._metrics.rust_operations += 1
            elif backend == VectorBackend.GPU:
                self._metrics.gpu_operations += 1
            elif backend == VectorBackend.NUMPY:
                self._metrics.numpy_operations += 1
            elif backend == VectorBackend.SCIPY:
                self._metrics.scipy_operations += 1
    
    async def _update_metrics(self, job: VectorProcessingJob, success: bool):
        """Update service metrics"""
        with self._metrics_lock:
            self._metrics.total_operations += 1
            
            if success:
                self._metrics.successful_operations += 1
            else:
                self._metrics.failed_operations += 1
            
            # Update processing time
            if job.started_at and job.completed_at:
                processing_time_ms = (job.completed_at - job.started_at).total_seconds() * 1000
                
                # Running average
                if self._metrics.total_operations == 1:
                    self._metrics.average_processing_time_ms = processing_time_ms
                else:
                    alpha = 0.1  # Smoothing factor
                    self._metrics.average_processing_time_ms = (
                        alpha * processing_time_ms + 
                        (1 - alpha) * self._metrics.average_processing_time_ms
                    )
            
            # Update vector count
            if hasattr(job.vectors, 'shape'):
                if job.vectors.ndim == 1:
                    self._metrics.total_vectors_processed += 1
                else:
                    self._metrics.total_vectors_processed += job.vectors.shape[0]
            
            self._metrics.last_updated = datetime.now(timezone.utc)
    
    async def _metrics_collector(self):
        """Collect and update metrics periodically"""
        while True:
            try:
                # Update cache hit rate from vector optimizer
                if self.vector_optimizer:
                    cache_stats = self.vector_optimizer.get_cache_statistics()
                    with self._metrics_lock:
                        # Simple cache hit rate calculation
                        if self._metrics.total_operations > 0:
                            self._metrics.cache_hit_rate = min(
                                cache_stats.get("cache_entries", 0) / self._metrics.total_operations,
                                1.0
                            )
                
                # Update memory usage
                import psutil
                process = psutil.Process()
                with self._metrics_lock:
                    self._metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting vector service metrics: {e}")
                await asyncio.sleep(10)
    
    def _get_next_job_id(self) -> int:
        """Get next job ID"""
        with self._job_lock:
            self._job_counter += 1
            return self._job_counter
    
    async def get_metrics(self) -> VectorServiceMetrics:
        """Get current service metrics"""
        with self._metrics_lock:
            return self._metrics
    
    async def get_active_jobs(self) -> List[VectorProcessingJob]:
        """Get list of active jobs"""
        with self._job_lock:
            return list(self._active_jobs.values())
    
    async def benchmark_backends(self, test_vectors: np.ndarray) -> Dict[str, Any]:
        """Benchmark different backends with test vectors"""
        if not self.vector_optimizer:
            return {}
        
        # Run benchmarks in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._vector_thread_pool,
            self.vector_optimizer.benchmark_backends,
            test_vectors
        )
    
    async def get_backend_availability(self) -> Dict[VectorBackend, bool]:
        """Get availability status of all backends"""
        if not self.vector_optimizer:
            return {}
        
        availability = self.vector_optimizer.get_backend_availability()
        
        # Add FFI availability
        if self.vector_ffi:
            availability[VectorBackend.RUST] = self.vector_ffi.is_available()
        
        return availability
    
    async def shutdown(self):
        """Shutdown the enhanced vector service"""
        self.logger.info("Shutting down enhanced vector service")
        
        # Wait for active jobs to complete
        max_wait_time = 30  # seconds
        wait_start = time.time()
        
        while self._active_jobs and (time.time() - wait_start) < max_wait_time:
            await asyncio.sleep(1)
        
        # Shutdown thread pool
        if self._vector_thread_pool:
            self._vector_thread_pool.shutdown(wait=True)
        
        # Cleanup vector optimizer
        if self.vector_optimizer:
            self.vector_optimizer.shutdown()
        
        # Cleanup FFI
        if self.vector_ffi:
            self.vector_ffi.cleanup()
        
        self.logger.info("Enhanced vector service shutdown complete")


# Global service instance
_enhanced_vector_service: Optional[EnhancedVectorService] = None


async def get_enhanced_vector_service() -> EnhancedVectorService:
    """Get or create global enhanced vector service instance"""
    global _enhanced_vector_service
    
    if _enhanced_vector_service is None:
        _enhanced_vector_service = EnhancedVectorService()
        await _enhanced_vector_service.initialize()
    
    return _enhanced_vector_service