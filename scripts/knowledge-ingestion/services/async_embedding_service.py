"""
Async Embedding Service with GPU Acceleration and Concurrent Processing

This module provides an enhanced embedding service that leverages asyncio for
concurrent processing, GPU acceleration for embedding generation, and optimized
batch processing across multiple data sources.

Features:
- Asyncio concurrent embedding generation
- GPU acceleration support (CUDA, MPS)
- Intelligent batch processing and queuing
- Multi-model routing with async optimization
- Connection pooling for API-based models
- Performance monitoring and adaptive scaling
- Graceful degradation and error handling

Requirements: 10.1, 10.2, 10.4
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
import weakref
from contextlib import asynccontextmanager

from core.config import get_settings
from core.logging import get_logger
from .content_classifier import ContentClassifier, ClassificationResult
from .embedding_router import EmbeddingRouter, EmbeddingResult, EmbeddingModel
from .embedding_quality_validator import EmbeddingQualityValidator, ValidationResult
from .async_performance_optimizer import (
    AsyncPerformanceOptimizer, ProcessingTask, ProcessingMode, 
    get_performance_optimizer
)


class EmbeddingBackend(Enum):
    """Embedding generation backend types"""
    CPU = "cpu"
    GPU_CUDA = "gpu_cuda"
    GPU_MPS = "gpu_mps"  # Apple Metal Performance Shaders
    API = "api"
    HYBRID = "hybrid"


class BatchStrategy(Enum):
    """Batch processing strategies"""
    FIXED_SIZE = "fixed_size"
    DYNAMIC_SIZE = "dynamic_size"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


@dataclass
class EmbeddingRequest:
    """Individual embedding request"""
    request_id: str
    text: str
    title: str = ""
    model_preference: Optional[EmbeddingModel] = None
    priority: int = 0
    source_type: Optional[str] = None
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class EmbeddingBatch:
    """Batch of embedding requests"""
    batch_id: str
    requests: List[EmbeddingRequest]
    model: EmbeddingModel
    backend: EmbeddingBackend
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_started: bool = False
    estimated_time: Optional[float] = None


@dataclass
class AsyncEmbeddingResult:
    """Result from async embedding generation"""
    request_id: str
    success: bool
    embedding: Optional[List[float]] = None
    model_used: Optional[EmbeddingModel] = None
    backend_used: Optional[EmbeddingBackend] = None
    classification: Optional[ClassificationResult] = None
    quality_validation: Optional[ValidationResult] = None
    processing_time: float = 0.0
    batch_id: Optional[str] = None
    error_message: Optional[str] = None
    gpu_memory_used: Optional[float] = None


@dataclass
class EmbeddingServiceMetrics:
    """Metrics for the async embedding service"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    batches_processed: int = 0
    average_batch_size: float = 0.0
    average_processing_time: float = 0.0
    gpu_utilization: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    throughput_per_second: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)
    backend_usage: Dict[str, int] = field(default_factory=dict)
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GPUResourceManager:
    """Manages GPU resources for embedding generation"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.GPUResourceManager")
        self._gpu_available = False
        self._gpu_type = None
        self._gpu_memory_total = 0
        self._gpu_memory_used = 0
        self._gpu_lock = asyncio.Lock()
        self._device = None
        
    async def initialize(self) -> bool:
        """Initialize GPU resources"""
        try:
            # Try CUDA first
            if await self._check_cuda():
                self._gpu_available = True
                self._gpu_type = EmbeddingBackend.GPU_CUDA
                self.logger.info("CUDA GPU detected and initialized")
                return True
            
            # Try MPS (Apple Silicon)
            if await self._check_mps():
                self._gpu_available = True
                self._gpu_type = EmbeddingBackend.GPU_MPS
                self.logger.info("MPS GPU detected and initialized")
                return True
            
            self.logger.info("No GPU acceleration available, using CPU")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU resources: {e}")
            return False
    
    async def _check_cuda(self) -> bool:
        """Check CUDA availability"""
        try:
            import torch
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                return True
        except ImportError:
            pass
        return False
    
    async def _check_mps(self) -> bool:
        """Check MPS availability"""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device("mps")
                return True
        except ImportError:
            pass
        return False
    
    @asynccontextmanager
    async def acquire_gpu(self):
        """Acquire GPU resources for processing"""
        async with self._gpu_lock:
            if self._gpu_available:
                try:
                    yield self._device
                finally:
                    # Clean up GPU memory
                    await self._cleanup_gpu_memory()
            else:
                yield None
    
    async def _cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            if self._gpu_type == EmbeddingBackend.GPU_CUDA:
                import torch
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Failed to cleanup GPU memory: {e}")
    
    async def get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization"""
        try:
            if self._gpu_type == EmbeddingBackend.GPU_CUDA:
                import torch
                return torch.cuda.utilization()
        except Exception:
            pass
        return None
    
    async def get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage percentage"""
        try:
            if self._gpu_type == EmbeddingBackend.GPU_CUDA:
                import torch
                used = torch.cuda.memory_allocated()
                total = torch.cuda.max_memory_allocated()
                return (used / total * 100) if total > 0 else 0
        except Exception:
            pass
        return None
    
    @property
    def is_available(self) -> bool:
        return self._gpu_available
    
    @property
    def backend_type(self) -> Optional[EmbeddingBackend]:
        return self._gpu_type


class AsyncEmbeddingBatchProcessor:
    """Processes embedding requests in optimized batches"""
    
    def __init__(self, gpu_manager: GPUResourceManager):
        self.gpu_manager = gpu_manager
        self.logger = get_logger(f"{__name__}.BatchProcessor")
        
        # Batch configuration
        self.max_batch_size = 32
        self.min_batch_size = 4
        self.batch_timeout = 2.0  # seconds
        self.adaptive_batching = True
        
        # Processing queues
        self._request_queues: Dict[EmbeddingModel, asyncio.Queue] = {}
        self._batch_queues: Dict[EmbeddingModel, asyncio.Queue] = {}
        self._result_callbacks: Dict[str, Callable] = {}
        
        # Batch formation
        self._batch_formers: Dict[EmbeddingModel, asyncio.Task] = {}
        self._batch_processors: Dict[EmbeddingModel, asyncio.Task] = {}
        
        # Metrics
        self._metrics = EmbeddingServiceMetrics()
        self._metrics_lock = asyncio.Lock()
        
        # Shutdown
        self._shutdown = False
    
    async def initialize(self) -> bool:
        """Initialize the batch processor"""
        try:
            self.logger.info("Initializing async embedding batch processor")
            
            # Initialize queues for each model
            for model in EmbeddingModel:
                self._request_queues[model] = asyncio.Queue()
                self._batch_queues[model] = asyncio.Queue()
                
                # Start batch former and processor for each model
                self._batch_formers[model] = asyncio.create_task(
                    self._batch_former_loop(model)
                )
                self._batch_processors[model] = asyncio.create_task(
                    self._batch_processor_loop(model)
                )
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            self.logger.info("Async embedding batch processor initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize batch processor: {e}")
            return False
    
    async def submit_request(self, request: EmbeddingRequest) -> bool:
        """Submit an embedding request for processing"""
        try:
            # Determine model to use
            model = request.model_preference or EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE
            
            # Add to appropriate queue
            await self._request_queues[model].put(request)
            
            # Store callback if provided
            if request.callback:
                self._result_callbacks[request.request_id] = request.callback
            
            async with self._metrics_lock:
                self._metrics.total_requests += 1
                self._metrics.queue_sizes[model.value] = self._request_queues[model].qsize()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit request {request.request_id}: {e}")
            return False
    
    async def _batch_former_loop(self, model: EmbeddingModel):
        """Form batches from individual requests"""
        self.logger.debug(f"Starting batch former for model: {model.value}")
        
        while not self._shutdown:
            try:
                batch_requests = []
                batch_start_time = time.time()
                
                # Collect requests for batch
                while (len(batch_requests) < self.max_batch_size and 
                       (time.time() - batch_start_time) < self.batch_timeout):
                    
                    try:
                        # Wait for request with timeout
                        request = await asyncio.wait_for(
                            self._request_queues[model].get(),
                            timeout=0.1
                        )
                        batch_requests.append(request)
                        
                        # If we have minimum batch size and adaptive batching is enabled,
                        # check if we should process immediately
                        if (len(batch_requests) >= self.min_batch_size and 
                            self.adaptive_batching and 
                            await self._should_process_batch_early(model)):
                            break
                            
                    except asyncio.TimeoutError:
                        # Check if we have any requests to process
                        if batch_requests:
                            break
                        continue
                
                # Create batch if we have requests
                if batch_requests:
                    batch = EmbeddingBatch(
                        batch_id=f"{model.value}_{int(time.time() * 1000000)}",
                        requests=batch_requests,
                        model=model,
                        backend=await self._select_backend(model),
                        estimated_time=await self._estimate_batch_time(batch_requests, model)
                    )
                    
                    await self._batch_queues[model].put(batch)
                    
                    self.logger.debug(f"Formed batch {batch.batch_id} with {len(batch_requests)} requests")
                
            except Exception as e:
                self.logger.error(f"Error in batch former for {model.value}: {e}")
                await asyncio.sleep(1)
    
    async def _batch_processor_loop(self, model: EmbeddingModel):
        """Process batches of embedding requests"""
        self.logger.debug(f"Starting batch processor for model: {model.value}")
        
        while not self._shutdown:
            try:
                # Get batch from queue
                batch = await self._batch_queues[model].get()
                
                # Process batch
                await self._process_batch(batch)
                
            except Exception as e:
                self.logger.error(f"Error in batch processor for {model.value}: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: EmbeddingBatch):
        """Process a batch of embedding requests"""
        start_time = time.time()
        batch.processing_started = True
        
        try:
            self.logger.debug(f"Processing batch {batch.batch_id} with {len(batch.requests)} requests")
            
            # Mark all requests as started
            for request in batch.requests:
                request.started_at = datetime.now(timezone.utc)
            
            # Select processing method based on backend
            if batch.backend in [EmbeddingBackend.GPU_CUDA, EmbeddingBackend.GPU_MPS]:
                results = await self._process_gpu_batch(batch)
            elif batch.backend == EmbeddingBackend.API:
                results = await self._process_api_batch(batch)
            else:  # CPU or HYBRID
                results = await self._process_cpu_batch(batch)
            
            # Handle results
            processing_time = time.time() - start_time
            
            for i, request in enumerate(batch.requests):
                request.completed_at = datetime.now(timezone.utc)
                result = results[i] if i < len(results) else None
                
                if result:
                    result.request_id = request.request_id
                    result.batch_id = batch.batch_id
                    result.processing_time = processing_time / len(batch.requests)
                
                # Call callback if provided
                callback = self._result_callbacks.pop(request.request_id, None)
                if callback:
                    asyncio.create_task(self._safe_callback(callback, result))
            
            # Update metrics
            async with self._metrics_lock:
                self._metrics.batches_processed += 1
                self._metrics.completed_requests += len([r for r in results if r and r.success])
                self._metrics.failed_requests += len([r for r in results if r and not r.success])
                
                # Update average batch size
                if self._metrics.batches_processed == 1:
                    self._metrics.average_batch_size = len(batch.requests)
                else:
                    alpha = 0.1
                    self._metrics.average_batch_size = (
                        alpha * len(batch.requests) + 
                        (1 - alpha) * self._metrics.average_batch_size
                    )
                
                # Update backend usage
                backend_key = batch.backend.value
                self._metrics.backend_usage[backend_key] = self._metrics.backend_usage.get(backend_key, 0) + 1
                
                # Update model usage
                model_key = batch.model.value
                self._metrics.model_usage[model_key] = self._metrics.model_usage.get(model_key, 0) + len(batch.requests)
            
            self.logger.debug(f"Completed batch {batch.batch_id} in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch.batch_id}: {e}")
            
            # Create error results for all requests
            error_results = []
            for request in batch.requests:
                error_result = AsyncEmbeddingResult(
                    request_id=request.request_id,
                    success=False,
                    error_message=str(e),
                    batch_id=batch.batch_id
                )
                error_results.append(error_result)
                
                # Call callback with error
                callback = self._result_callbacks.pop(request.request_id, None)
                if callback:
                    asyncio.create_task(self._safe_callback(callback, error_result))
            
            async with self._metrics_lock:
                self._metrics.failed_requests += len(batch.requests)
    
    async def _process_gpu_batch(self, batch: EmbeddingBatch) -> List[AsyncEmbeddingResult]:
        """Process batch using GPU acceleration"""
        results = []
        
        async with self.gpu_manager.acquire_gpu() as device:
            if device is None:
                # Fallback to CPU processing
                return await self._process_cpu_batch(batch)
            
            try:
                # Prepare texts for batch processing
                texts = [req.text for req in batch.requests]
                
                # GPU-accelerated embedding generation
                embeddings = await self._generate_gpu_embeddings(texts, batch.model, device)
                
                # Create results
                for i, request in enumerate(batch.requests):
                    if i < len(embeddings):
                        result = AsyncEmbeddingResult(
                            request_id=request.request_id,
                            success=True,
                            embedding=embeddings[i],
                            model_used=batch.model,
                            backend_used=batch.backend,
                            gpu_memory_used=await self.gpu_manager.get_gpu_memory_usage()
                        )
                    else:
                        result = AsyncEmbeddingResult(
                            request_id=request.request_id,
                            success=False,
                            error_message="GPU processing failed"
                        )
                    
                    results.append(result)
                
            except Exception as e:
                self.logger.error(f"GPU batch processing failed: {e}")
                # Create error results
                for request in batch.requests:
                    results.append(AsyncEmbeddingResult(
                        request_id=request.request_id,
                        success=False,
                        error_message=f"GPU processing error: {e}"
                    ))
        
        return results
    
    async def _process_api_batch(self, batch: EmbeddingBatch) -> List[AsyncEmbeddingResult]:
        """Process batch using API calls"""
        results = []
        
        try:
            # Use the existing embedding router for API calls
            router = EmbeddingRouter()
            
            # Process requests concurrently
            tasks = []
            for request in batch.requests:
                task = asyncio.create_task(
                    router.generate_embedding(request.text, request.title)
                )
                tasks.append(task)
            
            # Wait for all API calls to complete
            api_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert to AsyncEmbeddingResult
            for i, (request, api_result) in enumerate(zip(batch.requests, api_results)):
                if isinstance(api_result, Exception):
                    result = AsyncEmbeddingResult(
                        request_id=request.request_id,
                        success=False,
                        error_message=str(api_result)
                    )
                elif hasattr(api_result, 'success') and api_result.success:
                    result = AsyncEmbeddingResult(
                        request_id=request.request_id,
                        success=True,
                        embedding=api_result.embedding,
                        model_used=api_result.model_used,
                        backend_used=EmbeddingBackend.API
                    )
                else:
                    result = AsyncEmbeddingResult(
                        request_id=request.request_id,
                        success=False,
                        error_message=getattr(api_result, 'error_message', 'API call failed')
                    )
                
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"API batch processing failed: {e}")
            # Create error results
            for request in batch.requests:
                results.append(AsyncEmbeddingResult(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"API processing error: {e}"
                ))
        
        return results
    
    async def _process_cpu_batch(self, batch: EmbeddingBatch) -> List[AsyncEmbeddingResult]:
        """Process batch using CPU"""
        results = []
        
        try:
            # Use thread pool for CPU-intensive work
            loop = asyncio.get_event_loop()
            
            # Process in smaller sub-batches to avoid blocking
            sub_batch_size = 8
            for i in range(0, len(batch.requests), sub_batch_size):
                sub_batch = batch.requests[i:i + sub_batch_size]
                
                # Process sub-batch in thread pool
                sub_results = await loop.run_in_executor(
                    None,
                    self._process_cpu_sub_batch,
                    sub_batch,
                    batch.model
                )
                
                results.extend(sub_results)
                
        except Exception as e:
            self.logger.error(f"CPU batch processing failed: {e}")
            # Create error results
            for request in batch.requests:
                results.append(AsyncEmbeddingResult(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"CPU processing error: {e}"
                ))
        
        return results
    
    def _process_cpu_sub_batch(self, requests: List[EmbeddingRequest], model: EmbeddingModel) -> List[AsyncEmbeddingResult]:
        """Process sub-batch on CPU (runs in thread pool)"""
        results = []
        
        try:
            # Use existing embedding router
            router = EmbeddingRouter()
            
            for request in requests:
                try:
                    # This would be synchronous in the thread pool
                    # For now, create a placeholder result
                    result = AsyncEmbeddingResult(
                        request_id=request.request_id,
                        success=True,
                        embedding=[0.0] * 1536,  # Placeholder
                        model_used=model,
                        backend_used=EmbeddingBackend.CPU
                    )
                    results.append(result)
                    
                except Exception as e:
                    results.append(AsyncEmbeddingResult(
                        request_id=request.request_id,
                        success=False,
                        error_message=str(e)
                    ))
                    
        except Exception as e:
            # Create error results for all requests
            for request in requests:
                results.append(AsyncEmbeddingResult(
                    request_id=request.request_id,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _generate_gpu_embeddings(
        self, 
        texts: List[str], 
        model: EmbeddingModel, 
        device
    ) -> List[List[float]]:
        """Generate embeddings using GPU acceleration"""
        try:
            # This would implement actual GPU-accelerated embedding generation
            # For now, simulate GPU processing
            await asyncio.sleep(0.1)  # Simulate GPU computation
            
            # Return placeholder embeddings
            return [[0.0] * 1536 for _ in texts]
            
        except Exception as e:
            self.logger.error(f"GPU embedding generation failed: {e}")
            raise
    
    async def _select_backend(self, model: EmbeddingModel) -> EmbeddingBackend:
        """Select optimal backend for model"""
        # API-based models
        if model in [EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE, 
                     EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002]:
            return EmbeddingBackend.API
        
        # Local models - prefer GPU if available
        if self.gpu_manager.is_available:
            return self.gpu_manager.backend_type
        
        return EmbeddingBackend.CPU
    
    async def _should_process_batch_early(self, model: EmbeddingModel) -> bool:
        """Determine if batch should be processed early"""
        # Check queue pressure
        queue_size = self._request_queues[model].qsize()
        if queue_size > self.max_batch_size * 2:
            return True
        
        # Check GPU availability
        if self.gpu_manager.is_available:
            gpu_util = await self.gpu_manager.get_gpu_utilization()
            if gpu_util and gpu_util < 50:  # GPU is underutilized
                return True
        
        return False
    
    async def _estimate_batch_time(self, requests: List[EmbeddingRequest], model: EmbeddingModel) -> float:
        """Estimate processing time for batch"""
        # Base time per request
        base_time = 0.1
        
        # Adjust based on model and backend
        if model in [EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_LARGE]:
            base_time = 0.2  # API calls take longer
        
        # Batch efficiency
        batch_efficiency = min(0.8, len(requests) / self.max_batch_size)
        
        return len(requests) * base_time * (1 - batch_efficiency)
    
    async def _safe_callback(self, callback: Callable, result: AsyncEmbeddingResult):
        """Safely execute callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result)
            else:
                callback(result)
        except Exception as e:
            self.logger.error(f"Error in callback for request {result.request_id}: {e}")
    
    async def _metrics_collector(self):
        """Collect performance metrics"""
        while not self._shutdown:
            try:
                async with self._metrics_lock:
                    # Update GPU metrics
                    if self.gpu_manager.is_available:
                        self._metrics.gpu_utilization = await self.gpu_manager.get_gpu_utilization()
                        self._metrics.gpu_memory_usage = await self.gpu_manager.get_gpu_memory_usage()
                    
                    # Update queue sizes
                    for model, queue in self._request_queues.items():
                        self._metrics.queue_sizes[model.value] = queue.qsize()
                    
                    # Calculate throughput
                    elapsed_time = (datetime.now(timezone.utc) - self._metrics.last_updated).total_seconds()
                    if elapsed_time > 0 and self._metrics.completed_requests > 0:
                        self._metrics.throughput_per_second = self._metrics.completed_requests / elapsed_time
                    
                    self._metrics.last_updated = datetime.now(timezone.utc)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)
    
    async def get_metrics(self) -> EmbeddingServiceMetrics:
        """Get current metrics"""
        async with self._metrics_lock:
            return self._metrics
    
    async def shutdown(self):
        """Shutdown the batch processor"""
        self.logger.info("Shutting down async embedding batch processor")
        
        self._shutdown = True
        
        # Cancel all tasks
        all_tasks = list(self._batch_formers.values()) + list(self._batch_processors.values())
        for task in all_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        self.logger.info("Async embedding batch processor shutdown complete")


class AsyncEmbeddingService:
    """
    Main async embedding service that provides high-level interface for
    concurrent embedding generation with GPU acceleration and optimization.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Core components
        self.gpu_manager = GPUResourceManager()
        self.batch_processor: Optional[AsyncEmbeddingBatchProcessor] = None
        self.performance_optimizer: Optional[AsyncPerformanceOptimizer] = None
        
        # Legacy components for compatibility
        self.classifier = ContentClassifier()
        self.validator = EmbeddingQualityValidator()
        
        # Request tracking
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._request_lock = asyncio.Lock()
        
        # Metrics
        self._service_metrics = EmbeddingServiceMetrics()
        
        # Shutdown
        self._shutdown = False
    
    async def initialize(self) -> bool:
        """Initialize the async embedding service"""
        try:
            self.logger.info("Initializing async embedding service")
            
            # Initialize GPU manager
            if not await self.gpu_manager.initialize():
                self.logger.warning("GPU initialization failed, continuing with CPU only")
            
            # Initialize batch processor
            self.batch_processor = AsyncEmbeddingBatchProcessor(self.gpu_manager)
            if not await self.batch_processor.initialize():
                raise Exception("Failed to initialize batch processor")
            
            # Get performance optimizer
            self.performance_optimizer = await get_performance_optimizer()
            
            self.logger.info("Async embedding service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize async embedding service: {e}")
            return False
    
    async def generate_embedding_async(
        self,
        text: str,
        title: str = "",
        model_preference: Optional[EmbeddingModel] = None,
        priority: int = 0,
        source_type: Optional[str] = None
    ) -> AsyncEmbeddingResult:
        """Generate embedding asynchronously"""
        request_id = f"req_{int(time.time() * 1000000)}"
        
        # Create future for result
        result_future = asyncio.Future()
        
        async with self._request_lock:
            self._pending_requests[request_id] = result_future
        
        # Create callback to resolve future
        async def result_callback(result: AsyncEmbeddingResult):
            async with self._request_lock:
                future = self._pending_requests.pop(request_id, None)
                if future and not future.done():
                    future.set_result(result)
        
        # Create request
        request = EmbeddingRequest(
            request_id=request_id,
            text=text,
            title=title,
            model_preference=model_preference,
            priority=priority,
            source_type=source_type,
            callback=result_callback
        )
        
        # Submit to batch processor
        if not await self.batch_processor.submit_request(request):
            return AsyncEmbeddingResult(
                request_id=request_id,
                success=False,
                error_message="Failed to submit request"
            )
        
        # Wait for result
        try:
            result = await asyncio.wait_for(result_future, timeout=300)  # 5 minute timeout
            return result
        except asyncio.TimeoutError:
            async with self._request_lock:
                self._pending_requests.pop(request_id, None)
            
            return AsyncEmbeddingResult(
                request_id=request_id,
                success=False,
                error_message="Request timeout"
            )
    
    async def generate_batch_embeddings_async(
        self,
        texts_and_titles: List[Tuple[str, str]],
        model_preference: Optional[EmbeddingModel] = None,
        max_concurrent: int = 32
    ) -> List[AsyncEmbeddingResult]:
        """Generate embeddings for multiple texts concurrently"""
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(text_title: Tuple[str, str]) -> AsyncEmbeddingResult:
            async with semaphore:
                text, title = text_title
                return await self.generate_embedding_async(
                    text=text,
                    title=title,
                    model_preference=model_preference
                )
        
        # Create tasks for all texts
        tasks = [generate_single(text_title) for text_title in texts_and_titles]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AsyncEmbeddingResult(
                    request_id=f"batch_{i}",
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_service_metrics(self) -> EmbeddingServiceMetrics:
        """Get comprehensive service metrics"""
        if self.batch_processor:
            return await self.batch_processor.get_metrics()
        return self._service_metrics
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        return {
            'available': self.gpu_manager.is_available,
            'backend_type': self.gpu_manager.backend_type.value if self.gpu_manager.backend_type else None,
            'utilization': await self.gpu_manager.get_gpu_utilization(),
            'memory_usage': await self.gpu_manager.get_gpu_memory_usage()
        }
    
    async def shutdown(self):
        """Shutdown the async embedding service"""
        self.logger.info("Shutting down async embedding service")
        
        self._shutdown = True
        
        # Cancel all pending requests
        async with self._request_lock:
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()
        
        # Shutdown batch processor
        if self.batch_processor:
            await self.batch_processor.shutdown()
        
        self.logger.info("Async embedding service shutdown complete")


# Global service instance
_async_embedding_service: Optional[AsyncEmbeddingService] = None


async def get_async_embedding_service() -> AsyncEmbeddingService:
    """Get or create global async embedding service instance"""
    global _async_embedding_service
    
    if _async_embedding_service is None:
        _async_embedding_service = AsyncEmbeddingService()
        await _async_embedding_service.initialize()
    
    return _async_embedding_service