"""
Enhanced Batch Ingestion Manager with Async Optimizations

This module provides an enhanced version of the batch ingestion manager that
leverages all async performance optimizations including concurrent processing,
GPU acceleration, optimized database operations, and intelligent resource
management across all data sources.

Features:
- Async concurrent processing across all data sources
- GPU-accelerated embedding generation
- Optimized database operations with connection pooling
- Intelligent resource allocation and backpressure handling
- Performance monitoring and adaptive scaling
- Cross-source batch optimization
- Graceful degradation and error recovery

Requirements: 10.1, 10.2, 10.4
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Set, Callable, AsyncIterator, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import uuid
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil

from core.config import get_settings
from core.logging import get_logger
from .multi_source_auth import DataSourceType, get_auth_service
from .unified_browsing_service import UniversalFileMetadata, get_browsing_service
from .async_performance_optimizer import (
    AsyncPerformanceOptimizer, ProcessingTask, ProcessingMode,
    get_performance_optimizer
)
from .async_embedding_service import (
    AsyncEmbeddingService, EmbeddingRequest, AsyncEmbeddingResult,
    get_async_embedding_service
)
from .async_database_service import (
    AsyncDatabaseService, get_async_database_service
)
from .supabase_storage import (
    DocumentMetadata, ChunkData, EmbeddedChunk, IngestionLogEntry,
    ProcessingStatus, IngestionPhase, IngestionStatus
)

# Import from original batch manager for compatibility
from .batch_ingestion_manager import (
    JobStatus, JobPriority, FileProcessingStatus, ProcessingOptions,
    FileJobItem, SourceProgress, BatchIngestionJob, JobStatistics,
    WebSocketConnection
)


@dataclass
class EnhancedProcessingOptions(ProcessingOptions):
    """Enhanced processing options with async optimizations"""
    # Async processing options
    enable_async_processing: bool = True
    max_concurrent_files: int = 16
    max_concurrent_sources: int = 4
    
    # GPU acceleration options
    enable_gpu_acceleration: bool = True
    gpu_batch_size: int = 32
    gpu_memory_limit_mb: int = 4096
    
    # Database optimization options
    enable_batch_database_ops: bool = True
    database_batch_size: int = 100
    max_database_connections: int = 10
    
    # Performance optimization options
    enable_adaptive_batching: bool = True
    enable_intelligent_queuing: bool = True
    enable_resource_monitoring: bool = True
    
    # Backpressure and throttling
    enable_backpressure_handling: bool = True
    memory_threshold_mb: int = 8192
    cpu_threshold_percent: float = 80.0
    
    # Error handling and recovery
    enable_graceful_degradation: bool = True
    fallback_to_sync_processing: bool = True
    auto_retry_failed_optimizations: bool = True


@dataclass
class ProcessingMetrics:
    """Enhanced processing metrics"""
    # Basic metrics
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    # Performance metrics
    average_processing_time: float = 0.0
    throughput_files_per_second: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_cpu_usage: float = 0.0
    
    # Async processing metrics
    concurrent_operations: int = 0
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    worker_utilization: Dict[str, float] = field(default_factory=dict)
    
    # GPU metrics
    gpu_utilization: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    gpu_accelerated_operations: int = 0
    
    # Database metrics
    database_operations: int = 0
    database_batch_operations: int = 0
    average_database_latency: float = 0.0
    connection_pool_utilization: float = 0.0
    
    # Error and recovery metrics
    optimization_failures: int = 0
    fallback_operations: int = 0
    backpressure_events: int = 0
    
    # Source-specific metrics
    source_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ResourceMonitor:
    """Monitors system resources and provides optimization recommendations"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ResourceMonitor")
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._metrics_history: List[Dict[str, float]] = []
        self._max_history = 100
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            self.logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            self.logger.info("Resource monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect current metrics
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
                    'disk_io_read': psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
                    'disk_io_write': psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
                    'network_sent': psutil.net_io_counters().bytes_sent,
                    'network_recv': psutil.net_io_counters().bytes_recv,
                }
                
                # Add to history
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history.pop(0)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(5)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics"""
        if self._metrics_history:
            return self._metrics_history[-1]
        return {}
    
    def should_throttle_processing(self, options: EnhancedProcessingOptions) -> bool:
        """Determine if processing should be throttled"""
        current = self.get_current_metrics()
        
        if not current:
            return False
        
        # Check CPU threshold
        if current.get('cpu_percent', 0) > options.cpu_threshold_percent:
            return True
        
        # Check memory threshold
        if current.get('memory_used_mb', 0) > options.memory_threshold_mb:
            return True
        
        return False
    
    def get_optimal_concurrency(self, base_concurrency: int) -> int:
        """Get optimal concurrency based on current resources"""
        current = self.get_current_metrics()
        
        if not current:
            return base_concurrency
        
        cpu_usage = current.get('cpu_percent', 0)
        memory_usage = current.get('memory_percent', 0)
        
        # Reduce concurrency if resources are stressed
        if cpu_usage > 80 or memory_usage > 80:
            return max(1, base_concurrency // 2)
        elif cpu_usage > 60 or memory_usage > 60:
            return max(1, int(base_concurrency * 0.75))
        elif cpu_usage < 30 and memory_usage < 30:
            return min(base_concurrency * 2, 64)  # Cap at reasonable maximum
        
        return base_concurrency


class EnhancedBatchIngestionManager:
    """
    Enhanced batch ingestion manager with comprehensive async optimizations
    and intelligent resource management across all data sources.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Service dependencies
        self._auth_service = None
        self._browsing_service = None
        self._performance_optimizer: Optional[AsyncPerformanceOptimizer] = None
        self._embedding_service: Optional[AsyncEmbeddingService] = None
        self._database_service: Optional[AsyncDatabaseService] = None
        
        # Job management (inherited from original)
        self._jobs: Dict[str, BatchIngestionJob] = {}
        self._job_queue: List[str] = []
        self._active_jobs: Set[str] = set()
        self._job_lock = asyncio.Lock()
        
        # Enhanced processing
        self._processing_semaphores: Dict[DataSourceType, asyncio.Semaphore] = {}
        self._processing_queues: Dict[DataSourceType, asyncio.Queue] = {}
        self._processing_workers: Dict[DataSourceType, List[asyncio.Task]] = {}
        
        # Resource monitoring
        self._resource_monitor = ResourceMonitor()
        
        # WebSocket connections (inherited from original)
        self._websocket_connections: Dict[str, List[WebSocketConnection]] = {}
        self._connection_lock = asyncio.Lock()
        
        # Enhanced metrics
        self._enhanced_metrics = ProcessingMetrics()
        self._metrics_lock = asyncio.Lock()
        
        # Configuration
        self._default_enhanced_options = EnhancedProcessingOptions()
        
        # Shutdown flag
        self._shutdown = False
    
    async def initialize(self) -> bool:
        """Initialize the enhanced batch ingestion manager"""
        try:
            self.logger.info("Initializing enhanced batch ingestion manager")
            
            # Initialize service dependencies
            self._auth_service = get_auth_service()
            self._browsing_service = get_browsing_service()
            await self._browsing_service.initialize()
            
            # Initialize performance optimizer
            self._performance_optimizer = await get_performance_optimizer()
            
            # Initialize async embedding service
            self._embedding_service = await get_async_embedding_service()
            
            # Initialize async database service
            self._database_service = await get_async_database_service()
            
            # Initialize processing infrastructure
            await self._initialize_processing_infrastructure()
            
            # Start resource monitoring
            await self._resource_monitor.start_monitoring()
            
            # Start background tasks
            asyncio.create_task(self._enhanced_job_processor())
            asyncio.create_task(self._enhanced_metrics_collector())
            asyncio.create_task(self._websocket_cleanup())
            
            self.logger.info("Enhanced batch ingestion manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced batch ingestion manager: {e}")
            return False
    
    async def _initialize_processing_infrastructure(self):
        """Initialize processing infrastructure for each data source"""
        for source_type in DataSourceType:
            # Create semaphore for concurrency control
            max_concurrent = self._default_enhanced_options.max_concurrent_files
            self._processing_semaphores[source_type] = asyncio.Semaphore(max_concurrent)
            
            # Create processing queue
            self._processing_queues[source_type] = asyncio.Queue()
            
            # Create worker tasks
            num_workers = min(4, max_concurrent // 2)
            workers = []
            for i in range(num_workers):
                worker_task = asyncio.create_task(
                    self._processing_worker(source_type, f"{source_type.value}_worker_{i}")
                )
                workers.append(worker_task)
            
            self._processing_workers[source_type] = workers
            
            self.logger.info(f"Initialized {num_workers} workers for {source_type.value}")
    
    async def create_enhanced_job(
        self,
        user_id: str,
        name: str,
        file_selections: List[Dict[str, Any]],
        processing_options: Optional[EnhancedProcessingOptions] = None,
        description: Optional[str] = None,
        priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """Create an enhanced batch ingestion job with async optimizations"""
        try:
            job_id = str(uuid.uuid4())
            
            # Use enhanced processing options
            enhanced_options = processing_options or self._default_enhanced_options
            
            # Create base job (reuse logic from original manager)
            job = BatchIngestionJob(
                job_id=job_id,
                user_id=user_id,
                name=name,
                description=description,
                priority=priority,
                processing_options=enhanced_options  # This will be cast to ProcessingOptions
            )
            
            # Process file selections and organize by source
            source_files: Dict[str, List[UniversalFileMetadata]] = {}
            
            for selection in file_selections:
                connection_id = selection['connection_id']
                file_ids = selection['file_ids']
                source_type = DataSourceType(selection['source_type'])
                
                # Get file metadata for each selected file
                for file_id in file_ids:
                    file_metadata = await self._browsing_service.get_file_metadata(file_id, connection_id)
                    if file_metadata:
                        key = f"{source_type.value}:{connection_id}"
                        if key not in source_files:
                            source_files[key] = []
                        source_files[key].append(file_metadata)
            
            # Create source progress tracking with enhanced metrics
            for key, files in source_files.items():
                source_type_str, connection_id = key.split(':', 1)
                source_type = DataSourceType(source_type_str)
                
                source_progress = SourceProgress(
                    source_type=source_type,
                    connection_id=connection_id,
                    total_files=len(files)
                )
                
                # Create file job items
                for file_metadata in files:
                    file_item = FileJobItem(
                        file_id=file_metadata.file_id,
                        file_metadata=file_metadata
                    )
                    source_progress.files.append(file_item)
                
                job.source_progress.append(source_progress)
            
            # Update job totals
            job.update_progress()
            
            # Enhanced estimation with async optimizations
            job.estimated_duration_ms = await self._estimate_enhanced_processing_duration(job, enhanced_options)
            
            # Store job
            async with self._job_lock:
                self._jobs[job_id] = job
                self._job_queue.append(job_id)
            
            # Persist to database (using async database service)
            await self._persist_job_async(job)
            
            # Update enhanced metrics
            await self._update_enhanced_metrics()
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Created enhanced batch ingestion job {job_id} with {job.total_files} files")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced batch ingestion job: {e}")
            raise
    
    async def _enhanced_job_processor(self):
        """Enhanced job processor with async optimizations"""
        while not self._shutdown:
            try:
                # Get next job from queue with priority handling
                job_id = None
                async with self._job_lock:
                    if self._job_queue and len(self._active_jobs) < self.settings.max_concurrent_jobs:
                        # Find highest priority job that can be processed
                        available_jobs = []
                        for jid in self._job_queue:
                            job = self._jobs.get(jid)
                            if job and job.status == JobStatus.QUEUED:
                                # Check if we have resources to process this job
                                if not self._resource_monitor.should_throttle_processing(self._default_enhanced_options):
                                    available_jobs.append((jid, job))
                        
                        if available_jobs:
                            # Sort by priority and creation time
                            priority_order = {
                                JobPriority.URGENT: 0,
                                JobPriority.HIGH: 1,
                                JobPriority.NORMAL: 2,
                                JobPriority.LOW: 3
                            }
                            available_jobs.sort(key=lambda x: (
                                priority_order.get(x[1].priority, 999),
                                x[1].created_at
                            ))
                            job_id, job = available_jobs[0]
                            
                            # Mark as running
                            job.status = JobStatus.RUNNING
                            self._active_jobs.add(job_id)
                            self._job_queue.remove(job_id)
                
                if job_id:
                    # Start enhanced processing
                    asyncio.create_task(self._process_job_enhanced(job_id))
                
                # Wait before checking again
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in enhanced job processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_job_enhanced(self, job_id: str):
        """Process job with enhanced async optimizations"""
        try:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            self.logger.info(f"Starting enhanced processing of job {job_id}")
            start_time = time.time()
            
            # Get enhanced processing options
            enhanced_options = self._get_enhanced_options(job)
            
            # Process sources concurrently with intelligent resource allocation
            if enhanced_options.enable_async_processing:
                await self._process_sources_concurrently(job, enhanced_options)
            else:
                # Fallback to sequential processing
                await self._process_sources_sequentially(job, enhanced_options)
            
            # Update job completion
            async with self._job_lock:
                if job.status == JobStatus.RUNNING:
                    job.update_progress()
                    
                    if job.failed_files > 0 and job.completed_files == 0:
                        job.status = JobStatus.FAILED
                        job.error_message = f"All {job.failed_files} files failed to process"
                    else:
                        job.status = JobStatus.COMPLETED
                    
                    job.completed_at = datetime.now(timezone.utc)
                    job.actual_duration_ms = int((time.time() - start_time) * 1000)
                
                self._active_jobs.discard(job_id)
            
            # Persist final status
            await self._persist_job_async(job)
            
            # Update enhanced metrics
            await self._update_enhanced_metrics()
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Enhanced processing of job {job_id} completed in {job.actual_duration_ms}ms")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced job processing {job_id}: {e}")
            
            # Mark job as failed
            async with self._job_lock:
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.now(timezone.utc)
                
                self._active_jobs.discard(job_id)
            
            # Persist and notify
            if job_id in self._jobs:
                await self._persist_job_async(self._jobs[job_id])
                await self._broadcast_job_update(self._jobs[job_id])
    
    async def _process_sources_concurrently(self, job: BatchIngestionJob, options: EnhancedProcessingOptions):
        """Process sources concurrently with resource optimization"""
        # Create semaphore for source-level concurrency
        source_semaphore = asyncio.Semaphore(options.max_concurrent_sources)
        
        async def process_source_with_semaphore(source_progress: SourceProgress):
            async with source_semaphore:
                await self._process_source_enhanced(job, source_progress, options)
        
        # Create tasks for all sources
        tasks = [
            process_source_with_semaphore(source_progress)
            for source_progress in job.source_progress
        ]
        
        # Execute with monitoring
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_sources_sequentially(self, job: BatchIngestionJob, options: EnhancedProcessingOptions):
        """Process sources sequentially (fallback mode)"""
        for source_progress in job.source_progress:
            if job.status != JobStatus.RUNNING:
                break
            
            await self._process_source_enhanced(job, source_progress, options)
    
    async def _process_source_enhanced(
        self,
        job: BatchIngestionJob,
        source_progress: SourceProgress,
        options: EnhancedProcessingOptions
    ):
        """Process files from a source with enhanced optimizations"""
        try:
            # Get optimal concurrency based on current resources
            optimal_concurrency = self._resource_monitor.get_optimal_concurrency(
                options.max_concurrent_files
            )
            
            # Create semaphore for file-level concurrency
            file_semaphore = asyncio.Semaphore(optimal_concurrency)
            
            async def process_file_with_semaphore(file_item: FileJobItem):
                async with file_semaphore:
                    await self._process_file_enhanced(job, source_progress, file_item, options)
            
            # Create tasks for all files
            tasks = [
                process_file_with_semaphore(file_item)
                for file_item in source_progress.files
                if file_item.status == FileProcessingStatus.PENDING
            ]
            
            # Execute with adaptive batching if enabled
            if options.enable_adaptive_batching and len(tasks) > 10:
                await self._execute_adaptive_batches(tasks, optimal_concurrency)
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error processing source {source_progress.source_type}: {e}")
            
            # Mark remaining files as failed
            for file_item in source_progress.files:
                if file_item.status == FileProcessingStatus.PENDING:
                    file_item.status = FileProcessingStatus.FAILED
                    file_item.error_message = f"Source processing failed: {e}"
                    source_progress.failed_files += 1
    
    async def _execute_adaptive_batches(self, tasks: List[asyncio.Task], concurrency: int):
        """Execute tasks in adaptive batches based on system performance"""
        batch_size = min(concurrency * 2, len(tasks))
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            # Execute batch
            await asyncio.gather(*batch, return_exceptions=True)
            
            # Check if we should adjust batch size based on performance
            if self._resource_monitor.should_throttle_processing(self._default_enhanced_options):
                batch_size = max(1, batch_size // 2)
                await asyncio.sleep(1)  # Brief pause to let system recover
            elif batch_size < concurrency * 2:
                batch_size = min(batch_size * 2, concurrency * 2)
    
    async def _process_file_enhanced(
        self,
        job: BatchIngestionJob,
        source_progress: SourceProgress,
        file_item: FileJobItem,
        options: EnhancedProcessingOptions
    ):
        """Process a single file with all enhanced optimizations"""
        try:
            file_item.status = FileProcessingStatus.ACCESSING
            file_item.started_at = datetime.now(timezone.utc)
            file_item.current_step = "Accessing file"
            file_item.progress = 10
            
            await self._broadcast_file_progress(job, source_progress, file_item)
            
            # Check if we should skip existing files
            if options.skip_existing:
                existing_doc = await self._database_service.execute_query_async(
                    query_type='SELECT',
                    table='documents',
                    filters={'file_id': file_item.file_id}
                )
                if existing_doc and existing_doc.data and existing_doc.data[0].get('processing_status') == ProcessingStatus.COMPLETED.value:
                    file_item.status = FileProcessingStatus.SKIPPED
                    file_item.current_step = "Skipped (already processed)"
                    file_item.progress = 100
                    file_item.completed_at = datetime.now(timezone.utc)
                    source_progress.skipped_files += 1
                    await self._broadcast_file_progress(job, source_progress, file_item)
                    return
            
            # Enhanced file processing pipeline
            await self._enhanced_file_pipeline(job, source_progress, file_item, options)
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_item.file_id}: {e}")
            
            file_item.status = FileProcessingStatus.FAILED
            file_item.error_message = str(e)
            file_item.current_step = f"Failed: {str(e)[:100]}"
            file_item.completed_at = datetime.now(timezone.utc)
            if file_item.started_at:
                file_item.processing_time_ms = int((file_item.completed_at - file_item.started_at).total_seconds() * 1000)
            source_progress.failed_files += 1
        
        finally:
            await self._broadcast_file_progress(job, source_progress, file_item)
    
    async def _enhanced_file_pipeline(
        self,
        job: BatchIngestionJob,
        source_progress: SourceProgress,
        file_item: FileJobItem,
        options: EnhancedProcessingOptions
    ):
        """Enhanced file processing pipeline with async optimizations"""
        # Download/access file
        file_item.current_step = "Downloading file"
        file_item.progress = 20
        await self._broadcast_file_progress(job, source_progress, file_item)
        
        connector = self._auth_service.get_connector(source_progress.source_type)
        file_content = await connector.download_file(file_item.file_metadata)
        
        # Parse PDF (submit to performance optimizer)
        file_item.status = FileProcessingStatus.PARSING
        file_item.current_step = "Parsing PDF"
        file_item.progress = 40
        await self._broadcast_file_progress(job, source_progress, file_item)
        
        # Submit parsing task to performance optimizer
        parsing_task_data = {
            'operation_type': 'pdf_parsing',
            'file_content': file_content,
            'use_llm': options.use_llm_parsing,
            'file_metadata': file_item.file_metadata
        }
        
        parsed_doc = await self._performance_optimizer.submit_task(
            pool_name='cpu_processing',
            task_data=parsing_task_data,
            processing_mode=ProcessingMode.CPU_BOUND,
            source_type=source_progress.source_type
        )
        
        # Chunk content
        file_item.status = FileProcessingStatus.CHUNKING
        file_item.current_step = "Creating chunks"
        file_item.progress = 60
        await self._broadcast_file_progress(job, source_progress, file_item)
        
        # Submit chunking task
        chunking_task_data = {
            'operation_type': 'semantic_chunking',
            'document': parsed_doc,
            'chunk_size': options.chunk_size,
            'overlap': options.chunk_overlap
        }
        
        chunks = await self._performance_optimizer.submit_task(
            pool_name='cpu_processing',
            task_data=chunking_task_data,
            processing_mode=ProcessingMode.CPU_BOUND,
            source_type=source_progress.source_type
        )
        
        file_item.chunks_created = len(chunks)
        
        # Generate embeddings (using async embedding service)
        file_item.status = FileProcessingStatus.EMBEDDING
        file_item.current_step = "Generating embeddings"
        file_item.progress = 80
        await self._broadcast_file_progress(job, source_progress, file_item)
        
        # Prepare embedding requests
        embedding_requests = []
        for chunk in chunks:
            embedding_requests.append((chunk.content, file_item.file_metadata.name))
        
        # Generate embeddings concurrently
        embedding_results = await self._embedding_service.generate_batch_embeddings_async(
            embedding_requests,
            max_concurrent=options.gpu_batch_size if options.enable_gpu_acceleration else 8
        )
        
        # Create embedded chunks
        embedded_chunks = []
        for i, (chunk, embedding_result) in enumerate(zip(chunks, embedding_results)):
            if embedding_result.success:
                embedded_chunk = EmbeddedChunk(
                    chunk_data=ChunkData(
                        document_id="",  # Will be set after document storage
                        content=chunk.content,
                        chunk_order=chunk.chunk_order,
                        section_header=chunk.section_header,
                        semantic_metadata=chunk.semantic_metadata,
                        token_count=chunk.token_count,
                        embedding_model=embedding_result.model_used.value if embedding_result.model_used else None,
                        embedding_dimension=len(embedding_result.embedding) if embedding_result.embedding else None,
                        quality_score=embedding_result.quality_validation.quality_score if embedding_result.quality_validation else None
                    ),
                    embedding_vector=embedding_result.embedding or []
                )
                embedded_chunks.append(embedded_chunk)
        
        if embedded_chunks:
            file_item.embedding_model = embedded_chunks[0].chunk_data.embedding_model
            file_item.quality_score = sum(ec.chunk_data.quality_score or 0 for ec in embedded_chunks) / len(embedded_chunks)
        
        # Store in database (using async database service)
        file_item.status = FileProcessingStatus.STORING
        file_item.current_step = "Storing in database"
        file_item.progress = 90
        await self._broadcast_file_progress(job, source_progress, file_item)
        
        # Create document metadata
        document_metadata = DocumentMetadata(
            file_id=file_item.file_id,
            title=file_item.file_metadata.name,
            source_url=file_item.file_metadata.access_url,
            content=parsed_doc.content if hasattr(parsed_doc, 'content') else None,
            structure=parsed_doc.structure if hasattr(parsed_doc, 'structure') else None,
            parsing_method=parsed_doc.parsing_method.value if hasattr(parsed_doc, 'parsing_method') else None,
            quality_score=file_item.quality_score,
            domain_classification=file_item.file_metadata.domain_classification,
            processing_status=ProcessingStatus.COMPLETED,
            file_size_bytes=file_item.file_metadata.size
        )
        
        # Store document and chunks using async database service
        result = await self._database_service.store_document_with_chunks_async(
            document_metadata,
            embedded_chunks
        )
        
        if result.success:
            file_item.status = FileProcessingStatus.COMPLETED
            file_item.current_step = "Completed"
            file_item.progress = 100
            file_item.document_id = result.document_id
            file_item.completed_at = datetime.now(timezone.utc)
            file_item.processing_time_ms = int((file_item.completed_at - file_item.started_at).total_seconds() * 1000)
            source_progress.completed_files += 1
        else:
            raise Exception(f"Storage failed: {result.error_message}")
    
    def _get_enhanced_options(self, job: BatchIngestionJob) -> EnhancedProcessingOptions:
        """Get enhanced processing options from job"""
        # Convert base ProcessingOptions to EnhancedProcessingOptions
        base_options = job.processing_options
        enhanced_options = EnhancedProcessingOptions()
        
        # Copy base options
        for field in ProcessingOptions.__dataclass_fields__:
            if hasattr(base_options, field):
                setattr(enhanced_options, field, getattr(base_options, field))
        
        return enhanced_options
    
    async def _estimate_enhanced_processing_duration(
        self,
        job: BatchIngestionJob,
        options: EnhancedProcessingOptions
    ) -> int:
        """Estimate processing duration with async optimizations"""
        # Base time per file (reduced due to async optimizations)
        base_time_per_file = 15000  # 15 seconds (vs 30 in original)
        
        # Adjust based on processing options
        multiplier = 1.0
        if options.use_llm_parsing:
            multiplier *= 1.5  # Reduced from 2.0 due to async processing
        if not options.enable_async_processing:
            multiplier *= 2.0
        if options.enable_gpu_acceleration:
            multiplier *= 0.7  # GPU acceleration reduces time
        
        # Calculate total time
        total_time = job.total_files * base_time_per_file * multiplier
        
        # Adjust for concurrent processing
        if options.enable_async_processing:
            concurrency_factor = min(options.max_concurrent_files, job.total_files)
            total_time = total_time / concurrency_factor
        
        return int(total_time)
    
    async def _enhanced_metrics_collector(self):
        """Collect enhanced performance metrics"""
        while not self._shutdown:
            try:
                async with self._metrics_lock:
                    # Update basic metrics from jobs
                    self._enhanced_metrics.total_files = sum(job.total_files for job in self._jobs.values())
                    self._enhanced_metrics.processed_files = sum(job.completed_files for job in self._jobs.values())
                    self._enhanced_metrics.failed_files = sum(job.failed_files for job in self._jobs.values())
                    self._enhanced_metrics.skipped_files = sum(job.skipped_files for job in self._jobs.values())
                    
                    # Get performance optimizer metrics
                    if self._performance_optimizer:
                        perf_metrics = await self._performance_optimizer.get_global_metrics()
                        self._enhanced_metrics.concurrent_operations = sum(perf_metrics.active_workers.values())
                        self._enhanced_metrics.queue_sizes = perf_metrics.queue_sizes
                        self._enhanced_metrics.peak_memory_usage_mb = perf_metrics.memory_usage_mb
                    
                    # Get embedding service metrics
                    if self._embedding_service:
                        embedding_metrics = await self._embedding_service.get_service_metrics()
                        self._enhanced_metrics.gpu_utilization = embedding_metrics.gpu_utilization
                        self._enhanced_metrics.gpu_memory_usage = embedding_metrics.gpu_memory_usage
                        self._enhanced_metrics.gpu_accelerated_operations = sum(
                            count for backend, count in embedding_metrics.backend_usage.items()
                            if 'gpu' in backend.lower()
                        )
                    
                    # Get database service metrics
                    if self._database_service:
                        db_metrics = await self._database_service.get_service_metrics()
                        self._enhanced_metrics.database_operations = db_metrics.total_queries
                        self._enhanced_metrics.average_database_latency = db_metrics.average_query_time
                        self._enhanced_metrics.connection_pool_utilization = db_metrics.pool_utilization
                    
                    # Get resource metrics
                    resource_metrics = self._resource_monitor.get_current_metrics()
                    if resource_metrics:
                        self._enhanced_metrics.average_cpu_usage = resource_metrics.get('cpu_percent', 0)
                    
                    # Calculate throughput
                    elapsed_time = (datetime.now(timezone.utc) - self._enhanced_metrics.last_updated).total_seconds()
                    if elapsed_time > 0 and self._enhanced_metrics.processed_files > 0:
                        self._enhanced_metrics.throughput_files_per_second = self._enhanced_metrics.processed_files / elapsed_time
                    
                    self._enhanced_metrics.last_updated = datetime.now(timezone.utc)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting enhanced metrics: {e}")
                await asyncio.sleep(5)
    
    async def get_enhanced_metrics(self) -> ProcessingMetrics:
        """Get current enhanced metrics"""
        async with self._metrics_lock:
            return self._enhanced_metrics
    
    async def _persist_job_async(self, job: BatchIngestionJob):
        """Persist job using async database service"""
        try:
            job_data = job.to_dict()
            
            # Store in jobs table using async database service
            await self._database_service.execute_query_async(
                query_type='INSERT',
                table='ingestion_jobs',
                data=job_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to persist job {job.job_id}: {e}")
    
    async def _processing_worker(self, source_type: DataSourceType, worker_name: str):
        """Processing worker for specific data source type"""
        self.logger.debug(f"Starting processing worker: {worker_name}")
        
        while not self._shutdown:
            try:
                # Get task from queue
                queue = self._processing_queues[source_type]
                task = await queue.get()
                
                # Process task
                await self._process_worker_task(task, worker_name)
                
                # Mark task as done
                queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in processing worker {worker_name}: {e}")
                await asyncio.sleep(1)
    
    async def _process_worker_task(self, task: Any, worker_name: str):
        """Process a worker task"""
        # Implementation would depend on task type
        # This is a placeholder for the worker task processing
        await asyncio.sleep(0.1)  # Simulate work
    
    # Inherit other methods from original BatchIngestionManager
    async def _broadcast_job_update(self, job: BatchIngestionJob):
        """Broadcast job update to WebSocket clients"""
        message = {
            'type': 'job_update',
            'job': job.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self._broadcast_to_user(job.user_id, message)
    
    async def _broadcast_file_progress(
        self,
        job: BatchIngestionJob,
        source_progress: SourceProgress,
        file_item: FileJobItem
    ):
        """Broadcast file progress update to WebSocket clients"""
        message = {
            'type': 'file_progress',
            'job_id': job.job_id,
            'source_type': source_progress.source_type.value,
            'connection_id': source_progress.connection_id,
            'file': file_item.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self._broadcast_to_user(job.user_id, message)
    
    async def _broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections for a user"""
        async with self._connection_lock:
            connections = self._websocket_connections.get(user_id, [])
            
            # Send to all connections
            for connection in connections[:]:  # Copy list to avoid modification during iteration
                success = await connection.send_message(message)
                if not success:
                    # Remove dead connection
                    connections.remove(connection)
    
    async def _websocket_cleanup(self):
        """Background task to clean up dead WebSocket connections"""
        while not self._shutdown:
            try:
                current_time = datetime.now(timezone.utc)
                
                async with self._connection_lock:
                    for user_id, connections in list(self._websocket_connections.items()):
                        # Check each connection
                        active_connections = []
                        for connection in connections:
                            # Check if connection is still alive
                            if (current_time - connection.last_ping).total_seconds() < 300:  # 5 minutes
                                if await connection.ping():
                                    active_connections.append(connection)
                        
                        # Update connections list
                        if active_connections:
                            self._websocket_connections[user_id] = active_connections
                        else:
                            del self._websocket_connections[user_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _update_enhanced_metrics(self):
        """Update enhanced metrics"""
        # This would update various enhanced metrics
        pass
    
    async def shutdown(self):
        """Shutdown the enhanced batch ingestion manager"""
        self.logger.info("Shutting down enhanced batch ingestion manager")
        
        self._shutdown = True
        
        # Stop resource monitoring
        await self._resource_monitor.stop_monitoring()
        
        # Cancel all processing workers
        for workers in self._processing_workers.values():
            for worker in workers:
                worker.cancel()
        
        # Wait for workers to finish
        all_workers = []
        for workers in self._processing_workers.values():
            all_workers.extend(workers)
        
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        
        # Shutdown service dependencies
        if self._embedding_service:
            await self._embedding_service.shutdown()
        
        if self._database_service:
            await self._database_service.shutdown()
        
        if self._performance_optimizer:
            await self._performance_optimizer.shutdown()
        
        # Close WebSocket connections
        async with self._connection_lock:
            for connections in self._websocket_connections.values():
                for connection in connections:
                    try:
                        await connection.websocket.close()
                    except Exception:
                        pass
            self._websocket_connections.clear()
        
        self.logger.info("Enhanced batch ingestion manager shutdown complete")


# Global service instance
_enhanced_batch_manager: Optional[EnhancedBatchIngestionManager] = None


async def get_enhanced_batch_manager() -> EnhancedBatchIngestionManager:
    """Get or create global enhanced batch ingestion manager instance"""
    global _enhanced_batch_manager
    
    if _enhanced_batch_manager is None:
        _enhanced_batch_manager = EnhancedBatchIngestionManager()
        await _enhanced_batch_manager.initialize()
    
    return _enhanced_batch_manager