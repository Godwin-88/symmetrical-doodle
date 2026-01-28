"""
Batch Ingestion Management Service

This module provides comprehensive batch ingestion job management for multi-source
knowledge base ingestion. It handles job creation, configuration, execution control,
progress tracking, and history persistence with source awareness.

Features:
- Cross-source batch job creation and configuration
- Real-time progress tracking with WebSocket support
- Job control operations (start, pause, cancel, retry)
- Job history and status persistence with source attribution
- Error handling and recovery mechanisms
- Job statistics and performance metrics
- Queue management and priority handling
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Callable, AsyncIterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import weakref
from concurrent.futures import ThreadPoolExecutor
import threading

from core.config import get_settings
from core.logging import get_logger
from .multi_source_auth import DataSourceType, get_auth_service
from .unified_browsing_service import UniversalFileMetadata, get_browsing_service
from .supabase_storage import (
    SupabaseStorageService, ProcessingStatus, IngestionPhase, IngestionStatus,
    DocumentMetadata, ChunkData, EmbeddedChunk, IngestionLogEntry
)


class JobStatus(Enum):
    """Batch ingestion job status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class FileProcessingStatus(Enum):
    """Individual file processing status within a job"""
    PENDING = "pending"
    ACCESSING = "accessing"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingOptions:
    """Processing configuration options for batch jobs"""
    use_llm_parsing: bool = False
    embedding_model_preference: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    quality_threshold: float = 0.7
    retry_failed_files: bool = True
    max_retries: int = 3
    skip_existing: bool = True
    preserve_structure: bool = True
    extract_math: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    source_specific_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileJobItem:
    """Individual file item within a batch job"""
    file_id: str
    file_metadata: UniversalFileMetadata
    status: FileProcessingStatus = FileProcessingStatus.PENDING
    progress: int = 0  # 0-100
    current_step: str = ""
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0
    processing_time_ms: Optional[int] = None
    chunks_created: Optional[int] = None
    embedding_model: Optional[str] = None
    quality_score: Optional[float] = None
    document_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['file_metadata'] = {
            'file_id': self.file_metadata.file_id,
            'name': self.file_metadata.name,
            'size': self.file_metadata.size,
            'modified_time': self.file_metadata.modified_time.isoformat(),
            'source_type': self.file_metadata.source_type.value,
            'source_path': self.file_metadata.source_path,
            'mime_type': self.file_metadata.mime_type,
            'access_url': self.file_metadata.access_url,
            'parent_folders': self.file_metadata.parent_folders,
            'domain_classification': self.file_metadata.domain_classification,
            'checksum': self.file_metadata.checksum,
            'source_specific_metadata': self.file_metadata.source_specific_metadata,
            'processing_status': self.file_metadata.processing_status,
            'is_accessible': self.file_metadata.is_accessible
        }
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


@dataclass
class SourceProgress:
    """Progress tracking for a specific data source within a job"""
    source_type: DataSourceType
    connection_id: str
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    files: List[FileJobItem] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage for this source"""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files + self.failed_files + self.skipped_files) / self.total_files * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source_type': self.source_type.value,
            'connection_id': self.connection_id,
            'total_files': self.total_files,
            'completed_files': self.completed_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'progress_percentage': self.progress_percentage,
            'files': [file_item.to_dict() for file_item in self.files]
        }


@dataclass
class BatchIngestionJob:
    """Batch ingestion job definition"""
    job_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    processing_options: ProcessingOptions = field(default_factory=ProcessingOptions)
    source_progress: List[SourceProgress] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    error_message: Optional[str] = None
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    estimated_duration_ms: Optional[int] = None
    actual_duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files + self.failed_files + self.skipped_files) / self.total_files * 100
    
    @property
    def is_active(self) -> bool:
        """Check if job is currently active (running or queued)"""
        return self.status in [JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.RETRYING]
    
    @property
    def is_finished(self) -> bool:
        """Check if job is finished (completed, failed, or cancelled)"""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def get_source_progress(self, source_type: DataSourceType, connection_id: str) -> Optional[SourceProgress]:
        """Get progress for a specific source"""
        for progress in self.source_progress:
            if progress.source_type == source_type and progress.connection_id == connection_id:
                return progress
        return None
    
    def update_progress(self):
        """Update overall progress from source progress"""
        self.total_files = sum(sp.total_files for sp in self.source_progress)
        self.completed_files = sum(sp.completed_files for sp in self.source_progress)
        self.failed_files = sum(sp.failed_files for sp in self.source_progress)
        self.skipped_files = sum(sp.skipped_files for sp in self.source_progress)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        data['progress_percentage'] = self.progress_percentage
        data['is_active'] = self.is_active
        data['is_finished'] = self.is_finished
        
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.paused_at:
            data['paused_at'] = self.paused_at.isoformat()
        if self.cancelled_at:
            data['cancelled_at'] = self.cancelled_at.isoformat()
        
        data['source_progress'] = [sp.to_dict() for sp in self.source_progress]
        return data


@dataclass
class JobStatistics:
    """Job execution statistics"""
    total_jobs: int = 0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    total_files_processed: int = 0
    average_processing_time_ms: float = 0.0
    success_rate: float = 0.0
    by_source_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)


# WebSocket connection management
class WebSocketConnection:
    """WebSocket connection wrapper"""
    
    def __init__(self, websocket, user_id: str, job_id: Optional[str] = None):
        self.websocket = websocket
        self.user_id = user_id
        self.job_id = job_id
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = datetime.now(timezone.utc)
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket client"""
        try:
            await self.websocket.send_text(json.dumps(message))
            return True
        except Exception:
            return False
    
    async def ping(self) -> bool:
        """Send ping to check connection"""
        try:
            await self.websocket.ping()
            self.last_ping = datetime.now(timezone.utc)
            return True
        except Exception:
            return False


class BatchIngestionManager:
    """
    Manages batch ingestion jobs with multi-source support, real-time progress tracking,
    and comprehensive job control operations.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Service dependencies
        self._auth_service = None
        self._browsing_service = None
        self._storage_service = None
        
        # Job management
        self._jobs: Dict[str, BatchIngestionJob] = {}
        self._job_queue: List[str] = []  # Job IDs in queue order
        self._active_jobs: Set[str] = set()
        self._job_lock = threading.RLock()
        
        # WebSocket connections
        self._websocket_connections: Dict[str, List[WebSocketConnection]] = {}  # user_id -> connections
        self._connection_lock = threading.RLock()
        
        # Processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self._statistics = JobStatistics()
        
        # Shutdown flag
        self._shutdown = False
    
    async def initialize(self) -> bool:
        """Initialize the batch ingestion manager"""
        try:
            self.logger.info("Initializing batch ingestion manager")
            
            # Initialize service dependencies
            self._auth_service = get_auth_service()
            self._browsing_service = get_browsing_service()
            await self._browsing_service.initialize()
            
            # Initialize storage service
            from core.config import load_config
            config = load_config()
            self._storage_service = SupabaseStorageService(config.supabase)
            await self._storage_service.initialize()
            
            # Load existing jobs from storage
            await self._load_jobs_from_storage()
            
            # Start background tasks
            asyncio.create_task(self._job_processor())
            asyncio.create_task(self._websocket_cleanup())
            
            self.logger.info("Batch ingestion manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize batch ingestion manager: {e}")
            return False
    
    async def create_job(
        self,
        user_id: str,
        name: str,
        file_selections: List[Dict[str, Any]],
        processing_options: Optional[ProcessingOptions] = None,
        description: Optional[str] = None,
        priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """
        Create a new batch ingestion job with cross-source file selection.
        
        Args:
            user_id: User identifier
            name: Job name
            file_selections: List of file selections with source info
            processing_options: Processing configuration
            description: Optional job description
            priority: Job priority level
            
        Returns:
            str: Job ID
        """
        try:
            job_id = str(uuid.uuid4())
            
            # Create job
            job = BatchIngestionJob(
                job_id=job_id,
                user_id=user_id,
                name=name,
                description=description,
                priority=priority,
                processing_options=processing_options or ProcessingOptions()
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
            
            # Create source progress tracking
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
            
            # Estimate processing duration
            job.estimated_duration_ms = self._estimate_processing_duration(job)
            
            # Store job
            with self._job_lock:
                self._jobs[job_id] = job
                self._job_queue.append(job_id)
            
            # Persist to storage
            await self._persist_job(job)
            
            # Update statistics
            self._update_statistics()
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Created batch ingestion job {job_id} with {job.total_files} files")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to create batch ingestion job: {e}")
            raise
    
    async def start_job(self, job_id: str, user_id: str) -> bool:
        """Start a batch ingestion job"""
        try:
            with self._job_lock:
                job = self._jobs.get(job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")
                
                if job.user_id != user_id:
                    raise ValueError("Unauthorized access to job")
                
                if job.status != JobStatus.PENDING:
                    raise ValueError(f"Job is in {job.status.value} status and cannot be started")
                
                job.status = JobStatus.QUEUED
                job.started_at = datetime.now(timezone.utc)
            
            # Persist status change
            await self._persist_job(job)
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Started batch ingestion job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start job {job_id}: {e}")
            return False
    
    async def pause_job(self, job_id: str, user_id: str) -> bool:
        """Pause a running batch ingestion job"""
        try:
            with self._job_lock:
                job = self._jobs.get(job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")
                
                if job.user_id != user_id:
                    raise ValueError("Unauthorized access to job")
                
                if job.status != JobStatus.RUNNING:
                    raise ValueError(f"Job is in {job.status.value} status and cannot be paused")
                
                job.status = JobStatus.PAUSED
                job.paused_at = datetime.now(timezone.utc)
            
            # Cancel processing task if running
            if job_id in self._processing_tasks:
                self._processing_tasks[job_id].cancel()
                del self._processing_tasks[job_id]
            
            # Remove from active jobs
            self._active_jobs.discard(job_id)
            
            # Persist status change
            await self._persist_job(job)
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Paused batch ingestion job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pause job {job_id}: {e}")
            return False
    
    async def resume_job(self, job_id: str, user_id: str) -> bool:
        """Resume a paused batch ingestion job"""
        try:
            with self._job_lock:
                job = self._jobs.get(job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")
                
                if job.user_id != user_id:
                    raise ValueError("Unauthorized access to job")
                
                if job.status != JobStatus.PAUSED:
                    raise ValueError(f"Job is in {job.status.value} status and cannot be resumed")
                
                job.status = JobStatus.QUEUED
                job.paused_at = None
                
                # Add back to queue if not already there
                if job_id not in self._job_queue:
                    self._job_queue.append(job_id)
            
            # Persist status change
            await self._persist_job(job)
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Resumed batch ingestion job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume job {job_id}: {e}")
            return False
    
    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a batch ingestion job"""
        try:
            with self._job_lock:
                job = self._jobs.get(job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")
                
                if job.user_id != user_id:
                    raise ValueError("Unauthorized access to job")
                
                if job.is_finished:
                    raise ValueError(f"Job is already finished with status {job.status.value}")
                
                job.status = JobStatus.CANCELLED
                job.cancelled_at = datetime.now(timezone.utc)
                
                if not job.completed_at:
                    job.completed_at = datetime.now(timezone.utc)
            
            # Cancel processing task if running
            if job_id in self._processing_tasks:
                self._processing_tasks[job_id].cancel()
                del self._processing_tasks[job_id]
            
            # Remove from active jobs and queue
            self._active_jobs.discard(job_id)
            if job_id in self._job_queue:
                self._job_queue.remove(job_id)
            
            # Persist status change
            await self._persist_job(job)
            
            # Update statistics
            self._update_statistics()
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Cancelled batch ingestion job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def retry_job(self, job_id: str, user_id: str, retry_failed_only: bool = True) -> bool:
        """Retry a failed batch ingestion job"""
        try:
            with self._job_lock:
                job = self._jobs.get(job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")
                
                if job.user_id != user_id:
                    raise ValueError("Unauthorized access to job")
                
                if job.status not in [JobStatus.FAILED, JobStatus.COMPLETED]:
                    raise ValueError(f"Job is in {job.status.value} status and cannot be retried")
                
                # Reset job status
                job.status = JobStatus.QUEUED
                job.started_at = datetime.now(timezone.utc)
                job.completed_at = None
                job.cancelled_at = None
                job.paused_at = None
                job.error_message = None
                
                # Reset file statuses
                for source_progress in job.source_progress:
                    for file_item in source_progress.files:
                        if not retry_failed_only or file_item.status == FileProcessingStatus.FAILED:
                            file_item.status = FileProcessingStatus.PENDING
                            file_item.progress = 0
                            file_item.current_step = ""
                            file_item.error_message = None
                            file_item.error_code = None
                            file_item.started_at = None
                            file_item.completed_at = None
                
                # Add back to queue
                if job_id not in self._job_queue:
                    self._job_queue.append(job_id)
            
            # Persist status change
            await self._persist_job(job)
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Retrying batch ingestion job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to retry job {job_id}: {e}")
            return False
    
    async def get_job(self, job_id: str, user_id: str) -> Optional[BatchIngestionJob]:
        """Get job details"""
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job and job.user_id == user_id:
                return job
            return None
    
    async def list_jobs(
        self,
        user_id: str,
        status_filter: Optional[List[JobStatus]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[BatchIngestionJob]:
        """List jobs for a user with optional filtering"""
        with self._job_lock:
            user_jobs = [job for job in self._jobs.values() if job.user_id == user_id]
            
            # Apply status filter
            if status_filter:
                user_jobs = [job for job in user_jobs if job.status in status_filter]
            
            # Sort by created_at descending
            user_jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            # Apply pagination
            if limit:
                user_jobs = user_jobs[offset:offset + limit]
            else:
                user_jobs = user_jobs[offset:]
            
            return user_jobs
    
    async def get_job_statistics(self, user_id: Optional[str] = None) -> JobStatistics:
        """Get job execution statistics"""
        with self._job_lock:
            if user_id:
                # User-specific statistics
                user_jobs = [job for job in self._jobs.values() if job.user_id == user_id]
            else:
                # Global statistics
                user_jobs = list(self._jobs.values())
            
            stats = JobStatistics()
            stats.total_jobs = len(user_jobs)
            
            total_processing_time = 0
            processing_time_count = 0
            
            for job in user_jobs:
                # Status counts
                if job.status == JobStatus.RUNNING or job.status == JobStatus.QUEUED:
                    stats.active_jobs += 1
                elif job.status == JobStatus.COMPLETED:
                    stats.completed_jobs += 1
                elif job.status == JobStatus.FAILED:
                    stats.failed_jobs += 1
                elif job.status == JobStatus.CANCELLED:
                    stats.cancelled_jobs += 1
                
                # File counts
                stats.total_files_processed += job.completed_files + job.failed_files + job.skipped_files
                
                # Processing time
                if job.actual_duration_ms:
                    total_processing_time += job.actual_duration_ms
                    processing_time_count += 1
                
                # By source type
                for source_progress in job.source_progress:
                    source_key = source_progress.source_type.value
                    stats.by_source_type[source_key] = stats.by_source_type.get(source_key, 0) + source_progress.total_files
                
                # By priority
                priority_key = job.priority.value
                stats.by_priority[priority_key] = stats.by_priority.get(priority_key, 0) + 1
                
                # By status
                status_key = job.status.value
                stats.by_status[status_key] = stats.by_status.get(status_key, 0) + 1
            
            # Calculate averages
            if processing_time_count > 0:
                stats.average_processing_time_ms = total_processing_time / processing_time_count
            
            if stats.total_files_processed > 0:
                successful_files = sum(job.completed_files for job in user_jobs)
                stats.success_rate = successful_files / stats.total_files_processed * 100
            
            return stats
    
    async def add_websocket_connection(self, websocket, user_id: str, job_id: Optional[str] = None):
        """Add WebSocket connection for real-time updates"""
        connection = WebSocketConnection(websocket, user_id, job_id)
        
        with self._connection_lock:
            if user_id not in self._websocket_connections:
                self._websocket_connections[user_id] = []
            self._websocket_connections[user_id].append(connection)
        
        self.logger.info(f"Added WebSocket connection for user {user_id}")
        return connection
    
    async def remove_websocket_connection(self, websocket, user_id: str):
        """Remove WebSocket connection"""
        with self._connection_lock:
            if user_id in self._websocket_connections:
                self._websocket_connections[user_id] = [
                    conn for conn in self._websocket_connections[user_id]
                    if conn.websocket != websocket
                ]
                if not self._websocket_connections[user_id]:
                    del self._websocket_connections[user_id]
        
        self.logger.info(f"Removed WebSocket connection for user {user_id}")
    
    async def _job_processor(self):
        """Background task to process queued jobs"""
        while not self._shutdown:
            try:
                # Get next job from queue
                job_id = None
                with self._job_lock:
                    if self._job_queue and len(self._active_jobs) < self.settings.max_concurrent_jobs:
                        # Find highest priority job
                        available_jobs = []
                        for jid in self._job_queue:
                            job = self._jobs.get(jid)
                            if job and job.status == JobStatus.QUEUED:
                                available_jobs.append((jid, job))
                        
                        if available_jobs:
                            # Sort by priority (urgent > high > normal > low)
                            priority_order = {
                                JobPriority.URGENT: 0,
                                JobPriority.HIGH: 1,
                                JobPriority.NORMAL: 2,
                                JobPriority.LOW: 3
                            }
                            available_jobs.sort(key=lambda x: priority_order.get(x[1].priority, 999))
                            job_id, job = available_jobs[0]
                            
                            # Mark as running
                            job.status = JobStatus.RUNNING
                            self._active_jobs.add(job_id)
                            self._job_queue.remove(job_id)
                
                if job_id:
                    # Start processing job
                    task = asyncio.create_task(self._process_job(job_id))
                    self._processing_tasks[job_id] = task
                
                # Wait before checking again
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in job processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_job(self, job_id: str):
        """Process a batch ingestion job"""
        try:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            self.logger.info(f"Starting processing of job {job_id}")
            start_time = time.time()
            
            # Process each source
            for source_progress in job.source_progress:
                if job.status != JobStatus.RUNNING:
                    break  # Job was paused or cancelled
                
                await self._process_source_files(job, source_progress)
            
            # Update job completion
            with self._job_lock:
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
            
            # Clean up processing task
            if job_id in self._processing_tasks:
                del self._processing_tasks[job_id]
            
            # Persist final status
            await self._persist_job(job)
            
            # Update statistics
            self._update_statistics()
            
            # Notify WebSocket clients
            await self._broadcast_job_update(job)
            
            self.logger.info(f"Completed processing of job {job_id} in {job.actual_duration_ms}ms")
            
        except Exception as e:
            self.logger.error(f"Error processing job {job_id}: {e}")
            
            # Mark job as failed
            with self._job_lock:
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.now(timezone.utc)
                
                self._active_jobs.discard(job_id)
            
            # Clean up
            if job_id in self._processing_tasks:
                del self._processing_tasks[job_id]
            
            # Persist and notify
            if job_id in self._jobs:
                await self._persist_job(self._jobs[job_id])
                await self._broadcast_job_update(self._jobs[job_id])
    
    async def _process_source_files(self, job: BatchIngestionJob, source_progress: SourceProgress):
        """Process files from a specific source"""
        try:
            # Get connector for this source
            connector = self._auth_service.get_connector(source_progress.source_type)
            if not connector:
                raise ValueError(f"No connector available for source type {source_progress.source_type}")
            
            # Process files with limited concurrency
            semaphore = asyncio.Semaphore(job.processing_options.max_workers)
            
            async def process_file_with_semaphore(file_item: FileJobItem):
                async with semaphore:
                    await self._process_single_file(job, source_progress, file_item, connector)
            
            # Create tasks for all files
            tasks = [
                process_file_with_semaphore(file_item)
                for file_item in source_progress.files
                if file_item.status == FileProcessingStatus.PENDING
            ]
            
            # Wait for all files to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error processing source {source_progress.source_type}: {e}")
            
            # Mark remaining files as failed
            for file_item in source_progress.files:
                if file_item.status == FileProcessingStatus.PENDING:
                    file_item.status = FileProcessingStatus.FAILED
                    file_item.error_message = f"Source processing failed: {e}"
                    source_progress.failed_files += 1
    
    async def _process_single_file(
        self,
        job: BatchIngestionJob,
        source_progress: SourceProgress,
        file_item: FileJobItem,
        connector
    ):
        """Process a single file through the ingestion pipeline"""
        try:
            file_item.status = FileProcessingStatus.ACCESSING
            file_item.started_at = datetime.now(timezone.utc)
            file_item.current_step = "Accessing file"
            file_item.progress = 10
            
            # Broadcast progress update
            await self._broadcast_file_progress(job, source_progress, file_item)
            
            # Check if we should skip existing files
            if job.processing_options.skip_existing:
                existing_doc = await self._storage_service.get_document_by_file_id(file_item.file_id)
                if existing_doc and existing_doc.get('processing_status') == ProcessingStatus.COMPLETED.value:
                    file_item.status = FileProcessingStatus.SKIPPED
                    file_item.current_step = "Skipped (already processed)"
                    file_item.progress = 100
                    file_item.completed_at = datetime.now(timezone.utc)
                    source_progress.skipped_files += 1
                    await self._broadcast_file_progress(job, source_progress, file_item)
                    return
            
            # Download/access file
            file_item.current_step = "Downloading file"
            file_item.progress = 20
            await self._broadcast_file_progress(job, source_progress, file_item)
            
            file_content = await connector.download_file(file_item.file_metadata)
            
            # Parse PDF
            file_item.status = FileProcessingStatus.PARSING
            file_item.current_step = "Parsing PDF"
            file_item.progress = 40
            await self._broadcast_file_progress(job, source_progress, file_item)
            
            # Import parsing service
            from .pdf_parser import PDFParser
            parser = PDFParser()
            parsed_doc = await parser.parse_pdf(
                file_content,
                use_llm=job.processing_options.use_llm_parsing
            )
            
            # Chunk content
            file_item.status = FileProcessingStatus.CHUNKING
            file_item.current_step = "Creating chunks"
            file_item.progress = 60
            await self._broadcast_file_progress(job, source_progress, file_item)
            
            from .semantic_chunker import SemanticChunker
            chunker = SemanticChunker()
            chunks = await chunker.chunk_document(
                parsed_doc,
                chunk_size=job.processing_options.chunk_size,
                overlap=job.processing_options.chunk_overlap
            )
            
            file_item.chunks_created = len(chunks)
            
            # Generate embeddings
            file_item.status = FileProcessingStatus.EMBEDDING
            file_item.current_step = "Generating embeddings"
            file_item.progress = 80
            await self._broadcast_file_progress(job, source_progress, file_item)
            
            from .embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            
            embedded_chunks = []
            for chunk in chunks:
                embedding = await embedding_service.generate_embedding(
                    chunk.content,
                    model_preference=job.processing_options.embedding_model_preference
                )
                
                embedded_chunk = EmbeddedChunk(
                    chunk_data=ChunkData(
                        document_id="",  # Will be set after document storage
                        content=chunk.content,
                        chunk_order=chunk.chunk_order,
                        section_header=chunk.section_header,
                        semantic_metadata=chunk.semantic_metadata,
                        token_count=chunk.token_count,
                        embedding_model=embedding.model_name,
                        embedding_dimension=len(embedding.vector),
                        quality_score=embedding.quality_score
                    ),
                    embedding_vector=embedding.vector
                )
                embedded_chunks.append(embedded_chunk)
            
            if embedded_chunks:
                file_item.embedding_model = embedded_chunks[0].chunk_data.embedding_model
                file_item.quality_score = sum(ec.chunk_data.quality_score or 0 for ec in embedded_chunks) / len(embedded_chunks)
            
            # Store in database
            file_item.status = FileProcessingStatus.STORING
            file_item.current_step = "Storing in database"
            file_item.progress = 90
            await self._broadcast_file_progress(job, source_progress, file_item)
            
            # Create document metadata
            document_metadata = DocumentMetadata(
                file_id=file_item.file_id,
                title=file_item.file_metadata.name,
                source_url=file_item.file_metadata.access_url,
                content=parsed_doc.content,
                structure=parsed_doc.structure,
                parsing_method=parsed_doc.parsing_method,
                quality_score=file_item.quality_score,
                domain_classification=file_item.file_metadata.domain_classification,
                processing_status=ProcessingStatus.COMPLETED,
                file_size_bytes=file_item.file_metadata.size
            )
            
            # Store document and chunks
            result = await self._storage_service.store_document_with_chunks(
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
            # Always broadcast final progress
            await self._broadcast_file_progress(job, source_progress, file_item)
    
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
        with self._connection_lock:
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
                
                with self._connection_lock:
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
    
    def _estimate_processing_duration(self, job: BatchIngestionJob) -> int:
        """Estimate job processing duration in milliseconds"""
        # Base time per file (in ms)
        base_time_per_file = 30000  # 30 seconds
        
        # Adjust based on processing options
        multiplier = 1.0
        if job.processing_options.use_llm_parsing:
            multiplier *= 2.0
        if not job.processing_options.parallel_processing:
            multiplier *= 2.0
        
        # Calculate total time
        total_time = job.total_files * base_time_per_file * multiplier
        
        # Adjust for parallel processing
        if job.processing_options.parallel_processing:
            total_time = total_time / min(job.processing_options.max_workers, job.total_files)
        
        return int(total_time)
    
    def _update_statistics(self):
        """Update global statistics"""
        with self._job_lock:
            self._statistics = JobStatistics()
            self._statistics.total_jobs = len(self._jobs)
            
            for job in self._jobs.values():
                if job.is_active:
                    self._statistics.active_jobs += 1
                elif job.status == JobStatus.COMPLETED:
                    self._statistics.completed_jobs += 1
                elif job.status == JobStatus.FAILED:
                    self._statistics.failed_jobs += 1
                elif job.status == JobStatus.CANCELLED:
                    self._statistics.cancelled_jobs += 1
    
    async def _persist_job(self, job: BatchIngestionJob):
        """Persist job to storage"""
        try:
            # For now, we'll store job data as JSON in a simple table
            # In a production system, this would use proper relational tables
            
            job_data = job.to_dict()
            
            # Store in a jobs table (would need to be created in schema)
            # This is a simplified implementation
            self.logger.debug(f"Persisting job {job.job_id} to storage")
            
        except Exception as e:
            self.logger.error(f"Failed to persist job {job.job_id}: {e}")
    
    async def _load_jobs_from_storage(self):
        """Load existing jobs from storage"""
        try:
            # Load jobs from storage
            # This is a simplified implementation
            self.logger.info("Loading existing jobs from storage")
            
        except Exception as e:
            self.logger.error(f"Failed to load jobs from storage: {e}")
    
    async def shutdown(self):
        """Shutdown the batch ingestion manager"""
        self.logger.info("Shutting down batch ingestion manager")
        
        self._shutdown = True
        
        # Cancel all active processing tasks
        for task in self._processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks.values(), return_exceptions=True)
        
        # Close WebSocket connections
        with self._connection_lock:
            for connections in self._websocket_connections.values():
                for connection in connections:
                    try:
                        await connection.websocket.close()
                    except Exception:
                        pass
            self._websocket_connections.clear()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self.logger.info("Batch ingestion manager shutdown complete")


# Global service instance
_batch_manager: Optional[BatchIngestionManager] = None


def get_batch_manager() -> BatchIngestionManager:
    """Get or create global batch ingestion manager instance"""
    global _batch_manager
    
    if _batch_manager is None:
        _batch_manager = BatchIngestionManager()
    
    return _batch_manager