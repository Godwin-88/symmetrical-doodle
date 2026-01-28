#!/usr/bin/env python3
"""
Main Multi-Source Pipeline Orchestration System

This module provides the main orchestration script that coordinates all three phases
(discovery → ingestion → audit) across all supported data sources with comprehensive
progress tracking, status reporting, logging, and monitoring.

Features:
- Complete multi-source pipeline orchestration
- Progress tracking across all phases and data sources
- Comprehensive logging and monitoring with source attribution
- Error handling and recovery mechanisms
- Performance monitoring and optimization
- Integration with all existing services
- Configuration management for orchestration settings
- Real-time status reporting and WebSocket updates

Requirements: All requirements integration (19.1)
"""

import asyncio
import logging
import sys
import time
import signal
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_settings, KnowledgeIngestionSettings
from core.logging import get_logger, set_correlation_id, configure_logging
from core.state_manager import StateManager, ExecutionPhase, ExecutionStatus

# Multi-source services
from services.multi_source_auth import (
    get_auth_service, MultiSourceAuthenticationService, DataSourceType
)
from services.unified_browsing_service import (
    get_browsing_service, UnifiedBrowsingService, UniversalFileMetadata
)
from services.enhanced_batch_manager import (
    get_enhanced_batch_manager, EnhancedBatchIngestionManager,
    EnhancedProcessingOptions, ProcessingMetrics
)

# Core processing services
from services.async_performance_optimizer import (
    get_performance_optimizer, AsyncPerformanceOptimizer
)
from services.async_embedding_service import get_async_embedding_service
from services.async_database_service import get_async_database_service

# Quality and audit services
from services.quality_audit_service import QualityAuditService
from services.coverage_analysis_service import CoverageAnalysisService
from services.knowledge_readiness_memo import KnowledgeReadinessMemoService

class OrchestrationPhase(Enum):
    """Orchestration phases for the complete pipeline"""
    INITIALIZATION = "initialization"
    DISCOVERY = "discovery"
    INGESTION = "ingestion"
    AUDIT = "audit"
    COMPLETION = "completion"


class OrchestrationStatus(Enum):
    """Overall orchestration status"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SourceOrchestrationState:
    """Orchestration state for a specific data source"""
    source_type: DataSourceType
    connection_id: str
    source_name: str
    total_files: int = 0
    discovered_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    current_phase: OrchestrationPhase = OrchestrationPhase.INITIALIZATION
    phase_status: OrchestrationStatus = OrchestrationStatus.NOT_STARTED
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    processing_metrics: Optional[Dict[str, Any]] = None


@dataclass
class OrchestrationProgress:
    """Overall orchestration progress tracking"""
    orchestration_id: str
    user_id: str
    started_at: datetime
    current_phase: OrchestrationPhase = OrchestrationPhase.INITIALIZATION
    overall_status: OrchestrationStatus = OrchestrationStatus.NOT_STARTED
    
    # Source-specific progress
    source_states: Dict[str, SourceOrchestrationState] = field(default_factory=dict)
    
    # Aggregate metrics
    total_sources: int = 0
    active_sources: int = 0
    total_files: int = 0
    discovered_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    # Phase timing
    phase_start_times: Dict[OrchestrationPhase, datetime] = field(default_factory=dict)
    phase_durations: Dict[OrchestrationPhase, float] = field(default_factory=dict)
    
    # Performance metrics
    throughput_files_per_second: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    
    # Error tracking
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    
    # Resource utilization
    peak_memory_usage_mb: float = 0.0
    average_cpu_usage: float = 0.0
    
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration behavior"""
    # Source selection
    enabled_sources: List[DataSourceType] = field(default_factory=list)
    source_priorities: Dict[DataSourceType, int] = field(default_factory=dict)
    
    # Processing options
    max_concurrent_sources: int = 4
    max_concurrent_files_per_source: int = 16
    enable_cross_source_optimization: bool = True
    
    # Progress reporting
    progress_update_interval_seconds: int = 5
    enable_websocket_updates: bool = True
    enable_detailed_logging: bool = True
    
    # Error handling
    continue_on_source_failure: bool = True
    max_retries_per_source: int = 3
    retry_delay_seconds: float = 5.0
    
    # Performance optimization
    enable_adaptive_batching: bool = True
    enable_resource_monitoring: bool = True
    memory_threshold_mb: int = 8192
    cpu_threshold_percent: float = 80.0
    
    # Audit and quality
    enable_quality_audit: bool = True
    enable_coverage_analysis: bool = True
    generate_readiness_memo: bool = True

class MultiSourcePipelineOrchestrator:
    """
    Main orchestrator for the complete multi-source knowledge ingestion pipeline.
    Coordinates discovery, ingestion, and audit phases across all data sources
    with comprehensive progress tracking, monitoring, and error handling.
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.config = config or OrchestrationConfig()
        
        # Service dependencies
        self._auth_service: Optional[MultiSourceAuthenticationService] = None
        self._browsing_service: Optional[UnifiedBrowsingService] = None
        self._batch_manager: Optional[EnhancedBatchIngestionManager] = None
        self._performance_optimizer: Optional[AsyncPerformanceOptimizer] = None
        self._embedding_service = None
        self._database_service = None
        
        # Quality and audit services
        self._quality_audit_service: Optional[QualityAuditService] = None
        self._coverage_service: Optional[CoverageAnalysisService] = None
        self._readiness_service: Optional[KnowledgeReadinessMemoService] = None
        
        # State management
        self._state_manager: Optional[StateManager] = None
        self._orchestration_progress: Optional[OrchestrationProgress] = None
        self._progress_lock = asyncio.Lock()
        
        # Background tasks
        self._progress_reporter_task: Optional[asyncio.Task] = None
        self._metrics_collector_task: Optional[asyncio.Task] = None
        self._websocket_broadcaster_task: Optional[asyncio.Task] = None
        
        # Shutdown handling
        self._shutdown_requested = False
        self._shutdown_event = asyncio.Event()
        
        # WebSocket connections for real-time updates
        self._websocket_connections: Dict[str, List[Any]] = {}
        self._websocket_lock = asyncio.Lock()
        
        # Performance tracking
        self._start_time: Optional[datetime] = None
        self._phase_metrics: Dict[OrchestrationPhase, Dict[str, Any]] = {}
        
        # Error tracking
        self._error_history: List[Dict[str, Any]] = []
        self._warning_history: List[Dict[str, Any]] = []
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator and all service dependencies"""
        try:
            self.logger.info("Initializing multi-source pipeline orchestrator")
            
            # Set correlation ID for this orchestration session
            correlation_id = set_correlation_id()
            self.logger.info(f"Orchestration session started", correlation_id=correlation_id)
            
            # Initialize service dependencies
            self.logger.info("Initializing service dependencies")
            
            # Authentication service
            self._auth_service = get_auth_service()
            await self._auth_service.initialize()
            
            # Browsing service
            self._browsing_service = get_browsing_service()
            await self._browsing_service.initialize()
            
            # Enhanced batch manager
            self._batch_manager = await get_enhanced_batch_manager()
            
            # Performance optimizer
            self._performance_optimizer = await get_performance_optimizer()
            
            # Async services
            self._embedding_service = await get_async_embedding_service()
            self._database_service = await get_async_database_service()
            
            # Quality and audit services
            self._quality_audit_service = QualityAuditService()
            self._coverage_service = CoverageAnalysisService()
            self._readiness_service = KnowledgeReadinessMemoService()
            
            # State manager
            self._state_manager = StateManager()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Multi-source pipeline orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
            return False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self._shutdown_requested = True
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _start_background_tasks(self):
        """Start background monitoring and reporting tasks"""
        self._progress_reporter_task = asyncio.create_task(self._progress_reporter())
        self._metrics_collector_task = asyncio.create_task(self._metrics_collector())
        
        if self.config.enable_websocket_updates:
            self._websocket_broadcaster_task = asyncio.create_task(self._websocket_broadcaster())
    
    async def orchestrate_complete_pipeline(
        self,
        user_id: str,
        source_selections: List[Dict[str, Any]],
        processing_options: Optional[EnhancedProcessingOptions] = None
    ) -> OrchestrationProgress:
        """
        Orchestrate the complete multi-source pipeline from discovery to audit.
        
        Args:
            user_id: User identifier
            source_selections: List of source configurations with connection details
            processing_options: Enhanced processing options
            
        Returns:
            OrchestrationProgress: Final orchestration progress and results
        """
        try:
            # Initialize orchestration progress
            orchestration_id = str(uuid.uuid4())
            self._orchestration_progress = OrchestrationProgress(
                orchestration_id=orchestration_id,
                user_id=user_id,
                started_at=datetime.now(timezone.utc)
            )
            
            self._start_time = datetime.now(timezone.utc)
            
            self.logger.info(
                f"Starting complete pipeline orchestration",
                orchestration_id=orchestration_id,
                user_id=user_id,
                sources_count=len(source_selections)
            )
            
            # Phase 1: Initialization and Discovery
            await self._execute_discovery_phase(source_selections)
            
            if self._shutdown_requested:
                return await self._handle_shutdown()
            
            # Phase 2: Multi-Source Ingestion
            await self._execute_ingestion_phase(processing_options)
            
            if self._shutdown_requested:
                return await self._handle_shutdown()
            
            # Phase 3: Quality Audit and Coverage Analysis
            if self.config.enable_quality_audit:
                await self._execute_audit_phase()
            
            # Phase 4: Completion and Reporting
            await self._execute_completion_phase()
            
            self.logger.info(
                f"Pipeline orchestration completed successfully",
                orchestration_id=orchestration_id,
                total_duration_seconds=(datetime.now(timezone.utc) - self._start_time).total_seconds(),
                total_files_processed=self._orchestration_progress.processed_files
            )
            
            return self._orchestration_progress
            
        except Exception as e:
            self.logger.error(f"Pipeline orchestration failed: {e}", exc_info=True)
            
            if self._orchestration_progress:
                async with self._progress_lock:
                    self._orchestration_progress.overall_status = OrchestrationStatus.FAILED
                    self._orchestration_progress.last_error = str(e)
                    self._orchestration_progress.error_count += 1
            
            raise
    
    async def _execute_discovery_phase(self, source_selections: List[Dict[str, Any]]):
        """Execute the discovery phase across all selected sources"""
        try:
            self.logger.info("Starting discovery phase")
            
            async with self._progress_lock:
                self._orchestration_progress.current_phase = OrchestrationPhase.DISCOVERY
                self._orchestration_progress.overall_status = OrchestrationStatus.RUNNING
                self._orchestration_progress.phase_start_times[OrchestrationPhase.DISCOVERY] = datetime.now(timezone.utc)
            
            # Initialize source states
            for selection in source_selections:
                source_type = DataSourceType(selection['source_type'])
                connection_id = selection['connection_id']
                source_key = f"{source_type.value}:{connection_id}"
                
                source_state = SourceOrchestrationState(
                    source_type=source_type,
                    connection_id=connection_id,
                    source_name=selection.get('source_name', source_type.value),
                    current_phase=OrchestrationPhase.DISCOVERY,
                    phase_status=OrchestrationStatus.RUNNING
                )
                
                async with self._progress_lock:
                    self._orchestration_progress.source_states[source_key] = source_state
                    self._orchestration_progress.total_sources += 1
                    self._orchestration_progress.active_sources += 1
            
            # Discover files from all sources concurrently
            discovery_tasks = []
            for selection in source_selections:
                task = asyncio.create_task(
                    self._discover_source_files(selection)
                )
                discovery_tasks.append(task)
            
            # Execute discovery with concurrency control
            semaphore = asyncio.Semaphore(self.config.max_concurrent_sources)
            
            async def discover_with_semaphore(task):
                async with semaphore:
                    return await task
            
            discovery_results = await asyncio.gather(
                *[discover_with_semaphore(task) for task in discovery_tasks],
                return_exceptions=True
            )
            
            # Process discovery results
            total_discovered = 0
            for i, result in enumerate(discovery_results):
                selection = source_selections[i]
                source_type = DataSourceType(selection['source_type'])
                connection_id = selection['connection_id']
                source_key = f"{source_type.value}:{connection_id}"
                
                if isinstance(result, Exception):
                    self.logger.error(f"Discovery failed for {source_key}: {result}")
                    async with self._progress_lock:
                        source_state = self._orchestration_progress.source_states[source_key]
                        source_state.phase_status = OrchestrationStatus.FAILED
                        source_state.error_message = str(result)
                        self._orchestration_progress.error_count += 1
                        self._orchestration_progress.active_sources -= 1
                else:
                    discovered_files = result or []
                    file_count = len(discovered_files)
                    total_discovered += file_count
                    
                    async with self._progress_lock:
                        source_state = self._orchestration_progress.source_states[source_key]
                        source_state.discovered_files = file_count
                        source_state.total_files = file_count
                        source_state.phase_status = OrchestrationStatus.COMPLETED
            
            # Update overall progress
            async with self._progress_lock:
                self._orchestration_progress.discovered_files = total_discovered
                self._orchestration_progress.total_files = total_discovered
                
                # Calculate phase duration
                phase_duration = (datetime.now(timezone.utc) - 
                                self._orchestration_progress.phase_start_times[OrchestrationPhase.DISCOVERY]).total_seconds()
                self._orchestration_progress.phase_durations[OrchestrationPhase.DISCOVERY] = phase_duration
            
            self.logger.info(
                f"Discovery phase completed",
                total_files_discovered=total_discovered,
                active_sources=self._orchestration_progress.active_sources,
                duration_seconds=phase_duration
            )
            
        except Exception as e:
            self.logger.error(f"Discovery phase failed: {e}", exc_info=True)
            async with self._progress_lock:
                self._orchestration_progress.overall_status = OrchestrationStatus.FAILED
                self._orchestration_progress.last_error = f"Discovery phase failed: {e}"
            raise
    
    async def _discover_source_files(self, selection: Dict[str, Any]) -> List[UniversalFileMetadata]:
        """Discover files from a specific source"""
        try:
            source_type = DataSourceType(selection['source_type'])
            connection_id = selection['connection_id']
            
            self.logger.info(f"Discovering files from {source_type.value}:{connection_id}")
            
            # Get files from the browsing service
            files = await self._browsing_service.list_files(
                connection_id=connection_id,
                folder_path=selection.get('folder_path', '/'),
                recursive=selection.get('recursive', True),
                file_filter={'mime_types': ['application/pdf']}
            )
            
            self.logger.info(f"Discovered {len(files)} files from {source_type.value}:{connection_id}")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to discover files from {selection}: {e}")
            raise
    
    async def _execute_ingestion_phase(self, processing_options: Optional[EnhancedProcessingOptions]):
        """Execute the ingestion phase using the enhanced batch manager"""
        try:
            self.logger.info("Starting ingestion phase")
            
            async with self._progress_lock:
                self._orchestration_progress.current_phase = OrchestrationPhase.INGESTION
                self._orchestration_progress.phase_start_times[OrchestrationPhase.INGESTION] = datetime.now(timezone.utc)
            
            # Prepare file selections for batch processing
            file_selections = []
            for source_key, source_state in self._orchestration_progress.source_states.items():
                if source_state.phase_status == OrchestrationStatus.COMPLETED and source_state.discovered_files > 0:
                    # Get discovered files for this source
                    discovered_files = await self._get_discovered_files_for_source(source_state)
                    
                    if discovered_files:
                        file_selection = {
                            'source_type': source_state.source_type.value,
                            'connection_id': source_state.connection_id,
                            'file_ids': [f.file_id for f in discovered_files]
                        }
                        file_selections.append(file_selection)
            
            if not file_selections:
                self.logger.warning("No files available for ingestion")
                return
            
            # Create enhanced batch ingestion job
            job_id = await self._batch_manager.create_enhanced_job(
                user_id=self._orchestration_progress.user_id,
                name=f"Multi-Source Pipeline Orchestration - {self._orchestration_progress.orchestration_id}",
                file_selections=file_selections,
                processing_options=processing_options,
                description="Automated multi-source pipeline orchestration job"
            )
            
            self.logger.info(f"Created batch ingestion job: {job_id}")
            
            # Monitor job progress
            await self._monitor_ingestion_job(job_id)
            
            # Update orchestration progress with ingestion results
            await self._update_progress_from_job(job_id)
            
            # Calculate phase duration
            async with self._progress_lock:
                phase_duration = (datetime.now(timezone.utc) - 
                                self._orchestration_progress.phase_start_times[OrchestrationPhase.INGESTION]).total_seconds()
                self._orchestration_progress.phase_durations[OrchestrationPhase.INGESTION] = phase_duration
            
            self.logger.info(
                f"Ingestion phase completed",
                job_id=job_id,
                processed_files=self._orchestration_progress.processed_files,
                duration_seconds=phase_duration
            )
            
        except Exception as e:
            self.logger.error(f"Ingestion phase failed: {e}", exc_info=True)
            async with self._progress_lock:
                self._orchestration_progress.overall_status = OrchestrationStatus.FAILED
                self._orchestration_progress.last_error = f"Ingestion phase failed: {e}"
            raise
    
    async def _get_discovered_files_for_source(self, source_state: SourceOrchestrationState) -> List[UniversalFileMetadata]:
        """Get discovered files for a specific source"""
        try:
            # This would typically retrieve files from the browsing service cache
            # or re-query the source if needed
            files = await self._browsing_service.list_files(
                connection_id=source_state.connection_id,
                folder_path='/',
                recursive=True,
                file_filter={'mime_types': ['application/pdf']}
            )
            return files
        except Exception as e:
            self.logger.error(f"Failed to get files for source {source_state.source_type}: {e}")
            return []
    
    async def _monitor_ingestion_job(self, job_id: str):
        """Monitor the progress of an ingestion job"""
        try:
            self.logger.info(f"Monitoring ingestion job: {job_id}")
            
            while not self._shutdown_requested:
                # Get job status
                job = await self._batch_manager.get_job_status(job_id)
                if not job:
                    break
                
                # Update orchestration progress
                await self._update_progress_from_job_status(job)
                
                # Check if job is complete
                if job.status in ['completed', 'failed', 'cancelled']:
                    break
                
                # Wait before next check
                await asyncio.sleep(self.config.progress_update_interval_seconds)
            
        except Exception as e:
            self.logger.error(f"Error monitoring ingestion job {job_id}: {e}")
    
    async def _update_progress_from_job_status(self, job):
        """Update orchestration progress from job status"""
        try:
            async with self._progress_lock:
                # Update file counts
                self._orchestration_progress.processed_files = job.completed_files
                self._orchestration_progress.failed_files = job.failed_files
                self._orchestration_progress.skipped_files = job.skipped_files
                
                # Update source-specific progress
                for source_progress in job.source_progress:
                    source_key = f"{source_progress.source_type.value}:{source_progress.connection_id}"
                    if source_key in self._orchestration_progress.source_states:
                        source_state = self._orchestration_progress.source_states[source_key]
                        source_state.processed_files = source_progress.completed_files
                        source_state.failed_files = source_progress.failed_files
                        source_state.skipped_files = source_progress.skipped_files
                        source_state.current_phase = OrchestrationPhase.INGESTION
                        
                        if source_progress.completed_files + source_progress.failed_files + source_progress.skipped_files >= source_progress.total_files:
                            source_state.phase_status = OrchestrationStatus.COMPLETED
                        else:
                            source_state.phase_status = OrchestrationStatus.RUNNING
                
                # Calculate throughput
                if self._start_time:
                    elapsed_seconds = (datetime.now(timezone.utc) - self._start_time).total_seconds()
                    if elapsed_seconds > 0:
                        self._orchestration_progress.throughput_files_per_second = self._orchestration_progress.processed_files / elapsed_seconds
                
                self._orchestration_progress.last_updated = datetime.now(timezone.utc)
                
        except Exception as e:
            self.logger.error(f"Error updating progress from job status: {e}")
    
    async def _update_progress_from_job(self, job_id: str):
        """Update final progress from completed job"""
        try:
            job = await self._batch_manager.get_job_status(job_id)
            if job:
                await self._update_progress_from_job_status(job)
        except Exception as e:
            self.logger.error(f"Error updating progress from job {job_id}: {e}")
    
    async def _execute_audit_phase(self):
        """Execute the quality audit and coverage analysis phase"""
        try:
            self.logger.info("Starting audit phase")
            
            async with self._progress_lock:
                self._orchestration_progress.current_phase = OrchestrationPhase.AUDIT
                self._orchestration_progress.phase_start_times[OrchestrationPhase.AUDIT] = datetime.now(timezone.utc)
            
            # Run quality audit
            if self.config.enable_quality_audit:
                self.logger.info("Running quality audit")
                audit_results = await self._quality_audit_service.run_comprehensive_audit()
                
                # Store audit results
                self._phase_metrics[OrchestrationPhase.AUDIT] = {
                    'quality_audit': audit_results
                }
            
            # Run coverage analysis
            if self.config.enable_coverage_analysis:
                self.logger.info("Running coverage analysis")
                coverage_results = await self._coverage_service.analyze_coverage()
                
                # Update phase metrics
                if OrchestrationPhase.AUDIT not in self._phase_metrics:
                    self._phase_metrics[OrchestrationPhase.AUDIT] = {}
                self._phase_metrics[OrchestrationPhase.AUDIT]['coverage_analysis'] = coverage_results
            
            # Generate readiness memo
            if self.config.generate_readiness_memo:
                self.logger.info("Generating knowledge readiness memo")
                readiness_memo = await self._readiness_service.generate_comprehensive_memo(
                    audit_results=self._phase_metrics.get(OrchestrationPhase.AUDIT, {}).get('quality_audit'),
                    coverage_results=self._phase_metrics.get(OrchestrationPhase.AUDIT, {}).get('coverage_analysis')
                )
                
                self._phase_metrics[OrchestrationPhase.AUDIT]['readiness_memo'] = readiness_memo
            
            # Calculate phase duration
            async with self._progress_lock:
                phase_duration = (datetime.now(timezone.utc) - 
                                self._orchestration_progress.phase_start_times[OrchestrationPhase.AUDIT]).total_seconds()
                self._orchestration_progress.phase_durations[OrchestrationPhase.AUDIT] = phase_duration
            
            self.logger.info(f"Audit phase completed in {phase_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Audit phase failed: {e}", exc_info=True)
            async with self._progress_lock:
                self._orchestration_progress.warning_count += 1
            # Don't fail the entire pipeline for audit issues
    
    async def _execute_completion_phase(self):
        """Execute the completion phase with final reporting"""
        try:
            self.logger.info("Starting completion phase")
            
            async with self._progress_lock:
                self._orchestration_progress.current_phase = OrchestrationPhase.COMPLETION
                self._orchestration_progress.overall_status = OrchestrationStatus.COMPLETED
                self._orchestration_progress.phase_start_times[OrchestrationPhase.COMPLETION] = datetime.now(timezone.utc)
            
            # Generate final summary
            summary = await self._generate_final_summary()
            
            # Log completion metrics
            total_duration = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            
            self.logger.info(
                "Pipeline orchestration completed",
                orchestration_id=self._orchestration_progress.orchestration_id,
                total_duration_seconds=total_duration,
                total_sources=self._orchestration_progress.total_sources,
                total_files=self._orchestration_progress.total_files,
                processed_files=self._orchestration_progress.processed_files,
                failed_files=self._orchestration_progress.failed_files,
                skipped_files=self._orchestration_progress.skipped_files,
                throughput_fps=self._orchestration_progress.throughput_files_per_second,
                error_count=self._orchestration_progress.error_count,
                warning_count=self._orchestration_progress.warning_count
            )
            
            # Store final metrics
            self._phase_metrics[OrchestrationPhase.COMPLETION] = {
                'summary': summary,
                'total_duration_seconds': total_duration,
                'final_status': self._orchestration_progress.overall_status.value
            }
            
        except Exception as e:
            self.logger.error(f"Completion phase failed: {e}", exc_info=True)
            async with self._progress_lock:
                self._orchestration_progress.overall_status = OrchestrationStatus.FAILED
                self._orchestration_progress.last_error = f"Completion phase failed: {e}"
    
    async def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate final orchestration summary"""
        try:
            summary = {
                'orchestration_id': self._orchestration_progress.orchestration_id,
                'user_id': self._orchestration_progress.user_id,
                'started_at': self._orchestration_progress.started_at.isoformat(),
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'total_duration_seconds': (datetime.now(timezone.utc) - self._start_time).total_seconds(),
                
                # Source summary
                'sources': {
                    'total_sources': self._orchestration_progress.total_sources,
                    'successful_sources': sum(1 for s in self._orchestration_progress.source_states.values() 
                                            if s.phase_status == OrchestrationStatus.COMPLETED),
                    'failed_sources': sum(1 for s in self._orchestration_progress.source_states.values() 
                                        if s.phase_status == OrchestrationStatus.FAILED),
                    'source_details': {
                        key: {
                            'source_type': state.source_type.value,
                            'source_name': state.source_name,
                            'total_files': state.total_files,
                            'processed_files': state.processed_files,
                            'failed_files': state.failed_files,
                            'skipped_files': state.skipped_files,
                            'status': state.phase_status.value,
                            'error_message': state.error_message
                        }
                        for key, state in self._orchestration_progress.source_states.items()
                    }
                },
                
                # File processing summary
                'files': {
                    'total_files': self._orchestration_progress.total_files,
                    'discovered_files': self._orchestration_progress.discovered_files,
                    'processed_files': self._orchestration_progress.processed_files,
                    'failed_files': self._orchestration_progress.failed_files,
                    'skipped_files': self._orchestration_progress.skipped_files,
                    'success_rate': (self._orchestration_progress.processed_files / 
                                   max(1, self._orchestration_progress.total_files)) * 100
                },
                
                # Performance metrics
                'performance': {
                    'throughput_files_per_second': self._orchestration_progress.throughput_files_per_second,
                    'peak_memory_usage_mb': self._orchestration_progress.peak_memory_usage_mb,
                    'average_cpu_usage': self._orchestration_progress.average_cpu_usage,
                    'phase_durations': {
                        phase.value: duration 
                        for phase, duration in self._orchestration_progress.phase_durations.items()
                    }
                },
                
                # Error and quality metrics
                'quality': {
                    'error_count': self._orchestration_progress.error_count,
                    'warning_count': self._orchestration_progress.warning_count,
                    'last_error': self._orchestration_progress.last_error,
                    'audit_results': self._phase_metrics.get(OrchestrationPhase.AUDIT, {})
                },
                
                # Configuration used
                'configuration': {
                    'max_concurrent_sources': self.config.max_concurrent_sources,
                    'max_concurrent_files_per_source': self.config.max_concurrent_files_per_source,
                    'enable_quality_audit': self.config.enable_quality_audit,
                    'enable_coverage_analysis': self.config.enable_coverage_analysis
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating final summary: {e}")
            return {'error': str(e)}
    
    async def _handle_shutdown(self) -> OrchestrationProgress:
        """Handle graceful shutdown"""
        self.logger.info("Handling graceful shutdown")
        
        async with self._progress_lock:
            self._orchestration_progress.overall_status = OrchestrationStatus.CANCELLED
            self._orchestration_progress.last_updated = datetime.now(timezone.utc)
        
        return self._orchestration_progress
    
    async def _progress_reporter(self):
        """Background task for progress reporting"""
        while not self._shutdown_requested:
            try:
                if self._orchestration_progress:
                    # Update resource metrics
                    await self._update_resource_metrics()
                    
                    # Log progress if detailed logging is enabled
                    if self.config.enable_detailed_logging:
                        await self._log_detailed_progress()
                
                await asyncio.sleep(self.config.progress_update_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in progress reporter: {e}")
                await asyncio.sleep(self.config.progress_update_interval_seconds)
    
    async def _update_resource_metrics(self):
        """Update resource utilization metrics"""
        try:
            if self._performance_optimizer:
                global_metrics = await self._performance_optimizer.get_global_metrics()
                
                async with self._progress_lock:
                    self._orchestration_progress.peak_memory_usage_mb = max(
                        self._orchestration_progress.peak_memory_usage_mb,
                        global_metrics.memory_usage_mb
                    )
                    self._orchestration_progress.average_cpu_usage = global_metrics.resource_utilization.get('cpu', 0)
                    
        except Exception as e:
            self.logger.error(f"Error updating resource metrics: {e}")
    
    async def _log_detailed_progress(self):
        """Log detailed progress information"""
        try:
            async with self._progress_lock:
                progress = self._orchestration_progress
                
                self.logger.info(
                    "Orchestration progress update",
                    orchestration_id=progress.orchestration_id,
                    current_phase=progress.current_phase.value,
                    overall_status=progress.overall_status.value,
                    total_files=progress.total_files,
                    processed_files=progress.processed_files,
                    failed_files=progress.failed_files,
                    throughput_fps=progress.throughput_files_per_second,
                    active_sources=progress.active_sources,
                    memory_usage_mb=progress.peak_memory_usage_mb,
                    cpu_usage=progress.average_cpu_usage
                )
                
        except Exception as e:
            self.logger.error(f"Error logging detailed progress: {e}")
    
    async def _metrics_collector(self):
        """Background task for metrics collection"""
        while not self._shutdown_requested:
            try:
                # Collect metrics from all services
                if self._batch_manager:
                    batch_metrics = await self._batch_manager.get_enhanced_metrics()
                    # Process batch metrics
                
                if self._performance_optimizer:
                    perf_metrics = await self._performance_optimizer.get_global_metrics()
                    # Process performance metrics
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(10)
    
    async def _websocket_broadcaster(self):
        """Background task for WebSocket broadcasting"""
        while not self._shutdown_requested:
            try:
                if self._orchestration_progress and self.config.enable_websocket_updates:
                    # Broadcast progress to connected WebSocket clients
                    message = {
                        'type': 'orchestration_progress',
                        'data': {
                            'orchestration_id': self._orchestration_progress.orchestration_id,
                            'current_phase': self._orchestration_progress.current_phase.value,
                            'overall_status': self._orchestration_progress.overall_status.value,
                            'total_files': self._orchestration_progress.total_files,
                            'processed_files': self._orchestration_progress.processed_files,
                            'failed_files': self._orchestration_progress.failed_files,
                            'throughput_fps': self._orchestration_progress.throughput_files_per_second,
                            'source_states': {
                                key: {
                                    'source_type': state.source_type.value,
                                    'source_name': state.source_name,
                                    'processed_files': state.processed_files,
                                    'total_files': state.total_files,
                                    'status': state.phase_status.value
                                }
                                for key, state in self._orchestration_progress.source_states.items()
                            }
                        },
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    
                    await self._broadcast_websocket_message(message)
                
                await asyncio.sleep(self.config.progress_update_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket broadcaster: {e}")
                await asyncio.sleep(self.config.progress_update_interval_seconds)
    
    async def _broadcast_websocket_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        try:
            async with self._websocket_lock:
                user_id = self._orchestration_progress.user_id if self._orchestration_progress else None
                if user_id and user_id in self._websocket_connections:
                    connections = self._websocket_connections[user_id]
                    
                    # Send to all connections for this user
                    for connection in connections[:]:  # Copy to avoid modification during iteration
                        try:
                            await connection.send_json(message)
                        except Exception:
                            # Remove dead connection
                            connections.remove(connection)
                            
        except Exception as e:
            self.logger.error(f"Error broadcasting WebSocket message: {e}")
    
    async def get_orchestration_progress(self) -> Optional[OrchestrationProgress]:
        """Get current orchestration progress"""
        async with self._progress_lock:
            return self._orchestration_progress
    
    async def add_websocket_connection(self, user_id: str, websocket):
        """Add WebSocket connection for progress updates"""
        async with self._websocket_lock:
            if user_id not in self._websocket_connections:
                self._websocket_connections[user_id] = []
            self._websocket_connections[user_id].append(websocket)
    
    async def remove_websocket_connection(self, user_id: str, websocket):
        """Remove WebSocket connection"""
        async with self._websocket_lock:
            if user_id in self._websocket_connections:
                try:
                    self._websocket_connections[user_id].remove(websocket)
                    if not self._websocket_connections[user_id]:
                        del self._websocket_connections[user_id]
                except ValueError:
                    pass
    
    async def shutdown(self):
        """Shutdown the orchestrator and all services"""
        self.logger.info("Shutting down multi-source pipeline orchestrator")
        
        self._shutdown_requested = True
        self._shutdown_event.set()
        
        # Cancel background tasks
        tasks_to_cancel = [
            self._progress_reporter_task,
            self._metrics_collector_task,
            self._websocket_broadcaster_task
        ]
        
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if tasks_to_cancel:
            await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        
        # Shutdown services
        if self._batch_manager:
            await self._batch_manager.shutdown()
        
        if self._performance_optimizer:
            await self._performance_optimizer.shutdown()
        
        if self._embedding_service:
            await self._embedding_service.shutdown()
        
        if self._database_service:
            await self._database_service.shutdown()
        
        # Close WebSocket connections
        async with self._websocket_lock:
            for connections in self._websocket_connections.values():
                for connection in connections:
                    try:
                        await connection.close()
                    except Exception:
                        pass
            self._websocket_connections.clear()
        
        self.logger.info("Multi-source pipeline orchestrator shutdown complete")


# Global orchestrator instance
_orchestrator: Optional[MultiSourcePipelineOrchestrator] = None


async def get_orchestrator(config: Optional[OrchestrationConfig] = None) -> MultiSourcePipelineOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = MultiSourcePipelineOrchestrator(config)
        await _orchestrator.initialize()
    
    return _orchestrator


async def main():
    """Main entry point for standalone orchestration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Source Knowledge Base Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --user-id user123 --sources google_drive:conn1,local_zip:conn2
  %(prog)s --config orchestration_config.json --enable-audit
  %(prog)s --user-id user123 --sources google_drive:conn1 --max-concurrent-sources 2
        """
    )
    
    parser.add_argument(
        "--user-id",
        required=True,
        help="User identifier for the orchestration session"
    )
    
    parser.add_argument(
        "--sources",
        required=True,
        help="Comma-separated list of source_type:connection_id pairs"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to orchestration configuration file"
    )
    
    parser.add_argument(
        "--max-concurrent-sources",
        type=int,
        default=4,
        help="Maximum concurrent sources to process"
    )
    
    parser.add_argument(
        "--max-concurrent-files",
        type=int,
        default=16,
        help="Maximum concurrent files per source"
    )
    
    parser.add_argument(
        "--enable-audit",
        action="store_true",
        help="Enable quality audit and coverage analysis"
    )
    
    parser.add_argument(
        "--disable-websocket",
        action="store_true",
        help="Disable WebSocket progress updates"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    try:
        # Configure logging
        configure_logging()
        logger = get_logger(__name__)
        
        logger.info("Starting multi-source pipeline orchestration")
        
        # Parse source selections
        source_selections = []
        for source_spec in args.sources.split(','):
            source_type_str, connection_id = source_spec.strip().split(':')
            source_selections.append({
                'source_type': source_type_str,
                'connection_id': connection_id,
                'source_name': f"{source_type_str}_{connection_id}",
                'recursive': True
            })
        
        # Load configuration
        config = OrchestrationConfig()
        if args.config and args.config.exists():
            with open(args.config) as f:
                config_data = json.load(f)
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # Override with command line arguments
        config.max_concurrent_sources = args.max_concurrent_sources
        config.max_concurrent_files_per_source = args.max_concurrent_files
        config.enable_quality_audit = args.enable_audit
        config.enable_coverage_analysis = args.enable_audit
        config.enable_websocket_updates = not args.disable_websocket
        
        # Create and initialize orchestrator
        orchestrator = await get_orchestrator(config)
        
        # Create processing options
        processing_options = EnhancedProcessingOptions(
            max_concurrent_files=args.max_concurrent_files,
            enable_async_processing=True,
            enable_gpu_acceleration=True,
            enable_adaptive_batching=True
        )
        
        # Run orchestration
        progress = await orchestrator.orchestrate_complete_pipeline(
            user_id=args.user_id,
            source_selections=source_selections,
            processing_options=processing_options
        )
        
        # Print final summary
        if progress.overall_status == OrchestrationStatus.COMPLETED:
            logger.info("Pipeline orchestration completed successfully")
            print(f"\n=== ORCHESTRATION COMPLETED ===")
            print(f"Orchestration ID: {progress.orchestration_id}")
            print(f"Total Sources: {progress.total_sources}")
            print(f"Total Files: {progress.total_files}")
            print(f"Processed Files: {progress.processed_files}")
            print(f"Failed Files: {progress.failed_files}")
            print(f"Success Rate: {(progress.processed_files / max(1, progress.total_files)) * 100:.1f}%")
            print(f"Throughput: {progress.throughput_files_per_second:.2f} files/second")
            
            if progress.phase_durations:
                print(f"\nPhase Durations:")
                for phase, duration in progress.phase_durations.items():
                    print(f"  {phase.value}: {duration:.2f} seconds")
            
            return 0
        else:
            logger.error("Pipeline orchestration failed")
            print(f"\n=== ORCHESTRATION FAILED ===")
            print(f"Status: {progress.overall_status.value}")
            print(f"Error: {progress.last_error}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Orchestration interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        if _orchestrator:
            await _orchestrator.shutdown()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))