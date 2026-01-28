"""
Orchestration Monitoring and Logging Service

This module provides comprehensive monitoring, logging, and status reporting
for the multi-source pipeline orchestration system with source attribution,
performance metrics, and real-time updates.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
import psutil
import threading
from collections import defaultdict, deque

from core.logging import get_logger, set_correlation_id
from core.config import get_settings
from services.multi_source_auth import DataSourceType
from config.orchestration_config import (
    OrchestrationConfiguration, MonitoringConfiguration, ProgressReportingLevel
)


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    source_attribution: Optional[str] = None


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: Optional[str] = None
    source_type: Optional[DataSourceType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    active_tasks: int
    
    # Orchestration-specific metrics
    active_sources: int = 0
    total_files_processing: int = 0
    throughput_files_per_second: float = 0.0
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    
    # GPU metrics (if available)
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None


@dataclass
class SourceMetrics:
    """Metrics for a specific data source"""
    source_type: DataSourceType
    connection_id: str
    source_name: str
    
    # File processing metrics
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    # Performance metrics
    average_processing_time_seconds: float = 0.0
    throughput_files_per_second: float = 0.0
    error_rate_percent: float = 0.0
    
    # Resource usage
    cpu_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    network_bytes_transferred: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    # Error tracking
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None


class OrchestrationMonitor:
    """
    Comprehensive monitoring service for multi-source pipeline orchestration.
    Provides metrics collection, performance monitoring, alerting, and logging
    with source attribution and real-time updates.
    """
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._metrics_lock = threading.Lock()
        
        # Performance snapshots
        self._performance_history: deque = deque(maxlen=1000)
        self._performance_lock = threading.Lock()
        
        # Source metrics
        self._source_metrics: Dict[str, SourceMetrics] = {}
        self._source_metrics_lock = threading.Lock()
        
        # Alerts
        self._alerts: Dict[str, Alert] = {}
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._alerts_lock = threading.Lock()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_collector_task: Optional[asyncio.Task] = None
        self._alert_processor_task: Optional[asyncio.Task] = None
        
        # State
        self._monitoring_active = False
        self._shutdown_requested = False
        
        # Correlation tracking
        self._correlation_id = set_correlation_id()
        
        # Performance baselines
        self._baseline_metrics: Dict[str, float] = {}
        self._performance_thresholds: Dict[str, float] = {
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'error_rate_percent': 5.0,
            'throughput_degradation_percent': 50.0
        }
    
    async def start_monitoring(self):
        """Start the monitoring service"""
        try:
            if self._monitoring_active:
                return
            
            self.logger.info("Starting orchestration monitoring service")
            
            self._monitoring_active = True
            self._shutdown_requested = False
            
            # Start background tasks
            if self.config.enable_metrics_collection:
                self._metrics_collector_task = asyncio.create_task(self._metrics_collector())
            
            self._monitoring_task = asyncio.create_task(self._performance_monitor())
            self._alert_processor_task = asyncio.create_task(self._alert_processor())
            
            # Record startup metrics
            await self._record_startup_metrics()
            
            self.logger.info("Orchestration monitoring service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring service: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop the monitoring service"""
        try:
            self.logger.info("Stopping orchestration monitoring service")
            
            self._shutdown_requested = True
            self._monitoring_active = False
            
            # Cancel background tasks
            tasks = [
                self._monitoring_task,
                self._metrics_collector_task,
                self._alert_processor_task
            ]
            
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if tasks:
                await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
            
            # Record shutdown metrics
            await self._record_shutdown_metrics()
            
            self.logger.info("Orchestration monitoring service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring service: {e}")
    
    async def _record_startup_metrics(self):
        """Record metrics at startup"""
        await self.record_metric("orchestration.startup", 1, MetricType.COUNTER)
        await self.record_metric("orchestration.active", 1, MetricType.GAUGE)
    
    async def _record_shutdown_metrics(self):
        """Record metrics at shutdown"""
        await self.record_metric("orchestration.shutdown", 1, MetricType.COUNTER)
        await self.record_metric("orchestration.active", 0, MetricType.GAUGE)
    
    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        source_attribution: Optional[str] = None
    ):
        """Record a metric point"""
        try:
            metric_point = MetricPoint(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(timezone.utc),
                tags=tags or {},
                source_attribution=source_attribution
            )
            
            with self._metrics_lock:
                self._metrics[name].append(metric_point)
            
            # Log metric if detailed logging is enabled
            if self.config.enable_detailed_logging:
                self.logger.debug(
                    f"Recorded metric: {name}={value}",
                    metric_type=metric_type.value,
                    tags=tags,
                    source=source_attribution
                )
                
        except Exception as e:
            self.logger.error(f"Error recording metric {name}: {e}")
    
    async def record_source_metric(
        self,
        source_key: str,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a metric for a specific source"""
        tags = {"source": source_key}
        await self.record_metric(
            f"source.{metric_name}",
            value,
            metric_type,
            tags=tags,
            source_attribution=source_key
        )
    
    async def update_source_metrics(self, source_key: str, metrics_update: Dict[str, Any]):
        """Update metrics for a specific source"""
        try:
            with self._source_metrics_lock:
                if source_key not in self._source_metrics:
                    # Initialize source metrics if not exists
                    parts = source_key.split(':', 1)
                    if len(parts) == 2:
                        source_type_str, connection_id = parts
                        self._source_metrics[source_key] = SourceMetrics(
                            source_type=DataSourceType(source_type_str),
                            connection_id=connection_id,
                            source_name=metrics_update.get('source_name', source_key)
                        )
                
                source_metrics = self._source_metrics[source_key]
                
                # Update metrics
                for key, value in metrics_update.items():
                    if hasattr(source_metrics, key):
                        setattr(source_metrics, key, value)
                
                source_metrics.last_activity = datetime.now(timezone.utc)
            
            # Record individual metrics
            for key, value in metrics_update.items():
                if isinstance(value, (int, float)):
                    await self.record_source_metric(source_key, key, float(value))
                    
        except Exception as e:
            self.logger.error(f"Error updating source metrics for {source_key}: {e}")
    
    async def _performance_monitor(self):
        """Background performance monitoring"""
        while not self._shutdown_requested:
            try:
                # Collect system performance metrics
                snapshot = await self._collect_performance_snapshot()
                
                with self._performance_lock:
                    self._performance_history.append(snapshot)
                
                # Record performance metrics
                await self._record_performance_metrics(snapshot)
                
                # Check for performance alerts
                await self._check_performance_alerts(snapshot)
                
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(self.config.health_check_interval_seconds)
    
    async def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_info = process.as_dict(['num_threads'])
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / 1024 / 1024,
                memory_usage_percent=memory.percent,
                disk_io_read_mb=(disk_io.read_bytes / 1024 / 1024) if disk_io else 0,
                disk_io_write_mb=(disk_io.write_bytes / 1024 / 1024) if disk_io else 0,
                network_sent_mb=(network_io.bytes_sent / 1024 / 1024) if network_io else 0,
                network_recv_mb=(network_io.bytes_recv / 1024 / 1024) if network_io else 0,
                active_threads=process_info.get('num_threads', 0),
                active_tasks=len([t for t in asyncio.all_tasks() if not t.done()])
            )
            
            # Add orchestration-specific metrics
            with self._source_metrics_lock:
                snapshot.active_sources = len([
                    s for s in self._source_metrics.values()
                    if s.started_at and not s.completed_at
                ])
                snapshot.total_files_processing = sum(
                    s.total_files - s.processed_files - s.failed_files - s.skipped_files
                    for s in self._source_metrics.values()
                )
            
            # Try to get GPU metrics
            try:
                snapshot.gpu_utilization_percent = await self._get_gpu_utilization()
                snapshot.gpu_memory_usage_mb = await self._get_gpu_memory_usage()
            except Exception:
                pass  # GPU metrics not available
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error collecting performance snapshot: {e}")
            # Return minimal snapshot
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=0,
                memory_usage_mb=0,
                memory_usage_percent=0,
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                network_sent_mb=0,
                network_recv_mb=0,
                active_threads=0,
                active_tasks=0
            )
    
    async def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except Exception:
            return None
    
    async def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(memory_info.used / 1024 / 1024)  # Convert to MB
        except Exception:
            return None
    
    async def _record_performance_metrics(self, snapshot: PerformanceSnapshot):
        """Record performance metrics"""
        metrics = [
            ("system.cpu_usage_percent", snapshot.cpu_usage_percent),
            ("system.memory_usage_mb", snapshot.memory_usage_mb),
            ("system.memory_usage_percent", snapshot.memory_usage_percent),
            ("system.disk_io_read_mb", snapshot.disk_io_read_mb),
            ("system.disk_io_write_mb", snapshot.disk_io_write_mb),
            ("system.network_sent_mb", snapshot.network_sent_mb),
            ("system.network_recv_mb", snapshot.network_recv_mb),
            ("orchestration.active_threads", snapshot.active_threads),
            ("orchestration.active_tasks", snapshot.active_tasks),
            ("orchestration.active_sources", snapshot.active_sources),
            ("orchestration.files_processing", snapshot.total_files_processing),
        ]
        
        for name, value in metrics:
            await self.record_metric(name, value, MetricType.GAUGE)
        
        # GPU metrics if available
        if snapshot.gpu_utilization_percent is not None:
            await self.record_metric("gpu.utilization_percent", snapshot.gpu_utilization_percent, MetricType.GAUGE)
        
        if snapshot.gpu_memory_usage_mb is not None:
            await self.record_metric("gpu.memory_usage_mb", snapshot.gpu_memory_usage_mb, MetricType.GAUGE)
    
    async def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance-related alerts"""
        try:
            # CPU usage alert
            if snapshot.cpu_usage_percent > self._performance_thresholds['cpu_usage_percent']:
                await self.create_alert(
                    AlertLevel.WARNING,
                    "High CPU Usage",
                    f"CPU usage is {snapshot.cpu_usage_percent:.1f}%, exceeding threshold of {self._performance_thresholds['cpu_usage_percent']:.1f}%"
                )
            
            # Memory usage alert
            if snapshot.memory_usage_percent > self._performance_thresholds['memory_usage_percent']:
                await self.create_alert(
                    AlertLevel.WARNING,
                    "High Memory Usage",
                    f"Memory usage is {snapshot.memory_usage_percent:.1f}%, exceeding threshold of {self._performance_thresholds['memory_usage_percent']:.1f}%"
                )
            
            # Check source-specific alerts
            await self._check_source_alerts()
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    async def _check_source_alerts(self):
        """Check for source-specific alerts"""
        try:
            with self._source_metrics_lock:
                for source_key, metrics in self._source_metrics.items():
                    # Error rate alert
                    if metrics.total_files > 0:
                        error_rate = (metrics.failed_files / metrics.total_files) * 100
                        if error_rate > self._performance_thresholds['error_rate_percent']:
                            await self.create_alert(
                                AlertLevel.ERROR,
                                f"High Error Rate - {source_key}",
                                f"Error rate for {source_key} is {error_rate:.1f}%, exceeding threshold",
                                source=source_key,
                                source_type=metrics.source_type
                            )
                    
                    # Stalled processing alert
                    if (metrics.last_activity and 
                        (datetime.now(timezone.utc) - metrics.last_activity).total_seconds() > 300):  # 5 minutes
                        await self.create_alert(
                            AlertLevel.WARNING,
                            f"Processing Stalled - {source_key}",
                            f"No activity detected for {source_key} in the last 5 minutes",
                            source=source_key,
                            source_type=metrics.source_type
                        )
                        
        except Exception as e:
            self.logger.error(f"Error checking source alerts: {e}")
    
    async def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: Optional[str] = None,
        source_type: Optional[DataSourceType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a new alert"""
        try:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                level=level,
                title=title,
                message=message,
                timestamp=datetime.now(timezone.utc),
                source=source,
                source_type=source_type,
                metadata=metadata or {}
            )
            
            with self._alerts_lock:
                self._alerts[alert.alert_id] = alert
            
            # Log alert
            log_level = {
                AlertLevel.INFO: "info",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "error",
                AlertLevel.CRITICAL: "error"
            }.get(level, "info")
            
            getattr(self.logger, log_level)(
                f"Alert: {title}",
                message=message,
                alert_id=alert.alert_id,
                source=source,
                source_type=source_type.value if source_type else None
            )
            
            # Notify alert handlers
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
    
    async def _metrics_collector(self):
        """Background metrics collection"""
        while not self._shutdown_requested:
            try:
                # Collect and aggregate metrics
                await self._collect_aggregated_metrics()
                
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)
    
    async def _collect_aggregated_metrics(self):
        """Collect aggregated metrics"""
        try:
            # Calculate throughput metrics
            with self._source_metrics_lock:
                total_throughput = sum(
                    s.throughput_files_per_second for s in self._source_metrics.values()
                )
                await self.record_metric("orchestration.total_throughput_fps", total_throughput, MetricType.GAUGE)
                
                # Calculate success rates
                total_files = sum(s.total_files for s in self._source_metrics.values())
                total_processed = sum(s.processed_files for s in self._source_metrics.values())
                
                if total_files > 0:
                    success_rate = (total_processed / total_files) * 100
                    await self.record_metric("orchestration.success_rate_percent", success_rate, MetricType.GAUGE)
                    
        except Exception as e:
            self.logger.error(f"Error collecting aggregated metrics: {e}")
    
    async def _alert_processor(self):
        """Background alert processing"""
        while not self._shutdown_requested:
            try:
                # Process and clean up old alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in alert processor: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            with self._alerts_lock:
                alerts_to_remove = [
                    alert_id for alert_id, alert in self._alerts.items()
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                
                for alert_id in alerts_to_remove:
                    del self._alerts[alert_id]
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler"""
        self._alert_handlers.append(handler)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        try:
            with self._metrics_lock:
                latest_metrics = {}
                for name, points in self._metrics.items():
                    if points:
                        latest_metrics[name] = points[-1].value
            
            with self._performance_lock:
                latest_performance = None
                if self._performance_history:
                    latest_performance = asdict(self._performance_history[-1])
            
            with self._source_metrics_lock:
                source_metrics = {
                    key: asdict(metrics) for key, metrics in self._source_metrics.items()
                }
            
            return {
                "metrics": latest_metrics,
                "performance": latest_performance,
                "sources": source_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {}
    
    def get_alerts(self, include_resolved: bool = False) -> List[Alert]:
        """Get current alerts"""
        try:
            with self._alerts_lock:
                if include_resolved:
                    return list(self._alerts.values())
                else:
                    return [alert for alert in self._alerts.values() if not alert.resolved]
                    
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []


# Global monitoring instance
_monitor: Optional[OrchestrationMonitor] = None


def get_orchestration_monitor(config: Optional[MonitoringConfiguration] = None) -> OrchestrationMonitor:
    """Get or create global orchestration monitor"""
    global _monitor
    
    if _monitor is None:
        if config is None:
            from config.orchestration_config import load_orchestration_config
            orchestration_config = load_orchestration_config()
            config = orchestration_config.monitoring
        
        _monitor = OrchestrationMonitor(config)
    
    return _monitor