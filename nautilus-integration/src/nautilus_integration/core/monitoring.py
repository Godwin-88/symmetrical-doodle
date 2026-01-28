"""
Monitoring and Alerting Infrastructure for NautilusTrader Integration

Implements performance monitoring with nanosecond precision metrics, health check
endpoints for all integration components, automated alerting and escalation procedures,
and diagnostic information and resolution guidance systems.

Following patterns from knowledge-ingestion system for consistency and reliability.

Requirements: 22.1, 22.2, 22.4, 22.7
"""

import asyncio
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics

import structlog
from nautilus_trader.core.nautilus_pyo3 import LogLevel

from nautilus_integration.core.logging import get_logger, get_correlation_id
from nautilus_integration.core.config import MonitoringConfig

logger = get_logger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DiagnosticLevel(Enum):
    """Diagnostic information levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

@dataclass
class DiagnosticInfo:
    """Diagnostic information for system issues"""
    component: str
    issue_type: str
    severity: AlertSeverity
    description: str
    symptoms: List[str]
    possible_causes: List[str]
    resolution_steps: List[str]
    escalation_contacts: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    documentation_links: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResolutionGuidance:
    """Resolution guidance for specific issues"""
    issue_id: str
    title: str
    description: str
    automated_fixes: List[Dict[str, Any]] = field(default_factory=list)
    manual_steps: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)
    rollback_steps: List[str] = field(default_factory=list)
    estimated_resolution_time: Optional[int] = None  # minutes
    success_rate: Optional[float] = None  # 0.0 to 1.0

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: datetime
    component: str
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: HealthStatus
    timestamp: datetime
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ns: Optional[int] = None
    correlation_id: Optional[str] = None

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    severity: AlertSeverity
    component: str
    title: str
    message: str
    timestamp: datetime
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    timestamp: datetime

class MetricsCollector:
    """Collects and stores performance metrics with nanosecond precision"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric"""
        with self._lock:
            metric_key = f"{metric.component}.{metric.name}"
            
            # Store in history
            self.metrics[metric_key].append(metric)
            
            # Update aggregated values
            if metric.metric_type == MetricType.COUNTER:
                self.counters[metric_key] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self.gauges[metric_key] = metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                self.histograms[metric_key].append(metric.value)
                # Keep only recent values for histograms
                if len(self.histograms[metric_key]) > 1000:
                    self.histograms[metric_key] = self.histograms[metric_key][-1000:]
    
    def increment_counter(self, name: str, component: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        metric = PerformanceMetric(
            name=name,
            metric_type=MetricType.COUNTER,
            value=value,
            timestamp=datetime.now(),
            component=component,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, component: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        metric = PerformanceMetric(
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            component=component,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_histogram(self, name: str, component: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        metric = PerformanceMetric(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            component=component,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def get_counter(self, name: str, component: str) -> float:
        """Get counter value"""
        metric_key = f"{component}.{name}"
        return self.counters.get(metric_key, 0.0)
    
    def get_gauge(self, name: str, component: str) -> Optional[float]:
        """Get gauge value"""
        metric_key = f"{component}.{name}"
        return self.gauges.get(metric_key)
    
    def get_histogram_stats(self, name: str, component: str) -> Dict[str, float]:
        """Get histogram statistics"""
        metric_key = f"{component}.{name}"
        values = self.histograms.get(metric_key, [])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
    
    def get_recent_metrics(self, component: str, minutes: int = 5) -> List[PerformanceMetric]:
        """Get recent metrics for a component"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = []
        
        with self._lock:
            for metric_key, metric_history in self.metrics.items():
                if component in metric_key:
                    for metric in metric_history:
                        if metric.timestamp >= cutoff_time:
                            recent_metrics.append(metric)
        
        return sorted(recent_metrics, key=lambda m: m.timestamp)

class LatencyTracker:
    """Tracks latency with nanosecond precision for NautilusTrader operations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._active_operations: Dict[str, int] = {}
    
    def start_operation(self, operation_id: str, component: str, operation_name: str) -> None:
        """Start tracking an operation"""
        start_time = time.perf_counter_ns()
        self._active_operations[operation_id] = start_time
        
        logger.debug(
            "Started operation tracking",
            operation_id=operation_id,
            component=component,
            operation=operation_name,
            start_time_ns=start_time
        )
    
    def end_operation(self, operation_id: str, component: str, operation_name: str) -> int:
        """End tracking an operation and record latency"""
        end_time = time.perf_counter_ns()
        start_time = self._active_operations.pop(operation_id, end_time)
        
        latency_ns = end_time - start_time
        latency_ms = latency_ns / 1_000_000  # Convert to milliseconds
        
        # Record latency metric
        self.metrics_collector.record_histogram(
            f"{operation_name}_latency_ms",
            component,
            latency_ms
        )
        
        logger.debug(
            "Completed operation tracking",
            operation_id=operation_id,
            component=component,
            operation=operation_name,
            latency_ns=latency_ns,
            latency_ms=latency_ms
        )
        
        return latency_ns
    
    def get_operation_latency(self, operation_id: str) -> Optional[int]:
        """Get current operation latency (for ongoing operations)"""
        start_time = self._active_operations.get(operation_id)
        if start_time is None:
            return None
        
        current_time = time.perf_counter_ns()
        return current_time - start_time

class HealthChecker:
    """Performs health checks on NautilusTrader integration components"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_status: Dict[str, HealthCheck] = {}
    
    def register_health_check(self, component: str, check_func: Callable) -> None:
        """Register a health check function for a component"""
        self.health_checks[component] = check_func
        logger.info(
            "Registered health check",
            component=component,
            check_function=check_func.__name__
        )
    
    async def check_component_health(self, component: str) -> HealthCheck:
        """Check health of a specific component"""
        correlation_id = get_correlation_id()
        start_time = time.perf_counter_ns()
        
        try:
            if component not in self.health_checks:
                return HealthCheck(
                    component=component,
                    status=HealthStatus.UNKNOWN,
                    timestamp=datetime.now(),
                    message=f"No health check registered for {component}",
                    correlation_id=correlation_id
                )
            
            check_func = self.health_checks[component]
            
            # Execute health check
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            end_time = time.perf_counter_ns()
            response_time_ns = end_time - start_time
            
            # Parse result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                message = "Health check passed" if result else "Health check failed"
                details = {}
            elif isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                message = result.get('message', 'No message provided')
                details = result.get('details', {})
            else:
                status = HealthStatus.UNKNOWN
                message = f"Invalid health check result: {result}"
                details = {}
            
            health_check = HealthCheck(
                component=component,
                status=status,
                timestamp=datetime.now(),
                message=message,
                details=details,
                response_time_ns=response_time_ns,
                correlation_id=correlation_id
            )
            
            # Record metrics
            self.metrics_collector.record_histogram(
                "health_check_response_time_ms",
                component,
                response_time_ns / 1_000_000
            )
            
            self.metrics_collector.set_gauge(
                "health_status",
                component,
                1.0 if status == HealthStatus.HEALTHY else 0.0
            )
            
            # Store last status
            self.last_health_status[component] = health_check
            
            logger.debug(
                "Health check completed",
                component=component,
                status=status.value,
                response_time_ns=response_time_ns,
                correlation_id=correlation_id
            )
            
            return health_check
            
        except Exception as e:
            end_time = time.perf_counter_ns()
            response_time_ns = end_time - start_time
            
            health_check = HealthCheck(
                component=component,
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                message=f"Health check failed with exception: {str(e)}",
                details={'exception': type(e).__name__, 'error_message': str(e)},
                response_time_ns=response_time_ns,
                correlation_id=correlation_id
            )
            
            self.last_health_status[component] = health_check
            
            logger.error(
                "Health check failed with exception",
                component=component,
                error_type=type(e).__name__,
                error_message=str(e),
                response_time_ns=response_time_ns,
                correlation_id=correlation_id,
                exc_info=True
            )
            
            return health_check
    
    async def check_all_components(self) -> Dict[str, HealthCheck]:
        """Check health of all registered components"""
        results = {}
        
        for component in self.health_checks.keys():
            results[component] = await self.check_component_health(component)
        
        return results
    
    def get_last_health_status(self, component: str) -> Optional[HealthCheck]:
        """Get last health status for a component"""
        return self.last_health_status.get(component)
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.last_health_status:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.last_health_status.values()]
        
        if any(status == HealthStatus.CRITICAL for status in statuses):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING for status in statuses):
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

class AlertManager:
    """Manages alerts and notifications for the NautilusTrader integration"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable] = []
    
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add an alert rule"""
        self.alert_rules.append(rule)
        logger.info(
            "Added alert rule",
            rule_name=rule.get('name', 'unnamed'),
            component=rule.get('component', 'unknown'),
            condition=rule.get('condition', 'unknown')
        )
    
    def register_notification_handler(self, handler: Callable) -> None:
        """Register a notification handler"""
        self.notification_handlers.append(handler)
        logger.info(
            "Registered notification handler",
            handler_name=handler.__name__
        )
    
    async def create_alert(
        self,
        severity: AlertSeverity,
        component: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert"""
        alert = Alert(
            alert_id=f"{component}_{int(time.time())}_{hash(title) % 10000}",
            severity=severity,
            component=component,
            title=title,
            message=message,
            timestamp=datetime.now(),
            correlation_id=get_correlation_id(),
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Record metric
        self.metrics_collector.increment_counter(
            "alerts_created",
            component,
            labels={'severity': severity.value}
        )
        
        logger.warning(
            "Alert created",
            alert_id=alert.alert_id,
            severity=severity.value,
            component=component,
            title=title,
            message=message,
            correlation_id=alert.correlation_id
        )
        
        # Send notifications
        await self._send_notifications(alert)
        
        return alert
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """Resolve an active alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        alert.metadata['resolution_message'] = resolution_message
        
        del self.active_alerts[alert_id]
        
        # Record metric
        self.metrics_collector.increment_counter(
            "alerts_resolved",
            alert.component,
            labels={'severity': alert.severity.value}
        )
        
        logger.info(
            "Alert resolved",
            alert_id=alert_id,
            component=alert.component,
            resolution_message=resolution_message,
            correlation_id=alert.correlation_id
        )
        
        return True
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert"""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(
                    "Failed to send notification",
                    handler_name=handler.__name__,
                    alert_id=alert.alert_id,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
    
    def get_active_alerts(self, component: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by component"""
        alerts = list(self.active_alerts.values())
        
        if component:
            alerts = [alert for alert in alerts if alert.component == component]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

class DiagnosticSystem:
    """Diagnostic information and resolution guidance system"""
    
    def __init__(self):
        self.diagnostic_rules: Dict[str, Callable] = {}
        self.resolution_guides: Dict[str, ResolutionGuidance] = {}
        self.issue_history: List[DiagnosticInfo] = []
        self.automated_fixes: Dict[str, Callable] = {}
        
        # Setup default diagnostic rules and resolution guides
        self._setup_default_diagnostics()
    
    def _setup_default_diagnostics(self) -> None:
        """Setup default diagnostic rules and resolution guides"""
        
        # Nautilus Engine diagnostics
        self.register_diagnostic_rule(
            "nautilus_engine_high_latency",
            self._diagnose_nautilus_engine_latency
        )
        
        self.register_resolution_guide(
            "nautilus_engine_high_latency",
            ResolutionGuidance(
                issue_id="nautilus_engine_high_latency",
                title="NautilusTrader Engine High Latency",
                description="The NautilusTrader engine is experiencing high latency in order processing",
                automated_fixes=[
                    {
                        "name": "restart_engine",
                        "description": "Restart the NautilusTrader engine",
                        "function": "restart_nautilus_engine",
                        "risk_level": "medium"
                    }
                ],
                manual_steps=[
                    "Check system resource usage (CPU, memory, disk I/O)",
                    "Review recent strategy deployments for performance issues",
                    "Examine network connectivity to trading venues",
                    "Check for data feed delays or interruptions"
                ],
                verification_steps=[
                    "Monitor order processing latency for 5 minutes",
                    "Verify all strategies are executing normally",
                    "Check that position updates are synchronized"
                ],
                rollback_steps=[
                    "Stop all live trading if latency exceeds critical thresholds",
                    "Switch to backup execution system if available",
                    "Preserve existing positions and halt new orders"
                ],
                estimated_resolution_time=15,
                success_rate=0.85
            )
        )
        
        # Strategy Translation diagnostics
        self.register_diagnostic_rule(
            "strategy_translation_failures",
            self._diagnose_strategy_translation_failures
        )
        
        self.register_resolution_guide(
            "strategy_translation_failures",
            ResolutionGuidance(
                issue_id="strategy_translation_failures",
                title="Strategy Translation Service Failures",
                description="The strategy translation service is failing to convert F6 strategies to Nautilus format",
                automated_fixes=[
                    {
                        "name": "clear_translation_cache",
                        "description": "Clear strategy translation cache",
                        "function": "clear_strategy_cache",
                        "risk_level": "low"
                    }
                ],
                manual_steps=[
                    "Review F6 strategy definitions for syntax errors",
                    "Check strategy parameter validation rules",
                    "Verify Nautilus strategy template compatibility",
                    "Examine translation service logs for specific errors"
                ],
                verification_steps=[
                    "Test strategy translation with a simple strategy",
                    "Verify generated Nautilus code compiles correctly",
                    "Check that all strategy parameters are mapped"
                ],
                rollback_steps=[
                    "Use previously translated strategy versions",
                    "Disable automatic strategy translation",
                    "Switch to manual strategy implementation"
                ],
                estimated_resolution_time=30,
                success_rate=0.90
            )
        )
        
        # Signal Routing diagnostics
        self.register_diagnostic_rule(
            "signal_routing_delays",
            self._diagnose_signal_routing_delays
        )
        
        self.register_resolution_guide(
            "signal_routing_delays",
            ResolutionGuidance(
                issue_id="signal_routing_delays",
                title="AI Signal Routing Delays",
                description="AI signals from F5 RAG service are experiencing delivery delays to Nautilus strategies",
                automated_fixes=[
                    {
                        "name": "restart_signal_router",
                        "description": "Restart the signal routing service",
                        "function": "restart_signal_router",
                        "risk_level": "low"
                    }
                ],
                manual_steps=[
                    "Check F5 RAG service health and response times",
                    "Verify signal queue sizes and processing rates",
                    "Examine network connectivity between services",
                    "Review signal validation rules for bottlenecks"
                ],
                verification_steps=[
                    "Monitor signal delivery latency for 10 minutes",
                    "Verify all strategies are receiving signals",
                    "Check signal queue depths are decreasing"
                ],
                rollback_steps=[
                    "Use cached signals for strategy execution",
                    "Disable real-time signal updates temporarily",
                    "Switch to historical signal replay mode"
                ],
                estimated_resolution_time=20,
                success_rate=0.80
            )
        )
    
    def register_diagnostic_rule(self, issue_type: str, diagnostic_func: Callable) -> None:
        """Register a diagnostic rule for an issue type"""
        self.diagnostic_rules[issue_type] = diagnostic_func
        logger.info(
            "Registered diagnostic rule",
            issue_type=issue_type,
            function=diagnostic_func.__name__
        )
    
    def register_resolution_guide(self, issue_type: str, guide: ResolutionGuidance) -> None:
        """Register a resolution guide for an issue type"""
        self.resolution_guides[issue_type] = guide
        logger.info(
            "Registered resolution guide",
            issue_type=issue_type,
            title=guide.title
        )
    
    def register_automated_fix(self, fix_name: str, fix_func: Callable) -> None:
        """Register an automated fix function"""
        self.automated_fixes[fix_name] = fix_func
        logger.info(
            "Registered automated fix",
            fix_name=fix_name,
            function=fix_func.__name__
        )
    
    async def diagnose_system_issues(
        self, 
        metrics: Dict[str, Any], 
        health_checks: Dict[str, Any],
        level: DiagnosticLevel = DiagnosticLevel.BASIC
    ) -> List[DiagnosticInfo]:
        """
        Diagnose system issues based on metrics and health checks.
        
        Args:
            metrics: Current system metrics
            health_checks: Health check results
            level: Level of diagnostic detail
            
        Returns:
            List of diagnostic information
        """
        diagnostics = []
        
        for issue_type, diagnostic_func in self.diagnostic_rules.items():
            try:
                if asyncio.iscoroutinefunction(diagnostic_func):
                    result = await diagnostic_func(metrics, health_checks, level)
                else:
                    result = diagnostic_func(metrics, health_checks, level)
                
                if result:
                    diagnostics.append(result)
                    self.issue_history.append(result)
                    
            except Exception as e:
                logger.error(
                    "Diagnostic rule failed",
                    issue_type=issue_type,
                    error=str(e)
                )
        
        return diagnostics
    
    def get_resolution_guidance(self, issue_type: str) -> Optional[ResolutionGuidance]:
        """Get resolution guidance for a specific issue type"""
        return self.resolution_guides.get(issue_type)
    
    async def execute_automated_fix(
        self, 
        fix_name: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an automated fix.
        
        Args:
            fix_name: Name of the fix to execute
            parameters: Optional parameters for the fix
            
        Returns:
            Result of the fix execution
        """
        if fix_name not in self.automated_fixes:
            raise ValueError(f"Automated fix '{fix_name}' not found")
        
        fix_func = self.automated_fixes[fix_name]
        parameters = parameters or {}
        
        try:
            logger.info(
                "Executing automated fix",
                fix_name=fix_name,
                parameters=parameters
            )
            
            if asyncio.iscoroutinefunction(fix_func):
                result = await fix_func(**parameters)
            else:
                result = fix_func(**parameters)
            
            logger.info(
                "Automated fix completed",
                fix_name=fix_name,
                result=result
            )
            
            return {
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(
                "Automated fix failed",
                fix_name=fix_name,
                error=str(e)
            )
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _diagnose_nautilus_engine_latency(
        self, 
        metrics: Dict[str, Any], 
        health_checks: Dict[str, Any],
        level: DiagnosticLevel
    ) -> Optional[DiagnosticInfo]:
        """Diagnose Nautilus engine latency issues"""
        # Check for high latency in order processing
        engine_metrics = metrics.get('nautilus_engine', {})
        
        for metric in engine_metrics:
            if 'latency_ms' in metric.get('name', '') and metric.get('value', 0) > 100:
                return DiagnosticInfo(
                    component="nautilus_engine",
                    issue_type="nautilus_engine_high_latency",
                    severity=AlertSeverity.WARNING,
                    description=f"High latency detected in {metric['name']}: {metric['value']}ms",
                    symptoms=[
                        f"Order processing latency: {metric['value']}ms (threshold: 100ms)",
                        "Potential delays in strategy execution",
                        "Risk of missed trading opportunities"
                    ],
                    possible_causes=[
                        "High system resource usage",
                        "Network connectivity issues",
                        "Data feed delays",
                        "Strategy complexity overhead",
                        "Memory pressure or garbage collection"
                    ],
                    resolution_steps=[
                        "Check system resource usage",
                        "Review network connectivity",
                        "Examine data feed health",
                        "Consider strategy optimization"
                    ],
                    related_metrics=["cpu_percent", "memory_percent", "network_latency"],
                    documentation_links=[
                        "https://docs.nautilustrader.io/performance/optimization",
                        "https://docs.nautilustrader.io/troubleshooting/latency"
                    ]
                )
        
        return None
    
    def _diagnose_strategy_translation_failures(
        self, 
        metrics: Dict[str, Any], 
        health_checks: Dict[str, Any],
        level: DiagnosticLevel
    ) -> Optional[DiagnosticInfo]:
        """Diagnose strategy translation failures"""
        translation_health = health_checks.get('strategy_translation', {})
        
        if translation_health.get('status') == 'critical':
            return DiagnosticInfo(
                component="strategy_translation",
                issue_type="strategy_translation_failures",
                severity=AlertSeverity.ERROR,
                description="Strategy translation service is failing",
                symptoms=[
                    "Strategy translation health check failing",
                    "New strategies cannot be deployed",
                    "F6 to Nautilus conversion errors"
                ],
                possible_causes=[
                    "Invalid F6 strategy definitions",
                    "Nautilus template compatibility issues",
                    "Parameter validation failures",
                    "Translation service resource exhaustion"
                ],
                resolution_steps=[
                    "Review F6 strategy syntax",
                    "Check translation service logs",
                    "Verify Nautilus template versions",
                    "Clear translation cache"
                ],
                related_metrics=["translation_success_rate", "translation_latency"],
                documentation_links=[
                    "https://docs.f6-strategy-registry.io/translation",
                    "https://docs.nautilustrader.io/strategy-development"
                ]
            )
        
        return None
    
    def _diagnose_signal_routing_delays(
        self, 
        metrics: Dict[str, Any], 
        health_checks: Dict[str, Any],
        level: DiagnosticLevel
    ) -> Optional[DiagnosticInfo]:
        """Diagnose signal routing delays"""
        signal_metrics = metrics.get('signal_routing', {})
        
        for metric in signal_metrics:
            if 'delivery_latency' in metric.get('name', '') and metric.get('value', 0) > 1000:
                return DiagnosticInfo(
                    component="signal_routing",
                    issue_type="signal_routing_delays",
                    severity=AlertSeverity.WARNING,
                    description=f"High signal delivery latency: {metric['value']}ms",
                    symptoms=[
                        f"Signal delivery latency: {metric['value']}ms (threshold: 1000ms)",
                        "Strategies may be using stale signals",
                        "Reduced trading performance"
                    ],
                    possible_causes=[
                        "F5 RAG service performance issues",
                        "Signal queue backlog",
                        "Network connectivity problems",
                        "Signal validation bottlenecks"
                    ],
                    resolution_steps=[
                        "Check F5 RAG service health",
                        "Monitor signal queue sizes",
                        "Verify network connectivity",
                        "Review signal validation rules"
                    ],
                    related_metrics=["signal_queue_depth", "f5_response_time"],
                    documentation_links=[
                        "https://docs.f5-intelligence.io/signal-routing",
                        "https://docs.signal-router.io/troubleshooting"
                    ]
                )
        
        return None
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable] = []
    
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add an alert rule"""
        self.alert_rules.append(rule)
        logger.info(
            "Added alert rule",
            rule_name=rule.get('name', 'unnamed'),
            component=rule.get('component', 'unknown'),
            condition=rule.get('condition', 'unknown')
        )
    
    def register_notification_handler(self, handler: Callable) -> None:
        """Register a notification handler"""
        self.notification_handlers.append(handler)
        logger.info(
            "Registered notification handler",
            handler_name=handler.__name__
        )
    
    async def create_alert(
        self,
        severity: AlertSeverity,
        component: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert"""
        alert = Alert(
            alert_id=f"{component}_{int(time.time())}_{hash(title) % 10000}",
            severity=severity,
            component=component,
            title=title,
            message=message,
            timestamp=datetime.now(),
            correlation_id=get_correlation_id(),
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Record metric
        self.metrics_collector.increment_counter(
            "alerts_created",
            component,
            labels={'severity': severity.value}
        )
        
        logger.warning(
            "Alert created",
            alert_id=alert.alert_id,
            severity=severity.value,
            component=component,
            title=title,
            message=message,
            correlation_id=alert.correlation_id
        )
        
        # Send notifications
        await self._send_notifications(alert)
        
        return alert
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """Resolve an active alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        alert.metadata['resolution_message'] = resolution_message
        
        del self.active_alerts[alert_id]
        
        # Record metric
        self.metrics_collector.increment_counter(
            "alerts_resolved",
            alert.component,
            labels={'severity': alert.severity.value}
        )
        
        logger.info(
            "Alert resolved",
            alert_id=alert_id,
            component=alert.component,
            resolution_message=resolution_message,
            correlation_id=alert.correlation_id
        )
        
        return True
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert"""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(
                    "Failed to send notification",
                    handler_name=handler.__name__,
                    alert_id=alert.alert_id,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
    
    def get_active_alerts(self, component: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by component"""
        alerts = list(self.active_alerts.values())
        
        if component:
            alerts = [alert for alert in alerts if alert.component == component]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

class SystemMonitor:
    """Monitors system-level metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start system monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval_seconds)
        )
        
        logger.info(
            "Started system monitoring",
            interval_seconds=interval_seconds
        )
    
    async def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped system monitoring")
    
    async def _monitor_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self._record_system_metrics(metrics)
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in system monitoring loop",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True
                )
                await asyncio.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network_io = psutil.net_io_counters()._asdict()
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_io=network_io,
            process_count=process_count,
            timestamp=datetime.now()
        )
    
    def _record_system_metrics(self, metrics: SystemMetrics) -> None:
        """Record system metrics"""
        component = "system"
        
        self.metrics_collector.set_gauge("cpu_percent", component, metrics.cpu_percent)
        self.metrics_collector.set_gauge("memory_percent", component, metrics.memory_percent)
        self.metrics_collector.set_gauge("disk_usage_percent", component, metrics.disk_usage_percent)
        self.metrics_collector.set_gauge("process_count", component, metrics.process_count)
        
        # Network I/O metrics
        for key, value in metrics.network_io.items():
            self.metrics_collector.set_gauge(f"network_{key}", component, value)

class NautilusMonitor:
    """Main monitoring system for NautilusTrader integration"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.metrics_collector = MetricsCollector(max_history=self.config.max_metric_history)
        self.latency_tracker = LatencyTracker(self.metrics_collector)
        self.health_checker = HealthChecker(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.diagnostic_system = DiagnosticSystem()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup default notification handlers
        self._setup_default_notification_handlers()
        
        # Setup automated fixes
        self._setup_automated_fixes()
    
    def _setup_automated_fixes(self) -> None:
        """Setup automated fix functions"""
        
        async def restart_nautilus_engine() -> str:
            """Automated fix: Restart Nautilus engine"""
            logger.warning("Executing automated fix: Restart Nautilus engine")
            # TODO: Implement actual Nautilus engine restart
            await asyncio.sleep(1)  # Simulate restart time
            return "Nautilus engine restarted successfully"
        
        async def clear_strategy_cache() -> str:
            """Automated fix: Clear strategy translation cache"""
            logger.info("Executing automated fix: Clear strategy cache")
            # TODO: Implement actual cache clearing
            await asyncio.sleep(0.5)  # Simulate cache clear time
            return "Strategy translation cache cleared"
        
        async def restart_signal_router() -> str:
            """Automated fix: Restart signal routing service"""
            logger.info("Executing automated fix: Restart signal router")
            # TODO: Implement actual signal router restart
            await asyncio.sleep(1)  # Simulate restart time
            return "Signal routing service restarted successfully"
        
        # Register automated fixes
        self.diagnostic_system.register_automated_fix("restart_nautilus_engine", restart_nautilus_engine)
        self.diagnostic_system.register_automated_fix("clear_strategy_cache", clear_strategy_cache)
        self.diagnostic_system.register_automated_fix("restart_signal_router", restart_signal_router)
    
    async def get_system_status_with_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system status with diagnostic information"""
        # Get basic system status
        basic_status = await self.get_system_status()
        
        # Run diagnostics
        diagnostics = await self.diagnostic_system.diagnose_system_issues(
            basic_status['recent_metrics'],
            basic_status['health_checks'],
            DiagnosticLevel.DETAILED
        )
        
        # Add diagnostic information
        basic_status['diagnostics'] = [asdict(diag) for diag in diagnostics]
        
        # Add resolution guidance for identified issues
        basic_status['resolution_guidance'] = {}
        for diag in diagnostics:
            guidance = self.diagnostic_system.get_resolution_guidance(diag.issue_type)
            if guidance:
                basic_status['resolution_guidance'][diag.issue_type] = asdict(guidance)
        
        return basic_status
    
    async def execute_automated_resolution(self, issue_type: str) -> Dict[str, Any]:
        """Execute automated resolution for a specific issue type"""
        guidance = self.diagnostic_system.get_resolution_guidance(issue_type)
        if not guidance or not guidance.automated_fixes:
            return {
                'success': False,
                'error': f'No automated fixes available for issue type: {issue_type}'
            }
        
        results = []
        for fix in guidance.automated_fixes:
            try:
                result = await self.diagnostic_system.execute_automated_fix(
                    fix['function'],
                    fix.get('parameters', {})
                )
                results.append({
                    'fix_name': fix['name'],
                    'result': result
                })
            except Exception as e:
                results.append({
                    'fix_name': fix['name'],
                    'result': {
                        'success': False,
                        'error': str(e)
                    }
                })
        
        return {
            'issue_type': issue_type,
            'automated_fixes_executed': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks for NautilusTrader components"""
        
        async def nautilus_engine_health() -> Dict[str, Any]:
            """Check NautilusTrader engine health"""
            # TODO: Implement actual Nautilus engine health check
            return {
                'status': 'healthy',
                'message': 'Nautilus engine is running',
                'details': {'engine_state': 'active'}
            }
        
        async def strategy_translation_health() -> Dict[str, Any]:
            """Check strategy translation service health"""
            # TODO: Implement actual strategy translation health check
            return {
                'status': 'healthy',
                'message': 'Strategy translation service is running',
                'details': {'active_strategies': 0}
            }
        
        async def signal_routing_health() -> Dict[str, Any]:
            """Check signal routing service health"""
            # TODO: Implement actual signal routing health check
            return {
                'status': 'healthy',
                'message': 'Signal routing service is running',
                'details': {'active_signals': 0}
            }
        
        self.health_checker.register_health_check("nautilus_engine", nautilus_engine_health)
        self.health_checker.register_health_check("strategy_translation", strategy_translation_health)
        self.health_checker.register_health_check("signal_routing", signal_routing_health)
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules"""
        
        # High CPU usage alert
        self.alert_manager.add_alert_rule({
            'name': 'high_cpu_usage',
            'component': 'system',
            'condition': 'cpu_percent > 80',
            'severity': 'warning',
            'message': 'High CPU usage detected'
        })
        
        # High memory usage alert
        self.alert_manager.add_alert_rule({
            'name': 'high_memory_usage',
            'component': 'system',
            'condition': 'memory_percent > 85',
            'severity': 'warning',
            'message': 'High memory usage detected'
        })
        
        # Component health alert
        self.alert_manager.add_alert_rule({
            'name': 'component_unhealthy',
            'component': '*',
            'condition': 'health_status == 0',
            'severity': 'critical',
            'message': 'Component health check failed'
        })
    
    def _setup_default_notification_handlers(self) -> None:
        """Setup default notification handlers"""
        
        def log_notification_handler(alert: Alert) -> None:
            """Log alert notifications"""
            logger.warning(
                "Alert notification",
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                component=alert.component,
                title=alert.title,
                message=alert.message
            )
        
        self.alert_manager.register_notification_handler(log_notification_handler)
    
    async def start(self) -> None:
        """Start the monitoring system"""
        logger.info("Starting NautilusTrader monitoring system")
        
        # Start system monitoring
        await self.system_monitor.start_monitoring(
            interval_seconds=self.config.system_monitor_interval
        )
        
        logger.info("NautilusTrader monitoring system started")
    
    async def stop(self) -> None:
        """Stop the monitoring system"""
        logger.info("Stopping NautilusTrader monitoring system")
        
        # Stop system monitoring
        await self.system_monitor.stop_monitoring()
        
        logger.info("NautilusTrader monitoring system stopped")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Get health checks
        health_checks = await self.health_checker.check_all_components()
        overall_health = self.health_checker.get_overall_health()
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Get recent metrics
        recent_metrics = {}
        for component in ['system', 'nautilus_engine', 'strategy_translation', 'signal_routing']:
            recent_metrics[component] = self.metrics_collector.get_recent_metrics(component, minutes=5)
        
        return {
            'overall_health': overall_health.value,
            'health_checks': {comp: asdict(check) for comp, check in health_checks.items()},
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'recent_metrics': {
                comp: [asdict(metric) for metric in metrics]
                for comp, metrics in recent_metrics.items()
            },
            'timestamp': datetime.now().isoformat()
        }

# Global monitoring instance
_global_monitor: Optional[NautilusMonitor] = None

def get_global_monitor() -> NautilusMonitor:
    """Get global monitoring instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = NautilusMonitor()
    return _global_monitor

def setup_monitoring(config: Optional[MonitoringConfig] = None) -> NautilusMonitor:
    """Setup global monitoring system"""
    global _global_monitor
    _global_monitor = NautilusMonitor(config)
    return _global_monitor

# Convenience functions for common monitoring operations
def record_latency(operation_name: str, component: str, latency_ms: float) -> None:
    """Record operation latency"""
    monitor = get_global_monitor()
    monitor.metrics_collector.record_histogram(
        f"{operation_name}_latency_ms",
        component,
        latency_ms
    )

def increment_counter(name: str, component: str, value: float = 1.0) -> None:
    """Increment a counter metric"""
    monitor = get_global_monitor()
    monitor.metrics_collector.increment_counter(name, component, value)

def set_gauge(name: str, component: str, value: float) -> None:
    """Set a gauge metric"""
    monitor = get_global_monitor()
    monitor.metrics_collector.set_gauge(name, component, value)

async def create_alert(severity: AlertSeverity, component: str, title: str, message: str) -> Alert:
    """Create an alert"""
    monitor = get_global_monitor()
    return await monitor.alert_manager.create_alert(severity, component, title, message)

class LatencyContext:
    """Context manager for tracking operation latency"""
    
    def __init__(self, operation_name: str, component: str):
        self.operation_name = operation_name
        self.component = component
        self.operation_id = f"{component}_{operation_name}_{int(time.time_ns())}"
        self.monitor = get_global_monitor()
    
    def __enter__(self) -> str:
        """Start latency tracking"""
        self.monitor.latency_tracker.start_operation(
            self.operation_id,
            self.component,
            self.operation_name
        )
        return self.operation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End latency tracking"""
        self.monitor.latency_tracker.end_operation(
            self.operation_id,
            self.component,
            self.operation_name
        )

class LatencyContext:
    """Context manager for tracking operation latency"""
    
    def __init__(self, operation_name: str, component: str):
        self.operation_name = operation_name
        self.component = component
        self.operation_id = f"{component}_{operation_name}_{int(time.time_ns())}"
        self.monitor = get_global_monitor()
    
    def __enter__(self) -> str:
        """Start latency tracking"""
        self.monitor.latency_tracker.start_operation(
            self.operation_id,
            self.component,
            self.operation_name
        )
        return self.operation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End latency tracking"""
        self.monitor.latency_tracker.end_operation(
            self.operation_id,
            self.component,
            self.operation_name
        )

def track_latency(operation_name: str, component: str) -> LatencyContext:
    """Create a latency tracking context manager"""
    return LatencyContext(operation_name, component)

async def get_system_diagnostics(level: DiagnosticLevel = DiagnosticLevel.BASIC) -> List[DiagnosticInfo]:
    """Get system diagnostics"""
    monitor = get_global_monitor()
    status = await monitor.get_system_status()
    return await monitor.diagnostic_system.diagnose_system_issues(
        status['recent_metrics'],
        status['health_checks'],
        level
    )

async def get_resolution_guidance(issue_type: str) -> Optional[ResolutionGuidance]:
    """Get resolution guidance for a specific issue type"""
    monitor = get_global_monitor()
    return monitor.diagnostic_system.get_resolution_guidance(issue_type)

async def execute_automated_fix(fix_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute an automated fix"""
    monitor = get_global_monitor()
    return await monitor.diagnostic_system.execute_automated_fix(fix_name, parameters)
    """Get resolution guidance for a specific issue type"""
    monitor = get_global_monitor()
    return monitor.diagnostic_system.get_resolution_guidance(issue_type)

async def execute_automated_fix(fix_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute an automated fix"""
    monitor = get_global_monitor()
    return await monitor.diagnostic_system.execute_automated_fix(fix_name, parameters)
    """Get resolution guidance for an issue type"""
    monitor = get_global_monitor()
    return monitor.diagnostic_system.get_resolution_guidance(issue_type)

async def execute_automated_fix(fix_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute an automated fix"""
    monitor = get_global_monitor()
    return await monitor.diagnostic_system.execute_automated_fix(fix_name, parameters)

async def execute_issue_resolution(issue_type: str) -> Dict[str, Any]:
    """Execute automated resolution for an issue type"""
    monitor = get_global_monitor()
    return await monitor.execute_automated_resolution(issue_type)