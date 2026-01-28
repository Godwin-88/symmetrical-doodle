"""
Orchestration Error Handling and Recovery System

This module provides comprehensive error handling, recovery mechanisms,
and resilience features for the multi-source pipeline orchestration system.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union, Type
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback
import uuid
from pathlib import Path

from core.logging import get_logger
from services.multi_source_auth import DataSourceType
from config.orchestration_config import ErrorHandlingConfiguration


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    PERMISSION = "permission"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL_LOGIC = "internal_logic"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SKIP_AND_CONTINUE = "skip_and_continue"
    FALLBACK_METHOD = "fallback_method"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class ErrorContext:
    """Context information for an error"""
    error_id: str
    timestamp: datetime
    source_type: Optional[DataSourceType] = None
    source_connection_id: Optional[str] = None
    file_id: Optional[str] = None
    operation: Optional[str] = None
    phase: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Complete error record with classification and recovery info"""
    context: ErrorContext
    exception: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    recovery_strategy: RecoveryStrategy
    
    # Error details
    error_message: str
    stack_trace: str
    
    # Recovery tracking
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_message: Optional[str] = None
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None


@dataclass
class RecoveryAction:
    """Recovery action to be executed"""
    action_id: str
    error_record: ErrorRecord
    strategy: RecoveryStrategy
    action_function: Callable
    action_args: tuple = field(default_factory=tuple)
    action_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed: bool = False
    execution_result: Optional[Any] = None
    execution_error: Optional[Exception] = None


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ErrorClassifier")
        
        # Error classification rules
        self._classification_rules = {
            # Authentication errors
            'authentication': {
                'patterns': ['auth', 'credential', 'token', 'permission denied', 'unauthorized'],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.AUTHENTICATION,
                'strategy': RecoveryStrategy.MANUAL_INTERVENTION
            },
            
            # Network errors
            'network': {
                'patterns': ['connection', 'timeout', 'network', 'unreachable', 'dns'],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.NETWORK,
                'strategy': RecoveryStrategy.RETRY_WITH_BACKOFF
            },
            
            # Rate limiting
            'rate_limit': {
                'patterns': ['rate limit', 'quota exceeded', 'too many requests', '429'],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.RATE_LIMIT,
                'strategy': RecoveryStrategy.RETRY_WITH_BACKOFF
            },
            
            # Permission errors
            'permission': {
                'patterns': ['permission', 'access denied', 'forbidden', '403'],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.PERMISSION,
                'strategy': RecoveryStrategy.SKIP_AND_CONTINUE
            },
            
            # Data corruption
            'data_corruption': {
                'patterns': ['corrupt', 'invalid format', 'parse error', 'malformed'],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.DATA_CORRUPTION,
                'strategy': RecoveryStrategy.SKIP_AND_CONTINUE
            },
            
            # Resource exhaustion
            'resource_exhaustion': {
                'patterns': ['memory', 'disk space', 'out of space', 'resource limit'],
                'severity': ErrorSeverity.CRITICAL,
                'category': ErrorCategory.RESOURCE_EXHAUSTION,
                'strategy': RecoveryStrategy.GRACEFUL_DEGRADATION
            }
        }
    
    def classify_error(self, exception: Exception, context: ErrorContext) -> tuple[ErrorSeverity, ErrorCategory, RecoveryStrategy]:
        """Classify an error and determine recovery strategy"""
        try:
            error_message = str(exception).lower()
            exception_type = type(exception).__name__.lower()
            
            # Check classification rules
            for rule_name, rule in self._classification_rules.items():
                for pattern in rule['patterns']:
                    if pattern in error_message or pattern in exception_type:
                        return rule['severity'], rule['category'], rule['strategy']
            
            # Default classification for unmatched errors
            if isinstance(exception, (ConnectionError, TimeoutError)):
                return ErrorSeverity.MEDIUM, ErrorCategory.NETWORK, RecoveryStrategy.RETRY_WITH_BACKOFF
            elif isinstance(exception, PermissionError):
                return ErrorSeverity.HIGH, ErrorCategory.PERMISSION, RecoveryStrategy.SKIP_AND_CONTINUE
            elif isinstance(exception, ValueError):
                return ErrorSeverity.MEDIUM, ErrorCategory.DATA_CORRUPTION, RecoveryStrategy.SKIP_AND_CONTINUE
            elif isinstance(exception, MemoryError):
                return ErrorSeverity.CRITICAL, ErrorCategory.RESOURCE_EXHAUSTION, RecoveryStrategy.GRACEFUL_DEGRADATION
            else:
                return ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN, RecoveryStrategy.RETRY
                
        except Exception as e:
            self.logger.error(f"Error classifying exception: {e}")
            return ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN, RecoveryStrategy.RETRY


class OrchestrationErrorHandler:
    """
    Comprehensive error handling and recovery system for multi-source
    pipeline orchestration with intelligent error classification,
    recovery strategies, and resilience mechanisms.
    """
    
    def __init__(self, config: ErrorHandlingConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Error tracking
        self._error_records: Dict[str, ErrorRecord] = {}
        self._error_history: List[ErrorRecord] = []
        self._error_lock = asyncio.Lock()
        
        # Recovery management
        self._recovery_queue: asyncio.Queue = asyncio.Queue()
        self._recovery_workers: List[asyncio.Task] = []
        self._recovery_lock = asyncio.Lock()
        
        # Error classification
        self._classifier = ErrorClassifier()
        
        # Circuit breaker state
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._circuit_breaker_lock = asyncio.Lock()
        
        # Background tasks
        self._recovery_processor_task: Optional[asyncio.Task] = None
        self._error_cleanup_task: Optional[asyncio.Task] = None
        
        # State
        self._active = False
        self._shutdown_requested = False
        
        # Statistics
        self._error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'by_category': {},
            'by_severity': {},
            'by_source': {}
        }
    
    async def start(self):
        """Start the error handling system"""
        try:
            if self._active:
                return
            
            self.logger.info("Starting orchestration error handling system")
            
            self._active = True
            self._shutdown_requested = False
            
            # Start recovery workers
            num_workers = min(4, self.config.max_retries_per_source)
            for i in range(num_workers):
                worker = asyncio.create_task(self._recovery_worker(f"recovery_worker_{i}"))
                self._recovery_workers.append(worker)
            
            # Start background tasks
            self._recovery_processor_task = asyncio.create_task(self._recovery_processor())
            self._error_cleanup_task = asyncio.create_task(self._error_cleanup())
            
            self.logger.info("Error handling system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start error handling system: {e}")
            raise
    
    async def stop(self):
        """Stop the error handling system"""
        try:
            self.logger.info("Stopping orchestration error handling system")
            
            self._shutdown_requested = True
            self._active = False
            
            # Cancel background tasks
            tasks_to_cancel = [
                self._recovery_processor_task,
                self._error_cleanup_task
            ] + self._recovery_workers
            
            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if tasks_to_cancel:
                await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
            
            self.logger.info("Error handling system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping error handling system: {e}")
    
    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        recovery_function: Optional[Callable] = None,
        recovery_args: tuple = (),
        recovery_kwargs: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """Handle an error with classification and recovery"""
        try:
            # Classify the error
            severity, category, strategy = self._classifier.classify_error(exception, context)
            
            # Create error record
            error_record = ErrorRecord(
                context=context,
                exception=exception,
                severity=severity,
                category=category,
                recovery_strategy=strategy,
                error_message=str(exception),
                stack_trace=traceback.format_exc(),
                max_retries=self._get_max_retries_for_category(category)
            )
            
            # Store error record
            async with self._error_lock:
                self._error_records[context.error_id] = error_record
                self._error_history.append(error_record)
                
                # Update statistics
                self._error_stats['total_errors'] += 1
                self._error_stats['by_category'][category.value] = self._error_stats['by_category'].get(category.value, 0) + 1
                self._error_stats['by_severity'][severity.value] = self._error_stats['by_severity'].get(severity.value, 0) + 1
                
                if context.source_type:
                    source_key = f"{context.source_type.value}:{context.source_connection_id}"
                    self._error_stats['by_source'][source_key] = self._error_stats['by_source'].get(source_key, 0) + 1
            
            # Log the error
            log_level = {
                ErrorSeverity.LOW: "info",
                ErrorSeverity.MEDIUM: "warning",
                ErrorSeverity.HIGH: "error",
                ErrorSeverity.CRITICAL: "error"
            }.get(severity, "error")
            
            getattr(self.logger, log_level)(
                f"Error handled: {category.value}",
                error_id=context.error_id,
                severity=severity.value,
                strategy=strategy.value,
                message=error_record.error_message,
                source_type=context.source_type.value if context.source_type else None,
                source_connection_id=context.source_connection_id,
                operation=context.operation,
                correlation_id=context.correlation_id
            )
            
            # Check circuit breaker
            if await self._should_circuit_break(context, error_record):
                self.logger.warning(f"Circuit breaker activated for {context.source_type}:{context.source_connection_id}")
                error_record.recovery_strategy = RecoveryStrategy.FAIL_FAST
            
            # Schedule recovery if appropriate
            if recovery_function and strategy != RecoveryStrategy.FAIL_FAST:
                await self._schedule_recovery(error_record, recovery_function, recovery_args, recovery_kwargs or {})
            
            return error_record
            
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            raise
    
    def _get_max_retries_for_category(self, category: ErrorCategory) -> int:
        """Get maximum retries based on error category"""
        category_retries = {
            ErrorCategory.AUTHENTICATION: 1,  # Don't retry auth errors much
            ErrorCategory.NETWORK: self.config.max_retries_per_source,
            ErrorCategory.RATE_LIMIT: self.config.max_retries_per_source * 2,  # More retries for rate limits
            ErrorCategory.PERMISSION: 1,  # Don't retry permission errors
            ErrorCategory.DATA_CORRUPTION: 1,  # Don't retry data corruption
            ErrorCategory.RESOURCE_EXHAUSTION: 2,
            ErrorCategory.CONFIGURATION: 1,
            ErrorCategory.EXTERNAL_SERVICE: self.config.max_retries_per_source,
            ErrorCategory.INTERNAL_LOGIC: 2,
            ErrorCategory.UNKNOWN: self.config.max_retries_per_source
        }
        return category_retries.get(category, self.config.max_retries_per_source)
    
    async def _should_circuit_break(self, context: ErrorContext, error_record: ErrorRecord) -> bool:
        """Check if circuit breaker should activate"""
        if not context.source_type or not context.source_connection_id:
            return False
        
        source_key = f"{context.source_type.value}:{context.source_connection_id}"
        
        async with self._circuit_breaker_lock:
            if source_key not in self._circuit_breakers:
                self._circuit_breakers[source_key] = {
                    'error_count': 0,
                    'last_error_time': datetime.now(timezone.utc),
                    'circuit_open': False,
                    'circuit_opened_at': None
                }
            
            breaker = self._circuit_breakers[source_key]
            now = datetime.now(timezone.utc)
            
            # Reset error count if enough time has passed
            if (now - breaker['last_error_time']).total_seconds() > 300:  # 5 minutes
                breaker['error_count'] = 0
            
            breaker['error_count'] += 1
            breaker['last_error_time'] = now
            
            # Open circuit if too many errors
            if breaker['error_count'] >= 5 and not breaker['circuit_open']:
                breaker['circuit_open'] = True
                breaker['circuit_opened_at'] = now
                return True
            
            # Check if circuit should close (after 10 minutes)
            if (breaker['circuit_open'] and breaker['circuit_opened_at'] and
                (now - breaker['circuit_opened_at']).total_seconds() > 600):
                breaker['circuit_open'] = False
                breaker['error_count'] = 0
                self.logger.info(f"Circuit breaker closed for {source_key}")
            
            return breaker['circuit_open']
    
    async def _schedule_recovery(
        self,
        error_record: ErrorRecord,
        recovery_function: Callable,
        recovery_args: tuple,
        recovery_kwargs: Dict[str, Any]
    ):
        """Schedule a recovery action"""
        try:
            # Calculate retry delay
            delay = self._calculate_retry_delay(error_record)
            next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
            
            error_record.next_retry_at = next_retry_at
            
            # Create recovery action
            recovery_action = RecoveryAction(
                action_id=str(uuid.uuid4()),
                error_record=error_record,
                strategy=error_record.recovery_strategy,
                action_function=recovery_function,
                action_args=recovery_args,
                action_kwargs=recovery_kwargs,
                scheduled_at=next_retry_at
            )
            
            # Add to recovery queue
            await self._recovery_queue.put(recovery_action)
            
            self.logger.info(
                f"Recovery scheduled for error {error_record.context.error_id}",
                strategy=error_record.recovery_strategy.value,
                delay_seconds=delay,
                retry_count=error_record.retry_count
            )
            
        except Exception as e:
            self.logger.error(f"Error scheduling recovery: {e}")
    
    def _calculate_retry_delay(self, error_record: ErrorRecord) -> float:
        """Calculate retry delay based on strategy and retry count"""
        base_delay = self.config.retry_delay_seconds
        
        if error_record.recovery_strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
            if self.config.exponential_backoff:
                delay = base_delay * (2 ** error_record.retry_count)
                return min(delay, self.config.max_retry_delay_seconds)
            else:
                return base_delay * (error_record.retry_count + 1)
        elif error_record.category == ErrorCategory.RATE_LIMIT:
            # Longer delays for rate limiting
            return min(base_delay * 5 * (error_record.retry_count + 1), self.config.max_retry_delay_seconds)
        else:
            return base_delay
    
    async def _recovery_processor(self):
        """Background task to process scheduled recoveries"""
        while not self._shutdown_requested:
            try:
                # Check for due recovery actions
                now = datetime.now(timezone.utc)
                
                # Process recovery queue
                try:
                    recovery_action = await asyncio.wait_for(self._recovery_queue.get(), timeout=1.0)
                    
                    if recovery_action.scheduled_at <= now:
                        # Execute recovery immediately
                        await self._execute_recovery(recovery_action)
                    else:
                        # Put back in queue for later
                        await self._recovery_queue.put(recovery_action)
                        await asyncio.sleep(1)
                        
                except asyncio.TimeoutError:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in recovery processor: {e}")
                await asyncio.sleep(5)
    
    async def _recovery_worker(self, worker_name: str):
        """Recovery worker to execute recovery actions"""
        self.logger.debug(f"Starting recovery worker: {worker_name}")
        
        while not self._shutdown_requested:
            try:
                # Wait for recovery actions
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in recovery worker {worker_name}: {e}")
                await asyncio.sleep(5)
    
    async def _execute_recovery(self, recovery_action: RecoveryAction):
        """Execute a recovery action"""
        try:
            error_record = recovery_action.error_record
            
            self.logger.info(
                f"Executing recovery for error {error_record.context.error_id}",
                strategy=recovery_action.strategy.value,
                retry_count=error_record.retry_count
            )
            
            # Update retry count
            error_record.retry_count += 1
            error_record.recovery_attempted = True
            
            # Execute recovery function
            try:
                result = await recovery_action.action_function(
                    *recovery_action.action_args,
                    **recovery_action.action_kwargs
                )
                
                recovery_action.execution_result = result
                recovery_action.executed = True
                
                # Mark as recovered
                error_record.recovery_successful = True
                error_record.resolved = True
                error_record.resolved_at = datetime.now(timezone.utc)
                error_record.resolution_method = "automatic_recovery"
                
                async with self._error_lock:
                    self._error_stats['recovered_errors'] += 1
                
                self.logger.info(
                    f"Recovery successful for error {error_record.context.error_id}",
                    retry_count=error_record.retry_count
                )
                
            except Exception as recovery_error:
                recovery_action.execution_error = recovery_error
                recovery_action.executed = True
                
                self.logger.warning(
                    f"Recovery failed for error {error_record.context.error_id}",
                    recovery_error=str(recovery_error),
                    retry_count=error_record.retry_count
                )
                
                # Check if we should retry again
                if error_record.retry_count < error_record.max_retries:
                    await self._schedule_recovery(
                        error_record,
                        recovery_action.action_function,
                        recovery_action.action_args,
                        recovery_action.action_kwargs
                    )
                else:
                    # Max retries reached
                    error_record.recovery_successful = False
                    error_record.resolved = True
                    error_record.resolved_at = datetime.now(timezone.utc)
                    error_record.resolution_method = "max_retries_exceeded"
                    
                    async with self._error_lock:
                        self._error_stats['failed_recoveries'] += 1
                    
                    self.logger.error(
                        f"Recovery failed permanently for error {error_record.context.error_id}",
                        max_retries=error_record.max_retries
                    )
                    
        except Exception as e:
            self.logger.error(f"Error executing recovery: {e}")
    
    async def _error_cleanup(self):
        """Background task to clean up old error records"""
        while not self._shutdown_requested:
            try:
                # Clean up old resolved errors
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                
                async with self._error_lock:
                    # Remove old resolved errors from active records
                    resolved_error_ids = [
                        error_id for error_id, error_record in self._error_records.items()
                        if error_record.resolved and error_record.resolved_at and error_record.resolved_at < cutoff_time
                    ]
                    
                    for error_id in resolved_error_ids:
                        del self._error_records[error_id]
                    
                    # Limit error history size
                    if len(self._error_history) > 1000:
                        self._error_history = self._error_history[-1000:]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Error in error cleanup: {e}")
                await asyncio.sleep(3600)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return {
            'total_errors': self._error_stats['total_errors'],
            'recovered_errors': self._error_stats['recovered_errors'],
            'failed_recoveries': self._error_stats['failed_recoveries'],
            'recovery_rate': (self._error_stats['recovered_errors'] / max(1, self._error_stats['total_errors'])) * 100,
            'by_category': self._error_stats['by_category'],
            'by_severity': self._error_stats['by_severity'],
            'by_source': self._error_stats['by_source'],
            'active_errors': len(self._error_records),
            'circuit_breakers': {
                key: {
                    'error_count': breaker['error_count'],
                    'circuit_open': breaker['circuit_open'],
                    'last_error_time': breaker['last_error_time'].isoformat()
                }
                for key, breaker in self._circuit_breakers.items()
            }
        }
    
    def get_active_errors(self) -> List[ErrorRecord]:
        """Get currently active (unresolved) errors"""
        return [error for error in self._error_records.values() if not error.resolved]
    
    async def resolve_error(self, error_id: str, resolution_method: str = "manual"):
        """Manually resolve an error"""
        async with self._error_lock:
            if error_id in self._error_records:
                error_record = self._error_records[error_id]
                error_record.resolved = True
                error_record.resolved_at = datetime.now(timezone.utc)
                error_record.resolution_method = resolution_method
                
                self.logger.info(f"Error {error_id} resolved manually")


# Global error handler instance
_error_handler: Optional[OrchestrationErrorHandler] = None


async def get_orchestration_error_handler(config: Optional[ErrorHandlingConfiguration] = None) -> OrchestrationErrorHandler:
    """Get or create global orchestration error handler"""
    global _error_handler
    
    if _error_handler is None:
        if config is None:
            from config.orchestration_config import load_orchestration_config
            orchestration_config = load_orchestration_config()
            config = orchestration_config.error_handling
        
        _error_handler = OrchestrationErrorHandler(config)
        await _error_handler.start()
    
    return _error_handler