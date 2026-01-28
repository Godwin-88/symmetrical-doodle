"""
Signal Router & Buffer (SIG)

This module provides the Signal Router & Buffer component that manages AI signal
delivery from F5 Intelligence Layer to NautilusTrader strategies with robust
delivery mechanisms, buffering, and replay capabilities.
"""

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field, validator

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.core.logging import (
    get_correlation_id,
    get_logger,
    log_error_with_context,
    with_correlation_id,
)


class SignalType(str, Enum):
    """AI signal types from F5 Intelligence Layer."""
    
    REGIME_PREDICTION = "regime_prediction"
    CORRELATION_SHIFT = "correlation_shift"
    SENTIMENT_SCORE = "sentiment_score"
    VOLATILITY_FORECAST = "volatility_forecast"
    MOMENTUM_SIGNAL = "momentum_signal"
    MEAN_REVERSION_SIGNAL = "mean_reversion_signal"
    BREAKOUT_SIGNAL = "breakout_signal"
    RISK_ALERT = "risk_alert"


class SignalConfidence(str, Enum):
    """Signal confidence levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeliveryStatus(str, Enum):
    """Signal delivery status."""
    
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    ACKNOWLEDGED = "acknowledged"


class AISignal(BaseModel):
    """AI signal from F5 Intelligence Layer."""
    
    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    signal_type: SignalType
    instrument_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: SignalConfidence
    value: Union[float, str, Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Signal provenance
    source_model: str
    model_version: str = "1.0"
    generation_time: datetime = Field(default_factory=datetime.now)
    
    # Delivery tracking
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    ttl_seconds: int = 300  # 5 minutes default TTL
    
    @validator('confidence_level', pre=True, always=True)
    def set_confidence_level(cls, v, values):
        """Set confidence level based on confidence score."""
        if 'confidence' in values:
            confidence = values['confidence']
            if confidence >= 0.8:
                return SignalConfidence.CRITICAL
            elif confidence >= 0.6:
                return SignalConfidence.HIGH
            elif confidence >= 0.4:
                return SignalConfidence.MEDIUM
            else:
                return SignalConfidence.LOW
        return v or SignalConfidence.MEDIUM
    
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        expiry_time = self.timestamp + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.value,
            "instrument_id": self.instrument_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "value": self.value,
            "metadata": self.metadata,
            "source_model": self.source_model,
            "model_version": self.model_version,
            "correlation_id": self.correlation_id,
        }


class SignalSubscription(BaseModel):
    """Signal subscription configuration."""
    
    subscription_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    signal_types: List[SignalType]
    instrument_ids: List[str] = Field(default_factory=list)  # Empty = all instruments
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    callback_function: Optional[str] = None
    
    # Filtering options
    filter_duplicates: bool = True
    max_signals_per_minute: int = 60
    
    # Delivery options
    delivery_timeout: int = 30  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


class SignalDeliveryRecord(BaseModel):
    """Record of signal delivery attempt."""
    
    delivery_id: str = Field(default_factory=lambda: str(uuid4()))
    signal_id: str
    strategy_id: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # Delivery details
    attempt_count: int = 0
    last_error: Optional[str] = None
    delivery_latency_ms: Optional[float] = None


class SignalBuffer(BaseModel):
    """Signal buffer for backtesting and replay."""
    
    buffer_id: str = Field(default_factory=lambda: str(uuid4()))
    instrument_id: str
    start_time: datetime
    end_time: datetime
    signals: List[AISignal] = Field(default_factory=list)
    
    # Buffer metadata
    created_at: datetime = Field(default_factory=datetime.now)
    signal_count: int = 0
    total_size_bytes: int = 0
    
    def add_signal(self, signal: AISignal) -> None:
        """Add signal to buffer."""
        if self.start_time <= signal.timestamp <= self.end_time:
            self.signals.append(signal)
            self.signal_count = len(self.signals)
            self.total_size_bytes += len(json.dumps(signal.to_dict()))
    
    def get_signals_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[AISignal]:
        """Get signals within time range."""
        return [
            signal for signal in self.signals
            if start <= signal.timestamp <= end
        ]


class DeadLetterQueue:
    """Dead letter queue for failed signal deliveries."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize dead letter queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queue: deque = deque(maxlen=max_size)
        self._failed_count = 0
        self.logger = get_logger("nautilus_integration.dead_letter_queue")
    
    def add_failed_delivery(self, delivery_record: SignalDeliveryRecord, signal: AISignal) -> None:
        """Add failed delivery to queue."""
        try:
            failed_item = {
                "delivery_record": delivery_record,
                "signal": signal,
                "failed_at": datetime.now(),
                "failure_reason": delivery_record.last_error,
            }
            
            self._queue.append(failed_item)
            self._failed_count += 1
            
            self.logger.warning(
                "Added failed delivery to dead letter queue",
                signal_id=signal.signal_id,
                strategy_id=delivery_record.strategy_id,
                failure_reason=delivery_record.last_error,
                queue_size=len(self._queue),
            )
            
        except Exception as error:
            self.logger.error(
                "Failed to add item to dead letter queue",
                error=str(error),
            )
    
    def get_failed_deliveries(
        self,
        limit: Optional[int] = None,
        strategy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get failed deliveries from queue."""
        try:
            items = list(self._queue)
            
            # Filter by strategy if specified
            if strategy_id:
                items = [
                    item for item in items
                    if item["delivery_record"].strategy_id == strategy_id
                ]
            
            # Apply limit
            if limit:
                items = items[-limit:]
            
            return items
            
        except Exception as error:
            self.logger.error(
                "Failed to get failed deliveries",
                error=str(error),
            )
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dead letter queue statistics."""
        return {
            "queue_size": len(self._queue),
            "total_failed_count": self._failed_count,
            "max_size": self.max_size,
            "queue_full": len(self._queue) >= self.max_size,
        }


class SignalProvenance(BaseModel):
    """Signal provenance tracking for audit trails."""
    
    signal_id: str
    correlation_id: str
    
    # Source information
    source_system: str = "F5_Intelligence_Layer"
    source_model: str
    model_version: str
    generation_timestamp: datetime
    
    # Processing pipeline
    processing_steps: List[Dict[str, Any]] = Field(default_factory=list)
    validation_results: Optional[Dict[str, Any]] = None
    
    # Delivery tracking
    routing_timestamp: Optional[datetime] = None
    delivery_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    final_delivery_timestamp: Optional[datetime] = None
    
    # Performance metrics
    total_latency_ms: Optional[float] = None
    processing_latency_ms: Optional[float] = None
    delivery_latency_ms: Optional[float] = None
    
    def add_processing_step(self, step_name: str, timestamp: datetime, details: Dict[str, Any] = None) -> None:
        """Add a processing step to the provenance trail."""
        step = {
            "step_name": step_name,
            "timestamp": timestamp.isoformat(),
            "details": details or {},
        }
        self.processing_steps.append(step)
    
    def add_delivery_attempt(self, strategy_id: str, timestamp: datetime, success: bool, error: Optional[str] = None) -> None:
        """Add a delivery attempt to the provenance trail."""
        attempt = {
            "strategy_id": strategy_id,
            "timestamp": timestamp.isoformat(),
            "success": success,
            "error": error,
        }
        self.delivery_attempts.append(attempt)
    
    def calculate_latencies(self) -> None:
        """Calculate various latency metrics."""
        if self.final_delivery_timestamp and self.generation_timestamp:
            self.total_latency_ms = (
                self.final_delivery_timestamp - self.generation_timestamp
            ).total_seconds() * 1000
        
        if self.routing_timestamp and self.generation_timestamp:
            self.processing_latency_ms = (
                self.routing_timestamp - self.generation_timestamp
            ).total_seconds() * 1000
        
        if self.final_delivery_timestamp and self.routing_timestamp:
            self.delivery_latency_ms = (
                self.final_delivery_timestamp - self.routing_timestamp
            ).total_seconds() * 1000


class RealTimeSignalDelivery:
    """Real-time signal delivery system with sub-second latency."""
    
    def __init__(self, config: NautilusConfig):
        """
        Initialize real-time signal delivery system.
        
        Args:
            config: NautilusTrader integration configuration
        """
        self.config = config
        self.logger = get_logger("nautilus_integration.realtime_delivery")
        
        # Performance tracking
        self._delivery_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._latency_targets = {
            "processing": 100.0,  # 100ms target for processing
            "delivery": 500.0,    # 500ms target for delivery
            "total": 1000.0,      # 1s target for total latency
        }
        
        # Provenance tracking
        self._provenance_records: Dict[str, SignalProvenance] = {}
        self._provenance_cleanup_interval = 3600  # 1 hour
        
        # Fallback cache for delivery failures
        self._cached_signals: Dict[str, List[AISignal]] = defaultdict(list)
        self._cache_ttl = timedelta(minutes=5)
        
    async def deliver_signal_realtime(
        self,
        signal: AISignal,
        strategy_id: str,
        callback: callable,
        timeout: float = 0.5
    ) -> Dict[str, Any]:
        """
        Deliver signal with real-time performance guarantees.
        
        Args:
            signal: AI signal to deliver
            strategy_id: Target strategy ID
            callback: Strategy callback function
            timeout: Delivery timeout in seconds
            
        Returns:
            Delivery result with performance metrics
        """
        start_time = datetime.now()
        
        # Initialize provenance tracking
        provenance = SignalProvenance(
            signal_id=signal.signal_id,
            correlation_id=signal.correlation_id,
            source_model=signal.source_model,
            model_version=signal.model_version,
            generation_timestamp=signal.generation_time,
            routing_timestamp=start_time,
        )
        
        provenance.add_processing_step("delivery_started", start_time)
        
        try:
            # Pre-delivery validation
            validation_start = datetime.now()
            if not await self._validate_realtime_delivery(signal, strategy_id):
                raise RuntimeError("Real-time delivery validation failed")
            
            validation_end = datetime.now()
            provenance.add_processing_step(
                "validation_completed",
                validation_end,
                {"duration_ms": (validation_end - validation_start).total_seconds() * 1000}
            )
            
            # Attempt delivery with timeout
            delivery_start = datetime.now()
            
            try:
                await asyncio.wait_for(callback(signal), timeout=timeout)
                delivery_success = True
                delivery_error = None
                
            except asyncio.TimeoutError:
                delivery_success = False
                delivery_error = f"Delivery timeout after {timeout}s"
                
                # Fallback to cached signal
                await self._cache_signal_for_fallback(signal, strategy_id)
                
            except Exception as error:
                delivery_success = False
                delivery_error = str(error)
                
                # Fallback to cached signal
                await self._cache_signal_for_fallback(signal, strategy_id)
            
            delivery_end = datetime.now()
            
            # Record delivery attempt
            provenance.add_delivery_attempt(
                strategy_id,
                delivery_end,
                delivery_success,
                delivery_error
            )
            
            if delivery_success:
                provenance.final_delivery_timestamp = delivery_end
            
            # Calculate performance metrics
            provenance.calculate_latencies()
            
            # Store provenance record
            self._provenance_records[signal.signal_id] = provenance
            
            # Update performance metrics
            await self._update_performance_metrics(provenance)
            
            # Log performance
            self.logger.debug(
                "Real-time signal delivery completed",
                signal_id=signal.signal_id,
                strategy_id=strategy_id,
                success=delivery_success,
                total_latency_ms=provenance.total_latency_ms,
                delivery_latency_ms=provenance.delivery_latency_ms,
                error=delivery_error,
            )
            
            return {
                "success": delivery_success,
                "signal_id": signal.signal_id,
                "strategy_id": strategy_id,
                "total_latency_ms": provenance.total_latency_ms,
                "delivery_latency_ms": provenance.delivery_latency_ms,
                "processing_latency_ms": provenance.processing_latency_ms,
                "error": delivery_error,
                "provenance_id": provenance.correlation_id,
            }
            
        except Exception as error:
            # Record failed delivery
            provenance.add_delivery_attempt(
                strategy_id,
                datetime.now(),
                False,
                str(error)
            )
            
            self.logger.error(
                "Real-time signal delivery failed",
                signal_id=signal.signal_id,
                strategy_id=strategy_id,
                error=str(error),
            )
            
            return {
                "success": False,
                "signal_id": signal.signal_id,
                "strategy_id": strategy_id,
                "error": str(error),
                "provenance_id": provenance.correlation_id,
            }
    
    async def get_cached_signals(self, strategy_id: str) -> List[AISignal]:
        """
        Get cached signals for strategy (fallback mechanism).
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of cached signals
        """
        try:
            cached_signals = self._cached_signals.get(strategy_id, [])
            
            # Filter out expired signals
            now = datetime.now()
            valid_signals = [
                signal for signal in cached_signals
                if now - signal.timestamp < self._cache_ttl
            ]
            
            # Update cache
            self._cached_signals[strategy_id] = valid_signals
            
            return valid_signals
            
        except Exception as error:
            self.logger.error(
                "Failed to get cached signals",
                strategy_id=strategy_id,
                error=str(error),
            )
            return []
    
    async def get_signal_provenance(self, signal_id: str) -> Optional[SignalProvenance]:
        """
        Get signal provenance record.
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Signal provenance record if found
        """
        return self._provenance_records.get(signal_id)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time delivery performance metrics."""
        try:
            metrics = {
                "latency_targets": self._latency_targets,
                "current_performance": {},
                "sla_compliance": {},
            }
            
            # Calculate current performance
            for metric_name, values in self._delivery_metrics.items():
                if values:
                    recent_values = list(values)[-100:]  # Last 100 deliveries
                    metrics["current_performance"][metric_name] = {
                        "average": sum(recent_values) / len(recent_values),
                        "p95": sorted(recent_values)[int(len(recent_values) * 0.95)],
                        "p99": sorted(recent_values)[int(len(recent_values) * 0.99)],
                        "count": len(recent_values),
                    }
            
            # Calculate SLA compliance
            for target_name, target_value in self._latency_targets.items():
                metric_key = f"{target_name}_latency_ms"
                if metric_key in self._delivery_metrics:
                    values = list(self._delivery_metrics[metric_key])[-1000:]  # Last 1000
                    if values:
                        compliant_count = sum(1 for v in values if v <= target_value)
                        metrics["sla_compliance"][target_name] = {
                            "target_ms": target_value,
                            "compliance_rate": compliant_count / len(values),
                            "violations": len(values) - compliant_count,
                        }
            
            return metrics
            
        except Exception as error:
            self.logger.error(
                "Failed to get performance metrics",
                error=str(error),
            )
            return {}
    
    async def _validate_realtime_delivery(self, signal: AISignal, strategy_id: str) -> bool:
        """Validate signal for real-time delivery."""
        try:
            # Check signal age
            age_seconds = (datetime.now() - signal.timestamp).total_seconds()
            if age_seconds > 5.0:  # 5 second threshold for real-time
                self.logger.warning(
                    "Signal too old for real-time delivery",
                    signal_id=signal.signal_id,
                    age_seconds=age_seconds,
                )
                return False
            
            # Check confidence threshold
            if signal.confidence < 0.1:  # Minimum confidence for real-time
                self.logger.warning(
                    "Signal confidence too low for real-time delivery",
                    signal_id=signal.signal_id,
                    confidence=signal.confidence,
                )
                return False
            
            return True
            
        except Exception as error:
            self.logger.error(
                "Real-time delivery validation failed",
                signal_id=signal.signal_id,
                error=str(error),
            )
            return False
    
    async def _cache_signal_for_fallback(self, signal: AISignal, strategy_id: str) -> None:
        """Cache signal for fallback delivery."""
        try:
            self._cached_signals[strategy_id].append(signal)
            
            # Limit cache size
            if len(self._cached_signals[strategy_id]) > 100:
                self._cached_signals[strategy_id] = self._cached_signals[strategy_id][-50:]
            
            self.logger.debug(
                "Cached signal for fallback delivery",
                signal_id=signal.signal_id,
                strategy_id=strategy_id,
            )
            
        except Exception as error:
            self.logger.error(
                "Failed to cache signal for fallback",
                signal_id=signal.signal_id,
                error=str(error),
            )
    
    async def _update_performance_metrics(self, provenance: SignalProvenance) -> None:
        """Update performance metrics from provenance data."""
        try:
            if provenance.total_latency_ms is not None:
                self._delivery_metrics["total_latency_ms"].append(provenance.total_latency_ms)
            
            if provenance.processing_latency_ms is not None:
                self._delivery_metrics["processing_latency_ms"].append(provenance.processing_latency_ms)
            
            if provenance.delivery_latency_ms is not None:
                self._delivery_metrics["delivery_latency_ms"].append(provenance.delivery_latency_ms)
            
        except Exception as error:
            self.logger.error(
                "Failed to update performance metrics",
                error=str(error),
            )


class SignalRouterService:
    """
    Signal Router & Buffer service for AI signal management.
    
    This service provides:
    - Real-time signal routing from F5 to Nautilus strategies
    - Signal validation and format checking
    - Signal buffering for backtesting consistency
    - Delivery retry mechanisms with exponential backoff
    - Signal replay capabilities
    - Comprehensive audit trails and monitoring
    """
    
    def __init__(self, config: NautilusConfig):
        """
        Initialize signal router service.
        
        Args:
            config: NautilusTrader integration configuration
        """
        self.config = config
        self.logger = get_logger("nautilus_integration.signal_router")
        
        # Signal routing
        self._subscriptions: Dict[str, SignalSubscription] = {}
        self._strategy_callbacks: Dict[str, callable] = {}
        
        # Signal buffering
        self._signal_buffers: Dict[str, SignalBuffer] = {}
        self._buffer_max_size = config.signal_router.buffer_max_size_mb * 1024 * 1024
        
        # Delivery tracking
        self._delivery_records: Dict[str, SignalDeliveryRecord] = {}
        self._pending_deliveries: deque = deque()
        self._failed_deliveries: deque = deque(maxlen=1000)
        self._dead_letter_queue = DeadLetterQueue(max_size=config.signal_router.max_pending_deliveries)
        
        # Real-time delivery system
        self._realtime_delivery = RealTimeSignalDelivery(config)
        
        # Rate limiting
        self._signal_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Background tasks
        self._delivery_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._f5_heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        
        # F5 connection tracking
        self._f5_connection: Optional[Dict[str, Any]] = None
        
        self.logger.info(
            "Signal Router Service initialized",
            buffer_max_size_mb=config.signal_router.buffer_max_size_mb,
        )
    
    async def initialize(self) -> None:
        """Initialize signal router service."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Initializing Signal Router Service")
            
            try:
                # Start background tasks
                self._running = True
                self._delivery_task = asyncio.create_task(self._delivery_worker())
                self._cleanup_task = asyncio.create_task(self._cleanup_worker())
                
                # Initialize F5 connection
                await self._initialize_f5_connection()
                
                self.logger.info(
                    "Signal Router Service initialization completed",
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "initialize", "correlation_id": correlation_id},
                    "Failed to initialize Signal Router Service"
                )
                raise
    
    async def subscribe_strategy(
        self,
        strategy_id: str,
        signal_types: List[SignalType],
        callback: callable,
        instrument_ids: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        **kwargs
    ) -> str:
        """
        Subscribe strategy to AI signals.
        
        Args:
            strategy_id: Strategy identifier
            signal_types: List of signal types to subscribe to
            callback: Callback function for signal delivery
            instrument_ids: Optional list of instrument IDs (None = all)
            min_confidence: Minimum confidence threshold
            **kwargs: Additional subscription options
            
        Returns:
            Subscription ID
            
        Raises:
            ValueError: If subscription parameters are invalid
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Creating signal subscription",
                strategy_id=strategy_id,
                signal_types=[st.value for st in signal_types],
                instrument_ids=instrument_ids,
                min_confidence=min_confidence,
            )
            
            try:
                # Validate parameters
                if not strategy_id:
                    raise ValueError("Strategy ID is required")
                
                if not signal_types:
                    raise ValueError("At least one signal type is required")
                
                if not callable(callback):
                    raise ValueError("Callback must be callable")
                
                # Create subscription
                subscription = SignalSubscription(
                    strategy_id=strategy_id,
                    signal_types=signal_types,
                    instrument_ids=instrument_ids or [],
                    min_confidence=min_confidence,
                    **kwargs
                )
                
                # Store subscription and callback
                self._subscriptions[subscription.subscription_id] = subscription
                self._strategy_callbacks[strategy_id] = callback
                
                self.logger.info(
                    "Signal subscription created",
                    strategy_id=strategy_id,
                    subscription_id=subscription.subscription_id,
                    correlation_id=correlation_id,
                )
                
                return subscription.subscription_id
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "subscribe_strategy",
                        "strategy_id": strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to create signal subscription"
                )
                raise
    
    async def unsubscribe_strategy(self, strategy_id: str) -> None:
        """
        Unsubscribe strategy from AI signals.
        
        Args:
            strategy_id: Strategy identifier
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Unsubscribing strategy from signals",
                strategy_id=strategy_id,
            )
            
            try:
                # Remove subscriptions for strategy
                subscriptions_to_remove = [
                    sub_id for sub_id, sub in self._subscriptions.items()
                    if sub.strategy_id == strategy_id
                ]
                
                for sub_id in subscriptions_to_remove:
                    del self._subscriptions[sub_id]
                
                # Remove callback
                self._strategy_callbacks.pop(strategy_id, None)
                
                self.logger.info(
                    "Strategy unsubscribed from signals",
                    strategy_id=strategy_id,
                    removed_subscriptions=len(subscriptions_to_remove),
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "unsubscribe_strategy",
                        "strategy_id": strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to unsubscribe strategy"
                )
                raise
    
    async def route_signal(self, signal: AISignal) -> Dict[str, Any]:
        """
        Route AI signal to subscribed strategies.
        
        Args:
            signal: AI signal to route
            
        Returns:
            Routing results with delivery status
            
        Raises:
            ValueError: If signal is invalid
        """
        with with_correlation_id() as correlation_id:
            self.logger.debug(
                "Routing AI signal",
                signal_id=signal.signal_id,
                signal_type=signal.signal_type.value,
                instrument_id=signal.instrument_id,
                confidence=signal.confidence,
            )
            
            try:
                # Validate signal
                validation_result = await self._validate_signal(signal)
                if not validation_result["valid"]:
                    raise ValueError(f"Signal validation failed: {validation_result['errors']}")
                
                # Buffer signal for replay
                await self._buffer_signal(signal)
                
                # Find matching subscriptions
                matching_subscriptions = self._find_matching_subscriptions(signal)
                
                routing_results = {
                    "signal_id": signal.signal_id,
                    "total_subscriptions": len(matching_subscriptions),
                    "deliveries": [],
                    "errors": [],
                }
                
                # Route to matching strategies
                for subscription in matching_subscriptions:
                    try:
                        # Check rate limits
                        if not self._check_rate_limit(subscription.strategy_id, subscription):
                            self.logger.warning(
                                "Rate limit exceeded for strategy",
                                strategy_id=subscription.strategy_id,
                                signal_id=signal.signal_id,
                            )
                            continue
                        
                        # Create delivery record
                        delivery_record = SignalDeliveryRecord(
                            signal_id=signal.signal_id,
                            strategy_id=subscription.strategy_id,
                        )
                        
                        self._delivery_records[delivery_record.delivery_id] = delivery_record
                        
                        # Queue for delivery
                        self._pending_deliveries.append({
                            "signal": signal,
                            "subscription": subscription,
                            "delivery_record": delivery_record,
                        })
                        
                        routing_results["deliveries"].append({
                            "strategy_id": subscription.strategy_id,
                            "delivery_id": delivery_record.delivery_id,
                            "status": "queued",
                        })
                        
                    except Exception as error:
                        error_msg = f"Failed to queue signal for {subscription.strategy_id}: {error}"
                        routing_results["errors"].append(error_msg)
                        
                        self.logger.error(
                            "Failed to queue signal delivery",
                            strategy_id=subscription.strategy_id,
                            signal_id=signal.signal_id,
                            error=str(error),
                        )
                
                self.logger.debug(
                    "Signal routing completed",
                    signal_id=signal.signal_id,
                    deliveries_queued=len(routing_results["deliveries"]),
                    errors=len(routing_results["errors"]),
                )
                
                return routing_results
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "route_signal",
                        "signal_id": signal.signal_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to route signal"
                )
                raise
    
    async def buffer_signal(self, signal: AISignal) -> None:
        """
        Buffer signal for backtesting replay.
        
        Args:
            signal: AI signal to buffer
        """
        await self._buffer_signal(signal)
    
    async def replay_signals(
        self,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        signal_types: Optional[List[SignalType]] = None
    ) -> List[AISignal]:
        """
        Replay buffered signals for backtesting.
        
        Args:
            instrument_id: Instrument identifier
            start_time: Replay start time
            end_time: Replay end time
            signal_types: Optional signal type filter
            
        Returns:
            List of signals in chronological order
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Replaying buffered signals",
                instrument_id=instrument_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                signal_types=[st.value for st in signal_types] if signal_types else None,
            )
            
            try:
                replayed_signals = []
                
                # Find relevant buffers
                for buffer in self._signal_buffers.values():
                    if buffer.instrument_id == instrument_id:
                        # Get signals in range
                        range_signals = buffer.get_signals_in_range(start_time, end_time)
                        
                        # Filter by signal types if specified
                        if signal_types:
                            range_signals = [
                                signal for signal in range_signals
                                if signal.signal_type in signal_types
                            ]
                        
                        replayed_signals.extend(range_signals)
                
                # Sort by timestamp
                replayed_signals.sort(key=lambda s: s.timestamp)
                
                self.logger.info(
                    "Signal replay completed",
                    instrument_id=instrument_id,
                    signals_count=len(replayed_signals),
                    correlation_id=correlation_id,
                )
                
                return replayed_signals
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "replay_signals",
                        "instrument_id": instrument_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to replay signals"
                )
                raise
    
    async def get_signal_statistics(
        self,
        instrument_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get signal statistics and metrics.
        
        Args:
            instrument_id: Optional instrument filter
            time_range: Optional time range filter
            
        Returns:
            Signal statistics
        """
        try:
            stats = {
                "total_signals": 0,
                "signals_by_type": defaultdict(int),
                "signals_by_confidence": defaultdict(int),
                "delivery_success_rate": 0.0,
                "average_delivery_latency_ms": 0.0,
                "buffer_statistics": {},
                "dead_letter_queue_statistics": self._dead_letter_queue.get_statistics(),
            }
            
            # Count buffered signals
            for buffer in self._signal_buffers.values():
                if instrument_id and buffer.instrument_id != instrument_id:
                    continue
                
                signals = buffer.signals
                if time_range:
                    start_time, end_time = time_range
                    signals = [s for s in signals if start_time <= s.timestamp <= end_time]
                
                stats["total_signals"] += len(signals)
                
                for signal in signals:
                    stats["signals_by_type"][signal.signal_type.value] += 1
                    stats["signals_by_confidence"][signal.confidence_level.value] += 1
            
            # Calculate delivery statistics
            successful_deliveries = [
                record for record in self._delivery_records.values()
                if record.status == DeliveryStatus.DELIVERED
            ]
            
            total_deliveries = len(self._delivery_records)
            if total_deliveries > 0:
                stats["delivery_success_rate"] = len(successful_deliveries) / total_deliveries
                
                # Calculate average latency
                latencies = [
                    record.delivery_latency_ms for record in successful_deliveries
                    if record.delivery_latency_ms is not None
                ]
                if latencies:
                    stats["average_delivery_latency_ms"] = sum(latencies) / len(latencies)
            
            # Buffer statistics
            stats["buffer_statistics"] = {
                "total_buffers": len(self._signal_buffers),
                "total_buffer_size_mb": sum(
                    buffer.total_size_bytes for buffer in self._signal_buffers.values()
                ) / (1024 * 1024),
            }
            
            return stats
            
        except Exception as error:
            self.logger.error(
                "Failed to get signal statistics",
                error=str(error),
            )
            return {}
    
    async def get_failed_deliveries(
        self,
        limit: Optional[int] = None,
        strategy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get failed signal deliveries from dead letter queue.
        
        Args:
            limit: Optional limit on number of results
            strategy_id: Optional strategy filter
            
        Returns:
            List of failed deliveries
        """
        return self._dead_letter_queue.get_failed_deliveries(limit, strategy_id)
    
    async def retry_failed_delivery(self, delivery_id: str) -> bool:
        """
        Retry a failed signal delivery.
        
        Args:
            delivery_id: Delivery record ID to retry
            
        Returns:
            True if retry was queued, False otherwise
        """
        try:
            # Find failed delivery in dead letter queue
            failed_deliveries = self._dead_letter_queue.get_failed_deliveries()
            
            for failed_item in failed_deliveries:
                delivery_record = failed_item["delivery_record"]
                if delivery_record.delivery_id == delivery_id:
                    # Reset delivery record for retry
                    delivery_record.status = DeliveryStatus.PENDING
                    delivery_record.attempt_count = 0
                    delivery_record.last_error = None
                    
                    # Find matching subscription
                    matching_subscription = None
                    for subscription in self._subscriptions.values():
                        if subscription.strategy_id == delivery_record.strategy_id:
                            matching_subscription = subscription
                            break
                    
                    if matching_subscription:
                        # Queue for retry
                        self._pending_deliveries.append({
                            "signal": failed_item["signal"],
                            "subscription": matching_subscription,
                            "delivery_record": delivery_record,
                        })
                        
                        self.logger.info(
                            "Queued failed delivery for retry",
                            delivery_id=delivery_id,
                            signal_id=failed_item["signal"].signal_id,
                            strategy_id=delivery_record.strategy_id,
                        )
                        
                        return True
            
            return False
            
        except Exception as error:
            self.logger.error(
                "Failed to retry delivery",
                delivery_id=delivery_id,
                error=str(error),
            )
            return False
    
    async def deliver_signal_realtime(
        self,
        signal: AISignal,
        strategy_id: str,
        timeout: float = 0.5
    ) -> Dict[str, Any]:
        """
        Deliver signal with real-time performance guarantees.
        
        Args:
            signal: AI signal to deliver
            strategy_id: Target strategy ID
            timeout: Delivery timeout in seconds
            
        Returns:
            Delivery result with performance metrics
        """
        try:
            # Get strategy callback
            callback = self._strategy_callbacks.get(strategy_id)
            if not callback:
                raise RuntimeError(f"No callback found for strategy {strategy_id}")
            
            # Use real-time delivery system
            return await self._realtime_delivery.deliver_signal_realtime(
                signal, strategy_id, callback, timeout
            )
            
        except Exception as error:
            self.logger.error(
                "Real-time signal delivery failed",
                signal_id=signal.signal_id,
                strategy_id=strategy_id,
                error=str(error),
            )
            return {
                "success": False,
                "signal_id": signal.signal_id,
                "strategy_id": strategy_id,
                "error": str(error),
            }
    
    async def get_signal_provenance(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get signal provenance and audit trail.
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Signal provenance data if found
        """
        try:
            provenance = await self._realtime_delivery.get_signal_provenance(signal_id)
            if provenance:
                return provenance.model_dump()
            return None
            
        except Exception as error:
            self.logger.error(
                "Failed to get signal provenance",
                signal_id=signal_id,
                error=str(error),
            )
            return None
    
    async def get_cached_signals(self, strategy_id: str) -> List[AISignal]:
        """
        Get cached signals for fallback delivery.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of cached signals
        """
        return await self._realtime_delivery.get_cached_signals(strategy_id)
    
    async def get_realtime_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time delivery performance metrics."""
        return await self._realtime_delivery.get_performance_metrics()
    
    async def shutdown(self) -> None:
        """Shutdown signal router service."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Shutting down Signal Router Service")
            
            try:
                # Stop background tasks
                self._running = False
                
                if self._delivery_task:
                    self._delivery_task.cancel()
                    try:
                        await self._delivery_task
                    except asyncio.CancelledError:
                        pass
                
                if self._cleanup_task:
                    self._cleanup_task.cancel()
                    try:
                        await self._cleanup_task
                    except asyncio.CancelledError:
                        pass
                
                if self._f5_heartbeat_task:
                    self._f5_heartbeat_task.cancel()
                    try:
                        await self._f5_heartbeat_task
                    except asyncio.CancelledError:
                        pass
                
                # Clear data structures
                self._subscriptions.clear()
                self._strategy_callbacks.clear()
                self._signal_buffers.clear()
                self._delivery_records.clear()
                self._pending_deliveries.clear()
                
                self.logger.info(
                    "Signal Router Service shutdown completed",
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "shutdown", "correlation_id": correlation_id},
                    "Error during Signal Router Service shutdown"
                )
    
    # Private helper methods
    
    async def _initialize_f5_connection(self) -> None:
        """Initialize connection to F5 Intelligence Layer."""
        try:
            self.logger.debug("Initializing F5 Intelligence Layer connection")
            
            # Initialize F5 connection configuration
            f5_config = {
                "enabled": self.config.signal_router.f5_connection_enabled,
                "timeout": self.config.signal_router.f5_connection_timeout,
                "heartbeat_interval": self.config.signal_router.f5_heartbeat_interval,
            }
            
            if not f5_config["enabled"]:
                self.logger.info("F5 connection disabled in configuration")
                return
            
            # In a real implementation, this would:
            # 1. Establish WebSocket or HTTP connection to F5 RAG service
            # 2. Authenticate with F5 service
            # 3. Set up signal subscription channels
            # 4. Initialize heartbeat mechanism
            # 5. Set up reconnection logic
            
            # For now, create a mock connection
            self._f5_connection = {
                "status": "connected",
                "connected_at": datetime.now(),
                "heartbeat_count": 0,
                "last_heartbeat": datetime.now(),
            }
            
            # Start heartbeat task if connection is enabled
            if f5_config["enabled"]:
                self._f5_heartbeat_task = asyncio.create_task(self._f5_heartbeat_worker())
            
            self.logger.info(
                "F5 Intelligence Layer connection initialized",
                enabled=f5_config["enabled"],
                timeout=f5_config["timeout"],
                heartbeat_interval=f5_config["heartbeat_interval"],
            )
            
        except Exception as error:
            self.logger.warning(
                "Failed to initialize F5 connection",
                error=str(error),
            )
            # Don't fail initialization for this - graceful degradation
            self._f5_connection = {
                "status": "failed",
                "error": str(error),
                "failed_at": datetime.now(),
            }
    
    async def _f5_heartbeat_worker(self) -> None:
        """Background worker for F5 connection heartbeat."""
        while self._running and self.config.signal_router.f5_connection_enabled:
            try:
                await asyncio.sleep(self.config.signal_router.f5_heartbeat_interval)
                
                # Send heartbeat to F5 service
                if self._f5_connection and self._f5_connection.get("status") == "connected":
                    # In a real implementation, this would send actual heartbeat
                    self._f5_connection["heartbeat_count"] += 1
                    self._f5_connection["last_heartbeat"] = datetime.now()
                    
                    self.logger.debug(
                        "F5 heartbeat sent",
                        heartbeat_count=self._f5_connection["heartbeat_count"],
                    )
                
            except Exception as error:
                self.logger.error(
                    "Error in F5 heartbeat worker",
                    error=str(error),
                )
                await asyncio.sleep(5.0)  # Wait before retry
    
    async def _validate_signal(self, signal: AISignal) -> Dict[str, Any]:
        """Validate AI signal format and content."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        try:
            # Check signal expiry
            if signal.is_expired():
                validation_result["valid"] = False
                validation_result["errors"].append("Signal has expired")
            
            # Validate confidence
            if not 0.0 <= signal.confidence <= 1.0:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid confidence: {signal.confidence}")
            
            # Validate instrument ID
            if not signal.instrument_id:
                validation_result["valid"] = False
                validation_result["errors"].append("Instrument ID is required")
            
            # Validate source model
            if not signal.source_model:
                validation_result["valid"] = False
                validation_result["errors"].append("Source model is required")
            
            # Signal-specific validation
            if signal.signal_type == SignalType.REGIME_PREDICTION:
                if not isinstance(signal.value, str):
                    validation_result["warnings"].append("Regime prediction should be string")
            
            elif signal.signal_type == SignalType.VOLATILITY_FORECAST:
                if not isinstance(signal.value, (int, float)):
                    validation_result["warnings"].append("Volatility forecast should be numeric")
            
        except Exception as error:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {error}")
        
        return validation_result
    
    def _find_matching_subscriptions(self, signal: AISignal) -> List[SignalSubscription]:
        """Find subscriptions that match the signal."""
        matching_subscriptions = []
        
        for subscription in self._subscriptions.values():
            # Check signal type
            if signal.signal_type not in subscription.signal_types:
                continue
            
            # Check instrument ID
            if subscription.instrument_ids and signal.instrument_id not in subscription.instrument_ids:
                continue
            
            # Check confidence threshold
            if signal.confidence < subscription.min_confidence:
                continue
            
            matching_subscriptions.append(subscription)
        
        return matching_subscriptions
    
    def _check_rate_limit(self, strategy_id: str, subscription: SignalSubscription) -> bool:
        """Check if strategy is within rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        signal_times = self._signal_counts[strategy_id]
        while signal_times and signal_times[0] < minute_ago:
            signal_times.popleft()
        
        # Check limit
        if len(signal_times) >= subscription.max_signals_per_minute:
            return False
        
        # Add current signal
        signal_times.append(now)
        return True
    
    async def _buffer_signal(self, signal: AISignal) -> None:
        """Buffer signal for replay."""
        try:
            # Find or create buffer for instrument
            buffer_key = f"{signal.instrument_id}_{signal.timestamp.date()}"
            
            if buffer_key not in self._signal_buffers:
                # Create new buffer for the day
                start_of_day = signal.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = start_of_day + timedelta(days=1)
                
                self._signal_buffers[buffer_key] = SignalBuffer(
                    instrument_id=signal.instrument_id,
                    start_time=start_of_day,
                    end_time=end_of_day,
                )
            
            # Add signal to buffer
            buffer = self._signal_buffers[buffer_key]
            buffer.add_signal(signal)
            
            # Check buffer size limits
            await self._manage_buffer_size()
            
        except Exception as error:
            self.logger.warning(
                "Failed to buffer signal",
                signal_id=signal.signal_id,
                error=str(error),
            )
    
    async def _manage_buffer_size(self) -> None:
        """Manage buffer size to stay within limits."""
        total_size = sum(buffer.total_size_bytes for buffer in self._signal_buffers.values())
        
        if total_size > self._buffer_max_size:
            # Remove oldest buffers
            sorted_buffers = sorted(
                self._signal_buffers.items(),
                key=lambda x: x[1].created_at
            )
            
            while total_size > self._buffer_max_size * 0.8 and sorted_buffers:
                buffer_key, buffer = sorted_buffers.pop(0)
                total_size -= buffer.total_size_bytes
                del self._signal_buffers[buffer_key]
                
                self.logger.debug(
                    "Removed old signal buffer",
                    buffer_id=buffer.buffer_id,
                    instrument_id=buffer.instrument_id,
                )
    
    async def _delivery_worker(self) -> None:
        """Background worker for signal delivery."""
        while self._running:
            try:
                if self._pending_deliveries:
                    delivery_item = self._pending_deliveries.popleft()
                    await self._deliver_signal(
                        delivery_item["signal"],
                        delivery_item["subscription"],
                        delivery_item["delivery_record"]
                    )
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as error:
                self.logger.error(
                    "Error in delivery worker",
                    error=str(error),
                )
                await asyncio.sleep(1.0)
    
    async def _deliver_signal(
        self,
        signal: AISignal,
        subscription: SignalSubscription,
        delivery_record: SignalDeliveryRecord
    ) -> None:
        """Deliver signal to strategy."""
        start_time = datetime.now()
        
        try:
            delivery_record.attempt_count += 1
            
            # Get strategy callback
            callback = self._strategy_callbacks.get(subscription.strategy_id)
            if not callback:
                raise RuntimeError(f"No callback found for strategy {subscription.strategy_id}")
            
            # Deliver signal with timeout
            try:
                await asyncio.wait_for(
                    callback(signal),
                    timeout=subscription.delivery_timeout
                )
                
                # Mark as delivered
                delivery_record.status = DeliveryStatus.DELIVERED
                delivery_record.delivered_at = datetime.now()
                delivery_record.delivery_latency_ms = (
                    delivery_record.delivered_at - start_time
                ).total_seconds() * 1000
                
                self.logger.debug(
                    "Signal delivered successfully",
                    signal_id=signal.signal_id,
                    strategy_id=subscription.strategy_id,
                    delivery_id=delivery_record.delivery_id,
                    latency_ms=delivery_record.delivery_latency_ms,
                )
                
            except asyncio.TimeoutError:
                raise RuntimeError("Delivery timeout")
            
        except Exception as error:
            delivery_record.status = DeliveryStatus.FAILED
            delivery_record.last_error = str(error)
            
            self.logger.warning(
                "Signal delivery failed",
                signal_id=signal.signal_id,
                strategy_id=subscription.strategy_id,
                delivery_id=delivery_record.delivery_id,
                attempt=delivery_record.attempt_count,
                error=str(error),
            )
            
            # Retry if attempts remaining
            if delivery_record.attempt_count < subscription.retry_attempts:
                # Add back to queue with delay
                await asyncio.sleep(subscription.retry_delay * delivery_record.attempt_count)
                self._pending_deliveries.append({
                    "signal": signal,
                    "subscription": subscription,
                    "delivery_record": delivery_record,
                })
            else:
                # Move to failed deliveries and dead letter queue
                self._failed_deliveries.append(delivery_record)
                self._dead_letter_queue.add_failed_delivery(delivery_record, signal)
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleanup tasks."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up expired delivery records
                now = datetime.now()
                expired_cutoff = now - timedelta(hours=24)
                
                expired_records = [
                    record_id for record_id, record in self._delivery_records.items()
                    if record.created_at < expired_cutoff
                ]
                
                for record_id in expired_records:
                    del self._delivery_records[record_id]
                
                if expired_records:
                    self.logger.debug(
                        "Cleaned up expired delivery records",
                        count=len(expired_records),
                    )
                
            except Exception as error:
                self.logger.error(
                    "Error in cleanup worker",
                    error=str(error),
                )


class F5SignalClient:
    """Client for receiving signals from F5 Intelligence Layer."""
    
    def __init__(self, signal_router: Optional["SignalRouterService"] = None):
        """
        Initialize F5 signal client.
        
        Args:
            signal_router: Optional signal router service
        """
        self.signal_router = signal_router
        self.logger = get_logger("nautilus_integration.f5_signal_client")
        
        # Local signal cache
        self._signal_cache: Dict[str, List[AISignal]] = defaultdict(list)
        self._cache_ttl = timedelta(minutes=5)
    
    async def subscribe(self, signal_type: SignalType, callback: callable) -> None:
        """Subscribe to signal type."""
        if self.signal_router:
            # Use signal router for subscription
            pass
        else:
            # Direct F5 subscription
            pass
    
    async def get_current_signals(self, instrument_id: str) -> List[AISignal]:
        """Get current signals for instrument."""
        # Check cache first
        cached_signals = self._signal_cache.get(instrument_id, [])
        
        # Filter out expired signals
        now = datetime.now()
        valid_signals = [
            signal for signal in cached_signals
            if now - signal.timestamp < self._cache_ttl
        ]
        
        return valid_signals
    
    async def get_signal_by_type(
        self,
        instrument_id: str,
        signal_type: SignalType
    ) -> Optional[AISignal]:
        """Get latest signal of specific type for instrument."""
        signals = await self.get_current_signals(instrument_id)
        
        # Find latest signal of requested type
        type_signals = [s for s in signals if s.signal_type == signal_type]
        if type_signals:
            return max(type_signals, key=lambda s: s.timestamp)
        
        return None
    
    # Additional methods for comprehensive monitoring and performance tracking
    
    async def get_delivery_metrics(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive delivery metrics and performance statistics.
        
        Args:
            time_window: Optional time window for metrics (default: last hour)
            
        Returns:
            Comprehensive delivery metrics
        """
        try:
            if time_window is None:
                time_window = timedelta(hours=1)
            
            cutoff_time = datetime.now() - time_window
            
            # Filter records within time window
            recent_records = [
                record for record in self._delivery_records.values()
                if record.created_at >= cutoff_time
            ]
            
            metrics = {
                "time_window_hours": time_window.total_seconds() / 3600,
                "total_deliveries": len(recent_records),
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "pending_deliveries": 0,
                "average_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "delivery_rate_per_minute": 0.0,
                "error_rate": 0.0,
                "retry_rate": 0.0,
                "strategies_served": set(),
                "signal_types_delivered": defaultdict(int),
                "top_error_reasons": defaultdict(int),
            }
            
            if not recent_records:
                return metrics
            
            # Calculate basic metrics
            successful_records = []
            failed_records = []
            pending_records = []
            
            for record in recent_records:
                metrics["strategies_served"].add(record.strategy_id)
                
                if record.status == DeliveryStatus.DELIVERED:
                    successful_records.append(record)
                    metrics["successful_deliveries"] += 1
                elif record.status == DeliveryStatus.FAILED:
                    failed_records.append(record)
                    metrics["failed_deliveries"] += 1
                    
                    if record.last_error:
                        metrics["top_error_reasons"][record.last_error] += 1
                elif record.status == DeliveryStatus.PENDING:
                    pending_records.append(record)
                    metrics["pending_deliveries"] += 1
                
                # Count retries
                if record.attempt_count > 1:
                    metrics["retry_rate"] += 1
            
            # Calculate rates
            total_deliveries = len(recent_records)
            if total_deliveries > 0:
                metrics["error_rate"] = len(failed_records) / total_deliveries
                metrics["retry_rate"] = metrics["retry_rate"] / total_deliveries
                
                # Delivery rate per minute
                time_minutes = time_window.total_seconds() / 60
                metrics["delivery_rate_per_minute"] = total_deliveries / time_minutes
            
            # Calculate latency metrics
            latencies = [
                record.delivery_latency_ms for record in successful_records
                if record.delivery_latency_ms is not None
            ]
            
            if latencies:
                latencies.sort()
                metrics["average_latency_ms"] = sum(latencies) / len(latencies)
                metrics["p95_latency_ms"] = latencies[int(len(latencies) * 0.95)]
                metrics["p99_latency_ms"] = latencies[int(len(latencies) * 0.99)]
            
            # Convert sets to counts
            metrics["strategies_served"] = len(metrics["strategies_served"])
            metrics["top_error_reasons"] = dict(metrics["top_error_reasons"])
            metrics["signal_types_delivered"] = dict(metrics["signal_types_delivered"])
            
            return metrics
            
        except Exception as error:
            self.logger.error(
                "Failed to get delivery metrics",
                error=str(error),
            )
            return {}
    
    async def get_subscription_statistics(self) -> Dict[str, Any]:
        """Get statistics about active subscriptions."""
        try:
            stats = {
                "total_subscriptions": len(self._subscriptions),
                "active_strategies": len(self._strategy_callbacks),
                "subscriptions_by_signal_type": defaultdict(int),
                "subscriptions_by_strategy": defaultdict(int),
                "average_confidence_threshold": 0.0,
                "subscription_details": [],
            }
            
            confidence_thresholds = []
            
            for subscription in self._subscriptions.values():
                # Count by signal type
                for signal_type in subscription.signal_types:
                    stats["subscriptions_by_signal_type"][signal_type.value] += 1
                
                # Count by strategy
                stats["subscriptions_by_strategy"][subscription.strategy_id] += 1
                
                # Collect confidence thresholds
                confidence_thresholds.append(subscription.min_confidence)
                
                # Add subscription details
                stats["subscription_details"].append({
                    "subscription_id": subscription.subscription_id,
                    "strategy_id": subscription.strategy_id,
                    "signal_types": [st.value for st in subscription.signal_types],
                    "min_confidence": subscription.min_confidence,
                    "instrument_count": len(subscription.instrument_ids),
                    "max_signals_per_minute": subscription.max_signals_per_minute,
                })
            
            # Calculate average confidence threshold
            if confidence_thresholds:
                stats["average_confidence_threshold"] = sum(confidence_thresholds) / len(confidence_thresholds)
            
            # Convert defaultdicts to regular dicts
            stats["subscriptions_by_signal_type"] = dict(stats["subscriptions_by_signal_type"])
            stats["subscriptions_by_strategy"] = dict(stats["subscriptions_by_strategy"])
            
            return stats
            
        except Exception as error:
            self.logger.error(
                "Failed to get subscription statistics",
                error=str(error),
            )
            return {}
    
    async def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get statistics about signal buffers."""
        try:
            stats = {
                "total_buffers": len(self._signal_buffers),
                "total_signals_buffered": 0,
                "total_buffer_size_mb": 0.0,
                "buffers_by_instrument": defaultdict(int),
                "oldest_buffer_age_hours": 0.0,
                "newest_buffer_age_hours": 0.0,
                "buffer_details": [],
            }
            
            if not self._signal_buffers:
                return stats
            
            now = datetime.now()
            buffer_ages = []
            
            for buffer in self._signal_buffers.values():
                stats["total_signals_buffered"] += buffer.signal_count
                stats["total_buffer_size_mb"] += buffer.total_size_bytes / (1024 * 1024)
                stats["buffers_by_instrument"][buffer.instrument_id] += 1
                
                # Calculate buffer age
                age_hours = (now - buffer.created_at).total_seconds() / 3600
                buffer_ages.append(age_hours)
                
                stats["buffer_details"].append({
                    "buffer_id": buffer.buffer_id,
                    "instrument_id": buffer.instrument_id,
                    "signal_count": buffer.signal_count,
                    "size_mb": buffer.total_size_bytes / (1024 * 1024),
                    "age_hours": age_hours,
                    "start_time": buffer.start_time.isoformat(),
                    "end_time": buffer.end_time.isoformat(),
                })
            
            # Calculate age statistics
            if buffer_ages:
                stats["oldest_buffer_age_hours"] = max(buffer_ages)
                stats["newest_buffer_age_hours"] = min(buffer_ages)
            
            # Convert defaultdict to regular dict
            stats["buffers_by_instrument"] = dict(stats["buffers_by_instrument"])
            
            return stats
            
        except Exception as error:
            self.logger.error(
                "Failed to get buffer statistics",
                error=str(error),
            )
            return {}
    
    async def cleanup_expired_data(self, max_age_hours: float = 24.0) -> Dict[str, int]:
        """
        Clean up expired data from buffers and delivery records.
        
        Args:
            max_age_hours: Maximum age in hours for data retention
            
        Returns:
            Cleanup statistics
        """
        try:
            self.logger.info(
                "Starting expired data cleanup",
                max_age_hours=max_age_hours,
            )
            
            cleanup_stats = {
                "buffers_removed": 0,
                "delivery_records_removed": 0,
                "signals_removed": 0,
                "bytes_freed": 0,
            }
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean up old buffers
            expired_buffer_keys = []
            for buffer_key, buffer in self._signal_buffers.items():
                if buffer.created_at < cutoff_time:
                    expired_buffer_keys.append(buffer_key)
                    cleanup_stats["signals_removed"] += buffer.signal_count
                    cleanup_stats["bytes_freed"] += buffer.total_size_bytes
            
            for buffer_key in expired_buffer_keys:
                del self._signal_buffers[buffer_key]
                cleanup_stats["buffers_removed"] += 1
            
            # Clean up old delivery records
            expired_record_ids = []
            for record_id, record in self._delivery_records.items():
                if record.created_at < cutoff_time:
                    expired_record_ids.append(record_id)
            
            for record_id in expired_record_ids:
                del self._delivery_records[record_id]
                cleanup_stats["delivery_records_removed"] += 1
            
            self.logger.info(
                "Expired data cleanup completed",
                **cleanup_stats,
            )
            
            return cleanup_stats
            
        except Exception as error:
            self.logger.error(
                "Failed to cleanup expired data",
                error=str(error),
            )
            return {}
    
    async def optimize_buffer_storage(self) -> Dict[str, Any]:
        """
        Optimize buffer storage by compacting and reorganizing data.
        
        Returns:
            Optimization results
        """
        try:
            self.logger.info("Starting buffer storage optimization")
            
            optimization_results = {
                "buffers_optimized": 0,
                "signals_compacted": 0,
                "bytes_saved": 0,
                "optimization_time_ms": 0.0,
            }
            
            start_time = datetime.now()
            
            for buffer in self._signal_buffers.values():
                try:
                    # Remove duplicate signals
                    original_count = len(buffer.signals)
                    original_size = buffer.total_size_bytes
                    
                    # Remove duplicates based on signal_id
                    seen_ids = set()
                    unique_signals = []
                    
                    for signal in buffer.signals:
                        if signal.signal_id not in seen_ids:
                            unique_signals.append(signal)
                            seen_ids.add(signal.signal_id)
                    
                    # Update buffer
                    buffer.signals = unique_signals
                    buffer.signal_count = len(unique_signals)
                    
                    # Recalculate size
                    new_size = sum(len(json.dumps(signal.to_dict())) for signal in unique_signals)
                    buffer.total_size_bytes = new_size
                    
                    # Update statistics
                    if original_count != len(unique_signals):
                        optimization_results["buffers_optimized"] += 1
                        optimization_results["signals_compacted"] += (original_count - len(unique_signals))
                        optimization_results["bytes_saved"] += (original_size - new_size)
                    
                except Exception as error:
                    self.logger.warning(
                        "Failed to optimize buffer",
                        buffer_id=buffer.buffer_id,
                        error=str(error),
                    )
                    continue
            
            end_time = datetime.now()
            optimization_results["optimization_time_ms"] = (end_time - start_time).total_seconds() * 1000
            
            self.logger.info(
                "Buffer storage optimization completed",
                **optimization_results,
            )
            
            return optimization_results
            
        except Exception as error:
            self.logger.error(
                "Failed to optimize buffer storage",
                error=str(error),
            )
            return {}
    
    async def export_signal_data(
        self,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        output_format: str = "json"
    ) -> Optional[str]:
        """
        Export signal data for analysis or backup.
        
        Args:
            instrument_id: Instrument identifier
            start_time: Export start time
            end_time: Export end time
            output_format: Output format ("json" or "csv")
            
        Returns:
            Path to exported file if successful
        """
        try:
            self.logger.info(
                "Exporting signal data",
                instrument_id=instrument_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                format=output_format,
            )
            
            # Collect signals from buffers
            signals_to_export = []
            
            for buffer in self._signal_buffers.values():
                if buffer.instrument_id == instrument_id:
                    range_signals = buffer.get_signals_in_range(start_time, end_time)
                    signals_to_export.extend(range_signals)
            
            if not signals_to_export:
                self.logger.warning(
                    "No signals found for export",
                    instrument_id=instrument_id,
                )
                return None
            
            # Sort by timestamp
            signals_to_export.sort(key=lambda s: s.timestamp)
            
            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signals_{instrument_id}_{timestamp_str}.{output_format}"
            
            # Export data
            if output_format == "json":
                export_data = [signal.to_dict() for signal in signals_to_export]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif output_format == "csv":
                import csv
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    if signals_to_export:
                        fieldnames = signals_to_export[0].to_dict().keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for signal in signals_to_export:
                            writer.writerow(signal.to_dict())
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            self.logger.info(
                "Signal data exported successfully",
                filename=filename,
                signals_count=len(signals_to_export),
            )
            
            return filename
            
        except Exception as error:
            self.logger.error(
                "Failed to export signal data",
                instrument_id=instrument_id,
                error=str(error),
            )
            return None
    
    async def validate_signal_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of buffered signals and delivery records.
        
        Returns:
            Validation results
        """
        try:
            self.logger.info("Starting signal integrity validation")
            
            validation_results = {
                "total_signals_checked": 0,
                "corrupted_signals": 0,
                "missing_fields": 0,
                "invalid_timestamps": 0,
                "duplicate_signals": 0,
                "validation_errors": [],
            }
            
            seen_signal_ids = set()
            
            # Validate buffered signals
            for buffer in self._signal_buffers.values():
                for signal in buffer.signals:
                    validation_results["total_signals_checked"] += 1
                    
                    try:
                        # Check for duplicates
                        if signal.signal_id in seen_signal_ids:
                            validation_results["duplicate_signals"] += 1
                            validation_results["validation_errors"].append(
                                f"Duplicate signal ID: {signal.signal_id}"
                            )
                        else:
                            seen_signal_ids.add(signal.signal_id)
                        
                        # Validate required fields
                        required_fields = ["signal_id", "signal_type", "instrument_id", "timestamp", "confidence"]
                        for field in required_fields:
                            if not hasattr(signal, field) or getattr(signal, field) is None:
                                validation_results["missing_fields"] += 1
                                validation_results["validation_errors"].append(
                                    f"Missing field {field} in signal {signal.signal_id}"
                                )
                        
                        # Validate timestamp
                        if hasattr(signal, 'timestamp'):
                            if not isinstance(signal.timestamp, datetime):
                                validation_results["invalid_timestamps"] += 1
                                validation_results["validation_errors"].append(
                                    f"Invalid timestamp type in signal {signal.signal_id}"
                                )
                        
                        # Validate confidence range
                        if hasattr(signal, 'confidence'):
                            if not (0.0 <= signal.confidence <= 1.0):
                                validation_results["validation_errors"].append(
                                    f"Invalid confidence value {signal.confidence} in signal {signal.signal_id}"
                                )
                        
                    except Exception as error:
                        validation_results["corrupted_signals"] += 1
                        validation_results["validation_errors"].append(
                            f"Corrupted signal {getattr(signal, 'signal_id', 'unknown')}: {error}"
                        )
            
            self.logger.info(
                "Signal integrity validation completed",
                **{k: v for k, v in validation_results.items() if k != "validation_errors"},
                errors_count=len(validation_results["validation_errors"]),
            )
            
            return validation_results
            
        except Exception as error:
            self.logger.error(
                "Failed to validate signal integrity",
                error=str(error),
            )
            return {}


class F5SignalClient:
    """Enhanced client for receiving signals from F5 Intelligence Layer."""
    
    def __init__(self, signal_router: Optional["SignalRouterService"] = None):
        """
        Initialize F5 signal client.
        
        Args:
            signal_router: Optional signal router service
        """
        self.signal_router = signal_router
        self.logger = get_logger("nautilus_integration.f5_signal_client")
        
        # Local signal cache
        self._signal_cache: Dict[str, List[AISignal]] = defaultdict(list)
        self._cache_ttl = timedelta(minutes=5)
        
        # Connection state
        self._connected = False
        self._connection_retries = 0
        self._max_retries = 3
    
    async def connect_to_f5(self, connection_config: Dict[str, Any]) -> bool:
        """
        Connect to F5 Intelligence Layer.
        
        Args:
            connection_config: F5 connection configuration
            
        Returns:
            True if connection successful
        """
        try:
            self.logger.info("Connecting to F5 Intelligence Layer")
            
            # TODO: Implement actual F5 connection
            # This would establish connection to F5 RAG service
            
            # Simulate connection
            await asyncio.sleep(0.1)
            
            self._connected = True
            self._connection_retries = 0
            
            self.logger.info("Connected to F5 Intelligence Layer successfully")
            return True
            
        except Exception as error:
            self._connection_retries += 1
            self.logger.error(
                "Failed to connect to F5 Intelligence Layer",
                error=str(error),
                retry_count=self._connection_retries,
            )
            
            if self._connection_retries < self._max_retries:
                # Retry with exponential backoff
                retry_delay = 2 ** self._connection_retries
                await asyncio.sleep(retry_delay)
                return await self.connect_to_f5(connection_config)
            
            return False
    
    async def subscribe(self, signal_type: SignalType, callback: callable) -> None:
        """Subscribe to signal type with enhanced error handling."""
        try:
            if not self._connected:
                raise RuntimeError("Not connected to F5 Intelligence Layer")
            
            self.logger.info(
                "Subscribing to F5 signal type",
                signal_type=signal_type.value,
            )
            
            if self.signal_router:
                # Use signal router for subscription
                await self.signal_router.subscribe_strategy(
                    strategy_id="f5_client",
                    signal_types=[signal_type],
                    callback=callback,
                )
            else:
                # Direct F5 subscription
                # TODO: Implement direct F5 subscription
                pass
                
        except Exception as error:
            self.logger.error(
                "Failed to subscribe to F5 signal type",
                signal_type=signal_type.value,
                error=str(error),
            )
            raise
    
    async def get_current_signals(self, instrument_id: str) -> List[AISignal]:
        """Get current signals for instrument with caching."""
        try:
            # Check cache first
            cached_signals = self._signal_cache.get(instrument_id, [])
            
            # Filter out expired signals
            now = datetime.now()
            valid_signals = [
                signal for signal in cached_signals
                if now - signal.timestamp < self._cache_ttl and not signal.is_expired()
            ]
            
            # Update cache
            self._signal_cache[instrument_id] = valid_signals
            
            return valid_signals
            
        except Exception as error:
            self.logger.error(
                "Failed to get current signals",
                instrument_id=instrument_id,
                error=str(error),
            )
            return []
    
    async def get_signal_by_type(
        self,
        instrument_id: str,
        signal_type: SignalType
    ) -> Optional[AISignal]:
        """Get latest signal of specific type for instrument."""
        try:
            signals = await self.get_current_signals(instrument_id)
            
            # Find latest signal of requested type
            type_signals = [s for s in signals if s.signal_type == signal_type]
            if type_signals:
                return max(type_signals, key=lambda s: s.timestamp)
            
            return None
            
        except Exception as error:
            self.logger.error(
                "Failed to get signal by type",
                instrument_id=instrument_id,
                signal_type=signal_type.value,
                error=str(error),
            )
            return None
    
    async def refresh_signals(self, instrument_id: str) -> int:
        """
        Refresh signals from F5 for instrument.
        
        Args:
            instrument_id: Instrument identifier
            
        Returns:
            Number of new signals retrieved
        """
        try:
            if not self._connected:
                await self.connect_to_f5({})
            
            # TODO: Implement actual signal refresh from F5
            # This would query F5 for latest signals
            
            self.logger.debug(
                "Refreshing signals from F5",
                instrument_id=instrument_id,
            )
            
            # Simulate signal refresh
            await asyncio.sleep(0.05)
            
            return 0  # Placeholder
            
        except Exception as error:
            self.logger.error(
                "Failed to refresh signals",
                instrument_id=instrument_id,
                error=str(error),
            )
            return 0
    
    async def disconnect(self) -> None:
        """Disconnect from F5 Intelligence Layer."""
        try:
            if self._connected:
                self.logger.info("Disconnecting from F5 Intelligence Layer")
                
                # TODO: Implement actual disconnection
                
                self._connected = False
                self._signal_cache.clear()
                
                self.logger.info("Disconnected from F5 Intelligence Layer")
                
        except Exception as error:
            self.logger.error(
                "Error during F5 disconnection",
                error=str(error),
            )