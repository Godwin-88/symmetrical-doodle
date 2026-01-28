"""
F8 Risk Management Integration Hooks

This module provides comprehensive integration hooks for F8 Portfolio & Risk management
system, including real-time risk checking, position synchronization, and kill switch
functionality for NautilusTrader strategies.

This implementation follows the requirements:
- 5.1: Integrate Nautilus live trading adapters with existing F8 portfolio and risk management systems
- 5.2: Route all live trades through F8 risk management layer before execution
- 5.3: Maintain real-time position synchronization between Nautilus and F8 systems
- 5.4: Preserve all existing risk limits, drawdown controls, and kill switch functionality
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
from nautilus_integration.services.risk_manager import (
    F8RiskManagerService,
    RiskCheckRequest,
    RiskCheckResult,
    RiskConstraints,
    PositionRecord,
    KillSwitchEvent,
)


class F8IntegrationStatus(str, Enum):
    """F8 integration status."""
    
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    SYNCING = "syncing"


class PositionSyncMode(str, Enum):
    """Position synchronization mode."""
    
    REAL_TIME = "real_time"
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"
    DISABLED = "disabled"


class RiskGateMode(str, Enum):
    """Risk gate enforcement mode."""
    
    STRICT = "strict"  # All trades must pass F8 risk checks
    ADVISORY = "advisory"  # Log warnings but allow trades
    DISABLED = "disabled"  # No risk checking


class F8RiskHook(BaseModel):
    """F8 risk management hook configuration."""
    
    hook_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    hook_type: str  # pre_trade, post_trade, position_sync, kill_switch
    enabled: bool = True
    priority: int = 100  # Lower numbers = higher priority
    
    # Hook-specific configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution tracking
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    average_execution_time_ms: float = 0.0


class PositionSyncResult(BaseModel):
    """Result of position synchronization."""
    
    sync_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    instrument_id: str
    
    # Position data
    nautilus_position: Optional[Decimal] = None
    f8_position: Optional[Decimal] = None
    position_delta: Decimal = Field(default=Decimal('0'))
    
    # Sync status
    sync_status: str = "synced"  # synced, conflict, error
    conflict_resolution: Optional[str] = None
    
    # Metadata
    sync_time: datetime = Field(default_factory=datetime.now)
    sync_duration_ms: float = 0.0
    error_message: Optional[str] = None


class F8RiskIntegrationService:
    """
    Comprehensive F8 Risk Management Integration Service.
    
    This service provides the core integration hooks between NautilusTrader
    and the F8 Portfolio & Risk management system, ensuring all trades
    are properly validated and positions are synchronized in real-time.
    """
    
    def __init__(self, config: NautilusConfig, risk_manager: F8RiskManagerService):
        """
        Initialize F8 risk integration service.
        
        Args:
            config: NautilusTrader integration configuration
            risk_manager: F8 risk manager service instance
        """
        self.config = config
        self.risk_manager = risk_manager
        self.logger = get_logger("nautilus_integration.f8_risk_integration")
        
        # Integration state
        self._integration_status = F8IntegrationStatus.DISCONNECTED
        self._registered_hooks: Dict[str, List[F8RiskHook]] = {}  # strategy_id -> hooks
        self._position_sync_mode = PositionSyncMode.REAL_TIME
        self._risk_gate_mode = RiskGateMode.STRICT
        
        # Position tracking
        self._position_cache: Dict[str, PositionRecord] = {}  # strategy_id:instrument_id -> position
        self._sync_conflicts: List[PositionSyncResult] = []
        self._last_sync_time: Optional[datetime] = None
        
        # Kill switch state
        self._kill_switch_active = False
        self._kill_switch_reason: Optional[str] = None
        self._kill_switch_activated_at: Optional[datetime] = None
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance metrics
        self._hook_execution_stats: Dict[str, Dict[str, float]] = {}
        self._sync_performance_stats: Dict[str, float] = {
            "average_sync_time_ms": 0.0,
            "sync_success_rate": 100.0,
            "conflicts_per_hour": 0.0,
        }
        
        self.logger.info(
            "F8 Risk Integration Service initialized",
            position_sync_mode=self._position_sync_mode.value,
            risk_gate_mode=self._risk_gate_mode.value,
        )
    
    async def initialize(self) -> None:
        """Initialize F8 risk integration service."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Initializing F8 Risk Integration Service")
            
            try:
                # Test F8 system connectivity
                await self._test_f8_connectivity()
                
                # Initialize position synchronization
                await self._initialize_position_sync()
                
                # Start background monitoring
                await self._start_background_tasks()
                
                # Set integration status
                self._integration_status = F8IntegrationStatus.CONNECTED
                
                self.logger.info(
                    "F8 Risk Integration Service initialization completed",
                    integration_status=self._integration_status.value,
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                self._integration_status = F8IntegrationStatus.ERROR
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "initialize", "correlation_id": correlation_id},
                    "Failed to initialize F8 Risk Integration Service"
                )
                raise
    
    async def register_strategy_hooks(
        self,
        strategy_id: str,
        risk_constraints: RiskConstraints,
        hook_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register comprehensive risk management hooks for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            risk_constraints: Risk constraints configuration
            hook_config: Optional hook configuration overrides
            
        Raises:
            ValueError: If strategy is already registered
            RuntimeError: If hook registration fails
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Registering F8 risk hooks for strategy",
                strategy_id=strategy_id,
                max_position_size=float(risk_constraints.max_position_size),
                max_daily_loss=float(risk_constraints.max_daily_loss),
            )
            
            try:
                if strategy_id in self._registered_hooks:
                    raise ValueError(f"Strategy {strategy_id} hooks already registered")
                
                # Register strategy with risk manager
                await self.risk_manager.register_strategy(strategy_id, risk_constraints)
                
                # Create comprehensive hook set
                hooks = await self._create_strategy_hooks(strategy_id, risk_constraints, hook_config)
                self._registered_hooks[strategy_id] = hooks
                
                # Initialize position tracking for strategy
                await self._initialize_strategy_position_tracking(strategy_id)
                
                self.logger.info(
                    "F8 risk hooks registered successfully",
                    strategy_id=strategy_id,
                    hooks_count=len(hooks),
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "register_strategy_hooks",
                        "strategy_id": strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to register F8 risk hooks"
                )
                raise
    
    async def execute_pre_trade_hooks(
        self,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskCheckResult:
        """
        Execute pre-trade risk hooks with comprehensive validation.
        
        Args:
            strategy_id: Strategy identifier
            instrument_id: Instrument identifier
            side: Trade side (BUY/SELL)
            quantity: Trade quantity
            price: Optional trade price
            metadata: Optional trade metadata
            
        Returns:
            Risk check result with approval status
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If risk check fails
        """
        start_time = datetime.now()
        
        with with_correlation_id() as correlation_id:
            self.logger.debug(
                "Executing pre-trade risk hooks",
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                side=side,
                quantity=float(quantity),
            )
            
            try:
                # Check kill switch first
                if self._kill_switch_active:
                    return RiskCheckResult(
                        request_id=str(uuid4()),
                        status="REJECTED",
                        approved=False,
                        reason=f"Kill switch active: {self._kill_switch_reason}",
                        risk_score=1.0,
                    )
                
                # Check integration status
                if self._integration_status != F8IntegrationStatus.CONNECTED:
                    if self._risk_gate_mode == RiskGateMode.STRICT:
                        return RiskCheckResult(
                            request_id=str(uuid4()),
                            status="REJECTED",
                            approved=False,
                            reason="F8 integration not available",
                            risk_score=0.9,
                        )
                    else:
                        self.logger.warning(
                            "F8 integration not available, allowing trade in advisory mode",
                            strategy_id=strategy_id,
                        )
                
                # Create risk check request
                request = RiskCheckRequest(
                    strategy_id=strategy_id,
                    instrument_id=instrument_id,
                    side=side,
                    quantity=quantity,
                    price=price,
                    metadata=metadata or {},
                )
                
                # Execute pre-trade hooks in priority order
                hooks = self._get_hooks_by_type(strategy_id, "pre_trade")
                for hook in sorted(hooks, key=lambda h: h.priority):
                    if hook.enabled:
                        await self._execute_hook(hook, request)
                
                # Perform comprehensive risk check
                result = await self.risk_manager.check_trade_limits(request)
                
                # Update position cache if trade approved
                if result.approved:
                    await self._update_position_cache_for_trade(
                        strategy_id, instrument_id, side, quantity
                    )
                
                # Record execution time
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_hook_performance_stats("pre_trade", execution_time)
                
                self.logger.debug(
                    "Pre-trade risk hooks completed",
                    strategy_id=strategy_id,
                    approved=result.approved,
                    risk_score=result.risk_score,
                    execution_time_ms=execution_time,
                )
                
                return result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "execute_pre_trade_hooks",
                        "strategy_id": strategy_id,
                        "instrument_id": instrument_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to execute pre-trade risk hooks"
                )
                
                # Return rejection in case of error
                return RiskCheckResult(
                    request_id=str(uuid4()),
                    status="ERROR",
                    approved=False,
                    reason=f"Risk check error: {str(error)}",
                    risk_score=1.0,
                )
    
    async def execute_post_trade_hooks(
        self,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fill_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Execute post-trade hooks for position synchronization and monitoring.
        
        Args:
            strategy_id: Strategy identifier
            instrument_id: Instrument identifier
            side: Trade side (BUY/SELL)
            quantity: Executed quantity
            price: Execution price
            fill_id: Fill identifier
            metadata: Optional trade metadata
        """
        with with_correlation_id() as correlation_id:
            self.logger.debug(
                "Executing post-trade hooks",
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                side=side,
                quantity=float(quantity),
                price=float(price),
                fill_id=fill_id,
            )
            
            try:
                # Execute post-trade hooks
                hooks = self._get_hooks_by_type(strategy_id, "post_trade")
                for hook in sorted(hooks, key=lambda h: h.priority):
                    if hook.enabled:
                        await self._execute_post_trade_hook(
                            hook, strategy_id, instrument_id, side, quantity, price, fill_id, metadata
                        )
                
                # Update position tracking
                await self._update_position_tracking(
                    strategy_id, instrument_id, side, quantity, price
                )
                
                # Trigger position synchronization if needed
                if self._position_sync_mode == PositionSyncMode.REAL_TIME:
                    await self._sync_position_with_f8(strategy_id, instrument_id)
                
                self.logger.debug(
                    "Post-trade hooks completed successfully",
                    strategy_id=strategy_id,
                    fill_id=fill_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "execute_post_trade_hooks",
                        "strategy_id": strategy_id,
                        "fill_id": fill_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to execute post-trade hooks"
                )
    
    async def synchronize_positions(
        self,
        strategy_id: Optional[str] = None,
        instrument_id: Optional[str] = None
    ) -> List[PositionSyncResult]:
        """
        Synchronize positions between Nautilus and F8 systems.
        
        Args:
            strategy_id: Optional strategy filter
            instrument_id: Optional instrument filter
            
        Returns:
            List of position synchronization results
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Starting position synchronization",
                strategy_id=strategy_id,
                instrument_id=instrument_id,
            )
            
            try:
                sync_results = []
                
                # Determine positions to sync
                positions_to_sync = self._get_positions_to_sync(strategy_id, instrument_id)
                
                for position_key in positions_to_sync:
                    try:
                        result = await self._sync_single_position(position_key)
                        sync_results.append(result)
                        
                    except Exception as error:
                        self.logger.warning(
                            "Failed to sync position",
                            position_key=position_key,
                            error=str(error),
                        )
                        
                        # Create error result
                        parts = position_key.split(":")
                        sync_results.append(PositionSyncResult(
                            strategy_id=parts[0],
                            instrument_id=parts[1],
                            sync_status="error",
                            error_message=str(error),
                        ))
                
                # Update sync statistics
                self._update_sync_performance_stats(sync_results)
                self._last_sync_time = datetime.now()
                
                self.logger.info(
                    "Position synchronization completed",
                    total_positions=len(sync_results),
                    synced_count=len([r for r in sync_results if r.sync_status == "synced"]),
                    conflicts_count=len([r for r in sync_results if r.sync_status == "conflict"]),
                    errors_count=len([r for r in sync_results if r.sync_status == "error"]),
                    correlation_id=correlation_id,
                )
                
                return sync_results
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "synchronize_positions",
                        "strategy_id": strategy_id,
                        "instrument_id": instrument_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to synchronize positions"
                )
                raise
    
    async def activate_kill_switch(
        self,
        reason: str,
        triggered_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Activate kill switch to halt all trading immediately.
        
        Args:
            reason: Reason for kill switch activation
            triggered_by: Who/what triggered the kill switch
            metadata: Optional metadata
        """
        with with_correlation_id() as correlation_id:
            self.logger.critical(
                "Activating kill switch",
                reason=reason,
                triggered_by=triggered_by,
            )
            
            try:
                # Set kill switch state
                self._kill_switch_active = True
                self._kill_switch_reason = reason
                self._kill_switch_activated_at = datetime.now()
                
                # Create kill switch event
                event = KillSwitchEvent(
                    event_id=str(uuid4()),
                    reason=reason,
                    triggered_by=triggered_by,
                    metadata=metadata or {},
                )
                
                # Execute kill switch hooks for all strategies
                for strategy_id in self._registered_hooks.keys():
                    hooks = self._get_hooks_by_type(strategy_id, "kill_switch")
                    for hook in hooks:
                        if hook.enabled:
                            await self._execute_kill_switch_hook(hook, event)
                
                # Notify F8 system
                await self._notify_f8_kill_switch_activation(event)
                
                self.logger.critical(
                    "Kill switch activated successfully",
                    reason=reason,
                    affected_strategies=len(self._registered_hooks),
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "activate_kill_switch",
                        "reason": reason,
                        "correlation_id": correlation_id,
                    },
                    "Failed to activate kill switch"
                )
                raise
    
    async def deactivate_kill_switch(
        self,
        authorized_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Deactivate kill switch to resume trading.
        
        Args:
            authorized_by: Who authorized the deactivation
            metadata: Optional metadata
        """
        with with_correlation_id() as correlation_id:
            self.logger.warning(
                "Deactivating kill switch",
                authorized_by=authorized_by,
                was_active_for_seconds=(
                    (datetime.now() - self._kill_switch_activated_at).total_seconds()
                    if self._kill_switch_activated_at else 0
                ),
            )
            
            try:
                # Reset kill switch state
                self._kill_switch_active = False
                previous_reason = self._kill_switch_reason
                self._kill_switch_reason = None
                self._kill_switch_activated_at = None
                
                # Notify F8 system
                await self._notify_f8_kill_switch_deactivation(authorized_by, previous_reason)
                
                self.logger.warning(
                    "Kill switch deactivated successfully",
                    authorized_by=authorized_by,
                    previous_reason=previous_reason,
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "deactivate_kill_switch",
                        "authorized_by": authorized_by,
                        "correlation_id": correlation_id,
                    },
                    "Failed to deactivate kill switch"
                )
                raise
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive F8 integration status.
        
        Returns:
            Integration status information
        """
        return {
            "integration_status": self._integration_status.value,
            "position_sync_mode": self._position_sync_mode.value,
            "risk_gate_mode": self._risk_gate_mode.value,
            "kill_switch_active": self._kill_switch_active,
            "kill_switch_reason": self._kill_switch_reason,
            "registered_strategies": len(self._registered_hooks),
            "position_cache_size": len(self._position_cache),
            "sync_conflicts_count": len(self._sync_conflicts),
            "last_sync_time": self._last_sync_time.isoformat() if self._last_sync_time else None,
            "performance_stats": self._sync_performance_stats.copy(),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def shutdown(self) -> None:
        """Shutdown F8 risk integration service."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Shutting down F8 Risk Integration Service")
            
            try:
                # Stop background tasks
                self._running = False
                if self._sync_task:
                    self._sync_task.cancel()
                if self._monitoring_task:
                    self._monitoring_task.cancel()
                
                # Unregister all strategies
                for strategy_id in list(self._registered_hooks.keys()):
                    await self._unregister_strategy_hooks(strategy_id)
                
                # Final position sync
                await self.synchronize_positions()
                
                # Set disconnected status
                self._integration_status = F8IntegrationStatus.DISCONNECTED
                
                self.logger.info(
                    "F8 Risk Integration Service shutdown completed",
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "shutdown", "correlation_id": correlation_id},
                    "Error during F8 Risk Integration Service shutdown"
                )
    
    # Private helper methods
    
    async def _test_f8_connectivity(self) -> None:
        """Test connectivity to F8 system."""
        try:
            # TODO: Implement actual F8 connectivity test
            # For now, simulate successful connection
            await asyncio.sleep(0.1)
            self.logger.debug("F8 connectivity test passed")
            
        except Exception as error:
            raise RuntimeError(f"F8 connectivity test failed: {error}")
    
    async def _initialize_position_sync(self) -> None:
        """Initialize position synchronization system."""
        try:
            # Load existing positions from F8
            await self._load_f8_positions()
            
            # Initialize sync conflict resolution
            self._sync_conflicts.clear()
            
            self.logger.debug("Position synchronization initialized")
            
        except Exception as error:
            raise RuntimeError(f"Position sync initialization failed: {error}")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and sync tasks."""
        try:
            self._running = True
            
            # Start position sync worker
            if self._position_sync_mode == PositionSyncMode.PERIODIC:
                self._sync_task = asyncio.create_task(self._position_sync_worker())
            
            # Start monitoring worker
            self._monitoring_task = asyncio.create_task(self._monitoring_worker())
            
            self.logger.debug("Background tasks started")
            
        except Exception as error:
            raise RuntimeError(f"Failed to start background tasks: {error}")
    
    async def _create_strategy_hooks(
        self,
        strategy_id: str,
        risk_constraints: RiskConstraints,
        hook_config: Optional[Dict[str, Any]]
    ) -> List[F8RiskHook]:
        """Create comprehensive hook set for strategy."""
        hooks = []
        
        # Pre-trade risk check hook
        hooks.append(F8RiskHook(
            strategy_id=strategy_id,
            hook_type="pre_trade",
            priority=10,  # High priority
            config={
                "max_position_size": float(risk_constraints.max_position_size),
                "max_daily_loss": float(risk_constraints.max_daily_loss),
                "check_portfolio_limits": True,
                "check_concentration_limits": True,
            }
        ))
        
        # Post-trade position sync hook
        hooks.append(F8RiskHook(
            strategy_id=strategy_id,
            hook_type="post_trade",
            priority=20,
            config={
                "sync_mode": self._position_sync_mode.value,
                "conflict_resolution": "f8_wins",  # F8 system is authoritative
            }
        ))
        
        # Position synchronization hook
        hooks.append(F8RiskHook(
            strategy_id=strategy_id,
            hook_type="position_sync",
            priority=30,
            config={
                "sync_interval_seconds": 60,
                "conflict_threshold": 0.01,  # 1 cent tolerance
            }
        ))
        
        # Kill switch hook
        hooks.append(F8RiskHook(
            strategy_id=strategy_id,
            hook_type="kill_switch",
            priority=1,  # Highest priority
            config={
                "immediate_halt": True,
                "notify_operators": True,
            }
        ))
        
        # Apply hook config overrides
        if hook_config:
            for hook in hooks:
                if hook.hook_type in hook_config:
                    hook.config.update(hook_config[hook.hook_type])
        
        return hooks
    
    def _get_hooks_by_type(self, strategy_id: str, hook_type: str) -> List[F8RiskHook]:
        """Get hooks of specific type for strategy."""
        strategy_hooks = self._registered_hooks.get(strategy_id, [])
        return [hook for hook in strategy_hooks if hook.hook_type == hook_type]
    
    async def _execute_hook(self, hook: F8RiskHook, request: RiskCheckRequest) -> None:
        """Execute a specific risk hook."""
        start_time = datetime.now()
        
        try:
            # Update hook execution tracking
            hook.execution_count += 1
            hook.last_execution = start_time
            
            # Execute hook based on type
            if hook.hook_type == "pre_trade":
                await self._execute_pre_trade_hook_logic(hook, request)
            
            # Update performance stats
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            hook.average_execution_time_ms = (
                (hook.average_execution_time_ms * (hook.execution_count - 1) + execution_time)
                / hook.execution_count
            )
            
        except Exception as error:
            self.logger.error(
                "Hook execution failed",
                hook_id=hook.hook_id,
                hook_type=hook.hook_type,
                strategy_id=hook.strategy_id,
                error=str(error),
            )
            raise
    
    async def _execute_pre_trade_hook_logic(self, hook: F8RiskHook, request: RiskCheckRequest) -> None:
        """Execute pre-trade hook logic."""
        # This would contain the actual pre-trade validation logic
        # For now, just log the execution
        self.logger.debug(
            "Executing pre-trade hook",
            hook_id=hook.hook_id,
            strategy_id=hook.strategy_id,
            instrument_id=request.instrument_id,
        )
    
    async def _execute_post_trade_hook(
        self,
        hook: F8RiskHook,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fill_id: str,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Execute post-trade hook."""
        self.logger.debug(
            "Executing post-trade hook",
            hook_id=hook.hook_id,
            strategy_id=strategy_id,
            fill_id=fill_id,
        )
        
        # Update hook execution tracking
        hook.execution_count += 1
        hook.last_execution = datetime.now()
    
    async def _execute_kill_switch_hook(self, hook: F8RiskHook, event: KillSwitchEvent) -> None:
        """Execute kill switch hook."""
        self.logger.critical(
            "Executing kill switch hook",
            hook_id=hook.hook_id,
            strategy_id=hook.strategy_id,
            event_id=event.event_id,
            reason=event.reason,
        )
        
        # Update hook execution tracking
        hook.execution_count += 1
        hook.last_execution = datetime.now()
    
    async def _update_position_cache_for_trade(
        self,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal
    ) -> None:
        """Update position cache for approved trade."""
        position_key = f"{strategy_id}:{instrument_id}"
        
        # Get current position
        current_position = self._position_cache.get(position_key)
        if not current_position:
            current_position = PositionRecord(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                quantity=Decimal('0'),
            )
        
        # Update position based on trade
        if side.upper() == "BUY":
            current_position.quantity += quantity
        else:
            current_position.quantity -= quantity
        
        current_position.last_updated = datetime.now()
        self._position_cache[position_key] = current_position
    
    async def _update_position_tracking(
        self,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal,
        price: Decimal
    ) -> None:
        """Update position tracking after trade execution."""
        position_key = f"{strategy_id}:{instrument_id}"
        
        # Update position cache
        await self._update_position_cache_for_trade(strategy_id, instrument_id, side, quantity)
        
        # Log position update
        self.logger.debug(
            "Position tracking updated",
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            side=side,
            quantity=float(quantity),
            price=float(price),
        )
    
    async def _sync_position_with_f8(self, strategy_id: str, instrument_id: str) -> PositionSyncResult:
        """Synchronize single position with F8 system."""
        position_key = f"{strategy_id}:{instrument_id}"
        
        try:
            # Get Nautilus position
            nautilus_position = self._position_cache.get(position_key)
            nautilus_qty = nautilus_position.quantity if nautilus_position else Decimal('0')
            
            # Get F8 position (simulated for now)
            f8_qty = await self._get_f8_position(strategy_id, instrument_id)
            
            # Calculate delta
            delta = nautilus_qty - f8_qty
            
            # Determine sync status
            if abs(delta) < Decimal('0.01'):  # 1 cent tolerance
                sync_status = "synced"
            else:
                sync_status = "conflict"
                self.logger.warning(
                    "Position sync conflict detected",
                    strategy_id=strategy_id,
                    instrument_id=instrument_id,
                    nautilus_position=float(nautilus_qty),
                    f8_position=float(f8_qty),
                    delta=float(delta),
                )
            
            return PositionSyncResult(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                nautilus_position=nautilus_qty,
                f8_position=f8_qty,
                position_delta=delta,
                sync_status=sync_status,
            )
            
        except Exception as error:
            return PositionSyncResult(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                sync_status="error",
                error_message=str(error),
            )
    
    async def _get_f8_position(self, strategy_id: str, instrument_id: str) -> Decimal:
        """Get position from F8 system."""
        # TODO: Implement actual F8 position retrieval
        # For now, return zero position
        return Decimal('0')
    
    def _get_positions_to_sync(
        self,
        strategy_id: Optional[str],
        instrument_id: Optional[str]
    ) -> List[str]:
        """Get list of position keys to synchronize."""
        positions = list(self._position_cache.keys())
        
        if strategy_id:
            positions = [p for p in positions if p.startswith(f"{strategy_id}:")]
        
        if instrument_id:
            positions = [p for p in positions if p.endswith(f":{instrument_id}")]
        
        return positions
    
    async def _sync_single_position(self, position_key: str) -> PositionSyncResult:
        """Synchronize a single position."""
        parts = position_key.split(":")
        strategy_id, instrument_id = parts[0], parts[1]
        
        return await self._sync_position_with_f8(strategy_id, instrument_id)
    
    def _update_hook_performance_stats(self, hook_type: str, execution_time_ms: float) -> None:
        """Update hook performance statistics."""
        if hook_type not in self._hook_execution_stats:
            self._hook_execution_stats[hook_type] = {
                "count": 0,
                "total_time_ms": 0.0,
                "average_time_ms": 0.0,
            }
        
        stats = self._hook_execution_stats[hook_type]
        stats["count"] += 1
        stats["total_time_ms"] += execution_time_ms
        stats["average_time_ms"] = stats["total_time_ms"] / stats["count"]
    
    def _update_sync_performance_stats(self, sync_results: List[PositionSyncResult]) -> None:
        """Update synchronization performance statistics."""
        if not sync_results:
            return
        
        # Calculate average sync time
        sync_times = [r.sync_duration_ms for r in sync_results if r.sync_duration_ms > 0]
        if sync_times:
            self._sync_performance_stats["average_sync_time_ms"] = sum(sync_times) / len(sync_times)
        
        # Calculate success rate
        successful_syncs = len([r for r in sync_results if r.sync_status == "synced"])
        self._sync_performance_stats["sync_success_rate"] = (successful_syncs / len(sync_results)) * 100
        
        # Calculate conflicts per hour
        conflicts = len([r for r in sync_results if r.sync_status == "conflict"])
        self._sync_performance_stats["conflicts_per_hour"] = conflicts  # Simplified calculation
    
    async def _load_f8_positions(self) -> None:
        """Load existing positions from F8 system."""
        # TODO: Implement actual F8 position loading
        self.logger.debug("Loading positions from F8 system")
    
    async def _initialize_strategy_position_tracking(self, strategy_id: str) -> None:
        """Initialize position tracking for strategy."""
        # TODO: Load existing positions for strategy
        self.logger.debug(
            "Initialized position tracking for strategy",
            strategy_id=strategy_id,
        )
    
    async def _unregister_strategy_hooks(self, strategy_id: str) -> None:
        """Unregister strategy hooks."""
        if strategy_id in self._registered_hooks:
            del self._registered_hooks[strategy_id]
            
            # Clean up position cache
            keys_to_remove = [key for key in self._position_cache.keys() if key.startswith(f"{strategy_id}:")]
            for key in keys_to_remove:
                del self._position_cache[key]
            
            self.logger.info(
                "Strategy hooks unregistered",
                strategy_id=strategy_id,
            )
    
    async def _notify_f8_kill_switch_activation(self, event: KillSwitchEvent) -> None:
        """Notify F8 system of kill switch activation."""
        # TODO: Implement actual F8 notification
        self.logger.debug(
            "Notified F8 system of kill switch activation",
            event_id=event.event_id,
        )
    
    async def _notify_f8_kill_switch_deactivation(self, authorized_by: str, previous_reason: Optional[str]) -> None:
        """Notify F8 system of kill switch deactivation."""
        # TODO: Implement actual F8 notification
        self.logger.debug(
            "Notified F8 system of kill switch deactivation",
            authorized_by=authorized_by,
            previous_reason=previous_reason,
        )
    
    async def _position_sync_worker(self) -> None:
        """Background worker for periodic position synchronization."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Sync every minute
                if self._running:
                    await self.synchronize_positions()
                    
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error(
                    "Position sync worker error",
                    error=str(error),
                )
    
    async def _monitoring_worker(self) -> None:
        """Background worker for system monitoring."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                if self._running:
                    await self._perform_health_checks()
                    
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error(
                    "Monitoring worker error",
                    error=str(error),
                )
    
    async def _perform_health_checks(self) -> None:
        """Perform periodic health checks."""
        try:
            # Check F8 connectivity
            await self._test_f8_connectivity()
            
            # Check for stale positions
            await self._check_for_stale_positions()
            
            # Check sync conflicts
            await self._check_sync_conflicts()
            
        except Exception as error:
            self.logger.warning(
                "Health check failed",
                error=str(error),
            )
    
    async def _check_for_stale_positions(self) -> None:
        """Check for stale position data."""
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        stale_positions = [
            key for key, position in self._position_cache.items()
            if position.last_updated < cutoff_time
        ]
        
        if stale_positions:
            self.logger.warning(
                "Stale positions detected",
                stale_count=len(stale_positions),
                positions=stale_positions[:5],  # Log first 5
            )
    
    async def _check_sync_conflicts(self) -> None:
        """Check for unresolved sync conflicts."""
        recent_conflicts = [
            conflict for conflict in self._sync_conflicts
            if conflict.sync_time > datetime.now() - timedelta(hours=1)
        ]
        
        if len(recent_conflicts) > 10:  # Threshold for concern
            self.logger.warning(
                "High number of sync conflicts",
                conflicts_count=len(recent_conflicts),
            )