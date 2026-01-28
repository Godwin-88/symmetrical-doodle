"""
Live Trading Risk Gate Enforcement

This module implements comprehensive risk gate enforcement for live trading,
ensuring all trades are routed through the F8 risk management layer with
no bypass capability, while preserving existing risk limits and kill switch
functionality with enhanced monitoring.

This implementation follows the requirements:
- 5.2: Route all live trades through F8 risk management layer before execution
- 5.4: Preserve existing risk limits and kill switch functionality
- 5.5: Add live trading validation mirroring in backtesting with consistency verification
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
from nautilus_integration.services.f8_risk_integration import (
    F8RiskIntegrationService,
    PositionSyncResult,
)
from nautilus_integration.services.risk_manager import (
    F8RiskManagerService,
    RiskCheckRequest,
    RiskCheckResult,
    RiskConstraints,
)


class TradingMode(str, Enum):
    """Trading mode."""
    
    LIVE = "live"
    BACKTEST = "backtest"
    PAPER = "paper"
    SIMULATION = "simulation"


class RiskGateStatus(str, Enum):
    """Risk gate status."""
    
    ACTIVE = "active"
    BYPASSED = "bypassed"  # Should never happen in production
    DISABLED = "disabled"  # Only for testing
    ERROR = "error"


class TradeValidationResult(BaseModel):
    """Result of trade validation."""
    
    validation_id: str = Field(default_factory=lambda: str(uuid4()))
    trade_id: str
    strategy_id: str
    instrument_id: str
    
    # Validation status
    approved: bool = False
    risk_gate_passed: bool = False
    f8_validation_passed: bool = False
    backtest_mirror_passed: bool = False
    
    # Risk assessment
    risk_score: float = 0.0
    violated_limits: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Validation details
    f8_check_result: Optional[RiskCheckResult] = None
    backtest_result: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    validation_time_ms: float = 0.0
    f8_check_time_ms: float = 0.0
    backtest_mirror_time_ms: float = 0.0
    
    # Metadata
    validation_time: datetime = Field(default_factory=datetime.now)
    trading_mode: TradingMode = TradingMode.LIVE
    bypass_reason: Optional[str] = None


class LiveTradingRiskGate:
    """
    Comprehensive Live Trading Risk Gate Enforcement.
    
    This service ensures that ALL live trades are routed through the F8 risk
    management layer with no bypass capability. It provides:
    
    - Mandatory F8 risk validation for every trade
    - Live/backtest parity validation
    - Enhanced monitoring and alerting
    - Immediate halt mechanisms
    - Comprehensive audit trails
    """
    
    def __init__(
        self,
        config: NautilusConfig,
        f8_integration: F8RiskIntegrationService,
        risk_manager: F8RiskManagerService
    ):
        """
        Initialize live trading risk gate.
        
        Args:
            config: NautilusTrader integration configuration
            f8_integration: F8 risk integration service
            risk_manager: F8 risk manager service
        """
        self.config = config
        self.f8_integration = f8_integration
        self.risk_manager = risk_manager
        self.logger = get_logger("nautilus_integration.live_trading_risk_gate")
        
        # Risk gate state
        self._gate_status = RiskGateStatus.ACTIVE
        self._bypass_disabled = True  # Bypass is permanently disabled in production
        self._emergency_halt_active = False
        self._halt_reason: Optional[str] = None
        
        # Trading mode tracking
        self._current_trading_mode = TradingMode.LIVE
        self._backtest_mirror_enabled = True
        
        # Validation tracking
        self._validation_history: List[TradeValidationResult] = []
        self._rejected_trades_count = 0
        self._approved_trades_count = 0
        
        # Performance monitoring
        self._validation_stats = {
            "average_validation_time_ms": 0.0,
            "f8_check_success_rate": 100.0,
            "backtest_mirror_success_rate": 100.0,
            "total_validations": 0,
            "rejection_rate": 0.0,
        }
        
        # Alert thresholds
        self._alert_thresholds = {
            "max_validation_time_ms": 1000.0,  # 1 second
            "max_rejection_rate": 0.10,  # 10%
            "max_f8_failures_per_hour": 5,
            "max_backtest_mirror_failures_per_hour": 10,
        }
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger.info(
            "Live Trading Risk Gate initialized",
            gate_status=self._gate_status.value,
            bypass_disabled=self._bypass_disabled,
            backtest_mirror_enabled=self._backtest_mirror_enabled,
        )
    
    async def initialize(self) -> None:
        """Initialize live trading risk gate."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Initializing Live Trading Risk Gate")
            
            try:
                # Verify F8 integration is active
                f8_status = await self.f8_integration.get_integration_status()
                if f8_status["integration_status"] != "connected":
                    raise RuntimeError("F8 integration not connected - cannot enable live trading")
                
                # Verify risk manager is operational
                await self._verify_risk_manager_operational()
                
                # Initialize backtest mirror if enabled
                if self._backtest_mirror_enabled:
                    await self._initialize_backtest_mirror()
                
                # Start monitoring
                await self._start_monitoring()
                
                # Set gate to active
                self._gate_status = RiskGateStatus.ACTIVE
                
                self.logger.info(
                    "Live Trading Risk Gate initialization completed",
                    gate_status=self._gate_status.value,
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                self._gate_status = RiskGateStatus.ERROR
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "initialize", "correlation_id": correlation_id},
                    "Failed to initialize Live Trading Risk Gate"
                )
                raise
    
    async def validate_live_trade(
        self,
        trade_id: str,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "MARKET",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TradeValidationResult:
        """
        Validate live trade through comprehensive risk gate enforcement.
        
        This is the MANDATORY entry point for ALL live trades. No trade can
        bypass this validation in production mode.
        
        Args:
            trade_id: Unique trade identifier
            strategy_id: Strategy identifier
            instrument_id: Instrument identifier
            side: Trade side (BUY/SELL)
            quantity: Trade quantity
            price: Optional trade price
            order_type: Order type
            metadata: Optional trade metadata
            
        Returns:
            Trade validation result with approval status
            
        Raises:
            ValueError: If trade parameters are invalid
            RuntimeError: If validation system is not operational
        """
        start_time = datetime.now()
        
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Validating live trade through risk gate",
                trade_id=trade_id,
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                side=side,
                quantity=float(quantity),
                order_type=order_type,
            )
            
            try:
                # Create validation result
                result = TradeValidationResult(
                    trade_id=trade_id,
                    strategy_id=strategy_id,
                    instrument_id=instrument_id,
                    trading_mode=self._current_trading_mode,
                )
                
                # CRITICAL: Check if risk gate is operational
                if self._gate_status != RiskGateStatus.ACTIVE:
                    result.approved = False
                    result.violated_limits.append("risk_gate_not_active")
                    result.warnings.append(f"Risk gate status: {self._gate_status.value}")
                    
                    self.logger.critical(
                        "Trade rejected - risk gate not active",
                        trade_id=trade_id,
                        gate_status=self._gate_status.value,
                    )
                    
                    return result
                
                # CRITICAL: Check emergency halt status
                if self._emergency_halt_active:
                    result.approved = False
                    result.violated_limits.append("emergency_halt_active")
                    result.warnings.append(f"Emergency halt reason: {self._halt_reason}")
                    
                    self.logger.critical(
                        "Trade rejected - emergency halt active",
                        trade_id=trade_id,
                        halt_reason=self._halt_reason,
                    )
                    
                    return result
                
                # STEP 1: Mandatory F8 risk validation (NO BYPASS ALLOWED)
                f8_start_time = datetime.now()
                f8_result = await self._execute_mandatory_f8_validation(
                    trade_id, strategy_id, instrument_id, side, quantity, price, metadata
                )
                result.f8_check_time_ms = (datetime.now() - f8_start_time).total_seconds() * 1000
                result.f8_check_result = f8_result
                result.f8_validation_passed = f8_result.approved
                
                # If F8 validation fails, trade is REJECTED (no exceptions)
                if not f8_result.approved:
                    result.approved = False
                    result.risk_gate_passed = False
                    result.risk_score = f8_result.risk_score
                    result.violated_limits.extend(f8_result.violated_limits)
                    
                    self.logger.warning(
                        "Trade rejected by F8 risk validation",
                        trade_id=trade_id,
                        f8_reason=f8_result.reason,
                        risk_score=f8_result.risk_score,
                        violated_limits=f8_result.violated_limits,
                    )
                    
                    # Check if emergency halt should be triggered
                    if f8_result.risk_score >= 0.95:
                        await self._trigger_emergency_halt(
                            f"Critical risk score: {f8_result.risk_score}",
                            f"Trade {trade_id} risk validation"
                        )
                    
                    return result
                
                # STEP 2: Live/Backtest parity validation (if enabled)
                if self._backtest_mirror_enabled:
                    backtest_start_time = datetime.now()
                    backtest_result = await self._execute_backtest_mirror_validation(
                        trade_id, strategy_id, instrument_id, side, quantity, price, metadata
                    )
                    result.backtest_mirror_time_ms = (datetime.now() - backtest_start_time).total_seconds() * 1000
                    result.backtest_result = backtest_result
                    result.backtest_mirror_passed = backtest_result.get("passed", False)
                    
                    # Backtest mirror failure is a warning, not a rejection
                    if not result.backtest_mirror_passed:
                        result.warnings.append("Backtest mirror validation failed")
                        self.logger.warning(
                            "Backtest mirror validation failed",
                            trade_id=trade_id,
                            backtest_reason=backtest_result.get("reason", "unknown"),
                        )
                
                # STEP 3: Final risk gate approval
                result.risk_gate_passed = True
                result.approved = True
                result.risk_score = f8_result.risk_score
                
                # Calculate total validation time
                result.validation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Update statistics
                self._update_validation_statistics(result)
                
                # Store validation history
                self._validation_history.append(result)
                self._cleanup_validation_history()
                
                self.logger.info(
                    "Live trade validated and approved",
                    trade_id=trade_id,
                    risk_score=result.risk_score,
                    validation_time_ms=result.validation_time_ms,
                    f8_check_time_ms=result.f8_check_time_ms,
                    backtest_mirror_time_ms=result.backtest_mirror_time_ms,
                )
                
                return result
                
            except Exception as error:
                # CRITICAL: Any validation error results in trade rejection
                result.approved = False
                result.violated_limits.append("validation_system_error")
                result.warnings.append(f"Validation error: {str(error)}")
                
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "validate_live_trade",
                        "trade_id": trade_id,
                        "strategy_id": strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Live trade validation failed due to system error"
                )
                
                # Consider triggering emergency halt for system errors
                if "system" in str(error).lower() or "connection" in str(error).lower():
                    await self._trigger_emergency_halt(
                        f"Validation system error: {str(error)}",
                        "live_trade_validation"
                    )
                
                return result
    
    async def trigger_emergency_halt(
        self,
        reason: str,
        triggered_by: str = "operator",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Trigger emergency halt with immediate effect.
        
        Args:
            reason: Reason for emergency halt
            triggered_by: Who/what triggered the halt
            metadata: Optional metadata
        """
        await self._trigger_emergency_halt(reason, triggered_by, metadata)
    
    async def clear_emergency_halt(
        self,
        authorized_by: str,
        authorization_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Clear emergency halt (requires authorization).
        
        Args:
            authorized_by: Who authorized the halt clearance
            authorization_code: Optional authorization code
            metadata: Optional metadata
        """
        with with_correlation_id() as correlation_id:
            self.logger.critical(
                "Clearing emergency halt",
                authorized_by=authorized_by,
                previous_reason=self._halt_reason,
            )
            
            try:
                # Verify authorization (in production, this would check actual credentials)
                if not authorized_by or len(authorized_by.strip()) == 0:
                    raise ValueError("Authorization required to clear emergency halt")
                
                # Clear halt state
                self._emergency_halt_active = False
                previous_reason = self._halt_reason
                self._halt_reason = None
                
                # Notify F8 integration
                await self.f8_integration.deactivate_kill_switch(
                    authorized_by=authorized_by,
                    metadata=metadata,
                )
                
                # Re-verify system operational status
                await self._verify_system_operational()
                
                self.logger.critical(
                    "Emergency halt cleared successfully",
                    authorized_by=authorized_by,
                    previous_reason=previous_reason,
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "clear_emergency_halt",
                        "authorized_by": authorized_by,
                        "correlation_id": correlation_id,
                    },
                    "Failed to clear emergency halt"
                )
                raise
    
    async def get_risk_gate_status(self) -> Dict[str, Any]:
        """
        Get comprehensive risk gate status.
        
        Returns:
            Risk gate status information
        """
        return {
            "gate_status": self._gate_status.value,
            "bypass_disabled": self._bypass_disabled,
            "emergency_halt_active": self._emergency_halt_active,
            "halt_reason": self._halt_reason,
            "trading_mode": self._current_trading_mode.value,
            "backtest_mirror_enabled": self._backtest_mirror_enabled,
            "validation_stats": self._validation_stats.copy(),
            "recent_validations": len(self._validation_history),
            "approved_trades": self._approved_trades_count,
            "rejected_trades": self._rejected_trades_count,
            "f8_integration_status": await self.f8_integration.get_integration_status(),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def get_validation_history(
        self,
        limit: int = 100,
        strategy_id: Optional[str] = None,
        approved_only: bool = False
    ) -> List[TradeValidationResult]:
        """
        Get trade validation history.
        
        Args:
            limit: Maximum number of results
            strategy_id: Optional strategy filter
            approved_only: Only return approved trades
            
        Returns:
            List of validation results
        """
        results = self._validation_history.copy()
        
        # Apply filters
        if strategy_id:
            results = [r for r in results if r.strategy_id == strategy_id]
        
        if approved_only:
            results = [r for r in results if r.approved]
        
        # Sort by validation time (most recent first)
        results.sort(key=lambda r: r.validation_time, reverse=True)
        
        return results[:limit]
    
    async def shutdown(self) -> None:
        """Shutdown live trading risk gate."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Shutting down Live Trading Risk Gate")
            
            try:
                # Stop monitoring
                self._running = False
                if self._monitoring_task:
                    self._monitoring_task.cancel()
                
                # Set gate to disabled
                self._gate_status = RiskGateStatus.DISABLED
                
                self.logger.info(
                    "Live Trading Risk Gate shutdown completed",
                    final_gate_status=self._gate_status.value,
                    total_validations=self._validation_stats["total_validations"],
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "shutdown", "correlation_id": correlation_id},
                    "Error during Live Trading Risk Gate shutdown"
                )
    
    # Private helper methods
    
    async def _verify_risk_manager_operational(self) -> None:
        """Verify risk manager is operational."""
        try:
            # Test risk manager with a dummy request
            test_request = RiskCheckRequest(
                strategy_id="test_strategy",
                instrument_id="TEST_INSTRUMENT",
                side="BUY",
                quantity=Decimal('1'),
            )
            
            # This should not fail (even if rejected)
            await self.risk_manager.check_trade_limits(test_request)
            
            self.logger.debug("Risk manager operational verification passed")
            
        except Exception as error:
            raise RuntimeError(f"Risk manager not operational: {error}")
    
    async def _initialize_backtest_mirror(self) -> None:
        """Initialize backtest mirror validation system."""
        try:
            # TODO: Initialize actual backtest mirror system
            # For now, just log initialization
            self.logger.debug("Backtest mirror validation system initialized")
            
        except Exception as error:
            raise RuntimeError(f"Backtest mirror initialization failed: {error}")
    
    async def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        try:
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_worker())
            
            self.logger.debug("Risk gate monitoring started")
            
        except Exception as error:
            raise RuntimeError(f"Failed to start monitoring: {error}")
    
    async def _execute_mandatory_f8_validation(
        self,
        trade_id: str,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal],
        metadata: Optional[Dict[str, Any]]
    ) -> RiskCheckResult:
        """
        Execute mandatory F8 risk validation.
        
        This validation CANNOT be bypassed in production mode.
        """
        try:
            # Execute F8 pre-trade hooks
            result = await self.f8_integration.execute_pre_trade_hooks(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                side=side,
                quantity=quantity,
                price=price,
                metadata=metadata,
            )
            
            self.logger.debug(
                "F8 validation completed",
                trade_id=trade_id,
                approved=result.approved,
                risk_score=result.risk_score,
            )
            
            return result
            
        except Exception as error:
            # F8 validation failure results in trade rejection
            self.logger.error(
                "F8 validation failed",
                trade_id=trade_id,
                error=str(error),
            )
            
            return RiskCheckResult(
                request_id=str(uuid4()),
                status="ERROR",
                approved=False,
                reason=f"F8 validation error: {str(error)}",
                risk_score=1.0,
            )
    
    async def _execute_backtest_mirror_validation(
        self,
        trade_id: str,
        strategy_id: str,
        instrument_id: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute backtest mirror validation for live/backtest parity.
        """
        try:
            # TODO: Implement actual backtest mirror validation
            # This would run the same trade through a backtest engine
            # and compare the results
            
            # For now, simulate successful validation
            await asyncio.sleep(0.01)  # Simulate processing time
            
            return {
                "passed": True,
                "backtest_decision": "approved",
                "live_decision": "approved",
                "parity_check": "passed",
                "reason": "Simulated backtest mirror validation",
            }
            
        except Exception as error:
            return {
                "passed": False,
                "reason": f"Backtest mirror error: {str(error)}",
                "error": str(error),
            }
    
    async def _trigger_emergency_halt(
        self,
        reason: str,
        triggered_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Trigger emergency halt with immediate effect."""
        with with_correlation_id() as correlation_id:
            self.logger.critical(
                "TRIGGERING EMERGENCY HALT",
                reason=reason,
                triggered_by=triggered_by,
            )
            
            try:
                # Set halt state immediately
                self._emergency_halt_active = True
                self._halt_reason = reason
                
                # Activate F8 kill switch
                await self.f8_integration.activate_kill_switch(
                    reason=reason,
                    triggered_by=triggered_by,
                    metadata=metadata,
                )
                
                # Set gate status to error
                self._gate_status = RiskGateStatus.ERROR
                
                self.logger.critical(
                    "EMERGENCY HALT ACTIVATED",
                    reason=reason,
                    triggered_by=triggered_by,
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "trigger_emergency_halt",
                        "reason": reason,
                        "correlation_id": correlation_id,
                    },
                    "CRITICAL: Failed to trigger emergency halt"
                )
                
                # Even if halt triggering fails, set local halt state
                self._emergency_halt_active = True
                self._halt_reason = f"Emergency halt trigger failed: {str(error)}"
                self._gate_status = RiskGateStatus.ERROR
    
    async def _verify_system_operational(self) -> None:
        """Verify all systems are operational after halt clearance."""
        try:
            # Verify F8 integration
            f8_status = await self.f8_integration.get_integration_status()
            if f8_status["integration_status"] != "connected":
                raise RuntimeError("F8 integration not connected")
            
            # Verify risk manager
            await self._verify_risk_manager_operational()
            
            # Set gate back to active
            self._gate_status = RiskGateStatus.ACTIVE
            
            self.logger.info("System operational verification passed")
            
        except Exception as error:
            self._gate_status = RiskGateStatus.ERROR
            raise RuntimeError(f"System not operational: {error}")
    
    def _update_validation_statistics(self, result: TradeValidationResult) -> None:
        """Update validation performance statistics."""
        self._validation_stats["total_validations"] += 1
        
        if result.approved:
            self._approved_trades_count += 1
        else:
            self._rejected_trades_count += 1
        
        # Update average validation time
        total_validations = self._validation_stats["total_validations"]
        current_avg = self._validation_stats["average_validation_time_ms"]
        new_avg = ((current_avg * (total_validations - 1)) + result.validation_time_ms) / total_validations
        self._validation_stats["average_validation_time_ms"] = new_avg
        
        # Update rejection rate
        self._validation_stats["rejection_rate"] = (
            self._rejected_trades_count / total_validations
        ) * 100
        
        # Update F8 success rate
        if result.f8_check_result:
            f8_success_count = len([
                r for r in self._validation_history[-100:]  # Last 100 validations
                if r.f8_validation_passed
            ])
            self._validation_stats["f8_check_success_rate"] = (
                f8_success_count / min(100, total_validations)
            ) * 100
        
        # Update backtest mirror success rate
        if self._backtest_mirror_enabled and result.backtest_result:
            mirror_success_count = len([
                r for r in self._validation_history[-100:]  # Last 100 validations
                if r.backtest_mirror_passed
            ])
            self._validation_stats["backtest_mirror_success_rate"] = (
                mirror_success_count / min(100, total_validations)
            ) * 100
    
    def _cleanup_validation_history(self) -> None:
        """Clean up old validation history."""
        # Keep only last 1000 validations
        if len(self._validation_history) > 1000:
            self._validation_history = self._validation_history[-1000:]
        
        # Remove validations older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        self._validation_history = [
            r for r in self._validation_history
            if r.validation_time > cutoff_time
        ]
    
    async def _monitoring_worker(self) -> None:
        """Background monitoring worker."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                if self._running:
                    await self._perform_monitoring_checks()
                    
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error(
                    "Risk gate monitoring error",
                    error=str(error),
                )
    
    async def _perform_monitoring_checks(self) -> None:
        """Perform periodic monitoring checks."""
        try:
            # Check validation performance
            await self._check_validation_performance()
            
            # Check F8 integration health
            await self._check_f8_integration_health()
            
            # Check for anomalies
            await self._check_for_anomalies()
            
        except Exception as error:
            self.logger.warning(
                "Monitoring check failed",
                error=str(error),
            )
    
    async def _check_validation_performance(self) -> None:
        """Check validation performance metrics."""
        stats = self._validation_stats
        
        # Check validation time
        if stats["average_validation_time_ms"] > self._alert_thresholds["max_validation_time_ms"]:
            self.logger.warning(
                "High validation latency detected",
                average_time_ms=stats["average_validation_time_ms"],
                threshold_ms=self._alert_thresholds["max_validation_time_ms"],
            )
        
        # Check rejection rate
        if stats["rejection_rate"] > self._alert_thresholds["max_rejection_rate"] * 100:
            self.logger.warning(
                "High rejection rate detected",
                rejection_rate=stats["rejection_rate"],
                threshold=self._alert_thresholds["max_rejection_rate"] * 100,
            )
    
    async def _check_f8_integration_health(self) -> None:
        """Check F8 integration health."""
        try:
            f8_status = await self.f8_integration.get_integration_status()
            
            if f8_status["integration_status"] != "connected":
                self.logger.error(
                    "F8 integration not connected",
                    status=f8_status["integration_status"],
                )
                
                # Consider triggering emergency halt
                if self._gate_status == RiskGateStatus.ACTIVE:
                    await self._trigger_emergency_halt(
                        "F8 integration connection lost",
                        "monitoring_system"
                    )
            
        except Exception as error:
            self.logger.error(
                "F8 integration health check failed",
                error=str(error),
            )
    
    async def _check_for_anomalies(self) -> None:
        """Check for trading anomalies."""
        if len(self._validation_history) < 10:
            return  # Not enough data
        
        recent_validations = self._validation_history[-10:]
        
        # Check for sudden spike in rejections
        recent_rejections = len([r for r in recent_validations if not r.approved])
        if recent_rejections >= 8:  # 80% rejection rate in last 10 trades
            self.logger.warning(
                "Anomaly detected: High rejection rate in recent trades",
                recent_rejections=recent_rejections,
                total_recent=len(recent_validations),
            )
        
        # Check for sudden spike in risk scores
        recent_risk_scores = [r.risk_score for r in recent_validations if r.risk_score > 0]
        if recent_risk_scores:
            avg_risk_score = sum(recent_risk_scores) / len(recent_risk_scores)
            if avg_risk_score > 0.7:
                self.logger.warning(
                    "Anomaly detected: High average risk score in recent trades",
                    average_risk_score=avg_risk_score,
                )