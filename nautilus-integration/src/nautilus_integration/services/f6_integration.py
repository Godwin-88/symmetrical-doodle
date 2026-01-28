"""
F6 Strategy Registry Integration

This module provides integration with the F6 Strategy Registry, including
strategy definition parsing, validation, and synchronization with the
Strategy Translation Component.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.core.logging import (
    get_correlation_id,
    get_logger,
    log_error_with_context,
    with_correlation_id,
)
from nautilus_integration.services.strategy_translation import (
    F6StrategyDefinition,
    StrategyTranslationService,
    StrategyTranslationResult,
)


class F6RegistryConnection(BaseModel):
    """Configuration for F6 registry connection."""
    
    host: str = "localhost"
    port: int = 8000
    api_version: str = "v1"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


class F6StrategyUpdate(BaseModel):
    """F6 strategy update notification."""
    
    update_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    update_type: str  # 'created', 'updated', 'deleted', 'activated', 'deactivated'
    timestamp: datetime = Field(default_factory=datetime.now)
    updated_fields: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class F6IntegrationService:
    """
    Service for integrating with F6 Strategy Registry.
    
    This service provides:
    - F6 strategy definition retrieval and parsing
    - Real-time strategy update monitoring
    - Automatic strategy translation triggering
    - Strategy lifecycle management
    - Error handling and recovery
    """
    
    def __init__(
        self,
        config: NautilusConfig,
        translation_service: StrategyTranslationService,
        f6_connection: Optional[F6RegistryConnection] = None
    ):
        """
        Initialize F6 integration service.
        
        Args:
            config: NautilusTrader integration configuration
            translation_service: Strategy translation service
            f6_connection: F6 registry connection configuration
        """
        self.config = config
        self.translation_service = translation_service
        self.f6_connection = f6_connection or F6RegistryConnection()
        self.logger = get_logger("nautilus_integration.f6_integration")
        
        # Strategy tracking
        self._monitored_strategies: Dict[str, F6StrategyDefinition] = {}
        self._translation_results: Dict[str, StrategyTranslationResult] = {}
        self._update_subscriptions: Dict[str, List[callable]] = {}
        
        # Connection state
        self._connected = False
        self._monitoring_active = False
        
        self.logger.info(
            "F6 Integration Service initialized",
            f6_host=self.f6_connection.host,
            f6_port=self.f6_connection.port,
        )
    
    async def initialize(self) -> None:
        """Initialize F6 integration service."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Initializing F6 integration service")
            
            try:
                # Test F6 registry connection
                await self._test_f6_connection()
                
                # Load existing strategy definitions
                await self._load_existing_strategies()
                
                # Start monitoring for updates
                await self._start_update_monitoring()
                
                self._connected = True
                
                self.logger.info(
                    "F6 integration service initialization completed",
                    strategies_loaded=len(self._monitored_strategies),
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "initialize", "correlation_id": correlation_id},
                    "Failed to initialize F6 integration service"
                )
                raise
    
    async def get_f6_strategy(self, strategy_id: str) -> Optional[F6StrategyDefinition]:
        """
        Get F6 strategy definition by ID.
        
        Args:
            strategy_id: F6 strategy identifier
            
        Returns:
            F6 strategy definition if found
            
        Raises:
            RuntimeError: If F6 registry is not accessible
        """
        with with_correlation_id() as correlation_id:
            self.logger.debug(
                "Retrieving F6 strategy",
                strategy_id=strategy_id,
            )
            
            try:
                # Check local cache first
                if strategy_id in self._monitored_strategies:
                    return self._monitored_strategies[strategy_id]
                
                # Fetch from F6 registry
                strategy_data = await self._fetch_strategy_from_f6(strategy_id)
                
                if strategy_data:
                    # Parse and validate strategy definition
                    f6_definition = await self._parse_f6_strategy_data(strategy_data)
                    
                    # Cache the strategy
                    self._monitored_strategies[strategy_id] = f6_definition
                    
                    self.logger.debug(
                        "F6 strategy retrieved successfully",
                        strategy_id=strategy_id,
                        strategy_name=f6_definition.name,
                    )
                    
                    return f6_definition
                else:
                    self.logger.warning(
                        "F6 strategy not found",
                        strategy_id=strategy_id,
                    )
                    return None
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "get_f6_strategy",
                        "strategy_id": strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to retrieve F6 strategy"
                )
                raise RuntimeError(f"Failed to retrieve F6 strategy {strategy_id}: {error}")
    
    async def list_f6_strategies(
        self,
        family: Optional[str] = None,
        production_ready: Optional[bool] = None,
        active_only: bool = True
    ) -> List[F6StrategyDefinition]:
        """
        List F6 strategies with optional filtering.
        
        Args:
            family: Optional strategy family filter
            production_ready: Optional production readiness filter
            active_only: Only return active strategies
            
        Returns:
            List of F6 strategy definitions
        """
        with with_correlation_id() as correlation_id:
            self.logger.debug(
                "Listing F6 strategies",
                family=family,
                production_ready=production_ready,
                active_only=active_only,
            )
            
            try:
                # Fetch strategies from F6 registry
                strategies_data = await self._fetch_strategies_list_from_f6(
                    family=family,
                    production_ready=production_ready,
                    active_only=active_only
                )
                
                strategies = []
                for strategy_data in strategies_data:
                    try:
                        f6_definition = await self._parse_f6_strategy_data(strategy_data)
                        strategies.append(f6_definition)
                        
                        # Update cache
                        self._monitored_strategies[f6_definition.strategy_id] = f6_definition
                        
                    except Exception as error:
                        self.logger.warning(
                            "Failed to parse F6 strategy",
                            strategy_data=strategy_data,
                            error=str(error),
                        )
                        continue
                
                self.logger.debug(
                    "F6 strategies listed successfully",
                    strategies_count=len(strategies),
                )
                
                return strategies
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "list_f6_strategies",
                        "correlation_id": correlation_id,
                    },
                    "Failed to list F6 strategies"
                )
                raise RuntimeError(f"Failed to list F6 strategies: {error}")
    
    async def translate_and_deploy_strategy(
        self,
        strategy_id: str,
        auto_deploy: bool = False
    ) -> StrategyTranslationResult:
        """
        Translate F6 strategy to Nautilus and optionally deploy.
        
        Args:
            strategy_id: F6 strategy identifier
            auto_deploy: Automatically deploy if translation succeeds
            
        Returns:
            Translation result
            
        Raises:
            ValueError: If strategy not found or invalid
            RuntimeError: If translation fails
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Translating and deploying F6 strategy",
                strategy_id=strategy_id,
                auto_deploy=auto_deploy,
            )
            
            try:
                # Get F6 strategy definition
                f6_definition = await self.get_f6_strategy(strategy_id)
                if not f6_definition:
                    raise ValueError(f"F6 strategy not found: {strategy_id}")
                
                # Translate to Nautilus strategy
                translation_result = await self.translation_service.translate_f6_strategy(
                    f6_definition
                )
                
                # Cache translation result
                self._translation_results[strategy_id] = translation_result
                
                # Deploy if requested and translation successful
                if auto_deploy and translation_result.compilation_successful and translation_result.validation_passed:
                    await self._deploy_nautilus_strategy(translation_result)
                
                self.logger.info(
                    "F6 strategy translation completed",
                    strategy_id=strategy_id,
                    translation_id=translation_result.translation_id,
                    success=translation_result.compilation_successful and translation_result.validation_passed,
                    auto_deployed=auto_deploy,
                )
                
                return translation_result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "translate_and_deploy_strategy",
                        "strategy_id": strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to translate and deploy F6 strategy"
                )
                raise
    
    async def subscribe_to_strategy_updates(
        self,
        strategy_id: str,
        callback: callable
    ) -> None:
        """
        Subscribe to F6 strategy update notifications.
        
        Args:
            strategy_id: F6 strategy identifier
            callback: Callback function for updates
        """
        if strategy_id not in self._update_subscriptions:
            self._update_subscriptions[strategy_id] = []
        
        self._update_subscriptions[strategy_id].append(callback)
        
        self.logger.debug(
            "Subscribed to F6 strategy updates",
            strategy_id=strategy_id,
            subscribers_count=len(self._update_subscriptions[strategy_id]),
        )
    
    async def handle_strategy_update(self, update: F6StrategyUpdate) -> None:
        """
        Handle F6 strategy update notification.
        
        Args:
            update: Strategy update notification
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Handling F6 strategy update",
                strategy_id=update.strategy_id,
                update_type=update.update_type,
                update_id=update.update_id,
            )
            
            try:
                if update.update_type == 'updated':
                    # Re-fetch and re-translate strategy
                    await self._handle_strategy_updated(update)
                    
                elif update.update_type == 'deleted':
                    # Clean up strategy
                    await self._handle_strategy_deleted(update)
                    
                elif update.update_type == 'activated':
                    # Deploy strategy if not already deployed
                    await self._handle_strategy_activated(update)
                    
                elif update.update_type == 'deactivated':
                    # Stop strategy execution
                    await self._handle_strategy_deactivated(update)
                
                # Notify subscribers
                await self._notify_update_subscribers(update)
                
                self.logger.info(
                    "F6 strategy update handled successfully",
                    strategy_id=update.strategy_id,
                    update_type=update.update_type,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "handle_strategy_update",
                        "strategy_id": update.strategy_id,
                        "update_type": update.update_type,
                        "correlation_id": correlation_id,
                    },
                    "Failed to handle F6 strategy update"
                )
                # Don't re-raise to avoid breaking update processing
    
    async def get_translation_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get translation status for F6 strategy.
        
        Args:
            strategy_id: F6 strategy identifier
            
        Returns:
            Translation status information
        """
        translation_result = self._translation_results.get(strategy_id)
        
        if translation_result:
            return {
                "strategy_id": strategy_id,
                "translation_id": translation_result.translation_id,
                "nautilus_strategy_id": translation_result.nautilus_strategy_id,
                "compilation_successful": translation_result.compilation_successful,
                "validation_passed": translation_result.validation_passed,
                "translation_time": translation_result.translation_time.isoformat(),
                "errors_count": len(translation_result.validation_errors) + len(translation_result.compilation_errors),
                "warnings_count": len(translation_result.safety_warnings),
            }
        
        return None
    
    async def shutdown(self) -> None:
        """Shutdown F6 integration service."""
        with with_correlation_id() as correlation_id:
            self.logger.info("Shutting down F6 integration service")
            
            try:
                # Stop update monitoring
                self._monitoring_active = False
                
                # Clear caches
                self._monitored_strategies.clear()
                self._translation_results.clear()
                self._update_subscriptions.clear()
                
                self._connected = False
                
                self.logger.info(
                    "F6 integration service shutdown completed",
                    correlation_id=correlation_id,
                )
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {"operation": "shutdown", "correlation_id": correlation_id},
                    "Error during F6 integration service shutdown"
                )
    
    # Private helper methods
    
    async def _test_f6_connection(self) -> None:
        """Test connection to F6 registry."""
        try:
            self.logger.debug("Testing F6 registry connection")
            
            # TODO: Implement actual F6 registry connection test
            # This would make an HTTP request to F6 registry health endpoint
            
            # Simulate connection test
            await asyncio.sleep(0.1)
            
            self.logger.debug("F6 registry connection test successful")
            
        except Exception as error:
            self.logger.error("F6 registry connection test failed", error=str(error))
            raise RuntimeError(f"F6 registry connection failed: {error}")
    
    async def _load_existing_strategies(self) -> None:
        """Load existing strategies from F6 registry."""
        try:
            self.logger.debug("Loading existing F6 strategies")
            
            # Get list of all strategies
            strategies = await self.list_f6_strategies()
            
            self.logger.info(
                "Existing F6 strategies loaded",
                strategies_count=len(strategies),
            )
            
        except Exception as error:
            self.logger.warning(
                "Failed to load existing F6 strategies",
                error=str(error),
            )
            # Don't fail initialization for this
    
    async def _start_update_monitoring(self) -> None:
        """Start monitoring F6 registry for strategy updates."""
        try:
            self.logger.debug("Starting F6 update monitoring")
            
            self._monitoring_active = True
            
            # TODO: Implement actual F6 update monitoring
            # This would establish WebSocket connection or polling mechanism
            
            self.logger.debug("F6 update monitoring started")
            
        except Exception as error:
            self.logger.warning(
                "Failed to start F6 update monitoring",
                error=str(error),
            )
            # Don't fail initialization for this
    
    async def _fetch_strategy_from_f6(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Fetch strategy data from F6 registry."""
        try:
            self.logger.debug(
                "Fetching strategy from F6 registry",
                strategy_id=strategy_id,
            )
            
            # TODO: Implement actual F6 API call
            # This would make HTTP request to F6 registry API
            
            # For now, return mock data based on strategy registry
            from intelligence_layer.strategy_registry import strategy_registry
            
            strategy_spec = strategy_registry.get_strategy(strategy_id)
            if strategy_spec:
                return {
                    "strategy_id": strategy_spec.id,
                    "name": strategy_spec.name,
                    "family": strategy_spec.family.value,
                    "horizon": strategy_spec.horizon.value,
                    "asset_classes": [ac.value for ac in strategy_spec.asset_classes],
                    "description": strategy_spec.description,
                    "signal_logic": strategy_spec.signal_logic,
                    "entry_rules": strategy_spec.entry_rules,
                    "exit_rules": strategy_spec.exit_rules,
                    "risk_controls": strategy_spec.risk_controls,
                    "parameters": strategy_spec.parameters,
                    "typical_sharpe": strategy_spec.typical_sharpe,
                    "typical_max_dd": strategy_spec.typical_max_dd,
                    "typical_win_rate": strategy_spec.typical_win_rate,
                    "max_position_size": strategy_spec.max_position_size,
                    "max_leverage": strategy_spec.max_leverage,
                    "stop_loss_pct": strategy_spec.stop_loss_pct,
                    "production_ready": strategy_spec.production_ready,
                    "complexity": strategy_spec.complexity,
                }
            
            return None
            
        except Exception as error:
            self.logger.error(
                "Failed to fetch strategy from F6 registry",
                strategy_id=strategy_id,
                error=str(error),
            )
            raise
    
    async def _fetch_strategies_list_from_f6(
        self,
        family: Optional[str] = None,
        production_ready: Optional[bool] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch strategies list from F6 registry."""
        try:
            self.logger.debug(
                "Fetching strategies list from F6 registry",
                family=family,
                production_ready=production_ready,
                active_only=active_only,
            )
            
            # TODO: Implement actual F6 API call
            # For now, use strategy registry
            from intelligence_layer.strategy_registry import strategy_registry, StrategyFamily
            
            # Convert family string to enum if provided
            family_enum = None
            if family:
                try:
                    family_enum = StrategyFamily(family)
                except ValueError:
                    self.logger.warning(f"Invalid strategy family: {family}")
            
            strategies = strategy_registry.list_strategies(
                family=family_enum,
                production_ready=production_ready
            )
            
            strategies_data = []
            for strategy_spec in strategies:
                strategy_data = {
                    "strategy_id": strategy_spec.id,
                    "name": strategy_spec.name,
                    "family": strategy_spec.family.value,
                    "horizon": strategy_spec.horizon.value,
                    "asset_classes": [ac.value for ac in strategy_spec.asset_classes],
                    "description": strategy_spec.description,
                    "signal_logic": strategy_spec.signal_logic,
                    "entry_rules": strategy_spec.entry_rules,
                    "exit_rules": strategy_spec.exit_rules,
                    "risk_controls": strategy_spec.risk_controls,
                    "parameters": strategy_spec.parameters,
                    "typical_sharpe": strategy_spec.typical_sharpe,
                    "typical_max_dd": strategy_spec.typical_max_dd,
                    "typical_win_rate": strategy_spec.typical_win_rate,
                    "max_position_size": strategy_spec.max_position_size,
                    "max_leverage": strategy_spec.max_leverage,
                    "stop_loss_pct": strategy_spec.stop_loss_pct,
                    "production_ready": strategy_spec.production_ready,
                    "complexity": strategy_spec.complexity,
                }
                strategies_data.append(strategy_data)
            
            return strategies_data
            
        except Exception as error:
            self.logger.error(
                "Failed to fetch strategies list from F6 registry",
                error=str(error),
            )
            raise
    
    async def _parse_f6_strategy_data(self, strategy_data: Dict[str, Any]) -> F6StrategyDefinition:
        """Parse F6 strategy data into definition object."""
        try:
            return F6StrategyDefinition(**strategy_data)
            
        except Exception as error:
            self.logger.error(
                "Failed to parse F6 strategy data",
                strategy_data=strategy_data,
                error=str(error),
            )
            raise ValueError(f"Invalid F6 strategy data: {error}")
    
    async def _deploy_nautilus_strategy(self, translation_result: StrategyTranslationResult) -> None:
        """Deploy translated Nautilus strategy."""
        try:
            self.logger.info(
                "Deploying Nautilus strategy",
                translation_id=translation_result.translation_id,
                nautilus_strategy_id=translation_result.nautilus_strategy_id,
            )
            
            # TODO: Implement actual strategy deployment
            # This would involve:
            # 1. Saving generated code to appropriate location
            # 2. Registering strategy with Nautilus engine
            # 3. Setting up monitoring and logging
            # 4. Notifying deployment system
            
            self.logger.info(
                "Nautilus strategy deployed successfully",
                translation_id=translation_result.translation_id,
            )
            
        except Exception as error:
            self.logger.error(
                "Failed to deploy Nautilus strategy",
                translation_id=translation_result.translation_id,
                error=str(error),
            )
            raise RuntimeError(f"Strategy deployment failed: {error}")
    
    async def _handle_strategy_updated(self, update: F6StrategyUpdate) -> None:
        """Handle strategy updated notification."""
        try:
            # Re-translate strategy
            translation_result = await self.translate_and_deploy_strategy(
                update.strategy_id,
                auto_deploy=True
            )
            
            self.logger.info(
                "Strategy update handled - re-translated",
                strategy_id=update.strategy_id,
                translation_id=translation_result.translation_id,
            )
            
        except Exception as error:
            self.logger.error(
                "Failed to handle strategy update",
                strategy_id=update.strategy_id,
                error=str(error),
            )
    
    async def _handle_strategy_deleted(self, update: F6StrategyUpdate) -> None:
        """Handle strategy deleted notification."""
        try:
            # Remove from caches
            self._monitored_strategies.pop(update.strategy_id, None)
            self._translation_results.pop(update.strategy_id, None)
            
            # TODO: Stop any running instances of the strategy
            
            self.logger.info(
                "Strategy deletion handled",
                strategy_id=update.strategy_id,
            )
            
        except Exception as error:
            self.logger.error(
                "Failed to handle strategy deletion",
                strategy_id=update.strategy_id,
                error=str(error),
            )
    
    async def _handle_strategy_activated(self, update: F6StrategyUpdate) -> None:
        """Handle strategy activated notification."""
        try:
            # Deploy strategy if not already deployed
            await self.translate_and_deploy_strategy(
                update.strategy_id,
                auto_deploy=True
            )
            
            self.logger.info(
                "Strategy activation handled",
                strategy_id=update.strategy_id,
            )
            
        except Exception as error:
            self.logger.error(
                "Failed to handle strategy activation",
                strategy_id=update.strategy_id,
                error=str(error),
            )
    
    async def _handle_strategy_deactivated(self, update: F6StrategyUpdate) -> None:
        """Handle strategy deactivated notification."""
        try:
            # TODO: Stop strategy execution
            
            self.logger.info(
                "Strategy deactivation handled",
                strategy_id=update.strategy_id,
            )
            
        except Exception as error:
            self.logger.error(
                "Failed to handle strategy deactivation",
                strategy_id=update.strategy_id,
                error=str(error),
            )
    
    async def _notify_update_subscribers(self, update: F6StrategyUpdate) -> None:
        """Notify subscribers of strategy update."""
        subscribers = self._update_subscriptions.get(update.strategy_id, [])
        
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(update)
                else:
                    callback(update)
                    
            except Exception as error:
                self.logger.warning(
                    "Failed to notify update subscriber",
                    strategy_id=update.strategy_id,
                    error=str(error),
                )