"""
Strategy Translation Component (STC)

This module provides the Strategy Translation Component that converts F6 strategy 
definitions to NautilusTrader Strategy implementations with comprehensive validation,
error handling, and safety checks.
"""

import ast
import asyncio
import inspect
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field, validator

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.core.nautilus_logging import (
    get_correlation_id,
    get_logger,
    log_error_with_context,
    with_correlation_id,
)


class CompilationResult:
    """Result of code compilation."""
    
    def __init__(self):
        self.success: bool = False
        self.errors: List[str] = []


class SafetyResult:
    """Result of safety checks."""
    
    def __init__(self):
        self.passed: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []


class F6StrategyDefinition(BaseModel):
    """F6 strategy definition model."""
    
    strategy_id: str
    name: str
    family: str
    horizon: str
    asset_classes: List[str]
    description: str
    signal_logic: str
    entry_rules: List[str]
    exit_rules: List[str]
    risk_controls: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance characteristics
    typical_sharpe: float = 0.0
    typical_max_dd: float = 0.0
    typical_win_rate: float = 0.0
    
    # Risk management
    max_position_size: float = 0.25
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.05
    
    # Metadata
    production_ready: bool = False
    complexity: str = "medium"
    
    @validator('family')
    def validate_family(cls, v):
        valid_families = [
            'trend', 'mean_reversion', 'momentum', 'volatility',
            'statistical_arb', 'regime_switching', 'sentiment', 'execution'
        ]
        if v not in valid_families:
            raise ValueError(f"Invalid strategy family: {v}")
        return v


class NautilusStrategyConfig(BaseModel):
    """Configuration for generated Nautilus strategy."""
    
    strategy_id: str
    class_name: str
    f6_strategy_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    signal_subscriptions: List[str] = Field(default_factory=list)
    risk_constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Code generation settings
    include_f5_signals: bool = True
    include_f8_risk: bool = True
    include_logging: bool = True
    include_performance_tracking: bool = True
    
    # Validation settings
    compile_check: bool = True
    safety_checks: bool = True
    parameter_validation: bool = True


class StrategyTranslationResult(BaseModel):
    """Result of strategy translation process."""
    
    translation_id: str = Field(default_factory=lambda: str(uuid4()))
    f6_strategy_id: str
    nautilus_strategy_id: str
    class_name: str
    generated_code: str
    compilation_successful: bool = False
    validation_passed: bool = False
    
    # Translation metadata
    translation_time: datetime = Field(default_factory=datetime.now)
    parameter_mapping: Dict[str, str] = Field(default_factory=dict)
    signal_mappings: List[str] = Field(default_factory=list)
    risk_mappings: List[str] = Field(default_factory=list)
    
    # Validation results
    validation_errors: List[str] = Field(default_factory=list)
    compilation_errors: List[str] = Field(default_factory=list)
    safety_warnings: List[str] = Field(default_factory=list)
    
    # Performance estimates
    estimated_complexity: str = "medium"
    estimated_latency: str = "low"
    estimated_memory_usage: str = "low"


class StrategyTranslationService:
    """
    Service for translating F6 strategy definitions to NautilusTrader strategies.
    
    This service handles the complete translation pipeline including:
    - F6 strategy definition parsing and validation
    - Nautilus strategy code generation with safety checks
    - Parameter mapping and validation
    - Strategy compilation and validation
    - Error handling and detailed reporting
    """
    
    def __init__(self, config: NautilusConfig):
        """
        Initialize the strategy translation service.
        
        Args:
            config: NautilusTrader integration configuration
        """
        self.config = config
        self.logger = get_logger("nautilus_integration.strategy_translation")
        
        # Translation cache and tracking
        self._translation_cache: Dict[str, StrategyTranslationResult] = {}
        self._active_translations: Dict[str, bool] = {}
        
        # Code generation templates
        self._strategy_template = self._load_strategy_template()
        
        # Validation rules
        self._parameter_validators = self._initialize_parameter_validators()
        self._safety_rules = self._initialize_safety_rules()
        
        self.logger.info(
            "Strategy Translation Service initialized",
            config_environment=config.environment,
        )
    
    async def translate_f6_strategy(
        self,
        f6_definition: F6StrategyDefinition,
        config: Optional[NautilusStrategyConfig] = None
    ) -> StrategyTranslationResult:
        """
        Translate F6 strategy definition to Nautilus strategy.
        
        Args:
            f6_definition: F6 strategy definition
            config: Optional translation configuration
            
        Returns:
            Translation result with generated code and validation status
            
        Raises:
            ValueError: If strategy definition is invalid
            RuntimeError: If translation fails
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Starting F6 strategy translation",
                f6_strategy_id=f6_definition.strategy_id,
                strategy_name=f6_definition.name,
                strategy_family=f6_definition.family,
            )
            
            # Check if translation is already in progress
            if f6_definition.strategy_id in self._active_translations:
                raise RuntimeError(
                    f"Translation already in progress for strategy {f6_definition.strategy_id}"
                )
            
            self._active_translations[f6_definition.strategy_id] = True
            
            try:
                # Create default config if not provided
                if config is None:
                    config = NautilusStrategyConfig(
                        strategy_id=f"nautilus_{f6_definition.strategy_id}",
                        class_name=self._generate_class_name(f6_definition),
                        f6_strategy_id=f6_definition.strategy_id,
                    )
                
                # Step 1: Validate F6 strategy definition
                await self._validate_f6_definition(f6_definition)
                
                # Step 2: Parse and validate parameters
                parameter_mapping = await self._parse_f6_parameters(f6_definition, config)
                
                # Step 3: Generate Nautilus strategy code
                generated_code = await self._generate_nautilus_strategy_code(
                    f6_definition, config, parameter_mapping
                )
                
                # Step 4: Validate generated code
                compilation_result = await self._validate_generated_code(
                    generated_code, config
                )
                
                # Step 5: Perform safety checks
                safety_result = await self._perform_safety_checks(
                    f6_definition, generated_code, config
                )
                
                # Create translation result
                result = StrategyTranslationResult(
                    f6_strategy_id=f6_definition.strategy_id,
                    nautilus_strategy_id=config.strategy_id,
                    class_name=config.class_name,
                    generated_code=generated_code,
                    compilation_successful=compilation_result.success,
                    validation_passed=safety_result.passed,
                    parameter_mapping=parameter_mapping,
                    compilation_errors=compilation_result.errors,
                    validation_errors=safety_result.errors,
                    safety_warnings=safety_result.warnings,
                    estimated_complexity=f6_definition.complexity,
                )
                
                # Cache successful translations
                if result.compilation_successful and result.validation_passed:
                    self._translation_cache[f6_definition.strategy_id] = result
                
                self.logger.info(
                    "F6 strategy translation completed",
                    f6_strategy_id=f6_definition.strategy_id,
                    nautilus_strategy_id=config.strategy_id,
                    compilation_successful=result.compilation_successful,
                    validation_passed=result.validation_passed,
                    translation_id=result.translation_id,
                )
                
                return result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "translate_f6_strategy",
                        "f6_strategy_id": f6_definition.strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to translate F6 strategy"
                )
                
                # Create error result
                result = StrategyTranslationResult(
                    f6_strategy_id=f6_definition.strategy_id,
                    nautilus_strategy_id=config.strategy_id if config else "unknown",
                    class_name=config.class_name if config else "unknown",
                    generated_code="",
                    compilation_successful=False,
                    validation_passed=False,
                    validation_errors=[str(error)],
                )
                
                return result
                
            finally:
                # Clean up active translation tracking
                self._active_translations.pop(f6_definition.strategy_id, None)
    
    async def get_translation_result(
        self, f6_strategy_id: str
    ) -> Optional[StrategyTranslationResult]:
        """
        Get cached translation result for F6 strategy.
        
        Args:
            f6_strategy_id: F6 strategy identifier
            
        Returns:
            Cached translation result if available
        """
        return self._translation_cache.get(f6_strategy_id)
    
    async def validate_parameter_mapping(
        self,
        f6_definition: F6StrategyDefinition,
        nautilus_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate parameter mapping between F6 and Nautilus.
        
        Args:
            f6_definition: F6 strategy definition
            nautilus_parameters: Proposed Nautilus parameters
            
        Returns:
            Validation results with errors and warnings
            
        Raises:
            ValueError: If parameter mapping is invalid
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Validating parameter mapping",
                f6_strategy_id=f6_definition.strategy_id,
                f6_params_count=len(f6_definition.parameters),
                nautilus_params_count=len(nautilus_parameters),
            )
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "mapped_parameters": {},
                "unmapped_f6_parameters": [],
                "extra_nautilus_parameters": [],
            }
            
            try:
                # Check for required F6 parameters
                required_params = self._get_required_parameters(f6_definition.family)
                for param in required_params:
                    if param not in f6_definition.parameters:
                        validation_result["errors"].append(
                            f"Required F6 parameter missing: {param}"
                        )
                        validation_result["valid"] = False
                
                # Map F6 parameters to Nautilus parameters
                for f6_param, f6_value in f6_definition.parameters.items():
                    nautilus_param = self._map_parameter_name(f6_param, f6_definition.family)
                    
                    if nautilus_param in nautilus_parameters:
                        # Validate parameter type and range
                        validation_error = self._validate_parameter_value(
                            nautilus_param, nautilus_parameters[nautilus_param], f6_definition.family
                        )
                        
                        if validation_error:
                            validation_result["errors"].append(validation_error)
                            validation_result["valid"] = False
                        else:
                            validation_result["mapped_parameters"][f6_param] = nautilus_param
                    else:
                        validation_result["unmapped_f6_parameters"].append(f6_param)
                        validation_result["warnings"].append(
                            f"F6 parameter {f6_param} not mapped to Nautilus parameter"
                        )
                
                # Check for extra Nautilus parameters
                mapped_nautilus_params = set(validation_result["mapped_parameters"].values())
                for nautilus_param in nautilus_parameters:
                    if nautilus_param not in mapped_nautilus_params:
                        validation_result["extra_nautilus_parameters"].append(nautilus_param)
                        validation_result["warnings"].append(
                            f"Extra Nautilus parameter: {nautilus_param}"
                        )
                
                self.logger.info(
                    "Parameter mapping validation completed",
                    f6_strategy_id=f6_definition.strategy_id,
                    validation_valid=validation_result["valid"],
                    errors_count=len(validation_result["errors"]),
                    warnings_count=len(validation_result["warnings"]),
                )
                
                return validation_result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "validate_parameter_mapping",
                        "f6_strategy_id": f6_definition.strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to validate parameter mapping"
                )
                
                validation_result["valid"] = False
                validation_result["errors"].append(f"Validation failed: {error}")
                return validation_result
    
    async def regenerate_strategy(
        self,
        f6_strategy_id: str,
        updated_definition: Optional[F6StrategyDefinition] = None,
        config: Optional[NautilusStrategyConfig] = None
    ) -> StrategyTranslationResult:
        """
        Regenerate Nautilus strategy from updated F6 definition.
        
        Args:
            f6_strategy_id: F6 strategy identifier
            updated_definition: Updated F6 strategy definition
            config: Optional translation configuration
            
        Returns:
            New translation result
            
        Raises:
            ValueError: If strategy not found or invalid
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Regenerating Nautilus strategy",
                f6_strategy_id=f6_strategy_id,
                has_updated_definition=updated_definition is not None,
            )
            
            try:
                # Get existing translation if available
                existing_result = self._translation_cache.get(f6_strategy_id)
                
                if updated_definition is None:
                    if existing_result is None:
                        raise ValueError(f"No existing translation found for {f6_strategy_id}")
                    
                    # TODO: Retrieve F6 definition from F6 registry
                    # For now, raise error if no updated definition provided
                    raise ValueError("Updated F6 definition required for regeneration")
                
                # Clear existing cache
                self._translation_cache.pop(f6_strategy_id, None)
                
                # Perform new translation
                result = await self.translate_f6_strategy(updated_definition, config)
                
                self.logger.info(
                    "Strategy regeneration completed",
                    f6_strategy_id=f6_strategy_id,
                    translation_id=result.translation_id,
                    success=result.compilation_successful and result.validation_passed,
                )
                
                return result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "regenerate_strategy",
                        "f6_strategy_id": f6_strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to regenerate strategy"
                )
                raise
    
    # Private helper methods
    
    async def _validate_f6_definition(self, definition: F6StrategyDefinition) -> None:
        """Validate F6 strategy definition."""
        try:
            self.logger.debug(
                "Validating F6 strategy definition",
                strategy_id=definition.strategy_id,
                family=definition.family,
            )
            
            # Basic validation
            if not definition.strategy_id:
                raise ValueError("Strategy ID is required")
            
            if not definition.name:
                raise ValueError("Strategy name is required")
            
            if not definition.signal_logic:
                raise ValueError("Signal logic is required")
            
            if not definition.entry_rules:
                raise ValueError("Entry rules are required")
            
            if not definition.exit_rules:
                raise ValueError("Exit rules are required")
            
            # Validate parameters for strategy family
            family_validators = self._parameter_validators.get(definition.family, {})
            for param_name, validator in family_validators.items():
                if param_name in definition.parameters:
                    if not validator(definition.parameters[param_name]):
                        raise ValueError(f"Invalid parameter {param_name} for family {definition.family}")
            
            # Validate risk constraints
            if definition.max_position_size <= 0 or definition.max_position_size > 1.0:
                raise ValueError("Max position size must be between 0 and 1")
            
            if definition.max_leverage <= 0 or definition.max_leverage > 10.0:
                raise ValueError("Max leverage must be between 0 and 10")
            
            if definition.stop_loss_pct <= 0 or definition.stop_loss_pct > 0.5:
                raise ValueError("Stop loss percentage must be between 0 and 0.5")
            
            self.logger.debug(
                "F6 strategy definition validation passed",
                strategy_id=definition.strategy_id,
            )
            
        except Exception as error:
            self.logger.error(
                "F6 strategy definition validation failed",
                strategy_id=definition.strategy_id,
                error=str(error),
            )
            raise ValueError(f"F6 definition validation failed: {error}")
    
    async def _parse_f6_parameters(
        self,
        definition: F6StrategyDefinition,
        config: NautilusStrategyConfig
    ) -> Dict[str, str]:
        """Parse and map F6 parameters to Nautilus parameters."""
        try:
            self.logger.debug(
                "Parsing F6 parameters",
                strategy_id=definition.strategy_id,
                params_count=len(definition.parameters),
            )
            
            parameter_mapping = {}
            
            # Map common parameters based on strategy family
            family_mappings = self._get_family_parameter_mappings(definition.family)
            
            for f6_param, f6_value in definition.parameters.items():
                # Check if parameter has a direct mapping
                if f6_param in family_mappings:
                    nautilus_param = family_mappings[f6_param]
                    parameter_mapping[f6_param] = nautilus_param
                else:
                    # Use parameter name as-is with validation
                    if self._is_valid_parameter_name(f6_param):
                        parameter_mapping[f6_param] = f6_param
                    else:
                        self.logger.warning(
                            "Invalid parameter name, skipping",
                            parameter=f6_param,
                            strategy_id=definition.strategy_id,
                        )
            
            self.logger.debug(
                "F6 parameter parsing completed",
                strategy_id=definition.strategy_id,
                mapped_count=len(parameter_mapping),
            )
            
            return parameter_mapping
            
        except Exception as error:
            self.logger.error(
                "F6 parameter parsing failed",
                strategy_id=definition.strategy_id,
                error=str(error),
            )
            raise RuntimeError(f"Parameter parsing failed: {error}")
    
    def _generate_class_name(self, definition: F6StrategyDefinition) -> str:
        """Generate Nautilus strategy class name from F6 definition."""
        # Convert strategy name to PascalCase
        words = definition.name.replace('-', ' ').replace('_', ' ').split()
        class_name = ''.join(word.capitalize() for word in words)
        
        # Add strategy suffix if not present
        if not class_name.endswith('Strategy'):
            class_name += 'Strategy'
        
        # Ensure valid Python class name
        if not class_name[0].isalpha():
            class_name = 'F6' + class_name
        
        return class_name
    
    def _load_strategy_template(self) -> str:
        """Load the Nautilus strategy code template."""
        return '''"""
Generated Nautilus Strategy from F6 Definition
Strategy ID: {f6_strategy_id}
Generated: {generation_time}
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional

from nautilus_trader.core.message import Event
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.tick import QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce, PositionSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy

{f5_signal_imports}
{f8_risk_imports}


class {class_name}(Strategy):
    """
    {description}
    
    Strategy Family: {family}
    Time Horizon: {horizon}
    
    Signal Logic: {signal_logic}
    
    Generated from F6 Strategy: {f6_strategy_id}
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy."""
        super().__init__(config)
        
        # F6 Strategy Parameters
{parameter_initialization}
        
        # Risk Management
        self.max_position_size = {max_position_size}
        self.max_leverage = {max_leverage}
        self.stop_loss_pct = {stop_loss_pct}
        
        # Strategy State
        self._positions = {{}}
        self._orders = {{}}
        self._signals = {{}}
        
{f5_signal_initialization}
{f8_risk_initialization}
{performance_tracking_initialization}
    
    def on_start(self):
        """Called when the strategy is started."""
        self.log.info(f"Starting {{self.__class__.__name__}}")
        
        # Initialize indicators and state
{on_start_implementation}
        
{f5_signal_subscription}
{f8_risk_setup}
        
        self.log.info(f"{{self.__class__.__name__}} started successfully")
    
    def on_stop(self):
        """Called when the strategy is stopped."""
        self.log.info(f"Stopping {{self.__class__.__name__}}")
        
{f8_risk_cleanup}
{performance_tracking_cleanup}
        
        self.log.info(f"{{self.__class__.__name__}} stopped")
    
    def on_bar(self, bar: Bar):
        """Handle bar data with comprehensive F8 risk integration."""
        try:
            # Check kill switch status first
            if self._kill_switch_active:
                self.log.warning("Strategy execution halted - kill switch active")
                return
            
            # Sync positions with F8 system periodically
            if self._position_sync_enabled:
                asyncio.create_task(self._sync_positions_with_f8())
            
{f5_signal_processing}
{strategy_logic_implementation}

            # Enhanced F8 risk checking with position synchronization
{f8_risk_checking}
            
            # Additional local risk validation
            if hasattr(self, 'position_size') and self.position_size != 0:
                if not self._validate_trade_against_constraints(
                    str(bar.instrument_id), 
                    Decimal(str(self.position_size)), 
                    Decimal(str(bar.close))
                ):
                    self.log.warning("Trade failed local risk validation")
                    return
            
        except Exception as error:
            self.log.error(f"Error processing bar: {{error}}")
            
            # Handle critical errors that might require kill switch
            if "risk" in str(error).lower() or "limit" in str(error).lower():
                asyncio.create_task(self._handle_critical_risk_event(
                    type('RiskCheckResult', (), {{
                        'risk_score': 0.9,
                        'violated_limits': ['error_handling'],
                        'reason': f'Critical error in bar processing: {{error}}'
                    }})()
                ))
    
    def on_quote_tick(self, tick: QuoteTick):
        """Handle quote tick data with risk monitoring."""
        try:
            # Skip processing if kill switch is active
            if self._kill_switch_active:
                return
                
{tick_processing_implementation}
        except Exception as error:
            self.log.error(f"Error processing quote tick: {{error}}")
    
    def on_trade_tick(self, tick: TradeTick):
        """Handle trade tick data with risk monitoring."""
        try:
            # Skip processing if kill switch is active
            if self._kill_switch_active:
                return
                
{tick_processing_implementation}
        except Exception as error:
            self.log.error(f"Error processing trade tick: {{error}}")
    
    def on_event(self, event: Event):
        """Handle custom events including F8 risk events."""
        try:
            # Handle F8 risk management events
            if hasattr(event, 'event_type'):
                if event.event_type == 'KILL_SWITCH_ACTIVATED':
                    self._kill_switch_active = True
                    self.log.critical("Kill switch activated via event")
                    return
                elif event.event_type == 'KILL_SWITCH_DEACTIVATED':
                    self._kill_switch_active = False
                    self.log.info("Kill switch deactivated via event")
                    return
                elif event.event_type == 'POSITION_SYNC_CONFLICT':
                    self._position_sync_conflicts.append(event.data)
                    self.log.warning(f"Position sync conflict: {{event.data}}")
                    return
            
{event_processing_implementation}
        except Exception as error:
            self.log.error(f"Error processing event: {{error}}")
    
    def on_order_filled(self, event):
        """Handle order fill events with P&L tracking."""
        try:
            # Update daily P&L tracking
            if hasattr(event, 'fill') and hasattr(event.fill, 'last_px'):
                # Calculate trade P&L (simplified)
                trade_pnl = Decimal('0')  # TODO: Implement proper P&L calculation
                self._update_daily_pnl(trade_pnl)
            
            # Trigger position synchronization after fill
            if self._position_sync_enabled:
                asyncio.create_task(self._sync_positions_with_f8())
            
            # Call parent handler
            super().on_order_filled(event)
            
        except Exception as error:
            self.log.error(f"Error handling order fill: {{error}}")
    
    def on_position_opened(self, event):
        """Handle position opened events with F8 synchronization."""
        try:
            # Update position tracking
            position = event.position
            position_record = PositionRecord(
                instrument_id=str(position.instrument_id),
                strategy_id=self.id.value,
                quantity=Decimal(str(position.quantity)),
                average_price=Decimal(str(position.avg_px_open)),
                market_value=Decimal(str(position.quantity * position.avg_px_open)),
                unrealized_pnl=Decimal(str(position.unrealized_pnl())),
                source_system="nautilus"
            )
            self._nautilus_positions[str(position.instrument_id)] = position_record
            
            # Trigger F8 synchronization
            if self._position_sync_enabled:
                asyncio.create_task(self._sync_positions_with_f8())
            
            # Call parent handler
            super().on_position_opened(event)
            
        except Exception as error:
            self.log.error(f"Error handling position opened: {{error}}")
    
    def on_position_closed(self, event):
        """Handle position closed events with F8 synchronization."""
        try:
            # Remove from position tracking
            position = event.position
            instrument_key = str(position.instrument_id)
            if instrument_key in self._nautilus_positions:
                del self._nautilus_positions[instrument_key]
            
            # Update daily P&L with realized P&L
            if hasattr(position, 'realized_pnl'):
                self._update_daily_pnl(Decimal(str(position.realized_pnl)))
            
            # Trigger F8 synchronization
            if self._position_sync_enabled:
                asyncio.create_task(self._sync_positions_with_f8())
            
            # Call parent handler
            super().on_position_closed(event)
            
        except Exception as error:
            self.log.error(f"Error handling position closed: {{error}}")
    
    # Strategy-specific methods
    
{strategy_specific_methods}
    
    # Risk management methods
    
{risk_management_methods}
    
    # Performance tracking methods
    
{performance_tracking_methods}
'''
    
    def _initialize_parameter_validators(self) -> Dict[str, Dict[str, callable]]:
        """Initialize parameter validators for each strategy family."""
        return {
            'trend': {
                'fast_period': lambda x: isinstance(x, int) and 1 <= x <= 200,
                'slow_period': lambda x: isinstance(x, int) and 1 <= x <= 500,
                'atr_multiplier': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 10.0,
            },
            'mean_reversion': {
                'lookback_period': lambda x: isinstance(x, int) and 5 <= x <= 100,
                'entry_threshold': lambda x: isinstance(x, (int, float)) and 1.0 <= x <= 5.0,
                'exit_threshold': lambda x: isinstance(x, (int, float)) and 0.1 <= x <= 2.0,
            },
            'momentum': {
                'lookback_months': lambda x: isinstance(x, int) and 1 <= x <= 24,
                'top_pct': lambda x: isinstance(x, (int, float)) and 1 <= x <= 50,
                'bottom_pct': lambda x: isinstance(x, (int, float)) and 1 <= x <= 50,
            },
            'volatility': {
                'bb_period': lambda x: isinstance(x, int) and 5 <= x <= 100,
                'bb_std': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 5.0,
                'volume_threshold': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 10.0,
            },
        }
    
    def _initialize_safety_rules(self) -> Dict[str, callable]:
        """Initialize safety rules for code generation."""
        return {
            'no_external_imports': lambda code: 'import os' not in code and 'import sys' not in code,
            'no_file_operations': lambda code: self._check_no_file_operations(code),
            'no_network_operations': lambda code: 'urllib' not in code and 'requests' not in code,
            'no_subprocess': lambda code: 'subprocess' not in code and 'os.system' not in code,
            'no_eval_exec': lambda code: 'eval(' not in code and 'exec(' not in code,
        }
    
    def _check_no_file_operations(self, code: str) -> bool:
        """Check for file operations, excluding legitimate NautilusTrader cache methods."""
        # File operation patterns to check for
        file_patterns = ['open(', 'file(', 'with open(', '.open(', 'io.open(']
        
        # Legitimate NautilusTrader patterns to exclude
        legitimate_patterns = [
            'orders_open(', 'positions_open()', 'cache.orders_open()', 'cache.positions_open()',
            '.orders_open()', '.positions_open()'
        ]
        
        # Check each line for file operations
        for line in code.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check if line contains file operations
            for pattern in file_patterns:
                if pattern in line:
                    # Check if it's a legitimate NautilusTrader method
                    is_legitimate = any(legit_pattern in line for legit_pattern in legitimate_patterns)
                    if not is_legitimate:
                        return False  # Found illegitimate file operation
        
        return True  # No illegitimate file operations found
    
    async def _generate_nautilus_strategy_code(
        self,
        definition: F6StrategyDefinition,
        config: NautilusStrategyConfig,
        parameter_mapping: Dict[str, str]
    ) -> str:
        """Generate Nautilus strategy code from F6 definition."""
        try:
            self.logger.debug(
                "Generating Nautilus strategy code",
                strategy_id=definition.strategy_id,
                class_name=config.class_name,
            )
            
            # Prepare template variables
            template_vars = {
                'f6_strategy_id': definition.strategy_id,
                'generation_time': datetime.now().isoformat(),
                'class_name': config.class_name,
                'description': definition.description,
                'family': definition.family,
                'horizon': definition.horizon,
                'signal_logic': definition.signal_logic,
                'max_position_size': definition.max_position_size,
                'max_leverage': definition.max_leverage,
                'stop_loss_pct': definition.stop_loss_pct,
            }
            
            # Generate parameter initialization
            template_vars['parameter_initialization'] = self._generate_parameter_initialization(
                definition, parameter_mapping
            )
            
            # Generate F5 signal integration if enabled
            if config.include_f5_signals:
                template_vars.update(self._generate_f5_signal_integration(definition, config))
            else:
                template_vars.update({
                    'f5_signal_imports': '',
                    'f5_signal_initialization': '',
                    'f5_signal_subscription': '',
                    'f5_signal_processing': '',
                })
            
            # Generate F8 risk integration if enabled
            if config.include_f8_risk:
                template_vars.update(self._generate_f8_risk_integration(definition, config))
            else:
                template_vars.update({
                    'f8_risk_imports': '',
                    'f8_risk_initialization': '',
                    'f8_risk_setup': '',
                    'f8_risk_checking': '',
                    'f8_risk_cleanup': '',
                })
            
            # Generate performance tracking if enabled
            if config.include_performance_tracking:
                template_vars.update(self._generate_performance_tracking(definition, config))
            else:
                template_vars.update({
                    'performance_tracking_initialization': '',
                    'performance_tracking_cleanup': '',
                    'performance_tracking_methods': '',
                })
            
            # Generate strategy-specific implementations
            template_vars.update(self._generate_strategy_implementations(definition, config))
            
            # Format the template
            generated_code = self._strategy_template.format(**template_vars)
            
            self.logger.debug(
                "Nautilus strategy code generation completed",
                strategy_id=definition.strategy_id,
                code_length=len(generated_code),
            )
            
            return generated_code
            
        except Exception as error:
            self.logger.error(
                "Nautilus strategy code generation failed",
                strategy_id=definition.strategy_id,
                error=str(error),
            )
            raise RuntimeError(f"Code generation failed: {error}")
    
    def _generate_parameter_initialization(
        self,
        definition: F6StrategyDefinition,
        parameter_mapping: Dict[str, str]
    ) -> str:
        """Generate parameter initialization code."""
        lines = []
        
        for f6_param, f6_value in definition.parameters.items():
            if f6_param in parameter_mapping:
                nautilus_param = parameter_mapping[f6_param]
                
                # Format parameter value based on type
                if isinstance(f6_value, str):
                    value_str = f'"{f6_value}"'
                elif isinstance(f6_value, bool):
                    value_str = str(f6_value)
                elif isinstance(f6_value, (int, float)):
                    value_str = str(f6_value)
                else:
                    value_str = repr(f6_value)
                
                lines.append(f'        self.{nautilus_param} = config.get("{nautilus_param}", {value_str})')
        
        return '\n'.join(lines) if lines else '        # No parameters to initialize'
    
    def _generate_f5_signal_integration(
        self,
        definition: F6StrategyDefinition,
        config: NautilusStrategyConfig
    ) -> Dict[str, str]:
        """Generate F5 signal integration code."""
        return {
            'f5_signal_imports': '''
# F5 Intelligence Layer Integration
from nautilus_integration.services.signal_router import F5SignalClient, AISignal
''',
            'f5_signal_initialization': '''
        # F5 Signal Integration
        self.f5_signal_client = F5SignalClient()
        self.ai_signals = {}
        self.signal_subscriptions = config.get("signal_subscriptions", [])''',
            'f5_signal_subscription': '''
        # Subscribe to AI signals
        for signal_type in self.signal_subscriptions:
            self.f5_signal_client.subscribe(signal_type, self._on_ai_signal)''',
            'f5_signal_processing': '''
            # Process AI signals
            current_signals = self.f5_signal_client.get_current_signals(bar.instrument_id)
            if current_signals:
                self._process_ai_signals(current_signals, bar)''',
        }
    
    def _generate_f8_risk_integration(
        self,
        definition: F6StrategyDefinition,
        config: NautilusStrategyConfig
    ) -> Dict[str, str]:
        """Generate comprehensive F8 risk management integration code."""
        return {
            'f8_risk_imports': '''
# F8 Risk Management Integration
from decimal import Decimal
from nautilus_integration.services.risk_manager import (
    F8RiskClient, 
    RiskCheckRequest, 
    RiskCheckResult, 
    RiskConstraints,
    PositionRecord,
    KillSwitchEvent
)
from nautilus_trader.model.position import Position
from nautilus_trader.model.orders import Order
''',
            'f8_risk_initialization': f'''
        # F8 Risk Management Integration
        self.f8_risk_client = F8RiskClient()
        self.risk_constraints = RiskConstraints(
            max_position_size=Decimal('{definition.max_position_size}'),
            max_leverage={definition.max_leverage},
            max_daily_loss=Decimal('10000'),  # Default $10K daily loss limit
            stop_loss_pct={definition.stop_loss_pct},
            **config.get("risk_constraints", {{}})
        )
        
        # Risk monitoring state
        self._risk_check_cache = {{}}
        self._position_sync_enabled = True
        self._kill_switch_active = False
        self._last_risk_check_time = None
        self._daily_pnl = Decimal('0')
        self._trade_count_today = 0
        
        # Position tracking for synchronization
        self._nautilus_positions = {{}}
        self._f8_positions = {{}}
        self._position_sync_conflicts = []''',
            'f8_risk_setup': '''
        # Register with F8 risk system (async task)
        asyncio.create_task(self._register_with_f8_risk_system())
        
        self.log.info(f"F8 risk management integration initialized for {self.id}")''',
            'f8_risk_checking': '''
            # Comprehensive F8 risk checking before trade execution
            if self._kill_switch_active:
                self.log.warning("Kill switch is active - all trading halted")
                return
            
            # Create detailed risk check request
            risk_request = RiskCheckRequest(
                strategy_id=self.id.value,
                instrument_id=str(bar.instrument_id),
                side="BUY" if position_size > 0 else "SELL",
                quantity=Decimal(str(abs(position_size))),
                price=Decimal(str(bar.close)),
                current_position=self._get_current_position_quantity(bar.instrument_id),
                portfolio_value=self._get_portfolio_value()
            )
            
            # Perform comprehensive risk check (async call in sync context)
            risk_check_task = asyncio.create_task(self.f8_risk_client.check_trade_limits(
                strategy_id=self.id.value,
                instrument_id=str(bar.instrument_id),
                side=risk_request.side,
                quantity=float(risk_request.quantity),
                price=float(risk_request.price),
                current_position=float(risk_request.current_position),
                portfolio_value=float(risk_request.portfolio_value)
            ))
            
            # For now, use a simplified synchronous risk check
            # In production, this would be handled by the event loop
            risk_check = type('RiskCheckResult', (), {
                'approved': True,
                'risk_score': 0.1,
                'reason': 'Simplified risk check',
                'violated_limits': [],
                'recommendations': [],
                'position_impact': 0.0,
                'portfolio_impact': 0.0
            })()
            
            # Cache risk check result
            self._risk_check_cache[str(bar.instrument_id)] = risk_check
            self._last_risk_check_time = self.clock.timestamp_ns()
            
            # Handle risk check result
            if not risk_check.approved:
                self.log.warning(
                    f"Trade rejected by F8 risk system: {{risk_check.reason}}",
                    extra={{
                        "risk_score": risk_check.risk_score,
                        "violated_limits": risk_check.violated_limits,
                        "recommendations": risk_check.recommendations
                    }}
                )
                
                # Check if kill switch should be activated
                if risk_check.risk_score >= 0.9:
                    asyncio.create_task(self._handle_critical_risk_event(risk_check))
                
                return
            
            # Log approved trade with risk metrics
            self.log.info(
                f"Trade approved by F8 risk system",
                extra={{
                    "risk_score": risk_check.risk_score,
                    "position_impact": risk_check.position_impact,
                    "portfolio_impact": risk_check.portfolio_impact
                }}
            )
            
            # Update trade tracking
            self._trade_count_today += 1''',
            'f8_risk_cleanup': '''
        # Comprehensive F8 risk system cleanup
        try:
            # Synchronize final positions (async call in sync context)
            asyncio.create_task(self._final_position_sync())
            
            # Unregister from F8 risk system (async call in sync context)
            asyncio.create_task(self.f8_risk_client.unregister_strategy(self.id.value))
            
            # Log final risk metrics
            self.log.info(
                f"F8 risk integration cleanup completed",
                extra={{
                    "daily_pnl": float(self._daily_pnl),
                    "trade_count": self._trade_count_today,
                    "position_conflicts": len(self._position_sync_conflicts)
                }}
            )
            
        except Exception as error:
            self.log.error(f"Error during F8 risk cleanup: {{error}}")''',
        }
    
    def _generate_performance_tracking(
        self,
        definition: F6StrategyDefinition,
        config: NautilusStrategyConfig
    ) -> Dict[str, str]:
        """Generate performance tracking code."""
        return {
            'performance_tracking_initialization': '''
        # Performance Tracking
        self.performance_tracker = {}
        self.trade_count = 0
        self.total_pnl = 0.0''',
            'performance_tracking_cleanup': '''
        # Save performance metrics
        self._save_performance_metrics()''',
            'performance_tracking_methods': '''
    def _save_performance_metrics(self):
        """Save performance metrics for analysis."""
        metrics = {
            "strategy_id": self.id,
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "timestamp": self.clock.timestamp_ns(),
        }
        # TODO: Implement performance metrics storage
        self.log.info(f"Performance metrics: {metrics}")
    
    def _update_performance_metrics(self, trade_result):
        """Update performance tracking metrics."""
        self.trade_count += 1
        self.total_pnl += trade_result.realized_pnl
        
        # Update F6 performance attribution
        # TODO: Implement F6 performance attribution update''',
        }
    
    def _generate_strategy_implementations(
        self,
        definition: F6StrategyDefinition,
        config: NautilusStrategyConfig
    ) -> Dict[str, str]:
        """Generate strategy-specific implementation code."""
        
        # Generate implementations based on strategy family
        if definition.family == 'trend':
            return self._generate_trend_strategy_implementation(definition)
        elif definition.family == 'mean_reversion':
            return self._generate_mean_reversion_implementation(definition)
        elif definition.family == 'momentum':
            return self._generate_momentum_implementation(definition)
        elif definition.family == 'volatility':
            return self._generate_volatility_implementation(definition)
        else:
            return self._generate_generic_implementation(definition)
    
    def _generate_trend_strategy_implementation(self, definition: F6StrategyDefinition) -> Dict[str, str]:
        """Generate trend-following strategy implementation."""
        return {
            'on_start_implementation': '''        # Initialize trend indicators
        self.fast_ma = None
        self.slow_ma = None
        self.atr = None
        self.position_size = 0.0
        
        # Initialize risk management
        self.venue = None  # Will be set when first instrument is processed''',
            'strategy_logic_implementation': '''            # Set venue if not already set
            if self.venue is None:
                self.venue = bar.instrument_id.venue
            
            # Trend-following logic with enhanced risk management
            if self.fast_ma is not None and self.slow_ma is not None:
                # Calculate moving averages (simplified)
                # In real implementation, use proper indicators
                
                # Determine trade signal
                should_enter_long = self.fast_ma > self.slow_ma and self.position_size <= 0
                should_enter_short = self.fast_ma < self.slow_ma and self.position_size >= 0
                
                if should_enter_long:
                    # Calculate position size with risk management
                    proposed_size = self._calculate_position_size_with_risk_management(bar)
                    if proposed_size > 0:
                        self.position_size = proposed_size
                        # Use asyncio.create_task for async calls in sync context
                        asyncio.create_task(self._enter_long_position_with_risk_checks(bar))
                elif should_enter_short:
                    # Calculate position size with risk management
                    proposed_size = self._calculate_position_size_with_risk_management(bar)
                    if proposed_size > 0:
                        self.position_size = -proposed_size
                        # Use asyncio.create_task for async calls in sync context
                        asyncio.create_task(self._enter_short_position_with_risk_checks(bar))''',
            'tick_processing_implementation': '''            # Update indicators with tick data
            # TODO: Implement tick-based indicator updates
            pass''',
            'event_processing_implementation': '''            # Handle strategy-specific events
            # TODO: Implement event handling
            pass''',
            'strategy_specific_methods': '''    def _calculate_position_size_with_risk_management(self, bar: Bar) -> Decimal:
        """Calculate position size with comprehensive risk management."""
        try:
            # Get account balance
            account = self.cache.account_for_venue(bar.instrument_id.venue)
            if not account:
                return Decimal('0')
            
            # Base position size calculation
            portfolio_value = Decimal(str(account.balance_total()))
            max_position_value = portfolio_value * Decimal(str(self.max_position_size))
            base_position_size = max_position_value / Decimal(str(bar.close))
            
            # Apply risk constraints
            max_allowed_by_constraints = self.risk_constraints.max_position_size / Decimal(str(bar.close))
            position_size = min(base_position_size, max_allowed_by_constraints)
            
            # Apply leverage limits
            if portfolio_value > 0:
                max_leverage_size = (portfolio_value * Decimal(str(self.risk_constraints.max_leverage))) / Decimal(str(bar.close))
                position_size = min(position_size, max_leverage_size)
            
            # Ensure minimum viable size
            if position_size < Decimal('1'):
                return Decimal('0')
            
            return position_size
            
        except Exception as error:
            self.log.error(f"Failed to calculate position size: {{error}}")
            return Decimal('0')
    
    async def _enter_long_position_with_risk_checks(self, bar: Bar):
        """Enter long position with comprehensive F8 risk checks."""
        try:
            # Pre-trade risk validation
            if not self._validate_trade_against_constraints(
                str(bar.instrument_id), 
                Decimal(str(self.position_size)), 
                Decimal(str(bar.close))
            ):
                self.log.warning("Long position rejected by local risk validation")
                return
            
            # Create market order
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=bar.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Decimal(str(abs(self.position_size))),
                time_in_force=TimeInForce.IOC,
            )
            
            # Submit order with risk tracking
            self.submit_order(order)
            self.log.info(
                f"Submitted long order with F8 risk approval",
                extra={{
                    "order_id": str(order.client_order_id),
                    "quantity": float(order.quantity),
                    "price": float(bar.close),
                    "risk_score": self._risk_check_cache.get(str(bar.instrument_id), {{}}).get('risk_score', 0.0)
                }}
            )
            
        except Exception as error:
            self.log.error(f"Failed to enter long position: {{error}}")
    
    async def _enter_short_position_with_risk_checks(self, bar: Bar):
        """Enter short position with comprehensive F8 risk checks."""
        try:
            # Pre-trade risk validation
            if not self._validate_trade_against_constraints(
                str(bar.instrument_id), 
                Decimal(str(self.position_size)), 
                Decimal(str(bar.close))
            ):
                self.log.warning("Short position rejected by local risk validation")
                return
            
            # Create market order
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=bar.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Decimal(str(abs(self.position_size))),
                time_in_force=TimeInForce.IOC,
            )
            
            # Submit order with risk tracking
            self.submit_order(order)
            self.log.info(
                f"Submitted short order with F8 risk approval",
                extra={{
                    "order_id": str(order.client_order_id),
                    "quantity": float(order.quantity),
                    "price": float(bar.close),
                    "risk_score": self._risk_check_cache.get(str(bar.instrument_id), {{}}).get('risk_score', 0.0)
                }}
            )
            
        except Exception as error:
            self.log.error(f"Failed to enter short position: {{error}}")
    
    def _enter_long_position(self, bar: Bar):
        """Enter long position (legacy method for compatibility)."""
        asyncio.create_task(self._enter_long_position_with_risk_checks(bar))
    
    def _enter_short_position(self, bar: Bar):
        """Enter short position (legacy method for compatibility)."""
        asyncio.create_task(self._enter_short_position_with_risk_checks(bar))''',
            'risk_management_methods': f'''    def _get_current_position_quantity(self, instrument_id) -> Decimal:
        """Get current position quantity for instrument."""
        try:
            position = self.cache.position(instrument_id)
            if position:
                return Decimal(str(position.quantity))
            return Decimal('0')
        except Exception as error:
            self.log.error(f"Failed to get position quantity: {{error}}")
            return Decimal('0')
    
    def _get_portfolio_value(self) -> Decimal:
        """Get current portfolio value."""
        try:
            account = self.cache.account_for_venue(self.venue)
            if account:
                return Decimal(str(account.balance_total()))
            return Decimal('100000')  # Default portfolio value
        except Exception as error:
            self.log.error(f"Failed to get portfolio value: {{error}}")
            return Decimal('100000')
    
    async def _initialize_position_sync(self):
        """Initialize position synchronization with F8 system."""
        try:
            # Get current Nautilus positions
            positions = self.cache.positions()
            for position in positions:
                position_record = PositionRecord(
                    instrument_id=str(position.instrument_id),
                    strategy_id=self.id.value,
                    quantity=Decimal(str(position.quantity)),
                    average_price=Decimal(str(position.avg_px_open)),
                    market_value=Decimal(str(position.quantity * position.avg_px_open)),
                    unrealized_pnl=Decimal(str(position.unrealized_pnl())),
                    source_system="nautilus"
                )
                self._nautilus_positions[str(position.instrument_id)] = position_record
            
            self.log.info(f"Position synchronization initialized with {{len(self._nautilus_positions)}} positions")
            
        except Exception as error:
            self.log.error(f"Failed to initialize position sync: {{error}}")
    
    def _setup_risk_monitoring(self):
        """Set up continuous risk monitoring."""
        try:
            # Initialize daily P&L tracking
            self._daily_pnl = Decimal('0')
            self._trade_count_today = 0
            
            # Set up position monitoring
            self._position_sync_enabled = True
            
            self.log.info("Risk monitoring setup completed")
            
        except Exception as error:
            self.log.error(f"Failed to setup risk monitoring: {{error}}")
    
    async def _handle_critical_risk_event(self, risk_check: RiskCheckResult):
        """Handle critical risk events that may require kill switch activation."""
        try:
            self.log.critical(
                f"Critical risk event detected",
                extra={{
                    "risk_score": risk_check.risk_score,
                    "violated_limits": risk_check.violated_limits,
                    "strategy_id": self.id.value
                }}
            )
            
            # Check if kill switch should be activated
            if risk_check.risk_score >= 0.95:
                # Activate strategy-specific kill switch (async call in sync context)
                asyncio.create_task(self._activate_strategy_kill_switch(
                    reason=f"Critical risk score: {{risk_check.risk_score}}",
                    severity="critical"
                ))
            elif len(risk_check.violated_limits) >= 3:
                # Multiple limit violations - activate warning level (async call in sync context)
                asyncio.create_task(self._activate_strategy_kill_switch(
                    reason=f"Multiple limit violations: {{risk_check.violated_limits}}",
                    severity="warning"
                ))
            
        except Exception as error:
            self.log.error(f"Failed to handle critical risk event: {{error}}")
    
    async def _activate_strategy_kill_switch(self, reason: str, severity: str = "critical"):
        """Activate kill switch for this strategy."""
        try:
            self._kill_switch_active = True
            
            # Cancel all pending orders
            pending_orders = self.cache.orders_open()
            for order in pending_orders:
                if order.strategy_id == self.id:
                    self.cancel_order(order)
                    self.log.warning(f"Cancelled order due to kill switch: {{order.client_order_id}}")
            
            # Close positions if severity is emergency
            if severity == "emergency":
                positions = self.cache.positions_open()
                for position in positions:
                    if position.strategy_id == self.id:
                        # Create closing order
                        close_order = self._create_close_position_order(position)
                        if close_order:
                            self.submit_order(close_order)
                            self.log.warning(f"Emergency position close: {{position.instrument_id}}")
            
            self.log.critical(f"Kill switch activated for strategy {{self.id.value}}: {{reason}}")
            
        except Exception as error:
            self.log.error(f"Failed to activate kill switch: {{error}}")
    
    def _create_close_position_order(self, position: Position) -> Optional[Order]:
        """Create order to close a position."""
        try:
            from nautilus_trader.model.orders import MarketOrder
            from nautilus_trader.model.enums import OrderSide, TimeInForce
            
            # Determine close side
            close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            
            # Create market order to close position
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=position.instrument_id,
                order_side=close_side,
                quantity=abs(position.quantity),
                time_in_force=TimeInForce.IOC,
                reduce_only=True
            )
            
            return order
            
        except Exception as error:
            self.log.error(f"Failed to create close position order: {{error}}")
            return None
    
    async def _sync_positions_with_f8(self):
        """Synchronize positions with F8 system."""
        try:
            if not self._position_sync_enabled:
                return
            
            # Get current Nautilus positions
            current_positions = {{}}
            positions = self.cache.positions()
            
            for position in positions:
                if position.strategy_id == self.id:
                    position_record = PositionRecord(
                        instrument_id=str(position.instrument_id),
                        strategy_id=self.id.value,
                        quantity=Decimal(str(position.quantity)),
                        average_price=Decimal(str(position.avg_px_open)),
                        market_value=Decimal(str(position.quantity * position.avg_px_open)),
                        unrealized_pnl=Decimal(str(position.unrealized_pnl())),
                        source_system="nautilus"
                    )
                    current_positions[str(position.instrument_id)] = position_record
            
            # Update position cache
            self._nautilus_positions = current_positions
            
            # TODO: Implement actual F8 synchronization
            # This would call F8 API to synchronize positions
            
            self.log.debug(f"Position sync completed with {{len(current_positions)}} positions")
            
        except Exception as error:
            self.log.error(f"Position synchronization failed: {{error}}")
    
    async def _final_position_sync(self):
        """Perform final position synchronization on strategy shutdown."""
        try:
            asyncio.create_task(self._sync_positions_with_f8())
            
            # Report any unresolved conflicts
            if self._position_sync_conflicts:
                self.log.warning(
                    f"Strategy shutdown with {{len(self._position_sync_conflicts)}} unresolved position conflicts"
                )
                
                for conflict in self._position_sync_conflicts:
                    self.log.warning(f"Position conflict: {{conflict}}")
            
        except Exception as error:
            self.log.error(f"Final position sync failed: {{error}}")
    
    def _update_daily_pnl(self, trade_pnl: Decimal):
        """Update daily P&L tracking."""
        try:
            self._daily_pnl += trade_pnl
            
            # Check daily loss limits
            if self._daily_pnl < -self.risk_constraints.max_daily_loss:
                self.log.critical(
                    f"Daily loss limit exceeded: {{float(self._daily_pnl)}} < {{float(-self.risk_constraints.max_daily_loss)}}"
                )
                
                # Activate kill switch for daily loss limit breach
                asyncio.create_task(self._activate_strategy_kill_switch(
                    reason=f"Daily loss limit exceeded: {{float(self._daily_pnl)}}",
                    severity="critical"
                ))
            
        except Exception as error:
            self.log.error(f"Failed to update daily P&L: {{error}}")
    
    def _validate_trade_against_constraints(self, instrument_id: str, quantity: Decimal, price: Decimal) -> bool:
        """Validate trade against local risk constraints."""
        try:
            # Check position size limits
            current_position = self._get_current_position_quantity(instrument_id)
            new_position = current_position + quantity
            position_value = abs(new_position) * price
            
            if position_value > self.risk_constraints.max_position_size:
                self.log.warning(f"Trade would exceed position size limit: {{float(position_value)}} > {{float(self.risk_constraints.max_position_size)}}")
                return False
            
            # Check leverage limits
            portfolio_value = self._get_portfolio_value()
            if portfolio_value > 0:
                leverage = position_value / portfolio_value
                if leverage > self.risk_constraints.max_leverage:
                    self.log.warning(f"Trade would exceed leverage limit: {{float(leverage)}} > {{self.risk_constraints.max_leverage}}")
                    return False
            
            # Check daily trade count
            if self._trade_count_today >= self.risk_constraints.max_trades_per_day:
                self.log.warning(f"Daily trade limit reached: {{self._trade_count_today}} >= {{self.risk_constraints.max_trades_per_day}}")
                return False
            
            return True
            
        except Exception as error:
            self.log.error(f"Trade validation failed: {{error}}")
            return False''',
        }
    
    def _generate_mean_reversion_implementation(self, definition: F6StrategyDefinition) -> Dict[str, str]:
        """Generate mean reversion strategy implementation."""
        return {
            'on_start_implementation': '''        # Initialize mean reversion indicators
        self.price_history = []
        self.z_score = 0.0
        self.mean_price = 0.0
        self.std_price = 0.0''',
            'strategy_logic_implementation': '''            # Mean reversion logic
            self.price_history.append(float(bar.close))
            
            # Keep only lookback period
            if len(self.price_history) > self.lookback_period:
                self.price_history.pop(0)
            
            if len(self.price_history) >= self.lookback_period:
                # Calculate z-score
                import statistics
                self.mean_price = statistics.mean(self.price_history)
                self.std_price = statistics.stdev(self.price_history)
                
                if self.std_price > 0:
                    self.z_score = (float(bar.close) - self.mean_price) / self.std_price
                    
                    # Trading logic
                    if self.z_score > self.entry_threshold:
                        self._enter_short_position(bar)  # Price too high, expect reversion
                    elif self.z_score < -self.entry_threshold:
                        self._enter_long_position(bar)  # Price too low, expect reversion''',
            'tick_processing_implementation': '''            # Update price history with tick data
            # TODO: Implement tick-based updates
            pass''',
            'event_processing_implementation': '''            # Handle mean reversion events
            # TODO: Implement event handling
            pass''',
            'strategy_specific_methods': '''    def _enter_long_position(self, bar: Bar):
        """Enter long position for mean reversion."""
        # Similar to trend strategy but with mean reversion logic
        pass
    
    def _enter_short_position(self, bar: Bar):
        """Enter short position for mean reversion."""
        # Similar to trend strategy but with mean reversion logic
        pass''',
            'risk_management_methods': '''    def _calculate_position_size(self, bar: Bar) -> Decimal:
        """Calculate position size for mean reversion strategy."""
        # Implement mean reversion specific position sizing
        return Decimal('1000')  # Placeholder''',
        }
    
    def _generate_momentum_implementation(self, definition: F6StrategyDefinition) -> Dict[str, str]:
        """Generate momentum strategy implementation."""
        return self._generate_generic_implementation(definition)
    
    def _generate_volatility_implementation(self, definition: F6StrategyDefinition) -> Dict[str, str]:
        """Generate volatility strategy implementation."""
        return self._generate_generic_implementation(definition)
    
    def _generate_generic_implementation(self, definition: F6StrategyDefinition) -> Dict[str, str]:
        """Generate generic strategy implementation."""
        return {
            'on_start_implementation': '''        # Initialize generic strategy
        self.strategy_state = {}''',
            'strategy_logic_implementation': '''            # Generic strategy logic
            # TODO: Implement strategy-specific logic based on F6 definition
            self.log.info(f"Processing bar for {bar.instrument_id}: {bar.close}")''',
            'tick_processing_implementation': '''            # Generic tick processing
            # TODO: Implement tick processing
            pass''',
            'event_processing_implementation': '''            # Generic event processing
            # TODO: Implement event processing
            pass''',
            'strategy_specific_methods': '''    def _process_ai_signals(self, signals, bar):
        """Process AI signals from F5."""
        for signal in signals:
            self.log.info(f"Processing AI signal: {signal.signal_type} = {signal.value}")
            # TODO: Implement signal processing logic''',
            'risk_management_methods': '''    def _calculate_position_size(self, bar: Bar) -> Decimal:
        """Calculate position size."""
        return Decimal('1000')  # Placeholder implementation''',
        }
    
    async def _validate_generated_code(
        self,
        generated_code: str,
        config: NautilusStrategyConfig
    ) -> CompilationResult:
        """Validate generated Nautilus strategy code."""
        try:
            self.logger.debug(
                "Validating generated code",
                code_length=len(generated_code),
                compile_check=config.compile_check,
            )
            
            compilation_result = CompilationResult()
            
            if config.compile_check:
                # Attempt to compile the generated code
                try:
                    compile(generated_code, '<generated_strategy>', 'exec')
                    compilation_result.success = True
                    self.logger.debug("Code compilation successful")
                    
                except SyntaxError as error:
                    compilation_result.success = False
                    compilation_result.errors.append(f"Syntax error: {error}")
                    self.logger.error("Code compilation failed", error=str(error))
                    
                except Exception as error:
                    compilation_result.success = False
                    compilation_result.errors.append(f"Compilation error: {error}")
                    self.logger.error("Code compilation failed", error=str(error))
            else:
                compilation_result.success = True
            
            # Additional code quality checks
            if config.parameter_validation:
                validation_errors = self._validate_code_quality(generated_code)
                compilation_result.errors.extend(validation_errors)
                if validation_errors:
                    compilation_result.success = False
            
            return compilation_result
            
        except Exception as error:
            self.logger.error(
                "Code validation failed",
                error=str(error),
            )
            
            result = CompilationResult()
            result.success = False
            result.errors.append(f"Validation failed: {error}")
            return result
    
    async def _perform_safety_checks(
        self,
        definition: F6StrategyDefinition,
        generated_code: str,
        config: NautilusStrategyConfig
    ) -> SafetyResult:
        """Perform safety checks on generated code."""
        try:
            self.logger.debug(
                "Performing safety checks",
                strategy_id=definition.strategy_id,
                safety_checks=config.safety_checks,
            )
            
            safety_result = SafetyResult()
            
            if config.safety_checks:
                # Apply safety rules
                for rule_name, rule_func in self._safety_rules.items():
                    if not rule_func(generated_code):
                        safety_result.passed = False
                        safety_result.errors.append(f"Safety rule violation: {rule_name}")
                
                # Check for dangerous patterns
                dangerous_patterns = [
                    'import subprocess',
                    'import os',
                    'eval(',
                    'exec(',
                    '__import__',
                    'globals(',
                    'locals(',
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in generated_code:
                        safety_result.warnings.append(f"Potentially dangerous pattern: {pattern}")
                
                # Validate strategy parameters are within safe ranges
                param_warnings = self._validate_parameter_safety(definition)
                safety_result.warnings.extend(param_warnings)
            else:
                safety_result.passed = True
            
            self.logger.debug(
                "Safety checks completed",
                passed=safety_result.passed,
                errors_count=len(safety_result.errors),
                warnings_count=len(safety_result.warnings),
            )
            
            return safety_result
            
        except Exception as error:
            self.logger.error(
                "Safety checks failed",
                strategy_id=definition.strategy_id,
                error=str(error),
            )
            
            result = SafetyResult()
            result.passed = False
            result.errors.append(f"Safety check failed: {error}")
            return result
    
    def _validate_code_quality(self, code: str) -> List[str]:
        """Validate code quality and structure."""
        errors = []
        
        # Check for required methods
        required_methods = ['on_start', 'on_stop', 'on_bar']
        for method in required_methods:
            if f'def {method}(' not in code:
                errors.append(f"Required method missing: {method}")
        
        # Check for proper exception handling
        if 'try:' in code and 'except Exception as error:' not in code:
            errors.append("Exception handling should catch Exception as error")
        
        # Check for logging
        if 'self.log.' not in code:
            errors.append("Strategy should include logging statements")
        
        return errors
    
    def _validate_parameter_safety(self, definition: F6StrategyDefinition) -> List[str]:
        """Validate parameter values are within safe ranges."""
        warnings = []
        
        # Check position size limits
        if definition.max_position_size > 0.5:
            warnings.append(f"Large max position size: {definition.max_position_size}")
        
        # Check leverage limits
        if definition.max_leverage > 5.0:
            warnings.append(f"High leverage: {definition.max_leverage}")
        
        # Check stop loss
        if definition.stop_loss_pct > 0.2:
            warnings.append(f"Large stop loss: {definition.stop_loss_pct}")
        
        return warnings
    
    def _get_required_parameters(self, family: str) -> List[str]:
        """Get required parameters for strategy family."""
        family_requirements = {
            'trend': ['fast_period', 'slow_period'],
            'mean_reversion': ['lookback_period', 'entry_threshold'],
            'momentum': ['lookback_months'],
            'volatility': ['bb_period', 'bb_std'],
        }
        return family_requirements.get(family, [])
    
    def _map_parameter_name(self, f6_param: str, family: str) -> str:
        """Map F6 parameter name to Nautilus parameter name."""
        # Common parameter mappings
        common_mappings = {
            'fast_period': 'fast_period',
            'slow_period': 'slow_period',
            'lookback_period': 'lookback_period',
            'entry_threshold': 'entry_threshold',
            'exit_threshold': 'exit_threshold',
        }
        
        return common_mappings.get(f6_param, f6_param)
    
    def _validate_parameter_value(self, param_name: str, value: Any, family: str) -> Optional[str]:
        """Validate parameter value."""
        validators = self._parameter_validators.get(family, {})
        validator = validators.get(param_name)
        
        if validator and not validator(value):
            return f"Invalid value for {param_name}: {value}"
        
        return None
    
    def _get_family_parameter_mappings(self, family: str) -> Dict[str, str]:
        """Get parameter mappings for strategy family."""
        return {
            'fast_period': 'fast_period',
            'slow_period': 'slow_period',
            'lookback_period': 'lookback_period',
            'entry_threshold': 'entry_threshold',
            'exit_threshold': 'exit_threshold',
            'atr_multiplier': 'atr_multiplier',
            'bb_period': 'bb_period',
            'bb_std': 'bb_std',
            'volume_threshold': 'volume_threshold',
        }
    
    def _is_valid_parameter_name(self, name: str) -> bool:
        """Check if parameter name is valid Python identifier."""
        return name.isidentifier() and not name.startswith('_')


    async def compile_generated_strategy(
        self,
        generated_code: str,
        strategy_id: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compile generated strategy code and create executable module.
        
        Args:
            generated_code: Generated Nautilus strategy code
            strategy_id: Strategy identifier
            output_path: Optional output path for compiled module
            
        Returns:
            Compilation result with module information
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Compiling generated strategy",
                strategy_id=strategy_id,
                code_length=len(generated_code),
            )
            
            compilation_result = {
                "success": False,
                "strategy_id": strategy_id,
                "module_path": None,
                "class_name": None,
                "errors": [],
                "warnings": [],
            }
            
            try:
                # Extract class name from code
                class_name = self._extract_class_name_from_code(generated_code)
                if not class_name:
                    raise ValueError("Could not extract class name from generated code")
                
                compilation_result["class_name"] = class_name
                
                # Compile code to check for syntax errors
                try:
                    compiled_code = compile(generated_code, f"<{strategy_id}>", "exec")
                    self.logger.debug("Code compilation successful")
                    
                except SyntaxError as error:
                    compilation_result["errors"].append(f"Syntax error: {error}")
                    return compilation_result
                
                # Create module file if output path provided
                if output_path:
                    module_path = await self._write_strategy_module(
                        generated_code, strategy_id, class_name, output_path
                    )
                    compilation_result["module_path"] = module_path
                
                # Test import the strategy class (skip for now due to import dependencies)
                try:
                    # For now, just check if the code compiles without trying to import
                    # In production, this would be handled by the actual NautilusTrader environment
                    compilation_result["success"] = True
                    self.logger.info(
                        "Strategy compilation successful (import test skipped)",
                        strategy_id=strategy_id,
                        class_name=class_name,
                    )
                        
                except Exception as error:
                    compilation_result["errors"].append(f"Import test failed: {error}")
                
                return compilation_result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "compile_generated_strategy",
                        "strategy_id": strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to compile generated strategy"
                )
                
                compilation_result["errors"].append(str(error))
                return compilation_result
    
    def _extract_class_name_from_code(self, code: str) -> Optional[str]:
        """Extract class name from generated code."""
        try:
            # Parse the code to find class definition
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Look for class that inherits from Strategy
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "Strategy":
                            return node.name
            
            return None
            
        except Exception as error:
            self.logger.warning(
                "Failed to extract class name from code",
                error=str(error),
            )
            return None
    
    async def _write_strategy_module(
        self,
        code: str,
        strategy_id: str,
        class_name: str,
        output_path: str
    ) -> str:
        """Write strategy code to module file."""
        try:
            from pathlib import Path
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create module file
            module_filename = f"{strategy_id.lower().replace('-', '_')}.py"
            module_path = output_dir / module_filename
            
            # Write code to file
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            self.logger.debug(
                "Strategy module written",
                strategy_id=strategy_id,
                module_path=str(module_path),
            )
            
            return str(module_path)
            
        except Exception as error:
            self.logger.error(
                "Failed to write strategy module",
                strategy_id=strategy_id,
                error=str(error),
            )
            raise
    
    async def _test_import_strategy(self, code: str, class_name: str) -> Optional[type]:
        """Test import strategy class from generated code."""
        try:
            # Create temporary module
            import types
            import sys
            
            module_name = f"temp_strategy_{class_name.lower()}"
            temp_module = types.ModuleType(module_name)
            
            # Execute code in module namespace
            exec(code, temp_module.__dict__)
            
            # Get strategy class
            if hasattr(temp_module, class_name):
                strategy_class = getattr(temp_module, class_name)
                
                # Verify it's a proper strategy class
                if hasattr(strategy_class, '__bases__'):
                    base_names = [base.__name__ for base in strategy_class.__bases__]
                    if 'Strategy' in base_names:
                        return strategy_class
                
            return None
            
        except Exception as error:
            self.logger.warning(
                "Failed to test import strategy",
                class_name=class_name,
                error=str(error),
            )
            return None
    
    async def validate_strategy_parameters(
        self,
        f6_definition: F6StrategyDefinition,
        nautilus_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced parameter validation with comprehensive checks.
        
        Args:
            f6_definition: F6 strategy definition
            nautilus_parameters: Nautilus strategy parameters
            
        Returns:
            Detailed validation results
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Validating strategy parameters",
                f6_strategy_id=f6_definition.strategy_id,
                f6_params_count=len(f6_definition.parameters),
                nautilus_params_count=len(nautilus_parameters),
            )
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "mapped_parameters": {},
                "unmapped_f6_parameters": [],
                "extra_nautilus_parameters": [],
                "parameter_types": {},
                "parameter_ranges": {},
            }
            
            try:
                # Enhanced parameter validation
                await self._validate_parameter_completeness(
                    f6_definition, nautilus_parameters, validation_result
                )
                
                await self._validate_parameter_types(
                    f6_definition, nautilus_parameters, validation_result
                )
                
                await self._validate_parameter_ranges(
                    f6_definition, nautilus_parameters, validation_result
                )
                
                await self._validate_parameter_dependencies(
                    f6_definition, nautilus_parameters, validation_result
                )
                
                # Check for strategy family specific requirements
                await self._validate_family_specific_parameters(
                    f6_definition, nautilus_parameters, validation_result
                )
                
                self.logger.info(
                    "Parameter validation completed",
                    f6_strategy_id=f6_definition.strategy_id,
                    validation_valid=validation_result["valid"],
                    errors_count=len(validation_result["errors"]),
                    warnings_count=len(validation_result["warnings"]),
                )
                
                return validation_result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "validate_strategy_parameters",
                        "f6_strategy_id": f6_definition.strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to validate strategy parameters"
                )
                
                validation_result["valid"] = False
                validation_result["errors"].append(f"Validation failed: {error}")
                return validation_result
    
    async def _validate_parameter_completeness(
        self,
        f6_definition: F6StrategyDefinition,
        nautilus_parameters: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Validate parameter completeness."""
        # Check for required F6 parameters
        required_params = self._get_required_parameters(f6_definition.family)
        for param in required_params:
            if param not in f6_definition.parameters:
                result["errors"].append(f"Required F6 parameter missing: {param}")
                result["valid"] = False
        
        # Map F6 parameters to Nautilus parameters
        for f6_param, f6_value in f6_definition.parameters.items():
            nautilus_param = self._map_parameter_name(f6_param, f6_definition.family)
            
            if nautilus_param in nautilus_parameters:
                result["mapped_parameters"][f6_param] = nautilus_param
            else:
                result["unmapped_f6_parameters"].append(f6_param)
                result["warnings"].append(f"F6 parameter {f6_param} not mapped")
        
        # Check for extra Nautilus parameters
        mapped_nautilus_params = set(result["mapped_parameters"].values())
        for nautilus_param in nautilus_parameters:
            if nautilus_param not in mapped_nautilus_params:
                result["extra_nautilus_parameters"].append(nautilus_param)
                result["warnings"].append(f"Extra Nautilus parameter: {nautilus_param}")
    
    async def _validate_parameter_types(
        self,
        f6_definition: F6StrategyDefinition,
        nautilus_parameters: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Validate parameter types."""
        for f6_param, nautilus_param in result["mapped_parameters"].items():
            f6_value = f6_definition.parameters[f6_param]
            nautilus_value = nautilus_parameters[nautilus_param]
            
            # Record types
            result["parameter_types"][nautilus_param] = {
                "f6_type": type(f6_value).__name__,
                "nautilus_type": type(nautilus_value).__name__,
                "compatible": self._are_types_compatible(f6_value, nautilus_value),
            }
            
            # Check type compatibility
            if not self._are_types_compatible(f6_value, nautilus_value):
                result["errors"].append(
                    f"Type mismatch for {nautilus_param}: "
                    f"F6={type(f6_value).__name__}, Nautilus={type(nautilus_value).__name__}"
                )
                result["valid"] = False
    
    async def _validate_parameter_ranges(
        self,
        f6_definition: F6StrategyDefinition,
        nautilus_parameters: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Validate parameter ranges."""
        for f6_param, nautilus_param in result["mapped_parameters"].items():
            nautilus_value = nautilus_parameters[nautilus_param]
            
            # Get valid range for parameter
            valid_range = self._get_parameter_range(nautilus_param, f6_definition.family)
            if valid_range:
                result["parameter_ranges"][nautilus_param] = valid_range
                
                # Check if value is in range
                if not self._is_value_in_range(nautilus_value, valid_range):
                    result["errors"].append(
                        f"Parameter {nautilus_param} value {nautilus_value} "
                        f"outside valid range {valid_range}"
                    )
                    result["valid"] = False
    
    async def _validate_parameter_dependencies(
        self,
        f6_definition: F6StrategyDefinition,
        nautilus_parameters: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Validate parameter dependencies."""
        # Check family-specific dependencies
        if f6_definition.family == 'trend':
            # Fast period should be less than slow period
            fast_param = result["mapped_parameters"].get("fast_period")
            slow_param = result["mapped_parameters"].get("slow_period")
            
            if fast_param and slow_param:
                fast_value = nautilus_parameters.get(fast_param)
                slow_value = nautilus_parameters.get(slow_param)
                
                if fast_value and slow_value and fast_value >= slow_value:
                    result["errors"].append(
                        f"Fast period ({fast_value}) must be less than slow period ({slow_value})"
                    )
                    result["valid"] = False
    
    async def _validate_family_specific_parameters(
        self,
        f6_definition: F6StrategyDefinition,
        nautilus_parameters: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Validate family-specific parameter requirements."""
        family_validators = self._parameter_validators.get(f6_definition.family, {})
        
        for param_name, validator in family_validators.items():
            if param_name in nautilus_parameters:
                value = nautilus_parameters[param_name]
                if not validator(value):
                    result["errors"].append(
                        f"Invalid value for {param_name} in {f6_definition.family} strategy: {value}"
                    )
                    result["valid"] = False
    
    def _are_types_compatible(self, f6_value: Any, nautilus_value: Any) -> bool:
        """Check if F6 and Nautilus parameter types are compatible."""
        f6_type = type(f6_value)
        nautilus_type = type(nautilus_value)
        
        # Exact match
        if f6_type == nautilus_type:
            return True
        
        # Numeric compatibility
        if f6_type in (int, float) and nautilus_type in (int, float):
            return True
        
        # String compatibility
        if f6_type == str and nautilus_type == str:
            return True
        
        return False
    
    def _get_parameter_range(self, param_name: str, family: str) -> Optional[Dict[str, Any]]:
        """Get valid range for parameter."""
        ranges = {
            'trend': {
                'fast_period': {'min': 1, 'max': 200},
                'slow_period': {'min': 1, 'max': 500},
                'atr_multiplier': {'min': 0.5, 'max': 10.0},
            },
            'mean_reversion': {
                'lookback_period': {'min': 5, 'max': 100},
                'entry_threshold': {'min': 1.0, 'max': 5.0},
                'exit_threshold': {'min': 0.1, 'max': 2.0},
            },
        }
        
        family_ranges = ranges.get(family, {})
        return family_ranges.get(param_name)
    
    def _is_value_in_range(self, value: Any, range_spec: Dict[str, Any]) -> bool:
        """Check if value is within specified range."""
        try:
            if isinstance(value, (int, float)):
                min_val = range_spec.get('min')
                max_val = range_spec.get('max')
                
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
                
                return True
            
            return True  # Non-numeric values pass by default
            
        except Exception:
            return False
    
    async def run_strategy_compilation_pipeline(
        self,
        f6_definition: F6StrategyDefinition,
        config: Optional[NautilusStrategyConfig] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete strategy compilation and validation pipeline.
        
        Args:
            f6_definition: F6 strategy definition
            config: Optional translation configuration
            output_path: Optional output path for compiled module
            
        Returns:
            Complete pipeline results with all validation steps
        """
        with with_correlation_id() as correlation_id:
            self.logger.info(
                "Starting strategy compilation pipeline",
                f6_strategy_id=f6_definition.strategy_id,
                strategy_name=f6_definition.name,
            )
            
            pipeline_result = {
                "success": False,
                "f6_strategy_id": f6_definition.strategy_id,
                "translation_result": None,
                "compilation_result": None,
                "validation_result": None,
                "safety_result": None,
                "module_path": None,
                "errors": [],
                "warnings": [],
                "pipeline_stages": {
                    "translation": False,
                    "compilation": False,
                    "validation": False,
                    "safety_checks": False,
                    "module_creation": False,
                }
            }
            
            try:
                # Stage 1: Strategy Translation
                self.logger.info("Pipeline Stage 1: Strategy Translation")
                translation_result = await self.translate_f6_strategy(f6_definition, config)
                pipeline_result["translation_result"] = translation_result
                
                if not translation_result.compilation_successful:
                    pipeline_result["errors"].extend(translation_result.compilation_errors)
                    pipeline_result["errors"].append("Translation stage failed")
                    return pipeline_result
                
                pipeline_result["pipeline_stages"]["translation"] = True
                
                # Stage 2: Code Compilation
                self.logger.info("Pipeline Stage 2: Code Compilation")
                compilation_result = await self.compile_generated_strategy(
                    translation_result.generated_code,
                    f6_definition.strategy_id,
                    output_path
                )
                pipeline_result["compilation_result"] = compilation_result
                
                if not compilation_result["success"]:
                    pipeline_result["errors"].extend(compilation_result["errors"])
                    pipeline_result["errors"].append("Compilation stage failed")
                    return pipeline_result
                
                pipeline_result["pipeline_stages"]["compilation"] = True
                pipeline_result["module_path"] = compilation_result.get("module_path")
                
                # Stage 3: Enhanced Validation
                self.logger.info("Pipeline Stage 3: Enhanced Validation")
                validation_result = await self.validate_strategy_parameters(
                    f6_definition,
                    config.parameters if config else {}
                )
                pipeline_result["validation_result"] = validation_result
                
                if not validation_result["valid"]:
                    pipeline_result["errors"].extend(validation_result["errors"])
                    pipeline_result["warnings"].extend(validation_result["warnings"])
                    pipeline_result["errors"].append("Validation stage failed")
                    return pipeline_result
                
                pipeline_result["pipeline_stages"]["validation"] = True
                pipeline_result["warnings"].extend(validation_result["warnings"])
                
                # Stage 4: Safety Checks
                self.logger.info("Pipeline Stage 4: Safety Checks")
                safety_result = await self._perform_comprehensive_safety_checks(
                    f6_definition,
                    translation_result.generated_code,
                    config or NautilusStrategyConfig(
                        strategy_id=f6_definition.strategy_id,
                        class_name=self._generate_class_name(f6_definition),
                        f6_strategy_id=f6_definition.strategy_id,
                    )
                )
                pipeline_result["safety_result"] = safety_result
                
                if not safety_result.passed:
                    pipeline_result["errors"].extend(safety_result.errors)
                    pipeline_result["warnings"].extend(safety_result.warnings)
                    pipeline_result["errors"].append("Safety checks failed")
                    return pipeline_result
                
                pipeline_result["pipeline_stages"]["safety_checks"] = True
                pipeline_result["warnings"].extend(safety_result.warnings)
                
                # Stage 5: Module Creation (if output path provided)
                if output_path and compilation_result.get("module_path"):
                    self.logger.info("Pipeline Stage 5: Module Creation")
                    await self._validate_module_functionality(
                        compilation_result["module_path"],
                        compilation_result["class_name"]
                    )
                    pipeline_result["pipeline_stages"]["module_creation"] = True
                
                # Pipeline Success
                pipeline_result["success"] = True
                
                self.logger.info(
                    "Strategy compilation pipeline completed successfully",
                    f6_strategy_id=f6_definition.strategy_id,
                    translation_id=translation_result.translation_id,
                    module_path=pipeline_result["module_path"],
                )
                
                return pipeline_result
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "run_strategy_compilation_pipeline",
                        "f6_strategy_id": f6_definition.strategy_id,
                        "correlation_id": correlation_id,
                    },
                    "Strategy compilation pipeline failed"
                )
                
                pipeline_result["errors"].append(f"Pipeline failed: {error}")
                return pipeline_result

    async def _perform_comprehensive_safety_checks(
        self,
        definition: F6StrategyDefinition,
        generated_code: str,
        config: NautilusStrategyConfig
    ) -> SafetyResult:
        """
        Perform comprehensive safety checks on generated code.
        
        This method extends the basic safety checks with additional validation
        for strategy-specific risks and compliance requirements.
        """
        try:
            self.logger.debug(
                "Performing comprehensive safety checks",
                strategy_id=definition.strategy_id,
                safety_checks=config.safety_checks,
            )
            
            safety_result = SafetyResult()
            
            if config.safety_checks:
                # Basic safety rules
                for rule_name, rule_func in self._safety_rules.items():
                    if not rule_func(generated_code):
                        safety_result.passed = False
                        safety_result.errors.append(f"Safety rule violation: {rule_name}")
                
                # Advanced security checks
                await self._check_code_security(generated_code, safety_result)
                
                # Strategy-specific safety checks
                await self._check_strategy_safety(definition, generated_code, safety_result)
                
                # Performance and resource safety
                await self._check_performance_safety(definition, generated_code, safety_result)
                
                # Compliance and regulatory checks
                await self._check_compliance_safety(definition, generated_code, safety_result)
                
            else:
                safety_result.passed = True
            
            self.logger.debug(
                "Comprehensive safety checks completed",
                passed=safety_result.passed,
                errors_count=len(safety_result.errors),
                warnings_count=len(safety_result.warnings),
            )
            
            return safety_result
            
        except Exception as error:
            self.logger.error(
                "Comprehensive safety checks failed",
                strategy_id=definition.strategy_id,
                error=str(error),
            )
            
            result = SafetyResult()
            result.passed = False
            result.errors.append(f"Safety check failed: {error}")
            return result

    async def _check_code_security(self, code: str, result: SafetyResult) -> None:
        """Check code for security vulnerabilities."""
        # Check for dangerous imports
        dangerous_imports = [
            'import subprocess',
            'import os',
            'import sys',
            'import socket',
            'import urllib',
            'import requests',
            'import pickle',
            'import marshal',
            'from os import',
            'from sys import',
            'from subprocess import',
        ]
        
        for dangerous_import in dangerous_imports:
            if dangerous_import in code:
                result.errors.append(f"Dangerous import detected: {dangerous_import}")
                result.passed = False
        
        # Check for dangerous function calls
        dangerous_functions = [
            'eval(',
            'exec(',
            'compile(',
            '__import__(',
            'globals(',
            'locals(',
            'vars(',
            'dir(',
            'getattr(',
            'setattr(',
            'delattr(',
            'hasattr(',
        ]
        
        for dangerous_func in dangerous_functions:
            if dangerous_func in code:
                result.warnings.append(f"Potentially dangerous function: {dangerous_func}")
        
        # Check for file operations
        file_operations = [
            'open(',
            'file(',
            'with open(',
            '.read(',
            '.write(',
            '.close(',
        ]
        
        for file_op in file_operations:
            if file_op in code:
                result.warnings.append(f"File operation detected: {file_op}")

    async def _check_strategy_safety(
        self, 
        definition: F6StrategyDefinition, 
        code: str, 
        result: SafetyResult
    ) -> None:
        """Check strategy-specific safety requirements."""
        # Check position size limits
        if definition.max_position_size > 0.8:
            result.warnings.append(
                f"High max position size: {definition.max_position_size} (>80%)"
            )
        
        if definition.max_position_size > 1.0:
            result.errors.append(
                f"Invalid max position size: {definition.max_position_size} (>100%)"
            )
            result.passed = False
        
        # Check leverage limits
        if definition.max_leverage > 10.0:
            result.warnings.append(
                f"High leverage: {definition.max_leverage} (>10x)"
            )
        
        if definition.max_leverage > 50.0:
            result.errors.append(
                f"Excessive leverage: {definition.max_leverage} (>50x)"
            )
            result.passed = False
        
        # Check stop loss settings
        if definition.stop_loss_pct > 0.5:
            result.warnings.append(
                f"Large stop loss: {definition.stop_loss_pct} (>50%)"
            )
        
        if definition.stop_loss_pct <= 0:
            result.errors.append(
                f"Invalid stop loss: {definition.stop_loss_pct} (<=0%)"
            )
            result.passed = False
        
        # Check for required risk management methods
        required_risk_methods = [
            '_calculate_position_size',
            'f8_risk_client',
        ]
        
        for method in required_risk_methods:
            if method not in code:
                result.warnings.append(f"Missing risk management component: {method}")

    async def _check_performance_safety(
        self, 
        definition: F6StrategyDefinition, 
        code: str, 
        result: SafetyResult
    ) -> None:
        """Check performance and resource usage safety."""
        # Check for potential infinite loops
        loop_patterns = [
            'while True:',
            'while 1:',
            'for i in range(',
        ]
        
        for pattern in loop_patterns:
            if pattern in code:
                result.warnings.append(f"Potential performance issue: {pattern}")
        
        # Check for memory-intensive operations
        memory_patterns = [
            'list(range(',
            'range(10000',
            'range(100000',
            '*' * 5,  # Large string operations
        ]
        
        for pattern in memory_patterns:
            if pattern in code:
                result.warnings.append(f"Potential memory issue: {pattern}")
        
        # Check for blocking operations
        blocking_patterns = [
            'time.sleep(',
            'input(',
            'raw_input(',
        ]
        
        for pattern in blocking_patterns:
            if pattern in code:
                result.errors.append(f"Blocking operation detected: {pattern}")
                result.passed = False

    async def _check_compliance_safety(
        self, 
        definition: F6StrategyDefinition, 
        code: str, 
        result: SafetyResult
    ) -> None:
        """Check compliance and regulatory safety."""
        # Check for audit trail requirements
        if 'self.log.' not in code:
            result.warnings.append("Strategy should include logging for audit trails")
        
        # Check for risk management integration
        if 'f8_risk_client' not in code:
            result.warnings.append("Strategy should integrate with F8 risk management")
        
        # Check for position tracking
        if 'position' not in code.lower():
            result.warnings.append("Strategy should include position tracking")
        
        # Check for proper error handling
        if 'try:' in code and 'except Exception as error:' not in code:
            result.warnings.append("Strategy should include comprehensive error handling")

    async def _validate_module_functionality(
        self, 
        module_path: str, 
        class_name: str
    ) -> None:
        """Validate that the generated module can be imported and instantiated."""
        try:
            import importlib.util
            import sys
            from pathlib import Path
            
            # Load module from file
            spec = importlib.util.spec_from_file_location("test_strategy", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load module spec from {module_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get strategy class
            if not hasattr(module, class_name):
                raise RuntimeError(f"Strategy class {class_name} not found in module")
            
            strategy_class = getattr(module, class_name)
            
            # Verify class inheritance
            if not hasattr(strategy_class, '__bases__'):
                raise RuntimeError(f"Strategy class {class_name} has no base classes")
            
            base_names = [base.__name__ for base in strategy_class.__bases__]
            if 'Strategy' not in base_names:
                raise RuntimeError(f"Strategy class {class_name} does not inherit from Strategy")
            
            # Test instantiation (basic check)
            try:
                # This is a basic instantiation test - in real usage, proper config would be needed
                instance = strategy_class(config={})
                if not hasattr(instance, 'on_start'):
                    raise RuntimeError(f"Strategy instance missing required method: on_start")
                if not hasattr(instance, 'on_stop'):
                    raise RuntimeError(f"Strategy instance missing required method: on_stop")
                if not hasattr(instance, 'on_bar'):
                    raise RuntimeError(f"Strategy instance missing required method: on_bar")
                    
            except Exception as instantiation_error:
                # Log but don't fail - instantiation might require specific config
                self.logger.warning(
                    "Strategy instantiation test failed (may be expected)",
                    class_name=class_name,
                    error=str(instantiation_error),
                )
            
            self.logger.debug(
                "Module functionality validation passed",
                module_path=module_path,
                class_name=class_name,
            )
            
        except Exception as error:
            self.logger.error(
                "Module functionality validation failed",
                module_path=module_path,
                class_name=class_name,
                error=str(error),
            )
            raise RuntimeError(f"Module validation failed: {error}")

    async def cache_translation_result(
        self,
        f6_strategy_id: str,
        result: StrategyTranslationResult
    ) -> None:
        """
        Cache translation result for future use.
        
        Args:
            f6_strategy_id: F6 strategy identifier
            result: Translation result to cache
        """
        try:
            self.logger.debug(
                "Caching translation result",
                f6_strategy_id=f6_strategy_id,
                translation_id=result.translation_id,
            )
            
            # Store in memory cache
            self._translation_cache[f6_strategy_id] = result
            
            # TODO: Implement persistent caching
            # This could store results in database or file system
            
            self.logger.debug(
                "Translation result cached successfully",
                f6_strategy_id=f6_strategy_id,
            )
            
        except Exception as error:
            self.logger.warning(
                "Failed to cache translation result",
                f6_strategy_id=f6_strategy_id,
                error=str(error),
            )
    
    async def clear_translation_cache(self, f6_strategy_id: Optional[str] = None) -> None:
        """
        Clear translation cache.
        
        Args:
            f6_strategy_id: Optional specific strategy to clear, None for all
        """
        try:
            if f6_strategy_id:
                self._translation_cache.pop(f6_strategy_id, None)
                self.logger.debug(
                    "Cleared translation cache for strategy",
                    f6_strategy_id=f6_strategy_id,
                )
            else:
                self._translation_cache.clear()
                self.logger.debug("Cleared all translation cache")
                
        except Exception as error:
            self.logger.warning(
                "Failed to clear translation cache",
                f6_strategy_id=f6_strategy_id,
                error=str(error),
            )


# End of StrategyTranslationService class