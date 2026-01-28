import asyncio
from unittest.mock import Mock, patch
from nautilus_integration.services.strategy_translation import StrategyTranslationService, F6StrategyDefinition

async def test():
    mock_config = Mock()
    mock_config.environment = 'testing'
    mock_config.log_level = 'DEBUG'
    
    with patch('nautilus_integration.services.strategy_translation.get_logger') as mock_logger:
        mock_logger.return_value = Mock()
        service = StrategyTranslationService(mock_config)
    
    f6_def = F6StrategyDefinition(
        strategy_id='test_trend_strategy',
        name='Test Trend Following',
        family='trend',
        horizon='daily',
        asset_classes=['crypto'],
        description='A simple trend following strategy for testing',
        signal_logic='Buy when fast MA > slow MA, sell when fast MA < slow MA',
        entry_rules=['Fast MA crosses above slow MA'],
        exit_rules=['Fast MA crosses below slow MA'],
        risk_controls=['Max 25% position size', '5% stop loss'],
        parameters={'fast_period': 10, 'slow_period': 20, 'atr_multiplier': 2.0},
        max_position_size=0.25,
        max_leverage=2.0,
        stop_loss_pct=0.05,
        production_ready=False,
        complexity='medium',
    )
    
    result = await service.run_strategy_compilation_pipeline(f6_def)
    print(f'Pipeline success: {result["success"]}')
    print(f'Pipeline stages: {result["pipeline_stages"]}')
    if result['errors']:
        print(f'Errors: {result["errors"]}')
    if result['translation_result']:
        print(f'Translation successful: {result["translation_result"].compilation_successful}')
        print(f'Validation passed: {result["translation_result"].validation_passed}')
        if result['translation_result'].compilation_errors:
            print(f'Compilation errors: {result["translation_result"].compilation_errors}')

if __name__ == "__main__":
    asyncio.run(test())