"""
End-to-end integration tests for NautilusTrader integration.

This module tests complete workflows from F6 strategy definition to Nautilus execution,
validating multi-system interactions and data consistency.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pandas as pd
import numpy as np

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.services.integration_service import BacktestConfig
from nautilus_integration.services.integration_service import NautilusIntegrationService
from nautilus_integration.services.strategy_translation import StrategyTranslationService
from nautilus_integration.services.signal_router import SignalRouterService
from nautilus_integration.services.data_catalog_adapter import DataCatalogAdapter
from nautilus_integration.services.f8_risk_integration import F8RiskIntegration


class TestEndToEndIntegration:
    """End-to-end integration tests for the complete NautilusTrader integration."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return NautilusConfig(
            environment="testing",  # Use valid environment value
            log_level="INFO",
            data_path=tempfile.mkdtemp(),
        )
    
    @pytest.fixture
    async def integration_services(self, config):
        """Create all integration services for testing."""
        services = {
            "integration": NautilusIntegrationService(config),
            "strategy_translation": StrategyTranslationService(config),
            "signal_router": SignalRouterService(config),
            "data_catalog": DataCatalogAdapter(config),
            "f8_risk": F8RiskIntegration(config),
        }
        
        # Initialize all services
        for service in services.values():
            await service.initialize()
        
        yield services
        
        # Cleanup all services
        for service in services.values():
            await service.shutdown()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='1H'
        )
        
        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible tests
        base_price = 1.1000
        
        data = []
        current_price = base_price
        
        for i, timestamp in enumerate(dates):
            # Generate price movement
            change = np.random.normal(0, 0.001)
            current_price *= (1 + change)
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.0005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.0005)))
            open_price = current_price + np.random.normal(0, 0.0002)
            close_price = current_price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'instrument_id': 'EUR/USD',
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close_price, 5),
                'volume': round(volume, 2),
                'bar_type': '1-HOUR-MID',
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_f6_strategy(self):
        """Create sample F6 strategy configuration."""
        return {
            "strategy_id": "trend_following_eur_usd",
            "strategy_name": "EUR/USD Trend Following",
            "family": "TREND_FOLLOWING",
            "version": "1.0.0",
            "parameters": {
                "fast_period": 10,
                "slow_period": 20,
                "signal_threshold": 0.001,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "position_size": 10000,
            },
            "instruments": ["EUR/USD"],
            "risk_constraints": {
                "max_position_size": 50000.0,
                "max_daily_loss": -5000.0,
                "max_leverage": 3.0,
            },
            "ai_signal_subscriptions": ["regime_prediction", "volatility_forecast"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
    
    @pytest.fixture
    def sample_ai_signals(self):
        """Create sample AI signals for testing."""
        return [
            {
                "signal_id": "regime_001",
                "signal_type": "REGIME_PREDICTION",
                "instrument_id": "EUR/USD",
                "timestamp": datetime.now() - timedelta(minutes=5),
                "confidence": 0.85,
                "value": "LOW_VOL_TRENDING",
                "source_model": "regime_classifier_v2",
                "metadata": {"model_version": "2.1", "training_date": "2023-12-01"},
            },
            {
                "signal_id": "vol_001",
                "signal_type": "VOLATILITY_FORECAST",
                "instrument_id": "EUR/USD",
                "timestamp": datetime.now() - timedelta(minutes=3),
                "confidence": 0.78,
                "value": 0.012,
                "source_model": "volatility_garch_v1",
                "metadata": {"forecast_horizon": "1H", "confidence_interval": "95%"},
            },
        ]
    
    @pytest.mark.integration
    async def test_complete_workflow_f6_to_nautilus_execution(
        self, 
        integration_services, 
        sample_f6_strategy, 
        sample_market_data, 
        sample_ai_signals
    ):
        """
        Test complete workflow from F6 strategy definition to Nautilus execution.
        
        This test validates the entire integration pipeline:
        1. F6 strategy translation to Nautilus
        2. AI signal routing and integration
        3. Market data preparation and loading
        4. Backtest execution with NautilusTrader
        5. Results analysis and validation
        """
        services = integration_services
        
        # Step 1: Translate F6 strategy to Nautilus
        translation_result = await services["strategy_translation"].translate_strategy(
            sample_f6_strategy
        )
        
        assert translation_result["success"] is True
        assert "generated_code" in translation_result
        
        nautilus_strategy = translation_result["generated_code"]
        
        # Step 2: Set up AI signal routing
        for signal in sample_ai_signals:
            await services["signal_router"].route_signal(signal)
        
        # Verify signals were buffered for backtesting
        buffered_signals = await services["signal_router"].get_buffered_signals(
            instrument_id="EUR/USD",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert len(buffered_signals) == len(sample_ai_signals)
        
        # Step 3: Prepare market data
        data_migration_result = await services["data_catalog"].prepare_market_data_for_nautilus(
            market_data=sample_market_data,
            instruments=["EUR/USD"],
            bar_types=["1-HOUR-MID"]
        )
        
        assert data_migration_result["success"] is True
        assert "parquet_files" in data_migration_result
        
        # Step 4: Execute backtest with NautilusTrader
        backtest_config = BacktestConfig(
            backtest_id="e2e_test_backtest",
            strategy_ids=[sample_f6_strategy["strategy_id"]],
            start_date=sample_market_data['timestamp'].min(),
            end_date=sample_market_data['timestamp'].max(),
            initial_capital=100000.0,
            data_sources=["test_data"],
            instruments=["EUR/USD"],
            bar_types=["1-HOUR-MID"],
        )
        
        # Mock NautilusTrader execution
        with patch('nautilus_integration.services.integration_service.BacktestEngine') as mock_engine:
            mock_engine_instance = MagicMock()
            mock_engine_instance.run.return_value = None
            mock_engine_instance.get_result.return_value = {
                "backtest_id": "e2e_test_backtest",
                "status": "completed",
                "total_return": 0.156,
                "sharpe_ratio": 1.42,
                "max_drawdown": -0.087,
                "total_trades": 23,
                "winning_trades": 15,
                "execution_log": [
                    {
                        "timestamp_ns": int(datetime.now().timestamp() * 1e9),
                        "event_type": "ORDER_FILLED",
                        "order_id": "order_001",
                        "instrument_id": "EUR/USD",
                        "side": "BUY",
                        "quantity": 10000,
                        "price": 1.1025,
                        "ai_signal_influence": 0.65,
                    }
                ]
            }
            mock_engine.return_value = mock_engine_instance
            
            backtest_result = await services["integration"].create_backtest(
                config=backtest_config,
                strategies=[nautilus_strategy],
                market_data=sample_market_data,
                ai_signals=buffered_signals
            )
        
        # Step 5: Validate results
        assert backtest_result["status"] == "completed"
        assert backtest_result["engine_type"] == "NautilusTrader"
        assert "results" in backtest_result
        
        results = backtest_result["results"]
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "total_trades" in results
        
        # Verify AI signal integration
        execution_log = results.get("execution_log", [])
        ai_influenced_trades = [
            trade for trade in execution_log 
            if "ai_signal_influence" in trade and trade["ai_signal_influence"] > 0
        ]
        
        assert len(ai_influenced_trades) > 0, "AI signals should influence trading decisions"
        
        # Verify strategy attribution
        assert "strategy_attribution" in backtest_result
        attribution = backtest_result["strategy_attribution"]
        
        assert attribution["f6_strategy_id"] == sample_f6_strategy["strategy_id"]
        assert attribution["f6_strategy_name"] == sample_f6_strategy["strategy_name"]
        assert attribution["f6_family"] == sample_f6_strategy["family"]
    
    @pytest.mark.integration
    async def test_live_backtest_parity_validation(
        self, 
        integration_services, 
        sample_f6_strategy, 
        sample_market_data
    ):
        """
        Test live/backtest parity with identical market data.
        
        Validates that identical strategies produce identical results in both modes.
        """
        services = integration_services
        
        # Translate strategy
        translation_result = await services["strategy_translation"].translate_strategy(
            sample_f6_strategy
        )
        
        assert translation_result["success"] is True
        nautilus_strategy = translation_result["generated_code"]
        
        # Prepare identical market data for both modes
        data_prep_result = await services["data_catalog"].prepare_market_data_for_nautilus(
            market_data=sample_market_data,
            instruments=["EUR/USD"],
            bar_types=["1-HOUR-MID"]
        )
        
        # Mock identical execution results for both modes
        mock_execution_result = {
            "total_return": 0.123,
            "sharpe_ratio": 1.35,
            "max_drawdown": -0.065,
            "total_trades": 18,
            "winning_trades": 12,
            "execution_decisions": [
                {"timestamp": "2023-12-01T10:00:00", "action": "BUY", "quantity": 10000},
                {"timestamp": "2023-12-01T14:00:00", "action": "SELL", "quantity": 10000},
                {"timestamp": "2023-12-01T18:00:00", "action": "BUY", "quantity": 15000},
            ]
        }
        
        # Execute backtest mode
        with patch('nautilus_integration.services.integration_service.BacktestEngine') as mock_backtest_engine:
            mock_backtest_instance = MagicMock()
            mock_backtest_instance.run.return_value = None
            mock_backtest_instance.get_result.return_value = mock_execution_result
            mock_backtest_engine.return_value = mock_backtest_instance
            
            backtest_config = BacktestConfig(
                backtest_id="parity_test_backtest",
                strategy_ids=[sample_f6_strategy["strategy_id"]],
                start_date=sample_market_data['timestamp'].min(),
                end_date=sample_market_data['timestamp'].max(),
                initial_capital=100000.0,
                data_sources=["test_data"],
                instruments=["EUR/USD"],
                bar_types=["1-HOUR-MID"],
            )
            
            backtest_result = await services["integration"].create_backtest(
                config=backtest_config,
                strategies=[nautilus_strategy],
                market_data=sample_market_data
            )
        
        # Execute live trading mode (with replay data)
        with patch('nautilus_integration.services.integration_service.TradingNode') as mock_trading_node:
            mock_node_instance = MagicMock()
            mock_node_instance.start.return_value = None
            mock_node_instance.replay_market_data = AsyncMock(return_value=sample_market_data)
            mock_node_instance.get_execution_summary.return_value = mock_execution_result
            mock_trading_node.return_value = mock_node_instance
            
            live_result = await services["integration"].start_live_trading_with_replay(
                strategies=[nautilus_strategy],
                replay_data=sample_market_data,
                risk_config=sample_f6_strategy["risk_constraints"]
            )
        
        # Validate parity
        assert backtest_result["status"] == "completed"
        assert live_result["status"] == "completed"
        
        # Compare execution results
        backtest_results = backtest_result["results"]
        live_results = live_result["results"]
        
        # Performance metrics should match
        assert abs(backtest_results["total_return"] - live_results["total_return"]) < 1e-6
        assert abs(backtest_results["sharpe_ratio"] - live_results["sharpe_ratio"]) < 1e-6
        assert abs(backtest_results["max_drawdown"] - live_results["max_drawdown"]) < 1e-6
        assert backtest_results["total_trades"] == live_results["total_trades"]
        
        # Execution decisions should match
        backtest_decisions = backtest_results["execution_decisions"]
        live_decisions = live_results["execution_decisions"]
        
        assert len(backtest_decisions) == len(live_decisions)
        
        for i, (bt_decision, live_decision) in enumerate(zip(backtest_decisions, live_decisions)):
            assert bt_decision["timestamp"] == live_decision["timestamp"], (
                f"Decision {i} timestamp mismatch"
            )
            assert bt_decision["action"] == live_decision["action"], (
                f"Decision {i} action mismatch"
            )
            assert bt_decision["quantity"] == live_decision["quantity"], (
                f"Decision {i} quantity mismatch"
            )
    
    @pytest.mark.integration
    async def test_error_handling_and_recovery_scenarios(
        self, 
        integration_services, 
        sample_f6_strategy, 
        sample_market_data
    ):
        """
        Test error handling and recovery scenarios with failure injection.
        
        Validates system resilience and graceful degradation.
        """
        services = integration_services
        
        # Test scenario 1: Strategy translation failure
        corrupted_strategy = sample_f6_strategy.copy()
        corrupted_strategy["parameters"] = {"invalid_param": "invalid_value"}
        
        translation_result = await services["strategy_translation"].translate_strategy(
            corrupted_strategy
        )
        
        # Should handle gracefully
        if not translation_result["success"]:
            assert "error" in translation_result
            assert "error_type" in translation_result
            assert translation_result["error_type"] in [
                "PARAMETER_VALIDATION_ERROR",
                "CODE_GENERATION_ERROR",
                "COMPILATION_ERROR"
            ]
        
        # Test scenario 2: Market data corruption
        corrupted_data = sample_market_data.copy()
        corrupted_data.loc[0, 'high'] = -100.0  # Invalid negative price
        corrupted_data.loc[1, 'volume'] = None   # Missing volume
        
        data_prep_result = await services["data_catalog"].prepare_market_data_for_nautilus(
            market_data=corrupted_data,
            instruments=["EUR/USD"],
            bar_types=["1-HOUR-MID"],
            validate_data=True
        )
        
        # Should detect and handle data quality issues
        assert "data_quality_report" in data_prep_result
        quality_report = data_prep_result["data_quality_report"]
        
        assert "issues_detected" in quality_report
        assert quality_report["issues_detected"] > 0
        
        # Test scenario 3: Signal routing failure
        invalid_signal = {
            "signal_id": "invalid_001",
            "signal_type": "INVALID_TYPE",  # Invalid signal type
            "instrument_id": "EUR/USD",
            "timestamp": datetime.now(),
            "confidence": 1.5,  # Invalid confidence > 1.0
            "value": None,      # Missing value
            "source_model": "",  # Empty source model
            "metadata": {},
        }
        
        signal_result = await services["signal_router"].route_signal(invalid_signal)
        
        # Should handle invalid signals gracefully
        assert "success" in signal_result
        if not signal_result["success"]:
            assert "validation_errors" in signal_result
            assert len(signal_result["validation_errors"]) > 0
        
        # Test scenario 4: Backtest engine failure
        valid_strategy = await services["strategy_translation"].translate_strategy(
            sample_f6_strategy
        )
        
        with patch('nautilus_integration.services.integration_service.BacktestEngine') as mock_engine:
            # Simulate engine failure
            mock_engine.side_effect = Exception("Simulated engine failure")
            
            backtest_config = BacktestConfig(
                backtest_id="failure_test_backtest",
                strategy_ids=[sample_f6_strategy["strategy_id"]],
                start_date=sample_market_data['timestamp'].min(),
                end_date=sample_market_data['timestamp'].max(),
                initial_capital=100000.0,
                data_sources=["test_data"],
                instruments=["EUR/USD"],
                bar_types=["1-HOUR-MID"],
            )
            
            backtest_result = await services["integration"].create_backtest(
                config=backtest_config,
                strategies=[valid_strategy["generated_code"]],
                market_data=sample_market_data
            )
            
            # Should handle engine failure gracefully
            assert backtest_result["status"] == "failed"
            assert "error" in backtest_result
            assert "fallback_attempted" in backtest_result
            
            # Should attempt fallback to F7 engine if configured
            if backtest_result.get("fallback_attempted"):
                assert "fallback_result" in backtest_result
    
    @pytest.mark.integration
    async def test_multi_system_interactions_and_data_consistency(
        self, 
        integration_services, 
        sample_f6_strategy, 
        sample_market_data, 
        sample_ai_signals
    ):
        """
        Test multi-system interactions and data consistency.
        
        Validates data flow and consistency across F5, F6, F8, and Nautilus systems.
        """
        services = integration_services
        
        # Step 1: Set up multi-system data flow
        
        # F6 Strategy Registry interaction
        f6_strategy_data = {
            "strategy_registry_id": "f6_reg_001",
            "strategy_config": sample_f6_strategy,
            "deployment_status": "active",
            "performance_history": [
                {"date": "2023-11-01", "return": 0.02, "sharpe": 1.1},
                {"date": "2023-11-02", "return": 0.015, "sharpe": 1.2},
            ]
        }
        
        # F5 Intelligence Layer interaction
        f5_intelligence_data = {
            "rag_session_id": "f5_rag_001",
            "signals": sample_ai_signals,
            "market_insights": [
                {"insight_type": "regime_change", "confidence": 0.82, "description": "Trending to ranging"},
                {"insight_type": "volatility_spike", "confidence": 0.75, "description": "Expected vol increase"},
            ],
            "knowledge_graph_updates": [
                {"node_id": "EUR_USD", "property": "current_regime", "value": "LOW_VOL_TRENDING"},
            ]
        }
        
        # F8 Risk Management interaction
        f8_risk_data = {
            "risk_session_id": "f8_risk_001",
            "portfolio_state": {
                "total_equity": 100000.0,
                "available_margin": 50000.0,
                "current_positions": [],
                "daily_pnl": 0.0,
            },
            "risk_limits": sample_f6_strategy["risk_constraints"],
            "risk_alerts": [],
        }
        
        # Step 2: Execute integrated workflow
        
        # Mock F6 registry interaction
        with patch.object(services["strategy_translation"], 'sync_with_f6_registry') as mock_f6_sync:
            mock_f6_sync.return_value = f6_strategy_data
            
            # Mock F5 intelligence interaction
            with patch.object(services["signal_router"], 'sync_with_f5_intelligence') as mock_f5_sync:
                mock_f5_sync.return_value = f5_intelligence_data
                
                # Mock F8 risk interaction
                with patch.object(services["f8_risk"], 'sync_portfolio_state') as mock_f8_sync:
                    mock_f8_sync.return_value = f8_risk_data
                    
                    # Execute integrated backtest
                    integration_result = await services["integration"].execute_integrated_backtest(
                        f6_strategy_config=sample_f6_strategy,
                        market_data=sample_market_data,
                        enable_f5_signals=True,
                        enable_f8_risk_checks=True,
                        sync_with_registries=True
                    )
        
        # Step 3: Validate multi-system consistency
        
        assert integration_result["success"] is True
        assert "system_interactions" in integration_result
        
        interactions = integration_result["system_interactions"]
        
        # Verify F6 interaction
        assert "f6_registry_sync" in interactions
        f6_sync = interactions["f6_registry_sync"]
        assert f6_sync["success"] is True
        assert f6_sync["strategy_id"] == sample_f6_strategy["strategy_id"]
        
        # Verify F5 interaction
        assert "f5_intelligence_sync" in interactions
        f5_sync = interactions["f5_intelligence_sync"]
        assert f5_sync["success"] is True
        assert f5_sync["signals_received"] == len(sample_ai_signals)
        
        # Verify F8 interaction
        assert "f8_risk_sync" in interactions
        f8_sync = interactions["f8_risk_sync"]
        assert f8_sync["success"] is True
        assert f8_sync["risk_checks_performed"] > 0
        
        # Verify data consistency across systems
        assert "data_consistency_report" in integration_result
        consistency_report = integration_result["data_consistency_report"]
        
        assert "strategy_consistency" in consistency_report
        assert "signal_consistency" in consistency_report
        assert "risk_consistency" in consistency_report
        
        # Strategy data should be consistent
        strategy_consistency = consistency_report["strategy_consistency"]
        assert strategy_consistency["f6_nautilus_match"] is True
        assert strategy_consistency["parameter_consistency_score"] >= 0.95
        
        # Signal data should be consistent
        signal_consistency = consistency_report["signal_consistency"]
        assert signal_consistency["f5_nautilus_match"] is True
        assert signal_consistency["signal_delivery_success_rate"] >= 0.95
        
        # Risk data should be consistent
        risk_consistency = consistency_report["risk_consistency"]
        assert risk_consistency["f8_nautilus_match"] is True
        assert risk_consistency["risk_limit_enforcement"] is True
        
        # Step 4: Verify audit trail completeness
        assert "audit_trail" in integration_result
        audit_trail = integration_result["audit_trail"]
        
        required_audit_events = [
            "f6_strategy_loaded",
            "f5_signals_received",
            "f8_risk_limits_applied",
            "nautilus_backtest_started",
            "nautilus_backtest_completed",
            "results_synchronized",
        ]
        
        audit_event_types = [event["event_type"] for event in audit_trail]
        
        for required_event in required_audit_events:
            assert required_event in audit_event_types, (
                f"Missing required audit event: {required_event}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])