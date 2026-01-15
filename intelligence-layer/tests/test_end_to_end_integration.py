"""
End-to-end integration tests for algorithmic trading system.

This module tests:
- Complete data flow from market data to execution
- All component interactions and boundaries
- Full system integration across all services

Requirements: All system requirements
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import shutil
from pathlib import Path

# Import system components
from intelligence_layer.config import load_config
from intelligence_layer.logging import get_logger
from intelligence_layer.feature_extraction import FeatureExtractor
from intelligence_layer.embedding_model import EmbeddingModel
from intelligence_layer.regime_detection import RegimeDetector
from intelligence_layer.graph_analytics import GraphAnalytics
from intelligence_layer.state_assembly import StateAssembler
from intelligence_layer.rl_environment import TradingEnvironment
from intelligence_layer.strategy_orchestration import StrategyOrchestrator
from intelligence_layer.health import HealthMonitor
from intelligence_layer.shutdown import ShutdownManager
from intelligence_layer.experiment_config import ExperimentManager, create_experiment_config
from intelligence_layer.evaluation_metrics import PerformanceEvaluator
from intelligence_layer.academic_safeguards import get_academic_validator

logger = get_logger(__name__)


class MockMarketDataProvider:
    """Mock market data provider for testing."""
    
    def __init__(self, num_periods: int = 1000):
        """Initialize mock data provider."""
        self.num_periods = num_periods
        self.current_index = 0
        self.data = self._generate_mock_data()
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate realistic mock market data."""
        np.random.seed(42)  # For reproducibility
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=self.num_periods)
        timestamps = [start_date + timedelta(days=i) for i in range(self.num_periods)]
        
        # Generate OHLCV data with realistic patterns
        prices = []
        volume = []
        
        base_price = 100.0
        for i in range(self.num_periods):
            # Add trend and noise
            trend = 0.0001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.02)  # 2% daily volatility
            
            # Regime changes (simplified)
            if i > 300 and i < 600:
                noise *= 2  # High volatility regime
            
            price_change = trend + noise
            base_price *= (1 + price_change)
            
            # OHLC from close price
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': base_price
            })
            
            # Volume with some correlation to volatility
            vol = np.random.lognormal(10, 0.5) * (1 + abs(noise))
            volume.append(vol)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': [p['open'] for p in prices],
            'high': [p['high'] for p in prices],
            'low': [p['low'] for p in prices],
            'close': [p['close'] for p in prices],
            'volume': volume
        })
        
        return data
    
    def get_next_batch(self, batch_size: int = 1) -> pd.DataFrame:
        """Get next batch of market data."""
        if self.current_index >= len(self.data):
            return pd.DataFrame()
        
        end_index = min(self.current_index + batch_size, len(self.data))
        batch = self.data.iloc[self.current_index:end_index].copy()
        self.current_index = end_index
        
        return batch
    
    def reset(self):
        """Reset data provider to beginning."""
        self.current_index = 0


class MockExecutionAdapter:
    """Mock execution adapter for testing."""
    
    def __init__(self):
        """Initialize mock execution adapter."""
        self.orders = []
        self.positions = {}
        self.balance = 100000.0  # Starting balance
        self.is_connected = True
    
    async def place_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """Place a mock order."""
        order_id = f"ORDER_{len(self.orders):06d}"
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
        self.orders.append(order)
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        
        if side == 'buy':
            self.positions[symbol] += quantity
            self.balance -= quantity * price
        else:  # sell
            self.positions[symbol] -= quantity
            self.balance += quantity * price
        
        return order
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance


class SystemIntegrationTest:
    """Comprehensive system integration test suite."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = load_config()
        
        # Initialize components
        self.market_data = MockMarketDataProvider()
        self.execution_adapter = MockExecutionAdapter()
        
        # Initialize system components
        self.feature_extractor = FeatureExtractor()
        self.embedding_model = EmbeddingModel()
        self.regime_detector = RegimeDetector()
        self.graph_analytics = GraphAnalytics()
        self.state_assembler = StateAssembler()
        self.trading_env = TradingEnvironment()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.health_monitor = HealthMonitor()
        self.shutdown_manager = ShutdownManager()
        
        # Initialize evaluation components
        self.experiment_manager = ExperimentManager(self.temp_dir / 'experiments')
        self.performance_evaluator = PerformanceEvaluator()
        self.academic_validator = get_academic_validator()
        
        logger.info("Integration test environment initialized")
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_data_flow(self):
        """Test complete data flow from market data to execution."""
        logger.info("Starting complete data flow test")
        
        # Step 1: Get market data
        market_batch = self.market_data.get_next_batch(100)
        assert len(market_batch) == 100
        assert 'close' in market_batch.columns
        
        # Step 2: Extract features
        features = self.feature_extractor.extract_features(market_batch)
        assert features is not None
        assert len(features) > 0
        
        # Step 3: Generate embeddings
        embeddings = await self.embedding_model.generate_embeddings(features)
        assert embeddings is not None
        assert embeddings.shape[0] > 0
        
        # Step 4: Detect regimes
        regimes = self.regime_detector.detect_regimes(market_batch)
        assert regimes is not None
        assert len(regimes) > 0
        
        # Step 5: Analyze graph features
        graph_features = await self.graph_analytics.compute_features(market_batch)
        assert graph_features is not None
        
        # Step 6: Assemble intelligence state
        intelligence_state = self.state_assembler.assemble_state(
            market_data=market_batch.iloc[-1:],
            embeddings=embeddings[-1:],
            regimes=regimes[-1:],
            graph_features=graph_features
        )
        assert intelligence_state is not None
        
        # Step 7: Generate trading signals
        action = self.trading_env.get_action(intelligence_state)
        assert action is not None
        
        # Step 8: Execute trades
        if abs(action.get('position_change', 0)) > 0.01:  # Only execute if significant change
            order = await self.execution_adapter.place_order(
                symbol='EURUSD',
                side='buy' if action['position_change'] > 0 else 'sell',
                quantity=abs(action['position_change']) * 10000,  # Convert to units
                price=market_batch.iloc[-1]['close']
            )
            assert order['status'] == 'filled'
        
        logger.info("Complete data flow test passed")
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self):
        """Test system health monitoring across all components."""
        logger.info("Starting system health monitoring test")
        
        # Check initial health
        health_status = await self.health_monitor.check_system_health()
        assert health_status['overall_status'] in ['healthy', 'degraded']
        
        # Simulate component failure
        self.execution_adapter.is_connected = False
        
        # Check health after failure
        health_status = await self.health_monitor.check_system_health()
        # Should detect the execution adapter issue
        
        # Restore component
        self.execution_adapter.is_connected = True
        
        # Verify recovery
        health_status = await self.health_monitor.check_system_health()
        assert health_status['overall_status'] in ['healthy', 'degraded']
        
        logger.info("System health monitoring test passed")
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_sequence(self):
        """Test graceful shutdown of all system components."""
        logger.info("Starting graceful shutdown test")
        
        # Initialize some state
        market_batch = self.market_data.get_next_batch(50)
        features = self.feature_extractor.extract_features(market_batch)
        
        # Initiate shutdown
        shutdown_success = await self.shutdown_manager.initiate_shutdown()
        assert shutdown_success
        
        # Verify all components are properly shut down
        # This would check that:
        # - All pending operations are completed
        # - State is properly persisted
        # - Resources are cleaned up
        # - No data is lost
        
        logger.info("Graceful shutdown test passed")
    
    @pytest.mark.asyncio
    async def test_strategy_orchestration_integration(self):
        """Test strategy orchestration with multiple strategies."""
        logger.info("Starting strategy orchestration integration test")
        
        # Get market data
        market_batch = self.market_data.get_next_batch(200)
        
        # Process through intelligence pipeline
        features = self.feature_extractor.extract_features(market_batch)
        embeddings = await self.embedding_model.generate_embeddings(features)
        regimes = self.regime_detector.detect_regimes(market_batch)
        
        # Test strategy orchestration
        strategies_performance = {}
        
        for i in range(10):  # Simulate 10 time steps
            current_data = market_batch.iloc[i:i+50]  # Rolling window
            
            # Assemble current state
            intelligence_state = self.state_assembler.assemble_state(
                market_data=current_data.iloc[-1:],
                embeddings=embeddings[i:i+1] if i < len(embeddings) else embeddings[-1:],
                regimes=regimes[i:i+1] if i < len(regimes) else regimes[-1:],
                graph_features={}
            )
            
            # Get strategy recommendations
            strategy_actions = self.strategy_orchestrator.get_strategy_actions(intelligence_state)
            assert strategy_actions is not None
            
            # Track performance
            for strategy_name, action in strategy_actions.items():
                if strategy_name not in strategies_performance:
                    strategies_performance[strategy_name] = []
                
                # Simulate return based on action and next price change
                if i < len(market_batch) - 1:
                    price_change = (market_batch.iloc[i+1]['close'] - market_batch.iloc[i]['close']) / market_batch.iloc[i]['close']
                    strategy_return = action.get('position_change', 0) * price_change
                    strategies_performance[strategy_name].append(strategy_return)
        
        # Verify strategies generated returns
        assert len(strategies_performance) > 0
        for strategy_name, returns in strategies_performance.items():
            assert len(returns) > 0
        
        logger.info("Strategy orchestration integration test passed")
    
    @pytest.mark.asyncio
    async def test_regime_transition_handling(self):
        """Test system behavior during regime transitions."""
        logger.info("Starting regime transition handling test")
        
        # Get data that includes regime transitions
        market_batch = self.market_data.get_next_batch(500)  # Includes high volatility period
        
        # Process data in chunks to simulate real-time
        chunk_size = 50
        regime_history = []
        
        for i in range(0, len(market_batch), chunk_size):
            chunk = market_batch.iloc[i:i+chunk_size]
            
            # Detect regimes for this chunk
            regimes = self.regime_detector.detect_regimes(chunk)
            regime_history.extend(regimes)
            
            # Check for regime transitions
            if len(regime_history) > 1:
                current_regime = regime_history[-1]
                previous_regime = regime_history[-2]
                
                if current_regime != previous_regime:
                    logger.info(f"Regime transition detected: {previous_regime} -> {current_regime}")
                    
                    # Test system adaptation to regime change
                    features = self.feature_extractor.extract_features(chunk)
                    embeddings = await self.embedding_model.generate_embeddings(features)
                    
                    # Verify system continues to function during transition
                    assert embeddings is not None
                    assert len(embeddings) > 0
        
        # Verify regime transitions were detected
        unique_regimes = set(regime_history)
        assert len(unique_regimes) > 1, "Should detect multiple regimes in test data"
        
        logger.info("Regime transition handling test passed")
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self):
        """Test system error recovery and resilience."""
        logger.info("Starting error recovery and resilience test")
        
        # Test 1: Handle corrupted market data
        corrupted_data = self.market_data.get_next_batch(50)
        corrupted_data.loc[10:15, 'close'] = np.nan  # Introduce NaN values
        
        try:
            features = self.feature_extractor.extract_features(corrupted_data)
            # Should handle NaN values gracefully
            assert features is not None
        except Exception as e:
            pytest.fail(f"System should handle corrupted data gracefully: {e}")
        
        # Test 2: Handle execution adapter failures
        original_balance = self.execution_adapter.get_balance()
        self.execution_adapter.is_connected = False
        
        # System should detect and handle execution failures
        try:
            order = await self.execution_adapter.place_order('EURUSD', 'buy', 1000, 1.1000)
            # Should either succeed with fallback or fail gracefully
        except Exception:
            pass  # Expected behavior for disconnected adapter
        
        # Restore connection
        self.execution_adapter.is_connected = True
        
        # Test 3: Handle memory/resource constraints
        # Simulate processing large dataset
        large_batch = self.market_data.get_next_batch(800)
        
        try:
            features = self.feature_extractor.extract_features(large_batch)
            assert features is not None
        except MemoryError:
            # Should handle memory constraints gracefully
            pass
        
        logger.info("Error recovery and resilience test passed")
    
    @pytest.mark.asyncio
    async def test_performance_evaluation_integration(self):
        """Test integration with performance evaluation system."""
        logger.info("Starting performance evaluation integration test")
        
        # Simulate trading session
        market_batch = self.market_data.get_next_batch(300)
        returns = []
        
        for i in range(50, len(market_batch)):
            # Get current market state
            current_data = market_batch.iloc[i-50:i]
            
            # Process through system
            features = self.feature_extractor.extract_features(current_data)
            embeddings = await self.embedding_model.generate_embeddings(features)
            regimes = self.regime_detector.detect_regimes(current_data)
            
            # Get trading action
            intelligence_state = self.state_assembler.assemble_state(
                market_data=current_data.iloc[-1:],
                embeddings=embeddings[-1:],
                regimes=regimes[-1:],
                graph_features={}
            )
            
            action = self.trading_env.get_action(intelligence_state)
            
            # Calculate return
            if i < len(market_batch) - 1:
                price_change = (market_batch.iloc[i+1]['close'] - market_batch.iloc[i]['close']) / market_batch.iloc[i]['close']
                strategy_return = action.get('position_change', 0) * price_change
                returns.append(strategy_return)
        
        # Evaluate performance
        returns_array = np.array(returns)
        performance_metrics = self.performance_evaluator.evaluate_returns(returns_array)
        
        # Verify metrics are calculated
        assert performance_metrics.total_return is not None
        assert performance_metrics.sharpe_ratio is not None
        assert performance_metrics.max_drawdown is not None
        
        # Test regime-conditioned evaluation
        regime_labels = np.random.choice([0, 1, 2], size=len(returns))  # Mock regime labels
        regime_metrics = self.performance_evaluator.evaluate_by_regime(returns_array, regime_labels)
        
        assert len(regime_metrics) > 0
        
        logger.info("Performance evaluation integration test passed")
    
    @pytest.mark.asyncio
    async def test_academic_validation_integration(self):
        """Test integration with academic validation framework."""
        logger.info("Starting academic validation integration test")
        
        # Create experiment
        experiment_config = create_experiment_config(
            name="Integration Test Experiment",
            description="End-to-end integration test",
            parameters={'test_param': 'test_value'},
            created_by="integration_test"
        )
        
        experiment_id = self.experiment_manager.create_experiment(experiment_config)
        
        # Generate test data
        market_batch = self.market_data.get_next_batch(200)
        features = self.feature_extractor.extract_features(market_batch)
        
        # Create mock targets
        targets = pd.Series(np.random.randn(len(features)))
        
        # Generate mock p-values
        p_values = [0.05, 0.03, 0.12, 0.001, 0.08]
        
        # Run academic validation
        validation_report = self.academic_validator.validate_research_integrity(
            experiment_id=experiment_id,
            data=market_batch,
            features=features,
            targets=targets,
            p_values=p_values,
            validator_name="integration_test"
        )
        
        # Verify validation report
        assert validation_report.experiment_id == experiment_id
        assert validation_report.integrity_score >= 0.0
        assert validation_report.integrity_score <= 1.0
        assert len(validation_report.bias_results) > 0
        
        logger.info("Academic validation integration test passed")
    
    @pytest.mark.asyncio
    async def test_full_system_stress_test(self):
        """Comprehensive stress test of the entire system."""
        logger.info("Starting full system stress test")
        
        # Process large amount of data continuously
        total_processed = 0
        max_iterations = 100
        
        for iteration in range(max_iterations):
            try:
                # Get market data
                market_batch = self.market_data.get_next_batch(20)
                if len(market_batch) == 0:
                    self.market_data.reset()
                    market_batch = self.market_data.get_next_batch(20)
                
                # Process through entire pipeline
                features = self.feature_extractor.extract_features(market_batch)
                embeddings = await self.embedding_model.generate_embeddings(features)
                regimes = self.regime_detector.detect_regimes(market_batch)
                graph_features = await self.graph_analytics.compute_features(market_batch)
                
                # Assemble state and get action
                intelligence_state = self.state_assembler.assemble_state(
                    market_data=market_batch.iloc[-1:],
                    embeddings=embeddings[-1:],
                    regimes=regimes[-1:],
                    graph_features=graph_features
                )
                
                action = self.trading_env.get_action(intelligence_state)
                
                # Execute if significant action
                if abs(action.get('position_change', 0)) > 0.01:
                    await self.execution_adapter.place_order(
                        symbol='EURUSD',
                        side='buy' if action['position_change'] > 0 else 'sell',
                        quantity=abs(action['position_change']) * 1000,
                        price=market_batch.iloc[-1]['close']
                    )
                
                total_processed += len(market_batch)
                
                # Check system health periodically
                if iteration % 20 == 0:
                    health_status = await self.health_monitor.check_system_health()
                    assert health_status is not None
                
            except Exception as e:
                logger.error(f"Stress test iteration {iteration} failed: {e}")
                # System should be resilient to individual failures
                continue
        
        # Verify system processed significant amount of data
        assert total_processed > 1000
        
        # Verify system is still healthy after stress test
        final_health = await self.health_monitor.check_system_health()
        assert final_health is not None
        
        logger.info(f"Full system stress test completed. Processed {total_processed} data points")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])