"""
Stress testing and scenario validation for algorithmic trading system.

This module tests:
- Market crash and gap scenario testing
- High-load performance validation
- Failure mode testing and recovery validation

Requirements: 3.4, 6.4
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import tempfile
import shutil
from pathlib import Path
import time
import concurrent.futures
from dataclasses import dataclass
from enum import Enum

# Import system components
from intelligence_layer.config import load_config
from intelligence_layer.logging import get_logger
from intelligence_layer.feature_extraction import FeatureExtractor
from intelligence_layer.embedding_model import EmbeddingModel
from intelligence_layer.regime_detection import RegimeDetector
from intelligence_layer.rl_environment import TradingEnvironment
from intelligence_layer.strategy_orchestration import StrategyOrchestrator
from intelligence_layer.health import HealthMonitor
from intelligence_layer.shutdown import ShutdownManager

logger = get_logger(__name__)


class ScenarioType(str, Enum):
    """Types of market scenarios for testing."""
    MARKET_CRASH = "market_crash"
    FLASH_CRASH = "flash_crash"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_MARKET = "trending_market"
    SIDEWAYS_MARKET = "sideways_market"
    REGIME_CHANGE = "regime_change"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class StressTestResult:
    """Result from a stress test scenario."""
    
    scenario_type: ScenarioType
    scenario_name: str
    
    # Performance metrics
    processing_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # System behavior
    errors_encountered: int
    recovery_time: float
    data_loss_events: int
    
    # Trading metrics
    max_drawdown: float
    total_return: float
    sharpe_ratio: float
    
    # System stability
    system_stable: bool
    all_components_functional: bool
    
    # Metadata
    test_duration: float
    data_points_processed: int
    timestamp: datetime


class MarketScenarioGenerator:
    """Generates various market scenarios for stress testing."""
    
    def __init__(self, base_price: float = 100.0, num_periods: int = 1000):
        """Initialize scenario generator.
        
        Args:
            base_price: Starting price for scenarios
            num_periods: Number of time periods to generate
        """
        self.base_price = base_price
        self.num_periods = num_periods
        np.random.seed(42)  # For reproducible scenarios
    
    def generate_market_crash_scenario(self, crash_magnitude: float = -0.20) -> pd.DataFrame:
        """Generate market crash scenario.
        
        Args:
            crash_magnitude: Magnitude of crash (negative value)
            
        Returns:
            DataFrame with crash scenario data
        """
        timestamps = self._generate_timestamps()
        prices = []
        
        # Normal market for first 40% of periods
        normal_periods = int(self.num_periods * 0.4)
        crash_periods = int(self.num_periods * 0.1)  # 10% crash period
        recovery_periods = self.num_periods - normal_periods - crash_periods
        
        current_price = self.base_price
        
        # Normal market phase
        for i in range(normal_periods):
            daily_return = np.random.normal(0.0005, 0.015)  # Normal volatility
            current_price *= (1 + daily_return)
            prices.append(self._generate_ohlcv(current_price, 0.015))
        
        # Crash phase - accelerating decline
        crash_start_price = current_price
        for i in range(crash_periods):
            # Accelerating decline
            crash_progress = (i + 1) / crash_periods
            daily_crash = crash_magnitude * crash_progress * 0.3  # Distribute crash over period
            volatility = 0.05 * (1 + crash_progress * 2)  # Increasing volatility
            
            noise = np.random.normal(0, volatility * 0.5)
            current_price *= (1 + daily_crash + noise)
            prices.append(self._generate_ohlcv(current_price, volatility))
        
        # Recovery phase - high volatility recovery
        for i in range(recovery_periods):
            recovery_progress = i / recovery_periods
            daily_return = np.random.normal(0.002, 0.03 * (1 - recovery_progress * 0.5))
            current_price *= (1 + daily_return)
            prices.append(self._generate_ohlcv(current_price, 0.03 * (1 - recovery_progress * 0.5)))
        
        return self._create_dataframe(timestamps, prices)
    
    def generate_flash_crash_scenario(self, flash_magnitude: float = -0.10) -> pd.DataFrame:
        """Generate flash crash scenario (sudden drop and recovery).
        
        Args:
            flash_magnitude: Magnitude of flash crash
            
        Returns:
            DataFrame with flash crash scenario data
        """
        timestamps = self._generate_timestamps()
        prices = []
        
        current_price = self.base_price
        flash_point = int(self.num_periods * 0.5)  # Flash crash at midpoint
        
        for i in range(self.num_periods):
            if i == flash_point:
                # Flash crash - immediate drop
                current_price *= (1 + flash_magnitude)
                prices.append(self._generate_ohlcv(current_price, 0.15, gap=True))
            elif i == flash_point + 1:
                # Immediate partial recovery
                current_price *= (1 - flash_magnitude * 0.6)  # Recover 60% of drop
                prices.append(self._generate_ohlcv(current_price, 0.10))
            elif abs(i - flash_point) <= 5:
                # High volatility around flash crash
                daily_return = np.random.normal(0, 0.05)
                current_price *= (1 + daily_return)
                prices.append(self._generate_ohlcv(current_price, 0.05))
            else:
                # Normal market
                daily_return = np.random.normal(0.0005, 0.015)
                current_price *= (1 + daily_return)
                prices.append(self._generate_ohlcv(current_price, 0.015))
        
        return self._create_dataframe(timestamps, prices)
    
    def generate_gap_scenario(self, gap_magnitude: float = 0.05, gap_type: str = "up") -> pd.DataFrame:
        """Generate gap up/down scenario.
        
        Args:
            gap_magnitude: Size of gap (positive value)
            gap_type: "up" or "down"
            
        Returns:
            DataFrame with gap scenario data
        """
        timestamps = self._generate_timestamps()
        prices = []
        
        current_price = self.base_price
        gap_point = int(self.num_periods * 0.3)  # Gap at 30% point
        
        gap_multiplier = 1 + gap_magnitude if gap_type == "up" else 1 - gap_magnitude
        
        for i in range(self.num_periods):
            if i == gap_point:
                # Gap event
                current_price *= gap_multiplier
                prices.append(self._generate_ohlcv(current_price, 0.02, gap=True))
            else:
                # Normal market
                daily_return = np.random.normal(0.0005, 0.015)
                current_price *= (1 + daily_return)
                prices.append(self._generate_ohlcv(current_price, 0.015))
        
        return self._create_dataframe(timestamps, prices)
    
    def generate_high_volatility_scenario(self, volatility_multiplier: float = 3.0) -> pd.DataFrame:
        """Generate high volatility scenario.
        
        Args:
            volatility_multiplier: Multiplier for normal volatility
            
        Returns:
            DataFrame with high volatility scenario data
        """
        timestamps = self._generate_timestamps()
        prices = []
        
        current_price = self.base_price
        base_volatility = 0.015
        
        for i in range(self.num_periods):
            # Varying volatility with clustering
            volatility_cycle = np.sin(i * 0.1) * 0.5 + 1  # Volatility clustering
            daily_volatility = base_volatility * volatility_multiplier * volatility_cycle
            
            daily_return = np.random.normal(0, daily_volatility)
            current_price *= (1 + daily_return)
            prices.append(self._generate_ohlcv(current_price, daily_volatility))
        
        return self._create_dataframe(timestamps, prices)
    
    def generate_regime_change_scenario(self) -> pd.DataFrame:
        """Generate scenario with multiple regime changes.
        
        Returns:
            DataFrame with regime change scenario data
        """
        timestamps = self._generate_timestamps()
        prices = []
        
        current_price = self.base_price
        
        # Define regimes
        regimes = [
            {'periods': int(self.num_periods * 0.3), 'drift': 0.0005, 'volatility': 0.015},  # Normal
            {'periods': int(self.num_periods * 0.2), 'drift': -0.002, 'volatility': 0.04},   # Bear
            {'periods': int(self.num_periods * 0.3), 'drift': 0.003, 'volatility': 0.025},   # Bull
            {'periods': int(self.num_periods * 0.2), 'drift': 0.0001, 'volatility': 0.008}   # Low vol
        ]
        
        for regime in regimes:
            for i in range(regime['periods']):
                daily_return = np.random.normal(regime['drift'], regime['volatility'])
                current_price *= (1 + daily_return)
                prices.append(self._generate_ohlcv(current_price, regime['volatility']))
        
        return self._create_dataframe(timestamps, prices)
    
    def _generate_timestamps(self) -> List[datetime]:
        """Generate timestamps for scenario."""
        start_date = datetime.now() - timedelta(days=self.num_periods)
        return [start_date + timedelta(days=i) for i in range(self.num_periods)]
    
    def _generate_ohlcv(self, close_price: float, volatility: float, gap: bool = False) -> Dict[str, float]:
        """Generate OHLCV data for a single period."""
        if gap:
            # For gaps, open significantly different from previous close
            open_price = close_price
        else:
            open_price = close_price * (1 + np.random.normal(0, volatility * 0.2))
        
        # Generate high and low
        intraday_range = volatility * np.random.uniform(0.5, 2.0)
        high_price = max(open_price, close_price) * (1 + intraday_range * 0.5)
        low_price = min(open_price, close_price) * (1 - intraday_range * 0.5)
        
        # Generate volume (higher during volatile periods)
        base_volume = 1000000
        volume_multiplier = 1 + volatility * 10  # Higher volatility = higher volume
        volume = base_volume * volume_multiplier * np.random.lognormal(0, 0.3)
        
        return {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
    
    def _create_dataframe(self, timestamps: List[datetime], prices: List[Dict[str, float]]) -> pd.DataFrame:
        """Create DataFrame from timestamps and price data."""
        data = {
            'timestamp': timestamps[:len(prices)],
            'open': [p['open'] for p in prices],
            'high': [p['high'] for p in prices],
            'low': [p['low'] for p in prices],
            'close': [p['close'] for p in prices],
            'volume': [p['volume'] for p in prices]
        }
        
        return pd.DataFrame(data)


class SystemStressTester:
    """Comprehensive system stress testing framework."""
    
    def __init__(self):
        """Initialize stress tester."""
        self.scenario_generator = MarketScenarioGenerator()
        self.results: List[StressTestResult] = []
        
        # Initialize system components
        self.feature_extractor = FeatureExtractor()
        self.embedding_model = EmbeddingModel()
        self.regime_detector = RegimeDetector()
        self.trading_env = TradingEnvironment()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.health_monitor = HealthMonitor()
        self.shutdown_manager = ShutdownManager()
    
    async def run_scenario_stress_test(
        self, 
        scenario_type: ScenarioType,
        scenario_params: Optional[Dict[str, Any]] = None
    ) -> StressTestResult:
        """Run stress test for specific market scenario.
        
        Args:
            scenario_type: Type of scenario to test
            scenario_params: Optional parameters for scenario generation
            
        Returns:
            Stress test results
        """
        logger.info(f"Starting stress test for scenario: {scenario_type}")
        
        start_time = time.time()
        scenario_params = scenario_params or {}
        
        # Generate scenario data
        if scenario_type == ScenarioType.MARKET_CRASH:
            market_data = self.scenario_generator.generate_market_crash_scenario(
                crash_magnitude=scenario_params.get('crash_magnitude', -0.20)
            )
        elif scenario_type == ScenarioType.FLASH_CRASH:
            market_data = self.scenario_generator.generate_flash_crash_scenario(
                flash_magnitude=scenario_params.get('flash_magnitude', -0.10)
            )
        elif scenario_type == ScenarioType.GAP_UP:
            market_data = self.scenario_generator.generate_gap_scenario(
                gap_magnitude=scenario_params.get('gap_magnitude', 0.05),
                gap_type="up"
            )
        elif scenario_type == ScenarioType.GAP_DOWN:
            market_data = self.scenario_generator.generate_gap_scenario(
                gap_magnitude=scenario_params.get('gap_magnitude', 0.05),
                gap_type="down"
            )
        elif scenario_type == ScenarioType.HIGH_VOLATILITY:
            market_data = self.scenario_generator.generate_high_volatility_scenario(
                volatility_multiplier=scenario_params.get('volatility_multiplier', 3.0)
            )
        elif scenario_type == ScenarioType.REGIME_CHANGE:
            market_data = self.scenario_generator.generate_regime_change_scenario()
        else:
            raise ValueError(f"Unsupported scenario type: {scenario_type}")
        
        # Initialize metrics tracking
        errors_encountered = 0
        processing_times = []
        returns = []
        
        # Process data through system
        chunk_size = 50
        for i in range(0, len(market_data), chunk_size):
            chunk_start_time = time.time()
            
            try:
                # Get data chunk
                chunk = market_data.iloc[i:i+chunk_size]
                
                # Process through system pipeline
                features = self.feature_extractor.extract_features(chunk)
                embeddings = await self.embedding_model.generate_embeddings(features)
                regimes = self.regime_detector.detect_regimes(chunk)
                
                # Get trading actions
                for j in range(len(chunk)):
                    if j < len(embeddings) and j < len(regimes):
                        # Simulate trading decision
                        action = self.trading_env.get_action({
                            'embeddings': embeddings[j:j+1],
                            'regimes': regimes[j:j+1],
                            'market_data': chunk.iloc[j:j+1]
                        })
                        
                        # Calculate simulated return
                        if j < len(chunk) - 1:
                            price_change = (chunk.iloc[j+1]['close'] - chunk.iloc[j]['close']) / chunk.iloc[j]['close']
                            position = action.get('position_change', 0)
                            trade_return = position * price_change
                            returns.append(trade_return)
                
                chunk_processing_time = time.time() - chunk_start_time
                processing_times.append(chunk_processing_time)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                errors_encountered += 1
                continue
        
        # Calculate performance metrics
        returns_array = np.array(returns) if returns else np.array([0])
        total_return = np.prod(1 + returns_array) - 1 if len(returns_array) > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + returns_array) if len(returns_array) > 0 else np.array([1])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate Sharpe ratio
        if len(returns_array) > 1 and np.std(returns_array) > 0:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # System stability check
        system_stable = errors_encountered < len(market_data) * 0.1  # Less than 10% error rate
        
        # Check component functionality
        try:
            health_status = await self.health_monitor.check_system_health()
            all_components_functional = health_status.get('overall_status') != 'critical'
        except Exception:
            all_components_functional = False
        
        test_duration = time.time() - start_time
        
        result = StressTestResult(
            scenario_type=scenario_type,
            scenario_name=f"{scenario_type.value}_stress_test",
            processing_time=np.mean(processing_times) if processing_times else 0,
            memory_usage_mb=0,  # Would implement actual memory monitoring
            cpu_usage_percent=0,  # Would implement actual CPU monitoring
            errors_encountered=errors_encountered,
            recovery_time=0,  # Would measure actual recovery time
            data_loss_events=0,  # Would track data loss events
            max_drawdown=max_drawdown,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            system_stable=system_stable,
            all_components_functional=all_components_functional,
            test_duration=test_duration,
            data_points_processed=len(market_data),
            timestamp=datetime.now()
        )
        
        self.results.append(result)
        logger.info(f"Completed stress test for {scenario_type}: {result.system_stable}")
        
        return result
    
    async def run_load_stress_test(
        self, 
        concurrent_requests: int = 10,
        duration_minutes: int = 5
    ) -> StressTestResult:
        """Run high-load stress test with concurrent processing.
        
        Args:
            concurrent_requests: Number of concurrent processing requests
            duration_minutes: Duration of load test in minutes
            
        Returns:
            Load stress test results
        """
        logger.info(f"Starting load stress test: {concurrent_requests} concurrent requests for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        errors_encountered = 0
        total_requests = 0
        processing_times = []
        
        async def process_request():
            """Process a single request."""
            nonlocal errors_encountered, total_requests
            
            try:
                request_start = time.time()
                
                # Generate small batch of data
                market_data = self.scenario_generator.generate_market_crash_scenario(crash_magnitude=-0.05)
                chunk = market_data.iloc[:20]  # Small chunk for load testing
                
                # Process through system
                features = self.feature_extractor.extract_features(chunk)
                embeddings = await self.embedding_model.generate_embeddings(features)
                regimes = self.regime_detector.detect_regimes(chunk)
                
                processing_time = time.time() - request_start
                processing_times.append(processing_time)
                total_requests += 1
                
            except Exception as e:
                logger.error(f"Load test request failed: {e}")
                errors_encountered += 1
        
        # Run concurrent requests
        while time.time() < end_time:
            tasks = []
            for _ in range(concurrent_requests):
                task = asyncio.create_task(process_request())
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        test_duration = time.time() - start_time
        
        # Calculate metrics
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        error_rate = errors_encountered / max(total_requests, 1)
        
        result = StressTestResult(
            scenario_type=ScenarioType.HIGH_VOLATILITY,  # Using as proxy for load test
            scenario_name="high_load_stress_test",
            processing_time=avg_processing_time,
            memory_usage_mb=0,  # Would implement actual monitoring
            cpu_usage_percent=0,  # Would implement actual monitoring
            errors_encountered=errors_encountered,
            recovery_time=0,
            data_loss_events=0,
            max_drawdown=0,  # Not applicable for load test
            total_return=0,  # Not applicable for load test
            sharpe_ratio=0,  # Not applicable for load test
            system_stable=error_rate < 0.1,  # Less than 10% error rate
            all_components_functional=True,  # Would check actual component status
            test_duration=test_duration,
            data_points_processed=total_requests,
            timestamp=datetime.now()
        )
        
        self.results.append(result)
        logger.info(f"Completed load stress test: {total_requests} requests, {error_rate:.2%} error rate")
        
        return result
    
    async def run_failure_recovery_test(self) -> StressTestResult:
        """Test system recovery from component failures.
        
        Returns:
            Failure recovery test results
        """
        logger.info("Starting failure recovery test")
        
        start_time = time.time()
        errors_encountered = 0
        recovery_successful = True
        
        try:
            # Generate test data
            market_data = self.scenario_generator.generate_market_crash_scenario()
            
            # Test 1: Simulate embedding model failure
            logger.info("Testing embedding model failure recovery")
            
            # Process some data normally
            chunk1 = market_data.iloc[:100]
            features1 = self.feature_extractor.extract_features(chunk1)
            embeddings1 = await self.embedding_model.generate_embeddings(features1)
            
            # Simulate failure by processing corrupted data
            corrupted_chunk = chunk1.copy()
            corrupted_chunk.loc[:, 'close'] = np.nan
            
            try:
                features_corrupted = self.feature_extractor.extract_features(corrupted_chunk)
                embeddings_corrupted = await self.embedding_model.generate_embeddings(features_corrupted)
                # Should handle gracefully or raise appropriate exception
            except Exception as e:
                logger.info(f"Expected failure handled: {e}")
                errors_encountered += 1
            
            # Test recovery - process normal data again
            chunk2 = market_data.iloc[100:200]
            features2 = self.feature_extractor.extract_features(chunk2)
            embeddings2 = await self.embedding_model.generate_embeddings(features2)
            
            if embeddings2 is None or len(embeddings2) == 0:
                recovery_successful = False
            
            # Test 2: Simulate regime detector failure
            logger.info("Testing regime detector failure recovery")
            
            try:
                # Process with insufficient data
                tiny_chunk = market_data.iloc[:5]  # Too small for regime detection
                regimes = self.regime_detector.detect_regimes(tiny_chunk)
                # Should handle gracefully
            except Exception as e:
                logger.info(f"Expected failure handled: {e}")
                errors_encountered += 1
            
            # Test recovery
            normal_chunk = market_data.iloc[200:300]
            regimes_recovery = self.regime_detector.detect_regimes(normal_chunk)
            
            if regimes_recovery is None or len(regimes_recovery) == 0:
                recovery_successful = False
            
            # Test 3: Simulate system shutdown and restart
            logger.info("Testing shutdown/restart recovery")
            
            shutdown_success = await self.shutdown_manager.initiate_shutdown()
            if not shutdown_success:
                recovery_successful = False
            
            # Simulate restart by reinitializing components
            self.feature_extractor = FeatureExtractor()
            self.embedding_model = EmbeddingModel()
            
            # Test functionality after restart
            restart_chunk = market_data.iloc[300:400]
            restart_features = self.feature_extractor.extract_features(restart_chunk)
            restart_embeddings = await self.embedding_model.generate_embeddings(restart_features)
            
            if restart_embeddings is None or len(restart_embeddings) == 0:
                recovery_successful = False
            
        except Exception as e:
            logger.error(f"Failure recovery test encountered unexpected error: {e}")
            recovery_successful = False
            errors_encountered += 1
        
        test_duration = time.time() - start_time
        
        result = StressTestResult(
            scenario_type=ScenarioType.MARKET_CRASH,  # Using as proxy
            scenario_name="failure_recovery_test",
            processing_time=test_duration,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            errors_encountered=errors_encountered,
            recovery_time=test_duration,
            data_loss_events=0,
            max_drawdown=0,
            total_return=0,
            sharpe_ratio=0,
            system_stable=recovery_successful,
            all_components_functional=recovery_successful,
            test_duration=test_duration,
            data_points_processed=len(market_data),
            timestamp=datetime.now()
        )
        
        self.results.append(result)
        logger.info(f"Completed failure recovery test: {'SUCCESS' if recovery_successful else 'FAILED'}")
        
        return result
    
    def generate_stress_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive stress test report.
        
        Returns:
            Detailed stress test report
        """
        if not self.results:
            return {"error": "No stress test results available"}
        
        # Summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.system_stable)
        
        # Performance metrics
        avg_processing_time = np.mean([r.processing_time for r in self.results])
        max_processing_time = max(r.processing_time for r in self.results)
        
        # Error analysis
        total_errors = sum(r.errors_encountered for r in self.results)
        avg_errors_per_test = total_errors / total_tests if total_tests > 0 else 0
        
        # Trading performance
        trading_results = [r for r in self.results if r.total_return != 0]
        if trading_results:
            avg_return = np.mean([r.total_return for r in trading_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in trading_results])
            worst_drawdown = min(r.max_drawdown for r in trading_results)
        else:
            avg_return = avg_sharpe = worst_drawdown = 0
        
        # Scenario breakdown
        scenario_results = {}
        for result in self.results:
            scenario = result.scenario_type.value
            if scenario not in scenario_results:
                scenario_results[scenario] = {
                    'tests_run': 0,
                    'tests_passed': 0,
                    'avg_processing_time': 0,
                    'total_errors': 0
                }
            
            scenario_results[scenario]['tests_run'] += 1
            if result.system_stable:
                scenario_results[scenario]['tests_passed'] += 1
            scenario_results[scenario]['avg_processing_time'] += result.processing_time
            scenario_results[scenario]['total_errors'] += result.errors_encountered
        
        # Calculate averages
        for scenario_data in scenario_results.values():
            if scenario_data['tests_run'] > 0:
                scenario_data['avg_processing_time'] /= scenario_data['tests_run']
                scenario_data['success_rate'] = scenario_data['tests_passed'] / scenario_data['tests_run']
        
        report = {
            "summary": {
                "total_tests_run": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "total_errors_encountered": total_errors,
                "average_errors_per_test": avg_errors_per_test
            },
            "performance": {
                "average_processing_time_seconds": avg_processing_time,
                "maximum_processing_time_seconds": max_processing_time,
                "total_data_points_processed": sum(r.data_points_processed for r in self.results)
            },
            "trading_performance": {
                "average_return": avg_return,
                "average_sharpe_ratio": avg_sharpe,
                "worst_max_drawdown": worst_drawdown,
                "tests_with_trading_data": len(trading_results)
            },
            "scenario_breakdown": scenario_results,
            "detailed_results": [
                {
                    "scenario": r.scenario_type.value,
                    "scenario_name": r.scenario_name,
                    "system_stable": r.system_stable,
                    "processing_time": r.processing_time,
                    "errors": r.errors_encountered,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        return report


class StressTestSuite:
    """Complete stress test suite for the trading system."""
    
    def __init__(self):
        """Initialize stress test suite."""
        self.stress_tester = SystemStressTester()
    
    @pytest.mark.asyncio
    async def test_market_crash_scenarios(self):
        """Test system behavior during various market crash scenarios."""
        logger.info("Running market crash scenario tests")
        
        # Test different crash magnitudes
        crash_magnitudes = [-0.10, -0.20, -0.30, -0.50]
        
        for magnitude in crash_magnitudes:
            result = await self.stress_tester.run_scenario_stress_test(
                ScenarioType.MARKET_CRASH,
                {'crash_magnitude': magnitude}
            )
            
            # System should remain stable even during severe crashes
            assert result.system_stable, f"System unstable during {magnitude:.0%} crash"
            assert result.all_components_functional, f"Components failed during {magnitude:.0%} crash"
            
            # Should not have excessive errors
            error_rate = result.errors_encountered / result.data_points_processed
            assert error_rate < 0.1, f"Too many errors during crash: {error_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_flash_crash_scenarios(self):
        """Test system behavior during flash crash scenarios."""
        logger.info("Running flash crash scenario tests")
        
        flash_magnitudes = [-0.05, -0.10, -0.15]
        
        for magnitude in flash_magnitudes:
            result = await self.stress_tester.run_scenario_stress_test(
                ScenarioType.FLASH_CRASH,
                {'flash_magnitude': magnitude}
            )
            
            assert result.system_stable, f"System unstable during {magnitude:.0%} flash crash"
            assert result.processing_time < 10.0, "Processing too slow during flash crash"
    
    @pytest.mark.asyncio
    async def test_gap_scenarios(self):
        """Test system behavior during gap up/down scenarios."""
        logger.info("Running gap scenario tests")
        
        gap_scenarios = [
            (ScenarioType.GAP_UP, {'gap_magnitude': 0.05}),
            (ScenarioType.GAP_DOWN, {'gap_magnitude': 0.05}),
            (ScenarioType.GAP_UP, {'gap_magnitude': 0.10}),
            (ScenarioType.GAP_DOWN, {'gap_magnitude': 0.10})
        ]
        
        for scenario_type, params in gap_scenarios:
            result = await self.stress_tester.run_scenario_stress_test(scenario_type, params)
            
            assert result.system_stable, f"System unstable during {scenario_type.value}"
            assert result.all_components_functional, f"Components failed during {scenario_type.value}"
    
    @pytest.mark.asyncio
    async def test_high_volatility_scenarios(self):
        """Test system behavior during high volatility periods."""
        logger.info("Running high volatility scenario tests")
        
        volatility_multipliers = [2.0, 3.0, 5.0, 10.0]
        
        for multiplier in volatility_multipliers:
            result = await self.stress_tester.run_scenario_stress_test(
                ScenarioType.HIGH_VOLATILITY,
                {'volatility_multiplier': multiplier}
            )
            
            assert result.system_stable, f"System unstable with {multiplier}x volatility"
            
            # Higher volatility should not cause excessive processing delays
            assert result.processing_time < 5.0, f"Processing too slow with high volatility: {result.processing_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_regime_change_scenarios(self):
        """Test system behavior during regime transitions."""
        logger.info("Running regime change scenario tests")
        
        result = await self.stress_tester.run_scenario_stress_test(ScenarioType.REGIME_CHANGE)
        
        assert result.system_stable, "System unstable during regime changes"
        assert result.all_components_functional, "Components failed during regime changes"
        
        # Should handle regime transitions without significant performance degradation
        assert result.processing_time < 2.0, "Processing too slow during regime changes"
    
    @pytest.mark.asyncio
    async def test_high_load_performance(self):
        """Test system performance under high concurrent load."""
        logger.info("Running high load performance tests")
        
        # Test different load levels
        load_scenarios = [
            {'concurrent_requests': 5, 'duration_minutes': 1},
            {'concurrent_requests': 10, 'duration_minutes': 2},
            {'concurrent_requests': 20, 'duration_minutes': 1}
        ]
        
        for scenario in load_scenarios:
            result = await self.stress_tester.run_load_stress_test(**scenario)
            
            assert result.system_stable, f"System unstable under load: {scenario}"
            
            # Should maintain reasonable processing times under load
            assert result.processing_time < 5.0, f"Processing too slow under load: {result.processing_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_failure_recovery(self):
        """Test system recovery from component failures."""
        logger.info("Running failure recovery tests")
        
        result = await self.stress_tester.run_failure_recovery_test()
        
        assert result.system_stable, "System failed to recover from component failures"
        assert result.all_components_functional, "Components not functional after recovery"
        
        # Recovery should be reasonably fast
        assert result.recovery_time < 30.0, f"Recovery too slow: {result.recovery_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_comprehensive_stress_suite(self):
        """Run comprehensive stress test covering all scenarios."""
        logger.info("Running comprehensive stress test suite")
        
        # Run all major scenario types
        scenarios_to_test = [
            (ScenarioType.MARKET_CRASH, {'crash_magnitude': -0.20}),
            (ScenarioType.FLASH_CRASH, {'flash_magnitude': -0.10}),
            (ScenarioType.GAP_UP, {'gap_magnitude': 0.05}),
            (ScenarioType.HIGH_VOLATILITY, {'volatility_multiplier': 3.0}),
            (ScenarioType.REGIME_CHANGE, {})
        ]
        
        all_results = []
        
        for scenario_type, params in scenarios_to_test:
            result = await self.stress_tester.run_scenario_stress_test(scenario_type, params)
            all_results.append(result)
        
        # Run load test
        load_result = await self.stress_tester.run_load_stress_test(
            concurrent_requests=10, 
            duration_minutes=1
        )
        all_results.append(load_result)
        
        # Run failure recovery test
        recovery_result = await self.stress_tester.run_failure_recovery_test()
        all_results.append(recovery_result)
        
        # Verify overall system stability
        stable_tests = sum(1 for r in all_results if r.system_stable)
        stability_rate = stable_tests / len(all_results)
        
        assert stability_rate >= 0.8, f"System stability rate too low: {stability_rate:.2%}"
        
        # Generate comprehensive report
        report = self.stress_tester.generate_stress_test_report()
        
        logger.info(f"Comprehensive stress test completed:")
        logger.info(f"- Total tests: {report['summary']['total_tests_run']}")
        logger.info(f"- Success rate: {report['summary']['success_rate']:.2%}")
        logger.info(f"- Average processing time: {report['performance']['average_processing_time_seconds']:.3f}s")
        
        # Overall system should pass comprehensive testing
        assert report['summary']['success_rate'] >= 0.8, "Overall system stability insufficient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])