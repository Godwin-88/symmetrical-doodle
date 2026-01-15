"""
Evaluation metrics and ablation framework for algorithmic trading system.

This module provides:
- Offline evaluation metrics for strategy performance
- Regime-conditioned performance analysis
- Ablation study framework for component evaluation
- Statistical significance testing

Requirements: 7.1-7.4, 7.8
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config import load_config
from .logging import get_logger

logger = get_logger(__name__)
config = load_config()


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    REGIME_SPECIFIC = "regime_specific"
    PREDICTION = "prediction"


@dataclass
class PerformanceMetrics:
    """Container for performance evaluation metrics."""
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    
    # Additional metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    num_trades: int = 0
    num_periods: int = 0


@dataclass
class RegimeMetrics:
    """Performance metrics conditioned on market regime."""
    
    regime_id: str
    regime_name: str
    metrics: PerformanceMetrics
    regime_duration: timedelta
    regime_frequency: float  # Fraction of total time in this regime
    
    # Regime-specific metrics
    regime_detection_accuracy: float = 0.0
    regime_transition_timing: float = 0.0  # How quickly transitions are detected


@dataclass
class PredictionMetrics:
    """Metrics for prediction accuracy evaluation."""
    
    # Regression metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r_squared: float = 0.0
    
    # Classification metrics (for regime prediction)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Time series specific
    directional_accuracy: float = 0.0  # Fraction of correct direction predictions
    hit_rate: float = 0.0  # Fraction of predictions within threshold
    
    # Information content
    information_coefficient: float = 0.0
    rank_correlation: float = 0.0


@dataclass
class AblationResult:
    """Results from an ablation study."""
    
    component_name: str
    baseline_metrics: PerformanceMetrics
    ablated_metrics: PerformanceMetrics
    
    # Statistical significance
    p_value: float = 1.0
    is_significant: bool = False
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Effect size
    effect_size: float = 0.0
    relative_improvement: float = 0.0
    
    # Component importance
    importance_score: float = 0.0


class PerformanceEvaluator:
    """Evaluates trading strategy performance with comprehensive metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02, benchmark_returns: Optional[np.ndarray] = None):
        """Initialize performance evaluator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            benchmark_returns: Benchmark returns for excess return calculation
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        
    def evaluate_returns(
        self, 
        returns: np.ndarray, 
        timestamps: Optional[np.ndarray] = None,
        trades: Optional[List[Dict]] = None
    ) -> PerformanceMetrics:
        """Evaluate performance metrics for a return series.
        
        Args:
            returns: Array of period returns
            timestamps: Optional timestamps for each return
            trades: Optional list of trade records
            
        Returns:
            Comprehensive performance metrics
        """
        if len(returns) == 0:
            return PerformanceMetrics()
        
        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = self._annualize_return(total_return, len(returns))
        
        # Excess return
        excess_return = 0.0
        if self.benchmark_returns is not None and len(self.benchmark_returns) == len(returns):
            benchmark_total = np.prod(1 + self.benchmark_returns) - 1
            excess_return = total_return - benchmark_total
        
        # Risk metrics
        volatility = np.std(returns)
        annualized_volatility = volatility * np.sqrt(252)  # Assuming daily returns
        
        # Downside deviation (volatility of negative returns)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        information_ratio = self._calculate_information_ratio(returns)
        
        # Drawdown metrics
        max_drawdown, max_dd_duration, avg_drawdown = self._calculate_drawdown_metrics(returns)
        
        # Additional metrics
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # Metadata
        start_date = timestamps[0] if timestamps is not None else None
        end_date = timestamps[-1] if timestamps is not None else None
        num_trades = len(trades) if trades is not None else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=excess_return,
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            downside_deviation=downside_deviation,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            average_drawdown=avg_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            start_date=start_date,
            end_date=end_date,
            num_trades=num_trades,
            num_periods=len(returns)
        )
    
    def evaluate_by_regime(
        self, 
        returns: np.ndarray, 
        regime_labels: np.ndarray,
        regime_names: Optional[Dict[str, str]] = None
    ) -> List[RegimeMetrics]:
        """Evaluate performance metrics conditioned on market regime.
        
        Args:
            returns: Array of period returns
            regime_labels: Array of regime identifiers for each period
            regime_names: Optional mapping from regime ID to human-readable name
            
        Returns:
            List of regime-specific performance metrics
        """
        regime_metrics = []
        unique_regimes = np.unique(regime_labels)
        
        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            # Calculate performance metrics for this regime
            metrics = self.evaluate_returns(regime_returns)
            
            # Regime-specific calculations
            regime_duration = timedelta(days=np.sum(regime_mask))
            regime_frequency = np.sum(regime_mask) / len(regime_labels)
            
            # Regime detection accuracy (simplified - would need ground truth)
            regime_detection_accuracy = 0.0  # Placeholder
            regime_transition_timing = 0.0   # Placeholder
            
            regime_name = regime_names.get(str(regime_id), f"Regime_{regime_id}") if regime_names else f"Regime_{regime_id}"
            
            regime_metrics.append(RegimeMetrics(
                regime_id=str(regime_id),
                regime_name=regime_name,
                metrics=metrics,
                regime_duration=regime_duration,
                regime_frequency=regime_frequency,
                regime_detection_accuracy=regime_detection_accuracy,
                regime_transition_timing=regime_transition_timing
            ))
        
        return regime_metrics
    
    def evaluate_predictions(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray,
        prediction_type: str = "regression"
    ) -> PredictionMetrics:
        """Evaluate prediction accuracy metrics.
        
        Args:
            predictions: Array of predicted values
            actuals: Array of actual values
            prediction_type: Type of prediction ("regression" or "classification")
            
        Returns:
            Prediction accuracy metrics
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        metrics = PredictionMetrics()
        
        if prediction_type == "regression":
            # Regression metrics
            metrics.mse = mean_squared_error(actuals, predictions)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(actuals, predictions)
            
            # R-squared
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            metrics.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(actuals))
            pred_direction = np.sign(np.diff(predictions))
            metrics.directional_accuracy = np.mean(actual_direction == pred_direction)
            
            # Information coefficient (correlation)
            if np.std(predictions) > 0 and np.std(actuals) > 0:
                metrics.information_coefficient = np.corrcoef(predictions, actuals)[0, 1]
                metrics.rank_correlation = stats.spearmanr(predictions, actuals)[0]
        
        elif prediction_type == "classification":
            # Classification metrics (assuming binary or multi-class)
            metrics.accuracy = np.mean(predictions == actuals)
            
            # For binary classification
            if len(np.unique(actuals)) == 2:
                tp = np.sum((predictions == 1) & (actuals == 1))
                fp = np.sum((predictions == 1) & (actuals == 0))
                fn = np.sum((predictions == 0) & (actuals == 1))
                
                metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0.0
        
        return metrics
    
    def _annualize_return(self, total_return: float, num_periods: int, periods_per_year: int = 252) -> float:
        """Annualize a total return."""
        if num_periods == 0:
            return 0.0
        return (1 + total_return) ** (periods_per_year / num_periods) - 1
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0
        
        annualized_return = self._annualize_return(np.prod(1 + returns) - 1, len(returns))
        max_drawdown, _, _ = self._calculate_drawdown_metrics(returns)
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / abs(max_drawdown)
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate Information ratio."""
        if self.benchmark_returns is None or len(returns) != len(self.benchmark_returns):
            return 0.0
        
        excess_returns = returns - self.benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(excess_returns) / tracking_error * np.sqrt(252)
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Tuple[float, int, float]:
        """Calculate drawdown metrics."""
        if len(returns) == 0:
            return 0.0, 0, 0.0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Maximum drawdown duration
        max_dd_duration = 0
        current_dd_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # Average drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0.0
        
        return max_drawdown, max_dd_duration, avg_drawdown
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (fraction of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns > 0)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss


class AblationFramework:
    """Framework for conducting ablation studies on trading system components."""
    
    def __init__(self, evaluator: PerformanceEvaluator):
        """Initialize ablation framework.
        
        Args:
            evaluator: Performance evaluator instance
        """
        self.evaluator = evaluator
        self.results: List[AblationResult] = []
    
    def run_ablation_study(
        self, 
        baseline_returns: np.ndarray,
        ablated_returns_dict: Dict[str, np.ndarray],
        significance_level: float = 0.05
    ) -> List[AblationResult]:
        """Run ablation study comparing baseline to ablated versions.
        
        Args:
            baseline_returns: Returns from complete system
            ablated_returns_dict: Dict mapping component names to returns without that component
            significance_level: Statistical significance threshold
            
        Returns:
            List of ablation results
        """
        baseline_metrics = self.evaluator.evaluate_returns(baseline_returns)
        results = []
        
        for component_name, ablated_returns in ablated_returns_dict.items():
            ablated_metrics = self.evaluator.evaluate_returns(ablated_returns)
            
            # Statistical significance test
            p_value = self._test_significance(baseline_returns, ablated_returns)
            is_significant = p_value < significance_level
            
            # Confidence interval for difference in Sharpe ratios
            ci = self._calculate_confidence_interval(
                baseline_metrics.sharpe_ratio, 
                ablated_metrics.sharpe_ratio,
                len(baseline_returns)
            )
            
            # Effect size (difference in Sharpe ratios)
            effect_size = baseline_metrics.sharpe_ratio - ablated_metrics.sharpe_ratio
            
            # Relative improvement
            relative_improvement = (baseline_metrics.total_return - ablated_metrics.total_return) / abs(ablated_metrics.total_return) if ablated_metrics.total_return != 0 else 0.0
            
            # Importance score (normalized effect size)
            importance_score = abs(effect_size) / (1 + abs(baseline_metrics.sharpe_ratio))
            
            result = AblationResult(
                component_name=component_name,
                baseline_metrics=baseline_metrics,
                ablated_metrics=ablated_metrics,
                p_value=p_value,
                is_significant=is_significant,
                confidence_interval=ci,
                effect_size=effect_size,
                relative_improvement=relative_improvement,
                importance_score=importance_score
            )
            
            results.append(result)
        
        # Sort by importance score
        results.sort(key=lambda x: x.importance_score, reverse=True)
        self.results = results
        
        return results
    
    def _test_significance(self, baseline_returns: np.ndarray, ablated_returns: np.ndarray) -> float:
        """Test statistical significance of performance difference."""
        if len(baseline_returns) != len(ablated_returns):
            logger.warning("Return series have different lengths, using minimum length")
            min_len = min(len(baseline_returns), len(ablated_returns))
            baseline_returns = baseline_returns[:min_len]
            ablated_returns = ablated_returns[:min_len]
        
        # Paired t-test for difference in returns
        try:
            statistic, p_value = stats.ttest_rel(baseline_returns, ablated_returns)
            return p_value
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return 1.0
    
    def _calculate_confidence_interval(
        self, 
        baseline_sharpe: float, 
        ablated_sharpe: float, 
        n_observations: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for Sharpe ratio difference."""
        # Simplified confidence interval calculation
        # In practice, would use more sophisticated methods for Sharpe ratio CI
        
        diff = baseline_sharpe - ablated_sharpe
        
        # Approximate standard error (simplified)
        se = np.sqrt(2 / n_observations)  # Rough approximation
        
        # Critical value for confidence level
        alpha = 1 - confidence_level
        critical_value = stats.norm.ppf(1 - alpha / 2)
        
        margin_of_error = critical_value * se
        
        return (diff - margin_of_error, diff + margin_of_error)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive ablation study report."""
        if not self.results:
            return {"error": "No ablation results available"}
        
        # Summary statistics
        significant_components = [r for r in self.results if r.is_significant]
        
        report = {
            "summary": {
                "total_components_tested": len(self.results),
                "significant_components": len(significant_components),
                "most_important_component": self.results[0].component_name if self.results else None,
                "average_effect_size": np.mean([r.effect_size for r in self.results]),
                "total_importance_score": sum(r.importance_score for r in self.results)
            },
            "component_rankings": [
                {
                    "component": r.component_name,
                    "importance_score": r.importance_score,
                    "effect_size": r.effect_size,
                    "p_value": r.p_value,
                    "is_significant": r.is_significant,
                    "relative_improvement": r.relative_improvement
                }
                for r in self.results
            ],
            "significant_findings": [
                {
                    "component": r.component_name,
                    "effect_size": r.effect_size,
                    "confidence_interval": r.confidence_interval,
                    "baseline_sharpe": r.baseline_metrics.sharpe_ratio,
                    "ablated_sharpe": r.ablated_metrics.sharpe_ratio
                }
                for r in significant_components
            ]
        }
        
        return report


# Global evaluator instance
_performance_evaluator = None


def get_performance_evaluator() -> PerformanceEvaluator:
    """Get global performance evaluator instance."""
    global _performance_evaluator
    if _performance_evaluator is None:
        _performance_evaluator = PerformanceEvaluator()
    return _performance_evaluator


def create_ablation_framework() -> AblationFramework:
    """Create new ablation framework instance."""
    evaluator = get_performance_evaluator()
    return AblationFramework(evaluator)