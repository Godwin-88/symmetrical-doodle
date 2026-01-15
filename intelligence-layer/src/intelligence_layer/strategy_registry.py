"""
Strategy Registry for Production Trading Strategies.

Provides a comprehensive catalog of implementable trading strategies with
configuration, backtesting, and execution capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from datetime import datetime


class StrategyFamily(str, Enum):
    """Strategy family classification."""
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    STATISTICAL_ARB = "statistical_arb"
    REGIME_SWITCHING = "regime_switching"
    SENTIMENT = "sentiment"
    EXECUTION = "execution"


class TimeHorizon(str, Enum):
    """Trading time horizon."""
    INTRADAY = "intraday"  # Minutes to hours
    DAILY = "daily"  # Days
    SWING = "swing"  # Days to weeks
    POSITION = "position"  # Weeks to months


class AssetClass(str, Enum):
    """Asset class."""
    FX = "fx"
    EQUITIES = "equities"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    FIXED_INCOME = "fixed_income"


@dataclass
class StrategySpec:
    """Specification for a trading strategy."""
    id: str
    name: str
    family: StrategyFamily
    horizon: TimeHorizon
    asset_classes: List[AssetClass]
    description: str
    signal_logic: str
    entry_rules: List[str]
    exit_rules: List[str]
    risk_controls: List[str]
    strengths: List[str]
    weaknesses: List[str]
    best_for: List[str]
    production_ready: bool
    complexity: Literal["low", "medium", "high"]
    data_requirements: Literal["small", "medium", "large"]
    latency_sensitivity: Literal["low", "medium", "high"]
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance characteristics
    typical_sharpe: float = 0.0
    typical_max_dd: float = 0.0
    typical_win_rate: float = 0.0
    typical_turnover: float = 0.0  # trades per day
    
    # Risk management
    max_position_size: float = 0.25  # % of portfolio
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.05
    
    # Regime affinity
    regime_affinity: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    paper_url: Optional[str] = None
    implementation_notes: Optional[str] = None


# ============================================================================
# 1. TREND-FOLLOWING STRATEGIES
# ============================================================================

TREND_STRATEGIES = [
    StrategySpec(
        id="ma_crossover",
        name="Moving Average Crossover with Volatility Filter",
        family=StrategyFamily.TREND,
        horizon=TimeHorizon.DAILY,
        asset_classes=[AssetClass.FX, AssetClass.EQUITIES, AssetClass.CRYPTO],
        description="Classic trend-following using fast/slow MA crossover with volatility filter",
        signal_logic="Fast MA (20) crosses above Slow MA (100) → Long; crosses below → Flat/Short",
        entry_rules=[
            "Fast MA crosses above Slow MA",
            "Rolling volatility > threshold",
            "Volume confirmation (optional)"
        ],
        exit_rules=[
            "Fast MA crosses below Slow MA",
            "ATR-based stop loss hit",
            "Time-based exit (max holding period)"
        ],
        risk_controls=[
            "ATR-based stop loss",
            "Max position size per asset",
            "Daily drawdown limit",
            "Volatility scaling"
        ],
        strengths=[
            "Simple and robust",
            "Works in trending markets",
            "Easy to interpret",
            "Battle-tested",
            "Low overfitting risk"
        ],
        weaknesses=[
            "Whipsaws in ranging markets",
            "Lagging indicator",
            "Poor in choppy conditions",
            "Transaction costs can erode profits"
        ],
        best_for=[
            "Trending markets",
            "Medium to long-term horizons",
            "Liquid instruments",
            "Infrastructure validation"
        ],
        production_ready=True,
        complexity="low",
        data_requirements="small",
        latency_sensitivity="low",
        parameters={
            "fast_period": 20,
            "slow_period": 100,
            "vol_lookback": 20,
            "vol_threshold": 0.01,
            "atr_multiplier": 2.0,
            "max_holding_days": 30
        },
        typical_sharpe=1.2,
        typical_max_dd=15.0,
        typical_win_rate=0.45,
        typical_turnover=0.5,
        max_position_size=0.20,
        max_leverage=2.0,
        stop_loss_pct=0.03,
        regime_affinity={
            "LOW_VOL_TRENDING": 0.85,
            "MEDIUM_VOL_TRENDING": 0.75,
            "HIGH_VOL_RANGING": 0.20
        }
    ),
]


# ============================================================================
# 2. MEAN REVERSION STRATEGIES
# ============================================================================

MEAN_REVERSION_STRATEGIES = [
    StrategySpec(
        id="zscore_reversion",
        name="Z-Score Mean Reversion",
        family=StrategyFamily.MEAN_REVERSION,
        horizon=TimeHorizon.INTRADAY,
        asset_classes=[AssetClass.FX, AssetClass.EQUITIES],
        description="Trade mean reversion using z-score of price deviations",
        signal_logic="Enter when z-score exceeds ±2, exit at mean (z ≈ 0)",
        entry_rules=[
            "Z-score > +2 → Short",
            "Z-score < -2 → Long",
            "Only in range-bound regimes"
        ],
        exit_rules=[
            "Z-score returns to 0",
            "Time-based stop (N bars)",
            "Stop loss at ±3 z-score"
        ],
        risk_controls=[
            "Position sizing by volatility",
            "Time-based stops",
            "Regime filter (no trending markets)",
            "Max correlation exposure"
        ],
        strengths=[
            "Works in ranging markets",
            "High win rate",
            "Quick trades",
            "Statistical foundation"
        ],
        weaknesses=[
            "Fails in trends",
            "Transaction cost sensitive",
            "Requires regime detection",
            "Can have large losses in breakouts"
        ],
        best_for=[
            "Range-bound markets",
            "High-frequency trading",
            "Pairs trading",
            "Market making"
        ],
        production_ready=True,
        complexity="medium",
        data_requirements="medium",
        latency_sensitivity="high",
        parameters={
            "lookback_period": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "max_holding_bars": 50,
            "stop_loss_zscore": 3.0
        },
        typical_sharpe=1.5,
        typical_max_dd=8.0,
        typical_win_rate=0.65,
        typical_turnover=5.0,
        max_position_size=0.15,
        max_leverage=1.5,
        stop_loss_pct=0.02,
        regime_affinity={
            "LOW_VOL_RANGING": 0.90,
            "MEDIUM_VOL_RANGING": 0.70,
            "HIGH_VOL_TRENDING": 0.10
        }
    ),
]


# ============================================================================
# 3. MOMENTUM STRATEGIES
# ============================================================================

MOMENTUM_STRATEGIES = [
    StrategySpec(
        id="cross_sectional_momentum",
        name="Cross-Sectional Momentum Rotation",
        family=StrategyFamily.MOMENTUM,
        horizon=TimeHorizon.SWING,
        asset_classes=[AssetClass.EQUITIES, AssetClass.FX],
        description="Rank assets by momentum, long top decile, short bottom decile",
        signal_logic="Rank by 3-12 month returns, rebalance monthly",
        entry_rules=[
            "Rank assets by momentum score",
            "Long top 10% performers",
            "Short bottom 10% (or flat)",
            "Monthly rebalance"
        ],
        exit_rules=[
            "Monthly rebalancing",
            "Asset drops out of top/bottom decile",
            "Stop loss per position"
        ],
        risk_controls=[
            "Volatility scaling per asset",
            "Sector caps",
            "Turnover limits",
            "Max concentration"
        ],
        strengths=[
            "Strong academic backing",
            "Works across asset classes",
            "Diversified portfolio",
            "Robust to market conditions"
        ],
        weaknesses=[
            "High turnover",
            "Crowded trade",
            "Momentum crashes",
            "Requires many assets"
        ],
        best_for=[
            "Portfolio management",
            "Multi-asset strategies",
            "Medium-term trading",
            "Institutional investors"
        ],
        production_ready=True,
        complexity="medium",
        data_requirements="large",
        latency_sensitivity="low",
        parameters={
            "lookback_months": 12,
            "skip_month": 1,
            "top_pct": 10,
            "bottom_pct": 10,
            "rebalance_frequency": "monthly",
            "min_assets": 20
        },
        typical_sharpe=1.0,
        typical_max_dd=20.0,
        typical_win_rate=0.55,
        typical_turnover=2.0,
        max_position_size=0.10,
        max_leverage=1.5,
        stop_loss_pct=0.10,
        regime_affinity={
            "LOW_VOL_TRENDING": 0.75,
            "MEDIUM_VOL_TRENDING": 0.80,
            "CRISIS": 0.30
        }
    ),
]


# ============================================================================
# 4. VOLATILITY BREAKOUT STRATEGIES
# ============================================================================

VOLATILITY_STRATEGIES = [
    StrategySpec(
        id="bollinger_breakout",
        name="Bollinger Band Breakout",
        family=StrategyFamily.VOLATILITY,
        horizon=TimeHorizon.INTRADAY,
        asset_classes=[AssetClass.FX, AssetClass.CRYPTO, AssetClass.EQUITIES],
        description="Trade breakouts from Bollinger Bands with volume confirmation",
        signal_logic="Enter when price closes above upper band, exit at mid-band",
        entry_rules=[
            "Price closes above upper band → Long",
            "Price closes below lower band → Short",
            "Volume > average (confirmation)",
            "High volatility regime"
        ],
        exit_rules=[
            "Price returns to mid-band",
            "Opposite band touched",
            "Time-based exit",
            "Stop loss"
        ],
        risk_controls=[
            "ATR-based position sizing",
            "Volume filter",
            "Regime filter",
            "Max daily trades"
        ],
        strengths=[
            "Captures volatility expansion",
            "Works in breakout markets",
            "Visual and intuitive",
            "Adaptable to volatility"
        ],
        weaknesses=[
            "False breakouts common",
            "Whipsaws in ranging markets",
            "Requires quick execution",
            "Parameter sensitive"
        ],
        best_for=[
            "Volatile markets",
            "Event-driven trading",
            "Crypto markets",
            "News-based breakouts"
        ],
        production_ready=True,
        complexity="low",
        data_requirements="small",
        latency_sensitivity="medium",
        parameters={
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_threshold": 1.5,
            "min_volatility": 0.015,
            "max_holding_bars": 20
        },
        typical_sharpe=0.9,
        typical_max_dd=12.0,
        typical_win_rate=0.40,
        typical_turnover=3.0,
        max_position_size=0.15,
        max_leverage=2.0,
        stop_loss_pct=0.025,
        regime_affinity={
            "HIGH_VOL_RANGING": 0.70,
            "MEDIUM_VOL_TRENDING": 0.60,
            "LOW_VOL_RANGING": 0.30
        }
    ),
]


# ============================================================================
# 5. STATISTICAL ARBITRAGE STRATEGIES
# ============================================================================

STATISTICAL_ARB_STRATEGIES = [
    StrategySpec(
        id="pairs_trading",
        name="Cointegration-Based Pairs Trading",
        family=StrategyFamily.STATISTICAL_ARB,
        horizon=TimeHorizon.DAILY,
        asset_classes=[AssetClass.EQUITIES, AssetClass.FX],
        description="Trade mean-reverting spread between cointegrated pairs",
        signal_logic="Identify cointegrated pairs, trade spread z-score",
        entry_rules=[
            "Identify cointegrated pairs (ADF test)",
            "Compute spread = price_A - beta * price_B",
            "Enter when spread z-score > ±2",
            "Long spread when z < -2, short when z > +2"
        ],
        exit_rules=[
            "Spread returns to mean (z ≈ 0)",
            "Cointegration breaks (rolling test)",
            "Time-based stop",
            "Stop loss at ±3 z-score"
        ],
        risk_controls=[
            "Cointegration monitoring",
            "Correlation breakdown detection",
            "Position sizing by spread volatility",
            "Max pairs per portfolio"
        ],
        strengths=[
            "Market-neutral",
            "Statistical foundation",
            "Lower directional risk",
            "Works in various markets"
        ],
        weaknesses=[
            "Cointegration can break",
            "Requires pair discovery",
            "Transaction costs matter",
            "Structural changes hurt"
        ],
        best_for=[
            "Market-neutral strategies",
            "Hedge fund strategies",
            "Low-correlation portfolios",
            "Statistical traders"
        ],
        production_ready=True,
        complexity="high",
        data_requirements="large",
        latency_sensitivity="medium",
        parameters={
            "lookback_period": 60,
            "cointegration_pvalue": 0.05,
            "entry_zscore": 2.0,
            "exit_zscore": 0.5,
            "stop_zscore": 3.0,
            "retest_frequency": 20
        },
        typical_sharpe=1.8,
        typical_max_dd=6.0,
        typical_win_rate=0.70,
        typical_turnover=2.0,
        max_position_size=0.20,
        max_leverage=1.0,
        stop_loss_pct=0.03,
        regime_affinity={
            "LOW_VOL_RANGING": 0.85,
            "MEDIUM_VOL_RANGING": 0.75,
            "CRISIS": 0.20
        }
    ),
]


# ============================================================================
# 6. REGIME-SWITCHING STRATEGIES
# ============================================================================

REGIME_SWITCHING_STRATEGIES = [
    StrategySpec(
        id="adaptive_regime",
        name="Adaptive Regime-Switching Strategy",
        family=StrategyFamily.REGIME_SWITCHING,
        horizon=TimeHorizon.DAILY,
        asset_classes=[AssetClass.FX, AssetClass.EQUITIES, AssetClass.CRYPTO],
        description="Switch between trend-following and mean-reversion based on regime",
        signal_logic="Detect regime, apply appropriate strategy",
        entry_rules=[
            "Detect current regime (HMM/Hurst/Volatility)",
            "Trend regime → use trend-following",
            "Range regime → use mean-reversion",
            "Crisis regime → reduce exposure"
        ],
        exit_rules=[
            "Regime change detected",
            "Strategy-specific exits",
            "Risk limit breached"
        ],
        risk_controls=[
            "Regime confidence threshold",
            "Gradual position transitions",
            "Max leverage per regime",
            "Drawdown limits"
        ],
        strengths=[
            "Adapts to market conditions",
            "Reduces whipsaws",
            "Professional-grade",
            "Robust across cycles"
        ],
        weaknesses=[
            "Complex implementation",
            "Regime detection lag",
            "Requires multiple strategies",
            "Higher turnover"
        ],
        best_for=[
            "Sophisticated traders",
            "Multi-strategy funds",
            "Adaptive systems",
            "Professional trading"
        ],
        production_ready=True,
        complexity="high",
        data_requirements="large",
        latency_sensitivity="low",
        parameters={
            "regime_lookback": 60,
            "hurst_threshold": 0.5,
            "vol_threshold": 0.02,
            "confidence_min": 0.7,
            "transition_period": 5
        },
        typical_sharpe=1.6,
        typical_max_dd=10.0,
        typical_win_rate=0.60,
        typical_turnover=1.5,
        max_position_size=0.25,
        max_leverage=2.0,
        stop_loss_pct=0.04,
        regime_affinity={
            "ALL_REGIMES": 0.80
        }
    ),
]


# ============================================================================
# 7. SENTIMENT STRATEGIES
# ============================================================================

SENTIMENT_STRATEGIES = [
    StrategySpec(
        id="news_sentiment",
        name="News Sentiment Reaction Strategy",
        family=StrategyFamily.SENTIMENT,
        horizon=TimeHorizon.INTRADAY,
        asset_classes=[AssetClass.EQUITIES, AssetClass.CRYPTO],
        description="Trade on large sentiment shocks from news/social media",
        signal_logic="Trade only on large sentiment deltas",
        entry_rules=[
            "Ingest news sentiment score",
            "Sentiment delta > threshold → Long",
            "Sentiment delta < -threshold → Short",
            "Latency critical (first mover advantage)"
        ],
        exit_rules=[
            "Sentiment normalizes",
            "Time-based exit (sentiment decays)",
            "Opposite sentiment shock",
            "Stop loss"
        ],
        risk_controls=[
            "False positive filtering",
            "Source reliability weighting",
            "Max position per event",
            "Rapid stop loss"
        ],
        strengths=[
            "Captures information edge",
            "Works with modern NLP",
            "High potential alpha",
            "Event-driven"
        ],
        weaknesses=[
            "Latency critical",
            "False positives common",
            "Requires NLP infrastructure",
            "Crowded in popular stocks"
        ],
        best_for=[
            "High-frequency trading",
            "Event-driven funds",
            "Crypto markets",
            "News-sensitive assets"
        ],
        production_ready=False,  # Requires NLP infrastructure
        complexity="high",
        data_requirements="large",
        latency_sensitivity="high",
        parameters={
            "sentiment_threshold": 0.7,
            "decay_halflife": 30,  # minutes
            "min_source_reliability": 0.8,
            "max_holding_minutes": 120
        },
        typical_sharpe=1.3,
        typical_max_dd=8.0,
        typical_win_rate=0.55,
        typical_turnover=10.0,
        max_position_size=0.10,
        max_leverage=1.5,
        stop_loss_pct=0.02,
        regime_affinity={
            "HIGH_VOL_RANGING": 0.75,
            "MEDIUM_VOL_TRENDING": 0.60,
            "CRISIS": 0.85
        }
    ),
]


# ============================================================================
# 8. EXECUTION STRATEGIES
# ============================================================================

EXECUTION_STRATEGIES = [
    StrategySpec(
        id="vwap_execution",
        name="VWAP Execution Strategy",
        family=StrategyFamily.EXECUTION,
        horizon=TimeHorizon.INTRADAY,
        asset_classes=[AssetClass.FX, AssetClass.EQUITIES, AssetClass.CRYPTO],
        description="Execute large orders with minimal market impact using VWAP",
        signal_logic="Split order into time slices, execute proportionally to volume",
        entry_rules=[
            "Divide order into N time slices",
            "Execute proportionally to historical volume profile",
            "Monitor real-time volume",
            "Adjust execution rate dynamically"
        ],
        exit_rules=[
            "Order fully executed",
            "Time window expired",
            "Market impact threshold exceeded"
        ],
        risk_controls=[
            "Market impact monitoring",
            "Slippage limits",
            "Participation rate caps",
            "Adverse selection detection"
        ],
        strengths=[
            "Minimizes market impact",
            "Industry standard",
            "Predictable execution",
            "Works for large orders"
        ],
        weaknesses=[
            "Not alpha-generating",
            "Vulnerable to gaming",
            "Requires volume data",
            "May miss opportunities"
        ],
        best_for=[
            "Large order execution",
            "Institutional trading",
            "Minimizing slippage",
            "Algorithmic execution"
        ],
        production_ready=True,
        complexity="medium",
        data_requirements="medium",
        latency_sensitivity="medium",
        parameters={
            "num_slices": 20,
            "participation_rate": 0.10,
            "urgency": 0.5,
            "max_deviation": 0.005
        },
        typical_sharpe=0.0,  # Execution strategy, not alpha
        typical_max_dd=0.0,
        typical_win_rate=0.0,
        typical_turnover=0.0,
        max_position_size=1.0,
        max_leverage=1.0,
        stop_loss_pct=0.0,
        regime_affinity={}
    ),
]


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

class StrategyRegistry:
    """Central registry for all trading strategies."""
    
    def __init__(self):
        self.strategies: Dict[str, StrategySpec] = {}
        self._register_all_strategies()
    
    def _register_all_strategies(self):
        """Register all strategies."""
        all_strategies = (
            TREND_STRATEGIES +
            MEAN_REVERSION_STRATEGIES +
            MOMENTUM_STRATEGIES +
            VOLATILITY_STRATEGIES +
            STATISTICAL_ARB_STRATEGIES +
            REGIME_SWITCHING_STRATEGIES +
            SENTIMENT_STRATEGIES +
            EXECUTION_STRATEGIES
        )
        
        for strategy in all_strategies:
            self.strategies[strategy.id] = strategy
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategySpec]:
        """Get strategy by ID."""
        return self.strategies.get(strategy_id)
    
    def list_strategies(
        self,
        family: Optional[StrategyFamily] = None,
        horizon: Optional[TimeHorizon] = None,
        asset_class: Optional[AssetClass] = None,
        production_ready: Optional[bool] = None,
    ) -> List[StrategySpec]:
        """List strategies with optional filtering."""
        strategies = list(self.strategies.values())
        
        if family:
            strategies = [s for s in strategies if s.family == family]
        
        if horizon:
            strategies = [s for s in strategies if s.horizon == horizon]
        
        if asset_class:
            strategies = [s for s in strategies if asset_class in s.asset_classes]
        
        if production_ready is not None:
            strategies = [s for s in strategies if s.production_ready == production_ready]
        
        return strategies
    
    def get_recommended_strategies(
        self,
        asset_class: AssetClass,
        horizon: TimeHorizon,
        regime: Optional[str] = None
    ) -> List[StrategySpec]:
        """Get recommended strategies for given conditions."""
        strategies = self.list_strategies(
            asset_class=asset_class,
            horizon=horizon,
            production_ready=True
        )
        
        # Sort by regime affinity if regime provided
        if regime:
            strategies = [s for s in strategies if regime in s.regime_affinity]
            strategies.sort(key=lambda s: s.regime_affinity.get(regime, 0), reverse=True)
        
        return strategies
    
    def get_strategy_families(self) -> List[str]:
        """Get all strategy families."""
        return [f.value for f in StrategyFamily]
    
    def get_time_horizons(self) -> List[str]:
        """Get all time horizons."""
        return [h.value for h in TimeHorizon]
    
    def get_asset_classes(self) -> List[str]:
        """Get all asset classes."""
        return [a.value for a in AssetClass]


# Global registry instance
strategy_registry = StrategyRegistry()
