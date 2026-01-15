/**
 * Trading Strategies API Service
 */

import { intelligenceApi } from './api';

export interface StrategySpec {
  id: string;
  name: string;
  family: string;
  horizon: string;
  asset_classes: string[];
  description: string;
  signal_logic: string;
  entry_rules: string[];
  exit_rules: string[];
  risk_controls: string[];
  strengths: string[];
  weaknesses: string[];
  best_for: string[];
  production_ready: boolean;
  complexity: 'low' | 'medium' | 'high';
  data_requirements: 'small' | 'medium' | 'large';
  latency_sensitivity: 'low' | 'medium' | 'high';
  parameters: Record<string, any>;
  typical_sharpe: number;
  typical_max_dd: number;
  typical_win_rate: number;
  typical_turnover: number;
  max_position_size: number;
  max_leverage: number;
  stop_loss_pct: number;
  regime_affinity: Record<string, number>;
  paper_url?: string;
  implementation_notes?: string;
}

export interface StrategyFamily {
  id: string;
  name: string;
  count: number;
}

export interface TimeHorizon {
  id: string;
  name: string;
  count: number;
}

export interface StrategyRecommendation {
  id: string;
  name: string;
  family: string;
  description: string;
  typical_sharpe: number;
  typical_max_dd: number;
  complexity: string;
  regime_affinity: number;
}


// ============================================================================
// HARDCODED STRATEGY DATA (Fallback when backend is unavailable)
// ============================================================================

const HARDCODED_STRATEGIES: StrategySpec[] = [
  // TREND-FOLLOWING
  {
    id: 'ma_crossover',
    name: 'Moving Average Crossover with Volatility Filter',
    family: 'trend',
    horizon: 'daily',
    asset_classes: ['fx', 'equities', 'crypto'],
    description: 'Classic trend-following using fast/slow MA crossover with volatility filter',
    signal_logic: 'Fast MA (20) crosses above Slow MA (100) → Long; crosses below → Flat/Short',
    entry_rules: [
      'Fast MA crosses above Slow MA',
      'Rolling volatility > threshold',
      'Volume confirmation (optional)'
    ],
    exit_rules: [
      'Fast MA crosses below Slow MA',
      'ATR-based stop loss hit',
      'Time-based exit (max holding period)'
    ],
    risk_controls: [
      'ATR-based stop loss',
      'Max position size per asset',
      'Daily drawdown limit',
      'Volatility scaling'
    ],
    strengths: [
      'Simple and robust',
      'Works in trending markets',
      'Easy to interpret',
      'Battle-tested',
      'Low overfitting risk'
    ],
    weaknesses: [
      'Whipsaws in ranging markets',
      'Lagging indicator',
      'Poor in choppy conditions',
      'Transaction costs can erode profits'
    ],
    best_for: [
      'Trending markets',
      'Medium to long-term horizons',
      'Liquid instruments',
      'Infrastructure validation'
    ],
    production_ready: true,
    complexity: 'low',
    data_requirements: 'small',
    latency_sensitivity: 'low',
    parameters: {
      fast_period: 20,
      slow_period: 100,
      vol_lookback: 20,
      vol_threshold: 0.01,
      atr_multiplier: 2.0,
      max_holding_days: 30
    },
    typical_sharpe: 1.2,
    typical_max_dd: 15.0,
    typical_win_rate: 0.45,
    typical_turnover: 0.5,
    max_position_size: 0.20,
    max_leverage: 2.0,
    stop_loss_pct: 0.03,
    regime_affinity: {
      'LOW_VOL_TRENDING': 0.85,
      'MEDIUM_VOL_TRENDING': 0.75,
      'HIGH_VOL_RANGING': 0.20
    }
  },

  // MEAN REVERSION
  {
    id: 'zscore_reversion',
    name: 'Z-Score Mean Reversion',
    family: 'mean_reversion',
    horizon: 'intraday',
    asset_classes: ['fx', 'equities'],
    description: 'Trade mean reversion using z-score of price deviations',
    signal_logic: 'Enter when z-score exceeds ±2, exit at mean (z ≈ 0)',
    entry_rules: ['Z-score > +2 → Short', 'Z-score < -2 → Long', 'Only in range-bound regimes'],
    exit_rules: ['Z-score returns to 0', 'Time-based stop (N bars)', 'Stop loss at ±3 z-score'],
    risk_controls: ['Position sizing by volatility', 'Time-based stops', 'Regime filter', 'Max correlation exposure'],
    strengths: ['Works in ranging markets', 'High win rate', 'Quick trades', 'Statistical foundation'],
    weaknesses: ['Fails in trends', 'Transaction cost sensitive', 'Requires regime detection', 'Large losses in breakouts'],
    best_for: ['Range-bound markets', 'High-frequency trading', 'Pairs trading', 'Market making'],
    production_ready: true,
    complexity: 'medium',
    data_requirements: 'medium',
    latency_sensitivity: 'high',
    parameters: { lookback_period: 20, entry_threshold: 2.0, exit_threshold: 0.5, max_holding_bars: 50, stop_loss_zscore: 3.0 },
    typical_sharpe: 1.5,
    typical_max_dd: 8.0,
    typical_win_rate: 0.65,
    typical_turnover: 5.0,
    max_position_size: 0.15,
    max_leverage: 1.5,
    stop_loss_pct: 0.02,
    regime_affinity: { 'LOW_VOL_RANGING': 0.90, 'MEDIUM_VOL_RANGING': 0.70, 'HIGH_VOL_TRENDING': 0.10 }
  },
  // MOMENTUM
  {
    id: 'cross_sectional_momentum',
    name: 'Cross-Sectional Momentum Rotation',
    family: 'momentum',
    horizon: 'swing',
    asset_classes: ['equities', 'fx'],
    description: 'Rank assets by momentum, long top decile, short bottom decile',
    signal_logic: 'Rank by 3-12 month returns, rebalance monthly',
    entry_rules: ['Rank assets by momentum score', 'Long top 10% performers', 'Short bottom 10% (or flat)', 'Monthly rebalance'],
    exit_rules: ['Monthly rebalancing', 'Asset drops out of top/bottom decile', 'Stop loss per position'],
    risk_controls: ['Volatility scaling per asset', 'Sector caps', 'Turnover limits', 'Max concentration'],
    strengths: ['Strong academic backing', 'Works across asset classes', 'Diversified portfolio', 'Robust to market conditions'],
    weaknesses: ['High turnover', 'Crowded trade', 'Momentum crashes', 'Requires many assets'],
    best_for: ['Portfolio management', 'Multi-asset strategies', 'Medium-term trading', 'Institutional investors'],
    production_ready: true,
    complexity: 'medium',
    data_requirements: 'large',
    latency_sensitivity: 'low',
    parameters: { lookback_months: 12, skip_month: 1, top_pct: 10, bottom_pct: 10, rebalance_frequency: 'monthly', min_assets: 20 },
    typical_sharpe: 1.0,
    typical_max_dd: 20.0,
    typical_win_rate: 0.55,
    typical_turnover: 2.0,
    max_position_size: 0.10,
    max_leverage: 1.5,
    stop_loss_pct: 0.10,
    regime_affinity: { 'LOW_VOL_TRENDING': 0.75, 'MEDIUM_VOL_TRENDING': 0.80, 'CRISIS': 0.30 }
  },
  // VOLATILITY BREAKOUT
  {
    id: 'bollinger_breakout',
    name: 'Bollinger Band Breakout',
    family: 'volatility',
    horizon: 'intraday',
    asset_classes: ['fx', 'crypto', 'equities'],
    description: 'Trade breakouts from Bollinger Bands with volume confirmation',
    signal_logic: 'Enter when price closes above upper band, exit at mid-band',
    entry_rules: ['Price closes above upper band → Long', 'Price closes below lower band → Short', 'Volume > average', 'High volatility regime'],
    exit_rules: ['Price returns to mid-band', 'Opposite band touched', 'Time-based exit', 'Stop loss'],
    risk_controls: ['ATR-based position sizing', 'Volume filter', 'Regime filter', 'Max daily trades'],
    strengths: ['Captures volatility expansion', 'Works in breakout markets', 'Visual and intuitive', 'Adaptable to volatility'],
    weaknesses: ['False breakouts common', 'Whipsaws in ranging markets', 'Requires quick execution', 'Parameter sensitive'],
    best_for: ['Volatile markets', 'Event-driven trading', 'Crypto markets', 'News-based breakouts'],
    production_ready: true,
    complexity: 'low',
    data_requirements: 'small',
    latency_sensitivity: 'medium',
    parameters: { bb_period: 20, bb_std: 2.0, volume_threshold: 1.5, min_volatility: 0.015, max_holding_bars: 20 },
    typical_sharpe: 0.9,
    typical_max_dd: 12.0,
    typical_win_rate: 0.40,
    typical_turnover: 3.0,
    max_position_size: 0.15,
    max_leverage: 2.0,
    stop_loss_pct: 0.025,
    regime_affinity: { 'HIGH_VOL_RANGING': 0.70, 'MEDIUM_VOL_TRENDING': 0.60, 'LOW_VOL_RANGING': 0.30 }
  },

  // STATISTICAL ARBITRAGE
  {
    id: 'pairs_trading',
    name: 'Cointegration-Based Pairs Trading',
    family: 'statistical_arb',
    horizon: 'daily',
    asset_classes: ['equities', 'fx'],
    description: 'Trade mean-reverting spread between cointegrated pairs',
    signal_logic: 'Identify cointegrated pairs, trade spread z-score',
    entry_rules: ['Identify cointegrated pairs (ADF test)', 'Compute spread = price_A - beta * price_B', 'Enter when spread z-score > ±2', 'Long spread when z < -2, short when z > +2'],
    exit_rules: ['Spread returns to mean (z ≈ 0)', 'Cointegration breaks (rolling test)', 'Time-based stop', 'Stop loss at ±3 z-score'],
    risk_controls: ['Cointegration monitoring', 'Correlation breakdown detection', 'Position sizing by spread volatility', 'Max pairs per portfolio'],
    strengths: ['Market-neutral', 'Statistical foundation', 'Lower directional risk', 'Works in various markets'],
    weaknesses: ['Cointegration can break', 'Requires pair discovery', 'Transaction costs matter', 'Structural changes hurt'],
    best_for: ['Market-neutral strategies', 'Hedge fund strategies', 'Low-correlation portfolios', 'Statistical traders'],
    production_ready: true,
    complexity: 'high',
    data_requirements: 'large',
    latency_sensitivity: 'medium',
    parameters: { lookback_period: 60, cointegration_pvalue: 0.05, entry_zscore: 2.0, exit_zscore: 0.5, stop_zscore: 3.0, retest_frequency: 20 },
    typical_sharpe: 1.8,
    typical_max_dd: 6.0,
    typical_win_rate: 0.70,
    typical_turnover: 2.0,
    max_position_size: 0.20,
    max_leverage: 1.0,
    stop_loss_pct: 0.03,
    regime_affinity: { 'LOW_VOL_RANGING': 0.85, 'MEDIUM_VOL_RANGING': 0.75, 'CRISIS': 0.20 }
  },
  // REGIME-SWITCHING
  {
    id: 'adaptive_regime',
    name: 'Adaptive Regime-Switching Strategy',
    family: 'regime_switching',
    horizon: 'daily',
    asset_classes: ['fx', 'equities', 'crypto'],
    description: 'Switch between trend-following and mean-reversion based on regime',
    signal_logic: 'Detect regime, apply appropriate strategy',
    entry_rules: ['Detect current regime (HMM/Hurst/Volatility)', 'Trend regime → use trend-following', 'Range regime → use mean-reversion', 'Crisis regime → reduce exposure'],
    exit_rules: ['Regime change detected', 'Strategy-specific exits', 'Risk limit breached'],
    risk_controls: ['Regime confidence threshold', 'Gradual position transitions', 'Max leverage per regime', 'Drawdown limits'],
    strengths: ['Adapts to market conditions', 'Reduces whipsaws', 'Professional-grade', 'Robust across cycles'],
    weaknesses: ['Complex implementation', 'Regime detection lag', 'Requires multiple strategies', 'Higher turnover'],
    best_for: ['Sophisticated traders', 'Multi-strategy funds', 'Adaptive systems', 'Professional trading'],
    production_ready: true,
    complexity: 'high',
    data_requirements: 'large',
    latency_sensitivity: 'low',
    parameters: { regime_lookback: 60, hurst_threshold: 0.5, vol_threshold: 0.02, confidence_min: 0.7, transition_period: 5 },
    typical_sharpe: 1.6,
    typical_max_dd: 10.0,
    typical_win_rate: 0.60,
    typical_turnover: 1.5,
    max_position_size: 0.25,
    max_leverage: 2.0,
    stop_loss_pct: 0.04,
    regime_affinity: { 'ALL_REGIMES': 0.80 }
  },
  // SENTIMENT
  {
    id: 'news_sentiment',
    name: 'News Sentiment Reaction Strategy',
    family: 'sentiment',
    horizon: 'intraday',
    asset_classes: ['equities', 'crypto'],
    description: 'Trade on large sentiment shocks from news/social media',
    signal_logic: 'Trade only on large sentiment deltas',
    entry_rules: ['Ingest news sentiment score', 'Sentiment delta > threshold → Long', 'Sentiment delta < -threshold → Short', 'Latency critical'],
    exit_rules: ['Sentiment normalizes', 'Time-based exit (sentiment decays)', 'Opposite sentiment shock', 'Stop loss'],
    risk_controls: ['False positive filtering', 'Source reliability weighting', 'Max position per event', 'Rapid stop loss'],
    strengths: ['Captures information edge', 'Works with modern NLP', 'High potential alpha', 'Event-driven'],
    weaknesses: ['Latency critical', 'False positives common', 'Requires NLP infrastructure', 'Crowded in popular stocks'],
    best_for: ['High-frequency trading', 'Event-driven funds', 'Crypto markets', 'News-sensitive assets'],
    production_ready: false,
    complexity: 'high',
    data_requirements: 'large',
    latency_sensitivity: 'high',
    parameters: { sentiment_threshold: 0.7, decay_halflife: 30, min_source_reliability: 0.8, max_holding_minutes: 120 },
    typical_sharpe: 1.3,
    typical_max_dd: 8.0,
    typical_win_rate: 0.55,
    typical_turnover: 10.0,
    max_position_size: 0.10,
    max_leverage: 1.5,
    stop_loss_pct: 0.02,
    regime_affinity: { 'HIGH_VOL_RANGING': 0.75, 'MEDIUM_VOL_TRENDING': 0.60, 'CRISIS': 0.85 }
  },
  // EXECUTION
  {
    id: 'vwap_execution',
    name: 'VWAP Execution Strategy',
    family: 'execution',
    horizon: 'intraday',
    asset_classes: ['fx', 'equities', 'crypto'],
    description: 'Execute large orders with minimal market impact using VWAP',
    signal_logic: 'Split order into time slices, execute proportionally to volume',
    entry_rules: ['Divide order into N time slices', 'Execute proportionally to historical volume profile', 'Monitor real-time volume', 'Adjust execution rate dynamically'],
    exit_rules: ['Order fully executed', 'Time window expired', 'Market impact threshold exceeded'],
    risk_controls: ['Market impact monitoring', 'Slippage limits', 'Participation rate caps', 'Adverse selection detection'],
    strengths: ['Minimizes market impact', 'Industry standard', 'Predictable execution', 'Works for large orders'],
    weaknesses: ['Not alpha-generating', 'Vulnerable to gaming', 'Requires volume data', 'May miss opportunities'],
    best_for: ['Large order execution', 'Institutional trading', 'Minimizing slippage', 'Algorithmic execution'],
    production_ready: true,
    complexity: 'medium',
    data_requirements: 'medium',
    latency_sensitivity: 'medium',
    parameters: { num_slices: 20, participation_rate: 0.10, urgency: 0.5, max_deviation: 0.005 },
    typical_sharpe: 0.0,
    typical_max_dd: 0.0,
    typical_win_rate: 0.0,
    typical_turnover: 0.0,
    max_position_size: 1.0,
    max_leverage: 1.0,
    stop_loss_pct: 0.0,
    regime_affinity: {}
  },
];


/**
 * List all available strategies with optional filtering
 * Falls back to hardcoded data if backend is unavailable
 */
export async function listStrategies(filters?: {
  family?: string;
  horizon?: string;
  asset_class?: string;
  production_ready?: boolean;
}): Promise<StrategySpec[]> {
  try {
    const params = new URLSearchParams();
    if (filters?.family) params.append('family', filters.family);
    if (filters?.horizon) params.append('horizon', filters.horizon);
    if (filters?.asset_class) params.append('asset_class', filters.asset_class);
    if (filters?.production_ready !== undefined) {
      params.append('production_ready', filters.production_ready.toString());
    }
    
    const response = await intelligenceApi.get(`/strategies/list?${params.toString()}`);
    return (response as any).strategies as StrategySpec[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded strategies:', error);
    
    // Filter hardcoded strategies
    let strategies = [...HARDCODED_STRATEGIES];
    
    if (filters?.family) {
      strategies = strategies.filter(s => s.family === filters.family);
    }
    
    if (filters?.horizon) {
      strategies = strategies.filter(s => s.horizon === filters.horizon);
    }
    
    if (filters?.asset_class) {
      strategies = strategies.filter(s => s.asset_classes.includes(filters.asset_class!));
    }
    
    if (filters?.production_ready !== undefined) {
      strategies = strategies.filter(s => s.production_ready === filters.production_ready);
    }
    
    return strategies;
  }
}

/**
 * Get detailed information about a specific strategy
 * Falls back to hardcoded data if backend is unavailable
 */
export async function getStrategyDetails(strategyId: string): Promise<StrategySpec> {
  try {
    const response = await intelligenceApi.get(`/strategies/${strategyId}`);
    return response as StrategySpec;
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded strategy details:', error);
    
    const strategy = HARDCODED_STRATEGIES.find(s => s.id === strategyId);
    if (!strategy) {
      throw new Error(`Strategy ${strategyId} not found`);
    }
    
    return strategy;
  }
}

/**
 * Get recommended strategies for specific conditions
 * Falls back to hardcoded data if backend is unavailable
 */
export async function recommendStrategies(
  assetClass: string,
  horizon: string,
  regime?: string
): Promise<{
  asset_class: string;
  horizon: string;
  regime?: string;
  recommendations: StrategyRecommendation[];
  count: number;
}> {
  try {
    const params = new URLSearchParams();
    params.append('asset_class', assetClass);
    params.append('horizon', horizon);
    if (regime) params.append('regime', regime);
    
    const response = await intelligenceApi.get(`/strategies/recommend?${params.toString()}`);
    return response as {
      asset_class: string;
      horizon: string;
      regime?: string;
      recommendations: StrategyRecommendation[];
      count: number;
    };
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded recommendations:', error);
    
    // Filter strategies by asset class and horizon
    let strategies = HARDCODED_STRATEGIES.filter(s => 
      s.asset_classes.includes(assetClass) && s.horizon === horizon
    );
    
    // Filter by production ready
    strategies = strategies.filter(s => s.production_ready);
    
    // Sort by regime affinity if regime provided
    if (regime) {
      strategies = strategies.filter(s => regime in s.regime_affinity);
      strategies.sort((a, b) => {
        const affinityA = a.regime_affinity[regime] || 0;
        const affinityB = b.regime_affinity[regime] || 0;
        return affinityB - affinityA;
      });
    }
    
    // Convert to recommendation format
    const recommendations: StrategyRecommendation[] = strategies.map(s => ({
      id: s.id,
      name: s.name,
      family: s.family,
      description: s.description,
      typical_sharpe: s.typical_sharpe,
      typical_max_dd: s.typical_max_dd,
      complexity: s.complexity,
      regime_affinity: regime ? (s.regime_affinity[regime] || 0) : 0,
    }));
    
    return {
      asset_class: assetClass,
      horizon,
      regime,
      recommendations,
      count: recommendations.length,
    };
  }
}

/**
 * Get all strategy families
 * Falls back to hardcoded data if backend is unavailable
 */
export async function getStrategyFamilies(): Promise<StrategyFamily[]> {
  try {
    const response = await intelligenceApi.get('/strategies/families');
    return (response as any).families as StrategyFamily[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded families:', error);
    
    // Extract unique families from hardcoded strategies
    const familyMap = new Map<string, number>();
    
    HARDCODED_STRATEGIES.forEach(strategy => {
      const count = familyMap.get(strategy.family) || 0;
      familyMap.set(strategy.family, count + 1);
    });
    
    return Array.from(familyMap.entries()).map(([id, count]) => ({
      id,
      name: formatFamily(id),
      count,
    }));
  }
}

/**
 * Get all time horizons
 * Falls back to hardcoded data if backend is unavailable
 */
export async function getTimeHorizons(): Promise<TimeHorizon[]> {
  try {
    const response = await intelligenceApi.get('/strategies/horizons');
    return (response as any).horizons as TimeHorizon[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded horizons:', error);
    
    // Extract unique horizons from hardcoded strategies
    const horizonMap = new Map<string, number>();
    
    HARDCODED_STRATEGIES.forEach(strategy => {
      const count = horizonMap.get(strategy.horizon) || 0;
      horizonMap.set(strategy.horizon, count + 1);
    });
    
    return Array.from(horizonMap.entries()).map(([id, count]) => ({
      id,
      name: formatHorizon(id),
      count,
    }));
  }
}

/**
 * Helper function to format strategy family for display
 */
export function formatFamily(family: string): string {
  return family.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Helper function to format time horizon for display
 */
export function formatHorizon(horizon: string): string {
  return horizon.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Helper function to get complexity color
 */
export function getComplexityColor(complexity: string): string {
  switch (complexity) {
    case 'low': return '#00ff00';
    case 'medium': return '#ffff00';
    case 'high': return '#ff8c00';
    default: return '#666';
  }
}

/**
 * Helper function to get data requirements color
 */
export function getDataReqColor(dataReq: string): string {
  switch (dataReq) {
    case 'small': return '#00ff00';
    case 'medium': return '#ffff00';
    case 'large': return '#ff8c00';
    default: return '#666';
  }
}

/**
 * Helper function to get latency sensitivity color
 */
export function getLatencySensitivityColor(latency: string): string {
  switch (latency) {
    case 'low': return '#00ff00';
    case 'medium': return '#ffff00';
    case 'high': return '#ff0000';
    default: return '#666';
  }
}
