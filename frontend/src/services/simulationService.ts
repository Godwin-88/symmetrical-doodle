/**
 * Simulation & Backtesting API Service
 * 
 * Architecture:
 * - Rust simulation-engine (port 8002): Backtesting execution, performance-critical
 * - Python intelligence-layer (port 8000): Experiment tracking, analytics, ML
 */

import { intelligenceApi } from './api';

// Note: simulationApi would point to Rust simulation-engine on port 8002
// For now, using intelligenceApi with fallback until Rust HTTP endpoints are added

export interface Experiment {
  id: string;
  name: string;
  strategyId: string;
  strategyVersion: string;
  researcher: string;
  status: 'DRAFT' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  gitCommit?: string;
  hypothesis: string;
  tags: string[];
  
  assetUniverse: string[];
  dataSource: string;
  startDate: string;
  endDate: string;
  frequency: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  survivorshipBias: boolean;
  lookAheadBias: boolean;
  
  parameters: Record<string, any>;
  
  executionModel: {
    orderTypes: string[];
    slippageModel: 'FIXED' | 'VOLUME_BASED' | 'IMPACT_CURVE';
    slippageBps: number;
    transactionCostBps: number;
    latencyMs: number;
  };
  
  portfolioConfig: {
    positionSizing: 'FIXED' | 'VOL_TARGET' | 'KELLY';
    maxLeverage: number;
    maxPositionPct: number;
    longShortRatio: number;
  };
  
  riskConfig: {
    stopLossPct: number;
    takeProfitPct: number;
    maxDrawdownPct: number;
    dailyLossLimitPct: number;
  };
  
  results?: ExperimentResults;
}

export interface ExperimentResults {
  totalReturn: number;
  cagr: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  profitFactor: number;
  winRate: number;
  turnover: number;
  
  volatility: number;
  cvar95: number;
  tailRisk: number;
  
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
  avgHoldingPeriod: number;
  
  pnlByAsset: Record<string, number>;
  pnlByRegime: Record<string, number>;
  transactionCosts: number;
  
  equityCurve: Array<{ date: string; value: number }>;
  drawdownCurve: Array<{ date: string; value: number }>;
  rollingSharpe: Array<{ date: string; value: number }>;
}

export interface ScenarioTest {
  id: string;
  name: string;
  type: 'HISTORICAL' | 'HYPOTHETICAL';
  description: string;
  parameters: Record<string, any>;
  results?: {
    totalReturn: number;
    maxDrawdown: number;
    sharpe: number;
    recoveryDays: number;
  };
}

export interface ParameterSweep {
  id: string;
  experimentId: string;
  parameterName: string;
  values: number[];
  status: 'PENDING' | 'RUNNING' | 'COMPLETED';
  results: Array<{
    value: number;
    sharpe: number;
    totalReturn: number;
    maxDrawdown: number;
  }>;
}

// ============================================================================
// HARDCODED DATA
// ============================================================================

const HARDCODED_EXPERIMENTS: Experiment[] = [
  {
    id: 'EXP-001',
    name: 'REGIME SWITCHING V2.1 - EURUSD',
    strategyId: 'regime_switching',
    strategyVersion: '2.1.0',
    researcher: 'research_team',
    status: 'COMPLETED',
    createdAt: '2024-01-10T00:00:00Z',
    startedAt: '2024-01-10T00:05:00Z',
    completedAt: '2024-01-10T00:45:00Z',
    gitCommit: 'a3f5b2c',
    hypothesis: 'Regime detection improves risk-adjusted returns in trending markets',
    tags: ['regime', 'ml', 'eurusd'],
    assetUniverse: ['EURUSD'],
    dataSource: 'DERIV_HISTORICAL',
    startDate: '2020-01-01',
    endDate: '2023-12-31',
    frequency: '1h',
    survivorshipBias: false,
    lookAheadBias: false,
    parameters: {
      regimeWindow: 100,
      trendThreshold: 0.02,
      volMultiplier: 1.5,
    },
    executionModel: {
      orderTypes: ['MARKET', 'LIMIT'],
      slippageModel: 'VOLUME_BASED',
      slippageBps: 0.5,
      transactionCostBps: 0.2,
      latencyMs: 50,
    },
    portfolioConfig: {
      positionSizing: 'VOL_TARGET',
      maxLeverage: 2.0,
      maxPositionPct: 25,
      longShortRatio: 1.0,
    },
    riskConfig: {
      stopLossPct: 2.0,
      takeProfitPct: 4.0,
      maxDrawdownPct: 15.0,
      dailyLossLimitPct: 3.0,
    },
    results: {
      totalReturn: 42.5,
      cagr: 12.3,
      sharpe: 1.85,
      sortino: 2.34,
      maxDrawdown: 8.7,
      maxDrawdownDuration: 45,
      profitFactor: 1.92,
      winRate: 58.3,
      turnover: 145.2,
      volatility: 12.5,
      cvar95: 2.8,
      tailRisk: 0.15,
      totalTrades: 1247,
      winningTrades: 727,
      losingTrades: 520,
      avgWin: 125.50,
      avgLoss: -78.30,
      avgHoldingPeriod: 18.5,
      pnlByAsset: { EURUSD: 42500 },
      pnlByRegime: { TRENDING: 28000, RANGING: 14500 },
      transactionCosts: 2450,
      equityCurve: [],
      drawdownCurve: [],
      rollingSharpe: [],
    },
  },
  {
    id: 'EXP-002',
    name: 'MOMENTUM ROTATION - MULTI ASSET',
    strategyId: 'momentum_rotation',
    strategyVersion: '1.5.0',
    researcher: 'quant_team',
    status: 'COMPLETED',
    createdAt: '2024-01-12T00:00:00Z',
    completedAt: '2024-01-12T01:15:00Z',
    gitCommit: 'b4c6d7e',
    hypothesis: 'Cross-asset momentum signals improve diversification',
    tags: ['momentum', 'multi-asset'],
    assetUniverse: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    dataSource: 'DERIV_HISTORICAL',
    startDate: '2021-01-01',
    endDate: '2023-12-31',
    frequency: '4h',
    survivorshipBias: false,
    lookAheadBias: false,
    parameters: {
      lookbackPeriod: 20,
      momentumThreshold: 0.015,
      rebalanceFrequency: 'DAILY',
    },
    executionModel: {
      orderTypes: ['MARKET'],
      slippageModel: 'FIXED',
      slippageBps: 0.3,
      transactionCostBps: 0.15,
      latencyMs: 30,
    },
    portfolioConfig: {
      positionSizing: 'FIXED',
      maxLeverage: 1.5,
      maxPositionPct: 30,
      longShortRatio: 1.2,
    },
    riskConfig: {
      stopLossPct: 1.5,
      takeProfitPct: 3.0,
      maxDrawdownPct: 12.0,
      dailyLossLimitPct: 2.5,
    },
    results: {
      totalReturn: 35.8,
      cagr: 10.9,
      sharpe: 1.62,
      sortino: 2.01,
      maxDrawdown: 9.2,
      maxDrawdownDuration: 38,
      profitFactor: 1.78,
      winRate: 55.7,
      turnover: 198.5,
      volatility: 14.2,
      cvar95: 3.1,
      tailRisk: 0.18,
      totalTrades: 2103,
      winningTrades: 1171,
      losingTrades: 932,
      avgWin: 98.20,
      avgLoss: -65.40,
      avgHoldingPeriod: 12.3,
      pnlByAsset: { EURUSD: 12000, GBPUSD: 10500, USDJPY: 8300, AUDUSD: 5000 },
      pnlByRegime: { TRENDING: 22000, RANGING: 13800 },
      transactionCosts: 3180,
      equityCurve: [],
      drawdownCurve: [],
      rollingSharpe: [],
    },
  },
  {
    id: 'EXP-003',
    name: 'MEAN REVERSION - INTRADAY',
    strategyId: 'mean_reversion',
    strategyVersion: '3.0.0',
    researcher: 'hft_team',
    status: 'RUNNING',
    createdAt: '2024-01-15T10:00:00Z',
    startedAt: '2024-01-15T10:05:00Z',
    gitCommit: 'c5d7e8f',
    hypothesis: 'Intraday mean reversion with vol filtering reduces drawdowns',
    tags: ['mean-reversion', 'intraday', 'vol-filter'],
    assetUniverse: ['EURUSD', 'GBPUSD'],
    dataSource: 'DERIV_HISTORICAL',
    startDate: '2022-01-01',
    endDate: '2023-12-31',
    frequency: '5m',
    survivorshipBias: false,
    lookAheadBias: false,
    parameters: {
      zScoreThreshold: 2.0,
      lookbackWindow: 50,
      volFilter: true,
    },
    executionModel: {
      orderTypes: ['LIMIT'],
      slippageModel: 'IMPACT_CURVE',
      slippageBps: 0.4,
      transactionCostBps: 0.1,
      latencyMs: 20,
    },
    portfolioConfig: {
      positionSizing: 'VOL_TARGET',
      maxLeverage: 3.0,
      maxPositionPct: 20,
      longShortRatio: 1.0,
    },
    riskConfig: {
      stopLossPct: 1.0,
      takeProfitPct: 2.0,
      maxDrawdownPct: 10.0,
      dailyLossLimitPct: 2.0,
    },
  },
  {
    id: 'EXP-004',
    name: 'VOL ARB - OPTIONS STRATEGY',
    strategyId: 'volatility_arb',
    strategyVersion: '1.0.0',
    researcher: 'derivatives_team',
    status: 'FAILED',
    createdAt: '2024-01-14T00:00:00Z',
    startedAt: '2024-01-14T00:05:00Z',
    completedAt: '2024-01-14T00:15:00Z',
    gitCommit: 'd6e8f9a',
    hypothesis: 'Volatility arbitrage between implied and realized vol',
    tags: ['volatility', 'options', 'arbitrage'],
    assetUniverse: ['EURUSD'],
    dataSource: 'DERIV_OPTIONS',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    frequency: '1h',
    survivorshipBias: false,
    lookAheadBias: false,
    parameters: {
      volWindow: 30,
      entryThreshold: 0.05,
      exitThreshold: 0.02,
    },
    executionModel: {
      orderTypes: ['MARKET'],
      slippageModel: 'FIXED',
      slippageBps: 1.0,
      transactionCostBps: 0.5,
      latencyMs: 100,
    },
    portfolioConfig: {
      positionSizing: 'FIXED',
      maxLeverage: 1.0,
      maxPositionPct: 50,
      longShortRatio: 1.0,
    },
    riskConfig: {
      stopLossPct: 3.0,
      takeProfitPct: 5.0,
      maxDrawdownPct: 20.0,
      dailyLossLimitPct: 5.0,
    },
  },
];

const HARDCODED_SCENARIOS: ScenarioTest[] = [
  {
    id: 'SCENARIO-001',
    name: '2008 FINANCIAL CRISIS',
    type: 'HISTORICAL',
    description: 'Lehman Brothers collapse - extreme volatility and correlation breakdown',
    parameters: {
      startDate: '2008-09-15',
      endDate: '2008-12-31',
      volMultiplier: 3.5,
      correlationShock: 0.9,
    },
    results: {
      totalReturn: -18.5,
      maxDrawdown: 22.3,
      sharpe: -0.85,
      recoveryDays: 120,
    },
  },
  {
    id: 'SCENARIO-002',
    name: 'COVID-19 CRASH',
    type: 'HISTORICAL',
    description: 'March 2020 pandemic crash - rapid drawdown and recovery',
    parameters: {
      startDate: '2020-03-01',
      endDate: '2020-06-30',
      volMultiplier: 4.0,
      liquidityShock: 0.5,
    },
    results: {
      totalReturn: -15.2,
      maxDrawdown: 19.8,
      sharpe: -0.92,
      recoveryDays: 95,
    },
  },
  {
    id: 'SCENARIO-003',
    name: 'FLASH CRASH',
    type: 'HISTORICAL',
    description: 'May 2010 flash crash - extreme intraday volatility',
    parameters: {
      startDate: '2010-05-06',
      endDate: '2010-05-07',
      volMultiplier: 10.0,
      liquidityDrain: 0.8,
    },
    results: {
      totalReturn: -8.5,
      maxDrawdown: 12.1,
      sharpe: -1.25,
      recoveryDays: 15,
    },
  },
  {
    id: 'SCENARIO-004',
    name: 'VOLATILITY SPIKE +50%',
    type: 'HYPOTHETICAL',
    description: 'Sudden 50% increase in market volatility',
    parameters: {
      volIncrease: 0.50,
      duration: '30d',
      correlationBreakdown: true,
    },
    results: {
      totalReturn: -12.3,
      maxDrawdown: 15.7,
      sharpe: -0.78,
      recoveryDays: 60,
    },
  },
];

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * List all experiments
 * Uses Python intelligence-layer for experiment tracking
 */
export async function listExperiments(filters?: {
  status?: string;
  strategyId?: string;
  researcher?: string;
}): Promise<Experiment[]> {
  try {
    const params = new URLSearchParams();
    if (filters?.status) params.append('status', filters.status);
    if (filters?.strategyId) params.append('strategy_id', filters.strategyId);
    if (filters?.researcher) params.append('researcher', filters.researcher);
    
    const response = await intelligenceApi.get(`/simulation/experiments?${params.toString()}`);
    return (response as any).experiments as Experiment[];
  } catch (error) {
    console.warn('Intelligence layer unavailable, using hardcoded experiments:', error);
    let experiments = [...HARDCODED_EXPERIMENTS];
    
    if (filters?.status) {
      experiments = experiments.filter(e => e.status === filters.status);
    }
    if (filters?.strategyId) {
      experiments = experiments.filter(e => e.strategyId === filters.strategyId);
    }
    if (filters?.researcher) {
      experiments = experiments.filter(e => e.researcher === filters.researcher);
    }
    
    return experiments;
  }
}

/**
 * Get experiment by ID
 */
export async function getExperiment(experimentId: string): Promise<Experiment> {
  try {
    const response = await intelligenceApi.get(`/simulation/experiments/${experimentId}`);
    return response as Experiment;
  } catch (error) {
    console.warn('Intelligence layer unavailable, using hardcoded experiment:', error);
    const experiment = HARDCODED_EXPERIMENTS.find(e => e.id === experimentId);
    if (!experiment) {
      throw new Error(`Experiment ${experimentId} not found`);
    }
    return experiment;
  }
}

/**
 * Create new experiment
 */
export async function createExperiment(experiment: Omit<Experiment, 'id' | 'createdAt' | 'status'>): Promise<Experiment> {
  try {
    const response = await intelligenceApi.post('/simulation/experiments/create', experiment);
    return response as Experiment;
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock experiment:', error);
    return {
      ...experiment,
      id: `EXP-${String(Math.floor(Math.random() * 1000)).padStart(3, '0')}`,
      createdAt: new Date().toISOString(),
      status: 'DRAFT',
    };
  }
}

/**
 * Update experiment
 */
export async function updateExperiment(experimentId: string, updates: Partial<Experiment>): Promise<Experiment> {
  try {
    const response = await intelligenceApi.put(`/simulation/experiments/${experimentId}`, updates);
    return response as Experiment;
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock update:', error);
    const experiment = HARDCODED_EXPERIMENTS.find(e => e.id === experimentId);
    if (!experiment) {
      throw new Error(`Experiment ${experimentId} not found`);
    }
    return { ...experiment, ...updates };
  }
}

/**
 * Delete experiment
 */
export async function deleteExperiment(experimentId: string): Promise<void> {
  try {
    await intelligenceApi.delete(`/simulation/experiments/${experimentId}`);
  } catch (error) {
    console.warn('Intelligence layer unavailable, mock delete:', error);
  }
}

/**
 * Clone experiment
 */
export async function cloneExperiment(experimentId: string, newName: string): Promise<Experiment> {
  try {
    const response = await intelligenceApi.post(`/simulation/experiments/${experimentId}/clone`, { name: newName });
    return response as Experiment;
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock clone:', error);
    const experiment = HARDCODED_EXPERIMENTS.find(e => e.id === experimentId);
    if (!experiment) {
      throw new Error(`Experiment ${experimentId} not found`);
    }
    return {
      ...experiment,
      id: `EXP-${String(Math.floor(Math.random() * 1000)).padStart(3, '0')}`,
      name: newName,
      status: 'DRAFT',
      createdAt: new Date().toISOString(),
      results: undefined,
    };
  }
}

/**
 * Run experiment
 * Uses Rust simulation-engine for backtesting execution
 */
export async function runExperiment(experimentId: string): Promise<Experiment> {
  try {
    // TODO: Route to Rust simulation-engine on port 8002
    const response = await intelligenceApi.post(`/simulation/experiments/${experimentId}/run`);
    return response as Experiment;
  } catch (error) {
    console.warn('Simulation engine unavailable, returning mock running experiment:', error);
    const experiment = HARDCODED_EXPERIMENTS.find(e => e.id === experimentId);
    if (!experiment) {
      throw new Error(`Experiment ${experimentId} not found`);
    }
    return {
      ...experiment,
      status: 'RUNNING',
      startedAt: new Date().toISOString(),
    };
  }
}

/**
 * Stop experiment
 */
export async function stopExperiment(experimentId: string): Promise<Experiment> {
  try {
    const response = await intelligenceApi.post(`/simulation/experiments/${experimentId}/stop`);
    return response as Experiment;
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock stopped experiment:', error);
    const experiment = HARDCODED_EXPERIMENTS.find(e => e.id === experimentId);
    if (!experiment) {
      throw new Error(`Experiment ${experimentId} not found`);
    }
    return {
      ...experiment,
      status: 'CANCELLED',
    };
  }
}

/**
 * List scenario tests
 */
export async function listScenarios(): Promise<ScenarioTest[]> {
  try {
    const response = await intelligenceApi.get('/simulation/scenarios');
    return (response as any).scenarios as ScenarioTest[];
  } catch (error) {
    console.warn('Intelligence layer unavailable, using hardcoded scenarios:', error);
    return [...HARDCODED_SCENARIOS];
  }
}

/**
 * Run scenario test
 */
export async function runScenario(experimentId: string, scenarioId: string): Promise<ScenarioTest> {
  try {
    const response = await intelligenceApi.post(`/simulation/experiments/${experimentId}/scenario/${scenarioId}`);
    return response as ScenarioTest;
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock scenario result:', error);
    const scenario = HARDCODED_SCENARIOS.find(s => s.id === scenarioId);
    if (!scenario) {
      throw new Error(`Scenario ${scenarioId} not found`);
    }
    return scenario;
  }
}

/**
 * Run parameter sweep
 */
export async function runParameterSweep(experimentId: string, parameterName: string, values: number[]): Promise<ParameterSweep> {
  try {
    const response = await intelligenceApi.post(`/simulation/experiments/${experimentId}/parameter-sweep`, {
      parameterName,
      values,
    });
    return response as ParameterSweep;
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock parameter sweep:', error);
    return {
      id: `SWEEP-${Math.floor(Math.random() * 1000)}`,
      experimentId,
      parameterName,
      values,
      status: 'RUNNING',
      results: values.map(v => ({
        value: v,
        sharpe: 1.5 + Math.random() * 0.5,
        totalReturn: 30 + Math.random() * 20,
        maxDrawdown: 8 + Math.random() * 5,
      })),
    };
  }
}

/**
 * Compare experiments
 */
export async function compareExperiments(experimentIds: string[]): Promise<{
  experiments: Experiment[];
  comparison: Record<string, any>;
}> {
  try {
    const response = await intelligenceApi.post('/simulation/experiments/compare', { experimentIds });
    return response as { experiments: Experiment[]; comparison: Record<string, any> };
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock comparison:', error);
    const experiments = HARDCODED_EXPERIMENTS.filter(e => experimentIds.includes(e.id));
    return {
      experiments,
      comparison: {
        bestSharpe: experiments[0]?.id,
        bestReturn: experiments[0]?.id,
        lowestDrawdown: experiments[1]?.id,
      },
    };
  }
}
