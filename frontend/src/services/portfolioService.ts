/**
 * Portfolio & Risk Management API Service
 */

import { intelligenceApi } from './api';

export interface PortfolioDefinition {
  id: string;
  name: string;
  baseCurrency: string;
  initialCapital: number;
  currentCapital: number;
  mode: 'LIVE' | 'PAPER' | 'SIMULATED';
  status: 'ACTIVE' | 'PAUSED' | 'CLOSED';
  createdAt: string;
  
  strategyAllocations: Array<{
    strategyId: string;
    weight: number;
    capitalAllocated: number;
  }>;
  
  allocationModel: 'EQUAL_WEIGHT' | 'VOL_TARGET' | 'RISK_PARITY' | 'MAX_DIVERSIFICATION' | 'KELLY' | 'CUSTOM';
  rebalanceFrequency: 'DAILY' | 'WEEKLY' | 'MONTHLY' | 'QUARTERLY';
  turnoverConstraint: number;
}

export interface RiskLimit {
  id: string;
  portfolioId: string;
  type: 'HARD' | 'SOFT';
  category: 'POSITION' | 'LEVERAGE' | 'SECTOR' | 'CORRELATION' | 'LOSS' | 'EXPOSURE';
  name: string;
  threshold: number;
  currentValue: number;
  breached: boolean;
  action: 'ALERT' | 'BLOCK' | 'REDUCE' | 'HALT';
}

export interface StressScenario {
  id: string;
  name: string;
  type: 'HISTORICAL' | 'HYPOTHETICAL';
  description: string;
  parameters: Record<string, any>;
  impact: {
    portfolioLoss: number;
    maxDrawdown: number;
    recoveryDays: number;
  };
}

// ============================================================================
// HARDCODED DATA (Fallback when backend is unavailable)
// ============================================================================

const HARDCODED_PORTFOLIOS: PortfolioDefinition[] = [
  {
    id: 'PORT-001',
    name: 'MAIN TRADING PORTFOLIO',
    baseCurrency: 'USD',
    initialCapital: 100000,
    currentCapital: 104127.89,
    mode: 'PAPER',
    status: 'ACTIVE',
    createdAt: '2024-01-01T00:00:00Z',
    strategyAllocations: [
      { strategyId: 'regime_switching', weight: 0.40, capitalAllocated: 40000 },
      { strategyId: 'momentum_rotation', weight: 0.30, capitalAllocated: 30000 },
      { strategyId: 'mean_reversion', weight: 0.20, capitalAllocated: 20000 },
      { strategyId: 'volatility_arb', weight: 0.10, capitalAllocated: 10000 },
    ],
    allocationModel: 'VOL_TARGET',
    rebalanceFrequency: 'WEEKLY',
    turnoverConstraint: 20,
  },
  {
    id: 'PORT-002',
    name: 'RESEARCH PORTFOLIO',
    baseCurrency: 'USD',
    initialCapital: 50000,
    currentCapital: 51234.56,
    mode: 'SIMULATED',
    status: 'ACTIVE',
    createdAt: '2024-01-10T00:00:00Z',
    strategyAllocations: [
      { strategyId: 'trend_following', weight: 0.50, capitalAllocated: 25000 },
      { strategyId: 'pairs_trading', weight: 0.50, capitalAllocated: 25000 },
    ],
    allocationModel: 'EQUAL_WEIGHT',
    rebalanceFrequency: 'MONTHLY',
    turnoverConstraint: 30,
  },
];

const HARDCODED_RISK_LIMITS: RiskLimit[] = [
  {
    id: 'LIMIT-001',
    portfolioId: 'PORT-001',
    type: 'HARD',
    category: 'POSITION',
    name: 'MAX POSITION SIZE',
    threshold: 15.0,
    currentValue: 10.5,
    breached: false,
    action: 'BLOCK',
  },
  {
    id: 'LIMIT-002',
    portfolioId: 'PORT-001',
    type: 'HARD',
    category: 'LEVERAGE',
    name: 'MAX LEVERAGE',
    threshold: 3.0,
    currentValue: 2.1,
    breached: false,
    action: 'BLOCK',
  },
  {
    id: 'LIMIT-003',
    portfolioId: 'PORT-001',
    type: 'HARD',
    category: 'LOSS',
    name: 'MAX DAILY LOSS',
    threshold: 5.0,
    currentValue: 1.2,
    breached: false,
    action: 'HALT',
  },
];

const HARDCODED_STRESS_SCENARIOS: StressScenario[] = [
  {
    id: 'STRESS-001',
    name: '2008 FINANCIAL CRISIS',
    type: 'HISTORICAL',
    description: 'Lehman Brothers collapse scenario',
    parameters: { startDate: '2008-09-15', duration: '90d', volMultiplier: 3.5 },
    impact: {
      portfolioLoss: -18500,
      maxDrawdown: 17.8,
      recoveryDays: 120,
    },
  },
  {
    id: 'STRESS-002',
    name: 'COVID-19 CRASH',
    type: 'HISTORICAL',
    description: 'March 2020 pandemic crash',
    parameters: { startDate: '2020-03-01', duration: '45d', volMultiplier: 4.0 },
    impact: {
      portfolioLoss: -22300,
      maxDrawdown: 21.4,
      recoveryDays: 95,
    },
  },
];

/**
 * List all portfolios
 */
export async function listPortfolios(): Promise<PortfolioDefinition[]> {
  try {
    const response = await intelligenceApi.get('/portfolios/list');
    return (response as any).portfolios as PortfolioDefinition[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded portfolios:', error);
    return [...HARDCODED_PORTFOLIOS];
  }
}

/**
 * Get portfolio by ID
 */
export async function getPortfolio(portfolioId: string): Promise<PortfolioDefinition> {
  try {
    const response = await intelligenceApi.get(`/portfolios/${portfolioId}`);
    return response as PortfolioDefinition;
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded portfolio:', error);
    const portfolio = HARDCODED_PORTFOLIOS.find(p => p.id === portfolioId);
    if (!portfolio) {
      throw new Error(`Portfolio ${portfolioId} not found`);
    }
    return portfolio;
  }
}

/**
 * Create new portfolio
 */
export async function createPortfolio(portfolio: Omit<PortfolioDefinition, 'id' | 'createdAt' | 'currentCapital'>): Promise<PortfolioDefinition> {
  try {
    const response = await intelligenceApi.post('/portfolios/create', portfolio);
    return response as PortfolioDefinition;
  } catch (error) {
    console.warn('Backend unavailable, returning mock portfolio:', error);
    return {
      ...portfolio,
      id: `PORT-${String(Math.floor(Math.random() * 1000)).padStart(3, '0')}`,
      createdAt: new Date().toISOString(),
      currentCapital: portfolio.initialCapital,
    };
  }
}

/**
 * Update portfolio
 */
export async function updatePortfolio(portfolioId: string, updates: Partial<PortfolioDefinition>): Promise<PortfolioDefinition> {
  try {
    const response = await intelligenceApi.put(`/portfolios/${portfolioId}`, updates);
    return response as PortfolioDefinition;
  } catch (error) {
    console.warn('Backend unavailable, returning mock update:', error);
    const portfolio = HARDCODED_PORTFOLIOS.find(p => p.id === portfolioId);
    if (!portfolio) {
      throw new Error(`Portfolio ${portfolioId} not found`);
    }
    return { ...portfolio, ...updates };
  }
}

/**
 * Delete portfolio
 */
export async function deletePortfolio(portfolioId: string): Promise<void> {
  try {
    await intelligenceApi.delete(`/portfolios/${portfolioId}`);
  } catch (error) {
    console.warn('Backend unavailable, mock delete:', error);
  }
}

/**
 * List risk limits for portfolio
 */
export async function listRiskLimits(portfolioId: string): Promise<RiskLimit[]> {
  try {
    const response = await intelligenceApi.get(`/portfolios/${portfolioId}/risk-limits`);
    return (response as any).limits as RiskLimit[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded risk limits:', error);
    return HARDCODED_RISK_LIMITS.filter(l => l.portfolioId === portfolioId);
  }
}

/**
 * Create risk limit
 */
export async function createRiskLimit(limit: Omit<RiskLimit, 'id' | 'currentValue' | 'breached'>): Promise<RiskLimit> {
  try {
    const response = await intelligenceApi.post('/risk-limits/create', limit);
    return response as RiskLimit;
  } catch (error) {
    console.warn('Backend unavailable, returning mock risk limit:', error);
    return {
      ...limit,
      id: `LIMIT-${String(Math.floor(Math.random() * 1000)).padStart(3, '0')}`,
      currentValue: 0,
      breached: false,
    };
  }
}

/**
 * Update risk limit
 */
export async function updateRiskLimit(limitId: string, updates: Partial<RiskLimit>): Promise<RiskLimit> {
  try {
    const response = await intelligenceApi.put(`/risk-limits/${limitId}`, updates);
    return response as RiskLimit;
  } catch (error) {
    console.warn('Backend unavailable, returning mock update:', error);
    const limit = HARDCODED_RISK_LIMITS.find(l => l.id === limitId);
    if (!limit) {
      throw new Error(`Risk limit ${limitId} not found`);
    }
    return { ...limit, ...updates };
  }
}

/**
 * Delete risk limit
 */
export async function deleteRiskLimit(limitId: string): Promise<void> {
  try {
    await intelligenceApi.delete(`/risk-limits/${limitId}`);
  } catch (error) {
    console.warn('Backend unavailable, mock delete:', error);
  }
}

/**
 * List stress scenarios
 */
export async function listStressScenarios(): Promise<StressScenario[]> {
  try {
    const response = await intelligenceApi.get('/stress-scenarios/list');
    return (response as any).scenarios as StressScenario[];
  } catch (error) {
    console.warn('Backend unavailable, using hardcoded stress scenarios:', error);
    return [...HARDCODED_STRESS_SCENARIOS];
  }
}

/**
 * Run stress test
 */
export async function runStressTest(portfolioId: string, scenarioId: string): Promise<any> {
  try {
    const response = await intelligenceApi.post('/stress-test/run', { portfolioId, scenarioId });
    return response;
  } catch (error) {
    console.warn('Backend unavailable, returning mock stress test result:', error);
    const scenario = HARDCODED_STRESS_SCENARIOS.find(s => s.id === scenarioId);
    return {
      portfolioId,
      scenarioId,
      impact: scenario?.impact || { portfolioLoss: -10000, maxDrawdown: 10.0, recoveryDays: 60 },
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Rebalance portfolio
 */
export async function rebalancePortfolio(portfolioId: string, newAllocations: Array<{ strategyId: string; weight: number }>): Promise<PortfolioDefinition> {
  try {
    const response = await intelligenceApi.post(`/portfolios/${portfolioId}/rebalance`, { allocations: newAllocations });
    return response as PortfolioDefinition;
  } catch (error) {
    console.warn('Backend unavailable, returning mock rebalance:', error);
    const portfolio = HARDCODED_PORTFOLIOS.find(p => p.id === portfolioId);
    if (!portfolio) {
      throw new Error(`Portfolio ${portfolioId} not found`);
    }
    return {
      ...portfolio,
      strategyAllocations: newAllocations.map(alloc => ({
        ...alloc,
        capitalAllocated: portfolio.currentCapital * alloc.weight,
      })),
    };
  }
}
