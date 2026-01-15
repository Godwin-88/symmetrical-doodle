import { useState, useEffect } from 'react';
import { useTradingStore } from '@/app/store/tradingStore';
import { listStrategies, type StrategySpec } from '../../services/strategiesService';
import {
  CreatePortfolioModal,
  EditPortfolioModal,
  RiskLimitModal,
  StressTestModal,
  AllocationModal,
  AttributionModal
} from './PortfolioModals';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface PortfolioDefinition {
  id: string;
  name: string;
  baseCurrency: string;
  initialCapital: number;
  currentCapital: number;
  mode: 'LIVE' | 'PAPER' | 'SIMULATED';
  status: 'ACTIVE' | 'PAUSED' | 'CLOSED';
  createdAt: string;
  
  // Strategy Allocations
  strategyAllocations: Array<{
    strategyId: string;
    weight: number;
    capitalAllocated: number;
  }>;
  
  // Capital Allocation Model
  allocationModel: 'EQUAL_WEIGHT' | 'VOL_TARGET' | 'RISK_PARITY' | 'MAX_DIVERSIFICATION' | 'KELLY' | 'CUSTOM';
  rebalanceFrequency: 'DAILY' | 'WEEKLY' | 'MONTHLY' | 'QUARTERLY';
  turnoverConstraint: number; // max % per rebalance
}

export interface Position {
  symbol: string;
  size: number;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  holdingPeriod: number; // hours
  strategyId: string;
  exposure: number; // % of portfolio
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

export interface PerformanceAttribution {
  byStrategy: Record<string, number>;
  byAsset: Record<string, number>;
  byFactor: Record<string, number>;
  byRegime: Record<string, number>;
}

export function Portfolio() {
  const { positions: storePositions, netPnl, riskUtilization } = useTradingStore();
  
  // State
  const [portfolios, setPortfolios] = useState<PortfolioDefinition[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<PortfolioDefinition | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [riskLimits, setRiskLimits] = useState<RiskLimit[]>([]);
  const [stressScenarios, setStressScenarios] = useState<StressScenario[]>([]);
  const [availableStrategies, setAvailableStrategies] = useState<StrategySpec[]>([]);
  
  // Modal states
  const [showCreatePortfolioModal, setShowCreatePortfolioModal] = useState(false);
  const [showEditPortfolioModal, setShowEditPortfolioModal] = useState(false);
  const [showRiskLimitModal, setShowRiskLimitModal] = useState(false);
  const [showStressTestModal, setShowStressTestModal] = useState(false);
  const [showAllocationModal, setShowAllocationModal] = useState(false);
  const [showAttributionModal, setShowAttributionModal] = useState(false);
  
  // View states
  const [activeView, setActiveView] = useState<'POSITIONS' | 'EXPOSURE' | 'RISK' | 'ATTRIBUTION'>('POSITIONS');
  
  // Initialize data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const strategies = await listStrategies();
        setAvailableStrategies(strategies);
      } catch (err) {
        console.warn('Failed to fetch strategies:', err);
      }
    };
    
    fetchData();
    
    // Initialize mock portfolios
    setPortfolios([
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
    ]);
    
    // Initialize mock positions
    setPositions([
      {
        symbol: 'EURUSD',
        size: 100000,
        side: 'LONG',
        entryPrice: 1.0845,
        currentPrice: 1.0894,
        unrealizedPnl: 490.00,
        realizedPnl: 1234.56,
        holdingPeriod: 12.5,
        strategyId: 'regime_switching',
        exposure: 10.5,
      },
      {
        symbol: 'GBPUSD',
        size: 75000,
        side: 'LONG',
        entryPrice: 1.2650,
        currentPrice: 1.2698,
        unrealizedPnl: 360.00,
        realizedPnl: 890.12,
        holdingPeriod: 8.2,
        strategyId: 'momentum_rotation',
        exposure: 9.5,
      },
      {
        symbol: 'USDJPY',
        size: 50000,
        side: 'SHORT',
        entryPrice: 148.50,
        currentPrice: 148.20,
        unrealizedPnl: 150.00,
        realizedPnl: -234.78,
        holdingPeriod: 24.1,
        strategyId: 'mean_reversion',
        exposure: 7.4,
      },
      {
        symbol: 'AUDUSD',
        size: 60000,
        side: 'LONG',
        entryPrice: 0.6580,
        currentPrice: 0.6612,
        unrealizedPnl: 192.00,
        realizedPnl: 456.89,
        holdingPeriod: 6.8,
        strategyId: 'volatility_arb',
        exposure: 6.3,
      },
    ]);
    
    // Initialize risk limits
    setRiskLimits([
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
      {
        id: 'LIMIT-004',
        portfolioId: 'PORT-001',
        type: 'SOFT',
        category: 'EXPOSURE',
        name: 'NET EXPOSURE WARNING',
        threshold: 60.0,
        currentValue: 33.7,
        breached: false,
        action: 'ALERT',
      },
      {
        id: 'LIMIT-005',
        portfolioId: 'PORT-001',
        type: 'HARD',
        category: 'CORRELATION',
        name: 'MAX CORRELATION',
        threshold: 0.85,
        currentValue: 0.72,
        breached: false,
        action: 'REDUCE',
      },
    ]);
    
    // Initialize stress scenarios
    setStressScenarios([
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
      {
        id: 'STRESS-003',
        name: 'VOLATILITY SPIKE +50%',
        type: 'HYPOTHETICAL',
        description: 'Sudden 50% increase in market volatility',
        parameters: { volIncrease: 0.50, correlationBreakdown: true },
        impact: {
          portfolioLoss: -8900,
          maxDrawdown: 8.5,
          recoveryDays: 30,
        },
      },
      {
        id: 'STRESS-004',
        name: 'LIQUIDITY DROUGHT',
        type: 'HYPOTHETICAL',
        description: 'Market liquidity drops 80%',
        parameters: { liquidityReduction: 0.80, slippageMultiplier: 5.0 },
        impact: {
          portfolioLoss: -12400,
          maxDrawdown: 11.9,
          recoveryDays: 60,
        },
      },
    ]);
  }, []);
  
  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPnl = (pnl: number) => {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}${formatNumber(pnl)}`;
  };
  
  // CRUD Operations
  const createPortfolio = (portfolio: Omit<PortfolioDefinition, 'id' | 'createdAt' | 'currentCapital'>) => {
    const newPortfolio: PortfolioDefinition = {
      ...portfolio,
      id: `PORT-${String(portfolios.length + 1).padStart(3, '0')}`,
      createdAt: new Date().toISOString(),
      currentCapital: portfolio.initialCapital,
    };
    setPortfolios([...portfolios, newPortfolio]);
    return newPortfolio;
  };
  
  const updatePortfolio = (id: string, updates: Partial<PortfolioDefinition>) => {
    setPortfolios(portfolios.map(p => p.id === id ? { ...p, ...updates } : p));
  };
  
  const deletePortfolio = (id: string) => {
    setPortfolios(portfolios.filter(p => p.id !== id));
    if (selectedPortfolio?.id === id) {
      setSelectedPortfolio(null);
    }
  };
  
  const createRiskLimit = (limit: Omit<RiskLimit, 'id' | 'currentValue' | 'breached'>) => {
    const newLimit: RiskLimit = {
      ...limit,
      id: `LIMIT-${String(riskLimits.length + 1).padStart(3, '0')}`,
      currentValue: 0,
      breached: false,
    };
    setRiskLimits([...riskLimits, newLimit]);
  };
  
  const updateRiskLimit = (id: string, updates: Partial<RiskLimit>) => {
    setRiskLimits(riskLimits.map(l => l.id === id ? { ...l, ...updates } : l));
  };
  
  const deleteRiskLimit = (id: string) => {
    setRiskLimits(riskLimits.filter(l => l.id !== id));
  };
  
  const runStressTest = (scenarioId: string) => {
    const scenario = stressScenarios.find(s => s.id === scenarioId);
    if (scenario) {
      alert(`Running stress test: ${scenario.name}\n\nEstimated Impact:\nLoss: $${formatNumber(Math.abs(scenario.impact.portfolioLoss))}\nMax DD: ${formatNumber(scenario.impact.maxDrawdown, 1)}%\nRecovery: ${scenario.impact.recoveryDays} days`);
    }
  };
  
  // Calculate portfolio metrics
  const calculateMetrics = () => {
    if (!selectedPortfolio) return null;
    
    const totalUnrealizedPnl = positions.reduce((sum, p) => sum + p.unrealizedPnl, 0);
    const totalRealizedPnl = positions.reduce((sum, p) => sum + p.realizedPnl, 0);
    const totalPnl = totalUnrealizedPnl + totalRealizedPnl;
    
    const grossExposure = positions.reduce((sum, p) => sum + Math.abs(p.size * p.currentPrice), 0);
    const netExposure = positions.reduce((sum, p) => {
      const value = p.size * p.currentPrice;
      return sum + (p.side === 'LONG' ? value : -value);
    }, 0);
    
    const leverage = grossExposure / selectedPortfolio.currentCapital;
    
    return {
      totalEquity: selectedPortfolio.currentCapital,
      totalPnl,
      unrealizedPnl: totalUnrealizedPnl,
      realizedPnl: totalRealizedPnl,
      grossExposure,
      netExposure,
      leverage,
      roi: (totalPnl / selectedPortfolio.initialCapital) * 100,
    };
  };
  
  const metrics = calculateMetrics();
  
  // Get breached limits
  const breachedLimits = riskLimits.filter(l => l.breached && l.portfolioId === selectedPortfolio?.id);
  const hardBreaches = breachedLimits.filter(l => l.type === 'HARD');
  const softBreaches = breachedLimits.filter(l => l.type === 'SOFT');

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel - Portfolio List */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          {/* Header */}
          <div className="mb-4">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">PORTFOLIO REGISTRY</div>
            
            {/* Summary Stats */}
            <div className="grid grid-cols-2 gap-2 mb-3">
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">TOTAL PORTFOLIOS</div>
                <div className="text-[#fff] text-[14px]">{portfolios.length}</div>
              </div>
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">ACTIVE</div>
                <div className="text-[#00ff00] text-[14px]">{portfolios.filter(p => p.status === 'ACTIVE').length}</div>
              </div>
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">TOTAL AUM</div>
                <div className="text-[#fff] text-[14px]">
                  ${formatNumber(portfolios.reduce((sum, p) => sum + p.currentCapital, 0), 0)}
                </div>
              </div>
              <div className="border border-[#444] p-2 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">BREACHES</div>
                <div className={`text-[14px] ${hardBreaches.length > 0 ? 'text-[#ff0000]' : 'text-[#00ff00]'}`}>
                  {hardBreaches.length}
                </div>
              </div>
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="space-y-2 mb-4">
            <button
              onClick={() => setShowCreatePortfolioModal(true)}
              className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
            >
              + NEW PORTFOLIO
            </button>
            <button
              onClick={() => setShowRiskLimitModal(true)}
              className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
            >
              CONFIGURE RISK LIMITS
            </button>
            <button
              onClick={() => setShowStressTestModal(true)}
              className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors"
            >
              STRESS TESTING
            </button>
          </div>
          
          {/* Portfolio List */}
          <div className="space-y-2">
            {portfolios.map((portfolio) => (
              <div
                key={portfolio.id}
                onClick={() => setSelectedPortfolio(portfolio)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedPortfolio?.id === portfolio.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="text-[#00ff00] text-[10px] font-bold">{portfolio.name}</div>
                    <div className="text-[#666] text-[9px]">{portfolio.id}</div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span className={`
                      text-[8px] px-1
                      ${portfolio.status === 'ACTIVE' ? 'text-[#00ff00]' : ''}
                      ${portfolio.status === 'PAUSED' ? 'text-[#ffff00]' : ''}
                      ${portfolio.status === 'CLOSED' ? 'text-[#666]' : ''}
                    `}>
                      {portfolio.status}
                    </span>
                    <span className={`
                      text-[8px] px-1
                      ${portfolio.mode === 'LIVE' ? 'text-[#ff0000]' : ''}
                      ${portfolio.mode === 'PAPER' ? 'text-[#ffff00]' : ''}
                      ${portfolio.mode === 'SIMULATED' ? 'text-[#666]' : ''}
                    `}>
                      {portfolio.mode}
                    </span>
                  </div>
                </div>
                
                <div className="space-y-1 text-[9px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">CAPITAL:</span>
                    <span className="text-[#fff]">${formatNumber(portfolio.currentCapital, 0)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">P&L:</span>
                    <span className={portfolio.currentCapital >= portfolio.initialCapital ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {formatPnl(portfolio.currentCapital - portfolio.initialCapital)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">STRATEGIES:</span>
                    <span className="text-[#fff]">{portfolio.strategyAllocations.length}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Center Panel - Portfolio Details */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="text-[#ff8c00] text-sm tracking-wider">
            {selectedPortfolio ? `PORTFOLIO: ${selectedPortfolio.name}` : 'PORTFOLIO & RISK MANAGEMENT'}
          </div>
        </div>

        {selectedPortfolio && metrics ? (
          <div className="space-y-6">
            {/* Portfolio Summary */}
            <div className="grid grid-cols-4 gap-4">
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">TOTAL EQUITY</div>
                <div className="text-[#00ff00] text-[14px]">${formatNumber(metrics.totalEquity)}</div>
                <div className="text-[#666] text-[9px] mt-1">
                  INITIAL: ${formatNumber(selectedPortfolio.initialCapital, 0)}
                </div>
              </div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">TOTAL P&L</div>
                <div className={`text-[14px] ${metrics.totalPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                  {formatPnl(metrics.totalPnl)}
                </div>
                <div className={`text-[9px] mt-1 ${metrics.roi >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                  {formatPnl(metrics.roi)}% ROI
                </div>
              </div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">LEVERAGE</div>
                <div className={`text-[14px] ${metrics.leverage > 2 ? 'text-[#ffff00]' : 'text-[#00ff00]'}`}>
                  {formatNumber(metrics.leverage)}x
                </div>
                <div className="text-[#666] text-[9px] mt-1">
                  GROSS: ${formatNumber(metrics.grossExposure, 0)}
                </div>
              </div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">NET EXPOSURE</div>
                <div className="text-[#00ff00] text-[14px]">${formatNumber(metrics.netExposure, 0)}</div>
                <div className="text-[#666] text-[9px] mt-1">
                  {formatNumber((metrics.netExposure / metrics.totalEquity) * 100, 1)}% OF EQUITY
                </div>
              </div>
            </div>

            {/* Risk Alerts */}
            {(hardBreaches.length > 0 || softBreaches.length > 0) && (
              <div className="border-2 border-[#ff0000] bg-[#1a0000] p-3">
                <div className="text-[#ff0000] text-[10px] font-bold mb-2">⚠ RISK LIMIT BREACHES</div>
                <div className="space-y-1">
                  {hardBreaches.map(limit => (
                    <div key={limit.id} className="text-[#ff0000] text-[9px]">
                      ● HARD: {limit.name} - {formatNumber(limit.currentValue, 1)} / {formatNumber(limit.threshold, 1)} - ACTION: {limit.action}
                    </div>
                  ))}
                  {softBreaches.map(limit => (
                    <div key={limit.id} className="text-[#ffff00] text-[9px]">
                      ● SOFT: {limit.name} - {formatNumber(limit.currentValue, 1)} / {formatNumber(limit.threshold, 1)} - ACTION: {limit.action}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* View Tabs */}
            <div className="flex gap-2 border-b border-[#444]">
              {(['POSITIONS', 'EXPOSURE', 'RISK', 'ATTRIBUTION'] as const).map(view => (
                <button
                  key={view}
                  onClick={() => setActiveView(view)}
                  className={`
                    px-4 py-2 text-[10px] transition-colors
                    ${activeView === view
                      ? 'border-b-2 border-[#ff8c00] text-[#ff8c00]'
                      : 'text-[#666] hover:text-[#fff]'
                    }
                  `}
                >
                  {view}
                </button>
              ))}
            </div>

            {/* Positions View */}
            {activeView === 'POSITIONS' && (
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">CURRENT POSITIONS</div>
                <div className="border border-[#444] bg-[#0a0a0a]">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                        <th className="px-3 py-2 text-left border-b border-[#444]">SYMBOL</th>
                        <th className="px-3 py-2 text-center border-b border-[#444]">SIDE</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">SIZE</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">ENTRY</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">CURRENT</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">UNREALIZED P&L</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">EXPOSURE</th>
                        <th className="px-3 py-2 text-left border-b border-[#444]">STRATEGY</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((pos, idx) => (
                        <tr key={idx} className="border-b border-[#222]">
                          <td className="px-3 py-2 text-[#00ff00]">{pos.symbol}</td>
                          <td className="px-3 py-2 text-center">
                            <span className={pos.side === 'LONG' ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                              {pos.side}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(pos.size, 0)}</td>
                          <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(pos.entryPrice, 4)}</td>
                          <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(pos.currentPrice, 4)}</td>
                          <td className={`px-3 py-2 text-right ${pos.unrealizedPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                            {formatPnl(pos.unrealizedPnl)}
                          </td>
                          <td className="px-3 py-2 text-right text-[#ffff00]">{formatNumber(pos.exposure, 1)}%</td>
                          <td className="px-3 py-2 text-[#666] text-[9px]">{pos.strategyId}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Exposure View */}
            {activeView === 'EXPOSURE' && (
              <div className="space-y-4">
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">EXPOSURE BREAKDOWN</div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">GROSS EXPOSURE</div>
                      <div className="text-[#fff] text-[14px]">${formatNumber(metrics.grossExposure, 0)}</div>
                      <div className="text-[#666] text-[9px] mt-1">
                        {formatNumber((metrics.grossExposure / metrics.totalEquity) * 100, 1)}% OF EQUITY
                      </div>
                    </div>
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">NET EXPOSURE</div>
                      <div className="text-[#00ff00] text-[14px]">${formatNumber(metrics.netExposure, 0)}</div>
                      <div className="text-[#666] text-[9px] mt-1">
                        {formatNumber((metrics.netExposure / metrics.totalEquity) * 100, 1)}% OF EQUITY
                      </div>
                    </div>
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">LONG / SHORT</div>
                      <div className="text-[#fff] text-[14px]">
                        {formatNumber(positions.filter(p => p.side === 'LONG').length / positions.length * 100, 0)}% / 
                        {formatNumber(positions.filter(p => p.side === 'SHORT').length / positions.length * 100, 0)}%
                      </div>
                      <div className="text-[#666] text-[9px] mt-1">
                        {positions.filter(p => p.side === 'LONG').length}L / {positions.filter(p => p.side === 'SHORT').length}S
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">EXPOSURE BY STRATEGY</div>
                  <div className="border border-[#444] bg-[#0a0a0a]">
                    {selectedPortfolio.strategyAllocations.map((alloc, idx) => {
                      const strategyPositions = positions.filter(p => p.strategyId === alloc.strategyId);
                      const strategyExposure = strategyPositions.reduce((sum, p) => sum + p.exposure, 0);
                      return (
                        <div key={idx} className="p-3 border-b border-[#222] last:border-b-0">
                          <div className="flex justify-between items-center mb-2">
                            <div className="text-[#00ff00] text-[10px]">{alloc.strategyId}</div>
                            <div className="text-[#fff] text-[10px]">{formatNumber(strategyExposure, 1)}%</div>
                          </div>
                          <div className="h-2 bg-[#222]">
                            <div
                              className="h-full bg-[#ff8c00]"
                              style={{ width: `${(strategyExposure / 100) * 100}%` }}
                            />
                          </div>
                          <div className="flex justify-between text-[9px] text-[#666] mt-1">
                            <span>ALLOCATED: ${formatNumber(alloc.capitalAllocated, 0)}</span>
                            <span>WEIGHT: {formatNumber(alloc.weight * 100, 0)}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Risk View */}
            {activeView === 'RISK' && (
              <div className="space-y-4">
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RISK LIMITS</div>
                  <div className="border border-[#444] bg-[#0a0a0a]">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                          <th className="px-3 py-2 text-left border-b border-[#444]">TYPE</th>
                          <th className="px-3 py-2 text-left border-b border-[#444]">LIMIT</th>
                          <th className="px-3 py-2 text-right border-b border-[#444]">CURRENT</th>
                          <th className="px-3 py-2 text-right border-b border-[#444]">THRESHOLD</th>
                          <th className="px-3 py-2 text-right border-b border-[#444]">UTILIZATION</th>
                          <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
                          <th className="px-3 py-2 text-left border-b border-[#444]">ACTION</th>
                        </tr>
                      </thead>
                      <tbody>
                        {riskLimits
                          .filter(l => l.portfolioId === selectedPortfolio.id)
                          .map((limit, idx) => {
                            const utilization = (limit.currentValue / limit.threshold) * 100;
                            return (
                              <tr key={idx} className="border-b border-[#222]">
                                <td className="px-3 py-2">
                                  <span className={limit.type === 'HARD' ? 'text-[#ff0000]' : 'text-[#ffff00]'}>
                                    {limit.type}
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-[#fff]">{limit.name}</td>
                                <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(limit.currentValue, 1)}</td>
                                <td className="px-3 py-2 text-right text-[#666]">{formatNumber(limit.threshold, 1)}</td>
                                <td className="px-3 py-2 text-right">
                                  <span className={
                                    utilization >= 90 ? 'text-[#ff0000]' :
                                    utilization >= 70 ? 'text-[#ffff00]' :
                                    'text-[#00ff00]'
                                  }>
                                    {formatNumber(utilization, 0)}%
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-center">
                                  {limit.breached ? (
                                    <span className="text-[#ff0000]">⚠ BREACH</span>
                                  ) : (
                                    <span className="text-[#00ff00]">✓ OK</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 text-[#666] text-[9px]">{limit.action}</td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RISK METRICS</div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">VOLATILITY (30D)</div>
                      <div className="text-[#ffff00] text-[14px]">12.5%</div>
                      <div className="text-[#666] text-[9px] mt-1">ANNUALIZED</div>
                    </div>
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">VAR (95%, 1D)</div>
                      <div className="text-[#ffff00] text-[14px]">$2,500</div>
                      <div className="text-[#666] text-[9px] mt-1">2.4% OF EQUITY</div>
                    </div>
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">CVAR (95%, 1D)</div>
                      <div className="text-[#ff0000] text-[14px]">$3,200</div>
                      <div className="text-[#666] text-[9px] mt-1">3.1% OF EQUITY</div>
                    </div>
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">MAX DRAWDOWN</div>
                      <div className="text-[#ff0000] text-[14px]">5.8%</div>
                      <div className="text-[#666] text-[9px] mt-1">DURATION: 12 DAYS</div>
                    </div>
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">SHARPE RATIO</div>
                      <div className="text-[#00ff00] text-[14px]">1.42</div>
                      <div className="text-[#666] text-[9px] mt-1">ANNUALIZED</div>
                    </div>
                    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                      <div className="text-[#666] text-[9px]">SORTINO RATIO</div>
                      <div className="text-[#00ff00] text-[14px]">1.87</div>
                      <div className="text-[#666] text-[9px] mt-1">DOWNSIDE RISK</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Attribution View */}
            {activeView === 'ATTRIBUTION' && (
              <div className="space-y-4">
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">P&L BY STRATEGY</div>
                  <div className="border border-[#444] bg-[#0a0a0a]">
                    {selectedPortfolio.strategyAllocations.map((alloc, idx) => {
                      const strategyPnl = positions
                        .filter(p => p.strategyId === alloc.strategyId)
                        .reduce((sum, p) => sum + p.unrealizedPnl + p.realizedPnl, 0);
                      return (
                        <div key={idx} className="p-3 border-b border-[#222] last:border-b-0">
                          <div className="flex justify-between items-center">
                            <div className="text-[#00ff00] text-[10px]">{alloc.strategyId}</div>
                            <div className={`text-[10px] ${strategyPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                              {formatPnl(strategyPnl)}
                            </div>
                          </div>
                          <div className="flex justify-between text-[9px] text-[#666] mt-1">
                            <span>CONTRIBUTION: {formatNumber((strategyPnl / metrics.totalPnl) * 100, 1)}%</span>
                            <span>ROI: {formatNumber((strategyPnl / alloc.capitalAllocated) * 100, 1)}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">P&L BY ASSET</div>
                  <div className="border border-[#444] bg-[#0a0a0a]">
                    {positions.map((pos, idx) => {
                      const totalPnl = pos.unrealizedPnl + pos.realizedPnl;
                      return (
                        <div key={idx} className="p-3 border-b border-[#222] last:border-b-0">
                          <div className="flex justify-between items-center">
                            <div className="text-[#00ff00] text-[10px]">{pos.symbol}</div>
                            <div className={`text-[10px] ${totalPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                              {formatPnl(totalPnl)}
                            </div>
                          </div>
                          <div className="flex justify-between text-[9px] text-[#666] mt-1">
                            <span>UNREALIZED: {formatPnl(pos.unrealizedPnl)}</span>
                            <span>REALIZED: {formatPnl(pos.realizedPnl)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center text-[#666] text-[10px] mt-20">
            SELECT A PORTFOLIO TO VIEW DETAILS
          </div>
        )}
      </div>

      {/* Right Panel - Actions & Controls */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">PORTFOLIO ACTIONS</div>
          
          {selectedPortfolio ? (
            <div className="space-y-2">
              {/* Portfolio Controls */}
              <button
                onClick={() => setShowEditPortfolioModal(true)}
                className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
              >
                ✎ EDIT PORTFOLIO
              </button>
              
              <button
                onClick={() => setShowAllocationModal(true)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                REBALANCE ALLOCATION
              </button>
              
              <button
                onClick={() => {
                  updatePortfolio(selectedPortfolio.id, {
                    status: selectedPortfolio.status === 'ACTIVE' ? 'PAUSED' : 'ACTIVE'
                  });
                }}
                className={`w-full py-2 px-3 border text-[10px] transition-colors ${
                  selectedPortfolio.status === 'ACTIVE'
                    ? 'border-[#ffff00] text-[#ffff00] hover:bg-[#ffff00] hover:text-black'
                    : 'border-[#00ff00] text-[#00ff00] hover:bg-[#00ff00] hover:text-black'
                }`}
              >
                {selectedPortfolio.status === 'ACTIVE' ? '⏸ PAUSE PORTFOLIO' : '▶ RESUME PORTFOLIO'}
              </button>
              
              <button
                onClick={() => {
                  if (confirm(`Delete portfolio ${selectedPortfolio.name}?`)) {
                    deletePortfolio(selectedPortfolio.id);
                  }
                }}
                className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
              >
                🗑 DELETE PORTFOLIO
              </button>
            </div>
          ) : (
            <div className="text-[#666] text-[10px]">
              Select a portfolio to view actions
            </div>
          )}
          
          {/* Stress Testing */}
          <div className="mt-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">STRESS SCENARIOS</div>
            <div className="space-y-2">
              {stressScenarios.map(scenario => (
                <div key={scenario.id} className="border border-[#444] p-2">
                  <div className="flex justify-between items-start mb-1">
                    <div className="text-[#00ff00] text-[9px]">{scenario.name}</div>
                    <span className={`text-[8px] ${scenario.type === 'HISTORICAL' ? 'text-[#ffff00]' : 'text-[#666]'}`}>
                      {scenario.type}
                    </span>
                  </div>
                  <div className="text-[#666] text-[8px] mb-2">{scenario.description}</div>
                  <div className="grid grid-cols-2 gap-1 text-[8px] mb-2">
                    <div>
                      <span className="text-[#666]">LOSS:</span>
                      <span className="text-[#ff0000] ml-1">${formatNumber(Math.abs(scenario.impact.portfolioLoss), 0)}</span>
                    </div>
                    <div>
                      <span className="text-[#666]">MAX DD:</span>
                      <span className="text-[#ff0000] ml-1">{formatNumber(scenario.impact.maxDrawdown, 1)}%</span>
                    </div>
                  </div>
                  <button
                    onClick={() => runStressTest(scenario.id)}
                    className="w-full py-1 border border-[#444] text-[#00ff00] text-[8px] hover:border-[#00ff00] transition-colors"
                  >
                    RUN TEST
                  </button>
                </div>
              ))}
            </div>
          </div>
          
          {/* Quick Stats */}
          {selectedPortfolio && metrics && (
            <div className="mt-6">
              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">QUICK STATS</div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">POSITIONS</span>
                    <span className="text-[#fff]">{positions.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">STRATEGIES</span>
                    <span className="text-[#fff]">{selectedPortfolio.strategyAllocations.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">LEVERAGE</span>
                    <span className={metrics.leverage > 2 ? 'text-[#ffff00]' : 'text-[#00ff00]'}>
                      {formatNumber(metrics.leverage)}x
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">RISK UTIL</span>
                    <span className="text-[#00ff00]">{formatNumber(riskUtilization, 1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">BREACHES</span>
                    <span className={hardBreaches.length > 0 ? 'text-[#ff0000]' : 'text-[#00ff00]'}>
                      {hardBreaches.length + softBreaches.length}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Modals */}
      <CreatePortfolioModal
        show={showCreatePortfolioModal}
        onClose={() => setShowCreatePortfolioModal(false)}
        onCreate={createPortfolio}
        availableStrategies={availableStrategies}
      />
      
      <EditPortfolioModal
        show={showEditPortfolioModal}
        onClose={() => setShowEditPortfolioModal(false)}
      />
      
      <RiskLimitModal
        show={showRiskLimitModal}
        onClose={() => setShowRiskLimitModal(false)}
      />
      
      <StressTestModal
        show={showStressTestModal}
        onClose={() => setShowStressTestModal(false)}
      />
      
      <AllocationModal
        show={showAllocationModal}
        onClose={() => setShowAllocationModal(false)}
      />
      
      <AttributionModal
        show={showAttributionModal}
        onClose={() => setShowAttributionModal(false)}
      />
    </div>
  );
}
