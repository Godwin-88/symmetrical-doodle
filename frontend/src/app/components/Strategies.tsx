import { useTradingStore } from '@/app/store/tradingStore';
import { useState, useEffect } from 'react';
import {
  listStrategies,
  getStrategyFamilies,
  getTimeHorizons,
  formatFamily,
  formatHorizon,
  getComplexityColor,
  getDataReqColor,
  getLatencySensitivityColor,
  type StrategySpec,
  type StrategyFamily as StrategyFamilyType,
  type TimeHorizon as TimeHorizonType,
} from '../../services/strategiesService';

interface StrategyConfig {
  id: string;
  strategy_id: string;
  name: string;
  enabled_markets: string[];
  parameters: Record<string, any>;
  risk_limits: {
    max_position_size: number;
    max_leverage: number;
    stop_loss_pct: number;
  };
  status: 'ACTIVE' | 'PAUSED' | 'STOPPED';
}

interface BacktestResult {
  strategy_id: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  equity_curve: Array<{ date: string; value: number }>;
  drawdown_curve: Array<{ date: string; value: number }>;
}

export function Strategies() {
  const { strategies } = useTradingStore();
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  
  // Strategy browser state
  const [showStrategyBrowser, setShowStrategyBrowser] = useState(false);
  const [availableStrategies, setAvailableStrategies] = useState<StrategySpec[]>([]);
  const [strategyFamilies, setStrategyFamilies] = useState<StrategyFamilyType[]>([]);
  const [timeHorizons, setTimeHorizons] = useState<TimeHorizonType[]>([]);
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [selectedHorizon, setSelectedHorizon] = useState<string | null>(null);
  const [productionOnlyFilter, setProductionOnlyFilter] = useState(true);
  const [selectedBrowserStrategy, setSelectedBrowserStrategy] = useState<StrategySpec | null>(null);
  
  // Configuration state
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showBacktestModal, setShowBacktestModal] = useState(false);
  const [editingConfig, setEditingConfig] = useState<StrategyConfig | null>(null);
  
  // Mock strategy configurations
  const [strategyConfigs, setStrategyConfigs] = useState<StrategyConfig[]>([
    {
      id: '1',
      strategy_id: 'ma_crossover',
      name: 'MA CROSSOVER - EURUSD',
      enabled_markets: ['EURUSD', 'GBPUSD'],
      parameters: { fast_period: 20, slow_period: 100, vol_threshold: 0.01 },
      risk_limits: { max_position_size: 0.20, max_leverage: 2.0, stop_loss_pct: 0.03 },
      status: 'ACTIVE'
    },
  ]);
  
  // Mock backtest results
  const generateMockBacktest = (strategyId: string): BacktestResult => {
    const days = 252;
    const equityCurve: Array<{ date: string; value: number }> = [];
    const drawdownCurve: Array<{ date: string; value: number }> = [];
    let equity = 100000;
    let peak = equity;
    
    for (let i = 0; i < days; i++) {
      const dailyReturn = (Math.random() - 0.48) * 0.02; // Slight positive bias
      equity *= (1 + dailyReturn);
      peak = Math.max(peak, equity);
      const drawdown = ((equity - peak) / peak) * 100;
      
      equityCurve.push({
        date: new Date(Date.now() - (days - i) * 86400000).toISOString().split('T')[0],
        value: equity
      });
      
      drawdownCurve.push({
        date: new Date(Date.now() - (days - i) * 86400000).toISOString().split('T')[0],
        value: drawdown
      });
    }
    
    const totalReturn = ((equity - 100000) / 100000) * 100;
    const maxDrawdown = Math.min(...drawdownCurve.map(d => d.value));
    
    return {
      strategy_id: strategyId,
      total_return: totalReturn,
      sharpe_ratio: 1.2 + Math.random() * 0.8,
      max_drawdown: Math.abs(maxDrawdown),
      win_rate: 0.45 + Math.random() * 0.15,
      total_trades: Math.floor(50 + Math.random() * 100),
      equity_curve: equityCurve,
      drawdown_curve: drawdownCurve
    };
  };

  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);

  // Fetch strategies on mount
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const [strats, families, horizons] = await Promise.all([
          listStrategies({ production_ready: productionOnlyFilter }),
          getStrategyFamilies(),
          getTimeHorizons(),
        ]);
        setAvailableStrategies(strats);
        setStrategyFamilies(families);
        setTimeHorizons(horizons);
      } catch (err) {
        console.warn('Failed to fetch strategies:', err);
      }
    };
    
    fetchStrategies();
  }, [productionOnlyFilter]);

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPnl = (pnl: number) => {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}${formatNumber(pnl)}`;
  };

  // Mock strategy details
  const strategyDetails = {
    'MOMENTUM ALPHA': {
      family: 'TREND',
      horizon: 'DAILY',
      description: 'MOMENTUM-BASED TREND FOLLOWING USING TECHNICAL INDICATORS',
      enabledMarkets: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
      parameters: {
        LOOKBACK_PERIOD: 20,
        MOMENTUM_THRESHOLD: 0.02,
        STOP_LOSS: 0.015,
        TAKE_PROFIT: 0.03,
      },
      regimeAffinity: [
        { regime: 'LOW_VOL_TRENDING', affinity: 0.85 },
        { regime: 'MEDIUM_VOL_TRENDING', affinity: 0.72 },
        { regime: 'HIGH_VOL_RANGING', affinity: 0.23 },
      ],
      riskBudget: {
        MAX_ALLOCATION: 25,
        MAX_LEVERAGE: 2.0,
        STOP_LOSS: 5.0,
      },
    },
  };

  const currentStrategy = selectedStrategy || 'MOMENTUM ALPHA';
  const details = strategyDetails[currentStrategy as keyof typeof strategyDetails] || strategyDetails['MOMENTUM ALPHA'];

  // CRUD operations
  const createStrategyConfig = (config: Omit<StrategyConfig, 'id'>) => {
    const newConfig: StrategyConfig = {
      ...config,
      id: Date.now().toString(),
    };
    setStrategyConfigs([...strategyConfigs, newConfig]);
  };

  const updateStrategyConfig = (id: string, updates: Partial<StrategyConfig>) => {
    setStrategyConfigs(strategyConfigs.map(c => c.id === id ? { ...c, ...updates } : c));
  };

  const deleteStrategyConfig = (id: string) => {
    setStrategyConfigs(strategyConfigs.filter(c => c.id !== id));
  };

  const handleRunBacktest = (strategyId: string) => {
    const result = generateMockBacktest(strategyId);
    setBacktestResult(result);
    setShowBacktestModal(true);
  };

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel - Strategy List */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">STRATEGY CATALOG</div>
          <div className="space-y-2">
            {strategies.map((strategy, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedStrategy(strategy.name)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedStrategy === strategy.name || (!selectedStrategy && idx === 0)
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{strategy.name}</div>
                  <span className={`
                    text-[10px]
                    ${strategy.status === 'ACTIVE' ? 'text-[#00ff00]' : ''}
                    ${strategy.status === 'PAUSED' ? 'text-[#ffff00]' : ''}
                    ${strategy.status === 'STOPPED' ? 'text-[#ff0000]' : ''}
                  `}>
                    {strategy.status}
                  </span>
                </div>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">ALLOC:</span>
                    <span className="text-[#fff]">{strategy.allocation}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">P&L:</span>
                    <span className={strategy.pnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {formatPnl(strategy.pnl)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">SHARPE:</span>
                    <span className="text-[#fff]">{formatNumber(strategy.sharpe)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>


      {/* Center Panel - Strategy Details */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="text-[#ff8c00] text-sm tracking-wider">
            STRATEGY: {currentStrategy}
          </div>
        </div>

        <div className="space-y-6">
          {/* Definition */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">DEFINITION</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="space-y-2 text-[10px]">
                <div className="flex gap-4">
                  <span className="text-[#666]">FAMILY:</span>
                  <span className="text-[#00ff00]">{details.family}</span>
                  <span className="mx-2 text-[#666]">|</span>
                  <span className="text-[#666]">HORIZON:</span>
                  <span className="text-[#00ff00]">{details.horizon}</span>
                </div>
                <div className="mt-2">
                  <span className="text-[#666]">DESCRIPTION:</span>
                  <div className="text-[#fff] mt-1">{details.description}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Enabled Markets */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">ENABLED MARKETS</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="flex flex-wrap gap-2">
                {details.enabledMarkets.map((market, idx) => (
                  <div key={idx} className="border border-[#00ff00] px-2 py-1 text-[#00ff00] text-[10px]">
                    {market}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Parameters */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">PARAMETERS</div>
            <div className="border border-[#444] bg-[#0a0a0a]">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                    <th className="px-3 py-2 text-left border-b border-[#444]">PARAMETER</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">VALUE</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(details.parameters).map(([key, value], idx) => (
                    <tr key={idx} className="border-b border-[#222]">
                      <td className="px-3 py-2 text-[#666]">{key}</td>
                      <td className="px-3 py-2 text-right text-[#00ff00]">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Regime Affinity */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">REGIME AFFINITY</div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2">
              {details.regimeAffinity.map((item, idx) => (
                <div key={idx} className="flex items-center gap-4">
                  <div className="w-48 text-[#00ff00]">{item.regime}</div>
                  <div className="w-16 text-right text-[#fff]">{formatNumber(item.affinity * 100, 0)}%</div>
                  <div className="flex-1">
                    <div className="h-2 bg-[#222]">
                      <div
                        className="h-full bg-[#ff8c00]"
                        style={{ width: `${item.affinity * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Risk Budget */}
          <div>
            <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RISK BUDGET</div>
            <div className="border border-[#444] bg-[#0a0a0a]">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                    <th className="px-3 py-2 text-left border-b border-[#444]">LIMIT</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">VALUE</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(details.riskBudget).map(([key, value], idx) => (
                    <tr key={idx} className="border-b border-[#222]">
                      <td className="px-3 py-2 text-[#666]">{key}</td>
                      <td className="px-3 py-2 text-right text-[#ffff00]">
                        {typeof value === 'number' && value < 10 ? formatNumber(value) : value}
                        {key.includes('ALLOCATION') || key.includes('LOSS') ? '%' : ''}
                        {key.includes('LEVERAGE') ? 'x' : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">STRATEGY ACTIONS</div>
          
          <div className="space-y-2 mb-6">
            <button
              onClick={() => setShowStrategyBrowser(true)}
              className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
            >
              BROWSE STRATEGIES ({availableStrategies.length})
            </button>
          </div>
          
          <div className="space-y-2">
            <button className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors">
              ACTIVATE STRATEGY
            </button>
            <button className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors">
              PAUSE STRATEGY
            </button>
            <button className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors">
              STOP STRATEGY
            </button>
          </div>

          <div className="mt-6 space-y-3">
            <button
              onClick={() => setShowConfigModal(true)}
              className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
            >
              EDIT PARAMETERS
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              CONFIGURE REGIMES
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              ADJUST RISK BUDGET
            </button>
            <button
              onClick={() => handleRunBacktest('ma_crossover')}
              className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
            >
              VIEW BACKTEST RESULTS
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              AUDIT TRAIL
            </button>
          </div>
        </div>
      </div>


      {/* Strategy Browser Modal */}
      {showStrategyBrowser && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">STRATEGY REGISTRY</h2>
              <button
                onClick={() => {
                  setShowStrategyBrowser(false);
                  setSelectedBrowserStrategy(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            {/* Filters */}
            <div className="flex gap-4 mb-4 flex-wrap">
              <div>
                <label className="text-[#666] text-[10px] block mb-1">FAMILY</label>
                <select
                  value={selectedFamily || ''}
                  onChange={(e) => setSelectedFamily(e.target.value || null)}
                  className="bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="">All Families</option>
                  {strategyFamilies.map(fam => (
                    <option key={fam.id} value={fam.id}>{fam.name} ({fam.count})</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[#666] text-[10px] block mb-1">HORIZON</label>
                <select
                  value={selectedHorizon || ''}
                  onChange={(e) => setSelectedHorizon(e.target.value || null)}
                  className="bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="">All Horizons</option>
                  {timeHorizons.map(hor => (
                    <option key={hor.id} value={hor.id}>{hor.name} ({hor.count})</option>
                  ))}
                </select>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-[10px] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={productionOnlyFilter}
                    onChange={(e) => setProductionOnlyFilter(e.target.checked)}
                    className="form-checkbox"
                  />
                  <span className="text-[#666]">Production Ready Only</span>
                </label>
              </div>
            </div>

            {/* Strategy List and Details */}
            <div className="flex-1 flex gap-4 overflow-hidden">
              {/* Strategy List */}
              <div className="w-1/2 border border-[#444] overflow-y-auto">
                <div className="space-y-1">
                  {availableStrategies
                    .filter(s => !selectedFamily || s.family === selectedFamily)
                    .filter(s => !selectedHorizon || s.horizon === selectedHorizon)
                    .map(strategy => (
                      <div
                        key={strategy.id}
                        onClick={() => setSelectedBrowserStrategy(strategy)}
                        className={`p-3 border-b border-[#222] cursor-pointer transition-colors ${
                          selectedBrowserStrategy?.id === strategy.id
                            ? 'bg-[#1a1a1a] border-l-4 border-l-[#ff8c00]'
                            : 'hover:bg-[#0f0f0f]'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="text-[#00ff00] text-[11px] font-bold">{strategy.name}</div>
                            <div className="text-[#666] text-[9px] mt-1">
                              {formatFamily(strategy.family)} | {formatHorizon(strategy.horizon)}
                            </div>
                            <div className="text-[#fff] text-[10px] mt-1 line-clamp-2">{strategy.description}</div>
                          </div>
                          {strategy.production_ready && (
                            <div className="ml-2 px-2 py-1 bg-[#00ff00] text-black text-[8px]">PROD</div>
                          )}
                        </div>
                        <div className="flex gap-2 mt-2 text-[9px]">
                          <span className="text-[#00ff00]">SHARPE: {strategy.typical_sharpe.toFixed(1)}</span>
                          <span className="text-[#666]">|</span>
                          <span className="text-[#ff8c00]">DD: {strategy.typical_max_dd.toFixed(1)}%</span>
                          <span className="text-[#666]">|</span>
                          <span style={{ color: getComplexityColor(strategy.complexity) }}>
                            {strategy.complexity.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    ))
                  }
                </div>
              </div>

              {/* Strategy Details */}
              <div className="w-1/2 border border-[#444] overflow-y-auto p-4">
                {selectedBrowserStrategy ? (
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-[#ff8c00] text-[12px] mb-2">{selectedBrowserStrategy.name}</h3>
                      <p className="text-[#fff] text-[10px]">{selectedBrowserStrategy.description}</p>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">SIGNAL LOGIC</div>
                      <p className="text-[#00ff00] text-[9px]">{selectedBrowserStrategy.signal_logic}</p>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">ENTRY RULES</div>
                      <ul className="space-y-1">
                        {selectedBrowserStrategy.entry_rules.map((rule, i) => (
                          <li key={i} className="text-[#00ff00] text-[9px]">→ {rule}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">EXIT RULES</div>
                      <ul className="space-y-1">
                        {selectedBrowserStrategy.exit_rules.map((rule, i) => (
                          <li key={i} className="text-[#ffff00] text-[9px]">← {rule}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">STRENGTHS</div>
                      <ul className="space-y-1">
                        {selectedBrowserStrategy.strengths.map((s, i) => (
                          <li key={i} className="text-[#00ff00] text-[9px]">✓ {s}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <div className="text-[#ff8c00] text-[10px] mb-2">WEAKNESSES</div>
                      <ul className="space-y-1">
                        {selectedBrowserStrategy.weaknesses.map((w, i) => (
                          <li key={i} className="text-[#ff0000] text-[9px]">✗ {w}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-[#666] text-[9px]">Complexity</div>
                        <div style={{ color: getComplexityColor(selectedBrowserStrategy.complexity) }} className="text-[10px]">
                          {selectedBrowserStrategy.complexity.toUpperCase()}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Data Requirements</div>
                        <div style={{ color: getDataReqColor(selectedBrowserStrategy.data_requirements) }} className="text-[10px]">
                          {selectedBrowserStrategy.data_requirements.toUpperCase()}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Latency Sensitivity</div>
                        <div style={{ color: getLatencySensitivityColor(selectedBrowserStrategy.latency_sensitivity) }} className="text-[10px]">
                          {selectedBrowserStrategy.latency_sensitivity.toUpperCase()}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Typical Sharpe</div>
                        <div className="text-[#00ff00] text-[10px]">{selectedBrowserStrategy.typical_sharpe.toFixed(2)}</div>
                      </div>
                    </div>

                    <div className="pt-4">
                      <button
                        onClick={() => {
                          // Use this strategy
                          createStrategyConfig({
                            strategy_id: selectedBrowserStrategy.id,
                            name: `${selectedBrowserStrategy.name} - CONFIG`,
                            enabled_markets: ['EURUSD'],
                            parameters: selectedBrowserStrategy.parameters,
                            risk_limits: {
                              max_position_size: selectedBrowserStrategy.max_position_size,
                              max_leverage: selectedBrowserStrategy.max_leverage,
                              stop_loss_pct: selectedBrowserStrategy.stop_loss_pct,
                            },
                            status: 'STOPPED'
                          });
                          setShowStrategyBrowser(false);
                          setSelectedBrowserStrategy(null);
                        }}
                        className="w-full px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
                      >
                        USE THIS STRATEGY
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-[#666] text-[10px]">
                    SELECT A STRATEGY TO VIEW DETAILS
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}


      {/* Backtest Results Modal */}
      {showBacktestModal && backtestResult && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">BACKTEST RESULTS</h2>
              <button
                onClick={() => setShowBacktestModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">TOTAL RETURN</div>
                <div className={`text-[14px] ${backtestResult.total_return >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                  {formatPnl(backtestResult.total_return)}%
                </div>
              </div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">SHARPE RATIO</div>
                <div className="text-[#00ff00] text-[14px]">{backtestResult.sharpe_ratio.toFixed(2)}</div>
              </div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">MAX DRAWDOWN</div>
                <div className="text-[#ff0000] text-[14px]">{backtestResult.max_drawdown.toFixed(2)}%</div>
              </div>
              <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                <div className="text-[#666] text-[9px]">WIN RATE</div>
                <div className="text-[#00ff00] text-[14px]">{(backtestResult.win_rate * 100).toFixed(1)}%</div>
              </div>
            </div>

            {/* Equity Curve */}
            <div className="mb-6">
              <div className="text-[#ff8c00] text-[10px] mb-2">EQUITY CURVE</div>
              <div className="border border-[#444] p-4 bg-[#0a0a0a]">
                <div className="h-48 relative">
                  <svg className="w-full h-full">
                    {/* Grid lines */}
                    {[0, 25, 50, 75, 100].map((pct) => (
                      <line
                        key={pct}
                        x1="0"
                        y1={`${pct}%`}
                        x2="100%"
                        y2={`${pct}%`}
                        stroke="#222"
                        strokeWidth="1"
                      />
                    ))}
                    
                    {/* Equity curve */}
                    <polyline
                      points={backtestResult.equity_curve.map((point, i) => {
                        const x = (i / (backtestResult.equity_curve.length - 1)) * 100;
                        const minValue = Math.min(...backtestResult.equity_curve.map(p => p.value));
                        const maxValue = Math.max(...backtestResult.equity_curve.map(p => p.value));
                        const y = 100 - ((point.value - minValue) / (maxValue - minValue)) * 100;
                        return `${x},${y}`;
                      }).join(' ')}
                      fill="none"
                      stroke="#00ff00"
                      strokeWidth="2"
                    />
                  </svg>
                  
                  {/* Y-axis labels */}
                  <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-[8px] text-[#666]">
                    <div>${formatNumber(Math.max(...backtestResult.equity_curve.map(p => p.value)), 0)}</div>
                    <div>${formatNumber(Math.min(...backtestResult.equity_curve.map(p => p.value)), 0)}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Drawdown Curve */}
            <div className="mb-6">
              <div className="text-[#ff8c00] text-[10px] mb-2">DRAWDOWN CURVE</div>
              <div className="border border-[#444] p-4 bg-[#0a0a0a]">
                <div className="h-32 relative">
                  <svg className="w-full h-full">
                    {/* Grid lines */}
                    {[0, 25, 50, 75, 100].map((pct) => (
                      <line
                        key={pct}
                        x1="0"
                        y1={`${pct}%`}
                        x2="100%"
                        y2={`${pct}%`}
                        stroke="#222"
                        strokeWidth="1"
                      />
                    ))}
                    
                    {/* Drawdown area */}
                    <polygon
                      points={[
                        '0,0',
                        ...backtestResult.drawdown_curve.map((point, i) => {
                          const x = (i / (backtestResult.drawdown_curve.length - 1)) * 100;
                          const y = Math.abs(point.value / backtestResult.max_drawdown) * 100;
                          return `${x},${y}`;
                        }),
                        '100,0'
                      ].join(' ')}
                      fill="rgba(255, 0, 0, 0.2)"
                      stroke="#ff0000"
                      strokeWidth="2"
                    />
                  </svg>
                  
                  {/* Y-axis labels */}
                  <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-[8px] text-[#666]">
                    <div>0%</div>
                    <div>-{backtestResult.max_drawdown.toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Trade Statistics */}
            <div>
              <div className="text-[#ff8c00] text-[10px] mb-2">TRADE STATISTICS</div>
              <div className="border border-[#444] bg-[#0a0a0a]">
                <table className="w-full">
                  <tbody>
                    <tr className="border-b border-[#222]">
                      <td className="px-3 py-2 text-[#666] text-[10px]">Total Trades</td>
                      <td className="px-3 py-2 text-right text-[#fff] text-[10px]">{backtestResult.total_trades}</td>
                    </tr>
                    <tr className="border-b border-[#222]">
                      <td className="px-3 py-2 text-[#666] text-[10px]">Winning Trades</td>
                      <td className="px-3 py-2 text-right text-[#00ff00] text-[10px]">
                        {Math.floor(backtestResult.total_trades * backtestResult.win_rate)}
                      </td>
                    </tr>
                    <tr className="border-b border-[#222]">
                      <td className="px-3 py-2 text-[#666] text-[10px]">Losing Trades</td>
                      <td className="px-3 py-2 text-right text-[#ff0000] text-[10px]">
                        {Math.floor(backtestResult.total_trades * (1 - backtestResult.win_rate))}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-3 py-2 text-[#666] text-[10px]">Avg Trade Duration</td>
                      <td className="px-3 py-2 text-right text-[#fff] text-[10px]">
                        {(252 / backtestResult.total_trades).toFixed(1)} days
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-6 flex gap-2">
              <button
                onClick={() => setShowBacktestModal(false)}
                className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
              >
                CLOSE
              </button>
              <button
                className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px]"
              >
                DEPLOY STRATEGY
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Configuration Modal */}
      {showConfigModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">STRATEGY CONFIGURATION</h2>
              <button
                onClick={() => setShowConfigModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <div className="text-[#666] text-[10px] mb-4">
              Configure strategy parameters, risk limits, and enabled markets
            </div>

            <div className="space-y-4">
              <div className="border border-[#444] p-4">
                <div className="text-[#ff8c00] text-[10px] mb-3">PARAMETERS</div>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(details.parameters).map(([key, value]) => (
                    <div key={key}>
                      <label className="text-[#666] block mb-1 text-[9px]">{key}</label>
                      <input
                        type="number"
                        defaultValue={value as number}
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="border border-[#444] p-4">
                <div className="text-[#ff8c00] text-[10px] mb-3">RISK LIMITS</div>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(details.riskBudget).map(([key, value]) => (
                    <div key={key}>
                      <label className="text-[#666] block mb-1 text-[9px]">{key}</label>
                      <input
                        type="number"
                        defaultValue={value as number}
                        step="0.01"
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setShowConfigModal(false)}
                className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
              >
                CANCEL
              </button>
              <button
                onClick={() => setShowConfigModal(false)}
                className="flex-1 px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
              >
                SAVE CONFIGURATION
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
