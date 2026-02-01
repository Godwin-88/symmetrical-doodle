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
import {
  translateStrategy,
  listStrategyTranslations,
  createBacktest,
  listBacktests,
  type StrategyTranslation,
  type BacktestResult,
  type BacktestConfig,
  StrategyTranslationStatus,
  BacktestStatus,
  getStatusColor,
  formatDuration,
} from '../../services/nautilusService';

// Types
type StrategyCategory = 'DEPLOYED' | 'CATALOG' | 'TRANSLATIONS' | 'BACKTESTS' | 'TEMPLATES';

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
  nautilusTranslationId?: string;
}

interface LocalBacktestResult {
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

  // Category selection state
  const [selectedCategory, setSelectedCategory] = useState<StrategyCategory>('DEPLOYED');
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
  const [showTranslationModal, setShowTranslationModal] = useState(false);
  const [showNewBacktestModal, setShowNewBacktestModal] = useState(false);
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [editingConfig, setEditingConfig] = useState<StrategyConfig | null>(null);

  // Nautilus integration state
  const [strategyTranslations, setStrategyTranslations] = useState<StrategyTranslation[]>([]);
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);
  const [selectedTranslation, setSelectedTranslation] = useState<StrategyTranslation | null>(null);
  const [selectedBacktest, setSelectedBacktest] = useState<BacktestResult | null>(null);

  // Status state
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

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
    {
      id: '2',
      strategy_id: 'trend_following',
      name: 'TREND FOLLOWING - MULTI',
      enabled_markets: ['EURUSD', 'GBPUSD', 'USDJPY'],
      parameters: { lookback: 50, entry_threshold: 0.02, exit_threshold: 0.01 },
      risk_limits: { max_position_size: 0.15, max_leverage: 1.5, stop_loss_pct: 0.025 },
      status: 'ACTIVE'
    },
    {
      id: '3',
      strategy_id: 'mean_reversion',
      name: 'MEAN REVERSION - USDJPY',
      enabled_markets: ['USDJPY'],
      parameters: { z_score_entry: 2.0, z_score_exit: 0.5, lookback: 30 },
      risk_limits: { max_position_size: 0.10, max_leverage: 1.0, stop_loss_pct: 0.02 },
      status: 'PAUSED'
    },
  ]);

  // Strategy templates
  const strategyTemplates = [
    { id: 'template_trend', name: 'Trend Following Template', family: 'TREND', horizon: 'DAILY' },
    { id: 'template_mean', name: 'Mean Reversion Template', family: 'MEAN_REVERSION', horizon: 'INTRADAY' },
    { id: 'template_momentum', name: 'Momentum Template', family: 'MOMENTUM', horizon: 'SWING' },
    { id: 'template_volatility', name: 'Volatility Arbitrage Template', family: 'VOLATILITY', horizon: 'INTRADAY' },
  ];

  // Mock backtest results generator
  const generateMockBacktest = (strategyId: string): LocalBacktestResult => {
    const days = 252;
    const equityCurve: Array<{ date: string; value: number }> = [];
    const drawdownCurve: Array<{ date: string; value: number }> = [];
    let equity = 100000;
    let peak = equity;

    for (let i = 0; i < days; i++) {
      const dailyReturn = (Math.random() - 0.48) * 0.02;
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

  const [backtestResult, setBacktestResult] = useState<LocalBacktestResult | null>(null);

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

  // Fetch Nautilus data
  useEffect(() => {
    const fetchNautilusData = async () => {
      try {
        const [translations, backtests] = await Promise.all([
          listStrategyTranslations(),
          listBacktests(),
        ]);
        setStrategyTranslations(translations);
        setBacktestResults(backtests);
      } catch (err) {
        console.warn('Failed to fetch Nautilus data:', err);
      }
    };

    fetchNautilusData();
  }, []);

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPnl = (pnl: number) => {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}${formatNumber(pnl)}`;
  };

  // Get current strategy details
  const getCurrentStrategy = () => {
    if (selectedCategory === 'DEPLOYED') {
      return strategyConfigs.find(s => s.id === selectedStrategy) || strategyConfigs[0];
    }
    return null;
  };

  const currentConfig = getCurrentStrategy();

  // Strategy details for display
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

  // CRUD operations
  const createStrategyConfig = (config: Omit<StrategyConfig, 'id'>) => {
    const newConfig: StrategyConfig = {
      ...config,
      id: Date.now().toString(),
    };
    setStrategyConfigs([...strategyConfigs, newConfig]);
    setActionStatus('Strategy configuration created');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const updateStrategyConfig = (id: string, updates: Partial<StrategyConfig>) => {
    setStrategyConfigs(strategyConfigs.map(c => c.id === id ? { ...c, ...updates } : c));
    setActionStatus('Strategy configuration updated');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const deleteStrategyConfig = (id: string) => {
    setStrategyConfigs(strategyConfigs.filter(c => c.id !== id));
    setActionStatus('Strategy configuration deleted');
    setTimeout(() => setActionStatus(null), 3000);
  };

  // Nautilus operations
  const handleTranslateStrategy = async (strategyId: string) => {
    setIsLoading(true);
    setActionStatus('Translating strategy to Nautilus format...');
    try {
      const result = await translateStrategy(strategyId, {
        validateOnly: false,
        includeSignals: true,
        riskIntegration: true,
      });
      setStrategyTranslations([...strategyTranslations, result]);
      setSelectedTranslation(result);
      setShowTranslationModal(true);
      setActionStatus('Strategy translation initiated');
    } catch (err: any) {
      setActionStatus(`Translation failed: ${err.message}`);
    } finally {
      setIsLoading(false);
      setTimeout(() => setActionStatus(null), 3000);
    }
  };

  const handleCreateBacktest = async (config: BacktestConfig) => {
    setIsLoading(true);
    setActionStatus('Creating backtest...');
    try {
      const result = await createBacktest(config);
      setBacktestResults([...backtestResults, result]);
      setSelectedBacktest(result);
      setShowNewBacktestModal(false);
      setActionStatus('Backtest created and running');
    } catch (err: any) {
      setActionStatus(`Backtest creation failed: ${err.message}`);
    } finally {
      setIsLoading(false);
      setTimeout(() => setActionStatus(null), 3000);
    }
  };

  const handleRunBacktest = (strategyId: string) => {
    const result = generateMockBacktest(strategyId);
    setBacktestResult(result);
    setShowBacktestModal(true);
  };

  const handleActivateStrategy = (id: string) => {
    updateStrategyConfig(id, { status: 'ACTIVE' });
    setActionStatus('Strategy activated');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const handlePauseStrategy = (id: string) => {
    updateStrategyConfig(id, { status: 'PAUSED' });
    setActionStatus('Strategy paused');
    setTimeout(() => setActionStatus(null), 3000);
  };

  const handleStopStrategy = (id: string) => {
    updateStrategyConfig(id, { status: 'STOPPED' });
    setActionStatus('Strategy stopped');
    setTimeout(() => setActionStatus(null), 3000);
  };

  // Categories configuration
  const categories: { key: StrategyCategory; label: string; count: number }[] = [
    { key: 'DEPLOYED', label: 'DEPLOYED STRATEGIES', count: strategyConfigs.length },
    { key: 'CATALOG', label: 'STRATEGY CATALOG', count: availableStrategies.length },
    { key: 'TRANSLATIONS', label: 'NAUTILUS TRANSLATIONS', count: strategyTranslations.length },
    { key: 'BACKTESTS', label: 'BACKTEST RESULTS', count: backtestResults.length },
    { key: 'TEMPLATES', label: 'TEMPLATES', count: strategyTemplates.length },
  ];

  // Render left panel content based on category
  const renderLeftPanelContent = () => {
    switch (selectedCategory) {
      case 'DEPLOYED':
        return (
          <div className="space-y-2">
            {strategyConfigs.map((config) => (
              <div
                key={config.id}
                onClick={() => setSelectedStrategy(config.id)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedStrategy === config.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{config.name}</div>
                  <span className={`
                    text-[10px] px-2 py-0.5 border
                    ${config.status === 'ACTIVE' ? 'border-[#00ff00] text-[#00ff00]' : ''}
                    ${config.status === 'PAUSED' ? 'border-[#ffff00] text-[#ffff00]' : ''}
                    ${config.status === 'STOPPED' ? 'border-[#ff0000] text-[#ff0000]' : ''}
                  `}>
                    {config.status}
                  </span>
                </div>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">MARKETS:</span>
                    <span className="text-[#fff]">{config.enabled_markets.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">MAX LEV:</span>
                    <span className="text-[#fff]">{config.risk_limits.max_leverage}x</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        );

      case 'CATALOG':
        return (
          <div className="space-y-2">
            {availableStrategies.slice(0, 10).map((strategy) => (
              <div
                key={strategy.id}
                onClick={() => setSelectedBrowserStrategy(strategy)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedBrowserStrategy?.id === strategy.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00] text-[11px]">{strategy.name}</div>
                  {strategy.production_ready && (
                    <span className="text-[8px] px-1 bg-[#00ff00] text-black">PROD</span>
                  )}
                </div>
                <div className="text-[#666] text-[9px]">
                  {formatFamily(strategy.family)} | {formatHorizon(strategy.horizon)}
                </div>
                <div className="mt-1 text-[9px]">
                  <span className="text-[#00ff00]">SR: {strategy.typical_sharpe.toFixed(1)}</span>
                  <span className="mx-2 text-[#666]">|</span>
                  <span className="text-[#ff8c00]">DD: {strategy.typical_max_dd.toFixed(1)}%</span>
                </div>
              </div>
            ))}
            <button
              onClick={() => setShowStrategyBrowser(true)}
              className="w-full py-2 border border-dashed border-[#444] text-[#666] text-[10px] hover:border-[#ff8c00] hover:text-[#ff8c00]"
            >
              VIEW ALL ({availableStrategies.length})
            </button>
          </div>
        );

      case 'TRANSLATIONS':
        return (
          <div className="space-y-2">
            {strategyTranslations.map((translation) => (
              <div
                key={translation.id}
                onClick={() => setSelectedTranslation(translation)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedTranslation?.id === translation.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{translation.f6StrategyName}</div>
                  <span className={`text-[10px] ${getStatusColor(translation.status)}`}>
                    {translation.status}
                  </span>
                </div>
                <div className="text-[#666] text-[10px]">
                  Progress: {translation.progress}%
                </div>
              </div>
            ))}
            {strategyTranslations.length === 0 && (
              <div className="text-[#666] text-[10px] text-center py-4">
                No translations yet. Translate a strategy to use with Nautilus.
              </div>
            )}
          </div>
        );

      case 'BACKTESTS':
        return (
          <div className="space-y-2">
            {backtestResults.map((backtest) => (
              <div
                key={backtest.id}
                onClick={() => setSelectedBacktest(backtest)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedBacktest?.id === backtest.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{backtest.config.name}</div>
                  <span className={`text-[10px] ${getStatusColor(backtest.status)}`}>
                    {backtest.status}
                  </span>
                </div>
                <div className="text-[#666] text-[10px]">
                  {backtest.progress}% complete
                </div>
                {backtest.metrics && (
                  <div className="mt-1 text-[9px]">
                    <span className="text-[#00ff00]">RET: {backtest.metrics.totalReturn.toFixed(1)}%</span>
                    <span className="mx-2 text-[#666]">|</span>
                    <span className="text-[#ff8c00]">SR: {backtest.metrics.sharpeRatio.toFixed(2)}</span>
                  </div>
                )}
              </div>
            ))}
            <button
              onClick={() => setShowNewBacktestModal(true)}
              className="w-full py-2 border border-dashed border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black"
            >
              + NEW BACKTEST
            </button>
          </div>
        );

      case 'TEMPLATES':
        return (
          <div className="space-y-2">
            {strategyTemplates.map((template) => (
              <div
                key={template.id}
                className="border border-[#333] p-3 cursor-pointer hover:border-[#ff8c00] transition-colors"
              >
                <div className="text-[#00ff00]">{template.name}</div>
                <div className="text-[#666] text-[10px] mt-1">
                  {template.family} | {template.horizon}
                </div>
              </div>
            ))}
          </div>
        );

      default:
        return null;
    }
  };

  // Render center panel content based on category
  const renderCenterPanelContent = () => {
    switch (selectedCategory) {
      case 'DEPLOYED':
        if (!currentConfig) return <div className="text-[#666]">Select a strategy</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                STRATEGY: {currentConfig.name}
              </div>
            </div>

            <div className="space-y-6">
              {/* Status */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">STATUS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="flex items-center gap-4">
                    <div className={`
                      px-3 py-1 border text-[10px]
                      ${currentConfig.status === 'ACTIVE' ? 'border-[#00ff00] text-[#00ff00] bg-[#00ff00]/10' : ''}
                      ${currentConfig.status === 'PAUSED' ? 'border-[#ffff00] text-[#ffff00] bg-[#ffff00]/10' : ''}
                      ${currentConfig.status === 'STOPPED' ? 'border-[#ff0000] text-[#ff0000] bg-[#ff0000]/10' : ''}
                    `}>
                      {currentConfig.status}
                    </div>
                    <div className="text-[#666] text-[10px]">
                      ID: {currentConfig.strategy_id}
                    </div>
                  </div>
                </div>
              </div>

              {/* Enabled Markets */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">ENABLED MARKETS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="flex flex-wrap gap-2">
                    {currentConfig.enabled_markets.map((market, idx) => (
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
                      {Object.entries(currentConfig.parameters).map(([key, value], idx) => (
                        <tr key={idx} className="border-b border-[#222]">
                          <td className="px-3 py-2 text-[#666]">{key.toUpperCase()}</td>
                          <td className="px-3 py-2 text-right text-[#00ff00]">{value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Risk Limits */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RISK LIMITS</div>
                <div className="border border-[#444] bg-[#0a0a0a]">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                        <th className="px-3 py-2 text-left border-b border-[#444]">LIMIT</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">VALUE</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-[#222]">
                        <td className="px-3 py-2 text-[#666]">MAX POSITION SIZE</td>
                        <td className="px-3 py-2 text-right text-[#ffff00]">{(currentConfig.risk_limits.max_position_size * 100).toFixed(0)}%</td>
                      </tr>
                      <tr className="border-b border-[#222]">
                        <td className="px-3 py-2 text-[#666]">MAX LEVERAGE</td>
                        <td className="px-3 py-2 text-right text-[#ffff00]">{currentConfig.risk_limits.max_leverage}x</td>
                      </tr>
                      <tr className="border-b border-[#222]">
                        <td className="px-3 py-2 text-[#666]">STOP LOSS</td>
                        <td className="px-3 py-2 text-right text-[#ff0000]">{(currentConfig.risk_limits.stop_loss_pct * 100).toFixed(1)}%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Nautilus Integration Status */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">NAUTILUS INTEGRATION</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  {currentConfig.nautilusTranslationId ? (
                    <div className="space-y-2 text-[10px]">
                      <div className="flex justify-between">
                        <span className="text-[#666]">Translation ID:</span>
                        <span className="text-[#00ff00]">{currentConfig.nautilusTranslationId}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">Status:</span>
                        <span className="text-[#00ff00]">READY FOR BACKTEST</span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-[#666] text-[10px]">
                      Not translated to Nautilus format yet. Click "Translate to Nautilus" to enable backtesting.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        );

      case 'CATALOG':
        if (!selectedBrowserStrategy) return <div className="text-[#666] p-4">Select a strategy from the catalog</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                {selectedBrowserStrategy.name}
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">DESCRIPTION</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <p className="text-[#fff] text-[10px]">{selectedBrowserStrategy.description}</p>
                </div>
              </div>

              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">SIGNAL LOGIC</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <p className="text-[#00ff00] text-[9px]">{selectedBrowserStrategy.signal_logic}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">ENTRY RULES</div>
                  <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                    <ul className="space-y-1">
                      {selectedBrowserStrategy.entry_rules.map((rule, i) => (
                        <li key={i} className="text-[#00ff00] text-[9px]">→ {rule}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">EXIT RULES</div>
                  <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                    <ul className="space-y-1">
                      {selectedBrowserStrategy.exit_rules.map((rule, i) => (
                        <li key={i} className="text-[#ffff00] text-[9px]">← {rule}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">STRENGTHS</div>
                  <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                    <ul className="space-y-1">
                      {selectedBrowserStrategy.strengths.map((s, i) => (
                        <li key={i} className="text-[#00ff00] text-[9px]">✓ {s}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">WEAKNESSES</div>
                  <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                    <ul className="space-y-1">
                      {selectedBrowserStrategy.weaknesses.map((w, i) => (
                        <li key={i} className="text-[#ff0000] text-[9px]">✗ {w}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">METRICS</div>
                <div className="border border-[#444] bg-[#0a0a0a]">
                  <div className="grid grid-cols-4 gap-4 p-3">
                    <div>
                      <div className="text-[#666] text-[9px]">Typical Sharpe</div>
                      <div className="text-[#00ff00] text-[14px]">{selectedBrowserStrategy.typical_sharpe.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">Max Drawdown</div>
                      <div className="text-[#ff0000] text-[14px]">{selectedBrowserStrategy.typical_max_dd.toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">Complexity</div>
                      <div style={{ color: getComplexityColor(selectedBrowserStrategy.complexity) }} className="text-[14px]">
                        {selectedBrowserStrategy.complexity.toUpperCase()}
                      </div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">Data Req</div>
                      <div style={{ color: getDataReqColor(selectedBrowserStrategy.data_requirements) }} className="text-[14px]">
                        {selectedBrowserStrategy.data_requirements.toUpperCase()}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        );

      case 'TRANSLATIONS':
        if (!selectedTranslation) return <div className="text-[#666] p-4">Select a translation to view details</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                TRANSLATION: {selectedTranslation.f6StrategyName}
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">STATUS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="flex items-center gap-4 mb-3">
                    <span className={`text-[12px] ${getStatusColor(selectedTranslation.status)}`}>
                      {selectedTranslation.status}
                    </span>
                    <span className="text-[#666] text-[10px]">
                      Progress: {selectedTranslation.progress}%
                    </span>
                  </div>
                  <div className="h-2 bg-[#222]">
                    <div
                      className="h-full bg-[#ff8c00]"
                      style={{ width: `${selectedTranslation.progress}%` }}
                    />
                  </div>
                </div>
              </div>

              {selectedTranslation.nautilusStrategyCode && (
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">GENERATED CODE</div>
                  <div className="border border-[#444] p-3 bg-[#000] overflow-x-auto">
                    <pre className="text-[#00ff00] text-[9px] whitespace-pre-wrap">
                      {selectedTranslation.nautilusStrategyCode}
                    </pre>
                  </div>
                </div>
              )}

              {selectedTranslation.validationResults && (
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">VALIDATION RESULTS</div>
                  <div className="border border-[#444] bg-[#0a0a0a]">
                    {selectedTranslation.validationResults.map((result, idx) => (
                      <div key={idx} className="p-2 border-b border-[#222] last:border-0">
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] ${
                            result.severity === 'INFO' ? 'text-[#00ff00]' :
                            result.severity === 'WARNING' ? 'text-[#ffff00]' :
                            'text-[#ff0000]'
                          }`}>
                            [{result.severity}]
                          </span>
                          <span className="text-[#666] text-[10px]">{result.type}:</span>
                          <span className="text-[#fff] text-[10px]">{result.message}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </>
        );

      case 'BACKTESTS':
        if (!selectedBacktest) return <div className="text-[#666] p-4">Select a backtest to view results</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                BACKTEST: {selectedBacktest.config.name}
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">STATUS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="flex items-center gap-4 mb-3">
                    <span className={`text-[12px] ${getStatusColor(selectedBacktest.status)}`}>
                      {selectedBacktest.status}
                    </span>
                    <span className="text-[#666] text-[10px]">
                      Progress: {selectedBacktest.progress}%
                    </span>
                    {selectedBacktest.duration && (
                      <span className="text-[#666] text-[10px]">
                        Duration: {formatDuration(selectedBacktest.duration)}
                      </span>
                    )}
                  </div>
                  <div className="h-2 bg-[#222]">
                    <div
                      className="h-full bg-[#ff8c00]"
                      style={{ width: `${selectedBacktest.progress}%` }}
                    />
                  </div>
                </div>
              </div>

              {selectedBacktest.metrics && (
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">PERFORMANCE METRICS</div>
                  <div className="border border-[#444] bg-[#0a0a0a]">
                    <div className="grid grid-cols-4 gap-4 p-3">
                      <div>
                        <div className="text-[#666] text-[9px]">Total Return</div>
                        <div className={`text-[14px] ${selectedBacktest.metrics.totalReturn >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                          {formatPnl(selectedBacktest.metrics.totalReturn)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Sharpe Ratio</div>
                        <div className="text-[#00ff00] text-[14px]">{selectedBacktest.metrics.sharpeRatio.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Max Drawdown</div>
                        <div className="text-[#ff0000] text-[14px]">{selectedBacktest.metrics.maxDrawdown.toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Win Rate</div>
                        <div className="text-[#00ff00] text-[14px]">{(selectedBacktest.metrics.winRate * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-4 gap-4 p-3 border-t border-[#222]">
                      <div>
                        <div className="text-[#666] text-[9px]">Total Trades</div>
                        <div className="text-[#fff] text-[14px]">{selectedBacktest.metrics.totalTrades}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Profit Factor</div>
                        <div className="text-[#00ff00] text-[14px]">{selectedBacktest.metrics.profitFactor.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Sortino</div>
                        <div className="text-[#00ff00] text-[14px]">{selectedBacktest.metrics.sortinoRatio.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Calmar</div>
                        <div className="text-[#00ff00] text-[14px]">{selectedBacktest.metrics.calmarRatio.toFixed(2)}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">CONFIGURATION</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Initial Capital:</span>
                    <span className="text-[#00ff00]">${selectedBacktest.config.initialCapital.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Date Range:</span>
                    <span className="text-[#fff]">{selectedBacktest.config.dataRange.startDate} to {selectedBacktest.config.dataRange.endDate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Venue:</span>
                    <span className="text-[#fff]">{selectedBacktest.config.venue}</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        );

      case 'TEMPLATES':
        return (
          <div className="p-4">
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                STRATEGY TEMPLATES
              </div>
            </div>
            <div className="text-[#666] text-[10px]">
              Select a template from the left panel to create a new strategy configuration.
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel - Category Selection & List */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          {/* Category Selection */}
          <div className="mb-4">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">CATEGORY</div>
            <div className="space-y-1">
              {categories.map((cat) => (
                <button
                  key={cat.key}
                  onClick={() => {
                    setSelectedCategory(cat.key);
                    setSelectedStrategy(null);
                    setSelectedBrowserStrategy(null);
                    setSelectedTranslation(null);
                    setSelectedBacktest(null);
                  }}
                  className={`
                    w-full py-2 px-3 text-left text-[10px] border transition-colors
                    ${selectedCategory === cat.key
                      ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                      : 'border-[#333] text-[#666] hover:border-[#ff8c00] hover:text-[#ff8c00]'
                    }
                  `}
                >
                  {cat.label} ({cat.count})
                </button>
              ))}
            </div>
          </div>

          {/* List Content */}
          <div>
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">
              {categories.find(c => c.key === selectedCategory)?.label}
            </div>
            {renderLeftPanelContent()}
          </div>
        </div>
      </div>

      {/* Center Panel - View */}
      <div className="flex-1 overflow-y-auto p-4">
        {renderCenterPanelContent()}
      </div>

      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">STRATEGY ACTIONS</div>

          {/* Action Status */}
          {actionStatus && (
            <div className="mb-4 p-2 border border-[#ffff00] bg-[#1a1a1a] text-[#ffff00] text-[10px]">
              {actionStatus}
            </div>
          )}

          {/* Category-specific actions */}
          {selectedCategory === 'DEPLOYED' && currentConfig && (
            <>
              <div className="space-y-2 mb-6">
                <button
                  onClick={() => handleActivateStrategy(currentConfig.id)}
                  disabled={currentConfig.status === 'ACTIVE'}
                  className={`w-full py-2 px-3 border text-[10px] transition-colors ${
                    currentConfig.status === 'ACTIVE'
                      ? 'border-[#333] text-[#333] cursor-not-allowed'
                      : 'border-[#00ff00] text-[#00ff00] hover:bg-[#00ff00] hover:text-black'
                  }`}
                >
                  ACTIVATE STRATEGY
                </button>
                <button
                  onClick={() => handlePauseStrategy(currentConfig.id)}
                  disabled={currentConfig.status === 'PAUSED'}
                  className={`w-full py-2 px-3 border text-[10px] transition-colors ${
                    currentConfig.status === 'PAUSED'
                      ? 'border-[#333] text-[#333] cursor-not-allowed'
                      : 'border-[#ffff00] text-[#ffff00] hover:bg-[#ffff00] hover:text-black'
                  }`}
                >
                  PAUSE STRATEGY
                </button>
                <button
                  onClick={() => handleStopStrategy(currentConfig.id)}
                  disabled={currentConfig.status === 'STOPPED'}
                  className={`w-full py-2 px-3 border text-[10px] transition-colors ${
                    currentConfig.status === 'STOPPED'
                      ? 'border-[#333] text-[#333] cursor-not-allowed'
                      : 'border-[#ff0000] text-[#ff0000] hover:bg-[#ff0000] hover:text-black'
                  }`}
                >
                  STOP STRATEGY
                </button>
              </div>

              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">CONFIGURATION</div>
              <div className="space-y-2 mb-6">
                <button
                  onClick={() => {
                    setEditingConfig(currentConfig);
                    setShowConfigModal(true);
                  }}
                  className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                >
                  EDIT PARAMETERS
                </button>
                <button
                  onClick={() => setShowConfigModal(true)}
                  className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                >
                  ADJUST RISK LIMITS
                </button>
                <button
                  onClick={() => handleRunBacktest(currentConfig.strategy_id)}
                  className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                >
                  VIEW BACKTEST RESULTS
                </button>
              </div>

              <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">NAUTILUS INTEGRATION</div>
              <div className="space-y-2">
                <button
                  onClick={() => handleTranslateStrategy(currentConfig.strategy_id)}
                  disabled={isLoading}
                  className="w-full py-2 px-3 border border-[#ff8c00] text-[10px] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors disabled:opacity-50"
                >
                  {isLoading ? 'TRANSLATING...' : 'TRANSLATE TO NAUTILUS'}
                </button>
                <button
                  onClick={() => setShowNewBacktestModal(true)}
                  className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                >
                  CREATE NAUTILUS BACKTEST
                </button>
                <button
                  onClick={() => setShowDeployModal(true)}
                  className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
                >
                  DEPLOY TO LIVE TRADING
                </button>
              </div>
            </>
          )}

          {selectedCategory === 'CATALOG' && selectedBrowserStrategy && (
            <div className="space-y-2">
              <button
                onClick={() => {
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
                  setSelectedCategory('DEPLOYED');
                }}
                className="w-full py-2 px-3 bg-[#ff8c00] text-black text-[10px] hover:bg-[#ffa500] transition-colors"
              >
                USE THIS STRATEGY
              </button>
              <button
                onClick={() => handleTranslateStrategy(selectedBrowserStrategy.id)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[10px] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                TRANSLATE TO NAUTILUS
              </button>
              <button
                onClick={() => setShowStrategyBrowser(true)}
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                BROWSE ALL STRATEGIES
              </button>
            </div>
          )}

          {selectedCategory === 'TRANSLATIONS' && selectedTranslation && (
            <div className="space-y-2">
              <button
                onClick={() => setShowNewBacktestModal(true)}
                disabled={selectedTranslation.status !== StrategyTranslationStatus.COMPLETED}
                className="w-full py-2 px-3 bg-[#ff8c00] text-black text-[10px] hover:bg-[#ffa500] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                CREATE BACKTEST
              </button>
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                VIEW CODE
              </button>
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                RE-TRANSLATE
              </button>
            </div>
          )}

          {selectedCategory === 'BACKTESTS' && selectedBacktest && (
            <div className="space-y-2">
              <button
                disabled={selectedBacktest.status !== BacktestStatus.COMPLETED}
                className="w-full py-2 px-3 bg-[#00ff00] text-black text-[10px] hover:bg-[#00ff00]/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                DEPLOY TO LIVE
              </button>
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                EXPORT RESULTS
              </button>
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                VIEW TRADE LOG
              </button>
              <button
                onClick={() => setShowNewBacktestModal(true)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[10px] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                NEW BACKTEST
              </button>
            </div>
          )}

          {/* Quick Actions */}
          <div className="mt-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">QUICK ACTIONS</div>
            <div className="space-y-2">
              <button
                onClick={() => setShowStrategyBrowser(true)}
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors"
              >
                BROWSE STRATEGY CATALOG
              </button>
              <button
                onClick={() => setShowNewBacktestModal(true)}
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors"
              >
                NEW BACKTEST
              </button>
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors"
              >
                VIEW AUDIT LOG
              </button>
            </div>
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

                    <div className="pt-4 space-y-2">
                      <button
                        onClick={() => {
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
                          setSelectedCategory('DEPLOYED');
                        }}
                        className="w-full px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
                      >
                        USE THIS STRATEGY
                      </button>
                      <button
                        onClick={() => {
                          handleTranslateStrategy(selectedBrowserStrategy.id);
                          setShowStrategyBrowser(false);
                        }}
                        className="w-full px-4 py-2 border border-[#ff8c00] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black text-[10px]"
                      >
                        TRANSLATE TO NAUTILUS
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
                </div>
              </div>
            </div>

            <div className="flex gap-2">
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
                onClick={() => {
                  setShowConfigModal(false);
                  setEditingConfig(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.currentTarget);
                const updates = {
                  parameters: {
                    fast_period: parseInt(formData.get('fast_period') as string) || 20,
                    slow_period: parseInt(formData.get('slow_period') as string) || 100,
                    vol_threshold: parseFloat(formData.get('vol_threshold') as string) || 0.01,
                  },
                  risk_limits: {
                    max_position_size: parseFloat(formData.get('max_position_size') as string) || 0.20,
                    max_leverage: parseFloat(formData.get('max_leverage') as string) || 2.0,
                    stop_loss_pct: parseFloat(formData.get('stop_loss_pct') as string) || 0.03,
                  },
                };

                if (editingConfig) {
                  updateStrategyConfig(editingConfig.id, updates);
                }
                setShowConfigModal(false);
                setEditingConfig(null);
              }}
            >
              <div className="space-y-4">
                <div className="border border-[#444] p-4">
                  <div className="text-[#ff8c00] text-[10px] mb-3">PARAMETERS</div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[#666] block mb-1 text-[9px]">FAST PERIOD</label>
                      <input
                        type="number"
                        name="fast_period"
                        defaultValue={editingConfig?.parameters.fast_period || 20}
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                    <div>
                      <label className="text-[#666] block mb-1 text-[9px]">SLOW PERIOD</label>
                      <input
                        type="number"
                        name="slow_period"
                        defaultValue={editingConfig?.parameters.slow_period || 100}
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                    <div>
                      <label className="text-[#666] block mb-1 text-[9px]">VOL THRESHOLD</label>
                      <input
                        type="number"
                        step="0.001"
                        name="vol_threshold"
                        defaultValue={editingConfig?.parameters.vol_threshold || 0.01}
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                  </div>
                </div>

                <div className="border border-[#444] p-4">
                  <div className="text-[#ff8c00] text-[10px] mb-3">RISK LIMITS</div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[#666] block mb-1 text-[9px]">MAX POSITION SIZE</label>
                      <input
                        type="number"
                        step="0.01"
                        name="max_position_size"
                        defaultValue={editingConfig?.risk_limits.max_position_size || 0.20}
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                    <div>
                      <label className="text-[#666] block mb-1 text-[9px]">MAX LEVERAGE</label>
                      <input
                        type="number"
                        step="0.1"
                        name="max_leverage"
                        defaultValue={editingConfig?.risk_limits.max_leverage || 2.0}
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                    <div>
                      <label className="text-[#666] block mb-1 text-[9px]">STOP LOSS %</label>
                      <input
                        type="number"
                        step="0.001"
                        name="stop_loss_pct"
                        defaultValue={editingConfig?.risk_limits.stop_loss_pct || 0.03}
                        className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex gap-2 mt-6">
                <button
                  type="button"
                  onClick={() => {
                    setShowConfigModal(false);
                    setEditingConfig(null);
                  }}
                  className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px]"
                >
                  SAVE CONFIGURATION
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* New Backtest Modal */}
      {showNewBacktestModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">CREATE NAUTILUS BACKTEST</h2>
              <button
                onClick={() => setShowNewBacktestModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.currentTarget);
                const config: BacktestConfig = {
                  name: formData.get('name') as string,
                  description: formData.get('description') as string,
                  strategies: [],
                  dataRange: {
                    startDate: formData.get('startDate') as string,
                    endDate: formData.get('endDate') as string,
                  },
                  initialCapital: parseFloat(formData.get('initialCapital') as string),
                  baseCurrency: 'USD',
                  venue: formData.get('venue') as string,
                  riskConfig: {
                    maxDrawdown: parseFloat(formData.get('maxDrawdown') as string),
                    maxLeverage: parseFloat(formData.get('maxLeverage') as string),
                    positionLimits: {},
                    stopLossEnabled: true,
                    killSwitchEnabled: true,
                  },
                  executionConfig: {
                    latencyMode: 'MEDIUM',
                    slippageModel: 'LINEAR',
                    commissionModel: 'PERCENTAGE',
                    fillModel: 'REALISTIC',
                  },
                  outputConfig: {
                    generateReports: true,
                    saveTradeLog: true,
                    savePositionLog: true,
                    saveMetrics: true,
                    exportFormat: 'JSON',
                  },
                };
                handleCreateBacktest(config);
              }}
            >
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-2">
                    <label className="text-[#666] block mb-1 text-[9px]">BACKTEST NAME</label>
                    <input
                      type="text"
                      name="name"
                      required
                      placeholder="e.g., MA Crossover Q1 2024"
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="text-[#666] block mb-1 text-[9px]">DESCRIPTION</label>
                    <textarea
                      name="description"
                      rows={2}
                      placeholder="Optional description..."
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">START DATE</label>
                    <input
                      type="date"
                      name="startDate"
                      required
                      defaultValue="2024-01-01"
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">END DATE</label>
                    <input
                      type="date"
                      name="endDate"
                      required
                      defaultValue="2024-12-31"
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">INITIAL CAPITAL</label>
                    <input
                      type="number"
                      name="initialCapital"
                      required
                      defaultValue={100000}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">VENUE</label>
                    <select
                      name="venue"
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    >
                      <option value="DERIV">DERIV</option>
                      <option value="MOCK">MOCK (Simulated)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">MAX DRAWDOWN %</label>
                    <input
                      type="number"
                      name="maxDrawdown"
                      required
                      defaultValue={20}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">MAX LEVERAGE</label>
                    <input
                      type="number"
                      step="0.1"
                      name="maxLeverage"
                      required
                      defaultValue={2.0}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                </div>
              </div>

              <div className="flex gap-2 mt-6">
                <button
                  type="button"
                  onClick={() => setShowNewBacktestModal(false)}
                  className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="flex-1 px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px] disabled:opacity-50"
                >
                  {isLoading ? 'CREATING...' : 'CREATE BACKTEST'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Translation Modal */}
      {showTranslationModal && selectedTranslation && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">STRATEGY TRANSLATION</h2>
              <button
                onClick={() => setShowTranslationModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px]">TRANSLATION STATUS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="flex items-center gap-4 mb-2">
                    <span className={`text-[12px] ${getStatusColor(selectedTranslation.status)}`}>
                      {selectedTranslation.status}
                    </span>
                    <span className="text-[#666] text-[10px]">
                      Progress: {selectedTranslation.progress}%
                    </span>
                  </div>
                  <div className="h-2 bg-[#222]">
                    <div
                      className="h-full bg-[#ff8c00]"
                      style={{ width: `${selectedTranslation.progress}%` }}
                    />
                  </div>
                </div>
              </div>

              {selectedTranslation.nautilusStrategyCode && (
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px]">GENERATED NAUTILUS CODE</div>
                  <div className="border border-[#444] p-3 bg-[#000] overflow-x-auto">
                    <pre className="text-[#00ff00] text-[9px] whitespace-pre-wrap font-mono">
                      {selectedTranslation.nautilusStrategyCode}
                    </pre>
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setShowTranslationModal(false)}
                className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
              >
                CLOSE
              </button>
              <button
                disabled={selectedTranslation.status !== StrategyTranslationStatus.COMPLETED}
                onClick={() => {
                  setShowTranslationModal(false);
                  setShowNewBacktestModal(true);
                }}
                className="flex-1 px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                CREATE BACKTEST
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
