import { useState, useEffect } from 'react';
import {
  listAssets,
  getTick,
  getOptionsChain,
  priceOption,
  priceFutures,
  priceStructuredProduct,
  startBacktest,
  listBacktests,
  getBacktestStatus,
  getStrategyTemplates,
  getProductTemplates,
  getMockAssets,
  getMockTick,
  getMockOptionsChain,
  type AssetInfo,
  type MarketTick,
  type OptionsChain,
  type OptionPriceResult,
  type FuturesPrice,
  type StructuredProductResult,
  type BacktestStatus,
  type StrategyTemplate,
  type ProductTemplate,
} from '../../services/derivativesService';

type DerivativesCategory = 'ASSETS' | 'OPTIONS' | 'FUTURES' | 'STRUCTURED' | 'BACKTEST';

export function Derivatives() {
  // Category state
  const [selectedCategory, setSelectedCategory] = useState<DerivativesCategory>('ASSETS');

  // Data state
  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<AssetInfo | null>(null);
  const [currentTick, setCurrentTick] = useState<MarketTick | null>(null);
  const [optionsChain, setOptionsChain] = useState<OptionsChain | null>(null);
  const [optionResult, setOptionResult] = useState<OptionPriceResult | null>(null);
  const [futuresResult, setFuturesResult] = useState<FuturesPrice | null>(null);
  const [structuredResult, setStructuredResult] = useState<StructuredProductResult | null>(null);
  const [backtests, setBacktests] = useState<BacktestStatus[]>([]);
  const [selectedBacktest, setSelectedBacktest] = useState<BacktestStatus | null>(null);
  const [strategyTemplates, setStrategyTemplates] = useState<StrategyTemplate[]>([]);
  const [productTemplates, setProductTemplates] = useState<ProductTemplate[]>([]);

  // Loading state
  const [isLoading, setIsLoading] = useState(false);
  const [useMockData, setUseMockData] = useState(false);

  // Form state
  const [optionForm, setOptionForm] = useState({
    strike: '',
    expiry: '',
    optionType: 'call' as 'call' | 'put',
    volatility: '0.2',
  });
  const [futuresForm, setFuturesForm] = useState({
    expiry: '',
    convenienceYield: '0',
    storageCost: '0',
  });
  const [structuredForm, setStructuredForm] = useState({
    productType: 'straddle' as 'straddle' | 'strangle' | 'butterfly' | 'iron_condor',
    expiry: '',
    strike: '',
    volatility: '0.2',
  });
  const [backtestForm, setBacktestForm] = useState({
    name: '',
    strategyType: 'covered_call' as 'covered_call' | 'iron_condor',
    startDate: '',
    endDate: '',
    initialCapital: '100000',
  });

  const formatNumber = (num: number, decimals = 4) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPercent = (num: number) => {
    const sign = num >= 0 ? '+' : '';
    return `${sign}${(num * 100).toFixed(2)}%`;
  };

  // Initialize data
  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const data = await listAssets();
        setAssets(data);
        setUseMockData(false);
      } catch {
        console.warn('Using mock assets data');
        setAssets(getMockAssets());
        setUseMockData(true);
      }
    };
    fetchAssets();
  }, []);

  // Fetch tick when asset selected
  useEffect(() => {
    if (!selectedAsset) return;

    const fetchTick = async () => {
      try {
        const tick = await getTick(selectedAsset.symbol);
        setCurrentTick(tick);
      } catch {
        setCurrentTick(getMockTick(selectedAsset.symbol));
      }
    };

    fetchTick();
    const interval = setInterval(fetchTick, 5000);
    return () => clearInterval(interval);
  }, [selectedAsset]);

  // Fetch options chain
  useEffect(() => {
    if (!selectedAsset || !currentTick || selectedCategory !== 'OPTIONS') return;

    const fetchChain = async () => {
      try {
        const expiry = new Date();
        expiry.setDate(expiry.getDate() + 30);
        const chain = await getOptionsChain({
          underlying: selectedAsset.symbol,
          spot_price: currentTick.last,
          expiry_date: expiry.toISOString().split('T')[0],
        });
        setOptionsChain(chain);
      } catch {
        setOptionsChain(getMockOptionsChain(selectedAsset.symbol, currentTick.last));
      }
    };

    fetchChain();
  }, [selectedAsset, currentTick, selectedCategory]);

  // Fetch backtests
  useEffect(() => {
    if (selectedCategory !== 'BACKTEST') return;

    const fetchBacktests = async () => {
      try {
        const data = await listBacktests();
        setBacktests(data);
      } catch {
        console.warn('Failed to fetch backtests');
      }
    };

    const fetchTemplates = async () => {
      try {
        const data = await getStrategyTemplates();
        setStrategyTemplates(data.strategies);
      } catch {
        console.warn('Failed to fetch strategy templates');
      }
    };

    fetchBacktests();
    fetchTemplates();
  }, [selectedCategory]);

  // Price option
  const handlePriceOption = async () => {
    if (!selectedAsset || !currentTick) return;
    setIsLoading(true);
    try {
      const result = await priceOption({
        underlying: selectedAsset.symbol,
        spot_price: currentTick.last,
        strike: parseFloat(optionForm.strike) || currentTick.last,
        expiry_date: optionForm.expiry || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        option_type: optionForm.optionType,
        volatility: parseFloat(optionForm.volatility),
      });
      setOptionResult(result);
    } catch (e) {
      console.error('Failed to price option:', e);
    } finally {
      setIsLoading(false);
    }
  };

  // Price futures
  const handlePriceFutures = async () => {
    if (!selectedAsset || !currentTick) return;
    setIsLoading(true);
    try {
      const result = await priceFutures({
        underlying: selectedAsset.symbol,
        spot_price: currentTick.last,
        expiry_date: futuresForm.expiry || new Date(Date.now() + 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        convenience_yield: parseFloat(futuresForm.convenienceYield),
        storage_cost: parseFloat(futuresForm.storageCost),
      });
      setFuturesResult(result);
    } catch (e) {
      console.error('Failed to price futures:', e);
    } finally {
      setIsLoading(false);
    }
  };

  // Price structured product
  const handlePriceStructured = async () => {
    if (!selectedAsset || !currentTick) return;
    setIsLoading(true);
    try {
      const result = await priceStructuredProduct({
        product_type: structuredForm.productType,
        underlying: selectedAsset.symbol,
        spot_price: currentTick.last,
        volatility: parseFloat(structuredForm.volatility),
        expiry_date: structuredForm.expiry || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        strike: parseFloat(structuredForm.strike) || currentTick.last,
      });
      setStructuredResult(result);
    } catch (e) {
      console.error('Failed to price structured product:', e);
    } finally {
      setIsLoading(false);
    }
  };

  // Run backtest
  const handleRunBacktest = async () => {
    if (!selectedAsset) return;
    setIsLoading(true);
    try {
      const result = await startBacktest({
        name: backtestForm.name || `Backtest ${selectedAsset.symbol}`,
        strategy_type: backtestForm.strategyType,
        underlying: selectedAsset.symbol,
        start_date: backtestForm.startDate || new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        end_date: backtestForm.endDate || new Date().toISOString().split('T')[0],
        initial_capital: parseFloat(backtestForm.initialCapital),
      });
      setBacktests([result, ...backtests]);
      setSelectedBacktest(result);
    } catch (e) {
      console.error('Failed to start backtest:', e);
    } finally {
      setIsLoading(false);
    }
  };

  const categories: { key: DerivativesCategory; label: string }[] = [
    { key: 'ASSETS', label: 'ASSET SELECTION' },
    { key: 'OPTIONS', label: 'OPTIONS PRICING' },
    { key: 'FUTURES', label: 'FUTURES PRICING' },
    { key: 'STRUCTURED', label: 'STRUCTURED PRODUCTS' },
    { key: 'BACKTEST', label: 'BACKTESTING' },
  ];

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel - Category Selection */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">DERIVATIVES TRADING</div>

          {/* Category Buttons */}
          <div className="space-y-2 mb-4">
            {categories.map(cat => (
              <button
                key={cat.key}
                onClick={() => setSelectedCategory(cat.key)}
                className={`w-full py-2 px-3 border text-left text-[10px] transition-colors ${
                  selectedCategory === cat.key
                    ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                    : 'border-[#444] text-[#00ff00] hover:border-[#ff8c00]'
                }`}
              >
                {cat.label}
              </button>
            ))}
          </div>

          {/* Asset List */}
          <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">AVAILABLE ASSETS</div>
          <div className="space-y-1">
            {assets.map(asset => (
              <div
                key={asset.symbol}
                onClick={() => setSelectedAsset(asset)}
                className={`border p-2 cursor-pointer transition-colors ${
                  selectedAsset?.symbol === asset.symbol
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="text-[#00ff00]">{asset.symbol}</span>
                  <span className="text-[#666] text-[9px]">{asset.asset_class.toUpperCase()}</span>
                </div>
                <div className="text-[#666] text-[9px]">{asset.name}</div>
              </div>
            ))}
          </div>

          {useMockData && (
            <div className="mt-4 p-2 border border-[#ffff00] text-[#ffff00] text-[9px]">
              USING MOCK DATA - Backend offline
            </div>
          )}
        </div>
      </div>

      {/* Center Panel - Main View */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="flex items-center justify-between">
            <div className="text-[#ff8c00] text-sm tracking-wider">
              DERIVATIVES - {selectedCategory.replace('_', ' ')}
            </div>
            {isLoading && <span className="text-[#ffff00] text-[10px]">LOADING...</span>}
          </div>
        </div>

        {/* Current Price Display */}
        {selectedAsset && currentTick && (
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="border border-[#444] p-3">
              <div className="text-[#666] text-[9px]">SYMBOL</div>
              <div className="text-[#00ff00] text-lg">{selectedAsset.symbol}</div>
            </div>
            <div className="border border-[#444] p-3">
              <div className="text-[#666] text-[9px]">BID / ASK</div>
              <div className="text-[#fff] text-sm">
                {formatNumber(currentTick.bid)} / {formatNumber(currentTick.ask)}
              </div>
            </div>
            <div className="border border-[#444] p-3">
              <div className="text-[#666] text-[9px]">LAST</div>
              <div className="text-[#00ff00] text-lg">{formatNumber(currentTick.last)}</div>
            </div>
            <div className="border border-[#444] p-3">
              <div className="text-[#666] text-[9px]">SPREAD (BPS)</div>
              <div className="text-[#ffff00]">{formatNumber(currentTick.spread_bps, 1)}</div>
            </div>
          </div>
        )}

        {/* Options View */}
        {selectedCategory === 'OPTIONS' && selectedAsset && (
          <div className="space-y-4">
            {/* Options Chain */}
            {optionsChain && (
              <div className="border border-[#444]">
                <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                  <div className="text-[#ff8c00]">OPTIONS CHAIN - {optionsChain.expiry_date}</div>
                </div>
                <table className="w-full">
                  <thead>
                    <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                      <th className="px-2 py-2 border-b border-[#444]" colSpan={5}>CALLS</th>
                      <th className="px-2 py-2 border-b border-[#444] bg-[#1a1a1a]">STRIKE</th>
                      <th className="px-2 py-2 border-b border-[#444]" colSpan={5}>PUTS</th>
                    </tr>
                    <tr className="bg-[#0a0a0a] text-[#666] text-[9px]">
                      <th className="px-2 py-1">PRICE</th>
                      <th className="px-2 py-1">DELTA</th>
                      <th className="px-2 py-1">GAMMA</th>
                      <th className="px-2 py-1">THETA</th>
                      <th className="px-2 py-1">IV%</th>
                      <th className="px-2 py-1 bg-[#1a1a1a]"></th>
                      <th className="px-2 py-1">PRICE</th>
                      <th className="px-2 py-1">DELTA</th>
                      <th className="px-2 py-1">GAMMA</th>
                      <th className="px-2 py-1">THETA</th>
                      <th className="px-2 py-1">IV%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {optionsChain.chain.map((row, idx) => {
                      const isATM = Math.abs(row.strike - (currentTick?.last || 0)) < (currentTick?.last || 1) * 0.02;
                      return (
                        <tr key={idx} className={`border-b border-[#222] ${isATM ? 'bg-[#1a1a1a]' : ''}`}>
                          <td className="px-2 py-1 text-right text-[#00ff00]">{formatNumber(row.call.price, 2)}</td>
                          <td className="px-2 py-1 text-right">{formatNumber(row.call.delta, 2)}</td>
                          <td className="px-2 py-1 text-right">{formatNumber(row.call.gamma, 4)}</td>
                          <td className="px-2 py-1 text-right text-[#ff0000]">{formatNumber(row.call.theta, 2)}</td>
                          <td className="px-2 py-1 text-right text-[#ffff00]">{row.call.iv.toFixed(1)}</td>
                          <td className={`px-2 py-1 text-center font-bold ${isATM ? 'text-[#ff8c00]' : 'text-[#fff]'}`}>
                            {formatNumber(row.strike, 2)}
                          </td>
                          <td className="px-2 py-1 text-right text-[#00ff00]">{formatNumber(row.put.price, 2)}</td>
                          <td className="px-2 py-1 text-right">{formatNumber(row.put.delta, 2)}</td>
                          <td className="px-2 py-1 text-right">{formatNumber(row.put.gamma, 4)}</td>
                          <td className="px-2 py-1 text-right text-[#ff0000]">{formatNumber(row.put.theta, 2)}</td>
                          <td className="px-2 py-1 text-right text-[#ffff00]">{row.put.iv.toFixed(1)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Option Pricing Result */}
            {optionResult && (
              <div className="border border-[#00ff00] p-4">
                <div className="text-[#ff8c00] mb-3">OPTION PRICING RESULT</div>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <div className="text-[#666] text-[9px]">PRICE</div>
                    <div className="text-[#00ff00] text-lg">${formatNumber(optionResult.price, 2)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">DELTA</div>
                    <div className="text-[#fff]">{formatNumber(optionResult.greeks.delta, 4)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">GAMMA</div>
                    <div className="text-[#fff]">{formatNumber(optionResult.greeks.gamma, 6)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">THETA</div>
                    <div className="text-[#ff0000]">{formatNumber(optionResult.greeks.theta, 4)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">VEGA</div>
                    <div className="text-[#fff]">{formatNumber(optionResult.greeks.vega, 4)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">INTRINSIC</div>
                    <div className="text-[#fff]">${formatNumber(optionResult.intrinsic_value, 2)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">TIME VALUE</div>
                    <div className="text-[#fff]">${formatNumber(optionResult.time_value, 2)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">PROB ITM</div>
                    <div className="text-[#ffff00]">{formatPercent(optionResult.probability_itm)}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Futures View */}
        {selectedCategory === 'FUTURES' && selectedAsset && futuresResult && (
          <div className="border border-[#00ff00] p-4">
            <div className="text-[#ff8c00] mb-3">FUTURES PRICING RESULT</div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-[#666] text-[9px]">SPOT PRICE</div>
                <div className="text-[#fff] text-lg">{formatNumber(futuresResult.spot_price)}</div>
              </div>
              <div>
                <div className="text-[#666] text-[9px]">FAIR VALUE</div>
                <div className="text-[#00ff00] text-lg">{formatNumber(futuresResult.fair_value)}</div>
              </div>
              <div>
                <div className="text-[#666] text-[9px]">BASIS</div>
                <div className={`text-lg ${futuresResult.basis >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                  {formatNumber(futuresResult.basis)} ({futuresResult.basis_pct.toFixed(2)}%)
                </div>
              </div>
              <div>
                <div className="text-[#666] text-[9px]">IMPLIED REPO RATE</div>
                <div className="text-[#ffff00]">{(futuresResult.implied_repo_rate * 100).toFixed(2)}%</div>
              </div>
              <div>
                <div className="text-[#666] text-[9px]">TIME TO EXPIRY</div>
                <div className="text-[#fff]">{futuresResult.time_to_expiry_years.toFixed(3)} years</div>
              </div>
              <div>
                <div className="text-[#666] text-[9px]">RISK-FREE RATE</div>
                <div className="text-[#fff]">{(futuresResult.risk_free_rate * 100).toFixed(2)}%</div>
              </div>
            </div>
          </div>
        )}

        {/* Structured Products View */}
        {selectedCategory === 'STRUCTURED' && selectedAsset && structuredResult && (
          <div className="space-y-4">
            <div className="border border-[#00ff00] p-4">
              <div className="text-[#ff8c00] mb-3">{structuredResult.product_name}</div>
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div>
                  <div className="text-[#666] text-[9px]">TOTAL PRICE</div>
                  <div className="text-[#00ff00] text-lg">${formatNumber(structuredResult.total_price, 2)}</div>
                </div>
                <div>
                  <div className="text-[#666] text-[9px]">NET DELTA</div>
                  <div className="text-[#fff]">{formatNumber(structuredResult.net_greeks.delta, 4)}</div>
                </div>
                <div>
                  <div className="text-[#666] text-[9px]">MAX PROFIT</div>
                  <div className="text-[#00ff00]">${formatNumber(structuredResult.max_profit, 2)}</div>
                </div>
                <div>
                  <div className="text-[#666] text-[9px]">MAX LOSS</div>
                  <div className="text-[#ff0000]">${formatNumber(Math.abs(structuredResult.max_loss), 2)}</div>
                </div>
              </div>

              {/* Legs */}
              <div className="text-[#ff8c00] mb-2 text-[10px]">LEGS</div>
              <div className="space-y-1">
                {structuredResult.legs.map((leg, idx) => (
                  <div key={idx} className="flex justify-between border border-[#333] p-2">
                    <span className="text-[#00ff00]">
                      {leg.quantity > 0 ? 'LONG' : 'SHORT'} {Math.abs(leg.quantity)} {leg.option_type?.toUpperCase()} @ {leg.strike}
                    </span>
                    <span className="text-[#fff]">${formatNumber(leg.price, 2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Backtest View */}
        {selectedCategory === 'BACKTEST' && (
          <div className="space-y-4">
            {/* Backtest List */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">BACKTEST HISTORY</div>
              </div>
              <div className="p-3">
                {backtests.length === 0 ? (
                  <div className="text-[#666] text-center py-4">No backtests yet</div>
                ) : (
                  <div className="space-y-2">
                    {backtests.map(bt => (
                      <div
                        key={bt.id}
                        onClick={() => setSelectedBacktest(bt)}
                        className={`border p-2 cursor-pointer transition-colors ${
                          selectedBacktest?.id === bt.id
                            ? 'border-[#ff8c00] bg-[#1a1a1a]'
                            : 'border-[#333] hover:border-[#ff8c00]'
                        }`}
                      >
                        <div className="flex justify-between items-center">
                          <span className="text-[#00ff00]">{bt.name}</span>
                          <span className={`text-[9px] ${
                            bt.status === 'completed' ? 'text-[#00ff00]' :
                            bt.status === 'running' ? 'text-[#ffff00]' :
                            'text-[#ff0000]'
                          }`}>
                            {bt.status.toUpperCase()}
                          </span>
                        </div>
                        {bt.result && (
                          <div className="text-[#666] text-[9px] mt-1">
                            Return: {formatPercent(bt.result.performance.total_return)} |
                            Sharpe: {bt.result.performance.sharpe_ratio.toFixed(2)}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Selected Backtest Results */}
            {selectedBacktest?.result && (
              <div className="border border-[#00ff00] p-4">
                <div className="text-[#ff8c00] mb-3">BACKTEST RESULTS: {selectedBacktest.name}</div>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <div className="text-[#666] text-[9px]">TOTAL RETURN</div>
                    <div className={`text-lg ${selectedBacktest.result.performance.total_return >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                      {formatPercent(selectedBacktest.result.performance.total_return)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">SHARPE RATIO</div>
                    <div className="text-[#fff] text-lg">{selectedBacktest.result.performance.sharpe_ratio.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">MAX DRAWDOWN</div>
                    <div className="text-[#ff0000] text-lg">{formatPercent(-selectedBacktest.result.performance.max_drawdown)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">WIN RATE</div>
                    <div className="text-[#fff] text-lg">{formatPercent(selectedBacktest.result.performance.win_rate)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">TOTAL TRADES</div>
                    <div className="text-[#fff]">{selectedBacktest.result.trade_statistics.total_trades}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">PROFIT FACTOR</div>
                    <div className="text-[#fff]">{selectedBacktest.result.performance.profit_factor.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">CALMAR RATIO</div>
                    <div className="text-[#fff]">{selectedBacktest.result.performance.calmar_ratio.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-[#666] text-[9px]">SORTINO RATIO</div>
                    <div className="text-[#fff]">{selectedBacktest.result.performance.sortino_ratio.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {!selectedAsset && (
          <div className="text-center text-[#666] mt-20">
            SELECT AN ASSET FROM THE LEFT PANEL TO BEGIN
          </div>
        )}
      </div>

      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">DERIVATIVES ACTIONS</div>

          {/* Options Actions */}
          {selectedCategory === 'OPTIONS' && selectedAsset && (
            <div className="space-y-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">PRICE OPTION</div>
              <div>
                <label className="text-[#666] text-[9px]">STRIKE</label>
                <input
                  type="number"
                  value={optionForm.strike}
                  onChange={e => setOptionForm({ ...optionForm, strike: e.target.value })}
                  placeholder={currentTick?.last.toString()}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] text-[9px]">EXPIRY DATE</label>
                <input
                  type="date"
                  value={optionForm.expiry}
                  onChange={e => setOptionForm({ ...optionForm, expiry: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setOptionForm({ ...optionForm, optionType: 'call' })}
                  className={`py-1 border text-[10px] ${
                    optionForm.optionType === 'call'
                      ? 'border-[#00ff00] bg-[#00ff00] text-black'
                      : 'border-[#444] text-[#00ff00]'
                  }`}
                >
                  CALL
                </button>
                <button
                  onClick={() => setOptionForm({ ...optionForm, optionType: 'put' })}
                  className={`py-1 border text-[10px] ${
                    optionForm.optionType === 'put'
                      ? 'border-[#ff0000] bg-[#ff0000] text-black'
                      : 'border-[#444] text-[#ff0000]'
                  }`}
                >
                  PUT
                </button>
              </div>
              <div>
                <label className="text-[#666] text-[9px]">VOLATILITY</label>
                <input
                  type="number"
                  step="0.01"
                  value={optionForm.volatility}
                  onChange={e => setOptionForm({ ...optionForm, volatility: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <button
                onClick={handlePriceOption}
                disabled={isLoading}
                className="w-full py-2 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                CALCULATE PRICE
              </button>
            </div>
          )}

          {/* Futures Actions */}
          {selectedCategory === 'FUTURES' && selectedAsset && (
            <div className="space-y-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">PRICE FUTURES</div>
              <div>
                <label className="text-[#666] text-[9px]">EXPIRY DATE</label>
                <input
                  type="date"
                  value={futuresForm.expiry}
                  onChange={e => setFuturesForm({ ...futuresForm, expiry: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] text-[9px]">CONVENIENCE YIELD</label>
                <input
                  type="number"
                  step="0.01"
                  value={futuresForm.convenienceYield}
                  onChange={e => setFuturesForm({ ...futuresForm, convenienceYield: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] text-[9px]">STORAGE COST</label>
                <input
                  type="number"
                  step="0.01"
                  value={futuresForm.storageCost}
                  onChange={e => setFuturesForm({ ...futuresForm, storageCost: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <button
                onClick={handlePriceFutures}
                disabled={isLoading}
                className="w-full py-2 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                CALCULATE FAIR VALUE
              </button>
            </div>
          )}

          {/* Structured Product Actions */}
          {selectedCategory === 'STRUCTURED' && selectedAsset && (
            <div className="space-y-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">STRUCTURE PRODUCT</div>
              <div>
                <label className="text-[#666] text-[9px]">PRODUCT TYPE</label>
                <select
                  value={structuredForm.productType}
                  onChange={e => setStructuredForm({ ...structuredForm, productType: e.target.value as typeof structuredForm.productType })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="straddle">STRADDLE</option>
                  <option value="strangle">STRANGLE</option>
                  <option value="butterfly">BUTTERFLY</option>
                  <option value="iron_condor">IRON CONDOR</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] text-[9px]">STRIKE</label>
                <input
                  type="number"
                  value={structuredForm.strike}
                  onChange={e => setStructuredForm({ ...structuredForm, strike: e.target.value })}
                  placeholder={currentTick?.last.toString()}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] text-[9px]">EXPIRY</label>
                <input
                  type="date"
                  value={structuredForm.expiry}
                  onChange={e => setStructuredForm({ ...structuredForm, expiry: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <button
                onClick={handlePriceStructured}
                disabled={isLoading}
                className="w-full py-2 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                PRICE STRUCTURE
              </button>
            </div>
          )}

          {/* Backtest Actions */}
          {selectedCategory === 'BACKTEST' && selectedAsset && (
            <div className="space-y-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">RUN BACKTEST</div>
              <div>
                <label className="text-[#666] text-[9px]">NAME</label>
                <input
                  type="text"
                  value={backtestForm.name}
                  onChange={e => setBacktestForm({ ...backtestForm, name: e.target.value })}
                  placeholder={`Backtest ${selectedAsset.symbol}`}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] text-[9px]">STRATEGY</label>
                <select
                  value={backtestForm.strategyType}
                  onChange={e => setBacktestForm({ ...backtestForm, strategyType: e.target.value as typeof backtestForm.strategyType })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="covered_call">COVERED CALL</option>
                  <option value="iron_condor">IRON CONDOR</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] text-[9px]">START DATE</label>
                <input
                  type="date"
                  value={backtestForm.startDate}
                  onChange={e => setBacktestForm({ ...backtestForm, startDate: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] text-[9px]">END DATE</label>
                <input
                  type="date"
                  value={backtestForm.endDate}
                  onChange={e => setBacktestForm({ ...backtestForm, endDate: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] text-[9px]">INITIAL CAPITAL</label>
                <input
                  type="number"
                  value={backtestForm.initialCapital}
                  onChange={e => setBacktestForm({ ...backtestForm, initialCapital: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <button
                onClick={handleRunBacktest}
                disabled={isLoading}
                className="w-full py-2 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
              >
                RUN BACKTEST
              </button>
            </div>
          )}

          {/* Quick Stats */}
          <div className="mt-6 border border-[#444] p-3">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">QUICK STATS</div>
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">ASSETS</span>
                <span className="text-[#fff]">{assets.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">BACKTESTS</span>
                <span className="text-[#fff]">{backtests.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">DATA MODE</span>
                <span className={useMockData ? 'text-[#ffff00]' : 'text-[#00ff00]'}>
                  {useMockData ? 'MOCK' : 'LIVE'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
