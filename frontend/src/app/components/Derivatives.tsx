import { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import {
  listAssets,
  getTick,
  getOptionsChain,
  priceOption,
  priceFutures,
  priceStructuredProduct,
  startBacktest,
  listBacktests,
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
} from '../../services/derivativesService';

type DerivativesCategory = 'ASSETS' | 'OPTIONS' | 'FUTURES' | 'STRUCTURED' | 'BACKTEST';
type AnalysisTab = 'PRICING' | 'GREEKS' | 'PAYOFF' | 'SENSITIVITY' | 'COMPARISON';

export function Derivatives() {
  // Panel collapse state
  const [isLeftCollapsed, setIsLeftCollapsed] = useState(false);
  const [isRightCollapsed, setIsRightCollapsed] = useState(false);

  // Category state
  const [selectedCategory, setSelectedCategory] = useState<DerivativesCategory>('OPTIONS');
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>('PRICING');

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

  // Sensitivity analysis parameters
  const [sensitivityParams, setSensitivityParams] = useState({
    spotRange: 10, // percent
    volRange: 50, // percent of current vol
    timeDecay: 7, // days
  });

  const formatNumber = (num: number, decimals = 4) => {
    if (num === undefined || num === null || isNaN(num)) return '-';
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPercent = (num: number) => {
    if (num === undefined || num === null || isNaN(num)) return '-';
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
        if (data.length > 0) setSelectedAsset(data[0]);
      } catch {
        console.warn('Using mock assets data');
        const mockData = getMockAssets();
        setAssets(mockData);
        setUseMockData(true);
        if (mockData.length > 0) setSelectedAsset(mockData[0]);
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
    fetchBacktests();
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

  // Generate Greeks data for visualization
  const greeksChartData = useMemo(() => {
    if (!optionsChain || !currentTick) return [];
    return optionsChain.chain.map(row => ({
      strike: row.strike,
      callDelta: row.call.delta,
      putDelta: row.put.delta,
      gamma: row.call.gamma,
      callTheta: row.call.theta,
      putTheta: row.put.theta,
      vega: row.call.vega,
      callIV: row.call.iv,
      putIV: row.put.iv,
      isATM: Math.abs(row.strike - currentTick.last) < currentTick.last * 0.02,
    }));
  }, [optionsChain, currentTick]);

  // Generate Payoff diagram data
  const payoffData = useMemo(() => {
    if (!currentTick || !optionResult) return [];
    const spot = currentTick.last;
    const strike = parseFloat(optionForm.strike) || spot;
    const premium = optionResult.price;
    const isCall = optionForm.optionType === 'call';

    const data = [];
    const range = spot * 0.3;
    for (let price = spot - range; price <= spot + range; price += range / 50) {
      const intrinsic = isCall
        ? Math.max(0, price - strike)
        : Math.max(0, strike - price);
      const pnl = intrinsic - premium;
      data.push({
        spotPrice: price,
        payoff: intrinsic,
        pnl: pnl,
        breakeven: price === (isCall ? strike + premium : strike - premium) ? pnl : null,
      });
    }
    return data;
  }, [currentTick, optionResult, optionForm]);

  // Generate structured product payoff data
  const structuredPayoffData = useMemo(() => {
    if (!structuredResult || !structuredResult.payoff_diagram) return [];
    const { spots, payoffs } = structuredResult.payoff_diagram;
    return spots.map((spot: number, i: number) => ({
      spotPrice: spot,
      payoff: payoffs[i],
      pnl: payoffs[i] - structuredResult.total_price,
    }));
  }, [structuredResult]);

  // Generate sensitivity analysis data
  const sensitivityData = useMemo(() => {
    if (!currentTick || !optionResult) return { spotSensitivity: [], volSensitivity: [], timeSensitivity: [] };

    const spot = currentTick.last;
    const vol = parseFloat(optionForm.volatility);
    const delta = optionResult.greeks.delta;
    const gamma = optionResult.greeks.gamma;
    const vega = optionResult.greeks.vega;
    const theta = optionResult.greeks.theta;
    const basePrice = optionResult.price;

    // Spot sensitivity
    const spotSensitivity = [];
    for (let pctChange = -sensitivityParams.spotRange; pctChange <= sensitivityParams.spotRange; pctChange += 1) {
      const newSpot = spot * (1 + pctChange / 100);
      const dS = newSpot - spot;
      const priceChange = delta * dS + 0.5 * gamma * dS * dS;
      spotSensitivity.push({
        spotChange: pctChange,
        priceChange: priceChange,
        newPrice: basePrice + priceChange,
        pnlPct: (priceChange / basePrice) * 100,
      });
    }

    // Volatility sensitivity
    const volSensitivity = [];
    for (let pctChange = -sensitivityParams.volRange; pctChange <= sensitivityParams.volRange; pctChange += 5) {
      const dVol = vol * (pctChange / 100);
      const priceChange = vega * dVol;
      volSensitivity.push({
        volChange: pctChange,
        priceChange: priceChange,
        newPrice: basePrice + priceChange,
        newVol: (vol + dVol) * 100,
      });
    }

    // Time decay
    const timeSensitivity = [];
    for (let days = 0; days <= sensitivityParams.timeDecay; days++) {
      const priceDecay = theta * days;
      timeSensitivity.push({
        days: days,
        priceDecay: priceDecay,
        newPrice: basePrice + priceDecay,
        pctDecay: (priceDecay / basePrice) * 100,
      });
    }

    return { spotSensitivity, volSensitivity, timeSensitivity };
  }, [currentTick, optionResult, optionForm.volatility, sensitivityParams]);

  // Volatility smile data
  const volSmileData = useMemo(() => {
    if (!optionsChain || !currentTick) return [];
    return optionsChain.chain.map(row => ({
      strike: row.strike,
      moneyness: (row.strike / currentTick.last - 1) * 100,
      callIV: row.call.iv,
      putIV: row.put.iv,
      avgIV: (row.call.iv + row.put.iv) / 2,
    }));
  }, [optionsChain, currentTick]);

  // Greeks radar data for current option
  const greeksRadarData = useMemo(() => {
    if (!optionResult) return [];
    const g = optionResult.greeks;
    return [
      { metric: 'Delta', value: Math.abs(g.delta) * 100, fullMark: 100 },
      { metric: 'Gamma', value: Math.min(g.gamma * 1000, 100), fullMark: 100 },
      { metric: 'Theta', value: Math.min(Math.abs(g.theta) * 10, 100), fullMark: 100 },
      { metric: 'Vega', value: Math.min(g.vega * 5, 100), fullMark: 100 },
      { metric: 'Rho', value: Math.min(Math.abs(g.rho) * 20, 100), fullMark: 100 },
    ];
  }, [optionResult]);

  const categories: { key: DerivativesCategory; label: string; shortLabel: string }[] = [
    { key: 'ASSETS', label: 'ASSET SELECTION', shortLabel: 'AST' },
    { key: 'OPTIONS', label: 'OPTIONS', shortLabel: 'OPT' },
    { key: 'FUTURES', label: 'FUTURES', shortLabel: 'FUT' },
    { key: 'STRUCTURED', label: 'STRUCTURED', shortLabel: 'STR' },
    { key: 'BACKTEST', label: 'BACKTEST', shortLabel: 'BT' },
  ];

  const analysisTabs: { key: AnalysisTab; label: string }[] = [
    { key: 'PRICING', label: 'PRICING' },
    { key: 'GREEKS', label: 'GREEKS ANALYSIS' },
    { key: 'PAYOFF', label: 'PAYOFF DIAGRAM' },
    { key: 'SENSITIVITY', label: 'SENSITIVITY' },
    { key: 'COMPARISON', label: 'VOL SURFACE' },
  ];

  return (
    <div className="flex h-full font-mono text-xs bg-[#0a0a0a]">
      {/* Left Panel - Collapsible Category/Asset Selection */}
      <div className={`${isLeftCollapsed ? 'w-12' : 'w-56'} border-r border-[#444] bg-[#0a0a0a] overflow-y-auto transition-all duration-300 flex-shrink-0`}>
        <button
          onClick={() => setIsLeftCollapsed(!isLeftCollapsed)}
          className="w-full p-2 border-b border-[#444] text-[#ff8c00] hover:bg-[#1a1a1a] transition-colors flex items-center justify-center text-[10px]"
        >
          {isLeftCollapsed ? '»' : '« HIDE'}
        </button>

        {!isLeftCollapsed && (
          <div className="p-2">
            {/* Category Buttons */}
            <div className="space-y-1 mb-3">
              {categories.map(cat => (
                <button
                  key={cat.key}
                  onClick={() => setSelectedCategory(cat.key)}
                  className={`w-full py-1.5 px-2 border text-left text-[9px] transition-colors ${
                    selectedCategory === cat.key
                      ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                      : 'border-[#333] text-[#00ff00] hover:border-[#ff8c00]'
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>

            {/* Asset List */}
            <div className="text-[#ff8c00] mb-1 text-[9px] tracking-wider">ASSETS</div>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {assets.map(asset => (
                <div
                  key={asset.symbol}
                  onClick={() => setSelectedAsset(asset)}
                  className={`border p-1.5 cursor-pointer transition-colors ${
                    selectedAsset?.symbol === asset.symbol
                      ? 'border-[#ff8c00] bg-[#1a1a1a]'
                      : 'border-[#333] hover:border-[#ff8c00]'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="text-[#00ff00] text-[9px]">{asset.symbol}</span>
                    <span className="text-[#666] text-[8px]">{asset.asset_class.toUpperCase()}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Current Price */}
            {selectedAsset && currentTick && (
              <div className="mt-3 p-2 border border-[#444]">
                <div className="text-[#ff8c00] text-[8px] mb-1">CURRENT PRICE</div>
                <div className="text-[#00ff00] text-lg">{formatNumber(currentTick.last, 2)}</div>
                <div className="text-[#666] text-[8px]">
                  {formatNumber(currentTick.bid, 4)} / {formatNumber(currentTick.ask, 4)}
                </div>
              </div>
            )}

            {useMockData && (
              <div className="mt-2 p-1 border border-[#ffff00] text-[#ffff00] text-[8px] text-center">
                MOCK DATA
              </div>
            )}
          </div>
        )}

        {/* Collapsed view shows icons */}
        {isLeftCollapsed && (
          <div className="p-1 space-y-1">
            {categories.map(cat => (
              <button
                key={cat.key}
                onClick={() => { setSelectedCategory(cat.key); setIsLeftCollapsed(false); }}
                className={`w-full p-1.5 border text-[8px] transition-colors ${
                  selectedCategory === cat.key
                    ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                    : 'border-[#333] text-[#00ff00] hover:border-[#ff8c00]'
                }`}
                title={cat.label}
              >
                {cat.shortLabel}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Center Panel - Main Analysis Workspace */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Header with Analysis Tabs */}
        <div className="border-b border-[#444] bg-[#0d0d0d]">
          <div className="flex items-center justify-between px-3 py-2 border-b border-[#333]">
            <div className="text-[#ff8c00] text-sm tracking-wider flex items-center gap-2">
              <span>{selectedAsset?.symbol || 'SELECT ASSET'}</span>
              <span className="text-[#666]">|</span>
              <span className="text-[#00ff00]">{selectedCategory}</span>
              {isLoading && <span className="text-[#ffff00] text-[10px] animate-pulse">CALCULATING...</span>}
            </div>
            <div className={`px-2 py-0.5 border text-[9px] ${useMockData ? 'border-[#ffff00] text-[#ffff00]' : 'border-[#00ff00] text-[#00ff00]'}`}>
              {useMockData ? '● MOCK' : '● LIVE'}
            </div>
          </div>

          {/* Analysis tabs for OPTIONS */}
          {selectedCategory === 'OPTIONS' && (
            <div className="flex gap-1 px-2 py-1">
              {analysisTabs.map(tab => (
                <button
                  key={tab.key}
                  onClick={() => setAnalysisTab(tab.key)}
                  className={`px-3 py-1 text-[9px] border transition-colors ${
                    analysisTab === tab.key
                      ? 'border-[#ff8c00] bg-[#ff8c00] text-black'
                      : 'border-[#444] text-[#888] hover:text-[#ff8c00] hover:border-[#ff8c00]'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto p-3">
          {/* OPTIONS ANALYSIS */}
          {selectedCategory === 'OPTIONS' && selectedAsset && (
            <div className="space-y-4">
              {/* PRICING TAB */}
              {analysisTab === 'PRICING' && (
                <div className="space-y-4">
                  {/* Options Chain Table */}
                  {optionsChain && (
                    <div className="border border-[#444]">
                      <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444] flex justify-between items-center">
                        <span className="text-[#ff8c00]">OPTIONS CHAIN - {optionsChain.expiry_date}</span>
                        <span className="text-[#666] text-[9px]">SPOT: {formatNumber(optionsChain.spot_price, 2)} | IV: {(optionsChain.volatility * 100).toFixed(1)}%</span>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-[9px]">
                          <thead>
                            <tr className="bg-[#0a0a0a] text-[#ff8c00]">
                              <th className="px-2 py-1.5 border-b border-[#444]" colSpan={5}>CALLS</th>
                              <th className="px-2 py-1.5 border-b border-[#444] bg-[#1a1a1a]">STRIKE</th>
                              <th className="px-2 py-1.5 border-b border-[#444]" colSpan={5}>PUTS</th>
                            </tr>
                            <tr className="bg-[#0a0a0a] text-[#666]">
                              <th className="px-1 py-1">PRICE</th>
                              <th className="px-1 py-1">Δ</th>
                              <th className="px-1 py-1">Γ</th>
                              <th className="px-1 py-1">Θ</th>
                              <th className="px-1 py-1">IV%</th>
                              <th className="px-1 py-1 bg-[#1a1a1a]"></th>
                              <th className="px-1 py-1">PRICE</th>
                              <th className="px-1 py-1">Δ</th>
                              <th className="px-1 py-1">Γ</th>
                              <th className="px-1 py-1">Θ</th>
                              <th className="px-1 py-1">IV%</th>
                            </tr>
                          </thead>
                          <tbody>
                            {optionsChain.chain.map((row, idx) => {
                              const isATM = Math.abs(row.strike - (currentTick?.last || 0)) < (currentTick?.last || 1) * 0.02;
                              return (
                                <tr key={idx} className={`border-b border-[#222] hover:bg-[#1a1a1a] ${isATM ? 'bg-[#1a1a0a]' : ''}`}>
                                  <td className="px-1 py-1 text-right text-[#00ff00]">{formatNumber(row.call.price, 2)}</td>
                                  <td className="px-1 py-1 text-right">{formatNumber(row.call.delta, 2)}</td>
                                  <td className="px-1 py-1 text-right">{formatNumber(row.call.gamma, 4)}</td>
                                  <td className="px-1 py-1 text-right text-[#ff4444]">{formatNumber(row.call.theta, 2)}</td>
                                  <td className="px-1 py-1 text-right text-[#ffff00]">{row.call.iv.toFixed(1)}</td>
                                  <td className={`px-1 py-1 text-center font-bold bg-[#1a1a1a] ${isATM ? 'text-[#ff8c00]' : 'text-[#fff]'}`}>
                                    {formatNumber(row.strike, 2)}
                                  </td>
                                  <td className="px-1 py-1 text-right text-[#00ff00]">{formatNumber(row.put.price, 2)}</td>
                                  <td className="px-1 py-1 text-right">{formatNumber(row.put.delta, 2)}</td>
                                  <td className="px-1 py-1 text-right">{formatNumber(row.put.gamma, 4)}</td>
                                  <td className="px-1 py-1 text-right text-[#ff4444]">{formatNumber(row.put.theta, 2)}</td>
                                  <td className="px-1 py-1 text-right text-[#ffff00]">{row.put.iv.toFixed(1)}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Option Pricing Result */}
                  {optionResult && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="border border-[#00ff00] p-4">
                        <div className="text-[#ff8c00] mb-3 text-sm">PRICING RESULT</div>
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <div className="text-[#666] text-[9px]">OPTION PRICE</div>
                            <div className="text-[#00ff00] text-2xl">${formatNumber(optionResult.price, 2)}</div>
                          </div>
                          <div>
                            <div className="text-[#666] text-[9px]">BREAKEVEN</div>
                            <div className="text-[#fff] text-lg">{formatNumber(optionResult.breakeven, 2)}</div>
                          </div>
                          <div>
                            <div className="text-[#666] text-[9px]">INTRINSIC VALUE</div>
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
                          <div>
                            <div className="text-[#666] text-[9px]">MODEL</div>
                            <div className="text-[#888]">{optionResult.pricing_model}</div>
                          </div>
                        </div>
                      </div>

                      {/* Greeks Radar Chart */}
                      <div className="border border-[#444] p-3">
                        <div className="text-[#ff8c00] mb-2 text-[10px]">GREEKS PROFILE</div>
                        <ResponsiveContainer width="100%" height={200}>
                          <RadarChart data={greeksRadarData}>
                            <PolarGrid stroke="#333" />
                            <PolarAngleAxis dataKey="metric" tick={{ fill: '#888', fontSize: 9 }} />
                            <PolarRadiusAxis tick={{ fill: '#666', fontSize: 8 }} domain={[0, 100]} />
                            <Radar name="Greeks" dataKey="value" stroke="#ff8c00" fill="#ff8c00" fillOpacity={0.3} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* GREEKS TAB */}
              {analysisTab === 'GREEKS' && greeksChartData.length > 0 && (
                <div className="grid grid-cols-2 gap-4">
                  {/* Delta Chart */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">DELTA (Δ) ACROSS STRIKES</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={greeksChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="strike" tick={{ fill: '#888', fontSize: 8 }} />
                        <YAxis tick={{ fill: '#888', fontSize: 8 }} domain={[-1, 1]} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                          labelStyle={{ color: '#ff8c00' }}
                        />
                        <ReferenceLine y={0} stroke="#444" />
                        {currentTick && <ReferenceLine x={currentTick.last} stroke="#ff8c00" strokeDasharray="5 5" label={{ value: 'ATM', fill: '#ff8c00', fontSize: 8 }} />}
                        <Line type="monotone" dataKey="callDelta" stroke="#00ff00" name="Call Δ" dot={false} strokeWidth={2} />
                        <Line type="monotone" dataKey="putDelta" stroke="#ff0000" name="Put Δ" dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Gamma Chart */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">GAMMA (Γ) ACROSS STRIKES</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={greeksChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="strike" tick={{ fill: '#888', fontSize: 8 }} />
                        <YAxis tick={{ fill: '#888', fontSize: 8 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                          labelStyle={{ color: '#ff8c00' }}
                        />
                        {currentTick && <ReferenceLine x={currentTick.last} stroke="#ff8c00" strokeDasharray="5 5" />}
                        <Area type="monotone" dataKey="gamma" stroke="#00ffff" fill="#00ffff" fillOpacity={0.3} name="Gamma" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Theta Chart */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">THETA (Θ) - TIME DECAY</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <ComposedChart data={greeksChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="strike" tick={{ fill: '#888', fontSize: 8 }} />
                        <YAxis tick={{ fill: '#888', fontSize: 8 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                          labelStyle={{ color: '#ff8c00' }}
                        />
                        <ReferenceLine y={0} stroke="#444" />
                        {currentTick && <ReferenceLine x={currentTick.last} stroke="#ff8c00" strokeDasharray="5 5" />}
                        <Bar dataKey="callTheta" fill="#ff4444" name="Call Θ" opacity={0.7} />
                        <Bar dataKey="putTheta" fill="#ff8888" name="Put Θ" opacity={0.7} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Vega Chart */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">VEGA (ν) - VOLATILITY SENSITIVITY</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={greeksChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="strike" tick={{ fill: '#888', fontSize: 8 }} />
                        <YAxis tick={{ fill: '#888', fontSize: 8 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                          labelStyle={{ color: '#ff8c00' }}
                        />
                        {currentTick && <ReferenceLine x={currentTick.last} stroke="#ff8c00" strokeDasharray="5 5" />}
                        <Area type="monotone" dataKey="vega" stroke="#ff00ff" fill="#ff00ff" fillOpacity={0.3} name="Vega" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Greeks Summary Table */}
                  {optionResult && (
                    <div className="col-span-2 border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">CURRENT OPTION GREEKS (FULL SUITE)</div>
                      <div className="grid grid-cols-8 gap-3">
                        {[
                          { label: 'Delta (Δ)', value: optionResult.greeks.delta, fmt: 4, color: '#00ff00' },
                          { label: 'Gamma (Γ)', value: optionResult.greeks.gamma, fmt: 6, color: '#00ffff' },
                          { label: 'Theta (Θ)', value: optionResult.greeks.theta, fmt: 4, color: '#ff4444' },
                          { label: 'Vega (ν)', value: optionResult.greeks.vega, fmt: 4, color: '#ff00ff' },
                          { label: 'Rho (ρ)', value: optionResult.greeks.rho, fmt: 4, color: '#ffff00' },
                          { label: 'Vanna', value: optionResult.greeks.vanna || 0, fmt: 6, color: '#88ff88' },
                          { label: 'Volga', value: optionResult.greeks.volga || 0, fmt: 6, color: '#8888ff' },
                          { label: 'Charm', value: optionResult.greeks.charm || 0, fmt: 6, color: '#ff8888' },
                        ].map((g, i) => (
                          <div key={i} className="text-center">
                            <div className="text-[#666] text-[8px]">{g.label}</div>
                            <div className="text-sm" style={{ color: g.color }}>{formatNumber(g.value, g.fmt)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* PAYOFF TAB */}
              {analysisTab === 'PAYOFF' && (
                <div className="space-y-4">
                  {payoffData.length > 0 ? (
                    <div className="border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">
                        PAYOFF DIAGRAM - {optionForm.optionType.toUpperCase()} @ STRIKE {optionForm.strike || currentTick?.last}
                      </div>
                      <ResponsiveContainer width="100%" height={350}>
                        <ComposedChart data={payoffData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis
                            dataKey="spotPrice"
                            tick={{ fill: '#888', fontSize: 9 }}
                            tickFormatter={(v) => formatNumber(v, 0)}
                            label={{ value: 'Spot Price at Expiry', position: 'bottom', fill: '#666', fontSize: 9 }}
                          />
                          <YAxis
                            tick={{ fill: '#888', fontSize: 9 }}
                            label={{ value: 'P&L', angle: -90, position: 'insideLeft', fill: '#666', fontSize: 9 }}
                          />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                            labelStyle={{ color: '#ff8c00' }}
                            formatter={(value: number) => ['$' + formatNumber(value, 2)]}
                            labelFormatter={(label) => `Spot: $${formatNumber(label, 2)}`}
                          />
                          <ReferenceLine y={0} stroke="#666" strokeWidth={2} />
                          {currentTick && <ReferenceLine x={currentTick.last} stroke="#ff8c00" strokeDasharray="5 5" label={{ value: 'Current', fill: '#ff8c00', fontSize: 9 }} />}
                          <Area
                            type="monotone"
                            dataKey="pnl"
                            stroke="#00ff00"
                            fill="#00ff00"
                            fillOpacity={0.2}
                            name="P&L"
                          />
                          <Line type="monotone" dataKey="payoff" stroke="#ffff00" name="Payoff" dot={false} strokeDasharray="5 5" />
                        </ComposedChart>
                      </ResponsiveContainer>
                      <div className="mt-3 grid grid-cols-4 gap-4 text-center">
                        <div>
                          <div className="text-[#666] text-[8px]">MAX PROFIT</div>
                          <div className="text-[#00ff00]">{optionForm.optionType === 'call' ? 'UNLIMITED' : `$${formatNumber((parseFloat(optionForm.strike) || currentTick?.last || 0) - (optionResult?.price || 0), 2)}`}</div>
                        </div>
                        <div>
                          <div className="text-[#666] text-[8px]">MAX LOSS</div>
                          <div className="text-[#ff0000]">${formatNumber(optionResult?.price || 0, 2)}</div>
                        </div>
                        <div>
                          <div className="text-[#666] text-[8px]">BREAKEVEN</div>
                          <div className="text-[#ffff00]">{formatNumber(optionResult?.breakeven || 0, 2)}</div>
                        </div>
                        <div>
                          <div className="text-[#666] text-[8px]">RISK/REWARD</div>
                          <div className="text-[#fff]">{optionForm.optionType === 'call' ? '∞' : formatNumber(((parseFloat(optionForm.strike) || currentTick?.last || 0) - (optionResult?.price || 0)) / (optionResult?.price || 1), 2)}</div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-[#666] py-20">
                      CALCULATE AN OPTION PRICE TO VIEW PAYOFF DIAGRAM
                    </div>
                  )}
                </div>
              )}

              {/* SENSITIVITY TAB */}
              {analysisTab === 'SENSITIVITY' && optionResult && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    {/* Spot Sensitivity */}
                    <div className="border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">SPOT PRICE SENSITIVITY (Δ + Γ)</div>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={sensitivityData.spotSensitivity}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="spotChange" tick={{ fill: '#888', fontSize: 8 }} tickFormatter={(v) => `${v}%`} />
                          <YAxis tick={{ fill: '#888', fontSize: 8 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                            labelFormatter={(l) => `Spot Δ: ${l}%`}
                          />
                          <ReferenceLine x={0} stroke="#ff8c00" />
                          <ReferenceLine y={0} stroke="#666" />
                          <Line type="monotone" dataKey="priceChange" stroke="#00ff00" name="Price Δ" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Vol Sensitivity */}
                    <div className="border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">VOLATILITY SENSITIVITY (ν)</div>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={sensitivityData.volSensitivity}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="volChange" tick={{ fill: '#888', fontSize: 8 }} tickFormatter={(v) => `${v}%`} />
                          <YAxis tick={{ fill: '#888', fontSize: 8 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                            labelFormatter={(l) => `Vol Δ: ${l}%`}
                          />
                          <ReferenceLine x={0} stroke="#ff8c00" />
                          <ReferenceLine y={0} stroke="#666" />
                          <Line type="monotone" dataKey="priceChange" stroke="#ff00ff" name="Price Δ" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Time Decay */}
                    <div className="border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">TIME DECAY (Θ)</div>
                      <ResponsiveContainer width="100%" height={200}>
                        <AreaChart data={sensitivityData.timeSensitivity}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="days" tick={{ fill: '#888', fontSize: 8 }} />
                          <YAxis tick={{ fill: '#888', fontSize: 8 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                            labelFormatter={(l) => `Day ${l}`}
                          />
                          <ReferenceLine y={0} stroke="#666" />
                          <Area type="monotone" dataKey="priceDecay" stroke="#ff4444" fill="#ff4444" fillOpacity={0.3} name="Decay" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Sensitivity Parameters */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">SENSITIVITY PARAMETERS</div>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="text-[#666] text-[8px]">SPOT RANGE (%)</label>
                        <input
                          type="range"
                          min="5"
                          max="30"
                          value={sensitivityParams.spotRange}
                          onChange={(e) => setSensitivityParams({ ...sensitivityParams, spotRange: parseInt(e.target.value) })}
                          className="w-full"
                        />
                        <div className="text-center text-[#fff] text-[9px]">±{sensitivityParams.spotRange}%</div>
                      </div>
                      <div>
                        <label className="text-[#666] text-[8px]">VOL RANGE (%)</label>
                        <input
                          type="range"
                          min="10"
                          max="100"
                          value={sensitivityParams.volRange}
                          onChange={(e) => setSensitivityParams({ ...sensitivityParams, volRange: parseInt(e.target.value) })}
                          className="w-full"
                        />
                        <div className="text-center text-[#fff] text-[9px]">±{sensitivityParams.volRange}%</div>
                      </div>
                      <div>
                        <label className="text-[#666] text-[8px]">TIME HORIZON (DAYS)</label>
                        <input
                          type="range"
                          min="1"
                          max="30"
                          value={sensitivityParams.timeDecay}
                          onChange={(e) => setSensitivityParams({ ...sensitivityParams, timeDecay: parseInt(e.target.value) })}
                          className="w-full"
                        />
                        <div className="text-center text-[#fff] text-[9px]">{sensitivityParams.timeDecay} days</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* VOL SURFACE TAB */}
              {analysisTab === 'COMPARISON' && volSmileData.length > 0 && (
                <div className="space-y-4">
                  {/* Volatility Smile */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">VOLATILITY SMILE</div>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={volSmileData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis
                          dataKey="moneyness"
                          tick={{ fill: '#888', fontSize: 9 }}
                          tickFormatter={(v) => `${v.toFixed(0)}%`}
                          label={{ value: 'Moneyness (% from ATM)', position: 'bottom', fill: '#666', fontSize: 9 }}
                        />
                        <YAxis
                          tick={{ fill: '#888', fontSize: 9 }}
                          label={{ value: 'Implied Volatility %', angle: -90, position: 'insideLeft', fill: '#666', fontSize: 9 }}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                          labelFormatter={(l) => `Moneyness: ${l.toFixed(1)}%`}
                        />
                        <Legend />
                        <ReferenceLine x={0} stroke="#ff8c00" strokeDasharray="5 5" label={{ value: 'ATM', fill: '#ff8c00', fontSize: 9 }} />
                        <Line type="monotone" dataKey="callIV" stroke="#00ff00" name="Call IV" dot={{ r: 3 }} />
                        <Line type="monotone" dataKey="putIV" stroke="#ff0000" name="Put IV" dot={{ r: 3 }} />
                        <Line type="monotone" dataKey="avgIV" stroke="#ffff00" name="Avg IV" strokeDasharray="5 5" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* IV by Strike */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">IV BY STRIKE</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={volSmileData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="strike" tick={{ fill: '#888', fontSize: 8 }} tickFormatter={(v) => formatNumber(v, 0)} />
                        <YAxis tick={{ fill: '#888', fontSize: 8 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }} />
                        <Bar dataKey="avgIV" fill="#ff8c00" name="IV %" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* FUTURES */}
          {selectedCategory === 'FUTURES' && selectedAsset && (
            <div className="space-y-4">
              {futuresResult && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="border border-[#00ff00] p-4">
                    <div className="text-[#ff8c00] mb-3">FUTURES PRICING</div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-[#666] text-[9px]">SPOT PRICE</div>
                        <div className="text-[#fff] text-xl">{formatNumber(futuresResult.spot_price, 2)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">FAIR VALUE</div>
                        <div className="text-[#00ff00] text-xl">{formatNumber(futuresResult.fair_value, 2)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">BASIS</div>
                        <div className={`text-lg ${futuresResult.basis >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                          {formatNumber(futuresResult.basis, 2)} ({futuresResult.basis_pct.toFixed(2)}%)
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">IMPLIED REPO</div>
                        <div className="text-[#ffff00]">{(futuresResult.implied_repo_rate * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                  </div>

                  <div className="border border-[#444] p-4">
                    <div className="text-[#ff8c00] mb-3">BASIS VISUALIZATION</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={[
                        { name: 'Spot', value: futuresResult.spot_price, fill: '#888' },
                        { name: 'Fair Value', value: futuresResult.fair_value, fill: '#00ff00' },
                        { name: 'Basis', value: futuresResult.basis, fill: futuresResult.basis >= 0 ? '#00ff00' : '#ff0000' },
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="name" tick={{ fill: '#888', fontSize: 9 }} />
                        <YAxis tick={{ fill: '#888', fontSize: 9 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }} />
                        <Bar dataKey="value" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* STRUCTURED PRODUCTS */}
          {selectedCategory === 'STRUCTURED' && selectedAsset && (
            <div className="space-y-4">
              {structuredResult && (
                <>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="border border-[#00ff00] p-4">
                      <div className="text-[#ff8c00] mb-2">{structuredResult.product_name}</div>
                      <div className="text-[#00ff00] text-2xl mb-2">${formatNumber(structuredResult.total_price, 2)}</div>
                      <div className="grid grid-cols-2 gap-2 text-[9px]">
                        <div><span className="text-[#666]">Max Profit:</span> <span className="text-[#00ff00]">${formatNumber(structuredResult.max_profit, 2)}</span></div>
                        <div><span className="text-[#666]">Max Loss:</span> <span className="text-[#ff0000]">${formatNumber(Math.abs(structuredResult.max_loss), 2)}</span></div>
                      </div>
                    </div>

                    <div className="border border-[#444] p-4">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">NET GREEKS</div>
                      <div className="grid grid-cols-2 gap-2">
                        <div><span className="text-[#666] text-[9px]">Delta:</span> <span className="text-[#00ff00]">{formatNumber(structuredResult.net_greeks.delta, 4)}</span></div>
                        <div><span className="text-[#666] text-[9px]">Gamma:</span> <span className="text-[#00ffff]">{formatNumber(structuredResult.net_greeks.gamma, 6)}</span></div>
                        <div><span className="text-[#666] text-[9px]">Theta:</span> <span className="text-[#ff4444]">{formatNumber(structuredResult.net_greeks.theta, 4)}</span></div>
                        <div><span className="text-[#666] text-[9px]">Vega:</span> <span className="text-[#ff00ff]">{formatNumber(structuredResult.net_greeks.vega, 4)}</span></div>
                      </div>
                    </div>

                    <div className="border border-[#444] p-4">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">BREAKEVENS</div>
                      <div className="space-y-1">
                        {structuredResult.breakevens.map((be: number, i: number) => (
                          <div key={i} className="text-[#ffff00]">{formatNumber(be, 2)}</div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Payoff Diagram */}
                  {structuredPayoffData.length > 0 && (
                    <div className="border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">PAYOFF DIAGRAM</div>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={structuredPayoffData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="spotPrice" tick={{ fill: '#888', fontSize: 9 }} tickFormatter={(v) => formatNumber(v, 0)} />
                          <YAxis tick={{ fill: '#888', fontSize: 9 }} />
                          <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }} />
                          <ReferenceLine y={0} stroke="#666" strokeWidth={2} />
                          {currentTick && <ReferenceLine x={currentTick.last} stroke="#ff8c00" strokeDasharray="5 5" />}
                          <Area type="monotone" dataKey="pnl" stroke="#00ff00" fill="#00ff00" fillOpacity={0.2} name="P&L" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Legs */}
                  <div className="border border-[#444] p-3">
                    <div className="text-[#ff8c00] mb-2 text-[10px]">STRUCTURE LEGS</div>
                    <table className="w-full text-[9px]">
                      <thead>
                        <tr className="text-[#666]">
                          <th className="text-left py-1">TYPE</th>
                          <th className="text-right py-1">STRIKE</th>
                          <th className="text-right py-1">QTY</th>
                          <th className="text-right py-1">PRICE</th>
                          <th className="text-right py-1">DELTA</th>
                        </tr>
                      </thead>
                      <tbody>
                        {structuredResult.legs.map((leg, idx) => (
                          <tr key={idx} className="border-t border-[#333]">
                            <td className={`py-1 ${leg.quantity > 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                              {leg.quantity > 0 ? 'LONG' : 'SHORT'} {leg.option_type?.toUpperCase()}
                            </td>
                            <td className="text-right py-1 text-[#fff]">{leg.strike}</td>
                            <td className="text-right py-1">{Math.abs(leg.quantity)}</td>
                            <td className="text-right py-1">${formatNumber(leg.price, 2)}</td>
                            <td className="text-right py-1">{formatNumber(leg.delta || 0, 3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          {/* BACKTEST */}
          {selectedCategory === 'BACKTEST' && (
            <div className="space-y-4">
              {backtests.length > 0 && (
                <div className="border border-[#444]">
                  <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                    <span className="text-[#ff8c00]">BACKTEST HISTORY</span>
                  </div>
                  <div className="p-2 space-y-1 max-h-40 overflow-y-auto">
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
                            bt.status === 'running' ? 'text-[#ffff00]' : 'text-[#ff0000]'
                          }`}>
                            {bt.status.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Backtest Results */}
              {selectedBacktest?.result && (
                <div className="space-y-4">
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { label: 'Total Return', value: formatPercent(selectedBacktest.result.performance.total_return), color: selectedBacktest.result.performance.total_return >= 0 ? '#00ff00' : '#ff0000' },
                      { label: 'Sharpe Ratio', value: selectedBacktest.result.performance.sharpe_ratio.toFixed(2), color: '#fff' },
                      { label: 'Max Drawdown', value: formatPercent(-selectedBacktest.result.performance.max_drawdown), color: '#ff0000' },
                      { label: 'Win Rate', value: formatPercent(selectedBacktest.result.performance.win_rate), color: '#ffff00' },
                    ].map((m, i) => (
                      <div key={i} className="border border-[#444] p-3 text-center">
                        <div className="text-[#666] text-[8px]">{m.label}</div>
                        <div className="text-lg" style={{ color: m.color }}>{m.value}</div>
                      </div>
                    ))}
                  </div>

                  {/* Equity Curve */}
                  {selectedBacktest.result.equity_curve && (
                    <div className="border border-[#444] p-3">
                      <div className="text-[#ff8c00] mb-2 text-[10px]">EQUITY CURVE</div>
                      <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={selectedBacktest.result.equity_curve}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="timestamp" tick={{ fill: '#888', fontSize: 8 }} tickFormatter={(v) => new Date(v).toLocaleDateString()} />
                          <YAxis tick={{ fill: '#888', fontSize: 8 }} tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
                            labelFormatter={(l) => new Date(l).toLocaleDateString()}
                            formatter={(v: number) => [`$${formatNumber(v, 2)}`, 'Equity']}
                          />
                          <Area type="monotone" dataKey="equity" stroke="#00ff00" fill="#00ff00" fillOpacity={0.2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {!selectedAsset && (
            <div className="text-center text-[#666] mt-20">
              SELECT AN ASSET FROM THE LEFT PANEL TO BEGIN ANALYSIS
            </div>
          )}
        </div>
      </div>

      {/* Right Panel - Collapsible Actions */}
      <div className={`${isRightCollapsed ? 'w-12' : 'w-64'} border-l border-[#444] bg-[#0a0a0a] overflow-y-auto transition-all duration-300 flex-shrink-0`}>
        <button
          onClick={() => setIsRightCollapsed(!isRightCollapsed)}
          className="w-full p-2 border-b border-[#444] text-[#ff8c00] hover:bg-[#1a1a1a] transition-colors flex items-center justify-center text-[10px]"
        >
          {isRightCollapsed ? '«' : 'HIDE »'}
        </button>

        {!isRightCollapsed && (
          <div className="p-3">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">ACTIONS</div>

            {/* Options Actions */}
            {selectedCategory === 'OPTIONS' && selectedAsset && (
              <div className="space-y-2">
                <div className="text-[#ff8c00] text-[9px] mb-1">PRICE OPTION</div>
                <div>
                  <label className="text-[#666] text-[8px]">STRIKE</label>
                  <input
                    type="number"
                    value={optionForm.strike}
                    onChange={e => setOptionForm({ ...optionForm, strike: e.target.value })}
                    placeholder={currentTick?.last.toString()}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] text-[8px]">EXPIRY</label>
                  <input
                    type="date"
                    value={optionForm.expiry}
                    onChange={e => setOptionForm({ ...optionForm, expiry: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <div className="grid grid-cols-2 gap-1">
                  <button
                    onClick={() => setOptionForm({ ...optionForm, optionType: 'call' })}
                    className={`py-1 border text-[9px] ${
                      optionForm.optionType === 'call'
                        ? 'border-[#00ff00] bg-[#00ff00] text-black'
                        : 'border-[#444] text-[#00ff00]'
                    }`}
                  >
                    CALL
                  </button>
                  <button
                    onClick={() => setOptionForm({ ...optionForm, optionType: 'put' })}
                    className={`py-1 border text-[9px] ${
                      optionForm.optionType === 'put'
                        ? 'border-[#ff0000] bg-[#ff0000] text-black'
                        : 'border-[#444] text-[#ff0000]'
                    }`}
                  >
                    PUT
                  </button>
                </div>
                <div>
                  <label className="text-[#666] text-[8px]">VOLATILITY</label>
                  <input
                    type="number"
                    step="0.01"
                    value={optionForm.volatility}
                    onChange={e => setOptionForm({ ...optionForm, volatility: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <button
                  onClick={handlePriceOption}
                  disabled={isLoading}
                  className="w-full py-1.5 border border-[#ff8c00] text-[#ff8c00] text-[9px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                >
                  {isLoading ? 'CALCULATING...' : 'CALCULATE'}
                </button>
              </div>
            )}

            {/* Futures Actions */}
            {selectedCategory === 'FUTURES' && selectedAsset && (
              <div className="space-y-2">
                <div className="text-[#ff8c00] text-[9px] mb-1">PRICE FUTURES</div>
                <div>
                  <label className="text-[#666] text-[8px]">EXPIRY</label>
                  <input
                    type="date"
                    value={futuresForm.expiry}
                    onChange={e => setFuturesForm({ ...futuresForm, expiry: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] text-[8px]">CONVENIENCE YIELD</label>
                  <input
                    type="number"
                    step="0.01"
                    value={futuresForm.convenienceYield}
                    onChange={e => setFuturesForm({ ...futuresForm, convenienceYield: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <button
                  onClick={handlePriceFutures}
                  disabled={isLoading}
                  className="w-full py-1.5 border border-[#ff8c00] text-[#ff8c00] text-[9px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                >
                  CALCULATE
                </button>
              </div>
            )}

            {/* Structured Actions */}
            {selectedCategory === 'STRUCTURED' && selectedAsset && (
              <div className="space-y-2">
                <div className="text-[#ff8c00] text-[9px] mb-1">STRUCTURE</div>
                <div>
                  <label className="text-[#666] text-[8px]">TYPE</label>
                  <select
                    value={structuredForm.productType}
                    onChange={e => setStructuredForm({ ...structuredForm, productType: e.target.value as typeof structuredForm.productType })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  >
                    <option value="straddle">STRADDLE</option>
                    <option value="strangle">STRANGLE</option>
                    <option value="butterfly">BUTTERFLY</option>
                    <option value="iron_condor">IRON CONDOR</option>
                  </select>
                </div>
                <div>
                  <label className="text-[#666] text-[8px]">STRIKE</label>
                  <input
                    type="number"
                    value={structuredForm.strike}
                    onChange={e => setStructuredForm({ ...structuredForm, strike: e.target.value })}
                    placeholder={currentTick?.last.toString()}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] text-[8px]">EXPIRY</label>
                  <input
                    type="date"
                    value={structuredForm.expiry}
                    onChange={e => setStructuredForm({ ...structuredForm, expiry: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <button
                  onClick={handlePriceStructured}
                  disabled={isLoading}
                  className="w-full py-1.5 border border-[#ff8c00] text-[#ff8c00] text-[9px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                >
                  PRICE
                </button>
              </div>
            )}

            {/* Backtest Actions */}
            {selectedCategory === 'BACKTEST' && selectedAsset && (
              <div className="space-y-2">
                <div className="text-[#ff8c00] text-[9px] mb-1">RUN BACKTEST</div>
                <div>
                  <label className="text-[#666] text-[8px]">NAME</label>
                  <input
                    type="text"
                    value={backtestForm.name}
                    onChange={e => setBacktestForm({ ...backtestForm, name: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] text-[8px]">STRATEGY</label>
                  <select
                    value={backtestForm.strategyType}
                    onChange={e => setBacktestForm({ ...backtestForm, strategyType: e.target.value as typeof backtestForm.strategyType })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  >
                    <option value="covered_call">COVERED CALL</option>
                    <option value="iron_condor">IRON CONDOR</option>
                  </select>
                </div>
                <div className="grid grid-cols-2 gap-1">
                  <div>
                    <label className="text-[#666] text-[8px]">START</label>
                    <input
                      type="date"
                      value={backtestForm.startDate}
                      onChange={e => setBacktestForm({ ...backtestForm, startDate: e.target.value })}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-1 py-1 text-[8px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] text-[8px]">END</label>
                    <input
                      type="date"
                      value={backtestForm.endDate}
                      onChange={e => setBacktestForm({ ...backtestForm, endDate: e.target.value })}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-1 py-1 text-[8px]"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-[#666] text-[8px]">CAPITAL</label>
                  <input
                    type="number"
                    value={backtestForm.initialCapital}
                    onChange={e => setBacktestForm({ ...backtestForm, initialCapital: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                  />
                </div>
                <button
                  onClick={handleRunBacktest}
                  disabled={isLoading}
                  className="w-full py-1.5 border border-[#00ff00] text-[#00ff00] text-[9px] hover:bg-[#00ff00] hover:text-black transition-colors"
                >
                  RUN
                </button>
              </div>
            )}

            {/* Quick Stats */}
            <div className="mt-4 border border-[#444] p-2">
              <div className="text-[#ff8c00] mb-2 text-[8px] tracking-wider">STATS</div>
              <div className="space-y-1 text-[9px]">
                <div className="flex justify-between">
                  <span className="text-[#666]">Assets</span>
                  <span className="text-[#fff]">{assets.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">Backtests</span>
                  <span className="text-[#fff]">{backtests.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">Mode</span>
                  <span className={useMockData ? 'text-[#ffff00]' : 'text-[#00ff00]'}>
                    {useMockData ? 'MOCK' : 'LIVE'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Collapsed view */}
        {isRightCollapsed && (
          <div className="p-1 space-y-1">
            <button
              onClick={() => setIsRightCollapsed(false)}
              className="w-full p-2 border border-[#444] text-[#ff8c00] text-[8px] hover:border-[#ff8c00]"
              title="Actions"
            >
              ACT
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
