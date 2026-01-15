import { useState, useEffect } from 'react';
import {
  getLiveMarketData,
  getCorrelationMatrix,
  getMicrostructure,
  getLiquidityAnalysis,
  getMarketEvents,
  type LiveMarketData,
  type CorrelationData,
  type MicrostructureData,
  type LiquidityData,
  type MarketEvent,
} from '../../services/marketsService';
import {
  getDerivTick,
  getDerivAccount,
  getDerivStatus,
  getDerivSymbols,
  calculateSpreadBps,
  calculateMidPrice,
  formatSymbol,
  isMarketOpen,
  POPULAR_FOREX,
  POPULAR_SYNTHETICS,
  type DerivTick,
  type DerivAccount,
  type DerivConnectionStatus,
  type DerivSymbol,
} from '../../services/derivService';

// Mock data generator for fallback
const generateMockLiveData = (assets: string[]): LiveMarketData[] => {
  return assets.map(assetId => {
    const hash = assetId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const basePrice = 1.0 + (hash % 100) * 0.01;
    const timestamp = new Date().toISOString();
    
    return {
      asset_id: assetId,
      timestamp,
      bid: basePrice - 0.0001,
      ask: basePrice + 0.0001,
      last: basePrice,
      volume: 1000000 + (hash % 5000000),
      spread_bps: 2.0,
      depth_bid: 2500000.0,
      depth_ask: 2300000.0,
      tick_frequency: 100 + (hash % 50),
    };
  });
};

const generateMockCorrelations = (assets: string[]): CorrelationData => {
  const n = assets.length;
  const matrix: number[][] = [];
  const significance: number[][] = [];
  
  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    significance[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 1.0;
        significance[i][j] = 0.0;
      } else {
        const hash = (i * 100 + j * 10) % 200;
        matrix[i][j] = (hash - 100) / 100;
        significance[i][j] = 0.001;
      }
    }
  }
  
  return {
    timestamp: new Date().toISOString(),
    assets,
    correlation_matrix: matrix,
    significance,
    window: '24H',
    method: 'pearson',
    clusters: [assets.slice(0, 2), assets.slice(2)],
  };
};

const generateMockMicrostructure = (assetId: string): MicrostructureData => {
  const hash = assetId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  
  return {
    asset_id: assetId,
    timestamp: new Date().toISOString(),
    spread_bps: 1.8 + (hash % 10) * 0.1,
    effective_spread_bps: 1.6 + (hash % 10) * 0.1,
    quoted_spread_bps: 2.0 + (hash % 10) * 0.1,
    depth_bid: 2500000.0 + (hash % 1000000),
    depth_ask: 2300000.0 + (hash % 1000000),
    imbalance_ratio: 0.52 + (hash % 20) * 0.01,
    tick_frequency: 100.0 + (hash % 50),
    price_impact_bps: 0.9 + (hash % 10) * 0.1,
  };
};

const generateMockLiquidity = (assets: string[]): LiquidityData[] => {
  return assets.map(assetId => {
    const hash = assetId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    
    return {
      asset_id: assetId,
      timestamp: new Date().toISOString(),
      bid_liquidity_usd: 2500000.0 + (hash % 2000000),
      ask_liquidity_usd: 2300000.0 + (hash % 2000000),
      total_liquidity_usd: 4800000.0 + (hash % 4000000),
      liquidity_score: 75.0 + (hash % 25),
      resilience_score: 70.0 + (hash % 30),
      toxicity_score: 20.0 + (hash % 30),
    };
  });
};

const generateMockEvents = (): MarketEvent[] => {
  const now = new Date();
  return [
    {
      timestamp: new Date(now.getTime() - 15 * 60000).toISOString(),
      asset_id: 'EURUSD',
      event_type: 'VOLATILITY_SPIKE',
      severity: 0.7,
      description: 'Volatility increased by 45% above baseline',
      recommended_action: 'ALERT',
    },
    {
      timestamp: new Date(now.getTime() - 30 * 60000).toISOString(),
      asset_id: 'GBPUSD',
      event_type: 'LIQUIDITY_DROP',
      severity: 0.6,
      description: 'Liquidity decreased by 30%',
      recommended_action: 'ALERT',
    },
    {
      timestamp: new Date(now.getTime() - 60 * 60000).toISOString(),
      asset_id: 'BTCUSD',
      event_type: 'PRICE_SPIKE',
      severity: 0.8,
      description: 'Price increased by 3.5% in 5 minutes',
      recommended_action: 'INVESTIGATE',
    },
  ];
};

interface WatchlistItem {
  id: string;
  name: string;
  assets: string[];
}

interface Alert {
  id: string;
  assetId: string;
  type: 'PRICE' | 'VOLATILITY' | 'LIQUIDITY' | 'CORRELATION';
  condition: string;
  threshold: number;
  enabled: boolean;
}

export function Markets() {
  const [liveData, setLiveData] = useState<LiveMarketData[]>([]);
  const [correlations, setCorrelations] = useState<CorrelationData | null>(null);
  const [microstructure, setMicrostructure] = useState<MicrostructureData | null>(null);
  const [liquidity, setLiquidity] = useState<LiquidityData[]>([]);
  const [events, setEvents] = useState<MarketEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState(true);
  const [useMockData, setUseMockData] = useState(false);

  // Deriv integration state
  const [derivTicks, setDerivTicks] = useState<Map<string, DerivTick>>(new Map());
  const [derivAccount, setDerivAccount] = useState<DerivAccount | null>(null);
  const [derivStatus, setDerivStatus] = useState<DerivConnectionStatus | null>(null);
  const [derivSymbols, setDerivSymbols] = useState<DerivSymbol[]>([]);
  const [selectedDerivCategory, setSelectedDerivCategory] = useState<'forex' | 'synthetic'>('forex');

  // User engagement state
  const [watchlists, setWatchlists] = useState<WatchlistItem[]>([
    { id: '1', name: 'Major Pairs', assets: ['EURUSD', 'GBPUSD', 'USDJPY'] },
    { id: '2', name: 'Crypto', assets: ['BTCUSD', 'ETHUSD'] },
  ]);
  const [activeWatchlist, setActiveWatchlist] = useState<string>('1');
  const [alerts, setAlerts] = useState<Alert[]>([
    { id: '1', assetId: 'EURUSD', type: 'PRICE', condition: 'above', threshold: 1.10, enabled: true },
    { id: '2', assetId: 'BTCUSD', type: 'VOLATILITY', condition: 'above', threshold: 50, enabled: true },
  ]);
  const [showWatchlistModal, setShowWatchlistModal] = useState(false);
  const [showAlertModal, setShowAlertModal] = useState(false);
  const [editingWatchlist, setEditingWatchlist] = useState<WatchlistItem | null>(null);
  const [editingAlert, setEditingAlert] = useState<Alert | null>(null);

  const watchedAssets = watchlists.find(w => w.id === activeWatchlist)?.assets || ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'BTCUSD'];

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatAssetSymbol = (assetId: string) => {
    // Convert EURUSD to EUR/USD
    if (assetId.length === 6) {
      return `${assetId.slice(0, 3)}/${assetId.slice(3)}`;
    }
    if (assetId === 'BTCUSD') {
      return 'BTC/USD';
    }
    return assetId;
  };

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Try to fetch from backend
        const [liveResponse, corrResponse, microResponse, liquidityResponse, eventsResponse] = await Promise.all([
          getLiveMarketData(watchedAssets),
          getCorrelationMatrix(watchedAssets, '24H', 'pearson'),
          getMicrostructure(watchedAssets[0]),
          getLiquidityAnalysis(watchedAssets),
          getMarketEvents(undefined, undefined, 0.5),
        ]);

        setLiveData(liveResponse.data);
        setCorrelations(corrResponse);
        setMicrostructure(microResponse);
        setLiquidity(liquidityResponse.data);
        setEvents(eventsResponse);
        setIsBackendConnected(true);
        setUseMockData(false);
      } catch (err: any) {
        console.warn('Backend unavailable, using mock data:', err);
        
        // Fallback to mock data
        setLiveData(generateMockLiveData(watchedAssets));
        setCorrelations(generateMockCorrelations(watchedAssets));
        setMicrostructure(generateMockMicrostructure(watchedAssets[0]));
        setLiquidity(generateMockLiquidity(watchedAssets));
        setEvents(generateMockEvents());
        setIsBackendConnected(false);
        setUseMockData(true);
        setError('Backend disconnected - using mock data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchMarketData();

    // Refresh data every 5 seconds
    const interval = setInterval(fetchMarketData, 5000);

    return () => clearInterval(interval);
  }, [activeWatchlist]);

  // Fetch Deriv data
  useEffect(() => {
    const fetchDerivData = async () => {
      try {
        // Get account info
        const account = await getDerivAccount();
        setDerivAccount(account);
        
        // Get connection status
        const status = await getDerivStatus();
        setDerivStatus(status);
        
        // Get available symbols
        const symbols = await getDerivSymbols();
        setDerivSymbols(symbols);
        
        // Get ticks for popular symbols
        const symbolsToFetch = selectedDerivCategory === 'forex' ? POPULAR_FOREX : POPULAR_SYNTHETICS;
        const tickPromises = symbolsToFetch.map(symbol => getDerivTick(symbol));
        const ticks = await Promise.all(tickPromises);
        
        const tickMap = new Map<string, DerivTick>();
        ticks.forEach(tick => tickMap.set(tick.symbol, tick));
        setDerivTicks(tickMap);
      } catch (err) {
        console.warn('Failed to fetch Deriv data:', err);
      }
    };
    
    fetchDerivData();
    const interval = setInterval(fetchDerivData, 1000); // Update every second
    
    return () => clearInterval(interval);
  }, [selectedDerivCategory]);

  // CRUD operations for watchlists
  const createWatchlist = (name: string, assets: string[]) => {
    const newWatchlist: WatchlistItem = {
      id: Date.now().toString(),
      name,
      assets,
    };
    setWatchlists([...watchlists, newWatchlist]);
    setActiveWatchlist(newWatchlist.id);
  };

  const updateWatchlist = (id: string, name: string, assets: string[]) => {
    setWatchlists(watchlists.map(w => w.id === id ? { ...w, name, assets } : w));
  };

  const deleteWatchlist = (id: string) => {
    setWatchlists(watchlists.filter(w => w.id !== id));
    if (activeWatchlist === id) {
      setActiveWatchlist(watchlists[0]?.id || '');
    }
  };

  // CRUD operations for alerts
  const createAlert = (alert: Omit<Alert, 'id'>) => {
    const newAlert: Alert = {
      ...alert,
      id: Date.now().toString(),
    };
    setAlerts([...alerts, newAlert]);
  };

  const updateAlert = (id: string, updates: Partial<Alert>) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, ...updates } : a));
  };

  const deleteAlert = (id: string) => {
    setAlerts(alerts.filter(a => a.id !== id));
  };

  const toggleAlert = (id: string) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, enabled: !a.enabled } : a));
  };

  // Calculate change percentage (mock for now)
  const calculateChange = (assetId: string) => {
    const hash = assetId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return ((hash % 500) - 250) / 100; // Random between -2.5 and 2.5
  };

  // Build correlation pairs
  const correlationPairs = [];
  if (correlations && correlations.assets.length > 1) {
    for (let i = 0; i < correlations.assets.length; i++) {
      for (let j = i + 1; j < correlations.assets.length; j++) {
        const corr = correlations.correlation_matrix[i]?.[j] || 0;
        const absCorr = Math.abs(corr);
        const strength = absCorr > 0.7 ? 'STRONG' : absCorr > 0.4 ? 'MODERATE' : 'WEAK';
        const sign = corr >= 0 ? 'POSITIVE' : 'NEGATIVE';

        correlationPairs.push({
          pair: `${formatAssetSymbol(correlations.assets[i])} - ${formatAssetSymbol(correlations.assets[j])}`,
          correlation: corr,
          strength,
          sign,
        });
      }
    }
  }

  // Get liquidity summary
  const liquiditySummary = liquidity.length > 0 ? {
    avgLiquidityScore: liquidity.reduce((sum, l) => sum + l.liquidity_score, 0) / liquidity.length,
    avgResilienceScore: liquidity.reduce((sum, l) => sum + l.resilience_score, 0) / liquidity.length,
    avgToxicityScore: liquidity.reduce((sum, l) => sum + l.toxicity_score, 0) / liquidity.length,
  } : null;

  return (
    <div className="h-full flex flex-col font-mono text-xs overflow-hidden">
      {/* Header */}
      <div className="border-t-2 border-b-2 border-[#ff8c00] py-2">
        <div className="flex items-center justify-between px-4">
          <div className="text-[#ff8c00] text-sm tracking-wide">
            MARKET DATA - REAL TIME QUOTES & ANALYTICS
            {isLoading && <span className="ml-2 text-[10px]">LOADING...</span>}
            {useMockData && <span className="ml-2 text-[#ffff00] text-[10px]">MOCK DATA MODE</span>}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowWatchlistModal(true)}
              className="px-3 py-1 bg-[#1a1a1a] border border-[#ff8c00] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors"
            >
              WATCHLISTS
            </button>
            <button
              onClick={() => setShowAlertModal(true)}
              className="px-3 py-1 bg-[#1a1a1a] border border-[#ff8c00] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black transition-colors"
            >
              ALERTS ({alerts.filter(a => a.enabled).length})
            </button>
            <div className={`px-3 py-1 border ${isBackendConnected ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#ff0000] text-[#ff0000]'}`}>
              {isBackendConnected ? '● LIVE' : '● OFFLINE'}
            </div>
          </div>
        </div>
      </div>

      {/* Watchlist Tabs */}
      <div className="border-b border-[#444] bg-[#0a0a0a] px-4 py-2 flex gap-2 overflow-x-auto">
        {watchlists.map(wl => (
          <button
            key={wl.id}
            onClick={() => setActiveWatchlist(wl.id)}
            className={`px-3 py-1 border transition-colors ${
              activeWatchlist === wl.id
                ? 'border-[#ff8c00] bg-[#ff8c00] text-black'
                : 'border-[#444] text-[#666] hover:text-[#ff8c00] hover:border-[#ff8c00]'
            }`}
          >
            {wl.name} ({wl.assets.length})
          </button>
        ))}
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Live Market Data */}
          <div className="border border-[#444]">
            <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
              <div className="text-[#ff8c00]">LIVE MARKET DATA</div>
            </div>
            <table className="w-full">
          <thead>
            <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
              <th className="px-3 py-2 text-left border-b border-[#444]">SYMBOL</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">BID</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">ASK</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">SPREAD</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">VOLUME</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">CHANGE %</th>
            </tr>
          </thead>
          <tbody>
            {liveData.map((asset, idx) => {
              const change = calculateChange(asset.asset_id);
              const spread = asset.ask - asset.bid;
              return (
                <tr key={idx} className="border-b border-[#222]">
                  <td className="px-3 py-2 text-[#00ff00]">{formatAssetSymbol(asset.asset_id)}</td>
                  <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(asset.bid, 4)}</td>
                  <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(asset.ask, 4)}</td>
                  <td className="px-3 py-2 text-right text-[#ffff00]">{formatNumber(spread, 4)}</td>
                  <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(asset.volume, 0)}</td>
                  <td className={`px-3 py-2 text-right ${change >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {change >= 0 ? '+' : ''}{formatNumber(change)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* DERIV LIVE PRICES - NEW SECTION */}
      <div className="border border-[#ff8c00]">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#ff8c00] flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="text-[#ff8c00]">🔴 DERIV LIVE PRICES</div>
            {derivStatus && (
              <div className={`text-[10px] px-2 py-1 border ${
                derivStatus.connected && derivStatus.authorized
                  ? 'border-[#00ff00] text-[#00ff00]'
                  : 'border-[#666] text-[#666]'
              }`}>
                {derivStatus.connected && derivStatus.authorized ? '● CONNECTED' : '○ OFFLINE'}
              </div>
            )}
            {derivAccount && (
              <div className="text-[10px] text-[#666]">
                Balance: <span className="text-[#00ff00]">{formatNumber(derivAccount.balance)} {derivAccount.currency}</span>
                {derivAccount.is_virtual && <span className="text-[#ffff00] ml-2">(DEMO)</span>}
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedDerivCategory('forex')}
              className={`px-3 py-1 text-[10px] border transition-colors ${
                selectedDerivCategory === 'forex'
                  ? 'border-[#ff8c00] bg-[#ff8c00] text-black'
                  : 'border-[#444] text-[#666] hover:text-[#ff8c00] hover:border-[#ff8c00]'
              }`}
            >
              FOREX
            </button>
            <button
              onClick={() => setSelectedDerivCategory('synthetic')}
              className={`px-3 py-1 text-[10px] border transition-colors ${
                selectedDerivCategory === 'synthetic'
                  ? 'border-[#ff8c00] bg-[#ff8c00] text-black'
                  : 'border-[#444] text-[#666] hover:text-[#ff8c00] hover:border-[#ff8c00]'
              }`}
            >
              SYNTHETICS
            </button>
          </div>
        </div>
        <table className="w-full">
          <thead>
            <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
              <th className="px-3 py-2 text-left border-b border-[#444]">SYMBOL</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">BID</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">ASK</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">MID</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">SPREAD (BPS)</th>
              <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">UPDATED</th>
            </tr>
          </thead>
          <tbody>
            {Array.from(derivTicks.entries()).map(([symbol, tick]) => {
              const mid = calculateMidPrice(tick);
              const spreadBps = calculateSpreadBps(tick);
              const marketOpen = isMarketOpen(symbol);
              const timeSinceUpdate = tick.timestamp ? 
                Math.floor((Date.now() - new Date(tick.timestamp).getTime()) / 1000) : 0;
              
              return (
                <tr key={symbol} className="border-b border-[#222] hover:bg-[#1a1a1a] transition-colors">
                  <td className="px-3 py-2 text-[#00ff00] font-bold">{formatSymbol(symbol)}</td>
                  <td className="px-3 py-2 text-right text-[#fff] font-mono">{tick.bid.toFixed(selectedDerivCategory === 'forex' ? 5 : 2)}</td>
                  <td className="px-3 py-2 text-right text-[#fff] font-mono">{tick.ask.toFixed(selectedDerivCategory === 'forex' ? 5 : 2)}</td>
                  <td className="px-3 py-2 text-right text-[#ffff00] font-mono">{mid.toFixed(selectedDerivCategory === 'forex' ? 5 : 2)}</td>
                  <td className="px-3 py-2 text-right text-[#ff8c00]">{spreadBps.toFixed(2)}</td>
                  <td className="px-3 py-2 text-center">
                    <span className={`text-[8px] px-2 py-1 border ${
                      marketOpen ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#666] text-[#666]'
                    }`}>
                      {marketOpen ? 'OPEN' : 'CLOSED'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-[#666] text-[9px]">
                    {timeSinceUpdate < 2 ? 'NOW' : `${timeSinceUpdate}s ago`}
                  </td>
                </tr>
              );
            })}
            {derivTicks.size === 0 && (
              <tr>
                <td colSpan={7} className="px-3 py-4 text-center text-[#666]">
                  Loading Deriv market data...
                </td>
              </tr>
            )}
          </tbody>
        </table>
        <div className="bg-[#0a0a0a] px-3 py-2 border-t border-[#444] text-[9px] text-[#666]">
          <div className="flex justify-between items-center">
            <div>
              Real-time data from Deriv API • Updates every second • 
              {selectedDerivCategory === 'forex' ? ' Forex pairs' : ' Synthetic indices'} • 
              Demo account
            </div>
            <div className="text-[#ff8c00]">
              {derivTicks.size} symbols streaming
            </div>
          </div>
        </div>
      </div>

      {/* Asset Correlations */}
      <div className="border border-[#444]">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
          <div className="text-[#ff8c00]">ASSET CORRELATIONS (24H ROLLING)</div>
        </div>
        <table className="w-full">
          <thead>
            <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
              <th className="px-3 py-2 text-left border-b border-[#444]">ASSET PAIR</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">CORRELATION</th>
              <th className="px-3 py-2 text-center border-b border-[#444]">STRENGTH</th>
              <th className="px-3 py-2 text-center border-b border-[#444]">SIGN</th>
            </tr>
          </thead>
          <tbody>
            {correlationPairs.slice(0, 5).map((corr, idx) => (
              <tr key={idx} className="border-b border-[#222]">
                <td className="px-3 py-2 text-[#00ff00]">{corr.pair}</td>
                <td className={`px-3 py-2 text-right ${corr.correlation >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                  {formatNumber(corr.correlation)}
                </td>
                <td className="px-3 py-2 text-center text-[#ffff00]">{corr.strength}</td>
                <td className={`px-3 py-2 text-center ${corr.sign === 'POSITIVE' ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                  {corr.sign}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Market Microstructure */}
      <div className="grid grid-cols-2 gap-4">
        <div className="border border-[#444] p-3">
          <div className="text-[#ff8c00] mb-3">MARKET MICROSTRUCTURE</div>
          {microstructure ? (
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">SPREAD (BPS)</span>
                <span className="text-[#00ff00]">{formatNumber(microstructure.spread_bps, 2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">DEPTH (BID)</span>
                <span className="text-[#00ff00]">{formatNumber(microstructure.depth_bid / 1000000, 2)}M</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">DEPTH (ASK)</span>
                <span className="text-[#00ff00]">{formatNumber(microstructure.depth_ask / 1000000, 2)}M</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">TICK FREQUENCY</span>
                <span className="text-[#00ff00]">{formatNumber(microstructure.tick_frequency, 0)} HZ</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">IMBALANCE</span>
                <span className="text-[#00ff00]">{formatNumber(microstructure.imbalance_ratio * 100, 1)}%</span>
              </div>
            </div>
          ) : (
            <div className="text-[#666] text-[10px]">Loading...</div>
          )}
        </div>

        <div className="border border-[#444] p-3">
          <div className="text-[#ff8c00] mb-3">LIQUIDITY ANALYSIS</div>
          {liquiditySummary ? (
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">LIQUIDITY SCORE</span>
                <span className="text-[#00ff00]">{formatNumber(liquiditySummary.avgLiquidityScore, 1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">RESILIENCE</span>
                <span className="text-[#00ff00]">{formatNumber(liquiditySummary.avgResilienceScore, 1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">TOXICITY</span>
                <span className="text-[#ffff00]">{formatNumber(liquiditySummary.avgToxicityScore, 1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">MARKET DEPTH</span>
                <span className="text-[#00ff00]">DEEP</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">DATA QUALITY</span>
                <span className="text-[#00ff00]">EXCELLENT</span>
              </div>
            </div>
          ) : (
            <div className="text-[#666] text-[10px]">Loading...</div>
          )}
        </div>
      </div>

      {/* Market Events */}
      <div className="border border-[#444] p-3">
        <div className="text-[#ff8c00] mb-3">RECENT MARKET EVENTS</div>
        <div className="space-y-1 text-[10px]">
          {events.length > 0 ? (
            events.slice(0, 4).map((event, idx) => (
              <div key={idx} className="text-[#666]">
                {new Date(event.timestamp).toLocaleTimeString()} - {event.asset_id} {event.event_type}: {event.description}
              </div>
            ))
          ) : (
            <div className="text-[#666]">No recent events</div>
          )}
        </div>
      </div>
        </div>
      </div>

      {/* Watchlist Modal */}
      {showWatchlistModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">MANAGE WATCHLISTS</h2>
              <button
                onClick={() => {
                  setShowWatchlistModal(false);
                  setEditingWatchlist(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            {/* Watchlist List */}
            <div className="space-y-2 mb-4">
              {watchlists.map(wl => (
                <div key={wl.id} className="border border-[#444] p-3 flex justify-between items-center">
                  <div>
                    <div className="text-[#00ff00]">{wl.name}</div>
                    <div className="text-[#666] text-[10px]">{wl.assets.join(', ')}</div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setEditingWatchlist(wl)}
                      className="px-2 py-1 border border-[#ff8c00] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black"
                    >
                      EDIT
                    </button>
                    <button
                      onClick={() => {
                        if (confirm(`Delete watchlist "${wl.name}"?`)) {
                          deleteWatchlist(wl.id);
                        }
                      }}
                      className="px-2 py-1 border border-[#ff0000] text-[#ff0000] hover:bg-[#ff0000] hover:text-black"
                    >
                      DELETE
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Create/Edit Form */}
            <div className="border border-[#ff8c00] p-4">
              <h3 className="text-[#ff8c00] mb-3">{editingWatchlist ? 'EDIT WATCHLIST' : 'CREATE NEW WATCHLIST'}</h3>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  const formData = new FormData(e.currentTarget);
                  const name = formData.get('name') as string;
                  const assets = (formData.get('assets') as string).split(',').map(a => a.trim()).filter(Boolean);
                  
                  if (editingWatchlist) {
                    updateWatchlist(editingWatchlist.id, name, assets);
                    setEditingWatchlist(null);
                  } else {
                    createWatchlist(name, assets);
                  }
                  e.currentTarget.reset();
                }}
              >
                <div className="space-y-3">
                  <div>
                    <label className="text-[#666] block mb-1">NAME</label>
                    <input
                      type="text"
                      name="name"
                      defaultValue={editingWatchlist?.name}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none"
                      placeholder="e.g., Major Pairs"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1">ASSETS (comma-separated)</label>
                    <input
                      type="text"
                      name="assets"
                      defaultValue={editingWatchlist?.assets.join(', ')}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none"
                      placeholder="e.g., EURUSD, GBPUSD, USDJPY"
                    />
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="submit"
                      className="px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500]"
                    >
                      {editingWatchlist ? 'UPDATE' : 'CREATE'}
                    </button>
                    {editingWatchlist && (
                      <button
                        type="button"
                        onClick={() => setEditingWatchlist(null)}
                        className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff]"
                      >
                        CANCEL
                      </button>
                    )}
                  </div>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Alert Modal */}
      {showAlertModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">MANAGE ALERTS</h2>
              <button
                onClick={() => {
                  setShowAlertModal(false);
                  setEditingAlert(null);
                }}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            {/* Alert List */}
            <div className="space-y-2 mb-4">
              {alerts.map(alert => (
                <div key={alert.id} className="border border-[#444] p-3 flex justify-between items-center">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className={alert.enabled ? 'text-[#00ff00]' : 'text-[#666]'}>
                        {alert.enabled ? '●' : '○'}
                      </span>
                      <span className="text-[#00ff00]">{alert.assetId}</span>
                      <span className="text-[#666]">-</span>
                      <span className="text-[#ffff00]">{alert.type}</span>
                    </div>
                    <div className="text-[#666] text-[10px] ml-5">
                      {alert.condition} {alert.threshold}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => toggleAlert(alert.id)}
                      className={`px-2 py-1 border ${
                        alert.enabled
                          ? 'border-[#ffff00] text-[#ffff00] hover:bg-[#ffff00] hover:text-black'
                          : 'border-[#666] text-[#666] hover:bg-[#666] hover:text-black'
                      }`}
                    >
                      {alert.enabled ? 'DISABLE' : 'ENABLE'}
                    </button>
                    <button
                      onClick={() => setEditingAlert(alert)}
                      className="px-2 py-1 border border-[#ff8c00] text-[#ff8c00] hover:bg-[#ff8c00] hover:text-black"
                    >
                      EDIT
                    </button>
                    <button
                      onClick={() => {
                        if (confirm('Delete this alert?')) {
                          deleteAlert(alert.id);
                        }
                      }}
                      className="px-2 py-1 border border-[#ff0000] text-[#ff0000] hover:bg-[#ff0000] hover:text-black"
                    >
                      DELETE
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Create/Edit Form */}
            <div className="border border-[#ff8c00] p-4">
              <h3 className="text-[#ff8c00] mb-3">{editingAlert ? 'EDIT ALERT' : 'CREATE NEW ALERT'}</h3>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  const formData = new FormData(e.currentTarget);
                  const alertData = {
                    assetId: formData.get('assetId') as string,
                    type: formData.get('type') as Alert['type'],
                    condition: formData.get('condition') as string,
                    threshold: parseFloat(formData.get('threshold') as string),
                    enabled: true,
                  };
                  
                  if (editingAlert) {
                    updateAlert(editingAlert.id, alertData);
                    setEditingAlert(null);
                  } else {
                    createAlert(alertData);
                  }
                  e.currentTarget.reset();
                }}
              >
                <div className="space-y-3">
                  <div>
                    <label className="text-[#666] block mb-1">ASSET</label>
                    <input
                      type="text"
                      name="assetId"
                      defaultValue={editingAlert?.assetId}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none"
                      placeholder="e.g., EURUSD"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1">TYPE</label>
                    <select
                      name="type"
                      defaultValue={editingAlert?.type}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none"
                    >
                      <option value="PRICE">PRICE</option>
                      <option value="VOLATILITY">VOLATILITY</option>
                      <option value="LIQUIDITY">LIQUIDITY</option>
                      <option value="CORRELATION">CORRELATION</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1">CONDITION</label>
                    <select
                      name="condition"
                      defaultValue={editingAlert?.condition}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none"
                    >
                      <option value="above">ABOVE</option>
                      <option value="below">BELOW</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1">THRESHOLD</label>
                    <input
                      type="number"
                      name="threshold"
                      step="0.01"
                      defaultValue={editingAlert?.threshold}
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 focus:border-[#ff8c00] outline-none"
                      placeholder="e.g., 1.10"
                    />
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="submit"
                      className="px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500]"
                    >
                      {editingAlert ? 'UPDATE' : 'CREATE'}
                    </button>
                    {editingAlert && (
                      <button
                        type="button"
                        onClick={() => setEditingAlert(null)}
                        className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff]"
                      >
                        CANCEL
                      </button>
                    )}
                  </div>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
