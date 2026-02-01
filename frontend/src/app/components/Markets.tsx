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

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

type MarketCategory = 'WATCHLISTS' | 'ALERTS' | 'INSTRUMENTS' | 'ANALYTICS' | 'EVENTS';

interface WatchlistItem {
  id: string;
  name: string;
  assets: string[];
  createdAt: string;
}

interface Alert {
  id: string;
  assetId: string;
  type: 'PRICE' | 'VOLATILITY' | 'LIQUIDITY' | 'CORRELATION' | 'SPREAD';
  condition: 'above' | 'below' | 'crosses';
  threshold: number;
  enabled: boolean;
  triggeredAt?: string;
  notificationMethod: 'UI' | 'EMAIL' | 'BOTH';
}

interface Instrument {
  id: string;
  symbol: string;
  name: string;
  type: 'FOREX' | 'CRYPTO' | 'SYNTHETIC' | 'STOCK' | 'INDEX';
  exchange: string;
  tradable: boolean;
  marginRequired: number;
  minLotSize: number;
  maxLotSize: number;
}

// ============================================================================
// MOCK DATA GENERATORS
// ============================================================================

const generateMockLiveData = (assets: string[]): LiveMarketData[] => {
  return assets.map(assetId => {
    const hash = assetId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const basePrice = 1.0 + (hash % 100) * 0.01;
    return {
      asset_id: assetId,
      timestamp: new Date().toISOString(),
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

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export function Markets() {
  // Category selection state
  const [selectedCategory, setSelectedCategory] = useState<MarketCategory>('WATCHLISTS');

  // Market data state
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
  const [selectedDerivCategory, setSelectedDerivCategory] = useState<'forex' | 'synthetic'>('forex');

  // User engagement state
  const [watchlists, setWatchlists] = useState<WatchlistItem[]>([
    { id: '1', name: 'Major Pairs', assets: ['EURUSD', 'GBPUSD', 'USDJPY'], createdAt: '2024-01-01' },
    { id: '2', name: 'Crypto', assets: ['BTCUSD', 'ETHUSD'], createdAt: '2024-01-05' },
    { id: '3', name: 'Emerging Markets', assets: ['USDTRY', 'USDZAR', 'USDMXN'], createdAt: '2024-01-10' },
  ]);
  const [activeWatchlist, setActiveWatchlist] = useState<string>('1');

  const [alerts, setAlerts] = useState<Alert[]>([
    { id: '1', assetId: 'EURUSD', type: 'PRICE', condition: 'above', threshold: 1.10, enabled: true, notificationMethod: 'UI' },
    { id: '2', assetId: 'BTCUSD', type: 'VOLATILITY', condition: 'above', threshold: 50, enabled: true, notificationMethod: 'BOTH' },
    { id: '3', assetId: 'GBPUSD', type: 'SPREAD', condition: 'above', threshold: 3.0, enabled: false, notificationMethod: 'EMAIL' },
  ]);

  const [instruments, setInstruments] = useState<Instrument[]>([
    { id: '1', symbol: 'EURUSD', name: 'Euro / US Dollar', type: 'FOREX', exchange: 'DERIV', tradable: true, marginRequired: 3.33, minLotSize: 0.01, maxLotSize: 100 },
    { id: '2', symbol: 'GBPUSD', name: 'British Pound / US Dollar', type: 'FOREX', exchange: 'DERIV', tradable: true, marginRequired: 3.33, minLotSize: 0.01, maxLotSize: 100 },
    { id: '3', symbol: 'USDJPY', name: 'US Dollar / Japanese Yen', type: 'FOREX', exchange: 'DERIV', tradable: true, marginRequired: 3.33, minLotSize: 0.01, maxLotSize: 100 },
    { id: '4', symbol: 'BTCUSD', name: 'Bitcoin / US Dollar', type: 'CRYPTO', exchange: 'DERIV', tradable: true, marginRequired: 50, minLotSize: 0.001, maxLotSize: 10 },
    { id: '5', symbol: 'Volatility_100', name: 'Volatility 100 Index', type: 'SYNTHETIC', exchange: 'DERIV', tradable: true, marginRequired: 0.5, minLotSize: 0.01, maxLotSize: 500 },
  ]);

  // Selection state
  const [selectedWatchlist, setSelectedWatchlist] = useState<WatchlistItem | null>(watchlists[0]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [selectedInstrument, setSelectedInstrument] = useState<Instrument | null>(null);

  // Modal states
  const [showWatchlistModal, setShowWatchlistModal] = useState(false);
  const [showAlertModal, setShowAlertModal] = useState(false);
  const [showInstrumentModal, setShowInstrumentModal] = useState(false);
  const [editingWatchlist, setEditingWatchlist] = useState<WatchlistItem | null>(null);
  const [editingAlert, setEditingAlert] = useState<Alert | null>(null);

  const watchedAssets = selectedWatchlist?.assets || ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'BTCUSD'];

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatAssetSymbol = (assetId: string) => {
    if (assetId.length === 6) return `${assetId.slice(0, 3)}/${assetId.slice(3)}`;
    if (assetId === 'BTCUSD') return 'BTC/USD';
    return assetId;
  };

  // Fetch market data
  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        setIsLoading(true);
        setError(null);

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
    const interval = setInterval(fetchMarketData, 5000);
    return () => clearInterval(interval);
  }, [activeWatchlist]);

  // Fetch Deriv data
  useEffect(() => {
    const fetchDerivData = async () => {
      try {
        const account = await getDerivAccount();
        setDerivAccount(account);
        const status = await getDerivStatus();
        setDerivStatus(status);

        const symbolsToFetch = selectedDerivCategory === 'forex' ? POPULAR_FOREX : POPULAR_SYNTHETICS;
        const ticks = await Promise.all(symbolsToFetch.map(symbol => getDerivTick(symbol)));

        const tickMap = new Map<string, DerivTick>();
        ticks.forEach(tick => tickMap.set(tick.symbol, tick));
        setDerivTicks(tickMap);
      } catch (err) {
        console.warn('Failed to fetch Deriv data:', err);
      }
    };

    fetchDerivData();
    const interval = setInterval(fetchDerivData, 1000);
    return () => clearInterval(interval);
  }, [selectedDerivCategory]);

  // CRUD operations for watchlists
  const createWatchlist = (name: string, assets: string[]) => {
    const newWatchlist: WatchlistItem = {
      id: Date.now().toString(),
      name,
      assets,
      createdAt: new Date().toISOString(),
    };
    setWatchlists([...watchlists, newWatchlist]);
    setSelectedWatchlist(newWatchlist);
  };

  const updateWatchlist = (id: string, name: string, assets: string[]) => {
    setWatchlists(watchlists.map(w => w.id === id ? { ...w, name, assets } : w));
    if (selectedWatchlist?.id === id) {
      setSelectedWatchlist({ ...selectedWatchlist, name, assets });
    }
  };

  const deleteWatchlist = (id: string) => {
    setWatchlists(watchlists.filter(w => w.id !== id));
    if (selectedWatchlist?.id === id) {
      setSelectedWatchlist(watchlists[0] || null);
    }
  };

  // CRUD operations for alerts
  const createAlert = (alert: Omit<Alert, 'id'>) => {
    const newAlert: Alert = { ...alert, id: Date.now().toString() };
    setAlerts([...alerts, newAlert]);
  };

  const updateAlert = (id: string, updates: Partial<Alert>) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, ...updates } : a));
  };

  const deleteAlert = (id: string) => {
    setAlerts(alerts.filter(a => a.id !== id));
    if (selectedAlert?.id === id) setSelectedAlert(null);
  };

  const toggleAlert = (id: string) => {
    setAlerts(alerts.map(a => a.id === id ? { ...a, enabled: !a.enabled } : a));
  };

  // Calculate change percentage
  const calculateChange = (assetId: string) => {
    const hash = assetId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return ((hash % 500) - 250) / 100;
  };

  // Build correlation pairs
  const correlationPairs = [];
  if (correlations && correlations.assets.length > 1) {
    for (let i = 0; i < correlations.assets.length; i++) {
      for (let j = i + 1; j < correlations.assets.length; j++) {
        const corr = correlations.correlation_matrix[i]?.[j] || 0;
        const absCorr = Math.abs(corr);
        const strength = absCorr > 0.7 ? 'STRONG' : absCorr > 0.4 ? 'MODERATE' : 'WEAK';
        correlationPairs.push({
          pair: `${formatAssetSymbol(correlations.assets[i])} - ${formatAssetSymbol(correlations.assets[j])}`,
          correlation: corr,
          strength,
          sign: corr >= 0 ? 'POSITIVE' : 'NEGATIVE',
        });
      }
    }
  }

  // Liquidity summary
  const liquiditySummary = liquidity.length > 0 ? {
    avgLiquidityScore: liquidity.reduce((sum, l) => sum + l.liquidity_score, 0) / liquidity.length,
    avgResilienceScore: liquidity.reduce((sum, l) => sum + l.resilience_score, 0) / liquidity.length,
    avgToxicityScore: liquidity.reduce((sum, l) => sum + l.toxicity_score, 0) / liquidity.length,
  } : null;

  // Category items for left panel
  const categories: { key: MarketCategory; label: string; count: number }[] = [
    { key: 'WATCHLISTS', label: 'WATCHLISTS', count: watchlists.length },
    { key: 'ALERTS', label: 'PRICE ALERTS', count: alerts.filter(a => a.enabled).length },
    { key: 'INSTRUMENTS', label: 'INSTRUMENTS', count: instruments.length },
    { key: 'ANALYTICS', label: 'ANALYTICS', count: 4 },
    { key: 'EVENTS', label: 'MARKET EVENTS', count: events.length },
  ];

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel - Category Selection */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">MARKET CATEGORIES</div>

          {/* Category Buttons */}
          <div className="space-y-2 mb-4">
            {categories.map(cat => (
              <button
                key={cat.key}
                onClick={() => setSelectedCategory(cat.key)}
                className={`w-full py-2 px-3 border text-left text-[10px] transition-colors flex justify-between items-center ${
                  selectedCategory === cat.key
                    ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                    : 'border-[#444] text-[#00ff00] hover:border-[#ff8c00]'
                }`}
              >
                <span>{cat.label}</span>
                <span className="text-[#666]">{cat.count}</span>
              </button>
            ))}
          </div>

          {/* Category-specific list */}
          {selectedCategory === 'WATCHLISTS' && (
            <div className="space-y-2">
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">YOUR WATCHLISTS</div>
              {watchlists.map(wl => (
                <div
                  key={wl.id}
                  onClick={() => setSelectedWatchlist(wl)}
                  className={`border p-2 cursor-pointer transition-colors ${
                    selectedWatchlist?.id === wl.id
                      ? 'border-[#ff8c00] bg-[#1a1a1a]'
                      : 'border-[#333] hover:border-[#ff8c00]'
                  }`}
                >
                  <div className="text-[#00ff00] text-[10px]">{wl.name}</div>
                  <div className="text-[#666] text-[9px]">{wl.assets.length} assets</div>
                </div>
              ))}
            </div>
          )}

          {selectedCategory === 'ALERTS' && (
            <div className="space-y-2">
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">ACTIVE ALERTS</div>
              {alerts.map(alert => (
                <div
                  key={alert.id}
                  onClick={() => setSelectedAlert(alert)}
                  className={`border p-2 cursor-pointer transition-colors ${
                    selectedAlert?.id === alert.id
                      ? 'border-[#ff8c00] bg-[#1a1a1a]'
                      : 'border-[#333] hover:border-[#ff8c00]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={alert.enabled ? 'text-[#00ff00]' : 'text-[#666]'}>
                      {alert.enabled ? '●' : '○'} {alert.assetId}
                    </span>
                    <span className="text-[#ffff00] text-[9px]">{alert.type}</span>
                  </div>
                  <div className="text-[#666] text-[9px]">{alert.condition} {alert.threshold}</div>
                </div>
              ))}
            </div>
          )}

          {selectedCategory === 'INSTRUMENTS' && (
            <div className="space-y-2">
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">TRADABLE INSTRUMENTS</div>
              {instruments.map(inst => (
                <div
                  key={inst.id}
                  onClick={() => setSelectedInstrument(inst)}
                  className={`border p-2 cursor-pointer transition-colors ${
                    selectedInstrument?.id === inst.id
                      ? 'border-[#ff8c00] bg-[#1a1a1a]'
                      : 'border-[#333] hover:border-[#ff8c00]'
                  }`}
                >
                  <div className="text-[#00ff00] text-[10px]">{inst.symbol}</div>
                  <div className="text-[#666] text-[9px]">{inst.name}</div>
                  <div className="text-[#ffff00] text-[8px]">{inst.type}</div>
                </div>
              ))}
            </div>
          )}

          {selectedCategory === 'ANALYTICS' && (
            <div className="space-y-2">
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">ANALYTICS VIEWS</div>
              {['CORRELATIONS', 'MICROSTRUCTURE', 'LIQUIDITY', 'VOLATILITY'].map(view => (
                <div key={view} className="border border-[#333] p-2 hover:border-[#ff8c00] cursor-pointer">
                  <div className="text-[#00ff00] text-[10px]">{view}</div>
                </div>
              ))}
            </div>
          )}

          {selectedCategory === 'EVENTS' && (
            <div className="space-y-2">
              <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RECENT EVENTS</div>
              {events.slice(0, 5).map((event, idx) => (
                <div key={idx} className="border border-[#333] p-2">
                  <div className="text-[#00ff00] text-[10px]">{event.asset_id}</div>
                  <div className="text-[#ffff00] text-[9px]">{event.event_type}</div>
                  <div className="text-[#666] text-[8px]">{new Date(event.timestamp).toLocaleTimeString()}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Center Panel - Main View */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="flex items-center justify-between">
            <div className="text-[#ff8c00] text-sm tracking-wider">
              MARKET DATA - {selectedCategory}
              {isLoading && <span className="ml-2 text-[10px]">LOADING...</span>}
              {useMockData && <span className="ml-2 text-[#ffff00] text-[10px]">MOCK DATA MODE</span>}
            </div>
            <div className={`px-3 py-1 border text-[10px] ${isBackendConnected ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#ff0000] text-[#ff0000]'}`}>
              {isBackendConnected ? '● LIVE' : '● OFFLINE'}
            </div>
          </div>
        </div>

        {/* Watchlists View */}
        {selectedCategory === 'WATCHLISTS' && selectedWatchlist && (
          <div className="space-y-4">
            {/* Live Market Data Table */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444] flex justify-between items-center">
                <div className="text-[#ff8c00]">{selectedWatchlist.name} - LIVE PRICES</div>
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

            {/* Deriv Live Prices */}
            <div className="border border-[#ff8c00]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#ff8c00] flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div className="text-[#ff8c00]">DERIV LIVE PRICES</div>
                  {derivStatus && (
                    <div className={`text-[10px] px-2 py-1 border ${derivStatus.connected ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#666] text-[#666]'}`}>
                      {derivStatus.connected ? '● CONNECTED' : '○ OFFLINE'}
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedDerivCategory('forex')}
                    className={`px-3 py-1 text-[10px] border ${selectedDerivCategory === 'forex' ? 'border-[#ff8c00] bg-[#ff8c00] text-black' : 'border-[#444] text-[#666]'}`}
                  >
                    FOREX
                  </button>
                  <button
                    onClick={() => setSelectedDerivCategory('synthetic')}
                    className={`px-3 py-1 text-[10px] border ${selectedDerivCategory === 'synthetic' ? 'border-[#ff8c00] bg-[#ff8c00] text-black' : 'border-[#444] text-[#666]'}`}
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
                    <th className="px-3 py-2 text-right border-b border-[#444]">SPREAD (BPS)</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from(derivTicks.entries()).map(([symbol, tick]) => {
                    const spreadBps = calculateSpreadBps(tick);
                    const marketOpen = isMarketOpen(symbol);
                    return (
                      <tr key={symbol} className="border-b border-[#222]">
                        <td className="px-3 py-2 text-[#00ff00] font-bold">{formatSymbol(symbol)}</td>
                        <td className="px-3 py-2 text-right text-[#fff]">{tick.bid.toFixed(5)}</td>
                        <td className="px-3 py-2 text-right text-[#fff]">{tick.ask.toFixed(5)}</td>
                        <td className="px-3 py-2 text-right text-[#ff8c00]">{spreadBps.toFixed(2)}</td>
                        <td className="px-3 py-2 text-center">
                          <span className={`text-[8px] px-2 py-1 border ${marketOpen ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#666] text-[#666]'}`}>
                            {marketOpen ? 'OPEN' : 'CLOSED'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Alerts View */}
        {selectedCategory === 'ALERTS' && (
          <div className="space-y-4">
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">ALERT CONFIGURATION</div>
              </div>
              {selectedAlert ? (
                <div className="p-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-[#666] text-[10px]">ASSET</div>
                      <div className="text-[#00ff00]">{selectedAlert.assetId}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">TYPE</div>
                      <div className="text-[#ffff00]">{selectedAlert.type}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">CONDITION</div>
                      <div className="text-[#fff]">{selectedAlert.condition} {selectedAlert.threshold}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">STATUS</div>
                      <div className={selectedAlert.enabled ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                        {selectedAlert.enabled ? 'ENABLED' : 'DISABLED'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">NOTIFICATION</div>
                      <div className="text-[#fff]">{selectedAlert.notificationMethod}</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-4 text-[#666] text-center">SELECT AN ALERT TO VIEW DETAILS</div>
              )}
            </div>
          </div>
        )}

        {/* Instruments View */}
        {selectedCategory === 'INSTRUMENTS' && (
          <div className="space-y-4">
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">INSTRUMENT DETAILS</div>
              </div>
              {selectedInstrument ? (
                <div className="p-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-[#666] text-[10px]">SYMBOL</div>
                      <div className="text-[#00ff00] text-lg">{selectedInstrument.symbol}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">NAME</div>
                      <div className="text-[#fff]">{selectedInstrument.name}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">TYPE</div>
                      <div className="text-[#ffff00]">{selectedInstrument.type}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">EXCHANGE</div>
                      <div className="text-[#fff]">{selectedInstrument.exchange}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">MARGIN REQUIRED</div>
                      <div className="text-[#fff]">{selectedInstrument.marginRequired}%</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">LOT SIZE RANGE</div>
                      <div className="text-[#fff]">{selectedInstrument.minLotSize} - {selectedInstrument.maxLotSize}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[10px]">TRADABLE</div>
                      <div className={selectedInstrument.tradable ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                        {selectedInstrument.tradable ? 'YES' : 'NO'}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-4 text-[#666] text-center">SELECT AN INSTRUMENT TO VIEW DETAILS</div>
              )}
            </div>
          </div>
        )}

        {/* Analytics View */}
        {selectedCategory === 'ANALYTICS' && (
          <div className="space-y-4">
            {/* Correlations */}
            <div className="border border-[#444]">
              <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
                <div className="text-[#ff8c00]">ASSET CORRELATIONS (24H)</div>
              </div>
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                    <th className="px-3 py-2 text-left border-b border-[#444]">PAIR</th>
                    <th className="px-3 py-2 text-right border-b border-[#444]">CORRELATION</th>
                    <th className="px-3 py-2 text-center border-b border-[#444]">STRENGTH</th>
                  </tr>
                </thead>
                <tbody>
                  {correlationPairs.slice(0, 6).map((corr, idx) => (
                    <tr key={idx} className="border-b border-[#222]">
                      <td className="px-3 py-2 text-[#00ff00]">{corr.pair}</td>
                      <td className={`px-3 py-2 text-right ${corr.correlation >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                        {formatNumber(corr.correlation)}
                      </td>
                      <td className="px-3 py-2 text-center text-[#ffff00]">{corr.strength}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Microstructure & Liquidity */}
            <div className="grid grid-cols-2 gap-4">
              <div className="border border-[#444] p-3">
                <div className="text-[#ff8c00] mb-3">MARKET MICROSTRUCTURE</div>
                {microstructure && (
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
                  </div>
                )}
              </div>

              <div className="border border-[#444] p-3">
                <div className="text-[#ff8c00] mb-3">LIQUIDITY ANALYSIS</div>
                {liquiditySummary && (
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
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Events View */}
        {selectedCategory === 'EVENTS' && (
          <div className="border border-[#444]">
            <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
              <div className="text-[#ff8c00]">MARKET EVENTS LOG</div>
            </div>
            <div className="p-4 space-y-2">
              {events.map((event, idx) => (
                <div key={idx} className="border border-[#333] p-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="text-[#00ff00]">{event.asset_id}</div>
                      <div className="text-[#ffff00] text-[10px]">{event.event_type}</div>
                    </div>
                    <div className="text-[#666] text-[10px]">{new Date(event.timestamp).toLocaleString()}</div>
                  </div>
                  <div className="text-[#fff] text-[10px] mt-2">{event.description}</div>
                  <div className="flex justify-between mt-2 text-[9px]">
                    <span className="text-[#666]">SEVERITY: <span className={event.severity > 0.7 ? 'text-[#ff0000]' : 'text-[#ffff00]'}>{formatNumber(event.severity * 100, 0)}%</span></span>
                    <span className="text-[#ff8c00]">{event.recommended_action}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">MARKET ACTIONS</div>

          {/* Watchlist Actions */}
          {selectedCategory === 'WATCHLISTS' && (
            <div className="space-y-2 mb-6">
              <button
                onClick={() => { setEditingWatchlist(null); setShowWatchlistModal(true); }}
                className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
              >
                + CREATE WATCHLIST
              </button>
              {selectedWatchlist && (
                <>
                  <button
                    onClick={() => { setEditingWatchlist(selectedWatchlist); setShowWatchlistModal(true); }}
                    className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                  >
                    EDIT WATCHLIST
                  </button>
                  <button
                    onClick={() => { if (confirm(`Delete "${selectedWatchlist.name}"?`)) deleteWatchlist(selectedWatchlist.id); }}
                    className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                  >
                    DELETE WATCHLIST
                  </button>
                </>
              )}
            </div>
          )}

          {/* Alert Actions */}
          {selectedCategory === 'ALERTS' && (
            <div className="space-y-2 mb-6">
              <button
                onClick={() => { setEditingAlert(null); setShowAlertModal(true); }}
                className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
              >
                + CREATE ALERT
              </button>
              {selectedAlert && (
                <>
                  <button
                    onClick={() => toggleAlert(selectedAlert.id)}
                    className={`w-full py-2 px-3 border text-[10px] transition-colors ${
                      selectedAlert.enabled
                        ? 'border-[#ffff00] text-[#ffff00] hover:bg-[#ffff00] hover:text-black'
                        : 'border-[#00ff00] text-[#00ff00] hover:bg-[#00ff00] hover:text-black'
                    }`}
                  >
                    {selectedAlert.enabled ? 'DISABLE ALERT' : 'ENABLE ALERT'}
                  </button>
                  <button
                    onClick={() => { setEditingAlert(selectedAlert); setShowAlertModal(true); }}
                    className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                  >
                    EDIT ALERT
                  </button>
                  <button
                    onClick={() => { if (confirm('Delete this alert?')) deleteAlert(selectedAlert.id); }}
                    className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                  >
                    DELETE ALERT
                  </button>
                </>
              )}
            </div>
          )}

          {/* Instrument Actions */}
          {selectedCategory === 'INSTRUMENTS' && selectedInstrument && (
            <div className="space-y-2 mb-6">
              <button className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors">
                ADD TO WATCHLIST
              </button>
              <button className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors">
                CREATE ALERT
              </button>
              <button className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors">
                VIEW CHART
              </button>
            </div>
          )}

          {/* Analytics Actions */}
          {selectedCategory === 'ANALYTICS' && (
            <div className="space-y-2 mb-6">
              <button className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors">
                REFRESH CORRELATIONS
              </button>
              <button className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors">
                EXPORT ANALYSIS
              </button>
              <button className="w-full py-2 px-3 border border-[#444] text-[#00ff00] text-[10px] hover:border-[#00ff00] transition-colors">
                CONFIGURE WINDOW
              </button>
            </div>
          )}

          {/* Quick Stats */}
          <div className="border border-[#444] p-3 bg-[#0a0a0a]">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">MARKET SUMMARY</div>
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">WATCHLISTS</span>
                <span className="text-[#fff]">{watchlists.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">ACTIVE ALERTS</span>
                <span className="text-[#00ff00]">{alerts.filter(a => a.enabled).length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">INSTRUMENTS</span>
                <span className="text-[#fff]">{instruments.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">DERIV STATUS</span>
                <span className={derivStatus?.connected ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                  {derivStatus?.connected ? 'CONNECTED' : 'OFFLINE'}
                </span>
              </div>
              {derivAccount && (
                <div className="flex justify-between">
                  <span className="text-[#666]">BALANCE</span>
                  <span className="text-[#00ff00]">{formatNumber(derivAccount.balance)} {derivAccount.currency}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Watchlist Modal */}
      {showWatchlistModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">{editingWatchlist ? 'EDIT WATCHLIST' : 'CREATE WATCHLIST'}</h2>
              <button onClick={() => setShowWatchlistModal(false)} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
            </div>
            <form onSubmit={(e) => {
              e.preventDefault();
              const formData = new FormData(e.currentTarget);
              const name = formData.get('name') as string;
              const assets = (formData.get('assets') as string).split(',').map(a => a.trim()).filter(Boolean);
              if (editingWatchlist) {
                updateWatchlist(editingWatchlist.id, name, assets);
              } else {
                createWatchlist(name, assets);
              }
              setShowWatchlistModal(false);
            }}>
              <div className="space-y-4">
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">NAME</label>
                  <input type="text" name="name" required defaultValue={editingWatchlist?.name}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-3 py-2 text-[10px]" placeholder="e.g., Major Pairs" />
                </div>
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">ASSETS (comma-separated)</label>
                  <input type="text" name="assets" required defaultValue={editingWatchlist?.assets.join(', ')}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-3 py-2 text-[10px]" placeholder="e.g., EURUSD, GBPUSD" />
                </div>
                <button type="submit" className="w-full py-2 bg-[#ff8c00] text-black text-[10px] hover:bg-[#ffa500]">
                  {editingWatchlist ? 'UPDATE' : 'CREATE'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Alert Modal */}
      {showAlertModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">{editingAlert ? 'EDIT ALERT' : 'CREATE ALERT'}</h2>
              <button onClick={() => setShowAlertModal(false)} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
            </div>
            <form onSubmit={(e) => {
              e.preventDefault();
              const formData = new FormData(e.currentTarget);
              const alertData = {
                assetId: formData.get('assetId') as string,
                type: formData.get('type') as Alert['type'],
                condition: formData.get('condition') as 'above' | 'below' | 'crosses',
                threshold: parseFloat(formData.get('threshold') as string),
                enabled: true,
                notificationMethod: formData.get('notificationMethod') as 'UI' | 'EMAIL' | 'BOTH',
              };
              if (editingAlert) {
                updateAlert(editingAlert.id, alertData);
              } else {
                createAlert(alertData);
              }
              setShowAlertModal(false);
            }}>
              <div className="space-y-4">
                <div>
                  <label className="text-[#666] block mb-1 text-[10px]">ASSET</label>
                  <input type="text" name="assetId" required defaultValue={editingAlert?.assetId}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-3 py-2 text-[10px]" placeholder="e.g., EURUSD" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">TYPE</label>
                    <select name="type" defaultValue={editingAlert?.type || 'PRICE'}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-3 py-2 text-[10px]">
                      <option value="PRICE">PRICE</option>
                      <option value="VOLATILITY">VOLATILITY</option>
                      <option value="LIQUIDITY">LIQUIDITY</option>
                      <option value="SPREAD">SPREAD</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">CONDITION</label>
                    <select name="condition" defaultValue={editingAlert?.condition || 'above'}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-3 py-2 text-[10px]">
                      <option value="above">ABOVE</option>
                      <option value="below">BELOW</option>
                      <option value="crosses">CROSSES</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">THRESHOLD</label>
                    <input type="number" name="threshold" step="0.0001" required defaultValue={editingAlert?.threshold}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-3 py-2 text-[10px]" />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[10px]">NOTIFICATION</label>
                    <select name="notificationMethod" defaultValue={editingAlert?.notificationMethod || 'UI'}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-3 py-2 text-[10px]">
                      <option value="UI">UI ONLY</option>
                      <option value="EMAIL">EMAIL</option>
                      <option value="BOTH">BOTH</option>
                    </select>
                  </div>
                </div>
                <button type="submit" className="w-full py-2 bg-[#ff8c00] text-black text-[10px] hover:bg-[#ffa500]">
                  {editingAlert ? 'UPDATE' : 'CREATE'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
