import { useState, useEffect } from 'react';
import {
  getNautilusSystemStatus,
  listBacktests,
  getBacktestStatus,
  createBacktest,
  cancelBacktest,
  pauseBacktest,
  resumeBacktest,
  startLiveTrading,
  getLiveTradingStatus,
  stopLiveTrading,
  pauseLiveTrading,
  resumeLiveTrading,
  listStrategyTranslations,
  startDataMigration,
  getDataMigrationStatus,
  getStatusColor,
  formatDuration,
  formatPercentage,
  formatCurrency,
  type NautilusSystemStatus,
  type BacktestResult,
  type BacktestConfig,
  type LiveTradingSession,
  type StrategyTranslation,
  type DataMigration,
  type ServiceStatus,
  BacktestStatus,
  LiveTradingStatus,
  StrategyTranslationStatus,
} from '../../services/nautilusService';

// Types
type NautilusCategory = 'LIVE_SESSIONS' | 'BACKTESTS' | 'TRANSLATIONS' | 'DATA_CATALOG' | 'SYSTEM_STATUS';

export function NautilusTrading() {
  // Category selection state
  const [selectedCategory, setSelectedCategory] = useState<NautilusCategory>('LIVE_SESSIONS');

  // Data state
  const [systemStatus, setSystemStatus] = useState<NautilusSystemStatus | null>(null);
  const [backtests, setBacktests] = useState<BacktestResult[]>([]);
  const [liveSessions, setLiveSessions] = useState<LiveTradingSession[]>([]);
  const [translations, setTranslations] = useState<StrategyTranslation[]>([]);
  const [dataMigrations, setDataMigrations] = useState<DataMigration[]>([]);

  // Selection state
  const [selectedBacktest, setSelectedBacktest] = useState<BacktestResult | null>(null);
  const [selectedSession, setSelectedSession] = useState<LiveTradingSession | null>(null);
  const [selectedTranslation, setSelectedTranslation] = useState<StrategyTranslation | null>(null);
  const [selectedMigration, setSelectedMigration] = useState<DataMigration | null>(null);

  // Modal state
  const [showNewBacktestModal, setShowNewBacktestModal] = useState(false);
  const [showNewSessionModal, setShowNewSessionModal] = useState(false);
  const [showDataMigrationModal, setShowDataMigrationModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);

  // Status state
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(true);

  // Mock live sessions
  const [mockLiveSessions] = useState<LiveTradingSession[]>([
    {
      id: 'lt_001',
      name: 'Trend Following Live',
      description: 'Live trend following strategy on EURUSD',
      strategies: [],
      status: LiveTradingStatus.ACTIVE,
      startedAt: new Date(Date.now() - 86400000).toISOString(),
      currentCapital: 102500,
      initialCapital: 100000,
      totalPnl: 2500,
      dailyPnl: 750,
      activePositions: [
        {
          id: 'pos_001',
          strategyId: 'trend_following',
          instrument: 'EURUSD',
          side: 'LONG',
          quantity: 50000,
          avgPrice: 1.0845,
          unrealizedPnl: 125.50,
          realizedPnl: 0,
          timestamp: new Date().toISOString(),
        }
      ],
      recentTrades: [],
      riskMetrics: {
        currentDrawdown: 0.5,
        maxDrawdown: 2.1,
        currentLeverage: 1.5,
        maxLeverage: 3.0,
        var95: 1250,
        var99: 2100,
        riskLimitBreaches: [],
      },
    },
    {
      id: 'lt_002',
      name: 'Mean Reversion Paper',
      description: 'Paper trading mean reversion strategy',
      strategies: [],
      status: LiveTradingStatus.PAUSED,
      startedAt: new Date(Date.now() - 172800000).toISOString(),
      currentCapital: 98500,
      initialCapital: 100000,
      totalPnl: -1500,
      dailyPnl: -200,
      activePositions: [],
      recentTrades: [],
      riskMetrics: {
        currentDrawdown: 1.5,
        maxDrawdown: 3.2,
        currentLeverage: 0,
        maxLeverage: 2.0,
        var95: 0,
        var99: 0,
        riskLimitBreaches: [],
      },
    },
  ]);

  // Fetch data on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [status, backtestList, translationList] = await Promise.all([
          getNautilusSystemStatus(),
          listBacktests(),
          listStrategyTranslations(),
        ]);
        setSystemStatus(status);
        setBacktests(backtestList);
        setTranslations(translationList);
        setLiveSessions(mockLiveSessions);
        setIsBackendConnected(true);
      } catch (err) {
        console.warn('Failed to fetch Nautilus data:', err);
        setIsBackendConnected(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPnl = (pnl: number) => {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}${formatNumber(pnl)}`;
  };

  // Action handlers
  const handleStartLiveTrading = async () => {
    setIsLoading(true);
    setActionStatus('Starting live trading session...');
    try {
      const session = await startLiveTrading({
        name: 'New Live Session',
        strategies: [],
        initialCapital: 100000,
        riskConfig: {
          maxDrawdown: 20,
          maxLeverage: 3,
          positionLimits: {},
          stopLossEnabled: true,
          killSwitchEnabled: true,
        },
      });
      setLiveSessions([...liveSessions, session]);
      setSelectedSession(session);
      setShowNewSessionModal(false);
      setActionStatus('Live trading session started');
    } catch (err: any) {
      setActionStatus(`Failed to start session: ${err.message}`);
    } finally {
      setIsLoading(false);
      setTimeout(() => setActionStatus(null), 3000);
    }
  };

  const handleStopSession = async (sessionId: string) => {
    setActionStatus('Stopping session...');
    try {
      await stopLiveTrading(sessionId);
      setLiveSessions(liveSessions.map(s =>
        s.id === sessionId ? { ...s, status: LiveTradingStatus.STOPPED } : s
      ));
      setActionStatus('Session stopped');
    } catch (err: any) {
      setActionStatus(`Failed to stop session: ${err.message}`);
    }
    setTimeout(() => setActionStatus(null), 3000);
  };

  const handlePauseSession = async (sessionId: string) => {
    setActionStatus('Pausing session...');
    try {
      await pauseLiveTrading(sessionId);
      setLiveSessions(liveSessions.map(s =>
        s.id === sessionId ? { ...s, status: LiveTradingStatus.PAUSED } : s
      ));
      setActionStatus('Session paused');
    } catch (err: any) {
      setActionStatus(`Failed to pause session: ${err.message}`);
    }
    setTimeout(() => setActionStatus(null), 3000);
  };

  const handleResumeSession = async (sessionId: string) => {
    setActionStatus('Resuming session...');
    try {
      await resumeLiveTrading(sessionId);
      setLiveSessions(liveSessions.map(s =>
        s.id === sessionId ? { ...s, status: LiveTradingStatus.ACTIVE } : s
      ));
      setActionStatus('Session resumed');
    } catch (err: any) {
      setActionStatus(`Failed to resume session: ${err.message}`);
    }
    setTimeout(() => setActionStatus(null), 3000);
  };

  const handleCreateBacktest = async (config: BacktestConfig) => {
    setIsLoading(true);
    setActionStatus('Creating backtest...');
    try {
      const result = await createBacktest(config);
      setBacktests([...backtests, result]);
      setSelectedBacktest(result);
      setShowNewBacktestModal(false);
      setActionStatus('Backtest created');
    } catch (err: any) {
      setActionStatus(`Failed to create backtest: ${err.message}`);
    } finally {
      setIsLoading(false);
      setTimeout(() => setActionStatus(null), 3000);
    }
  };

  const handleCancelBacktest = async (backtestId: string) => {
    setActionStatus('Cancelling backtest...');
    try {
      await cancelBacktest(backtestId);
      setBacktests(backtests.map(b =>
        b.id === backtestId ? { ...b, status: BacktestStatus.CANCELLED } : b
      ));
      setActionStatus('Backtest cancelled');
    } catch (err: any) {
      setActionStatus(`Failed to cancel backtest: ${err.message}`);
    }
    setTimeout(() => setActionStatus(null), 3000);
  };

  const handleStartDataMigration = async (config: any) => {
    setIsLoading(true);
    setActionStatus('Starting data migration...');
    try {
      const migration = await startDataMigration(config);
      setDataMigrations([...dataMigrations, migration]);
      setSelectedMigration(migration);
      setShowDataMigrationModal(false);
      setActionStatus('Data migration started');
    } catch (err: any) {
      setActionStatus(`Failed to start migration: ${err.message}`);
    } finally {
      setIsLoading(false);
      setTimeout(() => setActionStatus(null), 3000);
    }
  };

  // Categories configuration
  const categories: { key: NautilusCategory; label: string; count: number }[] = [
    { key: 'LIVE_SESSIONS', label: 'LIVE TRADING', count: liveSessions.length },
    { key: 'BACKTESTS', label: 'BACKTESTS', count: backtests.length },
    { key: 'TRANSLATIONS', label: 'STRATEGY TRANSLATIONS', count: translations.length },
    { key: 'DATA_CATALOG', label: 'DATA CATALOG', count: dataMigrations.length },
    { key: 'SYSTEM_STATUS', label: 'SYSTEM STATUS', count: systemStatus ? 7 : 0 },
  ];

  // Render service status card
  const renderServiceStatus = (name: string, status: ServiceStatus) => (
    <div className="border border-[#444] p-3 bg-[#0a0a0a]">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[#fff] text-[10px]">{name}</span>
        <span className={`text-[10px] px-2 py-0.5 border ${
          status.status === 'HEALTHY' ? 'border-[#00ff00] text-[#00ff00]' :
          status.status === 'DEGRADED' ? 'border-[#ffff00] text-[#ffff00]' :
          'border-[#ff0000] text-[#ff0000]'
        }`}>
          {status.status}
        </span>
      </div>
      <div className="space-y-1 text-[9px]">
        <div className="flex justify-between">
          <span className="text-[#666]">Version:</span>
          <span className="text-[#00ff00]">{status.version}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#666]">Uptime:</span>
          <span className="text-[#fff]">{formatDuration(status.uptime * 1000)}</span>
        </div>
        {status.metrics && (
          <>
            <div className="flex justify-between">
              <span className="text-[#666]">CPU:</span>
              <span className="text-[#fff]">{status.metrics.cpuUsage.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#666]">Memory:</span>
              <span className="text-[#fff]">{status.metrics.memoryUsage.toFixed(1)}%</span>
            </div>
          </>
        )}
      </div>
    </div>
  );

  // Render left panel content
  const renderLeftPanelContent = () => {
    switch (selectedCategory) {
      case 'LIVE_SESSIONS':
        return (
          <div className="space-y-2">
            {liveSessions.map((session) => (
              <div
                key={session.id}
                onClick={() => setSelectedSession(session)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedSession?.id === session.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{session.name}</div>
                  <span className={`text-[10px] px-2 py-0.5 border ${
                    session.status === LiveTradingStatus.ACTIVE ? 'border-[#00ff00] text-[#00ff00]' :
                    session.status === LiveTradingStatus.PAUSED ? 'border-[#ffff00] text-[#ffff00]' :
                    'border-[#ff0000] text-[#ff0000]'
                  }`}>
                    {session.status}
                  </span>
                </div>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">P&L:</span>
                    <span className={session.totalPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {formatPnl(session.totalPnl)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Daily:</span>
                    <span className={session.dailyPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {formatPnl(session.dailyPnl)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Positions:</span>
                    <span className="text-[#fff]">{session.activePositions.length}</span>
                  </div>
                </div>
              </div>
            ))}
            <button
              onClick={() => setShowNewSessionModal(true)}
              className="w-full py-2 border border-dashed border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black"
            >
              + NEW LIVE SESSION
            </button>
          </div>
        );

      case 'BACKTESTS':
        return (
          <div className="space-y-2">
            {backtests.map((backtest) => (
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
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Progress:</span>
                    <span className="text-[#fff]">{backtest.progress}%</span>
                  </div>
                  {backtest.metrics && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-[#666]">Return:</span>
                        <span className={backtest.metrics.totalReturn >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                          {formatPnl(backtest.metrics.totalReturn)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#666]">Sharpe:</span>
                        <span className="text-[#00ff00]">{backtest.metrics.sharpeRatio.toFixed(2)}</span>
                      </div>
                    </>
                  )}
                </div>
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

      case 'TRANSLATIONS':
        return (
          <div className="space-y-2">
            {translations.map((translation) => (
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
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Progress:</span>
                    <span className="text-[#fff]">{translation.progress}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">ID:</span>
                    <span className="text-[#666]">{translation.id}</span>
                  </div>
                </div>
              </div>
            ))}
            {translations.length === 0 && (
              <div className="text-[#666] text-[10px] text-center py-4">
                No translations yet. Go to Strategy Builder to translate strategies.
              </div>
            )}
          </div>
        );

      case 'DATA_CATALOG':
        return (
          <div className="space-y-2">
            {dataMigrations.map((migration) => (
              <div
                key={migration.id}
                onClick={() => setSelectedMigration(migration)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedMigration?.id === migration.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{migration.sourceType} → NAUTILUS</div>
                  <span className={`text-[10px] ${getStatusColor(migration.status)}`}>
                    {migration.status}
                  </span>
                </div>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Progress:</span>
                    <span className="text-[#fff]">{migration.progress}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Records:</span>
                    <span className="text-[#fff]">{migration.migratedRecords.toLocaleString()} / {migration.totalRecords.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            ))}
            <button
              onClick={() => setShowDataMigrationModal(true)}
              className="w-full py-2 border border-dashed border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black"
            >
              + NEW DATA MIGRATION
            </button>
          </div>
        );

      case 'SYSTEM_STATUS':
        if (!systemStatus) return <div className="text-[#666] text-[10px]">Loading system status...</div>;
        return (
          <div className="space-y-2">
            {Object.entries(systemStatus).map(([key, status]) => (
              <div
                key={key}
                className="border border-[#333] p-3 hover:border-[#ff8c00] cursor-pointer"
              >
                <div className="flex justify-between items-center">
                  <span className="text-[#fff] text-[10px]">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                  <span className={`text-[10px] px-2 py-0.5 border ${
                    status.status === 'HEALTHY' ? 'border-[#00ff00] text-[#00ff00]' :
                    status.status === 'DEGRADED' ? 'border-[#ffff00] text-[#ffff00]' :
                    'border-[#ff0000] text-[#ff0000]'
                  }`}>
                    {status.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        );

      default:
        return null;
    }
  };

  // Render center panel content
  const renderCenterPanelContent = () => {
    switch (selectedCategory) {
      case 'LIVE_SESSIONS':
        if (!selectedSession) return <div className="text-[#666] p-4">Select a live session to view details</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                LIVE SESSION: {selectedSession.name}
              </div>
            </div>

            <div className="space-y-6">
              {/* Status & P&L */}
              <div className="grid grid-cols-4 gap-4">
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">STATUS</div>
                  <div className={`text-[14px] ${
                    selectedSession.status === LiveTradingStatus.ACTIVE ? 'text-[#00ff00]' :
                    selectedSession.status === LiveTradingStatus.PAUSED ? 'text-[#ffff00]' :
                    'text-[#ff0000]'
                  }`}>
                    {selectedSession.status}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">TOTAL P&L</div>
                  <div className={`text-[14px] ${selectedSession.totalPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {formatCurrency(selectedSession.totalPnl)}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">DAILY P&L</div>
                  <div className={`text-[14px] ${selectedSession.dailyPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {formatCurrency(selectedSession.dailyPnl)}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">CAPITAL</div>
                  <div className="text-[#00ff00] text-[14px]">
                    {formatCurrency(selectedSession.currentCapital)}
                  </div>
                </div>
              </div>

              {/* Risk Metrics */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RISK METRICS</div>
                <div className="border border-[#444] bg-[#0a0a0a]">
                  <div className="grid grid-cols-3 gap-4 p-3">
                    <div>
                      <div className="text-[#666] text-[9px]">Current Drawdown</div>
                      <div className="text-[#ff0000] text-[12px]">{selectedSession.riskMetrics.currentDrawdown.toFixed(2)}%</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">Max Drawdown</div>
                      <div className="text-[#ff0000] text-[12px]">{selectedSession.riskMetrics.maxDrawdown.toFixed(2)}%</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">Current Leverage</div>
                      <div className="text-[#ffff00] text-[12px]">{selectedSession.riskMetrics.currentLeverage.toFixed(2)}x</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">VaR 95%</div>
                      <div className="text-[#fff] text-[12px]">{formatCurrency(selectedSession.riskMetrics.var95)}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">VaR 99%</div>
                      <div className="text-[#fff] text-[12px]">{formatCurrency(selectedSession.riskMetrics.var99)}</div>
                    </div>
                    <div>
                      <div className="text-[#666] text-[9px]">Max Leverage</div>
                      <div className="text-[#fff] text-[12px]">{selectedSession.riskMetrics.maxLeverage.toFixed(1)}x</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Active Positions */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">ACTIVE POSITIONS ({selectedSession.activePositions.length})</div>
                <div className="border border-[#444] bg-[#0a0a0a]">
                  {selectedSession.activePositions.length > 0 ? (
                    <table className="w-full">
                      <thead>
                        <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                          <th className="px-3 py-2 text-left border-b border-[#444]">INSTRUMENT</th>
                          <th className="px-3 py-2 text-center border-b border-[#444]">SIDE</th>
                          <th className="px-3 py-2 text-right border-b border-[#444]">QTY</th>
                          <th className="px-3 py-2 text-right border-b border-[#444]">AVG PRICE</th>
                          <th className="px-3 py-2 text-right border-b border-[#444]">UNREALIZED P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedSession.activePositions.map((pos) => (
                          <tr key={pos.id} className="border-b border-[#222]">
                            <td className="px-3 py-2 text-[#00ff00]">{pos.instrument}</td>
                            <td className={`px-3 py-2 text-center ${pos.side === 'LONG' ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                              {pos.side}
                            </td>
                            <td className="px-3 py-2 text-right text-[#fff]">{pos.quantity.toLocaleString()}</td>
                            <td className="px-3 py-2 text-right text-[#fff]">{pos.avgPrice.toFixed(5)}</td>
                            <td className={`px-3 py-2 text-right ${pos.unrealizedPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                              {formatCurrency(pos.unrealizedPnl)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="p-4 text-[#666] text-[10px] text-center">No active positions</div>
                  )}
                </div>
              </div>

              {/* Session Info */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">SESSION INFO</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Started:</span>
                    <span className="text-[#fff]">{new Date(selectedSession.startedAt || '').toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Initial Capital:</span>
                    <span className="text-[#00ff00]">{formatCurrency(selectedSession.initialCapital)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Return:</span>
                    <span className={selectedSession.totalPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {formatPercentage((selectedSession.totalPnl / selectedSession.initialCapital) * 100)}
                    </span>
                  </div>
                  {selectedSession.description && (
                    <div className="flex justify-between">
                      <span className="text-[#666]">Description:</span>
                      <span className="text-[#fff]">{selectedSession.description}</span>
                    </div>
                  )}
                </div>
              </div>
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

            <div className="space-y-6">
              {/* Status */}
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
                      className="h-full bg-[#ff8c00] transition-all"
                      style={{ width: `${selectedBacktest.progress}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Metrics */}
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
                        <div className="text-[#666] text-[9px]">Sortino Ratio</div>
                        <div className="text-[#00ff00] text-[14px]">{selectedBacktest.metrics.sortinoRatio.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[9px]">Calmar Ratio</div>
                        <div className="text-[#00ff00] text-[14px]">{selectedBacktest.metrics.calmarRatio.toFixed(2)}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Configuration */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">CONFIGURATION</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Initial Capital:</span>
                    <span className="text-[#00ff00]">{formatCurrency(selectedBacktest.config.initialCapital)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Date Range:</span>
                    <span className="text-[#fff]">{selectedBacktest.config.dataRange.startDate} to {selectedBacktest.config.dataRange.endDate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Venue:</span>
                    <span className="text-[#fff]">{selectedBacktest.config.venue}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Slippage Model:</span>
                    <span className="text-[#fff]">{selectedBacktest.config.executionConfig.slippageModel}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Fill Model:</span>
                    <span className="text-[#fff]">{selectedBacktest.config.executionConfig.fillModel}</span>
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

            <div className="space-y-6">
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

      case 'DATA_CATALOG':
        if (!selectedMigration) return (
          <div className="p-4">
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">DATA CATALOG</div>
            </div>
            <div className="text-[#666] text-[10px]">
              Nautilus Trader uses Parquet files for high-performance data storage.
              Migrate your existing data to Nautilus format for backtesting.
            </div>
          </div>
        );
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                MIGRATION: {selectedMigration.sourceType} → NAUTILUS
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">STATUS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="flex items-center gap-4 mb-3">
                    <span className={`text-[12px] ${getStatusColor(selectedMigration.status)}`}>
                      {selectedMigration.status}
                    </span>
                    <span className="text-[#666] text-[10px]">
                      Progress: {selectedMigration.progress}%
                    </span>
                  </div>
                  <div className="h-2 bg-[#222]">
                    <div
                      className="h-full bg-[#ff8c00]"
                      style={{ width: `${selectedMigration.progress}%` }}
                    />
                  </div>
                </div>
              </div>

              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">MIGRATION STATS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Total Records:</span>
                    <span className="text-[#fff]">{selectedMigration.totalRecords.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Migrated:</span>
                    <span className="text-[#00ff00]">{selectedMigration.migratedRecords.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Failed:</span>
                    <span className="text-[#ff0000]">{selectedMigration.failedRecords.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Started:</span>
                    <span className="text-[#fff]">{new Date(selectedMigration.startedAt).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        );

      case 'SYSTEM_STATUS':
        if (!systemStatus) return <div className="text-[#666] p-4">Loading system status...</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">NAUTILUS SYSTEM STATUS</div>
            </div>

            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                {renderServiceStatus('Integration Service', systemStatus.integrationService)}
                {renderServiceStatus('Backtest Engine', systemStatus.backtestEngine)}
                {renderServiceStatus('Trading Node', systemStatus.tradingNode)}
                {renderServiceStatus('Strategy Translation', systemStatus.strategyTranslation)}
                {renderServiceStatus('Signal Router', systemStatus.signalRouter)}
                {renderServiceStatus('Data Catalog', systemStatus.dataCatalog)}
                {renderServiceStatus('Risk Integration', systemStatus.riskIntegration)}
              </div>
            </div>
          </>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          {/* Status Indicator */}
          <div className="mb-4 flex items-center justify-between">
            <div className="text-[#ff8c00] text-[10px] tracking-wider">NAUTILUS TRADER</div>
            <div className={`px-2 py-1 border text-[10px] ${isBackendConnected ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#ff0000] text-[#ff0000]'}`}>
              {isBackendConnected ? '● CONNECTED' : '● OFFLINE'}
            </div>
          </div>

          {/* Category Selection */}
          <div className="mb-4">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">CATEGORY</div>
            <div className="space-y-1">
              {categories.map((cat) => (
                <button
                  key={cat.key}
                  onClick={() => {
                    setSelectedCategory(cat.key);
                    setSelectedSession(null);
                    setSelectedBacktest(null);
                    setSelectedTranslation(null);
                    setSelectedMigration(null);
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

      {/* Center Panel */}
      <div className="flex-1 overflow-y-auto p-4">
        {renderCenterPanelContent()}
      </div>

      {/* Right Panel - Actions */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">NAUTILUS ACTIONS</div>

          {/* Action Status */}
          {actionStatus && (
            <div className="mb-4 p-2 border border-[#ffff00] bg-[#1a1a1a] text-[#ffff00] text-[10px]">
              {actionStatus}
            </div>
          )}

          {/* Category-specific actions */}
          {selectedCategory === 'LIVE_SESSIONS' && selectedSession && (
            <div className="space-y-2 mb-6">
              {selectedSession.status === LiveTradingStatus.ACTIVE && (
                <>
                  <button
                    onClick={() => handlePauseSession(selectedSession.id)}
                    className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors"
                  >
                    PAUSE SESSION
                  </button>
                  <button
                    onClick={() => handleStopSession(selectedSession.id)}
                    className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                  >
                    STOP SESSION
                  </button>
                </>
              )}
              {selectedSession.status === LiveTradingStatus.PAUSED && (
                <>
                  <button
                    onClick={() => handleResumeSession(selectedSession.id)}
                    className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
                  >
                    RESUME SESSION
                  </button>
                  <button
                    onClick={() => handleStopSession(selectedSession.id)}
                    className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                  >
                    STOP SESSION
                  </button>
                </>
              )}
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                VIEW TRADE LOG
              </button>
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                EXPORT REPORT
              </button>
            </div>
          )}

          {selectedCategory === 'BACKTESTS' && selectedBacktest && (
            <div className="space-y-2 mb-6">
              {selectedBacktest.status === BacktestStatus.RUNNING && (
                <>
                  <button
                    onClick={() => handleCancelBacktest(selectedBacktest.id)}
                    className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                  >
                    CANCEL BACKTEST
                  </button>
                </>
              )}
              {selectedBacktest.status === BacktestStatus.COMPLETED && (
                <>
                  <button
                    className="w-full py-2 px-3 bg-[#00ff00] text-black text-[10px] hover:bg-[#00ff00]/80 transition-colors"
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
                </>
              )}
            </div>
          )}

          {selectedCategory === 'TRANSLATIONS' && selectedTranslation && (
            <div className="space-y-2 mb-6">
              {selectedTranslation.status === StrategyTranslationStatus.COMPLETED && (
                <>
                  <button
                    onClick={() => setShowNewBacktestModal(true)}
                    className="w-full py-2 px-3 bg-[#ff8c00] text-black text-[10px] hover:bg-[#ffa500] transition-colors"
                  >
                    CREATE BACKTEST
                  </button>
                  <button
                    onClick={() => setShowNewSessionModal(true)}
                    className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
                  >
                    START LIVE TRADING
                  </button>
                </>
              )}
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

          {/* Quick Actions */}
          <div className="mt-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">QUICK ACTIONS</div>
            <div className="space-y-2">
              <button
                onClick={() => setShowNewSessionModal(true)}
                className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
              >
                NEW LIVE SESSION
              </button>
              <button
                onClick={() => setShowNewBacktestModal(true)}
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                NEW BACKTEST
              </button>
              <button
                onClick={() => setShowDataMigrationModal(true)}
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors"
              >
                MIGRATE DATA
              </button>
              <button
                className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors"
              >
                VIEW SYSTEM LOGS
              </button>
            </div>
          </div>

          {/* Kill Switch */}
          <div className="mt-6">
            <div className="text-[#ff0000] mb-3 text-[10px] tracking-wider">EMERGENCY CONTROLS</div>
            <button
              className="w-full py-3 px-3 border-2 border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
            >
              KILL SWITCH - STOP ALL TRADING
            </button>
          </div>
        </div>
      </div>

      {/* New Live Session Modal */}
      {showNewSessionModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">NEW LIVE TRADING SESSION</h2>
              <button
                onClick={() => setShowNewSessionModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleStartLiveTrading();
              }}
            >
              <div className="space-y-4">
                <div>
                  <label className="text-[#666] block mb-1 text-[9px]">SESSION NAME</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g., Trend Following Live"
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] block mb-1 text-[9px]">INITIAL CAPITAL</label>
                  <input
                    type="number"
                    required
                    defaultValue={100000}
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">MAX DRAWDOWN %</label>
                    <input
                      type="number"
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
                      required
                      defaultValue={3.0}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    />
                  </div>
                </div>
                <div className="border border-[#ffff00] bg-[#1a1a1a] p-3 text-[#ffff00] text-[10px]">
                  Warning: Live trading involves real financial risk. Ensure all risk parameters are configured correctly.
                </div>
              </div>

              <div className="flex gap-2 mt-6">
                <button
                  type="button"
                  onClick={() => setShowNewSessionModal(false)}
                  className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px] disabled:opacity-50"
                >
                  {isLoading ? 'STARTING...' : 'START TRADING'}
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
              <h2 className="text-[#ff8c00] text-lg">NEW NAUTILUS BACKTEST</h2>
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
                <div>
                  <label className="text-[#666] block mb-1 text-[9px]">BACKTEST NAME</label>
                  <input
                    type="text"
                    name="name"
                    required
                    placeholder="e.g., MA Crossover Q1 2024"
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
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
                </div>
                <div className="grid grid-cols-2 gap-4">
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
                </div>
                <div className="grid grid-cols-2 gap-4">
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

      {/* Data Migration Modal */}
      {showDataMigrationModal && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-[#ff8c00] text-lg">DATA MIGRATION</h2>
              <button
                onClick={() => setShowDataMigrationModal(false)}
                className="text-[#ff8c00] hover:text-[#fff]"
              >
                ✕
              </button>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.currentTarget);
                handleStartDataMigration({
                  sourceType: formData.get('sourceType'),
                  sourcePath: formData.get('sourcePath'),
                  targetPath: '/data/nautilus',
                  dataType: formData.get('dataType'),
                  instruments: (formData.get('instruments') as string).split(',').map(i => i.trim()),
                  dateRange: {
                    startDate: formData.get('startDate'),
                    endDate: formData.get('endDate'),
                  },
                });
              }}
            >
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">SOURCE TYPE</label>
                    <select
                      name="sourceType"
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    >
                      <option value="POSTGRESQL">PostgreSQL</option>
                      <option value="CSV">CSV Files</option>
                      <option value="PARQUET">Parquet Files</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">DATA TYPE</label>
                    <select
                      name="dataType"
                      required
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                    >
                      <option value="OHLCV">OHLCV Bars</option>
                      <option value="TICK">Tick Data</option>
                      <option value="ORDER_BOOK">Order Book</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="text-[#666] block mb-1 text-[9px]">SOURCE PATH / CONNECTION</label>
                  <input
                    type="text"
                    name="sourcePath"
                    required
                    placeholder="e.g., postgresql://localhost/trading_data"
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div>
                  <label className="text-[#666] block mb-1 text-[9px]">INSTRUMENTS (comma-separated)</label>
                  <input
                    type="text"
                    name="instruments"
                    required
                    placeholder="e.g., EURUSD, GBPUSD, USDJPY"
                    className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
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
                </div>
              </div>

              <div className="flex gap-2 mt-6">
                <button
                  type="button"
                  onClick={() => setShowDataMigrationModal(false)}
                  className="flex-1 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="flex-1 px-4 py-2 bg-[#ff8c00] text-black hover:bg-[#ffa500] text-[10px] disabled:opacity-50"
                >
                  {isLoading ? 'STARTING...' : 'START MIGRATION'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
