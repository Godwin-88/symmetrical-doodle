import { useTradingStore } from '@/app/store/tradingStore';
import { useState, useEffect } from 'react';
import {
  getNautilusSystemStatus,
  type NautilusSystemStatus,
} from '../../services/nautilusService';

// Types
type DashboardCategory = 'OVERVIEW' | 'POSITIONS' | 'STRATEGIES' | 'ALERTS' | 'ACTIVITY';

interface Alert {
  id: string;
  type: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  source: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

interface ActivityLog {
  id: string;
  type: 'TRADE' | 'SIGNAL' | 'REGIME' | 'RISK' | 'SYSTEM';
  message: string;
  timestamp: string;
  details?: string;
}

export function Dashboard() {
  const {
    systemStatus,
    netPnl,
    dailyPnlPercent,
    riskUtilization,
    maxRisk,
    activeStrategies,
    totalStrategies,
    positions,
    strategies,
    currentRegime,
  } = useTradingStore();

  // Category selection state
  const [selectedCategory, setSelectedCategory] = useState<DashboardCategory>('OVERVIEW');
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);

  // Data state
  const [nautilusStatus, setNautilusStatus] = useState<NautilusSystemStatus | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState(true);

  // Mock alerts
  const [alerts] = useState<Alert[]>([
    { id: '1', type: 'WARNING', source: 'RISK ENGINE', message: 'Position size approaching limit on EURUSD', timestamp: new Date(Date.now() - 300000).toISOString(), acknowledged: false },
    { id: '2', type: 'INFO', source: 'INTELLIGENCE', message: 'Regime change detected: LOW_VOL_TRENDING', timestamp: new Date(Date.now() - 600000).toISOString(), acknowledged: true },
    { id: '3', type: 'INFO', source: 'EXECUTION', message: 'Order filled: EURUSD BUY 50000 @ 1.0845', timestamp: new Date(Date.now() - 900000).toISOString(), acknowledged: true },
    { id: '4', type: 'ERROR', source: 'MARKET DATA', message: 'Brief disconnection detected - reconnected', timestamp: new Date(Date.now() - 1200000).toISOString(), acknowledged: true },
  ]);

  // Mock activity logs
  const [activityLogs] = useState<ActivityLog[]>([
    { id: '1', type: 'TRADE', message: 'EURUSD BUY 50000 @ 1.0845', timestamp: new Date(Date.now() - 300000).toISOString(), details: 'Strategy: MOMENTUM ALPHA' },
    { id: '2', type: 'SIGNAL', message: 'Long signal generated for GBPUSD', timestamp: new Date(Date.now() - 600000).toISOString(), details: 'Confidence: 0.85' },
    { id: '3', type: 'REGIME', message: 'Regime change: HIGH_VOL_RANGING → LOW_VOL_TRENDING', timestamp: new Date(Date.now() - 900000).toISOString() },
    { id: '4', type: 'RISK', message: 'Risk limit updated: Max position size 25%', timestamp: new Date(Date.now() - 1200000).toISOString() },
    { id: '5', type: 'SYSTEM', message: 'Strategy rebalance completed', timestamp: new Date(Date.now() - 1500000).toISOString() },
    { id: '6', type: 'TRADE', message: 'USDJPY SELL 30000 @ 148.25', timestamp: new Date(Date.now() - 1800000).toISOString(), details: 'Strategy: MEAN REVERSION' },
  ]);

  // Fetch Nautilus status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await getNautilusSystemStatus();
        setNautilusStatus(status);
        setIsBackendConnected(true);
      } catch (err) {
        console.warn('Failed to fetch Nautilus status:', err);
        setIsBackendConnected(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPnl = (pnl: number) => {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}${formatNumber(pnl)}`;
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  // Get unacknowledged alerts count
  const unacknowledgedAlerts = alerts.filter(a => !a.acknowledged).length;

  // Categories configuration
  const categories: { key: DashboardCategory; label: string; count?: number }[] = [
    { key: 'OVERVIEW', label: 'OVERVIEW' },
    { key: 'POSITIONS', label: 'POSITIONS', count: positions.length },
    { key: 'STRATEGIES', label: 'STRATEGIES', count: strategies.length },
    { key: 'ALERTS', label: 'ALERTS', count: unacknowledgedAlerts },
    { key: 'ACTIVITY', label: 'ACTIVITY', count: activityLogs.length },
  ];

  // Render left panel content
  const renderLeftPanelContent = () => {
    switch (selectedCategory) {
      case 'OVERVIEW':
        return (
          <div className="space-y-4">
            {/* Key Metrics Summary */}
            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">SYSTEM STATUS</div>
              <div className={`text-[14px] ${systemStatus === 'OPERATIONAL' ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                {systemStatus}
              </div>
            </div>
            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">NET P&L</div>
              <div className={`text-[14px] ${netPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                {formatPnl(netPnl)} USD
              </div>
              <div className={`text-[10px] mt-1 ${dailyPnlPercent >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                {formatPnl(dailyPnlPercent)}% today
              </div>
            </div>
            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">RISK UTILIZATION</div>
              <div className="text-[#00ff00] text-[14px]">{formatNumber(riskUtilization, 1)}%</div>
              <div className="h-2 bg-[#222] mt-2">
                <div
                  className={`h-full ${riskUtilization > 80 ? 'bg-[#ff0000]' : riskUtilization > 60 ? 'bg-[#ffff00]' : 'bg-[#00ff00]'}`}
                  style={{ width: `${riskUtilization}%` }}
                />
              </div>
            </div>
            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] text-[10px] mb-2">ACTIVE STRATEGIES</div>
              <div className="text-[#00ff00] text-[14px]">{activeStrategies} / {totalStrategies}</div>
            </div>
          </div>
        );

      case 'POSITIONS':
        return (
          <div className="space-y-2">
            {positions.map((pos, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedPosition(pos.symbol)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedPosition === pos.symbol
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{pos.symbol}</div>
                  <span className={`text-[10px] ${pos.pnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {formatPnl(pos.pnl)}
                  </span>
                </div>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">SIZE:</span>
                    <span className="text-[#fff]">{formatNumber(pos.size, 0)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">EXPOSURE:</span>
                    <span className="text-[#fff]">{formatNumber(pos.exposure, 1)}%</span>
                  </div>
                </div>
              </div>
            ))}
            {positions.length === 0 && (
              <div className="text-[#666] text-[10px] text-center py-4">No open positions</div>
            )}
          </div>
        );

      case 'STRATEGIES':
        return (
          <div className="space-y-2">
            {strategies.map((strat, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedStrategy(strat.name)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedStrategy === strat.name
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                `}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-[#00ff00]">{strat.name}</div>
                  <span className={`text-[10px] px-2 py-0.5 border ${
                    strat.status === 'ACTIVE' ? 'border-[#00ff00] text-[#00ff00]' :
                    strat.status === 'PAUSED' ? 'border-[#ffff00] text-[#ffff00]' :
                    'border-[#ff0000] text-[#ff0000]'
                  }`}>
                    {strat.status}
                  </span>
                </div>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">ALLOC:</span>
                    <span className="text-[#fff]">{strat.allocation}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">P&L:</span>
                    <span className={strat.pnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {formatPnl(strat.pnl)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">SHARPE:</span>
                    <span className="text-[#fff]">{formatNumber(strat.sharpe)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        );

      case 'ALERTS':
        return (
          <div className="space-y-2">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                onClick={() => setSelectedAlert(alert.id)}
                className={`
                  border p-3 cursor-pointer transition-colors
                  ${selectedAlert === alert.id
                    ? 'border-[#ff8c00] bg-[#1a1a1a]'
                    : 'border-[#333] hover:border-[#ff8c00]'
                  }
                  ${!alert.acknowledged ? 'border-l-4 border-l-[#ffff00]' : ''}
                `}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className={`text-[10px] px-1 ${
                    alert.type === 'INFO' ? 'text-[#00ff00] bg-[#00ff00]/10' :
                    alert.type === 'WARNING' ? 'text-[#ffff00] bg-[#ffff00]/10' :
                    alert.type === 'ERROR' ? 'text-[#ff0000] bg-[#ff0000]/10' :
                    'text-[#ff0000] bg-[#ff0000]/20'
                  }`}>
                    {alert.type}
                  </span>
                  <span className="text-[#666] text-[9px]">{formatTime(alert.timestamp)}</span>
                </div>
                <div className="text-[#fff] text-[10px]">{alert.message}</div>
                <div className="text-[#666] text-[9px] mt-1">{alert.source}</div>
              </div>
            ))}
          </div>
        );

      case 'ACTIVITY':
        return (
          <div className="space-y-2">
            {activityLogs.map((log) => (
              <div
                key={log.id}
                className="border border-[#333] p-3 hover:border-[#ff8c00] cursor-pointer"
              >
                <div className="flex justify-between items-start mb-1">
                  <span className={`text-[9px] px-1 ${
                    log.type === 'TRADE' ? 'text-[#00ff00] bg-[#00ff00]/10' :
                    log.type === 'SIGNAL' ? 'text-[#ff8c00] bg-[#ff8c00]/10' :
                    log.type === 'REGIME' ? 'text-[#ffff00] bg-[#ffff00]/10' :
                    log.type === 'RISK' ? 'text-[#ff0000] bg-[#ff0000]/10' :
                    'text-[#666] bg-[#666]/10'
                  }`}>
                    {log.type}
                  </span>
                  <span className="text-[#666] text-[9px]">{formatTime(log.timestamp)}</span>
                </div>
                <div className="text-[#fff] text-[10px]">{log.message}</div>
                {log.details && (
                  <div className="text-[#666] text-[9px] mt-1">{log.details}</div>
                )}
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
      case 'OVERVIEW':
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                TRADING DASHBOARD - OVERVIEW
              </div>
            </div>

            <div className="space-y-6">
              {/* Key Metrics Grid */}
              <div className="grid grid-cols-4 gap-4">
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">SYSTEM STATUS</div>
                  <div className={`text-[14px] ${systemStatus === 'OPERATIONAL' ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {systemStatus}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">NET P&L (USD)</div>
                  <div className={`text-[14px] ${netPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {formatPnl(netPnl)}
                  </div>
                  <div className={`text-[10px] ${dailyPnlPercent >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {formatPnl(dailyPnlPercent)}% today
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">RISK UTILIZATION</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(riskUtilization, 1)}%</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">ACTIVE STRATEGIES</div>
                  <div className="text-[#00ff00] text-[14px]">{activeStrategies} / {totalStrategies}</div>
                </div>
              </div>

              {/* Current Positions Summary */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">CURRENT POSITIONS</div>
                <div className="border border-[#444] bg-[#0a0a0a]">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-[#000] text-[#ff8c00] text-[10px]">
                        <th className="px-3 py-2 text-left border-b border-[#444]">SYMBOL</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">SIZE</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">P&L</th>
                        <th className="px-3 py-2 text-right border-b border-[#444]">EXPOSURE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((pos, idx) => (
                        <tr key={idx} className="border-b border-[#222]">
                          <td className="px-3 py-2 text-[#00ff00]">{pos.symbol}</td>
                          <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(pos.size, 0)}</td>
                          <td className={`px-3 py-2 text-right ${pos.pnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                            {formatPnl(pos.pnl)}
                          </td>
                          <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(pos.exposure, 1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* System Health and Market Regime */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">SYSTEM HEALTH</div>
                  <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                    <div className="flex justify-between">
                      <span className="text-[#666]">EXECUTION CORE</span>
                      <span className="text-[#00ff00]">HEALTHY</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">INTELLIGENCE LAYER</span>
                      <span className="text-[#00ff00]">HEALTHY</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">MARKET DATA</span>
                      <span className="text-[#00ff00]">CONNECTED</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">RISK ENGINE</span>
                      <span className="text-[#00ff00]">ACTIVE</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#666]">NAUTILUS</span>
                      <span className={isBackendConnected ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                        {isBackendConnected ? 'CONNECTED' : 'OFFLINE'}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">MARKET REGIME</div>
                  <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                    {currentRegime && (
                      <div className="space-y-2 text-[10px]">
                        <div className="flex justify-between">
                          <span className="text-[#666]">CURRENT</span>
                          <span className="text-[#00ff00]">{currentRegime.name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#666]">CONFIDENCE</span>
                          <span className="text-[#00ff00]">{currentRegime.probability}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#666]">DURATION</span>
                          <span className="text-[#fff]">{currentRegime.duration}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#666]">VOLATILITY</span>
                          <span className="text-[#fff]">{currentRegime.volatility}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#666]">TREND</span>
                          <span className="text-[#fff]">{currentRegime.trend}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">RECENT ACTIVITY</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-1 text-[10px]">
                  {activityLogs.slice(0, 5).map((log) => (
                    <div key={log.id} className="flex items-center gap-2">
                      <span className="text-[#666]">{formatTime(log.timestamp)}</span>
                      <span className={`text-[9px] px-1 ${
                        log.type === 'TRADE' ? 'text-[#00ff00]' :
                        log.type === 'SIGNAL' ? 'text-[#ff8c00]' :
                        'text-[#666]'
                      }`}>
                        {log.type}
                      </span>
                      <span className="text-[#fff]">{log.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        );

      case 'POSITIONS':
        const currentPosition = positions.find(p => p.symbol === selectedPosition) || positions[0];
        if (!currentPosition) return <div className="text-[#666] p-4">No positions to display</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                POSITION: {currentPosition.symbol}
              </div>
            </div>

            <div className="space-y-6">
              <div className="grid grid-cols-4 gap-4">
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">SIZE</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(currentPosition.size, 0)}</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">P&L</div>
                  <div className={`text-[14px] ${currentPosition.pnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {formatPnl(currentPosition.pnl)}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">EXPOSURE</div>
                  <div className="text-[#ffff00] text-[14px]">{formatNumber(currentPosition.exposure, 1)}%</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">ENTRY PRICE</div>
                  <div className="text-[#fff] text-[14px]">1.0845</div>
                </div>
              </div>

              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">POSITION DETAILS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Direction:</span>
                    <span className="text-[#00ff00]">LONG</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Strategy:</span>
                    <span className="text-[#fff]">MOMENTUM ALPHA</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Entry Time:</span>
                    <span className="text-[#fff]">2024-01-15 10:30:00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Stop Loss:</span>
                    <span className="text-[#ff0000]">1.0800</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Take Profit:</span>
                    <span className="text-[#00ff00]">1.0950</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        );

      case 'STRATEGIES':
        const currentStrategy = strategies.find(s => s.name === selectedStrategy) || strategies[0];
        if (!currentStrategy) return <div className="text-[#666] p-4">No strategies to display</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">
                STRATEGY: {currentStrategy.name}
              </div>
            </div>

            <div className="space-y-6">
              <div className="grid grid-cols-4 gap-4">
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">STATUS</div>
                  <div className={`text-[14px] ${
                    currentStrategy.status === 'ACTIVE' ? 'text-[#00ff00]' :
                    currentStrategy.status === 'PAUSED' ? 'text-[#ffff00]' :
                    'text-[#ff0000]'
                  }`}>
                    {currentStrategy.status}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">P&L</div>
                  <div className={`text-[14px] ${currentStrategy.pnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                    {formatPnl(currentStrategy.pnl)}
                  </div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">ALLOCATION</div>
                  <div className="text-[#fff] text-[14px]">{currentStrategy.allocation}%</div>
                </div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="text-[#666] text-[9px]">SHARPE</div>
                  <div className="text-[#00ff00] text-[14px]">{formatNumber(currentStrategy.sharpe)}</div>
                </div>
              </div>

              <div>
                <div className="text-[#ff8c00] mb-2 text-[10px] tracking-wider">STRATEGY DETAILS</div>
                <div className="border border-[#444] p-3 bg-[#0a0a0a] space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">Family:</span>
                    <span className="text-[#fff]">TREND FOLLOWING</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Horizon:</span>
                    <span className="text-[#fff]">DAILY</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Max Leverage:</span>
                    <span className="text-[#fff]">2.0x</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">Stop Loss:</span>
                    <span className="text-[#ff0000]">3.0%</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        );

      case 'ALERTS':
        const currentAlert = alerts.find(a => a.id === selectedAlert) || alerts[0];
        if (!currentAlert) return <div className="text-[#666] p-4">No alerts to display</div>;
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">ALERT DETAILS</div>
            </div>

            <div className="space-y-6">
              <div className="border border-[#444] p-4 bg-[#0a0a0a]">
                <div className="flex items-center gap-2 mb-4">
                  <span className={`text-[12px] px-2 py-1 ${
                    currentAlert.type === 'INFO' ? 'text-[#00ff00] bg-[#00ff00]/10 border border-[#00ff00]' :
                    currentAlert.type === 'WARNING' ? 'text-[#ffff00] bg-[#ffff00]/10 border border-[#ffff00]' :
                    currentAlert.type === 'ERROR' ? 'text-[#ff0000] bg-[#ff0000]/10 border border-[#ff0000]' :
                    'text-[#ff0000] bg-[#ff0000]/20 border border-[#ff0000]'
                  }`}>
                    {currentAlert.type}
                  </span>
                  <span className={`text-[10px] ${currentAlert.acknowledged ? 'text-[#666]' : 'text-[#ffff00]'}`}>
                    {currentAlert.acknowledged ? 'ACKNOWLEDGED' : 'UNACKNOWLEDGED'}
                  </span>
                </div>
                <div className="text-[#fff] text-[12px] mb-2">{currentAlert.message}</div>
                <div className="text-[#666] text-[10px] mb-4">Source: {currentAlert.source}</div>
                <div className="text-[#666] text-[10px]">
                  Time: {new Date(currentAlert.timestamp).toLocaleString()}
                </div>
              </div>

              {!currentAlert.acknowledged && (
                <button className="w-full py-2 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black">
                  ACKNOWLEDGE ALERT
                </button>
              )}
            </div>
          </>
        );

      case 'ACTIVITY':
        return (
          <>
            <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
              <div className="text-[#ff8c00] text-sm tracking-wider">ACTIVITY LOG</div>
            </div>

            <div className="space-y-2">
              {activityLogs.map((log) => (
                <div key={log.id} className="border border-[#444] p-3 bg-[#0a0a0a]">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-[10px] px-1 ${
                      log.type === 'TRADE' ? 'text-[#00ff00] bg-[#00ff00]/10' :
                      log.type === 'SIGNAL' ? 'text-[#ff8c00] bg-[#ff8c00]/10' :
                      log.type === 'REGIME' ? 'text-[#ffff00] bg-[#ffff00]/10' :
                      log.type === 'RISK' ? 'text-[#ff0000] bg-[#ff0000]/10' :
                      'text-[#666] bg-[#666]/10'
                    }`}>
                      {log.type}
                    </span>
                    <span className="text-[#666] text-[9px]">{new Date(log.timestamp).toLocaleString()}</span>
                  </div>
                  <div className="text-[#fff] text-[10px]">{log.message}</div>
                  {log.details && (
                    <div className="text-[#666] text-[9px] mt-1">{log.details}</div>
                  )}
                </div>
              ))}
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
            <div className="text-[#ff8c00] text-[10px] tracking-wider">DASHBOARD</div>
            <div className={`px-2 py-1 border text-[10px] ${systemStatus === 'OPERATIONAL' ? 'border-[#00ff00] text-[#00ff00]' : 'border-[#ff0000] text-[#ff0000]'}`}>
              {systemStatus === 'OPERATIONAL' ? '● LIVE' : '● OFFLINE'}
            </div>
          </div>

          {/* Category Selection */}
          <div className="mb-4">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">VIEW</div>
            <div className="space-y-1">
              {categories.map((cat) => (
                <button
                  key={cat.key}
                  onClick={() => {
                    setSelectedCategory(cat.key);
                    setSelectedPosition(null);
                    setSelectedStrategy(null);
                    setSelectedAlert(null);
                  }}
                  className={`
                    w-full py-2 px-3 text-left text-[10px] border transition-colors flex justify-between items-center
                    ${selectedCategory === cat.key
                      ? 'border-[#ff8c00] bg-[#1a1a1a] text-[#ff8c00]'
                      : 'border-[#333] text-[#666] hover:border-[#ff8c00] hover:text-[#ff8c00]'
                    }
                  `}
                >
                  <span>{cat.label}</span>
                  {cat.count !== undefined && (
                    <span className={`px-1 ${cat.key === 'ALERTS' && cat.count > 0 ? 'bg-[#ff0000] text-black' : ''}`}>
                      {cat.count}
                    </span>
                  )}
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
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">QUICK ACTIONS</div>

          {/* Trading Actions */}
          <div className="space-y-2 mb-6">
            <button className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors">
              NEW ORDER
            </button>
            <button className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors">
              CLOSE ALL POSITIONS
            </button>
            <button className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors">
              EMERGENCY HALT
            </button>
          </div>

          <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">NAVIGATION</div>
          <div className="space-y-2 mb-6">
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              GO TO MARKETS (F2)
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              GO TO STRATEGIES (F4)
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              GO TO PORTFOLIO (F5)
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#00ff00] hover:border-[#ff8c00] hover:bg-[#1a1a1a] transition-colors">
              GO TO NAUTILUS (F9)
            </button>
          </div>

          <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">SYSTEM</div>
          <div className="space-y-2">
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors">
              REFRESH DATA
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors">
              VIEW SYSTEM LOGS
            </button>
            <button className="w-full py-2 px-3 border border-[#444] text-left text-[10px] text-[#666] hover:border-[#666] hover:text-[#fff] transition-colors">
              SETTINGS
            </button>
          </div>

          {/* Alerts Summary */}
          {unacknowledgedAlerts > 0 && (
            <div className="mt-6 p-3 border border-[#ffff00] bg-[#ffff00]/10">
              <div className="text-[#ffff00] text-[10px]">
                {unacknowledgedAlerts} UNACKNOWLEDGED ALERT{unacknowledgedAlerts > 1 ? 'S' : ''}
              </div>
              <button
                onClick={() => setSelectedCategory('ALERTS')}
                className="mt-2 text-[#ffff00] text-[9px] underline"
              >
                View Alerts →
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
