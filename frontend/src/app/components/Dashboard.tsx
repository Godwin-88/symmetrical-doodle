import { useTradingStore } from '@/app/store/tradingStore';

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

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatPnl = (pnl: number) => {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}${formatNumber(pnl)}`;
  };

  return (
    <div className="h-full flex flex-col font-mono text-xs overflow-hidden">
      {/* Header */}
      <div className="border-t-2 border-b-2 border-[#ff8c00] py-2">
        <div className="text-center text-[#ff8c00] text-sm tracking-wide">
          TRADING DASHBOARD - REAL TIME MARKET DATA
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-4 gap-4">
            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] mb-2">SYSTEM STATUS</div>
              <div className={`text-sm ${systemStatus === 'OPERATIONAL' ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                {systemStatus}
              </div>
              <div className="text-[#666] text-[10px] mt-1">ALL NOMINAL</div>
            </div>

            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] mb-2">NET P&L (USD)</div>
              <div className={`text-sm ${netPnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                {formatPnl(netPnl)}
              </div>
              <div className={`text-[10px] mt-1 ${dailyPnlPercent >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                {formatPnl(dailyPnlPercent)}% TODAY
              </div>
            </div>

            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] mb-2">RISK UTIL</div>
              <div className="text-sm text-[#00ff00]">{formatNumber(riskUtilization, 1)}%</div>
              <div className="text-[#666] text-[10px] mt-1">
                ${formatNumber(riskUtilization * 50000 / 100, 0)} / ${formatNumber(maxRisk * 50000 / 100, 0)}
              </div>
            </div>

            <div className="border border-[#444] p-3">
              <div className="text-[#ff8c00] mb-2">ACTIVE STRAT</div>
              <div className="text-sm text-[#00ff00]">{activeStrategies} / {totalStrategies}</div>
              <div className="text-[#ffff00] text-[10px] mt-1">1 PAUSED</div>
            </div>
          </div>

          {/* Current Positions */}
          <div className="border border-[#444]">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
          <div className="text-[#ff8c00]">CURRENT POSITIONS</div>
        </div>
        <table className="w-full">
          <thead>
            <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
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

      {/* Strategy Performance */}
      <div className="border border-[#444]">
        <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
          <div className="text-[#ff8c00]">STRATEGY PERFORMANCE</div>
        </div>
        <table className="w-full">
          <thead>
            <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
              <th className="px-3 py-2 text-left border-b border-[#444]">STRATEGY</th>
              <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">ALLOC</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">P&L</th>
              <th className="px-3 py-2 text-right border-b border-[#444]">SHARPE</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map((strat, idx) => (
              <tr key={idx} className="border-b border-[#222]">
                <td className="px-3 py-2 text-[#00ff00]">{strat.name}</td>
                <td className="px-3 py-2 text-center">
                  <span className={`
                    ${strat.status === 'ACTIVE' ? 'text-[#00ff00]' : ''}
                    ${strat.status === 'PAUSED' ? 'text-[#ffff00]' : ''}
                    ${strat.status === 'STOPPED' ? 'text-[#ff0000]' : ''}
                  `}>
                    {strat.status}
                  </span>
                </td>
                <td className="px-3 py-2 text-right text-[#fff]">{strat.allocation}%</td>
                <td className={`px-3 py-2 text-right ${strat.pnl >= 0 ? 'text-[#00ff00]' : 'text-[#ff0000]'}`}>
                  {formatPnl(strat.pnl)}
                </td>
                <td className="px-3 py-2 text-right text-[#fff]">{formatNumber(strat.sharpe)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* System Health and Market Regime */}
      <div className="grid grid-cols-2 gap-4">
        <div className="border border-[#444] p-3">
          <div className="text-[#ff8c00] mb-3">SYSTEM HEALTH</div>
          <div className="space-y-2 text-[10px]">
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
              <span className="text-[#666]">DATABASE</span>
              <span className="text-[#00ff00]">HEALTHY</span>
            </div>
          </div>
        </div>

        <div className="border border-[#444] p-3">
          <div className="text-[#ff8c00] mb-3">MARKET REGIME</div>
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

      {/* Recent Activity */}
      <div className="border border-[#444] p-3">
        <div className="text-[#ff8c00] mb-3">RECENT ACTIVITY</div>
        <div className="space-y-1 text-[10px]">
          <div className="text-[#666]">10:30 STRATEGY REBALANCE COMPLETED</div>
          <div className="text-[#666]">09:15 MARKET REGIME CHANGE DETECTED</div>
          <div className="text-[#666]">08:45 RISK LIMIT ADJUSTMENT</div>
          <div className="text-[#666]">08:30 POSITION EUR/USD OPENED</div>
        </div>
      </div>
        </div>
      </div>
    </div>
  );
}
