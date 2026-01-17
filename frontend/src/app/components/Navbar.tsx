import React from 'react';
import { 
  AlertTriangle, 
  Activity, 
  Zap, 
  Bell, 
  Search, 
  User, 
  Settings, 
  HelpCircle,
  Square,
  Play,
  RotateCcw,
  TrendingUp,
  BarChart3,
  Eye,
  MessageSquare
} from 'lucide-react';
import { useTradingStore } from '../store/tradingStore';

const Navbar: React.FC = () => {
  const {
    globalStatus,
    notifications,
    emergencyControls,
    showQuickChart,
    showFastOrder,
    showWatchlist,
    showSymbolLookup,
    showNotifications,
    emergencyHalt,
    pauseTrading,
    resumeTrading,
    forceReconnect,
    toggleQuickChart,
    toggleFastOrder,
    toggleWatchlist,
    toggleSymbolLookup,
    toggleNotifications,
  } = useTradingStore();

  const getConnectionStatusColor = () => {
    switch (globalStatus.connectionStatus) {
      case 'LIVE': return 'text-green-400';
      case 'DELAYED': return 'text-yellow-400';
      case 'DISCONNECTED': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getConnectionStatusIcon = () => {
    switch (globalStatus.connectionStatus) {
      case 'LIVE': return '🟢';
      case 'DELAYED': return '🟡';
      case 'DISCONNECTED': return '🔴';
      default: return '⚪';
    }
  };

  const getPnLColor = () => {
    return globalStatus.dailyPnL.amount >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getRiskColor = () => {
    const percentage = globalStatus.riskUtilization.percentage;
    if (percentage >= 80) return 'text-red-400';
    if (percentage >= 60) return 'text-yellow-400';
    return 'text-green-400';
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatNumber = (num: number, decimals: number = 1) => {
    return num.toFixed(decimals);
  };

  const totalNotifications = notifications.systemAlerts + notifications.tradingAlerts + notifications.messages;

  return (
    <nav className="bg-black border-b border-gray-600 px-4 py-2 flex items-center justify-between font-mono text-xs">
      {/* Left Side - Logo & Global Status */}
      <div className="flex items-center space-x-6">
        {/* Logo */}
        <div className="text-orange-400 font-bold text-sm">
          ALGO<span className="text-white">TRADE</span>
        </div>

        {/* Global Status */}
        <div className="flex items-center space-x-4">
          {/* Connection Status */}
          <div className={`flex items-center space-x-1 ${getConnectionStatusColor()}`}>
            <span>{getConnectionStatusIcon()}</span>
            <span className="font-bold">{globalStatus.connectionStatus}</span>
          </div>

          {/* Latency */}
          <div className="text-gray-300">
            <span className="text-gray-500">LAT:</span>
            <span className={globalStatus.latency > 50 ? 'text-yellow-400' : 'text-green-400'}>
              {globalStatus.latency}ms
            </span>
          </div>

          {/* Current Regime */}
          <div className="text-gray-300">
            <span className="text-gray-500">REGIME:</span>
            <span className="text-orange-400">{globalStatus.currentRegime.name}</span>
            <span className="text-gray-500 ml-1">
              ({formatNumber(globalStatus.currentRegime.confidence)}%)
            </span>
          </div>

          {/* Daily P&L */}
          <div className="text-gray-300">
            <span className="text-gray-500">P&L:</span>
            <span className={getPnLColor()}>
              {globalStatus.dailyPnL.amount >= 0 ? '+' : ''}
              {formatCurrency(globalStatus.dailyPnL.amount)}
            </span>
            <span className={`ml-1 ${getPnLColor()}`}>
              ({globalStatus.dailyPnL.percentage >= 0 ? '+' : ''}
              {formatNumber(globalStatus.dailyPnL.percentage, 2)}%)
            </span>
          </div>

          {/* Risk Utilization */}
          <div className="text-gray-300">
            <span className="text-gray-500">RISK:</span>
            <span className={getRiskColor()}>
              {formatNumber(globalStatus.riskUtilization.percentage)}%
            </span>
            <span className="text-gray-500 ml-1">
              ({formatCurrency(globalStatus.riskUtilization.current / 1000000)}M/
              {formatCurrency(globalStatus.riskUtilization.limit / 1000000)}M)
            </span>
          </div>
        </div>
      </div>

      {/* Center - Emergency Controls & Quick Actions */}
      <div className="flex items-center space-x-3">
        {/* Emergency Controls */}
        <div className="flex items-center space-x-2">
          {/* Emergency Halt */}
          <button
            onClick={emergencyHalt}
            disabled={!emergencyControls.canHalt || emergencyControls.systemStatus === 'HALTED'}
            className="px-3 py-1 bg-red-900 border border-red-600 text-red-400 hover:bg-red-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-1"
          >
            <Square className="w-3 h-3" />
            <span>HALT</span>
          </button>

          {/* Pause/Resume Trading */}
          {emergencyControls.systemStatus === 'PAUSED' ? (
            <button
              onClick={resumeTrading}
              className="px-3 py-1 bg-green-900 border border-green-600 text-green-400 hover:bg-green-800 transition-colors flex items-center space-x-1"
            >
              <Play className="w-3 h-3" />
              <span>RESUME</span>
            </button>
          ) : (
            <button
              onClick={pauseTrading}
              disabled={emergencyControls.systemStatus === 'HALTED'}
              className="px-3 py-1 bg-yellow-900 border border-yellow-600 text-yellow-400 hover:bg-yellow-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-1"
            >
              <Square className="w-3 h-3" />
              <span>PAUSE</span>
            </button>
          )}

          {/* Force Reconnect */}
          <button
            onClick={forceReconnect}
            disabled={globalStatus.connectionStatus === 'LIVE'}
            className="px-3 py-1 bg-blue-900 border border-blue-600 text-blue-400 hover:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-1"
          >
            <RotateCcw className="w-3 h-3" />
            <span>RECONNECT</span>
          </button>
        </div>

        {/* Divider */}
        <div className="h-6 w-px bg-gray-600"></div>

        {/* Quick Actions */}
        <div className="flex items-center space-x-2">
          {/* Quick Chart */}
          <button
            onClick={toggleQuickChart}
            className={`p-2 border transition-colors ${
              showQuickChart 
                ? 'bg-orange-900 border-orange-600 text-orange-400' 
                : 'border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400'
            }`}
            title="Quick Chart"
          >
            <TrendingUp className="w-4 h-4" />
          </button>

          {/* Fast Order */}
          <button
            onClick={toggleFastOrder}
            className={`p-2 border transition-colors ${
              showFastOrder 
                ? 'bg-orange-900 border-orange-600 text-orange-400' 
                : 'border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400'
            }`}
            title="Fast Order"
          >
            <Zap className="w-4 h-4" />
          </button>

          {/* Watchlist */}
          <button
            onClick={toggleWatchlist}
            className={`p-2 border transition-colors ${
              showWatchlist 
                ? 'bg-orange-900 border-orange-600 text-orange-400' 
                : 'border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400'
            }`}
            title="Watchlist"
          >
            <Eye className="w-4 h-4" />
          </button>

          {/* Symbol Lookup */}
          <button
            onClick={toggleSymbolLookup}
            className={`p-2 border transition-colors ${
              showSymbolLookup 
                ? 'bg-orange-900 border-orange-600 text-orange-400' 
                : 'border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400'
            }`}
            title="Symbol Lookup"
          >
            <Search className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Right Side - Notifications, Search, User */}
      <div className="flex items-center space-x-4">
        {/* Notifications */}
        <div className="flex items-center space-x-2">
          {/* System Alerts */}
          {notifications.systemAlerts > 0 && (
            <button
              onClick={toggleNotifications}
              className="relative p-2 border border-red-600 text-red-400 hover:bg-red-900 transition-colors"
              title="System Alerts"
            >
              <AlertTriangle className="w-4 h-4" />
              <span className="absolute -top-1 -right-1 bg-red-600 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                {notifications.systemAlerts}
              </span>
            </button>
          )}

          {/* Trading Alerts */}
          {notifications.tradingAlerts > 0 && (
            <button
              onClick={toggleNotifications}
              className="relative p-2 border border-yellow-600 text-yellow-400 hover:bg-yellow-900 transition-colors"
              title="Trading Alerts"
            >
              <Activity className="w-4 h-4" />
              <span className="absolute -top-1 -right-1 bg-yellow-600 text-black text-xs rounded-full w-4 h-4 flex items-center justify-center">
                {notifications.tradingAlerts}
              </span>
            </button>
          )}

          {/* Messages */}
          {notifications.messages > 0 && (
            <button
              onClick={toggleNotifications}
              className="relative p-2 border border-blue-600 text-blue-400 hover:bg-blue-900 transition-colors"
              title="Messages"
            >
              <MessageSquare className="w-4 h-4" />
              <span className="absolute -top-1 -right-1 bg-blue-600 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                {notifications.messages}
              </span>
            </button>
          )}

          {/* All Notifications (if any) */}
          {totalNotifications === 0 && (
            <button
              onClick={toggleNotifications}
              className="p-2 border border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400 transition-colors"
              title="Notifications"
            >
              <Bell className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Global Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search anything..."
            className="bg-gray-900 border border-gray-600 text-white px-3 py-1 pr-8 w-48 focus:outline-none focus:border-orange-400 transition-colors"
          />
          <Search className="absolute right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        </div>

        {/* User Controls */}
        <div className="flex items-center space-x-2">
          {/* Settings */}
          <button
            className="p-2 border border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400 transition-colors"
            title="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>

          {/* Help */}
          <button
            className="p-2 border border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400 transition-colors"
            title="Help"
          >
            <HelpCircle className="w-4 h-4" />
          </button>

          {/* User Menu */}
          <button
            className="p-2 border border-gray-600 text-gray-400 hover:bg-gray-900 hover:text-orange-400 transition-colors"
            title="User Menu"
          >
            <User className="w-4 h-4" />
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;