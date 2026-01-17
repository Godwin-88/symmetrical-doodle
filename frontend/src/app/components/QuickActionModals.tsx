import React, { useState } from 'react';
import { X, TrendingUp, Zap, Eye, Search, Bell } from 'lucide-react';
import { useTradingStore } from '../store/tradingStore';

// Quick Chart Modal
export const QuickChartModal: React.FC = () => {
  const { showQuickChart, toggleQuickChart } = useTradingStore();
  const [symbol, setSymbol] = useState('EURUSD');
  const [timeframe, setTimeframe] = useState('1H');

  if (!showQuickChart) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-black border border-gray-600 w-96 max-w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-600">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-orange-400" />
            <h3 className="text-orange-400 font-mono text-sm font-bold">QUICK CHART</h3>
          </div>
          <button
            onClick={toggleQuickChart}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 text-xs font-mono mb-1">SYMBOL</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
                placeholder="EURUSD"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-xs font-mono mb-1">TIMEFRAME</label>
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className="w-full bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
              >
                <option value="1M">1M</option>
                <option value="5M">5M</option>
                <option value="15M">15M</option>
                <option value="1H">1H</option>
                <option value="4H">4H</option>
                <option value="1D">1D</option>
              </select>
            </div>
          </div>

          {/* Mock Chart Area */}
          <div className="h-48 bg-gray-900 border border-gray-600 flex items-center justify-center">
            <div className="text-gray-500 font-mono text-sm">
              Chart for {symbol} ({timeframe})
              <br />
              <span className="text-xs">Integration with charting library needed</span>
            </div>
          </div>

          <div className="flex space-x-2">
            <button
              onClick={() => {
                console.log(`Opening chart for ${symbol} ${timeframe}`);
                toggleQuickChart();
              }}
              className="flex-1 bg-orange-900 border border-orange-600 text-orange-400 py-2 px-4 hover:bg-orange-800 transition-colors font-mono text-sm"
            >
              OPEN CHART
            </button>
            <button
              onClick={toggleQuickChart}
              className="px-4 py-2 border border-gray-600 text-gray-400 hover:bg-gray-900 transition-colors font-mono text-sm"
            >
              CANCEL
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Fast Order Modal
export const FastOrderModal: React.FC = () => {
  const { showFastOrder, toggleFastOrder } = useTradingStore();
  const [symbol, setSymbol] = useState('EURUSD');
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [size, setSize] = useState('100000');
  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT'>('MARKET');
  const [price, setPrice] = useState('');

  if (!showFastOrder) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-black border border-gray-600 w-96 max-w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-600">
          <div className="flex items-center space-x-2">
            <Zap className="w-5 h-5 text-orange-400" />
            <h3 className="text-orange-400 font-mono text-sm font-bold">FAST ORDER</h3>
          </div>
          <button
            onClick={toggleFastOrder}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 text-xs font-mono mb-1">SYMBOL</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-xs font-mono mb-1">SIDE</label>
              <div className="flex space-x-1">
                <button
                  onClick={() => setSide('BUY')}
                  className={`flex-1 py-2 px-3 font-mono text-sm transition-colors ${
                    side === 'BUY'
                      ? 'bg-green-900 border border-green-600 text-green-400'
                      : 'border border-gray-600 text-gray-400 hover:bg-gray-900'
                  }`}
                >
                  BUY
                </button>
                <button
                  onClick={() => setSide('SELL')}
                  className={`flex-1 py-2 px-3 font-mono text-sm transition-colors ${
                    side === 'SELL'
                      ? 'bg-red-900 border border-red-600 text-red-400'
                      : 'border border-gray-600 text-gray-400 hover:bg-gray-900'
                  }`}
                >
                  SELL
                </button>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 text-xs font-mono mb-1">SIZE</label>
              <input
                type="text"
                value={size}
                onChange={(e) => setSize(e.target.value)}
                className="w-full bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-xs font-mono mb-1">TYPE</label>
              <select
                value={orderType}
                onChange={(e) => setOrderType(e.target.value as 'MARKET' | 'LIMIT')}
                className="w-full bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
              >
                <option value="MARKET">MARKET</option>
                <option value="LIMIT">LIMIT</option>
              </select>
            </div>
          </div>

          {orderType === 'LIMIT' && (
            <div>
              <label className="block text-gray-400 text-xs font-mono mb-1">PRICE</label>
              <input
                type="text"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                className="w-full bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
                placeholder="1.0850"
              />
            </div>
          )}

          <div className="flex space-x-2">
            <button
              onClick={() => {
                console.log(`Placing ${side} order: ${size} ${symbol} ${orderType}${orderType === 'LIMIT' ? ` @ ${price}` : ''}`);
                toggleFastOrder();
              }}
              className={`flex-1 py-2 px-4 font-mono text-sm transition-colors ${
                side === 'BUY'
                  ? 'bg-green-900 border border-green-600 text-green-400 hover:bg-green-800'
                  : 'bg-red-900 border border-red-600 text-red-400 hover:bg-red-800'
              }`}
            >
              {side} {symbol}
            </button>
            <button
              onClick={toggleFastOrder}
              className="px-4 py-2 border border-gray-600 text-gray-400 hover:bg-gray-900 transition-colors font-mono text-sm"
            >
              CANCEL
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Watchlist Modal
export const WatchlistModal: React.FC = () => {
  const { showWatchlist, toggleWatchlist } = useTradingStore();

  const watchlistItems = [
    { symbol: 'EURUSD', price: '1.0845', change: '+0.0012', changePercent: '+0.11%' },
    { symbol: 'GBPUSD', price: '1.2634', change: '-0.0023', changePercent: '-0.18%' },
    { symbol: 'USDJPY', price: '149.85', change: '+0.45', changePercent: '+0.30%' },
    { symbol: 'AUDUSD', price: '0.6523', change: '+0.0008', changePercent: '+0.12%' },
    { symbol: 'USDCHF', price: '0.8756', change: '-0.0015', changePercent: '-0.17%' },
  ];

  if (!showWatchlist) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-black border border-gray-600 w-96 max-w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-600">
          <div className="flex items-center space-x-2">
            <Eye className="w-5 h-5 text-orange-400" />
            <h3 className="text-orange-400 font-mono text-sm font-bold">WATCHLIST</h3>
          </div>
          <button
            onClick={toggleWatchlist}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          <div className="space-y-2">
            {watchlistItems.map((item) => (
              <div
                key={item.symbol}
                className="flex items-center justify-between p-2 border border-gray-700 hover:bg-gray-900 transition-colors cursor-pointer"
              >
                <div className="font-mono text-sm text-white">{item.symbol}</div>
                <div className="text-right">
                  <div className="font-mono text-sm text-white">{item.price}</div>
                  <div className={`font-mono text-xs ${
                    item.change.startsWith('+') ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {item.change} ({item.changePercent})
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 flex space-x-2">
            <input
              type="text"
              placeholder="Add symbol..."
              className="flex-1 bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
            />
            <button className="px-4 py-2 bg-orange-900 border border-orange-600 text-orange-400 hover:bg-orange-800 transition-colors font-mono text-sm">
              ADD
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Symbol Lookup Modal
export const SymbolLookupModal: React.FC = () => {
  const { showSymbolLookup, toggleSymbolLookup } = useTradingStore();
  const [searchQuery, setSearchQuery] = useState('');

  const searchResults = [
    { symbol: 'EURUSD', name: 'Euro / US Dollar', type: 'Forex' },
    { symbol: 'AAPL', name: 'Apple Inc.', type: 'Stock' },
    { symbol: 'BTC-USD', name: 'Bitcoin USD', type: 'Crypto' },
    { symbol: 'GC=F', name: 'Gold Futures', type: 'Commodity' },
    { symbol: '^GSPC', name: 'S&P 500', type: 'Index' },
  ].filter(item => 
    searchQuery === '' || 
    item.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (!showSymbolLookup) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-black border border-gray-600 w-96 max-w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-600">
          <div className="flex items-center space-x-2">
            <Search className="w-5 h-5 text-orange-400" />
            <h3 className="text-orange-400 font-mono text-sm font-bold">SYMBOL LOOKUP</h3>
          </div>
          <button
            onClick={toggleSymbolLookup}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          <div className="mb-4">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search symbols..."
              className="w-full bg-gray-900 border border-gray-600 text-white px-3 py-2 font-mono text-sm focus:outline-none focus:border-orange-400"
              autoFocus
            />
          </div>

          <div className="space-y-1 max-h-64 overflow-y-auto">
            {searchResults.map((item) => (
              <div
                key={item.symbol}
                className="flex items-center justify-between p-2 border border-gray-700 hover:bg-gray-900 transition-colors cursor-pointer"
                onClick={() => {
                  console.log(`Selected symbol: ${item.symbol}`);
                  toggleSymbolLookup();
                }}
              >
                <div>
                  <div className="font-mono text-sm text-white">{item.symbol}</div>
                  <div className="text-xs text-gray-400">{item.name}</div>
                </div>
                <div className="text-xs text-orange-400 font-mono">{item.type}</div>
              </div>
            ))}
          </div>

          {searchResults.length === 0 && searchQuery && (
            <div className="text-center text-gray-500 font-mono text-sm py-8">
              No symbols found for "{searchQuery}"
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Notifications Modal
export const NotificationsModal: React.FC = () => {
  const { showNotifications, toggleNotifications, notifications } = useTradingStore();

  const allNotifications = [
    { type: 'trading', time: '14:28:45', message: 'Risk limit approaching 80%', level: 'warning' },
    { type: 'system', time: '14:15:22', message: 'Database connection restored', level: 'info' },
    { type: 'message', time: '13:45:10', message: 'Strategy performance report ready', level: 'info' },
    { type: 'message', time: '13:30:05', message: 'Daily P&L summary generated', level: 'info' },
    { type: 'message', time: '12:15:30', message: 'Market regime change detected', level: 'info' },
  ];

  if (!showNotifications) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-black border border-gray-600 w-96 max-w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-600">
          <div className="flex items-center space-x-2">
            <Bell className="w-5 h-5 text-orange-400" />
            <h3 className="text-orange-400 font-mono text-sm font-bold">NOTIFICATIONS</h3>
            <span className="bg-orange-900 text-orange-400 px-2 py-1 rounded text-xs font-mono">
              {notifications.systemAlerts + notifications.tradingAlerts + notifications.messages}
            </span>
          </div>
          <button
            onClick={toggleNotifications}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {allNotifications.map((notification, index) => (
              <div
                key={index}
                className="flex items-start space-x-3 p-2 border border-gray-700 hover:bg-gray-900 transition-colors"
              >
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  notification.level === 'warning' ? 'bg-yellow-400' : 'bg-blue-400'
                }`}></div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-mono px-2 py-1 rounded ${
                      notification.type === 'trading' ? 'bg-yellow-900 text-yellow-400' :
                      notification.type === 'system' ? 'bg-red-900 text-red-400' :
                      'bg-blue-900 text-blue-400'
                    }`}>
                      {notification.type.toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-500 font-mono">{notification.time}</span>
                  </div>
                  <div className="text-sm text-white mt-1">{notification.message}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 flex space-x-2">
            <button className="flex-1 py-2 px-4 border border-gray-600 text-gray-400 hover:bg-gray-900 transition-colors font-mono text-sm">
              MARK ALL READ
            </button>
            <button className="py-2 px-4 border border-gray-600 text-gray-400 hover:bg-gray-900 transition-colors font-mono text-sm">
              CLEAR
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Combined component that renders all modals
const QuickActionModals: React.FC = () => {
  return (
    <>
      <QuickChartModal />
      <FastOrderModal />
      <WatchlistModal />
      <SymbolLookupModal />
      <NotificationsModal />
    </>
  );
};

export default QuickActionModals;