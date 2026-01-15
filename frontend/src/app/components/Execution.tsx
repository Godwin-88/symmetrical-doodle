import { useState, useEffect } from 'react';
import {
  listOrders,
  listAdapters,
  getExecutionMetrics,
  listCircuitBreakers,
  createOrder,
  cancelOrder,
  modifyOrder,
  updateAdapter,
  reconnectAdapter,
  updateCircuitBreaker,
  killSwitch,
  getReconciliationReport,
  runReconciliation,
  type Order,
  type Adapter,
  type ExecutionMetrics,
  type CircuitBreaker,
  type ReconciliationReport,
} from '../../services/executionService';
import {
  CreateOrderModal,
  OrderDetailsModal,
  AdapterConfigModal,
  CircuitBreakerModal,
  TCAReportModal,
} from './ExecutionModals';

export function Execution() {
  // State
  const [orders, setOrders] = useState<Order[]>([]);
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [metrics, setMetrics] = useState<ExecutionMetrics | null>(null);
  const [circuitBreakers, setCircuitBreakers] = useState<CircuitBreaker[]>([]);
  const [reconciliation, setReconciliation] = useState<ReconciliationReport | null>(null);
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null);
  const [selectedAdapter, setSelectedAdapter] = useState<Adapter | null>(null);
  
  // Modal states
  const [showCreateOrderModal, setShowCreateOrderModal] = useState(false);
  const [showOrderDetailsModal, setShowOrderDetailsModal] = useState(false);
  const [showAdapterConfigModal, setShowAdapterConfigModal] = useState(false);
  const [showCircuitBreakerModal, setShowCircuitBreakerModal] = useState(false);
  const [showTCAReportModal, setShowTCAReportModal] = useState(false);
  const [tcaOrderId, setTcaOrderId] = useState<string | null>(null);
  
  // Filter states
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [assetFilter, setAssetFilter] = useState<string>('ALL');
  
  // Initialize data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [ordersData, adaptersData, metricsData, breakersData, reconData] = await Promise.all([
          listOrders(),
          listAdapters(),
          getExecutionMetrics(),
          listCircuitBreakers(),
          getReconciliationReport(),
        ]);
        
        setOrders(ordersData);
        setAdapters(adaptersData);
        setMetrics(metricsData);
        setCircuitBreakers(breakersData);
        setReconciliation(reconData);
      } catch (err) {
        console.error('Failed to fetch execution data:', err);
      }
    };
    
    fetchData();
    
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  // CRUD Operations
  const handleCreateOrder = async (orderData: any) => {
    try {
      const newOrder = await createOrder(orderData);
      setOrders([newOrder, ...orders]);
    } catch (err) {
      console.error('Failed to create order:', err);
    }
  };

  const handleCancelOrder = async (orderId: string) => {
    if (!confirm(`Cancel order ${orderId}?`)) return;
    try {
      const cancelledOrder = await cancelOrder(orderId);
      setOrders(orders.map(o => o.id === orderId ? cancelledOrder : o));
    } catch (err) {
      console.error('Failed to cancel order:', err);
    }
  };

  const handleReconnectAdapter = async (adapterId: string) => {
    try {
      const reconnectedAdapter = await reconnectAdapter(adapterId);
      setAdapters(adapters.map(a => a.id === adapterId ? reconnectedAdapter : a));
    } catch (err) {
      console.error('Failed to reconnect adapter:', err);
    }
  };

  const handleUpdateCircuitBreaker = async (breakerId: string, updates: Partial<CircuitBreaker>) => {
    try {
      const updatedBreaker = await updateCircuitBreaker(breakerId, updates);
      setCircuitBreakers(circuitBreakers.map(b => b.id === breakerId ? updatedBreaker : b));
    } catch (err) {
      console.error('Failed to update circuit breaker:', err);
    }
  };

  const handleKillSwitch = async () => {
    if (!confirm('⚠ EMERGENCY KILL SWITCH - Cancel all active orders?')) return;
    try {
      const result = await killSwitch();
      alert(`Kill switch activated. ${result.cancelled} orders cancelled.`);
      const updatedOrders = await listOrders();
      setOrders(updatedOrders);
    } catch (err) {
      console.error('Failed to execute kill switch:', err);
    }
  };

  const handleRunReconciliation = async () => {
    try {
      const report = await runReconciliation();
      setReconciliation(report);
      if (report.mismatches.length > 0 || report.fillMismatches.length > 0) {
        alert(`⚠ Reconciliation found ${report.mismatches.length} position mismatches and ${report.fillMismatches.length} fill mismatches`);
      } else {
        alert('✓ Reconciliation complete - No mismatches found');
      }
    } catch (err) {
      console.error('Failed to run reconciliation:', err);
    }
  };

  // Filter orders
  const filteredOrders = orders.filter(order => {
    if (statusFilter !== 'ALL' && order.status !== statusFilter) return false;
    if (assetFilter !== 'ALL' && order.asset !== assetFilter) return false;
    return true;
  });

  // Get unique assets for filter
  const uniqueAssets = Array.from(new Set(orders.map(o => o.asset)));

  return (
    <div className="flex h-full font-mono text-xs">
      {/* Left Panel - Order Blotter */}
      <div className="w-80 border-r border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">ORDER BLOTTER</div>
          
          {/* Filters */}
          <div className="space-y-2 mb-4">
            <div>
              <label className="text-[#666] block mb-1 text-[9px]">STATUS</label>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
              >
                <option value="ALL">ALL</option>
                <option value="FILLED">FILLED</option>
                <option value="PARTIALLY_FILLED">PARTIALLY FILLED</option>
                <option value="SENT">SENT</option>
                <option value="REJECTED">REJECTED</option>
                <option value="CANCELLED">CANCELLED</option>
              </select>
            </div>
            <div>
              <label className="text-[#666] block mb-1 text-[9px]">ASSET</label>
              <select
                value={assetFilter}
                onChange={(e) => setAssetFilter(e.target.value)}
                className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
              >
                <option value="ALL">ALL</option>
                {uniqueAssets.map(asset => (
                  <option key={asset} value={asset}>{asset}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Create Order Button */}
          <button
            onClick={() => setShowCreateOrderModal(true)}
            className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors mb-4"
          >
            + NEW ORDER
          </button>

          {/* Order List */}
          <div className="space-y-2">
            {filteredOrders.map((order) => (
              <div
                key={order.id}
                onClick={() => {
                  setSelectedOrder(order);
                  setShowOrderDetailsModal(true);
                }}
                className="border border-[#333] p-2 cursor-pointer hover:border-[#ff8c00] transition-colors"
              >
                <div className="flex justify-between items-start mb-1">
                  <div className="text-[#00ff00] text-[9px]">{order.id}</div>
                  <span className={`text-[8px] ${
                    order.status === 'FILLED' ? 'text-[#00ff00]' :
                    order.status === 'REJECTED' ? 'text-[#ff0000]' :
                    'text-[#ffff00]'
                  }`}>
                    {order.status}
                  </span>
                </div>
                <div className="space-y-1 text-[9px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">{order.asset}</span>
                    <span className={order.side === 'BUY' ? 'text-[#00ff00]' : 'text-[#ff0000]'}>
                      {order.side}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">SIZE:</span>
                    <span className="text-[#fff]">{formatNumber(order.size, 0)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">FILLED:</span>
                    <span className="text-[#00ff00]">{formatNumber((order.filledSize / order.size) * 100, 0)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Center Panel - Adapters & Metrics */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="border-t-2 border-b-2 border-[#ff8c00] py-2 mb-4">
          <div className="text-[#ff8c00] text-sm tracking-wider">
            EXECUTION MANAGEMENT - ORDER FLOW & ADAPTER STATUS
          </div>
        </div>

        <div className="space-y-4">
          {/* Execution Adapters */}
          <div className="border border-[#444]">
            <div className="bg-[#1a1a1a] px-3 py-2 border-b border-[#444]">
              <div className="text-[#ff8c00]">EXECUTION ADAPTERS</div>
            </div>
            <table className="w-full">
              <thead>
                <tr className="bg-[#0a0a0a] text-[#ff8c00] text-[10px]">
                  <th className="px-3 py-2 text-left border-b border-[#444]">ADAPTER</th>
                  <th className="px-3 py-2 text-center border-b border-[#444]">STATUS</th>
                  <th className="px-3 py-2 text-center border-b border-[#444]">HEALTH</th>
                  <th className="px-3 py-2 text-right border-b border-[#444]">LATENCY</th>
                  <th className="px-3 py-2 text-right border-b border-[#444]">UPTIME</th>
                  <th className="px-3 py-2 text-right border-b border-[#444]">FILLS</th>
                  <th className="px-3 py-2 text-right border-b border-[#444]">REJECTS</th>
                  <th className="px-3 py-2 text-center border-b border-[#444]">ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {adapters.map((adapter) => (
                  <tr key={adapter.id} className="border-b border-[#222]">
                    <td className="px-3 py-2 text-[#00ff00]">{adapter.name}</td>
                    <td className="px-3 py-2 text-center">
                      <span className={
                        adapter.status === 'CONNECTED' ? 'text-[#00ff00]' :
                        adapter.status === 'DEGRADED' ? 'text-[#ffff00]' :
                        'text-[#ff0000]'
                      }>
                        {adapter.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-center">
                      <span className={
                        adapter.health === 'HEALTHY' ? 'text-[#00ff00]' :
                        adapter.health === 'WARNING' ? 'text-[#ffff00]' :
                        'text-[#ff0000]'
                      }>
                        {adapter.health}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right text-[#fff]">{adapter.latencyMs}ms</td>
                    <td className="px-3 py-2 text-right text-[#00ff00]">{formatNumber(adapter.uptimePercent)}%</td>
                    <td className="px-3 py-2 text-right text-[#00ff00]">{adapter.fillsToday}</td>
                    <td className="px-3 py-2 text-right text-[#ff0000]">{adapter.rejectsToday}</td>
                    <td className="px-3 py-2 text-center">
                      <button
                        onClick={() => handleReconnectAdapter(adapter.id)}
                        className="text-[#ff8c00] hover:text-[#fff] text-[9px]"
                      >
                        RECONNECT
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Execution Metrics */}
          {metrics && (
            <div className="grid grid-cols-3 gap-4">
              <div className="border border-[#444] p-3">
                <div className="text-[#ff8c00] mb-3">LATENCY METRICS</div>
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">AVG LATENCY</span>
                    <span className="text-[#00ff00]">{formatNumber(metrics.avgLatencyMs, 1)}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">P95 LATENCY</span>
                    <span className="text-[#00ff00]">{formatNumber(metrics.p95LatencyMs, 1)}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">P99 LATENCY</span>
                    <span className="text-[#ffff00]">{formatNumber(metrics.p99LatencyMs, 1)}ms</span>
                  </div>
                </div>
              </div>

              <div className="border border-[#444] p-3">
                <div className="text-[#ff8c00] mb-3">EXECUTION QUALITY</div>
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">FILL RATE</span>
                    <span className="text-[#00ff00]">{formatNumber(metrics.fillRate)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">REJECTION RATE</span>
                    <span className="text-[#ffff00]">{formatNumber(metrics.rejectionRate)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">AVG SLIPPAGE</span>
                    <span className="text-[#00ff00]">{formatNumber(metrics.avgSlippageBps, 2)} bps</span>
                  </div>
                </div>
              </div>

              <div className="border border-[#444] p-3">
                <div className="text-[#ff8c00] mb-3">THROUGHPUT</div>
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#666]">ORDERS/SEC</span>
                    <span className="text-[#00ff00]">{formatNumber(metrics.ordersPerSecond, 1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">PEAK LOAD</span>
                    <span className="text-[#ffff00]">{formatNumber(metrics.peakLoad, 1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#666]">IMPL. SHORTFALL</span>
                    <span className="text-[#00ff00]">{formatNumber(metrics.implementationShortfall, 2)}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Circuit Breakers */}
          <div className="border border-[#444] p-3">
            <div className="flex justify-between items-center mb-3">
              <div className="text-[#ff8c00]">CIRCUIT BREAKERS</div>
              <button
                onClick={() => setShowCircuitBreakerModal(true)}
                className="text-[#ff8c00] hover:text-[#fff] text-[9px]"
              >
                CONFIGURE
              </button>
            </div>
            <div className="space-y-2">
              {circuitBreakers.map(breaker => (
                <div key={breaker.id} className="flex justify-between items-center text-[10px]">
                  <span className="text-[#666]">{breaker.name}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[#fff]">{formatNumber(breaker.currentValue, 1)} / {formatNumber(breaker.threshold, 1)}</span>
                    <span className={breaker.breached ? 'text-[#ff0000]' : 'text-[#00ff00]'}>
                      {breaker.breached ? '⚠ BREACH' : '✓ OK'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Reconciliation */}
          {reconciliation && (
            <div className="border border-[#444] p-3">
              <div className="flex justify-between items-center mb-3">
                <div className="text-[#ff8c00]">RECONCILIATION STATUS</div>
                <button
                  onClick={handleRunReconciliation}
                  className="text-[#ff8c00] hover:text-[#fff] text-[9px]"
                >
                  RUN RECONCILIATION
                </button>
              </div>
              <div className="space-y-2 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-[#666]">POSITION MISMATCHES</span>
                  <span className={reconciliation.mismatches.length > 0 ? 'text-[#ff0000]' : 'text-[#00ff00]'}>
                    {reconciliation.mismatches.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">FILL MISMATCHES</span>
                  <span className={reconciliation.fillMismatches.length > 0 ? 'text-[#ff0000]' : 'text-[#00ff00]'}>
                    {reconciliation.fillMismatches.length}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">CASH BALANCE DIFF</span>
                  <span className={reconciliation.cashBalance.difference !== 0 ? 'text-[#ff0000]' : 'text-[#00ff00]'}>
                    ${formatNumber(Math.abs(reconciliation.cashBalance.difference))}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right Panel - Actions & Controls */}
      <div className="w-80 border-l border-[#444] bg-[#0a0a0a] overflow-y-auto">
        <div className="p-4">
          <div className="text-[#ff8c00] mb-4 text-[10px] tracking-wider">EXECUTION CONTROLS</div>
          
          {/* Emergency Controls */}
          <div className="space-y-2 mb-6">
            <button
              onClick={handleKillSwitch}
              className="w-full py-2 px-3 border-2 border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors font-bold"
            >
              ⚠ KILL SWITCH
            </button>
            <button
              onClick={() => setShowCircuitBreakerModal(true)}
              className="w-full py-2 px-3 border border-[#ffff00] text-[#ffff00] text-[10px] hover:bg-[#ffff00] hover:text-black transition-colors"
            >
              CIRCUIT BREAKERS
            </button>
          </div>

          {/* Order Actions */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">ORDER ACTIONS</div>
            <div className="space-y-2">
              <button
                onClick={() => setShowCreateOrderModal(true)}
                className="w-full py-2 px-3 border border-[#00ff00] text-[#00ff00] text-[10px] hover:bg-[#00ff00] hover:text-black transition-colors"
              >
                + CREATE ORDER
              </button>
              {selectedOrder && (
                <>
                  <button
                    onClick={() => setShowOrderDetailsModal(true)}
                    className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                  >
                    VIEW DETAILS
                  </button>
                  <button
                    onClick={() => {
                      setTcaOrderId(selectedOrder.id);
                      setShowTCAReportModal(true);
                    }}
                    className="w-full py-2 px-3 border border-[#ff8c00] text-[#ff8c00] text-[10px] hover:bg-[#ff8c00] hover:text-black transition-colors"
                  >
                    TCA REPORT
                  </button>
                  {(selectedOrder.status === 'SENT' || selectedOrder.status === 'PARTIALLY_FILLED') && (
                    <button
                      onClick={() => handleCancelOrder(selectedOrder.id)}
                      className="w-full py-2 px-3 border border-[#ff0000] text-[#ff0000] text-[10px] hover:bg-[#ff0000] hover:text-black transition-colors"
                    >
                      CANCEL ORDER
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Adapter Actions */}
          <div className="mb-6">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">ADAPTER ACTIONS</div>
            <div className="space-y-2">
              {adapters.map(adapter => (
                <div key={adapter.id} className="border border-[#444] p-2">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[#00ff00] text-[9px]">{adapter.name}</span>
                    <span className={`text-[8px] ${
                      adapter.status === 'CONNECTED' ? 'text-[#00ff00]' :
                      adapter.status === 'DEGRADED' ? 'text-[#ffff00]' :
                      'text-[#ff0000]'
                    }`}>
                      {adapter.status}
                    </span>
                  </div>
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleReconnectAdapter(adapter.id)}
                      className="flex-1 py-1 border border-[#444] text-[#00ff00] text-[8px] hover:border-[#00ff00]"
                    >
                      RECONNECT
                    </button>
                    <button
                      onClick={() => {
                        setSelectedAdapter(adapter);
                        setShowAdapterConfigModal(true);
                      }}
                      className="flex-1 py-1 border border-[#444] text-[#ff8c00] text-[8px] hover:border-[#ff8c00]"
                    >
                      CONFIG
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="border border-[#444] p-3 bg-[#0a0a0a]">
            <div className="text-[#ff8c00] mb-3 text-[10px] tracking-wider">QUICK STATS</div>
            <div className="space-y-2 text-[10px]">
              <div className="flex justify-between">
                <span className="text-[#666]">TOTAL ORDERS</span>
                <span className="text-[#fff]">{orders.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">ACTIVE ORDERS</span>
                <span className="text-[#ffff00]">
                  {orders.filter(o => o.status === 'SENT' || o.status === 'PARTIALLY_FILLED').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">FILLED TODAY</span>
                <span className="text-[#00ff00]">
                  {orders.filter(o => o.status === 'FILLED').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">REJECTED TODAY</span>
                <span className="text-[#ff0000]">
                  {orders.filter(o => o.status === 'REJECTED').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#666]">ADAPTERS ONLINE</span>
                <span className="text-[#00ff00]">
                  {adapters.filter(a => a.status === 'CONNECTED').length} / {adapters.length}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modals */}
      <CreateOrderModal
        show={showCreateOrderModal}
        onClose={() => setShowCreateOrderModal(false)}
        onCreate={handleCreateOrder}
        adapters={adapters}
      />
      
      <OrderDetailsModal
        show={showOrderDetailsModal}
        onClose={() => setShowOrderDetailsModal(false)}
        order={selectedOrder}
      />
      
      <AdapterConfigModal
        show={showAdapterConfigModal}
        onClose={() => setShowAdapterConfigModal(false)}
        adapter={selectedAdapter}
        onUpdate={updateAdapter}
      />
      
      <CircuitBreakerModal
        show={showCircuitBreakerModal}
        onClose={() => setShowCircuitBreakerModal(false)}
        breakers={circuitBreakers}
        onUpdate={handleUpdateCircuitBreaker}
      />
      
      <TCAReportModal
        show={showTCAReportModal}
        onClose={() => setShowTCAReportModal(false)}
        orderId={tcaOrderId}
      />
    </div>
  );
}
