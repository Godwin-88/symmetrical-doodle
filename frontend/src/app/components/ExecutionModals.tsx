/**
 * Execution Modals - CRUD Forms for Execution Management
 */

import { useState } from 'react';
import type { Order, ExecutionAlgo, Adapter, CircuitBreaker } from '../../services/executionService';

// ============================================================================
// CREATE ORDER MODAL
// ============================================================================

interface CreateOrderModalProps {
  show: boolean;
  onClose: () => void;
  onCreate: (order: any) => void;
  adapters: Adapter[];
}

export function CreateOrderModal({ show, onClose, onCreate, adapters }: CreateOrderModalProps) {
  const [formData, setFormData] = useState({
    strategyId: '',
    portfolioId: 'PORT-001',
    asset: 'EURUSD',
    side: 'BUY' as 'BUY' | 'SELL',
    size: 10000,
    orderType: 'MARKET' as any,
    limitPrice: 0,
    adapterId: '',
    executionAlgo: {
      type: 'MARKET' as any,
      aggressiveness: 'MEDIUM' as any,
      timeHorizon: 30,
      participationRate: 0.2,
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onCreate(formData);
    onClose();
  };

  if (!show) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[#ff8c00] text-lg">CREATE NEW ORDER</h2>
          <button onClick={onClose} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4">
            {/* Basic Order Info */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">ASSET</label>
                <select
                  value={formData.asset}
                  onChange={(e) => setFormData({ ...formData, asset: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="EURUSD">EUR/USD</option>
                  <option value="GBPUSD">GBP/USD</option>
                  <option value="USDJPY">USD/JPY</option>
                  <option value="AUDUSD">AUD/USD</option>
                  <option value="USDCAD">USD/CAD</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">SIDE</label>
                <select
                  value={formData.side}
                  onChange={(e) => setFormData({ ...formData, side: e.target.value as any })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </select>
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">SIZE</label>
                <input
                  type="number"
                  required
                  value={formData.size}
                  onChange={(e) => setFormData({ ...formData, size: parseFloat(e.target.value) })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">ORDER TYPE</label>
                <select
                  value={formData.orderType}
                  onChange={(e) => setFormData({ ...formData, orderType: e.target.value as any })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                >
                  <option value="MARKET">MARKET</option>
                  <option value="LIMIT">LIMIT</option>
                  <option value="VWAP">VWAP</option>
                  <option value="TWAP">TWAP</option>
                  <option value="POV">POV</option>
                  <option value="ICEBERG">ICEBERG</option>
                </select>
              </div>
            </div>

            {/* Limit Price (if LIMIT order) */}
            {formData.orderType === 'LIMIT' && (
              <div>
                <label className="text-[#666] block mb-1 text-[10px]">LIMIT PRICE</label>
                <input
                  type="number"
                  step="0.0001"
                  value={formData.limitPrice}
                  onChange={(e) => setFormData({ ...formData, limitPrice: parseFloat(e.target.value) })}
                  className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
                />
              </div>
            )}

            {/* Adapter Selection */}
            <div>
              <label className="text-[#666] block mb-1 text-[10px]">EXECUTION ADAPTER</label>
              <select
                value={formData.adapterId}
                onChange={(e) => setFormData({ ...formData, adapterId: e.target.value })}
                required
                className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[10px]"
              >
                <option value="">Select Adapter...</option>
                {adapters.filter(a => a.status === 'CONNECTED').map(adapter => (
                  <option key={adapter.id} value={adapter.id}>
                    {adapter.name} ({adapter.latencyMs}ms)
                  </option>
                ))}
              </select>
            </div>

            {/* Execution Algorithm */}
            {(formData.orderType === 'VWAP' || formData.orderType === 'TWAP' || formData.orderType === 'POV') && (
              <div className="border border-[#444] p-3">
                <div className="text-[#ff8c00] mb-2 text-[10px]">EXECUTION ALGORITHM</div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">AGGRESSIVENESS</label>
                    <select
                      value={formData.executionAlgo.aggressiveness}
                      onChange={(e) => setFormData({
                        ...formData,
                        executionAlgo: { ...formData.executionAlgo, aggressiveness: e.target.value as any }
                      })}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                    >
                      <option value="LOW">LOW</option>
                      <option value="MEDIUM">MEDIUM</option>
                      <option value="HIGH">HIGH</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">TIME HORIZON (MIN)</label>
                    <input
                      type="number"
                      value={formData.executionAlgo.timeHorizon}
                      onChange={(e) => setFormData({
                        ...formData,
                        executionAlgo: { ...formData.executionAlgo, timeHorizon: parseInt(e.target.value) }
                      })}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                    />
                  </div>
                  <div>
                    <label className="text-[#666] block mb-1 text-[9px]">PARTICIPATION RATE</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={formData.executionAlgo.participationRate}
                      onChange={(e) => setFormData({
                        ...formData,
                        executionAlgo: { ...formData.executionAlgo, participationRate: parseFloat(e.target.value) }
                      })}
                      className="w-full bg-[#1a1a1a] border border-[#444] text-[#fff] px-2 py-1 text-[9px]"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Warning */}
            <div className="border border-[#ffff00] bg-[#1a1a1a] p-3 text-[#ffff00] text-[10px]">
              ⚠ Order will be sent to {adapters.find(a => a.id === formData.adapterId)?.name || 'selected adapter'}
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={!formData.adapterId}
                className="flex-1 px-4 py-2 bg-[#00ff00] text-black hover:bg-[#00ff00]/80 text-[10px] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                CREATE ORDER
              </button>
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
              >
                CANCEL
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

// ============================================================================
// ORDER DETAILS MODAL
// ============================================================================

interface OrderDetailsModalProps {
  show: boolean;
  onClose: () => void;
  order: Order | null;
}

export function OrderDetailsModal({ show, onClose, order }: OrderDetailsModalProps) {
  if (!show || !order) return null;

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[#ff8c00] text-lg">ORDER DETAILS: {order.id}</h2>
          <button onClick={onClose} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
        </div>

        <div className="space-y-4">
          {/* Order Info */}
          <div className="grid grid-cols-2 gap-4">
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#ff8c00] mb-2 text-[10px]">ORDER INFORMATION</div>
              <div className="space-y-1 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-[#666]">INTERNAL ID:</span>
                  <span className="text-[#fff]">{order.internalId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">VENUE ID:</span>
                  <span className="text-[#fff]">{order.venueId || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">ASSET:</span>
                  <span className="text-[#00ff00]">{order.asset}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">SIDE:</span>
                  <span className={order.side === 'BUY' ? 'text-[#00ff00]' : 'text-[#ff0000]'}>{order.side}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">SIZE:</span>
                  <span className="text-[#fff]">{formatNumber(order.size, 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">FILLED:</span>
                  <span className="text-[#00ff00]">{formatNumber(order.filledSize, 0)} ({formatNumber((order.filledSize / order.size) * 100, 1)}%)</span>
                </div>
              </div>
            </div>

            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#ff8c00] mb-2 text-[10px]">EXECUTION DETAILS</div>
              <div className="space-y-1 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-[#666]">ORDER TYPE:</span>
                  <span className="text-[#fff]">{order.orderType}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">STATUS:</span>
                  <span className={
                    order.status === 'FILLED' ? 'text-[#00ff00]' :
                    order.status === 'REJECTED' ? 'text-[#ff0000]' :
                    'text-[#ffff00]'
                  }>{order.status}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">AVG FILL PRICE:</span>
                  <span className="text-[#fff]">{order.avgFillPrice ? formatNumber(order.avgFillPrice, 4) : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">SLIPPAGE:</span>
                  <span className="text-[#ffff00]">{order.slippageBps ? formatNumber(order.slippageBps, 2) : 'N/A'} bps</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">LATENCY:</span>
                  <span className="text-[#00ff00]">{order.latencyMs}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#666]">ADAPTER:</span>
                  <span className="text-[#fff]">{order.adapterId}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Lifecycle Timeline */}
          <div className="border border-[#444] p-3 bg-[#0a0a0a]">
            <div className="text-[#ff8c00] mb-2 text-[10px]">LIFECYCLE TIMELINE</div>
            <div className="space-y-1 text-[10px]">
              {order.createdAt && (
                <div className="flex justify-between">
                  <span className="text-[#666]">CREATED:</span>
                  <span className="text-[#fff]">{new Date(order.createdAt).toLocaleString()}</span>
                </div>
              )}
              {order.validatedAt && (
                <div className="flex justify-between">
                  <span className="text-[#666]">VALIDATED:</span>
                  <span className="text-[#fff]">{new Date(order.validatedAt).toLocaleString()}</span>
                </div>
              )}
              {order.sentAt && (
                <div className="flex justify-between">
                  <span className="text-[#666]">SENT:</span>
                  <span className="text-[#fff]">{new Date(order.sentAt).toLocaleString()}</span>
                </div>
              )}
              {order.acknowledgedAt && (
                <div className="flex justify-between">
                  <span className="text-[#666]">ACKNOWLEDGED:</span>
                  <span className="text-[#fff]">{new Date(order.acknowledgedAt).toLocaleString()}</span>
                </div>
              )}
              {order.filledAt && (
                <div className="flex justify-between">
                  <span className="text-[#666]">FILLED:</span>
                  <span className="text-[#00ff00]">{new Date(order.filledAt).toLocaleString()}</span>
                </div>
              )}
            </div>
          </div>

          {/* Rejection Reason */}
          {order.rejectionReason && (
            <div className="border-2 border-[#ff0000] bg-[#1a0000] p-3">
              <div className="text-[#ff0000] text-[10px] font-bold mb-1">REJECTION REASON</div>
              <div className="text-[#ff0000] text-[10px]">{order.rejectionReason}</div>
            </div>
          )}

          <button
            onClick={onClose}
            className="w-full px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
          >
            CLOSE
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// ADAPTER CONFIGURATION MODAL
// ============================================================================

interface AdapterConfigModalProps {
  show: boolean;
  onClose: () => void;
  adapter: Adapter | null;
  onUpdate: (adapterId: string, updates: Partial<Adapter>) => void;
}

export function AdapterConfigModal({ show, onClose, adapter, onUpdate }: AdapterConfigModalProps) {
  if (!show || !adapter) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[#ff8c00] text-lg">ADAPTER CONFIGURATION: {adapter.name}</h2>
          <button onClick={onClose} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
        </div>

        <div className="space-y-4">
          <div className="text-[#666] text-[10px]">
            Adapter configuration interface - Coming soon
          </div>

          <button
            onClick={onClose}
            className="w-full px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
          >
            CLOSE
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// CIRCUIT BREAKER MODAL
// ============================================================================

interface CircuitBreakerModalProps {
  show: boolean;
  onClose: () => void;
  breakers: CircuitBreaker[];
  onUpdate: (breakerId: string, updates: Partial<CircuitBreaker>) => void;
}

export function CircuitBreakerModal({ show, onClose, breakers, onUpdate }: CircuitBreakerModalProps) {
  if (!show) return null;

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-3xl w-full">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[#ff8c00] text-lg">CIRCUIT BREAKERS</h2>
          <button onClick={onClose} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
        </div>

        <div className="space-y-3">
          {breakers.map(breaker => (
            <div key={breaker.id} className="border border-[#444] p-3">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <div className="text-[#00ff00] text-[10px] font-bold">{breaker.name}</div>
                  <div className="text-[#666] text-[9px]">{breaker.type}</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-[9px] ${breaker.breached ? 'text-[#ff0000]' : 'text-[#00ff00]'}`}>
                    {breaker.breached ? '⚠ BREACHED' : '✓ OK'}
                  </span>
                  <button
                    onClick={() => onUpdate(breaker.id, { enabled: !breaker.enabled })}
                    className={`px-2 py-1 text-[8px] border ${
                      breaker.enabled
                        ? 'border-[#00ff00] text-[#00ff00]'
                        : 'border-[#666] text-[#666]'
                    }`}
                  >
                    {breaker.enabled ? 'ENABLED' : 'DISABLED'}
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-[9px]">
                <div>
                  <span className="text-[#666]">THRESHOLD:</span>
                  <span className="text-[#fff] ml-1">{formatNumber(breaker.threshold, 1)}</span>
                </div>
                <div>
                  <span className="text-[#666]">CURRENT:</span>
                  <span className={`ml-1 ${breaker.breached ? 'text-[#ff0000]' : 'text-[#00ff00]'}`}>
                    {formatNumber(breaker.currentValue, 1)}
                  </span>
                </div>
                <div>
                  <span className="text-[#666]">ACTION:</span>
                  <span className="text-[#ffff00] ml-1">{breaker.action}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <button
          onClick={onClose}
          className="w-full mt-4 px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
        >
          CLOSE
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// TCA REPORT MODAL
// ============================================================================

interface TCAReportModalProps {
  show: boolean;
  onClose: () => void;
  orderId: string | null;
}

export function TCAReportModal({ show, onClose, orderId }: TCAReportModalProps) {
  if (!show || !orderId) return null;

  // Mock TCA data
  const tca = {
    orderId,
    expectedCost: 10.5,
    realizedCost: 10.2,
    spreadCapture: 0.3,
    marketImpact: 0.15,
    timingCost: 0.05,
    opportunityCost: 0.1,
    executionQuality: 'GOOD',
  };

  const formatNumber = (num: number, decimals = 2) => {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#0a0a0a] border-2 border-[#ff8c00] p-6 max-w-2xl w-full">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[#ff8c00] text-lg">TCA REPORT: {orderId}</h2>
          <button onClick={onClose} className="text-[#ff8c00] hover:text-[#fff]">✕</button>
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#666] text-[9px]">EXPECTED COST</div>
              <div className="text-[#fff] text-[14px]">${formatNumber(tca.expectedCost)}</div>
            </div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#666] text-[9px]">REALIZED COST</div>
              <div className="text-[#00ff00] text-[14px]">${formatNumber(tca.realizedCost)}</div>
            </div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#666] text-[9px]">SPREAD CAPTURE</div>
              <div className="text-[#00ff00] text-[14px]">${formatNumber(tca.spreadCapture)}</div>
            </div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#666] text-[9px]">MARKET IMPACT</div>
              <div className="text-[#ffff00] text-[14px]">${formatNumber(tca.marketImpact)}</div>
            </div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#666] text-[9px]">TIMING COST</div>
              <div className="text-[#fff] text-[14px]">${formatNumber(tca.timingCost)}</div>
            </div>
            <div className="border border-[#444] p-3 bg-[#0a0a0a]">
              <div className="text-[#666] text-[9px]">OPPORTUNITY COST</div>
              <div className="text-[#fff] text-[14px]">${formatNumber(tca.opportunityCost)}</div>
            </div>
          </div>

          <div className="border border-[#444] p-3 bg-[#0a0a0a]">
            <div className="text-[#666] text-[9px]">EXECUTION QUALITY</div>
            <div className={`text-[14px] ${
              tca.executionQuality === 'EXCELLENT' ? 'text-[#00ff00]' :
              tca.executionQuality === 'GOOD' ? 'text-[#00ff00]' :
              tca.executionQuality === 'FAIR' ? 'text-[#ffff00]' :
              'text-[#ff0000]'
            }`}>
              {tca.executionQuality}
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-full px-4 py-2 border border-[#666] text-[#666] hover:text-[#fff] hover:border-[#fff] text-[10px]"
          >
            CLOSE
          </button>
        </div>
      </div>
    </div>
  );
}
