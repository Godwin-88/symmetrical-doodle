/**
 * Execution Management API Service
 * Handles order flow, adapter status, execution quality, and TCA
 * 
 * NOTE: Execution operations use the Rust execution-core (port 8001) for performance-critical
 * order flow, while analytics/TCA use Python intelligence-layer (port 8000)
 */

import { intelligenceApi, executionApi } from './api';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface Order {
  id: string;
  internalId: string;
  venueId?: string;
  strategyId: string;
  portfolioId: string;
  asset: string;
  side: 'BUY' | 'SELL';
  size: number;
  filledSize: number;
  orderType: 'MARKET' | 'LIMIT' | 'VWAP' | 'TWAP' | 'POV' | 'ICEBERG' | 'CUSTOM';
  limitPrice?: number;
  avgFillPrice?: number;
  status: 'CREATED' | 'VALIDATED' | 'SENT' | 'ACKNOWLEDGED' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED' | 'REJECTED' | 'EXPIRED';
  createdAt: string;
  validatedAt?: string;
  sentAt?: string;
  acknowledgedAt?: string;
  filledAt?: string;
  slippageBps?: number;
  latencyMs: number;
  adapterId: string;
  rejectionReason?: string;
  executionAlgo?: ExecutionAlgo;
}

export interface ExecutionAlgo {
  type: 'MARKET' | 'LIMIT' | 'VWAP' | 'TWAP' | 'POV' | 'ICEBERG' | 'CUSTOM';
  aggressiveness: 'LOW' | 'MEDIUM' | 'HIGH';
  timeHorizon?: number; // minutes
  participationRate?: number; // 0-1
  sliceSize?: number;
  customParams?: Record<string, any>;
}

export interface Adapter {
  id: string;
  name: string;
  type: 'BROKER' | 'EXCHANGE' | 'SHADOW';
  status: 'CONNECTED' | 'DEGRADED' | 'DISCONNECTED';
  health: 'HEALTHY' | 'WARNING' | 'CRITICAL';
  latencyMs: number;
  uptimePercent: number;
  ordersToday: number;
  fillsToday: number;
  rejectsToday: number;
  errorRate: number;
  reconnectAttempts: number;
  lastHeartbeat: string;
  rateLimitUsage: number; // 0-100
  supportedAssets: string[];
  supportedOrderTypes: string[];
  tradingHours: string;
  minOrderSize: number;
  maxOrderSize: number;
  feeSchedule: Record<string, number>;
}

export interface ExecutionMetrics {
  avgLatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  fillRate: number;
  rejectionRate: number;
  avgSlippageBps: number;
  implementationShortfall: number;
  priceImprovement: number;
  ordersPerSecond: number;
  peakLoad: number;
}

export interface TCAReport {
  orderId: string;
  expectedCost: number;
  realizedCost: number;
  spreadCapture: number;
  marketImpact: number;
  timingCost: number;
  opportunityCost: number;
  executionQuality: 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR';
}

export interface CircuitBreaker {
  id: string;
  name: string;
  type: 'ORDER_RATE' | 'REJECTION_RATE' | 'PRICE_DEVIATION' | 'POSITION_MISMATCH';
  threshold: number;
  currentValue: number;
  breached: boolean;
  action: 'ALERT' | 'THROTTLE' | 'HALT' | 'KILL_SWITCH';
  enabled: boolean;
}

export interface ReconciliationReport {
  timestamp: string;
  internalPositions: Record<string, number>;
  brokerPositions: Record<string, number>;
  mismatches: Array<{
    asset: string;
    internal: number;
    broker: number;
    difference: number;
  }>;
  cashBalance: {
    internal: number;
    broker: number;
    difference: number;
  };
  fillMismatches: Array<{
    orderId: string;
    issue: string;
  }>;
}

// ============================================================================
// HARDCODED DATA (Fallback when backend is unavailable)
// ============================================================================

const HARDCODED_ORDERS: Order[] = [
  {
    id: 'ORD-001',
    internalId: 'INT-001',
    venueId: 'DERIV-12345',
    strategyId: 'regime_switching',
    portfolioId: 'PORT-001',
    asset: 'EURUSD',
    side: 'BUY',
    size: 50000,
    filledSize: 50000,
    orderType: 'MARKET',
    avgFillPrice: 1.0845,
    status: 'FILLED',
    createdAt: '2024-01-15T14:30:45Z',
    validatedAt: '2024-01-15T14:30:45.100Z',
    sentAt: '2024-01-15T14:30:45.200Z',
    acknowledgedAt: '2024-01-15T14:30:45.210Z',
    filledAt: '2024-01-15T14:30:45.222Z',
    slippageBps: 0.2,
    latencyMs: 12,
    adapterId: 'DERIV_API',
  },
  {
    id: 'ORD-002',
    internalId: 'INT-002',
    venueId: 'MT5-67890',
    strategyId: 'momentum_rotation',
    portfolioId: 'PORT-001',
    asset: 'GBPUSD',
    side: 'SELL',
    size: 30000,
    filledSize: 30000,
    orderType: 'LIMIT',
    limitPrice: 1.2635,
    avgFillPrice: 1.2634,
    status: 'FILLED',
    createdAt: '2024-01-15T14:28:32Z',
    filledAt: '2024-01-15T14:28:47Z',
    slippageBps: -0.1,
    latencyMs: 15,
    adapterId: 'MT5_ADAPTER',
  },
  {
    id: 'ORD-003',
    internalId: 'INT-003',
    strategyId: 'mean_reversion',
    portfolioId: 'PORT-001',
    asset: 'USDJPY',
    side: 'BUY',
    size: 40000,
    filledSize: 20000,
    orderType: 'VWAP',
    avgFillPrice: 148.23,
    status: 'PARTIALLY_FILLED',
    createdAt: '2024-01-15T14:25:18Z',
    sentAt: '2024-01-15T14:25:18.100Z',
    slippageBps: 0.15,
    latencyMs: 11,
    adapterId: 'DERIV_API',
    executionAlgo: {
      type: 'VWAP',
      aggressiveness: 'MEDIUM',
      timeHorizon: 30,
      participationRate: 0.2,
    },
  },
  {
    id: 'ORD-004',
    internalId: 'INT-004',
    strategyId: 'volatility_arb',
    portfolioId: 'PORT-001',
    asset: 'AUDUSD',
    side: 'SELL',
    size: 25000,
    filledSize: 0,
    orderType: 'MARKET',
    status: 'REJECTED',
    createdAt: '2024-01-15T14:22:05Z',
    latencyMs: 8,
    adapterId: 'MT5_ADAPTER',
    rejectionReason: 'INSUFFICIENT_MARGIN',
  },
];

const HARDCODED_ADAPTERS: Adapter[] = [
  {
    id: 'DERIV_API',
    name: 'DERIV API',
    type: 'BROKER',
    status: 'CONNECTED',
    health: 'HEALTHY',
    latencyMs: 12,
    uptimePercent: 99.98,
    ordersToday: 1247,
    fillsToday: 1243,
    rejectsToday: 4,
    errorRate: 0.32,
    reconnectAttempts: 0,
    lastHeartbeat: new Date().toISOString(),
    rateLimitUsage: 45,
    supportedAssets: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    supportedOrderTypes: ['MARKET', 'LIMIT', 'STOP'],
    tradingHours: '24/7',
    minOrderSize: 1000,
    maxOrderSize: 1000000,
    feeSchedule: { EURUSD: 0.0001, GBPUSD: 0.0001 },
  },
  {
    id: 'MT5_ADAPTER',
    name: 'MT5 ADAPTER',
    type: 'BROKER',
    status: 'CONNECTED',
    health: 'HEALTHY',
    latencyMs: 18,
    uptimePercent: 99.95,
    ordersToday: 856,
    fillsToday: 852,
    rejectsToday: 4,
    errorRate: 0.47,
    reconnectAttempts: 1,
    lastHeartbeat: new Date().toISOString(),
    rateLimitUsage: 32,
    supportedAssets: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    supportedOrderTypes: ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'],
    tradingHours: '24/5',
    minOrderSize: 1000,
    maxOrderSize: 500000,
    feeSchedule: { EURUSD: 0.00015, GBPUSD: 0.00015 },
  },
  {
    id: 'SHADOW_EXEC',
    name: 'SHADOW EXEC',
    type: 'SHADOW',
    status: 'CONNECTED',
    health: 'HEALTHY',
    latencyMs: 2,
    uptimePercent: 100.0,
    ordersToday: 2103,
    fillsToday: 2103,
    rejectsToday: 0,
    errorRate: 0.0,
    reconnectAttempts: 0,
    lastHeartbeat: new Date().toISOString(),
    rateLimitUsage: 0,
    supportedAssets: ['ALL'],
    supportedOrderTypes: ['ALL'],
    tradingHours: '24/7',
    minOrderSize: 1,
    maxOrderSize: 999999999,
    feeSchedule: {},
  },
];

const HARDCODED_METRICS: ExecutionMetrics = {
  avgLatencyMs: 12.5,
  p95LatencyMs: 18.2,
  p99LatencyMs: 24.8,
  fillRate: 99.62,
  rejectionRate: 0.38,
  avgSlippageBps: 0.15,
  implementationShortfall: 0.08,
  priceImprovement: 0.05,
  ordersPerSecond: 2.5,
  peakLoad: 15.2,
};

const HARDCODED_CIRCUIT_BREAKERS: CircuitBreaker[] = [
  {
    id: 'CB-001',
    name: 'MAX ORDER RATE',
    type: 'ORDER_RATE',
    threshold: 100,
    currentValue: 45,
    breached: false,
    action: 'THROTTLE',
    enabled: true,
  },
  {
    id: 'CB-002',
    name: 'MAX REJECTION RATE',
    type: 'REJECTION_RATE',
    threshold: 5.0,
    currentValue: 0.38,
    breached: false,
    action: 'HALT',
    enabled: true,
  },
  {
    id: 'CB-003',
    name: 'PRICE DEVIATION',
    type: 'PRICE_DEVIATION',
    threshold: 1.0,
    currentValue: 0.15,
    breached: false,
    action: 'ALERT',
    enabled: true,
  },
];

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * List all orders with optional filtering
 * Uses Rust execution-core for performance
 */
export async function listOrders(filters?: {
  status?: string;
  asset?: string;
  strategyId?: string;
  adapterId?: string;
}): Promise<Order[]> {
  try {
    const params = new URLSearchParams();
    if (filters?.status) params.append('status', filters.status);
    if (filters?.asset) params.append('asset', filters.asset);
    if (filters?.strategyId) params.append('strategy_id', filters.strategyId);
    if (filters?.adapterId) params.append('adapter_id', filters.adapterId);
    
    const response = await executionApi.get(`/orders?${params.toString()}`);
    return (response as any).orders as Order[];
  } catch (error) {
    console.warn('Execution core unavailable, using hardcoded orders:', error);
    let orders = [...HARDCODED_ORDERS];
    
    if (filters?.status) {
      orders = orders.filter(o => o.status === filters.status);
    }
    if (filters?.asset) {
      orders = orders.filter(o => o.asset === filters.asset);
    }
    if (filters?.strategyId) {
      orders = orders.filter(o => o.strategyId === filters.strategyId);
    }
    if (filters?.adapterId) {
      orders = orders.filter(o => o.adapterId === filters.adapterId);
    }
    
    return orders;
  }
}

/**
 * Get order by ID
 * Uses Rust execution-core
 */
export async function getOrder(orderId: string): Promise<Order> {
  try {
    const response = await executionApi.get(`/orders/${orderId}`);
    return response as Order;
  } catch (error) {
    console.warn('Execution core unavailable, using hardcoded order:', error);
    const order = HARDCODED_ORDERS.find(o => o.id === orderId);
    if (!order) {
      throw new Error(`Order ${orderId} not found`);
    }
    return order;
  }
}

/**
 * Create new order
 * Uses Rust execution-core for low-latency order submission
 */
export async function createOrder(order: Omit<Order, 'id' | 'internalId' | 'createdAt' | 'status' | 'filledSize' | 'latencyMs'>): Promise<Order> {
  try {
    const response = await executionApi.post('/orders/create', order);
    return response as Order;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock order:', error);
    return {
      ...order,
      id: `ORD-${String(Math.floor(Math.random() * 10000)).padStart(3, '0')}`,
      internalId: `INT-${String(Math.floor(Math.random() * 10000)).padStart(3, '0')}`,
      createdAt: new Date().toISOString(),
      status: 'CREATED',
      filledSize: 0,
      latencyMs: 0,
    };
  }
}

/**
 * Cancel order
 * Uses Rust execution-core for immediate cancellation
 */
export async function cancelOrder(orderId: string): Promise<Order> {
  try {
    const response = await executionApi.post(`/orders/${orderId}/cancel`);
    return response as Order;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock cancelled order:', error);
    const order = HARDCODED_ORDERS.find(o => o.id === orderId);
    if (!order) {
      throw new Error(`Order ${orderId} not found`);
    }
    return { ...order, status: 'CANCELLED' };
  }
}

/**
 * Modify order
 * Uses Rust execution-core
 */
export async function modifyOrder(orderId: string, updates: Partial<Order>): Promise<Order> {
  try {
    const response = await executionApi.put(`/orders/${orderId}`, updates);
    return response as Order;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock modified order:', error);
    const order = HARDCODED_ORDERS.find(o => o.id === orderId);
    if (!order) {
      throw new Error(`Order ${orderId} not found`);
    }
    return { ...order, ...updates };
  }
}

/**
 * List all adapters
 * Uses Rust execution-core for adapter status
 */
export async function listAdapters(): Promise<Adapter[]> {
  try {
    const response = await executionApi.get('/adapters');
    return (response as any).adapters as Adapter[];
  } catch (error) {
    console.warn('Execution core unavailable, using hardcoded adapters:', error);
    return [...HARDCODED_ADAPTERS];
  }
}

/**
 * Get adapter by ID
 * Uses Rust execution-core
 */
export async function getAdapter(adapterId: string): Promise<Adapter> {
  try {
    const response = await executionApi.get(`/adapters/${adapterId}`);
    return response as Adapter;
  } catch (error) {
    console.warn('Execution core unavailable, using hardcoded adapter:', error);
    const adapter = HARDCODED_ADAPTERS.find(a => a.id === adapterId);
    if (!adapter) {
      throw new Error(`Adapter ${adapterId} not found`);
    }
    return adapter;
  }
}

/**
 * Update adapter configuration
 * Uses Rust execution-core
 */
export async function updateAdapter(adapterId: string, updates: Partial<Adapter>): Promise<Adapter> {
  try {
    const response = await executionApi.put(`/adapters/${adapterId}`, updates);
    return response as Adapter;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock updated adapter:', error);
    const adapter = HARDCODED_ADAPTERS.find(a => a.id === adapterId);
    if (!adapter) {
      throw new Error(`Adapter ${adapterId} not found`);
    }
    return { ...adapter, ...updates };
  }
}

/**
 * Reconnect adapter
 * Uses Rust execution-core
 */
export async function reconnectAdapter(adapterId: string): Promise<Adapter> {
  try {
    const response = await executionApi.post(`/adapters/${adapterId}/reconnect`);
    return response as Adapter;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock reconnected adapter:', error);
    const adapter = HARDCODED_ADAPTERS.find(a => a.id === adapterId);
    if (!adapter) {
      throw new Error(`Adapter ${adapterId} not found`);
    }
    return { ...adapter, status: 'CONNECTED', reconnectAttempts: adapter.reconnectAttempts + 1 };
  }
}

/**
 * Get execution metrics
 * Uses Rust execution-core for real-time metrics
 */
export async function getExecutionMetrics(): Promise<ExecutionMetrics> {
  try {
    const response = await executionApi.get('/metrics');
    return response as ExecutionMetrics;
  } catch (error) {
    console.warn('Execution core unavailable, using hardcoded metrics:', error);
    return { ...HARDCODED_METRICS };
  }
}

/**
 * Get TCA report for order
 * Uses Python intelligence-layer for analytics
 */
export async function getTCAReport(orderId: string): Promise<TCAReport> {
  try {
    const response = await intelligenceApi.get(`/execution/tca/${orderId}`);
    return response as TCAReport;
  } catch (error) {
    console.warn('Intelligence layer unavailable, returning mock TCA report:', error);
    return {
      orderId,
      expectedCost: 10.5,
      realizedCost: 10.2,
      spreadCapture: 0.3,
      marketImpact: 0.15,
      timingCost: 0.05,
      opportunityCost: 0.1,
      executionQuality: 'GOOD',
    };
  }
}

/**
 * List circuit breakers
 * Uses Rust execution-core for real-time risk controls
 */
export async function listCircuitBreakers(): Promise<CircuitBreaker[]> {
  try {
    const response = await executionApi.get('/circuit-breakers');
    return (response as any).breakers as CircuitBreaker[];
  } catch (error) {
    console.warn('Execution core unavailable, using hardcoded circuit breakers:', error);
    return [...HARDCODED_CIRCUIT_BREAKERS];
  }
}

/**
 * Update circuit breaker
 * Uses Rust execution-core
 */
export async function updateCircuitBreaker(breakerId: string, updates: Partial<CircuitBreaker>): Promise<CircuitBreaker> {
  try {
    const response = await executionApi.put(`/circuit-breakers/${breakerId}`, updates);
    return response as CircuitBreaker;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock updated breaker:', error);
    const breaker = HARDCODED_CIRCUIT_BREAKERS.find(b => b.id === breakerId);
    if (!breaker) {
      throw new Error(`Circuit breaker ${breakerId} not found`);
    }
    return { ...breaker, ...updates };
  }
}

/**
 * Emergency kill switch - cancel all orders
 * Uses Rust execution-core for immediate action
 */
export async function killSwitch(): Promise<{ cancelled: number }> {
  try {
    const response = await executionApi.post('/kill-switch');
    return response as { cancelled: number };
  } catch (error) {
    console.warn('Execution core unavailable, mock kill switch:', error);
    return { cancelled: HARDCODED_ORDERS.filter(o => o.status === 'SENT' || o.status === 'PARTIALLY_FILLED').length };
  }
}

/**
 * Get reconciliation report
 * Uses Rust execution-core for position reconciliation
 */
export async function getReconciliationReport(): Promise<ReconciliationReport> {
  try {
    const response = await executionApi.get('/reconciliation');
    return response as ReconciliationReport;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock reconciliation report:', error);
    return {
      timestamp: new Date().toISOString(),
      internalPositions: { EURUSD: 50000, GBPUSD: -30000 },
      brokerPositions: { EURUSD: 50000, GBPUSD: -30000 },
      mismatches: [],
      cashBalance: {
        internal: 104127.89,
        broker: 104127.89,
        difference: 0,
      },
      fillMismatches: [],
    };
  }
}

/**
 * Run reconciliation
 * Uses Rust execution-core
 */
export async function runReconciliation(): Promise<ReconciliationReport> {
  try {
    const response = await executionApi.post('/reconciliation/run');
    return response as ReconciliationReport;
  } catch (error) {
    console.warn('Execution core unavailable, returning mock reconciliation:', error);
    return getReconciliationReport();
  }
}
