/**
 * NautilusTrader WebSocket Service
 * Handles real-time updates for backtesting progress, live trading updates, and system monitoring
 * Following multiSourceService patterns for robust error handling and automatic reconnection
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export enum NautilusWebSocketEventType {
  BACKTEST_PROGRESS = 'backtest_progress',
  BACKTEST_COMPLETED = 'backtest_completed',
  BACKTEST_FAILED = 'backtest_failed',
  LIVE_TRADING_UPDATE = 'live_trading_update',
  POSITION_UPDATE = 'position_update',
  TRADE_EXECUTED = 'trade_executed',
  RISK_LIMIT_BREACH = 'risk_limit_breach',
  STRATEGY_TRANSLATION_UPDATE = 'strategy_translation_update',
  SIGNAL_ROUTING_UPDATE = 'signal_routing_update',
  DATA_MIGRATION_UPDATE = 'data_migration_update',
  SYSTEM_STATUS_UPDATE = 'system_status_update',
  ERROR = 'error',
  CONNECTION_STATUS = 'connection_status'
}

export interface NautilusWebSocketMessage {
  type: NautilusWebSocketEventType;
  timestamp: string;
  data: any;
  correlationId?: string;
}

export interface BacktestProgressUpdate {
  backtestId: string;
  progress: number;
  status: string;
  currentStep: string;
  estimatedTimeRemaining?: number;
  metrics?: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    totalTrades: number;
  };
}

export interface LiveTradingUpdate {
  sessionId: string;
  status: string;
  currentCapital: number;
  totalPnl: number;
  dailyPnl: number;
  activePositions: number;
  recentTrades: number;
  riskMetrics: {
    currentDrawdown: number;
    currentLeverage: number;
    var95: number;
  };
}

export interface PositionUpdate {
  sessionId?: string;
  backtestId?: string;
  strategyId: string;
  instrument: string;
  side: 'LONG' | 'SHORT' | 'FLAT';
  quantity: number;
  avgPrice: number;
  unrealizedPnl: number;
  timestamp: string;
}

export interface TradeExecuted {
  sessionId?: string;
  backtestId?: string;
  tradeId: string;
  strategyId: string;
  instrument: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: string;
  pnl?: number;
  commission: number;
  slippage: number;
}

export interface RiskLimitBreach {
  sessionId?: string;
  backtestId?: string;
  limitType: string;
  limitName: string;
  threshold: number;
  currentValue: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  action: 'ALERT' | 'BLOCK' | 'REDUCE' | 'HALT';
  timestamp: string;
}

export interface StrategyTranslationUpdate {
  translationId: string;
  f6StrategyId: string;
  status: string;
  progress: number;
  currentStep: string;
  validationResults?: any[];
  error?: string;
}

export interface SignalRoutingUpdate {
  routingId: string;
  f5SignalId: string;
  targetStrategies: string[];
  status: string;
  deliveredAt?: string;
  latencyMs?: number;
  error?: string;
}

export interface DataMigrationUpdate {
  migrationId: string;
  status: string;
  progress: number;
  totalRecords: number;
  migratedRecords: number;
  failedRecords: number;
  currentStep: string;
  error?: string;
}

export interface SystemStatusUpdate {
  service: string;
  status: 'HEALTHY' | 'DEGRADED' | 'DOWN' | 'UNKNOWN';
  uptime: number;
  lastHeartbeat: string;
  metrics?: {
    requestsPerSecond: number;
    avgResponseTime: number;
    errorRate: number;
    memoryUsage: number;
    cpuUsage: number;
  };
}

export interface ConnectionStatusUpdate {
  connected: boolean;
  reconnectAttempts: number;
  lastError?: string;
  uptime: number;
}

// ============================================================================
// EVENT LISTENER TYPES
// ============================================================================

export type NautilusWebSocketEventListener<T = any> = (data: T) => void;

export interface NautilusWebSocketEventListeners {
  [NautilusWebSocketEventType.BACKTEST_PROGRESS]: NautilusWebSocketEventListener<BacktestProgressUpdate>;
  [NautilusWebSocketEventType.BACKTEST_COMPLETED]: NautilusWebSocketEventListener<BacktestProgressUpdate>;
  [NautilusWebSocketEventType.BACKTEST_FAILED]: NautilusWebSocketEventListener<BacktestProgressUpdate>;
  [NautilusWebSocketEventType.LIVE_TRADING_UPDATE]: NautilusWebSocketEventListener<LiveTradingUpdate>;
  [NautilusWebSocketEventType.POSITION_UPDATE]: NautilusWebSocketEventListener<PositionUpdate>;
  [NautilusWebSocketEventType.TRADE_EXECUTED]: NautilusWebSocketEventListener<TradeExecuted>;
  [NautilusWebSocketEventType.RISK_LIMIT_BREACH]: NautilusWebSocketEventListener<RiskLimitBreach>;
  [NautilusWebSocketEventType.STRATEGY_TRANSLATION_UPDATE]: NautilusWebSocketEventListener<StrategyTranslationUpdate>;
  [NautilusWebSocketEventType.SIGNAL_ROUTING_UPDATE]: NautilusWebSocketEventListener<SignalRoutingUpdate>;
  [NautilusWebSocketEventType.DATA_MIGRATION_UPDATE]: NautilusWebSocketEventListener<DataMigrationUpdate>;
  [NautilusWebSocketEventType.SYSTEM_STATUS_UPDATE]: NautilusWebSocketEventListener<SystemStatusUpdate>;
  [NautilusWebSocketEventType.ERROR]: NautilusWebSocketEventListener<{ message: string; code?: string }>;
  [NautilusWebSocketEventType.CONNECTION_STATUS]: NautilusWebSocketEventListener<ConnectionStatusUpdate>;
}

// ============================================================================
// WEBSOCKET SERVICE CLASS
// ============================================================================

export class NautilusWebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay = 30000; // Max 30 seconds
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private heartbeatInterval = 30000; // 30 seconds
  private isConnected = false;
  private connectionStartTime = 0;
  private listeners: Map<NautilusWebSocketEventType, Set<NautilusWebSocketEventListener>> = new Map();
  private subscriptions: Set<string> = new Set();
  private messageQueue: NautilusWebSocketMessage[] = [];
  private isReconnecting = false;

  constructor(url: string = 'ws://localhost:8000/nautilus/ws') {
    this.url = url;
    this.initializeListeners();
  }

  private initializeListeners(): void {
    // Initialize listener sets for all event types
    Object.values(NautilusWebSocketEventType).forEach(eventType => {
      this.listeners.set(eventType, new Set());
    });
  }

  // ============================================================================
  // CONNECTION MANAGEMENT
  // ============================================================================

  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
          resolve();
          return;
        }

        this.ws = new WebSocket(this.url);
        this.connectionStartTime = Date.now();

        this.ws.onopen = () => {
          console.log('NautilusTrader WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          this.isReconnecting = false;
          
          this.startHeartbeat();
          this.resubscribeAll();
          this.flushMessageQueue();
          
          this.emit(NautilusWebSocketEventType.CONNECTION_STATUS, {
            connected: true,
            reconnectAttempts: this.reconnectAttempts,
            uptime: Date.now() - this.connectionStartTime
          });
          
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: NautilusWebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
            this.emit(NautilusWebSocketEventType.ERROR, {
              message: 'Failed to parse WebSocket message',
              code: 'PARSE_ERROR'
            });
          }
        };

        this.ws.onclose = (event) => {
          console.log('NautilusTrader WebSocket disconnected:', event.code, event.reason);
          this.isConnected = false;
          this.stopHeartbeat();
          
          this.emit(NautilusWebSocketEventType.CONNECTION_STATUS, {
            connected: false,
            reconnectAttempts: this.reconnectAttempts,
            lastError: event.reason,
            uptime: Date.now() - this.connectionStartTime
          });

          if (!this.isReconnecting && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('NautilusTrader WebSocket error:', error);
          this.emit(NautilusWebSocketEventType.ERROR, {
            message: 'WebSocket connection error',
            code: 'CONNECTION_ERROR'
          });
          reject(error);
        };

      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        reject(error);
      }
    });
  }

  public disconnect(): void {
    this.isReconnecting = false;
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.isConnected = false;
    this.subscriptions.clear();
    this.messageQueue = [];
  }

  private scheduleReconnect(): void {
    if (this.isReconnecting || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }

    this.isReconnecting = true;
    this.reconnectAttempts++;
    
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectDelay);
    
    console.log(`Scheduling NautilusTrader WebSocket reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnect attempt failed:', error);
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        } else {
          console.error('Max reconnect attempts reached, giving up');
          this.emit(NautilusWebSocketEventType.ERROR, {
            message: 'Max reconnect attempts reached',
            code: 'MAX_RECONNECT_ATTEMPTS'
          });
        }
      });
    }, delay);
  }

  // ============================================================================
  // HEARTBEAT MANAGEMENT
  // ============================================================================

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({
          type: NautilusWebSocketEventType.CONNECTION_STATUS,
          timestamp: new Date().toISOString(),
          data: { ping: true }
        });
      }
    }, this.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // ============================================================================
  // MESSAGE HANDLING
  // ============================================================================

  private handleMessage(message: NautilusWebSocketMessage): void {
    // Handle heartbeat responses
    if (message.type === NautilusWebSocketEventType.CONNECTION_STATUS && message.data?.pong) {
      return;
    }

    // Emit the message to listeners
    this.emit(message.type, message.data);

    // Log important events
    if (message.type === NautilusWebSocketEventType.RISK_LIMIT_BREACH) {
      console.warn('Risk limit breach:', message.data);
    } else if (message.type === NautilusWebSocketEventType.ERROR) {
      console.error('NautilusTrader error:', message.data);
    }
  }

  private send(message: NautilusWebSocketMessage): void {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        this.messageQueue.push(message);
      }
    } else {
      // Queue message for later delivery
      this.messageQueue.push(message);
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  // ============================================================================
  // EVENT LISTENER MANAGEMENT
  // ============================================================================

  public on<T extends NautilusWebSocketEventType>(
    eventType: T,
    listener: NautilusWebSocketEventListeners[T]
  ): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.add(listener as NautilusWebSocketEventListener);
    }
  }

  public off<T extends NautilusWebSocketEventType>(
    eventType: T,
    listener: NautilusWebSocketEventListeners[T]
  ): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.delete(listener as NautilusWebSocketEventListener);
    }
  }

  private emit<T extends NautilusWebSocketEventType>(
    eventType: T,
    data: Parameters<NautilusWebSocketEventListeners[T]>[0]
  ): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in WebSocket event listener for ${eventType}:`, error);
        }
      });
    }
  }

  // ============================================================================
  // SUBSCRIPTION MANAGEMENT
  // ============================================================================

  public subscribeToBacktest(backtestId: string): void {
    const subscription = `backtest:${backtestId}`;
    this.subscriptions.add(subscription);
    this.send({
      type: NautilusWebSocketEventType.BACKTEST_PROGRESS,
      timestamp: new Date().toISOString(),
      data: { action: 'subscribe', backtestId }
    });
  }

  public unsubscribeFromBacktest(backtestId: string): void {
    const subscription = `backtest:${backtestId}`;
    this.subscriptions.delete(subscription);
    this.send({
      type: NautilusWebSocketEventType.BACKTEST_PROGRESS,
      timestamp: new Date().toISOString(),
      data: { action: 'unsubscribe', backtestId }
    });
  }

  public subscribeToLiveTrading(sessionId: string): void {
    const subscription = `live_trading:${sessionId}`;
    this.subscriptions.add(subscription);
    this.send({
      type: NautilusWebSocketEventType.LIVE_TRADING_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'subscribe', sessionId }
    });
  }

  public unsubscribeFromLiveTrading(sessionId: string): void {
    const subscription = `live_trading:${sessionId}`;
    this.subscriptions.delete(subscription);
    this.send({
      type: NautilusWebSocketEventType.LIVE_TRADING_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'unsubscribe', sessionId }
    });
  }

  public subscribeToStrategyTranslation(translationId: string): void {
    const subscription = `translation:${translationId}`;
    this.subscriptions.add(subscription);
    this.send({
      type: NautilusWebSocketEventType.STRATEGY_TRANSLATION_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'subscribe', translationId }
    });
  }

  public unsubscribeFromStrategyTranslation(translationId: string): void {
    const subscription = `translation:${translationId}`;
    this.subscriptions.delete(subscription);
    this.send({
      type: NautilusWebSocketEventType.STRATEGY_TRANSLATION_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'unsubscribe', translationId }
    });
  }

  public subscribeToSignalRouting(routingId: string): void {
    const subscription = `signal_routing:${routingId}`;
    this.subscriptions.add(subscription);
    this.send({
      type: NautilusWebSocketEventType.SIGNAL_ROUTING_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'subscribe', routingId }
    });
  }

  public unsubscribeFromSignalRouting(routingId: string): void {
    const subscription = `signal_routing:${routingId}`;
    this.subscriptions.delete(subscription);
    this.send({
      type: NautilusWebSocketEventType.SIGNAL_ROUTING_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'unsubscribe', routingId }
    });
  }

  public subscribeToDataMigration(migrationId: string): void {
    const subscription = `data_migration:${migrationId}`;
    this.subscriptions.add(subscription);
    this.send({
      type: NautilusWebSocketEventType.DATA_MIGRATION_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'subscribe', migrationId }
    });
  }

  public unsubscribeFromDataMigration(migrationId: string): void {
    const subscription = `data_migration:${migrationId}`;
    this.subscriptions.delete(subscription);
    this.send({
      type: NautilusWebSocketEventType.DATA_MIGRATION_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'unsubscribe', migrationId }
    });
  }

  public subscribeToSystemStatus(): void {
    const subscription = 'system_status';
    this.subscriptions.add(subscription);
    this.send({
      type: NautilusWebSocketEventType.SYSTEM_STATUS_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'subscribe' }
    });
  }

  public unsubscribeFromSystemStatus(): void {
    const subscription = 'system_status';
    this.subscriptions.delete(subscription);
    this.send({
      type: NautilusWebSocketEventType.SYSTEM_STATUS_UPDATE,
      timestamp: new Date().toISOString(),
      data: { action: 'unsubscribe' }
    });
  }

  private resubscribeAll(): void {
    // Re-establish all subscriptions after reconnection
    this.subscriptions.forEach(subscription => {
      const [type, id] = subscription.split(':');
      
      switch (type) {
        case 'backtest':
          this.subscribeToBacktest(id);
          break;
        case 'live_trading':
          this.subscribeToLiveTrading(id);
          break;
        case 'translation':
          this.subscribeToStrategyTranslation(id);
          break;
        case 'signal_routing':
          this.subscribeToSignalRouting(id);
          break;
        case 'data_migration':
          this.subscribeToDataMigration(id);
          break;
        case 'system_status':
          this.subscribeToSystemStatus();
          break;
      }
    });
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  public getConnectionStatus(): {
    connected: boolean;
    reconnectAttempts: number;
    uptime: number;
    subscriptions: number;
    queuedMessages: number;
  } {
    return {
      connected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      uptime: this.connectionStartTime ? Date.now() - this.connectionStartTime : 0,
      subscriptions: this.subscriptions.size,
      queuedMessages: this.messageQueue.length
    };
  }

  public clearMessageQueue(): void {
    this.messageQueue = [];
  }

  public resetReconnectAttempts(): void {
    this.reconnectAttempts = 0;
    this.reconnectDelay = 1000;
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

// Create a singleton instance for global use
export const nautilusWebSocketService = new NautilusWebSocketService();

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Format WebSocket connection status for display
 */
export function formatConnectionStatus(status: ConnectionStatusUpdate): string {
  if (status.connected) {
    const uptimeMinutes = Math.floor(status.uptime / 60000);
    return `Connected (${uptimeMinutes}m uptime)`;
  } else {
    return `Disconnected (${status.reconnectAttempts} reconnect attempts)`;
  }
}

/**
 * Get status color for WebSocket connection
 */
export function getConnectionStatusColor(connected: boolean): string {
  return connected ? 'text-green-400' : 'text-red-400';
}

/**
 * Format backtest progress update for display
 */
export function formatBacktestProgress(update: BacktestProgressUpdate): string {
  const progress = `${update.progress}%`;
  const step = update.currentStep;
  const eta = update.estimatedTimeRemaining 
    ? ` (ETA: ${Math.ceil(update.estimatedTimeRemaining / 1000)}s)`
    : '';
  
  return `${progress} - ${step}${eta}`;
}

/**
 * Format live trading update for display
 */
export function formatLiveTradingUpdate(update: LiveTradingUpdate): string {
  const pnl = update.totalPnl >= 0 ? `+$${update.totalPnl.toFixed(2)}` : `-$${Math.abs(update.totalPnl).toFixed(2)}`;
  const positions = `${update.activePositions} positions`;
  const leverage = `${update.riskMetrics.currentLeverage.toFixed(1)}x leverage`;
  
  return `${pnl} | ${positions} | ${leverage}`;
}

/**
 * Format risk limit breach for display
 */
export function formatRiskLimitBreach(breach: RiskLimitBreach): string {
  const severity = breach.severity;
  const limit = breach.limitName;
  const value = `${breach.currentValue.toFixed(2)} / ${breach.threshold.toFixed(2)}`;
  
  return `${severity}: ${limit} - ${value}`;
}

/**
 * Get severity color for risk limit breach
 */
export function getRiskBreachColor(severity: string): string {
  switch (severity.toLowerCase()) {
    case 'critical':
      return 'text-red-400';
    case 'high':
      return 'text-orange-400';
    case 'medium':
      return 'text-yellow-400';
    case 'low':
      return 'text-blue-400';
    default:
      return 'text-gray-400';
  }
}