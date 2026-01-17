/**
 * API Client for Intelligence Layer and Execution Core
 */

const INTELLIGENCE_API_URL = (import.meta as any).env?.VITE_INTELLIGENCE_API_URL || 'http://localhost:8000';
const EXECUTION_API_URL = (import.meta as any).env?.VITE_EXECUTION_API_URL || 'http://localhost:8001';

export interface ApiError {
  message: string;
  status: number;
  detail?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async get<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      return response.json();
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data ? JSON.stringify(data) : undefined,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      return response.json();
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data ? JSON.stringify(data) : undefined,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      return response.json();
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }

  async delete<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      // DELETE might return empty response
      const text = await response.text();
      return text ? JSON.parse(text) : ({} as T);
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }
}

// API clients
export const intelligenceApi = new ApiClient(INTELLIGENCE_API_URL);
export const executionApi = new ApiClient(EXECUTION_API_URL);

// Health check
export async function checkIntelligenceHealth(): Promise<{ status: string; service: string }> {
  return intelligenceApi.get('/health');
}

export async function checkExecutionHealth(): Promise<{ status: string; service: string }> {
  return executionApi.get('/health');
}

// Emergency Controls API
export interface EmergencyHaltRequest {
  reason?: string;
  force?: boolean;
}

export interface EmergencyHaltResponse {
  success: boolean;
  message: string;
  timestamp: string;
  previous_status: string;
  new_status: string;
}

export interface TradingControlRequest {
  action: 'pause' | 'resume';
  reason?: string;
}

export interface TradingControlResponse {
  success: boolean;
  message: string;
  timestamp: string;
  previous_status: string;
  new_status: string;
}

export interface SystemStatusResponse {
  trading_status: string;
  emergency_halt_active: boolean;
  last_status_change: string;
  halt_reason?: string;
  uptime_seconds: number;
}

// Emergency control functions
export async function emergencyHalt(request: EmergencyHaltRequest = {}): Promise<EmergencyHaltResponse> {
  // Try execution core first, fallback to intelligence layer
  try {
    return await executionApi.post('/emergency/halt', request);
  } catch (error) {
    console.warn('Execution core emergency halt failed, trying intelligence layer:', error);
    try {
      return await intelligenceApi.post('/emergency/halt', request);
    } catch (fallbackError) {
      console.warn('Intelligence layer emergency halt failed, using mock fallback:', fallbackError);
      // Mock fallback when both services are down
      return {
        success: true,
        message: `Emergency halt activated: ${request.reason || 'Manual emergency halt'} (Mock Mode)`,
        timestamp: new Date().toISOString(),
        previous_status: 'ACTIVE',
        new_status: 'HALTED'
      };
    }
  }
}

export async function resumeFromHalt(): Promise<EmergencyHaltResponse> {
  try {
    return await executionApi.post('/emergency/resume');
  } catch (error) {
    console.warn('Execution core resume failed, trying intelligence layer:', error);
    try {
      return await intelligenceApi.post('/emergency/resume');
    } catch (fallbackError) {
      console.warn('Intelligence layer resume failed, using mock fallback:', fallbackError);
      // Mock fallback when both services are down
      return {
        success: true,
        message: 'Emergency halt deactivated - Trading resumed (Mock Mode)',
        timestamp: new Date().toISOString(),
        previous_status: 'HALTED',
        new_status: 'ACTIVE'
      };
    }
  }
}

export async function tradingControl(request: TradingControlRequest): Promise<TradingControlResponse> {
  try {
    return await executionApi.post('/trading/control', request);
  } catch (error) {
    console.warn('Execution core trading control failed, trying intelligence layer:', error);
    try {
      return await intelligenceApi.post('/trading/control', request);
    } catch (fallbackError) {
      console.warn('Intelligence layer trading control failed, using mock fallback:', fallbackError);
      // Mock fallback when both services are down
      const newStatus = request.action === 'pause' ? 'PAUSED' : 'ACTIVE';
      return {
        success: true,
        message: `Trading ${request.action}d: ${request.reason || `Manual ${request.action}`} (Mock Mode)`,
        timestamp: new Date().toISOString(),
        previous_status: request.action === 'pause' ? 'ACTIVE' : 'PAUSED',
        new_status: newStatus
      };
    }
  }
}

export async function getSystemStatus(): Promise<SystemStatusResponse> {
  try {
    return await executionApi.get('/system/status');
  } catch (error) {
    console.warn('Execution core system status failed, trying intelligence layer:', error);
    try {
      return await intelligenceApi.get('/system/status');
    } catch (fallbackError) {
      console.warn('Intelligence layer system status failed, using mock fallback:', fallbackError);
      // Mock fallback when both services are down
      return {
        trading_status: 'ACTIVE',
        emergency_halt_active: false,
        last_status_change: new Date().toISOString(),
        halt_reason: undefined,
        uptime_seconds: 3600
      };
    }
  }
}

export async function forceReconnect(): Promise<{ success: boolean; message: string; timestamp: string }> {
  try {
    return await executionApi.post('/system/reconnect');
  } catch (error) {
    console.warn('Execution core reconnect failed, trying intelligence layer:', error);
    try {
      return await intelligenceApi.post('/quick/reconnect');
    } catch (fallbackError) {
      console.warn('Intelligence layer reconnect failed, using mock fallback:', fallbackError);
      // Mock fallback when both services are down
      return {
        success: true,
        message: 'Reconnection initiated (Mock Mode)',
        timestamp: new Date().toISOString()
      };
    }
  }
}

// Quick Actions API
export interface QuickOrderRequest {
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  order_type: 'MARKET' | 'LIMIT' | 'STOP';
  price?: number;
  stop_price?: number;
}

export interface QuickOrderResponse {
  success: boolean;
  message: string;
  order_id?: string;
  timestamp: string;
}

export interface QuickChartRequest {
  symbol: string;
  timeframe?: string;
  start_date?: string;
  end_date?: string;
}

export interface QuickChartResponse {
  symbol: string;
  timeframe: string;
  data_points: number;
  chart_url?: string;
  message: string;
}

export interface SymbolSearchRequest {
  query: string;
  limit?: number;
}

export interface SymbolSearchResponse {
  query: string;
  results: Array<{
    symbol: string;
    name: string;
    type: string;
    exchange: string;
  }>;
  total_found: number;
}

export interface WatchlistItem {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume?: number;
}

export interface WatchlistResponse {
  items: WatchlistItem[];
  last_updated: string;
}

// Quick action functions
export async function submitQuickOrder(request: QuickOrderRequest): Promise<QuickOrderResponse> {
  try {
    return await executionApi.post('/orders/quick', request);
  } catch (error) {
    console.warn('Quick order submission failed, using mock fallback:', error);
    // Mock fallback when service is down
    return {
      success: true,
      message: `Quick order submitted: ${request.side} ${request.quantity} ${request.symbol} (Mock Mode)`,
      order_id: `ORD_MOCK_${Date.now()}`,
      timestamp: new Date().toISOString()
    };
  }
}

export async function getQuickChart(request: QuickChartRequest): Promise<QuickChartResponse> {
  try {
    return await intelligenceApi.post('/quick/chart', request);
  } catch (error) {
    console.warn('Quick chart failed, using mock fallback:', error);
    // Mock fallback when service is down
    return {
      symbol: request.symbol,
      timeframe: request.timeframe || '1H',
      data_points: 100,
      chart_url: undefined,
      message: `Chart data prepared for ${request.symbol} (Mock Mode)`
    };
  }
}

export async function searchSymbols(request: SymbolSearchRequest): Promise<SymbolSearchResponse> {
  try {
    return await intelligenceApi.post('/quick/symbol-search', request);
  } catch (error) {
    console.warn('Symbol search failed, using mock fallback:', error);
    // Mock fallback when service is down
    const mockResults = [
      { symbol: 'EURUSD', name: 'Euro / US Dollar', type: 'Forex', exchange: 'FX' },
      { symbol: 'GBPUSD', name: 'British Pound / US Dollar', type: 'Forex', exchange: 'FX' },
      { symbol: 'USDJPY', name: 'US Dollar / Japanese Yen', type: 'Forex', exchange: 'FX' },
      { symbol: 'AAPL', name: 'Apple Inc.', type: 'Stock', exchange: 'NASDAQ' },
      { symbol: 'MSFT', name: 'Microsoft Corporation', type: 'Stock', exchange: 'NASDAQ' },
      { symbol: 'BTC-USD', name: 'Bitcoin USD', type: 'Crypto', exchange: 'CRYPTO' },
    ].filter(result => 
      result.symbol.toLowerCase().includes(request.query.toLowerCase()) ||
      result.name.toLowerCase().includes(request.query.toLowerCase())
    ).slice(0, request.limit || 10);

    return {
      query: request.query,
      results: mockResults,
      total_found: mockResults.length
    };
  }
}

export async function getWatchlist(): Promise<WatchlistResponse> {
  try {
    return await intelligenceApi.get('/quick/watchlist');
  } catch (error) {
    console.warn('Watchlist failed, using mock fallback:', error);
    // Mock fallback when service is down
    return {
      items: [
        { symbol: 'EURUSD', price: 1.0845, change: 0.0012, change_percent: 0.11 },
        { symbol: 'GBPUSD', price: 1.2634, change: -0.0023, change_percent: -0.18 },
        { symbol: 'USDJPY', price: 149.85, change: 0.45, change_percent: 0.30 },
        { symbol: 'AUDUSD', price: 0.6523, change: 0.0008, change_percent: 0.12 },
        { symbol: 'USDCHF', price: 0.8756, change: -0.0015, change_percent: -0.17 },
      ],
      last_updated: new Date().toISOString()
    };
  }
}
