/**
 * Deriv API Service - Real-time Market Data Integration
 * 
 * Provides access to Deriv demo trading platform:
 * - Real-time tick data (bid/ask prices)
 * - Account information
 * - Position tracking
 * - Order placement
 * 
 * Architecture:
 * - Frontend → Python intelligence-layer → Deriv WebSocket API
 * - Automatic fallback to mock data when backend unavailable
 */

import { intelligenceApi } from './api';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface DerivTick {
  symbol: string;
  bid: number;
  ask: number;
  timestamp: string;
  epoch?: number;
}

export interface DerivAccount {
  balance: number;
  currency: string;
  loginid: string;
  is_virtual: boolean;
}

export interface DerivPosition {
  contract_id: string;
  symbol: string;
  contract_type: string;
  buy_price: number;
  current_spot: number;
  profit: number;
  payout: number;
  purchase_time: string;
  pnl_pct: number;
}

export interface DerivSymbol {
  symbol: string;
  display_name: string;
  market: string;
  submarket: string;
  is_trading_suspended: boolean;
}

export interface DerivConnectionStatus {
  connected: boolean;
  authorized: boolean;
  account_balance?: number;
  last_update: string;
}

// ============================================================================
// HARDCODED FALLBACK DATA
// ============================================================================

const FALLBACK_ACCOUNT: DerivAccount = {
  balance: 10000.00,
  currency: 'USD',
  loginid: 'DEMO_ACCOUNT',
  is_virtual: true
};

const FALLBACK_SYMBOLS: DerivSymbol[] = [
  { symbol: 'frxEURUSD', display_name: 'EUR/USD', market: 'forex', submarket: 'major_pairs', is_trading_suspended: false },
  { symbol: 'frxGBPUSD', display_name: 'GBP/USD', market: 'forex', submarket: 'major_pairs', is_trading_suspended: false },
  { symbol: 'frxUSDJPY', display_name: 'USD/JPY', market: 'forex', submarket: 'major_pairs', is_trading_suspended: false },
  { symbol: 'frxAUDUSD', display_name: 'AUD/USD', market: 'forex', submarket: 'major_pairs', is_trading_suspended: false },
  { symbol: 'frxUSDCAD', display_name: 'USD/CAD', market: 'forex', submarket: 'major_pairs', is_trading_suspended: false },
  { symbol: 'R_100', display_name: 'Volatility 100 Index', market: 'synthetic_index', submarket: 'random_index', is_trading_suspended: false },
  { symbol: 'R_50', display_name: 'Volatility 50 Index', market: 'synthetic_index', submarket: 'random_index', is_trading_suspended: false },
  { symbol: 'R_25', display_name: 'Volatility 25 Index', market: 'synthetic_index', submarket: 'random_index', is_trading_suspended: false },
  { symbol: 'BOOM300', display_name: 'Boom 300 Index', market: 'synthetic_index', submarket: 'crash_boom', is_trading_suspended: false },
  { symbol: 'CRASH300', display_name: 'Crash 300 Index', market: 'synthetic_index', submarket: 'crash_boom', is_trading_suspended: false },
];

// Generate realistic tick data
function generateFallbackTick(symbol: string): DerivTick {
  const basePrice = symbol.startsWith('frx') ? 1.0850 : 1234.56;
  const spread = symbol.startsWith('frx') ? 0.0002 : 0.02;
  const volatility = symbol.startsWith('frx') ? 0.0001 : 0.5;
  
  const mid = basePrice + (Math.random() - 0.5) * volatility * 10;
  const bid = mid - spread / 2;
  const ask = mid + spread / 2;
  
  return {
    symbol,
    bid: parseFloat(bid.toFixed(symbol.startsWith('frx') ? 5 : 2)),
    ask: parseFloat(ask.toFixed(symbol.startsWith('frx') ? 5 : 2)),
    timestamp: new Date().toISOString(),
    epoch: Math.floor(Date.now() / 1000)
  };
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * Get Deriv account information
 */
export async function getDerivAccount(): Promise<DerivAccount> {
  try {
    const response = await intelligenceApi.get('/deriv/account');
    return response as DerivAccount;
  } catch (error) {
    console.warn('Deriv account unavailable, using fallback:', error);
    return FALLBACK_ACCOUNT;
  }
}

/**
 * Get connection status
 */
export async function getDerivStatus(): Promise<DerivConnectionStatus> {
  try {
    const response = await intelligenceApi.get('/deriv/status');
    return response as DerivConnectionStatus;
  } catch (error) {
    console.warn('Deriv status unavailable, using fallback');
    return {
      connected: false,
      authorized: false,
      account_balance: FALLBACK_ACCOUNT.balance,
      last_update: new Date().toISOString()
    };
  }
}

/**
 * Get real-time tick for a symbol
 */
export async function getDerivTick(symbol: string): Promise<DerivTick> {
  try {
    const response = await intelligenceApi.get(`/deriv/tick/${symbol}`);
    return response as DerivTick;
  } catch (error) {
    // Return realistic fallback data
    return generateFallbackTick(symbol);
  }
}

/**
 * Get multiple ticks at once
 */
export async function getDerivTicks(symbols: string[]): Promise<Map<string, DerivTick>> {
  try {
    const response = await intelligenceApi.post('/deriv/ticks', { symbols });
    const ticks = (response as any).ticks as DerivTick[];
    return new Map(ticks.map(tick => [tick.symbol, tick]));
  } catch (error) {
    // Return fallback ticks for all symbols
    const tickMap = new Map<string, DerivTick>();
    symbols.forEach(symbol => {
      tickMap.set(symbol, generateFallbackTick(symbol));
    });
    return tickMap;
  }
}

/**
 * Get list of available trading symbols
 */
export async function getDerivSymbols(): Promise<DerivSymbol[]> {
  try {
    const response = await intelligenceApi.get('/deriv/symbols');
    return (response as any).symbols as DerivSymbol[];
  } catch (error) {
    console.warn('Deriv symbols unavailable, using fallback');
    return FALLBACK_SYMBOLS;
  }
}

/**
 * Subscribe to tick updates for a symbol
 * Returns a cleanup function to unsubscribe
 */
export async function subscribeToTicks(
  symbol: string,
  callback: (tick: DerivTick) => void
): Promise<() => void> {
  try {
    // Try to subscribe via backend
    await intelligenceApi.post(`/deriv/subscribe/${symbol}`);
    
    // Poll for updates
    const interval = setInterval(async () => {
      const tick = await getDerivTick(symbol);
      callback(tick);
    }, 1000);
    
    return () => clearInterval(interval);
  } catch (error) {
    console.warn('Deriv subscription unavailable, using polling fallback');
    
    // Fallback: poll with generated data
    const interval = setInterval(() => {
      callback(generateFallbackTick(symbol));
    }, 1000);
    
    return () => clearInterval(interval);
  }
}

/**
 * Get all open positions
 */
export async function getDerivPositions(): Promise<DerivPosition[]> {
  try {
    const response = await intelligenceApi.get('/deriv/positions');
    return (response as any).positions as DerivPosition[];
  } catch (error) {
    console.warn('Deriv positions unavailable');
    return [];
  }
}

/**
 * Place an order (for future use in F6)
 */
export async function placeDerivOrder(
  symbol: string,
  type: 'CALL' | 'PUT',
  amount: number,
  duration: number,
  duration_unit: 't' | 'm' | 'h' = 't'
): Promise<any> {
  try {
    const response = await intelligenceApi.post('/deriv/order', {
      symbol,
      type,
      amount,
      duration,
      duration_unit
    });
    return response;
  } catch (error) {
    console.error('Failed to place Deriv order:', error);
    throw error;
  }
}

/**
 * Close a position (for future use in F5)
 */
export async function closeDerivPosition(contractId: string): Promise<any> {
  try {
    const response = await intelligenceApi.post(`/deriv/position/${contractId}/close`);
    return response;
  } catch (error) {
    console.error('Failed to close Deriv position:', error);
    throw error;
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Calculate spread in basis points
 */
export function calculateSpreadBps(tick: DerivTick): number {
  if (tick.bid === 0) return 0;
  return ((tick.ask - tick.bid) / tick.bid) * 10000;
}

/**
 * Calculate mid price
 */
export function calculateMidPrice(tick: DerivTick): number {
  return (tick.bid + tick.ask) / 2;
}

/**
 * Format symbol for display
 */
export function formatSymbol(symbol: string): string {
  if (symbol.startsWith('frx')) {
    return symbol.substring(3).replace(/(.{3})/, '$1/');
  }
  return symbol.replace(/_/g, ' ');
}

/**
 * Get symbol category
 */
export function getSymbolCategory(symbol: string): 'forex' | 'synthetic' | 'commodity' {
  if (symbol.startsWith('frx')) return 'forex';
  if (symbol.startsWith('R_') || symbol.includes('BOOM') || symbol.includes('CRASH')) return 'synthetic';
  return 'commodity';
}

/**
 * Check if market is open (simplified - Deriv synthetics are 24/7)
 */
export function isMarketOpen(symbol: string): boolean {
  const category = getSymbolCategory(symbol);
  if (category === 'synthetic') return true; // Synthetics trade 24/7
  
  // Forex: check if weekend
  const now = new Date();
  const day = now.getUTCDay();
  const hour = now.getUTCHours();
  
  // Forex closed on weekends
  if (day === 0 || day === 6) return false;
  
  // Forex closed Friday 22:00 UTC to Sunday 22:00 UTC
  if (day === 5 && hour >= 22) return false;
  
  return true;
}

// ============================================================================
// POPULAR SYMBOL LISTS
// ============================================================================

export const POPULAR_FOREX = [
  'frxEURUSD',
  'frxGBPUSD',
  'frxUSDJPY',
  'frxAUDUSD',
  'frxUSDCAD'
];

export const POPULAR_SYNTHETICS = [
  'R_100',
  'R_50',
  'R_25',
  'BOOM300',
  'CRASH300'
];

export const ALL_POPULAR_SYMBOLS = [
  ...POPULAR_FOREX,
  ...POPULAR_SYNTHETICS
];
