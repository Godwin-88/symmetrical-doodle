/**
 * Derivatives Service
 * Client for derivatives pricing, structuring, and backtesting API
 */

const API_BASE = import.meta.env.VITE_INTELLIGENCE_API_URL || 'http://localhost:8000';
const DERIVATIVES_API = `${API_BASE}/api/v1/derivatives`;

// ============================================================================
// Types
// ============================================================================

export interface AssetInfo {
  symbol: string;
  name: string;
  asset_class: 'forex' | 'crypto' | 'commodity';
  base_currency: string;
  quote_currency: string;
  margin_requirement: number;
}

export interface MarketTick {
  symbol: string;
  timestamp: string;
  bid: number;
  ask: number;
  last: number;
  mid: number;
  spread_bps: number;
  volume?: number;
  provider?: string;
}

export interface OHLCV {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  interval: string;
  provider?: string;
}

export interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  vanna?: number;
  volga?: number;
  charm?: number;
}

export interface OptionPriceResult {
  price: number;
  greeks: Greeks;
  intrinsic_value: number;
  time_value: number;
  breakeven: number;
  probability_itm: number;
  pricing_model: string;
  valuation_date: string;
}

export interface OptionsChainEntry {
  strike: number;
  call: {
    price: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    iv: number;
  };
  put: {
    price: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    iv: number;
  };
}

export interface OptionsChain {
  underlying: string;
  spot_price: number;
  expiry_date: string;
  volatility: number;
  chain: OptionsChainEntry[];
}

export interface FuturesPrice {
  underlying: string;
  spot_price: number;
  fair_value: number;
  basis: number;
  basis_pct: number;
  implied_repo_rate: number;
  time_to_expiry_years: number;
  risk_free_rate: number;
  convenience_yield: number;
  storage_cost: number;
}

export interface StructuredProductResult {
  product_name: string;
  product_type: string;
  total_price: number;
  net_greeks: {
    delta: number;
    gamma: number;
    vega: number;
    theta: number;
  };
  legs: Array<{
    type: string;
    option_type?: string;
    strike?: number;
    expiry?: string;
    quantity: number;
    price: number;
    delta?: number;
  }>;
  payoff_diagram: {
    spots: number[];
    payoffs: number[];
  };
  max_profit: number;
  max_loss: number;
  breakevens: number[];
  valuation_date: string;
}

export interface BacktestConfig {
  name: string;
  strategy_type: 'covered_call' | 'iron_condor' | 'custom';
  underlying: string;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  slippage_bps?: number;
  commission_per_contract?: number;
  risk_free_rate?: number;
  strategy_params?: Record<string, unknown>;
}

export interface BacktestResult {
  config: BacktestConfig;
  performance: {
    start_equity: number;
    end_equity: number;
    total_return: number;
    annualized_return: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    max_drawdown_duration: number;
    calmar_ratio: number;
    win_rate: number;
    profit_factor: number;
  };
  trade_statistics: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    avg_trade_duration_hours: number;
    total_commission: number;
    total_slippage: number;
  };
  greeks_statistics: {
    avg_net_delta: number;
    max_net_delta: number;
    avg_net_gamma: number;
    avg_net_vega: number;
    avg_net_theta: number;
  };
  equity_curve: Array<{ timestamp: string; equity: number }>;
  trades: Array<Record<string, unknown>>;
}

export interface BacktestStatus {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  progress: number;
  start_time: string;
  end_time?: string;
  result?: BacktestResult;
}

export interface StrategyTemplate {
  type: string;
  name: string;
  description: string;
  parameters: Record<string, { type: string; default: unknown; description: string }>;
  risk_level: string;
  market_outlook: string;
}

export interface ProductTemplate {
  type: string;
  description: string;
  required_fields: string[];
  risk_profile: string;
}

// ============================================================================
// API Functions
// ============================================================================

async function fetchWithFallback<T>(url: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `API error: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    console.warn(`API call failed: ${url}`, error);
    throw error;
  }
}

// ============================================================================
// Market Data
// ============================================================================

export async function listAssets(): Promise<AssetInfo[]> {
  return fetchWithFallback<AssetInfo[]>(`${DERIVATIVES_API}/assets`);
}

export async function listAssetsByClass(assetClass: 'forex' | 'crypto' | 'commodity'): Promise<AssetInfo[]> {
  return fetchWithFallback<AssetInfo[]>(`${DERIVATIVES_API}/assets/${assetClass}`);
}

export async function getTick(symbol: string): Promise<MarketTick> {
  return fetchWithFallback<MarketTick>(`${DERIVATIVES_API}/tick`, {
    method: 'POST',
    body: JSON.stringify({ symbol }),
  });
}

export async function getMultipleTicks(symbols: string[]): Promise<Record<string, MarketTick>> {
  return fetchWithFallback<Record<string, MarketTick>>(`${DERIVATIVES_API}/ticks`, {
    method: 'POST',
    body: JSON.stringify(symbols),
  });
}

export async function getOHLCV(
  symbol: string,
  interval: string,
  startDate: string,
  endDate: string
): Promise<{ symbol: string; interval: string; data: OHLCV[]; count: number }> {
  return fetchWithFallback(`${DERIVATIVES_API}/ohlcv`, {
    method: 'POST',
    body: JSON.stringify({
      symbol,
      interval,
      start_date: startDate,
      end_date: endDate,
    }),
  });
}

// ============================================================================
// Options Pricing
// ============================================================================

export async function priceOption(params: {
  underlying: string;
  spot_price: number;
  strike: number;
  expiry_date: string;
  option_type: 'call' | 'put';
  volatility?: number;
  option_style?: 'european' | 'american';
  risk_free_rate?: number;
  dividend_yield?: number;
}): Promise<OptionPriceResult> {
  return fetchWithFallback<OptionPriceResult>(`${DERIVATIVES_API}/options/price`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function calculateImpliedVol(params: {
  market_price: number;
  spot_price: number;
  strike: number;
  expiry_date: string;
  option_type: 'call' | 'put';
}): Promise<{ implied_volatility: number; implied_volatility_pct: number }> {
  return fetchWithFallback(`${DERIVATIVES_API}/options/implied-vol`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getOptionsChain(params: {
  underlying: string;
  spot_price: number;
  expiry_date: string;
  volatility?: number;
  strikes_count?: number;
}): Promise<OptionsChain> {
  const queryParams = new URLSearchParams({
    underlying: params.underlying,
    spot_price: params.spot_price.toString(),
    expiry_date: params.expiry_date,
  });

  if (params.volatility) queryParams.append('volatility', params.volatility.toString());
  if (params.strikes_count) queryParams.append('strikes_count', params.strikes_count.toString());

  return fetchWithFallback<OptionsChain>(`${DERIVATIVES_API}/options/chain?${queryParams}`);
}

// ============================================================================
// Futures Pricing
// ============================================================================

export async function priceFutures(params: {
  underlying: string;
  spot_price: number;
  expiry_date: string;
  risk_free_rate?: number;
  convenience_yield?: number;
  storage_cost?: number;
}): Promise<FuturesPrice> {
  return fetchWithFallback<FuturesPrice>(`${DERIVATIVES_API}/futures/price`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// ============================================================================
// Structured Products
// ============================================================================

export async function priceStructuredProduct(params: {
  product_type: 'straddle' | 'strangle' | 'butterfly' | 'iron_condor' | 'calendar_spread';
  underlying: string;
  spot_price: number;
  volatility: number;
  expiry_date: string;
  strike?: number;
  call_strike?: number;
  put_strike?: number;
  lower_strike?: number;
  middle_strike?: number;
  upper_strike?: number;
  put_lower?: number;
  put_upper?: number;
  call_lower?: number;
  call_upper?: number;
  near_expiry?: string;
  far_expiry?: string;
}): Promise<StructuredProductResult> {
  return fetchWithFallback<StructuredProductResult>(`${DERIVATIVES_API}/structured/price`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getProductTemplates(): Promise<{ templates: ProductTemplate[] }> {
  return fetchWithFallback(`${DERIVATIVES_API}/structured/templates`);
}

// ============================================================================
// Backtesting
// ============================================================================

export async function startBacktest(config: BacktestConfig): Promise<BacktestStatus> {
  return fetchWithFallback<BacktestStatus>(`${DERIVATIVES_API}/backtest/run`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function getBacktestStatus(backtestId: string): Promise<BacktestStatus> {
  return fetchWithFallback<BacktestStatus>(`${DERIVATIVES_API}/backtest/${backtestId}`);
}

export async function listBacktests(): Promise<BacktestStatus[]> {
  return fetchWithFallback<BacktestStatus[]>(`${DERIVATIVES_API}/backtest`);
}

export async function deleteBacktest(backtestId: string): Promise<void> {
  await fetchWithFallback(`${DERIVATIVES_API}/backtest/${backtestId}`, {
    method: 'DELETE',
  });
}

export async function getStrategyTemplates(): Promise<{ strategies: StrategyTemplate[] }> {
  return fetchWithFallback(`${DERIVATIVES_API}/strategies/templates`);
}

// ============================================================================
// Health Check
// ============================================================================

export async function checkDerivativesHealth(): Promise<{
  status: string;
  timestamp: string;
  market_data_providers: string[];
  active_backtests: number;
  pricing_engine: string;
}> {
  return fetchWithFallback(`${DERIVATIVES_API}/health`);
}

// ============================================================================
// Mock Data Fallbacks (for offline development)
// ============================================================================

export function getMockAssets(): AssetInfo[] {
  return [
    { symbol: 'XAUUSD', name: 'Gold', asset_class: 'commodity', base_currency: 'XAU', quote_currency: 'USD', margin_requirement: 5.0 },
    { symbol: 'XAGUSD', name: 'Silver', asset_class: 'commodity', base_currency: 'XAG', quote_currency: 'USD', margin_requirement: 5.0 },
    { symbol: 'EURUSD', name: 'Euro/US Dollar', asset_class: 'forex', base_currency: 'EUR', quote_currency: 'USD', margin_requirement: 3.33 },
    { symbol: 'GBPUSD', name: 'British Pound/US Dollar', asset_class: 'forex', base_currency: 'GBP', quote_currency: 'USD', margin_requirement: 3.33 },
    { symbol: 'USDJPY', name: 'US Dollar/Japanese Yen', asset_class: 'forex', base_currency: 'USD', quote_currency: 'JPY', margin_requirement: 3.33 },
    { symbol: 'BTCUSD', name: 'Bitcoin/US Dollar', asset_class: 'crypto', base_currency: 'BTC', quote_currency: 'USD', margin_requirement: 50.0 },
    { symbol: 'ETHUSD', name: 'Ethereum/US Dollar', asset_class: 'crypto', base_currency: 'ETH', quote_currency: 'USD', margin_requirement: 50.0 },
  ];
}

export function getMockTick(symbol: string): MarketTick {
  const prices: Record<string, number> = {
    'XAUUSD': 2045.50,
    'XAGUSD': 23.15,
    'EURUSD': 1.0892,
    'GBPUSD': 1.2698,
    'USDJPY': 148.25,
    'BTCUSD': 43250.00,
    'ETHUSD': 2285.00,
  };

  const price = prices[symbol] || 100;
  const spread = price * 0.0002;

  return {
    symbol,
    timestamp: new Date().toISOString(),
    bid: price - spread / 2,
    ask: price + spread / 2,
    last: price,
    mid: price,
    spread_bps: 2.0,
    volume: Math.random() * 1000000,
    provider: 'mock',
  };
}

export function getMockOptionsChain(underlying: string, spotPrice: number): OptionsChain {
  const chain: OptionsChainEntry[] = [];
  const step = spotPrice * 0.025;
  const center = Math.round(spotPrice / step) * step;

  for (let i = -5; i <= 5; i++) {
    const strike = center + i * step;
    chain.push({
      strike,
      call: {
        price: Math.max(spotPrice - strike, 0) + Math.random() * 5,
        delta: 0.5 - i * 0.08,
        gamma: 0.02,
        theta: -0.05,
        vega: 0.15,
        iv: 20 + Math.abs(i) * 2,
      },
      put: {
        price: Math.max(strike - spotPrice, 0) + Math.random() * 5,
        delta: -0.5 - i * 0.08,
        gamma: 0.02,
        theta: -0.05,
        vega: 0.15,
        iv: 20 + Math.abs(i) * 2,
      },
    });
  }

  return {
    underlying,
    spot_price: spotPrice,
    expiry_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    volatility: 0.2,
    chain,
  };
}
