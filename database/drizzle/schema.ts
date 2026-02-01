/**
 * Drizzle ORM Schema for Algorithmic Trading System
 *
 * This schema defines all database tables for Supabase PostgreSQL
 * with support for derivatives trading, backtesting, and market data.
 *
 * Usage:
 *   npm install drizzle-orm @neondatabase/serverless
 *   npm install -D drizzle-kit
 */

import { pgTable, pgSchema, uuid, text, timestamp, decimal, boolean, jsonb, integer, date, uniqueIndex, index } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

// ============================================================================
// SCHEMA DEFINITIONS
// ============================================================================

export const intelligenceSchema = pgSchema('intelligence');
export const executionSchema = pgSchema('execution');
export const simulationSchema = pgSchema('simulation');
export const derivativesSchema = pgSchema('derivatives');

// ============================================================================
// INTELLIGENCE SCHEMA
// ============================================================================

export const marketStateEmbeddings = intelligenceSchema.table('market_state_embeddings', {
  id: uuid('id').primaryKey().defaultRandom(),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  assetId: text('asset_id').notNull(),
  regimeId: text('regime_id'),
  embedding: text('embedding').notNull(), // Vector stored as text for compatibility
  volatility: decimal('volatility', { precision: 10, scale: 6 }),
  liquidity: decimal('liquidity', { precision: 10, scale: 6 }),
  horizon: text('horizon').notNull().default('1h'),
  sourceModel: text('source_model').notNull(),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  timestampIdx: index('idx_market_embeddings_timestamp').on(table.timestamp),
  assetTimeIdx: index('idx_market_embeddings_asset').on(table.assetId, table.timestamp),
}));

export const strategyStateEmbeddings = intelligenceSchema.table('strategy_state_embeddings', {
  id: uuid('id').primaryKey().defaultRandom(),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  strategyId: text('strategy_id').notNull(),
  embedding: text('embedding').notNull(),
  pnlState: decimal('pnl_state', { precision: 18, scale: 8 }),
  drawdown: decimal('drawdown', { precision: 10, scale: 6 }),
  exposure: decimal('exposure', { precision: 10, scale: 6 }),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  timestampIdx: index('idx_strategy_embeddings_timestamp').on(table.timestamp),
}));

export const regimeTrajectoryEmbeddings = intelligenceSchema.table('regime_trajectory_embeddings', {
  id: uuid('id').primaryKey().defaultRandom(),
  startTime: timestamp('start_time', { withTimezone: true }).notNull(),
  endTime: timestamp('end_time', { withTimezone: true }).notNull(),
  embedding: text('embedding').notNull(),
  realizedVol: decimal('realized_vol', { precision: 10, scale: 6 }),
  transitionPath: jsonb('transition_path'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  timeRangeIdx: index('idx_regime_embeddings_time_range').on(table.startTime, table.endTime),
}));

// ============================================================================
// EXECUTION SCHEMA
// ============================================================================

export const orders = executionSchema.table('orders', {
  id: uuid('id').primaryKey().defaultRandom(),
  assetId: text('asset_id').notNull(),
  side: text('side').notNull(), // 'buy' | 'sell'
  quantity: decimal('quantity', { precision: 18, scale: 8 }).notNull(),
  orderType: text('order_type').notNull(), // 'market' | 'limit' | 'stop'
  price: decimal('price', { precision: 18, scale: 8 }),
  status: text('status').notNull().default('pending'),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  assetTimeIdx: index('idx_orders_asset_time').on(table.assetId, table.createdAt),
}));

export const fills = executionSchema.table('fills', {
  id: uuid('id').primaryKey().defaultRandom(),
  orderId: uuid('order_id').references(() => orders.id),
  assetId: text('asset_id').notNull(),
  side: text('side').notNull(),
  quantity: decimal('quantity', { precision: 18, scale: 8 }).notNull(),
  price: decimal('price', { precision: 18, scale: 8 }).notNull(),
  commission: decimal('commission', { precision: 18, scale: 8 }).default('0'),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  orderIdIdx: index('idx_fills_order_id').on(table.orderId),
  timestampIdx: index('idx_fills_timestamp').on(table.timestamp),
}));

export const positions = executionSchema.table('positions', {
  id: uuid('id').primaryKey().defaultRandom(),
  assetId: text('asset_id').notNull().unique(),
  quantity: decimal('quantity', { precision: 18, scale: 8 }).notNull().default('0'),
  averagePrice: decimal('average_price', { precision: 18, scale: 8 }).notNull().default('0'),
  unrealizedPnl: decimal('unrealized_pnl', { precision: 18, scale: 8 }).default('0'),
  realizedPnl: decimal('realized_pnl', { precision: 18, scale: 8 }).default('0'),
  lastUpdated: timestamp('last_updated', { withTimezone: true }).defaultNow(),
});

// ============================================================================
// SIMULATION SCHEMA
// ============================================================================

export const experiments = simulationSchema.table('experiments', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: text('name').notNull(),
  description: text('description'),
  config: jsonb('config').notNull(),
  status: text('status').notNull().default('pending'),
  startTime: timestamp('start_time', { withTimezone: true }),
  endTime: timestamp('end_time', { withTimezone: true }),
  results: jsonb('results'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  statusIdx: index('idx_experiments_status').on(table.status),
}));

export const marketData = simulationSchema.table('market_data', {
  id: uuid('id').primaryKey().defaultRandom(),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  assetId: text('asset_id').notNull(),
  open: decimal('open', { precision: 18, scale: 8 }).notNull(),
  high: decimal('high', { precision: 18, scale: 8 }).notNull(),
  low: decimal('low', { precision: 18, scale: 8 }).notNull(),
  close: decimal('close', { precision: 18, scale: 8 }).notNull(),
  volume: decimal('volume', { precision: 18, scale: 8 }).notNull(),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  assetTimeIdx: index('idx_market_data_asset_time').on(table.assetId, table.timestamp),
}));

// ============================================================================
// DERIVATIVES SCHEMA
// ============================================================================

// Supported assets for derivatives trading
export const assets = derivativesSchema.table('assets', {
  id: uuid('id').primaryKey().defaultRandom(),
  symbol: text('symbol').notNull().unique(),
  name: text('name').notNull(),
  assetClass: text('asset_class').notNull(), // 'forex' | 'crypto' | 'commodity' | 'equity' | 'index'
  baseCurrency: text('base_currency').notNull(),
  quoteCurrency: text('quote_currency').notNull(),
  marginRequirement: decimal('margin_requirement', { precision: 8, scale: 4 }).notNull().default('5.0'),
  minTradeSize: decimal('min_trade_size', { precision: 18, scale: 8 }).notNull().default('0.01'),
  maxTradeSize: decimal('max_trade_size', { precision: 18, scale: 8 }).notNull().default('1000000'),
  tickSize: decimal('tick_size', { precision: 18, scale: 8 }).notNull().default('0.0001'),
  isActive: boolean('is_active').notNull().default(true),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
});

// Market ticks
export const marketTicks = derivativesSchema.table('market_ticks', {
  id: uuid('id').primaryKey().defaultRandom(),
  symbol: text('symbol').notNull().references(() => assets.symbol),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  bid: decimal('bid', { precision: 18, scale: 8 }).notNull(),
  ask: decimal('ask', { precision: 18, scale: 8 }).notNull(),
  last: decimal('last', { precision: 18, scale: 8 }).notNull(),
  mid: decimal('mid', { precision: 18, scale: 8 }).notNull(),
  spreadBps: decimal('spread_bps', { precision: 10, scale: 4 }).notNull(),
  volume: decimal('volume', { precision: 18, scale: 8 }),
  provider: text('provider').notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  symbolTimeIdx: index('idx_market_ticks_symbol_time').on(table.symbol, table.timestamp),
}));

// OHLCV data
export const ohlcv = derivativesSchema.table('ohlcv', {
  id: uuid('id').primaryKey().defaultRandom(),
  symbol: text('symbol').notNull().references(() => assets.symbol),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  interval: text('interval').notNull(), // '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M'
  open: decimal('open', { precision: 18, scale: 8 }).notNull(),
  high: decimal('high', { precision: 18, scale: 8 }).notNull(),
  low: decimal('low', { precision: 18, scale: 8 }).notNull(),
  close: decimal('close', { precision: 18, scale: 8 }).notNull(),
  volume: decimal('volume', { precision: 18, scale: 8 }).notNull(),
  provider: text('provider'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  symbolIntervalTimeIdx: uniqueIndex('idx_ohlcv_symbol_interval_time').on(table.symbol, table.interval, table.timestamp),
}));

// Options contracts
export const optionsContracts = derivativesSchema.table('options_contracts', {
  id: uuid('id').primaryKey().defaultRandom(),
  contractId: text('contract_id').notNull().unique(),
  underlying: text('underlying').notNull().references(() => assets.symbol),
  optionType: text('option_type').notNull(), // 'call' | 'put'
  optionStyle: text('option_style').notNull().default('european'), // 'european' | 'american'
  strike: decimal('strike', { precision: 18, scale: 8 }).notNull(),
  expiryDate: date('expiry_date').notNull(),
  multiplier: decimal('multiplier', { precision: 10, scale: 4 }).notNull().default('1'),
  isActive: boolean('is_active').notNull().default(true),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  underlyingIdx: index('idx_options_underlying').on(table.underlying),
  expiryIdx: index('idx_options_expiry').on(table.expiryDate),
}));

// Options prices
export const optionsPrices = derivativesSchema.table('options_prices', {
  id: uuid('id').primaryKey().defaultRandom(),
  contractId: text('contract_id').notNull().references(() => optionsContracts.contractId),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  spotPrice: decimal('spot_price', { precision: 18, scale: 8 }).notNull(),
  price: decimal('price', { precision: 18, scale: 8 }).notNull(),
  impliedVolatility: decimal('implied_volatility', { precision: 10, scale: 6 }).notNull(),
  delta: decimal('delta', { precision: 12, scale: 8 }).notNull(),
  gamma: decimal('gamma', { precision: 12, scale: 8 }).notNull(),
  theta: decimal('theta', { precision: 12, scale: 8 }).notNull(),
  vega: decimal('vega', { precision: 12, scale: 8 }).notNull(),
  rho: decimal('rho', { precision: 12, scale: 8 }).notNull(),
  vanna: decimal('vanna', { precision: 12, scale: 8 }),
  volga: decimal('volga', { precision: 12, scale: 8 }),
  charm: decimal('charm', { precision: 12, scale: 8 }),
  intrinsicValue: decimal('intrinsic_value', { precision: 18, scale: 8 }).notNull(),
  timeValue: decimal('time_value', { precision: 18, scale: 8 }).notNull(),
  probabilityItm: decimal('probability_itm', { precision: 8, scale: 6 }),
  pricingModel: text('pricing_model').notNull().default('black_scholes'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  contractTimeIdx: index('idx_options_prices_contract_time').on(table.contractId, table.timestamp),
}));

// Futures contracts
export const futuresContracts = derivativesSchema.table('futures_contracts', {
  id: uuid('id').primaryKey().defaultRandom(),
  contractId: text('contract_id').notNull().unique(),
  underlying: text('underlying').notNull().references(() => assets.symbol),
  expiryDate: date('expiry_date').notNull(),
  contractSize: decimal('contract_size', { precision: 18, scale: 8 }).notNull().default('1'),
  tickValue: decimal('tick_value', { precision: 18, scale: 8 }).notNull(),
  isActive: boolean('is_active').notNull().default(true),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  underlyingIdx: index('idx_futures_underlying').on(table.underlying),
  expiryIdx: index('idx_futures_expiry').on(table.expiryDate),
}));

// Futures prices
export const futuresPrices = derivativesSchema.table('futures_prices', {
  id: uuid('id').primaryKey().defaultRandom(),
  contractId: text('contract_id').notNull().references(() => futuresContracts.contractId),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  spotPrice: decimal('spot_price', { precision: 18, scale: 8 }).notNull(),
  fairValue: decimal('fair_value', { precision: 18, scale: 8 }).notNull(),
  basis: decimal('basis', { precision: 18, scale: 8 }).notNull(),
  basisPct: decimal('basis_pct', { precision: 10, scale: 6 }).notNull(),
  impliedRepoRate: decimal('implied_repo_rate', { precision: 10, scale: 6 }).notNull(),
  timeToExpiryYears: decimal('time_to_expiry_years', { precision: 10, scale: 6 }).notNull(),
  riskFreeRate: decimal('risk_free_rate', { precision: 10, scale: 6 }).notNull(),
  convenienceYield: decimal('convenience_yield', { precision: 10, scale: 6 }),
  storageCost: decimal('storage_cost', { precision: 10, scale: 6 }),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  contractTimeIdx: index('idx_futures_prices_contract_time').on(table.contractId, table.timestamp),
}));

// Structured products
export const structuredProducts = derivativesSchema.table('structured_products', {
  id: uuid('id').primaryKey().defaultRandom(),
  productId: text('product_id').notNull().unique(),
  productName: text('product_name').notNull(),
  productType: text('product_type').notNull(), // 'straddle' | 'strangle' | 'butterfly' | 'iron_condor' | 'calendar_spread' | 'custom'
  underlying: text('underlying').notNull().references(() => assets.symbol),
  expiryDate: date('expiry_date').notNull(),
  legs: jsonb('legs').notNull(),
  isActive: boolean('is_active').notNull().default(true),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  underlyingIdx: index('idx_structured_underlying').on(table.underlying),
  typeIdx: index('idx_structured_type').on(table.productType),
}));

// Structured product prices
export const structuredProductPrices = derivativesSchema.table('structured_product_prices', {
  id: uuid('id').primaryKey().defaultRandom(),
  productId: text('product_id').notNull().references(() => structuredProducts.productId),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  spotPrice: decimal('spot_price', { precision: 18, scale: 8 }).notNull(),
  totalPrice: decimal('total_price', { precision: 18, scale: 8 }).notNull(),
  netDelta: decimal('net_delta', { precision: 12, scale: 8 }).notNull(),
  netGamma: decimal('net_gamma', { precision: 12, scale: 8 }).notNull(),
  netTheta: decimal('net_theta', { precision: 12, scale: 8 }).notNull(),
  netVega: decimal('net_vega', { precision: 12, scale: 8 }).notNull(),
  maxProfit: decimal('max_profit', { precision: 18, scale: 8 }),
  maxLoss: decimal('max_loss', { precision: 18, scale: 8 }),
  breakevens: jsonb('breakevens'),
  payoffDiagram: jsonb('payoff_diagram'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  productTimeIdx: index('idx_structured_prices_product_time').on(table.productId, table.timestamp),
}));

// Derivatives positions
export const derivativesPositions = derivativesSchema.table('positions', {
  id: uuid('id').primaryKey().defaultRandom(),
  positionId: text('position_id').notNull().unique(),
  accountId: text('account_id').notNull().default('default'),
  instrumentType: text('instrument_type').notNull(), // 'option' | 'future' | 'structured_product'
  instrumentId: text('instrument_id').notNull(),
  underlying: text('underlying').notNull().references(() => assets.symbol),
  side: text('side').notNull(), // 'long' | 'short'
  quantity: decimal('quantity', { precision: 18, scale: 8 }).notNull(),
  entryPrice: decimal('entry_price', { precision: 18, scale: 8 }).notNull(),
  currentPrice: decimal('current_price', { precision: 18, scale: 8 }),
  unrealizedPnl: decimal('unrealized_pnl', { precision: 18, scale: 8 }).default('0'),
  realizedPnl: decimal('realized_pnl', { precision: 18, scale: 8 }).default('0'),
  delta: decimal('delta', { precision: 12, scale: 8 }),
  gamma: decimal('gamma', { precision: 12, scale: 8 }),
  theta: decimal('theta', { precision: 12, scale: 8 }),
  vega: decimal('vega', { precision: 12, scale: 8 }),
  status: text('status').notNull().default('open'), // 'open' | 'closed' | 'expired'
  openedAt: timestamp('opened_at', { withTimezone: true }).notNull().defaultNow(),
  closedAt: timestamp('closed_at', { withTimezone: true }),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  accountIdx: index('idx_positions_account').on(table.accountId),
  underlyingIdx: index('idx_positions_underlying').on(table.underlying),
  statusIdx: index('idx_positions_status').on(table.status),
  instrumentIdx: index('idx_positions_instrument').on(table.instrumentType, table.instrumentId),
}));

// Derivatives orders
export const derivativesOrders = derivativesSchema.table('orders', {
  id: uuid('id').primaryKey().defaultRandom(),
  orderId: text('order_id').notNull().unique(),
  accountId: text('account_id').notNull().default('default'),
  instrumentType: text('instrument_type').notNull(),
  instrumentId: text('instrument_id').notNull(),
  underlying: text('underlying').notNull().references(() => assets.symbol),
  side: text('side').notNull(), // 'buy' | 'sell'
  orderType: text('order_type').notNull(), // 'market' | 'limit' | 'stop' | 'stop_limit'
  quantity: decimal('quantity', { precision: 18, scale: 8 }).notNull(),
  price: decimal('price', { precision: 18, scale: 8 }),
  stopPrice: decimal('stop_price', { precision: 18, scale: 8 }),
  filledQuantity: decimal('filled_quantity', { precision: 18, scale: 8 }).notNull().default('0'),
  averageFillPrice: decimal('average_fill_price', { precision: 18, scale: 8 }),
  status: text('status').notNull().default('pending'), // 'pending' | 'open' | 'partial' | 'filled' | 'cancelled' | 'rejected' | 'expired'
  timeInForce: text('time_in_force').notNull().default('GTC'), // 'GTC' | 'DAY' | 'IOC' | 'FOK'
  strategyId: text('strategy_id'),
  correlationId: text('correlation_id'),
  rejectionReason: text('rejection_reason'),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  accountIdx: index('idx_deriv_orders_account').on(table.accountId),
  statusIdx: index('idx_deriv_orders_status').on(table.status),
  instrumentIdx: index('idx_deriv_orders_instrument').on(table.instrumentType, table.instrumentId),
}));

// Derivatives fills
export const derivativesFills = derivativesSchema.table('fills', {
  id: uuid('id').primaryKey().defaultRandom(),
  fillId: text('fill_id').notNull().unique(),
  orderId: text('order_id').notNull().references(() => derivativesOrders.orderId),
  instrumentType: text('instrument_type').notNull(),
  instrumentId: text('instrument_id').notNull(),
  side: text('side').notNull(),
  quantity: decimal('quantity', { precision: 18, scale: 8 }).notNull(),
  price: decimal('price', { precision: 18, scale: 8 }).notNull(),
  commission: decimal('commission', { precision: 18, scale: 8 }).notNull().default('0'),
  slippage: decimal('slippage', { precision: 18, scale: 8 }).notNull().default('0'),
  executedAt: timestamp('executed_at', { withTimezone: true }).notNull(),
  venue: text('venue'),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  orderIdx: index('idx_deriv_fills_order').on(table.orderId),
  executedIdx: index('idx_deriv_fills_executed').on(table.executedAt),
}));

// Backtests
export const backtests = derivativesSchema.table('backtests', {
  id: uuid('id').primaryKey().defaultRandom(),
  backtestId: text('backtest_id').notNull().unique(),
  name: text('name').notNull(),
  strategyType: text('strategy_type').notNull(),
  underlying: text('underlying').notNull().references(() => assets.symbol),
  startDate: date('start_date').notNull(),
  endDate: date('end_date').notNull(),
  initialCapital: decimal('initial_capital', { precision: 18, scale: 2 }).notNull().default('100000'),
  slippageBps: decimal('slippage_bps', { precision: 10, scale: 4 }).notNull().default('5'),
  commissionPerContract: decimal('commission_per_contract', { precision: 10, scale: 4 }).notNull().default('1'),
  riskFreeRate: decimal('risk_free_rate', { precision: 10, scale: 6 }).notNull().default('0.05'),
  strategyParams: jsonb('strategy_params').default({}),
  status: text('status').notNull().default('pending'), // 'pending' | 'running' | 'completed' | 'failed'
  progress: decimal('progress', { precision: 5, scale: 2 }).notNull().default('0'),
  startTime: timestamp('start_time', { withTimezone: true }),
  endTime: timestamp('end_time', { withTimezone: true }),
  errorMessage: text('error_message'),
  metadata: jsonb('metadata').default({}),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  underlyingIdx: index('idx_backtests_underlying').on(table.underlying),
  statusIdx: index('idx_backtests_status').on(table.status),
  strategyIdx: index('idx_backtests_strategy').on(table.strategyType),
}));

// Backtest results
export const backtestResults = derivativesSchema.table('backtest_results', {
  id: uuid('id').primaryKey().defaultRandom(),
  backtestId: text('backtest_id').notNull().unique().references(() => backtests.backtestId),
  startEquity: decimal('start_equity', { precision: 18, scale: 2 }).notNull(),
  endEquity: decimal('end_equity', { precision: 18, scale: 2 }).notNull(),
  totalReturn: decimal('total_return', { precision: 12, scale: 6 }).notNull(),
  annualizedReturn: decimal('annualized_return', { precision: 12, scale: 6 }).notNull(),
  sharpeRatio: decimal('sharpe_ratio', { precision: 10, scale: 4 }),
  sortinoRatio: decimal('sortino_ratio', { precision: 10, scale: 4 }),
  maxDrawdown: decimal('max_drawdown', { precision: 10, scale: 6 }).notNull(),
  maxDrawdownDuration: integer('max_drawdown_duration'),
  calmarRatio: decimal('calmar_ratio', { precision: 10, scale: 4 }),
  winRate: decimal('win_rate', { precision: 8, scale: 6 }),
  profitFactor: decimal('profit_factor', { precision: 10, scale: 4 }),
  totalTrades: integer('total_trades').notNull().default(0),
  winningTrades: integer('winning_trades').notNull().default(0),
  losingTrades: integer('losing_trades').notNull().default(0),
  avgTradeDurationHours: decimal('avg_trade_duration_hours', { precision: 10, scale: 2 }),
  totalCommission: decimal('total_commission', { precision: 18, scale: 2 }).notNull().default('0'),
  totalSlippage: decimal('total_slippage', { precision: 18, scale: 2 }).notNull().default('0'),
  avgNetDelta: decimal('avg_net_delta', { precision: 12, scale: 8 }),
  maxNetDelta: decimal('max_net_delta', { precision: 12, scale: 8 }),
  avgNetGamma: decimal('avg_net_gamma', { precision: 12, scale: 8 }),
  avgNetVega: decimal('avg_net_vega', { precision: 12, scale: 8 }),
  avgNetTheta: decimal('avg_net_theta', { precision: 12, scale: 8 }),
  equityCurve: jsonb('equity_curve'),
  trades: jsonb('trades'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
});

// Strategy templates
export const strategyTemplates = derivativesSchema.table('strategy_templates', {
  id: uuid('id').primaryKey().defaultRandom(),
  templateId: text('template_id').notNull().unique(),
  strategyType: text('strategy_type').notNull(),
  name: text('name').notNull(),
  description: text('description'),
  parameters: jsonb('parameters').notNull(),
  riskLevel: text('risk_level'), // 'low' | 'medium' | 'high'
  marketOutlook: text('market_outlook'), // 'bullish' | 'bearish' | 'neutral' | 'volatile'
  suitableFor: jsonb('suitable_for'),
  isActive: boolean('is_active').notNull().default(true),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
});

// Volatility surfaces
export const volatilitySurfaces = derivativesSchema.table('volatility_surfaces', {
  id: uuid('id').primaryKey().defaultRandom(),
  underlying: text('underlying').notNull().references(() => assets.symbol),
  timestamp: timestamp('timestamp', { withTimezone: true }).notNull(),
  atmVol: decimal('atm_vol', { precision: 10, scale: 6 }).notNull(),
  skew: decimal('skew', { precision: 10, scale: 6 }),
  kurtosis: decimal('kurtosis', { precision: 10, scale: 6 }),
  termStructure: jsonb('term_structure'),
  surfaceData: jsonb('surface_data'),
  model: text('model').notNull().default('svi'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => ({
  underlyingTimeIdx: uniqueIndex('idx_vol_surfaces_underlying_time').on(table.underlying, table.timestamp),
}));

// ============================================================================
// RELATIONS
// ============================================================================

export const ordersRelations = relations(orders, ({ many }) => ({
  fills: many(fills),
}));

export const fillsRelations = relations(fills, ({ one }) => ({
  order: one(orders, {
    fields: [fills.orderId],
    references: [orders.id],
  }),
}));

export const optionsContractsRelations = relations(optionsContracts, ({ one, many }) => ({
  asset: one(assets, {
    fields: [optionsContracts.underlying],
    references: [assets.symbol],
  }),
  prices: many(optionsPrices),
}));

export const optionsPricesRelations = relations(optionsPrices, ({ one }) => ({
  contract: one(optionsContracts, {
    fields: [optionsPrices.contractId],
    references: [optionsContracts.contractId],
  }),
}));

export const futuresContractsRelations = relations(futuresContracts, ({ one, many }) => ({
  asset: one(assets, {
    fields: [futuresContracts.underlying],
    references: [assets.symbol],
  }),
  prices: many(futuresPrices),
}));

export const futuresPricesRelations = relations(futuresPrices, ({ one }) => ({
  contract: one(futuresContracts, {
    fields: [futuresPrices.contractId],
    references: [futuresContracts.contractId],
  }),
}));

export const structuredProductsRelations = relations(structuredProducts, ({ one, many }) => ({
  asset: one(assets, {
    fields: [structuredProducts.underlying],
    references: [assets.symbol],
  }),
  prices: many(structuredProductPrices),
}));

export const structuredProductPricesRelations = relations(structuredProductPrices, ({ one }) => ({
  product: one(structuredProducts, {
    fields: [structuredProductPrices.productId],
    references: [structuredProducts.productId],
  }),
}));

export const backtestsRelations = relations(backtests, ({ one }) => ({
  asset: one(assets, {
    fields: [backtests.underlying],
    references: [assets.symbol],
  }),
  result: one(backtestResults),
}));

export const backtestResultsRelations = relations(backtestResults, ({ one }) => ({
  backtest: one(backtests, {
    fields: [backtestResults.backtestId],
    references: [backtests.backtestId],
  }),
}));
