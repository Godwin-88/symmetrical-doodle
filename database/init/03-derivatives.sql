-- Derivatives Trading Database Schema Extension
-- For Supabase PostgreSQL with pgvector support

-- Create derivatives schema
CREATE SCHEMA IF NOT EXISTS derivatives;

-- ============================================================================
-- ASSETS & MARKET DATA
-- ============================================================================

-- Supported assets for derivatives trading
CREATE TABLE derivatives.assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    asset_class TEXT NOT NULL CHECK (asset_class IN ('forex', 'crypto', 'commodity', 'equity', 'index')),
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    margin_requirement DECIMAL(8, 4) NOT NULL DEFAULT 5.0,
    min_trade_size DECIMAL(18, 8) NOT NULL DEFAULT 0.01,
    max_trade_size DECIMAL(18, 8) NOT NULL DEFAULT 1000000,
    tick_size DECIMAL(18, 8) NOT NULL DEFAULT 0.0001,
    is_active BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Real-time market ticks
CREATE TABLE derivatives.market_ticks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    bid DECIMAL(18, 8) NOT NULL,
    ask DECIMAL(18, 8) NOT NULL,
    last DECIMAL(18, 8) NOT NULL,
    mid DECIMAL(18, 8) NOT NULL,
    spread_bps DECIMAL(10, 4) NOT NULL,
    volume DECIMAL(18, 8),
    provider TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- OHLCV historical data
CREATE TABLE derivatives.ohlcv (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    interval TEXT NOT NULL CHECK (interval IN ('1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M')),
    open DECIMAL(18, 8) NOT NULL,
    high DECIMAL(18, 8) NOT NULL,
    low DECIMAL(18, 8) NOT NULL,
    close DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 8) NOT NULL,
    provider TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, interval)
);

-- ============================================================================
-- OPTIONS CONTRACTS
-- ============================================================================

-- Options contracts
CREATE TABLE derivatives.options_contracts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id TEXT NOT NULL UNIQUE,
    underlying TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    option_type TEXT NOT NULL CHECK (option_type IN ('call', 'put')),
    option_style TEXT NOT NULL CHECK (option_style IN ('european', 'american')) DEFAULT 'european',
    strike DECIMAL(18, 8) NOT NULL,
    expiry_date DATE NOT NULL,
    multiplier DECIMAL(10, 4) NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Options pricing snapshots
CREATE TABLE derivatives.options_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id TEXT NOT NULL REFERENCES derivatives.options_contracts(contract_id),
    timestamp TIMESTAMPTZ NOT NULL,
    spot_price DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    implied_volatility DECIMAL(10, 6) NOT NULL,
    -- Greeks
    delta DECIMAL(12, 8) NOT NULL,
    gamma DECIMAL(12, 8) NOT NULL,
    theta DECIMAL(12, 8) NOT NULL,
    vega DECIMAL(12, 8) NOT NULL,
    rho DECIMAL(12, 8) NOT NULL,
    vanna DECIMAL(12, 8),
    volga DECIMAL(12, 8),
    charm DECIMAL(12, 8),
    -- Additional metrics
    intrinsic_value DECIMAL(18, 8) NOT NULL,
    time_value DECIMAL(18, 8) NOT NULL,
    probability_itm DECIMAL(8, 6),
    pricing_model TEXT NOT NULL DEFAULT 'black_scholes',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- FUTURES CONTRACTS
-- ============================================================================

-- Futures contracts
CREATE TABLE derivatives.futures_contracts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id TEXT NOT NULL UNIQUE,
    underlying TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    expiry_date DATE NOT NULL,
    contract_size DECIMAL(18, 8) NOT NULL DEFAULT 1,
    tick_value DECIMAL(18, 8) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Futures pricing snapshots
CREATE TABLE derivatives.futures_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id TEXT NOT NULL REFERENCES derivatives.futures_contracts(contract_id),
    timestamp TIMESTAMPTZ NOT NULL,
    spot_price DECIMAL(18, 8) NOT NULL,
    fair_value DECIMAL(18, 8) NOT NULL,
    basis DECIMAL(18, 8) NOT NULL,
    basis_pct DECIMAL(10, 6) NOT NULL,
    implied_repo_rate DECIMAL(10, 6) NOT NULL,
    time_to_expiry_years DECIMAL(10, 6) NOT NULL,
    risk_free_rate DECIMAL(10, 6) NOT NULL,
    convenience_yield DECIMAL(10, 6),
    storage_cost DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- STRUCTURED PRODUCTS
-- ============================================================================

-- Structured products definitions
CREATE TABLE derivatives.structured_products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id TEXT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    product_type TEXT NOT NULL CHECK (product_type IN ('straddle', 'strangle', 'butterfly', 'iron_condor', 'calendar_spread', 'custom')),
    underlying TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    expiry_date DATE NOT NULL,
    legs JSONB NOT NULL, -- Array of leg definitions
    is_active BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Structured products pricing
CREATE TABLE derivatives.structured_product_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id TEXT NOT NULL REFERENCES derivatives.structured_products(product_id),
    timestamp TIMESTAMPTZ NOT NULL,
    spot_price DECIMAL(18, 8) NOT NULL,
    total_price DECIMAL(18, 8) NOT NULL,
    -- Net Greeks
    net_delta DECIMAL(12, 8) NOT NULL,
    net_gamma DECIMAL(12, 8) NOT NULL,
    net_theta DECIMAL(12, 8) NOT NULL,
    net_vega DECIMAL(12, 8) NOT NULL,
    -- P&L boundaries
    max_profit DECIMAL(18, 8),
    max_loss DECIMAL(18, 8),
    breakevens JSONB, -- Array of breakeven prices
    payoff_diagram JSONB, -- {spots: [], payoffs: []}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- DERIVATIVES POSITIONS
-- ============================================================================

-- Derivatives positions
CREATE TABLE derivatives.positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id TEXT NOT NULL UNIQUE,
    account_id TEXT NOT NULL DEFAULT 'default',
    instrument_type TEXT NOT NULL CHECK (instrument_type IN ('option', 'future', 'structured_product')),
    instrument_id TEXT NOT NULL, -- References contract_id or product_id
    underlying TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    side TEXT NOT NULL CHECK (side IN ('long', 'short')),
    quantity DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 8) NOT NULL,
    current_price DECIMAL(18, 8),
    unrealized_pnl DECIMAL(18, 8) DEFAULT 0,
    realized_pnl DECIMAL(18, 8) DEFAULT 0,
    -- Position Greeks (for options/structured)
    delta DECIMAL(12, 8),
    gamma DECIMAL(12, 8),
    theta DECIMAL(12, 8),
    vega DECIMAL(12, 8),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed', 'expired')) DEFAULT 'open',
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- DERIVATIVES ORDERS
-- ============================================================================

-- Derivatives orders
CREATE TABLE derivatives.orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id TEXT NOT NULL UNIQUE,
    account_id TEXT NOT NULL DEFAULT 'default',
    instrument_type TEXT NOT NULL CHECK (instrument_type IN ('option', 'future', 'structured_product')),
    instrument_id TEXT NOT NULL,
    underlying TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8), -- For limit orders
    stop_price DECIMAL(18, 8), -- For stop orders
    filled_quantity DECIMAL(18, 8) NOT NULL DEFAULT 0,
    average_fill_price DECIMAL(18, 8),
    status TEXT NOT NULL CHECK (status IN ('pending', 'open', 'partial', 'filled', 'cancelled', 'rejected', 'expired')) DEFAULT 'pending',
    time_in_force TEXT NOT NULL CHECK (time_in_force IN ('GTC', 'DAY', 'IOC', 'FOK')) DEFAULT 'GTC',
    strategy_id TEXT,
    correlation_id TEXT,
    rejection_reason TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Derivatives fills
CREATE TABLE derivatives.fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fill_id TEXT NOT NULL UNIQUE,
    order_id TEXT NOT NULL REFERENCES derivatives.orders(order_id),
    instrument_type TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    commission DECIMAL(18, 8) NOT NULL DEFAULT 0,
    slippage DECIMAL(18, 8) NOT NULL DEFAULT 0,
    executed_at TIMESTAMPTZ NOT NULL,
    venue TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- BACKTESTING
-- ============================================================================

-- Backtest configurations
CREATE TABLE derivatives.backtests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    underlying TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(18, 2) NOT NULL DEFAULT 100000,
    slippage_bps DECIMAL(10, 4) NOT NULL DEFAULT 5,
    commission_per_contract DECIMAL(10, 4) NOT NULL DEFAULT 1,
    risk_free_rate DECIMAL(10, 6) NOT NULL DEFAULT 0.05,
    strategy_params JSONB DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')) DEFAULT 'pending',
    progress DECIMAL(5, 2) NOT NULL DEFAULT 0,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Backtest results
CREATE TABLE derivatives.backtest_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id TEXT NOT NULL UNIQUE REFERENCES derivatives.backtests(backtest_id),
    -- Performance metrics
    start_equity DECIMAL(18, 2) NOT NULL,
    end_equity DECIMAL(18, 2) NOT NULL,
    total_return DECIMAL(12, 6) NOT NULL,
    annualized_return DECIMAL(12, 6) NOT NULL,
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 6) NOT NULL,
    max_drawdown_duration INTEGER, -- in hours
    calmar_ratio DECIMAL(10, 4),
    win_rate DECIMAL(8, 6),
    profit_factor DECIMAL(10, 4),
    -- Trade statistics
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    avg_trade_duration_hours DECIMAL(10, 2),
    total_commission DECIMAL(18, 2) NOT NULL DEFAULT 0,
    total_slippage DECIMAL(18, 2) NOT NULL DEFAULT 0,
    -- Greeks statistics
    avg_net_delta DECIMAL(12, 8),
    max_net_delta DECIMAL(12, 8),
    avg_net_gamma DECIMAL(12, 8),
    avg_net_vega DECIMAL(12, 8),
    avg_net_theta DECIMAL(12, 8),
    -- Equity curve and trades stored as JSONB
    equity_curve JSONB, -- Array of {timestamp, equity}
    trades JSONB, -- Array of trade details
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- STRATEGY TEMPLATES
-- ============================================================================

-- Strategy templates for derivatives trading
CREATE TABLE derivatives.strategy_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id TEXT NOT NULL UNIQUE,
    strategy_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    parameters JSONB NOT NULL, -- Schema of parameters with types and defaults
    risk_level TEXT CHECK (risk_level IN ('low', 'medium', 'high')),
    market_outlook TEXT CHECK (market_outlook IN ('bullish', 'bearish', 'neutral', 'volatile')),
    suitable_for JSONB, -- Array of suitable asset classes/conditions
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- VOLATILITY SURFACES
-- ============================================================================

-- Implied volatility surfaces
CREATE TABLE derivatives.volatility_surfaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    underlying TEXT NOT NULL REFERENCES derivatives.assets(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    atm_vol DECIMAL(10, 6) NOT NULL,
    skew DECIMAL(10, 6), -- 25 delta risk reversal
    kurtosis DECIMAL(10, 6), -- 25 delta butterfly
    term_structure JSONB, -- {expiries: [], vols: []}
    surface_data JSONB, -- Full surface grid
    model TEXT NOT NULL DEFAULT 'svi',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(underlying, timestamp)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Market data indexes
CREATE INDEX idx_market_ticks_symbol_time ON derivatives.market_ticks (symbol, timestamp DESC);
CREATE INDEX idx_ohlcv_symbol_interval_time ON derivatives.ohlcv (symbol, interval, timestamp DESC);

-- Options indexes
CREATE INDEX idx_options_underlying ON derivatives.options_contracts (underlying);
CREATE INDEX idx_options_expiry ON derivatives.options_contracts (expiry_date);
CREATE INDEX idx_options_prices_contract_time ON derivatives.options_prices (contract_id, timestamp DESC);

-- Futures indexes
CREATE INDEX idx_futures_underlying ON derivatives.futures_contracts (underlying);
CREATE INDEX idx_futures_expiry ON derivatives.futures_contracts (expiry_date);
CREATE INDEX idx_futures_prices_contract_time ON derivatives.futures_prices (contract_id, timestamp DESC);

-- Structured products indexes
CREATE INDEX idx_structured_underlying ON derivatives.structured_products (underlying);
CREATE INDEX idx_structured_type ON derivatives.structured_products (product_type);
CREATE INDEX idx_structured_prices_product_time ON derivatives.structured_product_prices (product_id, timestamp DESC);

-- Positions indexes
CREATE INDEX idx_positions_account ON derivatives.positions (account_id);
CREATE INDEX idx_positions_underlying ON derivatives.positions (underlying);
CREATE INDEX idx_positions_status ON derivatives.positions (status);
CREATE INDEX idx_positions_instrument ON derivatives.positions (instrument_type, instrument_id);

-- Orders indexes
CREATE INDEX idx_orders_account ON derivatives.orders (account_id);
CREATE INDEX idx_orders_status ON derivatives.orders (status);
CREATE INDEX idx_orders_instrument ON derivatives.orders (instrument_type, instrument_id);

-- Fills indexes
CREATE INDEX idx_fills_order ON derivatives.fills (order_id);
CREATE INDEX idx_fills_executed ON derivatives.fills (executed_at DESC);

-- Backtests indexes
CREATE INDEX idx_backtests_underlying ON derivatives.backtests (underlying);
CREATE INDEX idx_backtests_status ON derivatives.backtests (status);
CREATE INDEX idx_backtests_strategy ON derivatives.backtests (strategy_type);

-- Volatility surfaces indexes
CREATE INDEX idx_vol_surfaces_underlying_time ON derivatives.volatility_surfaces (underlying, timestamp DESC);

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION derivatives.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers
CREATE TRIGGER update_assets_timestamp BEFORE UPDATE ON derivatives.assets
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

CREATE TRIGGER update_options_contracts_timestamp BEFORE UPDATE ON derivatives.options_contracts
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

CREATE TRIGGER update_futures_contracts_timestamp BEFORE UPDATE ON derivatives.futures_contracts
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

CREATE TRIGGER update_structured_products_timestamp BEFORE UPDATE ON derivatives.structured_products
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

CREATE TRIGGER update_positions_timestamp BEFORE UPDATE ON derivatives.positions
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

CREATE TRIGGER update_orders_timestamp BEFORE UPDATE ON derivatives.orders
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

CREATE TRIGGER update_backtests_timestamp BEFORE UPDATE ON derivatives.backtests
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

CREATE TRIGGER update_strategy_templates_timestamp BEFORE UPDATE ON derivatives.strategy_templates
    FOR EACH ROW EXECUTE FUNCTION derivatives.update_updated_at();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert supported assets
INSERT INTO derivatives.assets (symbol, name, asset_class, base_currency, quote_currency, margin_requirement) VALUES
('XAUUSD', 'Gold', 'commodity', 'XAU', 'USD', 5.0),
('XAGUSD', 'Silver', 'commodity', 'XAG', 'USD', 5.0),
('EURUSD', 'Euro/US Dollar', 'forex', 'EUR', 'USD', 3.33),
('GBPUSD', 'British Pound/US Dollar', 'forex', 'GBP', 'USD', 3.33),
('USDJPY', 'US Dollar/Japanese Yen', 'forex', 'USD', 'JPY', 3.33),
('AUDUSD', 'Australian Dollar/US Dollar', 'forex', 'AUD', 'USD', 3.33),
('USDCHF', 'US Dollar/Swiss Franc', 'forex', 'USD', 'CHF', 3.33),
('BTCUSD', 'Bitcoin/US Dollar', 'crypto', 'BTC', 'USD', 50.0),
('ETHUSD', 'Ethereum/US Dollar', 'crypto', 'ETH', 'USD', 50.0);

-- Insert strategy templates
INSERT INTO derivatives.strategy_templates (template_id, strategy_type, name, description, parameters, risk_level, market_outlook) VALUES
('covered_call', 'covered_call', 'Covered Call', 'Sell call options against long underlying position',
 '{"delta_target": {"type": "number", "default": 0.3, "description": "Target delta for short calls"}, "roll_days": {"type": "integer", "default": 7, "description": "Days before expiry to roll"}}',
 'low', 'neutral'),
('iron_condor', 'iron_condor', 'Iron Condor', 'Sell OTM put spread and OTM call spread',
 '{"wing_width": {"type": "number", "default": 0.05, "description": "Width of spreads as % of spot"}, "delta_target": {"type": "number", "default": 0.15, "description": "Target delta for short strikes"}}',
 'medium', 'neutral'),
('straddle', 'straddle', 'Long Straddle', 'Buy ATM call and put for volatility exposure',
 '{"strike_offset": {"type": "number", "default": 0, "description": "Offset from ATM strike"}}',
 'high', 'volatile'),
('butterfly', 'butterfly', 'Butterfly Spread', 'Buy low/high strikes, sell 2x middle strike',
 '{"wing_width": {"type": "number", "default": 0.03, "description": "Width of wings as % of spot"}}',
 'medium', 'neutral'),
('calendar_spread', 'calendar_spread', 'Calendar Spread', 'Sell near-term, buy far-term same strike',
 '{"front_expiry_days": {"type": "integer", "default": 30, "description": "Days to front expiry"}, "back_expiry_days": {"type": "integer", "default": 60, "description": "Days to back expiry"}}',
 'medium', 'neutral');

-- Grant permissions (adjust as needed for your Supabase setup)
-- GRANT ALL ON SCHEMA derivatives TO authenticated;
-- GRANT ALL ON ALL TABLES IN SCHEMA derivatives TO authenticated;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA derivatives TO authenticated;
