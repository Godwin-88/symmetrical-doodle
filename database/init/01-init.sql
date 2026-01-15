-- Initialize trading system database with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS intelligence;
CREATE SCHEMA IF NOT EXISTS execution;
CREATE SCHEMA IF NOT EXISTS simulation;

-- Market state embeddings table
CREATE TABLE intelligence.market_state_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id TEXT NOT NULL,
    regime_id TEXT,
    embedding VECTOR(256) NOT NULL,
    volatility REAL,
    liquidity REAL,
    horizon TEXT NOT NULL DEFAULT '1h',
    source_model TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Strategy state embeddings table
CREATE TABLE intelligence.strategy_state_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    embedding VECTOR(128) NOT NULL,
    pnl_state REAL,
    drawdown REAL,
    exposure REAL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Regime trajectory embeddings table
CREATE TABLE intelligence.regime_trajectory_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    embedding VECTOR(128) NOT NULL,
    realized_vol REAL,
    transition_path JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create vector indexes for similarity search
CREATE INDEX ON intelligence.market_state_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

CREATE INDEX ON intelligence.strategy_state_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

CREATE INDEX ON intelligence.regime_trajectory_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Time-based indexes
CREATE INDEX idx_market_embeddings_timestamp ON intelligence.market_state_embeddings (timestamp);
CREATE INDEX idx_market_embeddings_asset ON intelligence.market_state_embeddings (asset_id, timestamp);
CREATE INDEX idx_strategy_embeddings_timestamp ON intelligence.strategy_state_embeddings (timestamp);
CREATE INDEX idx_regime_embeddings_time_range ON intelligence.regime_trajectory_embeddings (start_time, end_time);

-- Execution schema tables
CREATE TABLE execution.orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(18, 8) NOT NULL,
    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit', 'stop')),
    price DECIMAL(18, 8),
    status TEXT NOT NULL DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE execution.fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES execution.orders(id),
    asset_id TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    commission DECIMAL(18, 8) DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE execution.positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id TEXT NOT NULL UNIQUE,
    quantity DECIMAL(18, 8) NOT NULL DEFAULT 0,
    average_price DECIMAL(18, 8) NOT NULL DEFAULT 0,
    unrealized_pnl DECIMAL(18, 8) DEFAULT 0,
    realized_pnl DECIMAL(18, 8) DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Simulation schema tables
CREATE TABLE simulation.experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE simulation.market_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id TEXT NOT NULL,
    open DECIMAL(18, 8) NOT NULL,
    high DECIMAL(18, 8) NOT NULL,
    low DECIMAL(18, 8) NOT NULL,
    close DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 8) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_orders_asset_time ON execution.orders (asset_id, created_at);
CREATE INDEX idx_fills_order_id ON execution.fills (order_id);
CREATE INDEX idx_fills_timestamp ON execution.fills (timestamp);
CREATE INDEX idx_market_data_asset_time ON simulation.market_data (asset_id, timestamp);
CREATE INDEX idx_experiments_status ON simulation.experiments (status);

-- Insert some initial data
INSERT INTO execution.positions (asset_id, quantity, average_price) VALUES
('EURUSD', 0, 0),
('GBPUSD', 0, 0),
('USDJPY', 0, 0);

-- Create a function to update position timestamps
CREATE OR REPLACE FUNCTION update_position_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for position updates
CREATE TRIGGER update_position_timestamp_trigger
    BEFORE UPDATE ON execution.positions
    FOR EACH ROW
    EXECUTE FUNCTION update_position_timestamp();