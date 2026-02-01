# Database Schema Reference

This document describes the complete database schema for the algorithmic trading system, including PostgreSQL (Supabase) and Neo4j graph database.

## PostgreSQL (Supabase)

### Schemas

The database is organized into four schemas:

| Schema | Purpose |
|--------|---------|
| `intelligence` | Market embeddings and AI-related data |
| `execution` | Orders, fills, and positions |
| `simulation` | Backtesting experiments and market data |
| `derivatives` | Derivatives contracts, pricing, and positions |

### Drizzle ORM

The system uses Drizzle ORM for TypeScript type safety and migrations.

**Setup:**

```bash
cd database
npm install
npm run db:generate  # Generate migrations
npm run db:push      # Push to database
npm run db:studio    # Open Drizzle Studio
```

**Configuration:**

```typescript
// database/drizzle/drizzle.config.ts
export default {
  schema: './schema.ts',
  out: './migrations',
  driver: 'pg',
  dbCredentials: {
    connectionString: process.env.DATABASE_URL,
  },
};
```

### Intelligence Schema

#### market_state_embeddings

Stores market state vector embeddings for similarity search.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| timestamp | TIMESTAMPTZ | Embedding timestamp |
| asset_id | TEXT | Asset identifier |
| regime_id | TEXT | Market regime ID |
| embedding | VECTOR(256) | State embedding vector |
| volatility | REAL | Volatility metric |
| liquidity | REAL | Liquidity metric |
| horizon | TEXT | Time horizon (default: '1h') |
| source_model | TEXT | Model that generated embedding |
| metadata | JSONB | Additional metadata |

#### strategy_state_embeddings

Stores strategy state embeddings for performance tracking.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| timestamp | TIMESTAMPTZ | State timestamp |
| strategy_id | TEXT | Strategy identifier |
| embedding | VECTOR(128) | State embedding |
| pnl_state | REAL | P&L state |
| drawdown | REAL | Current drawdown |
| exposure | REAL | Market exposure |

### Execution Schema

#### orders

Stores all trading orders.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| asset_id | TEXT | Asset identifier |
| side | TEXT | 'buy' or 'sell' |
| quantity | DECIMAL(18,8) | Order quantity |
| order_type | TEXT | 'market', 'limit', 'stop' |
| price | DECIMAL(18,8) | Limit price (if applicable) |
| status | TEXT | Order status |
| metadata | JSONB | Additional data |

#### fills

Stores order execution fills.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| order_id | UUID | Reference to orders |
| asset_id | TEXT | Asset identifier |
| side | TEXT | Fill side |
| quantity | DECIMAL(18,8) | Filled quantity |
| price | DECIMAL(18,8) | Fill price |
| commission | DECIMAL(18,8) | Commission paid |
| timestamp | TIMESTAMPTZ | Fill timestamp |

#### positions

Stores current positions.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| asset_id | TEXT | Asset (unique) |
| quantity | DECIMAL(18,8) | Position size |
| average_price | DECIMAL(18,8) | Average entry price |
| unrealized_pnl | DECIMAL(18,8) | Unrealized P&L |
| realized_pnl | DECIMAL(18,8) | Realized P&L |

### Derivatives Schema

#### assets

Supported trading assets for derivatives.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| symbol | TEXT | Unique symbol |
| name | TEXT | Asset name |
| asset_class | TEXT | forex/crypto/commodity/equity |
| base_currency | TEXT | Base currency |
| quote_currency | TEXT | Quote currency |
| margin_requirement | DECIMAL(8,4) | Margin requirement % |
| tick_size | DECIMAL(18,8) | Minimum price increment |
| is_active | BOOLEAN | Trading enabled |

#### options_contracts

Options contract definitions.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| contract_id | TEXT | Unique contract ID |
| underlying | TEXT | Underlying asset |
| option_type | TEXT | 'call' or 'put' |
| option_style | TEXT | 'european' or 'american' |
| strike | DECIMAL(18,8) | Strike price |
| expiry_date | DATE | Expiration date |
| multiplier | DECIMAL(10,4) | Contract multiplier |
| is_active | BOOLEAN | Contract tradeable |

#### options_prices

Options pricing snapshots with Greeks.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| contract_id | TEXT | Reference to options_contracts |
| timestamp | TIMESTAMPTZ | Price timestamp |
| spot_price | DECIMAL(18,8) | Underlying spot |
| price | DECIMAL(18,8) | Option price |
| implied_volatility | DECIMAL(10,6) | Implied vol |
| delta | DECIMAL(12,8) | Delta |
| gamma | DECIMAL(12,8) | Gamma |
| theta | DECIMAL(12,8) | Theta |
| vega | DECIMAL(12,8) | Vega |
| rho | DECIMAL(12,8) | Rho |
| vanna | DECIMAL(12,8) | Vanna |
| volga | DECIMAL(12,8) | Volga |
| charm | DECIMAL(12,8) | Charm |
| intrinsic_value | DECIMAL(18,8) | Intrinsic value |
| time_value | DECIMAL(18,8) | Time value |
| pricing_model | TEXT | Model used |

#### structured_products

Structured product definitions.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| product_id | TEXT | Unique product ID |
| product_name | TEXT | Display name |
| product_type | TEXT | straddle/strangle/etc |
| underlying | TEXT | Underlying asset |
| expiry_date | DATE | Expiration date |
| legs | JSONB | Leg definitions |
| is_active | BOOLEAN | Product tradeable |

#### backtests

Backtest configurations and status.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| backtest_id | TEXT | Unique backtest ID |
| name | TEXT | Backtest name |
| strategy_type | TEXT | Strategy type |
| underlying | TEXT | Underlying asset |
| start_date | DATE | Backtest start |
| end_date | DATE | Backtest end |
| initial_capital | DECIMAL(18,2) | Starting capital |
| slippage_bps | DECIMAL(10,4) | Slippage in bps |
| commission_per_contract | DECIMAL(10,4) | Commission |
| status | TEXT | pending/running/completed/failed |
| progress | DECIMAL(5,2) | Progress % |
| strategy_params | JSONB | Strategy parameters |

#### backtest_results

Backtest performance results.

| Column | Type | Description |
|--------|------|-------------|
| backtest_id | TEXT | Reference to backtests |
| total_return | DECIMAL(12,6) | Total return |
| annualized_return | DECIMAL(12,6) | Annualized return |
| sharpe_ratio | DECIMAL(10,4) | Sharpe ratio |
| sortino_ratio | DECIMAL(10,4) | Sortino ratio |
| max_drawdown | DECIMAL(10,6) | Maximum drawdown |
| win_rate | DECIMAL(8,6) | Win rate |
| profit_factor | DECIMAL(10,4) | Profit factor |
| total_trades | INTEGER | Number of trades |
| equity_curve | JSONB | Equity curve data |
| trades | JSONB | Trade details |

### Indexes

Key indexes for performance:

```sql
-- Market data
CREATE INDEX idx_market_ticks_symbol_time ON derivatives.market_ticks (symbol, timestamp DESC);
CREATE INDEX idx_ohlcv_symbol_interval_time ON derivatives.ohlcv (symbol, interval, timestamp DESC);

-- Options
CREATE INDEX idx_options_underlying ON derivatives.options_contracts (underlying);
CREATE INDEX idx_options_expiry ON derivatives.options_contracts (expiry_date);

-- Positions
CREATE INDEX idx_positions_account ON derivatives.positions (account_id);
CREATE INDEX idx_positions_status ON derivatives.positions (status);

-- Backtests
CREATE INDEX idx_backtests_status ON derivatives.backtests (status);
```

## Neo4j Graph Database

### Node Types

#### Asset Nodes

```cypher
(:Asset {
    asset_id: 'XAUUSD',
    asset_class: 'Commodity',
    venue: 'Deriv',
    name: 'Gold',
    base_currency: 'XAU',
    quote_currency: 'USD',
    margin_requirement: 5.0
})
```

#### OptionContract Nodes

```cypher
(:OptionContract {
    contract_id: 'XAUUSD_C_2400_2024Q1',
    underlying: 'XAUUSD',
    option_type: 'call',
    option_style: 'european',
    strike: 2400.0,
    expiry_date: date('2024-03-29'),
    multiplier: 1.0
})
```

#### StructuredProduct Nodes

```cypher
(:StructuredProduct {
    product_id: 'BTCUSD_IC_2024Q1',
    product_name: 'Bitcoin Iron Condor',
    product_type: 'iron_condor',
    underlying: 'BTCUSD'
})
```

#### StrategyTemplate Nodes

```cypher
(:StrategyTemplate {
    template_id: 'covered_call',
    strategy_type: 'covered_call',
    name: 'Covered Call Writing',
    risk_level: 'low',
    market_outlook: 'neutral'
})
```

#### MarketRegime Nodes

```cypher
(:MarketRegime {
    regime_id: 'low_vol_trending',
    volatility_level: 'low',
    trend_state: 'trending',
    liquidity_state: 'normal'
})
```

#### TradingConcept Nodes (GraphRAG)

```cypher
(:TradingConcept {
    concept_id: 'delta',
    name: 'Delta',
    category: 'greeks',
    description: 'Rate of change of option price...',
    formula: 'dV/dS'
})
```

### Relationships

#### Core Relationships

| Relationship | From | To | Properties |
|--------------|------|----|-----------|
| `HAS_UNDERLYING` | OptionContract/FuturesContract | Asset | - |
| `CONTAINS_LEG` | StructuredProduct | OptionContract | quantity, leg_type |
| `CORRELATED` | Asset | Asset | window, strength, sign |
| `TRANSITIONS_TO` | MarketRegime | MarketRegime | probability, avg_duration |

#### Strategy Relationships

| Relationship | From | To | Properties |
|--------------|------|----|-----------|
| `SUITABLE_FOR_REGIME` | StrategyTemplate | MarketRegime | confidence, expected_sharpe |
| `USES_STRATEGY` | Backtest | StrategyTemplate | - |
| `BACKTESTED_ON` | Backtest | Asset | - |
| `PERFORMS_IN` | Strategy | MarketRegime | sharpe, max_dd, sample_size |

#### GraphRAG Relationships

| Relationship | From | To | Description |
|--------------|------|----|-----------|
| `EXPLAINS` | Document | TradingConcept | Document explains concept |
| `DERIVATIVE_OF` | TradingConcept | TradingConcept | Mathematical derivative |
| `MEASURES_SENSITIVITY_TO` | TradingConcept | TradingConcept | Sensitivity relationship |
| `AFFECTED_BY` | TradingConcept | TradingConcept | Causal relationship |

### Constraints

```cypher
CREATE CONSTRAINT asset_id_unique FOR (a:Asset) REQUIRE a.asset_id IS UNIQUE;
CREATE CONSTRAINT option_contract_id_unique FOR (o:OptionContract) REQUIRE o.contract_id IS UNIQUE;
CREATE CONSTRAINT strategy_template_id_unique FOR (st:StrategyTemplate) REQUIRE st.template_id IS UNIQUE;
```

### Example Queries

#### Find Options for an Underlying

```cypher
MATCH (o:OptionContract)-[:HAS_UNDERLYING]->(a:Asset {asset_id: 'XAUUSD'})
RETURN o.contract_id, o.option_type, o.strike, o.expiry_date
ORDER BY o.strike
```

#### Find Strategies for Current Regime

```cypher
MATCH (st:StrategyTemplate)-[r:SUITABLE_FOR_REGIME]->(regime:MarketRegime {regime_id: 'low_vol_trending'})
RETURN st.name, st.strategy_type, r.confidence, r.expected_sharpe
ORDER BY r.expected_sharpe DESC
```

#### Find Correlated Assets

```cypher
MATCH (a:Asset {asset_id: 'BTCUSD'})-[c:CORRELATED]-(related:Asset)
WHERE c.strength > 0.5
RETURN related.asset_id, related.name, c.strength, c.sign
ORDER BY c.strength DESC
```

#### GraphRAG: Find Relevant Documents

```cypher
MATCH (d:Document)-[:EXPLAINS]->(c:TradingConcept)
WHERE c.category = 'greeks'
RETURN d.title, collect(c.name) as concepts
```

## Migration Scripts

### Running PostgreSQL Migrations

```bash
# Initialize database
psql $DATABASE_URL -f database/init/01-init.sql
psql $DATABASE_URL -f database/init/03-derivatives.sql

# Or use Drizzle
cd database && npm run db:push
```

### Running Neo4j Scripts

```bash
# Using cypher-shell
cypher-shell -a $NEO4J_URI -u $NEO4J_USERNAME -p $NEO4J_PASSWORD \
  < database/init/02-neo4j-init.cypher
cypher-shell -a $NEO4J_URI -u $NEO4J_USERNAME -p $NEO4J_PASSWORD \
  < database/init/04-neo4j-derivatives.cypher
```

---

**Last Updated**: February 2026
