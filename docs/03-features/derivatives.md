# Derivatives Trading Module

The Derivatives Trading Module provides comprehensive derivatives pricing, structuring, and backtesting capabilities for the algorithmic trading system.

## Overview

This module enables:
- **Options Pricing**: Black-Scholes and Binomial Tree models
- **Futures Pricing**: Cost-of-carry model with convenience yield
- **Structured Products**: Straddles, Strangles, Butterflies, Iron Condors, Calendar Spreads
- **Backtesting**: Strategy backtesting with realistic slippage and commission modeling
- **Greeks Calculation**: Full suite including second-order Greeks (vanna, volga, charm)
- **Real-time Market Data**: Multi-provider aggregation with fallback

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (React/TypeScript)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Assets    │  │   Options   │  │  Structured │  │  Backtest  │ │
│  │   Browser   │  │    Chain    │  │   Products  │  │   Runner   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────────────┐
│                    Intelligence Layer (Python)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Market Data     │  │ Derivatives     │  │ Backtesting         │ │
│  │ Aggregator      │  │ Pricing Engine  │  │ Engine              │ │
│  │ - Yahoo Finance │  │ - Black-Scholes │  │ - Portfolio Sim     │ │
│  │ - Alpha Vantage │  │ - Binomial Tree │  │ - Greeks Tracking   │ │
│  │ - Binance       │  │ - Greeks Calc   │  │ - P&L Attribution   │ │
│  │ - Polygon       │  │ - IV Surface    │  │ - Risk Metrics      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    Execution Core (Rust)                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Derivatives     │  │ Risk Limits     │  │ Position            │ │
│  │ Types           │  │ (Greeks-based)  │  │ Management          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                         Database Layer                               │
│  ┌─────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ PostgreSQL (Supabase)   │  │ Neo4j (GraphRAG)                │  │
│  │ - Contracts             │  │ - Asset relationships          │  │
│  │ - Prices                │  │ - Strategy performance         │  │
│  │ - Positions             │  │ - Regime correlations          │  │
│  │ - Backtests             │  │ - Knowledge graph              │  │
│  └─────────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Supported Assets

| Symbol | Name | Asset Class | Margin Req |
|--------|------|-------------|------------|
| XAUUSD | Gold | Commodity | 5% |
| XAGUSD | Silver | Commodity | 5% |
| EURUSD | Euro/USD | Forex | 3.33% |
| GBPUSD | GBP/USD | Forex | 3.33% |
| USDJPY | USD/JPY | Forex | 3.33% |
| AUDUSD | AUD/USD | Forex | 3.33% |
| USDCHF | USD/CHF | Forex | 3.33% |
| BTCUSD | Bitcoin | Crypto | 50% |
| ETHUSD | Ethereum | Crypto | 50% |

## Options Pricing

### Black-Scholes Model

The system implements the Black-Scholes model for European options:

```python
from intelligence_layer.derivatives_pricing import BlackScholes

bs = BlackScholes(
    spot=2045.50,      # Current spot price
    strike=2100.00,    # Strike price
    time_to_expiry=0.25,  # Years to expiry
    volatility=0.15,   # Annualized volatility
    risk_free_rate=0.05,  # Risk-free rate
    dividend_yield=0.0    # Dividend yield
)

# Get option price
call_price = bs.price('call')
put_price = bs.price('put')

# Get all Greeks
greeks = bs.greeks('call')
# Returns: delta, gamma, theta, vega, rho, vanna, volga, charm
```

### Binomial Tree Model

For American options with early exercise:

```python
from intelligence_layer.derivatives_pricing import BinomialTree

tree = BinomialTree(
    spot=2045.50,
    strike=2100.00,
    time_to_expiry=0.25,
    volatility=0.15,
    risk_free_rate=0.05,
    steps=100  # Number of tree steps
)

american_put_price = tree.price('put')
```

### Implied Volatility Calculation

```python
from intelligence_layer.derivatives_pricing import BlackScholes

# Given market price, solve for implied volatility
iv = BlackScholes.implied_volatility(
    market_price=45.50,
    spot=2045.50,
    strike=2100.00,
    time_to_expiry=0.25,
    option_type='call',
    risk_free_rate=0.05
)
```

## Greeks

### First-Order Greeks

| Greek | Symbol | Measures |
|-------|--------|----------|
| Delta | Δ | Price sensitivity to underlying |
| Gamma | Γ | Delta sensitivity to underlying |
| Theta | Θ | Time decay |
| Vega | ν | Volatility sensitivity |
| Rho | ρ | Interest rate sensitivity |

### Second-Order Greeks

| Greek | Symbol | Measures |
|-------|--------|----------|
| Vanna | - | Delta sensitivity to volatility |
| Volga | - | Vega sensitivity to volatility |
| Charm | - | Delta decay over time |

## Structured Products

### Straddle

Buy ATM call and put for volatility exposure:

```python
from intelligence_layer.derivatives_pricing import DerivativesPricingEngine

engine = DerivativesPricingEngine()
straddle = engine.price_straddle(
    underlying='XAUUSD',
    spot_price=2045.50,
    strike=2045.50,  # ATM
    expiry_date='2024-03-29',
    volatility=0.15
)
# Returns: total_price, net_greeks, max_profit, max_loss, breakevens
```

### Iron Condor

Sell OTM put spread and call spread:

```python
iron_condor = engine.price_iron_condor(
    underlying='BTCUSD',
    spot_price=43000,
    put_lower=38000,
    put_upper=40000,
    call_lower=46000,
    call_upper=48000,
    expiry_date='2024-03-29',
    volatility=0.65
)
```

### Butterfly

Buy low/high strikes, sell 2x middle:

```python
butterfly = engine.price_butterfly(
    underlying='EURUSD',
    spot_price=1.0892,
    lower_strike=1.07,
    middle_strike=1.09,
    upper_strike=1.11,
    expiry_date='2024-03-29',
    volatility=0.08
)
```

## Backtesting

### Configuration

```python
from intelligence_layer.derivatives_backtesting import BacktestConfig, BacktestEngine

config = BacktestConfig(
    name='Gold Covered Call 2023',
    strategy_type='covered_call',
    underlying='XAUUSD',
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    slippage_bps=5,
    commission_per_contract=1.0,
    risk_free_rate=0.05,
    strategy_params={
        'delta_target': 0.30,
        'roll_days': 7
    }
)

engine = BacktestEngine(config)
result = engine.run()
```

### Performance Metrics

The backtest results include:

- **Total Return**: Overall strategy return
- **Annualized Return**: Annualized rate of return
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss

### Greeks Statistics

- **Avg Net Delta**: Average portfolio delta exposure
- **Max Net Delta**: Maximum delta exposure reached
- **Avg Net Gamma**: Average gamma exposure
- **Avg Net Vega**: Average volatility exposure
- **Avg Net Theta**: Average time decay

## API Endpoints

### Market Data

```
GET  /api/v1/derivatives/assets              # List all assets
GET  /api/v1/derivatives/assets/{class}      # List by asset class
POST /api/v1/derivatives/tick                # Get current tick
POST /api/v1/derivatives/ticks               # Get multiple ticks
POST /api/v1/derivatives/ohlcv               # Get historical OHLCV
```

### Options

```
POST /api/v1/derivatives/options/price       # Price an option
POST /api/v1/derivatives/options/implied-vol # Calculate IV
GET  /api/v1/derivatives/options/chain       # Get options chain
```

### Futures

```
POST /api/v1/derivatives/futures/price       # Price a futures contract
```

### Structured Products

```
POST /api/v1/derivatives/structured/price    # Price a structured product
GET  /api/v1/derivatives/structured/templates # Get product templates
```

### Backtesting

```
POST /api/v1/derivatives/backtest/run        # Start a backtest
GET  /api/v1/derivatives/backtest/{id}       # Get backtest status
GET  /api/v1/derivatives/backtest            # List all backtests
DELETE /api/v1/derivatives/backtest/{id}     # Delete a backtest
GET  /api/v1/derivatives/strategies/templates # Get strategy templates
```

## Database Schema

### PostgreSQL Tables

The derivatives schema includes:

- `derivatives.assets` - Supported trading assets
- `derivatives.market_ticks` - Real-time tick data
- `derivatives.ohlcv` - Historical OHLCV data
- `derivatives.options_contracts` - Option contract definitions
- `derivatives.options_prices` - Option price snapshots with Greeks
- `derivatives.futures_contracts` - Futures contract definitions
- `derivatives.futures_prices` - Futures price snapshots
- `derivatives.structured_products` - Structured product definitions
- `derivatives.positions` - Current positions
- `derivatives.orders` - Order history
- `derivatives.fills` - Fill history
- `derivatives.backtests` - Backtest configurations
- `derivatives.backtest_results` - Backtest results
- `derivatives.strategy_templates` - Strategy templates
- `derivatives.volatility_surfaces` - IV surface snapshots

### Neo4j Nodes

The graph database includes:

- `OptionContract` - Option contract nodes
- `FuturesContract` - Futures contract nodes
- `StructuredProduct` - Structured product nodes
- `StrategyTemplate` - Strategy template nodes
- `Backtest` - Backtest result nodes
- `VolatilitySurface` - IV surface nodes
- `GreekSnapshot` - Portfolio Greeks snapshots
- `TradingConcept` - Educational concept nodes (for RAG)
- `Document` - Document nodes (for RAG)

### Key Relationships

- `HAS_UNDERLYING` - Derivatives to underlying asset
- `CONTAINS_LEG` - Structured products to component options
- `USES_STRATEGY` - Backtests to strategy templates
- `SUITABLE_FOR_REGIME` - Strategies to market regimes
- `CORRELATED` - Asset correlations
- `EXPLAINS` - Documents to concepts (RAG)

## UI Components

### Derivatives Tab (F2 > MKTS > Derivatives)

The derivatives UI provides:

1. **Asset Browser**: Browse available assets by class
2. **Options Chain**: View and price options chains
3. **Futures Calculator**: Price futures contracts
4. **Structured Products**: Build and price complex strategies
5. **Backtest Runner**: Configure and run backtests
6. **Results Viewer**: Analyze backtest performance

### Component Layout

Following the List-View-Edit pattern:
- **Left Panel**: Asset/contract selection
- **Center Panel**: Pricing details and charts
- **Right Panel**: Order entry and Greeks display

## Risk Management

### Greeks-Based Risk Limits

The Rust execution core enforces Greeks-based risk limits:

```rust
pub struct DerivativeRiskLimits {
    pub max_delta: f64,      // Maximum absolute delta
    pub max_gamma: f64,      // Maximum gamma exposure
    pub max_vega: f64,       // Maximum vega exposure
    pub max_theta: f64,      // Maximum theta (time decay)
    pub max_position_value: f64,
    pub max_notional: f64,
}
```

### Pre-Trade Checks

Before executing derivative orders:
1. Calculate position Greeks impact
2. Check against portfolio limits
3. Validate margin requirements
4. Verify contract specifications

## Market Data Providers

The system aggregates data from multiple sources with automatic fallback:

1. **Yahoo Finance** (Primary for most assets)
2. **Alpha Vantage** (Backup for forex/crypto)
3. **Binance** (Crypto-specific)
4. **Polygon** (US equities backup)

Provider priority and fallback are automatic.

## Configuration

### Environment Variables

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key

# Database
DATABASE_URL=postgresql://...
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

# Feature Flags
DERIVATIVES_ENABLED=true
BACKTESTING_ENABLED=true
```

## Examples

### Price a Gold Call Option

```bash
curl -X POST http://localhost:8000/api/v1/derivatives/options/price \
  -H "Content-Type: application/json" \
  -d '{
    "underlying": "XAUUSD",
    "spot_price": 2045.50,
    "strike": 2100,
    "expiry_date": "2024-03-29",
    "option_type": "call",
    "volatility": 0.15
  }'
```

### Run a Backtest

```bash
curl -X POST http://localhost:8000/api/v1/derivatives/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Gold Covered Call Test",
    "strategy_type": "covered_call",
    "underlying": "XAUUSD",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000
  }'
```

---

**Last Updated**: February 2026
**Version**: 1.0.0
