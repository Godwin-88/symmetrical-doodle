# Quick Start: F7 (Simulation & Backtesting)

## Running the Application

```bash
cd frontend
npm run dev
```

Navigate to the Simulation (F7) tab.

---

## F7: SIMULATION & BACKTESTING - Quick Actions

### 1. View Experiments
**Left Panel**: Experiment Registry
- See all experiments with status, metrics, and tags
- Filter by status (DRAFT, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
- Filter by tags (PRODUCTION_CANDIDATE, TESTING, EXPERIMENTAL, STRESS_TEST, REJECTED)
- View summary stats (Total, Avg Sharpe, Running, Completed)

### 2. Create New Experiment
**Button**: "+ NEW EXPERIMENT"
- Configure strategy and parameters
- Set market & data configuration
- Define execution model
- Configure portfolio construction
- Set risk management rules

### 3. Run Experiment
**Button**: "▶ RUN EXPERIMENT" (when experiment selected)
- Queue experiment for execution
- Status changes: DRAFT → QUEUED → RUNNING
- Real-time progress updates
- Automatic result calculation

### 4. View Results
**Button**: "📊 VIEW FULL RESULTS" (when completed)
- Performance metrics (Sharpe, Sortino, Max DD)
- Trade statistics (Win rate, Profit factor)
- Risk metrics (CVaR, Tail risk)
- Attribution analysis (by asset, by regime)
- Equity curve visualization
- Drawdown curve visualization

### 5. Clone Experiment
**Button**: "⎘ CLONE EXPERIMENT"
- Duplicate existing experiment
- Modify parameters for variations
- Run parameter sweeps
- A/B testing

### 6. Compare Experiments
**Checkboxes**: Select 2-4 experiments
**Button**: "COMPARE (N)"
- Side-by-side metric comparison
- Equity curve overlays
- Risk profile comparison
- Parameter sensitivity analysis

### 7. Scenario Testing
**Button**: "SCENARIO TESTS"
- Run stress tests (2008 Crisis, COVID Crash)
- Volatility shock simulations
- Liquidity drought scenarios
- Execution failure tests

### 8. Parameter Sweep
**Button**: "PARAMETER SWEEP"
- Batch run experiments
- Grid search parameters
- Walk-forward optimization
- Monte Carlo resampling

---

## Mock Data Available

### 4 Experiments
1. **EXP-001: REGIME DETECTION V2.1** (COMPLETED)
   - Sharpe: 1.45, Max DD: 8.2%, Win Rate: 58%
   - 1,247 trades, $18,500 profit
   - Tags: PRODUCTION_CANDIDATE, REGIME_BASED

2. **EXP-002: MOMENTUM ALPHA BACKTEST** (RUNNING)
   - Sharpe: 1.32, Max DD: 12.5%, Win Rate: 54%
   - 892 trades, 67% complete
   - Tags: TESTING, MOMENTUM

3. **EXP-003: MULTI-REGIME STRATEGY** (QUEUED)
   - Awaiting execution
   - Tags: EXPERIMENTAL

4. **EXP-004: CRISIS SCENARIO TEST** (FAILED)
   - Sharpe: -0.45, Max DD: 25.3%
   - Failed during COVID crash period
   - Tags: STRESS_TEST, REJECTED

### 4 Scenario Tests
1. 2008 Financial Crisis (STRESS)
2. COVID-19 Crash (STRESS)
3. Flash Crash (LIQUIDITY_DROUGHT)
4. Volatility Spike (VOLATILITY_SHOCK)

---

## Institutional-Grade Features

### Bias Controls ✅
- **Survivorship Bias Toggle**: Prevent look-ahead bias
- **Look-Ahead Bias Guard**: No future data leakage
- **Visual Warnings**: Red flags for enabled biases

### Execution Realism ✅
- **Order Types**: MARKET, LIMIT, VWAP, TWAP
- **Slippage Models**: FIXED, VOLUME_BASED, IMPACT_CURVE
- **Transaction Costs**: Configurable in basis points
- **Latency Simulation**: 50-200ms delays

### Risk-First Metrics ✅
- **Sharpe & Sortino**: Risk-adjusted returns
- **CVaR 95%**: Tail risk measurement
- **Max Drawdown Duration**: Recovery time
- **Profit Factor**: Win/loss ratio
- **Turnover**: Transaction cost impact

### Reproducibility ✅
- **Immutable IDs**: EXP-001, EXP-002, etc.
- **Git Commit Tracking**: Version control
- **Hypothesis Field**: Research intent
- **Timestamp Tracking**: Full audit trail

### Governance ✅
- **Status Progression**: DRAFT → QUEUED → RUNNING → COMPLETED
- **Tag System**: Workflow management
- **Researcher Attribution**: Who ran what
- **Approval Gates**: Review workflow

---

## Configuration Sections

### Market & Data
- Asset universe: EURUSD, GBPUSD, USDJPY, etc.
- Data source: BLOOMBERG_TICK, REFINITIV_TICK
- Date range: 2020-01-01 to 2024-01-01
- Frequency: 1m, 5m, 15m, 1h, 4h, 1d
- Bias controls: Survivorship, Look-ahead

### Execution Model
- Order types: MARKET, LIMIT, VWAP, TWAP
- Slippage: FIXED (2 bps), VOLUME_BASED (2.5 bps), IMPACT_CURVE (3 bps)
- Transaction costs: 1-2 bps
- Latency: 50-200ms

### Portfolio Construction
- Position sizing: FIXED, VOL_TARGET, KELLY
- Max leverage: 1.0x - 2.0x
- Max position: 20-30%
- Long/short ratio: 1.0

### Risk Management
- Stop loss: 2-5%
- Take profit: 4-10%
- Max drawdown: 10-20%
- Daily loss limit: 3-10%

---

## Performance Metrics

### Standard Metrics
- Total Return (%)
- CAGR (%)
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown (%)
- Profit Factor
- Win Rate (%)
- Turnover (x)

### Risk Metrics
- Volatility (%)
- CVaR 95% (%)
- Tail Risk (%)
- Max DD Duration (days)

### Trade Statistics
- Total Trades
- Winning / Losing Trades
- Avg Win / Loss ($)
- Avg Holding Period (hours)

### Attribution
- P&L by Asset
- P&L by Regime
- Transaction Costs

---

## All Buttons Work Offline

F7 has automatic mock data fallback:
- ✅ No backend required
- ✅ All CRUD operations functional
- ✅ All 15 buttons clickable
- ✅ Realistic experiment data
- ✅ Complete performance metrics

---

## Build & Deploy

```bash
# Development
npm run dev

# Production build
npm run build

# Output: 375.73 KB (gzipped: 90.26 KB)
```

---

## Documentation

- `F7_SIMULATION_COMPLETE.md` - Full implementation details
- `F3_VS_F8_COMPARISON.md` - Intelligence vs Data & Models
- `STRATEGIES_COMPLETE.md` - Trading strategies
- `ML_MODELS_COMPLETE.md` - Model registry

---

## Support

All 15 buttons are functional with mock data. The system works completely offline and is ready for backend integration with the Rust simulation engine.
