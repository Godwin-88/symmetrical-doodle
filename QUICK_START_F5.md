# Quick Start: F5 (Portfolio & Risk Management)

## Running the Application

```bash
cd frontend
npm run dev
```

Navigate to the Portfolio (F5) tab.

---

## F5: PORTFOLIO & RISK MANAGEMENT - Quick Actions

### 1. View Portfolios
**Left Panel**: Portfolio Registry
- See all portfolios with status, mode, and metrics
- View summary stats (Total portfolios, Active, Total AUM, Breaches)
- Color-coded status (ACTIVE/PAUSED/CLOSED) and mode (LIVE/PAPER/SIMULATED)

### 2. Create New Portfolio
**Button**: "+ NEW PORTFOLIO"
- Define portfolio name and base currency
- Set initial capital
- Choose mode (LIVE, PAPER, SIMULATED)
- Configure strategy allocations
- Select capital allocation model
- Set rebalance frequency

### 3. View Positions
**Tab**: "POSITIONS"
- See all current positions
- Symbol, Side (LONG/SHORT), Size, Entry, Current, P&L
- Exposure percentage
- Strategy attribution

### 4. Monitor Exposure
**Tab**: "EXPOSURE"
- Gross vs Net exposure
- Long vs Short distribution
- Exposure by strategy (visual bars)
- Capital allocation vs actual exposure

### 5. Check Risk Limits
**Tab**: "RISK"
- View all risk limits (Hard & Soft)
- Current vs Threshold values
- Utilization percentage
- Breach status
- Risk metrics (Vol, VaR, CVaR, Drawdown, Sharpe, Sortino)

### 6. Analyze Attribution
**Tab**: "ATTRIBUTION"
- P&L by strategy
- P&L by asset
- Contribution percentages
- ROI per strategy
- Unrealized vs Realized breakdown

### 7. Configure Risk Limits
**Button**: "CONFIGURE RISK LIMITS"
- Create new risk limits
- Set thresholds (Hard or Soft)
- Define actions (ALERT, BLOCK, REDUCE, HALT)
- Categories: POSITION, LEVERAGE, SECTOR, CORRELATION, LOSS, EXPOSURE

### 8. Run Stress Tests
**Button**: "STRESS TESTING"
- Select scenario (2008 Crisis, COVID Crash, Volatility Spike, Liquidity Drought)
- View estimated impact (Loss, Max DD, Recovery days)
- Run test to see portfolio behavior

### 9. Rebalance Portfolio
**Button**: "REBALANCE ALLOCATION"
- Adjust strategy weights
- Reallocate capital
- Respect turnover constraints
- Apply allocation model

### 10. Pause/Resume Portfolio
**Button**: "⏸ PAUSE / ▶ RESUME PORTFOLIO"
- Pause active portfolio (stop trading)
- Resume paused portfolio (restart trading)
- Status changes: ACTIVE ↔ PAUSED

---

## Mock Data Available

### 2 Portfolios

1. **PORT-001: MAIN TRADING PORTFOLIO** (PAPER, ACTIVE)
   - Capital: $104,127.89 (Initial: $100,000)
   - P&L: +$4,127.89 (+4.13% ROI)
   - Strategies: 4 (Regime Switching, Momentum, Mean Reversion, Volatility Arb)
   - Allocation Model: VOL_TARGET
   - Rebalance: WEEKLY

2. **PORT-002: RESEARCH PORTFOLIO** (SIMULATED, ACTIVE)
   - Capital: $51,234.56 (Initial: $50,000)
   - P&L: +$1,234.56 (+2.47% ROI)
   - Strategies: 2 (Trend Following, Pairs Trading)
   - Allocation Model: EQUAL_WEIGHT
   - Rebalance: MONTHLY

### 4 Positions
1. **EURUSD**: LONG 100,000 @ 1.0845 → 1.0894 (+$490 unrealized, +$1,234.56 realized)
2. **GBPUSD**: LONG 75,000 @ 1.2650 → 1.2698 (+$360 unrealized, +$890.12 realized)
3. **USDJPY**: SHORT 50,000 @ 148.50 → 148.20 (+$150 unrealized, -$234.78 realized)
4. **AUDUSD**: LONG 60,000 @ 0.6580 → 0.6612 (+$192 unrealized, +$456.89 realized)

### 5 Risk Limits
1. **MAX POSITION SIZE** (HARD): 15.0% threshold, 10.5% current, BLOCK action
2. **MAX LEVERAGE** (HARD): 3.0x threshold, 2.1x current, BLOCK action
3. **MAX DAILY LOSS** (HARD): 5.0% threshold, 1.2% current, HALT action
4. **NET EXPOSURE WARNING** (SOFT): 60.0% threshold, 33.7% current, ALERT action
5. **MAX CORRELATION** (HARD): 0.85 threshold, 0.72 current, REDUCE action

### 4 Stress Scenarios
1. **2008 FINANCIAL CRISIS** (HISTORICAL): -$18,500 loss, 17.8% max DD, 120 days recovery
2. **COVID-19 CRASH** (HISTORICAL): -$22,300 loss, 21.4% max DD, 95 days recovery
3. **VOLATILITY SPIKE +50%** (HYPOTHETICAL): -$8,900 loss, 8.5% max DD, 30 days recovery
4. **LIQUIDITY DROUGHT** (HYPOTHETICAL): -$12,400 loss, 11.9% max DD, 60 days recovery

---

## Institutional-Grade Features

### Portfolio Control Plane ✅
- **Multiple Portfolios**: Research, hedge, execution
- **Mode Selection**: LIVE (real money), PAPER (simulated), SIMULATED (backtest)
- **Status Management**: ACTIVE, PAUSED, CLOSED
- **Capital Tracking**: Initial vs current capital

### Capital Allocation Models ✅
- **EQUAL_WEIGHT**: Equal distribution
- **VOL_TARGET**: Volatility-based
- **RISK_PARITY**: Risk-balanced
- **MAX_DIVERSIFICATION**: Maximum diversification
- **KELLY**: Kelly criterion (bounded)
- **CUSTOM**: User-defined

### Risk Controls ✅
- **Hard Limits**: BLOCK or HALT actions
- **Soft Limits**: ALERT or REDUCE actions
- **Categories**: Position, Leverage, Sector, Correlation, Loss, Exposure
- **Breach Detection**: Real-time monitoring
- **Visual Alerts**: Red banners for breaches

### Exposure Management ✅
- **Gross Exposure**: Total absolute exposure
- **Net Exposure**: Long - Short
- **Leverage**: Gross / Equity
- **Long vs Short**: Position distribution
- **By Strategy**: Exposure breakdown
- **By Asset**: Asset-level exposure

### Risk Metrics ✅
- **Volatility**: 30-day rolling (12.5%)
- **VaR (95%, 1D)**: Value at Risk ($2,500)
- **CVaR (95%, 1D)**: Conditional VaR ($3,200)
- **Max Drawdown**: Historical maximum (5.8%)
- **Sharpe Ratio**: Risk-adjusted return (1.42)
- **Sortino Ratio**: Downside risk-adjusted (1.87)

### Stress Testing ✅
- **Historical Scenarios**: 2008 Crisis, COVID Crash
- **Hypothetical Scenarios**: Volatility Spike, Liquidity Drought
- **Impact Metrics**: Loss, Max DD, Recovery days
- **Run Tests**: Execute scenarios on portfolio

### Performance Attribution ✅
- **By Strategy**: P&L contribution per strategy
- **By Asset**: P&L contribution per asset
- **Contribution %**: Percentage of total P&L
- **ROI per Strategy**: Return on allocated capital

---

## Configuration Options

### Portfolio Modes
- **LIVE**: Real money trading (red indicator)
- **PAPER**: Simulated trading with real data (yellow indicator)
- **SIMULATED**: Backtesting mode (gray indicator)

### Portfolio Status
- **ACTIVE**: Trading enabled (green indicator)
- **PAUSED**: Trading suspended (yellow indicator)
- **CLOSED**: Portfolio closed (gray indicator)

### Allocation Models
- **EQUAL_WEIGHT**: Simple equal distribution
- **VOL_TARGET**: Target volatility level
- **RISK_PARITY**: Equal risk contribution
- **MAX_DIVERSIFICATION**: Maximize diversification ratio
- **KELLY**: Kelly criterion allocation
- **CUSTOM**: User-defined algorithm

### Rebalance Frequencies
- **DAILY**: Intraday rebalancing
- **WEEKLY**: Weekly rebalancing
- **MONTHLY**: Monthly rebalancing
- **QUARTERLY**: Quarterly rebalancing

### Risk Limit Types
- **HARD**: Strict enforcement (BLOCK or HALT)
- **SOFT**: Warning only (ALERT or REDUCE)

### Risk Limit Categories
- **POSITION**: Max position size
- **LEVERAGE**: Max leverage
- **SECTOR**: Max sector exposure
- **CORRELATION**: Max correlation
- **LOSS**: Max daily/weekly loss
- **EXPOSURE**: Max net/gross exposure

### Risk Limit Actions
- **ALERT**: Send notification
- **BLOCK**: Prevent new trades
- **REDUCE**: Reduce positions
- **HALT**: Stop all trading

---

## All Buttons Work Offline

F5 has automatic mock data fallback:
- ✅ No backend required
- ✅ All CRUD operations functional
- ✅ All 12 buttons clickable
- ✅ Realistic portfolio data
- ✅ Complete risk metrics
- ✅ Stress test scenarios

---

## Build & Deploy

```bash
# Development
npm run dev

# Production build
npm run build

# Output: 393.54 KB (gzipped: 93.21 KB)
```

---

## Documentation

- `F5_PORTFOLIO_COMPLETE.md` - Full implementation details
- `ALL_FEATURES_SUMMARY.md` - Complete feature summary
- `STRATEGIES_COMPLETE.md` - Trading strategies

---

## Support

All 12 buttons are functional with mock data. The system works completely offline and is ready for backend integration with the Rust portfolio manager and risk engine.

**Key Principle**: Risk management is enforced at the portfolio layer, not the strategy layer. This ensures strategies remain interchangeable while risk remains centralized and controlled.
