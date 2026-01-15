# Deriv API Integration Guide

## Overview

Your algorithmic trading system now has full integration with Deriv's demo trading platform. This allows you to:

- **Trade with real market data** using a demo account (no real money)
- **Test strategies** in live market conditions
- **Monitor positions** and P&L in real-time
- **Execute orders** through the Deriv API
- **Stream market data** via WebSocket

## Configuration

### Environment Variables

The following variables are configured in `.env`:

```bash
# Deriv API Configuration (Demo Account)
DERIV_APP_ID=118029
DERIV_API_TOKEN=gxF5pHUCgjDTOGI
DERIV_WEBSOCKET_URL=wss://ws.derivws.com/websockets/v3?app_id=118029
DERIV_DEMO_MODE=true
DERIV_MAX_POSITION_SIZE=1.0
DERIV_MAX_DAILY_TRADES=50
DERIV_MAX_DAILY_LOSS=1000.0
```

### Safety Limits

- **Max Position Size**: 1.0 USD per trade
- **Max Daily Trades**: 50 trades per day
- **Max Daily Loss**: 1000 USD per day
- **Demo Mode**: Enabled (no real money at risk)

## Architecture

### Components

1. **Rust Adapter** (`execution-core/src/deriv_adapter.rs`)
   - Low-level WebSocket client
   - High-performance order execution
   - Real-time market data streaming
   - Position tracking

2. **Python Bridge** (`intelligence-layer/src/intelligence_layer/deriv_integration.py`)
   - High-level API interface
   - Strategy integration
   - Risk management
   - Analytics and reporting

3. **Frontend Integration**
   - Real-time position monitoring (F5 - Portfolio)
   - Order placement interface (F6 - Execution)
   - Market data display (F3 - Markets)

### Data Flow

```
Strategy Signal
    ↓
Python Intelligence Layer
    ↓
Risk Checks & Validation
    ↓
Rust Execution Core
    ↓
Deriv WebSocket API
    ↓
Demo Trading Account
```

## Testing the Connection

### Quick Test

Run the connection test script:

```bash
# Windows
python scripts\test-deriv-connection.py

# Linux/Mac
python3 scripts/test-deriv-connection.py
```

This will:
1. Connect to Deriv API
2. Authorize with your API token
3. Display account information
4. List available trading symbols
5. Subscribe to real-time ticks
6. Optionally place a test trade

### Expected Output

```
============================================================
DERIV API CONNECTION TEST
============================================================

Configuration:
  App ID: 118029
  API Token: gxF5pHUCg...
  WebSocket URL: wss://ws.derivws.com/websockets/v3?app_id=118029
  Demo Mode: True
  Max Position Size: 1.0
  Max Daily Trades: 50
  Max Daily Loss: 1000.0

Connecting to Deriv API...
✅ Connected successfully

Account Information:
  Login ID: VRTC12345678
  Balance: 10000.00 USD
  Account Type: DEMO

Fetching available symbols...
✅ Found 150+ available symbols

Forex Symbols (20):
  - frxEURUSD: EUR/USD
  - frxGBPUSD: GBP/USD
  - frxUSDJPY: USD/JPY
  ...

Synthetic Indices (50):
  - R_100: Volatility 100 Index
  - R_50: Volatility 50 Index
  - R_25: Volatility 25 Index
  ...

Testing tick subscription (R_100 - Volatility 100 Index)...
Waiting for ticks...
  Tick 1: Bid=1234.56, Ask=1234.58, Mid=1234.57
  Tick 2: Bid=1234.57, Ask=1234.59, Mid=1234.58
  ...

✅ All tests passed!

============================================================
READY FOR TRADING
============================================================
```

## Available Trading Symbols

### Forex Pairs
- **Major Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD
- **Minor Pairs**: EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY
- **Exotic Pairs**: USD/MXN, USD/ZAR, USD/TRY

### Synthetic Indices
- **Volatility Indices**: R_10, R_25, R_50, R_75, R_100
- **Crash/Boom Indices**: CRASH300, CRASH500, BOOM300, BOOM500
- **Step Indices**: STEP_INDEX

### Commodities
- Gold, Silver, Oil

## Using the Integration

### Python Example

```python
from intelligence_layer.deriv_integration import DerivClient, OrderType

async def trade_example():
    # Create and connect client
    client = DerivClient()
    await client.connect()
    
    # Check account
    account = client.get_account_info()
    print(f"Balance: {account['balance']} {account['currency']}")
    
    # Subscribe to market data
    await client.subscribe_ticks("R_100")
    
    # Place a trade
    contract = await client.buy_contract(
        symbol="R_100",
        contract_type=OrderType.CALL,  # Buy/Up
        amount=0.50,  # 50 cents
        duration=5,   # 5 ticks
        duration_unit="t"
    )
    
    if contract:
        print(f"Trade placed: {contract['contract_id']}")
    
    # Monitor positions
    positions = client.get_positions()
    for pos in positions:
        print(f"Position: {pos['symbol']}, P&L: {pos['profit']}")
    
    # Disconnect
    await client.disconnect()
```

### Rust Example

```rust
use execution_core::deriv_adapter::{DerivAdapter, DerivConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Create adapter
    let config = DerivConfig::default();
    let (adapter, mut events) = DerivAdapter::new(config);
    
    // Start connection
    adapter.start().await?;
    
    // Listen for events
    while let Some(event) = events.recv().await {
        match event {
            DerivEvent::Authorized(account) => {
                println!("Authorized: {} - Balance: {}", 
                    account.loginid, account.balance);
            }
            DerivEvent::TickUpdate(tick) => {
                println!("Tick: {} - Bid: {}, Ask: {}", 
                    tick.symbol, tick.bid, tick.ask);
            }
            DerivEvent::OrderFilled(order) => {
                println!("Order filled: {}", order.id);
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

## Integration with Trading System

### F3 - Markets (Real-time Data)

The Markets tab can now display real-time Deriv market data:

```typescript
// Subscribe to Deriv ticks
await derivClient.subscribe_ticks("frxEURUSD");

// Display in Markets component
const tick = derivClient.get_tick("frxEURUSD");
console.log(`EUR/USD: ${tick.bid} / ${tick.ask}`);
```

### F5 - Portfolio (Position Tracking)

Monitor your Deriv positions in the Portfolio tab:

```typescript
// Get all open positions
const positions = await derivClient.get_positions();

// Display in Portfolio component
positions.forEach(pos => {
    console.log(`${pos.symbol}: P&L ${pos.profit}`);
});
```

### F6 - Execution (Order Placement)

Place orders through the Execution Management tab:

```typescript
// Place a buy order
const order = await derivClient.buy_contract(
    "R_100",
    OrderType.CALL,
    0.50,
    5,
    "t"
);

// Monitor order status
console.log(`Order placed: ${order.contract_id}`);
```

## Risk Management

### Built-in Safety Controls

1. **Position Size Limits**
   - Maximum 1.0 USD per trade
   - Prevents over-exposure

2. **Daily Trade Limits**
   - Maximum 50 trades per day
   - Prevents excessive trading

3. **Daily Loss Limits**
   - Maximum 1000 USD loss per day
   - Automatic trading halt if exceeded

4. **Demo Mode Enforcement**
   - Only demo accounts allowed
   - No real money at risk

### Custom Risk Rules

You can add custom risk rules in the Python bridge:

```python
class CustomRiskManager:
    def check_trade(self, symbol, amount, account_balance):
        # Custom risk checks
        if amount > account_balance * 0.01:
            raise ValueError("Trade exceeds 1% of account")
        
        if symbol not in ALLOWED_SYMBOLS:
            raise ValueError("Symbol not allowed")
        
        return True
```

## Monitoring and Logging

### Event Logging

All Deriv events are logged:

```
[INFO] Connected to Deriv WebSocket
[INFO] Authorized - Account: VRTC12345678 (DEMO), Balance: 10000.00 USD
[DEBUG] Subscribed to ticks for R_100
[INFO] Contract purchased - ID: 123456789, Price: 0.50
[DEBUG] Position update: 123456789 - P&L: 0.15
[INFO] Contract closed - ID: 123456789, P&L: 0.25, Status: won
```

### Performance Metrics

Track your trading performance:

```python
# Get daily statistics
stats = {
    "trades": client.daily_trades,
    "pnl": client.daily_pnl,
    "win_rate": calculate_win_rate(client.orders),
    "avg_profit": calculate_avg_profit(client.orders)
}
```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to Deriv API

**Solutions**:
1. Check internet connection
2. Verify API token is correct
3. Ensure firewall allows WebSocket connections
4. Check Deriv API status: https://deriv.com/status

### Authorization Failures

**Problem**: Authorization failed

**Solutions**:
1. Verify API token in `.env` file
2. Check token hasn't expired
3. Ensure token has correct scopes: `read`, `trade`, `trading_information`
4. Generate new token at: https://app.deriv.com/account/api-token

### Trade Placement Errors

**Problem**: Cannot place trades

**Solutions**:
1. Check account balance is sufficient
2. Verify symbol is available for trading
3. Ensure daily limits haven't been reached
4. Check trade amount is within limits

## Next Steps

### 1. Install Dependencies

```bash
# Python dependencies
cd intelligence-layer
pip install -e .

# Rust dependencies
cd ../execution-core
cargo build
```

### 2. Test Connection

```bash
python scripts/test-deriv-connection.py
```

### 3. Integrate with Strategies

Connect your trading strategies to the Deriv adapter:

```python
from intelligence_layer.deriv_integration import DerivClient
from intelligence_layer.strategy_orchestration import StrategyOrchestrator

async def run_strategy_with_deriv():
    # Create Deriv client
    deriv = DerivClient()
    await deriv.connect()
    
    # Create strategy orchestrator
    orchestrator = StrategyOrchestrator()
    
    # Connect strategy signals to Deriv orders
    async for signal in orchestrator.generate_signals():
        if signal.action == "BUY":
            await deriv.buy_contract(
                symbol=signal.symbol,
                contract_type=OrderType.CALL,
                amount=signal.size,
                duration=signal.duration
            )
```

### 4. Monitor Performance

Track your strategy performance in real-time:

```python
# Get account balance
account = deriv.get_account_info()
print(f"Balance: {account['balance']}")

# Get open positions
positions = deriv.get_positions()
print(f"Open positions: {len(positions)}")

# Calculate P&L
total_pnl = sum(pos['profit'] for pos in positions)
print(f"Total P&L: {total_pnl}")
```

## Security Notes

### API Token Security

- ✅ API token is stored in `.env` file (not committed to git)
- ✅ Token is for demo account only
- ✅ Token has limited scopes
- ⚠️ Never share your API token
- ⚠️ Never commit `.env` to version control

### Demo Account Safety

- ✅ Using demo account (no real money)
- ✅ Safety limits enforced
- ✅ Can reset demo balance anytime
- ℹ️ Demo account replicates real market conditions

## Support

### Deriv Resources

- **API Documentation**: https://api.deriv.com/
- **Developer Portal**: https://developers.deriv.com/
- **Community Forum**: https://community.deriv.com/
- **Support**: https://deriv.com/contact-us/

### System Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: See `README.md` for system overview
- **Architecture**: See `ARCHITECTURE_SUMMARY.md` for technical details

## Summary

Your trading system is now connected to Deriv's demo trading platform with:

✅ Real-time market data streaming  
✅ Order placement and execution  
✅ Position tracking and monitoring  
✅ Account balance updates  
✅ Safety controls and risk limits  
✅ Demo account (no real money)  

You can now test your strategies with real market data in a safe, controlled environment!
