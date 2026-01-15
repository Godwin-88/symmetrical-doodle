# Deriv Integration - Quick Start

## 🚀 Get Started in 5 Minutes

Your algorithmic trading system is now connected to Deriv's demo trading platform!

### Step 1: Install Dependencies

```bash
# Install Python dependencies
cd intelligence-layer
pip install websockets python-dotenv

# Or install all dependencies
pip install -e .
```

### Step 2: Verify Configuration

Your `.env` file is already configured with:

```bash
DERIV_APP_ID=118029
DERIV_API_TOKEN=gxF5pHUCgjDTOGI
DERIV_DEMO_MODE=true
```

### Step 3: Test Connection

```bash
# Run the test script
python scripts/test-deriv-connection.py
```

Expected output:
```
✅ Connected successfully
✅ Account: VRTC12345678
✅ Balance: 10000.00 USD
✅ Demo Mode: True
✅ Available symbols: 150+
```

### Step 4: Start Trading!

You can now:

1. **View Real-time Market Data**
   - Open F3 (Markets) tab
   - See live Deriv prices

2. **Place Demo Trades**
   - Open F6 (Execution) tab
   - Place orders on forex pairs or synthetic indices

3. **Monitor Positions**
   - Open F5 (Portfolio) tab
   - Track P&L in real-time

4. **Test Strategies**
   - Connect your strategies to Deriv
   - Backtest with real market data

## 📊 Available Markets

### Forex Pairs
- EUR/USD, GBP/USD, USD/JPY
- AUD/USD, USD/CAD, EUR/GBP
- And 20+ more pairs

### Synthetic Indices
- Volatility: R_10, R_25, R_50, R_75, R_100
- Crash/Boom: CRASH300, CRASH500, BOOM300, BOOM500
- Step Indices

### Commodities
- Gold, Silver, Oil

## 🛡️ Safety Features

✅ **Demo Account Only** - No real money at risk  
✅ **Position Limits** - Max 1.0 USD per trade  
✅ **Daily Limits** - Max 50 trades, 1000 USD loss  
✅ **Automatic Stops** - Trading halts if limits exceeded  

## 📖 Next Steps

- **Full Guide**: See `DERIV_INTEGRATION_GUIDE.md` for complete documentation
- **API Docs**: https://api.deriv.com/
- **Support**: https://deriv.com/contact-us/

## 🎯 Quick Example

```python
from intelligence_layer.deriv_integration import DerivClient, OrderType

async def quick_trade():
    # Connect
    client = DerivClient()
    await client.connect()
    
    # Check balance
    account = client.get_account_info()
    print(f"Balance: {account['balance']} USD")
    
    # Place trade
    contract = await client.buy_contract(
        symbol="R_100",
        contract_type=OrderType.CALL,
        amount=0.50,
        duration=5,
        duration_unit="t"
    )
    
    print(f"Trade placed: {contract['contract_id']}")
    
    # Disconnect
    await client.disconnect()
```

## ✅ You're Ready!

Your system is configured and ready to trade on Deriv's demo platform. Start testing your strategies with real market data!

---

**Need Help?**
- Check `DERIV_INTEGRATION_GUIDE.md` for detailed documentation
- Visit https://api.deriv.com/ for API reference
- Contact Deriv support at https://deriv.com/contact-us/
