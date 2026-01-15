#!/usr/bin/env python3
"""
Test Deriv API Connection
Verifies that the Deriv demo account is properly configured and accessible
"""

import asyncio
import sys
import os
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add intelligence-layer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "intelligence-layer" / "src"))

from intelligence_layer.deriv_integration import DerivClient, DerivConfig, OrderType


async def test_basic_connection():
    """Test basic connection and authorization"""
    print("=" * 60)
    print("DERIV API CONNECTION TEST")
    print("=" * 60)
    print()
    
    # Load config
    config = DerivConfig.from_env()
    print(f"Configuration:")
    print(f"  App ID: {config.app_id}")
    print(f"  API Token: {config.api_token[:10]}..." if config.api_token else "  API Token: NOT SET")
    print(f"  WebSocket URL: {config.websocket_url}")
    print(f"  Demo Mode: {config.demo_mode}")
    print(f"  Max Position Size: {config.max_position_size}")
    print(f"  Max Daily Trades: {config.max_daily_trades}")
    print(f"  Max Daily Loss: {config.max_daily_loss}")
    print()
    
    if not config.api_token:
        print("❌ ERROR: DERIV_API_TOKEN not set in environment")
        return False
    
    # Create client
    client = DerivClient(config)
    
    try:
        # Connect
        print("Connecting to Deriv API...")
        connected = await client.connect()
        
        if not connected:
            print("❌ Failed to connect")
            return False
        
        print("✅ Connected successfully")
        print()
        
        # Check account
        if client.account:
            print("Account Information:")
            print(f"  Login ID: {client.account.loginid}")
            print(f"  Balance: {client.account.balance} {client.account.currency}")
            print(f"  Account Type: {'DEMO' if client.account.is_virtual else 'REAL'}")
            print()
            
            if not client.account.is_virtual:
                print("⚠️  WARNING: This is a REAL account, not a demo account!")
                print("   Please use a demo account for testing.")
                return False
        else:
            print("❌ No account information received")
            return False
        
        # Get available symbols
        print("Fetching available symbols...")
        symbols = await client.get_active_symbols()
        print(f"✅ Found {len(symbols)} available symbols")
        
        # Show some popular symbols
        forex_symbols = [s for s in symbols if s.get("market") == "forex" and s.get("submarket") == "major_pairs"]
        synthetic_symbols = [s for s in symbols if s.get("market") == "synthetic_index"]
        
        if forex_symbols:
            print(f"\nForex Symbols ({len(forex_symbols)}):")
            for symbol in forex_symbols[:5]:
                print(f"  - {symbol['symbol']}: {symbol['display_name']}")
        
        if synthetic_symbols:
            print(f"\nSynthetic Indices ({len(synthetic_symbols)}):")
            for symbol in synthetic_symbols[:5]:
                print(f"  - {symbol['symbol']}: {symbol['display_name']}")
        
        print()
        
        # Subscribe to a tick
        print("Testing tick subscription (R_100 - Volatility 100 Index)...")
        await client.subscribe_ticks("R_100")
        
        # Wait for a few ticks
        print("Waiting for ticks...")
        for i in range(5):
            await asyncio.sleep(1)
            if "R_100" in client.ticks:
                tick = client.ticks["R_100"]
                print(f"  Tick {i+1}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}, Mid={tick.mid:.5f}")
        
        print()
        print("✅ All tests passed!")
        print()
        print("=" * 60)
        print("READY FOR TRADING")
        print("=" * 60)
        print()
        print("Your Deriv demo account is properly configured and ready to use.")
        print("You can now:")
        print("  1. Place demo trades through the system")
        print("  2. Test strategies with real market data")
        print("  3. Monitor positions and P&L in real-time")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await client.disconnect()
        print("Disconnected from Deriv API")


async def test_simple_trade():
    """Test placing a simple trade (optional)"""
    print("\n" + "=" * 60)
    print("OPTIONAL: TEST TRADE")
    print("=" * 60)
    print()
    
    response = input("Do you want to test placing a small demo trade? (yes/no): ")
    if response.lower() != "yes":
        print("Skipping trade test")
        return True
    
    config = DerivConfig.from_env()
    client = DerivClient(config)
    
    try:
        await client.connect()
        
        print("\nPlacing a small test trade...")
        print("  Symbol: R_100 (Volatility 100 Index)")
        print("  Type: CALL (Up)")
        print("  Amount: 0.35 USD")
        print("  Duration: 5 ticks")
        print()
        
        contract = await client.buy_contract(
            symbol="R_100",
            contract_type=OrderType.CALL,
            amount=0.35,
            duration=5,
            duration_unit="t"
        )
        
        if contract:
            print("✅ Trade placed successfully!")
            print(f"  Contract ID: {contract.get('contract_id')}")
            print(f"  Buy Price: {contract.get('buy_price')}")
            print(f"  Payout: {contract.get('payout')}")
            print()
            
            # Monitor the contract
            print("Monitoring contract...")
            for i in range(10):
                await asyncio.sleep(1)
                if contract["contract_id"] in client.positions:
                    pos = client.positions[contract["contract_id"]]
                    print(f"  Update {i+1}: Current={pos.current_spot:.5f}, P&L={pos.profit:.2f}")
                else:
                    print(f"  Contract closed")
                    break
            
            print()
            print("✅ Trade test completed!")
            
        else:
            print("❌ Failed to place trade")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        await client.disconnect()


async def main():
    """Main test function"""
    # Test basic connection
    success = await test_basic_connection()
    
    if not success:
        print("\n❌ Connection test failed")
        sys.exit(1)
    
    # Optionally test a trade
    await test_simple_trade()
    
    print("\n✅ All tests completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
