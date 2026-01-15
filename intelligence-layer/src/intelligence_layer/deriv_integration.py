"""
Deriv API Integration - Python Bridge
Provides high-level interface to Deriv demo trading functionality
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DerivConfig:
    """Deriv API configuration"""
    app_id: str
    api_token: str
    websocket_url: str
    demo_mode: bool = True
    max_position_size: float = 1.0
    max_daily_trades: int = 50
    max_daily_loss: float = 1000.0
    
    @classmethod
    def from_env(cls) -> "DerivConfig":
        """Load configuration from environment variables"""
        app_id = os.getenv("DERIV_APP_ID", "118029")
        api_token = os.getenv("DERIV_API_TOKEN", "")
        websocket_url = os.getenv(
            "DERIV_WEBSOCKET_URL",
            f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        )
        
        return cls(
            app_id=app_id,
            api_token=api_token,
            websocket_url=websocket_url,
            demo_mode=os.getenv("DERIV_DEMO_MODE", "true").lower() == "true",
            max_position_size=float(os.getenv("DERIV_MAX_POSITION_SIZE", "1.0")),
            max_daily_trades=int(os.getenv("DERIV_MAX_DAILY_TRADES", "50")),
            max_daily_loss=float(os.getenv("DERIV_MAX_DAILY_LOSS", "1000.0")),
        )


# ============================================================================
# DATA MODELS
# ============================================================================

class OrderType(str, Enum):
    """Order types"""
    CALL = "CALL"  # Buy/Up
    PUT = "PUT"    # Sell/Down


class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "pending"
    OPEN = "open"
    WON = "won"
    LOST = "lost"
    SOLD = "sold"


@dataclass
class DerivTick:
    """Market tick data"""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    
    @property
    def mid(self) -> float:
        """Mid price"""
        return (self.bid + self.ask) / 2


@dataclass
class DerivAccount:
    """Account information"""
    balance: float
    currency: str
    loginid: str
    is_virtual: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivPosition:
    """Open position"""
    contract_id: str
    symbol: str
    contract_type: str
    buy_price: float
    current_spot: float
    profit: float
    payout: float
    purchase_time: datetime
    
    @property
    def pnl_pct(self) -> float:
        """P&L percentage"""
        if self.buy_price == 0:
            return 0.0
        return (self.profit / self.buy_price) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['pnl_pct'] = self.pnl_pct
        return data


# ============================================================================
# DERIV CLIENT
# ============================================================================

class DerivClient:
    """
    Deriv API WebSocket client
    Handles connection, authentication, and trading operations
    """
    
    def __init__(self, config: Optional[DerivConfig] = None):
        self.config = config or DerivConfig.from_env()
        self.ws: Optional[WebSocketClientProtocol] = None
        self.account: Optional[DerivAccount] = None
        self.positions: Dict[str, DerivPosition] = {}
        self.ticks: Dict[str, DerivTick] = {}
        self.subscriptions: Dict[str, str] = {}  # symbol -> subscription_id
        self.connected = False
        self.authorized = False
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        logger.info(f"Deriv client initialized - Demo mode: {self.config.demo_mode}")
    
    async def connect(self) -> bool:
        """Connect to Deriv WebSocket API"""
        try:
            logger.info(f"Connecting to Deriv API: {self.config.websocket_url}")
            self.ws = await websockets.connect(self.config.websocket_url)
            self.connected = True
            logger.info("Connected to Deriv WebSocket")
            
            # Authorize
            await self.authorize()
            
            # Subscribe to balance updates
            await self.subscribe_balance()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Deriv API: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Deriv API"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            self.authorized = False
            logger.info("Disconnected from Deriv API")
    
    async def authorize(self) -> bool:
        """Authorize with API token"""
        if not self.ws:
            raise RuntimeError("Not connected to Deriv API")
        
        auth_request = {
            "authorize": self.config.api_token
        }
        
        await self.ws.send(json.dumps(auth_request))
        response = await self.ws.recv()
        data = json.loads(response)
        
        if "error" in data:
            logger.error(f"Authorization failed: {data['error']['message']}")
            return False
        
        if "authorize" in data:
            auth_data = data["authorize"]
            self.account = DerivAccount(
                balance=auth_data.get("balance", 0.0),
                currency=auth_data.get("currency", "USD"),
                loginid=auth_data.get("loginid", "unknown"),
                is_virtual=auth_data.get("is_virtual", 0) == 1
            )
            self.authorized = True
            
            logger.info(
                f"Authorized - Account: {self.account.loginid} "
                f"({'DEMO' if self.account.is_virtual else 'REAL'}), "
                f"Balance: {self.account.balance} {self.account.currency}"
            )
            
            return True
        
        return False
    
    async def subscribe_balance(self):
        """Subscribe to balance updates"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        request = {
            "balance": 1,
            "subscribe": 1
        }
        
        await self.ws.send(json.dumps(request))
        logger.debug("Subscribed to balance updates")
    
    async def subscribe_ticks(self, symbol: str):
        """Subscribe to tick updates for a symbol"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        request = {
            "ticks": symbol,
            "subscribe": 1
        }
        
        await self.ws.send(json.dumps(request))
        response = await self.ws.recv()
        data = json.loads(response)
        
        if "subscription" in data:
            sub_id = data["subscription"]["id"]
            self.subscriptions[symbol] = sub_id
            logger.info(f"Subscribed to ticks for {symbol}")
        
        return data
    
    async def get_active_symbols(self) -> List[Dict[str, Any]]:
        """Get list of available trading symbols"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        request = {
            "active_symbols": "brief",
            "product_type": "basic"
        }
        
        await self.ws.send(json.dumps(request))
        response = await self.ws.recv()
        data = json.loads(response)
        
        if "active_symbols" in data:
            return data["active_symbols"]
        
        return []
    
    async def buy_contract(
        self,
        symbol: str,
        contract_type: OrderType,
        amount: float,
        duration: int = 5,
        duration_unit: str = "t"  # t=ticks, m=minutes, h=hours
    ) -> Optional[Dict[str, Any]]:
        """
        Buy a contract (place an order)
        
        Args:
            symbol: Trading symbol (e.g., "R_100", "frxEURUSD")
            contract_type: CALL (buy/up) or PUT (sell/down)
            amount: Stake amount
            duration: Contract duration
            duration_unit: Duration unit (t=ticks, m=minutes, h=hours)
        
        Returns:
            Contract details if successful
        """
        if not self.authorized:
            raise RuntimeError("Not authorized")
        
        # Safety checks
        if amount > self.config.max_position_size:
            logger.warning(f"Amount {amount} exceeds max position size {self.config.max_position_size}")
            return None
        
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trades}")
            return None
        
        if abs(self.daily_pnl) >= self.config.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return None
        
        # Get proposal first
        proposal_request = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type.value,
            "currency": self.account.currency if self.account else "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol
        }
        
        await self.ws.send(json.dumps(proposal_request))
        proposal_response = await self.ws.recv()
        proposal_data = json.loads(proposal_response)
        
        if "error" in proposal_data:
            logger.error(f"Proposal error: {proposal_data['error']['message']}")
            return None
        
        if "proposal" not in proposal_data:
            logger.error("No proposal in response")
            return None
        
        proposal_id = proposal_data["proposal"]["id"]
        
        # Buy the contract
        buy_request = {
            "buy": proposal_id,
            "price": amount
        }
        
        await self.ws.send(json.dumps(buy_request))
        buy_response = await self.ws.recv()
        buy_data = json.loads(buy_response)
        
        if "error" in buy_data:
            logger.error(f"Buy error: {buy_data['error']['message']}")
            return None
        
        if "buy" in buy_data:
            contract = buy_data["buy"]
            self.daily_trades += 1
            
            logger.info(
                f"Contract purchased - ID: {contract.get('contract_id')}, "
                f"Price: {contract.get('buy_price')}"
            )
            
            # Subscribe to contract updates
            await self.subscribe_contract(contract["contract_id"])
            
            return contract
        
        return None
    
    async def subscribe_contract(self, contract_id: str):
        """Subscribe to contract updates"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1
        }
        
        await self.ws.send(json.dumps(request))
        logger.debug(f"Subscribed to contract {contract_id}")
    
    async def sell_contract(self, contract_id: str, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Sell an open contract"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        request = {
            "sell": contract_id
        }
        
        if price is not None:
            request["price"] = price
        
        await self.ws.send(json.dumps(request))
        response = await self.ws.recv()
        data = json.loads(response)
        
        if "error" in data:
            logger.error(f"Sell error: {data['error']['message']}")
            return None
        
        if "sell" in data:
            logger.info(f"Contract sold - ID: {contract_id}, Price: {data['sell'].get('sold_for')}")
            return data["sell"]
        
        return None
    
    async def get_portfolio(self) -> List[DerivPosition]:
        """Get all open positions"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        request = {
            "portfolio": 1
        }
        
        await self.ws.send(json.dumps(request))
        response = await self.ws.recv()
        data = json.loads(response)
        
        positions = []
        if "portfolio" in data:
            for contract in data["portfolio"]["contracts"]:
                position = DerivPosition(
                    contract_id=contract["contract_id"],
                    symbol=contract["symbol"],
                    contract_type=contract["contract_type"],
                    buy_price=contract["buy_price"],
                    current_spot=contract.get("current_spot", 0.0),
                    profit=contract.get("profit", 0.0),
                    payout=contract.get("payout", 0.0),
                    purchase_time=datetime.fromtimestamp(contract["purchase_time"])
                )
                positions.append(position)
                self.positions[position.contract_id] = position
        
        return positions
    
    async def process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle tick updates
            if "tick" in data:
                tick_data = data["tick"]
                tick = DerivTick(
                    symbol=tick_data["symbol"],
                    bid=tick_data.get("bid", 0.0),
                    ask=tick_data.get("ask", 0.0),
                    timestamp=datetime.fromtimestamp(tick_data["epoch"])
                )
                self.ticks[tick.symbol] = tick
            
            # Handle balance updates
            if "balance" in data:
                balance_data = data["balance"]
                if self.account:
                    old_balance = self.account.balance
                    self.account.balance = balance_data["balance"]
                    pnl_change = self.account.balance - old_balance
                    self.daily_pnl += pnl_change
                    logger.debug(f"Balance updated: {self.account.balance}")
            
            # Handle contract updates
            if "proposal_open_contract" in data:
                poc = data["proposal_open_contract"]
                contract_id = poc["contract_id"]
                
                position = DerivPosition(
                    contract_id=contract_id,
                    symbol=poc["underlying"],
                    contract_type=poc["contract_type"],
                    buy_price=poc["buy_price"],
                    current_spot=poc.get("current_spot", 0.0),
                    profit=poc.get("profit", 0.0),
                    payout=poc.get("payout", 0.0),
                    purchase_time=datetime.fromtimestamp(poc["purchase_time"])
                )
                
                self.positions[contract_id] = position
                
                # Check if contract is closed
                if poc.get("is_sold") or poc.get("status") in ["won", "lost"]:
                    logger.info(
                        f"Contract closed - ID: {contract_id}, "
                        f"P&L: {position.profit}, Status: {poc.get('status')}"
                    )
                    if contract_id in self.positions:
                        del self.positions[contract_id]
            
            # Handle errors
            if "error" in data:
                logger.error(f"Deriv API error: {data['error']['message']}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def listen(self):
        """Listen for incoming messages"""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        try:
            async for message in self.ws:
                await self.process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
            self.authorized = False
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get current account information"""
        if self.account:
            return self.account.to_dict()
        return None
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        return [pos.to_dict() for pos in self.positions.values()]
    
    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest tick for a symbol"""
        if symbol in self.ticks:
            return asdict(self.ticks[symbol])
        return None
    
    def is_ready(self) -> bool:
        """Check if client is connected and authorized"""
        return self.connected and self.authorized


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_deriv_client() -> DerivClient:
    """Create and connect a Deriv client"""
    client = DerivClient()
    await client.connect()
    return client


async def test_connection():
    """Test Deriv API connection"""
    client = DerivClient()
    
    try:
        connected = await client.connect()
        if connected:
            print(f"✓ Connected to Deriv API")
            print(f"✓ Account: {client.account.loginid}")
            print(f"✓ Balance: {client.account.balance} {client.account.currency}")
            print(f"✓ Demo Mode: {client.account.is_virtual}")
            
            # Get available symbols
            symbols = await client.get_active_symbols()
            print(f"✓ Available symbols: {len(symbols)}")
            
            return True
        else:
            print("✗ Failed to connect")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Test the connection
    asyncio.run(test_connection())
