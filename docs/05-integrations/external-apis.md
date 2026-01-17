# External API Integrations

This guide covers the integration of external APIs and data sources with the algorithmic trading platform.

## Overview

The platform supports integration with various external APIs for market data, economic indicators, news feeds, and execution services. These integrations provide comprehensive market coverage and enable sophisticated trading strategies.

## Supported Integrations

### Market Data Providers

#### Yahoo Finance (Implemented)
- **Purpose**: Historical and real-time market data
- **Assets**: Stocks, forex, crypto, indices, commodities
- **Features**: OHLCV data, corporate actions, dividends
- **Rate Limits**: No official limits, but respectful usage recommended
- **Implementation**: Direct API integration via Data Workspace (F10)

**Usage Example:**
```python
# Via Data Workspace import functionality
# Search: "AAPL" → Import OHLCV data
# Supports: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo intervals
```

#### Alpha Vantage (Planned)
- **Purpose**: Professional market data and technical indicators
- **Assets**: Stocks, forex, crypto, commodities
- **Features**: Real-time quotes, technical indicators, fundamental data
- **Rate Limits**: 5 calls/minute (free), 75 calls/minute (premium)
- **API Key**: Required for all requests

**Configuration:**
```python
# .env configuration
ALPHA_VANTAGE_API_KEY=your_api_key_here
ALPHA_VANTAGE_BASE_URL=https://www.alphavantage.co/query
```

#### Quandl (Planned)
- **Purpose**: Economic and financial datasets
- **Assets**: Economic indicators, alternative data, commodities
- **Features**: Historical data, economic calendars, research datasets
- **Rate Limits**: 50 calls/day (free), unlimited (premium)

#### FRED (Federal Reserve Economic Data) (Planned)
- **Purpose**: US economic indicators and monetary policy data
- **Assets**: Interest rates, inflation, employment, GDP
- **Features**: Historical economic time series
- **Rate Limits**: No official limits, but respectful usage recommended

### Execution Venues

#### Deriv API (Implemented)
- **Purpose**: CFD and forex trading execution
- **Assets**: Forex pairs, indices, commodities, synthetic indices
- **Features**: Real-time quotes, order execution, account management
- **Connection**: WebSocket-based real-time API

**Configuration:**
```python
# Deriv API configuration
DERIV_APP_ID=your_app_id
DERIV_API_TOKEN=your_api_token
DERIV_ENDPOINT=wss://ws.binaryws.com/websockets/v3
```

**Implementation Status:**
- ✅ Connection established
- ✅ Real-time price streaming
- ✅ Account information retrieval
- ✅ Order placement (shadow mode)
- ⏳ Live trading (pending final testing)

#### Interactive Brokers (Planned)
- **Purpose**: Professional trading platform integration
- **Assets**: Stocks, options, futures, forex, bonds
- **Features**: Advanced order types, portfolio margin, global markets
- **Connection**: TWS API or IB Gateway

#### MetaTrader 5 (Planned)
- **Purpose**: Retail forex and CFD trading
- **Assets**: Forex, indices, commodities, cryptocurrencies
- **Features**: Expert Advisors, custom indicators, backtesting
- **Connection**: MT5 Python API

### News and Sentiment

#### NewsAPI (Planned)
- **Purpose**: Financial news aggregation
- **Sources**: Reuters, Bloomberg, Financial Times, etc.
- **Features**: Real-time news, sentiment analysis, keyword filtering
- **Rate Limits**: 1000 requests/day (free), unlimited (premium)

#### Twitter API (Planned)
- **Purpose**: Social sentiment analysis
- **Data**: Tweets, mentions, hashtags, user metrics
- **Features**: Real-time streaming, sentiment scoring, influence metrics
- **Rate Limits**: Various limits based on endpoint and plan

### Economic Data

#### Trading Economics (Planned)
- **Purpose**: Economic indicators and calendar events
- **Data**: GDP, inflation, employment, central bank decisions
- **Features**: Real-time updates, historical data, forecasts
- **Coverage**: 196 countries, 300,000+ indicators

#### Bloomberg API (Future)
- **Purpose**: Professional-grade market and economic data
- **Data**: Real-time prices, news, analytics, research
- **Features**: Terminal-quality data, advanced analytics
- **Requirements**: Bloomberg Terminal subscription

## Integration Architecture

### Data Flow
```
External APIs → API Adapters → Data Normalization → Database Storage
     ↓
Intelligence Layer → Analysis → Trading Signals → Execution Core
```

### API Adapter Pattern
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime

class MarketDataAdapter(ABC):
    """Abstract base class for market data adapters."""
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str
    ) -> List[Dict[str, Any]]:
        """Fetch historical OHLCV data."""
        pass
    
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for symbol."""
        pass
    
    @abstractmethod
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Search for symbols matching query."""
        pass

class YahooFinanceAdapter(MarketDataAdapter):
    """Yahoo Finance API adapter implementation."""
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str) -> List[Dict[str, Any]]:
        # Implementation for Yahoo Finance API
        pass
```

### Data Normalization
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class NormalizedOHLCV:
    """Normalized OHLCV data structure."""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    source: str
    raw_data: Optional[Dict[str, Any]] = None

class DataNormalizer:
    """Normalize data from different sources to common format."""
    
    def normalize_yahoo_data(self, raw_data: Dict[str, Any]) -> NormalizedOHLCV:
        """Normalize Yahoo Finance data."""
        return NormalizedOHLCV(
            symbol=raw_data['symbol'],
            timestamp=datetime.fromtimestamp(raw_data['timestamp']),
            open_price=raw_data['open'],
            high_price=raw_data['high'],
            low_price=raw_data['low'],
            close_price=raw_data['close'],
            volume=raw_data['volume'],
            source='yahoo_finance',
            raw_data=raw_data
        )
```

## Implementation Examples

### Yahoo Finance Integration
```python
import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime

class YahooFinanceService:
    """Service for Yahoo Finance data integration."""
    
    async def fetch_historical_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """Fetch historical data from Yahoo Finance."""
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            data = []
            for timestamp, row in hist.iterrows():
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume'],
                    'source': 'yahoo_finance'
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            raise
    
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Search for symbols on Yahoo Finance."""
        
        # Implementation would use Yahoo Finance search API
        # or maintain a symbol database
        pass
```

### Deriv API Integration
```python
import websockets
import json
from typing import Dict, Any, Callable

class DerivAPIClient:
    """Deriv API WebSocket client."""
    
    def __init__(self, app_id: str, api_token: str):
        self.app_id = app_id
        self.api_token = api_token
        self.websocket = None
        self.callbacks = {}
    
    async def connect(self):
        """Connect to Deriv API WebSocket."""
        uri = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.websocket = await websockets.connect(uri)
        
        # Authorize connection
        await self.authorize()
    
    async def authorize(self):
        """Authorize API connection."""
        auth_request = {
            "authorize": self.api_token
        }
        await self.websocket.send(json.dumps(auth_request))
        response = await self.websocket.recv()
        # Handle authorization response
    
    async def subscribe_to_ticks(self, symbol: str, callback: Callable):
        """Subscribe to real-time tick data."""
        request = {
            "ticks": symbol,
            "subscribe": 1
        }
        
        self.callbacks[f"tick_{symbol}"] = callback
        await self.websocket.send(json.dumps(request))
    
    async def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place trading order."""
        request = {
            "buy": 1,
            "price": order_params['price'],
            "parameters": {
                "contract_type": order_params['contract_type'],
                "currency": order_params['currency'],
                "symbol": order_params['symbol'],
                "amount": order_params['amount']
            }
        }
        
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)
```

## Configuration Management

### Environment Variables
```bash
# Market Data APIs
YAHOO_FINANCE_ENABLED=true
ALPHA_VANTAGE_API_KEY=your_key_here
ALPHA_VANTAGE_ENABLED=false
QUANDL_API_KEY=your_key_here
QUANDL_ENABLED=false

# Execution APIs
DERIV_APP_ID=your_app_id
DERIV_API_TOKEN=your_token
DERIV_ENABLED=true
IB_ENABLED=false
MT5_ENABLED=false

# News and Sentiment
NEWS_API_KEY=your_key_here
NEWS_API_ENABLED=false
TWITTER_API_KEY=your_key_here
TWITTER_ENABLED=false

# Rate Limiting
API_RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### Configuration Classes
```python
from pydantic import BaseSettings, validator
from typing import Optional

class ExternalAPISettings(BaseSettings):
    """External API configuration settings."""
    
    # Yahoo Finance
    yahoo_finance_enabled: bool = True
    
    # Alpha Vantage
    alpha_vantage_api_key: Optional[str] = None
    alpha_vantage_enabled: bool = False
    
    # Deriv
    deriv_app_id: Optional[str] = None
    deriv_api_token: Optional[str] = None
    deriv_enabled: bool = False
    
    # Rate limiting
    api_rate_limit_enabled: bool = True
    api_rate_limit_requests_per_minute: int = 60
    
    @validator('alpha_vantage_enabled')
    def validate_alpha_vantage(cls, v, values):
        if v and not values.get('alpha_vantage_api_key'):
            raise ValueError('Alpha Vantage API key required when enabled')
        return v
    
    @validator('deriv_enabled')
    def validate_deriv(cls, v, values):
        if v and not all([values.get('deriv_app_id'), values.get('deriv_api_token')]):
            raise ValueError('Deriv credentials required when enabled')
        return v
    
    class Config:
        env_file = '.env'
```

## Rate Limiting and Error Handling

### Rate Limiting Implementation
```python
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(list)
    
    async def acquire(self, api_name: str) -> bool:
        """Acquire rate limit token."""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.time_window)
        
        # Remove old calls
        self.calls[api_name] = [
            call_time for call_time in self.calls[api_name]
            if call_time > cutoff
        ]
        
        # Check if we can make another call
        if len(self.calls[api_name]) < self.max_calls:
            self.calls[api_name].append(now)
            return True
        
        # Calculate wait time
        oldest_call = min(self.calls[api_name])
        wait_time = (oldest_call + timedelta(seconds=self.time_window) - now).total_seconds()
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            return await self.acquire(api_name)
        
        return True
```

### Error Handling and Retry Logic
```python
import asyncio
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Base exception for API errors."""
    pass

class RateLimitError(APIError):
    """Rate limit exceeded error."""
    pass

class APIUnavailableError(APIError):
    """API temporarily unavailable error."""
    pass

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Any:
    """Retry function with exponential backoff."""
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        
        except RateLimitError:
            if attempt == max_retries:
                raise
            
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1})")
            await asyncio.sleep(delay)
        
        except APIUnavailableError:
            if attempt == max_retries:
                raise
            
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            logger.warning(f"API unavailable, retrying in {delay}s (attempt {attempt + 1})")
            await asyncio.sleep(delay)
        
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise
    
    raise APIError("Max retries exceeded")
```

## Data Quality and Validation

### Data Validation
```python
from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional

class MarketDataPoint(BaseModel):
    """Validated market data point."""
    
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    source: str
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 2:
            raise ValueError('Symbol must be at least 2 characters')
        return v.upper()
    
    @validator('high_price')
    def validate_high_price(cls, v, values):
        if 'low_price' in values and v < values['low_price']:
            raise ValueError('High price cannot be less than low price')
        return v
    
    @validator('volume')
    def validate_volume(cls, v):
        if v < 0:
            raise ValueError('Volume cannot be negative')
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.utcnow():
            raise ValueError('Timestamp cannot be in the future')
        return v
```

### Data Quality Checks
```python
class DataQualityChecker:
    """Check data quality and detect anomalies."""
    
    def check_price_continuity(self, data: List[MarketDataPoint]) -> List[str]:
        """Check for price gaps and anomalies."""
        issues = []
        
        for i in range(1, len(data)):
            prev_point = data[i-1]
            curr_point = data[i]
            
            # Check for large price gaps (>10%)
            price_change = abs(curr_point.close_price - prev_point.close_price) / prev_point.close_price
            if price_change > 0.10:
                issues.append(f"Large price gap at {curr_point.timestamp}: {price_change:.2%}")
            
            # Check for zero volume
            if curr_point.volume == 0:
                issues.append(f"Zero volume at {curr_point.timestamp}")
        
        return issues
    
    def check_data_completeness(self, data: List[MarketDataPoint], expected_interval: str) -> List[str]:
        """Check for missing data points."""
        issues = []
        
        if len(data) < 2:
            return issues
        
        # Calculate expected time delta based on interval
        interval_deltas = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
        
        expected_delta = interval_deltas.get(expected_interval)
        if not expected_delta:
            return issues
        
        for i in range(1, len(data)):
            prev_time = data[i-1].timestamp
            curr_time = data[i].timestamp
            actual_delta = curr_time - prev_time
            
            if actual_delta > expected_delta * 1.5:  # Allow 50% tolerance
                issues.append(f"Missing data between {prev_time} and {curr_time}")
        
        return issues
```

## Monitoring and Alerting

### API Health Monitoring
```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

class APIHealthMonitor:
    """Monitor API health and performance."""
    
    def __init__(self):
        self.health_status = {}
        self.response_times = {}
        self.error_counts = {}
    
    async def check_api_health(self, api_name: str, health_check_func: Callable) -> Dict[str, Any]:
        """Check health of specific API."""
        start_time = datetime.utcnow()
        
        try:
            await health_check_func()
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.health_status[api_name] = {
                'status': 'healthy',
                'last_check': start_time,
                'response_time': response_time,
                'error_count': self.error_counts.get(api_name, 0)
            }
            
            return self.health_status[api_name]
        
        except Exception as e:
            self.error_counts[api_name] = self.error_counts.get(api_name, 0) + 1
            
            self.health_status[api_name] = {
                'status': 'unhealthy',
                'last_check': start_time,
                'error': str(e),
                'error_count': self.error_counts[api_name]
            }
            
            return self.health_status[api_name]
    
    async def monitor_all_apis(self):
        """Continuously monitor all configured APIs."""
        while True:
            tasks = []
            
            if settings.yahoo_finance_enabled:
                tasks.append(self.check_api_health('yahoo_finance', self._check_yahoo_health))
            
            if settings.deriv_enabled:
                tasks.append(self.check_api_health('deriv', self._check_deriv_health))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(60)  # Check every minute
    
    async def _check_yahoo_health(self):
        """Health check for Yahoo Finance."""
        # Simple health check - fetch a known symbol
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        if not info:
            raise APIUnavailableError("Yahoo Finance API not responding")
    
    async def _check_deriv_health(self):
        """Health check for Deriv API."""
        # Implement Deriv-specific health check
        pass
```

## Security Considerations

### API Key Management
```python
import os
from cryptography.fernet import Fernet

class SecureAPIKeyManager:
    """Secure storage and retrieval of API keys."""
    
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for storage."""
        encrypted = self.cipher.encrypt(api_key.encode())
        return encrypted.decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key for use."""
        decrypted = self.cipher.decrypt(encrypted_key.encode())
        return decrypted.decode()
    
    @classmethod
    def generate_encryption_key(cls) -> bytes:
        """Generate new encryption key."""
        return Fernet.generate_key()
```

### Request Signing and Authentication
```python
import hmac
import hashlib
from datetime import datetime

class APIAuthenticator:
    """Handle API authentication and request signing."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
    
    def sign_request(self, method: str, endpoint: str, params: Dict[str, Any]) -> Dict[str, str]:
        """Sign API request with HMAC."""
        timestamp = str(int(datetime.utcnow().timestamp()))
        
        # Create signature payload
        payload = f"{method}{endpoint}{timestamp}"
        if params:
            payload += "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Generate HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'X-API-Key': self.api_key,
            'X-Timestamp': timestamp,
            'X-Signature': signature
        }
```

## Future Enhancements

### Planned Integrations
1. **Bloomberg Terminal API**: Professional-grade market data
2. **Refinitiv Eikon**: Comprehensive financial data platform
3. **IEX Cloud**: Real-time and historical market data
4. **Polygon.io**: High-frequency market data
5. **CoinGecko/CoinMarketCap**: Cryptocurrency data
6. **Economic Calendar APIs**: Central bank decisions, economic releases

### Advanced Features
1. **Smart Routing**: Automatically route requests to best available data source
2. **Data Fusion**: Combine data from multiple sources for enhanced accuracy
3. **Predictive Caching**: Pre-fetch data based on usage patterns
4. **Real-time Arbitrage**: Detect price discrepancies across venues
5. **Sentiment Integration**: Incorporate news and social sentiment into trading signals

### Performance Optimizations
1. **Connection Pooling**: Reuse connections across requests
2. **Batch Processing**: Group multiple requests for efficiency
3. **Compression**: Use gzip/deflate for large data transfers
4. **CDN Integration**: Cache static reference data
5. **Edge Computing**: Deploy API adapters closer to data sources

This comprehensive external API integration framework provides the foundation for connecting the algorithmic trading platform with the broader financial data ecosystem, enabling sophisticated trading strategies and comprehensive market analysis.