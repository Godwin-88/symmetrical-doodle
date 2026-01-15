# External Data Import Feature

## Overview

The Data Workspace now includes powerful external data import capabilities, allowing users to fetch market data from Yahoo Finance and other sources directly through the UI, analyze it in real-time, and integrate it with the trading system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│                                                              │
│  DataWorkspace Component                                     │
│  ├── Symbol Search UI                                        │
│  ├── Date Range Selector                                     │
│  ├── Interval Selector                                       │
│  └── Import Button                                           │
│                                                              │
│  dataImportService.ts                                        │
│  ├── searchSymbols()                                         │
│  ├── getSymbolInfo()                                         │
│  └── importExternalData()                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Intelligence Layer (Python)                     │
│                                                              │
│  FastAPI Endpoints                                           │
│  ├── GET /data/search                                        │
│  ├── GET /data/symbol-info                                   │
│  └── POST /data/import                                       │
│                                                              │
│  data_import.py                                              │
│  └── DataImporter                                            │
│      ├── fetch_data()                                        │
│      ├── search_symbols()                                    │
│      └── get_symbol_info()                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ yfinance library
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Yahoo Finance API                           │
│                                                              │
│  - Historical market data                                    │
│  - Real-time quotes                                          │
│  - Symbol search                                             │
│  - Company information                                       │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. Symbol Search

**Endpoint**: `GET /data/search`

**Parameters:**
- `query`: Search term (e.g., "Apple", "EUR", "BTC")
- `source`: Data source (default: "yahoo_finance")
- `limit`: Max results (default: 10)

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "type": "stock",
    "exchange": "NASDAQ"
  },
  {
    "symbol": "EURUSD=X",
    "name": "EUR/USD",
    "type": "forex",
    "exchange": "FX"
  }
]
```

**Supported Asset Types:**
- **Stocks**: AAPL, MSFT, GOOGL, TSLA, etc.
- **Forex**: EURUSD=X, GBPUSD=X, USDJPY=X, etc.
- **Crypto**: BTC-USD, ETH-USD, XRP-USD, etc.
- **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones), etc.
- **Commodities**: GC=F (Gold), CL=F (Crude Oil), etc.

### 2. Symbol Information

**Endpoint**: `GET /data/symbol-info`

**Parameters:**
- `symbol`: Asset symbol
- `source`: Data source

**Response:**
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "type": "EQUITY",
  "exchange": "NASDAQ",
  "currency": "USD",
  "market_cap": 3000000000000,
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "description": "Apple Inc. designs, manufactures..."
}
```

### 3. Data Import

**Endpoint**: `POST /data/import`

**Parameters:**
- `symbol`: Asset symbol (required)
- `source`: Data source (default: "yahoo_finance")
- `start_date`: Start date in ISO format (optional)
- `end_date`: End date in ISO format (optional)
- `interval`: Data interval (default: "1d")

**Intervals:**
- `1m`: 1 minute (intraday, limited history)
- `5m`: 5 minutes
- `15m`: 15 minutes
- `1h`: 1 hour
- `1d`: 1 day (default)
- `1wk`: 1 week
- `1mo`: 1 month

**Response:**
```json
{
  "symbol": "AAPL",
  "source": "yahoo_finance",
  "interval": "1d",
  "data_points": 252,
  "start_date": "2024-01-01T00:00:00",
  "end_date": "2024-12-31T23:59:59",
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "asset_id": "AAPL",
      "open": 150.25,
      "high": 152.50,
      "low": 149.75,
      "close": 151.80,
      "volume": 50000000
    },
    ...
  ]
}
```

## Frontend Integration

### UI Components

**Location**: Data Workspace (F10) → Left Panel → "IMPORT EXTERNAL DATA"

**Components:**
1. **Toggle Button**: Show/hide import panel
2. **Source Selector**: Choose data provider
3. **Search Input**: Enter symbol query
4. **Search Button**: Execute search
5. **Results List**: Display matching symbols
6. **Date Inputs**: Start and end dates
7. **Interval Selector**: Choose data granularity
8. **Import Button**: Fetch data
9. **Status Display**: Show progress/errors

### User Flow

```
1. User clicks "SHOW IMPORT"
   ↓
2. User selects source (Yahoo Finance)
   ↓
3. User types search query ("AAPL")
   ↓
4. User presses Enter or clicks Search
   ↓
5. Frontend calls GET /data/search
   ↓
6. Results displayed in list
   ↓
7. User clicks on desired symbol
   ↓
8. User sets date range (optional)
   ↓
9. User selects interval (1d)
   ↓
10. User clicks "IMPORT DATA"
    ↓
11. Frontend calls POST /data/import
    ↓
12. Backend fetches from Yahoo Finance
    ↓
13. Data returned to frontend
    ↓
14. Success message displayed
    ↓
15. Data ready for visualization
```

## Backend Implementation

### DataImporter Class

**File**: `intelligence-layer/src/intelligence_layer/data_import.py`

**Methods:**

```python
class DataImporter:
    async def fetch_data(
        source: DataSource,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        interval: str
    ) -> List[MarketData]
    
    async def search_symbols(
        query: str,
        source: DataSource,
        limit: int
    ) -> List[Dict[str, Any]]
    
    async def get_symbol_info(
        symbol: str,
        source: DataSource
    ) -> Dict[str, Any]
```

### Yahoo Finance Integration

Uses `yfinance` library:
- **Installation**: `pip install yfinance`
- **Documentation**: https://pypi.org/project/yfinance/
- **Rate Limits**: Reasonable use, no official limits
- **Data Quality**: High quality, widely used

**Example Usage:**
```python
import yfinance as yf

# Get ticker
ticker = yf.Ticker("AAPL")

# Get historical data
df = ticker.history(
    start="2024-01-01",
    end="2024-12-31",
    interval="1d"
)

# Get info
info = ticker.info
```

## Use Cases

### 1. Import Stock Data for Analysis

```
Goal: Analyze Apple stock performance
Steps:
1. Search "AAPL"
2. Select AAPL from results
3. Set date range: 2024-01-01 to 2024-12-31
4. Interval: 1 day
5. Import
6. Visualize with time series chart
7. Run descriptive statistics
8. Analyze returns distribution
```

### 2. Compare Forex Pairs

```
Goal: Compare EUR/USD and GBP/USD correlation
Steps:
1. Import EURUSD=X (last 30 days, 1h interval)
2. Import GBPUSD=X (last 30 days, 1h interval)
3. Select correlation visualization
4. Run correlation analysis
5. View correlation matrix
6. Export results
```

### 3. Crypto Market Analysis

```
Goal: Analyze Bitcoin volatility
Steps:
1. Search "BTC"
2. Select BTC-USD
3. Import last 90 days, 1d interval
4. Visualize with candlestick chart
5. Run time series analysis
6. Calculate GARCH model
7. Forecast volatility
```

### 4. Multi-Asset Portfolio

```
Goal: Build diversified portfolio
Steps:
1. Import AAPL (stocks)
2. Import EURUSD=X (forex)
3. Import BTC-USD (crypto)
4. Import GC=F (gold)
5. Run correlation analysis
6. Calculate optimal weights
7. Visualize efficient frontier
```

## Error Handling

### Frontend Errors

**No Results:**
```
Status: "No symbols found for query: XYZ"
Action: Try different search term
```

**Import Failed:**
```
Status: "Import failed: No data found for symbol"
Action: Check symbol format, date range
```

**Network Error:**
```
Status: "Import failed: Network error"
Action: Check backend connection
```

### Backend Errors

**Invalid Symbol:**
```python
raise ValueError(f"No data found for symbol: {symbol}")
```

**Date Range Error:**
```python
raise ValueError("Start date must be before end date")
```

**Rate Limit:**
```python
raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

## Performance

### Typical Response Times

- **Symbol Search**: 100-500ms
- **Symbol Info**: 200-800ms
- **Data Import (1 day, 1 year)**: 1-3 seconds
- **Data Import (1 hour, 1 month)**: 2-5 seconds
- **Data Import (1 minute, 1 day)**: 3-8 seconds

### Optimization

- **Async Operations**: Non-blocking I/O
- **Thread Pool**: CPU-bound operations
- **Caching**: Cache symbol info (future)
- **Batch Import**: Multiple symbols (future)

## Security

### API Keys

Yahoo Finance doesn't require API keys for basic usage.

For other sources (future):
- Store API keys in environment variables
- Never expose keys in frontend
- Use backend proxy for all requests

### Rate Limiting

- Implement rate limiting on backend
- Queue requests if needed
- Display wait time to user

### Data Validation

- Validate all inputs
- Sanitize symbol names
- Check date ranges
- Verify data integrity

## Future Enhancements

### Short-term
- [ ] Cache imported data in PostgreSQL
- [ ] Support multiple symbol import
- [ ] Add more data sources (Alpha Vantage, Quandl)
- [ ] Real-time streaming data
- [ ] Custom date presets (YTD, 1Y, 5Y)

### Medium-term
- [ ] Fundamental data import (earnings, ratios)
- [ ] News and sentiment data
- [ ] Economic indicators (FRED)
- [ ] Options data
- [ ] Insider trading data

### Long-term
- [ ] AI-powered symbol recommendations
- [ ] Automated data refresh
- [ ] Data quality scoring
- [ ] Custom data sources
- [ ] Data marketplace integration

## Testing

### Manual Testing

1. **Start Backend**:
   ```bash
   cd intelligence-layer
   pip install -e .
   uvicorn intelligence_layer.main:app --reload
   ```

2. **Test Search**:
   ```bash
   curl "http://localhost:8000/data/search?query=AAPL"
   ```

3. **Test Import**:
   ```bash
   curl -X POST "http://localhost:8000/data/import?symbol=AAPL&interval=1d"
   ```

4. **Test Frontend**:
   ```bash
   cd frontend
   npm run dev
   # Open http://localhost:5173
   # Press F10
   # Click "SHOW IMPORT"
   # Search for "AAPL"
   # Import data
   ```

### Automated Testing

```python
# test_data_import.py
import pytest
from intelligence_layer.data_import import data_importer, DataSource

@pytest.mark.asyncio
async def test_search_symbols():
    results = await data_importer.search_symbols("AAPL", DataSource.YAHOO_FINANCE, 10)
    assert len(results) > 0
    assert results[0]['symbol'] == 'AAPL'

@pytest.mark.asyncio
async def test_import_data():
    data = await data_importer.fetch_data(
        DataSource.YAHOO_FINANCE,
        "AAPL",
        None,
        None,
        "1d"
    )
    assert len(data) > 0
    assert data[0].asset_id == "AAPL"
```

## Documentation

- **User Guide**: `frontend/DATA_WORKSPACE_GUIDE.md`
- **API Docs**: http://localhost:8000/docs
- **Integration Guide**: `INTEGRATION_GUIDE.md`
- **This Document**: `DATA_IMPORT_FEATURE.md`

## Support

For issues:
1. Check symbol format (Yahoo Finance format)
2. Verify date range is valid
3. Check backend logs
4. Review API documentation
5. Test with curl/Postman

## Conclusion

The external data import feature provides a seamless way to bring market data from Yahoo Finance and other sources into the trading platform. Users can search, select, and import data with just a few clicks, then immediately analyze it using the full suite of visualization and analysis tools in the Data Workspace.

This feature bridges the gap between external market data and the platform's internal analytics, enabling comprehensive market research and strategy development.
