# Markets Page (F2) - Implementation Guide

## Overview

The Markets page provides real-time market data, analytics, and visualization for trading assets. It uses a hybrid Rust/Python architecture for optimal performance and analytical capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  - Markets Component (F2)                                    │
│  - Markets Service (TypeScript)                              │
│  - Auto-refresh (5 seconds)                                  │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Intelligence Layer (Python/FastAPI)             │
│  - /markets/live-data                                        │
│  - /markets/correlations                                     │
│  - /markets/microstructure                                   │
│  - /markets/liquidity                                        │
│  - /markets/events                                           │
│  - Market Analytics Engine                                   │
└────────────────────┬────────────────────────────────────────┘
                     │ (Future: IPC/gRPC)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Execution Core (Rust)                           │
│  - Market Data Collector                                     │
│  - Microstructure Engine                                     │
│  - Event Bus                                                 │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. Live Market Data
- Real-time bid/ask/last prices
- Volume tracking
- Spread calculation
- Tick frequency monitoring
- Optional order book depth
- **Automatic fallback to mock data when offline**

### 2. Correlation Analysis
- Multi-asset correlation matrix
- Statistical significance testing
- Hierarchical clustering
- Multiple methods (Pearson, Spearman, Kendall)
- Configurable rolling windows (1H, 4H, 24H, 7D, 30D)

### 3. Market Microstructure
- Bid-ask spread (BPS)
- Effective and quoted spreads
- Market depth (bid/ask)
- Order imbalance ratio
- Tick frequency
- Price impact estimation

### 4. Liquidity Analysis
- Liquidity in USD (bid, ask, total)
- Liquidity score (0-100)
- Resilience score
- Toxicity score
- Historical comparison

### 5. Market Events
- Volatility spikes
- Liquidity drops
- Price anomalies
- Correlation breakdowns
- Circuit breakers
- Severity-based filtering

### 6. User Engagement (CRUD Operations)
- **Watchlist Management:**
  - Create custom watchlists
  - Edit watchlist names and assets
  - Delete watchlists
  - Switch between watchlists
  - Tab-based navigation
  
- **Alert Management:**
  - Create price/volatility/liquidity/correlation alerts
  - Set conditions (above/below) and thresholds
  - Enable/disable alerts
  - Edit alert parameters
  - Delete alerts
  - Active alert count display

### 7. Offline Mode
- **Automatic Mock Data Fallback:**
  - Detects backend disconnection
  - Switches to deterministic mock data
  - Visual indicators (MOCK DATA MODE, OFFLINE status)
  - All features remain functional
  - Automatic reconnection when backend available

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Rust 1.70+ (for building)

### 1. Start Backend

```bash
# Navigate to intelligence layer
cd intelligence-layer

# Install dependencies (if not already installed)
pip install -e .

# Start the API server
python -m intelligence_layer.main
```

The backend will start on `http://localhost:8000`

### 2. Start Frontend

```bash
# Navigate to frontend
cd frontend

# Install dependencies (if not already installed)
npm install

# Start development server
npm run dev
```

The frontend will start on `http://localhost:5173`

### 3. Access Markets Page

- Open browser to `http://localhost:5173`
- Press **F2** or click **Markets** in the navigation
- Data will auto-refresh every 5 seconds

### 4. Manage Watchlists and Alerts

**Create a Watchlist:**
1. Click "WATCHLISTS" button in header
2. Fill in name and comma-separated assets
3. Click "CREATE"

**Create an Alert:**
1. Click "ALERTS" button in header
2. Select asset, type, condition, and threshold
3. Click "CREATE"

**Switch Watchlists:**
- Click on watchlist tabs below the header
- Data automatically updates for selected watchlist

### 5. Offline Mode

If backend is not running:
- Markets page automatically uses mock data
- Yellow "MOCK DATA MODE" indicator appears
- Red "● OFFLINE" status shown
- All features continue to work
- Reconnects automatically when backend starts

## API Endpoints

### GET /markets/live-data

Get real-time market data for assets.

**Parameters:**
- `assets` (required): Comma-separated list of asset IDs (e.g., "EURUSD,GBPUSD")
- `include_depth` (optional): Include order book depth (default: false)

**Example:**
```bash
curl "http://localhost:8000/markets/live-data?assets=EURUSD,GBPUSD&include_depth=true"
```

**Response:**
```json
{
  "data": [
    {
      "asset_id": "EURUSD",
      "timestamp": "2024-01-15T14:30:00Z",
      "bid": 1.0845,
      "ask": 1.0847,
      "last": 1.0846,
      "volume": 1250000,
      "spread_bps": 2.0,
      "depth_bid": 2500000.0,
      "depth_ask": 2300000.0,
      "tick_frequency": 125
    }
  ],
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### GET /markets/correlations

Calculate rolling correlation matrix for assets.

**Parameters:**
- `assets` (required): Comma-separated list of asset IDs
- `window` (optional): Rolling window (1H, 4H, 24H, 7D, 30D) (default: 24H)
- `method` (optional): Correlation method (pearson, spearman, kendall) (default: pearson)

**Example:**
```bash
curl "http://localhost:8000/markets/correlations?assets=EURUSD,GBPUSD,USDJPY&window=24H&method=pearson"
```

**Response:**
```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "assets": ["EURUSD", "GBPUSD", "USDJPY"],
  "correlation_matrix": [
    [1.0, 0.75, -0.45],
    [0.75, 1.0, -0.38],
    [-0.45, -0.38, 1.0]
  ],
  "significance": [...],
  "window": "24H",
  "method": "pearson",
  "clusters": [["EURUSD", "GBPUSD"], ["USDJPY"]]
}
```

### GET /markets/microstructure

Get market microstructure metrics for an asset.

**Parameters:**
- `asset_id` (required): Asset identifier

**Example:**
```bash
curl "http://localhost:8000/markets/microstructure?asset_id=EURUSD"
```

**Response:**
```json
{
  "asset_id": "EURUSD",
  "timestamp": "2024-01-15T14:30:00Z",
  "spread_bps": 1.8,
  "effective_spread_bps": 1.6,
  "quoted_spread_bps": 2.0,
  "depth_bid": 2500000.0,
  "depth_ask": 2300000.0,
  "imbalance_ratio": 0.52,
  "tick_frequency": 125.0,
  "price_impact_bps": 0.9
}
```

### GET /markets/liquidity

Analyze market liquidity for assets.

**Parameters:**
- `assets` (required): Comma-separated list of asset IDs

**Example:**
```bash
curl "http://localhost:8000/markets/liquidity?assets=EURUSD,GBPUSD"
```

**Response:**
```json
{
  "data": [
    {
      "asset_id": "EURUSD",
      "timestamp": "2024-01-15T14:30:00Z",
      "bid_liquidity_usd": 2500000.0,
      "ask_liquidity_usd": 2300000.0,
      "total_liquidity_usd": 4800000.0,
      "liquidity_score": 85.0,
      "resilience_score": 78.0,
      "toxicity_score": 22.0
    }
  ],
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### GET /markets/events

Get recent market events.

**Parameters:**
- `since` (optional): ISO timestamp to get events since
- `event_types` (optional): Comma-separated list of event types
- `severity_min` (optional): Minimum severity (0-1) (default: 0.5)

**Example:**
```bash
curl "http://localhost:8000/markets/events?severity_min=0.6"
```

**Response:**
```json
[
  {
    "timestamp": "2024-01-15T14:15:00Z",
    "asset_id": "EURUSD",
    "event_type": "VOLATILITY_SPIKE",
    "severity": 0.7,
    "description": "Volatility increased by 45% above baseline",
    "recommended_action": "ALERT"
  }
]
```

## Testing

### Manual Testing

1. **Test API Endpoints:**
```bash
python test_markets_api.py
```

2. **Test Frontend:**
- Open browser to `http://localhost:5173`
- Press F2 to open Markets page
- Verify data loads and refreshes

### Automated Testing

```bash
# Test Rust code
cd execution-core
cargo test market_data

# Test Python code
cd intelligence-layer
pytest tests/test_market_analytics.py

# Test Frontend
cd frontend
npm test
```

## Configuration

### Backend Configuration

Edit `intelligence-layer/src/intelligence_layer/config.py`:

```python
# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Market data settings
MARKET_DATA_REFRESH_INTERVAL = 1  # seconds
MARKET_DATA_HISTORY_SIZE = 1000   # data points
```

### Frontend Configuration

Edit `frontend/src/services/api.ts`:

```typescript
// API base URLs
const INTELLIGENCE_API_URL = 'http://localhost:8000';
const EXECUTION_API_URL = 'http://localhost:8080';

// Polling interval
const POLLING_INTERVAL = 5000; // milliseconds
```

## Troubleshooting

### Backend Not Starting

**Problem:** `ModuleNotFoundError: No module named 'yfinance'`

**Solution:**
```bash
cd intelligence-layer
pip install -e .
```

### Frontend Not Connecting

**Problem:** `Failed to fetch market data`

**Solution:**
The frontend will automatically fall back to mock data. To restore live data:
1. Check backend is running: `curl http://localhost:8000/health`
2. Check CORS settings in `intelligence-layer/src/intelligence_layer/main.py`
3. Verify frontend API URL in `frontend/src/services/api.ts`
4. Look for yellow "MOCK DATA MODE" indicator - this is normal when backend is offline

### No Data Showing

**Problem:** Markets page shows "Loading..." indefinitely

**Solution:**
1. Open browser console (F12)
2. Check for network errors
3. If backend is offline, mock data should load automatically
4. Check backend logs for errors if using live data

### Watchlist/Alert Not Saving

**Problem:** Changes to watchlists or alerts are lost on refresh

**Solution:**
Currently, watchlists and alerts are stored in component state (in-memory). To persist:
1. Add localStorage integration (future enhancement)
2. Or connect to backend API for persistence (future enhancement)

### Slow Performance

**Problem:** Data takes long to load

**Solution:**
1. Reduce number of assets in watchlist
2. Increase polling interval
3. Enable caching in backend
4. Check network latency
5. Use mock data mode for testing (instant response)

## Development

### Adding New Metrics

1. **Add to Rust (if performance-critical):**
```rust
// execution-core/src/market_data.rs
pub struct NewMetric {
    pub timestamp: i64,
    pub value: f64,
}
```

2. **Add to Python (if analytical):**
```python
# intelligence-layer/src/intelligence_layer/market_analytics.py
@dataclass
class NewMetric:
    timestamp: datetime
    value: float
```

3. **Add API endpoint:**
```python
# intelligence-layer/src/intelligence_layer/main.py
@app.get("/markets/new-metric")
async def get_new_metric(asset_id: str):
    # Implementation
    pass
```

4. **Add to frontend service:**
```typescript
// frontend/src/services/marketsService.ts
export async function getNewMetric(assetId: string) {
  return intelligenceApi.get(`/markets/new-metric?asset_id=${assetId}`);
}
```

5. **Update component:**
```typescript
// frontend/src/app/components/Markets.tsx
const [newMetric, setNewMetric] = useState(null);
// Fetch and display
```

### Adding New Assets

Edit the watched assets list in `Markets.tsx`:

```typescript
const watchedAssets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'BTCUSD', 'ETHUSD'];
```

## Performance Benchmarks

### Backend
- API response time: < 50ms (P95)
- Correlation calculation: < 100ms for 10 assets
- Throughput: > 1000 requests/second

### Frontend
- Initial load: < 2 seconds
- Refresh cycle: < 500ms
- Memory usage: < 100MB

## Future Enhancements

### Short Term
- [ ] Persist watchlists and alerts to localStorage
- [ ] WebSocket streaming for real-time updates
- [ ] Historical data charts
- [ ] Export data to CSV/JSON
- [ ] Alert notifications (browser notifications)
- [ ] Alert history log

### Medium Term
- [ ] Backend API for watchlist/alert persistence
- [ ] Connect to real exchange feeds
- [ ] Database persistence
- [ ] Redis caching layer
- [ ] Advanced filtering and search
- [ ] Custom alert actions (email, webhook)

### Long Term
- [ ] Machine learning predictions
- [ ] Custom indicators
- [ ] Backtesting integration
- [ ] Multi-timeframe analysis
- [ ] Alert backtesting
- [ ] Social features (share watchlists)

## Support

For issues or questions:
1. Check this README
2. Review `MARKETS_DATA_ARCHITECTURE.md`
3. Check `MARKETS_IMPLEMENTATION_SUMMARY.md`
4. Review API documentation at `http://localhost:8000/docs`

## License

MIT License - See LICENSE file for details
