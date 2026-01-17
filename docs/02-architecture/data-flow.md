# Markets Page (F2) - Full-Stack Data Architecture

## Overview

The Markets page (F2) collects and processes real-time market data using a hybrid Rust/Python architecture for maximum robustness, performance, and analytical capability.

## Implementation Status

✅ **COMPLETED:**
- Rust market data structures (`execution-core/src/market_data.rs`)
- Python market analytics engine (`intelligence-layer/src/intelligence_layer/market_analytics.py`)
- API endpoints in Intelligence Layer (`/markets/*`)
- Frontend markets service (`frontend/src/services/marketsService.ts`)
- Updated Markets component with real backend integration
- Correlation matrix calculation with clustering
- Volatility metrics calculation
- Anomaly detection
- Market regime indicators
- Live data polling (5-second refresh)

🚧 **IN PROGRESS:**
- WebSocket streaming for real-time updates
- Integration with actual exchange data feeds
- Database persistence for historical data

📋 **TODO:**
- Connect Rust market data collector to real exchanges
- Implement time-series database writer
- Add Redis caching layer
- Implement Neo4j graph relationships
- Add property-based tests for market data processing

## Data Collection Strategy

### 1. Real-Time Market Data (Rust - Execution Core)

**Why Rust:**
- Sub-millisecond latency requirements
- Zero-copy data structures
- Memory safety for 24/7 operation
- Concurrent processing without GIL

**Data to Collect:**

```rust
// execution-core/src/market_data.rs

pub struct MarketTick {
    pub timestamp: i64,           // Nanosecond precision
    pub asset_id: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub last_size: f64,
    pub volume: f64,
    pub sequence_number: u64,     // For gap detection
    pub exchange: String,
    pub latency_us: u32,          // Capture latency
}

pub struct OrderBookSnapshot {
    pub timestamp: i64,
    pub asset_id: String,
    pub bids: Vec<(f64, f64)>,    // Price, Size
    pub asks: Vec<(f64, f64)>,
    pub depth_levels: usize,
    pub checksum: u64,            // Data integrity
}

pub struct MarketMicrostructure {
    pub timestamp: i64,
    pub asset_id: String,
    pub spread_bps: f64,
    pub effective_spread_bps: f64,
    pub quoted_spread_bps: f64,
    pub depth_bid: f64,           // Cumulative depth
    pub depth_ask: f64,
    pub imbalance_ratio: f64,     // Bid/Ask imbalance
    pub tick_frequency: f64,      // Ticks per second
    pub price_impact_bps: f64,    // For standard size
}

pub struct LiquidityMetrics {
    pub timestamp: i64,
    pub asset_id: String,
    pub bid_liquidity_usd: f64,
    pub ask_liquidity_usd: f64,
    pub total_liquidity_usd: f64,
    pub liquidity_score: f64,     // 0-100
    pub resilience_score: f64,    // How fast liquidity recovers
    pub toxicity_score: f64,      // Adverse selection risk
}
```

### 2. Market Analytics (Python - Intelligence Layer)

**Why Python:**
- Rich ecosystem for statistical analysis
- NumPy/Pandas for vectorized operations
- SciPy for advanced analytics
- Easy integration with ML models

**Data to Collect:**

```python
# intelligence-layer/src/intelligence_layer/market_analytics.py

@dataclass
class CorrelationMatrix:
    timestamp: datetime
    assets: List[str]
    correlation_matrix: np.ndarray  # NxN matrix
    rolling_window: str             # e.g., "24H"
    method: str                     # pearson, spearman, kendall
    significance: np.ndarray        # p-values
    
@dataclass
class VolatilityMetrics:
    timestamp: datetime
    asset_id: str
    realized_vol: float             # Historical volatility
    implied_vol: float              # From options (if available)
    parkinson_vol: float            # High-low estimator
    garman_klass_vol: float         # OHLC estimator
    vol_of_vol: float               # Volatility clustering
    vol_regime: str                 # LOW, MEDIUM, HIGH
    
@dataclass
class MarketRegimeIndicators:
    timestamp: datetime
    asset_id: str
    trend_strength: float           # ADX-like
    trend_direction: str            # UP, DOWN, SIDEWAYS
    volatility_regime: str          # LOW, MEDIUM, HIGH
    liquidity_regime: str           # NORMAL, STRESSED, CRISIS
    correlation_regime: str         # NORMAL, BREAKDOWN, CRISIS
    regime_probability: float       # Confidence
    
@dataclass
class MarketAnomalies:
    timestamp: datetime
    asset_id: str
    anomaly_type: str               # PRICE_SPIKE, VOLUME_SURGE, etc.
    severity: float                 # 0-1
    z_score: float
    description: str
    recommended_action: str         # ALERT, PAUSE, INVESTIGATE
```

### 3. Market Events (Rust - Event Bus)

**Why Rust:**
- High-throughput event processing
- Lock-free data structures
- Guaranteed delivery semantics

```rust
// execution-core/src/market_events.rs

pub enum MarketEvent {
    PriceUpdate {
        asset_id: String,
        price: f64,
        change_pct: f64,
        timestamp: i64,
    },
    VolatilitySpike {
        asset_id: String,
        current_vol: f64,
        baseline_vol: f64,
        spike_factor: f64,
        timestamp: i64,
    },
    LiquidityDrop {
        asset_id: String,
        current_liquidity: f64,
        baseline_liquidity: f64,
        drop_pct: f64,
        timestamp: i64,
    },
    CorrelationBreakdown {
        asset_pair: (String, String),
        current_corr: f64,
        historical_corr: f64,
        deviation: f64,
        timestamp: i64,
    },
    CircuitBreaker {
        asset_id: String,
        reason: String,
        duration_sec: u32,
        timestamp: i64,
    },
}
```

## Backend Architecture

### Rust Components (Execution Core)

```
┌─────────────────────────────────────────────────────────────┐
│                    Execution Core (Rust)                     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Market Data Collector                                  │ │
│  │  - WebSocket connections to exchanges                   │ │
│  │  - Sub-millisecond tick processing                      │ │
│  │  - Order book reconstruction                            │ │
│  │  - Gap detection & recovery                             │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Market Microstructure Engine                           │ │
│  │  - Real-time spread calculation                         │ │
│  │  - Depth analysis                                       │ │
│  │  - Liquidity scoring                                    │ │
│  │  - Price impact estimation                              │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Event Bus                                              │ │
│  │  - Lock-free queue (crossbeam)                          │ │
│  │  - Event filtering & routing                            │ │
│  │  - Guaranteed delivery                                  │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Time-Series Database Writer                            │ │
│  │  - Batch writes to PostgreSQL                           │ │
│  │  - Compression & partitioning                           │ │
│  │  - Write-ahead logging                                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Python Components (Intelligence Layer)

```
┌─────────────────────────────────────────────────────────────┐
│                Intelligence Layer (Python)                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Market Analytics Engine                                │ │
│  │  - Correlation analysis (NumPy)                         │ │
│  │  - Volatility modeling (GARCH)                          │ │
│  │  - Regime detection (HMM)                               │ │
│  │  - Anomaly detection (Isolation Forest)                 │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Feature Engineering                                    │ │
│  │  - Technical indicators                                 │ │
│  │  - Statistical features                                 │ │
│  │  - Graph features (Neo4j)                               │ │
│  │  - Embedding generation                                 │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  ML Models                                              │ │
│  │  - Volatility forecasting                               │ │
│  │  - Correlation prediction                               │ │
│  │  - Liquidity prediction                                 │ │
│  │  - Regime classification                                │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  API Endpoints (FastAPI)                                │ │
│  │  - GET /markets/live-data                               │ │
│  │  - GET /markets/correlations                            │ │
│  │  - GET /markets/microstructure                          │ │
│  │  - GET /markets/liquidity                               │ │
│  │  - GET /markets/events                                  │ │
│  │  - WS /markets/stream                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints Design

### 1. Live Market Data

```python
@app.get("/markets/live-data")
async def get_live_market_data(
    assets: str,  # Comma-separated
    include_depth: bool = False,
) -> Dict[str, Any]:
    """
    Get real-time market data from Rust execution core.
    
    Returns:
    - Current bid/ask/last
    - Volume
    - Spread metrics
    - Optional: Order book depth
    """
    pass
```

### 2. Correlation Matrix

```python
@app.get("/markets/correlations")
async def get_correlation_matrix(
    assets: str,
    window: str = "24H",  # 1H, 4H, 24H, 7D, 30D
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Calculate rolling correlation matrix.
    
    Returns:
    - Correlation matrix
    - Significance levels
    - Correlation changes
    - Cluster analysis
    """
    pass
```

### 3. Market Microstructure

```python
@app.get("/markets/microstructure")
async def get_market_microstructure(
    asset_id: str,
) -> Dict[str, Any]:
    """
    Get real-time microstructure metrics from Rust.
    
    Returns:
    - Spread (bid-ask, effective, quoted)
    - Depth (bid, ask, total)
    - Imbalance ratio
    - Tick frequency
    - Price impact
    """
    pass
```

### 4. Liquidity Analysis

```python
@app.get("/markets/liquidity")
async def get_liquidity_analysis(
    assets: str,
) -> Dict[str, Any]:
    """
    Analyze market liquidity.
    
    Returns:
    - Liquidity scores
    - Resilience metrics
    - Toxicity indicators
    - Historical comparison
    """
    pass
```

### 5. Market Events

```python
@app.get("/markets/events")
async def get_market_events(
    since: Optional[datetime] = None,
    event_types: Optional[str] = None,
    severity_min: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Get recent market events.
    
    Returns:
    - Price spikes
    - Volatility events
    - Liquidity drops
    - Correlation breakdowns
    - Circuit breakers
    """
    pass
```

### 6. WebSocket Stream

```python
@app.websocket("/markets/stream")
async def market_data_stream(
    websocket: WebSocket,
    assets: str,
    include_events: bool = True,
):
    """
    Stream real-time market data.
    
    Streams:
    - Tick data
    - Microstructure updates
    - Market events
    - Liquidity changes
    """
    pass
```

## Frontend Integration

### Updated Markets Component

```typescript
// frontend/src/services/marketsService.ts

export interface LiveMarketData {
  asset_id: string;
  timestamp: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  spread_bps: number;
  depth_bid: number;
  depth_ask: number;
  tick_frequency: number;
}

export interface CorrelationData {
  timestamp: string;
  assets: string[];
  correlation_matrix: number[][];
  significance: number[][];
  clusters: string[][];
}

export interface MicrostructureData {
  asset_id: string;
  timestamp: string;
  spread_bps: number;
  effective_spread_bps: number;
  quoted_spread_bps: number;
  depth_bid: number;
  depth_ask: number;
  imbalance_ratio: number;
  tick_frequency: number;
  price_impact_bps: number;
}

export interface LiquidityData {
  asset_id: string;
  timestamp: string;
  bid_liquidity_usd: number;
  ask_liquidity_usd: number;
  total_liquidity_usd: number;
  liquidity_score: number;
  resilience_score: number;
  toxicity_score: number;
}

export interface MarketEvent {
  timestamp: string;
  asset_id: string;
  event_type: string;
  severity: number;
  description: string;
  recommended_action: string;
}

// API calls
export async function getLiveMarketData(
  assets: string[],
  includeDepth: boolean = false
): Promise<LiveMarketData[]> {
  return intelligenceApi.get(
    `/markets/live-data?assets=${assets.join(',')}&include_depth=${includeDepth}`
  );
}

export async function getCorrelationMatrix(
  assets: string[],
  window: string = '24H',
  method: string = 'pearson'
): Promise<CorrelationData> {
  return intelligenceApi.get(
    `/markets/correlations?assets=${assets.join(',')}&window=${window}&method=${method}`
  );
}

export async function getMicrostructure(
  assetId: string
): Promise<MicrostructureData> {
  return intelligenceApi.get(`/markets/microstructure?asset_id=${assetId}`);
}

export async function getLiquidityAnalysis(
  assets: string[]
): Promise<LiquidityData[]> {
  return intelligenceApi.get(`/markets/liquidity?assets=${assets.join(',')}`);
}

export async function getMarketEvents(
  since?: string,
  eventTypes?: string[],
  severityMin: number = 0.5
): Promise<MarketEvent[]> {
  const params = new URLSearchParams();
  if (since) params.append('since', since);
  if (eventTypes) params.append('event_types', eventTypes.join(','));
  params.append('severity_min', severityMin.toString());
  
  return intelligenceApi.get(`/markets/events?${params.toString()}`);
}

// WebSocket connection
export function connectMarketStream(
  assets: string[],
  onData: (data: any) => void,
  onEvent: (event: MarketEvent) => void
): WebSocket {
  const ws = new WebSocket(
    `ws://localhost:8000/markets/stream?assets=${assets.join(',')}`
  );
  
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'tick') {
      onData(message.data);
    } else if (message.type === 'event') {
      onEvent(message.data);
    }
  };
  
  return ws;
}
```

## Data Storage Strategy

### PostgreSQL (Time-Series Data)

```sql
-- Market ticks (high-frequency)
CREATE TABLE market_ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(20) NOT NULL,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    last_price DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    sequence_number BIGINT,
    latency_us INTEGER
) PARTITION BY RANGE (timestamp);

-- Create partitions by day
CREATE INDEX idx_market_ticks_asset_time ON market_ticks (asset_id, timestamp DESC);

-- Market microstructure (1-second aggregates)
CREATE TABLE market_microstructure (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(20) NOT NULL,
    spread_bps DOUBLE PRECISION,
    depth_bid DOUBLE PRECISION,
    depth_ask DOUBLE PRECISION,
    imbalance_ratio DOUBLE PRECISION,
    tick_frequency DOUBLE PRECISION,
    PRIMARY KEY (timestamp, asset_id)
) PARTITION BY RANGE (timestamp);

-- Liquidity metrics (1-minute aggregates)
CREATE TABLE liquidity_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(20) NOT NULL,
    bid_liquidity_usd DOUBLE PRECISION,
    ask_liquidity_usd DOUBLE PRECISION,
    liquidity_score DOUBLE PRECISION,
    resilience_score DOUBLE PRECISION,
    toxicity_score DOUBLE PRECISION,
    PRIMARY KEY (timestamp, asset_id)
);

-- Market events
CREATE TABLE market_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(20),
    event_type VARCHAR(50) NOT NULL,
    severity DOUBLE PRECISION,
    description TEXT,
    metadata JSONB,
    INDEX idx_events_time (timestamp DESC),
    INDEX idx_events_asset (asset_id, timestamp DESC)
);
```

### Neo4j (Relationship Data)

```cypher
// Asset correlation network
CREATE (a:Asset {id: 'EURUSD', name: 'EUR/USD'})
CREATE (b:Asset {id: 'GBPUSD', name: 'GBP/USD'})
CREATE (a)-[:CORRELATES_WITH {
    correlation: 0.75,
    window: '24H',
    timestamp: datetime(),
    significance: 0.001
}]->(b)

// Market regime transitions
CREATE (r1:Regime {name: 'LOW_VOL_TRENDING'})
CREATE (r2:Regime {name: 'HIGH_VOL_RANGING'})
CREATE (r1)-[:TRANSITIONS_TO {
    probability: 0.15,
    avg_duration_hours: 4.5,
    trigger_conditions: ['volatility_spike', 'news_event']
}]->(r2)
```

### Redis (Real-Time Cache)

```python
# Cache latest market data
redis.setex(
    f"market:tick:{asset_id}",
    60,  # 60 second TTL
    json.dumps(tick_data)
)

# Cache correlation matrix
redis.setex(
    f"market:corr:24H",
    300,  # 5 minute TTL
    json.dumps(correlation_matrix)
)

# Pub/Sub for real-time updates
redis.publish(
    f"market:events",
    json.dumps(market_event)
)
```

## Performance Optimizations

### Rust Optimizations

1. **Zero-Copy Deserialization**
```rust
use serde_json::from_slice;
use bytes::Bytes;

// Avoid string allocation
let tick: MarketTick = from_slice(&bytes)?;
```

2. **Lock-Free Data Structures**
```rust
use crossbeam::queue::ArrayQueue;

let tick_queue = ArrayQueue::new(10000);
// Producer
tick_queue.push(tick);
// Consumer
if let Some(tick) = tick_queue.pop() {
    process_tick(tick);
}
```

3. **SIMD for Calculations**
```rust
use packed_simd::f64x4;

// Vectorized spread calculation
let bids = f64x4::new(bid1, bid2, bid3, bid4);
let asks = f64x4::new(ask1, ask2, ask3, ask4);
let spreads = asks - bids;
```

### Python Optimizations

1. **NumPy Vectorization**
```python
# Instead of loops
correlations = np.corrcoef(returns.T)
```

2. **Numba JIT Compilation**
```python
from numba import jit

@jit(nopython=True)
def calculate_volatility(prices):
    returns = np.diff(np.log(prices))
    return np.std(returns) * np.sqrt(252)
```

3. **Async I/O**
```python
async def fetch_multiple_assets(assets):
    tasks = [fetch_asset_data(asset) for asset in assets]
    return await asyncio.gather(*tasks)
```

## Monitoring & Alerting

### Metrics to Track

```python
# Data quality metrics
- Tick gap rate (should be < 0.1%)
- Latency percentiles (P50, P95, P99)
- Data completeness (% of expected ticks)
- Sequence number gaps

# Performance metrics
- Processing throughput (ticks/second)
- Memory usage
- CPU usage
- Database write latency

# Business metrics
- Spread width (bps)
- Liquidity depth (USD)
- Market event frequency
- Correlation stability
```

## Benefits of This Architecture

### Robustness
1. **Fault Tolerance**: Rust's memory safety prevents crashes
2. **Data Integrity**: Checksums and sequence numbers detect corruption
3. **Graceful Degradation**: System continues with reduced functionality
4. **Automatic Recovery**: Gap detection and backfilling

### Performance
1. **Low Latency**: Rust processes ticks in microseconds
2. **High Throughput**: Handle 100K+ ticks/second
3. **Efficient Memory**: Zero-copy and arena allocation
4. **Scalability**: Horizontal scaling with load balancing

### Analytics
1. **Rich Insights**: Python's ML ecosystem
2. **Real-Time Analysis**: Streaming analytics
3. **Historical Analysis**: Time-series database
4. **Predictive Models**: Volatility and liquidity forecasting

### Maintainability
1. **Type Safety**: Rust and TypeScript catch errors at compile time
2. **Clear Separation**: Rust for speed, Python for intelligence
3. **Testability**: Property-based testing in both languages
4. **Documentation**: Auto-generated API docs

This architecture makes the Markets page a robust, high-performance, analytically-rich component that can handle institutional-grade trading requirements.
