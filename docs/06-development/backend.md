# Backend Development Guide

This guide covers the Rust and Python backend development for the algorithmic trading platform.

## Architecture Overview

The backend consists of three main components:

### 1. Execution Core (Rust) - Port 8001
- **Purpose**: High-performance order execution, portfolio accounting, risk management
- **Language**: Rust for memory safety and performance
- **Key Features**: Event-driven architecture, deterministic replay, emergency halt

### 2. Intelligence Layer (Python) - Port 8000
- **Purpose**: Machine learning, market analysis, regime detection
- **Language**: Python for ML ecosystem compatibility
- **Key Features**: FastAPI, async/await, pgvector integration, Neo4j GDS

### 3. Simulation Engine (Rust) - Port 8002
- **Purpose**: Backtesting, scenario analysis, deterministic simulation
- **Language**: Rust for reproducible results
- **Key Features**: Event-driven backtesting, clock abstraction

## Execution Core (Rust)

### Project Structure
```
execution-core/
├── src/
│   ├── lib.rs              # Core traits and interfaces
│   ├── main.rs             # Application entry point
│   ├── config.rs           # Configuration management
│   ├── event_bus.rs        # Event-driven architecture
│   ├── portfolio.rs        # Portfolio accounting
│   ├── risk.rs             # Risk management
│   ├── execution_manager.rs # Order execution
│   ├── execution_adapter.rs # Adapter pattern for brokers
│   ├── shadow_execution.rs  # Paper trading mode
│   ├── health.rs           # Health checks
│   └── shutdown.rs         # Graceful shutdown
├── Cargo.toml
└── Dockerfile
```

### Key Components

#### Event Bus System
```rust
// Event-driven architecture for deterministic replay
pub trait EventBus {
    fn publish(&mut self, event: Event) -> Result<(), EventError>;
    fn subscribe<H: EventHandler>(&mut self, handler: H);
    fn replay(&self, from: Timestamp) -> Vec<Event>;
}

pub enum Event {
    MarketData(MarketDataEvent),
    OrderPlaced(OrderEvent),
    OrderFilled(FillEvent),
    RiskBreach(RiskEvent),
    SystemShutdown,
}
```

#### Portfolio Management
```rust
pub struct Portfolio {
    positions: HashMap<AssetId, Position>,
    cash: Money,
    unrealized_pnl: Money,
    realized_pnl: Money,
}

impl Portfolio {
    pub fn update_position(&mut self, fill: &Fill) -> Result<(), PortfolioError>;
    pub fn calculate_risk_metrics(&self) -> RiskMetrics;
    pub fn get_exposure(&self) -> ExposureBreakdown;
}
```

#### Risk Management
```rust
pub struct RiskManager {
    limits: RiskLimits,
    metrics: RiskMetrics,
}

impl RiskManager {
    pub fn check_order(&self, order: &Order) -> Result<(), RiskError>;
    pub fn update_metrics(&mut self, portfolio: &Portfolio);
    pub fn emergency_halt(&mut self) -> Result<(), RiskError>;
}
```

### Development Setup

#### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# Install required tools
cargo install cargo-watch
cargo install cargo-audit
```

#### Development Commands
```bash
# Run in development mode
cargo run

# Watch for changes
cargo watch -x run

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run

# Check code quality
cargo fmt
cargo clippy
cargo audit
```

#### Configuration
```toml
# Cargo.toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio-rustls"] }
redis = "0.24"
uuid = { version = "1.0", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
```

### API Endpoints

#### Health Check
```rust
#[get("/health")]
async fn health() -> impl Responder {
    Json(HealthStatus {
        status: "healthy".to_string(),
        timestamp: Utc::now(),
        components: check_components().await,
    })
}
```

#### Portfolio Endpoints
```rust
#[get("/portfolio/positions")]
async fn get_positions() -> Result<Json<Vec<Position>>, ApiError>;

#[get("/portfolio/metrics")]
async fn get_risk_metrics() -> Result<Json<RiskMetrics>, ApiError>;

#[post("/portfolio/emergency-halt")]
async fn emergency_halt() -> Result<Json<HaltResponse>, ApiError>;
```

## Intelligence Layer (Python)

### Project Structure
```
intelligence-layer/
├── src/intelligence_layer/
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── config.py           # Configuration
│   ├── models.py           # Pydantic models
│   ├── logging.py          # Structured logging
│   ├── embedding_model.py  # TCN/VAE models
│   ├── regime_detection.py # HMM regime detection
│   ├── graph_analytics.py  # Neo4j GDS integration
│   ├── market_analytics.py # Market analysis
│   ├── strategy_registry.py # Strategy management
│   ├── model_registry.py   # Model versioning
│   ├── data_import.py      # External data import
│   └── deriv_integration.py # Deriv API integration
├── tests/
├── pyproject.toml
└── Dockerfile
```

### Key Components

#### FastAPI Application
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Intelligence Layer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

#### Embedding Models
```python
class TCNEmbeddingModel:
    def __init__(self, config: ModelConfig):
        self.model = self._build_tcn_model(config)
        
    def embed_market_state(self, market_data: MarketData) -> np.ndarray:
        """Generate market state embedding using TCN."""
        features = self._extract_features(market_data)
        embedding = self.model.predict(features)
        return embedding
        
    def _build_tcn_model(self, config: ModelConfig):
        # Temporal Convolutional Network implementation
        pass
```

#### Regime Detection
```python
class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.model = GaussianHMM(n_components=n_regimes)
        
    def detect_regime(self, market_data: MarketData) -> RegimeInference:
        """Detect current market regime using HMM."""
        features = self._prepare_features(market_data)
        probabilities = self.model.predict_proba(features)
        
        return RegimeInference(
            regime_probabilities=probabilities[-1],
            confidence=np.max(probabilities[-1]),
            transition_matrix=self.model.transmat_
        )
```

#### Neo4j Integration
```python
class GraphAnalytics:
    def __init__(self, neo4j_uri: str, auth: tuple):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        
    async def get_asset_correlations(self, asset_id: str) -> List[AssetCorrelation]:
        """Get asset correlations from Neo4j graph."""
        query = """
        MATCH (a:Asset {id: $asset_id})-[r:CORRELATES_WITH]->(b:Asset)
        RETURN b.id as asset, r.correlation as correlation
        ORDER BY r.correlation DESC
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, asset_id=asset_id)
            return [AssetCorrelation(**record) for record in result]
```

### Development Setup

#### Prerequisites
```bash
# Install Python 3.9+
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .
```

#### Development Commands
```bash
# Run development server
uvicorn intelligence_layer.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Type checking
mypy src/

# Code formatting
black src/
isort src/

# Linting
flake8 src/
```

#### Configuration
```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://user:pass@localhost/trading"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    redis_url: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### API Endpoints

#### Intelligence Endpoints
```python
@app.post("/intelligence/embedding")
async def get_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate market state embedding."""
    model = get_embedding_model()
    embedding = model.embed_market_state(request.market_data)
    
    return EmbeddingResponse(
        embedding=embedding.tolist(),
        confidence=calculate_confidence(embedding),
        timestamp=datetime.utcnow()
    )

@app.get("/intelligence/regime")
async def get_regime(asset_id: str) -> RegimeResponse:
    """Get current market regime for asset."""
    detector = get_regime_detector()
    market_data = await get_market_data(asset_id)
    regime = detector.detect_regime(market_data)
    
    return RegimeResponse(
        asset_id=asset_id,
        regime_probabilities=regime.regime_probabilities,
        confidence=regime.confidence,
        timestamp=datetime.utcnow()
    )
```

## Database Integration

### PostgreSQL with pgvector
```python
# Database models
class MarketStateEmbedding(Base):
    __tablename__ = "market_state_embeddings"
    
    id = Column(UUID, primary_key=True, default=uuid4)
    asset_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    embedding = Column(Vector(128))  # pgvector
    confidence = Column(Float)
    
    __table_args__ = (
        Index("idx_asset_timestamp", "asset_id", "timestamp"),
        Index("idx_embedding_similarity", "embedding", postgresql_using="ivfflat"),
    )
```

### Neo4j Graph Schema
```cypher
// Create constraints
CREATE CONSTRAINT asset_id IF NOT EXISTS FOR (a:Asset) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT regime_name IF NOT EXISTS FOR (r:MarketRegime) REQUIRE r.name IS UNIQUE;

// Create nodes
CREATE (a:Asset {id: 'EURUSD', name: 'Euro/US Dollar', type: 'forex'});
CREATE (r:MarketRegime {name: 'LOW_VOL_TRENDING', volatility: 'low', trend: 'trending'});

// Create relationships
MATCH (a1:Asset {id: 'EURUSD'}), (a2:Asset {id: 'GBPUSD'})
CREATE (a1)-[:CORRELATES_WITH {correlation: 0.85, period: '24h'}]->(a2);
```

## Testing

### Rust Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_portfolio_update() {
        let mut portfolio = Portfolio::new();
        let fill = Fill::new(/* ... */);
        
        let result = portfolio.update_position(&fill);
        assert!(result.is_ok());
        
        let position = portfolio.get_position(&fill.asset_id);
        assert_eq!(position.quantity, fill.quantity);
    }
    
    #[test]
    fn test_risk_check() {
        let risk_manager = RiskManager::new(RiskLimits::default());
        let order = Order::new(/* ... */);
        
        let result = risk_manager.check_order(&order);
        assert!(result.is_ok());
    }
}
```

### Python Testing
```python
import pytest
from intelligence_layer.regime_detection import HMMRegimeDetector

@pytest.fixture
def regime_detector():
    return HMMRegimeDetector(n_regimes=3)

@pytest.mark.asyncio
async def test_regime_detection(regime_detector):
    market_data = generate_test_market_data()
    regime = regime_detector.detect_regime(market_data)
    
    assert len(regime.regime_probabilities) == 3
    assert 0 <= regime.confidence <= 1
    assert np.sum(regime.regime_probabilities) == pytest.approx(1.0)

def test_embedding_model():
    model = TCNEmbeddingModel(ModelConfig())
    market_data = generate_test_market_data()
    
    embedding = model.embed_market_state(market_data)
    
    assert embedding.shape == (128,)  # Expected embedding dimension
    assert not np.isnan(embedding).any()
```

## Deployment

### Docker Configuration
```dockerfile
# Rust Dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/execution-core /usr/local/bin/
EXPOSE 8001
CMD ["execution-core"]
```

```dockerfile
# Python Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "intelligence_layer.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  execution-core:
    build: ./execution-core
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/trading
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
      
  intelligence-layer:
    build: ./intelligence-layer
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/trading
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - neo4j
      - redis
```

## Performance Optimization

### Rust Optimization
```rust
// Use async/await for I/O operations
async fn process_order(order: Order) -> Result<Fill, ExecutionError> {
    let validation = validate_order(&order).await?;
    let fill = execute_order(order).await?;
    update_portfolio(&fill).await?;
    Ok(fill)
}

// Use channels for inter-thread communication
use tokio::sync::mpsc;

let (tx, mut rx) = mpsc::channel(1000);

// Producer
tokio::spawn(async move {
    while let Some(event) = event_stream.next().await {
        tx.send(event).await.unwrap();
    }
});

// Consumer
tokio::spawn(async move {
    while let Some(event) = rx.recv().await {
        process_event(event).await;
    }
});
```

### Python Optimization
```python
# Use async/await for database operations
async def get_market_data(asset_id: str) -> MarketData:
    async with database.transaction():
        query = "SELECT * FROM market_data WHERE asset_id = $1 ORDER BY timestamp DESC LIMIT 1000"
        result = await database.fetch_all(query, asset_id)
        return [MarketData(**row) for row in result]

# Use connection pooling
from databases import Database

database = Database(
    "postgresql://user:pass@localhost/trading",
    min_size=5,
    max_size=20
)

# Cache expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_correlation_matrix(asset_ids: tuple) -> np.ndarray:
    # Expensive computation
    pass
```

## Monitoring and Observability

### Structured Logging
```rust
// Rust logging
use tracing::{info, warn, error, instrument};

#[instrument]
async fn execute_order(order: Order) -> Result<Fill, ExecutionError> {
    info!(order_id = %order.id, "Executing order");
    
    match broker.submit_order(order).await {
        Ok(fill) => {
            info!(fill_id = %fill.id, "Order filled successfully");
            Ok(fill)
        }
        Err(e) => {
            error!(error = %e, "Order execution failed");
            Err(e)
        }
    }
}
```

```python
# Python logging
import structlog

logger = structlog.get_logger()

async def detect_regime(market_data: MarketData) -> RegimeInference:
    logger.info("Starting regime detection", asset_id=market_data.asset_id)
    
    try:
        regime = await _compute_regime(market_data)
        logger.info("Regime detected", 
                   regime=regime.most_likely_regime,
                   confidence=regime.confidence)
        return regime
    except Exception as e:
        logger.error("Regime detection failed", error=str(e))
        raise
```

### Health Checks
```rust
// Rust health checks
#[derive(Serialize)]
struct HealthStatus {
    status: String,
    timestamp: DateTime<Utc>,
    components: HashMap<String, ComponentHealth>,
}

async fn check_health() -> HealthStatus {
    let mut components = HashMap::new();
    
    components.insert("database".to_string(), check_database().await);
    components.insert("redis".to_string(), check_redis().await);
    components.insert("risk_engine".to_string(), check_risk_engine().await);
    
    let overall_status = if components.values().all(|c| c.healthy) {
        "healthy"
    } else {
        "degraded"
    };
    
    HealthStatus {
        status: overall_status.to_string(),
        timestamp: Utc::now(),
        components,
    }
}
```

## Security Considerations

### Input Validation
```rust
// Rust input validation
use validator::Validate;

#[derive(Deserialize, Validate)]
struct OrderRequest {
    #[validate(length(min = 1, max = 10))]
    asset_id: String,
    
    #[validate(range(min = 0.01, max = 1000000.0))]
    quantity: f64,
    
    #[validate(custom = "validate_order_type")]
    order_type: OrderType,
}

fn validate_order_type(order_type: &OrderType) -> Result<(), ValidationError> {
    match order_type {
        OrderType::Market | OrderType::Limit => Ok(()),
        _ => Err(ValidationError::new("invalid_order_type")),
    }
}
```

### Authentication (Future)
```python
# Python JWT authentication
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/protected")
async def protected_endpoint(current_user: str = Depends(get_current_user)):
    return {"user": current_user}
```

## Best Practices

### Error Handling
```rust
// Rust error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    
    #[error("Risk limit exceeded: {limit}")]
    RiskLimitExceeded { limit: String },
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
}
```

### Configuration Management
```python
# Python configuration
from pydantic import BaseSettings, validator

class DatabaseSettings(BaseSettings):
    url: str
    pool_size: int = 10
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith('postgresql://'):
            raise ValueError('Database URL must use postgresql:// scheme')
        return v

class Settings(BaseSettings):
    database: DatabaseSettings
    debug: bool = False
    
    class Config:
        env_nested_delimiter = '__'
        env_file = '.env'
```

This comprehensive backend development guide provides the foundation for building a robust, scalable, and maintainable algorithmic trading system using Rust and Python.