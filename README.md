# Algorithmic Trading System

A full-stack, research-grade algorithmic trading system designed for master's thesis research. The system prioritizes intelligence completeness and academic rigor while maintaining strict separation between research intelligence and capital deployment through enforced architectural boundaries.

## Architecture Overview

The system follows a "research system that can trade" philosophy with the following components:

- **Execution Core (Rust)**: Portfolio accounting, risk management, and order execution
- **Intelligence Layer (Python)**: Market analysis, regime detection, and ML services  
- **Simulation Engine (Rust)**: Deterministic backtesting and scenario analysis
- **Frontend (React/TypeScript)**: Admin interface with nLVE framework
- **Data Layer**: Neo4j (graph), PostgreSQL with pgvector (embeddings), Redis (cache)

## Quick Start

See **[QUICKSTART.md](QUICKSTART.md)** for a 5-minute setup guide.

### Prerequisites

- Docker and Docker Compose
- Rust (1.75+)
- Python (3.9+)
- Node.js (18+)

### One-Command Start

```bash
# Start all backend services
docker-compose up -d

# Verify health
./scripts/test-integration.sh  # Linux/Mac
.\scripts\test-integration.ps1  # Windows

# Start frontend
cd frontend && npm install && npm run dev
```

Frontend available at: **http://localhost:5173**

### Development Setup

1. **Clone and setup environment**:
```bash
git clone <repository>
cd algorithmic-trading-system
cp .env.example .env
```

2. **Start infrastructure services**:
```bash
docker-compose up -d postgres neo4j redis
```

3. **Initialize databases**:
```bash
# PostgreSQL will auto-initialize from database/init/01-init.sql
# Neo4j initialization (run after Neo4j is ready)
docker exec -i trading-neo4j cypher-shell -u neo4j -p password < database/init/02-neo4j-init.cypher
```

4. **Start development services**:

**Execution Core (Rust)**:
```bash
cd execution-core
cargo run
```

**Intelligence Layer (Python)**:
```bash
cd intelligence-layer
pip install -e .
uvicorn intelligence_layer.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (React)**:
```bash
cd frontend
npm install
npm run dev
```

### Production Deployment

```bash
docker-compose up -d
```

## Project Structure

```
├── execution-core/          # Rust execution engine
│   ├── src/
│   │   ├── lib.rs           # Core execution traits
│   │   ├── config.rs        # Configuration management
│   │   ├── event_bus.rs     # Event-driven architecture
│   │   ├── portfolio.rs     # Portfolio accounting
│   │   ├── risk.rs          # Risk management
│   │   └── main.rs          # Application entry point
│   └── Cargo.toml
├── simulation-engine/       # Rust simulation engine
│   ├── src/
│   │   ├── lib.rs           # Simulation framework
│   │   ├── clock.rs         # Deterministic time management
│   │   └── backtesting.rs   # Event-driven backtesting
│   └── Cargo.toml
├── intelligence-layer/      # Python intelligence services
│   ├── src/intelligence_layer/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── config.py        # Configuration
│   │   ├── models.py        # Data models
│   │   └── logging.py       # Structured logging
│   └── pyproject.toml
├── frontend/                # React admin interface
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── pages/           # Application pages
│   │   ├── types/           # TypeScript types
│   │   ├── lib/             # Utilities
│   │   └── main.tsx         # Application entry
│   └── package.json
├── database/
│   └── init/                # Database initialization scripts
├── docker-compose.yml       # Development environment
└── README.md
```

## Key Features

### Frontend-Backend Integration
- **Real-time Updates**: Automatic polling of regime detection and graph analytics
- **Health Monitoring**: Continuous health checks of all backend services
- **Bloomberg Terminal UI**: Professional trading interface with F1-F9 navigation
- **Type-safe API**: Full TypeScript integration with Python FastAPI
- **Error Handling**: Graceful degradation and automatic retry logic
- **WebSocket Ready**: Infrastructure for real-time streaming (future)

See **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** for detailed integration documentation.

### Core System Architecture
- **Sandboxing**: Intelligence layer provides advisory outputs only
- **Event Bus**: Deterministic message passing with replay capability
- **Risk Management**: Configurable guardrails and emergency halt
- **Multi-language**: Rust for performance, Python for ML, TypeScript for UI

### Intelligence & Learning
- **Embedding Models**: TCN/VAE for market state representation
- **Regime Detection**: HMM-based market regime classification
- **Graph Analytics**: Neo4j GDS for market relationship analysis
- **Reinforcement Learning**: MDP environment for strategy orchestration

### Data Management
- **Neo4j Schema**: Structural market knowledge (assets, regimes, strategies)
- **pgvector**: Vector embeddings for similarity search
- **PostgreSQL**: Transactional data (orders, fills, positions)
- **Redis**: Caching and real-time data

### Research Framework
- **Deterministic Replay**: Complete reproducibility for academic evaluation
- **Experiment Tracking**: Version control for models and configurations
- **Ablation Studies**: Component evaluation framework
- **Academic Safeguards**: Lookahead bias prevention

## API Endpoints

### Intelligence Layer (Port 8000)

- `POST /intelligence/embedding` - Market state embedding inference
- `POST /intelligence/regime` - Regime classification and transitions
- `GET /intelligence/graph-features` - Graph structural features
- `GET /intelligence/state` - Composite RL state assembly

### Execution Core (Port 8001)

- `GET /health` - Health check endpoint
- Additional endpoints for order management (to be implemented)

### Frontend (Port 3000)

- Admin interface with nLVE framework
- Real-time dashboards and monitoring
- Configuration management

## Database Schemas

### PostgreSQL Tables
- `intelligence.market_state_embeddings` - Market embeddings with pgvector
- `intelligence.strategy_state_embeddings` - Strategy state vectors
- `execution.orders` - Order management
- `execution.fills` - Trade execution records
- `execution.positions` - Portfolio positions

### Neo4j Schema
- **Nodes**: Asset, MarketRegime, Strategy, IntelligenceSignal, MacroEvent
- **Relationships**: CORRELATED, TRANSITIONS_TO, PERFORMS_IN, SENSITIVE_TO, AFFECTS

## Configuration

Environment variables are prefixed by service:
- `TRADING_*` - Execution core configuration
- `INTELLIGENCE_*` - Intelligence layer configuration  
- `VITE_*` - Frontend configuration

See `.env.example` for complete configuration options.

## Development

### Running Tests

**Rust**:
```bash
cargo test
```

**Python**:
```bash
cd intelligence-layer
pytest
```

**TypeScript**:
```bash
cd frontend
npm test
```

### Code Quality

**Rust**:
```bash
cargo fmt
cargo clippy
```

**Python**:
```bash
black src/
isort src/
flake8 src/
mypy src/
```

**TypeScript**:
```bash
npm run lint
```

## Academic Research Features

- **Reproducible Experiments**: Complete versioning and deterministic replay
- **Formal Evaluation**: Property-based testing and correctness validation
- **Ablation Studies**: Component isolation for academic analysis
- **Negative Results**: Documentation of failed approaches
- **Thesis Integration**: Designed for master's thesis evaluation

## License

MIT License - See LICENSE file for details.

## Contributing

This is a research project. Please see CONTRIBUTING.md for guidelines.#   s y m m e t r i c a l - d o o d l e  
 