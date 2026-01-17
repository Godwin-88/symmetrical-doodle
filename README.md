# Algorithmic Trading System

A full-stack, research-grade algorithmic trading system designed for master's thesis research. The system prioritizes intelligence completeness and academic rigor while maintaining strict separation between research intelligence and capital deployment through enforced architectural boundaries.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository>
cd algorithmic-trading-system
cp .env.example .env

# 2. Start all services (one command)
./scripts/start-all.sh        # Linux/Mac
.\scripts\start-all.ps1       # Windows

# 3. Verify system health
./scripts/test-integration.sh  # Linux/Mac
.\scripts\test-integration.ps1 # Windows
```

**Frontend**: http://localhost:5173  
**Intelligence API**: http://localhost:8000  
**Execution API**: http://localhost:8001

## 📚 Documentation

Complete documentation is available in the [`docs/`](./docs/) folder:

### [📖 Getting Started](./docs/01-getting-started/)
- **[Quick Start Guide](./docs/01-getting-started/quick-start.md)** - 5-minute setup
- **[System Overview](./docs/01-getting-started/system-overview.md)** - High-level introduction
- **[Installation Guide](./docs/01-getting-started/installation.md)** - Complete setup instructions

### [🏗️ Architecture](./docs/02-architecture/)
- **[System Architecture](./docs/02-architecture/system-architecture.md)** - Overall design
- **[Component Overview](./docs/02-architecture/component-overview.md)** - Individual components
- **[Data Flow](./docs/02-architecture/data-flow.md)** - How data moves through the system
- **[Technology Stack](./docs/02-architecture/technology-stack.md)** - Technologies used

### [⚡ Features](./docs/03-features/)
- **[Markets (F2)](./docs/03-features/markets.md)** - Live market data and analysis
- **[Intelligence (F3)](./docs/03-features/intelligence.md)** - ML models and analytics
- **[Portfolio (F5)](./docs/03-features/portfolio.md)** - Risk management
- **[Simulation (F7)](./docs/03-features/simulation.md)** - Backtesting
- **[Data Workspace (F10)](./docs/03-features/data-workspace.md)** - Advanced analytics

### [🚀 Deployment](./docs/04-deployment/)
- **[Startup Scripts](./docs/04-deployment/startup-scripts.md)** - Automated deployment
- **[Environment Config](./docs/04-deployment/environment-config.md)** - Configuration guide
- **[Production Deployment](./docs/04-deployment/production-deployment.md)** - Production setup

### [🔌 Integrations](./docs/05-integrations/)
- **[Deriv API](./docs/05-integrations/deriv-api.md)** - Live trading integration
- **[Neo4j Aura](./docs/05-integrations/neo4j-aura.md)** - Cloud graph database
- **[External APIs](./docs/05-integrations/external-apis.md)** - Third-party integrations

### [💻 Development](./docs/06-development/)
- **[Frontend Development](./docs/06-development/frontend.md)** - React/TypeScript guide
- **[Backend Development](./docs/06-development/backend.md)** - Rust/Python guide
- **[API Reference](./docs/06-development/api-reference.md)** - Complete API docs
- **[Testing Guide](./docs/06-development/testing.md)** - Testing strategies

### [🔧 Troubleshooting](./docs/07-troubleshooting/)
- **[Common Issues](./docs/07-troubleshooting/common-issues.md)** - Frequently encountered problems
- **[Database Issues](./docs/07-troubleshooting/database-issues.md)** - Database troubleshooting

## 🏛️ Architecture Overview

The system follows a "research system that can trade" philosophy:

- **Execution Core (Rust)**: Portfolio accounting, risk management, order execution
- **Intelligence Layer (Python)**: Market analysis, regime detection, ML services  
- **Simulation Engine (Rust)**: Deterministic backtesting and scenario analysis
- **Frontend (React/TypeScript)**: Bloomberg Terminal-inspired interface
- **Data Layer**: Neo4j (graph), PostgreSQL with pgvector (embeddings), Redis (cache)

## 🎯 Key Features

### Professional Trading Interface
- **Bloomberg Terminal UI**: F1-F9 navigation, dark theme, dense information
- **Real-time Data**: Live market feeds, regime detection, risk metrics
- **Data Workspace**: Advanced analytics with 8 visualization types
- **Emergency Controls**: Kill switch, risk limits, health monitoring

### Intelligence & ML
- **Regime Detection**: HMM-based market classification
- **Embedding Models**: TCN/VAE for market state representation
- **Graph Analytics**: Neo4j GDS for relationship analysis
- **Strategy Orchestration**: RL environment for systematic trading

### Academic Rigor
- **Deterministic Replay**: Complete reproducibility
- **Experiment Tracking**: Version control for models
- **Audit Trails**: Full system observability
- **Property Testing**: Formal verification

### Production Ready
- **Multi-language**: Rust (performance), Python (ML), TypeScript (UI)
- **Event-driven**: Deterministic message passing
- **Risk Management**: Configurable guardrails
- **Health Monitoring**: Comprehensive system health

## 📊 System Status (January 2025)

**✅ COMPLETED COMPONENTS:**
- **Frontend UI**: Complete React/TypeScript interface with 10 functional modules
- **Intelligence Layer**: AI-powered analysis with LLM/RAG integration (NEWLY ENHANCED)
- **Market Data**: Real-time data feeds and analysis
- **Portfolio Management**: Position tracking and risk management
- **Strategy Engine**: Algorithm development and backtesting
- **Execution Core**: Order management and trade execution
- **Data Workspace**: Import/export and data management
- **MLOps Pipeline**: Model training and deployment
- **System Monitoring**: Health checks and performance metrics

**🔄 IN PROGRESS:**
- Backend service integration and API endpoints
- Database schema optimization
- Real-time WebSocket connections

**📊 SYSTEM METRICS:**
- **Frontend**: 10/10 modules functional with mock fallbacks
- **Backend**: 8/10 services implemented
- **Database**: PostgreSQL + Neo4j + Vector DB ready
- **Testing**: Comprehensive test suite with property-based testing
- **Documentation**: Complete user and developer guides

**🆕 LATEST UPDATES (January 2025):**
- **Intelligence Module Rewritten**: Complete CRUD functionality with AI chat, research reports, document management, and analysis models
- **Mock Fallback System**: Full UI functionality even when backend services are offline
- **Enhanced Testing**: Comprehensive user testing guide created
- **Documentation Updates**: All docs updated with current status

## 🛠️ Development

### Prerequisites
- Docker and Docker Compose
- Rust (1.75+)
- Python (3.9+)
- Node.js (18+)

### Development Commands
```bash
# Backend (Rust)
cd execution-core && cargo run

# Intelligence (Python)
cd intelligence-layer && uvicorn intelligence_layer.main:app --reload

# Frontend (React)
cd frontend && npm run dev

# Tests
cargo test                    # Rust tests
pytest                        # Python tests
npm test                      # Frontend tests
```

## 📈 Production Deployment

```bash
# One-command production deployment
docker-compose up -d

# Health verification
./scripts/test-integration.sh
```

## 🎓 Academic Use

This system is designed for:
- Master's thesis research
- Quantitative finance education
- Algorithmic trading research
- Financial ML experimentation

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a research project. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**For complete documentation, visit the [`docs/`](./docs/) folder.**