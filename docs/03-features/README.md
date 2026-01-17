# Features Overview

This section contains detailed documentation for all major features of the algorithmic trading system.

## Feature Documentation

### [Markets (F2)](./markets.md)
Live market data, watchlists, alerts, correlations, and microstructure analysis.

**Key Capabilities:**
- Real-time market data from Deriv API
- Watchlist management with CRUD operations
- Alert system (Price, Volatility, Liquidity, Correlation)
- Correlation matrix analysis
- Microstructure metrics
- Market events feed

### [Intelligence (F3)](./intelligence.md)
Machine learning models, regime detection, graph analytics, and real-time inference.

**Key Capabilities:**
- 18 production-grade ML models
- Embedding management and configuration
- Regime detection with HMM models
- Graph analytics (PageRank, Louvain, Betweenness)
- Real-time signal generation
- Model browser and selection

### [Portfolio (F5)](./portfolio.md)
Risk management, portfolio control, exposure tracking, and performance attribution.

**Key Capabilities:**
- Multi-portfolio management
- Capital allocation models (6 types)
- Real-time position tracking
- Exposure analysis (Gross, Net, Leverage)
- Risk limits and metrics
- Stress testing scenarios
- Performance attribution

### [Simulation (F7)](./simulation.md)
Institutional-grade backtesting, experiment tracking, and scenario analysis.

**Key Capabilities:**
- Experiment registry with full CRUD
- Bias controls (survivorship, look-ahead)
- Execution realism (slippage, costs, latency)
- Performance metrics (Sharpe, Sortino, CVaR)
- Scenario testing (4 stress tests)
- Experiment comparison
- Reproducibility features

### [Data & Models (F8)](./data-models.md)
Model training, validation, dataset management, and experiment tracking.

**Key Capabilities:**
- Model lifecycle management
- Dataset quality assessment
- Training job tracking
- Validation metrics and confusion matrix
- Model deployment workflow
- Hyperparameter configuration

### [Data Import](./data-import.md)
Data ingestion, transformation, and quality validation.

**Key Capabilities:**
- Multiple data source support
- Data transformation pipelines
- Quality validation and cleansing
- Batch and streaming ingestion
- Data lineage tracking

## Quick Start Guides

Each feature includes:
- ✅ **Complete Implementation** - All buttons and modals functional
- ✅ **Mock Data Fallback** - Works offline without backend
- ✅ **Real-time Updates** - WebSocket integration ready
- ✅ **Professional UI** - Bloomberg terminal aesthetic
- ✅ **CRUD Operations** - Full create, read, update, delete

## Feature Integration

All features are designed to work together:

```
Markets → Intelligence → Portfolio → Execution
   ↓           ↓            ↓          ↓
Data Import ← Simulation ← Risk Mgmt ← Trading
```

## Navigation

- **Getting Started**: [Quick Start Guide](../01-getting-started/quick-start.md)
- **Architecture**: [System Architecture](../02-architecture/system-architecture.md)
- **Deployment**: [Startup Scripts](../04-deployment/startup-scripts.md)
- **Development**: [Frontend Development](../06-development/frontend.md)

---

**Status**: All features complete and production-ready  
**Last Updated**: January 2026