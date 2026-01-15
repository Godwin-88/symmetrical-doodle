# Algorithmic Trading Platform - Frontend

Bloomberg Terminal-inspired frontend for the algorithmic trading system.

## Quick Start

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

## Architecture

### Design System
- **Theme**: Bloomberg Terminal aesthetic
- **Colors**: Dark background (#0a0a0a), Orange accents (#ff8c00), Green/Red for P&L
- **Typography**: Monospace fonts (JetBrains Mono, Courier New)
- **Layout**: Dense information grids, sharp borders, minimal whitespace

### Navigation (Function Keys)
- **F1**: Dashboard - System overview, positions, strategy performance
- **F2**: Markets - Live market data, correlations, microstructure
- **F3**: Intelligence - Regime detection, embeddings, signals
- **F4**: Strategies - Strategy catalog, parameters, regime affinity
- **F5**: Portfolio - Positions, exposure, risk metrics, performance
- **F6**: Execution - Adapters, latency, order activity, quality metrics
- **F7**: Simulation - Experiments, backtests, scenario analysis
- **F8**: Data & Models - Deployed models, training, validation, provenance
- **F9**: System - Component health, resources, logs, configuration

### Components

```
frontend/src/app/
├── components/
│   ├── Dashboard.tsx       (F1)
│   ├── Markets.tsx         (F2)
│   ├── Intelligence.tsx    (F3)
│   ├── Strategies.tsx      (F4)
│   ├── Portfolio.tsx       (F5)
│   ├── Execution.tsx       (F6)
│   ├── Simulation.tsx      (F7)
│   ├── DataModels.tsx      (F8)
│   ├── System.tsx          (F9)
│   ├── FunctionKeyBar.tsx
│   └── StatusBar.tsx
├── store/
│   └── tradingStore.ts     (Zustand state management)
└── App.tsx
```

### State Management
- **Zustand** for centralized state
- Type-safe with TypeScript interfaces
- Mock data for development

## Features

### Real-Time Data Display
- System health monitoring
- Live P&L tracking
- Position management
- Strategy performance
- Market regime detection
- Execution quality metrics

### Professional UI
- Terminal-style tables
- Color-coded status indicators
- Dense information layout
- Keyboard shortcuts (F1-F9)
- Emergency halt button

### Academic Rigor
- Model provenance tracking
- Experiment reproducibility
- Audit trails
- Validation metrics
- Training protocol compliance

## Backend Integration

### API Endpoints
- **Intelligence Layer**: http://localhost:8000 (Python FastAPI)
- **Execution Core**: http://localhost:8001 (Rust)

### WebSocket Streams
- Market data updates
- Intelligence signals
- Execution fills
- Risk metrics
- System health

### Databases
- **PostgreSQL + pgvector**: Time-series data, embeddings
- **Neo4j + GDS**: Graph analytics, regime transitions
- **Redis**: Real-time state, caching

## Development

### Tech Stack
- **React 18** with TypeScript
- **Vite** for build tooling
- **TailwindCSS** for styling
- **Zustand** for state management
- **Lucide React** for icons

### Code Style
- TypeScript strict mode
- Functional components with hooks
- Consistent formatting (2 spaces)
- Terminal-style naming (UPPERCASE for labels)

### Build Output
- Production build: ~226KB (gzipped: ~58KB)
- CSS: ~89KB (gzipped: ~14KB)
- No TypeScript errors
- No linting errors

## Next Steps

### Backend Integration
1. Replace mock data with API calls
2. Implement WebSocket connections
3. Add authentication/authorization
4. Error handling and retry logic

### Enhanced Features
1. Charts and visualizations (t-SNE, regime graphs, P&L charts)
2. Interactive forms (order placement, parameter tuning)
3. Advanced filtering and search
4. Export functionality (CSV, JSON)
5. Custom alerts and notifications

### Performance
1. Virtual scrolling for large tables
2. Memoization for expensive computations
3. WebSocket connection pooling
4. Optimistic UI updates

## Documentation

- **IMPLEMENTATION_SUMMARY.md**: Detailed implementation overview
- **frontend.md**: Comprehensive architecture description
- **ATTRIBUTIONS.md**: Third-party licenses and credits

## License

See parent project LICENSE file.
