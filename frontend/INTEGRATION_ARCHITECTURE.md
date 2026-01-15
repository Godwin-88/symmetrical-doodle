# Frontend Integration Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (User)                               │
│                     http://localhost:5173                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ HTTP/REST
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Frontend (React)                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Components (F1-F9)                                           │  │
│  │  - Dashboard, Markets, Intelligence, Strategies, etc.        │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│  ┌────────────────────────▼─────────────────────────────────────┐  │
│  │  Zustand Store (State Management)                            │  │
│  │  - regimes, positions, strategies                            │  │
│  │  - fetchRegimeData(), checkHealth()                          │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│  ┌────────────────────────▼─────────────────────────────────────┐  │
│  │  API Services                                                 │  │
│  │  - intelligenceService.ts                                    │  │
│  │  - api.ts (base client)                                      │  │
│  │  - websocketService.ts (future)                              │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        │                                       │
┌───────▼──────────────┐            ┌──────────▼─────────────┐
│ Intelligence Layer   │            │   Execution Core       │
│   (Python FastAPI)   │            │      (Rust)            │
│  localhost:8000      │            │   localhost:8001       │
│                      │            │                        │
│  Endpoints:          │            │  Endpoints:            │
│  - /health           │            │  - /health             │
│  - /intelligence/*   │            │  - /portfolio/*        │
│                      │            │  - /risk/*             │
└──────┬───────────────┘            └────────┬───────────────┘
       │                                     │
       │                                     │
┌──────▼─────────────────────────────────────▼───────────────┐
│                    Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ PostgreSQL   │  │   Neo4j      │  │    Redis     │    │
│  │ + pgvector   │  │   + GDS      │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

### 1. Initial Load

```
User opens browser
    │
    ▼
App.tsx mounts
    │
    ├─► checkHealth() every 30s
    │   │
    │   ├─► GET /health (Intelligence)
    │   │   └─► Update systemStatus
    │   │
    │   └─► GET /health (Execution)
    │       └─► Update connectionStatus
    │
    └─► Intelligence.tsx mounts
        │
        ├─► fetchRegimeData('EURUSD')
        │   │
        │   └─► GET /intelligence/regime?asset_id=EURUSD
        │       │
        │       └─► Transform response
        │           │
        │           └─► Update regimes[] in store
        │               │
        │               └─► UI re-renders
        │
        └─► fetchGraphFeatures('EURUSD')
            │
            └─► GET /intelligence/graph-features?asset_id=EURUSD
                │
                └─► Update intelligenceSignals[] in store
                    │
                    └─► UI re-renders
```

### 2. Periodic Updates (Every 30 seconds)

```
setInterval(30000)
    │
    ├─► checkHealth()
    │   └─► Update system status
    │
    ├─► fetchRegimeData()
    │   └─► Update regime probabilities
    │
    └─► fetchGraphFeatures()
        └─► Update graph metrics
```

### 3. User Interaction

```
User presses F3 (Intelligence)
    │
    ▼
setCurrentDomain('INTL')
    │
    ▼
Intelligence component renders
    │
    ├─► Display regimes from store
    ├─► Display embeddings from store
    └─► Display signals from store
        │
        └─► Data already loaded (no API call)
```

## Component Architecture

```
App.tsx
├── FunctionKeyBar.tsx
│   └── Navigation (F1-F9)
│
├── Current Domain Component
│   ├── Dashboard.tsx (F1)
│   │   └── Uses: positions, strategies, systemStatus
│   │
│   ├── Markets.tsx (F2)
│   │   └── Uses: marketData (mock for now)
│   │
│   ├── Intelligence.tsx (F3)
│   │   ├── Uses: regimes, embeddings, signals
│   │   ├── Calls: fetchRegimeData()
│   │   └── Calls: fetchGraphFeatures()
│   │
│   ├── Strategies.tsx (F4)
│   │   └── Uses: strategies (mock for now)
│   │
│   ├── Portfolio.tsx (F5)
│   │   └── Uses: positions, exposure (mock for now)
│   │
│   ├── Execution.tsx (F6)
│   │   └── Uses: orders, fills (mock for now)
│   │
│   ├── Simulation.tsx (F7)
│   │   └── Uses: experiments (mock for now)
│   │
│   ├── DataModels.tsx (F8)
│   │   └── Uses: models, datasets (mock for now)
│   │
│   └── System.tsx (F9)
│       └── Uses: systemStatus, health (from backend)
│
└── StatusBar.tsx
    └── Uses: systemStatus, connectionStatus, latency
```

## State Management Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Zustand Store                             │
│                                                              │
│  State:                                                      │
│  ├── currentDomain: 'DASH' | 'MKTS' | ...                  │
│  ├── systemStatus: 'OPERATIONAL' | 'DEGRADED' | 'DOWN'     │
│  ├── connectionStatus: 'LIVE' | 'DELAYED' | 'DISCONNECTED' │
│  ├── regimes: MarketRegime[]                                │
│  ├── positions: Position[]                                  │
│  ├── strategies: Strategy[]                                 │
│  ├── isLoading: boolean                                     │
│  ├── error: string | null                                   │
│  └── lastUpdate: Date | null                                │
│                                                              │
│  Actions:                                                    │
│  ├── setCurrentDomain(domain)                               │
│  ├── fetchRegimeData(assetId)                               │
│  │   └─► intelligenceService.getRegimeInference()           │
│  │       └─► Transform & update state                       │
│  │                                                           │
│  ├── fetchGraphFeatures(assetId)                            │
│  │   └─► intelligenceService.getGraphFeatures()             │
│  │       └─► Transform & update state                       │
│  │                                                           │
│  ├── fetchRLState(assets, strategies)                       │
│  │   └─► intelligenceService.assembleRLState()              │
│  │       └─► Transform & update state                       │
│  │                                                           │
│  ├── checkHealth()                                          │
│  │   └─► api.checkIntelligenceHealth()                      │
│  │   └─► api.checkExecutionHealth()                         │
│  │       └─► Update systemStatus                            │
│  │                                                           │
│  └── clearError()                                           │
│      └─► Set error to null                                  │
└─────────────────────────────────────────────────────────────┘
```

## API Service Layer

```
┌─────────────────────────────────────────────────────────────┐
│                   api.ts (Base Client)                       │
│                                                              │
│  class ApiClient {                                           │
│    async get<T>(endpoint): Promise<T>                        │
│    async post<T>(endpoint, data): Promise<T>                 │
│  }                                                           │
│                                                              │
│  export const intelligenceApi = new ApiClient(8000)          │
│  export const executionApi = new ApiClient(8001)             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            intelligenceService.ts (Typed API)                │
│                                                              │
│  export async function getRegimeInference(assetId) {         │
│    return intelligenceApi.get(                               │
│      `/intelligence/regime?asset_id=${assetId}`              │
│    );                                                        │
│  }                                                           │
│                                                              │
│  export async function getGraphFeatures(assetId) {           │
│    return intelligenceApi.get(                               │
│      `/intelligence/graph-features?asset_id=${assetId}`      │
│    );                                                        │
│  }                                                           │
│                                                              │
│  export async function assembleRLState(assets, strategies) { │
│    return intelligenceApi.get(                               │
│      `/intelligence/state?asset_ids=${assets.join(',')}`     │
│    );                                                        │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
API Call
    │
    ├─► Success
    │   └─► Update state with data
    │       └─► UI re-renders
    │
    └─► Error
        │
        ├─► Network Error (backend down)
        │   └─► Set error in state
        │       └─► Display error banner in UI
        │           └─► Retry on next poll
        │
        ├─► HTTP Error (4xx/5xx)
        │   └─► Set error in state
        │       └─► Display error message
        │           └─► Log to console
        │
        └─► Timeout
            └─► Set error in state
                └─► Display timeout message
                    └─► Retry on next poll
```

## Polling Strategy

```
Component Mount
    │
    ▼
Initial Fetch
    │
    └─► fetchData()
        │
        └─► Success or Error
            │
            ▼
        setInterval(30000)
            │
            └─► Every 30 seconds
                │
                └─► fetchData()
                    │
                    ├─► Success → Update UI
                    └─► Error → Show error, retry next time
```

## Type Safety

```
Backend (Python)
    │
    ├─► RegimeResponse (Pydantic)
    │   └─► JSON Response
    │
Frontend (TypeScript)
    │
    ├─► RegimeResponse (interface)
    │   └─► Type-checked at compile time
    │
    └─► MarketRegime (interface)
        └─► Transformed from backend response
```

## Environment Configuration

```
.env.development
    │
    ├─► VITE_INTELLIGENCE_API_URL=http://localhost:8000
    ├─► VITE_EXECUTION_API_URL=http://localhost:8001
    ├─► VITE_POLLING_INTERVAL=5000
    └─► VITE_HEALTH_CHECK_INTERVAL=30000
        │
        ▼
    import.meta.env.VITE_*
        │
        ▼
    Used in api.ts
        │
        ▼
    API calls use correct URLs
```

## Build Process

```
Source Code (TypeScript)
    │
    ├─► Type Checking
    │   └─► tsc --noEmit
    │
    ├─► Bundling
    │   └─► Vite build
    │       │
    │       ├─► Tree shaking
    │       ├─► Minification
    │       └─► Code splitting
    │
    └─► Output
        │
        ├─► dist/index.html (0.45 KB)
        ├─► dist/assets/index.css (89 KB → 14 KB gzipped)
        └─► dist/assets/index.js (230 KB → 60 KB gzipped)
```

## Deployment Architecture

```
Production Environment
    │
    ├─► Frontend (Static Files)
    │   └─► Nginx / CDN
    │       └─► Serves React app
    │
    ├─► Intelligence Layer
    │   └─► Docker Container
    │       └─► Python FastAPI
    │           └─► Port 8000
    │
    ├─► Execution Core
    │   └─► Docker Container
    │       └─► Rust Binary
    │           └─► Port 8001
    │
    └─► Data Layer
        ├─► PostgreSQL + pgvector
        ├─► Neo4j + GDS
        └─► Redis
```

## Security Considerations

```
Frontend
    │
    ├─► HTTPS only in production
    ├─► CORS validation
    ├─► No sensitive data in localStorage
    └─► API keys in environment variables
        │
        ▼
Backend
    │
    ├─► CORS middleware
    ├─► Rate limiting (future)
    ├─► Authentication (future)
    └─► Input validation
```

## Performance Optimization

```
Frontend
    │
    ├─► Code splitting by route
    ├─► Lazy loading components
    ├─► Memoization (React.memo)
    ├─► Debounced API calls
    └─► Efficient re-renders
        │
        ▼
Backend
    │
    ├─► Response caching
    ├─► Database connection pooling
    ├─► Async/await patterns
    └─► Efficient queries
```

## Monitoring & Observability

```
Frontend
    │
    ├─► Browser DevTools
    │   ├─► Console logs
    │   ├─► Network tab
    │   └─► React DevTools
    │
    └─► Error tracking (future)
        └─► Sentry / LogRocket
            │
            ▼
Backend
    │
    ├─► Structured logging
    │   └─► JSON logs
    │
    ├─► Health endpoints
    │   └─► /health
    │
    └─► Metrics (future)
        └─► Prometheus / Grafana
```

This architecture provides a solid foundation for the trading platform with clear separation of concerns, type safety, error handling, and scalability.
