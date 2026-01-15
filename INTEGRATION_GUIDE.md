# Frontend-Backend Integration Guide

This guide explains how the frontend integrates with the backend services (Intelligence Layer and Execution Core).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│                    (React + TypeScript)                      │
│                    http://localhost:5173                     │
└────────────┬────────────────────────────────┬───────────────┘
             │                                 │
             │ HTTP/REST                       │ HTTP/REST
             │                                 │
┌────────────▼──────────────┐    ┌────────────▼──────────────┐
│   Intelligence Layer      │    │     Execution Core        │
│   (Python FastAPI)        │    │        (Rust)             │
│   http://localhost:8000   │    │   http://localhost:8001   │
└────────────┬──────────────┘    └────────────┬──────────────┘
             │                                 │
             │                                 │
┌────────────▼──────────────┐    ┌────────────▼──────────────┐
│  PostgreSQL + pgvector    │    │      Event Bus            │
│  Neo4j + GDS              │    │   Portfolio Manager       │
│  Redis                    │    │   Risk Engine             │
└───────────────────────────┘    └───────────────────────────┘
```

## Services

### Intelligence Layer (Port 8000)
- **Health Check**: `GET /health`
- **Regime Inference**: `GET /intelligence/regime?asset_id={asset}`
- **Graph Features**: `GET /intelligence/graph-features?asset_id={asset}`
- **RL State Assembly**: `GET /intelligence/state?asset_ids={assets}&strategy_ids={strategies}`
- **Regime Training**: `POST /intelligence/regime/train`
- **Graph Analysis**: `POST /intelligence/graph/analyze?analysis_type={type}`

### Execution Core (Port 8001)
- **Health Check**: `GET /health`
- Portfolio management
- Risk monitoring
- Order execution
- Shadow execution mode

## Frontend Integration

### API Services

#### 1. API Client (`frontend/src/services/api.ts`)
Base HTTP client with error handling:
```typescript
import { intelligenceApi, executionApi } from '@/services/api';

// Health checks
await checkIntelligenceHealth();
await checkExecutionHealth();
```

#### 2. Intelligence Service (`frontend/src/services/intelligenceService.ts`)
Typed API calls for Intelligence Layer:
```typescript
import { 
  getRegimeInference, 
  getGraphFeatures, 
  assembleRLState 
} from '@/services/intelligenceService';

// Get regime data
const regime = await getRegimeInference('EURUSD');

// Get graph features
const graph = await getGraphFeatures('EURUSD');

// Assemble RL state
const state = await assembleRLState(['EURUSD', 'GBPUSD'], ['MOMENTUM_ALPHA']);
```

#### 3. WebSocket Service (`frontend/src/services/websocketService.ts`)
Real-time updates (when backend implements WebSocket):
```typescript
import { intelligenceWs } from '@/services/websocketService';

// Connect
intelligenceWs.connect();

// Subscribe to events
intelligenceWs.on('regime_update', (data) => {
  console.log('Regime updated:', data);
});

// Send messages
intelligenceWs.send('subscribe', { channel: 'regimes' });
```

### State Management

The Zustand store (`frontend/src/app/store/tradingStore.ts`) integrates with backend:

```typescript
const {
  // Data
  regimes,
  currentRegime,
  intelligenceSignals,
  
  // Loading state
  isLoading,
  error,
  lastUpdate,
  
  // Actions
  fetchRegimeData,
  fetchGraphFeatures,
  fetchRLState,
  checkHealth,
} = useTradingStore();

// Fetch data
await fetchRegimeData('EURUSD');
await fetchGraphFeatures('EURUSD');
await fetchRLState(['EURUSD', 'GBPUSD']);
await checkHealth();
```

### Component Integration

Components automatically fetch and display backend data:

#### Intelligence Component
```typescript
// Fetches regime data on mount and every 30 seconds
useEffect(() => {
  const fetchData = async () => {
    await fetchRegimeData('EURUSD');
    await fetchGraphFeatures('EURUSD');
  };
  
  fetchData();
  const interval = setInterval(fetchData, 30000);
  return () => clearInterval(interval);
}, []);
```

#### App Component
```typescript
// Health checks every 30 seconds
useEffect(() => {
  checkHealth();
  const interval = setInterval(checkHealth, 30000);
  return () => clearInterval(interval);
}, []);
```

## Environment Configuration

### Development (`.env.development`)
```bash
VITE_INTELLIGENCE_API_URL=http://localhost:8000
VITE_EXECUTION_API_URL=http://localhost:8001
VITE_WS_INTELLIGENCE_URL=ws://localhost:8000/ws
VITE_WS_EXECUTION_URL=ws://localhost:8001/ws
VITE_POLLING_INTERVAL=5000
VITE_HEALTH_CHECK_INTERVAL=30000
```

### Production (`.env.production`)
```bash
VITE_INTELLIGENCE_API_URL=https://api.trading.example.com
VITE_EXECUTION_API_URL=https://execution.trading.example.com
VITE_WS_INTELLIGENCE_URL=wss://api.trading.example.com/ws
VITE_WS_EXECUTION_URL=wss://execution.trading.example.com/ws
VITE_POLLING_INTERVAL=10000
VITE_HEALTH_CHECK_INTERVAL=60000
```

## Running the System

### 1. Start Backend Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Or start individually
docker-compose up -d postgres neo4j redis
docker-compose up -d intelligence-layer
docker-compose up -d execution-core
```

### 2. Verify Backend Health

```bash
# Windows PowerShell
.\scripts\test-integration.ps1

# Linux/Mac
./scripts/test-integration.sh
```

Expected output:
```
=== Frontend-Backend Integration Test ===

1. Checking backend services...
   Testing Intelligence Layer (port 8000)... OK
   Testing Execution Core (port 8001)... OK

2. Testing Intelligence Layer API endpoints...
   Testing /intelligence/regime... OK
      Regime probabilities: {"low_vol_trending":0.33,"high_vol_ranging":0.42,"crisis":0.11}
   Testing /intelligence/graph-features... OK
      Cluster: cluster_2
   Testing /intelligence/state... OK
      Current regime: regime_1

=== Integration Test Summary ===
Intelligence Layer: HEALTHY
Execution Core:     HEALTHY

All services are running. You can now start the frontend:
  cd frontend
  npm run dev

Frontend will be available at: http://localhost:5173
```

### 3. Start Frontend

```bash
cd frontend
npm install  # First time only
npm run dev
```

Frontend will be available at: http://localhost:5173

## Data Flow

### 1. Initial Load
```
User opens frontend
  → App component mounts
  → checkHealth() called
  → Intelligence component mounts
  → fetchRegimeData('EURUSD') called
  → fetchGraphFeatures('EURUSD') called
  → Data displayed in UI
```

### 2. Periodic Updates
```
Every 30 seconds:
  → checkHealth() updates system status
  → fetchRegimeData() updates regime probabilities
  → fetchGraphFeatures() updates graph metrics
  → UI automatically re-renders with new data
```

### 3. User Interactions
```
User clicks on regime
  → setSelectedRegime(regimeId) called
  → UI shows regime details
  → No backend call needed (data already loaded)

User switches to different asset
  → fetchRegimeData(newAssetId) called
  → fetchGraphFeatures(newAssetId) called
  → UI updates with new asset data
```

## Error Handling

### Network Errors
```typescript
try {
  await fetchRegimeData('EURUSD');
} catch (error) {
  // Error stored in state
  // Displayed in UI as red banner
  // Automatic retry on next poll
}
```

### Backend Down
```typescript
// Health check fails
checkHealth() → systemStatus: 'DOWN', connectionStatus: 'DISCONNECTED'

// UI shows:
// - Red system status indicator
// - "DISCONNECTED" in status bar
// - Error messages in components
```

### Partial Failure
```typescript
// Intelligence Layer up, Execution Core down
checkHealth() → systemStatus: 'DEGRADED', connectionStatus: 'DELAYED'

// UI shows:
// - Yellow system status indicator
// - "DELAYED" in status bar
// - Some features still work
```

## Testing Integration

### Manual Testing
1. Start backend services
2. Run integration test script
3. Start frontend
4. Open browser to http://localhost:5173
5. Check browser console for API calls
6. Verify data updates every 30 seconds

### API Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Regime inference
curl "http://localhost:8000/intelligence/regime?asset_id=EURUSD"

# Graph features
curl "http://localhost:8000/intelligence/graph-features?asset_id=EURUSD"

# RL state
curl "http://localhost:8000/intelligence/state?asset_ids=EURUSD,GBPUSD"
```

### Browser DevTools
1. Open browser DevTools (F12)
2. Go to Network tab
3. Filter by "Fetch/XHR"
4. Watch API calls in real-time
5. Inspect request/response data

## Troubleshooting

### Frontend can't connect to backend
- Check backend services are running: `docker-compose ps`
- Check ports are not blocked: `netstat -an | findstr "8000 8001"`
- Check CORS configuration in `intelligence-layer/src/intelligence_layer/main.py`
- Check environment variables in `.env.development`

### Data not updating
- Check browser console for errors
- Verify polling intervals in store
- Check network tab for failed requests
- Verify backend is returning valid JSON

### CORS errors
- Ensure frontend URL is in CORS allow_origins
- Check browser console for specific CORS error
- Verify backend CORS middleware is configured

### WebSocket not connecting (future)
- Check WebSocket URL in environment
- Verify backend WebSocket endpoint exists
- Check browser console for WebSocket errors
- Ensure firewall allows WebSocket connections

## Next Steps

### Immediate
- [x] HTTP REST API integration
- [x] Health checks
- [x] Regime data fetching
- [x] Graph features fetching
- [x] Error handling
- [x] Loading states

### Short-term
- [ ] WebSocket implementation in backend
- [ ] Real-time data streaming
- [ ] Authentication/authorization
- [ ] User session management
- [ ] Advanced error recovery

### Long-term
- [ ] Offline mode with service workers
- [ ] Data caching strategies
- [ ] Optimistic UI updates
- [ ] Request batching
- [ ] GraphQL migration (optional)

## API Documentation

Full API documentation available at:
- Intelligence Layer: http://localhost:8000/docs (Swagger UI)
- Intelligence Layer: http://localhost:8000/redoc (ReDoc)

## Support

For issues or questions:
1. Check logs: `docker-compose logs intelligence-layer`
2. Check logs: `docker-compose logs execution-core`
3. Review this integration guide
4. Check browser console for frontend errors
