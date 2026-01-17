# Quick Start Guide

Get the algorithmic trading platform running in 5 minutes.

## Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)
- Rust 1.70+ (for execution core development)

## Step 1: Start Backend Services

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd <repository-name>

# Start all backend services
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
docker-compose ps
```

Expected output:
```
NAME                    STATUS              PORTS
postgres                Up                  5432
neo4j                   Up                  7474, 7687
redis                   Up                  6379
intelligence-layer      Up                  8000
execution-core          Up                  8001
```

## Step 2: Verify Backend Health

### Windows PowerShell
```powershell
.\scripts\test-integration.ps1
```

### Linux/Mac
```bash
./scripts/test-integration.sh
```

You should see:
```
=== Integration Test Summary ===
Intelligence Layer: HEALTHY
Execution Core:     HEALTHY
```

## Step 3: Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at: **http://localhost:5173**

## Step 4: Explore the Platform

Open your browser to http://localhost:5173 and use function keys to navigate:

- **F1**: Dashboard - System overview
- **F2**: Markets - Live market data
- **F3**: Intelligence - Regime detection
- **F4**: Strategies - Strategy management
- **F5**: Portfolio - Positions and risk
- **F6**: Execution - Order management
- **F7**: Simulation - Backtesting
- **F8**: Data & Models - ML models
- **F9**: System - Health monitoring

## Verify Integration

1. Open browser DevTools (F12)
2. Go to Network tab
3. You should see API calls to:
   - `http://localhost:8000/health`
   - `http://localhost:8000/intelligence/regime`
   - `http://localhost:8000/intelligence/graph-features`

4. Check the Intelligence page (F3):
   - Regime probabilities should update from backend
   - Graph features should display cluster information

## Troubleshooting

### Backend services not starting
```bash
# Check logs
docker-compose logs intelligence-layer
docker-compose logs execution-core

# Restart services
docker-compose restart
```

### Frontend can't connect
```bash
# Check backend is running
curl http://localhost:8000/health
curl http://localhost:8001/health

# Check environment variables
cat frontend/.env.development
```

### Port conflicts
```bash
# Check what's using the ports
netstat -an | findstr "5173 8000 8001"  # Windows
lsof -i :5173,8000,8001                 # Linux/Mac

# Change ports in docker-compose.yml and .env.development
```

## Next Steps

1. **Read the Integration Guide**: See `INTEGRATION_GUIDE.md` for detailed architecture
2. **Review API Documentation**: http://localhost:8000/docs
3. **Check System Validation**: See `SYSTEM_VALIDATION_REPORT.md`
4. **Deploy to Production**: See `DEPLOYMENT_GUIDE.md`

## Development Workflow

### Backend Development
```bash
# Intelligence Layer (Python)
cd intelligence-layer
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
pytest

# Execution Core (Rust)
cd execution-core
cargo build
cargo test
```

### Frontend Development
```bash
cd frontend
npm run dev      # Development server
npm run build    # Production build
npm run preview  # Preview production build
```

### Full System Test
```bash
# Run all tests
make test

# Or individually
cd intelligence-layer && pytest
cd execution-core && cargo test
cd frontend && npm test
```

## Stopping the System

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Stop frontend
# Press Ctrl+C in the terminal running npm run dev
```

## Support

- **Documentation**: See `README.md` and `INTEGRATION_GUIDE.md`
- **API Docs**: http://localhost:8000/docs
- **Logs**: `docker-compose logs -f <service-name>`
- **Health Checks**: `./scripts/test-integration.sh`

## Architecture Overview

```
Frontend (React)          → Intelligence Layer (Python)  → PostgreSQL + pgvector
http://localhost:5173       http://localhost:8000          Neo4j + GDS
                                                           Redis
                          → Execution Core (Rust)
                            http://localhost:8001
```

## Key Features

✅ **Real-time regime detection** with HMM and graph analytics  
✅ **Bloomberg Terminal UI** with professional trading interface  
✅ **Academic rigor** with experiment tracking and reproducibility  
✅ **Shadow execution** for safe strategy testing  
✅ **Risk management** with real-time monitoring  
✅ **Multi-strategy orchestration** with RL-based allocation  

Enjoy trading! 🚀
