# Trading System - Startup Guide

## Quick Start

### Windows (PowerShell)

```powershell
# Start all services
.\scripts\start-all.ps1

# Stop all services
.\scripts\stop-all.ps1
```

### Linux/Mac (Bash)

```bash
# Make scripts executable (first time only)
chmod +x scripts/start-all.sh scripts/stop-all.sh

# Start all services
./scripts/start-all.sh

# Stop all services
./scripts/stop-all.sh
```

## What Gets Started

The startup script launches all necessary services:

### 1. **Database Services** (Docker)
- **PostgreSQL** (port 5432) - Main database
- **Neo4j** (ports 7474, 7687) - Graph database
- **Redis** (port 6379) - Cache and message broker

### 2. **Python Intelligence Layer** (port 8000)
- Machine learning models
- Strategy orchestration
- Market analytics
- Deriv API integration
- API docs at: http://localhost:8000/docs

### 3. **Rust Execution Core** (port 8001)
- Order execution
- Risk management
- Portfolio tracking
- High-performance trading engine

### 4. **Rust Simulation Engine** (port 8002)
- Backtesting
- Strategy simulation
- Performance analysis

### 5. **React Frontend** (port 5173)
- Trading dashboard
- Real-time market data
- Portfolio management
- Strategy monitoring

## Access URLs

Once started, access the system at:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | Main trading interface |
| **Intelligence API** | http://localhost:8000 | Python backend API |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs |
| **Execution API** | http://localhost:8001 | Rust execution engine |
| **Simulation API** | http://localhost:8002 | Rust simulation engine |
| **Neo4j Browser** | http://localhost:7474 | Graph database UI |

## Monitoring

### View Logs

**Windows:**
```powershell
# View specific service logs
Receive-Job -Id <JobId> -Keep

# View all jobs
Get-Job
```

**Linux/Mac:**
```bash
# View specific service logs
tail -f logs/intelligence-layer.log
tail -f logs/execution-core.log
tail -f logs/simulation-engine.log
tail -f logs/frontend.log

# View all logs
tail -f logs/*.log
```

### Check Service Health

```bash
# Python Intelligence Layer
curl http://localhost:8000/health

# Rust Execution Core
curl http://localhost:8001/health

# Rust Simulation Engine
curl http://localhost:8002/health

# Frontend
curl http://localhost:5173
```

## Prerequisites

Before running the startup script, ensure you have:

### Required Software

1. **Docker & Docker Compose**
   - Download: https://www.docker.com/products/docker-desktop

2. **Python 3.9+**
   - Download: https://www.python.org/downloads/
   - Virtual environment recommended

3. **Rust & Cargo**
   - Download: https://rustup.rs/

4. **Node.js 18+**
   - Download: https://nodejs.org/

### Installation Steps

```bash
# 1. Install Python dependencies
cd intelligence-layer
pip install -e .
cd ..

# 2. Build Rust projects
cd execution-core
cargo build --release
cd ../simulation-engine
cargo build --release
cd ..

# 3. Install frontend dependencies
cd frontend
npm install
cd ..

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings
```

## Configuration

### Environment Variables

Edit `.env` file to configure:

```bash
# Deriv API (Demo Trading)
DERIV_APP_ID=118029
DERIV_API_TOKEN=your_token_here
DERIV_DEMO_MODE=true

# Database
TRADING_DATABASE__POSTGRES_URL=postgresql://postgres:password@localhost:5432/trading_system
TRADING_DATABASE__NEO4J_URL=bolt://localhost:7687

# API Ports
INTELLIGENCE_API_PORT=8000
```

## Troubleshooting

### Port Already in Use

If you get "port already in use" errors:

**Windows:**
```powershell
# Find process using port
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
# Kill process
Stop-Process -Id <ProcessId> -Force
```

**Linux/Mac:**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

### Docker Services Won't Start

```bash
# Check Docker is running
docker ps

# Restart Docker services
docker-compose down
docker-compose up -d

# View Docker logs
docker-compose logs
```

### Python Service Fails

```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
cd intelligence-layer
pip install -e . --force-reinstall

# Check for errors
python -m uvicorn intelligence_layer.main:app --host 0.0.0.0 --port 8000
```

### Rust Services Fail to Build

```bash
# Update Rust
rustup update

# Clean and rebuild
cd execution-core
cargo clean
cargo build --release

# Check for errors
cargo run
```

### Frontend Won't Start

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install

# Start manually
npm run dev
```

## Development Mode

For development, you can start services individually:

### Start Database Only

```bash
docker-compose up -d
```

### Start Python Backend Only

```bash
cd intelligence-layer
python -m uvicorn intelligence_layer.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Rust Execution Only

```bash
cd execution-core
cargo run
```

### Start Frontend Only

```bash
cd frontend
npm run dev
```

## Production Deployment

For production, use:

```bash
# Build optimized versions
cd frontend
npm run build

cd ../execution-core
cargo build --release

cd ../simulation-engine
cargo build --release

# Use production Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## Deriv Integration

### Test Deriv Connection

```bash
# Test your Deriv API connection
python scripts/test-deriv-connection.py
```

Expected output:
```
✅ Connected successfully
✅ Account: VRTC12345678
✅ Balance: 10000.00 USD
✅ Demo Mode: True
```

### Enable Deriv Trading

1. Get API token from: https://app.deriv.com/account/api-token
2. Add to `.env`:
   ```bash
   DERIV_API_TOKEN=your_token_here
   ```
3. Restart services

## System Health Check

Run the validation script to check all components:

**Windows:**
```powershell
.\scripts\validate-setup.ps1
```

**Linux/Mac:**
```bash
./scripts/validate-setup.sh
```

## Stopping Services

### Stop All Services

**Windows:**
```powershell
.\scripts\stop-all.ps1
```

**Linux/Mac:**
```bash
./scripts/stop-all.sh
```

### Stop Individual Services

**Windows:**
```powershell
# Stop specific job
Stop-Job -Id <JobId>
Remove-Job -Id <JobId>

# Stop Docker
docker-compose down
```

**Linux/Mac:**
```bash
# Kill by PID
kill $(cat logs/intelligence-layer.pid)

# Stop Docker
docker-compose down
```

## Next Steps

Once all services are running:

1. **Open Frontend**: http://localhost:5173
2. **Check System Status**: F1 - System Dashboard
3. **View Live Market Data**: F3 - Markets (with Deriv integration)
4. **Monitor Portfolio**: F5 - Portfolio
5. **Test Strategies**: F4 - Strategies
6. **Run Simulations**: F7 - Simulation

## Support

- **Documentation**: See `README.md` for system overview
- **API Docs**: http://localhost:8000/docs
- **Deriv Guide**: See `DERIV_INTEGRATION_GUIDE.md`
- **Architecture**: See `ARCHITECTURE_SUMMARY.md`

## Summary

✅ **One command starts everything**  
✅ **All services run in parallel**  
✅ **Automatic health checks**  
✅ **Easy monitoring and logs**  
✅ **Simple stop command**  
✅ **Works on Windows, Linux, and Mac**  

Your algorithmic trading system is now ready to use!
