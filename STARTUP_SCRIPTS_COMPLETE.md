# Startup Scripts - Implementation Complete

## Overview

Comprehensive startup scripts have been created to launch all trading system services with a single command.

## Files Created

### 1. **Windows PowerShell Scripts**

**`scripts/start-all.ps1`**
- Starts all services in parallel
- Shows colored output and status
- Monitors service health
- Displays live logs
- Creates PowerShell background jobs

**`scripts/stop-all.ps1`**
- Stops all running services
- Kills processes on ports
- Stops Docker containers
- Cleans up background jobs

### 2. **Linux/Mac Bash Scripts**

**`scripts/start-all.sh`**
- Starts all services in background
- Creates PID files for tracking
- Logs to `logs/` directory
- Shows service status

**`scripts/stop-all.sh`**
- Stops all services by PID
- Kills processes on ports
- Stops Docker containers
- Cleans up PID files

### 3. **Documentation**

**`STARTUP_GUIDE.md`**
- Complete startup documentation
- Prerequisites and installation
- Troubleshooting guide
- Monitoring commands
- Configuration instructions

**`scripts/README.md`**
- Quick reference for scripts
- Usage examples

## Services Started

The scripts start all necessary services:

### 1. Database Services (Docker)
- PostgreSQL (port 5432)
- Neo4j (ports 7474, 7687)
- Redis (port 6379)

### 2. Python Intelligence Layer (port 8000)
- FastAPI backend
- ML models
- Deriv integration
- Strategy orchestration

### 3. Rust Execution Core (port 8001)
- Order execution
- Risk management
- Portfolio tracking

### 4. Rust Simulation Engine (port 8002)
- Backtesting
- Strategy simulation

### 5. React Frontend (port 5173)
- Trading dashboard
- Real-time UI

## Usage

### Quick Start

**Windows:**
```powershell
.\scripts\start-all.ps1
```

**Linux/Mac:**
```bash
./scripts/start-all.sh
```

### Stop All

**Windows:**
```powershell
.\scripts\stop-all.ps1
```

**Linux/Mac:**
```bash
./scripts/stop-all.sh
```

## Features

### ✅ Parallel Startup
- All services start simultaneously
- Faster than sequential startup
- Efficient resource usage

### ✅ Health Checks
- Automatic service health verification
- Shows which services are ready
- Displays connection status

### ✅ Colored Output
- Green for success
- Yellow for warnings
- Red for errors
- Cyan for info

### ✅ Process Management
- Tracks all running processes
- Easy to stop individual services
- Clean shutdown

### ✅ Logging
- All output logged to files (Linux/Mac)
- Live log viewing (Windows)
- Easy debugging

### ✅ Status Display
- Shows all access URLs
- Displays process IDs
- Shows Deriv connection status

## Access URLs

After startup, services are available at:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Intelligence API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Execution API | http://localhost:8001 |
| Simulation API | http://localhost:8002 |
| Neo4j Browser | http://localhost:7474 |

## Monitoring

### Windows

```powershell
# View job logs
Receive-Job -Id <JobId> -Keep

# List all jobs
Get-Job

# Stop specific job
Stop-Job -Id <JobId>
```

### Linux/Mac

```bash
# View logs
tail -f logs/intelligence-layer.log
tail -f logs/execution-core.log
tail -f logs/simulation-engine.log
tail -f logs/frontend.log

# View all logs
tail -f logs/*.log
```

## Error Handling

### Port Conflicts

Scripts automatically detect and handle:
- Ports already in use
- Failed service starts
- Docker connection issues

### Service Failures

If a service fails to start:
1. Check the logs
2. Verify prerequisites installed
3. Check port availability
4. Review error messages

## Platform Support

### ✅ Windows
- PowerShell 5.1+
- PowerShell Core 7+
- Background jobs
- Colored output

### ✅ Linux
- Bash 4.0+
- Background processes
- PID file management
- Log files

### ✅ macOS
- Bash 3.2+
- Background processes
- PID file management
- Log files

## Prerequisites

Before running scripts:

1. **Docker Desktop** - For database services
2. **Python 3.9+** - For intelligence layer
3. **Rust & Cargo** - For execution/simulation
4. **Node.js 18+** - For frontend
5. **Dependencies installed** - Run setup first

## Configuration

### Environment Variables

Scripts use `.env` file for configuration:

```bash
# Deriv API
DERIV_APP_ID=118029
DERIV_API_TOKEN=your_token_here

# Database
TRADING_DATABASE__POSTGRES_URL=postgresql://...

# Ports
INTELLIGENCE_API_PORT=8000
```

## Troubleshooting

### Docker Won't Start

```bash
# Check Docker is running
docker ps

# Restart Docker Desktop
# Then run script again
```

### Python Service Fails

```bash
# Check Python version
python --version

# Reinstall dependencies
cd intelligence-layer
pip install -e .
```

### Rust Build Fails

```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

### Port Already in Use

**Windows:**
```powershell
# Find and kill process
Get-NetTCPConnection -LocalPort 8000
Stop-Process -Id <PID> -Force
```

**Linux/Mac:**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

## Development Tips

### Start Individual Services

You can start services individually for development:

```bash
# Just databases
docker-compose up -d

# Just Python backend
cd intelligence-layer
python -m uvicorn intelligence_layer.main:app --reload

# Just frontend
cd frontend
npm run dev
```

### Watch Logs in Real-Time

**Windows:**
```powershell
# Watch specific service
while($true) { Receive-Job -Id <JobId>; Start-Sleep 1 }
```

**Linux/Mac:**
```bash
# Watch all logs
tail -f logs/*.log
```

## Production Deployment

For production:

1. Build optimized versions
2. Use production Docker Compose
3. Configure production environment
4. Set up monitoring
5. Enable SSL/TLS

See `DEPLOYMENT_GUIDE.md` for details.

## Testing

### Verify All Services

```bash
# Run validation script
./scripts/validate-setup.sh  # Linux/Mac
.\scripts\validate-setup.ps1  # Windows

# Test Deriv connection
python scripts/test-deriv-connection.py

# Run integration tests
./scripts/test-integration.sh  # Linux/Mac
.\scripts\test-integration.ps1  # Windows
```

## Summary

✅ **One-command startup** - Start everything with one script  
✅ **Cross-platform** - Works on Windows, Linux, and Mac  
✅ **Parallel execution** - All services start simultaneously  
✅ **Health monitoring** - Automatic service health checks  
✅ **Easy logging** - View logs for debugging  
✅ **Clean shutdown** - Stop all services cleanly  
✅ **Error handling** - Graceful failure handling  
✅ **Status display** - See all URLs and PIDs  

Your trading system can now be started and stopped with a single command!
