# Deployment Scripts Guide

This guide covers the automated deployment scripts for starting, stopping, and managing the algorithmic trading system.

## Overview

The platform includes comprehensive scripts for managing all system components across different operating systems. These scripts handle database services, backend components, and the frontend interface with proper dependency management and health checking.

## Script Locations

All deployment scripts are located in the `scripts/` directory:

```
scripts/
├── start-all.ps1           # Windows PowerShell startup
├── start-all.sh            # Linux/Mac startup
├── stop-all.ps1            # Windows PowerShell shutdown
├── stop-all.sh             # Linux/Mac shutdown
├── test-integration.ps1    # Windows integration tests
├── test-integration.sh     # Linux/Mac integration tests
├── validate-setup.ps1      # Windows system validation
├── validate-setup.sh       # Linux/Mac system validation
├── test-deriv-connection.py # Deriv API connection test
└── test-neo4j-aura.py      # Neo4j Aura connection test
```

## Startup Scripts

### Windows PowerShell (start-all.ps1)

```powershell
# Start All Services - Windows PowerShell
# Usage: .\start-all.ps1

Write-Host "Starting Algorithmic Trading System..." -ForegroundColor Green

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH"
    exit 1
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust/Cargo is not installed or not in PATH"
    exit 1
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH"
    exit 1
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js/npm is not installed or not in PATH"
    exit 1
}

# Start database services
Write-Host "Starting database services..." -ForegroundColor Yellow
docker-compose up -d postgres neo4j redis

# Wait for databases to be ready
Write-Host "Waiting for databases to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Initialize databases
Write-Host "Initializing databases..." -ForegroundColor Yellow
docker exec -i trading-postgres psql -U postgres -d trading < database/init/01-init.sql
docker exec -i trading-neo4j cypher-shell -u neo4j -p password < database/init/02-neo4j-init.cypher

# Start Execution Core (Rust)
Write-Host "Starting Execution Core..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd execution-core; cargo run" -WindowStyle Minimized

# Wait for Execution Core to start
Start-Sleep -Seconds 10

# Start Intelligence Layer (Python)
Write-Host "Starting Intelligence Layer..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd intelligence-layer; python -m uvicorn intelligence_layer.main:app --reload --host 0.0.0.0 --port 8000" -WindowStyle Minimized

# Wait for Intelligence Layer to start
Start-Sleep -Seconds 10

# Start Simulation Engine (Rust)
Write-Host "Starting Simulation Engine..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd simulation-engine; cargo run" -WindowStyle Minimized

# Wait for Simulation Engine to start
Start-Sleep -Seconds 5

# Start Frontend (React)
Write-Host "Starting Frontend..." -ForegroundColor Yellow
Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd frontend; npm run dev" -WindowStyle Minimized

Write-Host "System startup complete!" -ForegroundColor Green
Write-Host "Frontend available at: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Intelligence Layer API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Execution Core API: http://localhost:8001" -ForegroundColor Cyan
Write-Host "Simulation Engine API: http://localhost:8002" -ForegroundColor Cyan

Write-Host "Run integration tests with: .\scripts\test-integration.ps1" -ForegroundColor Yellow
```

### Linux/Mac Bash (start-all.sh)

```bash
#!/bin/bash
# Start All Services - Linux/Mac
# Usage: ./start-all.sh

set -e

echo "Starting Algorithmic Trading System..."

# Check prerequisites
echo "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "Rust/Cargo is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "Node.js/npm is required but not installed. Aborting." >&2; exit 1; }

# Start database services
echo "Starting database services..."
docker-compose up -d postgres neo4j redis

# Wait for databases to be ready
echo "Waiting for databases to initialize..."
sleep 30

# Initialize databases
echo "Initializing databases..."
docker exec -i trading-postgres psql -U postgres -d trading < database/init/01-init.sql
docker exec -i trading-neo4j cypher-shell -u neo4j -p password < database/init/02-neo4j-init.cypher

# Start Execution Core (Rust)
echo "Starting Execution Core..."
cd execution-core
cargo run &
EXECUTION_PID=$!
cd ..

# Wait for Execution Core to start
sleep 10

# Start Intelligence Layer (Python)
echo "Starting Intelligence Layer..."
cd intelligence-layer
python3 -m uvicorn intelligence_layer.main:app --reload --host 0.0.0.0 --port 8000 &
INTELLIGENCE_PID=$!
cd ..

# Wait for Intelligence Layer to start
sleep 10

# Start Simulation Engine (Rust)
echo "Starting Simulation Engine..."
cd simulation-engine
cargo run &
SIMULATION_PID=$!
cd ..

# Wait for Simulation Engine to start
sleep 5

# Start Frontend (React)
echo "Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "System startup complete!"
echo "Frontend available at: http://localhost:5173"
echo "Intelligence Layer API: http://localhost:8000"
echo "Execution Core API: http://localhost:8001"
echo "Simulation Engine API: http://localhost:8002"

echo "Process IDs:"
echo "Execution Core: $EXECUTION_PID"
echo "Intelligence Layer: $INTELLIGENCE_PID"
echo "Simulation Engine: $SIMULATION_PID"
echo "Frontend: $FRONTEND_PID"

echo "Run integration tests with: ./scripts/test-integration.sh"
echo "Stop all services with: ./scripts/stop-all.sh"

# Save PIDs for stop script
echo "$EXECUTION_PID" > .execution.pid
echo "$INTELLIGENCE_PID" > .intelligence.pid
echo "$SIMULATION_PID" > .simulation.pid
echo "$FRONTEND_PID" > .frontend.pid
```

## Shutdown Scripts

### Windows PowerShell (stop-all.ps1)

```powershell
# Stop All Services - Windows PowerShell
# Usage: .\stop-all.ps1

Write-Host "Stopping Algorithmic Trading System..." -ForegroundColor Red

# Stop frontend processes
Write-Host "Stopping frontend processes..." -ForegroundColor Yellow
Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*npm run dev*"} | Stop-Process -Force

# Stop Rust processes
Write-Host "Stopping Rust processes..." -ForegroundColor Yellow
Get-Process -Name "execution-core" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "simulation-engine" -ErrorAction SilentlyContinue | Stop-Process -Force

# Stop Python processes
Write-Host "Stopping Python processes..." -ForegroundColor Yellow
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*uvicorn*"} | Stop-Process -Force

# Stop Docker services
Write-Host "Stopping Docker services..." -ForegroundColor Yellow
docker-compose down

Write-Host "System shutdown complete!" -ForegroundColor Green
```

### Linux/Mac Bash (stop-all.sh)

```bash
#!/bin/bash
# Stop All Services - Linux/Mac
# Usage: ./stop-all.sh

echo "Stopping Algorithmic Trading System..."

# Stop processes using saved PIDs
if [ -f .execution.pid ]; then
    echo "Stopping Execution Core..."
    kill $(cat .execution.pid) 2>/dev/null || true
    rm .execution.pid
fi

if [ -f .intelligence.pid ]; then
    echo "Stopping Intelligence Layer..."
    kill $(cat .intelligence.pid) 2>/dev/null || true
    rm .intelligence.pid
fi

if [ -f .simulation.pid ]; then
    echo "Stopping Simulation Engine..."
    kill $(cat .simulation.pid) 2>/dev/null || true
    rm .simulation.pid
fi

if [ -f .frontend.pid ]; then
    echo "Stopping Frontend..."
    kill $(cat .frontend.pid) 2>/dev/null || true
    rm .frontend.pid
fi

# Fallback: kill by process name
echo "Cleaning up remaining processes..."
pkill -f "cargo run" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

# Stop Docker services
echo "Stopping Docker services..."
docker-compose down

echo "System shutdown complete!"
```

## Integration Testing Scripts

### Windows PowerShell (test-integration.ps1)

```powershell
# Integration Tests - Windows PowerShell
# Usage: .\test-integration.ps1

Write-Host "Running Integration Tests..." -ForegroundColor Green

$ErrorCount = 0

# Test database connections
Write-Host "Testing database connections..." -ForegroundColor Yellow

# Test PostgreSQL
try {
    $result = docker exec trading-postgres pg_isready -U postgres
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ PostgreSQL connection successful" -ForegroundColor Green
    } else {
        Write-Host "✗ PostgreSQL connection failed" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "✗ PostgreSQL connection failed: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test Neo4j
try {
    $result = docker exec trading-neo4j cypher-shell -u neo4j -p password "RETURN 1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Neo4j connection successful" -ForegroundColor Green
    } else {
        Write-Host "✗ Neo4j connection failed" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "✗ Neo4j connection failed: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test Redis
try {
    $result = docker exec trading-redis redis-cli ping
    if ($result -eq "PONG") {
        Write-Host "✓ Redis connection successful" -ForegroundColor Green
    } else {
        Write-Host "✗ Redis connection failed" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "✗ Redis connection failed: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test API endpoints
Write-Host "Testing API endpoints..." -ForegroundColor Yellow

# Test Intelligence Layer
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 10
    if ($response.status -eq "healthy") {
        Write-Host "✓ Intelligence Layer API healthy" -ForegroundColor Green
    } else {
        Write-Host "✗ Intelligence Layer API unhealthy" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "✗ Intelligence Layer API failed: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test Execution Core
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 10
    if ($response.status -eq "healthy") {
        Write-Host "✓ Execution Core API healthy" -ForegroundColor Green
    } else {
        Write-Host "✗ Execution Core API unhealthy" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "✗ Execution Core API failed: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test Frontend
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5173" -Method Get -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Frontend accessible" -ForegroundColor Green
    } else {
        Write-Host "✗ Frontend not accessible" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "✗ Frontend failed: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test Deriv API connection
Write-Host "Testing external integrations..." -ForegroundColor Yellow
try {
    python scripts/test-deriv-connection.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Deriv API connection successful" -ForegroundColor Green
    } else {
        Write-Host "✗ Deriv API connection failed" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "✗ Deriv API test failed: $_" -ForegroundColor Red
    $ErrorCount++
}

# Summary
Write-Host "`nIntegration Test Summary:" -ForegroundColor Cyan
if ($ErrorCount -eq 0) {
    Write-Host "All tests passed! ✓" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$ErrorCount test(s) failed! ✗" -ForegroundColor Red
    exit 1
}
```

### Linux/Mac Bash (test-integration.sh)

```bash
#!/bin/bash
# Integration Tests - Linux/Mac
# Usage: ./test-integration.sh

set -e

echo "Running Integration Tests..."

ERROR_COUNT=0

# Test database connections
echo "Testing database connections..."

# Test PostgreSQL
if docker exec trading-postgres pg_isready -U postgres >/dev/null 2>&1; then
    echo "✓ PostgreSQL connection successful"
else
    echo "✗ PostgreSQL connection failed"
    ((ERROR_COUNT++))
fi

# Test Neo4j
if docker exec trading-neo4j cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1; then
    echo "✓ Neo4j connection successful"
else
    echo "✗ Neo4j connection failed"
    ((ERROR_COUNT++))
fi

# Test Redis
if [ "$(docker exec trading-redis redis-cli ping)" = "PONG" ]; then
    echo "✓ Redis connection successful"
else
    echo "✗ Redis connection failed"
    ((ERROR_COUNT++))
fi

# Test API endpoints
echo "Testing API endpoints..."

# Test Intelligence Layer
if curl -s -f "http://localhost:8000/health" | grep -q "healthy"; then
    echo "✓ Intelligence Layer API healthy"
else
    echo "✗ Intelligence Layer API unhealthy"
    ((ERROR_COUNT++))
fi

# Test Execution Core
if curl -s -f "http://localhost:8001/health" | grep -q "healthy"; then
    echo "✓ Execution Core API healthy"
else
    echo "✗ Execution Core API unhealthy"
    ((ERROR_COUNT++))
fi

# Test Frontend
if curl -s -f "http://localhost:5173" >/dev/null 2>&1; then
    echo "✓ Frontend accessible"
else
    echo "✗ Frontend not accessible"
    ((ERROR_COUNT++))
fi

# Test Deriv API connection
echo "Testing external integrations..."
if python3 scripts/test-deriv-connection.py; then
    echo "✓ Deriv API connection successful"
else
    echo "✗ Deriv API connection failed"
    ((ERROR_COUNT++))
fi

# Summary
echo
echo "Integration Test Summary:"
if [ $ERROR_COUNT -eq 0 ]; then
    echo "All tests passed! ✓"
    exit 0
else
    echo "$ERROR_COUNT test(s) failed! ✗"
    exit 1
fi
```

## System Validation Scripts

### Windows PowerShell (validate-setup.ps1)

```powershell
# System Validation - Windows PowerShell
# Usage: .\validate-setup.ps1

Write-Host "Validating System Setup..." -ForegroundColor Green

$ErrorCount = 0

# Check required software
Write-Host "Checking required software..." -ForegroundColor Yellow

$RequiredSoftware = @(
    @{Name="Docker"; Command="docker --version"},
    @{Name="Docker Compose"; Command="docker-compose --version"},
    @{Name="Rust"; Command="cargo --version"},
    @{Name="Python"; Command="python --version"},
    @{Name="Node.js"; Command="node --version"},
    @{Name="npm"; Command="npm --version"}
)

foreach ($Software in $RequiredSoftware) {
    try {
        $version = Invoke-Expression $Software.Command
        Write-Host "✓ $($Software.Name): $version" -ForegroundColor Green
    } catch {
        Write-Host "✗ $($Software.Name): Not installed or not in PATH" -ForegroundColor Red
        $ErrorCount++
    }
}

# Check environment files
Write-Host "Checking environment configuration..." -ForegroundColor Yellow

$EnvFiles = @(".env", "frontend/.env.development")
foreach ($EnvFile in $EnvFiles) {
    if (Test-Path $EnvFile) {
        Write-Host "✓ $EnvFile exists" -ForegroundColor Green
    } else {
        Write-Host "✗ $EnvFile missing" -ForegroundColor Red
        $ErrorCount++
    }
}

# Check Docker daemon
Write-Host "Checking Docker daemon..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    Write-Host "✓ Docker daemon running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker daemon not running" -ForegroundColor Red
    $ErrorCount++
}

# Check ports availability
Write-Host "Checking port availability..." -ForegroundColor Yellow

$RequiredPorts = @(5432, 7687, 6379, 8000, 8001, 8002, 5173)
foreach ($Port in $RequiredPorts) {
    $Connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    if ($Connection.TcpTestSucceeded) {
        Write-Host "⚠ Port $Port is already in use" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Port $Port available" -ForegroundColor Green
    }
}

# Check disk space
Write-Host "Checking disk space..." -ForegroundColor Yellow
$Drive = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
$FreeSpaceGB = [math]::Round($Drive.FreeSpace / 1GB, 2)
if ($FreeSpaceGB -gt 10) {
    Write-Host "✓ Disk space: $FreeSpaceGB GB available" -ForegroundColor Green
} else {
    Write-Host "⚠ Low disk space: $FreeSpaceGB GB available" -ForegroundColor Yellow
}

# Summary
Write-Host "`nValidation Summary:" -ForegroundColor Cyan
if ($ErrorCount -eq 0) {
    Write-Host "System validation passed! Ready to start. ✓" -ForegroundColor Green
    Write-Host "Run: .\scripts\start-all.ps1" -ForegroundColor Cyan
} else {
    Write-Host "$ErrorCount validation error(s) found! ✗" -ForegroundColor Red
    Write-Host "Please fix the issues above before starting the system." -ForegroundColor Yellow
}
```

## Usage Instructions

### Quick Start
```bash
# 1. Validate system setup
./scripts/validate-setup.sh        # Linux/Mac
.\scripts\validate-setup.ps1       # Windows

# 2. Start all services
./scripts/start-all.sh              # Linux/Mac
.\scripts\start-all.ps1             # Windows

# 3. Run integration tests
./scripts/test-integration.sh       # Linux/Mac
.\scripts\test-integration.ps1      # Windows

# 4. Stop all services
./scripts/stop-all.sh               # Linux/Mac
.\scripts\stop-all.ps1              # Windows
```

### Development Workflow
```bash
# Start only databases for development
docker-compose up -d postgres neo4j redis

# Start individual components manually
cd execution-core && cargo run
cd intelligence-layer && uvicorn intelligence_layer.main:app --reload
cd frontend && npm run dev

# Run specific tests
python scripts/test-deriv-connection.py
python scripts/test-neo4j-aura.py
```

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check what's using a port
netstat -tulpn | grep :8000    # Linux
netstat -ano | findstr :8000   # Windows

# Kill process using port
kill -9 $(lsof -t -i:8000)     # Linux
taskkill /PID <PID> /F          # Windows
```

#### Docker Issues
```bash
# Reset Docker state
docker-compose down -v
docker system prune -f

# Check Docker logs
docker-compose logs postgres
docker-compose logs neo4j
docker-compose logs redis
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
docker exec -it trading-postgres psql -U postgres -d trading

# Test Neo4j connection
docker exec -it trading-neo4j cypher-shell -u neo4j -p password

# Test Redis connection
docker exec -it trading-redis redis-cli
```

### Script Permissions

#### Linux/Mac
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Fix line endings if needed
dos2unix scripts/*.sh
```

#### Windows
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run as administrator if needed
Start-Process powershell -Verb runAs
```

## Monitoring and Logging

### Process Monitoring
```bash
# Monitor all processes
ps aux | grep -E "(cargo|uvicorn|npm)"

# Monitor resource usage
top -p $(pgrep -d, -f "cargo|uvicorn|npm")
```

### Log Collection
```bash
# Collect all logs
mkdir -p logs
docker-compose logs > logs/docker.log
journalctl -u docker > logs/system.log

# Application logs
tail -f execution-core/logs/app.log
tail -f intelligence-layer/logs/app.log
```

## Automation and CI/CD

### GitHub Actions Integration
```yaml
# .github/workflows/integration-test.yml
name: Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup environment
        run: |
          cp .env.example .env
          ./scripts/validate-setup.sh
      - name: Start services
        run: ./scripts/start-all.sh
      - name: Run tests
        run: ./scripts/test-integration.sh
      - name: Cleanup
        run: ./scripts/stop-all.sh
```

### Cron Job Setup
```bash
# Daily health check
0 9 * * * /path/to/project/scripts/test-integration.sh

# Weekly system restart
0 2 * * 0 /path/to/project/scripts/stop-all.sh && sleep 30 && /path/to/project/scripts/start-all.sh
```

This comprehensive script system provides robust, automated deployment and management capabilities for the algorithmic trading platform across different operating systems and environments.