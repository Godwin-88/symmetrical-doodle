#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Simple startup script for all trading system servers
.DESCRIPTION
    A simplified version that starts all services without fancy formatting
.EXAMPLE
    .\scripts\start-all-simple.ps1
#>

Write-Host "Starting Algorithmic Trading System..."
Write-Host ""

# Check if running from project root
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "ERROR: Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Step 1: Start Docker services
Write-Host "Step 1: Starting Database Services..." -ForegroundColor Yellow
try {
    docker-compose up -d postgres neo4j redis
    Start-Sleep -Seconds 5
    Write-Host "SUCCESS: Database services started" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Failed to start Docker services. Make sure Docker is running." -ForegroundColor Yellow
}

# Step 2: Start Python Intelligence Layer
Write-Host "Step 2: Starting Python Intelligence Layer (Port 8000)..." -ForegroundColor Yellow
$pythonJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location intelligence-layer
    if (Test-Path "../.venv/Scripts/Activate.ps1") {
        & "../.venv/Scripts/Activate.ps1"
    }
    python -m uvicorn intelligence_layer.main:app --host 0.0.0.0 --port 8000 --reload
}
Write-Host "SUCCESS: Python Intelligence Layer starting (Job ID: $($pythonJob.Id))" -ForegroundColor Green

# Step 3: Start Rust Execution Core
Write-Host "Step 3: Starting Rust Execution Core (Port 8001)..." -ForegroundColor Yellow
$rustExecutionJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location execution-core
    cargo run --release
}
Write-Host "SUCCESS: Rust Execution Core starting (Job ID: $($rustExecutionJob.Id))" -ForegroundColor Green

# Step 4: Start Rust Simulation Engine
Write-Host "Step 4: Starting Rust Simulation Engine (Port 8002)..." -ForegroundColor Yellow
$rustSimulationJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location simulation-engine
    cargo run --release
}
Write-Host "SUCCESS: Rust Simulation Engine starting (Job ID: $($rustSimulationJob.Id))" -ForegroundColor Green

# Step 5: Start React Frontend
Write-Host "Step 5: Starting React Frontend (Port 5173)..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location frontend
    npm run dev
}
Write-Host "SUCCESS: React Frontend starting (Job ID: $($frontendJob.Id))" -ForegroundColor Green

# Wait for services to start
Write-Host ""
Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Display access URLs
Write-Host ""
Write-Host "=== ACCESS URLS ===" -ForegroundColor Cyan
Write-Host "Frontend:              http://localhost:5173" -ForegroundColor Cyan
Write-Host "Intelligence API:      http://localhost:8000" -ForegroundColor Cyan
Write-Host "Intelligence Docs:     http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Execution API:         http://localhost:8001" -ForegroundColor Cyan
Write-Host "Simulation API:        http://localhost:8002" -ForegroundColor Cyan
Write-Host "Neo4j Browser:         http://localhost:7474" -ForegroundColor Cyan
Write-Host ""

# Display job information
Write-Host "=== RUNNING JOBS ===" -ForegroundColor Cyan
Write-Host "Python Intelligence Layer: Job $($pythonJob.Id)"
Write-Host "Rust Execution Core: Job $($rustExecutionJob.Id)"
Write-Host "Rust Simulation Engine: Job $($rustSimulationJob.Id)"
Write-Host "React Frontend: Job $($frontendJob.Id)"
Write-Host ""

Write-Host "=== SYSTEM READY ===" -ForegroundColor Green
Write-Host "All services are starting up!" -ForegroundColor Green
Write-Host "Open http://localhost:5173 in your browser to access the trading system" -ForegroundColor Green
Write-Host ""
Write-Host "To stop all services, run: .\scripts\stop-all.ps1" -ForegroundColor Yellow
Write-Host "To view job output: Get-Job | Receive-Job -Keep" -ForegroundColor Yellow