#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start all trading system servers
.DESCRIPTION
    Launches all necessary servers for the algorithmic trading system:
    - Database services (PostgreSQL, Neo4j, Redis)
    - Python Intelligence Layer (port 8000)
    - Rust Execution Core (port 8001)
    - Rust Simulation Engine (port 8002)
    - React Frontend (port 5173)
.EXAMPLE
    .\scripts\start-all.ps1
#>

$ErrorActionPreference = "Continue"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header($message) {
    Write-Host ""
    Write-ColorOutput Yellow "============================================================"
    Write-ColorOutput Yellow $message
    Write-ColorOutput Yellow "============================================================"
    Write-Host ""
}

function Write-Success($message) {
    Write-ColorOutput Green "✓ $message"
}

function Write-Info($message) {
    Write-ColorOutput Cyan "→ $message"
}

function Write-Warning($message) {
    Write-ColorOutput Yellow "⚠ $message"
}

function Write-Error($message) {
    Write-ColorOutput Red "✗ $message"
}

# Check if running from project root
if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "Please run this script from the project root directory"
    exit 1
}

Write-Header "ALGORITHMIC TRADING SYSTEM - STARTUP"

# Step 1: Start Docker services
Write-Header "STEP 1: Starting Database Services (Docker)"

# Check if using Neo4j Aura (cloud) or local Neo4j
$usingNeo4jAura = $false
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "neo4j\+s://.*\.databases\.neo4j\.io") {
        $usingNeo4jAura = $true
        Write-Info "Detected Neo4j Aura (cloud database) configuration"
    }
}

Write-Info "Starting PostgreSQL and Redis..."
if ($usingNeo4jAura) {
    Write-Info "Using Neo4j Aura (cloud) - skipping local Neo4j"
} else {
    Write-Info "Starting local Neo4j..."
}

try {
    docker-compose up -d
    Start-Sleep -Seconds 5
    Write-Success "Database services started"
    Write-Info "PostgreSQL: localhost:5432"
    if (-not $usingNeo4jAura) {
        Write-Info "Neo4j: localhost:7474 (browser), localhost:7687 (bolt)"
    } else {
        Write-Info "Neo4j: Using Neo4j Aura (cloud)"
    }
    Write-Info "Redis: localhost:6379"
} catch {
    Write-Warning "Failed to start Docker services. Make sure Docker is running."
    Write-Info "You can start them manually with: docker-compose up -d"
}

# Step 2: Start Python Intelligence Layer
Write-Header "STEP 2: Starting Python Intelligence Layer (Port 8000)"

$pythonJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location intelligence-layer
    
    # Activate virtual environment if it exists
    if (Test-Path "../.venv/Scripts/Activate.ps1") {
        & "../.venv/Scripts/Activate.ps1"
    }
    
    # Start the server
    python -m uvicorn intelligence_layer.main:app --host 0.0.0.0 --port 8000 --reload
}

Write-Success "Python Intelligence Layer starting (Job ID: $($pythonJob.Id))"
Write-Info "API will be available at: http://localhost:8000"
Write-Info "API docs at: http://localhost:8000/docs"

# Step 3: Start Rust Execution Core
Write-Header "STEP 3: Starting Rust Execution Core (Port 8001)"

$rustExecutionJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location execution-core
    
    # Build and run
    cargo run --release
}

Write-Success "Rust Execution Core starting (Job ID: $($rustExecutionJob.Id))"
Write-Info "Execution API will be available at: http://localhost:8001"

# Step 4: Start Rust Simulation Engine
Write-Header "STEP 4: Starting Rust Simulation Engine (Port 8002)"

$rustSimulationJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location simulation-engine
    
    # Build and run
    cargo run --release
}

Write-Success "Rust Simulation Engine starting (Job ID: $($rustSimulationJob.Id))"
Write-Info "Simulation API will be available at: http://localhost:8002"

# Step 5: Start React Frontend
Write-Header "STEP 5: Starting React Frontend (Port 5173)"

$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location frontend
    
    # Start dev server
    npm run dev
}

Write-Success "React Frontend starting (Job ID: $($frontendJob.Id))"
Write-Info "Frontend will be available at: http://localhost:5173"

# Wait for services to start
Write-Header "Waiting for services to initialize..."
Start-Sleep -Seconds 10

# Check service status
Write-Header "SERVICE STATUS"

function Test-ServiceHealth($url, $name) {
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Success "$name is running"
            return $true
        }
    } catch {
        Write-Warning "$name is starting... (may take a moment)"
        return $false
    }
}

$pythonReady = Test-ServiceHealth "http://localhost:8000/health" "Python Intelligence Layer"
$executionReady = Test-ServiceHealth "http://localhost:8001/health" "Rust Execution Core"
$simulationReady = Test-ServiceHealth "http://localhost:8002/health" "Rust Simulation Engine"
$frontendReady = Test-ServiceHealth "http://localhost:5173" "React Frontend"

# Display job information
Write-Header "RUNNING JOBS"
Write-Info "Python Intelligence Layer: Job $($pythonJob.Id)"
Write-Info "Rust Execution Core: Job $($rustExecutionJob.Id)"
Write-Info "Rust Simulation Engine: Job $($rustSimulationJob.Id)"
Write-Info "React Frontend: Job $($frontendJob.Id)"

# Display access URLs
Write-Header "ACCESS URLS"
Write-ColorOutput Cyan "Frontend:              http://localhost:5173"
Write-ColorOutput Cyan "Intelligence API:      http://localhost:8000"
Write-ColorOutput Cyan "Intelligence Docs:     http://localhost:8000/docs"
Write-ColorOutput Cyan "Execution API:         http://localhost:8001"
Write-ColorOutput Cyan "Simulation API:        http://localhost:8002"
if (-not $usingNeo4jAura) {
    Write-ColorOutput Cyan "Neo4j Browser:         http://localhost:7474"
} else {
    Write-ColorOutput Cyan "Neo4j Aura Console:    https://console.neo4j.io/"
}
Write-Host ""

# Display monitoring commands
Write-Header "MONITORING COMMANDS"
Write-Info "View Python logs:      Receive-Job -Id $($pythonJob.Id) -Keep"
Write-Info "View Execution logs:   Receive-Job -Id $($rustExecutionJob.Id) -Keep"
Write-Info "View Simulation logs:  Receive-Job -Id $($rustSimulationJob.Id) -Keep"
Write-Info "View Frontend logs:    Receive-Job -Id $($frontendJob.Id) -Keep"
Write-Info "View all jobs:         Get-Job"
Write-Info "Stop all services:     .\scripts\stop-all.ps1"
Write-Host ""

# Display Deriv status
Write-Header "DERIV API STATUS"
if ($env:DERIV_API_TOKEN) {
    Write-Success "Deriv API token configured"
    Write-Info "Demo trading enabled"
    Write-Info "Test connection: python scripts\test-deriv-connection.py"
} else {
    Write-Warning "Deriv API token not configured"
    Write-Info "Set DERIV_API_TOKEN in .env file to enable demo trading"
}

Write-Header "SYSTEM READY"
Write-ColorOutput Green "All services are starting up!"
Write-ColorOutput Green "Open http://localhost:5173 in your browser to access the trading system"
Write-Host ""
Write-Info "Press Ctrl+C to stop monitoring (services will continue running)"
Write-Info "To stop all services, run: .\scripts\stop-all.ps1"
Write-Host ""

# Keep script running and show logs
Write-Header "LIVE LOGS (Press Ctrl+C to exit)"
Write-Host ""

try {
    while ($true) {
        # Show recent output from jobs
        $jobs = @($pythonJob, $rustExecutionJob, $rustSimulationJob, $frontendJob)
        foreach ($job in $jobs) {
            $output = Receive-Job -Id $job.Id -ErrorAction SilentlyContinue
            if ($output) {
                Write-Host $output
            }
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host ""
    Write-Info "Monitoring stopped. Services are still running."
    Write-Info "To stop all services, run: .\scripts\stop-all.ps1"
}
