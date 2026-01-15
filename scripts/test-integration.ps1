# Integration Test Script for Frontend-Backend Connection
# Tests the connection between frontend and backend services

Write-Host "=== Frontend-Backend Integration Test ===" -ForegroundColor Cyan
Write-Host ""

# Check if services are running
Write-Host "1. Checking backend services..." -ForegroundColor Yellow

# Test Intelligence Layer
Write-Host "   Testing Intelligence Layer (port 8000)..." -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host " OK" -ForegroundColor Green
        $intelligenceHealthy = $true
    } else {
        Write-Host " FAILED (Status: $($response.StatusCode))" -ForegroundColor Red
        $intelligenceHealthy = $false
    }
} catch {
    Write-Host " FAILED (Not responding)" -ForegroundColor Red
    $intelligenceHealthy = $false
}

# Test Execution Core
Write-Host "   Testing Execution Core (port 8001)..." -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host " OK" -ForegroundColor Green
        $executionHealthy = $true
    } else {
        Write-Host " FAILED (Status: $($response.StatusCode))" -ForegroundColor Red
        $executionHealthy = $false
    }
} catch {
    Write-Host " FAILED (Not responding)" -ForegroundColor Red
    $executionHealthy = $false
}

Write-Host ""

# Test Intelligence Layer API endpoints
if ($intelligenceHealthy) {
    Write-Host "2. Testing Intelligence Layer API endpoints..." -ForegroundColor Yellow
    
    # Test regime inference
    Write-Host "   Testing /intelligence/regime..." -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/intelligence/regime?asset_id=EURUSD" -Method GET -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $data = $response.Content | ConvertFrom-Json
            if ($data.regime_probabilities) {
                Write-Host " OK" -ForegroundColor Green
                Write-Host "      Regime probabilities: $($data.regime_probabilities | ConvertTo-Json -Compress)" -ForegroundColor Gray
            } else {
                Write-Host " FAILED (Invalid response)" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host " FAILED ($_)" -ForegroundColor Red
    }
    
    # Test graph features
    Write-Host "   Testing /intelligence/graph-features..." -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/intelligence/graph-features?asset_id=EURUSD" -Method GET -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $data = $response.Content | ConvertFrom-Json
            if ($data.centrality_metrics) {
                Write-Host " OK" -ForegroundColor Green
                Write-Host "      Cluster: $($data.cluster_membership)" -ForegroundColor Gray
            } else {
                Write-Host " FAILED (Invalid response)" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host " FAILED ($_)" -ForegroundColor Red
    }
    
    # Test RL state assembly
    Write-Host "   Testing /intelligence/state..." -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/intelligence/state?asset_ids=EURUSD,GBPUSD" -Method GET -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $data = $response.Content | ConvertFrom-Json
            if ($data.composite_state) {
                Write-Host " OK" -ForegroundColor Green
                Write-Host "      Current regime: $($data.composite_state.current_regime_label)" -ForegroundColor Gray
            } else {
                Write-Host " FAILED (Invalid response)" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host " FAILED ($_)" -ForegroundColor Red
    }
    
    Write-Host ""
}

# Summary
Write-Host "=== Integration Test Summary ===" -ForegroundColor Cyan
Write-Host "Intelligence Layer: $(if ($intelligenceHealthy) { 'HEALTHY' } else { 'DOWN' })" -ForegroundColor $(if ($intelligenceHealthy) { 'Green' } else { 'Red' })
Write-Host "Execution Core:     $(if ($executionHealthy) { 'HEALTHY' } else { 'DOWN' })" -ForegroundColor $(if ($executionHealthy) { 'Green' } else { 'Red' })
Write-Host ""

if ($intelligenceHealthy -and $executionHealthy) {
    Write-Host "All services are running. You can now start the frontend:" -ForegroundColor Green
    Write-Host "  cd frontend" -ForegroundColor Cyan
    Write-Host "  npm run dev" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Frontend will be available at: http://localhost:5173" -ForegroundColor Green
} else {
    Write-Host "Some services are not running. Please start them first:" -ForegroundColor Yellow
    if (-not $intelligenceHealthy) {
        Write-Host "  Intelligence Layer: docker-compose up intelligence-layer" -ForegroundColor Cyan
    }
    if (-not $executionHealthy) {
        Write-Host "  Execution Core: docker-compose up execution-core" -ForegroundColor Cyan
    }
}

Write-Host ""
