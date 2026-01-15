#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Stop all trading system servers
.DESCRIPTION
    Stops all running services for the algorithmic trading system
.EXAMPLE
    .\scripts\stop-all.ps1
#>

$ErrorActionPreference = "Continue"

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success($message) {
    Write-ColorOutput Green "✓ $message"
}

function Write-Info($message) {
    Write-ColorOutput Cyan "→ $message"
}

Write-Host ""
Write-ColorOutput Yellow "============================================================"
Write-ColorOutput Yellow "STOPPING ALL SERVICES"
Write-ColorOutput Yellow "============================================================"
Write-Host ""

# Stop all PowerShell jobs
Write-Info "Stopping PowerShell jobs..."
$jobs = Get-Job | Where-Object { $_.State -eq "Running" }
if ($jobs) {
    $jobs | Stop-Job
    $jobs | Remove-Job
    Write-Success "Stopped $($jobs.Count) PowerShell job(s)"
} else {
    Write-Info "No PowerShell jobs running"
}

# Stop Docker services
Write-Info "Stopping Docker services..."
try {
    docker-compose down
    Write-Success "Docker services stopped"
} catch {
    Write-Info "No Docker services to stop"
}

# Kill any remaining processes on our ports
Write-Info "Checking for processes on ports 5173, 8000, 8001, 8002..."

$ports = @(5173, 8000, 8001, 8002)
foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($conn in $connections) {
            $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                Write-Info "Stopping process $($process.Name) (PID: $($process.Id)) on port $port"
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host ""
Write-Success "All services stopped"
Write-Host ""
