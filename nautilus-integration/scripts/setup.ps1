# Idempotent setup script for NautilusTrader integration (Windows PowerShell)

param(
    [switch]$SkipVenv,
    [switch]$SkipDeps,
    [switch]$SkipConfig,
    [switch]$SkipRust,
    [switch]$Development,
    [switch]$Verbose,
    [switch]$Help
)

# Script configuration
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$IntegrationRoot = $ProjectRoot

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

# Error handling
function Exit-WithError {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

# Check if command exists
function Test-Command {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Show help
function Show-Help {
    @"
Usage: .\setup.ps1 [OPTIONS]

Idempotent setup script for NautilusTrader integration.

OPTIONS:
    -SkipVenv       Skip virtual environment creation
    -SkipDeps       Skip dependency installation
    -SkipConfig     Skip configuration setup
    -SkipRust       Skip Rust installation
    -Development    Setup for development environment
    -Verbose        Enable verbose output
    -Help           Show this help message

"@
}

# Validate prerequisites
function Test-Prerequisites {
    Write-Info "Validating prerequisites..."
    
    # Check Python version
    if (-not (Test-Command "python")) {
        Exit-WithError "Python is required but not installed"
    }
    
    try {
        $pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        $versionCheck = python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
        if ($LASTEXITCODE -ne 0) {
            Exit-WithError "Python 3.11 or higher is required (found: $pythonVersion)"
        }
    }
    catch {
        Exit-WithError "Failed to check Python version"
    }
    
    # Check for required tools
    $requiredTools = @("git", "curl")
    $missingTools = @()
    
    foreach ($tool in $requiredTools) {
        if (-not (Test-Command $tool)) {
            $missingTools += $tool
        }
    }
    
    if ($missingTools.Count -gt 0) {
        Exit-WithError "Missing required tools: $($missingTools -join ', ')"
    }
    
    # Check for Rust (will install if missing)
    if (-not (Test-Command "rustc")) {
        Write-Warning "Rust not found - will attempt to install during setup"
    }
    
    Write-Success "Prerequisites validation completed"
}

# Install Rust if not present
function Install-Rust {
    if (Test-Command "rustc") {
        Write-Info "Rust already installed, skipping"
        return
    }
    
    Write-Info "Installing Rust..."
    try {
        # Download and run rustup installer
        $rustupUrl = "https://win.rustup.rs/x86_64"
        $rustupPath = "$env:TEMP\rustup-init.exe"
        
        Invoke-WebRequest -Uri $rustupUrl -OutFile $rustupPath
        & $rustupPath -y --default-toolchain stable
        
        # Add Rust to PATH for current session
        $env:PATH += ";$env:USERPROFILE\.cargo\bin"
        
        if (Test-Command "rustc") {
            Write-Success "Rust installed successfully"
        }
        else {
            Write-Warning "Rust installation may require shell restart"
        }
    }
    catch {
        Write-Warning "Failed to install Rust automatically: $_"
    }
    finally {
        if (Test-Path $rustupPath) {
            Remove-Item $rustupPath -Force
        }
    }
}

# Create directory structure
function New-Directories {
    Write-Info "Creating directory structure..."
    
    $directories = @(
        "$IntegrationRoot\src\nautilus_integration"
        "$IntegrationRoot\tests"
        "$IntegrationRoot\scripts"
        "$IntegrationRoot\config"
        "$IntegrationRoot\data\catalog\bars"
        "$IntegrationRoot\data\catalog\ticks"
        "$IntegrationRoot\data\catalog\order_book"
        "$IntegrationRoot\data\catalog\instruments"
        "$IntegrationRoot\data\logs"
        "$IntegrationRoot\monitoring\grafana\dashboards"
        "$IntegrationRoot\monitoring\grafana\datasources"
        "$IntegrationRoot\database\init"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        
        # Create .gitkeep for empty directories
        $gitkeepPath = Join-Path $dir ".gitkeep"
        if (-not (Test-Path $gitkeepPath) -and ((Get-ChildItem $dir -Force).Count -eq 0)) {
            New-Item -ItemType File -Path $gitkeepPath -Force | Out-Null
        }
    }
    
    Write-Success "Directory structure created"
}

# Setup virtual environment
function New-VirtualEnvironment {
    $venvPath = Join-Path $IntegrationRoot ".venv"
    
    if (Test-Path $venvPath) {
        Write-Info "Virtual environment already exists, skipping creation"
        return
    }
    
    Write-Info "Creating virtual environment..."
    python -m venv $venvPath
    
    # Activate virtual environment and upgrade pip
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        python -m pip install --upgrade pip setuptools wheel
    }
    
    Write-Success "Virtual environment created"
}

# Install dependencies
function Install-Dependencies {
    param([bool]$Development = $false)
    
    Write-Info "Installing dependencies (development: $Development)..."
    
    $venvPath = Join-Path $IntegrationRoot ".venv"
    if (-not (Test-Path $venvPath)) {
        Exit-WithError "Virtual environment not found. Run setup with -SkipVenv:$false first."
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    }
    
    # Install main dependencies
    $requirementsPath = Join-Path $IntegrationRoot "requirements.txt"
    if (Test-Path $requirementsPath) {
        python -m pip install -r $requirementsPath
    }
    else {
        Write-Warning "requirements.txt not found, skipping dependency installation"
    }
    
    # Install development dependencies
    $pyprojectPath = Join-Path $IntegrationRoot "pyproject.toml"
    if ($Development -and (Test-Path $pyprojectPath)) {
        python -m pip install -e ".[dev]"
    }
    
    # Install package in development mode
    if (Test-Path $pyprojectPath) {
        python -m pip install -e .
    }
    
    Write-Success "Dependencies installed"
}

# Setup configuration
function New-Configuration {
    Write-Info "Setting up configuration..."
    
    $configDir = Join-Path $IntegrationRoot "config"
    $envExample = Join-Path $configDir ".env.example"
    $envFile = Join-Path $configDir ".env"
    
    # Copy example configuration if .env doesn't exist
    if ((Test-Path $envExample) -and (-not (Test-Path $envFile))) {
        Copy-Item $envExample $envFile
        Write-Success "Created configuration file from example"
    }
    elseif (Test-Path $envFile) {
        Write-Info "Configuration file already exists, skipping"
    }
    else {
        Write-Warning "No configuration example found"
    }
    
    # Create monitoring configuration
    New-MonitoringConfig
    
    # Create database initialization scripts
    New-DatabaseInit
    
    Write-Success "Configuration setup completed"
}

# Create monitoring configuration
function New-MonitoringConfig {
    $monitoringDir = Join-Path $IntegrationRoot "monitoring"
    $prometheusConfig = Join-Path $monitoringDir "prometheus.yml"
    
    if (-not (Test-Path $prometheusConfig)) {
        $prometheusContent = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nautilus-integration'
    static_configs:
      - targets: ['nautilus-integration:8002']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"@
        Set-Content -Path $prometheusConfig -Value $prometheusContent
        Write-Info "Created Prometheus configuration"
    }
}

# Create database initialization scripts
function New-DatabaseInit {
    $initDir = Join-Path $IntegrationRoot "database\init"
    $postgresInit = Join-Path $initDir "01-nautilus-init.sql"
    
    if (-not (Test-Path $postgresInit)) {
        $postgresContent = @"
-- NautilusTrader Integration Database Initialization

-- Create extension for pgvector if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for Nautilus data
CREATE SCHEMA IF NOT EXISTS nautilus;

-- Create tables for Nautilus integration
CREATE TABLE IF NOT EXISTS nautilus.backtests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id VARCHAR(255) UNIQUE NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    results JSONB
);

CREATE TABLE IF NOT EXISTS nautilus.trading_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_backtests_status ON nautilus.backtests(status);
CREATE INDEX IF NOT EXISTS idx_backtests_created_at ON nautilus.backtests(created_at);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_status ON nautilus.trading_sessions(status);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_created_at ON nautilus.trading_sessions(created_at);

-- Grant permissions
GRANT USAGE ON SCHEMA nautilus TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nautilus TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA nautilus TO postgres;
"@
        Set-Content -Path $postgresInit -Value $postgresContent
        Write-Info "Created PostgreSQL initialization script"
    }
}

# Validate installation
function Test-Installation {
    Write-Info "Validating installation..."
    
    $venvPath = Join-Path $IntegrationRoot ".venv"
    if (-not (Test-Path $venvPath)) {
        Exit-WithError "Virtual environment not found"
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    }
    
    # Test NautilusTrader import
    try {
        $nautilusVersion = python -c "import nautilus_trader; print(f'NautilusTrader {nautilus_trader.__version__}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "NautilusTrader validation passed: $nautilusVersion"
        }
        else {
            Exit-WithError "NautilusTrader validation failed"
        }
    }
    catch {
        Exit-WithError "NautilusTrader validation failed: $_"
    }
    
    # Test integration package import
    try {
        python -c "import nautilus_integration; print('Integration package loaded')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Integration package validation passed"
        }
        else {
            Write-Warning "Integration package validation failed (may be expected during initial setup)"
        }
    }
    catch {
        Write-Warning "Integration package validation failed: $_"
    }
    
    Write-Success "Installation validation completed"
}

# Main setup function
function Invoke-Setup {
    if ($Help) {
        Show-Help
        return
    }
    
    if ($Verbose) {
        $VerbosePreference = "Continue"
    }
    
    Write-Info "Starting NautilusTrader integration setup..."
    Write-Info "Project root: $ProjectRoot"
    Write-Info "Integration root: $IntegrationRoot"
    
    try {
        # Run setup steps
        Test-Prerequisites
        
        if (-not $SkipRust) {
            Install-Rust
        }
        
        New-Directories
        
        if (-not $SkipVenv) {
            New-VirtualEnvironment
        }
        
        if (-not $SkipDeps) {
            Install-Dependencies -Development $Development
        }
        
        if (-not $SkipConfig) {
            New-Configuration
        }
        
        Test-Installation
        
        Write-Success "NautilusTrader integration setup completed successfully!"
        
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "1. Review and customize config\.env"
        Write-Host "2. Run: docker-compose up -d --profile with-database"
        Write-Host "3. Test the integration: .\.venv\Scripts\Activate.ps1 && python -m nautilus_integration.main --help"
        Write-Host ""
    }
    catch {
        Exit-WithError "Setup failed: $_"
    }
}

# Run main function
Invoke-Setup