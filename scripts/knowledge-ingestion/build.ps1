# Google Drive Knowledge Base Ingestion - Container Build Script (PowerShell)

param(
    [string]$Tag = "latest",
    [string]$Registry = "",
    [ValidateSet("development", "production", "dev", "prod")]
    [string]$Environment = "production",
    [switch]$NoCache,
    [switch]$Push,
    [switch]$Help
)

# Configuration
$ImageName = "knowledge-ingestion"
$BuildTarget = "production"

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

# Function to show usage
function Show-Usage {
    @"
Usage: .\build.ps1 [OPTIONS]

Build the Google Drive Knowledge Base Ingestion container.

OPTIONS:
    -Tag TAG                Set the image tag (default: latest)
    -Registry URL           Set the Docker registry URL
    -Environment ENV        Set the target environment (development|production)
    -NoCache                Build without using cache
    -Push                   Push the image to registry after building
    -Help                   Show this help message

EXAMPLES:
    .\build.ps1                                         # Build with default settings
    .\build.ps1 -Tag "v1.0.0" -Environment production  # Build production image with tag v1.0.0
    .\build.ps1 -Tag "dev" -Environment development -NoCache  # Build development image without cache
    .\build.ps1 -Tag "v1.0.0" -Push -Registry "myregistry.com"  # Build and push to registry

ENVIRONMENT VARIABLES:
    IMAGE_TAG               Default image tag (default: latest)
    BUILD_TARGET            Build target (development|production, default: production)
    DOCKER_REGISTRY         Docker registry URL
"@
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Process environment parameter
switch ($Environment) {
    { $_ -in @("development", "dev") } {
        $BuildTarget = "production"  # Always use production target, but with dev config
    }
    { $_ -in @("production", "prod") } {
        $BuildTarget = "production"
    }
}

# Construct full image name
if ($Registry) {
    $FullImageName = "$Registry/$ImageName`:$Tag"
} else {
    $FullImageName = "$ImageName`:$Tag"
}

# Pre-build validation
Write-Info "Starting container build process..."
Write-Info "Image name: $FullImageName"
Write-Info "Build target: $BuildTarget"

# Check if Docker is available
try {
    $null = Get-Command docker -ErrorAction Stop
} catch {
    Write-Error "Docker is not installed or not in PATH"
    exit 1
}

# Check if Dockerfile exists
if (-not (Test-Path "Dockerfile")) {
    Write-Error "Dockerfile not found in current directory"
    exit 1
}

# Check if requirements.txt exists
if (-not (Test-Path "requirements.txt")) {
    Write-Error "requirements.txt not found in current directory"
    exit 1
}

# Build the Docker image
Write-Info "Building Docker image..."
$BuildStartTime = Get-Date

# Prepare build arguments
$BuildArgs = @(
    "build"
    "--target", $BuildTarget
    "--tag", $FullImageName
    "--label", "build.timestamp=$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    "--label", "build.version=$Tag"
    "--label", "build.target=$BuildTarget"
)

if ($NoCache) {
    $BuildArgs += "--no-cache"
}

$BuildArgs += "."

try {
    & docker @BuildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE"
    }
    
    $BuildEndTime = Get-Date
    $BuildDuration = [math]::Round(($BuildEndTime - $BuildStartTime).TotalSeconds, 1)
    Write-Success "Docker image built successfully in ${BuildDuration}s"
    Write-Success "Image: $FullImageName"
} catch {
    Write-Error "Docker build failed: $_"
    exit 1
}

# Show image information
Write-Info "Image information:"
& docker images $FullImageName --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Push to registry if requested
if ($Push) {
    if (-not $Registry) {
        Write-Error "Cannot push: no registry specified. Use -Registry option."
        exit 1
    }
    
    Write-Info "Pushing image to registry..."
    try {
        & docker push $FullImageName
        if ($LASTEXITCODE -ne 0) {
            throw "Docker push failed with exit code $LASTEXITCODE"
        }
        Write-Success "Image pushed successfully to $Registry"
    } catch {
        Write-Error "Failed to push image to registry: $_"
        exit 1
    }
}

# Provide usage examples
Write-Info "Build completed successfully!"
Write-Host ""
Write-Info "To run the container:"
Write-Host "  docker run --rm -it $FullImageName"
Write-Host ""
Write-Info "To run with Docker Compose:"
Write-Host "  docker-compose up knowledge-ingestion"
Write-Host ""
Write-Info "To run with custom environment:"
Write-Host "  docker run --rm -it -e KNOWLEDGE_INGESTION_ENV=development $FullImageName"