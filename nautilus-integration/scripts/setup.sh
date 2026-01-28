#!/bin/bash
# Idempotent setup script for NautilusTrader integration (Unix/Linux/macOS)

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INTEGRATION_ROOT="$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check Python version
    if ! command_exists python3; then
        error_exit "Python 3 is required but not installed"
    fi
    
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local required_version="3.11"
    
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
        error_exit "Python 3.11 or higher is required (found: $python_version)"
    fi
    
    # Check for required tools
    local required_tools=("git" "curl")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command_exists "$tool"; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error_exit "Missing required tools: ${missing_tools[*]}"
    fi
    
    # Check for Rust (will install if missing)
    if ! command_exists rustc; then
        log_warning "Rust not found - will attempt to install during setup"
    fi
    
    log_success "Prerequisites validation completed"
}

# Install Rust if not present
install_rust() {
    if command_exists rustc; then
        log_info "Rust already installed, skipping"
        return 0
    fi
    
    log_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Source Rust environment
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env" 2>/dev/null || true
    
    if command_exists rustc; then
        log_success "Rust installed successfully"
    else
        log_warning "Rust installation may require shell restart"
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    local directories=(
        "$INTEGRATION_ROOT/src/nautilus_integration"
        "$INTEGRATION_ROOT/tests"
        "$INTEGRATION_ROOT/scripts"
        "$INTEGRATION_ROOT/config"
        "$INTEGRATION_ROOT/data/catalog/bars"
        "$INTEGRATION_ROOT/data/catalog/ticks"
        "$INTEGRATION_ROOT/data/catalog/order_book"
        "$INTEGRATION_ROOT/data/catalog/instruments"
        "$INTEGRATION_ROOT/data/logs"
        "$INTEGRATION_ROOT/monitoring/grafana/dashboards"
        "$INTEGRATION_ROOT/monitoring/grafana/datasources"
        "$INTEGRATION_ROOT/database/init"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        # Create .gitkeep for empty directories
        if [ ! -f "$dir/.gitkeep" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
            touch "$dir/.gitkeep"
        fi
    done
    
    log_success "Directory structure created"
}

# Setup virtual environment
setup_venv() {
    local venv_path="$INTEGRATION_ROOT/.venv"
    
    if [ -d "$venv_path" ]; then
        log_info "Virtual environment already exists, skipping creation"
        return 0
    fi
    
    log_info "Creating virtual environment..."
    python3 -m venv "$venv_path"
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$venv_path/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created"
}

# Install dependencies
install_dependencies() {
    local development=${1:-false}
    
    log_info "Installing dependencies (development: $development)..."
    
    local venv_path="$INTEGRATION_ROOT/.venv"
    if [ ! -d "$venv_path" ]; then
        error_exit "Virtual environment not found. Run setup with --create-venv first."
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$venv_path/bin/activate"
    
    # Install main dependencies
    if [ -f "$INTEGRATION_ROOT/requirements.txt" ]; then
        pip install -r "$INTEGRATION_ROOT/requirements.txt"
    else
        log_warning "requirements.txt not found, skipping dependency installation"
    fi
    
    # Install development dependencies
    if [ "$development" = true ] && [ -f "$INTEGRATION_ROOT/pyproject.toml" ]; then
        pip install -e ".[dev]"
    fi
    
    # Install package in development mode
    if [ -f "$INTEGRATION_ROOT/pyproject.toml" ]; then
        pip install -e .
    fi
    
    log_success "Dependencies installed"
}

# Setup configuration
setup_configuration() {
    log_info "Setting up configuration..."
    
    local config_dir="$INTEGRATION_ROOT/config"
    local env_example="$config_dir/.env.example"
    local env_file="$config_dir/.env"
    
    # Copy example configuration if .env doesn't exist
    if [ -f "$env_example" ] && [ ! -f "$env_file" ]; then
        cp "$env_example" "$env_file"
        log_success "Created configuration file from example"
    elif [ -f "$env_file" ]; then
        log_info "Configuration file already exists, skipping"
    else
        log_warning "No configuration example found"
    fi
    
    # Create monitoring configuration
    create_monitoring_config
    
    # Create database initialization scripts
    create_database_init
    
    log_success "Configuration setup completed"
}

# Create monitoring configuration
create_monitoring_config() {
    local monitoring_dir="$INTEGRATION_ROOT/monitoring"
    local prometheus_config="$monitoring_dir/prometheus.yml"
    
    if [ ! -f "$prometheus_config" ]; then
        cat > "$prometheus_config" << 'EOF'
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
EOF
        log_info "Created Prometheus configuration"
    fi
}

# Create database initialization scripts
create_database_init() {
    local init_dir="$INTEGRATION_ROOT/database/init"
    local postgres_init="$init_dir/01-nautilus-init.sql"
    
    if [ ! -f "$postgres_init" ]; then
        cat > "$postgres_init" << 'EOF'
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
EOF
        log_info "Created PostgreSQL initialization script"
    fi
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."
    
    local venv_path="$INTEGRATION_ROOT/.venv"
    if [ ! -d "$venv_path" ]; then
        error_exit "Virtual environment not found"
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$venv_path/bin/activate"
    
    # Test NautilusTrader import
    if ! python -c "import nautilus_trader; print(f'NautilusTrader {nautilus_trader.__version__}')" 2>/dev/null; then
        error_exit "NautilusTrader validation failed"
    fi
    log_success "NautilusTrader validation passed"
    
    # Test integration package import
    if ! python -c "import nautilus_integration; print('Integration package loaded')" 2>/dev/null; then
        log_warning "Integration package validation failed (may be expected during initial setup)"
    else
        log_success "Integration package validation passed"
    fi
    
    log_success "Installation validation completed"
}

# Make scripts executable
make_scripts_executable() {
    log_info "Making scripts executable..."
    
    local scripts=(
        "$INTEGRATION_ROOT/scripts/setup.sh"
        "$INTEGRATION_ROOT/scripts/build.sh"
        "$INTEGRATION_ROOT/scripts/deploy.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            chmod +x "$script"
        fi
    done
    
    log_success "Scripts made executable"
}

# Main setup function
main() {
    local skip_venv=false
    local skip_deps=false
    local skip_config=false
    local skip_rust=false
    local development=false
    local verbose=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-venv)
                skip_venv=true
                shift
                ;;
            --skip-deps)
                skip_deps=true
                shift
                ;;
            --skip-config)
                skip_config=true
                shift
                ;;
            --skip-rust)
                skip_rust=true
                shift
                ;;
            --development)
                development=true
                shift
                ;;
            --verbose)
                verbose=true
                set -x
                shift
                ;;
            --help)
                cat << EOF
Usage: $0 [OPTIONS]

Idempotent setup script for NautilusTrader integration.

OPTIONS:
    --skip-venv      Skip virtual environment creation
    --skip-deps      Skip dependency installation
    --skip-config    Skip configuration setup
    --skip-rust      Skip Rust installation
    --development    Setup for development environment
    --verbose        Enable verbose output
    --help           Show this help message

EOF
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
    
    log_info "Starting NautilusTrader integration setup..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Integration root: $INTEGRATION_ROOT"
    
    # Run setup steps
    validate_prerequisites
    
    if [ "$skip_rust" = false ]; then
        install_rust
    fi
    
    create_directories
    
    if [ "$skip_venv" = false ]; then
        setup_venv
    fi
    
    if [ "$skip_deps" = false ]; then
        install_dependencies "$development"
    fi
    
    if [ "$skip_config" = false ]; then
        setup_configuration
    fi
    
    make_scripts_executable
    validate_installation
    
    log_success "NautilusTrader integration setup completed successfully!"
    
    echo
    echo "Next steps:"
    echo "1. Review and customize config/.env"
    echo "2. Run: docker-compose up -d --profile with-database"
    echo "3. Test the integration: source .venv/bin/activate && python -m nautilus_integration.main --help"
    echo
}

# Run main function with all arguments
main "$@"