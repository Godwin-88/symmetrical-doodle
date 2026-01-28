#!/bin/bash
#
# Enhanced Startup Script with Health Monitoring and Auto-Recovery
# Provides the most ideal feasible experience for system startup
#

set -e

# Colors and styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
MAX_RETRIES=3
HEALTH_CHECK_TIMEOUT=30
AUTO_RECOVERY=${AUTO_RECOVERY:-true}
VERBOSE=${VERBOSE:-false}

# Helper functions
print_banner() {
    echo ""
    echo -e "${PURPLE}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}${BOLD}║                ALGORITHMIC TRADING SYSTEM                    ║${NC}"
    echo -e "${PURPLE}${BOLD}║              Enhanced Startup Experience                     ║${NC}"
    echo -e "${PURPLE}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}▶ $1${NC}"
    echo -e "${BLUE}${BOLD}$(printf '%.0s─' {1..60})${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_progress() {
    local current=$1
    local total=$2
    local desc=$3
    local percent=$((current * 100 / total))
    local filled=$((percent / 5))
    local empty=$((20 - filled))
    
    printf "\r${CYAN}[%s%s] %d%% - %s${NC}" \
        "$(printf '%.0s█' $(seq 1 $filled))" \
        "$(printf '%.0s░' $(seq 1 $empty))" \
        "$percent" "$desc"
    
    if [ $current -eq $total ]; then
        echo ""
    fi
}

# System checks
check_prerequisites() {
    print_header "SYSTEM PREREQUISITES CHECK"
    
    local checks=("docker" "docker-compose" "node" "npm" "python" "cargo")
    local passed=0
    local total=${#checks[@]}
    
    for i in "${!checks[@]}"; do
        local cmd=${checks[$i]}
        print_progress $((i+1)) $total "Checking $cmd..."
        
        if command -v $cmd &> /dev/null; then
            print_success "$cmd is available"
            ((passed++))
        else
            print_error "$cmd is not installed"
        fi
        sleep 0.5
    done
    
    if [ $passed -eq $total ]; then
        print_success "All prerequisites satisfied ($passed/$total)"
        return 0
    else
        print_error "Missing prerequisites ($passed/$total)"
        return 1
    fi
}

# Enhanced health check with retry logic
health_check() {
    local service=$1
    local url=$2
    local max_attempts=${3:-10}
    local delay=${4:-3}
    
    for attempt in $(seq 1 $max_attempts); do
        if curl -s -f -o /dev/null --max-time 5 "$url" 2>/dev/null; then
            print_success "$service is healthy (attempt $attempt/$max_attempts)"
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            print_info "$service not ready, retrying in ${delay}s... (attempt $attempt/$max_attempts)"
            sleep $delay
        fi
    done
    
    print_error "$service failed health check after $max_attempts attempts"
    return 1
}

# Intelligent service startup with dependency management
start_service() {
    local service=$1
    local command=$2
    local health_url=$3
    local log_file=$4
    local pid_file=$5
    
    print_info "Starting $service..."
    
    # Kill existing process if running
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            print_warning "Stopping existing $service process (PID: $old_pid)"
            kill "$old_pid" 2>/dev/null || true
            sleep 2
        fi
        rm -f "$pid_file"
    fi
    
    # Start service
    nohup bash -c "$command" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    # Wait a moment for startup
    sleep 3
    
    # Check if process is still running
    if ! kill -0 "$pid" 2>/dev/null; then
        print_error "$service failed to start (process died)"
        return 1
    fi
    
    # Health check if URL provided
    if [ -n "$health_url" ]; then
        if health_check "$service" "$health_url"; then
            print_success "$service started successfully (PID: $pid)"
            return 0
        else
            print_error "$service started but failed health check"
            return 1
        fi
    else
        print_success "$service started successfully (PID: $pid)"
        return 0
    fi
}

# Auto-recovery mechanism
auto_recover() {
    local service=$1
    local command=$2
    local health_url=$3
    local log_file=$4
    local pid_file=$5
    
    print_warning "Attempting auto-recovery for $service..."
    
    # Try to restart the service
    if start_service "$service" "$command" "$health_url" "$log_file" "$pid_file"; then
        print_success "$service recovered successfully"
        return 0
    else
        print_error "$service auto-recovery failed"
        return 1
    fi
}

# Main startup sequence
main() {
    print_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --with-health-checks)
                HEALTH_CHECKS=true
                shift
                ;;
            --auto-recovery)
                AUTO_RECOVERY=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --with-health-checks    Enable comprehensive health monitoring"
                echo "  --auto-recovery        Enable automatic service recovery"
                echo "  --verbose              Enable verbose logging"
                echo "  --help                 Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if running from project root
    if [ ! -f "docker-compose.yml" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # System prerequisites check
    if ! check_prerequisites; then
        print_error "Prerequisites check failed. Please install missing dependencies."
        exit 1
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Step 1: Start Infrastructure Services
    print_header "STEP 1: INFRASTRUCTURE SERVICES"
    
    print_info "Starting database services..."
    if docker-compose up -d postgres neo4j redis; then
        print_success "Database services started"
        
        # Wait for databases to be ready
        print_info "Waiting for databases to initialize..."
        sleep 10
        
        # Health check databases
        if health_check "PostgreSQL" "http://localhost:5432" 5 2; then
            print_success "PostgreSQL is ready"
        else
            print_warning "PostgreSQL health check failed, but continuing..."
        fi
        
    else
        print_error "Failed to start database services"
        exit 1
    fi
    
    # Step 2: Start Backend Services
    print_header "STEP 2: BACKEND SERVICES"
    
    # Python Intelligence Layer
    if start_service "Intelligence Layer" \
        "cd intelligence-layer && python -m uvicorn intelligence_layer.main:app --host 0.0.0.0 --port 8000 --reload" \
        "http://localhost:8000/health" \
        "../logs/intelligence-layer.log" \
        "../logs/intelligence-layer.pid"; then
        print_success "Intelligence Layer is running"
    else
        if [ "$AUTO_RECOVERY" = true ]; then
            auto_recover "Intelligence Layer" \
                "cd intelligence-layer && python -m uvicorn intelligence_layer.main:app --host 0.0.0.0 --port 8000 --reload" \
                "http://localhost:8000/health" \
                "../logs/intelligence-layer.log" \
                "../logs/intelligence-layer.pid"
        fi
    fi
    
    # Rust Execution Core
    if start_service "Execution Core" \
        "cd execution-core && cargo run --release" \
        "http://localhost:8001/health" \
        "../logs/execution-core.log" \
        "../logs/execution-core.pid"; then
        print_success "Execution Core is running"
    else
        if [ "$AUTO_RECOVERY" = true ]; then
            auto_recover "Execution Core" \
                "cd execution-core && cargo run --release" \
                "http://localhost:8001/health" \
                "../logs/execution-core.log" \
                "../logs/execution-core.pid"
        fi
    fi
    
    # Rust Simulation Engine
    if start_service "Simulation Engine" \
        "cd simulation-engine && cargo run --release" \
        "http://localhost:8002/health" \
        "../logs/simulation-engine.log" \
        "../logs/simulation-engine.pid"; then
        print_success "Simulation Engine is running"
    else
        if [ "$AUTO_RECOVERY" = true ]; then
            auto_recover "Simulation Engine" \
                "cd simulation-engine && cargo run --release" \
                "http://localhost:8002/health" \
                "../logs/simulation-engine.log" \
                "../logs/simulation-engine.pid"
        fi
    fi
    
    # Step 3: Start Frontend
    print_header "STEP 3: FRONTEND APPLICATION"
    
    if start_service "Frontend" \
        "cd frontend && npm run dev" \
        "http://localhost:5173" \
        "../logs/frontend.log" \
        "../logs/frontend.pid"; then
        print_success "Frontend is running"
    else
        if [ "$AUTO_RECOVERY" = true ]; then
            auto_recover "Frontend" \
                "cd frontend && npm run dev" \
                "http://localhost:5173" \
                "../logs/frontend.log" \
                "../logs/frontend.pid"
        fi
    fi
    
    # Step 4: System Health Summary
    print_header "STEP 4: SYSTEM HEALTH SUMMARY"
    
    local services=(
        "Frontend:http://localhost:5173"
        "Intelligence API:http://localhost:8000/health"
        "Execution Core:http://localhost:8001/health"
        "Simulation Engine:http://localhost:8002/health"
    )
    
    local healthy=0
    local total=${#services[@]}
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service_info"
        if curl -s -f -o /dev/null --max-time 3 "$url" 2>/dev/null; then
            print_success "$name is healthy"
            ((healthy++))
        else
            print_error "$name is not responding"
        fi
    done
    
    # Final status
    print_header "SYSTEM READY"
    
    if [ $healthy -eq $total ]; then
        echo -e "${GREEN}${BOLD}🎉 ALL SYSTEMS OPERATIONAL ($healthy/$total)${NC}"
        echo ""
        echo -e "${CYAN}${BOLD}Access URLs:${NC}"
        echo -e "${CYAN}  Frontend:              http://localhost:5173${NC}"
        echo -e "${CYAN}  Intelligence API:      http://localhost:8000${NC}"
        echo -e "${CYAN}  Intelligence Docs:     http://localhost:8000/docs${NC}"
        echo -e "${CYAN}  Execution API:         http://localhost:8001${NC}"
        echo -e "${CYAN}  Simulation API:        http://localhost:8002${NC}"
        echo ""
        echo -e "${YELLOW}${BOLD}Monitoring Commands:${NC}"
        echo -e "${YELLOW}  View all logs:         tail -f logs/*.log${NC}"
        echo -e "${YELLOW}  System health:         curl http://localhost:8000/health${NC}"
        echo -e "${YELLOW}  Stop all services:     ./scripts/stop-all.sh${NC}"
        echo ""
        echo -e "${GREEN}${BOLD}🚀 Ready for algorithmic trading!${NC}"
    else
        echo -e "${YELLOW}${BOLD}⚠ PARTIAL SYSTEM STARTUP ($healthy/$total services healthy)${NC}"
        echo -e "${YELLOW}Check logs in the 'logs/' directory for troubleshooting.${NC}"
    fi
    
    # Continuous monitoring option
    if [ "$VERBOSE" = true ]; then
        print_header "CONTINUOUS MONITORING (Press Ctrl+C to exit)"
        while true; do
            sleep 30
            echo -e "${CYAN}$(date): Performing health checks...${NC}"
            for service_info in "${services[@]}"; do
                IFS=':' read -r name url <<< "$service_info"
                if curl -s -f -o /dev/null --max-time 3 "$url" 2>/dev/null; then
                    echo -e "${GREEN}  ✓ $name${NC}"
                else
                    echo -e "${RED}  ✗ $name${NC}"
                fi
            done
        done
    fi
}

# Run main function
main "$@"