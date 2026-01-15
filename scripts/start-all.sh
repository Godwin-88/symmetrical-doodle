#!/bin/bash
#
# Start all trading system servers
# Launches all necessary services for the algorithmic trading system
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running from project root
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_header "ALGORITHMIC TRADING SYSTEM - STARTUP"

# Create logs directory
mkdir -p logs

# Step 1: Start Docker services
print_header "STEP 1: Starting Database Services (Docker)"
print_info "Starting PostgreSQL, Neo4j, and Redis..."

if command -v docker-compose &> /dev/null; then
    docker-compose up -d
    sleep 5
    print_success "Database services started"
    print_info "PostgreSQL: localhost:5432"
    print_info "Neo4j: localhost:7474 (browser), localhost:7687 (bolt)"
    print_info "Redis: localhost:6379"
else
    print_warning "Docker Compose not found. Please start database services manually."
fi

# Step 2: Start Python Intelligence Layer
print_header "STEP 2: Starting Python Intelligence Layer (Port 8000)"

cd intelligence-layer

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

# Start the server in background
nohup python -m uvicorn intelligence_layer.main:app --host 0.0.0.0 --port 8000 --reload > ../logs/intelligence-layer.log 2>&1 &
PYTHON_PID=$!
echo $PYTHON_PID > ../logs/intelligence-layer.pid

cd ..

print_success "Python Intelligence Layer started (PID: $PYTHON_PID)"
print_info "API will be available at: http://localhost:8000"
print_info "API docs at: http://localhost:8000/docs"
print_info "Logs: logs/intelligence-layer.log"

# Step 3: Start Rust Execution Core
print_header "STEP 3: Starting Rust Execution Core (Port 8001)"

cd execution-core

# Build and run in background
nohup cargo run --release > ../logs/execution-core.log 2>&1 &
EXECUTION_PID=$!
echo $EXECUTION_PID > ../logs/execution-core.pid

cd ..

print_success "Rust Execution Core started (PID: $EXECUTION_PID)"
print_info "Execution API will be available at: http://localhost:8001"
print_info "Logs: logs/execution-core.log"

# Step 4: Start Rust Simulation Engine
print_header "STEP 4: Starting Rust Simulation Engine (Port 8002)"

cd simulation-engine

# Build and run in background
nohup cargo run --release > ../logs/simulation-engine.log 2>&1 &
SIMULATION_PID=$!
echo $SIMULATION_PID > ../logs/simulation-engine.pid

cd ..

print_success "Rust Simulation Engine started (PID: $SIMULATION_PID)"
print_info "Simulation API will be available at: http://localhost:8002"
print_info "Logs: logs/simulation-engine.log"

# Step 5: Start React Frontend
print_header "STEP 5: Starting React Frontend (Port 5173)"

cd frontend

# Start dev server in background
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid

cd ..

print_success "React Frontend started (PID: $FRONTEND_PID)"
print_info "Frontend will be available at: http://localhost:5173"
print_info "Logs: logs/frontend.log"

# Wait for services to start
print_header "Waiting for services to initialize..."
sleep 10

# Check service health
print_header "SERVICE STATUS"

check_service() {
    local url=$1
    local name=$2
    
    if curl -s -f -o /dev/null "$url"; then
        print_success "$name is running"
        return 0
    else
        print_warning "$name is starting... (may take a moment)"
        return 1
    fi
}

check_service "http://localhost:8000/health" "Python Intelligence Layer"
check_service "http://localhost:8001/health" "Rust Execution Core"
check_service "http://localhost:8002/health" "Rust Simulation Engine"
check_service "http://localhost:5173" "React Frontend"

# Display process information
print_header "RUNNING PROCESSES"
print_info "Python Intelligence Layer: PID $PYTHON_PID"
print_info "Rust Execution Core: PID $EXECUTION_PID"
print_info "Rust Simulation Engine: PID $SIMULATION_PID"
print_info "React Frontend: PID $FRONTEND_PID"

# Display access URLs
print_header "ACCESS URLS"
echo -e "${CYAN}Frontend:              http://localhost:5173${NC}"
echo -e "${CYAN}Intelligence API:      http://localhost:8000${NC}"
echo -e "${CYAN}Intelligence Docs:     http://localhost:8000/docs${NC}"
echo -e "${CYAN}Execution API:         http://localhost:8001${NC}"
echo -e "${CYAN}Simulation API:        http://localhost:8002${NC}"
echo -e "${CYAN}Neo4j Browser:         http://localhost:7474${NC}"
echo ""

# Display monitoring commands
print_header "MONITORING COMMANDS"
print_info "View Python logs:      tail -f logs/intelligence-layer.log"
print_info "View Execution logs:   tail -f logs/execution-core.log"
print_info "View Simulation logs:  tail -f logs/simulation-engine.log"
print_info "View Frontend logs:    tail -f logs/frontend.log"
print_info "View all logs:         tail -f logs/*.log"
print_info "Stop all services:     ./scripts/stop-all.sh"
echo ""

# Display Deriv status
print_header "DERIV API STATUS"
if [ -n "$DERIV_API_TOKEN" ]; then
    print_success "Deriv API token configured"
    print_info "Demo trading enabled"
    print_info "Test connection: python scripts/test-deriv-connection.py"
else
    print_warning "Deriv API token not configured"
    print_info "Set DERIV_API_TOKEN in .env file to enable demo trading"
fi

print_header "SYSTEM READY"
echo -e "${GREEN}All services are running!${NC}"
echo -e "${GREEN}Open http://localhost:5173 in your browser to access the trading system${NC}"
echo ""
print_info "To stop all services, run: ./scripts/stop-all.sh"
print_info "To view logs, run: tail -f logs/*.log"
echo ""
