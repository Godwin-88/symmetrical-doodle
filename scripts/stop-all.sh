#!/bin/bash
#
# Stop all trading system servers
#

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

echo ""
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}STOPPING ALL SERVICES${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""

# Stop processes from PID files
print_info "Stopping services..."

if [ -f "logs/intelligence-layer.pid" ]; then
    PID=$(cat logs/intelligence-layer.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        print_success "Stopped Python Intelligence Layer (PID: $PID)"
    fi
    rm logs/intelligence-layer.pid
fi

if [ -f "logs/execution-core.pid" ]; then
    PID=$(cat logs/execution-core.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        print_success "Stopped Rust Execution Core (PID: $PID)"
    fi
    rm logs/execution-core.pid
fi

if [ -f "logs/simulation-engine.pid" ]; then
    PID=$(cat logs/simulation-engine.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        print_success "Stopped Rust Simulation Engine (PID: $PID)"
    fi
    rm logs/simulation-engine.pid
fi

if [ -f "logs/frontend.pid" ]; then
    PID=$(cat logs/frontend.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        print_success "Stopped React Frontend (PID: $PID)"
    fi
    rm logs/frontend.pid
fi

# Stop Docker services
print_info "Stopping Docker services..."
if command -v docker-compose &> /dev/null; then
    docker-compose down
    print_success "Docker services stopped"
fi

# Kill any remaining processes on our ports
print_info "Checking for processes on ports 5173, 8000, 8001, 8002..."

for port in 5173 8000 8001 8002; do
    PID=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null || true
        print_info "Killed process on port $port (PID: $PID)"
    fi
done

echo ""
print_success "All services stopped"
echo ""
