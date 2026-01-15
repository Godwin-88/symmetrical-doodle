#!/bin/bash

# Integration Test Script for Frontend-Backend Connection
# Tests the connection between frontend and backend services

echo "=== Frontend-Backend Integration Test ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Check if services are running
echo -e "${YELLOW}1. Checking backend services...${NC}"

# Test Intelligence Layer
echo -n "   Testing Intelligence Layer (port 8000)..."
if curl -s -f -m 5 http://localhost:8000/health > /dev/null 2>&1; then
    echo -e " ${GREEN}OK${NC}"
    intelligence_healthy=true
else
    echo -e " ${RED}FAILED (Not responding)${NC}"
    intelligence_healthy=false
fi

# Test Execution Core
echo -n "   Testing Execution Core (port 8001)..."
if curl -s -f -m 5 http://localhost:8001/health > /dev/null 2>&1; then
    echo -e " ${GREEN}OK${NC}"
    execution_healthy=true
else
    echo -e " ${RED}FAILED (Not responding)${NC}"
    execution_healthy=false
fi

echo ""

# Test Intelligence Layer API endpoints
if [ "$intelligence_healthy" = true ]; then
    echo -e "${YELLOW}2. Testing Intelligence Layer API endpoints...${NC}"
    
    # Test regime inference
    echo -n "   Testing /intelligence/regime..."
    response=$(curl -s -m 5 "http://localhost:8000/intelligence/regime?asset_id=EURUSD" 2>/dev/null)
    if echo "$response" | grep -q "regime_probabilities"; then
        echo -e " ${GREEN}OK${NC}"
        echo -e "      ${GRAY}Regime probabilities: $(echo $response | jq -c '.regime_probabilities' 2>/dev/null || echo 'N/A')${NC}"
    else
        echo -e " ${RED}FAILED (Invalid response)${NC}"
    fi
    
    # Test graph features
    echo -n "   Testing /intelligence/graph-features..."
    response=$(curl -s -m 5 "http://localhost:8000/intelligence/graph-features?asset_id=EURUSD" 2>/dev/null)
    if echo "$response" | grep -q "centrality_metrics"; then
        echo -e " ${GREEN}OK${NC}"
        cluster=$(echo "$response" | jq -r '.cluster_membership' 2>/dev/null || echo 'N/A')
        echo -e "      ${GRAY}Cluster: $cluster${NC}"
    else
        echo -e " ${RED}FAILED (Invalid response)${NC}"
    fi
    
    # Test RL state assembly
    echo -n "   Testing /intelligence/state..."
    response=$(curl -s -m 5 "http://localhost:8000/intelligence/state?asset_ids=EURUSD,GBPUSD" 2>/dev/null)
    if echo "$response" | grep -q "composite_state"; then
        echo -e " ${GREEN}OK${NC}"
        regime=$(echo "$response" | jq -r '.composite_state.current_regime_label' 2>/dev/null || echo 'N/A')
        echo -e "      ${GRAY}Current regime: $regime${NC}"
    else
        echo -e " ${RED}FAILED (Invalid response)${NC}"
    fi
    
    echo ""
fi

# Summary
echo -e "${CYAN}=== Integration Test Summary ===${NC}"
if [ "$intelligence_healthy" = true ]; then
    echo -e "Intelligence Layer: ${GREEN}HEALTHY${NC}"
else
    echo -e "Intelligence Layer: ${RED}DOWN${NC}"
fi

if [ "$execution_healthy" = true ]; then
    echo -e "Execution Core:     ${GREEN}HEALTHY${NC}"
else
    echo -e "Execution Core:     ${RED}DOWN${NC}"
fi
echo ""

if [ "$intelligence_healthy" = true ] && [ "$execution_healthy" = true ]; then
    echo -e "${GREEN}All services are running. You can now start the frontend:${NC}"
    echo -e "  ${CYAN}cd frontend${NC}"
    echo -e "  ${CYAN}npm run dev${NC}"
    echo ""
    echo -e "${GREEN}Frontend will be available at: http://localhost:5173${NC}"
else
    echo -e "${YELLOW}Some services are not running. Please start them first:${NC}"
    if [ "$intelligence_healthy" = false ]; then
        echo -e "  ${CYAN}Intelligence Layer: docker-compose up intelligence-layer${NC}"
    fi
    if [ "$execution_healthy" = false ]; then
        echo -e "  ${CYAN}Execution Core: docker-compose up execution-core${NC}"
    fi
fi

echo ""
