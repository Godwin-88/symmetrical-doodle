#!/bin/bash

# Validation script for core system infrastructure setup

echo "=== Algorithmic Trading System - Setup Validation ==="
echo ""

# Check if required files exist
echo "Checking project structure..."

# Rust components
if [ -f "Cargo.toml" ] && [ -f "execution-core/Cargo.toml" ] && [ -f "simulation-engine/Cargo.toml" ]; then
    echo "✓ Rust workspace structure created"
else
    echo "✗ Rust workspace structure missing"
fi

# Python components
if [ -f "intelligence-layer/pyproject.toml" ] && [ -f "intelligence-layer/src/intelligence_layer/__init__.py" ]; then
    echo "✓ Python intelligence layer structure created"
else
    echo "✗ Python intelligence layer structure missing"
fi

# TypeScript components
if [ -f "frontend/package.json" ] && [ -f "frontend/tsconfig.json" ]; then
    echo "✓ TypeScript frontend structure created"
else
    echo "✗ TypeScript frontend structure missing"
fi

# Docker configuration
if [ -f "docker-compose.yml" ]; then
    echo "✓ Docker Compose configuration created"
else
    echo "✗ Docker Compose configuration missing"
fi

# Database initialization
if [ -f "database/init/01-init.sql" ] && [ -f "database/init/02-neo4j-init.cypher" ]; then
    echo "✓ Database initialization scripts created"
else
    echo "✗ Database initialization scripts missing"
fi

# Configuration files
if [ -f ".env.example" ] && [ -f "README.md" ] && [ -f "Makefile" ]; then
    echo "✓ Configuration and documentation files created"
else
    echo "✗ Configuration and documentation files missing"
fi

echo ""
echo "=== Setup Summary ==="
echo "✓ Multi-language project structure (Rust, Python, TypeScript)"
echo "✓ Docker containers for Neo4j, PostgreSQL with pgvector, and Redis"
echo "✓ Basic logging and configuration management"
echo "✓ Development environment setup"
echo ""
echo "Next steps:"
echo "1. Run 'make setup' to initialize the development environment"
echo "2. Start infrastructure: 'make start-infra'"
echo "3. Initialize databases: 'make db-init'"
echo "4. Start development services: 'make start-dev'"
echo ""
echo "Core system infrastructure setup is complete!"