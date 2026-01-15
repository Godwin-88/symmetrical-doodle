# Algorithmic Trading System - Development Makefile

.PHONY: help setup start stop clean test lint format build

# Default target
help:
	@echo "Algorithmic Trading System - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Initial project setup"
	@echo "  setup-env      - Copy environment template"
	@echo ""
	@echo "Development:"
	@echo "  start          - Start all services"
	@echo "  start-infra    - Start infrastructure only (databases)"
	@echo "  start-dev      - Start development services"
	@echo "  stop           - Stop all services"
	@echo "  restart        - Restart all services"
	@echo ""
	@echo "Database:"
	@echo "  db-init        - Initialize databases"
	@echo "  db-reset       - Reset databases (WARNING: destroys data)"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-rust      - Run Rust tests"
	@echo "  test-python    - Run Python tests"
	@echo "  test-frontend  - Run frontend tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run all linters"
	@echo "  format         - Format all code"
	@echo "  check          - Run all checks (lint + test)"
	@echo ""
	@echo "Build:"
	@echo "  build          - Build all services"
	@echo "  clean          - Clean build artifacts"

# Setup
setup: setup-env
	@echo "Setting up development environment..."
	@make start-infra
	@sleep 10
	@make db-init
	@echo "Setup complete! Run 'make start-dev' to start development services."

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template"; \
	else \
		echo ".env file already exists"; \
	fi

# Development
start:
	docker-compose up -d

start-infra:
	docker-compose up -d postgres neo4j redis

start-dev:
	@echo "Starting development services..."
	@echo "Execution Core: http://localhost:8001"
	@echo "Intelligence API: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo ""
	@echo "Run in separate terminals:"
	@echo "  cd execution-core && cargo run"
	@echo "  cd intelligence-layer && uvicorn intelligence_layer.main:app --reload"
	@echo "  cd frontend && npm run dev"

stop:
	docker-compose down

restart: stop start

# Database
db-init:
	@echo "Initializing databases..."
	@echo "PostgreSQL will auto-initialize from init scripts"
	@sleep 5
	@echo "Initializing Neo4j..."
	@docker exec -i trading-neo4j cypher-shell -u neo4j -p password < database/init/02-neo4j-init.cypher || true
	@echo "Database initialization complete"

db-reset:
	@echo "WARNING: This will destroy all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker-compose up -d postgres neo4j redis; \
		sleep 10; \
		make db-init; \
	fi

# Testing
test: test-rust test-python test-frontend

test-rust:
	@echo "Running Rust tests..."
	cargo test

test-python:
	@echo "Running Python tests..."
	cd intelligence-layer && python -m pytest

test-frontend:
	@echo "Running frontend tests..."
	cd frontend && npm test

# Code Quality
lint: lint-rust lint-python lint-frontend

lint-rust:
	@echo "Linting Rust code..."
	cargo clippy -- -D warnings

lint-python:
	@echo "Linting Python code..."
	cd intelligence-layer && flake8 src/
	cd intelligence-layer && mypy src/

lint-frontend:
	@echo "Linting frontend code..."
	cd frontend && npm run lint

format: format-rust format-python format-frontend

format-rust:
	@echo "Formatting Rust code..."
	cargo fmt

format-python:
	@echo "Formatting Python code..."
	cd intelligence-layer && black src/
	cd intelligence-layer && isort src/

format-frontend:
	@echo "Formatting frontend code..."
	cd frontend && npm run lint -- --fix

check: lint test

# Build
build:
	@echo "Building all services..."
	docker-compose build

build-rust:
	@echo "Building Rust services..."
	cargo build --release

build-python:
	@echo "Building Python package..."
	cd intelligence-layer && pip install -e .

build-frontend:
	@echo "Building frontend..."
	cd frontend && npm run build

# Clean
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	cd frontend && rm -rf dist node_modules
	docker-compose down --rmi local -v
	docker system prune -f

# Health checks
health:
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health || echo "Intelligence API: DOWN"
	@curl -f http://localhost:8001/health || echo "Execution Core: DOWN"
	@curl -f http://localhost:3000 || echo "Frontend: DOWN"

# Logs
logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f intelligence-api

logs-core:
	docker-compose logs -f execution-core

logs-frontend:
	docker-compose logs -f frontend

# Development utilities
shell-postgres:
	docker exec -it trading-postgres psql -U postgres -d trading_system

shell-neo4j:
	docker exec -it trading-neo4j cypher-shell -u neo4j -p password

shell-redis:
	docker exec -it trading-redis redis-cli