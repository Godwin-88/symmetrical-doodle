# Makefile for Google Drive Knowledge Base Ingestion System

.PHONY: help setup install test clean lint format type-check security-check run validate build deploy

# Default target
help:
	@echo "Google Drive Knowledge Base Ingestion System"
	@echo "Available targets:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup          - Set up virtual environment and install dependencies"
	@echo "  install        - Install dependencies only"
	@echo "  setup-dev      - Setup development environment"
	@echo "  setup-prod     - Setup production environment"
	@echo ""
	@echo "Testing and Validation:"
	@echo "  test           - Run test suite"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-property  - Run property-based tests"
	@echo "  validate       - Validate deployment configuration"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code"
	@echo "  type-check     - Run type checking"
	@echo "  security-check - Run security scanning"
	@echo ""
	@echo "Building and Deployment:"
	@echo "  build          - Build Docker image"
	@echo "  deploy-local   - Deploy for local execution"
	@echo "  deploy-docker  - Deploy using Docker"
	@echo "  deploy-compose - Deploy using Docker Compose"
	@echo ""
	@echo "Execution:"
	@echo "  run            - Run ingestion pipeline"
	@echo "  run-dev        - Run in development mode"
	@echo "  run-discovery  - Run discovery phase only"
	@echo "  run-resume     - Resume previous execution"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          - Clean temporary files"
	@echo "  clean-all      - Clean all generated files"
	@echo "  logs           - Show recent logs"
	@echo ""

# Environment setup
setup:
	python setup_environment.py

install:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt
	python deploy.py local --config-env development
	@echo "Development environment setup complete"

setup-prod:
	pip install -r requirements.txt
	python deploy.py local --config-env production
	@echo "Production environment setup complete"

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term

test-property:
	pytest tests/ -k "property" -v

test-integration:
	pytest tests/ -k "integration" -v

# Code quality
lint:
	flake8 . --max-line-length=100 --exclude=venv,__pycache__,.pytest_cache
	isort . --check-only --diff

format:
	black . --line-length=100
	isort .

type-check:
	mypy . --ignore-missing-imports

security-check:
	bandit -r . -x tests/,venv/

# Building and Deployment
build:
	python build.py --tag latest

build-dev:
	python build.py --tag dev --env development

build-prod:
	python build.py --tag latest --env production

deploy-local:
	python deploy.py local

deploy-docker:
	python deploy.py docker

deploy-compose:
	python deploy.py docker-compose

# Application execution
run:
	python run_ingestion.py

run-dev:
	python run_ingestion.py --config-env development --log-level DEBUG

run-prod:
	python run_ingestion.py --config-env production

run-discovery:
	python run_ingestion.py --phases discovery

run-resume:
	python run_ingestion.py --resume

run-fresh:
	python run_ingestion.py --no-resume

# Validation
validate:
	python validate_deployment.py --verbose

validate-config:
	python -c "from core.config import ConfigManager; cm = ConfigManager(); result = cm.validate_config(); print('Valid:', result['valid']); [print('Error:', e) for e in result['errors']]"

# Docker targets
docker-build:
	docker build --target production --tag knowledge-ingestion:latest .

docker-run:
	docker run --rm -it \
		-v $(PWD)/credentials:/app/data/credentials:ro \
		-v $(PWD)/data/logs:/app/data/logs \
		knowledge-ingestion:latest

docker-shell:
	docker run --rm -it \
		-v $(PWD)/credentials:/app/data/credentials:ro \
		-v $(PWD)/data/logs:/app/data/logs \
		knowledge-ingestion:latest /bin/bash

# Docker Compose targets
compose-up:
	docker-compose up knowledge-ingestion

compose-build:
	docker-compose build

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f knowledge-ingestion

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type f -name "*.tmp" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

clean-all: clean
	rm -rf data/logs/*
	rm -rf data/state/*
	rm -rf data/extracted_pdfs/*
	rm -rf .hypothesis/

# Development helpers
dev-setup: setup
	pip install pytest-cov pytest-xdist pre-commit flake8 bandit
	pre-commit install

logs:
	@if [ -d "data/logs" ]; then \
		echo "Recent log entries:"; \
		find data/logs -name "*.log" -type f -exec tail -n 20 {} + 2>/dev/null || echo "No log files found"; \
	else \
		echo "No logs directory found"; \
	fi

# Status and information
status:
	@echo "System Status:"
	@echo "=============="
	@python -c "from core.config import get_settings; s = get_settings(); print(f'Environment: {s.environment}')" 2>/dev/null || echo "Environment: Not configured"
	@python -c "from core.state_manager import StateManager; sm = StateManager(); state = sm.load_state(); print(f'Last execution: {state.execution_id if state else \"None\"}')" 2>/dev/null || echo "Last execution: None"
	@echo ""
	@echo "Configuration status:"
	@python validate_deployment.py 2>/dev/null | grep -E "(passed|failed)" || echo "Run 'make validate' for detailed status"

info:
	@echo "Google Drive Knowledge Base Ingestion System"
	@echo "==========================================="
	@echo ""
	@echo "Project Structure:"
	@echo "  config/          - Configuration files"
	@echo "  core/            - Core system modules"
	@echo "  services/        - Service implementations"
	@echo "  tests/           - Test files"
	@echo "  data/            - Runtime data (logs, state, cache)"
	@echo "  credentials/     - Authentication credentials"
	@echo ""
	@echo "Key Files:"
	@echo "  run_ingestion.py     - Main execution script"
	@echo "  validate_deployment.py - Deployment validation"
	@echo "  deploy.py           - Deployment management"
	@echo "  Dockerfile          - Container definition"
	@echo "  docker-compose.yml  - Multi-container setup"