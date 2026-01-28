# Production Deployment Strategy

## Overview

This document outlines the comprehensive production deployment strategy for the Algorithmic Trading System using a modern cloud-native approach with Vercel, Render, Supabase, and Neo4j Aura.

## 🏗️ Architecture Decision: Docker on Render (Recommended)

### Why Docker on Render?
1. **Consistency**: Same environment across development, staging, and production
2. **Multi-language Support**: Seamless Rust + Python deployment
3. **Resource Optimization**: Better control over resource allocation
4. **Scalability**: Easy horizontal scaling with container orchestration
5. **Cost Efficiency**: Single Render account vs multiple service plans
6. **Monitoring**: Unified logging and monitoring across services

## 🎯 Deployment Architecture

```mermaid
graph TB
    subgraph "Frontend (Vercel)"
        FE[React/TypeScript App<br/>algorithmic-trading.vercel.app]
    end

    subgraph "Backend Services (Render - Docker)"
        subgraph "Intelligence Service"
            IL[Intelligence Layer<br/>Python/FastAPI<br/>Port 8000]
        end
        
        subgraph "Execution Service"
            EC[Execution Core<br/>Rust/Axum<br/>Port 8001]
        end
        
        subgraph "Simulation Service"
            SE[Simulation Engine<br/>Rust<br/>Port 8002]
        end
        
        subgraph "Redis Cache"
            REDIS[Redis Container<br/>Port 6379]
        end
    end

    subgraph "Managed Databases"
        SUPA[(Supabase<br/>PostgreSQL + Vector)]
        NEO[(Neo4j Aura<br/>Graph Database)]
    end

    subgraph "External APIs"
        DERIV[Deriv API]
        LLM[LLM Providers<br/>OpenAI/Anthropic]
        MARKET[Market Data APIs]
    end

    %% Connections
    FE --> IL
    FE --> EC
    FE --> SE
    
    IL --> SUPA
    IL --> NEO
    IL --> REDIS
    IL --> LLM
    
    EC --> SUPA
    EC --> REDIS
    EC --> DERIV
    
    SE --> SUPA
    SE --> REDIS

    classDef frontend fill:#ff8c00,stroke:#000,stroke-width:2px,color:#000
    classDef backend fill:#4a90e2,stroke:#000,stroke-width:2px,color:#fff
    classDef database fill:#2ecc71,stroke:#000,stroke-width:2px,color:#fff
    classDef external fill:#e74c3c,stroke:#000,stroke-width:2px,color:#fff

    class FE frontend
    class IL,EC,SE,REDIS backend
    class SUPA,NEO database
    class DERIV,LLM,MARKET external
```

## 📦 Containerization Strategy

### Multi-Service Docker Setup

#### 1. Intelligence Layer Dockerfile
```dockerfile
# intelligence-layer/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "intelligence_layer.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Execution Core Dockerfile
```dockerfile
# execution-core/Dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /app/target/release/execution-core /usr/local/bin/execution-core

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["execution-core"]
```

#### 3. Simulation Engine Dockerfile
```dockerfile
# simulation-engine/Dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/simulation-engine /usr/local/bin/simulation-engine

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["simulation-engine"]
```

#### 4. Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  intelligence-layer:
    build:
      context: ./intelligence-layer
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${SUPABASE_URL}
      - NEO4J_URI=${NEO4J_URI}
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  execution-core:
    build:
      context: ./execution-core
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=${SUPABASE_URL}
      - REDIS_URL=redis://redis:6379
      - DERIV_API_TOKEN=${DERIV_API_TOKEN}
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  simulation-engine:
    build:
      context: ./simulation-engine
      dockerfile: Dockerfile
    ports:
      - "8002:8002"
    environment:
      - DATABASE_URL=${SUPABASE_URL}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  redis_data:

networks:
  default:
    name: trading-system
```

## 🚀 Deployment Configuration

### 1. Vercel Frontend Deployment

#### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "frontend/package.json",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "https://your-render-backend.onrender.com/api/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/frontend/$1"
    }
  ],
  "env": {
    "VITE_API_BASE_URL": "https://your-render-backend.onrender.com",
    "VITE_WS_URL": "wss://your-render-backend.onrender.com"
  },
  "functions": {
    "frontend/src/pages/api/*.ts": {
      "runtime": "nodejs18.x"
    }
  }
}
```

#### Frontend Environment Configuration
```typescript
// frontend/src/config/environment.ts
const config = {
  development: {
    API_BASE_URL: 'http://localhost:8000',
    WS_URL: 'ws://localhost:8000',
    EXECUTION_API_URL: 'http://localhost:8001',
    SIMULATION_API_URL: 'http://localhost:8002'
  },
  production: {
    API_BASE_URL: process.env.VITE_API_BASE_URL || 'https://your-render-backend.onrender.com',
    WS_URL: process.env.VITE_WS_URL || 'wss://your-render-backend.onrender.com',
    EXECUTION_API_URL: process.env.VITE_EXECUTION_API_URL || 'https://your-execution-service.onrender.com',
    SIMULATION_API_URL: process.env.VITE_SIMULATION_API_URL || 'https://your-simulation-service.onrender.com'
  }
};

export default config[process.env.NODE_ENV || 'development'];
```

### 2. Render Backend Deployment

#### render.yaml (Infrastructure as Code)
```yaml
# render.yaml
services:
  # Intelligence Layer Service
  - type: web
    name: intelligence-layer
    env: docker
    dockerfilePath: ./intelligence-layer/Dockerfile
    plan: standard
    region: oregon
    branch: main
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: supabase-connection
          property: connectionString
      - key: NEO4J_URI
        value: bolt+s://your-neo4j-instance.databases.neo4j.io
      - key: NEO4J_USER
        value: neo4j
      - key: NEO4J_PASSWORD
        sync: false
      - key: REDIS_URL
        fromService:
          type: redis
          name: trading-redis
          property: connectionString
      - key: OPENAI_API_KEY
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
    healthCheckPath: /health
    domains:
      - intelligence-api.your-domain.com

  # Execution Core Service
  - type: web
    name: execution-core
    env: docker
    dockerfilePath: ./execution-core/Dockerfile
    plan: standard
    region: oregon
    branch: main
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: supabase-connection
          property: connectionString
      - key: REDIS_URL
        fromService:
          type: redis
          name: trading-redis
          property: connectionString
      - key: DERIV_API_TOKEN
        sync: false
      - key: DERIV_APP_ID
        sync: false
    healthCheckPath: /health
    domains:
      - execution-api.your-domain.com

  # Simulation Engine Service
  - type: web
    name: simulation-engine
    env: docker
    dockerfilePath: ./simulation-engine/Dockerfile
    plan: starter
    region: oregon
    branch: main
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: supabase-connection
          property: connectionString
      - key: REDIS_URL
        fromService:
          type: redis
          name: trading-redis
          property: connectionString
    healthCheckPath: /health

  # Redis Cache
  - type: redis
    name: trading-redis
    plan: starter
    region: oregon
    maxmemoryPolicy: allkeys-lru

databases:
  # This represents the external Supabase connection
  - name: supabase-connection
    databaseName: postgres
    user: postgres
```

### 3. Database Configuration

#### Supabase Setup
```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS intelligence;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Create tables for trading data
CREATE TABLE trading.portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES trading.portfolios(id),
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 8) NOT NULL,
    current_price DECIMAL(18, 8),
    pnl DECIMAL(18, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for intelligence data
CREATE TABLE intelligence.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id VARCHAR(20) NOT NULL,
    embedding vector(512),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE intelligence.research_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    query TEXT NOT NULL,
    content JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    confidence DECIMAL(5, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_positions_portfolio_id ON trading.positions(portfolio_id);
CREATE INDEX idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX idx_embeddings_asset_id ON intelligence.embeddings(asset_id);
CREATE INDEX idx_research_reports_status ON intelligence.research_reports(status);

-- Row Level Security (RLS)
ALTER TABLE trading.portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading.positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.research_reports ENABLE ROW LEVEL SECURITY;
```

#### Neo4j Aura Configuration
```cypher
// Create constraints
CREATE CONSTRAINT asset_id_unique IF NOT EXISTS FOR (a:Asset) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT regime_id_unique IF NOT EXISTS FOR (r:Regime) REQUIRE r.id IS UNIQUE;

// Create indexes
CREATE INDEX asset_symbol_index IF NOT EXISTS FOR (a:Asset) ON (a.symbol);
CREATE INDEX regime_name_index IF NOT EXISTS FOR (r:Regime) ON (r.name);

// Create sample nodes
CREATE (a1:Asset {id: 'EURUSD', symbol: 'EURUSD', type: 'currency_pair'})
CREATE (a2:Asset {id: 'GBPUSD', symbol: 'GBPUSD', type: 'currency_pair'})
CREATE (a3:Asset {id: 'USDJPY', symbol: 'USDJPY', type: 'currency_pair'})

CREATE (r1:Regime {id: 'low_vol_trending', name: 'Low Volatility Trending'})
CREATE (r2:Regime {id: 'high_vol_ranging', name: 'High Volatility Ranging'})
CREATE (r3:Regime {id: 'crisis', name: 'Crisis Mode'})

// Create relationships
CREATE (a1)-[:CORRELATES_WITH {strength: 0.85}]->(a2)
CREATE (a1)-[:IN_REGIME {probability: 0.65}]->(r1)
CREATE (a2)-[:IN_REGIME {probability: 0.45}]->(r2)
```

## 🔧 Environment Configuration

### 1. Environment Variables Structure

#### .env.production
```bash
# Database URLs
SUPABASE_URL=postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Neo4j Configuration
NEO4J_URI=bolt+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Redis Configuration (Render managed)
REDIS_URL=redis://red-xxxxx:6379

# External APIs
DERIV_API_TOKEN=your_deriv_token
DERIV_APP_ID=your_deriv_app_id
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Service URLs
INTELLIGENCE_API_URL=https://intelligence-api.your-domain.com
EXECUTION_API_URL=https://execution-api.your-domain.com
SIMULATION_API_URL=https://simulation-api.your-domain.com

# Security
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key

# Monitoring
SENTRY_DSN=your_sentry_dsn
LOG_LEVEL=info
```

### 2. Service Configuration Files

#### Intelligence Layer Configuration
```python
# intelligence-layer/src/intelligence_layer/config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = os.getenv("DATABASE_URL")
    neo4j_uri: str = os.getenv("NEO4J_URI")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")
    redis_url: str = os.getenv("REDIS_URL")
    
    # External APIs
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY")
    
    # Service Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    # CORS
    allowed_origins: list = [
        "https://your-frontend.vercel.app",
        "https://localhost:5173",
        "http://localhost:5173"
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### Execution Core Configuration
```rust
// execution-core/src/config.rs
use serde::Deserialize;
use std::env;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub database_url: String,
    pub redis_url: String,
    pub deriv_api_token: String,
    pub deriv_app_id: String,
    pub host: String,
    pub port: u16,
    pub log_level: String,
}

impl Config {
    pub fn from_env() -> Result<Self, env::VarError> {
        Ok(Config {
            database_url: env::var("DATABASE_URL")?,
            redis_url: env::var("REDIS_URL")?,
            deriv_api_token: env::var("DERIV_API_TOKEN")?,
            deriv_app_id: env::var("DERIV_APP_ID")?,
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("PORT")
                .unwrap_or_else(|_| "8001".to_string())
                .parse()
                .unwrap_or(8001),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        })
    }
}
```

## 🚀 Deployment Pipeline

### 1. GitHub Actions Workflow

#### .github/workflows/deploy.yml
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd frontend && npm ci
          cd ../intelligence-layer && pip install -r requirements.txt
          cd ../execution-core && cargo build
      
      - name: Run tests
        run: |
          cd frontend && npm test
          cd ../intelligence-layer && pytest
          cd ../execution-core && cargo test

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    strategy:
      matrix:
        service: [intelligence-layer, execution-core, simulation-engine]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: ./${{ matrix.service }}
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.service }}:latest

  deploy-frontend:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: ./frontend
          vercel-args: '--prod'

  deploy-backend:
    needs: [test, build-and-push]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to Render
        uses: johnbeynon/render-deploy-action@v0.0.8
        with:
          service-id: ${{ secrets.RENDER_SERVICE_ID }}
          api-key: ${{ secrets.RENDER_API_KEY }}
```

### 2. Deployment Scripts

#### scripts/deploy-production.sh
```bash
#!/bin/bash

set -e

echo "🚀 Starting production deployment..."

# Build and push Docker images
echo "📦 Building Docker images..."
docker build -t ghcr.io/your-org/trading-system/intelligence-layer:latest ./intelligence-layer
docker build -t ghcr.io/your-org/trading-system/execution-core:latest ./execution-core
docker build -t ghcr.io/your-org/trading-system/simulation-engine:latest ./simulation-engine

echo "📤 Pushing Docker images..."
docker push ghcr.io/your-org/trading-system/intelligence-layer:latest
docker push ghcr.io/your-org/trading-system/execution-core:latest
docker push ghcr.io/your-org/trading-system/simulation-engine:latest

# Deploy to Render
echo "🚀 Deploying to Render..."
curl -X POST \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clearCache": false}' \
  "https://api.render.com/v1/services/$RENDER_SERVICE_ID/deploys"

# Deploy frontend to Vercel
echo "🌐 Deploying frontend to Vercel..."
cd frontend
vercel --prod --token $VERCEL_TOKEN

echo "✅ Deployment completed successfully!"
```

## 📊 Monitoring & Observability

### 1. Health Check Endpoints

#### Intelligence Layer Health Check
```python
# intelligence-layer/src/intelligence_layer/health.py
from fastapi import APIRouter, HTTPException
from sqlalchemy import text
import redis
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check():
    checks = {
        "service": "intelligence-layer",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Database check
    try:
        async with get_db() as db:
            await db.execute(text("SELECT 1"))
        checks["checks"]["database"] = "healthy"
    except Exception as e:
        checks["checks"]["database"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    # Redis check
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        checks["checks"]["redis"] = "healthy"
    except Exception as e:
        checks["checks"]["redis"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    # Neo4j check
    try:
        with get_neo4j_driver() as driver:
            with driver.session() as session:
                session.run("RETURN 1")
        checks["checks"]["neo4j"] = "healthy"
    except Exception as e:
        checks["checks"]["neo4j"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    if checks["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=checks)
    
    return checks
```

### 2. Logging Configuration

#### Structured Logging Setup
```python
# intelligence-layer/src/intelligence_layer/logging.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": "intelligence-layer",
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        return json.dumps(log_entry)

def setup_logging():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

### 3. Monitoring Dashboard

#### Render Service Monitoring
```python
# monitoring/render_monitor.py
import requests
import time
from datetime import datetime

class RenderMonitor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.render.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_service_status(self, service_id: str):
        response = requests.get(
            f"{self.base_url}/services/{service_id}",
            headers=self.headers
        )
        return response.json()
    
    def get_service_metrics(self, service_id: str):
        response = requests.get(
            f"{self.base_url}/services/{service_id}/metrics",
            headers=self.headers
        )
        return response.json()
    
    def monitor_all_services(self, service_ids: list):
        while True:
            for service_id in service_ids:
                try:
                    status = self.get_service_status(service_id)
                    metrics = self.get_service_metrics(service_id)
                    
                    print(f"Service {service_id}: {status['service']['state']}")
                    print(f"CPU: {metrics.get('cpu', 'N/A')}%")
                    print(f"Memory: {metrics.get('memory', 'N/A')}MB")
                    print("---")
                    
                except Exception as e:
                    print(f"Error monitoring {service_id}: {e}")
            
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor = RenderMonitor(os.getenv("RENDER_API_KEY"))
    service_ids = [
        "srv-intelligence-layer",
        "srv-execution-core",
        "srv-simulation-engine"
    ]
    monitor.monitor_all_services(service_ids)
```

## 💰 Cost Optimization

### 1. Service Tier Recommendations

#### Render Services
- **Intelligence Layer**: Standard ($25/month) - CPU intensive AI/ML operations
- **Execution Core**: Standard ($25/month) - Critical trading operations
- **Simulation Engine**: Starter ($7/month) - Batch processing, can scale up when needed
- **Redis**: Starter ($7/month) - Sufficient for caching and sessions

**Total Render Cost**: ~$64/month

#### External Services
- **Vercel**: Pro ($20/month) - Professional frontend hosting
- **Supabase**: Pro ($25/month) - Production database with vector support
- **Neo4j Aura**: Professional ($65/month) - Graph database for analytics

**Total Monthly Cost**: ~$174/month

### 2. Cost Optimization Strategies

#### Auto-scaling Configuration
```yaml
# render.yaml - Auto-scaling settings
services:
  - type: web
    name: intelligence-layer
    autoDeploy: true
    scaling:
      minInstances: 1
      maxInstances: 3
      targetCPUPercent: 70
      targetMemoryPercent: 80
```

#### Resource Optimization
```python
# Optimize database connections
DATABASE_POOL_SIZE = 5
DATABASE_MAX_OVERFLOW = 10
DATABASE_POOL_TIMEOUT = 30
DATABASE_POOL_RECYCLE = 3600

# Optimize Redis connections
REDIS_POOL_SIZE = 10
REDIS_CONNECTION_TIMEOUT = 5
```

## 🔒 Security Configuration

### 1. Environment Security

#### Secrets Management
```bash
# Use Render's secret management
render secrets set OPENAI_API_KEY "your-secret-key"
render secrets set DATABASE_URL "your-database-url"
render secrets set JWT_SECRET "your-jwt-secret"
```

#### CORS Configuration
```python
# intelligence-layer/src/intelligence_layer/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend.vercel.app",
        "https://localhost:5173"  # Development only
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### 2. Database Security

#### Supabase RLS Policies
```sql
-- Row Level Security policies
CREATE POLICY "Users can only access their own portfolios"
ON trading.portfolios FOR ALL
USING (auth.uid() = user_id);

CREATE POLICY "Users can only access their own positions"
ON trading.positions FOR ALL
USING (
  portfolio_id IN (
    SELECT id FROM trading.portfolios WHERE user_id = auth.uid()
  )
);
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] All environment variables configured
- [ ] Database migrations completed
- [ ] Docker images built and tested
- [ ] Health checks implemented
- [ ] Monitoring configured
- [ ] Security policies applied

### Deployment
- [ ] Frontend deployed to Vercel
- [ ] Backend services deployed to Render
- [ ] Database connections verified
- [ ] External API integrations tested
- [ ] SSL certificates configured
- [ ] Domain names configured

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring dashboards active
- [ ] Performance metrics within targets
- [ ] Security scans completed
- [ ] Backup procedures verified
- [ ] Disaster recovery tested

## 🚨 Troubleshooting Guide

### Common Issues

#### 1. Container Build Failures
```bash
# Check build logs
render logs --service-id srv-your-service-id --type build

# Local debugging
docker build -t test-image ./intelligence-layer
docker run -it test-image /bin/bash
```

#### 2. Database Connection Issues
```python
# Test database connectivity
import asyncpg

async def test_connection():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        result = await conn.fetchval("SELECT 1")
        print(f"Database connection successful: {result}")
        await conn.close()
    except Exception as e:
        print(f"Database connection failed: {e}")
```

#### 3. Service Communication Issues
```bash
# Test service endpoints
curl -f https://intelligence-api.your-domain.com/health
curl -f https://execution-api.your-domain.com/health

# Check service logs
render logs --service-id srv-intelligence-layer --tail
```

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Deployment Target**: Production  
**Estimated Setup Time**: 2-3 days  
**Monthly Cost**: ~$174