# Algorithmic Trading System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying and operating the algorithmic trading system in academic and research environments.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Memory**: Minimum 8GB RAM, 16GB recommended
- **Storage**: 50GB available disk space
- **CPU**: 4+ cores recommended
- **Network**: Stable internet connection for market data

### Software Dependencies
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Python**: 3.9+ (for development)
- **Node.js**: 16+ (for frontend development)
- **Rust**: 1.70+ (for core components)

## Quick Start Deployment

### 1. Clone Repository
```bash
git clone <repository-url>
cd algorithmic-trading-system
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

### 3. Deploy with Docker Compose
```bash
# Start all services
docker-compose up -d

# Verify deployment
docker-compose ps
```

### 4. Initialize Database
```bash
# Run database initialization
docker-compose exec intelligence-layer python -m intelligence_layer.setup_database

# Verify database schema
docker-compose exec postgres psql -U postgres -d trading_system -c "\dt"
docker-compose exec neo4j cypher-shell -u neo4j -p password "SHOW DATABASES"
```

### 5. Access System
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474
- **Health Dashboard**: http://localhost:8000/health

## Detailed Configuration

### Environment Variables (.env)

```bash
# Database Configuration
POSTGRES_URL=postgresql://postgres:password@postgres:5432/trading_system
NEO4J_URL=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://redis:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here

# Execution Configuration
DERIV_API_TOKEN=your-deriv-token
EXECUTION_MODE=shadow  # shadow or live
RISK_LIMITS_ENABLED=true

# Intelligence Configuration
EMBEDDING_MODEL_PATH=/models/embedding_model.pkl
REGIME_MODEL_PATH=/models/regime_model.pkl
FEATURE_WINDOW_SIZE=64

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/logs/system.log

# Research Configuration
EXPERIMENTS_DIR=/data/experiments
NEGATIVE_FINDINGS_DIR=/data/negative_findings
ACADEMIC_VALIDATION_ENABLED=true
```

### Database Configuration

#### PostgreSQL Setup
```sql
-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS intelligence;
CREATE SCHEMA IF NOT EXISTS research;
```

#### Neo4j Configuration
```cypher
// Create constraints
CREATE CONSTRAINT asset_symbol IF NOT EXISTS FOR (a:Asset) REQUIRE a.symbol IS UNIQUE;
CREATE CONSTRAINT strategy_name IF NOT EXISTS FOR (s:Strategy) REQUIRE s.name IS UNIQUE;

// Create indexes
CREATE INDEX regime_timestamp IF NOT EXISTS FOR (r:MarketRegime) ON (r.timestamp);
CREATE INDEX signal_timestamp IF NOT EXISTS FOR (s:IntelligenceSignal) ON (s.timestamp);
```

## Service Architecture

### Core Services

#### 1. Execution Core (Rust)
```yaml
execution-core:
  build: ./execution-core
  ports:
    - "8001:8001"
  environment:
    - RUST_LOG=info
    - DATABASE_URL=${POSTGRES_URL}
  volumes:
    - ./logs:/logs
    - ./config:/config
```

#### 2. Intelligence Layer (Python)
```yaml
intelligence-layer:
  build: ./intelligence-layer
  ports:
    - "8000:8000"
  environment:
    - PYTHONPATH=/app/src
    - DATABASE_URL=${POSTGRES_URL}
    - NEO4J_URL=${NEO4J_URL}
  volumes:
    - ./data:/data
    - ./models:/models
    - ./logs:/logs
```

#### 3. Frontend (React/TypeScript)
```yaml
frontend:
  build: ./frontend
  ports:
    - "3000:80"
  environment:
    - REACT_APP_API_URL=http://localhost:8000
  volumes:
    - ./frontend/nginx.conf:/etc/nginx/nginx.conf
```

### Supporting Services

#### Database Services
```yaml
postgres:
  image: pgvector/pgvector:pg15
  environment:
    POSTGRES_DB: trading_system
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: password
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./database/init:/docker-entrypoint-initdb.d

neo4j:
  image: neo4j:5.0
  environment:
    NEO4J_AUTH: neo4j/password
    NEO4J_PLUGINS: '["graph-data-science"]'
  volumes:
    - neo4j_data:/data
    - ./database/neo4j:/import

redis:
  image: redis:7-alpine
  volumes:
    - redis_data:/data
```

## Monitoring and Observability

### Health Checks
```bash
# System health overview
curl http://localhost:8000/health

# Component-specific health
curl http://localhost:8000/health/intelligence
curl http://localhost:8001/health/execution
curl http://localhost:3000/health/frontend
```

### Logging Configuration

#### Centralized Logging
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "5"
    labels: "service,version"
```

#### Log Aggregation (Optional)
```yaml
# Add to docker-compose.yml for production
elasticsearch:
  image: elasticsearch:8.0.0
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false

kibana:
  image: kibana:8.0.0
  ports:
    - "5601:5601"
  environment:
    - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
```

### Metrics Collection
```yaml
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana
  ports:
    - "3001:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
  volumes:
    - grafana_data:/var/lib/grafana
```

## Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificates for development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ./certs/private.key \
  -out ./certs/certificate.crt

# Update nginx configuration
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/certs/certificate.crt;
    ssl_certificate_key /etc/ssl/private/private.key;
}
```

### API Security
```python
# JWT Configuration
JWT_SECRET_KEY = "your-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Rate Limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds
```

### Database Security
```bash
# PostgreSQL security
echo "host all all 0.0.0.0/0 md5" >> /var/lib/postgresql/data/pg_hba.conf

# Neo4j security
NEO4J_AUTH=neo4j/strong-password-here
NEO4J_dbms_security_auth__enabled=true
```

## Backup and Recovery

### Automated Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# PostgreSQL backup
docker-compose exec postgres pg_dump -U postgres trading_system > $BACKUP_DIR/postgres.sql

# Neo4j backup
docker-compose exec neo4j neo4j-admin database dump --to-path=/backups neo4j

# Application data backup
tar -czf $BACKUP_DIR/app_data.tar.gz ./data ./logs ./config

# Upload to cloud storage (optional)
aws s3 cp $BACKUP_DIR s3://your-backup-bucket/$(basename $BACKUP_DIR) --recursive
```

### Recovery Procedures
```bash
#!/bin/bash
# restore.sh

BACKUP_DIR=$1

# Stop services
docker-compose down

# Restore PostgreSQL
docker-compose up -d postgres
sleep 10
docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS trading_system;"
docker-compose exec postgres psql -U postgres -c "CREATE DATABASE trading_system;"
docker-compose exec postgres psql -U postgres trading_system < $BACKUP_DIR/postgres.sql

# Restore Neo4j
docker-compose up -d neo4j
sleep 10
docker-compose exec neo4j neo4j-admin database load --from-path=/backups neo4j

# Restore application data
tar -xzf $BACKUP_DIR/app_data.tar.gz

# Start all services
docker-compose up -d
```

## Performance Tuning

### Database Optimization

#### PostgreSQL Tuning
```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

#### Neo4j Tuning
```properties
# neo4j.conf optimizations
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G
dbms.tx_log.rotation.retention_policy=1 days
```

### Application Performance
```python
# Intelligence Layer optimizations
BATCH_SIZE = 32
MAX_WORKERS = 4
CACHE_TTL = 300  # seconds
EMBEDDING_CACHE_SIZE = 1000

# Connection pooling
DATABASE_POOL_SIZE = 10
DATABASE_MAX_OVERFLOW = 20
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Restart specific service
docker-compose restart service-name
```

#### 2. Database Connection Issues
```bash
# Test PostgreSQL connection
docker-compose exec postgres psql -U postgres -c "SELECT 1;"

# Test Neo4j connection
docker-compose exec neo4j cypher-shell -u neo4j -p password "RETURN 1;"

# Check network connectivity
docker-compose exec intelligence-layer ping postgres
```

#### 3. Performance Issues
```bash
# Monitor resource usage
docker stats --no-stream

# Check database performance
docker-compose exec postgres psql -U postgres -c "SELECT * FROM pg_stat_activity;"

# Profile application
docker-compose exec intelligence-layer python -m cProfile -o profile.stats main.py
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debugger
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up

# Access debug endpoints
curl http://localhost:8000/debug/metrics
curl http://localhost:8000/debug/config
```

## Production Deployment

### Infrastructure Requirements
- **Load Balancer**: Nginx or HAProxy
- **Container Orchestration**: Kubernetes or Docker Swarm
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack or similar
- **Backup**: Automated daily backups with offsite storage

### Security Hardening
- **Network Segmentation**: Separate networks for different tiers
- **Secrets Management**: Use Kubernetes secrets or HashiCorp Vault
- **Image Scanning**: Regular vulnerability scans of container images
- **Access Control**: RBAC with principle of least privilege

### Scaling Considerations
- **Horizontal Scaling**: Stateless service design enables easy scaling
- **Database Scaling**: Read replicas and connection pooling
- **Caching**: Redis cluster for high availability
- **CDN**: Static asset delivery optimization

## Maintenance Procedures

### Regular Maintenance
```bash
# Weekly maintenance script
#!/bin/bash

# Update container images
docker-compose pull

# Clean up old containers and images
docker system prune -f

# Rotate logs
find ./logs -name "*.log" -mtime +7 -delete

# Database maintenance
docker-compose exec postgres vacuumdb -U postgres --all --analyze

# Health check
curl -f http://localhost:8000/health || exit 1
```

### Version Updates
```bash
# Update procedure
git pull origin main
docker-compose build --no-cache
docker-compose down
docker-compose up -d

# Verify update
docker-compose ps
curl http://localhost:8000/health
```

## Support and Documentation

### Additional Resources
- **API Documentation**: http://localhost:8000/docs
- **System Architecture**: See `ARCHITECTURE.md`
- **Development Guide**: See `DEVELOPMENT.md`
- **Security Guide**: See `SECURITY.md`

### Getting Help
- **Issues**: Create GitHub issue with logs and configuration
- **Performance**: Include system metrics and profiling data
- **Security**: Report security issues privately

---

**Last Updated**: January 13, 2026  
**Version**: 1.0.0  
**Maintainer**: System Development Team