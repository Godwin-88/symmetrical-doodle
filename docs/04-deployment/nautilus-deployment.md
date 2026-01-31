# NautilusTrader Integration Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the NautilusTrader integration in various environments, from development to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Deployment](#development-deployment)
3. [Staging Deployment](#staging-deployment)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Configuration Management](#configuration-management)
8. [Monitoring Setup](#monitoring-setup)
9. [Backup and Recovery](#backup-and-recovery)
10. [Maintenance Procedures](#maintenance-procedures)

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 100 Mbps
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows 10+

#### Recommended Requirements (Production)
- **CPU**: 16 cores, 3.0 GHz
- **RAM**: 32 GB
- **Storage**: 500 GB NVMe SSD
- **Network**: 1 Gbps
- **OS**: Ubuntu 22.04 LTS

### Software Dependencies

#### Core Dependencies
```bash
# Python 3.11+
python3 --version

# Node.js 18+
node --version
npm --version

# Docker & Docker Compose
docker --version
docker-compose --version

# Git
git --version
```

#### Database Systems
```bash
# PostgreSQL 14+
psql --version

# Redis 6+
redis-server --version

# Neo4j 5+ (optional)
neo4j version
```

#### NautilusTrader Platform
```bash
# Install NautilusTrader
pip install nautilus_trader

# Verify installation
python -c "import nautilus_trader; print(nautilus_trader.__version__)"
```

## Development Deployment

### Quick Setup

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd algorithmic-trading-system
   ```

2. **Setup Nautilus Integration**:
   ```bash
   cd nautilus-integration
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   # Copy configuration template
   cp config/.env.example config/.env
   
   # Edit configuration
   nano config/.env
   ```

4. **Initialize Database**:
   ```bash
   # Start database services
   docker-compose -f docker-compose.dev.yml up -d postgres redis
   
   # Initialize schema
   python scripts/setup.py --init-db
   ```

5. **Start Services**:
   ```bash
   # Start Nautilus integration
   python -m nautilus_integration.main
   
   # Start frontend (separate terminal)
   cd ../frontend
   npm install
   npm run dev
   ```

### Development Configuration

**File**: `nautilus-integration/config/.env`
```env
# Development Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database Configuration
DATABASE_URL=postgresql://nautilus:password@localhost:5432/nautilus_dev
REDIS_URL=redis://localhost:6379/0

# NautilusTrader Configuration
NAUTILUS_CONFIG_PATH=./config/nautilus_dev.json
NAUTILUS_DATA_PATH=./data/dev

# Risk Management (Relaxed for development)
MAX_POSITION_SIZE=10000
MAX_DAILY_LOSS=1000
RISK_CHECK_INTERVAL=5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000"]

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

### Development Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run UI tests
cd frontend
npm run test

# Run end-to-end tests
npm run test:e2e
```

## Staging Deployment

### Infrastructure Setup

1. **Provision Infrastructure**:
   ```bash
   # Using Terraform (example)
   cd infrastructure/staging
   terraform init
   terraform plan
   terraform apply
   ```

2. **Setup Load Balancer**:
   ```nginx
   # /etc/nginx/sites-available/nautilus-staging
   upstream nautilus_backend {
       server 127.0.0.1:8000;
       server 127.0.0.1:8001;
   }
   
   server {
       listen 80;
       server_name staging.nautilus.example.com;
       
       location / {
           proxy_pass http://nautilus_backend;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Database Setup**:
   ```bash
   # Setup PostgreSQL cluster
   sudo apt install postgresql-14-repmgr
   
   # Configure replication
   sudo -u postgres createuser --replication replicator
   sudo -u postgres createdb nautilus_staging
   ```

### Staging Configuration

**File**: `nautilus-integration/config/.env.staging`
```env
# Staging Environment
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration (with connection pooling)
DATABASE_URL=postgresql://nautilus:secure_password@db-cluster:5432/nautilus_staging?sslmode=require
REDIS_URL=redis://redis-cluster:6379/0

# NautilusTrader Configuration
NAUTILUS_CONFIG_PATH=./config/nautilus_staging.json
NAUTILUS_DATA_PATH=/data/staging

# Risk Management (Production-like)
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=5000
RISK_CHECK_INTERVAL=1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["https://staging.nautilus.example.com"]

# Security
JWT_SECRET_KEY=staging_secret_key_change_in_production
ENCRYPTION_KEY=staging_encryption_key

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_URL=https://monitoring.example.com
```

### Deployment Script

**File**: `scripts/deploy_staging.sh`
```bash
#!/bin/bash
set -e

echo "Starting staging deployment..."

# Variables
DEPLOY_DIR="/opt/nautilus-integration"
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
SERVICE_NAME="nautilus-integration"

# Create backup
echo "Creating backup..."
sudo mkdir -p $BACKUP_DIR
sudo cp -r $DEPLOY_DIR $BACKUP_DIR/

# Stop services
echo "Stopping services..."
sudo systemctl stop $SERVICE_NAME

# Update code
echo "Updating code..."
cd $DEPLOY_DIR
git fetch origin
git checkout staging
git pull origin staging

# Update dependencies
echo "Updating dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Run database migrations
echo "Running migrations..."
python scripts/migrate.py

# Update configuration
echo "Updating configuration..."
cp config/.env.staging config/.env

# Start services
echo "Starting services..."
sudo systemctl start $SERVICE_NAME

# Verify deployment
echo "Verifying deployment..."
sleep 30
curl -f http://localhost:8000/health || exit 1

echo "Staging deployment completed successfully!"
```

## Production Deployment

### Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (HAProxy)                  │
├─────────────────────────────────────────────────────────────┤
│    App Server 1    │    App Server 2    │    App Server 3  │
│   (Auto-scaling)   │   (Auto-scaling)   │   (Auto-scaling) │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL Primary │ PostgreSQL Replica │ PostgreSQL Replica│
├─────────────────────────────────────────────────────────────┤
│    Redis Cluster    │    Redis Cluster   │    Redis Cluster │
├─────────────────────────────────────────────────────────────┤
│                    Monitoring Stack                         │
│         (Prometheus, Grafana, AlertManager)                 │
└─────────────────────────────────────────────────────────────┘
```

### Production Setup

1. **Infrastructure Provisioning**:
   ```bash
   # Using Terraform for AWS
   cd infrastructure/production
   terraform init
   terraform plan -var-file="production.tfvars"
   terraform apply -var-file="production.tfvars"
   ```

2. **Database Cluster Setup**:
   ```bash
   # Setup PostgreSQL with streaming replication
   # Primary server
   sudo -u postgres initdb -D /var/lib/postgresql/data
   
   # Configure postgresql.conf
   echo "
   listen_addresses = '*'
   wal_level = replica
   max_wal_senders = 3
   wal_keep_segments = 64
   " >> /var/lib/postgresql/data/postgresql.conf
   
   # Configure pg_hba.conf
   echo "host replication replicator 10.0.0.0/8 md5" >> /var/lib/postgresql/data/pg_hba.conf
   ```

3. **Application Deployment**:
   ```bash
   # Deploy using Ansible
   ansible-playbook -i inventory/production playbooks/deploy.yml
   ```

### Production Configuration

**File**: `nautilus-integration/config/.env.production`
```env
# Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Database Configuration (High Availability)
DATABASE_URL=postgresql://nautilus:${DB_PASSWORD}@db-primary:5432/nautilus_prod?sslmode=require
DATABASE_REPLICA_URL=postgresql://nautilus:${DB_PASSWORD}@db-replica:5432/nautilus_prod?sslmode=require
REDIS_URL=redis://redis-cluster:6379/0

# NautilusTrader Configuration
NAUTILUS_CONFIG_PATH=./config/nautilus_production.json
NAUTILUS_DATA_PATH=/data/production

# Risk Management (Strict Production Limits)
MAX_POSITION_SIZE=1000000
MAX_DAILY_LOSS=50000
RISK_CHECK_INTERVAL=0.1
KILL_SWITCH_ENABLED=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["https://nautilus.example.com"]

# Security (Use environment variables)
JWT_SECRET_KEY=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
API_KEY=${API_KEY}

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_URL=https://monitoring.example.com
SENTRY_DSN=${SENTRY_DSN}

# Performance
WORKER_PROCESSES=4
MAX_CONNECTIONS=1000
CONNECTION_POOL_SIZE=20
```

### Production Deployment Script

**File**: `scripts/deploy_production.sh`
```bash
#!/bin/bash
set -e

# Production deployment with zero-downtime
echo "Starting production deployment..."

# Variables
DEPLOY_DIR="/opt/nautilus-integration"
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
SERVICE_NAME="nautilus-integration"
HEALTH_CHECK_URL="http://localhost:8000/health"

# Pre-deployment checks
echo "Running pre-deployment checks..."
python scripts/pre_deployment_checks.py || exit 1

# Create backup
echo "Creating backup..."
sudo mkdir -p $BACKUP_DIR
sudo cp -r $DEPLOY_DIR $BACKUP_DIR/
sudo pg_dump nautilus_prod > $BACKUP_DIR/database_backup.sql

# Blue-green deployment
echo "Starting blue-green deployment..."

# Deploy to inactive servers first
for server in server2 server3; do
    echo "Deploying to $server..."
    ssh $server "cd $DEPLOY_DIR && git pull && pip install -r requirements.txt"
    ssh $server "sudo systemctl restart $SERVICE_NAME"
    
    # Health check
    ssh $server "curl -f $HEALTH_CHECK_URL" || exit 1
done

# Switch load balancer to new servers
echo "Switching load balancer..."
sudo systemctl reload haproxy

# Deploy to primary server
echo "Deploying to primary server..."
cd $DEPLOY_DIR
git pull
pip install -r requirements.txt
python scripts/migrate.py
sudo systemctl restart $SERVICE_NAME

# Final health check
echo "Final health check..."
sleep 30
curl -f $HEALTH_CHECK_URL || exit 1

# Notify monitoring systems
echo "Notifying monitoring systems..."
curl -X POST https://monitoring.example.com/api/deployments \
  -H "Content-Type: application/json" \
  -d '{"service": "nautilus-integration", "version": "'$(git rev-parse HEAD)'", "status": "deployed"}'

echo "Production deployment completed successfully!"
```

## Docker Deployment

### Docker Configuration

**File**: `nautilus-integration/Dockerfile`
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 nautilus && chown -R nautilus:nautilus /app
USER nautilus

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "nautilus_integration.main"]
```

**File**: `docker-compose.yml`
```yaml
version: '3.8'

services:
  nautilus-integration:
    build: ./nautilus-integration
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://nautilus:password@postgres:5432/nautilus
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=nautilus
      - POSTGRES_USER=nautilus
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - nautilus-integration
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Docker Deployment Commands

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f nautilus-integration

# Scale services
docker-compose up -d --scale nautilus-integration=3

# Update services
docker-compose pull
docker-compose up -d

# Backup data
docker-compose exec postgres pg_dump -U nautilus nautilus > backup.sql

# Restore data
docker-compose exec -T postgres psql -U nautilus nautilus < backup.sql
```

## Kubernetes Deployment

### Kubernetes Manifests

**File**: `k8s/namespace.yaml`
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nautilus-integration
```

**File**: `k8s/configmap.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nautilus-config
  namespace: nautilus-integration
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  METRICS_ENABLED: "true"
```

**File**: `k8s/secret.yaml`
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: nautilus-secrets
  namespace: nautilus-integration
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  API_KEY: <base64-encoded-api-key>
```

**File**: `k8s/deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nautilus-integration
  namespace: nautilus-integration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nautilus-integration
  template:
    metadata:
      labels:
        app: nautilus-integration
    spec:
      containers:
      - name: nautilus-integration
        image: nautilus-integration:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: nautilus-config
        - secretRef:
            name: nautilus-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**File**: `k8s/service.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: nautilus-integration-service
  namespace: nautilus-integration
spec:
  selector:
    app: nautilus-integration
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**File**: `k8s/ingress.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nautilus-integration-ingress
  namespace: nautilus-integration
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - nautilus.example.com
    secretName: nautilus-tls
  rules:
  - host: nautilus.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nautilus-integration-service
            port:
              number: 80
```

### Kubernetes Deployment Commands

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n nautilus-integration

# View logs
kubectl logs -f deployment/nautilus-integration -n nautilus-integration

# Scale deployment
kubectl scale deployment nautilus-integration --replicas=5 -n nautilus-integration

# Update deployment
kubectl set image deployment/nautilus-integration nautilus-integration=nautilus-integration:v2.0.0 -n nautilus-integration

# Port forward for testing
kubectl port-forward service/nautilus-integration-service 8000:80 -n nautilus-integration
```

## Configuration Management

### Environment-Specific Configurations

**Development**:
- Relaxed security settings
- Verbose logging
- Mock external services
- Local database connections

**Staging**:
- Production-like security
- Moderate logging
- Real external services
- Clustered databases

**Production**:
- Strict security settings
- Minimal logging
- All external services
- High-availability setup

### Configuration Validation

**File**: `scripts/validate_config.py`
```python
#!/usr/bin/env python3
"""Configuration validation script."""

import os
import sys
from urllib.parse import urlparse

def validate_database_url(url):
    """Validate database URL format."""
    try:
        parsed = urlparse(url)
        assert parsed.scheme in ['postgresql', 'postgres']
        assert parsed.hostname
        assert parsed.port
        assert parsed.username
        assert parsed.password
        return True
    except:
        return False

def validate_config():
    """Validate all configuration settings."""
    errors = []
    
    # Required environment variables
    required_vars = [
        'DATABASE_URL',
        'REDIS_URL',
        'JWT_SECRET_KEY',
        'API_KEY'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate database URL
    db_url = os.getenv('DATABASE_URL')
    if db_url and not validate_database_url(db_url):
        errors.append("Invalid DATABASE_URL format")
    
    # Validate numeric settings
    try:
        max_position_size = int(os.getenv('MAX_POSITION_SIZE', '0'))
        if max_position_size <= 0:
            errors.append("MAX_POSITION_SIZE must be positive")
    except ValueError:
        errors.append("MAX_POSITION_SIZE must be a number")
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("Configuration validation passed!")

if __name__ == "__main__":
    validate_config()
```

## Monitoring Setup

### Prometheus Configuration

**File**: `monitoring/prometheus.yml`
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'nautilus-integration'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

**File**: `monitoring/grafana-dashboard.json`
```json
{
  "dashboard": {
    "id": null,
    "title": "Nautilus Integration Monitoring",
    "tags": ["nautilus", "trading"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Order Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(orders_processed_total[5m])",
            "legendFormat": "Orders/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Risk Check Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, risk_check_duration_seconds)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Active Positions",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_positions_total",
            "legendFormat": "Positions"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

**File**: `monitoring/rules/nautilus-alerts.yml`
```yaml
groups:
  - name: nautilus-integration
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "PostgreSQL database is not responding"

      - alert: RiskLimitBreach
        expr: current_portfolio_risk > risk_limit_threshold
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "Risk limit breached"
          description: "Portfolio risk exceeds configured limits"
```

## Backup and Recovery

### Database Backup

**File**: `scripts/backup_database.sh`
```bash
#!/bin/bash
set -e

# Configuration
BACKUP_DIR="/backups/database"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
echo "Creating database backup..."
pg_dump -h localhost -U nautilus nautilus_prod > $BACKUP_DIR/nautilus_$TIMESTAMP.sql

# Compress backup
gzip $BACKUP_DIR/nautilus_$TIMESTAMP.sql

# Upload to S3 (optional)
if [ "$UPLOAD_TO_S3" = "true" ]; then
    aws s3 cp $BACKUP_DIR/nautilus_$TIMESTAMP.sql.gz s3://nautilus-backups/database/
fi

# Clean old backups
find $BACKUP_DIR -name "nautilus_*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "Database backup completed: nautilus_$TIMESTAMP.sql.gz"
```

### Application Backup

**File**: `scripts/backup_application.sh`
```bash
#!/bin/bash
set -e

# Configuration
BACKUP_DIR="/backups/application"
APP_DIR="/opt/nautilus-integration"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Create application backup
echo "Creating application backup..."
tar -czf $BACKUP_DIR/nautilus-app_$TIMESTAMP.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs' \
    -C $APP_DIR .

# Upload to S3 (optional)
if [ "$UPLOAD_TO_S3" = "true" ]; then
    aws s3 cp $BACKUP_DIR/nautilus-app_$TIMESTAMP.tar.gz s3://nautilus-backups/application/
fi

echo "Application backup completed: nautilus-app_$TIMESTAMP.tar.gz"
```

### Recovery Procedures

**File**: `scripts/restore_database.sh`
```bash
#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

BACKUP_FILE=$1

# Verify backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Stop application
echo "Stopping application..."
systemctl stop nautilus-integration

# Drop and recreate database
echo "Recreating database..."
sudo -u postgres dropdb nautilus_prod
sudo -u postgres createdb nautilus_prod

# Restore database
echo "Restoring database..."
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | psql -h localhost -U nautilus nautilus_prod
else
    psql -h localhost -U nautilus nautilus_prod < $BACKUP_FILE
fi

# Start application
echo "Starting application..."
systemctl start nautilus-integration

# Verify restoration
sleep 30
curl -f http://localhost:8000/health || exit 1

echo "Database restoration completed successfully!"
```

## Maintenance Procedures

### Regular Maintenance Tasks

**Daily**:
```bash
# Check system health
curl -f http://localhost:8000/health

# Monitor disk space
df -h

# Check log files
tail -n 100 /var/log/nautilus-integration/error.log

# Backup database
./scripts/backup_database.sh
```

**Weekly**:
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Rotate logs
sudo logrotate -f /etc/logrotate.d/nautilus-integration

# Analyze database performance
psql -d nautilus_prod -c "SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Clean temporary files
find /tmp -name "nautilus_*" -mtime +7 -delete
```

**Monthly**:
```bash
# Update Python dependencies
pip list --outdated
pip install -r requirements.txt --upgrade

# Database maintenance
psql -d nautilus_prod -c "VACUUM ANALYZE;"
psql -d nautilus_prod -c "REINDEX DATABASE nautilus_prod;"

# Security updates
sudo apt update && sudo apt upgrade
```

### Performance Optimization

**File**: `scripts/optimize_performance.sh`
```bash
#!/bin/bash
set -e

echo "Starting performance optimization..."

# Database optimization
echo "Optimizing database..."
psql -d nautilus_prod -c "
    -- Update statistics
    ANALYZE;
    
    -- Vacuum tables
    VACUUM (ANALYZE, VERBOSE);
    
    -- Reindex if needed
    REINDEX DATABASE nautilus_prod;
"

# Clear application cache
echo "Clearing application cache..."
redis-cli FLUSHDB

# Restart services for memory cleanup
echo "Restarting services..."
systemctl restart nautilus-integration

# Verify performance
echo "Verifying performance..."
sleep 30
response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/health)
echo "Health check response time: ${response_time}s"

if (( $(echo "$response_time > 1.0" | bc -l) )); then
    echo "WARNING: Response time is high (${response_time}s)"
else
    echo "Performance optimization completed successfully!"
fi
```

### Troubleshooting Commands

```bash
# Check service status
systemctl status nautilus-integration

# View recent logs
journalctl -u nautilus-integration -n 100

# Check database connections
psql -d nautilus_prod -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor system resources
htop
iotop
nethogs

# Test API endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8000/api/status

# Check disk space
df -h
du -sh /var/log/*
du -sh /opt/nautilus-integration/*
```

---

*This deployment guide is maintained by the DevOps and Engineering teams. Last updated: January 2026*