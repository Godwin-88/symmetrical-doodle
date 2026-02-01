# MLflow Setup Guide

This guide provides instructions for setting up MLflow for the algorithmic trading system's ML operations.

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Network access to ports 5000, 5433, 9000, 9001

## Quick Start

### 1. Start MLflow Infrastructure

```bash
# Navigate to project root
cd symmetrical-doodle

# Start all MLflow services
docker-compose -f docker-compose.mlflow.yml up -d

# Verify services are running
docker-compose -f docker-compose.mlflow.yml ps
```

### 2. Access Web Interfaces

| Service | URL | Description |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | Experiment tracking and model registry |
| MinIO Console | http://localhost:9001 | Artifact storage management |

**Default MinIO credentials:**
- Username: `minioadmin`
- Password: `minioadmin`

### 3. Configure Intelligence Layer

Set environment variables for the Python intelligence layer:

```bash
# Add to .env file
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

### 4. Verify Connection

```python
from intelligence_layer.mlflow_service import get_mlflow_service

service = get_mlflow_service()
if service._initialized:
    print("MLflow connection successful!")
    experiments = service.list_experiments()
    print(f"Found {len(experiments)} experiments")
```

## Architecture

### Services

```
+-------------------+     +-------------------+     +-------------------+
|   MLflow Server   |---->|    PostgreSQL     |     |      MinIO        |
|   Port: 5000      |     |    Port: 5433     |     |  Port: 9000,9001  |
+-------------------+     +-------------------+     +-------------------+
         |                        |                        |
         |                        |                        |
         +------------------------+------------------------+
                                  |
                          mlflow-network
```

### Data Persistence

| Service | Volume | Purpose |
|---------|--------|---------|
| PostgreSQL | `mlflow_db_data` | Experiment tracking, model registry metadata |
| MinIO | `mlflow_minio_data` | Model artifacts, plots, configurations |

## Configuration Options

### Docker Compose Environment Variables

#### MLflow Server

```yaml
environment:
  MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow_password@mlflow-db:5432/mlflow
  MLFLOW_ARTIFACTS_DESTINATION: s3://mlflow-artifacts
  AWS_ACCESS_KEY_ID: minioadmin
  AWS_SECRET_ACCESS_KEY: minioadmin
  MLFLOW_S3_ENDPOINT_URL: http://mlflow-minio:9000
```

#### PostgreSQL

```yaml
environment:
  POSTGRES_USER: mlflow
  POSTGRES_PASSWORD: mlflow_password
  POSTGRES_DB: mlflow
```

#### MinIO

```yaml
environment:
  MINIO_ROOT_USER: minioadmin
  MINIO_ROOT_PASSWORD: minioadmin
```

### Production Configuration

For production deployments, update the following:

1. **Strong Passwords**: Change default passwords
2. **External PostgreSQL**: Use managed database service
3. **External S3**: Use AWS S3 or managed MinIO
4. **TLS/SSL**: Enable HTTPS for all services
5. **Authentication**: Enable MLflow authentication

Example production configuration:

```yaml
# docker-compose.prod.yml
services:
  mlflow-server:
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/mlflow
      MLFLOW_ARTIFACTS_DESTINATION: s3://${S3_BUCKET}
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
      AWS_DEFAULT_REGION: ${AWS_REGION}
```

## Operations

### Starting Services

```bash
# Start all services
docker-compose -f docker-compose.mlflow.yml up -d

# Start specific service
docker-compose -f docker-compose.mlflow.yml up -d mlflow-server
```

### Stopping Services

```bash
# Stop all services
docker-compose -f docker-compose.mlflow.yml down

# Stop and remove volumes (data loss!)
docker-compose -f docker-compose.mlflow.yml down -v
```

### Viewing Logs

```bash
# All services
docker-compose -f docker-compose.mlflow.yml logs -f

# Specific service
docker-compose -f docker-compose.mlflow.yml logs -f mlflow-server
```

### Health Checks

```bash
# Check MLflow server
curl http://localhost:5000/health

# Check MinIO
curl http://localhost:9000/minio/health/live

# Check PostgreSQL
docker exec mlflow-postgres pg_isready -U mlflow -d mlflow
```

### Backup and Recovery

#### Backup PostgreSQL

```bash
# Backup database
docker exec mlflow-postgres pg_dump -U mlflow mlflow > mlflow_backup.sql

# Restore database
cat mlflow_backup.sql | docker exec -i mlflow-postgres psql -U mlflow mlflow
```

#### Backup MinIO Artifacts

```bash
# Install mc (MinIO Client)
brew install minio/stable/mc  # macOS
# or
wget https://dl.min.io/client/mc/release/linux-amd64/mc  # Linux

# Configure mc
mc alias set local http://localhost:9000 minioadmin minioadmin

# Backup artifacts
mc mirror local/mlflow-artifacts ./mlflow-artifacts-backup

# Restore artifacts
mc mirror ./mlflow-artifacts-backup local/mlflow-artifacts
```

## Integration

### Python Client

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
mlflow.create_experiment("my-experiment")

# Start run
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.pytorch.log_model(model, "model")
```

### Intelligence Layer Integration

```python
from intelligence_layer.mlflow_service import MLflowService, MLflowConfig

# Create service with custom config
config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    default_experiment="financial-models"
)
service = MLflowService(config)
service.initialize()

# Use context manager for runs
with service.run_context("financial-models", "my-run") as run_id:
    service.log_params({"lr": 0.001})
    service.log_metrics({"loss": 0.05})
```

## Troubleshooting

### Common Issues

#### MLflow Server Won't Start

```bash
# Check logs
docker-compose -f docker-compose.mlflow.yml logs mlflow-server

# Common causes:
# - PostgreSQL not ready: wait for health check
# - MinIO bucket not created: check mlflow-minio-init logs
# - Port conflict: check if 5000 is in use
```

#### Cannot Connect to MLflow

```bash
# Verify network
docker network ls | grep mlflow

# Check server is listening
netstat -an | grep 5000

# Test connection
curl -v http://localhost:5000/health
```

#### Artifact Upload Fails

```bash
# Check MinIO logs
docker-compose -f docker-compose.mlflow.yml logs mlflow-minio

# Verify bucket exists
mc ls local/mlflow-artifacts

# Check credentials
# Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY match MinIO config
```

#### Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose -f docker-compose.mlflow.yml logs mlflow-db

# Test connection
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT 1"

# Check connection string
# postgresql://mlflow:mlflow_password@mlflow-db:5432/mlflow
```

### Reset Everything

```bash
# Stop and remove all containers and volumes
docker-compose -f docker-compose.mlflow.yml down -v

# Remove any leftover networks
docker network rm mlflow-network

# Start fresh
docker-compose -f docker-compose.mlflow.yml up -d
```

## Security Considerations

### Production Checklist

- [ ] Change all default passwords
- [ ] Enable TLS/SSL for all services
- [ ] Configure network isolation
- [ ] Enable MLflow authentication
- [ ] Set up regular backups
- [ ] Configure log rotation
- [ ] Enable audit logging
- [ ] Set resource limits

### MLflow Authentication

MLflow supports basic authentication via the `mlflow-auth` plugin:

```bash
# Install auth plugin
pip install mlflow[auth]

# Configure authentication
mlflow server \
  --backend-store-uri postgresql://... \
  --app-name basic-auth
```

---

**Last Updated**: February 2026
