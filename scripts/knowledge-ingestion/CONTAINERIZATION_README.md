# Google Drive Knowledge Base Ingestion - Containerization Guide

This document provides comprehensive guidance for containerizing and deploying the Google Drive Knowledge Base Ingestion system.

## Overview

The system supports multiple deployment methods:
- **Local Development**: Direct Python execution with virtual environments
- **Docker**: Single container deployment with volume mounts
- **Docker Compose**: Multi-container deployment with orchestration

## Quick Start

### Local Deployment
```bash
# Install dependencies and setup
make setup-dev

# Configure environment
cp config/.env.example config/.env
# Edit config/.env with your settings

# Run ingestion
make run
```

### Docker Deployment
```bash
# Build container
make build

# Run with Docker
make docker-run
```

### Docker Compose Deployment
```bash
# Setup environment
cp .env.docker .env
# Edit .env with your settings

# Deploy
make deploy-compose
```

## Container Architecture

### Multi-Stage Dockerfile

The `Dockerfile` uses a multi-stage build approach:

1. **Builder Stage**: Installs build dependencies and Python packages
2. **Production Stage**: Creates optimized runtime image with minimal dependencies

Key features:
- Python 3.11 slim base image for reduced size
- Non-root user for security
- Health checks for container monitoring
- Proper signal handling for graceful shutdown

### Environment Configuration

The system supports flexible environment configuration:

- **Environment Variables**: Override any configuration setting
- **Config Files**: YAML-based configuration with environment-specific overrides
- **Volume Mounts**: External configuration and data persistence

## Deployment Methods

### 1. Local Development

Best for development and testing:

```bash
# Setup development environment
make setup-dev

# Run with development settings
make run-dev

# Validate configuration
make validate
```

**Pros:**
- Fast iteration and debugging
- Direct access to logs and state
- Easy configuration changes

**Cons:**
- Requires Python environment setup
- Platform-dependent dependencies

### 2. Docker Container

Best for consistent deployment across environments:

```bash
# Build production image
make build-prod

# Run with environment variables
docker run --rm -it \
  -e SUPABASE_URL=your_url \
  -e SUPABASE_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -v ./credentials:/app/data/credentials:ro \
  -v ./data/logs:/app/data/logs \
  knowledge-ingestion:latest
```

**Pros:**
- Consistent runtime environment
- Easy deployment and scaling
- Isolated dependencies

**Cons:**
- Requires Docker installation
- Volume management for persistence

### 3. Docker Compose

Best for production deployments with multiple services:

```bash
# Configure environment
cp .env.docker .env
vim .env  # Edit with your settings

# Deploy services
docker-compose up knowledge-ingestion

# Scale if needed
docker-compose up --scale knowledge-ingestion=2
```

**Pros:**
- Multi-service orchestration
- Environment management
- Service discovery and networking

**Cons:**
- More complex configuration
- Requires Docker Compose

## Configuration Management

### Environment Variables

The system supports comprehensive environment variable configuration:

```bash
# Core settings
KNOWLEDGE_INGESTION_ENV=production
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
OPENAI_API_KEY=your-openai-key

# Processing settings
MAX_CONCURRENT_DOWNLOADS=5
MAX_CONCURRENT_PROCESSING=3
CHUNK_SIZE=1000
EMBEDDING_BATCH_SIZE=32

# Google Drive settings
GOOGLE_DRIVE_CREDENTIALS_PATH=/app/data/credentials/service-account.json
GOOGLE_DRIVE_FOLDER_IDS=folder1,folder2,folder3
```

### Configuration Files

Environment-specific YAML configuration:

- `config/config.yaml` - Base configuration
- `config/config.development.yaml` - Development overrides
- `config/config.production.yaml` - Production overrides

### Volume Mounts

Required and optional volume mounts:

```yaml
volumes:
  # Required: Google Drive credentials
  - ./credentials:/app/data/credentials:ro
  
  # Optional: Configuration overrides
  - ./config:/app/scripts/knowledge-ingestion/config:ro
  
  # Optional: Persistent data
  - ./data/logs:/app/data/logs
  - ./data/extracted_pdfs:/app/data/extracted_pdfs
```

## Idempotent Execution

The system implements comprehensive state management for idempotent execution:

### State Management

- **Execution State**: Tracks overall pipeline progress
- **File State**: Individual file processing status
- **Checkpoint Recovery**: Resume from last successful operation
- **Configuration Validation**: Detect configuration changes

### Checkpoint Handling

```bash
# Resume previous execution
python run_ingestion.py --resume

# Start fresh execution
python run_ingestion.py --no-resume

# Run specific phases only
python run_ingestion.py --phases discovery download
```

### State Persistence

State is persisted in `data/state/execution_state.json`:

```json
{
  "execution_id": "exec_1234567890_abc123",
  "start_time": "2024-01-01T00:00:00Z",
  "current_phase": "embedding",
  "configuration_hash": "abc123def456",
  "files_discovered": 150,
  "files_processed": 120,
  "file_states": {
    "file_id_1": {
      "discovery_status": "completed",
      "download_status": "completed",
      "parsing_status": "completed",
      "chunking_status": "completed",
      "embedding_status": "in_progress"
    }
  }
}
```

## Deployment Validation

The system includes comprehensive deployment validation:

### Pre-deployment Validation

```bash
# Validate configuration and dependencies
make validate

# Validate specific environment
python validate_deployment.py --config-env production

# Generate validation report
python validate_deployment.py --output validation_report.json
```

### Validation Checks

- **Configuration**: Required settings and file paths
- **Dependencies**: Python packages and system requirements
- **Authentication**: Google Drive and Supabase connectivity
- **Permissions**: File and directory access
- **Containerization**: Docker setup and volume mounts

### Health Checks

Container health checks monitor:
- Python interpreter availability
- Configuration validity
- Service connectivity
- Resource utilization

## Security Considerations

### Container Security

- **Non-root User**: Runs as `appuser` with limited privileges
- **Read-only Mounts**: Credentials mounted read-only
- **Minimal Base Image**: Python slim image reduces attack surface
- **No Secrets in Image**: All secrets via environment variables or volumes

### Credential Management

- **Service Account**: Google Drive authentication via JSON key file
- **Environment Variables**: API keys via secure environment injection
- **Volume Mounts**: Credentials directory mounted from host
- **Access Control**: Proper file permissions on credential files

### Network Security

- **HTTPS Only**: All external API calls use HTTPS
- **No Exposed Ports**: Container doesn't expose network ports
- **Outbound Only**: Only outbound connections to APIs

## Performance Optimization

### Resource Allocation

```yaml
# Docker Compose resource limits
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

### Concurrency Settings

```bash
# Environment variables for performance tuning
MAX_CONCURRENT_DOWNLOADS=10
MAX_CONCURRENT_PROCESSING=5
EMBEDDING_BATCH_SIZE=64
SUPABASE_MAX_CONNECTIONS=20
```

### Caching Strategy

- **PDF Caching**: Downloaded PDFs cached in `data/extracted_pdfs/`
- **State Persistence**: Execution state for resume capability
- **Connection Pooling**: Database connection reuse

## Monitoring and Logging

### Structured Logging

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "INFO",
  "logger": "ingestion.discovery",
  "message": "Discovered 150 PDF files",
  "correlation_id": "abc123",
  "execution_id": "exec_1234567890",
  "phase": "discovery"
}
```

### Log Management

- **Log Rotation**: Automatic rotation with size limits
- **Structured Format**: JSON format for log aggregation
- **Correlation IDs**: Request tracing across components
- **Volume Persistence**: Logs persisted via volume mounts

### Metrics Collection

- **Execution Metrics**: Files processed, success rates, timing
- **Resource Metrics**: Memory usage, CPU utilization
- **Error Metrics**: Failure rates, error categories
- **Business Metrics**: Knowledge base coverage, quality scores

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Fix file permissions
   chmod 600 credentials/service-account.json
   chown -R 1000:1000 data/
   ```

2. **Configuration Errors**
   ```bash
   # Validate configuration
   make validate-config
   
   # Check environment variables
   docker run --rm knowledge-ingestion:latest env
   ```

3. **Network Connectivity**
   ```bash
   # Test API connectivity
   docker run --rm knowledge-ingestion:latest \
     python -c "import requests; print(requests.get('https://api.openai.com/v1/models').status_code)"
   ```

4. **Resource Constraints**
   ```bash
   # Monitor resource usage
   docker stats knowledge-ingestion
   
   # Adjust resource limits
   docker-compose up --scale knowledge-ingestion=1
   ```

### Debug Mode

```bash
# Run with debug logging
docker run --rm -it \
  -e LOG_LEVEL=DEBUG \
  -v ./credentials:/app/data/credentials:ro \
  knowledge-ingestion:latest

# Interactive shell for debugging
make docker-shell
```

### Log Analysis

```bash
# View recent logs
make logs

# Follow logs in real-time
docker-compose logs -f knowledge-ingestion

# Search logs for errors
grep -r "ERROR" data/logs/
```

## Best Practices

### Development Workflow

1. **Local Development**: Use `make setup-dev` for development
2. **Configuration Management**: Keep sensitive data in environment variables
3. **Testing**: Run `make test` before building containers
4. **Validation**: Always run `make validate` before deployment

### Production Deployment

1. **Environment Separation**: Use separate configurations for dev/prod
2. **Resource Planning**: Allocate appropriate CPU/memory resources
3. **Monitoring**: Implement log aggregation and alerting
4. **Backup Strategy**: Regular backup of state and configuration

### Security Practices

1. **Credential Rotation**: Regular rotation of API keys and service accounts
2. **Access Control**: Minimal required permissions for service accounts
3. **Network Security**: Use private networks where possible
4. **Image Security**: Regular base image updates and vulnerability scanning

## Migration Guide

### From Local to Container

1. **Export Configuration**:
   ```bash
   # Create container-compatible environment file
   python -c "from core.config import get_settings; import os; [print(f'{k}={v}') for k,v in os.environ.items() if k.startswith(('SUPABASE_', 'OPENAI_', 'GOOGLE_'))]" > .env
   ```

2. **Migrate Credentials**:
   ```bash
   # Copy credentials to container-accessible location
   mkdir -p credentials
   cp /path/to/service-account.json credentials/
   ```

3. **Test Migration**:
   ```bash
   # Validate container deployment
   make deploy-docker
   make validate
   ```

### From Docker to Docker Compose

1. **Create Compose Configuration**:
   ```bash
   # Use provided docker-compose.yml as template
   cp docker-compose.yml docker-compose.prod.yml
   ```

2. **Environment Configuration**:
   ```bash
   # Create production environment file
   cp .env.docker .env.production
   ```

3. **Deploy and Validate**:
   ```bash
   # Deploy with compose
   docker-compose -f docker-compose.prod.yml up
   ```

## Support and Maintenance

### Regular Maintenance

- **Log Cleanup**: Regular cleanup of old log files
- **State Cleanup**: Cleanup of old execution state files
- **Image Updates**: Regular base image and dependency updates
- **Configuration Review**: Periodic review of configuration settings

### Monitoring Checklist

- [ ] Container health status
- [ ] Resource utilization (CPU, memory, disk)
- [ ] Log file sizes and rotation
- [ ] API rate limits and quotas
- [ ] Database connection health
- [ ] Execution success rates

### Update Procedures

1. **Test Updates**: Always test in development environment first
2. **Backup State**: Backup execution state before updates
3. **Rolling Updates**: Use rolling updates for zero-downtime deployment
4. **Rollback Plan**: Have rollback procedures ready

For additional support, refer to the main README.md and system documentation.