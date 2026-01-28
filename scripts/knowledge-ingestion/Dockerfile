# Google Drive Knowledge Base Ingestion Container
# Multi-stage build for optimized production image

# Build stage for Python dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    KNOWLEDGE_INGESTION_ENV=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directories
RUN mkdir -p /app/scripts/knowledge-ingestion && \
    mkdir -p /app/data/credentials && \
    mkdir -p /app/data/extracted_pdfs && \
    mkdir -p /app/data/logs && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app/scripts/knowledge-ingestion

# Copy application code
COPY --chown=appuser:appuser . .

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Environment validation
echo "Starting Google Drive Knowledge Base Ingestion Container"
echo "Environment: ${KNOWLEDGE_INGESTION_ENV:-development}"
echo "Python version: $(python --version)"

# Validate required environment variables
required_vars=(
    "SUPABASE_URL"
    "SUPABASE_KEY"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: Missing required environment variables:"
    printf '%s\n' "${missing_vars[@]}"
    exit 1
fi

# Validate credentials directory
if [[ ! -d "/app/data/credentials" ]]; then
    echo "ERROR: Credentials directory not mounted at /app/data/credentials"
    exit 1
fi

# Initialize logging directory
mkdir -p /app/data/logs
chmod 755 /app/data/logs

# Run the application
exec "$@"
EOF

RUN chmod +x /app/entrypoint.sh && chown appuser:appuser /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python", "main.py", "--help"]

# Labels for metadata
LABEL maintainer="Algorithmic Trading Platform" \
      version="1.0.0" \
      description="Google Drive Knowledge Base Ingestion System" \
      component="knowledge-ingestion"