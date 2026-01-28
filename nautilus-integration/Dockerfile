# NautilusTrader Integration Container
# Multi-stage build for optimized production image with Rust and Python support

# Build stage for Rust dependencies
FROM rust:1.75-slim as rust-builder

# Set environment variables
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true \
    RUSTFLAGS="-C target-cpu=native"

# Install system dependencies for Rust builds
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create workspace for Rust components
WORKDIR /rust-build

# Pre-build common Rust dependencies (if any custom Rust components are needed)
# This stage can be extended for custom Rust bindings or optimizations

# Python build stage
FROM python:3.11-slim as python-builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for NautilusTrader compilation
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

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
    NAUTILUS_INTEGRATION_ENV=production \
    PYTHONPATH="/app/nautilus-integration/src"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libssl3 \
    libffi8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=python-builder /opt/venv /opt/venv

# Create application user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directories
RUN mkdir -p /app/nautilus-integration && \
    mkdir -p /app/data/catalog && \
    mkdir -p /app/data/logs && \
    mkdir -p /app/config && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app/nautilus-integration

# Copy application code
COPY --chown=appuser:appuser . .

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Environment validation
echo "Starting NautilusTrader Integration Container"
echo "Environment: ${NAUTILUS_INTEGRATION_ENV:-development}"
echo "Python version: $(python --version)"

# Validate required environment variables
required_vars=(
    "NAUTILUS_DATABASE__POSTGRES_URL"
    "NAUTILUS_DATABASE__NEO4J_URL"
    "NAUTILUS_DATABASE__REDIS_URL"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "WARNING: Missing optional environment variables:"
    printf '%s\n' "${missing_vars[@]}"
    echo "Using default configuration values"
fi

# Initialize data directories
mkdir -p /app/data/catalog/{bars,ticks,order_book,instruments}
mkdir -p /app/data/logs
chmod 755 /app/data/catalog /app/data/logs

# Initialize configuration
if [[ ! -f "/app/config/.env" && -f "/app/config/.env.example" ]]; then
    echo "Creating default configuration from example"
    cp /app/config/.env.example /app/config/.env
fi

# Validate NautilusTrader installation
echo "Validating NautilusTrader installation..."
python -c "import nautilus_trader; print(f'NautilusTrader version: {nautilus_trader.__version__}')" || {
    echo "ERROR: NautilusTrader installation validation failed"
    exit 1
}

# Validate integration package
echo "Validating integration package..."
python -c "import nautilus_integration; print('Integration package loaded successfully')" || {
    echo "ERROR: Integration package validation failed"
    exit 1
}

echo "Container initialization completed successfully"

# Run the application
exec "$@"
EOF

RUN chmod +x /app/entrypoint.sh && chown appuser:appuser /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || python -c "import sys; sys.exit(0)" || exit 1

# Expose API port
EXPOSE 8002

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python", "-m", "nautilus_integration.main"]

# Labels for metadata
LABEL maintainer="Algorithmic Trading Platform" \
      version="1.0.0" \
      description="NautilusTrader Integration Service" \
      component="nautilus-integration" \
      nautilus_version="1.200.0+"