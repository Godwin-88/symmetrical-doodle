#!/bin/bash
# Google Drive Knowledge Base Ingestion - Container Build Script

set -e

# Configuration
IMAGE_NAME="knowledge-ingestion"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_TARGET="${BUILD_TARGET:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build the Google Drive Knowledge Base Ingestion container.

OPTIONS:
    -t, --tag TAG           Set the image tag (default: latest)
    -r, --registry URL      Set the Docker registry URL
    -e, --env ENV           Set the target environment (development|production)
    --no-cache              Build without using cache
    --push                  Push the image to registry after building
    -h, --help              Show this help message

EXAMPLES:
    $0                                          # Build with default settings
    $0 -t v1.0.0 -e production                # Build production image with tag v1.0.0
    $0 -t dev -e development --no-cache       # Build development image without cache
    $0 -t v1.0.0 --push -r myregistry.com    # Build and push to registry

ENVIRONMENT VARIABLES:
    IMAGE_TAG               Default image tag (default: latest)
    BUILD_TARGET            Build target (development|production, default: production)
    DOCKER_REGISTRY         Docker registry URL
EOF
}

# Parse command line arguments
PUSH_IMAGE=false
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -e|--env)
            case $2 in
                development|dev)
                    BUILD_TARGET="production"  # Always use production target, but with dev config
                    ;;
                production|prod)
                    BUILD_TARGET="production"
                    ;;
                *)
                    log_error "Invalid environment: $2. Use 'development' or 'production'"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Construct full image name
if [[ -n "$DOCKER_REGISTRY" ]]; then
    FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
fi

# Pre-build validation
log_info "Starting container build process..."
log_info "Image name: $FULL_IMAGE_NAME"
log_info "Build target: $BUILD_TARGET"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    log_error "Dockerfile not found in current directory"
    exit 1
fi

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    log_error "requirements.txt not found in current directory"
    exit 1
fi

# Build the Docker image
log_info "Building Docker image..."
BUILD_START_TIME=$(date +%s)

if docker build \
    --target "$BUILD_TARGET" \
    --tag "$FULL_IMAGE_NAME" \
    --label "build.timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --label "build.version=$IMAGE_TAG" \
    --label "build.target=$BUILD_TARGET" \
    $NO_CACHE \
    .; then
    
    BUILD_END_TIME=$(date +%s)
    BUILD_DURATION=$((BUILD_END_TIME - BUILD_START_TIME))
    log_success "Docker image built successfully in ${BUILD_DURATION}s"
    log_success "Image: $FULL_IMAGE_NAME"
else
    log_error "Docker build failed"
    exit 1
fi

# Show image information
log_info "Image information:"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Push to registry if requested
if [[ "$PUSH_IMAGE" == true ]]; then
    if [[ -z "$DOCKER_REGISTRY" ]]; then
        log_error "Cannot push: no registry specified. Use -r/--registry option."
        exit 1
    fi
    
    log_info "Pushing image to registry..."
    if docker push "$FULL_IMAGE_NAME"; then
        log_success "Image pushed successfully to $DOCKER_REGISTRY"
    else
        log_error "Failed to push image to registry"
        exit 1
    fi
fi

# Provide usage examples
log_info "Build completed successfully!"
echo
log_info "To run the container:"
echo "  docker run --rm -it $FULL_IMAGE_NAME"
echo
log_info "To run with Docker Compose:"
echo "  docker-compose up knowledge-ingestion"
echo
log_info "To run with custom environment:"
echo "  docker run --rm -it -e KNOWLEDGE_INGESTION_ENV=development $FULL_IMAGE_NAME"