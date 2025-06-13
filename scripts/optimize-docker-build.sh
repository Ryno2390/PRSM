#!/bin/bash

# PRSM Docker Build Optimization Script
# Optimizes Docker build process for faster builds and smaller images

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
REGISTRY=${REGISTRY:-"ghcr.io/prsm-ai"}
IMAGE_NAME=${IMAGE_NAME:-"prsm"}
VERSION=${VERSION:-"latest"}
CACHE_FROM=${CACHE_FROM:-"${REGISTRY}/${IMAGE_NAME}:cache"}

# Functions
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

check_requirements() {
    log_info "Checking Docker requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is required but not available"
        exit 1
    fi
    
    # Check Docker version
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' | cut -d. -f1,2)
    MIN_VERSION="20.10"
    
    if [ "$(printf '%s\n' "$MIN_VERSION" "$DOCKER_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
        log_error "Docker version $DOCKER_VERSION is too old. Minimum required: $MIN_VERSION"
        exit 1
    fi
    
    log_success "Docker requirements satisfied"
}

setup_buildx() {
    log_info "Setting up Docker Buildx..."
    
    # Create buildx builder if it doesn't exist
    if ! docker buildx inspect prsm-builder &> /dev/null; then
        log_info "Creating new buildx builder: prsm-builder"
        docker buildx create --name prsm-builder --use --driver docker-container \
            --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=50000000 \
            --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=10000000
    else
        log_info "Using existing buildx builder: prsm-builder"
        docker buildx use prsm-builder
    fi
    
    # Bootstrap the builder
    docker buildx inspect --bootstrap
    
    log_success "Buildx setup complete"
}

optimize_build_context() {
    log_info "Optimizing build context..."
    
    # Calculate build context size
    CONTEXT_SIZE=$(du -sh . | cut -f1)
    log_info "Current build context size: $CONTEXT_SIZE"
    
    # Check for large files that should be excluded
    log_info "Checking for large files in build context..."
    find . -type f -size +50M -not -path './.git/*' -not -path './venv/*' -not -path './dev-data/*' | while read -r file; do
        log_warning "Large file found: $file ($(du -h "$file" | cut -f1))"
    done
    
    # Validate .dockerignore
    if [ ! -f .dockerignore ]; then
        log_warning ".dockerignore not found - build context may be larger than necessary"
    else
        log_success ".dockerignore found"
    fi
}

build_image() {
    local target=$1
    local output_type=${2:-"docker"}
    
    log_info "Building PRSM image (target: $target, output: $output_type)..."
    
    # Build arguments
    BUILD_ARGS=(
        --target "$target"
        --build-arg "BUILD_ENV=$target"
        --build-arg "PRSM_VERSION=$VERSION"
        --build-arg "BUILDKIT_INLINE_CACHE=1"
        --cache-from "type=registry,ref=$CACHE_FROM"
        --cache-to "type=registry,ref=$CACHE_FROM,mode=max"
        --platform "linux/amd64,linux/arm64"
        --progress "plain"
    )
    
    # Add output configuration
    if [ "$output_type" = "registry" ]; then
        BUILD_ARGS+=(--push --tag "$REGISTRY/$IMAGE_NAME:$target-$VERSION")
    else
        BUILD_ARGS+=(--load --tag "$IMAGE_NAME:$target-$VERSION")
    fi
    
    # Add metadata
    BUILD_ARGS+=(
        --label "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        --label "org.opencontainers.image.version=$VERSION"
        --label "org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    )
    
    # Execute build
    DOCKER_BUILDKIT=1 docker buildx build "${BUILD_ARGS[@]}" .
    
    if [ $? -eq 0 ]; then
        log_success "$target image built successfully"
    else
        log_error "Failed to build $target image"
        return 1
    fi
}

analyze_image() {
    local image_tag=$1
    
    log_info "Analyzing image: $image_tag"
    
    # Image size
    IMAGE_SIZE=$(docker images --format "table {{.Size}}" "$image_tag" | tail -n 1)
    log_info "Image size: $IMAGE_SIZE"
    
    # Layer analysis
    log_info "Layer breakdown:"
    docker history --human --format "table {{.CreatedBy}}\t{{.Size}}" "$image_tag" | head -10
    
    # Security scan (if available)
    if command -v docker scan &> /dev/null; then
        log_info "Running security scan..."
        docker scan "$image_tag" || log_warning "Security scan failed or not available"
    fi
}

cleanup() {
    log_info "Cleaning up build artifacts..."
    
    # Remove dangling images
    docker image prune -f
    
    # Clean build cache (keep recent)
    docker buildx prune --keep-storage 10GB -f
    
    log_success "Cleanup complete"
}

main() {
    log_info "Starting PRSM Docker build optimization"
    
    # Parse command line arguments
    TARGET=${1:-"production"}
    OUTPUT=${2:-"docker"}
    
    case $TARGET in
        production|development|all)
            ;;
        *)
            log_error "Invalid target: $TARGET. Valid options: production, development, all"
            exit 1
            ;;
    esac
    
    # Run optimization steps
    check_requirements
    setup_buildx
    optimize_build_context
    
    # Build requested targets
    if [ "$TARGET" = "all" ]; then
        build_image "production" "$OUTPUT"
        build_image "development" "$OUTPUT"
        
        if [ "$OUTPUT" = "docker" ]; then
            analyze_image "$IMAGE_NAME:production-$VERSION"
            analyze_image "$IMAGE_NAME:development-$VERSION"
        fi
    else
        build_image "$TARGET" "$OUTPUT"
        
        if [ "$OUTPUT" = "docker" ]; then
            analyze_image "$IMAGE_NAME:$TARGET-$VERSION"
        fi
    fi
    
    cleanup
    
    log_success "Docker build optimization complete!"
    
    # Print usage instructions
    echo ""
    log_info "Usage instructions:"
    echo "  Local development: docker-compose -f docker-compose.yml -f docker-compose.dev.yml up"
    echo "  Production: docker-compose -f docker-compose.yml -f docker-compose.performance.yml up"
    echo "  Custom image: docker run -p 8000:8000 $IMAGE_NAME:$TARGET-$VERSION"
}

# Handle script interruption
trap cleanup EXIT INT TERM

# Run main function
main "$@"