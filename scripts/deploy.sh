#!/bin/bash
# PRSM Deployment Script
# Automated deployment for production and development environments

set -euo pipefail

# ===================================
# Configuration
# ===================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===================================
# Helper Functions
# ===================================
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# ===================================
# Deployment Functions
# ===================================
check_requirements() {
    log "Checking deployment requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        warning "Environment file not found. Copying from template..."
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        warning "Please edit .env file with your configuration before proceeding."
        return 1
    fi
    
    success "Requirements check passed"
}

pull_images() {
    log "Pulling latest Docker images..."
    cd "$PROJECT_ROOT"
    docker-compose pull
    success "Images pulled successfully"
}

build_images() {
    log "Building PRSM Docker images..."
    cd "$PROJECT_ROOT"
    docker-compose build --no-cache
    success "Images built successfully"
}

start_infrastructure() {
    local env_type="$1"
    log "Starting PRSM infrastructure (${env_type})..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$env_type" == "development" ]]; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    else
        docker-compose up -d
    fi
    
    success "Infrastructure started"
}

run_migrations() {
    log "Running database migrations..."
    cd "$PROJECT_ROOT"
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 10
    
    # Run migrations
    docker-compose exec prsm-api alembic upgrade head
    success "Database migrations completed"
}

health_check() {
    log "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            success "PRSM API is healthy"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: Waiting for API to be ready..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
}

show_status() {
    log "PRSM Infrastructure Status:"
    echo
    docker-compose ps
    echo
    log "Service URLs:"
    echo "🌐 PRSM API: http://localhost:8000"
    echo "📊 Grafana: http://localhost:3000"
    echo "🔍 Prometheus: http://localhost:9090"
    echo "🗄️  PostgreSQL: localhost:5432"
    echo "🔄 Redis: localhost:6379"
    echo "📁 IPFS Gateway: http://localhost:8080"
    echo "🧠 Weaviate: http://localhost:8080 (Vector DB)"
}

backup_data() {
    log "Creating data backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database
    docker-compose exec -T postgres pg_dump -U prsm prsm > "$backup_dir/database.sql"
    
    # Backup Redis data
    docker-compose exec -T redis redis-cli BGSAVE
    docker cp "$(docker-compose ps -q redis):/data/dump.rdb" "$backup_dir/redis.rdb"
    
    # Backup volumes
    docker run --rm -v prsm-data:/data -v "$backup_dir:/backup" alpine tar czf /backup/prsm-data.tar.gz -C /data .
    
    success "Backup created: $backup_dir"
}

# ===================================
# Main Script Logic
# ===================================
show_help() {
    echo "PRSM Deployment Script"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  prod              Deploy production environment"
    echo "  dev               Deploy development environment"
    echo "  stop              Stop all services"
    echo "  restart           Restart all services"
    echo "  logs              Show service logs"
    echo "  status            Show service status"
    echo "  backup            Create data backup"
    echo "  migrate           Run database migrations"
    echo "  build             Build Docker images"
    echo "  health            Check service health"
    echo "  clean             Clean up containers and volumes"
    echo
    echo "Options:"
    echo "  --no-build        Skip building images"
    echo "  --no-migrate      Skip database migrations"
    echo "  --help, -h        Show this help message"
}

# Parse command line arguments
COMMAND="${1:-help}"
NO_BUILD=false
NO_MIGRATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --no-migrate)
            NO_MIGRATE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "${COMMAND:-}" ]]; then
                COMMAND="$1"
            fi
            shift
            ;;
    esac
done

# Execute command
case "$COMMAND" in
    prod)
        log "🚀 Deploying PRSM Production Environment"
        check_requirements || exit 1
        
        if [[ "$NO_BUILD" == false ]]; then
            pull_images
            build_images
        fi
        
        start_infrastructure "production"
        
        if [[ "$NO_MIGRATE" == false ]]; then
            run_migrations
        fi
        
        health_check
        show_status
        success "🎉 Production deployment completed!"
        ;;
    
    dev)
        log "🛠️  Deploying PRSM Development Environment"
        check_requirements || exit 1
        
        if [[ "$NO_BUILD" == false ]]; then
            build_images
        fi
        
        start_infrastructure "development"
        
        if [[ "$NO_MIGRATE" == false ]]; then
            run_migrations
        fi
        
        health_check
        show_status
        success "🎉 Development deployment completed!"
        ;;
    
    stop)
        log "Stopping PRSM services..."
        cd "$PROJECT_ROOT"
        docker-compose down
        success "Services stopped"
        ;;
    
    restart)
        log "Restarting PRSM services..."
        cd "$PROJECT_ROOT"
        docker-compose restart
        success "Services restarted"
        ;;
    
    logs)
        cd "$PROJECT_ROOT"
        docker-compose logs -f
        ;;
    
    status)
        show_status
        ;;
    
    backup)
        backup_data
        ;;
    
    migrate)
        run_migrations
        ;;
    
    build)
        build_images
        ;;
    
    health)
        health_check
        ;;
    
    clean)
        log "Cleaning up PRSM containers and volumes..."
        cd "$PROJECT_ROOT"
        docker-compose down -v --remove-orphans
        docker system prune -f
        success "Cleanup completed"
        ;;
    
    help|*)
        show_help
        ;;
esac