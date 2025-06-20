#!/bin/bash
# PRSM Docker Helper Script
# Simplifies Docker operations for developers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILES=(
    "quickstart:docker-compose.quickstart.yml"
    "tutorial:docker-compose.tutorial.yml"
    "onboarding:docker-compose.onboarding.yml"
    "dev:docker-compose.dev.yml"
    "production:docker-compose.yml"
)

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

show_banner() {
    echo -e "${BLUE}"
    echo "🐳 PRSM Docker Helper"
    echo "===================="
    echo -e "${NC}"
}

show_help() {
    show_banner
    echo "Usage: $0 <command> [environment] [options]"
    echo ""
    echo "Commands:"
    echo "  start <env>     - Start Docker environment"
    echo "  stop <env>      - Stop Docker environment"
    echo "  restart <env>   - Restart Docker environment"
    echo "  status <env>    - Show environment status"
    echo "  logs <env>      - Show environment logs"
    echo "  clean <env>     - Clean environment (remove containers and volumes)"
    echo "  reset <env>     - Reset environment (clean + restart)"
    echo "  list            - List available environments"
    echo "  health          - Check Docker daemon and prerequisites"
    echo ""
    echo "Environments:"
    echo "  quickstart      - Minimal setup (30s startup)"
    echo "  tutorial        - Tutorial environment with tools"
    echo "  onboarding      - Developer onboarding setup"
    echo "  dev             - Full development environment"
    echo "  production      - Production deployment"
    echo ""
    echo "Options:"
    echo "  --profile <name> - Enable specific profile (e.g., jupyter, tools)"
    echo "  --follow         - Follow logs in real-time"
    echo "  --verbose        - Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 start quickstart"
    echo "  $0 start tutorial --profile jupyter"
    echo "  $0 logs dev --follow"
    echo "  $0 clean quickstart"
    echo ""
}

get_compose_file() {
    local env=$1
    for item in "${COMPOSE_FILES[@]}"; do
        if [[ $item =~ ^$env:(.+)$ ]]; then
            echo "${BASH_REMATCH[1]}"
            return 0
        fi
    done
    return 1
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        return 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        return 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        return 1
    fi
    
    log_success "All prerequisites met"
    return 0
}

start_environment() {
    local env=$1
    local profile=$2
    local verbose=$3
    
    local compose_file
    compose_file=$(get_compose_file "$env")
    if [[ $? -ne 0 ]]; then
        log_error "Unknown environment: $env"
        return 1
    fi
    
    log_info "Starting $env environment..."
    
    local cmd="docker-compose -f $compose_file"
    
    if [[ -n "$profile" ]]; then
        cmd="$cmd --profile $profile"
    fi
    
    cmd="$cmd up -d"
    
    if [[ "$verbose" == "true" ]]; then
        log_info "Running: $cmd"
    fi
    
    if eval "$cmd"; then
        log_success "$env environment started successfully"
        
        # Show access information
        case $env in
            "tutorial")
                echo ""
                log_info "Access URLs:"
                echo "  📚 Tutorial Dashboard: http://localhost:3000"
                echo "  🔍 Redis Commander: http://localhost:8081 (tutorial/prsm_tutorial)"
                if [[ "$profile" == "jupyter" ]]; then
                    echo "  📓 Jupyter: http://localhost:8888 (token: prsm_tutorial_jupyter)"
                fi
                if [[ "$profile" == "web-dev" ]]; then
                    echo "  💻 VS Code: http://localhost:8443 (password: prsm_tutorial_code)"
                fi
                ;;
            "onboarding")
                echo ""
                log_info "Access URLs:"
                if [[ "$profile" == "tools" ]]; then
                    echo "  🔍 Redis Commander: http://localhost:8082"
                    echo "  🗄️  SQLite Browser: http://localhost:3001"
                    echo "  📁 IPFS Web UI: http://localhost:5002"
                fi
                ;;
            "dev")
                echo ""
                log_info "Access URLs:"
                echo "  🗄️  PgAdmin: http://localhost:5050 (admin@prsm.dev/dev_password)"
                echo "  🔍 Redis Commander: http://localhost:8081"
                echo "  📓 Jupyter: http://localhost:8888 (token: prsm_dev_token)"
                ;;
            "production")
                echo ""
                log_info "Access URLs:"
                echo "  📊 Grafana: http://localhost:3000"
                echo "  📈 Prometheus: http://localhost:9090"
                echo "  🚨 Alertmanager: http://localhost:9093"
                ;;
        esac
        
        echo ""
        log_info "Next steps:"
        echo "  1. Check status: $0 status $env"
        echo "  2. Run tutorial: python examples/tutorials/hello_world_complete.py"
        echo "  3. View logs: $0 logs $env"
        
        return 0
    else
        log_error "Failed to start $env environment"
        return 1
    fi
}

stop_environment() {
    local env=$1
    
    local compose_file
    compose_file=$(get_compose_file "$env")
    if [[ $? -ne 0 ]]; then
        log_error "Unknown environment: $env"
        return 1
    fi
    
    log_info "Stopping $env environment..."
    
    if docker-compose -f "$compose_file" down; then
        log_success "$env environment stopped"
        return 0
    else
        log_error "Failed to stop $env environment"
        return 1
    fi
}

show_status() {
    local env=$1
    
    local compose_file
    compose_file=$(get_compose_file "$env")
    if [[ $? -ne 0 ]]; then
        log_error "Unknown environment: $env"
        return 1
    fi
    
    log_info "Status for $env environment:"
    echo ""
    
    docker-compose -f "$compose_file" ps --format table
    
    echo ""
    log_info "Health checks:"
    docker-compose -f "$compose_file" ps --format "table {{.Service}}\t{{.Status}}\t{{.Health}}"
}

show_logs() {
    local env=$1
    local follow=$2
    local service=$3
    
    local compose_file
    compose_file=$(get_compose_file "$env")
    if [[ $? -ne 0 ]]; then
        log_error "Unknown environment: $env"
        return 1
    fi
    
    local cmd="docker-compose -f $compose_file logs"
    
    if [[ "$follow" == "true" ]]; then
        cmd="$cmd -f"
    fi
    
    if [[ -n "$service" ]]; then
        cmd="$cmd $service"
    fi
    
    eval "$cmd"
}

clean_environment() {
    local env=$1
    
    local compose_file
    compose_file=$(get_compose_file "$env")
    if [[ $? -ne 0 ]]; then
        log_error "Unknown environment: $env"
        return 1
    fi
    
    log_warning "This will remove all containers and data for $env environment"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning $env environment..."
        
        if docker-compose -f "$compose_file" down -v --remove-orphans; then
            log_success "$env environment cleaned"
            return 0
        else
            log_error "Failed to clean $env environment"
            return 1
        fi
    else
        log_info "Cancelled"
        return 0
    fi
}

list_environments() {
    log_info "Available environments:"
    echo ""
    
    for item in "${COMPOSE_FILES[@]}"; do
        if [[ $item =~ ^([^:]+):(.+)$ ]]; then
            local env="${BASH_REMATCH[1]}"
            local file="${BASH_REMATCH[2]}"
            
            local description=""
            case $env in
                "quickstart") description="Minimal setup (30s startup)" ;;
                "tutorial") description="Tutorial environment with tools" ;;
                "onboarding") description="Developer onboarding setup" ;;
                "dev") description="Full development environment" ;;
                "production") description="Production deployment" ;;
            esac
            
            printf "  %-12s - %s\n" "$env" "$description"
        fi
    done
    
    echo ""
    log_info "Usage: $0 start <environment>"
}

# Parse command line arguments
COMMAND=""
ENVIRONMENT=""
PROFILE=""
FOLLOW=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|status|logs|clean|reset|list|health)
            COMMAND=$1
            shift
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --follow)
            FOLLOW=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$ENVIRONMENT" ]]; then
                ENVIRONMENT=$1
            fi
            shift
            ;;
    esac
done

# Main command handling
case $COMMAND in
    "start")
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for start command"
            show_help
            exit 1
        fi
        check_prerequisites || exit 1
        start_environment "$ENVIRONMENT" "$PROFILE" "$VERBOSE"
        ;;
    "stop")
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for stop command"
            exit 1
        fi
        stop_environment "$ENVIRONMENT"
        ;;
    "restart")
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for restart command"
            exit 1
        fi
        stop_environment "$ENVIRONMENT" && start_environment "$ENVIRONMENT" "$PROFILE" "$VERBOSE"
        ;;
    "status")
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for status command"
            exit 1
        fi
        show_status "$ENVIRONMENT"
        ;;
    "logs")
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for logs command"
            exit 1
        fi
        show_logs "$ENVIRONMENT" "$FOLLOW"
        ;;
    "clean")
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for clean command"
            exit 1
        fi
        clean_environment "$ENVIRONMENT"
        ;;
    "reset")
        if [[ -z "$ENVIRONMENT" ]]; then
            log_error "Environment required for reset command"
            exit 1
        fi
        clean_environment "$ENVIRONMENT" && start_environment "$ENVIRONMENT" "$PROFILE" "$VERBOSE"
        ;;
    "list")
        list_environments
        ;;
    "health")
        check_prerequisites
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac