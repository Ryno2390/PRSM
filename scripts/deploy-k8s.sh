#!/bin/bash
set -euo pipefail

# PRSM Kubernetes Deployment Script
# Production-grade automated deployment with comprehensive health checks

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/deployment-$(date +%Y%m%d-%H%M%S).log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENVIRONMENT="staging"
NAMESPACE="prsm-system"
CONTEXT=""
DRY_RUN=false
SKIP_TESTS=false
FORCE_DEPLOY=false
BACKUP_BEFORE_DEPLOY=true
WAIT_TIMEOUT=600
IMAGE_TAG="latest"
HELM_CHART_VERSION=""

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

PRSM Kubernetes Deployment Script

OPTIONS:
    -e, --environment ENV     Target environment (staging|production) [default: staging]
    -n, --namespace NAMESPACE Kubernetes namespace [default: prsm-system]
    -c, --context CONTEXT     Kubernetes context to use
    -t, --tag TAG             Docker image tag [default: latest]
    -v, --version VERSION     Helm chart version
    --dry-run                 Perform a dry run without making changes
    --skip-tests              Skip deployment validation tests
    --force                   Force deployment even if health checks fail
    --no-backup               Skip pre-deployment backup
    --timeout SECONDS         Wait timeout for deployment [default: 600]
    -h, --help                Show this help message

EXAMPLES:
    $0 --environment staging --tag v1.2.3
    $0 --environment production --context prod-cluster --version 1.0.0
    $0 --dry-run --environment staging

EOF
}

# Function to log messages
log() {
    local level=$1
    shift
    local message="$*"
    echo "[$TIMESTAMP] [$level] $message" | tee -a "$LOG_FILE"
}

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%H:%M:%S')] $message${NC}"
    log "INFO" "$message"
}

# Function to handle errors
error_exit() {
    print_status "$RED" "ERROR: $1"
    exit 1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "$BLUE" "Checking deployment prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "kustomize" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "Required tool '$tool' is not installed"
        fi
    done
    
    # Check Kubernetes connectivity
    if [[ -n "$CONTEXT" ]]; then
        kubectl config use-context "$CONTEXT" || error_exit "Failed to switch to context: $CONTEXT"
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check namespace exists or create it
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_status "$YELLOW" "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE" || error_exit "Failed to create namespace"
    fi
    
    # Verify Docker images exist
    local images=("prsm-api:$IMAGE_TAG" "prsm-worker:$IMAGE_TAG")
    for image in "${images[@]}"; do
        if ! docker manifest inspect "$image" &> /dev/null; then
            print_status "$YELLOW" "Warning: Image $image not found locally, assuming it exists in registry"
        fi
    done
    
    print_status "$GREEN" "Prerequisites check passed"
}

# Function to backup existing deployment
backup_deployment() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == "false" ]]; then
        print_status "$YELLOW" "Skipping pre-deployment backup"
        return 0
    fi
    
    print_status "$BLUE" "Creating pre-deployment backup..."
    
    local backup_dir="${PROJECT_ROOT}/backups/pre-deployment-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Kubernetes resources
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/kubernetes-resources.yaml" 2>/dev/null || true
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml" 2>/dev/null || true
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml" 2>/dev/null || true
    kubectl get pvc -n "$NAMESPACE" -o yaml > "$backup_dir/persistent-volumes.yaml" 2>/dev/null || true
    
    # Backup database if PostgreSQL is running
    if kubectl get deployment postgres -n "$NAMESPACE" &> /dev/null; then
        print_status "$BLUE" "Creating database backup..."
        kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_dumpall -U postgres > "$backup_dir/database-backup.sql" 2>/dev/null || {
            print_status "$YELLOW" "Warning: Database backup failed, continuing with deployment"
        }
    fi
    
    print_status "$GREEN" "Backup created in: $backup_dir"
}

# Function to prepare deployment manifests
prepare_manifests() {
    print_status "$BLUE" "Preparing deployment manifests..."
    
    local overlay_dir="${PROJECT_ROOT}/deploy/kubernetes/overlays/$ENVIRONMENT"
    local temp_dir="$(mktemp -d)"
    
    # Check if environment overlay exists
    if [[ ! -d "$overlay_dir" ]]; then
        error_exit "Environment overlay not found: $overlay_dir"
    fi
    
    # Generate manifests using kustomize
    cd "$overlay_dir"
    kustomize build . > "$temp_dir/manifests.yaml" || error_exit "Failed to build manifests with kustomize"
    
    # Replace image tags if specified
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        sed -i.bak "s/:latest/:$IMAGE_TAG/g" "$temp_dir/manifests.yaml"
    fi
    
    echo "$temp_dir/manifests.yaml"
}

# Function to validate manifests
validate_manifests() {
    local manifest_file=$1
    print_status "$BLUE" "Validating Kubernetes manifests..."
    
    # Dry run validation
    kubectl apply --dry-run=client -f "$manifest_file" || error_exit "Manifest validation failed"
    
    # Server-side validation
    kubectl apply --dry-run=server -f "$manifest_file" || error_exit "Server-side validation failed"
    
    print_status "$GREEN" "Manifest validation passed"
}

# Function to apply manifests
apply_manifests() {
    local manifest_file=$1
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "$YELLOW" "DRY RUN: Would apply manifests"
        kubectl apply --dry-run=client -f "$manifest_file"
        return 0
    fi
    
    print_status "$BLUE" "Applying Kubernetes manifests..."
    
    # Apply with server-side apply for better conflict handling
    kubectl apply --server-side=true --force-conflicts -f "$manifest_file" || error_exit "Failed to apply manifests"
    
    print_status "$GREEN" "Manifests applied successfully"
}

# Function to wait for deployment readiness
wait_for_deployment() {
    print_status "$BLUE" "Waiting for deployment to be ready..."
    
    local deployments=("prsm-api" "prsm-worker" "postgres" "redis" "ipfs")
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            print_status "$BLUE" "Waiting for deployment: $deployment"
            kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s" || {
                if [[ "$FORCE_DEPLOY" == "true" ]]; then
                    print_status "$YELLOW" "Warning: Deployment $deployment not ready, but continuing due to --force"
                else
                    error_exit "Deployment $deployment failed to become ready"
                fi
            }
        fi
    done
    
    print_status "$GREEN" "All deployments are ready"
}

# Function to run health checks
run_health_checks() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_status "$YELLOW" "Skipping health checks"
        return 0
    fi
    
    print_status "$BLUE" "Running post-deployment health checks..."
    
    # Check API health
    local api_service="prsm-api-service"
    if kubectl get service "$api_service" -n "$NAMESPACE" &> /dev/null; then
        print_status "$BLUE" "Checking API health..."
        
        # Port forward for health check
        kubectl port-forward -n "$NAMESPACE" service/"$api_service" 8080:8000 &
        local port_forward_pid=$!
        sleep 5
        
        # Health check with timeout
        if timeout 30 curl -f http://localhost:8080/health &> /dev/null; then
            print_status "$GREEN" "API health check passed"
        else
            kill $port_forward_pid 2>/dev/null || true
            if [[ "$FORCE_DEPLOY" == "true" ]]; then
                print_status "$YELLOW" "Warning: API health check failed, but continuing due to --force"
            else
                error_exit "API health check failed"
            fi
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    # Check database connectivity
    if kubectl get deployment postgres -n "$NAMESPACE" &> /dev/null; then
        print_status "$BLUE" "Checking database connectivity..."
        kubectl exec -n "$NAMESPACE" deployment/postgres -- psql -U postgres -c "SELECT version();" &> /dev/null || {
            if [[ "$FORCE_DEPLOY" == "true" ]]; then
                print_status "$YELLOW" "Warning: Database connectivity check failed, but continuing due to --force"
            else
                error_exit "Database connectivity check failed"
            fi
        }
        print_status "$GREEN" "Database connectivity check passed"
    fi
    
    # Check Redis connectivity
    if kubectl get deployment redis -n "$NAMESPACE" &> /dev/null; then
        print_status "$BLUE" "Checking Redis connectivity..."
        kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli ping &> /dev/null || {
            if [[ "$FORCE_DEPLOY" == "true" ]]; then
                print_status "$YELLOW" "Warning: Redis connectivity check failed, but continuing due to --force"
            else
                error_exit "Redis connectivity check failed"
            fi
        }
        print_status "$GREEN" "Redis connectivity check passed"
    fi
    
    print_status "$GREEN" "All health checks passed"
}

# Function to display deployment summary
display_summary() {
    print_status "$GREEN" "Deployment Summary"
    echo "==========================================="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Image Tag: $IMAGE_TAG"
    echo "Deployment Time: $(date)"
    echo "==========================================="
    
    print_status "$BLUE" "Deployment Resources:"
    kubectl get all -n "$NAMESPACE" --show-labels
    
    print_status "$BLUE" "Service Endpoints:"
    kubectl get ingress -n "$NAMESPACE" -o wide 2>/dev/null || echo "No ingress resources found"
    
    print_status "$GREEN" "Deployment completed successfully!"
}

# Function to handle rollback
rollback_deployment() {
    print_status "$YELLOW" "Rolling back deployment..."
    
    local deployments=("prsm-api" "prsm-worker")
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE" || {
                print_status "$YELLOW" "Warning: Failed to rollback deployment: $deployment"
            }
        fi
    done
    
    print_status "$GREEN" "Rollback completed"
}

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        print_status "$RED" "Deployment failed with exit code: $exit_code"
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            rollback_deployment
        fi
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -c|--context)
            CONTEXT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -v|--version)
            HELM_CHART_VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        --no-backup)
            BACKUP_BEFORE_DEPLOY=false
            shift
            ;;
        --timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    error_exit "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
 fi

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

# Main deployment flow
main() {
    print_status "$BLUE" "Starting PRSM Kubernetes deployment..."
    print_status "$BLUE" "Environment: $ENVIRONMENT"
    print_status "$BLUE" "Namespace: $NAMESPACE"
    print_status "$BLUE" "Image Tag: $IMAGE_TAG"
    
    # Deployment steps
    check_prerequisites
    backup_deployment
    
    local manifest_file
    manifest_file=$(prepare_manifests)
    
    validate_manifests "$manifest_file"
    apply_manifests "$manifest_file"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        wait_for_deployment
        run_health_checks
        display_summary
    fi
    
    print_status "$GREEN" "PRSM deployment completed successfully!"
}

# Run main function
main "$@"