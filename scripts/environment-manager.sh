#!/bin/bash
set -euo pipefail

# PRSM Environment Management System
# Comprehensive environment configuration and promotion pipeline

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENTS_DIR="${PROJECT_ROOT}/deploy/environments"
LOG_FILE="${PROJECT_ROOT}/logs/environment-$(date +%Y%m%d-%H%M%S).log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
OPERATION="status"
SOURCE_ENV=""
TARGET_ENV=""
CONFIG_TEMPLATE=""
FORCE=false
DRY_RUN=false
VERBOSE=false
VALIDATE_ONLY=false

# Supported environments
SUPPORTED_ENVIRONMENTS=("development" "staging" "production")

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPERATION] [OPTIONS]

PRSM Environment Management System

OPERATIONS:
    create ENV                Create a new environment configuration
    clone SOURCE TARGET       Clone configuration from source to target environment
    promote SOURCE TARGET     Promote application from source to target environment
    validate ENV              Validate environment configuration
    deploy ENV                Deploy to specific environment
    status [ENV]              Show environment status (all if ENV not specified)
    diff ENV1 ENV2            Compare configurations between environments
    backup ENV                Backup environment configuration
    restore ENV BACKUP_ID     Restore environment from backup
    secrets ENV               Manage environment secrets
    switch ENV                Switch kubectl context to environment

OPTIONS:
    -t, --template TEMPLATE   Configuration template to use for new environment
    -f, --force               Force operation without confirmation prompts
    -d, --dry-run             Show what would be done without making changes
    -v, --verbose             Enable verbose output
    --validate-only           Only validate, don't deploy
    -h, --help                Show this help message

SUPPORTED ENVIRONMENTS:
    development, staging, production

EXAMPLES:
    $0 create staging --template base
    $0 promote staging production
    $0 validate production
    $0 status
    $0 diff staging production
    $0 secrets production

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
    if [[ "$VERBOSE" == "true" ]]; then
        log "INFO" "$message"
    fi
}

# Function to handle errors
error_exit() {
    print_status "$RED" "ERROR: $1"
    exit 1
}

# Function to validate environment name
validate_environment() {
    local env=$1
    
    if [[ ! " ${SUPPORTED_ENVIRONMENTS[*]} " =~ " $env " ]]; then
        error_exit "Unsupported environment: $env. Supported: ${SUPPORTED_ENVIRONMENTS[*]}"
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "$BLUE" "Checking environment management prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "kustomize" "helm" "jq" "yq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "Required tool '$tool' is not installed"
        fi
    done
    
    # Create environments directory if it doesn't exist
    mkdir -p "$ENVIRONMENTS_DIR"
    
    print_status "$GREEN" "Prerequisites check passed"
}

# Function to create environment structure
create_environment_structure() {
    local env=$1
    local template=${2:-"base"}
    
    print_status "$BLUE" "Creating environment structure for: $env"
    
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    if [[ -d "$env_dir" && "$FORCE" != "true" ]]; then
        error_exit "Environment directory already exists: $env_dir. Use --force to overwrite."
    fi
    
    # Create directory structure
    mkdir -p "$env_dir"/{configs,secrets,manifests,scripts,backups}
    
    # Create base configuration files
    create_base_config "$env" "$template"
    create_secrets_template "$env"
    create_deployment_config "$env"
    create_environment_scripts "$env"
    
    print_status "$GREEN" "Environment structure created: $env_dir"
}

# Function to create base configuration
create_base_config() {
    local env=$1
    local template=$2
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    # Create environment-specific configuration
    cat > "$env_dir/configs/environment.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prsm-environment-config
  namespace: prsm-$env
  labels:
    app.kubernetes.io/name: prsm
    app.kubernetes.io/environment: $env
data:
  ENVIRONMENT: "$env"
  LOG_LEVEL: "$(get_log_level "$env")"
  DEBUG_MODE: "$(get_debug_mode "$env")"
  API_WORKERS: "$(get_worker_count "$env")"
  DATABASE_POOL_SIZE: "$(get_db_pool_size "$env")"
  REDIS_MAX_CONNECTIONS: "$(get_redis_connections "$env")"
  CACHE_TTL_SECONDS: "$(get_cache_ttl "$env")"
  RATE_LIMIT_REQUESTS: "$(get_rate_limit "$env")"
  MONITORING_ENABLED: "$(get_monitoring_enabled "$env")"
  TRACING_SAMPLE_RATE: "$(get_tracing_rate "$env")"
  BACKUP_RETENTION_DAYS: "$(get_backup_retention "$env")"
EOF

    # Create resource limits configuration
    cat > "$env_dir/configs/resources.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prsm-resource-limits
  namespace: prsm-$env
data:
  api_cpu_request: "$(get_api_cpu_request "$env")"
  api_cpu_limit: "$(get_api_cpu_limit "$env")"
  api_memory_request: "$(get_api_memory_request "$env")"
  api_memory_limit: "$(get_api_memory_limit "$env")"
  worker_cpu_request: "$(get_worker_cpu_request "$env")"
  worker_cpu_limit: "$(get_worker_cpu_limit "$env")"
  worker_memory_request: "$(get_worker_memory_request "$env")"
  worker_memory_limit: "$(get_worker_memory_limit "$env")"
  storage_size: "$(get_storage_size "$env")"
EOF

    # Create scaling configuration
    cat > "$env_dir/configs/scaling.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prsm-scaling-config
  namespace: prsm-$env
data:
  api_min_replicas: "$(get_api_min_replicas "$env")"
  api_max_replicas: "$(get_api_max_replicas "$env")"
  worker_min_replicas: "$(get_worker_min_replicas "$env")"
  worker_max_replicas: "$(get_worker_max_replicas "$env")"
  cpu_target_utilization: "$(get_cpu_target "$env")"
  memory_target_utilization: "$(get_memory_target "$env")"
  scale_up_stabilization: "$(get_scale_up_window "$env")"
  scale_down_stabilization: "$(get_scale_down_window "$env")"
EOF
}

# Function to create secrets template
create_secrets_template() {
    local env=$1
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    # Create secrets template (values should be replaced with actual secrets)
    cat > "$env_dir/secrets/secrets.env.template" << EOF
# PRSM Secrets Configuration for $env environment
# WARNING: This is a template file. Replace all placeholder values with actual secrets.
# Never commit actual secrets to version control.

# Database Configuration
DATABASE_URL=postgresql://username:password@postgres-service:5432/prsm_$env
DATABASE_PASSWORD=REPLACE_WITH_SECURE_PASSWORD
POSTGRES_USER=prsm_user
POSTGRES_PASSWORD=REPLACE_WITH_SECURE_PASSWORD
POSTGRES_DB=prsm_$env

# Redis Configuration
REDIS_URL=redis://redis-service:6379/0
REDIS_PASSWORD=REPLACE_WITH_SECURE_PASSWORD

# JWT Configuration
JWT_SECRET_KEY=REPLACE_WITH_SECURE_JWT_SECRET
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API Keys for External Services
OPENAI_API_KEY=REPLACE_WITH_OPENAI_KEY
ANTHROPIC_API_KEY=REPLACE_WITH_ANTHROPIC_KEY
HUGGINGFACE_API_KEY=REPLACE_WITH_HUGGINGFACE_KEY

# Web3 Configuration
WEB3_PRIVATE_KEY=REPLACE_WITH_WEB3_PRIVATE_KEY
POLYGON_RPC_URL=REPLACE_WITH_POLYGON_RPC_URL
FTNS_CONTRACT_ADDRESS=REPLACE_WITH_CONTRACT_ADDRESS

# IPFS Configuration
IPFS_API_URL=http://ipfs-service:5001
IPFS_GATEWAY_URL=http://ipfs-service:8080

# Monitoring and Observability
PROMETHEUS_URL=http://prometheus-service:9090
GRAFANA_ADMIN_PASSWORD=REPLACE_WITH_GRAFANA_PASSWORD
JAEGER_ENDPOINT=http://jaeger-service:14268/api/traces

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=REPLACE_WITH_EMAIL
SMTP_PASSWORD=REPLACE_WITH_EMAIL_PASSWORD
NOTIFICATION_EMAIL=admin@prsm.org

# Security Configuration
ENCRYPTION_KEY=REPLACE_WITH_32_BYTE_ENCRYPTION_KEY
HASH_SALT=REPLACE_WITH_SECURE_SALT

# Storage Configuration
S3_ACCESS_KEY_ID=REPLACE_WITH_S3_ACCESS_KEY
S3_SECRET_ACCESS_KEY=REPLACE_WITH_S3_SECRET_KEY
S3_BUCKET_NAME=prsm-$env-storage
S3_REGION=us-east-1
EOF

    # Create instructions for secrets management
    cat > "$env_dir/secrets/README.md" << EOF
# Secrets Management for $env Environment

## Overview
This directory contains templates and instructions for managing secrets in the $env environment.

## Files
- \`secrets.env.template\`: Template file with placeholder values
- \`secrets.env\`: Actual secrets file (should be created from template)
- \`vault-config.yaml\`: HashiCorp Vault configuration
- \`external-secrets.yaml\`: Kubernetes External Secrets configuration

## Setup Instructions

1. Copy the template file:
   \`\`\`bash
   cp secrets.env.template secrets.env
   \`\`\`

2. Replace all placeholder values in \`secrets.env\` with actual secrets

3. Create Kubernetes secret:
   \`\`\`bash
   kubectl create secret generic prsm-secrets \
     --from-env-file=secrets.env \
     --namespace=prsm-$env
   \`\`\`

4. Verify secret creation:
   \`\`\`bash
   kubectl get secrets -n prsm-$env
   \`\`\`

## Security Best Practices

- Never commit \`secrets.env\` to version control
- Use strong, unique passwords for each environment
- Rotate secrets regularly
- Use HashiCorp Vault or cloud-native secret management when possible
- Limit access to secrets using RBAC
- Monitor secret access and usage

## External Secret Management

For production environments, consider using:
- HashiCorp Vault
- AWS Secrets Manager
- Google Secret Manager
- Azure Key Vault
- Kubernetes External Secrets Operator
EOF
}

# Function to create deployment configuration
create_deployment_config() {
    local env=$1
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    # Create kustomization for the environment
    cat > "$env_dir/manifests/kustomization.yaml" << EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: prsm-$env

resources:
  - ../../../deploy/kubernetes/base
  - namespace.yaml
  - ingress.yaml

patches:
  - path: patches.yaml
    target:
      kind: Deployment
      name: prsm-api
  - path: patches.yaml
    target:
      kind: Deployment
      name: prsm-worker

images:
  - name: prsm-api
    newTag: $(get_image_tag "$env")
  - name: prsm-worker
    newTag: $(get_image_tag "$env")

replicas:
  - name: prsm-api
    count: $(get_api_min_replicas "$env")
  - name: prsm-worker
    count: $(get_worker_min_replicas "$env")

configMapGenerator:
  - name: prsm-config
    files:
      - ../configs/environment.yaml
      - ../configs/resources.yaml
      - ../configs/scaling.yaml
    behavior: replace

secretGenerator:
  - name: prsm-secrets
    env: ../secrets/secrets.env
    type: Opaque
    behavior: replace

commonLabels:
  environment: $env
  app.kubernetes.io/instance: prsm-$env

commonAnnotations:
  deployment.kubernetes.io/environment: $env
  deployment.kubernetes.io/managed-by: environment-manager
EOF

    # Create namespace definition
    cat > "$env_dir/manifests/namespace.yaml" << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: prsm-$env
  labels:
    app.kubernetes.io/name: prsm
    app.kubernetes.io/environment: $env
    pod-security.kubernetes.io/enforce: $(get_pod_security_level "$env")
  annotations:
    description: "PRSM $env environment"
    contact: "team@prsm.org"
    environment: "$env"
EOF

    # Create ingress configuration
    cat > "$env_dir/manifests/ingress.yaml" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prsm-ingress
  namespace: prsm-$env
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: $(get_cert_issuer "$env")
    nginx.ingress.kubernetes.io/ssl-redirect: "$(get_ssl_redirect "$env")"
    nginx.ingress.kubernetes.io/rate-limit: "$(get_rate_limit "$env")"
    nginx.ingress.kubernetes.io/cors-allow-origin: "$(get_cors_origin "$env")"
spec:
  tls:
  - hosts:
    - $(get_domain "$env")
    secretName: prsm-$env-tls
  rules:
  - host: $(get_domain "$env")
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prsm-api-service
            port:
              number: 8000
EOF

    # Create patches for environment-specific modifications
    cat > "$env_dir/manifests/patches.yaml" << EOF
# Environment-specific patches for $env
- op: replace
  path: /spec/template/spec/containers/0/resources/requests/memory
  value: "$(get_api_memory_request "$env")"
- op: replace
  path: /spec/template/spec/containers/0/resources/limits/memory
  value: "$(get_api_memory_limit "$env")"
- op: replace
  path: /spec/template/spec/containers/0/resources/requests/cpu
  value: "$(get_api_cpu_request "$env")"
- op: replace
  path: /spec/template/spec/containers/0/resources/limits/cpu
  value: "$(get_api_cpu_limit "$env")"
EOF
}

# Function to create environment-specific scripts
create_environment_scripts() {
    local env=$1
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    # Create deployment script for the environment
    cat > "$env_dir/scripts/deploy.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# Environment-specific deployment script
ENV="__ENV__"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$ENV_DIR")")")" 

echo "Deploying PRSM to $ENV environment..."

# Switch to environment context
kubectl config use-context "prsm-$ENV" 2>/dev/null || echo "Warning: Context prsm-$ENV not found"

# Apply manifests
cd "$ENV_DIR/manifests"
kustomize build . | kubectl apply -f -

# Wait for deployment
kubectl rollout status deployment/prsm-api -n "prsm-$ENV" --timeout=300s
kubectl rollout status deployment/prsm-worker -n "prsm-$ENV" --timeout=300s

echo "Deployment to $ENV completed successfully!"
EOF

    # Replace placeholder with actual environment
    sed -i.bak "s/__ENV__/$env/g" "$env_dir/scripts/deploy.sh" && rm "$env_dir/scripts/deploy.sh.bak"
    chmod +x "$env_dir/scripts/deploy.sh"
    
    # Create health check script
    cat > "$env_dir/scripts/health-check.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# Environment health check script
ENV="__ENV__"
NAMESPACE="prsm-$ENV"

echo "Checking health of $ENV environment..."

# Check pod status
echo "Pod Status:"
kubectl get pods -n "$NAMESPACE"

# Check service endpoints
echo "\nService Status:"
kubectl get services -n "$NAMESPACE"

# Check ingress
echo "\nIngress Status:"
kubectl get ingress -n "$NAMESPACE"

# Test API health endpoint
echo "\nAPI Health Check:"
if kubectl get service prsm-api-service -n "$NAMESPACE" &>/dev/null; then
    kubectl port-forward -n "$NAMESPACE" service/prsm-api-service 8080:8000 &
    PF_PID=$!
    sleep 2
    
    if curl -f http://localhost:8080/health &>/dev/null; then
        echo "✅ API health check passed"
    else
        echo "❌ API health check failed"
    fi
    
    kill $PF_PID 2>/dev/null || true
else
    echo "⚠️  API service not found"
fi

echo "\nHealth check completed for $ENV environment"
EOF

    # Replace placeholder with actual environment
    sed -i.bak "s/__ENV__/$env/g" "$env_dir/scripts/health-check.sh" && rm "$env_dir/scripts/health-check.sh.bak"
    chmod +x "$env_dir/scripts/health-check.sh"
    
    # Create rollback script
    cat > "$env_dir/scripts/rollback.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# Environment rollback script
ENV="__ENV__"
NAMESPACE="prsm-$ENV"

echo "Rolling back deployment in $ENV environment..."

# Rollback API deployment
if kubectl get deployment prsm-api -n "$NAMESPACE" &>/dev/null; then
    echo "Rolling back prsm-api deployment..."
    kubectl rollout undo deployment/prsm-api -n "$NAMESPACE"
    kubectl rollout status deployment/prsm-api -n "$NAMESPACE" --timeout=300s
fi

# Rollback worker deployment
if kubectl get deployment prsm-worker -n "$NAMESPACE" &>/dev/null; then
    echo "Rolling back prsm-worker deployment..."
    kubectl rollout undo deployment/prsm-worker -n "$NAMESPACE"
    kubectl rollout status deployment/prsm-worker -n "$NAMESPACE" --timeout=300s
fi

echo "Rollback completed for $ENV environment"
EOF

    # Replace placeholder with actual environment
    sed -i.bak "s/__ENV__/$env/g" "$env_dir/scripts/rollback.sh" && rm "$env_dir/scripts/rollback.sh.bak"
    chmod +x "$env_dir/scripts/rollback.sh"
}

# Environment-specific configuration functions
get_log_level() {
    case $1 in
        development) echo "DEBUG" ;;
        staging) echo "INFO" ;;
        production) echo "WARNING" ;;
        *) echo "INFO" ;;
    esac
}

get_debug_mode() {
    case $1 in
        development) echo "true" ;;
        *) echo "false" ;;
    esac
}

get_worker_count() {
    case $1 in
        development) echo "2" ;;
        staging) echo "4" ;;
        production) echo "8" ;;
        *) echo "4" ;;
    esac
}

get_db_pool_size() {
    case $1 in
        development) echo "5" ;;
        staging) echo "10" ;;
        production) echo "20" ;;
        *) echo "10" ;;
    esac
}

get_redis_connections() {
    case $1 in
        development) echo "10" ;;
        staging) echo "50" ;;
        production) echo "100" ;;
        *) echo "50" ;;
    esac
}

get_cache_ttl() {
    case $1 in
        development) echo "60" ;;
        staging) echo "300" ;;
        production) echo "600" ;;
        *) echo "300" ;;
    esac
}

get_rate_limit() {
    case $1 in
        development) echo "1000" ;;
        staging) echo "500" ;;
        production) echo "100" ;;
        *) echo "500" ;;
    esac
}

get_monitoring_enabled() {
    case $1 in
        development) echo "false" ;;
        *) echo "true" ;;
    esac
}

get_tracing_rate() {
    case $1 in
        development) echo "1.0" ;;
        staging) echo "0.1" ;;
        production) echo "0.01" ;;
        *) echo "0.1" ;;
    esac
}

get_backup_retention() {
    case $1 in
        development) echo "7" ;;
        staging) echo "14" ;;
        production) echo "30" ;;
        *) echo "14" ;;
    esac
}

get_api_cpu_request() {
    case $1 in
        development) echo "100m" ;;
        staging) echo "250m" ;;
        production) echo "500m" ;;
        *) echo "250m" ;;
    esac
}

get_api_cpu_limit() {
    case $1 in
        development) echo "500m" ;;
        staging) echo "1000m" ;;
        production) echo "2000m" ;;
        *) echo "1000m" ;;
    esac
}

get_api_memory_request() {
    case $1 in
        development) echo "256Mi" ;;
        staging) echo "512Mi" ;;
        production) echo "1Gi" ;;
        *) echo "512Mi" ;;
    esac
}

get_api_memory_limit() {
    case $1 in
        development) echo "1Gi" ;;
        staging) echo "2Gi" ;;
        production) echo "4Gi" ;;
        *) echo "2Gi" ;;
    esac
}

get_worker_cpu_request() {
    case $1 in
        development) echo "200m" ;;
        staging) echo "500m" ;;
        production) echo "1000m" ;;
        *) echo "500m" ;;
    esac
}

get_worker_cpu_limit() {
    case $1 in
        development) echo "1000m" ;;
        staging) echo "2000m" ;;
        production) echo "4000m" ;;
        *) echo "2000m" ;;
    esac
}

get_worker_memory_request() {
    case $1 in
        development) echo "512Mi" ;;
        staging) echo "1Gi" ;;
        production) echo "2Gi" ;;
        *) echo "1Gi" ;;
    esac
}

get_worker_memory_limit() {
    case $1 in
        development) echo "2Gi" ;;
        staging) echo "4Gi" ;;
        production) echo "8Gi" ;;
        *) echo "4Gi" ;;
    esac
}

get_storage_size() {
    case $1 in
        development) echo "5Gi" ;;
        staging) echo "20Gi" ;;
        production) echo "100Gi" ;;
        *) echo "20Gi" ;;
    esac
}

get_api_min_replicas() {
    case $1 in
        development) echo "1" ;;
        staging) echo "2" ;;
        production) echo "3" ;;
        *) echo "2" ;;
    esac
}

get_api_max_replicas() {
    case $1 in
        development) echo "3" ;;
        staging) echo "10" ;;
        production) echo "20" ;;
        *) echo "10" ;;
    esac
}

get_worker_min_replicas() {
    case $1 in
        development) echo "1" ;;
        staging) echo "1" ;;
        production) echo "2" ;;
        *) echo "1" ;;
    esac
}

get_worker_max_replicas() {
    case $1 in
        development) echo "2" ;;
        staging) echo "5" ;;
        production) echo "10" ;;
        *) echo "5" ;;
    esac
}

get_cpu_target() {
    case $1 in
        development) echo "80" ;;
        staging) echo "70" ;;
        production) echo "60" ;;
        *) echo "70" ;;
    esac
}

get_memory_target() {
    case $1 in
        development) echo "85" ;;
        staging) echo "80" ;;
        production) echo "75" ;;
        *) echo "80" ;;
    esac
}

get_scale_up_window() {
    case $1 in
        development) echo "60" ;;
        staging) echo "120" ;;
        production) echo "180" ;;
        *) echo "120" ;;
    esac
}

get_scale_down_window() {
    case $1 in
        development) echo "60" ;;
        staging) echo "300" ;;
        production) echo "600" ;;
        *) echo "300" ;;
    esac
}

get_image_tag() {
    case $1 in
        development) echo "dev-latest" ;;
        staging) echo "staging-latest" ;;
        production) echo "v1.0.0" ;;
        *) echo "latest" ;;
    esac
}

get_pod_security_level() {
    case $1 in
        development) echo "baseline" ;;
        staging) echo "baseline" ;;
        production) echo "restricted" ;;
        *) echo "baseline" ;;
    esac
}

get_cert_issuer() {
    case $1 in
        development) echo "letsencrypt-staging" ;;
        staging) echo "letsencrypt-staging" ;;
        production) echo "letsencrypt-prod" ;;
        *) echo "letsencrypt-staging" ;;
    esac
}

get_ssl_redirect() {
    case $1 in
        development) echo "false" ;;
        *) echo "true" ;;
    esac
}

get_cors_origin() {
    case $1 in
        development) echo "*" ;;
        staging) echo "https://staging.prsm.org" ;;
        production) echo "https://prsm.org" ;;
        *) echo "https://staging.prsm.org" ;;
    esac
}

get_domain() {
    case $1 in
        development) echo "dev.prsm.local" ;;
        staging) echo "staging.prsm.org" ;;
        production) echo "prsm.org" ;;
        *) echo "staging.prsm.org" ;;
    esac
}

# Function to validate environment configuration
validate_environment_config() {
    local env=$1
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    print_status "$BLUE" "Validating configuration for environment: $env"
    
    if [[ ! -d "$env_dir" ]]; then
        error_exit "Environment directory not found: $env_dir"
    fi
    
    local validation_errors=0
    
    # Check required files
    local required_files=(
        "configs/environment.yaml"
        "configs/resources.yaml"
        "configs/scaling.yaml"
        "manifests/kustomization.yaml"
        "manifests/namespace.yaml"
        "manifests/ingress.yaml"
        "secrets/secrets.env.template"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$env_dir/$file" ]]; then
            print_status "$RED" "Missing required file: $file"
            ((validation_errors++))
        fi
    done
    
    # Validate Kubernetes manifests
    if [[ -d "$env_dir/manifests" ]]; then
        print_status "$BLUE" "Validating Kubernetes manifests..."
        cd "$env_dir/manifests"
        
        if kustomize build . > /tmp/manifests.yaml; then
            if kubectl apply --dry-run=client -f /tmp/manifests.yaml &> /dev/null; then
                print_status "$GREEN" "Kubernetes manifests validation passed"
            else
                print_status "$RED" "Kubernetes manifests validation failed"
                ((validation_errors++))
            fi
        else
            print_status "$RED" "Failed to build manifests with kustomize"
            ((validation_errors++))
        fi
        
        rm -f /tmp/manifests.yaml
    fi
    
    # Check if secrets file exists (not template)
    if [[ ! -f "$env_dir/secrets/secrets.env" ]]; then
        print_status "$YELLOW" "Warning: secrets.env file not found. Please create from template."
    fi
    
    if [[ $validation_errors -eq 0 ]]; then
        print_status "$GREEN" "Environment validation passed: $env"
        return 0
    else
        print_status "$RED" "Environment validation failed: $validation_errors errors found"
        return 1
    fi
}

# Function to show environment status
show_environment_status() {
    local env=${1:-""}
    
    if [[ -n "$env" ]]; then
        validate_environment "$env"
        show_single_environment_status "$env"
    else
        show_all_environments_status
    fi
}

# Function to show single environment status
show_single_environment_status() {
    local env=$1
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    print_status "$BLUE" "Environment Status: $env"
    echo "======================================"
    
    if [[ ! -d "$env_dir" ]]; then
        print_status "$RED" "Environment not configured"
        return 1
    fi
    
    # Show configuration status
    echo "Configuration Status:"
    if [[ -f "$env_dir/configs/environment.yaml" ]]; then
        echo "  ✅ Environment config: Present"
    else
        echo "  ❌ Environment config: Missing"
    fi
    
    if [[ -f "$env_dir/secrets/secrets.env" ]]; then
        echo "  ✅ Secrets: Configured"
    else
        echo "  ⚠️  Secrets: Not configured (template available)"
    fi
    
    # Show Kubernetes status
    echo "\nKubernetes Status:"
    local namespace="prsm-$env"
    
    if kubectl get namespace "$namespace" &> /dev/null; then
        echo "  ✅ Namespace: $namespace exists"
        
        # Show deployment status
        local deployments
        deployments=$(kubectl get deployments -n "$namespace" --no-headers 2>/dev/null | wc -l || echo "0")
        echo "  📦 Deployments: $deployments"
        
        # Show pod status
        local pods_running
        pods_running=$(kubectl get pods -n "$namespace" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l || echo "0")
        local pods_total
        pods_total=$(kubectl get pods -n "$namespace" --no-headers 2>/dev/null | wc -l || echo "0")
        echo "  🚀 Pods: $pods_running/$pods_total running"
        
        # Show service status
        local services
        services=$(kubectl get services -n "$namespace" --no-headers 2>/dev/null | wc -l || echo "0")
        echo "  🌐 Services: $services"
        
        # Show ingress status
        if kubectl get ingress -n "$namespace" &> /dev/null; then
            local ingress_hosts
            ingress_hosts=$(kubectl get ingress -n "$namespace" -o jsonpath='{.items[*].spec.rules[*].host}' 2>/dev/null || echo "")
            if [[ -n "$ingress_hosts" ]]; then
                echo "  🌍 Ingress: $ingress_hosts"
            else
                echo "  🌍 Ingress: Configured (no hosts)"
            fi
        else
            echo "  ❌ Ingress: Not found"
        fi
    else
        echo "  ❌ Namespace: $namespace not found"
    fi
    
    echo "======================================"
}

# Function to show all environments status
show_all_environments_status() {
    print_status "$BLUE" "All Environments Status"
    echo "========================================"
    
    printf "%-15s %-12s %-10s %-8s %s\n" "ENVIRONMENT" "CONFIG" "SECRETS" "DEPLOYED" "DOMAIN"
    printf "%-15s %-12s %-10s %-8s %s\n" "---------------" "------------" "----------" "--------" "------------------"
    
    for env in "${SUPPORTED_ENVIRONMENTS[@]}"; do
        local env_dir="$ENVIRONMENTS_DIR/$env"
        local config_status="❌"
        local secrets_status="❌"
        local deployed_status="❌"
        local domain="-"
        
        if [[ -d "$env_dir" ]]; then
            config_status="✅"
            
            if [[ -f "$env_dir/secrets/secrets.env" ]]; then
                secrets_status="✅"
            fi
            
            domain=$(get_domain "$env")
        fi
        
        if kubectl get namespace "prsm-$env" &> /dev/null; then
            deployed_status="✅"
        fi
        
        printf "%-15s %-12s %-10s %-8s %s\n" "$env" "$config_status" "$secrets_status" "$deployed_status" "$domain"
    done
    
    echo "========================================"
}

# Function to compare environments
compare_environments() {
    local env1=$1
    local env2=$2
    
    validate_environment "$env1"
    validate_environment "$env2"
    
    print_status "$BLUE" "Comparing environments: $env1 vs $env2"
    
    local env1_dir="$ENVIRONMENTS_DIR/$env1"
    local env2_dir="$ENVIRONMENTS_DIR/$env2"
    
    if [[ ! -d "$env1_dir" ]]; then
        error_exit "Environment $env1 not found"
    fi
    
    if [[ ! -d "$env2_dir" ]]; then
        error_exit "Environment $env2 not found"
    fi
    
    echo "Configuration Differences:"
    echo "============================"
    
    # Compare configuration files
    local config_files=("configs/environment.yaml" "configs/resources.yaml" "configs/scaling.yaml")
    
    for config_file in "${config_files[@]}"; do
        if [[ -f "$env1_dir/$config_file" && -f "$env2_dir/$config_file" ]]; then
            echo "\n📄 $config_file:"
            if diff -u "$env1_dir/$config_file" "$env2_dir/$config_file" | head -20; then
                echo "  ✅ No differences"
            fi
        elif [[ -f "$env1_dir/$config_file" ]]; then
            echo "\n📄 $config_file: Only in $env1"
        elif [[ -f "$env2_dir/$config_file" ]]; then
            echo "\n📄 $config_file: Only in $env2"
        fi
    done
    
    # Compare manifests
    echo "\n🚀 Kubernetes Manifests:"
    if [[ -d "$env1_dir/manifests" && -d "$env2_dir/manifests" ]]; then
        local temp1="/tmp/manifests_${env1}.yaml"
        local temp2="/tmp/manifests_${env2}.yaml"
        
        (cd "$env1_dir/manifests" && kustomize build . > "$temp1") || true
        (cd "$env2_dir/manifests" && kustomize build . > "$temp2") || true
        
        if [[ -f "$temp1" && -f "$temp2" ]]; then
            if diff -u "$temp1" "$temp2" | head -30; then
                echo "  ✅ No differences in generated manifests"
            fi
        fi
        
        rm -f "$temp1" "$temp2"
    fi
    
    echo "\n============================"
}

# Function to deploy environment
deploy_environment() {
    local env=$1
    
    validate_environment "$env"
    
    local env_dir="$ENVIRONMENTS_DIR/$env"
    
    if [[ ! -d "$env_dir" ]]; then
        error_exit "Environment $env not configured. Run: $0 create $env"
    fi
    
    # Validate configuration before deployment
    if ! validate_environment_config "$env"; then
        if [[ "$FORCE" != "true" ]]; then
            error_exit "Environment validation failed. Use --force to deploy anyway."
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "$YELLOW" "DRY RUN: Would deploy to $env environment"
        return 0
    fi
    
    print_status "$BLUE" "Deploying to $env environment..."
    
    # Check if secrets file exists
    if [[ ! -f "$env_dir/secrets/secrets.env" ]]; then
        error_exit "Secrets file not found: $env_dir/secrets/secrets.env. Please create from template."
    fi
    
    # Execute environment-specific deploy script if it exists
    if [[ -f "$env_dir/scripts/deploy.sh" ]]; then
        print_status "$BLUE" "Running environment-specific deployment script..."
        bash "$env_dir/scripts/deploy.sh"
    else
        # Fallback to direct kustomize deployment
        print_status "$BLUE" "Deploying using kustomize..."
        cd "$env_dir/manifests"
        kustomize build . | kubectl apply -f -
        
        # Wait for deployments
        kubectl rollout status deployment/prsm-api -n "prsm-$env" --timeout=300s || true
        kubectl rollout status deployment/prsm-worker -n "prsm-$env" --timeout=300s || true
    fi
    
    print_status "$GREEN" "Deployment to $env completed!"
    
    # Run health check
    if [[ -f "$env_dir/scripts/health-check.sh" ]]; then
        print_status "$BLUE" "Running health check..."
        bash "$env_dir/scripts/health-check.sh"
    fi
}

# Main function
main() {
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    case "$OPERATION" in
        create)
            if [[ -z "$TARGET_ENV" ]]; then
                error_exit "Environment name required for create operation"
            fi
            validate_environment "$TARGET_ENV"
            check_prerequisites
            create_environment_structure "$TARGET_ENV" "$CONFIG_TEMPLATE"
            ;;
        validate)
            if [[ -z "$TARGET_ENV" ]]; then
                error_exit "Environment name required for validate operation"
            fi
            validate_environment "$TARGET_ENV"
            validate_environment_config "$TARGET_ENV"
            ;;
        deploy)
            if [[ -z "$TARGET_ENV" ]]; then
                error_exit "Environment name required for deploy operation"
            fi
            check_prerequisites
            deploy_environment "$TARGET_ENV"
            ;;
        status)
            show_environment_status "$TARGET_ENV"
            ;;
        diff)
            if [[ -z "$SOURCE_ENV" || -z "$TARGET_ENV" ]]; then
                error_exit "Two environment names required for diff operation"
            fi
            compare_environments "$SOURCE_ENV" "$TARGET_ENV"
            ;;
        *)
            error_exit "Unknown operation: $OPERATION"
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        create|clone|promote|validate|deploy|status|diff|backup|restore|secrets|switch)
            OPERATION="$1"
            shift
            ;;
        -t|--template)
            CONFIG_TEMPLATE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            error_exit "Unknown option: $1"
            ;;
        *)
            # Positional arguments
            if [[ -z "$SOURCE_ENV" ]]; then
                SOURCE_ENV="$1"
                TARGET_ENV="$1"  # For single-env operations
            elif [[ -z "$TARGET_ENV" ]]; then
                TARGET_ENV="$1"
            fi
            shift
            ;;
    esac
done

# Run main function
main "$@"