#!/bin/bash
# =============================================================================
# PRSM Bootstrap Server GCP Deployment Script
# =============================================================================
# This script deploys the PRSM bootstrap server to Google Cloud Platform.
#
# Prerequisites:
#   - Google Cloud SDK installed and configured
#   - Docker installed
#   - jq installed
#   - Valid SSL certificates
#
# Usage:
#   ./deploy_bootstrap_gcp.sh [options]
#
# Options:
#   -r, --region       GCP region (default: us-east1)
#   -e, --environment  Environment (staging/production)
#   -d, --domain       Domain name for the bootstrap server
#   -p, --project      GCP project ID
#   -h, --help         Show this help message
#
# Environment Variables:
#   GCP_REGION         GCP region
#   GCP_PROJECT        GCP project ID
#   PRSM_DOMAIN        Domain name
#   PRSM_ENV           Environment name
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Default values
DEFAULT_REGION="us-east1"
DEFAULT_ZONE="us-east1-b"
DEFAULT_ENVIRONMENT="production"
DEFAULT_DOMAIN="prsm-network.com"
DEFAULT_MACHINE_TYPE="e2-medium"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

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

show_help() {
    cat << EOF
PRSM Bootstrap Server GCP Deployment Script

Usage: $0 [options]

Options:
    -r, --region       GCP region (default: $DEFAULT_REGION)
    -z, --zone         GCP zone (default: $DEFAULT_ZONE)
    -e, --environment  Environment: staging/production (default: $DEFAULT_ENVIRONMENT)
    -d, --domain       Domain name (default: $DEFAULT_DOMAIN)
    -p, --project      GCP project ID (required)
    -m, --machine-type GCE machine type (default: $DEFAULT_MACHINE_TYPE)
    -h, --help         Show this help message

Environment Variables:
    GCP_REGION         GCP region
    GCP_PROJECT        GCP project ID
    PRSM_DOMAIN        Domain name
    PRSM_ENV           Environment name
    POSTGRES_PASSWORD  PostgreSQL password

Examples:
    $0 --project my-project --region us-west1 --environment staging
    $0 -p my-project -r europe-west1 -e production

EOF
    exit 0
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=()
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        missing+=("gcloud")
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them before running this script."
        exit 1
    fi
    
    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 > /dev/null 2>&1; then
        log_error "GCloud not authenticated. Run 'gcloud auth login' first."
        exit 1
    fi
    
    # Check project is set
    if [ -z "${GCP_PROJECT:-}" ]; then
        log_error "GCP project ID is required. Use -p option or set GCP_PROJECT environment variable."
        exit 1
    fi
    
    # Set the project
    gcloud config set project "$GCP_PROJECT"
    
    log_success "All prerequisites met"
}

# =============================================================================
# GCP Resource Functions
# =============================================================================

enable_apis() {
    log_info "Enabling required GCP APIs..."
    
    local apis=(
        "compute.googleapis.com"
        "dns.googleapis.com"
        "container.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        gcloud services enable "$api" --project="$GCP_PROJECT" 2>/dev/null || true
    done
    
    log_success "APIs enabled"
}

create_vpc_network() {
    local network_name="prsm-bootstrap-${ENVIRONMENT}-network"
    
    log_info "Creating VPC network: $network_name"
    
    # Check if network exists
    if gcloud compute networks describe "$network_name" --project="$GCP_PROJECT" > /dev/null 2>&1; then
        log_warning "VPC network already exists: $network_name"
        echo "$network_name"
        return 0
    fi
    
    # Create VPC network
    gcloud compute networks create "$network_name" \
        --project="$GCP_PROJECT" \
        --subnet-mode=custom \
        --bgp-routing-mode=global
    
    log_success "Created VPC network: $network_name"
    echo "$network_name"
}

create_subnet() {
    local network_name=$1
    local subnet_name="prsm-bootstrap-${ENVIRONMENT}-subnet"
    local subnet_region="${GCP_REGION}"
    
    log_info "Creating subnet: $subnet_name"
    
    # Check if subnet exists
    if gcloud compute networks subnets describe "$subnet_name" \
        --region="$subnet_region" \
        --project="$GCP_PROJECT" > /dev/null 2>&1; then
        log_warning "Subnet already exists: $subnet_name"
        echo "$subnet_name"
        return 0
    fi
    
    # Create subnet
    gcloud compute networks subnets create "$subnet_name" \
        --project="$GCP_PROJECT" \
        --network="$network_name" \
        --region="$subnet_region" \
        --range=10.0.1.0/24 \
        --enable-private-ip-google-access \
        --enable-flow-logs
    
    log_success "Created subnet: $subnet_name"
    echo "$subnet_name"
}

create_firewall_rules() {
    local network_name=$1
    
    log_info "Creating firewall rules..."
    
    # Allow SSH
    gcloud compute firewall-rules create "prsm-bootstrap-${ENVIRONMENT}-allow-ssh" \
        --project="$GCP_PROJECT" \
        --network="$network_name" \
        --allow=tcp:22 \
        --source-ranges=0.0.0.0/0 \
        --target-tags="prsm-bootstrap" \
        --description="Allow SSH to bootstrap server" 2>/dev/null || true
    
    # Allow WebSocket P2P
    gcloud compute firewall-rules create "prsm-bootstrap-${ENVIRONMENT}-allow-websocket" \
        --project="$GCP_PROJECT" \
        --network="$network_name" \
        --allow=tcp:8765 \
        --source-ranges=0.0.0.0/0 \
        --target-tags="prsm-bootstrap" \
        --description="Allow WebSocket P2P connections" 2>/dev/null || true
    
    # Allow HTTP API
    gcloud compute firewall-rules create "prsm-bootstrap-${ENVIRONMENT}-allow-api" \
        --project="$GCP_PROJECT" \
        --network="$network_name" \
        --allow=tcp:8000 \
        --source-ranges=0.0.0.0/0 \
        --target-tags="prsm-bootstrap" \
        --description="Allow HTTP API access" 2>/dev/null || true
    
    # Allow HTTPS
    gcloud compute firewall-rules create "prsm-bootstrap-${ENVIRONMENT}-allow-https" \
        --project="$GCP_PROJECT" \
        --network="$network_name" \
        --allow=tcp:443 \
        --source-ranges=0.0.0.0/0 \
        --target-tags="prsm-bootstrap" \
        --description="Allow HTTPS access" 2>/dev/null || true
    
    # Allow internal communication
    gcloud compute firewall-rules create "prsm-bootstrap-${ENVIRONMENT}-allow-internal" \
        --project="$GCP_PROJECT" \
        --network="$network_name" \
        --allow=tcp,udp,icmp \
        --source-ranges=10.0.0.0/16 \
        --target-tags="prsm-bootstrap" \
        --description="Allow internal communication" 2>/dev/null || true
    
    log_success "Firewall rules created"
}

create_static_ip() {
    local ip_name="prsm-bootstrap-${ENVIRONMENT}-ip"
    
    log_info "Creating static IP address..."
    
    # Check if IP exists
    if gcloud compute addresses describe "$ip_name" \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" > /dev/null 2>&1; then
        log_warning "Static IP already exists"
        local existing_ip=$(gcloud compute addresses describe "$ip_name" \
            --region="$GCP_REGION" \
            --project="$GCP_PROJECT" \
            --format="value(address)")
        echo "$existing_ip"
        return 0
    fi
    
    # Reserve static IP
    local static_ip=$(gcloud compute addresses create "$ip_name" \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(address)")
    
    log_success "Created static IP: $static_ip"
    echo "$static_ip"
}

create_vm_instance() {
    local network_name=$1
    local subnet_name=$2
    local static_ip=$3
    local instance_name="prsm-bootstrap-${ENVIRONMENT}"
    
    log_info "Creating VM instance: $instance_name"
    
    # Check if instance exists
    if gcloud compute instances describe "$instance_name" \
        --zone="$GCP_ZONE" \
        --project="$GCP_PROJECT" > /dev/null 2>&1; then
        log_warning "VM instance already exists: $instance_name"
        echo "$instance_name"
        return 0
    fi
    
    # Create startup script
    local startup_script=$(cat << 'STARTUP'
#!/bin/bash
# PRSM Bootstrap Server GCE Startup Script

# Update system
apt-get update
apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release jq

# Install Docker
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Enable Docker
systemctl enable docker
systemctl start docker

# Install Google Cloud Ops Agent for logging and monitoring
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
bash add-google-cloud-ops-agent-repo.sh --also-install

# Create PRSM directory
mkdir -p /opt/prsm
cd /opt/prsm

# Clone or copy PRSM repository
# In production, you would clone from your repository
# git clone https://github.com/prsm-network/prsm.git .

# Create environment file
cat > .env << EOF
PRSM_ENV=production
PRSM_DOMAIN=${PRSM_DOMAIN}
PRSM_REGION=${GCP_REGION}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
PRSM_AUTH_SECRET=${PRSM_AUTH_SECRET}
EOF

# Pull and run bootstrap server
# docker compose -f docker-compose.bootstrap.yml up -d

# Setup log rotation
cat > /etc/logrotate.d/prsm << EOF
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    size=100M
    missingok
    delaycompress
    copytruncate
}
EOF

STARTUP
)
    
    # Create instance
    gcloud compute instances create "$instance_name" \
        --project="$GCP_PROJECT" \
        --zone="$GCP_ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --network-interface="network=$network_name,subnet=$subnet_name,address=$static_ip" \
        --tags="prsm-bootstrap" \
        --image-family="debian-11" \
        --image-project="debian-cloud" \
        --boot-disk-size="50GB" \
        --boot-disk-type="pd-ssd" \
        --metadata="startup-script=$startup_script" \
        --labels="environment=$ENVIRONMENT,project=prsm"
    
    log_success "Created VM instance: $instance_name"
    echo "$instance_name"
}

setup_dns() {
    local static_ip=$1
    local domain=$2
    
    log_info "Setting up Cloud DNS..."
    
    local dns_zone_name="prsm-bootstrap-${ENVIRONMENT}"
    local dns_name="bootstrap.${domain}."
    
    # Check if DNS zone exists
    if ! gcloud dns managed-zones describe "$dns_zone_name" --project="$GCP_PROJECT" > /dev/null 2>&1; then
        # Create DNS zone
        gcloud dns managed-zones create "$dns_zone_name" \
            --project="$GCP_PROJECT" \
            --dns-name="$dns_name" \
            --description="PRSM Bootstrap Server DNS zone"
    fi
    
    # Create DNS record
    local transaction_file="/tmp/dns-transaction-$$.yaml"
    
    gcloud dns record-sets transaction start \
        --zone="$dns_zone_name" \
        --project="$GCP_PROJECT"
    
    gcloud dns record-sets transaction add "$static_ip" \
        --name="bootstrap.${domain}." \
        --ttl=300 \
        --type=A \
        --zone="$dns_zone_name" \
        --project="$GCP_PROJECT"
    
    gcloud dns record-sets transaction execute \
        --zone="$dns_zone_name" \
        --project="$GCP_PROJECT"
    
    log_success "DNS record created: bootstrap.${domain} -> $static_ip"
}

setup_ssl() {
    local domain=$1
    
    log_info "Setting up SSL certificate..."
    
    # Create Google-managed SSL certificate
    local cert_name="prsm-bootstrap-${ENVIRONMENT}-ssl"
    
    if gcloud compute ssl-certificates describe "$cert_name" --project="$GCP_PROJECT" > /dev/null 2>&1; then
        log_warning "SSL certificate already exists: $cert_name"
        return 0
    fi
    
    gcloud compute ssl-certificates create "$cert_name" \
        --project="$GCP_PROJECT" \
        --domains="bootstrap.${domain}" \
        --global
    
    log_success "SSL certificate created: $cert_name"
}

create_health_check() {
    local health_check_name="prsm-bootstrap-${ENVIRONMENT}-health"
    
    log_info "Creating health check..."
    
    if gcloud compute health-checks describe "$health_check_name" --project="$GCP_PROJECT" > /dev/null 2>&1; then
        log_warning "Health check already exists"
        return 0
    fi
    
    gcloud compute health-checks create http "$health_check_name" \
        --project="$GCP_PROJECT" \
        --port=8000 \
        --request-path="/health" \
        --check-interval=30s \
        --timeout=10s \
        --healthy-threshold=2 \
        --unhealthy-threshold=3
    
    log_success "Health check created"
}

# =============================================================================
# Main Deployment
# =============================================================================

deploy_bootstrap_server() {
    log_info "Starting PRSM Bootstrap Server deployment to GCP..."
    log_info "Project: $GCP_PROJECT"
    log_info "Region: $GCP_REGION"
    log_info "Zone: $GCP_ZONE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Domain: $PRSM_DOMAIN"
    
    # Enable APIs
    enable_apis
    
    # Create network infrastructure
    local network_name=$(create_vpc_network)
    local subnet_name=$(create_subnet "$network_name")
    create_firewall_rules "$network_name"
    
    # Create static IP
    local static_ip=$(create_static_ip)
    
    # Create VM instance
    local instance_name=$(create_vm_instance "$network_name" "$subnet_name" "$static_ip")
    
    # Setup DNS
    setup_dns "$static_ip" "$PRSM_DOMAIN"
    
    # Setup SSL
    setup_ssl "$PRSM_DOMAIN"
    
    # Create health check
    create_health_check
    
    # Print summary
    echo ""
    log_success "=========================================="
    log_success "Deployment Complete!"
    log_success "=========================================="
    echo ""
    echo "Instance Name: $instance_name"
    echo "Static IP: $static_ip"
    echo "WebSocket URL: wss://bootstrap.${PRSM_DOMAIN}:8765"
    echo "API URL: https://bootstrap.${PRSM_DOMAIN}:8000"
    echo ""
    echo "Next Steps:"
    echo "1. SSH to instance: gcloud compute ssh $instance_name --zone=$GCP_ZONE"
    echo "2. Check logs: docker logs prsm-bootstrap"
    echo "3. Configure monitoring in Cloud Console"
    echo ""
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--region)
                GCP_REGION="$2"
                shift 2
                ;;
            -z|--zone)
                GCP_ZONE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--domain)
                PRSM_DOMAIN="$2"
                shift 2
                ;;
            -p|--project)
                GCP_PROJECT="$2"
                shift 2
                ;;
            -m|--machine-type)
                MACHINE_TYPE="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

# =============================================================================
# Script Entry Point
# =============================================================================

main() {
    # Set defaults from environment or defaults
    GCP_REGION="${GCP_REGION:-$DEFAULT_REGION}"
    GCP_ZONE="${GCP_ZONE:-$DEFAULT_ZONE}"
    ENVIRONMENT="${PRSM_ENV:-$DEFAULT_ENVIRONMENT}"
    PRSM_DOMAIN="${PRSM_DOMAIN:-$DEFAULT_DOMAIN}"
    MACHINE_TYPE="${MACHINE_TYPE:-$DEFAULT_MACHINE_TYPE}"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Check required environment variables
    if [ -z "${POSTGRES_PASSWORD:-}" ]; then
        log_warning "POSTGRES_PASSWORD not set, using default (not recommended for production)"
        POSTGRES_PASSWORD="prsm_secure_pass_$(date +%s)"
    fi
    
    if [ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]; then
        log_warning "GRAFANA_ADMIN_PASSWORD not set, using default"
        GRAFANA_ADMIN_PASSWORD="admin"
    fi
    
    if [ -z "${PRSM_AUTH_SECRET:-}" ]; then
        PRSM_AUTH_SECRET=$(openssl rand -hex 32 2>/dev/null || uuidgen | tr -d '-')
        log_info "Generated PRSM_AUTH_SECRET"
    fi
    
    # Run deployment
    deploy_bootstrap_server
}

# Run main function
main "$@"
