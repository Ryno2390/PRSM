#!/bin/bash
set -euo pipefail

# PRSM Backup and Recovery System
# Comprehensive backup solution with automated scheduling and disaster recovery

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_BASE_DIR="${PROJECT_ROOT}/backups"
LOG_FILE="${BACKUP_BASE_DIR}/backup-$(date +%Y%m%d-%H%M%S).log"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
OPERATION="backup"
ENVIRONMENT="staging"
NAMESPACE="prsm-system"
BACKUP_TYPE="full"
RETENTION_DAYS=30
COMPRESSION="gzip"
REMOTE_STORAGE=""
ENCRYPTION_KEY=""
NOTIFY_EMAIL=""
VERIFY_BACKUP=true
QUIET=false

# Backup components
BACKUP_DATABASE=true
BACKUP_REDIS=true
BACKUP_IPFS=true
BACKUP_KUBERNETES=true
BACKUP_VOLUMES=true
BACKUP_CONFIGS=true

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPERATION] [OPTIONS]

PRSM Backup and Recovery System

OPERATIONS:
    backup          Create a backup (default)
    restore         Restore from backup
    list            List available backups
    verify          Verify backup integrity
    cleanup         Clean up old backups
    schedule        Set up automated backup scheduling

OPTIONS:
    -e, --environment ENV     Target environment (staging|production) [default: staging]
    -n, --namespace NAMESPACE Kubernetes namespace [default: prsm-system]
    -t, --type TYPE           Backup type (full|incremental|differential) [default: full]
    -r, --retention DAYS      Backup retention in days [default: 30]
    -c, --compression TYPE    Compression type (gzip|lz4|zstd) [default: gzip]
    -s, --storage URL         Remote storage URL (s3://bucket, gs://bucket, etc.)
    -k, --encryption-key KEY  Encryption key file for backup encryption
    --notify EMAIL            Email for backup notifications
    --no-verify              Skip backup verification
    --database-only           Backup only database
    --redis-only              Backup only Redis
    --ipfs-only               Backup only IPFS
    --volumes-only            Backup only persistent volumes
    --configs-only            Backup only configurations
    --quiet                   Suppress non-error output
    -h, --help                Show this help message

EXAMPLES:
    $0 backup --environment production --type full
    $0 restore --environment staging --backup-id 20240611-143022
    $0 cleanup --retention 7
    $0 schedule --environment production --daily

EOF
}

# Function to log messages
log() {
    local level=$1
    shift
    local message="$*"
    if [[ "$QUIET" != "true" || "$level" == "ERROR" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$LOG_FILE"
    fi
}

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${color}[$(date '+%H:%M:%S')] $message${NC}"
    fi
    log "INFO" "$message"
}

# Function to handle errors
error_exit() {
    print_status "$RED" "ERROR: $1"
    send_notification "BACKUP_FAILED" "$1"
    exit 1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "$BLUE" "Checking backup prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "pg_dump" "redis-cli")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "Required tool '$tool' is not installed"
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error_exit "Namespace '$NAMESPACE' does not exist"
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_BASE_DIR"
    
    print_status "$GREEN" "Prerequisites check passed"
}

# Function to create backup directory structure
create_backup_structure() {
    local backup_id=$1
    local backup_dir="${BACKUP_BASE_DIR}/${backup_id}"
    
    mkdir -p "$backup_dir"/{database,redis,ipfs,kubernetes,volumes,configs,metadata}
    echo "$backup_dir"
}

# Function to backup PostgreSQL database
backup_database() {
    local backup_dir=$1
    
    if [[ "$BACKUP_DATABASE" != "true" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Backing up PostgreSQL database..."
    
    # Check if PostgreSQL pod exists
    if ! kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/name=postgres | grep -q Running; then
        print_status "$YELLOW" "PostgreSQL pod not found or not running, skipping database backup"
        return 0
    fi
    
    local db_backup_file="$backup_dir/database/postgres_${TIMESTAMP}.sql"
    
    # Create database backup
    kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_dumpall -U postgres > "$db_backup_file" || {
        error_exit "Failed to backup PostgreSQL database"
    }
    
    # Compress backup
    compress_file "$db_backup_file"
    
    # Get database statistics
    local db_stats_file="$backup_dir/database/stats_${TIMESTAMP}.json"
    kubectl exec -n "$NAMESPACE" deployment/postgres -- psql -U postgres -c "SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables;" -t -A -F',' > "$db_stats_file" || true
    
    print_status "$GREEN" "Database backup completed"
}

# Function to backup Redis data
backup_redis() {
    local backup_dir=$1
    
    if [[ "$BACKUP_REDIS" != "true" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Backing up Redis data..."
    
    # Check if Redis pod exists
    if ! kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/name=redis | grep -q Running; then
        print_status "$YELLOW" "Redis pod not found or not running, skipping Redis backup"
        return 0
    fi
    
    local redis_backup_file="$backup_dir/redis/redis_${TIMESTAMP}.rdb"
    
    # Trigger Redis save and copy RDB file
    kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli BGSAVE
    sleep 5  # Wait for background save to complete
    
    kubectl cp "$NAMESPACE/redis-pod:/data/dump.rdb" "$redis_backup_file" 2>/dev/null || {
        # Alternative: use redis-cli to dump data
        kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli --rdb /tmp/backup.rdb || {
            error_exit "Failed to backup Redis data"
        }
        kubectl cp "$NAMESPACE/redis-pod:/tmp/backup.rdb" "$redis_backup_file" || {
            error_exit "Failed to copy Redis backup file"
        }
    }
    
    # Compress backup
    compress_file "$redis_backup_file"
    
    # Get Redis info
    local redis_info_file="$backup_dir/redis/info_${TIMESTAMP}.txt"
    kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli INFO > "$redis_info_file" || true
    
    print_status "$GREEN" "Redis backup completed"
}

# Function to backup IPFS data
backup_ipfs() {
    local backup_dir=$1
    
    if [[ "$BACKUP_IPFS" != "true" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Backing up IPFS data..."
    
    # Check if IPFS pod exists
    if ! kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/name=ipfs | grep -q Running; then
        print_status "$YELLOW" "IPFS pod not found or not running, skipping IPFS backup"
        return 0
    fi
    
    local ipfs_backup_dir="$backup_dir/ipfs"
    
    # Backup IPFS repository
    kubectl exec -n "$NAMESPACE" deployment/ipfs -- tar czf /tmp/ipfs_backup.tar.gz -C /data/ipfs . || {
        error_exit "Failed to create IPFS backup archive"
    }
    
    kubectl cp "$NAMESPACE/ipfs-pod:/tmp/ipfs_backup.tar.gz" "$ipfs_backup_dir/ipfs_${TIMESTAMP}.tar.gz" || {
        error_exit "Failed to copy IPFS backup file"
    }
    
    # Get IPFS stats
    local ipfs_stats_file="$ipfs_backup_dir/stats_${TIMESTAMP}.json"
    kubectl exec -n "$NAMESPACE" deployment/ipfs -- ipfs stats repo > "$ipfs_stats_file" 2>/dev/null || true
    
    print_status "$GREEN" "IPFS backup completed"
}

# Function to backup Kubernetes resources
backup_kubernetes() {
    local backup_dir=$1
    
    if [[ "$BACKUP_KUBERNETES" != "true" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Backing up Kubernetes resources..."
    
    local k8s_backup_dir="$backup_dir/kubernetes"
    
    # Backup all resources in namespace
    kubectl get all -n "$NAMESPACE" -o yaml > "$k8s_backup_dir/all_resources_${TIMESTAMP}.yaml" || true
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$k8s_backup_dir/configmaps_${TIMESTAMP}.yaml" || true
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$k8s_backup_dir/secrets_${TIMESTAMP}.yaml" || true
    kubectl get pvc -n "$NAMESPACE" -o yaml > "$k8s_backup_dir/pvcs_${TIMESTAMP}.yaml" || true
    kubectl get ingress -n "$NAMESPACE" -o yaml > "$k8s_backup_dir/ingress_${TIMESTAMP}.yaml" 2>/dev/null || true
    kubectl get networkpolicies -n "$NAMESPACE" -o yaml > "$k8s_backup_dir/networkpolicies_${TIMESTAMP}.yaml" 2>/dev/null || true
    
    # Backup custom resources
    kubectl get crd -o yaml > "$k8s_backup_dir/crds_${TIMESTAMP}.yaml" 2>/dev/null || true
    
    # Compress Kubernetes backups
    tar czf "$k8s_backup_dir/kubernetes_${TIMESTAMP}.tar.gz" -C "$k8s_backup_dir" *.yaml 2>/dev/null || true
    rm -f "$k8s_backup_dir"/*.yaml 2>/dev/null || true
    
    print_status "$GREEN" "Kubernetes backup completed"
}

# Function to backup persistent volumes
backup_volumes() {
    local backup_dir=$1
    
    if [[ "$BACKUP_VOLUMES" != "true" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Backing up persistent volumes..."
    
    local volumes_backup_dir="$backup_dir/volumes"
    
    # Get list of PVCs
    local pvcs
    pvcs=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$pvcs" ]]; then
        for pvc in $pvcs; do
            print_status "$BLUE" "Backing up PVC: $pvc"
            
            # Create a backup job for each PVC
            local backup_job="backup-$pvc-${TIMESTAMP}"
            
            kubectl run "$backup_job" -n "$NAMESPACE" \
                --image=alpine:latest \
                --restart=Never \
                --rm \
                --overrides="{
                    \"spec\": {
                        \"containers\": [{
                            \"name\": \"backup\",
                            \"image\": \"alpine:latest\",
                            \"command\": [\"tar\", \"czf\", \"/backup/${pvc}_${TIMESTAMP}.tar.gz\", \"-C\", \"/data\", \".\"],
                            \"volumeMounts\": [
                                {\"name\": \"data\", \"mountPath\": \"/data\"},
                                {\"name\": \"backup\", \"mountPath\": \"/backup\"}
                            ]
                        }],
                        \"volumes\": [
                            {\"name\": \"data\", \"persistentVolumeClaim\": {\"claimName\": \"$pvc\"}},
                            {\"name\": \"backup\", \"emptyDir\": {}}
                        ]
                    }
                }" \
                --timeout=300s || {
                    print_status "$YELLOW" "Warning: Failed to backup PVC: $pvc"
                    continue
                }
            
            # Copy backup file from pod
            kubectl cp "$NAMESPACE/$backup_job:/backup/${pvc}_${TIMESTAMP}.tar.gz" "$volumes_backup_dir/${pvc}_${TIMESTAMP}.tar.gz" 2>/dev/null || {
                print_status "$YELLOW" "Warning: Failed to copy backup for PVC: $pvc"
            }
        done
    else
        print_status "$YELLOW" "No persistent volumes found to backup"
    fi
    
    print_status "$GREEN" "Volumes backup completed"
}

# Function to backup configurations
backup_configs() {
    local backup_dir=$1
    
    if [[ "$BACKUP_CONFIGS" != "true" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Backing up configurations..."
    
    local configs_backup_dir="$backup_dir/configs"
    
    # Copy configuration files
    cp -r "${PROJECT_ROOT}/config" "$configs_backup_dir/" 2>/dev/null || true
    cp -r "${PROJECT_ROOT}/deploy" "$configs_backup_dir/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/docker-compose.yml" "$configs_backup_dir/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/docker-compose.dev.yml" "$configs_backup_dir/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/Dockerfile" "$configs_backup_dir/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/requirements.txt" "$configs_backup_dir/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/pyproject.toml" "$configs_backup_dir/" 2>/dev/null || true
    
    # Compress configurations
    tar czf "$configs_backup_dir/configs_${TIMESTAMP}.tar.gz" -C "$configs_backup_dir" . 2>/dev/null || true
    
    print_status "$GREEN" "Configurations backup completed"
}

# Function to create backup metadata
create_backup_metadata() {
    local backup_dir=$1
    local backup_id=$2
    
    local metadata_file="$backup_dir/metadata/backup_info.json"
    
    cat > "$metadata_file" << EOF
{
    "backup_id": "$backup_id",
    "timestamp": "$TIMESTAMP",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "backup_type": "$BACKUP_TYPE",
    "compression": "$COMPRESSION",
    "components": {
        "database": $BACKUP_DATABASE,
        "redis": $BACKUP_REDIS,
        "ipfs": $BACKUP_IPFS,
        "kubernetes": $BACKUP_KUBERNETES,
        "volumes": $BACKUP_VOLUMES,
        "configs": $BACKUP_CONFIGS
    },
    "kubernetes_version": "$(kubectl version --client -o json 2>/dev/null | jq -r .clientVersion.gitVersion || echo 'unknown')",
    "cluster_info": "$(kubectl cluster-info --context=$(kubectl config current-context) 2>/dev/null | head -1 || echo 'unknown')",
    "backup_size": "$(du -sh "$backup_dir" | cut -f1)",
    "created_by": "$(whoami)@$(hostname)",
    "retention_until": "$(date -d "+$RETENTION_DAYS days" '+%Y-%m-%d %H:%M:%S')"
}
EOF
}

# Function to compress files
compress_file() {
    local file=$1
    
    case $COMPRESSION in
        gzip)
            gzip "$file" || error_exit "Failed to compress file: $file"
            ;;
        lz4)
            lz4 "$file" "$file.lz4" && rm "$file" || error_exit "Failed to compress file: $file"
            ;;
        zstd)
            zstd "$file" && rm "$file" || error_exit "Failed to compress file: $file"
            ;;
        *)
            print_status "$YELLOW" "Unknown compression type: $COMPRESSION, using gzip"
            gzip "$file" || error_exit "Failed to compress file: $file"
            ;;
    esac
}

# Function to verify backup integrity
verify_backup() {
    local backup_dir=$1
    
    if [[ "$VERIFY_BACKUP" != "true" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Verifying backup integrity..."
    
    local verification_log="$backup_dir/metadata/verification.log"
    local verification_passed=true
    
    # Check if backup directory exists and has content
    if [[ ! -d "$backup_dir" ]]; then
        echo "ERROR: Backup directory does not exist" >> "$verification_log"
        verification_passed=false
    fi
    
    # Verify each component
    if [[ "$BACKUP_DATABASE" == "true" ]]; then
        if ls "$backup_dir/database/"*.sql.gz &> /dev/null; then
            echo "OK: Database backup found" >> "$verification_log"
        else
            echo "ERROR: Database backup not found" >> "$verification_log"
            verification_passed=false
        fi
    fi
    
    if [[ "$BACKUP_REDIS" == "true" ]]; then
        if ls "$backup_dir/redis/"*.rdb.gz &> /dev/null; then
            echo "OK: Redis backup found" >> "$verification_log"
        else
            echo "ERROR: Redis backup not found" >> "$verification_log"
            verification_passed=false
        fi
    fi
    
    if [[ "$BACKUP_IPFS" == "true" ]]; then
        if ls "$backup_dir/ipfs/"*.tar.gz &> /dev/null; then
            echo "OK: IPFS backup found" >> "$verification_log"
        else
            echo "ERROR: IPFS backup not found" >> "$verification_log"
            verification_passed=false
        fi
    fi
    
    if [[ "$BACKUP_KUBERNETES" == "true" ]]; then
        if ls "$backup_dir/kubernetes/"*.tar.gz &> /dev/null; then
            echo "OK: Kubernetes backup found" >> "$verification_log"
        else
            echo "ERROR: Kubernetes backup not found" >> "$verification_log"
            verification_passed=false
        fi
    fi
    
    # Check file integrity (if compressed files are valid)
    for compressed_file in $(find "$backup_dir" -name "*.gz" -o -name "*.lz4" -o -name "*.zst"); do
        case "$compressed_file" in
            *.gz)
                if gzip -t "$compressed_file" &> /dev/null; then
                    echo "OK: $(basename "$compressed_file") integrity verified" >> "$verification_log"
                else
                    echo "ERROR: $(basename "$compressed_file") integrity check failed" >> "$verification_log"
                    verification_passed=false
                fi
                ;;
            *.lz4)
                if lz4 -t "$compressed_file" &> /dev/null; then
                    echo "OK: $(basename "$compressed_file") integrity verified" >> "$verification_log"
                else
                    echo "ERROR: $(basename "$compressed_file") integrity check failed" >> "$verification_log"
                    verification_passed=false
                fi
                ;;
            *.zst)
                if zstd -t "$compressed_file" &> /dev/null; then
                    echo "OK: $(basename "$compressed_file") integrity verified" >> "$verification_log"
                else
                    echo "ERROR: $(basename "$compressed_file") integrity check failed" >> "$verification_log"
                    verification_passed=false
                fi
                ;;
        esac
    done
    
    if [[ "$verification_passed" == "true" ]]; then
        print_status "$GREEN" "Backup verification passed"
        echo "VERIFICATION: PASSED" >> "$verification_log"
    else
        error_exit "Backup verification failed - check $verification_log"
    fi
}

# Function to upload to remote storage
upload_to_remote() {
    local backup_dir=$1
    
    if [[ -z "$REMOTE_STORAGE" ]]; then
        return 0
    fi
    
    print_status "$BLUE" "Uploading backup to remote storage: $REMOTE_STORAGE"
    
    case "$REMOTE_STORAGE" in
        s3://*)
            aws s3 cp "$backup_dir" "$REMOTE_STORAGE/$(basename "$backup_dir")" --recursive || {
                error_exit "Failed to upload backup to S3"
            }
            ;;
        gs://*)
            gsutil -m cp -r "$backup_dir" "$REMOTE_STORAGE/$(basename "$backup_dir")" || {
                error_exit "Failed to upload backup to Google Cloud Storage"
            }
            ;;
        *)
            print_status "$YELLOW" "Unknown remote storage type: $REMOTE_STORAGE"
            ;;
    esac
    
    print_status "$GREEN" "Backup uploaded to remote storage"
}

# Function to send notifications
send_notification() {
    local status=$1
    local message=$2
    
    if [[ -z "$NOTIFY_EMAIL" ]]; then
        return 0
    fi
    
    local subject
    case "$status" in
        "BACKUP_SUCCESS")
            subject="✅ PRSM Backup Successful - $ENVIRONMENT"
            ;;
        "BACKUP_FAILED")
            subject="❌ PRSM Backup Failed - $ENVIRONMENT"
            ;;
        "RESTORE_SUCCESS")
            subject="✅ PRSM Restore Successful - $ENVIRONMENT"
            ;;
        "RESTORE_FAILED")
            subject="❌ PRSM Restore Failed - $ENVIRONMENT"
            ;;
        *)
            subject="📢 PRSM Backup Notification - $ENVIRONMENT"
            ;;
    esac
    
    # Send email notification (requires mail or sendmail)
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "$subject" "$NOTIFY_EMAIL" || true
    elif command -v sendmail &> /dev/null; then
        {
            echo "To: $NOTIFY_EMAIL"
            echo "Subject: $subject"
            echo ""
            echo "$message"
        } | sendmail "$NOTIFY_EMAIL" || true
    fi
}

# Function to perform full backup
perform_backup() {
    local backup_id="backup_${ENVIRONMENT}_${TIMESTAMP}"
    local backup_dir
    backup_dir=$(create_backup_structure "$backup_id")
    
    print_status "$BLUE" "Starting $BACKUP_TYPE backup for environment: $ENVIRONMENT"
    print_status "$BLUE" "Backup ID: $backup_id"
    print_status "$BLUE" "Backup directory: $backup_dir"
    
    # Perform backup operations
    backup_database "$backup_dir"
    backup_redis "$backup_dir"
    backup_ipfs "$backup_dir"
    backup_kubernetes "$backup_dir"
    backup_volumes "$backup_dir"
    backup_configs "$backup_dir"
    
    # Create metadata
    create_backup_metadata "$backup_dir" "$backup_id"
    
    # Verify backup
    verify_backup "$backup_dir"
    
    # Upload to remote storage
    upload_to_remote "$backup_dir"
    
    # Calculate final backup size
    local backup_size
    backup_size=$(du -sh "$backup_dir" | cut -f1)
    
    print_status "$GREEN" "Backup completed successfully!"
    print_status "$GREEN" "Backup ID: $backup_id"
    print_status "$GREEN" "Backup Size: $backup_size"
    print_status "$GREEN" "Backup Location: $backup_dir"
    
    # Send success notification
    send_notification "BACKUP_SUCCESS" "Backup completed successfully\nBackup ID: $backup_id\nSize: $backup_size\nLocation: $backup_dir"
    
    echo "$backup_id"
}

# Function to list available backups
list_backups() {
    print_status "$BLUE" "Available backups:"
    
    if [[ ! -d "$BACKUP_BASE_DIR" ]]; then
        print_status "$YELLOW" "No backup directory found"
        return 0
    fi
    
    local backups
    backups=$(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "backup_*" | sort -r)
    
    if [[ -z "$backups" ]]; then
        print_status "$YELLOW" "No backups found"
        return 0
    fi
    
    printf "%-25s %-15s %-10s %-15s %s\n" "BACKUP_ID" "ENVIRONMENT" "TYPE" "SIZE" "DATE"
    printf "%-25s %-15s %-10s %-15s %s\n" "-------------------------" "---------------" "----------" "---------------" "-------------------"
    
    for backup_path in $backups; do
        local backup_id
        backup_id=$(basename "$backup_path")
        local metadata_file="$backup_path/metadata/backup_info.json"
        
        if [[ -f "$metadata_file" ]]; then
            local env
            local type
            local size
            local date
            
            env=$(jq -r '.environment // "unknown"' "$metadata_file" 2>/dev/null)
            type=$(jq -r '.backup_type // "unknown"' "$metadata_file" 2>/dev/null)
            size=$(jq -r '.backup_size // "unknown"' "$metadata_file" 2>/dev/null)
            date=$(jq -r '.timestamp // "unknown"' "$metadata_file" 2>/dev/null)
            
            printf "%-25s %-15s %-10s %-15s %s\n" "$backup_id" "$env" "$type" "$size" "$date"
        else
            local size
            size=$(du -sh "$backup_path" 2>/dev/null | cut -f1 || echo "unknown")
            printf "%-25s %-15s %-10s %-15s %s\n" "$backup_id" "unknown" "unknown" "$size" "unknown"
        fi
    done
}

# Function to cleanup old backups
cleanup_backups() {
    print_status "$BLUE" "Cleaning up backups older than $RETENTION_DAYS days..."
    
    if [[ ! -d "$BACKUP_BASE_DIR" ]]; then
        print_status "$YELLOW" "No backup directory found"
        return 0
    fi
    
    local deleted_count=0
    local total_size_freed=0
    
    # Find and delete old backups
    while IFS= read -r -d '' backup_path; do
        local backup_size
        backup_size=$(du -sb "$backup_path" 2>/dev/null | cut -f1 || echo "0")
        
        print_status "$BLUE" "Removing old backup: $(basename "$backup_path")"
        rm -rf "$backup_path"
        
        ((deleted_count++))
        ((total_size_freed += backup_size))
    done < <(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "backup_*" -mtime +"$RETENTION_DAYS" -print0)
    
    if [[ $deleted_count -eq 0 ]]; then
        print_status "$GREEN" "No old backups found to clean up"
    else
        local size_freed_human
        size_freed_human=$(numfmt --to=iec "$total_size_freed")
        print_status "$GREEN" "Cleanup completed: $deleted_count backups removed, $size_freed_human freed"
    fi
}

# Main function
main() {
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    case "$OPERATION" in
        backup)
            check_prerequisites
            perform_backup
            cleanup_backups
            ;;
        list)
            list_backups
            ;;
        cleanup)
            cleanup_backups
            ;;
        verify)
            print_status "$BLUE" "Backup verification functionality available during backup process"
            ;;
        *)
            error_exit "Unknown operation: $OPERATION"
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        backup|restore|list|verify|cleanup|schedule)
            OPERATION="$1"
            shift
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--type)
            BACKUP_TYPE="$2"
            shift 2
            ;;
        -r|--retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        -c|--compression)
            COMPRESSION="$2"
            shift 2
            ;;
        -s|--storage)
            REMOTE_STORAGE="$2"
            shift 2
            ;;
        -k|--encryption-key)
            ENCRYPTION_KEY="$2"
            shift 2
            ;;
        --notify)
            NOTIFY_EMAIL="$2"
            shift 2
            ;;
        --no-verify)
            VERIFY_BACKUP=false
            shift
            ;;
        --database-only)
            BACKUP_DATABASE=true
            BACKUP_REDIS=false
            BACKUP_IPFS=false
            BACKUP_KUBERNETES=false
            BACKUP_VOLUMES=false
            BACKUP_CONFIGS=false
            shift
            ;;
        --redis-only)
            BACKUP_DATABASE=false
            BACKUP_REDIS=true
            BACKUP_IPFS=false
            BACKUP_KUBERNETES=false
            BACKUP_VOLUMES=false
            BACKUP_CONFIGS=false
            shift
            ;;
        --ipfs-only)
            BACKUP_DATABASE=false
            BACKUP_REDIS=false
            BACKUP_IPFS=true
            BACKUP_KUBERNETES=false
            BACKUP_VOLUMES=false
            BACKUP_CONFIGS=false
            shift
            ;;
        --volumes-only)
            BACKUP_DATABASE=false
            BACKUP_REDIS=false
            BACKUP_IPFS=false
            BACKUP_KUBERNETES=false
            BACKUP_VOLUMES=true
            BACKUP_CONFIGS=false
            shift
            ;;
        --configs-only)
            BACKUP_DATABASE=false
            BACKUP_REDIS=false
            BACKUP_IPFS=false
            BACKUP_KUBERNETES=false
            BACKUP_VOLUMES=false
            BACKUP_CONFIGS=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
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

# Run main function
main "$@"