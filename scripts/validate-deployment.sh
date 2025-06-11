#!/bin/bash
set -euo pipefail

# PRSM Deployment Validation Script
# Comprehensive testing and validation of deployed PRSM infrastructure

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/validation-$(date +%Y%m%d-%H%M%S).log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENVIRONMENT="staging"
NAMESPACE="prsm-staging"
TIMEOUT=300
VERBOSE=false
CONTINUE_ON_ERROR=false

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

PRSM Deployment Validation Script

OPTIONS:
    -e, --environment ENV     Target environment (staging|production) [default: staging]
    -n, --namespace NAMESPACE Kubernetes namespace [default: prsm-staging]
    -t, --timeout SECONDS     Timeout for validation checks [default: 300]
    -v, --verbose             Enable verbose output
    -c, --continue-on-error   Continue validation even if some tests fail
    -h, --help                Show this help message

EXAMPLES:
    $0 --environment production
    $0 --namespace prsm-staging --verbose
    $0 --environment production --continue-on-error

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
    if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
        exit 1
    fi
}

# Function to validate Kubernetes connectivity
validate_kubernetes() {
    print_status "$BLUE" "Validating Kubernetes connectivity..."
    
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error_exit "Namespace '$NAMESPACE' does not exist"
        return 1
    fi
    
    print_status "$GREEN" "Kubernetes connectivity validated"
    return 0
}

# Function to validate pod status
validate_pods() {
    print_status "$BLUE" "Validating pod status..."
    
    local required_pods=("prsm-api" "prsm-worker" "postgres" "redis" "ipfs")
    local validation_passed=true
    
    for pod_label in "${required_pods[@]}"; do
        local pod_count
        pod_count=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$pod_label" --field-selector=status.phase=Running --no-headers | wc -l)
        
        if [[ $pod_count -eq 0 ]]; then
            print_status "$RED" "No running pods found for: $pod_label"
            validation_passed=false
        else
            print_status "$GREEN" "Found $pod_count running pod(s) for: $pod_label"
        fi
    done
    
    # Check for any failed pods
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed --no-headers | wc -l)
    
    if [[ $failed_pods -gt 0 ]]; then
        print_status "$RED" "Found $failed_pods failed pods"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed
        validation_passed=false
    fi
    
    if [[ "$validation_passed" == "true" ]]; then
        print_status "$GREEN" "Pod validation passed"
        return 0
    else
        error_exit "Pod validation failed"
        return 1
    fi
}

# Function to validate services
validate_services() {
    print_status "$BLUE" "Validating services..."
    
    local required_services=("prsm-api-service" "postgres-service" "redis-service" "ipfs-service")
    local validation_passed=true
    
    for service in "${required_services[@]}"; do
        if kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
            local endpoints
            endpoints=$(kubectl get endpoints "$service" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
            
            if [[ $endpoints -gt 0 ]]; then
                print_status "$GREEN" "Service $service has $endpoints endpoint(s)"
            else
                print_status "$RED" "Service $service has no endpoints"
                validation_passed=false
            fi
        else
            print_status "$RED" "Service not found: $service"
            validation_passed=false
        fi
    done
    
    if [[ "$validation_passed" == "true" ]]; then
        print_status "$GREEN" "Service validation passed"
        return 0
    else
        error_exit "Service validation failed"
        return 1
    fi
}

# Function to validate ingress
validate_ingress() {
    print_status "$BLUE" "Validating ingress..."
    
    local ingress_name
    case "$ENVIRONMENT" in
        production) ingress_name="prsm-production-ingress" ;;
        staging) ingress_name="prsm-staging-ingress" ;;
        *) ingress_name="prsm-ingress" ;;
    esac
    
    if kubectl get ingress "$ingress_name" -n "$NAMESPACE" &> /dev/null; then
        local ingress_ip
        ingress_ip=$(kubectl get ingress "$ingress_name" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        
        if [[ -n "$ingress_ip" ]]; then
            print_status "$GREEN" "Ingress has IP: $ingress_ip"
        else
            print_status "$YELLOW" "Ingress IP not yet assigned"
        fi
    else
        print_status "$YELLOW" "Ingress not found: $ingress_name"
    fi
    
    return 0
}

# Function to validate API health
validate_api_health() {
    print_status "$BLUE" "Validating API health..."
    
    # Port forward to API service
    kubectl port-forward -n "$NAMESPACE" service/prsm-api-service 8080:8000 &
    local pf_pid=$!
    sleep 3
    
    local health_check_passed=false
    
    # Try health check multiple times
    for i in {1..5}; do
        if curl -f http://localhost:8080/health --max-time 10 &> /dev/null; then
            print_status "$GREEN" "API health check passed"
            health_check_passed=true
            break
        else
            print_status "$YELLOW" "API health check failed (attempt $i/5)"
            sleep 5
        fi
    done
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
    wait $pf_pid 2>/dev/null || true
    
    if [[ "$health_check_passed" == "true" ]]; then
        return 0
    else
        error_exit "API health validation failed"
        return 1
    fi
}

# Function to validate database connectivity
validate_database() {
    print_status "$BLUE" "Validating database connectivity..."
    
    local db_check_result
    db_check_result=$(kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_isready -U postgres 2>/dev/null | grep "accepting connections" || echo "failed")
    
    if [[ "$db_check_result" != "failed" ]]; then
        print_status "$GREEN" "Database connectivity validated"
        return 0
    else
        error_exit "Database connectivity validation failed"
        return 1
    fi
}

# Function to validate Redis connectivity
validate_redis() {
    print_status "$BLUE" "Validating Redis connectivity..."
    
    local redis_check_result
    redis_check_result=$(kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli ping 2>/dev/null || echo "failed")
    
    if [[ "$redis_check_result" == "PONG" ]]; then
        print_status "$GREEN" "Redis connectivity validated"
        return 0
    else
        error_exit "Redis connectivity validation failed"
        return 1
    fi
}

# Function to validate IPFS connectivity
validate_ipfs() {
    print_status "$BLUE" "Validating IPFS connectivity..."
    
    local ipfs_check_result
    ipfs_check_result=$(kubectl exec -n "$NAMESPACE" deployment/ipfs -- ipfs id 2>/dev/null | grep '"ID"' || echo "failed")
    
    if [[ "$ipfs_check_result" != "failed" ]]; then
        print_status "$GREEN" "IPFS connectivity validated"
        return 0
    else
        error_exit "IPFS connectivity validation failed"
        return 1
    fi
}

# Function to validate resource usage
validate_resources() {
    print_status "$BLUE" "Validating resource usage..."
    
    local high_cpu_pods
    high_cpu_pods=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$2 ~ /[0-9]+m/ && $2+0 > 1000 {print $1}' || echo "")
    
    if [[ -n "$high_cpu_pods" ]]; then
        print_status "$YELLOW" "High CPU usage detected in pods: $high_cpu_pods"
    else
        print_status "$GREEN" "CPU usage within normal limits"
    fi
    
    local high_memory_pods
    high_memory_pods=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$3 ~ /[0-9]+Mi/ && $3+0 > 2048 {print $1}' || echo "")
    
    if [[ -n "$high_memory_pods" ]]; then
        print_status "$YELLOW" "High memory usage detected in pods: $high_memory_pods"
    else
        print_status "$GREEN" "Memory usage within normal limits"
    fi
    
    return 0
}

# Function to validate persistent volumes
validate_volumes() {
    print_status "$BLUE" "Validating persistent volumes..."
    
    local pvc_status
    pvc_status=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$2 != "Bound" {print $1}' || echo "")
    
    if [[ -n "$pvc_status" ]]; then
        print_status "$RED" "Unbound PVCs found: $pvc_status"
        error_exit "PVC validation failed"
        return 1
    else
        print_status "$GREEN" "All PVCs are bound"
        return 0
    fi
}

# Function to run performance tests
run_performance_tests() {
    print_status "$BLUE" "Running basic performance tests..."
    
    # Port forward to API service
    kubectl port-forward -n "$NAMESPACE" service/prsm-api-service 8080:8000 &
    local pf_pid=$!
    sleep 3
    
    local performance_passed=true
    
    # Simple load test with curl
    for i in {1..10}; do
        local start_time=$(date +%s%3N)
        if curl -f http://localhost:8080/health --max-time 5 &> /dev/null; then
            local end_time=$(date +%s%3N)
            local response_time=$((end_time - start_time))
            
            if [[ $response_time -gt 2000 ]]; then
                print_status "$YELLOW" "Slow response time: ${response_time}ms"
                performance_passed=false
            fi
        else
            print_status "$RED" "Health check failed during performance test"
            performance_passed=false
        fi
    done
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
    wait $pf_pid 2>/dev/null || true
    
    if [[ "$performance_passed" == "true" ]]; then
        print_status "$GREEN" "Basic performance tests passed"
        return 0
    else
        print_status "$YELLOW" "Performance tests showed potential issues"
        return 0  # Don't fail validation for performance issues
    fi
}

# Function to generate validation report
generate_report() {
    local report_file="${PROJECT_ROOT}/logs/validation-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# PRSM Deployment Validation Report

**Environment:** $ENVIRONMENT  
**Namespace:** $NAMESPACE  
**Validation Date:** $(date)  
**Validation Status:** $(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l) pods validated

## Deployment Overview

\`\`\`
$(kubectl get all -n "$NAMESPACE")
\`\`\`

## Resource Usage

\`\`\`
$(kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics not available")
\`\`\`

## Events (Last 10)

\`\`\`
$(kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10)
\`\`\`

## Validation Log

\`\`\`
$(tail -50 "$LOG_FILE")
\`\`\`

EOF
    
    print_status "$GREEN" "Validation report generated: $report_file"
}

# Main validation function
main() {
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    print_status "$BLUE" "Starting PRSM deployment validation..."
    print_status "$BLUE" "Environment: $ENVIRONMENT"
    print_status "$BLUE" "Namespace: $NAMESPACE"
    
    local validation_results=()
    
    # Run all validation checks
    validate_kubernetes && validation_results+=("kubernetes:PASS") || validation_results+=("kubernetes:FAIL")
    validate_pods && validation_results+=("pods:PASS") || validation_results+=("pods:FAIL")
    validate_services && validation_results+=("services:PASS") || validation_results+=("services:FAIL")
    validate_ingress && validation_results+=("ingress:PASS") || validation_results+=("ingress:FAIL")
    validate_api_health && validation_results+=("api:PASS") || validation_results+=("api:FAIL")
    validate_database && validation_results+=("database:PASS") || validation_results+=("database:FAIL")
    validate_redis && validation_results+=("redis:PASS") || validation_results+=("redis:FAIL")
    validate_ipfs && validation_results+=("ipfs:PASS") || validation_results+=("ipfs:FAIL")
    validate_volumes && validation_results+=("volumes:PASS") || validation_results+=("volumes:FAIL")
    validate_resources && validation_results+=("resources:PASS") || validation_results+=("resources:FAIL")
    run_performance_tests && validation_results+=("performance:PASS") || validation_results+=("performance:FAIL")
    
    # Summary
    print_status "$BLUE" "Validation Summary:"
    local failed_count=0
    for result in "${validation_results[@]}"; do
        local check_name=$(echo "$result" | cut -d: -f1)
        local check_result=$(echo "$result" | cut -d: -f2)
        
        if [[ "$check_result" == "PASS" ]]; then
            print_status "$GREEN" "  ✅ $check_name: PASSED"
        else
            print_status "$RED" "  ❌ $check_name: FAILED"
            ((failed_count++))
        fi
    done
    
    # Generate report
    generate_report
    
    if [[ $failed_count -eq 0 ]]; then
        print_status "$GREEN" "🎉 All validation checks passed!"
        return 0
    else
        print_status "$RED" "❌ $failed_count validation check(s) failed"
        return 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            NAMESPACE="prsm-$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main "$@"