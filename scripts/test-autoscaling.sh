#!/bin/bash

# PRSM Kubernetes Autoscaling Test Script
# Tests HPA, VPA, and Cluster Autoscaler functionality

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-"prsm-system"}
API_ENDPOINT=${API_ENDPOINT:-"http://localhost:8000"}
LOAD_TEST_DURATION=${LOAD_TEST_DURATION:-"300"}  # 5 minutes
CONCURRENT_USERS=${CONCURRENT_USERS:-"1000"}
TARGET_RPS=${TARGET_RPS:-"100"}

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check for load testing tools
    if ! command -v hey &> /dev/null && ! command -v ab &> /dev/null; then
        log_warning "No load testing tool found. Installing hey..."
        if command -v go &> /dev/null; then
            go install github.com/rakyll/hey@latest
        else
            log_error "Please install 'hey' or 'apache2-utils' for load testing"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

check_autoscaler_status() {
    log_info "Checking autoscaler status..."
    
    # Check HPA status
    log_info "HPA Status:"
    kubectl get hpa -n "$NAMESPACE" -o wide || log_warning "No HPA found"
    
    # Check VPA status
    log_info "VPA Status:"
    kubectl get vpa -n "$NAMESPACE" -o wide || log_warning "No VPA found"
    
    # Check Cluster Autoscaler
    log_info "Cluster Autoscaler Status:"
    kubectl get deployment cluster-autoscaler -n kube-system -o wide || log_warning "Cluster autoscaler not found"
    
    # Check node capacity
    log_info "Node Capacity:"
    kubectl top nodes || log_warning "Metrics server not available"
    
    # Check pod status
    log_info "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -o wide
}

get_baseline_metrics() {
    log_info "Collecting baseline metrics..."
    
    # Current replica count
    INITIAL_REPLICAS=$(kubectl get deployment prsm-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    log_info "Initial replica count: $INITIAL_REPLICAS"
    
    # Current resource usage
    kubectl top pods -n "$NAMESPACE" --containers || log_warning "Pod metrics not available"
    
    # HPA current metrics
    kubectl describe hpa prsm-api-hpa -n "$NAMESPACE" || log_warning "HPA metrics not available"
    
    return $INITIAL_REPLICAS
}

run_load_test() {
    local duration=$1
    local concurrent=$2
    local rps=$3
    
    log_info "Starting load test: ${concurrent} concurrent users, ${rps} RPS for ${duration}s"
    
    # Create a simple test endpoint if it doesn't exist
    local test_endpoint="${API_ENDPOINT}/health"
    
    # Run load test based on available tool
    if command -v hey &> /dev/null; then
        hey -z "${duration}s" -c "$concurrent" -q "$rps" "$test_endpoint" > load_test_results.txt 2>&1 &
    elif command -v ab &> /dev/null; then
        local total_requests=$((rps * duration))
        ab -n "$total_requests" -c "$concurrent" "$test_endpoint" > load_test_results.txt 2>&1 &
    else
        log_error "No load testing tool available"
        return 1
    fi
    
    local load_test_pid=$!
    log_info "Load test started with PID: $load_test_pid"
    
    # Monitor scaling during load test
    local monitoring_interval=30
    local elapsed=0
    
    while [ $elapsed -lt $duration ]; do
        sleep $monitoring_interval
        elapsed=$((elapsed + monitoring_interval))
        
        log_info "=== Monitoring at ${elapsed}s ==="
        
        # Current replica count
        local current_replicas=$(kubectl get deployment prsm-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        log_info "Current replicas: $current_replicas"
        
        # HPA status
        kubectl get hpa prsm-api-hpa -n "$NAMESPACE" --no-headers | awk '{print "HPA: CPU=" $3 " Memory=" $4 " Replicas=" $6 "/" $7}'
        
        # Pod resource usage
        kubectl top pods -n "$NAMESPACE" --containers | grep prsm-api | head -5
        
        # Node resource usage
        kubectl top nodes | head -3
    done
    
    # Wait for load test to complete
    wait $load_test_pid
    log_success "Load test completed"
    
    # Show load test results
    if [ -f load_test_results.txt ]; then
        log_info "Load test results:"
        cat load_test_results.txt
    fi
}

monitor_scale_down() {
    log_info "Monitoring scale-down behavior..."
    
    local monitoring_duration=600  # 10 minutes
    local monitoring_interval=60   # Check every minute
    local elapsed=0
    
    while [ $elapsed -lt $monitoring_duration ]; do
        sleep $monitoring_interval
        elapsed=$((elapsed + monitoring_interval))
        
        log_info "=== Scale-down monitoring at ${elapsed}s ==="
        
        # Current replica count
        local current_replicas=$(kubectl get deployment prsm-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        log_info "Current replicas: $current_replicas"
        
        # Check if we've scaled down
        if [ "$current_replicas" -le "$INITIAL_REPLICAS" ]; then
            log_success "Scale-down detected after ${elapsed}s"
            break
        fi
        
        # HPA status
        kubectl get hpa prsm-api-hpa -n "$NAMESPACE" --no-headers
    done
}

analyze_results() {
    log_info "Analyzing autoscaling results..."
    
    # Final replica count
    local final_replicas=$(kubectl get deployment prsm-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    
    # HPA events
    log_info "HPA Events:"
    kubectl describe hpa prsm-api-hpa -n "$NAMESPACE" | grep -A 20 "Events:" || log_warning "No HPA events found"
    
    # Deployment events
    log_info "Deployment Events:"
    kubectl describe deployment prsm-api -n "$NAMESPACE" | grep -A 10 "Events:" || log_warning "No deployment events found"
    
    # Pod scaling timeline
    log_info "Pod Creation Timeline:"
    kubectl get events -n "$NAMESPACE" --sort-by='.firstTimestamp' | grep -E "(Scaled|Created|Started)" | tail -20
    
    # Cluster autoscaler logs (if available)
    log_info "Cluster Autoscaler Logs (last 50 lines):"
    kubectl logs -n kube-system deployment/cluster-autoscaler --tail=50 2>/dev/null || log_warning "Cluster autoscaler logs not available"
    
    # Summary
    echo ""
    log_info "=== AUTOSCALING TEST SUMMARY ==="
    echo "Initial replicas: $INITIAL_REPLICAS"
    echo "Final replicas: $final_replicas"
    echo "Max replicas during test: $(get_max_replicas_from_events)"
    echo "Load test duration: ${LOAD_TEST_DURATION}s"
    echo "Target concurrent users: $CONCURRENT_USERS"
    echo "Target RPS: $TARGET_RPS"
}

get_max_replicas_from_events() {
    # Extract max replica count from HPA events
    kubectl describe hpa prsm-api-hpa -n "$NAMESPACE" | \
        grep -E "scaled.*to [0-9]+" | \
        sed -E 's/.*to ([0-9]+).*/\1/' | \
        sort -n | \
        tail -1 || echo "unknown"
}

test_vpa_functionality() {
    log_info "Testing VPA functionality..."
    
    # Check VPA recommendations
    if kubectl get vpa prsm-api-vpa -n "$NAMESPACE" &> /dev/null; then
        log_info "VPA Recommendations:"
        kubectl describe vpa prsm-api-vpa -n "$NAMESPACE" | grep -A 20 "Recommendation:" || log_warning "No VPA recommendations available"
    else
        log_warning "VPA not configured for prsm-api"
    fi
}

cleanup() {
    log_info "Cleaning up test artifacts..."
    
    # Remove load test results
    rm -f load_test_results.txt
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill
    
    log_success "Cleanup completed"
}

main() {
    log_info "Starting PRSM Autoscaling Test"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --duration)
                LOAD_TEST_DURATION="$2"
                shift 2
                ;;
            --concurrent)
                CONCURRENT_USERS="$2"
                shift 2
                ;;
            --rps)
                TARGET_RPS="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --endpoint)
                API_ENDPOINT="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--duration <seconds>] [--concurrent <users>] [--rps <rate>] [--namespace <ns>] [--endpoint <url>]"
                exit 1
                ;;
        esac
    done
    
    # Run test phases
    check_prerequisites
    check_autoscaler_status
    
    INITIAL_REPLICAS=$(get_baseline_metrics)
    
    # Run the load test
    run_load_test "$LOAD_TEST_DURATION" "$CONCURRENT_USERS" "$TARGET_RPS"
    
    # Monitor scale-down
    monitor_scale_down
    
    # Test VPA
    test_vpa_functionality
    
    # Analyze results
    analyze_results
    
    log_success "Autoscaling test completed!"
    
    # Generate test report
    cat > autoscaling_test_report.md << EOF
# PRSM Autoscaling Test Report

## Test Configuration
- **Duration**: ${LOAD_TEST_DURATION}s
- **Concurrent Users**: ${CONCURRENT_USERS}
- **Target RPS**: ${TARGET_RPS}
- **Namespace**: ${NAMESPACE}
- **API Endpoint**: ${API_ENDPOINT}

## Results
- **Initial Replicas**: ${INITIAL_REPLICAS}
- **Final Replicas**: ${final_replicas}
- **Max Replicas**: $(get_max_replicas_from_events)

## Performance Validation
$([ -f load_test_results.txt ] && echo "See load_test_results.txt for detailed performance metrics")

## Recommendations
- HPA scaling behavior: $(kubectl get hpa prsm-api-hpa -n "$NAMESPACE" -o jsonpath='{.status.conditions[0].message}' 2>/dev/null || echo "Check HPA configuration")
- VPA status: $(kubectl get vpa prsm-api-vpa -n "$NAMESPACE" -o jsonpath='{.status.conditions[0].message}' 2>/dev/null || echo "VPA not configured")

Generated at: $(date)
EOF
    
    log_info "Test report saved to autoscaling_test_report.md"
}

# Handle script interruption
trap cleanup EXIT INT TERM

# Run main function
main "$@"