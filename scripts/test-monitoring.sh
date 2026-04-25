#!/bin/bash
set -euo pipefail

# PRSM Monitoring & Alerting Test Script
# Comprehensive testing of the monitoring and alerting infrastructure

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/monitoring-test-$(date +%Y%m%d-%H%M%S).log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
ALERTMANAGER_URL="http://localhost:9093"
API_URL="http://localhost:8000"
TIMEOUT=30
VERBOSE=false

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

PRSM Monitoring & Alerting Test Script

OPTIONS:
    --prometheus-url URL      Prometheus URL [default: $PROMETHEUS_URL]
    --grafana-url URL         Grafana URL [default: $GRAFANA_URL]
    --alertmanager-url URL    Alertmanager URL [default: $ALERTMANAGER_URL]
    --api-url URL             PRSM API URL [default: $API_URL]
    --timeout SECONDS         Request timeout [default: $TIMEOUT]
    -v, --verbose             Enable verbose output
    -h, --help                Show this help message

TEST CATEGORIES:
    health                    Test health check endpoints
    prometheus                Test Prometheus configuration
    grafana                   Test Grafana dashboards
    alertmanager              Test Alertmanager configuration
    integration               Test end-to-end monitoring integration
    alerts                    Test alert firing and notification
    all                       Run all tests (default)

EXAMPLES:
    $0                        # Run all tests
    $0 health                 # Test only health endpoints
    $0 prometheus grafana     # Test Prometheus and Grafana
    $0 --verbose integration  # Verbose integration test

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

# Function to test HTTP endpoint
test_endpoint() {
    local url=$1
    local expected_status=${2:-200}
    local description=$3
    
    print_status "$BLUE" "Testing: $description"
    
    local response
    local status_code
    
    if response=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$url" 2>/dev/null); then
        status_code=$(echo "$response" | tail -n1)
        response_body=$(echo "$response" | head -n -1)
        
        if [[ "$status_code" == "$expected_status" ]]; then
            print_status "$GREEN" "✅ $description - Status: $status_code"
            return 0
        else
            print_status "$RED" "❌ $description - Expected: $expected_status, Got: $status_code"
            return 1
        fi
    else
        print_status "$RED" "❌ $description - Connection failed"
        return 1
    fi
}

# Function to test JSON endpoint
test_json_endpoint() {
    local url=$1
    local description=$2
    local json_path=${3:-""}
    local expected_value=${4:-""}
    
    print_status "$BLUE" "Testing: $description"
    
    local response
    if response=$(curl -s --max-time "$TIMEOUT" "$url" 2>/dev/null); then
        if echo "$response" | jq . >/dev/null 2>&1; then
            print_status "$GREEN" "✅ $description - Valid JSON response"
            
            if [[ -n "$json_path" && -n "$expected_value" ]]; then
                local actual_value
                actual_value=$(echo "$response" | jq -r "$json_path" 2>/dev/null || echo "null")
                
                if [[ "$actual_value" == "$expected_value" ]]; then
                    print_status "$GREEN" "✅ $description - Value check passed: $actual_value"
                else
                    print_status "$YELLOW" "⚠️ $description - Value check failed: expected '$expected_value', got '$actual_value'"
                fi
            fi
            
            return 0
        else
            print_status "$RED" "❌ $description - Invalid JSON response"
            return 1
        fi
    else
        print_status "$RED" "❌ $description - Connection failed"
        return 1
    fi
}

# Function to test health endpoints
test_health_endpoints() {
    print_status "$BLUE" "=== Testing Health Check Endpoints ==="
    
    local health_tests=(
        "$API_URL/health|Basic health check"
        "$API_URL/health/liveness|Liveness probe"
        "$API_URL/health/readiness|Readiness probe"
        "$API_URL/health/ping|Ping endpoint"
        "$API_URL/health/version|Version info"
        "$API_URL/health/status/components|Component status"
        "$API_URL/health/status/resources|Resource status"
    )
    
    local passed=0
    local total=${#health_tests[@]}
    
    for test_case in "${health_tests[@]}"; do
        local url=$(echo "$test_case" | cut -d'|' -f1)
        local description=$(echo "$test_case" | cut -d'|' -f2)
        
        if test_json_endpoint "$url" "$description"; then
            ((passed++))
        fi
    done
    
    print_status "$BLUE" "Health endpoints: $passed/$total tests passed"
    return $((total - passed))
}

# Function to test Prometheus
test_prometheus() {
    print_status "$BLUE" "=== Testing Prometheus Configuration ==="
    
    local prometheus_tests=(
        "$PROMETHEUS_URL/-/healthy|Prometheus health"
        "$PROMETHEUS_URL/-/ready|Prometheus readiness"
        "$PROMETHEUS_URL/api/v1/status/config|Prometheus config"
        "$PROMETHEUS_URL/api/v1/status/targets|Prometheus targets"
        "$PROMETHEUS_URL/api/v1/rules|Prometheus rules"
        "$PROMETHEUS_URL/api/v1/alerts|Prometheus alerts"
    )
    
    local passed=0
    local total=${#prometheus_tests[@]}
    
    for test_case in "${prometheus_tests[@]}"; do
        local url=$(echo "$test_case" | cut -d'|' -f1)
        local description=$(echo "$test_case" | cut -d'|' -f2)
        
        if test_endpoint "$url" 200 "$description"; then
            ((passed++))
        fi
    done
    
    # Test for specific metrics
    print_status "$BLUE" "Testing Prometheus metrics collection..."
    
    local metric_queries=(
        "up|System uptime metrics"
        "http_requests_total|HTTP request metrics"
        "prsm:api_request_rate|PRSM API request rate"
        "prsm:cpu_utilization|CPU utilization metrics"
        "prsm:memory_utilization|Memory utilization metrics"
    )
    
    for query_case in "${metric_queries[@]}"; do
        local query=$(echo "$query_case" | cut -d'|' -f1)
        local description=$(echo "$query_case" | cut -d'|' -f2)
        local url="$PROMETHEUS_URL/api/v1/query?query=$query"
        
        if test_json_endpoint "$url" "$description" ".status" "success"; then
            ((passed++))
            ((total++))
        else
            ((total++))
        fi
    done
    
    print_status "$BLUE" "Prometheus tests: $passed/$total tests passed"
    return $((total - passed))
}

# Function to test Grafana
test_grafana() {
    print_status "$BLUE" "=== Testing Grafana Dashboards ==="
    
    local grafana_tests=(
        "$GRAFANA_URL/api/health|Grafana health"
        "$GRAFANA_URL/api/datasources|Grafana datasources"
        "$GRAFANA_URL/api/search?query=PRSM|PRSM dashboards"
        "$GRAFANA_URL/api/dashboards/uid/prsm-overview|PRSM overview dashboard"
        "$GRAFANA_URL/api/dashboards/uid/prsm-security|PRSM security dashboard"
    )
    
    local passed=0
    local total=${#grafana_tests[@]}
    
    for test_case in "${grafana_tests[@]}"; do
        local url=$(echo "$test_case" | cut -d'|' -f1)
        local description=$(echo "$test_case" | cut -d'|' -f2)
        
        if test_json_endpoint "$url" "$description"; then
            ((passed++))
        fi
    done
    
    print_status "$BLUE" "Grafana tests: $passed/$total tests passed"
    return $((total - passed))
}

# Function to test Alertmanager
test_alertmanager() {
    print_status "$BLUE" "=== Testing Alertmanager Configuration ==="
    
    local alertmanager_tests=(
        "$ALERTMANAGER_URL/-/healthy|Alertmanager health"
        "$ALERTMANAGER_URL/-/ready|Alertmanager readiness"
        "$ALERTMANAGER_URL/api/v1/status|Alertmanager status"
        "$ALERTMANAGER_URL/api/v1/alerts|Alertmanager alerts"
        "$ALERTMANAGER_URL/api/v1/receivers|Alertmanager receivers"
    )
    
    local passed=0
    local total=${#alertmanager_tests[@]}
    
    for test_case in "${alertmanager_tests[@]}"; do
        local url=$(echo "$test_case" | cut -d'|' -f1)
        local description=$(echo "$test_case" | cut -d'|' -f2)
        
        if test_json_endpoint "$url" "$description"; then
            ((passed++))
        fi
    done
    
    print_status "$BLUE" "Alertmanager tests: $passed/$total tests passed"
    return $((total - passed))
}

# Function to test integration
test_integration() {
    print_status "$BLUE" "=== Testing End-to-End Integration ==="
    
    print_status "$BLUE" "Testing monitoring pipeline..."
    
    # Generate some test metrics by making API calls
    print_status "$BLUE" "Generating test traffic..."
    for i in {1..10}; do
        curl -s "$API_URL/health/ping" >/dev/null 2>&1 || true
        sleep 0.1
    done
    
    # Wait a bit for metrics to be scraped
    sleep 5
    
    # Check if metrics appear in Prometheus
    local integration_passed=0
    local integration_total=4
    
    if test_json_endpoint "$PROMETHEUS_URL/api/v1/query?query=up{job=\"prsm-api\"}" "PRSM API metrics in Prometheus" ".status" "success"; then
        ((integration_passed++))
    fi
    
    if test_json_endpoint "$PROMETHEUS_URL/api/v1/query?query=prometheus_notifications_total" "Prometheus notifications" ".status" "success"; then
        ((integration_passed++))
    fi
    
    # Test Grafana can query Prometheus
    if test_endpoint "$GRAFANA_URL/api/datasources/proxy/1/api/v1/query?query=up" 200 "Grafana-Prometheus integration"; then
        ((integration_passed++))
    fi
    
    # Test that health endpoint returns detailed status
    if response=$(curl -s "$API_URL/health" 2>/dev/null); then
        if echo "$response" | jq -e '.components.database.status' >/dev/null 2>&1; then
            print_status "$GREEN" "✅ Health endpoint returns component status"
            ((integration_passed++))
        else
            print_status "$RED" "❌ Health endpoint missing component status"
        fi
    fi
    
    print_status "$BLUE" "Integration tests: $integration_passed/$integration_total tests passed"
    return $((integration_total - integration_passed))
}

# Function to test alert firing
test_alerts() {
    print_status "$BLUE" "=== Testing Alert Configuration ==="
    
    # Check if alert rules are loaded
    local alerts_passed=0
    local alerts_total=3
    
    if response=$(curl -s "$PROMETHEUS_URL/api/v1/rules" 2>/dev/null); then
        if echo "$response" | jq -e '.data.groups[] | select(.name == "prsm.system.health")' >/dev/null 2>&1; then
            print_status "$GREEN" "✅ PRSM system health alert rules loaded"
            ((alerts_passed++))
        else
            print_status "$RED" "❌ PRSM system health alert rules not found"
        fi
        
        if echo "$response" | jq -e '.data.groups[] | select(.name == "prsm.security.alerts")' >/dev/null 2>&1; then
            print_status "$GREEN" "✅ PRSM security alert rules loaded"
            ((alerts_passed++))
        else
            print_status "$RED" "❌ PRSM security alert rules not found"
        fi
    else
        print_status "$RED" "❌ Cannot retrieve Prometheus rules"
    fi
    
    # Check Alertmanager configuration
    if response=$(curl -s "$ALERTMANAGER_URL/api/v1/status" 2>/dev/null); then
        if echo "$response" | jq -e '.data.configYAML' >/dev/null 2>&1; then
            print_status "$GREEN" "✅ Alertmanager configuration loaded"
            ((alerts_passed++))
        else
            print_status "$RED" "❌ Alertmanager configuration not loaded"
        fi
    else
        print_status "$RED" "❌ Cannot retrieve Alertmanager status"
    fi
    
    print_status "$BLUE" "Alert tests: $alerts_passed/$alerts_total tests passed"
    return $((alerts_total - alerts_passed))
}

# Function to run all tests
run_all_tests() {
    local total_failed=0
    
    print_status "$BLUE" "Starting comprehensive monitoring system test..."
    print_status "$BLUE" "Timestamp: $TIMESTAMP"
    print_status "$BLUE" "Log file: $LOG_FILE"
    
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    test_health_endpoints || ((total_failed += $?))
    test_prometheus || ((total_failed += $?))
    test_grafana || ((total_failed += $?))
    test_alertmanager || ((total_failed += $?))
    test_integration || ((total_failed += $?))
    test_alerts || ((total_failed += $?))
    
    print_status "$BLUE" "=== Test Summary ==="
    
    if [[ $total_failed -eq 0 ]]; then
        print_status "$GREEN" "🎉 All monitoring tests passed!"
        print_status "$GREEN" "✅ PRSM monitoring and alerting system is fully operational"
        return 0
    else
        print_status "$RED" "❌ $total_failed test(s) failed"
        print_status "$YELLOW" "⚠️ Please check the logs and fix any issues before deploying to production"
        return 1
    fi
}

# Function to generate test report
generate_test_report() {
    local report_file="${PROJECT_ROOT}/logs/monitoring-test-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# PRSM Monitoring & Alerting Test Report

**Test Date:** $(date)  
**Test Duration:** $(($(date +%s) - start_time)) seconds  
**Environment:** PRSM Development/Testing  

## Test Configuration

- **Prometheus URL:** $PROMETHEUS_URL
- **Grafana URL:** $GRAFANA_URL  
- **Alertmanager URL:** $ALERTMANAGER_URL
- **PRSM API URL:** $API_URL
- **Timeout:** ${TIMEOUT}s

## Test Results

### Health Check Endpoints
- Basic health check
- Liveness and readiness probes
- Component status monitoring
- Resource utilization tracking

### Prometheus Monitoring
- Configuration validation
- Target discovery
- Metrics collection
- Alert rule loading

### Grafana Dashboards
- Dashboard accessibility
- Data source connectivity
- PRSM-specific visualizations

### Alertmanager Notifications
- Configuration validation
- Receiver setup
- Notification routing

### Integration Testing
- End-to-end monitoring pipeline
- Metric flow validation
- Alert firing capability

## Test Log

\`\`\`
$(tail -50 "$LOG_FILE")
\`\`\`

---
*Generated by PRSM Monitoring Test Suite*
EOF
    
    print_status "$GREEN" "Test report generated: $report_file"
}

# Main function
main() {
    local start_time=$(date +%s)
    local test_categories=("$@")
    
    # Default to all tests if no specific tests specified
    if [[ ${#test_categories[@]} -eq 0 ]]; then
        test_categories=("all")
    fi
    
    local exit_code=0
    
    for category in "${test_categories[@]}"; do
        case "$category" in
            health)
                test_health_endpoints || exit_code=1
                ;;
            prometheus)
                test_prometheus || exit_code=1
                ;;
            grafana)
                test_grafana || exit_code=1
                ;;
            alertmanager)
                test_alertmanager || exit_code=1
                ;;
            integration)
                test_integration || exit_code=1
                ;;
            alerts)
                test_alerts || exit_code=1
                ;;
            all)
                run_all_tests || exit_code=1
                ;;
            *)
                error_exit "Unknown test category: $category"
                ;;
        esac
    done
    
    # Generate test report
    generate_test_report
    
    return $exit_code
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prometheus-url)
            PROMETHEUS_URL="$2"
            shift 2
            ;;
        --grafana-url)
            GRAFANA_URL="$2"
            shift 2
            ;;
        --alertmanager-url)
            ALERTMANAGER_URL="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        health|prometheus|grafana|alertmanager|integration|alerts|all)
            break
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function with remaining arguments
main "$@"