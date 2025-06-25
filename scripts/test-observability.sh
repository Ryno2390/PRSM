#!/bin/bash

# PRSM Observability Stack Testing Script
# Validates monitoring, logging, and tracing functionality

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROMETHEUS_URL=${PROMETHEUS_URL:-"http://localhost:9090"}
GRAFANA_URL=${GRAFANA_URL:-"http://localhost:3000"}
JAEGER_URL=${JAEGER_URL:-"http://localhost:16686"}
LOKI_URL=${LOKI_URL:-"http://localhost:3100"}
METRICS_EXPORTER_URL=${METRICS_EXPORTER_URL:-"http://localhost:9091"}
TEST_DURATION=${TEST_DURATION:-"300"}  # 5 minutes

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

check_service_health() {
    local service_name=$1
    local url=$2
    local endpoint=${3:-"/"}
    
    log_info "Checking $service_name health..."
    
    if curl -s --max-time 10 "$url$endpoint" > /dev/null 2>&1; then
        log_success "$service_name is healthy"
        return 0
    else
        log_error "$service_name is not responding"
        return 1
    fi
}

test_prometheus_metrics() {
    log_info "Testing Prometheus metrics collection..."
    
    # Check if Prometheus is collecting PRSM metrics
    local metrics_query="up{service=~\"prsm-.*\"}"
    local response=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=$metrics_query")
    
    if echo "$response" | jq -e '.data.result | length > 0' > /dev/null 2>&1; then
        log_success "Prometheus is collecting PRSM metrics"
        
        # Show available metrics
        log_info "Available PRSM services:"
        echo "$response" | jq -r '.data.result[].metric.service' | sort | uniq
    else
        log_error "Prometheus is not collecting PRSM metrics"
        return 1
    fi
    
    # Test custom PRSM metrics
    local custom_metrics=(
        "prsm_concurrent_sessions_total"
        "ftns_transactions_per_second"
        "agent_task_queue_depth"
        "p2p_active_connections"
    )
    
    for metric in "${custom_metrics[@]}"; do
        local metric_response=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=$metric")
        if echo "$metric_response" | jq -e '.data.result | length > 0' > /dev/null 2>&1; then
            log_success "Custom metric $metric is available"
        else
            log_warning "Custom metric $metric is not available"
        fi
    done
}

test_grafana_dashboards() {
    log_info "Testing Grafana dashboard access..."
    
    # Check if Grafana is accessible
    if ! check_service_health "Grafana" "$GRAFANA_URL" "/api/health"; then
        return 1
    fi
    
    # Test dashboard API (requires authentication)
    local dashboard_response=$(curl -s "$GRAFANA_URL/api/search" \
        -u "admin:prsm_admin" 2>/dev/null || echo '[]')
    
    local dashboard_count=$(echo "$dashboard_response" | jq 'length' 2>/dev/null || echo "0")
    
    if [ "$dashboard_count" -gt 0 ]; then
        log_success "Found $dashboard_count Grafana dashboards"
    else
        log_warning "No Grafana dashboards found or authentication failed"
    fi
}

test_jaeger_tracing() {
    log_info "Testing Jaeger tracing functionality..."
    
    if ! check_service_health "Jaeger" "$JAEGER_URL" "/api/services"; then
        return 1
    fi
    
    # Check for PRSM services in Jaeger
    local services_response=$(curl -s "$JAEGER_URL/api/services")
    
    if echo "$services_response" | jq -e '.data | length > 0' > /dev/null 2>&1; then
        log_success "Jaeger is collecting traces"
        
        # Show available services
        log_info "Services with traces:"
        echo "$services_response" | jq -r '.data[]' | grep -i prsm || echo "No PRSM services found"
    else
        log_warning "No traces found in Jaeger"
    fi
}

test_loki_logging() {
    log_info "Testing Loki log aggregation..."
    
    if ! check_service_health "Loki" "$LOKI_URL" "/ready"; then
        return 1
    fi
    
    # Query for PRSM logs
    local log_query='%7Bservice%3D%22prsm-api%22%7D'  # URL encoded {service="prsm-api"}
    local logs_response=$(curl -s "$LOKI_URL/loki/api/v1/query?query=$log_query&limit=10")
    
    if echo "$logs_response" | jq -e '.data.result | length > 0' > /dev/null 2>&1; then
        log_success "Loki is collecting PRSM logs"
        
        # Show log count
        local log_count=$(echo "$logs_response" | jq '.data.result | length')
        log_info "Found $log_count log streams"
    else
        log_warning "No PRSM logs found in Loki"
    fi
}

test_custom_metrics_exporter() {
    log_info "Testing PRSM custom metrics exporter..."
    
    if ! check_service_health "Metrics Exporter" "$METRICS_EXPORTER_URL" "/metrics"; then
        return 1
    fi
    
    # Check for PRSM-specific metrics
    local metrics_response=$(curl -s "$METRICS_EXPORTER_URL/metrics")
    
    local prsm_metrics=(
        "prsm_query_processing_duration_seconds"
        "ftns_cost_calculation_duration_seconds"
        "agent_task_queue_depth"
        "circuit_breaker_activations_total"
    )
    
    local found_metrics=0
    for metric in "${prsm_metrics[@]}"; do
        if echo "$metrics_response" | grep -q "$metric"; then
            log_success "Found custom metric: $metric"
            ((found_metrics++))
        else
            log_warning "Missing custom metric: $metric"
        fi
    done
    
    if [ $found_metrics -gt 0 ]; then
        log_success "Custom metrics exporter is working ($found_metrics/$((${#prsm_metrics[@]})) metrics)"
    else
        log_error "Custom metrics exporter is not working properly"
        return 1
    fi
}

test_alerting_rules() {
    log_info "Testing Prometheus alerting rules..."
    
    # Check alerting rules
    local rules_response=$(curl -s "$PROMETHEUS_URL/api/v1/rules")
    
    if echo "$rules_response" | jq -e '.data.groups | length > 0' > /dev/null 2>&1; then
        local rule_count=$(echo "$rules_response" | jq '[.data.groups[].rules[]] | length')
        log_success "Found $rule_count alerting rules"
        
        # Check for PRSM-specific rules
        local prsm_rules=$(echo "$rules_response" | jq -r '[.data.groups[] | select(.name | contains("prsm")) | .name] | @json')
        if [ "$prsm_rules" != "[]" ]; then
            log_success "PRSM-specific alerting rules found"
            echo "$prsm_rules" | jq -r '.[]' | while read rule; do
                log_info "  - $rule"
            done
        else
            log_warning "No PRSM-specific alerting rules found"
        fi
    else
        log_warning "No alerting rules configured"
    fi
}

test_phase1_metrics() {
    log_info "Testing Phase 1 validation metrics..."
    
    # Test Phase 1 specific metrics
    local phase1_queries=(
        "prsm_concurrent_sessions_total"
        "histogram_quantile(0.95, rate(prsm_query_processing_duration_seconds_bucket[5m]))"
        "histogram_quantile(0.99, rate(ftns_cost_calculation_duration_seconds_bucket[5m]))"
        "rate(http_requests_total{service=\"prsm-api\"}[5m])"
    )
    
    local phase1_names=(
        "Concurrent Sessions"
        "Query Latency P95"
        "FTNS Calculation Latency P99"
        "API Throughput"
    )
    
    for i in "${!phase1_queries[@]}"; do
        local query="${phase1_queries[$i]}"
        local name="${phase1_names[$i]}"
        
        local encoded_query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$query'))")
        local response=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=$encoded_query")
        
        if echo "$response" | jq -e '.data.result | length > 0' > /dev/null 2>&1; then
            local value=$(echo "$response" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
            log_success "$name: $value"
        else
            log_warning "$name: No data available"
        fi
    done
}

generate_test_load() {
    log_info "Generating test load to validate metrics collection..."
    
    # Simple load generation to create metrics
    local prsm_api_url="http://localhost:8000"
    
    # Check if PRSM API is available
    if curl -s --max-time 5 "$prsm_api_url/health" > /dev/null 2>&1; then
        log_info "Generating test requests for $TEST_DURATION seconds..."
        
        # Generate load in background
        for i in {1..10}; do
            (
                local end_time=$(($(date +%s) + TEST_DURATION))
                while [ $(date +%s) -lt $end_time ]; do
                    curl -s "$prsm_api_url/health" > /dev/null 2>&1 || true
                    curl -s "$prsm_api_url/metrics" > /dev/null 2>&1 || true
                    sleep 1
                done
            ) &
        done
        
        # Wait for load generation
        sleep $((TEST_DURATION + 10))
        
        # Kill any remaining background processes
        jobs -p | xargs -r kill > /dev/null 2>&1 || true
        
        log_success "Test load generation completed"
    else
        log_warning "PRSM API not available, skipping load generation"
    fi
}

run_comprehensive_test() {
    log_info "Running comprehensive observability stack test..."
    
    local failed_tests=0
    local total_tests=0
    
    # Test each component
    local tests=(
        "test_prometheus_metrics"
        "test_grafana_dashboards"
        "test_jaeger_tracing"
        "test_loki_logging"
        "test_custom_metrics_exporter"
        "test_alerting_rules"
        "test_phase1_metrics"
    )
    
    for test_func in "${tests[@]}"; do
        ((total_tests++))
        echo ""
        if ! $test_func; then
            ((failed_tests++))
        fi
    done
    
    # Generate summary
    echo ""
    log_info "=== OBSERVABILITY TEST SUMMARY ==="
    echo "Total tests: $total_tests"
    echo "Passed: $((total_tests - failed_tests))"
    echo "Failed: $failed_tests"
    
    if [ $failed_tests -eq 0 ]; then
        log_success "All observability tests passed!"
        return 0
    else
        log_warning "$failed_tests test(s) failed"
        return 1
    fi
}

show_monitoring_urls() {
    echo ""
    log_info "=== MONITORING DASHBOARD URLS ==="
    echo "Grafana:    $GRAFANA_URL (admin/prsm_admin)"
    echo "Prometheus: $PROMETHEUS_URL"
    echo "Jaeger:     $JAEGER_URL"
    echo "Kibana:     http://localhost:5601"
    echo "Metrics:    $METRICS_EXPORTER_URL/metrics"
    echo ""
}

main() {
    log_info "Starting PRSM Observability Stack Test"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --generate-load)
                GENERATE_LOAD=true
                shift
                ;;
            --duration)
                TEST_DURATION="$2"
                shift 2
                ;;
            --urls-only)
                show_monitoring_urls
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--generate-load] [--duration <seconds>] [--urls-only]"
                exit 1
                ;;
        esac
    done
    
    # Show monitoring URLs
    show_monitoring_urls
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Generate load if requested
    if [ "${GENERATE_LOAD:-false}" = "true" ]; then
        generate_test_load
    fi
    
    # Run comprehensive test
    if run_comprehensive_test; then
        log_success "Observability stack validation completed successfully!"
        exit 0
    else
        log_error "Observability stack validation failed"
        exit 1
    fi
}

# Run main function
main "$@"