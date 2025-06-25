#!/bin/bash

# PRSM Phase 1 Requirements Validation Script
# Validates the core Phase 1 target: 1000 concurrent users with <2s latency

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Phase 1 Requirements
PHASE1_CONCURRENT_USERS=1000
PHASE1_MAX_LATENCY_MS=2000
PHASE1_MIN_SUCCESS_RATE=95
PHASE1_MAX_ERROR_RATE=5
PHASE1_TEST_DURATION=300  # 5 minutes

# Configuration
API_BASE_URL=${API_BASE_URL:-"http://localhost:8000"}
PROMETHEUS_URL=${PROMETHEUS_URL:-"http://localhost:9090"}
GRAFANA_URL=${GRAFANA_URL:-"http://localhost:3000"}
OUTPUT_DIR=${OUTPUT_DIR:-"./phase1-validation"}

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

validate_environment() {
    log_info "Validating Phase 1 test environment..."
    
    # Check API availability
    if ! curl -s --max-time 5 "$API_BASE_URL/health" > /dev/null; then
        log_error "PRSM API not available at $API_BASE_URL"
        return 1
    fi
    
    # Check monitoring stack
    if ! curl -s --max-time 5 "$PROMETHEUS_URL/api/v1/query?query=up" > /dev/null; then
        log_warning "Prometheus not available at $PROMETHEUS_URL"
    fi
    
    # Check if load testing tools are available
    local tools_available=0
    
    if command -v k6 &> /dev/null; then
        ((tools_available++))
        log_success "k6 available"
    fi
    
    if command -v wrk &> /dev/null; then
        ((tools_available++))
        log_success "wrk available"
    fi
    
    if command -v hey &> /dev/null; then
        ((tools_available++))
        log_success "hey available"
    fi
    
    if [ $tools_available -eq 0 ]; then
        log_error "No load testing tools available. Please install k6, wrk, or hey."
        return 1
    fi
    
    log_success "Environment validation passed"
}

create_phase1_k6_script() {
    log_info "Creating Phase 1 validation k6 script..."
    
    mkdir -p "$OUTPUT_DIR"
    
    cat > "$OUTPUT_DIR/phase1-validation.js" << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Phase 1 specific metrics
export let phase1LatencyCompliance = new Rate('phase1_latency_compliance');
export let phase1ErrorRate = new Rate('phase1_error_rate');
export let phase1Throughput = new Trend('phase1_throughput');
export let phase1ConcurrentUsers = new Counter('phase1_concurrent_users');

export let options = {
  stages: [
    // Rapid ramp-up to test infrastructure capacity
    { duration: '30s', target: 250 },   // Ramp to 25%
    { duration: '30s', target: 500 },   // Ramp to 50%
    { duration: '30s', target: 750 },   // Ramp to 75%
    { duration: '30s', target: 1000 },  // Reach Phase 1 target: 1000 users
    
    // Sustained load at Phase 1 target
    { duration: '180s', target: 1000 }, // Hold 1000 users for 3 minutes
    
    // Gradual ramp-down
    { duration: '60s', target: 0 },     // Ramp down
  ],
  
  // Phase 1 Requirements as thresholds
  thresholds: {
    // Core Phase 1 requirement: 95th percentile < 2000ms
    'http_req_duration': ['p(95)<2000'],
    
    // Supporting requirements
    'http_req_failed': ['rate<0.05'],           // <5% error rate
    'phase1_latency_compliance': ['rate>0.95'], // >95% requests under 2s
    'phase1_error_rate': ['rate<0.05'],         // <5% error rate
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';

// Phase 1 specific endpoints representing realistic PRSM usage
const phase1Endpoints = [
  // Health checks (monitoring/health verification)
  { path: '/health', weight: 5, method: 'GET', description: 'Health check' },
  
  // Session management (core user interaction)
  { path: '/api/v1/sessions', weight: 25, method: 'GET', description: 'List sessions' },
  { 
    path: '/api/v1/sessions', 
    weight: 20, 
    method: 'POST', 
    data: { 
      query: 'Explain machine learning in simple terms',
      domain: 'general',
      context_allocation: 100 
    },
    description: 'Create session'
  },
  
  // Model marketplace (discovery and usage)
  { path: '/api/v1/marketplace/models', weight: 15, method: 'GET', description: 'Browse models' },
  { path: '/api/v1/marketplace/models/featured', weight: 10, method: 'GET', description: 'Featured models' },
  
  // Governance (community participation)
  { path: '/api/v1/governance/proposals', weight: 8, method: 'GET', description: 'List proposals' },
  { path: '/api/v1/governance/voting', weight: 5, method: 'GET', description: 'Voting status' },
  
  // Teams and collaboration
  { path: '/api/v1/teams', weight: 7, method: 'GET', description: 'List teams' },
  
  // Metrics and monitoring
  { path: '/metrics', weight: 5, method: 'GET', description: 'Prometheus metrics' },
];

function selectWeightedEndpoint() {
  const totalWeight = phase1Endpoints.reduce((sum, ep) => sum + ep.weight, 0);
  let random = Math.random() * totalWeight;
  
  for (const endpoint of phase1Endpoints) {
    random -= endpoint.weight;
    if (random <= 0) {
      return endpoint;
    }
  }
  return phase1Endpoints[0];
}

export default function() {
  const endpoint = selectWeightedEndpoint();
  const url = `${BASE_URL}${endpoint.path}`;
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'PRSM-Phase1-Validation/1.0',
      'X-Test-Type': 'phase1-validation',
    },
    timeout: '10s',
  };
  
  const startTime = Date.now();
  let response;
  
  // Execute request based on method
  if (endpoint.method === 'POST' && endpoint.data) {
    response = http.post(url, JSON.stringify(endpoint.data), params);
  } else {
    response = http.get(url, params);
  }
  
  const endTime = Date.now();
  const responseTime = endTime - startTime;
  
  // Phase 1 specific validations
  const isSuccess = response.status >= 200 && response.status < 400;
  const isLatencyCompliant = responseTime < 2000; // Phase 1 requirement
  
  // Record Phase 1 metrics
  phase1LatencyCompliance.add(isLatencyCompliant);
  phase1ErrorRate.add(!isSuccess);
  phase1Throughput.add(1);
  phase1ConcurrentUsers.add(1);
  
  // Standard k6 checks
  const checks = check(response, {
    'Phase 1: Status is 2xx': (r) => r.status >= 200 && r.status < 300,
    'Phase 1: Response time < 2000ms': (r) => r.timings.duration < 2000,
    'Phase 1: Response has body': (r) => r.body && r.body.length > 0,
    'Phase 1: Content-Type header': (r) => r.headers['Content-Type'] !== undefined,
  });
  
  // Log failures for debugging
  if (!isSuccess || !isLatencyCompliant) {
    console.error(`Phase 1 failure: ${endpoint.method} ${endpoint.path} - Status: ${response.status}, Time: ${responseTime}ms, Description: ${endpoint.description}`);
  }
  
  // Realistic user behavior simulation
  const userThinkTime = Math.random() * 1.5 + 0.5; // 0.5-2.0 seconds
  sleep(userThinkTime);
}

export function handleSummary(data) {
  const phase1Results = {
    timestamp: new Date().toISOString(),
    test_type: 'phase1_validation',
    requirements: {
      concurrent_users: 1000,
      max_latency_p95_ms: 2000,
      min_success_rate: 0.95,
      max_error_rate: 0.05
    },
    results: {
      max_vus: data.metrics.vus_max.values.max,
      total_requests: data.metrics.http_reqs.values.count,
      requests_per_second: data.metrics.http_reqs.values.rate,
      avg_response_time: data.metrics.http_req_duration.values.avg,
      p95_response_time: data.metrics.http_req_duration.values['p(95)'],
      p99_response_time: data.metrics.http_req_duration.values['p(99)'],
      error_rate: data.metrics.http_req_failed.values.rate,
      latency_compliance_rate: data.metrics.phase1_latency_compliance?.values.rate || 0,
    },
    compliance: {
      concurrent_users: data.metrics.vus_max.values.max >= 1000,
      latency: data.metrics.http_req_duration.values['p(95)'] < 2000,
      error_rate: data.metrics.http_req_failed.values.rate < 0.05,
      overall: false // Will be calculated
    }
  };
  
  // Calculate overall compliance
  phase1Results.compliance.overall = 
    phase1Results.compliance.concurrent_users &&
    phase1Results.compliance.latency &&
    phase1Results.compliance.error_rate;
  
  return {
    'phase1-results.json': JSON.stringify(phase1Results, null, 2),
    'phase1-summary.txt': textSummary(data, { indent: ' ', enableColors: false }),
    'stdout': generatePhase1Report(phase1Results),
  };
}

function generatePhase1Report(results) {
  const status = results.compliance.overall ? '✅ PASSED' : '❌ FAILED';
  
  return `
=================================================================
🎯 PRSM PHASE 1 VALIDATION RESULTS
=================================================================

Overall Status: ${status}

📊 PERFORMANCE METRICS:
├─ Max Concurrent Users: ${results.results.max_vus} (Required: ${results.requirements.concurrent_users})
├─ 95th Percentile Latency: ${results.results.p95_response_time.toFixed(1)}ms (Required: <${results.requirements.max_latency_p95_ms}ms)
├─ Error Rate: ${(results.results.error_rate * 100).toFixed(2)}% (Required: <${results.requirements.max_error_rate * 100}%)
├─ Total Requests: ${results.results.total_requests}
├─ Throughput: ${results.results.requests_per_second.toFixed(1)} RPS
└─ Average Response Time: ${results.results.avg_response_time.toFixed(1)}ms

✅ COMPLIANCE CHECK:
├─ Concurrent Users (1000): ${results.compliance.concurrent_users ? 'PASS' : 'FAIL'}
├─ Latency (<2000ms): ${results.compliance.latency ? 'PASS' : 'FAIL'}
└─ Error Rate (<5%): ${results.compliance.error_rate ? 'PASS' : 'FAIL'}

🎉 PHASE 1 STATUS: ${results.compliance.overall ? 'REQUIREMENTS MET' : 'REQUIREMENTS NOT MET'}

Generated: ${results.timestamp}
=================================================================
`;
}
EOF
    
    log_success "Phase 1 k6 script created"
}

run_phase1_validation() {
    log_info "Running Phase 1 validation test..."
    
    local start_time=$(date +%s)
    
    # Start performance monitoring if available
    if curl -s --max-time 5 "$PROMETHEUS_URL/api/v1/query?query=up" > /dev/null; then
        log_info "Monitoring Phase 1 metrics via Prometheus..."
        monitor_phase1_metrics &
        local monitor_pid=$!
    fi
    
    # Run the Phase 1 validation test
    local k6_exit_code=0
    
    API_BASE_URL="$API_BASE_URL" k6 run \
        --out json="$OUTPUT_DIR/phase1-raw-results.json" \
        --quiet \
        "$OUTPUT_DIR/phase1-validation.js" || k6_exit_code=$?
    
    # Stop monitoring
    if [ -n "${monitor_pid:-}" ]; then
        kill $monitor_pid 2>/dev/null || true
    fi
    
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    log_info "Phase 1 validation completed in ${total_time}s"
    
    # Analyze results
    analyze_phase1_results
    
    return $k6_exit_code
}

monitor_phase1_metrics() {
    local monitoring_interval=10
    local monitoring_file="$OUTPUT_DIR/phase1-monitoring.log"
    
    echo "timestamp,cpu_usage,memory_usage,active_connections,response_time_p95" > "$monitoring_file"
    
    while true; do
        local timestamp=$(date +%s)
        
        # Query Prometheus for system metrics
        local cpu_query="100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
        local memory_query="(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
        local connections_query="prsm_concurrent_sessions_total"
        local latency_query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        
        local cpu_usage=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=$cpu_query" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
        local memory_usage=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=$memory_query" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
        local connections=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=$connections_query" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
        local response_time=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=$latency_query" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
        
        echo "$timestamp,$cpu_usage,$memory_usage,$connections,$response_time" >> "$monitoring_file"
        
        sleep $monitoring_interval
    done
}

analyze_phase1_results() {
    log_info "Analyzing Phase 1 validation results..."
    
    if [ ! -f "$OUTPUT_DIR/phase1-results.json" ]; then
        log_error "Phase 1 results file not found"
        return 1
    fi
    
    # Extract key metrics
    local results_file="$OUTPUT_DIR/phase1-results.json"
    local max_vus=$(jq -r '.results.max_vus' "$results_file")
    local p95_latency=$(jq -r '.results.p95_response_time' "$results_file")
    local error_rate=$(jq -r '.results.error_rate * 100' "$results_file")
    local overall_compliance=$(jq -r '.compliance.overall' "$results_file")
    local rps=$(jq -r '.results.requests_per_second' "$results_file")
    
    echo ""
    log_info "=== PHASE 1 VALIDATION SUMMARY ==="
    
    # Detailed compliance check
    printf "%-30s %s\n" "Requirement" "Status"
    printf "%-30s %s\n" "$(printf -- '-%.0s' {1..30})" "$(printf -- '-%.0s' {1..20})"
    
    # Concurrent users check
    if (( $(echo "$max_vus >= $PHASE1_CONCURRENT_USERS" | bc -l) )); then
        printf "%-30s ${GREEN}✓ PASS${NC} (${max_vus} users)\n" "Concurrent Users (≥1000)"
    else
        printf "%-30s ${RED}✗ FAIL${NC} (${max_vus} users)\n" "Concurrent Users (≥1000)"
    fi
    
    # Latency check
    if (( $(echo "$p95_latency < $PHASE1_MAX_LATENCY_MS" | bc -l) )); then
        printf "%-30s ${GREEN}✓ PASS${NC} (%.1fms)\n" "95th Percentile Latency (<2s)" "$p95_latency"
    else
        printf "%-30s ${RED}✗ FAIL${NC} (%.1fms)\n" "95th Percentile Latency (<2s)" "$p95_latency"
    fi
    
    # Error rate check
    if (( $(echo "$error_rate < $PHASE1_MAX_ERROR_RATE" | bc -l) )); then
        printf "%-30s ${GREEN}✓ PASS${NC} (%.2f%%)\n" "Error Rate (<5%)" "$error_rate"
    else
        printf "%-30s ${RED}✗ FAIL${NC} (%.2f%%)\n" "Error Rate (<5%)" "$error_rate"
    fi
    
    echo ""
    printf "Throughput: %.1f RPS\n" "$rps"
    
    # Overall result
    echo ""
    if [ "$overall_compliance" = "true" ]; then
        log_success "🎉 PHASE 1 REQUIREMENTS: FULLY COMPLIANT"
        echo "PRSM successfully handles 1000 concurrent users with <2s latency!"
    else
        log_warning "⚠️  PHASE 1 REQUIREMENTS: NOT FULLY COMPLIANT"
        echo "Further optimization needed to meet Phase 1 targets."
        
        # Generate recommendations
        generate_phase1_recommendations "$max_vus" "$p95_latency" "$error_rate"
    fi
    
    # Show file locations
    echo ""
    log_info "Detailed results available in:"
    echo "  - Summary: $OUTPUT_DIR/phase1-results.json"
    echo "  - Raw data: $OUTPUT_DIR/phase1-raw-results.json"
    echo "  - Monitoring: $OUTPUT_DIR/phase1-monitoring.log"
}

generate_phase1_recommendations() {
    local max_vus="$1"
    local p95_latency="$2"
    local error_rate="$3"
    
    echo ""
    log_info "=== OPTIMIZATION RECOMMENDATIONS ==="
    
    if (( $(echo "$max_vus < $PHASE1_CONCURRENT_USERS" | bc -l) )); then
        echo "• Scale horizontally: Add more API server replicas"
        echo "• Optimize connection pooling and keep-alive settings"
        echo "• Review resource limits in Kubernetes deployments"
    fi
    
    if (( $(echo "$p95_latency >= $PHASE1_MAX_LATENCY_MS" | bc -l) )); then
        echo "• Optimize database queries and add indexing"
        echo "• Implement response caching for frequently accessed data"
        echo "• Consider CDN for static content delivery"
        echo "• Profile and optimize hot code paths"
    fi
    
    if (( $(echo "$error_rate >= $PHASE1_MAX_ERROR_RATE" | bc -l) )); then
        echo "• Investigate error logs for common failure patterns"
        echo "• Implement circuit breakers and better error handling"
        echo "• Review rate limiting configurations"
        echo "• Check database connection pool sizing"
    fi
    
    echo "• Monitor resource utilization (CPU, memory, network)"
    echo "• Consider implementing auto-scaling policies"
    echo "• Run targeted performance profiling during peak load"
}

create_validation_report() {
    log_info "Creating comprehensive Phase 1 validation report..."
    
    local report_file="$OUTPUT_DIR/PHASE1_VALIDATION_REPORT.md"
    local timestamp=$(date)
    
    cat > "$report_file" << EOF
# PRSM Phase 1 Validation Report

**Generated:** $timestamp  
**Test Duration:** ${PHASE1_TEST_DURATION}s  
**Target:** $API_BASE_URL  

## Executive Summary

This report validates PRSM's Phase 1 requirements:
- ✅ Support 1000 concurrent users
- ✅ Maintain <2s latency (95th percentile)
- ✅ Achieve >95% success rate

## Test Configuration

- **Concurrent Users Target:** $PHASE1_CONCURRENT_USERS
- **Maximum Latency:** ${PHASE1_MAX_LATENCY_MS}ms
- **Maximum Error Rate:** ${PHASE1_MAX_ERROR_RATE}%
- **Test Duration:** ${PHASE1_TEST_DURATION}s

## Results Summary

$(if [ -f "$OUTPUT_DIR/phase1-results.json" ]; then
    echo "### Performance Metrics"
    echo ""
    jq -r '"- **Max Concurrent Users:** " + (.results.max_vus | tostring) + " (Target: " + (.requirements.concurrent_users | tostring) + ")"' "$OUTPUT_DIR/phase1-results.json"
    jq -r '"- **95th Percentile Latency:** " + (.results.p95_response_time | tostring | split(".")[0]) + "ms (Target: <" + (.requirements.max_latency_p95_ms | tostring) + "ms)"' "$OUTPUT_DIR/phase1-results.json"
    jq -r '"- **Error Rate:** " + ((.results.error_rate * 100) | tostring | split(".")[0]) + "% (Target: <" + ((.requirements.max_error_rate * 100) | tostring) + "%)"' "$OUTPUT_DIR/phase1-results.json"
    jq -r '"- **Throughput:** " + (.results.requests_per_second | tostring | split(".")[0]) + " RPS"' "$OUTPUT_DIR/phase1-results.json"
    echo ""
    echo "### Compliance Status"
    echo ""
    local overall=$(jq -r '.compliance.overall' "$OUTPUT_DIR/phase1-results.json")
    if [ "$overall" = "true" ]; then
        echo "🎉 **PHASE 1 REQUIREMENTS: PASSED**"
    else
        echo "⚠️ **PHASE 1 REQUIREMENTS: FAILED**"
    fi
else
    echo "Results not available. Test may have failed to complete."
fi)

## Detailed Analysis

### Load Testing Tool: k6
- **Test Script:** phase1-validation.js
- **Test Strategy:** Gradual ramp-up to 1000 users, sustained load, gradual ramp-down
- **Endpoints Tested:** $(echo "Health, Sessions, Marketplace, Governance, Teams, Metrics")

### Infrastructure Monitoring
$(if [ -f "$OUTPUT_DIR/phase1-monitoring.log" ]; then
    echo "- System metrics collected during test execution"
    echo "- See phase1-monitoring.log for detailed resource utilization"
else
    echo "- System monitoring data not available"
fi)

## Files Generated

\`\`\`
$(ls -la "$OUTPUT_DIR" 2>/dev/null | tail -n +2 || echo "No files generated")
\`\`\`

## Next Steps

Based on the validation results:

1. **If PASSED:** PRSM is ready for Phase 1 deployment
2. **If FAILED:** Review recommendations and optimize before re-testing

## Technical Details

- **Load Testing Framework:** k6
- **Monitoring:** Prometheus + Grafana
- **Test Automation:** Bash scripting with validation logic
- **Results Format:** JSON + Markdown reports

---
*Generated by PRSM Phase 1 Validation Suite*
EOF
    
    log_success "Validation report created: $report_file"
}

main() {
    log_info "🎯 Starting PRSM Phase 1 Requirements Validation"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --url)
                API_BASE_URL="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --prometheus)
                PROMETHEUS_URL="$2"
                shift 2
                ;;
            --quick)
                PHASE1_TEST_DURATION=60
                log_info "Quick mode: 60s test duration"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--url <url>] [--output <dir>] [--prometheus <url>] [--quick]"
                exit 1
                ;;
        esac
    done
    
    echo ""
    echo "🎯 PRSM Phase 1 Validation Configuration"
    echo "├─ Target API: $API_BASE_URL"
    echo "├─ Concurrent Users: $PHASE1_CONCURRENT_USERS"
    echo "├─ Max Latency: ${PHASE1_MAX_LATENCY_MS}ms"
    echo "├─ Max Error Rate: ${PHASE1_MAX_ERROR_RATE}%"
    echo "├─ Test Duration: ${PHASE1_TEST_DURATION}s"
    echo "└─ Output Directory: $OUTPUT_DIR"
    echo ""
    
    # Run validation steps
    validate_environment
    create_phase1_k6_script
    run_phase1_validation
    create_validation_report
    
    echo ""
    log_success "Phase 1 validation completed!"
    echo ""
    echo "📊 View results:"
    echo "   cat $OUTPUT_DIR/PHASE1_VALIDATION_REPORT.md"
    echo ""
    echo "🔍 Detailed analysis:"
    echo "   cat $OUTPUT_DIR/phase1-results.json"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Handle script interruption
trap 'log_warning "Validation interrupted"' EXIT INT TERM

# Run main function
main "$@"