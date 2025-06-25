#!/bin/bash

# PRSM Comprehensive Load Testing Suite
# Tests Phase 1 requirements: 1000 concurrent users with <2s latency

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL=${API_BASE_URL:-"http://localhost:8000"}
DURATION=${DURATION:-"300s"}  # 5 minutes
CONCURRENT_USERS=${CONCURRENT_USERS:-"1000"}
RAMP_UP_TIME=${RAMP_UP_TIME:-"60s"}
TARGET_RPS=${TARGET_RPS:-"100"}
MAX_LATENCY=${MAX_LATENCY:-"2000"}  # 2s in milliseconds
OUTPUT_DIR=${OUTPUT_DIR:-"./load-test-results"}

# Phase 1 targets
PHASE1_CONCURRENT_USERS=1000
PHASE1_MAX_LATENCY_MS=2000
PHASE1_MIN_SUCCESS_RATE=95

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

check_dependencies() {
    log_info "Checking load testing dependencies..."
    
    local missing_tools=()
    
    # Check for load testing tools
    if ! command -v k6 &> /dev/null; then
        missing_tools+=("k6")
    fi
    
    if ! command -v wrk &> /dev/null; then
        missing_tools+=("wrk")
    fi
    
    if ! command -v hey &> /dev/null && ! command -v ab &> /dev/null; then
        missing_tools+=("hey or apache2-utils")
    fi
    
    # Check for monitoring tools
    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi
    
    if ! command -v jq &> /dev/null; then
        missing_tools+=("jq")
    fi
    
    if [ ${#missing_tools[@]} -eq 0 ]; then
        log_success "All dependencies are available"
        return 0
    else
        log_error "Missing dependencies: ${missing_tools[*]}"
        log_info "Installing missing tools..."
        install_dependencies "${missing_tools[@]}"
    fi
}

install_dependencies() {
    local tools=("$@")
    
    for tool in "${tools[@]}"; do
        case $tool in
            "k6")
                log_info "Installing k6..."
                if command -v brew &> /dev/null; then
                    brew install k6
                elif command -v apt-get &> /dev/null; then
                    sudo apt-get update && sudo apt-get install -y k6
                else
                    log_warning "Please install k6 manually: https://k6.io/docs/getting-started/installation/"
                fi
                ;;
            "wrk")
                log_info "Installing wrk..."
                if command -v brew &> /dev/null; then
                    brew install wrk
                elif command -v apt-get &> /dev/null; then
                    sudo apt-get update && sudo apt-get install -y wrk
                else
                    log_warning "Please install wrk manually"
                fi
                ;;
            "hey or apache2-utils")
                log_info "Installing hey..."
                if command -v go &> /dev/null; then
                    go install github.com/rakyll/hey@latest
                elif command -v apt-get &> /dev/null; then
                    sudo apt-get update && sudo apt-get install -y apache2-utils
                else
                    log_warning "Please install hey or apache2-utils manually"
                fi
                ;;
            "jq")
                if command -v brew &> /dev/null; then
                    brew install jq
                elif command -v apt-get &> /dev/null; then
                    sudo apt-get update && sudo apt-get install -y jq
                fi
                ;;
        esac
    done
}

setup_test_environment() {
    log_info "Setting up load test environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Generate timestamp for this test run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    TEST_RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
    mkdir -p "$TEST_RUN_DIR"
    
    log_info "Test results will be saved to: $TEST_RUN_DIR"
    
    # Check API health
    if ! curl -s --max-time 5 "$API_BASE_URL/health" > /dev/null; then
        log_error "API health check failed. Is the PRSM API running at $API_BASE_URL?"
        return 1
    fi
    
    log_success "Test environment ready"
}

create_k6_script() {
    log_info "Creating k6 load test script..."
    
    cat > "$TEST_RUN_DIR/k6-script.js" << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export let errorRate = new Rate('errors');
export let responseTimeTrend = new Trend('response_time');

export let options = {
  stages: [
    { duration: '60s', target: 200 },   // Ramp up to 200 users
    { duration: '60s', target: 500 },   // Ramp up to 500 users
    { duration: '60s', target: 1000 },  // Ramp up to 1000 users (Phase 1 target)
    { duration: '180s', target: 1000 }, // Stay at 1000 users for 3 minutes
    { duration: '60s', target: 0 },     // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<2000'], // 95th percentile < 2s (Phase 1 requirement)
    'http_req_failed': ['rate<0.05'],    // Error rate < 5%
    'errors': ['rate<0.05'],             // Custom error rate < 5%
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';

// Endpoint weights for realistic traffic distribution
const endpoints = [
  { path: '/health', weight: 10, method: 'GET' },
  { path: '/api/v1/sessions', weight: 30, method: 'GET' },
  { path: '/api/v1/sessions', weight: 20, method: 'POST', data: { query: 'Test query', domain: 'general' } },
  { path: '/api/v1/marketplace/models', weight: 15, method: 'GET' },
  { path: '/api/v1/governance/proposals', weight: 10, method: 'GET' },
  { path: '/api/v1/teams', weight: 10, method: 'GET' },
  { path: '/metrics', weight: 5, method: 'GET' },
];

function selectEndpoint() {
  const totalWeight = endpoints.reduce((sum, ep) => sum + ep.weight, 0);
  let random = Math.random() * totalWeight;
  
  for (const endpoint of endpoints) {
    random -= endpoint.weight;
    if (random <= 0) {
      return endpoint;
    }
  }
  return endpoints[0];
}

export default function() {
  const endpoint = selectEndpoint();
  const url = `${BASE_URL}${endpoint.path}`;
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'PRSM-LoadTest-k6/1.0',
    },
    timeout: '10s',
  };
  
  let response;
  const startTime = Date.now();
  
  if (endpoint.method === 'POST' && endpoint.data) {
    response = http.post(url, JSON.stringify(endpoint.data), params);
  } else {
    response = http.get(url, params);
  }
  
  const endTime = Date.now();
  const responseTime = endTime - startTime;
  
  // Record custom metrics
  responseTimeTrend.add(responseTime);
  errorRate.add(response.status >= 400);
  
  // Validate response
  const success = check(response, {
    'status is 2xx': (r) => r.status >= 200 && r.status < 300,
    'response time < 2000ms': (r) => r.timings.duration < 2000,
    'response has body': (r) => r.body && r.body.length > 0,
  });
  
  if (!success) {
    console.error(`Request failed: ${endpoint.method} ${url} - Status: ${response.status}, Time: ${responseTime}ms`);
  }
  
  // Realistic user think time
  sleep(Math.random() * 2 + 0.5); // 0.5-2.5 seconds
}

export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data, null, 2),
    'summary.txt': textSummary(data, { indent: ' ', enableColors: false }),
  };
}
EOF
    
    log_success "k6 script created"
}

run_k6_test() {
    log_info "Running k6 load test..."
    
    local k6_output="$TEST_RUN_DIR/k6-results"
    mkdir -p "$k6_output"
    
    # Run k6 test
    API_BASE_URL="$API_BASE_URL" k6 run \
        --out json="$k6_output/results.json" \
        --out csv="$k6_output/results.csv" \
        "$TEST_RUN_DIR/k6-script.js" \
        > "$k6_output/output.log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "k6 test completed successfully"
    else
        log_warning "k6 test completed with warnings (exit code: $exit_code)"
    fi
    
    # Parse results
    if [ -f "$k6_output/summary.json" ]; then
        parse_k6_results "$k6_output/summary.json"
    fi
    
    return $exit_code
}

parse_k6_results() {
    local results_file="$1"
    
    log_info "Parsing k6 test results..."
    
    # Extract key metrics
    local avg_duration=$(jq -r '.metrics.http_req_duration.values.avg // 0' "$results_file")
    local p95_duration=$(jq -r '.metrics.http_req_duration.values["p(95)"] // 0' "$results_file")
    local p99_duration=$(jq -r '.metrics.http_req_duration.values["p(99)"] // 0' "$results_file")
    local error_rate=$(jq -r '.metrics.http_req_failed.values.rate // 0' "$results_file")
    local total_requests=$(jq -r '.metrics.http_reqs.values.count // 0' "$results_file")
    local rps=$(jq -r '.metrics.http_reqs.values.rate // 0' "$results_file")
    local vus_max=$(jq -r '.metrics.vus_max.values.max // 0' "$results_file")
    
    # Convert to readable format
    avg_duration_ms=$(echo "$avg_duration" | awk '{printf "%.1f", $1}')
    p95_duration_ms=$(echo "$p95_duration" | awk '{printf "%.1f", $1}')
    p99_duration_ms=$(echo "$p99_duration" | awk '{printf "%.1f", $1}')
    error_rate_pct=$(echo "$error_rate * 100" | bc -l | awk '{printf "%.2f", $1}')
    rps_formatted=$(echo "$rps" | awk '{printf "%.1f", $1}')
    
    echo ""
    log_info "=== K6 LOAD TEST RESULTS ==="
    echo "Total Requests: $total_requests"
    echo "Requests/Second: $rps_formatted"
    echo "Max Virtual Users: $vus_max"
    echo "Average Response Time: ${avg_duration_ms}ms"
    echo "95th Percentile: ${p95_duration_ms}ms"
    echo "99th Percentile: ${p99_duration_ms}ms"
    echo "Error Rate: ${error_rate_pct}%"
    
    # Check Phase 1 compliance
    check_phase1_compliance "$p95_duration_ms" "$error_rate_pct" "$vus_max"
}

run_wrk_test() {
    log_info "Running wrk benchmark test..."
    
    if ! command -v wrk &> /dev/null; then
        log_warning "wrk not available, skipping wrk test"
        return 0
    fi
    
    local wrk_output="$TEST_RUN_DIR/wrk-results.txt"
    
    # Run wrk test
    wrk -t12 -c400 -d"$DURATION" --latency "$API_BASE_URL/health" > "$wrk_output" 2>&1
    
    log_info "wrk test completed. Results saved to: $wrk_output"
    
    # Extract key metrics from wrk output
    if [ -f "$wrk_output" ]; then
        log_info "=== WRK BENCHMARK RESULTS ==="
        grep -E "(Requests/sec|Latency|requests in)" "$wrk_output" || true
    fi
}

run_hey_test() {
    log_info "Running hey stress test..."
    
    if ! command -v hey &> /dev/null && ! command -v ab &> /dev/null; then
        log_warning "hey/ab not available, skipping stress test"
        return 0
    fi
    
    local hey_output="$TEST_RUN_DIR/hey-results.txt"
    
    # Run hey test if available
    if command -v hey &> /dev/null; then
        hey -z "$DURATION" -c "$CONCURRENT_USERS" "$API_BASE_URL/health" > "$hey_output" 2>&1
    else
        # Fall back to apache bench
        local total_requests=$((100 * 60 * 5))  # ~5 minutes worth
        ab -n "$total_requests" -c 100 "$API_BASE_URL/health" > "$hey_output" 2>&1
    fi
    
    log_info "Stress test completed. Results saved to: $hey_output"
    
    # Extract key metrics
    if [ -f "$hey_output" ]; then
        log_info "=== STRESS TEST RESULTS ==="
        grep -E "(Requests per second|Time per request|Total:|Percentage of the requests)" "$hey_output" | head -10 || true
    fi
}

run_custom_python_test() {
    log_info "Running custom Python load test..."
    
    # Check if Python test is available
    if [ ! -f "prsm/performance/load_testing.py" ]; then
        log_warning "Custom Python load test not found, skipping"
        return 0
    fi
    
    local python_output="$TEST_RUN_DIR/python-results.json"
    
    # Run Python load test
    python3 -c "
import asyncio
import sys
import json
sys.path.append('.')
from prsm.performance.load_testing import LoadTestSuite, LoadTestConfig

async def run_test():
    config = LoadTestConfig(
        test_name='phase1_validation',
        description='PRSM Phase 1 Load Test Validation',
        concurrent_users=$CONCURRENT_USERS,
        duration_seconds=300,
        base_url='$API_BASE_URL',
        max_response_time_ms=$MAX_LATENCY,
        min_success_rate=0.95,
        test_scenarios=['api_endpoints']
    )
    
    suite = LoadTestSuite()
    result = await suite.run_load_test(config)
    
    # Save results
    with open('$python_output', 'w') as f:
        json.dump({
            'test_name': result.test_name,
            'duration_seconds': result.duration_seconds,
            'total_requests': result.total_requests,
            'success_rate': result.success_rate,
            'avg_response_time_ms': result.avg_response_time_ms,
            'p95_response_time_ms': result.p95_response_time_ms,
            'requests_per_second': result.requests_per_second,
            'passed_thresholds': result.passed_thresholds,
            'performance_score': result.performance_score,
            'recommendations': result.recommendations
        }, f, indent=2)
    
    return result

result = asyncio.run(run_test())
print(f'Python test completed: {result.passed_thresholds}')
" 2>&1 | tee "$TEST_RUN_DIR/python-output.log"
    
    if [ -f "$python_output" ]; then
        log_info "=== PYTHON LOAD TEST RESULTS ==="
        jq -r '. | "Success Rate: \(.success_rate * 100)%, Avg Response: \(.avg_response_time_ms)ms, RPS: \(.requests_per_second), Passed: \(.passed_thresholds)"' "$python_output"
    fi
}

check_phase1_compliance() {
    local p95_latency="$1"
    local error_rate="$2"
    local max_users="$3"
    
    echo ""
    log_info "=== PHASE 1 COMPLIANCE CHECK ==="
    
    local compliance_score=0
    local total_checks=3
    
    # Check latency requirement
    if (( $(echo "$p95_latency < $PHASE1_MAX_LATENCY_MS" | bc -l) )); then
        log_success "✓ Latency requirement: ${p95_latency}ms < ${PHASE1_MAX_LATENCY_MS}ms"
        ((compliance_score++))
    else
        log_error "✗ Latency requirement: ${p95_latency}ms >= ${PHASE1_MAX_LATENCY_MS}ms"
    fi
    
    # Check error rate requirement
    local max_error_rate=5.0
    if (( $(echo "$error_rate < $max_error_rate" | bc -l) )); then
        log_success "✓ Error rate requirement: ${error_rate}% < ${max_error_rate}%"
        ((compliance_score++))
    else
        log_error "✗ Error rate requirement: ${error_rate}% >= ${max_error_rate}%"
    fi
    
    # Check concurrency requirement
    if (( $(echo "$max_users >= $PHASE1_CONCURRENT_USERS" | bc -l) )); then
        log_success "✓ Concurrency requirement: ${max_users} >= ${PHASE1_CONCURRENT_USERS} users"
        ((compliance_score++))
    else
        log_error "✗ Concurrency requirement: ${max_users} < ${PHASE1_CONCURRENT_USERS} users"
    fi
    
    # Overall compliance
    local compliance_percentage=$((compliance_score * 100 / total_checks))
    
    if [ $compliance_score -eq $total_checks ]; then
        log_success "🎉 PHASE 1 REQUIREMENTS: PASSED ($compliance_percentage%)"
        return 0
    else
        log_warning "⚠️  PHASE 1 REQUIREMENTS: PARTIAL ($compliance_percentage%)"
        return 1
    fi
}

generate_test_report() {
    log_info "Generating comprehensive test report..."
    
    local report_file="$TEST_RUN_DIR/load-test-report.md"
    
    cat > "$report_file" << EOF
# PRSM Load Test Report

**Test Run:** $(date)
**Test Directory:** $TEST_RUN_DIR
**Target API:** $API_BASE_URL
**Duration:** $DURATION
**Max Concurrent Users:** $CONCURRENT_USERS

## Phase 1 Requirements

- **Concurrent Users:** 1000
- **Max Latency (95th percentile):** 2000ms
- **Min Success Rate:** 95%

## Test Results Summary

### k6 Load Test
$([ -f "$TEST_RUN_DIR/k6-results/summary.json" ] && echo "See k6-results/summary.json for detailed metrics" || echo "k6 test not completed")

### wrk Benchmark
$([ -f "$TEST_RUN_DIR/wrk-results.txt" ] && echo "See wrk-results.txt for benchmark results" || echo "wrk test not completed")

### Stress Test (hey/ab)
$([ -f "$TEST_RUN_DIR/hey-results.txt" ] && echo "See hey-results.txt for stress test results" || echo "Stress test not completed")

### Custom Python Test
$([ -f "$TEST_RUN_DIR/python-results.json" ] && echo "See python-results.json for custom test results" || echo "Python test not completed")

## Files Generated

\`\`\`
$(ls -la "$TEST_RUN_DIR" | tail -n +2)
\`\`\`

## Next Steps

1. Review detailed results in individual test output files
2. Analyze performance bottlenecks if Phase 1 requirements not met
3. Optimize infrastructure and application code as needed
4. Re-run tests to validate improvements

Generated at: $(date)
EOF
    
    log_success "Test report generated: $report_file"
}

cleanup() {
    log_info "Cleaning up test artifacts..."
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill > /dev/null 2>&1 || true
    
    log_success "Cleanup completed"
}

main() {
    log_info "Starting PRSM Load Test Suite"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --url)
                API_BASE_URL="$2"
                shift 2
                ;;
            --duration)
                DURATION="$2"
                shift 2
                ;;
            --users)
                CONCURRENT_USERS="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --quick)
                DURATION="60s"
                CONCURRENT_USERS="100"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--url <url>] [--duration <time>] [--users <count>] [--output <dir>] [--quick]"
                exit 1
                ;;
        esac
    done
    
    # Run test suite
    check_dependencies
    setup_test_environment
    
    # Create and run different load tests
    create_k6_script
    
    log_info "Running comprehensive load test suite..."
    echo "Target: $API_BASE_URL"
    echo "Duration: $DURATION"
    echo "Max Concurrent Users: $CONCURRENT_USERS"
    echo ""
    
    # Run tests in sequence
    run_k6_test
    run_wrk_test
    run_hey_test
    run_custom_python_test
    
    # Generate final report
    generate_test_report
    
    log_success "Load test suite completed!"
    echo ""
    echo "Results saved to: $TEST_RUN_DIR"
    echo "View report: cat $TEST_RUN_DIR/load-test-report.md"
}

# Handle script interruption
trap cleanup EXIT INT TERM

# Run main function
main "$@"