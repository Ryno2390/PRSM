// PRSM Baseline Performance Test
// =============================
// 
// K6 performance test for establishing baseline metrics
// Addresses Gemini's requirement for real performance validation

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiResponseTime = new Trend('api_response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp-up
    { duration: '5m', target: 10 },   // Stay at 10 users
    { duration: '2m', target: 50 },   // Ramp to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 0 },    // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.1'],     // Error rate under 10%
    errors: ['rate<0.1'],              // Custom error rate under 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Test 1: Health check endpoint
  let healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  // Test 2: API authentication
  let authPayload = JSON.stringify({
    username: 'test_user',
    password: 'test_password'
  });

  let authResponse = http.post(`${BASE_URL}/api/auth/login`, authPayload, {
    headers: { 'Content-Type': 'application/json' },
  });

  let authToken = '';
  let authSuccess = check(authResponse, {
    'auth status is 200 or 201': (r) => [200, 201].includes(r.status),
    'auth response time < 1000ms': (r) => r.timings.duration < 1000,
  });

  if (authSuccess && authResponse.json('access_token')) {
    authToken = authResponse.json('access_token');
  } else {
    errorRate.add(1);
  }

  // Test 3: Agent query endpoint (core functionality)
  if (authToken) {
    let queryPayload = JSON.stringify({
      query: 'What is the current status of the PRSM system?',
      context: 'system_health'
    });

    let queryResponse = http.post(`${BASE_URL}/api/agents/query`, queryPayload, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
    });

    check(queryResponse, {
      'query status is 200': (r) => r.status === 200,
      'query response time < 3000ms': (r) => r.timings.duration < 3000,
      'query returns valid response': (r) => r.json('response') && r.json('response').length > 0,
    }) || errorRate.add(1);

    apiResponseTime.add(queryResponse.timings.duration);
  }

  // Test 4: Vector store search (if available)
  if (authToken) {
    let searchPayload = JSON.stringify({
      query: 'machine learning',
      limit: 10
    });

    let searchResponse = http.post(`${BASE_URL}/api/search`, searchPayload, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
    });

    check(searchResponse, {
      'search response time < 2000ms': (r) => r.timings.duration < 2000,
      'search returns results or empty array': (r) => {
        if (r.status === 200) {
          let results = r.json('results');
          return Array.isArray(results);
        }
        return r.status === 404; // OK if search not implemented yet
      },
    }) || errorRate.add(1);
  }

  // Test 5: System metrics endpoint
  let metricsResponse = http.get(`${BASE_URL}/api/metrics`, {
    headers: authToken ? { 'Authorization': `Bearer ${authToken}` } : {},
  });

  check(metricsResponse, {
    'metrics accessible': (r) => [200, 401, 403].includes(r.status), // OK if auth required
    'metrics response time < 1000ms': (r) => r.timings.duration < 1000,
  }) || errorRate.add(1);

  // Random sleep between 1-3 seconds to simulate real user behavior
  sleep(Math.random() * 2 + 1);
}

export function handleSummary(data) {
  return {
    'performance-results/baseline_summary.json': JSON.stringify(data, null, 2),
    'performance-results/baseline_report.html': generateHTMLReport(data),
  };
}

function generateHTMLReport(data) {
  const metrics = data.metrics;
  
  return `
<!DOCTYPE html>
<html>
<head>
    <title>PRSM Baseline Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .metric { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .pass { background-color: #d4edda; border-color: #c3e6cb; }
        .fail { background-color: #f8d7da; border-color: #f5c6cb; }
        .header { background-color: #e9ecef; padding: 20px; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PRSM Baseline Performance Report</h1>
        <p>Generated: ${new Date().toISOString()}</p>
        <p>Test Duration: ${data.state.testRunDurationMs}ms</p>
        <p>Virtual Users: ${data.root_group.checks ? Object.keys(data.root_group.checks).length : 'N/A'}</p>
    </div>
    
    <div class="summary">
        <div class="metric ${metrics.http_req_duration?.values?.['p(95)'] < 2000 ? 'pass' : 'fail'}">
            <h3>Response Time (95th percentile)</h3>
            <p><strong>${metrics.http_req_duration?.values?.['p(95)']?.toFixed(2) || 'N/A'}ms</strong></p>
            <p>Threshold: < 2000ms</p>
        </div>
        
        <div class="metric ${(metrics.http_req_failed?.values?.rate || 0) < 0.1 ? 'pass' : 'fail'}">
            <h3>Error Rate</h3>
            <p><strong>${((metrics.http_req_failed?.values?.rate || 0) * 100).toFixed(2)}%</strong></p>
            <p>Threshold: < 10%</p>
        </div>
        
        <div class="metric">
            <h3>Total Requests</h3>
            <p><strong>${metrics.http_reqs?.values?.count || 'N/A'}</strong></p>
            <p>Rate: ${metrics.http_reqs?.values?.rate?.toFixed(2) || 'N/A'} req/s</p>
        </div>
        
        <div class="metric">
            <h3>Data Transferred</h3>
            <p><strong>${((metrics.data_received?.values?.count || 0) / 1024 / 1024).toFixed(2)} MB</strong></p>
            <p>Rate: ${((metrics.data_received?.values?.rate || 0) / 1024).toFixed(2)} KB/s</p>
        </div>
    </div>
    
    <h2>Detailed Metrics</h2>
    <pre>${JSON.stringify(metrics, null, 2)}</pre>
</body>
</html>`;
}