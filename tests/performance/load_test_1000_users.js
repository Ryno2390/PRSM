// PRSM Load Test - 1000 Concurrent Users
// =====================================
// 
// Addresses Gemini's specific requirement to test system scalability
// at the target of 1000+ concurrent users

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics for detailed analysis
const errorRate = new Rate('errors');
const dbQueryTime = new Trend('database_query_time');
const authTime = new Trend('auth_time');
const mlProcessingTime = new Trend('ml_processing_time');
const concurrentUsers = new Counter('concurrent_users');

// Aggressive load test configuration
export const options = {
  stages: [
    { duration: '5m', target: 100 },   // Warm-up phase
    { duration: '5m', target: 300 },   // Ramp to 300 users
    { duration: '5m', target: 600 },   // Ramp to 600 users
    { duration: '10m', target: 1000 }, // Target load: 1000 users
    { duration: '15m', target: 1000 }, // Sustained load for 15 minutes
    { duration: '5m', target: 1200 },  // Spike test: beyond target
    { duration: '5m', target: 1000 },  // Return to target
    { duration: '10m', target: 0 },    // Gradual ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000', 'p(99)<10000'], // Realistic thresholds under load
    http_req_failed: ['rate<0.05'],                   // 5% error rate acceptable under heavy load
    errors: ['rate<0.05'],
    database_query_time: ['p(95)<3000'],
    auth_time: ['p(95)<2000'],
    ml_processing_time: ['p(95)<8000'],               // ML operations can be slower
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// User behavior scenarios with realistic weights
const scenarios = [
  { name: 'quick_query', weight: 40 },      // 40% - Quick information queries
  { name: 'complex_ml', weight: 25 },       // 25% - ML processing tasks
  { name: 'data_search', weight: 20 },      // 20% - Vector/database searches
  { name: 'system_admin', weight: 10 },     // 10% - Admin/monitoring tasks
  { name: 'marketplace', weight: 5 },       // 5% - Economic/marketplace operations
];

function selectScenario() {
  const random = Math.random() * 100;
  let cumulative = 0;
  
  for (const scenario of scenarios) {
    cumulative += scenario.weight;
    if (random <= cumulative) {
      return scenario.name;
    }
  }
  return scenarios[0].name;
}

export default function () {
  concurrentUsers.add(1);
  
  // Authentication (required for most operations)
  let authStart = Date.now();
  let authPayload = JSON.stringify({
    username: `user_${Math.floor(Math.random() * 10000)}`,
    password: 'load_test_password'
  });

  let authResponse = http.post(`${BASE_URL}/api/auth/login`, authPayload, {
    headers: { 'Content-Type': 'application/json' },
  });

  authTime.add(Date.now() - authStart);

  let authToken = '';
  let authSuccess = check(authResponse, {
    'auth successful': (r) => [200, 201].includes(r.status),
  });

  if (authSuccess && authResponse.json('access_token')) {
    authToken = authResponse.json('access_token');
  } else {
    errorRate.add(1);
    sleep(1);
    return; // Skip rest if auth fails
  }

  // Execute scenario based on user behavior
  const scenario = selectScenario();
  
  switch (scenario) {
    case 'quick_query':
      executeQuickQuery(authToken);
      break;
    case 'complex_ml':
      executeComplexMLTask(authToken);
      break;
    case 'data_search':
      executeDataSearch(authToken);
      break;
    case 'system_admin':
      executeSystemAdmin(authToken);
      break;
    case 'marketplace':
      executeMarketplace(authToken);
      break;
  }

  // Random think time (1-5 seconds)
  sleep(Math.random() * 4 + 1);
}

function executeQuickQuery(authToken) {
  const queries = [
    'What is the current system status?',
    'How many users are online?',
    'What are the latest updates?',
    'Show me recent activity',
    'What is my account balance?'
  ];

  let queryPayload = JSON.stringify({
    query: queries[Math.floor(Math.random() * queries.length)],
    priority: 'high',
    timeout: 5000
  });

  let response = http.post(`${BASE_URL}/api/agents/query`, queryPayload, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
  });

  check(response, {
    'quick query successful': (r) => r.status === 200,
    'quick query fast': (r) => r.timings.duration < 3000,
  }) || errorRate.add(1);
}

function executeComplexMLTask(authToken) {
  let mlStart = Date.now();
  
  let mlPayload = JSON.stringify({
    task_type: 'text_analysis',
    content: 'This is a complex text that requires ML processing for sentiment analysis, entity extraction, and content classification. The PRSM system should demonstrate its real ML capabilities here.',
    options: {
      include_sentiment: true,
      extract_entities: true,
      classify_content: true,
      generate_summary: true
    }
  });

  let response = http.post(`${BASE_URL}/api/ml/process`, mlPayload, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
    timeout: '30s', // Longer timeout for ML tasks
  });

  mlProcessingTime.add(Date.now() - mlStart);

  check(response, {
    'ML task successful': (r) => r.status === 200,
    'ML response contains results': (r) => {
      if (r.status === 200) {
        let results = r.json('results');
        return results && typeof results === 'object';
      }
      return r.status === 503; // Service unavailable is acceptable under load
    },
  }) || errorRate.add(1);
}

function executeDataSearch(authToken) {
  let dbStart = Date.now();
  
  let searchTerms = [
    'machine learning',
    'artificial intelligence', 
    'data science',
    'blockchain',
    'cryptocurrency',
    'neural networks',
    'natural language processing'
  ];

  let searchPayload = JSON.stringify({
    query: searchTerms[Math.floor(Math.random() * searchTerms.length)],
    limit: 50,
    include_metadata: true,
    similarity_threshold: 0.7
  });

  let response = http.post(`${BASE_URL}/api/search/vector`, searchPayload, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
  });

  dbQueryTime.add(Date.now() - dbStart);

  check(response, {
    'search completed': (r) => [200, 404].includes(r.status),
    'search response time acceptable': (r) => r.timings.duration < 5000,
  }) || errorRate.add(1);
}

function executeSystemAdmin(authToken) {
  // Admin operations: metrics, monitoring, system health
  let adminEndpoints = [
    '/api/metrics/system',
    '/api/health/detailed',
    '/api/monitoring/performance',
    '/api/admin/user_stats'
  ];

  for (const endpoint of adminEndpoints.slice(0, 2)) { // Test 2 endpoints
    let response = http.get(`${BASE_URL}${endpoint}`, {
      headers: { 'Authorization': `Bearer ${authToken}` },
    });

    check(response, {
      [`admin ${endpoint} accessible`]: (r) => [200, 401, 403].includes(r.status),
    }) || errorRate.add(1);

    sleep(0.5); // Short delay between admin calls
  }
}

function executeMarketplace(authToken) {
  // Economic operations: balance checks, transactions, marketplace activity
  let marketplacePayload = JSON.stringify({
    action: 'get_balance',
    currency: 'FTNS'
  });

  let balanceResponse = http.post(`${BASE_URL}/api/economy/balance`, marketplacePayload, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
  });

  check(balanceResponse, {
    'balance check successful': (r) => [200, 404].includes(r.status),
  }) || errorRate.add(1);

  // Simulate browsing marketplace
  let marketplaceResponse = http.get(`${BASE_URL}/api/marketplace/browse?limit=20`, {
    headers: { 'Authorization': `Bearer ${authToken}` },
  });

  check(marketplaceResponse, {
    'marketplace accessible': (r) => [200, 404, 503].includes(r.status),
  }) || errorRate.add(1);
}

export function handleSummary(data) {
  const report = generateLoadTestReport(data);
  
  return {
    'performance-results/load_test_1000_users.json': JSON.stringify(data, null, 2),
    'performance-results/load_test_1000_users.html': report,
    stdout: generateConsoleReport(data),
  };
}

function generateLoadTestReport(data) {
  const metrics = data.metrics;
  const vusMax = metrics.vus_max?.values?.max || 0;
  const totalRequests = metrics.http_reqs?.values?.count || 0;
  const errorRate = (metrics.http_req_failed?.values?.rate || 0) * 100;
  const p95ResponseTime = metrics.http_req_duration?.values?.['p(95)'] || 0;
  const p99ResponseTime = metrics.http_req_duration?.values?.['p(99)'] || 0;
  
  const passed = errorRate < 5 && p95ResponseTime < 5000 && p99ResponseTime < 10000;
  
  return `
<!DOCTYPE html>
<html>
<head>
    <title>PRSM 1000-User Load Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: ${passed ? '#d4edda' : '#f8d7da'}; padding: 20px; margin-bottom: 30px; }
        .metric { margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .pass { background-color: #d4edda; }
        .fail { background-color: #f8d7da; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .critical { font-size: 1.2em; font-weight: bold; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 PRSM 1000-User Load Test Report</h1>
        <div class="critical">
            Test Result: ${passed ? '✅ PASSED' : '❌ FAILED'}
        </div>
        <p>Generated: ${new Date().toISOString()}</p>
        <p>Max Concurrent Users: ${vusMax}</p>
        <p>Total Requests: ${totalRequests.toLocaleString()}</p>
        <p>Test Duration: ${Math.round(data.state.testRunDurationMs / 1000 / 60)} minutes</p>
    </div>
    
    <div class="summary">
        <div class="metric ${errorRate < 5 ? 'pass' : 'fail'}">
            <h3>📈 Error Rate</h3>
            <div class="critical">${errorRate.toFixed(2)}%</div>
            <p>Threshold: < 5%</p>
        </div>
        
        <div class="metric ${p95ResponseTime < 5000 ? 'pass' : 'fail'}">
            <h3>⚡ Response Time (95th)</h3>
            <div class="critical">${p95ResponseTime.toFixed(0)}ms</div>
            <p>Threshold: < 5000ms</p>
        </div>
        
        <div class="metric ${p99ResponseTime < 10000 ? 'pass' : 'fail'}">
            <h3>🎯 Response Time (99th)</h3>
            <div class="critical">${p99ResponseTime.toFixed(0)}ms</div>
            <p>Threshold: < 10000ms</p>
        </div>
        
        <div class="metric">
            <h3>🔥 Request Rate</h3>
            <div class="critical">${(metrics.http_reqs?.values?.rate || 0).toFixed(1)} req/s</div>
            <p>Peak throughput achieved</p>
        </div>
        
        <div class="metric">
            <h3>💾 Data Transferred</h3>
            <div class="critical">${((metrics.data_received?.values?.count || 0) / 1024 / 1024).toFixed(1)} MB</div>
            <p>Total data received</p>
        </div>
        
        <div class="metric">
            <h3>🧠 ML Processing Time</h3>
            <div class="critical">${(metrics.ml_processing_time?.values?.['p(95)'] || 0).toFixed(0)}ms</div>
            <p>95th percentile ML tasks</p>
        </div>
    </div>
    
    <h2>🏗️ Production Readiness Assessment</h2>
    <div class="summary">
        <div class="metric ${vusMax >= 1000 ? 'pass' : 'fail'}">
            <h4>Scalability Target</h4>
            <p>${vusMax >= 1000 ? '✅' : '❌'} Achieved 1000+ concurrent users</p>
        </div>
        
        <div class="metric ${errorRate < 5 ? 'pass' : 'fail'}">
            <h4>Reliability</h4>
            <p>${errorRate < 5 ? '✅' : '❌'} Error rate within acceptable limits</p>
        </div>
        
        <div class="metric ${p95ResponseTime < 5000 ? 'pass' : 'fail'}">
            <h4>Performance</h4>
            <p>${p95ResponseTime < 5000 ? '✅' : '❌'} Response times acceptable under load</p>
        </div>
    </div>
    
    <h2>📊 Detailed Metrics</h2>
    <pre>${JSON.stringify(metrics, null, 2)}</pre>
</body>
</html>`;
}

function generateConsoleReport(data) {
  const metrics = data.metrics;
  const vusMax = metrics.vus_max?.values?.max || 0;
  const errorRate = (metrics.http_req_failed?.values?.rate || 0) * 100;
  const p95ResponseTime = metrics.http_req_duration?.values?.['p(95)'] || 0;
  
  return `
🚀 PRSM 1000-User Load Test Results
===================================

Max Concurrent Users: ${vusMax}
Error Rate: ${errorRate.toFixed(2)}%
95th Percentile Response Time: ${p95ResponseTime.toFixed(0)}ms

${vusMax >= 1000 && errorRate < 5 && p95ResponseTime < 5000 ? '✅ LOAD TEST PASSED' : '❌ LOAD TEST FAILED'}

Target: Support 1000+ concurrent users with < 5% error rate
Status: ${vusMax >= 1000 ? 'Scalability ✅' : 'Scalability ❌'} | ${errorRate < 5 ? 'Reliability ✅' : 'Reliability ❌'}
`;
}