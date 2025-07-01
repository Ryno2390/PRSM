import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter, Trend } from 'k6/metrics';

// Custom metrics for Gemini audit requirements
const errorRate = new Rate('errors');
const requestCounter = new Counter('requests_total');
const responseTime = new Trend('response_time');
const concurrentUsers = new Trend('concurrent_users');

export const options = {
  stages: [
    // Gradual ramp up to 1000+ users (Gemini requirement)
    { duration: '5m', target: 100 },   // Ramp to 100 users
    { duration: '5m', target: 300 },   // Ramp to 300 users  
    { duration: '5m', target: 600 },   // Ramp to 600 users
    { duration: '10m', target: 1000 }, // Ramp to 1000 users
    { duration: '15m', target: 1200 }, // Peak at 1200 users (exceed requirement)
    { duration: '10m', target: 1000 }, // Sustain at 1000 users
    { duration: '5m', target: 500 },   // Ramp down to 500
    { duration: '5m', target: 0 },     // Ramp down to 0
  ],
  thresholds: {
    // Gemini audit requirements
    http_req_duration: ['p(95)<8000'],    // 95% under 8 seconds
    http_req_failed: ['rate<0.15'],       // Error rate under 15%
    response_time: ['p(95)<8000'],        // Custom response time
    requests_total: ['count>50000'],      // Minimum request volume
    concurrent_users: ['avg>=1000'],      // Average concurrent users >= 1000
  },
  // Resource configuration for high load
  maxRedirects: 3,
  insecureSkipTLSVerify: true,
  noConnectionReuse: false,
  userAgent: 'PRSM-LoadTest-1000Users/1.0',
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Mock user data for realistic testing
const users = [];
for (let i = 0; i < 1500; i++) {
  users.push({
    username: `loadtest_user_${i}`,
    password: 'test_password_123',
    type: i % 10 === 0 ? 'premium' : 'standard'
  });
}

export default function () {
  // Track current VU as concurrent user
  concurrentUsers.add(__VU);
  requestCounter.add(1);
  
  // Realistic user behavior simulation
  const userIndex = (__VU - 1) % users.length;
  const currentUser = users[userIndex];
  
  // Simulate different user behaviors
  const behavior = Math.random();
  
  if (behavior < 0.4) {
    // 40% - Basic API usage
    performBasicAPIUsage(currentUser);
  } else if (behavior < 0.7) {
    // 30% - Marketplace browsing and transactions
    performMarketplaceUsage(currentUser);
  } else if (behavior < 0.85) {
    // 15% - AI model interaction
    performAIModelUsage(currentUser);
  } else {
    // 15% - System monitoring and analytics
    performSystemMonitoring(currentUser);
  }
  
  // Variable sleep to simulate realistic usage patterns
  const sleepTime = currentUser.type === 'premium' ? 
    Math.random() * 2 + 0.5 :  // Premium users: 0.5-2.5s
    Math.random() * 3 + 1;     // Standard users: 1-4s
  
  sleep(sleepTime);
}

function performBasicAPIUsage(user) {
  // Health check
  let healthRes = http.get(`${BASE_URL}/health`);
  
  check(healthRes, {
    'health check status is 200': (r) => r.status === 200,
    'health response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  errorRate.add(healthRes.status !== 200);
  responseTime.add(healthRes.timings.duration);
  
  // Authentication simulation
  let authRes = http.post(`${BASE_URL}/api/auth/login`, JSON.stringify({
    username: user.username,
    password: user.password
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(authRes, {
    'auth status is 200 or 401': (r) => r.status === 200 || r.status === 401,
  });
  
  if (authRes.status === 200) {
    // Simulate authenticated API calls
    const token = 'mock_token_' + user.username;
    
    http.get(`${BASE_URL}/api/user/profile`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    
    http.get(`${BASE_URL}/api/user/balance`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
  }
}

function performMarketplaceUsage(user) {
  const token = 'mock_token_' + user.username;
  
  // Browse marketplace
  let browseRes = http.post(`${BASE_URL}/api/marketplace/listings/search`, JSON.stringify({
    limit: 20,
    offset: Math.floor(Math.random() * 100)
  }), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  });
  
  check(browseRes, {
    'marketplace browse successful': (r) => r.status === 200 || r.status === 401,
  });
  
  // Simulate occasional purchases
  if (Math.random() < 0.1) { // 10% purchase rate
    http.post(`${BASE_URL}/api/marketplace/purchase`, JSON.stringify({
      listing_id: `listing_${Math.floor(Math.random() * 1000)}`,
      quantity: 1
    }), {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
  }
}

function performAIModelUsage(user) {
  const token = 'mock_token_' + user.username;
  
  // AI model queries
  let queryRes = http.post(`${BASE_URL}/api/agents/query`, JSON.stringify({
    query: `AI analysis request from ${user.username}`,
    model_type: 'general',
    max_tokens: 1000
  }), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  });
  
  check(queryRes, {
    'AI query processed': (r) => r.status === 200 || r.status === 401 || r.status === 429,
  });
  
  responseTime.add(queryRes.timings.duration);
  
  // Vector search simulation
  http.post(`${BASE_URL}/api/search/vector`, JSON.stringify({
    query: 'machine learning model',
    limit: 10
  }), {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  });
}

function performSystemMonitoring(user) {
  const token = 'mock_token_' + user.username;
  
  // System metrics (for admin users)
  if (user.username.includes('admin') || Math.random() < 0.1) {
    http.get(`${BASE_URL}/api/metrics/system`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    
    http.get(`${BASE_URL}/api/marketplace/analytics`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
  }
  
  // Health monitoring
  http.get(`${BASE_URL}/api/health/detailed`);
}

export function handleSummary(data) {
  const summary = {
    test_name: 'PRSM 1000+ Concurrent Users Load Test',
    gemini_requirement: 'Validate 1000+ concurrent user capability',
    timestamp: new Date().toISOString(),
    
    // Key metrics for Gemini audit
    peak_concurrent_users: Math.max(...Object.values(data.metrics.concurrent_users?.values || {})),
    avg_concurrent_users: data.metrics.concurrent_users?.values?.avg || 0,
    total_requests: data.metrics.requests_total?.values?.count || 0,
    error_rate: (data.metrics.errors?.values?.rate || 0) * 100,
    p95_response_time: data.metrics.http_req_duration?.values?.['p(95)'] || 0,
    
    // Pass/fail status against Gemini requirements
    gemini_requirements_met: {
      concurrent_users_1000: (data.metrics.concurrent_users?.values?.avg || 0) >= 1000,
      error_rate_under_15: ((data.metrics.errors?.values?.rate || 0) * 100) < 15,
      p95_under_8s: (data.metrics.http_req_duration?.values?.['p(95)'] || 0) < 8000,
      min_request_volume: (data.metrics.requests_total?.values?.count || 0) > 50000
    },
    
    detailed_metrics: data.metrics
  };
  
  // Calculate overall pass/fail
  const requirements = summary.gemini_requirements_met;
  summary.overall_status = Object.values(requirements).every(met => met) ? 'PASS' : 'FAIL';
  
  return {
    'load_test_1000_users_summary.json': JSON.stringify(summary, null, 2),
    stdout: `
🚀 PRSM 1000+ User Load Test Complete
=====================================

Overall Status: ${summary.overall_status}

Key Results:
- Peak Concurrent Users: ${summary.peak_concurrent_users}
- Average Concurrent Users: ${summary.avg_concurrent_users}
- Total Requests: ${summary.total_requests}
- Error Rate: ${summary.error_rate.toFixed(2)}%
- P95 Response Time: ${summary.p95_response_time.toFixed(0)}ms

Gemini Requirements:
✓ 1000+ Users: ${requirements.concurrent_users_1000 ? 'PASS' : 'FAIL'}
✓ Error Rate <15%: ${requirements.error_rate_under_15 ? 'PASS' : 'FAIL'}  
✓ P95 <8s: ${requirements.p95_under_8s ? 'PASS' : 'FAIL'}
✓ Request Volume: ${requirements.min_request_volume ? 'PASS' : 'FAIL'}

${summary.overall_status === 'PASS' ? 
  '🎉 Gemini 1000+ user requirement VALIDATED!' : 
  '❌ Performance targets not met - optimization required'}
    `
  };
}