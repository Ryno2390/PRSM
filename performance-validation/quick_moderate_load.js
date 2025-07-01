
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const requestCounter = new Counter('requests_total');
const responseTime = new Trend('response_time');

export const options = {
  stages: [
    { duration: '30s', target: 25 },
    { duration: '5m', target: 100 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'],
    http_req_failed: ['rate<0.10'],
    response_time: ['p(95)<5000'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  requestCounter.add(1);
  
  // Basic health check
  let healthRes = http.get(`${BASE_URL}/health`);
  
  check(healthRes, {
    'health status is 200': (r) => r.status === 200,
    'response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  errorRate.add(healthRes.status !== 200);
  responseTime.add(healthRes.timings.duration);
  
  // Simulate API usage
  if (Math.random() < 0.7) {
    let apiRes = http.post(`${BASE_URL}/api/auth/login`, JSON.stringify({
      username: `test_user_${__VU}`,
      password: 'test_password'
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
    
    check(apiRes, {
      'auth response acceptable': (r) => r.status === 200 || r.status === 401,
    });
  }
  
  // Variable sleep based on load
  sleep(Math.random() * 2 + 0.5);
}

export function handleSummary(data) {
  const summary = {
    scenario: 'moderate_load',
    max_concurrent_users: 100,
    total_requests: data.metrics.requests_total?.values?.count || 0,
    error_rate: (data.metrics.errors?.values?.rate || 0) * 100,
    avg_response_time: data.metrics.http_req_duration?.values?.avg || 0,
    p95_response_time: data.metrics.http_req_duration?.values?.['p(95)'] || 0,
    requests_per_second: data.metrics.http_reqs?.values?.rate || 0,
    
    performance_grade: {
      concurrent_users: 100,
      error_rate_pct: (data.metrics.errors?.values?.rate || 0) * 100,
      p95_response_ms: data.metrics.http_req_duration?.values?.['p(95)'] || 0,
      rps: data.metrics.http_reqs?.values?.rate || 0
    }
  };
  
  return {
    'quick_moderate_load_summary.json': JSON.stringify(summary, null, 2),
  };
}
