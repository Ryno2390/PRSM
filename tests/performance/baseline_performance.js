import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

export const options = {
  stages: [
    { duration: '2m', target: 5 },   // Ramp up to 5 users
    { duration: '5m', target: 5 },   // Stay at 5 users
    { duration: '2m', target: 10 },  // Ramp up to 10 users
    { duration: '5m', target: 10 },  // Stay at 10 users
    { duration: '1m', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.05'],    // Error rate under 5%
    response_time: ['p(95)<2000'],     // Custom metric threshold
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Test basic health endpoint
  let healthRes = http.get(`${BASE_URL}/health`);
  
  check(healthRes, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  // Record custom metrics
  errorRate.add(healthRes.status !== 200);
  responseTime.add(healthRes.timings.duration);
  
  // Test API endpoints if available
  let apiRes = http.get(`${BASE_URL}/api/health`, {
    headers: { 'Accept': 'application/json' }
  });
  
  check(apiRes, {
    'api health status is 200': (r) => r.status === 200 || r.status === 404,
  });
  
  // Random sleep between 1-3 seconds
  sleep(Math.random() * 2 + 1);
}

export function handleSummary(data) {
  return {
    'baseline_summary.json': JSON.stringify(data, null, 2),
  };
}