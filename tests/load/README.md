# PRSM Load Testing

This directory contains Locust-based load tests for the PRSM API.

## Prerequisites

```bash
pip install locust
```

## Running Tests

### Quick Test (Headless)

```bash
# 50 users, 5/second ramp-up, 2 minute duration
locust -f tests/load/locustfile.py --headless -u 50 -r 5 -t 120s

# 100 users, 10/second ramp-up, 5 minute duration
locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 300s
```

### Interactive Test (Web UI)

```bash
locust -f tests/load/locustfile.py
# Open http://localhost:8089 in your browser
```

### Distributed Testing

For high-volume testing across multiple machines:

```bash
# On master machine
locust -f tests/load/locustfile.py --master --expect-workers=4

# On each worker machine
locust -f tests/load/locustfile.py --worker --master-host=<master-ip>
```

## Test Scenarios

### PRSMUser (Default)
Simulates typical user behavior with weighted tasks:
- Health checks (10x weight) - Most frequent
- FTNS balance (5x weight)
- Marketplace browse (3x weight)
- AI queries (2x weight) - Less frequent due to cost
- Metrics (1x weight)

### PRSMPowerUser
Simulates high-activity users:
- Shorter wait times between requests
- More queries and marketplace interactions

### APIStressTest
Aggressive stress testing:
- Minimal wait times
- Focus on identifying breaking points

## Performance Baselines (Phase 7)

| Endpoint         | Median  | 99th %ile | Notes                    |
|-----------------|---------|-----------|--------------------------|
| /health         | < 10ms  | < 50ms    | Should be very fast      |
| /api/v1/ftns/*  | < 50ms  | < 200ms   | Database queries         |
| /api/v1/marketplace/* | < 100ms | < 500ms | Complex queries       |
| /api/v1/query   | < 2000ms | < 5000ms | External AI API calls    |

## Interpreting Results

### Good Performance
- All response times within baseline
- Zero errors
- Linear scaling with users

### Warning Signs
- Response times increasing exponentially
- High error rates (> 1%)
- Connection timeouts
- 503 Service Unavailable responses

### Common Bottlenecks
1. Database connection pool exhaustion
2. External API rate limits
3. Memory pressure
4. CPU saturation

## Configuration

Environment variables:
- `PRSM_LOAD_TEST_HOST`: Target host (default: http://localhost:8000)
- `PRSM_LOAD_TEST_USERS`: Number of users (default: 50)
- `PRSM_LOAD_TEST_SPAWN_RATE`: Users per second (default: 5)
