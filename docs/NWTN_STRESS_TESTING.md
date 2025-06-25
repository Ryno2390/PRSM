# NWTN Orchestrator Stress Testing Framework

## Overview

The NWTN (Neural Web for Transformation Networking) Orchestrator is the core coordination system for PRSM's 5-agent pipeline. This document describes the comprehensive stress testing framework designed to validate Phase 1 requirements:

- **Target**: 1000 concurrent users
- **Latency**: <2s response time (95th percentile)
- **Success Rate**: >95% successful requests
- **Agent Pipeline**: Architect → Prompter → Router → Executor → Compiler

## Testing Architecture

### 1. Agent Pipeline Validation (`validate-nwtn-agents.py`)

Validates the coordination and performance of individual agents in the pipeline:

```bash
# Run agent validation
make nwtn-test

# Or run directly
python scripts/validate-nwtn-agents.py
```

**Validation Scenarios:**
- Simple queries (single agent execution)
- Medium complexity (3-4 agent coordination)
- Complex research tasks (full 5-agent pipeline)
- Multi-domain technical queries

**Metrics Tracked:**
- Individual agent execution times
- Agent coordination scores
- Data flow integrity
- Safety compliance
- Output quality assessment

### 2. Orchestrator Stress Testing (`nwtn-stress-test.py`)

Simulates realistic user loads to validate performance under Phase 1 requirements:

```bash
# Full stress test (1000 users, 5 minutes)
make nwtn-stress

# Quick test (100 users, 1 minute)
make nwtn-stress-quick

# Extended test (2000 users, 10 minutes)
make nwtn-stress-extended
```

**Test Phases:**
1. **Ramp-up**: Gradually increase to maximum concurrent users
2. **Steady State**: Maintain peak load for validation period
3. **Ramp-down**: Graceful reduction of concurrent users

**User Simulation:**
- Realistic query distribution (40% simple, 40% medium, 20% complex)
- Variable think times (2-10 seconds between requests)
- Mixed workload patterns
- Error injection and recovery testing

## Test Scenarios

### Simple Queries (40% of traffic)
```
"What is machine learning?"
"Explain the difference between AI and ML"
```
- **Expected Agents**: Architect, Executor
- **Target Time**: <1s
- **Context Allocation**: 50 FTNS

### Medium Complexity (40% of traffic)
```
"Create a comprehensive analysis of renewable energy trends"
"Design a microservices architecture for e-commerce"
```
- **Expected Agents**: Architect, Router, Executor, Compiler
- **Target Time**: <5s
- **Context Allocation**: 100 FTNS

### Complex Queries (20% of traffic)
```
"Research quantum computing + AI intersection with enterprise roadmap"
"Optimize distributed system for 1M concurrent requests"
```
- **Expected Agents**: Full 5-agent pipeline
- **Target Time**: <15s
- **Context Allocation**: 200 FTNS

## Performance Metrics

### Core Metrics
- **Total Requests**: All requests attempted
- **Success Rate**: Percentage of successful completions
- **Response Time**: P50, P95, P99 latency percentiles
- **Throughput**: Requests per second (RPS)
- **Concurrent Users**: Maximum sustained concurrency

### Agent-Specific Metrics
- **Architect Performance**: Task decomposition speed
- **Prompter Optimization**: Prompt enhancement quality
- **Router Efficiency**: Model selection accuracy and speed
- **Executor Throughput**: API call performance and parallelization
- **Compiler Quality**: Result synthesis coherence and speed

### System Resource Metrics
- **CPU Usage**: System-wide processor utilization
- **Memory Consumption**: RAM usage patterns and peaks
- **Network I/O**: Request/response bandwidth
- **Context Usage**: FTNS token consumption accuracy

## Phase 1 Compliance Validation

### Latency Requirements
```python
# P95 latency must be < 2000ms
assert p95_latency < 2000, "Latency requirement not met"

# Average latency should be reasonable
assert avg_latency < 1000, "Average latency too high"
```

### Concurrency Requirements
```python
# Must sustain 1000 concurrent users
assert max_concurrent_users >= 1000, "Concurrency target not reached"

# Should handle user ramp-up smoothly
assert successful_ramp_up, "User ramp-up failed"
```

### Success Rate Requirements
```python
# Success rate must exceed 95%
assert success_rate >= 0.95, "Success rate below threshold"

# Error patterns should be acceptable
assert circuit_breaker_activations == 0, "System instability detected"
```

## Running the Tests

### Prerequisites
```bash
# Install dependencies
pip install aiohttp psutil structlog

# Ensure PRSM API is running
make run

# Verify API health
curl http://localhost:8000/health
```

### Quick Validation
```bash
# Test agent pipeline coordination
make nwtn-test

# Quick stress test (100 users)
make nwtn-stress-quick

# Phase 1 validation
make test-phase1
```

### Full Stress Testing
```bash
# Complete Phase 1 validation
make nwtn-stress

# Extended testing
make nwtn-stress-extended

# Full performance suite
make test-performance
```

### Custom Testing
```bash
# Custom user count and duration
python scripts/nwtn-stress-test.py \
  --users 500 \
  --duration 180 \
  --latency 1500 \
  --success-rate 0.97

# Test against remote environment
python scripts/nwtn-stress-test.py \
  --url https://prsm-api.example.com \
  --users 1000
```

## Interpreting Results

### Success Criteria
✅ **PASSED** - All requirements met:
- P95 latency < 2000ms
- 1000+ concurrent users sustained
- >95% success rate
- All agents functioning correctly

❌ **FAILED** - Requirements not met:
- Review detailed error breakdown
- Check agent-specific performance issues
- Investigate system resource constraints
- Validate safety compliance

### Performance Ratings
- **Excellent**: P95 < 1000ms, >98% success rate
- **Good**: P95 < 1500ms, >96% success rate  
- **Needs Improvement**: P95 < 2000ms, >95% success rate
- **Poor**: Above thresholds not met

### Common Issues and Solutions

#### High Latency
```
🔴 P95 latency (2500ms) exceeds target (2000ms)
   - Optimize agent pipeline execution
   - Implement request caching
   - Consider agent parallelization
```

#### Low Success Rate
```
🔴 Success rate (92%) below target (95%)
   - Investigate error patterns
   - Implement better error handling
   - Add circuit breaker fallbacks
```

#### Agent Performance Issues
```
🟡 Executor agent P95 time high: 3.2s
   - Optimize model API calls
   - Implement connection pooling
   - Add timeout handling
```

## Continuous Integration

### Automated Testing
```yaml
# .github/workflows/nwtn-stress-test.yml
name: NWTN Stress Test
on: [push, pull_request]
jobs:
  stress-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: make install-dev
      - name: Start PRSM API
        run: make run &
      - name: Wait for API
        run: sleep 30
      - name: Run agent validation
        run: make nwtn-test
      - name: Run stress test
        run: make nwtn-stress-quick
```

### Performance Monitoring
```bash
# Continuous monitoring setup
./scripts/setup-monitoring.sh

# Performance regression detection
python scripts/compare-performance.py \
  --baseline results/baseline.json \
  --current results/current.json
```

## Troubleshooting

### API Not Responding
```bash
# Check API status
curl -v http://localhost:8000/health

# Check logs
make docker-logs

# Restart services
make docker-down && make docker-run
```

### Test Failures
```bash
# Run with verbose logging
PYTHONPATH=. python scripts/nwtn-stress-test.py --quick

# Check agent validation first
make nwtn-test

# Review detailed results
cat nwtn_stress_test_results.json
```

### Performance Issues
```bash
# Monitor system resources
htop

# Check database connections
make db-status

# Review observability stack
make obs-dashboards
```

## Best Practices

### Development Workflow
1. **Before Code Changes**: Run `make nwtn-test` to establish baseline
2. **After Changes**: Run `make nwtn-stress-quick` to validate impact
3. **Before Deployment**: Run `make test-phase1` for full validation
4. **Production**: Monitor with observability stack

### Load Testing Strategy
1. **Start Small**: Begin with `nwtn-stress-quick`
2. **Increment Gradually**: Increase user counts progressively
3. **Validate Components**: Use `nwtn-test` to isolate issues
4. **Monitor Resources**: Watch system metrics during tests
5. **Document Results**: Save test reports for comparison

### Performance Optimization
1. **Profile First**: Identify bottlenecks before optimizing
2. **Test Incrementally**: Validate each optimization
3. **Monitor Regressions**: Compare against baselines
4. **Document Changes**: Track optimization impact

## Results Analysis

### Automated Reports
The stress testing framework generates comprehensive reports:

- **Summary Dashboard**: High-level pass/fail status
- **Performance Metrics**: Detailed latency and throughput analysis
- **Agent Breakdown**: Individual agent performance assessment
- **Error Analysis**: Categorized failure patterns
- **Recommendations**: Specific optimization suggestions

### Report Files
- `nwtn_stress_test_results.json`: Complete test results
- `nwtn_agent_validation_results.json`: Agent pipeline validation
- `performance_timeline.json`: Time-series performance data
- `error_breakdown.json`: Detailed error analysis

This framework ensures PRSM's NWTN Orchestrator meets Phase 1 requirements and provides a foundation for continuous performance validation and optimization.