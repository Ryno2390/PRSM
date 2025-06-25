# Performance Tuning API

The Performance Tuning API provides tools for optimizing AI model performance, monitoring system metrics, and implementing performance best practices.

## Overview

Optimize performance through:
- Model configuration tuning
- Resource allocation optimization
- Caching strategy management
- Load balancing configuration
- Performance monitoring and alerting

## Endpoints

### Performance Configuration

#### Get Performance Profile
```http
GET /v1/performance/profile/{model_id}
```

**Response:**
```json
{
  "model_id": "gpt-4",
  "current_config": {
    "batch_size": 8,
    "max_tokens": 4096,
    "temperature": 0.7,
    "timeout_ms": 30000
  },
  "performance_metrics": {
    "avg_latency_ms": 1200,
    "throughput_rps": 25.5,
    "error_rate": 0.002,
    "cache_hit_rate": 0.75
  },
  "optimization_recommendations": [
    {
      "parameter": "batch_size",
      "current_value": 8,
      "recommended_value": 12,
      "expected_improvement": "15% latency reduction"
    }
  ]
}
```

#### Optimize Model Configuration
```http
POST /v1/performance/optimize
```

**Request Body:**
```json
{
  "model_id": "gpt-4",
  "optimization_goals": {
    "primary": "latency",
    "secondary": "cost",
    "constraints": {
      "max_cost_increase": 0.1,
      "min_quality_score": 0.9
    }
  },
  "workload_characteristics": {
    "avg_request_size": 500,
    "peak_rps": 100,
    "response_time_sla": 2000
  }
}
```

### Monitoring

#### Get Real-time Metrics
```http
GET /v1/performance/metrics?window=5m
```

**Response:**
```json
{
  "timestamp": "2025-06-22T10:30:00Z",
  "window": "5m",
  "metrics": {
    "requests_per_second": 45.2,
    "avg_response_time_ms": 950,
    "p95_response_time_ms": 1800,
    "p99_response_time_ms": 2500,
    "error_rate": 0.001,
    "cache_hit_rate": 0.82,
    "active_connections": 150
  },
  "resource_utilization": {
    "cpu_percent": 65.5,
    "memory_percent": 78.2,
    "gpu_utilization": 85.0
  }
}
```

## Python SDK

```python
from prsm_sdk import PRSMClient

client = PRSMClient(api_key="your-api-key")

# Get performance profile
profile = await client.performance.get_profile("gpt-4")
print(f"Current latency: {profile.metrics.avg_latency_ms}ms")

# Optimize configuration
optimization = await client.performance.optimize(
    model_id="gpt-4",
    goals={"primary": "latency", "secondary": "cost"},
    constraints={"max_cost_increase": 0.1}
)

# Monitor real-time metrics
metrics = await client.performance.get_metrics(window="5m")
print(f"Current RPS: {metrics.requests_per_second}")
```

## Related Documentation

- [Model Inference API](./model-inference.md)
- [Cost Optimization API](./cost-optimization.md)
- [Monitoring Guide](./monitoring.md)