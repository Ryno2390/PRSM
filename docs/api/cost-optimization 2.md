# Cost Optimization API

The PRSM Cost Optimization API provides tools for monitoring, analyzing, and optimizing AI infrastructure costs across your distributed AI workloads.

## Overview

The Cost Optimization API enables:
- Real-time cost monitoring
- Budget management and alerts
- Cost analytics and reporting
- Resource optimization recommendations
- Automated cost reduction strategies

## Authentication

All API endpoints require authentication using PRSM API keys:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.network/v1/cost-optimization/budget
```

## Endpoints

### Budget Management

#### Get Current Budget Status
```http
GET /v1/cost-optimization/budget
```

**Response:**
```json
{
  "budget_id": "budget_12345",
  "total_budget": 1000.00,
  "spent": 450.75,
  "remaining": 549.25,
  "period": "monthly",
  "spending_rate": 15.02,
  "projected_exhaustion": "2025-07-15T10:30:00Z"
}
```

#### Set Budget Limits
```http
PUT /v1/cost-optimization/budget
```

**Request Body:**
```json
{
  "total_budget": 1500.00,
  "period": "monthly",
  "alert_thresholds": [0.5, 0.8, 0.95],
  "auto_throttle_enabled": true
}
```

### Cost Analytics

#### Get Cost Breakdown
```http
GET /v1/cost-optimization/analytics/breakdown?period=30d
```

**Response:**
```json
{
  "period": "30d",
  "total_cost": 450.75,
  "breakdown": {
    "api_calls": 280.50,
    "compute": 120.25,
    "storage": 35.00,
    "network": 15.00
  },
  "top_consumers": [
    {
      "service": "model_inference",
      "cost": 200.30,
      "percentage": 44.5
    }
  ]
}
```

#### Get Optimization Recommendations
```http
GET /v1/cost-optimization/recommendations
```

**Response:**
```json
{
  "recommendations": [
    {
      "type": "model_routing",
      "description": "Route 30% of requests to more cost-effective models",
      "estimated_savings": 85.50,
      "confidence": 0.87,
      "implementation_effort": "low"
    }
  ]
}
```

### Resource Optimization

#### Optimize Request Routing
```http
POST /v1/cost-optimization/route-request
```

**Request Body:**
```json
{
  "request": {
    "text": "Analyze this document",
    "model_requirements": {
      "task": "text_analysis",
      "min_quality": 0.8
    }
  },
  "constraints": {
    "max_cost": 0.05,
    "max_latency": 5.0
  }
}
```

**Response:**
```json
{
  "selected_model": "efficient-analyzer-v2",
  "estimated_cost": 0.035,
  "estimated_latency": 2.1,
  "cost_savings": 0.015,
  "provider": "local_cluster"
}
```

## SDKs

### Python SDK

```python
from prsm_sdk import PRSMClient

client = PRSMClient(api_key="your-api-key")

# Get budget status
budget = await client.cost_optimization.get_budget()
print(f"Remaining budget: ${budget.remaining}")

# Get cost breakdown
breakdown = await client.cost_optimization.get_cost_breakdown(
    period="30d"
)

# Optimize request routing
optimized = await client.cost_optimization.optimize_request(
    text="Analyze sentiment",
    max_cost=0.02
)
```

### JavaScript SDK

```javascript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({ apiKey: 'your-api-key' });

// Get budget status
const budget = await client.costOptimization.getBudget();
console.log(`Remaining budget: $${budget.remaining}`);

// Get optimization recommendations
const recommendations = await client.costOptimization.getRecommendations();
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `429` - Rate Limited
- `500` - Internal Server Error

Error responses include detailed information:

```json
{
  "error": {
    "code": "BUDGET_EXCEEDED",
    "message": "Request would exceed monthly budget limit",
    "details": {
      "current_budget": 1000.00,
      "remaining": 15.50,
      "request_cost": 25.00
    }
  }
}
```

## Rate Limits

- 1000 requests per minute per API key
- Cost optimization endpoints: 100 requests per minute
- Analytics endpoints: 50 requests per minute

## Webhooks

Subscribe to cost-related events:

```json
{
  "event": "budget.threshold_reached",
  "data": {
    "budget_id": "budget_12345",
    "threshold": 0.8,
    "current_usage": 0.82,
    "remaining_budget": 180.00
  }
}
```

## Best Practices

1. **Monitor Regularly**: Check budget status daily
2. **Set Alerts**: Configure multiple threshold alerts
3. **Use Optimization**: Leverage request routing optimization
4. **Analyze Trends**: Review cost analytics weekly
5. **Implement Recommendations**: Act on optimization suggestions

## Related Documentation

- [Model Inference API](./model-inference.md)
- [Performance Tuning Guide](./performance-tuning.md)
- [Budget Management Tutorial](../tutorials/02-foundation/configuration.md)