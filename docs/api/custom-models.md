# Custom Model Deployment API

The Custom Model Deployment API enables organizations to deploy, manage, and serve their own AI models within the PRSM network.

## Overview

Deploy and manage:
- Custom fine-tuned models
- Proprietary model architectures
- Domain-specific models
- Private model endpoints
- Model versioning and rollback

## Endpoints

### Model Deployment

#### Deploy Custom Model
```http
POST /v1/custom-models/deploy
```

**Request Body:**
```json
{
  "model_name": "company-legal-analyzer-v2",
  "model_type": "text-classification",
  "deployment_config": {
    "container_image": "company/legal-analyzer:v2.1",
    "resource_requirements": {
      "cpu": "2",
      "memory": "8Gi",
      "gpu": "1"
    },
    "scaling": {
      "min_replicas": 1,
      "max_replicas": 10,
      "target_utilization": 0.7
    }
  },
  "model_metadata": {
    "description": "Legal document analysis model",
    "version": "2.1.0",
    "capabilities": ["classification", "entity-extraction"],
    "domains": ["legal", "compliance"]
  }
}
```

**Response:**
```json
{
  "deployment_id": "deploy_12345",
  "model_id": "custom_legal_analyzer_v2",
  "status": "deploying",
  "endpoint_url": "https://api.prsm.ai/v1/models/custom_legal_analyzer_v2/infer",
  "estimated_deployment_time": "5-10 minutes",
  "resource_allocation": {
    "cpu": "2",
    "memory": "8Gi",
    "gpu": "1 x NVIDIA T4"
  }
}
```

#### Get Deployment Status
```http
GET /v1/custom-models/deployments/{deployment_id}
```

**Response:**
```json
{
  "deployment_id": "deploy_12345",
  "status": "running",
  "health_status": "healthy",
  "replicas": {
    "desired": 2,
    "ready": 2,
    "available": 2
  },
  "performance_metrics": {
    "requests_per_second": 15.2,
    "avg_latency_ms": 450,
    "error_rate": 0.001
  },
  "resource_usage": {
    "cpu_percent": 65,
    "memory_percent": 72,
    "gpu_percent": 45
  }
}
```

### Model Management

#### Update Model
```http
PUT /v1/custom-models/{model_id}
```

#### Scale Model
```http
POST /v1/custom-models/{model_id}/scale
```

**Request Body:**
```json
{
  "target_replicas": 5,
  "scaling_strategy": "gradual"
}
```

### Model Inference

#### Invoke Custom Model
```http
POST /v1/models/{custom_model_id}/infer
```

**Request Body:**
```json
{
  "inputs": {
    "text": "This is a legal document for review",
    "document_type": "contract"
  },
  "parameters": {
    "confidence_threshold": 0.8,
    "return_entities": true
  }
}
```

**Response:**
```json
{
  "model_id": "custom_legal_analyzer_v2",
  "outputs": {
    "classification": "commercial_contract",
    "confidence": 0.92,
    "entities": [
      {
        "type": "party",
        "text": "ABC Corp",
        "start": 45,
        "end": 53
      }
    ]
  },
  "metadata": {
    "inference_time_ms": 234,
    "model_version": "2.1.0"
  }
}
```

## Python SDK

```python
from prsm_sdk import PRSMClient

client = PRSMClient(api_key="your-api-key")

# Deploy custom model
deployment = await client.custom_models.deploy(
    model_name="legal-analyzer-v2",
    container_image="company/legal-analyzer:v2.1",
    resource_requirements={
        "cpu": "2",
        "memory": "8Gi",
        "gpu": "1"
    }
)

print(f"Deployment ID: {deployment.deployment_id}")

# Check deployment status
status = await client.custom_models.get_deployment_status(
    deployment.deployment_id
)
print(f"Status: {status.status}")

# Use deployed model
result = await client.models.infer(
    model_id="custom_legal_analyzer_v2",
    inputs={"text": "Contract text here"}
)
```

## Security and Access Control

### Private Models
- Models are private to your organization by default
- Fine-grained access control with roles and permissions
- Secure model storage and execution
- Audit logging for all model operations

### Model Sharing
```http
POST /v1/custom-models/{model_id}/share
```

**Request Body:**
```json
{
  "share_type": "organization",
  "permissions": ["read", "infer"],
  "allowed_organizations": ["partner-org-id"],
  "usage_limits": {
    "requests_per_day": 1000,
    "total_requests": 10000
  }
}
```

## Best Practices

1. **Resource Sizing**: Start with conservative resource allocations
2. **Health Checks**: Implement proper health check endpoints
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Versioning**: Use semantic versioning for model releases
5. **Testing**: Thoroughly test models before production deployment

## Related Documentation

- [Model Inference API](./model-inference.md)
- [Performance Tuning API](./performance-tuning.md)
- [Security Guide](../SECURITY_HARDENING.md)