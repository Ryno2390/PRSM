# Provider Integration API

The Provider Integration API enables seamless integration with multiple AI model providers, allowing dynamic routing and unified access across different AI services.

## Supported Providers

- OpenAI (GPT models)
- Anthropic (Claude models)
- Local models (Llama, Mistral)
- Azure OpenAI
- Google Vertex AI
- Hugging Face
- Custom endpoints

## Endpoints

### Provider Management

#### List Available Providers
```http
GET /v1/providers
```

**Response:**
```json
{
  "providers": [
    {
      "id": "openai",
      "name": "OpenAI",
      "status": "active",
      "models": ["gpt-4", "gpt-3.5-turbo"],
      "capabilities": ["text-generation", "chat"],
      "current_cost_per_1k_tokens": 0.03,
      "avg_latency_ms": 1200
    },
    {
      "id": "anthropic",
      "name": "Anthropic",
      "status": "active",
      "models": ["claude-3-opus", "claude-3-sonnet"],
      "capabilities": ["text-generation", "chat", "analysis"],
      "current_cost_per_1k_tokens": 0.025,
      "avg_latency_ms": 950
    }
  ]
}
```

#### Configure Provider
```http
PUT /v1/providers/{provider_id}/config
```

**Request Body:**
```json
{
  "api_key": "sk-...",
  "endpoint_url": "https://api.openai.com/v1",
  "rate_limits": {
    "requests_per_minute": 3000,
    "tokens_per_minute": 150000
  },
  "retry_config": {
    "max_retries": 3,
    "backoff_strategy": "exponential"
  }
}
```

### Intelligent Routing

#### Route Request
```http
POST /v1/providers/route
```

**Request Body:**
```json
{
  "prompt": "Explain quantum computing",
  "requirements": {
    "task_type": "explanation",
    "max_cost": 0.02,
    "max_latency_ms": 2000,
    "min_quality_score": 0.85
  },
  "routing_strategy": "cost_optimized"
}
```

**Response:**
```json
{
  "selected_provider": "anthropic",
  "selected_model": "claude-3-sonnet",
  "estimated_cost": 0.018,
  "estimated_latency_ms": 950,
  "confidence_score": 0.92,
  "routing_reason": "Best cost-performance ratio for explanation tasks"
}
```

## Python SDK

```python
from prsm_sdk import PRSMClient

client = PRSMClient(api_key="your-api-key")

# List providers
providers = await client.providers.list()
for provider in providers:
    print(f"{provider.name}: {provider.status}")

# Configure provider
await client.providers.configure(
    provider_id="openai",
    api_key="sk-...",
    rate_limits={"requests_per_minute": 3000}
)

# Intelligent routing
routed = await client.providers.route(
    prompt="Analyze this text",
    requirements={
        "max_cost": 0.02,
        "task_type": "analysis"
    }
)
print(f"Using {routed.selected_provider}: {routed.selected_model}")
```

## Related Documentation

- [Model Inference API](./model-inference.md)
- [Cost Optimization API](./cost-optimization.md)
- [Performance Tuning API](./performance-tuning.md)