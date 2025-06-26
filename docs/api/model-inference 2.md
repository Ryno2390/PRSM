# Model Inference API

Execute inference across multiple LLM providers with PRSM's unified model inference infrastructure.

## 🎯 Overview

The Model Inference API provides a unified interface for executing inference across multiple language model providers including OpenAI, Anthropic, Hugging Face, and custom models. It handles provider abstraction, load balancing, fallback strategies, and cost optimization.

## 📋 Base URL

```
https://api.prsm.ai/v1/inference
```

## 🔐 Authentication

All requests require authentication via API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.ai/v1/inference
```

## 🚀 Quick Start

### Basic Inference Request

```python
import prsm

client = prsm.Client(api_key="your-api-key")

response = client.inference.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=150
)

print(response.content)
```

## 📊 Endpoints

### POST /inference
Execute a model inference request.

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user", 
      "content": "Explain quantum computing"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "provider": "openai",
  "fallback_providers": ["anthropic", "huggingface"],
  "cost_optimization": true
}
```

**Response:**
```json
{
  "id": "inf_abc123",
  "object": "inference",
  "created": 1703521234,
  "model": "gpt-3.5-turbo",
  "provider": "openai",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is a revolutionary..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 135,
    "total_tokens": 150
  },
  "cost": {
    "amount": 0.0023,
    "currency": "USD",
    "provider_cost": 0.0025,
    "savings": 0.0002
  }
}
```

### GET /inference/{inference_id}
Retrieve details about a specific inference request.

**Response:**
```json
{
  "id": "inf_abc123",
  "status": "completed",
  "model": "gpt-3.5-turbo",
  "provider": "openai",
  "created": 1703521234,
  "completed": 1703521236,
  "latency_ms": 2150,
  "result": {
    "content": "Quantum computing is a revolutionary...",
    "tokens": 150,
    "cost": 0.0023
  }
}
```

### POST /inference/batch
Execute multiple inference requests in batch.

**Request Body:**
```json
{
  "requests": [
    {
      "id": "req_1",
      "model": "gpt-3.5-turbo",
      "messages": [{"role": "user", "content": "Question 1"}]
    },
    {
      "id": "req_2", 
      "model": "claude-3-sonnet",
      "messages": [{"role": "user", "content": "Question 2"}]
    }
  ],
  "parallel": true,
  "fallback_strategy": "best_available"
}
```

### POST /inference/stream
Execute streaming inference with real-time token generation.

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Write a story about AI"}
  ],
  "stream": true,
  "max_tokens": 500
}
```

**Streaming Response:**
```
data: {"id":"inf_stream_123","object":"inference.delta","choices":[{"delta":{"content":"Once"}}]}

data: {"id":"inf_stream_123","object":"inference.delta","choices":[{"delta":{"content":" upon"}}]}

data: {"id":"inf_stream_123","object":"inference.delta","choices":[{"delta":{"content":" a"}}]}
```

## 🎛️ Model Configuration

### Supported Models

**OpenAI:**
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-16k`

**Anthropic:**
- `claude-3-opus`
- `claude-3-sonnet`
- `claude-3-haiku`
- `claude-2.1`

**Hugging Face:**
- `microsoft/DialoGPT-large`
- `meta-llama/Llama-2-70b-chat-hf`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

**Custom Models:**
- Upload and host your own models
- Fine-tuned model support
- Local model integration

### Provider Configuration

```python
# Configure provider preferences
client.inference.configure_providers({
    "primary": "openai",
    "fallback": ["anthropic", "huggingface"],
    "cost_optimization": True,
    "latency_optimization": False,
    "quality_threshold": 0.85
})
```

## ⚙️ Advanced Features

### Cost Optimization

```python
# Enable automatic cost optimization
response = client.inference.create(
    model="gpt-3.5-turbo",
    messages=messages,
    cost_optimization={
        "enabled": True,
        "max_cost": 0.10,
        "prefer_cheaper": True,
        "quality_threshold": 0.8
    }
)
```

### Fallback Strategies

```python
# Configure sophisticated fallback logic
response = client.inference.create(
    model="gpt-4",
    messages=messages,
    fallback_config={
        "strategies": [
            {"provider": "openai", "timeout": 5000},
            {"provider": "anthropic", "timeout": 10000},
            {"provider": "huggingface", "model": "mixtral"}
        ],
        "retry_attempts": 3,
        "backoff_multiplier": 2.0
    }
)
```

### Caching and Performance

```python
# Enable response caching
response = client.inference.create(
    model="gpt-3.5-turbo",
    messages=messages,
    cache_config={
        "enabled": True,
        "ttl": 3600,  # 1 hour
        "key_strategy": "semantic_similarity",
        "similarity_threshold": 0.95
    }
)
```

## 📊 Monitoring and Analytics

### Usage Analytics

```python
# Get usage statistics
analytics = client.inference.analytics(
    timeframe="last_30_days",
    metrics=["requests", "tokens", "costs", "latency"]
)

print(f"Total requests: {analytics.total_requests}")
print(f"Average latency: {analytics.avg_latency_ms}ms")
print(f"Total cost: ${analytics.total_cost}")
```

### Performance Monitoring

```python
# Monitor model performance
performance = client.inference.performance(
    model="gpt-3.5-turbo",
    metrics=["success_rate", "avg_latency", "p95_latency"]
)
```

## 🔧 Error Handling

### Common Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `400` | Invalid request parameters | Check request format |
| `401` | Authentication failed | Verify API key |
| `402` | Insufficient credits | Add credits to account |
| `429` | Rate limit exceeded | Implement backoff |
| `500` | Internal server error | Retry request |
| `503` | Provider unavailable | Use fallback provider |

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Missing required parameter: model",
    "details": {
      "parameter": "model",
      "expected_type": "string"
    }
  }
}
```

### Error Handling Example

```python
try:
    response = client.inference.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
except prsm.errors.InvalidRequestError as e:
    print(f"Invalid request: {e.message}")
except prsm.errors.RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}")
except prsm.errors.ProviderError as e:
    print(f"Provider error: {e.provider} - {e.message}")
```

## 🛠️ SDK Examples

### Python SDK

```python
import prsm
import asyncio

# Async inference
async def async_inference():
    client = prsm.AsyncClient(api_key="your-key")
    
    response = await client.inference.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Explain machine learning"}
        ]
    )
    
    return response.content

# Batch processing
responses = client.inference.batch([
    {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": q}]}
    for q in questions
])
```

### JavaScript SDK

```javascript
import { PRSM } from '@prsm/sdk';

const client = new PRSM({
  apiKey: 'your-api-key'
});

// Streaming inference
const stream = await client.inference.stream({
  model: 'gpt-3.5-turbo',
  messages: [
    { role: 'user', content: 'Write a poem about coding' }
  ]
});

for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

### Go SDK

```go
package main

import (
    "context"
    "github.com/Ryno2390/PRSM/sdks/go"
)

func main() {
    client := prsm.NewClient("your-api-key")
    
    response, err := client.Inference.Create(context.Background(), &prsm.InferenceRequest{
        Model: "gpt-3.5-turbo",
        Messages: []prsm.Message{
            {Role: "user", Content: "Explain Go programming"},
        },
    })
    
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(response.Content)
}
```

## 📈 Rate Limits

### Default Limits

| Tier | Requests/minute | Tokens/minute | Concurrent |
|------|----------------|---------------|------------|
| Free | 20 | 40,000 | 2 |
| Basic | 100 | 200,000 | 5 |
| Pro | 500 | 1,000,000 | 20 |
| Enterprise | Custom | Custom | Custom |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1703521800
X-RateLimit-Retry-After: 45
```

## 🧪 Testing

### Test Environment

```bash
# Use test environment
export PRSM_API_URL=https://api-test.prsm.ai
export PRSM_API_KEY=test_key_123
```

### Mock Responses

```python
# Enable mock mode for testing
client = prsm.Client(
    api_key="test-key",
    mock_mode=True,
    mock_responses={
        "gpt-3.5-turbo": {
            "content": "This is a mock response",
            "latency_ms": 100
        }
    }
)
```

## 📚 Additional Resources

- [Model Comparison Guide](./model-comparison.md)
- [Cost Optimization Strategies](./cost-optimization.md)
- [Performance Tuning](./performance-tuning.md)
- [Provider Integration Guide](./provider-integration.md)
- [Custom Model Deployment](./custom-models.md)

## 📞 Support

- **API Issues**: api-support@prsm.ai
- **SDK Issues**: sdk-support@prsm.ai
- **Documentation**: docs@prsm.ai
- **Status Page**: https://status.prsm.ai