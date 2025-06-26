# Model Comparison API

The Model Comparison API enables systematic evaluation and comparison of different AI models across various metrics and use cases.

## Overview

Compare models based on:
- Performance metrics (accuracy, speed, quality)
- Cost efficiency 
- Capability matching
- Resource requirements
- User ratings and feedback

## Endpoints

### Compare Models

#### Compare Multiple Models
```http
POST /v1/models/compare
```

**Request Body:**
```json
{
  "models": ["gpt-4", "claude-3", "llama-2-70b"],
  "evaluation_criteria": {
    "tasks": ["text_generation", "reasoning", "code_generation"],
    "metrics": ["accuracy", "speed", "cost", "quality"],
    "benchmarks": ["mmlu", "hellaswag", "humaneval"]
  },
  "test_inputs": [
    {
      "task": "text_generation",
      "prompt": "Write a technical blog post introduction"
    }
  ]
}
```

**Response:**
```json
{
  "comparison_id": "comp_12345",
  "models": {
    "gpt-4": {
      "overall_score": 8.7,
      "metrics": {
        "accuracy": 9.2,
        "speed": 7.8,
        "cost": 6.5,
        "quality": 9.5
      },
      "estimated_cost_per_1k_tokens": 0.03,
      "avg_latency_ms": 1200
    },
    "claude-3": {
      "overall_score": 8.5,
      "metrics": {
        "accuracy": 9.0,
        "speed": 8.2,
        "cost": 7.0,
        "quality": 9.3
      },
      "estimated_cost_per_1k_tokens": 0.025,
      "avg_latency_ms": 950
    }
  },
  "recommendations": [
    {
      "use_case": "cost_sensitive",
      "recommended_model": "claude-3",
      "reason": "Best cost-performance ratio"
    }
  ]
}
```

#### Get Model Capabilities
```http
GET /v1/models/{model_id}/capabilities
```

**Response:**
```json
{
  "model_id": "gpt-4",
  "capabilities": {
    "text_generation": {
      "supported": true,
      "quality_score": 9.5,
      "max_tokens": 4096
    },
    "code_generation": {
      "supported": true,
      "quality_score": 8.8,
      "languages": ["python", "javascript", "rust", "go"]
    },
    "multimodal": {
      "supported": true,
      "image_understanding": true,
      "image_generation": false
    }
  }
}
```

### Benchmarking

#### Run Benchmark Suite
```http
POST /v1/models/{model_id}/benchmark
```

**Request Body:**
```json
{
  "benchmark_suite": "comprehensive",
  "tasks": ["mmlu", "hellaswag", "arc", "truthfulqa"],
  "custom_prompts": [
    {
      "category": "reasoning",
      "prompt": "If all cats are animals and some animals are pets...",
      "expected_response_type": "logical_conclusion"
    }
  ]
}
```

## Python SDK

```python
from prsm_sdk import PRSMClient

client = PRSMClient(api_key="your-api-key")

# Compare models
comparison = await client.models.compare(
    models=["gpt-4", "claude-3"],
    tasks=["text_generation", "reasoning"],
    test_inputs=[
        {"prompt": "Explain quantum computing"}
    ]
)

print(f"Best model for cost: {comparison.recommendations['cost_effective']}")

# Get model capabilities
capabilities = await client.models.get_capabilities("gpt-4")
print(f"Supports code generation: {capabilities.code_generation.supported}")

# Run benchmarks
benchmark_results = await client.models.benchmark(
    model_id="custom-model-v1",
    benchmark_suite="standard"
)
```

## Related Documentation

- [Model Inference API](./model-inference.md)
- [Cost Optimization API](./cost-optimization.md)
- [Performance Tuning Guide](./performance-tuning.md)