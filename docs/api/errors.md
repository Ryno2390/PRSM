# Error Handling and Status Codes

This document describes the error handling patterns, status codes, and error response formats used throughout the PRSM API.

## HTTP Status Codes

PRSM API uses standard HTTP status codes to indicate success or failure:

### Success Codes
- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `202 Accepted` - Request accepted for processing
- `204 No Content` - Request succeeded with no response body

### Client Error Codes
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict
- `422 Unprocessable Entity` - Valid request but semantic errors
- `429 Too Many Requests` - Rate limit exceeded

### Server Error Codes
- `500 Internal Server Error` - Unexpected server error
- `502 Bad Gateway` - Upstream service error
- `503 Service Unavailable` - Service temporarily unavailable
- `504 Gateway Timeout` - Upstream service timeout

## Error Response Format

All error responses follow a consistent JSON format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error description",
    "details": {
      "field": "Additional context or field-specific errors"
    },
    "request_id": "req_12345",
    "timestamp": "2025-06-22T10:30:00Z"
  }
}
```

## Error Categories

### Authentication Errors

#### INVALID_API_KEY
```json
{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid or expired",
    "details": {
      "key_format": "API keys should start with 'prsm_'"
    }
  }
}
```

#### INSUFFICIENT_PERMISSIONS
```json
{
  "error": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "message": "Your API key does not have permission to access this resource",
    "details": {
      "required_permissions": ["models.infer"],
      "current_permissions": ["models.read"]
    }
  }
}
```

### Resource Errors

#### MODEL_NOT_FOUND
```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "The specified model does not exist or is not available",
    "details": {
      "model_id": "nonexistent-model",
      "available_models": ["gpt-4", "claude-3", "llama-2"]
    }
  }
}
```

#### RESOURCE_UNAVAILABLE
```json
{
  "error": {
    "code": "RESOURCE_UNAVAILABLE",
    "message": "The requested resource is temporarily unavailable",
    "details": {
      "resource_type": "gpu_cluster",
      "estimated_availability": "2025-06-22T11:00:00Z"
    }
  }
}
```

### Validation Errors

#### INVALID_INPUT
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "One or more input parameters are invalid",
    "details": {
      "validation_errors": [
        {
          "field": "max_tokens",
          "error": "Value must be between 1 and 4096",
          "provided_value": 5000
        },
        {
          "field": "temperature",
          "error": "Value must be between 0.0 and 2.0",
          "provided_value": 3.5
        }
      ]
    }
  }
}
```

#### REQUEST_TOO_LARGE
```json
{
  "error": {
    "code": "REQUEST_TOO_LARGE",
    "message": "Request payload exceeds maximum allowed size",
    "details": {
      "max_size_bytes": 1048576,
      "actual_size_bytes": 2097152
    }
  }
}
```

### Rate Limiting Errors

#### RATE_LIMIT_EXCEEDED
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "API rate limit exceeded",
    "details": {
      "limit_type": "requests_per_minute",
      "limit": 1000,
      "reset_time": "2025-06-22T10:31:00Z",
      "retry_after_seconds": 60
    }
  }
}
```

### Budget and Cost Errors

#### BUDGET_EXCEEDED
```json
{
  "error": {
    "code": "BUDGET_EXCEEDED",
    "message": "Request would exceed your current budget limit",
    "details": {
      "current_budget": 1000.00,
      "remaining_budget": 15.50,
      "request_cost": 25.00,
      "budget_period": "monthly"
    }
  }
}
```

#### INSUFFICIENT_CREDITS
```json
{
  "error": {
    "code": "INSUFFICIENT_CREDITS",
    "message": "Insufficient FTNS credits to complete request",
    "details": {
      "required_credits": 100,
      "available_credits": 25,
      "credit_purchase_url": "https://prsm.network/credits"
    }
  }
}
```

### Provider and Infrastructure Errors

#### PROVIDER_ERROR
```json
{
  "error": {
    "code": "PROVIDER_ERROR",
    "message": "Error from upstream AI provider",
    "details": {
      "provider": "openai",
      "provider_error": "Rate limit exceeded",
      "retry_recommended": true,
      "retry_after_seconds": 60
    }
  }
}
```

#### INFRASTRUCTURE_ERROR
```json
{
  "error": {
    "code": "INFRASTRUCTURE_ERROR",
    "message": "Internal infrastructure error",
    "details": {
      "error_type": "timeout",
      "component": "model_inference_service",
      "incident_id": "inc_67890"
    }
  }
}
```

## Error Handling Best Practices

### Retry Logic

```python
import asyncio
import random
from prsm_sdk import PRSMClient, PRSMError

async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except PRSMError as e:
            if e.code in ['RATE_LIMIT_EXCEEDED', 'PROVIDER_ERROR'] and attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                continue
            raise

# Usage
client = PRSMClient(api_key="your-api-key")

result = await retry_with_backoff(
    lambda: client.models.infer("gpt-4", "Hello world")
)
```

### Error Handling Patterns

```python
from prsm_sdk import PRSMClient, PRSMError, BudgetExceededError, RateLimitError

client = PRSMClient(api_key="your-api-key")

try:
    result = await client.models.infer("gpt-4", "Analyze this text")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e.remaining_budget} remaining")
    # Handle budget exhaustion
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after_seconds} seconds")
    # Handle rate limiting
except PRSMError as e:
    print(f"API error: {e.code} - {e.message}")
    # Handle other API errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

### Graceful Degradation

```python
async def robust_inference(prompt, preferred_model="gpt-4"):
    fallback_models = ["gpt-4", "claude-3", "gpt-3.5-turbo"]
    
    for model in fallback_models:
        try:
            return await client.models.infer(model, prompt)
        except PRSMError as e:
            if e.code == "MODEL_NOT_FOUND":
                continue  # Try next model
            elif e.code == "BUDGET_EXCEEDED":
                # Try cheaper model
                if model == "gpt-4":
                    continue
                else:
                    raise  # No cheaper options
            else:
                raise  # Don't retry for other errors
    
    raise PRSMError("No available models for inference")
```

## SDK Error Classes

### Python SDK

```python
class PRSMError(Exception):
    """Base exception for PRSM API errors"""
    def __init__(self, message, code=None, details=None, request_id=None):
        self.message = message
        self.code = code
        self.details = details or {}
        self.request_id = request_id
        super().__init__(message)

class AuthenticationError(PRSMError):
    """Authentication-related errors"""
    pass

class ValidationError(PRSMError):
    """Input validation errors"""
    pass

class RateLimitError(PRSMError):
    """Rate limiting errors"""
    def __init__(self, *args, retry_after_seconds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_after_seconds = retry_after_seconds

class BudgetExceededError(PRSMError):
    """Budget limit errors"""
    def __init__(self, *args, remaining_budget=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining_budget = remaining_budget
```

### JavaScript SDK

```typescript
class PRSMError extends Error {
  constructor(
    message: string,
    public code?: string,
    public details?: Record<string, any>,
    public requestId?: string
  ) {
    super(message);
    this.name = 'PRSMError';
  }
}

class AuthenticationError extends PRSMError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'AUTHENTICATION_ERROR', details);
    this.name = 'AuthenticationError';
  }
}

class RateLimitError extends PRSMError {
  constructor(
    message: string,
    public retryAfterSeconds?: number,
    details?: Record<string, any>
  ) {
    super(message, 'RATE_LIMIT_EXCEEDED', details);
    this.name = 'RateLimitError';
  }
}
```

## Monitoring and Alerting

Set up monitoring for:
- Error rate trends
- Specific error patterns
- Rate limit violations
- Budget threshold alerts
- Provider availability issues

## Related Documentation

- [Authentication Guide](./auth-security.md)
- [Rate Limiting](./monitoring.md#rate-limiting)
- [Cost Management](./cost-optimization.md)
- [SDK Documentation](./sdks/)