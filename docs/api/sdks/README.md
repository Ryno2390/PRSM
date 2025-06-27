# PRSM SDKs

Official Software Development Kits for the PRSM API, providing idiomatic interfaces for popular programming languages.

## Available SDKs

### Python SDK
**Installation:** `pip install prsm-sdk`
- Full async/await support
- Type hints and IntelliSense
- Comprehensive error handling
- Built-in retry logic

```python
from prsm_sdk import PRSMClient

client = PRSMClient(api_key="your-api-key")
result = await client.models.infer("gpt-4", "Hello, world!")
```

### JavaScript/TypeScript SDK
**Installation:** `npm install @prsm/sdk`
- Promise-based API
- TypeScript definitions included
- Browser and Node.js support
- Streaming response support

```typescript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({ apiKey: 'your-api-key' });
const result = await client.models.infer('gpt-4', 'Hello, world!');
```

### Go SDK
**Installation:** `go get github.com/prsm-ai/prsm-go`
- Idiomatic Go interfaces
- Context support
- Channel-based streaming
- Built-in connection pooling

```go
import "github.com/prsm-ai/prsm-go"

client := prsm.NewClient("your-api-key")
result, err := client.Models.Infer(ctx, "gpt-4", "Hello, world!")
```

## SDK Features

### Common Features
All SDKs provide:
- **Authentication**: Automatic API key management
- **Error Handling**: Detailed error information and types
- **Rate Limiting**: Built-in respect for API limits
- **Retries**: Automatic retry with exponential backoff
- **Streaming**: Support for streaming responses
- **Pagination**: Automatic handling of paginated results

### Language-Specific Features

#### Python SDK
- **Async/Await**: Full asyncio support
- **Type Hints**: Complete type annotations
- **Pydantic Models**: Structured response objects
- **Context Managers**: Automatic resource cleanup

#### JavaScript SDK
- **Promise/Async**: Modern JavaScript patterns
- **TypeScript**: First-class TypeScript support
- **Browser Support**: Works in browsers with CORS
- **Node.js Streams**: Native stream support

#### Go SDK
- **Context**: Full context.Context support
- **Channels**: Channel-based streaming
- **Interfaces**: Clean Go interfaces
- **Generics**: Type-safe generic methods

## Getting Started

### 1. Get API Key
Sign up at [PRSM Network](https://prsm.network) and get your API key.

### 2. Install SDK
Choose your preferred language and install the SDK.

### 3. Initialize Client
Create a client instance with your API key.

### 4. Make Requests
Start making API calls to PRSM services.

## Configuration

### Environment Variables
All SDKs support configuration via environment variables:

```bash
export PRSM_API_KEY="your-api-key"
export PRSM_BASE_URL="https://api.prsm.ai"
export PRSM_TIMEOUT="30"
export PRSM_MAX_RETRIES="3"
```

### Configuration Objects

#### Python
```python
from prsm_sdk import PRSMClient, Config

config = Config(
    api_key="your-api-key",
    base_url="https://api.prsm.ai",
    timeout=30.0,
    max_retries=3
)

client = PRSMClient(config=config)
```

#### JavaScript
```typescript
const client = new PRSMClient({
  apiKey: 'your-api-key',
  baseURL: 'https://api.prsm.ai',
  timeout: 30000,
  maxRetries: 3
});
```

#### Go
```go
config := prsm.Config{
    APIKey:     "your-api-key",
    BaseURL:    "https://api.prsm.ai",
    Timeout:    30 * time.Second,
    MaxRetries: 3,
}

client := prsm.NewClientWithConfig(config)
```

## Examples

### Basic Model Inference
```python
# Python
result = await client.models.infer(
    model="gpt-4",
    prompt="Explain quantum computing",
    max_tokens=500
)
```

```javascript
// JavaScript
const result = await client.models.infer({
  model: 'gpt-4',
  prompt: 'Explain quantum computing',
  maxTokens: 500
});
```

```go
// Go
result, err := client.Models.Infer(ctx, prsm.InferRequest{
    Model:     "gpt-4",
    Prompt:    "Explain quantum computing",
    MaxTokens: 500,
})
```

### Streaming Responses
```python
# Python
async for chunk in client.models.stream(
    model="gpt-4",
    prompt="Write a story"
):
    print(chunk.content, end="")
```

```javascript
// JavaScript
const stream = client.models.stream({
  model: 'gpt-4',
  prompt: 'Write a story'
});

for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

```go
// Go
stream, err := client.Models.Stream(ctx, prsm.StreamRequest{
    Model:  "gpt-4",
    Prompt: "Write a story",
})

for chunk := range stream {
    fmt.Print(chunk.Content)
}
```

### Error Handling
```python
# Python
try:
    result = await client.models.infer("gpt-4", "Hello")
except prsm_sdk.RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after_seconds}s")
except prsm_sdk.BudgetExceededError as e:
    print(f"Budget exceeded. Remaining: ${e.remaining_budget}")
```

```javascript
// JavaScript
try {
  const result = await client.models.infer('gpt-4', 'Hello');
} catch (error) {
  if (error instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${error.retryAfterSeconds}s`);
  } else if (error instanceof BudgetExceededError) {
    console.log(`Budget exceeded. Remaining: $${error.remainingBudget}`);
  }
}
```

```go
// Go
result, err := client.Models.Infer(ctx, "gpt-4", "Hello")
if err != nil {
    switch e := err.(type) {
    case *prsm.RateLimitError:
        fmt.Printf("Rate limited. Retry after %ds\n", e.RetryAfterSeconds)
    case *prsm.BudgetExceededError:
        fmt.Printf("Budget exceeded. Remaining: $%.2f\n", e.RemainingBudget)
    default:
        fmt.Printf("Error: %v\n", err)
    }
}
```

## Community SDKs

Community-maintained SDKs are available for additional languages:
- **Rust**: `cargo add prsm-rs`
- **Ruby**: `gem install prsm`
- **PHP**: `composer require prsm/sdk`
- **Java**: Available on Maven Central

## Support

- **Documentation**: [docs.prsm.network](https://docs.prsm.network)
- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Join our developer community
- **Email**: sdk-support@prsm.ai

## Contributing

We welcome contributions to our SDKs! See our [contributing guide](https://github.com/prsm-ai/prsm-python/blob/main/CONTRIBUTING.md) for details.

## License

All official PRSM SDKs are released under the MIT License.