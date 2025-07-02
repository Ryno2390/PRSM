# PRSM Go SDK Examples

This directory contains comprehensive examples demonstrating how to use the PRSM Go SDK for AI queries, token management, and marketplace interactions.

## Prerequisites

1. **Go Installation**: Ensure you have Go 1.19 or later installed
2. **API Key**: Set your PRSM API key as an environment variable:
   ```bash
   export PRSM_API_KEY="your_api_key_here"
   ```

## Running Examples

Each example is a standalone Go program that you can run directly:

```bash
# Basic AI query example
go run basic_query.go

# Cost estimation example
go run cost_estimation.go

# Model marketplace exploration
go run model_marketplace.go

# FTNS token management
go run ftns_management.go

# Advanced features demonstration
go run advanced_features.go
```

## Example Descriptions

### 1. Basic Query (`basic_query.go`)

Demonstrates the fundamental PRSM AI query functionality:

- Creating a PRSM client
- Configuring query parameters (prompt, tokens, temperature, safety level)
- Executing queries and handling responses
- Displaying results including reasoning traces

**Key Features Shown:**
- Simple client initialization
- Basic query execution
- Response handling and data extraction

### 2. Cost Estimation (`cost_estimation.go`)

Shows how to estimate FTNS costs before executing queries:

- Estimating costs for different prompt complexities
- Checking account balance before execution
- Model-specific cost estimation
- Cost optimization strategies

**Key Features Shown:**
- Pre-query cost estimation
- Balance checking and validation
- Model-specific pricing
- Cost-aware query planning

### 3. Model Marketplace (`model_marketplace.go`)

Explores the PRSM model marketplace:

- Listing available models
- Searching for models by criteria
- Finding optimal models for specific tasks
- Comparing model performance and costs
- Analyzing model metrics

**Key Features Shown:**
- Model discovery and search
- Performance comparisons
- Cost-benefit analysis
- Marketplace navigation

### 4. FTNS Management (`ftns_management.go`)

Comprehensive token management operations:

- Checking FTNS balances
- Viewing transaction history
- Token transfers (example structure)
- Staking opportunities
- Governance voting power
- Network revenue tracking

**Key Features Shown:**
- Balance monitoring
- Transaction tracking
- Token economics
- Governance participation

### 5. Advanced Features (`advanced_features.go`)

Advanced PRSM capabilities and error handling:

- Safety monitoring and status
- Complex queries with system prompts
- Tool execution (MCP integration)
- Health checks and diagnostics
- Model performance analysis
- Comprehensive error handling

**Key Features Shown:**
- Safety system integration
- Advanced query configuration
- Tool execution framework
- System monitoring
- Error handling patterns

## Common Patterns

### Client Configuration

```go
// Basic client
client := client.New("your-api-key")

// Custom configuration
config := client.DefaultConfig()
config.APIKey = "your-api-key"
config.Timeout = 120 * time.Second
config.MaxRetries = 5
client := client.NewWithConfig(config)
```

### Error Handling

```go
response, err := client.Query(ctx, request)
if err != nil {
    switch e := err.(type) {
    case *types.AuthenticationError:
        // Handle auth errors
    case *types.InsufficientFundsError:
        // Handle balance issues
    case *types.RateLimitError:
        // Handle rate limiting
    default:
        // Handle other errors
    }
    return
}
```

### Context Management

```go
// Basic context
ctx := context.Background()

// Context with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// Context with cancellation
ctx, cancel := context.WithCancel(context.Background())
defer cancel()
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific test package
go test ./types
go test ./client
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PRSM_API_KEY` | Your PRSM API key | Yes |
| `PRSM_BASE_URL` | Custom API base URL | No |
| `PRSM_TIMEOUT` | Request timeout in seconds | No |

## Error Handling Best Practices

1. **Always check errors**: Every SDK method returns an error that should be checked
2. **Type-specific handling**: Use type assertions to handle specific error types
3. **Retry logic**: Implement appropriate retry logic for transient errors
4. **Context timeouts**: Use context timeouts for long-running operations
5. **Resource cleanup**: Always close resources and cancel contexts when done

## Performance Tips

1. **Reuse clients**: Create one client instance and reuse it across requests
2. **Context pooling**: Reuse context objects where appropriate
3. **Rate limiting**: Respect rate limits to avoid throttling
4. **Batch operations**: Group related operations when possible
5. **Cost optimization**: Use cost estimation to optimize query parameters

## Security Considerations

1. **API Key Protection**: Never hardcode API keys in source code
2. **Environment Variables**: Use environment variables for secrets
3. **HTTPS Only**: All communications use HTTPS by default
4. **Safety Levels**: Use appropriate safety levels for your use case
5. **Input Validation**: Validate all user inputs before sending to PRSM

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Check API key format and validity
2. **Network Timeouts**: Increase timeout values for complex queries
3. **Rate Limiting**: Implement exponential backoff for rate limit errors
4. **Insufficient Funds**: Monitor FTNS balance and top up as needed
5. **Model Availability**: Check model status before making requests

### Debug Mode

Enable verbose logging for debugging:

```go
import "log"

// Enable debug logging
log.SetFlags(log.LstdFlags | log.Lshortfile)
```

### Support

For additional support:

1. Check the [PRSM Documentation](https://docs.prsm.ai)
2. Review the [API Reference](https://api.prsm.ai/docs)
3. Join the [PRSM Community](https://community.prsm.ai)
4. Submit issues on [GitHub](https://github.com/PRSM-AI/prsm-go-sdk)

## License

This SDK is licensed under the MIT License. See the LICENSE file for details.