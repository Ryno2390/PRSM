# PRSM Go SDK

Official Go SDK for PRSM (Protocol for Recursive Scientific Modeling)

## 🚀 Quick Start

```bash
go get github.com/PRSM-AI/prsm-go-sdk
```

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    
    "github.com/PRSM-AI/prsm-go-sdk/client"
    "github.com/PRSM-AI/prsm-go-sdk/types"
)

func main() {
    // Get API key from environment variable
    apiKey := os.Getenv("PRSM_API_KEY")
    if apiKey == "" {
        log.Fatal("PRSM_API_KEY environment variable is required")
    }

    // Create PRSM client
    prsmClient := client.New(apiKey)

    // Prepare query request
    queryReq := &types.QueryRequest{
        Prompt:      "What are the key principles of machine learning?",
        MaxTokens:   500,
        Temperature: 0.7,
        SafetyLevel: types.SafetyLevelModerate,
    }

    fmt.Println("Executing query...")

    // Execute query
    response, err := prsmClient.Query(context.Background(), queryReq)
    if err != nil {
        log.Fatalf("Query failed: %v", err)
    }

    // Display results
    fmt.Printf("Response: %s\n", response.Content)
    fmt.Printf("Model ID: %s\n", response.ModelID)
    fmt.Printf("Provider: %s\n", response.Provider)
    fmt.Printf("Execution Time: %.2f seconds\n", response.ExecutionTime)
    fmt.Printf("FTNS Cost: %.4f\n", response.FTNSCost)
    fmt.Printf("Safety Status: %s\n", response.SafetyStatus)
}
```

## 📦 Installation

```bash
go get github.com/PRSM-AI/prsm-go-sdk
```

## 🔧 Configuration

### Basic Configuration

```go
// Simple client with API key
client := client.New("your-api-key")
```

### Advanced Configuration

```go
config := &client.Config{
    APIKey:       "your-api-key",
    BaseURL:      "https://api.prsm.ai/v1",
    WebSocketURL: "wss://ws.prsm.ai/v1",
    Timeout:      60 * time.Second,
    MaxRetries:   3,
    RateLimit:    10, // requests per second
}

prsmClient := client.NewWithConfig(config)
```

## 🧠 Core Features

### 1. AI Query Execution

Execute AI queries with comprehensive configuration:

```go
// Basic query
request := &types.QueryRequest{
    Prompt:      "Explain machine learning fundamentals",
    MaxTokens:   1000,
    Temperature: 0.7,
    SafetyLevel: types.SafetyLevelModerate,
}

response, err := client.Query(ctx, request)
if err != nil {
    return err
}

fmt.Printf("Response: %s\n", response.Content)
fmt.Printf("Cost: %.4f FTNS\n", response.FTNSCost)

// Advanced query with system prompt and tools
systemPrompt := "You are an expert AI researcher"
advancedRequest := &types.QueryRequest{
    Prompt:       "Compare transformer architectures for NLP tasks",
    ModelID:      &modelID,
    MaxTokens:    800,
    Temperature:  0.3,
    SystemPrompt: &systemPrompt,
    Context: map[string]interface{}{
        "domain": "natural_language_processing",
        "level":  "advanced",
    },
    Tools:       []string{"web_search", "arxiv_search"},
    SafetyLevel: types.SafetyLevelHigh,
}

response, err := client.Query(ctx, advancedRequest)
```

### 2. Model Discovery & Management

Discover and interact with available AI models:

```go
// List all available models
models, err := client.ListAvailableModels(ctx)
if err != nil {
    return err
}

for _, model := range models {
    fmt.Printf("Model: %s (%s)\n", model.Name, model.ID)
    fmt.Printf("  Provider: %s\n", model.Provider)
    fmt.Printf("  Cost per token: %.6f FTNS\n", model.CostPerToken)
    fmt.Printf("  Performance: %.2f/5.0\n", model.PerformanceRating)
    fmt.Printf("  Available: %t\n", model.IsAvailable)
}

// Search marketplace for specific models
searchQuery := &types.MarketplaceQuery{
    Query:          "language model",
    MaxCost:        &maxCost,
    MinPerformance: &minPerformance,
    Capabilities:   []string{"text-generation", "reasoning"},
    Limit:          10,
}

searchResults, err := client.Marketplace.SearchModels(ctx, searchQuery)
```

### 3. FTNS Token Management

Manage FTNS tokens for the PRSM ecosystem:

```go
// Check current balance
balance, err := client.FTNS.GetBalance(ctx)
if err != nil {
    return err
}

fmt.Printf("Total Balance: %.4f FTNS\n", balance.TotalBalance)
fmt.Printf("Available: %.4f FTNS\n", balance.AvailableBalance)
fmt.Printf("Reserved: %.4f FTNS\n", balance.ReservedBalance)

// Get transaction history
transactions, err := client.FTNS.GetTransactionHistory(ctx, 10)
if err != nil {
    return err
}

for _, tx := range transactions {
    fmt.Printf("%s: %+.4f FTNS (%s)\n", 
        tx.Timestamp.Format("Jan 02 15:04"), 
        tx.Amount, 
        tx.Type)
}

// Estimate cost before executing query
cost, err := client.EstimateCost(ctx, "Complex research query", nil)
if err != nil {
    return err
}

fmt.Printf("Estimated cost: %.4f FTNS\n", cost)
```

### 4. Safety & Monitoring

Monitor system safety and health:

```go
// Check safety status
safetyStatus, err := client.GetSafetyStatus(ctx)
if err != nil {
    return err
}

fmt.Printf("Safety Status: %s\n", safetyStatus.OverallStatus)
fmt.Printf("Active Monitors: %d\n", safetyStatus.ActiveMonitors)
fmt.Printf("Network Health: %.2f%%\n", safetyStatus.NetworkHealth*100)

// Health check
health, err := client.HealthCheck(ctx)
if err != nil {
    return err
}

fmt.Printf("API Status: %v\n", health["status"])
fmt.Printf("Response Time: %v ms\n", health["response_time"])
```

### 5. Tool Execution (MCP Integration)

Execute external tools for enhanced functionality:

```go
// Execute web search tool
toolReq := &types.ToolExecutionRequest{
    ToolName: "web_search",
    Parameters: map[string]interface{}{
        "query":       "latest AI research 2024",
        "max_results": 5,
    },
    SafetyLevel: types.SafetyLevelModerate,
}

toolResponse, err := client.Tools.Execute(ctx, toolReq)
if err != nil {
    return err
}

if toolResponse.Success {
    fmt.Printf("Tool execution successful!\n")
    fmt.Printf("Result: %v\n", toolResponse.Result)
    fmt.Printf("Cost: %.4f FTNS\n", toolResponse.FTNSCost)
}
```

## 🔄 Error Handling

The SDK provides comprehensive error handling with specific error types:

```go
response, err := client.Query(ctx, request)
if err != nil {
    switch e := err.(type) {
    case *types.AuthenticationError:
        fmt.Printf("Authentication failed: %s\n", e.Message)
        // Handle auth error (check API key, refresh token, etc.)
    case *types.ValidationError:
        fmt.Printf("Validation error: %s - %s\n", e.Field, e.ValidationMessage)
        // Handle validation error (fix request parameters)
    case *types.RateLimitError:
        if e.RetryAfter != nil {
            fmt.Printf("Rate limited. Retry after: %d seconds\n", *e.RetryAfter)
            time.Sleep(time.Duration(*e.RetryAfter) * time.Second)
        }
        // Retry request with backoff
    case *types.SafetyViolationError:
        fmt.Printf("Safety violation [%s]: %s\n", e.SafetyLevel, e.Message)
        // Handle safety violation (modify content, use different safety level)
    case *types.InsufficientFundsError:
        fmt.Printf("Insufficient FTNS balance: %.2f required, %.2f available\n", 
            e.Required, e.Available)
        // Handle insufficient funds (top up balance, reduce query complexity)
    case *types.ModelNotFoundError:
        fmt.Printf("Model not found: %s\n", e.ModelID)
        // Handle model not found (choose different model)
    case *types.NetworkError:
        fmt.Printf("Network error: %s\n", e.Message)
        // Handle network error (retry with backoff, check connectivity)
    case *types.ToolExecutionError:
        fmt.Printf("Tool execution failed: %s - %s\n", e.ToolName, e.ErrorMessage)
        // Handle tool execution error
    default:
        fmt.Printf("Unexpected error: %s\n", err)
    }
}
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run race condition detection
go test -race ./...

# Use the provided test runner script
./test_runner.sh
```

### Example Tests

The SDK includes comprehensive tests for all components:

```go
func TestClientQuery(t *testing.T) {
    client := client.New("test-api-key")
    
    request := &types.QueryRequest{
        Prompt:      "Test prompt",
        MaxTokens:   100,
        Temperature: 0.7,
        SafetyLevel: types.SafetyLevelModerate,
    }
    
    // This would use mock server in real tests
    response, err := client.Query(context.Background(), request)
    
    assert.NoError(t, err)
    assert.NotEmpty(t, response.Content)
    assert.Greater(t, response.FTNSCost, 0.0)
}

func TestErrorHandling(t *testing.T) {
    client := client.New("invalid-key")
    
    request := &types.QueryRequest{
        Prompt: "Test prompt",
    }
    
    _, err := client.Query(context.Background(), request)
    
    assert.Error(t, err)
    assert.IsType(t, &types.AuthenticationError{}, err)
}
```

## 📊 Examples and Documentation

Comprehensive examples are available in the `examples/` directory:

- **basic_query.go** - Simple AI query execution
- **cost_estimation.go** - Cost estimation and balance checking
- **model_marketplace.go** - Model discovery and marketplace features
- **ftns_management.go** - Token management and transactions
- **advanced_features.go** - Safety monitoring, tools, and error handling

See the [Examples README](examples/README.md) for detailed usage instructions.

## 🔒 Security Best Practices

1. **Environment Variables**: Store API keys in environment variables
   ```bash
   export PRSM_API_KEY="your-api-key"
   ```

2. **Input Validation**: Always validate user inputs before sending to PRSM
3. **Safety Levels**: Use appropriate safety levels for your use case
4. **Error Handling**: Implement proper error handling for all operations
5. **Rate Limiting**: Respect API rate limits to avoid throttling

## 📈 Advanced Usage

### Context and Timeouts

```go
// Request with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

response, err := client.Query(ctx, request)

// Request with cancellation
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

// Cancel from another goroutine if needed
go func() {
    time.Sleep(10 * time.Second)
    cancel()
}()

response, err := client.Query(ctx, request)
```

### Custom Configuration

```go
// Custom rate limiting and timeouts
config := &client.Config{
    APIKey:     "your-api-key",
    BaseURL:    "https://api.prsm.ai/v1",
    Timeout:    120 * time.Second,
    MaxRetries: 5,
    RateLimit:  5, // 5 requests per second
}

prsmClient := client.NewWithConfig(config)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/PRSM-AI/PRSM.git
cd PRSM/sdks/go

# Install dependencies
go mod tidy

# Run tests
go test ./...

# Run linting
golangci-lint run
```

## 📄 License

This SDK is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

## 🆘 Support

- 📧 Email: sdk-support@prsm.ai
- 💬 Discord: [PRSM Community](https://discord.gg/prsm)
- 📖 Documentation: [docs.prsm.ai](https://docs.prsm.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/PRSM-AI/PRSM/issues)