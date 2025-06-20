# PRSM Go SDK

Official Go client for the Protocol for Recursive Scientific Modeling (PRSM).

[![Go Reference](https://pkg.go.dev/badge/github.com/PRSM-AI/prsm-go-sdk.svg)](https://pkg.go.dev/github.com/PRSM-AI/prsm-go-sdk)
[![Go Version](https://img.shields.io/github/go-mod/go-version/PRSM-AI/prsm-go-sdk)](https://golang.org/doc/devel/release.html)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

## Installation

```bash
go get github.com/PRSM-AI/prsm-go-sdk
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/PRSM-AI/prsm-go-sdk/client"
    "github.com/PRSM-AI/prsm-go-sdk/types"
)

func main() {
    // Initialize client
    c := client.New("your_api_key_here")
    
    // Simple AI query
    response, err := c.Query(context.Background(), &types.QueryRequest{
        Prompt: "Explain quantum computing in simple terms",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Response:", response.Content)
    fmt.Printf("Cost: %.4f FTNS\n", response.FTNSCost)
    
    // Check token balance
    balance, err := c.FTNS.GetBalance(context.Background())
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("FTNS Balance: %.2f\n", balance.AvailableBalance)
    
    // Search for models
    models, err := c.Marketplace.SearchModels(context.Background(), &types.MarketplaceQuery{
        Query: "gpt",
        Limit: 10,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Found %d models\n", len(models))
}
```

## Features

- 🤖 **AI Query Interface** - Simple access to PRSM's distributed AI network
- 💰 **FTNS Token Management** - Built-in token balance and cost tracking
- 🔒 **Authentication** - Secure API key and JWT token handling
- 📊 **Model Marketplace** - Browse and use community-contributed models
- 🛡️ **Safety Integration** - Built-in safety monitoring and circuit breakers
- 📈 **Performance Monitoring** - Request tracking and optimization
- 🌐 **P2P Network Access** - Direct access to distributed computing resources
- 🔧 **Tool Integration** - MCP tool protocol support for enhanced capabilities
- ⚡ **High Performance** - Optimized for concurrent operations and low latency

## Advanced Usage

### Context and Cancellation

```go
func contextExample() {
    c := client.New("your_api_key")
    
    // Create context with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    response, err := c.Query(ctx, &types.QueryRequest{
        Prompt:      "Complex scientific computation",
        MaxTokens:   2000,
        Temperature: 0.3,
    })
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            fmt.Println("Query timed out")
            return
        }
        log.Fatal(err)
    }
    
    fmt.Println(response.Content)
}
```

### Cost Estimation

```go
func costExample() {
    c := client.New("your_api_key")
    ctx := context.Background()
    
    // Estimate cost before running
    cost, err := c.EstimateCost(ctx, "Complex scientific query about protein folding", nil)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Estimated cost: %.4f FTNS\n", cost)
    
    // Check if we have enough balance
    balance, err := c.FTNS.GetBalance(ctx)
    if err != nil {
        log.Fatal(err)
    }
    
    if balance.AvailableBalance >= cost {
        response, err := c.Query(ctx, &types.QueryRequest{
            Prompt: "Complex scientific query about protein folding",
        })
        if err != nil {
            log.Fatal(err)
        }
        fmt.Println(response.Content)
    } else {
        fmt.Println("Insufficient FTNS balance")
    }
}
```

### Model Marketplace

```go
func marketplaceExample() {
    c := client.New("your_api_key")
    ctx := context.Background()
    
    // Search for specific models
    minPerf := 0.8
    maxCost := 0.001
    models, err := c.Marketplace.SearchModels(ctx, &types.MarketplaceQuery{
        Query:          "scientific research",
        MinPerformance: &minPerf,
        MaxCost:        &maxCost,
        Capabilities:   []string{"text_generation", "reasoning"},
        Limit:          5,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Use a specific model
    if len(models) > 0 {
        model := models[0]
        response, err := c.Query(ctx, &types.QueryRequest{
            Prompt:  "Explain CRISPR gene editing",
            ModelID: &model.ID,
        })
        if err != nil {
            log.Fatal(err)
        }
        
        fmt.Printf("Response from %s: %s\n", model.Name, response.Content)
    }
}
```

### Tool Execution (MCP)

```go
func toolsExample() {
    c := client.New("your_api_key")
    ctx := context.Background()
    
    // List available tools
    tools, err := c.Tools.ListAvailable(ctx)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Available tools: ")
    for i, tool := range tools {
        if i > 0 {
            fmt.Print(", ")
        }
        fmt.Print(tool.Name)
    }
    fmt.Println()
    
    // Execute a tool
    for _, tool := range tools {
        if tool.Name == "web_search" {
            result, err := c.Tools.Execute(ctx, &types.ToolExecutionRequest{
                ToolName: "web_search",
                Parameters: map[string]interface{}{
                    "query": "latest AI research papers",
                },
                SafetyLevel: types.SafetyLevelModerate,
            })
            if err != nil {
                log.Fatal(err)
            }
            
            fmt.Printf("Search results: %v\n", result.Result)
            break
        }
    }
}
```

### Concurrent Operations

```go
import (
    "sync"
    "golang.org/x/sync/errgroup"
)

func concurrentExample() {
    c := client.New("your_api_key")
    ctx := context.Background()
    
    queries := []string{
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does AI work?",
    }
    
    // Use errgroup for concurrent execution
    g, ctx := errgroup.WithContext(ctx)
    responses := make([]*types.PRSMResponse, len(queries))
    
    for i, query := range queries {
        i, query := i, query // Capture loop variables
        g.Go(func() error {
            response, err := c.Query(ctx, &types.QueryRequest{
                Prompt: query,
            })
            if err != nil {
                return err
            }
            responses[i] = response
            return nil
        })
    }
    
    if err := g.Wait(); err != nil {
        log.Fatal(err)
    }
    
    for i, response := range responses {
        fmt.Printf("Query %d: %s\n", i+1, response.Content[:100]+"...")
    }
}
```

### Error Handling

```go
import (
    "errors"
    "github.com/PRSM-AI/prsm-go-sdk/types"
)

func errorHandlingExample() {
    c := client.New("invalid_api_key")
    ctx := context.Background()
    
    _, err := c.Query(ctx, &types.QueryRequest{
        Prompt: "Hello world",
    })
    
    if err != nil {
        var authErr *types.AuthenticationError
        var fundsErr *types.InsufficientFundsError
        var safetyErr *types.SafetyViolationError
        
        switch {
        case errors.As(err, &authErr):
            fmt.Println("Authentication failed:", authErr.Message)
        case errors.As(err, &fundsErr):
            fmt.Printf("Insufficient funds: required %.2f, available %.2f\n", 
                      fundsErr.Required, fundsErr.Available)
        case errors.As(err, &safetyErr):
            fmt.Printf("Safety violation [%s]: %s\n", 
                      safetyErr.SafetyLevel, safetyErr.Message)
        default:
            fmt.Println("Unexpected error:", err)
        }
    }
}
```

## Configuration

### Environment Variables

```bash
export PRSM_API_KEY="your_api_key_here"
export PRSM_BASE_URL="https://api.prsm.ai/v1"  # Optional
```

### Custom Configuration

```go
import (
    "time"
    "github.com/PRSM-AI/prsm-go-sdk/client"
    "golang.org/x/time/rate"
)

func customConfigExample() {
    config := &client.Config{
        APIKey:       "your_api_key",
        BaseURL:      "https://custom-api.prsm.ai/v1",
        WebSocketURL: "wss://custom-ws.prsm.ai/v1",
        Timeout:      2 * time.Minute,
        MaxRetries:   5,
        RateLimit:    rate.Limit(20), // 20 requests per second
    }
    
    c := client.NewWithConfig(config)
    
    // Use client...
}
```

## API Reference

### Client

The main client struct for interacting with PRSM.

```go
type Client struct {
    Auth        *auth.Manager
    FTNS        *ftns.Manager
    Marketplace *marketplace.Manager
    Tools       *tools.Executor
}

// Main methods
func (c *Client) Query(ctx context.Context, req *types.QueryRequest) (*types.PRSMResponse, error)
func (c *Client) EstimateCost(ctx context.Context, prompt string, modelID *string) (float64, error)
func (c *Client) GetSafetyStatus(ctx context.Context) (*types.SafetyStatus, error)
func (c *Client) ListAvailableModels(ctx context.Context) ([]*types.ModelInfo, error)
func (c *Client) HealthCheck(ctx context.Context) (map[string]interface{}, error)
```

### Types

All data structures are defined in the `types` package:

```go
type QueryRequest struct {
    Prompt       string                 `json:"prompt"`
    ModelID      *string                `json:"model_id,omitempty"`
    MaxTokens    int                    `json:"max_tokens"`
    Temperature  float64                `json:"temperature"`
    SystemPrompt *string                `json:"system_prompt,omitempty"`
    Context      map[string]interface{} `json:"context"`
    Tools        []string               `json:"tools,omitempty"`
    SafetyLevel  SafetyLevel            `json:"safety_level"`
}

type PRSMResponse struct {
    Content        string                 `json:"content"`
    ModelID        string                 `json:"model_id"`
    Provider       ModelProvider          `json:"provider"`
    ExecutionTime  float64                `json:"execution_time"`
    TokenUsage     map[string]int         `json:"token_usage"`
    FTNSCost       float64                `json:"ftns_cost"`
    ReasoningTrace []string               `json:"reasoning_trace,omitempty"`
    SafetyStatus   SafetyLevel            `json:"safety_status"`
    Metadata       map[string]interface{} `json:"metadata"`
    RequestID      string                 `json:"request_id"`
    Timestamp      time.Time              `json:"timestamp"`
}
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/PRSM-AI/PRSM.git
cd PRSM/sdks/go

# Download dependencies
go mod download

# Run tests
go test ./...

# Run linting (requires golangci-lint)
golangci-lint run

# Build examples
go build -o examples/basic examples/basic/main.go
```

### Testing

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests with race detection
go test -race ./...

# Run specific package tests
go test ./client

# Verbose output
go test -v ./...
```

### Building

```bash
# Build for current platform
go build ./...

# Build for specific platforms
GOOS=linux GOARCH=amd64 go build ./...
GOOS=windows GOARCH=amd64 go build ./...
GOOS=darwin GOARCH=arm64 go build ./...
```

## Examples

See the [examples](examples/) directory for more comprehensive examples:

- [Basic Usage](examples/basic/main.go)
- [Concurrent Operations](examples/concurrent/main.go)
- [Marketplace Integration](examples/marketplace/main.go)
- [Tool Execution](examples/tools/main.go)
- [Cost Management](examples/cost/main.go)
- [Streaming (WebSocket)](examples/streaming/main.go)

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Support

- 📚 **Documentation**: [docs.prsm.ai/go-sdk](https://docs.prsm.ai/go-sdk)
- 🐛 **Issues**: [GitHub Issues](https://github.com/PRSM-AI/PRSM/issues)
- 💬 **Community**: [Discord](https://discord.gg/prsm)
- 📧 **Email**: [dev@prsm.ai](mailto:dev@prsm.ai)