# PRSM Go SDK

Official Go SDK for PRSM (Protocol for Recursive Scientific Modeling)

## 🚀 Quick Start

```bash
go get github.com/PRSM-AI/PRSM/sdks/go
```

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/PRSM-AI/PRSM/sdks/go/prsm"
)

func main() {
    // Initialize client
    client, err := prsm.NewClient(&prsm.Config{
        BaseURL: "https://api.prsm.ai",
        APIKey:  "your-api-key",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Submit research query with SEAL enhancement
    query := &prsm.NWTNQuery{
        Query:  "Analyze the effectiveness of renewable energy policies",
        Domain: "environmental_policy",
        SEALEnhancement: &prsm.SEALConfig{
            Enabled:              true,
            AutonomousImprovement: true,
            TargetLearningGain:   0.25,
        },
    }

    session, err := client.NWTN.SubmitQuery(context.Background(), query)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Research session started: %s\n", session.ID)

    // Monitor progress with real-time updates
    progressCh, err := client.WebSocket.SubscribeToSession(session.ID)
    if err != nil {
        log.Fatal(err)
    }

    for progress := range progressCh {
        fmt.Printf("Progress: %.1f%% - %s\n", 
            progress.Progress, progress.CurrentPhase)
        
        if progress.Status == prsm.SessionStatusCompleted {
            fmt.Printf("Research completed! Results: %s\n", 
                progress.Results.Summary)
            break
        }
    }
}
```

## 📦 Installation

```bash
go get github.com/PRSM-AI/PRSM/sdks/go
```

## 🔧 Configuration

### Basic Configuration

```go
config := &prsm.Config{
    BaseURL: "https://api.prsm.ai",
    APIKey:  "your-api-key",
    Timeout: 30 * time.Second,
}

client, err := prsm.NewClient(config)
```

### Advanced Configuration

```go
config := &prsm.Config{
    BaseURL: "https://api.prsm.ai",
    APIKey:  "your-api-key",
    Timeout: 60 * time.Second,
    RetryConfig: &prsm.RetryConfig{
        MaxRetries:    3,
        RetryDelay:    time.Second,
        ExponentialBackoff: true,
    },
    WebSocket: &prsm.WebSocketConfig{
        ReconnectAttempts: 5,
        ReconnectDelay:    2 * time.Second,
    },
    FTNS: &prsm.FTNSConfig{
        WalletAddress: "0x...",
        PrivateKey:    "your-private-key",
        Network:       "polygon",
    },
}

client, err := prsm.NewClient(config)
```

## 🧠 Core Features

### 1. NWTN (Neural Web of Thought Networks)

Submit complex research queries and get comprehensive analysis:

```go
// Basic research query
query := &prsm.NWTNQuery{
    Query:    "Impact of AI on healthcare efficiency",
    Domain:   "healthcare_technology",
    Priority: prsm.PriorityHigh,
}

session, err := client.NWTN.SubmitQuery(ctx, query)
if err != nil {
    return err
}

// Advanced query with SEAL enhancement
advancedQuery := &prsm.NWTNQuery{
    Query:  "Optimize renewable energy grid integration",
    Domain: "energy_systems",
    SEALEnhancement: &prsm.SEALConfig{
        Enabled:                true,
        AutonomousImprovement:  true,
        TargetLearningGain:     0.30,
        RecursiveDepth:         5,
        QualityThreshold:       0.85,
    },
    Constraints: &prsm.QueryConstraints{
        MaxCost:      100.0,
        MaxDuration:  time.Hour * 2,
        RequiredQuality: 0.9,
    },
}

session, err := client.NWTN.SubmitQuery(ctx, advancedQuery)
```

### 2. Model Management

Discover and interact with AI models:

```go
// Browse available models
models, err := client.Models.Browse(ctx, &prsm.ModelFilter{
    Category:    "scientific",
    Provider:    prsm.ProviderAnthropic,
    MinQuality:  0.8,
    MaxCost:     0.01,
})

// Execute with specific model
request := &prsm.ModelRequest{
    ModelID: "claude-3-sonnet",
    Prompt:  "Explain quantum entanglement",
    SystemPrompt: "You are a quantum physics expert",
    MaxTokens: 1000,
    Temperature: 0.7,
}

response, err := client.Models.Execute(ctx, request)
if err != nil {
    return err
}

fmt.Printf("Response: %s\n", response.Content)
fmt.Printf("Cost: $%.4f\n", response.Cost)
```

### 3. FTNS Token Management

Manage FTNS tokens for the PRSM ecosystem:

```go
// Check token balance
balance, err := client.FTNS.GetBalance(ctx)
if err != nil {
    return err
}

fmt.Printf("FTNS Balance: %.2f\n", balance.Available)

// Transfer tokens
transfer := &prsm.FTNSTransfer{
    ToAddress: "0x742d35Cc6634C0532925a3b8D138C6B9be7b6s5a3",
    Amount:    100.0,
    Note:      "Research collaboration payment",
}

txHash, err := client.FTNS.Transfer(ctx, transfer)
if err != nil {
    return err
}

fmt.Printf("Transfer completed: %s\n", txHash)
```

### 4. Real-time WebSocket

Monitor sessions and receive live updates:

```go
// Connect to WebSocket
err := client.WebSocket.Connect(ctx)
if err != nil {
    return err
}

// Subscribe to session updates
progressCh, err := client.WebSocket.SubscribeToSession(sessionID)
if err != nil {
    return err
}

// Handle real-time updates
go func() {
    for progress := range progressCh {
        switch progress.Status {
        case prsm.SessionStatusRunning:
            fmt.Printf("Progress: %.1f%% - %s\n", 
                progress.Progress, progress.CurrentPhase)
        case prsm.SessionStatusCompleted:
            fmt.Printf("Session completed!\n")
            handleResults(progress.Results)
        case prsm.SessionStatusFailed:
            fmt.Printf("Session failed: %s\n", progress.Error)
        }
    }
}()
```

## 🔄 Error Handling

The SDK provides comprehensive error handling with specific error types:

```go
_, err := client.NWTN.SubmitQuery(ctx, query)
if err != nil {
    switch e := err.(type) {
    case *prsm.AuthenticationError:
        fmt.Printf("Authentication failed: %s\n", e.Message)
        // Handle auth error (refresh token, re-login, etc.)
    case *prsm.ValidationError:
        fmt.Printf("Validation error: %s\n", e.Message)
        for field, message := range e.FieldErrors {
            fmt.Printf("  %s: %s\n", field, message)
        }
    case *prsm.RateLimitError:
        fmt.Printf("Rate limited. Retry after: %s\n", e.RetryAfter)
        time.Sleep(e.RetryAfter)
        // Retry request
    case *prsm.SafetyViolationError:
        fmt.Printf("Safety violation: %s\n", e.Violation)
        // Handle safety violation
    case *prsm.InsufficientFundsError:
        fmt.Printf("Insufficient FTNS balance: %.2f required, %.2f available\n", 
            e.Required, e.Available)
        // Handle insufficient funds
    default:
        fmt.Printf("Unexpected error: %s\n", err)
    }
}
```

## 🧪 Testing

### Unit Tests

```go
func TestNWTNQuerySubmission(t *testing.T) {
    client := prsm.NewTestClient()
    
    query := &prsm.NWTNQuery{
        Query:  "Test research query",
        Domain: "test_domain",
    }
    
    session, err := client.NWTN.SubmitQuery(context.Background(), query)
    
    assert.NoError(t, err)
    assert.NotEmpty(t, session.ID)
    assert.Equal(t, prsm.SessionStatusPending, session.Status)
}
```

### Integration Tests

```go
func TestFullResearchWorkflow(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    client := setupIntegrationClient(t)
    
    // Submit query
    query := &prsm.NWTNQuery{
        Query:  "Integration test query",
        Domain: "test",
    }
    
    session, err := client.NWTN.SubmitQuery(context.Background(), query)
    require.NoError(t, err)
    
    // Monitor progress
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()
    
    progressCh, err := client.WebSocket.SubscribeToSession(session.ID)
    require.NoError(t, err)
    
    for {
        select {
        case progress := <-progressCh:
            if progress.Status == prsm.SessionStatusCompleted {
                assert.NotEmpty(t, progress.Results.Summary)
                return
            }
        case <-ctx.Done():
            t.Fatal("Test timeout")
        }
    }
}
```

## 📊 Performance and Monitoring

### Metrics Collection

```go
// Enable metrics collection
client.EnableMetrics()

// Get client metrics
metrics := client.GetMetrics()

fmt.Printf("Total requests: %d\n", metrics.TotalRequests)
fmt.Printf("Average response time: %s\n", metrics.AvgResponseTime)
fmt.Printf("Error rate: %.2f%%\n", metrics.ErrorRate)
```

## 🔒 Security

### API Key Management

```go
// Rotate API key
newKey, err := client.Auth.RotateAPIKey(ctx)
if err != nil {
    return err
}

// Update client configuration
client.UpdateAPIKey(newKey)
```

## 📈 Advanced Usage

### Custom HTTP Client

```go
import "net/http"

httpClient := &http.Client{
    Timeout: 30 * time.Second,
    Transport: &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
    },
}

config := &prsm.Config{
    BaseURL:    "https://api.prsm.ai",
    APIKey:     "your-api-key",
    HTTPClient: httpClient,
}

client, err := prsm.NewClient(config)
```

### Context and Cancellation

```go
// Request with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

session, err := client.NWTN.SubmitQuery(ctx, query)

// Request with cancellation
ctx, cancel := context.WithCancel(context.Background())

go func() {
    time.Sleep(10 * time.Second)
    cancel() // Cancel after 10 seconds
}()

session, err := client.NWTN.SubmitQuery(ctx, query)
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