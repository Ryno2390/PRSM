# Building on PRSM — SDK Developer Guide

A practical guide for developers building applications on the PRSM platform.

## Overview

PRSM provides SDKs in three languages, allowing you to integrate decentralized AI capabilities into your applications:

| Language | Package | Status | Import |
|----------|---------|--------|--------|
| Python | `prsm-python-sdk` | ✅ Complete | `pip install prsm-python-sdk` |
| JavaScript/TypeScript | `@prsm/js-sdk` | ✅ Complete | `npm install @prsm/js-sdk` |
| Go | `prsm-go-sdk` | ✅ Complete | `go get github.com/prsm-network/prsm-go-sdk` |

All SDKs provide:
- Ring 1-10 compute query execution with cost control (quote → run → settle)
- FTNS token balance checks and transfers
- ContentStore upload/download with royalty tracking
- Node status and hardware benchmarking
- WebSocket streaming for long-running compute jobs

PRSM itself does not host models — reasoning happens in your third-party LLM of choice. The SDKs let you drive the PRSM infrastructure layer (compute dispatch, storage, FTNS settlement) from your application.

---

## Python SDK

### Installation

```bash
pip install prsm-python-sdk
```

### Quick Start

```python
import asyncio
from prsm_sdk import PRSMClient

async def main():
    # Initialize client
    async with PRSMClient(api_key="your_api_key") as client:
        # Simple query
        response = await client.query("Explain quantum entanglement")
        print(response.content)

        # Check balance
        balance = await client.ftns.get_balance()
        print(f"Available: {balance.available_balance} FTNS")

asyncio.run(main())
```

### Common Use Cases

#### AI Query with Budget Control

```python
async def query_with_budget(client, prompt: str, max_cost: float = 1.0):
    """Execute a query with cost controls."""
    # Estimate cost first
    estimated = await client.estimate_cost(prompt)

    if estimated > max_cost:
        raise ValueError(f"Cost {estimated} exceeds budget {max_cost}")

    # Execute with budget limit
    response = await client.query(
        prompt=prompt,
        max_tokens=2000,
        temperature=0.7
    )

    return response
```

#### Upload and Share a Dataset

```python
async def upload_dataset(client, filepath: str):
    """Upload a dataset to PRSM storage."""
    # Read file
    with open(filepath, "rb") as f:
        data = f.read()

    # Upload to IPFS
    result = await client.storage.upload_bytes(
        data=data,
        filename=filepath.split("/")[-1],
        content_type="dataset",
        tags=["research", "ml-training"],
        is_public=True
    )

    print(f"Uploaded: {result.cid}")
    print(f"Gateway URL: {result.gateway_url}")

    return result
```

#### Stream Real-Time Results

```python
async def stream_query(client, prompt: str):
    """Stream AI response in real-time."""
    print("Response: ", end="", flush=True)

    async for chunk in client.stream(prompt):
        print(chunk.content, end="", flush=True)

    print()  # Newline at end
```

#### Cost Quote Before Executing

```python
async def quote_and_run(client, query: str):
    """Get a free quote first, then run the query if the cost is acceptable."""
    quote = await client.quote(query, shards=5, tier="t2")
    print(f"Estimated cost: {quote.total_ftns} FTNS")

    if quote.total_ftns > 5.0:
        print("Query too expensive, skipping")
        return None

    return await client.query(query, budget=quote.total_ftns * 1.1)
```

### Error Handling

```python
from prsm_sdk import PRSMClient
from prsm_sdk.exceptions import (
    AuthenticationError,
    InsufficientFundsError,
    RateLimitError,
    PRSMError
)

async def safe_query(client, prompt: str):
    try:
        return await client.query(prompt)

    except AuthenticationError:
        print("Invalid API key")
        raise

    except InsufficientFundsError as e:
        print(f"Need {e.required} FTNS, have {e.available}")
        # Optionally: transfer tokens or wait for rewards

    except RateLimitError as e:
        print(f"Rate limited. Retry after {e.retry_after}s")
        await asyncio.sleep(e.retry_after)
        return await client.query(prompt)  # Retry

    except PRSMError as e:
        print(f"PRSM error: {e}")
        raise
```

---

## JavaScript/TypeScript SDK

### Installation

```bash
npm install @prsm/js-sdk
# or
yarn add @prsm/js-sdk
```

### Quick Start

```typescript
import { PRSMClient } from '@prsm/js-sdk';

async function main() {
    const client = new PRSMClient({
        apiKey: process.env.PRSM_API_KEY
    });

    // Simple query
    const response = await client.query("What is machine learning?");
    console.log(response.content);

    // Check balance
    const balance = await client.ftns.getBalance();
    console.log(`Available: ${balance.availableBalance} FTNS`);

    await client.close();
}

main();
```

### Common Use Cases

#### Query with Streaming

```typescript
async function streamQuery(client: PRSMClient, prompt: string) {
    const stream = await client.stream(prompt);

    for await (const chunk of stream) {
        process.stdout.write(chunk.content);
    }

    console.log(); // Newline
}
```

#### Publish Data with Royalty Tracking

```typescript
async function publishDataset(client: PRSMClient, filePath: string) {
    const content = await fs.promises.readFile(filePath);

    const result = await client.storage.upload({
        filename: path.basename(filePath),
        content,
        description: "Vehicle registration data 2025",
        royaltyRate: 0.05,   // 0.05 FTNS per access
        replicas: 5,
    });

    console.log(`Uploaded: ${result.cid}`);
    console.log(`Earn 80% of every query that hits this content`);
    return result;
}
```

#### Express.js Integration

```typescript
import express from 'express';
import { PRSMClient } from '@prsm/js-sdk';

const app = express();
const prsm = new PRSMClient({ apiKey: process.env.PRSM_API_KEY });

app.post('/query', async (req, res) => {
    try {
        const { prompt } = req.body;

        // Estimate cost
        const cost = await prsm.estimateCost(prompt);

        // Check budget
        if (cost > parseFloat(process.env.MAX_QUERY_COST || '10')) {
            return res.status(402).json({ error: 'Query too expensive' });
        }

        // Execute
        const response = await prsm.query(prompt);
        res.json({ result: response.content, cost: response.ftnsCost });

    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});
```

---

## Go SDK

### Installation

```bash
go get github.com/prsm-network/prsm-go-sdk
```

### Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    prsm "github.com/prsm-network/prsm-go-sdk"
    "github.com/prsm-network/prsm-go-sdk/types"
)

func main() {
    ctx := context.Background()

    // Initialize client
    client := prsm.New("your_api_key")

    // Initialize connection
    if err := client.Initialize(ctx); err != nil {
        log.Fatal(err)
    }
    defer client.Destroy()

    // Simple query
    req := &types.QueryRequest{
        Prompt:      "Explain neural networks",
        MaxTokens:   1000,
        Temperature: 0.7,
    }

    response, err := client.Query(ctx, req)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Content)
}
```

### Common Use Cases

#### Query with Cost Estimation

```go
func queryWithBudget(ctx context.Context, client *prsm.Client, prompt string, maxCost float64) (*types.PRSMResponse, error) {
    // Estimate cost
    cost, err := client.EstimateCost(ctx, prompt, nil)
    if err != nil {
        return nil, err
    }

    if cost > maxCost {
        return nil, fmt.Errorf("estimated cost %.2f exceeds budget %.2f", cost, maxCost)
    }

    // Execute query
    req := &types.QueryRequest{
        Prompt:    prompt,
        MaxTokens: 1000,
    }

    return client.Query(ctx, req)
}
```

#### FTNS Operations

```go
func ftnsOperations(ctx context.Context, client *prsm.Client) error {
    // Get balance
    balance, err := client.FTNS.GetBalance(ctx)
    if err != nil {
        return err
    }
    fmt.Printf("Balance: %.2f FTNS\n", balance.AvailableBalance)

    // Transfer tokens
    transferResp, err := client.FTNS.Transfer(ctx, &ftns.TransferRequest{
        ToAddress: "recipient_address",
        Amount:    10.0,
        Memo:      "Payment for services",
    })
    if err != nil {
        return err
    }
    fmt.Printf("Transfer ID: %s\n", transferResp.TransactionID)

    return nil
}
```

#### Ring 1-10 Compute Query with Budget

```go
func runQuery(ctx context.Context, client *prsm.Client) error {
    // Get a free quote first
    quote, err := client.Compute.Quote(ctx, &compute.QuoteRequest{
        Query:  "EV adoption trends in NC",
        Shards: 5,
        Tier:   "t2",
    })
    if err != nil {
        return err
    }
    fmt.Printf("Estimated cost: %.2f FTNS\n", quote.TotalFTNS)

    // Execute the query with a budget
    result, err := client.Compute.Run(ctx, &compute.RunRequest{
        Query:   "EV adoption trends in NC",
        Budget:  quote.TotalFTNS * 1.1,   // 10% headroom
        Privacy: "standard",
    })
    if err != nil {
        return err
    }

    fmt.Printf("Result: %s\n", result.Content)
    fmt.Printf("FTNS spent: %.4f\n", result.FTNSSpent)
    return nil
}
```

---

## Authentication

### API Keys

API keys authenticate requests and track usage. Get your key from the PRSM dashboard or CLI:

```bash
# Create API key via CLI
prsm auth create-key --name "my-app" --expires 365d
```

**Security Best Practices:**
- Store keys in environment variables, never in code
- Use different keys for different environments
- Rotate keys periodically
- Use minimal required permissions

### Environment Variables

```bash
# Recommended: Set via environment
export PRSM_API_KEY="your_api_key"

# Optional: Override defaults
export PRSM_BASE_URL="https://api.prsm.ai/v1"
export PRSM_WEBSOCKET_URL="wss://ws.prsm.ai/v1"
```

### JWT Tokens (Advanced)

For long-running sessions or OAuth flows:

```python
# Get JWT token
token = await client.auth.get_token()

# Use token for subsequent requests
client2 = PRSMClient(token=token)
```

---

## Error Handling

### Error Types

| Error | Meaning | Action |
|-------|---------|--------|
| `AuthenticationError` | Invalid/expired API key | Check credentials |
| `InsufficientFundsError` | Not enough FTNS | Earn or purchase tokens |
| `RateLimitError` | Too many requests | Wait and retry |
| `NetworkError` | Connection failed | Check connectivity |
| `ModelNotFoundError` | Requested model unavailable | Try another model |
| `SafetyViolationError` | Content policy violation | Modify query content |

### Retry Logic

```python
import asyncio
from prsm_sdk.exceptions import RateLimitError, NetworkError

async def query_with_retry(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.query(prompt)

        except RateLimitError as e:
            wait_time = e.retry_after or (2 ** attempt)
            await asyncio.sleep(wait_time)

        except NetworkError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    raise Exception("Max retries exceeded")
```

---

## Rate Limits

### Default Limits

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/compute/run` | 100 requests | 1 minute |
| `/compute/quote` | 300 requests | 1 minute |
| `/ftns/*` | 50 requests | 1 minute |
| `/storage/*` | 30 requests | 1 minute |
| `/mcp/*` | 200 requests | 1 minute |

### Handling Rate Limits

```python
async def handle_rate_limit(client, prompt):
    try:
        return await client.query(prompt)
    except RateLimitError as e:
        # Option 1: Wait and retry
        await asyncio.sleep(e.retry_after)
        return await client.query(prompt)

        # Option 2: Fail gracefully
        # return {"error": "Rate limited", "retry_after": e.retry_after}
```

### Custom Rate Limits

Enterprise users can request higher limits. Contact support@prsm.ai.

---

## Examples Repository

See `sdks/python/examples/` for production-ready patterns:

| Example | Description |
|---------|-------------|
| `basic_usage.py` | Getting started with the SDK |
| `streaming.py` | Real-time response streaming |
| `storage_upload.py` | Publish data with royalty tracking |
| `tools.py` | MCP tool execution |
| `cost_management.py` | Budget control and cost tracking |
| `production/fastapi_integration.py` | FastAPI backend integration |
| `production/docker_deployment.py` | Container deployment examples |

### Running Examples

```bash
cd sdks/python/examples

# Set API key
export PRSM_API_KEY="your_key"

# Run example
python basic_usage.py
```

---

## Additional Resources

- **API Reference**: [docs.prsm.ai/api](https://docs.prsm.ai/api)
- **SDK Source**: [github.com/prsm-network/PRSM/tree/main/sdks](https://github.com/prsm-network/PRSM/tree/main/sdks)
- **Changelog**: See `CHANGELOG.md` in each SDK directory
- **Contributing**: [github.com/prsm-network/PRSM/blob/main/CONTRIBUTING.md](https://github.com/prsm-network/PRSM/blob/main/CONTRIBUTING.md)

---

## Support

- **GitHub Issues**: [github.com/prsm-network/PRSM/issues](https://github.com/prsm-network/PRSM/issues)
- **Discord**: [discord.gg/prsm](https://discord.gg/prsm)
- **Email**: dev@prsm.ai
