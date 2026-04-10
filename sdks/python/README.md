# PRSM Python SDK

Official Python client for **PRSM — a P2P infrastructure protocol for open-source collaboration**. PRSM aggregates consumer-node storage, compute, and data into a mesh network any third-party LLM can reach through MCP tools. This SDK lets your Python application drive the PRSM infrastructure layer directly. Reasoning happens in your LLM of choice — PRSM supplies the distributed compute, storage, and FTNS settlement.

[![PyPI version](https://badge.fury.io/py/prsm-python-sdk.svg)](https://badge.fury.io/py/prsm-python-sdk)
[![Python versions](https://img.shields.io/pypi/pyversions/prsm-python-sdk.svg)](https://pypi.org/project/prsm-python-sdk/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

## Installation

```bash
pip install prsm-python-sdk
```

## Quick Start

```python
import asyncio
from prsm_sdk import PRSMClient

async def main():
    # Initialize client
    client = PRSMClient(api_key="your_api_key_here")
    
    # Ring 1-10 compute query
    response = await client.query(
        "Explain quantum computing in simple terms",
        budget=1.0,
    )
    print(response.content)

    # Check FTNS balance
    balance = await client.ftns.get_balance()
    print(f"FTNS Balance: {balance.available_balance}")

    # Publish a dataset to the ContentStore
    cid = await client.storage.upload(
        "./my_dataset.parquet",
        description="EV registrations 2025",
        royalty_rate=0.05,
    )
    print(f"Uploaded: {cid}")

    await client.close()

# Run the example
asyncio.run(main())
```

## Features

- **Ring 1-10 Compute Client** — submit quotes and queries through the full PRSM pipeline
- **FTNS Token Management** — balance checks, transfers, yield estimation
- **Authentication** — secure API key and JWT token handling
- **ContentStore** — upload content with royalty tracking, download by CID
- **MCP Tool Surface** — drive the same 16 tools third-party LLMs use
- **WebSocket Streaming** — live job progress updates for long-running compute jobs
- **P2P Network Access** — direct access to the PRSM mesh via your local node

## Advanced Usage

### Streaming Responses

```python
async def stream_example():
    client = PRSMClient(api_key="your_api_key")
    
    print("AI Response: ", end="")
    async for chunk in client.stream("Write a short story about AI"):
        print(chunk.content, end="", flush=True)
    print()  # New line at end
    
    await client.close()
```

### Cost Estimation

```python
async def cost_example():
    client = PRSMClient(api_key="your_api_key")
    
    # Estimate cost before running
    cost = await client.estimate_cost("Complex scientific query about protein folding")
    print(f"Estimated cost: {cost} FTNS")
    
    # Check if we have enough balance
    balance = await client.ftns.get_balance()
    if balance.available_balance >= cost:
        response = await client.query("Complex scientific query about protein folding")
        print(response.content)
    else:
        print("Insufficient FTNS balance")
    
    await client.close()
```

### Publish Data with Royalty Tracking

```python
async def storage_example():
    client = PRSMClient(api_key="your_api_key")

    # Upload content through your local node's ContentStore
    result = await client.storage.upload(
        "./nada_registrations_2025.parquet",
        description="NADA NC Vehicle Registrations 2025",
        royalty_rate=0.05,   # 0.05 FTNS earned per access
        replicas=5,
    )
    print(f"CID: {result.cid}")
    print(f"You earn 80% of every query that hits this content")

    await client.close()
```

### Tool Execution (MCP)

```python
async def tools_example():
    client = PRSMClient(api_key="your_api_key")
    
    # List available tools
    tools = await client.tools.list_available()
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    # Execute a tool
    if "web_search" in [tool.name for tool in tools]:
        result = await client.tools.execute(
            tool_name="web_search",
            parameters={"query": "latest AI research papers"}
        )
        print(f"Search results: {result.result}")
    
    await client.close()
```

### Error Handling

```python
from prsm_sdk import (
    PRSMClient, 
    InsufficientFundsError, 
    SafetyViolationError,
    AuthenticationError
)

async def error_handling_example():
    try:
        client = PRSMClient(api_key="invalid_key")
        response = await client.query("Hello world")
    except AuthenticationError:
        print("Invalid API key")
    except InsufficientFundsError as e:
        print(f"Not enough FTNS: need {e.details['required']}, have {e.details['available']}")
    except SafetyViolationError as e:
        print(f"Content safety violation: {e.message}")
    finally:
        await client.close()
```

## Configuration

### Environment Variables

```bash
export PRSM_API_KEY="your_api_key_here"
export PRSM_BASE_URL="https://api.prsm.ai/v1"  # Optional
```

### Custom Configuration

```python
client = PRSMClient(
    api_key="your_key",
    base_url="https://custom-api.prsm.ai/v1",
    websocket_url="wss://custom-ws.prsm.ai/v1", 
    timeout=120,  # 2 minutes
    max_retries=5
)
```

## API Reference

### PRSMClient

The main client class for interacting with PRSM.

#### Methods

- `query(prompt, **kwargs)` - Execute AI query
- `stream(prompt, **kwargs)` - Stream AI response
- `estimate_cost(prompt, **kwargs)` - Estimate query cost
- `get_safety_status()` - Get safety monitoring status
- `list_available_models()` - List available models
- `health_check()` - Check API health
- `close()` - Close client connections

### FTNSManager

Manage FTNS token balance and transactions.

#### Methods

- `get_balance()` - Get current token balance
- `get_transaction_history(limit=50)` - Get recent transactions
- `transfer(to_address, amount)` - Transfer tokens

### ModelMarketplace

Browse and interact with the model marketplace.

#### Methods

- `search_models(query, **filters)` - Search for models
- `get_model_info(model_id)` - Get detailed model information
- `list_categories()` - List model categories

### ToolExecutor

Execute MCP tools and manage tool interactions.

#### Methods

- `list_available()` - List available tools
- `execute(tool_name, parameters)` - Execute a tool
- `get_tool_info(tool_name)` - Get tool specifications

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/PRSM-AI/PRSM.git
cd PRSM/sdks/python

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy prsm_sdk

# Format code
black prsm_sdk
isort prsm_sdk
```

### Running Tests

```bash
# Run unit tests (no server needed)
pytest tests/test_client.py tests/test_models.py

# Run all tests
pytest

# Run with coverage
pytest --cov=prsm_sdk --cov-report=html

# Run specific test file
pytest tests/test_client.py
```

### Running Integration Tests

Integration tests require a live PRSM server to be running:

```bash
# Start a PRSM node
prsm node start &

# Set your API key and run integration tests
PRSM_TEST_API_KEY=your_key pytest tests/test_integration.py -v

# Or with a custom server URL
PRSM_TEST_URL=http://localhost:8000 PRSM_TEST_API_KEY=your_key pytest tests/test_integration.py -v
```

Integration tests will be skipped automatically if no server is available.

## Examples

See the [examples](examples/) directory for more comprehensive examples:

- [Basic Usage](examples/basic_usage.py)
- [Streaming Responses](examples/streaming.py)
- [Marketplace Integration](examples/marketplace.py)
- [Tool Execution](examples/tools.py)
- [Cost Management](examples/cost_management.py)

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

- 📚 **Documentation**: [docs.prsm.ai/python-sdk](https://docs.prsm.ai/python-sdk)
- 🐛 **Issues**: [GitHub Issues](https://github.com/PRSM-AI/PRSM/issues)
- 💬 **Community**: [Discord](https://discord.gg/prsm)
- 📧 **Email**: [dev@prsm.ai](mailto:dev@prsm.ai)