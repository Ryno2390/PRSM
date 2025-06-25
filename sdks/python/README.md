# PRSM Python SDK

Official Python client for the Protocol for Recursive Scientific Modeling (PRSM).

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
    
    # Simple AI query
    response = await client.query("Explain quantum computing in simple terms")
    print(response.content)
    
    # Check token balance
    balance = await client.ftns.get_balance()
    print(f"FTNS Balance: {balance.available_balance}")
    
    # Search for models
    models = await client.marketplace.search_models("gpt")
    print(f"Found {len(models)} models")
    
    # Clean up
    await client.close()

# Run the example
asyncio.run(main())
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

### Model Marketplace

```python
async def marketplace_example():
    client = PRSMClient(api_key="your_api_key")
    
    # Search for specific models
    science_models = await client.marketplace.search_models(
        query="scientific research",
        min_performance=0.8,
        max_cost=0.001
    )
    
    # Use a specific model
    if science_models:
        model = science_models[0]
        response = await client.query(
            "Explain CRISPR gene editing",
            model_id=model.id
        )
        print(f"Response from {model.name}: {response.content}")
    
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
# Run all tests
pytest

# Run with coverage
pytest --cov=prsm_sdk --cov-report=html

# Run specific test file
pytest tests/test_client.py
```

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