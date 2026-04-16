# PRSM Quickstart Guide

Get from zero to a running PRSM instance in under 5 minutes.

## Prerequisites

- **Python**: 3.11 or higher
- **OS**: macOS, Linux, or Windows with WSL
- **Memory**: 4GB RAM minimum (16GB recommended for ML workloads)

## Installation

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM
python3 -m venv .venv && source .venv/bin/activate
```

### 2. Install PRSM

```bash
pip install -e .
```

This installs PRSM in editable mode with the `prsm` CLI command.

### 3. Configure (Optional)

```bash
cp .env.example .env
```

Edit `.env` if you want to configure AI provider API keys (OpenAI, Anthropic) or external services (PostgreSQL, Redis, IPFS). The system works with defaults — no external services required for basic operation.

### 4. Start the API Server

```bash
prsm serve
```

The server starts on `http://localhost:8000`. Verify it's running:

```bash
curl http://localhost:8000/health
```

You should see a JSON response with `"status": "healthy"`.

## CLI Commands

```bash
prsm --help              # Show all commands
prsm serve               # Start the API server
prsm serve --port 9000   # Start on a different port
prsm serve --reload      # Start with auto-reload (development)
prsm status              # Show system status
prsm init                # Initialize configuration (copies .env.example)
prsm node start          # Start a PRSM node (single-node mode)
prsm node start --wizard # Interactive setup wizard
prsm query "your query"  # Submit a query (requires running server)
```

## API Endpoints

Once the server is running, key endpoints include:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/v1/` | GET | API root |
| `/docs` | GET | Interactive API documentation (Swagger) |
| `/redoc` | GET | Alternative API docs |

Visit `http://localhost:8000/docs` for the full interactive API reference.

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev,test]"

# Run the full suite (793 tests)
pytest

# Run with coverage
pytest --cov=prsm --cov-report=term-missing

# Run by category
pytest -m unit
pytest -m integration
```

## Project Structure

```
prsm/
  cli.py                          # CLI entry point (prsm command)
  core/
    config.py                     # Application settings
    database.py                   # Database models and queries
  interface/api/
    main.py                       # FastAPI application
    app_factory.py                # Application factory
    core_endpoints.py             # Health, root endpoints
  compute/
    nwtn/                         # NWTN orchestration engine
      orchestrator.py             # Main 5-agent orchestrator
      engines/                    # Reasoning engines (MCTS, meta-reasoning)
      architectures/              # SSM core, model architectures
    agents/                       # Agent pipeline (Architect, Solver, etc.)
    teachers/                     # Teacher model framework
  tokenomics/                     # FTNS token economy
  economics/                      # Economic simulation models
  storage/                        # Decentralized storage layer
  safety/                         # Safety constraints and governance
  federation/                     # P2P networking (in development)
```

## What Works Today

- **API Server**: Full FastAPI application with health checks, CORS, WebSocket support
- **NWTN Pipeline**: 5-agent orchestration (Architect, Primer, Solver, Verifier, Scribe)
- **FTNS Ledger**: Token accounting with microsecond precision
- **Teacher Models**: Create and train specialized AI models
- **Deterministic Inference**: Reproducible SSM computation across instances
- **Search Reasoning**: MCTS-based hypothesis exploration

## What's In Development

- **P2P Networking**: Multi-node discovery and communication
- **Blockchain Consensus**: Distributed proof-of-inference validation
- **IPFS Integration**: Full decentralized model/data storage
- **Production Deployment**: Kubernetes orchestration and scaling

## Troubleshooting

### `prsm: command not found`
Make sure you've activated the virtual environment and installed with `pip install -e .`:
```bash
source .venv/bin/activate
pip install -e .
```

### Import errors on startup
Ensure you're using Python 3.11+:
```bash
python3 --version
```

### Server won't start
Check if port 8000 is already in use:
```bash
prsm serve --port 8001
```

## Next Steps

- Explore the API at `http://localhost:8000/docs`
- Read the [Architecture Guide](ARCHITECTURE.md) for deeper technical details
- Check `docs/business/` for project roadmap and tokenomics
- Run `pytest` to verify everything works
