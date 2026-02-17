# PRSM: Protocol for Recursive Scientific Modeling

PRSM is a peer-to-peer collaboration framework for neuro-symbolic AI research. It combines three pillars — a compute network for AI orchestration, decentralized storage for scientific artifacts, and a token economy (FTNS) that incentivizes contributions. The goal is to make scientific AI development open, reproducible, and collectively owned.

**Current version: 0.1.0 (Alpha)**

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM
python3 -m venv .venv && source .venv/bin/activate

# Install
pip install -e .

# Configure (optional — works with defaults)
cp .env.example .env   # edit if needed

# Start the API server
prsm serve

# Verify it's running
curl http://localhost:8000/health
```

The server starts on `localhost:8000` by default. See `prsm --help` for all CLI options.

---

## Running a PRSM Node

Any user can join the PRSM network by running a node. Nodes contribute compute, storage, or both, and earn FTNS tokens for their contributions.

```bash
# Interactive setup wizard (recommended for first run)
prsm node start --wizard

# Or start with defaults
prsm node start

# Start with a specific bootstrap node
prsm node start --bootstrap 203.0.113.10:9001

# Check node identity and config
prsm node info

# List connected peers (while node is running)
prsm node peers
```

### What the wizard configures

1. **Display name** — how your node appears on the network
2. **Role** — full (compute + storage), compute-only, or storage-only
3. **Resources** — auto-detects CPU/RAM/GPU, you set allocation percentages
4. **IPFS** — auto-detects local daemon for storage features
5. **Ports** — P2P (default 9001) and management API (default 8000)
6. **Bootstrap** — address of an existing node to join the network

### Node management API

While a node is running, a local management API is available:

| Endpoint | Description |
|---|---|
| `GET /status` | Node status, peers, balance, capabilities |
| `GET /peers` | Connected and known peers |
| `GET /balance` | FTNS balance and recent transactions |
| `POST /compute/submit` | Submit a compute job to the network |
| `POST /content/upload` | Upload content with provenance tracking |
| `GET /transactions` | Transaction history |
| `GET /health` | Health check |

### Optional: IPFS for storage

Storage features require a local IPFS daemon ([Kubo](https://docs.ipfs.tech/install/)). Without it, the node operates normally but skips storage-related tasks. To enable:

```bash
# Install Kubo, then:
ipfs init
ipfs daemon &
prsm node start   # storage features auto-detected
```

---

## Architecture Overview

PRSM is organized around three pillars:

### 1. Compute Network (NWTN)
The Neural Web for Transformation Networking orchestrates multi-agent AI pipelines. It includes state-space models for efficient inference, Monte Carlo tree search for hypothesis exploration, and a 5-agent pipeline (Architect, Primer, Solver, Verifier, Scribe).

**Key modules:** `prsm/compute/nwtn/`, `prsm/compute/agents/`, `prsm/compute/teachers/`

### 2. Decentralized Storage
IPFS-based content-addressed storage for models, datasets, and research artifacts. Includes sharding, retrieval, and integrity verification.

**Key modules:** `prsm/storage/`, `prsm/core/ipfs_model.py`

### 3. Token Economy (FTNS)
The Federated Token for Networked Science handles resource accounting, staking, and incentive distribution. Includes a microsecond-precision accounting ledger and a local SQLite-backed ledger for individual node accounting.

**Key modules:** `prsm/tokenomics/`, `prsm/economics/`, `prsm/node/local_ledger.py`

### 4. P2P Node Network
WebSocket-based peer-to-peer connectivity with gossip protocol, peer discovery, and a compute/storage marketplace. Each node has an Ed25519 identity and earns FTNS by contributing resources.

**Key modules:** `prsm/node/`

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| FastAPI server + health endpoint | **Working** | `prsm serve` starts the API |
| CLI (`prsm` command) | **Working** | `serve`, `status`, `init`, `node` commands |
| NWTN 5-agent pipeline | **Working** | Orchestration with mocked LLM backends |
| FTNS accounting ledger | **Working** | Microsecond-precision transactions |
| Deterministic SSM inference | **Working** | Reproducible results across instances |
| MCTS search reasoning | **Working** | Hypothesis tree exploration |
| Teacher model framework | **Working** | Create and train specialized models |
| Test suite | **Working** | 793+ tests passing |
| P2P networking | **Working** | WebSocket transport with gossip protocol |
| Node identity | **Working** | Ed25519 keypair, persisted to `~/.prsm/` |
| Local FTNS ledger | **Working** | SQLite-backed, zero config |
| Compute marketplace | **Working** | Job offers, acceptance, execution, payment |
| IPFS storage integration | **Working** | Requires local IPFS daemon (Kubo) |
| Blockchain consensus | **In Development** | Local validation only |
| Production deployment (K8s) | **Planned** | Configs exist, not production-tested |

---

## For Developers

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev,test]"

# Run the full test suite
pytest

# Run with coverage
pytest --cov=prsm --cov-report=term-missing

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Project Structure

```
prsm/
  cli.py                    # CLI entry point
  core/                     # Config, database, validation, errors
  interface/api/            # FastAPI application and endpoints
  node/                     # P2P node implementation
    node.py                 # Main runtime orchestrator
    identity.py             # Ed25519 keypair and signing
    local_ledger.py         # SQLite FTNS ledger
    transport.py            # WebSocket P2P connections
    discovery.py            # Bootstrap + gossip peer discovery
    gossip.py               # Epidemic gossip protocol
    compute_provider.py     # Accept and execute compute jobs
    compute_requester.py    # Submit compute jobs to network
    storage_provider.py     # IPFS pin space contribution
    content_uploader.py     # Upload content with provenance
    api.py                  # Node management API
    config.py               # Node configuration
  compute/
    nwtn/                   # Neural orchestration engine
    agents/                 # Multi-agent pipeline
    teachers/               # Teacher model framework
  tokenomics/               # FTNS token economy
  economics/                # Economic modeling
  storage/                  # Decentralized storage
  safety/                   # Safety and governance
tests/                      # Test suite (793+ tests)
docs/                       # Documentation
config/                     # Configuration templates
scripts/                    # Utility scripts
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Run the test suite: `pytest`
4. Submit a pull request

---

## For Investors

See the `docs/business/` directory for:
- Business model and tokenomics documentation
- Technical architecture deep-dives
- Development roadmap and milestones

---

**License:** MIT | **Python:** >=3.11 | **Repo:** [github.com/Ryno2390/PRSM](https://github.com/Ryno2390/PRSM)
