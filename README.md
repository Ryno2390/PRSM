# PRSM: Protocol for Recursive Scientific Modeling

PRSM is a peer-to-peer collaboration framework for neuro-symbolic AI research. It combines three pillars — a compute network for AI orchestration, decentralized storage for scientific artifacts, and a token economy (FTNS) that incentivizes contributions. The goal is to make scientific AI development open, reproducible, and collectively owned.

**Current version: 0.2.0 (Alpha)** | [www.prsm-network.com](https://www.prsm-network.com)

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM
python3 -m venv .venv && source .venv/bin/activate

# Install (includes all runtime dependencies)
pip install -e .
```

### Start a PRSM node

This is the primary way to use PRSM. A node runs locally, contributes compute resources, and connects to the P2P network when peers are available.

```bash
# Recommended: interactive setup wizard
prsm node start --wizard

# Or start with defaults (zero config required)
prsm node start
```

The node starts a live dashboard showing your identity, FTNS balance, connected peers, compute activity, and more. A local management API is also available at `http://localhost:8000`.

### Verify your node is running

In a separate terminal:

```bash
# Health check
curl http://localhost:8000/health

# Full node status (identity, peers, balance, compute, storage)
curl http://localhost:8000/status

# Check your FTNS balance (new nodes receive a 100 FTNS welcome grant)
curl http://localhost:8000/balance
```

---

## Try It: Submit a Compute Job

A single PRSM node can execute compute jobs locally. Try submitting a CPU benchmark:

```bash
# Submit a benchmark job (costs 1.0 FTNS from your balance)
curl -s -X POST http://localhost:8000/compute/submit \
  -H 'Content-Type: application/json' \
  -d '{"job_type": "benchmark", "payload": {"iterations": 100000}, "ftns_budget": 1.0}'
```

The response includes a `job_id`. Check the result:

```bash
# Replace <job_id> with the job_id from the response above
curl -s http://localhost:8000/compute/job/<job_id> | python3 -m json.tool
```

Once the job completes, verify your balance reflects the compute earnings:

```bash
curl -s http://localhost:8000/balance | python3 -m json.tool
```

You can also submit `inference` and `embedding` job types. See `POST /compute/submit` in the API reference below.

---

## Connect Two Nodes Locally

To test P2P features (peer discovery, cross-node compute, gossip), run two nodes on different ports:

```bash
# Terminal 1: Start the first node
prsm node start --p2p-port 9001 --api-port 8001 --no-dashboard

# Terminal 2: Start the second node, bootstrapping to the first
prsm node start --p2p-port 9002 --api-port 8002 --bootstrap 127.0.0.1:9001 --no-dashboard
```

Verify they connected:

```bash
# Check peers on node 1
curl -s http://localhost:8001/peers | python3 -m json.tool

# Check peers on node 2
curl -s http://localhost:8002/peers | python3 -m json.tool
```

Both should show `connected_count: 1`. Now submit a compute job on node 2 — node 1 will pick it up and execute it:

```bash
# Submit job from node 2
curl -s -X POST http://localhost:8002/compute/submit \
  -H 'Content-Type: application/json' \
  -d '{"job_type": "benchmark", "payload": {"iterations": 50000}, "ftns_budget": 1.0}'
```

---

## Running the PRSM API Server

For application development against PRSM's platform APIs (separate from the P2P node), use:

```bash
# Configure (optional — works with defaults)
cp .env.example .env   # edit if needed

# Start the API server
prsm serve

# Verify it's running
curl http://localhost:8000/health
```

The server starts on `localhost:8000`. External services (Redis, PostgreSQL, IPFS) show as "unhealthy" until configured, but the core API is fully functional. See `prsm --help` for all CLI options.

---

## Node Configuration

### What the wizard configures

1. **Display name** — how your node appears on the network
2. **Role** — full (compute + storage), compute-only, or storage-only
3. **Resources** — auto-detects CPU/RAM/GPU, you set allocation percentages
4. **IPFS** — auto-detects local daemon for storage features
5. **Ports** — P2P (default 9001) and management API (default 8000)
6. **Bootstrap** — address of an existing node to join the network

Configuration is saved to `~/.prsm/node_config.json`. You can edit this file directly or re-run the wizard.

### Node management API

While a node is running, a local management API is available:

| Endpoint | Description |
|---|---|
| `GET /status` | Node status, peers, balance, capabilities |
| `GET /health` | Health check |
| `GET /peers` | Connected and known peers |
| `GET /balance` | FTNS balance and recent transactions |
| `POST /compute/submit` | Submit a compute job (`benchmark`, `inference`, `embedding`) |
| `GET /compute/job/{id}` | Check job status and result |
| `GET /compute/stats` | Compute provider statistics |
| `POST /content/upload` | Upload content with provenance tracking |
| `GET /content/search?q=` | Search the content index |
| `GET /transactions` | Transaction history |
| `GET /agents` | List known agents |
| `GET /storage/stats` | Storage provider statistics |
| `POST /ledger/transfer` | Transfer FTNS to another wallet |

### CLI commands

```bash
prsm node start              # Start with defaults or saved config
prsm node start --wizard     # Interactive setup wizard
prsm node start --bootstrap HOST:PORT  # Join via a specific peer
prsm node start --no-dashboard         # Static output (no live TUI)
prsm node info               # Show node identity and config
prsm node peers              # List connected peers (requires running node)
```

### Optional: IPFS for storage

Storage features require a local IPFS daemon ([Kubo](https://docs.ipfs.tech/install/)). Without it, the node operates normally but skips storage-related tasks. To enable:

```bash
# Install Kubo, then:
ipfs init
ipfs daemon &
prsm node start   # storage features auto-detected
```

### Network and bootstrap

PRSM nodes discover each other through bootstrap peers. The default configuration points to `wss://bootstrap.prsm-network.com`. If the bootstrap server is unavailable, the node starts in **local mode** — fully functional for local compute, but peer discovery is deferred until inbound connections arrive or bootstrap recovers. See `docs/SECURE_SETUP.md` for bootstrap configuration details.

---

## Architecture Overview

PRSM is organized around four pillars:

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
| P2P node (`prsm node start`) | **Working** | Single-node and multi-node |
| CLI (`prsm` command) | **Working** | `serve`, `node start`, `node info`, `node peers` |
| Local compute (single-node) | **Working** | Benchmark, inference, embedding jobs |
| FastAPI server (`prsm serve`) | **Working** | Platform API for app development |
| NWTN 5-agent pipeline | **Working** | Orchestration with mocked LLM backends |
| FTNS accounting ledger | **Working** | SQLite-backed, zero config, DAG-based |
| P2P networking | **Working** | WebSocket transport with gossip protocol |
| Node identity | **Working** | Ed25519 keypair, persisted to `~/.prsm/` |
| Compute marketplace | **Working** | Job offers, acceptance, execution, payment |
| IPFS storage integration | **Working** | Requires local IPFS daemon (Kubo) |
| Test suite | **Working** | 920+ tests passing |
| Bootstrap network | **Alpha** | `wss://bootstrap.prsm-network.com` |
| Production deployment (K8s) | **Planned** | Configs exist, not production-tested |

---

## For Developers

### Running Tests

```bash
# Install dev + test dependencies
pip install -e ".[dev,test]"

# Run the full test suite
pytest --timeout=120

# Run with coverage
pytest --timeout=120 --cov=prsm --cov-report=term-missing

# Run specific test categories
pytest -m unit
pytest -m integration
```

### PostgreSQL Integration Tests

Some integration tests require PostgreSQL for testing database concurrency features. To run these:

```bash
# Option 1: Using Docker Compose (recommended)
docker-compose -f docker-compose.test.yml up -d
DATABASE_URL=postgresql+asyncpg://prsm:test@localhost:5433/prsm_test pytest tests/integration/
docker-compose -f docker-compose.test.yml down -v

# Option 2: Using local PostgreSQL
# Set up a test database:
createdb prsm_test
DATABASE_URL=postgresql+asyncpg://your_user:your_password@localhost:5432/prsm_test pytest tests/integration/
```

The GitHub CI pipeline automatically provisions PostgreSQL 16 for integration tests.

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
tests/                      # Test suite (920+ tests)
docs/                       # Documentation
config/                     # Configuration templates
scripts/                    # Utility scripts
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Run the test suite: `pytest`
4. Submit a pull request

See `docs/CONTRIBUTOR_ONBOARDING.md` for detailed contributor guides at all experience levels.

---

## For Investors

See the `docs/business/` directory for:
- Business model and tokenomics documentation
- Technical architecture deep-dives
- Development roadmap and milestones

---

**License:** MIT | **Python:** 3.11+ | **Website:** [www.prsm-network.com](https://www.prsm-network.com) | **Repo:** [github.com/Ryno2390/PRSM](https://github.com/Ryno2390/PRSM)
