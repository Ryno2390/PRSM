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

## Architecture Overview

PRSM is organized around three pillars:

### 1. Compute Network (NWTN)
The Neural Web for Transformation Networking orchestrates multi-agent AI pipelines. It includes state-space models for efficient inference, Monte Carlo tree search for hypothesis exploration, and a 5-agent pipeline (Architect, Primer, Solver, Verifier, Scribe).

**Key modules:** `prsm/compute/nwtn/`, `prsm/compute/agents/`, `prsm/compute/teachers/`

### 2. Decentralized Storage
IPFS-based content-addressed storage for models, datasets, and research artifacts. Includes sharding, retrieval, and integrity verification.

**Key modules:** `prsm/storage/`, `prsm/core/ipfs_model.py`

### 3. Token Economy (FTNS)
The Federated Token for Networked Science handles resource accounting, staking, and incentive distribution. Includes a microsecond-precision accounting ledger.

**Key modules:** `prsm/tokenomics/`, `prsm/economics/`

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| FastAPI server + health endpoint | **Working** | `prsm serve` starts the API |
| CLI (`prsm` command) | **Working** | `serve`, `status`, `init`, `query` commands |
| NWTN 5-agent pipeline | **Working** | Orchestration with mocked LLM backends |
| FTNS accounting ledger | **Working** | Microsecond-precision transactions |
| Deterministic SSM inference | **Working** | Reproducible results across instances |
| MCTS search reasoning | **Working** | Hypothesis tree exploration |
| Teacher model framework | **Working** | Create and train specialized models |
| Test suite | **Working** | 793/793 tests passing |
| P2P networking | **In Development** | Currently single-node mode |
| Blockchain consensus | **In Development** | Local validation only |
| IPFS storage integration | **In Development** | Works with local IPFS daemon |
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
  compute/
    nwtn/                   # Neural orchestration engine
    agents/                 # Multi-agent pipeline
    teachers/               # Teacher model framework
  tokenomics/               # FTNS token economy
  economics/                # Economic modeling
  storage/                  # Decentralized storage
  safety/                   # Safety and governance
  federation/               # P2P networking (in development)
tests/                      # Test suite (793 tests)
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
