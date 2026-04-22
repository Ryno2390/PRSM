# PRSM: P2P Infrastructure Protocol for Open-Source Collaboration

**The code goes to the data. Not the other way around.**

PRSM is a P2P infrastructure protocol that aggregates latent storage, compute, and data from consumer nodes — gaming PCs, consoles, laptops, phones — into a mesh network accessible to third-party LLMs via MCP tools. Contributors earn FTNS tokens for sharing their latent resources; users leverage PRSM infrastructure through their preferred LLMs (local or via OAuth/API). PRSM is not an AGI framework. Reasoning happens in third-party LLMs; PRSM provides the infrastructure those LLMs use to access distributed resources and data.

**Version 1.7.0** | [PyPI](https://pypi.org/project/prsm-network/) | [Getting Started](docs/GETTING_STARTED.md) | [Architecture Spec](docs/SOVEREIGN_EDGE_AI_SPEC.md)

[![PyPI version](https://badge.fury.io/py/prsm-network.svg)](https://pypi.org/project/prsm-network/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Tools](https://img.shields.io/badge/MCP%20tools-16-blue.svg)](#mcp-integration)

---

## Why PRSM Exists

**The problem:** Frontier AI labs hoard data, compute, and models behind API walls. Every query you send to a centralized API is logged, stored, and potentially trained on. Meanwhile, billions of consumer devices sit idle — gaming PCs, laptops, phones, tablets — each with storage, compute, and sometimes proprietary data that never leave the device.

**The PRSM thesis:** Build an open-source commons for AI infrastructure. Aggregate the latent resources of consumer electronics into a P2P mesh. Let any LLM — local, OAuth, or API — use that mesh via MCP tools. Pay contributors in FTNS tokens so sharing beats hoarding.

| Centralized AI (today) | PRSM |
|------------------------|------|
| You upload data to their cloud | WASM agents travel to the data |
| They see your query | Zero-persistence sandbox — nothing logged |
| One datacenter processes everything | Thousands of edge nodes work in parallel |
| You pay per token | Hybrid pricing — commodity compute + market-rate data |
| Vendor lock-in | Works with any LLM via MCP |

**The result:** A network where the model is open but the computation is private, and where contributing your idle storage/compute/data earns you a share of every query it serves.

---

## Quick Start

```bash
# Install
pip install prsm-network

# Check your hardware
prsm node benchmark

# Start your node
prsm node start

# Expose PRSM tools to any MCP-compatible LLM
prsm mcp-server
```

> **FTNS tokens:** Providers earn FTNS for sharing storage, compute, and data through their node. New nodes receive a 100 FTNS welcome grant. Third-party LLMs invoke PRSM tools via MCP; reasoning happens in the LLM, execution happens on PRSM nodes.

See the full [Getting Started Guide](docs/GETTING_STARTED.md) for detailed setup.

---

## How It Works

### The 10-Ring Architecture

PRSM is built as concentric capability rings. Each ring wraps and enriches the ones inside it.

| Ring | Name | What It Does |
|------|------|-------------|
| 1 | **The Sandbox** | WASM runtime with Wasmtime — sandboxed execution with memory/time limits |
| 2 | **The Courier** | Mobile agent dispatch — agents travel to data via P2P gossip + bidding |
| 3 | **The Swarm** | Semantic sharding — data split by meaning, parallel map-reduce across nodes |
| 4 | **The Economy** | Hybrid pricing — deterministic compute rates + market-rate data + 80/15/5 revenue splits |
| 5 | **Agent Forge** | WASM mobile agent runtime — query decomposition is performed by the caller's third-party LLM; PRSM dispatches the resulting WASM agents |
| 6 | **The Polish** | Production hardening — dynamic gas, RPC failover, CLI commands |
| 7 | **The Vault** | Confidential compute — TEE abstraction + differential privacy noise |
| 8 | **The Shield** | Model sharding — tensor parallelism + randomized pipelines + collision detection |
| 9 | **The Mind** | NWTN training pipeline — collect traces, evaluate quality, deploy fine-tuned models (reserved for future work) |
| 10 | **The Fortress** | Security — integrity verification, privacy budgets, hash-chained audit logs |

### End-to-End Flow

```
Third-party LLM (Claude/GPT/local): calls prsm_analyze via MCP

  → LLM:     Decomposes the query into WASM agent instructions
  → Ring 3:  Finds relevant semantic shards by embedding similarity
  → Ring 4:  Quotes cost: compute + data + network fee
  → Ring 3:  Fans out parallel agents to shard-holding nodes
  → Ring 2:  Each agent dispatched via gossip bidding
  → Ring 1:  Executed in WASM sandbox on provider hardware
  → Ring 7:  Differential privacy noise applied
  → Ring 3:  Results aggregated when quorum met
  → Ring 4:  FTNS settled: 80% data owner / 15% compute / 5% treasury

  ← Result returned to the LLM for final synthesis
```

---

## MCP Integration

Any LLM can use PRSM as a compute backend via the Model Context Protocol. **16 tools** are exposed:

```bash
prsm mcp-server    # Start the MCP server
```

Configure in Claude Desktop (`~/.claude/claude_desktop_config.json`):
```json
{"mcpServers": {"prsm": {"command": "python", "args": ["scripts/prsm_mcp_server.py"]}}}
```

Then Claude (or any MCP-compatible LLM) can:

| Tool | What It Does |
|------|-------------|
| `prsm_analyze` | Full Ring 1-10 pipeline — query in, answer out |
| `prsm_quote` | Cost estimate before committing (free) |
| `prsm_create_agent` | Build custom agent with 11 data operations |
| `prsm_dispatch_agent` | Execute agent on the network |
| `prsm_upload_dataset` | Publish data with pricing |
| `prsm_list_datasets` | Browse available datasets |
| `prsm_search_shards` | Find relevant data shards |
| `prsm_yield_estimate` | "What would I earn?" |
| `prsm_stake` | Staking tier info |
| `prsm_revenue_split` | Calculate 80/15/5 distribution |
| `prsm_hardware_benchmark` | GPU, TFLOPS, tier, TEE detection |
| `prsm_node_status` | Ring 1-10 health check |
| `prsm_agent_status` | Check running agent |
| `prsm_settlement_stats` | FTNS settlement queue |
| `prsm_privacy_status` | Differential privacy budget |
| `prsm_training_status` | NWTN training corpus quality |

---

## For Data Providers

Publish data through your node's ContentStore and earn 80% of every query against it:

```bash
prsm storage upload ./my_dataset.parquet \
  --description "NADA NC Vehicle Registrations 2025" \
  --royalty-rate 0.05 \
  --replicas 5
```

Revenue split: **80% to you**, 15% to compute providers, 5% to PRSM treasury.

---

## For Compute Providers

Check what you'd earn and start providing:

```bash
prsm node benchmark                           # See your hardware tier
prsm ftns yield-estimate --hours 8 --stake 1000   # Monthly earnings estimate
prsm node start                                # Start earning
```

**Staking tiers:**

| Tier | Stake | Yield Boost |
|------|-------|-------------|
| Casual | 0 FTNS | 1.0x |
| Pledged | 100 FTNS | 1.25x |
| Dedicated | 1,000 FTNS | 1.5x |
| Sentinel | 10,000 FTNS | 2.0x + aggregator fees |

---

## Privacy Architecture

PRSM provides three layers of privacy by construction — not by policy:

1. **WASM Zero-Persistence** — The sandbox has no filesystem, no network, no state after execution. The agent literally *cannot* persist data.
2. **Semantic Data Sharding** — No single node holds the full dataset. Each node sees only its assigned shard.
3. **Differential Privacy** — Calibrated Gaussian noise on all intermediate activations (configurable ε: 8.0 standard, 4.0 high, 1.0 maximum).

For proprietary models, **tensor-parallel model sharding** distributes weights across nodes so no single operator can reconstruct the model. **Randomized pipeline assignment** changes the topology per inference. **Collision detection** catches tampering.

See [Confidential Compute Spec](docs/CONFIDENTIAL_COMPUTE_SPEC.md) for details.

---

## Built on Ethereum + Base

PRSM's smart contracts live on **Base**, an Ethereum Layer 2 operated by Coinbase. The FTNS token is a standard ERC-20 at [`0x5276a375...16e5`](https://basescan.org/address/0x5276a3756C85f2E9e46f6D34386167a209aa16e5).

**Why Ethereum:**

| Criterion | Ethereum delivers |
|---|---|
| Credible decentralization | ~1M validators, zero chain halts since PoS launch (Sept 2022) |
| Developer ecosystem | Largest by an order of magnitude; Hardhat, Foundry, OpenZeppelin are Ethereum-native |
| Auditor coverage | Dozens of reputable Solidity audit firms vs. 3-4 on alternatives |
| US regulatory posture | ETH classified as commodity by CFTC; spot and futures ETFs approved |
| Standards + composability | ERC-20 / ERC-721 / ERC-4337 — every wallet, exchange, indexer speaks EVM |

**Why Base (not Ethereum L1 directly, not Solana, not a new chain):**

- **~$0.01 transactions, 2-second blocks** — viable economics for micropayment royalty flow
- **Coinbase as operator** — immediate legitimacy narrative for investor conversations
- **Native USDC from Circle** — zero-friction fiat on-ramp for Phase 5 (Coinbase → Base in seconds)
- **OP Stack foundation** — most battle-tested L2 codebase, portable to any OP Stack chain if Base ever falters
- **Inherits Ethereum L1 security** — fraud proofs and trust-minimized withdrawals, not a trust-in-Coinbase promise

**One-sentence answer for investors asking "why not Solana?":** *"We need Ethereum's decentralization and tooling, we don't need Solana's peak speed for our use case, and we get near-Solana performance via Base L2 without giving up any of Ethereum's other advantages."*

For the full argument (evaluation framework, honest comparison of major chains, responses to common investor pushback), see [Technology Choices](docs/TECH_CHOICES.md).

---

## SDKs

| Language | Install | Docs |
|----------|---------|------|
| **Python** | `pip install prsm-network` | [SDK Guide](docs/SDK_DEVELOPER_GUIDE.md) |
| **JavaScript** | `npm install prsm-sdk` | [sdks/javascript/](sdks/javascript/) |
| **Go** | `go get github.com/Ryno2390/PRSM/sdks/go@v0.37.0` | [sdks/go/](sdks/go/) |

```python
from prsm.sdk import PRSMClient

client = PRSMClient("http://localhost:8000")
result = await client.query("EV trends in NC", budget=10.0, privacy="standard")
quote = await client.quote("EV trends", shards=5, tier="t2")
```

---

## Production Deployment

```bash
# Systemd service
sudo cp deploy/production/prsm-node.service /etc/systemd/system/
sudo cp deploy/production/prsm.env.template /opt/prsm/.env
sudo systemctl enable prsm-node && sudo systemctl start prsm-node

# Docker (2-node demo)
docker-compose -f docker/docker-compose.demo.yml up
```

See [Deployment Guide](deploy/production/DEPLOYMENT_GUIDE.md) for full instructions.

---

## Project Stats

| Metric | Value |
|--------|-------|
| Version | 1.7.0 |
| MCP Tools | 16 |
| SDKs | Python, JavaScript, Go |
| FTNS Token | [Base mainnet](https://basescan.org/address/0x5276a3756C85f2E9e46f6D34386167a209aa16e5) |
| Bootstrap | `wss://bootstrap1.prsm-network.com:8765` |
| License | MIT |

---

## Documentation

**Start here:** [`docs/INDEX.md`](docs/INDEX.md) — navigable map of the full docs surface (89 docs) with audience-oriented entry points (investors / auditors / Foundation officers / research partners / new engineers).

**Commonly-referenced docs:**

| Document | Description |
|----------|------------|
| [Getting Started](docs/GETTING_STARTED.md) | Install → configure → first query in 5 minutes |
| [Executive Summary](docs/2026-04-22-prsm-investor-executive-summary.md) | Investor / partner positioning + shipped-coverage summary |
| [Technology Choices](docs/TECH_CHOICES.md) | Why Ethereum + Base — investor-facing chain-choice rationale |
| [Sovereign-Edge AI Spec](docs/SOVEREIGN_EDGE_AI_SPEC.md) | Phase 1 architecture (Rings 1-6) |
| [Confidential Compute Spec](docs/CONFIDENTIAL_COMPUTE_SPEC.md) | Phase 2 architecture (Rings 7-10) |
| [Audit-Gap Roadmap](docs/2026-04-10-audit-gap-roadmap.md) | Master roadmap — phase status + scope + planning-artifact pointers |
| [Implementation Status](docs/IMPLEMENTATION_STATUS.md) | Subsystem status and test coverage |
| [Deployment Guide](deploy/production/DEPLOYMENT_GUIDE.md) | Production deployment walkthrough |
| [SDK Developer Guide](docs/SDK_DEVELOPER_GUIDE.md) | Building on PRSM |

---

## Contributing

```bash
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM && pip install -e ".[dev]"
pytest --timeout=120    # Run test suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**License:** MIT | **Website:** [prsm-network.com](https://www.prsm-network.com) | **PyPI:** [prsm-network](https://pypi.org/project/prsm-network/)
