# PRSM: Sovereign-Edge AI Protocol

**The code goes to the data. Not the other way around.**

PRSM is a decentralized protocol that turns idle consumer hardware — gaming PCs, consoles, laptops, phones — into a distributed AI compute fabric. Instead of uploading your data to a cloud provider, PRSM sends lightweight WASM mobile agents to the nodes that already have the data. The computation happens at the edge, the results come back, and nobody in between sees your query or your data.

**Version 1.0.0** | [PyPI](https://pypi.org/project/prsm-network/) | [Getting Started](docs/GETTING_STARTED.md) | [Architecture Spec](docs/SOVEREIGN_EDGE_AI_SPEC.md)

[![PyPI version](https://badge.fury.io/py/prsm-network.svg)](https://pypi.org/project/prsm-network/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-479%20passing-brightgreen.svg)](#)
[![MCP Tools](https://img.shields.io/badge/MCP%20tools-17-blue.svg)](#mcp-integration)

---

## Why PRSM Exists

**The problem:** Frontier AI is a "brain in a vat" — incredibly smart, but with no hands to touch real-world data securely and no wallet to pay for its own compute. Every query you send to a centralized API is logged, stored, and potentially trained on. Meanwhile, 300 million gaming PCs sit idle 90% of the day.

**The PRSM solution:**

| Centralized AI (today) | PRSM |
|------------------------|------|
| You upload data to their cloud | WASM agents travel to the data |
| They see your query | Zero-persistence sandbox — nothing logged |
| One datacenter processes everything | Thousands of edge nodes work in parallel |
| You pay per token | Hybrid pricing — commodity compute + market-rate data |
| Vendor lock-in | Works with any LLM via MCP |

**The result:** A network where the model is open but the computation is private. You can read every weight, but you can't see what anyone asked or what it answered.

---

## Quick Start

```bash
# Install
pip install prsm-network

# See it work (no config needed)
prsm demo

# Check your hardware
prsm node benchmark

# Start your node
export OPENROUTER_API_KEY=your-key    # Free at openrouter.ai/keys
prsm node start

# Run a query through the full pipeline
prsm compute run --query "What causes climate change?" --budget 1.0
```

> **FTNS tokens required:** PRSM's distributed compute requires FTNS tokens to pay providers. New nodes receive a 100 FTNS welcome grant. Use `prsm compute quote` to estimate costs first.

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
| 5 | **The Brain** | Agent Forge — LLM decomposes queries into WASM agent instructions |
| 6 | **The Polish** | Production hardening — dynamic gas, RPC failover, CLI commands |
| 7 | **The Vault** | Confidential compute — TEE abstraction + differential privacy noise |
| 8 | **The Shield** | Model sharding — tensor parallelism + randomized pipelines + collision detection |
| 9 | **The Mind** | NWTN training pipeline — collect traces, evaluate quality, deploy fine-tuned models |
| 10 | **The Fortress** | Security — integrity verification, privacy budgets, hash-chained audit logs |

### End-to-End Flow

```
Researcher: prsm compute run --query "EV adoption trends in NC" --budget 10.0

  → Ring 5:  AgentForge decomposes query via LLM
  → Ring 3:  Finds relevant semantic shards by embedding similarity
  → Ring 4:  Quotes cost: compute + data + network fee
  → Ring 3:  Fans out parallel agents to shard-holding nodes
  → Ring 2:  Each agent dispatched via gossip bidding
  → Ring 1:  Executed in WASM sandbox on provider hardware
  → Ring 7:  Differential privacy noise applied
  → Ring 3:  Results aggregated when quorum met
  → Ring 4:  FTNS settled: 80% data owner / 15% compute / 5% treasury
  → Ring 9:  AgentTrace saved for NWTN fine-tuning

  ← Result returned to researcher
```

---

## MCP Integration

Any LLM can use PRSM as a compute backend via the Model Context Protocol. **17 tools** are exposed:

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
| `prsm_decompose` | Preview how a query would be decomposed |
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

Publish a dataset and earn 80% of every query against it:

```bash
prsm marketplace list-dataset \
  --title "NADA NC Vehicle Registrations 2025" \
  --dataset-id nada-nc-2025 \
  --base-fee 5.0 \
  --per-shard 0.5 \
  --shards 12 \
  --require-stake 100
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
| Version | 1.0.0 |
| Tests | 479 passing |
| MCP Tools | 17 |
| SDKs | Python, JavaScript, Go |
| FTNS Token | [Base mainnet](https://basescan.org/address/0x5276a3756C85f2E9e46f6D34386167a209aa16e5) |
| Bootstrap | `wss://bootstrap1.prsm-network.com:8765` |
| License | MIT |

---

## Documentation

| Document | Description |
|----------|------------|
| [Getting Started](docs/GETTING_STARTED.md) | Install → configure → first query in 5 minutes |
| [Sovereign-Edge AI Spec](docs/SOVEREIGN_EDGE_AI_SPEC.md) | Phase 1 architecture (Rings 1-6) |
| [Confidential Compute Spec](docs/CONFIDENTIAL_COMPUTE_SPEC.md) | Phase 2 architecture (Rings 7-10) |
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
