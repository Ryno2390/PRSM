# Getting Started with PRSM

## Quick Start (5 minutes)

### 1. Install

```bash
pip install prsm-network
```

Or from source:
```bash
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM
pip install -e .
```

### 2. Configure

**Set up your LLM backend** (needed for the Agent Forge to decompose queries):

```bash
# Option A: OpenRouter (recommended — many free models available)
export OPENROUTER_API_KEY="your-key-here"

# Option B: Store in PRSM config
mkdir -p ~/.prsm
echo 'OPENROUTER_API_KEY=your-key-here' >> ~/.prsm/.env
```

Get a free OpenRouter key at https://openrouter.ai/keys

**Optional: Set API key for node security:**
```bash
export PRSM_NODE_API_KEY="your-secret-key"
```

### 3. Check Your Hardware

```bash
prsm node benchmark
```

This shows your compute tier (T1-T4), GPU detection, TFLOPS, and thermal classification.

### 4. Run Your First Query

**Start your node:**
```bash
prsm node start
```

In another terminal:

**Simple query (direct LLM):**
```bash
prsm compute run --query "What is the capital of France?" --budget 0.01
```

**Get a cost estimate first:**
```bash
prsm compute quote "EV adoption trends" --shards 5 --tier t2
```

**Check yield estimate (what would I earn as a provider?):**
```bash
prsm ftns yield-estimate --hours 8 --stake 1000
```

### 5. Explore the Forge Pipeline

The `--query` flag routes through the full Ring 1-10 forge pipeline:

1. **Decompose** -- LLM analyzes what data and operations are needed
2. **Plan** -- Selects execution route (direct LLM / single agent / swarm)
3. **Quote** -- Calculates cost before execution
4. **Execute** -- Routes to appropriate ring
5. **Trace** -- Collects training data for future NWTN fine-tuning

```bash
# See how a complex query gets decomposed
prsm agent forge "Analyze vehicle registration trends in North Carolina"
```

## Architecture Overview

PRSM is built as 10 concentric "Capability Rings":

| Ring | Name | What It Does |
|------|------|-------------|
| 1 | The Sandbox | WASM runtime + hardware profiling |
| 2 | The Courier | Mobile agent dispatch + settlement |
| 3 | The Swarm | Semantic sharding + parallel execution |
| 4 | The Economy | Hybrid pricing + prosumer staking |
| 5 | The Brain | LLM agent forge + MCP tools |
| 6 | The Polish | Dynamic gas, CLI, signatures |
| 7 | The Vault | TEE + differential privacy |
| 8 | The Shield | Model sharding + collusion resistance |
| 9 | The Mind | NWTN training pipeline |
| 10 | The Fortress | Security audit + integrity |

Each ring wraps and enriches the ones inside it. See `docs/SOVEREIGN_EDGE_AI_SPEC.md` for details.

## For Data Providers

Publish a dataset with pricing:

```bash
prsm marketplace list-dataset \
  --title "My Dataset 2025" \
  --dataset-id my-dataset \
  --base-fee 5.0 \
  --per-shard 0.5 \
  --shards 10 \
  --require-stake 100
```

Revenue split: **80% to you** (data owner), 15% to compute providers, 5% to network.

## For Compute Providers

Check what you'd earn:

```bash
prsm ftns yield-estimate --hours 20 --stake 1000
```

Staking tiers:
- **Casual** (0 FTNS): 1.0x base rate
- **Pledged** (100 FTNS): 1.25x boost
- **Dedicated** (1,000 FTNS): 1.5x boost
- **Sentinel** (10,000 FTNS): 2.0x boost + aggregator eligibility

## Python SDK

```python
from prsm.sdk import PRSMClient

client = PRSMClient("http://localhost:8000", api_key="your-key")

# Full forge pipeline
result = await client.query("EV trends in NC", budget=10.0, privacy="standard")

# Get cost estimate
quote = await client.quote("EV trends", shards=5, tier="t2")

# Upload dataset
await client.upload_dataset("my-data", "My Dataset", content=data_bytes, shard_count=4)
```

## Node Status API

```bash
# Check ring initialization
curl http://localhost:8000/rings/status

# Node health
curl http://localhost:8000/status
```

## Next Steps

- Read the [Sovereign-Edge AI Spec](SOVEREIGN_EDGE_AI_SPEC.md) for architecture details
- Read the [Confidential Compute Spec](CONFIDENTIAL_COMPUTE_SPEC.md) for privacy architecture
- Browse [MCP tools](../prsm/compute/nwtn/agent_forge/mcp_tools.py) for LLM integration
- Check [Implementation Status](IMPLEMENTATION_STATUS.md) for subsystem details
