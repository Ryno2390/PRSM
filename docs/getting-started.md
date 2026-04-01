# Getting Started with PRSM

This guide takes you from zero to a running PRSM node in about five minutes. By the end you'll have a local node contributing to the decentralized AI network, earning FTNS tokens, and ready to accept compute jobs.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or later (`python3 --version` to check) |
| **pip** | Recent version (`pip install --upgrade pip`) |
| **OS** | macOS, Linux, or Windows (WSL recommended) |
| **IPFS** *(optional)* | For decentralized storage features. Not required to start. |

If you plan to contribute GPU compute, you'll also need CUDA-compatible drivers installed.

---

## Install PRSM

### From PyPI (recommended)

```bash
pip install prsm-network
```

### From source

```bash
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Verify the install:

```bash
prsm --version
```

---

## First Run: The Setup Wizard

Run the interactive setup wizard to configure your node:

```bash
prsm setup
```

The wizard walks you through **7 steps**. Each step has sensible defaults — press Enter to accept them if you're unsure.

### Step 1: Welcome & System Detection

The wizard detects your hardware (CPU cores, RAM, GPU, disk space) and shows what it found. This information is used to suggest resource limits in later steps.

### Step 2: Role Selection

Choose how your node participates in the network:

| Role | Description |
|---|---|
| **Full** *(default)* | Contribute compute and storage, consume services, earn FTNS |
| **Contributor** | Contribute resources only — optimized for headless/server deployments |
| **Consumer** | Use the network without contributing resources |

### Step 3: Resource Allocation

Set how much of your machine PRSM can use:

- **CPU %** — Percentage of CPU cores (default: 50%, range: 10–90%)
- **Memory %** — Percentage of RAM (default: 50%, range: 10–90%)
- **GPU %** — Percentage of GPU memory, if available (default: 80%)
- **Storage** — Disk space in GB for cached data and artifacts (default: 10 GB)
- **Max concurrent jobs** — How many jobs can run at once (default: 3)
- **Bandwidth limit** — Upload speed cap in Mbps (default: unlimited)
- **Active hours/days** — Schedule when your node is active (optional)

### Step 4: API Keys & Wallet

Optionally provide API keys for AI providers and your FTNS wallet:

- **OpenAI API key** — For GPT model inference
- **Anthropic API key** — For Claude model inference
- **HuggingFace token** — For model/dataset access
- **FTNS wallet address** — Your Base mainnet wallet for token earnings

Keys are stored in `~/.prsm/.env`, not in the main config file. You can skip all of these and add them later.

### Step 5: Network Configuration

Configure how your node connects to the P2P network:

- **P2P port** — Port for peer connections (default: 9001)
- **API port** — Port for the local management API (default: 8000)
- **Bootstrap nodes** — Initial peers to connect to. The default bootstrap server (`wss://bootstrap1.prsm-network.com:8765`) is pre-configured.

### Step 6: AI Integration

Enable the MCP (Model Context Protocol) server so AI assistants can use your node's capabilities:

- **MCP server** — Enable/disable (default: enabled)
- **MCP port** — Port for the MCP server (default: 9100)

See the [AI Integration Guide](ai-integration.md) for details on connecting AI clients.

### Step 7: Review & Launch

The wizard displays a summary of all your settings. Confirm to save the configuration to `~/.prsm/config.yaml` and optionally launch the node immediately.

---

## Starting Your Node

After setup, start your node:

```bash
# Interactive mode — shows a live dashboard
prsm start

# Daemon mode — runs in the background
prsm start --daemon
```

The live dashboard displays:
- Your node identity and peer ID
- FTNS balance (new nodes receive a **100 FTNS welcome grant**)
- Connected peers
- Active compute jobs
- Resource utilization

A local management API starts at `http://localhost:8000` (or your configured API port).

---

## Checking Status

### From the CLI

```bash
prsm status
```

This shows your node's current state: identity, role, connected peers, active jobs, FTNS balance, and resource usage.

### From the API

```bash
# Health check
curl http://localhost:8000/health

# Full status
curl http://localhost:8000/status

# FTNS balance
curl http://localhost:8000/balance
```

---

## Adjusting Configuration

View your current config:

```bash
prsm config show
```

Change individual settings:

```bash
prsm config set cpu_pct 70
prsm config set display_name "my-research-node"
prsm config set max_concurrent_jobs 5
```

Read a single value:

```bash
prsm config get cpu_pct
```

See the [Configuration Reference](configuration.md) for all available settings and the full set of `prsm config` commands.

---

## Next Steps

**Explore skills** — PRSM ships with built-in skill packages for datasets, compute, and network operations:

```bash
prsm skills list
prsm skills info prsm-datasets
```

**Connect an AI assistant** — Expose your node's tools to Claude, Hermes, or other MCP-compatible agents:

```bash
prsm mcp start
prsm mcp config-snippet
```

See the [AI Integration Guide](ai-integration.md) for setup instructions.

**Submit a compute job** — Try running a benchmark:

```bash
curl -s -X POST http://localhost:8000/compute/submit \
  -H 'Content-Type: application/json' \
  -d '{"job_type": "benchmark", "payload": {"iterations": 100000}, "ftns_budget": 1.0}'
```

**Read the config reference** — Fine-tune resource limits, scheduling, and network settings in the [Configuration Reference](configuration.md).
