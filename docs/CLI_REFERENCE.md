# PRSM CLI Reference

## Global

```bash
prsm --help          # Show all command groups
prsm --version       # Show version
```

## Node Management

```bash
prsm node start                    # Start with defaults (all 10 rings)
prsm node start --wizard           # Interactive setup wizard
prsm node start --no-dashboard     # Headless mode (for servers)
prsm node start --api-port 8001    # Custom API port
prsm node start --p2p-port 9002    # Custom P2P port
prsm node benchmark                # Hardware profiler (tier, TFLOPS, GPU, thermal, TEE)
prsm setup                         # Interactive configuration wizard
```

## Compute

```bash
# Full forge pipeline (Rings 1-10) — requires FTNS budget
prsm compute run --query "What causes climate change?" --budget 1.0
prsm compute run --query "EV trends in NC" --budget 10.0 --privacy high

# Legacy NWTN path
prsm compute run --prompt "What is 2+2?" --budget 0.01

# Cost estimate (free, no tokens spent)
prsm compute quote "your query"
prsm compute quote "EV trends" --shards 5 --tier t2

# Privacy levels: none, standard (default), high, maximum
```

## FTNS Tokens

```bash
prsm ftns yield-estimate                        # Default estimate
prsm ftns yield-estimate --hours 20 --stake 1000  # Custom estimate
```

## Data Marketplace

```bash
prsm marketplace list-dataset \
  --title "My Dataset" \
  --dataset-id my-data \
  --base-fee 5.0 \
  --per-shard 0.5 \
  --shards 10 \
  --require-stake 100
```

## Agent Operations

```bash
prsm agent forge "Analyze vehicle registration trends in NC"  # Preview decomposition
prsm demo                                                       # Full Ring 1-10 demo
```

## MCP Server

```bash
prsm mcp-server                      # Start MCP server (17 tools)
python scripts/prsm_mcp_server.py    # Clean entry point (no log noise)
```

## Daemon

```bash
prsm daemon start    # Background daemon
prsm daemon stop     # Stop daemon
prsm daemon status   # Check status
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM backend for forge | `sk-or-...` |
| `PRSM_NODE_API_KEY` | API authentication | Any strong secret |
| `PRSM_BOOTSTRAP_NODES` | Custom bootstrap | `ws://host:9001` |
| `PRSM_FAUCET_ENABLED` | Enable/disable faucet | `1` or `0` |
| `FTNS_WALLET_PRIVATE_KEY` | On-chain settlement | Ethereum private key |
| `BASE_RPC_URL` | Base L2 RPC endpoint | `https://mainnet.base.org` |
