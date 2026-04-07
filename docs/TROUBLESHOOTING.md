# PRSM Troubleshooting Guide

## Installation Issues

### `pip install prsm-network` fails
- **Python version:** Requires Python 3.11+. Check with `python3 --version`.
- **Permissions:** Use `pip install --user prsm-network` or a virtual environment.
- **WASM support:** For WASM sandbox, use `pip install prsm-network[wasm]`.

### `prsm: command not found`
- The `prsm` CLI is installed with the package. Ensure your Python scripts directory is in PATH.
- Try: `python -m prsm.cli --help`
- Or: `pip show prsm-network` to find the install location.

## Node Startup Issues

### "DEGRADED local mode"
**What it means:** The bootstrap server is unreachable. Your node works fine for local compute but can't discover peers.

**Fixes:**
1. Check internet connection
2. The node retries every 60 seconds — it may connect shortly
3. For local-only use, this is fine — `prsm compute run` works in single-node mode
4. Set custom bootstrap: `export PRSM_BOOTSTRAP_NODES=ws://your-server:9001`

### "Agent forge not initialized"
**What it means:** No LLM backend is configured, so the forge can't decompose queries.

**Fix:**
```bash
# Get a free API key at https://openrouter.ai/keys
export OPENROUTER_API_KEY=your-key-here
prsm node start  # Restart with key configured
```

### Port already in use
**Fix:**
```bash
prsm node start --api-port 8001 --p2p-port 9002
```

## Query Issues

### "FTNS budget is required"
All forge pipeline queries cost FTNS tokens. You cannot set budget to 0.

**Fix:**
```bash
# Check your balance
curl http://localhost:8000/balance

# Estimate cost first (free)
prsm compute quote "your query" --shards 3

# Then run with budget
prsm compute run --query "your query" --budget 1.0

# Need more FTNS? Use the faucet (dev/testnet):
curl -X POST http://localhost:8000/ftns/faucet \
  -H "Content-Type: application/json" \
  -d '{"amount": 100}'
```

### Query returns empty or default result
**Cause:** The LLM decomposition failed or returned defaults (no datasets, route=direct_llm).

**Fixes:**
1. Ensure `OPENROUTER_API_KEY` is set and valid
2. Check that the free model isn't rate-limited (try again after 30 seconds)
3. Try a simpler query first: `prsm compute run --query "What is 2+2?" --budget 0.01`

### "wasmtime not installed"
**Fix:** `pip install wasmtime` or `pip install prsm-network[wasm]`

## MCP Server Issues

### LLM can't connect to MCP server
1. Start the server: `prsm mcp-server` or `python scripts/prsm_mcp_server.py`
2. For Claude Desktop, configure `~/.claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "prsm": {
      "command": "python",
      "args": ["/path/to/PRSM/scripts/prsm_mcp_server.py"]
    }
  }
}
```
3. Use absolute paths in the config

### MCP tools return "node not running"
Some tools (`prsm_analyze`, `prsm_upload_dataset`) need a running PRSM node.
Start it in another terminal: `prsm node start`

Local tools that work without a node: `prsm_quote`, `prsm_hardware_benchmark`, `prsm_yield_estimate`, `prsm_stake`, `prsm_revenue_split`, `prsm_decompose`.

## FTNS Token Issues

### Balance is 0
New nodes receive a 100 FTNS welcome grant on first start. If you've spent it all:
```bash
# Faucet (dev/testnet only)
curl -X POST http://localhost:8000/ftns/faucet \
  -H "Content-Type: application/json" \
  -d '{"amount": 100}'
```

### Settlement pending
On-chain FTNS settlement batches periodically. Check queue:
```bash
curl http://localhost:8000/settlement/stats
```

## Hardware Detection

### GPU not detected
- **NVIDIA:** Ensure `nvidia-smi` is in PATH
- **Apple Silicon:** Detected automatically via Metal framework
- **Check:** `prsm node benchmark`

### Wrong compute tier
Tiers are based on TFLOPS:
- T1: < 5 TFLOPS (mobile/IoT)
- T2: 5-30 TFLOPS (consumer)
- T3: 30-80 TFLOPS (high-end)
- T4: 80+ TFLOPS (datacenter)

Run `prsm node benchmark` to see your classification.

## Getting Help

```bash
prsm --help              # All commands
prsm node start --help   # Node options
prsm compute run --help  # Compute options
```

Check ring status: `curl http://localhost:8000/rings/status`
