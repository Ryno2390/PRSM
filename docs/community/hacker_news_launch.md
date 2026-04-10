# Hacker News — Show HN Post

## Title
Show HN: PRSM – P2P infrastructure protocol that lets any LLM reach distributed compute

## Body

We built PRSM (Protocol for Recursive Scientific Modeling) because we wanted a way to point any LLM — Claude, GPT, local Llama — at distributed compute, storage, and proprietary data without routing everything through a centralized API. Today we're opening it up.

The short pitch: PRSM is a P2P infrastructure protocol where consumer nodes share their latent storage, compute, and data. Contributors earn FTNS tokens. Any third-party LLM can reach the network through MCP tools. **Reasoning always happens in the caller's LLM** — PRSM doesn't host models. When your LLM calls `prsm_analyze`, it decomposes the query into WASM mobile-agent instructions; PRSM dispatches those agents to the nodes holding the relevant data, the agents execute in zero-persistence Wasmtime sandboxes, and the results flow back. The code goes to the data.

Architecture is organized as ten "Sovereign-Edge AI Capability Rings" — WASM sandbox (Wasmtime), mobile agent dispatch, semantic swarm compute, hybrid FTNS pricing, TEE abstraction with differential privacy, tensor-parallel model sharding, and a hash-chained audit log. FTNS is live on Base mainnet (`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`) with a Chronos bridge to USD/USDT.

Getting started takes about 30 seconds:

```bash
pip install prsm-network
prsm node start
# Auto-connects to wss://bootstrap1.prsm-network.com:8765
# You receive a 100 FTNS welcome grant
# REST API live at localhost:8000

prsm mcp-server
# Exposes 16 tools to any MCP-compatible LLM
```

Real talk on state: this is v1.6.2. Rings 1-10 are shipped and tested. A fine-tuned NWTN LLM for WASM agent generation (Ring 9) is future work — only the training pipeline ships today. Node bootstrap is single-region (NYC3) with EU/APAC deployment docs ready. The network has a small number of nodes. What we want right now is node operators, early integrators, and people who will break things and tell us about it.

GitHub: https://github.com/Ryno2390/PRSM

Happy to answer questions about the ring architecture, the MCP tool surface, the WASM agent SDK, or the 80/15/5 revenue split (80% to data owner, 15% to compute, 5% to treasury).
