# Reddit — r/ethereum Post

## Title
FTNS token live on Base mainnet — powering a P2P infrastructure network where nodes earn for sharing compute/storage/data

## Body

We've shipped the FTNS (Fungible Tokens for Node Support) contract to **Base mainnet** as part of PRSM — a P2P infrastructure protocol for open-source collaboration. Sharing here because the token economic model avoids a lot of the anti-patterns common in crypto projects, and I'd value feedback from people who think carefully about incentive design.

**Contract address (Base mainnet):** `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`

**How the token economy works:**

FTNS is the native unit of exchange for sharing resources on the PRSM network. Consumer-class nodes — gaming PCs, laptops, phones — contribute latent storage, compute, and sometimes proprietary data, and earn FTNS for doing so. Any third-party LLM (Claude, GPT, local Llama) reaches the network through MCP (Model Context Protocol) tools. When an LLM dispatches a WASM mobile agent to your node, you get paid.

The split is deterministic: **80% to the data owner, 15% to compute providers, 5% to treasury**. Atomic settlement on a DAG ledger with Ed25519 signatures.

**What's different from most token projects:**

FTNS is **minted at contribution time**. No pre-mine, no ICO, no foundation reserve to dump. New nodes receive a 100 FTNS welcome grant on first connect. All additional supply enters circulation as settlement payouts when queries are actually served. If the network does no work, no FTNS is minted.

We run a Chronos fiat bridge (`prsm/compute/chronos/`) that converts FTNS ↔ USD/USDT, so node operators can actually cash out. Protocol governance is fully on-network: node operators vote on upgrades, treasury spending, and parameter updates. There's no institutional tier — nodes are nodes.

**Current state:**

- v1.6.2 shipped on PyPI
- Ring 1-10 Sovereign-Edge AI architecture fully implemented
- Base mainnet FTNS contract live
- Python / JavaScript / Go SDKs published
- Bootstrap node live at `wss://bootstrap1.prsm-network.com:8765`

```bash
pip install prsm-network
prsm node start
# Auto-connects to bootstrap, 100 FTNS welcome grant
prsm mcp-server
# Exposes 16 tools to any MCP-compatible LLM
```

GitHub: https://github.com/Ryno2390/PRSM

If you've thought about contribution-minted tokens, stake-weighted settlement, or Sybil resistance in permissionless node networks — I'd like to hear where the economic model has holes.
