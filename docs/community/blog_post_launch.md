# Blog Post — Launch Announcement

## We Built a P2P Infrastructure Protocol That Lets Any LLM Reach Distributed Compute, Storage, and Data

The way most people reach AI compute today is broken. You route your query to a centralized provider, your data and outputs pass through infrastructure you don't control, and every token you spend funds a datacenter you'll never visit. Meanwhile, billions of consumer devices — gaming PCs, consoles, laptops, phones — sit idle with storage, compute, and sometimes proprietary data that never leave the device.

We built PRSM to take a different path.

---

### What PRSM Is

PRSM stands for Protocol for Research, Storage, and Modeling, but the short version is this: **it's a P2P infrastructure protocol for open-source collaboration**. Nodes on a mesh network share their latent storage, compute, and data. Contributors earn FTNS tokens for sharing those resources. Any third-party LLM — Claude, GPT, a local Llama, whatever — can reach the network through MCP (Model Context Protocol) tools.

Crucially, **PRSM is not an AGI framework**. Reasoning happens inside the caller's LLM. PRSM doesn't host models. When you ask a question, your LLM decomposes it into WASM mobile-agent instructions, PRSM dispatches those agents to the nodes holding the relevant data, the agents execute in zero-persistence Wasmtime sandboxes, and the results flow back to the LLM for final synthesis. The code goes to the data — not the other way around.

Storage is handled through a native ContentStore with semantic provenance tracking built in. Every piece of content carries metadata about who produced it and what it was derived from. This feeds directly into the royalty pipeline: when a query hits your data, you earn FTNS.

---

### What Actually Works Today

v1.6.2 shipped this week. Here's what's real:

- `pip install prsm-network` works. `prsm node start` connects your node to the bootstrap network at `wss://bootstrap1.prsm-network.com:8765`, and you receive a 100 FTNS welcome grant.
- The Ring 1-10 Sovereign-Edge AI architecture is fully shipped: Wasmtime sandbox, mobile agent dispatch, semantic swarm compute, hybrid pricing, TEE abstraction + differential privacy, tensor-parallel model sharding, and a hash-chained audit log.
- 16 MCP tools are exposed by `prsm mcp-server`, so Claude Desktop (or any MCP-compatible LLM) gains hands (WASM agents) and a wallet (FTNS settlement).
- The **FTNS token is live on Base mainnet** at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`, with the Chronos bridge converting between FTNS and USD/USDT so node operators can cash out.
- Python, JavaScript/TypeScript, and Go SDKs are all published.

This is still early — and we're direct about that. The network has a small number of nodes, the FTNS economy is being calibrated against real usage, and there are rough edges. We're not asking you to run production workloads on PRSM yet. We're asking you to run a node, plug it into your LLM of choice, and tell us what breaks.

---

### What's Coming

Near-term focus is geographic bootstrap resilience (EU + APAC nodes) and a fine-tuned NWTN LLM (Ring 9) trained on real execution traces collected by the network. Ring 9 currently ships only the training pipeline; the fine-tuned model itself is future work. When it arrives, any node will be able to run it locally as an alternative to OpenRouter/Anthropic/OpenAI for WASM agent generation.

Longer term, protocol governance is entirely on-network: node operators vote on upgrades, treasury spending, and parameter updates. No institutional tier. Nodes are nodes.

---

### Get Involved

If you're frustrated with the cost and opacity of centralized AI compute, if you want to point your favorite LLM at a network you control, or if you have idle hardware and want to earn FTNS for sharing it — we want to hear from you.

```bash
pip install prsm-network
prsm node start
```

GitHub: https://github.com/prsm-network/PRSM

The network gets more useful with every node that joins. Come help us build it.
