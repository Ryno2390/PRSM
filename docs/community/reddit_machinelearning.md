# Reddit — r/MachineLearning Post

## Title
P2P infrastructure protocol that lets any LLM reach distributed compute and data via MCP — v1.6.2 [project]

## Body

Hi r/MachineLearning — sharing PRSM (Protocol for Recursive Scientific Modeling), a **P2P infrastructure protocol** that lets any LLM reach distributed compute, storage, and data through MCP (Model Context Protocol) tools. Flagging upfront: **PRSM is not an AGI framework**. Reasoning happens in the caller's LLM. We just provide the distributed infrastructure that LLM can use.

**What's interesting architecturally:**

Instead of routing your query to a centralized provider, you point your LLM at your local `prsm mcp-server`. The LLM decomposes the query into WASM mobile-agent instructions, PRSM finds the right semantic shards, dispatches the agents to the nodes holding the data, and the agents execute in zero-persistence Wasmtime sandboxes. **The code goes to the data, not the other way around.** Results flow back to the LLM for final synthesis.

Privacy is enforced structurally, not by policy:

1. **WASM zero-persistence** — Wasmtime sandbox has no filesystem, no network, no state after execution. Agents literally *cannot* persist data.
2. **Semantic data sharding** — datasets split by meaning, no single node sees the full dataset.
3. **Differential privacy** — calibrated Gaussian noise on intermediate activations (configurable ε: 8.0 standard → 1.0 maximum).
4. **Tensor-parallel model sharding** — for proprietary models, Ring 8 distributes weights so no single operator can reconstruct the model. Randomized pipelines per inference. Collision detection catches tampering.

The architecture is organized as ten concentric "Sovereign-Edge AI Capability Rings": Wasmtime sandbox (Ring 1), mobile agent dispatch (Ring 2), semantic swarm compute (Ring 3), hybrid FTNS pricing (Ring 4), WASM agent runtime (Ring 5), production hardening (Ring 6), TEE + differential privacy (Ring 7), model sharding (Ring 8), NWTN training pipeline (Ring 9), security audit chain (Ring 10).

Contributors earn FTNS tokens on Base mainnet for sharing their latent storage, compute, and data. New nodes get a 100 FTNS welcome grant. The revenue split is 80% data owner / 15% compute / 5% treasury, settled atomically on a DAG ledger.

**Current state:**

- v1.6.2 shipped on PyPI — `pip install prsm-network`
- All 10 rings shipped and tested
- 16 MCP tools exposed to any compatible LLM
- FTNS live on Base mainnet (`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`)
- Python / JavaScript / Go SDKs all published
- Bootstrap node live at `wss://bootstrap1.prsm-network.com:8765`

Ring 9 currently ships only the NWTN training pipeline (AgentTrace collection → JSONL export → model card). A fine-tuned NWTN LLM for WASM agent generation is future work once we have enough production traces.

```bash
pip install prsm-network
prsm node start
prsm mcp-server    # Point Claude Desktop / any MCP client at this
```

GitHub: https://github.com/prsm-network/PRSM

What I'd value feedback on: the WASM agent SDK ergonomics, the semantic sharding clustering approach (centroid + cosine similarity), and the incentive surface for contributing proprietary data to an otherwise-open network. Happy to go deep on any of the design decisions in the comments.
