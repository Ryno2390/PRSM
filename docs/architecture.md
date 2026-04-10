# PRSM Architecture

PRSM is a **P2P infrastructure protocol for open-source collaboration**. It aggregates the latent storage, compute, and data of consumer-class nodes — gaming PCs, consoles, laptops, phones — into a mesh network that any third-party LLM can reach through MCP (Model Context Protocol) tools. Contributors earn FTNS tokens for sharing resources; users leverage PRSM through the LLM of their choice.

PRSM is **not** an AGI framework, a teacher-model or distillation platform, a coordinated multi-agent system, or a hosted-model marketplace. Reasoning happens inside third-party LLMs. PRSM provides the infrastructure those LLMs use to reach distributed data, dispatch sandboxed compute, and settle payments.

---

## The 10-Ring Sovereign-Edge AI Architecture

PRSM is built as ten concentric capability rings. Each ring wraps and enriches the rings inside it.

| Ring | Name | Responsibility | Key Modules |
|------|------|---------------|-------------|
| 1 | **The Sandbox** | WASM runtime (Wasmtime) with memory/fuel limits, hardware profiling | `prsm/compute/wasm/` |
| 2 | **The Courier** | Mobile agent dispatch via gossip-based bidding + escrow settlement | `prsm/compute/agents/` |
| 3 | **The Swarm** | Semantic sharding, parallel map-reduce, quorum-based aggregation | `prsm/compute/swarm/` |
| 4 | **The Economy** | Hybrid pricing (PCU menu + market rates), prosumer staking, 80/15/5 splits | `prsm/economy/tokenomics/` |
| 5 | **Agent Forge** | WASM mobile agent runtime — query decomposition is done by the caller's third-party LLM | `prsm/compute/agents/` |
| 6 | **The Polish** | Dynamic gas pricing, RPC failover, settler signature verification, CLI UX | — |
| 7 | **The Vault** | Confidential compute — TEE runtime abstraction + differential privacy noise | `prsm/compute/tee/` |
| 8 | **The Shield** | Tensor-parallel model sharding, randomized pipelines, collusion detection | `prsm/compute/model_sharding/` |
| 9 | **The Mind** | NWTN training pipeline — collects execution traces to fine-tune a future NWTN LLM | `prsm/compute/nwtn/training/` |
| 10 | **The Fortress** | Integrity verification, privacy budget tracking, hash-chained audit logs | `prsm/compute/security/` |

See [`docs/SOVEREIGN_EDGE_AI_SPEC.md`](SOVEREIGN_EDGE_AI_SPEC.md) for the original Rings 1-6 design and [`docs/CONFIDENTIAL_COMPUTE_SPEC.md`](CONFIDENTIAL_COMPUTE_SPEC.md) for Rings 7-10.

---

## Query Flow

```
User → Third-party LLM (Claude / GPT / local)
       │
       │ The LLM calls prsm_analyze via MCP
       ▼
Ring 5:  LLM decomposes the query into WASM mobile-agent instructions
Ring 3:  Finds relevant semantic shards by embedding similarity
Ring 4:  Quotes cost: compute + data + network fee
Ring 3:  Fans out parallel agents to shard-holding nodes
Ring 2:  Each agent dispatched via gossip bidding
Ring 1:  Executed in zero-persistence WASM sandbox on provider hardware
Ring 7:  Differential privacy noise applied to intermediate activations
Ring 3:  Results aggregated when quorum is met
Ring 4:  FTNS settled — 80% to data owner, 15% to compute, 5% to treasury
       │
       │ Result returned to the LLM for final synthesis
       ▼
Third-party LLM → User
```

Reasoning always happens inside the caller's LLM. PRSM never hosts the model. All execution on PRSM nodes runs inside WASM sandboxes with no filesystem, no network, and no state after return.

---

## Privacy by Construction

Privacy is enforced structurally — not by policy.

1. **WASM zero-persistence** — The Wasmtime sandbox has no filesystem, no network, no state after execution. An agent literally *cannot* persist data.
2. **Semantic data sharding** — Datasets are split by meaning across nodes; no single node holds the full dataset.
3. **Differential privacy** — Calibrated Gaussian noise is injected into intermediate activations (configurable ε: 8.0 standard, 4.0 high, 1.0 maximum).
4. **Model sharding** — For proprietary models, Ring 8 distributes weights via tensor parallelism across nodes so no single operator can reconstruct the model. Randomized pipeline assignment changes the topology per inference. Collision detection catches tampering.

---

## FTNS Economics

FTNS (Fungible Tokens for Node Support) is the protocol's economic primitive. It is **minted at contribution time** — there is no pre-mine, no ICO, and no foundation reserve to dump. New nodes receive a 100 FTNS welcome grant.

- **Data providers** earn 80% of each query against their content (Ring 4 `ContentStore` + royalty pipeline).
- **Compute providers** earn 15% of each query they execute (Ring 1 sandbox + Ring 2 dispatch).
- **Treasury** receives 5% — used exclusively for protocol upkeep under on-chain governance.

FTNS lives on Base mainnet at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`. The Chronos bridge (`prsm/compute/chronos/`) converts between FTNS and USD/USDT so node operators can cash out.

---

## Governance

Protocol evolution is governed **on-network** by node operators. Stake-weighted voting with term-decay protections determines protocol changes, treasury spending, and parameter updates. There is no institutional tier — every node is a node.

---

## Where Third-Party LLMs Fit

PRSM exposes MCP (Model Context Protocol) tools that any compatible LLM can call:

| Tool | Purpose |
|------|---------|
| `prsm_analyze` | Full Ring 1-10 pipeline — query in, answer out |
| `prsm_quote` | Cost estimate before committing (free) |
| `prsm_create_agent` / `prsm_dispatch_agent` | Build and execute a custom WASM agent |
| `prsm_upload_dataset` / `prsm_list_datasets` / `prsm_search_shards` | Data publishing and discovery |
| `prsm_yield_estimate` / `prsm_stake` / `prsm_revenue_split` | Economic queries |
| `prsm_hardware_benchmark` / `prsm_node_status` | Node operations |

All 16 MCP tools are exposed by `prsm/mcp_server.py`. Configure any MCP-compatible LLM (Claude Desktop, OpenClaw, etc.) to point at your local `prsm mcp-server` and the LLM gains hands (WASM agents) and a wallet (FTNS settlement).

---

## Future Work: Ring 9 NWTN

Ring 9 currently ships the **NWTN training pipeline** — the infrastructure that collects `AgentTrace` records from real query executions, evaluates trace quality, and exports a JSONL training corpus. The eventual goal is a fine-tuned NWTN LLM specialized for WASM agent generation, which any node can then run locally as an alternative to OpenRouter/Anthropic/OpenAI. The fine-tuned model itself does not exist yet; only the training pipeline is shipped.
