# PRSM Sovereign-Edge AI Architecture Spec

**Version:** 1.0
**Date:** 2026-04-06
**Status:** Approved — ready for implementation planning

---

## Executive Summary

PRSM pivots from cloud-first AI to **sovereign-edge AI** — a distributed computing fabric where idle consumer GPUs (gaming PCs, consoles, AI laptops) provide the compute, WASM mobile agents carry the logic to the data instead of moving data to the cloud, and FTNS tokens create a self-sustaining prosumer economy.

The architecture is organized into six **Capability Rings**, each independently valuable and demoable. Each ring wraps and enriches the rings inside it.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Phasing strategy | Interleave original Phase 3/5/7 gaps into new rings | Existing work is foundational, not wasted |
| MVP milestone | Mobile Agent end-to-end (code-to-data) | Atomic building block everything else depends on |
| WASM runtime | Runtime-agnostic interface, Wasmtime default | Best sandbox security; abstraction avoids lock-in |
| NWTN fine-tune | Late-stage goal; frontier models generate agents for now | Need real execution traces before training makes sense |
| Quality bar | Tiered — production-grade for security/settlement, prototype for intelligence layers | Can't half-bake sandboxing or payments; intelligence evolves iteratively |

---

## Ring 1 — "The Sandbox": WASM Runtime + Hardware Profiling

**Goal:** A PRSM node can safely execute an untrusted WASM module and accurately report its hardware capabilities to the network.

### 1.1 — WASM Runtime Abstraction

**New module:** `prsm/compute/wasm/`

A `WASMRuntime` interface with Wasmtime as the default implementation:

- `load(wasm_bytes) → Module` — validate and compile a WASM binary
- `execute(module, input_data, resource_limits) → ExecutionResult` — run in sandbox with enforced caps
- `capabilities() → RuntimeInfo` — what WASI features this runtime supports

**Resource limits** (enforced by the runtime, not by trust):

| Resource | Default | Configurable |
|----------|---------|-------------|
| Max memory | 256 MB | Per job |
| Max execution time | 30 seconds | Per job |
| Max output size | 10 MB | Per job |
| Filesystem access | Virtual `/data` mount only | No |
| Network access | None | No |

The WASM module sees only what the host explicitly provides — the input data shard, a write buffer for output, and nothing else. This is WASI's capability-based security model.

### 1.2 — Hardware Profiler

**Extend:** `prsm/node/capability_detection.py`

Produces a `HardwareProfile` with:

- **Compute:** TFLOPS estimate (FP32 and FP16), CUDA/Metal/ROCm capability, core count
- **Memory:** Total VRAM, total RAM, available at time of measurement
- **Storage:** Available disk, I/O throughput estimate
- **Network:** Upload/download bandwidth (measured via bootstrap handshake)
- **Thermal:** Headroom classification — SUSTAINED, BURST, or THROTTLED

Gossiped via `GOSSIP_HARDWARE_PROFILE` (24-hour retention).

**Compute tiers:**

| Tier | TFLOPS Range | Typical Hardware | Use Case |
|------|-------------|------------------|----------|
| T1 | < 5 | Mobile, IoT, old laptops | Symbolic filtering |
| T2 | 5–30 | Consoles, mid-range GPUs | Data analysis, embeddings |
| T3 | 30–80 | High-end desktops, M-series | Complex reasoning, large inference |
| T4 | 80+ | Datacenter GPUs, multi-GPU | Training, fine-tuning, frontier inference |

### 1.3 — Integration

- Existing `compute_provider.py` gets a new code path: `.wasm` payloads route to `WASMRuntime.execute()` instead of subprocess
- Hardware profile included in gossip heartbeats
- `agent_registry` learns to filter by hardware tier for node discovery

### 1.4 — Deliverable

```
prsm node start                                        # reports hardware profile
prsm node benchmark                                    # shows tier classification
prsm compute run --wasm agent.wasm --input data.json   # local sandbox execution
```

---

## Ring 2 — "The Courier": Mobile Agent Dispatch + Settlement

**Goal:** A WASM agent travels to a remote node holding the data it needs, executes in the Ring 1 sandbox, and triggers FTNS payment on completion. The core "code-to-data" shift.

### 2.1 — Mobile Agent Definition

**New module:** `prsm/compute/agents/`

```
MobileAgent {
    agent_id:       UUID
    wasm_binary:    bytes           # Compiled WASM (typically 10–500 KB)
    manifest:       AgentManifest   # Data needs, hardware reqs, output schema
    origin_node:    str
    signature:      bytes           # Ed25519 from origin
    ftns_budget:    Decimal
    ttl:            int             # Max seconds before expiry
}
```

`AgentManifest` describes:
- Required data CIDs
- Minimum hardware tier and capabilities
- Output schema (JSON schema for validation)
- Execution constraints (memory, time, output size)

### 2.2 — Dispatch Protocol

Three new gossip message types:

| Message | Purpose | Payload |
|---------|---------|---------|
| `GOSSIP_AGENT_DISPATCH` | Origin announces job need | Manifest (no binary) |
| `GOSSIP_AGENT_ACCEPT` | Qualifying node bids | Price, node profile, shard proof |
| `GOSSIP_AGENT_RESULT` | Executor returns output | Signed result hash, metrics |

WASM binary transfers via existing WebSocket direct-message channel (not gossip).

### 2.3 — Execution Flow

```
 1. Researcher submits query via CLI or MCP
 2. NWTN determines required data shard
 3. MobileAgent created with WASM logic + manifest
 4. FTNS escrow locked (existing Phase 1 escrow)
 5. GOSSIP_AGENT_DISPATCH broadcasts manifest
 6. Nodes holding shard + meeting hardware reqs bid
 7. Origin selects best bid (price × reputation × latency)
 8. WASM binary transferred via direct WebSocket
 9. Executor validates: signature, manifest hash, shard availability
10. Ring 1 sandbox executes with shard mounted at /data
11. Signed result returned via GOSSIP_AGENT_RESULT
12. Origin verifies signature, releases escrow to provider
13. Result returned to researcher
```

### 2.4 — Fault Handling

| Scenario | Response |
|----------|----------|
| Agent TTL expires | Escrow refunded, automatic retry (max 3, exponential backoff) |
| Node disappears mid-execution | 2 missed heartbeats (60s) → mark failed, redispatch to next-best bidder |
| Malicious/invalid output | Reject if output doesn't conform to manifest schema, redispatch |
| WASM binary too large | Dispatch rejected at configurable limit (default 5 MB) |

### 2.5 — Settlement

Reuses existing infrastructure — no new on-chain contracts:
- `PaymentEscrow.create_escrow()` at step 4
- `PaymentEscrow.release_escrow()` at step 12
- `BatchSettlementManager` for gas-efficient on-chain confirmation
- Phase 6 settler staking validates high-value batch settlements

### 2.6 — CLI & MCP

**CLI:**
```
prsm agent dispatch --wasm agent.wasm --manifest manifest.json --budget 5.0
prsm agent status <agent-id>
prsm agent results <agent-id>
```

**MCP tool:** `prsm_dispatch_agent` — any LLM can dispatch agents without knowing about the P2P network.

### 2.7 — Deliverable

Full "code-to-data" demo: Node A dispatches a WASM agent to Node B, agent executes on B's local data, result returned, FTNS paid.

---

## Ring 3 — "The Swarm": Semantic Sharding + Parallel Map-Reduce

**Goal:** Large datasets are sharded by meaning across the network. A single query triggers parallel agent execution across the entire swarm — the "100 chefs cooking one dish each" model.

### 3.1 — Semantic Vector Sharding

**New module:** `prsm/data/semantic_shard.py`

**Extend:** `prsm/core/ipfs_sharding.py` (currently byte-offset only)

Upload process:
1. Parse data into records (format-aware: CSV rows, JSON objects, document paragraphs)
2. Embed each record (existing embedding infrastructure or lightweight local model)
3. Cluster by meaning (k-means / locality-sensitive hashing) into **neighborhoods**
4. Each neighborhood → shard → IPFS pin with `ShardManifest`:

```
ShardManifest {
    shard_id:       str
    parent_dataset: str
    centroid:       float[]    # Embedding centroid of this neighborhood
    record_count:   int
    size_bytes:     int
    cid:            str        # IPFS content address
    keywords:       str[]      # Human-readable topic tags
}
```

5. Manifests registered in content index, gossiped via `GOSSIP_CONTENT_ADVERTISE`

**Query-time lookup:** NWTN embeds the query, finds nearest shard centroids via cosine similarity against the existing vector store.

**Integration:** IPFS sharding handles byte-level storage. BitTorrent handles replication. Content economy tracks replication status. Vector store indexes centroids.

### 3.2 — Swarm Job Decomposition

**New module:** `prsm/compute/swarm/`

```
SwarmJob {
    job_id:         UUID
    query:          str
    agent_template: MobileAgent      # Same WASM for all shards
    target_shards:  ShardManifest[]
    reduce_agent:   MobileAgent      # Aggregation logic (optional)
    budget_ftns:    Decimal
    strategy:       MapReduceStrategy
}
```

`MapReduceStrategy` defines:
- Budget split (equal or weighted by shard size)
- Reduce location (origin node, staked aggregator, or hierarchical)
- Quorum (minimum shard completion %, default 80%)
- Timeouts (per-shard and global)

### 3.3 — Parallel Dispatch

Fan-out of Ring 2 dispatch:
1. NWTN embeds query → finds relevant shards (e.g., 50 shards across 30 nodes)
2. SwarmJob created with agent template + shard list
3. Per-shard MobileAgent clones dispatched simultaneously
4. Nodes bid on shards they hold
5. All agents execute in parallel
6. Mini-results (10 KB summaries, not raw data) stream back

### 3.4 — Aggregator Nodes

**Extend:** `prsm/node/settler_registry.py`

Phase 6 staked settlers double as **regional aggregators**:

1. Map agents send mini-results to designated aggregator (by proximity + stake)
2. Aggregator runs `reduce_agent` WASM on collected mini-results
3. Aggregator produces final result, signs it, submits batch settlement

**Hierarchical aggregation** (1000+ shard jobs):
- Nodes grouped by network proximity
- Regional micro-aggregators compress local results
- Global aggregator synthesizes — only compressed proofs travel long-distance

### 3.5 — Result Consensus

**Extend:** Existing `ResultConsensus` module (present but not fully integrated)

| Job Value | Consensus Model |
|-----------|----------------|
| < 1 FTNS | Single-node execution (Ring 2) |
| 1–10 FTNS | 2-of-3 consensus |
| > 10 FTNS | 3-of-5 consensus |

Disagreements resolved by staked arbitrator (settler) re-execution.

### 3.6 — Edge Caching

**New module:** `prsm/data/edge_cache.py`

The "Netflix model" from the design notes:
- Off-peak hours (default 1–5 AM local): nodes pre-fetch popular shards via BitTorrent
- "Popular" defined by access frequency in content economy
- Only caches shards matching storage pledge and available disk
- Nodes earn FTNS for caching (storage provider reward loop)

### 3.7 — Deliverable

```
prsm storage upload --file nada_nc_2025.parquet --semantic-shard --replicas 3
prsm compute run --query "EV adoption trends in NC 2025 vs 2026" --budget 10.0
# → 12 agents dispatched across 8 nodes, aggregated result in seconds
```

---

## Ring 4 — "The Economy": Hybrid Pricing + Data Marketplace + Prosumer Staking

**Goal:** Self-sustaining network economics. Deterministic compute pricing, value-based data pricing, and prosumer incentives that make "PS5 mining FTNS while you're at work" real.

### 4.1 — Standardized Compute Pricing (The Menu)

**New module:** `prsm/economy/pricing/`

**Unit:** PRSM Compute Unit (PCU) — normalized metric combining:
- TFLOPS consumed × seconds
- Memory allocated (GB-seconds)
- Network egress (MB transferred)

Metered precisely by Ring 1 hardware profiler and sandbox resource reporting.

**Tier pricing:**

| Tier | PCU Rate (FTNS) | Typical Hardware |
|------|----------------|------------------|
| T1 | 0.001 | Mobile, IoT, old laptops |
| T2 | 0.005 | Consoles, mid-range GPUs |
| T3 | 0.02 | High-end desktops, M-series |
| T4 | 0.10 | Datacenter GPUs |

**Spot pricing:** Rates decrease up to 50% when utilization < 40%, increase up to 25% when utilization > 80%. Automatic supply-demand curve, no manual governance.

### 4.2 — Data Marketplace (The Market)

**Extend:** `prsm/economy/marketplace/`

```
DataListing {
    dataset_id:         UUID
    owner_id:           str
    title:              str
    shard_manifests:    ShardManifest[]

    # Pricing (owner-controlled)
    base_access_fee:    Decimal       # FTNS per query
    per_shard_fee:      Decimal
    bulk_discount:      float         # % discount for 10+ shards
    exclusive_premium:  Decimal       # Premium for early/exclusive access

    # Access control
    requires_stake:     Decimal       # Anti-scraping stake requirement
    max_queries_per_day: int
    allowed_operations: str[]         # "aggregate", "filter", "export"
}
```

**Anti-scraping:** Accessors stake FTNS. If agent output exceeds expected schema size (data exfiltration attempt), stake is slashed. WASM sandbox makes this enforceable.

**Exclusive access auctions:** Time-limited listings where researchers bid for early access to premium data.

### 4.3 — NWTN as Pricing Broker

**Extend:** `prsm/compute/nwtn/orchestrator.py`

Before execution, NWTN produces a cost quote:

```
Quote {
    compute_cost:   Decimal     # PCU rate × estimated PCUs
    data_cost:      Decimal     # Dataset access fees
    network_fee:    Decimal     # 5% of total
    total:          Decimal
    breakdown:      [...]       # Per-dataset, per-shard detail
    confidence:     float       # Estimate confidence (0–1)
    alternatives:   [...]       # Cheaper options
}
```

Researcher approves before FTNS escrow locks. Actual cost settled post-execution — under-budget remainder refunded.

**Arbitrage:** When job queue is high and acceptance rate low, NWTN triggers spot pricing automatically.

### 4.4 — Prosumer Staking & Yields

**New module:** `prsm/economy/prosumer.py`

| Tier | Stake | Commitment | Yield Boost |
|------|-------|------------|-------------|
| Casual | 0 FTNS | None | 1.0× |
| Pledged | 100 FTNS | 8+ hrs/day | 1.25× |
| Dedicated | 1,000 FTNS | 20+ hrs/day, 95% uptime | 1.5× |
| Sentinel | 10,000 FTNS | Always-on, aggregator-eligible | 2.0× + aggregator fees |

**Slashing:**
- Availability pledge not met (7-day rolling window): 5% of stake
- Job abandoned mid-execution: 10% per incident
- Fabricated results (consensus detection): 100% of stake

### 4.5 — Revenue Distribution

```
Total Payment
├── Data fees      → Data Owner (per listing price)
├── Compute fees   → Provider(s) (per PCU metering)
└── Network fee    → PRSM Treasury (5% of total)
```

Handled by existing Phase 4 multi-party escrow and Phase 6 settler validation.

### 4.6 — Deliverable

```
prsm marketplace list-dataset --file nada.parquet --base-fee 5.0
prsm ftns yield-estimate           # "Estimated monthly: 72 FTNS"
prsm ftns stake 1000 --tier dedicated
prsm compute quote --query "EV trends in NC"
# → Compute: 0.50 | Data: 5.60 | Network: 0.31 | Total: 6.41 FTNS
```

---

## Ring 5 — "The Brain": NWTN as Agent Architect

**Goal:** NWTN evolves into an agent architect that turns natural language queries into optimized WASM mobile agents. Frontier models do the generation now; the specialized fine-tune comes later from real execution traces.

### 5.1 — Prompt-to-Manifest Pipeline

**New module:** `prsm/compute/nwtn/agent_forge/`

Four-stage pipeline:

**Stage 1 — Intent Decomposition** (extends existing orchestrator):
- What data sources needed? (shard manifest lookup)
- What operations? (filter, aggregate, join, stats, similarity)
- Parallelizable? (single-shard vs. swarm)
- Minimum hardware tier?

Output: `TaskDecomposition`

**Stage 2 — Hardware-Aware Planning:**
Match decomposition against available hardware profiles and shard locations.

```
TaskPlan {
    decomposition:    TaskDecomposition
    target_nodes:     NodeMatch[]
    execution_model:  SINGLE | SWARM
    estimated_pcu:    float
    thermal_class:    SUSTAINED | BURST
}
```

Thermal classification prevents dispatching long jobs to devices that will throttle.

**Stage 3 — Agent Code Generation:**
LLM receives TaskPlan + WASM Agent SDK templates + target hardware profile + shard schema. Outputs:

```
AgentSource {
    language:         RUST | ASSEMBLYSCRIPT
    source_code:      str
    entry_point:      str
    dependencies:     str[]          # SDK templates used
    estimated_memory: int
}
```

Uses frontier models via existing backend registry (OpenRouter, Anthropic, local). Fine-tuned NWTN replaces this in a future phase.

**Stage 4 — Compilation & Validation:**
1. Compile: Rust → `wasm32-wasi` or AssemblyScript → WASM
2. Static analysis: scan for disallowed WASI imports
3. Sandbox dry-run: execute against synthetic test shard
4. Sign: origin node signs validated binary

Retry up to 3 times on failure — LLM receives error + failing code.

### 5.2 — WASM Agent SDK

**New module:** `prsm/compute/wasm/sdk/`

Pre-compiled helper functions agents can import:

| Module | Operations |
|--------|-----------|
| `prsm_io` | Read input shard (JSON, CSV, Parquet), write output |
| `prsm_filter` | Field value, range, regex filtering |
| `prsm_aggregate` | Sum, count, avg, min, max, percentile, group-by |
| `prsm_stats` | StdDev, correlation, regression, time-series |
| `prsm_embed` | Cosine similarity against pre-computed vectors |
| `prsm_crypto` | Hash output for proof-of-execution, sign results |

Versioned. Nodes cache last 3 SDK versions.

### 5.3 — Training Data Collection

Every successful agent generation is logged as an `AgentTrace`:

```
AgentTrace {
    query:               str
    task_decomposition:  JSON
    task_plan:           JSON
    generated_source:    str
    compilation_success: bool
    validation_success:  bool
    execution_result:    JSON
    execution_metrics:   JSON
    hardware_profile:    JSON
    user_satisfaction:   float    # Optional
}
```

Stored locally, optionally contributed to shared training corpus (opt-in, anonymized). Becomes the fine-tune training set when volume is sufficient.

### 5.4 — MCP Integration ("MCP Everywhere")

Full pipeline exposed as MCP tools for any LLM:

| Tool | Purpose |
|------|---------|
| `prsm_analyze` | Submit query → get results (wraps entire pipeline) |
| `prsm_quote` | Cost estimate before committing |
| `prsm_list_datasets` | Browse datasets and pricing |
| `prsm_dispatch_agent` | Low-level: dispatch pre-built WASM agent |
| `prsm_swarm_status` | Check running swarm job status |

**Key insight:** PRSM doesn't compete with Claude or Gemini — it gives them hands (WASM agents) and a wallet (FTNS settlement).

### 5.5 — Graceful Degradation

NWTN routes to the lightest path:

| Query Type | Route |
|------------|-------|
| Simple factual | Existing NWTN reasoning, no agents |
| Local data | Ring 1 local sandbox |
| Network data, single shard | Ring 2 single dispatch |
| Network data, multi-shard | Ring 3 swarm dispatch |
| Remote model inference | Ring 2/3 with T3+ hardware requirement |

### 5.6 — Deliverable

```
prsm compute run --query "Analyze NADA vehicle registration trends in NC
    for 2025 vs 2026, looking specifically for EV adoption shifts" --budget 10.0
# NWTN decomposes → forges Rust agent → compiles to WASM → swarm dispatches
# → aggregates → settles → returns result
```

Or via MCP:
```
User → Claude: "What are the EV adoption trends in North Carolina?"
Claude → [calls prsm_analyze tool]
PRSM → [forge → dispatch → swarm → settle]
Claude → User: "Based on PRSM's analysis of the NADA dataset..."
```

---

## Ring 6 — "The Polish": Production Hardening + CLI UX + Phase Gaps

**Goal:** Everything from Rings 1–5 hardened for real-world deployment. Original Phase 3/5/7 gaps closed. Security battle-tested. CLI polished for humans and AI agents.

### 6.1 — Security Hardening

**WASM Sandbox (Ring 1):**
- Adversarial fuzzing of sandbox escape vectors
- Explicit WASI import deny-list audit
- Binary size limits enforced at gossip layer
- Module signature verification mandatory

**Agent Dispatch (Ring 2):**
- Rate limiting on `GOSSIP_AGENT_DISPATCH` (per-node cap)
- Bid proof-of-data (hash of first 1KB of shard)
- Result replay protection (dispatch-specific nonce)

**Settler/Aggregator (Ring 3/4):**
- Real ECDSA signature verification (replaces Phase 6 simulated signatures)
- Governance-gated slashing (quorum required, not open to anyone)
- JWT/Ed25519 API authentication on all settler and marketplace endpoints
- On-chain bond escrow (FTNS contract upgrade or companion contract)

**Agent Forge (Ring 5):**
- Static analysis on generated WASM for known vulnerability patterns
- Compilation in isolated environment
- Rate limiting on forge requests

### 6.2 — On-Chain Resilience (Phase 1 Hardening)

- Dynamic gas pricing via `eth_gasPrice` RPC + configurable multiplier
- RPC failover with configurable endpoint list and automatic rotation
- Transaction reconciliation on startup (pending tx scan and resubmit)
- Batch retry queue with exponential backoff (max 3 retries over 1 hour)

### 6.3 — CLI UX (Phase 7 Completion)

**New commands:**

```
# Cluster management
prsm cluster status | add | health

# Prosumer experience
prsm ftns yield-estimate | stake | earnings

# Data marketplace
prsm marketplace list-dataset | browse | quote

# Agent operations
prsm agent forge | dispatch | status | traces

# Diagnostics
prsm node benchmark | connectivity
prsm debug last-error
```

**Bootstrap reliability:**
1. Primary → EU fallback → APAC fallback (existing)
2. mDNS local discovery (new — find PRSM nodes on same LAN)
3. Local-only mode with clear status and 60s auto-retry
4. Seamless transition to full P2P when peers connect

**Error messages:** Structured output with cause, suggestion, and detail pointer.

### 6.4 — Agent Collaboration (Phase 5 Gaps)

- **Sealed-bid auction:** Replaces basic first-come bidding. Configurable bid window (default 5s), scored by price × reputation × latency × hardware match
- **Subagent spawning:** Running WASM agents can request child agents via manifest `max_subagents` + sub-budget. Children inherit parent's delegation certificate
- **Reputation integration:** Job completion, consensus, uptime feed into existing reputation system. Reputation influences bid ranking

### 6.5 — Observability

- OpenTelemetry traces extended across: forge → dispatch → execution → settlement
- Streamlit dashboard: network-wide swarm jobs, FTNS flow, shard heatmap, node health
- Operator alerts: earnings threshold, uptime drops, slash risk

### 6.6 — Documentation & SDK Updates

- Operator Guide (prosumer flow)
- Data Provider Guide (listing, pricing, access controls)
- Agent Developer Guide (manual WASM SDK usage)
- MCP Integration Guide (connecting any LLM)
- Python/JS/Go SDK updates for new endpoints

### 6.7 — Deliverable

PRSM is ready for external developers and investors:
- Developer: clone → setup → stake → earn FTNS (end-to-end in Operator Guide)
- Data provider: list dataset → earn royalties (minutes to onboard)
- Researcher: submit query from any LLM via MCP → get results
- Investor: working P2P, real token economics, secure sandbox, swarm compute, hybrid marketplace

---

## Ring Dependency Map

```
Ring 1: The Sandbox
  └─→ Ring 2: The Courier (needs sandbox + hardware profiles)
        └─→ Ring 3: The Swarm (needs agent dispatch)
              └─→ Ring 4: The Economy (needs swarm + sharding for pricing)
                    └─→ Ring 5: The Brain (needs economy for quoting)
                          └─→ Ring 6: The Polish (hardens everything)
```

Each ring is independently valuable:
- Ring 1 alone = secure compute provider
- Ring 1 + 2 = working mobile agent system
- Ring 1 + 2 + 3 = distributed analytics platform
- Ring 1–4 = self-sustaining network economy
- Ring 1–5 = zero-code agent creation from any LLM
- Ring 1–6 = production-ready, investor-presentable platform

---

## Existing Infrastructure Reuse

| Existing Component | Used In | How |
|-------------------|---------|-----|
| NWTN orchestrator + backends | Ring 5 | Intent decomposition, agent code generation |
| OpenRouter integration | Ring 5 | LLM backend for agent forge |
| Gossip protocol (56 msg types) | Ring 2, 3 | Agent dispatch, shard advertisement |
| WebSocket transport | Ring 2 | WASM binary direct transfer |
| IPFS sharding | Ring 3 | Byte-level shard storage |
| BitTorrent client | Ring 3 | Shard replication across swarm |
| Content economy | Ring 3, 4 | Replication tracking, royalty distribution |
| Multi-party escrow | Ring 4 | Batched revenue distribution |
| Payment escrow | Ring 2 | Per-agent FTNS settlement |
| Batch settlement | Ring 2, 4 | Gas-efficient on-chain confirmation |
| Settler registry | Ring 3, 4 | Aggregator nodes, batch validation |
| Agent registry | Ring 2 | Node discovery by capability |
| Agent collaboration | Ring 6 | Bidding, subagent spawning |
| Marketplace | Ring 4 | Data listings, reputation |
| Vector store backend | Ring 3 | Shard centroid indexing |
| MCP integration | Ring 5 | "MCP Everywhere" tool exposure |
| Hardware profiling | Ring 1 | Extended with TFLOPS, thermal, bandwidth |
| CLI | Ring 6 | New commands, polish |

---

## What This Spec Does NOT Cover

- **NWTN fine-tuned model training** — deferred until sufficient AgentTrace corpus exists (post-Ring 5)
- **Console/mobile platform daemons** — PS5/Xbox/smartphone PRSM clients are a separate engineering effort beyond the core protocol
- **FTNS contract upgrades** — on-chain bond escrow in Ring 6 may require a companion contract; design TBD
- **Regulatory/compliance** — token economics and data marketplace may have jurisdiction-specific requirements
- **UI/frontend** — this spec covers CLI and MCP interfaces; a web dashboard is out of scope
