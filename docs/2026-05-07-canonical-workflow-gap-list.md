# Canonical 8-Step Workflow — Reality-Check Gap-List

**Date:** 2026-05-07
**Method:** Code-survey against the canonical user-workflow memory
(validated 2026-04-26). No live runtime; reads against tip
`origin/main @ 11ce30a3`.

## Summary

PRSM has **two distinct end-to-end paths**, of which **one is real
on-chain-ready and the other is broken since v1.6.0**:

| Path | Use case | Status |
|------|----------|--------|
| **Inference path** (`prsm_inference` → `/compute/inference`) | TEE-attested LLM inference with verifiable receipt | ✅ **REAL** end-to-end. Phase 3.x.1-3.x.11 builds (10 sprints) wired the full TensorParallelExecutor + ParallaxScheduledExecutor + streaming + receipt + escrow + privacy-budget pipeline. |
| **Data-query path** (`prsm_analyze` / `prsm_dispatch_agent` → `/compute/forge`) | Natural-language data query → DSL manifest → fan out WASM agents to shard-holders → aggregate → settle | ❌ **BROKEN.** `node.agent_forge = None` unconditionally per `prsm/node/node.py:1196-1197`. The `/compute/forge` endpoint guards on `agent_forge is None` and returns HTTP 503. |

**This is the load-bearing gap between current state and "fully
operational PRSM" as the canonical workflow defines it.** The inference
path is one of two product surfaces; the data-query path — the path the
Vision describes in §1 ("the user never chooses — the model routes")
and the path that mints PCU-style billing on commons-content royalties
(Prismatica revenue stream #1) — has been non-functional since v1.6.0
shipped on 2026-04-09.

## 8-step canonical workflow vs reality

### Step 1 — Install: `pip install prsm-network` ✅ REAL
- Package on PyPI: 1.6.0 published 2026-04-09 (per memory).
- `prsm` CLI entrypoint registered.

### Step 2 — Configure: `prsm node start` + MCP registration ✅ REAL
- `prsm/node/node.py` brings up the node runtime.
- `prsm/mcp_server.py` registers 18 tools that auto-discover.
- `prsm/skills/` MCP tool package system.

### Step 3 — LLM-routed invocation ✅ REAL (from MCP perspective)
- 18 MCP tools register correctly: `prsm_analyze`, `prsm_quote`,
  `prsm_inference`, `prsm_create_agent`, `prsm_dispatch_agent`,
  `prsm_agent_status`, `prsm_search_shards`, `prsm_upload_dataset`,
  `prsm_node_status`, `prsm_hardware_benchmark`, `prsm_yield_estimate`,
  `prsm_stake`, `prsm_revenue_split`, `prsm_settlement_stats`,
  `prsm_privacy_status`, `prsm_training_status`, `prsm_billing_status`,
  `prsm_list_datasets`.
- Tool descriptions advertise full functionality regardless of backend
  reality (see Step 4 caveat below).

### Step 4 — Manifest construction via constrained DSL ✅ REAL (DSL primitive) / ❌ BROKEN (handoff)
**The DSL itself is real:** `prsm/compute/agents/instruction_set.py:19`
defines `class AgentOp(str, Enum)` with **exactly the 11 canonical
operators**: `filter`, `aggregate`, `group_by`, `sort`, `limit`,
`count`, `sum`, `average`, `select`, `compare`, `time_series`. This
matches the memory-validated workflow exactly.

**The MCP tool that builds manifests is real:** `handle_prsm_create_agent`
constructs `InstructionManifest`, validates op codes, returns JSON.

**The handoff is broken:** the manifest's only consumer is
`/compute/forge`, which is dead (Step 6).

### Step 5 — Parallel SPRK dispatch ⚠️ COMPONENTS REAL, ORCHESTRATOR DELETED
**Survives:**
- `prsm/compute/wasm/runtime.py` — Wasmtime adapter (`WasmtimeRuntime`)
- `prsm/compute/agents/dispatcher.py` — `class AgentDispatcher`
- `prsm/compute/agents/executor.py` — `class AgentExecutor`
- `prsm/compute/tee/confidential_executor.py` — TEE attestation
  (Ring 7, kept in v1.6.0)
- `prsm/compute/swarm/` — swarm fan-out primitives

**Deleted:** the orchestration glue ("Agent Forge", Ring 5) that
wired query decomposition → shard discovery → AgentDispatcher fan-out
→ AgentExecutor → results gather. Per `node.py:1196`:

```python
# Agent Forge (Ring 5) removed in v1.6.0 — legacy NWTN AGI framework
self.agent_forge = None
```

The deletion was deliberate per the v1.6.0 sprint memory (Ring 5 was
bundled with the NWTN AGI framework, which was out of scope). But the
data-query path was not re-wired with a non-AGI orchestrator, leaving
a permanent gap.

### Step 6 — Aggregator selection + DP noise + batch settlement ⚠️ COMPONENTS REAL, SELECTION LOGIC ABSENT
**Survives:**
- DP noise: `prsm/compute/tee/dp_noise.py` (Laplace + Gaussian
  mechanisms), `prsm/security/privacy_budget.py` (epsilon tracking),
  `prsm/compute/inference/executor.py` (per-call DP wiring).
- Batch settlement: `prsm/settlement/` (settler registry + batch
  flush), `prsm/node/node.py:1190-1194` (`_settler_registry.on_settlement_ready`).

**Absent:**
- No `select_aggregator` / `choose_aggregator` / `pick_aggregator`
  symbol exists anywhere in `prsm/`. Search returns zero hits.
- The Vision §6 commitment ("aggregator selected from T2+ pool,
  typically NOT the prompter's node") has no implementation. Even if
  Agent Forge were restored, an aggregator-selection layer would still
  be a fresh build.

### Step 7 — Contract-enforced settlement ✅ REAL on mainnet
- `RoyaltyDistributor` v1 (push-payment) live at
  `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2` on Base mainnet.
- v2 source (pull-payment + Ownable2Step + recoverStranded; A-08 fix
  shipped 2026-05-07) ready for the planned redeploy ceremony.
- 3-way split, 2% network fee, immutable network treasury — all
  Phase-1.3-Task-8 verified.
- Foundation Safe 2-of-3 multisig owns 7 audit-bundle / Phase-8 /
  Phase-7-storage contracts (per Phase B mainnet ceremony 2026-05-07).

**Caveat:** the canonical Tokenomics §5.1 split (20% burn, 1.6%
treasury, 6.4% creator royalty, 72% serving compute providers) is a
*post-aggregator* split that requires the data-query path (Step 6) to
fire to be exercised. The mainnet contract enforces a different split
(creator + network + serving-node) keyed off the
`distributeRoyalty(contentHash, servingNode, gross)` shape — closer to
Step 7 inference settlement than Step 6 data-query settlement.

### Step 8 — Result delivery with cost reconciliation ✅ REAL
- `_format_cost_footer` (Phase 3.x.1 Task 7) renders the standard
  cost footer for every MCP tool response.
- For `prsm_analyze` it works against the dead `/compute/forge` (so
  the footer renders but the response above it is "Agent forge not
  initialized"); for `prsm_inference` it renders against real
  inference receipts.

## Tool-by-tool real/broken matrix

| MCP tool | Backend route | Status | Notes |
|----------|--------------|--------|-------|
| `prsm_inference` | `/compute/inference` | ✅ REAL | Full Phase 3.x.1 wiring |
| `prsm_inference` (streaming) | `/compute/inference/stream` | ✅ REAL | Phase 3.x.8 |
| `prsm_analyze` | `/compute/forge` | ❌ BROKEN | 503 — `agent_forge=None` |
| `prsm_create_agent` | (local; just builds JSON manifest) | ✅ REAL | Manifest is unconsumable downstream |
| `prsm_dispatch_agent` | `/compute/forge` | ❌ BROKEN | 503 |
| `prsm_agent_status` | `/compute/status/{job_id}` | ❌ BROKEN | Backing endpoint does not exist on node API — call returns 404. Confirmed via grep on `prsm/node/api.py`. |
| `prsm_quote` | local `PricingEngine` | ✅ REAL | Uses pricing module directly; does NOT call `/compute/forge/quote`. Earlier guess was wrong. |
| `prsm_search_shards` | `/content/search` → `node.content_index.search()` | ✅ REAL | |
| `prsm_upload_dataset` | `/content/upload/shard` | ⚠️ **STUB — CRITICAL** | Registers a `SemanticShardManifest` in `data_listing_manager` but **does NOT upload to IPFS**. Per `prsm/node/api.py:1392` the CIDs are placeholders (`cid=f"Qm{dataset_id}-{i:04d}"`) with comment `# Placeholder until IPFS upload`. **Creator-economy implication:** the canonical creator flow ("Creators publish data via `prsm_upload_dataset`; earn 6.4% royalty continuously as queries hit their content") cannot work — queries would have nothing to route against because nothing real is actually published. |
| `prsm_node_status` | local node API | ✅ REAL | Status read |
| `prsm_hardware_benchmark` | local | ✅ REAL | Benchmarking framework live |
| `prsm_yield_estimate` | local | ✅ REAL | Calculator |
| `prsm_stake` | on-chain Phase 7 StakeBond | ✅ REAL | Wired to live mainnet contract |
| `prsm_revenue_split` | on-chain RoyaltyDistributor | ✅ REAL | Wired to live mainnet contract |
| `prsm_settlement_stats` | local settler registry | ✅ REAL | |
| `prsm_privacy_status` | local privacy budget | ✅ REAL | Phase 3.x.4 |
| `prsm_training_status` | `/rings/status` → DashboardMetrics + local TrainingEvaluator | ✅ REAL | Hits real Ring 9 surface; degrades gracefully when traces are scarce |
| `prsm_billing_status` | local billing tracker | ✅ REAL | Phase 3.x.1 Task 7 |
| `prsm_list_datasets` | `/content/search` → `node.content_index.search()` | ✅ REAL | Same backend as `prsm_search_shards` |

**Final tally after 2026-05-07 unverified-6 walk:**
- **REAL: 13 of 18** — quote, list_datasets, search_shards, training_status promoted from unverified.
- **BROKEN: 4 of 18** — analyze, dispatch_agent, agent_status (confirmed: `/compute/status/{job_id}` endpoint does not exist), and the implicit collateral on any tool that depends on `/compute/forge`.
- **STUB: 1 of 18** — `prsm_upload_dataset` registers a manifest but does not upload to IPFS (CIDs are placeholders). **This is the second load-bearing finding.** The creator-economy pillar of PRSM (Vision: creators earn 6.4% royalty as queries hit their content) cannot work end-to-end without (a) IPFS upload completion AND (b) QueryOrchestrator reconstruction. Either one alone is insufficient.

## What "fully operational" needs

Three categories of work, in order of leverage:

### A. RESTORE THE DATA-QUERY PATH (highest leverage)

The DSL exists. The agent runtime exists. WASM, TEE, P2P transport,
batch settlement all exist. **What's missing is a non-AGI
orchestrator** that ties them together — call it `QueryOrchestrator`
to avoid the Ring-5/Agent-Forge legacy framing. Estimated scope:

1. `prsm/compute/query_orchestrator/` — new package
2. `class QueryOrchestrator` with `async run(query, budget_ftns,
   shard_cids=None)` — entrypoint that mirrors the deleted
   AgentForge.run signature so `/compute/forge` can rewire with
   minimal API churn.
3. Sub-modules:
   - `decomposer.py` — natural-language query → DSL manifest. Can
     start with a simple LLM-prompted decomposition (the user's own
     LLM already routed here) before optimizing.
   - `shard_finder.py` — query content → relevant shard CIDs.
     Hooks into the existing PRSM-PROV-1 EmbeddingDHT (Item 3) +
     ManifestDHT (Phase 3.x.5).
   - `aggregator_selector.py` — pick a T2+ aggregator from the
     stake-weighted reputation set (StakeBond-backed). **NEW
     primitive** — does not exist anywhere today.
   - `swarm_runner.py` — fan out via existing AgentDispatcher,
     gather via existing AgentExecutor results, apply DP noise.
4. Wire `node.agent_forge = QueryOrchestrator(...)` in `node.py:1196`.
5. Tests: 1-node smoke, 3-node fan-out, settlement-on-completion.

**Estimated cost:** ~1,500 source LOC + ~2,000 test LOC. 1-2 weeks
of focused work. The aggregator-selector is the trickiest piece —
needs threat-modeling against stake-weighted-collusion before code.

### B. UPDATE TOOL DESCRIPTIONS TO MATCH REALITY (cheap, high integrity)

The `prsm_analyze` tool description currently reads
"Submit a natural language query to the PRSM distributed compute
network. Automatically decomposes the query via LLM, finds relevant
data shards, dispatches WASM mobile agents to edge nodes, aggregates
results, and settles FTNS token payments." — which an LLM client will
trust. Two options:
1. Hide `prsm_analyze` / `prsm_dispatch_agent` / `prsm_agent_status`
   from the registered tool list until A. lands. ~10 LoC.
2. Update descriptions to "Currently unavailable — Agent Forge backend
   removed in v1.6.0; under reconstruction. Use `prsm_inference` for
   the inference path." ~30 LoC.

Recommendation: **#1** until A. lands, then re-enable with the new
description. LLM clients honoring the tool list won't surface a
broken tool to users.

### C. LIGHT-UP THE UNVERIFIED 6 TOOLS (medium leverage)

Confirm or refute:
- `prsm_search_shards` — likely real (search infrastructure exists)
- `prsm_upload_dataset` — creator path; touches IPFS + registry
- `prsm_quote` — same `/compute/forge/quote` dep as `prsm_analyze`,
  needs verification
- `prsm_list_datasets` — catalog read
- `prsm_agent_status` — orphan if A. doesn't ship
- `prsm_training_status` — Ring 9 surface

Each is ~30 min of code-survey. 3-4 hours total. Output: this matrix
with the "⚠️ UNVERIFIED" rows promoted to ✅ or ❌.

## What this means for "fully operational"

**Honest framing (post-2026-05-07 walk):** PRSM today is a
**mainnet-live private inference network** (the canonical step-7
settlement is real, the `prsm_inference` path delivers TEE-attested
model inference with verifiable receipts on mainnet). The
**data-query path** that Tokenomics §5.1 routes 6.4% creator royalty +
72% compute provider splits through is broken at TWO layers:

1. **Orchestration layer** — `/compute/forge` returns 503 because Agent
   Forge was deleted in v1.6.0 (no QueryOrchestrator successor).
2. **Content-distribution layer** — `prsm_upload_dataset` is a
   registration stub; even with the orchestration layer rebuilt,
   queries would have no real content to route against because
   `/content/upload/shard` doesn't actually push to IPFS.

**Both layers must be functional for the canonical creator-and-querier
workflow to close the loop.** Fixing only one is insufficient.

The Risk Register should reflect this — current investor-facing
materials may overstate "fully operational" if they describe the
data-query path or creator-economy as functional. (Out of scope for
this gap-list to audit; flag for separate review.)

## Recommended next-step ordering

1. **B1 — hide `prsm_analyze` / `prsm_dispatch_agent` / `prsm_agent_status`** from MCP tool list (10-30 LoC, defensive). **Today.**
2. **C — light up the unverified 6 tools** (3-4 hours). **Today/tomorrow.**
3. **A — design + implement QueryOrchestrator** (1-2 weeks). **Next sprint.**
4. **A1 — aggregator-selector threat model first** (1-2 days), before A.
5. After A lands: end-to-end testnet validation against live Base Sepolia contracts (mirror the T10 validation pattern).

## Files referenced

| File | Lines | Why |
|---|---|---|
| `prsm/node/node.py` | 1196-1197 | `self.agent_forge = None` + comment |
| `prsm/node/api.py` | 678-705 | `/compute/forge` 503 guard |
| `prsm/node/api.py` | 802-1024 | `/compute/inference` (real path) |
| `prsm/mcp_server.py` | 75-445 | All 18 MCP tool definitions |
| `prsm/mcp_server.py` | 762-820 | `handle_prsm_analyze` (broken backend) |
| `prsm/mcp_server.py` | 972-1047 | `handle_prsm_create_agent` (real, orphan) |
| `prsm/mcp_server.py` | 1050-1099 | `handle_prsm_dispatch_agent` (broken backend) |
| `prsm/compute/agents/instruction_set.py` | 19-31 | `class AgentOp` — the canonical 11-op DSL |
| `prsm/compute/agents/dispatcher.py` | 31, 104 | `AgentDispatcher` survives |
| `prsm/compute/agents/executor.py` | 28, 68 | `AgentExecutor` survives |
| `prsm/compute/wasm/runtime.py` | 16-130 | `WasmtimeRuntime` survives |
| `prsm/compute/tee/dp_noise.py` | (whole) | DP noise primitives |
| `~/.../memory/project_v1_6_0_sprint_complete.md` | — | v1.6.0 deletion scope memory |
| `~/.../memory/project_canonical_user_workflow.md` | — | The 8-step ground truth |
