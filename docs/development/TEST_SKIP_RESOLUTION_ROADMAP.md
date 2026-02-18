# PRSM Development Roadmap

> **Test status:** 920 passed / 31 failed / 77 skipped (as of 2026-02-18)
>
> This document covers two areas:
> 1. **Collaboration Infrastructure Completion** — the remaining work to make
>    PRSM's P2P file sharing, provenance royalties, and token economy fully
>    functional across the network
> 2. **Test Skip Resolution** — every remaining skipped test, its root cause,
>    and the concrete fix
>
> Items are organized into prioritized phases.

---

# Part I: Collaboration Infrastructure Completion

> **Status as of 2026-02-18:** P2P node networking is fully functional.
> File sharing and the token economy are scaffolded — local implementations
> exist but are not wired into network-wide operations.

## Infrastructure Audit Summary

| Area | Status | What Works | What's Missing |
|------|--------|-----------|----------------|
| **P2P Node Networking** | FUNCTIONAL | WebSocket handshake, signed messages, gossip propagation, peer discovery, bootstrap via `wss://bootstrap.prsm-network.com` | — |
| **File Storage & Sharing** | SCAFFOLDED | IPFS client uploads/downloads to local daemon, storage provider pins content, content uploader records provenance locally | Cross-node file discovery, request/serve protocol, access event tracking |
| **FTNS Token Economy** | SCAFFOLDED | Local SQLite ledger, welcome grant, balance tracking, credit/debit/transfer | Network-wide ledger sync, provenance royalty triggers, compute/storage earning distribution |

---

## CI-1: Cross-Node File Discovery & Retrieval

**Goal:** A file pinned on node A can be discovered and retrieved by node B
through the P2P network.

**Current state:**
- `prsm/core/ipfs_client.py` — Real IPFS client with upload, download, chunked
  streaming, retry, and gateway fallback. Works with a local IPFS daemon.
- `prsm/node/storage_provider.py` — Pins content to local IPFS, tracks pinned
  CIDs, earns FTNS locally for storage contribution.
- `prsm/node/content_uploader.py` — Uploads files to IPFS and records
  provenance locally. Has a `record_access()` method that is never called.

**What's missing:**

### Step 1: Content Advertisement via Gossip

**File:** `prsm/node/storage_provider.py`

When a node pins new content, it should gossip a `content_available` message
containing the CID, content metadata (size, type, description), and the
announcing node's ID. Other nodes receive this via gossip and update a local
content directory.

```
Message type: "content_available"
Payload: { cid, metadata, provider_node_id, timestamp }
```

Register a gossip handler in `prsm/node/node.py` during `initialize()` that
processes incoming `content_available` messages and stores them in a local
content index.

### Step 2: Content Index (Local DHT-like Directory)

**New file:** `prsm/node/content_index.py`

A lightweight in-memory (with optional SQLite persistence) index that maps
CIDs to known providers:

```python
class ContentIndex:
    async def register(self, cid: str, provider_id: str, metadata: dict)
    async def lookup(self, cid: str) -> List[ContentProvider]
    async def search(self, query: str) -> List[ContentEntry]
    async def remove(self, cid: str, provider_id: str)
```

Populated by gossip `content_available` messages. Queried when a node wants
to find content.

### Step 3: Content Request/Serve Protocol

**Files:** `prsm/node/transport.py`, `prsm/node/node.py`

Add two new message types to the P2P transport:

- `content_request` — Node A asks Node B for a specific CID
  ```
  Payload: { cid, requester_node_id }
  ```
- `content_response` — Node B replies with the content data or a redirect
  to an IPFS gateway URL
  ```
  Payload: { cid, data_b64 | gateway_url, provider_node_id }
  ```

For large files, the response should include an IPFS gateway URL rather than
streaming the full content over the WebSocket. For small files (<1 MB), direct
transfer via the WebSocket is acceptable.

Register handlers in `node.py` that:
1. On `content_request`: check local IPFS, respond with data or gateway URL
2. On `content_response`: deliver content to the requesting application layer

### Step 4: Access Event Gossip

**Files:** `prsm/node/content_uploader.py`, `prsm/node/node.py`

When content is accessed (downloaded/used), gossip a `content_accessed` message:

```
Message type: "content_accessed"
Payload: { cid, accessor_node_id, timestamp }
```

This triggers provenance royalty credits on the creator's node (see CI-2 below).
Wire `content_uploader.record_access()` to be called when a `content_response`
is served, and have it gossip the access event.

**Effort:** High | **Estimated scope:** ~500 lines across 4-5 files

---

## CI-2: Provenance Royalty Distribution

**Goal:** When content is accessed anywhere on the network, the original
creator receives FTNS royalty credits.

**Current state:**
- `prsm/economy/tokenomics/strategic_provenance.py` — Defines provenance
  nodes, royalty rates, and multi-level provenance graphs. Data structures
  only; no distribution logic.
- `prsm/node/content_uploader.py` — Has `record_access()` (line 166) that
  increments access count and credits the local ledger. **Dead code** — never
  called by anything.
- `prsm/node/local_ledger.py` — Fully functional SQLite ledger with
  credit/debit/transfer. Per-node isolation.

**What's missing:**

### Step 1: Wire Access Events to Royalty Credits

**File:** `prsm/node/node.py`

Register a gossip handler for `content_accessed` messages (from CI-1 Step 4).
When received:
1. Look up the content's provenance record (creator node ID, royalty rate)
2. Credit the creator's balance in the local ledger
3. Record the transaction with type `ROYALTY`

```python
async def _handle_content_accessed(self, msg, peer):
    cid = msg.payload["cid"]
    provenance = self.content_index.get_provenance(cid)
    if provenance and provenance.creator_id == self.identity.node_id:
        royalty = provenance.royalty_rate  # e.g., 0.01 FTNS per access
        self.ledger.credit(royalty, f"royalty:{cid}:{msg.payload['accessor_node_id']}")
```

### Step 2: Provenance Registration at Upload Time

**File:** `prsm/node/content_uploader.py`

When content is uploaded, store a provenance record that includes:
- Creator node ID
- CID
- Royalty rate (configurable, default 0.01 FTNS per access)
- Timestamp
- Content type / description

Include provenance data in the `content_available` gossip message so all
nodes know who created what and what the royalty rate is.

### Step 3: Multi-Level Provenance (Derivative Works)

**File:** `prsm/economy/tokenomics/strategic_provenance.py`

When content is derived from other content (e.g., a model fine-tuned on a
dataset), record the provenance chain. On access, distribute royalties
up the chain:

- Creator of derivative: 70% of royalty
- Creator of source material: 25% of royalty
- Network fee: 5%

This leverages the existing `ProvenanceGraph` data structures in
`strategic_provenance.py` — they just need to be connected to the gossip
layer.

**Effort:** Medium | **Estimated scope:** ~300 lines across 3-4 files

---

## CI-3: Network-Wide FTNS Ledger Synchronization

**Goal:** Node balances stay consistent across the network so that transfers
and payments between nodes are valid.

**Current state:**
- `prsm/node/local_ledger.py` — SQLite-backed, per-node ledger. Tracks
  transactions locally. No network awareness.
- `prsm/economy/tokenomics/ftns_service.py` — In-memory token tracking
  with award/deduct/burn mechanics. Standalone, not integrated into nodes.

**What's missing:**

### Step 1: Transaction Gossip

**File:** `prsm/node/local_ledger.py`, `prsm/node/node.py`

When a node records a transaction (credit, debit, transfer, royalty), gossip
it to the network:

```
Message type: "ftns_transaction"
Payload: {
    tx_id, tx_type, amount, from_node, to_node,
    reason, timestamp, signature
}
```

All nodes maintain a transaction log. Signature verification ensures only
the debited node can authorize outgoing transfers.

### Step 2: Balance Reconciliation

**File:** `prsm/node/local_ledger.py`

Periodically (or on demand), nodes can request a balance proof from peers:

```
Message type: "balance_request" / "balance_response"
```

If a node's computed balance (from its transaction log) differs from the
network consensus, flag it for investigation. This is an eventually-consistent
model — not full Byzantine consensus, but sufficient for the current scale.

### Step 3: Double-Spend Prevention

**File:** `prsm/node/local_ledger.py`

Before processing an outgoing transfer:
1. Check local balance is sufficient
2. Create a signed transaction with a unique nonce
3. Gossip the transaction
4. Debit locally only after gossip confirmation

The nonce prevents replay attacks. At small network scale (<100 nodes),
gossip propagation is fast enough that double-spend windows are negligible.

### Step 4: Earning Mechanisms Beyond Welcome Grant

**Files:** `prsm/node/storage_provider.py`, `prsm/node/compute_provider.py`

Currently, storage providers earn FTNS locally via `_reward_loop()` but
don't sync with the network. Wire the following earning events into the
transaction gossip:

| Activity | Reward | Trigger |
|----------|--------|---------|
| Storage contribution | 0.1 FTNS/GB/day | Periodic proof-of-storage |
| Compute job completion | Variable per job | Job result verification |
| Content creation royalty | 0.01 FTNS/access | `content_accessed` gossip |
| Governance participation | 0.5 FTNS/vote | Governance vote recorded |

**Effort:** High | **Estimated scope:** ~600 lines across 4-5 files

---

## CI Execution Order

```
CI-1 (File Discovery & Retrieval)
 ├── Step 1: Content advertisement gossip
 ├── Step 2: Content index
 ├── Step 3: Request/serve protocol
 └── Step 4: Access event gossip ──────────┐
                                           │
CI-2 (Provenance Royalties)                │
 ├── Step 1: Wire access events ◄──────────┘
 ├── Step 2: Provenance at upload
 └── Step 3: Multi-level provenance
                                           │
CI-3 (FTNS Ledger Sync)                   │
 ├── Step 1: Transaction gossip ◄──────────┘
 ├── Step 2: Balance reconciliation
 ├── Step 3: Double-spend prevention
 └── Step 4: Earning mechanisms
```

**CI-1 should be built first** — it's the foundation that CI-2 and CI-3
depend on (access events trigger royalties, which trigger transactions).

**Total estimated scope:** ~1,400 lines of new code across ~12 files.
Most changes extend existing modules rather than creating new ones.

---

# Part II: Test Skip Resolution

---

## Phase 1: Quick Wins (0 code changes required)

### 1A. Database Environment Configuration (10 skips)

**File:** `tests/integration/test_ftns_concurrency_integration.py`

These 10 tests use PostgreSQL-specific features (row-level locking, `SELECT FOR UPDATE`,
`asyncpg`) and skip when `DATABASE_URL` or `TEST_DATABASE_URL` is not set.

| Test | Line | Skip Reason |
|------|------|-------------|
| `test_concurrent_balance_updates` | 124 | No DATABASE_URL |
| `test_concurrent_transaction_processing` | 247 | No DATABASE_URL |
| `test_double_spend_prevention` | 285 | No DATABASE_URL |
| `test_concurrent_reward_distribution` | 332 | No DATABASE_URL |
| `test_atomic_transfer_operations` | 373 | No DATABASE_URL |
| `test_concurrent_marketplace_purchases` | 408 | No DATABASE_URL |
| `test_concurrent_governance_voting` | 466 | No DATABASE_URL |
| `test_high_frequency_micro_transactions` | 517 | No DATABASE_URL |
| `test_concurrent_staking_operations` | 622 | No DATABASE_URL |
| `test_stress_test_transaction_throughput` | 638 | No DATABASE_URL |

**Resolution:** Add a PostgreSQL service to CI (GitHub Actions) and set `DATABASE_URL`.
For local development, add a `docker-compose.test.yml` with a Postgres container:

```yaml
services:
  test-db:
    image: postgres:16
    environment:
      POSTGRES_DB: prsm_test
      POSTGRES_USER: prsm
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
```

Then run: `DATABASE_URL=postgresql+asyncpg://prsm:test@localhost:5433/prsm_test pytest tests/integration/test_ftns_concurrency_integration.py`

**Effort:** Low (infrastructure only, no code changes)

---

### 1B. Optional Third-Party Dependencies (3 skips)

| Test File | Skip Reason | Fix |
|-----------|-------------|-----|
| `tests/integration/test_ui_integration.py` | `selenium` not installed | `pip install selenium` + browser driver |
| `tests/neural/test_semantic_similarity.py` | `sentence-transformers` not installed | `pip install sentence-transformers` (~420 MB) |
| `tests/new_integration_tests/test_integration_demo.py` | `integration_demo_pgvector` not found | Module is a local script, not a package; fix the import path |

**Resolution:**
- Add `selenium` and `sentence-transformers` to an `[optional]` or `[test-full]` extras group in `pyproject.toml`
- For `integration_demo_pgvector`: the test tries to import `from integration_demo_pgvector import ...` as a sibling module. Either add the directory to `sys.path` in the test or convert to a proper relative import.

**Effort:** Low

---

## Phase 2: Export / Import Fixes (6 skips)

These tests skip because existing code doesn't export the symbols the tests expect.

### 2A. FTNS Service Exports (3 skips)

| Test File | Missing Symbol | Source |
|-----------|---------------|--------|
| `tests/environment/persistent_test_environment.py` | `ftns_service` instance | `prsm.economy.tokenomics.ftns_service` |
| `tests/unit/tokenomics/test_ftns_service.py` | Service constants (`INITIAL_BALANCE`, etc.) | `prsm.economy.tokenomics.ftns_service` |
| `tests/test_ftns_budget_manager.py` | `FTNSBudgetManager` class | `prsm.economy.tokenomics.ftns_budget_manager` |

**Resolution:**
1. Export a module-level `ftns_service` singleton from `prsm/economy/tokenomics/ftns_service.py`
2. Export constants (`INITIAL_BALANCE`, `MIN_TRANSACTION_AMOUNT`, etc.) at module level
3. Verify `FTNSBudgetManager` is importable from `prsm.economy.tokenomics.ftns_budget_manager`

**Effort:** Low (add exports to existing modules)

---

### 2B. Specific Missing Symbols (3 skips)

| Test File | Missing Symbol | Fix |
|-----------|---------------|-----|
| `tests/test_budget_api.py` | `prsm.interface.auth` module | Create auth dependency module for FastAPI routes |
| `tests/test_expanded_marketplace.py` | Incomplete SQLAlchemy imports in marketplace models | Fix imports in `prsm/economy/marketplace/` models |
| `tests/test_hybrid_architecture_integration.py` | `AgentTask` class in `prsm.core.models` | Add `AgentTask` Pydantic model to core models |

**Resolution:** Each requires adding a small module or model class. The `prsm.interface.auth` module is a FastAPI dependency that extracts the current user from the JWT token - this is a standard pattern.

**Effort:** Medium

---

## Phase 3: NWTN Subsystem (10 skips + 6 downstream)

The NWTN (Newton) reasoning engine is referenced by many tests but key modules are missing.

### Missing Modules

| Module | Required By | Description |
|--------|-------------|-------------|
| `prsm.compute.nwtn.orchestrator` | 4 test files | Central orchestrator class (`NWTNOrchestrator`) |
| `prsm.compute.nwtn.meta_reasoning_engine` | 4 test files | Meta-reasoning pipeline |
| `prsm.compute.nwtn.complete_system` | 1 test file | End-to-end system entry point |
| `prsm.compute.nwtn.external_storage_config` | 1 test file | External storage configuration |

### Directly Blocked Tests (10)

| Test File | Lines | Blocked By |
|-----------|-------|------------|
| `tests/integration/api/test_endpoint_integration.py` | 28 | `orchestrator` |
| `tests/test_nwtn_direct_prompt_1.py` | 24 | `meta_reasoning_engine` |
| `tests/test_nwtn_final_clean.py` | 23 | `meta_reasoning_engine` |
| `tests/test_nwtn_final_fixed.py` | 23 | `meta_reasoning_engine` |
| `tests/test_nwtn_integration.py` | 36 | `orchestrator` |
| `tests/test_nwtn_prompt_1.py` | 31 | `complete_system` |
| `tests/test_nwtn_provenance_integration.py` | 54 | `meta_reasoning_engine` |
| `tests/test_nwtn_search_fix.py` | 15 | `external_storage_config` |
| `tests/test_nwtn_simple.py` | 14 | `orchestrator` |
| `tests/test_prsm_system_integration.py` | 63 | `orchestrator` |

### Downstream Tests Also Blocked (6)

These user-workflow and integration tests also depend on NWTN:

| Test File | Lines | Additional Dependencies |
|-----------|-------|------------------------|
| `tests/integration/workflows/test_user_workflows.py` | 53 | NWTNOrchestrator |
| `tests/integration/workflows/test_user_workflows.py` | 163 | Marketplace API |
| `tests/integration/workflows/test_user_workflows.py` | 342 | `prsm.collaboration` |
| `tests/integration/workflows/test_user_workflows.py` | 611 | NWTNOrchestrator |
| `tests/integration/workflows/test_user_workflows.py` | 674 | NWTNOrchestrator |
| `tests/test_hierarchical_consensus.py` | 25 | `prsm.performance` |

### Implementation Plan

The NWTN subsystem already has extensive code under `prsm/compute/nwtn/` (reasoning modules,
convergence detection, meta-generation engines, etc.). The missing pieces are the
**orchestrator** (which ties them together) and the **meta-reasoning engine** (which
routes queries through the reasoning pipeline).

**Step 1:** Create `prsm/compute/nwtn/orchestrator.py` with an `NWTNOrchestrator` class that:
- Accepts a query and optional context
- Routes through the existing reasoning pipeline
- Returns a structured response

**Step 2:** Create `prsm/compute/nwtn/meta_reasoning_engine.py` that:
- Implements the meta-reasoning loop (System 1 fast path + System 2 deep reasoning)
- Uses existing modules: `reasoning/`, `convergence_analyzer`, `meta_generation_engine`

**Step 3:** Create `prsm/compute/nwtn/complete_system.py` as a facade that combines
the orchestrator with provenance tracking and content grounding.

**Step 4:** Create `prsm/compute/nwtn/external_storage_config.py` for IPFS/external
storage configuration.

**Effort:** High (core feature implementation)

---

## Phase 4: Marketplace Service Completion (13 skips)

### Missing Methods on `RealMarketplaceService`

| Method | Test Files Using It |
|--------|-------------------|
| `create_resource_listing()` | marketplace_production, marketplace_activation |
| `create_ai_model_listing()` | marketplace_activation |
| `create_dataset_listing()` | marketplace_activation |
| `create_agent_listing()` | marketplace_activation |
| `create_tool_listing()` | marketplace_activation |
| `search_resources()` | marketplace_production, marketplace_activation |
| `get_comprehensive_stats()` | marketplace_production, marketplace_activation |

### Missing Enum Value

`ModelProvider.PRSM` needs to be added to the `ModelProvider` enum.

### Blocked Tests

| Test File | Count |
|-----------|-------|
| `tests/integration/test_marketplace_production_integration.py` | 5 |
| `tests/new_integration_tests/test_marketplace_activation.py` | 8 |

### Implementation Plan

**Step 1:** Add `PRSM = "prsm"` to the `ModelProvider` enum

**Step 2:** Implement CRUD methods on `RealMarketplaceService`:
- `create_resource_listing()` - Generic resource listing creation
- `create_ai_model_listing()` - AI model-specific listing
- `create_dataset_listing()` - Dataset-specific listing
- `create_agent_listing()` - Agent-specific listing
- `create_tool_listing()` - Tool-specific listing

**Step 3:** Implement query methods:
- `search_resources()` - Filtered search across listings
- `get_comprehensive_stats()` - Aggregate marketplace statistics

**Effort:** Medium-High (feature implementation on existing service)

---

## Phase 5: Performance Module (2 skips + 1 benchmark)

### Missing Module: `prsm.performance`

| Test File | What It Tests |
|-----------|--------------|
| `tests/test_performance_instrumentation.py` | Performance metric collection and instrumentation |
| `tests/test_performance_optimization.py` | Automatic performance optimization strategies |
| `tests/test_benchmark_orchestrator.py` | Benchmark orchestration and reporting |

### Implementation Plan

Create `prsm/performance/` package with:
- `instrumentation.py` - Decorators and context managers for timing, memory profiling
- `optimization.py` - Caching strategies, batch processing optimization
- `benchmark_orchestrator.py` - Test orchestration, result aggregation, reporting

**Effort:** Medium

---

## Phase 6: Generic "Not Yet Implemented" Tests (26 skips)

These 26 tests all skip with the generic message "Module dependencies not yet fully
implemented". Each needs individual investigation to determine the actual blocker.

### Grouped by Subsystem

#### Integration Tests (10 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_api_integration_comprehensive.py` | Full API stack |
| `test_collaboration_platform_integration.py` | `prsm.collaboration` module |
| `test_complete_collaboration_platform.py` | `prsm.collaboration` module |
| `test_end_to_end_prsm_workflow.py` | NWTN + FTNS + Marketplace |
| `test_full_spectrum_integration.py` | All subsystems |
| `test_integration_suite_runner.py` | Test orchestration infrastructure |
| `test_p2p_integration.py` | P2P networking stack |
| `test_phase7_integration.py` | Phase 7 features (governance?) |
| `test_real_data_integration.py` | Real data pipelines |
| `test_system_resilience_integration.py` | Fault tolerance infrastructure |

#### Tokenomics & Economy (4 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_advanced_ftns.py` | Advanced FTNS features |
| `test_advanced_tokenomics_integration.py` | Full tokenomics stack |
| `test_marketplace.py` | Marketplace + FTNS integration |
| `test_full_governance_system.py` | Governance + FTNS |

#### Core System (4 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_agent_framework.py` | Agent orchestration framework |
| `test_breakthrough_modes.py` | NWTN breakthrough detection |
| `test_consensus_integration.py` | Consensus mechanism |
| `test_150k_papers_provenance.py` | Provenance at scale |

#### External Integrations (3 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_openai_free_tier.py` | OpenAI API integration |
| `test_openai_integration.py` | OpenAI API integration |
| `test_openai_real_integration.py` | OpenAI API integration |

#### Scripts & Performance (3 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_governance.py` | Governance scripts |
| `simple_performance_test.py` | Performance testing |
| `test_performance_benchmarks_alt.py` | Benchmark infrastructure |

#### P2P & Federation (2 files)

| Test File | Likely Dependency |
|-----------|------------------|
| `test_p2p_federation.py` | P2P federation protocol |
| `test_production_p2p_federation.py` | Production P2P deployment |

### Resolution Strategy

Many of these will automatically resolve as Phases 3-5 are completed (NWTN, Marketplace,
Performance). The remaining ones likely need:
1. Individual investigation of each file's actual import block
2. Either implementing the missing module or fixing the import path
3. Some may be candidates for deletion if they test features that have been redesigned

**Effort:** High (requires investigation + implementation across many subsystems)

---

## Test Skip Summary

| Phase | Description | Skips Resolved | Effort |
|-------|-------------|---------------|--------|
| 1A | Database env config for CI | 10 | Low |
| 1B | Optional third-party deps | 3 | Low |
| 2A | FTNS service exports | 3 | Low |
| 2B | Missing symbols/modules | 3 | Medium |
| 3 | NWTN subsystem | 10 (+6 downstream) | High |
| 4 | Marketplace service | 13 | Medium-High |
| 5 | Performance module | 3 | Medium |
| 6 | Generic "not implemented" | 26 | High |
| **Total** | | **77** | |

### Recommended Execution Order

1. **Phase 1** (13 skips) - Zero or minimal code changes, immediate wins
2. **Phase 2** (6 skips) - Small export/import fixes, low risk
3. **Phase 4** (13 skips) - Marketplace is well-defined, existing service to extend
4. **Phase 5** (3 skips) - Performance module is self-contained
5. **Phase 3** (16 skips) - NWTN is the largest feature gap, high impact
6. **Phase 6** (26 skips) - Many will auto-resolve from earlier phases; remainder needs investigation

### Target: Zero Skips

After all phases are complete, the test suite should have **0 skips** (or only
infrastructure-dependent skips that are expected in environments without PostgreSQL,
Selenium, or GPU access). All "not yet implemented" skips should be resolved by either
implementing the feature or removing obsolete test files.

---

# Part III: Overall Project Status & Priorities

## What's Complete

| Component | Status | Notes |
|-----------|--------|-------|
| P2P node networking | DONE | WebSocket transport, gossip, discovery, signed messages |
| Bootstrap infrastructure | DONE | Cloudflare Tunnel at `wss://bootstrap.prsm-network.com`, launchd persistence |
| Health endpoint | DONE | Reports "healthy" for local-first installs, optional services shown as "not_configured" |
| Pure-Python merkle tree | DONE | Replaced `merkletools` C extension, works on Python 3.13+ |
| New-user onboarding | DONE | `git clone` → `pip install` → `prsm node start` → auto-connects to network |
| Local FTNS ledger | DONE | SQLite-backed, per-node balance tracking with transaction history |
| IPFS client | DONE | Real client with upload/download/pin, requires external IPFS daemon |
| Neuro-symbolic reasoning (S1) | DONE | System 1 fast-path reasoning via `prsm/compute/nwtn/reasoning/` |

## What's Scaffolded (Needs Network Wiring)

| Component | Blocking Issue | Roadmap Section |
|-----------|---------------|-----------------|
| Cross-node file sharing | No content directory or request/serve protocol | CI-1 |
| Provenance royalties | Access events never triggered; `record_access()` is dead code | CI-2 |
| Network FTNS distribution | Ledgers are per-node; no gossip sync or consensus | CI-3 |
| Marketplace CRUD | Service exists but missing listing creation/search methods | Phase 4 |
| NWTN orchestrator | Reasoning modules exist but top-level orchestrator missing | Phase 3 |

## Recommended Overall Priority

```
Priority 1 (Foundation):     CI-1 File Discovery + Phase 1-2 Test Fixes
Priority 2 (Economy):        CI-2 Provenance Royalties + CI-3 Ledger Sync
Priority 3 (Features):       Phase 3 NWTN Orchestrator + Phase 4 Marketplace
Priority 4 (Polish):         Phase 5-6 Performance + Remaining Test Skips
```

CI-1 through CI-3 represent approximately **1,400 lines of new code** across
12 files. Most changes extend existing modules. Once complete, PRSM's core
collaboration architecture — nodes joining, sharing files, earning tokens via
provenance royalties — will be fully operational end-to-end.
