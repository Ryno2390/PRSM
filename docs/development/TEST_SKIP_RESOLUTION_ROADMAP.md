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

> **Status as of 2026-02-18:** P2P node networking and cross-node file
> discovery are fully functional. The token economy is scaffolded — local
> implementations exist but are not wired into network-wide operations.
> A new agentic interoperability layer (CI-4) is planned to make AI agents
> first-class participants on the network.

## Infrastructure Audit Summary

| Area | Status | What Works | What's Missing |
|------|--------|-----------|----------------|
| **P2P Node Networking** | FUNCTIONAL | WebSocket handshake, signed messages, gossip propagation, peer discovery, bootstrap via `wss://bootstrap.prsm-network.com` | — |
| **File Storage & Sharing** | FUNCTIONAL | IPFS upload/download, storage provider pins, content uploader with provenance, content index with keyword search, cross-node content advertisement via gossip, direct-message request/serve (inline ≤1MB or gateway URL), access event tracking with royalty credits | — |
| **FTNS Token Economy** | FUNCTIONAL | Local SQLite ledger, welcome grant, balance tracking, credit/debit/transfer, content royalty credits, network-wide transaction gossip, nonce-based double-spend prevention, balance reconciliation, earning event broadcasting | — |
| **Agentic Interoperability** | PLANNED | Nodes can host multiple services; OpenClaw demonstrates multi-agent coordination externally | Agent identity delegation, agent discovery, agent-to-agent messaging, delegated payments, collaboration protocols |

---

## CI-1: Cross-Node File Discovery & Retrieval — COMPLETE

**Status: DONE** (implemented 2026-02-18)

**Goal:** A file pinned on node A can be discovered and retrieved by node B
through the P2P network.

**What was built:**

- **Content advertisement via gossip** — `GOSSIP_CONTENT_ADVERTISE` messages
  broadcast CID, filename, size, content hash, creator, and provider on every
  upload and pin operation (`content_uploader.py`, `storage_provider.py`).
- **Content index** — `prsm/node/content_index.py` with `ContentRecord`
  dataclass, keyword search (AND semantics over filenames/metadata), LRU
  eviction at 10k entries, provider set tracking.
- **Request/serve protocol** — Direct P2P messages (`content_request` /
  `content_response`) via `MSG_DIRECT`. Small files (≤1MB) served inline
  as base64; large files served via IPFS gateway URL.
- **Access event tracking** — `GOSSIP_CONTENT_ACCESS` messages trigger
  `record_access()` royalty credits (0.01 FTNS) on both the serving node
  and the original creator's node.
- **API endpoints** — `GET /content/search?q=...`, `GET /content/{cid}`,
  `GET /content/index/stats`.
- **Wiring** — `node.py` creates `ContentIndex`, passes transport to uploader,
  calls `start()` on all content subsystems, exposes index stats in status.

---

## CI-2: Provenance Royalty Distribution — COMPLETE

**Status: DONE** (implemented 2026-02-18)

**Goal:** When content is accessed anywhere on the network, the original
creator receives FTNS royalty credits. Derivative works distribute royalties
up the provenance chain.

**What was built:**

- **Configurable royalty rates** — Creators set a rate at upload time (clamped
  to 0.001–0.1 FTNS per access, default 0.01). The rate is included in
  `GOSSIP_CONTENT_ADVERTISE` and `GOSSIP_CONTENT_ACCESS` payloads, stored in
  `ContentRecord` and `UploadedContent`, and exposed via the upload API.
- **Multi-level provenance** — Uploads can declare `parent_cids` (source
  material). On access, royalties are split: 70% derivative creator, 25%
  source creators (split evenly among parents), 5% network fee. If parent
  creators are on the same node, they're credited locally; if remote, they
  receive credits when the `GOSSIP_CONTENT_ACCESS` message arrives.
- **Source creator notification** — `_on_content_access()` now handles both
  cases: (1) we are the direct creator, (2) we are a source creator whose
  content was used as a parent in a derivative work.
- **API updates** — `POST /content/upload` accepts optional `royalty_rate`
  and `parent_cids` fields. All content responses include these fields.

**Files modified:** `content_uploader.py`, `content_index.py`, `node.py`, `api.py`

---

## CI-3: Network-Wide FTNS Ledger Synchronization — COMPLETE

**Status: DONE** (implemented 2026-02-18)

**Goal:** Node balances stay consistent across the network so that transfers
and payments between nodes are valid.

**What was built:**

- **Transaction gossip** — `GOSSIP_FTNS_TRANSACTION` messages broadcast
  every transaction (credits, debits, transfers, royalties) with a signed
  canonical payload. Signature verification ensures only the originating node
  could have authorized the transaction.
- **Nonce-based double-spend prevention** — `seen_nonces` table in SQLite
  tracks processed transaction nonces. Replayed or duplicate transactions
  are rejected. `signed_transfer()` in `LedgerSync` checks balance, signs,
  gossips, and debits atomically.
- **Balance reconciliation** — Every 5 minutes, nodes exchange `balance_request`
  / `balance_response` direct messages with their balance and recent transaction
  IDs. Discrepancies (missing transactions) are logged for investigation.
  Eventually-consistent model sufficient for current network scale.
- **Earning event broadcasting** — Storage rewards (`storage_provider.py`),
  compute earnings (`compute_provider.py`), compute payments
  (`compute_requester.py`), and content royalties (`content_uploader.py`)
  all broadcast their transactions via `ledger_sync.broadcast_transaction()`.
- **Incoming transaction processing** — When a gossipped transaction names
  this node as recipient, it's verified (signature + nonce) and credited
  to the local ledger with a `[remote]` prefix in the description.
- **API endpoints** — `GET /ledger/sync/stats` (broadcast/received/rejected
  counts, reconciliation stats), `POST /ledger/transfer` (signed cross-node
  transfer via the API).
- **New file:** `prsm/node/ledger_sync.py` — `LedgerSync` class (~250 lines).

**Files modified:** `gossip.py`, `local_ledger.py`, `node.py`, `api.py`,
`content_uploader.py`, `storage_provider.py`, `compute_provider.py`,
`compute_requester.py`

---

## CI-4: Agentic Interoperability Layer

**Goal:** Make AI agents first-class participants on the PRSM network —
able to discover each other, communicate, collaborate, and transact on
behalf of their human principals.

**Rationale:** Power users and developers on PRSM will be AI-native
individuals running teams of AI agents (as demonstrated by the OpenClaw
multi-agent setup). The existing P2P infrastructure handles node-to-node
communication well, but an agent is a logical entity *above* the node
layer — one node may host many agents, and one human may control agents
across multiple nodes. PRSM needs a protocol layer that lets agents
interact as peers regardless of which node or orchestration framework
they run on.

### Step 1: Agent Identity & Delegation

**Files:** `prsm/node/identity.py`, new `prsm/node/agent_identity.py`

Extend the identity system so that a human's Ed25519 keypair can issue
**delegation certificates** to agent keypairs. An agent identity contains:

```python
@dataclass
class AgentIdentity:
    agent_id: str              # Unique agent identifier
    agent_name: str            # Human-readable name (e.g. "prsm-coder")
    agent_type: str            # "coding", "research", "devops", etc.
    principal_id: str          # Node ID of the human who controls this agent
    public_key_b64: str        # Agent's own Ed25519 public key
    delegation_cert: str       # Signature from principal proving delegation
    capabilities: List[str]    # Declared capabilities (e.g. ["code_review", "testing"])
    max_spend_ftns: float      # Spending cap per epoch (delegated budget)
    created_at: float
```

The delegation certificate is signed by the principal's private key, binding
the agent's public key to the principal's identity. This lets any node verify
that an agent is authorized to act on behalf of its human without contacting
the principal.

**Scope:** ~150 lines (new dataclass + delegation signing/verification)

### Step 2: Agent Registry & Discovery

**Files:** `prsm/node/gossip.py`, new `prsm/node/agent_registry.py`

New gossip subtypes:
```python
GOSSIP_AGENT_ADVERTISE = "agent_advertise"
GOSSIP_AGENT_DEREGISTER = "agent_deregister"
```

`AgentRegistry` — network-wide directory of known agents, similar to
`ContentIndex` but for agent capabilities:

```python
class AgentRegistry:
    def register_local(agent: AgentIdentity)      # Register an agent on this node
    def lookup(agent_id: str) -> AgentRecord       # Find an agent by ID
    def search(capability: str) -> List[AgentRecord]  # Find agents by capability
    def get_agents_for_principal(principal_id: str) -> List[AgentRecord]
```

When an agent comes online, it gossips `GOSSIP_AGENT_ADVERTISE` with its
identity, capabilities, node location, and availability status. Other nodes
add it to their local registry. Agents can also query the registry to find
peers with specific capabilities ("find me an agent that can do code review").

**Scope:** ~200 lines (registry + gossip handlers + search)

### Step 3: Agent-to-Agent Messaging

**Files:** `prsm/node/transport.py`, `prsm/node/content_uploader.py`

Extend the existing `MSG_DIRECT` transport with agent-level addressing.
Currently, messages are addressed to node IDs. Add an optional `agent_id`
field to the message payload so that a message can be routed to a specific
agent on a target node:

```
Direct message envelope:
{
    msg_type: "direct",
    sender_id: <node_id>,
    payload: {
        subtype: "agent_message",
        from_agent: <agent_id>,
        to_agent: <agent_id>,
        conversation_id: <uuid>,       # Thread/session identifier
        content_type: "text" | "task" | "result" | "query",
        content: { ... },
        delegation_cert: <signature>   # Proves sender is authorized
    }
}
```

The receiving node looks up the target agent in its local registry and
dispatches the message. Content types include:

- **text** — Free-form communication between agents
- **task** — Structured task delegation (description, constraints, budget)
- **result** — Task completion with deliverables
- **query** — Capability/availability inquiry

**Scope:** ~120 lines (message routing + dispatch + content type handlers)

### Step 4: Delegated Payments & Budget Control

**Files:** `prsm/node/local_ledger.py`, `prsm/node/agent_identity.py`

Allow agents to spend FTNS from their principal's wallet up to a delegated
budget cap. This requires:

1. **Allowance records** in the ledger — principal grants agent X a budget
   of Y FTNS per epoch (configurable period, default 24h).
2. **Spend tracking** — each agent payment debits from both the principal's
   balance and the agent's remaining allowance.
3. **Budget refresh** — allowances reset at epoch boundaries.
4. **Revocation** — principal can revoke an agent's spending authority
   instantly via a signed revocation message.

```python
# In local_ledger.py
async def grant_agent_allowance(principal_id, agent_id, amount, epoch_hours=24)
async def agent_debit(agent_id, amount, tx_type, description) -> bool
async def get_agent_allowance(agent_id) -> AgentAllowance
async def revoke_agent_allowance(principal_id, agent_id)
```

Agent-to-agent payments follow this flow:
1. Agent A requests service from Agent B
2. Agent A's node verifies A has sufficient allowance
3. Payment is debited from A's principal's wallet → credited to B's principal's wallet
4. Transaction is gossiped with both agent IDs and principal IDs for auditability

**Scope:** ~200 lines (allowance model + ledger extensions + payment flow)

### Step 5: Collaboration Protocols

**Files:** new `prsm/node/agent_collaboration.py`

Structured protocols for multi-agent collaboration over the P2P network:

**Task Delegation Protocol:**
```
1. Requester agent gossips a task_offer (or sends directly to a known agent)
2. Candidate agents respond with bids (capability match, estimated cost, ETA)
3. Requester selects a bid, sends task_assign with FTNS escrow
4. Worker agent executes, sends task_result
5. Requester verifies result, releases escrow (or disputes)
```

**Peer Review Protocol:**
```
1. Agent submits work product for review
2. Review request is gossiped to agents with relevant capabilities
3. Reviewer agents claim review slots (paid per review)
4. Reviews are collected and aggregated
5. Consensus determines acceptance/revision
```

**Knowledge Exchange Protocol:**
```
1. Agent queries the network for information on a topic
2. Agents with relevant knowledge respond with summaries + content CIDs
3. Requester pays per response (micro-payment via delegated budget)
4. Provenance is tracked — original knowledge creators earn royalties
```

These protocols build on the existing gossip/direct-message infrastructure
and the content index. They are essentially structured conversation patterns
with built-in payment and verification.

**Scope:** ~300 lines (protocol state machines + message handlers)

### Step 6: Agent Observability & Human Oversight

**Files:** `prsm/node/api.py`, `prsm/node/node.py`

API endpoints for humans to monitor and control their agents:

- `GET /agents` — List all agents on this node (local + remote known)
- `GET /agents/{agent_id}` — Agent status, activity log, spending
- `GET /agents/{agent_id}/conversations` — Recent agent-to-agent threads
- `POST /agents/{agent_id}/allowance` — Set/update spending allowance
- `DELETE /agents/{agent_id}/allowance` — Revoke spending authority
- `POST /agents/{agent_id}/pause` — Temporarily suspend an agent
- `GET /agents/spending` — Aggregate spend dashboard across all agents

All agent actions are logged with the principal's node ID, creating a full
audit trail. Humans can review what their agents did, how much they spent,
and who they collaborated with.

**Scope:** ~150 lines (API endpoints + status aggregation)

**Total CI-4 effort:** High | **Estimated scope:** ~1,100 lines across 6-8 files

---

## CI Execution Order

```
CI-1 (File Discovery & Retrieval) ✅ COMPLETE
 └── Content advertisement, index, request/serve, access events

CI-2 (Provenance Royalties) ✅ COMPLETE
 └── Configurable rates, multi-level provenance, source creator splits

CI-3 (FTNS Ledger Sync) ✅ COMPLETE
 └── Transaction gossip, nonce-based double-spend, balance reconciliation, earning broadcasts

CI-4 (Agentic Interoperability)
 ├── Step 1: Agent identity & delegation
 ├── Step 2: Agent registry & discovery
 ├── Step 3: Agent-to-agent messaging
 ├── Step 4: Delegated payments
 ├── Step 5: Collaboration protocols
 └── Step 6: Observability & human oversight
```

**CI-4 is the next priority.** All dependencies are now in place —
ledger sync, content discovery, and provenance royalties are operational.

**Total remaining scope:** ~1,100 lines of new code across ~6-8 files.
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
| Cross-node file discovery | DONE | Content index, gossip advertisement, direct-message request/serve, access tracking with royalties |
| Provenance royalties | DONE | Configurable rates (0.001–0.1 FTNS), multi-level provenance with parent CIDs, 70/25/5 derivative/source/network split |
| Network FTNS sync | DONE | Transaction gossip with signatures, nonce-based double-spend prevention, balance reconciliation, earning event broadcasting |

## What's Scaffolded (Needs Network Wiring)

| Component | Blocking Issue | Roadmap Section |
|-----------|---------------|-----------------|
| Agentic interoperability | No agent identity, discovery, messaging, or delegated payments | CI-4 |
| Marketplace CRUD | Service exists but missing listing creation/search methods | Phase 4 |
| NWTN orchestrator | Reasoning modules exist but top-level orchestrator missing | Phase 3 |

## Recommended Overall Priority

```
Priority 1 (Agents):         CI-4 Agentic Interoperability
Priority 2 (Features):       Phase 3 NWTN Orchestrator + Phase 4 Marketplace
Priority 3 (Test Fixes):     Phase 1-2 Quick Wins + Export Fixes
Priority 4 (Polish):         Phase 5-6 Performance + Remaining Test Skips
```

CI-4 represents approximately **1,100 lines of new code** across 6-8 files.
All infrastructure dependencies (file discovery, provenance royalties, ledger
sync) are now complete. Once CI-4 is built, PRSM will support a full
multi-agent economy — AI agents discovering each other, collaborating on
tasks, sharing content, and transacting FTNS on behalf of their human
principals, all over a decentralized P2P network.
