# PRSM Changelog

All notable changes to PRSM are documented here.

## [1.6.1] - 2026-04-10

### Polish Release — Test Debt Cleanup + Dead Code Removal

Follow-up polish release after v1.6.0's ~210K LoC scope alignment cleanup.
Addresses residual test debt, dead code, and dependency issues discovered
during post-release verification.

### Fixed

- **v1.5.0 IPFS migration stragglers**: test files still used old `cid=`,
  `required_cids=`, `shard_cid=`, `ipfs_cid=`, `parent_cids=`, `manifest_cid=`
  kwargs that were renamed to `content_id=` / `required_content_ids=` /
  `shard_content_id=` / `parent_content_ids=` / `manifest_content_id=` in
  commit `ceba564` (v1.5.0 native storage migration). Affected test files
  (13): `test_storage_proofs.py`, `test_content_economy.py` (node + e2e),
  `test_swarm_models.py`, `test_swarm_coordinator.py`, `test_mobile_agent_models.py`,
  `test_royalty_pipeline.py`, `test_provenance_persistence.py`,
  `test_ring2_dispatch.py`, `test_ring3_swarm.py`, `test_ring_cross_node.py`,
  `test_tier2_auth_consensus.py`, `test_enhanced_p2p_network_p1_tranche3.py`,
  `test_models.py` (FTNSTransaction `ipfs_cid` → `content_id`).
- **v1.5.0 production stragglers**: `SwarmCoordinator` still constructed
  `SwarmJob(shard_cids=...)` / `AgentManifest(required_cids=...)` /
  `ShardAssignment(shard_cid=...)` after the dataclasses were renamed;
  `ContentUploader._platform_royalty_transfer()` signature still used
  `cid=` but one caller had already been renamed to `content_id=`;
  `ProvenanceQueries.upsert_provenance` / `load_all_for_node` / `get_provenance`
  in `prsm/core/database.py` still round-tripped the legacy dict key names,
  so `content_uploader._hydrate_from_db()` was silently broken. All aligned
  to the post-v1.5.0 naming (SQL column names in the `content_provenance`
  table retain their legacy form — no migration required).
- **ContentEconomy method signature migration**: test calls to
  `process_content_access`, `track_content_upload`, `update_replication_status`,
  `request_content_retrieval`, `index_content_embedding`,
  `_resolve_provenance_chain`, `_resolve_parent_creators` updated to match
  the post-v1.5.0 `content_id=` / `parent_content_ids=` parameter names.
- **StorageProofVerifier / StorageProver / StorageChallenge / StorageProof**:
  test calls updated to match post-v1.5.0 `shard_hash=` (not `cid=`)
  constructor parameters; `StorageProver` assertion replaced
  `.ipfs_api_url` with `.content_client` (IPFS-agnostic client abstraction).
- **UploadedContent attribute access**: tests now use `.content_id` instead
  of `.cid`; `ProvenanceChain.original_content_id` instead of `.original_cid`.
- **AgentManifest / MobileAgent / ProviderBid / ShardAssignment**: test
  kwargs updated to match current `required_content_ids` / `shard_content_id`
  field names.
- **`asyncpg` added to base dependencies**: fresh `pip install prsm-network`
  now works without needing optional extras. Previously
  `prsm.compute.performance.database_optimization` hard-imported asyncpg
  at module load, causing transitive imports via `prsm.compute.federation`
  to fail on a bare install.
- **Missing `prsm/api/__init__.py`**: the `prsm/api` directory contained
  route modules but lacked `__init__.py`, so
  `test_python_package_structure` flagged it. Added the missing file.
- **`FTNSTransaction.content_id`**: `test_models.py::test_ftns_transaction_optional_fields`
  still referenced the deleted `ipfs_cid` field — aligned to `content_id`.
- **`ModelShard.model_content_id`**: `test_enhanced_p2p_network_p1_tranche3`
  still passed the legacy `model_cid` kwarg — renamed to `model_content_id`.

### Removed

- **Stale Ring 5 AgentForge tests**: `tests/e2e/test_forge_endpoint.py`
  (entire file), `test_sprint7_ux.py::TestDetectAvailableBackends` (whole
  class), `test_sprint7_ux.py::test_backend_detection_used_in_cli_startup`,
  `test_phase5_completeness.py::test_breakthrough_mode_exports`,
  `test_ring6_hardening.py::test_agent_forge_command_exists`, Ring 5
  test functions from `test_ring_cross_node.py`, and `AgentForge`
  references from "all rings import" tests in `test_ring6_polish.py`,
  `test_ring7_vault.py`, `test_ring8_shield.py`, `test_ring10_fortress.py`.
  Ring 5 was deleted in v1.6.0 as part of the legacy AGI framework removal.
- **Stale marketplace CLI test**: `tests/unit/test_marketplace_cli.py`
  (entire file) — `prsm marketplace list / buy / list-dataset` CLI commands
  were removed in v1.6.0 when the legacy marketplace was pruned. Also
  deleted `test_pricing_advanced.py::TestCLIListDataset::test_command_exists`.
- **Stale v1.5.0 provider-side self-credit tests**:
  `test_compute_provider.py::test_execute_benchmark_job` and
  `test_node_self_compute.py::test_self_compute_records_earnings` updated
  to reflect the v1.6.0 design: `compute_provider._execute_job` no longer
  self-credits — payment flows through requester-side escrow release. The
  stale assertions expected legacy self-credit behavior.
- **Stale governance safety_monitor patch**:
  `test_governance_persistence.py::test_create_proposal_persists_to_db`
  patched a `voting_system.safety_monitor` attribute that was removed in
  v1.6.0 along with the legacy AGI safety framework.
- **Orphan StakingProgram classes** in `prsm/compute/chronos/models.py`
  (`StakingProgram`, `StakePosition`, `StakingAuction`, `StakingBid`,
  `StakingProgramStatus`, `CHRONOSStakingRequest`). Unused after
  `staking_integration.py` was deleted in v1.6.0.
- **Orphan pydantic settings** in `prsm/core/config.py`
  (`backend_primary`, `backend_fallback_chain`, `backend_timeout_seconds`).
  Unused after NWTN backends were deleted in v1.6.0.
- **Stale build artifacts** from repo root: `.obsidian/` (editor state),
  `prsm_network-0.37.0/`, `prsm_network-1.5.0/` (old build directories),
  and eight `real_world_scenario_results_*.json` test output files.

### Added

- **UI mockup files** (`prsm/ui_mockup/*`, `prsm/ui_assets/`): HTML / CSS /
  JS prototype for the P2P dashboard, shard visualization, and security
  indicators. Previously untracked; now part of the repo.
- **`.gitignore` entry for `.obsidian/`**: editor state no longer shows
  up as untracked on machines running the Obsidian knowledge-base plugin.

### Test Suite Health

- Before v1.6.1: 132 failed / 3275 passed / 60 skipped / 4 xfailed / 0 errors
- After v1.6.1:    3 failed / 3378 passed / 60 skipped / 4 xfailed / 0 errors
- The 3 remaining failures are pre-existing integration/e2e flakiness in
  multi-node P2P bootstrap tests (`test_cross_node_peer_connection`,
  `test_peers_endpoint_if_available`, `test_two_nodes_compute_job_and_payment`)
  that depend on live P2P connectivity not available in the isolated test
  env. They were failing at v1.6.0 and are unrelated to test debt or
  scope alignment.
- Ring 9 regression gate: 6 passed / 0 errors (preserved).

### Dev Notes

- `prsm/compute/federation/p2p_network.py` / `enhanced_p2p_network.py`
  orphan helpers (`_select_hosting_peers`, `_store_shard_on_peers`,
  `_select_execution_peers`, `_find_hosting_peers_via_dht`,
  `_distribute_shard_to_peers`, `_find_execution_peers_via_dht`,
  `_execute_task_on_peer_rpc`, `_store_shard_metadata_in_dht`) were
  **kept** in v1.6.1: investigation showed that
  `test_enhanced_p2p_network_p1_tranche2.py` and `_tranche3.py` exercise
  `_find_execution_peers_via_dht` and `_execute_task_on_peer_rpc`
  directly as unit tests, so they are no longer orphan. Flagged for a
  future release to decide whether to delete tests + methods together.

---

## [1.6.0] - 2026-04-09

### Changed - Scope Alignment Release

PRSM's scope has been formally narrowed to match the current product thesis:
a P2P infrastructure protocol for open-source collaboration, not an AGI
framework. This release removes ~210K lines of legacy code from earlier
conceptions and fixes the real bugs that legacy was hiding.

### Removed

**Legacy AGI Framework (entire subsystems):**
- Old NWTN orchestrator and meta-reasoning subsystems (meta_reasoning_engine,
  hybrid_integration, complete_system, voicebox, agent_forge, hybrid_architecture,
  chemistry_hybrid_executor, backends, bsc, team, whiteboard, synthesis,
  openclaw, reasoning, engines, architectures, knowledge_graph) — Ring 9
  (NWTN training pipeline) preserved for future fine-tuned NWTN model
- Teacher model framework (prsm/compute/teachers/, distillation/, evolution/,
  improvement/, students/, candidates/, ai_orchestration/)
- Benchmarking, evaluation, validation, network subsystems
- 5-layer agent framework (architects, compilers, prompters, routers,
  executors) from prsm/compute/agents/ — WASM mobile agent runtime preserved
- AGI safety circuit breakers (prsm/core/safety/)
- Enterprise/institutional infrastructure (prsm/core/institutional/,
  enterprise/, compliance/, teams/)
- AI marketplace (prsm/economy/marketplace/, prsm/economy/tokenomics/marketplace.py)
- Coordinated multi-agent collaboration (prsm/collaboration/)
- Legacy top-level modules (knowledge_system, demo, performance, sdks,
  query, response, learning, nlp, optimization, dev_cli)
- AI-improvement governance (prsm/economy/governance/proposals.py)
- Legacy API routers: distillation_api, cdn_api, marketplace (3 variants),
  recommendation_api, reputation_api, monitoring_api, compliance_api,
  teams_api, budget_api legacy endpoints
- CLI command groups: `prsm teacher`, `prsm nwtn`, `prsm distill`,
  `prsm marketplace`, `prsm reputation`, `prsm nwtn agent-team`,
  `prsm agent forge` (~2600 lines of CLI wiring to deleted backends)
- MCP tool `prsm_decompose` (depended on deleted agent_forge)
- `prsm-dev` bootstrap CLI (generated NWTNOrchestrator sample code)

**Legacy files within kept subsystems:**
- prsm/compute/federation/: distributed_evolution, distributed_rlt_network,
  knowledge_transfer, phase5_demo, distributed_model_registry, model_registry
- prsm/compute/chronos/: enterprise_sdk, staking_integration, treasury_provider
- prsm/core/monitoring/: rlt_performance_monitor, enterprise_monitoring
- prsm/node/node.py: _NodeContextAdapter, _NodeFTNSAdapter, _NodeIPFSAdapter,
  _NodeModelRegistryAdapter (shims that bridged node subsystems into the
  NWTN orchestrator interface)
- prsm/core/config.py: get_backend_config + backend_config property
  (referenced deleted prsm.compute.nwtn.backends)

### Fixed

- Pydantic v2 migration in the API layer (Field `regex=` → `pattern=`)
- ContentStore wiring in content_provider.py and storage_provider.py
- StorageProofVerifier / StorageProver / ContentUploader: removed obsolete
  `ipfs_client` and `ipfs_api_url` kwargs (v1.5.0 IPFS migration stragglers)
- Removed stale test skip markers hiding passing tests
- 14 stale test import errors eliminated by deleting legacy test files:
  test_phase9_completeness, test_system_resource_integration,
  test_section28_integrations, test_release_scripts,
  test_node_model_registry; plus updating test_mcp_server and
  test_content_upload to the new storage interface

### Migration Notes

No public API changes for in-scope modules. If you were importing from any of
the removed legacy subsystems, you were depending on code that was never in
the current product scope. Migration paths:
- For distributed AI workloads: use third-party LLMs via MCP
- For content storage: use `prsm.storage` (native since v1.4.0)
- For P2P networking: use `prsm.compute.federation` (mesh + consensus)
- For WASM agent dispatch: use `prsm.compute.agents` (dispatcher/executor)
- For governance: use `prsm.governance` (protocol governance)

---

## [0.35.1] - 2026-04-07

### Added — Phase 1: Sovereign-Edge AI (Rings 1-6)
- **Ring 1 (The Sandbox):** WASM runtime with Wasmtime sandbox, hardware profiler with TFLOPS/thermal detection
- **Ring 2 (The Courier):** Mobile agent dispatch with gossip-based bidding, escrow settlement
- **Ring 3 (The Swarm):** Semantic vector sharding, parallel map-reduce across data shards
- **Ring 4 (The Economy):** Hybrid pricing (PCU menu + data market), prosumer staking tiers, yield estimation
- **Ring 5 (The Brain):** LLM-powered agent forge with task decomposition, 5 MCP tools for external LLMs
- **Ring 6 (The Polish):** Dynamic gas pricing, RPC failover, settler signature verification, CLI commands

### Added — Phase 2: Confidential Compute (Rings 7-10)
- **Ring 7 (The Vault):** TEE runtime abstraction, differential privacy noise injection (configurable ε)
- **Ring 8 (The Shield):** Tensor-parallel model sharding, randomized pipeline assignment, collision detection
- **Ring 9 (The Mind):** NWTN training pipeline with JSONL export, model registry and deployment service
- **Ring 10 (The Fortress):** Integrity verification, privacy budget tracking, hash-chained audit log

### Added — Pricing Infrastructure
- Revenue split engine (80% data owner / 15% compute / 5% treasury)
- Data listing marketplace with stake-based access control
- Spot market arbitrage with automatic price adjustment

### Added — End-to-End Pipeline
- `/compute/forge` API endpoint wiring full Ring 1-10 pipeline
- `prsm compute run --query` CLI for forge-powered queries
- Live verified with NVIDIA Nemotron 120B via OpenRouter
- 367 tests across 33 test files, all passing

## [0.25.0] - 2026-04-03

### Added - Phase 6: Governance & Staking (L2-Style)

**Settler Registry:**
- `SettlerRegistry` - L2-style staking for batch settlement security
  - **The Bond**: 10K FTNS minimum stake to become a settler
  - **The Multi-Sig**: 3-of-N signatures required for batch approval
  - **The Challenge**: Public ledger export for fraud detection
  - Governance-based slashing for proven misconduct

**Security Model:**
- Settlers stake FTNS as "skin in the game" for settlement rights
- Multi-signature approval prevents single-point-of-failure
- Ledger export enables community audit and fraud proofs
- Slashing via governance vote for fraudulent batch submissions

**API Endpoints:**
- `POST /settler/register` - Register as settler with bond
- `GET /settler/{settler_id}` - Get settler details
- `GET /settler/list/active` - List active settlers
- `POST /settler/unbond` - Initiate unbonding (30-day lock)
- `POST /settler/batch/sign` - Sign pending batch
- `GET /settler/batch/pending` - List batches awaiting approval
- `GET /settler/ledger/export` - Export ledger for audit
- `POST /settler/slash/propose` - Propose settler slashing
- `GET /settler/stats` - Registry statistics

**Integration:**
- Wired `SettlerRegistry` to `PRSMNode` for batch approval workflow
- Callback triggers settlement when multi-sig threshold reached
- Settlers can unbond after 30-day lock period

**Tests:**
- `tests/node/test_settler_registry.py` - Unit tests for settler lifecycle
- `scripts/test_settler_integration.py` - End-to-end integration test

### Changed
- Updated version from 0.24.0 to 0.25.0 across 8 files

## [0.24.0] - 2026-04-03

### Added - Phase 4: Storage Provider + IPFS Content Economy

**Core Modules:**
- `ContentEconomy` - Payment processing and royalty distribution
  - Phase 4 royalty model: 8% original creator, 1% derivative creators, 2% infrastructure
  - Legacy royalty model: 70% creator, 25% sources, 5% network
  - Automatic provenance chain resolution for derivative content
- `VectorStoreBackend` - Multi-backend vector store for semantic search
  - pgvector (PostgreSQL) support with IVFFlat indexing
  - Qdrant support with filtering
  - In-memory backend for testing
  - Configurable embedding dimensions and similarity metrics
- `MultiPartyEscrow` - Batch royalty settlements for gas efficiency
  - Accumulates royalties per creator
  - Batches on-chain transfers to reduce gas costs
  - Configurable thresholds (min batch size, min value, max age)
  - Background auto-settlement loop

**API Endpoints:**
- `POST /content-economy/access` - Process content access payment
- `GET /content-economy/payment/{payment_id}` - Get payment status
- `POST /content-economy/retrieval` - Request content retrieval with bidding
- `GET /content-economy/retrieval/{request_id}` - Get retrieval status
- `POST /content-economy/search` - Semantic search via vector DB
- `POST /content-economy/index/{cid}` - Index content embedding
- `GET /content-economy/replication/{cid}` - Get replication status
- `POST /content-economy/replication/{cid}/ensure` - Ensure minimum replicas
- `GET /content-economy/royalty/{cid}` - Get royalty info for content
- `GET /content-economy/stats` - Content economy statistics
- `GET /content-economy/models` - List royalty models

**Integration:**
- Wired `ContentEconomy` to `StorageProvider` for replication tracking via storage proofs
- Wired `ContentEconomy` to `ContentProvider` for payment on retrieval
- Wired `ContentEconomy` to `ContentUploader` for initial replication tracking
- Added vector store initialization to `PRSMNode`
- Added multi-party escrow to `PRSMNode` lifecycle

**Tests:**
- `tests/node/test_content_economy.py` - Unit/integration tests
- `tests/e2e/test_content_economy_e2e.py` - End-to-end multi-node tests

### Changed
- Updated version from 0.23.0 to 0.24.0 across 8 files

## [0.23.0] - 2026-03-XX

### Added - Phase 2: Multi-Node Federation
- Multi-node P2P federation verified
- Node B connects to primary via WebSocket
- Jobs route via gossip protocol
- Ed25519 identity handshake
- On-chain FTNS escrow system verified

### Changed
- Always call `node.initialize()` before `node.start()`
- Port conflict detection with `lsof -ti :PORT`
- Use `asyncio.to_thread` for sync web3 calls

## [0.22.0] - 2026-03-XX

### Added - Phase 1: Foundation
- Bootstrap server at `wss://bootstrap1.prsm-network.com:8765`
- FTNS token on Base mainnet (`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`)
- Real AI inference via Anthropic + OpenAI
- Python/JS/Go SDKs
- Interactive setup wizard (`prsm node start --wizard`)

---

## Release Naming Convention

- **Major (X.0.0)**: Breaking changes, major milestones
- **Minor (0.X.0)**: New features, phase completions
- **Patch (0.0.X)**: Bug fixes, minor improvements

## Roadmap

| Phase | Version | Status | Focus |
|-------|---------|--------|-------|
| Phase 1 | 0.22.0 | ✅ Complete | Foundation, bootstrap, FTNS |
| Phase 2 | 0.23.0 | ✅ Complete | Multi-node federation, escrow |
| Phase 3 | TBD | 🔄 Paused | NWTN orchestration, OpenRouter |
| Phase 4 | 0.24.0 | ✅ Complete | Content economy, royalties, vector search |
| Phase 5 | TBD | 📋 Planned | Agent collaboration, multi-agent workflows |
| Phase 6 | TBD | 📋 Planned | Governance, staking |
| Phase 7 | TBD | 📋 Planned | Production hardening, CLI UX |
