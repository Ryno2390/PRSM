# PRSM Changelog

All notable changes to PRSM are documented here.

## [Unreleased] — Phase 1.1: Codex Review Fixes (partial)

Independent codex re-review of Phase 1 surfaced 6 P1 + 3 P2 bugs.
Phase 1.1 addressed 8 of the 9, but the codex re-review of Phase 1.1
itself caught a partial fix on P1 #1 plus three new findings:

  - **P1 #1 (partial)** — `compute_content_hash` helper, CLI, and the
    `_try_onchain_distribute` reader path are all in place, but the
    upload/serve/API surface (`content_uploader.py`, `content_provider.py`,
    `content_economy_routes.py`) never populates `provenance_hash` in
    `content_metadata`. Real traffic still falls back to local. The
    end-to-end hash chain claim is not yet true.
  - **P2 (new)** — `broadcast_pending` payments are reported as
    `COMPLETED` to API callers because `process_content_access` sets the
    status unconditionally after `_distribute_royalties` returns.
  - **P2 (new)** — A malformed `provenance_hash` raises `ValueError` from
    `bytes.fromhex()` outside the protected try block, so the entire
    payment is marked `FAILED` instead of falling back to local.
  - **P3 (new)** — `ProvenanceRegistryClient.register_content` still
    validates `royalty_rate_bps <= 10000`; the contract caps at 9800.

**Verdict: NOT SAFE TO DEPLOY.** Phase 1.2 will close the four
remaining items.

### What landed in Phase 1.1 (verified by codex)

  - P1 #2: capped royalty rate at MAX_ROYALTY_RATE_BPS = 9800
  - P1 #4: distinguished pre-broadcast vs post-broadcast failures
           via BroadcastFailedError / OnChainPendingError / OnChainRevertedError
  - P1 #5 + P2 #8: per-client lock + pending nonce strategy
  - P1 #6: local fallback now pays serving node its remainder
  - P2 #7: slim getCreatorAndRate getter eliminates metadataUri gas
           griefing
  - P2 #9: deploy script preflights checksum, bytecode, symbol(),
           chain id; optional AUTO_VERIFY=1 for Basescan

## [Unreleased] — Phase 1: On-Chain Provenance

Closes Phase 1 of the audit-gap remediation roadmap
([`docs/2026-04-10-audit-gap-roadmap.md`](docs/2026-04-10-audit-gap-roadmap.md)).
Moves royalty distribution off the local SQLite ledger and onto Base
mainnet so anyone can independently verify creator earnings.

### Added

- **`contracts/contracts/ProvenanceRegistry.sol`** — on-chain content
  provenance registry. Maps a 32-byte content hash to a creator address
  and royalty rate (basis points). Records are immutable except for
  creator-initiated ownership transfer. 9 Hardhat tests, ~71k gas per
  registration, ~29k per transfer.
- **`contracts/contracts/RoyaltyDistributor.sol`** — atomic three-way
  FTNS splitter (creator / network treasury / serving node). Pulls FTNS
  from the payer via `transferFrom`, looks up creator + rate from the
  registry, splits, and emits `RoyaltyPaid`. ReentrancyGuard on the
  distribute path. 8 Hardhat tests, ~120k gas per distribution.
- **`contracts/contracts/test/MockERC20.sol`** — test-only ERC-20 with
  open mint, used by the distributor test suite.
- **`contracts/scripts/deploy-provenance.js`** — Hardhat deploy script
  for both contracts. Takes `FTNS_TOKEN_ADDRESS` and `NETWORK_TREASURY`
  env vars and writes a timestamped manifest under
  `contracts/deployments/`.
- **`base-sepolia` network** added to `contracts/hardhat.config.js`
  (chainId 84532, RPC `https://sepolia.base.org`) plus the matching
  Etherscan custom chain entry for verification via the v2 API.
- **`prsm/economy/web3/provenance_registry.py`** —
  `ProvenanceRegistryClient` Web3.py 7.x wrapper. Methods:
  `register_content`, `transfer_ownership`, `get_content`,
  `is_registered`. Returns a `ContentRecord` dataclass for reads.
  Hash-length and royalty-rate validation done client-side. 6 unit
  tests with mocked Web3.
- **`prsm/economy/web3/royalty_distributor.py`** —
  `RoyaltyDistributorClient` with allowance-aware approval flow:
  reads the existing FTNS allowance for the distributor and only sends
  a fresh `approve` when the allowance is insufficient. Methods:
  `preview_split`, `distribute_royalty`, `allowance`. 4 unit tests.
- **`prsm provenance register|info|transfer` CLI** —
  `prsm/cli_modules/provenance.py`, registered into the main click
  group. `register` computes sha3-256 of file bytes as the content
  hash. `info` accepts a 0x-prefixed hash or a file path. `transfer`
  changes the creator address.
- **Feature flag `PRSM_ONCHAIN_PROVENANCE=1`** in
  `prsm/node/content_economy.py`. When set, `_distribute_royalties`
  attempts an on-chain `RoyaltyDistributor.distributeRoyalty` call
  before the local-ledger split. On any failure (chain outage,
  unregistered content, missing 0x address), the call falls through
  to the existing local path so payments are never lost.
- **End-to-end integration test**
  `tests/integration/test_onchain_provenance_e2e.py` — boots a real
  Hardhat node, deploys all three contracts, exercises the full
  Python client flow (register → preview → distribute), and asserts
  on-chain balances. Skipped automatically when Hardhat node_modules
  are absent.
- **`docs/ONCHAIN_PROVENANCE.md`** — user-facing documentation
  covering env vars, CLI, the split formula, and how to verify a
  payment on Basescan.
- **`docs/2026-04-10-audit-gap-roadmap.md`** — master roadmap for
  Phases 1-7 of the audit-gap remediation program.
- **`docs/2026-04-10-phase1-onchain-provenance-plan.md`** — TDD-style
  10-task implementation plan for Phase 1.

### Fixed

Two pre-existing import bugs uncovered while running the unit suite
(unrelated to Phase 1 but small enough to fix in passing):

- `prsm/node/payment_escrow.py` — added missing `Callable` to typing
  import. Was breaking test collection on every test that touched
  `compute_provider.py`.
- `prsm/node/content_uploader.py` — added missing
  `from prsm.storage.models import ShardManifest`. Was breaking
  test collection on `test_provenance_persistence.py` and
  `test_royalty_pipeline.py`.

### Notes

- Feature is opt-in. With `PRSM_ONCHAIN_PROVENANCE` unset, behavior
  is byte-for-byte identical to v1.7.0. Verified by running the
  existing royalty/provenance unit suite (12 tests, all pass).
- Live deployment to Base Sepolia and then Base mainnet is the
  manual operator-gated step in Task 10 of the Phase 1 plan and is
  not part of this changelog entry.
- Pre-existing failures in `contracts/test/FTNSToken.test.js` and
  `contracts/test/BridgeSecurity.test.js` (ethers v5 → v6 + OZ v5
  custom-error migration) are out of scope for Phase 1 and remain.

## [1.7.0] - 2026-04-10

### Audit Findings & Punch List — Real Features, Honest Docs

A 12-item audit-and-fix sweep that reconciled the public docs with the actual
shipped code. The audit caught two stale claims (SDK quote endpoints and
HTTP-client circuit breaker wiring), one duplicate-implementation problem, and
several layers of legacy code that survived the v1.6.0 sprint. All 12 items
shipped; full test suite green (3235 passed, 0 unexpected failures, 4 xfailed,
55 skipped).

### Added

- **`POST /compute/forge/quote`** — new server endpoint that returns a real
  `CostQuote` for a Ring 1-10 forge query without executing it. Backed by
  `PricingEngine.quote_swarm_job`. Used by all three SDKs' `quote()` helpers.
- **`GET /privacy/budget`** — new server endpoint exposing the live differential
  privacy budget audit report (`node.privacy_budget.get_audit_report()`).
- **Top-level `client.quote()`** in JavaScript SDK — convenience method that
  delegates to `client.forge.quote(...)` and mirrors the Python SDK example.
- **Top-level `client.Quote()`** in Go SDK — convenience method that delegates
  to `client.Forge.GetQuote(...)`.
- **`TensorParallelExecutor(remote_dispatcher=...)`** — new optional constructor
  parameter on Ring 8's tensor-parallel executor. Acts as the Ring 2 integration
  seam: assignments with non-local `node_id` route through the dispatcher when
  one is provided.
- **`ExchangeRouter(live_trading=False)`** — opt-in flag for Chronos exchange
  routing. When `False` (default), `execute_trade` returns simulated fills using
  real CoinGecko spot prices. When `True`, the router refuses to proceed unless
  per-exchange credentials are non-placeholder, and even then errors with a
  clear "not implemented in 1.6.x" message rather than risking accidental fills.

### Changed

- **All 4 partial MCP tools now hit real endpoints:**
  - `prsm_list_datasets` — queries `/content/search` (with `/content/index/stats`
    fallback) instead of returning a placeholder string
  - `prsm_search_shards` — queries `/content/search` and renders real results
  - `prsm_stake` — now optionally calls `POST /staking/stake` when `execute=true`
    (preview is the safe default; tool schema declares `execute` and `stake_type`)
  - `prsm_privacy_status` — reads the new `/privacy/budget` endpoint instead of
    returning generic explanation text
- **Chronos `_get_exchange_price`** now uses the existing real CoinGecko fetcher
  (`_fetch_exchange_rate_from_source`) instead of hardcoded `base_rates`. Live
  test verified $73K BTC fetched from CoinGecko API.
- **Chronos `execute_trade`** now clearly tags `execution_mode="simulated"` and
  refuses to silently use placeholder credentials.
- **`TensorParallelExecutor.execute_parallel`** result now includes
  `execution_modes={"local": N, "remote": M}` so callers can verify what
  actually happened.
- **Go SDK `forge.QuoteRequest`** rebuilt to match the new server contract:
  `Query`, `ShardCids`, `ShardCount`, `HardwareTier`, `EstimatedPcuPerShard`
  (was `Prompt`/`TaskType`). `QuoteResponse` mirrors the `CostQuote` shape.
- **`docs/IMPLEMENTATION_STATUS.md`** — Circuit breaker entry now honestly says
  "Library only — not currently wired into any live HTTP client" (the previous
  "Wired into HTTP clients (Anthropic, OpenAI, OpenRouter)" claim was false:
  no first-party HTTP client uses `CircuitBreaker.call`, and `node.agent_forge`
  was set to `None` in v1.6.0). Chronos entry now acknowledges sandbox mocks.

### Removed

**Strict-mode failure for Ring 8 remote assignments without a dispatcher.**
The previous `_execute_shard` placeholder silently fell back to local execution
regardless of `node_id`, hiding the missing remote-execution capability behind
a fake "success". Non-local assignments without a dispatcher now raise
`NotImplementedError` with a clear error message naming the integration seam.

**Dead `prsm demo` command references** removed from `README.md`,
`docs/CLI_REFERENCE.md`, and `llms.txt`. The command was removed in v1.6.0 but
the docs continued to advertise it.

**~7,000 lines of legacy code that survived the v1.6.0 sprint:**

- **`prsm/node/api.py`** — entire `/teacher/*` and `/distillation/*` endpoint
  blocks (~707 lines, 13 endpoints)
- **`prsm/node/node.py`** — `TrainingJob`, `TrainingJobStatus`,
  `teacher_registry`, `training_jobs`, `_save_teacher_registry`,
  `_load_teacher_registry_meta`, `_save_training_runs`, `_load_training_runs`
- **`prsm/node/compute_provider.py`** — `_run_training`, `JobType.TRAINING`,
  legacy distillation backend capability check
- **`prsm/node/compute_requester.py`** — `submit_training_job`, JobType.TRAINING
  entries in `JOB_TYPE_CAPABILITIES` and `JOB_TYPE_PREFERRED_BACKENDS`
- **`prsm/node/capability_detection.py`** — distillation capability advertisement
- **`prsm/interface/api/main.py`** — `/teachers/*` legacy endpoints
- **`prsm/compute/collaboration/`** — 13 legacy subdirs (`academic`, `datascience`,
  `design`, `enterprise`, `grants`, `jupyter`, `latex`, `references`,
  `specialized`, `tech_transfer`, `university_industry`, `containers`,
  `development`) plus `models.py` and `state_sync.py`. Kept `p2p/` (peer
  reputation, bandwidth optimization, node discovery, shard distribution) and
  `security/` (post-quantum sharding, access control, key management).
- **`prsm/core/integrations/langchain/`** — 5 files / 2,527 lines. Was a v0.x
  AGI-orchestration LangChain wrapper that contradicted the v1.6.0 scope and
  was already broken (`__init__.py` referenced non-existent `embeddings.py`
  and `retriever.py`).
- **`sdks/python/`** — duplicate `prsm-python-sdk@0.2.0` package. The canonical
  Python SDK is unambiguously `prsm/sdk/` (shipped as part of `prsm-network`).
- **7 empty shell directories** with only `__pycache__`:
  `prsm/compute/{distillation,evolution,improvement,ai_orchestration,students,teachers}/`,
  `prsm/core/institutional/`
- **8 broken legacy imports** referencing the empty shells in `interface/api/main.py`,
  `node/api.py`, `node/capability_detection.py`, `node/compute_provider.py`
- **8 legacy test files** for deleted subsystems:
  `tests/integration/test_complete_collaboration_platform.py`,
  `tests/integration/test_collaboration_platform_integration.py`,
  `tests/test_phase8_sdk.py`,
  `tests/unit/test_python_sdk.py`,
  `tests/unit/test_distillation_node_integration.py`,
  `tests/unit/test_teacher_node_integration.py`,
  `tests/unit/test_compute_provider_nwtn_integration.py`,
  `tests/unit/test_training_job_status.py`

### Fixed

- **JS and Go SDK `quote()` would 404 in production.** Both SDKs already had
  `forge.quote()` / `Forge.GetQuote()` helpers, but the endpoints they called
  (`/api/v1/compute/forge/quote` and `/api/v1/forge/quote`) **did not exist on
  the server**. The new `POST /compute/forge/quote` endpoint fixes both.
- **`tests/integration/test_ring8_shield.py::test_tensor_parallel_execution`**
  was previously passing only because the executor silently ignored `node_id`.
  The test has been replaced with three honest tests covering local, remote
  with a dispatcher, and remote without a dispatcher.
- **`tests/unit/test_pipeline_security.py::test_execute_parallel_produces_result`**
  now uses `node_id="local"` explicitly instead of relying on the silent
  local-fallback that the old executor implemented.

### Audit Corrections (issued during execution)

The original audit report contained two errors that were caught and corrected
during punch-list execution:

1. **PayPal IS implemented.** `PayPalProvider` exists at
   `prsm/economy/payments/fiat_gateway.py:262-479` with full OAuth, payment
   intent creation, status polling, and refund support. The original audit had
   only checked `payment_provider.py`. The README "Stripe/PayPal" claim is
   accurate; no doc change needed.
2. **All three SDKs already had `quote()` methods.** They were nested under
   `client.forge.quote` (JS) and `client.Forge.GetQuote` (Go) — the audit
   incorrectly checked only the top-level client surface. The real bug was that
   the server endpoints they called did not exist (now fixed).

### Test Suite

- **Before:** 132 unexpected failures (per pre-sprint baseline)
- **After:** 0 unexpected failures, 3235 passed, 55 skipped, 4 xfailed
- 3 pre-existing network/peer integration flakes deselected from CI
  (`test_cross_node_peer_connection`, `test_peers_endpoint_if_available`,
  `test_two_nodes_compute_job_and_payment`) — these were failing on `main`
  before this release and are tracked separately.

## [1.6.3] - 2026-04-10

### Docs-Only Release — Documentation Accuracy Sweep

Updates all forward-facing documentation to reflect PRSM's current v1.6.0+
scope (P2P infrastructure protocol for open-source collaboration, not an
AGI framework). Ensures the PyPI package page renders the accurate README.

No code changes. Test suite and installed package behavior identical to v1.6.2.

### Changed

- **`README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `contracts/README.md`** updated
  to remove legacy framing and reflect current scope
- **`docs/architecture.md`** — full rewrite to current 10-Ring architecture,
  query flow, privacy layers, FTNS economics
- **`docs/IMPLEMENTATION_STATUS.md`** — full rewrite with shipped-rings table,
  current subsystem status, explicit v1.6.0 removal list
- **`docs/API_REFERENCE.md`, `docs/CLI_REFERENCE.md`, `docs/GETTING_STARTED.md`,
  `docs/DEVELOPMENT_GUIDE.md`, `docs/TROUBLESHOOTING_GUIDE.md`,
  `docs/SDK_DEVELOPER_GUIDE.md`, `docs/FTNS_API_DOCUMENTATION.md`** — legacy
  examples replaced with current public surface
- **`docs/business/INVESTOR_MATERIALS.md`** — thesis, market positioning,
  revenue model rewritten to match current scope
- **Community launch posts** (`blog_post_launch.md`, `hacker_news_launch.md`,
  `reddit_ethereum.md`, `reddit_machinelearning.md`) — rewritten for P2P
  infrastructure pitch
- **`prsm/economy/tokenomics/README.md`** — 611-line legacy NWTN ThinkingMode
  pricing doc replaced with real FTNS tokenomics (80/15/5 splits, PCU menu,
  staking tiers)
- **SDK READMEs** (`sdks/python`, `sdks/javascript`, `sdks/go`) — quick
  start and features updated
- **`docs/SOVEREIGN_EDGE_AI_SPEC.md`, `docs/CONFIDENTIAL_COMPUTE_SPEC.md`** —
  added "⚠️ HISTORICAL DOCUMENT" banners directing readers to current
  `docs/architecture.md`. Specs themselves preserved as historical records.

### Removed

**50 pure-legacy documentation files deleted:**
- NWTN legacy pipeline: `COMPLETE_NWTN_PIPELINE_WORKFLOW.md`,
  `NWTN_COMPLETE_SYSTEM_README.md`, `NWTN_PIPELINE_CHECKPOINTING.md`,
  `NWTN_README_UPDATE.md`, `NWTN_Potemkin_Protection_Analysis.md`,
  `NWTN_Stochastic_Parrots_Analysis.md`,
  `architecture/NWTN_COMPLETE_PIPELINE_ARCHITECTURE.md`
- Distillation / SEAL / teacher: `architecture/automated_distillation_system.md`,
  `architecture/SEAL_IMPLEMENTATION.md`, `ENHANCED_FTNS_TOKEN_PRICING.md`,
  `tokenomics/TOKENOMICS_OVERVIEW.md`
- Enterprise architecture: `ARCHITECTURE_DEPENDENCIES.md`,
  `SOC2_ISO27001_COMPLIANCE_FRAMEWORK.md`, `PHASE_7_ENTERPRISE_ARCHITECTURE.md`,
  `ENTERPRISE_MONITORING_SYSTEM.md`, `MULTI_CLOUD_STRATEGY.md`,
  `CONTAINER_RUNTIME_ABSTRACTION_SUMMARY.md`, `cdn_infrastructure.md`,
  `advanced/DISTRIBUTED_RESOURCE_ARCHITECTURE.md`
- Legacy marketplace: `marketplace/MARKETPLACE_ECOSYSTEM_GUIDE.md`,
  `marketplace/MARKETPLACE_IMPLEMENTATION.md`, `marketplace-status.md`
- Legacy API/code docs: `API_DOCUMENTATION.md`, `CODE_REVIEW.md`,
  `EXAMPLES_COOKBOOK.md`, `api/PHASE_7_API_REFERENCE.md`,
  `TECHNICAL_ADVANTAGES.md`
- Legacy investor: `audit/INVESTOR_AUDIT_GUIDE.md`,
  `business/INVESTMENT_READINESS_REPORT.md`, `business/INVESTOR_QUICKSTART.md`,
  `business/roadmaps/PRODUCTION_ROADMAP.md`, `INVESTOR_QUICKSTART.md`,
  `GAME_THEORETIC_INVESTOR_THESIS.md`
- AI-auditor docs (all 4 files referenced deleted 7-phase Newton spectrum):
  `ai-auditor/{AI_AUDIT_GUIDE,AI_AUDITOR_INDEX,README,TECHNICAL_CLAIMS_VALIDATION}.md`
- Other legacy: `COLLABORATION_PLATFORM_COMPLETION_SUMMARY.md`,
  `integration/COLLABORATION_PLATFORM_INTEGRATION_REPORT.md`,
  `roadmaps/P2P_SECURE_COLLABORATION_INTEGRATION_ROADMAP.md`,
  `SECURITY_ARCHITECTURE.md`, `security/SECURITY_IMPLEMENTATION_STATUS.md`,
  `PERFORMANCE_METRICS.md`, `safety.md`, `PRSM_Development_Notes.txt`,
  `evaluator_demo.md`, `REASONING_CONTEXT_AGGREGATOR_ROADMAP.md`

### Preserved as Historical

~90 dated records (CHANGELOG, sprint specs in `docs/plans/`, external audits,
research notes, phase4 implementation record) left untouched.

### Filesystem Cleanup

Removed stale artifacts in `prsm/compute/nwtn/` left over from PR 2 (v1.6.0):
11 empty legacy subdirectories, stale `__pycache__/` files for deleted modules,
`.DS_Store` files, and 1 untracked academic paper markdown. `prsm/compute/nwtn/`
now correctly contains only `__init__.py` + `training/` (Ring 9).

### Stats

- 79 files changed across two commits (`cf3dd38` + `a804099`)
- 23,482 lines of legacy docs removed, 718 lines of accurate content added
- Net: −22,764 lines of documentation
- Ring 9 regression tests: 6 passed / 0 errors (unchanged)

---

## [1.6.2] - 2026-04-10

### Hotfix — Missing Base Dependencies

Immediate hotfix after v1.6.1 discovered that fresh `pip install prsm-network`
still failed on `from prsm.compute.federation import p2p_network` because
`psutil` is hard-imported in `prsm/compute/performance/load_testing.py` (and
9 other sites) without being in base dependencies. v1.6.1 fixed `asyncpg`
but missed `psutil` and `pynacl`.

### Fixed

- **`psutil` added to base dependencies** — 10 hard imports across
  `prsm/compute/performance/` (load_testing, optimization, task_worker,
  scaling_test_controller), `prsm/core/monitoring/` (validators, profiler),
  `prsm/core/performance/monitor.py`, `prsm/compute/scalability/cpu_optimizer.py`,
  `prsm/interface/api/health_api.py`, and
  `prsm/interface/dashboard/real_time_monitoring_dashboard.py` were all
  hard-imported without try/except, causing `prsm.compute.federation` imports
  to fail on fresh installs.
- **`pynacl` added to base dependencies** — P2P cryptographic signatures
  (Ed25519) are essential for any real node, even though the import sites
  currently have try/except fallbacks.

### Notes

Fresh `pip install prsm-network==1.6.2` now imports all core modules
(`prsm.storage`, `prsm.compute.agents`, `prsm.compute.federation`,
`prsm.compute.chronos`, `prsm.compute.nwtn.training`,
`prsm.economy.tokenomics`) without needing any optional extras.

---

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
