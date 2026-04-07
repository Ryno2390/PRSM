# PRSM Changelog

All notable changes to PRSM are documented here.

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
