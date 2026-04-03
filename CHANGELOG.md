# PRSM Changelog

All notable changes to PRSM are documented here.

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
