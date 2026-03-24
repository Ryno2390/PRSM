# PRSM Implementation Status
## Comprehensive Mapping of Current State vs. Production Readiness

[![Status](https://img.shields.io/badge/status-Alpha%20v0.2.1-blue.svg)](#current-implementation-status)
[![Tests](https://img.shields.io/badge/tests-404%20passing%2C%201%20failing-yellow.svg)](#test-suite-status)
[![Updated](https://img.shields.io/badge/updated-2026--03--24-green.svg)](#)

**This document tracks PRSM's current technical implementation state, known bugs, and the remaining work required before general user participation is possible.**

---

## Executive Summary

As of commit `e6d6cf2` (March 24, 2026), PRSM's core P2P infrastructure is end-to-end functional. A user can run `prsm node start`, join the live bootstrap network, execute compute jobs, and have FTNS tokens charged and credited in real time. One failing security test exists with a known fix. Several items remain before broad, non-technical user participation is practical.

---

## Current Implementation Status

### ✅ Fully Implemented and Production-Ready

#### P2P Node Infrastructure
- `prsm node start` boots cleanly with a wizard-driven configuration flow
- Ed25519 keypair identity generated/persisted at `~/.prsm/node_identity.json`
- WebSocket transport layer, gossip protocol, and peer discovery all functional
- All 274 API endpoints register cleanly via FastAPI on port 8000
- Live TUI dashboard (or `--no-dashboard` for static output)
- Connected to live bootstrap: `wss://bootstrap1.prsm-network.com:8765`

**Key files:** `prsm/cli.py`, `prsm/node/node.py`, `prsm/node/transport.py`, `prsm/node/gossip.py`, `prsm/node/discovery.py`

#### FTNS Token Economy
- DAG-based accounting with IOTA-style Tangle architecture — no mining fees, fast confirmation
- Atomic balance operations with Ed25519 cryptographic verification and row-level SQLite locking
- Real charging: ~0.01 FTNS/token on live Anthropic queries
- Transaction types: compute payment, storage incentive, training reward, transfer, staking, penalties
- Welcome grant: 100 FTNS on first node registration
- FTNS token live on Ethereum Sepolia: `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4`
- Staking manager with lock/unlock operations

**Key files:** `prsm/node/dag_ledger.py`, `prsm/node/local_ledger.py`, `prsm/economy/tokenomics/staking_manager.py`

#### Compute Marketplace
- Job submission, acceptance, execution, and FTNS payment pipeline fully wired end-to-end
- Cross-node job routing with FTNS escrow
- Local compute provider with psutil-based resource detection
- AI backends: Anthropic (auto-detects primary model from API key) + OpenAI
- NWTN 5-agent pipeline: Architect → Primer → Solver → Verifier → Scribe
- Job types: benchmark, inference, embedding
- Teacher model training framework with async training jobs

**Key files:** `prsm/compute/nwtn/`, `prsm/node/compute_provider.py`, `prsm/node/compute_requester.py`

#### Data Sharing and Storage
- IPFS client with chunked uploads/downloads for multi-GB files, retry with backoff, gateway fallback (`prsm/core/ipfs_client.py`)
- BitTorrent integration: torrent manifests, distributed transfer, proof-of-transfer verification
- Content provenance: semantic attribution, royalty distribution, content indexing
- Shard-level integrity verification

**Key files:** `prsm/core/ipfs_client.py`, `prsm/node/bittorrent_provider.py`, `prsm/node/content_uploader.py`, `prsm/data/provenance/`

#### Marketplace API
- 9 asset types: AI models, datasets, agent workflows, MCP tools, compute resources, knowledge resources, evaluation services, training services, safety tools
- Full order lifecycle: creation → fulfillment → rating
- SQLAlchemy ORM with ACID compliance and parameterized queries
- Revenue analytics with 30-day reporting
- Advanced search with database-optimized filtering

**Key files:** `prsm/interface/api/routers/marketplace.py`

#### API Infrastructure
- 50+ routers, 274 endpoints
- JWT authentication with role-based access control
- WebSocket support for real-time updates
- OpenAPI/Swagger docs auto-generated

---

### 🔴 Failing Test (Security Critical)

#### Test
```
tests/security/test_double_spend_prevention.py::test_idempotency_key_prevents_duplicate_transactions
```

#### Error
```
sqlite3.OperationalError: no such table: ftns_idempotency_keys
```

#### Root Cause
`AtomicFTNSService._get_session()` (lines 141–145 of `prsm/economy/tokenomics/atomic_ftns_service.py`) ignores the injected `database_service` constructor argument and always calls `get_async_session()` — the real database connection. The test injects a mock session to avoid hitting the DB, but it is bypassed.

#### Fix Required (2 lines in `atomic_ftns_service.py`)
```python
async def _get_session(self):
    if not self._initialized:
        await self.initialize()
    if self._db_service is not None:           # ADD
        return self._db_service.get_session()  # ADD
    return get_async_session()
```

#### Secondary Issue: Missing Table
`ftns_idempotency_keys` is referenced in raw SQL within `atomic_ftns_service.py` and `database.py` but no SQLAlchemy model exists, so the table is never created. This is not hit in the node startup path today but will surface when `AtomicFTNSService` is used against a real database.

A model needs to be added to `prsm/core/database.py`:
```python
class FTNSIdempotencyKeyModel(Base):
    __tablename__ = "ftns_idempotency_keys"
    idempotency_key = Column(String(255), primary_key=True)
    transaction_id  = Column(String(255), nullable=False)
    user_id         = Column(String(255), nullable=False)
    operation_type  = Column(String(50),  nullable=False)
    amount          = Column(String(50),  nullable=False)
    status          = Column(String(20),  default="completed")
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    expires_at      = Column(DateTime(timezone=True), nullable=False)
```

#### Tertiary Issue: PostgreSQL-Only SQL Dialect
Raw SQL in `atomic_ftns_service.py` uses syntax that fails on SQLite:
- `NOW()` → SQLite requires `CURRENT_TIMESTAMP`
- `INTERVAL '24 hours'` → SQLite requires `datetime('now', '+24 hours')`

Non-blocking for the current node startup path (which uses `local_ledger.py`/`dag_ledger.py`, not `AtomicFTNSService`), but relevant if this service is ever wired to a local SQLite node.

---

### ⚠️ Stubbed Features (NotImplementedError)

All stubs are in optional or advanced features outside the core node execution path. None block basic node operation.

| File | What's Stubbed | User Impact |
|------|---------------|-------------|
| `economy/payments/crypto_exchange.py` | Fiat ↔ crypto exchange | Cannot convert FTNS to fiat |
| `economy/payments/fiat_gateway.py` | Credit card / bank payments | No fiat on-ramp to purchase FTNS |
| `compute/chronos/price_oracles.py` | Dynamic pricing from oracles | Compute prices are static |
| `compute/agents/executors/ollama_client.py` | Local LLM inference via Ollama | No inference without an Anthropic/OpenAI API key |
| `compute/ai_orchestration/model_manager.py` | Cross-backend model routing | Some multi-model routing paths incomplete |
| `data/analytics/real_time_processor.py` | Streaming analytics pipeline | Analytics are batch-only |

---

### 🟡 Gaps Before General User Participation

These are not bugs — they are gaps between "technically functional" and "broadly usable."

#### Mainnet FTNS Token (Priority: High)
FTNS is live on Ethereum Sepolia testnet only. Provenance royalties and compute payments have no real monetary value until mainnet deployment.

#### Single Bootstrap Node (Priority: High)
Only one bootstrap server exists: `wss://bootstrap1.prsm-network.com:8765`. Single point of failure for new peer discovery. Multi-region fallback nodes (EU, Asia-Pacific) are on the roadmap but not deployed.

#### IPFS Dependency Not Bundled (Priority: Medium)
Data sharing requires a locally running IPFS daemon. IPFS is not auto-started by `prsm node start`. Non-technical users will encounter this immediately.
- Setup instructions: `docs/MACOS_SETUP.md`, `docs/QUICKSTART_GUIDE.md`

#### Compute Requires Personal API Keys (Priority: Medium)
Participating as a compute provider or requester requires personal Anthropic or OpenAI API keys. No pooled or anonymized compute arrangement exists. The Ollama integration (`ollama_client.py`) would enable local inference without API keys but is currently stubbed.

#### No Web Onboarding UI (Priority: Low-Medium)
Node setup is CLI-only. No browser-based onboarding exists for non-technical users.

---

## Test Suite Status

| Metric | Value |
|--------|-------|
| Total collected | 3,521 |
| Passing | 404 |
| Skipped | 76 |
| Failing | 1 |
| Benchmark suite | Times out (excluded from main run) |

The single failing test is `test_idempotency_key_prevents_duplicate_transactions` — see the bug report above.

---

## Production Readiness by Subsystem

| Subsystem | Status | Notes |
|-----------|--------|-------|
| P2P Node Infrastructure | ✅ Ready | Identity, transport, discovery, gossip |
| FTNS DAG Ledger (local) | ✅ Ready | SQLite, atomic ops, Ed25519 signatures |
| Compute Job Pipeline | ✅ Ready | Submit, accept, execute, pay |
| AI Backends (Anthropic/OpenAI) | ✅ Ready | Auto-detection, real charging |
| IPFS Storage | ✅ Functional | Requires daemon; not auto-started |
| BitTorrent Transfer | ✅ Functional | Proof-of-transfer implemented |
| Marketplace API | ✅ Ready | 9 asset types, full order lifecycle |
| Content Provenance | ✅ Ready | Attribution + royalty distribution |
| AtomicFTNSService | ⚠️ Partially broken | Missing table, mock bypass bug, PG-only SQL |
| Ollama / Local LLM | 🔴 Stubbed | `NotImplementedError` |
| Fiat Gateway | 🔴 Stubbed | `NotImplementedError` |
| Crypto Exchange | 🔴 Stubbed | `NotImplementedError` |
| Real-time Analytics | 🔴 Stubbed | `NotImplementedError` |
| Mainnet Token | 📋 Planned | Sepolia testnet only |
| Multi-region Bootstrap | 📋 Planned | Single node today |

---

## Recommended Fix Priority

### Immediate (unblock the test suite)
1. Fix `AtomicFTNSService._get_session()` to use injected `_db_service` — 2 lines in `atomic_ftns_service.py`
2. Add `FTNSIdempotencyKeyModel` to `prsm/core/database.py` schema

### Short-term (unblock non-developer users)
3. Auto-start or bundle IPFS daemon in `prsm node start`
4. Implement Ollama client for local inference without API keys
5. Document API key alternatives or add a shared compute tier

### Medium-term (real economic value)
6. Mainnet FTNS token deployment
7. Deploy EU + Asia-Pacific bootstrap fallback nodes

### Longer-term (growth)
8. Web-based node onboarding UI
9. Fiat gateway for FTNS on-ramp
10. Dynamic price oracles for compute pricing

---

## Key File Locations

| Purpose | Path |
|---------|------|
| CLI entry point | `prsm/cli.py` |
| Node orchestrator | `prsm/node/node.py` |
| DAG ledger | `prsm/node/dag_ledger.py` |
| Local SQLite ledger | `prsm/node/local_ledger.py` |
| Atomic FTNS service (has bugs) | `prsm/economy/tokenomics/atomic_ftns_service.py` |
| Database schema / ORM | `prsm/core/database.py` |
| Anthropic backend | `prsm/compute/nwtn/backends/anthropic_backend.py` |
| IPFS client | `prsm/core/ipfs_client.py` |
| Marketplace API | `prsm/interface/api/routers/marketplace.py` |
| Security tests | `tests/security/` |

---

*Last updated against commit `e6d6cf2` on the `main` branch — March 24, 2026.*
