# PRSM Audit Gap Remediation — Master Roadmap

> **Created:** 2026-04-10
> **Source:** Audit performed on PRSM v1.7.0 codebase against the meganode-bootstrap product plan
> **Status:** Phase 1 plan ready for execution. Phases 2–7 are scoped but not yet detail-planned.

## Goal

Close the seven gaps that today block PRSM from supporting the planned business model:
**P2P mesh + open protocol + FTNS-driven provenance royalties + storage/compute marketplace + meganode bootstrap on Base mainnet.**

## Audit Verdict (April 2026)

| # | Pillar | Status |
|---|---|---|
| 1 | P2P mesh (libp2p) | 🟡 Prototype — NAT/bootstrap unproven |
| 2 | FTNS token on Base | ✅ **Live** at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| 3 | Provenance royalty system | 🟡 Computed in SQLite, **not on-chain** |
| 4 | Marketplace | 🟠 Scaffold — no order book, no matching |
| 5 | Storage nodes | ✅ Production (no erasure coding / slashing) |
| 6 | Compute nodes | ✅ Production locally, ❌ remote dispatch `NotImplementedError` |
| 7 | Wallet / onboarding | 🟡 Raw Ed25519 keys, no Coinbase Wallet SDK |
| 8 | Fiat on-ramp | 🟡 Stripe/PayPal code exists, no keys, no KYC |

**Foundation is solid post-v1.6/1.7 cleanup.** What's missing is the *economic rails* that make the meganode flywheel real.

## Phasing Principle

Phases ordered by **how much each gap blocks the meganode bootstrap story**:
1. Without on-chain royalties, "trustless royalty distribution" is a lie. → **Phase 1**
2. Without remote compute dispatch, the compute mesh is a single-machine demo. → **Phase 2**
3. Without a marketplace, supply and demand cannot find each other. → **Phase 3**
4. Without consumer wallet onboarding, FTNS demand growth is gated by crypto-native users. → **Phase 4**
5. Without fiat on-ramp, meganode operators cannot realize ROI. → **Phase 5**
6. Without P2P hardening, the network falls over at scale. → **Phase 6**
7. Without storage hardening, data loss kills trust. → **Phase 7**

Each phase ships independently. Each phase produces working, demonstrable software.

---

## Phase 1 — On-Chain Provenance & Royalty Distribution

**Why first:** Meganode pitch = "earn FTNS by curating data others use." Today, royalty splits are computed in a local SQLite ledger and a node operator could rewrite history. Smart contracts on Base move the source of truth on-chain, making royalties verifiable by anyone.

**What ships:**
- `ProvenanceRegistry.sol` — content hash → creator address → royalty rate (basis points)
- `RoyaltyDistributor.sol` — pulls FTNS, splits creator/serving-node/network treasury per registry, emits `RoyaltyPaid`
- Web3.py clients (`prsm/economy/web3/provenance_registry.py`, `royalty_distributor.py`)
- Wired into `prsm/node/content_economy.py` payment flow with feature flag (`PRSM_ONCHAIN_PROVENANCE=1`)
- CLI: `prsm provenance register|info|transfer`
- Deployment to Base Sepolia → Base mainnet
- End-to-end integration test (register → use → pay → assert on-chain event)

**Detailed plan:** [`2026-04-10-phase1-onchain-provenance-plan.md`](./2026-04-10-phase1-onchain-provenance-plan.md)

**Acceptance criteria:**
- Anyone can independently verify a creator earned FTNS for content X by reading Base mainnet logs.
- Rolling back royalty payments requires a chain reorg.
- Existing batch settlement and local ledger remain functional under feature flag.

---

## Phase 2 — Remote Compute Dispatch (Ring 8 completion)

**Why second:** Ring 8 tensor-parallel sharding shipped, but `_execute_shard(node_id != "local")` raises `NotImplementedError`. Until remote nodes can accept compute jobs from other nodes, the "mesh compute network" is a single-machine demo.

**What ships:**
- gRPC or libp2p-stream RPC for shard execution requests
- `RemoteShardExecutor` — submit shard, await result, timeout, retry, fallback
- Job receipt protocol — signed proof a remote node executed correctly
- Slashing hook for nodes that accept jobs and fail to deliver
- Integration with FTNS escrow: payment locks on dispatch, releases on signed receipt
- Test: 3-node local cluster runs a sharded inference end-to-end

**Files involved:**
- `prsm/compute/model_sharding/` (Ring 8)
- `prsm/node/compute_provider.py`
- `prsm/economy/escrow/` (existing escrow → tie to Phase 1 RoyaltyDistributor)

**Estimated scope:** ~2-3 weeks

---

## Phase 3 — Marketplace Matching Engine

**Why third:** With Phases 1+2 complete, supply (compute, data, storage) is monetizable. But discovery is implicit — buyers and sellers find each other only through query routing. A real marketplace exposes listings, prices, and lets users explicitly buy compute time or data access.

**What ships:**
- Listing service: `prsm/economy/marketplace/listings.py` — datasets, model weights, compute slots, storage offers
- Order book + matching engine — limit orders, time-priority within price level
- On-chain settlement contract `Marketplace.sol` — escrow buyer FTNS, release on delivery proof
- REST API: `/api/v1/marketplace/listings`, `/orders`, `/match`
- CLI: `prsm market list|buy|sell|orders`
- Optional: minimal web UI scaffold (separate sub-phase if pursued)

**Files involved:**
- New: `prsm/economy/marketplace/` (currently empty)
- New: `contracts/contracts/Marketplace.sol`
- Modify: `prsm/storage/content_store.py` (listing hooks)

**Estimated scope:** ~3-4 weeks

---

## Phase 4 — Wallet SDK & Consumer Onboarding

**Why fourth:** Today, user onboarding requires CLI, raw Ed25519 keys, and manual FTNS contract interaction. Most users will never tolerate this. Coinbase Wallet SDK + WalletConnect lets users sign in with a familiar wallet UX.

**What ships:**
- Coinbase Wallet SDK integration — passkey login, smart-wallet contract account
- WalletConnect v2 fallback for other wallets
- Web onboarding flow: connect wallet → grant FTNS allowance → bind to PRSM node identity
- Account recovery doc for self-custody node operators (separate from consumer flow)
- `prsm/interface/onboarding/` rewritten with new flow
- Test: end-to-end wallet connect → register provenance → buy data

**Estimated scope:** ~2-3 weeks

---

## Phase 5 — Fiat On-Ramp & KYC

**Why fifth:** Meganode ROI story requires FTNS → USD conversion. Stripe/PayPal client code exists in `prsm/economy/payments/` but isn't wired with real keys, has no KYC, and Coinbase/Kraken integrations are sandbox stubs.

**What ships:**
- Selected KYC vendor integration (Persona, Sumsub, or Onfido)
- Stripe production keys + PCI scope review
- Coinbase Commerce or Coinbase Exchange API integration for FTNS↔USD swap path
- Aerodrome (Base DEX) integration for FTNS↔USDC liquidity pool
- Compliance review checklist (Howey test memo, MSB analysis)
- Withdrawal flow: user requests USD → KYC check → Stripe payout

**Estimated scope:** ~4-6 weeks (compliance-gated, not engineering-gated)

**Pre-work required:** Legal review of FTNS as security risk before this phase begins. Foundation entity formation likely needed.

---

## Phase 6 — P2P Network Hardening

**Why sixth:** libp2p prototype exists but bootstrap nodes, NAT traversal, and DHT have not been load-tested. Once Phases 1-5 attract users, these become production blockers.

**What ships:**
- Bootstrap node infrastructure (≥3 geographically distributed)
- ICE/STUN/TURN integration for NAT traversal
- Kademlia DHT replication tuning
- Connection liveness + automatic peer eviction
- Network observability dashboard
- Chaos test: 100-node simulated network with 30% churn

**Estimated scope:** ~3-4 weeks

---

## Phase 7 — Storage Hardening (Erasure Coding + Slashing)

**Why last:** Storage works for dev/staging but is a single-replica copy with no slashing. At scale, data loss is inevitable without redundancy guarantees.

**What ships:**
- Reed-Solomon erasure coding (k=6, n=10 default) in `prsm/storage/`
- Storage proof challenge/response protocol (PoR — Proof of Retrievability)
- Slashing contract `StorageSlashing.sol` — burn collateral on failed challenge
- Heartbeat enforcement
- Test: kill 4 of 10 shards, verify content still retrievable

**Estimated scope:** ~3-4 weeks

---

## Out of scope (deferred indefinitely)

- **Cross-chain bridge** — `FTNSBridge.sol` legacy template exists. Defer until multi-chain demand surfaces.
- **NWTN / AGI components** — already deleted in v1.6.0 sprint.
- **Governance contract** — out of scope until token holder base exists.

## Total program estimate

Phases 1-7 sum to **~17-25 weeks** of focused engineering, plus compliance lead time for Phase 5. Phases 1-3 (~7-10 weeks) deliver the minimum viable economic loop.

## Execution model

- One phase = one or more dedicated detailed TDD plans (this doc + Phase 1 plan are first two artifacts).
- Each phase ends with a release tag and CHANGELOG entry.
- After Phase 1, re-audit before starting Phase 2 — gaps may have shifted.
