# PRSM Audit Gap Remediation — Master Roadmap

> **Created:** 2026-04-10
> **Last updated:** 2026-04-16
> **Source:** Audit performed on PRSM v1.7.0 codebase against the meganode-bootstrap product plan
> **Status:** Phase 1 in Sepolia bake-in (Day 5 of 7 as of 2026-04-16; mainnet deploy imminent). Phase 2 design + implementation plan complete; execution pending Phase 1 ship. Phase 3 (marketplace) + Phase 3.1 (batch settlement) shipped; three Phase 3.x follow-ons (MCP completion / liquidity / operator toolkit) scoped in PRSM-PHASE3-STATUS-1. Phase 4+ research track (R1-R7) tracked as parallel workstream. Phase 7 scope expanded to include content confidentiality tiers B/C. Phase 8 added for on-chain halving schedule enforcement. Hybrid tokenomics legal/governance workstream running on independent cadence.

## Goal

Close the seven gaps that today block PRSM from supporting the planned business model:
**P2P mesh + open protocol + FTNS-driven provenance royalties + storage/compute marketplace + meganode bootstrap on Base mainnet.**

## Audit Verdict (April 2026, updated 2026-04-16)

| # | Pillar | Status |
|---|---|---|
| 1 | P2P mesh (libp2p) | 🟡 Prototype — NAT/bootstrap unproven (→ Phase 6) |
| 2 | FTNS token on Base | ✅ **Live** at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| 3 | Provenance royalty system | 🟢 **Testnet deployed + source-verified; Sepolia bake-in Day 5/7; mainnet imminent** (Phase 1) |
| 4 | Marketplace | 🟠 Scaffold — no order book, no matching (→ Phase 3) |
| 5 | Storage nodes | ✅ Production (erasure coding / slashing / content confidentiality tiers B+C → Phase 7) |
| 6 | Compute nodes | 🟡 Production locally; remote dispatch design + implementation plan complete (→ Phase 2 execution) |
| 7 | Wallet / onboarding | 🟡 Raw Ed25519 keys, no Coinbase Wallet SDK (→ Phase 4) |
| 8 | Fiat on-ramp | 🟡 Stripe/PayPal code exists, no keys, no KYC (→ Phase 5) |

**Foundation is solid post-v1.6/1.7 cleanup.** Phase 1 gap is closing now; economic rails become real at mainnet deploy.

## Phasing Principle

Phases ordered by how much each gap blocks the meganode bootstrap story:

1. Without on-chain royalties, "trustless royalty distribution" is a lie. → **Phase 1** (in bake-in)
2. Without remote compute dispatch, the compute mesh is a single-machine demo. → **Phase 2** (design+plan complete)
3. Without a marketplace, supply and demand cannot find each other. → **Phase 3** (preplanning)
4. Without consumer wallet onboarding, FTNS demand growth is gated by crypto-native users. → **Phase 4**
5. Without fiat on-ramp, meganode operators cannot realize ROI. → **Phase 5**
6. Without P2P hardening, the network falls over at scale. → **Phase 6**
7. Without storage hardening + content confidentiality, regulated-industry workloads cannot use PRSM. → **Phase 7** (expanded scope)
8. Without on-chain halving enforcement, FTNS's Bitcoin-style value trajectory depends on foundation operational discipline rather than protocol code. → **Phase 8** (added 2026-04-16)

Each phase ships independently. Each phase produces working, demonstrable software.

---

## Phase 1 — On-Chain Provenance & Royalty Distribution

**Status: in Sepolia bake-in Day 5 of 7 as of 2026-04-16. Mainnet deploy imminent post-bake-in.**

**Why first:** Meganode pitch = "earn FTNS by curating data others use." Pre-Phase-1, royalty splits were computed in a local SQLite ledger and a node operator could rewrite history. Smart contracts on Base move the source of truth on-chain, making royalties verifiable by anyone.

**What shipped:**
- `ProvenanceRegistry.sol` — content hash → creator address → royalty rate (basis points). Base Sepolia: [`0x3744D1104c236f0Bd68473E35927587EB919198B`](https://sepolia.basescan.org/address/0x3744D1104c236f0Bd68473E35927587EB919198B#code) (source-verified).
- `RoyaltyDistributor.sol` — pulls FTNS, splits creator/serving-node/network treasury per registry, emits `RoyaltyPaid`. Base Sepolia: [`0x95F59fA1EDe8958407f7b003d2B089730109BD54`](https://sepolia.basescan.org/address/0x95F59fA1EDe8958407f7b003d2B089730109BD54#code) (source-verified; passed 7 rounds of codex review with SAFE TO DEPLOY verdict).
- Web3.py clients (`prsm/economy/web3/provenance_registry.py`, `royalty_distributor.py`)
- Wired into `prsm/node/content_economy.py` payment flow with feature flag (`PRSM_ONCHAIN_PROVENANCE=1`)
- CLI: `prsm provenance register|info|transfer`
- End-to-end integration test (register → use → pay → assert on-chain event)

**Planning artifacts:**
- Original plan: [`2026-04-10-phase1-onchain-provenance-plan.md`](./2026-04-10-phase1-onchain-provenance-plan.md)
- Codex review fixes (round 1): [`archive/2026-04-10-phase1.1-codex-fixes-plan.md`](./archive/2026-04-10-phase1.1-codex-fixes-plan.md) (archived — superseded by Phase 1.3)
- Codex review fixes (round 2): [`archive/2026-04-10-phase1.2-codex-rereview-fixes-plan.md`](./archive/2026-04-10-phase1.2-codex-rereview-fixes-plan.md) (archived — superseded by Phase 1.3)
- Phase 1.3 completion + Sepolia deploy + bake-in: [`2026-04-11-phase1.3-completion-plan.md`](./2026-04-11-phase1.3-completion-plan.md)
- Task 0 (pre-deploy audit) findings: [`2026-04-11-phase1.3-task0-findings.md`](./2026-04-11-phase1.3-task0-findings.md)
- Live bake-in log: [`2026-04-11-phase1.3-sepolia-bakein-log.md`](./2026-04-11-phase1.3-sepolia-bakein-log.md)

**Acceptance criteria:**
- Anyone can independently verify a creator earned FTNS for content X by reading Base mainnet logs.
- Rolling back royalty payments requires a chain reorg.
- Existing batch settlement and local ledger remain functional under feature flag.

**Remaining work post-bake-in:** mainnet deploy (Task 8), production rollout (Task 9), v1.8.0 tag (Task 10). Tracked in the Phase 1.3 completion plan.

---

## Phase 2 — Remote Compute Dispatch (Ring 8 completion)

**Status: design spec approved 2026-04-12; implementation plan complete 2026-04-12 (with Vision-doc-alignment addenda added 2026-04-14 through 2026-04-16). Execution pending Phase 1 completion.**

**Why second:** Pre-audit Ring 8 tensor-parallel sharding shipped in v0.35.0 (Rings 7-10 confidential compute), but `TensorParallelExecutor._execute_shard(node_id != "local")` raises `NotImplementedError`. Until remote nodes can accept compute jobs from other nodes, the "mesh compute network" is a single-machine demo.

**Planning artifacts:**
- Design spec: [`2026-04-12-phase2-remote-compute-design.md`](./2026-04-12-phase2-remote-compute-design.md) — brainstormed architecture, key Q&A decisions, interfaces
- Implementation plan: [`2026-04-12-phase2-remote-compute-plan.md`](./2026-04-12-phase2-remote-compute-plan.md) — task-level TDD plan with tests, file-level changes, and Vision-doc-alignment addenda (preemption handling, topology randomization, TEE attestation schema deferred-to-Phase-2.1+, tier-naming disambiguation, R7 KV/activation compression cross-reference)

**Key design decisions:**
- Transport: WebSocket MSG_DIRECT (reuses existing transport; gRPC upgrade path in Phase 6)
- Verification: tiered interface, ship receipt-only (compute verification Tier A) in Phase 2; redundant-execution (Tier B) and stake-slash (Tier C) deferred to Phase 7
- 3-node test fidelity: in-process with real transport (multi-process chaos tests are Phase 6)

**What ships:**
- `RemoteShardDispatcher` — submit shard, await result, timeout, retry, fallback
- Signed-receipt protocol (Ed25519 over job_id / shard_index / output_hash / executed_at_unix)
- Integration with FTNS escrow: payment locks on dispatch, releases on signed receipt
- Topology randomization per Vision-doc §7 addendum
- TEE attestation schema field (verification deferred to Phase 2.1+)
- Slashing hook for nodes that accept jobs and fail to deliver
- 3-node in-process integration test

**Estimated scope:** ~9 commits, ~650 LoC production, ~500 LoC tests per implementation plan.

---

## Phase 3 — Marketplace Matching Engine

**Status: ✅ SHIPPED. Phase 3 marketplace matching engine + Phase 3.1 batch settlement both delivered (see tags / commit history). Three Phase 3.x follow-on workstreams scoped as partner-handoff-ready (MCP completion / liquidity guarantee / operator toolkit). See [`2026-04-22-phase3-status-and-forward-plan.md`](./2026-04-22-phase3-status-and-forward-plan.md) (PRSM-PHASE3-STATUS-1) for the current state.**

**Why third:** With Phases 1+2 complete, supply (compute, data, storage) is monetizable. But discovery is implicit — buyers and sellers find each other only through query routing. A real marketplace exposes listings, prices, and lets users explicitly buy compute time or data access.

**Planning artifacts:**
- Phase 3 design + TDD: [`2026-04-20-phase3-marketplace-design.md`](./2026-04-20-phase3-marketplace-design.md) + [`2026-04-20-phase3-marketplace-plan.md`](./2026-04-20-phase3-marketplace-plan.md) — DELIVERED
- Phase 3.1 batch settlement: [`2026-04-21-phase3.1-batch-settlement-design.md`](./2026-04-21-phase3.1-batch-settlement-design.md) — DELIVERED
- Phase 3 status + forward plan: [`2026-04-22-phase3-status-and-forward-plan.md`](./2026-04-22-phase3-status-and-forward-plan.md) — scopes MCP completion, liquidity guarantee, operator toolkit as Phase 3.x follow-ons
- Historical preplanning stub: [`2026-04-14-phase3-preplanning.md`](./2026-04-14-phase3-preplanning.md) — superseded by the status doc

**Workstreams captured from Vision doc (beyond original audit-gap scope):**
1. **MCP server** — highest-leverage adoption lever. Exposes PRSM's query / retrieval / compute interfaces via Model Context Protocol so Claude Desktop, ChatGPT Desktop, Gemini, and any MCP-compatible client can invoke PRSM without users switching interfaces. Per `PRSM_Vision.md` Executive Summary positioning: PRSM is a complement to frontier LLMs via MCP, not a replacement.
2. **Short-term FTNS→USDC liquidity guarantee** — Aerodrome integration + auto-swap in arbitrage node software. Makes T3 cloud-arbitrage tier retail-accessible by reducing FX friction for hourly-settlement operators.
3. **Cloud provider compatibility + operator toolkit** — provider compatibility matrix, one-click node deploy (Docker + Terraform), operator dashboard with earnings / preemption / auto-swap status.

**What ships:**
- Listing service: `prsm/economy/marketplace/listings.py` — datasets, model weights, compute slots, storage offers
- Order book + matching engine — limit orders, time-priority within price level
- On-chain settlement contract `Marketplace.sol` — escrow buyer FTNS, release on delivery proof
- REST API: `/api/v1/marketplace/listings`, `/orders`, `/match`
- CLI: `prsm market list|buy|sell|orders`
- MCP server + tool registrations (new; `prsm_retrieve`, `prsm_compute`, `prsm_inference`)
- Aerodrome auto-swap integration (new)
- Operator dashboard + one-click deployment templates (new)

**Estimated scope:** ~3-4 weeks for core marketplace; expect expansion to 4-6 weeks with MCP server, auto-swap, and operator toolkit additions.

---

## Phase 4 — Wallet SDK & Consumer Onboarding

**Status: design + TDD plan drafted 2026-04-22 (see [`2026-04-22-phase4-wallet-sdk-design-plan.md`](./2026-04-22-phase4-wallet-sdk-design-plan.md)). Target Q4 2026.**

**Why fourth:** Today, user onboarding requires CLI, raw Ed25519 keys, and manual FTNS contract interaction. Most users will never tolerate this. Coinbase Wallet SDK + WalletConnect lets users sign in with a familiar wallet UX.

**What ships:**
- Coinbase Wallet SDK integration — passkey login, smart-wallet contract account
- WalletConnect v2 fallback for other wallets
- Web onboarding flow: connect wallet → grant FTNS allowance → bind to PRSM node identity
- Embedded wallet pattern (Privy / Web3Auth / Magic.link) for users who want email-only onboarding per Vision §14 crypto-UX mitigation
- `prsm/interface/onboarding/` rewritten with new flow
- USD-equivalent displays throughout UX (FTNS quantities shown only on explicit request)

**Estimated scope:** ~2-3 weeks.

---

## Phase 5 — Fiat On-Ramp & KYC

**Status: design + TDD plan drafted 2026-04-22 (see [`2026-04-22-phase5-fiat-onramp-design-plan.md`](./2026-04-22-phase5-fiat-onramp-design-plan.md)). Target Q1 2027. Compliance-gated.**

**Why fifth:** Meganode ROI story requires FTNS → USD conversion. Stripe/PayPal client code exists in `prsm/economy/payments/` but isn't wired with real keys, has no KYC, and Coinbase/Kraken integrations are sandbox stubs.

**What ships:**
- Selected KYC vendor integration (Persona, Sumsub, or Onfido)
- Stripe production keys + PCI scope review
- Coinbase Commerce or Coinbase Exchange API integration for FTNS↔USD swap path
- Compliance review checklist (Howey test under equity-investment architecture per `PRSM_Tokenomics.md` §9; MSB analysis)
- Withdrawal flow: user requests USD → KYC check → Stripe payout

**Pre-work required:** Legal review of FTNS securities classification (materially reduced to ~5-10% adverse-action probability under equity-investment architecture per Tokenomics §9.1). Foundation entity formation likely needed. Some liquidity work (Aerodrome FTNS↔USDC pool) may ship earlier in Phase 3 per Workstream 2 above.

**Estimated scope:** ~4-6 weeks engineering + compliance lead time.

---

## Phase 6 — P2P Network Hardening

**Status: design + TDD plan drafted 2026-04-22 (see [`2026-04-22-phase6-p2p-hardening-design-plan.md`](./2026-04-22-phase6-p2p-hardening-design-plan.md)). Target Q2 2027.**

**Why sixth:** libp2p prototype exists but bootstrap nodes, NAT traversal, and DHT have not been load-tested. Once Phases 1-5 attract users, these become production blockers.

**What ships:**
- Bootstrap node infrastructure (≥3 geographically distributed)
- ICE/STUN/TURN integration for NAT traversal
- Kademlia DHT replication tuning
- Connection liveness + automatic peer eviction
- Network observability dashboard
- Chaos test: 100-node simulated network with 30% churn
- gRPC-streaming upgrade path for Phase 2 compute dispatch at scale (supports >10 MB shards deferred from Phase 2)

**Estimated scope:** ~3-4 weeks.

---

## Phase 7 — Storage Hardening + Content Confidentiality + Verification Tiers

**Status: design + TDD plan drafted 2026-04-22 for §7.1 + §7.2 workstreams (see [`2026-04-22-phase7-storage-design-plan.md`](./2026-04-22-phase7-storage-design-plan.md)). §7.3 (compute verification Tier B/C) already shipped as Phase 7 / 7.1 / 7.1x under separate numbering — the storage plan scopes only the remaining storage-hardening + content-confidentiality workstreams. Target Q3-Q4 2027. Scope expanded 2026-04-15 to match `PRSM_Vision.md` §2 tiered content model.**

**Why last of the core phases:** Storage works for dev/staging but is a single-replica copy with no slashing. At scale, data loss is inevitable without redundancy guarantees. Equally important, without Tier B/C content confidentiality, PRSM cannot serve regulated-industry workloads (healthcare, legal, financial services) — which is the addressable market the Vision doc's private-inference stack targets and which Prismatica's commissioned-dataset + domain-model + clean-rooms revenue streams depend on (per `Prismatica_Vision.md` §2.5, §2.6, §3.3).

**Three interrelated workstreams shipping together (because they share cryptographic infrastructure and consensus primitives):**

### 7.1 Storage hardening
- Reed-Solomon erasure coding (k=6, n=10 default) in `prsm/storage/` — serves both Tier B/C availability and durability goals
- Storage proof challenge/response protocol (PoR — Proof of Retrievability)
- `StorageSlashing.sol` — burn collateral on failed challenge
- Heartbeat enforcement
- Test: kill 4 of 10 shards, verify content still retrievable

### 7.2 Content confidentiality tiers (added per Vision §2)
- **Tier B (encryption-before-sharding):** publisher encrypts file with AES-256-GCM before sharding. Shards contain ciphertext only. Decryption key released to consumers via on-chain key-distribution contract triggered by verified royalty payment.
- **Tier C (zero-knowledge content):** encryption + Reed-Solomon erasure coding + Shamir-split decryption keys. K-of-N fragment reconstruction threshold + M-of-N key-share threshold. Reconstructing content requires crossing both thresholds.
- Shared cryptographic infrastructure: AES-256-GCM, Shamir Secret Sharing, on-chain key-distribution contract with payment-gated release, client-side encryption pipeline, decryption path for authenticated consumers.
- Until Phase 7 ships, content requiring confidentiality guarantees stronger than Tier A should not be published on PRSM (disclosed in Vision §2 and Risk Register G4).

### 7.3 Verification tiers (compute-side)
- Redundant-execution consensus (compute verification Tier B) — multiple providers execute same shard, consensus on output
- Stake-slash verification (compute verification Tier C) — providers stake FTNS, misbehavior slashed via on-chain `ComputeSlashing.sol`
- Required for TEE attestation gating to become effective (Phase 2 ships the schema field; Phase 7 ships enforcement logic)

**Terminology note — two independent Tier A/B/C systems:**

- **Compute verification tiers (A/B/C)** apply to the verification-strength spectrum for remote compute dispatch. Used in Phase 2 docs and in codebase (`remote_dispatcher.py`, `shard_receipt.py`).
- **Content confidentiality tiers (A/B/C)** apply to the confidentiality-strength spectrum for stored content. Used in `PRSM_Vision.md` §2 and §7.

The naming collision is unfortunate but the concepts are orthogonal. When disambiguation is needed, prefer fully-qualified terms "compute verification Tier A/B/C" and "content confidentiality Tier A/B/C" rather than bare "Tier A/B/C." Phase 7 plan documentation will include an explicit section confirming the two tier systems do not share code paths — they are independent subsystems co-located in the same phase delivery for scheduling convenience, not logical coupling.

For the **full disambiguation reference** — including the third independent "Tier" system (hardware supply tiers T1-T4 per `PRSM_Vision.md` §6), Ring vs Phase numbering, Prismatica vs Foundation, and the compensation-only vs bonding-curve pivot flag — see [`glossary.md`](./glossary.md).

**Estimated scope:** ~8-12 weeks (up from original 3-4-week estimate due to Tier B/C expansion).

---

## Phase 8 — On-Chain Halving Schedule Enforcement (Added 2026-04-16)

**Status: scope document ratified 2026-04-16. Implementation deferred to Phase 8 execution slot.**

**Why:** `PRSM_Tokenomics.md` §4 specifies a Bitcoin-style halving schedule (4-year epochs, 0.5 halving factor) as one of four mechanisms producing FTNS's value trajectory. Currently, the halving schedule is enforced via foundation operational policy (see Risk Register entries C5/C6/C7). Phase 8 migrates enforcement to on-chain smart contracts, eliminating foundation operational discretion over rates and making the schedule trust-coded rather than trust-promised. The hybrid operational-to-on-chain approach (Option C) was selected so the operational period (Epochs 1-2, years 0-8) produces real-world data that informs contract design before committing to immutable parameters.

**Target timing:** Q4 2028 (end of Epoch 1), before Epoch 2 halving takes effect so the first on-chain-enforced halving is a real mechanism rather than a no-op. Earliest viable Q2 2028; latest acceptable Q2 2029.

**What ships:**
- `EmissionController.sol` — new contract minting FTNS from governance-authorized allocation (up to 1B cap) at halving-controlled rates. Immutable halving schedule (4-year epochs, 0.5 factor); per-epoch rate limits; deterministic public epoch-rate calculation; monotone-decreasing invariant enforced at contract level.
- `CompensationDistributor.sol` — pulls from `EmissionController` and distributes to compensation pools per governance-set weights (creator royalty bonus pool, node operator pool, contributor grant pool).
- Migration from operational-policy-only enforcement (Epochs 1-2, years 0-8) to strict on-chain enforcement. Foundation genesis allocation continues operational-policy distribution in parallel.

**Planning artifacts:**
- Scope doc + governance safeguards + timeline: [`2026-04-16-halving-schedule-implementation-plan.md`](./2026-04-16-halving-schedule-implementation-plan.md)
- Engineering design + TDD plan (9-task breakdown, contract interfaces, risk register): [`2026-04-22-phase8-design-plan.md`](./2026-04-22-phase8-design-plan.md) (drafted 2.5 years ahead of target execution; parameter values calibrate on Epoch 1 operational data)

**Dependencies:** Phases 1-7 shipped and stable; ≥2 years of Epoch 1 operational-policy data; foundation governance structure finalized; security-audit firms engaged (min 2 audits + formal verification of mint paths); multi-stakeholder testnet exercise pre-mainnet.

**Safeguards:** 90-day advance notice for changes; 30-day stakeholder comment period; 75% supermajority threshold (above ordinary 60%); multi-class voting if classes are defined; forkability as ultimate kill-switch.

---

## Parallel Workstreams (Not Phase-Aligned)

These run on independent cadence from the engineering phases:

### Phase 4+ Research Track
[`2026-04-14-phase4plus-research-track.md`](./2026-04-14-phase4plus-research-track.md) — catalogues deferred research questions with named promotion-to-engineering triggers. Current items:

- **R1:** FHE for private inference — watch signals: Zama fhEVM, Intel HERACLES, TFHE-rs LLM benchmarks
- **R2:** MPC for sharded inference — integration research, moderate effort if pursued
- **R3:** Activation-inversion attack characterization under PRSM-specific threat model
- **R4:** Per-provider supply caps and geographic diversity incentives — governance parameter tuning
- **R5:** Tier C hardening against majority collusion — threshold-FHE and related active research
- **R6:** Post-quantum signatures — defer until NIST finalization + Ethereum L2 migration
- **R7:** KV/activation compression for consumer-edge inference (added 2026-04-16) — directly targets Risk Register G3's 9000× bandwidth gap via data-oblivious quantization (TurboQuant / PolarQuant / QJL lineage verified 2026-04-16)

Review cadence: semi-annual. Next review: 2026-10-14.

### Hybrid Tokenomics Legal/Governance Track
[`2026-04-14-hybrid-tokenomics-legal-tracking.md`](./2026-04-14-hybrid-tokenomics-legal-tracking.md) — tracks the 5-gate workflow for ratifying the Tokenomics §11 hybrid model (reduced 50 bps tax + 15-30% foundation equity in Prismatica). Gates:

1. **Internal alignment** (founders) — open; blocked on founder time + Prismatica founder conversation
2. **Prismatica founder alignment** — blocked on Gate 1
3. **Legal consultation** — blocked on Gate 2; estimated $100K-$300K spend, 3-6 months
4. **Governance ratification** — blocked on Gate 3
5. **Smart-contract parameter update** — blocked on Gate 4

Fallback: if any gate fails, PRSM continues with 2% pure-tax baseline. No shipped functionality depends on this workstream.

### Halving Schedule Operational Policy (pre-Phase-8 enforcement)
Operational policy effective immediately upon foundation formation; specified in the Phase 8 scope doc. Foundation applies halving rates off-chain until Phase 8 on-chain enforcement ships. Risk Register C5 tracks operational compliance; C6 tracks Epoch 1 rate-calibration risk; C7 tracks governance-capture risk.

---

## Numbering History (Rings → Phases)

**Pre-v1.6 era:** Work was organized as "Rings" 1-10. Ring plan artifacts were archived to [`docs/archive/`](./archive/) on 2026-04-16 (see `docs/archive/README.md`); the `docs/plans/` directory retains only still-active planning material (libp2p, native-storage). Rings 7-10 shipped as confidential compute in v0.35.0 (TEE runtime, DP noise, tensor-parallel model sharding, security hardening). Ring 8's tensor-parallel sharding shipped for single-machine execution; its remote-dispatch piece is exactly what Phase 2 of the current audit-gap roadmap completes.

**Post-v1.6 era (April 2026+):** The v1.6 scope-alignment sprint (shipped 2026-04-09) deleted ~210K LoC of legacy AGI-framework code (NWTN, teachers, distillation, self-improvement) and confirmed PRSM as a P2P infrastructure protocol for open-source collaboration, not an AGI framework. The April 2026 audit produced this roadmap's numbered Phases 1-7, later extended with Phase 8 (halving enforcement).

Phase numbering is independent of Ring numbering; "Ring 8 tensor-parallel sharding" being referenced in current Phase 2 docs is not a numbering collision — it's the earlier-era shipped work that the current Phase 2 completes via remote dispatch.

The Ring artifacts in `docs/archive/` are preserved as historical record of shipped and superseded work. New work uses Phase numbering exclusively.

---

## Out of Scope (Deferred Indefinitely)

- **Cross-chain bridge** — `FTNSBridge.sol` legacy template exists. Defer until multi-chain demand surfaces.
- **NWTN / AGI components** — deleted in v1.6.0 sprint; confirmed out of scope per Vision-doc scope clarification. PRSM is infrastructure protocol, not AGI framework.
- **Standalone governance contract** — out of scope until token holder base exists. Hybrid tokenomics governance ratification (Track Gate 4) may introduce lightweight voting mechanics earlier.
- **Full-scale bonding-curve / public FTNS sale** — architecturally removed per equity-investment pivot (see `PRSM_Tokenomics.md` §3). FTNS is compensation-only; bootstrap capital flows through Prismatica equity (Reg D 506(c)). No return to this design is planned.

---

## Total Program Estimate

**Phases 1-7:** ~17-25 weeks of focused engineering plus compliance lead time for Phase 5 and expanded scope for Phase 7 (Tier B/C + erasure coding + slashing adds ~5-8 weeks beyond original 3-4-week estimate).

**Phases 1-3** (~7-10 weeks) deliver the minimum viable economic loop.

**Phase 8:** separate Q4 2028 target; dependent on Phases 1-7 shipped and stable plus ≥2 years of Epoch 1 operational data.

Phases 1-7 deliver what `PRSM_Vision.md` §13 describes as the roadmap through Q3-Q4 2027. Phase 8 is the 2028 addition for halving enforcement per the halving implementation plan.

## Execution Model

- One phase = one or more dedicated detailed TDD plans; the plan documents are linked from each phase section above.
- Two-stage planning for complex phases: **design spec** (brainstormed, validated) → **implementation plan** (TDD-structured, file-level detail). Phase 2 established this pattern.
- Each phase ends with a release tag and CHANGELOG entry.
- After each phase ships, re-audit before starting the next — gaps may have shifted.
- Parallel workstreams (research track, legal gates, halving implementation) run on their own cadence independent of phase rhythm.
- Vision-doc-derived scope changes that arrive mid-phase are captured as **addenda** to the active plan (pattern established by the Phase 2 plan's multiple 2026-04-14 / 2026-04-15 / 2026-04-16 addenda) rather than triggering mid-flight replan.
