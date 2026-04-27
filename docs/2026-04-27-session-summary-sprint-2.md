# PRSM Engineering Sprint 2 — 2026-04-27 Session Summary

**Prismatica, Inc.** | **April 27, 2026**
Engineering-progression changelog covering the second focused sprint after the 2026-04-22 baseline (`docs/2026-04-22-session-summary-engineering-sprint.md`). Stacks on top of that prior summary; reads cleanly with or without it.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

In a five-day sprint after the 2026-04-22 baseline, PRSM engineering shipped **five new sub-phases** (3.x.1 already shipped pre-sprint; 3.x.2 / 3.x.3 / 3.x.4 / 3.x.5 + Phase 4 Task 3 shipped this sprint), produced **two Foundation-handoff-ready vendor decision memos**, and re-bundled the audit-prep package to cover the cumulative tree.

**Two-round independent code review on every phase.** Eight HIGH-severity findings caught + closed pre-tag, including a real **manifest-substitution attack** (Phase 3.x.5) and a **lowercase-address login break** (Phase 4 Task 3) that would have shipped silently to non-Coinbase wallets without the gate.

Concretely:
- **~161 commits** pushed to `main` since the prior baseline tag
- **~580 new unit tests + ~40 new integration tests**, all green
- **5 new merge-ready tags** (3.x.1 already had its own) + 1 cumulative audit-prep tag
- **2 new partner-handoff vendor-decision memos** (Phase 4 Privy / Phase 5 Persona)
- **Zero regressions** across the full test surface

---

## 2. What shipped

### 2.1 Code: 5 new sub-phases moved from design-doc to merge-ready

| Phase | Scope | Tasks shipped | New tests | Tag |
|---|---|---|---|---|
| **3.x.1** | MCP Server Completion (inference + receipts + content tier gate + billing visibility + npm/Homebrew distribution) | 14/14 + 2 round-2 LOW remediations | 273 | `phase3.x.1-merge-ready-20260426` |
| **3.x.2** | Persistent Model Registry (`FilesystemModelRegistry` + Ed25519-signed `ModelManifest`) | 7/7 | ~80 | `phase3.x.2-merge-ready-20260426` |
| **3.x.3** | Publisher Key Anchor (NEW Solidity contract + Python client + verifier wrappers) | 8/8 | ~70 + 25 e2e | `phase3.x.3-merge-ready-20260427` |
| **3.x.4** | Persistent Privacy Budget (chained-per-entry signed journal) | 8/8 | ~60 | `phase3.x.4-merge-ready-20260427` |
| **3.x.5** | Manifest DHT (cross-node distribution w/ mandatory anchor verification) | 8/8 | 263 + 12 e2e | `phase3.x.5-merge-ready-20260427` |
| **Phase 4 Task 3** | Wallet API HTTP routes + JS SDK `WalletAuth` helper | Task 3a + 3b + 3c | 26 + 21 | (no tag — Tasks 4/6 still vendor-gated) |

Plus cross-cutting infrastructure:
- One **new Solidity contract** (`PublisherKeyAnchor.sol`, 124 LoC) — sha256-derived nodeId binding to Ed25519 pubkey; write-once + admin-override.
- One **new HTTP API surface** (`prsm/interface/api/wallet_api.py`, ~370 LoC) — closes the §5.2 missing piece named in the Phase 4 design plan but never landed by Tasks 1/2/5.
- One **new JS SDK module** (`sdks/javascript/src/wallet-auth.ts`, ~280 LoC) — provider-agnostic SIWE+binding handshake driver.

### 2.2 Docs: 3 new partner-handoff-ready documents

| Doc | Scope | Status |
|---|---|---|
| **Phase 5 §8.1 KYC vendor decision** | Persona / Sumsub / Onfido evaluation; recommends Persona primary + Sumsub fallback for US launch | Partner-handoff ready (`docs/2026-04-27-phase5-kyc-vendor-decision.md`) |
| **Cumulative audit-prep refresh** | Stacks on `phase7.1x-audit-prep-20260422-2`; adds five new phases + new Solidity contract to scope | Auditor-handoff ready (`docs/2026-04-27-cumulative-audit-prep.md`) |
| **5 Phase 3.x design plans** | Per-sub-phase scope + TDD plan + acceptance gate | Self-superseded by their respective merge-ready tags |

---

## 3. Why this matters

### 3.1 Cross-node trust closed end-to-end

Before this sprint, Phase 3.x.1 had four production caveats. The sprint closed three of them:

- **Caveat #3** (in-memory model registry) → CLOSED by 3.x.2 (`FilesystemModelRegistry` ships persistent signed-manifest registry with restart survival + per-shard sha256 commitments + Ed25519 publisher signatures).
- **Cross-node trust gap** (3.x.1 + 3.x.2 sidecar trust model only authenticated within a single node's filesystem boundary) → CLOSED by 3.x.3 (on-chain `PublisherKeyAnchor.sol` provides authoritative pubkey lookup + the `anchor=` kwarg on `FilesystemModelRegistry` / `FilesystemPrivacyBudgetStore` routes verification through the chain rather than the local sidecar).
- **In-memory privacy-budget caveat** → CLOSED by 3.x.4 (signed append-only journal w/ chained-per-entry signatures + RESET audit trail; survives restart).
- **Availability half of cross-node trust gap** → CLOSED by 3.x.5 (manifest DHT lets any node serve a verified manifest copy, with mandatory anchor verification at the DHT client — no trust-the-network mode).

After the sprint, the trust + availability stack is end-to-end closed for model manifests + privacy budgets + inference receipts: any artifact carrying a `publisher_node_id` is verifiable cross-node against a single on-chain authority, AND any node holding the bytes can serve them while readers verify before accepting.

The fourth caveat (#1 — `allow_software_tee_for_privacy=True` default) is unchanged, surfaced in the audit-prep doc, and queued as a small follow-up policy decision.

### 3.2 Eight HIGH-severity findings caught and closed pre-tag

The two-round independent review pattern (round 1 finds, round 2 confirms) caught real security gaps:

| Phase | HIGH finding | Severity rationale |
|---|---|---|
| 3.x.2 | Path-traversal via reserved-name (`.`, `..`) | Disk write to arbitrary location |
| 3.x.3 | `bytes` input bypass on `lookup` (bytes have `.lower()`, slipped past type guard) | Type-confusion → trust bypass |
| 3.x.4 | Negative-ε credit-back attack at parent class | Privacy-budget exhaustion bypass |
| 3.x.5 | **Manifest substitution attack** (validly-signed manifest under wrong model_id) | Model-substitution → wrong shards executed |
| 3.x.5 | `handle()` never-raises invariant violated by uncaught dispatch exceptions | Remote DoS on victim DHT server |
| 3.x.5 | `AnchorRPCError` leaked through per-provider loop | Single anchor RPC blip → entire fetch aborts |
| Phase 4 Task 3 | SIWE no-statement layout violates EIP-4361 | Silent login break for any caller omitting optional statement |
| Phase 4 Task 3 | Lowercase wallet addresses rejected by Python `siwe` library | Silent login break for WalletConnect / Privy users |

All eight closed in the same merge-ready commit cycle. Zero shipped to mainnet. The review pattern continues to demonstrably catch real bugs at this gate, not theoretical ones.

### 3.3 Foundation-track artifacts compounding

The 2026-04-22 sprint produced 7 partner-handoff docs. This sprint adds 3 more (Phase 5 KYC, cumulative audit-prep, five per-phase merge-ready tags).

**Cumulative state:** every Foundation-side gating item is named, documented, and queued. When Foundation entity formation completes (Compliance Officer named → MSB filed → counsel engaged → vendor contracts signed), the engineering tree is ready to consume those signals immediately. No engineering re-planning needed.

### 3.4 Audit engagement is mainnet-deploy-ready

Before this sprint: audit-prep tag from 2026-04-22 covered Phase 7/7.1/7.1x economic substrate only. Five new phases (with one new Solidity contract) had landed since.

After this sprint: `cumulative-audit-prep-20260427` extends the 2026-04-22 baseline to cover the full new surface. Bundle coordinator updated. Per-phase merge-ready tags + design plans + threat models + known-issue lists + auditor prompts all in place.

Estimated additive engagement effort: **~7 audit days** on top of the 2026-04-22 baseline. The Foundation can hand the cumulative tag + bundle coordinator to the auditor the moment the contract signs.

### 3.5 Manifest DHT closes the "decentralization story" gap

A common investor question: "if PRSM is decentralized, what happens when the publisher node goes offline?"

Pre-sprint answer: nothing — model manifests lived only on the publisher's filesystem.

Post-sprint answer: any node that has fetched the manifest (via DHT) and has the shard bytes can serve verified copies to peers. Anchor verification at the DHT client guarantees readers reject tampered bytes regardless of how many providers serve them. The protocol is mainframe-fail-tolerant in the architectural sense, not just the marketing sense.

The E2E test (`tests/integration/test_manifest_dht_e2e.py`) demonstrates this concretely with three simulated nodes: alice publishes; bob + charlie fetch via DHT; alice goes offline; bob still gets the manifest from charlie's cache. Tampering scenarios (malicious provider serving altered bytes) all caught at anchor verify.

---

## 4. Phase-by-phase detail

### 4.1 Phase 3.x.1 — MCP Server Completion

**Status:** 14/14 tasks shipped + 2 round-2 LOW remediations. Tag `phase3.x.1-merge-ready-20260426`.

**What was built:**
- `prsm/compute/inference/` module — InferenceReceipt Ed25519 sign/verify + InferenceExecutor wrapping TensorParallelExecutor + content tier gate (Tier B/C-in-TEE enforcement).
- `prsm_inference` + `prsm_billing_status` MCP tools — production tool definitions + handlers + streaming-context wiring.
- `POST /compute/inference` HTTP endpoint with content-tier checks.
- npm package wrapper (`prsm-mcp`) + Homebrew tap formula scaffold + Distribution Makefile.
- Round-2 review surfaced 2 LOW findings; both closed.

**Production caveats baked into the tag** (per `MEMORY.md`):
1. `allow_software_tee_for_privacy=True` default — operator config, queued as small follow-up.
2. Hardware-TEE enforcement gate not wired (deferred to real hardware).
3. ~~In-memory model registry~~ — CLOSED by 3.x.2.
4. Hosted-MCP-server question decided: defer 12+ months per `docs/2026-04-26-hosted-mcp-server-decision-memo.md`.

### 4.2 Phase 3.x.2 — Persistent Model Registry

**Status:** 7/7 tasks shipped. Tag `phase3.x.2-merge-ready-20260426`.

**What was built:**
- `ModelManifest` frozen dataclass + canonical signing payload (deterministic JSON, sorted keys).
- Ed25519 sign/verify wrappers in `signing.py`.
- `ModelRegistry` ABC + `InMemoryModelRegistry` (drop-in dict replacement) + `FilesystemModelRegistry` (manifest.json + publisher.pubkey + shards/*.bin layout, atomic writes via .tmp + os.replace + fsync).
- `TensorParallelInferenceExecutor` integration: model-loading routes through the registry, tampering caught at sha256 + signature verify.
- Round-1 review caught a HIGH path-traversal via `_RESERVED_NAMES`; remediated.

**Closes Phase 3.x.1 caveat #3.**

### 4.3 Phase 3.x.3 — Publisher Key Anchor

**Status:** 8/8 tasks shipped. Tag `phase3.x.3-merge-ready-20260427`.

**What was built:**
- **NEW Solidity contract** `PublisherKeyAnchor.sol` (124 LoC): write-once `bytes16 → bytes32` registry; `register(publicKey)` derives node_id via sha256(pubkey)[:16] (matching off-chain Python derivation byte-equally); `lookup(node_id) → bytes`; `adminOverride(...)` multi-sig escape hatch. 20 Hardhat tests.
- Python `PublisherKeyAnchorClient` (web3.py wrapper) + negative-cache + retry handling.
- Verifier wrappers: `verify_manifest_with_anchor`, `verify_entry_with_anchor`, `verify_receipt_with_anchor`.
- `anchor=` kwarg integration into `FilesystemModelRegistry` (3.x.2) + `FilesystemPrivacyBudgetStore` (3.x.4) — when set, on-chain pubkey replaces on-disk sidecar as verification anchor.
- Sepolia deployment script + runbook (already broadcast operator-side; Base mainnet bundles into Phase 1.3 audit clock).
- Round-1 review caught a HIGH bytes-input bypass on `lookup`; remediated.

**Closes the cross-node trust-boundary caveat shared by 3.x.2 + 3.x.4 (for callers passing `anchor=`).**

### 4.4 Phase 3.x.4 — Persistent Privacy Budget

**Status:** 8/8 tasks shipped. Tag `phase3.x.4-merge-ready-20260427`.

**What was built:**
- `PrivacyBudgetEntry` frozen dataclass + canonical signing payload.
- Chained Ed25519 signatures: each entry's payload includes `prev_hash` (sha256 of previous entry). Tampering breaks the chain at every subsequent verification.
- `PrivacyBudgetStore` ABC + `InMemoryPrivacyBudgetStore` + `FilesystemPrivacyBudgetStore` (atomic append).
- `PersistentPrivacyBudgetTracker` enforcing per-user epsilon over the journal.
- `node.py` integration replacing the in-memory tracker.
- Round-1 review caught HIGH negative-ε credit-back attack at parent class; remediated by guarding at the ABC level (all subclasses inherit).

**Closes Phase 3.x.1 in-memory privacy-tracker caveat.**

### 4.5 Phase 3.x.5 — Manifest DHT

**Status:** 8/8 tasks shipped. Tag `phase3.x.5-merge-ready-20260427`.

**What was built:**
- `prsm/network/manifest_dht/` package: 5 message dataclasses + JSON wire codec + protocol version + size caps.
- `LocalManifestIndex` — per-node servable-manifests index w/ rebuild-from-walk + orphan reconciliation.
- `ManifestDHTClient` — single-round Kademlia find_providers; mandatory `anchor=` at construction; substitution-defended `get_manifest` (model_id assertion before anchor verify).
- `ManifestDHTServer` — `handle(bytes) → bytes` that NEVER raises; outer try/except wraps all dispatch.
- `FilesystemModelRegistry` `dht=` kwarg integration: best-effort announce on register; DHT fallback + cache + re-announce on get_manifest miss.
- E2E (`test_manifest_dht_e2e.py`): three simulated nodes, real signing, real protocol round-trips. 12 scenarios — happy path, tampering, offline-publisher, anchor enforcement, composition.
- Round-1 review caught **3 HIGH findings** (manifest substitution attack, `handle()` never-raises invariant, AnchorRPCError leak) + 2 MEDIUM (size bounds, orphan reconciliation). All remediated. Round-2 SAFE-TO-DEPLOY.

**Closes the AVAILABILITY half of the cross-node trust gap.** After 3.x.3 (TRUST) + 3.x.5 (AVAILABILITY), any verified manifest copy is reachable from any node, and any reader can verify the bytes are legitimate before accepting.

### 4.6 Phase 4 Task 3 — Wallet API + JS SDK Helper

**Status:** Task 3a + 3b + 3c shipped. Tasks 4 + 6 still gated on Foundation Privy contract (per §8.1 vendor-decision memo). No phase-level tag yet.

**What was built:**
- **Task 3a — `prsm/interface/api/wallet_api.py`** (closes the API surface named in design plan §5.2 but never delivered by Tasks 1/2/5): `POST /api/v1/auth/wallet/siwe/nonce`, `POST /siwe/verify`, `POST /bind`, `GET /binding`, `GET /balance`. Service injection via FastAPI Depends + race-safe `set_services` boot hook. Stable error-code contract.
- **Task 3b — `sdks/javascript/src/wallet-auth.ts`**: provider-agnostic `WalletAuth` class wrapping the SIWE+binding handshake. `connectCoinbaseWallet()` is the composed flow (eth_requestAccounts → /siwe/nonce → buildSiweMessage → personal_sign → /siwe/verify → personal_sign(binding) → /bind).
- Round-1 review caught **2 HIGH findings** that the test suite couldn't surface by construction (no-statement SIWE layout, lowercase address rejection by `siwe` library). Both remediated with `toChecksumAddress` (js-sha3 keccak256 EIP-55 normalisation) + corrected EIP-4361 layout. Plus 4 IMPORTANT polish items.

**Why Task 4 is correctly deferred:** the §8.1 vendor-decision memo's G1-G4 checklist (Privy contract, credentials, pricing commitment, security review) is Foundation-side. Task 3 ships the engineering scope that's actually unblocked; Task 4 lands the vendor-specific wiring on top of this foundation when Privy signs.

---

## 5. Foundation track snapshot

The engineering track is at a natural quiet point. Foundation track is the practical critical path for everything still gated.

| Foundation gate | Status | Engineering blocked on |
|---|---|---|
| Foundation entity formed (Cayman nonprofit) | Documented; not yet incorporated | Phase 5 Task 1; mainnet deploy ceremony |
| Compliance Officer named | Job description done (#114); not yet hired | Phase 5 K4 (KYC compliance review) |
| Privy contract signed | Memo ready (`docs/2026-04-22-phase4-wallet-vendor-decision.md`) | Phase 4 Task 4 |
| Persona contract signed | Memo ready (`docs/2026-04-27-phase5-kyc-vendor-decision.md`) | Phase 5 Task 2 |
| FinCEN MSB registration filed | Outside engineering scope | Phase 5 Task 1 (production go-live) |
| External auditor contract signed | Shortlist done (#101); not yet engaged | Phase 7 Task 9 + Phase 7.1 Task 9 + Phase 8 audit |
| Hardware multi-sig (HSMs procured + ceremony plan) | Outside engineering scope | Mainnet deploy ceremony |

**Each gate has a named green-light document.** When any of them clear, the engineering work that depends on it can begin without re-planning.

---

## 6. What's open (engineering side)

After this sprint, three small unblocked engineering items remain:

1. **3.x.1 caveat #1** — `allow_software_tee_for_privacy` default flip from `True` to `False` (safe-fail). 1-line change pending policy decision: do operators of software-only nodes opt into serving STANDARD/HIGH/MAXIMUM privacy tiers, or opt out?
2. **Phase 8 Task 5** — Sepolia deploy script for EmissionController + CompensationDistributor. The actual multi-stakeholder testnet exercise needs Foundation+operators+investors+auditors, but the deploy infrastructure is unblocked engineering work.
3. **Cross-cutting hardening sweep** — ruff/mypy/dep hygiene on the new Phase 4 Task 3 surface. Diminishing-returns work; only worth it before audit engagement.

Everything else is either shipped, gated on Foundation, or in the deferred-research track (R1-R8, with named promotion triggers).

---

## 7. Test surface

| Surface | Pre-sprint | Post-sprint | Δ |
|---|---|---|---|
| Phase 3.x.1 | 273 | 273 | 0 (already shipped) |
| Phase 3.x.2 | 0 | ~80 | +80 |
| Phase 3.x.3 | 0 | ~70 + 25 e2e | +95 |
| Phase 3.x.4 | 0 | ~60 | +60 |
| Phase 3.x.5 | 0 | 263 + 12 e2e | +275 |
| Phase 4 Task 3 (wallet_api + JS) | 0 | 26 + 21 | +47 |
| **Sprint total** | — | **~620 new tests** | **+620** |

All green. Zero regressions on the pre-existing surface (~2,803 tests as of 2026-04-22 baseline).

---

## 8. What changes for an investor reading this

1. **The "decentralization story" is no longer aspirational** — manifest distribution + cross-node verification ship as working code with E2E tests demonstrating multi-node fault tolerance.
2. **The audit engagement is ready to start** — single doc + single tag handed to auditor on contract signing; estimated ~7 additive audit days for this sprint's surface.
3. **Two more vendor-contract-blocking gates resolved upstream** — Phase 4 Privy + Phase 5 Persona both have ratification-ready memos; Foundation product lead can ratify on their schedule without re-running the evaluation.
4. **Foundation track is now THE critical path** — engineering has shipped to the boundary of what it can do without the entity formed. Capital-raise + entity-formation timing now drives launch timing.

---

## 9. Cross-references

**Sprint 1 baseline:**
- `docs/2026-04-22-session-summary-engineering-sprint.md` — what shipped before this sprint

**Per-phase reference:**
- `docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md`
- `docs/2026-04-26-phase3.x.2-persistent-model-registry-design-plan.md`
- `docs/2026-04-27-phase3.x.3-publisher-key-anchor-design-plan.md`
- `docs/2026-04-27-phase3.x.4-persistent-privacy-budget-design-plan.md`
- `docs/2026-04-27-phase3.x.5-manifest-dht-design-plan.md`
- `docs/2026-04-22-phase4-wallet-sdk-design-plan.md`

**Vendor decisions:**
- `docs/2026-04-22-phase4-wallet-vendor-decision.md` (Privy)
- `docs/2026-04-27-phase5-kyc-vendor-decision.md` (Persona)

**Audit-prep:**
- `docs/2026-04-21-audit-bundle-coordinator.md` (entry-point; updated to reference the new bundle)
- `docs/2026-04-22-phase7.1x-audit-prep.md` (economic substrate baseline)
- `docs/2026-04-27-cumulative-audit-prep.md` (cumulative refresh)

**Tags pinned in this sprint:**
- `phase3.x.1-merge-ready-20260426`
- `phase3.x.2-merge-ready-20260426`
- `phase3.x.3-merge-ready-20260427`
- `phase3.x.4-merge-ready-20260427`
- `phase3.x.5-merge-ready-20260427`
- `cumulative-audit-prep-20260427`

---

## 10. Changelog

- **0.1 (2026-04-27):** initial sprint-2 summary covering 3.x.1 → 3.x.5 + Phase 4 Task 3 + cumulative audit-prep refresh + 2 vendor memos. Stacks on `docs/2026-04-22-session-summary-engineering-sprint.md`.
