# Phase 3 — Shipped Status + Forward Plan for Remaining Workstreams

**Document identifier:** PRSM-PHASE3-STATUS-1
**Version:** 0.1 Draft
**Status:** Supersedes `docs/2026-04-14-phase3-preplanning.md` (historical stub). Documents actual Phase 3 / Phase 3.1 delivery status against the original three-workstream scope and scopes the residual workstreams as Phase 3.x follow-ons.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Related documents:**
- `docs/2026-04-14-phase3-preplanning.md` — original preplanning stub with three workstreams (MCP server / short-term FTNS↔USDC liquidity / operator toolkit).
- `docs/2026-04-20-phase3-marketplace-design.md` + `-plan.md` — Phase 3 matching-engine design + TDD plan (DELIVERED).
- `docs/2026-04-21-phase3.1-batch-settlement-design.md` — Phase 3.1 batch-settlement addendum (DELIVERED).
- `PRSM_Vision.md` §6 + Executive Summary — origin of each workstream.
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — governance interlock for operator-supply diversity (PRSM-SUPPLY-1).

---

## 1. Purpose

When `2026-04-14-phase3-preplanning.md` was written, Phase 3 was named as "the marketplace phase" with three parallel workstreams: (1) MCP server, (2) short-term FTNS↔USDC liquidity guarantee, (3) cloud-provider compatibility + operator toolkit. Those three were framed as equally important Phase 3 deliverables.

What actually shipped under the **Phase 3 label** is narrower than that: the marketplace matching engine + batch-settlement addendum. MCP server shipped IN PARALLEL under its own track and is ~80% feature-complete. The liquidity + operator-toolkit workstreams have not been executed.

This document does three things:

1. **Records delivery status** against the original three-workstream scope, so the Phase 3 bookkeeping is accurate rather than fictionally "unstarted."
2. **Scopes Phase 3.x follow-on workstreams** for the two residual deliverables (liquidity + operator toolkit) and the MCP completion gap. Each gets a promotion-trigger section matching the R1-R8 scoping-doc pattern.
3. **Realigns the preplanning stub with reality** — future readers who land on the stub will find a pointer here rather than treating stale framing as current.

**This is a status document, not a plan.** Execution of the residual workstreams requires its own design + TDD plans at promotion time, following the Phase 4/5/6/7-storage/8 pattern.

---

## 2. Shipped under the Phase 3 label

### 2.1 Phase 3 — Marketplace matching engine ✅ DELIVERED

**Design doc:** `docs/2026-04-20-phase3-marketplace-design.md` (386 lines).
**TDD plan:** `docs/2026-04-20-phase3-marketplace-plan.md` (659 lines, 8 tasks).

Shipped modules under `prsm/marketplace/`:
- `listing.py` — ProviderListing wire format + sign/verify.
- `directory.py` — provider-listing directory with TTL expiry.
- `advertiser.py` — provider-side listing broadcast.
- `filter.py` — policy-based selection (price, TEE, min reputation).
- `policy.py` — requester dispatch policy.
- `price_handshake.py` — fixed-price negotiation with the discovered provider.
- `reputation.py` — lightweight score without slashing (Phase 7 adds slash).
- `orchestrator.py` — wraps Phase 2 RemoteShardDispatcher.
- `errors.py` — marketplace-specific exception taxonomy.

Wrapping pattern: `MarketplaceOrchestrator` discovers + selects + hands off to the Phase 2 dispatcher. Phase 2 code unchanged.

### 2.2 Phase 3.1 — Batch-settlement addendum ✅ DELIVERED

**Design doc:** `docs/2026-04-21-phase3.1-batch-settlement-design.md` (545 lines).

Shipped modules under `prsm/marketplace/` + `contracts/contracts/`:
- `contracts/contracts/BatchSettlementRegistry.sol` — on-chain receipt commitment + challenge flow.
- `contracts/contracts/EscrowPool.sol` — per-dispatch escrow integrated with settlement.
- `prsm/marketplace/consensus_submitter.py` + `consensus_queue.py` — challenge-submitter service with persistence + leasing.
- Receipt Merkle accumulator + canonical encoding (`prsm/marketplace/`).

Cumulative effect: marketplace dispatch settles on-chain in batches, with challengeable receipts and per-receipt slashing hooks (Phase 7 tier gating layered on top).

### 2.3 MCP server — core tool surface ✅ DELIVERED, completeness gaps remain

**Status:** `prsm/mcp_server.py` (1073 lines) ships core PRSM tools over MCP stdio, runnable via `python -m prsm.mcp_server`.

**What's live today:**
- Tool definitions for query, retrieval, and compute-dispatch entry points.
- JSON-RPC stdio compliance (MCP stdout purity enforced — see §MCP Server header).
- CLI wrapper `prsm/cli_modules/mcp_server.py` for operator start/stop.

**Completeness gaps per original stub §Workstream-1:**
- `prsm_inference(prompt, model_id, budget_ftns, privacy_tier)` — private-inference tool. Privacy tier enforcement not fully wired (depends on Phase 7 content tier + Phase 2 TEE attestation integration).
- FTNS wallet delegation / billing settlement — current billing is basic; per-tool settlement through PaymentEscrow not fully wired.
- Distribution: npm / Python / Homebrew one-liners not shipped.
- Hosted "PRSM-as-a-service" MCP server decision: deferred.
- Streaming response semantics: partial (inference streaming not fully tested through the MCP path).

These gaps are scoped as **Phase 3.x — MCP Server Completion** in §3.1 below.

---

## 3. Residual workstreams from the original preplanning stub

Each of the three items below is framed as a Phase 3.x follow-on with its own scoping-doc-style trigger + partner-handoff sections. Full design + TDD plans get written at promotion time.

### 3.1 Phase 3.x — MCP Server Completion

**Origin:** `docs/2026-04-14-phase3-preplanning.md` §Workstream 1 — "highest-leverage adoption lever."

**Scope:**
- Complete the `prsm_inference` tool path once Phase 2 TEE attestation gating + Phase 7 content-tier gating are both live.
- Per-tool FTNS settlement wired through PaymentEscrow; per-call billing visible to the MCP client.
- Distribution: npm package + Python package + Homebrew formula. Each with a one-liner install.
- Streaming semantics verified through the MCP SDK's test harness.
- Decision memo on the hosted-MCP-server question (go/no-go based on observed user demand after npm launch).

**Promotion triggers:**
- **T1 — adoption floor.** Phase 3 marketplace has ≥20 active providers and ≥50 daily dispatches — the user-facing tool surface is worth distributing widely.
- **T2 — dependencies green.** Phase 7 content-tier gating LIVE + Phase 2 TEE attestation LIVE (both are known-future-gates per current roadmap).
- **T3 — partner ask.** At least one MCP-client-ecosystem partner (Claude Desktop integration team, another MCP-compat client vendor) expresses concrete interest in a distributable PRSM MCP package.

**Estimated scope:** ~3-5 engineering tasks + 1 distribution / packaging task. Calendar: 4-6 weeks including npm/Homebrew review cycles.

**Open issues:** hosted-MCP-server product question (Foundation-run vs. fully self-hosted); regulatory implications of Foundation holding user FTNS for MCP billing (likely defers to Phase 5 fiat/KYC outcome).

### 3.2 Phase 3.x — Short-Term FTNS↔USDC Liquidity Guarantee

**Origin:** `docs/2026-04-14-phase3-preplanning.md` §Workstream 2; `PRSM_Vision.md` §6 "Honest caveats" on T3 arbitrage.

**Scope (original three mutually-exclusive options unchanged):**
- **Option A — Foundation-operated settlement facility.** Foundation USDC float; arbitrageurs redeem FTNS at a price band (e.g., 95% of 1-hour VWAP). Foundation takes market-making risk.
- **Option B — AMM-integrated auto-swap on Aerodrome.** Arbitrage-node software auto-routes earned FTNS through Aerodrome's USDC pool with slippage bounds. No Foundation risk; relies on pool depth.
- **Option C — Third-party prime broker.** Regulated crypto prime broker offers T3 operators hedged settlement.

**Recommended sequencing (preserved from preplanning):**
- Ship **Option B** at Phase 3.x launch (minimal new infra, relies on Aerodrome pool existing).
- Add **Option C** as Phase 4+ maturity (integration complexity is substantial).
- Avoid Option A unless Option B proves insufficient at scale.

**Promotion triggers:**
- **T1 — T3 demand.** ≥5 arbitrage-style operators in the marketplace requesting predictable USD settlement.
- **T2 — Aerodrome depth.** Aerodrome FTNS↔USDC pool reaches a TVL threshold (suggested ≥$500k) that makes Option B viable without catastrophic slippage.
- **T3 — Foundation treasury readiness.** PRSM-TOK-1 / tokenomics legal track resolved to the point where Foundation can operate a USDC-denominated settlement facility if Option B is insufficient.

**Estimated scope:** Option B alone is ~2-3 engineering tasks (auto-swap client, slippage config, dashboard integration). Option C is ~8-12 tasks gated on a contracted prime-broker partner.

**Dependencies:** Aerodrome pool seeding (see `PRSM_Vision.md` §8 Phase-2 discussion); Phase 5 fiat/KYC outcome may reshape if a Foundation-run Option A becomes necessary.

### 3.3 Phase 3.x — Operator Toolkit + Cloud Provider Compatibility

**Origin:** `docs/2026-04-14-phase3-preplanning.md` §Workstream 3; `PRSM_Vision.md` §6 "Honest caveats" on operator onboarding.

**Scope:**

- **Provider compatibility matrix.** Publish `docs/cloud-provider-compatibility.md` enumerating which providers allow PRSM node operation under current AUP:
  - Confirmed compatible (permissive AUP as of 2026-04-22): RunPod, Lambda Cloud, CoreWeave (agreement-dependent).
  - Confirmed incompatible: AWS (crypto-mining AUP), GCP (similar), Azure (similar).
  - Requires individual confirmation: Vast.ai, Genesis Cloud, Fluidstack, and other specialised providers.
  - Review cadence: quarterly (2027-04-22 first refresh).
- **One-click deployment recipes.** Docker image + IaC templates (Terraform / Pulumi / Ansible) per compatible provider. Operator target: `make deploy-runpod-h100` → online node in ≤5 min.
- **Operator dashboard.** Minimal web UI: earnings, preemption events, FTNS balance, auto-swap status. Not a full control plane — enough for 1-10-node operators without CLI expertise.
- **Operator documentation.** Compute-pricing guide; dispatch-market reading guide; preemption-handling primer; basic tax-treatment guidance (tagged EXPLICITLY as non-legal-advice per Tokenomics §9 guidance).

**Promotion triggers:**
- **T1 — operator-friction signal.** Support channel sees ≥10 distinct operators ask "how do I deploy this on provider X" OR Discord / GitHub discussions consistently surface provider-compatibility questions.
- **T2 — PRSM-SUPPLY-1 interlock.** Once PRSM-SUPPLY-1's supply-diversity metrics start publishing, the compatibility matrix becomes the primary tool for improving diversity scores — ops-level priority rises.
- **T3 — Foundation capacity.** At least one Foundation DevRel-style role exists to maintain the matrix + deployment recipes quarterly.

**Estimated scope:** ~2 engineering tasks for the dashboard (Python + minimal React), ongoing docs maintenance. Calendar: 6-8 weeks initial + quarterly upkeep.

**Dependencies:** Phase 3.x liquidity (§3.2) — dashboard needs auto-swap status to surface meaningfully. Plan Option B ship before the dashboard.

---

## 4. What's still out-of-scope for Phase 3.x

These remain on explicit later-phase tracks and should not be pulled into Phase 3.x:

- **FHE / MPC research** — R1-SCOPING-1 + R2-SCOPING-1 (Phase 4+ research track).
- **Per-provider supply caps / geographic-diversity bonuses** — PRSM-SUPPLY-1 governance standard.
- **Hybrid tokenomics ratification** — Tokenomics §11 legal / governance workstream, independent cadence.
- **Secondary token markets / CEX listings** — Phase 4+ distribution.
- **Content confidentiality Tier B / Tier C** — Phase 7-storage (ALREADY SHIPPED as of this document's date, engineering complete; see `phase7-storage-merge-ready-20260422` tag).

---

## 5. Status summary

| Original Workstream | Status | Follow-on |
|---------------------|--------|-----------|
| **Phase 3 Marketplace (matching engine)** | ✅ Delivered (Phase 3 + Phase 3.1 tags) | n/a |
| **Workstream 1 — MCP Server** | ⚠️ Partial — core shipped, 5 gaps open | §3.1 Phase 3.x MCP Completion |
| **Workstream 2 — FTNS↔USDC Liquidity** | ❌ Not started | §3.2 Phase 3.x Liquidity Guarantee |
| **Workstream 3 — Operator Toolkit** | ❌ Not started | §3.3 Phase 3.x Operator Toolkit |

All three Phase 3.x follow-on workstreams are partner-handoff-ready at the scoping-doc level above. None is currently in execution. Promotion to a full design + TDD plan follows the same pattern as Phase 4/5/6/7-storage/8 — triggers in the above sections plus Foundation budget allocation.

---

## 6. Decision made by this document

1. **The 2026-04-14 preplanning stub is historical.** Future readers landing there should follow the pointer to this document.
2. **Phase 3 as a label refers to the marketplace + batch-settlement delivery only.** The three-workstream framing of the original stub is preserved as scope context but renamed to Phase 3.x for the residual items, matching the Phase 7 compute-vs-storage split precedent.
3. **No residual workstream enters execution without its own design + TDD plan.** This document does NOT authorize engineering work; it scopes the space so a future plan can be written confidently.

---

## 7. Cross-references

- **R4 SUPPLY-1** (governance standard) — operator-toolkit promotion trigger T2 ties into SUPPLY-1 metrics.
- **Phase 4 Wallet SDK** (`docs/2026-04-22-phase4-wallet-sdk-design-plan.md`) — MCP-server FTNS billing reuses Phase 4 wallet binding.
- **Phase 5 Fiat On-Ramp** (`docs/2026-04-22-phase5-fiat-onramp-design-plan.md`) — Phase 3.x liquidity Option A / C decisions depend on Phase 5 regulatory outcome.
- **Phase 7-storage** — content confidentiality gating is live (SHIPPED); MCP-server `privacy_tier` parameter can now meaningfully distinguish Tier A/B/C.

---

## 8. Changelog

- **0.1 (2026-04-22):** initial status + forward-plan doc. Supersedes `docs/2026-04-14-phase3-preplanning.md` as the current-state view of Phase 3 and its follow-on workstreams.
