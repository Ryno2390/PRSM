# Phase 3 Pre-Planning Stub

**Status:** HISTORICAL. Superseded by [`2026-04-22-phase3-status-and-forward-plan.md`](./2026-04-22-phase3-status-and-forward-plan.md) (PRSM-PHASE3-STATUS-1) as of 2026-04-22. The marketplace matching engine (Workstream 0, implicit) shipped as Phase 3 + Phase 3.1; the three workstreams listed below are now scoped as Phase 3.x follow-ons in the status doc. Please read the status doc first.

---

**Original status:** Pre-planning only. Not yet a formal phase plan. Captured 2026-04-14 during Phase 1.3 Sepolia bake-in so Vision-doc-derived requirements are not forgotten when Phase 2 ships.

**Context:** Phase 3 is the "Marketplace" phase in the current roadmap — the layer where PRSM transitions from protocol-level primitives (Phase 1: on-chain provenance; Phase 2: remote compute dispatch) to external-facing product surface. This document captures the workstreams that need to land in Phase 3 per `PRSM_Vision.md` (specifically the Executive Summary "Positioning" paragraph and Section 6's four-tier supply architecture).

Formal Phase 3 planning (subagent-driven, two-stage review) will occur after Phase 2 ships and a re-audit is complete.

## Workstream 1: MCP Server

**Vision doc reference:** Executive Summary, "The positioning" paragraph — *"Users continue to interact with Claude, Gemini, or GPT as their conversational interface; PRSM plugs in underneath via MCP (Model Context Protocol) as a retrieval and heavy-compute substrate."*

**Strategic importance:** highest-leverage adoption lever. If PRSM ships as a clean MCP server, Claude Desktop / ChatGPT Desktop / Gemini / any MCP-compatible client can invoke PRSM without the user switching interfaces. Adoption does not require changing chat clients — it requires installing an MCP server.

**Scope:**

1. **MCP server entry point.** Expose PRSM's query/retrieval/compute interfaces as an MCP server compliant with the current MCP spec (see `modelcontextprotocol.io` for version at Phase 3 start).
2. **Tool definitions.** At minimum:
   - `prsm_retrieve(query, budget_ftns, privacy_tier)` — retrieve from PRSM's data layer with creator royalty settlement.
   - `prsm_compute(task_spec, budget_ftns, privacy_tier)` — dispatch heavy compute via Phase 2 remote dispatcher.
   - `prsm_inference(prompt, model_id, budget_ftns, privacy_tier)` — run private inference via Section 7 zero-trust stack.
3. **Billing integration.** MCP server holds user's FTNS wallet (or delegated signer); per-tool calls settle on-chain via existing `PaymentEscrow`.
4. **Privacy tier parameter.** Clients can request privacy tiers that combine (a) compute-side TEE attestation from Phase 2 line item C and (b) content confidentiality tier (A/B/C per PRSM_Vision.md §2) from Phase 7. Until Phase 7 ships, only content Tier A is available, so MCP clients requesting Tier B or C privacy fail fast or fall back with clear user-visible warnings. Post-Phase 7, the full matrix (public-content + standard-compute through zero-knowledge-content + TEE-attested-compute) is exposed.
5. **Distribution.** npm package, Python package, Homebrew formula. Installation one-liner on each major platform.

**Dependencies:**
- Phase 2 remote compute dispatcher must be shipped and stable.
- Phase 2 line item C (TEE attestation tier gating) should be complete for the `tee_attested` / `maximum` tiers to be meaningful. Without it, privacy_tier is informational only.
- User wallet management UX — either bring-your-own-wallet via connected Metamask/Coinbase Wallet, or a simple "top up $X of FTNS" custodial option (regulatory implications TBD).

**Open questions:**
- Do we ship a hosted "PRSM-as-a-service" MCP server for users who don't want to run one locally? Simpler onboarding, centralization tradeoff.
- How are streaming responses handled? MCP streaming semantics must be respected for inference use cases.

## Workstream 2: Short-Term FTNS → USDC Liquidity Guarantee

**Vision doc reference:** Section 6 subsection, "Honest caveats" for T3 arbitrage — *"PRSM should ship short-term FTNS→USDC liquidity guarantees to reduce this friction."*

**Strategic importance:** Without this, T3 professional arbitrage tier is accessible only to operators sophisticated enough to hedge FTNS exposure via perps or off-exchange markets. With this, a T3 operator can run bot-driven hourly settlements: pay RunPod $2.50, earn $3.00 in FTNS, swap FTNS → USDC with guaranteed floor within (say) 1 hour at <2% slippage. Retail-accessible T3 participation depends on it.

**Scope options (mutually exclusive):**

**Option A: Foundation-operated settlement facility.** The PRSM Foundation holds a USDC float that arbitrageurs can draw from by returning FTNS at a guaranteed price band (e.g., 95% of 1-hour VWAP). Foundation takes market-making risk; retail operators get predictable USD income. Simplest design; requires foundation balance-sheet commitment.

**Option B: AMM-integrated auto-swap on Aerodrome.** Arbitrage node software automatically routes earned FTNS through Aerodrome's USDC pool at a slippage-bounded interval. No foundation risk; relies on Aerodrome pool depth being adequate. Works for small operators; breaks at scale during adverse markets.

**Option C: Third-party prime-broker integration.** Partner with a regulated crypto prime broker to offer T3 operators hedged settlement. Operator-facing product; foundation does not hold the risk. Best long-term; highest integration complexity.

**Recommendation:** ship Option B at Phase 3 launch (minimal new infra), add Option C as Phase 4 maturity. Option A avoided unless Option B proves insufficient.

**Dependencies:**
- Phase 1's Aerodrome AMM seeding (Vision doc Section 8 Phase 2 discussion) must be live with adequate depth.
- Arbitrage node software (workstream 3) must include auto-swap as a first-class feature.

## Workstream 3: Cloud Provider Compatibility Documentation & Operator Toolkit

**Vision doc reference:** Section 6 subsection, "Honest caveats" — *"PRSM's node documentation will specify provider compatibility explicitly."*

**Strategic importance:** Lowering T3 onboarding friction from "you figure it out" to "follow these steps for provider X." Current state: no official documentation. Professional operators can assemble their own toolchain; the middle-tier operator class cannot.

**Scope:**

1. **Provider compatibility matrix.** Which cloud providers allow PRSM node operation under current AUP:
   - **Confirmed compatible:** RunPod (permissive), Lambda Cloud (permissive), CoreWeave (varies by agreement).
   - **Confirmed incompatible or restricted:** AWS (crypto-mining AUP), GCP (similar), Azure (similar).
   - **Requires individual confirmation:** smaller specialized providers (Vast.ai, Genesis Cloud, etc.).
   Update quarterly; publish as `docs/cloud-provider-compatibility.md`.
2. **One-click node deployment.** Docker image + Terraform/Pulumi templates for each compatible provider. An operator runs `make deploy-runpod-h100` or equivalent and has a node online in under 5 minutes.
3. **Operator dashboard.** Minimal web UI showing node earnings, preemption events, FTNS balance, auto-swap status. Not a full control plane; just enough to let a small operator manage 1-10 nodes without CLI expertise.
4. **Operator documentation.** How to price your compute, how to read the PRSM dispatch market, what preemption handling looks like, basic tax guidance (see legal workstream).

**Dependencies:**
- Phase 2 remote dispatcher must be stable.
- Short-term liquidity guarantee (workstream 2) must be at least partially live for the dashboard to surface USD-denominated earnings.

## Out-of-Scope Clarifications

These belong in later phases and are named here only so they are not accidentally pulled into Phase 3:

- **FHE / MPC research track** → Phase 4+ research, not product.
- **Per-provider supply caps / geographic diversity bonuses** → governance parameters, Phase 4+ once monitoring data exists.
- **Hybrid tokenomics ratification** (Tokenomics Section 11) → legal/governance workstream, tracked separately.
- **Secondary token markets / CEX listings** → Phase 4+ distribution.
- **Content confidentiality Tier B and Tier C** → Phase 7 delivery. Phase 3 MCP server must handle "Tier A only" gracefully until Phase 7 ships; any MCP client requesting higher confidentiality tiers must receive a clear pre-Phase-7 error or degraded-mode fallback, never silently serve lower tier than requested.

## Next Steps

1. Phase 2 ships and is audited (not yet done as of 2026-04-14).
2. Post-Phase-2 retrospective identifies any scope changes to this stub.
3. Formal Phase 3 plan written via subagent-driven planning flow, referencing this stub as scope input.
4. Two-stage review (spec compliance → code quality) as standard.

No code work on Phase 3 should begin until the formal plan is drafted and reviewed.
