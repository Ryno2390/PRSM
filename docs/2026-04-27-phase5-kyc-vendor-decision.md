# Phase 5 §8.1 — KYC Vendor Decision

**Document identifier:** PHASE5-KYC-VENDOR-1
**Version:** 0.1 Draft
**Status:** Partner-handoff-ready decision memo. Resolves the Phase 5 design plan §8.1 open-issue by recommending a specific vendor, documenting evaluation criteria, and specifying the green-light checklist that unblocks Phase 5 Task 2 engineering. **Not yet a signed decision** — this memo recommends; Foundation Compliance Officer ratifies (jointly with product lead + legal).
**Date:** 2026-04-27
**Drafting authority:** PRSM founder
**Related documents:**
- `docs/2026-04-22-phase5-fiat-onramp-design-plan.md` §8.1 — source open issue with three candidates (Persona / Sumsub / Onfido).
- `docs/2026-04-22-phase4-wallet-vendor-decision.md` — Phase 4 Privy memo. Same pattern, different vendor space; this memo deliberately mirrors structure.
- `prsm/economy/payments/vendor_adapters.py` — Phase 5 Task 2-4 vendor adapter scaffolds (SHIPPED). Adapter interface is vendor-agnostic; choice here determines which scaffold gets production wiring.
- `prsm/economy/payments/withdrawal_orchestrator.py` — Phase 5 Task 5 (SHIPPED). Consumes the `KycService` Protocol; vendor-choice-blind by design.

---

## 1. Purpose

Phase 5 design plan §8.1 names three candidate KYC vendors as the path for "user identity verification → tier assignment → sanctions screening hook":

- **Persona** — "strong US coverage, modern API, mid-range pricing."
- **Sumsub** — "strong international coverage, lower cost at volume."
- **Onfido** — "older player, enterprise-focused, slower roll-out."

Plan §8.1 records a tentative recommendation for Persona (US-only launch) but defers the binding decision. Phase 5 Tasks 2 (KYC vendor integration), 3 (Stripe — depends on KYC tier output for charge limits), and 4 (Coinbase Exchange — depends on tier for swap volume caps) all flow through the chosen vendor's identity-verification + sanctions-hit signals:

- Task 2 scope IS "selected vendor (§8.1) integration."
- Task 3's user-tier-aware charge limits depend on Task 2's tier output.
- Task 4's swap-volume gating depends on Task 2's tier output.
- Task 7 review gate cannot pass without an end-to-end onboarding test against the chosen vendor.

Without a named choice, Phase 5 Tasks 2-4 stay in scoping limbo. This memo resolves that.

**What this memo does:**
1. Re-states the evaluation criteria from plan §8.1 in measurable terms.
2. Scores each candidate against criteria as of 2026-04-27.
3. Recommends a specific vendor with explicit rationale.
4. Lists the green-light checklist that must pass before Phase 5 Task 2 engineering begins.
5. Specifies the follow-up conditions under which the decision should be re-examined.

**What this memo does NOT do:**
- Not a contract. Foundation product + legal + Compliance Officer sign the actual KYC vendor agreement before integration.
- Not a Phase 5 Task 1 substitute. Compliance review (FinCEN MSB, OFAC, Howey, ToS+Privacy) is a separate Foundation track.
- Not an integration plan. Phase 5 Task 2 design + TDD follows this memo's choice.
- Not a permanent lock-in. §7 specifies re-review triggers (notably international-expansion trigger T4, which is the most likely to fire).

---

## 2. Criteria

From plan §8.1, normalised to measurable form:

| Criterion | What "pass" looks like |
|-----------|------------------------|
| **C1 US coverage** | All 50 states + DC + US territories. Document types: state-issued ID, driver's license, passport. Liveness: video selfie or equivalent biometric. SSN-collection optional + appropriately gated. |
| **C2 Sanctions screening** | OFAC SDN + sectoral lists, plus EU + UK + UN sanctions screening. Real-time check at onboarding + periodic re-screening for active users. PEP screening optional but supported. |
| **C3 API stability + latency** | Production-grade REST API. ≥99.5% uptime SLA on a published status page. p95 verification latency ≤30s end-to-end. Webhooks for async events (verification complete, sanctions match, document re-review required). |
| **C4 Pricing at PRSM scale** | Per-verification cost ≤$2 at <10k verifications/month; ≤$1 at 100k+ verifications/month. Pricing model published OR enterprise rates negotiable in writing. |
| **C5 Fraud-rate track record** | Public case studies or industry benchmarks demonstrating fraud-rate parity with peers. Vendor-disclosed false-positive rate at the chosen tier ≤2%. Acceptable tradeoff vs false-negative rate. |
| **C6 Compliance posture** | SOC 2 Type II current. ISO 27001 OR equivalent. Subject-access-request (GDPR Article 15 / CCPA equivalent) workflow documented. Data residency: US data stays in US. |
| **C7 Exit / data-portability story** | If the vendor is acquired / pivots / shuts down, can PRSM export user verification records? What's the operational cost to PRSM of vendor-switching mid-deployment? |

C1-C5 map directly to plan §8.1's stated criteria (coverage, cost, API stability, fraud rate). C6-C7 are additional criteria added by this memo: compliance posture matters for Foundation's own MSB-registration pre-checks (FinCEN expects vendor SOC 2 documentation in the registration packet), and exit-planning matters for any load-bearing dependency on a regulated third-party vendor.

---

## 3. Candidate scorecard

Scoring: ✅ pass / ⚠️ pass with caveats / ❌ fail / ❓ insufficient public information. All scores reflect the vendor's publicly-stated capability as of **2026-04-27**; scores drift and must be re-verified at decision time if this memo's recommendation isn't ratified within 30 days.

| Criterion | Persona | Sumsub | Onfido |
|-----------|---------|--------|--------|
| **C1 US coverage** | ✅ All 50 states + DC. Strong driver's license + passport coverage. Liveness via video selfie + active-anti-spoof. SSN gated optional. | ✅ Coverage parity in US; documentation skews international. SSN supported. | ✅ All 50 states covered; older + slower onboarding for new doc types. |
| **C2 Sanctions screening** | ✅ OFAC + EU + UK + UN built into the standard `inquiry` flow. Real-time at onboarding; periodic re-screening configurable. PEP screening as add-on. | ✅ Strongest international sanctions list coverage of the three (built for global-first). PEP screening included in standard tier. | ⚠️ OFAC + UK supported in standard flow; broader international + PEP requires enterprise tier. |
| **C3 API stability + latency** | ✅ Public status page; clean recent incident history. p95 verification latency ~15-25s per published benchmarks. Webhooks reliable. React + Python SDKs. | ✅ Status page; incident history clean. p95 latency ~20-30s. Webhooks reliable. SDKs in multiple languages including Python. | ⚠️ Status page exists; some 2024 incidents on document-review queue (review backlog spiked under load). p95 latency higher (~30-45s) at peak. |
| **C4 Pricing at PRSM scale** | ✅ Published per-verification pricing: ~$1.50 at <10k/mo; ~$0.85-$1.10 at 100k+/mo (enterprise quoted). Free tier covers PoC volume. | ⚠️ Published pricing is opaque; enterprise sales is the only path. Quoted rates competitive at volume but no published anchor. | ❌ Enterprise-sales-only. Smaller customers report pricing 2-3x Persona at PRSM-projected volume. |
| **C5 Fraud-rate track record** | ✅ Public case studies (consumer fintech + creator platforms). Disclosed false-positive ~1.5%. Industry-standard configurability. | ✅ Strong fraud-rate posture via global-scale data; FPR ~1-2% configurable. Public case studies focused on European fintech. | ✅ Long track record (oldest of the three); fraud-rate parity. Older flow design means higher friction at the user end. |
| **C6 Compliance posture** | ✅ SOC 2 Type II + ISO 27001. GDPR + CCPA workflows documented. US data residency available (default in US tier). | ✅ SOC 2 Type II + ISO 27001. GDPR-focused (vendor based in UK + Cyprus). US data residency is opt-in. | ✅ SOC 2 Type II + ISO 27001. Long-running compliance posture. US data residency available. |
| **C7 Exit / data-portability** | ⚠️ Verification records exportable via API; format is JSON + linked documents. Operational cost of vendor-switching: medium — re-onboarding existing users avoidable if export is clean. | ⚠️ Similar export path; documentation slightly less explicit on retention windows post-cancellation. | ⚠️ Older API; export exists but exports are slower + format documented less clearly. |

---

## 4. Qualitative observations

### 4.1 Persona

- **Strengths:** modern developer experience (clear docs, SDK ergonomics, sandbox is genuinely usable), pricing transparency that's rare in the KYC space, US-first orientation matches Phase 5 §8.2 launch jurisdiction. Active integration ecosystem (consumer fintech, creator platforms — same shape as PRSM's expected user base).
- **Risks:** vendor maturity is good but still smaller than Onfido. International expansion would need re-evaluation against Sumsub. Pricing may escalate at scale (no contractual guarantees beyond enterprise quotes).
- **PRSM fit:** strong. Persona's `inquiry` flow maps cleanly to the existing `KycService` Protocol shape in `withdrawal_orchestrator.py:147-148`. Their "decision-on-completion" model fits the orchestrator's REQUESTED → KYC_CHECK → ORACLE_QUOTE state machine without any orchestrator-side rewrite.

### 4.2 Sumsub

- **Strengths:** the strongest international story (≥220 jurisdictions). Best-in-class for sanctions + PEP coverage. Pricing competitive at volume.
- **Risks:** US documentation thinner than Persona; sales engagement required for PRSM-scale rates. Vendor based in UK/Cyprus — US data residency is opt-in, which the FinCEN MSB application would need to verify. Enterprise-only pricing path adds friction to Foundation's contracting timeline.
- **PRSM fit:** technically strong but US-first launch isn't where Sumsub's advantages compound. Becomes the natural default if/when Phase 5 expands beyond US.

### 4.3 Onfido

- **Strengths:** longest track record. Enterprise-grade compliance posture. Mature webhook + retry semantics.
- **Risks:** pricing 2-3x Persona at expected volume per industry reports. Slower onboarding for new document types (matters in 2-3 years, not at launch). 2024 incident history flags a document-review-queue scaling issue at peak load that hasn't been re-stress-tested publicly.
- **PRSM fit:** weakest of the three. The cost premium isn't justified by demonstrable advantage at PRSM's projected scale; the older flow design adds user-facing friction that hurts onboarding metrics (Phase 5 plan §7 implicitly cares about onboarding TTO).

---

## 5. Recommendation

**Primary: Persona** — conditional on PR-1 pricing + PR-2 compliance review.

**Rationale:**
- Best US coverage + developer experience among the three; Phase 5 §8.2 launches US-only, which is exactly Persona's strongest market.
- Pricing transparency that's rare in KYC + already inside PRSM's plan §10 cost envelope at projected first-18-month verification volume.
- API stability + latency are best-of-class for the US flow; webhook reliability matters because the orchestrator's KYC_CHECK state machine is webhook-driven.
- Compliance posture (SOC 2 Type II + ISO 27001 + US data residency) clears Foundation's MSB-registration packet requirements.
- Comfortable operational risk: SDK is stable, status page is clean, sandbox works.

**Runner-up: Sumsub** — retain as the explicit fallback if PR-1 pricing or PR-2 compliance review fails at sign-time, AND as the natural primary for Phase 5.x international expansion (re-review triggered by §7 T4).

**Rejected: Onfido** — cost premium without demonstrable advantage; slower flow + recent scaling-incident history make the additional spend hard to defend.

### 5.1 The conditions attached to Persona

This memo's recommendation IS conditional. Two items must clear at sign-time:

**PR-1 — pricing at 100k verifications/month must stay below $1.10 / verification.** Persona's published tiers as of 2026-04-27 support this. At sign-time, get a written 12-month price-stability commitment (standard for their enterprise customers) OR have Sumsub pricing as a signed-contract fallback.

**PR-2 — compliance documentation review by Foundation Compliance Officer must complete.** Scope: Persona's SOC 2 Type II report + ISO 27001 certificate + GDPR/CCPA workflow + US data residency confirmation, all included in the FinCEN MSB application packet as vendor-supporting documentation. If material concerns arise (e.g., an active audit finding on the SOC 2 report, or a data-residency caveat that conflicts with PRSM's MSB representations), fall back to Sumsub.

If PR-1 or PR-2 fails, switch to Sumsub. Sumsub's own conditions at that point would be:
- SS-1: lock in a written enterprise-tier per-verification rate ≤$1.10 at 100k/mo before engineering starts.
- SS-2: confirm US data residency can be enforced contractually (Sumsub's default is EU-resident; this is a non-default config).

---

## 6. Green-light checklist for Phase 5 Tasks 2-4

Phase 5 Tasks 2 (KYC integration), 3 (Stripe production), and 4 (Coinbase Exchange) should NOT begin engineering until **all** of the following clear. Tasks 3 and 4 depend on Task 2's vendor signal (tier output drives charge limits + swap-volume gating), so Task 2's gates are the load-bearing path.

- [ ] **K1 Vendor signed.** Foundation product + legal + Compliance Officer have countersigned the Persona (or fallback Sumsub) Master Services Agreement + DPA.
- [ ] **K2 Credentials issued.** Persona environment keys (sandbox + production) in Foundation secrets manager. Webhook signing secrets configured. Staging + production environments separate.
- [ ] **K3 Pricing commitment received.** PR-1 (or SS-1 if Sumsub) written 12-month commitment on file.
- [ ] **K4 Compliance review complete.** PR-2 report filed with Foundation Compliance Officer + included in FinCEN MSB application packet (or queued for inclusion). No blocking findings.
- [ ] **K5 MSB registration filed OR scheduled.** Phase 5 Task 1 explicitly requires FinCEN MSB registration before any production launch. KYC vendor integration can begin BEFORE registration completes (since sandbox doesn't require it), but production go-live blocks on the registration receipt.
- [ ] **K6 Integration proof-of-concept runs.** Minimum viable integration in a throwaway repo: orchestrator KYC_CHECK state → Persona inquiry create → user completes sandbox flow → webhook fires → orchestrator advances to ORACLE_QUOTE. Run time ≤10 minutes. Validates the chosen vendor composes with the shipped Phase 5 Task 5 orchestrator (`withdrawal_orchestrator.py`).
- [ ] **K7 Sandbox telemetry baseline.** 1 week of Persona sandbox usage to measure end-to-end verification latency under realistic test load. Establishes the p95 baseline that the production flow will be monitored against.

K1-K5 are Foundation-side concerns (vendor + legal + compliance + regulatory); K6-K7 are engineering-side. K1-K4 must clear before K6-K7 begin; K5 (MSB registration) blocks production go-live but does NOT block sandbox engineering.

**Important sequencing note vs Phase 4:** Phase 4 Task 4 was blocked on Foundation contracting alone. Phase 5 Task 2 has the same vendor-contracting gate PLUS a regulatory-filing gate (K5 / FinCEN MSB) that is separate from the vendor decision. Foundation's Phase 5 critical path is K5, not K1.

---

## 7. Re-review triggers

This memo's recommendation should be re-examined before engineering-commit if any of:

- **T1: Date drift** — memo older than 60 days. Vendor pricing / API stability / incident history drifts fast; stale recommendations shouldn't bind. KYC space sees 1-2 material vendor updates per quarter.
- **T2: Vendor event** — Persona acquired (reduces commitment continuity), Persona ships breaking API change (integration cost rises), Persona pricing changes materially (PR-1 commitment violated), Persona incident reveals systemic posture issue.
- **T3: New candidate emerges** — a fourth vendor publishes a meaningful US-launch offering with material pricing or compliance-posture advantage. Re-evaluate if their published capability clears C1-C5 with measurably better terms.
- **T4: International expansion** — when Phase 5.x extends beyond US (per Phase 5 plan §8.2 sequencing: Canada → UK → EU → APAC), re-run §3 evaluation. Sumsub becomes the structural primary for any non-US launch; Persona's US-first orientation is a decreasing fit as the user base globalizes.
- **T5: Foundation Compliance Officer changes** — the named Compliance Officer is the ratifier of the K4 review. A change in that role within 6 months of signing should trigger a re-confirm-not-re-decide review (does the new officer accept the prior review's findings? Yes → continue; No → run K4 again).

---

## 8. Relationship to other work

- **Phase 5 plan §8.1** — source open-issue; this memo resolves it.
- **Phase 5 plan §8.2** (jurisdictional scope) — this memo is US-only; T4 above triggers re-review at international expansion.
- **Phase 5 plan §8.4** (custody handling during swap window) — KYC tier output gates max swap-window exposure, so vendor choice indirectly affects custody-window risk. Sumsub's stronger PEP screening would reduce some classes of high-risk-user exposure; not a determinative factor but worth documenting.
- **Phase 5 Task 1 (compliance + legal sign-off — NOT YET DONE)** — drives the FinCEN MSB filing. K4 + K5 above are the engineering-visible artifacts of Foundation's separate compliance track. Coordinate timelines.
- **Phase 5 Task 5 (withdrawal orchestrator — SHIPPED)** — vendor-agnostic by design (`KycService` Protocol). Vendor choice does NOT change any shipped orchestrator code.
- **Phase 5 Tasks 2-4 vendor adapter scaffolds (SHIPPED)** — `prsm/economy/payments/vendor_adapters.py`. Adapter classes are stub implementations of the `KycService` / `OracleService` / `SwapService` / `PayoutService` Protocols; vendor choice determines which scaffold gets production wiring. Persona-specific wiring lands in Task 2 proper.
- **Phase 4 Privy memo** — same pattern applied to a different vendor space. Cross-reference for ratification process. Both memos block on Foundation operational items, not engineering.
- **Foundation jurisdiction scoping doc (#104)** — informs Phase 5 §8.2 and this memo's US-first framing.

---

## 9. Risks

### R1: Persona enterprise pricing escalates post-sign

KYC vendors raise prices periodically; Persona's last published-tier change was 2024. A second escalation within the 12-month commitment window would create a choice between (a) renegotiate, (b) tolerate, (c) migrate. Mitigation: PR-1 commitment makes option (a) the default within the window; re-review at window end. Sumsub fallback is always viable.

### R2: Persona acquired by a hostile-to-PRSM actor

KYC vendors aggregate sensitive identity data; an acquirer with conflicting incentives (e.g., advertising-driven acquirer, foreign-state-aligned acquirer) is a material risk. Low probability but high impact. Mitigation: K4 (Foundation Compliance Officer review) includes baseline acquisition-risk note. Fallback to Sumsub is always viable via re-running this memo's §5 evaluation.

### R3: Compliance review finds blocking issue on short timeline

If PR-2 review takes >4 weeks, Phase 5 Task 2 launch slips. Mitigation: PR-2 budget scoped as 2-week block at K4; if it slips past 4 weeks, auto-pivot to Sumsub. SOC 2 + ISO 27001 reviews are routine for procurement teams; the bottleneck is usually scheduling, not findings.

### R4: FinCEN MSB registration timeline slips

K5 is outside this memo's scope but blocks production go-live. FinCEN registration typically takes 30-60 days from submission; state-by-state MTL registrations take 6-18 months for full national coverage. Mitigation: launch with FinCEN-only + a small set of MTL-completed states; expand state coverage as registrations land. The orchestrator already supports per-user jurisdiction-aware tier assignment, so this is a configuration gate, not a re-engineering event.

### R5: User-data export degradation between API versions

Persona's data-export story is documented but not contractually guaranteed. Mitigation: Foundation Compliance Officer retains a quarterly spot-check of the verification-record export flow as part of ongoing operator-trust maintenance. Same risk pattern + mitigation as Phase 4 R4 (Privy).

### R6: This memo's recommendation ratified too slowly

Phase 5 Tasks 2-4 engineering stays blocked for the memo-ratification window. Phase 5 Task 5 (withdrawal orchestrator) is already shipped; Task 1 (legal) is the parallel critical-path work. Mitigation: this memo is short enough for 1-meeting ratification; Foundation product lead + legal + Compliance Officer alignment should take days, not weeks. If a Compliance Officer is not yet named, surface that as the actual blocker — it's a gating Foundation governance task, not a memo-quality issue.

### R7: PRSM's user data + KYC vendor data jurisdictional mismatch

If PRSM operates from a non-US Foundation entity (Cayman nonprofit per project memory) but contracts with Persona (US data residency), the inter-jurisdiction data flow needs explicit DPA terms. Mitigation: K1 includes a DPA review as part of the Master Services Agreement countersigning; legal should specifically vet the controller/processor split between Foundation, Prismatica (DE C-corp), and Persona.

---

## 10. Success criteria

This memo is considered successful when:

1. **A named vendor is chosen** — Persona or Sumsub — in writing by Foundation Compliance Officer + product lead.
2. **Green-light checklist K1-K7 is complete** (or K1-K4 are complete + K5-K7 are queued).
3. **Phase 5 Task 2 engineering begins** with unambiguous vendor context.
4. **The re-review cadence (§7) is entered into Foundation Compliance + DevRel calendars** — next review due at 60-day mark regardless of whether engineering has started.
5. **The Phase 5 Task 1 compliance track is in motion** — Compliance Officer named, MSB registration drafted or filed, ToS+Privacy updates on legal's queue. This memo's recommendation is moot if Task 1 doesn't progress.

---

## 11. Changelog

- **0.1 (2026-04-27):** initial decision memo. Recommends Persona conditional on PR-1 + PR-2. Phase 5 Tasks 2-4 remain blocked on K1-K7 green-light checklist; this memo does not itself unblock engineering. Phase 5 Task 1 (compliance + legal) is a parallel Foundation track and is the practical critical path.
