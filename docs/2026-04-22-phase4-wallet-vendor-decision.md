# Phase 4 §8.1 — Embedded-Wallet Vendor Decision

**Document identifier:** PHASE4-WALLET-VENDOR-1
**Version:** 0.1 Draft
**Status:** Partner-handoff-ready decision memo. Resolves the Phase 4 design plan §8.1 open-issue by recommending a specific vendor, documenting evaluation criteria, and specifying the green-light checklist that unblocks Phase 4 Tasks 3/4 engineering. **Not yet a signed decision** — this memo recommends; Foundation product lead ratifies.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Related documents:**
- `docs/2026-04-22-phase4-wallet-sdk-design-plan.md` §8.1 — source open issue with three candidates (Privy / Web3Auth / Magic.link).
- `prsm/interface/onboarding/siwe.py`, `prsm/interface/onboarding/wallet_binding.py`, `prsm/interface/display.py` — Phase 4 backend Tasks 1, 2, 5 (SHIPPED). Vendor choice determines what Tasks 3/4 actually build against.
- `tests/integration/test_phase4_wallet_sdk_e2e.py` — Phase 4 backend E2E suite. Any vendor selection must keep these 9 tests green against the same Protocol contracts.

---

## 1. Purpose

Phase 4 design plan §8.1 names three candidate embedded-wallet SDK vendors as the path for "sign in with email → generate custodial-or-MPC wallet → no seed phrase ever surfaced to user":

- **Privy** — "most polished UX, generous free tier, MPC + embedded-wallet export."
- **Web3Auth** — "older, widely-integrated, non-custodial option."
- **Magic.link** — "email + passkey, strong mobile support."

Plan §8.1 records a tentative recommendation for Privy but defers the binding decision to "Task 4 kickoff." Phase 4 Tasks 3 (Coinbase Wallet SDK primary) and 4 (WalletConnect v2 + embedded-vendor fallback) both depend on the vendor choice:

- Task 4 scope IS "integrate the selected embedded-wallet vendor."
- Task 3 scope depends on vendor because the Coinbase Wallet SDK + embedded-vendor complement each other (Coinbase Wallet = user already has one; embedded = doesn't).
- Task 6 review gate cannot pass without an end-to-end onboarding test against the chosen vendor.

Without a named choice, Phase 4 Tasks 3/4 stay in scoping limbo. This memo resolves that.

**What this memo does:**
1. Re-states the evaluation criteria from plan §8.1 in measurable terms.
2. Scores each candidate against criteria as of 2026-04-22.
3. Recommends a specific vendor with explicit rationale.
4. Lists the green-light checklist that must pass before Phase 4 Tasks 3/4 engineering begins.
5. Specifies the follow-up conditions under which the decision should be re-examined.

**What this memo does NOT do:**
- Not a contract. Foundation product lead + legal sign the actual SDK-license agreement before integration.
- Not an integration plan. Phase 4 Tasks 3/4 design + TDD follows this memo's choice.
- Not a permanent lock-in. §7 specifies re-review triggers.

---

## 2. Criteria

From plan §8.1, normalised to measurable form:

| Criterion | What "pass" looks like |
|-----------|------------------------|
| **C1 Non-custodial option** | Users can export their wallet to self-custody without vendor permission or additional fees. |
| **C2 Base mainnet support** | Base chain (8453) supported as first-class network in the SDK. EIP-1559 transaction signing, ERC-20 (FTNS) balance display, event subscription. |
| **C3 Pricing at PRSM scale** | Free tier supports ≥10k monthly active wallets; paid scale cost ≤$0.10 per MAU at 100k MAU. Pricing model published + stable. |
| **C4 SDK stability** | Production-grade React + Python SDK. Semver released. API changes between minor versions accepted with ≥6-month deprecation windows. Public incident history available. |
| **C5 Open-source posture** | SDK client code open-source OR auditable. Foundation security team can review the code that runs in our users' browsers. |
| **C6 Exit / migration story** | If the vendor is acquired / pivots / shuts down, how does a PRSM user migrate their wallet? What's the operational cost to PRSM of vendor-switching mid-deployment? |

C1-C4 are Phase 4 plan §8.1's explicit criteria. C5-C6 are additional criteria added by this memo — Phase 4's plan was written assuming vendor-level details but didn't name the security-review or exit-planning criteria explicitly. These matter for a load-bearing dependency on a third-party vendor.

---

## 3. Candidate scorecard

Scoring: ✅ pass / ⚠️ pass with caveats / ❌ fail / ❓ insufficient public information. All scores reflect the vendor's publicly-stated capability as of **2026-04-22**; scores drift and must be re-verified at decision time if this memo's recommendation isn't ratified within 30 days.

| Criterion | Privy | Web3Auth | Magic.link |
|-----------|-------|----------|------------|
| **C1 Non-custodial export** | ✅ MPC key split + user-exportable private key (documented). | ✅ Shamir-split shares; user holds device share + backup share; can reassemble off-vendor. | ⚠️ Delegated key management via Magic's DID; export requires migration flow that involves vendor consent. |
| **C2 Base mainnet support** | ✅ First-class. Base in supported-networks list; docs have Base-specific examples. | ✅ Base supported via wallet-core plugin. EIP-1559 support tested. | ⚠️ Base supported but less Base-specific documentation than Privy or Web3Auth. |
| **C3 Pricing at PRSM scale** | ✅ Free tier covers first 10k MAU. Paid tier ~$0.05-$0.08 / MAU at 100k+ per published pricing. | ⚠️ Free tier tighter (~1k MAU). Paid tier pricing opaque (requires sales contact). | ⚠️ Free tier ~10k MAU but reporting in 2024-2025 about price changes + enterprise-tier gating. |
| **C4 SDK stability** | ✅ React SDK 2.x stable; Python SDK available but thinner. Public changelog + migration notes. | ✅ SDK venerable (2020+); React + server-side SDKs robust. | ✅ SDK mature (2020+); well-documented React integration. |
| **C5 Open-source posture** | ⚠️ Client SDK source available but some server-side components proprietary. Enough for client-side security review. | ✅ Mostly open-source; client SDK + core.js open. | ❌ Client SDK is a thin wrapper around proprietary Magic API. Security review limited. |
| **C6 Exit / migration story** | ⚠️ User key export exists, but migration to self-custody requires user action at our prompt. Operational cost of switching: medium — SDK-specific login flow is unique to each vendor. | ⚠️ Similar export path; Shamir-backup recovery is user-driven. | ❌ Tight vendor coupling. Magic DIDs are portable in principle but not seamlessly; migration is a significant re-onboarding event. |

---

## 4. Qualitative observations

### 4.1 Privy

- **Strengths:** strong UX, progressive-trust model (email → passkey → self-custody available in-app). Active Base + Ethereum ecosystem engagement. React + server SDK parity. Public incident history on their status page is clean.
- **Risks:** some server-side infrastructure proprietary. Pricing has escalated once (2024) — unclear trajectory at PRSM scale. Vendor maturity is good but company is still Series A.
- **PRSM fit:** natural. Their MPC + exportable-key model matches the plan §3.4 "user trusts vendor's key-custody; not in PRSM's path" framing.

### 4.2 Web3Auth

- **Strengths:** oldest + most battle-tested of the three. Shamir-based recovery feels architecturally aligned with Phase 7-storage Task 7 (also Shamir). Strongest open-source posture.
- **Risks:** free tier is the tightest (~1k MAU). Enterprise pricing requires sales engagement — unknown at quote time. UX is less polished than Privy; documentation is dense but less marketing-smooth.
- **PRSM fit:** good technically. Pricing uncertainty makes launch-time scale planning harder.

### 4.3 Magic.link

- **Strengths:** simplest integration. Mobile story (iOS SDK) is the best of the three.
- **Risks:** closed-source client SDK. Vendor coupling is the strongest; migration is disruptive. 2024-2025 pricing changes created some operator churn.
- **PRSM fit:** weakest. PRSM's ethos (decentralised, user-sovereign) conflicts with Magic's more-centralised architecture. Ruling this out is straightforward.

---

## 5. Recommendation

**Primary: Privy** — conditional on C3 pricing + C5 security review.

**Rationale:**
- Best UX among the three candidates; Phase 4 acceptance criterion in plan §7 targets <90s onboarding, which Privy's flow directly supports.
- Non-custodial export model aligns with PRSM's sovereignty ethos.
- Base mainnet support is first-class + Base-specific.
- Pricing is transparent + covers our expected first-18-month scale under the free tier.
- Comfortable operational risk: SDK is stable, company is real, incident history is clean.

**Runner-up: Web3Auth** — retain as the explicit fallback if C3 or C5 conditions on Privy fail at sign-time.

**Rejected: Magic.link** — closed-source client SDK and weak exit story are structural misalignments with PRSM.

### 5.1 The conditions attached to Privy

This memo's recommendation IS conditional. Two items must clear at sign-time:

**PV-1 — pricing at 100k MAU must stay below $0.10 / MAU.** Privy's published tiers as of 2026-04-22 support this. At sign-time, get a written 12-month price-stability commitment (standard for their enterprise customers) OR have Web3Auth pricing as a signed-contract fallback.

**PV-2 — client-side SDK security review by Foundation security team must complete.** Scope: review of the JS bundle shipped in users' browsers. If material security concerns arise (key-material leak patterns, supply-chain audit findings on their npm publishing chain), fall back to Web3Auth.

If PV-1 or PV-2 fails, switch to Web3Auth. Web3Auth's own conditions at that point would be:
- WA-1: lock in a named enterprise-tier MAU rate before engineering starts.
- WA-2: no equivalent to PV-2 — Web3Auth's open-source posture satisfies it without the same review depth.

---

## 6. Green-light checklist for Phase 4 Tasks 3/4

Phase 4 Tasks 3 and 4 should NOT begin engineering until **all** of:

- [ ] **G1 Vendor signed.** Foundation product + legal have countersigned the Privy (or fallback Web3Auth) SDK license agreement.
- [ ] **G2 Credentials issued.** Privy app ID + environment keys in Foundation secrets manager. Staging + production environments separate.
- [ ] **G3 Pricing commitment received.** PV-1 (or WA-1 if Web3Auth) written 12-month commitment on file.
- [ ] **G4 Security review complete.** PV-2 report filed with Foundation security; no blocking findings.
- [ ] **G5 Integration proof-of-concept runs.** Minimum viable integration in a throwaway repo: email sign-in → SIWE → wallet binding → display balance. Run time ≤5 minutes. Validates that the chosen vendor composes with the shipped Phase 4 backend (SIWE / wallet_binding / display).
- [ ] **G6 Sandbox telemetry baseline.** 1 week of Privy-sandbox usage to baseline "onboarding < 90s" acceptance per plan §7. This is a sanity check, not a pass/fail gate; if sandbox onboarding is >2min we have an SDK integration problem, not a vendor problem.

G1-G4 are Foundation-side concerns; G5-G6 are engineering-side. All six must clear; sequencing in any order that suits Foundation ops.

---

## 7. Re-review triggers

This memo's recommendation should be re-examined before engineering-commit if any of:

- **T1: Date drift** — memo older than 60 days. Vendor pricing / SDK stability / incident history drifts fast; stale recommendations shouldn't bind.
- **T2: Vendor event** — Privy acquired (reduces commitment continuity), Privy ships breaking SDK change (integration cost rises), Privy pricing changes materially (PV-1 commitment violated).
- **T3: New candidate emerges** — a fourth vendor publishes a meaningful offering (e.g., Coinbase-native embedded-wallet). Re-evaluate if their published capability clears C1-C4.
- **T4: Phase 4 scope change** — if Phase 4 frontend drops embedded-wallet entirely (e.g., Coinbase Wallet + WalletConnect alone is deemed sufficient, leaving email onboarding to a Phase 5+ concern), this memo becomes moot.

---

## 8. Relationship to other work

- **Phase 4 plan §8.1** — source open-issue; this memo resolves it.
- **Phase 4 plan §8.5** — smart-wallet gas sponsorship open issue. Privy supports ERC-4337; if that feature becomes a requirement, Privy's commitment strengthens further.
- **Phase 4 backend Tasks 1, 2, 5 (SHIPPED)** — vendor choice does NOT change any shipped backend. SIWE + wallet-binding + USD display are vendor-agnostic by design.
- **Phase 5 plan** — Phase 5 §3.5 (withdrawal flow) uses the same Phase 4 wallet binding. Vendor choice here does NOT create new Phase 5 dependencies.

---

## 9. Risks

### R1: Privy enterprise pricing escalates post-sign

Privy raised prices once in 2024. A second escalation within the 12-month commitment window would create a choice between (a) renegotiate, (b) tolerate, (c) migrate. Mitigation: PV-1 commitment makes option (a) the default within the window; re-review at window end.

### R2: Privy acquired by a hostile-to-PRSM actor

Low probability but high impact. Mitigation: G4 (Foundation security review) includes baseline acquisition-risk note. Fallback to Web3Auth is always viable via re-running this memo's §5 evaluation.

### R3: Security review finds blocking issue on short timeline

If PV-2 review takes >4 weeks, Phase 4 frontend launch slips. Mitigation: PV-2 budget scoped as 2-week block at G4; if it slips past 4 weeks, auto-pivot to Web3Auth (where C5 is stronger).

### R4: User-export path degrades between SDK versions

Privy's non-custodial export story is written into their docs but not contractually guaranteed. Mitigation: Foundation security retains a quarterly spot-check of the exported-key flow as part of ongoing operator-trust maintenance.

### R5: This memo's recommendation ratified too slowly

Phase 4 Tasks 3/4 engineering stays blocked for the memo-ratification window. Phase 4 backend is already shipped; the frontend gap means consumers can't onboard via the wallet SDK path. Mitigation: this memo is short enough for 1-meeting ratification; Foundation product lead + legal alignment should take days, not weeks.

---

## 10. Success criteria

This memo is considered successful when:

1. **A named vendor is chosen** — Privy or Web3Auth — in writing by Foundation product lead.
2. **Green-light checklist G1-G6 is complete**.
3. **Phase 4 Tasks 3/4 engineering begins** with unambiguous vendor context.
4. **The re-review cadence (§7) is entered into Foundation DevRel's calendar** — next review due at 60-day mark regardless of whether engineering has started.

---

## 11. Changelog

- **0.1 (2026-04-22):** initial decision memo. Recommends Privy conditional on PV-1 + PV-2. Phase 4 Tasks 3/4 remain blocked on G1-G6 green-light checklist; this memo does not itself unblock engineering.
