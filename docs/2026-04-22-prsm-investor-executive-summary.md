# PRSM — Investor Executive Summary

**Prismatica, Inc.** | **Reg D Rule 506(c)** | **April 2026**
**Contact:** *[founder contact]* | **Data room:** *[access link]*

**Supersedes:** `docs/2026-04-21-prsm-investor-executive-summary.md` (superseded on 2026-04-22 after Phase 7 + 7.1 + 7.1x shipment + pre-audit hardening completion). The prior summary ended at Phase 3; this version covers the verification triad (A+B+C) now engineering-complete and audit-ready.

---

## The opportunity

Frontier AI inference is centralizing faster than any prior technology wave. Three companies control the GPU supply, one company controls the attestation root, and zero open-weights labs can safely publish state-of-the-art models without expecting them to be exfiltrated and relisted under different branding. The industry needs neutral infrastructure that is cryptographically credible, economically aligned, and permissionless — not another vertically-integrated stack.

**PRSM is a P2P protocol that turns underutilized consumer electronics, T3 cloud arbitrage, and dedicated T4 meganodes into a single priced mesh for AI inference, storage, and data.** Contributors earn FTNS tokens for sharing latent resources. Third-party LLMs (local, API, or OAuth) consume PRSM services via MCP tools. Reasoning happens in the LLM; execution happens on PRSM nodes.

## What's shipped

As of 2026-04-22, on `main` at `github.com/Ryno2390/PRSM`:

| Phase | Coverage | Status |
|---|---|---|
| **Phase 1.3** | On-chain provenance + royalty distribution | Engineering complete. Contracts deployed, 7-day Sepolia bake-in passed. Mainnet deploy pending multi-sig hardware quorum. |
| **Phase 2** | Remote compute dispatch + Ed25519 receipts + FTNS escrow + topology randomization + TEE attestation schema | Shipped. `phase2-merge-ready-20260420`. |
| **Phase 2 Rings 7-10** | Confidential compute: TEE runtime + DP noise + tensor-parallel sharding + security hardening | Shipped. v0.35.0 on PyPI. |
| **Phase 3** | FTNS-priced marketplace with price discovery + reputation + orchestration | Shipped. `phase3-merge-ready-20260420`. |
| **Phase 3.1** | Batched on-chain settlement (Tier A receipt-only verification) | Merge-ready. `phase3.1-merge-ready-20260421`. |
| **Phase 7** | Tier C: provider stake + slashing (DOUBLE_SPEND / INVALID_SIGNATURE) | Merge-ready. `phase7-merge-ready-20260421`. |
| **Phase 7.1** | Tier B: k-of-n redundant execution with CONSENSUS_MISMATCH slashing | Merge-ready. `phase7.1-merge-ready-20260421`. |
| **Phase 7.1x + pre-audit hardening** | MIN_SLASH_GAS floor + ConsensusChallengeSubmitter + SQLite queue + claim-lease + consensus_group_id sybil fix + default-deny retry + SettlementContractClient catch-up | Merge-ready. `phase7.1x-merge-ready-20260422-2`. |

**Verification triad engineering-complete.** Three verification tiers now ship in the marketplace:

- **Tier A (receipt-only)** for open jobs — Phase 3.1 batched settlement.
- **Tier B (redundant execution)** for high-value jobs — Phase 7.1 k-of-n consensus + CONSENSUS_MISMATCH challenges.
- **Tier C (single-provider + stake-slash)** for premium/critical jobs — Phase 7 StakeBond.

**Test suite: 427 green** (142 Solidity + 283 Python unit + 2 Python end-to-end on a live hardhat node).

**Audit bundle ready.** Three merge-ready trees (Phase 3.1 + Phase 7 + Phase 7.1x) bundled for a single external audit engagement. All six published review-gate findings resolved pre-audit; audit-scope coordinator at `docs/2026-04-21-audit-bundle-coordinator.md`. Hardware-gated on the Foundation multi-sig for deploy; no engineering blockers.

## The technical moat

1. **CIS-1 confidential-silicon standard.** RFC-style spec defining three conformance levels (C1 / C2 / C3) with quantified weight-exfiltration lower bounds: 2²⁰ inferences at baseline to 2⁴⁰ at SOTA tier. Governance-separated: Foundation owns the standard; any fab can implement. **No one else in this space has specified a frontier-grade confidentiality target.**
2. **Vendor-independent attestation.** On-chain AttestationRegistry on Base binds chip identity to protocol governance, not to NVIDIA or any single CA. Multi-manufacturer requirement at C2+ is a **prohibited amendment** (cannot be removed without dissolving the Foundation).
3. **Complete economic backstop.** StakeBond + BatchSettlementRegistry + three slashable reason codes (DOUBLE_SPEND, INVALID_SIGNATURE, CONSENSUS_MISMATCH) + 70/30 challenger/Foundation bounty split + consensus_group_id sybil mitigation that requires k-distinct-provider keys to attack. Production-viable consensus-challenge submitter service with SQLite persistence + multi-runner claim-lease for operational deployments.
4. **Eight-layer research track with 6 partner-handoff-ready docs.** Defense-stack composition analysis, MPC scoping, activation-inversion red-team methodology, PQ-signature watch memo, compression benchmark plan, supply-diversity standard. Remaining 2 items (FHE / Tier-C-hardening) are pure cryptography-partnership arcs pending partner identification.

## The governance-credibility moat

Investors who understand Web3 history know the real question isn't "what's the tech?" — it's "who captures the rents?" PRSM answers structurally, not promissorily:

- **Foundation cannot build chips.** Prohibited amendment.
- **Foundation cannot revenue-share with Prismatica.** Prohibited amendment.
- **Founder must pick one entity after Year 3.** Cannot serve both Foundation board and Prismatica executive indefinitely.
- **Foundation certifies Prismatica's chips on the same terms as any other implementer.** C2+ certification explicitly requires ≥2 independent manufacturers.
- **Transparency default.** Every on-chain transaction, committee vote, certification decision, and financial statement is public on a fixed cadence.
- **Supply-side diversity standard** (new: PRSM-SUPPLY-1). Measures provider / geographic concentration; activates soft-cap mechanisms + diversity bonus once any single provider exceeds 30% of supply or any single country exceeds 40%. Structural answer to "will PRSM effectively run on 3 hyperscalers?"

Details: `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md`, `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md`.

## The regulatory clarity

- **FTNS is distributed only as compensation for services rendered.** No retail sale. No bonding curve. No ICO.
- **Howey analysis passes cleanly** for FTNS recipients: no investment of money, no expectation of profit from others' efforts. Mirrors Bitcoin's regulatory posture.
- **Bootstrap capital raises through Prismatica equity** (Reg D 506(c)) — a cleanly-classified security — NOT through FTNS distribution. Investors buy equity; FTNS is a separate instrument with separate treatment.

Details: `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md`.

## The deal

- **Instrument:** Prismatica Inc. equity, Delaware C-corp, Reg D Rule 506(c) accredited-only.
- **Use of proceeds:** First-implementer CIS chip design and fabrication, T4 meganode deployment, ML research team, commissioned-data pipeline, external audit retainer for the three-phase bundle, 24-36 month operating runway.
- **Investor returns:** Prismatica's enterprise-value growth (operating revenue from managed meganodes + CIS chip licensing + FTNS treasury appreciation from organic earning, not speculation).
- **Round size, valuation, terms:** *[specified in pitch deck and data room]*.

## Why now

1. **The verification triad is engineering-complete.** Tier A receipt-only, Tier B redundant execution, Tier C stake-slash — all three ship at merge-ready as of 2026-04-22. Frontier labs asking "what defenses does PRSM have?" get a specific shipped-stack answer, not a roadmap.
2. **H100 Confidential Compute is shipping** — raising the baseline trust floor and making the market ready for a protocol that treats attestation as a first-class primitive.
3. **Frontier labs are publicly asking "what would it take for us to publish weights to a decentralized substrate?"** — PRSM's CIS-1 roadmap + the new R8 composition analysis (`docs/2026-04-22-r8-defense-stack-composition.md`) give the concrete technical answer.
4. **Regulatory posture has matured.** MiCA, FINMA guidance, Cayman Foundation Company framework all provide workable non-profit-foundation paths that did not exist in 2019.
5. **Audit engagement is the last hardware-gated step.** Three merge-ready trees bundled for a single auditor; six published review findings all resolved pre-audit. Multi-sig hardware assembly + retainer payment activates the engagement.

## Team

**Founder:** [name, role, background summary — bio in data room]

Foundation board and Prismatica advisory roles currently in recruitment per `PRSM-GOV-1 §8.4` (Year 1 milestones). First external board members targeted for Q3 2026.

---

## Read next

- `docs/TECH_CHOICES.md` — why Ethereum / Base / ERC-20, with investor-pushback responses.
- `docs/2026-04-21-prsm-cis-1-confidential-inference-silicon.md` — the silicon standard in full.
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — governance charter.
- `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` — tokenomics standard.
- `docs/2026-04-21-prsm-economic-model-white-paper.md` — bear/base/bull scenarios, 5/10-year horizons.
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — supply-diversity governance standard (new).
- `docs/2026-04-22-phase7.1x-audit-prep.md` — audit scope + known issues bundle (the document an external auditor reads first).
- `docs/2026-04-21-audit-bundle-coordinator.md` — cross-phase seams map for bundled audit engagement.
- `docs/2026-04-14-phase4plus-research-track.md` — 8-item research roadmap; 6 items have partner-handoff-ready docs as of 2026-04-22.
- `README.md` — protocol architecture and shipped rings.

*This document is a summary and is not an offer to sell securities. Any offer will be made only through formal offering materials to accredited investors under Regulation D Rule 506(c).*
