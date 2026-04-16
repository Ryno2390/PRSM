# Hybrid Tokenomics Model — Legal & Governance Tracking

**Status:** Proposal under review. Not ratified. Not yet sent to counsel. Captured 2026-04-14. Updated 2026-04-15 to reflect pivot to equity-investment bootstrap architecture (see `PRSM_Tokenomics.md` §3 and `PRSM_Vision.md` §9); the hybrid tokenomics model continues to apply but with reduced urgency since the primary regulatory risk from token sales has been eliminated.

**Context:** `PRSM_Tokenomics.md` Section 11 proposes replacing the baseline 2% protocol tax with a hybrid structure: reduced 50 bps tax floor + 15-30% foundation equity stake in Prismatica (the for-profit entity seeding PRSM's base knowledge layer). This is a governance-and-legal workstream running in parallel to engineering phases. This document tracks its progression so it does not stall.

**Cross-references:**
- Design: `PRSM_Tokenomics.md` Section 11.
- Cross-link: `PRSM_Tokenomics.md` Section 8 Open Question #6.
- Vision doc: `PRSM_Vision.md` Phase 2/3 revenue projections currently assume 2% tax; updates pending if hybrid ratified.

## Decision Gates

### Gate 1: Internal Alignment — Does the economic math hold?

**Owner:** founders.

**Inputs:** Tokenomics §11.4 scenario math (bull / base / bear across 10-50% equity stakes).

**Questions to resolve:**
- Is a 25% foundation equity stake the right target? (§11.4 recommends 25% as the upper end before Prismatica founder motivation erodes; 15-25% is the practical range.)
- Is a 50 bps tax floor the right floor? Too low → foundation underfunded during bear. Too high → defeats the "reads as free protocol" marketing advantage.
- Does the revenue mix (tax + dividends + treasury appreciation) match foundation's operating cost trajectory for years 1-5?

**Status:** open. Blocked on founder time + Prismatica founder conversation.

**Exit criterion:** founders sign off on a specific stake % and tax floor to present to counsel.

### Gate 2: Prismatica Founder Alignment

**Owner:** founders (PRSM) + Prismatica founders.

**Questions to resolve:**
- Are Prismatica founders willing to accept (e.g.) 25% foundation equity in exchange for baseline protocol positioning and reduced user-facing tax?
- What governance rights accompany the foundation equity? Board seat? Observer seat? Information rights only?
- Liquidity rights: can the foundation sell its Prismatica stake in the future? To whom? Approval conditions?

**Status:** open. Requires separate conversation with Prismatica founding team.

**Exit criterion:** term sheet signed between PRSM Foundation (in formation) and Prismatica covering stake %, governance rights, and exit mechanics. Non-binding but concrete enough for legal.

### Gate 3: Legal Consultation

**Owner:** external counsel (nonprofit + securities expertise).

**Inputs:** Signed term sheet from Gate 2, Tokenomics §11 in full, Vision doc Section 8 (bonding curve and AMM transition mechanics).

**Questions counsel must answer (per §11.7):**
1. **UBIT exposure.** Are dividends from Prismatica to the PRSM Foundation treated as unrelated business income? Structure to minimize.
2. **Private inurement / private benefit.** How to structure so Prismatica founders (likely overlapping with PRSM founders) don't trigger self-dealing violations?
3. **Nonprofit purpose alignment.** What language in the 501(c)(3) filing supports the Prismatica structure as serving charitable purpose?
4. **Foreign structuring.** If the foundation is offshore (Cayman / Switzerland / Singapore per §9.3), how does the dividend arrangement interact with tax treaties?
5. **SEC characterization of dividends.** If foundation distributes FTNS as part of activities, does this constitute a token issuance?
6. **Prismatica legal form.** C-corp / B-corp / PBC / LLC — which optimizes for this arrangement?

**Estimated engagement:** 3-6 months, $100K-$300K legal spend depending on offshore structuring complexity.

**Status:** not started. Blocked on Gate 2.

**Exit criterion:** counsel delivers opinion letter specifying a legal structure, with estimated compliance cost and ongoing reporting burden.

### Gate 4: Governance Ratification

**Owner:** PRSM governance (structure TBD — council / DAO / hybrid, per §8 open question #5).

**Inputs:** Legal opinion from Gate 3.

**Decisions:**
- Adopt the hybrid model? Yes / No / Modified.
- If adopted, ratify specific parameter changes:
  - Amend `PRSM_Tokenomics.md` §2 (network fee 200 → 50 bps).
  - Amend `PRSM_Tokenomics.md` §5.1 (payment distribution math: 20% burn / 0.4% treasury / 7.68% creator / 71.92% serving node, or similar).
  - Establish Prismatica equity ownership vehicle.

**Status:** not started.

**Exit criterion:** governance vote passes, on-chain contract upgrade scheduled, Vision/Tokenomics docs updated.

### Gate 5: Smart-Contract Parameter Update

**Owner:** engineering.

**Inputs:** Ratified governance decision from Gate 4.

**Scope:** Update `FTNSTokenSimple.sol` or the fee-routing contract to reflect the new network fee parameter. If the contract already has a settable fee parameter with admin access, this is a single governance transaction. If not, requires a contract upgrade via the UUPS proxy.

**Dependencies:**
- Must happen after a mainnet deploy (Phase 1.3 Task 8) but can happen any time post-launch via governance.
- Coordination with any live Phase 2+ contracts that consume the fee parameter.

**Status:** not started. Parametric implementation exists; no work required until Gate 4 completes.

## Timeline (Illustrative, Not Committed)

| Gate | Earliest start | Target completion |
|---|---|---|
| 1 | 2026-04-20 (post-bake-in) | 2026-05-15 |
| 2 | 2026-05-15 | 2026-07-01 |
| 3 | 2026-07-01 | 2027-01-01 |
| 4 | 2027-01-01 | 2027-02-15 |
| 5 | 2027-02-15 | 2027-03-01 |

If any gate stalls, subsequent gates slide. The 2% pure-tax baseline remains operative until Gate 5 completes.

## Fallback Plan

If any gate fails (founder misalignment, legal determination that the structure is infeasible, governance rejection), PRSM continues with the 2% pure-tax model from §2. No code changes required; Vision and Tokenomics docs remain unchanged. No shipped functionality depends on the hybrid ratification.

## Updates

- **2026-04-14:** Document created. No gate work started.
