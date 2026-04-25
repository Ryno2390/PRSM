# PRSM-TOK-1: FTNS Tokenomics Standard

**Document identifier:** PRSM-TOK-1
**Version:** 0.2 Draft (revised 2026-04-24)
**Status:** Consolidates prior tokenomics drafting into standards-track form. Parameter values below are the current recommendations, pending Foundation board ratification and counsel confirmation (§13).
**Date:** 2026-04-21 (revised 2026-04-24)
**Drafting authority:** PRSM founder, pending Foundation convocation

**Revision note (2026-04-24):** Per canonical Vision docs (`PRSM_Vision.md`, `Prismatica_Vision.md`, `PRSM_Tokenomics.md` in founder materials), PRSM-CIS-1 (confidential-silicon standard) is reclassified as deferred research exploration rather than an active Foundation standard. PRSM-ECON-WP-1 (economic-model white paper) is a CIS-derivative analysis priced around chip unit economics; it has been archived alongside the CIS-1 source docs. Tokenomics sections that referenced CIS-1 as a companion standard or relied on the CIS-priced ECON-WP framing have been updated inline. The **core tokenomics primitives (halving, burn, staking, POL, compensation-only distribution)** are independent of CIS-1 and remain unchanged.

**Companions:**
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — governance charter this standard operates under.

**Related standards (added 2026-04-22):**
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — **PRSM-SUPPLY-1**, supply-side diversity standard. §6.2 introduces a Foundation-reserve-funded diversity bonus paid in FTNS (≤10% of qualifying listing's gross earnings). Affects reserve-to-operator FTNS flow when §5.2 intervention triggers fire; enters monitoring-only mode during a 5,000-provider-or-36-month bootstrap grace period. TOK-1 §4.6 (Foundation reserve management) should be read alongside SUPPLY-1 §7.3 (diversity-bonus budget line item).
- `docs/2026-04-23-prsm-policy-jurisdiction-1.md` — **PRSM-POLICY-JURISDICTION-1**, Foundation boundary subsidiary policy codifying anti-operational commitments.

**Archived dependencies (see `docs/archive/research/` for full rationale):**
- `archive/research/2026-04-21-prsm-economic-model-cis-silicon.md` — **PRSM-ECON-WP-1**, the CIS-priced economic model. Archived 2026-04-24 alongside the CIS-1 source docs because its bottom-up methodology is chip-unit-economics-driven. A successor economic model priced around Prismatica's six-stream core business (see `Prismatica_Vision.md` §2-3: commons curation, T3/T4 compute, FTNS treasury, protocol-native VC fund, commissioned datasets, domain models) remains to be drafted.

**Source material consolidated by this document:**
- `PRSM_Tokenomics.md` (external vault, 701 lines) — detailed design + mathematical models.
- `docs/2026-04-16-halving-schedule-implementation-plan.md` — halving operational + migration plan.
- Hybrid-model legal-ratification track (relocated to private repo).

This document does NOT replace the source material; it formalizes the subset of parameters and commitments that constitute the ratifiable standard. The source material remains the authoritative reference for scenario math, sensitivity analyses, and worked examples.

**This document is not legal or investment advice.** Counsel consultation required before any allocation is finalized.

---

## 1. Preface

### 1.1 Purpose

PRSM-TOK-1 specifies the normative parameters governing the FTNS token: supply schedule, distribution method, fee flows, staking mechanics, and the structural commitments that keep FTNS legally defensible as a utility/compensation token rather than an investment contract.

The standard is designed to achieve four goals simultaneously:

1. **Reward early contributors.** Creators, operators, and contributors who join before product-market-fit receive compensation in a token whose future value depends on the network's success.
2. **Preserve orderly secondary markets** through protocol-level mechanisms (burn, halving, staking) rather than Foundation market-making.
3. **Converge to utility-driven stability** — FTNS/USD tracks actual network usage at steady state, not pure speculation.
4. **Remain regulatorily defensible** under the Howey framework and equivalent international standards. FTNS is distributed as compensation for services rendered; Foundation does not sell FTNS to investors.

### 1.2 Normative language

Per RFC 2119. Parameters marked **[RATIFY]** require Foundation board approval before mainnet; parameters marked **[COUNSEL]** require legal opinion before adoption.

### 1.3 Scope

This document specifies:

- (§3) Core token parameters.
- (§4) The compensation-only distribution principle and its implementation.
- (§5) Bootstrap capital architecture (Prismatica equity, not FTNS sale).
- (§6) Emission schedule via Bitcoin-style halving.
- (§7) Genesis allocation percentages **[RATIFY]**.
- (§8) Fee flows: network fee, burn, royalty splits.
- (§9) Staking mechanics.
- (§10) Protocol-Owned Liquidity (POL) reserve.
- (§11) Regulatory positioning and Howey analysis.
- (§12) Implementation staging: operational → on-chain.
- (§13) Open decisions blocking ratification.

Out of scope:
- Specific secondary-market venues or listing strategies.
- Jurisdiction-specific tax treatment (covered by PRSM-GOV-1 §5 jurisdiction selection).
- Contract-level implementation details (covered by PRSM-CONTRACT-n standards, pending).
- Investor-specific economics (Prismatica's equity terms, covered by Prismatica's Series A materials).

---

## 2. Normative references

- **FTNSTokenSimple.sol** — ERC-20 + UUPS proxy token contract. Deployed to Base Sepolia testnet (7-day bake-in passed); mainnet deploy hardware-gated pending Foundation multi-sig quorum + external audit engagement. Mainnet contract address will be assigned at deploy ceremony.
- **PRSM-GOV-1** — Foundation Governance Charter.
- **PRSM-CIS-1** — Confidential Inference Silicon Standard *(currently deferred research per 2026-04-24 revision; see `archive/research/README.md`)*.
- **Howey v. SEC (1946)** — foundational US securities-classification test.
- **SEC v. Telegram Group Inc. (2020)** — relevant precedent on pre-launch token sale treatment.
- **Regulation D, Rule 506(c)** — accredited-investor private-offering exemption relied upon by Prismatica's equity raise.
- **FinCEN Guidance FIN-2019-G001** — virtual-currency MSB treatment.

---

## 3. Core parameters

| Parameter | Value | Status |
|---|---|---|
| Token symbol | `FTNS` (Fungible Token for Node Support; "Photons") | Fixed |
| Blockchain | Base (Ethereum L2), chain ID 8453 | Fixed (mainnet target; currently deployed to Base Sepolia testnet) |
| Contract address | **TBD at mainnet-deploy ceremony** (Base Sepolia: see `docs/2026-04-11-phase1.3-sepolia-bakein-log.md` for current testnet address) | Pending |
| Standard | ERC-20 | Fixed |
| Decimals | 18 | Fixed |
| **Initial supply** | **100,000,000 FTNS** | Fixed (minted to Foundation treasury at genesis) |
| **Maximum supply** | **1,000,000,000 FTNS** | Fixed (hard cap enforced at contract) |
| Distribution method | Compensation-only | Normative (§4) |
| Bootstrap capital source | Prismatica equity (Reg D 506(c)) | Normative (§5) |
| Network fee baseline | 200 bps (2%) | **[RATIFY]** — hybrid proposal may reduce to 50 bps, see §8.5 |
| Burn rate | 2,000 bps (20%) of each payment | Normative (§8) |
| Max royalty rate | 9,800 bps (98%) per content, set by creator | Fixed (contract level) |
| Staking lock options | 30 / 90 / 365 days | Normative (§9) |
| Staking yield multipliers | 1× / 1.5× / 3× | Normative (§9) |
| Halving epoch duration | 4 years | Normative (§6) |
| Halving scope | Foundation-paid compensation rates only | Normative (§6.2) |
| Asymptotic emission limit | 1,000,000,000 FTNS | Fixed (hard cap; halving ensures asymptotic convergence) |

---

## 4. Distribution principle: compensation-only

### 4.1 The commitment

FTNS is distributed **exclusively** as compensation for services rendered to the PRSM network. The Foundation operates no retail sale, no bonding curve, no ICO, no public-market-making. Every FTNS in circulation corresponds to work performed on or for the network.

This is a load-bearing commitment. PRSM-GOV-1 §13.2 designates any amendment that would permit Foundation token sales a **prohibited amendment** requiring Foundation dissolution and reformation — the same protection applied to the Foundation-does-not-build-chips commitment.

### 4.2 Permitted distribution channels

FTNS enters circulation only through:

1. **Creator royalties** — paid when content is accessed per the on-chain RoyaltyDistributor.
2. **Node operator compensation** — paid for compute, storage, bandwidth, shard-execution services. Includes the Phase 3 marketplace escrow-release path.
3. **Contributor grants** — paid by the Foundation to engineers, researchers, maintainers, and strategic contributors performing protocol development work. Subject to standard vesting (4-year with 1-year cliff).
4. **Foundation operational distributions** — paid to Prismatica, ecosystem partners, and strategic contributors as documented compensation for specific services rendered.

### 4.3 What this excludes

The Foundation MUST NOT:

- Operate any window that exchanges USD/USDC/ETH for FTNS controlled by the Foundation.
- Operate a bonding curve where FTNS price is a function of Foundation-side variables.
- Seed AMM pools using Foundation treasury FTNS.
- Conduct any "public sale" under any branding.
- Allow the POL reserve (§10) to be used for FTNS issuance (POL is defensive buying, not selling).

### 4.4 Why this structure defeats Howey

The compensation-only distribution produces a clean Howey analysis for recipients:

| Howey prong | Compensation-received FTNS | Traditional ICO / bonding curve |
|---|---|---|
| Investment of money | Absent (tokens received as compensation) | Present |
| Common enterprise | Arguable | Present |
| Expectation of profit | Absent (compensation for current services) | Present |
| From efforts of others | Absent (recipient's own efforts produced the compensation) | Present |

Prior drafts of PRSM tokenomics included a bonding-curve-based investor sale that satisfied all four Howey prongs; that design was superseded by the equity-investment architecture (§5) following the 2026-04-15 pivot. **Prismatica raises equity from accredited investors under Reg D 506(c); the Foundation never sells FTNS.**

---

## 5. Bootstrap capital architecture

### 5.1 The equity-investment model

PRSM's bootstrap capital requirement (design, engineering, infrastructure, initial meganode deployment, certification testing costs, research grants, founder-team compensation) is raised through **Prismatica equity**, not FTNS distribution. Specifically:

1. Prismatica is incorporated as a Delaware C-corporation (PRSM-GOV-1 §6.3 recommendation).
2. Prismatica sells equity via standard Reg D Rule 506(c) to accredited investors.
3. Capital is deployed to build Prismatica's commercial business on PRSM — six core streams + three growth adjacencies per `Prismatica_Vision.md` §2-3: T3/T4 compute operations, commons-data curation, commissioned-dataset origination, domain-specific foundation models, protocol-native VC fund seed capacity, FTNS treasury accumulation, managed enterprise inference, scholarly publishing, data clean-rooms.
4. Prismatica accumulates FTNS organically through its own network participation — earning compensation by operating meganodes, by building and licensing PRSM-compatible services, and by receiving documented operational FTNS distributions from the Foundation when those are for specific services rendered.
5. Prismatica investor returns flow from Prismatica's enterprise value growth, driven by operating revenue + FTNS treasury appreciation + other commercial activities.

### 5.2 Why this architecture is the correct choice

**Regulatory:**
- Prismatica equity is cleanly classified as a security under existing frameworks. Investors buy a known instrument with known protections and known tax treatment.
- FTNS receives compensation-token treatment (§4.4). The two instruments have cleanly separated regulatory status.

**Incentive alignment:**
- Prismatica investors want Prismatica's commercial execution to succeed — which requires PRSM to succeed.
- Prismatica's FTNS treasury grows through earning (commerce + provided services), which aligns Prismatica's economic interest with network success rather than with secondary-market speculation.

**Governance clarity:**
- Foundation is not dependent on Prismatica investors for capital. Foundation funding comes from its FTNS treasury allocation + protocol fees + certification fees + donations (PRSM-GOV-1 §4.6).
- No individual Prismatica investor holds leverage over Foundation decisions.

### 5.3 Prismatica's FTNS holdings policy

Prismatica's FTNS treasury is disclosed in its investor reports. Prismatica's on-chain voting behavior (if any votes are cast on FTNS-holder matters) is subject to public summary per PRSM-GOV-1 §6.2 item 4.

Prismatica MUST NOT accumulate FTNS specifically to influence a Foundation-unfavorable vote. The enforcement is structural (PRSM-GOV-1 §4.4 caps single-implementer Foundation board seats at 1; §8 forces the founder to pick one entity after bootstrap) rather than contractual — token-contract-level vote-buying restrictions are unenforceable.

### 5.4 What the prior bonding-curve design would have produced

Historically, PRSM tokenomics included a bonding-curve + AMM-transition design targeting ~$20M treasury accumulation through month 18 with 18-100× investor returns. That design was:

- Unambiguously a securities offering under Howey.
- Dependent on retail appreciation expectation that creates price instability.
- Legally vulnerable to retroactive SEC action (Telegram TON precedent).
- Inconsistent with long-term protocol legitimacy.

The replacement (Prismatica equity raise of comparable dollar magnitude) provides the same bootstrap capital with cleaner regulatory treatment. The returns that would have flowed to bonding-curve investors now flow to Prismatica equity holders who hold a properly-classified security and exit via standard equity mechanisms (secondaries, IPO, Prismatica-funded buybacks).

---

## 6. Emission schedule

### 6.1 Bitcoin-style halving

Foundation-paid compensation rates halve every 4 years on a published schedule:

| Epoch | Years post-mainnet | Rate | Approx emission budget (of 100M genesis) |
|---|---|---|---|
| 1 | 0–4 | Baseline (R) | Up to ~40M FTNS distributed |
| 2 | 4–8 | R/2 | Up to ~20M FTNS |
| 3 | 8–12 | R/4 | Up to ~10M FTNS |
| 4 | 12–16 | R/8 | Up to ~5M FTNS |
| 5 | 16–20 | R/16 | Up to ~2.5M FTNS |
| 10 | 36–40 | R/512 | Negligible |
| Asymptotic | ∞ | → 0 | 1B hard cap never reached |

Rate halving applies to foundation-paid compensation streams at each epoch transition (one administrative action per epoch). **The 1B FTNS hard cap is never reached — emissions asymptotically converge to zero.**

### 6.2 Scope: what halves and what doesn't

Halving applies to:
- Foundation-paid creator royalty bonuses (layered on top of user-paid royalties).
- Foundation-paid node operator compensation bonuses (layered on top of user-paid job fees).
- Foundation contributor grants (denominated in USD but paid in FTNS at grant time).

Halving does NOT apply to:
- User-to-user payments for services (a user paying a creator 1 FTNS for content access is a transaction, not an emission).
- On-chain royalty-split percentages (a creator's 8% share remains 8%; what halves is the Foundation-issued bonus).
- Protocol fees and burn mechanics.
- Staking yield distribution.

### 6.3 Governance and enforcement

Per `docs/2026-04-16-halving-schedule-implementation-plan.md`, the halving is implemented in two stages:

**Stage 1 (current, Epochs 1-2, years 0-8 post-mainnet):** Operational policy.
- Foundation board ratifies Epoch 1 baseline rates pre-mainnet.
- Rate halvings applied at each epoch transition via Foundation treasury operations.
- Modifications require 90-day advance notice + 30-day stakeholder comment period.
- **Monotone-decreasing invariant:** rates can halve or pause but cannot increase. No exception permitted.
- Public logging on every transition (on-chain event or Foundation-signed IPFS attestation).

**Stage 2 (Phase 8 target, Q4 2028):** On-chain enforcement.
- New `EmissionController.sol` enforces per-epoch rate limits.
- Epoch-rate calculation publicly readable and deterministic.
- Foundation operational discretion over rates eliminated (aside from narrow pause provisions, §6.4).

### 6.4 Pause provision

Governance may pause the halving schedule (rates remain at current epoch level rather than halving further) in specific network-health-emergency scenarios. Pause requires:
- Unanimous Foundation board vote.
- Published rationale.
- Pause does NOT permit rate increase. Monotone-decreasing invariant is preserved.
- Pause duration is bounded — a pause not reaffirmed at 6 months automatically lapses and the next halving is executed.

### 6.5 Epoch 1 baseline rate calibration **[RATIFY before mainnet]**

Foundation board must ratify Epoch 1 baseline rates pre-mainnet. Calibration methodology per `docs/2026-04-16-halving-schedule-implementation-plan.md` §2.4:

1. Forward-simulation scenarios (bull / base / bear adoption) with total-FTNS-distributed, network-participation-level, and incentive-strength metrics.
2. Acceptance criteria:
   - Foundation allocation depletion through Epoch 1 (<40% under base case).
   - Incentive strength >2× operator opportunity cost under base case.
   - Epoch 1 distribution under bear case fits within 60% of allocation.
3. Ratification in a numbered Foundation governance document 90+ days before mainnet.

**Specific rate values are not set in this document.** They require simulation with modeled adoption data not yet available.

---

## 7. Genesis allocation **[RATIFY]**

All 100M initial supply mints to Foundation treasury at genesis. The Foundation distributes over time as compensation per §4.2. The proposed allocation categories:

| Category | Share | Vesting | Notes |
|---|---|---|---|
| Creator royalty pool | **30–40%** | As content is accessed, per RoyaltyDistributor | Funds the creator-bonus layer during bootstrap |
| Node operator pool | **25–35%** | As services rendered | Funds the operator-bonus layer during bootstrap |
| Contributor grants | **15–20%** | 4-year vesting, 1-year cliff | Engineers, researchers, maintainers |
| Foundation operational reserve | **10–15%** | Retained at Foundation discretion | Ecosystem grants, POL defense (§10), strategic distributions |
| Initial Prismatica allocation | **5–10%** | 4-year vesting | Granted to Prismatica as first-mover commercial-operator compensation (T4 meganode baseline capacity, commons-data curation, and anchor operational role during the bootstrap period) |

Constraints:
- Total MUST equal 100% of the 100M genesis supply.
- Contributor and Prismatica vesting schedules MUST be disclosed publicly in Foundation's annual report (PRSM-GOV-1 §11.1).
- Allocation percentages above are **recommended ranges**; Foundation board ratifies specific values pre-mainnet. Any ratification outside these ranges requires supermajority AND published rationale explaining the deviation.

### 7.1 Founder's personal allocation

The founder's personal FTNS holdings (whether from Prismatica's 5-10% allocation, from Foundation contributor grants for pre-mainnet work, or from Prismatica's on-market compensation) are:
- Subject to standard 4-year vesting with 1-year cliff.
- Publicly disclosed per PRSM-GOV-1 §12.2.
- Any secondary-market sales disclosed with 24-hour lag for market-integrity reasons (PRSM-GOV-1 §7.2).

### 7.2 Foundation's 20% long-run cap

Per PRSM-GOV-1 §4.3 item 4, the Foundation's FTNS treasury MUST NOT exceed 20% of circulating supply after Year 5. Distributions during Years 1-5 may push the Foundation's held share higher temporarily (if circulating supply is low relative to Foundation treasury); the 20% cap is a long-run commitment that shapes how quickly the Foundation deploys its genesis allocation.

---

## 8. Fee flows and burn

### 8.1 Payment distribution (from Phase 3 onward)

Every FTNS payment on the network is split as follows:

```
  burn:          20.0%  (permanently destroyed)
  treasury:       1.6%  (2% network fee × 80% of remainder)
  creator:        6.4%  (8% royalty × 80% of remainder)
  serving node: 72.0%  (90% × 80% of remainder)
                ─────
  total:        100.0%
```

The 20% burn is taken off the top; the remaining 80% is split per the royalty/treasury/operator ratios established in Phase 1.1.

### 8.2 Burn-on-use rationale

The 20% burn creates continuous deflationary pressure proportional to utility. At the $100M/year volume scenario with $2/FTNS price:
- Annual burn: `$100M × 20% / $2 = 10M FTNS/year`
- At 50M circulating supply: **20% annual supply reduction**

Crossover point (where burn exceeds emission) depends on halving epoch and transaction volume. At Epoch 1 low-volume conditions, emissions typically exceed burn. By Epoch 3 with compounded adoption and halved emissions, burn typically dominates.

### 8.3 Network fee destination

The 2% network fee (on the 80% non-burned portion, yielding 1.6% of total) flows to Foundation treasury. This funds:
- Foundation operations (staff, audits, certifications overhead not covered by cost-recovery certification fees).
- Research grants.
- POL reserve replenishment.
- Ecosystem partnerships.

Foundation fee revenue is disclosed in the annual report (PRSM-GOV-1 §11.1 item 3).

### 8.4 Creator royalty mechanics

User-set royalty rates 0–98% determine the creator's share of each access payment (before the 20% burn / 80% distribution above). A creator setting 8% royalty receives 8% of the 80% post-burn — i.e., 6.4% of the gross payment.

### 8.5 Hybrid fee model — **[COUNSEL REQUIRED]**

Per the hybrid-model legal-ratification track (private repo), an alternative fee structure is under legal review:

- Reduce baseline network fee from 2% to **50 bps (0.5%)**.
- Foundation takes a **15-25% equity stake in Prismatica** in exchange.
- Foundation revenue mix shifts from pure protocol-fee-extraction toward dividend income from Prismatica's commercial returns.

Hybrid advantages:
- Lower user-facing fees improve network competitiveness.
- Foundation revenue is less cyclical (Prismatica dividends smooth volume-driven protocol fees).
- Reads as "nearly-free protocol" for user acquisition narratives.

Hybrid risks:
- UBIT (Unrelated Business Income Tax) exposure on dividend flows.
- Private-inurement risk if Prismatica founders overlap with Foundation board members (mitigated by PRSM-GOV-1 §8 recusal + end-of-bootstrap choice).
- Private-benefit analysis under non-profit doctrine.

Four open gates before hybrid adoption:
1. Internal alignment on specific stake % and fee floor.
2. Prismatica founder alignment — term sheet ratified.
3. External counsel opinion (3-6 month engagement, $100K-$300K legal spend).
4. Foundation governance ratification.

**Until the hybrid model is ratified, this document specifies 2% baseline per §3.** Any migration to the hybrid structure requires PRSM-TOK-1 amendment per §13 amendment process.

---

## 9. Staking

### 9.1 Lock tiers and yield multipliers

| Lock period | Yield multiplier | Service discount | Priority access |
|---|---|---|---|
| None | 0× (not staked) | 0% | Standard |
| 30 days | 1× (baseline) | 2% | +10% |
| 90 days | 1.5× | 5% | +25% |
| 365 days | 3× | 10% | +50% |

### 9.2 Yield source

Staking rewards flow from the 2% network fee pool (or 50 bps under hybrid, §8.5), distributed pro-rata to stakers weighted by `(multiplier × locked amount)`.

### 9.3 Yield calculation example

Assumptions:
- Total staked: 15M FTNS (30% of 50M circulating)
  - 5M at 30-day (1×): weight 5M
  - 7M at 90-day (1.5×): weight 10.5M
  - 3M at 365-day (3×): weight 9M
  - Total weighted: 24.5M
- Annual network fee pool: `$100M volume × 2% = $2M`

Per-weighted-FTNS yield: `$2M / 24.5M = $0.0816 per weighted FTNS per year`

At $2/FTNS price:
- 30-day staker APY: ~4.1%
- 90-day staker APY: ~6.1%
- 365-day staker APY: ~12.2%

Yields scale with network volume. At low volume, yield is small; **service discounts (up to 10%) and priority access may be more valuable than yield for high-volume users.**

### 9.4 Staking is optional

No FTNS holder is required to stake. Staking is a user-optional mechanism that trades short-term liquidity for yield + service-access benefits. This matters for regulatory classification — a token that pays yield automatically to all holders carries different analysis than one where yield is a function of user-elected lock.

### 9.5 Staking implementation

Staking contract is deferred to **Phase 7+** (see `docs/2026-04-12-phase2-remote-compute-plan.md` for phasing). Pre-Phase-7, staking-tier signals in the Phase 3 marketplace (§3.1 of PRSM marketplace design) are self-reported and advisory only; the `stake_tier` field in ProviderListing carries no economic enforcement until the Phase 7 staking contract ships.

---

## 10. Protocol-Owned Liquidity (POL) reserve

### 10.1 Purpose

The POL reserve is a Foundation-held USDC position (funded by accumulated network fee revenue) that can be deployed defensively during severe market-price stress. It is **not** a price peg, guarantee, or commitment.

### 10.2 Operational rules

- **Trigger:** FTNS price drops >40% over 7 days across secondary venues.
- **Deployment cap per event:** ≤10% of current POL reserve.
- **Action:** Foundation may purchase FTNS from peer-to-peer markets at prevailing price, retire or hold as treasury.
- **Reserve size target:** 20-30% of Foundation's total liquid treasury at any time.
- **Transparency:** all POL operations disclosed on-chain post-execution (amounts, timing, venues).
- **Discretionary:** Foundation MAY choose not to intervene if intervention would be ineffective or counterproductive (e.g., during coordinated manipulation, during structural bear markets not caused by PRSM-specific events).

### 10.3 What POL explicitly is NOT

- Not a price floor.
- Not a retail-facing liquidity commitment.
- Not a mechanism for FTNS issuance (POL only purchases, never sells Foundation-held FTNS in a way that increases circulating supply — Foundation-to-market sales are prohibited per §4).
- Not an algorithmic peg (each intervention is governance-ratified).

---

## 11. Regulatory positioning

### 11.1 US Howey analysis

Per §4.4, the compensation-only distribution produces a clean Howey failure on three of four prongs for FTNS recipients. Prismatica equity carries standard securities treatment; FTNS itself is designed to mirror Bitcoin's regulatory posture as a utility/compensation token.

### 11.2 FinCEN considerations

The Foundation's operational role (distributing compensation; accepting payments for network services) does not constitute a Money Services Business under FIN-2019-G001 because:
- Foundation does not convert FTNS to fiat on behalf of users.
- Foundation does not operate as an exchange.
- Foundation's POL operations (§10) are treasury defensive-purchases, not exchange services.

### 11.3 International posture

Specific jurisdictional analysis is pending counsel engagement. Target jurisdictions and their relevant frameworks:
- **EU**: MiCA (Markets in Crypto-Assets) applies from 2024. Utility-token classification likely workable; counsel opinion required.
- **Switzerland**: FINMA guidance recognizes utility tokens clearly; Swiss Foundation form (PRSM-GOV-1 §5.1 Candidate A) aligns with FINMA expectations.
- **Singapore**: MAS guidance on payment tokens vs utility tokens; likely workable.
- **Cayman**: supports token-issuing foundations straightforwardly.

### 11.4 What the Foundation MUST NOT say in marketing

To preserve regulatory posture, Foundation and Foundation-employed persons MUST NOT:
- Promise, suggest, or imply FTNS price appreciation.
- Describe FTNS acquisition as "investing."
- Quote projected FTNS returns.
- Compare FTNS to any publicly-traded security.
- Describe the Foundation's activities as managing a "fund."

Prismatica's marketing of its own equity is separate — Prismatica is selling securities under Reg D 506(c) and the standard securities-marketing rules apply there. PRSM-CIS-1 and PRSM-GOV-1 content is technical/governance material, not FTNS marketing, and is not subject to these restrictions.

### 11.5 Counsel consultation dependencies

The following items REQUIRE counsel opinion before finalization (flagged **[COUNSEL]**):

1. §8.5 hybrid fee model — four legal gates per the hybrid-model legal-ratification track (private repo).
2. §11.3 international positioning in each target jurisdiction.
3. Genesis allocation §7 specifics under the chosen Foundation jurisdiction's non-profit rules.
4. Prismatica's initial allocation mechanics — whether it constitutes a compensation grant, a sale, or something requiring Reg D filing.
5. Founder's personal allocation mechanics — interaction with Foundation employment (if any) and Prismatica equity vesting.

---

## 12. Implementation staging

### 12.1 Stage mapping

| Stage | Period | What ships | Enforcement |
|-------|--------|-----------|-------------|
| Genesis | Mainnet launch (target post-hardware, ~2026 Q3) | 100M FTNS minted to Foundation treasury | Contract-level |
| Bootstrap | Year 0-3 | Compensation distribution begins; halving operational policy | Foundation policy + public logs |
| Phase 3 | Year 0-1 | Marketplace dispatch + Phase-3-rate burn on every payment | On-chain contract |
| Phase 7 | Year 1-3 | Staking contract + stake-tier enforcement | On-chain contract |
| Phase 8 | Year 2-3 | On-chain halving enforcement (`EmissionController.sol`) | Strict on-chain |
| Steady state | Year 5+ | All stages active; Foundation treasury long-run cap (20%) enforced by policy + disclosure | Hybrid |

### 12.2 What ships with what

**At mainnet (genesis):**
- ERC-20 token contract (already deployed).
- Foundation treasury multi-sig holds 100M FTNS (post-Phase 1.3 hardware mainnet deploy).
- Epoch 1 baseline rates ratified and published.
- POL reserve initial funding: $0 (grows from network fee revenue).

**Phase 3 (Q3 2026 target, post-hardware):**
- 20% burn on every payment.
- Network fee flows to Foundation treasury.
- Marketplace distributions into full payment-split pattern.

**Phase 7:**
- Staking contract with 30/90/365-day tiers.
- Marketplace stake-tier enforcement (replaces Phase 3 self-reported advisory).
- Staking yield pool funded from network fees.

**Phase 8:**
- `EmissionController.sol` deployed.
- Halving schedule moves from operational policy to strict on-chain enforcement.
- Foundation operational discretion on rates narrowed to pause-only (§6.4).

### 12.3 Independence from deferred research standards

FTNS tokenomics do not depend on any single protocol-standard arc (CIS-1, FHE, MPC, etc.). The core mechanisms — halving emissions, burn-on-use, staking locks, Protocol-Owned Liquidity reserve, compensation-only distribution — operate against any compute tier Prismatica or third-party operators provision on PRSM (T1 consumer edge, T2 prosumer, T3 cloud arbitrage, T4 meganodes). Revenue-mix composition shifts as the network matures and research tracks ship or are deferred, but the core tokenomics primitives remain identical.

*(Previous §12.3 "Interaction with PRSM-CIS-1" was removed 2026-04-24 alongside the CIS-1 reclassification as deferred research. If CIS-1 is revived under the §9 ratification path, a new §12.x covering that standard's FTNS-denomination semantics can be added at that time.)*

### 12.4 Interaction with PRSM-GOV-1

Foundation tokenomics decisions (allocation ratification, halving rate pauses, POL deployment) follow PRSM-GOV-1 §4.2 powers and §10 execution requirements. Specifically:
- All FTNS-treasury transactions above $250K equivalent require full board approval + 48-hour notice (PRSM-GOV-1 §4.2 item 5).
- Any parameter amendment in PRSM-TOK-1 requires PRSM-GOV-1 §13 amendment procedure.
- Founder FTNS holdings disclosed publicly per PRSM-GOV-1 §12.2.

---

## 13. Open decisions

Items requiring resolution before v1.0 ratification, in priority order:

### 13.1 Genesis allocation specifics **[HIGHEST PRIORITY, RATIFY pre-mainnet]**

§7 proposes ranges, not specific percentages. Foundation board must ratify:
- Creator royalty pool: specific % in the 30-40% range.
- Node operator pool: specific % in the 25-35% range.
- Contributor grants: specific % in the 15-20% range.
- Foundation operational reserve: specific % in the 10-15% range.
- Prismatica allocation: specific % in the 5-10% range.

Input data: PRSM_Tokenomics.md §6 scenario simulations (bull / base / bear) producing emission budget per epoch.

**Action:** Foundation board ratifies in a numbered governance document at least 90 days pre-mainnet, per PRSM-GOV-1 §9 ratification process.

### 13.2 Epoch 1 baseline rates **[RATIFY pre-mainnet]**

§6.5 specifies methodology but not values. Foundation board must publish:
- Creator royalty bonus rate (FTNS paid per content access event, on top of user-paid royalty).
- Node operator bonus rate (FTNS paid per PCU served, on top of user-paid job fee).
- Contributor grant rate reference table (FTNS paid per work-item category).

### 13.3 Hybrid fee model — adopt or retain 2% baseline **[COUNSEL + RATIFY]**

§8.5 + hybrid-model legal-ratification track (private repo). Four gates:
1. Internal alignment on 15-25% stake + 50 bps fee floor.
2. Prismatica founder term sheet.
3. Counsel opinion (3-6 months, $100K-$300K).
4. Foundation governance ratification.

Timeline: hybrid adoption, if pursued, ratifies ~Year 1 post-mainnet. This PRSM-TOK-1 v0.1 specifies the 2% baseline and treats hybrid as an amendment path.

### 13.4 Foundation jurisdiction dependency

PRSM-GOV-1 §5.1 jurisdiction choice affects PRSM-TOK-1 §7.2 (long-run treasury cap tax treatment), §10 (POL operations tax treatment), and §11.3 (international positioning). Until jurisdiction is chosen (PRSM-GOV-1 §14.1 highest-priority item), PRSM-TOK-1 parameters are provisional.

### 13.5 Prismatica allocation mechanics **[COUNSEL]**

§7 Prismatica allocation = 5-10%. Mechanics open:
- Is this a "grant for services rendered" (clean under §4 compensation principle)?
- Is this a "purchase at zero" (potentially Reg D territory)?
- Is this a founder-compensation-shared arrangement (private inurement risk per PRSM-GOV-1 §8)?

The cleanest structure: Prismatica's 5-10% vests over 4 years tied to specific services rendered (first-implementer CIS design, first T4 meganode operation, published research contributions). This aligns with §4 compensation principle. Counsel must confirm.

### 13.6 Staking contract specifics — **[DEFER to Phase 7 plan document]**

§9 specifies tiers, multipliers, yield source. Remaining contract-level questions (early-unlock penalty schedules, yield-compounding mechanics, slashing for any staking-related misbehavior) belong in a Phase 7 plan document (not yet authored).

### 13.7 FTNS minting contract beyond genesis **[DEFER]**

§3 specifies 100M genesis + 1B hard cap. The mechanism by which the next ~900M enters circulation over decades is:
- Protocol-level mining-like rewards for work performed (not investor sales).
- Enforced by `EmissionController.sol` (Phase 8 deliverable).
- Subject to the halving schedule's asymptotic convergence.

Contract-level implementation details for the minting mechanism beyond the existing FTNSTokenSimple.sol are deferred to the Phase 8 plan document.

### 13.8 Cross-jurisdictional tax on FTNS holdings

If Foundation is offshore (Swiss / Cayman) and contributor-recipients are US persons, the FTNS distributions trigger US-side tax for recipients at fair-market-value on date of receipt. This is standard for any compensation in crypto. Foundation MUST NOT represent itself as providing tax advice but SHOULD provide recipients with documentation sufficient for their own tax filings (value of FTNS at grant time, grant date, service category). **[COUNSEL]** for specifics.

---

## 14. Acknowledgements

This standard consolidates design work recorded in:
- `PRSM_Tokenomics.md` (external vault, authoritative mathematical models).
- `docs/2026-04-16-halving-schedule-implementation-plan.md` (halving operational + migration).
- Hybrid-model legal track (private repo).

Design inheritances:
- **Bitcoin** — halving cadence, compensation-based distribution, non-security-preserving structure.
- **Ethereum** — Foundation governance model (Ethereum Foundation / Zug pattern).
- **Helium, Filecoin, Arweave** — utility-token network designs; relevant as cautionary and positive examples across different regulatory outcomes.

Where this standard diverges from predecessors, it's typically toward stricter compensation-framing (§4) and more aggressive burn (20% vs network precedents closer to 5-10%). The 20% burn is calibrated for PRSM's expected adoption curve; see PRSM_Tokenomics.md §5.2 for the supporting scenario math.

---

## 15. Change log

**v0.1 (2026-04-21):** Initial consolidation of PRSM_Tokenomics.md + halving implementation plan + hybrid legal tracking into standards-track form. Parameter ranges specified; specific values deferred to Foundation board ratification (§13). Counsel consultation required before any allocation is finalized (§11.5).

---

**End of PRSM-TOK-1 v0.1 Draft.**
