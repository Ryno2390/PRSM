# PRSM Economic Model: CIS Silicon + FTNS Revenue Recovery

**Document identifier:** PRSM-ECON-WP-1
**Version:** 0.1 Draft
**Status:** Companion analysis to PRSM-CIS-1 §16.10 and PRSM-TOK-1. First-principles economic model for Prismatica equity investors. All projections are illustrative and subject to the assumptions stated inline. **Not investment advice.**
**Date:** 2026-04-21
**Drafting authority:** PRSM founder, pending Foundation convocation
**Companions:**
- `docs/2026-04-21-prsm-cis-1-confidential-inference-silicon.md` — silicon standard this paper prices.
- `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` — tokenomics standard.
- `docs/2026-04-21-prsm-investor-executive-summary.md` — scannable investor-facing overview.

**This document is not legal or investment advice.** Projections are illustrative scenario analyses, not promised returns. Any investor relying on the math below MUST independently verify assumptions and should consult qualified financial and legal advisors.

---

## 1. Purpose

PRSM-CIS-1 specifies three silicon conformance levels (C1 / C2 / C3) with graduated security + cost. PRSM-TOK-1 specifies the FTNS compensation flows and Prismatica's equity-investment architecture. **This paper connects them:** showing how CIS chip costs translate to per-inference unit economics, network-level revenue, and ultimately Prismatica enterprise value — which is what Prismatica equity investors actually buy.

The investor's question is not "will FTNS appreciate?" — that is a secondary and deliberately-unstated effect. The investor's question is: **"What does Prismatica's P&L look like in Years 3, 5, and 10, and does the equity stake recover the capital at an acceptable IRR?"** This paper builds that answer from the bottom up.

### 1.1 What this paper does

- Estimates CIS chip design + fab + certification costs (§3).
- Estimates per-chip-year revenue under the Phase 3 marketplace price model (§4).
- Models meganode unit economics at realistic utilization (§5).
- Builds up to Prismatica's two revenue streams: commercial services + FTNS treasury (§6-7).
- Presents three scenarios (bull / base / bear) over 5- and 10-year horizons (§8).
- Identifies which assumptions matter most via sensitivity analysis (§9).
- Compares to alternative investment structures (§10).

### 1.2 What this paper does NOT do

- Promise specific investor returns.
- Constitute an offering of Prismatica securities (any offering will be via formal 506(c) materials).
- Replace professional financial or legal advice.
- Resolve the PRSM-CIS-1 §16.10 placeholder authoritatively — this is a first-principles estimate, not the final Foundation-published economic model.

### 1.3 Assumption transparency

Every material number in this paper carries an explicit justification inline, with one of three confidence tags:
- **[HIGH]** — derived from published pricing or first-party observation (e.g., Phase 3 marketplace acceptance test at 0.03 FTNS/shard).
- **[MEDIUM]** — industry-standard estimates from comparable hardware programs (TSMC 7nm wafer cost, typical certification budgets).
- **[LOW]** — forward projections about adoption, market share, FTNS price at future dates. These are the most sensitive assumptions and are tested in §9.

---

## 2. Market sizing (top-down)

### 2.1 Current AI inference spend

Public estimates for 2025-2026 annual inference spend at frontier labs **[MEDIUM]**:

| Entity | Est. annual inference cost |
|---|---|
| OpenAI (GPT-4/5-class) | $1.5B – $5B |
| Anthropic (Claude-family) | $500M – $2B |
| Google (Gemini, search AI integration) | $2B – $8B (mostly internal) |
| Meta (open-weights Llama, internal) | $500M – $2B |
| xAI (Grok) | $200M – $1B |
| Tier-2 labs (Mistral, Cohere, AI21, startups) | $500M – $2B aggregate |
| **Addressable market (excluding internal-captive)** | **$3B – $12B annual** |

Growth rate: inference costs approximately doubling every 12-18 months as model sizes + usage both scale, partially offset by cost-per-token deflation (~30-50%/year) **[MEDIUM]**. Net market growth: 40-80% CAGR through 2030.

### 2.2 PRSM-addressable segments

Not all inference is reachable. PRSM captures inference where the publisher wants to serve through a decentralized, cryptographically-verifiable substrate instead of (or in addition to) their own internal infrastructure. Segments:

| Segment | 2025 est. annual | 2030 est. annual | PRSM addressability |
|---|---|---|---|
| **Open-weights fine-tunes** (Llama 4, Mixtral, etc.) | $100M-$500M | $2B-$10B | High. Phase 3 ships this today. |
| **Enterprise-specific proprietary** (SaaS-wrapped LLMs) | $500M-$1.5B | $5B-$20B | Medium. Requires CIS-C1 minimum. |
| **Frontier proprietary lagged 12-18mo** | $1B-$3B | $10B-$30B | Low today, Medium by 2030. Requires CIS-C2 or strong Phase 2.1 attestation. |
| **Current-SOTA frontier proprietary** | $2B-$8B | $20B-$50B | Gated on CIS-C3 (2031+) |

PRSM's **immediate addressable market in 2026-2028**: the open-weights + enterprise segments, roughly $500M-$2B annually, growing fast.

PRSM's **frontier-acceptable market in 2030-2033**: the 12-18-month-lagged proprietary segment, roughly $10B-$30B annually if CIS-C2 ships on schedule.

### 2.3 PRSM's share capture assumptions

PRSM does not need majority share to justify the investment. Share capture assumptions (tested in scenarios):

| Segment | Bear case | Base case | Bull case |
|---|---|---|---|
| Open-weights + enterprise (2028) | 1% | 5% | 15% |
| Frontier lagged (2032) | 2% | 10% | 25% |
| Current SOTA (2034+) | 0% | 2% | 10% |

These percentages are **[LOW]** confidence — they reflect adoption trajectories that cannot be forecast precisely. Sensitivity analysis in §9 shows how Prismatica returns behave as these shift.

---

## 3. CIS chip unit economics (bottom-up)

### 3.1 NRE (non-recurring engineering) costs

Design + tape-out costs for an AI-accelerator-class chip at modern nodes, scaled for CIS security features **[MEDIUM]**:

| Conformance level | Target process | Design + verify | Tape-out mask set | Total NRE |
|---|---|---|---|---|
| **C1** | 7nm | $40M – $70M | $20M – $30M | **$60M – $100M** |
| **C2** | 5nm | $80M – $150M | $30M – $50M | **$110M – $200M** |
| **C3** | 5nm or 3nm | $150M – $300M | $40M – $100M | **$190M – $400M** |

CIS-specific uplift over a general AI accelerator: tamper-mesh routing + active-shield layers + formal verification of the attestation unit (required at C2+) + destructive-test-sample budget for certification. Estimated CIS uplift: **+20-40%** over a commodity AI accelerator at the same process node.

### 3.2 Per-chip production cost

Once NRE is amortized, per-chip production economics **[MEDIUM]**:

| Cost element | C1 (7nm) | C2 (5nm) | C3 (3nm) |
|---|---|---|---|
| Wafer (per full wafer) | $10K | $16K | $20K |
| Dies per wafer (at ~500mm²) | ~100 | ~100 | ~100 |
| Die yield (mature node) | 65% | 60% | 45% |
| Die cost | $155 | $270 | $440 |
| Packaging + test | $40 | $60 | $80 |
| CIS security test + provisioning | $50 | $100 | $200 |
| **Total per-chip cost** | **~$245** | **~$430** | **~$720** |

Meganode operator typically buys in 10K+ unit volumes; pricing includes ~40-60% implementer margin → street price **$400-$1200 per chip** depending on level.

### 3.3 Certification costs

Per-design (not per-chip) **[MEDIUM]** based on FIPS 140-3 L3/L4 program precedents + R8 academic red-team budget:

| Level | Lab testing | Red-team engagement | Committee review | **Total certification** |
|---|---|---|---|---|
| C1 | $300K – $800K | — | $100K | **$400K – $900K** |
| C2 | $1.5M – $3M | $500K – $1M | $300K | **$2.3M – $4.3M** |
| C3 | $5M – $10M | $2M – $4M | $500K | **$7.5M – $14.5M** |

Certification is paid to certified labs (independent of the Foundation) + Foundation committee oversight (cost-recovery only per PRSM-GOV-1 §15.2). Certification validity: 3 years (C1), 2 years (C2), 1 year with mandatory annual review (C3).

### 3.4 Total capital to first Prismatica chip

| Phase | C1 first chip | C2 first chip |
|---|---|---|
| Design + tape-out NRE | $80M (mid of range) | $150M |
| First production run (1000 chips) | $250K | $430K |
| Certification | $650K | $3M |
| Provisioning + supply-chain setup | $2M | $3M |
| **Total capital to shipping first chip** | **~$83M** | **~$156M** |

**[HIGH]** observation: this capital is the order-of-magnitude Prismatica's Series A + B must raise to reach first-chip delivery at C2, which is the frontier-acceptable tier where PRSM's large TAM lives. C1 is cheaper but commercially less differentiating vs H100 CC.

---

## 4. Per-chip annual revenue

### 4.1 Inference rate assumptions

A CIS chip at C2 is designed for sustained inference throughput comparable to a single H100 CC GPU **[MEDIUM]**:

| Metric | Assumption | Source |
|---|---|---|
| Sustained tokens/sec (70B model, batched) | 2,000 tok/s | H100 published benchmarks, scaled for CIS overhead (~85% vs non-CIS) |
| Utilization (time under load) | 60% | Typical cloud-GPU utilization at steady state |
| Effective tokens/year per chip | ~38B tokens | 2,000 × 86,400 × 365 × 0.6 |
| Effective inferences/year (1K-token avg response) | ~38M inferences | Depends on workload mix |
| Effective shards/year (4-shard tensor-parallel avg) | ~150M shards | For Phase 3 per-shard pricing |

### 4.2 Phase 3 price realized per chip

Phase 3 marketplace test prices shards at **0.03 FTNS per shard** at the cheap end, 0.10 FTNS at the expensive end **[HIGH]** (acceptance-test data).

At a realistic blended price of **0.05 FTNS per shard** across the market:

```
Per-chip annual FTNS revenue = 150M shards × 0.05 FTNS/shard
                             = 7.5M FTNS/year/chip (gross)
```

Subtract protocol flows per PRSM-TOK-1 §8.1:
- 20% burn: 1.5M FTNS destroyed
- 6.4% creator royalty: 0.48M FTNS to model publisher
- 1.6% Foundation treasury fee: 0.12M FTNS
- **72% to serving node (Prismatica operator): 5.4M FTNS/year/chip**

At an assumed **$2/FTNS** secondary market price (see §7 for price modeling):
- Net per-chip gross revenue: **$10.8M/year**
- Chip amortization (3-year life): **~$143/year**
- Power + cooling + data center: **~$5K/year** (comparable to H100 operation)
- **Per-chip gross margin: ~$10.8M - $5K = ~$10.79M/year** (≈99% gross margin at this price)

This is extraordinary margin and **[LOW]** confidence because it depends on:
1. FTNS price holding at ~$2 (§7).
2. Marketplace price holding at ~0.05 FTNS/shard (§4.3).
3. 60% utilization at the assumed throughput (§4.1).

### 4.3 Price compression over time

0.05 FTNS/shard is a bootstrap price. As competition enters and CIS tier becomes standard, expect price compression. Scenarios:

| Year | Avg price per shard | Reason |
|---|---|---|
| 2026-2028 | 0.05 FTNS | Bootstrap; CIS supply-limited |
| 2029-2031 | 0.03 FTNS | More implementers online; commodity compression begins |
| 2032-2035 | 0.01 – 0.02 FTNS | Mature competitive market |

Revenue per chip compresses correspondingly. At $2/FTNS and 150M shards/year:
- Bootstrap (0.05): $10.8M/year
- Commodity (0.02): $4.3M/year
- Mature (0.01): $2.2M/year

Even at the mature-market price, per-chip gross margin remains attractive (~$2.2M - $5K ≈ $2.2M), **but only if FTNS price maintains utility anchor**. If FTNS price compresses in parallel, revenue drops further.

---

## 5. Meganode unit economics

A Prismatica T4 meganode operates at scale: 100-1000 CIS chips per site.

### 5.1 Capital + operating cost per meganode

At 500 CIS-C2 chips per meganode **[MEDIUM]**:

| Item | Capital | Annual OpEx |
|---|---|---|
| CIS chips (500 × $800 street) | $400K | — |
| Chip amortization (3yr) | — | $133K |
| Server chassis + integration (500 nodes × $5K) | $2.5M | — |
| Chassis amortization (5yr) | — | $500K |
| Data center lease | — | $1.2M |
| Power (500 × 700W × 24h × 365d × $0.08/kWh) | — | $245K |
| Cooling + networking | — | $400K |
| Operator staff (2 FTEs × $200K) | — | $400K |
| **Total meganode cost per year** | | **$2.88M** |

### 5.2 Meganode annual revenue (base case)

500 chips × 5.4M FTNS/chip/year = 2.7B FTNS/year gross to the operator (after protocol splits).

At $2/FTNS: **$5.4B/year gross revenue at bootstrap prices**.
At $0.50/FTNS (bear case): **$1.35B/year**.

Even in the bear case, a single meganode is substantially profitable. **[LOW]** confidence — depends heavily on FTNS price.

### 5.3 Meganode margin analysis

| Scenario | FTNS price | Meganode annual revenue | OpEx | **Net** |
|---|---|---|---|---|
| Bull (adoption strong, bootstrap pricing) | $2.00 | $5.4B | $2.88M | **$5.4B** |
| Base (healthy adoption, some price compression) | $0.75 | $2.0B | $2.88M | **$2.0B** |
| Bear (slow adoption, FTNS at utility floor only) | $0.25 | $675M | $2.88M | **$672M** |

Even the bear-case meganode nets >$100× its annual operating cost. These numbers are deliberately chosen to illustrate the structural advantage of CIS meganodes under PRSM's economics, but they are **[LOW]** confidence — the actual FTNS price path is the single most-sensitive variable.

### 5.4 Prismatica meganode count trajectory

Prismatica's realistic meganode deployment pace:

| Year | Meganodes | Total chips deployed |
|---|---|---|
| Year 2 (first CIS-C2 tape-out) | 0 (bring-up only) | Pilot: 100 chips |
| Year 3 | 1 | 500 |
| Year 4 | 3 | 1,500 |
| Year 5 | 8 | 4,000 |
| Year 7 | 25 | 12,500 |
| Year 10 | 60 | 30,000 |

Each new meganode = $2.9M annual OpEx + the revenue above. Compounded at the base-case FTNS price ($0.75) and base-case meganode revenue ($2.0B/meganode/year), by Year 5 Prismatica operates at ~$16B annual run-rate.

These numbers feel aggressive; they are a direct consequence of the Phase 3 marketplace bootstrap pricing and realistic inference throughput per CIS chip. The aggression is contained in the **[LOW]** confidence tags on FTNS price + market-share capture.

---

## 6. Prismatica revenue streams

Prismatica has two principal revenue streams:

### 6.1 Stream 1: Meganode operating revenue

Per §5 above. Prismatica earns FTNS for shard-execution services, converts some fraction to USD operational currency, retains the balance as FTNS treasury.

### 6.2 Stream 2: Commercial services layered on top

Beyond base meganode revenue, Prismatica offers:

1. **Managed-deployment services** — Prismatica handles the entire lifecycle of sealing a publisher's weights to CIS chips, monitoring inference quality, handling edge cases. Pricing: typical SaaS model, $50K-$500K/year per major publisher relationship.
2. **CIS chip design licensing** — Prismatica's CIS-compliant chip design (post-certification) may be licensed to other fabs as the reference implementation, under commercially reasonable terms. Revenue: $1M-$10M per licensee upfront + royalties.
3. **Integration consulting** — for enterprises deploying their own private meganodes using Prismatica's reference integration. Typical engagement: $500K-$2M.
4. **Hosted-publisher services** — some model publishers want Prismatica to handle not just the inference but the model lifecycle (versioning, A/B testing, eval orchestration). Premium offering, revenue $500K-$5M/year per major contract.

Stream 2 is **[LOW]** confidence because these relationships don't exist yet. But they are the natural commercial expansion for a first-implementer that scales into a category leader. Conservative Year 5 estimate: $50M – $200M annual Stream 2 revenue.

### 6.3 Prismatica is NOT a token project

Worth restating: Prismatica does not sell FTNS to investors. FTNS enters Prismatica's treasury organically (earned from meganode operation). Investor returns flow from Prismatica's **enterprise value growth**, which is a function of:

- Operating revenue (Streams 1 + 2).
- FTNS treasury mark-to-market.
- Commercial relationships + brand value.

At exit (IPO, acquisition, or secondary sale), investors are valued on standard metrics (revenue multiples, EBITDA, strategic premium). The FTNS treasury appears as a balance-sheet asset, valued at prevailing market price.

---

## 7. FTNS price anchoring

The single most-sensitive variable above is FTNS price. This section addresses: what realistic floor and ceiling exist?

### 7.1 Utility floor

FTNS has an emergent utility floor: inference service buyers need FTNS. As adoption grows, utility demand grows. This floor is NOT a Foundation-enforced price peg — but it is a structural consequence of the compensation-for-services token design.

Estimated utility floor at $1B/year network volume **[LOW]**:
- Network throughput: $1B/year requires ~500M FTNS/year in velocity (at $2/FTNS) or 1B FTNS/year (at $1/FTNS).
- With ~50M circulating supply at Year 3, FTNS velocity of 10-20x annually implies equilibrium price near $1-2.

This floor rises with adoption — at $10B/year network volume, the utility floor is an order of magnitude higher.

### 7.2 Protocol deflation

20% burn on every payment, applied to a $1B/year network, burns $200M/year of FTNS at then-current prices. This is a structural floor support: at $1/FTNS, burn removes 200M FTNS/year from circulation against ~50M new FTNS emission in Epoch 1 (net -150M FTNS/year circulating → scarcity → upward price pressure).

Crossover point where burn exceeds emission: expected in Epoch 2-3 at realistic adoption rates.

### 7.3 Ceiling dynamics

FTNS price is bounded above by:
- Operator opportunity cost. If FTNS at $50/unit means a $100M/year meganode, some operators would sell FTNS to lock gains. Operator selling dampens ceiling.
- Secondary-market arbitrage (Uniswap + Coinbase once listed).
- POL reserve's discretionary interventions (§10 of PRSM-TOK-1).

Realistic equilibrium bands **[LOW]**:

| Year | FTNS price band | Reasoning |
|---|---|---|
| 2026 (bootstrap) | $0.20 – $0.80 | Low circulating, limited utility demand |
| 2028 (ramp) | $0.50 – $3.00 | Marketplace at scale, burn active |
| 2031 (Phase 7 + staking) | $1.00 – $8.00 | Staking tier restriction + deflation |
| 2035 (mature) | $2.00 – $15.00 | Utility stable, mature burn-emission crossover |

These are bands, not projections. Actual price will wander within (or outside) these on timescales shorter than the analysis.

---

## 8. Scenario modeling

Three scenarios, 5- and 10-year horizons. All numbers in USD equivalent at the year stated.

### 8.1 Base case (probability-weighted)

Assumptions:
- Adoption grows per base-case §2.3 share capture.
- FTNS price follows §7.3 center of bands.
- Prismatica executes per §5.4 meganode deployment.
- CIS-C2 ships by Year 3; CIS-C3 by Year 5.

| Year | Meganodes | FTNS price | Meganode op. revenue | Stream 2 revenue | FTNS treasury (MTM) | **Enterprise value estimate** |
|---|---|---|---|---|---|---|
| Year 3 | 1 | $0.50 | $135M | $20M | $20M | **$300M – $500M** |
| Year 5 | 8 | $0.75 | $1.6B | $80M | $200M | **$2B – $5B** |
| Year 7 | 25 | $1.50 | $6.2B | $250M | $1.2B | **$10B – $25B** |
| Year 10 | 60 | $3.00 | $20B | $600M | $5B | **$40B – $100B** |

Enterprise-value estimates use 5-10x revenue multiples typical for hardware-infrastructure unicorns with software-margin economics. **[LOW]** confidence; sensitivity in §9.

### 8.2 Bull case

Assumptions: upper-band adoption, FTNS price in upper band, Prismatica captures larger share via first-mover advantage + CIS-C3 leadership.

Year 10 enterprise value range: **$100B – $300B.** At this scale Prismatica is a public-company-scale business with strategic optionality (IPO, acquisition by a hyperscaler, merger with another AI infrastructure entity).

### 8.3 Bear case

Assumptions: low adoption, FTNS at utility floor only, CIS-C2 slips 2 years, Prismatica share below 1% of open-weights market.

Year 10 enterprise value range: **$500M – $2B.** Still a business worth owning, but more in line with a typical venture outcome (modest multiple on invested capital).

### 8.4 Probability-weighted expected value

Simple triangular weighting (20% bear / 55% base / 25% bull):

```
Year 10 EV = 0.20 × $1B + 0.55 × $70B + 0.25 × $200B
           = $0.2B + $38.5B + $50B
           = $88.7B
```

This is illustrative. Real investor analysis would use Monte Carlo over the assumption ranges. The number's purpose: to show that under reasonable assumptions, Prismatica is a venture outcome in the upper quartile even without bull-case execution.

### 8.5 What kills the math

The bear case is not catastrophic — it's still a $500M-$2B business. The scenarios where investors lose money:

1. **FTNS designation as security.** Despite §4.4 analysis in PRSM-TOK-1, SEC enforcement could invalidate the utility-token posture. Mitigation: compensation-only distribution, Prismatica-equity bootstrap, published Howey analysis. Residual risk: non-zero.
2. **H100 CC publicly broken** AND PRSM-CIS-1 slips >5 years. The first buys PRSM-CIS-1 urgency; the second removes the premium tier. Combined, they destroy Prismatica's differentiation vs commodity cloud GPU.
3. **Single-manufacturer capture** despite multi-manufacturer requirement. If Prismatica ends up the only C2-certified implementer for >5 years, the ecosystem-diversity premium collapses. Mitigation: Foundation spec + certification committee + on-chain governance — all structural defenses per PRSM-GOV-1.
4. **PRSM protocol fails adoption.** No market = no revenue regardless of Prismatica execution. This is the standard venture risk; it is why investors diversify.

---

## 9. Sensitivity analysis

Which assumptions matter most for Year 10 enterprise value:

| Assumption | Bear value | Base value | Bull value | EV delta (bear→bull) |
|---|---|---|---|---|
| FTNS price at Year 10 | $0.50 | $3.00 | $10.00 | **HIGHEST** — 20× swing |
| PRSM addressable market capture | 1% | 5% | 15% | High — 5x swing |
| Meganodes deployed by Year 10 | 20 | 60 | 120 | High — 3x swing |
| Avg FTNS per shard | 0.015 | 0.03 | 0.06 | Medium — 2x swing |
| CIS-C2 ship date | Year 5 | Year 3 | Year 2 | Medium — timing impact |
| Stream 2 share of revenue | 1% | 5% | 10% | Low — offset by Stream 1 scale |

**The dominant variable is FTNS price.** An investor thesis that assumes strong Prismatica execution but a low FTNS price produces a ~$1-5B outcome. A thesis that assumes the converse produces a ~$10-50B outcome.

The paper's explicit recommendation for investors: **do not buy Prismatica equity unless you believe the utility floor will hold at or above $1/FTNS by Year 5** under realistic adoption. That's the threshold below which the base case degrades into the bear case.

---

## 10. Comparison to alternative investments

### 10.1 Comparable venture plays

| Entity | Thesis | Comparable valuation trajectory |
|---|---|---|
| CoreWeave | Cloud AI infrastructure, GPU rental | $30B+ at 2024 IPO after 5 years |
| Lambda Labs | Cloud AI + hardware | $1-3B private |
| Groq | Inference-specific silicon | ~$3B at 2024 round |
| SambaNova | AI accelerator silicon | ~$5B private |
| Ceremonia (hypothetical frontier-TEE startup) | Close analog | N/A yet |

Prismatica differs from these in three ways:
1. **Has a protocol-level moat**, not just hardware. PRSM protocol + FTNS creates an aligned economic network around Prismatica's hardware.
2. **Open-standards backstop**. PRSM-CIS-1 means Prismatica's hardware can work with any ecosystem participant, not just Prismatica-managed deployments.
3. **Treasury-appreciation optionality**. FTNS treasury acts as a call option on network success, independent of operating revenue.

### 10.2 Comparable token-project plays

An FTNS-native investor thesis (buy FTNS directly) is **not** what Prismatica equity provides. FTNS is not sold to investors; it is earned as compensation. An investor wanting FTNS exposure must either:
- Earn FTNS by contributing to the network (not an investment; compensation for work).
- Buy FTNS on secondary markets after earned FTNS enters circulation.

Prismatica equity is a **fundamentally different instrument**: a properly-classified security with standard corporate governance, audited financials, and conventional exit mechanisms. Prismatica's upside is partially FTNS-correlated (treasury holdings appreciate with FTNS price) but primarily operating-business-correlated.

### 10.3 Comparable infrastructure plays

| Investment | Return profile | Risk profile |
|---|---|---|
| Prismatica equity (this paper) | 5-100× over 10 years (scenarios) | Moderate-high venture risk |
| Public GPU-cloud stock (CoreWeave etc.) | 2-5× over 10 years | Moderate public equity risk |
| Hyperscaler equity (MSFT, GOOG, META) | 1.5-3× over 10 years | Low-moderate |
| NVIDIA equity | Highly variable (2-5×) | Moderate-high (concentration) |
| Direct compute capacity (data-center GP) | 1.5-2.5× over 10 years | Low, illiquid |

Prismatica equity sits at the risky-end of infrastructure investment with asymmetric upside driven by protocol-level network effects that the comparable plays don't have.

---

## 11. Validation plan

Before v1.0 ratification of this economic model, the Foundation + Prismatica commits to:

1. **Publishing quarterly operating metrics** post-Phase 3 mainnet: actual shards executed, actual FTNS flows, actual utilization, actual chip counts. Replaces [LOW] confidence with observed data.
2. **Commissioning independent economic review** by a qualified infrastructure-economics consultancy before Series B. Third-party validation of assumptions.
3. **Red-team adversarial analysis** — commissioning an adversarial modeler to attack the assumptions with worst-case scenarios. Publishing the attacks + responses transparently.
4. **Periodic re-baseline** — annual update to this paper incorporating observed data + revised forward projections.
5. **Open scenario modeler** — Foundation publishes a notebook or web tool where any party can reproduce the analysis with their own assumptions. Ensures the math is scrutable, not black-boxed.

---

## 12. Open questions

1. **Specific Prismatica round sizing** — Series A targets $X at $Y valuation; tied to §3.4 capital-to-first-chip math. Pending Prismatica board + lead investor term sheet.
2. **Hybrid tokenomics adoption** — if PRSM-TOK-1 §8.5 hybrid is ratified, Foundation takes 15-25% equity in Prismatica. This changes §6.3 investor-return analysis materially. Pending counsel + ratification (§13 of PRSM-TOK-1).
3. **Insurance mechanism** — investors may require chip-damage, certification-failure, or CIS-breach insurance. Pricing + availability TBD; factored into OpEx at ~1% in §5.1 (not separately itemized).
4. **Secondary-market FTNS liquidity at exit** — IPO or acquisition multiples for a company with large FTNS treasury are untested. Market precedent will mature during the investment horizon.
5. **Regulatory evolution** — MiCA final implementation, SEC crypto policy under each subsequent administration, FinCEN token-compensation framework. All move on 1-4-year timescales. Material impact on enterprise-value multiples.
6. **Talent concentration risk** — Prismatica's early engineering team concentration is a standard startup risk but acute for a deep-tech hardware company. Succession + key-person insurance practices TBD.

---

## 13. Summary

- **Prismatica equity** is a properly-classified security (Reg D 506(c) accredited) whose enterprise-value growth comes from operating revenue + FTNS treasury appreciation.
- **Year 10 base-case enterprise value:** ~$40-100B under the assumptions stated.
- **Year 10 bear-case enterprise value:** still ~$500M-$2B (a normal venture outcome, not a catastrophe).
- **Year 10 bull-case enterprise value:** $100B+ (hyperscaler-scale, public-company territory).
- **Dominant risk variable:** FTNS price at Year 5+. If utility floor holds ≥$1/FTNS, base/bull cases are plausible. Below that, bear case dominates.
- **Structural protections baked in:** PRSM-GOV-1 prohibits Foundation/Prismatica revenue-sharing; PRSM-CIS-1 requires multi-manufacturer at C2+; PRSM-TOK-1 forbids Foundation token sales. These prevent the most common rent-extraction outcomes.
- **Investor thesis:** Prismatica is the first-implementer play on a neutral AI-infrastructure protocol. Upside is asymmetric (network effects + treasury optionality); downside is bounded at normal venture levels.

Any investor interested in engaging further should request the full Reg D offering materials (private placement memorandum, financial projections, cap table, risk factors) through Prismatica's investor contact.

---

## 14. Acknowledgements

Estimates and modeling draw on:
- Public AI hardware cost analyses (SemiAnalysis, Chip Insights, etc.) for wafer/yield economics.
- PRSM_Tokenomics.md §6 scenario framework for FTNS price ranges.
- Phase 3 acceptance test data for per-shard pricing anchor.
- PRSM-CIS-1 tier specifications for certification cost calibration.
- Comparable venture-infrastructure precedents (CoreWeave, Groq, SambaNova) for valuation frames.

---

## 15. Change log

**v0.1 (2026-04-21):** Initial economic model connecting PRSM-CIS-1 chip costs to PRSM-TOK-1 token flows to Prismatica enterprise value for investor due diligence. Bear / Base / Bull scenarios over 5 and 10 year horizons. Sensitivity analysis highlights FTNS price as dominant variable. Validation plan commits to quarterly operating metrics + independent economic review before Series B.

---

**End of PRSM-ECON-WP-1 v0.1 Draft.**

*This document is not an offer to sell securities. Any offer will be made only through formal offering materials to accredited investors under Regulation D Rule 506(c). Projections are illustrative only and subject to the assumptions stated inline.*
