# PRSM-POL-1: Treasury Reserve & Risk-Transfer Policy

**Document ID:** PRSM-POL-1
**Version:** 1.0
**Type:** Foundation council governance policy
**Issued:** 2026-05-05
**Ratified:** 2026-05-06 by founder (sole council member pending
council expansion to 2-of-3) per Council Resolution
[`PRSM-CR-2026-05-06-1`](PRSM-CR-2026-05-06-1.md). All five §12 open
questions ratified at their drafted defaults.
**Author:** Founder (drafted for council review)
**Status:** **Ratified — v1.0** (effective 2026-05-06). Subject to
re-ratification by the full 2-of-3 Foundation council upon expansion;
the §6 annual review cadence allows in-place re-confirmation without
full re-drafting.

**Companion docs:**
- `audits/AUDIT_PLAN.md` §11 (risk-transfer layer rationale)
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` (governance authority)
- `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` §8.1 (2% network fee mechanics)
- `docs/2026-04-22-risk-register-track-2.md` (risk inventory)
- `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` (incident response — references this policy)

---

## 1. Purpose

The PRSM Foundation Safe accumulates a 2% network fee from every
RoyaltyDistributor settlement. This treasury serves three distinct
purposes:

1. **Operations:** auditor budgets, council compensation, security
   ops, infrastructure, legal counsel.
2. **Self-insurance reserve:** funds available for victim
   reimbursement in the event of a smart-contract exploit or
   key-compromise incident.
3. **Strategic optionality:** runway for protocol-level decisions
   (governance transition, jurisdiction changes, downstream investments).

This policy establishes:

- The **floor reserve** below which non-emergency operational
  disbursement is paused.
- The **target reserve** above which external risk-transfer
  (insurance) becomes a viable option.
- The **engagement criteria** for external coverage purchase.
- The **disbursement authorization** workflow.

---

## 2. Definitions

| Term | Meaning |
|------|---------|
| **Foundation Safe** | The 2-of-3 multisig at `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` on Base mainnet. |
| **Treasury balance** | USD-equivalent value of all assets held in the Foundation Safe at the spot rate of CoinGecko + Chainlink oracle aggregate, sampled at council quarterly review. |
| **TVL** | Total Value Locked across all PRSM smart contracts (escrow + stake + emission cap), denominated in FTNS at oracle rate, summed in USD. |
| **Floor reserve** | The minimum treasury balance below which non-emergency operational disbursement is paused. |
| **Target reserve** | The treasury balance above which external coverage purchase becomes a viable option. |
| **Max-single-user position** | The largest USD-equivalent FTNS position held by any single address (escrow + stake combined), measured at quarterly review. |

---

## 3. Reserve targets

### 3.1 Floor reserve

**Floor: $2,000,000 USD-equivalent.**

Below this floor:
- All non-emergency operational disbursement is paused.
- Audit budget commitments may continue (existing engagement
  letters honored).
- Council compensation continues.
- New hires / new contracts: paused until floor recovered.

The floor is sized to cover ~6 months of Foundation operating costs at
the current burn rate (audit-bundle in-progress, council operating)
plus a victim-reimbursement reserve for a small-to-medium incident.

### 3.2 Target reserve

**Target: $10,000,000 USD-equivalent.**

At or above this level:
- External coverage purchase becomes a viable option (see §4).
- Council may consider strategic disbursements (foundation grants,
  ecosystem investments, DAO transition costs).
- Annual reserve allocation adjustments per §6 may proceed.

### 3.3 Operational floor

**Operational reserve: 12 months of operating costs.**

Below this minimum:
- Audit pipeline pace adjusts: lower-priority RFPs deferred.
- Cost-cutting measures activate per Foundation operating handbook.

---

## 4. Risk-transfer (insurance) policy

### 4.1 Pre-mainnet (now through Gate B)

**No external coverage purchased.** TVL is small enough that
self-insurance from Foundation reserve is the cheaper, simpler path.

### 4.2 Post-Gate-B / Post-Phase-8 emission

**Re-evaluate quarterly.**

**Triggers for purchasing external coverage:**

1. Any single user position exceeds 1% of treasury reserve, OR
2. TVL exceeds $5M total, OR
3. Single insured-event maximum payout (per §3.1 floor reserve)
   would deplete < 50% of treasury post-payout.

When any trigger fires, the council must consider coverage purchase
within 30 days.

### 4.3 Post-DEX-listing

**Mandatory external coverage** at minimum 2× max-single-user
position. This is required to maintain credible TVL backstop for
institutional participants.

### 4.4 Coverage products

The Foundation evaluates the following products at each quarterly
review (with most-recent assessment dates):

| Product | Type | Notes |
|---------|------|-------|
| **Nexus Mutual** | Mutual member-pooled | Most established; per-protocol capacity limited; KYC required |
| **Sherlock** | Audit + cover combined | Tightly coupled to L4 vendor choice; useful only if Sherlock is selected for L4 |
| **Unslashed Finance** | Discretionary mutual | Smaller capacity; permissionless |
| **Risk Harbor** | Parametric | Narrow applicability (depeg-style triggers) |
| **InsurAce** | Mutual | Smaller capacity; available |

**Selection criterion:** capacity ≥ max-single-user position × 2,
premium ≤ 5% of covered amount per year.

---

## 5. Disbursement authorization

### 5.1 Operational disbursements (recurring)

| Category | Cap per disbursement | Authority |
|----------|----------------------|-----------|
| Audit engagement payment | Per engagement letter | Founder + 1 council signer (2-of-3 sig) |
| Council compensation | Per compensation policy | Founder + 1 council signer |
| Infrastructure (hosting, monitoring) | $5K/mo | Founder + 1 council signer |
| Legal counsel retainer | Per engagement letter | Founder + 1 council signer |

### 5.2 One-time / strategic disbursements

| Amount | Authority |
|--------|-----------|
| < $10K | Founder + 1 council signer (2-of-3 sig) |
| $10K – $100K | 2-of-3 council sig + 7-day council notice |
| $100K – $500K | 2-of-3 council sig + 14-day council notice |
| > $500K | 2-of-3 council sig + 30-day council notice + Foundation member vote (when chartered) |

### 5.3 Emergency disbursement (incident response)

In the event of an active exploit or key-compromise incident, the
**Exploit Response Playbook §3-§5** governs disbursement:

- Up to $250K victim reimbursement: 2-of-3 council sig within 4
  hours.
- Above $250K: same authority as §5.2 strategic disbursement, but
  14-day notice may be compressed to 24h with public council
  resolution.

---

## 6. Annual treasury allocation

Each calendar year, the council ratifies a treasury allocation:

| Category | Target % of post-floor treasury |
|----------|----------------------------------|
| Operations (12-month runway) | 40% |
| Self-insurance reserve | 30% |
| Strategic / grants | 20% |
| External coverage premium | 10% (or 0% if no policy active) |

Allocation deviations require council ratification; the council
publishes allocation results post-ratification.

---

## 7. Reporting cadence

### 7.1 Quarterly

- Treasury balance (USD-equivalent) at Q-end
- TVL snapshot
- Max-single-user position
- Triggers status (§4.2 #1-#3)
- Coverage premium expense (if applicable)
- Allocation actuals vs targets

Published within 14 days of quarter close to:
- Foundation council
- Foundation members (when chartered)
- Public-facing summary on prsm.network/transparency

### 7.2 Annual

- Comprehensive treasury report
- Allocation ratification for next year
- Re-evaluation of §3 reserve levels (inflation, TVL growth, etc.)
- External-coverage decision (purchase / renew / decline)

### 7.3 Ad-hoc (post-incident)

Any incident requiring §5.3 emergency disbursement triggers a
written council resolution within 7 days of resolution + public
disclosure within the disclosure-policy window (`AUDIT_PLAN.md` §12).

---

## 8. Reserve floor breach protocol

If the treasury balance falls below the floor (§3.1):

1. **Immediate:** founder notifies council via Slack/Signal within
   24 hours of detection (Foundation Safe balance is checked at
   each quarterly review and on every Forta P0/P1 alert).
2. **Within 7 days:** council convenes emergency review.
3. **Within 14 days:** disbursement-pause notice published; remediation
   plan published (revenue forecast, cost-cutting, emergency
   fundraise).
4. **Within 30 days:** floor recovery plan ratified or council
   decision to consciously operate below floor (with explicit
   justification).

---

## 9. External-counterparty notification

When external coverage is purchased:

- Coverage policy reference + insurer added to public transparency
  page.
- Premium amount disclosed quarterly.
- Coverage capacity disclosed (so participants can self-assess).
- Claim history (if any) disclosed within 30 days of claim
  resolution.

---

## 10. Amendment process

This policy is amendable by Foundation council. Amendment process:

1. Proposed amendment published on the council governance forum
   with 14-day comment period.
2. Council votes (2-of-3 ratification required for adoption).
3. Adopted amendment published with effective date and prior
   version archived in `docs/governance/`.

Material amendments (changes to §3 reserve floors, §4 trigger thresholds,
§5 disbursement authorities) require additional 30-day public notice.

---

## 11. Audit & compliance

This policy is reviewed:

- **Annually** by Foundation counsel (L8 retainer scope) for
  Cayman regulatory compliance.
- **Quarterly** by founder + council for operational adherence.
- **Per material amendment** for cross-jurisdictional review (Cayman
  + US securities counsel — L8 Track A + Track B).

---

## 12. Ratified parameters (was: open ratification questions)

The five parameters below were ratified at their drafted defaults
on 2026-05-06 by founder as sole council member. Each is subject to
the §10 amendment process (council vote with 30-day public notice
for material amendments) and the §7.2 annual review.

1. **✅ Floor reserve: $2,000,000 USD-equivalent.** Sized to ~6
   months Foundation operations + small-incident victim
   reimbursement reserve.
2. **✅ Target reserve: $10,000,000 USD-equivalent.** Threshold
   above which external coverage purchase becomes viable.
3. **✅ Trigger #1 threshold: 1% of treasury** for max-single-user-
   position concentration. Standard treasury-management threshold.
4. **✅ Disbursement authorization tiers (§5.2):** ratified as
   drafted (< $10K founder + 1 signer; $10K–$100K 7-day notice;
   $100K–$500K 14-day; > $500K 30-day + member vote).
5. **✅ Annual allocation: 40 / 30 / 20 / 10** (Operations /
   Self-insurance / Strategic / Coverage premium). Re-ratified
   annually per §6.

**Pre-mainnet caveat:** with mainnet TVL effectively zero (audit-
bundle not yet deployed) the $2M floor and 1% concentration
threshold are forward-looking; they bind when treasury inflows from
the 2% network fee begin accumulating post-mainnet.

---

## 13. Effective date + ratification

**Effective date:** 2026-05-06.

**Ratification record:** Council Resolution
[`PRSM-CR-2026-05-06-1`](PRSM-CR-2026-05-06-1.md). The ratifying
resolution identifies this policy by ID `PRSM-POL-1` and references
the commit at which the ratified content lives (see resolution
§4 for commit hash).

**Council composition at ratification:** 1-of-1 (founder as sole
council member). This satisfies the council-resolution path of §13
above; the policy commits the Foundation to expanding to 2-of-3
council membership and re-confirming this ratification with full
quorum once seats #2 and #3 are filled. See PRSM-CR-2026-05-06-1
§3 for the council-expansion commitment timeline.

**Stewardship:** Founder is policy steward until Foundation Director
of Engineering hire. Steward proposes amendments + monitors compliance.

---

## 14. Signoff

| Council member         | Signature                              | Date       |
|------------------------|----------------------------------------|------------|
| Ryne Schultz (Founder) | See PRSM-CR-2026-05-06-1 §5            | 2026-05-06 |
| Council seat 2         | _Vacant — pending council expansion_   | _pending_  |
| Council seat 3         | _Vacant — pending council expansion_   | _pending_  |

Sole-founder ratification is the legitimate ratification path under
§13 ("council resolution referencing this document by ID PRSM-POL-1 +
commit hash") at council size 1. The two empty seats are a council-
expansion commitment, not a ratification deficiency.

---

*This policy implements `audits/AUDIT_PLAN.md` §11 risk-transfer
layer recommendation. Companion to `EXPLOIT_RESPONSE_PLAYBOOK.md`
which references the §5.3 emergency disbursement authority.*
