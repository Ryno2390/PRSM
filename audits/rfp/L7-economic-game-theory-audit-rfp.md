# L7 Economic / Game-Theory Audit — Request for Proposal

**Engagement:** PRSM tokenomics, emission schedule, incentive alignment,
MEV exposure
**Issuing organization:** PRSM Foundation (Cayman Islands nonprofit)
**Issued:** 2026-05-05
**Response deadline:** 2026-05-26 (3 weeks)
**Engagement window:** 2026-06 to 2026-08
**Budget envelope:** $50,000 – $100,000 USD

**Primary contact:** schultzryne@gmail.com / security@prsm.network
**PGP:** see SECURITY.md
**Repository:** https://github.com/Ryno2390/PRSM

---

## 1. Engagement summary (TL;DR)

PRSM is a decentralized inference protocol with FTNS as the native
work-and-payment token. We are seeking a **DeFi economic risk firm**
or **token-engineering academic group** to perform an adversarial
review of:

- FTNS emission schedule (halving + supply cap dynamics)
- Network-fee + royalty split (creator / treasury / serving node)
  — note: burn-on-use was DROPPED; the deployed token has no
  burn-on-use (burnFrom exists for the bridge only)
- StakeBond slashing economics (70/30 challenger/Foundation split,
  100% self-slash protection)
- POL (Protocol-Owned Liquidity) parameterization
- Foundation revenue model (2% network fee → treasury reserve)
- Foundation governance charter (Cayman nonprofit + 2-of-3 multisig
  + future DAO transition)

**Standard smart-contract audits don't cover this.** L4 audits
*correctness* of the splitter; L7 audits *whether the parameters
incentivize the protocol's intended behavior under adversarial agents*.

We are seeking proposals from ML systems-economic security firms with
prior published agent-based simulation work on DeFi economics.

---

## 2. Why us / why this scope

PRSM's economic surface differs from typical DeFi in three
non-trivial ways:

1. **Inference work, not lending.** Most DeFi economic audits cover
   AMMs / lending / yield. PRSM's substrate is *compute work* (a
   provider serves an inference request, gets paid in FTNS). The
   incentive model is closer to compute-marketplace economics
   (Akash, Render, IO.NET) than to AMM economics.

2. **Slashing-backed quality assurance.** StakeBond + the challenge
   path mean providers post collateral that gets slashed (redistributed
   per the 70/30 challenger/Foundation split — this is distinct from
   token burn-on-use, which was dropped) on adversarial behavior.
   The economic question is whether the
   slash amount + bounty split create *correct incentives at
   equilibrium* — not just whether the math works on paper.

3. **2% network fee = Foundation's only revenue.** The 2% fee on
   every settlement is the Foundation's primary income stream
   (vs. Prismatica's equity raise, which is a separate revenue path
   for the for-profit subsidiary). The auditor's economic review is
   directly load-bearing on Foundation runway projections.

---

## 3. Vendor preferences

**Top picks:**

1. **Gauntlet.** Strongest DeFi economic-risk modeling firm; prior
   work on Aave / Compound / Uniswap / dYdX. Agent-based simulation
   capabilities.
2. **Chaos Labs.** Real-time DeFi simulation + risk management;
   prior work on Aave, dYdX, Spark, Frax.
3. **BlockScience.** Token-engineering academic firm (CADCAD-based
   formal modeling). Strong on incentive-design proofs.

**Academic candidates:**

4. **MIT DCI** (Digital Currency Initiative) — published work on
   token economics + governance.
5. **Stanford engineering** (e.g., Joe Bonneau group) — published
   work on DeFi MEV + tokenomics.
6. **Cornell IC3** — Sirer / Eyal group; published work on
   selfish-mining economics.

This RFP is being sent to **#1 Gauntlet** and **#2 Chaos Labs**
(commercial track) plus **#3 BlockScience** (CADCAD modeling) and
**#4 MIT DCI** (academic track). Proposals from #5 / #6 considered
if scope fit better.

**Selection criteria:**

- Published prior work on DeFi / token economics / agent-based
  simulation (last 24 months).
- Apache-2.0 / MIT-friendly engagement terms (we want to publish
  the report).
- Availability in the W2-W14 window (target start: 2026-06).
- Quoted price within the $50K-$100K envelope.

---

## 4. Scope of work

### 4.1 In scope

| Topic | Focus |
|-------|-------|
| **FTNS emission schedule** | Halving cadence, supply cap, mint authorization (EmissionController + FTNSTokenSimple) |
| **Network fee + royalty split** | 2% network fee, creator royalty rate (per content), serving-node share (burn-on-use was DROPPED — the v2 redeploy corrected the split but added no burn) |
| **StakeBond slashing economics** | 50/100% slash rates per tier, 70/30 challenger/Foundation split, 100%-to-Foundation on self-slash |
| **Challenge incentives** | Whether 70% bounty incentivizes honest challengers without creating griefing economics |
| **POL parameterization** | Per Q4 ratification: intervention thresholds, max-per-window, kill-switch params |
| **Foundation revenue model** | 2% fee → treasury reserve; runway projections at various network volumes |
| **Foundation governance** | Cayman nonprofit structure, 2-of-3 multisig, future DAO transition (Q6 ratification) |
| **MEV exposure** | Front-running of: commitBatch, finalizeBatch, challengeReceipt; sandwich opportunities on RoyaltyDistributor.distributeRoyalty + claim |
| **Equilibrium analysis** | Are honest providers + honest requesters Nash-equilibrium-rational under the parameters? |
| **Adversarial scenarios** | Sybil, collusion (provider + requester pair), wash-trading, slow-start griefing, mass-unbond cascade |

### 4.2 Out of scope

| Topic | Why out of scope |
|-------|------------------|
| Smart-contract correctness (split math, etc.) | L4 covers this |
| Crypto primitives | L3 covers this |
| Off-chain ML supply-chain | L5 covers this |
| Legal/regulatory securities classification | L8 covers this |
| Bug bounty design | L9 covers this |
| Frontend UX / wallet | Not adversarial economic surface |

### 4.3 Specific questions we want answered

1. **Is the 2% network fee economically sustainable** for Foundation
   operations at low network volume (year 1)? At medium volume
   (year 3)? Under a competitive scenario where another protocol
   undercuts on fees?

2. **Does the 70/30 challenger/Foundation split correctly incentivize
   honest challenges** without creating griefing economics? What's
   the minimum bounty size that makes a marginal challenge
   worthwhile after gas?

3. **Halving schedule + post-Q3 ratification parameters** — does
   the schedule achieve the inflation/deflation targets in the
   tokenomics spec under realistic adoption curves?

4. **Wash-trade economics** — the L2 audit's HIGH-1 finding pinned
   a 10× wash-trade arbitrage gap (resolved by the v2 RoyaltyDistributor
   redeploy's corrected split, NOT by burn — the originally-planned
   burn-on-use was dropped). Validate the post-fix economics.

5. **Foundation revenue runway** — at what network-volume threshold
   does the 2% fee cover Foundation operating costs (audit budget,
   council compensation, security ops)?

6. **MEV surface** — which on-chain functions are MEV-exposed, and
   what's the user-impact cost of that MEV under realistic
   conditions?

7. **POL kill-switch parameterization** — does the Q4 ratified
   per-window-max and intervention-threshold create the intended
   safety property without over-restricting honest market activity?

---

## 5. Deliverables

1. **Economic risk report (primary).**
   - Severity-classified findings (CRITICAL = protocol-breaking
     economics; HIGH = parameter mistuning; MEDIUM/LOW =
     optimization).
   - For each finding: agent-based simulation evidence,
     parameter-sensitivity analysis, recommended fix.
   - Executive summary suitable for council.
   - Detailed technical body.

2. **Agent-based simulation harness (if applicable).** Whether
   CADCAD / Gauntlet simulator / custom — the artifact should be
   licensed for Foundation re-runs as we tune parameters
   post-engagement.

3. **Specific quantitative answers** to the §4.3 questions.

4. **Optional: parameter sensitivity dashboard.** Web-accessible
   dashboard of the simulation harness so the council can re-run
   at parameter changes. Quote separately.

5. **License of deliverables.** Apache-2.0 or equivalent for
   methodology + report. Simulation code may be Foundation-private
   if vendor IP requires.

---

## 6. Pre-engagement artifacts

| Artifact | Location | What it gives |
|----------|----------|---------------|
| **Tokenomics spec PRSM-TOK-1** | `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md` | Full tokenomics design |
| **Governance charter PRSM-GOV-1** | `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` | Foundation structure, council powers |
| **Q1-Q7 ratification packet** | `docs/2026-04-21-tokenomics-ratification-packet.md` | Final pre-mainnet tokenomics decisions (epoch durations, halving factor, treasury split, etc.) |
| **Phase B Epoch 1 simulator** | `prsm/tokenomics/simulator/` (Phase B build) | Internal rate-calibration simulator (task #131 deliverable) |
| **Risk Register (Track 2)** | `docs/2026-04-22-risk-register-track-2.md` | Consolidated risk inventory |
| **Pre-mainnet exploit playbook** | `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md` | Incident response posture |
| **L4 firm RFP addendum** | `audits/rfp/L4-firm-rfp-addendum-20260505.md` | Adjacent audit context (L4 covers code; L7 covers economics) |
| **Master audit plan** | `audits/AUDIT_PLAN.md` v1.1 | Layer L7 = this engagement |

---

## 7. Engagement details

### 7.1 Timeline

- **2026-05-26:** Vendor proposals due
- **2026-06-02:** Vendor selection
- **2026-06-09:** Engagement starts
- **2026-06-09 to 2026-08-04:** Audit window (8 weeks for full
  agent-based simulation; 4-5 for narrower scope)
- **2026-08-11:** Final report delivered
- **2026-08-25:** Optional delta review on remediation parameters

### 7.2 Communication

- Primary contact: founder
- Backup: deputy-founder
- Engagement Slack/Signal channel can be set up
- Findings reported under coordinated-disclosure terms

### 7.3 Engagement terms

- **Compensation:** $50K-$100K depending on scope.
- **Payment:** 30% on engagement start, 30% on draft report,
  40% on final.
- **NDA:** mutual NDA covering pre-disclosure findings; superseded
  by public-report publication.
- **Liability:** auditor's standard MSA, capped at engagement fee.
- **Right to publish:** Foundation retains right; auditor may
  co-publish.

---

## 8. Proposal request

Proposals should include:

1. **Economic-modeling qualifications.** Prior agent-based
   simulation / DeFi economic-risk engagements (last 24 months).
2. **Lead modeler CV.**
3. **Methodology.** Agent-based simulation tooling, parameter-
   sensitivity approach, MEV-modeling approach.
4. **Scope quote.** Days of effort, total fee, breakdown by
   §4.1 topics.
5. **Timeline.** Earliest start, expected completion.
6. **Sample report.** Public sample of comparable economic risk
   audit work.
7. **Engagement terms.** License, NDA, liability per §7.3.
8. **References.** 2-3 prior clients we can contact.

---

## 9. Submission

Send proposals to: **schultzryne@gmail.com** (founder) and CC
**security@prsm.network**

Subject: `PRSM L7 Economic Audit RFP — [Your Firm]`

Attachments: PDF preferred. Sample reports as separate attachments.

We will acknowledge receipt within 24 hours and schedule a 30-minute
clarification call within 1 week.

---

## 10. Defense-in-depth context

| Layer | Surface | Vendor |
|-------|---------|--------|
| L3 | On-chain crypto | Crypto specialist (concurrent) |
| L4 | On-chain protocol composition | Code4rena public + firm pair (concurrent) |
| L5 | Off-chain ML supply chain | ML systems security firm (concurrent) |
| **L7 (this RFP)** | **Economic / game theory** | **DeFi economic-risk firm** |

A finding that escapes all four reviewers is the residual risk we
explicitly accept.

---

## 11. Signoff

**Issuing party:** PRSM Foundation (Cayman Islands)
**Authorized signatory:** Founder
**Date issued:** 2026-05-05

This RFP is non-binding until a mutually-executed engagement agreement
is in place. Foundation reserves the right to decline all proposals
or adjust scope based on responses received.

---

*See `audits/AUDIT_PLAN.md` §5 L7 for the strategic rationale behind
treating economic / game-theory as a distinct audit layer.*
