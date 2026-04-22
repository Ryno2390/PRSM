# PRSM-SUPPLY-1: Supply-Side Diversity Standard

**Document identifier:** PRSM-SUPPLY-1
**Version:** 0.1 Draft
**Status:** Pre-trigger governance spec. The thresholds this standard activates on (30% single-provider, 40% single-country) have not yet been observed; the document is written now so that when they are, PRSM has a ratification-ready policy rather than an ad-hoc response.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder, pending Foundation convocation
**Governs:** Research track R4 from `docs/2026-04-14-phase4plus-research-track.md` §R4 — promoted here from placeholder to standard draft.
**Related documents:**
- `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md` — parent charter; this standard ratifies under §9.
- `docs/2026-04-14-phase4plus-research-track.md` §R4 — original research stub.
- `PRSM_Vision.md` §6 "Honest caveats" — source of the concern this standard addresses.
- `prsm/marketplace/orchestrator.py` — implementation target for §6.1 enforcement (specifically `_select_top_k` and `EligibilityFilter`).

---

## 1. Purpose

PRSM's four-tier supply architecture (T1 consumer edge, T2 prosumer, T3 cloud arbitrage, T4 meganode) is designed to deliver frontier-adjacent inference latency on day 1 via T3 while the lower-cost T1/T2 tiers grow into meaningful supply share over time. The **launch UX thesis** (per `docs/2026-04-12-phase2-remote-compute-plan.md`) treats T3 concentration during bootstrap as acceptable precisely because the network needs competitive latency on its first production workloads.

What makes this acceptable on day 1 becomes a structural risk if sustained. Three specific failure modes:

1. **Hyperscaler-collapse risk.** If T3 exceeds 70-80% of supply long-term and is itself concentrated across 3-4 hyperscalers (AWS, GCP, Azure, Oracle), PRSM effectively runs on those hyperscalers' infrastructure. A single hyperscaler suspending PRSM workloads (capacity reclaim, policy change, AUP violation) could take a meaningful fraction of network capacity offline simultaneously. PRSM's decentralization claim becomes rhetorically thin.
2. **Geographic-concentration risk.** Single-country supply dominance exposes the network to that country's regulatory or political action. The 40% country threshold below is set conservatively because PRSM's promise to regulated-industry users includes jurisdictional diversity.
3. **Upward-price-pressure lock-in.** If T3 concentration drives pricing toward what any one hyperscaler charges its own AI workloads, PRSM loses its cost-arbitrage margin and the economic case for participation weakens.

This standard specifies **(a)** how PRSM measures supply concentration, **(b)** what thresholds trigger intervention, and **(c)** what intervention mechanisms exist, without destroying T3 economics during bootstrap.

---

## 2. Scope

### 2.1 Governs

- Measurement of supply concentration across providers and geographies.
- Orchestrator-layer diversity-aware scoring adjustments.
- Foundation-funded diversity-bonus payment mechanism.
- Triggers and escalation paths from "monitoring only" through "governance intervention."
- Data-collection requirements placed on operators at listing time.

### 2.2 Does not govern

- **Demand-side diversity.** Requesters are free to concentrate or diversify their dispatch patterns as they see fit. Only supply-side concentration is a protocol-governance concern.
- **Per-tier concentration within T1/T2.** Consumer-edge and prosumer nodes naturally diversify by virtue of being individually owned; a concentration policy at that tier would over-constrain bootstrap growth.
- **Specific cloud-provider prohibitions.** This standard is mechanism-neutral. It does not blacklist any provider. A hyperscaler's share rising above the cap simply triggers the soft-cap mechanism, not an eviction.
- **FTNS token-holder concentration.** Governed separately by PRSM-TOK-1.
- **Bootstrap-phase emergency exceptions.** See §9 for the grace period during which this standard's mechanisms run in monitoring-only mode.

### 2.3 Deferred to later standards

- **Algorithmic anti-sybil.** Operators willing to appear as multiple distinct providers to evade caps are a real concern. Deterrence mechanism (stake-based identity binding; see §8.3) is specified here at high level; full sybil-resistance protocol is a follow-up standard (PRSM-SUPPLY-2 candidate).
- **Compute-class weighting.** Not all "capacity units" are equivalent — a H100 at 3 TB/s HBM is not interchangeable with a consumer RTX 4090. Weighting schemes are deferred until monitoring data shows whether naive TFLOPS weighting over-indexes on raw compute vs effective usable capacity.

---

## 3. Definitions

### 3.1 Provider

The **corporate entity** providing the underlying hardware on which a PRSM node runs, as distinct from the operator running the node.

- For T3 cloud arbitrage: the cloud provider (AWS, GCP, Azure, Oracle, DigitalOcean, Hetzner, …).
- For T4 meganodes: the facility operator (PRSM-registered legal entity).
- For T1/T2: the operator themselves — the individual or small business.
- For co-located nodes (e.g., an operator's hardware in a colo datacenter): the colo facility operator is the provider.

Operators self-declare the provider at listing time via a new `provider_id` field on `ProviderListing`. The Foundation publishes a canonical provider registry keyed on the declared identity; unrecognized providers are treated as "Other (unverified)" and contribute to a bucket tracked separately.

Detection against self-declaration is out of scope for this standard but is tracked as a sybil-resistance follow-up (see §2.3).

### 3.2 Geographic region

The **country** of the node's operating jurisdiction, declared by the operator at listing time.

- Country is the unit for the 40% trigger in §5.2. Finer grains (state/province, datacenter region) are collected but not capped.
- Country is determined by where the operator is incorporated (for T3/T4) or resides (for T1/T2), not by IP geolocation. IP-based detection is easy to spoof and penalizes operators using VPNs for legitimate reasons.
- Multi-country operators (e.g., a T4 meganode operator with facilities in two countries) declare per-facility.

### 3.3 Capacity unit

The measured contribution of a provider / region to network supply. Initial unit:

**Available shard-second-equivalents over a rolling 7-day window.**

A shard-second-equivalent is: one shard dispatched on one provider's hardware for one second of wall time, normalized to a reference workload (Llama-3.1-8B forward pass at 4k context, fp16). Each provider's share is `their_shard_seconds / network_shard_seconds`.

Why this unit rather than TFLOPS or #nodes:
- **#nodes** over-indexes on T1/T2 (many small nodes) vs T3/T4 (few large nodes) in the direction that makes us *less* concerned about concentration, which inverts the policy intent.
- **TFLOPS** over-indexes on raw compute and under-counts bandwidth-bound workloads. PRSM inference is frequently bandwidth-bound under sharded execution (see Phase 2 Rings 7-10).
- **Realized dispatch work** captures what the network actually relies on, and is measurable from existing `ShardExecutionReceipt` events without new instrumentation.

### 3.4 Share

`Share(X) = CapacityUnit(X) / CapacityUnit(network_total)` over a rolling 7-day window, evaluated at midnight UTC daily.

Intermediate values are averaged over the window rather than computed on the instantaneous snapshot; a single-day anomaly (one meganode rebooting for maintenance) should not trigger policy action.

---

## 4. Measurement

### 4.1 Primary metric

`Share(provider_i)` and `Share(region_j)` as defined in §3.4.

### 4.2 Secondary metric: Herfindahl-Hirschman Index (HHI)

`HHI_providers = Σ_i (Share(provider_i) × 100)²`
`HHI_regions   = Σ_j (Share(region_j)   × 100)²`

HHI ranges from near 0 (perfectly diffuse) to 10,000 (single entity dominates). PRSM targets:

- `HHI_providers < 2500` (DoJ Merger Guidelines' "moderately concentrated" boundary).
- `HHI_regions   < 3000` (loosened vs provider HHI because many fewer countries than providers → natural HHI floor).

HHI trends matter more than any instantaneous value — a rising HHI with no single entity crossing 30% may still warrant attention.

### 4.3 Monitoring cadence

- **Daily:** share metrics, HHI metrics, 7-day moving averages published to an internal Foundation dashboard.
- **Weekly:** public disclosure of top-5 providers by share, top-5 countries by share, HHI indices. Published to the Foundation's transparency page.
- **Quarterly:** Foundation Quarterly Disclosure (PRSM-GOV-1 §11.2) includes a dedicated supply-concentration section with trend charts.
- **Continuous:** alert thresholds (see §5) fire in real-time to the Foundation's on-call rotation.

### 4.4 Data collection

Operators provide at listing time:

- `provider_id` — from the Foundation's canonical provider registry (or "Other").
- `region_country` — ISO 3166-1 alpha-2 code.
- `region_subdivision` — ISO 3166-2 subdivision (optional, for analytics only).

Listings missing either field are counted in an "undeclared" bucket that is capped at 5% of network share. Above 5%, undeclared listings are soft-penalized (see §6.1) until they declare.

---

## 5. Triggers

Thresholds set conservatively; once a trigger fires, the escalation path in §6 runs through its stages in sequence.

### 5.1 Informational triggers (monitoring intensifies; no enforcement action)

- Any single provider `Share` exceeds 20% for 14 consecutive days.
- Any single country `Share` exceeds 30% for 14 consecutive days.
- `HHI_providers` exceeds 2000 for 14 consecutive days.
- `HHI_regions` exceeds 2500 for 14 consecutive days.

**Foundation action:** alert, investigate, publish finding in next Quarterly Disclosure. No orchestrator-layer change.

### 5.2 Intervention triggers (enforcement mechanisms activate)

- Any single provider `Share` exceeds **30%** for 30 consecutive days.
- Any single country `Share` exceeds **40%** for 30 consecutive days.
- `HHI_providers` exceeds **2500** for 30 consecutive days.
- `HHI_regions` exceeds **3000** for 30 consecutive days.

**Foundation action:** activate §6.1 orchestrator-layer soft-cap scoring AND §6.2 diversity bonus. Publish activation in an immediate public notice. Required Quarterly Disclosure section on effectiveness.

### 5.3 Emergency triggers (governance convocation)

- Any single provider `Share` exceeds **50%** on any single measurement day.
- A single provider publicly announces suspension or restriction of PRSM workloads.
- A jurisdictional action (court order, regulatory mandate) affects provider(s) representing >25% of network share.

**Foundation action:** emergency Board convocation under PRSM-GOV-1 §9.4. Remediation mechanisms beyond §6 (hard caps, urgent solicitation of T1/T2 supply via incentive auction) are considered. Standard amendment may be proposed.

### 5.4 Trigger-level "operator concern" escalation

Any operator with ≥1000 successful dispatches may file a supply-concentration concern with the Foundation, backed by specific data. The Foundation evaluates the concern within 30 days and either:

- Confirms an existing trigger has fired (proceed to §6).
- Proposes an intermediate informational action (dashboard, disclosure addition).
- Declines the concern with published rationale.

This is the mechanism the R4 research-track stub called out as "An operator raises a governance concern backed by data."

---

## 6. Enforcement mechanisms

### 6.1 Orchestrator-layer soft-cap scoring

Implementation target: `prsm/marketplace/orchestrator.py` `_select_top_k` helper, plus a new diversity-aware term in `EligibilityFilter`.

Current `_select_top_k` ranks candidates by `tier_ordinal × (1 / price)` with lexicographic provider_id tiebreak. When a §5.2 trigger is active, the score gains a diversity multiplier:

```
score(listing) = tier_ordinal × (1 / price) × diversity_multiplier(listing)

diversity_multiplier(listing) =
    overconcentration_penalty(provider_share, region_share)
```

Where `overconcentration_penalty` is 1.0 (no penalty) for listings from providers/regions below the §5.2 threshold, and declines smoothly to 0.25 for listings at 50%+ share. Specific shape:

```
penalty(s) = 1.0                               if s ≤ threshold
           = 1.0 - 0.75 × (s - threshold) / (0.5 - threshold)   if threshold < s ≤ 0.5
           = 0.25                              if s > 0.5
```

Where `threshold` is the relevant §5.2 trigger (0.30 for provider, 0.40 for country). Linear interpolation between threshold and the 50% floor.

**Why soft-cap, not hard-cap:** a hard cap that refuses routing above X% share can degrade latency during peak demand if T1/T2 supply hasn't grown to fill the gap. Soft caps preserve routing availability while making diverse providers more attractive at the margin.

**Why 0.25 floor (not 0):** dropping overconcentrated providers to zero score would exclude them entirely once the cap is hit, equivalent to a hard cap. The 0.25 floor preserves them as fallback capacity for cases where diverse alternatives are unavailable.

### 6.2 Foundation diversity-bonus payment

Implementation target: a new Foundation-operated contract (future deliverable) that reads the published weekly share metrics and pays bonus FTNS to under-represented providers/regions.

Mechanics:

- **Eligibility:** any provider or region whose share is below `(max_share / 3)` where `max_share` is the currently largest provider/region share.
- **Bonus amount:** ≤ 10% of the eligible listing's gross FTNS earned during the billing period, paid from the Foundation reserve.
- **Cap:** total monthly bonus payout ≤ $X (TBD via Foundation budget process; see §7.3).
- **Self-extinguishing:** as diversity improves (gap between the top and the underrepresented narrows), fewer listings qualify for the bonus and total payout declines toward zero.

**Why bonuses, not penalties:** penalty-only mechanisms create operator flight risk during the correction period. Bonus-only mechanisms cost the Foundation reserve but preserve operator goodwill. §6.1 and §6.2 together combine carrot (bonus) with light stick (soft-cap score).

### 6.3 Required operator disclosure (non-breaking)

When a §5.2 trigger is active, the Foundation publishes a weekly concentration report naming the specific providers/regions triggering the policy. No operator is named as non-compliant — participating above a share threshold is not non-compliance — but transparency lets operators make informed routing decisions.

### 6.4 Fallback: governance-directed hard cap

If §6.1 + §6.2 do not reduce the triggering metric below the §5.2 threshold within 180 days of activation, the Foundation Board may vote (simple majority under PRSM-GOV-1 §9.2) to activate a hard cap at the triggering level. Hard-cap activation:

- Has a stated sunset date (≤ 180 days).
- Requires a Foundation statement on the expected harm-reduction vs the soft-cap status quo.
- May be extended only by a separate vote with a new 180-day sunset.

Hard caps are intended as a ratchet, not a permanent mode.

---

## 7. Parameter values (initial)

This section specifies the **initial values** for all parameters introduced above. Parameters are adjustable via PRSM-GOV-1 §9.2 standards amendment (simple majority); emergency adjustment under §9.4 is allowed.

### 7.1 Share thresholds

| Metric | Informational | Intervention | Emergency |
|---|---|---|---|
| Single provider share | 20% | 30% | 50% |
| Single country share  | 30% | 40% | — |
| HHI providers         | 2000 | 2500 | — |
| HHI regions           | 2500 | 3000 | — |

### 7.2 Measurement window

- Concentration metric averaging: **7-day rolling window.**
- Informational trigger persistence: **14 consecutive days.**
- Intervention trigger persistence: **30 consecutive days.**

### 7.3 Diversity bonus

- Bonus rate: **≤ 10%** of qualifying listing's gross FTNS.
- Eligibility threshold: qualifying listings must have `share ≤ (max_share / 3)`.
- Monthly cap: **$50,000 USD-equivalent** of Foundation reserve per month (initial). This is a loose placeholder; the actual value should be set by the Foundation treasury as part of the annual budget under PRSM-GOV-1 §4.6, weighing it against R&D spend and grant programs.
- Payout cadence: **monthly**, based on the prior month's share metrics.

### 7.4 Soft-cap penalty curve

- Floor: **0.25** (minimum diversity multiplier).
- Start of penalty: exactly at the §5.2 intervention threshold.
- Full-floor point: at 50% share.
- Interpolation: **linear** (§6.1 formula).

### 7.5 Hard-cap parameters (if ever activated)

- Activation vote: simple majority (§6.4).
- Sunset: ≤ 180 days per activation.
- Extension: separate vote, new sunset.

---

## 8. Appeals and exceptions

### 8.1 Provider misclassification appeal

If an operator believes their listing has been misclassified (wrong `provider_id`, wrong region), they may file an appeal via the Foundation's standard operator-support channel. Appeals are decided within **14 days** with published rationale.

### 8.2 Share-measurement dispute

If an operator disputes the published share for their provider or region, they may request a recomputation from the underlying shard-execution-receipt data, audited by the Foundation's designated independent auditor (PRSM-GOV-1 §11.4).

### 8.3 Stake-binding to prevent provider sybil

Evading caps via multiple provider identities is a real risk. The deterrent specified here:

- Listing under a `provider_id` requires that the operator's on-chain `StakeBond` bond amount meet or exceed a provider-specific floor, scaled to the tier they claim.
- If two listings with different `provider_id`s are traced back to the same underlying corporate entity, the Foundation may — upon evidence — consolidate them for share computation.
- Full sybil-resistance protocol (e.g., attested-operator-identity binding) is deferred to PRSM-SUPPLY-2.

### 8.4 T1/T2 aggregation

Individual T1/T2 operators should not be individually named in share reports (privacy concern). They are aggregated into a `T1/T2 collective` bucket for public reporting, but their shares sum toward share metrics at their declared `region_country`.

---

## 9. Bootstrap-phase grace period

From the date PRSM-SUPPLY-1 is ratified until the date PRSM's network supply exceeds **5,000 providers** OR **36 months after ratification** (whichever comes first), the standard operates in **monitoring-only mode**:

- Measurement (§4) runs.
- Public disclosures (§4.3) publish.
- §5.1 informational triggers may fire; Foundation investigates and publishes findings.
- §5.2 intervention triggers DO NOT activate §6.1 soft-cap or §6.2 diversity bonus.
- §5.3 emergency triggers DO activate the Board convocation path.

**Rationale:** during bootstrap, T3 concentration is part of the launch UX thesis — diverse supply hasn't had time to grow. Prematurely activating soft-caps would degrade the bootstrap-period latency story. Monitoring runs throughout so that when the grace period ends, the Foundation has a data-rich baseline for activation decisions.

The monitoring-only mode sunsets automatically when a grace-period exit condition is met. An active §5.3 emergency trigger can override this grace period at any time.

---

## 10. Implementation responsibilities

### 10.1 Foundation

- Maintains canonical provider registry (§3.1).
- Publishes measurement dashboard (§4.3).
- Runs trigger monitoring and convenes on §5.3 emergencies.
- Operates diversity-bonus contract (§6.2).
- Funds the diversity-bonus pool via annual budget (§7.3).
- Handles appeals under §8.

### 10.2 Prismatica (or any T4 meganode operator)

- Self-declares `provider_id` and `region_country` at listing time (§4.4).
- Complies with soft-cap scoring without circumvention attempts (§6.1).
- Reports honestly on its own share trend in any Foundation disclosure.

### 10.3 Engineering

- Extend `ProviderListing` (per `prsm/marketplace/listing.py`) with `provider_id` and `region_country` fields.
- Extend `EligibilityFilter` (per `prsm/marketplace/filter.py`) to read the diversity context.
- Extend `MarketplaceOrchestrator._select_top_k` (per `prsm/marketplace/orchestrator.py`) with the §6.1 `diversity_multiplier` term. Gate behind a policy flag that reads the Foundation's current activation state (TBD: via an on-chain registry or off-chain config signal).
- Emit share-contribution metrics from `ShardExecutionReceipt` into a queryable data store (Foundation-operated; design TBD).

### 10.4 Operators

- Declare truthfully at listing time.
- Cooperate with share-measurement audits under §8.2.
- Bond stake appropriately per §8.3.

---

## 11. Open issues

### 11.1 Canonical provider registry seed list

The §3.1 registry needs an initial populated list. Proposed initial entries:

- Hyperscalers: AWS, GCP, Azure, Oracle, IBM Cloud.
- Tier-2 clouds: Hetzner, DigitalOcean, OVH, Vultr, Linode, Fly.io, Lambda, RunPod, CoreWeave, Crusoe.
- Meganode operators: Prismatica (initial); populated as new operators onboard.
- "Other (unverified)" catch-all.

Foundation should publish this list as a versioned registry file and update via standards amendment when new significant providers emerge.

### 11.2 Stake-binding floor values

§8.3 requires a per-tier stake floor for each `provider_id` listing. Default proposal:

- T3 listings: floor = 5,000 FTNS (matches standard tier min, per `prsm/economy/web3/stake_manager.py` `TIER_STANDARD_MIN_WEI`).
- T4 meganode listings: floor = 50,000 FTNS (matches critical tier min).
- T1/T2: no additional floor beyond normal staking.

These numbers are tentative; PRSM-TOK-1 owns FTNS-denominated value judgments and may adjust.

### 11.3 Monthly USD cap on diversity bonus

$50k/month is a placeholder (§7.3). The Foundation's annual budget process (PRSM-GOV-1 §4.6) should set this figure in the context of:

- Total diversity bonus expected to flow (depends on network size).
- R&D spend, grant programs, reserve maintenance.
- Worst-case scenario: §5.2 triggers fire and bonus payout rises to the cap.

### 11.4 Enforcement integration with on-chain governance

§10.3 notes the orchestrator needs to read the Foundation's current activation state for each trigger. Candidate mechanisms:

- **On-chain flag contract:** a Foundation-owned contract exposing `isTriggerActive(triggerId)` reads. Orchestrators read on each dispatch.
- **Off-chain signed config:** Foundation publishes a signed JSON blob with current trigger state; orchestrators cache with TTL.
- **Hybrid:** on-chain flag as source of truth, off-chain mirror for low-latency reads.

TBD; will land in the engineering design doc that promotes this standard to implementation.

### 11.5 Grace-period exit condition calibration

§9's "5,000 providers OR 36 months" is a placeholder. The 5,000-provider count is motivated by "enough T1/T2 emergence that soft-capping hyperscalers doesn't catastrophically degrade supply"; the 36-month clock ensures the grace period doesn't stretch indefinitely if supply growth stalls. Both figures are subject to Foundation adjustment as monitoring data accumulates.

### 11.6 Interaction with ECON-WP-1 economic model

PRSM-ECON-WP-1 (economic model white paper) specifies overall network economics. This standard's diversity bonus (§6.2) introduces a Foundation-reserve-funded payment stream that may modify certain ECON-WP projections — specifically T1/T2 effective pricing once bonuses are factored in. ECON-WP v2 should incorporate this.

---

## 12. Ratification path

Per PRSM-GOV-1 §9.2:

1. **Discussion.** Public comment period via the Foundation's standards forum. Target 30 days.
2. **Revision.** Drafter incorporates comments; publishes revised version.
3. **Ratification vote.** Foundation Board vote; simple majority (this is an amendable standard, not a charter amendment).
4. **Activation.** Standard takes effect on ratification date but immediately enters §9 grace period.

Target ratification: **Q3 2026**, pending Foundation formation (PRSM-GOV-1 §5 open issues).

---

## 13. Amendment

Per PRSM-GOV-1 §9.2, §9.4 (emergency). Specific amendments that require additional process beyond the default:

- **Changes to grace-period exit conditions (§9).** Require explicit Board rationale referencing current monitoring data; cannot be extended indefinitely without new data justifying each extension.
- **Lowering share thresholds (§7.1) below 20% / 30%.** Requires supermajority (2/3) to prevent runaway constraint creep.
- **Removing soft-cap floor (§7.4).** Requires supermajority — a zero floor is a hard cap and is intentionally the ratchet-of-last-resort option via §6.4.

---

## 14. Relationship to R4 research stub

This standard promotes `docs/2026-04-14-phase4plus-research-track.md` §R4 from research placeholder to governance-ready specification. R4's triggers ("any single provider exceeds 30%", "any single country exceeds 40%") are preserved and integrated into §5 with a richer escalation ladder. R4's "operator raises governance concern backed by data" path is specified in §5.4.

The R4 stub said effort was "small — mostly governance-design work and parameter tuning once monitoring exists." This document IS that governance-design work. The "parameter tuning once monitoring exists" is now §7's explicit parameter block plus the §11 open issues that calibrate during the §9 grace period.

**Research-track disposition:** R4 is promoted from "research stub" to "governance standard draft." On ratification, R4 can be marked ✅ resolved in the Phase 4+ track doc, with a cross-reference to this standard.

---

## 15. Changelog

- **0.1 (2026-04-22):** initial draft, founder-authored. Promoted from R4 research stub. Pending Foundation convocation for ratification.
