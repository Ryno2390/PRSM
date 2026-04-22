# Phase 6 Task 4 — DHT Parameter Tuning Plan

**Document identifier:** PHASE6-TASK4-DHT-TUNING-1
**Version:** 0.1 Draft
**Status:** Partner-handoff-ready scoping doc. Not an execution plan yet — execution requires a Foundation-operated testnet and ~1 engineering-week of measurement time, both of which follow Phase 6 engineering merge and Foundation-ops bootstrap-node infrastructure (Task 2). This document pins the parameter space, methodology, pass/fail gates, and final-output shape so the measurement can proceed with minimum design overhead when resources are ready.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Related documents:**
- `docs/2026-04-22-phase6-p2p-hardening-design-plan.md` §3.3 + §6 Task 4 — origin of this plan; defines the parameter set to sweep.
- `docs/2026-04-22-r7-benchmark-plan.md` (R7-BENCH-1) — structural template for this doc (preregistered hypotheses + phased execution + promotion triggers).
- `tests/chaos/harness.py` — in-process chaos simulator that predates the real testnet; continues to catch regressions in liveness / rate-limit state machines at CI scale.
- `prsm/node/bootstrap.py`, `prsm/node/liveness.py`, `prsm/node/rate_limit.py`, `prsm/node/nat_traversal.py` — Phase 6 Tasks 1 + 3 + 5 (SHIPPED), the primitives whose default configuration this measurement finalises.

---

## 1. Purpose

Phase 6 Task 4 is a **measurement task**, not an engineering task. Its deliverable is a single tuned parameter set committed to `prsm/node/libp2p_*` defaults, backed by testnet data rather than default-library guesses. Phase 6 design doc §3.3 specifies the candidate parameter space with initial guesses; this document provides the methodology to validate or overturn those guesses empirically.

Scope-wise, this sits adjacent to R7 (KV/activation compression benchmarking): both are parameter-sweep / best-value-selection exercises against a preregistered hypothesis set. R7 measures a compression scheme; this measures a DHT configuration. Same shape, different subject.

**What this doc is:** a preregistered measurement plan with hypothesis gates, execution phases, and a final-output spec — handed to whoever runs the testnet. Observable parameters + pass/fail thresholds are fixed here BEFORE the measurement starts; post-hoc tuning against the observed data is explicitly disallowed.

**What this doc is NOT:**

- Not an engineering-tasks design doc. Code changes arising from the measurement (updated defaults, any new metric the measurement surfaces) are separate follow-up tasks.
- Not a full testnet-ops runbook. Cluster provisioning, node-identity generation, gossip bootstrap, monitoring dashboards are Phase 6 Task 2 territory.
- Not a permanent tuning. The tuned parameters are valid for the network-size / churn / geography regime measured. A re-run is expected at the next major inflection (e.g., ≥1000 nodes, introduction of mobile operators, etc.).

---

## 2. Non-goals

- **Not re-validating libp2p itself.** libp2p's Kademlia implementation is assumed correct; we are tuning, not debugging.
- **Not introducing new DHT parameters.** The sweep is limited to the five knobs in §4.1. New knobs (e.g., adaptive replication) are post-Phase-6 scope.
- **Not optimizing for worst-case adversarial scenarios.** That's Task 7 chaos testing. This tunes the NORMAL-operation envelope. Defaults must work for honest networks; the rate-limit + liveness + NAT primitives handle adversarial protection independently.
- **Not tuning across deployment topologies.** A single global network is measured. If PRSM later deploys per-region DHTs, each gets its own measurement.

---

## 3. Why this needs a preregistered plan

DHT measurement is well-known for three failure modes that preregistration avoids:

1. **P-hacking.** Sweep 5 parameters × 3-5 values each = 243-3125 configurations. Some config WILL win on any one metric by chance. Pre-committing the primary/secondary metrics and their pass/fail thresholds eliminates "the best config is whatever we test last."
2. **Goodhart's Law on success rate.** Success-rate measurements without latency constraints converge on "pick the largest k and longest timeout." We preregister the composite metric to avoid this.
3. **Size-of-network confound.** Results at 50 nodes don't generalize to 500. The measurement runs at two network sizes to surface scale sensitivity before committing defaults.

---

## 4. Preregistered hypotheses

Each hypothesis has a specific pass / fail / null condition keyed off the observed data. All thresholds are set here, before the measurement runs.

### H1: Replication factor `k = 10` is sufficient at 50-100 nodes

**Current default:** `k = 20` (libp2p standard). Phase 6 §3.3 conjecture: tune to `k = 10`.

**Measurement:** sweep `k ∈ {6, 10, 14, 20}`. For each, run N=5000 PUT+GET pairs across the cluster with 5% churn per hour. Report lookup success rate + p95 lookup latency.

**Pass:** `k = 10` achieves ≥99% lookup success AND p95 latency within 15% of `k = 20` baseline.

**Fail:** `k = 10` drops below 99% success OR p95 latency > 1.15× `k = 20`. Fallback to `k = 14` or `k = 20` as data dictates.

**Null:** `k ∈ {10, 14, 20}` all pass within tolerance → prefer `k = 10` for storage-overhead minimization (each entry replicated to k nodes). Document the null outcome explicitly.

### H2: Query timeout 10 s is sufficient; 30 s is over-budgeted

**Current default:** `query_timeout = 30 s` (libp2p standard). Phase 6 §3.3 conjecture: tune to 10 s.

**Measurement:** at selected-`k` from H1, sweep `timeout ∈ {5, 10, 15, 30} s`. Measure success rate + p95 latency. Cross-reference with NAT-traversal-success rate from the p2p_metrics module to check whether symmetric-NAT peers disproportionately fail at shorter timeouts.

**Pass:** `timeout = 10 s` achieves ≥99% success AND p95 latency ≤ 10 s (trivially) AND NAT-symmetric-peer success rate ≥ 95%.

**Fail:** symmetric-NAT subset below 95% at timeout 10 s. Fallback to 15 s.

**Null:** both 10 s and 15 s pass → prefer 10 s for faster timeout-to-retry on legitimately-failed lookups.

### H3: Bucket size 20 is correct; no tuning needed

**Current default:** `bucket_size = 20`. Phase 6 §3.3 marks this "keep."

**Measurement:** sweep `bucket_size ∈ {10, 20, 40}` at the selected `k + timeout`. Lookup success + p95 + routing-table-size observability.

**Pass:** `bucket_size = 20` matches or beats both extremes on the composite metric (success × (1 / p95)).

**Fail:** either extreme wins. Report and retune.

**Null:** all within tolerance → keep 20 (default minimizes stored state without obvious penalty).

### H4: Alpha = 3 is correct

**Current default:** `alpha = 3` (concurrent lookup queries). Phase 6 §3.3 marks this "keep."

**Measurement:** sweep `alpha ∈ {1, 3, 5}` at selected config. p95 latency + bandwidth consumption per lookup.

**Pass:** `alpha = 3` has p95 latency within 20% of `alpha = 5`, AND bandwidth-per-lookup at most 50% of `alpha = 5`.

**Fail:** `alpha = 5` wins on both latency and bandwidth (unlikely — more concurrency usually costs more bandwidth). Investigate for cluster-side interaction effects.

**Null:** both `alpha = 3` and `alpha = 5` within tolerance → keep 3 for bandwidth discipline.

### H5: Republish interval 22 h is adequate at low churn

**Current default:** `republish_interval = 22 h`. Phase 6 §3.3 marks this "keep."

**Measurement:** observe 24-hour entry-retention rate at 5% hourly churn. Report fraction of entries still retrievable 24 h after initial PUT.

**Pass:** ≥95% 24-hour retention.

**Fail:** below 95%. Tune to 12 h or 8 h.

**Null:** passes cleanly → keep 22 h for storage-traffic minimization.

---

## 5. Network conditions to measure

To surface scale sensitivity, the sweep runs at TWO cluster sizes. All other conditions fixed.

| Axis | Small cluster | Medium cluster |
|------|---------------|----------------|
| Node count | 50 | 100 |
| Churn rate | 5% / hour (legitimate peers) | 5% / hour |
| Geographic regions | 4 (us-east, us-west, eu-west, ap-south) | 4 |
| NAT distribution | 40% cone, 30% restricted, 20% port-restricted, 10% symmetric (see §5.1) | Same |
| Test duration | 4 hours steady-state (after 30-min warmup) | 4 hours steady-state |
| PUT + GET pairs | 5,000 | 10,000 |

The 50-node and 100-node runs test H1-H5 independently. If a hypothesis passes at 50 nodes but fails at 100, the selected parameter is SIZE-DEPENDENT and needs a second measurement at any future production scale above 100.

### 5.1 NAT distribution synthesis

Real-network NAT-type distributions vary by deployment. The 40/30/20/10 split above is a conservative synthesis from public residential-ISP studies; it is not empirically grounded in PRSM's future operator base. A follow-up measurement at real deployment scale (Phase 6 post-mainnet) should refresh this distribution.

---

## 6. Methodology

### 6.1 Controlled variables

- **libp2p version:** pinned to whatever ships with Phase 6 Task 1-7 merge-ready. Re-running this measurement on a future libp2p version invalidates the parameter set.
- **Geographic latency simulation:** inter-region RTT added via `tc qdisc` (Linux traffic control): us-east↔us-west 60 ms, us-east↔eu-west 80 ms, us-east↔ap-south 180 ms. Symmetric.
- **Bootstrap topology:** each node connects to 3 bootstrap peers per Phase 6 Task 1 design. Bootstrap peers placed in distinct regions.
- **Content distribution:** synthetic 256-byte values with random 32-byte keys. No clustering — each (key, value) is uniformly random.

### 6.2 Measurement protocol

For each (cluster-size, config) pair:

1. **Provision cluster.** 30-minute warm-up: all nodes join, DHT converges. Verified by bootstrap-peer-count + p2p_metrics / `p2p_connected_peers` gauge stabilizing.
2. **Initial PUT phase.** 30 min: distribute 5,000 or 10,000 (key, value) pairs across random-source nodes.
3. **Steady-state GET phase.** 4 hours: random-source nodes issue GETs against known keys at 1 req/sec/node. Log success + per-request latency histograms.
4. **Churn injection.** Throughout GET phase: every 12 min, randomly drop 5% of legitimate nodes (not bootstrap), replace within 2 min with fresh nodes.
5. **Metrics collection.** Per-minute p2p_metrics snapshot from every node. Stored as a single time-series for offline analysis.

### 6.3 Reported metrics

Primary:
- **lookup_success_rate** — GET returning any value (correct or not) within the timeout.
- **p95_lookup_latency_ms** — 95th-percentile wall-clock time from GET-issue to value-received (or timeout).

Secondary:
- **storage_overhead_ratio** — total bytes stored across cluster / bytes of content (expected ≈ k).
- **bandwidth_per_lookup_kb** — aggregate wire-bytes / lookup count.
- **nat_symmetric_subset_success** — H2 cross-reference.
- **24h_retention_rate** — H5.

Composite:
- **composite_score** = `lookup_success_rate × (1000 / (p95_lookup_latency_ms + 1))`. Used for ordering configs within a hypothesis test; NOT used as a pass/fail gate on its own.

---

## 7. Phased execution plan

### Phase 1 — Preparation (1 week)

- Foundation-ops provisions a Terraform-managed 100-node testnet cluster per Phase 6 Task 2 deploy playbook. Confirms: bootstrap reachability, `tc qdisc` latency injection works, NAT-type simulation verified via STUN probes.
- Instrumentation: each node emits Prometheus metrics; central scraper collects + persists to a time-series DB.
- Dry-run with libp2p defaults confirms the methodology pipeline end-to-end.

**Gate to Phase 2:** dry-run measurement matches libp2p's published baselines within 10%.

### Phase 2 — H1 (replication factor) sweep (1 week)

- Sweep `k ∈ {6, 10, 14, 20}` at 50 nodes, then at 100 nodes.
- 8 configs × 4 hours each = 32 hours of steady-state measurement.
- Report H1 pass/fail/null per §4.1.

**Gate to Phase 3:** H1 outcome selected + documented. This selection fixes `k` for subsequent hypothesis tests.

### Phase 3 — H2 (timeout) + H3 (bucket size) + H4 (alpha) (1 week)

- Hold `k` fixed at H1-selected. Run H2 + H3 + H4 sweeps in parallel (independent).
- 4 timeout configs × 2 sizes + 3 bucket configs × 2 sizes + 3 alpha configs × 2 sizes = 20 runs × 4 hours = 80 hours.

**Gate to Phase 4:** H2 + H3 + H4 outcomes selected.

### Phase 4 — H5 (republish interval) + final validation (1 week)

- H5 runs at the final-selected (k, timeout, bucket, alpha) config. Single 24+ hour run per cluster size.
- Final validation: re-run the full composite at the final config. Must still match Phase 2-3 individual-hypothesis results within tolerance — rules out parameter-interaction surprises.

**Gate to publication:** final config passes all five hypothesis gates.

### Phase 5 — Commit + document (1 week)

- Tuned parameters land as new defaults in `prsm/node/libp2p_*`.
- Final measurement report published as `docs/2026-MM-DD-phase6-task4-results.md` with raw-data archive reference.
- Review-cadence reminder set for the next major network-size inflection.

**Total calendar: 5 weeks.** Test-mode engineering effort: ~1 FTE for the duration (Foundation Site Reliability / DevRel role).

---

## 8. Pass / fail / null semantics

For clarity on what each hypothesis outcome means operationally:

| Outcome | Action |
|---------|--------|
| Pass | Commit the hypothesis's preferred value as default. |
| Fail | Use the next fallback in the hypothesis's `Fail:` line. If no fallback listed, re-run with finer parameter grid around observed best. |
| Null | Multiple values pass; prefer the one minimizing secondary cost (storage / bandwidth / latency, in that order). Document null outcome publicly. |

**Explicit anti-patterns the preregistration closes:**

- "Just pick the best observed value" — requires falling back to the `Fail:` line if the preferred value didn't pass. Prevents post-hoc drift.
- "Re-run with more values" — permitted ONLY after a `Fail:` outcome, not to hunt for a better result after `Pass:`.
- "Average across cluster sizes" — sizes are reported separately; no averaging. If they disagree, the hypothesis is size-dependent and must be documented as such.

---

## 9. Promotion triggers (the "why fund this now" test)

Task 4 is a Foundation-ops dependency, not an engineering-track dependency. It moves from "scoped" to "funded for execution" when **all** of:

### T1: Phase 6 Tasks 1 + 3 + 5 + 6 + 7 shipped

All the state machines this measurement tunes must be merge-ready. As of 2026-04-22 all five have shipped under the `phase6-merge-ready-20260422` tag.

### T2: Foundation bootstrap-node infrastructure live

Task 2 (Foundation-operated 3-region bootstrap cluster) must be operational. This measurement runs AGAINST the Foundation testnet bootstrap; measuring against an ad-hoc cluster produces results that don't apply to the production topology.

### T3: Foundation SRE / DevRel capacity

Requires ~1 FTE × 5 weeks of attention. Not a side-project effort — someone whose job is this measurement for that window.

### T4: Pre-mainnet-scale interest

The tuned defaults matter most when the network is at small-to-medium scale (50-500 nodes) where parameter sensitivity is highest. At tiny scale (≤20 nodes) any reasonable config works; at large scale (≥1000 nodes) the sensitivities shift enough to invalidate this measurement. The measurement targets the middle, where it does the most good.

**Operational reading:** Phase 6 Task 4 moves to execution when Foundation ops is live + staffing available + pre-mainnet testnet hitting 50+ nodes. Target window: Q3 2026 if on schedule; later if Phase 6 Task 2 slips.

---

## 10. Relationship to neighbouring work

- **R7 BENCH-1** (KV/activation compression benchmark) — structural template for this plan. Same preregistered-hypotheses + phased-execution pattern.
- **Phase 6 Task 2** (Foundation ops runbook) — prerequisite infrastructure.
- **Phase 6 Task 7** (chaos harness, SHIPPED) — complementary adversarial-conditions testing at CI scale. Task 4's measurements assume honest peers; Task 7 validates the state-machine primitives against adversarial peers. The two together cover the envelope.
- **PRSM-SUPPLY-1** (diversity standard) — the NAT distribution and geographic regions in §5 approximate what SUPPLY-1's diversity metrics will track in production. When SUPPLY-1 is live with real data, a re-run of this measurement should use the observed distribution instead of the synthesized one.

---

## 11. What a partner receives

Day 1 handoff package:

- This scoping doc (PHASE6-TASK4-DHT-TUNING-1).
- `docs/2026-04-22-phase6-p2p-hardening-design-plan.md` — the source Phase 6 plan.
- `docs/2026-04-22-r7-benchmark-plan.md` (R7-BENCH-1) — the structural template.
- Access to the Foundation testnet cluster (Task 2 runbook output).
- Terraform / Ansible templates for the measurement harness (§6.1 latency injection, NAT simulation).
- Prometheus / OTel metrics collector endpoints.

First-milestone deliverable: Phase 1 dry-run measurement matching libp2p baselines (§7 Phase 1 gate). 1 week.

---

## 12. Success criteria

Task 4 is considered successful if, at end of Phase 5:

1. **All five hypotheses resolved** — each with a Pass / Fail / Null outcome per §8.
2. **A tuned parameter set committed** to `prsm/node/libp2p_*` defaults, backed by data.
3. **A publishable measurement report** lands as `docs/2026-MM-DD-phase6-task4-results.md`. External reviewers can reproduce the methodology (data archive retained).
4. **No post-hoc parameter drift** — the committed defaults match the Phase 4 (final validation) outcome, not some later re-tune.

---

## 13. Risk register

### R1: Foundation testnet infrastructure slippage

Task 2 operational readiness gates this measurement. If Task 2 slips, so does Task 4. Mitigation: the chaos harness (Task 7, SHIPPED) continues to regress the liveness + rate-limit state machines at CI scale, so default-library Kademlia values remain reasonable stand-ins until the measurement completes.

### R2: libp2p version drift mid-measurement

A libp2p upgrade during the 5-week window invalidates partial results. Mitigation: version pin confirmed at Phase 1 start; no upgrades until Phase 5 completes.

### R3: Hypothesis-gate over-specification

If all five hypotheses fail at their preferred values, the default config lands closer to libp2p defaults than Phase 6 §3.3 conjectured. That's a legitimate outcome, not a failure — the measurement is the authority. The risk is treating "no changes from defaults" as a Phase-6-failed outcome; document clearly that null/fallback is expected behaviour.

### R4: 5-week measurement window conflicts with Phase 7 / 8 ramp

Task 4 runs 5 weeks of Foundation-ops attention. If Phase 7 / 8 execution competes for the same FTE, Task 4 must defer. Tuned-default absence is not a ship-blocker; libp2p defaults work adequately at small-to-medium scale.

### R5: Synthesized NAT distribution under-represents real deployment

§5.1 distribution is a conservative synthesis. If production operators skew heavily to a single NAT type (e.g., most use datacenter providers with public IPs), the measurement's implicit assumption about NAT-traversal cost is off. Mitigation: Phase 6 post-mainnet rerun using observed PRSM-SUPPLY-1 data.

---

## 14. Changelog

- **0.1 (2026-04-22):** initial scoping doc. Promotes Phase 6 Task 4 from design-doc bullet points to partner-handoff-ready measurement plan. Execution pending T1-T4 triggers per §9.
