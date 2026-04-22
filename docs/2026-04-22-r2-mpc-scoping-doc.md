# R2 Scoping Doc: MPC-for-Sharded-Inference under PRSM

**Document identifier:** R2-SCOPING-1
**Version:** 0.1 Draft
**Status:** Research scoping doc, not an execution plan. MPC for production-scale LLM inference is not solved at the state of the art; this document clarifies what solving it looks like *under PRSM's specific architecture* and what the first concrete milestone a partner org would target is.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Promotes:** `docs/2026-04-14-phase4plus-research-track.md` §R2 from stub to scoping doc.
**Related documents:**
- `docs/2026-04-14-phase4plus-research-track.md` §R2 — original research stub.
- `docs/2026-04-22-r7-benchmark-plan.md` — R7 benchmark plan. R2 composition with R7 specified in §7.
- `docs/2026-04-22-r3-threat-model.md` — R3 threat model. R2's collusion question uses R3's attacker-scale framework (S3 multi-shard coordinated attack).
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — supply-diversity standard. Non-collusion quantification (§5) reuses SUPPLY-1's diversity metrics.
- `PRSM_Vision.md` §7 "Honest limits" — vision-doc framing ("MPC is 10-100× overhead and requires non-colluding node assumptions").

---

## 1. Purpose

The R2 stub asked: *"Can MPC protocols be adapted to PRSM's tensor-sharded architecture such that non-colluding node assumptions are realistic under PRSM's tiered supply?"*

This document clarifies what "realistic" means in terms a research partner can act on:

1. **Decouple the two questions in the R2 stub** — the MPC-protocol question (which protocol, how does it compose with tensor-parallel sharding?) and the non-collusion question (how do we quantify that the assumption holds given PRSM's permissionless supply?). These require different expertise and probably different partner orgs.
2. **Frame MPC-on-sensitive-layers as the realistic first milestone.** Full-transformer MPC at frontier scale is unsolved; MPC on the first N transformer layers (where activation-inversion attacks are effective per R3) plus TEE-only on later layers is a scoped deliverable.
3. **Tie non-collusion quantification to PRSM-SUPPLY-1's diversity metrics.** The supply-diversity standard already specifies how PRSM measures operator / geographic / economic concentration. Those same metrics can parameterize a confidence interval on the non-collusion assumption.
4. **Specify the partner handoff.** R2 requires a cryptography-research partner (not general ML); this doc lists what we hand them day 1 and what first-milestone deliverable looks like.

R2 execution is substantially larger than R6 or R7. This doc is the scope; execution would be 2-4 research quarters depending on partner org capacity and prioritization.

---

## 2. Preregistered research questions

Unlike R7 (which has specific paper-replication hypotheses), R2 is an open design question at the state of the art. We preregister questions rather than hypotheses. Each question has a "how we'd answer it" specification so the partner team has a tractable first target.

### Q1: Which MPC protocol is most compatible with PRSM's pipeline-parallel SPRK architecture?

**How we'd answer:** comparison table scoring three candidate protocol families (ABY3 / SecureNN-line; MP-SPDZ general MPC; specialized transformer-MPC protocols like Iron / BumbleBee / Puma) against PRSM-specific criteria: composability with asynchronous SPRK dispatch, tolerance for node churn mid-inference, round-trip complexity per forward pass, public implementation maturity.

**Expected answer shape:** no single protocol wins outright. The table surfaces which protocols are "plausibly adaptable" vs "architecturally incompatible" with PRSM's flow.

### Q2: Can MPC be scoped to first-N transformer layers without losing privacy guarantees?

**How we'd answer:** adapt a candidate protocol (from Q1's shortlist) to cover just layers 0 through N in a sharded inference; let layers N+1 through end run in TEE-only (Phase 2 Line C) mode. Measure: (a) protocol overhead cost bounded by N, (b) R3-methodology inversion-fidelity measured on TEE-only tail layers to verify the R3 assumption that late layers leak less.

**Expected answer shape:** specific N (probably 4-8 based on activation-inversion literature) plus an overhead measurement showing the total inference-latency cost.

### Q3: Does non-collusion hold at PRSM's operator scale, and how do we measure it?

**How we'd answer:** derive a quantitative confidence interval on "no k operators colluding" from PRSM-SUPPLY-1's diversity metrics. Specifically — if the top-k providers span ≥3 legal jurisdictions, ≥3 operator classes (T1/T3/T4 mix), and have uncorrelated slashing stakes, the probability of k-collusion is bounded by some function of those parameters. Question 3 asks for that function.

**Expected answer shape:** a non-collusion confidence model that reads PRSM-SUPPLY-1 metrics as input and emits a bound on collusion probability. Partner org owns the modeling; SUPPLY-1 owns the data.

### Q4: What is the bandwidth-latency cost of MPC on first-N layers under PRSM's 9000× bandwidth handicap?

**How we'd answer:** prototype a 3-party secure protocol on the SPRK activation path for layers 0-N; measure end-to-end added latency and bandwidth on consumer-residential, T3-cloud, and T4-meganode paths. Compare against baseline plaintext inference.

**Expected answer shape:** a cost table (ms of added latency, MB of added bandwidth per inference). This is the go/no-go for whether R2 progresses past prototype.

### Q5: How does MPC compose with R7 compression (activation quantization)?

**How we'd answer:** run MPC-on-first-N combined with R7's TurboQuant activation compression on the same layers. Measure whether MPC overhead × compression gain produces net-acceptable cost vs baseline.

**Expected answer shape:** the composition either has additive overhead (acceptable; R2+R7 both ship) or multiplicative (R2 alone without R7, OR vice versa). Per the R7 plan's note: *"if R7 shows 4× bandwidth reduction and R2 adds 10× MPC overhead, net is 2.5× overhead vs today's plaintext streaming."* Q5 verifies that composition is in fact additive.

---

## 3. Candidate MPC protocol landscape

### 3.1 General-purpose MPC frameworks

- **MP-SPDZ** — widely used, supports many protocol variants. Good for prototyping but not production-optimized for transformers.
- **CrypTen (Meta)** — PyTorch-native MPC. More ergonomic than MP-SPDZ but less protocol flexibility; primarily 2PC semi-honest.
- **SecretFlow (Ant Group)** — open-source federated learning + MPC platform. Includes SPU (Secure Processing Unit) for accelerated MPC.

**PRSM implication:** these are good starting points for the Q1 comparison but are designed for data-center MPC, not permissionless-operator MPC. Adaptation to PRSM is non-trivial.

### 3.2 Transformer-specific MPC protocols

- **Iron** (Hao et al. 2022, [arXiv:2211.01452](https://arxiv.org/abs/2211.01452)). Practical secure inference for transformers. Evaluated on BERT-size; frontier-scale not reported.
- **BumbleBee** (Lu et al. 2023, [arXiv:2310.01737](https://arxiv.org/abs/2310.01737)). Post-LayerNorm MPC optimizations. BERT-size evaluations.
- **Puma** (Dong et al. 2023, [arXiv:2307.12533](https://arxiv.org/abs/2307.12533)). Practical secure transformer inference on Ant Group's SPU. Demos on Llama-7B.
- **MPC-Minimized Secure LLM Inference** (Rathee et al. 2024) — reduces MPC overhead by moving non-linear ops to plaintext where acceptable.

**Currently most promising lineage for R2:** **Puma line** — it's the only one with documented results at LLaMA-7B scale. Upcoming work from the same group targets Llama-13B and 30B. **Starting point for R2 prototyping.**

### 3.3 Protocol variants by party count + trust model

| Variant | Parties | Security model | PRSM fit |
|---|---|---|---|
| 2PC semi-honest | 2 | honest-but-curious | 🟡 (strong collusion bound but brittle) |
| 3PC semi-honest (ABY3-line) | 3 | honest majority (2-of-3) | ✅ (matches T1/T3/T4 diversity) |
| 4PC malicious (Fantastic Four) | 4 | malicious minority | 🟡 (strongest guarantees but high overhead) |
| N-party (MP-SPDZ) | variable | flexible | ❌ (coordination cost kills it at PRSM scale) |

**Starting assumption for R2:** 3PC semi-honest with an honest-majority (2-of-3) assumption. This aligns with PRSM-SUPPLY-1's diversity targets — if the three operators span three jurisdictions + three operator classes, the honest-majority assumption has strong non-collusion support (Q3).

### 3.4 Explicit non-candidates

- **Full homomorphic encryption (FHE)** — this is R1's scope, not R2. FHE is 10,000-100,000× slower; not practical.
- **Trusted execution environments alone** — already in PRSM's stack via Phase 2 Line C. R2 is additive to TEE, not replacement.
- **Zero-knowledge proofs** — orthogonal. ZK proves inference happened correctly; MPC hides the inputs. Different threat models.

---

## 4. MPC-on-first-N-layers as the realistic first milestone

Full-transformer MPC at Llama-3.1-8B scale is marginal at the state of the art (Puma reports reasonable latency but still 10-100× overhead). At 70B scale it is unsolved.

A scoped milestone: MPC on the **first N transformer layers**, TEE-only on the rest.

### 4.1 Rationale

R3's threat model (A1 direct inversion) establishes that activation-inversion attacks are effective mostly on early-layer activations — the attacker's best shot at reconstructing a prompt is when activations are still close to input embeddings. Mid-to-late layer activations carry less input-mutual-information and are increasingly task-specific rather than content-specific.

**If R3's A1 Phase 1 baseline confirms this assumption** (≤0.3 cosine similarity on layer-N-and-beyond inversions), then TEE alone is sufficient for layers N+1 onward. MPC buys privacy where it's needed and is out of the way where it isn't.

### 4.2 N-selection logic

N is chosen as the smallest layer index where R3's A1 baseline fidelity drops below the §6.4 safety claim threshold (cosine ≤ 0.3). Published activation-inversion work typically puts this around layers 3-8 for Llama-class models, but the exact value is R3-Phase-1-dependent.

The R3 plan and R2 plan should share this measurement. R3-Phase-1 produces the layer-by-layer fidelity curve; R2 reads N from the curve.

### 4.3 Overhead bound

Protocol overhead is O(N) in the number of MPC layers. If Puma-class protocols add ~20× overhead per layer and N=5, the full inference adds 5 × 20× = 100× overhead on 5 layers, which averages to 3-4× overhead across a 32-layer Llama-3.1-8B. This is the back-of-envelope; Q4 validates it empirically.

### 4.4 Composition with TEE-only tail

The transition from MPC to TEE-only happens between layers N and N+1. Activations at that boundary are reconstructed from MPC shares and passed in plaintext (within TEE) to layer N+1. The TEE attestation covers layers N+1 onward.

**Risk:** the boundary-crossing reconstruction moment is a potential attack point. A compromised TEE could capture reconstructed activations. Mitigation: the TEE at the boundary is chosen via Phase 2 Line B topology randomization from a different operator-class bucket than the MPC parties. Explicit separation-of-duty across the boundary.

### 4.5 Full-transformer MPC deferred

Puma-line work extending to Llama-30B is on the research frontier as of early 2026. If that lands at acceptable overhead (say, ≤5× end-to-end), R2's scope may expand to full-transformer MPC in a follow-up milestone. For the initial R2 milestone, first-N-layers is the target.

---

## 5. Non-collusion quantification via PRSM-SUPPLY-1 metrics

Q3 asks: how do we quantify that non-collusion holds in PRSM's permissionless network?

### 5.1 Diversity dimensions (from PRSM-SUPPLY-1)

SUPPLY-1 already measures:
- Per-provider share (hyperscaler concentration).
- Per-region share (geographic concentration).
- HHI indices for both.
- Stake concentration via `StakeBond` (economic diversity).

These are the inputs to non-collusion confidence.

### 5.2 Confidence model (sketch)

For a 3PC protocol to be secure under honest-majority, at most 1 of the 3 parties can collude with the adversary. Probability of 2+ colluders (protocol break) is the sum over pairwise-collusion probabilities.

A confidence model would look like:

```
P(collusion) = P(2+ parties collude)
             ≈ Σ_{pairs (i,j)} P(parties i and j have correlated interests)
```

Correlation factors that raise collusion probability:
- Same jurisdiction (subject to same legal coercion).
- Same hyperscaler provider (provider-level compromise affects both).
- Operated by same principal (sybil).
- Correlated stakes (slashing one is approximately slashing the other).
- Same operator class (e.g., three T3 cloud-arbitrage operators all dependent on one hyperscaler).

SUPPLY-1's diversity-bonus mechanism already creates economic pressure toward reducing these correlations. Q3 deliverable is the function that maps SUPPLY-1 metrics to collusion probability, plus the threshold at which PRSM can publish "3PC MPC is honest-majority secure under PRSM-supply-conditions."

### 5.3 Dynamic party selection

Party selection for an MPC inference reads the same SUPPLY-1 metrics:
- Pick 3 providers from 3 different jurisdictions.
- Prefer 3 different operator classes (T1 + T3 + T4 is ideal).
- Verify uncorrelated stake / slash history via ReputationTracker.
- Rotate selection each inference (Phase 2 Line B topology randomization).

The dispatcher's top-k selection (already in `MarketplaceOrchestrator._select_top_k`) gains a diversity-aware MPC-selection extension.

### 5.4 Adaptive adversary

An adversary who understands the selection algorithm can try to register operators across jurisdictions / classes to maximize selection probability. SUPPLY-1 §8.3's stake-binding makes this expensive (attacker must bond across k distinct registered-operator identities). The non-collusion model must account for adaptive registration; Q3's function should model this.

---

## 6. Bandwidth-latency cost under 9000× handicap

The R2 stub flagged this: *"Does MPC overhead compose acceptably with the 9000× bandwidth handicap between consumer nodes vs. datacenter NVLink?"*

### 6.1 Dimensional analysis

MPC protocols typically add:
- **Communication rounds:** 2-5 round trips per MAC operation in matrix multiplication.
- **Bandwidth:** 2-3× the activation-tensor size per round (shares of each tensor).
- **Computation:** modest overhead (+20-50%) from MPC-field arithmetic.

On NVLink (~900 GB/s): 3× bandwidth × 3 rounds = ~9× additional data, latency dominated by compute.

On PRSM's P2P consumer bandwidth (~100 MB/s residential): same 9× data × 9000× slower transport = ~81000× slower per round. Multiply by round count → practically unusable.

**Implication:** MPC-on-all-operators is incompatible with T1/T2 consumer-residential operators under current networking. MPC operators are realistically T3/T4 only at first.

### 6.2 R7 composition as mitigation

R7's compression cuts activation bandwidth 3-5×. If R7 ships, MPC-over-compressed-activations is 3-5× cheaper on the wire. Still far from consumer-viable but closes the gap to T3-feasible substantially.

### 6.3 Milestone bandwidth target

For Q4 to pass:
- T3-T3-T3 operator set: MPC on first N layers adds ≤3× end-to-end latency vs baseline plaintext sharded inference.
- T4-T4-T4 meganode set: MPC adds ≤2× (better networking).
- T3-T4-T1 heterogeneous set: deferred; this is R2-follow-up scope.

Q4 fail condition: T3-T3-T3 adds ≥10× latency, which would mean MPC is economically uncompetitive with TEE-only even at premium-tier prices.

---

## 7. Composition with R3 + R7

### 7.1 R3 ↔ R2

- R3 determines N (§4.2).
- R3's Phase 3 adaptive attacks include the MPC boundary crossing (§4.4) as a target — does an attacker exploit the boundary reconstruction to recover activations?
- R3's collusion attack (A4) is the direct test of R2's honest-majority assumption at protocol level: if A4 succeeds at k=2, 3PC honest-majority breaks and R2 must use 4PC malicious-minority.

### 7.2 R7 ↔ R2

- R7's compression composes with MPC per Q5 (§2).
- R7's H4 (consumer-hardware quantization cost) informs whether T1/T2 operators could participate in MPC-with-compression or whether R2 remains T3/T4-only.
- R7's data-oblivious property (shared-seed rotation, no per-block metadata) is useful for R2: MPC parties can derive the same rotation without exchanging calibration, simplifying MPC round reduction.

### 7.3 Scheduling

R2 cannot start meaningfully until R3 has produced at least its layer-by-layer fidelity curve (R3 Phase 1, weeks 1-3). R2's prototype also benefits from R7's scheme selection (R7 Phase 1-2 results). Therefore:

**Recommended order:** R3 Phase 1 → R7 Phase 1-2 → R2 starts. R2 is roughly quarter 2 or 3 of combined research, not quarter 1.

Budget implication: R3 + R7 first, R2 second. Combined R3+R7 is ~$250k-$500k; R2 is ~$300k-$500k more, across a second research quarter. Total R2+R3+R7 research program: $550k-$1M over 2-3 quarters.

---

## 8. Experiment phases

5 phases over 16-20 weeks, starting after R3 Phase 1 completes.

### Phase 1: Protocol selection (weeks 1-3)

- Produce the Q1 comparison table.
- Prototype two candidate 3PC protocols (from Puma-line and a non-transformer-specific baseline like ABY3) on Llama-3.1-8B, first 3 layers only, latency benchmark.
- **Gate:** shortlist of 1-2 protocols for Phase 2.

### Phase 2: First-N-layer scoping (weeks 4-8)

- N selected from R3 Phase 1 fidelity curve.
- Adapt the shortlisted protocol to cover layers 0 through N; TEE-only from N+1.
- Measure Q2 (privacy — via R3-methodology inversion on the TEE-tail).
- **Gate:** no privacy regression on TEE-tail vs full-MPC baseline.

### Phase 3: Bandwidth-latency measurement (weeks 9-13)

- Deploy Phase 2's prototype on Phase 2 Rings 7-10 SPRK infrastructure.
- Measure Q4 across T3-T3-T3, T4-T4-T4, T3-T4-T1 operator sets.
- **Gate:** T3-T3-T3 ≤3× overhead for first-milestone pass; T4-T4-T4 ≤2×.

### Phase 4: Composition with R7 (weeks 14-16, requires R7 Phase 4)

- Combine with R7 compression.
- Measure Q5.
- **Gate:** additive overhead (within 1.5× of sum of independent overheads).

### Phase 5: Non-collusion modeling (weeks 14-20, can parallel Phase 3-4)

- Academic partner (Q3 work) develops the non-collusion confidence function.
- Validates against PRSM-SUPPLY-1 diversity metrics.
- **Deliverable:** function + threshold + paper draft.

Total: 20 weeks if phases run sequentially; 16 weeks with parallelism.

---

## 9. Deliverables

1. **R2-RESULTS-1** — results document with Q1-Q5 answered.
2. **Prototype code repo** — Phase 1-4 implementations.
3. **Non-collusion model** — from Phase 5, plus reference Python implementation that consumes SUPPLY-1 metrics and emits collusion probability.
4. **Partner recommendation memo** — if milestone gates pass, proposed engineering-promotion path; if fail, specific reasons and recommended next research direction.
5. **Academic publication(s)** — Phase 3 bandwidth-measurement work and Phase 5 non-collusion work are independently publishable.

---

## 10. Partner handoff

### 10.1 Partner profile

Cryptography research group with prior MPC publications, ideally with transformer-MPC experience. Candidate orgs:
- Ant Group's SPU team (Puma authors).
- Facebook / Meta AI Research (CrypTen team).
- ETH Zurich secure-systems group.
- UCB Berkeley RISE / SkyLab.
- Cornell systems/crypto groups.

PRSM's existing cryptography-partnership footprint is limited; partner identification is a Foundation action.

### 10.2 Day 1 materials

- This scoping doc.
- R3 Phase 1 fidelity curve (when R3 reaches Phase 1).
- R7 scheme selection (when R7 reaches Phase 2).
- Phase 2 Rings 7-10 architecture walkthrough.
- §5.4 PRSM dispatch logs (shared with R3 and R7).
- Reference hardware allocation (T3-T3-T3 + T4-T4-T4 test sets).

### 10.3 Week 1 milestones

- Environment stood up with Puma CUDA kernels (or shortlisted protocol impl).
- Baseline Llama-3.1-8B running full-precision single-node.
- First 3-layer MPC prototype running (any protocol, any latency — proof of env-functional).

### 10.4 Month 1 milestones

- Q1 comparison table drafted.
- Two protocols benchmarked to the same spec.
- Go/no-go decision on Phase 2.

### 10.5 Quarter 1-2 milestones

- All Phase 1-5 deliverables (§9) shipped.

---

## 11. Budget

- **Personnel:** 2-3 researchers × 20 weeks + blue-team support (~0.1 FTE of Phase 2 implementer).
- **Hardware:** shared with R3+R7 T3/T4 pools + 1 additional high-bandwidth test rig for MPC round-trip benchmarking.
- **Rough total:** $300k-$500k for Phase 1-5. Lower bound if a single partner org absorbs some cost against grant funding.
- **Combined R2+R3+R7 research program:** $550k-$1M over 2-3 quarters.

Foundation R&D budget implications: this is multi-quarter spending. Justification rests on whether MPC becomes load-bearing for Tier B/C content — a judgment PRSM-SUPPLY-1 grace-period data can inform.

---

## 12. Open issues

### 12.1 MPC standardization risk

MPC protocols for transformers are an active research area. A scheme that looks best today may be superseded by a 2027 paper. R2 should choose a protocol on short-term merits (best-known-as-of-now) and expect to revisit on a 1-2 year cycle. This is opposite to R6's posture (wait for standards) because MPC has no NIST-equivalent.

### 12.2 Non-standard hardware (SPU)

Ant Group's SPU is a specialized MPC accelerator. Puma's results depend on it. If R2 adopts the Puma protocol, PRSM's operator hardware may need SPU-compatible accelerators — a non-trivial hardware constraint for T3 operators using stock H100s. Protocol choice implicitly decides hardware compatibility.

### 12.3 Semi-honest vs malicious

3PC semi-honest is the starting point (§3.3). But PRSM's slashing economics are malicious-behavior-aware — a provider willing to lose their stake IS a malicious adversary. Semi-honest is a weaker threat model than PRSM's actual adversary. Upgrading to 4PC malicious-minority multiplies overhead roughly 3-5×. Design choice: settle for semi-honest with economic mitigation (slash detected deviators) vs pay full malicious-security overhead.

### 12.4 Legal/regulatory implications of MPC

Some jurisdictions regulate MPC as export-controlled cryptography (similar to FHE). Operator-level deployment may have export-control implications in specific regions. Foundation legal review required before promoting R2 to engineering.

### 12.5 Q5 composition with R7's rotation schemes

R7's data-oblivious rotations (TurboQuant) share a seed across SPRKs. In an MPC context, the seed itself must be distributed in a way that doesn't reveal information to an attacker. Q5 may surface a subtle protocol interaction: a naive seed-distribution inside MPC could leak information. Deferred to Phase 4 investigation.

### 12.6 Full-transformer MPC deferral

§4.5 defers full-transformer MPC. If Puma-line research lands Llama-30B or -70B MPC at ≤5× overhead by the time R2 Phase 5 completes, reopen the scoping decision. R2 v2 could target full-transformer directly, skipping the first-N-layers intermediate.

---

## 13. Cross-references

### 13.1 To other research tracks

- **R1 FHE.** Strict superset of R2's privacy guarantees but at 100-1000× worse overhead. R2 is the practical-today path; R1 is long-horizon.
- **R3 activation-inversion.** R3 determines N (§4.2) and runs the A4 adaptive collusion attack against R2's honest-majority assumption (§7.1). Blocking dependencies in both directions.
- **R5 Tier C hardening.** R2's cryptographic weight sharding (implied by MPC) IS R5's missing primitive. If R2 ships, R5 inherits.
- **R7 compression.** §7.2 composition detail. R2 schedules after R7 Phase 2.
- **R8 anti-exfiltration.** R2's MPC protocol doubles as R8's weight-sharding defense (R8's defense layer 3). If R2 ships, R8 gets a scheme for free.

### 13.2 To engineering

- **Phase 2 Rings 7-10 (tensor-parallel sharding).** Implementation target.
- **MarketplaceOrchestrator._select_top_k.** Extends with diversity-aware MPC party selection (§5.3).
- **PRSM-SUPPLY-1.** Diversity metrics consumed as Q3 inputs.
- **ReputationTracker.** Used to verify uncorrelated slash history for party selection.

### 13.3 To strategy

- **Tier B/C content viability.** MPC is the privacy-tier mechanism for content too sensitive for TEE-only. If R2 succeeds, PRSM's Tier B/C offering strengthens materially.
- **PRSM_Vision.md §7 "Honest limits".** R2 tightens the disclosure from "MPC is 10-100× overhead" to a specific overhead number for PRSM's protocol choice.

---

## 14. Ratification

This document does not require governance ratification; it is a research scoping doc. Ratifies implicitly upon:

1. Foundation allocates budget and identifies partner org.
2. R3 Phase 1 completes (provides N).
3. R7 Phase 2 completes (provides scheme selection).
4. Partner org reviews, adjusts, commits.
5. Phase 1 kickoff.

Earliest realistic start: one quarter after R3+R7 kickoff.

---

## 15. Changelog

- **0.1 (2026-04-22):** initial draft, founder-authored. Promoted from R2 stub. Designed for post-R3+R7 execution.
