# R7 Benchmark Plan: Data-Oblivious KV + Activation Compression on PRSM

**Document identifier:** R7-BENCH-1
**Version:** 0.1 Draft
**Status:** Preregistered experimental design. Execution is a full research quarter; this document is the design that an academic partner or PRSM research contractor would pick up and run.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Promotes:** `docs/2026-04-14-phase4plus-research-track.md` §R7 from stub to preregistered benchmark plan.
**Related documents:**
- `docs/2026-04-14-phase4plus-research-track.md` §R7 — original research stub with the "concrete starting hypothesis."
- `docs/2026-04-12-phase2-remote-compute-plan.md` — Phase 2 launch UX thesis + SPRK dispatch architecture; R7 targets the activation-streaming path specified there.
- Phase 2 Rings 7-10 (tensor-parallel model sharding) — the production surface R7 results would eventually affect.
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — R7 compression gains shift the T1/T2 vs T3 cost curve, which this standard's policy reacts to.

---

## 1. Purpose

The R7 stub established three things:

1. A data-oblivious KV-quantization lineage (QJL → PolarQuant → TurboQuant) from a coherent research group is the right *starting* candidate for PRSM-specific benchmarking — specifically because per-block quantization metadata would otherwise travel between SPRKs on every pipeline-parallel stage handoff, defeating the compression gain.
2. A concrete starting hypothesis with three sub-experiments (KV quantization, activation-streaming quantization, composition with Phase 2's topology randomization / Ring 9 DP noise / TEE attestation).
3. A trigger for moving to engineering (≥3× bandwidth reduction at <1% accuracy loss, red-team clean, plus network-level signal that T1/T2 viability needs shifting).

**What the stub did NOT specify, and what this plan adds:**

- **Exact reference models** (§4).
- **Exact workloads / datasets** with accuracy metrics (§5).
- **Quantitative success thresholds per experiment** that, if met, advance the scheme to the next gate (§7).
- **Preregistered null hypotheses** per experiment so the research is falsifiable (§2).
- **Red-team integration with R3** (activation-inversion characterization) as a *blocker* rather than a nice-to-have (§9).
- **A partner-handoff checklist** (§12) so this plan is executable by someone who doesn't have the full PRSM context.

This document is a preregistered plan. Once executed, results go into a results document with the same numbering (R7-RESULTS-1).

---

## 2. Preregistered hypotheses

Five primary hypotheses. Each is paired with a null, a pass condition, and a fail condition. The pass condition defines the bar for "this scheme advances"; the fail condition defines the bar for "this scheme is eliminated." Ambiguous results (between pass and fail) trigger additional measurement, not a declaration.

### H1: TurboQuant KV quantization at 3.5 bits matches baseline accuracy

- **H1 (pass):** Llama-3.1-8B-Instruct at 3.5 bits/channel TurboQuant KV maintains ≥99% of full-precision accuracy on LongBench-V1, averaged across ≥10 subtasks.
- **H1 (fail):** average accuracy drops below 98% OR any single subtask drops below 95% of full-precision.
- **H1 (null):** paper results (≥100% average, ≥97% worst case) do not replicate on an independent run. Elimination of the scheme from further PRSM work.

### H2: TurboQuant activation streaming at 3.5 bits matches baseline accuracy

- **H2 (pass):** ≥99% of full-precision accuracy on LongBench-V1 AND ≥99% on HumanEval+ (code-gen tasks bring out compression artifacts faster than generic summarization) when inter-stage activations are TurboQuant-compressed to 3.5 bits/channel.
- **H2 (fail):** average accuracy drops below 97%.
- **H2 (null):** activations have distributional properties that break TurboQuant's unbiasedness guarantee; accuracy degrades more than KV compression at the same bit rate. **This is the plausible outcome given that activations differ statistically from KV tensors** — if null, fall back to the outlier-aware rotations of RoTateKV adapted for activations.

### H3: Bandwidth reduction on SPRK activation streaming is ≥3×

- **H3 (pass):** end-to-end inter-SPRK pipeline-stage activation transfers consume ≤ 33% of the baseline bytes on the Phase 2 reference inference workload at 3.5-bit quantization, measured as total P2P bytes over N=1000 requests.
- **H3 (fail):** bandwidth reduction below 2× (i.e., metadata/packing/overhead eats more than half the nominal compression).
- **H3 (null):** PRSM's P2P transport wraps activations in a payload envelope whose overhead dominates the raw tensor bytes, making compression gain at the wire less than compression gain in the tensor itself.

### H4: Per-vector quantization cost is compatible with T1/T2 hardware

- **H4 (pass):** TurboQuant forward pass at d=1536 runs in ≤3ms on consumer-class hardware (RTX 4090, Apple M2 Max) — 2-3× the paper's reported H100 cost is acceptable given consumer hardware.
- **H4 (fail):** >10ms on either consumer platform OR >5ms on both (implying fundamental algorithmic issue rather than hardware parity gap).
- **H4 (null):** the paper's CUDA kernel does not port to Metal (Apple) or consumer CUDA (RTX) without specialized rewrites that fall outside reasonable research-contractor scope. R7 then pivots to H2+H3 only on datacenter hardware, with T1/T2 viability deferred pending a separate kernel-porting effort.

### H5: Quantization does not compound with activation-inversion attack surface

- **H5 (pass):** R3-methodology red-team finds no new attack surface introduced by TurboQuant quantization; published inversion attacks (Zhu et al. 2019 and follow-ups) show ≥30% reduction in reconstruction fidelity when applied to quantized activations vs baseline.
- **H5 (fail):** red-team identifies a new attack vector OR reconstruction fidelity increases (quantization creates exploitable structure).
- **H5 (null):** results are inconclusive — inversion fidelity unchanged but no new vector found. R7 advances conditionally; R3 red-team work extends for another quarter specifically on this surface.

**Why H5 is blocking, not a nice-to-have.** PRSM's launch UX thesis depends on user trust in privacy claims. A compression scheme that reduces bandwidth but increases inversion fidelity even marginally is a net-negative regardless of bandwidth gains.

---

## 3. Candidate schemes — comparison table

Beyond the three lineage papers (QJL, PolarQuant, TurboQuant), four adjacent schemes serve as baselines.

| Scheme | Type | Data-obl? | Calib? | Claimed ratio | Claimed acc loss | Public impl? | Primary citation |
|---|---|---|---|---|---|---|---|
| **QJL** | KV-1-bit JL | ✅ | ❌ | >5× at 3b/FPN | 0% on Llama-2 | ✅ CUDA | Zandieh '24 [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) |
| **PolarQuant** | KV polar-transform | ✅ | ❌ | 4.2× | SOTA quality | 🟡 paper-only | Han '25 [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) |
| **TurboQuant** | KV MSE-optimal + QJL branch | ✅ | ❌ | 3.5b ≈ full; 2.5b ≈ 98% | <1% at 3.5b | 🟡 paper + QJL | Zandieh '25 [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| KIVI | KV asymmetric 2-bit | ❌ (per-block scale) | ❌ | 4× | minor | ✅ | Liu '24 [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| KVQuant | 10M-context KV 4-bit | ❌ | ❌ | up to 4× | minor | ✅ | Hooper '24 [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| QuaRot | End-to-end 4-bit rotation | ❌ | ✅ (required) | 4× at model + KV | ~1% | ✅ | Ashkboos '24 [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) |
| RoTateKV | 2-bit outlier-aware | ❌ | partial | 8× | <1% | ✅ | Su '25 [arXiv:2501.16383](https://arxiv.org/abs/2501.16383) |

**Column legend:**
- *Data-obl?* — does the scheme avoid per-block metadata that would travel between SPRKs? Load-bearing for PRSM per R7 stub.
- *Calib?* — requires per-deployment calibration? **Calibration is near-impossible in PRSM's permissionless topology** because nodes churn in/out on minute timescales and a calibration phase would gate every inference behind a lengthy warm-up.
- *Public impl?* — is there production-quality code available to prototype against? QJL's CUDA kernel at github.com/amirzandieh/QJL is the foundational primitive; PolarQuant and TurboQuant are currently paper-only and require implementation.

**Primary scheme selection rationale:**
- **TurboQuant is the scheme to benchmark first** because (a) it's the most recent synthesis, (b) it retains data-obliviousness of QJL while adding the MSE-optimal branch for when unbiasedness isn't required, (c) the QJL branch has a public CUDA kernel we can bootstrap from, (d) theoretical guarantees are strongest (near-Shannon-optimal).
- **QuaRot is the most important non-lineage baseline** because it's the only published scheme measuring activation-streaming quantization end-to-end. Its calibration requirement is a blocker for PRSM production but NOT a blocker for benchmark — we include it for comparison to establish an upper bound on what non-data-oblivious schemes can achieve.

---

## 4. Reference models

Three models, chosen to span the deployment spectrum and to enable comparison with published results.

### 4.1 Llama-3.1-8B-Instruct

- **Purpose:** primary benchmark target. TurboQuant's paper uses exactly this model on LongBench-V1; direct replication of their results is the first milestone.
- **Size:** 8B params, 32 transformer layers, head_dim=128.
- **Deployment:** Phase 2 reference inference workload. Plus the T1/T2-viability question (fits in 24GB consumer GPU VRAM at fp16).
- **License:** Llama 3.1 Community License — acceptable for research.

### 4.2 Mistral-7B-Instruct-v0.3

- **Purpose:** secondary model to check cross-family generalization. If results on Llama and Mistral diverge, the scheme depends on architectural details; if results agree, the scheme is likely robust.
- **Size:** 7B params.
- **Deployment:** same as Llama.
- **License:** Apache 2.0.

### 4.3 Llama-3.1-70B-Instruct

- **Purpose:** frontier-scale check. Compression schemes that work at 8B don't always hold at 70B — quantization error compounds across deeper networks. This model is the T3/T4-only tier for R7 and is the *actual* target of PRSM's sharded-inference path.
- **Size:** 70B, requires multi-GPU or fp8 quantization baseline to fit.
- **Deployment:** T3 / T4 only during benchmark. On H100 (~140GB with fp16 weights, fits on 2× H100 or 1× H200).
- **License:** Llama 3.1 Community License.

**Not benchmarked, with rationale:**
- **Models smaller than 7B.** R7's motivation is bandwidth-constrained sharded inference of frontier-adjacent models; sub-7B models don't need sharding and are less bandwidth-bound.
- **Models with non-standard attention (DeepSeek MoE, Gemini-style sparse).** These introduce confounds — any compression failure may be architectural rather than scheme-specific. Follow-up after TurboQuant is validated on dense transformers.

---

## 5. Reference workloads

Four workload types. Each targets a specific quality dimension that compression artifacts affect differently.

### 5.1 LongBench-V1

- **Coverage:** long-context understanding (16k-32k tokens), multi-document QA, summarization, few-shot learning, code completion.
- **Why:** this is TurboQuant's paper benchmark; direct replication test. Also stresses KV cache more than generic benchmarks — long-context amplifies any per-token compression error.
- **Subtasks of interest:** Qasper, MultiFieldQA-en, HotpotQA, 2WikiMultihopQA, GovReport, QMSum, TREC, TriviaQA, SAMSum, LCC, RepoBench-P.
- **Metric:** task-specific (F1, Rouge-L, accuracy depending on subtask); aggregated as macro-average.

### 5.2 Needle-in-a-Haystack (NIAH)

- **Coverage:** retrieval precision across context depth. Measures whether compression loses attention's ability to pick a specific fact from a 32k-token context.
- **Why:** complements LongBench — LongBench measures "can the model still do the task"; NIAH measures "did compression destroy attention's retrieval capability specifically." TurboQuant reports 0.997 on this at 3.5b; we replicate.
- **Metric:** exact-match retrieval accuracy across context depths (1k, 4k, 8k, 16k, 32k) and positions (0%, 25%, 50%, 75%, 100% of context).

### 5.3 HumanEval+

- **Coverage:** code generation, a domain where small output perturbations fail tests that small accuracy drops would not flag.
- **Why:** code-gen tasks bring out compression artifacts faster than summarization — a single wrong token can make code uncompilable. If TurboQuant at 3.5b preserves code-gen quality, it's probably safe for general downstream use.
- **Metric:** pass@1 over 164 problems × 100 samples per problem.

### 5.4 Phase 2 SPRK inference workload (PRSM-specific)

- **Coverage:** the actual sharded pipeline-parallel inference traces PRSM generates. Internal to the project; constructed from dispatch logs over a representative operating period.
- **Why:** synthetic benchmarks miss PRSM-specific characteristics (per-shard variance, topology-randomization interaction, churn during inference). Results here are the ones PRSM's supply/economic decisions react to.
- **Metric:** (a) wire bytes per request; (b) end-to-end request latency; (c) correctness vs full-precision sharded baseline (exact-match output).

---

## 6. Metrics

Each hypothesis's pass condition maps to one or more metrics. Comprehensive list:

| Metric | Unit | Used in | Measurement method |
|---|---|---|---|
| Task accuracy | task-specific | H1, H2 | LongBench / NIAH / HumanEval+ harness |
| Wire bytes per stage handoff | bytes | H3 | SPRK transport-layer byte counter |
| P2P transfer latency | ms | H3 | SPRK timestamps on send/recv |
| End-to-end inference latency | ms | H3 | orchestrator request-to-response timer |
| Per-vector quantization latency | ms | H4 | CUDA / Metal kernel timer on quantize op |
| KV memory footprint | bytes | H1, T1 viability | GPU VRAM sampling during generation |
| Activation-inversion reconstruction fidelity | cosine sim 0-1 | H5 | R3 red-team methodology (see §9) |

**Baseline convention:** "full-precision" means fp16 KV cache, fp16 activations on the wire, no rotation / projection / quantization — exactly the Phase 2 Rings 7-10 production stack at time of benchmark.

---

## 7. Success thresholds (trigger to promote to engineering)

R7's stub committed to: *"≥3× bandwidth reduction at <1% accuracy loss, with red-team clean."* This plan translates to per-hypothesis thresholds:

| Hypothesis | Pass threshold | Advances to |
|---|---|---|
| H1 | ≥99% avg accuracy at 3.5 bits on Llama-3.1-8B | H2, H3 parallel |
| H2 | ≥99% avg accuracy + ≥99% HumanEval+ on 3.5-bit activations | H3 (bandwidth measurement meaningful) |
| H3 | ≥3× P2P bandwidth reduction on §5.4 workload | T1/T2 viability recalculation |
| H4 | ≤3ms per-vector quant on consumer hardware | T1/T2 supply-economics update |
| H5 | No new attack surface + ≥30% inversion fidelity reduction | **Engineering promotion gate** |

**All five hypotheses must pass** for R7 to promote to engineering. If any hypothesis returns null or fails, the research plan branches:

- **H1 null:** eliminate TurboQuant → promote QJL-only (weaker but simpler).
- **H2 null:** KV-only deployment; activations remain full-precision. Substantial bandwidth gains deferred.
- **H3 null:** investigate transport-envelope overhead; potentially redesign SPRK transport to reduce overhead before re-benchmarking.
- **H4 null:** datacenter-only deployment; T1/T2 viability path deferred to a separate kernel-porting effort.
- **H5 fail:** full stop on R7 until R3 characterizes the new attack surface and either mitigates or disqualifies the scheme.

---

## 8. Experiment phases

Five phases, ordered by risk. Each builds on the prior so a fail at any phase cleanly shuts down downstream work without wasted effort.

### Phase 1: Replicate paper results (weeks 1-3)

- Exactly reproduce TurboQuant's Table 1 results on Llama-3.1-8B-Instruct / LongBench-V1 at the bit rates reported in the paper.
- Use the QJL public CUDA kernel for the 1-bit branch; implement the MSE-branch from paper pseudocode.
- **Gate:** results within 2% of paper values. Fail → H1 null, investigate implementation before advancing.

### Phase 2: Extend to additional models + workloads (weeks 4-6)

- Apply Phase 1 implementation to Mistral-7B and Llama-3.1-70B.
- Add NIAH + HumanEval+ to the benchmark suite.
- **Gate:** H1 pass on all three models across all three workloads. Fail → scheme is Llama-specific; eliminate or investigate.

### Phase 3: Activation-streaming quantization (weeks 7-10)

- Adapt TurboQuant from KV-cache to inter-SPRK activation tensors.
- Compare against QuaRot (with calibration) + RoTateKV (outlier-aware) baselines.
- **Gate:** H2 pass + ≥2× bandwidth observation (weak H3 signal; real measurement in Phase 4). Fail → fall back to KV-only deployment for R7.

### Phase 4: PRSM integration + bandwidth measurement (weeks 11-13)

- Integrate the scheme into the Phase 2 Rings 7-10 SPRK activation path behind a feature flag.
- Run §5.4 workload; measure all H3 metrics.
- **Gate:** H3 + H4 pass. Fail → understand the gap between sim and prod before real deployment.

### Phase 5: R3 red-team (weeks 14-16, parallel to Phase 4 where possible)

- Execute R3's activation-inversion attack methodology against the Phase-4 deployment.
- Compare inversion fidelity on compressed vs baseline activations.
- **Gate:** H5 pass. Fail → full stop.

**Total:** one calendar quarter (16 weeks) if all gates pass on the first attempt. Add ~3 weeks per null/fail branch explored.

---

## 9. Red-team plan (R3 integration)

H5 is blocking. The red-team is not a nice-to-have.

### 9.1 Attack methodology

- **Baseline:** Zhu et al. 2019 "Deep Leakage from Gradients" and follow-ups on transformer activation inversion. Apply to Phase 2's activation-streaming path with full-precision activations. Establish a reconstruction-fidelity baseline in cosine similarity.
- **Compressed:** apply the same attacks to TurboQuant-compressed activations at 3.5-bit and 2.5-bit settings.
- **Adaptive:** the red team develops a new attack specifically targeting TurboQuant's random-rotation structure. If TurboQuant's shared-seed rotation creates exploitable patterns (correlated outliers in specific dimensions after rotation), a sophisticated attacker may recover more than a naive baseline attack would.

### 9.2 Success criteria for H5 pass

- **Baseline attacks:** ≥30% reduction in reconstruction fidelity (cosine similarity) on compressed vs full-precision.
- **Adaptive attack:** no new vector identified, OR identified vector yields <20% reconstruction fidelity (strictly worse than baseline attacks on plaintext).

### 9.3 Output

- Published red-team report (internal Foundation document) listing every attack tried, its fidelity on baseline, its fidelity on compressed, and a recommendation.
- If H5 is null (inconclusive), recommend additional red-team quarter scope.

### 9.4 Personnel

- Red team: 1 researcher with prior experience in ML privacy / activation-inversion attacks. Academic partnership preferred (NYU, Stanford, ETH security groups have published on this).
- Blue team: scheme implementer from the Phase 1-4 execution team, available for design-clarification questions but not participating in the attack development.

---

## 10. Infrastructure

### 10.1 Reference hardware

| Tier | Hardware | Purpose |
|---|---|---|
| T4 datacenter | 2× H100 SXM, NVLink | Llama-3.1-70B benchmarks, paper replication |
| T3 cloud | 1× H100 PCIe (AWS p5.48xlarge) | cross-cloud parity check |
| T2 prosumer | RTX 4090 (24GB) | H4 consumer-hardware target |
| T1 Apple Silicon | M2 Max (32GB unified) | H4 second consumer target, unified-memory check |

### 10.2 Frameworks

- **Model runtime:** PyTorch 2.5+ with FlashAttention-3.
- **KV cache:** HuggingFace Transformers with custom cache extensions (QJL kernel as starting point).
- **Sharded inference:** Phase 2's SPRK runtime, patched with R7 kernel.
- **Benchmark harness:** lm-evaluation-harness (EleutherAI) for standard tasks; custom harness for §5.4.

### 10.3 Data

- Models downloaded from HuggingFace (Llama 3.1, Mistral) under respective licenses.
- LongBench-V1 from HuggingFace `THUDM/LongBench`.
- HumanEval+ from `evalplus/humanevalplus`.
- NIAH synthetic data generated per Greg Kamradt's spec.
- §5.4 PRSM traces anonymized from dispatch logs (requester/provider IDs hashed; payload hashes only, not payloads).

### 10.4 Reproducibility

- All code in a dedicated git repo (Foundation-owned).
- Random seeds fixed at 42 everywhere; per-run seeds published in result metadata.
- Reproducibility target: any reviewer with H100 access should reproduce paper-matching results within 5%.

---

## 11. Deliverables

### 11.1 Primary deliverables

1. **R7-RESULTS-1 results document** — companion to this plan, same numbering. Reports Phase 1-5 outcomes against hypotheses, with full metric tables.
2. **Code repository** — scheme implementations + benchmark harness + results-reproduction scripts.
3. **Red-team report** — §9.3 deliverable.
4. **Engineering-promotion recommendation** — if all gates pass, a short memo recommending the scheme for implementation behind a PRSM feature flag.

### 11.2 Secondary deliverables

- **Updated §R7 section** in `docs/2026-04-14-phase4plus-research-track.md` with status: either ✅ PROMOTED or the specific gate that failed.
- **Consumer-edge context-length table** — per the R7 stub, if H1 + H4 pass, publish a revised table of what context lengths are feasible on consumer GPUs and Apple Silicon with compressed KV.
- **ECON-WP-v2 feed-in** — if H3 passes, R7's bandwidth reduction changes PRSM's supply economics; ECON-WP-v2 incorporates.

### 11.3 Academic publication (optional but encouraged)

If Phase 1-5 complete with notable findings (either strong pass on H1-H5 or interesting null on H2), the research is publishable as a paper targeting ICLR / NeurIPS / MLSys. Partnership academic(s) would be first author on the publication; Foundation acknowledged as funder.

---

## 12. Partner handoff checklist

For a research partner or contractor picking this up day 1:

**Day 1 access:**
- [ ] This plan + R7 stub in the research-track doc.
- [ ] Phase 2 Rings 7-10 architecture walkthrough (1-hour synchronous session).
- [ ] §5.4 workload traces (access to anonymized dispatch logs).
- [ ] Reference hardware allocation (§10.1).
- [ ] Foundation contact for design clarifications.

**Week 1 deliverables:**
- [ ] Environment stood up; Llama-3.1-8B running full-precision baseline on reference hardware.
- [ ] QJL public kernel integrated and smoke-tested.
- [ ] Commit to LongBench-V1 task subset (subtasks from §5.1 confirmed or revised).

**Month 1 (Phase 1 complete):**
- [ ] Paper-replication results committed to the code repo.
- [ ] First gate decision filed (Phase 1 pass / fail).

**Quarter 1 (Phase 5 complete):**
- [ ] All deliverables from §11.1 shipped.
- [ ] Final recommendation (promote / hold / kill).

---

## 13. Cross-references

### 13.1 To other research tracks

- **R2 (MPC for sharded inference).** R7 is the plaintext-efficiency question; R2 is the privacy-overhead question. If R7 shows 4× bandwidth reduction at <1% accuracy loss and R2 adds 10× MPC overhead, net ~2.5× overhead vs today — plausibly acceptable for Tier B/C.
- **R3 (activation-inversion characterization).** §9 is R3 integration. H5 is blocking precisely because R3's threat model applies here directly.
- **R8 (anti-exfiltration).** R7's compressed activations may reduce information leaked to weight-exfiltration attackers performing activation-inversion — potentially strengthening R8's defense stack. Measured via §9's red-team; not assumed.

### 13.2 To engineering

- **Phase 2 Rings 7-10 (tensor-parallel model sharding).** Implementation target for Phase 4 integration.
- **MarketplaceOrchestrator / SPRK runtime.** Feature-flag wiring.
- **PRSM-SUPPLY-1.** R7's bandwidth gain shifts T1/T2 vs T3 supply-economics — directly relevant to the standard's grace-period exit conditions and diversity-bonus funding model.

### 13.3 To strategy

- **Launch UX thesis** (`docs/2026-04-12-phase2-remote-compute-plan.md`): R7 is NOT a launch-viability lever; it is a cost-curve lever once the network is in production. See R7 stub §"Framing clarification (2026-04-18)."
- **PRSM-TOK-1:** R7's T1/T2 viability shift affects FTNS demand curve modestly; not a blocker for TOK-1 v0.1.

---

## 14. Open issues

- **Personnel.** Plan assumes 1-2 implementers + 1 red-team researcher. Partner org identification (academic vs. contractor) is TBD and a Foundation decision.
- **Budget.** Hardware (§10.1) + personnel over 16 weeks + 3 weeks reserve. Preliminary estimate: $150k-$300k depending on academic-partner model. Foundation budget line.
- **Scheme-agnostic kernel.** Phase 1-2 benchmark only TurboQuant against baselines. An alternate approach — benchmark all three lineage schemes (QJL, PolarQuant, TurboQuant) — would add ~6 weeks but produce a fuller comparison. Tradeoff: depth vs breadth. Current plan prioritizes depth on TurboQuant.
- **Inter-family activation quantization.** §4 benchmarks on Llama + Mistral. A third model from a different family (e.g., DeepSeek architecture, once stable) would strengthen cross-family generalization claims. Deferred as a post-Phase-5 follow-up.
- **Hardware churn timing.** If a major new consumer GPU lands mid-benchmark (e.g., RTX 5090 release), the H4 results may need re-running on the new platform. Build a re-run protocol into the quarter's budget.

---

## 15. Ratification path

This document does not require governance ratification — it is a research plan, not a protocol standard. It ratifies implicitly upon:

1. Foundation allocates budget and identifies execution team.
2. Execution team reviews this plan, files objections or adjustments, commits to the scope.
3. Phase 1 kickoff.

Adjustments to this plan after execution starts are tracked in an addendum section (added here as §16 upon first adjustment).

---

## 16. Adjustments after execution starts

*(Empty; populated during execution.)*

---

## 17. Changelog

- **0.1 (2026-04-22):** initial draft, founder-authored. Promoted from R7 research stub. Ready for Foundation budget review and partner identification.
