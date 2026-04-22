# R1 Scoping Doc: Fully Homomorphic Encryption (FHE) for Private Inference under PRSM

**Document identifier:** R1-SCOPING-1
**Version:** 0.1 Draft
**Status:** Research scoping doc, not an execution plan. FHE for production-scale transformer inference is not solved at the state of the art — current overhead is 10,000×–100,000× plaintext. This document clarifies what "solving it for PRSM" looks like, what promotion triggers would move it from watch-item to funded execution, and what a research partner would own.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Promotes:** `docs/2026-04-14-phase4plus-research-track.md` §R1 from stub to scoping doc.
**Related documents:**
- `docs/2026-04-14-phase4plus-research-track.md` §R1 — original research stub.
- `docs/2026-04-22-r2-mpc-scoping-doc.md` — R2-SCOPING-1. R1 and R2 sit on a shared spectrum (plaintext → TEE-only → MPC → FHE). Composition in §7.
- `docs/2026-04-22-r5-tier-c-hardening-scoping-doc.md` — R5-SCOPING-1. Threshold-FHE primitives overlap substantially; §8 scopes the shared research budget.
- `docs/2026-04-22-r7-benchmark-plan.md` — R7-BENCH-1. KV-cache compression layers compose with FHE ciphertext size; §7.
- `docs/2026-04-22-r3-threat-model.md` — R3-TM-1. R1 would close the A1/A5/A6 paths in R3's attack catalog that TEE attestation cannot.
- `PRSM_Vision.md` §7 "Honest limits" — vision-doc framing.

---

## 1. Purpose

The R1 stub asked: *"Can FHE-based inference on PRSM become practical within a 3-5 year horizon?"*

This document clarifies:

1. **What "practical" means in PRSM-specific terms** — framed as an overhead budget compatible with the revenue-model constraints in `Prismatica_Vision.md` §3.3, not as generic "FHE-for-LLMs" research.
2. **What the research is NOT.** FHE for PRSM is not "add FHE to every inference path." It is a tiered capability that some consumers (regulated industries, cryptographic paranoia) would pay a premium for; most inference stays in the TEE + redundant-execution stack.
3. **Promotion triggers.** R1 is the most research-heavy item on the roadmap. The triggers that would move it from "academic watch" to "funded scoping" are specific and observable; this doc lists them so the question of "when should we fund R1" has an answer.
4. **Partner-handoff scope.** R1 requires a specialized cryptography-research partnership. This doc specifies the Day-1 package and the first-milestone deliverable.

**R1 execution is the largest on the research track** — plausibly 3-5 research quarters *after* triggers fire. This doc is scope, not a commitment to ship.

---

## 2. Non-goals

- **Not a commitment to ship FHE inference.** The trigger conditions in §6 are preregistered so the decision "ship FHE or not" is made against evidence, not hype.
- **Not a replacement for TEE + redundant execution.** FHE, if it ships, is a tier on top of the existing stack, not a replacement. Regulated-industry consumers pay a premium for the stronger guarantee; most traffic stays on the current path.
- **Not general-purpose FHE computation.** PRSM's FHE scope is transformer inference, not arbitrary MPC. Scope discipline bounds the research.
- **Not tied to a specific FHE library.** Zama, OpenFHE, Microsoft SEAL, Google's HE libraries — the scheme-family question (§4 Q1) is open and blocks the library question.

---

## 3. What "practical for PRSM" means

The vision-doc framing ("10,000–100,000× slower") is too coarse to act on. Concrete thresholds:

| Threshold | Meaning for PRSM | Source |
|-----------|------------------|--------|
| **Overhead ≤ 10,000×** | Academic milestone; no production relevance. Today's baseline. | Brakerski-Vaikuntanathan line; Zama fhEVM benchmarks |
| **Overhead ≤ 1,000×** | Economically infeasible but within demo range. A Prismatica commissioned-dataset customer could pay for a single exploratory query. | R1 Phase-2 gate |
| **Overhead ≤ 100×** | **Commercial threshold.** Regulated-industry premium can absorb this at ~100× the Tier B price; specific customer classes (healthcare genomics, legal discovery, financial modeling) are willing to pay. Composes with the Prismatica revenue model per `Prismatica_Vision.md` §3.3. | R1 Phase-3 gate / engineering-promotion trigger |
| **Overhead ≤ 10×** | Mainstream. Would displace TEE-only as the default privacy tier. Effectively never reached at today's scheme state. | Long-horizon aspiration; not in R1 scope |

**R1's job is to get from 10,000× to 100×**, or prove that the path to 100× is not achievable within the horizon. Both outcomes are valuable. A confident no lets PRSM commit to "FHE is not on our roadmap" rather than leaving it as an unresolved caveat.

---

## 4. Preregistered research questions

R1 is closer to open design exploration than R7's paper-replication work. We preregister research questions, not hypotheses with pass/fail gates. Each question has a "how we'd answer it" specification.

### Q1: Which FHE scheme family is most compatible with transformer inference under PRSM's tiered supply?

**Candidates:**
- **CKKS** (approximate arithmetic, natural fit for real-valued tensor operations). Zama's fhEVM and recent CKKS-for-ML papers (CryptoNets-2024, HELR). Precision management is the central research problem.
- **TFHE / FHEW** (bit-level, bootstrapping-heavy). Best for non-arithmetic ops (ReLU, softmax). Integration cost with CKKS-dominant pipelines is open.
- **BGV** (exact integer). Historically second-line for ML but recent work on polynomial approximations of attention softmax reopens it.
- **Hybrid CKKS + TFHE** with switching (modular approach). Higher engineering surface but matches transformer internals (matmul in CKKS, attention softmax in TFHE, bootstrap when noise budget thin).

**How we'd answer:** comparison table scoring each against (a) overhead on matmul-dominant paths, (b) overhead on softmax/GELU paths, (c) bootstrapping frequency required for an end-to-end transformer forward pass, (d) available library maturity, (e) hardware-acceleration roadmap clarity.

**Expected answer shape:** no single scheme wins outright. The answer looks like "hybrid CKKS + TFHE with scheme switching at specific transformer-block boundaries; CKKS for attention matmul, TFHE for softmax + GELU, CKKS for FFN matmul, TFHE for layer-norm." The specific boundary points are the research output.

### Q2: Can FHE overhead be bounded to ≤1,000× by selective-layer application and hardware acceleration?

**Framing:** not "can we make FHE fast enough for all of every transformer" (answer: no, today) but "can we make FHE fast enough for the layers where it matters, with the rest of the model running in TEE-only mode?"

**How we'd answer:**
- Identify the "privacy-critical" layers per R3 threat model (first N transformer layers where activation-inversion attacks are strongest; R3 Phase-4 output gates this with measured inversion fidelity).
- Run FHE-on-first-N-layers + TEE-on-remainder with best-available scheme from Q1. Measure end-to-end overhead vs. TEE-only baseline.
- Cross-check against hardware-acceleration pipeline: Intel HERACLES (expected 2027-2028 per Intel roadmap), specialized FHE ASICs (Optalysys, Cornami in development), GPU acceleration via cuFHE + cuCKKS.

**Expected answer shape:** specific N + measured overhead envelope + hardware assumption set. Example output: "N=4 layers, overhead 800× on H100, overhead projected ~150× on Intel HERACLES, overhead ~80× on a Cornami-class ASIC if one ships."

### Q3: Does FHE compose with PRSM's tensor-parallel sharding?

**Framing:** PRSM's inference is sharded across SPRKs (Phase 2 Rings 7-10). FHE literature almost always assumes single-node inference. The open question: can FHE ciphertexts be sharded the same way plaintext activations are, or does FHE require the whole encrypted computation to stay in one node?

**Sub-questions:**
- Can tensor-parallel matmul operate on FHE ciphertext shards, or does the all-reduce-equivalent step require decryption (i.e., can't be done on shards)?
- Does the shard boundary create a "partial plaintext" leakage vector (e.g., an attacker holding one shard of an encrypted activation learns something about the full activation)?
- Is threshold FHE (ciphertexts split across nodes, M-of-N decryption) the right architecture, or is single-node FHE + model sharding the right abstraction?

**How we'd answer:** formal analysis + small-scale prototype on a 2-shard, 6-layer transformer with a candidate scheme from Q1.

**Expected answer shape:** either (a) "yes, FHE composes — specifically this scheme under this sharding pattern," or (b) "no, FHE requires single-node execution; PRSM would need to run FHE on one SPRK that holds the entire model slice under encryption." Answer (b) constrains which consumer workloads FHE could serve (smaller models, specialized hardware).

### Q4: What are the composition costs with R2 (MPC) and R7 (compression)?

**Framing:** R1/R2/R7 are not independent research threads. They share a consumer-facing spectrum:
- TEE-only: fastest, weakest guarantee.
- TEE + MPC (R2): moderate overhead (10-100×), stronger guarantee but requires non-collusion assumption.
- TEE + FHE (R1): highest overhead (100-10,000×), strongest computational guarantee.

And R7's KV-cache compression composes with all three by reducing ciphertext/mask/KV memory footprint.

**How we'd answer:** compose R1-Phase-2 FHE stack with R7-Phase-3 compressed KV cache. Does compression reduce FHE overhead (smaller ciphertexts = cheaper ops)? Does FHE reduce compression's effectiveness (homomorphic ops don't compose with arbitrary lossy compression)?

**Expected answer shape:** a composition matrix. Columns: privacy tiers. Rows: compression schemes. Each cell: (overhead, quality-loss, security-level). Picks out the Pareto frontier.

### Q5: What are the promotion triggers from R1-research to R1-engineering?

Specified in §6. The answer shape is: a named list of observable events, each of which, if it occurs, increments PRSM's confidence that FHE is worth funding.

---

## 5. Phased execution plan

**Scheduling note.** R1 is funded AFTER (a) R2 Phase-2 demonstrates MPC feasibility or non-feasibility for PRSM, AND (b) R7 Phase-3 establishes the compression baseline FHE would compose with. Running R1 before either is premature — the "does it compose" questions are unanswerable without those inputs.

Earliest viable R1 Phase-1 start: **Q3 2027** (after R2+R7 complete per their plans). Latest acceptable: **Q4 2029** (before the next regulated-industry RFP cycle).

### Phase 1: Scheme selection + landscape report (4-6 months)

**Deliverable:** answer to Q1. Comparison table + rationale + shortlist of 1-2 scheme families for Phase-2 prototyping.

**Gate to Phase 2:** shortlisted scheme has a public implementation with documented performance benchmarks on at least one transformer-block-scale workload (not just MNIST-scale).

### Phase 2: Selective-layer prototype (6-9 months)

**Deliverable:** answer to Q2. Working prototype of FHE-on-first-N-layers for a small reference model (Llama-style 125M-1B params range, 6-12 transformer layers). Measured overhead on available hardware + projected overhead on near-term accelerators.

**Gate to Phase 3:** measured overhead ≤ 10,000× (confirming we can run at all); projected overhead with accelerators ≤ 1,000×.

### Phase 3: Sharding composition (4-6 months)

**Deliverable:** answer to Q3. Either a working sharded FHE prototype or a published result saying sharded FHE is not viable under tested conditions.

**Gate to Phase 4:** sharding result establishes which consumer workloads FHE could plausibly serve.

### Phase 4: R2/R7 composition + stack positioning (3-4 months)

**Deliverable:** answer to Q4. Composition matrix. Recommendation on whether FHE graduates to R1-engineering or stays as a permanent research watch item.

**Gate to promotion:** at least one composition row on the Pareto frontier shows a customer-addressable overhead bound (per §3 thresholds) for a specific workload class. If no row qualifies, R1 stays research-only.

---

## 6. Promotion triggers (the "why fund R1 now" test)

R1 moves from "watch item" to "funded Phase-1" when **at least two** of the following five triggers fire:

### T1: Hardware acceleration ships

Intel HERACLES, Cornami ASIC, or an equivalent FHE-specific accelerator reaches documented availability at a throughput that projects ≤500× overhead on a transformer-block-scale benchmark. Today's state (April 2026): in development; public benchmarks not yet available.

### T2: Academic breakthrough on scheme overhead

A peer-reviewed publication demonstrates ≤1,000× overhead on a full transformer forward pass at ≥1B-param scale without hardware acceleration. Today: best known is roughly 10,000-20,000× on sub-1B models.

### T3: Regulated-industry partner demand

A concrete Prismatica commissioned-dataset partner (healthcare genomics, financial modeling, legal discovery) expresses formal interest in an FHE tier at a price point consistent with §3's 100× commercial threshold. "Expresses formal interest" means RFP submission or letter of intent, not casual inquiry.

### T4: Threshold-FHE maturation (shared with R5)

Zama, Duality, Inpher, or equivalent publishes a production-ready threshold-FHE deployment. This triggers BOTH R1 and R5 (R5 uses threshold FHE for storage hardening; R1 could benefit from threshold FHE for sharded inference per Q3). See §8.

### T5: Competitor ships FHE inference

A peer private-compute platform (iExec, Phala, Oasis) announces FHE inference as a production tier. This is a market-positioning trigger: not sufficient on its own, but a meaningful signal when combined with any of T1-T4.

**Review cadence.** Triggers reviewed annually (2027-04-22, 2028-04-22, 2029-04-22). Any quarterly trigger firing also reopens the review out-of-cycle.

---

## 7. Relationship to R2, R5, R7

- **R2 MPC (SCOPING-1).** R2 and R1 together describe the privacy-strength spectrum beyond TEE. R2 is "earlier + cheaper + requires non-collusion assumption." R1 is "later + more expensive + no non-collusion assumption needed." If R2 ships and consistently satisfies consumer needs, R1 demand may stay low — the R5 nation-state / critical-infrastructure segment is the principal R1 justification beyond R2.

- **R5 Tier C hardening (SCOPING-1).** R1 and R5 share threshold-FHE as a primitive. Q3 of R1 and Q2 of R5 are both "can threshold FHE do X under our architecture?" A joint partner engagement is plausible and budget-advantageous — same research group can explore both applications. See §8.

- **R7 KV-cache compression (BENCH-1).** R7 Phase-3 output (compression scheme baseline) is a composition input to R1 Phase-4. If R7 establishes that aggressive compression is compatible with PRSM's streaming inference path, R1 can scope FHE-on-compressed-ciphertexts, which materially reduces overhead.

- **R3 threat model (TM-1).** R3 defines the A1/A5/A6 attacks (activation inversion, TEE-bypass, adversarial-operator coalitions) that TEE-only does not fully close. R1, if it ships, would close those paths definitively; R3's methodology would be re-run against the R1 prototype to verify.

---

## 8. Shared budget profile with R5

R1 and R5 share primitives (threshold FHE, underlying BFV/CKKS/TFHE schemes). A coordinated research engagement is cheaper than funding them independently:

| Funding pattern | Est. R1-only cost | Est. R5-only cost | Combined |
|-----------------|-------------------|-------------------|----------|
| Independent engagements | $500k-$1M | $300k-$600k | $800k-$1.6M |
| Joint partner (same group covers both) | — | — | **$600k-$1M** |

**Budget advantage:** $200k-$600k saved through shared scheme selection (Q1 of both specs), shared Phase-1 landscape report, shared partner onboarding.

**Operational implication:** when R1 triggers fire, the scoping-to-funding conversation should include R5 as a natural companion. Partner profile (§9) reflects this.

---

## 9. Partner profile + handoff package

**Partner profile:** cryptography research group (academic or industrial research lab) with:
- Published work on FHE for neural networks, explicitly at transformer scale (not just CNN/MLP).
- Familiarity with at least one of CKKS, TFHE, or BGV scheme families beyond toy-problem usage.
- Implementation capability — able to modify an existing FHE library, not just write papers.
- Ideally, overlapping interest in threshold FHE (to unlock the shared-engagement economics of §8).

**Day-1 handoff package:**
- This scoping doc (R1-SCOPING-1) + R5-SCOPING-1 + R3-TM-1.
- `PRSM_Vision.md` §7 — honest-limits framing.
- `Prismatica_Vision.md` §3.3 — commercial-tier economics.
- Phase 2 Rings 7-10 implementation summary + tensor-parallel sharding spec.
- Relevant compiled artifacts (model-shard structure, inference dispatch traces) so partners can see real PRSM workloads, not abstracted descriptions.

**First-milestone target:** Phase 1 deliverable (§5) in 6 months.

---

## 10. Risk register

### R1-risk-1: Triggers never fire

R1 stays research-only indefinitely. This is a non-risk — it's the expected outcome if FHE does not mature within the horizon. The cost is only the annual review effort.

### R1-risk-2: Triggers fire but Phase 2 fails the ≤10,000× gate

Current-state prototype cannot clear even the basic feasibility gate. R1 stops at Phase 2 with a "not viable at today's state" published result. This is also acceptable — a confident no is publishable.

### R1-risk-3: Partner engagement over-runs

R1 is the largest research item; burn can escalate. Mitigation: fixed-milestone deliverables gate continued funding. No Phase-N funding without Phase-(N−1) acceptance.

### R1-risk-4: Scheme selection locks early

Q1's answer commits to a scheme family; a later competing scheme could obsolete Phase-2 investment. Mitigation: Phase 1 shortlist is 1-2 schemes (not 1), Phase 2 prototypes the top pick but Phase 3 can pivot if needed.

### R1-risk-5: FHE hype cycle distorts evaluation

Market noise around FHE tends toward either over-promise ("FHE solves everything") or over-dismissal ("10,000× forever"). Trigger conditions (§6) are specified to be observable events, not narrative; the review uses evidence.

---

## 11. Success criteria

R1 is considered successful if, at the end of Phase 4:

1. **Answer to "is FHE viable for PRSM" is published** — either "yes at threshold T and conditions C" or "no until triggers X and Y fire," not "maybe."
2. **If yes, at least one workload class has a scoped engineering-promotion path** with overhead within the §3 commercial threshold.
3. **Published results contribute to the open FHE-for-ML literature** — PRSM's permissionless-supply framing is distinctive enough that research output is of external academic value, even if PRSM itself doesn't ship.

If (1) and (2) are "yes + workload identified," R1 closes with a named Phase-4.x engineering-scoping sub-project. If (1) is "no," R1 closes with a review-trigger update documenting what would need to change.

---

## 12. Changelog

- **0.1 (2026-04-22):** initial scoping doc. Promotes R1 from research stub to partner-handoff-ready scope. Execution pending trigger firing + budget allocation + partner identification.
