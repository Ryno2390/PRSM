# R5 Scoping Doc: Tier C Content-Confidentiality Hardening Beyond Shamir + AES

**Document identifier:** R5-SCOPING-1
**Version:** 0.1 Draft
**Status:** Research scoping doc, not an execution plan. The Phase-7 Tier-C design (AES-256-GCM + Reed-Solomon erasure coding + Shamir-split keys, per `docs/2026-04-22-phase7-storage-design-plan.md`) is correct, shippable, and serves the vast majority of regulated-industry use cases. This document scopes the research path to strengthen it further for threat models where computational-security is insufficient: nation-state adversaries, adversarial-foundation scenarios, critical-infrastructure content.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Promotes:** `docs/2026-04-14-phase4plus-research-track.md` §R5 from stub to scoping doc.
**Related documents:**
- `docs/2026-04-14-phase4plus-research-track.md` §R5 — original research stub.
- `docs/2026-04-22-phase7-storage-design-plan.md` — Phase 7 storage + content-confidentiality plan. Defines the Tier C baseline R5 hardens beyond.
- `docs/2026-04-22-r1-fhe-inference-scoping-doc.md` — R1-SCOPING-1. Threshold-FHE primitives overlap; shared engagement in §7.
- `docs/2026-04-22-r3-threat-model.md` — R3-TM-1. Attack scales S1-S3 parameterize when R5's stronger guarantees are worth paying for.
- `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — PRSM-SUPPLY-1. Non-collusion quantification (used below) reuses SUPPLY-1's diversity metrics.
- `PRSM_Vision.md` §2 (tiered content model) + §7 (honest limits) — vision-doc framing.

---

## 1. Purpose

The R5 stub asked: *"Can Tier C confidentiality be strengthened beyond the Phase 7 design to provide guarantees that hold even against a majority-colluding adversary in the right positions?"*

This document clarifies:

1. **The threat Tier C leaves open.** Phase 7 Tier C is secure under the assumption that no adversary simultaneously obtains (a) K-of-N erasure-coded shard fragments AND (b) M-of-N key-shares. Under economic-cost-of-collusion analysis this is strong for most use cases. For nation-state and adversarial-foundation threat models it is not unconditional. §3 specifies the gap.
2. **What "hardened Tier C" means concretely.** Not a single replacement — a menu of four candidate mechanisms (threshold FHE, secure aggregation with information-theoretic sum, entropy-bounded storage, functional encryption) each of which closes a different slice of the remaining threat surface.
3. **Scheduling and dependencies.** R5 is scheduled AFTER Phase 7 ships AND after R1 Phase 1 (FHE scheme selection) completes. Running R5 earlier is premature — there is no Tier C baseline to measure hardening against, and the threshold-FHE primitives R5 would use depend on R1's scheme-family decision.
4. **Partner profile + handoff.** R5 shares the R1 partner pipeline. §7 specifies the joint-engagement economics.

**R5 execution is moderate-sized** — roughly 2-3 research quarters depending on partner and which of the §4 candidate mechanisms is selected for prototyping. This is research scope, not a commitment to ship.

---

## 2. Non-goals

- **Not a replacement for the Phase 7 Tier C design.** Phase 7 Tier C ships first and is not blocked on R5. R5 is a potential future upgrade path, not a gating prerequisite.
- **Not intended for all Tier C workloads.** The overhead of information-theoretic primitives is high enough that only the most extreme threat models justify paying it. Most Tier C consumers stay on the Phase 7 baseline indefinitely.
- **Not a generalized "better encryption."** Tier C hardening is specifically about removing the reliance on computational-hardness assumptions for the subset of the threat model where that matters.
- **Not tied to a specific primitive.** The four candidates in §4 are not ranked; the research chooses.

---

## 3. What Phase 7 Tier C leaves open

Phase 7 Tier C design:

```
Content ─► AES-256-GCM encrypt ─► Reed-Solomon (K=6, N=10)
                │                              │
                ▼                              ▼
        Shamir secret share       Encrypted shard distribution
        (M=3, N=5) of key                (10 storage nodes)
                │                              │
                └─────── Two-threshold reconstruction ─────────┘
                         (need K shards AND M shares)
```

**Guaranteed under:** adversary obtains < K shards OR < M shares. That covers ~all realistic adversaries below nation-state scale.

**Not guaranteed under:**

- **A. Nation-state compelled disclosure.** An adversary with subpoena / compelled-assistance authority over multiple jurisdictions can plausibly obtain both ≥ K shards and ≥ M key-shares. PRSM's supply-diversity standard (PRSM-SUPPLY-1) caps geographic concentration; it does not make a legal process impossible. The computational AES is unbreakable, but the keys can be legally seized.

- **B. Foundation compromise.** Tier C trusts the foundation / key-management-service that holds one of the M shares. A compromise of that party's cryptographic hygiene (e.g., server key exfiltration) downgrades the M-of-N threshold by 1. If M=3 drops to effective M=2, Shamir is near-brittle.

- **C. Future quantum break of AES.** AES-256-GCM is quantum-secure under Grover's algorithm against the 128-bit post-quantum security level. That's still strong but it is a reduction, not preservation. For 20-50-year archival content (medical genomics, legal records), quantum speed-ups matter.

- **D. Adaptive / rushing adversary.** An adversary who compromises storage nodes sequentially (obtain one, learn metadata, target the next) is not bounded in the Phase 7 design by anything stronger than the K threshold. Information-theoretic schemes with "non-malleability across rounds" would close this.

**R5's scope:** close A, B, C, and/or D with mechanisms whose guarantee does NOT reduce to computational hardness alone.

---

## 4. Candidate mechanisms

Four directions, scored on (a) which of A/B/C/D it closes, (b) current state of the art, (c) integration cost with Phase 7.

### 4.1 Threshold Fully Homomorphic Encryption (threshold FHE)

**Idea.** Key is itself an FHE ciphertext; decryption requires M-of-N participants to run a joint decryption protocol. No single party ever sees the decryption key in cleartext.

**Closes.** A (no jurisdictional seize target — the key is distributed), B (foundation compromise is insufficient: needs M-of-N). Does NOT close C (still computational).

**State of the art.** Zama concrete-threshold-FHE (2024-2025), Inpher's threshold-BFV, Duality schemes. Production maturity: demonstrations at small-to-medium scale; full-document encryption workflows published but not at PRSM's expected throughput.

**Integration cost.** High for encryption (replaces AES in the pipeline). Moderate for key-share distribution (replaces Shamir). Overlaps strongly with R1 — see §7.

### 4.2 Secure Aggregation with Information-Theoretic Sum

**Idea.** Storage nodes hold cryptographically-masked shards. Recovery computes a sum across K nodes via a single-round secure aggregation protocol. The sum mathematically yields the plaintext only if K is reached AND all K cooperate.

**Closes.** A (legal seize must cross K participants). Partially closes D (masks do not compose across rounds — an adversary compromising nodes sequentially observes independent masks each round, not accumulated information).

**State of the art.** Federated-learning secure aggregation (Bonawitz et al. 2017, follow-ups through 2024) is production-tested at Google-scale. Direct application to content-storage is less mature but the primitives transfer.

**Integration cost.** Low-to-moderate for Phase 7 integration. Shard-side computation is the new cost; consumer-side decryption is simpler than current Shamir combine.

### 4.3 Entropy-Bounded Storage Schemes (All-or-Nothing Transform variants)

**Idea.** Encrypt content such that recovering any strict subset of ciphertext yields ZERO information about plaintext (not just computationally hard — information-theoretically zero). Rivest's 1997 AONT and modern extensions (DAON, Enrollment-based AONT) formalize this.

**Closes.** D definitively (any round-wise partial compromise leaks nothing). Partially closes A (but since the K-of-N threshold is preserved, a sufficient compelled disclosure still succeeds — the gain is tighter bounds on what "sufficient" means).

**State of the art.** Academic, with implementations. Full production deployment in content-storage contexts is limited.

**Integration cost.** Low — AONT can be applied as a pre-processor before the Phase 7 erasure-coding step without reworking the rest of the pipeline.

### 4.4 Functional Encryption with Delegatable Decryption

**Idea.** Keys can be delegated to perform only specific operations (e.g., "verify this content without retrieving it," "retrieve this for this account only"). Limits the blast radius of any single compromised key.

**Closes.** B (compromise of a delegated key is bounded by its scope — cannot recover the full plaintext if the key only permits verification).

**State of the art.** Academic, with small-scale deployments. Function-hiding variants are still research.

**Integration cost.** Moderate — requires rearchitecting the key-distribution contract (Phase 7's KeyDistribution.sol) to issue function-scoped keys rather than raw secret shares.

---

## 5. Preregistered research questions

### Q1: Which candidate mechanism gives the best closure of A-D per unit cost?

**How we'd answer:** scoring matrix across the four candidates. Each candidate scored on (closes-A, closes-B, closes-C, closes-D) ∈ {yes, partial, no} + (current-maturity, integration-cost). Recommend one or two for Phase-2 prototyping.

**Expected answer shape:** a ranked shortlist. Precommit: threshold FHE and/or AONT are the most likely picks — threshold FHE for its breadth of closure (A+B), AONT for its low integration cost and D-closure.

### Q2: Can threshold FHE be applied to PRSM Tier C key release in place of Shamir secret sharing, and at what operational cost?

**How we'd answer:** prototype threshold-FHE-based key release for a small example (5-10 consumer retrievals, 5 nodes, M=3). Measure: (a) decryption protocol latency, (b) per-retrieval overhead vs. Shamir combine, (c) key-rotation cost, (d) failure-recovery cost when one of the M participants is offline.

**Expected answer shape:** per-retrieval overhead envelope + operational-failure playbook + recommendation on whether threshold FHE is the right primitive for Tier C at PRSM's expected throughput.

### Q3: Does AONT pre-processing compose with Reed-Solomon without a quality penalty?

**How we'd answer:** test AONT-then-RS on realistic content sizes (1 MB to 1 GB). Measure: (a) end-to-end encryption + erasure time, (b) storage overhead (AONT typically adds ~O(log N) extra bytes — bounded, but verify), (c) decryption / decoding time with K of N shards.

**Expected answer shape:** measurements + go/no-go on whether AONT can be added to the Phase 7 pipeline without pipeline-level restructuring.

### Q4: What is the quantum resistance profile of each candidate?

**How we'd answer:** cross-reference each candidate against NIST PQC recommendations. Threshold FHE inherits scheme-level PQ properties (lattice-based schemes: yes; pre-lattice CKKS variants: depends). AONT is quantum-neutral — the information-theoretic property is unaffected by quantum compute. Secure aggregation inherits from underlying MAC/commitment scheme.

**Expected answer shape:** per-candidate PQ-score. Informs whether R5's output should be combined with R6's post-quantum migration (see §7).

### Q5: When should R5 graduate to engineering (promotion triggers)?

Specified in §6. Graduation criterion: one candidate mechanism is demonstrated to close at least one of A-D at overhead acceptable for a specific named customer class.

---

## 6. Promotion triggers (the "why fund R5 now" test)

R5 moves from "watch item" to "funded Phase-1" when **at least two** of the following fire:

### T1: Named regulated-industry customer demand

A concrete Prismatica commissioned-dataset partner (defense, critical infrastructure, high-sensitivity healthcare, high-sensitivity legal) expresses formal interest (RFP, LOI) in a stronger-than-Tier-C privacy tier. This is the principal trigger — research is driven by addressable demand, not speculation.

### T2: Threshold-FHE production maturity

Zama, Duality, Inpher, or equivalent publishes a production-ready threshold-FHE deployment at scale (≥ 1GB content, ≥ 10 participants, documented latency). Shared with R1 (see §7).

### T3: Jurisdictional / compliance change

A regulatory change (HIPAA evolution, EU Data Act interpretation, new sector-specific mandate) raises the confidentiality bar for Tier C's target customer classes to a level Phase 7 cannot unconditionally meet. PRSM-SUPPLY-1's reviewers flag this through quarterly compliance updates.

### T4: Academic-literature breakthrough

A peer-reviewed publication demonstrates a practical candidate mechanism (per §4) at PRSM-relevant scale (document-level, not MNIST-level). "Practical" means sub-10× overhead vs Phase 7 baseline.

### T5: R1 Phase 1 output indicates threshold-FHE maturity

R1's Phase 1 scheme-selection report concludes threshold FHE is viable at acceptable overhead for inference. This transitively indicates viability for R5's Q2. Triggers joint-engagement conversation per §7.

**Review cadence.** Triggers reviewed annually (2027-04-22, 2028-04-22, 2029-04-22). Out-of-cycle re-review on any T1 event (named customer request).

---

## 7. Relationship to R1, R3, R6

- **R1 FHE-for-inference (SCOPING-1).** R1 and R5 share threshold-FHE as a primitive candidate. Shared scheme-selection (R1 Q1 ≈ R5 Q1 restricted to threshold-FHE shortlist). Budget sharing per R1 §8 — R1+R5 joint engagement saves $200k-$600k over independent engagements. Scheduling: R5 Phase 1 can start in parallel with R1 Phase 2 IF the joint partner is engaged; otherwise R5 waits for R1 Phase 1 report.

- **R3 threat model (TM-1).** R3 defines attack scales S1 (single adversarial node), S2 (small coalition), S3 (multi-shard coordinated attack, approaching nation-state). R5 directly closes R3's S3-scale attacks against Tier C. R5 Phase 2 validation methodology re-uses R3's red-team framework.

- **R6 post-quantum signatures (WATCH-1).** R6 is about signatures; R5 is about confidentiality. Independent, BUT both push PRSM's stack toward post-quantum readiness. If R6 triggers fire around the same time as R5 T2 (threshold-FHE maturity), a combined migration strategy is more efficient than sequential.

---

## 8. Phased execution plan

**Scheduling note.** R5 Phase 1 starts AFTER Phase 7 ships (earliest Q3 2028 per Phase 7 target) AND after R1 Phase 1 completes OR a joint-engagement partner is confirmed. Earliest viable: Q1 2029. Latest acceptable: before regulated-industry contract expansion that expects Tier-C+ service.

### Phase 1: Candidate selection + landscape report (3-4 months)

**Deliverable:** answer to Q1. Scoring matrix + shortlist of 1-2 candidate mechanisms. Joint engagement with R1 Phase 1 if partner available.

**Gate to Phase 2:** shortlisted candidate has enough public implementation + documentation to prototype against.

### Phase 2: Prototype on Tier C subset (5-7 months)

**Deliverable:** answer to Q2 (threshold FHE) and/or Q3 (AONT), depending on shortlist. Working prototype replacing one layer of Phase 7 Tier C (key release OR pre-processing) with the candidate mechanism. Measured overhead vs baseline.

**Gate to Phase 3:** prototype closes at least one of A/B/C/D from §3 at bounded cost.

### Phase 3: Composition + PQ analysis + customer handoff (3-4 months)

**Deliverable:** answers to Q4 + Q5. Final report scoping the engineering-promotion path to a Phase-7.x hardening upgrade targeting a specific named customer class. If no acceptable path exists, R5 closes with a "not ready; review triggers" published result.

**Gate to promotion:** at least one customer-class-scoped path has a go decision from the named customer (T1 trigger partner) backed by budget.

---

## 9. Partner profile + handoff package

**Partner profile:** same as R1 — cryptography research group with:
- Published work on threshold schemes or all-or-nothing transforms at production scale.
- Implementation capability beyond paper-level.
- Willingness to engage on both R1 and R5 for budget-shared scope (§7).

**Day-1 handoff package:**
- This scoping doc (R5-SCOPING-1) + R1-SCOPING-1 + R3-TM-1.
- `docs/2026-04-22-phase7-storage-design-plan.md` — the Tier C baseline.
- `PRSM_Vision.md` §2 + §7.
- `Prismatica_Vision.md` §3.3 + §2.6 — commercial-tier economics + clean-rooms context.
- Named customer-class use case (once T1 fires) with the confidentiality threshold that customer needs.

**First-milestone target:** Phase 1 deliverable (§8) in 4 months.

---

## 10. Risk register

### R5-risk-1: Triggers never fire

R5 stays research-only indefinitely. Not a risk — current Tier C is sufficient for all addressable customers. Annual review cost only.

### R5-risk-2: Named customer demand materializes but no candidate mechanism closes their specific threat

Customer wants a guarantee none of the four §4 candidates provides. R5 Phase 1 reports "not viable with current state of the art," and the customer either accepts current Tier C with caveats or looks elsewhere. Published result remains valuable.

### R5-risk-3: R1 and R5 partner conflict

A single partner is engaged for joint R1+R5 work and delays one side. Mitigation: contract structure separates R1 and R5 milestone acceptance; either can stand alone if the other stalls.

### R5-risk-4: Overhead exceeds customer budget

Phase 2 prototype demonstrates closure of A/B/C/D but at a cost the addressable customer cannot pay. Output: "viable but uneconomic today; retriggers on hardware acceleration." Same review cadence applies.

### R5-risk-5: Post-quantum retrofit conflict

A threshold-FHE candidate is selected in Phase 1, then R6 triggers fire and post-quantum migration requires a different underlying scheme. Mitigation: Q4 (PQ analysis) bakes PQ-compatibility into selection, so Phase 2's choice is not a dead-end if R6 fires later.

---

## 11. Success criteria

R5 is considered successful at end of Phase 3 if:

1. **Answer to "can Tier C be hardened beyond computational assumptions for PRSM" is published** — yes with mechanism M at cost C, or no at today's state of the art.
2. **If yes, at least one customer-class-scoped engineering-promotion path** is identified with budget + customer buy-in.
3. **Published output contributes to the storage + confidentiality research literature** — PRSM's permissionless + erasure-coded + Tier C composition is distinctive enough that the research is of external value.

---

## 12. Changelog

- **0.1 (2026-04-22):** initial scoping doc. Promotes R5 from research stub to partner-handoff-ready scope. Execution pending Phase 7 ship + R1 Phase 1 OR joint engagement + trigger firing + budget allocation. Closes the partner-handoff sweep across the research track — R1-R8 all now at scoping-doc, threat-model, watch-memo, benchmark-plan, or promoted-standard status; the research track is hand-offable without further drafting.
