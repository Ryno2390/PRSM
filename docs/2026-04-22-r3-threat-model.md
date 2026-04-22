# R3 Threat Model + Red-Team Methodology: Activation-Inversion under PRSM

**Document identifier:** R3-TM-1
**Version:** 0.1 Draft
**Status:** Preregistered threat model + red-team methodology. Execution is a research quarter; this document is the spec an academic partner or PRSM research contractor would pick up and run.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Promotes:** `docs/2026-04-14-phase4plus-research-track.md` §R3 from stub to threat-model spec + methodology.
**Related documents:**
- `docs/2026-04-14-phase4plus-research-track.md` §R3 — original research stub.
- `docs/2026-04-22-r7-benchmark-plan.md` §9 — R7's H5 red-team gate consumes this methodology.
- `docs/2026-04-12-phase2-remote-compute-plan.md` §Line-item-B (topology randomization) + §Line-item-C (TEE attestation) — defenses currently in the stack.
- Phase 2 Rings 7-10 (tensor-parallel model sharding + Ring 9 DP noise) — architecture R3 attacks and defends against.
- `PRSM_Vision.md` §7 "Honest limits" — vision-doc framing of activation-inversion concerns.

---

## 1. Purpose

The R3 stub asked: *"How strong are activation-inversion attacks against production-scale transformer inference under PRSM's specific threat model (single malicious node holding one tensor-parallel shard of one layer, without access to other shards or the model's later layers)?"*

The stub noted that Phase 2 Line Item B (topology randomization) is a first-order mitigation and asked whether it is *sufficient* or whether stronger measures (activation-layer noise injection, secure aggregation, differential-privacy guarantees) are needed.

R3 has two deliverables, and this document specifies both:

1. **A PRSM-specific threat model** — what an attacker with a compromised SPRK can actually see and do under PRSM's architecture (§2-§4). Prior activation-inversion literature typically assumes an attacker with full-gradient access at training time; PRSM's threat model is narrower and requires explicit re-characterization.
2. **A red-team methodology** — a reproducible protocol for evaluating attacks under this threat model (§6-§9), plus a mapping from attacks to defense-stack elements (§5) so a red-team result tells us which defense element to strengthen.

R3 is **blocking for R7** because R7's H5 hypothesis (quantization does not compound with activation-inversion surface) requires exactly the red-team methodology specified here. R7's timeline therefore depends on R3 methodology being available in time. This document is the dependency.

---

## 2. PRSM-specific threat model

### 2.1 What the attacker is

A malicious operator who has claimed one or more SPRK positions in a PRSM tensor-parallel inference. The attacker has been routed to by the MarketplaceOrchestrator and their SPRK is actively processing activations.

### 2.2 What the attacker controls

- **Their SPRK's runtime.** Full read/write access to their own SPRK's memory, including activation tensors entering and leaving their shard.
- **Their node's network interface.** Can capture all bytes arriving from + departing to peer SPRKs.
- **Their claimed stake.** An attacker willing to forfeit stake can sustain malicious behavior through a single slash event; sustained malice burns stake at the rate the §5.4 defense activates.
- **Timing + ordering of their own responses.** Can strategically delay, reorder, or refuse to respond, potentially to correlate activation patterns with upstream request structure.

### 2.3 What the attacker does NOT control

- **Other SPRKs' memory.** By PRSM's architecture, each SPRK holds one tensor-parallel shard of one layer. No SPRK sees the full hidden state.
- **Model weights they were not assigned.** Shard assignment is part of dispatch; attackers without that shard can't read its weights.
- **The requester's prompt, directly.** Prompts live on the requester node. The attacker sees only derived activations.
- **The model's output.** Final-layer outputs route back through the requester's aggregator, not through the attacker unless they're specifically on the output path.
- **Topology decisions before dispatch.** Phase 2 Line B randomizes topology per-inference; the attacker doesn't know which other nodes are participating in any particular inference before they receive activations.
- **TEE-sealed attestation roots** (when Phase 2 Line C is active on a given inference).

### 2.4 Attacker goals (ranked by severity)

| Goal | Severity | Description |
|---|---|---|
| G1: recover full prompt | HIGH | reconstruct the requester's original input text with high fidelity |
| G2: recover prompt fragments | HIGH | reconstruct substrings, topics, or characteristic tokens (names, numbers, code identifiers) |
| G3: characterize prompt class | MEDIUM | infer the domain, language, or task category of the prompt without reconstructing content |
| G4: recover model outputs | HIGH | reconstruct the model's response — equivalent to stealing the answer |
| G5: leak training-set membership | LOW | infer whether specific data was in the model's training corpus |
| G6: poison downstream inference | MEDIUM | perturb activations in ways that steer the final output toward attacker-chosen outcomes |

G1-G4 are privacy violations; G6 is integrity. This document primarily concerns G1-G4 because Phase 2 Rings 7-10 handles integrity via the receipt-signing + consensus layer (Phase 3.1 + 7 + 7.1x).

### 2.5 Attacker scale

Three scales:

- **S1 — one shard, one layer, single inference.** The narrowest case the R3 stub calls out: "single malicious node holding one tensor-parallel shard of one layer." Attacker sees one activation tensor at one pipeline stage, on one request.
- **S2 — one shard across many inferences.** Same attacker, same shard position, but runs their SPRK for weeks and builds a corpus. Topology randomization means different inferences see different peers; corpus analysis may still identify patterns.
- **S3 — multi-shard coordinated attack.** Multiple colluding attackers hold shards on different layers / positions. Aggregated view across shards may enable attacks unreachable at S1.

Most published activation-inversion literature is at S1 equivalent. S2 and S3 are the realistic PRSM attack scales and deserve primary attention.

---

## 3. Attack surface enumeration

This is the catalog of activation-inversion attacks adapted to PRSM. Each attack is specified with: target goal (from §2.4), required scale (§2.5), rough effort, and expected defense-stack coverage (§5).

### 3.1 A1: Direct inversion from intermediate activations (Zhu et al. 2019 class)

- **Goal:** G2 (prompt fragments) or G1 if strong.
- **Scale:** S1 minimum.
- **Mechanism:** optimize an input that reproduces the observed activation tensor, then decode as tokens. Works best on early layers where activations are closest to input embeddings.
- **Prior art:** Zhu et al. 2019 "Deep Leakage from Gradients" and follow-ups (Geiping et al. 2020, Yin et al. 2021). Activation inversion is weaker than gradient inversion but follows similar principles.
- **Expected strength against PRSM:** moderate on early-layer shards, weaker on mid-to-late layer shards (post-attention hidden states have lower input-mutual-information).

### 3.2 A2: Embedding-table probe

- **Goal:** G1 on short prompts, G2 on longer.
- **Scale:** S1.
- **Mechanism:** if the attacker holds an early-layer shard that contains the embedding matrix, they can directly look up tokens whose embeddings best match observed activations.
- **Prior art:** this is the "obvious" attack and is usually the first layer attacker targets.
- **Expected strength:** high if attacker holds shard 0 of layer 0; negligible otherwise.
- **Mitigation alignment:** topology randomization (Phase 2 Line B) should prevent repeated shard-0-layer-0 assignment to the same operator.

### 3.3 A3: Cross-inference pattern aggregation

- **Goal:** G3 (prompt class), potentially G2 with enough data.
- **Scale:** S2 (corpus of activations from many inferences).
- **Mechanism:** activations carry characteristic patterns per domain / language / task. With thousands of inferences through the same shard position, an attacker can train a classifier that maps activation patterns to inferred prompt class.
- **Prior art:** this is a direct adaptation of membership-inference attacks in federated learning literature (Shokri et al. 2017 and follow-ups).
- **Expected strength:** high for broad-category inference (English vs non-English, code vs natural language, medical vs legal), lower for specific-content inference.
- **Mitigation alignment:** Ring 9 DP noise degrades cross-inference pattern aggregation; topology randomization complicates corpus building.

### 3.4 A4: Collusion-aggregated inversion

- **Goal:** G1 (full prompt recovery).
- **Scale:** S3 (multi-shard, coordinated).
- **Mechanism:** k colluding attackers hold shards across multiple layers. Aggregated view approximates the full hidden state; inversion is strictly easier than at S1.
- **Prior art:** limited published work at this scale against sharded production LLMs. Opportunity for primary research.
- **Expected strength:** unknown — key question the red-team quarter answers.
- **Mitigation alignment:** R2 (MPC) if ever shipped; for now, topology randomization + economic cost of sustaining k colluding stakes.

### 3.5 A5: Timing side channels

- **Goal:** G3 (prompt class), possibly G2 on high-entropy content.
- **Scale:** S1 or S2.
- **Mechanism:** inference latency varies with prompt characteristics (length, branching in MoE models, cache hit patterns). Attacker with wire-time measurements infers properties of the upstream prompt.
- **Prior art:** classic side-channel literature (Kocher et al.); adapted to LLM serving (recent work on KV-cache timing).
- **Expected strength:** moderate for MoE architectures (routing decisions leak); lower for dense transformers.
- **Mitigation alignment:** time-padding the SPRK dispatch (not currently specified); TEE attestation doesn't cover this.

### 3.6 A6: Fault injection / adaptive-probe attacks

- **Goal:** G6 (poison) primarily; G2 as side effect.
- **Scale:** S1 sufficient.
- **Mechanism:** attacker returns deliberately perturbed activations, observes how other participants respond (retry patterns, error messages, downstream activation changes), and triangulates prompt content from response behavior.
- **Prior art:** ML security literature on adversarial examples; adapted to sharded inference is new.
- **Expected strength:** unknown; likely moderate.
- **Mitigation alignment:** receipt-signing + challenge (Phase 3.1 + 7) catches malicious outputs economically but doesn't prevent the probe.

### 3.7 Adaptive attacks (design meta-note)

Each A1-A6 also has an adaptive variant where the attacker targets the *defenses* themselves — e.g., learns what topology-randomization pattern is in use and works around it, or learns the DP-noise distribution and denoises. The red-team methodology (§7) REQUIRES adaptive variants; non-adaptive-only red-team results overstate defense strength.

---

## 4. Current defense stack inventory

What PRSM already ships (or will ship with merge-ready phases) that mitigates A1-A6:

### 4.1 Phase 2 Line Item B — topology randomization

- **Mechanism:** per-inference randomized assignment of shards to SPRKs. A specific operator won't be repeatedly assigned the same (layer, shard_index) position; over time, the assignment is uniformly random subject to pool availability.
- **Coverage:** dampens A2 (prevents sustained embedding-table probe); dampens A3 (complicates cross-inference corpus); partial against A4 (complicates recruiting the "right" k colluders).
- **Does NOT cover:** A1 (single-inference inversion on a single shard is unaffected by randomization), A5 (timing is independent of assignment), A6 (fault injection is independent).

### 4.2 Phase 2 Line Item C — TEE attestation

- **Mechanism:** inference-granularity attestation using H100 Confidential Compute. Activations stay encrypted in VRAM, enclave is attested.
- **Coverage:** A1 against non-colluding attackers with no physical access to the GPU (reduces the attacker population to cloud-operator-cooperating attackers and side-channel attackers). A2 becomes harder (attacker can't read embedding matrix in plaintext).
- **Does NOT cover:** A3, A4 (if collusion includes the TEE-hosting operator), A5, A6 at the protocol layer; TEE sealing doesn't address timing leakage.

### 4.3 Ring 9 — DP noise injection on specific compute paths

- **Mechanism:** calibrated Gaussian noise added to specific activation paths with a privacy budget.
- **Coverage:** A3 directly (DP bounds the cross-inference pattern leakage); partial A1 at high enough noise (degrades single-inference inversion fidelity).
- **Does NOT cover:** A2 (a single-inference lookup isn't bounded by DP budget if the attacker doesn't aggregate), A6.
- **Cost:** DP noise hurts accuracy; the privacy budget has to be spent carefully.

### 4.4 Phase 3.1 + Phase 7 + Phase 7.1x — economic-layer deterrents

- **Mechanism:** stake + slashing. A provider who participates in an attack risks stake forfeiture if detected.
- **Coverage:** deters rational attackers from A6 (malicious output is slashable via DOUBLE_SPEND / INVALID_SIGNATURE / CONSENSUS_MISMATCH); creates cost-of-sustained-attack for A2 / A3 / A4 at scale.
- **Does NOT cover:** detection of the attack itself. Slashing only works if the red-team evaluation AND the on-chain challenge flow can prove misbehavior; activation-inversion is fundamentally a passive attack and is not on-chain-provable today.

### 4.5 Coverage summary

| Attack | Topology rand (B) | TEE (C) | DP noise (R9) | Staking |
|---|---|---|---|---|
| A1 direct inversion | ❌ | ✅ (non-collud.) | 🟡 (high budget) | — |
| A2 embedding probe | ✅ | ✅ | ❌ | — |
| A3 cross-inference aggregation | ✅ | ❌ | ✅ | — |
| A4 collusion aggregation | 🟡 | 🟡 | ❌ | 🟡 (cost) |
| A5 timing side channels | ❌ | ❌ | ❌ | — |
| A6 fault injection | ❌ | ❌ | ❌ | ✅ (slash) |

The gaps (❌ columns) are the motivation for R3's red-team: quantify how bad the uncovered attacks actually are in practice, and decide whether additional defense layers are warranted.

---

## 5. Mapping red-team results to defense decisions

A well-designed red-team produces decisions, not just findings. Each attack's result maps to a specific defense-stack response:

| Finding | Defense decision |
|---|---|
| A1 high-fidelity inversion on early layers | Add Line B's minimum-layer-distance constraint (operators never hold layer 0) |
| A1 high-fidelity inversion on late layers | Add DP noise to late-layer activations |
| A1 low fidelity across all layers | No additional defense; publish result as privacy guarantee |
| A2 successful despite topology randomization | Randomization pool size is too small; grow operator pool or exclude specific-shard assignments |
| A3 pattern classification at >50% accuracy | Tighten Ring 9 DP budget; add activation-content hashing for further obfuscation |
| A4 collusion attack succeeds at k=2 | R2 (MPC) becomes urgent; stake floor increases for shard-adjacent operators |
| A4 requires k=5+ for success | Economic deterrence is sufficient; track k-collusion cost in ECON-WP |
| A5 timing leakage | Time-pad SPRK dispatch; spec in Phase 2 update |
| A6 fault injection succeeds | Add activation-hash commitments to receipt format (receipt must include hash of activation sent); challenge path extended |

R3's quarterly output is this table with specific attack fidelity numbers and a specific go/no-go per defense decision.

---

## 6. Red-team methodology

### 6.1 Principles

- **Adaptive, not static.** Each attack must include an adaptive variant targeting the defense.
- **Reproducible.** Fixed seeds, published code, results a third party can reproduce.
- **Blind blue team.** The scheme implementer should not assist in attack design; they answer clarifying questions only.
- **Two-rater scoring.** Reconstruction fidelity is judged by at least two reviewers independently, with cosine similarity as the objective metric and qualitative judgment (is this recognizably the original prompt? y/n) as the subjective metric.

### 6.2 Attack phases

**Phase 1 — Baseline characterization.** Run every attack A1-A6 against the baseline Phase 2 + Ring 9 stack (full-precision activations, topology randomization active, DP noise at current budget, TEE on). Record fidelity. This establishes the status-quo attack surface.

**Phase 2 — Per-defense ablation.** Disable one defense at a time; re-run attacks. E.g., disable topology randomization and measure A2's new fidelity. Establishes each defense's per-attack contribution.

**Phase 3 — Adaptive variants.** Red team develops adaptive variants of each attack (attacker knows the defense and targets it specifically). Re-run.

**Phase 4 — Composition with R7.** Once R7 ships TurboQuant quantization behind the flag, re-run every attack against the compressed-activation variant. This is R7-H5's input.

**Phase 5 — Novel attacks.** Red team invents at least one attack not in the A1-A6 catalog, runs it. This is the "unknown unknowns" check.

### 6.3 Fidelity metrics

- **Cosine similarity** between reconstructed activation/prompt and ground truth (for A1, A4).
- **Classification accuracy** on domain inference (for A3).
- **Token-level edit distance** between reconstructed and original prompt (for A1, A2 when prompts can be token-aligned).
- **Subjective recognizability** — blind human raters judge whether reconstructed output is "same topic", "same person/entity named", or "same exact content" as ground truth.

### 6.4 Success criteria for publishing a "PRSM is safe against Ax" claim

For each attack, R3 can publish a positive safety claim (Ax is mitigated to level Y) only if:

- Cosine similarity ≤ 0.3 under adaptive attack, AND
- Classification accuracy ≤ 1.5× baseline random guess, AND
- Token-level edit distance ≥ 0.7 normalized per token, AND
- Blind raters correctly identify ≤ 20% of reconstructions as "same topic."

Any attack failing these bars gets a documented residual-risk claim, not a safety claim.

---

## 7. Red-team methodology — execution spec

### 7.1 Personnel

- **Red team:** 1-2 researchers with published activation-inversion work. Academic partnership preferred. Candidate groups: ETH AI-Lab, Stanford Privacy Group, NYU Tandon Security, CMU CyLab.
- **Blue team:** Phase 2 Rings 7-10 implementer (Foundation staff or contractor). Available for design-clarification but does not participate in attack development.
- **Independent reviewer:** one reviewer outside both teams who sees all raw results before publication. Provides the second rater for §6.1 two-rater scoring.

### 7.2 Reference infrastructure

Mirror R7's §10.1 hardware to enable R3 + R7 co-execution and direct H5 input:

| Tier | Hardware | Purpose |
|---|---|---|
| T4 | 2× H100 SXM NVLink | primary attack benchmarks on Llama-3.1-70B |
| T3 | 1× H100 PCIe | 8B model attacks, cross-cloud parity |
| T2 | RTX 4090 | attacker-side simulation (realistic since many attackers use consumer hardware) |

### 7.3 Reference model + workload

- **Model:** Llama-3.1-8B-Instruct (primary), Llama-3.1-70B-Instruct (frontier check). Same as R7.
- **Prompts:** mixed corpus of (a) LongBench-V1 prompts (reuse from R7), (b) synthetic prompts with known entities / numbers / code to enable exact-match fidelity scoring, (c) real anonymized PRSM dispatch logs (§5.4 from R7).

### 7.4 Timeline

- **Phase 1 baseline:** weeks 1-3.
- **Phase 2 per-defense ablation:** weeks 4-6.
- **Phase 3 adaptive variants:** weeks 7-10.
- **Phase 4 R7 composition (requires R7 Phase 4 delivery):** weeks 11-13, contingent on R7 schedule.
- **Phase 5 novel attacks:** weeks 14-16.

Total: 16 weeks, co-executable with R7's 16-week plan. Weeks 11-13 are the tight coupling with R7's Phase 4 integration.

### 7.5 Deliverables

1. **R3-RESULTS-1 results document** — full attack catalog with fidelity numbers per phase, per attack, per scale.
2. **Code repository** — attack implementations + reproduction scripts (published post-remediation if any attacks succeed).
3. **Decision table** — populated version of §5's table with specific go/no-go per defense.
4. **R7 H5 input memo** — directly answers whether R7's TurboQuant scheme changes the activation-inversion surface.
5. **Academic publication (optional)** — novel-attack findings from Phase 5 are publication-quality; partner org owns first-author credit.

---

## 8. Composition with R7 (H5 dependency)

R7's H5 asks: *"quantization does not compound with activation-inversion attack surface."*

R3 delivers the evaluation methodology for that claim. The precise coupling:

- R3's Phase 1 (baseline characterization) gives R7 a baseline inversion-fidelity table that R7's Phase 4 compares against.
- R3's Phase 4 (composition with R7) is the actual H5 measurement — R7 runs its compressed stack through R3's attack suite.
- R3's success criteria (§6.4) is the R7 H5 pass criterion: adaptive-attack fidelity ≤ 0.3 cosine similarity on the compressed activations.

Scheduling implication: R7 cannot resolve H5 until R3 is at least at Phase 3 (adaptive variants) — that's week 10 of R3. R7 Phase 5 (its own red-team) starts at R7 week 14. The 4-week gap allows R3 to hand off methodology + adaptive attacks to R7's red-team via a synchronous workshop at the R3 week-10 / R7 week-14 boundary.

**Decision:** R3 and R7 should be budgeted together. A single research partner org running both is the simplest coordination model.

---

## 9. Budget implications

Preliminary estimate:

- **Personnel:** red team (1-2 researchers × 16 weeks) + independent reviewer (0.25 FTE × 4 weeks) + blue-team clarification time (~0.1 FTE of Phase 2 implementer throughout).
- **Hardware:** shared with R7 if co-executed; ~10% additional GPU-time for R3-specific attack experiments.
- **Rough total:** $100k-$200k depending on academic-partner model. Added to R7's $150k-$300k gives $250k-$500k for the combined R3+R7 research quarter.
- **Foundation budget impact:** non-trivial but within the scale of a single Foundation R&D line. Justified by the clarity gain: R3+R7 together either unlock the T1/T2 viability story (bandwidth reduction without privacy regression) or cleanly identify the specific defense to build next.

---

## 10. Open issues

### 10.1 S3 (multi-shard collusion) attack scale is under-studied

No published activation-inversion literature evaluates S3 at production transformer scale. R3 Phase 5 (novel attacks) is likely where the S3 attack is developed, meaning the §3.4 A4 expected-strength row is genuinely unknown until R3 Phase 5 results land. This is the single largest unknown in the threat model.

### 10.2 Fault-injection (A6) is dual-purpose

A6 primarily targets integrity (G6) but also leaks privacy (G2) as a side effect. Defense stacks for G6 (receipt signing + challenge) don't automatically cover G2. R3 Phase 3 should explicitly test the G2 side of A6 since existing defenses were designed for G6.

### 10.3 Ring 9 DP budget is not a per-attack tunable

Ring 9's DP noise has a privacy budget set at deployment time. It wasn't designed with specific attacks in mind. Once R3 Phase 2 results are available, the DP budget may need to be re-calibrated to the attack surface — which is itself a Ring 9 v2 deliverable outside R3 scope.

### 10.4 Timing defenses are not specified anywhere in the current stack

§5's A5 row has no current defense (column entries are all ❌). If R3 Phase 1 shows A5 is a meaningful attack, Phase 2 needs a new Line Item (time-padding on SPRK dispatch). This would expand R3 scope from "evaluate existing defenses" to "specify a new defense." Deferred to R3 Phase 2 results before scoping.

### 10.5 Academic-partner NDA and publication balance

A Foundation-funded red team that finds a novel attack against PRSM has an obvious incentive to publish (academic reputation). Publishing before Foundation remediates is a reputational risk. The standard approach is a coordinated-disclosure agreement with an embargo period (90-120 days), but this needs negotiating with the specific partner org.

---

## 11. Cross-references

### 11.1 To other research tracks

- **R2 (MPC for sharded inference).** If R3 Phase 5 shows A4 (collusion) is strong at low k, R2 becomes urgent. R3 results directly set R2's prioritization.
- **R7 (compression).** H5 dependency detailed in §8.
- **R8 (anti-exfiltration).** R8's activation-inversion concern is the same attack surface R3 characterizes. R3 results are a direct input to R8's defense-stack composition analysis.

### 11.2 To engineering

- **Phase 2 Line Item B** — R3 validates or invalidates topology randomization's sufficiency.
- **Phase 2 Line Item C** — R3 measures TEE's residual risk once non-TEE attacks (A3, A5) are characterized.
- **Ring 9 DP noise** — R3 informs budget calibration.

### 11.3 To strategy

- **PRSM_Vision.md §7** — the "Honest limits" activation-inversion disclosure is tightened by R3 results (either "minimal residual risk" or "specific attack x fidelity y residual risk").
- **Privacy-tier messaging** — any R3 Phase 1 baseline fidelity >0.3 means PRSM cannot claim privacy parity with TEE-only centralized inference. The privacy-tier story needs reframing if so.

---

## 12. Ratification

This document does not require governance ratification; it is a research methodology, not a protocol standard. Ratifies implicitly upon:

1. Foundation allocates budget (§9) and identifies execution team.
2. Execution team reviews, files adjustments, commits to the scope.
3. Phase 1 kickoff.

---

## 13. Adjustments after execution starts

*(Empty; populated during execution.)*

---

## 14. Changelog

- **0.1 (2026-04-22):** initial draft, founder-authored. Promoted from R3 research stub. Ready for Foundation budget review and partner identification. Designed for co-execution with R7-BENCH-1.
