# R8 Defense-Stack Composition Analysis: Anti-Exfiltration for Frontier-Model Inference

**Document identifier:** R8-COMP-1
**Version:** 0.1 Draft
**Status:** Architectural composition analysis. Not an execution plan — R8 spans multiple research tracks (R2, R3, R7) and multi-year engineering (PRSM-CIS-1 silicon standard). This doc specifies how the pieces fit together, which combinations close which attack chains, and what the minimum-sufficient stack for frontier-lab acceptance looks like.
**Date:** 2026-04-22
**Drafting authority:** PRSM founder
**Promotes:** `docs/2026-04-14-phase4plus-research-track.md` §R8 from stub to composition analysis.
**Related documents:**
- `docs/2026-04-14-phase4plus-research-track.md` §R8 — research stub.
- `docs/2026-04-21-prsm-cis-1-confidential-inference-silicon.md` — PRSM-CIS-1, the silicon standard that provides one of R8's five defense layers.
- `docs/2026-04-22-r2-mpc-scoping-doc.md` — R2 scoping. R2's MPC scheme IS R8's cryptographic weight-sharding defense layer (same primitive, different name).
- `docs/2026-04-22-r3-threat-model.md` — R3 threat model. R8's "side channels" sub-threat is a strict superset of R3's A5 timing-attack vector.
- `docs/2026-04-22-r7-benchmark-plan.md` — R7 compression. R7's H5 tests whether compression strengthens or weakens the attack surface R8 defends.

---

## 1. Purpose

R8's stub framed the question clearly: what is the defense-stack composition sufficient for a frontier AI lab to publish SOTA weights to PRSM's permissionless meganode network?

What the stub did NOT specify:
- Which combinations of the five proposed defense layers actually close which attack chains.
- Which layers are load-bearing (required in every realistic stack) vs optional (strengthening specific sub-threats).
- What the minimum-sufficient stack looks like when (a) all layers mature, (b) only some mature by the time a frontier lab asks.
- How the five layers interact with the existing research tracks (R2, R3, R7) that also produce defense primitives.

This document fills that gap. It is the integration map.

**Not in scope:** specifying any individual layer. Each layer has its own primary document:
- TEE attestation layer: Phase 2 Line C (already shipped at merge-ready).
- Cryptographic weight sharding: R2-SCOPING-1 (MPC is the mechanism).
- Weight fingerprinting: specified as open research in §3.3 below.
- Output watermarking: specified as open research in §3.4 below.
- Custom silicon: PRSM-CIS-1 (already published standard draft).

R8-COMP-1's deliverable is the composition logic, not the layer designs.

---

## 2. Threat model (attack chain level)

R8's stub named the attack chain. Here we formalize it as a sequence of attacker actions, each of which a defense layer might block. This yields the attack-chain-vs-defense-layer matrix in §4.

### 2.1 Primary attack chain: W+δ re-monetization

Attacker is a meganode operator running inference for model `M` with weights `W`. The chain:

1. **Exfiltrate.** Capture `W` via GPU-VRAM dump / side-channel / invasive physical attack / cold-boot / firmware compromise.
2. **Perturb.** Apply small perturbation `δ` (LoRA-sized diff, quantization roundtrip, random noise) chosen to evade the publisher's fingerprinting while preserving quality.
3. **Re-register.** Upload `W+δ` to PRSM's `ProvenanceRegistry` as a new model `M'`.
4. **Re-monetize.** Set per-M'-token price `F' < F`, capture demand that would have gone to the publisher.
5. **Defect.** Operator has now externalized the publisher's $500M+ training cost.

Each step is a candidate blocking point. An attack chain must complete every step to succeed; any defense that breaks any step breaks the chain.

### 2.2 Adversary classes

Four adversary classes at increasing capability, matching PRSM-CIS-1's §4.1:

- **A-L1 — opportunistic operator.** No specialized equipment, no side-channel expertise. Tries public exfil techniques from papers. Blocked by commodity TEE.
- **A-L2 — funded attacker.** Commercial side-channel equipment (power-analysis rigs), 1-2 security researchers for weeks. Blocks: probably-not blocked by commodity TEE alone at scale; dedicated silicon (C3) + fingerprinting may block.
- **A-L3 — nation-state-level.** Full lab, physical-invasion equipment (de-cap + probing), years of dedicated effort. Full defense stack (TEE + MPC + fingerprinting + watermarking + dedicated silicon) probably-blocked at high cost.
- **A-L4 — vendor-colluding.** Cloud provider cooperates with attacker. This is the *unmitigated* threat under current H100 Confidential Compute because the cloud provider owns the physical infrastructure. Only dedicated PRSM silicon (CIS-1 C3) blocks this.

### 2.3 Sub-threats per step

Each attack chain step decomposes into multiple sub-threats. A defense-layer matrix tracks these.

| Step | Sub-threat | Example |
|---|---|---|
| 1 Exfiltrate | 1a VRAM dump (plaintext weights in device memory) | `cuda-memcheck`-style probe |
| | 1b Memory-bus side channel | DDR probing, Rowhammer, cold-boot |
| | 1c Power analysis | DPA / CPA on crypto ops |
| | 1d Physical de-cap | fab-level invasive analysis |
| | 1e Activation-inversion (indirect) | reconstruct weights from many activations |
| | 1f TEE firmware / attestation CA compromise | vendor backdoor |
| 2 Perturb | 2a LoRA-style additive delta | low-rank fine-tune |
| | 2b Quantization roundtrip | fp16→int8→fp16 |
| | 2c Random weight noise | Gaussian ε on all params |
| | 2d Targeted-delta-to-evade-fingerprint | trained adversarial delta |
| 3 Re-register | 3a ProvenanceRegistry has no clone check | direct upload |
| | 3b Fingerprint-hash evasion | hash differs after delta |
| 4 Re-monetize | 4a Price undercut | operate at F' < F |
| | 4b Claim the undercut is on "efficiency" | obfuscate source |
| 5 Defect | 5a No economic penalty on successful theft | attacker keeps revenue |
| | 5b Reputational insulation | attacker is anonymous wallet |

R8's five-layer defense stack exists to break at least one of these sub-threats per step. A complete defense closes every step's sub-threats or at least makes them uneconomical.

---

## 3. Defense layers

Five layers in the R8 stub's taxonomy. Each specified at the "what it blocks / what it doesn't" level.

### 3.1 Layer L1: TEE attestation

- **Mechanism:** H100 Confidential Compute or equivalent. Weights stay encrypted in VRAM, runtime is attested, attestation chains to NVIDIA's root CA.
- **Primary source:** Phase 2 Line C (already shipped at merge-ready).
- **Blocks:** 1a (encrypted VRAM), 1b partial (memory encryption mitigates bus probes against A-L1), 2b (roundtrip happens on plaintext weights the attacker doesn't have).
- **Does not block:** 1c (power analysis — orthogonal), 1d (physical invasion — beats commodity TEE), 1e (activation-inversion — R3 scope), 1f (vendor compromise — unmitigated under current attestation model), **1* under A-L4 (vendor collusion) — TEE is useless if the attacker IS the cloud provider**.
- **Current state:** shipped; the attestation CA is NVIDIA's, which is the unmitigated centralization risk.

### 3.2 Layer L2: Cryptographic weight sharding (MPC)

- **Mechanism:** split `W` across k operators under an MPC protocol; no operator ever holds the full weight matrix.
- **Primary source:** R2-SCOPING-1 (MPC as the mechanism).
- **Blocks:** 1a structurally (no operator has plaintext full weights to dump), 1b-1d for partial shards (reconstructing `W` from k-1 shards is infeasible under MPC's secret-sharing assumption), 1e partial (activation-inversion still sees activations, not weights — R3 addresses this separately).
- **Does not block:** 1e (activation-level leakage), 1f (vendor compromise if vendor provides k's hosting), 3b (fingerprinting evasion — orthogonal to weight access).
- **Current state:** R2 is research-scoping; no implementation. Earliest realistic availability: 2027-2028 conditional on R2 execution.
- **Composition note:** MPC overhead is 10-100× on full inference; R2's first-N-layers milestone brings this to ~3-4× average overhead across a Llama-3.1-8B. Usable for high-value inference, not universal.

### 3.3 Layer L3: Weight fingerprinting

- **Mechanism:** publisher embeds a cryptographic fingerprint in their model weights at training / release time. PRSM's `ProvenanceRegistry` runs a fingerprint check on every `registerContent`; rejects registrations where the fingerprint is present (indicating stolen + perturbed `W+δ`).
- **Primary source:** open research. Candidate approaches:
  - Output-distribution hashes on a fixed probe set (publisher specifies probe prompts; registry verifies output hashes match).
  - Layer-wise singular-value signatures.
  - Trained classifiers on weight statistics.
  - Random-basis rotation fingerprints (known to publisher's secret; preserved under small δ).
- **Blocks:** 3a (registry check blocks direct-upload clone), 3b partial (fingerprinting scheme must be robust to small-δ perturbations — this is the open research problem).
- **Does not block:** 1* (exfiltration itself — fingerprinting is post-hoc), 2d (a sufficiently-sophisticated adversarial delta may evade any particular fingerprint scheme).
- **Current state:** open research. No production-robust scheme against adaptive perturbation attackers exists in the literature. Academic partnership required.
- **Unique property:** L3 works *without* changing how inference happens. Publisher can adopt L3 unilaterally; no operator cooperation required. This makes L3 the single highest-leverage layer for frontier-lab persuasion.

### 3.4 Layer L4: Output watermarking

- **Mechanism:** publisher embeds a cryptographically-keyed watermark in the model's outputs. Third parties can detect watermark presence without the secret; the key stays with the publisher. A detected watermark on an output from model `M'` but under publisher's watermark key proves `M'` is derived from the publisher's `M`.
- **Primary source:** open research. Prior work includes Google SynthID (images), Kirchenbauer et al. (LLMs with input-independent watermarks), recent adversarial-robust LLM watermarking papers.
- **Blocks:** 4a detection (undercut price + watermarked outputs → watermark detection → publisher has evidence for §5.3 economic-layer slash), 5a partial (if slash mechanism exists).
- **Does not block:** 1* (exfiltration), 2* (perturbation — though perturbation may damage watermark, making output quality decline).
- **Current state:** production-robust LLM watermarking against adaptive attackers is not solved. Kirchenbauer-line work is evadable. Research-active.
- **Unique property:** L4 is *post-hoc detection*. Works even if L1-L3 all failed; the only requirement is that someone is generating outputs from a derived model for PRSM users.

### 3.5 Layer L5: Custom silicon (PRSM-CIS-1)

- **Mechanism:** chip-level confidentiality anchored in PRSM's own attestation root. Tamper mesh, active shield, per-chip HRoT, no vendor-CA dependency.
- **Primary source:** PRSM-CIS-1 (published silicon standard draft).
- **Blocks:** 1a architecturally, 1b physical-invasion-resistant, 1c power-analysis-hardened, 1d de-cap-resistant with tamper mesh, 1f (no vendor CA to compromise — PRSM's attestation anchors to its own ProvenanceRegistry), **1* under A-L4 (vendor collusion) — this is the only layer that blocks the vendor-colluding threat**.
- **Does not block:** 1e (activation-inversion is still orthogonal to hardware confidentiality), 3b (fingerprinting evasion).
- **Current state:** CIS-1 standard draft published; no fab has committed to C2/C3-compliant silicon yet. 3-5 year engineering arc from first-committed-fab to deployed hardware. Three conformance levels (C1 baseline, C2 frontier-acceptable, C3 current-SOTA-acceptable).
- **Unique property:** L5 is the only layer that addresses A-L4 (vendor collusion). Everything else assumes the vendor is neutral.

---

## 4. Defense-layer vs attack-sub-threat composition

Matrix of which layers block which sub-threats. `✅` = blocks; `🟡` = partial / probabilistic block; `❌` = does not block; `—` = not applicable.

| Sub-threat | L1 TEE | L2 MPC | L3 FP | L4 WM | L5 CIS-1 | Best combo for this sub-threat |
|---|---|---|---|---|---|---|
| 1a VRAM dump | ✅ | ✅ | ❌ | ❌ | ✅ | L1 or L2 or L5 |
| 1b memory-bus side channel | 🟡 | ✅ | ❌ | ❌ | ✅ | L2 or L5 |
| 1c power analysis | ❌ | 🟡 | ❌ | ❌ | ✅ | L5 required |
| 1d physical de-cap | ❌ | ❌ | ❌ | ❌ | ✅ | L5 required |
| 1e activation-inversion | ❌ | 🟡 | ❌ | ❌ | ❌ | R3 + R7 combined |
| 1f vendor CA compromise | ❌ | 🟡 | ❌ | ❌ | ✅ | L5 required |
| 2a-2c perturbation (generic) | ❌ | ❌ | 🟡 | 🟡 | ❌ | L3 + L4 (robust variants) |
| 2d adversarial-delta | ❌ | ❌ | 🟡 | 🟡 | ❌ | L3 + L4 + red-team verification |
| 3a clone registration | ❌ | ❌ | ✅ | ❌ | ❌ | L3 required |
| 3b fingerprint evasion | ❌ | ❌ | 🟡 | 🟡 | ❌ | L3 (robust) + L4 as backstop |
| 4a price undercut | — | — | — | — | — | Detectable via L4 + econ-layer |
| 4b source obfuscation | ❌ | ❌ | ❌ | ✅ | ❌ | L4 required |
| 5a no economic penalty | — | — | — | — | — | Phase 7 slashing (separate) |
| 5b anonymous attacker | ❌ | ❌ | ❌ | 🟡 | ❌ | Operator KYC (out of R8 scope) |

Key observations:

- **L5 (CIS-1) is the only layer that fully addresses A-L4 vendor collusion.** Every other layer assumes the hardware vendor isn't the adversary.
- **L3 (fingerprinting) is the only layer that blocks 3a clone registration.** A ProvenanceRegistry without L3 has no defense against direct-upload clones regardless of what else is shipped.
- **L2 + L5 together block the physical-exfiltration chain** (1a-1d). Neither alone suffices under A-L3 (nation-state).
- **No layer alone covers 1e activation-inversion.** R3 (characterization) + R7 (compression as partial mitigation) close this gap outside the R8 stack proper.
- **L4 (watermarking) is the only post-hoc detection layer.** If L1-L3-L5 all fail, L4 still produces evidence for downstream economic action.

---

## 5. Minimum-sufficient stacks per adversary class

Different frontier-lab partners have different threat tolerance. A-L1 threshold (opportunistic attackers) is much lower than A-L3 (nation-state). Three scenarios:

### 5.1 "First frontier lab" scenario (opportunistic + funded attackers, A-L1 + A-L2)

Target: a tier-1 frontier lab agrees to publish a SOTA model to PRSM under a *pilot* arrangement, given defenses sufficient against A-L1 and A-L2 but explicitly NOT A-L3 or A-L4.

**Minimum stack:** L1 (already shipped) + L3 (fingerprinting) + L4 (watermarking) + economic-layer slashing.

- Why L1 alone isn't enough: A-L2 can defeat commodity TEE at scale (side channels, cold-boot on DDR).
- Why L3 is required: blocks 3a clone registration. Without it, perturbation-and-reupload is trivial.
- Why L4 is required: post-hoc detection if L1+L3 both fail.
- Why L2 and L5 are NOT required: A-L3 and A-L4 are explicitly out-of-scope for pilot.

**Timeline:** L3 and L4 are the blocking deliverables. Both are research-active; both lack production-robust schemes. Estimated 2-4 quarters of focused research per layer before a minimal-viable implementation ships. **Earliest realistic pilot: late 2027.**

### 5.2 "Mainstream frontier lab" scenario (adds A-L3)

Target: second+ frontier lab adoption post-pilot. Nation-state attackers become a realistic concern (PRSM's decentralized supply means any actor can run a meganode).

**Minimum stack:** pilot stack + L2 (MPC sharding) OR L5 (CIS-1 C2 silicon). Pick whichever matures first.

- L2 on its own blocks A-L3's exfiltration attempts but not L5's coverage gaps (vendor CA dependency remains).
- L5 (C2) on its own blocks A-L3 including vendor-CA concerns (silicon is PRSM-anchored, not NVIDIA-anchored).
- **L2 is likely earlier** (R2's scoping doc targets late 2027 first milestone; CIS-1 C2 silicon is 2028+ best case).

**Path recommendation:** plan on L2 as the intermediate step, L5 (C2) as the long-term replacement. An L5 arrival later makes L2 for tier-2 workloads and L5 for tier-3 (highest-value) workloads.

### 5.3 "SOTA exclusive" scenario (adds A-L4)

Target: a frontier lab publishes weights for their CURRENT STATE-OF-THE-ART model (not a generation-behind release).

**Minimum stack:** full stack — L1 + L2 + L3 + L4 + L5 (C3 silicon, not just C2).

- A-L4 (vendor collusion) is load-bearing here. Only L5 C3 blocks it, and L5 C3 requires PRSM-specific silicon, not a profile of commodity silicon.
- L2 remains required because L5 + vendor-collusion-free doesn't mean L2-redundant; L2 adds shards-per-operator whereas L5 adds trust-base change.
- Full stack reduces residual risk to, essentially, "a perfect-coordination k-collusion across geographically and economically diverse operators holding CIS-1 C3 hardware" — which PRSM-SUPPLY-1's diversity mechanisms make exponentially expensive.

**Timeline:** CIS-1 C3 is 2030+ best case. SOTA-exclusive scenario is a multi-year destination, not a near-term target.

---

## 6. Cross-track integration

R8 isn't independent research — it's the integration target that the other research tracks feed into. Each track contributes a specific layer or sub-layer of R8's stack.

### 6.1 R2 (MPC) → R8 L2

R2's MPC scheme is L2. No R8-specific work required beyond recognizing this overlap; R2's deliverable IS R8 L2's specification.

### 6.2 R3 (activation-inversion) → R8 1e coverage

R3 characterizes sub-threat 1e (activation-inversion). R3's deliverables determine:

- Whether 1e is a meaningful residual threat given current R9 DP noise + topology randomization.
- Whether new defenses (e.g., MPC on activation tensors in addition to weights) are required.
- Red-team methodology R8 can reuse to verify any new L3/L4 scheme is robust.

R8's minimum-sufficient-stack analysis assumes R3 has reduced 1e to an "acceptable-residual-risk" state via R2+R7 composition. If R3 shows otherwise, the stacks in §5 need to add dedicated 1e defenses.

### 6.3 R7 (compression) → R8 1e partial mitigation + L2 cost reduction

R7 hypothesizes that compressed activations reduce inversion fidelity (H5). If H5 passes, R7 contributes to 1e coverage. R7 also reduces L2's bandwidth overhead (§6 of R2-SCOPING-1), making L2 T3-deployable sooner.

### 6.4 R6 (PQ signatures) → L3/L4 future-proofing

L3 fingerprint schemes and L4 watermarking schemes will rely on signature primitives. When R6 triggers migration, L3/L4 schemes must migrate too. R6's recommendation to add optional PQ-signature headroom in receipt format (§5.1 of R6-WATCH-1) extends to fingerprint/watermark formats.

### 6.5 PRSM-CIS-1 → R8 L5

CIS-1 IS L5. No bridging work required beyond tracking CIS-1's conformance-level roadmap.

### 6.6 Cross-track summary

```
R8 Layers:
  L1 TEE          ← Phase 2 Line C (shipped)
  L2 MPC sharding ← R2-SCOPING-1 (research-scoping)
  L3 Fingerprint  ← open research (needs commissioning)
  L4 Watermark    ← open research (needs commissioning)
  L5 CIS-1        ← PRSM-CIS-1 (published standard, pending fab)

Cross-layer support:
  1e coverage  ← R3 (characterization) + R7 (compression)
  L2 viability ← R7 (reduces L2 bandwidth cost)
  L3/L4 crypto ← R6 (PQ migration readiness)
```

---

## 7. Governance and ecosystem-coordination

R8's composition requires ecosystem coordination well beyond PRSM's solo development. The 2026-04-19 governance split (from the R8 stub) is the structural foundation:

### 7.1 Foundation owns

- The silicon standard (PRSM-CIS-1 + its conformance process).
- The on-chain fingerprint-check mechanism for `ProvenanceRegistry` (L3 enforcement).
- The watermark-detection oracle (L4 enforcement).
- The defense-stack composition requirements per publisher tier (i.e., "to publish at tier X, you need stack Y").
- The research agenda that commissions L3 and L4 work.

### 7.2 Prismatica (first implementer) contributes

- First T4 meganodes conforming to CIS-1 (initially C2, target C3).
- Reference implementations of L3/L4 as they land.
- Economic runway for early-stage deployment before ecosystem matures.

### 7.3 Other ecosystem actors needed

- **Foundation-funded academic partnerships** for L3 + L4 research. Candidate orgs: Stanford / CMU / Cornell / ETH / Oxford ML-security groups.
- **Fab / chip designer commitments** for CIS-1 C2 and C3 implementations. Foundation-led outreach required.
- **Frontier lab engagement** (the actual customer). Foundation-led partnership conversations conditioned on stack maturity.

### 7.4 Failure mode if governance slips

If PRSM ships L3/L4 schemes without rigorous robustness-review, frontier labs will not trust them. A prematurely-claimed "defense" that turns out to be evadable is strictly worse than no defense — it creates false confidence. Governance must commit to a **red-team-before-claim** discipline:

- No L3 scheme is marked "production-ready" until it passes an adaptive-attacker red team (R3 methodology) with ≤10% evasion rate.
- No L4 scheme is marked "production-ready" until similar robustness testing against perturbation + output-filtering attacks.

This discipline is the difference between R8 being real and R8 being security theater.

---

## 8. Decision framework: when to commission L3 + L4 research

Of the five layers, L1 and L5 have ongoing primary-source development (Phase 2 Line C; PRSM-CIS-1). L2 has R2 as its research-scoping doc. L3 and L4 have neither. Those are the unsponsored layers.

### 8.1 Why L3 is highest-leverage-per-dollar today

- L3 has **unilateral publisher adoption** (§3.3 unique property): a frontier lab can fingerprint their own model and register with PRSM's `ProvenanceRegistry` without any operator cooperation.
- L3 closes 3a (clone registration) which is TODAY an open gap in PRSM's ProvenanceRegistry; no other layer addresses this.
- L3 is research-active but not research-heavy: a Foundation-funded academic partnership could produce a production-robust scheme in 2-4 quarters.

**R8's highest-leverage recommendation: commission L3 fingerprinting research as the first R8-direct-investment.** Budget estimate: $150k-$300k academic partnership.

### 8.2 Why L4 is second-highest-leverage

- L4 is post-hoc detection — the backstop that works when other layers fail.
- L4 composes with the Phase 7 slashing economic layer: watermark-detected clone → stake slash.
- L4 research is more mature than L3 research (several candidate schemes; Kirchenbauer line + SynthID-like approaches).
- L4 is operator-transparent — no SPRK-level changes needed.

**R8 recommendation:** commission L4 in parallel with L3, ~$150k-$250k.

### 8.3 Why commissioning NOW matters (even before frontier-lab ask)

A frontier lab engaging PRSM for a pilot will ask "what defenses do you have?" The answer today is "L1 + L5 standard + research plans for L2/L3/L4." If L3 and L4 are *not actively in research*, the answer loses credibility. Having commissioned research with named partner orgs and a clear timeline is persuasive in a way that "defined as open research" isn't.

### 8.4 Combined R8-research budget

- L3 partnership: $150k-$300k, 2-4 quarters.
- L4 partnership: $150k-$250k, 2-3 quarters.
- R8 integration + composition testing (post-R3/R7/R2 outputs): $100k, 1 quarter.
- Total R8-direct: $400k-$650k.
- **Combined R2+R3+R7+R8 research program:** $950k-$1.65M over 6-8 quarters.

---

## 9. Partner handoff

### 9.1 Partner profiles

- **L3 fingerprinting:** ML-security academic group with model-provenance or deep-learning watermarking publications. Candidate orgs: UMich (Hsieh-line), UIUC, MIT (Madry group), ETH (Vechev group).
- **L4 watermarking:** ML-security group with generative-model watermarking expertise. Candidate orgs: UMd (Kirchenbauer's group), Google DeepMind (SynthID team), CMU (Fredrikson/Papernot orbit).
- **Silicon partners (L5):** fab + chip designer commitments for CIS-1. Separate procurement track.

### 9.2 Day 1 materials

- This composition analysis doc.
- PRSM-CIS-1 standard (for L5 context).
- R2-SCOPING-1 (for L2 composition context).
- R3-TM-1 (for red-team methodology).
- R7-BENCH-1 (for composition dependency).
- Phase 2 Line C TEE attestation architecture (for L1 context).
- `ProvenanceRegistry` contract interface (for L3 integration target).

### 9.3 Milestone cadence

- **L3 + L4 quarter 1:** scheme proposal + initial robustness experiments.
- **Quarter 2:** adversarial red-team (R3 methodology) against quarter-1 schemes.
- **Quarter 3:** scheme refinement based on red-team findings.
- **Quarter 4 (partially parallel with quarter 3):** PRSM integration prototype; pilot-stack assembly (§5.1).

### 9.4 Publication policy

Novel L3/L4 schemes are publication-worthy. Partner orgs retain first-author rights; Foundation acknowledged as funder. Embargo period for novel-attack findings from red-team work (90-120 days) per R3 §10.5 precedent.

---

## 10. Open issues

### 10.1 L3 robustness definition

"Production-robust" per §7.4 is ≤10% evasion rate. This bar is approximate; actual threshold depends on the economic attack model (how many evasion attempts a rational attacker would pay to make). PRSM-ECON-WP-2 should model this explicitly.

### 10.2 L4 quality-preservation bound

Watermarks degrade output quality. Frontier labs will not accept ≥5% benchmark-quality loss. This is a hard constraint on L4 scheme choice.

### 10.3 Stack composability under partial failure

§5's stacks assume each layer works *as designed*. What happens when L3 is adopted but the scheme turns out to be 30% evadable? Composition analysis should model partial-failure modes and answer questions like "does L4 compensate for L3 at evasion rate X?"

### 10.4 Frontier-lab engagement protocol

The §5 scenarios are hypotheticals. Actual frontier-lab engagement process — how PRSM approaches a lab, what conversations cover, who signs what — is not specified anywhere. Foundation business-development function is out of R8 scope but is the consumer of R8's composition analysis.

### 10.5 Revocation / incident response

If an L3 or L4 scheme is broken in the wild (novel attack published), what is the emergency process? Stack re-certification? Temporary freeze on affected publisher models? PRSM-GOV-1 §9.4 emergency amendments is the process layer; R8's incident-response playbook is TBD.

### 10.6 Non-adversarial derivation cases

A legitimate fine-tune of a base model (e.g., an MCP-style domain-adaptation) is technically W+δ. L3/L4 schemes must distinguish legitimate derivations from theft. This likely requires publisher-signed derivation licenses on-chain, and is a non-trivial design addition.

---

## 11. Cross-references

### 11.1 To other research tracks

- **R1 FHE.** Long-horizon superset of L2 privacy guarantees; not a near-term contributor to R8.
- **R2 MPC.** Contributes L2. Direct integration; R2's deliverable IS L2's specification.
- **R3 activation-inversion.** Covers sub-threat 1e outside the five-layer stack; its red-team methodology is the L3/L4 verification mechanism.
- **R5 Tier C hardening.** Threshold FHE / information-theoretic weight sharding would be an L2 successor under strongest threat models (A-L4 at scale).
- **R6 PQ signatures.** L3/L4 cryptographic primitives must migrate when R6 triggers.
- **R7 compression.** Reduces L2 bandwidth overhead; contributes to 1e coverage via H5.

### 11.2 To engineering

- **ProvenanceRegistry contract** — L3 enforcement target. Will need a fingerprint-check hook.
- **Phase 2 Line C TEE** — L1 primary source, already shipped.
- **PRSM-CIS-1 conformance registry** — L5 enforcement mechanism (registry specifies which silicon is C1/C2/C3-certified).
- **Phase 7 slashing economics** — enforcement downstream of L4 detection.

### 11.3 To strategy

- **PRSM_Vision.md §7 "Honest limits"** — R8's composition tightens the vision doc's anti-exfiltration section from "this is how we'd address it" to specific scenario-matched stacks.
- **PRSM-TOK-1 FTNS demand model** — frontier-lab adoption is a major FTNS demand driver; R8's timeline affects TOK-1 demand projections.
- **Launch UX thesis** — R8 is explicitly NOT a launch-viability lever. Launch is carried by open-weights models on Phase 2 Rings 7-10 + L1. R8 matters for expanding the addressable market to frontier-proprietary-weight hosting, a later phase.

---

## 12. Ratification

This document does not require governance ratification; it is an architectural analysis. Partner org identification and budget allocation under PRSM-GOV-1 §4.6 are Foundation decisions that do consume governance bandwidth.

Ratifies implicitly upon:
1. Foundation reviews and accepts the composition analysis.
2. Foundation allocates budget for L3 + L4 research commissioning (§8.4).
3. Partner orgs identified and engaged.
4. Cross-track dependencies (R2 scheduling, R3/R7 Phase timing) coordinated with R8 milestones.

---

## 13. Changelog

- **0.1 (2026-04-22):** initial draft, founder-authored. Promoted from R8 stub to composition analysis. Bridges PRSM-CIS-1 silicon standard with the five-layer defense stack and cross-track dependencies (R2, R3, R7). Recommends L3 + L4 as highest-leverage immediate R8-direct research investments.
