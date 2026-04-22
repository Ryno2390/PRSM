# Phase 4+ Research Track Stub

**Status:** Research roadmap placeholder. Not engineering work. Captured 2026-04-14 so Vision-doc-derived research items are tracked.

**Context:** Several items referenced in `PRSM_Vision.md` are explicitly positioned as research roadmap, not product. This document prevents them from being forgotten, without committing to a delivery schedule.

## R1: Fully Homomorphic Encryption (FHE) for Private Inference

**Vision doc reference:** Section 7, "Honest limits" — *"FHE is currently 10,000-100,000× slower than plaintext inference and not production-ready for frontier-scale models."*

**Research question:** Can FHE-based inference on PRSM become practical within a 3-5 year horizon, given continued improvements in FHE schemes (CKKS, TFHE), hardware acceleration (Intel HERACLES, specialized FHE ASICs in development), and model quantization co-design?

**What would change if this ships:** Eliminates the remaining insider-exposure risk from Section 7. Node holding a model shard could execute inference on encrypted activations without ever decrypting them. Would make PRSM's privacy tier strictly dominant over centralized inference for any use case, not just the PII-sensitive niche.

**Watch signals:** Zama's fhEVM progress, Intel HERACLES availability, TFHE-rs performance benchmarks on LLM workloads, academic papers demonstrating sub-100× FHE inference on transformer models.

**Effort if pursued:** substantial research partnership with a specialized cryptography group. Not a solo project.

## R2: Multi-Party Computation (MPC) for Sharded Inference ⏳ SCOPING DOC DRAFTED

**2026-04-22 update:** promoted from stub to partner-handoff-ready scoping doc at `docs/2026-04-22-r2-mpc-scoping-doc.md` (**R2-SCOPING-1**). Five preregistered research questions (Q1 protocol selection, Q2 MPC-on-first-N-layers scoping, Q3 non-collusion quantification via PRSM-SUPPLY-1 metrics, Q4 bandwidth-latency under 9000× handicap, Q5 composition with R7). MPC-on-first-N-layers framed as realistic first milestone with Puma-line as starting protocol. Scheduled AFTER R3 Phase 1 + R7 Phase 2 complete so N-layer selection + compression scheme are both fixed inputs. 20-week plan, $300k-$500k budget for R2 alone, $550k-$1M for combined R2+R3+R7 program. Partner profile: cryptography research group with transformer-MPC experience.

The framing below is preserved as originating context.

---


**Vision doc reference:** Section 7, "Honest limits" — *"MPC is 10-100× overhead and requires non-colluding node assumptions."*

**Research question:** Can MPC protocols be adapted to PRSM's tensor-sharded architecture such that non-colluding node assumptions are realistic under PRSM's tiered supply (T1 consumer edge, T3 cloud arbitrage, T4 meganodes across independent operators)?

**What would change if this ships:** Intermediate between plaintext (current Section 7 stack) and full FHE. Earlier delivery than FHE; stronger guarantees than TEE-only.

**Specific sub-questions:**
- How is the non-collusion assumption quantified in a permissionless network? Economic stake, geographic diversity, operator-class diversity?
- Does MPC overhead compose acceptably with the 9000× bandwidth handicap between consumer nodes vs. datacenter NVLink?
- Can MPC be scoped to the "sensitive layers" of an inference (first few transformer layers where activation-inversion attacks are effective), leaving later layers in TEE-only mode, to bound overhead?

**Effort if pursued:** moderate — this is closer to integration research than foundational cryptography research.

## R3: Activation-Inversion Attack Characterization & Mitigation ⏳ THREAT MODEL + METHODOLOGY DRAFTED

**2026-04-22 update:** the PRSM-specific threat model, attack catalog (A1-A6 ranked by attacker scale S1-S3), defense-stack coverage matrix, and red-team methodology are specified at `docs/2026-04-22-r3-threat-model.md` (**R3-TM-1**). 16-week execution plan co-designed with R7-BENCH-1 — R3 Phase 4 directly feeds R7's H5 gate. Execution pending Foundation budget allocation and research-partner identification (combined R3+R7 budget ~$250k-$500k).

The research content below is retained as framing context for the threat-model spec.

---


**Vision doc reference:** Section 7, "Honest limits" — *"Activation-inversion attacks (Zhu et al. 2019 and follow-up literature) can partially reconstruct input prompts from early-layer activations."*

**Research question:** How strong are activation-inversion attacks against production-scale transformer inference under PRSM's specific threat model (single malicious node holding one tensor-parallel shard of one layer, without access to other shards or the model's later layers)?

**Current state:** Phase 2 line item B (`docs/2026-04-12-phase2-remote-compute-plan.md` addendum) adds topology randomization as a first-order mitigation. This research track asks whether topology randomization is *sufficient* or whether stronger measures (activation-layer noise injection, secure aggregation across tensor-parallel peers, differential-privacy guarantees) are needed.

**Watch signals:** academic papers on activation inversion at scale, red-team reports from AI safety groups targeting sharded inference systems.

**Effort if pursued:** a quarter of red-team + blue-team work; reportable results.

## R4: Per-Provider Supply Caps and Geographic Diversity Incentives ✅ PROMOTED TO STANDARD DRAFT

**Original vision-doc reference:** Section 6 subsection, "Honest caveats" — *"If T3 dominates supply, PRSM effectively runs on 3-4 hyperscalers underneath. Mitigable via supply caps per provider and geographic-diversity bonuses."*

**Resolution (2026-04-22):** promoted from research stub to governance standard draft at `docs/2026-04-22-prsm-supply-1-supply-diversity-standard.md` — **PRSM-SUPPLY-1**. The standard specifies definitions, measurement, triggers (preserved from this stub: 30% single-provider, 40% single-country, operator concern), soft-cap + diversity-bonus enforcement mechanisms, a 5,000-provider / 36-month bootstrap grace period, and the ratification path under PRSM-GOV-1 §9.

**Remaining research-track disposition:** closed. Follow-up work is now engineering (provider-registry seed list, stake-binding floor values, orchestrator integration) and Foundation treasury (diversity-bonus budget). Neither is research.

## R5: Content Tier C Hardening Against Majority Collusion

**Vision doc reference:** `PRSM_Vision.md` §7 "Honest limits" — *"A colluder who obtains enough erasure-coded fragments (above the K-of-N reconstruction threshold) plus enough key-shares (above the M-of-N threshold) can reconstruct Tier C content."*

**Research question:** Can Tier C confidentiality be strengthened beyond the Phase 7 design, which relies on computational assumptions plus economic-cost-of-collusion, to provide guarantees that hold even against a majority-colluding adversary in the right positions?

**Why this is research, not engineering:** The Phase 7 Tier C design is correct and shippable, and it meets the threat model of the vast majority of regulated-industry use cases. But for the most extreme threat models (nation-state adversaries, adversarial-foundation scenarios, critical-infrastructure use cases), a stronger guarantee would be valuable. The techniques that provide it — threshold FHE, secure aggregation with information-theoretic properties, entropy-bounded storage schemes — are active research areas without production-ready implementations.

**What would change if this ships:** Tier C becomes information-theoretically secure rather than computationally secure for the portion of the threat model it covers. Opens defense, government, and critical-infrastructure use cases that current Tier C can serve only with caveats.

**Watch signals:** academic papers on threshold cryptography at scale; NIST post-quantum secure computation competitions; threshold-FHE deployment milestones from Zama, Duality, Inpher.

**Effort if pursued:** substantial research partnership with a specialized cryptography group. Not a solo project. Overlaps significantly with R1 (FHE for private inference) since many underlying primitives are shared.

## R6: Post-Quantum Signatures ⏳ WATCH MEMO PUBLISHED

**2026-04-22 update:** promoted from stub to a concise trigger-watch memo at `docs/2026-04-22-r6-pq-signatures-watch-memo.md` (**R6-WATCH-1**). The memo documents the current signature surface (three layers — node identity, receipt sigs, on-chain verifier — each with different migration implications), the signature-specific threat model (no harvest-now-forge-later; lead-time-driven rather than urgent), four trigger conditions (one already fired: NIST Aug 2024 finalization of ML-DSA + SLH-DSA), and a 3-phase dual-sign migration playbook when triggers promote. No engineering action today; scheduled re-review on 2027-04-22 or on any trigger firing.

The framing below is preserved as the originating context.

---


## R7: KV Cache & Activation Compression for Consumer-Edge Inference ⏳ BENCHMARK PLAN DRAFTED

**2026-04-22 update:** the "concrete starting hypothesis" has been promoted to a preregistered experimental design at `docs/2026-04-22-r7-benchmark-plan.md` (**R7-BENCH-1**). Five hypotheses with pass/fail/null conditions, five phases over 16 weeks, a partner-handoff checklist, and an R3 red-team integration gate that is explicitly blocking. Execution is pending Foundation budget allocation and research-partner identification.

The research content below is retained as the framing the benchmark plan promotes from.

---


**Vision doc reference:** Section 7 — *"In a datacenter, Megatron-style sharding runs over NVLink at ~900 GB/s. PRSM replaces NVLink with BitTorrent-mediated activation streaming between SPRKs. Orders of magnitude slower..."* Also Section 6, T1/T2 supply tier viability.

**Research question:** Can modern KV-cache and activation-tensor compression techniques (quantization, rotation-based schemes, 1-bit residual corrections) shrink the bandwidth and memory footprint of PRSM's sharded inference enough to make consumer-edge nodes (T1) and prosumer nodes (T2) economically competitive with T3 cloud arbitrage for a meaningful share of workloads?

**Current state.** Phase 2 Rings 7-10 shipped tensor-parallel model sharding with plaintext activation streaming between SPRKs. No compression layer on the activation wire; KV cache lives at native model precision (typically fp16/bf16). The 9000× bandwidth handicap between residential internet and NVLink is the single largest architectural constraint on T1 viability. R2 (MPC for Sharded Inference) treats this as a composition problem with privacy overhead. R7 treats it as a pure efficiency problem — what does aggressive lossy compression buy us, independently of privacy?

**Specific sub-questions:**

1. **Activation streaming compression.** What is the accuracy cost of quantizing inter-stage activations to int8 / fp8 / 4-bit during pipeline-parallel forward passes? Does the answer differ for tensor-parallel all-reduce vs. pipeline-parallel point-to-point transfers? PRSM's activation path is P2P between SPRKs, not NCCL all-reduce — literature mostly reports the latter.

2. **KV cache quantization.** Techniques like KIVI, KVQuant, SmoothQuant-KV, and newer rotation-based schemes (e.g., PolarQuant-style polar-coordinate transforms, Johnson-Lindenstrauss residual corrections) claim 4-6× memory reduction at near-zero accuracy loss on standard benchmarks. What holds up under PRSM's specific deployment: long-context (32k+), sharded across heterogeneous nodes, with per-token streaming output?

3. **Composition with DP noise (Ring 9).** PRSM already adds calibrated differential-privacy noise to specific compute paths. Does quantization noise compose additively with DP noise, or does one mask the other? Can we budget them jointly?

4. **Composition with TEE attestation (Phase 2 line item C).** Does quantizing activations before they leave the enclave change the attestation surface? If quantization happens post-enclave, the plaintext activation still exists in enclave memory; if it happens in-enclave, the enclave must carry the quantization library (attestation surface grows).

5. **Hardware heterogeneity.** PRSM's supply-side includes consumer GPUs (RTX 40-series), Apple Silicon, and AMD/Intel accelerators. Do the quantization schemes that work well on H100 tensor cores degrade on consumer hardware? Apple Silicon's unified memory changes the KV-cache economics significantly.

**Primary-source candidate schemes (verified 2026-04-16 against PDFs):**

A coherent three-paper research lineage from one group (Zandieh + Mirrokni as common authors across all three; collaborators at Google Research, NYU, Yale, KAIST, Adobe, DeepMind) establishes the data-oblivious KV-quantization track:

- **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead** (Zandieh, Daliri, Han — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482), Jul 2024). Foundational primitive. JL random projection followed by 1-bit sign quantization with an *asymmetric* inner-product estimator (quantize keys, keep queries unquantized) yielding unbiased attention-score estimation. Llama-2 KV cache at 3 bits/FPN: no accuracy drop vs. 16-bit baseline, >5× memory reduction, faster runtime than the full-precision baseline. **Public CUDA kernel available at github.com/amirzandieh/QJL** — lowers PRSM's prototyping cost; the primitive can be ported into a research branch rather than reimplemented from the paper.
- **PolarQuant: Quantizing KV Caches with Polar Transformation** (Han, Kacham, Karbasi, Mirrokni, Zandieh — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617), Feb 2025). Random preconditioning + recursive polar transformation (log₂(d) levels). Key property: after preconditioning, polar angles follow an analytically computable tightly-concentrated distribution, so the quantizer needs no per-block zero-point/scale metadata — eliminating the ~1+ bit per quantized number overhead that traditional block quantization pays. ×4.2 KV cache compression at SOTA quality.
- **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** (Zandieh, Daliri, Hadian, Mirrokni — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), Apr 2025). Synthesis: MSE-optimal quantization via random rotation → Beta-distributed coordinates → optimal scalar quantization per coordinate (MSE branch), plus optional 1-bit QJL residual for unbiased inner-product estimation (inner-product branch). Proves information-theoretic near-optimality (within ~2.7× constant of Shannon lower bound; ~1.45× at b=1). On Llama-3.1-8B-Instruct LongBench-V1: 3.5 bits/channel matches full-cache average (50.06 = 50.06); 2.5 bits/channel at 49.44. Needle-in-haystack at 3.5 bits: 0.997 (matches full precision). Per-vector quantization cost ~1.3ms at d=1536.

**Adjacent non-lineage work worth comparing against:**

- **KIVI** (Liu et al., [arXiv:2402.02750](https://arxiv.org/abs/2402.02750), 2024). Tuning-free 2-bit asymmetric KV quantization; incumbent baseline. Lineage papers consistently beat it at comparable or lower bit budgets.
- **KVQuant** (Hooper et al., [arXiv:2401.18079](https://arxiv.org/abs/2401.18079), 2024). 10M-context KV quantization demonstration.
- **QuaRot** (Ashkboos et al., [arXiv:2404.00456](https://arxiv.org/abs/2404.00456), 2024). Rotation-based 4-bit end-to-end inference including activations — closest existing work to activation-streaming quantization. Requires calibration (*not* data-oblivious), which is the cost PRSM cannot easily pay in a permissionless topology.
- **RoTateKV** (Su et al., [arXiv:2501.16383](https://arxiv.org/abs/2501.16383), 2025). 2-bit KV via outlier-aware adaptive rotations; relevant for comparison.

**Why data-obliviousness is load-bearing for PRSM specifically (new insight, 2026-04-16):**

In centralized inference, per-block quantization metadata (zero-point + scale per group of coordinates) is a minor storage overhead kept in the same memory hierarchy as the quantized tensor. In PRSM's sharded setting, per-block metadata would have to *travel between SPRKs on every pipeline-parallel stage handoff* alongside the quantized activations or KV shards. That metadata stream competes directly with the compression gain it is supposed to enable. The QJL → PolarQuant → TurboQuant lineage eliminates this metadata entirely: each SPRK derives the rotation/projection matrix from a shared random seed, independently, with no coordinating communication. This upgrades data-obliviousness from "convenient" in centralized inference to **structurally necessary in distributed PRSM inference**.

**Watch signals:** Vendor releases (TensorRT-LLM, vLLM, SGLang) shipping any of the above as production kernels; Zama/Duality threshold-cryptography work that touches on low-precision activations; red-team publications on activation-inversion under quantization.

**Still unverified as of 2026-04-16:** reports of a "Ramp Latent Briefing" method for inter-agent KV-cache sharing. Primary source not located. Even if real, Latent Briefing is narrow-fit for PRSM (requires shared-model KV between orchestrator and worker; SPRKs are WASM kernels not LLM agents, and cross-vendor MCP handoffs cannot share KV). R7 does not depend on Latent Briefing being real.

**Cross-references to other research tracks:**
- **R2 (MPC for sharded inference).** R7 asks the plaintext-efficiency question; R2 asks the privacy-overhead question. Outputs compose — if R7 shows 4× bandwidth reduction and R2 adds 10× MPC overhead, net is 2.5× overhead vs. today's plaintext streaming, which may be acceptable for Tier B/C content.
- **R3 (activation-inversion attack characterization).** Lossy quantization may *reduce* activation-inversion attack effectiveness by removing signal the adversary would reconstruct from. But this is not guaranteed — quantization can also create exploitable structure. Red-team work required before claiming quantization as a mitigation.

**What would change if this ships.** T1 (consumer-edge) viability expands from "edge inference on small models" to "meaningful contribution to frontier-scale sharded inference." Directly addresses the §6 "Honest caveats" concern that T3 (cloud arbitrage) could come to dominate supply. Reduces PRSM's effective dependency on hyperscaler infrastructure.

**Effort if pursued:** moderate. Closer to benchmarking and integration research than foundational cryptography. Core loop: implement candidate scheme in SPRK activation-streaming path → measure bandwidth / accuracy / latency on held-out workloads → red-team inversion surface → publish results. Estimated quarter of work per candidate scheme; multiple schemes should be compared.

**Concrete starting hypothesis (for the research quarter that opens R7):**

TurboQuant's data-oblivious property makes it the best-fit *starting* candidate for PRSM's permissionless topology. Investigate two applications in parallel:

1. **KV cache quantization on attention-holding SPRKs.** Direct application — paper's primary use case. Prototype using the public QJL CUDA kernel (github.com/amirzandieh/QJL) as the 1-bit primitive, extended with TurboQuant's MSE-optimal scalar quantizer for the MSE branch and/or PolarQuant's recursive polar transform for comparison. Target: replicate Llama-3.1-8B LongBench-V1 results on PRSM's reference models, then measure memory-ceiling impact on T1 (consumer-edge) nodes. Expected outcome: 3.5-5× KV memory reduction enables meaningfully longer context on 16-24GB consumer GPUs and 32GB Apple Silicon unified memory. Deliverable: updated consumer-edge context-length table for the four-tier supply architecture. The shared-seed rotation matrix must be distributed to all SPRKs participating in an inference (include in the job dispatch payload from the `RemoteShardDispatcher`); handshake overhead is O(32 bytes) per inference.

2. **Activation-streaming quantization between pipeline-parallel SPRKs.** Extrapolation — none of the three lineage papers benchmark this. Activation tensors are high-dimensional Euclidean vectors that satisfy TurboQuant's input assumptions, but inter-stage activations have different statistical structure than KV cache (no unit-norm assumption, different distributional properties after LayerNorm, occasional outlier channels). Requires empirical validation. Target: measure bandwidth reduction on forward-pass activation transfers at 3.5-bit and 2.5-bit settings against full-precision baseline, on the Phase 2 reference inference workload. QuaRot is the relevant baseline here (quantizes activations end-to-end but requires calibration; the lineage trades ~1-2% accuracy for zero calibration — potentially the right trade for permissionless PRSM, and the only feasible one given that per-node calibration across churning nodes is impractical). RoTateKV's outlier-aware rotation machinery may need to be adapted if activation outlier channels disproportionately affect reconstruction.

3. **Composition stack.** Both applications must be measured against: (a) Phase 2 topology randomization (line item B) — does per-inference topology change invalidate any cached quantization state? TurboQuant's rotation matrix is random per-quantizer but fixed across vectors; interaction with per-inference topology randomization needs explicit test; (b) Ring 9 DP noise — quantization noise ≈ Gaussian with variance set by bit-width; jointly budgeting with DP noise (rather than composing naively) may be strictly better; (c) R3 activation-inversion attack surface — preliminary intuition says quantization *reduces* inversion fidelity (information-theoretic bottleneck) but adversarial structure in the specific rotation matrix could create new attack surface. Red-team required.

**Framing clarification (2026-04-18):** R7 is a **cost-curve lever, not a launch-viability lever.** Phase 2's launch UX thesis (see `docs/2026-04-12-phase2-remote-compute-plan.md` "Launch UX thesis") is carried by T3 cloud-arbitrage nodes delivering frontier-adjacent latency on day 1. R7 matters when we need to shift the supply mix T3 → T1/T2 to keep the cost basis dropping over 2-5 years, or when T3 capacity approaches the rented-GPU supply ceiling. This removes timeline pressure: R7 can run as genuine research on a research calendar rather than as critical-path engineering.

**Trigger to move to engineering:**
- Research quarter identifies a scheme with ≥3× bandwidth reduction at <1% accuracy loss on PRSM's benchmark suite, with red-team report showing no regression on activation-inversion surface; AND
- Monitoring data shows T1/T2 supply share falling below target and T3 concentration exceeding R4 thresholds — forcing prioritization of T1-viability mechanisms. OR
- Observed T3 capacity approaches the rented-GPU spot/on-demand supply ceiling and prices begin to drift upward, making T1/T2 expansion the obvious cost-control lever.

## R8: Anti-Exfiltration Architecture for Frontier-Model Inference (Added 2026-04-19) ⏳ COMPOSITION ANALYSIS DRAFTED

**2026-04-22 update:** promoted from stub to defense-stack composition analysis at `docs/2026-04-22-r8-defense-stack-composition.md` (**R8-COMP-1**). Formalizes the attack chain as sub-threats, maps each sub-threat against the five defense layers (L1 TEE / L2 MPC / L3 fingerprinting / L4 watermarking / L5 CIS-1 silicon), and specifies minimum-sufficient stacks per adversary class (A-L1 opportunistic through A-L4 vendor-colluding). Identifies L3 + L4 as the unsponsored layers with highest leverage per research dollar (~$400k-$650k to commission both); recommends commissioning now ahead of frontier-lab asks so the answer to "what defenses do you have?" is "actively researching with named partners" rather than "defined as open research." Cross-track integration: R2 contributes L2, R3+R7 contribute 1e coverage, R6 contributes crypto-migration readiness for L3/L4, CIS-1 contributes L5. Combined R2+R3+R7+R8 research program scopes at $950k-$1.65M over 6-8 quarters.

The framing below is preserved as originating context.

---


**Research question:** What combination of hardware confidentiality, cryptographic weight sharding, fingerprint-based clone detection, and output watermarking is sufficient for a frontier AI lab to accept permissionless inference hosting on PRSM — i.e., to publish proprietary SOTA model weights to untrusted meganodes without expecting the weights to be exfiltrated, perturbed to evade provenance checks, and re-monetized by a thief at a lower FTNS rate?

**Threat model:** Meganode operator runs PRSM inference for model M with weights W. During inference, W exists in plaintext in GPU VRAM on hardware the operator physically controls. Attack chain: (1) operator exfiltrates W via memory dump / side channel / VRAM probe; (2) applies perturbation δ below the publisher's fingerprint-detection threshold; (3) uploads W+δ as new model M' to PRSM's ProvenanceRegistry; (4) charges F' < F per M tokens, undercutting the original publisher; (5) captures demand while externalizing the $500M+ training cost. Consequence if undefended: no frontier lab publishes SOTA weights to PRSM, and PRSM remains a market for open-weights models only — a substantial cap on the network's TAM.

This threat does not apply to proprietary inference providers (Anthropic, OpenAI, Google) because operator and model-owner are the same legal entity running on physically-controlled infrastructure. PRSM specifically breaks that coupling — which is the whole point (decentralized supply) and also the whole problem.

**Current state.** Phase 2 Line item C specifies TEE attestation at inference granularity using H100 Confidential Compute or equivalent. This covers ~95% of the naive exfiltration surface (encrypted VRAM, attested runtime, weights sealed to enclave key). Remaining threats: side channels (cache timing, memory-bus probes, Rowhammer, power analysis, cold-boot against DRAM), physical invasive attacks against the SoC, and — critically — the centralization risk of trusting NVIDIA's attestation root.

No current PRSM design addresses: (1) fingerprint-based clone detection in ProvenanceRegistry to block W+δ re-registration, (2) output watermarking for passive post-hoc detection, (3) k-of-n cryptographic weight sharding to raise collusion bars, or (4) hardware confidentiality beyond commodity TEE — i.e., chips whose physical design makes weight exfiltration architecturally impossible rather than merely expensive.

**Specific sub-questions:**

1. **Defense stack composition.** Given five candidate layers (TEE, k-of-n sharding, weight fingerprinting, output watermarking, custom silicon), what subset is (a) sufficient for a first-tier frontier lab to agree to publish SOTA weights, and (b) cost-rational for PRSM to build? Each layer has independent research and engineering cost; composition effects matter.

2. **Robust weight fingerprinting under adversarial perturbation.** Standard model fingerprinting (activation-pattern hashes, weight-hash Merkle trees) is brittle under small perturbations. What fingerprinting scheme tolerates legitimate fine-tunes (LoRA, DPO, quantization) but detects "steal + random perturb" attacks? This is an open problem in the model-provenance literature. Candidate approaches: output-distribution hashes on a fixed probe set, layer-wise singular-value signatures, trained classifiers on weight statistics. All need red-team evaluation against adaptive perturbation attacks.

3. **Output-level watermarking for generative models.** Can PRSM publishers embed cryptographically-keyed watermarks in model outputs that (a) survive small weight perturbations applied post-theft, (b) do not degrade output quality below publisher-acceptable thresholds, (c) are verifiable by third parties without revealing the watermark secret? Prior work: Google's SynthID (images), Kirchenbauer et al. for LLMs (input-independent but degradable). None are production-robust against adaptive attackers yet.

4. **Custom silicon as an open standard.** Can the Foundation specify a confidential-inference chip architecture — tamper mesh, active shield, power-analysis-hardened crypto, per-chip hardware root, attestation anchored in PRSM's ProvenanceRegistry rather than any single vendor's CA — such that multiple fabs/designers can produce compliant hardware under competitive implementation? See `docs/2026-04-19-confidential-inference-silicon-standard.md` for the in-flight spec.

5. **Economic-layer detection and slashing.** If a stolen+perturbed model M' is detected post-hoc via output watermarking, can PRSM's ProvenanceRegistry + staking contracts slash the offender's stake and redirect recovered FTNS to the original publisher? This is the economic backstop that makes hardware + cryptographic defenses bite — theft without profitable re-monetization is not rational.

**Cross-references to other research tracks:**

- **R3 (activation-inversion attack characterization).** The activation-inversion threat is adjacent: attackers with access to intermediate activations across many inferences can reconstruct weights, bypassing any TEE that only protects VRAM. R8 defenses against activation-inversion are the same defenses R3 is characterizing. Outputs compose.
- **R2 (MPC for sharded inference).** R2's cryptographic weight sharding IS R8 defense layer (3) under a different name. If R2 produces a practical scheme, R8 inherits it.
- **R7 (KV/activation compression).** Quantized activations may reduce the information leaked to an attacker performing activation inversion — potentially strengthening R8. Must be measured, not assumed.

**Relationship to the Phase 2 launch UX thesis (see `docs/2026-04-12-phase2-remote-compute-plan.md`).** The launch thesis assumes T3 cloud-arbitrage nodes serve the initial demand. Those operators run H100s they do not own (rented from AWS/GCP/Azure). The data-center operator can always dump VRAM without the PRSM operator's cooperation. Therefore *even with H100 CC*, residual trust sits with the underlying cloud provider, not with PRSM's meganode operator. Frontier labs will factor this in when deciding whether to publish. R8 ultimately pushes toward PRSM-specific silicon that removes this residual dependency.

**Governance split (decided 2026-04-19):**

- **PRSM Foundation** owns the confidential-inference silicon standard: threat model, attestation protocol, compliance test suite, on-chain attestation registry contracts, certification process.
- **Prismatica** (and any other interested party) is a first implementer. Prismatica commits to building T4 meganode hardware conforming to the published standard. It does not own the standard and cannot modify it without Foundation governance approval.
- **Any fab/chip designer** may implement the standard and petition the Foundation for certification. Compliance is the gate, not brand; a successful alternate implementer reduces Prismatica's leverage to zero.

This mirrors the governance pattern that worked for TCG/TPM, RISC-V, OpenCompute, and the Confidential Computing Consortium — separating standard-setting from implementation is the structural guarantee against rent extraction by the first implementer.

**Watch signals:**

- NVIDIA Confidential Compute SDK updates (attestation API changes, new side-channel mitigations).
- AMD SEV-SNP / Intel TDX adoption at hyperscalers — moves baseline trust floor.
- Academic red-team publications on H100 CC side channels.
- Frontier lab public statements on "what would it take for us to publish weights to a decentralized marketplace" — the definitive signal.
- Confidential Computing Consortium standards work on attested accelerator specifications.

**Effort if pursued:** substantial. Sub-questions 1-3 are 1-2 quarters of research each, feasibly done by Foundation + academic partners. Sub-question 4 (custom silicon) is a 5-10 year arc and is tracked separately in the silicon-standard planning doc. Sub-question 5 requires contract work tied to ProvenanceRegistry v2.

**Trigger to move to engineering:**

- A first-tier frontier lab publicly commits to publishing SOTA weights to PRSM *conditional on* named defense-stack elements being shipped. This is the market signal that overrides any in-house prioritization judgment.
- OR at least two complementary layers (e.g., robust fingerprinting + output watermarking) reach research maturity simultaneously, allowing a composition prototype to ship and be demoed to frontier labs as a persuasion artifact.
- OR H100 CC is publicly broken in a way that forces the defense stack forward regardless of market signal.

---

## Governance and Tracking

- This document is reviewed semi-annually (next: 2026-10-14) to add new research questions and retire resolved ones.
- No item here is prioritized over product-phase work.
- Items may be promoted to engineering phases when triggers (named above) fire.
- Research partnerships with academic groups are a viable delivery model for items R1-R3.
