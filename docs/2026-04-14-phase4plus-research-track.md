# Phase 4+ Research Track Stub

**Status:** Research roadmap placeholder. Not engineering work. Captured 2026-04-14 so Vision-doc-derived research items are tracked.

**Context:** Several items referenced in `PRSM_Vision.md` are explicitly positioned as research roadmap, not product. This document prevents them from being forgotten, without committing to a delivery schedule.

## R1: Fully Homomorphic Encryption (FHE) for Private Inference

**Vision doc reference:** Section 7, "Honest limits" — *"FHE is currently 10,000-100,000× slower than plaintext inference and not production-ready for frontier-scale models."*

**Research question:** Can FHE-based inference on PRSM become practical within a 3-5 year horizon, given continued improvements in FHE schemes (CKKS, TFHE), hardware acceleration (Intel HERACLES, specialized FHE ASICs in development), and model quantization co-design?

**What would change if this ships:** Eliminates the remaining insider-exposure risk from Section 7. Node holding a model shard could execute inference on encrypted activations without ever decrypting them. Would make PRSM's privacy tier strictly dominant over centralized inference for any use case, not just the PII-sensitive niche.

**Watch signals:** Zama's fhEVM progress, Intel HERACLES availability, TFHE-rs performance benchmarks on LLM workloads, academic papers demonstrating sub-100× FHE inference on transformer models.

**Effort if pursued:** substantial research partnership with a specialized cryptography group. Not a solo project.

## R2: Multi-Party Computation (MPC) for Sharded Inference

**Vision doc reference:** Section 7, "Honest limits" — *"MPC is 10-100× overhead and requires non-colluding node assumptions."*

**Research question:** Can MPC protocols be adapted to PRSM's tensor-sharded architecture such that non-colluding node assumptions are realistic under PRSM's tiered supply (T1 consumer edge, T3 cloud arbitrage, T4 meganodes across independent operators)?

**What would change if this ships:** Intermediate between plaintext (current Section 7 stack) and full FHE. Earlier delivery than FHE; stronger guarantees than TEE-only.

**Specific sub-questions:**
- How is the non-collusion assumption quantified in a permissionless network? Economic stake, geographic diversity, operator-class diversity?
- Does MPC overhead compose acceptably with the 9000× bandwidth handicap between consumer nodes vs. datacenter NVLink?
- Can MPC be scoped to the "sensitive layers" of an inference (first few transformer layers where activation-inversion attacks are effective), leaving later layers in TEE-only mode, to bound overhead?

**Effort if pursued:** moderate — this is closer to integration research than foundational cryptography research.

## R3: Activation-Inversion Attack Characterization & Mitigation

**Vision doc reference:** Section 7, "Honest limits" — *"Activation-inversion attacks (Zhu et al. 2019 and follow-up literature) can partially reconstruct input prompts from early-layer activations."*

**Research question:** How strong are activation-inversion attacks against production-scale transformer inference under PRSM's specific threat model (single malicious node holding one tensor-parallel shard of one layer, without access to other shards or the model's later layers)?

**Current state:** Phase 2 line item B (`docs/2026-04-12-phase2-remote-compute-plan.md` addendum) adds topology randomization as a first-order mitigation. This research track asks whether topology randomization is *sufficient* or whether stronger measures (activation-layer noise injection, secure aggregation across tensor-parallel peers, differential-privacy guarantees) are needed.

**Watch signals:** academic papers on activation inversion at scale, red-team reports from AI safety groups targeting sharded inference systems.

**Effort if pursued:** a quarter of red-team + blue-team work; reportable results.

## R4: Per-Provider Supply Caps and Geographic Diversity Incentives

**Vision doc reference:** Section 6 subsection, "Honest caveats" — *"If T3 dominates supply, PRSM effectively runs on 3-4 hyperscalers underneath. Mitigable via supply caps per provider and geographic-diversity bonuses."*

**Research question:** How should PRSM governance parameterize supply-side diversity constraints so that no single cloud provider or geographic region exceeds X% of network capacity, without destroying T3 economics?

**Why this is research, not engineering:** Policy parameters that depend on empirical network data that doesn't yet exist. Premature implementation risks over-constraining the network during bootstrap when diversity emerges organically anyway.

**Triggers to move this to engineering work:**
- Monitoring data shows any single provider exceeds 30% of supply.
- Monitoring data shows any single country exceeds 40% of supply.
- An operator raises a governance concern backed by data.

**Effort if pursued:** small — mostly governance-design work and parameter tuning once monitoring exists.

## R5: Content Tier C Hardening Against Majority Collusion

**Vision doc reference:** `PRSM_Vision.md` §7 "Honest limits" — *"A colluder who obtains enough erasure-coded fragments (above the K-of-N reconstruction threshold) plus enough key-shares (above the M-of-N threshold) can reconstruct Tier C content."*

**Research question:** Can Tier C confidentiality be strengthened beyond the Phase 7 design, which relies on computational assumptions plus economic-cost-of-collusion, to provide guarantees that hold even against a majority-colluding adversary in the right positions?

**Why this is research, not engineering:** The Phase 7 Tier C design is correct and shippable, and it meets the threat model of the vast majority of regulated-industry use cases. But for the most extreme threat models (nation-state adversaries, adversarial-foundation scenarios, critical-infrastructure use cases), a stronger guarantee would be valuable. The techniques that provide it — threshold FHE, secure aggregation with information-theoretic properties, entropy-bounded storage schemes — are active research areas without production-ready implementations.

**What would change if this ships:** Tier C becomes information-theoretically secure rather than computationally secure for the portion of the threat model it covers. Opens defense, government, and critical-infrastructure use cases that current Tier C can serve only with caveats.

**Watch signals:** academic papers on threshold cryptography at scale; NIST post-quantum secure computation competitions; threshold-FHE deployment milestones from Zama, Duality, Inpher.

**Effort if pursued:** substantial research partnership with a specialized cryptography group. Not a solo project. Overlaps significantly with R1 (FHE for private inference) since many underlying primitives are shared.

## R6: Post-Quantum Signatures

**Vision doc reference:** none yet, but implicit in any long-horizon cryptographic system.

**Research question:** When should PRSM's Ed25519 signing infrastructure (receipts, on-chain settlement) migrate to post-quantum primitives?

**Why it's here:** not urgent; post-quantum threat models put classical signatures safe through at least 2030+ for most workloads, and Ethereum L2 itself will migrate before PRSM needs to unilaterally.

**Trigger to move to engineering:** NIST finalization of post-quantum signature standards + Ethereum-level migration plans.

## R7: KV Cache & Activation Compression for Consumer-Edge Inference

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

**Trigger to move to engineering:**
- Research quarter identifies a scheme with ≥3× bandwidth reduction at <1% accuracy loss on PRSM's benchmark suite, with red-team report showing no regression on activation-inversion surface.
- OR monitoring data shows T1/T2 supply share falling below target and T3 concentration exceeding R4 thresholds — forcing prioritization of T1-viability mechanisms.

## Governance and Tracking

- This document is reviewed semi-annually (next: 2026-10-14) to add new research questions and retire resolved ones.
- No item here is prioritized over product-phase work.
- Items may be promoted to engineering phases when triggers (named above) fire.
- Research partnerships with academic groups are a viable delivery model for items R1-R3.
