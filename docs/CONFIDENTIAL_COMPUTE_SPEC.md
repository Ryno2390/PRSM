# PRSM Phase 2: Confidential Compute + Model Sharding Spec

**Version:** 1.0
**Date:** 2026-04-06
**Status:** Historical sprint design document — see notice below

> **⚠️ HISTORICAL DOCUMENT**
>
> This is the original Phase 2 design spec that drove implementation of Rings 7-10 during April 2026. It is preserved as a historical record of the sprint's design decisions.
>
> **Some details were reconsidered in v1.6.0 (2026-04-09).** In particular, where this document refers to "LLM-powered agent forge" (Ring 5) or "NWTN as Agent Architect," that implementation was re-scoped during the v1.6.0 scope alignment. PRSM no longer runs an internal LLM-backed reasoning layer — reasoning happens in third-party LLMs (local or via OAuth/API) that reach PRSM via MCP tools. Ring 9 ("The Mind") remains valid as the training pipeline for a **future** fine-tuned NWTN LLM that will become one of several LLM options alongside third-party models.
>
> For the **current** architecture, see:
> - [`docs/architecture.md`](architecture.md) — current 10-Ring architecture and query flow
> - [`docs/IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) — current subsystem status
> - [`docs/2026-04-09-v1.6-scope-alignment-design.md`](2026-04-09-v1.6-scope-alignment-design.md) — the scope narrowing design
>
> The rest of this document is preserved as-written for historical accuracy.

---

## Executive Summary

Phase 1 (Rings 1-6) built the sovereign-edge AI infrastructure: WASM sandboxes, mobile agents, swarm compute, hybrid pricing, LLM-powered agent forge, and production hardening. **Phase 2 extends this into confidential computing** — the ability to run proprietary models and process sensitive data across untrusted consumer nodes without any single party (node operator, model owner, PRSM Foundation) being able to see the data or reconstruct the model.

The core insight: PRSM's existing architecture (WASM sandbox zero-persistence + sharded data + ephemeral agents) already provides two of three privacy layers by construction. Phase 2 adds the third — model sharding via pipeline/tensor parallelism with cryptographic protections — making NWTN potentially the most secure frontier model available.

### Three Privacy Layers

| Layer | Mechanism | Already Built? |
|-------|-----------|---------------|
| **Data privacy** | Semantic sharding — no node sees full dataset | Yes (Ring 3) |
| **Compute privacy** | WASM sandbox zero-persistence — no logs, no history | Yes (Ring 1) |
| **Model privacy** | Pipeline/tensor parallelism — no node holds full model | **Phase 2** |

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Side-channel defense | Differential Privacy noise injection (default) | Nearly free, well-studied, 1-2% accuracy degradation acceptable |
| Collusion defense | Randomized pipeline assignment + stake-weighted trust | Makes collusion hard (random) and expensive (slashing) |
| Latency mitigation | Tensor parallelism (Megatron-LM style) with regional clustering | Converts sequential latency to parallel; leverages Ring 3 clustering |
| TEE integration | Runtime abstraction (like WASMRuntime) with platform-specific backends | Covers Intel SGX, ARM TrustZone, Apple Secure Enclave without lock-in |

---

## Security Architecture

### Threat Model

**Adversaries:**
- **Curious node operator:** Wants to see the data passing through their PS5 or extract model weights
- **Colluding nodes:** Multiple operators cooperate to reconstruct a proprietary model
- **Network observer:** Monitors traffic patterns to infer query content
- **Malicious researcher:** Attempts to exfiltrate a proprietary model via crafted queries

**Trust assumptions:**
- PRSM protocol is open-source and auditable
- Hardware TEEs (when available) are trusted at the silicon level
- The FTNS staking/slashing mechanism creates economic disincentives for misbehavior
- No single entity (including PRSM Foundation) has privileged access to data or models in transit

### Defense Layer 1: Differential Privacy on Intermediate Activations

**Problem:** A node running layer 47 of a 96-layer model sees the tensor flowing through. Repeated observations can leak information about inputs and model architecture.

**Solution:** Before an intermediate tensor leaves a node, add calibrated Gaussian noise.

**Implementation:**
- Noise calibrated per model architecture (done once at model registration, not per inference)
- Uses the DP-SGD framework (Abadi et al.) adapted for inference
- Privacy budget (ε) configurable per model: lower ε = more noise = stronger privacy = slightly lower accuracy
- Default ε = 8.0 (moderate privacy, <1% accuracy loss for most tasks)
- Model owners can set stricter ε for highly proprietary models

**Integration point:** Applied in the `ModelPipelineExecutor` between shard execution and network transfer. The executing node adds noise before sending the activation tensor to the next pipeline stage.

**Performance cost:** Negligible — noise generation is O(n) where n is tensor size, dominated by the actual matrix multiplication.

### Defense Layer 2: Anti-Collusion via Randomized Pipelines + Staking

**Problem:** If K nodes in a model pipeline collude, they can reconstruct enough layers to approximate the full model.

**Solution (two mechanisms working together):**

**Randomized Pipeline Assignment:**
- For each inference request, NWTN randomly assigns model layers to different nodes from the available pool
- No two inferences use the same pipeline topology
- An attacker controlling N nodes out of M available would need to be assigned to the same pipeline repeatedly — probability decreases exponentially with pool size
- Pool size is enforced: model pipelines require a minimum of 20 distinct nodes in the eligible pool before execution is permitted

**Stake-Weighted Trust:**
- Nodes participating in model pipelines must stake FTNS proportional to the model's declared value tier:

| Model Value Tier | Pipeline Stake Required | Slash Rate on Violation |
|-----------------|----------------------|----------------------|
| Open-weight | 0 FTNS | N/A |
| Standard proprietary | 5,000 FTNS | 50% |
| Premium proprietary | 25,000 FTNS | 100% |
| Critical/regulated | 50,000 FTNS | 100% + network ban |

- **Detection mechanism:** Run the same inference through 2 diversified pipelines for premium/critical models. If outputs diverge beyond a threshold (accounting for DP noise), one pipeline is compromised. All nodes in the divergent pipeline are slashed.
- This makes collusion economically irrational — the cost of losing staked FTNS exceeds the value of reconstructing one model

**Integration point:** Extends Ring 4's `ProsumerManager` and Ring 6's `SettlerRegistry` with model-pipeline-specific staking tiers and slashing conditions.

### Defense Layer 3: Tensor Parallelism (Latency Mitigation)

**Problem:** Pipeline parallelism across a WAN adds round-trip time per layer. A 96-layer model with 50ms per hop = 4.8 seconds just for network transit.

**Solution:** Tensor parallelism (Megatron-LM architecture) with regional clustering.

**How it works:**
- Instead of splitting the model by layers (sequential: node 1 → node 2 → node 3), split by tensor dimensions (parallel: all nodes compute simultaneously)
- Each node computes a *slice* of every layer at the same time
- Results are all-reduced (aggregated) after each layer — one synchronization point per layer, not one per node
- This converts sequential latency (N × round-trip) into parallel latency (1 × round-trip per layer)

**Regional clustering integration:**
- Leverages Ring 3's existing node proximity grouping
- Pipeline nodes are selected from the same regional cluster (same city/ISP/subnet)
- Intra-cluster latency: 2-5ms (vs 40-60ms cross-region)
- A 96-layer model with tensor parallelism across 8 regional nodes: ~96 × 3ms = 288ms network overhead
- Combined with compute time: total inference in 1-3 seconds for most models

**All-reduce implementation:**
- Ring all-reduce pattern (each node sends to neighbor, log₂(N) rounds)
- For 8 nodes: 3 rounds × 3ms = 9ms per layer synchronization
- Bandwidth-efficient: each node sends/receives only 1/N of the full tensor per round

**Integration point:** New `TensorParallelExecutor` that replaces simple pipeline execution for large models. Uses the existing gossip infrastructure for node discovery and the existing transport layer for tensor transfer.

---

## Ring Structure — Phase 2

### Ring 7 — "The Vault": Confidential Compute Foundation

**Goal:** A PRSM node can execute model inference inside a hardware-secured environment (TEE when available, WASM sandbox as fallback) with differential privacy on all intermediate activations.

**Components:**
- `TEERuntime` abstraction interface (mirrors `WASMRuntime` from Ring 1)
  - `IntelSGXRuntime` — for Intel CPUs with SGX/TDX support
  - `AppleSecureEnclaveRuntime` — for Apple Silicon devices
  - `ARMTrustZoneRuntime` — for ARM-based devices (consoles, mobile)
  - `WASMFallbackRuntime` — software-only fallback using Ring 1 sandbox
- `DPNoiseInjector` — calibrates and applies Gaussian noise to intermediate tensors
- `ActivationEncryptor` — encrypts tensors before network transfer, decrypts inside TEE
- Privacy budget tracker — enforces per-model ε budget across inference sessions
- `GOSSIP_TEE_CAPABILITY` message type — nodes advertise their TEE hardware support

**Deliverable:** A node with TEE hardware can execute a model shard where the node operator cannot inspect the data being processed, and all intermediate activations are DP-protected before leaving the secure boundary.

---

### Ring 8 — "The Shield": Model Sharding + Collusion Resistance

**Goal:** A proprietary model can be sharded across the network using tensor parallelism, with randomized pipeline assignment and stake-weighted trust ensuring no single node (or colluding group) can reconstruct the full model.

**Components:**
- `ModelSharder` — splits model weights into tensor-parallel shards using Megatron-LM partitioning
  - Supports: transformer attention heads (column-parallel), MLP layers (row-parallel), embedding tables (vocabulary-parallel)
  - Shard metadata registered in content index (like Ring 3's `SemanticShardManifest` but for model layers)
- `TensorParallelExecutor` — coordinates tensor-parallel inference across regional node clusters
  - Ring all-reduce for activation synchronization
  - Regional cluster selection via Ring 3's proximity grouping
  - Configurable parallelism degree (2, 4, 8, 16 nodes)
- `PipelineRandomizer` — assigns model shards to nodes randomly per inference
  - Minimum pool size enforcement (20+ eligible nodes)
  - Assignment entropy tracking (ensures sufficient randomization over time)
- `ModelPipelineStaking` — extends prosumer staking for model pipeline participation
  - Model value tier classification
  - Pipeline-specific slashing conditions
  - Diversified pipeline execution for premium models (2 pipelines, output comparison)
- `CollisionDetector` — compares outputs from diversified pipelines
  - Accounts for DP noise variance when comparing
  - Triggers slashing on statistically significant divergence

**Deliverable:** A model owner can upload a proprietary model to the PRSM network. It is automatically sharded, distributed across staked nodes, and serves inference requests where no individual node holds more than 1/N of the weights. Collusion attempts are detected and economically punished.

---

### Ring 9 — "The Mind": NWTN Open-Weight Model

**Goal:** Fine-tune and deploy NWTN as an open-weight model optimized for PRSM's architecture — task decomposition, WASM manifest generation, hardware-aware planning — running confidentially on the Phase 2 infrastructure.

**Components:**
- **Training pipeline:**
  - Corpus: AgentTrace data collected by Ring 5 during Phase 1 operations
  - Base model: Open-weight frontier model (Llama, Qwen, or Mistral family — selected based on 2026 landscape)
  - Fine-tune objectives: task decomposition accuracy, WASM manifest quality, hardware-tier prediction
  - Evaluation: agent execution success rate, PCU prediction accuracy, researcher satisfaction scores
- **Model deployment:**
  - NWTN weights sharded via Ring 8's `ModelSharder`
  - Served as a PRSM network service (any node can request NWTN inference, Ring 8 handles distribution)
  - Nodes can also run NWTN locally (for zero-latency, maximum privacy mode)
  - Cloud endpoint for low-power nodes that can't run locally
- **NWTN as pricing broker:**
  - Replaces the prompt-based decomposition in Ring 5 with fine-tuned decomposition
  - Produces WASM manifests directly (not just JSON task plans)
  - Hardware-aware: adjusts agent code based on target node's profile
- **Economic model:**
  - Running NWTN inference for the network earns FTNS (it's a service like any other)
  - NWTN model weights are open — anyone can verify, audit, or fork
  - The confidential compute infrastructure (Rings 7-8) protects the *data* flowing through NWTN, not the model itself

**Deliverable:** NWTN is a specialized open-weight model that any node can run. It produces better task decompositions and WASM agents than generic frontier models because it was trained on real PRSM execution traces. It runs on the same confidential compute infrastructure as any other model on the network.

---

### Ring 10 — "The Fortress": Production Hardening II

**Goal:** Harden Phase 2 infrastructure plus close remaining Phase 1 deferred items.

**Phase 2 hardening:**
- TEE attestation verification — cryptographic proof that a TEE is genuine (not emulated)
- Model shard integrity verification — hash chains on shards to detect tampering
- DP noise budget auditing — public verification that privacy guarantees are being met
- Pipeline randomization audit trail — provable randomness in node assignment
- Tensor transfer encryption — TLS 1.3 for all inter-node activation transfers
- Stress testing — adversarial scenarios: node dropout mid-pipeline, network partitions, coordinated collusion attempts

**Phase 1 deferred items (from Ring 6 scope):**
- Full API authentication (JWT/Ed25519 challenge-response on all endpoints)
- mDNS local peer discovery (find PRSM nodes on same LAN without bootstrap)
- OpenTelemetry traces across the full pipeline (forge → dispatch → shard → execute → settle)
- Streamlit dashboard extensions (swarm job visualization, FTNS flow, node health)
- Operator alerting (earnings threshold, uptime drops, slash risk)
- Comprehensive documentation (Operator Guide, Data Provider Guide, Agent Developer Guide, MCP Integration Guide)
- Python/JS/Go SDK updates for Rings 1-10 endpoints

---

## Ring Dependency Map — Phase 2

```
Phase 1 Complete (Rings 1-6)
  └─→ Ring 7: The Vault (TEE runtime + DP noise)
        └─→ Ring 8: The Shield (model sharding + collusion resistance)
              └─→ Ring 9: The Mind (NWTN fine-tune + deployment)
                    └─→ Ring 10: The Fortress (hardening + deferred items)
```

Each ring is independently valuable:
- Ring 7 alone = confidential data processing (even without model sharding)
- Ring 7 + 8 = full confidential model inference
- Ring 7 + 8 + 9 = NWTN as a confidential, open-weight frontier model
- Ring 7-10 = production-ready confidential compute platform

---

## Existing Infrastructure Reuse (Phase 1 → Phase 2)

| Phase 1 Component | Used In Phase 2 | How |
|-------------------|-----------------|-----|
| WASMRuntime (Ring 1) | Ring 7 | Fallback when TEE unavailable; WASM agents still used for data processing |
| HardwareProfiler (Ring 1) | Ring 7 | Extended to detect TEE capabilities (SGX, TrustZone, Secure Enclave) |
| AgentDispatcher (Ring 2) | Ring 8 | Dispatches model shards to pipeline nodes |
| SwarmCoordinator (Ring 3) | Ring 8 | Coordinates tensor-parallel execution across node clusters |
| Regional clustering (Ring 3) | Ring 8 | Selects low-latency node groups for tensor parallelism |
| ProsumerManager (Ring 4) | Ring 8 | Extended with model-pipeline staking tiers |
| PricingEngine (Ring 4) | Ring 8 | Extended with model inference pricing (per-token + per-shard) |
| SettlerRegistry (Ring 6) | Ring 8 | Validates model pipeline settlements; slashes collusion |
| AgentForge (Ring 5) | Ring 9 | Replaced by fine-tuned NWTN for better decomposition |
| AgentTrace (Ring 5) | Ring 9 | Training corpus for NWTN fine-tune |
| MCP tools (Ring 5) | Ring 9 | Extended with model deployment and inference tools |
| Dynamic gas pricing (Ring 6) | Ring 10 | Applies to model inference settlements |
| Signature verification (Ring 6) | Ring 10 | TEE attestation verification |

---

## What This Spec Does NOT Cover

- **Specific TEE SDK integration** — Intel SGX SDK, Apple CryptoKit, ARM TrustZone API details are implementation-phase decisions
- **NWTN base model selection** — depends on the open-weight landscape at time of fine-tuning
- **FHE (Fully Homomorphic Encryption)** — deferred until performance is viable for real-time inference (likely 2027+)
- **Console platform daemons** — PS5/Xbox PRSM clients remain a separate engineering effort
- **Regulatory/compliance** — confidential compute may have jurisdiction-specific requirements (GDPR, HIPAA) that need legal review
- **Token economics v2** — model inference pricing, pipeline staking rewards, and DP noise cost allocation may require tokenomics redesign

---

## The Vision: NWTN as the Most Secure Frontier Model

When Phase 2 is complete, NWTN will be:

1. **Open-weight** — anyone can inspect, audit, or fork the model
2. **Confidentially served** — inference runs across sharded nodes inside TEEs; no single entity sees the full computation
3. **Zero-persistence** — WASM sandbox ensures no query history, no training data leakage, no logs
4. **Collusion-resistant** — randomized pipelines + staked trust make reconstruction economically irrational
5. **Privacy-preserving** — differential privacy on all intermediate activations protects both data and model
6. **Economically self-sustaining** — nodes earn FTNS for serving NWTN inference, creating a flywheel

The paradox that makes this powerful: **the model is completely open, but the computation is completely private.** You can read every weight, but you can't see what anyone asked it or what it answered. This is the inverse of every centralized API — where the model is secret but your queries are logged.
