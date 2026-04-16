# `docs/archive/`

Historical planning and reference documents that no longer reflect current PRSM architecture. Preserved here as execution record and for understanding what existed pre-pivot. **Do not treat as authoritative for current work.**

Moved here on 2026-04-16 as part of a documentation triage pass. For current docs, see [`../2026-04-10-audit-gap-roadmap.md`](../2026-04-10-audit-gap-roadmap.md) (master roadmap) and related `2026-04-*` phase plans in `docs/`.

## What's in here

### Ring plans (pre-v1.6 era)

| File | Status | Notes |
|---|---|---|
| `2026-04-06-ring1-wasm-sandbox.md` | Shipped | Ring 1 WASM runtime (v0.30.0+). Code at `prsm/compute/wasm/`. |
| `2026-04-06-ring2-mobile-agent-dispatch.md` | Shipped | Ring 2 gossip bidding + escrow. Code at `prsm/compute/agents/`. |
| `2026-04-06-ring3-swarm-compute.md` | Shipped | Ring 3 semantic sharding, map-reduce. |
| `2026-04-06-ring4-economy.md` | Shipped, superseded | Ring 4 PCU pricing / staking. Superseded by Phase 1 (on-chain provenance) + Phase 7 (content confidentiality). |
| `2026-04-06-ring5-agent-forge.md` | Re-scoped post-v1.6 | References deleted NWTN 5-layer pipeline. Ring 5 is now WASM mobile-agent runtime only. |
| `2026-04-06-ring6-polish.md` | Shipped | Ring 6 dynamic gas, RPC failover, CLI UX. |
| `2026-04-06-ring7-confidential-compute.md` | Shipped | Ring 7 TEE + DP noise (v0.35.0). |
| `2026-04-06-ring8-model-sharding.md` | Shipped (single-node) | Ring 8 tensor parallelism. Remote-dispatch completion is current Phase 2. |
| `2026-04-06-ring9-nwtn-model.md` | Scope-reduced post-v1.6 | Ring 9 is now training-pipeline-only. NWTN orchestrator deleted in v1.6.0. |
| `2026-04-07-ring10-fortress.md` | Shipped | Ring 10 integrity verification (v0.35.0). |
| `2026-04-09-v1.6-scope-alignment.md` | Completed 2026-04-09 | v1.6.0 sprint plan (deleted ~210K LoC legacy). |

### Phase 1 superseded iterations

| File | Status | Notes |
|---|---|---|
| `2026-04-10-phase1.1-codex-fixes-plan.md` | Superseded | Codex review round 1 fixes. Replaced by Phase 1.3 completion plan. |
| `2026-04-10-phase1.2-codex-rereview-fixes-plan.md` | Superseded | Codex review round 2 fixes. Replaced by Phase 1.3 completion plan. |
| `phase4_implementation.md` | Superseded | Old Phase 4 stub. Replaced by master roadmap Phase 4 section. |

### Pre-pivot architecture docs

These documents reflect the pre-2026-04 bonding-curve tokenomics and/or pre-v1.6 AGI-framework architecture. Both have been explicitly removed from the current design; see `PRSM_Tokenomics.md` §3 and the v1.6 scope-alignment design for details.

| File | Superseded by |
|---|---|
| `FTNS_EARLY_INVESTOR_COMPENSATION_ARCHITECTURE.md` | `PRSM_Tokenomics.md` §3 (equity-investment architecture). FTNS is now compensation-only; bootstrap via Prismatica equity (Reg D 506(c)). |
| `BRIDGE_MULTI_SIGNATURE_VERIFICATION.md` | Phase 1 `RoyaltyDistributor.sol` on-chain settlement. |
| `DAG_TRANSACTION_SIGNATURES.md` | Phase 1 on-chain provenance contracts. |
| `REGULATORY_TRANSPARENCY_REPORT.md` | Superseded by hybrid tokenomics legal tracking + Risk Register post-equity-pivot disclosures. |
| `NWTN_Critical_Analysis_and_Improvement_Recommendations.md` | NWTN orchestrator deleted in v1.6.0. |
| `BOOTSTRAP_DEPLOYMENT.md` | Duplicate of `BOOTSTRAP_DEPLOYMENT_GUIDE.md` (kept live). |
| `QUICKSTART_GUIDE.md` | Pre-v1.6 quickstart referencing deleted NWTN 5-agent orchestrator and teacher models. Superseded by `quickstart.md` (post-v1.6, cross-node walkthrough). Archived 2026-04-16. |
