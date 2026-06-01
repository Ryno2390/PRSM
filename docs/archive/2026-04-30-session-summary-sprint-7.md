# PRSM Engineering Sprint 7 — 2026-04-30 Session Summary

**Prismatica, Inc.** | **April 30, 2026**
Engineering-progression changelog covering the seventh focused sprint after the 2026-04-30-b baseline (`docs/2026-04-30-session-summary-sprint-6.md`). Stacks on top of that summary; reads cleanly with or without it.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

In a one-day focused sprint after the 2026-04-30-b baseline, PRSM engineering shipped **one new merge-ready phase** that closes the most-cited remaining structural deferral the prior sprint left open:

**Phase 3.x.11.q.y — Constant-time speculation under Tier C.** Lifts the "Speculation + Tier C remains denied" line that was load-bearing in §3.7 of the threat-model addendum across the prior four sprints. Speculation can now flow under Tier C content on the executor → stage wire when the operator wires the full opt-in stack: AES-GCM `encrypted_proposed_token_probs` (closes the §3.6.2 plaintext-probs leak from sprint-5), tail-runner `constant_k_commitment` padding to K+1 regardless of actual acceptance (closes the §3.6.1 wire-shape leak via `verified_token_ids` length), client-side `flat_k_mode` gating off the adaptive-K state machine (closes §3.6.3 cross-round correlation), and a `tier_c_speculation_enabled` routing-layer gate on `ParallaxScheduledExecutor` (deploy-time policy boundary; structured failure on default-deny names the full opt-in contract).

**Honest-scope residual leak is documented, not hidden.** The `RollbackCacheRequest.n_positions_to_drop` field still encodes `K - accepted_count` on the wire when `accepted_count < K`. Closing this requires either always-rollback-K + replay-accepted-prefix (compute + transport tax) or a deeper re-architecture pushing the rollback decision server-side; both deferred to Phase 3.x.11.q.y'. The slice ships an E2E test (`test_e3_residual_rollback_leak_is_documented`) that ASSERTS the leak's wire presence — accidental quiet closure is caught as a CI regression that requires updating threat-model §3.8 in the same PR. This is the load-bearing honest-scope artifact of the sprint: the leak is operator-visible v1 trade, not a sprint-7 oversight.

**Round-1 independent code review caught one LOW idiomatic defect.** `ProbsCipher.encrypt` originally generated its 12-byte AES-GCM nonce via `AESGCM.generate_key(bit_length=128)[:12]` — functionally equivalent to `os.urandom(12)` (both are backed by the OS CSPRNG inside the `cryptography` library) but idiomatically wrong: `generate_key` is intended for key generation, and slicing 12 bytes off a 16-byte key-shaped output is non-obvious to an auditor reading the encrypt path. Closed pre-tag with a 1-line semantic-preserving swap.

Concretely:

- **~8 commits** pushed to `main` since the prior baseline tag
- **~38 new tests** (20 cipher + 7 wire-format + 6 constant-K runner + 6 routing-layer integration + 4 slow real-distilgpt2 E2E - 5 pre-existing E2E touched); 602 unit + 4 slow E2E cumulative across the streaming-inference subsystem
- **1 new merge-ready tag** (`phase3.x.11.q.y-merge-ready-20260430`)
- **1 new cumulative audit-prep tag** (`cumulative-audit-prep-20260430-c`) covering 14 streaming-inference phases (§7.1 through §7.14)
- **1 threat-model addendum revision** (1.3 → 1.4, adds §3.8 covering Phase 3.x.11.q.y v1 wire-surface analysis + the load-bearing residual rollback-drop leak)
- **Zero regressions** across the full test surface (existing 558 unit + 4 slow tests pass unchanged after the new wire field, cipher integration, and routing gate; Tier A/B paths and Tier C greedy paths are bit-identical to sprint-6)
- **Real-distilgpt2 E2E proves all four invariants end-to-end** — encrypted-wire smoke at T=0.7, every VERIFY request carries `encrypted_proposed_token_probs` and NO plaintext probs, every tail VERIFY response carries exactly K+1 entries in `verified_token_ids` regardless of acceptance, the residual rollback-drop leak is asserted to be present on the wire

This is a comparable-scope slice to sprint-6 (8 commits vs. sprint-6's 7) but covers a more-active threat-surface delta: sprint-6 closed Tier C structural deny on the executor → caller path; sprint-7 closes it on the executor → stage path under speculation specifically — the strongest content-correlated wire surface in the speculative-decoding stack.

---

## 2. Cumulative streaming-inference subsystem

After 7 sprints, the streaming-inference subsystem covers (read in order — each builds on the prior):

| Phase | Surface | Tag |
|-------|---------|-----|
| 3.x.6 | Parallax third-party scheduler integration + 4 PRSM-original trust adapters | `phase3.x.6-merge-ready-20260427` |
| 3.x.7 | Cross-host ChainExecutor (RpcChainExecutor + LayerStageServer + multi-stage TEE attestation) | `phase3.x.7-merge-ready-20260428` |
| 3.x.7.1 | Chunked activation streaming (v2 wire format) | `phase3.x.7.1-merge-ready-20260428` |
| 3.x.8 | Streaming-token output wire format + receipt streamed_output flag | `phase3.x.8-merge-ready-20260428` |
| 3.x.8.1 | SSE-framed POST /compute/inference/stream + settle-on-tokens-emitted billing | `phase3.x.8.1-merge-ready-20260428` |
| 3.x.10 | AutoregressiveStreamingRunner (real HF generate; replaces synthetic placeholder) | `phase3.x.10-merge-ready-20260428` |
| 3.x.10.x | Production wiring: max_tokens + temperature wire fields + factory + node-startup wiring | `phase3.x.10.x-merge-ready-20260428` |
| 3.x.10.y | Tier C constant-time padding decorators (M2 + M1) + dispatch-layer gate + HF prompt-echo fix | `phase3.x.10.y-merge-ready-20260429` |
| 3.x.11 | Sharded autoregressive decode (per-token chain redispatch + KV-cache handoff) | `phase3.x.11-merge-ready-20260430` |
| 3.x.11.x | Pipelining (wire-level chunked PREFILL composition) + per-iteration receipt attestation | `phase3.x.11.x-merge-ready-20260430` |
| 3.x.11.y | Speculative decoding (greedy at v1; HFDraftModel + verify-batched-K+1 + rollback) | `phase3.x.11.y-merge-ready-20260429` |
| 3.x.11.y.x | Sampling-correct speculation under T > 0 (Leviathan-2023 §2.2 + adaptive K) | `phase3.x.11.y.x-merge-ready-20260429` |
| 3.x.11.q | Tier C constant-time sharded decode via chain-level decorators (executor → caller wire) | `phase3.x.11.q-merge-ready-20260429` |
| **3.x.11.q.y** | **Constant-time speculation under Tier C (executor → stage wire)** — **NEW THIS SPRINT** | **`phase3.x.11.q.y-merge-ready-20260430`** |

**Cumulative test surface:** 602 unit + 4 slow real-distilgpt2 E2E across the streaming-inference subsystem alone. Every phase carries a round-1 independent code review with HIGH/MEDIUM remediations resolved pre-tag.

**Cumulative audit bundle:** `cumulative-audit-prep-20260430-c` — 14 sections (§7.1 through §7.14) of the per-phase auditor-facing summary, plus the threat-model addendum at `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` (1.4 revision; 8 numbered threat-surface sections).

---

## 3. What sprint-7 means for the audit engagement

This sprint adds one new section to the auditor reading path (§7.14) and one new threat-model section (§3.8) without changing the engagement model or the per-phase tag list of prior sprints — every sprint-1 through sprint-6 tag is byte-identical at this point.

**The §3.8 honest-scope artifact is the auditor-relevant story.** PRSM engineering chose to ship the residual rollback-drop leak as a documented v1 trade rather than block the slice on the deeper closure work. The argument: Phase 3.x.11.q.y closes 2.5 of the 3 §3.6 wire surfaces (the most-load-bearing two; the third — adaptive-K cross-round correlation — is structurally narrowed via operator-configurable opt-out), and the residual rollback-drop channel is materially weaker than what existed before the slice. The E2E pin (`test_e3_residual_rollback_leak_is_documented`) ensures the leak is operator-visible and CI-tracked, not buried.

**For an external auditor, this means:** the §3.8 Honest Scope subsection is the highest-density area to read first — it explicitly enumerates what's closed (encrypted probs, constant-K commitment, flat-K) vs. what's open (rollback drop value, on-wire DH key negotiation, multi-position §2.2 marginal narrowing). Auditors with limited engagement time can scope coverage proportionally without surprises.

---

## 4. What's next on the PRSM critical path

After sprint-7, the streaming-inference subsystem has **two remaining named structural deferrals** that fit a normal sprint scope:

1. **Phase 3.x.11.q.y'** (closing the residual rollback drop-value leak; on-wire DH key negotiation to drop the operator PSK burden)
2. **Phase 3.x.11.q.x** (per-stage cadence wrapping — full-network masking, complementary to chain-level decorators; M2 response-size operator-configurable padding)

Plus several research-track items that are not on the engineering critical path but bear watching: R5 Tier C TEE-resident PSK material (would let Phase 3.x.11.q.y' drop the entire on-wire DH layer), and R8 defense-stack composition analysis (would let auditors reason about M1 + flat-K + constant-K + chain-level M2 composition formally rather than per-pair).

The Foundation governance + audit engagement surface continues to track separately — that's the substrate every streaming-inference tag eventually rolls into.

---

## 5. Changelog

- **0.1 (2026-04-30)** — initial sprint-7 summary covering Phase 3.x.11.q.y. Tag pending at HEAD `e26d1bf0`. Cumulative audit-prep tag: `cumulative-audit-prep-20260430-c`.
