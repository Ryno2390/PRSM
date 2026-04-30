# PRSM Engineering Sprint 9 — 2026-04-30 Session Summary

**Prismatica, Inc.** | **April 30, 2026**
Engineering-progression changelog covering the ninth focused sprint after the 2026-04-30-d baseline (`docs/2026-04-30-session-summary-sprint-8.md`). **This sprint reaches the streaming-inference subsystem roadmap cap** — every named structural deferral introduced across sprints 1–8 is now closed.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

**Phase 3.x.11.q.x — Per-stage cadence + M2 response-size padding.** Closes the two named honest-scope items carrying since §7.13 (Phase 3.x.11.q baseline) and reaffirmed in §7.14.1 (Phase 3.x.11.q.y' delta):

- **Per-stage wire timing leak under sharded autoregressive decode.** New `RpcChainExecutor.per_stage_dispatch_cadence_seconds: Optional[float]` constructor kwarg + new `_wait_for_per_stage_cadence` helper integrated into BOTH the non-speculative `_execute_chain_streaming_sharded` AND speculative `_execute_chain_streaming_sharded_speculative` loops. When set, the helper clamps inter-iteration cadence: each per-token chain dispatch waits for ≥ cadence since the prior dispatch start. Wire-side effect: each per-stage RPC arrives at uniform inter-arrival cadence regardless of K and per-stage decode work. A network observer with visibility into a single stage's transport learns nothing about per-token complexity.

- **M2 response-size leak total joined-text length.** New `BatchedTrailingShardedExecutor.pad_to_bytes: Optional[int]` constructor kwarg pads the M2 trailing `StreamToken`'s `text_delta` to exactly `pad_to_bytes` UTF-8 bytes. New `_pad_or_truncate_utf8` helper handles codepoint-boundary safety: `bytes[:cap].decode(errors="ignore")` drops any partial multi-byte sequence at the cap, then whitespace-fill brings the byte count back to the exact target. When joined text exceeds cap, sets `finish_reason="length_capped"` to surface the truncation. Factory threads `pad_to_bytes` for `mode="m2"` and rejects for `mode="m1"`.

- **Composition with chain-level decorators (3.x.11.q).** Operator wires `RpcChainExecutor(per_stage_dispatch_cadence_seconds=...)` AS the inner of `BatchedTrailingShardedExecutor(pad_to_bytes=...)` for full-network constant-time masking: per-stage cadence clamps the executor → stage wire; M2 padding clamps the executor → caller wire byte count. Smoke-tested in `TestQXCompositionCadencePlusPadding`.

**Zero round-1 review remediations required.** Self-code-review found no HIGH/MEDIUM issues. The cadence helper uses `self._clock()` (defaults to `time.time`) for elapsed measurement and `time.sleep` to clamp; minor wall-clock-vs-monotonic drift is theoretically possible but only weakens the privacy guarantee, not functional correctness — not worth pre-tag remediation.

**Streaming-inference subsystem roadmap cap reached.** After 9 sprints in a row on the streaming-inference surface, every named structural deferral is closed. Only **Phase 3.x.11.q.y''** (multi-stage replay forward path; the q.y' best-effort honest-scope residual) remains as an optional follow-up — and that one is conditional on whether multi-stage Tier C deployment telemetry shows the gap materially affects stage > 0 cache correctness. Outside that conditional follow-up, the streaming-inference roadmap is functionally complete.

Concretely:

- **~5 commits** pushed to `main` since the prior baseline tag (4 task commits + 1 audit-prep refresh)
- **~16 new tests** (4 cadence + 1 composition + 11 padding); 605 unit + 7 slow E2E cumulative across the streaming-inference subsystem
- **1 new merge-ready tag** (`phase3.x.11.q.x-merge-ready-20260430`)
- **1 new cumulative audit-prep tag** (`cumulative-audit-prep-20260430-e`) covering 15 streaming-inference phases (§7.1 through §7.15)
- **1 threat-model addendum revision** (1.5 → 1.6) updating §3.7 + §3.8 to reflect the per-stage wire + M2 response-size leak closures
- **Zero regressions** across the full streaming-inference test surface
- **Default-None preserves legacy bit-identical behavior** — operators upgrade at their own cadence; pre-q.x deployments continue to work unchanged

This is the smallest sprint by commit count (5 vs. sprint-7 at 8 + sprint-8 at 10) but covers the highest-volume honest-scope closure: two named items that have been the load-bearing scope-honesty points for the prior two sprints.

---

## 2. Cumulative streaming-inference subsystem (final cap)

After 9 sprints, the streaming-inference subsystem covers every named phase in the roadmap (read in order — each builds on the prior):

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
| 3.x.11.q.y | Constant-time speculation under Tier C (executor → stage wire); v1 honest-scope residuals | `phase3.x.11.q.y-merge-ready-20260430` |
| 3.x.11.q.y' | Closure of v1 honest-scope residuals (drop-value leak + PSK distribution) | `phase3.x.11.q.y-prime-merge-ready-20260430` |
| **3.x.11.q.x** | **Per-stage cadence + M2 response-size padding (roadmap-cap closure)** — **NEW THIS SPRINT** | **`phase3.x.11.q.x-merge-ready-20260430`** |

**Cumulative test surface:** 605 unit + 7 slow real-distilgpt2 E2E across the streaming-inference subsystem. Every phase carries a round-1 independent code review with HIGH/MEDIUM remediations resolved pre-tag.

**Cumulative audit bundle:** `cumulative-audit-prep-20260430-e` — 15 sections (§7.1 through §7.15) of the per-phase auditor-facing summary plus §7.14.1 (q.y' delta), with the threat-model addendum at `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` (1.6 revision; 8 numbered threat-surface sections).

**Honest-scope residuals carrying forward (after roadmap cap):**

1. **Multi-stage replay best-effort (q.y' residual)** — Stage 0 only at v1; non-stage-0 needs upstream hidden state. Wire-side leak still closed regardless; cache-state correctness on stage > 0 falls back to TTL sweeper. Phase 3.x.11.q.y'' if multi-stage Tier C telemetry warrants.
2. **Replay window inside `deadline_unix` (q.y' residual)** — Per-stage nonce cache deferred (orthogonal).
3. **Post-quantum (R6 trigger-watch)** — X25519 ECDH not PQ-secure.
4. **Multi-position §2.2 marginal narrowing under constant-K (q.y baseline)** — User-facing output stays §2.2-correct via `accepted_count`; auditors note positions ac+1..K of `verified_token_ids` are argmax fillers when `accepted_count < K`.
5. **Adaptive K under flat-K is OFF (q.y baseline)** — Operator-configurable perf-vs-privacy trade.
6. **Adaptive cadence based on observed network load (q.x deferral)** — Phase 3.x.11.q.z (post-roadmap-cap research item).

None of these are blocking. The first three are conditional follow-ups; the last three are permanent honest-scope items the threat model accepts.

---

## 3. What sprint-9 means for the audit engagement

This sprint adds one new section to the auditor reading path (§7.15) and one threat-model revision (§3.7 + §3.8 → 1.6) without changing the engagement model or the per-phase tag list of prior sprints — every sprint-1 through sprint-8 tag is byte-identical at this point.

**The roadmap-cap story is the auditor-relevant arc.** The streaming-inference subsystem opened in sprint-1 (Phase 3.x.6 vendoring Parallax) and through 9 sprints incrementally added: cross-host execution (3.x.7), chunked transport (3.x.7.1), streaming UX (3.x.8 + 3.x.8.1), real-model runners (3.x.10 + 3.x.10.x + 3.x.10.y), sharded autoregressive decode (3.x.11 + 3.x.11.x), speculative decoding (3.x.11.y + 3.x.11.y.x), and Tier C constant-time guarantees (3.x.11.q + 3.x.11.q.y + 3.x.11.q.y' + this sprint's 3.x.11.q.x). At each step, the team identified named structural deferrals, documented them in audit-prep + threat-model, and shipped them in subsequent sprints with E2E pins flipping the documentation from "leak documented" to "leak absent." Sprint-9's q.x is the final cap on that arc.

**For an external auditor, this means:** the streaming-inference subsystem can be audited as a finished surface rather than a moving target. The 15 phase tags are stable; the audit-prep + threat-model are at their final revision; honest-scope residuals 1–6 above are explicit and either conditional follow-ups or accepted v1 trades. An auditor can scope coverage proportional to the §7.X numbering and trust that the surface won't shift mid-engagement.

---

## 4. What's next on the PRSM critical path

Streaming-inference roadmap is now functionally complete. Three honest options for the next focus area:

1. **Phase 3.x.11.q.y''** (conditional follow-up). Multi-stage replay forward path closes the q.y' best-effort honest-scope residual on stage > 0. Only worth doing if multi-stage Tier C deployment telemetry shows the gap materially affects cache correctness. Defer until production data exists.

2. **Pivot to other Phase 3 surfaces.** R5 Tier C TEE-resident PSK material (would let Phase 3.x.11.q.y'' drop on-wire DH entirely), R8 defense-stack composition formal analysis, or non-streaming Phase 3 surfaces.

3. **Foundation governance + audit engagement substrate.** Tags are stable; auditor RFP ready; legal counsel + jurisdiction work continues; this is where every streaming-inference tag eventually rolls into. The next high-leverage effort is likely on the engagement-readiness side rather than another engineering sprint.

The Foundation governance + audit engagement surface continues to track separately — that's where the streaming-inference work eventually rolls into.

---

## 5. Changelog

- **0.1 (2026-04-30)** — initial sprint-9 summary covering Phase 3.x.11.q.x roadmap-cap closure. Tag `phase3.x.11.q.x-merge-ready-20260430` at `adaa3ab3`. Cumulative audit-prep tag: `cumulative-audit-prep-20260430-e`.
