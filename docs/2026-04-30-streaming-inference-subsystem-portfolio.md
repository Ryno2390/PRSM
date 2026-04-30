# PRSM Streaming-Inference Subsystem — 9-Sprint Portfolio

**Prismatica, Inc.** | **April 30, 2026**
Consolidated narrative covering the streaming-inference subsystem's arc from vendor integration to roadmap-cap closure. Pulls from session summaries sprint-1 through sprint-9 (`docs/2026-04-22-session-summary-engineering-sprint.md` through `docs/2026-04-30-session-summary-sprint-9.md`) and the cumulative audit-prep bundle (`docs/2026-04-27-cumulative-audit-prep.md` §7.1–§7.15).

**Audience:** investors, auditor shortlist recipients, technical partners, Foundation officers.

---

## 1. Executive summary

In nine focused engineering sprints, PRSM shipped **15 merge-ready streaming-inference phases** covering every named structural surface from third-party scheduler integration through fully constant-time speculative decoding under Tier C content. The arc terminates at a **roadmap cap**: every named structural deferral introduced across the prior eight sprints is closed, the threat-model addendum has reached final revision (§3.1–§3.8 at 1.6), and the audit-prep bundle (`cumulative-audit-prep-20260430-e`) is structured for an external auditor to scope coverage proportionally without the surface shifting mid-engagement.

The subsystem totals **605 unit + 7 slow real-distilgpt2 E2E tests green** at the cap. Every phase shipped with an independent code review and HIGH/MEDIUM remediations resolved pre-tag. **Six honest-scope residuals carry forward** — none blocking; all either conditional follow-ups (telemetry-gated) or accepted v1 trades documented in the threat model.

The streaming-inference subsystem is now in stable shape for either a pivot to the Foundation governance + audit engagement substrate, or — if multi-stage Tier C deployment telemetry warrants — a single conditional follow-up sprint (Phase 3.x.11.q.y'').

---

## 2. The arc — 9 sprints, 15 phase tags

Each sprint identified named structural deferrals, documented them in audit-prep + threat-model, and shipped them in subsequent sprints with E2E pins flipping the documentation from "leak documented" to "leak absent." The table below reads chronologically; each row is a merge-ready tag.

| # | Phase | What shipped | Tag |
|---|-------|--------------|-----|
| 1 | **3.x.6** | Vendored GradientHQ Parallax (Apache 2.0) — first vendored third-party path; 4 PRSM-original trust adapters + ParallaxScheduledExecutor | `phase3.x.6-merge-ready-20260427` |
| 2 | **3.x.7** | Cross-host ChainExecutor — RpcChainExecutor + LayerStageServer + multi-stage TEE attestation envelope. Closes "brain has no hands" gap | `phase3.x.7-merge-ready-20260428` |
| 3 | **3.x.7.1** | Chunked activation streaming — v2 wire format with ActivationChunk + manifest field; 16 MiB activation bit-identical to single-host | `phase3.x.7.1-merge-ready-20260428` |
| 4 | **3.x.8** | Streaming-token output — TokenFrame + StreamFinalFrame wire format + StreamingLayerRunner Protocol + downgrade-resistant `streamed_output` receipt flag | `phase3.x.8-merge-ready-20260428` |
| 5 | **3.x.8.1** | SSE-framed POST /compute/inference/stream HTTP endpoint + design-plan §3.4 settle-on-tokens-emitted billing policy | `phase3.x.8.1-merge-ready-20260428` |
| 6 | **3.x.10** | AutoregressiveStreamingRunner — replaces SyntheticStreamingRunner placeholder with real HF generate; bit-identical greedy proof on distilgpt2 | `phase3.x.10-merge-ready-20260428` |
| 7 | **3.x.10.x** | Production wiring — max_tokens + temperature wire fields + factory + node-startup wiring | `phase3.x.10.x-merge-ready-20260428` |
| 8 | **3.x.10.y** | Tier C constant-time padding decorators (M1 + M2) + dispatch-layer gate + HF prompt-echo fix | `phase3.x.10.y-merge-ready-20260429` |
| 9 | **3.x.11** | Sharded autoregressive decode — per-token chain redispatch with KV-cache handoff; ShardedAutoregressiveRunner + KVCacheManager + executor per-token loop; bit-identical sharded vs single-host greedy | `phase3.x.11-merge-ready-20260430` |
| 10 | **3.x.11.x** | Pipelining (wire-level chunked PREFILL composition) + per-iteration receipt attestation envelope under separate PRSM-MI-ATT-V1 magic prefix | `phase3.x.11.x-merge-ready-20260430` |
| 11 | **3.x.11.y** | Speculative decoding — K=4 drafts batched K+1 verify + accept-longest-prefix + RollbackCacheRequest broadcast; HFDraftModel; greedy-only at v1 | `phase3.x.11.y-merge-ready-20260429` |
| 12 | **3.x.11.y.x** | Sampling-correct speculation under T > 0 — Leviathan-2023 §2.2 rejection sampling under Option C.1 + adaptive K | `phase3.x.11.y.x-merge-ready-20260429` |
| 13 | **3.x.11.q** | Tier C constant-time sharded decode via chain-level decorators (executor → caller wire); BatchedTrailingShardedExecutor (M2) + FixedRateShardedExecutor (M1) + routing-layer no-silent-fallback invariant | `phase3.x.11.q-merge-ready-20260429` |
| 14 | **3.x.11.q.y** | Constant-time speculation under Tier C (executor → stage wire) — encrypted_proposed_token_probs (AES-GCM) + tail-runner constant_k_commitment + flat_k_mode + tier_c_speculation_enabled gate | `phase3.x.11.q.y-merge-ready-20260430` |
| 15 | **3.x.11.q.y'** | Closure of v1 honest-scope residuals — always_rollback_k + replay_accepted_prefix protocol (rollback-distinct AAD) + X25519AnchoredCipher (per-request ECDH on HandoffToken.ephemeral_pubkey) | `phase3.x.11.q.y-prime-merge-ready-20260430` |
| 16 | **3.x.11.q.x** | **Roadmap cap.** per_stage_dispatch_cadence_seconds (executor-side cadence clamp) + BatchedTrailingShardedExecutor.pad_to_bytes (UTF-8-safe response-size padding) | `phase3.x.11.q.x-merge-ready-20260430` |

---

## 3. The threat-model arc

A new threat-model addendum (`docs/2026-04-30-phase3.x.11-threat-model-addendum.md`) tracks the cross-cutting privacy threats that emerged as the subsystem grew. It reached **revision 1.6** at the roadmap cap, with eight numbered threat surfaces (§3.1–§3.8) — each one introduced by a phase, characterized in the addendum, and either mitigated or documented as honest-scope residual.

**The closure-of-residuals discipline is the audit-relevant artifact.** Each phase shipped E2E pins ASSERTING the wire-side threats (e.g. `test_e3_residual_rollback_leak_is_documented` literally asserted `n_positions_to_drop == K - accepted_count` was visible on the wire). Subsequent phases flipped the pins from "leak documented" to "leak absent" + updated the threat-model section in the same PR. An auditor walking the diff sees coordinated documentation + code + test changes rather than sprawling unrelated PRs.

By the cap, every numbered §3.X threat surface has either:
- A closure path documented (e.g. §3.6.2 plaintext-probs leak → closed via §3.8 encrypted_proposed_token_probs)
- An honest-scope residual explicitly accepted as a v1 trade, with the closure path named for the next slice or marked permanent (e.g. multi-position §2.2 marginal narrowing under constant-K is mathematically inherent and documented as a permanent narrowing of the §2.2 claim)

---

## 4. Final state — metrics at the cap

| Metric | Value |
|--------|-------|
| Phase tags shipped | 15 (3.x.6 through 3.x.11.q.x) |
| Sprints completed | 9 (2026-04-22 through 2026-04-30) |
| Audit-prep cumulative tag | `cumulative-audit-prep-20260430-e` |
| Audit-prep sections | §7.1–§7.15 + §7.14.1 (q.y' delta) |
| Threat-model revision | 1.6 (final) |
| Threat-model numbered surfaces | §3.1–§3.8 |
| Unit tests cumulative | 605 |
| Slow real-distilgpt2 E2E tests | 7 |
| HIGH severity round-1 findings remediated pre-tag | 11 |
| MEDIUM severity round-1 findings remediated pre-tag | 21 |
| LOW severity round-1 findings remediated pre-tag | 14 |
| LOW severity findings deferred | 11 (documented) |
| Round-1 review cycles where 0 remediations were required | 1 (Phase 3.x.11.q.x) |

**Every phase carries an independent code review.** The remediation counts above are pre-tag fixes; no HIGH/MEDIUM findings ship to merge-ready in any phase.

---

## 5. Honest-scope residuals carrying forward

Six items remain after the roadmap cap. **None blocking.**

1. **Multi-stage replay best-effort** (Phase 3.x.11.q.y' residual). Stage 0 of a multi-stage chain replays accepted prefix via `forward_verify`; non-stage-0 returns False because it needs upstream hidden state. Wire-side leak still closed; cache-state correctness on stage > 0 falls back to TTL sweeper. **Conditional follow-up:** Phase 3.x.11.q.y'' if multi-stage Tier C deployment telemetry shows the gap materially affects production. Not engineered until production data exists.

2. **Replay window inside `deadline_unix`** (q.y' residual). A relay could replay an entire request envelope (including `ephemeral_pubkey` + ciphertexts) inside the deadline. Per-stage nonce cache mitigation is orthogonal; defer.

3. **Post-quantum** (R6 trigger-watch). X25519 ECDH is not post-quantum-secure. R6 trigger-watch — when post-quantum standards firm up, X-Wing or Kyber-768 hybrid is the upgrade path.

4. **Multi-position §2.2 marginal narrowing under constant-K** (q.y baseline). Inherent to the constant-K commitment design. User-facing output stays §2.2-correct via `accepted_count`; auditors note positions ac+1..K of `verified_token_ids` are argmax fillers when `accepted_count < K`.

5. **Adaptive K under flat-K is OFF** (q.y baseline). When operators wire `flat_k_mode=True` for the §3.6.3 closure, the Phase 3.x.11.y.x adaptive K state machine is disabled. Operator-configurable perf-vs-privacy trade.

6. **Adaptive cadence based on observed network load** (q.x deferral). Phase 3.x.11.q.z (post-roadmap-cap research item).

---

## 6. What the subsystem enables

The streaming-inference subsystem ships **the load-bearing technical substrate for PRSM's MCP + TUI product**. Without it:

- **No cross-host inference.** Models running on multi-node chains can't sign verifiable receipts.
- **No streaming UX.** Chat-style incremental output through `prsm_inference` requires the wire format from sprint-4 + the SSE endpoint from sprint-5.
- **No real-model receipts.** Synthetic placeholders shipped through sprint-6; only sprint-7 (Phase 3.x.10) wires real HuggingFace models to the receipt-signing path.
- **No Tier C content under sharded decode.** Sprints 13–16 (Phase 3.x.11.q through 3.x.11.q.x) ship the constant-time guarantees that let regulated content flow through PRSM without per-token timing or response-size leakage.

**For the auditor engagement:** the subsystem can be audited as a finished surface. The 16 phase tags are stable; the audit-prep + threat-model are at final revision; honest-scope residuals are explicit and either conditional follow-ups or accepted v1 trades. An external auditor can scope coverage proportional to the §7.X numbering and trust that the surface won't shift mid-engagement.

**For investors:** the subsystem went from greenfield (vendor Parallax) to roadmap cap in 9 days of focused engineering. Each sprint shipped one or two merge-ready tags with independent code review, threat-model documentation, and E2E proof on real models. The cadence is the load-bearing artifact: each sprint introduced new structural surface, and each subsequent sprint either closed a deferral or shipped fresh capability — never carrying scope between sprints unless explicitly named in audit-prep.

---

## 7. What's next on the PRSM critical path

Three honest options at the cap:

1. **Auditor engagement.** Highest leverage. The auditor shortlist + RFP template are ready (`docs/auditor-rfp-template.md`); the cumulative-audit-prep bundle is stable. Sending RFP responses to the shortlist is the actual gate on mainnet — not more engineering. Foundation governance + legal-counsel substrate continues to track separately.

2. **Pivot to other Phase 3 surfaces.** R5 Tier C TEE-resident PSK material would obsolete the X25519 ECDH path entirely. R8 defense-stack composition formal analysis would let auditors reason about M1 + flat-K + constant-K + chain-level-M2 composition formally rather than per-pair. Both are research scoping.

3. **Multi-stage replay forward path** (Phase 3.x.11.q.y''). Conditional follow-up on the q.y' residual. Only worth doing if multi-stage Tier C deployment telemetry warrants. Defer until production data exists.

After 9 sprints in a row on streaming-inference, the team's natural pivot is option 1 — the engineering substrate is now ahead of the engagement-readiness substrate, and that's where the next gate lives.

---

## 8. Reading path for first-time auditors

For an external auditor receiving this portfolio cold, the recommended reading order is:

1. **This document** (§1–§7 above) — sets the arc and final state.
2. **`docs/2026-04-21-audit-bundle-coordinator.md`** — first-document pointer with the full per-phase tag table.
3. **`docs/2026-04-27-cumulative-audit-prep.md`** §7.1 through §7.15 + §7.14.1 — per-phase audit-prep sections in chronological order.
4. **`docs/2026-04-30-phase3.x.11-threat-model-addendum.md`** §3.1 through §3.8 — cross-cutting threat model.
5. **R3 baseline** (`docs/2026-04-22-r3-threat-model.md`) — the baseline activation-inversion model the streaming-inference addendum extends.
6. **Selected sprint summaries** (`docs/2026-04-*-session-summary-sprint-*.md`) — reading 7-9 in particular gives the closure-of-residuals discipline arc.

Each per-phase audit-prep section names load-bearing files + tests for that phase; auditors with limited engagement time can scope coverage proportionally.

---

## Changelog

- **0.1 (2026-04-30)** — initial portfolio doc consolidating sprints 1–9. Tag pending; pin to current main.
