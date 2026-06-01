# PRSM Engineering Sprint 8 — 2026-04-30 Session Summary

**Prismatica, Inc.** | **April 30, 2026**
Engineering-progression changelog covering the eighth focused sprint after the 2026-04-30-c baseline (`docs/2026-04-30-session-summary-sprint-7.md`). Stacks on top of that summary; reads cleanly with or without it.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

In a one-day focused sprint after the 2026-04-30-c baseline, PRSM engineering shipped **one new merge-ready phase** that closes both v1 honest-scope residuals the prior sprint left open:

**Phase 3.x.11.q.y' — Closure of v1 honest-scope residuals.** Sprint-7's q.y stack closed two of the three §3.6 wire surfaces and structurally narrowed the third, but explicitly left two carry-forward residuals: the `RollbackCacheRequest.n_positions_to_drop` field still encoded `K - accepted_count` on the wire when acceptance was partial, and `ProbsCipher` PSK distribution remained an operator-managed out-of-band burden. Both are now closed.

**Channel 1: Drop-value leak closure.** New `RpcChainExecutor.always_rollback_k=True` mode dispatches a constant-K rollback per VERIFY round (drops K+1 positions regardless of accepted_count), accompanied by the accepted prefix tokens — encrypted under the wired cipher with rollback-distinct AAD (`b"|rollback"` suffix; AAD spaces disjoint from the probs path by construction so a relay cannot replay a probs ciphertext into a rollback envelope). Server decrypts at the boundary and drives `runner.replay_accepted_prefix` → forward over the prefix tokens through `forward_verify` to repopulate the cache to the correct length. Wire observer sees a constant-byte rollback per round; the §3.8 leak is closed.

**Channel 2: Operator-managed PSK distribution closure.** New `X25519AnchoredCipher` class — drop-in alternative to `ProbsCipher` with surface-compatible `encrypt`/`decrypt`/`encrypt_prefix`/`decrypt_prefix` methods. Per-request key derivation via ECDH on the executor's ephemeral X25519 public key (carried on `HandoffToken.ephemeral_pubkey`, signed under the existing settler-signature chain) + HKDF-SHA256 over `(request_id, stage_index, chain_total_stages)` → AES-256 key. Forward secrecy: per-request ephemeral keys mean compromise of one request's traffic doesn't compromise other requests'. Chain-length-forgery defense via the chain_total_stages binding in HKDF info. Operators wiring q.y' can drop the PSK rotation playbook entirely; long-term keys are anchor-signed and refreshable on the standard publisher-key cadence.

**Honest-scope residual under q.y'.** Multi-stage replay is best-effort: stage 0 (owns embeddings) drives the forward; non-stage-0 stages return False without raising because they need the upstream hidden state which the server doesn't have at rollback time. The wire-side leak is closed regardless; cache-state correctness on stage > 0 falls back to TTL sweeper bounds. Phase 3.x.11.q.y'' will lift this if multi-stage Tier C telemetry shows it materially affects deployments.

**Round-1 review caught one LOW issue.** Server-side decrypt failure originally surfaced as `MALFORMED_REQUEST`, but the truncation step had already run successfully — leaving the cache in a truncated-but-not-replayed state worse than just logging the decrypt failure and continuing. Fixed: convert the three failure paths (cipher unwired, missing decrypt_prefix method, all-K decrypt failures) to logger.warning + skip replay, consistent with the runner-side replay best-effort semantics.

Concretely:

- **~10 commits** pushed to `main` since the prior baseline tag (8 task commits + 1 audit-prep refresh + 1 LOW remediation)
- **~50 new tests** (12 wire-format + 6 HandoffToken + 9 X25519AnchoredCipher + 4 always_rollback_k + 3 E2E q.y' + helper-coverage growth across the existing suites)
- **1 new merge-ready tag** (`phase3.x.11.q.y-prime-merge-ready-20260430`)
- **1 new cumulative audit-prep tag** (`cumulative-audit-prep-20260430-d`) covering 14 streaming-inference phases (§7.1 through §7.14, with §7.14.1 covering the q.y' delta)
- **1 threat-model addendum revision** (1.4 → 1.5) updating §3.8 to reflect closure of both v1 residuals; mitigation table 2 rows updated; auditor reading path 3 entries added (14, 15, 16)
- **Zero regressions** across the full streaming-inference test surface (existing 558 q.y baseline tests pass unchanged)
- **Backwards-compat byte-equivalence** load-bearing pin on pre-q.y' rollbacks (no replay fields) AND pre-q.y' HandoffTokens (no ephemeral_pubkey) — both encode to byte-identical wire bytes via omit-when-None canonical encoding (test pins both)

This is a comparable-scope slice to sprint-7 (10 commits vs. sprint-7's 8) but covers two distinct residual channels with separate cryptographic primitives. The X25519 cipher is the heaviest single piece (~280 lines + 9 unit tests + new key-rotation E2E), but the surface-compatible Protocol design keeps the integration cost on `client.py` / `server.py` minimal.

---

## 2. Cumulative streaming-inference subsystem

After 8 sprints, the streaming-inference subsystem covers (read in order — each builds on the prior):

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
| **3.x.11.q.y'** | **Closure of v1 honest-scope residuals (drop-value leak + PSK distribution)** — **NEW THIS SPRINT** | **`phase3.x.11.q.y-prime-merge-ready-20260430`** |

**Cumulative test surface:** 592 unit + 7 slow real-distilgpt2 E2E across the streaming-inference subsystem. Every phase carries a round-1 independent code review with HIGH/MEDIUM remediations resolved pre-tag.

**Cumulative audit bundle:** `cumulative-audit-prep-20260430-d` — 14 sections (§7.1 through §7.14) of the per-phase auditor-facing summary plus §7.14.1 (q.y' delta), with the threat-model addendum at `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` (1.5 revision; 8 numbered threat-surface sections).

---

## 3. What sprint-8 means for the audit engagement

This sprint adds one new subsection to the auditor reading path (§7.14.1) and one threat-model revision (§3.8 → 1.5) without changing the engagement model or the per-phase tag list of prior sprints — every sprint-1 through sprint-7 tag is byte-identical at this point.

**The closure-of-residuals story is the auditor-relevant arc.** Sprint-7's q.y baseline was deliberately structured with E2E pins ASSERTING the residuals on the wire, so accidental quiet closure would be caught as a CI regression that required threat-model documentation updates in the same PR. Sprint-8 ships the closures as expected, flips those pins from "leak documented" to "leak absent", and updates §3.8 + §7.14 in one coordinated change. Auditors evaluating the q.y → q.y' transition can walk the diff and see that the wire-format extensions, the cipher surface, and the test pins were all coordinated rather than sprawling across multiple unrelated PRs.

**For an external auditor, this means:** the `TestAlwaysRollbackKE2E::test_e3_constant_k_rollback_pin` E2E test is the load-bearing real-model proof — `n_positions_to_drop == K + 1` on every observed rollback regardless of acceptance, plus `target_stage_index` is bound to a valid chain stage, plus the encrypted prefix is set (mutual-exclusion with plaintext field). The §7.14.1 trust-seam list flags six items including AAD distinctness probs ↔ rollback, backwards-compat byte-equivalence, and HandoffToken signing covering ephemeral_pubkey.

---

## 4. What's next on the PRSM critical path

After sprint-8, the streaming-inference subsystem has **one remaining named structural deferral** that fits a normal sprint scope:

**Phase 3.x.11.q.x** — per-stage cadence wrapping (full-network masking; complementary to the Phase 3.x.11.q chain-level decorator) + M2 response-size operator-configurable padding. Inherited from §7.13's honest-scope and reaffirmed in §7.14.1.

Plus several follow-up items at the residual level:
- **Phase 3.x.11.q.y''** — multi-stage replay forward path (closes the q.y' best-effort honest-scope residual on stage > 0). Optional; depends on whether multi-stage Tier C telemetry shows the gap materially affects deployments.
- **Per-stage nonce cache** for replay-window-inside-deadline_unix defense. Orthogonal; defer.

After q.x lands, the streaming-inference subsystem will have closed every named structural deferral and reduced to maintenance + research-track integration. That's a natural cap on the streaming roadmap and a reasonable point to pivot focus to other Phase 3 surfaces or the Foundation governance + audit engagement substrate.

The Foundation governance + audit engagement surface continues to track separately — that's where every streaming-inference tag eventually rolls into.

---

## 5. Changelog

- **0.1 (2026-04-30)** — initial sprint-8 summary covering Phase 3.x.11.q.y'. Tag `phase3.x.11.q.y-prime-merge-ready-20260430` at `cf03f821`. Cumulative audit-prep tag: `cumulative-audit-prep-20260430-d`.
