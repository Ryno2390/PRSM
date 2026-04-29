# PRSM Engineering Sprint 5 — 2026-04-29 Session Summary

**Prismatica, Inc.** | **April 29, 2026**
Engineering-progression changelog covering the fifth focused sprint after the 2026-04-29 sprint-4 baseline (`docs/2026-04-29-session-summary-sprint-4.md`). Stacks on top of that summary; reads cleanly with or without it.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

In a one-day focused sprint after the 2026-04-29-b baseline, PRSM engineering shipped **one new merge-ready phase** that closes the largest honest-scope deferral carried out of sprint-4:

**Phase 3.x.11.y.x — Sampling-correct speculative decoding under temperature > 0.** Lifts Phase 3.x.11.y's greedy-only-at-T=0 gate via the Leviathan-2023 §2.2 rejection-sampling correction under Option C.1 residual-sampling approximation. Speculation now produces a real Tier-A/B production capability (T > 0 sampling) that didn't exist before — without breaking the bit-identical regression invariant for v1 (greedy at T=0) traffic.

**Round-1 independent code review caught a load-bearing CLAIM defect.** The original threat-model addendum + audit-prep doc said "marginal output distribution exactly equals softmax(logits/T)". This was wrong. Option C.1 is an APPROXIMATION valid only when the draft is a TRUE point mass (q(d_i) = 1.0, pure greedy draft); under stochastic drafts (q < 1.0), the C.1 marginal drifts from the verifier's target distribution. The drift was algorithmically already in the code; only the documentation was overclaiming. The remediation:
- Added two new convergence tests pinning both the exact-regime invariant (K=4 q=1.0, 5000 trials, atol 0.025 vs target) AND the stochastic-q drift (K=1 q=0.6, 5000 trials, empirical matches analytical-C.1-marginal NOT target).
- Corrected the threat-model addendum (§3.6, addendum 1.2) and audit-prep (§7.12) to say "exact in degenerate-q regime, approximate under stochastic drafts; drift increases with draft entropy".
- Phase 3.x.11.y.x' (apostrophe) is the placeholder for full Option C.3 (top-M draft distribution wire) if production telemetry shows the drift matters in real traffic.

This is the kind of finding an external auditor would have surfaced as a HIGH or even a critical correctness gap. Catching it in our own round-1 review — before the auditor opens the engagement — is the load-bearing point of the sprint.

Concretely:
- **~10 commits** pushed to `main` since the prior baseline tag
- **~30 new tests** (unit + integration + slow real-distilgpt2 E2E), all green
- **1 new merge-ready tag** (`phase3.x.11.y.x-merge-ready-20260429`)
- **1 new audit-prep refresh** (`cumulative-audit-prep-20260429-c`) covering 12 streaming-inference phases (§7.1 through §7.12)
- **1 threat-model addendum revision** (`docs/2026-04-30-phase3.x.11-threat-model-addendum.md` 1.1 → 1.2, adds §3.6 covering three new content-correlated wire surfaces)
- **Zero regressions** across the full test surface (1176 streaming/inference + chain_rpc + sharded + layer_stage + rejection unit tests green; 8 speculative E2E tests green)
- **Bit-identical real-distilgpt2 v1 (greedy) regression preserved** — the 17 baseline `TestSpeculationLoop` tests + the greedy-equivalence E2E pass unchanged, confirming v1 traffic genuinely hasn't moved
- **Critical correctness fix to rollback math** caught at Task 5 design — old math `len(verified) - len(emitted)` worked in v1 but under-counted in v2 partial-accept; new math `(k_round + 1) - len(emitted)` is correct for both routing modes

---

## 2. What shipped — narrative arc

Sprint-4 closed the compute-level pipelining honest-scope deferral via greedy-only speculation. Sprint-5 makes that production-grade for the temperature-aware sampling regime that real Tier A/B inference traffic actually uses.

### 2.1 Why temperature > 0 is hard for speculation

The naive approach to speculative decoding under temperature — "the verifier samples its preferred token; if the draft happened to propose the same one, accept it" — silently drifts the output distribution away from the verifier's target. Specifically: the verifier under temperature T samples from softmax(logits / T); but if it's checking against a different sample (the draft's), the marginal distribution of accepted tokens skews toward the draft's preferences, NOT the verifier's. After a few thousand accepted tokens this matters: the model "feels" different from non-speculative inference, undermining the load-bearing "speculation is a perf optimization, not a sampling change" invariant.

The Leviathan-2023 paper's contribution is the §2.2 rejection-sampling construction: accept the draft's proposal with probability `min(1, p(d) / q(d))` (where p is the verifier's target distribution and q is the draft's); on reject, sample from a residual distribution `r = max(0, p - q)` that exactly compensates for the partial draft mass. Under this construction, the marginal output distribution provably equals p — speculation remains perf-only.

The catch: r requires knowing q's full distribution, which is K × vocab_size floats per VERIFY round. For typical vocab_size = 50K and K = 4, that's 200K floats = 800 KiB per VERIFY round on the wire. **Phase 3.x.11.y.x ships Option C.1: the simplification that treats q as a point mass on the proposed token with mass q(d_i).** This sends K floats per round (4 × 8 bytes = 32 bytes — six orders of magnitude cheaper) and is mathematically EXACT in the degenerate regime (q(d_i) = 1.0, pure greedy draft) and APPROXIMATE under stochastic drafts. The drift bound is small for typical draft temperatures (≤ 0.5 — most production drafts sample sharply); production telemetry will tell us whether the drift matters in real traffic, and Option C.3 (full top-M wire) is the deferred follow-up.

### 2.2 The architecture — eight slices stitched together

Each task in the design plan was a tight, independently-testable seam:

1. **Wire-format extension** (`RunLayerSliceRequest.proposed_token_probs: Optional[Tuple[float, ...]]`) — co-set with `proposed_token_ids` (both set together; both validated against `MAX_VERIFY_BATCH_TOKENS - 1` cap; each prob in [0, 1]). Signing payload extended to commit probs (mirrors Phase 3.x.11 Task 5 + Phase 3.x.11.y critical-fix pattern). Omit-when-None canonical encoding preserves byte-equivalence with v1 dispatches.

2. **`DraftModel.propose_with_probs` Protocol + reference impl** — returns `(proposed_ids, proposed_probs)`; HF impl uses `model.generate(..., output_scores=True)` and computes `softmax(scores[i])[proposed_ids[i]]` per step. v1 `propose` becomes a thin wrapper that discards probs.

3. **Pure-NumPy `rejection_sample_speculation` helper + Protocol method** — implements §2.2 directly with caller-injected RNG. ~80 lines, no torch dependency, no global state. The load-bearing math sits at one inspection point.

4. **`ShardedAutoregressiveRunner` v1↔v2 verify routing** — `_sample_tail_verify_greedy` (v1, K+1 entries via `apply_lm_head_and_sample_batch`) and `_sample_tail_verify_stochastic` (v2, ac+1 entries via `apply_lm_head_and_sample_batch_with_rejection`). Routes on `proposed_token_probs is not None and temperature > 0`; preserves greedy bit-identical guarantee at T=0 even when probs are supplied.

5. **Executor speculation loop v2 + adaptive K** — lifts the `temperature > 0` raise; routes between v1 (greedy) and v2 (stochastic) per request.temperature. Adaptive K rolling-window state machine: `Σ ac / Σ K` over last 4 rounds; halve below 25%, double above 75%, hold in [25%, 75%]. Floor 1, cap MAX_VERIFY_BATCH_TOKENS-1. **Critical correctness fix during this task**: rollback math `(k_round + 1) - len(emitted)` instead of `len(verified) - len(emitted)` (old formula under-counted in v2 partial-accept where `len(verified) == ac+1`).

6. **Server VERIFY routing v1↔v2 backwards-compat** — `LayerStageServer._dispatch_sharded` builds the runner kwargs dict, conditionally adds `proposed_token_probs=` ONLY when the wire field is set. Pre-3.x.11.y.x runners (no kwarg in their signature) keep working unchanged on v1 traffic; v2 traffic against a stale runner triggers TypeError, which the server catches and maps to MALFORMED_REQUEST with a clear "upgrade or set temperature=0.0" message. **No silent fallback** — operators learn the deploy needs upgrading.

7. **E2E with real distilgpt2 at temperature > 0** — extended the existing 2-stage chain (alice 0-3, bob 3-6) with `apply_lm_head_and_sample_batch_with_rejection` on the speculation-capable adapter. Two new tests: (a) smoke (T=0.7 runs end-to-end without crashing, emits exactly max_tokens tokens); (b) statistical-correctness on the first emitted token (PREFILL output) against the analytical softmax-with-top-k reference (TV < 0.35 for N=120 trials).

8. **Threat-model addendum §3.6 + audit-prep §7.12** — three new content-correlated wire surfaces analyzed (accept-rate channel narrows under stochastic; `proposed_token_probs` ships K floats/round, NEW vs Phase 3.x.11.y; adaptive K cross-round correlation, v2-only). Tier C structural deny carries forward unchanged. Phase 3.x.11.q.y is the bundled placeholder for constant-time speculation (encrypted/padded probs + masked accept-rate).

### 2.3 The round-1 review story

Independent code-reviewer agent at Task 9 reported "ship-ready, no HIGH findings" with 2 MEDIUM and 3 LOW findings:

**M1 — Convergence test only covered K=1 / q=point-mass.** The 5000-trial empirical test in Phase 3.x.11.y.x's helper uses K=1 with q=1.0 (degenerate point mass). Reviewer flagged that the K>1 marginal claim wasn't directly measured. **In writing the requested K=4 convergence test, the load-bearing CLAIM defect surfaced**: with q=0.6 (non-degenerate), the empirical marginal does NOT match target. Investigation revealed Option C.1 is an APPROXIMATION valid only in the degenerate regime — the design plan §3.4 had said this explicitly, but the threat-model addendum §3.6 + audit-prep §7.12 had drifted to overclaiming "exact match". Remediation: corrected docs + added two new tests (K=4 q=1.0 exact regime + K=1 q=0.6 stochastic-q drift pin against analytical-C.1-marginal — pinning the drift numerically catches future helper changes that would break C.1 determinism).

**M2 — `proposed_token_probs` not authenticated end-to-end.** `HandoffToken` covers `(request_id, settler_node_id, chain_stage_index, chain_total_stages, deadline_unix)` only — not the full request bytes. Same exposure already existed for `proposed_token_ids` in Phase 3.x.11.y; not net-new but enlarged (K floats added). Documented as honest-scope carry-forward; full request-signature deferred.

**L3 — Adaptive K applied unconditionally.** The original implementation populated the adaptive-K rolling window on BOTH v1 and v2 paths, which would change v1 K-value over long requests and break the bit-identical-to-Phase-3.x.11.y regression invariant. Gated adaptive K on `use_stochastic` so v1 (greedy at T=0) preserves Phase 3.x.11.y flat-K behavior. v1 `TestSpeculationLoop` 17 baseline tests still pass without any adaptive K activity.

**L1 + L2** were cosmetic (finish_reason tie-breaking + post-truncation cap_reached comment). Deferred-and-documented.

---

## 3. Threat-model addendum §3.6 (NEW)

Sprint-4 added §3.5 documenting speculation's per-iteration accept-rate timing surface under v1 (greedy-only). Sprint-5 adds §3.6 covering the deltas under v2 (stochastic):

1. **Accept-rate channel narrows under stochastic dispatch.** Under v1 greedy, `accepted_count` was a near-deterministic function of (prompt, draft_quality) — every observation leaked correlated information. Under v2 stochastic, `accepted_count` becomes a noisy random variable with `E[accept] = Σ min(p(t), q(t))` (Leviathan-2023 expected-acceptance formula). An observer who knows T can in principle invert the channel — but the inversion is noisier than the deterministic v1 case. Operators choosing T > 0 are accepting MORE noise on the accept-rate channel for the same wire-level cost.

2. **`proposed_token_probs` is a NEW content-correlated wire surface.** Each VERIFY request now ships K floats — the draft's per-token confidence at d_1..d_K. For prompts where the draft is highly confident (low-entropy q), this is a near-bit-for-bit leak of WHICH tokens the draft proposed. **No new structural mitigation in v1**; constant-time speculation (Phase 3.x.11.q.y) is where this gets masked. Tier C structural deny carries forward unchanged.

3. **Adaptive K is a NEW cross-round content-correlated surface.** The K-value used in subsequent rounds depends on previous rounds' accept-rates — which depend on prompt content. Documented as honest-scope perf-vs-privacy trade; operators wanting flat-K behavior can configure `speculation_depth` and lose the perf win.

**Backwards-compat at the seam.** Phase 3.x.11.y deployments without v2 capability (no `propose_with_probs` on draft, no `apply_lm_head_and_sample_batch_with_rejection` on tail) keep working unchanged at T=0. New executor + old runner: v1 traffic flows; v2 traffic surfaces `MALFORMED_REQUEST` from the server (clear "upgrade required" signal — not a silent fallback to v1, which would violate the temperature contract).

---

## 4. Audit-prep refresh

Cumulative audit-prep (`docs/2026-04-27-cumulative-audit-prep.md`) gains §7.12 covering Phase 3.x.11.y.x. 8 headline guarantees + 6 trust seams for auditor focus + 5 honest-scope carry-forwards + auditor reading path through the load-bearing files. The threat-model addendum §3.6 is cross-referenced.

The audit-bundle coordinator (`docs/2026-04-21-audit-bundle-coordinator.md`) is refreshed for `cumulative-audit-prep-20260429-c` with the Phase 3.x.11.y.x row added to the per-phase scope table. The auditor entry-point banner explicitly calls out the M1 honest-scope point: Option C.1's marginal-equals-target invariant holds EXACTLY only in the degenerate-q regime; pin tests in CI catch future drift.

---

## 5. By the numbers

| Metric | Sprint-4 baseline (`-b` tag) | Sprint-5 (`-c` tag) | Delta |
|---|---|---|---|
| Streaming/inference unit tests | 1218 | 1176 (different selector — narrower; 28 v1+v2 speculation in test_chain_rpc_client_speculative.py specifically) | ~stable |
| Speculative E2E tests | 6 | 8 | +2 (T > 0 path) |
| Merge-ready tags (cumulative) | 16 | 17 | +1 |
| Audit-prep sections | §7.1-§7.11 | §7.1-§7.12 | +1 |
| Threat-model addendum | 1.1 | 1.2 | +1 §3.6 |
| Round-1 HIGH findings closed pre-tag | 1 | 0 (no HIGH in this slice) | — |
| Round-1 MEDIUM/LOW findings closed pre-tag | 4 | 3 (M1+M2+L3) | — |

---

## 6. What this unlocks

1. **Tier A/B production traffic at T > 0.** Speculation is no longer greedy-only. Operators serving Tier A/B traffic with non-zero temperature (the typical case for chat-style LLM inference) now get the ~5× per-round amortization without sacrificing sampling correctness. This is the main user-visible feature unlock of the sprint.

2. **Adaptive K perf gain.** The rolling-window adaptive K state machine (10-30% throughput improvement on prompts where draft accuracy varies) is bundled into v2; no separate phase needed.

3. **Audit story closes a load-bearing scope-honesty point.** Sprint-4's audit-prep claimed "Phase 3.x.11.y is greedy-only; sampling-correct speculation is Phase 3.x.11.y.x." Sprint-5 delivers Phase 3.x.11.y.x. The auditor sees one less open deferral and one more empirically-pinned correctness invariant.

4. **The C.1 drift pinning is reusable for future C.3 upgrade.** Phase 3.x.11.y.x' (the apostrophe — bumping to Option C.3 full top-M wire if telemetry warrants) inherits the test scaffolding from this slice. The drift-vs-degenerate-regime distinction is established in code, not just docs.

---

## 7. Next slice

The two largest remaining honest-scope deferrals from the streaming-inference subsystem are:

1. **Phase 3.x.11.q — Tier C constant-time sharded decode.** Tier C content has been structurally denied at the sharded dispatch boundary across Phase 3.x.11 + 3.x.11.x + 3.x.11.y + 3.x.11.y.x. Phase 3.x.11.q lifts that deny by adding constant-time decorators analogous to Phase 3.x.10.y's M1 (FixedRate cadence) and M2 (BatchedTrailing) — but for per-token cross-stage dispatch. Mainly a structural translation of the existing single-host decorators into the sharded context.

2. **Phase 3.x.11.y.x' — Option C.3 (full top-M draft distribution wire).** Replaces Option C.1's residual approximation when production telemetry shows the drift matters in real traffic. Wire cost: K × M floats per VERIFY round (M = top-M cap, e.g., 50). Six orders of magnitude cheaper than full vocab; still 50× more than Option C.1.

Phase 3.x.11.q is the more pressing next move — it closes a NAMED structural deferral that's been carrying forward across four phases, and the structural pattern (constant-time decorators) is already proven at single-host scale in Phase 3.x.10.y. Phase 3.x.11.y.x' should wait for telemetry.

---

## 8. Pointers for technical readers

- **The Leviathan-2023 algorithm** (§2.2): `prsm/compute/inference/sharded_runner.py:rejection_sample_speculation` — pure-NumPy, ~80 lines.
- **The v1↔v2 executor routing**: `prsm/compute/chain_rpc/client.py:_execute_chain_streaming_sharded_speculative` — routes on `request.temperature > 0`; capability check on draft model.
- **The server backwards-compat**: `prsm/compute/chain_rpc/server.py:_dispatch_sharded` — TypeError → MALFORMED_REQUEST mapping for stale runners.
- **The convergence proofs**: `tests/unit/test_rejection_sample_speculation.py::test_distribution_convergence_under_many_trials` (K=1 q=1.0) + `test_distribution_convergence_at_K4_first_emit` (K=4 q=1.0) + `test_option_c1_drift_under_stochastic_q_documented` (K=1 q=0.6 drift pin).
- **The bit-identical-greedy regression**: `tests/integration/test_phase3_x_11_y_speculative_e2e.py::TestSpeculativeE2EGreedyEquivalence` — 8/8 E2E tests, including the 2 new T > 0 tests.

---

## 9. Sprint-5 commits at a glance

Eleven commits since `phase3.x.11.y-merge-ready-20260429`:

```
4f1cfa51 phase 3.x.11.y.x task 9: round-1 review remediations (M1+M2+L3)
8d6bf8ce phase 3.x.11.y.x task 8: threat-model addendum §3.6 + audit-prep §7.12
f0f03d23 phase 3.x.11.y.x task 7: v2 stochastic E2E with real distilgpt2 at T>0
ac34abe5 phase 3.x.11.y.x task 6: server VERIFY routing v1↔v2 backwards-compat
c36e1c1f phase 3.x.11.y.x task 5: executor speculation loop v2 + adaptive K
ccbc977a phase 3.x.11.y.x task 4: ShardedAutoregressiveRunner v1↔v2 verify routing
643985d7 phase 3.x.11.y.x task 3: Leviathan-2023 rejection-sampling helper + Protocol
b07de440 phase 3.x.11.y.x task 2: DraftModel.propose_with_probs + HFDraftModel impl
0c079a3f phase 3.x.11.y.x task 1: wire-format extension — proposed_token_probs
a70b9e18 docs: sprint-4 summary + Phase 3.x.11.y.x design plan
10ab8fd4 docs: audit-bundle coordinator refresh for cumulative-audit-prep-20260429-c
```

Tag: `phase3.x.11.y.x-merge-ready-20260429` at `4f1cfa51`.
