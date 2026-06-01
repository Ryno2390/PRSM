# PRSM Engineering Sprint 6 — 2026-04-30 Session Summary

**Prismatica, Inc.** | **April 30, 2026**
Engineering-progression changelog covering the sixth focused sprint after the 2026-04-29 sprint-5 baseline (`docs/2026-04-29-session-summary-sprint-5.md`). Stacks on top of that summary; reads cleanly with or without it.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

In a one-day focused sprint after the 2026-04-29-c baseline, PRSM engineering shipped **one new merge-ready phase** that closes the most-cited remaining structural deferral from the streaming-inference subsystem:

**Phase 3.x.11.q — Tier C constant-time sharded decode.** Closes the named "Phase 3.x.11.q" structural deferral that has been carrying across four prior phases (3.x.11 + 3.x.11.x + 3.x.11.y + 3.x.11.y.x). Tier C content can now flow through sharded autoregressive decode at Tier A/B perf characteristics via two operator-wired chain-level decorators: `BatchedTrailingShardedExecutor` (M2 — single trailing frame) and `FixedRateShardedExecutor` (M1 — cadence-driven yield). Routing through `ParallaxScheduledExecutor` enforces a no-silent-fallback invariant — Tier C without a wired decorator surfaces a structured failure naming the factory.

**Round-1 independent code review caught two MEDIUM defensive-coding issues.** The original M2 implementation drained the entire inner stream regardless of event order, which would have silently merged tokens emitted AFTER a `ChainExecutionResult` into the joined text — re-ordering content across the terminal boundary. The text_delta join also wasn't defended against non-string upstream values (None, bytes), which would have TypeError'd mid-generator and crashed the whole Tier C request. Both remediations are 3-line fixes pinned by dedicated tests; no production behavior change for compliant upstreams.

This is a structurally smaller slice than sprint-5 (6 commits + 1 round-1-remediation commit vs. sprint-5's 11) but closes a more-load-bearing honest-scope point: the Tier C structural deny in §3.4 + §3.5 + §3.6 of the threat-model addendum was the most-cited cross-phase deferral.

Concretely:
- **~7 commits** pushed to `main` since the prior baseline tag
- **~37 new tests** (27 unit + 6 routing integration + 3 real-distilgpt2 E2E + 1 round-1 test replacement); 82 streaming/inference cumulative
- **1 new merge-ready tag** (`phase3.x.11.q-merge-ready-20260429`)
- **1 new audit-prep refresh** (`cumulative-audit-prep-20260430-b`) covering 13 streaming-inference phases (§7.1 through §7.13)
- **1 threat-model addendum revision** (`docs/2026-04-30-phase3.x.11-threat-model-addendum.md` 1.2 → 1.3, adds §3.7 covering chain-level vs per-stage scope-honesty point)
- **Zero regressions** across the full test surface (existing 39 parallax_executor + 8 sharded E2E tests pass unchanged after the routing-layer integration; Tier A/B path is bit-identical to sprint-5)
- **Real-distilgpt2 E2E proves the timing-mask invariants end-to-end** — M2 emits exactly 1 trailing StreamToken regardless of `max_tokens=4`; M1's inter-token intervals are clamped to ≥ 50ms cadence

---

## 2. What shipped — narrative arc

Sprint-5 shipped sampling-correct speculation under T > 0 (Phase 3.x.11.y.x), unlocking real Tier A/B production traffic at non-zero temperature. That left two large honest-scope deferrals carried forward: (a) Tier C content was still structurally denied at the sharded dispatch boundary, and (b) the per-token timing surface was unmasked even for Tier A/B traffic. Sprint-6 closes (a).

### 2.1 The decorator-pattern translation

Phase 3.x.10.y shipped constant-time padding decorators (`BatchedTrailingStreamingRunner` = M2; `FixedRateStreamingRunner` = M1) for the single-host streaming path — wrapping `StreamingLayerRunner` at the per-stage server boundary. Sharded decode dispatches `T` per-stage RPCs per generated token, each completing at its own per-stage rate; M1/M2 never see those events because they live above `LayerStageServer.handle` (sharded uses `_dispatch_sharded`, not `handle_token_stream`).

**Phase 3.x.11.q's structural insight.** The decorator location must move from the per-stage runner (where 3.x.10.y wrapped) to the chain-executor's streaming surface (where the executor → caller iterator emits). The chain-level decorator drains the inner generator and gates emission on the outer iteration; the chain runs at native speed while the wire observation is masked.

Two decorators in this slice mirror the 3.x.10.y pattern at the new location:

- **`BatchedTrailingShardedExecutor` (M2).** Drains the inner executor's full stream, then emits ONE `StreamToken` (joined text) followed by ONE `ChainExecutionResult`. From a wire observer on the executor → caller path, exactly two events appear regardless of how many tokens the inner chain produced or at what per-token cadence. Sacrifices streaming UX for maximum executor-wire leak elimination.

- **`FixedRateShardedExecutor` (M1).** Each `StreamToken` from the inner executor is held until ≥ `cadence_seconds` have elapsed since the previous yield. Chain runs at native speed; decorator's `yield` gates emission. Inter-StreamToken intervals on the executor → caller wire are clamped to ≥ cadence regardless of per-token chain compute variance. Operator MUST set cadence ≥ chain native rate (recommended: 2× measured native).

A `make_tier_c_sharded_executor(inner, *, mode, cadence_seconds)` factory mirrors Phase 3.x.10.y's mode-string pattern.

### 2.2 The routing-layer integration

`ParallaxScheduledExecutor` is the trust-policy boundary — it already handles content-tier and privacy-tier gates before any chain dispatch. Phase 3.x.11.q adds a `tier_c_chain_executor: Optional[Any]` constructor kwarg and routes Tier C streaming requests through it:

- Tier A/B → default `chain_executor` (unchanged from sprint-5; **bit-identical regression preserved**)
- Tier C + `tier_c_chain_executor` wired → decorator
- Tier C + decorator unwired → structured `InferenceResult.failure(...)` naming `make_tier_c_sharded_executor`. **No silent fallback to the leaky path.**

The construction-time defense rejects a `tier_c_chain_executor` without `execute_chain_streaming` at `__init__` with a clear error naming the factory — operator misconfig surfaces at server-start time, not first-Tier-C-request time.

The per-stage `_dispatch_sharded` TIER_GATE deny stays in place as defense-in-depth. A misconfigured executor that tries to send Tier C dispatches directly to stages still gets rejected at the stage. The chain-level decorator is the trust-policy boundary, not the per-stage server.

### 2.3 The round-1 review story

Independent code-reviewer agent at Task 7 reported "ship-ready, no HIGH findings" with 2 MEDIUM and 3 LOW findings:

**M1 — M2 decorator out-of-order event handling.** The original implementation drained the entire inner stream regardless of order. If a future inner ever emitted token-then-result-then-token (which would be a bug in the inner, but defensive code should fail loud not silently corrupt), the post-terminal token would silently merge into the joined text — re-ordering content across the terminal boundary. **Remediation:** eager break on receipt of `ChainExecutionResult`; post-terminal tokens are explicitly DROPPED. Pinned by `test_post_result_tokens_dropped_round1_m1`.

**M2 — non-str text_delta defensive coerce.** The original `"".join(t.text_delta for t in tokens)` would TypeError mid-generator if upstream ever shipped `text_delta=None` or a non-string type. The `StreamToken` dataclass types text_delta as `str` but `RpcChainExecutor` doesn't enforce at runtime. **Remediation:** defensive coerce — `str(t.text_delta) if t.text_delta is not None else ""`. Pinned by `test_non_str_text_delta_coerced_round1_m2`.

**L1+L2+L3** were verified-correct or intentionally-structural; deferred-and-documented.

---

## 3. Threat-model addendum §3.7 (NEW)

Sprint-6 adds §3.7 to the Phase 3.x.11 threat-model addendum (1.3 revision) covering Phase 3.x.11.q chain-level Tier C constant-time decorators. The load-bearing scope-honesty point:

**The chain-executor decorator masks the executor → caller wire ONLY.** The executor → per-stage path continues to dispatch at chain native rate. A network observer with visibility into a single stage's transport learns the raw per-token cadence — the §3.1 timing surface is intact AT THE STAGE LEVEL.

For full-network masking, operators would compose the chain-level decorator with per-stage cadence wrappers (analogous to Phase 3.x.10.y's `FixedRateStreamingRunner` on each stage's `StreamingLayerRunner`). Phase 3.x.11.q.x is the bundled placeholder for that work; the per-stage variant requires per-stage coordination that's structurally harder than the single-host pattern and was left out of the sprint-6 scope.

**Speculation under Tier C remains structurally denied.** Phase 3.x.11.y.x's three new content-correlated wire surfaces (accept-rate channel, `proposed_token_probs` per VERIFY round, adaptive K's cross-round correlation) are NOT covered by chain-level decorators. Phase 3.x.11.q.y is the bundled placeholder that composes today's decorators with encrypted/padded probs + masked accept-rate + flat-K mode.

---

## 4. Audit-prep refresh

Cumulative audit-prep (`docs/2026-04-27-cumulative-audit-prep.md`) gains §7.13 covering Phase 3.x.11.q. 7 headline guarantees + 6 trust seams for auditor focus + 5 honest-scope items + auditor reading path. The threat-model addendum §3.7 is cross-referenced.

The audit-bundle coordinator (`docs/2026-04-21-audit-bundle-coordinator.md`) is refreshed for `cumulative-audit-prep-20260430-b` with the Phase 3.x.11.q row added to the per-phase scope table. The auditor entry-point banner explicitly calls out the load-bearing trust-policy boundary: the no-silent-fallback invariant. Tier C without a wired decorator surfaces a structured failure naming the factory — confirmed by `tests/unit/test_parallax_executor.py::TestTierCRoutingIntegration::test_tier_c_without_decorator_surfaces_failure` asserting BOTH the failure presence AND that the default chain_executor is NOT touched.

---

## 5. By the numbers

| Metric | Sprint-5 baseline (`-c` tag) | Sprint-6 (`-b` 2026-04-30 tag) | Delta |
|---|---|---|---|
| Streaming/inference unit + integration tests | 1184 | 1262 (with chain_rpc + tier_c + parallax_executor + sharded E2E selected) | +78 (most are pre-existing tests now collected by broader selectors) |
| Tier C decorator-specific tests | 0 | 27 | +27 |
| Real-distilgpt2 E2E tests (sharded) | 8 | 11 | +3 |
| Routing-layer integration tests | 0 (Tier C unsupported) | 6 | +6 |
| Merge-ready tags (cumulative) | 17 | 18 | +1 |
| Audit-prep sections | §7.1-§7.12 | §7.1-§7.13 | +1 |
| Threat-model addendum | 1.2 | 1.3 | +1 §3.7 |
| Round-1 HIGH findings closed pre-tag | 0 (no HIGH) | 0 (no HIGH) | — |
| Round-1 MEDIUM findings closed pre-tag | 2 | 2 (M1+M2) | — |

---

## 6. What this unlocks

1. **Tier C operators get sharded decode for the first time.** Until sprint-6, Tier C content was structurally denied at the sharded dispatch boundary — Tier C operators were stuck on Phase 3.x.10.y single-host streaming (loses the cross-host inference perf path entirely). Sprint-6 lets them opt-in to sharded decode with chain-level masking, paying the streaming-UX cost (M2) or the cadence-overhead cost (M1) for the timing surface.

2. **The trust-policy boundary moves from per-stage to chain-executor cleanly.** `ParallaxScheduledExecutor` is the natural place for content-tier policy decisions; the routing-layer integration is mechanically simple (one `if` branch + structured failure on misconfig). Operators wire one decorator per Tier C policy at config time.

3. **No-silent-fallback invariant is the load-bearing operator-experience point.** A misconfigured deploy doesn't silently leak — it surfaces a clear "Tier C streaming requires Phase 3.x.11.q constant-time decorator — wire `tier_c_chain_executor=` via `make_tier_c_sharded_executor(...)`" message. The auditor sees one less misconfig vector.

4. **The decorator pattern is reusable for Phase 3.x.11.q.x and 3.x.11.q.y.** The chain-level seam is now established; the follow-up phases (per-stage cadence wrapper for full-network masking; speculation-aware constant-time bundle) compose with what shipped today rather than adding new architectural seams.

---

## 7. Next slice

Two paths sit ahead of Phase 3.x.11.q:

1. **Phase 3.x.11.q.y — constant-time speculation (recommended).** Composes today's M1/M2 decorators with encrypted/padded `proposed_token_probs` + masked `accepted_count` + flat-K mode. Lifts the "speculation + Tier C still denied" line that §3.7 just documented. Most-cited remaining honest-scope deferral.

2. **Phase 3.x.11.q.x — per-stage cadence wrapper for full-network masking + M2 response-size padding.** Closes the "per-stage wire still leaks" honest-scope point. Touches more of the codebase (per-stage runner) but fundamentally completes the masking story.

Phase 3.x.11.q.y is the recommended next slice — structurally smaller (composes existing decorators rather than adding per-stage machinery), closes a more-immediately-cited deferral.

---

## 8. Pointers for technical readers

- **Chain-level decorators:** `prsm/compute/chain_rpc/tier_c_sharded_executors.py` — both decorators (M1 + M2). ~250 lines.
- **Factory:** `prsm/compute/chain_rpc/factories.py:make_tier_c_sharded_executor` — mode-string selection.
- **Routing-layer integration:** `prsm/compute/inference/parallax_executor.py:execute_streaming` — search "Phase 3.x.11.q" to find the routing block.
- **Round-1 remediations:** `prsm/compute/chain_rpc/tier_c_sharded_executors.py:103-138` (M1 post-terminal drop) + `:130-138` (M2 str coerce).
- **No-silent-fallback test:** `tests/unit/test_parallax_executor.py::TestTierCRoutingIntegration::test_tier_c_without_decorator_surfaces_failure` — asserts BOTH failure presence AND default chain_executor untouched.
- **Real-distilgpt2 E2E:** `tests/integration/test_phase3_x_11_sharded_e2e.py::TestTierCShardedDecoratorsE2E` — 3 tests proving timing-mask invariants end-to-end.

---

## 9. Sprint-6 commits at a glance

Eight commits since `cumulative-audit-prep-20260429-c`:

```
2241f8b9 docs: fixup audit-bundle tag pointer to -b suffix (existing 20260430 tag)
7999e5ea docs: audit-bundle coordinator refresh for cumulative-audit-prep-20260430
0dac3651 phase 3.x.11.q task 7: round-1 review remediations (M1+M2)
8862f953 phase 3.x.11.q task 6: threat-model addendum §3.7 + audit-prep §7.13
769f1118 phase 3.x.11.q task 5: E2E with mocked timing observer + real distilgpt2
7b7ac75a phase 3.x.11.q task 4: ParallaxScheduledExecutor Tier C routing-layer integration
d5fbd466 phase 3.x.11.q task 3: module exports + make_tier_c_sharded_executor factory
d7e3bb10 phase 3.x.11.q task 2: FixedRateShardedExecutor (M1) tests
b7b32ab2 phase 3.x.11.q task 1: BatchedTrailingShardedExecutor (M2)
4615e48b docs: sprint-5 summary + Phase 3.x.11.q design plan
```

Tag: `phase3.x.11.q-merge-ready-20260429` at `0dac3651`.
