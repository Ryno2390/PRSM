# PRSM Engineering Sprint 4 — 2026-04-29 Session Summary

**Prismatica, Inc.** | **April 29, 2026**
Engineering-progression changelog covering the fourth focused sprint after the 2026-04-30 sprint-3 baseline (`docs/2026-04-30-session-summary-sprint-3.md`). Stacks on top of that summary; reads cleanly with or without it.

**Audience:** investors tracking engineering velocity, technical partners evaluating integration readiness, Foundation officers planning execution-phase allocation, external auditors evaluating engagement scope.

---

## 1. Headline

In a one-day focused sprint after the 2026-04-30 baseline, PRSM engineering shipped **two new merge-ready phases** that close the two largest honest-scope deferrals carried out of sprint-3:

1. **Phase 3.x.11.x — Wire-level pipelining + per-iteration receipt attestation.** Closes the threat-model addendum §3.2 "no per-iteration cryptographic commitment in receipt" gap. Receipts now prove "stage K served EVERY dispatch", not just the last one. Chunked + sharded PREFILL composition lifts the unary-only guard for prompt-prefill (single-position INCREMENTAL stays unary-only by design — chunking single-position activations doesn't amortize).

2. **Phase 3.x.11.y — Speculative decoding (compute-level pipelining).** Closes the load-bearing compute-level pipelining honest-scope deferral. K=4 candidate tokens per round proposed by a co-located draft model; the sharded chain verifies them in a batched K+1-position forward; the executor accepts the longest matching prefix and broadcasts ``RollbackCacheRequest`` for the rejected suffix. Under matched draft + verifier (perfect-accept under greedy), 5× per-round amortization is the design point.

**Two-round independent code review on both phases.** Round-1 caught **one HIGH-severity finding** that would have silently corrupted non-tail KV-cache state in production speculation deployments — the existing rollback unit tests masked it because they pre-seeded a tail-shaped state, and the E2E used `draft == verifier` (perfect-accept means rollback never fires in the green path). The bug + the test gap are both closed pre-tag.

Concretely:
- **~9 commits** pushed to `main` since the prior baseline tag
- **~140 new tests** (unit + integration + slow real-distilgpt2 E2E), all green
- **2 new merge-ready tags** (`phase3.x.11.x-merge-ready-20260430`, `phase3.x.11.y-merge-ready-20260429`)
- **1 new audit-prep refresh** (`cumulative-audit-prep-20260429-b`) covering both phases
- **1 threat-model addendum revision** (`docs/2026-04-30-phase3.x.11-threat-model-addendum.md` 1.0 → 1.1, adds §3.5 covering speculation's per-iteration accept-rate timing surface)
- **Zero regressions** across the full test surface (1218 streaming/inference + chain_rpc unit tests green; 6 slow real-distilgpt2 E2E green)
- **Bit-identical real-distilgpt2 speculative vs single-host greedy** — speculation is a perf optimization, not a sampling change, and v1 ships with that invariant cryptographically pinned

---

## 2. What shipped — narrative arc

The two phases tell one continuous story: how the streaming-inference subsystem went from "honest about per-token wire latency tax" (sprint-3) to "production-grade pipelining with cryptographic commitments at every iteration" (sprint-4).

### 2.1 Wire-level pipelining + per-iteration receipts (Phase 3.x.11.x)

Sprint-3's Phase 3.x.11 closed the load-bearing tail-only contract by making each chain stage run once per generated token (sharded autoregressive decode). The threat-model addendum §3.2 noted that the receipt's signature still committed to wire activations only — NOT to the in-stage cache state. A malicious stage that swapped its KV-cache mid-INCREMENTAL could produce a "valid" receipt while serving manipulated K/V.

**Phase 3.x.11.x closes that gap.** Receipts now carry an ``IterationAttestation`` envelope with one ``StageAttestation`` per stage PER iteration (not flattened to "the last iteration only"). The envelope rides under a separate ``PRSM-MI-ATT-V1`` magic prefix; non-sharded receipts are byte-equivalent with pre-3.x.11.x signed bytes (golden-bytes pin enforces this).

The same slice opens chunked + sharded PREFILL composition. The 3.x.11 Task 9 unary-only guard was conservative; INCREMENTAL stays unary-only by design (single-position activations don't benefit from chunking — splitting <hidden_dim>×4 bytes into two chunks adds wire overhead without overlapping any compute), but PREFILL with long prompts now overlaps Stage 1's chunk emission with Stage 2's chunk consumption. Real-distilgpt2 E2E with ~30+ token prompt at 10 KiB chunk threshold proves the streamed transport actually fires (Stage 1 → Stage 2 hidden-state handoff at distilgpt2's 768 hidden_dim × 4 bytes/fp32 × 30+ positions ≈ 92+ KiB > threshold), and the bit-identical greedy invariant carries through the chunked path.

Round-1 review surfaced one LOW finding (defensive ``decode_mode == PREFILL`` assert at ``_dispatch_streamed_sharded`` entry, preventing Phase 3.x.11 Task 9 M1-class seam-bugs from a future refactor). Closed pre-tag.

### 2.2 Speculative decoding — compute-level pipelining (Phase 3.x.11.y)

This is the load-bearing slice of the sprint. It closes the compute-level pipelining honest-scope deferral that sprint-3 explicitly carried forward.

**The mechanism.** A small draft model proposes K=4 candidate tokens for the next round; the sharded chain verifies them in a single batched K+1-position forward; the executor accepts the longest matching prefix and broadcasts ``RollbackCacheRequest`` for the rejected suffix. Under perfect-accept (matched draft + verifier under greedy), each chain pass produces 5 tokens instead of 1. Network latency per token drops by the same factor.

**The architecture.** Five new components, each rigorously tested:

1. ``DraftModel`` Protocol + reference ``HFDraftModel`` impl — reset/propose/commit/evict lifecycle; v1 stateless re KV cache (re-runs ``model.generate()`` per propose for clarity; v2 stateful ``DynamicCache`` impl is a drop-in Protocol replacement).

2. VERIFY wire format extension — ``DecodeMode.VERIFY`` enum + ``RunLayerSliceResponse.verified_token_ids`` (capped at MAX_VERIFY_BATCH_TOKENS=65) + ``accepted_count`` (co-set invariant) + ``RollbackCacheRequest/Response`` envelopes. **Critical security extension**: the response's signing payload commits to ``verified_token_ids`` + ``accepted_count`` (mirrors Phase 3.x.11 Task 5's same-class fix for ``next_token_id`` + ``is_terminal``). Without this commitment, a downstream relay between the tail and the executor could swap verified tokens, causing the executor to emit wrong content. Pre-3.x.11.y signed bytes are byte-equivalent (omit-when-default canonical encoding).

3. ``ShardedAutoregressiveRunner`` VERIFY support — three new optional ``ShardedLayerForward`` Protocol methods (``forward_verify``, ``apply_lm_head_and_sample_batch``, ``truncate_cache``) implementing the K+1 batched forward + per-position sampling + cache rollback. Tail computes ``accepted_count`` via the standard speculative-decoding longest-prefix-match algorithm.

4. ``RpcChainExecutor`` speculation loop — ``draft_model`` + ``speculation_depth`` constructor args; ``_execute_chain_streaming_sharded_speculative`` branches when the draft is wired. **Greedy-only invariant at executor entry**: ``request.temperature > 0`` raises ``PROMPT_ENCODE_ERROR`` (correctness gate AND threat-model-containment gate). Mid-emit max_tokens truncation handles the case where ``accepted_count + 1`` would overshoot the cap. Eviction broadcast on every exit path (terminal, GeneratorExit, exception) calls ``draft.evict`` + ``manager.evict`` + ``RollbackCacheRequest`` broadcast.

5. Server-side ``RollbackCacheRequest`` handler — routes to ``ShardedAutoregressiveRunner.rollback_cache`` (which provides the model's ``truncate_cache`` as the manager's ``truncate_fn``). ``MissingVerifyCapabilityError`` mapped to ``MALFORMED_REQUEST`` so the executor can distinguish caller bug from internal crash.

**Real-model proof.** A 6-task slice of E2E with HuggingFace distilgpt2: split layers 0-2 / 3-5 across two chain stages, wire ``HFDraftModel`` against the SAME distilgpt2 (perfect-accept oracle under greedy). 8-token speculative decode with K=4 produces output **bit-identical to single-host greedy** — the load-bearing correctness proof.

**Critical adapter remediation during E2E bring-up.** The K+1 batched forward in HF GPT2's eager attention path defaults to FULL attention across new tokens (the auto-mask logic only kicks in for q_len=1 with cached past). Without an explicit 4D additive causal mask, each new query attends to all K+1 keys (including future positions), breaking greedy-equivalence vs single-token INCREMENTAL. Caught during E2E bring-up and remediated with an explicit causal mask in the test adapter. **Operators wiring other HF model adapters (Llama, Mistral, etc.) need to verify their attention impl handles K+1 batched cached forward correctly under their attention path** — this is documented in the adapter docstring and audit-prep §7.11.

### 2.3 Round-1 HIGH-1 finding — non-tail rollback silently no-op'd

Independent code review at Phase 3.x.11.y Task 9 surfaced one HIGH-severity correctness bug that would have silently corrupted non-tail KV-cache state the first time a real deployment with mixed-model speculation rejected a draft.

**The bug.** ``KVCacheManager.rollback`` clamped on ``handle.tokens_generated`` (bumped only by the tail-variant runner's ``_sample_tail`` / ``_sample_tail_verify``). Non-tail stages keep ``tokens_generated == 0`` for the entire request lifecycle. ``min(N, 0) == 0`` returned ``(False, 0)`` without invoking the truncate_fn callback. The non-tail model's KV cache retained the K+1 verified positions (including the rejected suffix), and the next VERIFY/INCREMENTAL forward computed wrong logits.

**Why the test suite missed it.** Three independent invisibility patterns:
1. The pre-tag E2E used ``draft == verifier`` (matched models under greedy = perfect-accept = rollback never fires in the green path).
2. ``test_rollback_cache.py`` pre-seeded ``tokens_generated`` directly, mimicking tail-shaped state.
3. ``test_chain_rpc_client_speculative.py`` used record-only rollback fakes that don't model manager state.

**Remediation pre-tag.** Added ``KVCacheHandle.cached_positions`` (separate from tail-only ``tokens_generated``); runner bumps it on EVERY successful forward (PREFILL by ``_input_n_positions``, INCREMENTAL by 1, VERIFY by K+1); manager clamps on ``cached_positions``. New regression test ``TestSpeculativeE2EPartialAcceptRollback::test_zero_accept_rolls_back_non_tail_cache`` uses a deliberate-mismatch draft proposing all-zeros to force ``accepted_count == 0`` every round; greedy-equivalence still holds against single-host (proves rollback didn't corrupt either stage's cache state). Adapter fix: HF ``DynamicCache.get_seq_length()`` defaults to layer 0 (empty on stages whose ``layer_range[0] > 0``) — would crop to 0 and blow away the entire cache; adapter now iterates layers to find any populated one.

The HIGH-1 + 3 MEDIUM findings (symmetric VERIFY ⇔ proposed_token_ids co-set; cap_hit_mid_emit at exact-cap; signing_payload silent None coercion) all closed pre-tag.

---

## 3. Threat-model addendum §3.5 (NEW)

Speculation introduces a NEW timing surface that the existing constant-time padding decorators don't cover: **per-iteration acceptance count is observable on the wire** (directly via ``RunLayerSliceResponse.accepted_count`` on tail responses, indirectly via ``RollbackCacheRequest.n_positions_to_drop`` per-round broadcast = K - accepted_count). A network observer learns the acceptance-rate distribution, which correlates weakly with prompt content (high-accept-rate prompts are typical natural-language continuations; low-accept-rate prompts are ambiguous or out-of-distribution).

**v1 mitigations** (carry forward from the addendum's existing patterns):
- **Tier C structural deny.** The Tier C dispatch gate from Phase 3.x.10.y carries forward unchanged: speculation runs entirely on Tier A/B paths. The deny gate fires at ``_dispatch_sharded`` BEFORE any VERIFY decoding, so the new accept-rate timing surface is structurally never exposed on Tier C.
- **Greedy-only invariant.** The executor's ``PROMPT_ENCODE_ERROR`` gate on ``request.temperature > 0`` is BOTH a correctness gate (no Leviathan-2023 sampling correction yet) AND a threat-model-containment gate (greedy speculation produces output bit-identical to non-speculative greedy, keeping the threat-surface comparison crisp).
- **Operator advisory.** Tier A + B content's accept-rate surface is honest-scope; operators choosing speculation for Tier A/B accept the trade in exchange for the 4-6× perf win.

**Honest-scope deferrals** (carry forward to roadmap):
- **Phase 3.x.11.y.x** — sampling-correct speculation under temperature > 0 (Leviathan-2023 rejection-sampling correction) + adaptive K tuning + authenticated rollback envelope (if telemetry warrants).
- **Phase 3.x.11.q.y bundle** — constant-time speculation for Tier C (combines 3.x.11.q's sharded constant-time decorators with speculation's accept-rate masking).
- **Phase 3.x.11.y'** (apostrophe) — multi-draft consensus.
- **Phase 3.x.11.y''** — cross-request draft caching.
- **Phase 3.x.11.z** — cache swap-out / paging.
- **Phase 3.x.12** — mid-stream re-routing.

---

## 4. Audit-prep refresh

`docs/2026-04-27-cumulative-audit-prep.md` extended from 9 streaming-inference phases (§7.1-§7.9) to 11 (§7.10 + §7.11 added). Each new section covers the 6-7 trust seams + auditor focus + honest-scope-deferred items + test coverage at tag. Auditor-bundle coordinator (`docs/2026-04-21-audit-bundle-coordinator.md`) refreshed to reference the new cumulative tag (`cumulative-audit-prep-20260429-b`).

All round-1 review HIGH/MEDIUM findings across both new phases are **RESOLVED pre-audit**.

---

## 5. By the numbers

| Metric | Sprint-3 baseline | Sprint-4 delta | Sprint-4 totals |
|---|---|---|---|
| Merge-ready tags (streaming-inference) | 10 (3.x.6 → 3.x.11.x) | +1 (3.x.11.y) | 11 |
| Cumulative audit-prep tag | `-20260430` | refreshed | `-20260429-b` |
| Streaming-inference unit tests green | 685 | +533 | 1218 |
| Slow real-distilgpt2 E2E tests green | 5+ | +6 | 11+ |
| Round-1 HIGH findings closed pre-tag | 4 | +1 | 5 |
| Round-1 MEDIUM findings closed pre-tag | 8 | +3 | 11 |
| Threat-model addendum sections | 4 (§3.1 - §3.4) | +1 (§3.5) | 5 |
| Audit-prep §7 sections | 10 (§7.1 - §7.10) | +1 (§7.11) | 11 |

---

## 6. What this unlocks

**Performance.** A 3-stage WAN chain at 50 ms RTT serving 70B-class models was capped at ~6-7 tokens/sec under sprint-3's per-token chain redispatch model. Under matched-draft speculation at K=4, that ceiling rises to ~25-30 tokens/sec — bringing distributed sharded inference into the same throughput class as single-host LAN deployments. (Real-world workload-dependent; operators benchmark before tuning K.)

**Trust posture.** Per-iteration receipt attestation closes the last "what about a malicious mid-stream cache swap?" auditor question. Combined with the response signing payload commitments (next_token_id + is_terminal from sprint-3, verified_token_ids + accepted_count from this sprint), the receipt now cryptographically binds every per-iteration outcome — relays cannot swap any signed field without invalidating the signature.

**Audit posture.** All 11 streaming-inference phases now stack cleanly into one cumulative bundle. Engagement scope is unchanged (single auditor, one remediation cycle, one Base mainnet deploy ceremony); the additional surface is two new sections in the same audit-prep doc + one new threat-model addendum revision.

---

## 7. Next slice

**Phase 3.x.11.y.x — sampling-correct speculation under temperature > 0** (Leviathan-2023 rejection sampling). Closes the most-cited honest-scope deferral from Phase 3.x.11.y; design plan kicked off this sprint. Adaptive K tuning + authenticated rollback envelope likely bundle into the same slice.

The streaming-inference subsystem is otherwise on autonomous-cleanup mode — Phase 3.x.11.q (Tier C sharded), Phase 3.x.11.z (cache paging), Phase 3.x.12 (mid-stream re-routing) are all roadmap items rather than near-term sprint targets.

---

## 8. Pointers for technical readers

- **Speculative decoding mechanics** — `prsm/compute/chain_rpc/client.py:_execute_chain_streaming_sharded_speculative`. Greedy-only gate at entry; mid-emit max_tokens truncation; finally-block eviction on every exit path.
- **VERIFY runner** — `prsm/compute/inference/sharded_runner.py:_sample_tail_verify`. Standard speculative-decoding longest-prefix-match algorithm with ``handle.tokens_generated += len(emitted)`` (only emitted tokens count against max_tokens).
- **Per-iteration attestation envelope** — `prsm/compute/inference/multi_stage_attestation.py:IterationAttestation`. Discriminator at magic-prefix level (`PRSM-MI-ATT-V1` vs legacy `PRSM-MS-ATT-V1`) preserves non-sharded receipt byte-equivalence.
- **Cache rollback** — `prsm/compute/chain_rpc/kv_cache.py:KVCacheManager.rollback`. Caller-injected ``truncate_fn`` keeps manager payload-opaque; HIGH-1 remediation introduces ``cached_positions`` counter bumped on every forward.
- **Threat-model addendum §3.5** — `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` v1.1.
- **Audit-prep §7.10 + §7.11** — `docs/2026-04-27-cumulative-audit-prep.md`.

---

## 9. Sprint-4 commits at a glance

```
a8ffdca0 phase 3.x.11.y task 9: round-1 review remediations (HIGH-1 + M1+M2+M3)
32c1a850 phase 3.x.11.y task 8: threat-model addendum §3.5 + audit-prep §7.11
b6617b8b phase 3.x.11.y task 7: E2E with real distilgpt2 + HFDraftModel
8f8ca31e phase 3.x.11.y task 6: RollbackCacheRequest server wire handler
d26b2a9e phase 3.x.11.y task 5: executor speculation loop + wire/server VERIFY plumbing
1fde33a8 phase 3.x.11.y task 4: ShardedAutoregressiveRunner VERIFY support
260a92e2 phase 3.x.11.y task 3: DraftModel Protocol + HFDraftModel reference impl
328d4e53 phase 3.x.11.y task 2: KVCacheManager.rollback
505222cf phase 3.x.11.y task 1: VERIFY wire-format extension + RollbackCacheRequest
18050a54 docs: Phase 3.x.11.y design plan — speculative decoding
```

Plus the audit-bundle coordinator refresh + cumulative tag (`cumulative-audit-prep-20260429-b`) at sprint close.

---

*PRSM continues to ship distributed-inference infrastructure on a measured cadence: each merge-ready tag is independently reviewed, threat-modeled, and tested against bit-identical real-model invariants. The next sprint slice (3.x.11.y.x) closes the largest remaining honest-scope deferral from the streaming-inference track.*
