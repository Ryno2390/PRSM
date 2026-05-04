# PRSM Mainnet Audit — Bundle Coordinator

**Date:** 2026-05-04 (refreshed for cumulative tag — adds §7.16 "CEREMONY EXECUTED 2026-05-04" subsection documenting Phase 1.3 Task 8 live deployment on Base mainnet + operational lessons doc + Multi-Sig Action Plan addendum; tag moved forward at HEAD; **streaming-inference subsystem roadmap-cap reached + Phase 1.3 Task 8 mainnet deploy ceremony EXECUTED + Foundation Safe `0x91b0...5791` (2-of-3) + ProvenanceRegistry + RoyaltyDistributor live and Basescan-verified on Base**)
**Bundle tag reference:** `cumulative-audit-prep-20260504-h` — now stacks all fifteen streaming-inference phases (§7.1-§7.15) PLUS §7.16 covering the Phase 1.3 Task 8 + post-audit deploy-ceremony infrastructure (mainnet-hardened deploy-provenance.js + 4-script T-0 pipeline + 2 transfer scripts + 2 rehearsal orchestrators + audit gap closures). §7.16 is a different axis from §7.1-§7.15: it covers operator-facing scripts and runbooks that wrap the on-chain contracts, rather than the contracts themselves. 13-commit deploy-prep sprint window 34b59c11..e4c52144.
**Engagement model:** single auditor, one remediation cycle, one Base mainnet deploy ceremony.

This is the first document an external auditor should read. It frames the full audit tree as one coherent engagement, points to the per-phase scope bundles that drill into each surface, and surfaces the cross-phase seams that an auditor should focus on first.

> **📌 For auditors starting here:** the most recent audit-prep bundle is **`docs/2026-04-27-cumulative-audit-prep.md`** at tag `cumulative-audit-prep-20260504-h` — it now stacks the original 2026-04-22 economic-layer baseline + 2026-04-27 multi-phase additions + the fifteen streaming-inference phases (§7.1-§7.15, with §7.14.1 covering the q.y' delta and §7.15 covering the q.x roadmap-cap closure) + §7.16 covering the Phase 1.3 Task 8 + post-audit deploy-ceremony infrastructure. All round-1 review HIGH/MEDIUM findings across all phases are **RESOLVED pre-audit**. The 2026-04-22 baseline (`docs/2026-04-22-phase7.1x-audit-prep.md`) remains authoritative for the 7/7.1/7.1x economic substrate; this refresh ADDS to it rather than replacing. **New for this refresh:** §7.16 deploy-ceremony infrastructure — operator-facing scripts and runbooks that wrap the on-chain contracts. Two ceremonies covered: Phase 1.3 Task 8 (immediate, post-hardware-arrival 2026-05-01) and post-audit (audit-bundle + Phase 8 + Phase 7-storage; gated on Phase 7 Task 9 + Phase 7.1 Task 9). Three new mainnet-only safety guards on `deploy-provenance.js` (chainId pin + canonical-FTNS pin + treasury-is-contract); 4-script T-0 pipeline (pre-task8-checklist → deploy-provenance → verify-provenance-deployment → post-task8-handoff-checklist) all proven green on hardhat-local; mainnet-fork dry-run safety verified against real Base mainnet RPC without burning gas. Engineering audit of operator-side Multi-Sig Action Plan (`docs/2026-04-30-multisig-action-plan-engineering-audit.md`, 10 findings) with executable F1+F3+F5+F9 closures. Stale-script purge (`deploy.js` + `verify-deployment.js` deleted; `package.json` + `README.md` + `FTNS_TESTNET_DEPLOYMENT.md` reconciled). 7 trust seams called out in §7.16 for auditor focus including disposable-deployer-key fund-and-sweep pattern, two-phase deploy model rationale, and `MINTER_ROLE → EmissionController` post-handoff governance tx.

---

## 1. What's being audited

Three phases ship the core PRSM economic layer and must be reviewed together because their challenge / slash paths are tightly coupled:

| Phase | Tier | Merge-ready tag | Surface |
|-------|------|-----------------|---------|
| **Phase 3.1** | A (receipt-only) | `phase3.1-merge-ready-20260421` | `BatchSettlementRegistry` + `EscrowPool` + challenge handlers (DOUBLE_SPEND / INVALID_SIGNATURE / NO_ESCROW / EXPIRED) |
| **Phase 7** | C (stake + slash) | `phase7-merge-ready-20260421` | `StakeBond.sol` (new) + registry slash-hook extension + Python StakeManager + marketplace on-chain tier gate |
| **Phase 7.1** | B (redundant execution) | `phase7.1-merge-ready-20260421` | `CONSENSUS_MISMATCH` reason code + cross-batch handler + Python MultiDispatcher / ShardConsensus + orchestrator consensus routing |
| **Phase 3.x.6** | A | `phase3.x.6-merge-ready-20260427` | Parallax inference scheduler (vendored GradientHQ/parallax) + 4 PRSM-original trust adapters + ParallaxScheduledExecutor |
| **Phase 3.x.7** | A | `phase3.x.7-merge-ready-20260428` | Cross-host ChainExecutor: RpcChainExecutor + LayerStageServer + multi-stage TEE attestation envelope |
| **Phase 3.x.7.1** | A | `phase3.x.7.1-merge-ready-20260428` | Chunked activation streaming (v2 wire format adds ActivationChunk + manifest field) |
| **Phase 3.x.8** | A | `phase3.x.8-merge-ready-20260428` | Streaming-token output: TokenFrame + StreamFinalFrame wire format + StreamingLayerRunner Protocol + receipt streamed_output flag |
| **Phase 3.x.8.1** | A | `phase3.x.8.1-merge-ready-20260428` | SSE-framed POST /compute/inference/stream endpoint + design plan §3.4 settle-on-tokens-emitted billing policy |
| **Phase 3.x.10** | A | `phase3.x.10-merge-ready-20260428` | AutoregressiveStreamingRunner (real HF generate; replaces SyntheticStreamingRunner placeholder) + per-token timing side-channel memo |
| **Phase 3.x.10.x** | A | `phase3.x.10.x-merge-ready-20260428` | Production wiring: max_tokens + temperature wire fields + StreamingSamplingShim + make_autoregressive_streaming_runner factory |
| **Phase 3.x.10.y** | A | `phase3.x.10.y-merge-ready-20260429` | Tier C constant-time padding decorators (M2 BatchedTrailing + M1 FixedRate at cadence) + Tier C dispatch-layer gate (default-deny) + HF prompt-echo fix |
| **Phase 3.x.11** | A/B (Tier C structurally denied at v1) | `phase3.x.11-merge-ready-20260430` | Sharded autoregressive decode: per-token chain redispatch with KV-cache handoff. ShardedAutoregressiveRunner + KVCacheManager + executor per-token chain loop + EvictCacheRequest wire envelope + RunLayerSliceResponse signing-payload extension to commit next_token_id + is_terminal. Bit-identical real-distilgpt2 sharded vs single-host greedy. |
| **Phase 3.x.11.x** | A/B | `phase3.x.11.x-merge-ready-20260430` | Pipelining (wire-level chunked PREFILL composition) + per-iteration receipt attestation. IterationAttestation under separate PRSM-MI-ATT-V1 magic prefix (golden-bytes pin on legacy multi-stage envelope = non-sharded receipts byte-equivalent). Closes 3.x.11 threat-model addendum §3.2 "no per-iteration cryptographic commitment" gap. |
| **Phase 3.x.11.y** | A/B (Tier C still structurally denied; greedy-only at v1) | `phase3.x.11.y-merge-ready-20260429` | Speculative decoding (compute-level pipelining): K=4 drafts proposed by HFDraftModel + batched K+1 verify + accept-longest-prefix + RollbackCacheRequest broadcast. ShardedAutoregressiveRunner forward_verify/apply_lm_head_and_sample_batch/truncate_cache. Bit-identical real-distilgpt2 greedy proven. Greedy-only invariant at executor entry (PROMPT_ENCODE_ERROR on temp > 0). Round-1 HIGH-1 (non-tail rollback no-op'd; KVCacheHandle.cached_positions counter remediation) + M1+M2+M3 closed pre-tag. |
| **Phase 3.x.11.y.x** | A/B (Tier C still structurally denied; T > 0 now supported) | `phase3.x.11.y.x-merge-ready-20260429` | Sampling-correct speculation under T > 0 via Leviathan-2023 §2.2 rejection sampling under Option C.1. New `proposed_token_probs` wire field (co-set with `proposed_token_ids` + signing-payload commitment + omit-when-None canonical encoding). DraftModel.propose_with_probs Protocol + HFDraftModel reference. `apply_lm_head_and_sample_batch_with_rejection` Protocol + pure-NumPy `rejection_sample_speculation` helper. Executor v1↔v2 routing on `request.temperature` + adaptive K (rolling 4-round window, halve <25% / double >75%, v2-only — preserves v1 bit-identical regression). Server backwards-compat (TypeError → MALFORMED_REQUEST on stale runner; no silent fallback). Critical correctness fix: rollback math `(k_round + 1) - len(emitted)` (was `len(verified) - len(emitted)` which under-counts in v2 partial-accept). Three-layered convergence proof: K=1 q=1.0 + K=4 q=1.0 (both atol 0.025 vs target) + stochastic-q drift pin against analytical-C.1-marginal. Round-1 M1+M2+L3 closed pre-tag (M1 caught + corrected a load-bearing CLAIM defect — Option C.1 marginal is exact only in q=1.0 regime); L1+L2 deferred. |
| **Phase 3.x.11.q** | A/B/**C** (chain-level masking; per-stage wire still leaks — 3.x.11.q.x) | `phase3.x.11.q-merge-ready-20260429` | Tier C constant-time sharded decode via chain-level decorators. `BatchedTrailingShardedExecutor` (M2) emits 1 StreamToken (joined text) + 1 ChainExecutionResult regardless of inner cadence; `FixedRateShardedExecutor` (M1) clamps inter-StreamToken intervals to ≥ cadence (injectable clock+sleep for test determinism). `make_tier_c_sharded_executor(inner, *, mode, cadence_seconds)` factory mirrors Phase 3.x.10.y mode-string pattern. `ParallaxScheduledExecutor` constructor gains `tier_c_chain_executor: Optional[Any]`; `execute_streaming` routes by `request.content_tier` with no-silent-fallback invariant (Tier C without decorator → structured `InferenceResult.failure` naming `make_tier_c_sharded_executor`). Per-stage `_dispatch_sharded` TIER_GATE deny stays as defense-in-depth. Speculation + Tier C remains structurally denied (Phase 3.x.11.q.y bundles speculation-aware constant-time). Round-1 M1 (post-terminal token drop) + M2 (non-str text_delta defensive coerce) closed pre-tag; 3 LOWs deferred. 82 unit + E2E tests. |
| **Phase 3.x.11.q.x** | A/B/**C** (closes the named per-stage timing + M2 response-size honest-scope items from §7.13; **streaming-inference subsystem roadmap-cap reached**) | `phase3.x.11.q.x-merge-ready-20260430` | Closes the two named honest-scope items carrying since §7.13: (1) per-stage wire timing leak under sharded autoregressive decode via `RpcChainExecutor.per_stage_dispatch_cadence_seconds=...` (clamps inter-iteration cadence in BOTH non-speculative + speculative sharded loops via new `_wait_for_per_stage_cadence` helper; each per-stage RPC arrives at uniform inter-arrival cadence regardless of K and decode work). (2) M2 response-size leak via `BatchedTrailingShardedExecutor.pad_to_bytes=...` (UTF-8-safe truncation via `decode(errors="ignore")` + whitespace fill to fixed byte target; `finish_reason="length_capped"` when joined exceeds cap). Factory threads `pad_to_bytes` for `mode="m2"` and rejects for `mode="m1"`. Composition: per-stage cadence + chain-level decorators wired together = full-network constant-time masking. 0 round-1 remediations required (self-code-review found no HIGH/MEDIUM issues). 16 new tests across TestPerStageDispatchCadence (4) + TestQXCompositionCadencePlusPadding (1) + TestBatchedTrailingPadding (11). Threat-model §3.7 + §3.8 (1.6 revision) + audit-prep §7.15. After q.x, only Phase 3.x.11.q.y'' (multi-stage replay forward, conditional on telemetry) remains as a follow-up. |
| **Phase 3.x.11.q.y'** | A/B/**C** (closes v1 honest-scope residuals from q.y; multi-stage replay best-effort honest-scope carries forward) | `phase3.x.11.q.y-prime-merge-ready-20260430` | Closes both v1 residuals from the q.y baseline: (1) `RollbackCacheRequest.n_positions_to_drop` drop-value leak via `RpcChainExecutor.always_rollback_k=True` + new `replay_accepted_prefix` / `encrypted_replay_accepted_prefix` / `target_stage_index` wire fields with mutual-exclusion + co-set + AAD-distinct (`b"|rollback"` suffix) validators; server decrypts at the boundary + drives `runner.replay_accepted_prefix` → forward over the prefix tokens to repopulate the cache (stage-0-only at v1; non-stage-0 best-effort honest-scope). (2) Operator-managed PSK distribution via new `X25519AnchoredCipher` class (drop-in alternative to ProbsCipher; surface-compatible) + `HandoffToken.ephemeral_pubkey` field (signed via the existing settler signature; relay substitution breaks `verify_with_anchor`); per-request ECDH + HKDF over `(request_id, stage_index, chain_total_stages)` → forward secrecy across requests + chain-length forgery defense. E2E pin transitions from `test_e3_residual_rollback_leak_is_documented` (q.y baseline; asserts leak presence) to `TestAlwaysRollbackKE2E::test_e3_constant_k_rollback_pin` (q.y' opt-in; asserts `n_positions_to_drop == K + 1` regardless of acceptance). New `TestX25519PerRequestKeyRotation::test_e5_x25519_per_request_key_rotation` proves substituted ephemeral_pubkey across requests fails decrypt. Round-1 L1 (server decrypt failure → best-effort, matches runner replay semantics) closed pre-tag. 592 unit + 7 slow E2E. Honest-scope residuals carry forward: multi-stage replay best-effort, replay window inside `deadline_unix`, post-quantum (R6 trigger-watch), multi-position §2.2 marginal narrowing under constant-K, per-stage timing leak (Phase 3.x.11.q.x), adaptive K under flat-K is OFF. |
| **Phase 3.x.11.q.y** | A/B/**C** (constant-time speculation under Tier C; rollback drop-value channel still leaks — 3.x.11.q.y') | `phase3.x.11.q.y-merge-ready-20260430` | Constant-time speculation under Tier C composes 3.x.11.q chain-level decorators with three new per-stage-wire mitigations. `encrypted_proposed_token_probs` wire field on `RunLayerSliceRequest` (Optional[bytes], 1024-byte cap, mutual-exclusion with plaintext probs, co-set with proposed_token_ids); `ProbsCipher` AES-256-GCM helper with AAD-bound `(request_id, stage_index)` + length-mismatch defense + HKDF-SHA256 key derivation. Tail-runner `constant_k_commitment=True` pads `verified_token_ids` to K+1 regardless of acceptance via `apply_lm_head_and_sample_batch` argmax fillers (multi-position §2.2 marginal narrows under constant-K; user-facing output stays §2.2-correct via `accepted_count`). Client-side `flat_k_mode=True` gates off Phase 3.x.11.y.x adaptive-K state machine (cross-round correlation closed at cost of perf win). `ParallaxScheduledExecutor` constructor gains `tier_c_speculation_enabled: bool = False` — `execute_streaming` denies Tier C + temperature > 0 by default; structured failure names full opt-in contract (cipher + flat_k_mode + tail constant_k_commitment). Server-side `LayerStageServer.encrypted_probs_cipher=` decrypts at AAD-bound boundary; unwired-cipher / decrypt failure surfaces `MALFORMED_REQUEST` (no silent fallback to plaintext probs). Honest-scope residual leak: `RollbackCacheRequest.n_positions_to_drop` still encodes `K - accepted_count` on the wire (E2E `test_e3_residual_rollback_leak_is_documented` ASSERTS leak's wire presence). Round-1 L1 (ProbsCipher.encrypt nonce idiom: `os.urandom(12)` not `AESGCM.generate_key(bit_length=128)[:12]`) closed pre-tag. 602 unit + 4 slow E2E tests. |

**Why bundled:**
- Phase 7's `stakeBond.slash(...)` call sits inside Phase 3.1's `challengeReceipt` flow.
- Phase 7.1 adds one more `ReasonCode` branch to that same `challengeReceipt` dispatch chain.
- All three phases share the same `ReceiptLeaf` struct, the same batched-settlement Merkle-tree machinery, and the same slash economics (70/30 challenger/Foundation split with self-slash 100%-to-Foundation).

Splitting the audits would risk seam-crossing findings falling between engagements. One auditor, one remediation cycle, one deploy.

---

## 2. Per-phase scope bundles (read in order)

1. **`docs/2026-04-27-cumulative-audit-prep.md`** — most recent; cumulative bundle covering all post-economic-layer phases. Read sections in order:
   - §2.1-§2.7 — Solidity + Python + JS SDK additions (3.x.2 / 3.x.3 / 3.x.4 / 3.x.5 + Phase 4 Task 3).
   - §7.1 — Parallax third-party-derived components (Phase 3.x.6).
   - §7.2 — Cross-Host ChainExecutor (Phase 3.x.7).
   - §7.3 — Chunked Activation Streaming (Phase 3.x.7.1).
   - §7.4 — Streaming-Token Output (Phase 3.x.8).
   - §7.5 — Streaming HTTP Endpoint (Phase 3.x.8.1).
   - §7.6 — Real Autoregressive Streaming Runner (Phase 3.x.10).
   - §7.7 — Production Wiring + Sampling-Param Plumbing (Phase 3.x.10.x).
   - §7.8 — Tier C Constant-Time Padding + HF Prompt-Echo Fix (Phase 3.x.10.y).
   - §7.9 — Sharded Autoregressive Decode (Phase 3.x.11). Read alongside `docs/2026-04-30-phase3.x.11-threat-model-addendum.md` for the per-token timing surface + KV-cache privacy threat model.
   - §7.10 — Pipelining + Per-Token Receipt Attestation (Phase 3.x.11.x). Closes the threat-model addendum §3.2 no-per-iteration-cryptographic-commitment gap.
   - §7.11 — Speculative Decoding (Phase 3.x.11.y). Read alongside the threat-model addendum's §3.5 (added in 1.1 — covers the per-iteration accept-rate timing surface) and the addendum's auditor-reading-path entries 8 + 9 (greedy-only gate + best-effort rollback).
   - §7.12 — Sampling-Correct Speculation under Temperature > 0 (Phase 3.x.11.y.x). Read alongside the threat-model addendum's §3.6 (added in 1.2 — covers the three new content-correlated wire surfaces: accept-rate channel narrows-but-doesn't-disappear, `proposed_token_probs` ships K floats per VERIFY round, adaptive K cross-round correlation). The §7.12 honest-scope section explicitly documents that Option C.1's marginal-equals-target invariant holds EXACTLY only in the degenerate-q regime (q=1.0); the stochastic-q drift is pinned numerically in `test_option_c1_drift_under_stochastic_q_documented` so a future helper change that broke C.1 determinism would surface in CI.
   - §7.13 — Tier C Constant-Time Sharded Decode (Phase 3.x.11.q). Read alongside the threat-model addendum's §3.7 (added in 1.3 — covers the chain-level vs per-stage scope-honesty point). The §7.13 trust-seam list flags the routing-layer no-silent-fallback invariant as load-bearing — confirm `tests/unit/test_parallax_executor.py::TestTierCRoutingIntegration::test_tier_c_without_decorator_surfaces_failure` asserts BOTH the structured failure AND that the default chain_executor is NOT touched (`primary.streaming_calls == []`).
   - §7.14 — Constant-Time Speculation under Tier C (Phase 3.x.11.q.y). **NEW. Read alongside the threat-model addendum's §3.8 (added in 1.4 — covers the v1 wire-surface analysis + the load-bearing residual leak: RollbackCacheRequest.n_positions_to_drop still encodes `K - accepted_count` on the wire). The §7.14 trust-seam list flags six items including AAD binding closes cross-slot replay, mutual-exclusion at validator, MALFORMED_REQUEST on unwired cipher (no silent downgrade to plaintext probs), routing-layer `tier_c_speculation_enabled` no-silent-route, multi-position §2.2 marginal narrowing under constant-K (user-facing output unchanged via `accepted_count`), residual rollback drop-value leak (operator-visible v1 trade). The E2E test `test_e3_residual_rollback_leak_is_documented` is the in-code documentation primary — auditors should confirm the leak's wire presence is asserted (so accidental quiet closure is caught as a regression requiring §3.8 update).**
2. **`docs/2026-04-21-phase7.1-audit-prep.md`** — covers the CONSENSUS_MISMATCH extension.
3. **`docs/2026-04-21-phase7-audit-prep.md`** — covers StakeBond + slash hook into Phase 3.1.
4. **Phase 3.1 surface** — covered by the Phase 7 bundle's §1.2 "out of scope" (Phase 3.1 was already merge-ready and forms the unchanged substrate Phase 7 and 7.1 build on).

Each per-phase bundle contains: in/out-of-scope, commit range, threat model, known issues with auditor prompts, engagement plan.

---

## 3. Cross-phase seams (auditor priority zones)

### 3.1 `challengeReceipt` dispatch chain

**Location:** `contracts/contracts/BatchSettlementRegistry.sol` lines ~369-445.

```
Phase 3.1: dispatch to 4 handlers + slash skipped
Phase 7:   extend slash-hook allowlist: DOUBLE_SPEND + INVALID_SIGNATURE
Phase 7.1: append CONSENSUS_MISMATCH to enum (value=5), add cross-batch
           handler, extend slash-hook allowlist
```

An auditor should read this function end-to-end once and verify:
- Enum ordering is stable across the three phases (no re-ordering).
- The slash-hook's reason allowlist exactly matches the set of handlers that represent provable misbehavior (not griefing, not hygiene).
- The `try/catch` around `stakeBond.slash` behaves correctly under every handler's return path.

### 3.2 `ReceiptLeaf` canonical form

The struct is defined in Phase 3.1 and consumed by all three phases:
- Phase 3.1's Merkle tree commits leaves.
- Phase 7's slash hook passes the batchId (not the leaf) into StakeBond; no leaf manipulation.
- Phase 7.1's handler decodes a second `ReceiptLeaf` from auxData and verifies its hash matches a proof against a different batch's root.

Please confirm `abi.encode(ReceiptLeaf)` encoding is deterministic across calldata and memory locations, and that the Python-side `_leaf_hash` helper in `tests/integration/test_phase7_1_consensus_e2e.py` produces byte-identical output.

### 3.3 Slash economics under three reason codes

`StakeBond.slash(provider, challenger, reasonId)` is unchanged since Phase 7 Task 2. Three reason codes now route through it:

| Reason | Slash target | Challenger | Self-slash guard |
|--------|--------------|------------|------------------|
| DOUBLE_SPEND | `b.provider` | caller of `challengeReceipt` | triggers if `caller == b.provider` → 100% Foundation |
| INVALID_SIGNATURE | `b.provider` | caller of `challengeReceipt` | triggers if `caller == b.provider` → 100% Foundation |
| CONSENSUS_MISMATCH | `b.provider` (= minority's committer) | `b.requester` (MVP auth) | triggers if `b.requester == b.provider` → 100% Foundation |

The sybil-requester griefing vector (Phase 7.1 audit-prep §5.2) bypasses the self-slash guard because the attacker uses TWO distinct EOAs. The design argument is that the 30% Foundation skim makes it negative-EV; please verify the math holds across realistic gas + stake + FTNS-price parameter ranges per PRSM-ECON-WP-1.

### 3.4 `tier_slash_rate_bps` snapshot invariant

Phase 7 introduced a two-field snapshot chain:
1. `Stake.tier_slash_rate_bps` — snapshotted at `bond()` time into the provider's stake record.
2. `Batch.tier_slash_rate_bps` — snapshotted at `commitBatch()` time into each batch.

The slash amount is computed from **Stake.tier_slash_rate_bps**, not Batch.tier_slash_rate_bps. The batch field is used only as the slash-hook's "is this batch slashable at all?" predicate (`> 0`). Phase 7.1 preserves this exactly — all three reason codes that slash (DOUBLE_SPEND / INVALID_SIGNATURE / CONSENSUS_MISMATCH) reach the same hook and the same `StakeBond.slash` that reads from `stakes[provider]`.

Please confirm a provider cannot dodge slash by:
- Re-bonding at a lower tier before a challenge lands (blocked by `AlreadyBonded` check in `bond()`).
- Unbonding mid-challenge (blocked by the UNBONDING status still being slashable per Phase 7 design §3.1).
- Withdrawing mid-challenge (blocked by unbond-delay timer + WITHDRAWN status not slashable, so race is provider-loses).

---

## 4. Consolidated test inventory

```bash
# Solidity (hardhat)
cd contracts && npx hardhat test \
  test/StakeBond.test.js \
  test/BatchSettlementRegistry.test.js \
  test/BatchSettlementChallenge.test.js \
  test/BatchSettlementSlashing.test.js \
  test/BatchSettlementConsensus.test.js \
  test/BatchSettlementGasFloor.test.js

# Python unit (excludes conftest-brittle legacy tests)
.venv/bin/python -m pytest \
  tests/contracts/ \
  tests/unit/test_reputation_tracker.py \
  tests/unit/test_marketplace_orchestrator.py \
  tests/unit/test_orchestrator_consensus.py \
  tests/unit/test_shard_consensus.py \
  tests/unit/test_multi_dispatcher.py \
  tests/unit/test_phase7_1_reason_code.py \
  tests/unit/test_phase7_1_reputation_observability.py

# Python E2E (live hardhat)
.venv/bin/python -m pytest \
  tests/integration/test_phase7_stake_slash_e2e.py \
  tests/integration/test_phase7_1_consensus_e2e.py -v
```

Expected at the current tip: **427 passing total** (142 Solidity + 283 Python unit + 2 Python E2E).

---

## 5. Consolidated known-issues list (across all three bundles)

From the three per-phase review gates. All are non-blockers for tag but tracked through the audit:

### From Phase 7 (Task 8 review)

- **§8.7 — Challenge-tx gas floor.** ✅ RESOLVED PRE-AUDIT. Landed contract-level `MIN_SLASH_GAS = 150_000` with `require(gasleft() >= MIN_SLASH_GAS)` before the `stakeBond.slash` try/catch. Preserves best-effort semantics for legitimate slash-ineligibility while excluding the OOG path. 7 new tests in `BatchSettlementGasFloor.test.js` pin the behavior.
- **§8.8 — Cross-process nonce-race on shared provider keys.** Not a regression from Phase 1.1; Phase 7 widens exposure. Operator runbook invariant now in `OPERATOR_GUIDE.md` §On-chain Keypairs.

### From Phase 7.1 (Task 8 review)

- **§8.6 — `consensus_minority_queue` persistence.** ✅ RESOLVED pre-audit via three shipped artifacts: (1) `ConsensusChallengeSubmitter` service (`prsm/marketplace/consensus_submitter.py`), (2) SQLite-backed `ConsensusChallengeQueue` (`prsm/marketplace/consensus_queue.py`) with PENDING → SUBMITTABLE → SUBMITTED/FAILED lifecycle + crash-safety test, (3) `process_submittable_queue` runner with outcome-based retry classification. Remaining follow-ups (exponential backoff, multi-process claim leases, `BatchCommitted` event watcher) are Phase 7.1x.next+ operational polish, not contract-security.
- **§8.7 — Sybil-requester griefing vector.** ✅ RESOLVED PRE-AUDIT via `consensus_group_id` binding in `Batch` struct + `_handleConsensusMismatch`. Both batches must share a non-zero `consensus_group_id` AND be committed by different providers, multiplying the attacker cost from `1×` stake burn to `k×`. As a bonus, the requester-only auth is replaced by open third-party challenger (closes §8.5 too). 3 new tests pin the behavior.
- **§8.8 — asyncio timeout/cancel propagation.** ✅ VERIFIED + FIXED PRE-AUDIT. Full grep pass completed; `asyncio.TimeoutError` is wrapped by `_dispatch_once`'s retry loop into `ShardDispatchError`; `asyncio.CancelledError` correctly propagates (outer-loop cancellation). One real gap found and fixed: `PeerNotConnectedError` now classified as partial-response in `MultiShardDispatcher` rather than aborting the gather.

---

## 6. Hardware-gated prerequisites

Shared across all three phases — these all fire once together, not per-phase:

- [ ] Foundation 2-of-3 multi-sig hardware wallets configured (Ledger + Trezor + Keystone per PRSM-GOV-1 §8).
- [ ] Treasury funded on Base mainnet with FTNS + USDC for auditor retainer.
- [ ] Auditor selected and engagement letter signed.

---

## 7. Deploy ceremony (post-audit)

Single ceremony covering all three phases. Follows the Phase 1.3 Task 8 multi-sig action plan (at `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/My Vault/Agent-Shared/Multi-Sig_Action_Plan.md`).

Key deploy steps in order:
1. Deploy FTNSToken (Phase 1.1 — already live on mainnet if Phase 1.3 Task 8 shipped; skip if so).
2. Deploy EscrowPool (Phase 3.1).
3. Deploy BatchSettlementRegistry (Phase 3.1 — includes Phase 7 + 7.1 extensions baked in; single deploy).
4. Deploy production Ed25519 signature verifier (replaces `MockSignatureVerifier`).
5. Deploy StakeBond (Phase 7).
6. Cross-wire: `setEscrowPool`, `setSignatureVerifier`, `setStakeBond`, `setSlasher`, `setFoundationReserveWallet`, `setSettlementRegistry`.
7. Verify all contracts on Basescan.
8. Freeze-tag the deployed-artifacts commit: `mainnet-20260XXX`.

---

## 8. Contact

- **Engineering lead:** Ryne Schultz (schultzryne@gmail.com).
- **Audit coordination:** TBD post-multi-sig setup.
- **Security disclosure:** security@prsm.ai.

For audit questions on specific design decisions, each per-phase bundle's §8 (Open issues) captures the rationale trail. For cross-phase questions, this coordinator is the integration map.
