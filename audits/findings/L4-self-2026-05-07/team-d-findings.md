# Team D — Lifecycle / State-Machine Re-Audit Findings (2026-05-07)

**Pinned commit:** `2c93c5bb` (HEAD of `main` after MED-5 + INFO-1..5 docstring/ordering cleanups)
**Re-audit authority:** PRSM-POL-2 §4.1 — re-audit of post-fix tip to confirm prior findings remain closed and no new HIGH/CRITICAL surfaces opened by the fixes.
**Prior round:** `audits/findings/L4-self-2026-05-06/team-d-findings.md` (1 HIGH, 2 MED, 3 lower).

## Summary

**Result: zero new HIGH or CRITICAL findings. Zero new MEDIUM findings. All prior Team-D findings (D-01 through D-06) are confirmed remediated.**

The lifecycle / state-machine surfaces re-evaluated:
- Batch lifecycle (NONEXISTENT → PENDING → FINALIZED / VOIDED)
- Stake lifecycle (none → BONDED → UNBONDING → WITHDRAWN)
- Per-batch snapshot composition (challengeWindow, lookbackWindow, totalPaused, escrowPool, stakeBond, signatureVerifier)
- Constructor cross-wire validation (StakeBond + EscrowPool zero-rejects)
- Setter cross-wire validation (BSR three setters, code.length guard with zero-permitted disable mode)
- Pause coordination (BSR + StakeBond independent OZ Pausable; documented operational invariant)
- `commitBatch` storage-pointer ordering (status flip is the LAST storage write)
- `_pause`/`_unpause` overrides safety
- `lastPendingBatchExpiry` lifecycle (monotonic, never decremented)

All checks pass. Test suite: 524 hardhat passing (no regression vs. 2026-05-06 post-fix tip); 16 / 16 L4 self-audit-specific tests in `test/L4SelfAuditFixes.test.js` passing.

## Re-check of prior findings

### D-01 (HIGH-1, ≡ A-01) — `requestUnbond` floor uses LIVE challengeWindow → **CONFIRMED CLOSED**

**Prior fix:** BSR maintains `mapping(address provider => uint64) public lastPendingBatchExpiry`, updated monotonically in `commitBatch` to `max(existing, commitTimestamp + challengeWindowSecondsAtCommit)`. `StakeBond.requestUnbond` reads it via the new `ISlasherWithProviderExpiry` interface and clamps `effectiveEligibleAt` against it.

**Verification:**
- `BatchSettlementRegistry.sol:282` declares the mapping.
- `BatchSettlementRegistry.sol:494-502` updates it inside `commitBatch` after the status flip; max-update only.
- `StakeBond.sol:39-41` defines `ISlasherWithProviderExpiry`; `:330-339` reads it inside `requestUnbond` with try/catch fallback to the local floor.
- The fix is **structural**, not parameter-tuned. A subsequent `setChallengeWindowSeconds` reduction cannot move the unbond floor below any PENDING batch's pinned expiry, because the floor is sourced from the per-batch snapshot via the per-provider tracker — not from the live mutable global.

**Lifecycle composition checks:**

1. *Tracker never decremented.* When a batch FINALIZES or VOIDS, the tracker remains at the high-water mark. This is intentional: the value naturally goes stale (past `block.timestamp`) and is dominated by `localFloor = now + unbondDelaySeconds` (always strictly in the future). Honest providers are NOT over-blocked. Verified by reading the StakeBond clamp logic at `:331-333`: `if (uint256(maxExpiry) > effectiveEligibleAt)` — past timestamps cannot exceed `now + 1 day`.
2. *Multiple PENDING batches.* Tracker holds the MAX expiry. When the oldest finalizes, the newer batch's expiry remains valid in the tracker (it was `max`'d in at its commit). When the newer also finalizes, both expiries are now stale-but-dominated by `localFloor`. ✓
3. *No external call between status flip and tracker update.* `commitBatch:492` sets `b.status = PENDING`; lines `:499-502` update the tracker. Both are storage writes; no external call between them. An external observer in the SAME transaction cannot see a half-updated state. ✓
4. *Per-provider isolation.* `lastPendingBatchExpiry[msg.sender]` is keyed by provider, so one provider's PENDING batches don't extend another provider's unbond floor. ✓

**Regression test:** `test/L4SelfAuditFixes.test.js` ("HIGH-1 (A-01 ≡ D-01) — lastPendingBatchExpiry tracker") — passing.

### D-02 (MED-6) — Constructor zero-reject for slasher / initialRegistry → **CONFIRMED CLOSED**

- `StakeBond.sol:224` — `if (initialSlasher == address(0)) revert ZeroAddress();`
- `EscrowPool.sol:106` — `if (initialRegistry == address(0)) revert ZeroAddress();`

Both contracts hard-reject `address(0)` for the immutable cross-wire. Permanent-brick-on-deploy primitive removed. Tests previously relying on `address(0)` deploys would fail loudly at construction. ✓

### D-03 (MED-7) — Setter `code.length > 0` + zero-permitted disable mode → **CONFIRMED CLOSED**

- `BatchSettlementRegistry.sol:929-931` — `setEscrowPool` rejects non-zero EOAs.
- `BatchSettlementRegistry.sol:945-947` — `setStakeBond` same.
- `BatchSettlementRegistry.sol:961-963` — `setSignatureVerifier` same.

All three setters preserve `address(0)` "disable" mode (intentional — slashing/INVALID_SIG/settlement can be disabled by the owner). Non-zero values must satisfy `code.length > 0`. New batches cannot be committed against an EOA target; in-flight batches were already insulated by the D-03 per-batch snapshot pattern. The MED-7 follow-up closes the remaining one-snapshot-cycle window where a compromised owner could re-acquire the in-flight-mutation primitive. ✓

**Note:** Static-analysis-grade warning — `code.length > 0` is FALSE during a constructor's own execution, but that's irrelevant here since BSR's setters target ALREADY-DEPLOYED contracts. ✓

### D-04 (LOW-3) — `drainFoundationReserve whenNotPaused` → **CONFIRMED CLOSED**

- `StakeBond.sol:571` — `function drainFoundationReserve() external onlyOwner nonReentrant whenNotPaused`.
- `StakeBond.sol:445` — `setFoundationReserveWallet` is now ALSO `whenNotPaused` AND rejects `address(0)` AND requires `code.length > 0` (MED-4 + LOW-3 stacked). The compromised-owner-during-pause exfiltration vector is closed: pause must be lifted before either the destination can be changed or the reserve can be drained. Lifting pause is itself an on-chain breadcrumb. ✓

### D-05 (INFO-4) — Cross-contract pause coordination DOCUMENTED → **CONFIRMED CLOSED**

`StakeBond.sol:79-101` carries the contract-level NatSpec describing:
- The operational invariant ("Pause both contracts together, in either order").
- Each cross-call surface and its degraded-state behavior:
  - `BSR.challengeReceipt → StakeBond.slash` (try/catch — emits `SlashSwallowed` for Forta observability if StakeBond is paused-while-BSR-is-not).
  - `StakeBond.requestUnbond → BSR.lastPendingBatchExpiry` (read-only view, not affected by either pause).
- The operational defense (PRSM-EXPLOIT-PLAYBOOK §11 + multisig batched pause UI).
- The future-work option (shared `PauseCoordinator`).

The docstring matches the actual code: `slash` is `whenNotPaused` so a paused StakeBond will revert and emit `SlashSwallowed`; `lastPendingBatchExpiry` is a public mapping not gated by Pausable. ✓

### D-06 (INFO-5) — `commitBatch` storage-pointer ordering → **CONFIRMED CLOSED**

`BatchSettlementRegistry.sol:455-492` performs all snapshot writes BEFORE the status flip:

| Line | Field | Pre/Post status |
|---|---|---|
| 455 | `b.provider` | PRE |
| 456 | `b.requester` | PRE |
| 457 | `b.merkleRoot` | PRE |
| 458 | `b.receiptCount` | PRE |
| 459 | `b.totalValueFTNS` | PRE |
| 461 | `b.commitTimestamp` | PRE |
| 462 | `b.tier_slash_rate_bps` | PRE |
| 463 | `b.consensus_group_id` | PRE |
| 468 | `b.challengeWindowSecondsAtCommit` | PRE |
| 473 | `b.totalPausedAtBatchOrigin` | PRE |
| 479 | `b.lookbackWindowSecondsAtCommit` | PRE |
| 481 | `b.escrowPoolAtCommit` | PRE |
| 482 | `b.stakeBondAtCommit` | PRE |
| 483 | `b.signatureVerifierAtCommit` | PRE |
| 484 | `b.metadataURI` | PRE |
| **492** | **`b.status = BatchStatus.PENDING`** | **LAST** |
| 499-502 | `lastPendingBatchExpiry[msg.sender]` update | POST (separate mapping, not part of the Batch struct) |

Confirmed: every per-batch snapshot field is populated BEFORE the status flip. The post-status-flip work is the per-provider tracker (separate mapping) and the event emission. Neither involves an external call. The fix is structurally defensive against future edits that might add an external call inside `commitBatch`. ✓

## Re-check of new surfaces opened by the HIGH-1 / HIGH-2 / INFO-5 fixes

### N-01 — `_pause` / `_unpause` overrides: re-pause + never-paused safety

**Concern:** The HIGH-2 fix overrides `_pause()` and `_unpause()` to maintain `pauseStartedAt` + `totalPausedSeconds`. Verified safety:

1. **Never-paused contract.** Defaults: `pauseStartedAt = 0`, `totalPausedSeconds = 0`. `_effectiveElapsed` computes `totalPausedSeconds (0) - b.totalPausedAtBatchOrigin (0) = 0`, then `wall - 0 = wall`. Pre-fix behavior recovered exactly. ✓
2. **Re-pause while paused.** Override at BSR:1100 sets `pauseStartedAt = block.timestamp` BEFORE calling `super._pause()`. If contract is already paused, `super._pause()` reverts (`EnforcedPause`). The entire tx reverts atomically — `pauseStartedAt` is NOT corrupted because EVM rolls back on revert. ✓
3. **Unpause while not paused.** Override at BSR:1111 does `totalPausedSeconds += block.timestamp - pauseStartedAt`; if not paused, `pauseStartedAt = 0`, so the increment would be `block.timestamp` (huge), but `super._unpause()` reverts (`ExpectedPause`) and rolls back the corrupted increment atomically. ✓
4. **Pause accumulator monotonicity.** `totalPausedSeconds` is only incremented (in `_unpause`); never decremented. Sequencer cannot manipulate it backwards. ✓

**No finding.**

### N-02 — `lastPendingBatchExpiry` lifecycle

Already evaluated under D-01 above. Specifically:
- Tracker is monotonic non-decreasing.
- Never decremented when a batch transitions PENDING → FINALIZED / VOIDED.
- Stale values (past timestamps) are naturally dominated by the always-future `localFloor = now + unbondDelaySeconds`.
- Honest providers are NOT over-blocked.
- The `try/catch` in `StakeBond.requestUnbond` gracefully degrades if the slasher contract is replaced with a non-conforming target post-deploy (operationally this branch only fires under a governance-driven contract swap, which is precluded by the `slasher` immutability).

**No finding.**

### N-03 — `_effectiveElapsed` defensive clamp

`BSR:1093` `return wall > pausedSinceCommit ? wall - pausedSinceCommit : 0;` — defends against a pathological invariant violation where `pausedSinceCommit > wall`. By construction this should be unreachable (every paused second was a wall second between commit and now), but the clamp avoids an underflow revert that could brick `finalizeBatch` / `challengeReceipt` / `isFinalizable` / `secondsUntilFinalizable` for a batch under future code paths. ✓

**No finding.**

### N-04 — Re-derivation under reorg

`batchId = keccak256(provider || requester || merkleRoot || receiptCount || block.number || sequence)`. On reorg, `block.number` may shift, the `sequence` (per-provider counter) might increment differently. Re-execution re-derives. The `lastPendingBatchExpiry` mapping is also a deterministic function of re-derived `block.timestamp + challengeWindowSecondsAtCommit`, so it re-derives correctly. ✓

**No finding.**

## Vectors re-evaluated and cleared (no new findings)

| Surface | Verdict |
|---|---|
| ERC-20 return values | Cleared (raw `bool ok = ...; if (!ok) revert TransferFailed();`). Same disposition as 2026-05-06. SafeERC20 migration remains defense-in-depth. |
| Loop bounds | Cleared. `MerkleProof.verify` is O(log N). |
| Reorg sensitivity | Cleared. `batchId` re-derives; `lastPendingBatchExpiry` re-derives. |
| `block.timestamp` manipulation | Cleared. Sequencer-bounded; accumulator is monotonic non-decreasing. |
| Self-destruct on dependencies | Cleared. Post-Solidity-0.8.18 self-destruct is gated; cross-wire is immutable post-HIGH-6/HIGH-7. |
| Storage layout / inheritance | Cleared. None upgradeable. |
| Reentrancy | Cleared. `nonReentrant` on every value-moving path. CEI followed. |
| `MIN_SLASH_GAS` floor | Cleared. 150K above OOG threshold; outer tx reverts cleanly if challenger underfunds. |
| `claimBounty` permissionless self-claim | Cleared. Caller can only claim from their own balance. |
| `Ownable2Step` migration | Cleared on both BSR + StakeBond + EscrowPool. |
| `totalEscrowedBalance` accumulator | Cleared. EscrowPool maintains in lockstep with `balances`. |
| Per-batch snapshot pattern (D-03 + D-05 + D-05-HIGH-2 + MED-3) | Cleared. All four parameters (`challengeWindowSecondsAtCommit`, `lookbackWindowSecondsAtCommit`, `totalPausedAtBatchOrigin`, plus the three cross-wire pointers `escrowPoolAtCommit` / `stakeBondAtCommit` / `signatureVerifierAtCommit`) snapshot at commit. Live setter affects only future commits. |
| `Ed25519Verifier` `verify` `pure` correctness | Cleared at the interface level (out of state-machine scope; covered in Team C surface). |
| Cross-chain replay | N/A (BSR is single-chain). |
| `claimBounty` reentrancy | Cleared. CEI: balance zeroed before transfer. |
| `slash` slasher-only access control | Cleared. `if (msg.sender != slasher) revert CallerNotSlasher(...)`; slasher is immutable. |

## Sign-off

This document is the L4-self-2026-05-07 Team-D re-audit artifact. **Result: zero new HIGH or CRITICAL findings; zero new MEDIUM findings; all prior Team-D findings (D-01 through D-06) confirmed closed.** PRSM-POL-2 §4.1 pass criterion satisfied from the lifecycle / state-machine angle.

— Generated by Team D agent (Lifecycle / State-Machine Auditor), 2026-05-07.
