# Team D — State Composition Self-Audit Findings (2026-05-06)

## Summary

Re-audit at the post-remediation tip, focused on whether HIGH-2/3/6/7 fixes introduced new state-composition bugs or left analogous patterns unfixed. The remediations are largely sound — `nonReentrant` + CEI intact, both immutable cross-wires correctly enforced, per-batch snapshot pattern (D-05) extended to escrowPool/stakeBond/signatureVerifier, `whenNotPaused` coverage hits all value-moving paths, `MIN_SLASH_GAS` floor closes the OOG corner.

**1 HIGH** the HIGH-2 fix did not fully close, **2 MEDIUMs** introduced or left by the immutability moves, plus 3 lower-severity composition observations.

## Findings

### D-01 — HIGH-2 fix is incomplete: D-05 per-batch snapshot reopens the slash-evasion race (Severity: HIGH)

**Location:**
- `StakeBond.requestUnbond` lines 248–262 (reads live `slasher.challengeWindowSeconds()`)
- `BatchSettlementRegistry.commitBatch` line 381 (snapshots `challengeWindowSecondsAtCommit`)
- `BatchSettlementRegistry.challengeReceipt` line 497 (gates challenge eligibility on per-batch snapshot, not live global)

**Description:**
HIGH-2 (`unbondDelay ≥ challengeWindow` cross-contract invariant) and D-05 (per-batch `challengeWindowSecondsAtCommit` snapshot) were remediated independently and their interaction was not modeled. `StakeBond.requestUnbond` clamps `unbond_eligible_at` to `now + slasher.challengeWindowSeconds()` — the LIVE mutable global. But because of D-05, an already-PENDING batch keeps its commit-time snapshot as the binding challenge window, even after the global is reduced.

**Attack scenario:**
1. Owner sets global `challengeWindowSeconds = 30 days`. Provider commits batch B1 → snapshot pinned at 30 days.
2. Provider double-spends in batch B2.
3. Owner sets global `challengeWindowSeconds = 1 hour` (allowed, ≥ MIN). B1's snapshot remains 30 days.
4. Provider calls `requestUnbond()`. Slasher returns LIVE 1h. With `unbondDelaySeconds = 1 day`, eligible-at = `now + 1 day`.
5. Day 1: provider `withdraw()`. Status → WITHDRAWN.
6. Day 5: challenger lands `challengeReceipt(B1, ..., DOUBLE_SPEND)`. Within snapshot 30d. Slash branch entered. `StakeBond.slash` reverts `NotSlashable(WITHDRAWN)`. `try/catch` swallows; `SlashSwallowed` fires; provider keeps full stake.

This is the same outcome as the original D-01 race — only the trigger has moved. No longer reachable from a pristine governance posture, but evaporates as soon as the live window is changed downward, which the D-05 remediation explicitly markets as "safe for in-flight batches."

**Impact:** HIGH. Reproduces the original CRIT/HIGH-2 economic outcome under a single subsequent governance action. Compromised-owner scenario is the obvious vector, but also reachable by a benign owner reducing the window after a hot-period spike — a normal operational move.

**Recommended fix:**
1. In `StakeBond.requestUnbond`, clamp eligibility against the **maximum challenge window currently in flight** for this provider's PENDING batches. BSR keeps `mapping(address provider => uint64 maxPendingDeadline)` updated on commitBatch.
2. In `BatchSettlementRegistry.setChallengeWindowSeconds`, reject reductions when any PENDING batch exists with `challengeWindowSecondsAtCommit > newSeconds` AND provider is BONDED.

Defense-in-depth: Forta should alert specifically on `SlashSwallowed{reason ∈ {DOUBLE_SPEND, INVALID_SIGNATURE, CONSENSUS_MISMATCH}}` paired with provider `StakeStatus == WITHDRAWN`.

---

### D-02 — Constructor accepts `address(0)` for the immutable slasher / settlementRegistry, with no production-deploy guard (Severity: MEDIUM)

**Location:**
- `EscrowPool.sol:90–102` constructor
- `StakeBond.sol:160–178` constructor

**Description:**
HIGH-6 / HIGH-7 made `settlementRegistry` and `slasher` immutable, removed setters, documented "production MUST set this." But constructors still accept `address(0)` to support unit-test convenience. With no setter, `address(0)` deploy is **permanently bricked**: `EscrowPool.settleFromRequester` reverts `CallerNotRegistry(_, address(0))` for every call; `StakeBond.slash` reverts `CallerNotSlasher(_, address(0))` likewise. No setter to recover.

The inscribed comment ("production MUST set this") is operational-policy-only — same shape as the B-CROSS-2 / D-07 docstring weakness rejected as inadequate.

**Attack scenario:**
Not adversarial; ceremony / deploy-script misconfiguration. But pre-handoff a malicious deployer or compromised CI machine could deliberately deploy with `address(0)` knowing slashing will never fire. Symptoms identical to D-01 from outside.

**Impact:** Bricks value-moving / slashing path with no recovery.

**Recommended fix:** Constructor require `initialRegistry != address(0)` / `initialSlasher != address(0)` in production builds. Either unconditionally + update tests to wire stub addresses, or gate behind `bool allowZeroForTesting`. Deploy verifier should also assert non-zero.

---

### D-03 — `BatchSettlementRegistry.setEscrowPool` / `setStakeBond` / `setSignatureVerifier` accept any address with no contract-bytecode or interface check (Severity: MEDIUM)

**Location:** `BatchSettlementRegistry.sol:792–820`.

**Description:**
The D-03 fix snapshots these three pointers per-batch (good — closes mid-flight rotation). But setters still accept arbitrary addresses — including EOAs, contracts at wrong interface, `address(0)`, self-references. A compromised owner who waits one snapshot cycle (one commitBatch) can re-acquire all the in-flight-mutation primitives the D-03 fix was supposed to remove. The fix only protected already-PENDING batches; new commits are still hostage.

**Attack scenario:**
1. Owner compromised; calls `bsr.setSignatureVerifier(maliciousVerifier)`.
2. Within next block, attacker calls `commitBatch(...)` → `signatureVerifierAtCommit = maliciousVerifier`.
3. INVALID_SIGNATURE challenges run against malicious verifier — false-positive on correct signatures, whitewash bad ones.

Same pattern works for `setEscrowPool` (route settlement to attacker-drained pool) and `setStakeBond` (point to attacker-controlled bond returning bogus `challengeWindowSeconds()` to weaken downstream).

**Recommended fix:** On all three setters require:
- `newAddress != address(0)` and `newAddress.code.length > 0`
- `ERC165 supportsInterface` or bespoke `isPRSMEscrowPool()` sentinel
- Bonus: 7-day timelock for non-emergency rotations

---

### D-04 — `drainFoundationReserve` not gated by `whenNotPaused`; drains during incident response (Severity: LOW)

**Location:** `StakeBond.sol:470–484`.

**Description:** Pause-coverage exempts admin setters and `drainFoundationReserve` for emergency-rotation rationale. For setters this is correct. For `drainFoundationReserve` it's questionable: the function moves real FTNS to `foundationReserveWallet`, which is itself owner-mutable. Compromised owner during pause can `setFoundationReserveWallet(attacker)` then `drainFoundationReserve()` — pause does not protect this value pool. Per Stated Invariant #3 ("when paused, no value can move in any direction"), this is a violation.

**Recommended fix:** Add `whenNotPaused` to `drainFoundationReserve`. Add `whenNotPaused` to `setFoundationReserveWallet`. Setters that don't move value can stay unpaused.

---

### D-05 — `requestUnbond` doesn't check `whenNotPaused` on the slasher before reading `challengeWindowSeconds()` (Severity: INFO)

Today's slasher is BSR with public storage variable returning regardless of pause — functionally harmless. But cross-contract pause coordination is implicit; future slasher upgrade could introduce coherence bugs. Tomorrow-problem; document the operational invariant or add a PauserCoordinator.

---

### D-06 — `commitBatch` mid-block sequence consumed under storage-pointer assignment refactor (Severity: INFO)

**Location:** `BatchSettlementRegistry.sol:349, 358–386`.

**Description:** HIGH-2 / D-03 / D-05 refactored commitBatch from struct-literal to storage-pointer field-by-field assignment to avoid stack-too-deep. Behavior equivalent today. But `b.status == PENDING` is set at line 374 BEFORE the snapshot writes at 383–385. No external call between, so no present-day exploit. But makes function fragile to future edits.

**Recommended fix:** Reorder so `b.status = PENDING` is the LAST storage write. Same gas, strictly safer invariant. Or add a `// invariant: status set last` comment.

## Notes / Out-of-scope

- **Reentrancy review:** cleared. CEI followed. `nonReentrant` applied where external calls happen.
- **Storage layout / inheritance:** cleared. None upgradeable.
- **ERC-20 return values:** raw `bool ok = ftns.transfer(...); if (!ok) revert TransferFailed();` — correct against OZ-style ERC-20s; rejects USDT-style. FTNS is OZ-based + `setFtnsToken` guarded by `totalEscrowedBalance == 0`. Acceptable. SafeERC20 migration would be defense-in-depth.
- **Loop bounds:** cleared. `MerkleProof.verify` is `O(log N)`.
- **`block.timestamp` manipulation:** bounded; sequencer can move forward but commit/challenge/finalize deltas are O(days).
- **Reorg sensitivity:** cleared. `batchId = keccak256(... block.number ...)` re-derives.
- **Out-of-scope handed to other teams:** signingMessageHash binding (Team C); access-control edge cases (Team B); cryptographic correctness (Team C / L3).
