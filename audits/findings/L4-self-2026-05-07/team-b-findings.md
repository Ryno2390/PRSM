# Team B — Access Control Re-Audit Findings (2026-05-07)

## Summary

Re-audit of the L4-self-2026-05-06 access-control surface against the
post-fix tip (`2c93c5bb`). All prior B-team findings (B-01 / B-02 / B-03)
are **properly closed**; the new code introduced by the remediations
(`_pause`/`_unpause` overrides, `pauseStartedAt`, `totalPausedSeconds`,
`lookbackWindowSecondsAtCommit`, `lastPendingBatchExpiry`,
`SetterTargetNotContract`, `WalletNotContract`, INFO-3 test factory
defaults) introduces **zero new HIGH or CRITICAL findings**. Two
INFO-level consistency observations are noted in §3 for code-hygiene
follow-up but neither is in scope to block the re-audit pass.

**Verdict:** **PASS**. 0 new CRITICAL · 0 new HIGH · 0 new MEDIUM ·
0 new LOW · 2 new INFO. Prior B-01 / B-02 / B-03 stay closed.

## 1. Re-checks of prior findings

### B-01 (HIGH-2) — Owner can shrink in-flight challenge windows by sustained pause

**Status: CLOSED.**

**Evidence:**
- `BatchSettlementRegistry.sol:298` declares `uint256 public totalPausedSeconds`,
  `:303` declares `uint256 public pauseStartedAt`.
- `:1100-1114` overrides `_pause`/`_unpause`:
  - `_pause` writes `pauseStartedAt = block.timestamp` BEFORE
    `super._pause()` (so an OZ revert on already-paused rolls the
    write back via tx-level revert).
  - `_unpause` adds `block.timestamp - pauseStartedAt` to
    `totalPausedSeconds`, zeros `pauseStartedAt`, then calls
    `super._unpause()` (so an OZ revert on already-unpaused rolls
    the accumulator update back).
- `:1084-1094` `_effectiveElapsed(b)` returns
  `wall - (totalPausedSeconds - b.totalPausedAtBatchOrigin)` clamped
  at zero (defensive against any future invariant break).
- `:543`, `:617`, `:1052`, `:1062` — all four call sites
  (finalizeBatch / challengeReceipt / isFinalizable /
  secondsUntilFinalizable) consult `_effectiveElapsed`.
- `:473` snapshots `b.totalPausedAtBatchOrigin = totalPausedSeconds`
  on commit so post-commit pauses are subtracted from the elapsed
  delta.

**Reasoning checked (per re-check checklist):**
- ✓ Pause-time accumulator only increments on COMPLETED pauses
  (`_unpause` is the only writer of `totalPausedSeconds`).
- ✓ Cannot be retroactively manipulated: `pauseStartedAt` is
  set ONLY by `_pause` and zeroed by `_unpause`. There is no
  setter, no admin path, no other writer.
- ✓ Symmetric across pause/unpause: every pause that completes
  contributes exactly `unpause_ts - pause_ts` to the accumulator.
- ✓ No reentrancy via OZ Pausable: `super._pause()` and
  `super._unpause()` only emit events; OZ `Pausable.sol` v5.3.0
  has no callback, no external call, no `_msgSender()` re-entry path.
- ✓ Symmetry in OZ guards: `_pause` has `whenNotPaused`,
  `_unpause` has `whenPaused`. Double-pause / double-unpause
  reverts at OZ level → tx-level revert rolls back the accumulator
  write.
- ✓ View functions `isFinalizable` / `secondsUntilFinalizable`
  documented as "momentarily under-report during in-progress
  pause"; safe because both are advisory (non-state-changing) and
  finalize/challenge gate on `whenNotPaused` so the paused-state
  numbers are unactionable until atomic catch-up at unpause.

**Regression test coverage:** `test/L4SelfAuditFixes.test.js`
HIGH-2 cluster passes; per-batch-snapshot tests exercise both pause
and unpause across multiple batches.

---

### B-02 (MED-3) — `setSettlementLookbackWindow` retroactively flips EXPIRED eligibility

**Status: CLOSED.**

**Evidence:**
- `BatchSettlementRegistry.sol:162-170` adds
  `uint64 lookbackWindowSecondsAtCommit` to `Batch` struct.
- `:479` snapshots `b.lookbackWindowSecondsAtCommit = uint64(settlementLookbackWindowSeconds)`
  on commit.
- `:641` passes `b.lookbackWindowSecondsAtCommit` (NOT the live
  global) into `_handleExpired`.
- `:808-818` `_handleExpired` consults `lookbackWindowAtCommit`
  exclusively. The live `settlementLookbackWindowSeconds` is no
  longer read on the challenge path.
- Pattern is identical to D-05's `challengeWindowSecondsAtCommit`.

**Setter `setSettlementLookbackWindow` (`:973`):** still mutates the
live global. By design — only affects FUTURE batches. Owner cannot
retroactively flip in-flight EXPIRED eligibility either direction.

---

### B-03 (MED-4) — `setFoundationReserveWallet` accepts zero/non-canonical address

**Status: CLOSED.**

**Evidence (`StakeBond.sol:445-451`):**
```
function setFoundationReserveWallet(address newWallet) external onlyOwner whenNotPaused {
    if (newWallet == address(0)) revert ZeroAddress();
    if (newWallet.code.length == 0) revert WalletNotContract(newWallet);
    address old = foundationReserveWallet;
    foundationReserveWallet = newWallet;
    emit FoundationReserveWalletUpdated(old, newWallet);
}
```

- ✓ Zero-reject (line 446).
- ✓ `code.length == 0` rejects EOAs / non-deployed addresses
  (line 447).
- ✓ `whenNotPaused` modifier (LOW-3 fix bundled).
- ✓ Custom `WalletNotContract(address)` error provides forensic
  detail for off-chain monitoring.
- ✓ `drainFoundationReserve` is also `whenNotPaused` (`:571`).

**Canonical-Safe pin** is documented as a deploy-script concern
(out of contract scope), per the original B-03 recommendation #3.

**Defense-in-depth (timelock)** noted but not implemented —
acceptable per the consolidated-findings disposition matrix
(reserve drain is owner-only and emits
`FoundationReserveWalletUpdated` for off-chain Forta monitoring).

## 2. Re-checks of NEW surfaces introduced by the fixes

### 2.1 New `whenNotPaused` on `drainFoundationReserve` + `setFoundationReserveWallet`

**No new pause-forever scenarios introduced.**

Re-check question: does the new `whenNotPaused` on these two
functions create any state where a paused contract is
unrecoverable?

**Analysis:**
- `pause()` and `unpause()` (`StakeBond.sol:471, :476`) are NOT
  `whenNotPaused`-gated. Owner retains the unpause primitive
  unconditionally.
- `setUnbondDelay`, `setFoundationReserveWallet`,
  `drainFoundationReserve` are now `whenNotPaused`. To recover
  drain capability or rotate the wallet, owner must `unpause()`
  first — which is unrestricted.
- B-PAUSE-1 (the original "all-PAUSERs renounce while paused →
  permanent freeze" vector on FTNS) is closed transitively via
  HIGH-5: `renounceRole(DEFAULT_ADMIN_ROLE)` reverts on
  FTNSTokenSimple, so even if every PAUSER renounced, admin can
  still `grantRole(PAUSER_ROLE, fresh)` to recover. The OZ
  Pausable surface in BSR / EscrowPool / StakeBond uses
  `Ownable2Step` not AccessControl, so the renounceRole vector
  doesn't apply — the owner cannot renounce ownership without
  going through the explicit `transferOwnership` + `acceptOwnership`
  two-step (see §2.5).
- StakeBond constructor (`:205-228`) hard-rejects address(0)
  for slasher (MED-6 fix). Combined with `Ownable(initialOwner)`
  (OZ rejects zero-owner at construction), there is no
  pause-forever construction path.

**Verdict:** No new permanent-freeze surface.

### 2.2 New `WalletNotContract` + `SetterTargetNotContract` errors

**No selector collision; no unexpected revert path.**

**Analysis:**
- `WalletNotContract(address provided)` is only thrown by
  `StakeBond.setFoundationReserveWallet` after the zero-check.
- `SetterTargetNotContract(address provided)` is only thrown by
  `BatchSettlementRegistry.setEscrowPool` / `setStakeBond` /
  `setSignatureVerifier` after the explicit non-zero gate
  (`newAddr != address(0) && newAddr.code.length == 0`).
  Address(0) "disable" mode preserved per the documented BSR
  semantics (see `:919` and similar comments).
- 4-byte selector check: errors are uniquely named and have
  unique `bytes` signatures. Solidity's selector collision
  surface is `0xffffffff`-scale; with two distinct names + types
  there's no collision risk in practice.
- Revert paths: both are pre-state-mutation guards; no partial
  state mutation occurs before the revert. Storage writes happen
  AFTER both checks pass.

**Edge case (informational, not a finding):**
`EscrowPool.setFtnsToken` (`:216-224`) does NOT have an analogous
`code.length > 0` check on `newToken`. The setter is gated by:
1. `onlyOwner`
2. `newToken != address(0)` (kept)
3. `totalEscrowedBalance == 0` (B-CROSS-2 fix preserved)

Because (3) requires every requester balance + every PENDING
batch to have settled to zero before the swap is allowed, and
because the swap itself is a documented escape hatch, the
absence of a `code.length > 0` check is consistent with the
pre-existing risk model. See INFO-B7-1 in §3 for the
consistency note.

### 2.3 `lastPendingBatchExpiry[provider]` public storage slot

**No access-control surface depends on this slot being
non-public.**

**Analysis:**
- Slot is `mapping(address provider => uint64) public lastPendingBatchExpiry`
  at `:282`.
- Read by `StakeBond.requestUnbond` via the
  `ISlasherWithProviderExpiry.lastPendingBatchExpiry(provider)`
  interface — public visibility is REQUIRED for this read to
  work.
- Written exclusively by `commitBatch` (`:500-501`) using
  monotonic max-update logic. Stale values (past timestamps from
  finalized batches) are naturally dominated by `now + unbondDelay`
  on the StakeBond side via max-of-floors.
- No off-chain caller can manipulate this slot. No setter exists.
  No reset on `finalizeBatch` (intentional — stale values are safe
  by design, and clearing on finalize would add gas + a
  reentrancy surface during finalize).
- Not invoked by any access-control check inside BSR (it's
  read-only output). The ABI exposure is the entire point of the
  fix; making it private would break the HIGH-1 remediation.

**Verdict:** Public is correct + intentional.

### 2.4 INFO-3 test factory hash defaults

**No prior audit-team-b test depends on the old `ZeroHash`
default.**

**Evidence:**
- `contracts/test/audit-team-b/B-AccessControl-PoC.test.js:96, :296`
  uses `ethers.ZeroHash` as the literal value of
  `DEFAULT_ADMIN_ROLE` (which IS `bytes32(0)` per OZ
  AccessControl convention) — not as a leaf-factory default.
- `B-RenounceRole-Override.test.js:50, :90` likewise uses
  `ethers.ZeroHash` as the `DEFAULT_ADMIN_ROLE` constant.
- The INFO-3 fix targeted the `makeLeaf` factory in
  `audit-team-c/C-INT-01-invalid-signature-forgery.test.js:43-63`
  (signing-message default), NOT any role-identifier constant.

**Test execution evidence:** All 20 audit-team-b tests pass on
the post-fix tip (`B-AccessControl-PoC.test.js` + `B-RenounceRole-Override.test.js`).

### 2.5 Ownable2Step on all 7 contracts

**Confirmed.**

- `BatchSettlementRegistry.sol:133`: `Ownable2Step, Pausable`
- `EscrowPool.sol:37`: `Ownable2Step, ReentrancyGuard, Pausable`
- `StakeBond.sol:103`: `Ownable2Step, ReentrancyGuard, Pausable`
- `CompensationDistributor.sol:49`: `Ownable2Step, ReentrancyGuard`
- `EmissionController.sol:41`: `Ownable2Step, ReentrancyGuard`
- `KeyDistribution.sol:69`: `Ownable2Step, ReentrancyGuard`
- `StorageSlashing.sol:54`: `Ownable2Step, ReentrancyGuard`

All 7 inherit OZ `Ownable2Step` v5.3.0 — the two-step
`transferOwnership` → `acceptOwnership` migration is the only
ownership-handoff path. No bare `transferOwnership` override.
No `renounceOwnership` override needed (OZ Ownable's renounce is
acceptable for the post-fully-decentralised end-state; not in
scope for current Foundation-Safe-controlled deploy).

### 2.6 HIGH-5 — `renounceRole(DEFAULT_ADMIN_ROLE)` revert

**Confirmed.** `FTNSTokenSimple.sol:142-150`:
```
function renounceRole(bytes32 role, address callerConfirmation)
    public
    override(AccessControlUpgradeable)
{
    if (role == DEFAULT_ADMIN_ROLE) {
        revert("DEFAULT_ADMIN_ROLE renounce disabled - use grantRole(new) + revokeRole(old)");
    }
    super.renounceRole(role, callerConfirmation);
}
```
Closes the foot-gun where a single bad multi-sig tx renouncing
admin would permanently lock UUPS upgrade authorization +
MINTER/BURNER/PAUSER role grants. B-PAUSE-1 (all-PAUSERs renounce
while paused) is closed transitively as documented in the docstring
(`:133-140`).

### 2.7 HIGH-6 — `EscrowPool.settlementRegistry` immutable

**Confirmed.** `EscrowPool.sol:54`:
```
address public immutable settlementRegistry;
```
Setter (`setSettlementRegistry`) is documented as removed (`:194-197`).
Constructor hard-rejects `initialRegistry == address(0)` at
`:106` (MED-6 fix bundled).

### 2.8 HIGH-7 — `StakeBond.slasher` immutable

**Confirmed.** `StakeBond.sol:131`:
```
address public immutable slasher;
```
Setter removed (`:416-419`). Constructor hard-rejects
`initialSlasher == address(0)` at `:224` (MED-6 fix bundled).

### 2.9 B-CROSS-2 — `setFtnsToken` with non-zero pending balance reverts

**Confirmed.** `EscrowPool.sol:216-224`:
```
function setFtnsToken(address newToken) external onlyOwner {
    if (newToken == address(0)) revert ZeroAddress();
    if (totalEscrowedBalance != 0) {
        revert PendingBalancesNonZero(totalEscrowedBalance);
    }
    ...
}
```
`totalEscrowedBalance` accumulator is updated by `deposit`
(`:127`), `withdraw` (`:150`), `settleFromRequester` (`:184`).
Owner cannot swap the token while any requester balance or
in-flight batch backing remains.

### 2.10 Constructor poisoning (zero-address checks)

| Contract | initialOwner | other immutables |
|---|---|---|
| BSR | OZ Ownable rejects 0 (`:401`) | n/a |
| EscrowPool | OZ Ownable rejects 0 (`:94`) | `ftnsAddress` rejected (`:95`); `initialRegistry` rejected (`:106`) — MED-6 |
| StakeBond | OZ Ownable rejects 0 (`:210`) | `ftnsAddress` rejected (`:211`); `initialSlasher` rejected (`:224`) — MED-6 |
| FTNSTokenSimple (initialize) | UUPS — initial admin from caller; `_grantRole` no-zero-check by design | n/a (not constructor) |

**Verdict:** All immutable cross-wires are zero-address protected
at construction. No permanently-bricked deploy is possible from
mis-configured immutables.

### 2.11 Pauser locking (B-PAUSE-1) — re-check

**Closed transitively** via HIGH-5. See §2.1 + §2.6.

### 2.12 Initializer re-entry (UUPS) — FTNSTokenSimple

`FTNSTokenSimple.sol:46` uses the `initializer` modifier. Constructor
disables initializers via `_disableInitializers()` at `:38`. UUPS
upgrade gate on `_authorizeUpgrade` requires `DEFAULT_ADMIN_ROLE`
(`:115`). No initializer re-entry vector visible.

## 3. New informational observations (no severity escalation)

### INFO-B7-1 — `EscrowPool.setFtnsToken` lacks `code.length > 0` check

**Severity: INFO. Not a finding for this re-audit pass.**

**Location:** `EscrowPool.sol:216-224`.

**Observation:** MED-7 added `code.length > 0` rejection on the
three BSR cross-wire setters (`setEscrowPool`, `setStakeBond`,
`setSignatureVerifier`). The analogous setter in EscrowPool
(`setFtnsToken`) does NOT enforce a `code.length > 0` check on
`newToken`.

**Risk assessment:**
- Setter is gated on `totalEscrowedBalance == 0` (B-CROSS-2):
  no requester funds at risk during swap.
- Setter is `onlyOwner` (Foundation Safe).
- Operationally, swapping FTNS → an EOA bricks the contract for
  future deposits (transferFrom decode error) but is recoverable
  by calling `setFtnsToken(real_ftns)` again.
- No fund loss path. No new attack primitive.

**Recommendation:** For consistency with MED-7's defensive style,
consider adding `if (newToken.code.length == 0) revert TokenNotContract(newToken);`
in a follow-up cleanup commit. Not blocking.

### INFO-B7-2 — `lastPendingBatchExpiry` not cleared on `finalizeBatch`

**Severity: INFO. Not a finding for this re-audit pass.**

**Location:** `BatchSettlementRegistry.sol:529-558` (finalizeBatch
does not touch `lastPendingBatchExpiry`).

**Observation:** The HIGH-1 tracker uses monotonic max-update on
commit and is never decremented. After the last PENDING batch for
a provider finalizes, the slot retains the (now-stale) max expiry
of its last batch.

**Risk assessment:**
- StakeBond.requestUnbond uses `max(localFloor, slasher.lastPendingBatchExpiry(provider))`.
  A stale past timestamp is dominated by `localFloor = now + unbondDelay`,
  which is always strictly in the future.
- If `unbondDelay` is configured >= the previous `challengeWindowSeconds`
  at the time of the last commit, no extra delay is imposed.
- If `unbondDelay` is configured < the previous `challengeWindowSeconds`,
  the stale value WILL impose a wait equal to the difference. This
  is conservative (over-locks honest providers slightly) but not
  unsafe.

**Recommendation:** No change. Documented in the field NatSpec
(`:265-281`). Clearing on finalize would add gas + a state-mutation
on the permissionless finalize path. Conservative-by-default
behavior is the right trade-off.

## 4. Vectors evaluated and cleared (post-fix tip)

| Vector | Verdict |
|--|--|
| Pause-time accumulator monotonicity (B-01 fix) | Cleared |
| `_pause`/`_unpause` override safety (re-entrancy, ordering, OZ guard symmetry) | Cleared |
| `pauseStartedAt` write/read symmetry | Cleared |
| `totalPausedSeconds` updated only on completed pauses | Cleared |
| `lookbackWindowSecondsAtCommit` snapshot consumed by `_handleExpired` (B-02 fix) | Cleared |
| `setFoundationReserveWallet` zero-reject + EOA-reject + `whenNotPaused` (B-03 fix) | Cleared |
| `drainFoundationReserve` `whenNotPaused` (LOW-3 fix) | Cleared |
| `SetterTargetNotContract` revert path on BSR cross-wire setters (MED-7 fix) | Cleared |
| Constructor zero-address rejection on EscrowPool + StakeBond immutables (MED-6 fix) | Cleared |
| `lastPendingBatchExpiry` public visibility intentional + correct (HIGH-1 fix) | Cleared |
| INFO-3 `makeLeaf` non-degenerate default doesn't break audit-team-b tests | Cleared |
| HIGH-5 FTNS `renounceRole(DEFAULT_ADMIN_ROLE)` revert | Cleared |
| HIGH-6 EscrowPool.settlementRegistry immutable | Cleared |
| HIGH-7 StakeBond.slasher immutable | Cleared |
| B-CROSS-2 setFtnsToken non-zero pending balance reverts | Cleared |
| Ownable2Step on all 7 in-scope contracts | Cleared |
| Initializer re-entry on FTNSTokenSimple | Cleared |
| Pauser locking (B-PAUSE-1) closed transitively via HIGH-5 | Cleared |

## 5. Notes / Out-of-scope

- Foundation Safe owners (out of hardhat scope per re-audit brief).
- Ed25519Verifier — pure, no state, no Ownable; nothing to re-check.
- `RoyaltyDistributor` / `ProvenanceRegistryV2` — out of L4 self-audit
  contract scope; covered separately by L3 / L11 / on-chain treasury
  workstreams.

## 6. Bottom-line

Post-remediation, the access-control surface of the L4 self-audit
contract bundle (`BatchSettlementRegistry`, `EscrowPool`, `StakeBond`,
`Ed25519Verifier`, plus FTNSTokenSimple for HIGH-5) is materially
tighter than the L4-self-2026-05-06 baseline. **All three prior
B-team findings (B-01 / B-02 / B-03) are properly closed; the
remediation code introduces zero new HIGH or CRITICAL access-control
findings.** Two INFO-level consistency observations
(`setFtnsToken` lacks contract-check; `lastPendingBatchExpiry` not
cleared on finalize) are documented for future cleanup but neither
blocks the re-audit pass nor materially affects security posture.

Re-audit verdict for Team B: **PASS** under PRSM-POL-2 §4.1
("0 unremediated CRITICAL; HIGH findings either remediated or
accepted-with-recorded-rationale"; today's run yields 0 new
CRITICAL ✓ and 0 new HIGH ✓).
