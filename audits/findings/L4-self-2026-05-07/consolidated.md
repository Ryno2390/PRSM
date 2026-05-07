# L4 Self-Audit Re-Run — Consolidated Findings (2026-05-07)

## 1. Run metadata

- **Tip audited:** `2c93c5bb` (post-MEDIUM-cluster + post-docstring tip).
- **Teams:** A (Economic), B (Access Control), C (Crypto + Integrity), D (Lifecycle / State Machine), each running independently against the same tip with full read access to the prior round's findings.
- **Pass criterion (POL-2 §4.1):** "0 unremediated CRITICAL findings; HIGH findings either remediated or accepted-with-recorded-rationale."
- **Headline:** **0 CRITICAL · 1 HIGH (now REMEDIATED) · 1 MEDIUM (re-classification) · 0 LOW · 2 INFO** across the 4 teams' new findings on the post-fix tip.

## 2. Severity rollup

| Severity | Count | Disposition |
|---|---|---|
| CRITICAL | 0 | n/a |
| HIGH | 1 (A-06) | **REMEDIATED 2026-05-07** in this same sprint via the paired-tracker patch (BSR `lastPendingBatchPausedAtAccrual` + `ISlasherWithProviderExpiryAndPause` reader in StakeBond). |
| MEDIUM | 1 (A-07) | Re-classification: MED-7 in 2026-05-06 disposition matrix moves from REMEDIATED → REMEDIATED-PARTIAL. The 7-day timelock the policy referenced was NOT shipped; only the `code.length > 0` check was. The per-batch D-03 snapshot still protects in-flight batches. |
| LOW | 0 | n/a |
| INFO | 2 (A-08, B-INFO-B7-1, B-INFO-B7-2) | A-08: RoyaltyDistributor donation strand mirrors LOW-1 (bundle when convenient). B-INFO-B7-1: `EscrowPool.setFtnsToken` lacks `code.length > 0` check (consistency, risk-bounded). B-INFO-B7-2: `lastPendingBatchExpiry` not cleared on finalize (conservative-by-design, non-exploitable). |

## 3. The HIGH (A-06) — fix shipped this sprint

### Finding
The 2026-05-06 HIGH-1 fix (`lastPendingBatchExpiry` per-provider tracker) and HIGH-2 fix (`totalPausedSeconds` accumulator + per-batch snapshot) were independently correct but did NOT compose. HIGH-1 stored wall-clock expiry (`commitTimestamp + windowAtCommit`); HIGH-2 made the actual challenge expiry pause-extended (wall-clock + `pausedSinceCommit`). Under the canonical A-01 trigger (governance `setChallengeWindowSeconds` reduction) PLUS any non-zero pause during the batch's window, the unbond floor read by `StakeBond.requestUnbond` was shorter than the actual effective challenge expiry by `pausedSinceCommit`. Provider could withdraw at the wall-clock boundary; a successful late challenge in the `pausedSinceCommit`-shaped tail of the effective window would hit `WITHDRAWN` → `SlashSwallowed`, re-opening the original A-01/D-01 economic outcome.

### Fix (commit this sprint, post-`2c93c5bb`)
Mirrors the HIGH-2 `_effectiveElapsed` arithmetic:

- **BSR-side:** new `mapping(address provider => uint64) public lastPendingBatchPausedAtAccrual` — snapshot of `totalPausedSeconds` at the moment the per-provider expiry was last updated. Written ATOMICALLY with `lastPendingBatchExpiry` in `commitBatch` (only when the new expiry strictly dominates the prior tracker value).
- **StakeBond-side:** new interface `ISlasherWithProviderExpiryAndPause` exposing `lastPendingBatchExpiry`, `lastPendingBatchPausedAtAccrual`, and `totalPausedSeconds`. `requestUnbond` computes:
  ```
  pauseAdjustedExpiry = lastPendingBatchExpiry(p)
                       + (totalPausedSeconds - lastPendingBatchPausedAtAccrual(p))
  ```
  and uses it as the slasher floor.
- **Backwards-compat:** the reader uses `try/catch` around each new getter, so a pre-A-06-fix BSR build (or non-BSR slasher contract) gracefully falls back to the un-pause-adjusted expiry. The wall-clock floor remains correct for the no-pause case (which is the common case).

### Tests added
3 new tests in `test/L4SelfAuditFixes.test.js`:
- `commitBatch snapshots totalPausedSeconds atomically with expiry`
- `A-06 vector: requestUnbond floor pause-extends after governance window-reduction + post-commit pause` — full attack scenario
- `no-pause path: A-06 fix is a no-op for the common case` — common-path regression guard

Total Hardhat suite: **527 passing** (was 524 pre-A-06 fix).

## 4. The MEDIUM (A-07) — re-classification only

### Finding
The 2026-05-06 disposition note for MED-7 read "Setter validation with `code.length > 0` + interface check" and was marked REMEDIATED in the matrix. The remediation shipped only the `code.length > 0` check — no interface check (low-cost, deferred), no 7-day timelock (referenced in the policy footnote but not in code). The per-batch D-03 snapshot correctly protects already-PENDING batches from mid-flight rotation, but a single-batch capture window remains for any batch committed after a malicious rotation.

### Disposition
Re-classify MED-7 in `audits/findings/L4-self-2026-05-06/consolidated.md` from `REMEDIATED` → `REMEDIATED-PARTIAL (code.length check only; 7-day timelock + ERC165 interface check deferred to v2 deploy)`. Operational defense (Foundation Safe 2-of-3 + 14-day public review window per POL-2 §4) covers the residual risk for the v1 deploy round. CompensationDistributor `setPoolAddresses` shares the same shape; documented in Team A's "vectors evaluated and cleared" table.

## 5. The 3 INFOs — defensive cleanups

- **A-08:** RoyaltyDistributor mirrors LOW-1 / A-04 donation strand. RoyaltyDistributor has no `Ownable*` so a `recoverStranded` surface would need an owner role added first. Bundle with the LOW-1 fix when convenient.
- **B-INFO-B7-1:** `EscrowPool.setFtnsToken` lacks `code.length > 0` check (inconsistency with the BSR setters). Risk-bounded by B-CROSS-2's `totalEscrowedBalance == 0` precondition. Trivial to add.
- **B-INFO-B7-2:** `lastPendingBatchExpiry` is never cleared when a batch finalizes; stale future values are dominated by the always-future `localFloor`, so no over-blocking; not exploitable. Conservative-by-design. Could be cleared in `finalizeBatch` for cleanliness.

## 6. Vectors evaluated and cleared (cross-team consensus)

- HIGH-1 (`lastPendingBatchExpiry` tracker) — monotonic, overflow-safe, stale-value-graceful. Cleared in the no-pause case; pause case fixed by A-06 above.
- HIGH-2 (`totalPausedSeconds` accumulator + per-batch snapshot) — symmetric, reentrancy-free under OZ Pausable v5.
- MED-3 (lookback snapshot) — correct semantics; pause does NOT extend receipt validity (correct direction for challengers).
- MED-4 (foundation wallet hardening) — zero + EOA + paused all rejected.
- MED-6 (constructor zero-reject) — both StakeBond + EscrowPool reject `address(0)`.
- LOW-3 (`drainFoundationReserve` whenNotPaused) — confirmed; compromised owner must unpause first (on-chain breadcrumb).
- INFO-3 (test factory hash defaults) — no PoC test silently passes.
- INFO-4 (cross-contract pause docstring) — accurately describes operational invariant.
- INFO-5 (`b.status = PENDING` reorder) — confirmed last write at line 492.
- HIGH-5 / HIGH-6 / HIGH-7 / B-CROSS-2 / B-OWNABLE-1 / Reentrancy / Ownable2Step / `MAX_SUPPLY` / `renounceRole` block / Ed25519 `pure` / `MIN_SLASH_GAS` floor / 3-way split arithmetic / pull-pattern reentrancy — all confirmed clean by the relevant team.

## 7. POL-2 §4.1 disposition decision

**Pass criterion met.** 0 unremediated CRITICAL; the 1 new HIGH (A-06) is REMEDIATED in this same sprint with a regression suite. The MEDIUM re-classification is a doc-update; the 3 INFOs are defensive cleanups bundled for a future sweep.

This consolidated document satisfies the POL-2 §4.1 re-audit requirement at the post-fix tip.

## 8. Sign-off

— Generated from 4-team self-audit (Teams A/B/C/D running Agent tool with `subagent_type=general-purpose`) targeting tip `2c93c5bb`, 2026-05-07. All four per-team findings files (`team-a-findings.md` … `team-d-findings.md`) live alongside this consolidated doc.
