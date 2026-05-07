# Team A — Economic Self-Audit Findings (2026-05-07 re-run)

## Summary

Re-audit of the post-fix tip (`2c93c5bb`) covering the same surfaces as the 2026-05-06 round plus the four newly-in-scope contracts (FTNSTokenSimple, RoyaltyDistributor, EmissionController, CompensationDistributor). Severity rollup: **1 HIGH (new), 1 MEDIUM (new, pre-existing-but-newly-relevant), 0 LOW, 1 INFO**. The 0-CRITICAL pass criterion of POL-2 §4.1 is met.

The new HIGH (A-06) is a composition gap between the HIGH-1 remediation (`lastPendingBatchExpiry` per-provider tracker) and the HIGH-2 remediation (`totalPausedSeconds` accumulator). HIGH-1 stores wall-clock expiry; HIGH-2 makes the effective challenge expiry pause-extended. The unbond floor read by `StakeBond.requestUnbond` therefore can be earlier than the effective challenge expiry under the canonical A-01 trigger condition (governance `setChallengeWindowSeconds` reduction) when ANY pause has occurred between commit and "wall-clock window elapsed" — re-opening the original slash-evasion race for the `pausedSinceCommit`-shaped slice.

All other prior-round economic findings remain in their accepted/remediated dispositions and have not regressed:

| Prior finding | Status this round |
|---|---|
| A-01 (HIGH-1) — `lastPendingBatchExpiry` + `ISlasherWithProviderExpiry` | Remediated for the no-pause case. Pause-composition gap raised as A-06 below. |
| A-02 (MED-1) — CONSENSUS_MISMATCH symmetric first-mover | Still accepted-with-rationale. No regression. |
| A-03 (MED-2) — NO_ESCROW per-receipt cherry-pick | Still accepted-with-rationale. No regression. |
| A-04 (LOW-1) — donations strand FTNS | Still pending. Now also applies to RoyaltyDistributor (same pattern). |
| A-05 (INFO-1) — slasher misconfigured silent degradation | Subsumed by MED-6 constructor zero-reject. ✓ Cleared. |
| HIGH-2 / B-01 — pause-as-window-consumer | Remediated for the BSR-internal challenge-window math. Composition with HIGH-1 raised as A-06. |
| MED-3 / B-02 — `lookbackWindowSecondsAtCommit` snapshot | Remediated correctly. ✓ |
| MED-4 / B-03 — `WalletNotContract` + zero check on `setFoundationReserveWallet` | Remediated correctly. ✓ |
| LOW-3 / D-04 — `whenNotPaused` on `setFoundationReserveWallet` + `drainFoundationReserve` | Remediated correctly. ✓ |
| MED-7 / D-03 — `SetterTargetNotContract` on the 3 BSR setters | Remediated correctly. **Gap noted as A-07 below:** the 7-day timelock the policy doc references is NOT implemented; only the EOA/non-contract guard is. |
| INFO-5 / D-06 — `b.status = PENDING` reorder to last write | Remediated correctly. ✓ Verified line 492. |

## Findings

### A-06 — `lastPendingBatchExpiry` is pause-blind; HIGH-1 + HIGH-2 fixes do not compose (Severity: HIGH)

**Location:**
- `contracts/contracts/BatchSettlementRegistry.sol:499-502` (commitBatch tracker update)
- `contracts/contracts/BatchSettlementRegistry.sol:1084-1094` (`_effectiveElapsed`)
- `contracts/contracts/StakeBond.sol:330-339` (requestUnbond reader)

**Description:**
The HIGH-1 remediation stores per-provider expiry as the wall-clock value `commitTimestamp + challengeWindowSecondsAtCommit` (BSR line 499). The HIGH-2 remediation makes the effective challenge-window expiry pause-aware: `_effectiveElapsed(b) = wall - pausedSinceCommit`, so the actual challenge can land any time `_effectiveElapsed < windowAtCommit`, i.e. up to wall-clock time `commitTimestamp + windowAtCommit + pausedSinceCommit`.

The unbond floor read by `StakeBond.requestUnbond` clamps against the wall-clock expiry but NOT against `pausedSinceCommit`. When the LIVE `challengeWindowSeconds()` has been reduced via `setChallengeWindowSeconds` (the canonical A-01 trigger), the slasherFloor (`now + LIVE_window`) shrinks accordingly. The localFloor (`now + unbondDelaySeconds`, min 1 day) is the only remaining clamp. Provider can therefore initiate unbond at the wall-clock-expiry boundary, withdraw `unbondDelaySeconds` later, and a successful challenge landed in the `pausedSinceCommit`-shaped tail of the effective challenge window hits `StakeStatus.WITHDRAWN` and is silently swallowed by the BSR `try/catch { emit SlashSwallowed }` path.

**Attack scenario:**
1. T=0: governance has `challengeWindowSeconds = 30 days`, `unbondDelaySeconds = 1 day`. Provider P bonds 25,000 FTNS at PREMIUM (10000 bps).
2. T=0: P commits a fraudulent batch. `b.challengeWindowSecondsAtCommit = 30 days`, `b.totalPausedAtBatchOrigin = 0`. `lastPendingBatchExpiry(P) = 30 days`.
3. T=10d → T=15d: BSR is paused (legitimate incident response, e.g. a different exploit triage). On unpause, `totalPausedSeconds += 5 days`.
4. T=20d: governance reduces `challengeWindowSeconds` to `1 hour` (e.g., honest operational pressure). P's batch retains its 30-day pinned window per D-05.
5. T=30d (wall-clock window boundary): P calls `requestUnbond`.
   - `localFloor = 30d + 1d = 31d`.
   - `slasherFloor = 30d + 1h ≈ 30d 1h` (live window, post-reduction).
   - `lastPendingBatchExpiry(P) = 30d` — NOT strictly greater than 31d, so the HIGH-1 clamp does not bind.
   - `effectiveEligibleAt = 31d`.
6. T=31d: P calls `withdraw`. Status → WITHDRAWN; full 25K FTNS returned.
7. T=33d: challenger lands `challengeReceipt`. `_effectiveElapsed(b) = (33d - 0) - (5d - 0) = 28d < 30d`, so within window. Receipt invalidated.
8. `IStakeBond.slash` reverts `NotSlashable(WITHDRAWN)`. Caught; `SlashSwallowed` fires. Provider keeps full stake.

**Impact:**
- Same economic outcome as the original A-01: attacker keeps full stake (25K FTNS PREMIUM, up to 50K CRITICAL).
- 70/30 bounty/Foundation slash split bypassed.
- Re-opens the L2 HIGH-2 / L4 HIGH-1 attack surface for the `pausedSinceCommit`-shaped tail of the effective challenge window.
- Trigger requires BOTH (a) `setChallengeWindowSeconds` reduction (the canonical A-01 trigger) AND (b) any non-zero pause duration during the batch's window. Pauses are part of normal operational defense (PRSM-EXPLOIT-PLAYBOOK §11) — the precondition is realistic, not pathological.

**Note on the `slasherFloor` LIVE-read:** if governance does NOT reduce the live window, `slasherFloor = now + 30d` dominates and the provider's unbond is gated 30 days into the future regardless of pause history. The exploit therefore inherits the original A-01 governance-reduction precondition. But the prior round explicitly remediated A-01 to close this exact precondition; under pause, the remediation no longer holds.

**Recommended fix (preferred):**
Track per-provider expiry as a `(maxPendingDeadline, totalPausedAtAccrual)` pair. In `commitBatch`, write both. The reader (`requestUnbond`) computes the pause-adjusted floor:

```solidity
uint64 expiry = bsr.lastPendingBatchExpiry(msg.sender);
uint64 pausedAtAccrual = bsr.lastPendingBatchPausedAtAccrual(msg.sender);
uint256 currentPaused = bsr.totalPausedSeconds();
uint256 pauseDelta = currentPaused > pausedAtAccrual ? currentPaused - pausedAtAccrual : 0;
uint256 pauseAdjustedExpiry = uint256(expiry) + pauseDelta;
if (pauseAdjustedExpiry > effectiveEligibleAt) effectiveEligibleAt = pauseAdjustedExpiry;
```

This mirrors the `_effectiveElapsed` arithmetic and guarantees the unbond floor tracks the effective challenge expiry across any pause sequence.

**One-line alternative:** as before, use `MAX_CHALLENGE_WINDOW_SECONDS` (30 days) as a permanent floor regardless of LIVE value. Penalises honest providers under low-window regimes but removes the LIVE-vs-snapshot AND pause-vs-wallclock mismatches in one stroke.

**Defense-in-depth (already in place):** Forta alert on `SlashSwallowed{reason ∈ {DOUBLE_SPEND, INVALID_SIGNATURE, CONSENSUS_MISMATCH}}` paired with provider `StakeStatus == WITHDRAWN` is the canonical fingerprint. Operationally, the alert distinguishes A-06 firing from benign double-challenges.

---

### A-07 — MED-7 setter timelock referenced by policy is NOT implemented in code (Severity: MEDIUM)

**Location:**
- `contracts/contracts/BatchSettlementRegistry.sol:923-967` (`setEscrowPool` / `setStakeBond` / `setSignatureVerifier`)

**Description:**
The 2026-05-06 disposition note for MED-7 in `consolidated.md` §3 explicitly recommended "bonus 7-day timelock" alongside the `code.length > 0` check. The remediation shipped only the `code.length > 0` check; no timelock surface (queue / commit / execute) exists in the contract today. The `D-03 per-batch snapshot` correctly protects already-PENDING batches from mid-flight rotation — but new batches commit against the live setter values, which can be rotated atomically by the owner.

**Attack scenario:**
Compromised Foundation Safe rotates `escrowPool` to an attacker-controlled stub contract. Within the next provider commit, the rotated pool is snapshotted into the batch's `escrowPoolAtCommit`. On finalize, the attacker stub receives the `settleFromRequester` call, draining requester escrow into the attacker's address.

**Impact:**
Captures one batch's settlement flow per rotation. Compromised Safe is in scope per HIGH-6/HIGH-7 precedent. Mitigated by Foundation Safe's 2-of-3 model (operational, not on-chain) and by the 14-day public review window POL-2 §4 imposes on mainnet deploys.

**Recommended fix:**
Implement the documented 7-day timelock OR explicitly downgrade the MED-7 disposition from "remediated" to "remediated-partial; 7-day timelock deferred to v2 deploy." The latter is the lower-cost path since the per-batch snapshot already captures most of the in-flight risk; the timelock is defense against the live-rotation slice only.

This is reported separately because the consolidated.md remediation row currently reads `REMEDIATED` (full) and the deploy gate at POL-2 §4 framework should reflect the actual code shape, not the recommendation.

---

### A-08 — RoyaltyDistributor donation strand mirrors LOW-1; aggregate fund-loss footgun across pull-payment contracts (Severity: INFO)

**Location:**
- `contracts/contracts/RoyaltyDistributor.sol:51-168`

**Description:**
RoyaltyDistributor uses the same pull-payment pattern as StakeBond (claimable[] mapping + claim() pull). The same A-04 / LOW-1 donation footgun applies: anyone can `ftns.transfer(distributor, X)` directly. Donations are not credited to any `claimable[]` slot, so they cannot be claimed and sit forever. RoyaltyDistributor was not in scope for the prior round; the LOW-1 disposition (pending owner-gated `recoverStranded`) does not currently extend to it, and there is no owner role on RoyaltyDistributor at all (no `Ownable*`).

**Impact:**
Same shape as A-04 — not exploitable, footgun. Magnitude is bounded by donation rate. Notably, RoyaltyDistributor has no owner, so even if the recovery surface were added later, there is no trusted recipient to gate it to.

**Recommended fix:**
Either (a) add an Ownable surface and `recoverStranded(address to)` mirroring the proposed StakeBond fix, or (b) accept the documented limitation; the donation pattern is not a privilege-escalation primitive.

## Vectors evaluated and cleared

| Vector | Verdict |
|--|--|
| A-01 + HIGH-1 fix correctness in the no-pause case (single batch, no governance change) | **Cleared.** The `lastPendingBatchExpiry` write-on-commit + `requestUnbond` clamp correctly handles the prior A-01 attack scenario absent pause. |
| HIGH-1 monotonicity (`if (newExpiry > lastPendingBatchExpiry)` only increments) | **Cleared.** Tracker is monotonic. Stale future values from finalized batches are suboptimal (over-conservative for honest providers) but not exploitable. |
| HIGH-1 overflow (uint64 narrowing of `block.timestamp + windowAtCommit`) | **Cleared.** ~580 billion years of headroom; not a near-term concern. |
| HIGH-2 `_pause` / `_unpause` symmetry — accumulator only updates on `_unpause` | **Cleared.** Per docstring (BSR line 285-298) and verified at lines 1100-1115. The currently-active pause is irrelevant to gated functions because they're `whenNotPaused`; views that read `_effectiveElapsed` may momentarily under-report during an active pause but callers can't act until unpause. |
| HIGH-2 accumulator overflow (`totalPausedSeconds` is uint256; per-batch snapshot is uint64) | **Cleared.** Per-batch uint64 narrowing is safe for centuries of cumulative pause time. |
| HIGH-2 commit-time snapshot vs effective-elapsed math (`wall - pausedSinceCommit`) | **Cleared at the within-BSR level.** Composition gap with HIGH-1 raised as A-06 above. |
| MED-3 `lookbackWindowSecondsAtCommit` × EXPIRED challenge × pause | **Cleared.** EXPIRED is wall-clock receipt-age (correct semantics — pause shouldn't extend receipt validity). Effective challenge window extension via pause means MORE receipts are challengeable as EXPIRED, which is in the challengers' favor (correct direction). |
| MED-4 `WalletNotContract` + `whenNotPaused` on `setFoundationReserveWallet` | **Cleared.** StakeBond:445-451. Zero rejected, EOA rejected, paused rejected. |
| LOW-3 `whenNotPaused` on `drainFoundationReserve` | **Cleared.** StakeBond:571 has `whenNotPaused`. Compromised owner must unpause first (on-chain breadcrumb) before draining. |
| MED-6 constructor zero-reject (StakeBond + EscrowPool) | **Cleared.** Both reject `address(0)` for slasher / initialRegistry. Subsumes A-05 / INFO-1. |
| MED-7 `SetterTargetNotContract` on 3 BSR setters | **Partially cleared** — `code.length > 0` check shipped; 7-day timelock NOT shipped. Raised separately as A-07. |
| INFO-5 `b.status = PENDING` ordering | **Cleared.** Line 492 is the LAST write in commitBatch. Verified by re-reading the function body. |
| Reentrancy (StakeBond, EscrowPool, RoyaltyDistributor) | **Cleared.** All state-mutating external functions carry `nonReentrant`; CEI ordering correct in claim/withdraw/settle/slash. |
| EscrowPool `totalEscrowedBalance` accumulator vs setFtnsToken | **Cleared.** B-CROSS-2 fix solid. |
| RoyaltyDistributor 3-way split arithmetic | **Cleared.** `creatorAmt + networkAmt <= gross` enforced; serving-node gets remainder; no underflow possible. |
| RoyaltyDistributor pull pattern reentrancy across distribute + claim | **Cleared.** Both nonReentrant on shared guard; no cross-function reentrancy possible. |
| FTNSTokenSimple `MAX_SUPPLY` enforcement on `mintReward` | **Cleared.** Line 71 caps at MAX_SUPPLY. |
| FTNSTokenSimple `renounceRole(DEFAULT_ADMIN_ROLE)` block | **Cleared.** Line 142-150 reverts with explicit message. |
| EmissionController immutable curve + halving | **Cleared.** `>>` halving + saturate-to-zero at e>=256. mintCap immutable. |
| EmissionController pause-and-burst risk | **Accepted-with-rationale per design §8.2.** lastMintTimestamp is not rewound on pause; a long quiet period (pause or natural) followed by `pullAndDistribute` mints `rate × elapsed` at once. Documented; monitoring alert on call-gap > 7 days is the operational defense. |
| CompensationDistributor 90-day weight schedule enforcement | **Cleared.** `scheduledAt >= now + 90 days` enforced at `updateWeights`. Replaceable by re-call. |
| CompensationDistributor `setPoolAddresses` rotation risk | **Same shape as MED-7.** No timelock; compromised owner can rotate pools and trigger `pullAndDistribute` to capture the next emission slice. Mitigated by Safe 2-of-3 + monitoring; not raised as a separate finding because the disposition matches the existing accepted MED-7 gap. |
| CompensationDistributor bps-sum validation | **Cleared.** `_validateWeights` requires sum == 10000; dust accrues to grant pool. |
| Front-running settle/finalize | **Cleared.** Bound at commit per D-03 / D-05 snapshots. |
| Self-payment / self-slash routing (challenger == provider → 100% Foundation) | **Cleared.** StakeBond:531-538. |
| MIN_SLASH_GAS gas-floor guard | **Cleared.** 150K floor + revert before try/catch entry on insufficient gas. |
| Cross-chain replay / signature replay | **Cleared.** Single-chain BSR; signing message hash bound at leaf level (C-INT-01 fix). |

## Cross-team adjacencies

- **A-06 (HIGH)** intersects Team D's state-composition angle. Recommend cross-confirmation in Team D's findings — same composition-gap shape as the original D-01 finding.
- **A-07 (MED)** intersects Team B's access-control angle. Setter rotation risk surfaces in both economic (compromised-Safe drain) and access-control (privilege boundary) framings.
- **A-08 (INFO)** is a benign extension of LOW-1; no cross-team coordination needed.

## POL-2 §4.1 disposition

**Pass criterion:** "0 unremediated CRITICAL findings; HIGH findings either remediated or accepted-with-recorded-rationale."

This re-run yields:
- **0 CRITICAL** ✓
- **1 NEW HIGH (A-06)** — requires founder disposition before audit-bundle mainnet deploy. The `lastPendingBatchExpiry` data structure must either be extended to `(expiry, pausedAtAccrual)` per provider OR the unbond floor must adopt the `MAX_CHALLENGE_WINDOW_SECONDS` permanent-floor alternative. Estimated remediation cost: ~1 day (mirrors the HIGH-1 patch shape; one extra storage slot per provider; ~2 lines of arithmetic in StakeBond.requestUnbond).
- **1 NEW MEDIUM (A-07)** — disposition recommendation: re-classify in `consolidated.md` from REMEDIATED to REMEDIATED-PARTIAL with timelock deferred. No code change required if accepted.
- **1 NEW INFO (A-08)** — defensive cleanup; bundle with the LOW-1 / A-04 fix when convenient.

The pass criterion can be met by remediating A-06 OR by accepting it with rationale (e.g., "pause-extension exploit requires both compromised-or-mistaken governance reduction AND a coincident pause window; combined precondition is operationally rare; Forta detects the WITHDRAWN+SlashSwallowed signature; defer fix to v2 deploy"). The founder's call.

— Team A (Economic), L4 self-audit re-run, 2026-05-07.
