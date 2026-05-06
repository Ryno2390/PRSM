# Team A — Economic Self-Audit Findings (2026-05-06)

## Summary

Five new findings against the post-remediation audit-bundle (HEAD of `main` 2026-05-06). Severity rollup: 1 HIGH, 2 MEDIUM, 1 LOW, 1 INFO. The HIGH is a partial-coverage gap in the L2 HIGH-2 remediation: `StakeBond.requestUnbond` reads the LIVE `challengeWindowSeconds` from BSR rather than the per-batch snapshot from the L2 D-05 fix. A governance change between commit and requestUnbond can re-open the original A-02/D-01 slash-evasion race without the attacker controlling timing of the change. The two MEDIUM items are an asymmetric "first-mover slashes the sibling" CONSENSUS_MISMATCH primitive and a per-receipt requester-side cherry-pick under NO_ESCROW. None of the prior 1 CRIT + 7 HIGH have regressed; signingMessageHash binding, immutable `slasher`/`settlementRegistry`, the `Pausable` surface, the per-batch D-03/D-05 snapshots, the MIN_SLASH_GAS floor, and the B-CROSS-2 totalEscrowedBalance accumulator all check out.

## Findings

### A-01 — `StakeBond.requestUnbond` floor uses LIVE challengeWindow, not per-batch snapshot (Severity: HIGH)

**Location:** `StakeBond.sol:248-262` (requestUnbond) + `BatchSettlementRegistry.sol:855-863` (setChallengeWindowSeconds)

**Description:**
The L2 HIGH-2 remediation makes `requestUnbond` clamp `unbond_eligible_at = max(now + unbondDelay, now + slasher.challengeWindowSeconds())`. Independently, the L2 D-05 remediation per-batch snapshots `challengeWindowSecondsAtCommit` so `setChallengeWindowSeconds` cannot retroactively shorten dispute periods for already-PENDING batches.

These two remediations are not composed. `requestUnbond` consults the LIVE `challengeWindowSeconds` storage slot, NOT any per-batch snapshot. If governance lowers the global window AFTER a malicious provider commits a high-window batch but BEFORE the provider initiates unbond, the unbond floor is computed against the new (shorter) window even though the provider's PENDING batch retains the old (longer) window per D-05. Provider can withdraw before the batch's still-valid challenge window elapses, and a successful later challenge falls into the same `try/catch { SlashSwallowed }` path the original A-02 finding identified.

**Attack scenario:**
1. T=0: governance has `challengeWindowSeconds = 30 days` (MAX). Provider P bonds 25,000 FTNS at PREMIUM rate (10000 bps).
2. T=0: P calls `commitBatch` with fraudulent receipts. Snapshot `challengeWindowSecondsAtCommit = 30 days`.
3. T=1 hour: governance — honest operational pressure or compromised owner — calls `setChallengeWindowSeconds(1 hour)`. Valid (≥ MIN). P's batch retains 30-day window per D-05.
4. T=1 hour + ε: P calls `requestUnbond`. The slasher returns LIVE 1 hour. With `unbondDelay = 1 day`, `effectiveEligibleAt = now + 1 day`.
5. T=1 day + 1 hour: P calls `withdraw`. Status → WITHDRAWN; full 25K FTNS returned.
6. T ∈ (1d, 30d): challenger lands `challengeReceipt`. Receipt invalidated (per-batch 30d window still open). `IStakeBond.slash` reverts `NotSlashable(WITHDRAWN)`. Caught; `SlashSwallowed` fires. Provider keeps full stake.

**Impact:**
- Attacker keeps full stake (25K FTNS PREMIUM, up to 50K CRITICAL).
- 70/30 bounty/Foundation slash split bypassed.
- Re-introduces the L2 HIGH-2 attack surface under a narrower precondition.

**Recommended fix:**
1. **(Preferred)** Maintain a per-provider `lastPendingBatchExpiry[provider]` in BSR, updated in `commitBatch`. Add an `IWithProviderExpiry` interface read in `requestUnbond`: `effectiveEligibleAt = max(localFloor, slasher.lastPendingBatchExpiry(msg.sender))`.
2. **(One-liner)** In `requestUnbond`, use `MAX_CHALLENGE_WINDOW_SECONDS` (30 days) as the floor regardless of LIVE value.

---

### A-02 — CONSENSUS_MISMATCH is symmetric: first mover slashes the sibling (Severity: MEDIUM)

**Location:** `BatchSettlementRegistry.sol:735-781` (`_handleConsensusMismatch`)

**Description:**
The contract has no on-chain notion of which output is "majority" — given any two batches in the same consensus group with disagreeing outputs, EITHER provider can call challengeReceipt against the OTHER and slash them. First mover wins.

**Attack scenario:**
Honest H and attacker A both dispatched to the same shard under k-of-n. A computes deliberately-wrong output, immediately calls `challengeReceipt(H_batch_id, ..., majorityProof = my_leaf)`. All checks pass; H slashed; A collects 70% bounty.

**Impact:**
Adversarial group member can slash honest sibling at 70% bounty rate, asymmetric downside. Disincentivizes participation in consensus dispatch.

**Recommended fix:**
1. **Off-chain quorum, on-chain proof:** auxData carries `k-1` other-provider leaves. Challenge succeeds only if `k-1` corroborating providers agree.
2. **Pre-registered consensus group manifest** with strict majority requirement.
3. **Stop-gap:** restrict CONSENSUS_MISMATCH to `challenger == requester`.

---

### A-03 — Requester can cherry-pick per-receipt NO_ESCROW invalidation to underpay batches (Severity: MEDIUM)

**Location:** `BatchSettlementRegistry.sol:670-675` (`_handleNoEscrow`) + 531-532

**Description:**
NO_ESCROW is requester-only with no proof. challengeReceipt is per-receipt. Requester can pick exactly which N out of M leaves to deny, since there's no on-chain commitment that "if I deny one I deny all."

**Attack scenario:**
Requester R deposits, P legitimately serves 10 shards. R challenges 5 with NO_ESCROW. Half-payment.

**Recommended fix:**
1. Make NO_ESCROW batch-level, not receipt-level.
2. Pre-commit authorized-receipt-set at deposit time (`bytes32 authorizedReceiptsRoot`).

---

### A-04 — Donations to StakeBond / EscrowPool strand FTNS in contract; no recovery surface (Severity: LOW)

Anyone can `ftns.transfer(stakeBond, X)` directly. Stranded forever. Not exploitable, but a fund-loss footgun.

**Recommended fix:** Add owner-gated `recoverStranded(address to)`.

---

### A-05 — `requestUnbond` slasher floor silently degrades when slasher misconfigured (Severity: INFO)

Constructor permits `slasher = address(0)` for tests. If production deploy ships with `address(0)`, A-02-class slash-evasion is open across all batches forever, with no event signaling.

**Recommended fix:** Constructor revert on `address(0)` for production builds, OR emit `SlasherFloorBypass` event when fallback hits.

## Notes / Out-of-scope

**Vectors evaluated and cleared:**
- Reentrancy on settle/withdraw/claim/slash — all nonReentrant + CEI ✓
- Donation/inflation on EscrowPool — B-CROSS-2 fix solid ✓
- MIN_SLASH_GAS floor (150K) — sufficient ✓
- Front-running settle/finalize — bound at commit ✓
- Self-payment / wash trade — negative EV ✓
- Self-slash via challenger == provider — correctly routes 100% to Foundation ✓

**Cross-team adjacencies:**
- A-02 (CONSENSUS_MISMATCH) intersects Team B's authorization-model angle if option 3 (challenger == requester) is taken.
- A-01 (lastPendingBatchExpiry counter) intersects Team D's state-composition angle.
