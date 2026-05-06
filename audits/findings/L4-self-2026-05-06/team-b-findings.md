# Team B — Access Control Self-Audit Findings (2026-05-06)

## Summary

Re-audit of the 4 contracts after L2 remediation. Most prior findings are properly closed (immutable cross-wires, `Ownable2Step`, `totalEscrowedBalance` accumulator, per-batch snapshots for D-03/D-05). **3 new findings** identified — 1 HIGH, 2 MEDIUM. None block Gate B independently, but the HIGH does expand the post-handoff blast-radius of a compromised owner key.

## Findings

### B-01 — Owner can shrink in-flight challenge windows by sustained `pause()` (Severity: HIGH)

**Location:** `BatchSettlementRegistry.sol:413` (finalizeBatch), `:486` (challengeReceipt), `:882` (pause), `:425` (challengeWindowSecondsAtCommit snapshot).

**Description:** L2 D-05 was remediated by snapshotting `challengeWindowSecondsAtCommit` per batch so `setChallengeWindowSeconds` cannot retroactively shrink the window. However, the `Pausable` surface added in HIGH-3/D-02 reintroduces an equivalent retroactive shrink primitive via a different vector: pausing BSR halts both `challengeReceipt` AND `finalizeBatch`, but the underlying clock keeps advancing. A pause that brackets most of a batch's challenge window irrecoverably consumes that window.

**Attack scenario:**
1. Provider (colluding with compromised Foundation Safe) commits fraudulent batch at T0 with default 3-day window.
2. Honest off-chain monitors detect within hours and prepare challenge tx.
3. Compromised Safe calls `pause()` at T0 + 1h citing "incident response."
4. Pause held until T0 + 3d + 1m. During pause, `challengeReceipt` reverts `EnforcedPause`.
5. Safe calls `unpause()`. `block.timestamp - b.commitTimestamp = 3d + 1m > b.challengeWindowSecondsAtCommit = 3d` → `challengeReceipt` reverts `ChallengeWindowElapsed`; `finalizeBatch` succeeds.
6. Provider receives full settlement. Stake never slashed.

**Impact:** A compromised 2-of-3 Foundation Safe can deprive challengers of their committed dispute period for any in-flight batch. Combined with provider collusion, the slasher becomes decorative for a window-sized batch of fraudulent settlements per pause cycle.

**Recommended fix:** Track cumulative pause duration overlapping each batch's window. Maintain `uint256 totalPausedAtBatchOrigin` snapshotted per batch + global `totalPausedSeconds` accumulator updated by `_pause`/`_unpause` overrides. Effective elapsed = `(now - commitTimestamp) - (totalPausedSeconds - totalPausedAtBatchOrigin)`.

Lighter alternative: REMOVE `whenNotPaused` from `challengeReceipt` so challenges remain landable during pause. Trade-off: weaker incident-response containment vs. closing this finding.

---

### B-02 — `setSettlementLookbackWindow` retroactively flips EXPIRED challenge eligibility (Severity: MEDIUM)

**Location:** `BatchSettlementRegistry.sol:681-687` (`_handleExpired`), `:826` (`setSettlementLookbackWindow`).

**Description:** L2 D-05 snapshots `challengeWindowSecondsAtCommit` per batch. D-03 snapshots `escrowPool`/`stakeBond`/`signatureVerifier`. But `settlementLookbackWindowSeconds` — which gates the EXPIRED challenge — is read live, NOT snapshotted at commit time. Owner can mutate it mid-flight to flip EXPIRED challenge eligibility either direction.

**Attack scenarios:**
- **Shortening:** Compromised Safe shortens window. Previously-valid receipts in PENDING batches become EXPIRED-challengeable.
- **Lengthening:** Provider colluding with Safe commits stale receipts; Safe lengthens window before honest EXPIRED challenge lands.

**Impact:** Owner-trust violation. EXPIRED is non-slashing, so blast radius is bounded to per-batch receipt value, but it's a clean primitive for owner-side payment manipulation.

**Recommended fix:** Add `uint64 lookbackWindowSecondsAtCommit` to `Batch` struct; snapshot in `commitBatch`; consult per-batch field in `_handleExpired`. Mirrors existing D-05 + D-03 pattern.

---

### B-03 — `setFoundationReserveWallet` accepts zero/non-canonical address (Severity: MEDIUM)

**Location:** `StakeBond.sol:346-350`.

**Description:** The destination for `drainFoundationReserve`'s 30%-of-slashes flow has WEAKER validation than even prior B-TREASURY-1: no zero-address check, no contract-bytecode check, no canonical-Safe pin. A typo at handoff or a compromised owner can route the entire Foundation reserve stream to an arbitrary EOA.

**Attack scenario:**
1. Foundation Safe (compromised OR operator typo) calls `setFoundationReserveWallet(attacker_eoa)`.
2. Slashes accrue or already-accrued via `foundationReserveBalance > 0`.
3. `drainFoundationReserve()` → entire balance to attacker_eoa. No timelock, no second-step.

**Recommended fix:**
1. Reject `address(0)` in `setFoundationReserveWallet`.
2. Add contract-bytecode check on mainnet.
3. Apply B-TREASURY-1 canonical-Safe pin at deploy-script level.
4. Defense-in-depth: 14-day timelock on the setter.

## Vectors evaluated and cleared

| Vector | Verdict |
|--|--|
| `EscrowPool.settlementRegistry` mutability | Cleared (immutable, HIGH-6 fix solid) |
| `StakeBond.slasher` mutability | Cleared (immutable, HIGH-7 fix solid) |
| `Ownable2Step` migration | Cleared |
| `totalEscrowedBalance` accumulator | Cleared |
| Per-batch D-03 cross-wire snapshots | Cleared |
| Per-batch D-05 challenge window | Cleared at setter level (B-01 raised separately for pause) |
| CRIT-1 C-INT-01 signingMessageHash binding | Cleared |
| HIGH-2 A-02/D-01 unbond race | Cleared at the cross-wire level (A-01 raised re LIVE-vs-snapshot composition) |
| `Pausable` PAUSER scope | Cleared as designed |
| Re-entrancy via slasher in requestUnbond | Cleared |
| `Ed25519Verifier` access surface | Cleared (pure, no state) |
| `claimBounty` permissionless self-claim | Cleared |
| `MIN_SLASH_GAS` floor | Cleared |
| `SlashSwallowed` event observability | Cleared |

## Notes / Out-of-scope

- CRIT-2 (B-FTNS-1) — RESOLVED 2026-05-06 via PRSM-CR-2026-05-06-3 role migration ceremony. No longer applicable.
- HIGH-1 (A-01 RoyaltyDistributor) — out of contract scope.
- B-FTNS-1 renounceRole override (HIGH-5) — applies to FTNSTokenSimple, out of scope.

## Bottom-line

Post-remediation, the audit-bundle's access-control surface is materially tighter than the L2 baseline. The 3 new findings are second-order surfaces opened by the L2 fixes themselves: B-01 is an emergent consequence of adding `Pausable` (HIGH-3) without per-batch pause-duration accounting. B-02 is the lookback-window analogue of D-05 the original snapshot pass missed. B-03 is a B-TREASURY-1-shaped finding that wasn't applied to StakeBond's reserve-wallet sink. All three are clean fixes in the existing snapshot/canonical-pin pattern.
