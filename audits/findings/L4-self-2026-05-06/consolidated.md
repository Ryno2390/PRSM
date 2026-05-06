# L4 Self-Audit (2026-05-06) — Consolidated Findings

**Authority:** PRSM-POL-2 §4.1 (Resource-Constrained Audit Strategy), ratified by PRSM-CR-2026-05-06-5. Agent-teams self-audit is the substitute external-audit modality for L3 + L4 pre-mainnet review.

**Date:** 2026-05-06
**Pinned commit:** `8a8cab3c` (HEAD of `main` after today's policy + V2 deploy + role migration commits)
**Teams:** A (economic) · B (access control) · C (signature/crypto) · D (state composition). All 4 teams ran independently in parallel via the `Agent` tool.
**Source documents:** `team-a-findings.md` · `team-b-findings.md` · `team-c-findings.md` · `team-d-findings.md`

**Scope (4 contracts, ~1,708 LOC):**
- `BatchSettlementRegistry.sol` (919 LOC)
- `EscrowPool.sol` (245 LOC)
- `StakeBond.sol` (485 LOC)
- `Ed25519Verifier.sol` (59 LOC)

## 1. Severity rollup (deduplicated, post-cross-team merge)

| Severity | Count | POL-2 §4.1 disposition |
|----------|-------|------------------------|
| **Critical** | **0** | n/a |
| **High** | **2** | Must remediate OR accept-with-rationale before audit-bundle mainnet deploy |
| **Medium** | **5** | Should remediate before deploy |
| **Low** | **3** | Best-practice cleanup |
| **Informational** | **5** | Code/doc quality |
| **TOTAL** | **15** | |

**Pass criterion (POL-2 §4.1):** "0 unremediated CRITICAL findings; HIGH findings either remediated or accepted-with-recorded-rationale." Today's run yields **0 CRITICAL** ✓ and **2 HIGH** that require disposition.

**Cross-team convergence (significance signal):** Teams A and D **independently identified the same HIGH severity finding** (A-01 ≡ D-01) — the `requestUnbond` / per-batch-snapshot composition gap. Two adversarial reviewers landing on the same issue from different angles strongly indicates this is a real and exploitable composition bug, not a false positive.

## 2. HIGH findings — must disposition before audit-bundle mainnet deploy

### HIGH-1 (A-01 ≡ D-01) — `requestUnbond` floor uses LIVE challengeWindow, not per-batch snapshot

**Location:** `StakeBond.sol:248-262` (requestUnbond) + `BatchSettlementRegistry.sol:381` (challengeWindowSecondsAtCommit snapshot) + `:497` (challengeReceipt eligibility gate)

**Convergence:** Found independently by Team A (economic angle — slash-evasion bypass) and Team D (state composition angle — incomplete fix interaction).

**Description:** Two prior remediations interact incorrectly:
- L2 HIGH-2 fix: `requestUnbond` clamps `unbond_eligible_at = max(now + unbondDelay, now + slasher.challengeWindowSeconds())`.
- L2 D-05 fix: each `Batch` snapshots `challengeWindowSecondsAtCommit` so `setChallengeWindowSeconds` cannot retroactively shorten in-flight batches.

The HIGH-2 reads the **live** global; D-05 makes already-PENDING batches use the **snapshot**. If governance lowers the live window AFTER a malicious provider commits a high-window batch but BEFORE provider initiates unbond, the unbond floor uses the new (shorter) window while the batch still has the old (longer) challenge window. Provider can withdraw before the batch's still-valid challenge window elapses; the same `try/catch { SlashSwallowed }` path the original A-02 finding identified silently swallows the failed slash.

**Trigger condition:** Single subsequent `setChallengeWindowSeconds` reduction. No attacker control of governance required — works against any provider whose batch was committed under a higher window than the current live value.

**Impact:** Reproduces the original L2 HIGH-2 economic outcome (provider keeps full stake despite successful malicious-batch challenge) under a precondition that the D-05 fix explicitly markets as "safe for in-flight batches."

**Recommended fix (preferred):** BSR maintains `mapping(address provider => uint64 maxPendingDeadline)` updated on `commitBatch` to `max(existing, commitTimestamp + challengeWindowSecondsAtCommit)`. `requestUnbond` clamps `unbond_eligible_at >= bsr.maxPendingDeadline(msg.sender)`.

**One-line alternative:** In `requestUnbond`, use `MAX_CHALLENGE_WINDOW_SECONDS` (30 days) as the floor regardless of LIVE value. Penalises honest providers under low-window regimes but removes the LIVE-vs-snapshot mismatch entirely.

**Defense-in-depth (already partially in place):** Forta alert on `SlashSwallowed{reason ∈ {DOUBLE_SPEND, INVALID_SIGNATURE, CONSENSUS_MISMATCH}}` paired with provider `StakeStatus == WITHDRAWN` is the canonical fingerprint of HIGH-1 firing.

---

### HIGH-2 (B-01) — Owner can shrink in-flight challenge windows by sustained `pause()`

**Location:** `BatchSettlementRegistry.sol:413` (finalizeBatch), `:486` (challengeReceipt), `:882` (pause), `:425` (challengeWindowSecondsAtCommit snapshot).

**Description:** L2 D-05 was remediated by snapshotting `challengeWindowSecondsAtCommit` per batch so `setChallengeWindowSeconds` cannot retroactively shrink the window. However, the `Pausable` surface added in HIGH-3/D-02 reintroduces an equivalent retroactive shrink primitive via a different vector: pausing BSR halts both `challengeReceipt` AND `finalizeBatch`, but the underlying clock (`block.timestamp - b.commitTimestamp`) keeps advancing. A pause that brackets most of a batch's challenge window irrecoverably consumes that window.

**Attack scenario:** Provider colludes with compromised Foundation Safe. Safe pauses BSR shortly after fraudulent batch commit, holds pause through the entire challenge window, unpauses just after window expiry. `challengeReceipt` reverts `ChallengeWindowElapsed`; `finalizeBatch` succeeds.

**Trigger condition:** Compromised 2-of-3 Foundation Safe. Same blast-radius model as HIGH-6/HIGH-7 (which were remediated specifically because compromised owner was within scope).

**Impact:** Slasher becomes decorative for a window-sized batch of fraudulent settlements per pause cycle. Affects every PENDING batch simultaneously since pause is global.

**Recommended fix:** Track cumulative pause duration overlapping each batch's window. Maintain `uint256 totalPausedAtBatchOrigin` snapshotted per batch + global `totalPausedSeconds` accumulator updated by `_pause`/`_unpause` overrides. Effective elapsed = `(now - commitTimestamp) - (totalPausedSeconds - totalPausedAtBatchOrigin)`. Mirrors the D-05 snapshot pattern.

**Lighter alternative:** Remove `whenNotPaused` from `challengeReceipt` so challenges remain landable during pause. Trade-off: weaker incident-response containment vs. closing this finding.

## 3. MEDIUM findings — should remediate before deploy

### MED-1 (A-02) — CONSENSUS_MISMATCH symmetric: first mover slashes the sibling
`BatchSettlementRegistry.sol:735-781`. No on-chain "majority" notion — given any two batches in the same consensus group with disagreeing outputs, EITHER provider can slash the other. First mover wins. **Fix:** require k≥3 corroborating leaves (off-chain quorum proof) OR pre-registered consensus group manifest with strict-majority requirement OR (stop-gap) restrict CONSENSUS_MISMATCH to `challenger == requester`.

### MED-2 (A-03) — Requester can cherry-pick per-receipt NO_ESCROW invalidation
`BatchSettlementRegistry.sol:670-675` + `:531-532`. NO_ESCROW is per-receipt with no proof; requester can pick exactly which N of M leaves to deny inside the challenge window. **Fix:** make NO_ESCROW batch-level (all-or-nothing), OR pre-commit `authorizedReceiptsRoot` at deposit time.

### MED-3 (B-02) — `setSettlementLookbackWindow` retroactively flips EXPIRED challenge eligibility
`BatchSettlementRegistry.sol:681-687, :826`. Lookback window is read live, not snapshotted at commit time. Owner can mutate it to flip EXPIRED challenge outcomes either direction. Same shape as D-05 but for a different parameter. **Fix:** add `uint64 lookbackWindowSecondsAtCommit` to `Batch` struct; snapshot in `commitBatch`; consult per-batch field in `_handleExpired`.

### MED-4 (B-03) — `setFoundationReserveWallet` accepts zero/non-canonical address
`StakeBond.sol:346-350`. The destination for `drainFoundationReserve`'s 30%-of-slashes flow has no zero-address check, no contract-bytecode check, no canonical-Safe pin. **Fix:** reject `address(0)`, add `code.length > 0` check, apply canonical-Safe pin at deploy-script level. Defense-in-depth: 14-day timelock on the setter.

### MED-5 (C-01) — `signing_message_hash` provider-chosen, no on-chain consistency to other leaf fields
`BatchSettlementRegistry.sol:55-70` (struct) + `:636-661` (`_handleInvalidSignature`). The C-INT-01 fix binds signature to message hash, but the message hash is not bound to the leaf content. Provider can sign arbitrary `m`, fabricate other leaf fields. INVALID_SIGNATURE challenges silently fail to catch receipt-content forgery. **Fix (recommended):** update leaf docstring to clarify that `signingMessageHash` is provider-supplied and only proves "valid Ed25519 signature over SOME 32-byte preimage." Document NO_ESCROW + CONSENSUS_MISMATCH as the actual primitives that catch receipt-content forgery. (Full on-chain binding via storing `jobId` string is ~3-5 days + parity rebuild and not recommended for this pass.)

### MED-6 (D-02) — Constructor accepts `address(0)` for immutable slasher / settlementRegistry
`EscrowPool.sol:90-102` + `StakeBond.sol:160-178`. With no setter post-HIGH-6/HIGH-7, an `address(0)` deploy is permanently bricked. Documentation-only ban is the same shape as the rejected B-CROSS-2 / D-07 docstring weakness. **Fix:** constructor revert on `address(0)` for production builds (gate behind `bool allowZeroForTesting` or unconditionally). Deploy verifier should also assert non-zero.

### MED-7 (D-03) — `setEscrowPool` / `setStakeBond` / `setSignatureVerifier` accept any address
`BatchSettlementRegistry.sol:792-820`. The D-03 fix snapshots these per-batch, protecting in-flight batches. But setters still accept arbitrary addresses including EOAs and wrong-interface contracts. Compromised owner who waits one snapshot cycle re-acquires all in-flight-mutation primitives the D-03 fix was supposed to remove. **Fix:** require `code.length > 0`, ERC165 `supportsInterface` (or bespoke sentinel), bonus 7-day timelock.

## 4. LOW + INFO findings — defensive cleanup

### LOW-1 (A-04) — Donations to StakeBond/EscrowPool strand FTNS in contract
Anyone can `ftns.transfer(stakeBond, X)` directly; sits forever. Not exploitable; fund-loss footgun. **Fix:** owner-gated `recoverStranded(address to)` computing `balance(this) - accountedSum`.

### LOW-2 (C-02) — Receipt signing payload lacks domain-prefix string
`prsm/compute/shard_receipt.py:26-42`. Cross-protocol replay primitive if Ed25519 provider keys reused across PRSM subsystems. Identical pattern to C-INT-07 HandoffToken finding. **Fix:** prefix payload with `b"PRSM-Receipt-v1\n"` before keccak256. Coordinate with C-INT-07 fix.

### LOW-3 (D-04) — `drainFoundationReserve` not gated by `whenNotPaused`
`StakeBond.sol:470-484`. Compromised owner during pause can `setFoundationReserveWallet(attacker)` → `drainFoundationReserve()`. Pause does not protect this value pool, violating Stated Invariant #3. **Fix:** add `whenNotPaused` to `drainFoundationReserve` and `setFoundationReserveWallet`.

### INFO-1 (A-05) — `requestUnbond` slasher floor silent degradation when slasher misconfigured
Constructor permits `address(0)` for tests. Production `address(0)` deploy opens A-02-class evasion across all batches with no signaling event. **Fix:** constructor revert OR emit `SlasherFloorBypass` event.

### INFO-2 (C-03) — `signingMessageHash` 32-byte unconstrained
`bytes32` accepts any value. Document in `audits/findings/L3-crypto/caller-assumptions.md`.

### INFO-3 (C-04) — Stale `signingMessageHash: ZeroHash` default in PoC test factory
`contracts/test/audit-team-c/C-INT-01-invalid-signature-forgery.test.js:43-57`. Future test author copying could land on degenerate input. Trivial fix.

### INFO-4 (D-05) — Cross-contract pause coordination is implicit
StakeBond-paused-while-BSR-running coherence concern. No present-day exploit. Document operational invariant.

### INFO-5 (D-06) — `commitBatch` storage-pointer assignment ordering
`b.status = PENDING` set BEFORE snapshot writes. No external call between, so no exploit. Future fragility. **Fix:** reorder `b.status = PENDING` to LAST write.

## 5. Vectors evaluated and cleared

Both teams independently confirmed the following are PROPERLY remediated and not regressed:

| Prior finding | Verdict |
|--|--|
| L2 CRIT-1 (C-INT-01) — adversarial slashing via unbound signingMessage | **Cleared.** All challenge paths consume `leaf.signingMessageHash` directly; auxData no longer carries challenger-supplied message; regression test passes. |
| L2 CRIT-2 (B-FTNS-1) — FTNS DEFAULT_ADMIN_ROLE on hot key | **RESOLVED 2026-05-06** via PRSM-CR-2026-05-06-3 role migration ceremony. No longer applicable. |
| L2 HIGH-2 (A-02/D-01) — slash race / unbondDelay >= challengeWindow | **Cleared at the cross-wire level.** HIGH-1 above raises the LIVE-vs-snapshot composition gap as a separate finding. |
| L2 HIGH-3 (D-02) — missing OZ Pausable | **Cleared as designed.** HIGH-2 above raises pause-as-window-consumer as a separate emergent surface. |
| L2 HIGH-5 (B-RENOUNCE-1) — FTNS renounceRole override | **Cleared on FTNSTokenSimple.** Out of this audit's scope. |
| L2 HIGH-6 (B-CROSS-1) — EscrowPool.settlementRegistry mutable | **Cleared.** Now `address public immutable`. |
| L2 HIGH-7 (B-CROSS-3) — StakeBond.slasher mutable | **Cleared.** Now `address public immutable`. |
| L2 HIGH-1 (A-01) — RoyaltyDistributor split divergence | Out of this audit's contract scope. |

Other surfaces evaluated and cleared (no findings):
- Reentrancy (CEI + nonReentrant); storage layout / inheritance; ERC-20 return values; loop bounds; `block.timestamp` manipulation; reorg sensitivity; self-destruct on dependencies; Ed25519Verifier wrapper input handling; `MIN_SLASH_GAS` floor accuracy (150K vs ~80K actual cost); `claimBounty` permissionless self-claim; `Ownable2Step` migration; `totalEscrowedBalance` accumulator; per-batch D-03 cross-wire snapshots; per-batch D-05 challenge window (setter-level only); `ISignatureVerifier.verify` `pure` correctness; cross-chain replay (BSR is single-chain); `claimBounty` reentrancy.

## 6. POL-2 §4.1 disposition decision matrix

Per the policy, the founder must decide for each HIGH: **remediate** OR **accept with recorded rationale**. Recommendations:

| Finding | Recommended disposition | Estimated engineering cost | Rationale |
|---|---|---|---|
| **HIGH-1** | **Remediate (preferred fix)** | ~1-2 days | Cross-team convergence (A+D) signals real bug. Fix is structural and one-time. Forta defense-in-depth helps but isn't a substitute. |
| **HIGH-2** | **Remediate (full fix preferred)** | ~1 day | Compromised-Safe is in scope per HIGH-6/HIGH-7 precedent. Full fix is mechanically straightforward (mirror D-05 pattern for pause-time accumulator). |
| MED-3 | Remediate | ~30 min | One struct field + setter capture. Mirrors D-05 pattern. |
| MED-4 | Remediate | ~1 hr | Three-line setter validation + canonical pin. |
| MED-6 | Remediate | ~10 min | Constructor require non-zero. Cheap defense against permanent brick. |
| MED-7 | Remediate | ~1 hr | Setter validation with `code.length > 0` + interface check. |
| MED-1 | **Accept with rationale** for this deploy round | (would require new ConsensusGroupRegistry) | Preserve push for now; revisit if k=2 consensus is widely deployed. Stop-gap option (`challenger == requester`) is a reasonable middle path if appetite for engineering exists. |
| MED-2 | **Accept with rationale** for this deploy round | (would require deposit-time auth-set commit) | Multi-receipt cherry-pick is an off-chain dispute concern; on-chain fix is invasive. Document as known limitation. |
| MED-5 | **Documentation-only** | ~30 min | Update docstrings + threat model; full on-chain fix not justified. |
| LOW-1, LOW-2, LOW-3 | Remediate when convenient | ~1 hr each | Defensive cleanups, batchable. |
| INFO-1 to INFO-5 | Remediate when touching adjacent code | minutes | Code/doc quality. |

## 7. Recommended next workstream

Before audit-bundle mainnet deploy under POL-2 §4 framework:

1. **Fix HIGH-1** (per-provider expiry tracker OR conservative MAX-window floor).
2. **Fix HIGH-2** (pause-time accumulator).
3. **Fix MEDIUM-3, MED-4, MED-6, MED-7** (small cluster of setter / constructor / snapshot hardening).
4. **Update docstrings** for MED-5 + INFO-1 through INFO-5.
5. **Re-run agent-teams self-audit** against the post-fix tip to confirm no new HIGH/CRITICAL surfaces opened by the fixes (mirrors the L2 → today regression-check pattern).
6. **Open 14-day public GitHub review window** per POL-2 §4.4 with this consolidated findings doc + remediation patches linked from the issue.
7. **Define TVL caps** per POL-2 §4.3: EscrowPool/StakeBond initial cap = $10K each.
8. **Verify Pausable wired** per POL-2 §4.2 + Foundation Safe holds PAUSER_ROLE.
9. **Deploy** under POL-2 §4 framework.

Estimated total engineering: 3-5 days for HIGH + cluster MEDIUM remediations + re-audit.

## 8. Sign-off

This consolidated document is the L4-self-2026-05-06 audit artifact required by PRSM-POL-2 §4.1. It satisfies the 0-CRITICAL pass criterion. The 2 HIGH findings require disposition (remediation OR accept-with-rationale) before audit-bundle mainnet deploy proceeds under POL-2 §4 framework.

— Generated by 4-team agent-teams self-audit (Teams A/B/C/D running Agent tool with `subagent_type=general-purpose`), 2026-05-06.
