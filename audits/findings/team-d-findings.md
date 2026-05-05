# Team D — State-Composition Findings

**Audit team:** D (state-machine composition)
**Pinned commit:** `589c14d2` (HEAD of `main`) — identical to `cumulative-audit-prep-20260504-h`
**Baseline test status:** `npx hardhat test` → **441 passing** at the pinned commit
**PoC test status:** `npx hardhat test test/audit-team-d/...` → **10 passing**
**Date:** 2026-05-04

---

## 0. Executive summary

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| **D-01** | Slash-evasion race when `unbondDelay < challengeWindow` | **HIGH** | Confirmed by PoC |
| **D-02** | Treasury contracts have no pause mechanism (BSR/EscrowPool/StakeBond/RoyaltyDistributor) | **HIGH** | Confirmed by PoC |
| **D-03** | Owner cross-wire mutation is instantaneous; in-flight batches can be soft-bricked | **MEDIUM** | Confirmed by PoC |
| **D-04** | Royalty push-payment with no creator-side recovery; permanent share-stranding via ownership transfer | **MEDIUM** | Confirmed by PoC |
| **D-05** | `setChallengeWindowSeconds` retroactively shortens window for already-PENDING batches | **MEDIUM** | Confirmed by code review |
| **D-06** | `setEscrowPool` / `setSettlementRegistry` / `setSlasher` / `setFtnsToken` lack on-chain timelock | **LOW** | Confirmed by code review |
| **D-07** | `EscrowPool.setFtnsToken` strands old-token balances with no on-chain guard | **LOW** | Confirmed by code review |
| **D-08** | `providerBatchSequence` increments before collision check (consumed even on revert path) | **INFO** | Confirmed by code review |

**Headline:** the **stated invariant #2** in the team prompt — *"Unbonding delay is non-zero and respects in-flight slashes"* — is **false** under permitted governance values. With the contract-allowed bounds (`unbondDelay ≥ 1 day`, `challengeWindow ≤ 30 days`), a provider can withdraw stake fully before a successful in-flight challenge can land, evading slashing entirely. This is reproduced end-to-end in `UnbondVsChallengeRace.test.js`.

The **stated invariant #3** — *"Pause covers the full attack surface"* — is also **false**. There is no pause surface on any of the four core treasury contracts. The only kill-switch in the bundle is `FTNSTokenSimple.pause()`, which globally halts all FTNS transfers (production-impacting).

---

## 1. BatchSettlementRegistry state-transition diagram (REQUIRED ARTIFACT)

This is the central nexus identified in the team prompt. Produced by reading every line of `contracts/contracts/BatchSettlementRegistry.sol`.

### 1.1 Per-batch state machine

```
                   ┌──────────────────┐
                   │   NONEXISTENT    │  (default value of mapping; never written)
                   │  (status == 0)   │
                   └────────┬─────────┘
                            │
                            │ commitBatch(requester, root, count, value, tier, group, uri)
                            │   guards:
                            │     • requester != 0
                            │     • merkleRoot != 0
                            │     • receiptCount != 0
                            │     • tierSlashRateBps <= 10000
                            │     • providerBatchSequence[msg.sender]++  (consumed UNCONDITIONALLY)
                            │     • batchId derived = keccak256(provider, requester, root, count, block.number, seq)
                            │     • batches[batchId].status == NONEXISTENT  (defensive collision check)
                            │   side-effects:
                            │     • full Batch struct written (status=PENDING, commitTimestamp=now)
                            │   external calls: NONE
                            ▼
                   ┌──────────────────┐
              ┌────│     PENDING      │◄──────┐
              │    │  (status == 1)   │       │
              │    └────────┬─────────┘       │
              │             │                 │
              │             │                 │ challengeReceipt(...)
              │             │                 │   guards:
              │             │                 │     • status == PENDING
              │             │                 │     • elapsed < challengeWindowSeconds
              │             │                 │     • leaf hash not already invalidated
              │             │                 │     • MerkleProof.verify(proof, root, leafHash)
              │             │                 │     • reason-specific check passes
              │             │                 │       └─ DOUBLE_SPEND        → _handleDoubleSpend (read other batch)
              │             │                 │       └─ INVALID_SIGNATURE   → ext call: ISignatureVerifier.verify
              │             │                 │       └─ NO_ESCROW          → caller must == b.requester
              │             │                 │       └─ EXPIRED            → time check
              │             │                 │       └─ CONSENSUS_MISMATCH → _handleConsensusMismatch
              │             │                 │       └─ MALFORMED          → revert
              │             │                 │   side-effects:
              │             │                 │     • invalidatedReceipts[batchId][leafHash] = true
              │             │                 │     • b.invalidatedValueFTNS += leaf.valueFtns
              │             │                 │   external calls (post-effects):
              │             │                 │     • IF stakeBond != 0 AND tier_slash_rate_bps > 0
              │             │                 │       AND reason ∈ {DOUBLE_SPEND, INVALID_SIGNATURE, CONSENSUS_MISMATCH}
              │             │                 │     • PRE-CHECK: gasleft() >= MIN_SLASH_GAS (150_000)
              │             │                 │     • try IStakeBond.slash(provider, msg.sender, batchId)
              │             │                 │       catch { swallow — receipt stays invalidated }
              │             │                 └─────────┐ (loops back to PENDING; multiple challenges allowed)
              │             │                           │
              │             │ finalizeBatch(batchId)    │
              │             │   guards:                 │
              │             │     • status == PENDING   │
              │             │     • elapsed >= challengeWindowSeconds (CURRENT value, not commit-time)
              │             │   effects (BEFORE external call):
              │             │     • status = FINALIZED
              │             │   external calls:
              │             │     • IF (totalValue - invalidatedValue) > 0:
              │             │         REQUIRE escrowPool != 0
              │             │         IEscrowPool.settleFromRequester(requester, provider, finalValue)
              │             │           └─ EscrowPool internally: balances[requester] -= amount; ftns.transfer(provider, amount)
              │             │   emits BatchFinalized
              │             ▼
              │    ┌──────────────────┐
              │    │    FINALIZED     │  ── TERMINAL ──
              │    │  (status == 2)   │
              │    └──────────────────┘
              │
              │ (no current code path; reserved by enum)
              ▼
     ┌──────────────────┐
     │     VOIDED       │  (status == 3, RESERVED — not reachable in current code)
     │  (status == 3)   │
     └──────────────────┘
```

### 1.2 Cross-contract interaction graph (BSR-centric)

```
                                ┌─────────────────────┐
                                │ external Challenger │
                                │ external Finalizer  │
                                │ external Provider   │
                                │ external Requester  │ ── deposit/withdraw ──┐
                                └─────────┬───────────┘                       │
                                          │                                   ▼
                                          │                        ┌──────────────────┐
                                          │ commitBatch /          │   EscrowPool     │
                                          │ challengeReceipt /     │   - balances[]   │
                                          │ finalizeBatch          │   - settleFR()   │◄──┐
                                          ▼                        │   nonReentrant   │   │
                              ┌─────────────────────────┐          └────────┬─────────┘   │
                              │ BatchSettlementRegistry │                   │             │
                              │ - batches[batchId]      │                   │ ftns.       │
                              │ - invalidatedReceipts[] │                   │ transfer    │
                              │ - providerBatchSequence │                   │ (recipient) │
                              │ NOT nonReentrant        │                   ▼             │
                              │ NOT pausable            │       ┌──────────────────┐      │
                              │ Ownable                 │       │   FTNSToken      │      │
                              │                         │       │   (Pausable;     │      │
                              │  ┌─ try slash ─────────┼──────►│    UUPS proxy)   │      │
                              │  │                     │       └──────────────────┘      │
                              │  │ MIN_SLASH_GAS guard │                                  │
                              │  │ catch swallows      │       ┌──────────────────┐      │
                              │  ▼                     │       │   StakeBond      │      │
                              │                         ├──────►│ - stakes[]       │      │
                              │  ┌─ verify ────────────┼──────►│ - slashedBounty[]│      │
                              │  │                     │       │ - foundationRsv  │      │
                              │  │ INVALID_SIGNATURE   │       │ nonReentrant     │      │
                              │  │ branch              │       │ NOT pausable     │      │
                              │  ▼                     │       │ slasher = BSR    │      │
                              │                         │       └────────┬─────────┘      │
                              │  reads: other batches   │                │                │
                              │  writes: own state      │                │ ftns.transfer  │
                              │                         │                ▼                │
                              │ owner setters (instant):│       ┌──────────────────┐      │
                              │  - setEscrowPool        │       │   FTNSToken      │      │
                              │  - setStakeBond         │       │   (same as above)│◄─────┘
                              │  - setSignatureVerifier │       └──────────────────┘
                              │  - setChallengeWindow   │
                              │  - setLookbackWindow    │
                              └─────────────────────────┘
                                          ▲
                                          │ Pause? NONE.
                                          │ Timelock? NONE.
                                          │ Owner = Foundation Safe (2-of-3)
                                          ▼
                              ┌─────────────────────────┐
                              │     Foundation Safe     │
                              │     0x91b0…5791         │
                              └─────────────────────────┘

                   Independent path (not wired through BSR):

       ┌────────────┐    distributeRoyalty(...)     ┌─────────────────────┐
       │   Payer    │──────────────────────────────►│ RoyaltyDistributor  │
       └────────────┘                               │ - immutable wires   │
                                                    │ - 3-way push-pay    │
                                                    │ - nonReentrant      │
                                                    │ - NOT pausable      │
                                                    └─────────┬───────────┘
                                                              │
                                                              │ reads
                                                              ▼
                                                    ┌─────────────────────┐
                                                    │ ProvenanceRegistry  │
                                                    │ - contents[]        │
                                                    │ - no admin / no     │
                                                    │   pause             │
                                                    └─────────────────────┘
```

### 1.3 StakeBond state-transition diagram (referenced by D-01)

```
   ┌────────┐
   │  NONE  │
   └───┬────┘
       │ bond(amount, tier)  [reverts if AlreadyBonded for BONDED/UNBONDING]
       ▼
   ┌────────┐                                ┌────────────────────────┐
   │ BONDED │ ────── slash() (slasher) ────►│ stake.amount reduced;  │
   └───┬────┘                                │ status UNCHANGED       │
       │ requestUnbond()                     └────────────────────────┘
       │   sets unbond_eligible_at = now + unbondDelay
       ▼
   ┌──────────┐                              ┌────────────────────────┐
   │ UNBONDING│ ────── slash() (slasher) ───►│ stake.amount reduced;  │
   └─────┬────┘                              │ status UNCHANGED       │
         │ withdraw()                        └────────────────────────┘
         │   guards: now >= unbond_eligible_at
         ▼
   ┌──────────┐
   │ WITHDRAWN│ ── slash() reverts NotSlashable ── caught silently by BSR try/catch
   └──────────┘   (THIS IS THE D-01 BUG)
```

---

## 2. Findings

### D-01 (HIGH) — Slash-evasion race when `unbondDelay < challengeWindow`

**Status:** Confirmed end-to-end via PoC `test/audit-team-d/UnbondVsChallengeRace.test.js`.

**Invariant broken:** team-prompt §Stated invariants #2 — *"Unbonding delay is non-zero and respects in-flight slashes. A pending slash must block unbond completion until resolved."*

**Setting:** the contract-permitted bounds:

- `StakeBond.unbondDelaySeconds`: bounded `[1 day, 30 days]` (governance-adjustable).
- `BatchSettlementRegistry.challengeWindowSeconds`: bounded `[1 hour, 30 days]` (governance-adjustable).

There is **no on-chain invariant** binding these to each other. Operationally, the documented intent is `unbondDelay ≥ challengeWindow` — but neither the constructor, the setters, nor the contract's invariants enforce this.

**Attack sequence** (state-machine reference: `BONDED → UNBONDING → WITHDRAWN` racing `PENDING → FINALIZED`):

1. Foundation governance sets `unbondDelay = 1 day` (MIN) and `challengeWindow = 30 days` (MAX). Both legal under current bounds.
2. Provider `bond(10000 FTNS, 5000bps)`. Status: `BONDED`.
3. Provider `commitBatch(...)` for batch B1. PENDING begins.
4. Provider double-spends: same Merkle root committed in batch B2.
5. Provider `requestUnbond()`. Status: `UNBONDING`, `unbond_eligible_at = now + 1 day`.
6. **Day 1:** provider `withdraw()`. Stake fully refunded. Status: `WITHDRAWN`.
7. **Day 5:** legitimate challenger fires `challengeReceipt(B1, leaf, proof, DOUBLE_SPEND, auxData)`.
   - The challenge **succeeds**: receipt invalidated, value subtracted from final payable.
   - BSR enters the slash branch: `gasleft() >= MIN_SLASH_GAS` ✓; calls `stakeBond.slash(provider, challenger, batchId)`.
   - `StakeBond.slash` reverts with `NotSlashable(provider, WITHDRAWN)`.
   - The `try/catch` in `BatchSettlementRegistry.challengeReceipt` (line ~489-494) **swallows** the revert.
   - Result: receipt invalidated, but `slashedBountyPayable[challenger] == 0`, `foundationReserveBalance` unchanged. Provider walks away with full stake.

**Root-cause taxonomy:**

- The `try/catch` (BSR.sol:489) is justified to swallow legitimate `NotSlashable` cases (e.g., provider wasn't bonded — challenge can still serve a network-cleanup purpose). But this conflates two regimes: (a) provider never bonded → swallow is correct; (b) provider unbonded-then-withdrew within challenge window → swallow rewards the attacker.
- StakeBond intentionally allows slashing during `UNBONDING` to "close the challenge-then-unbond race escape" (see contract docstring). This protection only holds while `unbondDelay >= challengeWindow`, an unenforced invariant.

**PoC excerpt** (full file: `contracts/test/audit-team-d/UnbondVsChallengeRace.test.js`):

```javascript
// Provider unbonds + withdraws inside the challenge window.
await stakeBond.connect(provider).requestUnbond();
await time.increase(UNBOND_DELAY + 60);  // 1 day + ε
await stakeBond.connect(provider).withdraw();
// stake.status == WITHDRAWN; provider has full STAKE in their wallet

// Challenger challenges within the still-open 30-day window.
await registry.connect(challenger).challengeReceipt(...);
// Result:
//   foundationReserveBalance unchanged (0)
//   slashedBountyPayable[challenger] unchanged (0)
//   Receipt invalidated (good); provider unslashed (BAD)
```

**Recommendations** (in priority order):

1. **Hard invariant in code:** in `StakeBond.requestUnbond`, require `s.unbond_eligible_at >= block.timestamp + maxObservedChallengeWindow`. Practically: import the BSR's `challengeWindowSeconds` view and use `max(unbondDelaySeconds, registry.challengeWindowSeconds())`.
2. **Per-stake "in-flight challenge" hold:** when a batch is committed by `provider`, register a hold on `stakes[provider]` blocking `withdraw()` until `commitTimestamp + challengeWindowSeconds` for the latest such batch. Cleared when batch finalizes.
3. **Soft (governance-only):** documented operational invariant + `setUnbondDelay` rejecting values < current `challengeWindowSeconds` (cross-contract read).
4. **Defense-in-depth:** narrow the `try/catch` in `BSR.challengeReceipt` to swallow only `NotSlashable(_, NONE | WITHDRAWN)` is the hostile case; revert on it instead. Trade-off: legitimate "provider never bonded" challenges become harder to land.

---

### D-02 (HIGH) — Treasury contracts have no pause mechanism

**Status:** Confirmed via PoC `test/audit-team-d/PauseCoverageGap.test.js`.

**Invariant broken:** team-prompt §Stated invariants #3 — *"Pause covers the full attack surface. When paused, no value can move in any direction."*

**Code evidence:** the string `whenNotPaused` does not appear in any of `BatchSettlementRegistry.sol`, `EscrowPool.sol`, `StakeBond.sol`, `RoyaltyDistributor.sol`. None inherit from `Pausable` / `PausableUpgradeable`. The PoC asserts the absence of `pause()` / `paused()` selectors on all four contracts.

**Why this matters in the composed system:**

- Suppose a SignatureVerifier exploit lands (out-of-scope for Team D but in-scope for Team B/C). There is no way to halt batch finalization while the verifier is patched. Owner can `setSignatureVerifier(0)` — but that only kills future `INVALID_SIGNATURE` challenges, not in-flight finalizations.
- Suppose the `unbond/withdraw/slash` race (D-01) is being exploited. There is no way to pause `StakeBond.withdraw` while a fix is deployed.
- Suppose `EscrowPool.settleFromRequester` is being abused (e.g., via D-03 cross-wire mutation). No pause.

**The only kill-switch in the bundle is `FTNSTokenSimple.pause()`**, which has the property of pausing every FTNS transfer in the system globally — including legitimate user activity, exchange withdrawals, etc. This is administratively too heavy for surgical incident response.

**Recommendations:**

1. Add `OpenZeppelin Pausable` to all four contracts. Gate all state-mutating external functions on `whenNotPaused`. Pauser role assigned to the Foundation Safe (or a fast-response operator multi-sig with a separate threshold).
2. Specifically audit unbond timer behavior under pause (D-10 from the team prompt): when paused, does the `unbond_eligible_at` clock stop, continue, or reset? Document and enforce one. Recommendation: clock continues (timestamp-based, not block-based), but `withdraw()` itself is gated. This means a 30-day pause doesn't extend unbond time but does freeze the action.
3. Bundle pause coverage with a "circuit-breaker" governance role (separate from full owner) to enable a fast-response signer set.

---

### D-03 (MEDIUM) — Owner cross-wire mutation is instantaneous; in-flight batches can be soft-bricked

**Status:** Confirmed via PoC `test/audit-team-d/CrossWireMutationMidFlight.test.js`.

**Functions involved (no on-chain timelock; revert with `OwnableUnauthorizedAccount` only):**

- `BatchSettlementRegistry.setEscrowPool`
- `BatchSettlementRegistry.setStakeBond`
- `BatchSettlementRegistry.setSignatureVerifier`
- `BatchSettlementRegistry.setChallengeWindowSeconds`
- `BatchSettlementRegistry.setSettlementLookbackWindow`
- `EscrowPool.setSettlementRegistry`
- `EscrowPool.setFtnsToken`
- `StakeBond.setSlasher`
- `StakeBond.setUnbondDelay`
- `StakeBond.setFoundationReserveWallet`

**Demonstrated breakage** (from the PoC):

1. Provider commits batch B1 against pool P1. Owner calls `registry.setEscrowPool(P2)`. After challenge window, `finalizeBatch(B1)` attempts `P2.settleFromRequester(requester, ...)` — the requester's funds are still in P1, so `P2` reverts `InsufficientBalance`. Batch is stuck `PENDING` until owner reverts the wire.
2. Provider commits B1. Owner calls `pool.setSettlementRegistry(evilRegistry)`. After challenge window, `registry.finalizeBatch(B1)` calls `pool.settleFromRequester` which reverts `CallerNotRegistry`. Same soft-brick.

The PRSM-GOV-1 §10.3 14-day notice period is **off-chain policy only**. A compromised owner key (or 2-of-3 multisig collusion) can move instantly. Combined with D-02 (no pause), there is no recovery window between an attacker controlling the owner key and value being misrouted.

**Recommendations:**

1. Add a built-in 2-step setter pattern with `delay >= 7 days` to all `set*` cross-wire functions. Stage 1: `proposeNewEscrowPool(P2)`; Stage 2 (after delay): `commitNewEscrowPool()`.
2. For `setChallengeWindowSeconds` and `setSettlementLookbackWindow`, the contract already documents the retroactive effect (BSR.sol:735-749) but takes no action to mitigate. Either:
   - Apply only to batches committed AFTER the change (snapshot the window per-batch), or
   - Enforce that the new window cannot be SHORTER than the current one (one-way ratchet upward is safe).
3. Cross-wire setters should at minimum emit events (they already do) — operational monitoring should alert on these as a tripwire.

---

### D-04 (MEDIUM) — Royalty push-payment with no creator-side recovery; permanent share-stranding via ownership transfer

**Status:** Confirmed via PoC `test/audit-team-d/RoyaltyContagionGrief.test.js`.

**Mechanism:** `RoyaltyDistributor.distributeRoyalty` does **three pushes** atomically: creator share → `ftns.transfer(creator, ...)`; network → `ftns.transfer(networkTreasury, ...)`; node → `ftns.transfer(servingNode, ...)`. There is no `pull-bounty` / `claim` indirection (compare `StakeBond.slashedBountyPayable + claimBounty` for the correct pattern).

**Two demonstrated griefs** (both from the PoC, both pass):

1. **Permanent creator-share strand via ownership transfer.** A registered creator can `transferContentOwnership(contentHash, R)` where `R` is any contract that does not forward FTNS onward (e.g., a contract with no entry point, or `RoyaltyDistributor` itself). Future `distributeRoyalty` calls succeed, but the 50% creator share lands in `R` and is permanently stranded — `RoyaltyDistributor` has no admin / withdraw / sweep / rescue function. The PoC confirms the absence of all of them.

2. **Caller self-trap via `servingNode = distributor`.** The `servingNode` parameter is caller-supplied with no self-recipient check. A misconfigured caller can permanently strand the node share inside the distributor. Single-call self-grief, but reachable.

**Why this is currently bounded** (and why it should still be flagged):

- `FTNSTokenSimple` is a plain ERC20 — `transfer` to any address (EOA or contract) succeeds without recipient cooperation. So a creator cannot brick OTHER recipients' shares by being a non-receiving contract; only their own share is at risk.
- This bound depends on FTNS staying a plain ERC20. The `EscrowPool.setFtnsToken` escape hatch (D-07) means a future FTNS replacement with hooks/blacklists could turn this into a full-DoS vector.

**Recommendations:**

1. **Switch to pull-payment.** Mirror `StakeBond.slashedBountyPayable + claimBounty`. Each recipient claims their own balance. Eliminates push contagion entirely.
2. **Or: per-transfer try/catch.** If push semantics are required, wrap each of the three transfers in `try/catch` so a single recipient's failure doesn't revert the whole call. Stranded shares route to a recovery balance.
3. **Add a sweep/recovery function** owned by the Foundation, gated to recover only unspent balances (`token.balanceOf(distributor)` minus any in-flight allocations).

---

### D-05 (MEDIUM) — `setChallengeWindowSeconds` retroactively shortens window for already-PENDING batches

**Status:** Confirmed by code review (`BatchSettlementRegistry.sol:344-384` reads CURRENT window; comment at lines 735-749 acknowledges this).

This is the well-documented retroactivity issue. The BSR finalize path reads the CURRENT `challengeWindowSeconds`, not a per-batch snapshot:

```solidity
// finalizeBatch (line 352):
uint256 elapsed = block.timestamp - b.commitTimestamp;
if (elapsed < challengeWindowSeconds) { revert ... }
```

If owner shrinks the window from 30 days to 1 hour mid-flight (allowed by bounds), a challenger who was working with the original 30-day window assumption is suddenly locked out. Conversely, expanding the window delays already-pending finalizations indefinitely — the documented governance trade.

**Already documented in source** (BSR.sol:733-750 docstring acknowledges this as deliberate). Listed here for completeness because in combination with D-02 (no pause) and D-03 (instant cross-wire), an attacker controlling the owner can `setChallengeWindowSeconds(MIN)` to deliberately void all in-flight challenges before they can land. This is not a code bug per se — it's the natural consequence of treating `challengeWindow` as a contract-level mutable parameter.

**Recommendation:** snapshot the challenge window per-batch at `commitBatch` time (add `Batch.challengeWindowSecondsAtCommit`). Trade-off: 32 bytes more per batch.

---

### D-06 (LOW) — Multiple mutable wires lack on-chain timelock

Already enumerated in D-03. Listed separately as the LOW finding because, absent a key compromise, the off-chain governance notice (PRSM-GOV-1 §10.3) is sufficient. Should be combined with on-chain timelock in audit remediation.

---

### D-07 (LOW) — `EscrowPool.setFtnsToken` strands old-token balances with no on-chain guard

`EscrowPool.setFtnsToken` (line 183-188) explicitly admits in its docstring:

> NOTE: we do NOT track total-balance-sum cheaply, so the "no pending balances" check is operational-policy only. Owner MUST verify via off-chain indexing before calling.

This is governance-trust by docstring. A misconfigured owner action (or a key compromise) silently strands all requester balances.

**Recommendation:** track `totalEscrowedBalance` as an on-chain sum (incremented on deposit, decremented on withdraw/settle). Reject `setFtnsToken` while `totalEscrowedBalance > 0`. Cost: ~5K gas per deposit/withdraw/settle.

---

### D-08 (INFO) — `providerBatchSequence` increments before collision check

`BatchSettlementRegistry.commitBatch` line 292:

```solidity
uint256 sequence = providerBatchSequence[msg.sender]++;
```

Sequence is incremented unconditionally. If `commitBatch` reverts later (e.g., on `BatchAlreadyCommitted`, hypothetically reachable only via collision), the sequence is consumed. Under EVM semantics the revert rolls this back too, so this is informational only — no exploit. Flagged because future refactors may move the check or introduce a non-revert exit, in which case ordering matters.

---

## 3. Vectors evaluated and cleared

| Vector | Verdict | Why |
|--------|---------|-----|
| **D1** Reentrancy: `EscrowPool.release` → recipient → callback into EscrowPool/BSR | **Cleared** | EscrowPool.settleFromRequester applies effects (balance decrement) before `ftns.transfer`, has `nonReentrant`. BSR.finalizeBatch sets `b.status = FINALIZED` BEFORE the external `escrowPool.settleFromRequester` call, so reentry into `finalizeBatch(sameBatchId)` reverts on `BatchNotPending`. FTNSTokenSimple is plain ERC20 (no recipient hooks). Cross-batch reentry into BSR is benign — each batch has its own state slot. |
| **D2** Challenge-window block.timestamp drift (off-by-one at boundary) | **Cleared** | At `T = commit + window`: `finalizeBatch` requires `elapsed >= window` → SUCCEEDS. `challengeReceipt` requires `elapsed < window` → FAILS. No overlap, no gap. Strictly correct boundary handling. |
| **D3** Unbond ↔ slash race | **NOT cleared — see D-01** | High-severity confirmed finding. |
| **D4** Pause coverage | **NOT cleared — see D-02** | High-severity confirmed finding. |
| **D5** Push-payment grief (RoyaltyDistributor) | **NOT cleared — see D-04** | Medium-severity confirmed finding. |
| **D6** Unbounded loops in user-callable functions | **Cleared** | The only loops are inside `MerkleProof.verify` (depth = `log2(N)`, bounded) and `abi.decode` of fixed-shape `auxData`. No user-callable function iterates over user-controlled arrays of unbounded length. |
| **D7** Storage corruption / packed-struct write amplification | **Cleared** | `Batch` struct fields use individual full slots for dynamic types (`string metadataURI`); `address + uint16` packs cleanly. `Stake` struct: `uint128 + uint64 + uint64 = 256 bits` (1 slot), then `status (1 byte) + tier_slash_rate_bps (2 bytes)` (next slot). Compiler-managed; no manual packing; no cross-field write amplification. |
| **D8** Re-finalization (same batch finalized twice) | **Cleared** | `finalizeBatch` requires `b.status == PENDING`. After finalize, status becomes `FINALIZED`. Second call reverts `BatchNotPending`. |
| **D9** Ghost challenges (challenge against non-existent batch) | **Cleared** | `challengeReceipt` reverts `BatchNotFound` if `b.status == NONEXISTENT`. `_handleDoubleSpend` and `_handleConsensusMismatch` revert `ConflictingBatchNotCommitted` on missing conflicting batch. No dangling state path. |
| **D10** Pause-during-unbond | **Cleared by absence — but see D-02** | There is no pause mechanism, so the question is moot in current code. Once D-02 is remediated, this MUST be specified — recommend: clock continues, withdraw gated. |
| **D11** Owner mutates cross-wire mid-flight | **NOT cleared — see D-03** | Medium-severity confirmed finding. |
| **D12** Storage-collision via proxy / delegatecall | **Cleared for in-scope contracts** | BSR, EscrowPool, StakeBond, RoyaltyDistributor, ProvenanceRegistry are all NON-upgradeable. No `delegatecall`. No proxies. FTNSTokenSimple is UUPSUpgradeable but its storage layout is owner-controlled and out of state-composition scope. |
| **D13** Reorg sensitivity | **Cleared (operational)** | Base has 2-second blocks with 1-block soft finality. `batchId` includes `block.number` so a reorg fully unwinds. Challenge-window timing depends on `block.timestamp` which can move forward across a reorg (sequencer-controlled). A challenger acting at exact window-end + 1s could theoretically miss the window in a sequencer reorg, but this is a standard L2 risk, not a contract bug. Documented as operational concern in the deploy ceremony. |

---

## 4. PoC test inventory

All PoCs live in `contracts/test/audit-team-d/` and are runnable via:

```bash
cd contracts && npx hardhat test \
  test/audit-team-d/UnbondVsChallengeRace.test.js \
  test/audit-team-d/PauseCoverageGap.test.js \
  test/audit-team-d/RoyaltyContagionGrief.test.js \
  test/audit-team-d/CrossWireMutationMidFlight.test.js
```

Result: **10 passing**. Demonstrated on commit `589c14d2` against the existing 441-test green baseline.

| File | Vector | What it proves |
|------|--------|----------------|
| `UnbondVsChallengeRace.test.js` | D-01 | End-to-end: provider unbonds + withdraws + later challenge succeeds but slash is silently swallowed. |
| `PauseCoverageGap.test.js` | D-02 | All four treasury contracts lack `pause()` selector and `whenNotPaused` modifiers. |
| `RoyaltyContagionGrief.test.js` | D-04 | Creator transfers ownership to a non-receiving contract → 50% share stranded permanently with no recovery. |
| `CrossWireMutationMidFlight.test.js` | D-03 | Owner re-points `escrowPool` mid-flight → finalizeBatch reverts on empty new pool. Owner re-points pool's registry → real registry's settle reverts `CallerNotRegistry`. |

---

## 5. Out-of-scope notes

Per team-prompt boundaries, the following were observed but not investigated (handed to other teams):

- **Math correctness** (Team A): the `slashAmount` arithmetic in `StakeBond.slash` is `(amount * tier_slash_rate_bps) / 10000` — looks correct but rounding/cap behavior wasn't independently verified.
- **Signature verification** (Team B): `_handleInvalidSignature` flow uses `keccak256` binding of pubkey/sig hashes pre-verify; the cryptographic soundness of the verifier is Team B's call.
- **Access control** (Team C): the `Ownable` pattern across BSR/EscrowPool/StakeBond delegates trust to the Foundation Safe; the multi-sig setup itself (3-of-3 hardware wallets, ceremony provenance) is Team C's surface.
