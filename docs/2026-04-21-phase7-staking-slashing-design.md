# Phase 7: Provider Staking + Slashing — Design + TDD Plan

**Document identifier:** PRSM-PHASE7
**Version:** 0.1 Draft
**Status:** Combined design + plan (same pattern as Phase 3.1). Scope-contained to Tier C (single-provider + stake-slashing via challenge). Tier B (k-of-n redundant-execution consensus) deferred to Phase 7.1.
**Date:** 2026-04-21

**Dependencies:**
- Phase 3 (`phase3-merge-ready-20260420`) — `DispatchPolicy.min_stake_tier` field; `ProviderListing.stake_tier` field. Both currently advisory (self-reported). Phase 7 makes them enforceable.
- Phase 3.1 (`phase3.1-merge-ready-20260421`) — `BatchSettlementRegistry.challengeReceipt` with DOUBLE_SPEND / INVALID_SIGNATURE / NO_ESCROW reason codes. Phase 7 wires successful challenges to slash the offending provider's stake.
- Phase 1.3 Task 8 (mainnet deploy, hardware-blocked) — StakeBond + updated BatchSettlementRegistry deploy alongside existing contracts.

---

## 1. Context & Goals

Phases 2 and 3 explicitly reference **three verification tiers**:
- **Tier A (receipt-only):** provider signs a receipt with output hash; requester verifies the signature. Cheap. Shipped in Phase 2.
- **Tier B (redundant execution):** same shard runs on k providers; verifier takes majority vote. Deferred — scope-contained to Phase 7.1.
- **Tier C (stake-slashing):** single-provider execution, but provider posts stake as collateral; stake is forfeited on proven misbehavior.

Phase 2 and Phase 3 ship with Tier A. The Phase 3 `min_stake_tier` field accepts `"open" | "standard" | "premium" | "critical"` with nominal stake thresholds (0 / 5K / 25K / 50K FTNS), but **no on-chain stake enforcement**. Providers advertise a tier in their listing; requesters trust the advertisement.

This is the most-frequently-questioned gap in investor and frontier-lab conversations: *"What prevents a provider from lying about its tier?"* The current answer is "reputation tracking + eligibility filter" — real but weak.

**Phase 7 replaces the trust with cryptographic + economic enforcement.** Providers post FTNS stake to a `StakeBond` contract; their on-chain balance determines their effective tier; successful Phase 3.1 challenges automatically slash the stake; the slashed portion routes to the challenger's bounty + Foundation reserve.

### 1.1 Non-goals for Phase 7

- **Not Tier B (redundant execution).** Deferred to Phase 7.1. Requires separate MultiDispatcher + consensus logic.
- **Not full governance redesign.** Slash-rate parameters are Foundation-board-adjustable per PRSM-GOV-1 §4.2; no new governance surface introduced.
- **Not cross-chain staking.** Phase 7 ships on Base only. Multi-chain staking (if pursued) is Phase 7.2.
- **Not advanced slash-rate tiers.** Phase 7 ships with a single `slash_rate_bps` per tier; flexible rate schedules (e.g., decaying slash for repeated offenses) are a future refinement.
- **Not insurance pool.** The design reserves a Foundation operational reserve path for slashed FTNS but does not create a depositor-insurance product.

### 1.2 Backwards compatibility

Phase 7 is **strictly additive for Phase 3/3.1 consumers who don't opt in**. Providers with zero on-chain stake continue to operate at `min_stake_tier="open"` eligibility. Requesters whose `DispatchPolicy.min_stake_tier="open"` (the default) never query the stake bond contract. Paying-attention requesters set `min_stake_tier="standard"` or higher and get enforcement.

---

## 2. Scope

**In scope:**
- `StakeBond.sol` contract: per-provider stake balance, bond / unbond / withdraw / slash operations.
- `BatchSettlementRegistry` extension: successful DOUBLE_SPEND and INVALID_SIGNATURE challenges automatically slash.
- `StakeManager` Python wrapper: on-chain stake query, local tier resolution, bond/unbond txn helpers.
- `MarketplaceOrchestrator` integration: `min_stake_tier` enforcement via on-chain query.
- `ReputationTracker` extension: `record_slash()` counter (observability only — does not affect score derivation in Phase 7).
- Tests + E2E integration test.

**Out of scope (later phases / different docs):**
- Redundant-execution (Tier B) — Phase 7.1.
- Insurance pool for slashed providers — never (explicit non-goal).
- Federated-reputation gossip of slashing events — Phase 6 (federated reputation).
- On-chain reputation anchoring — out of scope.
- Slash-on-preemption — explicitly forbidden (Phase 2.1 Line A: preemption is honest-work failure).
- Time-varying unbonding penalties — future refinement.

---

## 3. Protocol

### 3.1 Stake lifecycle

```
  [Provider wallet]
       │
       │ approve(StakeBond, amount)
       │ bond(tier)  ─────►  [StakeBond contract]
       │                          │
       │                          │ status: BONDED
       │                          │ tier: standard|premium|critical
       │                          │ bonded_at: unix
       │                          │
       │ request_unbond()  ────►  │ status: UNBONDING
       │                          │ unbond_eligible_at: bonded_at + unbond_delay
       │                          │
       │  ⟨unbond_delay elapses⟩  │
       │                          │
       │ withdraw()  ──────────►  │ status: WITHDRAWN
       │                          │ funds returned to provider wallet
       │                          │
  ⟨OR⟩
       │                          │ ← slash() called by BatchSettlementRegistry
       │                          │ balance reduced by slash_amount
       │                          │ slash_amount routed to Foundation reserve
```

**Unbonding delay** (default 7 days): prevents flash-stake attacks where a provider bonds right before a dispatch, collects payment, then immediately unbonds and disappears. During unbonding, the provider's effective tier drops to `"open"` (no stake-backed).

**Slash during unbonding**: permitted. A provider who initiated unbonding but is subsequently caught in a challenge still forfeits the staked amount. This prevents the "challenge-then-unbond race" escape.

### 3.2 Stake tiers + thresholds

Phase 7 uses the Phase 3 tier enum verbatim. Thresholds in **FTNS wei** (18 decimals):

| Tier | Min stake | Slash rate on proven misbehavior |
|---|---|---|
| `open` | 0 | — (no stake, no slash) |
| `standard` | 5,000 × 10^18 | 50% of stake |
| `premium` | 25,000 × 10^18 | 100% of stake |
| `critical` | 50,000 × 10^18 | 100% of stake |

Slash rate determined by the tier the provider claimed AT COMMIT TIME (snapshot into the Batch). A provider who downgrades mid-flight still gets slashed at their committed-at-time rate.

### 3.3 Slashing trigger

`BatchSettlementRegistry` gains a new state field `stakeBond` (address of StakeBond contract). When a challenge of type DOUBLE_SPEND or INVALID_SIGNATURE proves misbehavior, the Registry:

1. Looks up the provider's claimed tier (stored alongside the batch; Phase 7 extension).
2. Calls `StakeBond.slash(provider_address, slash_rate_bps)`.
3. Emits `ProviderSlashed(provider, batch_id, receipt_leaf_hash, amount, slash_rate_bps, reason)`.

**NO_ESCROW and EXPIRED** challenges do NOT trigger slashing. Rationale:
- NO_ESCROW is requester-attestation-based; any requester can make the claim. Triggering slash on NO_ESCROW creates a griefing vector where requesters over-attest to steal provider stake.
- EXPIRED is a contract-rule violation but arguably protocol-hygiene rather than malice — stale receipts are usually accumulator bugs, not adversarial behavior.

Challenges that succeed under these reason codes still invalidate the receipt value (no payment), but do not touch stake.

### 3.4 Slashed FTNS routing

Per PRSM-TOK-1 governance + PRSM-GOV-1 §4.2:

- **70% to challenger bounty** (if the challenge was submitted by a party who is NOT the provider themselves — prevents self-slash loops).
- **30% to Foundation operational reserve**.

Rationale:
- Large challenger bounty incentivizes vigilance + pays the gas challengers spend to prove misbehavior.
- Foundation reserve share funds the challenge-bounty-budget (§5.3 of Phase 3.1 design) + audits + research.
- Challenger-is-provider edge case: 100% to Foundation reserve (prevents self-slash schemes).

---

## 4. Data model

### 4.1 StakeBond contract interface

```solidity
enum StakeStatus { NONE, BONDED, UNBONDING, WITHDRAWN }

struct Stake {
    uint128 amount;             // FTNS wei bonded
    uint64 bonded_at_unix;
    uint64 unbond_eligible_at;  // 0 while BONDED; set when request_unbond fires
    StakeStatus status;
    uint16 tier_slash_rate_bps; // snapshot of slash rate at bond time
}

interface IStakeBond {
    function bond(uint128 amount, uint16 tierSlashRateBps) external;
    function requestUnbond() external;
    function withdraw() external;
    function slash(
        address provider,
        bytes32 reasonId  // opaque; typically batch_id for auditability
    ) external;  // only callable by the registered slasher (BatchSettlementRegistry)

    function stakeOf(address provider) external view returns (Stake memory);
    function effectiveTier(address provider) external view returns (string memory);
    function slashedBountyPayable(address challenger) external view returns (uint256);
    function claimBounty() external;
}
```

### 4.2 BatchSettlementRegistry extensions

Add two fields to the `Batch` struct (no wire-format break — new fields appended):

```solidity
struct Batch {
    // ... existing Phase 3.1 fields ...
    uint16 tier_slash_rate_bps;   // snapshot from provider's stake at commit
    address challenger;            // null until a slashing challenge fires
}
```

Add two new admin entry points:

```solidity
function setStakeBond(address newBond) external onlyOwner;
function getStakeBond() external view returns (address);
```

The challenge handlers for DOUBLE_SPEND and INVALID_SIGNATURE extend to call `stakeBond.slash(batch.provider, batch_id)` after marking the receipt invalidated.

### 4.3 Python StakeManager

```python
class StakeManager:
    """Read-through cache of on-chain stake state + bond/unbond tx helpers."""

    async def bond(self, tier: StakeTier, amount_ftns_wei: int) -> bytes:
        """Submit bond tx. Returns tx hash."""

    async def request_unbond(self) -> bytes: ...
    async def withdraw(self) -> bytes: ...

    async def effective_tier(self, provider: str) -> str:
        """Cached read of StakeBond.effectiveTier. Used by
        MarketplaceOrchestrator's eligibility filter."""

    async def claim_bounty(self) -> bytes: ...
```

---

## 5. Integration points

### 5.1 MarketplaceOrchestrator

When `DispatchPolicy.min_stake_tier != "open"`, the orchestrator consults the StakeManager for each listing's provider before dispatching:

```python
# In _dispatch_one_shard, before price handshake:
if policy.min_stake_tier != "open":
    on_chain_tier = await self.stake_manager.effective_tier(listing.provider_id)
    if _tier_rank(on_chain_tier) < _tier_rank(policy.min_stake_tier):
        last_reason = "insufficient_on_chain_stake"
        continue
```

The EligibilityFilter's existing `min_stake_tier` check stays as a fast-path advisory filter (operates on the listing's self-reported tier); the orchestrator's new check is the load-bearing on-chain enforcement. Two-level defense.

### 5.2 ReputationTracker

Add `record_slash(provider_id, reason)` counter. **Does NOT affect score derivation in Phase 7** — slashed providers are already economically punished; doubling the hit via reputation penalty compounds the pain without adding signal. Observability-only.

### 5.3 Phase 3.1 BatchSettlementRegistry

- Existing `Batch` struct extended with `tier_slash_rate_bps` + `challenger`.
- `commitBatch` signature gains a `tier_slash_rate_bps` parameter (0 for providers with no stake; positive for Phase 7-enabled providers).
- Challenge handlers invoke `stakeBond.slash(...)` on success.
- `setStakeBond` admin function to wire the contract.

---

## 6. TDD plan

**9 tasks**, same shape as Phase 3.1 minus the Merkle/canonical-parity work (already done):

### Task 1: `StakeBond.sol` — bond + withdraw lifecycle

- Contract with `bond`, `requestUnbond`, `withdraw`, `stakeOf`, `effectiveTier`, `setSlasher`, `setUnbondDelay`.
- Tests: bond succeeds, double-bond rejected, requestUnbond transitions state, premature withdraw reverts, post-delay withdraw succeeds, slasher-only enforcement of `setSlasher` + `slash`.

### Task 2: `slash()` + bounty accrual

- Extend StakeBond with `slash(provider, reasonId)` + `slashedBountyPayable` + `claimBounty`.
- 70/30 split between challenger and Foundation reserve.
- Challenger-is-provider edge case routes 100% to Foundation.
- Tests: slashing reduces stake balance, double-slash rejected, bounty payable to challenger, claimBounty idempotent.

### Task 3: `BatchSettlementRegistry` extension

- Add `tier_slash_rate_bps` + `challenger` to Batch.
- Extend `commitBatch` signature.
- Add `setStakeBond` governance entry.
- Challenge handlers call `stakeBond.slash` on DOUBLE_SPEND / INVALID_SIGNATURE success.
- Tests: extended commit flow, slash fires on successful challenge, NO_ESCROW + EXPIRED do NOT slash.

### Task 4: `StakeManager` (Python)

- `prsm/staking/__init__.py`
- `prsm/staking/manager.py`
- Tests: bond / requestUnbond / withdraw tx building, effective_tier cached read, claim_bounty helper. Unit tests with AsyncMock stake contract.

### Task 5: `MarketplaceOrchestrator` integration

- Add `stake_manager: Optional[StakeManager]` to orchestrator constructor.
- Insert on-chain stake check in `_dispatch_one_shard` (before price handshake, after eligibility filter).
- Skip when `min_stake_tier="open"` to avoid unnecessary RPC.
- Tests: insufficient-stake skips dispatch, sufficient-stake proceeds, open-tier bypasses check entirely.

### Task 6: `ReputationTracker.record_slash`

- Add counter + event.
- Tests: record_slash increments counter, does NOT affect score_for (score unchanged).

### Task 7: E2E integration test

- 3-node cluster, provider B has 25K FTNS bonded at `premium`, provider C has 0 stake.
- Policy `min_stake_tier="premium"` excludes C.
- Provider B serves shard, then requester challenges via DOUBLE_SPEND (synthetic second batch).
- Slashing fires; provider B's balance reduced; challenger bounty claimable.
- Verify Phase 3 preserved when stake_manager is None.

### Task 8: Review gate + tag

Independent Agent review per Phase 3.1 Task 9 pattern. Tag `phase7-merge-ready-YYYYMMDD` on SAFE TO MERGE.

### Task 9: External audit + mainnet deploy

Hardware-gated. Bundle with Phase 3.1 Task 10's audit + mainnet deploy (shared auditor engagement reduces cost).

---

## 7. Acceptance criterion

**Roadmap-level:** 3-node cluster where:

1. Provider B bonds 25K FTNS at `premium` tier via StakeManager.
2. Provider C has 0 on-chain stake.
3. Requester dispatches with `DispatchPolicy.min_stake_tier="premium"`.
4. EligibilityFilter + on-chain stake check route all traffic to B (C excluded at stake check).
5. B serves, emits a valid receipt.
6. Test simulates B committing the SAME receipt in a second batch (double-spend).
7. Requester submits DOUBLE_SPEND challenge.
8. Challenge succeeds → `stakeBond.slash(B)` fires → B's balance drops by 25K × 100% = 25K FTNS.
9. 70% of slashed FTNS (17.5K) becomes claimable bounty for the challenger.
10. 30% (7.5K) flows to Foundation reserve wallet.
11. B's subsequent `effective_tier` returns `"open"` (stake now below `standard` threshold).

This lands as `tests/integration/test_phase7_slashing_e2e.py` in Task 7.

---

## 8. Open issues

### 8.1 Unbonding delay tuning

7-day default balances flash-stake prevention vs. legitimate-operator friction. Post-mainnet operational data will inform refinement. Governance-adjustable.

### 8.2 Bounty-claim gas

If the bounty is smaller than the gas cost to claim, challengers rationally skip claiming → bounty accumulates unclaimed → discouragement. Mitigations:
- Minimum-bounty threshold: challenge only pays if bounty ≥ gas-cost × safety-factor.
- Batch-claim across multiple successful challenges (Phase 7.1 refinement).

For Phase 7 MVP: ship as-is; track empirically; refine in 7.x if needed.

### 8.3 Challenger-provider collusion

A determined attacker could run BOTH the provider AND the challenger entities, committing an intentionally-bad batch and then "discovering" it themselves to capture the 70% bounty. This is net-negative for them (they lose 100% stake, recover 70%; 30% drain to Foundation), but they could use it as a leaked-stake-extraction vector.

Mitigations:
- 30% Foundation skim makes the attack strictly unprofitable.
- Sophisticated analysis: if the same wallet that bonded is also the challenger, route 100% to Foundation (covered in §3.4).

### 8.4 Slash-rate governance

The 50% / 100% slash rates per tier are load-bearing. PRSM-GOV-1 §13.2 designates these as **requiring supermajority Foundation governance** for amendment. Not a prohibited-amendment (unlike the "Foundation doesn't build chips" commitment), but not adjustable by simple majority either. Flagged for GOV-1 v0.2 explicit entry.

### 8.5 Stake-tier decay

A provider who bonds 50K FTNS at `critical` and then 10 years later still sits at 50K — is that still `critical`? In a USD-denominated world, 50K FTNS at 2035 prices may be economically trivial. Options:
- Let tiers be USD-denominated (requires oracle).
- Recalibrate tier thresholds periodically via governance.
- Accept the drift (Phase 7 MVP default).

For Phase 7 MVP: accept the drift. Periodic governance recalibration via `setTierThresholds` admin function is a Phase 7.x add.

### 8.6 Ed25519 verifier for INVALID_SIGNATURE slash path

Phase 3.1 Task 3 ships `INVALID_SIGNATURE` with a pluggable verifier (mock in tests; production Ed25519 library TBD for pre-audit). Slashing based on `INVALID_SIGNATURE` challenges is only as sound as the verifier. Real-deploy substitution of the verifier is a pre-Phase-7-mainnet-deploy step (call it part of Phase 7 Task 9 audit prep, not a blocker for merge).

---

## 9. Estimated scope

- **9 tasks** (8 engineering + 1 shared audit engagement).
- **~400 LoC Solidity** (StakeBond + Registry extensions + tests).
- **~500 LoC Python** (StakeManager + orchestrator integration + tests).
- **~150 LoC integration test + conftest extensions**.
- **Pre-hardware deliverable:** Tasks 1-8 complete, `phase7-merge-ready-YYYYMMDD` tagged.
- **Post-hardware deliverable:** Task 9 audit (shared with Phase 3.1 audit) + mainnet deploy.

Assuming similar velocity to Phase 3.1, the 8 pre-hardware tasks fit in ~3-4 sessions. Audit is async. Mainnet deploy is gated on hardware arrival.

---

## 10. Change log

**v0.1 (2026-04-21):** Initial combined design + plan. Nine tasks. Pre-hardware scope fully executable. Six open issues documented (unbonding tuning, bounty-claim gas, challenger-provider collusion, slash-rate governance, tier-threshold drift, Ed25519 verifier). Ready to begin Task 1 when user approves.

---

**End of PRSM-PHASE7 v0.1 Draft.**
