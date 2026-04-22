# Phase 8: On-Chain Halving Schedule Enforcement — Design + TDD Plan

**Date:** 2026-04-22
**Status:** Combined design + TDD plan written ~2.5 years ahead of target execution. Extends the 2026-04-16 scope doc with engineering-phase task breakdown. Specific parameter values subject to calibration with Epoch 1 operational data (2026-2030).
**Promotes:** `docs/2026-04-16-halving-schedule-implementation-plan.md` from scope doc to engineering-ready plan.
**Depends on:**
- Phase 1.3 mainnet deploy (FTNS token live).
- Phase 2 + 3 + 3.1 + 7 + 7.1 / 7.1x shipped (this is the "Phases 1-7 shipped and stable" dependency per §3.4 of the 2026-04-16 scope doc).
- ≥2 years of Epoch 1 operational-policy data (~2028).
- Foundation governance finalized.
- 2+ security-audit firms engaged.
**Target execution:** Q4 2028 (end of Epoch 1). Earliest viable Q2 2028. Latest acceptable Q2 2029.

---

## 1. Context & Goals

Phase 8 migrates FTNS emission rate enforcement from Foundation operational policy to on-chain smart contracts. The transition is load-bearing for the PRSM token-value thesis: `PRSM_Tokenomics.md` §4 commits to a Bitcoin-style halving schedule (4-year epochs, 0.5 halving factor); that commitment is currently trust-promised rather than trust-coded.

Why hybrid rather than day-one on-chain: operational-period (Epochs 1-2) compensation distribution produces real-world telemetry (actual rates per pool, operator-acceptance patterns, governance-intervention frequency) that informs contract parameter design. A day-one on-chain commitment would hard-code parameters chosen from model assumptions rather than observation; the hybrid Option C (per 2026-04-16 §1) buys calibration at the cost of 8 years of operational-policy trust.

**Phase 8 is the execution of the on-chain half.** It takes the calibrated parameter set from Epoch 1 observation and locks it into contract code before Epoch 2's halving takes effect.

### 1.1 Non-goals for Phase 8

- **Not a tokenomics redesign.** Halving schedule remains 4-year epoch / 0.5 factor, matching `PRSM_Tokenomics.md` §4. Phase 8 commits these to contract — parameter values may differ from today's operational targets based on Epoch 1 data, but the structural mechanism (periodic halving, monotone-decreasing) does not change.
- **Not a burn-mechanism change.** Burn lives in other contracts; Phase 8 touches emission only.
- **Not a staking-mechanism change.** `StakeBond` (Phase 7) unchanged.
- **Not a UX change.** Operators interact with CompensationDistributor the same way they interact with today's operational-policy distribution.
- **Not a retroactive change.** All FTNS issued under Epochs 1-2 operational policy stays issued; Phase 8 controls NEW emission.

### 1.2 Backwards compatibility

Two continuity guarantees:

1. **Foundation genesis allocation (100M FTNS) is never touched by Phase 8.** Foundation operational-policy distribution of the 100M genesis continues in parallel to Phase 8 contracts. Phase 8 authorizes NEW minting beyond 100M up to the 1B cap.
2. **Post-migration rates match the calibrated operational rates.** The Epoch 2 on-chain rate equals (Epoch 1 observed operational rate) × 0.5, not a clean-slate derivation. Continuity of operator economics is the design target.

---

## 2. Scope

### 2.1 In scope

**Solidity (new):**
- `contracts/contracts/EmissionController.sol` — mints FTNS from the governance-authorized allocation (cap-enforced, epoch-rate-enforced). Immutable halving schedule; public rate calculator.
- `contracts/contracts/CompensationDistributor.sol` — pulls emission from EmissionController and distributes to compensation pools (creator royalty bonus pool, node operator pool, contributor grant pool) under governance-set weights.

**Solidity (modified):**
- `contracts/contracts/FTNSToken.sol` — Phase 8 becomes an authorized minter. Existing FTNSToken must already support minter authorization (Phase 1.1 deliverable; verify during dependency check).

**Python (new):**
- `prsm/emission/emission_client.py` — read-side wrapper for epoch-rate queries, upcoming-halving notifications, operational telemetry.
- `prsm/emission/watcher.py` — background watcher that emits alerts on epoch boundaries + rate changes.

**Python (modified):**
- `prsm/economy/*` — integration point for the calibrated rate transitioning from operational to on-chain. Exact module TBD in Task 1 design review.

**Governance artifacts:**
- Parameter calibration data collected from Epoch 1 operational telemetry (the calibration-input dataset that drives Task 1's parameter choices).
- Pre-deploy formal-verification report on mint paths.
- 2+ security audits from established firms.
- Multi-stakeholder testnet exercise results.

### 2.2 Out of scope

- Changes to the halving schedule mechanism itself (4-year epochs / 0.5 factor).
- Changes to the 1B supply cap.
- Changes to `FTNSToken` beyond minter authorization.
- Changes to Foundation genesis allocation distribution (continues under operational policy).
- Any UX changes — Phase 8 is a backend migration invisible to users.
- Fiat on-ramp integration (that's Phase 5 scope).

### 2.3 Deferred / Phase 8.x candidates

- **Per-pool weighted distribution granularity beyond 3 pools** — today the 2026-04-16 plan specifies 3 pools (creator royalty / node operator / contributor grant). A finer-grained future split is a Phase 8.x refinement.
- **Cross-chain emission** — Phase 8 is Base-only. Any future multi-chain emission requires new plan.
- **On-chain governance of pool weights** — the 2026-04-16 plan says weights are governance-set; whether "governance" means Foundation multi-sig (operational) or an on-chain voting contract is deferred until Foundation governance structure finalizes.

---

## 3. Protocol

### 3.1 Emission flow

```
Epoch counter (block.timestamp-based)
        │
        ▼
EmissionController.currentEpochRate()
        │
        ▼
CompensationDistributor.requestDistribution()
        ├── pulls from EmissionController
        ├── caps against per-call + per-epoch limits
        ├── splits per governance weights
        └── emits to pool contracts
```

### 3.2 Epoch calculation

Epoch index is pure function of `block.timestamp`:

```solidity
function currentEpoch() public view returns (uint32) {
    if (block.timestamp <= epochZeroStartTimestamp) return 0;
    return uint32((block.timestamp - epochZeroStartTimestamp) / EPOCH_DURATION_SECONDS);
}
```

Where `EPOCH_DURATION_SECONDS = 4 years in seconds` and `epochZeroStartTimestamp` is an immutable constructor parameter matching the mainnet-deploy-block timestamp.

### 3.3 Rate calculation

```solidity
function currentEpochRate() public view returns (uint256 ratePerSecond) {
    uint32 e = currentEpoch();
    // Halving: rate_n = baseline * (1/2)^n
    return BASELINE_RATE_PER_SECOND >> e; // right-shift = divide by 2^n
}
```

Where `BASELINE_RATE_PER_SECOND` is immutable, set at deploy time to match the Epoch 2 starting rate (= Epoch 1 calibrated rate / 2). Using bit-shift rather than multiplication keeps the rate exactly halving without float rounding.

**Monotone-decreasing invariant enforced at contract level:** any caller reads `currentEpochRate()` and gets a value less than or equal to the previous call's value. The right-shift guarantees this structurally.

### 3.4 Mint authorization + cap

```solidity
uint256 public constant MINT_CAP = 900_000_000 * 10**18;  // 900M FTNS (total supply - 100M genesis)
uint256 public mintedToDate;

function mintAuthorized(uint256 amount) external onlyDistributor {
    uint256 maxAllowedThisCall = currentEpochRate() * timeSinceLastMint;
    require(amount <= maxAllowedThisCall, "ExceedsRateLimit");
    require(mintedToDate + amount <= MINT_CAP, "ExceedsMintCap");
    mintedToDate += amount;
    ftnsToken.mint(address(distributor), amount);
}
```

The per-call rate check is what makes the epoch-rate non-bypassable. No authorized caller can mint more than the epoch rate × elapsed time from last mint.

### 3.5 Distribution flow

```solidity
function distribute() external {
    uint256 available = ftnsToken.balanceOf(address(this));
    // Check per-epoch distribution cap (redundant but defensive)
    // Split according to current weights
    uint256 toCreatorPool = available * creatorPoolWeightBps / 10000;
    uint256 toOperatorPool = available * operatorPoolWeightBps / 10000;
    uint256 toGrantPool = available - toCreatorPool - toOperatorPool;

    ftnsToken.transfer(creatorPool, toCreatorPool);
    ftnsToken.transfer(operatorPool, toOperatorPool);
    ftnsToken.transfer(grantPool, toGrantPool);

    emit Distributed(toCreatorPool, toOperatorPool, toGrantPool);
}
```

Weights adjustable via governance (Foundation multi-sig at minimum; on-chain voting if / when finalized); weight changes require §4.1 90-day advance notice.

### 3.6 Migration from operational policy

Epoch 2 (year 8) is the target first on-chain-enforced halving. Migration sequence:

- **T-12mo:** Phase 8 contracts deployed on testnet. Multi-stakeholder exercise (Foundation multi-sig + operator reps + investors + auditors) runs a simulated halving transition.
- **T-6mo:** Audit reports published. Formal verification of mint paths complete.
- **T-3mo:** 90-day advance notice published per §4.1.
- **T-30d:** Stakeholder comment period per §4.1.
- **T=0 (Epoch 2 start):** Phase 8 contracts deployed on mainnet; distributor authorized as minter on FTNSToken; operational policy for post-100M emissions retires. Foundation genesis distribution continues.

The Epoch 2 rate on Day 1 equals Epoch 1 calibrated rate × 0.5 — continuity over discontinuity. Operator economics change by the halving amount, not by a clean-slate recalibration.

---

## 4. Data model

### 4.1 EmissionController.sol interface

```solidity
interface IEmissionController {
    // Public reads
    function currentEpoch() external view returns (uint32);
    function currentEpochRate() external view returns (uint256 ratePerSecond);
    function mintedToDate() external view returns (uint256);
    function mintCap() external pure returns (uint256);
    function epochZeroStartTimestamp() external view returns (uint64);
    function timeUntilNextHalving() external view returns (uint256 seconds);

    // Authorized-only writes
    function mintAuthorized(uint256 amount) external;
    function setAuthorizedDistributor(address newDistributor) external; // onlyOwner

    // Governance
    function pauseMinting() external; // multi-sig; emergency only
    function resumeMinting() external;

    // Events
    event Minted(address indexed recipient, uint256 amount, uint32 epoch, uint256 epochRate);
    event EpochTransition(uint32 oldEpoch, uint32 newEpoch, uint256 newRate);
    event DistributorUpdated(address oldDistributor, address newDistributor);
    event MintingPaused(address indexed caller);
    event MintingResumed(address indexed caller);
}
```

### 4.2 CompensationDistributor.sol interface

```solidity
interface ICompensationDistributor {
    struct PoolWeights {
        uint16 creatorPoolBps;    // basis points, sum must == 10000
        uint16 operatorPoolBps;
        uint16 grantPoolBps;
    }

    // Public reads
    function creatorPool() external view returns (address);
    function operatorPool() external view returns (address);
    function grantPool() external view returns (address);
    function currentWeights() external view returns (PoolWeights memory);
    function lastDistributionTimestamp() external view returns (uint64);

    // Distribution (permissionless — anyone can trigger)
    function pullAndDistribute() external;

    // Governance
    function updateWeights(PoolWeights calldata newWeights, uint256 scheduledAt) external;
    function setPoolAddresses(address creator, address operator, address grant) external;

    // Events
    event Distributed(uint256 toCreator, uint256 toOperator, uint256 toGrant);
    event WeightsScheduled(PoolWeights newWeights, uint256 effectiveTimestamp);
    event WeightsActivated(PoolWeights newWeights);
}
```

Weight update two-phase: `scheduledAt` provides the §4.1 90-day advance notice period on-chain rather than just operationally. `updateWeights` with `scheduledAt < now + 90 days` reverts.

### 4.3 Python EmissionClient

```python
class EmissionClient:
    """Sync web3 wrapper for EmissionController reads + subscriptions."""
    def __init__(self, rpc_url: str, controller_address: str): ...
    def current_epoch(self) -> int: ...
    def current_epoch_rate_per_sec(self) -> int: ...
    def time_until_next_halving_sec(self) -> int: ...
    def minted_to_date_wei(self) -> int: ...
    def mint_cap_wei(self) -> int: ...

class EmissionWatcher:
    """Background watcher subscribing to epoch-transition events."""
    def __init__(self, client: EmissionClient, on_transition: Callable): ...
    async def run_forever(self) -> None: ...
```

Read-only wrapper (no signing). Operators consume via dashboards; Foundation reads for treasury modeling.

---

## 5. Integration points

### 5.1 FTNSToken.sol

Requires `FTNSToken` to support minter authorization (typical ERC-20 pattern with `MinterRole` or OpenZeppelin AccessControl). Phase 1.1 should have shipped this; Task 1 verification step confirms. If the deployed FTNSToken lacks minter-role support, Phase 8 requires a prerequisite token upgrade (which is its own governance operation and would push timeline).

### 5.2 Governance hooks

- `EmissionController.setAuthorizedDistributor` — Foundation multi-sig.
- `EmissionController.pauseMinting` — Foundation multi-sig; emergency only.
- `CompensationDistributor.updateWeights` — Foundation multi-sig with 90-day scheduledAt.
- `CompensationDistributor.setPoolAddresses` — Foundation multi-sig; non-trivial op, rare.

All four hooks emit events; all four operate under the Foundation multi-sig (2-of-3 hardware wallets) consistent with Phase 1.3 and Phase 7 deploy posture.

### 5.3 Observability

- `EmissionWatcher` dashboards (Foundation + operator-facing): current epoch, current rate, time-until-next-halving, total-minted, cap-utilization.
- Event-log consumers: Foundation Quarterly Disclosure (PRSM-GOV-1 §11.2) includes per-quarter mint totals; SUPPLY-1 §4.3 trends may incorporate emission-by-pool telemetry.
- Monitoring alert: if `currentEpochRate()` ever returns a value different from expected-calibrated, alert immediately (possible clock drift, contract bug, or misconfigured `epochZeroStartTimestamp`).

---

## 6. TDD plan

**9 tasks**, mirroring the Phase 3.1 / Phase 7 / Phase 7.1 shape.

### Task 1: `EmissionController.sol` — core mint + rate

- Contract implementing §4.1. Immutable `epochZeroStartTimestamp`, `BASELINE_RATE_PER_SECOND`. Public view functions. `mintAuthorized` with rate + cap enforcement. Pause/resume hooks.
- Authorized-distributor role via Ownable + a separate distributor address slot.
- Tests: `currentEpoch` at boundary timestamps (0, epoch_end, epoch_end+1); `currentEpochRate` halving at each boundary; `mintAuthorized` rejects if exceeds rate-limit; rejects if exceeds cap; accepts valid mint; owner can change distributor; non-owner cannot; pause blocks mint; resume restores; events emitted correctly.
- Expected: ~20 tests.

### Task 2: `CompensationDistributor.sol` — pull + weighted split

- Contract implementing §4.2. Pool addresses, weight struct, permissionless `pullAndDistribute`.
- Two-phase weight update with 90-day `scheduledAt` enforcement.
- Tests: distribute splits correctly at 70/25/5 weights (illustrative); updating weights requires `scheduledAt >= now + 90 days`; weights activate on time; scheduled weights do not apply before activation; pool addresses settable only by governance; `pullAndDistribute` callable by anyone; distribution event emitted.
- Expected: ~15 tests.

### Task 3: `FTNSToken` minter-authorization verification

- Not a new contract. Verify existing FTNSToken (Phase 1.1) supports minter authorization compatible with EmissionController.
- If supported: no work; test that EmissionController can be added/removed as a minter.
- If unsupported: scope a prerequisite FTNSToken upgrade. This would delay Phase 8 by a full audit cycle; surface as a blocker in Task 1 design review.
- Expected: 3-5 tests confirming the authorization surface.

### Task 4: Python `EmissionClient` + `EmissionWatcher`

- `prsm/emission/emission_client.py` — sync web3 wrapper matching StakeManagerClient pattern.
- `prsm/emission/watcher.py` — async watcher on EpochTransition + Minted events.
- Tests: mocked web3 — epoch calculation, rate calculation, time-until-halving, minted-to-date. Watcher fires callback on epoch transition; reconnects on RPC failure.
- Expected: ~15 tests.

### Task 5: Multi-stakeholder testnet exercise

- Deploy Phase 8 contracts to Sepolia.
- Foundation multi-sig + 3-5 operator reps + 2-3 investors + auditors participate in a simulated Epoch 2 halving.
- Deliverables: test report covering (a) weight-update governance flow, (b) pause/resume under adversarial scenario, (c) rate transition at simulated epoch boundary, (d) operator-side dashboard UX, (e) monitoring alert firing and resolution.
- Required before §6 T=0 per 2026-04-16 §3.4.

### Task 6: Formal verification of mint paths

- Contracted external work. Target properties:
  - `mintedToDate` monotone-increasing.
  - `mintedToDate ≤ MINT_CAP` invariant.
  - `mintAuthorized` call never exceeds `currentEpochRate() × (block.timestamp - lastMintTimestamp)`.
  - Pause state blocks all `mintAuthorized` calls.
  - Owner / distributor access controls hold under all reachable states.
- Required before §6 T=0 per 2026-04-16 §3.4.

### Task 7: 2+ security audits

- Two audit engagements from distinct firms. At least one should be a firm with formal-verification specialty (supports Task 6).
- Audit scope: EmissionController.sol + CompensationDistributor.sol + Python client.
- Bundled with Phase 7.1x audit-engagement relationship if feasible (lower cost on shared auditor-ramp).
- Required before §6 T=0.

### Task 8: Migration playbook + governance ratification

- Playbook for §3.6 T-12mo through T=0.
- PRSM-GOV-1 §9.2 standards-amendment vote ratifying Phase 8 contracts + weight parameters.
- 90-day advance notice published per §4.1.
- 30-day stakeholder comment period per §4.1.
- Emergency-pause test (multi-sig can pause within 1 block of a Foundation vote).

### Task 9: Mainnet deploy + operational retirement

- Deploy EmissionController + CompensationDistributor to Base mainnet.
- Authorize CompensationDistributor as minter on FTNSToken via Foundation multi-sig tx.
- Retire operational policy for post-100M emission (Foundation publishes notice; all post-100M emission now contract-governed).
- Foundation genesis allocation operational policy CONTINUES in parallel.
- Freeze-tag: `phase8-mainnet-YYYYMMDD`.

---

## 7. Acceptance criterion

Three concrete criteria:

1. **On-chain halving happens at the Epoch 2 boundary.** At `block.timestamp >= epochZeroStartTimestamp + 2 × EPOCH_DURATION_SECONDS`, `currentEpochRate()` returns exactly `BASELINE_RATE_PER_SECOND >> 2`. Verifiable by any caller at any time.

2. **No single entity can cause rate violation.** No sequence of governance-authorized calls can produce `mintedToDate > MINT_CAP` or a single-call `amount > currentEpochRate() × timeSinceLastMint`. Verified by formal verification (Task 6) and audit (Task 7).

3. **Operator continuity.** On Day 1 post-migration, operators earning from the compensation pools receive FTNS at the expected Epoch 2 rate (Epoch 1 calibrated × 0.5), with no UX change vs operational-policy period. Verified by multi-stakeholder exercise (Task 5).

---

## 8. Open issues

### 8.1 Baseline rate calibration

The single largest unknown — `BASELINE_RATE_PER_SECOND` must be set at deploy time and is immutable. Setting it correctly requires 2 years of Epoch 1 operational data. Possible approaches:

- **Direct match:** Epoch 2 rate = observed Epoch 1 rate × 0.5. Safe default.
- **Per-pool-weighted average:** compute time-weighted average of per-pool operational rates, derive a unified Epoch 1 rate, halve. More faithful to observed economics.
- **Target-driven:** set rate to produce a target total-emission-per-year consistent with the 1B cap distributed over some horizon (e.g., 60 years). Most theory-driven but least responsive to actual adoption patterns.

Task 1 design review selects one approach with data from Epochs 1-2.

### 8.2 Per-call rate-limit window

§3.4's rate check uses `timeSinceLastMint`. What if no one calls for a month? A single call could mint a full month's worth. Alternatives:

- **Running average window** — track minting over a trailing 24-hour window, cap at 24 hours × rate.
- **Per-block cap** — cap at `rate × block_time` per block. Requires more frequent distributor calls, but rate-limit violations are bounded to single blocks.
- **Accept current behavior** — as long as the distributor is called sufficiently often (once per day minimum), burstiness is bounded.

Defer to Task 1 implementation review; probably accept current behavior with a monitoring alert on distributor-call gap > 7 days.

### 8.3 Governance structure finalization

Phase 8 assumes Foundation governance is operational with multi-sig + optional on-chain voting for weight updates. PRSM-GOV-1 §5 jurisdiction-selection is still open. If governance structure changes materially before Phase 8 execution, the `owner` / governance hooks (§5.2) may need reworking.

### 8.4 Pool contract design deferred

Phase 8 designs the distributor but treats each pool (creator / operator / grant) as an address receiving FTNS. How each pool DISTRIBUTES to its underlying recipients (creators, operators, grant recipients) is deferred to each pool's own contract design — out of Phase 8 scope. Phase 8 delivers funding mechanism; downstream pools deliver recipient mechanics.

### 8.5 Epoch-boundary timing under block-time drift

Base's block time varies slightly. Over 4 years of block accumulation, cumulative drift could push epoch boundaries off their calendar target by minutes or hours. Since `epochZeroStartTimestamp` uses wall-clock seconds, not block number, drift is bounded by the block producer's timestamp accuracy (Ethereum L2s: seconds max). Negligible at 4-year horizons but worth noting in audit.

### 8.6 Parallel genesis-allocation distribution

Foundation continues operational-policy distribution of the 100M genesis allocation in parallel to Phase 8's emission. This means Foundation operational discretion over genesis distribution persists indefinitely (until the genesis allocation is exhausted). Is that acceptable long-term? Arguably yes — genesis is a one-time allocation and operational discretion there doesn't compound into ongoing trust debt. But explicit acknowledgment is warranted in Task 8 ratification vote.

### 8.7 Cross-chain emission deferred

If PRSM expands to multiple L2s (Optimism, Arbitrum, etc.) before Phase 8 target, Phase 8 must be extended to handle emission across chains or deferred until cross-chain architecture stabilizes. Low probability on current roadmap; flag in Task 1 design review.

---

## 9. Estimated scope

- **9 tasks** (7 engineering + testnet exercise + audits + deploy).
- **Expected LOC:** ~400 new Solidity, ~300 new Python.
- **Expected test footprint:** ~55 new tests (40 Solidity + 15 Python).
- **Calendar duration:** 9-12 months from Task 1 kickoff to mainnet. Dominated by audit cycles and the T-12mo testnet exercise, not code complexity.
- **Budget:** 2+ audits + formal verification contracts + multi-stakeholder testnet exercise costs. Estimated $200k-$400k depending on audit-firm selection.

Scope is MODEST in code terms and SIGNIFICANT in governance / audit / testnet terms. The engineering is a small fraction of the total work.

---

## 10. Risk register

Risks specific to Phase 8 timely delivery:

### R1: Epoch 1 operational telemetry is insufficient for calibration

If observed rates in Epochs 1-2 are dominated by noise or incomplete adoption, §8.1 calibration is unreliable. Mitigation: Foundation publishes Epoch 1 operational telemetry quarterly (per PRSM-GOV-1 §11.2) so the data is public and Phase 8 calibration is auditable. Review cadence quarterly starting 2026-04 to surface inadequate data early.

### R2: Foundation governance structure not finalized by Q2 2028

PRSM-GOV-1 §5 lists jurisdiction as DECISION REQUIRED; no Foundation entity exists yet. If governance structure isn't finalized by Q2 2028, `onlyOwner` hooks have no clear holder. Mitigation: Foundation formation deadline is a PRSM-GOV-1 Year-1 milestone (§8.4); active founder attention is the mitigation.

### R3: FTNSToken lacks minter-role support

Task 3 discovers this. Worst case: delays Phase 8 by one audit cycle (3-6 months) because a prerequisite FTNSToken upgrade requires its own audit. Mitigation: verify minter-role support early — during Task 1 design review, not Task 3.

### R4: Audit firm availability

2+ audit firms required. Stablecoin / DeFi audit-firm market is typically capacity-constrained 3-6 months out. Mitigation: engage audit firms 9+ months ahead of planned audit start; lock slots early.

### R5: Epoch 2 boundary slips past 2029 Q2 deadline

§3.2 of 2026-04-16 plan: latest acceptable is Q2 2029. If Phase 1-7 delivery slips, Phase 8 may slip too. Mitigation: Phase 1.3 mainnet deploy (currently hardware-gated) is the earliest-sliding dependency; monitor hardware-acquisition cadence. Unlikely to impact Phase 8 directly but worth tracking.

### R6: Formal-verification vendor churn

Formal verification is specialist work; a preferred vendor closing or pivoting between now and 2028 is possible. Mitigation: identify 2+ candidate FV firms during Task 6 pre-work; maintain a warm contact with at least two.

### R7: Token-economics parameter overfit to bootstrap conditions

If Epochs 1-2 bootstrap conditions don't generalize (e.g., today's operator mix skews heavily T3 because hardware supply is cloud-dominated, but long-term mix is T1/T2-dominant), the Phase 8 calibration may lock in a rate that's wrong for the long-run steady state. Mitigation: PRSM-SUPPLY-1 diversity metrics provide explicit supply-mix telemetry that the calibration analysis can use to weight observations toward expected long-run mix rather than instantaneous bootstrap mix.

---

## 11. Relationship to PRSM-SUPPLY-1

PRSM-SUPPLY-1 (supply-diversity standard, 2026-04-22) and Phase 8 are adjacent governance surfaces with distinct concerns:

- **SUPPLY-1** measures supply-side concentration and activates diversity-bonus payments from Foundation reserve. Affects reserve OUTFLOW.
- **Phase 8** controls FTNS emission from the 1B cap via halving. Affects reserve INFLOW (Foundation pool portion).

The two interact via the Foundation reserve:
- Phase 8 distributes a portion of post-100M emission to the grant / Foundation pool.
- SUPPLY-1 spends Foundation reserve on diversity bonuses.
- Net reserve trajectory depends on balance.

Explicit coordination: ECON-WP v2 (per PRSM-TOK-1 update 2026-04-22) should model Phase 8 inflow + SUPPLY-1 outflow jointly.

---

## 12. Ratification path

Per PRSM-GOV-1 §9.2:

1. **Discussion:** 30-day public comment period via Foundation standards forum when Task 1 design is finalized.
2. **Revision:** incorporate comments.
3. **Ratification vote:** Foundation Board; this is a standards amendment under §9.2, not a charter amendment. Simple majority default; but see §4.1 of 2026-04-16 plan (Phase 8 activation requires 75% supermajority as the safeguard against governance capture).
4. **Activation:** contracts deploy to mainnet per §3.6 timeline; operational policy retires for post-100M emission.

Target ratification: **Q2 2028** (ahead of Epoch 2 boundary).

---

## 13. Ratification of this plan

This document is a design + TDD plan, not a protocol standard. It ratifies implicitly when:

1. Foundation reviews and accepts as the engineering blueprint for Phase 8.
2. Task-1-design review approved.
3. Phase 8 budget allocated.

No governance vote required at the plan level; the vote happens at Phase 8 deployment per §12.

---

## 14. Adjustments after execution starts

*(Empty; populated during execution.)*

---

## 15. Changelog

- **0.1 (2026-04-22):** initial design + TDD plan. Extends `docs/2026-04-16-halving-schedule-implementation-plan.md` scope doc with engineering-phase task breakdown. ~2.5 years ahead of target execution; parameter values subject to Epoch 1 operational-data calibration.
