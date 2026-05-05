# Team A — Economic Value Extraction Findings

**Pinned commit:** 589c14d2 (HEAD of main, contract source identical to `cumulative-audit-prep-20260504-h`)
**Auditor:** Team A (AI agent)
**Date:** 2026-05-04

## Summary

Two confirmed findings, both with runnable PoC tests under
`contracts/test/audit-team-a/`:

- **A-01 (HIGH — economic-spec divergence):** `RoyaltyDistributor.sol`
  does not implement the 20% burn mandated by PRSM-TOK-1 §8.1, and pays
  the network treasury 2.0% of gross instead of the spec-mandated 1.6%.
  The contract's split is `creator(rateBps) / network(2%) / node(rest)`
  — there is no burn at all, total supply is never reduced. This breaks
  invariant #1 in the team-prompt verbatim.
- **A-02 (MEDIUM — slash-evasion race):** `StakeBond.unbondDelaySeconds`
  and `BatchSettlementRegistry.challengeWindowSeconds` are independent
  governance parameters with no on-chain coupling. When governance sets
  `unbondDelay < challengeWindow` (allowed by both contracts' bounds —
  unbond min is 1 day, challenge window default is 3 days), a malicious
  provider can `commitBatch` + `requestUnbond` in the same block, then
  `withdraw` the full stake before any challenge can fire. The
  challenge succeeds (receipt invalidated) but the slash is silently
  swallowed by the try/catch in `BatchSettlementRegistry.challengeReceipt`
  because `slash()` reverts `NotSlashable` once status is `WITHDRAWN`.

Three additional informational items are flagged at the end:
adjacency to access-control on `setFtnsToken` (orphaning balances),
the tier-bond-vs-tier-batch decoupling (off-chain trust assumption),
and a mild "self-payment is cheap" follow-on from A-01 (with no burn,
the cost of wash-trading content access is 2% of gross instead of
the spec-intended 21.6%).

No reentrancy, donation, first-depositor, ETH-direct, MEV, griefing-
to-DoS, or rounding-dust path was found that perturbs invariant
behavior. See §"Vectors evaluated and cleared" for per-vector reasoning.

## Findings

### A-01 [HIGH]: RoyaltyDistributor split contradicts PRSM-TOK-1 §8.1 (no burn, treasury at 2% instead of 1.6%)

**Severity:** High
**Contract:** `contracts/contracts/RoyaltyDistributor.sol:62-100`
**Status:** Confirmed (PoC: `contracts/test/audit-team-a/SplitInvariant.test.js`, 3 passing tests)

**Attack scenario / broken invariant:**

PRSM-TOK-1 §8.1 (`docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md`,
lines 300-311) is unambiguous:

```
burn:          20.0%  (permanently destroyed)
treasury:       1.6%  (2% network fee × 80% of remainder)
creator:        6.4%  (8% royalty × 80% of remainder)
serving node: 72.0%  (90% × 80% of remainder)
              ─────
total:        100.0%
```

§8.1 prose: "The 20% burn is taken off the top; the remaining 80% is
split per the royalty/treasury/operator ratios established in Phase 1.1."

The team-prompt invariant #1 ("Payment split is exactly 20/6.4/72/1.6
bps for burn/creator/node/treasury. Sum is exactly 10000 bps. Any
deviation under any caller-controllable input is a finding") echoes
this verbatim.

The deployed contract (live at Base mainnet `0x3E82...D6c2` per the
Phase 1.3 Task 8 ceremony) instead executes:

```solidity
// RoyaltyDistributor.sol:73-76
uint256 creatorAmt = (gross * rateBps) / 10000;
uint256 networkAmt = (gross * NETWORK_FEE_BPS) / 10000;  // NETWORK_FEE_BPS = 200
uint256 nodeAmt    = gross - creatorAmt - networkAmt;
```

There is no `burnFrom`, no transfer to a burn-sink, no supply reduction.
With the registry's documented "set rateBps = 800" convention (8%
royalty per §8.4), the actual split is:

| Recipient | §8.1 spec | Contract behavior | Delta |
|---|---|---|---|
| Burn | 20.0% | 0.0% | **−20.0 pp** |
| Treasury | 1.6% | 2.0% | +0.4 pp (25% over-payment) |
| Creator | 6.4% | 8.0% | +1.6 pp |
| Serving node | 72.0% | 90.0% | +18.0 pp |
| **Sum** | 100.0% | 100.0% | — |

This is not a rounding/edge issue — it's a structural absence of the
burn mechanism that anchors the entire deflationary tokenomics
narrative (§8.2 quotes a 20% annual supply reduction at the
$100M/year volume scenario).

**Proof of concept:**

`contracts/test/audit-team-a/SplitInvariant.test.js` — 3 passing tests:
1. `burn is permanently 0 — TOK-1 §8.1 requires 20%`: registers content
   at rateBps=800, calls `distributeRoyalty(1000 FTNS)`, asserts
   `totalSupplyAfter == totalSupplyBefore` (0 burned), and that
   treasury holds 20 FTNS (2%) not 16 FTNS (1.6%).
2. `serving node is over-paid by ~19.6 percentage points vs §8.1`: at
   gross=10,000 FTNS the node receives 9,000 (90%); §8.1 mandates
   7,200 (72%). Over-payment quantum: 1,800 FTNS / 10,000 = 18 pp.
3. `creator share equals 6.4% only by coincidence`: even if the team's
   defense is "rateBps in registry should be set to the post-burn
   effective rate (640 = 6.4%)", treasury is still wrong (2% vs 1.6%),
   node is still wrong (91.6% vs 72%), and burn is still 0%.

To run:
```bash
cd contracts && npx hardhat test test/audit-team-a/SplitInvariant.test.js
```

**Recommended fix:**

Two paths — pick one and pin it in PRSM-TOK-1's amendment process:

1. **Implement the burn (faithful to §8.1).** In
   `RoyaltyDistributor.distributeRoyalty`, after pulling `gross`,
   compute `burnAmt = gross * 2000 / 10000`, call
   `IFTNSToken(ftns).burn(burnAmt)` (requires `BURNER_ROLE` or a
   permissionless burn-from-self entry on the token), then split
   the remaining `gross - burnAmt` per the existing 8/2/90 ratios.
   Net split lands at the §8.1 numbers automatically. Add a `burned`
   field to the `RoyaltyPaid` event.
2. **Amend PRSM-TOK-1 §8.1.** If the team's intent is "burn-on-use is
   deferred to a later phase", §8.1 must be edited to reflect the
   actual on-chain split (8/2/90, no burn) so investors and creators
   see the real economic distribution before the audit closes.
   §8.2-§8.4 burn rationale must be updated in lockstep.

Path 2 is a documentation-only change but leaves §3 ("Burn rate 2,000
bps of each payment — Normative") in conflict with the live contract.
We strongly recommend Path 1.

The contract is deployed as non-Ownable + non-upgradeable per the
Phase 1.3 Task 8 design (cumulative-audit-prep §7.16). Path 1 therefore
requires re-deployment with the new bytecode, a new `RoyaltyDistributor`
address, and migration of off-chain integrations. This is consistent
with the "deploy ceremony" model used for Task 8 — but it must happen
BEFORE production billing volume scales, because every payment between
now and a re-deploy under-burns FTNS by 20% of gross relative to spec.

---

### A-02 [MEDIUM]: Stake-bond/challenge-window race lets attacker dodge slash when `unbondDelay < challengeWindow`

**Severity:** Medium
**Contract:** `contracts/contracts/StakeBond.sol:59-60, 219-239` + `contracts/contracts/BatchSettlementRegistry.sol:482-495`
**Status:** Confirmed (PoC: `contracts/test/audit-team-a/UnbondRace.test.js`, 1 passing test)

**Attack scenario:**

The contract pair has two independent governance parameters:
- `StakeBond.unbondDelaySeconds`, bounded `[1 day, 30 days]`.
- `BatchSettlementRegistry.challengeWindowSeconds`, bounded `[1 hour,
  30 days]`, default 3 days.

There is no on-chain check that `unbondDelay >= challengeWindow`. A
governance configuration that sets `unbondDelay = 1 day` (the MIN
allowed) while leaving `challengeWindow = 3 days` (the default) — or
any pairing where `unbondDelay < challengeWindow` — opens the
following 4-step attack:

1. **T = 0:** Attacker (a bonded provider) calls `commitBatch` with
   fraudulent receipts (e.g., a double-spend setup using two batches
   over the same Merkle leaf, or a forged-signature leaf).
2. **T = 0 (same block):** Attacker calls `requestUnbond`. State
   transitions BONDED → UNBONDING; `unbond_eligible_at = T + 1 day`.
3. **T = 1 day:** Attacker calls `withdraw`. Status now WITHDRAWN,
   `s.amount = 0`, the entire stake is back in the attacker's wallet.
4. **T ∈ (1 day, 7 days):** Honest challenger discovers the fraud and
   calls `challengeReceipt(...)` with a valid Merkle proof + auxData.

Inside `challengeReceipt` (BatchSettlementRegistry.sol:489-494):
- Receipt invalidation lands successfully (`invalidatedReceipts[...]
  = true`, `b.invalidatedValueFTNS += leaf.valueFtns`).
- The wrapped `try stakeBond.slash(...)` reverts with `NotSlashable`
  (StakeBond.sol:338-340 — status must be BONDED or UNBONDING; it's
  now WITHDRAWN).
- The `catch {}` block on BatchSettlementRegistry.sol:491 swallows
  the revert. **Foundation reserve is unchanged. Challenger bounty
  is unchanged. The receipt is invalidated, but no economic penalty
  has been applied to the attacker.**

The contract's stated defense (StakeBond.sol:30-32 doc comment:
"Slashing during UNBONDING is permitted: a provider who initiated
unbonding is still accountable for misbehavior caught within the
delay window. Prevents the challenge-then-unbond race escape.")
**only holds if `unbondDelay >= challengeWindow`**. Without that
relationship, the attacker traverses UNBONDING → WITHDRAWN before
the challenger can act, and slashing becomes uncatchable.

The economic impact:
- Attacker keeps full stake (e.g., 25,000 FTNS for a Premium tier).
- The receipt is invalidated, so the requester doesn't pay for the
  fraudulent work, but that is a *defensive* outcome, not a
  *deterrent* one. Attacker sets up future fraud campaigns
  knowing the worst case is "no payment" (rather than "no payment
  AND lose stake AND fund the challenger's bounty"). The 70/30
  bounty/Foundation slash split — the entire economic deterrent
  in Phase 7 — is bypassed.

**Proof of concept:**

`contracts/test/audit-team-a/UnbondRace.test.js` — 1 passing test
that wires `unbondDelay = 1 day`, `challengeWindow = 7 days`, has
the attacker bond 25,000 FTNS at PREMIUM_SLASH_BPS = 10000 (100%
slash), then walks through the 4 steps above. Final assertions:
- Attacker's wallet gained back the full PREMIUM_STAKE.
- Foundation reserve is unchanged (= 0).
- Challenger bounty is unchanged (= 0).
- Stake amount is 0 (already drained pre-challenge).
- The challenge succeeded (`ReceiptChallenged` event emitted) — so
  receipt invalidation worked, but slashing did not.

To run:
```bash
cd contracts && npx hardhat test test/audit-team-a/UnbondRace.test.js
```

**Recommended fix:**

Three layered options, ranked by intrusiveness:

1. **(Preferred) Enforce `unbondDelay >= challengeWindow` in code.**
   Either:
   - In `StakeBond.setUnbondDelay`, take a registry address at
     construction and require `newDelay >= registry.challengeWindowSeconds()`.
   - Symmetric: in `BatchSettlementRegistry.setChallengeWindowSeconds`,
     require `newSeconds <= stakeBond.unbondDelaySeconds()`.
   - Either way, both `MIN_UNBOND_DELAY_SECONDS` and
     `MIN_CHALLENGE_WINDOW_SECONDS` need re-evaluation; the safest
     floor pairing is `MIN_UNBOND_DELAY_SECONDS == MAX_CHALLENGE_WINDOW_SECONDS = 30 days`.

2. **Pessimistic: set `MIN_UNBOND_DELAY_SECONDS = MAX_CHALLENGE_WINDOW_SECONDS`.**
   I.e., raise StakeBond.sol:59 to `30 days`. Always-safe under any
   governance choice of `challengeWindow` within current bounds.
   Penalises honest providers (longer wait to access stake) but
   removes the misconfiguration footgun entirely.

3. **(Operational) Document the constraint.** In PRSM-GOV-1, encode
   the invariant `unbondDelay >= challengeWindow` as a governance
   precondition on any future window adjustment. This is the
   weakest fix — it transfers the risk to operator vigilance. Not
   recommended given that the §7.16 Multi-Sig Action Plan operator
   audit just identified 10 findings on operator-side runbook
   tightness; adding more invariants for humans to track increases
   that surface.

Adjacency note: the `try/catch` swallowing of `NotSlashable` is
itself a code-smell. A challenge-success-but-slash-fail outcome
should at minimum emit a structured event (e.g.,
`SlashSwallowed(provider, reason)`) so monitoring can flag the
condition off-chain. Currently the silent swallow makes the race
invisible to forensic indexing.

---

### A-03 [INFORMATIONAL]: `EscrowPool.setFtnsToken` orphans existing balances

**Severity:** Informational (adjacency to access-control)
**Contract:** `contracts/contracts/EscrowPool.sol:183-188`
**Status:** Speculative

The `setFtnsToken(newToken)` function lets the owner swap the FTNS
token address while the `balances[]` mapping still records pre-swap
deposit amounts. The doc comment acknowledges this:

> `Cannot be called if there are non-zero balances pending in the
> pool, because old-token balances would be stranded — owner must
> drain first. NOTE: we do NOT track total-balance-sum cheaply, so
> the "no pending balances" check is operational-policy only.`

Economic-lens flag: this is purely an operator-side risk, but if
exercised mid-flight, every depositor's balance becomes a claim
against a token contract that no longer holds their value. A
malicious or compromised owner could exploit this. Real-world
defense is the Foundation's 2-of-3 multi-sig — out of Team A's
scope. Recommend: either remove `setFtnsToken` (it's only an
"escape hatch"; emergency redeploy can be a fresh `EscrowPool`)
or add an on-chain `totalBalanceSum` counter and revert when
non-zero. Flagged for Team B (access-control) follow-up.

---

### A-04 [INFORMATIONAL]: Bond-tier and batch-tier are decoupled (off-chain trust assumption)

**Severity:** Informational
**Contract:** `contracts/contracts/StakeBond.sol:167-192` + `contracts/contracts/BatchSettlementRegistry.sol:278-328`
**Status:** Speculative

`StakeBond.bond(amount, tierSlashRateBps)` snapshots the slash rate
at bond time. `BatchSettlementRegistry.commitBatch(..., tierSlashRateBps, ...)`
also snapshots a slash rate, declared by the *committer*. These are
NOT compared. A provider can:
- bond at `tierSlashRateBps = 0` (the "open tier" no-slash rate; or
  a deliberately small stake at PREMIUM_SLASH_BPS to still pass
  effectiveTier checks),
- commitBatch claiming `tierSlashRateBps = 10000` (premium),
- collect premium-tier work,
- on a successful challenge, `slash()` reverts `NothingToSlash`
  (zero-or-too-low computed amount) → caught by the try/catch →
  receipt invalidated but no slash.

This is partially a marketplace-orchestrator concern (off-chain
DispatchPolicy in Phase 7-storage is supposed to enforce
`min_stake_tier`), but the on-chain slash invariant assumes the
two snapshots agree. Recommend: in `commitBatch`, optionally read
`stakeBond.stakeOf(msg.sender).tier_slash_rate_bps` and reject if
it's lower than the declared `tierSlashRateBps`. Adjacency to Team C
(state-machine sequencing).

---

### A-05 [INFORMATIONAL]: Wash-trade economics are 10× cheaper than §8.1 intends (follow-on of A-01)

**Severity:** Informational
**Contract:** `contracts/contracts/RoyaltyDistributor.sol:62-100`
**Status:** Confirmed (analytic; uses A-01's PoC infrastructure)

A self-payment loop (attacker = creator AND serving node) under §8.1
costs the attacker `20% (burn) + 1.6% (treasury) = 21.6%` of gross
per round. Under the deployed contract (no burn, 2% treasury) the
cost is only `2%`. An attacker farming false reputation / false
"content access" volume to pump on-chain analytics or qualify for
Foundation distribution programs (per PRSM-TOK-1 §7.x) faces a 10×
lower deterrent than spec-modeled. This is mitigation-grade, not
exploitation-grade — flagged because it compounds A-01's economic
divergence. Resolves automatically when A-01's burn is implemented.

---

## Vectors evaluated and cleared

For each of A1–A10 from the team-prompt §"Specific attack vectors to
evaluate", explicit verdict:

- **A1 — Re-entrancy on `distribute()` via ERC-777 / ERC-1363 token paths.**
  CLEARED. `RoyaltyDistributor` uses OZ `ReentrancyGuard` *and* the
  load-bearing token (`FTNSTokenSimple`) is a vanilla OZ ERC-20 with
  no transfer hooks. ERC-20 `transfer` does not call recipient
  contracts. Even if a future deployment swapped in an ERC-777
  variant via the registry's `ftns` immutable, the immutable cannot
  be replaced — `RoyaltyDistributor.ftns` is set in the constructor
  with no setter. Same logic protects `EscrowPool` (whose
  `setFtnsToken` is owner-gated and economically separate from the
  reentrancy guard).

- **A2 — Donation attack on `EscrowPool` (force-send tokens to
  manipulate per-job accounting).** CLEARED. `EscrowPool.balances[]`
  is a per-requester mapping incremented on `deposit`, decremented
  on `withdraw`/`settle`. It never reads `ftns.balanceOf(this)`.
  Force-sent tokens are not credited to any requester and can only
  be reclaimed via owner-side recovery (none implemented; tokens
  lost). No accounting perturbation.

- **A3 — Rounding-dust accumulation on the 6400/7200/1600/200 bps
  split.** CLEARED. With the actual contract math (creator =
  `gross * rate / 10000`, network = `gross * 200 / 10000`, node =
  `gross - creator - network`), worst-case dust per call is bounded
  at 2 wei (two flooring divisions) and is fully captured by the
  serving node via subtraction. No value escapes; no accumulation
  surface. This finding stands separately from A-01 (which is about
  the absent burn, not about dust).

- **A4 — First-depositor share-inflation if `EscrowPool` uses
  share/proportional accounting.** CLEARED. `EscrowPool` uses
  per-address absolute balance accounting; there is no shares
  abstraction, no totalSupply, no `assets-per-share` denominator
  anywhere. Classic ERC-4626 / vault inflation attack does not
  apply.

- **A5 — Self-payment loop.** CLEARED of direct profit (A-01-aware).
  Even with 0% burn the attacker net-pays `networkAmt = 2%` per
  round, so the inner loop cannot net positive. The economic
  consequence is captured separately as A-05 informational —
  wash-trading is cheaper than §8.1 intends, but never profitable.

- **A6 — Griefing via a recipient that always reverts on receive.**
  CLEARED. Vanilla ERC-20 `transfer(to, amount)` does not invoke
  any callback on `to`. Even contract recipients (e.g., a Safe
  with custom guard hooks) cannot reject the transfer, because
  there is no transfer-hook surface to revert in. `StakeBond`
  bounty/reserve drains use plain `ftns.transfer(...)` to the
  caller (`msg.sender`) for `claimBounty` and to a configured
  wallet for `drainFoundationReserve` — same protection. No DoS
  path to lock funds.

- **A7 — Front-running batch settlement to claim a share that
  should have gone to a different operator.** CLEARED.
  `BatchSettlementRegistry.commitBatch` derives `batchId` from
  `keccak256(msg.sender, requester, root, count, blockNumber, sequence)`,
  with `msg.sender` baked in. Two providers cannot collide on
  batchId. The provider field in `Batch` is `msg.sender`, so even
  if a watcher front-runs `finalizeBatch`, the registry pays
  `b.provider` (the original committer), not the finalizer. There
  is no caller-controllable redirect surface in the value flow.

- **A8 — MINTER_ROLE / FTNS supply manipulation.** CLEARED of
  in-scope economic exploits. `FTNSTokenSimple.mintReward` enforces
  the `MAX_SUPPLY = 1e9 FTNS` cap. `MINTER_ROLE` grant requires
  `DEFAULT_ADMIN_ROLE` per OZ `AccessControl` — that's not
  by-passable from the economic surface. Adjacency: who holds
  `DEFAULT_ADMIN_ROLE` post-deploy is a Team B (access-control)
  question and per the Multi-Sig Action Plan the roles are still
  on a hot key pending hardware-wallet handoff. Flagged for Team B.

- **A9 — Stake-bond unbond-during-slash race.** **NOT CLEARED →
  A-02.** Confirmed exploitable when `unbondDelay < challengeWindow`.
  See finding A-02 above.

- **A10 — External-token interactions if any contract holds or
  routes tokens other than FTNS (e.g., does EscrowPool accept ETH
  directly?).** CLEARED. `EscrowPool` has no `receive`, no
  `payable` functions, no `fallback`. ETH cannot enter except via
  `selfdestruct` (which is being deprecated in EVM and even when
  it lands does not influence the FTNS-balances accounting).
  `RoyaltyDistributor` is the same shape — no ETH ingress, FTNS
  is the only token reference. `BatchSettlementRegistry` holds no
  funds at all (it routes settlement via `EscrowPool` and slashes
  via `StakeBond`); no token-mismatch surface. `StakeBond`
  similarly only handles FTNS. No multi-token routing exists
  anywhere in the bundle.

## Adjacencies flagged for other teams

- **Team B (access-control / signatures):**
  - `FTNSTokenSimple` `DEFAULT_ADMIN_ROLE` → `MINTER_ROLE` grant
    surface; current operator policy has the admin on a hot key
    pending hardware-wallet handoff (per Multi-Sig Action Plan).
  - `EscrowPool.setFtnsToken` owner-only escape hatch (see A-03).
  - `BatchSettlementRegistry.setChallengeWindowSeconds` and
    `StakeBond.setUnbondDelay` together gate A-02. The race
    severity depends on operator-side discipline in setting these.

- **Team C (state-machine sequencing):**
  - The `try/catch` swallow of `NotSlashable` in
    `BatchSettlementRegistry.challengeReceipt` (line 491) silently
    accepts a state where receipt invalidation lands but slashing
    doesn't. Even after A-02 is fixed, structured-event emission
    on swallowed reverts would improve forensic observability.
  - Bond-tier / batch-tier decoupling (A-04) is a state-snapshot
    coherence question.
