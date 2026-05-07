# A-08 — RoyaltyDistributor.recoverStranded design

**Date:** 2026-05-07
**Status:** Decided — ready to implement
**Severity of underlying finding:** INFO (L4 self-audit re-run §A-08)
**Related findings:** A-04 / LOW-1 (StakeBond donation strand — same shape, deferred)

## Problem

`RoyaltyDistributor` (mainnet `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2`)
uses pull-payment via `claimable[address] mapping`. Anyone can
`ftns.transfer(distributor, X)` directly, bypassing `distributeRoyalty`.
Such donations are not credited to any `claimable[]` slot, so they cannot
be claimed and sit forever. Magnitude is bounded by donation rate; not
exploitable, footgun-only.

`RoyaltyDistributor` v1 has no `Ownable*` surface, so there is currently
no trusted recipient to gate a recovery call to.

## Decision

**Add `Ownable2Step` + `recoverStranded(address to)` to `RoyaltyDistributor`.**

The fix lands in the **v2 redeploy** that was already planned per
`RoyaltyDistributor.sol:44-49` to bundle:
- HIGH-1 / A-01 (push-payment → pull-payment migration; **already on disk** as the v2 source)
- HIGH-3 / D-02-deferred (Pausable; **separately scoped**, not in this PR)
- A-08 / this fix

A-08 piggybacks on the v2 deploy that has to happen anyway. No additional
on-chain operation beyond what was already required.

### Owner

Foundation Safe `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791` (Cayman 2-of-3
hardware-multisig). Mirrors every other audit-bundle / Phase 8 / Phase 7-
storage contract. Initial owner is the deployer hot key, transferred to
the Safe via `Ownable2Step.transferOwnership` + `acceptOwnership` in the
v2 ceremony.

### Recoverable amount

`recoverable = ftns.balanceOf(distributor) - totalClaimable`

Where `totalClaimable` is a new accumulator updated atomically with every
write to `claimable[]`:

| `claimable[]` write site         | `totalClaimable` delta      |
|----------------------------------|-----------------------------|
| `distributeRoyalty` (3 credits)  | `+= creatorAmt + networkAmt + nodeAmt` |
| `claim()`                        | `-= amount` (full claim, mapping zeroed) |
| `recoverStranded`                | unchanged (only touches the excess) |

The invariant **`balance(this) >= totalClaimable`** holds at all times:
- `distributeRoyalty` `transferFrom`'s `gross` and credits exactly `gross` to claimable pools (creatorAmt + networkAmt + nodeAmt = gross by construction).
- `claim()` `transfer`'s `amount` and decrements totalClaimable by `amount`. Both balance and accumulator drop equally.
- Donations only increment balance, not totalClaimable, and are always recoverable.

### `recoverStranded` semantics

```solidity
function recoverStranded(address to) external onlyOwner nonReentrant {
    if (to == address(0)) revert ZeroAddress();
    uint256 bal = ftns.balanceOf(address(this));
    if (bal <= totalClaimable) revert NothingToRecover();
    uint256 excess = bal - totalClaimable;
    if (!ftns.transfer(to, excess)) revert TransferFailed();
    emit StrandedRecovered(to, excess);
}
```

- Sweeps **all excess in one call** — no `amount` parameter. Owner can split off-chain after recovery if granular routing is needed.
- `nonReentrant` because the FTNS token is an external contract; canonical FTNSTokenSimple is hookless but defense-in-depth.
- Reverts with `NothingToRecover()` instead of silently emitting a 0-value event.
- **Crucially does NOT touch `claimable[]`** — recipients keep their full entitlement. The accumulator is a floor, not a ceiling.

### What cannot happen

- **Owner cannot drain claimable balances.** `recoverStranded` only sees `balance - totalClaimable`, which by the invariant equals strictly the donations.
- **Owner cannot front-run a `distributeRoyalty`.** The credit is atomic with the `transferFrom`, so an interleaved `recoverStranded` would see no excess.
- **Owner cannot front-run a `claim()`.** Same — the decrement is atomic with the outbound transfer.

## Alternatives considered

1. **Per-call recovery cap (`amount` parameter).** Rejected. A compromised
   Safe with the `recoverStranded` privilege would still be able to drain
   the full excess in one call regardless. The cap adds no security and
   complicates the API.

2. **Recompute `totalClaimable` on the fly.** Impossible — the `claimable`
   mapping cannot be iterated on-chain. Would require off-chain indexing
   to be authoritative, which fails the "trustless" property of the
   recovery surface.

3. **Sweep-to-self only.** Rejected. Foundation Safe is the natural
   recipient, but the donor may have sent funds to the wrong address by
   mistake; a `to` parameter lets the Safe route to the original donor
   if identified off-chain. The same security property holds either way
   (only owner can call).

4. **Don't add the recovery surface (option (b) of the audit
   recommendation).** Rejected. The fix is small (~30 LoC), deploys
   alongside the already-planned v2 redeploy at zero marginal cost, and
   closes a documented LOW-shape footgun. Carrying the known limitation
   forward into v2 would require explicit auditor sign-off later;
   landing the fix now is cheaper.

5. **Add `Pausable` (HIGH-3 / D-02 scope) at the same time.** Deferred.
   The HIGH-3 disposition for RoyaltyDistributor is separately tracked
   and would expand the PR scope beyond the A-08 finding. Bundling at
   v2 deploy time is fine if HIGH-3 disposition lands first, but this
   PR is A-08-only.

## Implementation checklist

- [x] ADR written (this file)
- [x] `RoyaltyDistributor.sol` patch — `Ownable2Step`, `totalClaimable`, `recoverStranded`, `StrandedRecovered` event, `NothingToRecover` / `ZeroAddress` / `TransferFailed` errors
- [x] Existing tests updated for new constructor signature (`initialOwner` arg) — 5 callsites across `RoyaltyDistributor.test.js`, `audit-team-a/SplitInvariant`, `audit-team-b/B-AccessControl-PoC`, `audit-team-d/PauseCoverageGap`, `audit-team-d/RoyaltyContagionGrief`
- [x] `deploy-provenance.js` updated for 4-arg constructor + optional `ROYALTY_DISTRIBUTOR_OWNER` env override (defaults to deployer)
- [x] New regression tests for A-08 (9 tests):
  - `totalClaimable` invariants (start at zero, lockstep with distribute, lockstep with claim)
  - Donate-only → recover full donation
  - Donate + distribute + claim → recover excess only; claimable balances untouched
  - `NothingToRecover` revert (balance == totalClaimable, fresh empty)
  - `ZeroAddress` revert (`to == 0`)
  - `OwnableUnauthorizedAccount` revert for non-owner
  - `Ownable2Step` `transferOwnership` → `acceptOwnership` flow
- [x] Full hardhat suite green (546 passing, +10 vs pre-PR)
- [x] Commit + push
- [ ] (Out of scope this PR) v2 mainnet redeploy ceremony — bundled with HIGH-1 + HIGH-3 disposition outcome.
