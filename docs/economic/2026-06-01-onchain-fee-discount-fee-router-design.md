# On-chain staking fee-discount enforcement — design scoping & go/no-go

**Status:** Scoping / RFC (no implementation). Decision-forcing doc.
**Date:** 2026-06-01
**Author:** PRSM engineering
**Trigger:** Sprint 906 implemented lock-based staking utility benefits
(service discount + dispatch priority) **off-chain** (`StakingManager` +
`RevenueSplitEngine`). The discount is enforced on the off-chain ledger
payment path. This doc scopes what it would take to also enforce the
**network-fee discount on the on-chain payment path**, and recommends
whether to do it.

---

## 1. The ask, precisely

Today a staker's network-fee discount (2% / 5% / 10% off the network fee
for a 30 / 90 / 365-day lock) is applied by `RevenueSplitEngine`
(`network_fee_discount_fraction`) on the **off-chain** payment ledger.
Purely **on-chain** payments — content royalties settled through
`RoyaltyDistributor.distributeRoyalty()` on Base mainnet — pay the full
`NETWORK_FEE_BPS = 200` (2%) with no discount, because the contract has no
knowledge of the payer's staking tier.

"On-chain fee-discount enforcement" = make `distributeRoyalty` (or its
successor) charge a staker the discounted network fee.

## 2. Why this is not a one-contract change

Two independent blockers, each of which forces a new contract or a new
trust surface:

### Blocker A — the fee-charging contract is immutable

`RoyaltyDistributor.sol` hardcodes the fee:

```solidity
uint16 public constant NETWORK_FEE_BPS = 200;           // constant — not settable
address public immutable networkTreasury;               // immutable
...
uint256 networkAmt = (gross * NETWORK_FEE_BPS) / 10000;  // fixed 2%, no per-payer logic
```

`NETWORK_FEE_BPS` is a compile-time `constant` and the split is computed
inline in `distributeRoyalty`. There is **no setter, no per-payer branch,
no upgrade proxy** (it is a plain immutable deployment, governed by
`INV-RD-1: NETWORK_FEE_BPS == 200`). The deployed contract therefore
**cannot** be made discount-aware. Any on-chain discount requires a
**new** contract:

- **Option A1 — RoyaltyDistributor v3.** A new distributor that reads the
  payer's tier and computes `networkAmt = gross * NETWORK_FEE_BPS *
  (1 - discount) / 10000`, crediting the waived portion back to the payer
  (or simply not pulling it). Requires re-pointing the off-chain
  payment path + any integrators to the v3 address, and re-running the
  ownership/wiring ceremony. Breaks `INV-RD-1` (intentionally — the fee
  becomes per-payer, not a flat 2%), so the formal-invariant registry
  needs a new invariant set for v3.
- **Option A2 — rebate router in front.** Keep the v2 distributor; a new
  `FeeRebateRouter` is what payers call. It pulls `gross`, calls
  `distributeRoyalty` (full 2%), then rebates the discounted fee fraction
  to the payer from a foundation-funded rebate pool. Avoids redeploying
  the distributor, but introduces a **funded rebate pool** the foundation
  must keep topped up, and the rebate is a second transfer (more gas).

Either way: a new audited contract + a deploy ceremony.

### Blocker B — there is no on-chain source of utility-staking tier

The sp906 utility-staking lock state lives entirely **off-chain**
(`StakingManager`, SQLite `ftns_stakes.stake_metadata`). The only
on-chain stake contract, `StakeBond.sol`, tracks **operator security
bonds** (per-provider `Stake{amount, unbond_eligible_at,
tier_slash_rate_bps}` with slash rates) — a different mechanism, keyed to
providers backing a service tier, **not** the 30/90/365-day utility-lock
discount tiers. So a fee contract has **nothing on-chain to read** to
learn a payer's discount tier. Three ways to supply it:

| # | Tier source | How it works | Trust surface | Cost / complexity | Latency / freshness |
|---|---|---|---|---|---|
| **B1** | **On-chain utility-staking lock contract** | New `StakingLock.sol`: users lock FTNS on-chain for 30/90/365d; exposes `tierOf(address) → bps`. Fee contract reads it directly. | Trustless (state is on-chain) | **Highest** — new staking contract + migrate the off-chain lock model on-chain (or run both); audit; users must lock on-chain (gas, UX change) | Real-time, exact |
| **B2** | **Oracle-signed tier attestation** | Foundation (or an oracle key) signs `(payer, tier, expiry)`; payer submits the signature; fee contract verifies `ecrecover`. | **Trusted signer** in the money path (a compromised/buggy signer can mint discounts; a securities-optics question — foundation actively grants a benefit per-tx) | Low–medium — `Ed25519Verifier.sol` / ECDSA verify already in repo; off-chain signer service | Real-time; signer must be live |
| **B3** | **Periodic merkle root of tiers** | Off-chain job publishes a merkle root of `(address → tier)` on-chain each epoch; payer submits a merkle proof; fee contract verifies against the current root. | Trust-minimized (root is committed on-chain; foundation only controls the root, publicly auditable) | Medium — root publisher job + on-chain root storage + proof verification (pattern already used by `BatchSettlementRegistry`) | Stale up to one epoch; cheap to verify |

## 3. The go/no-go question (read this first)

Before choosing an architecture, decide whether to build it **at all**.
The honest cost/benefit:

**Benefit is small and narrow.**
- The discount is on the **network fee**, which is 2% of the payment.
  Top tier (365-day lock) waives 10% of that → **0.2% of transaction
  value.** A 90-day staker saves 0.1%; a 30-day staker 0.04%.
- It applies **only to fully-on-chain royalty payments** through
  `RoyaltyDistributor`. The off-chain ledger path (where dev-mode and
  most current payments flow) **already enforces the discount** via
  sp906. So this closes a gap only for on-chain-settled royalties — a
  subset that is small today.
- It does **not** affect the dispatch-priority benefit (that's a
  marketplace-software concern, already wired, never on-chain).

**Cost is real and permanent.**
- A new audited Solidity contract (Blocker A) — PRSM's standard requires
  security review + Base Sepolia bake-in + a Foundation-multisig mainnet
  deploy ceremony.
- A permanent tier-source mechanism (Blocker B): either migrate staking
  on-chain (B1, big), stand up + secure a signer (B2, ongoing trust), or
  run a root-publisher job (B3, ongoing ops).
- New formal invariants, new emergency-pause template, new monitoring.
- If A1 (v3 distributor): a migration that re-points the payment path
  and supersedes the live, audited, invariant-pinned v2 distributor —
  non-trivial blast radius on a money contract.

**Verdict — recommend DEFER (no-go for now).** The marginal benefit
(≤0.2% of value, on the on-chain-royalty subset only) does not justify a
new money-path contract + a permanent trust/ops surface + a mainnet
ceremony, *while the off-chain path already enforces the discount*. This
is the documented honest-scope stopping point. Revisit when there is a
concrete demand driver — e.g., on-chain-settled royalty volume becomes
material **and** a high-value staker cohort cites the on-chain fee as an
adoption blocker. Build it then, with real data, rather than pre-committing
a ceremony now.

## 4. If we proceed anyway — recommended architecture

If the decision is go (e.g., a strategic partner requires on-chain
enforcement), the recommended combination is **A2 (rebate router) + B3
(merkle root)**:

- **B3 over B1**: avoids migrating the staking model on-chain (the
  off-chain `StakingManager` stays the source of truth; the root just
  mirrors it each epoch) and avoids a per-tx trusted signer in the money
  path (B2's securities-optics + key-risk problem). The merkle pattern is
  already proven in-repo (`BatchSettlementRegistry`).
- **A2 over A1**: a rebate router leaves the audited, invariant-pinned v2
  `RoyaltyDistributor` untouched (no money-contract migration, `INV-RD-1`
  preserved) and confines the new logic to an additive rebate path. The
  rebate pool is foundation-funded and capped, matching the "discount is
  a foundation fee concession, never an operator/creator haircut"
  economics from sp906.
- Net new on-chain surface: `FeeRebateRouter.sol` (verifies a merkle
  proof of the payer's tier against the current epoch root, rebates the
  discounted fee fraction from a funded pool) + a `tierRoot` storage slot
  updated by an owner-only `publishTierRoot(root)` behind the standard
  governance gate.

## 5. Path to a ceremony (so expectations are calibrated)

The mainnet multi-sig deploy is the **last** of six steps. We are at
step 0.

1. **Design ratification** — this doc's go decision + architecture choice.
2. **Implement** `FeeRebateRouter.sol` + the off-chain root-publisher +
   the merkle-proof client; TDD throughout.
3. **Formal invariants + halmos specs** for the new contract; emergency-
   pause template; monitoring.
4. **Security review** (the audit pipeline) + remediation.
5. **Base Sepolia bake-in** — deploy, exercise end-to-end, observe.
6. **Mainnet ceremony** — Foundation Safe + hardware-wallet signers
   deploy + wire the router and publish the first tier root. *(This is
   the only step that needs the Safe + signers; everything above is
   normal dev work.)*

## 6. Recommendation summary

- **Now:** No-go. Keep the sp906 off-chain enforcement as the stopping
  point. The on-chain forward item stays documented (PRSM_Tokenomics.md
  §5.3 / §8 #5) and gated on a real demand driver.
- **If/when go:** Architecture **A2 + B3** (rebate router + merkle tier
  root). Do **not** migrate staking on-chain (B1) or put a per-tx signer
  in the money path (B2) unless a specific requirement forces it.
- **Ceremony:** Only after steps 1–5. Having the Safe + signers ready is
  necessary but not sufficient — there is no contract to deploy yet.
