# Aerodrome USDC-FTNS Pool Seed Ceremony Plan

**Document identifier:** AERODROME-POOL-SEED-CEREMONY-PLAN-1
**Version:** 0.1 Draft
**Status:** Pre-ceremony planning. To be ratified by founder council resolution **before** execution. **NOT** a council resolution itself.
**Date drafted:** 2026-05-13
**Target ceremony date:** 2026-06-15 per `PRSM_Vision.md` §13 gantt (movable subject to council ratification of Resolution 8 from `PRSM-CR-2026-05-08.md` + Sepolia rehearsal completion).
**Drafting authority:** PRSM founder
**Companion docs:**
- `docs/governance/PRSM-CR-2026-05-08.md` §Resolution 8 — funding-source decision deferred; this ceremony **cannot** execute until that resolution exists.
- `docs/governance/2026-06-15-aerodrome-pool-seed-sepolia-rehearsal.md` — rehearsal runbook (this packet).
- `prsm/economy/web3/aerodrome_client.py` — read-only quoter that will exercise the pool post-seed.
- `Tokenomics §3.5` (Foundation seeds, does NOT market-make) + `§3.7` ($250K-$1M USDC + matched FTNS envelope, ratified per PRSM-CR-2026-05-08 §Resolution 8).

---

## 1. Purpose

PRSM's Phase 5 fiat surface ships commission-ready (sprints 276-286) but is **inert** until a USDC-FTNS pool exists on Aerodrome (Base mainnet, Velodrome fork). This ceremony seeds the pool with initial liquidity from the Foundation Safe — a one-time bootstrap event, NOT continuous market-making (see Tokenomics §3.5 / `2026-05-08` Resolution 3 alignment).

**Post-ceremony state:**
- Aerodrome pool exists with FTNS price discovery active (volatile pool, xy=k constant-product invariant)
- Foundation Safe holds the LP tokens (initial liquidity is non-transferable)
- `prsm/economy/web3/aerodrome_client.py` MCP endpoint returns real pool state to operators via `prsm_pool_quote`
- Coinbase CDP commissioning becomes unblocked (the offramp composer's quote endpoint pulls live USDC/FTNS prices)
- `PRSM_FTNS_USD_RATE` env-var fallback gets a real on-chain replacement; the operator-trust seam from audit-prep §7.23 closes

**What this ceremony is NOT:**
- NOT a Coinbase CDP commissioning. That's a separate engagement.
- NOT a continuous market-making activation. The Foundation provides initial liquidity once; subsequent market-making is operator/community-driven.
- NOT a veAERO gauge whitelisting. That's a downstream governance vote (separate ceremony).
- NOT a fleet kill-switch activation. Independent design.

---

## 2. Pre-ceremony council action (BLOCKING)

**This ceremony cannot execute until the following council resolution exists.** Drafted but not yet ratified.

### 2.1 PRSM-CR-2026-06-XX Resolution 8 follow-up (REQUIRED)

`docs/governance/PRSM-CR-2026-05-08.md` §Resolution 8 deferred the funding-source decision (Foundation treasury vs Prismatica balance sheet). Before this ceremony can execute, founder council must ratify a resolution that:

- Names the funding source (Foundation Safe holds USDC + FTNS, or Prismatica wires USDC to Foundation Safe with binding pre-agreement)
- Locks the seed amounts within the $250K-$1M USDC envelope (Tokenomics §3.7)
- Locks the matched FTNS amount (sets implied bootstrap price)
- Names the LP-token recipient (default = Foundation Safe)
- Records the ceremony date + signers

**Draft CR text** (founder fills bracketed fields):

> Resolution: Authorize Aerodrome USDC-FTNS pool seed ceremony per `AERODROME-POOL-SEED-CEREMONY-PLAN-1`.
> - Funding source: [Foundation treasury | Prismatica wire]
> - USDC seed amount: [$X within $250K-$1M envelope]
> - FTNS seed amount: [Y FTNS]
> - Implied initial price: [$X/Y per FTNS]
> - Pool type: VOLATILE (Aerodrome `stable=false`)
> - LP recipient: Foundation Safe `0x91b0...5791`
> - Ceremony date: 2026-06-15 (target)
> - Signers: 2-of-3 hardware (Ledger + Trezor; OneKey reserve)

---

## 3. Pre-flight checklist

### 3.1 Council + governance

- [ ] Resolution 8 follow-up ratified (§2.1 above)
- [ ] Funding source confirmed; if Prismatica-funded, USDC wire to Foundation Safe completed + Basescan-verified before D-1
- [ ] Council recorded in `docs/governance/PRSM-CR-2026-06-XX.md` with chosen amounts + funding source

### 3.2 Repo state

- [ ] `git status` clean on `main` at the commit anchoring the ceremony
- [ ] `contracts/scripts/build-aerodrome-pool-seed-tx.js` matches the audit-prep §7.x reference commit (shipped alongside this doc)
- [ ] `npx hardhat compile` succeeds without warnings (no contracts deployed here — script just constructs calldata)

### 3.3 Network + RPC

- [ ] Base mainnet RPC endpoint reachable (`PRSM_BASE_RPC_URL` or default `https://mainnet.base.org`)
- [ ] Basescan reachable (`https://basescan.org`)
- [ ] Aerodrome canonical contracts re-verified via Aerodrome docs (https://aerodrome.finance/docs) — operator confirms addresses at ceremony time, since contract addresses can change with protocol upgrades:
  - [ ] AerodromeRouter address (current canonical: `0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43` per 2024 deployments — **operator: re-verify at ceremony time**)
  - [ ] AerodromePoolFactory address (current canonical: `0x420DD381b31aEf6683db6B902084cB0FFECe40Da` — **operator: re-verify**)
  - [ ] USDC Base-native address (`0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` per Circle docs)

### 3.4 Foundation Safe state

- [ ] Foundation Safe `0x91b0...5791` reachable via Safe{Wallet} UI (https://app.safe.global)
- [ ] Safe holds ≥ council-ratified FTNS seed amount (per memory: Safe holds 100M FTNS after 2026-05-06 migration)
- [ ] Safe holds ≥ council-ratified USDC seed amount (if Prismatica-wired, balance confirmed on Basescan)
- [ ] Safe holds ≥ 0.05 ETH on Base for gas (3 txs × estimated 200K gas × current gas price)

### 3.5 Hardware + signer set

- [ ] Signer 1 (Ledger) available + firmware up-to-date, Ethereum app installed
- [ ] Signer 2 (Trezor) available + firmware up-to-date, Ethereum app installed
- [ ] Signer 3 (OneKey, reserve) location confirmed; not used unless Ledger or Trezor unavailable
- [ ] Test sign on Sepolia rehearsal completed within last 7 days (see Sepolia runbook)
- [ ] Quiet room / 60-min focused window booked

### 3.6 Sepolia rehearsal

- [ ] Full Sepolia rehearsal of the same 3-tx sequence executed (see `2026-06-15-aerodrome-pool-seed-sepolia-rehearsal.md`)
- [ ] Rehearsal pool exists on Sepolia Aerodrome (if Sepolia deploy available — otherwise rehearsal uses Base Sepolia or local Anvil fork)
- [ ] All 3 txs in rehearsal returned status=1; LP tokens visible in Sepolia Safe

---

## 4. Ceremony transaction sequence

Three sequential transactions, all signed by 2-of-3 Foundation Safe owners. Each tx is queued via Safe{Wallet} transaction builder + signed by both hardware wallets serially.

### TX 1 — FTNS approve(Router, ftns_amount)

| Field | Value |
|---|---|
| From | Foundation Safe `0x91b0...5791` |
| To | FTNS token `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| Function | `approve(address spender, uint256 amount)` |
| spender | AerodromeRouter (re-verify per §3.3) |
| amount | Council-ratified FTNS seed amount (wei: amount * 10^18) |
| ETH value | 0 |
| Expected gas | ~50K |

### TX 2 — USDC approve(Router, usdc_amount)

| Field | Value |
|---|---|
| From | Foundation Safe `0x91b0...5791` |
| To | USDC `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` |
| Function | `approve(address spender, uint256 amount)` |
| spender | AerodromeRouter (re-verify per §3.3) |
| amount | Council-ratified USDC seed amount (wei: amount * 10^6 — USDC has 6 decimals) |
| ETH value | 0 |
| Expected gas | ~50K |

### TX 3 — Router.addLiquidity(...)

Aerodrome Router `addLiquidity` ABI:

```solidity
function addLiquidity(
    address tokenA,
    address tokenB,
    bool stable,
    uint amountADesired,
    uint amountBDesired,
    uint amountAMin,
    uint amountBMin,
    address to,
    uint deadline
) external returns (uint amountA, uint amountB, uint liquidity);
```

| Field | Value |
|---|---|
| From | Foundation Safe `0x91b0...5791` |
| To | AerodromeRouter (re-verify per §3.3) |
| Function | `addLiquidity(address,address,bool,uint256,uint256,uint256,uint256,address,uint256)` |
| tokenA | FTNS `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| tokenB | USDC `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` |
| stable | `false` (volatile — FTNS price discovery, xy=k) |
| amountADesired | Council-ratified FTNS amount in wei (matches TX 1 approve) |
| amountBDesired | Council-ratified USDC amount in wei (matches TX 2 approve) |
| amountAMin | `amountADesired` (first liquidity has no slippage — pool is empty) |
| amountBMin | `amountBDesired` (first liquidity has no slippage) |
| to | Foundation Safe `0x91b0...5791` (LP-token recipient) |
| deadline | Unix epoch seconds; recommend `block.timestamp + 1800` (30-min window) |
| ETH value | 0 |
| Expected gas | ~250K |

**Slippage handling note:** for an initial liquidity provision, `amountAMin`/`amountBMin` SHOULD equal the desired amounts because no prior pool exists to slip against. If the pool already exists (e.g., a frontrunner created it before our ceremony), Aerodrome's Router will use the existing reserves to compute the proportional deposit — in which case our ceremony should ABORT, escalate to council, and re-plan.

---

## 5. Calldata generation

Use the helper script (shipped alongside this doc):

```bash
node contracts/scripts/build-aerodrome-pool-seed-tx.js \
  --network base-mainnet \
  --ftns-amount 1000000 \
  --usdc-amount 250000 \
  --safe-address 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
  --router 0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43 \
  --out /tmp/aerodrome-seed-bundle.json
```

The script emits a JSON bundle with three transactions in Safe{Wallet} transaction-builder import format. Operator imports `/tmp/aerodrome-seed-bundle.json` into Safe{Wallet} UI → queues all 3 txs at once → signs serially with 2-of-3 hardware wallets.

---

## 6. Post-flight verification

Run the verification script immediately after the 3rd tx confirms:

```bash
node contracts/scripts/verify-aerodrome-pool-seed.js \
  --network base-mainnet \
  --safe-address 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791
```

The script asserts:
1. Pool exists at `AerodromePoolFactory.getPool(FTNS, USDC, false)` and returns a non-zero address
2. Pool's `getReserves()` matches the seeded amounts (within rounding)
3. Pool's `token0()` + `token1()` resolve to {FTNS, USDC} (order is enforced by sort)
4. `LP_token.balanceOf(FoundationSafe)` > 0 (Safe holds LP tokens)
5. Safe's FTNS balance decreased by exactly the seeded amount
6. Safe's USDC balance decreased by exactly the seeded amount

**On any assertion failure:** STOP. Do NOT update the env var. File an incident per `docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md`. The pool exists either way; the question is whether the seed completed as intended.

---

## 7. Post-ceremony operator action

After verification passes:

1. Record the pool address in `prsm/config/networks.py` MAINNET config (new field: `aerodrome_usdc_ftns_pool`)
2. Set `AERODROME_USDC_FTNS_POOL_ADDRESS` env on all operator nodes (production deployment manifest, runbook §6 update)
3. Set `PRSM_FTNS_USD_RATE` to a sensible bootstrap value (the implied seed price) so the env-var fallback closes cleanly when operators migrate to the live pool quoter
4. Notify operators via Foundation operator broadcast: pool live + env-var update instructions
5. Draft + ratify `PRSM-CR-2026-06-XX-2`: ratifies post-ceremony state (block number, pool address, seeded amounts, signers, gas spent)
6. Update `docs/2026-04-27-cumulative-audit-prep.md` with §7.X entry referencing the ceremony
7. Update `Tokenomics §3.5` with the on-chain pool address (replaces "pool seeded per Vision gantt 2026-06-15" placeholder)
8. Update `PARTICIPANT_GUIDE.md` rate-disclosure paragraph to reference the live pool

---

## 8. Risk inventory

| Risk | Mitigation |
|---|---|
| Pool already exists (frontrunner) | Verification step §3.3 includes `AerodromePoolFactory.getPool(FTNS, USDC, false)` — if non-zero, abort + escalate. If frontrunner seeded with hostile price, council decides whether to add liquidity at the frontrunner's price (cheap insurance) or wait for arbitrage to settle. |
| Approve race-condition (USDC approve drift) | USDC's `approve()` is the standard ERC-20 variant; the increment-approve pattern isn't needed because no prior allowance exists for the Router. If a prior allowance somehow exists, set to 0 first (extra tx) before re-approving. |
| Sandwich attack on `addLiquidity` | First-liquidity provisions cannot be sandwiched — there's no prior pool state for the attacker to exploit. Subsequent liquidity adds CAN be sandwiched; that's a downstream concern, not this ceremony's. |
| Wrong slippage tolerance | First-liquidity: `amountAMin = amountADesired` (zero slippage acceptable since pool is empty). If pool exists, abort. |
| Hardware wallet failure mid-ceremony | OneKey is the reserve signer. If Ledger fails, OneKey + Trezor satisfies 2-of-3. If Trezor fails, OneKey + Ledger satisfies. If two fail, ABORT and reschedule. |
| Foundation Safe insufficient balance | §3.4 pre-flight balance checks. If short, council decision: top up before ceremony, OR reduce seed amount within envelope. |
| Aerodrome contract addresses changed | §3.3 re-verification step. If Aerodrome has redeployed contracts (e.g., V2 router), the ceremony pauses for an updated CR. |
| Founder unavailable on ceremony date | Reschedule. No urgency — Phase 5 commission gates on this, but Phase 5 has no external SLA. |
| Network congestion / failed tx | Re-broadcast with higher gas. Safe{Wallet} UI supports gas bumps. |
| Pool gets immediately drained by arbitrage | Expected behavior — the implied seed price is the founder's announcement, not a market price; arbitrageurs will pull it toward the prevailing OTC/secondary rate. Mitigation = council picks a defensible implied price within the Tokenomics §3.7 envelope. |

---

## 9. Non-scope (explicit)

The following are **NOT** authorized by this ceremony plan + any companion CR that ratifies it:

1. **Continuous market-making.** Foundation provides initial liquidity once; does NOT MM continuously (Tokenomics §3.5 invariant).
2. **veAERO bribe / gauge whitelisting.** Separate governance ceremony; gated on Foundation acquiring a veAERO position.
3. **Coinbase CDP commissioning.** Sibling external-party engagement; this ceremony is necessary but not sufficient.
4. **LP-token transfers.** Foundation Safe holds the LP tokens; no authorization to transfer them out exists. Any future transfer requires its own CR.
5. **Multiple pool tiers.** Only ONE pool (volatile USDC-FTNS) is in scope. No stable-pool variant.
6. **Withdrawal authority.** Removing liquidity later requires its own CR.
7. **Composite operations.** No interactions with Aerodrome's gauge / voter contracts. Just `addLiquidity` to a fresh pool.

---

## 10. Cancellation criteria

ABORT the ceremony (do NOT proceed to TX 3) if any of:

- Pool already exists with non-trivial reserves at ceremony time (frontrunner)
- USDC balance on Safe is below the council-ratified seed amount
- FTNS balance on Safe is below the council-ratified seed amount
- Either hardware-wallet signature attempt fails after 3 retries
- Sepolia rehearsal hasn't been completed in the last 7 days
- Aerodrome canonical contract addresses don't match expectations + can't be re-verified
- Founder concentration is degraded (illness, distraction, external interruption)

If aborting: revert any TX 1 / TX 2 approvals by re-issuing `approve(Router, 0)` to clear the allowance. Document the abort + reschedule in a post-mortem doc.

---

## Sign-off

| Approver | Role | Date | Signature method |
|---|---|---|---|
| _________ | Founder | _________ | Hand-written + Ledger acceptance test on Sepolia |
| _________ | Council signer 2 | _________ | Trezor acceptance test on Sepolia |
| _________ | Council signer 3 (reserve) | _________ | OneKey availability confirmed |
