# Aerodrome Pool-Seed Ceremony — Sepolia Rehearsal Runbook

**Document identifier:** AERODROME-POOL-SEED-SEPOLIA-RUNBOOK-1
**Version:** 0.1 Draft
**Status:** Pre-ceremony rehearsal aid. Companion to `2026-06-15-aerodrome-pool-seed-ceremony-plan.md`.
**Date drafted:** 2026-05-13
**Drafting authority:** PRSM founder
**Expected duration:** 90 minutes end-to-end (15 min setup + 30 min rehearsal + 30 min verification + 15 min teardown)

> Same shape as `2026-05-09-A-08-sepolia-rehearsal-runbook.md`. Rehearse the mainnet 3-tx sequence on a Sepolia mirror so the muscle memory + hardware-wallet UX is locked in before mainnet day.

---

## 1. Why rehearse

The mainnet ceremony moves council-ratified treasury funds ($250K-$1M USDC + matched FTNS per Tokenomics §3.7). Three failure modes that ONLY surface in rehearsal:

1. **Hardware-wallet UI quirks.** Ledger / Trezor render different transaction summaries; some addLiquidity calldata is hard to read on the device screen. Rehearsal teaches the founder what to look for.
2. **Calldata correctness.** The 3-tx sequence is constructed off-chain by `build-aerodrome-pool-seed-tx.js`. A buggy script could produce a syntactically-valid but semantically-wrong calldata that Safe{Wallet} happily queues. Rehearsal catches this on a network where the worst outcome is testnet ETH/USDC/FTNS lost.
3. **Aerodrome canonical contract drift.** If Aerodrome has redeployed contracts on Base mainnet but our doc still references old addresses, the rehearsal pool will fail to materialize, surfacing the discrepancy without burning mainnet gas.

**Rehearsal pass criteria:**
- All 3 txs return status=1 on Sepolia
- LP tokens visible in Sepolia Safe
- Verification script's 6 assertions all pass
- Founder reports comfort with the Ledger + Trezor flows (no surprises on device screens)

---

## 2. Prerequisites

### 2.1 Sepolia Safe (state survey 2026-05-13)

PRSM has an existing Sepolia rehearsal Safe at `0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889` (Base Sepolia, chainid 84532, Safe v1.4.1, 1-of-1 with sole owner Ledger `0xA3683EDDBed6622f132698D7DC36a7C2DAFe4Ed3`). This is the same Safe used for the 2026-05-09 A-08 v2 rehearsal.

**Threshold reality check.** This Safe is 1-of-1 (test wallet acting as Safe-equivalent per `2026-05-09-A-08-v2-redeploy-ceremony-plan.md` §8) rather than 2-of-3 like the mainnet Foundation Safe. This is acceptable: the rehearsal exercises Safe-UI compose + hardware-wallet signing UX — both invariant under threshold. The 2-of-3 mainnet threshold is then a "more of the same" multiplier on the signing UX validated here.

State verified 2026-05-13:
- [x] Safe address: `0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889`
- [x] Owner: `0xA3683EDDBed6622f132698D7DC36a7C2DAFe4Ed3` (Ledger)
- [ ] **GAP: Safe ETH balance is 4e-05 ETH — needs faucet top-up to ≥ 0.05 ETH** (faucet: https://www.alchemy.com/faucets/base-sepolia or https://docs.base.org/tools/network-faucets)

### 2.2 Sepolia FTNS (state survey 2026-05-13)

Existing Sepolia FTNS deployments (per `contracts/deployments/`):

- **Latest (recommended for this rehearsal):** `0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a` — `FTNSTokenSimple` UUPS proxy on Base Sepolia, deployed 2026-05-07 (manifest: `phase1-ftns-base-sepolia-1778159554505.json`)
- **A-08 rehearsal alternative:** `0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a` — used 2026-05-09 if needed

Initial 100M supply on the recommended FTNS was minted to deployer `0xCCAc7b21695De068979b1ca47B0cfBD328654220`, NOT the Safe.

- [x] Sepolia FTNS address known: `0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a`
- [ ] **GAP: Transfer ≥ 2M FTNS from deployer to Sepolia Safe `0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889`** before rehearsal. Use Sepolia FTNS's built-in `transfer(safe, 2_000_000 * 10^18)` from deployer EOA.

### 2.3 Sepolia USDC mock (state survey 2026-05-13)

Sepolia doesn't have canonical Circle USDC. PRSM ships `contracts/contracts/test/MockUSDC.sol` (6-decimal USDC stand-in; permissionless mint; intentionally distinct from the 18-decimal MockERC20 so a decimals-encoding bug surfaces on Sepolia rather than masking) + `contracts/scripts/deploy-mock-usdc.js` deploy + mint helper.

```bash
cd contracts
PRIVATE_KEY=<sepolia-deployer-key> \
SEPOLIA_SAFE=0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889 \
MINT_AMOUNT=500000 \
npx hardhat run scripts/deploy-mock-usdc.js --network base-sepolia
```

- [ ] **GAP: MockUSDC not yet deployed** — run the script above
- [ ] Resulting address recorded in `contracts/deployments/mock-usdc-base-sepolia-<ts>.json`
- [ ] Sepolia Safe holds 500K mUSDC (script mints automatically as part of deploy)

### 2.4 Aerodrome Sepolia mirror (verified 2026-05-13)

**Aerodrome has NO Base Sepolia testnet deployment.** Confirmed via `aerodrome-finance/contracts` GitHub — only Base mainnet (chainid 8453) is supported. The previous Option A (Base Sepolia mirror) is therefore NOT AVAILABLE.

This forces **Option B (Anvil fork)** as the only path that exercises the real Aerodrome contracts. The trade-off: Anvil fork runs against impersonated Foundation Safe (not actual Sepolia Safe), which means the hardware-wallet signing UX validates via the actual mainnet Foundation Safe in a one-off test sign (NOT a real ceremony) BEFORE the rehearsal — see §3.1 below for the revised sequence.

```bash
# Option B setup — fork Base mainnet at a recent block
anvil --fork-url https://mainnet.base.org --port 8545 --block-time 1 &

# In another terminal: impersonate Foundation Safe so we can send txs from it
cast rpc anvil_impersonateAccount 0x91b0000000000000000000000000000000005791 \
  --rpc-url http://127.0.0.1:8545

# Top up the impersonated Safe with ETH for gas
cast rpc anvil_setBalance 0x91b0000000000000000000000000000000005791 \
  0x56BC75E2D63100000 --rpc-url http://127.0.0.1:8545  # 100 ETH

# Confirm FTNS balance on the impersonated Safe (should already hold 100M from
# the 2026-05-06 mainnet migration captured in the fork state)
cast call 0x5276a3756C85f2E9e46f6D34386167a209aa16e5 \
  "balanceOf(address)(uint256)" 0x91b0000000000000000000000000000000005791 \
  --rpc-url http://127.0.0.1:8545
```

- [ ] Anvil fork running on port 8545
- [ ] Foundation Safe `0x91b0...5791` impersonated
- [ ] Safe's USDC top-up: `cast rpc anvil_setStorageAt` to write a 500K USDC balance into the Safe's slot (USDC uses storage slot 9 for balances mapping; key = keccak256(safe_addr ++ slot)) — OR mint via `anvil_impersonateAccount` of USDC's role holders if any. For convenience the script `setup-anvil-aerodrome-rehearsal.sh` (TBD ship in this packet) automates this.

**Hardware-wallet UX validation under Anvil.** Because Anvil signs from impersonated accounts (no real EOA), we CAN'T exercise Ledger/Trezor signing against the Anvil fork directly. The rehearsal therefore splits in two:

1. **Anvil fork (Option B):** exercises calldata + bundle import + the 3-tx execution sequence against real Aerodrome contracts. Validates that calldata is correct + verify-script assertions pass. NO hardware-wallet involvement.
2. **Sepolia Safe sign-test:** exercises Ledger/Trezor signing against the Sepolia Safe with a TRIVIAL tx (e.g., send 0.0001 ETH to self) — validates that the device flow works end-to-end without contract complexity. Run this immediately before the Anvil portion so muscle memory is fresh.

The mainnet ceremony is the combination: real Aerodrome (from Anvil rehearsal) + real hardware signing (from Sepolia sign-test).

### 2.5 Hardware

- [ ] Ledger: ETH app open, testnet support enabled (Settings → Blind signing ON if needed for complex calldata)
- [ ] Trezor: ETH app open, testnet enabled
- [ ] OneKey: available as reserve

---

## 3. Rehearsal procedure

### 3.1 Generate calldata

```bash
cd contracts
# Match the mainnet command but with testnet addresses
node scripts/build-aerodrome-pool-seed-tx.js \
  --network sepolia \
  --ftns-token <SEPOLIA_FTNS_ADDRESS> \
  --usdc-token <SEPOLIA_MOCK_USDC_ADDRESS> \
  --router <SEPOLIA_AERODROME_ROUTER> \
  --ftns-amount 1000000 \
  --usdc-amount 250000 \
  --safe-address <SEPOLIA_SAFE_ADDRESS> \
  --out /tmp/aerodrome-sepolia-bundle.json
```

- [ ] Bundle file produced; inspect it manually
- [ ] All 3 txs have correct `to` + `data` + `value=0`

### 3.2 Import into Safe{Wallet} (Sepolia testnet UI)

- [ ] Safe{Wallet} on Sepolia UI loaded
- [ ] Transaction Builder app opened
- [ ] `/tmp/aerodrome-sepolia-bundle.json` imported
- [ ] All 3 txs visible in queue

### 3.3 Sign + execute TX 1 (FTNS approve)

- [ ] Connect Ledger via WalletConnect or browser extension
- [ ] Sign tx in Safe UI
- [ ] Switch to Trezor; sign tx
- [ ] Execute tx; wait for confirmation
- [ ] Check Basescan: `FTNS.allowance(Safe, Router)` equals seed amount
- [ ] Note device-screen text for both wallets (write down literally what appeared, for mainnet-day reference)

### 3.4 Sign + execute TX 2 (USDC approve)

- [ ] Same flow as TX 1 with the Mock USDC contract
- [ ] Confirm allowance set

### 3.5 Sign + execute TX 3 (addLiquidity)

- [ ] Same signing flow. **This is the critical signing — verify the device screens carefully.**
- [ ] Pay special attention to: tokenA address, tokenB address, amounts, recipient address (must be Safe)
- [ ] Execute. Wait for confirmation.
- [ ] Pool should now exist

### 3.6 Run verification

```bash
node contracts/scripts/verify-aerodrome-pool-seed.js \
  --network sepolia \
  --safe-address <SEPOLIA_SAFE_ADDRESS>
```

- [ ] All 6 assertions pass
- [ ] Pool address recorded
- [ ] Reserves match seeded amounts
- [ ] LP token balance on Safe > 0

---

## 4. Failure-mode notes

Record these for mainnet-day reference:

| Symptom | Likely cause | Mainnet response |
|---|---|---|
| Ledger refuses to display addLiquidity calldata | Blind-signing disabled | Enable blind signing on Ledger (Settings → Ethereum) before mainnet day |
| Safe UI shows different gas estimate than calldata implies | Gas oracle drift | Acceptable on Sepolia. On mainnet, sanity-check via independent gas estimator (Basescan, etherscan) before signing |
| TX 3 reverts with `INSUFFICIENT_LIQUIDITY_MINTED` | Pool already exists OR amounts mismatch | ABORT mainnet ceremony; investigate. May indicate frontrunner. |
| TX 3 reverts with `INSUFFICIENT_A/B_AMOUNT` | `amountAMin` / `amountBMin` too tight against actual pool ratio | ABORT mainnet ceremony; pool exists with different ratio than expected. Council decision on whether to seed at the existing ratio. |
| Tx submitted but never confirms | Network congestion | Wait, then re-broadcast with higher gas via Safe UI |
| Wallet displays "wrong network" | App selected wrong chain | Switch the wallet's chain selector to Base (8453) for mainnet, Sepolia (11155111) for rehearsal |

---

## 5. Post-rehearsal action

- [ ] Pool address from rehearsal recorded in rehearsal artifact file
- [ ] Founder writes 1-paragraph rehearsal-lessons doc (`2026-06-XX-aerodrome-pool-seed-sepolia-rehearsal-lessons.md`) capturing surprises + hardware-wallet UX notes
- [ ] Sleep on it. Mainnet ceremony scheduled no earlier than 24 hours after rehearsal pass.
- [ ] If rehearsal failed at any step, root-cause analysis before mainnet ceremony.

---

## 6. Cleanup

After rehearsal:

- [ ] Optionally revoke Sepolia Router approvals: `FTNS.approve(Router, 0)` + `USDC.approve(Router, 0)` (low risk on testnet but good hygiene)
- [ ] Sepolia LP tokens left in Safe; can be withdrawn separately or left in place
- [ ] Rehearsal bundle file (`/tmp/aerodrome-sepolia-bundle.json`) archived to a personal notes location for reference
