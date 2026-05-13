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

### 2.1 Sepolia Safe

PRSM operates a Sepolia Foundation Safe mirror at `<SEPOLIA_SAFE_ADDRESS — fill in from prior rehearsal docs>`. If no mirror exists, deploy a fresh 2-of-3 Sepolia Safe with the same signer set (Ledger Sepolia testnet derivation + Trezor Sepolia derivation + OneKey reserve).

- [ ] Sepolia Safe address known
- [ ] All 3 signers' Sepolia-testnet addresses configured as Safe owners
- [ ] Sepolia ETH balance ≥ 0.05 on Safe (faucet: https://www.alchemy.com/faucets/ethereum-sepolia)

### 2.2 Sepolia FTNS

PRSM has Sepolia FTNS test contracts from prior rehearsals (per `project_t10_a08_2026_05_07.md` / `prsm/config/networks.py` SEPOLIA config). If no FTNS exists on Sepolia, deploy via:

```bash
cd contracts
PRSM_NETWORK=sepolia npx hardhat run scripts/deploy-ftns-token-simple.js --network sepolia
```

- [ ] Sepolia FTNS deployed; address recorded
- [ ] Sepolia Safe holds ≥ 1M FTNS test units (deployer mints to Safe in deploy script)

### 2.3 Sepolia USDC mock

Sepolia doesn't have canonical Circle USDC. Deploy a MockERC20 with 6 decimals + USDC-like behavior:

```bash
cd contracts
PRSM_NETWORK=sepolia npx hardhat run scripts/deploy-mock-usdc.js --network sepolia
```

If the script doesn't exist, deploy `contracts/test/MockERC20.sol` and rename the instance "Mock USDC" in deployment metadata.

- [ ] Sepolia mock-USDC deployed; address recorded
- [ ] Sepolia Safe holds ≥ 250K USDC test units (mint to Safe post-deploy)

### 2.4 Aerodrome Sepolia mirror

Aerodrome does NOT have a Sepolia deployment as of this drafting. Two options:

**Option A: Base Sepolia.** Aerodrome may have a Base Sepolia testnet deployment. Verify via `https://aerodrome.finance/docs` at rehearsal time. If exists, use Base Sepolia (chainid 84532) instead of mainnet Sepolia.

**Option B: Local Anvil fork of Base mainnet.** Fork Base mainnet block N into a local Anvil node; that mirror has the real Aerodrome contracts + the real USDC, and we top up the Safe with `anvil_setBalance` + `anvil_impersonateAccount` to mint test FTNS. This rehearsal is closest to mainnet behavior but most invasive to set up.

```bash
# Option B setup
anvil --fork-url https://mainnet.base.org --port 8545 &
# In another terminal:
cast send 0x91b0...5791 --value 1ether --from <hardhat-default-account>
# ... impersonate Foundation Safe, send mock txs
```

- [ ] Decided: Option A (Base Sepolia) OR Option B (Anvil fork) OR defer rehearsal to mainnet-day-only (NOT RECOMMENDED — see §1)
- [ ] If Option A: Aerodrome Sepolia Router + Factory addresses recorded
- [ ] If Option B: Anvil fork running + Foundation Safe impersonation working

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
