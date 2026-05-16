# FTNS-side Mainnet TX Test Plan

**Status:** READY FOR EXECUTION (user-funded; not yet run)
**Coverage matrix column:** OC (Real on-chain mutations) — last
untested dimension as of sprint 495
**Authored:** sprint 496, 2026-05-16
**Prerequisites:** Real FTNS tokens + ETH for gas

---

## Why this matters

Sprints 466 + 467 verified ETH-side transactions:
- Sprint 466: 2 testnet self-transfers + 2 contract reads
  (Sepolia, chainId 84532)
- Sprint 467: 1 mainnet self-transfer + INV-RD-3 + INV-EC-1
  PASS against live Base mainnet contracts (chainId 8453)

But **no FTNS-side mainnet TX has ever been executed**. Vision
§11 / §13 claims that depend on real FTNS movement remain
unattested:

| Claim | Surface | Verification today |
|-------|---------|--------------------|
| FTNS transfer between wallets works on-chain | `/balance/onchain` + RoyaltyDistributor | Schema-pinned only; never executed |
| Creator stake commission writes to chain | `/marketplace/creator-stake/stake` | Schema-pinned only; commissioning surface returns PENDING_COMMISSION |
| Royalty distribution credits creator's on-chain balance | `/wallet/royalty/claim` | Returns 503 "RoyaltyDistributor client not wired" without `PRSM_NETWORK=mainnet` + private key |
| Settler bond is REAL FTNS, not just a number | `/settler/register` (sprint 492 F32 fix attested local-FTNS lock; on-chain lock is separate) | Local lock attested; on-chain not |
| EmissionController distributes FTNS per 4-year epoch | `/admin/formal-verification/check?contract=emission_controller` | Constant verified (sprint 467); no actual emission claimed-against |

Until at least the first three are executed, the §11 economic
layer is **schema-verified but operationally unattested at the
FTNS layer.**

---

## Test wallet from sprint 467

| Field | Value |
|-------|-------|
| Address | `0x2Fd48D2d026bEf7563C85c647674cb945C4d4f57` |
| Current ETH balance (Base mainnet, last checked sprint 467) | `0.0004999874 ETH` |
| Current FTNS balance | `0` (never been funded) |
| Private key env var | `FTNS_WALLET_PRIVATE_KEY` |

This wallet has enough ETH for ~80-100 small TX at current
Base gas prices (~0.006 Gwei observed sprint 467). FTNS-side
TX still requires the wallet to HOLD FTNS first.

---

## Pre-conditions

1. **Acquire FTNS.** Two paths:
   - **Coinbase CDP onramp** (Phase 5 surface — sprint 451 verified the quote endpoint). Requires KYC commissioning. Best for $50+ purchases.
   - **Aerodrome USDC/FTNS pool swap** (Vision gantt 2026-06-15 seeding ceremony). Not yet live.
   - **Direct FTNS transfer from a known holder** (e.g., Foundation distribution if available).
   
   **Minimum recommended:** $5 worth of FTNS for the full
   test sequence below (covers ~20 small TX with margin).
   At a notional $0.01/FTNS that's 500 FTNS. At higher
   prices the test budget scales.

2. **Wallet funded.** Send FTNS to `0x2Fd48D2d026bEf7563C85c647674cb945C4d4f57`.
   Verify via `prsm wallet info` or `/balance/onchain`:
   ```
   {
     "address": "0x2Fd48D...",
     "balance_ftns": <expected>,
     ...
   }
   ```

3. **Daemon env vars set:**
   ```bash
   export PRSM_NETWORK=mainnet
   export BASE_RPC_URL=https://mainnet.base.org
   export PRSM_ONCHAIN_FTNS=1
   export PRSM_ONCHAIN_PROVENANCE=1     # for §11 royalty leg
   export FTNS_WALLET_PRIVATE_KEY=<test_wallet_priv_key>
   export PRSM_ROYALTY_DISTRIBUTOR_ADDRESS=0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e
   export PRSM_FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5
   ```

4. **Sanity check.** Restart daemon. Confirm `/health/detailed`:
   ```
   "ftns_ledger.connected_address" == test wallet address
   "ftns_ledger.canonical_match" == true
   "royalty_distributor.available" == true   # was false in sprint 467
   ```

---

## Test sequence (staged smallest → largest)

Each TX builds on the previous. Stop at any failure +
investigate before continuing.

### TX-1: FTNS self-transfer (smallest possible value)

Sends FTNS from the test wallet to itself. Minimum FTNS
move (1 wei worth) — proves the on-chain FTNS path works
without any balance change.

**Command:**
```bash
prsm wallet transfer-ftns \
  --to 0x2Fd48D2d026bEf7563C85c647674cb945C4d4f57 \
  --amount 0.000001
```
or HTTP:
```bash
curl -X POST http://127.0.0.1:8000/wallet/transfer/onchain \
  -H "Content-Type: application/json" \
  -d '{
    "to_address": "0x2Fd48D...",
    "amount_ftns": 0.000001
  }'
```

**Note:** If `/wallet/transfer/onchain` doesn't exist yet
(only `/wallet/transfer/gasless` was indexed), use the
RoyaltyDistributor's direct path or expose a real onchain
transfer surface as a sub-deliverable of this sprint.

**Expected outcome:**
- 200 with `{tx_hash: "0x...", block_number: N, status: 1}`
- Self-transfer → balance unchanged net (minus gas)
- BaseScan: confirm TX is mined + status=success
- Pin tests: TX-1's tx_hash gets recorded in PRSM_Testing.md
  with a sprint number for OC ✅ promotion

**Cost:** ~21,000 gas × 0.006 Gwei ≈ $0.0004

---

### TX-2: FTNS transfer to a different address

Move 1 FTNS to a 2nd address that the test wallet controls.

**Prerequisites:** Generate a second test wallet (or use
the node's identity address).

**Expected outcome:**
- 200 with tx_hash
- Source wallet balance drops by 1 FTNS (+ gas)
- Destination wallet balance increases by 1 FTNS
- `/balance/onchain` reflects both sides

**Cost:** ~21k gas (ERC-20 transfer)

---

### TX-3: Claim royalty (the §11 user-facing surface)

Most operators won't trigger TX-1/TX-2 directly. The real
production-relevant path is `prsm node claim-royalty` /
`POST /wallet/royalty/claim`. This invokes
`RoyaltyDistributor.claim()` on the canonical contract.

Today (sprint 471) this returns 503 because
`PRSM_ROYALTY_DISTRIBUTOR_ADDRESS` isn't wired by default.
With the env var set + wallet funded:

```bash
curl -X POST http://127.0.0.1:8000/wallet/royalty/claim
```

**Expected outcome:**
- If creator has unclaimed royalties: 200 with tx_hash;
  funds move from RoyaltyDistributor contract to wallet
- If no unclaimed royalties: 400 "nothing to claim" (clean
  empty-state, NOT 503)
- Pin: tx_hash recorded; `/admin/royalty-dispatch-history`
  shows the entry

**Cost:** ~50-80k gas (contract call with claim logic)

**Note:** to actually have unclaimed royalties, the test
wallet must be the creator of some content that's been
accessed. The cleanest E2E:
1. Upload content with `creator_eth_address` = test wallet
2. Trigger N retrievals from a different operator (multi-
   node bench — sprint 456-457 setup) so the royalty leg
   fires
3. Claim — test wallet gets the royalty

---

### TX-4: Stake commissioning (Vision §14 anti-Sybil)

Sprint 492 F32 fixed the LOCAL bond check. The on-chain
side commissions a real CreatorStakeClient — which today
returns PENDING_COMMISSION when contract address isn't
wired.

```bash
curl -X POST http://127.0.0.1:8000/marketplace/creator-stake/stake \
  -H "Content-Type: application/json" \
  -d '{
    "creator_eth_address": "0x2Fd48D...",
    "amount_ftns": 10
  }'
```

**Expected outcome:**
- Stake tx_hash recorded on chain
- `/marketplace/creator-stake/0x2Fd48D...` shows
  `balance_wei: 10000000000000000000, commissioned: true,
  high_tier_eligible: false` (10 FTNS < minimum 10000)
- A 10000-FTNS stake would flip `high_tier_eligible: true`
  (but costs more)

**Cost:** ~50-80k gas

---

### TX-5: Settler bond on-chain (deferred follow-on)

Sprint 492 F32 fixed the local bond check at
`_StakingFTNSAdapter.lock_tokens`. The on-chain settler
registration is a separate flow. Out of scope for
sprint 496's INITIAL test plan because:
- min bond is 10,000 FTNS (~$100+ depending on price)
- requires multi-sig coordination per Vision §11

Document for follow-on when the Foundation Safe ceremony
date approaches.

---

## Pin tests

After each TX completes:

1. **Record the tx_hash** in PRSM_Testing.md against the
   relevant row. Promote from 🟢 / ⚠️ → ✅ with sprint 496
   attribution.
2. **Update the dogfood-findings doc** with the live
   evidence (tx_hash + BaseScan URL).
3. **Static pin test** in `tests/unit/test_sprint_496_*.py`
   that asserts the BaseScan URL is included in the
   PRSM_Testing.md notes column (an audit-trail invariant —
   ✅ promotions on the OC column must cite a real tx_hash).

---

## Rollback / cleanup

FTNS-side TX are **irreversible**. There is no
"refund" path on chain.

Defenses:
- Always start with the smallest possible amount (TX-1's
  0.000001 FTNS = 1e12 wei = essentially zero).
- After TX-1 success, scale up only by 10x per step (TX-2
  uses 1 FTNS; TX-3/4 use 10 FTNS).
- For TX-4 (commissioning), keep the test stake at the
  minimum so the bond is recoverable via the 30-day unbond
  cooldown.

---

## Cost summary

| TX | Amount FTNS moved | Gas (Base mainnet) | USD cost (approx) |
|----|-------------------|--------------------|--------------------|
| TX-1 self-transfer | 0.000001 | 21k @ 0.006 Gwei | $0.0004 |
| TX-2 transfer to 2nd wallet | 1 | 21k | $0.0004 |
| TX-3 claim royalty | (received) | 50-80k | $0.001 |
| TX-4 stake 10 FTNS | 10 | 50-80k | $0.001 |
| **Total** | ~11 FTNS spent | ~170k gas total | ~$0.003 ETH |

Plus the **funded amount** (recommend $5 in FTNS as float).

---

## Why not testnet?

Sprint 466 already exercised Sepolia testnet for ETH-side
TX. **For FTNS-side TX, testnet has a 100M FTNS supply
that's founder-airdropped (not faucet-available)** — see
sprint 466 closing note:
> "Testnet-FTNS-distribution is founder-airdropped per
> config (not faucet) — Sprint 466 scope covers ETH-side
> TX + read-only contract verification; FTNS-side TX
> deferred."

Testnet FTNS would require coordination with the
Foundation to get tokens, at which point mainnet is just
as easy + the actual proof is what we care about.

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| RPC endpoint refuses TX | Low | Multi-fallback: mainnet.base.org + alchemy.com + ankr.com |
| Gas price spike during TX | Low | Use EIP-1559 with `maxFeePerGas` cap |
| Wrong network selected (catastrophe — burn funds to 0x0) | Critical | `chain_id == 8453` invariant pin pre-broadcast |
| Private key leaked via logs | Critical | F-arc memory note: never log `FTNS_WALLET_PRIVATE_KEY`. Verify before TX-1 by `grep "FTNS_WALLET_PRIVATE_KEY" /tmp/prsm-daemon-*.log` returning empty |
| TX nonce conflict | Low | Daemon uses `web3.eth.get_transaction_count` per call |

---

## Decision point

**Ready to execute when:**
1. Sprint 496 commits this runbook (✅ this commit)
2. User decides to fund the test wallet with FTNS
3. User notifies; the on-chain TX sequence executes per
   the above

Until then, the OC column of the coverage matrix remains
the only major dimension WITHOUT live attestation.
