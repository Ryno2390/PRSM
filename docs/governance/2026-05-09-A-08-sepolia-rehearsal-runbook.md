# A-08 RoyaltyDistributor v2 — Sepolia Rehearsal Runbook

**Document identifier:** A-08-SEPOLIA-REHEARSAL-1
**Status:** Pre-execution; founder runs each step + reports outputs.
**Date drafted:** 2026-05-09
**Companion docs:**
- `docs/governance/2026-05-09-A-08-v2-redeploy-ceremony-plan.md` (mainnet plan)
- `docs/governance/A-08-recoverStranded-design.md` (ADR)

---

## Purpose

Execute the full 4.1-4.8 mainnet ceremony sequence end-to-end on Base Sepolia. Catches script bugs, env-var mistakes, transferOwnership flow weirdness, and `recoverStranded` end-to-end correctness BEFORE mainnet gas is paid. Per ceremony plan §2.4 + §8, this rehearsal must complete within 7 days preceding mainnet ceremony.

**Time budget:** ~2 hours including pauses for output review.

**What you need:**

1. Sepolia private key with ≥ 0.05 Sepolia ETH (testnet faucet: https://sepoliafaucet.com or Coinbase faucet at https://portal.cdp.coinbase.com/products/faucet)
2. A second Sepolia address to act as the "Safe-equivalent" target for transferOwnership (can be a second hot key from MetaMask, or a real Safe deployed on Sepolia if you have one). Single-sig is fine for rehearsal.
3. `BASESCAN_API_KEY` already in env (same key works for mainnet + Sepolia)
4. Repo root checked out at the head commit

---

## Step 0 — Pre-flight

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM

# Verify clean state
git status
# Expect: clean working tree on main

cd contracts
npx hardhat compile
# Expect: "Nothing to compile" (clean) OR "Compiled N Solidity files successfully"

npx hardhat test test/RoyaltyDistributor*.js
# Expect: 20 passing
```

If any of the above fail, STOP and investigate before running rehearsal.

---

## Step 1 — Set rehearsal env vars

Open a fresh shell. Set these env vars:

```bash
# Sepolia deployer hot key (testnet burner; never share)
# Format check: must start with 0x and be 66 chars total
export PRIVATE_KEY="0x<YOUR_SEPOLIA_HOT_KEY_PRIVATE_KEY>"

# Sepolia constructor args
# These are the test addresses from the May 5 deploy. If you'd rather
# use fresh ones, deploy a new test FTNS + Registry first; otherwise
# the existing instances work fine for rehearsal purposes.
export FTNS_TOKEN_ADDRESS="0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a"
export NETWORK_TREASURY="0x40C81867987e1e07E5C8c9B3395aBE38EE95C911"

# Reuse existing Sepolia ProvenanceRegistry — same as mainnet ROYALTY_ONLY pattern.
# (For the rehearsal we don't need V2 — V1 registry is fine since the
# RoyaltyDistributor only cares that registry is a contract with the
# expected interface; canonical-pin check is mainnet-only.)
export ROYALTY_ONLY=1
export EXISTING_PROVENANCE_REGISTRY="0x2911f9a0a02896486CdF59d6d369764841DC0eA4"
export FORCE_NONCANONICAL_REGISTRY=1   # bypass mainnet-pin check (Sepolia)
export FORCE_NONCANONICAL_FTNS=1       # bypass mainnet-pin check (Sepolia)
export FORCE_NONCANONICAL_TREASURY=1   # bypass mainnet-pin check (Sepolia)

# Auto-verify post-deploy
export AUTO_VERIFY=1

# RPC (default works; override only if rate limited)
# export BASE_SEPOLIA_RPC_URL="https://your-rpc-provider..."

# "Safe-equivalent" target for transferOwnership (Step 4.4)
# Pick a different Sepolia address you control — second MetaMask
# account, second hot key, or real Safe. Single-sig is fine.
export REHEARSAL_SAFE_EQUIVALENT="0x<YOUR_SECOND_SEPOLIA_ADDRESS>"

# Sanity check
echo "Deployer key set:    $([ -n "$PRIVATE_KEY" ] && echo yes || echo NO)"
echo "FTNS:                $FTNS_TOKEN_ADDRESS"
echo "Registry:            $EXISTING_PROVENANCE_REGISTRY"
echo "Treasury:            $NETWORK_TREASURY"
echo "Safe-equivalent:     $REHEARSAL_SAFE_EQUIVALENT"
```

Verify the deployer EOA derived from PRIVATE_KEY has Sepolia ETH:

```bash
# Pull the EOA address from PRIVATE_KEY using `cast`
cast wallet address --private-key $PRIVATE_KEY
# Note this output — you'll need it later as DEPLOYER_HOT_KEY_ADDRESS

# Check Sepolia ETH balance (must be ≥ 0.005 ETH per plan §2.3)
cast balance --rpc-url https://sepolia.base.org \
  $(cast wallet address --private-key $PRIVATE_KEY) | \
  cast --from-wei
```

If balance is too low: get Sepolia ETH from a faucet, then continue.

---

## Step 2 — Deploy v2 RoyaltyDistributor on Sepolia

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM/contracts
npx hardhat run scripts/deploy-provenance.js --network base-sepolia 2>&1 | tee /tmp/a08-rehearsal-deploy.log
```

**Expected output highlights:**
- "Reusing existing ProvenanceRegistry (ROYALTY_ONLY=1):"
- "Deploying RoyaltyDistributor…"
- "  RoyaltyDistributor: 0x..." ← **note this address — you'll need it for all downstream steps**
- Manifest written: `contracts/deployments/provenance-base-sepolia-<timestamp>.json`
- (if AUTO_VERIFY=1) Basescan source-verify success

**Save the new distributor address:**

```bash
# Grep the deploy log
NEW_DIST=$(grep "RoyaltyDistributor:" /tmp/a08-rehearsal-deploy.log | tail -1 | awk '{print $NF}')
echo "New v2 distributor: $NEW_DIST"
export NEW_DIST_ADDRESS=$NEW_DIST
```

**If deploy fails:** STOP. Report the error. Most common causes:
- `PRIVATE_KEY` not set / wrong format
- Insufficient Sepolia ETH for gas
- RPC rate limit (retry with override RPC)
- Compile mismatch (re-run `npx hardhat compile` then retry)

---

## Step 3 — Run post-deploy verification

```bash
PROVENANCE_MANIFEST=$(ls -t contracts/deployments/provenance-base-sepolia-*.json | head -1) \
  npx hardhat run scripts/verify-royalty-distributor-v2-deployment.js \
  --network base-sepolia
```

**Expected:** all assertions pass. Specifically:
- Contract bytecode at the new address (non-empty)
- `ftns()` matches FTNS_TOKEN_ADDRESS
- `registry()` matches EXISTING_PROVENANCE_REGISTRY
- `networkTreasury()` matches NETWORK_TREASURY
- `owner()` == deployer hot key (BEFORE transferOwnership)
- `pendingOwner()` == 0x0
- `totalClaimable()` == 0

**If any assertion fails:** STOP. Investigate. Either compile mismatch, wrong constructor args, or contract bytecode unexpected.

---

## Step 4 — Initiate transferOwnership (deployer → Safe-equivalent)

```bash
cast send $NEW_DIST_ADDRESS \
  "transferOwnership(address)" \
  $REHEARSAL_SAFE_EQUIVALENT \
  --rpc-url https://sepolia.base.org \
  --private-key $PRIVATE_KEY
```

**Expected:** tx receipt with `OwnershipTransferStarted` event (NOT `OwnershipTransferred` yet — that fires on accept).

**Verify:**

```bash
# pendingOwner should now be Safe-equivalent
cast call $NEW_DIST_ADDRESS "pendingOwner()(address)" \
  --rpc-url https://sepolia.base.org

# owner is still deployer until accept
cast call $NEW_DIST_ADDRESS "owner()(address)" \
  --rpc-url https://sepolia.base.org
```

If `pendingOwner()` ≠ Safe-equivalent or `owner()` ≠ deployer: STOP. Investigate.

---

## Step 5 — acceptOwnership (Safe-equivalent → confirms transfer)

If REHEARSAL_SAFE_EQUIVALENT is a single-sig hot key:

```bash
# Switch to the Safe-equivalent's private key for this one tx
# (or use --private-key directly if you have it in env)
SAFE_EQ_PRIVATE_KEY="0x<PRIVATE_KEY_FOR_REHEARSAL_SAFE_EQUIVALENT>"

cast send $NEW_DIST_ADDRESS \
  "acceptOwnership()" \
  --rpc-url https://sepolia.base.org \
  --private-key $SAFE_EQ_PRIVATE_KEY
```

If REHEARSAL_SAFE_EQUIVALENT is a real Safe on Sepolia: use Safe UI at https://app.safe.global to compose `acceptOwnership()` call from the Safe; signers cosign per Safe threshold.

**Verify final ownership:**

```bash
PROVENANCE_MANIFEST=$(ls -t contracts/deployments/provenance-base-sepolia-*.json | head -1) \
EXPECT_FINAL_OWNER=$REHEARSAL_SAFE_EQUIVALENT \
  npx hardhat run scripts/verify-royalty-distributor-v2-deployment.js \
  --network base-sepolia
```

**Expected:** all checks pass. `owner()` == Safe-equivalent; `pendingOwner()` == 0x0.

---

## Step 6 — Test recoverStranded end-to-end

This is the unique-to-v2 path. Send some FTNS test tokens directly to the v2 distributor (a "stranded donation"), confirm `totalClaimable()` is unchanged, then call `recoverStranded` from the new owner and verify the donation is swept.

```bash
# Transfer 1 FTNS test token directly to the distributor (stranded donation)
cast send $FTNS_TOKEN_ADDRESS \
  "transfer(address,uint256)" \
  $NEW_DIST_ADDRESS \
  1000000000000000000 \
  --rpc-url https://sepolia.base.org \
  --private-key $PRIVATE_KEY

# Confirm distributor balance went up
cast call $FTNS_TOKEN_ADDRESS "balanceOf(address)(uint256)" \
  $NEW_DIST_ADDRESS \
  --rpc-url https://sepolia.base.org
# Expect: 1000000000000000000 (1 FTNS in wei)

# Confirm totalClaimable() is still 0 (donation didn't credit anyone)
cast call $NEW_DIST_ADDRESS "totalClaimable()(uint256)" \
  --rpc-url https://sepolia.base.org
# Expect: 0

# Recover the stranded donation to the deployer hot key (test recovery)
DEPLOYER_ADDR=$(cast wallet address --private-key $PRIVATE_KEY)
cast send $NEW_DIST_ADDRESS \
  "recoverStranded(address)" \
  $DEPLOYER_ADDR \
  --rpc-url https://sepolia.base.org \
  --private-key $SAFE_EQ_PRIVATE_KEY

# Confirm distributor balance is now 0
cast call $FTNS_TOKEN_ADDRESS "balanceOf(address)(uint256)" \
  $NEW_DIST_ADDRESS \
  --rpc-url https://sepolia.base.org
# Expect: 0

# Confirm deployer received the recovered token
cast call $FTNS_TOKEN_ADDRESS "balanceOf(address)(uint256)" \
  $DEPLOYER_ADDR \
  --rpc-url https://sepolia.base.org
# Expect: increased by 1 FTNS
```

If `recoverStranded` reverts or the balances don't move as expected: STOP and investigate. This indicates a contract-level issue that would also affect mainnet.

---

## Step 7 — Archive rehearsal manifest

```bash
# The deploy already wrote a manifest; just rename it to flag the rehearsal
TS=$(date +%s)
DEPLOY_MANIFEST=$(ls -t /Users/ryneschultz/Documents/GitHub/PRSM/contracts/deployments/provenance-base-sepolia-*.json | head -1)
cp "$DEPLOY_MANIFEST" "/Users/ryneschultz/Documents/GitHub/PRSM/contracts/deployments/sepolia-a08-rehearsal-${TS}.json"
ls /Users/ryneschultz/Documents/GitHub/PRSM/contracts/deployments/ | grep "sepolia-a08-rehearsal"
```

Commit the rehearsal manifest to the repo as evidence:

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM
git add contracts/deployments/sepolia-a08-rehearsal-*.json
git commit -m "Sepolia A-08 rehearsal manifest 2026-05-09

Rehearsal of v2 RoyaltyDistributor deploy + transferOwnership +
acceptOwnership + recoverStranded against Base Sepolia per ceremony
plan §2.4 + §8 commitment.

Distributor: <NEW_DIST_ADDRESS>
Deployer:    <DEPLOYER_ADDR>
Safe-equiv:  <REHEARSAL_SAFE_EQUIVALENT>

All ceremony plan steps 4.1-4.7 plus recoverStranded round-trip
executed successfully. Mainnet ceremony unblocked per plan §8."

git push origin main
```

---

## Step 8 — Report rehearsal outcome

Report back to me with:

1. New v2 distributor address (Sepolia)
2. Deployer EOA address
3. Safe-equivalent EOA address
4. Final `owner()` reading after Step 5
5. recoverStranded round-trip result (Step 6 outputs)
6. Whether any step required deviation from this runbook

If everything passed: I draft the PRSM-CR for mainnet ceremony ratification.

If anything failed: we triage. Per ceremony plan §8: "If the Sepolia rehearsal surfaces any issue, the mainnet ceremony slips by ≥ 7 days for re-rehearsal."

---

## Abort criteria

STOP and report immediately if any of:

- Any `cast` call returns revert
- Any verification script assertion fails
- Sepolia transactions hang past 5 minutes (RPC issue, retry with different RPC)
- `recoverStranded` doesn't move the donated tokens
- Deployer key has insufficient Sepolia ETH after the rehearsal (estimate ~0.01 ETH total spend)

---

**End of runbook.**
