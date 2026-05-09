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

1. Sepolia hot key `0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0` with 0.0248 Sepolia ETH (verified 2026-05-09; private key already in `contracts/.env` as `PRIVATE_KEY`)
2. Sepolia Safe at `0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889` — verified on-chain 2026-05-09 as Safe v1.4.1, threshold 1, sole owner `0xA3683EDDBed6622f132698D7DC36a7C2DAFe4Ed3` (Ledger). 1-of-1 is acceptable per ceremony plan §8 ("single-sig test wallet acting as Safe-equivalent"); rehearsal exercises Safe-UI compose + Ledger signing but not multi-sig threshold.
3. `ETHERSCAN_API_KEY` already in `contracts/.env` (Hardhat reads it auto for the verify step; same key serves Basescan mainnet + Sepolia per the Etherscan multi-chain v2 API)
4. Repo root checked out at the head commit
5. Foundry installed (`cast --version` shows ≥ 1.7) — installer at https://foundry.paradigm.xyz if missing

**Foundry / shell setup once at session start:**

```bash
# Foundry binaries live under ~/.foundry/bin; the installer added a
# line to ~/.zshenv. Make sure the current shell has it:
source ~/.zshenv

# Source the .env once so PRIVATE_KEY + ETHERSCAN_API_KEY are present:
cd /Users/ryneschultz/Documents/GitHub/PRSM/contracts
set -a; . ./.env; set +a
cd ..

# Sanity:
cast --version
echo "PRIVATE_KEY len: ${#PRIVATE_KEY}"      # expect 66 (0x + 64 hex)
echo "ETHERSCAN_API_KEY len: ${#ETHERSCAN_API_KEY}"
```

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

In the shell that already sourced `contracts/.env` (Step 0 setup), add:

```bash
# Sepolia constructor args (existing test instances from May 5 deploy)
export FTNS_TOKEN_ADDRESS="0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a"
# IMPORTANT: NETWORK_TREASURY must be the Safe (so distributor's
# 2% network-fee wei flows to the Safe — same pattern as mainnet
# where treasury == owner). The Safe-equivalent doubles as the
# treasury for rehearsal-fidelity purposes.
export NETWORK_TREASURY="0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889"

# Reuse existing Sepolia ProvenanceRegistry. v1 vs v2 doesn't matter
# for rehearsal — the RoyaltyDistributor only stores registry as an
# address; canonical-pin check is mainnet-only.
export ROYALTY_ONLY=1
export EXISTING_PROVENANCE_REGISTRY="0x2911f9a0a02896486CdF59d6d369764841DC0eA4"
export FORCE_NONCANONICAL_REGISTRY=1   # bypass mainnet-pin check (Sepolia)
export FORCE_NONCANONICAL_FTNS=1       # bypass mainnet-pin check (Sepolia)
export FORCE_NONCANONICAL_TREASURY=1   # bypass mainnet-pin check (Sepolia)

# Auto-verify post-deploy via Hardhat
export AUTO_VERIFY=1

# RPC (default works; override only if rate limited)
# export BASE_SEPOLIA_RPC_URL="https://your-rpc-provider..."

# Safe-equivalent target for transferOwnership (Step 4)
export REHEARSAL_SAFE_EQUIVALENT="0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889"

# Sanity check
echo "Deployer key set:    $([ -n "$PRIVATE_KEY" ] && echo yes || echo NO)"
echo "Etherscan key set:   $([ -n "$ETHERSCAN_API_KEY" ] && echo yes || echo NO)"
echo "FTNS:                $FTNS_TOKEN_ADDRESS"
echo "Registry:            $EXISTING_PROVENANCE_REGISTRY"
echo "Treasury:            $NETWORK_TREASURY"
echo "Safe-equivalent:     $REHEARSAL_SAFE_EQUIVALENT"
```

Verify the deployer EOA matches the expected `0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0`:

```bash
cast wallet address --private-key $PRIVATE_KEY
# Expect: 0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0

# Balance was 0.0248 Sepolia ETH at 2026-05-09 verification.
# Plan §2.3 requires ≥ 0.005 — comfortable margin.
cast balance --rpc-url https://sepolia.base.org \
  0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0 | cast --from-wei
```

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

## Step 5 — acceptOwnership via Safe UI (Safe → confirms transfer)

This step is executed via Safe UI, not `cast send`. The Safe is the entity that needs to call `acceptOwnership()` on the v2 distributor.

**5.1.** Open Safe UI at:
```
https://app.safe.global/transactions/queue?safe=basesep:0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889
```
(Note `basesep:` prefix for Base Sepolia network.)

**5.2.** Click "New transaction" → "Transaction Builder" (or "Contract interaction" depending on UI version).

**5.3.** Compose:
- **Address (To):** `<NEW_DIST_ADDRESS>` (the v2 RoyaltyDistributor, from Step 2)
- **ABI:** paste the function signature OR upload — easiest path:
  ```
  [{"inputs":[],"name":"acceptOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"}]
  ```
- **Method:** `acceptOwnership` (no args)
- **Value:** `0`

**5.4.** Click "Add transaction" → "Create Batch" → "Send Batch".

**5.5.** Sign with Ledger:
- Connect Ledger via WebUSB / Bluetooth
- Open Ethereum app on Ledger
- Safe UI prompts "Confirm Transaction" → click; Ledger displays tx details
- Verify on Ledger screen: target = v2 distributor address, function selector = `0x79ba5097` (`acceptOwnership()`)
- Approve on Ledger
- Safe UI submits the on-chain tx (single signer, threshold 1 — no co-signing needed)

**5.6.** Wait for tx confirmation. Note the tx hash from Safe UI or Basescan.

**Verify final ownership:**

```bash
PROVENANCE_MANIFEST=$(ls -t contracts/deployments/provenance-base-sepolia-*.json | head -1) \
EXPECT_FINAL_OWNER=$REHEARSAL_SAFE_EQUIVALENT \
  npx hardhat run scripts/verify-royalty-distributor-v2-deployment.js \
  --network base-sepolia
```

**Expected:** all checks pass. `owner()` == Safe; `pendingOwner()` == 0x0.

**If acceptOwnership tx reverts:** most likely cause is `transferOwnership` (Step 4) didn't actually fire OR fired with wrong target. Re-run cast call from Step 4 to confirm `pendingOwner() == Safe`.

---

## Step 6 — Test recoverStranded end-to-end (Safe UI)

This is the unique-to-v2 path. The deployer hot key first sends test FTNS directly to the v2 distributor (a "stranded donation"); the Safe then calls `recoverStranded(<destination>)` to sweep it. Verifies the recovery surface end-to-end.

**6.1.** Verified 2026-05-09: deployer holds 99,999,980 test FTNS — no minting needed, proceed directly to donation.

**6.2.** Send 1 FTNS test token directly to the v2 distributor (stranded donation):

```bash
cast send $FTNS_TOKEN_ADDRESS \
  "transfer(address,uint256)" \
  $NEW_DIST_ADDRESS \
  1000000000000000000 \
  --rpc-url https://sepolia.base.org \
  --private-key $PRIVATE_KEY

# Confirm donation arrived
cast call $FTNS_TOKEN_ADDRESS "balanceOf(address)(uint256)" \
  $NEW_DIST_ADDRESS \
  --rpc-url https://sepolia.base.org
# Expect: 1000000000000000000 (1 FTNS in wei)

# Confirm totalClaimable() is still 0 (donation didn't credit anyone)
cast call $NEW_DIST_ADDRESS "totalClaimable()(uint256)" \
  --rpc-url https://sepolia.base.org
# Expect: 0  ← this is the key invariant — recoverStranded
#              math depends on (balanceOf(this) - totalClaimable)
```

**6.3.** Call `recoverStranded` from the Safe via Safe UI:

- Safe UI: `https://app.safe.global/transactions/queue?safe=basesep:0xCb4Bfa18E5B166C2E13c18007b4F4E1b2CE8A889`
- New transaction → Transaction Builder
- **Address (To):** `<NEW_DIST_ADDRESS>`
- **ABI:**
  ```
  [{"inputs":[{"internalType":"address","name":"to","type":"address"}],"name":"recoverStranded","outputs":[],"stateMutability":"nonpayable","type":"function"}]
  ```
- **Method:** `recoverStranded`
- **Argument `to`:** `0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0` (deployer EOA — sweep destination)
- Submit → Ledger sign → broadcast.

**6.4.** Verify the donation moved:

```bash
# Distributor should be 0 now
cast call $FTNS_TOKEN_ADDRESS "balanceOf(address)(uint256)" \
  $NEW_DIST_ADDRESS \
  --rpc-url https://sepolia.base.org
# Expect: 0

# Deployer balance should have increased by exactly 1 FTNS
cast call $FTNS_TOKEN_ADDRESS "balanceOf(address)(uint256)" \
  0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0 \
  --rpc-url https://sepolia.base.org
# Expect: previous balance + 1000000000000000000
```

**Failure modes to watch for:**
- `recoverStranded` reverts with `OnlyOwner` → Safe didn't actually accept ownership in Step 5 (re-check `owner()`)
- `recoverStranded` reverts with `NoStranded` → totalClaimable accounting drift; donation must be > totalClaimable for any to be considered "stranded" (verify §7 of `A-08-recoverStranded-design.md`)
- Distributor balance stays 1 FTNS → Safe tx failed to broadcast; check Safe UI tx history for revert reason

If any failure mode triggers: STOP. The same failure would occur on mainnet.

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
