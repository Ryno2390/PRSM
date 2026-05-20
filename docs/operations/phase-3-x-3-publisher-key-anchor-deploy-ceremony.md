# Phase 3.x.3 — PublisherKeyAnchor Base mainnet deploy ceremony

> Status: preflight-cleared 2026-05-20 sprint 619.
> Pending: operator execution of the mainnet deploy TX.

## What this deploys

`contracts/contracts/PublisherKeyAnchor.sol` — the on-chain
`node_id → Ed25519 pubkey` registry that the §7 trust-stack
production path needs. After this is live on Base mainnet:

- `PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS` env can be set fleet-wide
- `prsm node section7-readiness` flips anchor=ok
- Phase 2 RPC chain executor's full
  `PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc` path constructs a real
  `RpcChainExecutor` instead of falling back to stub
- Phase 2F real-model HF inference becomes operationally testable
  end-to-end

## Preflight (sprint 619, autonomous) — DONE

```
$ cd contracts && npx hardhat compile
Compiled 1 Solidity file successfully (evm target: paris).

$ ANCHOR_ADMIN_ADDRESS=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
  npx hardhat run scripts/deploy-publisher-key-anchor.js \
  --network hardhat
...
✅ DEPLOY COMPLETE (hardhat-local)
PublisherKeyAnchor: 0x5FbDB2315678afecb367f032d93F642f64180aa3
Admin:              0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791
```

Post-deploy invariants observed:
- `anchor.admin() == Foundation Safe` ✓
- `anchor.lookup(0x00...) == 0x` (empty as expected) ✓

## Deployer wallet — two paths

The deploy TX itself does NOT need to come from the Foundation
Safe. The contract's constructor takes the admin address as an
argument; whatever EOA pays gas becomes irrelevant once the
contract is live (it has no privilege over `adminOverride`).

**Path A — funded hot EOA (recommended for speed):**
- Any operator-controlled wallet with ~$3 of Base ETH for gas
- Standard `PRIVATE_KEY=...` env var
- ~30 seconds to deploy
- Trade-off: deployer EOA's address shows in Basescan history but
  has no continuing authority over the contract

**Path B — Foundation Safe via Safe SDK:**
- Multi-sig hardware ceremony executes the deploy TX
- More auditable (the Foundation Safe IS the deployer in history)
- Slower (Safe transaction proposal + 2-of-3 signing + execute)
- Requires Safe wallet SDK / app

Recommendation: **Path A**. The contract's admin (Foundation Safe)
is the only authority that matters post-deploy; the deployer's
identity is cosmetic.

## Ceremony command (Path A — funded hot EOA)

```bash
# Prereqs:
# - PRIVATE_KEY: 0x-prefixed hex of a funded EOA on Base mainnet
#   (~$3 worth of ETH covers gas + buffer; check with
#   `cast balance <addr> --rpc-url $BASE_RPC_URL`)
# - BASE_RPC_URL: https://mainnet.base.org  (or your provider)
# - BASESCAN_API_KEY: optional, for auto-verify (recommended)

cd contracts/

PRIVATE_KEY=0x<your_funded_eoa_key> \
BASE_RPC_URL=https://mainnet.base.org \
ANCHOR_ADMIN_ADDRESS=0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 \
AUTO_VERIFY=1 \
BASESCAN_API_KEY=<your_basescan_key> \
  npx hardhat run scripts/deploy-publisher-key-anchor.js \
    --network base
```

Expected output's "DEPLOY COMPLETE" block prints:
```
PublisherKeyAnchor: 0x<NEW_ADDRESS>
Admin:              0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791
```

## Post-deploy steps (autonomous-able once we have the address)

1. Update `prsm/config/networks.py`:
   ```python
   # In base_mainnet block:
   publisher_key_anchor="0x<NEW_ADDRESS>",  # Phase 3.x.3 deployed YYYY-MM-DD
   ```
2. Commit + push the address update.
3. Re-run `prsm node anchor-probe` (sprint 581) — should report `ok`.
4. Re-run `prsm node section7-readiness` (sprint 585) — anchor → `ok`.
5. Restart any operator daemons with
   `PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS` set; sprint-598
   `_build_chain_executor` rpc branch now constructs the real
   `RpcChainExecutor`.
6. Verify on Basescan that `admin()` returns the Foundation Safe.

## Rollback / re-deploy

The contract has no upgrade path — that's by design (no
proxy = no admin can swap implementation). If the deploy goes
sideways:
- Don't update `networks.py`. The bad address stays unreferenced.
- Re-deploy fresh. The new contract gets a new address; update
  `networks.py` to point at it. The old broken contract sits idle.
- Foundation Safe `adminOverride(node_id, pubkey)` is the in-band
  remediation path for compromised publisher keys after deploy.

## Operator-side safety

- **NEVER use the Foundation Safe's signing key as the deployer's
  `PRIVATE_KEY`**. The Foundation Safe is a multi-sig; it doesn't
  have a single EOA key. Use a separate funded hot wallet.
- **DO confirm the deployed `admin()` matches Foundation Safe
  exactly** before announcing the contract live. A typo in
  `ANCHOR_ADMIN_ADDRESS` would brick the emergency-override path.
- **DO verify on Basescan** (auto-verify in the deploy command
  handles this).
