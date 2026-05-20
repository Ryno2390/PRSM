# Phase 3.x.3 Path B — Foundation Safe deploy via Singleton Factory

> Status: payload generated 2026-05-20 sprint 620.
> Pending: operator multi-sig signing + execute.

## What this is

Path B deploys `PublisherKeyAnchor` to Base mainnet **from the
Foundation Safe** (`0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`).
The Safe is the deployer-of-record on Basescan — more auditable
than Path A (a funded hot EOA).

The Safe can't natively do `to=null` contract creation (Safe's
`execTransaction` only does Call/DelegateCall on existing
addresses). Path B uses Arachnid's Deterministic Deployment
Proxy (`0xce0042B868300000d44A59004Da54A005ffdcf9f`) — already
deployed on Base mainnet (308 bytes verified). The Safe calls
the factory with `(salt, initcode)`; factory deploys via CREATE2;
contract address is deterministic.

## Pre-execution evidence (autonomous, sprint 620)

| Check | Result |
|---|---|
| Singleton Factory deployed on Base mainnet (`0xce0042...cf9f`) | ✅ 308 bytes |
| Predicted contract address empty (no collision) | ✅ 0 bytes |
| Foundation Safe nonce on Base mainnet | 1 |
| Contract compiles clean (hardhat) | ✅ sprint 619 |
| Hardhat-local dry-run post-deploy invariants pass | ✅ sprint 619 |

**Predicted contract address (CREATE2 deterministic):**

```
0xd811ad9986f44f404b0fd992168a7cc76206df03
```

This address depends ONLY on:
- factory: `0xce0042B868300000d44A59004Da54A005ffdcf9f`
- salt: `bytes32(0)`
- `keccak256(bytecode || abi.encode(admin=0x91b0...5791))`

So the address is fixed before signing — no surprises post-execute.

## Safe transaction parameters

Paste these into `https://app.safe.global` → New Transaction →
Contract Interaction (or "Custom Transaction" depending on UI
version):

| Field | Value |
|---|---|
| To | `0xce0042B868300000d44A59004Da54A005ffdcf9f` |
| Value | `0` |
| Operation | `Call` (NOT DelegateCall) |
| Data | (see calldata file below — single 2344-byte hex string) |

**Calldata file**: `docs/operations/phase-3-x-3-safe-deploy-calldata.txt`

Copy the entire single-line `0x...` from that file into the Safe
UI's Data field. Length: 2344 bytes / 4688 hex chars (+ `0x`).

## Generating the payload yourself (reproducibility)

If you want to independently verify the calldata and predicted
address:

```bash
cd contracts/
npx hardhat compile  # rebuild artifact
cd ..
python3 scripts/generate_safe_deploy_payload.py
```

The output prints the exact data + predicted address. Re-running
with no source changes produces byte-identical calldata.

## Signing flow

1. **Propose**: Open Safe UI → New Transaction → Custom. Fill
   the To / Value / Data fields per the table above. Submit.
2. **Sign 2-of-3**: With the Foundation Safe being a 2-of-3
   multi-sig (Ledger + Trezor + OneKey per `prsm/config/networks.py:110`),
   two of the three hardware signers approve in the Safe UI.
3. **Execute**: After the second signature, anyone (including the
   second signer) clicks "Execute" — the Safe submits the TX,
   the Singleton Factory deploys the contract.

Expected gas: ~250k–400k (proxy delegation overhead + ~100k for
the actual deploy). At Base mainnet base fees (~0.01 gwei) this
is roughly $0.10–$0.30 worth of ETH.

## Post-execution verification (sprint 621 prep — autonomous after deploy)

1. Confirm contract code at predicted address:
   ```bash
   curl -s -X POST -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_getCode","params":["0xd811ad9986f44f404b0fd992168a7cc76206df03","latest"],"id":1}' \
     https://mainnet.base.org | python3 -c "import json,sys; d=json.load(sys.stdin).get('result',''); print(f'code length: {len(d)//2 - 1} bytes — {\"OK\" if len(d) > 100 else \"NOT DEPLOYED\"}')"
   ```
2. Read `admin()` (Selector: `0xf851a440`, returns address):
   ```bash
   curl -s -X POST -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_call","params":[{"to":"0xd811ad9986f44f404b0fd992168a7cc76206df03","data":"0xf851a440"},"latest"],"id":1}' \
     https://mainnet.base.org
   # Expected result: 0x...91b0e6F85A371D82De94eD13A3812d9f5A4E5791 (Foundation Safe, left-padded to 32 bytes)
   ```
3. Update `prsm/config/networks.py`:
   ```python
   publisher_key_anchor="0xd811ad9986f44f404b0fd992168a7cc76206df03",  # Phase 3.x.3 deployed YYYY-MM-DD
   ```
4. Verify on Basescan (manual: paste source code) or via Hardhat:
   ```bash
   cd contracts/
   BASESCAN_API_KEY=... npx hardhat verify --network base \
     0xd811ad9986f44f404b0fd992168a7cc76206df03 \
     0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791
   ```
5. Re-run `prsm node anchor-probe` — should return `ok`.
6. Re-run `prsm node section7-readiness` — anchor → `ok`.

## If something goes wrong

- **Safe TX reverts**: most likely the calldata pasted got truncated
  or had a non-hex character. Regenerate via the Python script;
  the file in this repo is the source of truth.
- **Contract appears at predicted address but `admin()` returns
  wrong address**: somehow the calldata's constructor args got
  mangled. Re-deploy with a fresh salt (change `SALT_HEX` in
  `scripts/generate_safe_deploy_payload.py` to e.g.
  `0x...01`) — the new salt gives a fresh predicted address and
  doesn't collide with the broken one.
- **Predicted address occupied by another contract** (extremely
  unlikely given the unique initcode): pick a different salt as
  above.

## Why CREATE2 and not Safe's native createProxy

Safe v1.4's `createProxyWithNonce` is for spawning ANOTHER Safe
instance. We're deploying an arbitrary contract. CREATE2 via
Arachnid's Singleton Factory is the canonical pattern.
