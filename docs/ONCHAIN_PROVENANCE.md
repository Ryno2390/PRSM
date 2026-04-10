# On-Chain Provenance & Royalty Distribution (Phase 1)

> Status: shipped, opt-in via feature flag. See
> [`2026-04-10-audit-gap-roadmap.md`](./2026-04-10-audit-gap-roadmap.md) for context.

## What this is

PRSM adds two smart contracts on Base mainnet that move royalty
distribution from the local node ledger to the chain:

- **ProvenanceRegistry** — maps a 32-byte content hash to a creator
  address and a royalty rate (basis points). Anyone can register content;
  the registrant becomes the creator and earns royalties on use. Records
  are immutable except for creator-initiated ownership transfer.
- **RoyaltyDistributor** — atomic three-way splitter. Pulls FTNS from a
  payer via `transferFrom`, looks up the creator and royalty rate from
  the registry, and splits the gross amount into creator / network
  treasury / serving node shares in a single transaction.

Source contracts live under `contracts/contracts/`.

## Why

Before Phase 1, royalty distribution was computed in a local SQLite
ledger on each node. Trust depended entirely on the node operator —
nothing prevented an operator from rewriting their own ledger to
under-pay creators or over-pay themselves. With Phase 1, anyone can
audit royalty payments by reading Base mainnet event logs. Rolling back
a payment requires a chain reorg.

## The split

For a `gross` payment on content with rate `R` basis points:

| Recipient | Amount |
|---|---|
| Creator | `gross * R / 10000` |
| Network treasury | `gross * 200 / 10000` (2% fixed) |
| Serving node | `gross - creator - network` |

The contract reverts if `R + 200 > 10000`. The maximum royalty rate is
9800 bps (98%) so the network fee always fits.

## Enabling on a node

Set these environment variables before starting your node:

```bash
export PRSM_ONCHAIN_PROVENANCE=1
export PRSM_PROVENANCE_REGISTRY_ADDRESS=0x…   # from contracts/deployments/
export PRSM_ROYALTY_DISTRIBUTOR_ADDRESS=0x…   # from contracts/deployments/
export PRSM_BASE_RPC_URL=https://mainnet.base.org
export FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5
export FTNS_WALLET_PRIVATE_KEY=0x…            # creator/payer wallet
```

When the flag is set, content access payments route through
`RoyaltyDistributor` for content that has been registered on-chain.
For unregistered content, the call falls through to the existing
local-ledger path so payments are never lost during a chain outage or
unfunded gas wallet.

## CLI commands

```bash
# Register a file. Computes sha3-256 of the file bytes as the content hash.
prsm provenance register ./mydata.csv --royalty-bps 800 --metadata-uri ipfs://Qm…

# Look up a record by hash or by file path.
prsm provenance info 0x<64 hex chars>
prsm provenance info ./mydata.csv

# Transfer ownership.
prsm provenance transfer 0x<hash> 0x<new_creator>
```

## Verifying a royalty payment on Basescan

1. Find the `RoyaltyPaid` event on the deployed `RoyaltyDistributor`
   contract.
2. The `contentHash` topic identifies which piece of content was paid.
3. The `creator` topic confirms who received the royalty.
4. The `creatorAmount`, `networkAmount`, and `servingNodeAmount` data
   fields show the exact split.
5. Cross-reference the `creator` address against
   `ProvenanceRegistry.contents(contentHash)` — they must match.

## Limitations (Phase 1)

- Only original creator + serving node + treasury — derivative chains
  are not yet supported on-chain. Multi-hop derivative provenance can
  be added later via mapping additions without contract redeploy.
- Approval flow is plain ERC-20 `approve` + `transferFrom` (no permit).
  Each new payer pays one extra approve transaction the first time.
- Content addressed by `bytes32` only — no ERC-721 wrapper.

These will be addressed in subsequent phases or as upgrades become
necessary.

## Architecture pointers

| Concern | Source |
|---|---|
| Solidity contract: registry | `contracts/contracts/ProvenanceRegistry.sol` |
| Solidity contract: distributor | `contracts/contracts/RoyaltyDistributor.sol` |
| Hardhat tests | `contracts/test/ProvenanceRegistry.test.js` + `RoyaltyDistributor.test.js` |
| Deploy script | `contracts/scripts/deploy-provenance.js` |
| Python registry client | `prsm/economy/web3/provenance_registry.py` |
| Python distributor client | `prsm/economy/web3/royalty_distributor.py` |
| Payment wiring | `prsm/node/content_economy.py` (search `ONCHAIN_PROVENANCE_ENABLED`) |
| CLI | `prsm/cli_modules/provenance.py` |
| End-to-end test | `tests/integration/test_onchain_provenance_e2e.py` |
