# PRSM Public Testnet — Quickstart

The PRSM public testnet runs on **Base Sepolia** (chain id 84532). It's
a no-stakes environment for trying out the protocol: testnet-FTNS has
**zero monetary value**, and contracts can be redeployed at any time.
Use mainnet for anything that needs to be durable.

## What's deployed

All 9 audit-bundle + Phase 8 + Phase 7-storage contracts are live and
Basescan-verified as of 2026-05-07:

| Contract | Address | Basescan |
|---|---|---|
| FTNS (UUPS proxy) | `0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a` | [↗](https://sepolia.basescan.org/address/0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a) |
| BatchSettlementRegistry | `0xF8BEEb4362222b50109b6034767322B31aA92449` | [↗](https://sepolia.basescan.org/address/0xF8BEEb4362222b50109b6034767322B31aA92449) |
| EscrowPool | `0xaa28b5818242608e04C1773c3e34bF7bFfb96248` | [↗](https://sepolia.basescan.org/address/0xaa28b5818242608e04C1773c3e34bF7bFfb96248) |
| Ed25519Verifier | `0x208dc98545Fe062d0B13Ac07b073633E6a62b5A9` | [↗](https://sepolia.basescan.org/address/0x208dc98545Fe062d0B13Ac07b073633E6a62b5A9) |
| StakeBond | `0xF93aCa6551F408fFfe24292288d5488864D5264c` | [↗](https://sepolia.basescan.org/address/0xF93aCa6551F408fFfe24292288d5488864D5264c) |
| EmissionController | `0x1478F8f5F13a5BDeBc2a0b7C185D19BEE15f312e` | [↗](https://sepolia.basescan.org/address/0x1478F8f5F13a5BDeBc2a0b7C185D19BEE15f312e) |
| CompensationDistributor | `0xFd730f8E513eD184F255cb1a62791e711B2e81b9` | [↗](https://sepolia.basescan.org/address/0xFd730f8E513eD184F255cb1a62791e711B2e81b9) |
| StorageSlashing | `0x2ba1B361d2AD49f15F1131762fA3512d7824EB06` | [↗](https://sepolia.basescan.org/address/0x2ba1B361d2AD49f15F1131762fA3512d7824EB06) |
| KeyDistribution | `0xdB41A471AAC86285cD855bEdC27D7FC810dc3318` | [↗](https://sepolia.basescan.org/address/0xdB41A471AAC86285cD855bEdC27D7FC810dc3318) |

**Out of testnet scope:** `ProvenanceRegistry` and `RoyaltyDistributor`
are mainnet-only. The content-registration / royalty path is exercised
on mainnet, not testnet.

## What testnet does NOT have

- A real Foundation Safe — the "foundation" address on testnet is a
  single deployer EOA. Multisig governance lives on mainnet.
- ~~Accelerated halving~~ — **DELIVERED 2026-05-07 (T10).** Testnet
  `EmissionController` now uses a 1-hour epoch (vs mainnet 4 years).
  `EPOCH_DURATION_SECONDS` was refactored from `constant` to
  constructor-set `immutable`; mainnet (chainId 8453) constructor
  enforces exactly 4 years. The first halving on testnet fires at
  `epochZeroStart + 1h`; second at `+2h`, etc. — emission rate halves
  from 1 FTNS/s to 0.5 FTNS/s after the first hour and so on, so the
  full halving curve is observable in a single short session.
- Drainable foundation reserve — the foundation reserve wallet on
  StakeBond is set to the FTNS token address itself (passes the
  `code.length > 0` gate; foundation-share slashes accumulate there
  passively). On testnet there's no recovery path; the slashing
  flow is exercisable but the economic recovery is mainnet-only.

## Onboarding

### 1. Generate a fresh testnet burner wallet

```bash
prsm join-testnet
```

Creates `~/.prsm/testnet-deployer.env` with a freshly-generated private
key plus all the env vars below. The wallet starts with zero balance.

> **Warning:** the burner is only for testnet. Never send mainnet funds
> to this address. The key is stored in plaintext at
> `~/.prsm/testnet-deployer.env`.

The env file sets:

```bash
PRIVATE_KEY=0x<64-hex>             # the burner key (also as FTNS_WALLET_PRIVATE_KEY)
BASE_SEPOLIA_RPC_URL=https://sepolia.base.org
PRSM_NETWORK=testnet
PRSM_ONCHAIN_PROVENANCE=1
FTNS_TOKEN_ADDRESS=0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a
```

### 2. Fund the burner with Base Sepolia ETH

You need ~0.001 ETH (a few cents at mainnet prices, but free on testnet)
for transaction fees. Use any of these faucets — pick whichever is up:

- https://www.coinbase.com/faucets/base-sepolia
- https://www.alchemy.com/faucets/base-sepolia
- https://docs.base.org/tools/network-faucets/

The address to fund is printed at the top of `~/.prsm/testnet-deployer.env`
(also visible via `cat ~/.prsm/testnet-deployer.env | head -3`).

### 3. Request testnet-FTNS

For the first ~20 testnet users, ask in the PRSM project's
`#testnet-faucet` channel. Founder will airdrop within 24h. The airdrop
script is `contracts/scripts/airdrop-testnet-ftns.js`:

```bash
PRIVATE_KEY=0x<faucet-key> \
FTNS_TOKEN_ADDRESS=0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a \
RECIPIENT=0x<your-burner-address> \
AMOUNT_FTNS=1000 \
    npx hardhat run scripts/airdrop-testnet-ftns.js --network base-sepolia
```

(For the founder running this: `PRIVATE_KEY` is the deployer's key
which holds 100M testnet-FTNS at deploy time.)

### 4. Activate the env + start a node

```bash
source ~/.prsm/testnet-deployer.env
prsm wallet info             # sanity-check: should show your burner address
prsm node start --network testnet
```

Behind the scenes, `PRSM_NETWORK=testnet` flows through to every
on-chain client (`OnChainFTNSLedger`, batch-settlement client,
emission watcher, etc.) via `prsm.config.networks.resolve_endpoints()`.
The client picks up:

- RPC endpoint: `BASE_SEPOLIA_RPC_URL` (defaults to
  `https://sepolia.base.org`); pin a paid Alchemy / QuickNode endpoint
  via `BASE_SEPOLIA_RPC_URL=https://...` for higher throughput.
- All testnet contract addresses from `prsm/config/networks.py`'s
  `TESTNET` config — no need to set them individually.

### 5. Verify the wiring

A live E2E smoke test confirms the full path is correct:

```bash
PRSM_TESTNET_E2E=1 \
PRSM_NETWORK=testnet \
FTNS_WALLET_PRIVATE_KEY=$PRIVATE_KEY \
  pytest tests/e2e/test_testnet_smoke.py -v
```

3 tests, ~10 seconds. Locks in: testnet config resolves correctly,
`OnChainFTNSLedger` connects to Sepolia (chain 84532) on the right
contract, total-supply matches the post-deploy invariant.

## What you can actually do

| Operation | Status on testnet |
|---|---|
| `wallet info` (FTNS balance, address) | reads testnet contract |
| Bond stake (`prsm provider bond`) | testnet StakeBond accepts FTNS |
| Submit inference job (escrow → settle) | full Phase 3.1 batch settlement |
| Storage proof challenge (Phase 7-storage) | StorageSlashing live |
| Emission claim (Phase 8) | curve uses an accelerated 1-hour epoch — rate halves every hour so the full curve is visible in one session |
| Content registration / royalty | not on testnet (mainnet-only) |

## Troubleshooting

### `OnChainFTNSLedger` connects to mainnet despite `PRSM_NETWORK=testnet`

Make sure you ran `source ~/.prsm/testnet-deployer.env` in the SAME
shell where you started the node. The env-var resolution happens at
Python module-load time. If you set `PRSM_NETWORK=testnet` AFTER the
process started, it has no effect.

### `transaction execution reverted` on bond / settle

Check that the burner address has both:
1. Sepolia ETH for gas (faucet, see step 2)
2. Testnet-FTNS approved to the contract (StakeBond / EscrowPool —
   the SDK does this automatically; if running against the contracts
   manually, call `FTNS.approve(contract, amount)` first)

### Faucet exhausted / 24h-rate-limited

The Coinbase + Alchemy faucets enforce per-IP rate limits. Try a
different one from the list, or wait 24h. Testnet-FTNS shortages can
also happen — ping in `#testnet-faucet` and the founder will top up
from the deployer wallet.

### `prsm join-testnet` already wrote a config; how do I rotate burners?

```bash
prsm join-testnet --force
```

Overwrites `~/.prsm/testnet-deployer.env` with a fresh burner. Old
balance is stranded at the old address (testnet — no economic loss).

## Resources

- **Bundle deploy manifests:** `contracts/deployments/`
  (`audit-bundle-base-sepolia-*.json`,
  `phase8-emission-stack-base-sepolia-*.json`,
  `phase7-storage-base-sepolia-*.json`,
  `phase1-ftns-base-sepolia-*.json`).
- **Original deploy plan:** `docs/2026-05-05-public-testnet-deploy-plan.md`.
- **Source-of-truth network config:** `prsm/config/networks.py`
  (`TESTNET` dataclass).
- **Live RPC explorer:** https://sepolia.basescan.org/

## Reporting issues

Testnet failures are interesting — that's the whole point of running
one. File issues at https://github.com/prsm-network/PRSM/issues with:

- The PRSM commit SHA (`git rev-parse HEAD`)
- The contents of `~/.prsm/testnet-deployer.env` *minus* the
  `PRIVATE_KEY` line (DO NOT paste the private key)
- The exact command + the exact error
- The Basescan link to any failed tx (if applicable)
