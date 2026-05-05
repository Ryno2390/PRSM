# PRSM Testnet Quickstart

**Network:** Base Sepolia (chainId 84532)
**Status:** Live as of 2026-05-05

This guide gets you from zero to a running PRSM node earning real on-chain
testnet-FTNS in under 30 minutes.

> ⚠️ **Testnet only.** All FTNS on this network has zero monetary value.
> Do not use a real wallet's private key. Do not connect mainnet wallets.
> The "Foundation" on testnet is a single deployer EOA, not the real
> 2-of-3 hardware multisig used on mainnet.

---

## TL;DR

```bash
pip install prsm                     # 1. install
prsm join-testnet                    # 2. generate burner wallet
# (fund the printed address with Base Sepolia ETH)
source ~/.prsm/testnet-deployer.env  # 3. activate
prsm wallet info --network testnet   # 4. confirm balance
prsm storage upload <file>           # 5. upload + earn royalties
prsm wallet claim --network testnet  # 6. withdraw earnings
```

---

## What's deployed on Base Sepolia

| Contract | Address | Role |
|---|---|---|
| FTNSTokenSimple | `0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a` | ERC-20 token (1B cap, 100M genesis) |
| ProvenanceRegistry | `0x2911f9a0a02896486CdF59d6d369764841DC0eA4` | Content registration + creator binding |
| RoyaltyDistributor | `0xB790045ff826C76fe02DBc54a6ef0021951Fd892` | Pull-payment royalty splits |
| BatchSettlementRegistry | `0x200B35fCB68678717a355176e22321Dc3e703315` | Batch settlement + challenge windows |
| EscrowPool | `0x4BDf07b2BB23176469bdEFca2B103AdB3DCb3dd2` | Per-requester escrow |
| Ed25519Verifier | `0x1d7fCbC08792D649016703C4Be59635e619097EE` | On-chain signature verification |
| StakeBond | `0xDea103f33503BC7e73Ea447d43b2Cd7E2710D20A` | Provider stake bonds |
| EmissionController | `0x134552dbe2d235DB60be5A881A2c06d9E42d2613` | Halving emission curve |
| CompensationDistributor | `0xFa3610e87027b548B86859B105B1b39B30d9955B` | Pull-based provider rewards |
| StorageSlashing | `0x4FDd792fDcDcEe31861D23A1B0342058Ed32c766` | Storage-proof challenges |
| KeyDistribution | `0xC33ceA03455DB9246716ccF04cE1446EB56B439b` | Threshold key distribution |

All addresses are also pinned in `prsm/config/networks.py`.

---

## Step 1 — Install PRSM

```bash
pip install prsm
```

Or from source:

```bash
git clone https://github.com/Ryno2390/PRSM
cd PRSM
pip install -e .
```

Verify:

```bash
prsm --version
```

---

## Step 2 — Generate a testnet burner wallet

```bash
prsm join-testnet
```

This creates a fresh keypair and writes `~/.prsm/testnet-deployer.env`
(mode 600) containing:

- `PRIVATE_KEY` (the burner's signing key)
- `BASE_SEPOLIA_RPC_URL` (default `https://sepolia.base.org`)
- All testnet contract addresses
- `PRSM_ONCHAIN_PROVENANCE=1` (so uploads register on-chain)

The command prints your wallet address and a Basescan link. **Copy
the address — you'll need it to fund the wallet.**

> The burner key is in the env file in plaintext. That's fine — testnet
> tokens have zero value. **Never reuse this key for mainnet, and never
> commit the env file.**

---

## Step 3 — Fund the wallet with Base Sepolia ETH

The burner needs some Base Sepolia ETH to pay gas (gas is cheap on
Base Sepolia, but not free). Recommended faucets:

| Faucet | Notes | URL |
|---|---|---|
| **Coinbase Developer Platform** | No mainnet gate; ~0.0001 ETH/click, 0.1 ETH/day | https://portal.cdp.coinbase.com/faucet |
| **Alchemy** | Requires ≥ 0.001 ETH on Ethereum mainnet at the wallet | https://www.alchemy.com/faucets/base-sepolia |
| **pk910 PoW** (Sepolia → bridge to Base Sepolia) | Browser-based mining; bypasses mainnet gates but blocks hosting/VPN IPs | https://sepolia-faucet.pk910.de |

A few hundred clicks of the Coinbase faucet (~0.02 ETH) is enough for
many node operations. If you want a single big drop, the Alchemy faucet
requires a small mainnet-ETH stake at your wallet (~$5 once, satisfies
their Sybil gate).

**Verify your wallet is funded** at the Basescan link from Step 2 (or
plug the address into `https://sepolia.basescan.org`).

---

## Step 4 — Request testnet-FTNS

Faucet for testnet-FTNS itself is currently manual (per ratified
testnet-decision-policy):

- Drop your wallet address in the project's `#testnet-faucet` Discord
  channel (or email `security@prsm.network` if you don't have Discord
  access).
- The Foundation operator runs `airdrop-testnet-ftns.js` to send
  ~1000 testnet-FTNS within 24 hours.

You can use PRSM with 0 FTNS — uploading + earning still works. You
only need FTNS to *spend* (pay other users for content access).

---

## Step 5 — Activate + verify state

```bash
source ~/.prsm/testnet-deployer.env
prsm wallet info --network testnet
```

You should see something like:

```
Wallet on testnet (chainId 84532)
Address:        0x...
FTNS balance:   0.000000 FTNS
Claimable:      0.000000 FTNS
```

---

## Step 6 — Upload content + earn royalties

Pre-requisite: you need a local IPFS node running (or another IPFS
endpoint accessible at `http://127.0.0.1:5001`). Quickstart:

```bash
# install + run IPFS (one-time)
brew install ipfs    # or: apt install ipfs
ipfs init
ipfs daemon &
```

Then upload:

```bash
prsm storage upload my_dataset.csv \
  --description "My dataset" \
  --royalty-rate 0.05
```

This:

1. Uploads the file to local IPFS
2. Computes the canonical creator-bound provenance hash
3. **Registers the content on-chain** at ProvenanceRegistry (visible at
   https://sepolia.basescan.org/address/0x2911f9a0a02896486CdF59d6d369764841DC0eA4)
4. Stores the on-chain tx hash in the local upload record
5. Gossips the registration to other PRSM nodes on the network

The on-chain registration is what makes royalty distribution possible:
when someone else later pays to access your content, the
RoyaltyDistributor uses the on-chain creator binding to credit the
royalty share to your address.

---

## Step 7 — Withdraw earnings

When other users access your content, the RoyaltyDistributor
accumulates your share in `claimable[your-address]` (visible via
`prsm wallet info`). To withdraw:

```bash
prsm wallet claim --network testnet
```

The command:
- Pre-flight checks `claimable > 0` (clean error if nothing to claim)
- Prompts for confirmation
- Submits `RoyaltyDistributor.claim()`
- Prints tx hash + Basescan link

After confirmation, the FTNS lands in your wallet's regular balance —
verify with `prsm wallet info`.

---

## What testnet does NOT do (yet)

These are documented limitations vs. mainnet — see
`docs/2026-05-05-public-testnet-deploy-plan.md` for the full follow-up
list:

- **No accelerated halving.** EmissionController uses mainnet's 4-year
  halving constant, so you won't observe a halving event in a typical
  test window. (Tracked as task T10.)
- **No 2-of-3 multisig.** Testnet "Foundation" is a single deployer
  EOA. Operations like pause / parameter updates / emergency
  disbursement do NOT exercise the real 2-of-3 governance flow.
- **No external audit.** The contracts are L2-team-cleared internally
  (1 CRIT + 7 HIGH + 7 MEDIUM remediated) but external audit (L4 firm
  pair-review) hasn't started. Mainnet-equivalent security guarantees
  are not yet established.
- **Bootstrap discovery hardcoded for mainnet.** Currently the
  bootstrap node returns mainnet contract addresses by default; testnet
  users have to set env vars manually (handled by `prsm join-testnet`).
  Network-tagged bootstrap responses is task T4.

---

## Troubleshooting

### `prsm wallet info` shows balance: 0

Either the burner has no FTNS yet (request airdrop in Step 4), or the
RPC isn't reaching Base Sepolia. Verify:

```bash
echo $BASE_SEPOLIA_RPC_URL
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  $BASE_SEPOLIA_RPC_URL
# Expected: {"jsonrpc":"2.0","id":1,"result":"0x14a34"}  (chainId 84532)
```

### `prsm storage upload` fails with "creator_address: None"

Make sure you sourced the env file:

```bash
source ~/.prsm/testnet-deployer.env
echo $PRIVATE_KEY  # should be 66 chars (0x + 64 hex)
```

If the env vars are set but the upload still fails, the on-chain
provenance call may have errored — check logs. The upload should
still succeed locally even if the on-chain registration fails;
`provenance_tx_hash` will be None in the local record.

### Need to start over

```bash
rm ~/.prsm/testnet-deployer.env
prsm join-testnet
```

The new wallet has a different address, so any FTNS / earnings
attached to the old burner stay with that address.

---

## Architecture reference

For a deeper protocol walkthrough:

- **Two-entity structure** (Foundation + Prismatica): `docs/2026-04-21-prsm-gov-1-foundation-governance-charter.md`
- **Tokenomics** (FTNS, halving, emission cap): `docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md`
- **Audit pipeline** (L0-L11): `audits/AUDIT_PLAN.md`
- **Risk register**: `docs/2026-04-22-risk-register-track-2.md`

---

## Get help

- Discord: (project-specific channel — TBD)
- GitHub issues: https://github.com/Ryno2390/PRSM/issues
- Security: `security@prsm.network` (PGP key in `SECURITY.md`)

---

*This quickstart is part of the public testnet rollout
(`docs/2026-05-05-public-testnet-deploy-plan.md`). Last updated:
2026-05-05.*
