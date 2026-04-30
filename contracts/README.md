# PRSM FTNS Smart Contracts

Solidity contracts backing the FTNS (Fungible Tokens for Node Support) token. The live token is deployed on **Base mainnet** at `0x5276a3756C85f2E9e46f6D34386167a209aa16e5`.

## Contracts

| Contract | Purpose |
|----------|---------|
| `FTNSTokenSimple.sol` | Minimal ERC20 FTNS token — minted at contribution time, no pre-mine |
| `FTNSBridge.sol` | Cross-chain bridge for FTNS between supported networks |
| `BridgeSecurity.sol` | Multi-signature verification + rate limiting for bridge operations |

PRSM does **not** ship on-chain contracts for marketplace, governance, or staking. Those behaviors run off-chain in the Ring 4 tokenomics layer (`prsm/economy/tokenomics/`) and on the DAG ledger for atomic settlement. FTNS↔USD/USDT fiat conversion is handled by the Chronos bridge in `prsm/compute/chronos/`.

## Quick Start

### Prerequisites

```bash
node >= 16.0.0
npm >= 7.0.0
```

### Install

```bash
cd contracts
npm install
```

### Environment

Copy `.env.example` to `.env` and configure:

```env
# Base network RPC URL
BASE_RPC_URL=https://mainnet.base.org

# Deployer wallet (use a dedicated key — never your main wallet)
PRIVATE_KEY=your_deployment_private_key

# Contract verification (Etherscan V2 unified API covers Base mainnet + Sepolia)
ETHERSCAN_API_KEY=your_etherscan_api_key
```

### Compile + Test

```bash
npm run compile
npm test

# Gas reporting
REPORT_GAS=true npm test
```

## Deployment

The FTNS token is already live on Base mainnet (see "Live Addresses"
below). Subsequent contract ceremonies follow this layered structure:

| Ceremony | Scope | Script | Engineering runbook |
|---|---|---|---|
| **Phase 1.3 Task 8** | ProvenanceRegistry + RoyaltyDistributor with Foundation Safe as `NETWORK_TREASURY` | `scripts/deploy-provenance.js` | [`docs/2026-04-30-phase1.3-task8-engineering-runbook.md`](../docs/2026-04-30-phase1.3-task8-engineering-runbook.md) |
| **Post-audit** | Audit-bundle + Phase 8 emission + Phase 7-storage (9 contracts + transferOwnership) | `../scripts/rehearse-deploy.sh` orchestrator | [`docs/2026-04-30-post-audit-deploy-ceremony-runbook.md`](../docs/2026-04-30-post-audit-deploy-ceremony-runbook.md) |

For Phase 1.3 Task 8 (the immediate post-hardware-arrival ceremony):

```bash
# Pre-flight: 10 verifications BEFORE burning gas
../scripts/pre-task8-checklist.sh

# Hardhat-local rehearsal (mock FTNS + stub treasury + verifier)
../scripts/rehearse-task8.sh

# Base Sepolia rehearsal (real testnet RPC + real testnet treasury)
NETWORK=base-sepolia FTNS_TOKEN_ADDRESS=0x... NETWORK_TREASURY=0x... \
  PRIVATE_KEY=0x... BASE_SEPOLIA_RPC_URL=<rpc> \
  ../scripts/rehearse-task8.sh

# Mainnet (only after pre-flight green)
npx hardhat run scripts/deploy-provenance.js --network base 2>&1 \
  | tee /tmp/mainnet-deploy-$(date +%s).log

# Post-deploy verification
PROVENANCE_MANIFEST=deployments/provenance-base-<ts>.json \
  npx hardhat run scripts/verify-provenance-deployment.js --network base

# Post-deploy ops handoff (find OLD-address refs + generate PR + memory stub)
MAINNET_MANIFEST=deployments/provenance-base-<ts>.json \
  ../scripts/post-task8-handoff-checklist.sh
```

Verify on Basescan (Etherscan v2 unified API):

```bash
npx hardhat verify --network base <CONTRACT_ADDRESS> [constructor args...]
```

## Live Addresses

| Network | Contract | Address |
|---------|----------|---------|
| **Base mainnet** | FTNS Token | `0x5276a3756C85f2E9e46f6D34386167a209aa16e5` |
| Ethereum Sepolia | FTNS test token | `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4` |

## Security

- **No pre-mine**: FTNS is minted at contribution time by off-chain settlement signed by the DAG ledger. There is no foundation reserve to dump.
- **Multi-sig bridge**: `BridgeSecurity.sol` enforces multi-signature validation and rate limits on cross-chain transfers.
- **ReentrancyGuard** on all state-changing functions.
- **Pausable** for emergency halts under on-chain governance.

## License

MIT — see [`LICENSE`](../LICENSE).
