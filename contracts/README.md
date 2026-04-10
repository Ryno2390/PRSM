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

# Contract verification
BASESCAN_API_KEY=your_basescan_api_key
```

### Compile + Test

```bash
npm run compile
npm test

# Gas reporting
REPORT_GAS=true npm test
```

## Deployment

The live FTNS contract is already deployed to Base mainnet. For new test deployments or bridge contract upgrades:

```bash
# Base Sepolia (testnet)
npx hardhat run scripts/deploy.js --network base-sepolia

# Base mainnet (mainnet)
npx hardhat run scripts/deploy.js --network base-mainnet
```

Verify on Basescan:

```bash
npx hardhat verify --network base-mainnet <CONTRACT_ADDRESS>
```

See [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) for the full deployment walkthrough.

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
