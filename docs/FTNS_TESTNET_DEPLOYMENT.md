# FTNS Token Testnet Deployment Guide

This guide covers deploying the FTNS (Fungible Tokens for Node Support) token to testnet networks for on-chain economics demonstration.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Network Configuration](#network-configuration)
- [Environment Setup](#environment-setup)
- [Deployment Methods](#deployment-methods)
- [Contract Verification](#contract-verification)
- [Post-Deployment](#post-deployment)
- [Interacting with Deployed Contracts](#interacting-with-deployed-contracts)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Accounts

1. **Infura Account**
   - Sign up at [infura.io](https://infura.io)
   - Create a new project to get your API key
   - This provides RPC access to Ethereum and Polygon testnets

2. **Etherscan Account** (for Sepolia)
   - Sign up at [etherscan.io](https://etherscan.io)
   - Generate an API key from your account settings

3. **Polygonscan Account** (for Polygon Mumbai)
   - Sign up at [polygonscan.com](https://polygonscan.com)
   - Generate an API key from your account settings

### Testnet Funds

#### Sepolia ETH
- Get free Sepolia ETH from faucets:
  - [Alchemy Sepolia Faucet](https://sepoliafaucet.alchemy.com/)
  - [Infura Sepolia Faucet](https://www.infura.io/faucet/sepolia)
  - [Google Cloud Faucet](https://cloud.google.com/application/web3/faucet/ethereum/sepolia)

#### Polygon Mumbai MATIC
- Get free Mumbai MATIC from faucets:
  - [Polygon Mumbai Faucet](https://mumbai.polygonscan.com/faucet)
  - [Alchemy Mumbai Faucet](https://mumbai.alchemy.com/faucet)

### Local Development Setup

```bash
# Install Node.js 18+ (recommended via nvm)
nvm install 18
nvm use 18

# Install Python 3.11+
python --version  # Should be 3.11+

# Clone the repository
git clone https://github.com/your-org/PRSM.git
cd PRSM

# Install contract dependencies
cd contracts
npm install
cd ..
```

## Network Configuration

### Supported Testnets

| Network | Chain ID | Native Token | Explorer |
|---------|----------|--------------|----------|
| Sepolia | 11155111 | ETH | [sepolia.etherscan.io](https://sepolia.etherscan.io) |
| Polygon Mumbai | 80001 | MATIC | [mumbai.polygonscan.com](https://mumbai.polygonscan.com) |

### Configuration File

The deployment configuration is stored in [`contracts/deployment-config.json`](../contracts/deployment-config.json):

```json
{
  "testnets": {
    "sepolia": {
      "chainId": 11155111,
      "rpcUrl": "https://sepolia.infura.io/v3/${INFURA_KEY}",
      "explorer": "https://sepolia.etherscan.io"
    },
    "polygon_mumbai": {
      "chainId": 80001,
      "rpcUrl": "https://polygon-mumbai.infura.io/v3/${INFURA_KEY}",
      "explorer": "https://mumbai.polygonscan.com"
    }
  },
  "ftns_token": {
    "name": "PRSM Fungible Tokens for Node Support",
    "symbol": "FTNS",
    "decimals": 18,
    "initialSupply": "100000000000000000000000000"
  }
}
```

## Environment Setup

### 1. Create Environment File

Copy the template and configure your credentials:

```bash
cp config/secure.env.template config/secure.env
```

### 2. Configure Required Variables

Edit `config/secure.env` with your values:

```bash
# Infura API Key
INFURA_KEY=your_infura_api_key_here

# Deployer Private Key (create a NEW wallet for testnet - never use mainnet keys!)
DEPLOYER_PRIVATE_KEY=your_testnet_wallet_private_key_here

# Etherscan API Key (for Sepolia verification)
ETHERSCAN_API_KEY=your_etherscan_api_key_here

# Polygonscan API Key (for Mumbai verification)
POLYGONSCAN_API_KEY=your_polygonscan_api_key_here
```

### 3. Create `.env` in Contracts Directory

Create `contracts/.env`:

```bash
# RPC URLs
POLYGON_MUMBAI_RPC_URL=https://polygon-mumbai.infura.io/v3/YOUR_INFURA_KEY
POLYGON_MAINNET_RPC_URL=https://polygon-mainnet.infura.io/v3/YOUR_INFURA_KEY
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY

# Private Key (without 0x prefix)
PRIVATE_KEY=your_deployer_private_key

# API Keys for verification
POLYGONSCAN_API_KEY=your_polygonscan_api_key
ETHERSCAN_API_KEY=your_etherscan_api_key
```

## Deployment Methods

### Method 1: GitHub Actions (Recommended)

The easiest way to deploy is via GitHub Actions:

1. **Configure GitHub Secrets**
   
   Go to your repository → Settings → Secrets and variables → Actions
   
   Add the following secrets:
   - `INFURA_KEY` - Your Infura API key
   - `DEPLOYER_PRIVATE_KEY` - Your deployer wallet private key
   - `ETHERSCAN_API_KEY` - Your Etherscan API key
   - `POLYGONSCAN_API_KEY` - Your Polygonscan API key

2. **Trigger Deployment**
   
   - Go to Actions → "Deploy FTNS to Testnet"
   - Click "Run workflow"
   - Select network (`polygon_mumbai` or `sepolia`)
   - Select contract type (`simple` or `full`)
   - Click "Run workflow"

3. **Review Results**
   
   - Check the workflow logs for deployment details
   - Download the deployment artifacts for contract addresses
   - View the deployment summary in the workflow run

### Method 2: Local Hardhat Deployment

Deploy directly from your local machine:

```bash
cd contracts

# Deploy to Polygon Mumbai (Simple Token)
npx hardhat run scripts/deploy-simple.js --network polygon-mumbai

# Deploy to Sepolia (Simple Token)
npx hardhat run scripts/deploy-simple.js --network sepolia

# Deploy full FTNS system (if available)
npx hardhat run scripts/deploy.js --network polygon-mumbai
```

### Method 3: Python Deployment Script

Use the Python deployment CLI:

```bash
# Deploy to Polygon Mumbai
python scripts/deploy_contracts.py deploy --network polygon_mumbai

# Check deployment status
python scripts/deploy_contracts.py status --network polygon_mumbai

# Get deployment instructions
python scripts/deploy_contracts.py instructions --network polygon_mumbai
```

## Contract Verification

### Automatic Verification (Hardhat)

After deployment, Hardhat can automatically verify contracts:

```bash
cd contracts

# Verify on Polygon Mumbai
npx hardhat verify --network polygon-mumbai <CONTRACT_ADDRESS>

# Verify on Sepolia
npx hardhat verify --network sepolia <CONTRACT_ADDRESS>
```

### Manual Verification

If automatic verification fails, verify manually:

#### Etherscan (Sepolia)

1. Go to [sepolia.etherscan.io/verifyContract](https://sepolia.etherscan.io/verifyContract)
2. Enter contract address
3. Select "Solidity (Standard-Json-Input)"
4. Upload the compiled artifact from `contracts/artifacts/`
5. Click "Verify"

#### Polygonscan (Mumbai)

1. Go to [mumbai.polygonscan.com/verifyContract](https://mumbai.polygonscan.com/verifyContract)
2. Enter contract address
3. Select "Solidity (Standard-Json-Input)"
4. Upload the compiled artifact
5. Click "Verify"

### Proxy Contract Verification

For UUPS proxy contracts, verify both the proxy and implementation:

```bash
# Get implementation address
npx hardhat run scripts/verify-deployment.js --network polygon-mumbai

# Verify implementation
npx hardhat verify --network polygon-mumbai <IMPLEMENTATION_ADDRESS>
```

## Post-Deployment

### Update PRSM Configuration

After successful deployment, update your PRSM configuration:

```bash
# Add to your .env file
FTNS_TOKEN_ADDRESS=<deployed_contract_address>
WEB3_NETWORK=polygon_mumbai  # or sepolia
WEB3_MONITORING_ENABLED=true
```

### Save Deployment Information

Deployment data is automatically saved to:
- `contracts/deployments/<network>-simple-<timestamp>.json`

Example deployment record:
```json
{
  "network": "polygon-mumbai",
  "chainId": 80001,
  "deployer": "0x...",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "contracts": {
    "ftnsToken": "0x..."
  }
}
```

## Interacting with Deployed Contracts

### Using Hardhat Console

```bash
cd contracts
npx hardhat console --network polygon-mumbai

# Get contract instance
const FTNS = await ethers.getContractFactory("FTNSTokenSimple");
const ftns = await FTNS.attach("0x_YOUR_CONTRACT_ADDRESS");

// Check token info
await ftns.name();     // "PRSM Fungible Tokens for Node Support"
await ftns.symbol();   // "FTNS"
await ftns.decimals(); // 18
await ftns.totalSupply(); // BigNumber { value: "100000000000000000000000000" }

// Check balance
const [signer] = await ethers.getSigners();
await ftns.balanceOf(signer.address);
```

### Using Python (web3.py)

```python
from web3 import Web3
from prsm.web3.contract_manager import FTNSTokenManager

# Connect to deployed contract
manager = FTNSTokenManager(
    contract_address="0x_YOUR_CONTRACT_ADDRESS",
    network="polygon_mumbai"
)

# Get token info
print(f"Name: {manager.get_name()}")
print(f"Symbol: {manager.get_symbol()}")
print(f"Total Supply: {manager.get_total_supply()}")
```

### Using Etherscan/Polygonscan

1. Navigate to your contract on the block explorer
2. Click "Contract" → "Write Contract"
3. Connect your wallet (MetaMask, etc.)
4. Call contract functions directly

## Contract Functions

### FTNSTokenSimple

| Function | Role Required | Description |
|----------|---------------|-------------|
| `mintReward(address, uint256)` | MINTER_ROLE | Mint new tokens (up to MAX_SUPPLY) |
| `burnFrom(address, uint256)` | BURNER_ROLE | Burn tokens from address |
| `pause()` | PAUSER_ROLE | Pause all transfers |
| `unpause()` | PAUSER_ROLE | Resume transfers |
| `transfer(address, uint256)` | None | Transfer tokens |
| `approve(address, uint256)` | None | Approve spender |

### Granting Roles

```javascript
// In Hardhat console
const MINTER_ROLE = await ftns.MINTER_ROLE();
await ftns.grantRole(MINTER_ROLE, "0x_RECIPIENT_ADDRESS");
```

## Troubleshooting

### Common Issues

#### "insufficient funds for gas"
- Ensure your deployer wallet has enough testnet tokens
- Get more from faucets (see [Testnet Funds](#testnet-funds))

#### "nonce too low" or "nonce too high"
- Reset your wallet nonce in MetaMask: Settings → Advanced → Reset Account
- Or wait for pending transactions to complete

#### "contract creation code overlap"
- Clear Hardhat cache: `npx hardhat clean`
- Delete `artifacts/` and `cache/` directories
- Recompile: `npx hardhat compile`

#### Verification Failed
- Wait a few minutes after deployment for the block explorer to index
- Ensure you're using the correct compiler settings
- For proxy contracts, verify the implementation first

### Getting Help

1. Check the [contracts/README.md](../contracts/README.md) for contract-specific details
2. Review the [DEPLOYMENT_GUIDE.md](../contracts/DEPLOYMENT_GUIDE.md)
3. Open an issue on GitHub with deployment logs

## Security Considerations

### ⚠️ IMPORTANT

1. **Never use mainnet private keys for testnet deployment**
   - Create a new wallet specifically for testnet
   - Fund it only with testnet tokens

2. **Never commit private keys to version control**
   - Use environment variables
   - Add `.env` to `.gitignore`

3. **Testnet tokens have no real value**
   - They cannot be exchanged for real assets
   - Faucets provide free tokens for testing

4. **Contract upgrades**
   - FTNSTokenSimple uses UUPS proxy pattern
   - Only DEFAULT_ADMIN_ROLE can upgrade
   - Always test upgrades on testnet first

## Next Steps

After successful testnet deployment:

1. **Test Integration**
   ```bash
   python scripts/test_web3_integration.py
   ```

2. **Start PRSM with Web3 Support**
   ```bash
   python -m uvicorn prsm.api.main:app --reload
   ```

3. **Monitor Contract Events**
   - Set up event listeners in your application
   - Use block explorer for transaction history

4. **Prepare for Mainnet**
   - Complete security audit
   - Review all role assignments
   - Document operational procedures
