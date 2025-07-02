# PRSM FTNS Smart Contract Deployment Guide

Complete guide for deploying FTNS smart contracts to Polygon Mumbai testnet.

## 🚀 Quick Deployment (5 minutes)

### Prerequisites

1. **Node.js 16+** installed
2. **Git** repository access
3. **Polygon Mumbai testnet MATIC** for gas fees

### Step 1: Environment Setup

```bash
cd contracts
cp .env.example .env
```

### Step 2: Configure Environment

Edit `.env` file with your deployment wallet:

```env
# Required: Deployment wallet private key
PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef

# Required: RPC URL (default works for most cases)
POLYGON_MUMBAI_RPC_URL=https://rpc-mumbai.maticvigil.com

# Optional: For contract verification on PolygonScan
POLYGONSCAN_API_KEY=your_api_key_here

# Optional: Treasury address (defaults to deployer address)
TREASURY_ADDRESS=0x1234567890123456789012345678901234567890
```

### Step 3: Get Testnet MATIC

Visit [Polygon Faucet](https://faucet.polygon.technology/) and request MATIC for your deployment address.

**Minimum required**: 0.1 MATIC for deployment

### Step 4: Install Dependencies

```bash
npm install
```

### Step 5: Deploy Contracts

```bash
# Deploy to Mumbai testnet
npm run deploy:testnet

# Or use Hardhat directly
npx hardhat run scripts/deploy-simple.js --network polygon-mumbai
```

### Step 6: Verify Deployment

After deployment, run verification:

```bash
# Verify contract on PolygonScan (requires API key)
npx hardhat verify --network polygon-mumbai <CONTRACT_ADDRESS>
```

## 📋 Deployment Output

Successful deployment will show:

```
🚀 Starting PRSM FTNS Simple Token Deployment
Network: polygon-mumbai
Deployer address: 0x1234...
Deployer balance: 0.5 MATIC

📄 Deploying FTNS Token (Simple)...
✅ FTNS Token deployed to: 0xabcd1234...
   Implementation: 0xefgh5678...

🔍 Verifying deployment...
- Token: PRSM Fungible Tokens for Node Support (FTNS) with 18 decimals
- Total Supply: 100000000.0 FTNS
- Deployer Balance: 100000000.0 FTNS

📄 Deployment data saved to: deployments/polygon-mumbai-simple-1234567890.json

🎉 PRSM FTNS SIMPLE DEPLOYMENT COMPLETE!
FTNS Token: 0xabcd1234...

🧪 Testnet Verification Command:
npx hardhat verify --network polygon-mumbai 0xabcd1234...
```

## 🔧 Advanced Configuration

### Custom Gas Settings

```env
# Gas Configuration
GAS_LIMIT_MULTIPLIER=1.5    # 50% safety margin
GAS_PRICE_MULTIPLIER=1.2    # 20% faster confirmation
```

### Custom Token Parameters

```env
# Token Configuration
INITIAL_SUPPLY=100000000     # 100M tokens initial supply
MAX_SUPPLY=1000000000       # 1B tokens maximum supply
TREASURY_ADDRESS=0x...      # Custom treasury address
```

### Multiple Network Deployment

```bash
# Deploy to localhost (for testing)
npm run deploy:local

# Deploy to mainnet (production)
npm run deploy:mainnet
```

## 📊 Post-Deployment Integration

### Update PRSM Configuration

After deployment, update your PRSM environment:

```env
# Add to your PRSM .env file
FTNS_TOKEN_ADDRESS=0xabcd1234...
WEB3_NETWORK=polygon_mumbai
WEB3_MONITORING_ENABLED=true
```

### Initialize Web3 Services

```python
# In your PRSM application
from prsm.web3.web3_service import initialize_web3_services

# Initialize with deployed contract
await initialize_web3_services(
    network="polygon_mumbai",
    contract_addresses={
        "ftns_token": "0xabcd1234..."
    }
)
```

### Test Integration

```bash
# Test API endpoints
curl -X GET "http://localhost:8000/api/v1/web3/wallet/info"

# Test contract interaction
curl -X POST "http://localhost:8000/api/v1/web3/contracts/initialize" \
  -H "Content-Type: application/json" \
  -d '{"ftns_token": "0xabcd1234..."}'
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Insufficient funds"**
   - Get more MATIC from [faucet](https://faucet.polygon.technology/)
   - Check your wallet balance

2. **"Network connection failed"**
   - Verify RPC URL is correct
   - Check internet connection
   - Try alternative RPC: `https://matic-mumbai.chainstacklabs.com`

3. **"Compilation failed"**
   - Run `npm install` to install dependencies
   - Check Node.js version (requires 16+)

4. **"Private key invalid"**
   - Ensure private key starts with `0x`
   - Verify private key is 64 characters (32 bytes)
   - Never commit private keys to git

### Alternative RPC URLs

If default RPC fails, try these alternatives:

```env
# Alternative Mumbai RPC URLs
POLYGON_MUMBAI_RPC_URL=https://matic-mumbai.chainstacklabs.com
# or
POLYGON_MUMBAI_RPC_URL=https://rpc.ankr.com/polygon_mumbai
# or  
POLYGON_MUMBAI_RPC_URL=https://polygon-mumbai.blockpi.network/v1/rpc/public
```

### Gas Estimation Issues

If gas estimation fails:

```bash
# Deploy with higher gas limit
npx hardhat run scripts/deploy-simple.js --network polygon-mumbai --gas-limit 2000000
```

## 🔐 Security Notes

- **Never commit `.env` files** to version control
- **Use dedicated deployment wallets** with minimal funds
- **Verify contract addresses** before integration
- **Test thoroughly** on testnet before mainnet

## 📞 Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Verify all prerequisites are met
3. Check Polygon Mumbai network status
4. Review deployment logs for specific errors

## 🎯 Next Steps

After successful deployment:

1. ✅ Update PRSM configuration with contract address
2. ✅ Test Web3 API endpoints
3. ✅ Verify contract on PolygonScan
4. ✅ Begin user testing and onboarding
5. ✅ Monitor contract events and transactions

---

**⚠️ Important**: This deploys to testnet using test tokens. For production deployment to Polygon mainnet, ensure thorough testing and security audits.