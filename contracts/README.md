# PRSM FTNS Smart Contracts

Production-ready smart contracts for the PRSM Federated Token Network System (FTNS) deployed on Polygon.

## 🏗️ Architecture

### Core Contracts

1. **FTNSToken.sol** - ERC20 token with governance and staking features
2. **FTNSMarketplace.sol** - Decentralized marketplace for AI model rentals
3. **FTNSGovernance.sol** - Quadratic voting governance system
4. **TimelockController.sol** - Timelock for secure governance execution

### Key Features

- ✅ **Multi-tier Balance Management** - Liquid, locked, and staked balances
- ✅ **Governance Integration** - Quadratic voting with staking rewards
- ✅ **Context Allocation** - Token burning for AI computation costs
- ✅ **Marketplace Operations** - Model rental with escrow protection
- ✅ **Reward Distribution** - Automated rewards for participation
- ✅ **Upgradeability** - UUPS proxy pattern for future improvements
- ✅ **Privacy Ready** - Integration points for privacy pools

## 🚀 Quick Start

### Prerequisites

```bash
node >= 16.0.0
npm >= 7.0.0
```

### Installation

```bash
cd contracts
npm install
```

### Environment Setup

1. Copy environment template:
```bash
cp .env.example .env
```

2. Configure your `.env` file:
```env
# Polygon Network Configuration
POLYGON_MUMBAI_RPC_URL=https://rpc-mumbai.maticvigil.com
POLYGON_MAINNET_RPC_URL=https://polygon-rpc.com

# Wallet Configuration (Use a dedicated deployment wallet)
PRIVATE_KEY=your_private_key_here

# API Keys
POLYGONSCAN_API_KEY=your_polygonscan_api_key
```

### Compilation

```bash
npm run compile
```

### Testing

```bash
# Run all tests
npm test

# Run with gas reporting
REPORT_GAS=true npm test

# Run specific test file
npx hardhat test test/FTNSToken.test.js
```

## 📋 Deployment

### Polygon Mumbai Testnet

1. **Get Test MATIC**:
   - Visit [Polygon Faucet](https://faucet.polygon.technology/)
   - Request test MATIC for your deployment address

2. **Deploy Contracts**:
```bash
npm run deploy:testnet
```

3. **Verify Contracts**:
```bash
npx hardhat verify --network polygon-mumbai <CONTRACT_ADDRESS>
```

### Polygon Mainnet

⚠️ **WARNING**: Mainnet deployment involves real money. Test thoroughly on testnet first.

```bash
npm run deploy:mainnet
```

## 📊 Contract Addresses

### Polygon Mumbai Testnet
- **FTNS Token**: `TBD`
- **Marketplace**: `TBD`
- **Governance**: `TBD`
- **Timelock**: `TBD`

### Polygon Mainnet
- **FTNS Token**: `TBD`
- **Marketplace**: `TBD`
- **Governance**: `TBD`
- **Timelock**: `TBD`

## 🔧 Configuration

### Token Parameters
```solidity
MAX_SUPPLY = 1,000,000,000 FTNS  // 1 billion total supply
INITIAL_SUPPLY = 100,000,000 FTNS // 100 million initial mint
```

### Marketplace Parameters
```solidity
PLATFORM_FEE_RATE = 250         // 2.5% platform fee
MAX_FEE_RATE = 1000             // 10% maximum fee
```

### Governance Parameters
```solidity
VOTING_DELAY = 7200             // 1 day (in blocks)
VOTING_PERIOD = 50400           // 7 days (in blocks)
QUORUM_FRACTION = 10            // 10% quorum required
TIMELOCK_DELAY = 172800         // 2 days (in seconds)
```

## 🛠️ Development

### Local Development

1. **Start Local Node**:
```bash
npx hardhat node
```

2. **Deploy to Local Network**:
```bash
npx hardhat run scripts/deploy.js --network localhost
```

3. **Interact with Contracts**:
```bash
npx hardhat console --network localhost
```

### Testing Scenarios

```javascript
// Example: Test token staking
const ftnsToken = await ethers.getContractAt("FTNSToken", "0x...");
await ftnsToken.stakeForGovernance(
  ethers.utils.parseEther("1000"), // 1000 FTNS
  365 * 24 * 3600                  // 1 year duration
);
```

## 🔐 Security

### Access Control Roles

| Role | Description | Functions |
|------|-------------|-----------|
| `DEFAULT_ADMIN_ROLE` | Contract administration | Grant/revoke roles, upgrades |
| `MINTER_ROLE` | Token minting | Mint rewards, distribute dividends |
| `BURNER_ROLE` | Token burning | Burn tokens from accounts |
| `PAUSER_ROLE` | Emergency pause | Pause/unpause contracts |
| `GOVERNANCE_ROLE` | Governance actions | Unlock tokens, governance operations |

### Security Features

- ✅ **ReentrancyGuard** - Protection against reentrancy attacks
- ✅ **Pausable** - Emergency pause functionality
- ✅ **AccessControl** - Role-based permission system
- ✅ **TimelockController** - Delayed execution for governance
- ✅ **UUPS Upgrades** - Secure upgradeability pattern

### Audit Checklist

- [ ] External security audit
- [ ] Formal verification of critical functions
- [ ] Economic model validation
- [ ] Gas optimization review
- [ ] Integration testing with PRSM backend

## 🎯 Usage Examples

### Token Operations

```javascript
// Get account information
const accountInfo = await ftnsToken.getAccountInfo(userAddress);
console.log("Liquid balance:", ethers.utils.formatEther(accountInfo.liquid));

// Stake for governance
await ftnsToken.stakeForGovernance(
  ethers.utils.parseEther("1000"),
  365 * 24 * 3600
);

// Allocate context for AI operations
await ftnsToken.allocateContext(ethers.utils.parseEther("100"));
```

### Marketplace Operations

```javascript
// Create model listing
await marketplace.createListing(
  "QmModelHashHere",              // IPFS CID
  "High-Performance Language Model", // Title
  "Advanced AI model for...",     // Description
  ethers.utils.parseEther("10"),  // 10 FTNS per hour
  1,                              // Min 1 hour rental
  168                             // Max 1 week rental
);

// Rent a model
await marketplace.rentModel(
  1,                              // Listing ID
  24                              // 24 hour rental
);
```

### Governance Operations

```javascript
// Create proposal
await governance.proposeWithDetails(
  [targetContract.address],       // Targets
  [0],                           // Values
  [calldata],                    // Call data
  "Proposal description",        // Description
  0,                             // Technical category
  "QmProposalDocumentHash",      // IPFS hash
  false                          // Not emergency
);

// Vote on proposal
await governance.castVoteWithReason(
  proposalId,
  1,                             // Vote for
  "Supporting this proposal because..."
);
```

## 📈 Economic Model

### Token Distribution

- **40%** - Research Community Rewards
- **20%** - Development Team (4-year vesting)
- **15%** - Early Adopters and Partnerships
- **15%** - Treasury for Operations
- **10%** - Liquidity Provision

### Fee Structure

- **Context Allocation**: 0.1-10 FTNS per query
- **Marketplace Transactions**: 2.5% platform fee
- **Governance Participation**: 5 FTNS per vote
- **Privacy Transactions**: 0.01 FTNS mixer fee

### Reward Mechanisms

- **Data Contribution**: 0.05 FTNS per MB
- **Model Contribution**: 100 FTNS base + performance bonus
- **Research Publication**: 500 FTNS + citation rewards
- **Teaching Success**: 20 FTNS + improvement multiplier
- **Governance Participation**: 5 FTNS per vote/proposal

## 🔗 Integration

### Backend Integration

```python
# Python example using web3.py
from web3 import Web3

# Connect to Polygon
w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))

# Load contract
ftns_token = w3.eth.contract(
    address='0x...',
    abi=ftns_abi
)

# Check balance
balance = ftns_token.functions.balanceOf(user_address).call()
```

### Frontend Integration

```javascript
// JavaScript example using ethers.js
import { ethers } from 'ethers';

// Connect to user's wallet
const provider = new ethers.providers.Web3Provider(window.ethereum);
const signer = provider.getSigner();

// Load contract
const ftnsToken = new ethers.Contract(
  '0x...',
  ftnsAbi,
  signer
);

// Get account info
const accountInfo = await ftnsToken.getAccountInfo(userAddress);
```

## 🆘 Troubleshooting

### Common Issues

1. **"Insufficient funds" error**:
   - Ensure wallet has enough MATIC for gas fees
   - Check FTNS token balance for transactions

2. **"AccessControl" error**:
   - Verify account has required role
   - Check if using correct signer address

3. **"Transfer exceeds liquid balance"**:
   - Account for locked and staked tokens
   - Use `liquidBalance()` to check transferable amount

### Gas Optimization

- Use batch operations when possible
- Stake for longer periods to reduce transaction frequency
- Monitor gas prices and deploy during low-usage periods

## 📝 License

MIT License - see [LICENSE](../LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `npm test`
4. Submit a pull request

## 📞 Support

- **Documentation**: [PRSM Docs](https://docs.prsm.ai)
- **Discord**: [PRSM Community](https://discord.gg/prsm)
- **Issues**: [GitHub Issues](https://github.com/prsm/issues)

---

**⚠️ Important**: These contracts handle real financial value. Always test thoroughly on testnets before mainnet deployment. Consider professional security audits for production use.