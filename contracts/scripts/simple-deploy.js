/**
 * Simple deployment script for FTNS token to Polygon Mumbai testnet
 * Uses web3.js for direct blockchain interaction without Hardhat complexity
 */

const Web3 = require('web3');
const fs = require('fs');
const path = require('path');

// Contract ABI and bytecode (for FTNSTokenSimple)
const CONTRACT_ABI = [
  {
    "inputs": [],
    "name": "name",
    "outputs": [{"name": "", "type": "string"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "symbol",
    "outputs": [{"name": "", "type": "string"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "decimals",
    "outputs": [{"name": "", "type": "uint8"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "totalSupply",
    "outputs": [{"name": "", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"name": "_owner", "type": "address"}],
    "name": "balanceOf",
    "outputs": [{"name": "balance", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {"name": "_to", "type": "address"},
      {"name": "_value", "type": "uint256"}
    ],
    "name": "transfer",
    "outputs": [{"name": "", "type": "bool"}],
    "stateMutability": "nonpayable",
    "type": "function"
  }
];

// Configuration
const CONFIG = {
  MUMBAI_RPC: process.env.POLYGON_MUMBAI_RPC_URL || 'https://rpc-mumbai.maticvigil.com',
  PRIVATE_KEY: process.env.PRIVATE_KEY,
  INITIAL_SUPPLY: '100000000000000000000000000', // 100M tokens with 18 decimals
  TOKEN_NAME: 'PRSM Federated Token Network System',
  TOKEN_SYMBOL: 'FTNS'
};

// Simple ERC20 contract bytecode (minimal implementation)
const SIMPLE_ERC20_BYTECODE = '0x608060405234801561001057600080fd5b50604051610c9e380380610c9e833981810160405281019061003291906101a4565b81600390805190602001906100489291906100e8565b50806004908051906020019061005f9291906100e8565b506012600560006101000a81548160ff021916908360ff1602179055508260065461008a91906102f8565b6000803373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055508260065461013a91906102f8565b600281905550505050610415565b6101f3565b600080fd5b600080fd5b6000819050919050565b610110816100fd565b811461011b57600080fd5b50565b60008151905061012d81610107565b92915050565b6000806040838503121561014a5761014961010e565b5b60006101588582860161011e565b92505060206101698582860161011e565b9150509250929050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806101bb57607f821691505b6020821081036101ce576101cd610174565b5b50919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b6000610213826100fd565b915061021e836100fd565b9250817fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff048311821515161561025757610256610204565b5b828202905092915050565b600061026d826100fd565b9150610278836100fd565b92508261028857610287610233565b5b828204905092915050565b6108b4806102a16000396000f3fe';

async function deployToken() {
  console.log('🚀 PRSM FTNS Token Deployment to Polygon Mumbai');
  
  try {
    // Validate environment
    if (!CONFIG.PRIVATE_KEY) {
      throw new Error('PRIVATE_KEY environment variable is required');
    }
    
    // Initialize Web3
    console.log('🌐 Connecting to Polygon Mumbai...');
    const web3 = new Web3(CONFIG.MUMBAI_RPC);
    
    // Verify connection
    const isConnected = await web3.eth.net.isListening();
    if (!isConnected) {
      throw new Error('Failed to connect to Polygon Mumbai network');
    }
    
    const chainId = await web3.eth.getChainId();
    console.log('✅ Connected to network, Chain ID:', chainId);
    
    if (chainId !== 80001) {
      console.warn('⚠️  Warning: Not connected to Polygon Mumbai (Chain ID should be 80001)');
    }
    
    // Setup account
    const account = web3.eth.accounts.privateKeyToAccount(CONFIG.PRIVATE_KEY);
    web3.eth.accounts.wallet.add(account);
    web3.eth.defaultAccount = account.address;
    
    console.log('👤 Deployer address:', account.address);
    
    // Check balance
    const balance = await web3.eth.getBalance(account.address);
    const balanceInMatic = web3.utils.fromWei(balance, 'ether');
    console.log('💰 Deployer balance:', balanceInMatic, 'MATIC');
    
    if (parseFloat(balanceInMatic) < 0.01) {
      console.warn('⚠️  Low balance! Get MATIC from faucet: https://faucet.polygon.technology/');
      console.warn('   Minimum recommended: 0.01 MATIC');
    }
    
    // For now, let's create a deployment record as if we deployed
    // In a real deployment, this would deploy the actual contract
    console.log('\n📋 Simulating deployment (no actual contract deployed)...');
    
    // Generate a mock contract address for testing
    const mockContractAddress = '0x' + web3.utils.randomHex(20).slice(2);
    
    console.log('✅ Mock FTNS Token deployed to:', mockContractAddress);
    
    // Create deployment record
    const deploymentData = {
      network: 'polygon-mumbai',
      chainId: chainId,
      deployer: account.address,
      timestamp: new Date().toISOString(),
      contracts: {
        ftnsToken: mockContractAddress
      },
      config: {
        tokenName: CONFIG.TOKEN_NAME,
        tokenSymbol: CONFIG.TOKEN_SYMBOL,
        initialSupply: CONFIG.INITIAL_SUPPLY,
        decimals: 18
      },
      verification: {
        name: CONFIG.TOKEN_NAME,
        symbol: CONFIG.TOKEN_SYMBOL,
        totalSupply: CONFIG.INITIAL_SUPPLY
      },
      deployment_type: 'mock', // Indicates this is a test deployment
      notes: 'Mock deployment for development and testing purposes'
    };
    
    // Save deployment data
    const deploymentsDir = path.join(__dirname, '../deployments');
    if (!fs.existsSync(deploymentsDir)) {
      fs.mkdirSync(deploymentsDir, { recursive: true });
    }
    
    const deploymentFile = path.join(
      deploymentsDir,
      `polygon-mumbai-mock-${Date.now()}.json`
    );
    
    fs.writeFileSync(deploymentFile, JSON.stringify(deploymentData, null, 2));
    
    console.log('\n📄 Deployment record saved to:', deploymentFile);
    
    // Display next steps
    console.log('\n🎉 MOCK DEPLOYMENT COMPLETE!');
    console.log('\n📋 Integration Steps:');
    console.log('1. Update PRSM .env file:');
    console.log(`   FTNS_TOKEN_ADDRESS=${mockContractAddress}`);
    console.log(`   WEB3_NETWORK=polygon_mumbai`);
    console.log('   WEB3_MONITORING_ENABLED=true');
    
    console.log('\n2. Test Web3 integration:');
    console.log('   # Start PRSM API server');
    console.log('   cd .. && python -m uvicorn prsm.api.main:app --reload');
    console.log('   ');
    console.log('   # Test API endpoints');
    console.log('   curl http://localhost:8000/api/v1/web3/wallet/info');
    
    console.log('\n3. Initialize contracts in PRSM:');
    console.log('   curl -X POST http://localhost:8000/api/v1/web3/contracts/initialize \\');
    console.log('     -H "Content-Type: application/json" \\');
    console.log(`     -d '{"ftns_token": "${mockContractAddress}"}'`);
    
    console.log('\n📝 Note: This is a MOCK deployment for development.');
    console.log('🚀 For real deployment, implement actual contract deployment code.');
    
    return deploymentData;
    
  } catch (error) {
    console.error('\n❌ Deployment failed:', error.message);
    throw error;
  }
}

// Create environment setup helper
function createEnvironmentFile() {
  const envContent = `# PRSM FTNS Testnet Configuration
# Copy this to your main PRSM .env file

# Web3 Configuration
FTNS_TOKEN_ADDRESS=
WEB3_NETWORK=polygon_mumbai
WEB3_MONITORING_ENABLED=true
FAUCET_INTEGRATION_ENABLED=true

# Polygon Mumbai RPC
POLYGON_MUMBAI_RPC_URL=https://rpc-mumbai.maticvigil.com

# Development wallet (DO NOT use in production)
WALLET_PRIVATE_KEY=

# Optional: PolygonScan API for verification
POLYGONSCAN_API_KEY=
`;

  const envFile = path.join(__dirname, '../../.env.testnet');
  fs.writeFileSync(envFile, envContent);
  console.log('📝 Created environment template:', envFile);
}

if (require.main === module) {
  deployToken()
    .then((deploymentData) => {
      createEnvironmentFile();
      console.log('\n✅ Setup complete! Check the deployment guide for next steps.');
      process.exit(0);
    })
    .catch((error) => {
      console.error('💥 Setup failed:', error);
      process.exit(1);
    });
}

module.exports = { deployToken };