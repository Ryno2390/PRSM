/**
 * PRSM Web3 SDK
 * =============
 * 
 * Complete JavaScript SDK for integrating with PRSM smart contracts
 * Works in both browser and Node.js environments
 */

const { ethers } = require('ethers');

// Contract ABIs
const CONTRACT_ABIS = {
  FTNSToken: [
    "function name() view returns (string)",
    "function symbol() view returns (string)",
    "function decimals() view returns (uint8)",
    "function totalSupply() view returns (uint256)",
    "function balanceOf(address owner) view returns (uint256)",
    "function transfer(address to, uint256 amount) returns (bool)",
    "function approve(address spender, uint256 amount) returns (bool)",
    "function transferFrom(address from, address to, uint256 amount) returns (bool)",
    "function allowance(address owner, address spender) view returns (uint256)",
    "function getLiquidBalance(address user) view returns (uint256)",
    "function getContextAllocation(address user, string context) view returns (uint256)",
    "function getUserLocks(address user) view returns (tuple[])",
    "function getUserStakes(address user) view returns (tuple[])",
    "function stakeTokens(uint256 amount, string context)",
    "function unstakeTokens(uint256 stakeIndex)",
    "function lockTokens(uint256 amount, uint256 duration, string context)",
    "function unlockTokens(uint256 lockIndex)",
    "function calculateStakingRewards(uint256 amount, uint256 duration, string context) view returns (uint256)",
    "event Transfer(address indexed from, address indexed to, uint256 value)",
    "event Approval(address indexed owner, address indexed spender, uint256 value)",
    "event TokensStaked(address indexed user, uint256 amount, string context)",
    "event TokensUnstaked(address indexed user, uint256 amount, string context)",
    "event TokensLocked(address indexed user, uint256 amount, uint256 unlockTime, string context)",
    "event TokensUnlocked(address indexed user, uint256 amount, string context)"
  ],
  
  FTNSMarketplace: [
    "function listService(string context, string name, string description, uint8 serviceType, uint256 price, uint256 minStake) returns (uint256)",
    "function purchaseService(uint256 serviceId) returns (uint256)",
    "function releaseEscrow(uint256 purchaseId)",
    "function submitReview(uint256 serviceId, uint8 rating, string comment)",
    "function createDispute(uint256 purchaseId, string reason) returns (uint256)",
    "function getService(uint256 serviceId) view returns (tuple)",
    "function getServicesByContext(string context) view returns (uint256[])",
    "function getServicesByProvider(address provider) view returns (uint256[])",
    "function getServiceReviews(uint256 serviceId) view returns (tuple[])",
    "function getPurchasesByBuyer(address buyer) view returns (uint256[])",
    "event ServiceListed(uint256 indexed serviceId, address indexed provider, string context, uint256 price, string serviceType)",
    "event ServicePurchased(uint256 indexed serviceId, address indexed buyer, address indexed provider, uint256 amount, string context)",
    "event ReviewSubmitted(uint256 indexed serviceId, address indexed reviewer, uint8 rating, string comment)"
  ],
  
  FTNSGovernance: [
    "function propose(address[] targets, uint256[] values, bytes[] calldatas, string description) returns (uint256)",
    "function proposeContext(address[] targets, uint256[] values, bytes[] calldatas, string description, string context) returns (uint256)",
    "function castVote(uint256 proposalId, uint8 support) returns (uint256)",
    "function execute(address[] targets, uint256[] values, bytes[] calldatas, bytes32 descriptionHash) returns (uint256)",
    "function state(uint256 proposalId) view returns (uint8)",
    "function votingDelay() view returns (uint256)",
    "function votingPeriod() view returns (uint256)",
    "function proposalThreshold() view returns (uint256)",
    "function quorum(uint256 blockNumber) view returns (uint256)",
    "function getVotes(address account, uint256 blockNumber) view returns (uint256)",
    "function canProposeForContext(address proposer, string context) view returns (bool)"
  ]
};

// Network configurations
const NETWORKS = {
  polygon: {
    name: 'Polygon Mainnet',
    chainId: 137,
    rpcUrl: 'https://polygon-rpc.com',
    explorer: 'https://polygonscan.com',
    nativeCurrency: { name: 'MATIC', symbol: 'MATIC', decimals: 18 }
  },
  mumbai: {
    name: 'Polygon Mumbai',
    chainId: 80001,
    rpcUrl: 'https://rpc-mumbai.maticvigil.com',
    explorer: 'https://mumbai.polygonscan.com',
    nativeCurrency: { name: 'MATIC', symbol: 'MATIC', decimals: 18 }
  }
};

class PRSMWeb3SDK {
  constructor(options = {}) {
    this.network = options.network || 'mumbai';
    this.contractAddresses = options.contractAddresses || {};
    this.provider = null;
    this.signer = null;
    this.contracts = {};
    this.eventListeners = new Map();
    
    // Initialize provider
    if (options.provider) {
      this.provider = options.provider;
    } else if (options.privateKey) {
      this.provider = new ethers.JsonRpcProvider(NETWORKS[this.network].rpcUrl);
      this.signer = new ethers.Wallet(options.privateKey, this.provider);
    } else if (typeof window !== 'undefined' && window.ethereum) {
      this.provider = new ethers.BrowserProvider(window.ethereum);
    } else {
      this.provider = new ethers.JsonRpcProvider(NETWORKS[this.network].rpcUrl);
    }
  }

  /**
   * Initialize the SDK by loading contracts
   */
  async initialize() {
    try {
      // Get signer if in browser
      if (!this.signer && this.provider.getSigner) {
        try {
          this.signer = await this.provider.getSigner();
        } catch (e) {
          console.warn('No signer available, read-only mode');
        }
      }

      // Load contracts
      await this.loadContracts();
      
      console.log('PRSM Web3 SDK initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize SDK:', error);
      throw error;
    }
  }

  /**
   * Load smart contracts
   */
  async loadContracts() {
    const contractTypes = ['FTNSToken', 'FTNSMarketplace', 'FTNSGovernance'];
    
    for (const contractType of contractTypes) {
      const address = this.contractAddresses[contractType];
      if (address) {
        const contract = new ethers.Contract(
          address,
          CONTRACT_ABIS[contractType],
          this.signer || this.provider
        );
        this.contracts[contractType] = contract;
      }
    }
  }

  /**
   * Connect wallet (browser only)
   */
  async connectWallet() {
    if (typeof window === 'undefined' || !window.ethereum) {
      throw new Error('MetaMask not available');
    }

    await window.ethereum.request({ method: 'eth_requestAccounts' });
    this.provider = new ethers.BrowserProvider(window.ethereum);
    this.signer = await this.provider.getSigner();
    
    // Reload contracts with signer
    await this.loadContracts();
    
    return await this.signer.getAddress();
  }

  /**
   * Get network information
   */
  async getNetworkInfo() {
    const network = await this.provider.getNetwork();
    return {
      chainId: Number(network.chainId),
      name: network.name,
      ensAddress: network.ensAddress
    };
  }

  // ===================
  // FTNS Token Methods
  // ===================

  /**
   * Get token balance for an address
   */
  async getTokenBalance(address) {
    const contract = this.contracts.FTNSToken;
    if (!contract) throw new Error('FTNSToken contract not loaded');

    const balance = await contract.balanceOf(address);
    return ethers.formatEther(balance);
  }

  /**
   * Get liquid (unstaked/unlocked) balance
   */
  async getLiquidBalance(address) {
    const contract = this.contracts.FTNSToken;
    if (!contract) throw new Error('FTNSToken contract not loaded');

    const balance = await contract.getLiquidBalance(address);
    return ethers.formatEther(balance);
  }

  /**
   * Transfer tokens
   */
  async transferTokens(to, amount) {
    const contract = this.contracts.FTNSToken;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.transfer(to, ethers.parseEther(amount.toString()));
    return await tx.wait();
  }

  /**
   * Stake tokens for a context
   */
  async stakeTokens(amount, context) {
    const contract = this.contracts.FTNSToken;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.stakeTokens(ethers.parseEther(amount.toString()), context);
    return await tx.wait();
  }

  /**
   * Unstake tokens
   */
  async unstakeTokens(stakeIndex) {
    const contract = this.contracts.FTNSToken;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.unstakeTokens(stakeIndex);
    return await tx.wait();
  }

  /**
   * Lock tokens
   */
  async lockTokens(amount, duration, context) {
    const contract = this.contracts.FTNSToken;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.lockTokens(
      ethers.parseEther(amount.toString()),
      duration,
      context
    );
    return await tx.wait();
  }

  /**
   * Unlock tokens
   */
  async unlockTokens(lockIndex) {
    const contract = this.contracts.FTNSToken;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.unlockTokens(lockIndex);
    return await tx.wait();
  }

  /**
   * Get user's stakes
   */
  async getUserStakes(address) {
    const contract = this.contracts.FTNSToken;
    if (!contract) throw new Error('FTNSToken contract not loaded');

    return await contract.getUserStakes(address);
  }

  /**
   * Get user's locks
   */
  async getUserLocks(address) {
    const contract = this.contracts.FTNSToken;
    if (!contract) throw new Error('FTNSToken contract not loaded');

    return await contract.getUserLocks(address);
  }

  /**
   * Get context allocation
   */
  async getContextAllocation(address, context) {
    const contract = this.contracts.FTNSToken;
    if (!contract) throw new Error('FTNSToken contract not loaded');

    const allocation = await contract.getContextAllocation(address, context);
    return ethers.formatEther(allocation);
  }

  // ======================
  // Marketplace Methods
  // ======================

  /**
   * List a service on the marketplace
   */
  async listService(serviceData) {
    const contract = this.contracts.FTNSMarketplace;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const { context, name, description, serviceType, price, minStake } = serviceData;
    
    const tx = await contract.listService(
      context,
      name,
      description,
      serviceType,
      ethers.parseEther(price.toString()),
      ethers.parseEther((minStake || 0).toString())
    );
    
    return await tx.wait();
  }

  /**
   * Purchase a service
   */
  async purchaseService(serviceId) {
    const contract = this.contracts.FTNSMarketplace;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.purchaseService(serviceId);
    return await tx.wait();
  }

  /**
   * Get service details
   */
  async getService(serviceId) {
    const contract = this.contracts.FTNSMarketplace;
    if (!contract) throw new Error('FTNSMarketplace contract not loaded');

    return await contract.getService(serviceId);
  }

  /**
   * Get services by context
   */
  async getServicesByContext(context) {
    const contract = this.contracts.FTNSMarketplace;
    if (!contract) throw new Error('FTNSMarketplace contract not loaded');

    const serviceIds = await contract.getServicesByContext(context);
    
    // Get full service details
    const services = await Promise.all(
      serviceIds.map(id => this.getService(id))
    );
    
    return services;
  }

  /**
   * Submit a review
   */
  async submitReview(serviceId, rating, comment) {
    const contract = this.contracts.FTNSMarketplace;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.submitReview(serviceId, rating, comment);
    return await tx.wait();
  }

  /**
   * Get service reviews
   */
  async getServiceReviews(serviceId) {
    const contract = this.contracts.FTNSMarketplace;
    if (!contract) throw new Error('FTNSMarketplace contract not loaded');

    return await contract.getServiceReviews(serviceId);
  }

  // ===================
  // Governance Methods
  // ===================

  /**
   * Create a proposal
   */
  async createProposal(targets, values, calldatas, description) {
    const contract = this.contracts.FTNSGovernance;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.propose(targets, values, calldatas, description);
    return await tx.wait();
  }

  /**
   * Vote on a proposal
   */
  async vote(proposalId, support) {
    const contract = this.contracts.FTNSGovernance;
    if (!contract || !this.signer) throw new Error('Contract or signer not available');

    const tx = await contract.castVote(proposalId, support);
    return await tx.wait();
  }

  /**
   * Get proposal state
   */
  async getProposalState(proposalId) {
    const contract = this.contracts.FTNSGovernance;
    if (!contract) throw new Error('FTNSGovernance contract not loaded');

    return await contract.state(proposalId);
  }

  /**
   * Get voting power
   */
  async getVotingPower(address, blockNumber) {
    const contract = this.contracts.FTNSGovernance;
    if (!contract) throw new Error('FTNSGovernance contract not loaded');

    const votes = await contract.getVotes(address, blockNumber || 'latest');
    return ethers.formatEther(votes);
  }

  // =================
  // Event Listening
  // =================

  /**
   * Listen to contract events
   */
  addEventListener(contractName, eventName, callback) {
    const contract = this.contracts[contractName];
    if (!contract) throw new Error(`${contractName} contract not loaded`);

    const listener = (...args) => {
      const event = args[args.length - 1];
      callback(event, ...args.slice(0, -1));
    };

    contract.on(eventName, listener);
    
    const key = `${contractName}.${eventName}`;
    if (!this.eventListeners.has(key)) {
      this.eventListeners.set(key, []);
    }
    this.eventListeners.get(key).push(listener);

    return () => {
      contract.off(eventName, listener);
      const listeners = this.eventListeners.get(key);
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    };
  }

  /**
   * Remove all event listeners
   */
  removeAllEventListeners() {
    for (const [key, listeners] of this.eventListeners) {
      const [contractName, eventName] = key.split('.');
      const contract = this.contracts[contractName];
      
      if (contract) {
        listeners.forEach(listener => {
          contract.off(eventName, listener);
        });
      }
    }
    
    this.eventListeners.clear();
  }

  // =================
  // Utility Methods
  // =================

  /**
   * Get transaction receipt
   */
  async getTransactionReceipt(txHash) {
    return await this.provider.getTransactionReceipt(txHash);
  }

  /**
   * Wait for transaction confirmation
   */
  async waitForTransaction(txHash, confirmations = 1) {
    return await this.provider.waitForTransaction(txHash, confirmations);
  }

  /**
   * Format token amount
   */
  formatTokenAmount(amount, decimals = 18) {
    return ethers.formatUnits(amount, decimals);
  }

  /**
   * Parse token amount
   */
  parseTokenAmount(amount, decimals = 18) {
    return ethers.parseUnits(amount.toString(), decimals);
  }

  /**
   * Get current block number
   */
  async getCurrentBlock() {
    return await this.provider.getBlockNumber();
  }

  /**
   * Get gas price
   */
  async getGasPrice() {
    const feeData = await this.provider.getFeeData();
    return feeData.gasPrice;
  }

  /**
   * Estimate gas for transaction
   */
  async estimateGas(contractName, methodName, params = []) {
    const contract = this.contracts[contractName];
    if (!contract) throw new Error(`${contractName} contract not loaded`);

    return await contract[methodName].estimateGas(...params);
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PRSMWeb3SDK;
} else if (typeof window !== 'undefined') {
  window.PRSMWeb3SDK = PRSMWeb3SDK;
}

module.exports = PRSMWeb3SDK;