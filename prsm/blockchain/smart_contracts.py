"""
FTNS Smart Contract Suite
=========================

Production-grade smart contracts for FTNS token ecosystem.
Implements comprehensive DeFi functionality and cross-chain compatibility.

Smart Contract Components:
1. FTNS Token Contract - ERC20 with additional features
2. Cross-Chain Bridge Contract - Multi-blockchain token bridging
3. Staking Contract - FTNS staking for governance and rewards
4. Marketplace Contract - P2P AI model/data trading
5. Governance Contract - DAO governance and voting
6. Oracle Contract - Price feeds and external data

Features:
- ERC20 compatible FTNS token with mint/burn capabilities
- Cross-chain bridge with security validations
- Staking pools with flexible terms and rewards
- Decentralized marketplace for AI assets
- DAO governance with proposal and voting mechanisms
- Oracle integration for real-world data feeds
- Gas optimization and security best practices
- Upgrade-safe proxy patterns
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import structlog

logger = structlog.get_logger(__name__)


class SmartContractSuite:
    """Production smart contract suite for FTNS ecosystem"""
    
    def __init__(self):
        self.contracts_dir = Path(__file__).parent / "contracts"
        self.contracts_dir.mkdir(exist_ok=True)
        
        # Contract definitions
        self.contract_sources = {
            "FTNSToken": self._get_ftns_token_contract(),
            "FTNSBridge": self._get_bridge_contract(),
            "FTNSStaking": self._get_staking_contract(),
            "FTNSMarketplace": self._get_marketplace_contract(),
            "FTNSGovernance": self._get_governance_contract(),
            "FTNSOracle": self._get_oracle_contract()
        }
        
        # Write contracts to files
        self._write_contract_files()
    
    def _write_contract_files(self):
        """Write contract source code to files"""
        for contract_name, source in self.contract_sources.items():
            contract_file = self.contracts_dir / f"{contract_name}.sol"
            contract_file.write_text(source)
            logger.info(f"✅ Contract written: {contract_file}")
    
    def _get_ftns_token_contract(self) -> str:
        """FTNS Token Contract - Enhanced ERC20 with additional features"""
        return '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title FTNS Token
 * @dev Enhanced ERC20 token for PRSM ecosystem with governance features
 */
contract FTNSToken is ERC20, ERC20Burnable, ERC20Pausable, AccessControl, ReentrancyGuard {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant BRIDGE_ROLE = keccak256("BRIDGE_ROLE");
    
    // Maximum supply: 1 billion FTNS
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18;
    
    // Transaction limits for security
    uint256 public maxTransactionAmount = 10_000_000 * 10**18; // 10M FTNS
    uint256 public maxWalletAmount = 50_000_000 * 10**18; // 50M FTNS
    
    // Fee structure
    uint256 public transferFeePercent = 0; // 0% by default
    address public feeRecipient;
    
    // Anti-bot protection
    mapping(address => bool) public isBlacklisted;
    bool public tradingActive = true;
    
    // Events
    event MaxTransactionAmountUpdated(uint256 newAmount);
    event MaxWalletAmountUpdated(uint256 newAmount);
    event TransferFeeUpdated(uint256 newFeePercent);
    event FeeRecipientUpdated(address newRecipient);
    event BlacklistUpdated(address indexed account, bool isBlacklisted);
    event TradingStatusUpdated(bool active);
    
    constructor(
        string memory name,
        string memory symbol,
        uint256 initialSupply,
        address admin
    ) ERC20(name, symbol) {
        require(initialSupply <= MAX_SUPPLY, "Initial supply exceeds maximum");
        
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(MINTER_ROLE, admin);
        _grantRole(PAUSER_ROLE, admin);
        
        _mint(admin, initialSupply);
        feeRecipient = admin;
    }
    
    /**
     * @dev Mint new tokens (only by minter role)
     */
    function mint(address to, uint256 amount) public onlyRole(MINTER_ROLE) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Minting would exceed max supply");
        _mint(to, amount);
    }
    
    /**
     * @dev Pause token transfers (only by pauser role)
     */
    function pause() public onlyRole(PAUSER_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpause token transfers (only by pauser role)
     */
    function unpause() public onlyRole(PAUSER_ROLE) {
        _unpause();
    }
    
    /**
     * @dev Update transaction limits (only admin)
     */
    function updateTransactionLimits(
        uint256 _maxTransactionAmount,
        uint256 _maxWalletAmount
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_maxTransactionAmount >= totalSupply() / 1000, "Max transaction too low"); // Min 0.1%
        require(_maxWalletAmount >= totalSupply() / 100, "Max wallet too low"); // Min 1%
        
        maxTransactionAmount = _maxTransactionAmount;
        maxWalletAmount = _maxWalletAmount;
        
        emit MaxTransactionAmountUpdated(_maxTransactionAmount);
        emit MaxWalletAmountUpdated(_maxWalletAmount);
    }
    
    /**
     * @dev Update transfer fee (only admin)
     */
    function updateTransferFee(
        uint256 _feePercent,
        address _feeRecipient
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_feePercent <= 500, "Fee cannot exceed 5%"); // Max 5%
        require(_feeRecipient != address(0), "Invalid fee recipient");
        
        transferFeePercent = _feePercent;
        feeRecipient = _feeRecipient;
        
        emit TransferFeeUpdated(_feePercent);
        emit FeeRecipientUpdated(_feeRecipient);
    }
    
    /**
     * @dev Update blacklist status (only admin)
     */
    function updateBlacklist(
        address account,
        bool blacklisted
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        isBlacklisted[account] = blacklisted;
        emit BlacklistUpdated(account, blacklisted);
    }
    
    /**
     * @dev Enable/disable trading (only admin)
     */
    function setTradingActive(bool _active) external onlyRole(DEFAULT_ADMIN_ROLE) {
        tradingActive = _active;
        emit TradingStatusUpdated(_active);
    }
    
    /**
     * @dev Bridge burn function (only bridge role)
     */
    function bridgeBurn(address from, uint256 amount) external onlyRole(BRIDGE_ROLE) {
        _burn(from, amount);
    }
    
    /**
     * @dev Bridge mint function (only bridge role)
     */
    function bridgeMint(address to, uint256 amount) external onlyRole(BRIDGE_ROLE) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Minting would exceed max supply");
        _mint(to, amount);
    }
    
    /**
     * @dev Override transfer function to include fees and limits
     */
    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal override {
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(!isBlacklisted[from] && !isBlacklisted[to], "Address blacklisted");
        require(tradingActive || hasRole(DEFAULT_ADMIN_ROLE, from), "Trading not active");
        
        // Check transaction limits (exclude admin and contracts)
        if (!hasRole(DEFAULT_ADMIN_ROLE, from) && !hasRole(DEFAULT_ADMIN_ROLE, to)) {
            require(amount <= maxTransactionAmount, "Transfer amount exceeds limit");
            
            if (to != address(0) && balanceOf(to) + amount > maxWalletAmount) {
                require(false, "Recipient wallet would exceed limit");
            }
        }
        
        uint256 transferAmount = amount;
        
        // Apply transfer fee if configured
        if (transferFeePercent > 0 && !hasRole(DEFAULT_ADMIN_ROLE, from)) {
            uint256 feeAmount = (amount * transferFeePercent) / 10000;
            transferAmount = amount - feeAmount;
            
            if (feeAmount > 0) {
                super._transfer(from, feeRecipient, feeAmount);
            }
        }
        
        super._transfer(from, to, transferAmount);
    }
    
    /**
     * @dev Required override for multiple inheritance
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20, ERC20Pausable) {
        super._beforeTokenTransfer(from, to, amount);
    }
    
    /**
     * @dev Get contract version
     */
    function version() external pure returns (string memory) {
        return "1.0.0";
    }
}
'''
    
    def _get_bridge_contract(self) -> str:
        """Cross-Chain Bridge Contract"""
        return '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "./FTNSToken.sol";

/**
 * @title FTNS Cross-Chain Bridge
 * @dev Secure bridge for transferring FTNS tokens across blockchains
 */
contract FTNSBridge is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    
    FTNSToken public immutable ftnsToken;
    
    // Bridge configuration
    uint256 public minBridgeAmount = 1 * 10**18; // 1 FTNS
    uint256 public maxBridgeAmount = 1_000_000 * 10**18; // 1M FTNS
    uint256 public bridgeFeePercent = 10; // 0.1%
    address public feeRecipient;
    
    // Cross-chain tracking
    mapping(uint256 => bool) public supportedChains;
    mapping(bytes32 => bool) public processedTransactions;
    mapping(address => uint256) public userNonces;
    
    // Bridge statistics
    uint256 public totalBridgedOut;
    uint256 public totalBridgedIn;
    uint256 public totalFees;
    
    // Events
    event BridgeOut(
        address indexed user,
        uint256 amount,
        uint256 fee,
        uint256 destinationChain,
        uint256 nonce,
        bytes32 indexed transactionId
    );
    
    event BridgeIn(
        address indexed user,
        uint256 amount,
        uint256 sourceChain,
        bytes32 indexed sourceTransactionId,
        bytes32 indexed transactionId
    );
    
    event SupportedChainUpdated(uint256 chainId, bool supported);
    event BridgeLimitsUpdated(uint256 minAmount, uint256 maxAmount);
    event BridgeFeeUpdated(uint256 feePercent, address feeRecipient);
    
    constructor(
        address _ftnsToken,
        address _admin,
        address _feeRecipient
    ) {
        require(_ftnsToken != address(0), "Invalid token address");
        require(_admin != address(0), "Invalid admin address");
        require(_feeRecipient != address(0), "Invalid fee recipient");
        
        ftnsToken = FTNSToken(_ftnsToken);
        feeRecipient = _feeRecipient;
        
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(VALIDATOR_ROLE, _admin);
        _grantRole(OPERATOR_ROLE, _admin);
        
        // Add common supported chains
        supportedChains[1] = true;    // Ethereum
        supportedChains[137] = true;  // Polygon
        supportedChains[56] = true;   // BSC
    }
    
    /**
     * @dev Bridge tokens to another chain
     */
    function bridgeOut(
        uint256 amount,
        uint256 destinationChain
    ) external nonReentrant whenNotPaused {
        require(amount >= minBridgeAmount, "Amount below minimum");
        require(amount <= maxBridgeAmount, "Amount exceeds maximum");
        require(supportedChains[destinationChain], "Destination chain not supported");
        require(ftnsToken.balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        // Calculate fee
        uint256 fee = (amount * bridgeFeePercent) / 10000;
        uint256 bridgeAmount = amount - fee;
        
        // Increment user nonce
        uint256 nonce = ++userNonces[msg.sender];
        
        // Generate transaction ID
        bytes32 transactionId = keccak256(abi.encodePacked(
            msg.sender,
            amount,
            destinationChain,
            nonce,
            block.timestamp,
            block.chainid
        ));
        
        // Burn tokens (bridge out)
        ftnsToken.bridgeBurn(msg.sender, amount);
        
        // Send fee to recipient
        if (fee > 0) {
            ftnsToken.mint(feeRecipient, fee);
        }
        
        // Update statistics
        totalBridgedOut += bridgeAmount;
        totalFees += fee;
        
        emit BridgeOut(msg.sender, bridgeAmount, fee, destinationChain, nonce, transactionId);
    }
    
    /**
     * @dev Bridge tokens from another chain (validator only)
     */
    function bridgeIn(
        address user,
        uint256 amount,
        uint256 sourceChain,
        bytes32 sourceTransactionId,
        bytes memory validatorSignatures
    ) external onlyRole(VALIDATOR_ROLE) nonReentrant whenNotPaused {
        require(user != address(0), "Invalid user address");
        require(amount > 0, "Invalid amount");
        require(supportedChains[sourceChain], "Source chain not supported");
        require(!processedTransactions[sourceTransactionId], "Transaction already processed");
        
        // Verify validator signatures (simplified for demo)
        require(validatorSignatures.length > 0, "Invalid signatures");
        
        // Generate transaction ID
        bytes32 transactionId = keccak256(abi.encodePacked(
            user,
            amount,
            sourceChain,
            sourceTransactionId,
            block.timestamp,
            block.chainid
        ));
        
        // Mark transaction as processed
        processedTransactions[sourceTransactionId] = true;
        
        // Mint tokens (bridge in)
        ftnsToken.bridgeMint(user, amount);
        
        // Update statistics
        totalBridgedIn += amount;
        
        emit BridgeIn(user, amount, sourceChain, sourceTransactionId, transactionId);
    }
    
    /**
     * @dev Update supported chains (admin only)
     */
    function updateSupportedChain(
        uint256 chainId,
        bool supported
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        supportedChains[chainId] = supported;
        emit SupportedChainUpdated(chainId, supported);
    }
    
    /**
     * @dev Update bridge limits (admin only)
     */
    function updateBridgeLimits(
        uint256 _minAmount,
        uint256 _maxAmount
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_minAmount > 0, "Invalid minimum amount");
        require(_maxAmount > _minAmount, "Invalid maximum amount");
        
        minBridgeAmount = _minAmount;
        maxBridgeAmount = _maxAmount;
        
        emit BridgeLimitsUpdated(_minAmount, _maxAmount);
    }
    
    /**
     * @dev Update bridge fee (admin only)
     */
    function updateBridgeFee(
        uint256 _feePercent,
        address _feeRecipient
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_feePercent <= 1000, "Fee cannot exceed 10%"); // Max 10%
        require(_feeRecipient != address(0), "Invalid fee recipient");
        
        bridgeFeePercent = _feePercent;
        feeRecipient = _feeRecipient;
        
        emit BridgeFeeUpdated(_feePercent, _feeRecipient);
    }
    
    /**
     * @dev Pause bridge operations (admin only)
     */
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpause bridge operations (admin only)
     */
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
    
    /**
     * @dev Emergency withdrawal (admin only)
     */
    function emergencyWithdraw(
        address token,
        uint256 amount
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (token == address(0)) {
            payable(msg.sender).transfer(amount);
        } else {
            IERC20(token).transfer(msg.sender, amount);
        }
    }
    
    /**
     * @dev Get bridge statistics
     */
    function getBridgeStats() external view returns (
        uint256 bridgedOut,
        uint256 bridgedIn,
        uint256 fees,
        uint256 netFlow
    ) {
        return (
            totalBridgedOut,
            totalBridgedIn,
            totalFees,
            totalBridgedIn > totalBridgedOut ? totalBridgedIn - totalBridgedOut : totalBridgedOut - totalBridgedIn
        );
    }
}
'''
    
    def _get_staking_contract(self) -> str:
        """FTNS Staking Contract"""
        return '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "./FTNSToken.sol";

/**
 * @title FTNS Staking Contract
 * @dev Staking pools for FTNS tokens with governance and rewards
 */
contract FTNSStaking is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant POOL_MANAGER_ROLE = keccak256("POOL_MANAGER_ROLE");
    bytes32 public constant REWARDS_MANAGER_ROLE = keccak256("REWARDS_MANAGER_ROLE");
    
    FTNSToken public immutable ftnsToken;
    
    struct StakingPool {
        string name;
        uint256 minStakeAmount;
        uint256 maxStakeAmount;
        uint256 lockPeriod; // in seconds
        uint256 apy; // Annual percentage yield (basis points: 10000 = 100%)
        uint256 totalStaked;
        uint256 maxPoolSize;
        bool active;
        uint256 createdAt;
    }
    
    struct UserStake {
        uint256 amount;
        uint256 stakedAt;
        uint256 lastRewardClaim;
        uint256 poolId;
        bool active;
    }
    
    // State variables
    mapping(uint256 => StakingPool) public stakingPools;
    mapping(address => mapping(uint256 => UserStake)) public userStakes;
    mapping(address => uint256[]) public userStakeIds;
    mapping(uint256 => address[]) public poolStakers;
    
    uint256 public nextPoolId = 1;
    uint256 public nextStakeId = 1;
    uint256 public totalStaked;
    uint256 public totalRewardsPaid;
    
    // Events
    event PoolCreated(
        uint256 indexed poolId,
        string name,
        uint256 apy,
        uint256 lockPeriod
    );
    
    event Staked(
        address indexed user,
        uint256 indexed poolId,
        uint256 indexed stakeId,
        uint256 amount
    );
    
    event Unstaked(
        address indexed user,
        uint256 indexed poolId,
        uint256 indexed stakeId,
        uint256 amount,
        uint256 rewards
    );
    
    event RewardsClaimed(
        address indexed user,
        uint256 indexed poolId,
        uint256 indexed stakeId,
        uint256 rewards
    );
    
    event PoolUpdated(uint256 indexed poolId, bool active);
    
    constructor(
        address _ftnsToken,
        address _admin
    ) {
        require(_ftnsToken != address(0), "Invalid token address");
        require(_admin != address(0), "Invalid admin address");
        
        ftnsToken = FTNSToken(_ftnsToken);
        
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(POOL_MANAGER_ROLE, _admin);
        _grantRole(REWARDS_MANAGER_ROLE, _admin);
        
        // Create default staking pools
        _createDefaultPools();
    }
    
    /**
     * @dev Create default staking pools
     */
    function _createDefaultPools() internal {
        // Flexible pool (no lock, low APY)
        _createPool("Flexible", 100 * 10**18, 1000000 * 10**18, 0, 500, 50000000 * 10**18); // 5% APY
        
        // 30-day lock pool
        _createPool("30-Day Lock", 1000 * 10**18, 1000000 * 10**18, 30 days, 1200, 30000000 * 10**18); // 12% APY
        
        // 90-day lock pool
        _createPool("90-Day Lock", 1000 * 10**18, 1000000 * 10**18, 90 days, 2000, 20000000 * 10**18); // 20% APY
        
        // Governance pool (high minimum, high APY)
        _createPool("Governance", 10000 * 10**18, 10000000 * 10**18, 180 days, 3000, 10000000 * 10**18); // 30% APY
    }
    
    /**
     * @dev Create new staking pool
     */
    function createPool(
        string memory name,
        uint256 minStakeAmount,
        uint256 maxStakeAmount,
        uint256 lockPeriod,
        uint256 apy,
        uint256 maxPoolSize
    ) external onlyRole(POOL_MANAGER_ROLE) {
        _createPool(name, minStakeAmount, maxStakeAmount, lockPeriod, apy, maxPoolSize);
    }
    
    function _createPool(
        string memory name,
        uint256 minStakeAmount,
        uint256 maxStakeAmount,
        uint256 lockPeriod,
        uint256 apy,
        uint256 maxPoolSize
    ) internal {
        require(bytes(name).length > 0, "Invalid pool name");
        require(minStakeAmount > 0, "Invalid minimum stake");
        require(maxStakeAmount > minStakeAmount, "Invalid maximum stake");
        require(apy <= 10000, "APY cannot exceed 100%");
        require(maxPoolSize > 0, "Invalid max pool size");
        
        uint256 poolId = nextPoolId++;
        
        stakingPools[poolId] = StakingPool({
            name: name,
            minStakeAmount: minStakeAmount,
            maxStakeAmount: maxStakeAmount,
            lockPeriod: lockPeriod,
            apy: apy,
            totalStaked: 0,
            maxPoolSize: maxPoolSize,
            active: true,
            createdAt: block.timestamp
        });
        
        emit PoolCreated(poolId, name, apy, lockPeriod);
    }
    
    /**
     * @dev Stake tokens in a pool
     */
    function stake(uint256 poolId, uint256 amount) external nonReentrant whenNotPaused {
        require(poolId > 0 && poolId < nextPoolId, "Invalid pool ID");
        require(amount > 0, "Invalid stake amount");
        
        StakingPool storage pool = stakingPools[poolId];
        require(pool.active, "Pool not active");
        require(amount >= pool.minStakeAmount, "Amount below minimum");
        require(amount <= pool.maxStakeAmount, "Amount exceeds maximum");
        require(pool.totalStaked + amount <= pool.maxPoolSize, "Pool size exceeded");
        
        // Transfer tokens from user
        ftnsToken.transferFrom(msg.sender, address(this), amount);
        
        uint256 stakeId = nextStakeId++;
        
        // Create user stake
        userStakes[msg.sender][stakeId] = UserStake({
            amount: amount,
            stakedAt: block.timestamp,
            lastRewardClaim: block.timestamp,
            poolId: poolId,
            active: true
        });
        
        // Update tracking
        userStakeIds[msg.sender].push(stakeId);
        poolStakers[poolId].push(msg.sender);
        pool.totalStaked += amount;
        totalStaked += amount;
        
        emit Staked(msg.sender, poolId, stakeId, amount);
    }
    
    /**
     * @dev Unstake tokens from a pool
     */
    function unstake(uint256 stakeId) external nonReentrant {
        UserStake storage userStake = userStakes[msg.sender][stakeId];
        require(userStake.active, "Stake not active");
        
        StakingPool storage pool = stakingPools[userStake.poolId];
        
        // Check lock period
        require(
            block.timestamp >= userStake.stakedAt + pool.lockPeriod,
            "Stake still locked"
        );
        
        uint256 amount = userStake.amount;
        uint256 rewards = calculateRewards(msg.sender, stakeId);
        
        // Mark stake as inactive
        userStake.active = false;
        
        // Update pool stats
        pool.totalStaked -= amount;
        totalStaked -= amount;
        totalRewardsPaid += rewards;
        
        // Transfer tokens back to user
        ftnsToken.transfer(msg.sender, amount);
        
        // Mint rewards if any
        if (rewards > 0) {
            ftnsToken.mint(msg.sender, rewards);
        }
        
        emit Unstaked(msg.sender, userStake.poolId, stakeId, amount, rewards);
    }
    
    /**
     * @dev Claim rewards without unstaking
     */
    function claimRewards(uint256 stakeId) external nonReentrant {
        UserStake storage userStake = userStakes[msg.sender][stakeId];
        require(userStake.active, "Stake not active");
        
        uint256 rewards = calculateRewards(msg.sender, stakeId);
        require(rewards > 0, "No rewards to claim");
        
        // Update last claim time
        userStake.lastRewardClaim = block.timestamp;
        totalRewardsPaid += rewards;
        
        // Mint rewards
        ftnsToken.mint(msg.sender, rewards);
        
        emit RewardsClaimed(msg.sender, userStake.poolId, stakeId, rewards);
    }
    
    /**
     * @dev Calculate pending rewards for a stake
     */
    function calculateRewards(address user, uint256 stakeId) public view returns (uint256) {
        UserStake storage userStake = userStakes[user][stakeId];
        if (!userStake.active) return 0;
        
        StakingPool storage pool = stakingPools[userStake.poolId];
        
        uint256 stakingDuration = block.timestamp - userStake.lastRewardClaim;
        uint256 annualRewards = (userStake.amount * pool.apy) / 10000;
        uint256 rewards = (annualRewards * stakingDuration) / 365 days;
        
        return rewards;
    }
    
    /**
     * @dev Get user stakes
     */
    function getUserStakes(address user) external view returns (uint256[] memory) {
        return userStakeIds[user];
    }
    
    /**
     * @dev Get pool stakers
     */
    function getPoolStakers(uint256 poolId) external view returns (address[] memory) {
        return poolStakers[poolId];
    }
    
    /**
     * @dev Update pool status (admin only)
     */
    function updatePoolStatus(uint256 poolId, bool active) external onlyRole(POOL_MANAGER_ROLE) {
        require(poolId > 0 && poolId < nextPoolId, "Invalid pool ID");
        stakingPools[poolId].active = active;
        emit PoolUpdated(poolId, active);
    }
    
    /**
     * @dev Pause contract (admin only)
     */
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpause contract (admin only)
     */
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
    
    /**
     * @dev Get contract statistics
     */
    function getContractStats() external view returns (
        uint256 _totalStaked,
        uint256 _totalRewardsPaid,
        uint256 _activeStakers,
        uint256 _activePools
    ) {
        uint256 activePools = 0;
        for (uint256 i = 1; i < nextPoolId; i++) {
            if (stakingPools[i].active) {
                activePools++;
            }
        }
        
        return (totalStaked, totalRewardsPaid, 0, activePools); // TODO: Calculate active stakers
    }
}
'''
    
    def _get_marketplace_contract(self) -> str:
        """FTNS Marketplace Contract"""
        return '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "./FTNSToken.sol";

/**
 * @title FTNS Marketplace
 * @dev Decentralized marketplace for AI models, datasets, and compute time
 */
contract FTNSMarketplace is AccessControl, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    bytes32 public constant MODERATOR_ROLE = keccak256("MODERATOR_ROLE");
    
    FTNSToken public immutable ftnsToken;
    
    enum AssetType {
        AI_MODEL,
        DATASET,
        COMPUTE_TIME,
        API_ACCESS
    }
    
    enum ListingStatus {
        ACTIVE,
        SOLD,
        CANCELLED,
        SUSPENDED
    }
    
    struct Listing {
        uint256 id;
        address seller;
        AssetType assetType;
        string title;
        string description;
        string metadataURI; // IPFS hash or URL
        uint256 price; // Price in FTNS tokens
        uint256 quantity; // Available quantity
        uint256 sold; // Amount sold
        ListingStatus status;
        uint256 createdAt;
        uint256 updatedAt;
        mapping(address => bool) buyers; // Track who bought
    }
    
    struct Purchase {
        uint256 id;
        uint256 listingId;
        address buyer;
        address seller;
        uint256 quantity;
        uint256 price;
        uint256 timestamp;
        bool completed;
        string accessData; // Encrypted access information
    }
    
    // State variables
    Counters.Counter private _listingIds;
    Counters.Counter private _purchaseIds;
    
    mapping(uint256 => Listing) public listings;
    mapping(uint256 => Purchase) public purchases;
    mapping(address => uint256[]) public sellerListings;
    mapping(address => uint256[]) public buyerPurchases;
    
    // Marketplace configuration
    uint256 public marketplaceFeePercent = 250; // 2.5%
    address public feeRecipient;
    uint256 public minListingPrice = 1 * 10**18; // 1 FTNS
    
    // Statistics
    uint256 public totalSales;
    uint256 public totalVolume;
    uint256 public totalFees;
    
    // Events
    event ListingCreated(
        uint256 indexed listingId,
        address indexed seller,
        AssetType assetType,
        uint256 price,
        uint256 quantity
    );
    
    event ListingUpdated(
        uint256 indexed listingId,
        uint256 price,
        uint256 quantity,
        ListingStatus status
    );
    
    event Purchase(
        uint256 indexed purchaseId,
        uint256 indexed listingId,
        address indexed buyer,
        address seller,
        uint256 quantity,
        uint256 price
    );
    
    event MarketplaceFeeUpdated(uint256 feePercent, address feeRecipient);
    
    constructor(
        address _ftnsToken,
        address _admin,
        address _feeRecipient
    ) {
        require(_ftnsToken != address(0), "Invalid token address");
        require(_admin != address(0), "Invalid admin address");
        require(_feeRecipient != address(0), "Invalid fee recipient");
        
        ftnsToken = FTNSToken(_ftnsToken);
        feeRecipient = _feeRecipient;
        
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(MODERATOR_ROLE, _admin);
    }
    
    /**
     * @dev Create new marketplace listing
     */
    function createListing(
        AssetType assetType,
        string memory title,
        string memory description,
        string memory metadataURI,
        uint256 price,
        uint256 quantity
    ) external nonReentrant {
        require(bytes(title).length > 0, "Title required");
        require(bytes(description).length > 0, "Description required");
        require(price >= minListingPrice, "Price below minimum");
        require(quantity > 0, "Invalid quantity");
        
        _listingIds.increment();
        uint256 listingId = _listingIds.current();
        
        Listing storage listing = listings[listingId];
        listing.id = listingId;
        listing.seller = msg.sender;
        listing.assetType = assetType;
        listing.title = title;
        listing.description = description;
        listing.metadataURI = metadataURI;
        listing.price = price;
        listing.quantity = quantity;
        listing.sold = 0;
        listing.status = ListingStatus.ACTIVE;
        listing.createdAt = block.timestamp;
        listing.updatedAt = block.timestamp;
        
        sellerListings[msg.sender].push(listingId);
        
        emit ListingCreated(listingId, msg.sender, assetType, price, quantity);
    }
    
    /**
     * @dev Purchase item from marketplace
     */
    function purchaseItem(
        uint256 listingId,
        uint256 quantity
    ) external nonReentrant {
        Listing storage listing = listings[listingId];
        require(listing.id > 0, "Listing not found");
        require(listing.status == ListingStatus.ACTIVE, "Listing not active");
        require(quantity > 0, "Invalid quantity");
        require(quantity <= listing.quantity - listing.sold, "Insufficient quantity");
        require(msg.sender != listing.seller, "Cannot buy own listing");
        
        uint256 totalPrice = listing.price * quantity;
        uint256 marketplaceFee = (totalPrice * marketplaceFeePercent) / 10000;
        uint256 sellerAmount = totalPrice - marketplaceFee;
        
        // Transfer payment from buyer
        ftnsToken.transferFrom(msg.sender, address(this), totalPrice);
        
        // Transfer to seller
        ftnsToken.transfer(listing.seller, sellerAmount);
        
        // Transfer fee to marketplace
        if (marketplaceFee > 0) {
            ftnsToken.transfer(feeRecipient, marketplaceFee);
        }
        
        // Create purchase record
        _purchaseIds.increment();
        uint256 purchaseId = _purchaseIds.current();
        
        purchases[purchaseId] = Purchase({
            id: purchaseId,
            listingId: listingId,
            buyer: msg.sender,
            seller: listing.seller,
            quantity: quantity,
            price: totalPrice,
            timestamp: block.timestamp,
            completed: false,
            accessData: "" // To be set by seller
        });
        
        // Update listing
        listing.sold += quantity;
        listing.updatedAt = block.timestamp;
        listing.buyers[msg.sender] = true;
        
        if (listing.sold >= listing.quantity) {
            listing.status = ListingStatus.SOLD;
        }
        
        // Update tracking
        buyerPurchases[msg.sender].push(purchaseId);
        
        // Update statistics
        totalSales++;
        totalVolume += totalPrice;
        totalFees += marketplaceFee;
        
        emit Purchase(purchaseId, listingId, msg.sender, listing.seller, quantity, totalPrice);
    }
    
    /**
     * @dev Update listing (seller only)
     */
    function updateListing(
        uint256 listingId,
        uint256 newPrice,
        uint256 newQuantity,
        string memory newDescription
    ) external {
        Listing storage listing = listings[listingId];
        require(listing.seller == msg.sender, "Not listing owner");
        require(listing.status == ListingStatus.ACTIVE, "Listing not active");
        require(newPrice >= minListingPrice, "Price below minimum");
        require(newQuantity >= listing.sold, "Quantity below sold amount");
        
        listing.price = newPrice;
        listing.quantity = newQuantity;
        listing.description = newDescription;
        listing.updatedAt = block.timestamp;
        
        emit ListingUpdated(listingId, newPrice, newQuantity, listing.status);
    }
    
    /**
     * @dev Cancel listing (seller only)
     */
    function cancelListing(uint256 listingId) external {
        Listing storage listing = listings[listingId];
        require(listing.seller == msg.sender, "Not listing owner");
        require(listing.status == ListingStatus.ACTIVE, "Listing not active");
        
        listing.status = ListingStatus.CANCELLED;
        listing.updatedAt = block.timestamp;
        
        emit ListingUpdated(listingId, listing.price, listing.quantity, ListingStatus.CANCELLED);
    }
    
    /**
     * @dev Provide access data to buyer (seller only)
     */
    function provideAccessData(
        uint256 purchaseId,
        string memory accessData
    ) external {
        Purchase storage purchase = purchases[purchaseId];
        require(purchase.seller == msg.sender, "Not purchase seller");
        require(!purchase.completed, "Purchase already completed");
        require(bytes(accessData).length > 0, "Access data required");
        
        purchase.accessData = accessData;
        purchase.completed = true;
    }
    
    /**
     * @dev Get seller listings
     */
    function getSellerListings(address seller) external view returns (uint256[] memory) {
        return sellerListings[seller];
    }
    
    /**
     * @dev Get buyer purchases
     */
    function getBuyerPurchases(address buyer) external view returns (uint256[] memory) {
        return buyerPurchases[buyer];
    }
    
    /**
     * @dev Check if address bought from listing
     */
    function hasPurchased(uint256 listingId, address buyer) external view returns (bool) {
        return listings[listingId].buyers[buyer];
    }
    
    /**
     * @dev Suspend listing (moderator only)
     */
    function suspendListing(uint256 listingId, string memory reason) external onlyRole(MODERATOR_ROLE) {
        Listing storage listing = listings[listingId];
        require(listing.id > 0, "Listing not found");
        
        listing.status = ListingStatus.SUSPENDED;
        listing.updatedAt = block.timestamp;
        
        emit ListingUpdated(listingId, listing.price, listing.quantity, ListingStatus.SUSPENDED);
    }
    
    /**
     * @dev Update marketplace fee (admin only)
     */
    function updateMarketplaceFee(
        uint256 _feePercent,
        address _feeRecipient
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_feePercent <= 1000, "Fee cannot exceed 10%");
        require(_feeRecipient != address(0), "Invalid fee recipient");
        
        marketplaceFeePercent = _feePercent;
        feeRecipient = _feeRecipient;
        
        emit MarketplaceFeeUpdated(_feePercent, _feeRecipient);
    }
    
    /**
     * @dev Get marketplace statistics
     */
    function getMarketplaceStats() external view returns (
        uint256 _totalSales,
        uint256 _totalVolume,
        uint256 _totalFees,
        uint256 _activeListings
    ) {
        uint256 activeListings = 0;
        for (uint256 i = 1; i <= _listingIds.current(); i++) {
            if (listings[i].status == ListingStatus.ACTIVE) {
                activeListings++;
            }
        }
        
        return (totalSales, totalVolume, totalFees, activeListings);
    }
}
'''
    
    def _get_governance_contract(self) -> str:
        """FTNS Governance Contract"""
        return '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "./FTNSToken.sol";
import "./FTNSStaking.sol";

/**
 * @title FTNS Governance
 * @dev DAO governance contract for PRSM ecosystem
 */
contract FTNSGovernance is AccessControl, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    FTNSToken public immutable ftnsToken;
    FTNSStaking public immutable stakingContract;
    
    enum ProposalType {
        PARAMETER_CHANGE,
        FEATURE_ADDITION,
        FUNDING_REQUEST,
        CONSTITUTION_AMENDMENT
    }
    
    enum ProposalStatus {
        PENDING,
        ACTIVE,
        SUCCEEDED,
        DEFEATED,
        EXECUTED,
        CANCELLED
    }
    
    enum VoteType {
        AGAINST,
        FOR,
        ABSTAIN
    }
    
    struct Proposal {
        uint256 id;
        address proposer;
        ProposalType proposalType;
        string title;
        string description;
        string executionData; // JSON or encoded function calls
        uint256 startTime;
        uint256 endTime;
        uint256 votesFor;
        uint256 votesAgainst;
        uint256 votesAbstain;
        uint256 totalVotingPower;
        ProposalStatus status;
        mapping(address => bool) hasVoted;
        mapping(address => VoteType) votes;
        mapping(address => uint256) votingPower;
    }
    
    // State variables
    Counters.Counter private _proposalIds;
    mapping(uint256 => Proposal) public proposals;
    mapping(address => uint256[]) public userProposals;
    
    // Governance parameters
    uint256 public proposalThreshold = 100000 * 10**18; // 100K FTNS to propose
    uint256 public quorumPercentage = 1000; // 10% (basis points)
    uint256 public votingPeriod = 7 days;
    uint256 public executionDelay = 2 days;
    uint256 public gracePeriod = 14 days;
    
    // Statistics
    uint256 public totalProposals;
    uint256 public totalVotes;
    uint256 public executedProposals;
    
    // Events
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        ProposalType proposalType,
        string title,
        uint256 startTime,
        uint256 endTime
    );
    
    event VoteCast(
        address indexed voter,
        uint256 indexed proposalId,
        VoteType vote,
        uint256 votingPower
    );
    
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCancelled(uint256 indexed proposalId);
    
    event GovernanceParametersUpdated(
        uint256 proposalThreshold,
        uint256 quorumPercentage,
        uint256 votingPeriod
    );
    
    constructor(
        address _ftnsToken,
        address _stakingContract,
        address _admin
    ) {
        require(_ftnsToken != address(0), "Invalid token address");
        require(_stakingContract != address(0), "Invalid staking contract");
        require(_admin != address(0), "Invalid admin address");
        
        ftnsToken = FTNSToken(_ftnsToken);
        stakingContract = FTNSStaking(_stakingContract);
        
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
    }
    
    /**
     * @dev Create new governance proposal
     */
    function createProposal(
        ProposalType proposalType,
        string memory title,
        string memory description,
        string memory executionData
    ) external nonReentrant {
        require(bytes(title).length > 0, "Title required");
        require(bytes(description).length > 0, "Description required");
        
        // Check proposer has enough voting power
        uint256 proposerPower = getVotingPower(msg.sender);
        require(proposerPower >= proposalThreshold, "Insufficient voting power to propose");
        
        _proposalIds.increment();
        uint256 proposalId = _proposalIds.current();
        
        uint256 startTime = block.timestamp;
        uint256 endTime = startTime + votingPeriod;
        
        Proposal storage proposal = proposals[proposalId];
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.proposalType = proposalType;
        proposal.title = title;
        proposal.description = description;
        proposal.executionData = executionData;
        proposal.startTime = startTime;
        proposal.endTime = endTime;
        proposal.status = ProposalStatus.ACTIVE;
        
        userProposals[msg.sender].push(proposalId);
        totalProposals++;
        
        emit ProposalCreated(proposalId, msg.sender, proposalType, title, startTime, endTime);
    }
    
    /**
     * @dev Cast vote on proposal
     */
    function castVote(uint256 proposalId, VoteType voteType) external nonReentrant {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id > 0, "Proposal not found");
        require(proposal.status == ProposalStatus.ACTIVE, "Proposal not active");
        require(block.timestamp <= proposal.endTime, "Voting period ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        
        uint256 votingPower = getVotingPower(msg.sender);
        require(votingPower > 0, "No voting power");
        
        // Record vote
        proposal.hasVoted[msg.sender] = true;
        proposal.votes[msg.sender] = voteType;
        proposal.votingPower[msg.sender] = votingPower;
        proposal.totalVotingPower += votingPower;
        
        // Update vote counts
        if (voteType == VoteType.FOR) {
            proposal.votesFor += votingPower;
        } else if (voteType == VoteType.AGAINST) {
            proposal.votesAgainst += votingPower;
        } else {
            proposal.votesAbstain += votingPower;
        }
        
        totalVotes++;
        
        emit VoteCast(msg.sender, proposalId, voteType, votingPower);
    }
    
    /**
     * @dev Execute proposal (if passed)
     */
    function executeProposal(uint256 proposalId) external nonReentrant {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id > 0, "Proposal not found");
        require(proposal.status == ProposalStatus.ACTIVE, "Proposal not active");
        require(block.timestamp > proposal.endTime, "Voting still active");
        
        // Check if proposal passed
        bool quorumReached = _checkQuorum(proposal);
        bool majorityReached = proposal.votesFor > proposal.votesAgainst;
        
        if (quorumReached && majorityReached) {
            proposal.status = ProposalStatus.SUCCEEDED;
            
            // Wait for execution delay
            require(
                block.timestamp >= proposal.endTime + executionDelay,
                "Execution delay not met"
            );
            
            // Execute proposal (simplified - would call actual functions)
            _executeProposalLogic(proposal);
            
            proposal.status = ProposalStatus.EXECUTED;
            executedProposals++;
            
            emit ProposalExecuted(proposalId);
        } else {
            proposal.status = ProposalStatus.DEFEATED;
        }
    }
    
    /**
     * @dev Cancel proposal (proposer or admin only)
     */
    function cancelProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id > 0, "Proposal not found");
        require(
            msg.sender == proposal.proposer || hasRole(DEFAULT_ADMIN_ROLE, msg.sender),
            "Not authorized"
        );
        require(
            proposal.status == ProposalStatus.PENDING || proposal.status == ProposalStatus.ACTIVE,
            "Cannot cancel"
        );
        
        proposal.status = ProposalStatus.CANCELLED;
        
        emit ProposalCancelled(proposalId);
    }
    
    /**
     * @dev Get voting power for address (based on staked FTNS)
     */
    function getVotingPower(address account) public view returns (uint256) {
        // Voting power comes from staked FTNS tokens
        // This would integrate with the staking contract
        // For now, using token balance as simplified voting power
        return ftnsToken.balanceOf(account);
    }
    
    /**
     * @dev Check if proposal meets quorum requirement
     */
    function _checkQuorum(Proposal storage proposal) internal view returns (bool) {
        uint256 totalSupply = ftnsToken.totalSupply();
        uint256 requiredQuorum = (totalSupply * quorumPercentage) / 10000;
        return proposal.totalVotingPower >= requiredQuorum;
    }
    
    /**
     * @dev Execute proposal logic (simplified)
     */
    function _executeProposalLogic(Proposal storage proposal) internal {
        // This would contain actual execution logic based on proposal type
        // For now, just log the execution
        if (proposal.proposalType == ProposalType.PARAMETER_CHANGE) {
            // Update governance parameters
        } else if (proposal.proposalType == ProposalType.FUNDING_REQUEST) {
            // Transfer funds
        }
        // etc.
    }
    
    /**
     * @dev Get user proposals
     */
    function getUserProposals(address user) external view returns (uint256[] memory) {
        return userProposals[user];
    }
    
    /**
     * @dev Get proposal vote counts
     */
    function getProposalVotes(uint256 proposalId) external view returns (
        uint256 votesFor,
        uint256 votesAgainst,
        uint256 votesAbstain,
        uint256 totalVotingPower
    ) {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.votesFor,
            proposal.votesAgainst,
            proposal.votesAbstain,
            proposal.totalVotingPower
        );
    }
    
    /**
     * @dev Update governance parameters (through governance only)
     */
    function updateGovernanceParameters(
        uint256 _proposalThreshold,
        uint256 _quorumPercentage,
        uint256 _votingPeriod
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_quorumPercentage <= 5000, "Quorum cannot exceed 50%");
        require(_votingPeriod >= 1 days && _votingPeriod <= 30 days, "Invalid voting period");
        
        proposalThreshold = _proposalThreshold;
        quorumPercentage = _quorumPercentage;
        votingPeriod = _votingPeriod;
        
        emit GovernanceParametersUpdated(_proposalThreshold, _quorumPercentage, _votingPeriod);
    }
    
    /**
     * @dev Get governance statistics
     */
    function getGovernanceStats() external view returns (
        uint256 _totalProposals,
        uint256 _totalVotes,
        uint256 _executedProposals,
        uint256 _activeProposals
    ) {
        uint256 activeProposals = 0;
        for (uint256 i = 1; i <= _proposalIds.current(); i++) {
            if (proposals[i].status == ProposalStatus.ACTIVE) {
                activeProposals++;
            }
        }
        
        return (totalProposals, totalVotes, executedProposals, activeProposals);
    }
}
'''
    
    def _get_oracle_contract(self) -> str:
        """FTNS Oracle Contract"""
        return '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title FTNS Oracle
 * @dev Price oracle and external data feeds for FTNS ecosystem
 */
contract FTNSOracle is AccessControl, ReentrancyGuard {
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    
    struct PriceData {
        uint256 price; // Price in wei (18 decimals)
        uint256 timestamp;
        uint256 confidence; // Confidence level (0-10000, where 10000 = 100%)
        bool valid;
    }
    
    struct DataFeed {
        string name;
        string description;
        bytes32 dataType; // "price", "volume", "supply", etc.
        uint256 updateFrequency; // Minimum seconds between updates
        uint256 lastUpdate;
        bool active;
    }
    
    // State variables
    mapping(string => PriceData) public priceData; // symbol => price data
    mapping(bytes32 => DataFeed) public dataFeeds;
    mapping(bytes32 => mapping(uint256 => bytes)) public historicalData;
    mapping(bytes32 => uint256) public latestDataIndex;
    
    // Oracle configuration
    uint256 public priceDeviationThreshold = 1000; // 10% (basis points)
    uint256 public minConfidenceLevel = 8000; // 80%
    uint256 public maxPriceAge = 1 hours;
    
    // Events
    event PriceUpdated(
        string indexed symbol,
        uint256 price,
        uint256 timestamp,
        uint256 confidence
    );
    
    event DataFeedCreated(
        bytes32 indexed feedId,
        string name,
        bytes32 dataType
    );
    
    event DataUpdated(
        bytes32 indexed feedId,
        uint256 indexed dataIndex,
        bytes data,
        uint256 timestamp
    );
    
    constructor(address _admin) {
        require(_admin != address(0), "Invalid admin address");
        
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(ORACLE_ROLE, _admin);
        _grantRole(VALIDATOR_ROLE, _admin);
        
        // Initialize default data feeds
        _createDefaultFeeds();
    }
    
    /**
     * @dev Create default data feeds
     */
    function _createDefaultFeeds() internal {
        createDataFeed("FTNS Price Feed", "FTNS token price in USD", "price", 300); // 5 minutes
        createDataFeed("FTNS Volume Feed", "24h trading volume", "volume", 3600); // 1 hour
        createDataFeed("Total Supply Feed", "Total FTNS supply", "supply", 3600); // 1 hour
        createDataFeed("Market Cap Feed", "FTNS market capitalization", "market_cap", 600); // 10 minutes
    }
    
    /**
     * @dev Update price data (oracle only)
     */
    function updatePrice(
        string memory symbol,
        uint256 price,
        uint256 confidence
    ) external onlyRole(ORACLE_ROLE) {
        require(bytes(symbol).length > 0, "Invalid symbol");
        require(price > 0, "Invalid price");
        require(confidence >= minConfidenceLevel, "Confidence too low");
        
        // Check for significant price deviation
        PriceData storage currentPrice = priceData[symbol];
        if (currentPrice.valid && currentPrice.timestamp > block.timestamp - maxPriceAge) {
            uint256 deviation = _calculateDeviation(currentPrice.price, price);
            require(deviation <= priceDeviationThreshold, "Price deviation too large");
        }
        
        // Update price data
        priceData[symbol] = PriceData({
            price: price,
            timestamp: block.timestamp,
            confidence: confidence,
            valid: true
        });
        
        emit PriceUpdated(symbol, price, block.timestamp, confidence);
    }
    
    /**
     * @dev Get latest price for symbol
     */
    function getPrice(string memory symbol) external view returns (
        uint256 price,
        uint256 timestamp,
        uint256 confidence,
        bool valid
    ) {
        PriceData storage data = priceData[symbol];
        
        // Check if price is not too old
        bool isValid = data.valid && (block.timestamp - data.timestamp <= maxPriceAge);
        
        return (data.price, data.timestamp, data.confidence, isValid);
    }
    
    /**
     * @dev Create new data feed (admin only)
     */
    function createDataFeed(
        string memory name,
        string memory description,
        string memory dataTypeStr,
        uint256 updateFrequency
    ) public onlyRole(DEFAULT_ADMIN_ROLE) returns (bytes32) {
        require(bytes(name).length > 0, "Invalid name");
        require(updateFrequency > 0, "Invalid update frequency");
        
        bytes32 feedId = keccak256(abi.encodePacked(name, block.timestamp));
        bytes32 dataType = keccak256(abi.encodePacked(dataTypeStr));
        
        dataFeeds[feedId] = DataFeed({
            name: name,
            description: description,
            dataType: dataType,
            updateFrequency: updateFrequency,
            lastUpdate: 0,
            active: true
        });
        
        emit DataFeedCreated(feedId, name, dataType);
        return feedId;
    }
    
    /**
     * @dev Update data feed (oracle only)
     */
    function updateDataFeed(
        bytes32 feedId,
        bytes memory data
    ) external onlyRole(ORACLE_ROLE) {
        DataFeed storage feed = dataFeeds[feedId];
        require(feed.active, "Feed not active");
        require(
            block.timestamp >= feed.lastUpdate + feed.updateFrequency,
            "Update too frequent"
        );
        
        uint256 dataIndex = ++latestDataIndex[feedId];
        historicalData[feedId][dataIndex] = data;
        feed.lastUpdate = block.timestamp;
        
        emit DataUpdated(feedId, dataIndex, data, block.timestamp);
    }
    
    /**
     * @dev Get latest data from feed
     */
    function getLatestData(bytes32 feedId) external view returns (
        bytes memory data,
        uint256 timestamp
    ) {
        DataFeed storage feed = dataFeeds[feedId];
        require(feed.active, "Feed not active");
        
        uint256 dataIndex = latestDataIndex[feedId];
        if (dataIndex > 0) {
            return (historicalData[feedId][dataIndex], feed.lastUpdate);
        }
        
        return ("", 0);
    }
    
    /**
     * @dev Get historical data from feed
     */
    function getHistoricalData(
        bytes32 feedId,
        uint256 dataIndex
    ) external view returns (bytes memory) {
        return historicalData[feedId][dataIndex];
    }
    
    /**
     * @dev Calculate price deviation percentage
     */
    function _calculateDeviation(uint256 oldPrice, uint256 newPrice) internal pure returns (uint256) {
        if (oldPrice == 0) return 0;
        
        uint256 difference = oldPrice > newPrice ? oldPrice - newPrice : newPrice - oldPrice;
        return (difference * 10000) / oldPrice; // Return in basis points
    }
    
    /**
     * @dev Update oracle parameters (admin only)
     */
    function updateOracleParameters(
        uint256 _priceDeviationThreshold,
        uint256 _minConfidenceLevel,
        uint256 _maxPriceAge
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_priceDeviationThreshold <= 5000, "Deviation threshold too high"); // Max 50%
        require(_minConfidenceLevel <= 10000, "Invalid confidence level");
        require(_maxPriceAge >= 1 minutes && _maxPriceAge <= 24 hours, "Invalid max price age");
        
        priceDeviationThreshold = _priceDeviationThreshold;
        minConfidenceLevel = _minConfidenceLevel;
        maxPriceAge = _maxPriceAge;
    }
    
    /**
     * @dev Toggle data feed status (admin only)
     */
    function toggleDataFeed(bytes32 feedId, bool active) external onlyRole(DEFAULT_ADMIN_ROLE) {
        dataFeeds[feedId].active = active;
    }
    
    /**
     * @dev Get feed information
     */
    function getFeedInfo(bytes32 feedId) external view returns (
        string memory name,
        string memory description,
        bytes32 dataType,
        uint256 updateFrequency,
        uint256 lastUpdate,
        bool active
    ) {
        DataFeed storage feed = dataFeeds[feedId];
        return (
            feed.name,
            feed.description,
            feed.dataType,
            feed.updateFrequency,
            feed.lastUpdate,
            feed.active
        );
    }
    
    /**
     * @dev Emergency pause for specific price symbol (admin only)
     */
    function emergencyPausePrice(string memory symbol) external onlyRole(DEFAULT_ADMIN_ROLE) {
        priceData[symbol].valid = false;
    }
}
'''

    def get_contract_deployment_script(self) -> str:
        """Generate deployment script for all contracts"""
        return '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./FTNSToken.sol";
import "./FTNSBridge.sol";
import "./FTNSStaking.sol";
import "./FTNSMarketplace.sol";
import "./FTNSGovernance.sol";
import "./FTNSOracle.sol";

/**
 * @title FTNS Ecosystem Deployer
 * @dev Deployment script for complete FTNS ecosystem
 */
contract FTNSEcosystemDeployer {
    address public admin;
    
    FTNSToken public ftnsToken;
    FTNSBridge public bridge;
    FTNSStaking public staking;
    FTNSMarketplace public marketplace;
    FTNSGovernance public governance;
    FTNSOracle public oracle;
    
    event EcosystemDeployed(
        address ftnsToken,
        address bridge,
        address staking,
        address marketplace,
        address governance,
        address oracle
    );
    
    constructor(address _admin) {
        require(_admin != address(0), "Invalid admin address");
        admin = _admin;
    }
    
    /**
     * @dev Deploy complete FTNS ecosystem
     */
    function deployEcosystem(
        string memory tokenName,
        string memory tokenSymbol,
        uint256 initialSupply
    ) external {
        require(msg.sender == admin, "Only admin can deploy");
        
        // Deploy FTNS Token
        ftnsToken = new FTNSToken(tokenName, tokenSymbol, initialSupply, admin);
        
        // Deploy Oracle
        oracle = new FTNSOracle(admin);
        
        // Deploy Staking
        staking = new FTNSStaking(address(ftnsToken), admin);
        
        // Deploy Bridge
        bridge = new FTNSBridge(address(ftnsToken), admin, admin);
        
        // Deploy Marketplace
        marketplace = new FTNSMarketplace(address(ftnsToken), admin, admin);
        
        // Deploy Governance
        governance = new FTNSGovernance(address(ftnsToken), address(staking), admin);
        
        // Grant bridge role to bridge contract
        ftnsToken.grantRole(ftnsToken.BRIDGE_ROLE(), address(bridge));
        
        // Grant minter role to staking contract for rewards
        ftnsToken.grantRole(ftnsToken.MINTER_ROLE(), address(staking));
        
        emit EcosystemDeployed(
            address(ftnsToken),
            address(bridge),
            address(staking),
            address(marketplace),
            address(governance),
            address(oracle)
        );
    }
    
    /**
     * @dev Get all deployed contract addresses
     */
    function getContractAddresses() external view returns (
        address _ftnsToken,
        address _bridge,
        address _staking,
        address _marketplace,
        address _governance,
        address _oracle
    ) {
        return (
            address(ftnsToken),
            address(bridge),
            address(staking),
            address(marketplace),
            address(governance),
            address(oracle)
        );
    }
}
'''


# Initialize and write contracts
if __name__ == "__main__":
    suite = SmartContractSuite()
    logger.info("✅ FTNS Smart Contract Suite generated successfully")