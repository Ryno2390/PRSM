// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title FTNS Token
 * @dev The FTNS (Federated Token Network System) token for PRSM
 * 
 * Features:
 * - ERC20 compliant token with permit functionality
 * - Pausable transfers for emergency situations
 * - Ownable with transfer restrictions
 * - Context-based staking and locking mechanisms
 * - Usage-based minting and burning
 * - Governance integration ready
 */
contract FTNSToken is ERC20, ERC20Permit, Ownable, Pausable, ReentrancyGuard {
    
    // Events
    event TokensLocked(address indexed user, uint256 amount, uint256 unlockTime, string context);
    event TokensUnlocked(address indexed user, uint256 amount, string context);
    event TokensStaked(address indexed user, uint256 amount, string context);
    event TokensUnstaked(address indexed user, uint256 amount, string context);
    event ContextUsage(address indexed user, string context, uint256 tokens, uint256 computeUnits);
    event RewardsDistributed(address indexed user, uint256 amount, string reason);
    
    // Structs
    struct Lock {
        uint256 amount;
        uint256 unlockTime;
        string context;
        bool withdrawn;
    }
    
    struct Stake {
        uint256 amount;
        uint256 stakeTime;
        string context;
        uint256 rewards;
    }
    
    // State variables
    mapping(address => Lock[]) public userLocks;
    mapping(address => Stake[]) public userStakes;
    mapping(address => mapping(string => uint256)) public contextAllocations;
    mapping(string => uint256) public contextTotalStaked;
    mapping(address => bool) public authorizedMinters;
    mapping(address => bool) public authorizedBurners;
    
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1 billion tokens
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100 million tokens
    uint256 public totalStaked;
    uint256 public totalLocked;
    
    // Reward rates (per second)
    uint256 public baseStakingReward = 1e15; // 0.001 tokens per second
    uint256 public contextMultiplier = 2; // 2x for context staking
    
    constructor(address initialOwner) 
        ERC20("FTNS Token", "FTNS")
        ERC20Permit("FTNS Token")
        Ownable(initialOwner)
    {
        _mint(initialOwner, INITIAL_SUPPLY);
    }
    
    // Override required by Solidity
    function _beforeTokenTransfer(address from, address to, uint256 amount)
        internal
        override
        whenNotPaused
    {
        super._beforeTokenTransfer(from, to, amount);
    }
    
    /**
     * @dev Pause token transfers - emergency only
     */
    function pause() public onlyOwner {
        _pause();
    }
    
    /**
     * @dev Unpause token transfers
     */
    function unpause() public onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Mint tokens - only authorized minters
     */
    function mint(address to, uint256 amount) public {
        require(authorizedMinters[msg.sender] || msg.sender == owner(), "Not authorized to mint");
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }
    
    /**
     * @dev Burn tokens - only authorized burners
     */
    function burn(uint256 amount) public {
        require(authorizedBurners[msg.sender] || msg.sender == owner(), "Not authorized to burn");
        _burn(msg.sender, amount);
    }
    
    /**
     * @dev Burn tokens from account - only authorized burners
     */
    function burnFrom(address account, uint256 amount) public {
        require(authorizedBurners[msg.sender] || msg.sender == owner(), "Not authorized to burn");
        _spendAllowance(account, msg.sender, amount);
        _burn(account, amount);
    }
    
    /**
     * @dev Lock tokens for a specific context and duration
     */
    function lockTokens(uint256 amount, uint256 duration, string memory context) 
        public 
        nonReentrant 
    {
        require(amount > 0, "Amount must be greater than 0");
        require(duration > 0, "Duration must be greater than 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        _transfer(msg.sender, address(this), amount);
        
        uint256 unlockTime = block.timestamp + duration;
        userLocks[msg.sender].push(Lock({
            amount: amount,
            unlockTime: unlockTime,
            context: context,
            withdrawn: false
        }));
        
        totalLocked += amount;
        contextAllocations[msg.sender][context] += amount;
        
        emit TokensLocked(msg.sender, amount, unlockTime, context);
    }
    
    /**
     * @dev Unlock tokens that have reached their unlock time
     */
    function unlockTokens(uint256 lockIndex) public nonReentrant {
        require(lockIndex < userLocks[msg.sender].length, "Invalid lock index");
        
        Lock storage lock = userLocks[msg.sender][lockIndex];
        require(!lock.withdrawn, "Already withdrawn");
        require(block.timestamp >= lock.unlockTime, "Lock period not expired");
        
        lock.withdrawn = true;
        totalLocked -= lock.amount;
        contextAllocations[msg.sender][lock.context] -= lock.amount;
        
        _transfer(address(this), msg.sender, lock.amount);
        
        emit TokensUnlocked(msg.sender, lock.amount, lock.context);
    }
    
    /**
     * @dev Stake tokens for a specific context
     */
    function stakeTokens(uint256 amount, string memory context) 
        public 
        nonReentrant 
    {
        require(amount > 0, "Amount must be greater than 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        _transfer(msg.sender, address(this), amount);
        
        userStakes[msg.sender].push(Stake({
            amount: amount,
            stakeTime: block.timestamp,
            context: context,
            rewards: 0
        }));
        
        totalStaked += amount;
        contextTotalStaked[context] += amount;
        contextAllocations[msg.sender][context] += amount;
        
        emit TokensStaked(msg.sender, amount, context);
    }
    
    /**
     * @dev Unstake tokens and claim rewards
     */
    function unstakeTokens(uint256 stakeIndex) public nonReentrant {
        require(stakeIndex < userStakes[msg.sender].length, "Invalid stake index");
        
        Stake storage stake = userStakes[msg.sender][stakeIndex];
        require(stake.amount > 0, "Already unstaked");
        
        // Calculate rewards
        uint256 stakingDuration = block.timestamp - stake.stakeTime;
        uint256 rewards = calculateStakingRewards(stake.amount, stakingDuration, stake.context);
        
        // Update state
        uint256 totalReturn = stake.amount + rewards;
        totalStaked -= stake.amount;
        contextTotalStaked[stake.context] -= stake.amount;
        contextAllocations[msg.sender][stake.context] -= stake.amount;
        
        // Mint rewards if within max supply
        if (totalSupply() + rewards <= MAX_SUPPLY) {
            _mint(address(this), rewards);
        } else {
            rewards = 0;
        }
        
        // Reset stake
        stake.amount = 0;
        stake.rewards = rewards;
        
        // Transfer tokens back
        _transfer(address(this), msg.sender, totalReturn);
        
        emit TokensUnstaked(msg.sender, stake.amount, stake.context);
        if (rewards > 0) {
            emit RewardsDistributed(msg.sender, rewards, "staking_rewards");
        }
    }
    
    /**
     * @dev Record context usage and potentially distribute rewards
     */
    function recordContextUsage(
        address user, 
        string memory context, 
        uint256 computeUnits
    ) public {
        require(authorizedMinters[msg.sender] || msg.sender == owner(), "Not authorized");
        
        // Calculate tokens based on compute units
        uint256 tokensUsed = computeUnits / 1000; // 1 token per 1000 compute units
        
        // Distribute usage-based rewards
        if (contextAllocations[user][context] > 0) {
            uint256 rewardRate = (contextAllocations[user][context] * 1e18) / contextTotalStaked[context];
            uint256 rewards = (tokensUsed * rewardRate) / 1e18;
            
            if (totalSupply() + rewards <= MAX_SUPPLY) {
                _mint(user, rewards);
                emit RewardsDistributed(user, rewards, "usage_rewards");
            }
        }
        
        emit ContextUsage(user, context, tokensUsed, computeUnits);
    }
    
    /**
     * @dev Calculate staking rewards
     */
    function calculateStakingRewards(
        uint256 amount, 
        uint256 duration, 
        string memory context
    ) public view returns (uint256) {
        uint256 baseReward = (amount * baseStakingReward * duration) / 1e18;
        
        // Apply context multiplier for specific contexts
        if (keccak256(bytes(context)) != keccak256(bytes(""))) {
            baseReward = (baseReward * contextMultiplier);
        }
        
        return baseReward;
    }
    
    /**
     * @dev Get user's lock information
     */
    function getUserLocks(address user) public view returns (Lock[] memory) {
        return userLocks[user];
    }
    
    /**
     * @dev Get user's stake information
     */
    function getUserStakes(address user) public view returns (Stake[] memory) {
        return userStakes[user];
    }
    
    /**
     * @dev Get user's context allocation
     */
    function getContextAllocation(address user, string memory context) 
        public 
        view 
        returns (uint256) 
    {
        return contextAllocations[user][context];
    }
    
    /**
     * @dev Get total context staked
     */
    function getContextTotalStaked(string memory context) 
        public 
        view 
        returns (uint256) 
    {
        return contextTotalStaked[context];
    }
    
    /**
     * @dev Get user's liquid balance (not locked or staked)
     */
    function getLiquidBalance(address user) public view returns (uint256) {
        uint256 locked = 0;
        uint256 staked = 0;
        
        // Calculate locked tokens
        Lock[] memory locks = userLocks[user];
        for (uint i = 0; i < locks.length; i++) {
            if (!locks[i].withdrawn && block.timestamp < locks[i].unlockTime) {
                locked += locks[i].amount;
            }
        }
        
        // Calculate staked tokens
        Stake[] memory stakes = userStakes[user];
        for (uint i = 0; i < stakes.length; i++) {
            if (stakes[i].amount > 0) {
                staked += stakes[i].amount;
            }
        }
        
        return balanceOf(user) - locked - staked;
    }
    
    // Admin functions
    function setAuthorizedMinter(address minter, bool authorized) public onlyOwner {
        authorizedMinters[minter] = authorized;
    }
    
    function setAuthorizedBurner(address burner, bool authorized) public onlyOwner {
        authorizedBurners[burner] = authorized;
    }
    
    function setStakingReward(uint256 newRate) public onlyOwner {
        baseStakingReward = newRate;
    }
    
    function setContextMultiplier(uint256 newMultiplier) public onlyOwner {
        contextMultiplier = newMultiplier;
    }
    
    /**
     * @dev Emergency withdraw - only owner, only when paused
     */
    function emergencyWithdraw(address token, uint256 amount) public onlyOwner whenPaused {
        if (token == address(0)) {
            payable(owner()).transfer(amount);
        } else {
            IERC20(token).transfer(owner(), amount);
        }
    }
}