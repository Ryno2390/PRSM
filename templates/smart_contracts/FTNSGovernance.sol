// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorSettings.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorCountingSimple.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotes.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotesQuorumFraction.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorTimelockControl.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title FTNS Governance
 * @dev Governance contract for PRSM FTNS token ecosystem
 * 
 * Features:
 * - Token-based voting with delegation
 * - Configurable voting delay and period
 * - Quorum-based proposal execution
 * - Timelock integration for secure execution
 * - Context-specific governance proposals
 * - Emergency governance mechanisms
 */
contract FTNSGovernance is 
    Governor,
    GovernorSettings,
    GovernorCountingSimple,
    GovernorVotes,
    GovernorVotesQuorumFraction,
    GovernorTimelockControl,
    Ownable
{
    // Events
    event ContextProposalCreated(
        uint256 indexed proposalId,
        string context,
        address proposer,
        string description
    );
    
    event EmergencyActionExecuted(
        address indexed executor,
        address indexed target,
        bytes callData,
        string reason
    );
    
    // Structs
    struct ContextProposal {
        string context;
        uint256 requiredStake;
        uint256 contextQuorum;
        bool active;
    }
    
    // State variables
    mapping(uint256 => ContextProposal) public contextProposals;
    mapping(string => uint256) public contextMinStake;
    mapping(address => bool) public emergencyExecutors;
    
    bool public emergencyMode = false;
    uint256 public emergencyDelay = 0;
    
    constructor(
        IVotes _token,
        TimelockController _timelock,
        address initialOwner
    )
        Governor("FTNS Governance")
        GovernorSettings(
            1, /* 1 block voting delay */
            50400, /* 1 week voting period */
            1000e18 /* 1000 FTNS proposal threshold */
        )
        GovernorVotes(_token)
        GovernorVotesQuorumFraction(4) /* 4% quorum */
        GovernorTimelockControl(_timelock)
        Ownable(initialOwner)
    {}
    
    /**
     * @dev Create a context-specific proposal
     */
    function proposeContext(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description,
        string memory context
    ) public returns (uint256) {
        // Check if proposer has enough stake in the context
        uint256 requiredStake = contextMinStake[context];
        if (requiredStake > 0) {
            require(
                getVotes(msg.sender, block.number - 1) >= requiredStake,
                "Insufficient context stake"
            );
        }
        
        uint256 proposalId = propose(targets, values, calldatas, description);
        
        contextProposals[proposalId] = ContextProposal({
            context: context,
            requiredStake: requiredStake,
            contextQuorum: 0, // Use default quorum
            active: true
        });
        
        emit ContextProposalCreated(proposalId, context, msg.sender, description);
        
        return proposalId;
    }
    
    /**
     * @dev Execute proposal with context validation
     */
    function executeContext(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash,
        string memory context
    ) public payable returns (uint256) {
        uint256 proposalId = hashProposal(targets, values, calldatas, descriptionHash);
        
        // Validate context proposal
        ContextProposal storage contextProp = contextProposals[proposalId];
        require(contextProp.active, "Context proposal not active");
        require(
            keccak256(bytes(contextProp.context)) == keccak256(bytes(context)),
            "Context mismatch"
        );
        
        return execute(targets, values, calldatas, descriptionHash);
    }
    
    /**
     * @dev Override quorum calculation for context proposals
     */
    function quorum(uint256 blockNumber) public view override returns (uint256) {
        return super.quorum(blockNumber);
    }
    
    /**
     * @dev Get context-specific quorum
     */
    function contextQuorum(uint256 proposalId, uint256 blockNumber) 
        public 
        view 
        returns (uint256) 
    {
        ContextProposal storage contextProp = contextProposals[proposalId];
        
        if (contextProp.contextQuorum > 0) {
            return contextProp.contextQuorum;
        }
        
        return quorum(blockNumber);
    }
    
    /**
     * @dev Emergency proposal execution
     */
    function emergencyExecute(
        address target,
        bytes calldata callData,
        string calldata reason
    ) public {
        require(emergencyMode, "Emergency mode not active");
        require(emergencyExecutors[msg.sender], "Not authorized for emergency execution");
        
        (bool success, ) = target.call(callData);
        require(success, "Emergency execution failed");
        
        emit EmergencyActionExecuted(msg.sender, target, callData, reason);
    }
    
    /**
     * @dev Set emergency mode
     */
    function setEmergencyMode(bool _emergencyMode) public onlyOwner {
        emergencyMode = _emergencyMode;
        if (_emergencyMode) {
            emergencyDelay = 0; // Immediate execution in emergency
        } else {
            emergencyDelay = 1; // Restore normal delay
        }
    }
    
    /**
     * @dev Add/remove emergency executor
     */
    function setEmergencyExecutor(address executor, bool authorized) public onlyOwner {
        emergencyExecutors[executor] = authorized;
    }
    
    /**
     * @dev Set minimum stake required for context proposals
     */
    function setContextMinStake(string memory context, uint256 minStake) public onlyOwner {
        contextMinStake[context] = minStake;
    }
    
    /**
     * @dev Update governance settings
     */
    function updateSettings(
        uint256 newVotingDelay,
        uint256 newVotingPeriod,
        uint256 newProposalThreshold
    ) public onlyOwner {
        _setVotingDelay(newVotingDelay);
        _setVotingPeriod(newVotingPeriod);
        _setProposalThreshold(newProposalThreshold);
    }
    
    /**
     * @dev Update quorum fraction
     */
    function updateQuorumFraction(uint256 newQuorumFraction) public onlyOwner {
        _updateQuorumNumerator(newQuorumFraction);
    }
    
    // Override required functions
    function votingDelay() public view override(IGovernor, GovernorSettings) returns (uint256) {
        if (emergencyMode) {
            return emergencyDelay;
        }
        return super.votingDelay();
    }
    
    function votingPeriod() public view override(IGovernor, GovernorSettings) returns (uint256) {
        return super.votingPeriod();
    }
    
    function proposalThreshold() public view override(Governor, GovernorSettings) returns (uint256) {
        return super.proposalThreshold();
    }
    
    function state(uint256 proposalId)
        public
        view
        override(Governor, GovernorTimelockControl)
        returns (ProposalState)
    {
        return super.state(proposalId);
    }
    
    function propose(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description
    ) public override(Governor, IGovernor) returns (uint256) {
        return super.propose(targets, values, calldatas, description);
    }
    
    function _execute(
        uint256 proposalId,
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) internal override(Governor, GovernorTimelockControl) {
        super._execute(proposalId, targets, values, calldatas, descriptionHash);
    }
    
    function _cancel(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) internal override(Governor, GovernorTimelockControl) returns (uint256) {
        return super._cancel(targets, values, calldatas, descriptionHash);
    }
    
    function _executor() internal view override(Governor, GovernorTimelockControl) returns (address) {
        return super._executor();
    }
    
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(Governor, GovernorTimelockControl)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
    
    /**
     * @dev Get proposal context information
     */
    function getProposalContext(uint256 proposalId) 
        public 
        view 
        returns (ContextProposal memory) 
    {
        return contextProposals[proposalId];
    }
    
    /**
     * @dev Check if address can propose for context
     */
    function canProposeForContext(address proposer, string memory context) 
        public 
        view 
        returns (bool) 
    {
        uint256 requiredStake = contextMinStake[context];
        if (requiredStake == 0) {
            return getVotes(proposer, block.number - 1) >= proposalThreshold();
        }
        return getVotes(proposer, block.number - 1) >= requiredStake;
    }
    
    /**
     * @dev Get governance statistics
     */
    function getGovernanceStats() public view returns (
        uint256 totalProposals,
        uint256 activeProposals,
        uint256 totalVoters,
        uint256 currentQuorum
    ) {
        // This would need to be implemented with proposal tracking
        // For now, return basic info
        return (0, 0, 0, quorum(block.number - 1));
    }
}