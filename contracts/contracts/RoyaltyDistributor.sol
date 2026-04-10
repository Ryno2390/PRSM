// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

interface IProvenanceRegistry {
    /// @dev Slim getter — returns only what the distributor needs. The
    ///      registry's full `contents()` getter would also return an
    ///      unbounded `string metadataUri`, which a squatter could use to
    ///      grief payment gas. We deliberately avoid that path here.
    function getCreatorAndRate(bytes32 contentHash)
        external
        view
        returns (address creator, uint16 royaltyRateBps);
}

/**
 * @title RoyaltyDistributor
 * @notice Atomic three-way splitter for FTNS payments tied to registered content.
 * @dev Stateless. Pulls FTNS from msg.sender via transferFrom, so the payer must
 *      have approved this contract for at least `gross` first. Splits:
 *        creator share = gross * royaltyRateBps / 10000  (from registry)
 *        network share = gross * NETWORK_FEE_BPS / 10000 (constant 2%)
 *        serving node share = gross - creator - network
 *      Reverts if creator + network share exceed gross
 *      (i.e. royaltyRateBps + 200 > 10000).
 */
contract RoyaltyDistributor is ReentrancyGuard {
    IERC20 public immutable ftns;
    IProvenanceRegistry public immutable registry;
    address public immutable networkTreasury;

    /// @dev 2% network fee, basis points
    uint16 public constant NETWORK_FEE_BPS = 200;

    event RoyaltyPaid(
        bytes32 indexed contentHash,
        address indexed payer,
        address indexed creator,
        address servingNode,
        uint256 creatorAmount,
        uint256 networkAmount,
        uint256 servingNodeAmount
    );

    constructor(address _ftns, address _registry, address _networkTreasury) {
        require(_ftns != address(0), "Zero ftns");
        require(_registry != address(0), "Zero registry");
        require(_networkTreasury != address(0), "Zero treasury");
        ftns = IERC20(_ftns);
        registry = IProvenanceRegistry(_registry);
        networkTreasury = _networkTreasury;
    }

    /**
     * @notice Pull `gross` FTNS from msg.sender and split it 3 ways.
     * @param contentHash content registered in ProvenanceRegistry
     * @param servingNode address of node that served the content
     * @param gross total FTNS amount being distributed (in token base units)
     */
    function distributeRoyalty(
        bytes32 contentHash,
        address servingNode,
        uint256 gross
    ) external nonReentrant {
        require(gross > 0, "Zero amount");
        require(servingNode != address(0), "Zero serving node");

        (address creator, uint16 rateBps) = registry.getCreatorAndRate(contentHash);
        require(creator != address(0), "Not registered");

        uint256 creatorAmt = (gross * rateBps) / 10000;
        uint256 networkAmt = (gross * NETWORK_FEE_BPS) / 10000;
        require(creatorAmt + networkAmt <= gross, "Rate plus fee exceeds 100%");
        uint256 nodeAmt = gross - creatorAmt - networkAmt;

        // Pull full amount once, then push to recipients.
        require(ftns.transferFrom(msg.sender, address(this), gross), "Pull failed");

        if (creatorAmt > 0) {
            require(ftns.transfer(creator, creatorAmt), "Creator xfer failed");
        }
        if (networkAmt > 0) {
            require(ftns.transfer(networkTreasury, networkAmt), "Network xfer failed");
        }
        if (nodeAmt > 0) {
            require(ftns.transfer(servingNode, nodeAmt), "Node xfer failed");
        }

        emit RoyaltyPaid(
            contentHash,
            msg.sender,
            creator,
            servingNode,
            creatorAmt,
            networkAmt,
            nodeAmt
        );
    }

    /**
     * @notice Read-only preview of how `gross` would be split for `contentHash`.
     * @return creatorAmount FTNS that would be sent to the registered creator
     * @return networkAmount FTNS that would be sent to the network treasury
     * @return servingNodeAmount FTNS that would be sent to the serving node
     */
    function preview(bytes32 contentHash, uint256 gross)
        external
        view
        returns (
            uint256 creatorAmount,
            uint256 networkAmount,
            uint256 servingNodeAmount
        )
    {
        (address creator, uint16 rateBps) = registry.getCreatorAndRate(contentHash);
        require(creator != address(0), "Not registered");

        creatorAmount = (gross * rateBps) / 10000;
        networkAmount = (gross * NETWORK_FEE_BPS) / 10000;
        require(creatorAmount + networkAmount <= gross, "Rate plus fee exceeds 100%");
        servingNodeAmount = gross - creatorAmount - networkAmount;
    }
}
