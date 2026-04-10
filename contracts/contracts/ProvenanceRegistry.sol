// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/**
 * @title ProvenanceRegistry
 * @notice On-chain registry of PRSM content provenance — content hash to creator
 *         address and royalty rate. Source of truth for RoyaltyDistributor.
 * @dev Content is identified by an opaque bytes32 hash (keccak256 of canonical
 *      content bytes, set by the off-chain client). Royalty rates are basis
 *      points (1 bp = 0.01%, max 10000 = 100%). Records are immutable except
 *      for ownership transfer.
 */
contract ProvenanceRegistry {
    struct Content {
        address creator;        // Receives royalties
        uint16 royaltyRateBps;  // 0..10000
        uint64 registeredAt;    // unix seconds
        string metadataUri;     // ipfs://, https://, etc.
    }

    mapping(bytes32 => Content) public contents;

    event ContentRegistered(
        bytes32 indexed contentHash,
        address indexed creator,
        uint16 royaltyRateBps,
        string metadataUri
    );

    event OwnershipTransferred(
        bytes32 indexed contentHash,
        address indexed previousCreator,
        address indexed newCreator
    );

    /**
     * @notice Register a new piece of content. Caller becomes the creator.
     * @param contentHash keccak256 hash of canonical content bytes
     * @param royaltyRateBps royalty rate in basis points (max 10000)
     * @param metadataUri off-chain pointer to descriptive metadata
     */
    function registerContent(
        bytes32 contentHash,
        uint16 royaltyRateBps,
        string calldata metadataUri
    ) external {
        require(contents[contentHash].creator == address(0), "Already registered");
        require(royaltyRateBps <= 10000, "Invalid rate");

        contents[contentHash] = Content({
            creator: msg.sender,
            royaltyRateBps: royaltyRateBps,
            registeredAt: uint64(block.timestamp),
            metadataUri: metadataUri
        });

        emit ContentRegistered(contentHash, msg.sender, royaltyRateBps, metadataUri);
    }

    /**
     * @notice Transfer creator role for an existing piece of content.
     * @dev Only the current creator can transfer. New creator receives all
     *      future royalties. Royalty rate is preserved.
     */
    function transferContentOwnership(bytes32 contentHash, address newCreator) external {
        Content storage c = contents[contentHash];
        require(c.creator == msg.sender, "Not creator");
        require(newCreator != address(0), "Zero address");

        address previous = c.creator;
        c.creator = newCreator;

        emit OwnershipTransferred(contentHash, previous, newCreator);
    }

    /**
     * @notice Convenience accessor — true iff contentHash has a creator.
     */
    function isRegistered(bytes32 contentHash) external view returns (bool) {
        return contents[contentHash].creator != address(0);
    }
}
