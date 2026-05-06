// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/**
 * @title ProvenanceRegistryV2
 * @notice Successor to ProvenanceRegistry that adds an optional commitment to
 *         the content's semantic fingerprint (embedding hash + kind tag).
 *         PRSM-PROV-1 Item 7 — closes the "what was the content actually?"
 *         gap that the v1 hash-only registry leaves open in disputes.
 *
 * @dev v1 (deployed at Base mainnet 0xdF47...9915) is NOT upgradeable; this
 *      contract is a *separate* deployment. New uploads register here; old
 *      uploads remain in v1. Off-chain RoyaltyDistributor reads from both.
 *
 *      Mainnet deployment of THIS contract is gated behind L4 audit firm
 *      review per PRSM-PROV-1 plan §4.5 + PRSM-POL-1 §5. Sepolia deployment
 *      is unrestricted and is what T7.7 exercises.
 *
 *      Storage shape: identical to v1's `Content` struct, plus two bytes32
 *      slots. Both new fields can legitimately be zero — a creator may
 *      register byte-hash-only content (no embedding) and disputes fall
 *      back to v1-style raw-hash matching.
 */
contract ProvenanceRegistryV2 {
    struct Content {
        address creator;              // Receives royalties
        uint16 royaltyRateBps;        // 0..MAX_ROYALTY_RATE_BPS
        uint64 registeredAt;          // unix seconds
        bytes32 embeddingCommitment;  // keccak256(model_id || dim_be || vector_bytes); zero = no commitment
        bytes32 fingerprintKind;      // keccak256("text-vector" | "image-phash" | ...); zero = byte-hash only
        string metadataUri;           // ipfs://, https://, etc.
    }

    /// @dev Must equal 10000 - RoyaltyDistributor.NETWORK_FEE_BPS so a registered
    ///      rate can never make the distributor's split overflow gross. Unchanged
    ///      from v1 — the network-fee constant did not move.
    uint16 public constant MAX_ROYALTY_RATE_BPS = 9800;

    mapping(bytes32 => Content) public contents;

    event ContentRegistered(
        bytes32 indexed contentHash,
        address indexed creator,
        uint16 royaltyRateBps,
        bytes32 embeddingCommitment,
        bytes32 fingerprintKind,
        string metadataUri
    );

    event OwnershipTransferred(
        bytes32 indexed contentHash,
        address indexed previousCreator,
        address indexed newCreator
    );

    /**
     * @notice Register new content with optional embedding commitment.
     * @dev Pass `bytes32(0)` for both new args to register byte-hash-only
     *      content (legacy behavior). The two new fields are EVENT-emitted and
     *      stored so an indexer can reconstruct the dispute state without
     *      re-reading every storage slot.
     * @param contentHash keccak256 of canonical content bytes (creator-bound;
     *        compute via `prsm.economy.web3.provenance_registry.compute_content_hash`)
     * @param royaltyRateBps royalty rate in basis points (0..MAX_ROYALTY_RATE_BPS)
     * @param metadataUri off-chain pointer to descriptive metadata
     * @param embeddingCommitment keccak256(model_id || uint32_be(dim) || vector_bytes), or zero
     * @param fingerprintKind keccak256(kind_label_bytes), or zero
     */
    function registerContent(
        bytes32 contentHash,
        uint16 royaltyRateBps,
        string calldata metadataUri,
        bytes32 embeddingCommitment,
        bytes32 fingerprintKind
    ) external {
        require(contents[contentHash].creator == address(0), "Already registered");
        require(royaltyRateBps <= MAX_ROYALTY_RATE_BPS, "Rate exceeds max");

        contents[contentHash] = Content({
            creator: msg.sender,
            royaltyRateBps: royaltyRateBps,
            registeredAt: uint64(block.timestamp),
            embeddingCommitment: embeddingCommitment,
            fingerprintKind: fingerprintKind,
            metadataUri: metadataUri
        });

        emit ContentRegistered(
            contentHash,
            msg.sender,
            royaltyRateBps,
            embeddingCommitment,
            fingerprintKind,
            metadataUri
        );
    }

    /**
     * @notice Transfer creator role — same semantics as v1.
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
     * @notice True iff `contentHash` has a creator on this v2 registry.
     */
    function isRegistered(bytes32 contentHash) external view returns (bool) {
        return contents[contentHash].creator != address(0);
    }

    /**
     * @notice Slim payment-hot-path accessor — returns only creator + rate
     *         and avoids loading the unbounded `metadataUri` string.
     */
    function getCreatorAndRate(bytes32 contentHash)
        external
        view
        returns (address creator, uint16 royaltyRateBps)
    {
        Content storage c = contents[contentHash];
        return (c.creator, c.royaltyRateBps);
    }

    /**
     * @notice Returns the embedding commitment + kind for a registered piece
     *         of content. Both fields zero when the registrant did not
     *         commit an embedding (legacy / byte-hash-only path).
     */
    function getEmbeddingCommitment(bytes32 contentHash)
        external
        view
        returns (bytes32 embeddingCommitment, bytes32 fingerprintKind)
    {
        Content storage c = contents[contentHash];
        return (c.embeddingCommitment, c.fingerprintKind);
    }

    /**
     * @notice Pure dispute-helper: returns true iff `claimed` matches the
     *         on-chain commitment for `contentHash`. Off-chain code computes
     *         `claimed = keccak256(model_id || uint32_be(dim) || vector_bytes)`
     *         from a candidate vector and submits the result; on a true
     *         return, that vector is the one the creator committed to at
     *         registration. Zero commitment never matches (zero claimed
     *         passed into a zero-committed record returns false too — we
     *         require a real positive match for arbitration).
     */
    function verifyEmbeddingCommitment(bytes32 contentHash, bytes32 claimed)
        external
        view
        returns (bool)
    {
        bytes32 onChain = contents[contentHash].embeddingCommitment;
        if (onChain == bytes32(0)) {
            return false;
        }
        return onChain == claimed;
    }
}
