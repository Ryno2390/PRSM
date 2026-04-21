// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title BatchSettlementRegistry
 * @notice On-chain anchor for Phase 3.1 batched-settlement receipts.
 *
 * Providers accumulate Phase 2 ShardExecutionReceipts locally, build a
 * Merkle tree over them, and post the root here in a single commitBatch
 * transaction. After a challenge window elapses (default 3 days), the
 * provider calls finalizeBatch, which transitions state + emits the
 * BatchFinalized event that the Phase 3.1 settlement client consumes
 * to execute FTNS transfers.
 *
 * This file is the Phase 3.1 Task 1 deliverable: core batch-state machine
 * only. The actual FTNS transfer execution is wired in by Task 2
 * (EscrowPool.sol integration); the dispute/challenge surface is added in
 * Task 3. Both are additive — Task 1's state transitions remain the
 * authority on batch lifecycle.
 *
 * Key design choices (see docs/2026-04-21-phase3.1-batch-settlement-design.md):
 *   - Merkle-root-only commit. Individual receipt bytes are NOT on chain;
 *     challengers supply them inline (Task 3). Keeps commit gas ~100K.
 *   - Deterministic batchId = keccak256(provider || merkleRoot ||
 *     receiptCount || commitBlock || sequencePerProvider). Allows multiple
 *     simultaneous batches from the same provider without collision.
 *   - Challenge-window is a contract-level state variable (not per-batch)
 *     so Foundation governance can adjust it per PRSM-GOV-1 §4.2 without
 *     upgrading the contract. Default: 3 days.
 *   - Ownable for the governance-adjustable parameter only; batch
 *     operations are permissionless (anyone can commit; anyone can
 *     finalize a pending batch whose window has elapsed).
 */
contract BatchSettlementRegistry is Ownable {
    enum BatchStatus {
        NONEXISTENT, // default — batchId never committed
        PENDING,     // committed; within challenge window
        FINALIZED,   // past challenge window + finalizeBatch called
        VOIDED       // reserved for future governance voiding (e.g., if the
                     // whole batch is invalidated en bloc); not reachable in Task 1
    }

    struct Batch {
        address provider;           // who submitted + claims payment
        bytes32 merkleRoot;         // root of the receipt-hash tree
        uint256 receiptCount;       // number of receipts in the tree
        uint256 totalValueFTNS;     // sum of receipt values, pre-challenge
        uint256 invalidatedValueFTNS; // sum of successfully-challenged values; set by Task 3
        uint64  commitTimestamp;    // block.timestamp at commit
        BatchStatus status;
        string metadataURI;         // optional IPFS pointer (see §10.7 of design)
    }

    /// @dev Challenge window in seconds. Governance-adjustable.
    /// Default 3 days per design §5.1.
    uint256 public challengeWindowSeconds;

    /// @dev Per-provider monotonic counter used in batchId derivation to
    /// avoid collision when a provider commits multiple batches in the
    /// same block with identical merkle roots (pathological, but we want
    /// the contract to be correct under that case).
    mapping(address provider => uint256) public providerBatchSequence;

    /// @dev Primary state: all committed batches keyed by deterministic batchId.
    mapping(bytes32 batchId => Batch) public batches;

    /// @dev Minimum and maximum allowed challenge window values. Prevents
    /// governance from setting pathological values (e.g., 0 = no challenge
    /// period; 100 years = funds locked forever). Bounds themselves are
    /// owner-adjustable only via contract upgrade, not governance.
    uint256 public constant MIN_CHALLENGE_WINDOW_SECONDS = 1 hours;
    uint256 public constant MAX_CHALLENGE_WINDOW_SECONDS = 30 days;

    event BatchCommitted(
        bytes32 indexed batchId,
        address indexed provider,
        bytes32 merkleRoot,
        uint256 receiptCount,
        uint256 totalValueFTNS,
        uint64 commitTimestamp,
        string metadataURI
    );

    event BatchFinalized(
        bytes32 indexed batchId,
        address indexed provider,
        uint256 finalValueFTNS,
        uint256 invalidatedCount,
        uint64 finalizeTimestamp
    );

    event ChallengeWindowUpdated(uint256 oldSeconds, uint256 newSeconds);

    error InvalidChallengeWindow(uint256 provided);
    error BatchAlreadyCommitted(bytes32 batchId);
    error BatchNotFound(bytes32 batchId);
    error BatchNotPending(bytes32 batchId, BatchStatus current);
    error ChallengeWindowNotElapsed(bytes32 batchId, uint64 commitTimestamp, uint256 windowSeconds);
    error EmptyMerkleRoot();
    error ZeroReceiptCount();
    error ZeroProvider();

    constructor(address initialOwner, uint256 initialChallengeWindow) Ownable(initialOwner) {
        if (initialChallengeWindow < MIN_CHALLENGE_WINDOW_SECONDS ||
            initialChallengeWindow > MAX_CHALLENGE_WINDOW_SECONDS) {
            revert InvalidChallengeWindow(initialChallengeWindow);
        }
        challengeWindowSeconds = initialChallengeWindow;
    }

    /**
     * @notice Commit a batch of off-chain receipts as a single Merkle-root
     *         anchor. Permissionless: any provider can submit.
     * @param merkleRoot keccak256 Merkle root over the set of receipt-hash leaves
     * @param receiptCount number of receipts in the batch
     * @param totalValueFTNS sum of receipt values in FTNS base units
     * @param metadataURI optional off-chain pointer (e.g., ipfs://...)
     * @return batchId deterministic identifier for the committed batch
     */
    function commitBatch(
        bytes32 merkleRoot,
        uint256 receiptCount,
        uint256 totalValueFTNS,
        string calldata metadataURI
    ) external returns (bytes32 batchId) {
        if (merkleRoot == bytes32(0)) revert EmptyMerkleRoot();
        if (receiptCount == 0) revert ZeroReceiptCount();

        uint256 sequence = providerBatchSequence[msg.sender]++;

        batchId = keccak256(
            abi.encode(msg.sender, merkleRoot, receiptCount, block.number, sequence)
        );

        // Defensive: deterministic derivation means this should never
        // collide within the same block, but check explicitly in case
        // the derivation ever changes.
        if (batches[batchId].status != BatchStatus.NONEXISTENT) {
            revert BatchAlreadyCommitted(batchId);
        }

        batches[batchId] = Batch({
            provider: msg.sender,
            merkleRoot: merkleRoot,
            receiptCount: receiptCount,
            totalValueFTNS: totalValueFTNS,
            invalidatedValueFTNS: 0,
            commitTimestamp: uint64(block.timestamp),
            status: BatchStatus.PENDING,
            metadataURI: metadataURI
        });

        emit BatchCommitted(
            batchId,
            msg.sender,
            merkleRoot,
            receiptCount,
            totalValueFTNS,
            uint64(block.timestamp),
            metadataURI
        );
    }

    /**
     * @notice Finalize a PENDING batch after its challenge window has elapsed.
     *         Permissionless: anyone may call (typically the provider does,
     *         since they benefit from settlement, but the contract accepts
     *         calls from any address so a watchdog can finalize on behalf
     *         of absent providers).
     *
     *         Task 1 scope: transitions state + emits event. Task 2 will
     *         add the FTNS transfer invocation here against EscrowPool;
     *         Task 3 will account for invalidated receipt values against
     *         the final payable amount.
     *
     * @param batchId identifier returned by commitBatch
     */
    function finalizeBatch(bytes32 batchId) external {
        Batch storage b = batches[batchId];

        if (b.status == BatchStatus.NONEXISTENT) revert BatchNotFound(batchId);
        if (b.status != BatchStatus.PENDING) {
            revert BatchNotPending(batchId, b.status);
        }

        uint256 elapsed = block.timestamp - b.commitTimestamp;
        if (elapsed < challengeWindowSeconds) {
            revert ChallengeWindowNotElapsed(
                batchId, b.commitTimestamp, challengeWindowSeconds
            );
        }

        b.status = BatchStatus.FINALIZED;

        // Final payable = total - invalidated (Task 3 sets invalidatedValueFTNS).
        // In Task 1 scope, invalidatedValueFTNS is always 0, so finalValue ==
        // totalValueFTNS. Task 2 will pipe finalValue into EscrowPool transfers.
        uint256 finalValue = b.totalValueFTNS - b.invalidatedValueFTNS;

        emit BatchFinalized(
            batchId,
            b.provider,
            finalValue,
            0, // invalidatedCount; Task 3 will track and emit
            uint64(block.timestamp)
        );
    }

    // ── Governance surface ────────────────────────────────────────

    /**
     * @notice Update the challenge-window duration. Owner-only.
     *         Does NOT affect batches already in PENDING state — they
     *         retain the window value at their commit time, which is
     *         derived from the old challengeWindowSeconds at commit.
     *
     *         Wait: this is subtle. The storage holds batches keyed by
     *         commitTimestamp; the finalization check reads the CURRENT
     *         challengeWindowSeconds. So a post-change finalization of
     *         an already-pending batch uses the NEW window. This is
     *         acceptable because:
     *           1. Shrinking the window lets older pending batches
     *              finalize sooner (favorable to provider; no one loses
     *              value).
     *           2. Expanding the window delays older pending batches'
     *              finalization — this is a governance decision, publicly
     *              announced per PRSM-GOV-1 §10.3 (14-day notice period
     *              for non-emergency on-chain transactions).
     *         Callers relying on a specific finalization horizon must
     *         read challengeWindowSeconds at commit time AND monitor for
     *         governance-event adjustments.
     *
     * @param newSeconds new window duration in seconds
     */
    function setChallengeWindowSeconds(uint256 newSeconds) external onlyOwner {
        if (newSeconds < MIN_CHALLENGE_WINDOW_SECONDS ||
            newSeconds > MAX_CHALLENGE_WINDOW_SECONDS) {
            revert InvalidChallengeWindow(newSeconds);
        }
        uint256 old = challengeWindowSeconds;
        challengeWindowSeconds = newSeconds;
        emit ChallengeWindowUpdated(old, newSeconds);
    }

    // ── Views ────────────────────────────────────────────────────

    /// @notice Read the full Batch struct for a given batchId.
    function getBatch(bytes32 batchId) external view returns (Batch memory) {
        return batches[batchId];
    }

    /// @notice True iff the batch exists + is PENDING + window has elapsed.
    /// @dev Lightweight pre-check for would-be finalizers.
    function isFinalizable(bytes32 batchId) external view returns (bool) {
        Batch storage b = batches[batchId];
        if (b.status != BatchStatus.PENDING) return false;
        return (block.timestamp - b.commitTimestamp) >= challengeWindowSeconds;
    }

    /// @notice Seconds remaining until a PENDING batch can be finalized.
    ///         Returns 0 if the window has elapsed or the batch isn't PENDING.
    function secondsUntilFinalizable(bytes32 batchId) external view returns (uint256) {
        Batch storage b = batches[batchId];
        if (b.status != BatchStatus.PENDING) return 0;
        uint256 elapsed = block.timestamp - b.commitTimestamp;
        if (elapsed >= challengeWindowSeconds) return 0;
        return challengeWindowSeconds - elapsed;
    }
}
