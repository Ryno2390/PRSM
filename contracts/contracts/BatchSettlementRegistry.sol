// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

interface IEscrowPool {
    function settleFromRequester(address requester, address recipient, uint256 amount) external;
}

/// @notice Pluggable signature-verification surface used by the
/// INVALID_SIGNATURE challenge path. The production deployment
/// substitutes an audited Ed25519 verifier (per PRSM-PHASE3.1 §10.1
/// resolution). The interface is opaque to the cryptographic scheme
/// so a future migration (e.g., to BLS12-381 or secp256r1) doesn't
/// require contract surgery.
interface ISignatureVerifier {
    /// @return true iff `signature` is a valid signature of `messageHash`
    ///         under `publicKey` according to the implementing scheme.
    function verify(
        bytes32 messageHash,
        bytes calldata signature,
        bytes calldata publicKey
    ) external view returns (bool);
}

/// @notice Canonical on-chain representation of a Phase 2 ShardExecutionReceipt,
/// ABI-encoded + keccak256'd to produce Merkle-tree leaves.
///
/// NOTE: canonical-form alignment between this struct and the Python-side
/// off-chain encoding is a Task 5 deliverable. Task 3 ships the struct
/// + on-chain challenge dispatch; Task 5 locks the parity.
struct ReceiptLeaf {
    bytes32 jobIdHash;             // keccak256(utf8(receipt.job_id))
    uint32 shardIndex;             // receipt.shard_index
    bytes32 providerIdHash;        // keccak256(utf8(receipt.provider_id))
    bytes32 providerPubkeyHash;    // keccak256(base64decode(provider_pubkey_b64))
    bytes32 outputHash;            // bytes32 of hex-decoded receipt.output_hash
    uint64 executedAtUnix;         // receipt.executed_at_unix
    uint128 valueFtns;             // per-receipt quoted price in FTNS base units
    bytes32 signatureHash;         // keccak256(base64decode(receipt.signature))
}

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
        address requester;          // who owes payment (funds this batch from escrow)
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

    /// @dev Reason codes for challengeReceipt. Ordering is stable; adding
    /// new codes appends to the enum. MALFORMED is reserved but not
    /// implementable on-chain without extra infrastructure; use of it in
    /// Task 3 reverts with NotYetImplemented.
    enum ReasonCode {
        DOUBLE_SPEND,        // 0: receipt present in two committed batches
        INVALID_SIGNATURE,   // 1: Ed25519 sig doesn't verify under declared pubkey
        NO_ESCROW,           // 2: batch.requester attests no matching authorization
        EXPIRED,             // 3: receipt.executed_at_unix beyond lookback window
        MALFORMED            // 4: reserved (not yet implementable)
    }

    /// @dev EscrowPool contract authorized to execute settlement transfers.
    /// Set by owner via setEscrowPool; finalizeBatch invokes it to move
    /// FTNS from requester's escrow balance to provider's wallet.
    /// May be address(0) in Task 1 scope (no transfer happens); once set
    /// in Task 2, finalizeBatch requires it to be configured.
    IEscrowPool public escrowPool;

    /// @dev Pluggable Ed25519 verifier for INVALID_SIGNATURE challenges.
    /// May be address(0) until a production verifier is deployed;
    /// INVALID_SIGNATURE challenges revert with VerifierNotConfigured
    /// if it is unset.
    ISignatureVerifier public signatureVerifier;

    /// @dev Maximum receipt age (in seconds) that a provider may batch.
    /// EXPIRED challenges succeed if block.timestamp - leaf.executedAtUnix
    /// exceeds this window. Default 30 days per design §5.2 reason 3.
    /// Governance-adjustable within reasonable bounds.
    uint256 public settlementLookbackWindowSeconds;

    uint256 public constant MIN_LOOKBACK_SECONDS = 1 days;
    uint256 public constant MAX_LOOKBACK_SECONDS = 365 days;

    /// @dev invalidatedReceipts[batchId][receiptLeafHash] = true if this
    /// specific receipt has been successfully challenged. Prevents
    /// double-challenges that would double-subtract value.
    mapping(bytes32 batchId => mapping(bytes32 receiptLeafHash => bool))
        public invalidatedReceipts;

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
        uint256 invalidatedValueFTNS,
        uint64 finalizeTimestamp
    );

    event ChallengeWindowUpdated(uint256 oldSeconds, uint256 newSeconds);
    event EscrowPoolUpdated(address oldPool, address newPool);
    event SignatureVerifierUpdated(address oldVerifier, address newVerifier);
    event SettlementLookbackUpdated(uint256 oldSeconds, uint256 newSeconds);
    event ReceiptChallenged(
        bytes32 indexed batchId,
        bytes32 indexed receiptLeafHash,
        address indexed challenger,
        ReasonCode reason,
        uint128 invalidatedValueFTNS
    );

    error InvalidChallengeWindow(uint256 provided);
    error InvalidLookbackWindow(uint256 provided);
    error BatchAlreadyCommitted(bytes32 batchId);
    error BatchNotFound(bytes32 batchId);
    error BatchNotPending(bytes32 batchId, BatchStatus current);
    error ChallengeWindowNotElapsed(bytes32 batchId, uint64 commitTimestamp, uint256 windowSeconds);
    error ChallengeWindowElapsed(bytes32 batchId);
    error EmptyMerkleRoot();
    error ZeroReceiptCount();
    error ZeroRequester();
    error EscrowPoolNotConfigured();
    error InvalidMerkleProof(bytes32 batchId, bytes32 receiptLeafHash);
    error ReceiptAlreadyInvalidated(bytes32 batchId, bytes32 receiptLeafHash);
    error ChallengeNotProven(ReasonCode reason);
    error VerifierNotConfigured();
    error MalformedReasonNotImplemented();
    error CallerNotRequester(address caller, address requester);
    error ReceiptNotExpired(uint64 executedAtUnix, uint256 lookbackSeconds);
    error ConflictingBatchNotCommitted(bytes32 conflictingBatchId);

    constructor(address initialOwner, uint256 initialChallengeWindow) Ownable(initialOwner) {
        if (initialChallengeWindow < MIN_CHALLENGE_WINDOW_SECONDS ||
            initialChallengeWindow > MAX_CHALLENGE_WINDOW_SECONDS) {
            revert InvalidChallengeWindow(initialChallengeWindow);
        }
        challengeWindowSeconds = initialChallengeWindow;
        settlementLookbackWindowSeconds = 30 days; // default per design §5.2
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
        address requester,
        bytes32 merkleRoot,
        uint256 receiptCount,
        uint256 totalValueFTNS,
        string calldata metadataURI
    ) external returns (bytes32 batchId) {
        if (requester == address(0)) revert ZeroRequester();
        if (merkleRoot == bytes32(0)) revert EmptyMerkleRoot();
        if (receiptCount == 0) revert ZeroReceiptCount();

        uint256 sequence = providerBatchSequence[msg.sender]++;

        batchId = keccak256(
            abi.encode(msg.sender, requester, merkleRoot, receiptCount, block.number, sequence)
        );

        // Defensive: deterministic derivation means this should never
        // collide within the same block, but check explicitly in case
        // the derivation ever changes.
        if (batches[batchId].status != BatchStatus.NONEXISTENT) {
            revert BatchAlreadyCommitted(batchId);
        }

        batches[batchId] = Batch({
            provider: msg.sender,
            requester: requester,
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
        // In Task 2 scope, invalidatedValueFTNS is always 0, so finalValue ==
        // totalValueFTNS. Task 3 will track challenges and reduce finalValue.
        uint256 finalValue = b.totalValueFTNS - b.invalidatedValueFTNS;

        // Task 2: execute settlement via EscrowPool. A configured pool is
        // required once we've reached this code path — the contract is
        // meant to actually move value, not just emit events. Setting
        // finalValue=0 (pathological case where every receipt was
        // invalidated) skips the transfer but still finalizes state.
        if (finalValue > 0) {
            if (address(escrowPool) == address(0)) {
                revert EscrowPoolNotConfigured();
            }
            escrowPool.settleFromRequester(b.requester, b.provider, finalValue);
        }

        emit BatchFinalized(
            batchId,
            b.provider,
            finalValue,
            b.invalidatedValueFTNS,
            uint64(block.timestamp)
        );
    }

    // ── Challenges (Task 3) ───────────────────────────────────────

    /**
     * @notice Challenge a specific receipt inside a PENDING batch. Reason-
     *         code-specific verification determines whether the challenge
     *         succeeds; on success, the receipt's value is subtracted from
     *         the batch's final payable amount at finalizeBatch time.
     *
     * @param batchId the pending batch
     * @param leaf canonical ReceiptLeaf being challenged
     * @param merkleProof proof that keccak256(abi.encode(leaf)) is in
     *                    batches[batchId].merkleRoot
     * @param reason which check the contract performs; see ReasonCode
     * @param auxData additional data per-reason-code. See individual
     *                _handle* functions for encoding.
     */
    function challengeReceipt(
        bytes32 batchId,
        ReceiptLeaf calldata leaf,
        bytes32[] calldata merkleProof,
        ReasonCode reason,
        bytes calldata auxData
    ) external {
        Batch storage b = batches[batchId];
        if (b.status == BatchStatus.NONEXISTENT) revert BatchNotFound(batchId);
        if (b.status != BatchStatus.PENDING) {
            revert BatchNotPending(batchId, b.status);
        }
        uint256 elapsed = block.timestamp - b.commitTimestamp;
        if (elapsed >= challengeWindowSeconds) {
            // Cannot challenge after window elapses — finalize is
            // eligible.  Challengers must act within the window.
            revert ChallengeWindowElapsed(batchId);
        }

        bytes32 leafHash = _hashLeaf(leaf);
        if (invalidatedReceipts[batchId][leafHash]) {
            revert ReceiptAlreadyInvalidated(batchId, leafHash);
        }

        if (!MerkleProof.verify(merkleProof, b.merkleRoot, leafHash)) {
            revert InvalidMerkleProof(batchId, leafHash);
        }

        bool proven;
        if (reason == ReasonCode.DOUBLE_SPEND) {
            proven = _handleDoubleSpend(leaf, leafHash, auxData);
        } else if (reason == ReasonCode.INVALID_SIGNATURE) {
            proven = _handleInvalidSignature(leaf, auxData);
        } else if (reason == ReasonCode.NO_ESCROW) {
            proven = _handleNoEscrow(b);
        } else if (reason == ReasonCode.EXPIRED) {
            proven = _handleExpired(leaf);
        } else {
            // MALFORMED reserved — cannot prove on-chain in Task 3 scope.
            revert MalformedReasonNotImplemented();
        }

        if (!proven) revert ChallengeNotProven(reason);

        // Record + accumulate invalidated value.
        invalidatedReceipts[batchId][leafHash] = true;
        b.invalidatedValueFTNS += leaf.valueFtns;

        emit ReceiptChallenged(batchId, leafHash, msg.sender, reason, leaf.valueFtns);
    }

    /// @dev Compute the canonical keccak256 leaf hash for a ReceiptLeaf.
    function _hashLeaf(ReceiptLeaf calldata leaf) internal pure returns (bytes32) {
        return keccak256(abi.encode(leaf));
    }

    /**
     * @dev DOUBLE_SPEND: the same receipt was committed in a different
     *      batch. auxData layout:
     *        abi.encode(bytes32 conflictingBatchId, bytes32[] conflictingProof)
     *      Succeeds iff the receipt's leafHash is provable in both
     *      `this batch` (already verified by caller) and the conflicting
     *      batch.
     */
    function _handleDoubleSpend(
        ReceiptLeaf calldata leaf,
        bytes32 leafHash,
        bytes calldata auxData
    ) internal view returns (bool) {
        (bytes32 conflictingBatchId, bytes32[] memory conflictingProof) =
            abi.decode(auxData, (bytes32, bytes32[]));

        Batch storage other = batches[conflictingBatchId];
        if (other.status == BatchStatus.NONEXISTENT) {
            revert ConflictingBatchNotCommitted(conflictingBatchId);
        }
        // Conflicting batch may be PENDING, FINALIZED, or VOIDED — all
        // establish that the receipt was claimed there too.

        return MerkleProof.verify(conflictingProof, other.merkleRoot, leafHash);
    }

    /**
     * @dev INVALID_SIGNATURE: the receipt's signature doesn't verify under
     *      its declared pubkey. auxData layout:
     *        abi.encode(bytes signingMessage, bytes publicKey, bytes signature)
     *      Contract verifies:
     *        - keccak256(publicKey) == leaf.providerPubkeyHash
     *        - keccak256(signature) == leaf.signatureHash
     *        - ISignatureVerifier.verify returns FALSE
     *      If verify returns TRUE, the signature is genuinely valid →
     *      challenge is not proven (challenger wasted gas).
     */
    function _handleInvalidSignature(
        ReceiptLeaf calldata leaf,
        bytes calldata auxData
    ) internal view returns (bool) {
        if (address(signatureVerifier) == address(0)) {
            revert VerifierNotConfigured();
        }
        (bytes memory signingMessage, bytes memory publicKey, bytes memory signature) =
            abi.decode(auxData, (bytes, bytes, bytes));

        // Bind the submitted pubkey + signature to the leaf-committed hashes.
        if (keccak256(publicKey) != leaf.providerPubkeyHash) return false;
        if (keccak256(signature) != leaf.signatureHash) return false;

        bytes32 messageHash = keccak256(signingMessage);
        bool valid = signatureVerifier.verify(messageHash, signature, publicKey);
        return !valid; // challenge succeeds iff verification fails
    }

    /**
     * @dev NO_ESCROW: the batch's named requester attests (via msg.sender)
     *      that they did not authorize this receipt. No cryptographic
     *      proof of a negative is possible on-chain, so this reduces to
     *      an authorization check — only the batch.requester can invoke.
     *      Their signed transaction IS the attestation.
     */
    function _handleNoEscrow(Batch storage b) internal view returns (bool) {
        if (msg.sender != b.requester) {
            revert CallerNotRequester(msg.sender, b.requester);
        }
        return true;
    }

    /**
     * @dev EXPIRED: receipt older than settlementLookbackWindowSeconds.
     *      Pure time check against leaf.executedAtUnix.
     */
    function _handleExpired(ReceiptLeaf calldata leaf) internal view returns (bool) {
        uint256 age = block.timestamp - uint256(leaf.executedAtUnix);
        if (age <= settlementLookbackWindowSeconds) {
            revert ReceiptNotExpired(leaf.executedAtUnix, settlementLookbackWindowSeconds);
        }
        return true;
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
    /**
     * @notice Set the EscrowPool contract that executes settlement transfers.
     *         Owner-only. The pool must be deployed separately; this function
     *         registers its address with the registry. Setting to address(0)
     *         effectively disables finalization of non-zero-value batches.
     * @param newPool EscrowPool contract address
     */
    function setEscrowPool(address newPool) external onlyOwner {
        address old = address(escrowPool);
        escrowPool = IEscrowPool(newPool);
        emit EscrowPoolUpdated(old, newPool);
    }

    /**
     * @notice Set the pluggable Ed25519 verifier used by INVALID_SIGNATURE
     *         challenges. Owner-only. May be address(0) to effectively
     *         disable INVALID_SIGNATURE challenges (they revert with
     *         VerifierNotConfigured).
     */
    function setSignatureVerifier(address newVerifier) external onlyOwner {
        address old = address(signatureVerifier);
        signatureVerifier = ISignatureVerifier(newVerifier);
        emit SignatureVerifierUpdated(old, newVerifier);
    }

    /**
     * @notice Set the EXPIRED-receipt lookback window. Owner-only.
     *         Bounded [1 day, 365 days] to prevent pathological values.
     */
    function setSettlementLookbackWindow(uint256 newSeconds) external onlyOwner {
        if (newSeconds < MIN_LOOKBACK_SECONDS || newSeconds > MAX_LOOKBACK_SECONDS) {
            revert InvalidLookbackWindow(newSeconds);
        }
        uint256 old = settlementLookbackWindowSeconds;
        settlementLookbackWindowSeconds = newSeconds;
        emit SettlementLookbackUpdated(old, newSeconds);
    }

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
