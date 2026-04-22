// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title IRoyaltyPaymentVerifier
 * @notice Minimum surface `KeyDistribution` requires from a Phase 1.3-style
 *         royalty distributor — a view function that returns true iff the
 *         `payer` has settled at least `feeWei` against `contentHash`.
 *
 *         Phase 1.3 RoyaltyDistributor.sol adds this function per plan §5.4
 *         as a minimal contract change. For the duration of this unit test
 *         suite it is mocked via `MockRoyaltyVerifier`.
 */
interface IRoyaltyPaymentVerifier {
    function verifyPayment(
        address payer,
        bytes32 contentHash,
        uint256 feeWei
    ) external view returns (bool);
}

/**
 * @title KeyDistribution
 * @notice On-chain payment-gated release of content-encryption keys for
 *         Phase 7-storage Tier B / Tier C.
 *
 * Per docs/2026-04-22-phase7-storage-design-plan.md §4.3, §6 Task 6.
 *
 * Flow:
 *
 *   1. Publisher calls `depositKey(contentHash, encryptedKey, royalty,
 *      releaseFeeFtnsWei)` once per content. `encryptedKey` is the wire
 *      blob whose format is defined off-chain (Tier B: AES key encrypted
 *      under the consumer pubkey at release time off-chain, OR a
 *      publisher-encrypted envelope decrypted by the consumer after
 *      receipt; Tier C: one Shamir share envelope per holder contract).
 *      This contract treats `encryptedKey` as opaque bytes — verification
 *      is off-chain.
 *   2. Consumer pays `releaseFeeFtnsWei` FTNS through the configured
 *      `royalty` distributor. The distributor's `verifyPayment` returns
 *      true for that (payer, contentHash, fee) triple.
 *   3. Consumer (or anyone on their behalf) calls
 *      `release(contentHash, recipient)`. If the payment verifies, the
 *      `KeyReleased` event fires with the `encryptedKey` payload.
 *      Consumers listen for events matching their address and decrypt.
 *
 * Scope boundary — what this contract does NOT do:
 *
 *   * Does NOT hold FTNS. Payment flows through RoyaltyDistributor
 *     (Phase 1.3). KeyDistribution is a gate, not an escrow.
 *   * Does NOT revoke released keys retroactively — plan §8.5 is
 *     explicit: once a consumer has received the key via `release`, the
 *     publisher cannot un-release it. Publisher can stop releasing NEW
 *     keys by deauthorizing the content record.
 *   * Does NOT enforce Tier B vs Tier C distinction. Plan §2.3 splits
 *     the responsibilities: Tier B content publishes ONE KeyRecord with
 *     the full encrypted AES key; Tier C publishes ONE KeyRecord per
 *     Shamir share (M-of-N holders each operate their own share
 *     contract OR publish to this same contract with distinct
 *     `contentHash` derivatives).
 *   * Does NOT prevent duplicate releases to the same recipient —
 *     tracked via event log; idempotent on-chain (repeat release()
 *     calls by the same recipient emit additional events but cost
 *     nothing beyond gas).
 */
contract KeyDistribution is Ownable, ReentrancyGuard {
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    struct KeyRecord {
        address publisher;
        bytes encryptedKey;
        IRoyaltyPaymentVerifier royalty;
        uint256 releaseFeeFtnsWei;
        bool active;
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    mapping(bytes32 => KeyRecord) public records;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event KeyDeposited(
        bytes32 indexed contentHash,
        address indexed publisher,
        address indexed royalty,
        uint256 releaseFeeFtnsWei
    );
    event KeyReleased(
        bytes32 indexed contentHash,
        address indexed recipient,
        bytes encryptedKey
    );
    event KeyDeauthorized(
        bytes32 indexed contentHash,
        address indexed publisher
    );

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error InvalidAddress();
    error EmptyEncryptedKey();
    error KeyAlreadyDeposited(bytes32 contentHash);
    error KeyNotFound(bytes32 contentHash);
    error KeyNotActive(bytes32 contentHash);
    error NotPublisher(address caller, address publisher);
    error PaymentNotVerified(address recipient, bytes32 contentHash);
    error ZeroFee();

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(address _initialOwner) Ownable(_initialOwner) {}

    // -------------------------------------------------------------------------
    // depositKey
    // -------------------------------------------------------------------------

    /// @notice Publisher deposits the encrypted content-decryption key.
    ///
    /// @dev One deposit per `contentHash`. A second deposit for the same
    /// content reverts — the publisher must choose a distinct `contentHash`
    /// derivative for subsequent versions. Deposits are permanent (except
    /// for deactivation via `deauthorize`); there is no replace-in-place.
    function depositKey(
        bytes32 contentHash,
        bytes calldata encryptedKey,
        IRoyaltyPaymentVerifier royalty,
        uint256 releaseFeeFtnsWei
    ) external nonReentrant {
        if (encryptedKey.length == 0) revert EmptyEncryptedKey();
        if (address(royalty) == address(0)) revert InvalidAddress();
        if (releaseFeeFtnsWei == 0) revert ZeroFee();

        KeyRecord storage r = records[contentHash];
        if (r.publisher != address(0)) {
            revert KeyAlreadyDeposited(contentHash);
        }
        r.publisher = msg.sender;
        r.encryptedKey = encryptedKey;
        r.royalty = royalty;
        r.releaseFeeFtnsWei = releaseFeeFtnsWei;
        r.active = true;

        emit KeyDeposited(
            contentHash, msg.sender, address(royalty), releaseFeeFtnsWei
        );
    }

    // -------------------------------------------------------------------------
    // release
    // -------------------------------------------------------------------------

    /// @notice Release the deposited key to `recipient`. Gated on the
    ///         royalty distributor confirming `recipient` has paid the
    ///         release fee against `contentHash`.
    ///
    /// @dev Callable by anyone — allows a publisher or consumer agent to
    /// trigger release on behalf of the recipient. Payment verification
    /// binds to `recipient`, so third-party calls cannot leak keys to
    /// arbitrary addresses.
    function release(bytes32 contentHash, address recipient)
        external
        nonReentrant
    {
        if (recipient == address(0)) revert InvalidAddress();

        KeyRecord memory r = records[contentHash];
        if (r.publisher == address(0)) revert KeyNotFound(contentHash);
        if (!r.active) revert KeyNotActive(contentHash);

        if (!r.royalty.verifyPayment(
            recipient, contentHash, r.releaseFeeFtnsWei
        )) {
            revert PaymentNotVerified(recipient, contentHash);
        }

        emit KeyReleased(contentHash, recipient, r.encryptedKey);
    }

    // -------------------------------------------------------------------------
    // deauthorize
    // -------------------------------------------------------------------------

    /// @notice Publisher-only: deactivate future releases for this content.
    ///
    /// @dev Past releases CANNOT be revoked — consumers who already received
    /// the key via `KeyReleased` events keep it. Plan §8.5.
    function deauthorize(bytes32 contentHash) external nonReentrant {
        KeyRecord storage r = records[contentHash];
        if (r.publisher == address(0)) revert KeyNotFound(contentHash);
        if (msg.sender != r.publisher) {
            revert NotPublisher(msg.sender, r.publisher);
        }
        r.active = false;
        emit KeyDeauthorized(contentHash, msg.sender);
    }
}
