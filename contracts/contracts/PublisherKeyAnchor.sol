// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/**
 * @title PublisherKeyAnchor
 * @notice On-chain registry mapping `node_id` → Ed25519 public key.
 *         Closes the cross-node trust-boundary caveat shared by Phase
 *         3.x.2 (model registry) and Phase 3.x.4 (privacy-budget journal):
 *         a verifier holding only a signed artifact can resolve the
 *         publisher's pubkey from this contract instead of trusting a
 *         local sidecar file.
 *
 * Trust upgrade per design plan §1.4:
 * - Operator swaps `node.pubkey` sidecar locally → defeated. Verifier's
 *   on-chain lookup returns the original pubkey; tampered sidecar is
 *   ignored.
 * - Adversary attempts to register a victim's `node_id` → defeated.
 *   This contract derives `node_id` on-chain via `sha256(publicKey)` and
 *   takes the first 16 bytes; the adversary can only register pubkeys
 *   they actually own (matching PRSM's NodeIdentity construction at
 *   `prsm/node/identity.py`).
 * - Compromised publisher key needs revocation → multisig
 *   `adminOverride(node_id, newKey)` is the v1 emergency path. Versioned
 *   key-rotation history is deferred to v2 once a real rotation use case
 *   exists.
 *
 * Why `bytes16` for node_id: Solidity strings are awkward and expensive.
 * `bytes16` is the raw first 16 bytes of `sha256(pubkey)` — same width
 * PRSM's `NodeIdentity` truncates to and then hex-encodes for human
 * display. Verifiers convert the artifact's hex `node_id` to `bytes16`
 * client-side before calling `lookup`.
 *
 * Why raw 32-byte pubkey instead of base64 string: storage efficiency
 * (32 vs ~44 bytes per entry × N publishers). Client base64-encodes after
 * lookup for the existing `verify_X(public_key_b64=...)` API.
 *
 * Gas: register ≈ 50k. lookup: free (view).
 */
contract PublisherKeyAnchor {
    /// @dev Multisig allowed to invoke `adminOverride`. Immutable —
    ///      changing the trust anchor requires a new contract deploy +
    ///      migration.
    address public immutable admin;

    /// @dev node_id (first 16 bytes of sha256(pubkey)) → Ed25519 pubkey.
    mapping(bytes16 => bytes) public publisherKeys;

    /// @dev node_id → block timestamp at registration (or override).
    mapping(bytes16 => uint64) public registeredAt;

    /// @dev Ed25519 public keys are exactly 32 bytes.
    uint256 private constant PUBKEY_LEN = 32;

    event PublisherRegistered(
        bytes16 indexed nodeId,
        bytes publicKey,
        uint64 timestamp
    );
    event PublisherOverridden(
        bytes16 indexed nodeId,
        bytes oldKey,
        bytes newKey,
        uint64 timestamp
    );

    error AdminIsZeroAddress();
    error InvalidPublicKeyLength(uint256 actual);
    error AlreadyRegistered(bytes16 nodeId);
    error NotAdmin(address caller);
    error NotRegistered(bytes16 nodeId);

    constructor(address _admin) {
        if (_admin == address(0)) revert AdminIsZeroAddress();
        admin = _admin;
    }

    /**
     * @notice Register a publisher's Ed25519 public key. Permissionless;
     *         the contract derives `node_id` from `sha256(publicKey)` so
     *         the caller cannot forge a registration for a key they don't
     *         own.
     * @dev Write-once per node_id. Use `adminOverride` for the
     *      emergency rotation path.
     */
    function register(bytes calldata publicKey) external {
        if (publicKey.length != PUBKEY_LEN) {
            revert InvalidPublicKeyLength(publicKey.length);
        }
        bytes16 nodeId = bytes16(sha256(publicKey));
        if (publisherKeys[nodeId].length != 0) revert AlreadyRegistered(nodeId);
        publisherKeys[nodeId] = publicKey;
        registeredAt[nodeId] = uint64(block.timestamp);
        emit PublisherRegistered(nodeId, publicKey, uint64(block.timestamp));
    }

    /**
     * @notice Multisig-only override of a previously-registered key.
     *         Used for compromised-key revocation. Bypasses the
     *         write-once invariant of `register`.
     * @dev Caller MUST be `admin`. The `nodeId` MUST already exist —
     *      this is an override, not a registration.
     */
    function adminOverride(bytes16 nodeId, bytes calldata newKey) external {
        if (msg.sender != admin) revert NotAdmin(msg.sender);
        if (publisherKeys[nodeId].length == 0) revert NotRegistered(nodeId);
        if (newKey.length != PUBKEY_LEN) {
            revert InvalidPublicKeyLength(newKey.length);
        }
        bytes memory oldKey = publisherKeys[nodeId];
        publisherKeys[nodeId] = newKey;
        registeredAt[nodeId] = uint64(block.timestamp);
        emit PublisherOverridden(nodeId, oldKey, newKey, uint64(block.timestamp));
    }

    /**
     * @notice View: return the registered pubkey for `nodeId`, or empty
     *         bytes if not registered.
     * @dev Free. Verifiers call this on every cross-node artifact
     *      verification (cached client-side with TTL).
     */
    function lookup(bytes16 nodeId) external view returns (bytes memory) {
        return publisherKeys[nodeId];
    }
}
