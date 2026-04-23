// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import {ISignatureVerifier} from "./BatchSettlementRegistry.sol";
import {Ed25519Lib} from "./lib/Ed25519Lib.sol";

/**
 * @title Ed25519Verifier
 * @notice Production Ed25519 signature verifier for the INVALID_SIGNATURE
 *         challenge path of BatchSettlementRegistry.
 *
 * Wraps chengwenxi/Ed25519 (Apache-2.0) ported to Solidity 0.8.22. The
 * underlying library implements RFC 8032 §5.1.7 Ed25519 verify end-to-end,
 * including the SHA-512(R || A || M) hashing step (via in-tree Sha512.sol).
 *
 * Interface contract (ISignatureVerifier):
 *   verify(messageHash, signature, publicKey) -> bool
 *
 * Binding to Ed25519Lib.verify(k, r, s, m):
 *   k = publicKey       (32-byte Ed25519 public key A)
 *   r = signature[0:32] (first 32 bytes of 64-byte Ed25519 signature)
 *   s = signature[32:64](remaining 32 bytes)
 *   m = messageHash     (treated as the 32-byte message; providers sign
 *                        keccak256(canonical_receipt) via Ed25519)
 *
 * Gas: ~500K-2M per verify call depending on inputs. Only invoked on the
 * INVALID_SIGNATURE challenge path; happy-path commit + finalize paths
 * incur zero Ed25519 cost. See PRSM-PHASE3.1 §10.1 for the ADR.
 *
 * Security: pure-function verifier; no state. Malformed input sizes return
 * false rather than revert, matching the ISignatureVerifier contract.
 */
contract Ed25519Verifier is ISignatureVerifier {
    /// @dev Ed25519 public key + each signature scalar are 32 bytes.
    uint256 private constant PUBKEY_LEN = 32;
    /// @dev Ed25519 signature = R || s = 64 bytes total.
    uint256 private constant SIGNATURE_LEN = 64;

    /// @inheritdoc ISignatureVerifier
    function verify(
        bytes32 messageHash,
        bytes calldata signature,
        bytes calldata publicKey
    ) external pure override returns (bool) {
        if (signature.length != SIGNATURE_LEN) return false;
        if (publicKey.length != PUBKEY_LEN) return false;

        bytes32 k = bytes32(publicKey[0:32]);
        bytes32 r = bytes32(signature[0:32]);
        bytes32 s = bytes32(signature[32:64]);

        bytes memory m = new bytes(32);
        assembly {
            mstore(add(m, 32), messageHash)
        }

        return Ed25519Lib.verify(k, r, s, m);
    }
}
