// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import {Ed25519Lib} from "../lib/Ed25519Lib.sol";

/**
 * @title Ed25519LibHarness
 * @notice Test-only thin wrapper exposing Ed25519Lib.verify for variable-length
 *         messages. Used by L3 pre-engagement test-vector validation
 *         (RFC 8032 §7.1).
 *
 * The production Ed25519Verifier wraps the lib for a 32-byte messageHash
 * interface; this harness exposes the lib's full variable-length-message
 * surface so we can run RFC 8032 vectors directly against the underlying
 * crypto math.
 *
 * Not deployed to mainnet. Lives under contracts/contracts/test/ for the
 * existing test-mock convention.
 */
contract Ed25519LibHarness {
    function verify(
        bytes32 k,
        bytes32 r,
        bytes32 s,
        bytes memory m
    ) external pure returns (bool) {
        return Ed25519Lib.verify(k, r, s, m);
    }
}
