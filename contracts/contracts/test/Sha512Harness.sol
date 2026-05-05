// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import {Sha512} from "../lib/Sha512.sol";

/**
 * @title Sha512Harness
 * @notice Test-only wrapper exposing Sha512.hash for FIPS 180-4
 *         known-answer-test (KAT) validation in L3 pre-engagement.
 *
 * Sha512 is consumed only by Ed25519Lib in production. The lib has no
 * direct caller surface, so we expose .hash() here purely for the test
 * suite to verify FIPS 180-4 §7.4 conformance.
 *
 * Returns the 8 × uint64 internal state directly; the JS test packs
 * them to a 64-byte array for comparison against published KATs.
 */
contract Sha512Harness {
    function hashOf(bytes memory data) external pure returns (uint64[8] memory) {
        return Sha512.hash(data);
    }
}
