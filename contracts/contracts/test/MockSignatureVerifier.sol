// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import {ISignatureVerifier} from "../BatchSettlementRegistry.sol";

/// @notice Test-only signature verifier. Returns whatever the test
/// configured via setResult. Stands in for a real Ed25519 library during
/// Task 3 unit tests; production deploy replaces with an audited
/// on-chain Ed25519 implementation (per PRSM-PHASE3.1 §10.1).
contract MockSignatureVerifier is ISignatureVerifier {
    bool public result;

    function setResult(bool newResult) external {
        result = newResult;
    }

    function verify(
        bytes32, // messageHash (ignored in mock)
        bytes calldata, // signature (ignored)
        bytes calldata // publicKey (ignored)
    ) external view override returns (bool) {
        return result;
    }
}
