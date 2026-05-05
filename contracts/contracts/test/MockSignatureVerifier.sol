// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @notice Test-only signature verifier. Returns whatever the test
/// configured via setResult. Stands in for a real Ed25519 library during
/// Task 3 unit tests; production deploy replaces with an audited
/// on-chain Ed25519 implementation (per PRSM-PHASE3.1 §10.1).
///
/// L2 audit MEDIUM C-INT-02 fix: ISignatureVerifier is now declared
/// `pure`, but this togglable mock genuinely needs storage to flip
/// between "accept" and "reject" outcomes per test scenario. Rather
/// than weaken the production interface, this mock deliberately does
/// NOT inherit `is ISignatureVerifier` — it just exposes the same
/// `verify(bytes32, bytes, bytes) returns (bool)` selector, which the
/// registry's `ISignatureVerifier(address)` cast resolves at the EVM
/// level (selectors match; `pure`/`view` is a Solidity-side
/// classification, not enforced by the EVM during external calls).
/// Production verifiers (Ed25519Verifier) remain truly pure.
contract MockSignatureVerifier {
    bool public result;

    function setResult(bool newResult) external {
        result = newResult;
    }

    function verify(
        bytes32, // messageHash (ignored in mock)
        bytes calldata, // signature (ignored)
        bytes calldata // publicKey (ignored)
    ) external view returns (bool) {
        return result;
    }
}
