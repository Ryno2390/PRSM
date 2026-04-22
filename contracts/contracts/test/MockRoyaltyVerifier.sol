// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/**
 * @title MockRoyaltyVerifier
 * @notice Minimal IRoyaltyPaymentVerifier stub for KeyDistribution unit
 *         tests. Exposes a setPaid(payer, contentHash, feeWei, ok) toggle
 *         so tests can drive the paid/unpaid branches without spinning
 *         up a real RoyaltyDistributor.
 */
contract MockRoyaltyVerifier {
    mapping(bytes32 => bool) public paid;

    function _key(address payer, bytes32 contentHash, uint256 feeWei)
        internal
        pure
        returns (bytes32)
    {
        return keccak256(abi.encode(payer, contentHash, feeWei));
    }

    function setPaid(
        address payer,
        bytes32 contentHash,
        uint256 feeWei,
        bool ok
    ) external {
        paid[_key(payer, contentHash, feeWei)] = ok;
    }

    function verifyPayment(
        address payer,
        bytes32 contentHash,
        uint256 feeWei
    ) external view returns (bool) {
        return paid[_key(payer, contentHash, feeWei)];
    }
}
