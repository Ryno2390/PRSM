// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @title MockUSDC — Sepolia testnet stand-in for Circle USDC
/// @notice 6-decimal ERC-20 with permissionless mint, intended ONLY for the
///         Aerodrome pool-seed Sepolia rehearsal (per
///         `docs/governance/2026-06-15-aerodrome-pool-seed-sepolia-rehearsal.md`).
///         Decimals match Circle's USDC convention (6 vs the ERC-20 default 18)
///         so the rehearsal exercises the same amount-encoding math the mainnet
///         ceremony uses (mainnet USDC: ftnsAmount in 10^18 wei, usdcAmount in
///         10^6 wei). A bug in our calldata builder that gets the decimals wrong
///         WILL surface on Sepolia with this contract; using a vanilla 18-decimal
///         MockERC20 instead would mask that class of bug.
/// @dev Anyone can mint. Not for mainnet under any circumstances. The
///      contract name + symbol carry "Mock" so operators inspecting via
///      Basescan can't confuse this with real USDC.
contract MockUSDC is ERC20 {
    constructor() ERC20("Mock USDC", "mUSDC") {}

    /// @notice ERC-20 decimals override — USDC convention.
    function decimals() public pure override returns (uint8) {
        return 6;
    }

    /// @notice Permissionless mint for test-only seeding. Mainnet USDC has
    ///         no public mint — this divergence is intentional for the
    ///         rehearsal use case (operator mints 500K mUSDC to the Sepolia
    ///         Safe before the ceremony, mirroring what the Prismatica wire
    ///         provides on mainnet).
    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}
