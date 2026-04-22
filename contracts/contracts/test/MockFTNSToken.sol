// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/**
 * @title MockFTNSToken
 * @notice Minimal ERC20 for Phase 8 end-to-end unit tests. Exposes
 * `mintReward(to, amount)` to match the `IFTNSMinter` surface that
 * EmissionController calls into.
 *
 * No access control — tests are trusted. Real-token integration
 * (MINTER_ROLE gating, upgradeable proxy) is Phase 8 Task 3.
 */
contract MockFTNSToken is ERC20 {
    constructor() ERC20("Mock FTNS", "FTNS") {}

    function mintReward(address to, uint256 amount) external {
        _mint(to, amount);
    }
}
