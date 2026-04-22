// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/**
 * @title MockFTNSMinter
 * @notice Minimal IFTNSMinter stub for EmissionController unit tests.
 *
 * Records every mintReward call so assertions can verify recipient + amount
 * without wiring the full FTNSToken AccessControl / Upgradeable stack.
 * Task 3 of Phase 8 covers real-token integration.
 */
contract MockFTNSMinter {
    uint256 public totalMinted;
    address public lastRecipient;
    uint256 public lastAmount;
    uint256 public mintCallCount;

    // Optional toggle: make mintReward revert, simulating a token-side
    // authorization failure.
    bool public shouldRevert;

    function setShouldRevert(bool v) external {
        shouldRevert = v;
    }

    function mintReward(address to, uint256 amount) external {
        require(!shouldRevert, "MockFTNSMinter: forced revert");
        totalMinted += amount;
        lastRecipient = to;
        lastAmount = amount;
        mintCallCount += 1;
    }
}
