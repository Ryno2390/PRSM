// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/**
 * @title MockStakeBondSlasher
 * @notice Minimal IStakeBondSlasher stub for StorageSlashing unit tests.
 *
 * Records every `slash` call so assertions can verify arguments without
 * wiring the full StakeBond.sol + FTNS token stack. Phase 7-storage
 * Task 8 exercises real-StakeBond integration end-to-end.
 */
contract MockStakeBondSlasher {
    struct SlashRecord {
        address provider;
        address challenger;
        bytes32 reasonId;
    }

    SlashRecord[] public slashes;
    bool public shouldRevert;

    function setShouldRevert(bool v) external {
        shouldRevert = v;
    }

    function slash(
        address provider,
        address challenger,
        bytes32 reasonId
    ) external {
        require(!shouldRevert, "MockStakeBondSlasher: forced revert");
        slashes.push(SlashRecord(provider, challenger, reasonId));
    }

    function slashCount() external view returns (uint256) {
        return slashes.length;
    }
}
