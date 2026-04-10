// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @dev Test-only ERC20. Anyone can mint. Not for production.
contract MockERC20 is ERC20 {
    constructor() ERC20("MockFTNS", "MFTNS") {}

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}
