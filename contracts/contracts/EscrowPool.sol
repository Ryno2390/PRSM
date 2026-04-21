// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title EscrowPool
 * @notice Per-requester FTNS escrow for Phase 3.1 batched settlement.
 *
 * Requesters deposit FTNS into their per-address escrow balance via
 * `deposit(amount)` (with prior ERC-20 approval). The configured
 * BatchSettlementRegistry (§6.5 design) calls `settleFromRequester`
 * during batch finalization to transfer FTNS from the requester's
 * escrow balance directly to a provider's address. Requesters may
 * `withdraw` any unused balance at any time.
 *
 * Authorization model:
 *   - Anyone can deposit (to their own balance) or withdraw (from
 *     their own balance).
 *   - Only the registered `settlementRegistry` can call
 *     `settleFromRequester`. The registry is set once by the contract
 *     owner at deployment time (or updated via `setSettlementRegistry`
 *     with a published governance notice). The registry contract
 *     itself enforces its own access control on which batches can
 *     invoke settlement.
 *   - The owner (Foundation multi-sig in production) can update the
 *     registry address and the FTNS token address (the latter exists
 *     as a defensive upgrade escape hatch; normal operations hold the
 *     token immutable).
 *
 * ReentrancyGuard on deposit/withdraw/settle because the FTNS token
 * is an external contract — although the canonical FTNS implementation
 * is a standard ERC-20 without hooks, we can't guarantee the token
 * contract's behavior in every deployment context.
 */
contract EscrowPool is Ownable, ReentrancyGuard {
    IERC20 public ftns;
    address public settlementRegistry;

    /// @dev Per-requester escrow balance. Incremented on deposit,
    /// decremented on withdraw or settlement.
    mapping(address requester => uint256) public balances;

    event Deposited(address indexed requester, uint256 amount, uint256 newBalance);
    event Withdrawn(address indexed requester, uint256 amount, uint256 newBalance);
    event Settled(
        address indexed requester,
        address indexed recipient,
        uint256 amount,
        address indexed caller  // which registry invoked
    );
    event SettlementRegistryUpdated(address oldRegistry, address newRegistry);
    event FtnsTokenUpdated(address oldToken, address newToken);

    error ZeroAmount();
    error ZeroAddress();
    error InsufficientBalance(address requester, uint256 available, uint256 requested);
    error CallerNotRegistry(address caller, address expectedRegistry);
    error TransferFailed();

    constructor(
        address initialOwner,
        address ftnsAddress,
        address initialRegistry
    ) Ownable(initialOwner) {
        if (ftnsAddress == address(0)) revert ZeroAddress();
        ftns = IERC20(ftnsAddress);
        // initialRegistry may be address(0) at deploy time — registry
        // is often deployed after the pool so the owner updates it
        // via setSettlementRegistry once available.
        settlementRegistry = initialRegistry;
    }

    /**
     * @notice Deposit FTNS into the caller's escrow balance.
     *         Caller must have approved this contract for at least
     *         `amount` of FTNS before calling.
     * @param amount FTNS to deposit in token base units
     */
    function deposit(uint256 amount) external nonReentrant {
        if (amount == 0) revert ZeroAmount();

        // Record the credit before the external transferFrom to keep
        // the accounting correct even if the token's transferFrom
        // decides to call back into this contract (checks-effects-
        // interactions). The transfer cannot silently succeed with
        // zero tokens moved in a conforming ERC-20, but we check the
        // returned bool anyway.
        balances[msg.sender] += amount;

        bool ok = ftns.transferFrom(msg.sender, address(this), amount);
        if (!ok) revert TransferFailed();

        emit Deposited(msg.sender, amount, balances[msg.sender]);
    }

    /**
     * @notice Withdraw FTNS from the caller's own escrow balance back
     *         to the caller's address. Permissionless for the
     *         requester's own funds.
     * @param amount FTNS to withdraw in token base units
     */
    function withdraw(uint256 amount) external nonReentrant {
        if (amount == 0) revert ZeroAmount();
        uint256 bal = balances[msg.sender];
        if (bal < amount) {
            revert InsufficientBalance(msg.sender, bal, amount);
        }

        // Effects before interactions.
        balances[msg.sender] = bal - amount;

        bool ok = ftns.transfer(msg.sender, amount);
        if (!ok) revert TransferFailed();

        emit Withdrawn(msg.sender, amount, balances[msg.sender]);
    }

    /**
     * @notice Execute settlement from a requester's escrow balance to
     *         a recipient (typically a provider). Callable only by the
     *         registered BatchSettlementRegistry.
     * @param requester address whose balance funds the settlement
     * @param recipient address receiving the FTNS transfer
     * @param amount FTNS amount in token base units
     */
    function settleFromRequester(
        address requester,
        address recipient,
        uint256 amount
    ) external nonReentrant {
        if (msg.sender != settlementRegistry) {
            revert CallerNotRegistry(msg.sender, settlementRegistry);
        }
        if (amount == 0) revert ZeroAmount();
        if (recipient == address(0)) revert ZeroAddress();

        uint256 bal = balances[requester];
        if (bal < amount) {
            revert InsufficientBalance(requester, bal, amount);
        }

        // Effects before interactions.
        balances[requester] = bal - amount;

        bool ok = ftns.transfer(recipient, amount);
        if (!ok) revert TransferFailed();

        emit Settled(requester, recipient, amount, msg.sender);
    }

    // ── Governance surface ────────────────────────────────────────

    /**
     * @notice Update the settlement registry address. Owner-only.
     *         Should be paired with a PRSM-GOV-1 §10.3 14-day notice
     *         in production governance.
     * @param newRegistry address of the new BatchSettlementRegistry
     */
    function setSettlementRegistry(address newRegistry) external onlyOwner {
        address old = settlementRegistry;
        settlementRegistry = newRegistry;
        emit SettlementRegistryUpdated(old, newRegistry);
    }

    /**
     * @notice Update the FTNS token address. Owner-only. Defensive
     *         escape hatch — intended for use only if the FTNS token
     *         contract is ever replaced (e.g., emergency re-deploy).
     *         Cannot be called if there are non-zero balances pending
     *         in the pool, because old-token balances would be
     *         stranded — owner must drain first.
     *
     *         NOTE: we do NOT track total-balance-sum cheaply, so the
     *         "no pending balances" check is operational-policy only.
     *         Owner MUST verify via off-chain indexing before calling.
     *         Flagged for Task 3 review + potential on-chain
     *         balance-sum tracking.
     * @param newToken address of the new FTNS-compatible ERC-20
     */
    function setFtnsToken(address newToken) external onlyOwner {
        if (newToken == address(0)) revert ZeroAddress();
        address old = address(ftns);
        ftns = IERC20(newToken);
        emit FtnsTokenUpdated(old, newToken);
    }

    // ── Views ─────────────────────────────────────────────────────

    /// @notice Convenience view — alias for balances[requester].
    function balanceOf(address requester) external view returns (uint256) {
        return balances[requester];
    }
}
