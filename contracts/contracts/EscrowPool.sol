// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

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
 *     `settleFromRequester`. The registry is set ONCE in the constructor
 *     and is immutable thereafter (L2 audit HIGH-6 / B-CROSS-1 fix).
 *     The registry contract itself enforces its own access control on
 *     which batches can invoke settlement.
 *   - The owner (Foundation multi-sig in production) can update the
 *     FTNS token address as a defensive upgrade escape hatch (normal
 *     operations hold the token immutable).
 *
 * ReentrancyGuard on deposit/withdraw/settle because the FTNS token
 * is an external contract — although the canonical FTNS implementation
 * is a standard ERC-20 without hooks, we can't guarantee the token
 * contract's behavior in every deployment context.
 */
contract EscrowPool is Ownable, ReentrancyGuard, Pausable {
    IERC20 public ftns;

    /// @dev Settlement registry address. L2 audit HIGH-6 (B-CROSS-1)
    /// fix: this is now IMMUTABLE — set once at construction, no
    /// setter. Previously a compromised owner could re-point this to
    /// an attacker EOA and call settleFromRequester to drain every
    /// depositor's balance in 3 transactions. Closing this requires
    /// re-deploying the pool if the registry is ever rotated, which
    /// is the intended trade-off — the registry is meant to be a
    /// permanent trust anchor of the protocol.
    ///
    /// Deploy ordering implication: BatchSettlementRegistry must be
    /// deployed BEFORE EscrowPool. Registry's reciprocal pointer
    /// (its escrowPool field) is set after-the-fact via
    /// Registry.setEscrowPool — that direction remains mutable
    /// (separately scoped; see Registry.setEscrowPool semantics).
    address public immutable settlementRegistry;

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
        // initialRegistry may be address(0) at deploy time only when
        // running unit tests that don't exercise the settle path —
        // production deploys MUST set this to the registry's address
        // since there is no setter to fix it after-the-fact.
        ftns = IERC20(ftnsAddress);
        settlementRegistry = initialRegistry;
    }

    /**
     * @notice Deposit FTNS into the caller's escrow balance.
     *         Caller must have approved this contract for at least
     *         `amount` of FTNS before calling.
     * @param amount FTNS to deposit in token base units
     */
    function deposit(uint256 amount) external nonReentrant whenNotPaused {
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
    function withdraw(uint256 amount) external nonReentrant whenNotPaused {
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
    ) external nonReentrant whenNotPaused {
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

    // L2 audit HIGH-6 (B-CROSS-1) fix: setSettlementRegistry was
    // removed. The field is now immutable — set in the constructor.
    // Rotation requires re-deployment of EscrowPool; this is the
    // intended trade-off documented at the field declaration above.

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

    // ── Pause control (L2 audit HIGH-3 / D-02) ────────────────────

    /**
     * @notice Pause deposit / withdraw / settleFromRequester. Owner-only;
     *         intended for incident response per
     *         docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md. The remaining
     *         admin setter (setFtnsToken) stays accessible while paused
     *         so the owner can perform emergency token rotation.
     *         Note: settlementRegistry is immutable post-HIGH-6 — pool
     *         re-deploy is required for registry rotation.
     */
    function pause() external onlyOwner {
        _pause();
    }

    /// @notice Resume normal operations after pause.
    function unpause() external onlyOwner {
        _unpause();
    }

    // ── Views ─────────────────────────────────────────────────────

    /// @notice Convenience view — alias for balances[requester].
    function balanceOf(address requester) external view returns (uint256) {
        return balances[requester];
    }
}
