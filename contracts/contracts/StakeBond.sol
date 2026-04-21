// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title StakeBond
 * @notice Per-provider FTNS collateral for Phase 7 Tier-C verification.
 *
 * Providers post FTNS stake to back their claimed tier. On-chain
 * balance determines effective tier; successful Phase 3.1 DOUBLE_SPEND
 * or INVALID_SIGNATURE challenges automatically slash the stake.
 *
 * Scope (Task 1): bond / requestUnbond / withdraw lifecycle + read-only
 * queries. Slashing + bounty accrual are Task 2. Registry wiring is
 * Task 3.
 *
 * Lifecycle:
 *   [none] →  bond(amount, tierSlashRateBps)  →  BONDED
 *   BONDED →  requestUnbond()                  →  UNBONDING
 *   UNBONDING (after unbond_delay) → withdraw() → WITHDRAWN
 *   BONDED OR UNBONDING → slash() (Task 2) → stake reduced
 *
 * Unbonding delay (default 7 days) prevents flash-stake attacks where
 * a provider bonds right before a dispatch, collects payment, then
 * unbonds and disappears before a challenge can fire.
 *
 * Slashing during UNBONDING is permitted: a provider who initiated
 * unbonding is still accountable for misbehavior caught within the
 * delay window. Prevents the challenge-then-unbond race escape.
 *
 * Slasher authorization: owner-settable. In production, the slasher
 * is the BatchSettlementRegistry's address. Until set, slashing is
 * disabled (challenges still invalidate receipts but don't touch stake).
 */
contract StakeBond is Ownable, ReentrancyGuard {
    enum StakeStatus {
        NONE,        // never bonded (or fully withdrawn + re-bonded happens via new Stake)
        BONDED,      // active stake backing a tier
        UNBONDING,   // requestUnbond fired; waiting for delay to elapse
        WITHDRAWN    // funds returned to provider
    }

    struct Stake {
        uint128 amount;               // FTNS wei currently locked
        uint64 bonded_at_unix;
        uint64 unbond_eligible_at;    // 0 while BONDED; set when requestUnbond fires
        StakeStatus status;
        uint16 tier_slash_rate_bps;   // snapshot at bond time; survives slashes
    }

    IERC20 public immutable ftns;

    /// @dev Seconds a provider must wait between requestUnbond() and
    /// withdraw(). Governance-adjustable within sane bounds.
    uint256 public unbondDelaySeconds;
    uint256 public constant MIN_UNBOND_DELAY_SECONDS = 1 days;
    uint256 public constant MAX_UNBOND_DELAY_SECONDS = 30 days;

    /// @dev Authorized slasher address. Only this contract can call
    /// slash() in Task 2. Default address(0) = slashing disabled.
    /// Owner updates via setSlasher. In production, the slasher is
    /// the BatchSettlementRegistry.
    address public slasher;

    /// @dev Per-provider stake record. Re-bonding after a full withdraw
    /// overwrites the prior record.
    mapping(address provider => Stake) public stakes;

    event Bonded(
        address indexed provider,
        uint128 amount,
        uint16 tierSlashRateBps,
        uint64 bondedAt
    );
    event UnbondRequested(
        address indexed provider,
        uint64 eligibleAt
    );
    event Withdrawn(
        address indexed provider,
        uint128 amount
    );
    event UnbondDelayUpdated(uint256 oldDelay, uint256 newDelay);
    event SlasherUpdated(address oldSlasher, address newSlasher);

    error ZeroAmount();
    error ZeroAddress();
    error InvalidUnbondDelay(uint256 provided);
    error AlreadyBonded(address provider, StakeStatus status);
    error NotBonded(address provider, StakeStatus status);
    error NotUnbonding(address provider, StakeStatus status);
    error UnbondDelayNotElapsed(uint64 eligibleAt);
    error InvalidSlashRateBps(uint16 provided);
    error TransferFailed();

    constructor(
        address initialOwner,
        address ftnsAddress,
        uint256 initialUnbondDelay
    ) Ownable(initialOwner) {
        if (ftnsAddress == address(0)) revert ZeroAddress();
        if (initialUnbondDelay < MIN_UNBOND_DELAY_SECONDS ||
            initialUnbondDelay > MAX_UNBOND_DELAY_SECONDS) {
            revert InvalidUnbondDelay(initialUnbondDelay);
        }
        ftns = IERC20(ftnsAddress);
        unbondDelaySeconds = initialUnbondDelay;
    }

    // ── Provider lifecycle ────────────────────────────────────────

    /**
     * @notice Post FTNS stake. Caller must have approved this contract
     *         for at least `amount` of FTNS. The `tierSlashRateBps`
     *         snapshot is stored at bond time and used on subsequent
     *         slashes — so a provider cannot dodge slashing by
     *         downgrading their claimed tier mid-flight.
     * @param amount FTNS in base units (wei)
     * @param tierSlashRateBps basis-points slash rate (0-10000). In
     *        Phase 7 tier mapping: standard=5000 (50%), premium=
     *        critical=10000 (100%). 0 = no-slash tier (== "open"
     *        tier, typically wouldn't bond).
     */
    function bond(uint128 amount, uint16 tierSlashRateBps) external nonReentrant {
        if (amount == 0) revert ZeroAmount();
        if (tierSlashRateBps > 10000) revert InvalidSlashRateBps(tierSlashRateBps);

        Stake storage s = stakes[msg.sender];
        if (s.status == StakeStatus.BONDED || s.status == StakeStatus.UNBONDING) {
            revert AlreadyBonded(msg.sender, s.status);
        }

        // Effects before interaction.
        stakes[msg.sender] = Stake({
            amount: amount,
            bonded_at_unix: uint64(block.timestamp),
            unbond_eligible_at: 0,
            status: StakeStatus.BONDED,
            tier_slash_rate_bps: tierSlashRateBps
        });

        bool ok = ftns.transferFrom(msg.sender, address(this), amount);
        if (!ok) revert TransferFailed();

        emit Bonded(
            msg.sender, amount, tierSlashRateBps,
            uint64(block.timestamp)
        );
    }

    /**
     * @notice Begin the unbonding process. Transitions BONDED →
     *         UNBONDING and starts the delay timer. During UNBONDING
     *         the provider's effectiveTier drops to "open" (view-
     *         layer concern, see effectiveTier).
     *         Slashing remains active during UNBONDING — a provider
     *         caught in a challenge after requesting unbond still
     *         forfeits stake.
     */
    function requestUnbond() external {
        Stake storage s = stakes[msg.sender];
        if (s.status != StakeStatus.BONDED) {
            revert NotBonded(msg.sender, s.status);
        }
        s.status = StakeStatus.UNBONDING;
        s.unbond_eligible_at = uint64(block.timestamp + unbondDelaySeconds);
        emit UnbondRequested(msg.sender, s.unbond_eligible_at);
    }

    /**
     * @notice After the unbonding delay has elapsed, withdraw the
     *         remaining staked FTNS to the provider's wallet. If the
     *         stake was partially slashed during UNBONDING, the
     *         current `amount` is what gets returned.
     */
    function withdraw() external nonReentrant {
        Stake storage s = stakes[msg.sender];
        if (s.status != StakeStatus.UNBONDING) {
            revert NotUnbonding(msg.sender, s.status);
        }
        if (block.timestamp < s.unbond_eligible_at) {
            revert UnbondDelayNotElapsed(s.unbond_eligible_at);
        }

        uint128 payout = s.amount;
        // Effects before interaction.
        s.amount = 0;
        s.status = StakeStatus.WITHDRAWN;

        if (payout > 0) {
            bool ok = ftns.transfer(msg.sender, payout);
            if (!ok) revert TransferFailed();
        }

        emit Withdrawn(msg.sender, payout);
    }

    // ── Views ─────────────────────────────────────────────────────

    /// @notice Return the full Stake record for a provider.
    function stakeOf(address provider) external view returns (Stake memory) {
        return stakes[provider];
    }

    /// @notice Return the provider's currently-effective tier as a
    ///         short string. Consumed by the marketplace orchestrator
    ///         when enforcing DispatchPolicy.min_stake_tier.
    ///         Returns "open" during UNBONDING or WITHDRAWN states
    ///         even if the on-chain amount is high — a provider
    ///         who requested unbond is not reliably backing a tier.
    function effectiveTier(address provider) external view returns (string memory) {
        Stake storage s = stakes[provider];
        if (s.status != StakeStatus.BONDED) return "open";
        uint128 amt = s.amount;
        // Thresholds per Phase 7 design §3.2.
        if (amt >= 50_000 * 1e18) return "critical";
        if (amt >= 25_000 * 1e18) return "premium";
        if (amt >= 5_000 * 1e18) return "standard";
        return "open";
    }

    // ── Governance surface ────────────────────────────────────────

    /**
     * @notice Update the unbonding delay. Owner-only. Does NOT retro-
     *         actively affect already-UNBONDING stakes (they retain
     *         their `unbond_eligible_at` from requestUnbond time).
     *         Per PRSM-GOV-1 §10.3, governance changes to unbond
     *         delay are subject to the 14-day advance-notice period.
     */
    function setUnbondDelay(uint256 newDelay) external onlyOwner {
        if (newDelay < MIN_UNBOND_DELAY_SECONDS ||
            newDelay > MAX_UNBOND_DELAY_SECONDS) {
            revert InvalidUnbondDelay(newDelay);
        }
        uint256 old = unbondDelaySeconds;
        unbondDelaySeconds = newDelay;
        emit UnbondDelayUpdated(old, newDelay);
    }

    /**
     * @notice Set the authorized slasher address. Owner-only. In
     *         production this is the BatchSettlementRegistry. Setting
     *         to address(0) disables slashing — tests or emergency
     *         governance pause.
     */
    function setSlasher(address newSlasher) external onlyOwner {
        address old = slasher;
        slasher = newSlasher;
        emit SlasherUpdated(old, newSlasher);
    }
}
