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
    /// slash(). Default address(0) = slashing disabled. Owner updates
    /// via setSlasher. In production, the slasher is the
    /// BatchSettlementRegistry.
    address public slasher;

    /// @dev Per-provider stake record. Re-bonding after a full withdraw
    /// overwrites the prior record.
    mapping(address provider => Stake) public stakes;

    /// @dev Per-challenger claimable bounty balance. Accrues 70% of
    /// slashed FTNS when the challenger is NOT the slashed provider.
    /// Claimed via claimBounty(); zeroed on claim.
    mapping(address challenger => uint256) public slashedBountyPayable;

    /// @dev Accumulated Foundation-reserve balance from slashing
    /// (30% normal case; 100% on self-slash / anonymous challenger).
    /// Drained via drainFoundationReserve() by owner to the
    /// foundationReserveWallet address.
    uint256 public foundationReserveBalance;

    /// @dev Destination for drainFoundationReserve. Owner-set; must
    /// be non-zero before drain can succeed. Typically the Foundation's
    /// 2-of-3 multi-sig treasury wallet.
    address public foundationReserveWallet;

    /// @dev Bounty split in basis points. 7000 = 70% of slash amount
    /// goes to challenger; remainder (3000 = 30%) to Foundation reserve.
    /// When the challenger equals the provider or is address(0), the
    /// entire slash goes to the Foundation reserve (prevents self-slash
    /// schemes that would net-profit by capturing the challenger bounty).
    uint16 public constant CHALLENGER_BOUNTY_BPS = 7000;

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
    event FoundationReserveWalletUpdated(address oldWallet, address newWallet);
    event Slashed(
        address indexed provider,
        address indexed challenger,
        bytes32 indexed reasonId,
        uint256 slashAmount,
        uint256 challengerBounty,
        uint256 foundationShare
    );
    event BountyClaimed(address indexed challenger, uint256 amount);
    event FoundationReserveDrained(address indexed recipient, uint256 amount);

    error ZeroAmount();
    error ZeroAddress();
    error InvalidUnbondDelay(uint256 provided);
    error AlreadyBonded(address provider, StakeStatus status);
    error NotBonded(address provider, StakeStatus status);
    error NotUnbonding(address provider, StakeStatus status);
    error UnbondDelayNotElapsed(uint64 eligibleAt);
    error InvalidSlashRateBps(uint16 provided);
    error TransferFailed();
    error CallerNotSlasher(address caller, address expectedSlasher);
    error NotSlashable(address provider, StakeStatus status);
    error NothingToSlash(address provider);
    error NothingToClaim(address challenger);
    error FoundationReserveEmpty();
    error FoundationReserveWalletNotSet();

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

    /**
     * @notice Set the destination wallet for drainFoundationReserve.
     *         Owner-only. Must be non-zero before drain can succeed.
     */
    function setFoundationReserveWallet(address newWallet) external onlyOwner {
        address old = foundationReserveWallet;
        foundationReserveWallet = newWallet;
        emit FoundationReserveWalletUpdated(old, newWallet);
    }

    // ── Slashing (Task 2) ─────────────────────────────────────────

    /**
     * @notice Slash a provider's stake at their bonded slash rate.
     *         Slasher-only (the authorized BatchSettlementRegistry).
     *
     *         Amount slashed = current stake × tier_slash_rate_bps / 10000.
     *         Split:
     *           - If challenger ∈ {provider, address(0)}: 100% → Foundation
     *             reserve. Prevents self-slash schemes where a sophisticated
     *             adversary operates both the provider and the challenger
     *             to net-profit from the 70% bounty.
     *           - Otherwise: 70% → challenger's claimable bounty balance,
     *             30% → Foundation reserve.
     *
     *         Slashing is permitted in BOTH BonDED and UNBONDING states —
     *         closes the challenge-then-unbond race escape.
     *
     * @param provider  provider address whose stake is slashed
     * @param challenger address that submitted the successful challenge
     * @param reasonId   opaque identifier (typically the batchId that
     *                   contained the invalid receipt) for audit trail
     */
    function slash(
        address provider,
        address challenger,
        bytes32 reasonId
    ) external nonReentrant {
        if (msg.sender != slasher) {
            revert CallerNotSlasher(msg.sender, slasher);
        }
        Stake storage s = stakes[provider];
        if (s.status != StakeStatus.BONDED && s.status != StakeStatus.UNBONDING) {
            revert NotSlashable(provider, s.status);
        }
        if (s.amount == 0) revert NothingToSlash(provider);

        // slashAmount = amount * tier_slash_rate_bps / 10000, capped at amount.
        uint256 slashAmount = (uint256(s.amount) * uint256(s.tier_slash_rate_bps)) / 10000;
        if (slashAmount == 0) revert NothingToSlash(provider);
        if (slashAmount > s.amount) slashAmount = s.amount;  // defensive

        // Effects before interactions.
        s.amount -= uint128(slashAmount);

        // Split + credit to claimable pools. No ERC-20 transfer here —
        // FTNS stays in this contract until claimed via claimBounty /
        // drainFoundationReserve. Keeps the slash operation lightweight
        // and decouples the challenger's claim timing.
        uint256 challengerShare;
        uint256 foundationShare;
        if (challenger == provider || challenger == address(0)) {
            challengerShare = 0;
            foundationShare = slashAmount;
        } else {
            challengerShare = (slashAmount * uint256(CHALLENGER_BOUNTY_BPS)) / 10000;
            foundationShare = slashAmount - challengerShare;
            slashedBountyPayable[challenger] += challengerShare;
        }
        foundationReserveBalance += foundationShare;

        emit Slashed(
            provider, challenger, reasonId,
            slashAmount, challengerShare, foundationShare
        );
    }

    /**
     * @notice Claim accumulated bounty from successful challenges.
     *         Permissionless for the caller's own balance. Idempotent —
     *         a zero-balance claim reverts rather than emitting a
     *         noise event.
     */
    function claimBounty() external nonReentrant {
        uint256 amount = slashedBountyPayable[msg.sender];
        if (amount == 0) revert NothingToClaim(msg.sender);

        // Effects before interaction.
        slashedBountyPayable[msg.sender] = 0;

        bool ok = ftns.transfer(msg.sender, amount);
        if (!ok) revert TransferFailed();

        emit BountyClaimed(msg.sender, amount);
    }

    /**
     * @notice Move the accumulated Foundation-reserve balance to the
     *         configured foundationReserveWallet. Owner-only. Reverts
     *         if the wallet is unset or the reserve is empty.
     */
    function drainFoundationReserve() external onlyOwner nonReentrant {
        if (foundationReserveWallet == address(0)) {
            revert FoundationReserveWalletNotSet();
        }
        uint256 amount = foundationReserveBalance;
        if (amount == 0) revert FoundationReserveEmpty();

        // Effects before interaction.
        foundationReserveBalance = 0;

        bool ok = ftns.transfer(foundationReserveWallet, amount);
        if (!ok) revert TransferFailed();

        emit FoundationReserveDrained(foundationReserveWallet, amount);
    }
}
