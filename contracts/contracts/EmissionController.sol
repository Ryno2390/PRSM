// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title EmissionController
 * @notice On-chain FTNS emission with immutable Bitcoin-style halving schedule.
 *
 * Phase 8 Task 1 per docs/2026-04-22-phase8-design-plan.md §3.2–§3.4, §4.1.
 *
 * Design goals:
 *  - The emission rate is a pure function of block.timestamp and immutable
 *    constructor parameters. No entity — not even the owner — can alter the
 *    emission curve after deploy. Governance can rotate the distributor and
 *    pause/resume minting, but cannot accelerate or decelerate emission.
 *  - The halving is implemented with a right-shift (rate_n = baseline >> n)
 *    so the rate is EXACTLY half each epoch, with no float rounding.
 *  - Per-call rate-limit: a single mint cannot exceed
 *    currentEpochRate() × (block.timestamp - lastMintTimestamp). A single
 *    mint after a long quiet period can therefore still release a large
 *    amount at once — per plan §8.2 this is accepted with the mitigation
 *    that distributor calls happen sufficiently often and a monitoring
 *    alert fires on call-gap > 7 days. Future plan revisions may tighten
 *    this via a trailing-window cap.
 *  - Hard cap of `mintCap` total emission (constructor-immutable). The
 *    underlying FTNSToken has its own MAX_SUPPLY separately; the
 *    EmissionController cap is the slice above the Foundation genesis
 *    allocation (typically 900M FTNS = 1B supply cap − 100M genesis).
 *
 * Scope boundary for Task 1:
 *  - CompensationDistributor.sol (pull + weighted split) is Task 2.
 *  - Token-side minter-role integration verification is Task 3.
 *  - Python EmissionClient / EmissionWatcher is Task 4.
 */
interface IFTNSMinter {
    function mintReward(address to, uint256 amount) external;
}

contract EmissionController is Ownable, ReentrancyGuard {
    // -------------------------------------------------------------------------
    // Immutable configuration
    // -------------------------------------------------------------------------

    /// @notice Address of the FTNSToken contract this controller mints through.
    IFTNSMinter public immutable ftnsToken;

    /// @notice Unix timestamp of epoch 0 start. Emission cannot occur before
    /// this time. Set at deploy to match the mainnet Phase 8 activation moment.
    uint64 public immutable epochZeroStartTimestamp;

    /// @notice Emission rate at epoch 0, in FTNS wei per second. Rate halves
    /// at each epoch boundary; the curve is fully determined by this value
    /// and the epoch duration.
    uint256 public immutable baselineRatePerSecond;

    /// @notice Total FTNS this controller is authorised to mint across its
    /// lifetime. Typically 900M FTNS (total supply cap − genesis allocation).
    uint256 public immutable mintCap;

    /// @notice Halving epoch length. 4 years in seconds, matching the
    /// PRSM_Tokenomics.md §4 commitment.
    uint256 public constant EPOCH_DURATION_SECONDS = 4 * 365 days;

    // -------------------------------------------------------------------------
    // Mutable state
    // -------------------------------------------------------------------------

    /// @notice Sole address authorised to call mintAuthorized. In production
    /// this is the CompensationDistributor (Task 2). Zero means no
    /// distributor is authorised — minting is effectively blocked.
    address public authorizedDistributor;

    /// @notice Cumulative FTNS minted through this controller. Monotone
    /// non-decreasing; bounded by mintCap.
    uint256 public mintedToDate;

    /// @notice Timestamp of the most recent mintAuthorized call.
    /// Initialised to epochZeroStartTimestamp so the first mint's elapsed-
    /// time window begins at epoch 0, not the deploy block.
    uint64 public lastMintTimestamp;

    /// @notice When true, mintAuthorized reverts. Owner-controlled emergency
    /// lever.
    bool public paused;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event Minted(
        address indexed recipient,
        uint256 amount,
        uint32 epoch,
        uint256 epochRate
    );
    event DistributorUpdated(
        address indexed oldDistributor,
        address indexed newDistributor
    );
    event MintingPaused(address indexed caller);
    event MintingResumed(address indexed caller);

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error NotDistributor();
    error MintingIsPaused();
    error NotPaused();
    error AlreadyPaused();
    error ExceedsRateLimit(uint256 requested, uint256 allowed);
    error ExceedsMintCap(uint256 requested, uint256 remaining);
    error InvalidAddress();
    error ZeroAmount();

    // -------------------------------------------------------------------------
    // Modifiers
    // -------------------------------------------------------------------------

    modifier onlyDistributor() {
        if (msg.sender != authorizedDistributor) revert NotDistributor();
        _;
    }

    modifier whenNotPaused() {
        if (paused) revert MintingIsPaused();
        _;
    }

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(
        address _ftnsToken,
        uint64 _epochZeroStartTimestamp,
        uint256 _baselineRatePerSecond,
        uint256 _mintCap,
        address _initialOwner
    ) Ownable(_initialOwner) {
        if (_ftnsToken == address(0)) revert InvalidAddress();
        if (_baselineRatePerSecond == 0) revert ZeroAmount();
        if (_mintCap == 0) revert ZeroAmount();

        ftnsToken = IFTNSMinter(_ftnsToken);
        epochZeroStartTimestamp = _epochZeroStartTimestamp;
        baselineRatePerSecond = _baselineRatePerSecond;
        mintCap = _mintCap;
        lastMintTimestamp = _epochZeroStartTimestamp;
    }

    // -------------------------------------------------------------------------
    // Public views
    // -------------------------------------------------------------------------

    /// @notice Current epoch index (0 during the first 4-year window).
    function currentEpoch() public view returns (uint32) {
        if (block.timestamp <= epochZeroStartTimestamp) return 0;
        return uint32(
            (block.timestamp - epochZeroStartTimestamp) / EPOCH_DURATION_SECONDS
        );
    }

    /// @notice Current emission rate in FTNS wei per second. Halves each
    /// epoch; saturates to 0 past epoch 255 (well past any practical horizon).
    function currentEpochRate() public view returns (uint256 ratePerSecond) {
        uint32 e = currentEpoch();
        if (e >= 256) return 0;
        return baselineRatePerSecond >> e;
    }

    /// @notice Seconds remaining until the next halving.
    function timeUntilNextHalving() public view returns (uint256) {
        uint32 e = currentEpoch();
        uint256 nextBoundary =
            uint256(epochZeroStartTimestamp) +
            (uint256(e) + 1) * EPOCH_DURATION_SECONDS;
        if (block.timestamp >= nextBoundary) return 0;
        return nextBoundary - block.timestamp;
    }

    // -------------------------------------------------------------------------
    // Mint
    // -------------------------------------------------------------------------

    /// @notice Mint FTNS to the authorised distributor, bounded by the
    /// current epoch rate × elapsed time and the lifetime cap.
    ///
    /// @dev Rate and cap checks are independent — rate governs throughput,
    /// cap governs lifetime emission. Both must pass.
    function mintAuthorized(uint256 amount)
        external
        nonReentrant
        onlyDistributor
        whenNotPaused
    {
        if (amount == 0) revert ZeroAmount();

        uint256 nowTs = block.timestamp;
        uint256 elapsed =
            nowTs > lastMintTimestamp ? nowTs - lastMintTimestamp : 0;
        uint256 allowed = currentEpochRate() * elapsed;
        if (amount > allowed) revert ExceedsRateLimit(amount, allowed);

        uint256 remaining = mintCap - mintedToDate;
        if (amount > remaining) revert ExceedsMintCap(amount, remaining);

        mintedToDate += amount;
        lastMintTimestamp = uint64(nowTs);

        ftnsToken.mintReward(authorizedDistributor, amount);

        emit Minted(
            authorizedDistributor,
            amount,
            currentEpoch(),
            currentEpochRate()
        );
    }

    // -------------------------------------------------------------------------
    // Governance
    // -------------------------------------------------------------------------

    /// @notice Set the sole address authorised to call mintAuthorized.
    /// Foundation multi-sig expected to hold ownership.
    function setAuthorizedDistributor(address newDistributor)
        external
        onlyOwner
    {
        if (newDistributor == address(0)) revert InvalidAddress();
        address old = authorizedDistributor;
        authorizedDistributor = newDistributor;
        emit DistributorUpdated(old, newDistributor);
    }

    /// @notice Emergency pause. Blocks all mintAuthorized calls. Does not
    /// rewind the mint history — resuming continues from lastMintTimestamp.
    function pauseMinting() external onlyOwner {
        if (paused) revert AlreadyPaused();
        paused = true;
        emit MintingPaused(msg.sender);
    }

    function resumeMinting() external onlyOwner {
        if (!paused) revert NotPaused();
        paused = false;
        emit MintingResumed(msg.sender);
    }
}
