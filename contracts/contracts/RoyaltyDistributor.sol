// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable2Step.sol";

interface IProvenanceRegistry {
    /// @dev Slim getter — returns only what the distributor needs. The
    ///      registry's full `contents()` getter would also return an
    ///      unbounded `string metadataUri`, which a squatter could use to
    ///      grief payment gas. We deliberately avoid that path here.
    function getCreatorAndRate(bytes32 contentHash)
        external
        view
        returns (address creator, uint16 royaltyRateBps);
}

/**
 * @title RoyaltyDistributor
 * @notice Three-way splitter for FTNS payments tied to registered content.
 * @dev Pulls FTNS from msg.sender via transferFrom and credits each
 *      recipient's `claimable` balance for later claim() — pull-payment
 *      pattern mirroring StakeBond.slashedBountyPayable + claimBounty.
 *
 *      L2 audit MEDIUM D-04 fix (this version, post-mainnet v1):
 *      Pre-fix, distributeRoyalty PUSHED FTNS to creator, networkTreasury,
 *      and servingNode in the same tx. A reverting recipient (e.g., a
 *      contract creator that reverts on ERC-20 receive, or a content
 *      ownership change mid-flight via ProvenanceRegistry's
 *      transferContentOwnership) would soft-brick the entire distribute
 *      call AND strand any in-flight share. Post-fix, distribute credits
 *      claimableBalance[recipient]; recipients pull via claim() at their
 *      convenience. A reverting/changed creator no longer brick-holds
 *      the entire payment, and the creator at distribute-time keeps
 *      their accrued share even if ownership transfers afterwards.
 *
 *      Splits:
 *        creator share = gross * royaltyRateBps / 10000  (from registry)
 *        network share = gross * NETWORK_FEE_BPS / 10000 (constant 2%)
 *        serving node share = gross - creator - network
 *      Reverts if creator + network share exceed gross
 *      (i.e. royaltyRateBps + 200 > 10000).
 *
 *      DEPLOYMENT NOTE: mainnet v1 (deployed 2026-05-04) ships the
 *      pre-fix push-payment behaviour. This pull-payment version is the
 *      v2 source-of-truth, intended to ship as part of the Week 2 v2
 *      re-deploy bundled with HIGH-1 (A-01 burn fix) + HIGH-3-deferred
 *      (Pausable). networkTreasury is immutable so a v2 deploy is
 *      required regardless of which fix lands first.
 */
contract RoyaltyDistributor is Ownable2Step, ReentrancyGuard {
    IERC20 public immutable ftns;
    IProvenanceRegistry public immutable registry;
    address public immutable networkTreasury;

    /// @dev 2% network fee, basis points
    uint16 public constant NETWORK_FEE_BPS = 200;

    /// @dev Per-recipient claimable balance accumulated across all
    /// distributeRoyalty calls. Zeroed on claim().
    mapping(address recipient => uint256) public claimable;

    /// @dev L4 self-audit A-08: sum of every entry in `claimable`. Maintained
    /// in lockstep with `claimable` so `recoverStranded` can compute the
    /// excess (donations + push-payments stuck in the contract) without
    /// iterating the mapping. Invariant: `ftns.balanceOf(address(this)) >=
    /// totalClaimable` at all times.
    uint256 public totalClaimable;

    event RoyaltyPaid(
        bytes32 indexed contentHash,
        address indexed payer,
        address indexed creator,
        address servingNode,
        uint256 creatorAmount,
        uint256 networkAmount,
        uint256 servingNodeAmount
    );
    event RoyaltyClaimed(address indexed recipient, uint256 amount);
    /// @dev L4 self-audit A-08
    event StrandedRecovered(address indexed to, uint256 amount);

    /// @dev L4 self-audit A-08 errors
    error ZeroAddress();
    error NothingToRecover();
    error TransferFailed();

    constructor(
        address _ftns,
        address _registry,
        address _networkTreasury,
        address _initialOwner
    ) Ownable(_initialOwner) {
        require(_ftns != address(0), "Zero ftns");
        require(_registry != address(0), "Zero registry");
        require(_networkTreasury != address(0), "Zero treasury");
        ftns = IERC20(_ftns);
        registry = IProvenanceRegistry(_registry);
        networkTreasury = _networkTreasury;
    }

    /**
     * @notice Pull `gross` FTNS from msg.sender and credit the 3-way split
     *         into per-recipient `claimable` balances. Recipients claim
     *         via claim() at their convenience.
     * @param contentHash content registered in ProvenanceRegistry
     * @param servingNode address of node that served the content
     * @param gross total FTNS amount being distributed (in token base units)
     */
    function distributeRoyalty(
        bytes32 contentHash,
        address servingNode,
        uint256 gross
    ) external nonReentrant {
        require(gross > 0, "Zero amount");
        require(servingNode != address(0), "Zero serving node");

        (address creator, uint16 rateBps) = registry.getCreatorAndRate(contentHash);
        require(creator != address(0), "Not registered");

        uint256 creatorAmt = (gross * rateBps) / 10000;
        uint256 networkAmt = (gross * NETWORK_FEE_BPS) / 10000;
        require(creatorAmt + networkAmt <= gross, "Rate plus fee exceeds 100%");
        uint256 nodeAmt = gross - creatorAmt - networkAmt;

        // Pull full amount once, then credit to claimable pools.
        // No outbound transfers — recipients pull via claim().
        require(ftns.transferFrom(msg.sender, address(this), gross), "Pull failed");

        if (creatorAmt > 0) {
            claimable[creator] += creatorAmt;
        }
        if (networkAmt > 0) {
            claimable[networkTreasury] += networkAmt;
        }
        if (nodeAmt > 0) {
            claimable[servingNode] += nodeAmt;
        }
        // L4 self-audit A-08: maintain totalClaimable in lockstep. The
        // sum equals exactly `gross` because creatorAmt+networkAmt+nodeAmt
        // = gross by construction (line 105 above), so balance and
        // totalClaimable both increase by the same amount.
        totalClaimable += gross;

        emit RoyaltyPaid(
            contentHash,
            msg.sender,
            creator,
            servingNode,
            creatorAmt,
            networkAmt,
            nodeAmt
        );
    }

    /**
     * @notice Claim accumulated royalty balance to the caller. Idempotent:
     *         a zero-balance call reverts rather than emitting noise.
     */
    function claim() external nonReentrant {
        uint256 amount = claimable[msg.sender];
        require(amount > 0, "Nothing to claim");
        // Effects before interaction.
        claimable[msg.sender] = 0;
        // L4 self-audit A-08: balance and totalClaimable drop equally.
        totalClaimable -= amount;
        require(ftns.transfer(msg.sender, amount), "Claim transfer failed");
        emit RoyaltyClaimed(msg.sender, amount);
    }

    /**
     * @notice Recover FTNS that has accumulated in this contract beyond the
     *         sum of all claimable balances — i.e. donations sent via direct
     *         `ftns.transfer(distributor, X)` that bypass `distributeRoyalty`
     *         and therefore cannot be claimed by anyone.
     *
     *         L4 self-audit A-08 fix. See docs/governance/A-08-recoverStranded-design.md.
     *
     *         Owner-only (Foundation Safe in production). Only sweeps the
     *         strict excess `balance(this) - totalClaimable`; recipients'
     *         claimable entitlements are untouched.
     *
     * @param to address to receive the recovered FTNS
     */
    function recoverStranded(address to) external onlyOwner nonReentrant {
        if (to == address(0)) revert ZeroAddress();
        uint256 bal = ftns.balanceOf(address(this));
        if (bal <= totalClaimable) revert NothingToRecover();
        uint256 excess = bal - totalClaimable;
        if (!ftns.transfer(to, excess)) revert TransferFailed();
        emit StrandedRecovered(to, excess);
    }

    /**
     * @notice Read-only preview of how `gross` would be split for `contentHash`.
     * @return creatorAmount FTNS that would be sent to the registered creator
     * @return networkAmount FTNS that would be sent to the network treasury
     * @return servingNodeAmount FTNS that would be sent to the serving node
     */
    function preview(bytes32 contentHash, uint256 gross)
        external
        view
        returns (
            uint256 creatorAmount,
            uint256 networkAmount,
            uint256 servingNodeAmount
        )
    {
        (address creator, uint16 rateBps) = registry.getCreatorAndRate(contentHash);
        require(creator != address(0), "Not registered");

        creatorAmount = (gross * rateBps) / 10000;
        networkAmount = (gross * NETWORK_FEE_BPS) / 10000;
        require(creatorAmount + networkAmount <= gross, "Rate plus fee exceeds 100%");
        servingNodeAmount = gross - creatorAmount - networkAmount;
    }
}
