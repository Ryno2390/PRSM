/**
 * PRSM Contract Configuration
 *
 * Central registry of monitored contract addresses + relevant event signatures.
 * Per-network mapping; Sepolia addresses are pre-mainnet baseline, mainnet
 * addresses populate at Phase 1.3 deploy.
 *
 * Sources of truth:
 * - PRSM_Vision.md §11 (Sepolia deploy addresses)
 * - contracts/contracts/ (Solidity sources)
 */

export type NetworkChainId = 1 | 8453 | 11155111 | 84532;

export const NETWORKS: Record<string, NetworkChainId> = {
  ETH_MAINNET: 1,
  BASE_MAINNET: 8453,
  ETH_SEPOLIA: 11155111,
  BASE_SEPOLIA: 84532,
};

/**
 * Contract addresses per network. Sepolia values from PRSM_Vision.md §11.
 * Mainnet placeholders populate when Phase 1.3 deploy completes
 * (hardware-gated on multi-sig hot wallet provisioning).
 */
export const CONTRACT_ADDRESSES: Record<string, Record<NetworkChainId, string | null>> = {
  FTNSToken: {
    [NETWORKS.ETH_SEPOLIA]: "0xd979c096BE297F4C3a85175774Bc38C22b95E6a4",
    [NETWORKS.ETH_MAINNET]: null,
    [NETWORKS.BASE_SEPOLIA]: null,
    [NETWORKS.BASE_MAINNET]: null,
  },
  ProvenanceRegistry: {
    [NETWORKS.BASE_SEPOLIA]: "0x3744D1104c236f0Bd68473E35927587EB919198B",
    [NETWORKS.BASE_MAINNET]: null,
    [NETWORKS.ETH_SEPOLIA]: null,
    [NETWORKS.ETH_MAINNET]: null,
  },
  RoyaltyDistributor: {
    [NETWORKS.BASE_SEPOLIA]: "0x95F59fA1EDe8958407f7b003d2B089730109BD54",
    [NETWORKS.BASE_MAINNET]: null,
    [NETWORKS.ETH_SEPOLIA]: null,
    [NETWORKS.ETH_MAINNET]: null,
  },
  // Phase 3.1 contracts — populate Sepolia when deploy artifacts confirmed
  BatchSettlementRegistry: {
    [NETWORKS.BASE_SEPOLIA]: null,
    [NETWORKS.BASE_MAINNET]: null,
    [NETWORKS.ETH_SEPOLIA]: null,
    [NETWORKS.ETH_MAINNET]: null,
  },
  EscrowPool: {
    [NETWORKS.BASE_SEPOLIA]: null,
    [NETWORKS.BASE_MAINNET]: null,
    [NETWORKS.ETH_SEPOLIA]: null,
    [NETWORKS.ETH_MAINNET]: null,
  },
  // Phase 8 contracts — populate when Phase 8 deploys
  EmissionController: {
    [NETWORKS.BASE_SEPOLIA]: null,
    [NETWORKS.BASE_MAINNET]: null,
    [NETWORKS.ETH_SEPOLIA]: null,
    [NETWORKS.ETH_MAINNET]: null,
  },
  CompensationDistributor: {
    [NETWORKS.BASE_SEPOLIA]: null,
    [NETWORKS.BASE_MAINNET]: null,
    [NETWORKS.ETH_SEPOLIA]: null,
    [NETWORKS.ETH_MAINNET]: null,
  },
};

/**
 * Event signatures per contract. Used by detectors to filter transactions.
 * Standard ERC-20 events plus PRSM-specific custom events.
 */
export const EVENT_SIGNATURES = {
  // ERC-20 + OpenZeppelin AccessControl on FTNSToken
  ERC20_TRANSFER: "event Transfer(address indexed from, address indexed to, uint256 value)",
  ERC20_APPROVAL: "event Approval(address indexed owner, address indexed spender, uint256 value)",
  ROLE_GRANTED: "event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender)",
  ROLE_REVOKED: "event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender)",
  PAUSED: "event Paused(address account)",
  UNPAUSED: "event Unpaused(address account)",

  // ProvenanceRegistry custom events
  CONTENT_REGISTERED: "event ContentRegistered(bytes32 indexed contentHash, address indexed creator, uint256 royaltyRateBps)",
  OWNERSHIP_TRANSFERRED: "event ContentOwnershipTransferred(bytes32 indexed contentHash, address indexed previousOwner, address indexed newOwner)",
  ROYALTY_RATE_UPDATED: "event RoyaltyRateUpdated(bytes32 indexed contentHash, uint256 previousRate, uint256 newRate)",

  // RoyaltyDistributor custom events
  ROYALTY_PAID: "event RoyaltyPaid(bytes32 indexed contentHash, address indexed creator, address indexed servingNode, uint256 creatorShare, uint256 nodeShare, uint256 treasuryShare)",
  DISTRIBUTION_FAILED: "event DistributionFailed(bytes32 indexed contentHash, string reason)",

  // BatchSettlementRegistry custom events (Phase 3.1)
  BATCH_COMMITTED: "event BatchCommitted(bytes32 indexed batchRoot, address indexed submitter, uint256 receiptCount)",
  CHALLENGE_SUBMITTED: "event ChallengeSubmitted(bytes32 indexed batchRoot, bytes32 indexed receiptHash, address challenger)",
  CHALLENGE_RESOLVED: "event ChallengeResolved(bytes32 indexed batchRoot, bytes32 indexed receiptHash, bool challengerWon)",

  // EscrowPool custom events (Phase 3.1)
  ESCROW_CREATED: "event EscrowCreated(bytes32 indexed jobId, address indexed payer, uint256 amount)",
  ESCROW_RELEASED: "event EscrowReleased(bytes32 indexed jobId, address indexed provider, uint256 amount)",
  ESCROW_REFUNDED: "event EscrowRefunded(bytes32 indexed jobId, address indexed payer, uint256 amount, string reason)",

  // EmissionController custom events (Phase 8)
  HALVING_APPLIED: "event HalvingApplied(uint256 indexed epochId, uint256 newRateBps)",
  HALVING_PAUSED: "event HalvingPaused(uint256 indexed epochId, uint256 extensionSeconds, string rationale, bytes32 healthMetricCited)",
};

/**
 * Critical role identifiers (keccak256 of role name).
 * Used for AccessControl monitoring on FTNSToken + Phase 8 contracts.
 */
export const ROLES = {
  DEFAULT_ADMIN_ROLE: "0x0000000000000000000000000000000000000000000000000000000000000000",
  // keccak256("MINTER_ROLE")
  MINTER_ROLE: "0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6",
  // keccak256("BURNER_ROLE")
  BURNER_ROLE: "0x3c11d16cbaffd01df69ce1c404f6340ee057498f5f00246190ea54220576a848",
  // keccak256("PAUSER_ROLE")
  PAUSER_ROLE: "0x65d7a28e3265b37a6474929f336521b332c1681b933f6cb9f3376673440d862a",
};

/**
 * Tokenomics §5.1 + Q4 ratification — payment-distribution invariants.
 * Used by RoyaltyDistributor detector to verify split correctness.
 */
export const PAYMENT_SPLIT_BPS = {
  // Phase 3+ split (post-burn activation)
  BURN: 2000,        // 20%
  CREATOR: 640,      // 6.4%
  NODE: 7200,        // 72%
  TREASURY: 160,     // 1.6%
  TOTAL: 10000,
};

/**
 * Risk thresholds derived from Risk Register A1-A8 + Tokenomics scenarios.
 * Tunable via governance once mainnet operational data exists.
 */
export const THRESHOLDS = {
  // FTNSToken
  SUPPLY_CAP_FTNS: 1_000_000_000n * (10n ** 18n),         // 1B max supply
  SUPPLY_WARNING_PCT: 95,                                  // alert at 95% of cap
  ANOMALOUS_TRANSFER_VOLUME_PCT: 1000,                     // 10× rolling average

  // ProvenanceRegistry
  REGISTRATION_SPAM_PER_HOUR_PER_ADDRESS: 100,             // anti-spam
  HIGH_ROYALTY_RATE_BPS: 9000,                             // 90% — very high

  // RoyaltyDistributor
  DISTRIBUTION_FAILURE_RATE_PCT: 5,                        // alert if >5% fail
  ANOMALOUS_DISTRIBUTION_AMOUNT_PCT: 1000,                 // 10× rolling p99

  // BatchSettlementRegistry
  CHALLENGE_RATE_PER_HOUR: 50,                             // baseline
  CHALLENGE_WIN_RATE_PCT: 25,                              // alert if >25% win

  // EscrowPool
  STUCK_ESCROW_AGE_HOURS: 48,                              // alert if escrow >48h unreleased
  ESCROW_DRAIN_AMOUNT_PCT: 500,                            // 5× rolling p99

  // EmissionController
  HALVING_PAUSE_FREQUENCY_PER_EPOCH: 1,                    // alert if council pauses >1×/epoch
};

/**
 * Helper — get monitored contract addresses for a given chain.
 */
export function getMonitoredAddressesForChain(chainId: NetworkChainId): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [name, addresses] of Object.entries(CONTRACT_ADDRESSES)) {
    const addr = addresses[chainId];
    if (addr !== null) {
      result[name] = addr;
    }
  }
  return result;
}
