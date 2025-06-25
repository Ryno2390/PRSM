/**
 * TypeScript type definitions for PRSM SDK
 * Comprehensive type definitions for the Protocol for Recursive Scientific Modeling
 */

// ============================================================================
// ENUMS
// ============================================================================

export enum ModelProvider {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  HUGGINGFACE = 'huggingface',
  OLLAMA = 'ollama',
  OPENROUTER = 'openrouter',
  LOCAL = 'local',
  PRSM_DISTILLED = 'prsm_distilled'
}

export enum SafetyLevel {
  NONE = 'none',
  LOW = 'low',
  MODERATE = 'moderate',
  HIGH = 'high',
  CRITICAL = 'critical',
  EMERGENCY = 'emergency'
}

export enum SessionStatus {
  QUEUED = 'queued',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  PAUSED = 'paused'
}

export enum ModelCategory {
  LANGUAGE = 'language',
  VISION = 'vision',
  AUDIO = 'audio',
  MULTIMODAL = 'multimodal',
  SCIENTIFIC = 'scientific',
  SPECIALIZED = 'specialized'
}

export enum NetworkType {
  MAINNET = 'mainnet',
  TESTNET = 'testnet',
  LOCAL = 'local'
}

export enum ProposalStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  PASSED = 'passed',
  REJECTED = 'rejected',
  EXECUTED = 'executed',
  CANCELLED = 'cancelled'
}

export enum UserRole {
  USER = 'user',
  RESEARCHER = 'researcher',
  DEVELOPER = 'developer',
  VALIDATOR = 'validator',
  ADMIN = 'admin',
  GOVERNANCE = 'governance'
}

// ============================================================================
// CLIENT CONFIGURATION
// ============================================================================

export interface PRSMClientConfig {
  /** API key for authentication */
  apiKey?: string;
  /** Base URL for the PRSM API */
  baseUrl?: string;
  /** WebSocket URL for real-time features */
  websocketUrl?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Maximum number of retry attempts */
  maxRetries?: number;
  /** Custom headers to include with requests */
  headers?: Record<string, string>;
  /** Enable debug logging */
  debug?: boolean;
  /** Custom user agent */
  userAgent?: string;
  /** Circuit breaker configuration */
  circuitBreaker?: {
    enabled: boolean;
    failureThreshold: number;
    resetTimeout: number;
  };
  /** Rate limiting configuration */
  rateLimit?: {
    enabled: boolean;
    requestsPerSecond: number;
    burstSize: number;
  };
}

// ============================================================================
// CORE API TYPES
// ============================================================================

export interface QueryRequest {
  /** The research query or prompt */
  query: string;
  /** Specific model ID to use */
  modelId?: string;
  /** Research domain (e.g., 'environmental_science', 'biochemistry') */
  domain?: string;
  /** Analysis methodology */
  methodology?: 'comprehensive_analysis' | 'quick_analysis' | 'deep_research';
  /** Maximum number of iterations */
  maxIterations?: number;
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Temperature for randomness (0.0-2.0) */
  temperature?: number;
  /** System prompt to guide behavior */
  systemPrompt?: string;
  /** Additional context data */
  context?: Record<string, any>;
  /** Tools to make available during execution */
  tools?: string[];
  /** Safety level for content filtering */
  safetyLevel?: SafetyLevel;
  /** Include citations in response */
  includeCitations?: boolean;
  /** SEAL enhancement configuration */
  sealEnhancement?: SEALConfig;
  /** Streaming response configuration */
  stream?: boolean;
}

export interface SEALConfig {
  /** Enable SEAL autonomous improvement */
  enabled: boolean;
  /** Allow autonomous self-improvement */
  autonomousImprovement: boolean;
  /** Target learning gain (0.0-1.0) */
  targetLearningGain: number;
  /** Enable RESTEM methodology */
  restemMethodology: boolean;
  /** Maximum improvement iterations */
  maxImprovementIterations?: number;
}

export interface PRSMResponse {
  /** Generated content/response */
  content: string;
  /** Model ID that generated the response */
  modelId: string;
  /** Model provider */
  provider: ModelProvider;
  /** Execution time in milliseconds */
  executionTime: number;
  /** Token usage statistics */
  tokenUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  /** Cost in FTNS tokens */
  ftnsCost: number;
  /** Reasoning trace if available */
  reasoningTrace?: string[];
  /** Safety assessment result */
  safetyStatus: SafetyLevel;
  /** Additional metadata */
  metadata: Record<string, any>;
  /** Unique request identifier */
  requestId: string;
  /** Response timestamp */
  timestamp: string;
  /** Citations if requested */
  citations?: Citation[];
  /** SEAL performance metrics */
  sealPerformance?: SEALPerformance;
}

export interface Citation {
  title: string;
  authors: string[];
  journal?: string;
  year: number;
  doi?: string;
  url?: string;
  confidence: number;
}

export interface SEALPerformance {
  /** Number of autonomous improvements applied */
  autonomousImprovementsApplied: number;
  /** Learning gain achieved */
  learningGainAchieved: number;
  /** Knowledge incorporation improvement */
  knowledgeIncorporationImprovement: number;
  /** Self-edit examples generated */
  selfEditExamplesGenerated: number;
  /** RESTEM policy updates */
  restemPolicyUpdates: number;
}

// ============================================================================
// SESSION MANAGEMENT
// ============================================================================

export interface SessionInfo {
  /** Unique session identifier */
  sessionId: string;
  /** Current session status */
  status: SessionStatus;
  /** Progress percentage (0-100) */
  progress: number;
  /** Original query */
  query: string;
  /** Estimated completion time */
  estimatedCompletion?: string;
  /** Cost estimate */
  costEstimate: {
    ftnsTokens: number;
    usdEquivalent: number;
  };
  /** Actual cost (when completed) */
  costActual?: {
    ftnsTokens: number;
    usdEquivalent: number;
  };
  /** Session results (when completed) */
  results?: SessionResults;
  /** SEAL status for this session */
  sealStatus?: SEALSessionStatus;
  /** Session creation timestamp */
  createdAt: string;
  /** Session completion timestamp */
  completedAt?: string;
}

export interface SessionResults {
  /** Main summary/result */
  summary: string;
  /** Key findings */
  keyFindings: string[];
  /** Citations */
  citations: Citation[];
  /** Confidence score (0.0-1.0) */
  confidenceScore: number;
  /** Additional data/artifacts */
  artifacts?: Record<string, any>;
}

export interface SEALSessionStatus {
  /** SEAL enhancement enabled */
  enhancementEnabled: boolean;
  /** Autonomous improvement active */
  autonomousImprovementActive: boolean;
  /** Estimated learning gain */
  estimatedLearningGain: number;
  /** Self-edit generation rate */
  selfEditGenerationRate: number;
}

// ============================================================================
// FINANCIAL & TOKENS
// ============================================================================

export interface FTNSBalance {
  /** Total FTNS balance */
  totalBalance: number;
  /** Available balance for spending */
  availableBalance: number;
  /** Reserved/locked balance */
  reservedBalance: number;
  /** Staked tokens */
  stakedBalance?: number;
  /** Tokens earned today */
  earnedToday: number;
  /** Tokens spent today */
  spentToday: number;
  /** Wallet address */
  walletAddress?: string;
  /** Network type */
  network?: NetworkType;
  /** Last balance update */
  lastUpdated: string;
}

export interface TransactionInfo {
  /** Transaction hash */
  txHash: string;
  /** Transaction type */
  type: 'transfer' | 'purchase' | 'earn' | 'spend' | 'stake' | 'unstake';
  /** Amount in FTNS */
  amount: number;
  /** USD equivalent at time of transaction */
  usdEquivalent?: number;
  /** From address */
  from?: string;
  /** To address */
  to?: string;
  /** Transaction status */
  status: 'pending' | 'confirmed' | 'failed';
  /** Block number */
  blockNumber?: number;
  /** Transaction timestamp */
  timestamp: string;
  /** Gas used */
  gasUsed?: number;
  /** Transaction note/memo */
  note?: string;
}

export interface PaymentRequest {
  /** Amount in USD */
  amountUsd: number;
  /** Payment method */
  paymentMethod: 'stripe' | 'paypal' | 'crypto';
  /** Payment token from provider */
  paymentToken: string;
  /** Optional billing info */
  billingInfo?: BillingInfo;
}

export interface BillingInfo {
  name: string;
  email: string;
  address?: {
    street: string;
    city: string;
    state: string;
    postalCode: string;
    country: string;
  };
}

// ============================================================================
// MODELS & MARKETPLACE
// ============================================================================

export interface ModelInfo {
  /** Unique model identifier */
  id: string;
  /** Human-readable model name */
  name: string;
  /** Model provider */
  provider: ModelProvider;
  /** Model category */
  category: ModelCategory;
  /** Model description */
  description: string;
  /** Model capabilities */
  capabilities: string[];
  /** Cost per input token */
  costPerInputToken: number;
  /** Cost per output token */
  costPerOutputToken: number;
  /** Maximum tokens supported */
  maxTokens: number;
  /** Context window size */
  contextWindow: number;
  /** Whether model is currently available */
  isAvailable: boolean;
  /** Performance rating (0.0-5.0) */
  performanceRating: number;
  /** Safety rating (0.0-5.0) */
  safetyRating: number;
  /** Model creation timestamp */
  createdAt: string;
  /** Model version */
  version?: string;
  /** Model size information */
  size?: {
    parameters: string;
    diskSize: string;
  };
  /** Supported languages */
  languages?: string[];
  /** Fine-tuning support */
  fineTuningSupported?: boolean;
}

export interface MarketplaceQuery {
  /** Search query */
  query?: string;
  /** Filter by provider */
  provider?: ModelProvider;
  /** Filter by category */
  category?: ModelCategory;
  /** Maximum cost per token */
  maxCost?: number;
  /** Minimum performance rating */
  minPerformance?: number;
  /** Required capabilities */
  capabilities?: string[];
  /** Result limit */
  limit?: number;
  /** Result offset for pagination */
  offset?: number;
  /** Sort criteria */
  sortBy?: 'rating' | 'cost' | 'created' | 'popular';
  /** Sort direction */
  sortOrder?: 'asc' | 'desc';
  /** Only show featured models */
  featured?: boolean;
}

export interface MarketplaceModel {
  /** Model information */
  model: ModelInfo;
  /** Creator information */
  creator: {
    username: string;
    verified: boolean;
    reputation: number;
  };
  /** Pricing information */
  pricing: {
    ftnsPerRequest: number;
    bulkDiscount?: number;
    subscriptionOptions?: SubscriptionOption[];
  };
  /** Usage statistics */
  stats: {
    downloads: number;
    rating: number;
    reviews: number;
    uptime: number;
  };
  /** Model tags */
  tags: string[];
  /** Whether model is featured */
  featured: boolean;
  /** Model listing timestamp */
  listedAt: string;
}

export interface SubscriptionOption {
  /** Subscription type */
  type: 'hourly' | 'daily' | 'weekly' | 'monthly';
  /** Price in FTNS */
  price: number;
  /** Included requests */
  includedRequests: number;
  /** Overage rate */
  overageRate: number;
}

// ============================================================================
// TOOLS & EXECUTION
// ============================================================================

export interface ToolSpec {
  /** Tool name/identifier */
  name: string;
  /** Tool description */
  description: string;
  /** Tool parameters schema */
  parameters: Record<string, any>;
  /** Cost per execution in FTNS */
  costPerExecution: number;
  /** Required safety level */
  safetyLevel: SafetyLevel;
  /** Tool provider */
  provider: string;
  /** Tool version */
  version: string;
  /** Tool category */
  category?: string;
  /** Execution timeout in seconds */
  timeout?: number;
  /** Required permissions */
  requiredPermissions?: string[];
}

export interface ToolExecutionRequest {
  /** Tool name to execute */
  toolName: string;
  /** Execution parameters */
  parameters: Record<string, any>;
  /** Execution context */
  context?: Record<string, any>;
  /** Safety level override */
  safetyLevel?: SafetyLevel;
  /** Execution timeout override */
  timeout?: number;
  /** Async execution flag */
  async?: boolean;
}

export interface ToolExecutionResponse {
  /** Execution result */
  result: any;
  /** Execution time in milliseconds */
  executionTime: number;
  /** Cost in FTNS tokens */
  ftnsCost: number;
  /** Safety assessment */
  safetyStatus: SafetyLevel;
  /** Execution success flag */
  success: boolean;
  /** Error message if failed */
  error?: string;
  /** Additional metadata */
  metadata: Record<string, any>;
  /** Execution ID for tracking */
  executionId: string;
  /** Execution timestamp */
  timestamp: string;
}

// ============================================================================
// GOVERNANCE
// ============================================================================

export interface Proposal {
  /** Proposal ID */
  id: string;
  /** Proposal title */
  title: string;
  /** Proposal description */
  description: string;
  /** Proposal creator */
  proposer: string;
  /** Current status */
  status: ProposalStatus;
  /** Proposal category */
  category: 'technical' | 'funding' | 'governance' | 'community';
  /** Voting end time */
  votingEnds: string;
  /** Vote counts */
  votes: {
    yes: number;
    no: number;
    abstain: number;
  };
  /** Required quorum */
  quorumRequired: number;
  /** Passing threshold */
  threshold: number;
  /** Implementation plan */
  implementationPlan?: string;
  /** Budget required */
  budgetRequired?: number;
  /** Proposal creation time */
  createdAt: string;
}

export interface VoteRequest {
  /** Vote choice */
  vote: 'yes' | 'no' | 'abstain';
  /** Voting power to use */
  votingPower: number;
  /** Optional comment */
  comment?: string;
}

// ============================================================================
// USER MANAGEMENT
// ============================================================================

export interface UserProfile {
  /** User ID */
  id: string;
  /** Username */
  username: string;
  /** Email address */
  email: string;
  /** User role */
  role: UserRole;
  /** Organization */
  organization?: string;
  /** User bio */
  bio?: string;
  /** Research interests */
  researchInterests?: string[];
  /** Account creation date */
  createdAt: string;
  /** Last login */
  lastLogin?: string;
  /** Current FTNS balance */
  ftnsBalance: number;
  /** Research credits */
  researchCredits?: number;
  /** User permissions */
  permissions: string[];
  /** Account verification status */
  verified: boolean;
  /** Two-factor authentication enabled */
  twoFactorEnabled: boolean;
}

export interface AuthTokens {
  /** Access token */
  accessToken: string;
  /** Token type (usually 'bearer') */
  tokenType: string;
  /** Token expiration time in seconds */
  expiresIn: number;
  /** Refresh token */
  refreshToken?: string;
}

export interface LoginRequest {
  /** Username or email */
  username: string;
  /** Password */
  password: string;
  /** Two-factor authentication code */
  twoFactorCode?: string;
}

export interface RegisterRequest {
  /** Desired username */
  username: string;
  /** Email address */
  email: string;
  /** Password */
  password: string;
  /** User role */
  role?: UserRole;
  /** Organization */
  organization?: string;
  /** Invitation code (if required) */
  invitationCode?: string;
}

// ============================================================================
// SECURITY & MONITORING
// ============================================================================

export interface SecurityStatus {
  /** Overall security level */
  securityLevel: 'normal' | 'elevated' | 'high' | 'critical';
  /** Number of threats detected */
  threatsDetected: number;
  /** Last security scan */
  lastScan: string;
  /** Account status */
  accountStatus: 'active' | 'suspended' | 'locked' | 'verified';
  /** Recent activity summary */
  recentActivity: {
    loginAttempts: number;
    successfulLogins: number;
    apiCalls: number;
    suspiciousActivity: boolean;
  };
}

export interface SafetyStatus {
  /** Overall safety level */
  overallStatus: SafetyLevel;
  /** Number of active monitors */
  activeMonitors: number;
  /** Threats detected */
  threatsDetected: number;
  /** Circuit breakers triggered */
  circuitBreakersTriggered: number;
  /** Last safety assessment */
  lastAssessment: string;
  /** Network health score (0.0-1.0) */
  networkHealth: number;
}

export interface HealthStatus {
  /** Overall system status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** System version */
  version: string;
  /** Uptime in seconds */
  uptimeSeconds: number;
  /** Environment */
  environment: 'development' | 'staging' | 'production';
  /** Health check timestamp */
  timestamp: string;
  /** Component health details */
  components: {
    database: ComponentHealth;
    redis: ComponentHealth;
    ipfs: ComponentHealth;
    [key: string]: ComponentHealth;
  };
}

export interface ComponentHealth {
  /** Component status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Response time in milliseconds */
  responseTimeMs: number;
  /** Last health check */
  lastCheck: string;
  /** Additional component-specific data */
  metadata?: Record<string, any>;
}

// ============================================================================
// WEBSOCKET & REAL-TIME
// ============================================================================

export interface WebSocketMessage {
  /** Message type */
  type: string;
  /** Message payload data */
  data: Record<string, any>;
  /** Request ID for correlation */
  requestId?: string;
  /** Message timestamp */
  timestamp: string;
  /** Message ID */
  messageId?: string;
}

export interface WebSocketSubscription {
  /** Channel to subscribe to */
  channel: string;
  /** Channel-specific parameters */
  params?: Record<string, any>;
}

export interface StreamChunk {
  /** Chunk content */
  content: string;
  /** Whether this is the final chunk */
  isComplete: boolean;
  /** Chunk sequence number */
  sequence?: number;
  /** Additional chunk metadata */
  metadata?: Record<string, any>;
}

// ============================================================================
// API RESPONSES & PAGINATION
// ============================================================================

export interface PaginatedResponse<T> {
  /** Response data */
  data: T[];
  /** Total number of items */
  total: number;
  /** Current page number */
  page: number;
  /** Total number of pages */
  pages: number;
  /** Items per page */
  limit: number;
  /** Offset from start */
  offset: number;
  /** Whether there are more pages */
  hasMore: boolean;
}

export interface APIResponse<T> {
  /** Response data */
  data: T;
  /** Success flag */
  success: boolean;
  /** Response message */
  message?: string;
  /** Request ID */
  requestId: string;
  /** Response timestamp */
  timestamp: string;
}

export interface APIError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Additional error details */
  details?: Record<string, any>;
  /** Request ID */
  requestId: string;
  /** Error timestamp */
  timestamp: string;
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

export type Callback<T> = (data: T) => void;
export type ErrorCallback = (error: Error) => void;
export type EventHandler<T> = (event: T) => void;

export interface RequestOptions {
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Retry configuration */
  retry?: {
    attempts: number;
    delay: number;
  };
  /** Signal for request cancellation */
  signal?: AbortSignal;
}