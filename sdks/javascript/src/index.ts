/**
 * PRSM JavaScript/TypeScript SDK
 * Official client library for the Protocol for Recursive Scientific Modeling
 * 
 * Features:
 * - Complete API client with all PRSM endpoints
 * - TypeScript support with full type definitions
 * - WebSocket support for real-time features
 * - Error handling and retry logic
 * - Authentication and API key management
 * - FTNS token management and Web3 integration
 * - Marketplace operations and model management
 * - Tool execution and management
 * 
 * @example Basic Usage
 * ```typescript
 * import { PRSMClient } from '@prsm/sdk';
 * 
 * const client = new PRSMClient({ 
 *   apiKey: 'your_api_key',
 *   baseUrl: 'https://api.prsm.org'
 * });
 * 
 * // Submit a research query
 * const session = await client.nwtn.submitQuery({
 *   query: 'Analyze the impact of climate change on marine ecosystems',
 *   domain: 'environmental_science',
 *   maxIterations: 5,
 *   includeCitations: true
 * });
 * 
 * // Monitor progress
 * const result = await client.nwtn.waitForCompletion(session.sessionId);
 * console.log(result.summary);
 * ```
 * 
 * @example WebSocket Real-time Updates
 * ```typescript
 * import { PRSMClient } from '@prsm/sdk';
 * 
 * const client = new PRSMClient({ apiKey: 'your_api_key' });
 * 
 * // Connect WebSocket for real-time updates
 * await client.websocket.connect();
 * 
 * // Subscribe to session progress
 * client.websocket.subscribeToSession(sessionId, (progress) => {
 *   console.log(`Progress: ${progress.progress}%`);
 * });
 * ```
 * 
 * @example FTNS Token Management
 * ```typescript
 * // Check balance
 * const balance = await client.ftns.getBalance();
 * console.log(`Available: ${balance.availableBalance} FTNS`);
 * 
 * // Transfer tokens
 * const transfer = await client.ftns.transfer({
 *   toAddress: '0x...',
 *   amount: 100,
 *   note: 'Research collaboration'
 * });
 * ```
 * 
 * @example Marketplace Operations
 * ```typescript
 * // Browse models
 * const models = await client.marketplace.browseModels({
 *   category: 'scientific',
 *   featured: true,
 *   limit: 10
 * });
 * 
 * // Rent a model
 * const rental = await client.marketplace.rentModel(modelId, {
 *   durationHours: 24,
 *   maxRequests: 1000
 * });
 * ```
 */

// ============================================================================
// MAIN CLIENT EXPORT
// ============================================================================

export { PRSMClient } from './client';

// ============================================================================
// CORE TYPE EXPORTS
// ============================================================================

export type {
  // Client configuration
  PRSMClientConfig,
  RequestOptions,

  // Core API types
  QueryRequest,
  PRSMResponse,
  SessionInfo,
  SessionResults,
  
  // SEAL technology types
  SEALConfig,
  SEALPerformance,
  SEALSessionStatus,
  
  // Citations and references
  Citation,
  
  // Financial and token types
  FTNSBalance,
  TransactionInfo,
  PaymentRequest,
  BillingInfo,
  
  // Model and marketplace types
  ModelInfo,
  MarketplaceQuery,
  MarketplaceModel,
  SubscriptionOption,
  
  // Tool types
  ToolSpec,
  ToolExecutionRequest,
  ToolExecutionResponse,
  
  // User and authentication types
  UserProfile,
  AuthTokens,
  LoginRequest,
  RegisterRequest,
  
  // Governance types
  Proposal,
  VoteRequest,
  
  // Security and monitoring types
  SecurityStatus,
  SafetyStatus,
  HealthStatus,
  ComponentHealth,
  
  // WebSocket types
  WebSocketMessage,
  WebSocketSubscription,
  StreamChunk,
  
  // API response types
  PaginatedResponse,
  APIResponse,
  APIError,
  
  // Utility types
  Callback,
  ErrorCallback,
  EventHandler,
} from './types';

// ============================================================================
// ENUM EXPORTS
// ============================================================================

export {
  ModelProvider,
  SafetyLevel,
  SessionStatus,
  ModelCategory,
  NetworkType,
  ProposalStatus,
  UserRole,
} from './types';

// ============================================================================
// ERROR EXPORTS
// ============================================================================

export {
  // Base error
  PRSMError,
  
  // Authentication errors
  AuthenticationError,
  InvalidAPIKeyError,
  TokenExpiredError,
  InsufficientPermissionsError,
  
  // Financial errors
  InsufficientFundsError,
  PaymentError,
  Web3TransactionError,
  
  // Safety and security errors
  SafetyViolationError,
  ContentBlockedError,
  CircuitBreakerError,
  
  // Network errors
  NetworkError,
  TimeoutError,
  RateLimitError,
  
  // Resource errors
  ResourceNotFoundError,
  ModelNotFoundError,
  SessionNotFoundError,
  ToolNotFoundError,
  
  // Execution errors
  ToolExecutionError,
  ModelExecutionError,
  SessionExecutionError,
  
  // Validation errors
  ValidationError,
  MissingParameterError,
  InvalidParameterError,
  
  // WebSocket errors
  WebSocketError,
  WebSocketConnectionError,
  InvalidWebSocketMessageError,
  
  // Service errors
  ServiceUnavailableError,
  ConfigurationError,
  
  // Error utilities
  isPRSMError,
  getErrorCode,
  toPRSMError,
  formatErrorForLogging,
  isRetryableError,
  getRetryDelay,
} from './errors';

// ============================================================================
// MANAGER EXPORTS
// ============================================================================

export { AuthManager, type AuthManagerConfig } from './auth';
export { FTNSManager, type FTNSManagerConfig } from './ftns';
export { MarketplaceManager, type MarketplaceManagerConfig } from './marketplace';
export { ToolsManager, type ToolsManagerConfig } from './tools';
export { WebSocketManager, type WebSocketManagerConfig, WebSocketState, MessageType } from './websocket';

// ============================================================================
// TOKEN STORAGE EXPORTS
// ============================================================================

export {
  type TokenStorage,
  MemoryTokenStorage,
  LocalStorageTokenStorage,
} from './auth';

// ============================================================================
// ADDITIONAL TYPE EXPORTS
// ============================================================================

export type {
  // FTNS specific types
  StakingInfo,
  StakingPool,
  WalletInfo,
  TransferRequest,
} from './ftns';

export type {
  // Marketplace specific types
  ModelSubmission,
  ModelRental,
  RentalRequest,
  ModelReview,
  ReviewSubmission,
  MarketplaceFilters,
} from './marketplace';

export type {
  // Tools specific types
  ToolDiscoveryOptions,
  ToolExecutionContext,
  AsyncToolExecution,
  ToolInstallation,
  ToolValidationResult,
} from './tools';

export type {
  // WebSocket specific types
  ConnectionEvent,
  SessionProgressEvent,
  StreamEvent,
  SafetyAlertEvent,
} from './websocket';

// ============================================================================
// CONVENIENCE EXPORTS
// ============================================================================

/**
 * Create a new PRSM client with API key authentication
 */
export function createClient(apiKey: string, options: Omit<PRSMClientConfig, 'apiKey'> = {}): PRSMClient {
  return new PRSMClient({ ...options, apiKey });
}

/**
 * Create a new PRSM client for browser environments with localStorage token storage
 */
export function createBrowserClient(config: PRSMClientConfig): PRSMClient {
  return new PRSMClient(config);
}

/**
 * Create a new PRSM client for Node.js environments
 */
export function createNodeClient(config: PRSMClientConfig): PRSMClient {
  return new PRSMClient(config);
}

// ============================================================================
// VERSION AND METADATA
// ============================================================================

/** SDK version */
export const VERSION = '0.1.0';

/** SDK name */
export const SDK_NAME = '@prsm/sdk';

/** SDK user agent */
export const USER_AGENT = `${SDK_NAME}/${VERSION}`;

/** Supported API version */
export const API_VERSION = 'v1';

/** Default API endpoints */
export const DEFAULT_ENDPOINTS = {
  PRODUCTION: 'https://api.prsm.org',
  STAGING: 'https://staging-api.prsm.org',
  DEVELOPMENT: 'http://localhost:8000',
} as const;

/** Default WebSocket endpoints */
export const DEFAULT_WEBSOCKET_ENDPOINTS = {
  PRODUCTION: 'wss://api.prsm.org/ws',
  STAGING: 'wss://staging-api.prsm.org/ws',
  DEVELOPMENT: 'ws://localhost:8000/ws',
} as const;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if running in browser environment
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined' && typeof window.document !== 'undefined';
}

/**
 * Check if running in Node.js environment
 */
export function isNode(): boolean {
  return typeof process !== 'undefined' && process.versions != null && process.versions.node != null;
}

/**
 * Get appropriate storage for current environment
 */
export function getDefaultStorage(): TokenStorage {
  if (isBrowser()) {
    return new LocalStorageTokenStorage();
  } else {
    return new MemoryTokenStorage();
  }
}

/**
 * Format FTNS amount for display
 */
export function formatFTNS(amount: number, decimals: number = 4): string {
  return FTNSManager.formatFTNS(amount, decimals);
}

/**
 * Parse FTNS amount from string
 */
export function parseFTNS(value: string): number {
  return FTNSManager.parseFTNS(value);
}

/**
 * Format tool execution cost for display
 */
export function formatToolCost(cost: number): string {
  return ToolsManager.formatCost(cost);
}

/**
 * Format execution duration
 */
export function formatDuration(durationMs: number): string {
  return ToolsManager.formatDuration(durationMs);
}

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

export default PRSMClient;