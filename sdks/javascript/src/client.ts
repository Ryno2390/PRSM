/**
 * PRSM Main Client
 * The central client for interacting with the Protocol for Recursive Scientific Modeling
 */

import { EventEmitter } from 'eventemitter3';
import {
  PRSMClientConfig,
  QueryRequest,
  PRSMResponse,
  SessionInfo,
  SessionStatus,
  HealthStatus,
  RequestOptions,
  PaginatedResponse,
  ModelInfo,
  Proposal,
  UserProfile,
  SEALConfig,
  SafetyLevel,
} from './types';
import { AuthManager, AuthManagerConfig, MemoryTokenStorage } from './auth';
import { FTNSManager, FTNSManagerConfig } from './ftns';
import { MarketplaceManager, MarketplaceManagerConfig } from './marketplace';
import { ToolsManager, ToolsManagerConfig } from './tools';
import { WebSocketManager, WebSocketManagerConfig } from './websocket';
import {
  PRSMError,
  NetworkError,
  ValidationError,
  ConfigurationError,
  SessionNotFoundError,
  toPRSMError,
} from './errors';

// ============================================================================
// NWTN (Neural Web of Thought Networks) API
// ============================================================================

class NWTNApi {
  constructor(private client: PRSMClient) {}

  /**
   * Submit a research query to NWTN
   */
  async submitQuery(request: QueryRequest): Promise<SessionInfo> {
    this.validateQueryRequest(request);

    try {
      if (this.client.debug) {
        console.log('[PRSM NWTN] Submitting query:', request.query);
      }

      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<SessionInfo>(
        '/api/v1/nwtn/query', {
          method: 'POST',
          headers,
          body: {
            query: request.query,
            model_id: request.modelId,
            domain: request.domain,
            methodology: request.methodology,
            max_iterations: request.maxIterations,
            max_tokens: request.maxTokens,
            temperature: request.temperature,
            system_prompt: request.systemPrompt,
            context: request.context,
            tools: request.tools,
            safety_level: request.safetyLevel,
            include_citations: request.includeCitations,
            seal_enhancement: request.sealEnhancement,
            stream: request.stream,
          },
        }
      );

      if (this.client.debug) {
        console.log('[PRSM NWTN] Query submitted, session ID:', response.sessionId);
      }

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to submit NWTN query');
    }
  }

  /**
   * Get session status and details
   */
  async getSession(sessionId: string): Promise<SessionInfo> {
    try {
      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest<SessionInfo>(
        `/api/v1/nwtn/sessions/${sessionId}`, {
          method: 'GET',
          headers,
        }
      );
    } catch (error) {
      if (error instanceof NetworkError && error.statusCode === 404) {
        throw new SessionNotFoundError(sessionId);
      }
      throw toPRSMError(error, 'Failed to get session details');
    }
  }

  /**
   * List user's sessions
   */
  async listSessions(options: {
    limit?: number;
    offset?: number;
    status?: SessionStatus;
  } = {}): Promise<PaginatedResponse<SessionInfo>> {
    try {
      const params = new URLSearchParams();
      if (options.limit) params.append('limit', options.limit.toString());
      if (options.offset) params.append('offset', options.offset.toString());
      if (options.status) params.append('status', options.status);

      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest<PaginatedResponse<SessionInfo>>(
        `/api/v1/nwtn/sessions?${params}`, {
          method: 'GET',
          headers,
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to list sessions');
    }
  }

  /**
   * Cancel a running session
   */
  async cancelSession(sessionId: string): Promise<void> {
    try {
      if (this.client.debug) {
        console.log('[PRSM NWTN] Cancelling session:', sessionId);
      }

      const headers = await this.client.getAuthHeaders();
      await this.client.makeRequest(`/api/v1/nwtn/sessions/${sessionId}/cancel`, {
        method: 'POST',
        headers,
      });

      if (this.client.debug) {
        console.log('[PRSM NWTN] Session cancelled successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to cancel session');
    }
  }

  /**
   * Wait for session completion
   */
  async waitForCompletion(
    sessionId: string,
    options: {
      timeoutMs?: number;
      pollIntervalMs?: number;
      onProgress?: (session: SessionInfo) => void;
    } = {}
  ): Promise<SessionInfo> {
    const {
      timeoutMs = 600000, // 10 minutes
      pollIntervalMs = 5000, // 5 seconds
      onProgress,
    } = options;

    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const session = await this.getSession(sessionId);

      if (onProgress) {
        onProgress(session);
      }

      if (session.status === SessionStatus.COMPLETED) {
        return session;
      }

      if (session.status === SessionStatus.FAILED) {
        throw new PRSMError(
          'Session failed',
          'SESSION_FAILED',
          sessionId,
          { session }
        );
      }

      if (session.status === SessionStatus.CANCELLED) {
        throw new PRSMError(
          'Session was cancelled',
          'SESSION_CANCELLED',
          sessionId,
          { session }
        );
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }

    throw new PRSMError(
      'Session completion timeout',
      'SESSION_TIMEOUT',
      sessionId
    );
  }

  private validateQueryRequest(request: QueryRequest): void {
    if (!request.query || request.query.trim().length === 0) {
      throw new ValidationError('Query text is required');
    }

    if (request.maxTokens && (request.maxTokens < 1 || request.maxTokens > 32000)) {
      throw new ValidationError('Max tokens must be between 1 and 32000');
    }

    if (request.temperature !== undefined && 
        (request.temperature < 0 || request.temperature > 2)) {
      throw new ValidationError('Temperature must be between 0 and 2');
    }

    if (request.maxIterations && 
        (request.maxIterations < 1 || request.maxIterations > 10)) {
      throw new ValidationError('Max iterations must be between 1 and 10');
    }
  }
}

// ============================================================================
// SEAL TECHNOLOGY API
// ============================================================================

class SEALApi {
  constructor(private client: PRSMClient) {}

  /**
   * Get SEAL performance metrics
   */
  async getMetrics(): Promise<{
    sealSystemStatus: string;
    productionMetrics: {
      knowledgeIncorporationBaseline: number;
      knowledgeIncorporationCurrent: number;
      improvementPercentage: number;
      fewShotLearningSuccessRate: number;
      selfEditGenerationRate: number;
      autonomousImprovementCyclesCompleted: number;
    };
    realTimePerformance: {
      restemPolicyUpdatesPerSecond: number;
      sealRewardCalculationsPerSecond: number;
      autonomousImprovementRate: number;
    };
  }> {
    try {
      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest('/api/v1/seal/metrics', {
        method: 'GET',
        headers,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get SEAL metrics');
    }
  }

  /**
   * Trigger SEAL autonomous improvement
   */
  async triggerImprovement(config: {
    domain: string;
    targetImprovement: number;
    improvementStrategy: string;
    maxIterations: number;
  }): Promise<{ improvementId: string; status: string }> {
    try {
      if (this.client.debug) {
        console.log('[PRSM SEAL] Triggering autonomous improvement');
      }

      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest('/api/v1/seal/improve', {
        method: 'POST',
        headers,
        body: config,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to trigger SEAL improvement');
    }
  }

  /**
   * Get SEAL enhancement status for session
   */
  async getSessionStatus(sessionId: string): Promise<{
    enhancementEnabled: boolean;
    autonomousImprovementActive: boolean;
    estimatedLearningGain: number;
    selfEditGenerationRate: number;
  }> {
    try {
      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest(`/api/v1/seal/sessions/${sessionId}/status`, {
        method: 'GET',
        headers,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get SEAL session status');
    }
  }
}

// ============================================================================
// MODELS API
// ============================================================================

class ModelsApi {
  constructor(private client: PRSMClient) {}

  /**
   * List available models
   */
  async listModels(options: {
    category?: string;
    provider?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<PaginatedResponse<ModelInfo>> {
    try {
      const params = new URLSearchParams();
      if (options.category) params.append('category', options.category);
      if (options.provider) params.append('provider', options.provider);
      if (options.limit) params.append('limit', options.limit.toString());
      if (options.offset) params.append('offset', options.offset.toString());

      return await this.client.makeRequest<PaginatedResponse<ModelInfo>>(
        `/api/v1/models?${params}`, {
          method: 'GET',
          headers: {},
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to list models');
    }
  }

  /**
   * Get model details
   */
  async getModel(modelId: string): Promise<ModelInfo> {
    try {
      return await this.client.makeRequest<ModelInfo>(
        `/api/v1/models/${modelId}`, {
          method: 'GET',
          headers: {},
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get model details');
    }
  }

  /**
   * Train custom model
   */
  async trainModel(config: {
    name: string;
    baseModel: string;
    trainingData: string;
    trainingParameters: Record<string, any>;
    domain: string;
  }): Promise<{ modelId: string; trainingJobId: string }> {
    try {
      if (this.client.debug) {
        console.log('[PRSM Models] Starting model training:', config.name);
      }

      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest('/api/v1/models/train', {
        method: 'POST',
        headers,
        body: config,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to start model training');
    }
  }
}

// ============================================================================
// GOVERNANCE API
// ============================================================================

class GovernanceApi {
  constructor(private client: PRSMClient) {}

  /**
   * List active proposals
   */
  async listProposals(options: {
    status?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<PaginatedResponse<Proposal>> {
    try {
      const params = new URLSearchParams();
      if (options.status) params.append('status', options.status);
      if (options.limit) params.append('limit', options.limit.toString());
      if (options.offset) params.append('offset', options.offset.toString());

      return await this.client.makeRequest<PaginatedResponse<Proposal>>(
        `/api/v1/governance/proposals?${params}`, {
          method: 'GET',
          headers: {},
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to list proposals');
    }
  }

  /**
   * Submit new proposal
   */
  async submitProposal(proposal: {
    title: string;
    description: string;
    category: string;
    implementationPlan: string;
    budgetRequired: number;
  }): Promise<{ proposalId: string }> {
    try {
      if (this.client.debug) {
        console.log('[PRSM Governance] Submitting proposal:', proposal.title);
      }

      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest('/api/v1/governance/proposals', {
        method: 'POST',
        headers,
        body: proposal,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to submit proposal');
    }
  }

  /**
   * Vote on proposal
   */
  async vote(proposalId: string, vote: {
    vote: 'yes' | 'no' | 'abstain';
    votingPower: number;
    comment?: string;
  }): Promise<void> {
    try {
      if (this.client.debug) {
        console.log('[PRSM Governance] Voting on proposal:', proposalId, vote.vote);
      }

      const headers = await this.client.getAuthHeaders();
      await this.client.makeRequest(`/api/v1/governance/proposals/${proposalId}/vote`, {
        method: 'POST',
        headers,
        body: vote,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to vote on proposal');
    }
  }
}

// ============================================================================
// HEALTH API
// ============================================================================

class HealthApi {
  constructor(private client: PRSMClient) {}

  /**
   * Get system health status
   */
  async getSystemHealth(): Promise<HealthStatus> {
    try {
      return await this.client.makeRequest<HealthStatus>('/health', {
        method: 'GET',
        headers: {},
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get system health');
    }
  }

  /**
   * Get detailed health status (authenticated)
   */
  async getDetailedHealth(): Promise<HealthStatus> {
    try {
      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest<HealthStatus>('/health/detailed', {
        method: 'GET',
        headers,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get detailed health status');
    }
  }

  /**
   * Get Prometheus metrics
   */
  async getMetrics(): Promise<string> {
    try {
      const response = await fetch(`${this.client.baseUrl}/health/metrics`);
      if (!response.ok) {
        throw new NetworkError(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.text();
    } catch (error) {
      throw toPRSMError(error, 'Failed to get metrics');
    }
  }
}

// ============================================================================
// MAIN PRSM CLIENT
// ============================================================================

/**
 * Main PRSM Client
 * Provides access to all PRSM API functionality
 */
export class PRSMClient extends EventEmitter {
  public readonly baseUrl: string;
  public readonly websocketUrl: string;
  public readonly debug: boolean;
  public readonly timeout: number;
  public readonly maxRetries: number;
  public readonly headers: Record<string, string>;

  // API modules
  public readonly nwtn: NWTNApi;
  public readonly seal: SEALApi;
  public readonly models: ModelsApi;
  public readonly governance: GovernanceApi;
  public readonly health: HealthApi;

  // Manager instances
  public readonly auth: AuthManager;
  public readonly ftns: FTNSManager;
  public readonly marketplace: MarketplaceManager;
  public readonly tools: ToolsManager;
  public readonly websocket: WebSocketManager;

  constructor(config: PRSMClientConfig) {
    super();

    // Validate configuration
    this.validateConfig(config);

    // Set up base configuration
    this.baseUrl = config.baseUrl || 'https://api.prsm.org';
    this.websocketUrl = config.websocketUrl || 'wss://api.prsm.org/ws';
    this.debug = config.debug ?? false;
    this.timeout = config.timeout ?? 30000;
    this.maxRetries = config.maxRetries ?? 3;
    this.headers = {
      'User-Agent': '@prsm/sdk/0.1.0',
      'Content-Type': 'application/json',
      ...config.headers,
    };

    if (this.debug) {
      console.log('[PRSM Client] Initializing with config:', {
        baseUrl: this.baseUrl,
        websocketUrl: this.websocketUrl,
        timeout: this.timeout,
        maxRetries: this.maxRetries,
      });
    }

    // Initialize authentication manager
    const authConfig: AuthManagerConfig = {
      baseUrl: this.baseUrl,
      tokenStorage: new MemoryTokenStorage(),
      apiKey: config.apiKey,
      autoRefresh: true,
      refreshThresholdSeconds: 300,
      headers: this.headers,
      debug: this.debug,
    };
    this.auth = new AuthManager(authConfig);

    // Initialize FTNS manager
    const ftnsConfig: FTNSManagerConfig = {
      baseUrl: this.baseUrl,
      getAuthHeaders: () => this.getAuthHeaders(),
      headers: this.headers,
      debug: this.debug,
      timeout: this.timeout,
      autoRefreshInterval: 60, // 1 minute
    };
    this.ftns = new FTNSManager(ftnsConfig);

    // Initialize marketplace manager
    const marketplaceConfig: MarketplaceManagerConfig = {
      baseUrl: this.baseUrl,
      getAuthHeaders: () => this.getAuthHeaders(),
      headers: this.headers,
      debug: this.debug,
      timeout: this.timeout,
    };
    this.marketplace = new MarketplaceManager(marketplaceConfig);

    // Initialize tools manager
    const toolsConfig: ToolsManagerConfig = {
      baseUrl: this.baseUrl,
      getAuthHeaders: () => this.getAuthHeaders(),
      headers: this.headers,
      debug: this.debug,
      timeout: this.timeout * 2, // Tools may take longer
    };
    this.tools = new ToolsManager(toolsConfig);

    // Initialize WebSocket manager
    const websocketConfig: WebSocketManagerConfig = {
      websocketUrl: this.websocketUrl,
      getAuthHeaders: () => this.getAuthHeaders(),
      autoReconnect: true,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      connectionTimeout: 30000,
      heartbeatInterval: 30000,
      debug: this.debug,
    };
    this.websocket = new WebSocketManager(websocketConfig);

    // Initialize API modules
    this.nwtn = new NWTNApi(this);
    this.seal = new SEALApi(this);
    this.models = new ModelsApi(this);
    this.governance = new GovernanceApi(this);
    this.health = new HealthApi(this);

    // Set up event forwarding
    this.setupEventForwarding();

    if (this.debug) {
      console.log('[PRSM Client] Initialization complete');
    }
  }

  // ============================================================================
  // AUTHENTICATION CONVENIENCE METHODS
  // ============================================================================

  /**
   * Login with username and password
   */
  async login(username: string, password: string, twoFactorCode?: string): Promise<UserProfile> {
    return this.auth.login({ username, password, twoFactorCode });
  }

  /**
   * Register new user account
   */
  async register(userData: {
    username: string;
    email: string;
    password: string;
    role?: string;
    organization?: string;
  }): Promise<UserProfile> {
    return this.auth.register(userData);
  }

  /**
   * Logout current user
   */
  async logout(): Promise<void> {
    await this.auth.logout();
  }

  /**
   * Get current user profile
   */
  async getCurrentUser(): Promise<UserProfile> {
    return this.auth.getCurrentUser();
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return this.auth.isAuthenticated();
  }

  // ============================================================================
  // CONVENIENCE METHODS
  // ============================================================================

  /**
   * Quick research query with basic options
   */
  async query(
    query: string,
    options: {
      domain?: string;
      maxIterations?: number;
      includeCitations?: boolean;
      sealEnhancement?: Partial<SEALConfig>;
    } = {}
  ): Promise<PRSMResponse> {
    const session = await this.nwtn.submitQuery({
      query,
      domain: options.domain,
      maxIterations: options.maxIterations || 3,
      includeCitations: options.includeCitations ?? true,
      sealEnhancement: {
        enabled: true,
        autonomousImprovement: true,
        targetLearningGain: 0.15,
        restemMethodology: true,
        ...options.sealEnhancement,
      },
    });

    const result = await this.nwtn.waitForCompletion(session.sessionId);
    
    if (!result.results) {
      throw new PRSMError('Session completed without results', 'NO_RESULTS');
    }

    return {
      content: result.results.summary,
      modelId: 'nwtn-ensemble',
      provider: 'prsm_distilled',
      executionTime: Date.now() - new Date(session.createdAt).getTime(),
      tokenUsage: {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
      ftnsCost: result.costActual?.ftnsTokens || 0,
      reasoningTrace: [],
      safetyStatus: SafetyLevel.MODERATE,
      metadata: result.results.artifacts || {},
      requestId: session.sessionId,
      timestamp: new Date().toISOString(),
      citations: result.results.citations,
      sealPerformance: undefined, // Would need to fetch from SEAL API
    };
  }

  // ============================================================================
  // INTERNAL METHODS
  // ============================================================================

  /**
   * Get authentication headers
   */
  async getAuthHeaders(): Promise<Record<string, string>> {
    try {
      return await this.auth.getAuthHeaders();
    } catch (error) {
      if (this.debug) {
        console.warn('[PRSM Client] Failed to get auth headers:', error);
      }
      return {};
    }
  }

  /**
   * Make HTTP request with retry logic
   */
  async makeRequest<T = any>(
    endpoint: string,
    options: {
      method: string;
      headers?: Record<string, string>;
      body?: any;
      timeout?: number;
    }
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      ...this.headers,
      ...options.headers,
    };

    const requestOptions: RequestInit = {
      method: options.method,
      headers,
    };

    if (options.body) {
      requestOptions.body = JSON.stringify(options.body);
    }

    let lastError: Error;
    
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        // Add timeout
        const controller = new AbortController();
        requestOptions.signal = controller.signal;
        const timeoutId = setTimeout(
          () => controller.abort(), 
          options.timeout || this.timeout
        );

        const response = await fetch(url, requestOptions);
        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new NetworkError(
            errorData.message || `HTTP ${response.status}: ${response.statusText}`,
            undefined,
            undefined,
            { statusCode: response.status, ...errorData }
          );
        }

        const data = await response.json();
        return data.data || data; // Handle both APIResponse<T> and direct T

      } catch (error) {
        lastError = error as Error;
        
        if (error.name === 'AbortError') {
          lastError = new NetworkError('Request timeout');
        }

        // Don't retry on certain errors
        if (error instanceof NetworkError && 
            (error.statusCode === 401 || error.statusCode === 403 || error.statusCode === 404)) {
          break;
        }

        // Wait before retry (exponential backoff)
        if (attempt < this.maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
          await new Promise(resolve => setTimeout(resolve, delay));
          
          if (this.debug) {
            console.log(`[PRSM Client] Retrying request (attempt ${attempt + 2}/${this.maxRetries + 1})`);
          }
        }
      }
    }

    throw lastError;
  }

  /**
   * Set up event forwarding from managers
   */
  private setupEventForwarding(): void {
    // Forward WebSocket events
    this.websocket.on('connected', () => this.emit('websocket_connected'));
    this.websocket.on('disconnected', () => this.emit('websocket_disconnected'));
    this.websocket.on('error', (error) => this.emit('websocket_error', error));
    this.websocket.on('session_progress', (progress) => this.emit('session_progress', progress));
    this.websocket.on('safety_alert', (alert) => this.emit('safety_alert', alert));

    // Forward authentication events
    this.auth.on?.('token_refreshed', () => this.emit('token_refreshed'));
    this.auth.on?.('logout', () => this.emit('logout'));
  }

  /**
   * Validate client configuration
   */
  private validateConfig(config: PRSMClientConfig): void {
    if (!config.apiKey && !config.baseUrl?.includes('localhost')) {
      throw new ConfigurationError('API key is required for production environments');
    }

    if (config.timeout && (config.timeout < 1000 || config.timeout > 300000)) {
      throw new ConfigurationError('Timeout must be between 1 second and 5 minutes');
    }

    if (config.maxRetries && (config.maxRetries < 0 || config.maxRetries > 10)) {
      throw new ConfigurationError('Max retries must be between 0 and 10');
    }
  }

  // ============================================================================
  // LIFECYCLE METHODS
  // ============================================================================

  /**
   * Initialize client and establish connections
   */
  async initialize(): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Client] Initializing client');
      }

      // Validate authentication if API key is provided
      if (this.auth.getToken() || this.auth.apiKey) {
        this.auth.validateConfig();
      }

      // Test API connectivity
      await this.health.getSystemHealth();

      // Connect WebSocket if authenticated
      if (this.isAuthenticated()) {
        await this.websocket.connect();
      }

      if (this.debug) {
        console.log('[PRSM Client] Client initialized successfully');
      }

      this.emit('initialized');
    } catch (error) {
      if (this.debug) {
        console.error('[PRSM Client] Initialization failed:', error);
      }
      this.emit('initialization_failed', error);
      throw toPRSMError(error, 'Client initialization failed');
    }
  }

  /**
   * Clean up resources and disconnect
   */
  async destroy(): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Client] Destroying client');
      }

      // Disconnect WebSocket
      this.websocket.destroy();

      // Stop FTNS auto-refresh
      this.ftns.stopAutoRefresh();

      // Clear caches
      this.marketplace.clearCache();
      this.tools.clearCache();

      // Remove all listeners
      this.removeAllListeners();

      if (this.debug) {
        console.log('[PRSM Client] Client destroyed successfully');
      }

      this.emit('destroyed');
    } catch (error) {
      if (this.debug) {
        console.error('[PRSM Client] Destruction failed:', error);
      }
      throw toPRSMError(error, 'Client destruction failed');
    }
  }

  /**
   * Get client status and statistics
   */
  getStatus(): {
    isAuthenticated: boolean;
    isWebSocketConnected: boolean;
    baseUrl: string;
    websocketUrl: string;
    ftnsBalance: number | null;
    authStatus: any;
    websocketStats: any;
  } {
    return {
      isAuthenticated: this.isAuthenticated(),
      isWebSocketConnected: this.websocket.isConnected(),
      baseUrl: this.baseUrl,
      websocketUrl: this.websocketUrl,
      ftnsBalance: this.ftns.getCachedBalance()?.totalBalance || null,
      authStatus: this.auth.getAuthStatus(),
      websocketStats: this.websocket.getStats(),
    };
  }
}