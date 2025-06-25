/**
 * PRSM SDK Tools Manager
 * Handles tool discovery, execution, and management
 */

import {
  ToolSpec,
  ToolExecutionRequest,
  ToolExecutionResponse,
  SafetyLevel,
  PaginatedResponse,
} from './types';
import {
  ToolNotFoundError,
  ToolExecutionError,
  ValidationError,
  SafetyViolationError,
  NetworkError,
  InsufficientPermissionsError,
  toPRSMError,
} from './errors';

// ============================================================================
// TOOLS MANAGER CONFIGURATION
// ============================================================================

export interface ToolsManagerConfig {
  /** Base URL for tools API endpoints */
  baseUrl: string;
  /** Authentication headers function */
  getAuthHeaders: () => Promise<Record<string, string>>;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Debug logging */
  debug?: boolean;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Default safety level for tool execution */
  defaultSafetyLevel?: SafetyLevel;
}

// ============================================================================
// TOOL DISCOVERY TYPES
// ============================================================================

export interface ToolDiscoveryOptions {
  /** Search query */
  query?: string;
  /** Filter by category */
  category?: string;
  /** Filter by provider */
  provider?: string;
  /** Filter by required safety level */
  safetyLevel?: SafetyLevel;
  /** Filter by cost range */
  costRange?: {
    min: number;
    max: number;
  };
  /** Filter by capabilities */
  capabilities?: string[];
  /** Filter by required permissions */
  permissions?: string[];
  /** Only show verified tools */
  verifiedOnly?: boolean;
  /** Sort criteria */
  sortBy?: 'name' | 'cost' | 'rating' | 'usage' | 'created';
  /** Sort direction */
  sortOrder?: 'asc' | 'desc';
  /** Result limit */
  limit?: number;
  /** Result offset */
  offset?: number;
}

// ============================================================================
// TOOL EXECUTION TYPES
// ============================================================================

export interface ToolExecutionContext {
  /** Session ID for tracking */
  sessionId?: string;
  /** User context data */
  userContext?: Record<string, any>;
  /** Environment variables */
  environment?: Record<string, string>;
  /** Execution priority */
  priority?: 'low' | 'normal' | 'high';
  /** Maximum execution time in seconds */
  maxExecutionTime?: number;
  /** Memory limit in MB */
  memoryLimit?: number;
  /** CPU limit as percentage */
  cpuLimit?: number;
}

export interface AsyncToolExecution {
  /** Execution ID */
  executionId: string;
  /** Tool name */
  toolName: string;
  /** Execution status */
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  /** Progress percentage (0-100) */
  progress: number;
  /** Estimated completion time */
  estimatedCompletion?: string;
  /** Execution result (when completed) */
  result?: any;
  /** Error information (if failed) */
  error?: string;
  /** Execution metadata */
  metadata: Record<string, any>;
  /** Started timestamp */
  startedAt: string;
  /** Completed timestamp */
  completedAt?: string;
}

// ============================================================================
// TOOL INSTALLATION TYPES
// ============================================================================

export interface ToolInstallation {
  /** Tool name */
  toolName: string;
  /** Installation status */
  status: 'installing' | 'installed' | 'failed' | 'updating';
  /** Tool version */
  version: string;
  /** Installation progress */
  progress: number;
  /** Installation path */
  installPath?: string;
  /** Required dependencies */
  dependencies?: Array<{
    name: string;
    version: string;
    status: 'pending' | 'installed' | 'failed';
  }>;
  /** Installation log */
  log?: string[];
  /** Installation timestamp */
  installedAt?: string;
}

// ============================================================================
// TOOL VALIDATION TYPES
// ============================================================================

export interface ToolValidationResult {
  /** Validation status */
  valid: boolean;
  /** Validation errors */
  errors: string[];
  /** Validation warnings */
  warnings: string[];
  /** Security assessment */
  securityLevel: SafetyLevel;
  /** Performance estimate */
  performanceEstimate?: {
    expectedDuration: number;
    memoryUsage: number;
    cpuUsage: number;
  };
}

// ============================================================================
// TOOLS MANAGER CLASS
// ============================================================================

/**
 * Manages tool discovery, execution, and lifecycle
 */
export class ToolsManager {
  private readonly baseUrl: string;
  private readonly getAuthHeaders: () => Promise<Record<string, string>>;
  private readonly headers: Record<string, string>;
  private readonly debug: boolean;
  private readonly timeout: number;
  private readonly defaultSafetyLevel: SafetyLevel;

  private toolsCache: Map<string, ToolSpec> = new Map();
  private categoriesCache: string[] | null = null;
  private lastCacheUpdate: Date | null = null;
  private activeExecutions: Map<string, AsyncToolExecution> = new Map();

  constructor(config: ToolsManagerConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.getAuthHeaders = config.getAuthHeaders;
    this.headers = config.headers || {};
    this.debug = config.debug ?? false;
    this.timeout = config.timeout ?? 60000; // 1 minute default for tools
    this.defaultSafetyLevel = config.defaultSafetyLevel || SafetyLevel.MODERATE;
  }

  // ============================================================================
  // TOOL DISCOVERY
  // ============================================================================

  /**
   * Discover available tools
   */
  async discoverTools(
    options: ToolDiscoveryOptions = {}
  ): Promise<PaginatedResponse<ToolSpec>> {
    try {
      if (this.debug) {
        console.log('[PRSM Tools] Discovering tools with options:', options);
      }

      const params = this.buildDiscoveryParams(options);
      const headers = await this.getAuthHeaders();

      const response = await this.makeRequest<PaginatedResponse<ToolSpec>>(
        `/api/v1/tools?${params}`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );

      // Update cache with discovered tools
      response.data.forEach(tool => {
        this.toolsCache.set(tool.name, tool);
      });

      if (this.debug) {
        console.log('[PRSM Tools] Discovered', response.data.length, 'tools');
      }

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to discover tools');
    }
  }

  /**
   * Search tools by text query
   */
  async searchTools(
    query: string,
    options: Omit<ToolDiscoveryOptions, 'query'> = {}
  ): Promise<PaginatedResponse<ToolSpec>> {
    return this.discoverTools({ ...options, query });
  }

  /**
   * Get tools by category
   */
  async getToolsByCategory(
    category: string,
    options: Omit<ToolDiscoveryOptions, 'category'> = {}
  ): Promise<PaginatedResponse<ToolSpec>> {
    return this.discoverTools({ ...options, category });
  }

  /**
   * Get available tool categories
   */
  async getCategories(forceRefresh = false): Promise<string[]> {
    if (!forceRefresh && this.categoriesCache && this.isCacheFresh()) {
      return this.categoriesCache;
    }

    try {
      if (this.debug) {
        console.log('[PRSM Tools] Fetching tool categories');
      }

      const response = await this.makeRequest<{ categories: string[] }>(
        '/api/v1/tools/categories', {
          method: 'GET',
          headers: this.headers,
        }
      );

      this.categoriesCache = response.categories;
      this.lastCacheUpdate = new Date();

      return response.categories;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get tool categories');
    }
  }

  /**
   * Get specific tool details
   */
  async getTool(toolName: string, useCache = true): Promise<ToolSpec> {
    // Check cache first
    if (useCache && this.toolsCache.has(toolName)) {
      return this.toolsCache.get(toolName)!;
    }

    try {
      if (this.debug) {
        console.log('[PRSM Tools] Fetching tool details:', toolName);
      }

      const tool = await this.makeRequest<ToolSpec>(
        `/api/v1/tools/${encodeURIComponent(toolName)}`, {
          method: 'GET',
          headers: this.headers,
        }
      );

      // Update cache
      this.toolsCache.set(toolName, tool);

      return tool;
    } catch (error) {
      if (error instanceof NetworkError && error.statusCode === 404) {
        throw new ToolNotFoundError(toolName);
      }
      throw toPRSMError(error, 'Failed to get tool details');
    }
  }

  // ============================================================================
  // TOOL EXECUTION
  // ============================================================================

  /**
   * Execute a tool synchronously
   */
  async executeTool(
    toolName: string,
    parameters: Record<string, any>,
    context: ToolExecutionContext = {}
  ): Promise<ToolExecutionResponse> {
    const request: ToolExecutionRequest = {
      toolName,
      parameters,
      context: {
        ...context,
        sessionId: context.sessionId,
        userContext: context.userContext,
        environment: context.environment,
      },
      safetyLevel: context.priority === 'high' ? SafetyLevel.HIGH : this.defaultSafetyLevel,
      timeout: context.maxExecutionTime ? context.maxExecutionTime * 1000 : this.timeout,
      async: false,
    };

    return this.performToolExecution(request);
  }

  /**
   * Execute a tool asynchronously
   */
  async executeToolAsync(
    toolName: string,
    parameters: Record<string, any>,
    context: ToolExecutionContext = {}
  ): Promise<AsyncToolExecution> {
    const request: ToolExecutionRequest = {
      toolName,
      parameters,
      context: {
        ...context,
        sessionId: context.sessionId,
        userContext: context.userContext,
        environment: context.environment,
      },
      safetyLevel: context.priority === 'high' ? SafetyLevel.HIGH : this.defaultSafetyLevel,
      timeout: context.maxExecutionTime ? context.maxExecutionTime * 1000 : this.timeout,
      async: true,
    };

    try {
      if (this.debug) {
        console.log('[PRSM Tools] Starting async execution:', toolName);
      }

      const headers = await this.getAuthHeaders();
      const execution = await this.makeRequest<AsyncToolExecution>(
        '/api/v1/tools/execute', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: request,
        }
      );

      // Track execution
      this.activeExecutions.set(execution.executionId, execution);

      if (this.debug) {
        console.log('[PRSM Tools] Async execution started:', execution.executionId);
      }

      return execution;
    } catch (error) {
      throw toPRSMError(error, 'Failed to start async tool execution');
    }
  }

  /**
   * Get async execution status
   */
  async getExecutionStatus(executionId: string): Promise<AsyncToolExecution> {
    try {
      const headers = await this.getAuthHeaders();
      const execution = await this.makeRequest<AsyncToolExecution>(
        `/api/v1/tools/executions/${executionId}`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );

      // Update cache
      this.activeExecutions.set(executionId, execution);

      return execution;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get execution status');
    }
  }

  /**
   * Wait for async execution to complete
   */
  async waitForExecution(
    executionId: string,
    timeoutMs: number = 300000, // 5 minutes
    pollIntervalMs: number = 2000 // 2 seconds
  ): Promise<AsyncToolExecution> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const execution = await this.getExecutionStatus(executionId);

      if (execution.status === 'completed') {
        if (this.debug) {
          console.log('[PRSM Tools] Execution completed:', executionId);
        }
        return execution;
      }

      if (execution.status === 'failed') {
        throw new ToolExecutionError(
          execution.toolName,
          execution.error || 'Execution failed',
          executionId
        );
      }

      if (execution.status === 'cancelled') {
        throw new ToolExecutionError(
          execution.toolName,
          'Execution was cancelled',
          executionId
        );
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }

    throw new ToolExecutionError(
      'unknown',
      'Execution timeout',
      executionId
    );
  }

  /**
   * Cancel async execution
   */
  async cancelExecution(executionId: string): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Tools] Cancelling execution:', executionId);
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest(`/api/v1/tools/executions/${executionId}/cancel`, {
        method: 'POST',
        headers: { ...headers, ...this.headers },
      });

      // Update cache
      const execution = this.activeExecutions.get(executionId);
      if (execution) {
        execution.status = 'cancelled';
        this.activeExecutions.set(executionId, execution);
      }

      if (this.debug) {
        console.log('[PRSM Tools] Execution cancelled successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to cancel execution');
    }
  }

  /**
   * Get user's active executions
   */
  async getActiveExecutions(): Promise<AsyncToolExecution[]> {
    try {
      if (this.debug) {
        console.log('[PRSM Tools] Fetching active executions');
      }

      const headers = await this.getAuthHeaders();
      const response = await this.makeRequest<{ executions: AsyncToolExecution[] }>(
        '/api/v1/tools/executions', {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );

      // Update cache
      response.executions.forEach(execution => {
        this.activeExecutions.set(execution.executionId, execution);
      });

      return response.executions;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get active executions');
    }
  }

  // ============================================================================
  // TOOL VALIDATION
  // ============================================================================

  /**
   * Validate tool parameters before execution
   */
  async validateToolParameters(
    toolName: string,
    parameters: Record<string, any>
  ): Promise<ToolValidationResult> {
    try {
      if (this.debug) {
        console.log('[PRSM Tools] Validating parameters for:', toolName);
      }

      const headers = await this.getAuthHeaders();
      return await this.makeRequest<ToolValidationResult>(
        `/api/v1/tools/${encodeURIComponent(toolName)}/validate`, {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: { parameters },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to validate tool parameters');
    }
  }

  /**
   * Check if user has permission to use tool
   */
  async checkToolPermissions(toolName: string): Promise<{
    hasPermission: boolean;
    requiredPermissions: string[];
    missingPermissions: string[];
  }> {
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest(
        `/api/v1/tools/${encodeURIComponent(toolName)}/permissions`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to check tool permissions');
    }
  }

  /**
   * Estimate tool execution cost
   */
  async estimateExecutionCost(
    toolName: string,
    parameters: Record<string, any>
  ): Promise<{
    estimatedCost: number;
    breakdown: Array<{
      component: string;
      cost: number;
    }>;
  }> {
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest(
        `/api/v1/tools/${encodeURIComponent(toolName)}/cost`, {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: { parameters },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to estimate execution cost');
    }
  }

  // ============================================================================
  // TOOL INSTALLATION
  // ============================================================================

  /**
   * Install a tool
   */
  async installTool(
    toolName: string,
    version?: string
  ): Promise<ToolInstallation> {
    try {
      if (this.debug) {
        console.log('[PRSM Tools] Installing tool:', toolName, version || 'latest');
      }

      const headers = await this.getAuthHeaders();
      const installation = await this.makeRequest<ToolInstallation>(
        '/api/v1/tools/install', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: {
            tool_name: toolName,
            version: version,
          },
        }
      );

      if (this.debug) {
        console.log('[PRSM Tools] Tool installation started:', installation.toolName);
      }

      return installation;
    } catch (error) {
      throw toPRSMError(error, 'Failed to install tool');
    }
  }

  /**
   * Get tool installation status
   */
  async getInstallationStatus(toolName: string): Promise<ToolInstallation> {
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest<ToolInstallation>(
        `/api/v1/tools/${encodeURIComponent(toolName)}/installation`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get installation status');
    }
  }

  /**
   * Uninstall a tool
   */
  async uninstallTool(toolName: string): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Tools] Uninstalling tool:', toolName);
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest(`/api/v1/tools/${encodeURIComponent(toolName)}/uninstall`, {
        method: 'POST',
        headers: { ...headers, ...this.headers },
      });

      // Remove from cache
      this.toolsCache.delete(toolName);

      if (this.debug) {
        console.log('[PRSM Tools] Tool uninstalled successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to uninstall tool');
    }
  }

  /**
   * List installed tools
   */
  async getInstalledTools(): Promise<ToolInstallation[]> {
    try {
      if (this.debug) {
        console.log('[PRSM Tools] Fetching installed tools');
      }

      const headers = await this.getAuthHeaders();
      const response = await this.makeRequest<{ tools: ToolInstallation[] }>(
        '/api/v1/tools/installed', {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );

      return response.tools;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get installed tools');
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private async performToolExecution(
    request: ToolExecutionRequest
  ): Promise<ToolExecutionResponse> {
    this.validateExecutionRequest(request);

    try {
      if (this.debug) {
        console.log('[PRSM Tools] Executing tool:', request.toolName);
      }

      // Check permissions first
      const permissions = await this.checkToolPermissions(request.toolName);
      if (!permissions.hasPermission) {
        throw new InsufficientPermissionsError(
          `Tool execution: ${request.toolName}`,
          undefined,
          { missingPermissions: permissions.missingPermissions }
        );
      }

      // Validate parameters
      const validation = await this.validateToolParameters(
        request.toolName,
        request.parameters
      );

      if (!validation.valid) {
        throw new ValidationError(
          `Invalid tool parameters: ${validation.errors.join(', ')}`,
          undefined,
          request.parameters
        );
      }

      // Check safety level
      if (validation.securityLevel === SafetyLevel.CRITICAL ||
          validation.securityLevel === SafetyLevel.EMERGENCY) {
        throw new SafetyViolationError(
          'high_risk_tool',
          validation.securityLevel,
          'Tool execution blocked due to high security risk'
        );
      }

      const headers = await this.getAuthHeaders();
      const response = await this.makeRequest<ToolExecutionResponse>(
        '/api/v1/tools/execute', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: request,
        }
      );

      if (this.debug) {
        console.log('[PRSM Tools] Tool executed successfully:', response.executionId);
      }

      return response;
    } catch (error) {
      if (error instanceof ToolNotFoundError ||
          error instanceof ValidationError ||
          error instanceof SafetyViolationError ||
          error instanceof InsufficientPermissionsError) {
        throw error;
      }
      
      throw toPRSMError(error, 'Tool execution failed');
    }
  }

  private buildDiscoveryParams(options: ToolDiscoveryOptions): string {
    const params = new URLSearchParams();

    if (options.query) params.append('query', options.query);
    if (options.category) params.append('category', options.category);
    if (options.provider) params.append('provider', options.provider);
    if (options.safetyLevel) params.append('safety_level', options.safetyLevel);
    if (options.verifiedOnly) params.append('verified_only', 'true');
    if (options.sortBy) params.append('sort_by', options.sortBy);
    if (options.sortOrder) params.append('sort_order', options.sortOrder);
    if (options.limit) params.append('limit', options.limit.toString());
    if (options.offset) params.append('offset', options.offset.toString());

    if (options.costRange) {
      params.append('cost_min', options.costRange.min.toString());
      params.append('cost_max', options.costRange.max.toString());
    }

    if (options.capabilities) {
      options.capabilities.forEach(cap => params.append('capabilities', cap));
    }

    if (options.permissions) {
      options.permissions.forEach(perm => params.append('permissions', perm));
    }

    return params.toString();
  }

  private validateExecutionRequest(request: ToolExecutionRequest): void {
    if (!request.toolName || request.toolName.trim().length === 0) {
      throw new ValidationError('Tool name is required');
    }

    if (!request.parameters || typeof request.parameters !== 'object') {
      throw new ValidationError('Tool parameters must be an object');
    }

    if (request.timeout && (request.timeout < 1000 || request.timeout > 3600000)) {
      throw new ValidationError('Timeout must be between 1 second and 1 hour');
    }
  }

  private isCacheFresh(): boolean {
    if (!this.lastCacheUpdate) return false;
    const age = Date.now() - this.lastCacheUpdate.getTime();
    return age < 300000; // 5 minutes
  }

  /**
   * Make HTTP request to API
   */
  private async makeRequest<T = any>(
    endpoint: string,
    options: {
      method: string;
      headers?: Record<string, string>;
      body?: any;
    }
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    const requestOptions: RequestInit = {
      method: options.method,
      headers,
    };

    if (options.body) {
      requestOptions.body = JSON.stringify(options.body);
    }

    // Add timeout
    const controller = new AbortController();
    requestOptions.signal = controller.signal;
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, requestOptions);
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new NetworkError(
          errorData.message || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();
      return data.data || data; // Handle both APIResponse<T> and direct T
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        throw new NetworkError('Request timeout');
      }

      if (error instanceof NetworkError) {
        throw error;
      }

      throw new NetworkError('Network request failed', error as Error);
    }
  }

  /**
   * Clear cached data
   */
  clearCache(): void {
    this.toolsCache.clear();
    this.categoriesCache = null;
    this.lastCacheUpdate = null;

    if (this.debug) {
      console.log('[PRSM Tools] Cache cleared');
    }
  }

  /**
   * Get cached tool without API call
   */
  getCachedTool(toolName: string): ToolSpec | null {
    return this.toolsCache.get(toolName) || null;
  }

  /**
   * Get cached categories without API call
   */
  getCachedCategories(): string[] | null {
    return this.categoriesCache;
  }

  /**
   * Get cached active executions
   */
  getCachedActiveExecutions(): AsyncToolExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  /**
   * Clean up completed executions from cache
   */
  cleanupExecutions(): void {
    const cutoff = Date.now() - 3600000; // 1 hour

    for (const [id, execution] of this.activeExecutions.entries()) {
      if (execution.status === 'completed' || execution.status === 'failed') {
        const completedTime = execution.completedAt ? 
          new Date(execution.completedAt).getTime() : 
          Date.now();
        
        if (completedTime < cutoff) {
          this.activeExecutions.delete(id);
        }
      }
    }

    if (this.debug) {
      console.log('[PRSM Tools] Execution cache cleaned up');
    }
  }

  /**
   * Format tool cost for display
   */
  static formatCost(cost: number): string {
    return `${cost.toFixed(4)} FTNS`;
  }

  /**
   * Format execution duration
   */
  static formatDuration(durationMs: number): string {
    if (durationMs < 1000) {
      return `${durationMs}ms`;
    } else if (durationMs < 60000) {
      return `${(durationMs / 1000).toFixed(1)}s`;
    } else {
      return `${(durationMs / 60000).toFixed(1)}m`;
    }
  }
}