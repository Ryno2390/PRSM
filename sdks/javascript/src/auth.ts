/**
 * PRSM SDK Authentication Manager
 * Handles API key management, JWT tokens, and authentication flows
 */

import {
  AuthTokens,
  LoginRequest,
  RegisterRequest,
  UserProfile,
  APIResponse,
  RequestOptions,
} from './types';
import {
  AuthenticationError,
  InvalidAPIKeyError,
  TokenExpiredError,
  NetworkError,
  ValidationError,
  toPRSMError,
} from './errors';

// ============================================================================
// TOKEN STORAGE INTERFACE
// ============================================================================

/**
 * Interface for token storage implementations
 */
export interface TokenStorage {
  getToken(): string | null;
  setToken(token: string): void;
  removeToken(): void;
  getRefreshToken(): string | null;
  setRefreshToken(token: string): void;
  removeRefreshToken(): void;
}

/**
 * Memory-based token storage (default)
 */
export class MemoryTokenStorage implements TokenStorage {
  private accessToken: string | null = null;
  private refreshToken: string | null = null;

  getToken(): string | null {
    return this.accessToken;
  }

  setToken(token: string): void {
    this.accessToken = token;
  }

  removeToken(): void {
    this.accessToken = null;
  }

  getRefreshToken(): string | null {
    return this.refreshToken;
  }

  setRefreshToken(token: string): void {
    this.refreshToken = token;
  }

  removeRefreshToken(): void {
    this.refreshToken = null;
  }
}

/**
 * LocalStorage-based token storage (browser only)
 */
export class LocalStorageTokenStorage implements TokenStorage {
  private readonly accessTokenKey = 'prsm_access_token';
  private readonly refreshTokenKey = 'prsm_refresh_token';

  getToken(): string | null {
    if (typeof localStorage === 'undefined') return null;
    return localStorage.getItem(this.accessTokenKey);
  }

  setToken(token: string): void {
    if (typeof localStorage === 'undefined') return;
    localStorage.setItem(this.accessTokenKey, token);
  }

  removeToken(): void {
    if (typeof localStorage === 'undefined') return;
    localStorage.removeItem(this.accessTokenKey);
  }

  getRefreshToken(): string | null {
    if (typeof localStorage === 'undefined') return null;
    return localStorage.getItem(this.refreshTokenKey);
  }

  setRefreshToken(token: string): void {
    if (typeof localStorage === 'undefined') return;
    localStorage.setItem(this.refreshTokenKey, token);
  }

  removeRefreshToken(): void {
    if (typeof localStorage === 'undefined') return;
    localStorage.removeItem(this.refreshTokenKey);
  }
}

// ============================================================================
// JWT TOKEN UTILITIES
// ============================================================================

/**
 * JWT token payload interface
 */
interface JWTPayload {
  sub: string;
  exp: number;
  iat: number;
  iss?: string;
  aud?: string;
  scope?: string;
  role?: string;
  [key: string]: any;
}

/**
 * Parse JWT token payload (without verification)
 */
function parseJWTPayload(token: string): JWTPayload | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;

    const payload = parts[1];
    const decodedPayload = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(decodedPayload) as JWTPayload;
  } catch {
    return null;
  }
}

/**
 * Check if JWT token is expired
 */
function isTokenExpired(token: string): boolean {
  const payload = parseJWTPayload(token);
  if (!payload || !payload.exp) return true;

  const now = Math.floor(Date.now() / 1000);
  return payload.exp <= now;
}

/**
 * Get token expiration time
 */
function getTokenExpiration(token: string): Date | null {
  const payload = parseJWTPayload(token);
  if (!payload || !payload.exp) return null;

  return new Date(payload.exp * 1000);
}

// ============================================================================
// AUTHENTICATION MANAGER
// ============================================================================

export interface AuthManagerConfig {
  /** Base URL for authentication endpoints */
  baseUrl: string;
  /** Token storage implementation */
  tokenStorage?: TokenStorage;
  /** API key for server-to-server authentication */
  apiKey?: string;
  /** Automatic token refresh */
  autoRefresh?: boolean;
  /** Token refresh threshold in seconds (default: 300) */
  refreshThresholdSeconds?: number;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Debug logging */
  debug?: boolean;
}

/**
 * Manages authentication for PRSM API
 */
export class AuthManager {
  private readonly baseUrl: string;
  private readonly tokenStorage: TokenStorage;
  private readonly apiKey?: string;
  private readonly autoRefresh: boolean;
  private readonly refreshThresholdSeconds: number;
  private readonly headers: Record<string, string>;
  private readonly debug: boolean;

  private currentUser: UserProfile | null = null;
  private refreshPromise: Promise<AuthTokens> | null = null;

  constructor(config: AuthManagerConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.tokenStorage = config.tokenStorage || new MemoryTokenStorage();
    this.apiKey = config.apiKey;
    this.autoRefresh = config.autoRefresh ?? true;
    this.refreshThresholdSeconds = config.refreshThresholdSeconds ?? 300;
    this.headers = config.headers || {};
    this.debug = config.debug ?? false;
  }

  // ============================================================================
  // TOKEN MANAGEMENT
  // ============================================================================

  /**
   * Get current access token
   */
  getToken(): string | null {
    return this.tokenStorage.getToken();
  }

  /**
   * Get current refresh token
   */
  getRefreshToken(): string | null {
    return this.tokenStorage.getRefreshToken();
  }

  /**
   * Set authentication tokens
   */
  setTokens(tokens: AuthTokens): void {
    this.tokenStorage.setToken(tokens.accessToken);
    if (tokens.refreshToken) {
      this.tokenStorage.setRefreshToken(tokens.refreshToken);
    }
    
    if (this.debug) {
      console.log('[PRSM Auth] Tokens set successfully');
    }
  }

  /**
   * Clear authentication tokens
   */
  clearTokens(): void {
    this.tokenStorage.removeToken();
    this.tokenStorage.removeRefreshToken();
    this.currentUser = null;
    
    if (this.debug) {
      console.log('[PRSM Auth] Tokens cleared');
    }
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    const token = this.getToken();
    return token !== null && !isTokenExpired(token);
  }

  /**
   * Check if token needs refresh
   */
  needsRefresh(): boolean {
    const token = this.getToken();
    if (!token) return false;

    const payload = parseJWTPayload(token);
    if (!payload || !payload.exp) return true;

    const now = Math.floor(Date.now() / 1000);
    const expiresIn = payload.exp - now;
    
    return expiresIn <= this.refreshThresholdSeconds;
  }

  /**
   * Get token expiration date
   */
  getTokenExpiration(): Date | null {
    const token = this.getToken();
    return token ? getTokenExpiration(token) : null;
  }

  // ============================================================================
  // AUTHENTICATION HEADERS
  // ============================================================================

  /**
   * Get authentication headers for API requests
   */
  async getAuthHeaders(): Promise<Record<string, string>> {
    const headers: Record<string, string> = { ...this.headers };

    // Use API key if available (takes precedence)
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
      return headers;
    }

    // Get access token, refreshing if necessary
    const token = await this.getValidToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
  }

  /**
   * Get valid access token, refreshing if necessary
   */
  async getValidToken(): Promise<string | null> {
    let token = this.getToken();
    
    if (!token) {
      return null;
    }

    // Check if token is expired
    if (isTokenExpired(token)) {
      if (!this.autoRefresh) {
        throw new TokenExpiredError();
      }
      
      token = await this.refreshToken();
    }
    // Check if token needs refresh soon
    else if (this.autoRefresh && this.needsRefresh()) {
      try {
        token = await this.refreshToken();
      } catch (error) {
        // If refresh fails, continue with current token if still valid
        if (this.debug) {
          console.warn('[PRSM Auth] Token refresh failed:', error);
        }
      }
    }

    return token;
  }

  // ============================================================================
  // AUTHENTICATION FLOWS
  // ============================================================================

  /**
   * Login with username and password
   */
  async login(credentials: LoginRequest): Promise<UserProfile> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Attempting login for:', credentials.username);
      }

      const response = await this.makeRequest<{
        tokens: AuthTokens;
        user: UserProfile;
      }>('/api/v1/auth/login', {
        method: 'POST',
        body: credentials,
      });

      this.setTokens(response.tokens);
      this.currentUser = response.user;

      if (this.debug) {
        console.log('[PRSM Auth] Login successful for user:', response.user.username);
      }

      return response.user;
    } catch (error) {
      throw toPRSMError(error, 'Login failed');
    }
  }

  /**
   * Register new user account
   */
  async register(userData: RegisterRequest): Promise<UserProfile> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Attempting registration for:', userData.username);
      }

      const response = await this.makeRequest<{
        tokens: AuthTokens;
        user: UserProfile;
      }>('/api/v1/auth/register', {
        method: 'POST',
        body: userData,
      });

      this.setTokens(response.tokens);
      this.currentUser = response.user;

      if (this.debug) {
        console.log('[PRSM Auth] Registration successful for user:', response.user.username);
      }

      return response.user;
    } catch (error) {
      throw toPRSMError(error, 'Registration failed');
    }
  }

  /**
   * Logout current user
   */
  async logout(): Promise<void> {
    try {
      // Attempt to revoke token on server
      const token = this.getToken();
      if (token) {
        await this.makeRequest('/api/v1/auth/logout', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        });
      }
    } catch (error) {
      // Continue with local logout even if server logout fails
      if (this.debug) {
        console.warn('[PRSM Auth] Server logout failed:', error);
      }
    } finally {
      this.clearTokens();
      
      if (this.debug) {
        console.log('[PRSM Auth] Logout completed');
      }
    }
  }

  /**
   * Refresh access token
   */
  async refreshToken(): Promise<string> {
    // Prevent multiple concurrent refresh attempts
    if (this.refreshPromise) {
      const tokens = await this.refreshPromise;
      return tokens.accessToken;
    }

    this.refreshPromise = this.performTokenRefresh();
    
    try {
      const tokens = await this.refreshPromise;
      return tokens.accessToken;
    } finally {
      this.refreshPromise = null;
    }
  }

  private async performTokenRefresh(): Promise<AuthTokens> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      throw new AuthenticationError('No refresh token available');
    }

    try {
      if (this.debug) {
        console.log('[PRSM Auth] Refreshing access token');
      }

      const response = await this.makeRequest<AuthTokens>('/api/v1/auth/refresh', {
        method: 'POST',
        headers: { Authorization: `Bearer ${refreshToken}` },
      });

      this.setTokens(response);

      if (this.debug) {
        console.log('[PRSM Auth] Token refresh successful');
      }

      return response;
    } catch (error) {
      // Clear tokens if refresh fails
      this.clearTokens();
      throw toPRSMError(error, 'Token refresh failed');
    }
  }

  // ============================================================================
  // USER PROFILE
  // ============================================================================

  /**
   * Get current user profile
   */
  async getCurrentUser(): Promise<UserProfile> {
    if (this.currentUser) {
      return this.currentUser;
    }

    try {
      if (this.debug) {
        console.log('[PRSM Auth] Fetching current user profile');
      }

      const headers = await this.getAuthHeaders();
      const user = await this.makeRequest<UserProfile>('/api/v1/users/me', {
        method: 'GET',
        headers,
      });

      this.currentUser = user;
      return user;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get current user');
    }
  }

  /**
   * Update current user profile
   */
  async updateProfile(updates: Partial<UserProfile>): Promise<UserProfile> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Updating user profile');
      }

      const headers = await this.getAuthHeaders();
      const user = await this.makeRequest<UserProfile>('/api/v1/users/me', {
        method: 'PUT',
        headers,
        body: updates,
      });

      this.currentUser = user;
      return user;
    } catch (error) {
      throw toPRSMError(error, 'Failed to update profile');
    }
  }

  /**
   * Change user password
   */
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Changing user password');
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest('/api/v1/users/me/password', {
        method: 'POST',
        headers,
        body: {
          current_password: currentPassword,
          new_password: newPassword,
        },
      });

      if (this.debug) {
        console.log('[PRSM Auth] Password changed successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to change password');
    }
  }

  // ============================================================================
  // API KEY MANAGEMENT
  // ============================================================================

  /**
   * Generate new API key
   */
  async generateAPIKey(
    name: string,
    permissions: string[] = [],
    expiresInDays?: number
  ): Promise<{ key: string; id: string }> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Generating new API key:', name);
      }

      const headers = await this.getAuthHeaders();
      const response = await this.makeRequest<{ key: string; id: string }>(
        '/api/v1/security/api-keys',
        {
          method: 'POST',
          headers,
          body: {
            name,
            permissions,
            expires_in_days: expiresInDays,
          },
        }
      );

      if (this.debug) {
        console.log('[PRSM Auth] API key generated successfully');
      }

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to generate API key');
    }
  }

  /**
   * List user's API keys
   */
  async listAPIKeys(): Promise<Array<{
    id: string;
    name: string;
    permissions: string[];
    created: string;
    expires?: string;
    lastUsed?: string;
  }>> {
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest('/api/v1/security/api-keys', {
        method: 'GET',
        headers,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to list API keys');
    }
  }

  /**
   * Revoke API key
   */
  async revokeAPIKey(keyId: string): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Revoking API key:', keyId);
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest(`/api/v1/security/api-keys/${keyId}`, {
        method: 'DELETE',
        headers,
      });

      if (this.debug) {
        console.log('[PRSM Auth] API key revoked successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to revoke API key');
    }
  }

  // ============================================================================
  // TWO-FACTOR AUTHENTICATION
  // ============================================================================

  /**
   * Enable two-factor authentication
   */
  async enableTwoFactor(): Promise<{ qrCode: string; backupCodes: string[] }> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Enabling two-factor authentication');
      }

      const headers = await this.getAuthHeaders();
      return await this.makeRequest('/api/v1/security/2fa/enable', {
        method: 'POST',
        headers,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to enable two-factor authentication');
    }
  }

  /**
   * Disable two-factor authentication
   */
  async disableTwoFactor(code: string): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Auth] Disabling two-factor authentication');
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest('/api/v1/security/2fa/disable', {
        method: 'POST',
        headers,
        body: { code },
      });

      if (this.debug) {
        console.log('[PRSM Auth] Two-factor authentication disabled');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to disable two-factor authentication');
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Make authenticated HTTP request
   */
  private async makeRequest<T = any>(
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

    if (options.timeout) {
      const controller = new AbortController();
      requestOptions.signal = controller.signal;
      setTimeout(() => controller.abort(), options.timeout);
    }

    try {
      const response = await fetch(url, requestOptions);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new AuthenticationError(
          errorData.message || `HTTP ${response.status}: ${response.statusText}`,
          errorData.requestId,
          { statusCode: response.status, ...errorData.details }
        );
      }

      const data = await response.json();
      return data.data || data; // Handle both APIResponse<T> and direct T
    } catch (error) {
      if (error instanceof AuthenticationError) {
        throw error;
      }
      throw new NetworkError('Network request failed', error as Error);
    }
  }

  /**
   * Validate authentication configuration
   */
  validateConfig(): void {
    if (!this.baseUrl) {
      throw new ValidationError('Base URL is required');
    }

    if (!this.apiKey && !this.getToken()) {
      throw new ValidationError('Either API key or access token is required');
    }
  }

  /**
   * Get authentication status summary
   */
  getAuthStatus(): {
    isAuthenticated: boolean;
    hasApiKey: boolean;
    tokenExpiration: Date | null;
    needsRefresh: boolean;
    currentUser: UserProfile | null;
  } {
    return {
      isAuthenticated: this.isAuthenticated(),
      hasApiKey: !!this.apiKey,
      tokenExpiration: this.getTokenExpiration(),
      needsRefresh: this.needsRefresh(),
      currentUser: this.currentUser,
    };
  }
}