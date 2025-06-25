/**
 * PRSM SDK FTNS Token Manager
 * Handles FTNS (Fractal Token Network System) token operations, balances, and Web3 integration
 */

import {
  FTNSBalance,
  TransactionInfo,
  PaymentRequest,
  NetworkType,
  APIResponse,
  PaginatedResponse,
  RequestOptions,
} from './types';
import {
  InsufficientFundsError,
  PaymentError,
  Web3TransactionError,
  NetworkError,
  ValidationError,
  ResourceNotFoundError,
  toPRSMError,
} from './errors';

// ============================================================================
// FTNS MANAGER CONFIGURATION
// ============================================================================

export interface FTNSManagerConfig {
  /** Base URL for FTNS API endpoints */
  baseUrl: string;
  /** Authentication headers function */
  getAuthHeaders: () => Promise<Record<string, string>>;
  /** Web3 network type */
  network?: NetworkType;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Debug logging */
  debug?: boolean;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Auto-refresh balance interval in seconds */
  autoRefreshInterval?: number;
}

// ============================================================================
// TRANSACTION FILTERS
// ============================================================================

export interface TransactionFilters {
  /** Transaction type filter */
  type?: 'transfer' | 'purchase' | 'earn' | 'spend' | 'stake' | 'unstake';
  /** Start date for filtering */
  startDate?: Date;
  /** End date for filtering */
  endDate?: Date;
  /** Minimum amount */
  minAmount?: number;
  /** Maximum amount */
  maxAmount?: number;
  /** Transaction status */
  status?: 'pending' | 'confirmed' | 'failed';
  /** Address filter (from or to) */
  address?: string;
  /** Page limit */
  limit?: number;
  /** Page offset */
  offset?: number;
}

// ============================================================================
// STAKING INFORMATION
// ============================================================================

export interface StakingInfo {
  /** Total staked amount */
  totalStaked: number;
  /** Available for unstaking */
  availableForUnstaking: number;
  /** Pending unstaking */
  pendingUnstaking: number;
  /** Annual percentage yield */
  apy: number;
  /** Rewards earned */
  rewardsEarned: number;
  /** Next reward date */
  nextRewardDate: string;
  /** Staking pools */
  pools: StakingPool[];
}

export interface StakingPool {
  /** Pool ID */
  id: string;
  /** Pool name */
  name: string;
  /** Pool description */
  description: string;
  /** Minimum stake amount */
  minStake: number;
  /** Maximum stake amount */
  maxStake?: number;
  /** Annual percentage yield */
  apy: number;
  /** Lock period in days */
  lockPeriodDays: number;
  /** Pool capacity */
  capacity: number;
  /** Currently staked in pool */
  currentStaked: number;
  /** User's stake in this pool */
  userStake: number;
  /** Pool status */
  status: 'active' | 'paused' | 'full';
}

// ============================================================================
// WALLET OPERATIONS
// ============================================================================

export interface WalletInfo {
  /** Wallet address */
  address: string;
  /** Network type */
  network: NetworkType;
  /** Is wallet connected */
  isConnected: boolean;
  /** Wallet type/provider */
  provider?: string;
  /** ENS name if available */
  ensName?: string;
}

export interface TransferRequest {
  /** Recipient address */
  toAddress: string;
  /** Amount to transfer */
  amount: number;
  /** Optional note/memo */
  note?: string;
  /** Gas price override */
  gasPrice?: number;
  /** Gas limit override */
  gasLimit?: number;
}

// ============================================================================
// FTNS MANAGER CLASS
// ============================================================================

/**
 * Manages FTNS token operations and Web3 integration
 */
export class FTNSManager {
  private readonly baseUrl: string;
  private readonly getAuthHeaders: () => Promise<Record<string, string>>;
  private readonly network: NetworkType;
  private readonly headers: Record<string, string>;
  private readonly debug: boolean;
  private readonly timeout: number;

  private cachedBalance: FTNSBalance | null = null;
  private balanceRefreshInterval: NodeJS.Timeout | null = null;
  private lastBalanceUpdate: Date | null = null;

  constructor(config: FTNSManagerConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.getAuthHeaders = config.getAuthHeaders;
    this.network = config.network || NetworkType.MAINNET;
    this.headers = config.headers || {};
    this.debug = config.debug ?? false;
    this.timeout = config.timeout ?? 30000;

    // Set up auto-refresh if configured
    if (config.autoRefreshInterval) {
      this.startAutoRefresh(config.autoRefreshInterval);
    }
  }

  // ============================================================================
  // BALANCE MANAGEMENT
  // ============================================================================

  /**
   * Get current FTNS balance
   */
  async getBalance(forceRefresh = false): Promise<FTNSBalance> {
    if (!forceRefresh && this.cachedBalance && this.isBalanceFresh()) {
      return this.cachedBalance;
    }

    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Fetching FTNS balance');
      }

      const headers = await this.getAuthHeaders();
      const balance = await this.makeRequest<FTNSBalance>('/api/v1/web3/balance', {
        method: 'GET',
        headers: { ...headers, ...this.headers },
      });

      this.cachedBalance = balance;
      this.lastBalanceUpdate = new Date();

      if (this.debug) {
        console.log('[PRSM FTNS] Balance fetched:', balance.totalBalance, 'FTNS');
      }

      return balance;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get FTNS balance');
    }
  }

  /**
   * Check if balance cache is fresh (within 30 seconds)
   */
  private isBalanceFresh(): boolean {
    if (!this.lastBalanceUpdate) return false;
    const age = Date.now() - this.lastBalanceUpdate.getTime();
    return age < 30000; // 30 seconds
  }

  /**
   * Start auto-refresh of balance
   */
  startAutoRefresh(intervalSeconds: number): void {
    this.stopAutoRefresh();
    
    this.balanceRefreshInterval = setInterval(async () => {
      try {
        await this.getBalance(true);
      } catch (error) {
        if (this.debug) {
          console.warn('[PRSM FTNS] Auto-refresh failed:', error);
        }
      }
    }, intervalSeconds * 1000);

    if (this.debug) {
      console.log(`[PRSM FTNS] Auto-refresh started (${intervalSeconds}s interval)`);
    }
  }

  /**
   * Stop auto-refresh of balance
   */
  stopAutoRefresh(): void {
    if (this.balanceRefreshInterval) {
      clearInterval(this.balanceRefreshInterval);
      this.balanceRefreshInterval = null;
      
      if (this.debug) {
        console.log('[PRSM FTNS] Auto-refresh stopped');
      }
    }
  }

  /**
   * Check if user has sufficient balance
   */
  async hasSufficientBalance(amount: number): Promise<boolean> {
    const balance = await this.getBalance();
    return balance.availableBalance >= amount;
  }

  /**
   * Ensure user has sufficient balance or throw error
   */
  async ensureSufficientBalance(amount: number): Promise<void> {
    const balance = await this.getBalance();
    if (balance.availableBalance < amount) {
      throw new InsufficientFundsError(amount, balance.availableBalance);
    }
  }

  // ============================================================================
  // TRANSACTION MANAGEMENT
  // ============================================================================

  /**
   * Get transaction history
   */
  async getTransactions(
    filters: TransactionFilters = {}
  ): Promise<PaginatedResponse<TransactionInfo>> {
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Fetching transaction history');
      }

      const params = new URLSearchParams();
      
      if (filters.type) params.append('type', filters.type);
      if (filters.startDate) params.append('start_date', filters.startDate.toISOString());
      if (filters.endDate) params.append('end_date', filters.endDate.toISOString());
      if (filters.minAmount) params.append('min_amount', filters.minAmount.toString());
      if (filters.maxAmount) params.append('max_amount', filters.maxAmount.toString());
      if (filters.status) params.append('status', filters.status);
      if (filters.address) params.append('address', filters.address);
      if (filters.limit) params.append('limit', filters.limit.toString());
      if (filters.offset) params.append('offset', filters.offset.toString());

      const headers = await this.getAuthHeaders();
      return await this.makeRequest<PaginatedResponse<TransactionInfo>>(
        `/api/v1/web3/transactions?${params}`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get transaction history');
    }
  }

  /**
   * Get specific transaction details
   */
  async getTransaction(txHash: string): Promise<TransactionInfo> {
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Fetching transaction:', txHash);
      }

      const headers = await this.getAuthHeaders();
      return await this.makeRequest<TransactionInfo>(
        `/api/v1/web3/transactions/${txHash}`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get transaction details');
    }
  }

  /**
   * Wait for transaction confirmation
   */
  async waitForTransaction(
    txHash: string,
    timeoutMs: number = 300000, // 5 minutes
    pollIntervalMs: number = 5000 // 5 seconds
  ): Promise<TransactionInfo> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeoutMs) {
      try {
        const tx = await this.getTransaction(txHash);
        
        if (tx.status === 'confirmed') {
          if (this.debug) {
            console.log('[PRSM FTNS] Transaction confirmed:', txHash);
          }
          return tx;
        }
        
        if (tx.status === 'failed') {
          throw new Web3TransactionError(
            'Transaction failed',
            txHash,
            tx.gasUsed
          );
        }
        
        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
      } catch (error) {
        if (error instanceof ResourceNotFoundError) {
          // Transaction not found yet, continue polling
          await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
          continue;
        }
        throw error;
      }
    }
    
    throw new Web3TransactionError(
      'Transaction confirmation timeout',
      txHash
    );
  }

  // ============================================================================
  // TOKEN TRANSFERS
  // ============================================================================

  /**
   * Transfer FTNS tokens to another address
   */
  async transfer(request: TransferRequest): Promise<TransactionInfo> {
    this.validateTransferRequest(request);
    
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Initiating transfer:', request.amount, 'FTNS to', request.toAddress);
      }

      // Check balance before transfer
      await this.ensureSufficientBalance(request.amount);

      const headers = await this.getAuthHeaders();
      const transaction = await this.makeRequest<TransactionInfo>(
        '/api/v1/web3/transfer', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: {
            to_address: request.toAddress,
            amount: request.amount,
            note: request.note,
            gas_price: request.gasPrice,
            gas_limit: request.gasLimit,
          },
        }
      );

      // Invalidate balance cache
      this.cachedBalance = null;

      if (this.debug) {
        console.log('[PRSM FTNS] Transfer initiated, tx hash:', transaction.txHash);
      }

      return transaction;
    } catch (error) {
      throw toPRSMError(error, 'Failed to transfer FTNS tokens');
    }
  }

  /**
   * Estimate gas for transfer
   */
  async estimateTransferGas(request: Omit<TransferRequest, 'gasPrice' | 'gasLimit'>): Promise<{
    gasLimit: number;
    gasPrice: number;
    totalCost: number;
  }> {
    this.validateTransferRequest(request);
    
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest('/api/v1/web3/transfer/estimate', {
        method: 'POST',
        headers: { ...headers, ...this.headers },
        body: {
          to_address: request.toAddress,
          amount: request.amount,
        },
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to estimate transfer gas');
    }
  }

  private validateTransferRequest(request: Partial<TransferRequest>): void {
    if (!request.toAddress) {
      throw new ValidationError('Recipient address is required');
    }
    
    if (!request.amount || request.amount <= 0) {
      throw new ValidationError('Transfer amount must be greater than 0');
    }
    
    // Basic address validation (Ethereum format)
    if (!/^0x[a-fA-F0-9]{40}$/.test(request.toAddress)) {
      throw new ValidationError('Invalid recipient address format');
    }
  }

  // ============================================================================
  // FIAT PAYMENTS
  // ============================================================================

  /**
   * Purchase FTNS tokens with fiat currency
   */
  async purchaseTokens(request: PaymentRequest): Promise<{
    transactionId: string;
    status: string;
    ftnsAmount: number;
    paymentUrl?: string;
  }> {
    this.validatePaymentRequest(request);
    
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Purchasing tokens:', request.amountUsd, 'USD');
      }

      const headers = await this.getAuthHeaders();
      const result = await this.makeRequest('/api/v1/payments/purchase', {
        method: 'POST',
        headers: { ...headers, ...this.headers },
        body: {
          amount_usd: request.amountUsd,
          payment_method: request.paymentMethod,
          payment_token: request.paymentToken,
          billing_info: request.billingInfo,
        },
      });

      if (this.debug) {
        console.log('[PRSM FTNS] Purchase initiated:', result.transactionId);
      }

      return result;
    } catch (error) {
      throw toPRSMError(error, 'Failed to purchase FTNS tokens');
    }
  }

  /**
   * Get purchase status
   */
  async getPurchaseStatus(transactionId: string): Promise<{
    status: 'pending' | 'completed' | 'failed';
    ftnsAmount?: number;
    error?: string;
  }> {
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest(`/api/v1/payments/purchase/${transactionId}`, {
        method: 'GET',
        headers: { ...headers, ...this.headers },
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get purchase status');
    }
  }

  /**
   * Get current FTNS exchange rates
   */
  async getExchangeRates(): Promise<{
    usdPerFtns: number;
    ftnsPerUsd: number;
    lastUpdate: string;
  }> {
    try {
      return await this.makeRequest('/api/v1/payments/rates', {
        method: 'GET',
        headers: this.headers,
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get exchange rates');
    }
  }

  private validatePaymentRequest(request: PaymentRequest): void {
    if (!request.amountUsd || request.amountUsd <= 0) {
      throw new ValidationError('Payment amount must be greater than 0');
    }
    
    if (!request.paymentMethod) {
      throw new ValidationError('Payment method is required');
    }
    
    if (!request.paymentToken) {
      throw new ValidationError('Payment token is required');
    }
    
    const validMethods = ['stripe', 'paypal', 'crypto'];
    if (!validMethods.includes(request.paymentMethod)) {
      throw new ValidationError(`Invalid payment method: ${request.paymentMethod}`);
    }
  }

  // ============================================================================
  // STAKING OPERATIONS
  // ============================================================================

  /**
   * Get staking information
   */
  async getStakingInfo(): Promise<StakingInfo> {
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Fetching staking info');
      }

      const headers = await this.getAuthHeaders();
      return await this.makeRequest<StakingInfo>('/api/v1/web3/staking', {
        method: 'GET',
        headers: { ...headers, ...this.headers },
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get staking information');
    }
  }

  /**
   * Stake FTNS tokens
   */
  async stakeTokens(poolId: string, amount: number): Promise<TransactionInfo> {
    if (amount <= 0) {
      throw new ValidationError('Stake amount must be greater than 0');
    }
    
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Staking tokens:', amount, 'FTNS in pool', poolId);
      }

      await this.ensureSufficientBalance(amount);

      const headers = await this.getAuthHeaders();
      const transaction = await this.makeRequest<TransactionInfo>(
        '/api/v1/web3/staking/stake', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: {
            pool_id: poolId,
            amount: amount,
          },
        }
      );

      // Invalidate balance cache
      this.cachedBalance = null;

      if (this.debug) {
        console.log('[PRSM FTNS] Staking initiated, tx hash:', transaction.txHash);
      }

      return transaction;
    } catch (error) {
      throw toPRSMError(error, 'Failed to stake FTNS tokens');
    }
  }

  /**
   * Unstake FTNS tokens
   */
  async unstakeTokens(poolId: string, amount: number): Promise<TransactionInfo> {
    if (amount <= 0) {
      throw new ValidationError('Unstake amount must be greater than 0');
    }
    
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Unstaking tokens:', amount, 'FTNS from pool', poolId);
      }

      const headers = await this.getAuthHeaders();
      const transaction = await this.makeRequest<TransactionInfo>(
        '/api/v1/web3/staking/unstake', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: {
            pool_id: poolId,
            amount: amount,
          },
        }
      );

      if (this.debug) {
        console.log('[PRSM FTNS] Unstaking initiated, tx hash:', transaction.txHash);
      }

      return transaction;
    } catch (error) {
      throw toPRSMError(error, 'Failed to unstake FTNS tokens');
    }
  }

  /**
   * Claim staking rewards
   */
  async claimRewards(poolId?: string): Promise<TransactionInfo> {
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Claiming rewards', poolId ? `from pool ${poolId}` : 'from all pools');
      }

      const headers = await this.getAuthHeaders();
      const transaction = await this.makeRequest<TransactionInfo>(
        '/api/v1/web3/staking/claim', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: poolId ? { pool_id: poolId } : {},
        }
      );

      // Invalidate balance cache
      this.cachedBalance = null;

      if (this.debug) {
        console.log('[PRSM FTNS] Rewards claimed, tx hash:', transaction.txHash);
      }

      return transaction;
    } catch (error) {
      throw toPRSMError(error, 'Failed to claim staking rewards');
    }
  }

  // ============================================================================
  // WALLET OPERATIONS
  // ============================================================================

  /**
   * Get wallet information
   */
  async getWalletInfo(): Promise<WalletInfo> {
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest<WalletInfo>('/api/v1/web3/wallet', {
        method: 'GET',
        headers: { ...headers, ...this.headers },
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to get wallet information');
    }
  }

  /**
   * Connect external wallet
   */
  async connectWallet(walletAddress: string, signature: string): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Connecting wallet:', walletAddress);
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest('/api/v1/web3/wallet/connect', {
        method: 'POST',
        headers: { ...headers, ...this.headers },
        body: {
          wallet_address: walletAddress,
          signature: signature,
        },
      });

      if (this.debug) {
        console.log('[PRSM FTNS] Wallet connected successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to connect wallet');
    }
  }

  /**
   * Disconnect wallet
   */
  async disconnectWallet(): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM FTNS] Disconnecting wallet');
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest('/api/v1/web3/wallet/disconnect', {
        method: 'POST',
        headers: { ...headers, ...this.headers },
      });

      if (this.debug) {
        console.log('[PRSM FTNS] Wallet disconnected successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to disconnect wallet');
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

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
        
        if (response.status === 402) {
          throw new InsufficientFundsError(
            errorData.required || 0,
            errorData.available || 0,
            errorData.requestId
          );
        }
        
        if (response.status === 400 && errorData.code === 'PAYMENT_ERROR') {
          throw new PaymentError(
            errorData.message,
            errorData.paymentMethod,
            errorData.transactionId,
            errorData.requestId
          );
        }
        
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
      
      if (error instanceof InsufficientFundsError || 
          error instanceof PaymentError ||
          error instanceof NetworkError) {
        throw error;
      }
      
      throw new NetworkError('Network request failed', error as Error);
    }
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.stopAutoRefresh();
    this.cachedBalance = null;
    this.lastBalanceUpdate = null;
    
    if (this.debug) {
      console.log('[PRSM FTNS] Manager destroyed');
    }
  }

  /**
   * Get cached balance without API call
   */
  getCachedBalance(): FTNSBalance | null {
    return this.cachedBalance;
  }

  /**
   * Format FTNS amount for display
   */
  static formatFTNS(amount: number, decimals: number = 4): string {
    return `${amount.toFixed(decimals)} FTNS`;
  }

  /**
   * Parse FTNS amount from string
   */
  static parseFTNS(value: string): number {
    const cleaned = value.replace(/[^0-9.-]/g, '');
    const parsed = parseFloat(cleaned);
    
    if (isNaN(parsed)) {
      throw new ValidationError('Invalid FTNS amount format');
    }
    
    return parsed;
  }
}