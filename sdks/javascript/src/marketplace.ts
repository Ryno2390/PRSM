/**
 * PRSM SDK Marketplace Manager
 * Handles model marketplace operations, browsing, renting, and submissions
 */

import {
  MarketplaceQuery,
  MarketplaceModel,
  ModelInfo,
  ModelCategory,
  ModelProvider,
  SubscriptionOption,
  PaginatedResponse,
  RequestOptions,
} from './types';
import {
  ResourceNotFoundError,
  InsufficientFundsError,
  ValidationError,
  NetworkError,
  toPRSMError,
} from './errors';

// ============================================================================
// MARKETPLACE MANAGER CONFIGURATION
// ============================================================================

export interface MarketplaceManagerConfig {
  /** Base URL for marketplace API endpoints */
  baseUrl: string;
  /** Authentication headers function */
  getAuthHeaders: () => Promise<Record<string, string>>;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Debug logging */
  debug?: boolean;
  /** Request timeout in milliseconds */
  timeout?: number;
}

// ============================================================================
// MODEL SUBMISSION TYPES
// ============================================================================

export interface ModelSubmission {
  /** Model name */
  name: string;
  /** Model description */
  description: string;
  /** Model category */
  category: ModelCategory;
  /** Model file (IPFS hash or upload) */
  modelFile: string | File;
  /** Model configuration file */
  configFile?: string | File;
  /** Pricing configuration */
  pricing: {
    ftnsPerRequest: number;
    revenueShare: number; // 0.0 to 1.0
    bulkDiscount?: number;
    subscriptionOptions?: SubscriptionOption[];
  };
  /** Model tags */
  tags: string[];
  /** Capabilities */
  capabilities: string[];
  /** Technical specifications */
  specifications?: {
    parameters?: string;
    diskSize?: string;
    contextWindow?: number;
    maxTokens?: number;
    languages?: string[];
  };
  /** Demo/example inputs */
  examples?: Array<{
    input: string;
    output: string;
    description?: string;
  }>;
  /** License information */
  license?: string;
  /** Documentation URL */
  documentationUrl?: string;
}

// ============================================================================
// RENTAL TYPES
// ============================================================================

export interface ModelRental {
  /** Rental ID */
  id: string;
  /** Model information */
  model: ModelInfo;
  /** Rental duration in hours */
  durationHours: number;
  /** Maximum requests included */
  maxRequests: number;
  /** Cost in FTNS */
  cost: number;
  /** Rental start time */
  startTime: string;
  /** Rental end time */
  endTime: string;
  /** Requests used */
  requestsUsed: number;
  /** Rental status */
  status: 'active' | 'expired' | 'cancelled';
  /** Access token for the rental */
  accessToken?: string;
}

export interface RentalRequest {
  /** Duration in hours */
  durationHours: number;
  /** Maximum requests */
  maxRequests: number;
  /** Auto-renewal */
  autoRenew?: boolean;
}

// ============================================================================
// REVIEW TYPES
// ============================================================================

export interface ModelReview {
  /** Review ID */
  id: string;
  /** Model ID */
  modelId: string;
  /** Reviewer information */
  reviewer: {
    username: string;
    verified: boolean;
  };
  /** Rating (1-5) */
  rating: number;
  /** Review title */
  title: string;
  /** Review content */
  content: string;
  /** Review timestamp */
  createdAt: string;
  /** Helpful votes */
  helpfulVotes: number;
  /** Response from model creator */
  response?: {
    content: string;
    createdAt: string;
  };
}

export interface ReviewSubmission {
  /** Rating (1-5) */
  rating: number;
  /** Review title */
  title: string;
  /** Review content */
  content: string;
}

// ============================================================================
// MARKETPLACE FILTERS
// ============================================================================

export interface MarketplaceFilters extends MarketplaceQuery {
  /** Price range */
  priceRange?: {
    min: number;
    max: number;
  };
  /** Rating range */
  ratingRange?: {
    min: number;
    max: number;
  };
  /** Model size filter */
  modelSize?: 'small' | 'medium' | 'large' | 'xlarge';
  /** Language support */
  languages?: string[];
  /** Has fine-tuning support */
  fineTuningSupported?: boolean;
  /** Verified creators only */
  verifiedOnly?: boolean;
  /** Date range */
  dateRange?: {
    start: Date;
    end: Date;
  };
}

// ============================================================================
// MARKETPLACE MANAGER CLASS
// ============================================================================

/**
 * Manages PRSM model marketplace operations
 */
export class MarketplaceManager {
  private readonly baseUrl: string;
  private readonly getAuthHeaders: () => Promise<Record<string, string>>;
  private readonly headers: Record<string, string>;
  private readonly debug: boolean;
  private readonly timeout: number;

  private featuredModelsCache: MarketplaceModel[] | null = null;
  private categoriesCache: string[] | null = null;
  private lastCacheUpdate: Date | null = null;

  constructor(config: MarketplaceManagerConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.getAuthHeaders = config.getAuthHeaders;
    this.headers = config.headers || {};
    this.debug = config.debug ?? false;
    this.timeout = config.timeout ?? 30000;
  }

  // ============================================================================
  // MODEL BROWSING
  // ============================================================================

  /**
   * Browse marketplace models
   */
  async browseModels(
    filters: MarketplaceFilters = {}
  ): Promise<PaginatedResponse<MarketplaceModel>> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Browsing models with filters:', filters);
      }

      const params = this.buildQueryParams(filters);
      const headers = await this.getAuthHeaders();
      
      return await this.makeRequest<PaginatedResponse<MarketplaceModel>>(
        `/api/v1/marketplace/models?${params}`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to browse marketplace models');
    }
  }

  /**
   * Get featured models
   */
  async getFeaturedModels(forceRefresh = false): Promise<MarketplaceModel[]> {
    if (!forceRefresh && this.featuredModelsCache && this.isCacheFresh()) {
      return this.featuredModelsCache;
    }

    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Fetching featured models');
      }

      const response = await this.browseModels({
        featured: true,
        limit: 20,
        sortBy: 'rating',
        sortOrder: 'desc',
      });

      this.featuredModelsCache = response.data;
      this.lastCacheUpdate = new Date();

      return response.data;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get featured models');
    }
  }

  /**
   * Search models by text query
   */
  async searchModels(
    query: string,
    filters: Omit<MarketplaceFilters, 'query'> = {}
  ): Promise<PaginatedResponse<MarketplaceModel>> {
    return this.browseModels({ ...filters, query });
  }

  /**
   * Get models by category
   */
  async getModelsByCategory(
    category: ModelCategory,
    filters: Omit<MarketplaceFilters, 'category'> = {}
  ): Promise<PaginatedResponse<MarketplaceModel>> {
    return this.browseModels({ ...filters, category });
  }

  /**
   * Get models by provider
   */
  async getModelsByProvider(
    provider: ModelProvider,
    filters: Omit<MarketplaceFilters, 'provider'> = {}
  ): Promise<PaginatedResponse<MarketplaceModel>> {
    return this.browseModels({ ...filters, provider });
  }

  /**
   * Get available categories
   */
  async getCategories(forceRefresh = false): Promise<string[]> {
    if (!forceRefresh && this.categoriesCache && this.isCacheFresh()) {
      return this.categoriesCache;
    }

    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Fetching categories');
      }

      const response = await this.makeRequest<{ categories: string[] }>(
        '/api/v1/marketplace/categories', {
          method: 'GET',
          headers: this.headers,
        }
      );

      this.categoriesCache = response.categories;
      this.lastCacheUpdate = new Date();

      return response.categories;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get marketplace categories');
    }
  }

  // ============================================================================
  // MODEL DETAILS
  // ============================================================================

  /**
   * Get detailed model information
   */
  async getModel(modelId: string): Promise<MarketplaceModel> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Fetching model details:', modelId);
      }

      return await this.makeRequest<MarketplaceModel>(
        `/api/v1/marketplace/models/${modelId}`, {
          method: 'GET',
          headers: this.headers,
        }
      );
    } catch (error) {
      if (error instanceof NetworkError && error.statusCode === 404) {
        throw new ResourceNotFoundError('Model', modelId);
      }
      throw toPRSMError(error, 'Failed to get model details');
    }
  }

  /**
   * Get model usage statistics
   */
  async getModelStats(modelId: string): Promise<{
    totalDownloads: number;
    activeRentals: number;
    averageRating: number;
    totalReviews: number;
    revenueGenerated: number;
    popularityRank: number;
  }> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Fetching model stats:', modelId);
      }

      return await this.makeRequest(
        `/api/v1/marketplace/models/${modelId}/stats`, {
          method: 'GET',
          headers: this.headers,
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get model statistics');
    }
  }

  /**
   * Check model availability
   */
  async checkModelAvailability(modelId: string): Promise<{
    available: boolean;
    reason?: string;
    nextAvailable?: string;
  }> {
    try {
      return await this.makeRequest(
        `/api/v1/marketplace/models/${modelId}/availability`, {
          method: 'GET',
          headers: this.headers,
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to check model availability');
    }
  }

  // ============================================================================
  // MODEL RENTAL
  // ============================================================================

  /**
   * Rent a model
   */
  async rentModel(modelId: string, request: RentalRequest): Promise<ModelRental> {
    this.validateRentalRequest(request);
    
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Renting model:', modelId, request);
      }

      const headers = await this.getAuthHeaders();
      const rental = await this.makeRequest<ModelRental>(
        `/api/v1/marketplace/models/${modelId}/rent`, {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: {
            duration_hours: request.durationHours,
            max_requests: request.maxRequests,
            auto_renew: request.autoRenew,
          },
        }
      );

      if (this.debug) {
        console.log('[PRSM Marketplace] Model rented successfully:', rental.id);
      }

      return rental;
    } catch (error) {
      if (error instanceof NetworkError && error.statusCode === 402) {
        throw new InsufficientFundsError(0, 0); // Server should provide details
      }
      throw toPRSMError(error, 'Failed to rent model');
    }
  }

  /**
   * Get rental pricing estimate
   */
  async getRentalPricing(modelId: string, request: RentalRequest): Promise<{
    baseCost: number;
    discounts: number;
    totalCost: number;
    breakdown: Array<{
      item: string;
      cost: number;
    }>;
  }> {
    this.validateRentalRequest(request);
    
    try {
      return await this.makeRequest(
        `/api/v1/marketplace/models/${modelId}/rent/pricing`, {
          method: 'POST',
          headers: this.headers,
          body: {
            duration_hours: request.durationHours,
            max_requests: request.maxRequests,
          },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get rental pricing');
    }
  }

  /**
   * Get user's active rentals
   */
  async getActiveRentals(): Promise<ModelRental[]> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Fetching active rentals');
      }

      const headers = await this.getAuthHeaders();
      const response = await this.makeRequest<{ rentals: ModelRental[] }>(
        '/api/v1/marketplace/rentals', {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );

      return response.rentals;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get active rentals');
    }
  }

  /**
   * Get rental details
   */
  async getRental(rentalId: string): Promise<ModelRental> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Fetching rental details:', rentalId);
      }

      const headers = await this.getAuthHeaders();
      return await this.makeRequest<ModelRental>(
        `/api/v1/marketplace/rentals/${rentalId}`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get rental details');
    }
  }

  /**
   * Cancel rental
   */
  async cancelRental(rentalId: string): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Cancelling rental:', rentalId);
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest(`/api/v1/marketplace/rentals/${rentalId}/cancel`, {
        method: 'POST',
        headers: { ...headers, ...this.headers },
      });

      if (this.debug) {
        console.log('[PRSM Marketplace] Rental cancelled successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to cancel rental');
    }
  }

  /**
   * Extend rental
   */
  async extendRental(
    rentalId: string,
    additionalHours: number,
    additionalRequests?: number
  ): Promise<ModelRental> {
    if (additionalHours <= 0) {
      throw new ValidationError('Additional hours must be greater than 0');
    }
    
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Extending rental:', rentalId);
      }

      const headers = await this.getAuthHeaders();
      const rental = await this.makeRequest<ModelRental>(
        `/api/v1/marketplace/rentals/${rentalId}/extend`, {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: {
            additional_hours: additionalHours,
            additional_requests: additionalRequests,
          },
        }
      );

      if (this.debug) {
        console.log('[PRSM Marketplace] Rental extended successfully');
      }

      return rental;
    } catch (error) {
      throw toPRSMError(error, 'Failed to extend rental');
    }
  }

  // ============================================================================
  // MODEL SUBMISSION
  // ============================================================================

  /**
   * Submit model to marketplace
   */
  async submitModel(submission: ModelSubmission): Promise<{
    modelId: string;
    status: 'pending_review' | 'approved' | 'rejected';
    reviewEstimate: string;
  }> {
    this.validateModelSubmission(submission);
    
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Submitting model:', submission.name);
      }

      const headers = await this.getAuthHeaders();
      
      // Handle file uploads if necessary
      const formData = await this.prepareSubmissionData(submission);
      
      const result = await this.makeRequest(
        '/api/v1/marketplace/models', {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: formData,
        }
      );

      if (this.debug) {
        console.log('[PRSM Marketplace] Model submitted successfully:', result.modelId);
      }

      return result;
    } catch (error) {
      throw toPRSMError(error, 'Failed to submit model to marketplace');
    }
  }

  /**
   * Update submitted model
   */
  async updateModel(
    modelId: string,
    updates: Partial<ModelSubmission>
  ): Promise<void> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Updating model:', modelId);
      }

      const headers = await this.getAuthHeaders();
      await this.makeRequest(`/api/v1/marketplace/models/${modelId}`, {
        method: 'PUT',
        headers: { ...headers, ...this.headers },
        body: updates,
      });

      if (this.debug) {
        console.log('[PRSM Marketplace] Model updated successfully');
      }
    } catch (error) {
      throw toPRSMError(error, 'Failed to update model');
    }
  }

  /**
   * Get submission status
   */
  async getSubmissionStatus(modelId: string): Promise<{
    status: 'pending_review' | 'approved' | 'rejected' | 'published';
    reviewComments?: string;
    publishedAt?: string;
    rejectionReason?: string;
  }> {
    try {
      const headers = await this.getAuthHeaders();
      return await this.makeRequest(
        `/api/v1/marketplace/models/${modelId}/status`, {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get submission status');
    }
  }

  /**
   * Get user's submitted models
   */
  async getMyModels(): Promise<MarketplaceModel[]> {
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Fetching user models');
      }

      const headers = await this.getAuthHeaders();
      const response = await this.makeRequest<{ models: MarketplaceModel[] }>(
        '/api/v1/marketplace/my-models', {
          method: 'GET',
          headers: { ...headers, ...this.headers },
        }
      );

      return response.models;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get user models');
    }
  }

  // ============================================================================
  // REVIEWS
  // ============================================================================

  /**
   * Get model reviews
   */
  async getModelReviews(
    modelId: string,
    options: {
      limit?: number;
      offset?: number;
      sortBy?: 'rating' | 'date' | 'helpful';
      sortOrder?: 'asc' | 'desc';
    } = {}
  ): Promise<PaginatedResponse<ModelReview>> {
    try {
      const params = new URLSearchParams();
      if (options.limit) params.append('limit', options.limit.toString());
      if (options.offset) params.append('offset', options.offset.toString());
      if (options.sortBy) params.append('sort_by', options.sortBy);
      if (options.sortOrder) params.append('sort_order', options.sortOrder);

      return await this.makeRequest<PaginatedResponse<ModelReview>>(
        `/api/v1/marketplace/models/${modelId}/reviews?${params}`, {
          method: 'GET',
          headers: this.headers,
        }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get model reviews');
    }
  }

  /**
   * Submit model review
   */
  async submitReview(
    modelId: string,
    review: ReviewSubmission
  ): Promise<ModelReview> {
    this.validateReviewSubmission(review);
    
    try {
      if (this.debug) {
        console.log('[PRSM Marketplace] Submitting review for model:', modelId);
      }

      const headers = await this.getAuthHeaders();
      const result = await this.makeRequest<ModelReview>(
        `/api/v1/marketplace/models/${modelId}/reviews`, {
          method: 'POST',
          headers: { ...headers, ...this.headers },
          body: review,
        }
      );

      if (this.debug) {
        console.log('[PRSM Marketplace] Review submitted successfully');
      }

      return result;
    } catch (error) {
      throw toPRSMError(error, 'Failed to submit review');
    }
  }

  /**
   * Vote on review helpfulness
   */
  async voteOnReview(reviewId: string, helpful: boolean): Promise<void> {
    try {
      const headers = await this.getAuthHeaders();
      await this.makeRequest(`/api/v1/marketplace/reviews/${reviewId}/vote`, {
        method: 'POST',
        headers: { ...headers, ...this.headers },
        body: { helpful },
      });
    } catch (error) {
      throw toPRSMError(error, 'Failed to vote on review');
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private buildQueryParams(filters: MarketplaceFilters): string {
    const params = new URLSearchParams();
    
    if (filters.query) params.append('query', filters.query);
    if (filters.provider) params.append('provider', filters.provider);
    if (filters.category) params.append('category', filters.category);
    if (filters.maxCost) params.append('max_cost', filters.maxCost.toString());
    if (filters.minPerformance) params.append('min_performance', filters.minPerformance.toString());
    if (filters.capabilities) filters.capabilities.forEach(cap => params.append('capabilities', cap));
    if (filters.limit) params.append('limit', filters.limit.toString());
    if (filters.offset) params.append('offset', filters.offset.toString());
    if (filters.sortBy) params.append('sort_by', filters.sortBy);
    if (filters.sortOrder) params.append('sort_order', filters.sortOrder);
    if (filters.featured) params.append('featured', 'true');
    if (filters.verifiedOnly) params.append('verified_only', 'true');
    if (filters.fineTuningSupported) params.append('fine_tuning_supported', 'true');
    
    if (filters.priceRange) {
      params.append('price_min', filters.priceRange.min.toString());
      params.append('price_max', filters.priceRange.max.toString());
    }
    
    if (filters.ratingRange) {
      params.append('rating_min', filters.ratingRange.min.toString());
      params.append('rating_max', filters.ratingRange.max.toString());
    }
    
    if (filters.modelSize) {
      params.append('model_size', filters.modelSize);
    }
    
    if (filters.languages) {
      filters.languages.forEach(lang => params.append('languages', lang));
    }
    
    if (filters.dateRange) {
      params.append('date_start', filters.dateRange.start.toISOString());
      params.append('date_end', filters.dateRange.end.toISOString());
    }
    
    return params.toString();
  }

  private async prepareSubmissionData(submission: ModelSubmission): Promise<any> {
    // For now, assume JSON submission. In a real implementation,
    // this would handle file uploads using FormData
    return {
      name: submission.name,
      description: submission.description,
      category: submission.category,
      model_file: submission.modelFile,
      config_file: submission.configFile,
      pricing: submission.pricing,
      tags: submission.tags,
      capabilities: submission.capabilities,
      specifications: submission.specifications,
      examples: submission.examples,
      license: submission.license,
      documentation_url: submission.documentationUrl,
    };
  }

  private validateRentalRequest(request: RentalRequest): void {
    if (!request.durationHours || request.durationHours <= 0) {
      throw new ValidationError('Duration must be greater than 0 hours');
    }
    
    if (!request.maxRequests || request.maxRequests <= 0) {
      throw new ValidationError('Max requests must be greater than 0');
    }
    
    if (request.durationHours > 8760) { // 1 year
      throw new ValidationError('Duration cannot exceed 1 year');
    }
  }

  private validateModelSubmission(submission: ModelSubmission): void {
    if (!submission.name || submission.name.trim().length === 0) {
      throw new ValidationError('Model name is required');
    }
    
    if (!submission.description || submission.description.trim().length === 0) {
      throw new ValidationError('Model description is required');
    }
    
    if (!submission.category) {
      throw new ValidationError('Model category is required');
    }
    
    if (!submission.modelFile) {
      throw new ValidationError('Model file is required');
    }
    
    if (!submission.pricing?.ftnsPerRequest || submission.pricing.ftnsPerRequest <= 0) {
      throw new ValidationError('Valid pricing per request is required');
    }
    
    if (!submission.pricing?.revenueShare || 
        submission.pricing.revenueShare < 0 || 
        submission.pricing.revenueShare > 1) {
      throw new ValidationError('Revenue share must be between 0 and 1');
    }
    
    if (!submission.tags || submission.tags.length === 0) {
      throw new ValidationError('At least one tag is required');
    }
    
    if (!submission.capabilities || submission.capabilities.length === 0) {
      throw new ValidationError('At least one capability is required');
    }
  }

  private validateReviewSubmission(review: ReviewSubmission): void {
    if (!review.rating || review.rating < 1 || review.rating > 5) {
      throw new ValidationError('Rating must be between 1 and 5');
    }
    
    if (!review.title || review.title.trim().length === 0) {
      throw new ValidationError('Review title is required');
    }
    
    if (!review.content || review.content.trim().length === 0) {
      throw new ValidationError('Review content is required');
    }
    
    if (review.title.length > 100) {
      throw new ValidationError('Review title cannot exceed 100 characters');
    }
    
    if (review.content.length > 2000) {
      throw new ValidationError('Review content cannot exceed 2000 characters');
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
    this.featuredModelsCache = null;
    this.categoriesCache = null;
    this.lastCacheUpdate = null;
    
    if (this.debug) {
      console.log('[PRSM Marketplace] Cache cleared');
    }
  }

  /**
   * Get cached featured models without API call
   */
  getCachedFeaturedModels(): MarketplaceModel[] | null {
    return this.featuredModelsCache;
  }

  /**
   * Get cached categories without API call
   */
  getCachedCategories(): string[] | null {
    return this.categoriesCache;
  }
}