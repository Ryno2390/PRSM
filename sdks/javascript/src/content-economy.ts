/**
 * PRSM Content Economy Client
 * Ring 4 - The Economy: Data marketplace, content listings, access control
 */

import type { PRSMClient } from './client';
import { PRSMError, toPRSMError } from './errors';

// ============================================================================
// TYPES
// ============================================================================

export enum ContentStatus {
  AVAILABLE = 'available',
  PROCESSING = 'processing',
  DELISTED = 'delisted',
}

export interface DataListing {
  cid: string;
  title: string;
  description: string;
  owner: string;
  accessFee: number;
  category: string;
  tags: string[];
  schema?: Record<string, unknown>;
  rowCount?: number;
  sizeBytes: number;
  status: ContentStatus;
  createdAt: Date;
  updatedAt: Date;
}

export interface ContentAccessRequest {
  cid: string;
  requesterAddress: string;
  accessType: 'read' | 'compute' | 'download';
  duration?: number;
}

export interface ContentAccessResponse {
  accessId: string;
  cid: string;
  granted: boolean;
  accessToken?: string;
  expiresAt?: Date;
  paymentId?: string;
}

export interface SemanticSearchRequest {
  query: string;
  limit?: number;
  categories?: string[];
  minAccessFee?: number;
  maxAccessFee?: number;
}

export interface SemanticSearchResult {
  results: Array<{
    cid: string;
    title: string;
    description: string;
    similarity: number;
    accessFee: number;
    owner: string;
  }>;
  total: number;
  queryEmbedding?: number[];
}

export interface ReplicationStatus {
  cid: string;
  replicaCount: number;
  targetReplicas: number;
  nodes: Array<{
    nodeId: string;
    status: 'active' | 'syncing' | 'failed';
    lastSync: Date;
  }>;
}

export interface RoyaltyInfo {
  cid: string;
  totalRoyalties: number;
  pendingPayout: number;
  lastPayout?: Date;
  payoutHistory: Array<{
    amount: number;
    date: Date;
    txHash: string;
  }>;
}

export interface ContentEconomyStats {
  totalListings: number;
  totalAccesses: number;
  totalRevenue: number;
  topCategories: Array<{ category: string; count: number }>;
}

export interface ContentEconomyManagerConfig {
  defaultSearchLimit?: number;
}

// ============================================================================
// CONTENT ECONOMY MANAGER
// ============================================================================

export class ContentEconomyManager {
  private client: PRSMClient;
  private config: Required<ContentEconomyManagerConfig>;

  constructor(client: PRSMClient, config?: ContentEconomyManagerConfig) {
    this.client = client;
    this.config = {
      defaultSearchLimit: config?.defaultSearchLimit ?? 20,
    };
  }

  /**
   * List a dataset for sale on the marketplace
   */
  async listDataset(request: {
    cid: string;
    title: string;
    description: string;
    accessFee: number;
    category: string;
    tags?: string[];
    schema?: Record<string, unknown>;
  }): Promise<DataListing> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<DataListing>(
        '/api/v1/content-economy/listings',
        {
          method: 'POST',
          headers,
          body: {
            cid: request.cid,
            title: request.title,
            description: request.description,
            access_fee: request.accessFee,
            category: request.category,
            tags: request.tags ?? [],
            schema: request.schema,
          },
        }
      );

      return {
        ...response,
        createdAt: new Date(response.createdAt),
        updatedAt: new Date(response.updatedAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to list dataset');
    }
  }

  /**
   * Get a data listing by CID
   */
  async getListing(cid: string): Promise<DataListing> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<DataListing>(
        `/api/v1/content-economy/listings/${cid}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        createdAt: new Date(response.createdAt),
        updatedAt: new Date(response.updatedAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get listing');
    }
  }

  /**
   * Request access to a dataset
   */
  async requestAccess(
    request: ContentAccessRequest
  ): Promise<ContentAccessResponse> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<ContentAccessResponse>(
        '/api/v1/content-economy/access',
        {
          method: 'POST',
          headers,
          body: {
            cid: request.cid,
            requester_address: request.requesterAddress,
            access_type: request.accessType,
            duration: request.duration,
          },
        }
      );

      return {
        ...response,
        expiresAt: response.expiresAt
          ? new Date(response.expiresAt)
          : undefined,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to request access');
    }
  }

  /**
   * Get payment status for an access request
   */
  async getPaymentStatus(paymentId: string): Promise<ContentAccessResponse> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<ContentAccessResponse>(
        `/api/v1/content-economy/payment/${paymentId}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        expiresAt: response.expiresAt
          ? new Date(response.expiresAt)
          : undefined,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get payment status');
    }
  }

  /**
   * Request data retrieval (code-to-data compute)
   */
  async requestRetrieval(request: {
    cid: string;
    query?: string;
    format?: 'json' | 'parquet' | 'csv';
  }): Promise<{
    requestId: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    resultCid?: string;
    estimatedTime?: number;
  }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{
        request_id: string;
        status: string;
        result_cid?: string;
        estimated_time?: number;
      }>('/api/v1/content-economy/retrieval', {
        method: 'POST',
        headers,
        body: {
          cid: request.cid,
          query: request.query,
          format: request.format ?? 'json',
        },
      });

      return {
        requestId: response.request_id,
        status: response.status as
          | 'pending'
          | 'processing'
          | 'completed'
          | 'failed',
        resultCid: response.result_cid,
        estimatedTime: response.estimated_time,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to request retrieval');
    }
  }

  /**
   * Get retrieval request status
   */
  async getRetrievalStatus(requestId: string): Promise<{
    requestId: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    resultCid?: string;
    completedAt?: Date;
    error?: string;
  }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{
        request_id: string;
        status: string;
        result_cid?: string;
        completed_at?: string;
        error?: string;
      }>(`/api/v1/content-economy/retrieval/${requestId}`, {
        method: 'GET',
        headers,
      });

      return {
        requestId: response.request_id,
        status: response.status as
          | 'pending'
          | 'processing'
          | 'completed'
          | 'failed',
        resultCid: response.result_cid,
        completedAt: response.completed_at
          ? new Date(response.completed_at)
          : undefined,
        error: response.error,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get retrieval status');
    }
  }

  /**
   * Semantic search across datasets
   */
  async semanticSearch(
    request: SemanticSearchRequest
  ): Promise<SemanticSearchResult> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<SemanticSearchResult>(
        '/api/v1/content-economy/search',
        {
          method: 'POST',
          headers,
          body: {
            query: request.query,
            limit: request.limit ?? this.config.defaultSearchLimit,
            categories: request.categories,
            min_access_fee: request.minAccessFee,
            max_access_fee: request.maxAccessFee,
          },
        }
      );

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to search datasets');
    }
  }

  /**
   * Index a CID for search
   */
  async indexCid(cid: string): Promise<{ indexed: boolean }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{ indexed: boolean }>(
        `/api/v1/content-economy/index/${cid}`,
        { method: 'POST', headers }
      );
      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to index CID');
    }
  }

  /**
   * Get replication status for a CID
   */
  async getReplicationStatus(cid: string): Promise<ReplicationStatus> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<ReplicationStatus>(
        `/api/v1/content-economy/replication/${cid}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        nodes: response.nodes.map(n => ({
          ...n,
          lastSync: new Date(n.lastSync),
        })),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get replication status');
    }
  }

  /**
   * Ensure minimum replication for a CID
   */
  async ensureReplication(cid: string, targetReplicas: number): Promise<{
    success: boolean;
    newReplicaCount: number;
  }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{
        success: boolean;
        new_replica_count: number;
      }>(`/api/v1/content-economy/replication/${cid}/ensure`, {
        method: 'POST',
        headers,
        body: { target_replicas: targetReplicas },
      });

      return {
        success: response.success,
        newReplicaCount: response.new_replica_count,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to ensure replication');
    }
  }

  /**
   * Get royalty information for a dataset
   */
  async getRoyaltyInfo(cid: string): Promise<RoyaltyInfo> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<RoyaltyInfo>(
        `/api/v1/content-economy/royalty/${cid}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        lastPayout: response.lastPayout
          ? new Date(response.lastPayout)
          : undefined,
        payoutHistory: response.payoutHistory.map(p => ({
          ...p,
          date: new Date(p.date),
        })),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get royalty info');
    }
  }

  /**
   * Get content economy statistics
   */
  async getStats(): Promise<ContentEconomyStats> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<ContentEconomyStats>(
        '/api/v1/content-economy/stats',
        { method: 'GET', headers }
      );
      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get stats');
    }
  }

  /**
   * Get available models for content processing
   */
  async getModels(): Promise<
    Array<{ modelId: string; name: string; category: string }>
  > {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<
        Array<{ model_id: string; name: string; category: string }>
      >('/api/v1/content-economy/models', { method: 'GET', headers });

      return response.map(m => ({
        modelId: m.model_id,
        name: m.name,
        category: m.category,
      }));
    } catch (error) {
      throw toPRSMError(error, 'Failed to get models');
    }
  }
}
