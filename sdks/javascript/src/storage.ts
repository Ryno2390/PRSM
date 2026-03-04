/**
 * PRSM Storage Client
 * IPFS storage operations for decentralized data management
 */

import { PRSMClient } from './client';
import { PRSMError, NetworkError } from './errors';

// ============================================================================
// ENUMS
// ============================================================================

export enum StorageStatus {
  UPLOADING = 'uploading',
  AVAILABLE = 'available',
  PINNED = 'pinned',
  UNAVAILABLE = 'unavailable',
  EXPIRED = 'expired'
}

export enum ContentType {
  FILE = 'file',
  DATASET = 'dataset',
  MODEL = 'model',
  DOCUMENT = 'document',
  CODE = 'code',
  OTHER = 'other'
}

// ============================================================================
// TYPES
// ============================================================================

export interface StorageUploadOptions {
  /** Type of content */
  contentType?: ContentType;
  /** Original filename */
  filename?: string;
  /** Content description */
  description?: string;
  /** Tags for categorization */
  tags?: string[];
  /** Make content publicly accessible */
  isPublic?: boolean;
  /** Pin content for persistence */
  pin?: boolean;
  /** Replication factor */
  replication?: number;
}

export interface StorageUploadResult {
  /** IPFS content identifier */
  cid: string;
  /** Content size in bytes */
  size: number;
  /** Type of content */
  contentType: ContentType;
  /** Original filename */
  filename?: string;
  /** Upload timestamp */
  uploadTime: string;
  /** FTNS cost for upload */
  ftnsCost: number;
  /** Gateway URL for access */
  gatewayUrl: string;
  /** Whether content is pinned */
  isPinned: boolean;
}

export interface StorageInfo {
  /** IPFS content identifier */
  cid: string;
  /** Type of content */
  contentType: ContentType;
  /** Content size in bytes */
  size: number;
  /** Original filename */
  filename?: string;
  /** Content description */
  description?: string;
  /** Tags */
  tags: string[];
  /** Current status */
  status: StorageStatus;
  /** Public accessibility */
  isPublic: boolean;
  /** Pinned status */
  isPinned: boolean;
  /** Replication factor */
  replication: number;
  /** Upload timestamp */
  createdAt: string;
  /** Expiration time */
  expiresAt?: string;
  /** Owner address */
  owner: string;
  /** Number of accesses */
  accessCount: number;
  /** Gateway URL for access */
  gatewayUrl?: string;
}

export interface StorageSearchRequest {
  /** Search query */
  query?: string;
  /** Filter by type */
  contentType?: ContentType;
  /** Filter by tags */
  tags?: string[];
  /** Filter by owner */
  owner?: string;
  /** Filter by public status */
  isPublic?: boolean;
  /** Minimum size in bytes */
  minSize?: number;
  /** Maximum size in bytes */
  maxSize?: number;
  /** Maximum results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

export interface StorageSearchResult {
  /** Found items */
  items: StorageInfo[];
  /** Total matching items */
  total: number;
  /** Current offset */
  offset: number;
  /** Current limit */
  limit: number;
}

export interface PinInfo {
  /** Content identifier */
  cid: string;
  /** When pinned */
  pinnedAt: string;
  /** Content size */
  size: number;
  /** Replication factor */
  replication: number;
  /** Monthly FTNS cost */
  monthlyCost: number;
}

export interface StorageStats {
  /** Total storage used in bytes */
  totalStorage: number;
  /** Number of files */
  fileCount: number;
  /** Pinned content count */
  pinnedCount: number;
  /** Public content count */
  publicCount: number;
  /** Monthly FTNS cost */
  monthlyCost: number;
}

// ============================================================================
// STORAGE MANAGER
// ============================================================================

export interface StorageManagerConfig {
  /** Default replication factor */
  defaultReplication?: number;
  /** Auto-pin uploads */
  autoPin?: boolean;
  /** Default public visibility */
  defaultPublic?: boolean;
}

export class StorageManager {
  private client: PRSMClient;
  private config: StorageManagerConfig;

  constructor(client: PRSMClient, config?: StorageManagerConfig) {
    this.client = client;
    this.config = config || {
      defaultReplication: 3,
      autoPin: true,
      defaultPublic: false
    };
  }

  /**
   * Upload content to IPFS storage
   * 
   * @param data - Binary data to upload
   * @param options - Upload options
   * @returns Upload result with CID
   * 
   * @example
   * const result = await client.storage.upload(
   *   new Uint8Array([1, 2, 3]),
   *   { contentType: ContentType.FILE, filename: 'data.bin' }
   * );
   * console.log(`Uploaded to: ${result.cid}`);
   */
  async upload(
    data: ArrayBuffer | Uint8Array | Blob | string,
    options?: StorageUploadOptions
  ): Promise<StorageUploadResult> {
    const formData = new FormData();
    
    // Handle different data types
    if (typeof data === 'string') {
      formData.append('file', new Blob([data], { type: 'text/plain' }), options?.filename || 'file.txt');
    } else if (data instanceof Blob) {
      formData.append('file', data, options?.filename || 'file');
    } else {
      // Convert ArrayBuffer/Uint8Array to Blob - handle both ArrayBuffer and Uint8Array
      const uint8Array = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      formData.append('file', new Blob([uint8Array.buffer as ArrayBuffer]), options?.filename || 'file');
    }

    // Add metadata
    if (options?.contentType) {
      formData.append('contentType', options.contentType);
    }
    if (options?.description) {
      formData.append('description', options.description);
    }
    if (options?.tags && options.tags.length > 0) {
      formData.append('tags', JSON.stringify(options.tags));
    }
    formData.append('isPublic', String(options?.isPublic ?? this.config.defaultPublic ?? false));
    formData.append('pin', String(options?.pin ?? this.config.autoPin ?? true));
    formData.append('replication', String(options?.replication ?? this.config.defaultReplication ?? 3));

    const response = await this.client.makeRequest<StorageUploadResult>(
      '/api/v1/storage/upload',
      {
        method: 'POST',
        body: formData
      }
    );

    return response;
  }

  /**
   * Upload string content to IPFS storage
   * 
   * @param content - String content to upload
   * @param options - Upload options
   * @returns Upload result with CID
   */
  async uploadString(
    content: string,
    options?: StorageUploadOptions
  ): Promise<StorageUploadResult> {
    return this.upload(content, {
      ...options,
      contentType: options?.contentType ?? ContentType.DOCUMENT
    });
  }

  /**
   * Upload JSON data to IPFS storage
   * 
   * @param data - JSON data to upload
   * @param options - Upload options
   * @returns Upload result with CID
   */
  async uploadJSON(
    data: Record<string, any>,
    options?: StorageUploadOptions
  ): Promise<StorageUploadResult> {
    return this.upload(JSON.stringify(data), {
      ...options,
      contentType: options?.contentType ?? ContentType.DOCUMENT,
      filename: options?.filename ?? 'data.json'
    });
  }

  /**
   * Download content from IPFS
   * 
   * @param cid - IPFS content identifier
   * @returns Content as ArrayBuffer
   * 
   * @example
   * const data = await client.storage.download('QmXxx...');
   * const text = new TextDecoder().decode(data);
   */
  async download(cid: string): Promise<ArrayBuffer> {
    const response = await this.client.makeRequest<ArrayBuffer>(
      `/api/v1/storage/${cid}/download`,
      { method: 'GET' }
    );
    return response;
  }

  /**
   * Download content as string
   * 
   * @param cid - IPFS content identifier
   * @returns Content as string
   */
  async downloadString(cid: string): Promise<string> {
    const data = await this.download(cid);
    return new TextDecoder().decode(data);
  }

  /**
   * Download content as JSON
   * 
   * @param cid - IPFS content identifier
   * @returns Content as JSON object
   */
  async downloadJSON<T = any>(cid: string): Promise<T> {
    const text = await this.downloadString(cid);
    return JSON.parse(text);
  }

  /**
   * Get information about stored content
   * 
   * @param cid - IPFS content identifier
   * @returns Storage info
   */
  async getInfo(cid: string): Promise<StorageInfo> {
    return await this.client.makeRequest<StorageInfo>(
      `/api/v1/storage/${cid}`,
      { method: 'GET' }
    );
  }

  /**
   * Pin content for persistent storage
   * 
   * @param cid - IPFS content identifier
   * @param replication - Replication factor
   * @returns Pin info
   * 
   * @example
   * const pinInfo = await client.storage.pin('QmXxx...', 5);
   * console.log(`Pinned at ${pinInfo.pinnedAt}`);
   */
  async pin(cid: string, replication?: number): Promise<PinInfo> {
    return await this.client.makeRequest<PinInfo>(
      `/api/v1/storage/${cid}/pin`,
      {
        method: 'POST',
        body: { replication: replication ?? this.config.defaultReplication ?? 3 }
      }
    );
  }

  /**
   * Unpin content from storage
   * 
   * @param cid - IPFS content identifier
   * @returns True if unpinned successfully
   */
  async unpin(cid: string): Promise<boolean> {
    const response = await this.client.makeRequest<{ unpinned: boolean }>(
      `/api/v1/storage/${cid}/unpin`,
      { method: 'POST' }
    );
    return response.unpinned;
  }

  /**
   * List all pinned content
   *
   * @param limit - Maximum results
   * @returns List of pin info
   */
  async listPins(limit?: number): Promise<PinInfo[]> {
    const url = limit ? `/api/v1/storage/pins?limit=${limit}` : '/api/v1/storage/pins';
    const response = await this.client.makeRequest<{ pins: PinInfo[] }>(
      url,
      { method: 'GET' }
    );
    return response.pins;
  }

  /**
   * Search for stored content
   * 
   * @param request - Search request
   * @returns Search result with matching items
   * 
   * @example
   * const results = await client.storage.search({
   *   query: 'model weights',
   *   contentType: ContentType.MODEL,
   *   limit: 10
   * });
   */
  async search(request: StorageSearchRequest): Promise<StorageSearchResult> {
    return await this.client.makeRequest<StorageSearchResult>(
      '/api/v1/storage/search',
      {
        method: 'POST',
        body: request
      }
    );
  }

  /**
   * Delete content from storage
   * 
   * @param cid - IPFS content identifier
   * @returns True if deleted successfully
   */
  async delete(cid: string): Promise<boolean> {
    const response = await this.client.makeRequest<{ deleted: boolean }>(
      `/api/v1/storage/${cid}`,
      { method: 'DELETE' }
    );
    return response.deleted;
  }

  /**
   * Update content metadata
   * 
   * @param cid - IPFS content identifier
   * @param metadata - Metadata to update
   * @returns Updated storage info
   */
  async updateMetadata(
    cid: string,
    metadata: {
      description?: string;
      tags?: string[];
      isPublic?: boolean;
    }
  ): Promise<StorageInfo> {
    return await this.client.makeRequest<StorageInfo>(
      `/api/v1/storage/${cid}`,
      {
        method: 'PATCH',
        body: metadata
      }
    );
  }

  /**
   * Get HTTP gateway URL for content
   * 
   * @param cid - IPFS content identifier
   * @returns Gateway URL string
   */
  async getGatewayUrl(cid: string): Promise<string> {
    try {
      const info = await this.getInfo(cid);
      return info.gatewayUrl || `https://ipfs.io/ipfs/${cid}`;
    } catch {
      return `https://ipfs.io/ipfs/${cid}`;
    }
  }

  /**
   * Estimate FTNS cost for uploading content
   * 
   * @param sizeBytes - Size of content in bytes
   * @param replication - Replication factor
   * @param durationDays - Storage duration in days
   * @returns Estimated FTNS cost
   */
  async estimateUploadCost(
    sizeBytes: number,
    replication?: number,
    durationDays?: number
  ): Promise<number> {
    const response = await this.client.makeRequest<{ estimatedCost: number }>(
      '/api/v1/storage/estimate-cost',
      {
        method: 'POST',
        body: {
          size: sizeBytes,
          replication: replication ?? this.config.defaultReplication ?? 3,
          durationDays: durationDays ?? 30
        }
      }
    );
    return response.estimatedCost;
  }

  /**
   * Get storage usage statistics
   * 
   * @returns Storage statistics
   */
  async getStats(): Promise<StorageStats> {
    return await this.client.makeRequest<StorageStats>(
      '/api/v1/storage/stats',
      { method: 'GET' }
    );
  }
}