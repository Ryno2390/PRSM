/**
 * PRSM Compute Client
 * Submit and manage compute jobs on the PRSM network (Ring 1-10)
 */

import type { PRSMClient } from './client';
import { PRSMError, toPRSMError } from './errors';

// ============================================================================
// TYPES
// ============================================================================

export enum JobStatus {
  PENDING = 'pending',
  QUEUED = 'queued',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  TIMEOUT = 'timeout',
}

export enum JobPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  URGENT = 'urgent',
}

export interface JobRequest {
  prompt: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  budget?: number;
  priority?: JobPriority;
  timeout?: number;
  context?: Record<string, unknown>;
  tools?: string[];
  stream?: boolean;
}

export interface JobResponse {
  jobId: string;
  status: JobStatus;
  createdAt: Date;
  estimatedCost: number;
  estimatedDuration: number;
  queuePosition?: number;
}

export interface JobResult {
  jobId: string;
  status: JobStatus;
  content: string;
  model: string;
  provider: string;
  executionTime: number;
  tokenUsage: Record<string, number>;
  ftnsCost: number;
  reasoningTrace?: string[];
  citations?: Array<{ source: string; url?: string }>;
  metadata?: Record<string, unknown>;
  completedAt: Date;
}

export interface JobInfo {
  jobId: string;
  status: JobStatus;
  request: JobRequest;
  result?: JobResult;
  progress: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
  nodeId?: string;
}

export interface JobListResponse {
  jobs: JobInfo[];
  total: number;
  offset: number;
  limit: number;
}

export interface ComputeManagerConfig {
  maxRetries?: number;
  defaultTimeout?: number;
  pollInterval?: number;
}

// ============================================================================
// COMPUTE MANAGER
// ============================================================================

export class ComputeManager {
  private client: PRSMClient;
  private config: Required<ComputeManagerConfig>;

  constructor(client: PRSMClient, config?: ComputeManagerConfig) {
    this.client = client;
    this.config = {
      maxRetries: config?.maxRetries ?? 3,
      defaultTimeout: config?.defaultTimeout ?? 300,
      pollInterval: config?.pollInterval ?? 2000,
    };
  }

  /**
   * Submit a compute job to the PRSM network
   */
  async submitJob(request: JobRequest): Promise<JobResponse> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<JobResponse>(
        '/api/v1/compute/jobs',
        {
          method: 'POST',
          headers,
          body: {
            prompt: request.prompt,
            model: request.model ?? 'nwtn',
            max_tokens: request.maxTokens ?? 1000,
            temperature: request.temperature ?? 0.7,
            budget: request.budget,
            priority: request.priority ?? JobPriority.NORMAL,
            timeout: request.timeout ?? this.config.defaultTimeout,
            context: request.context,
            tools: request.tools,
            stream: request.stream ?? false,
          },
        }
      );

      return {
        ...response,
        createdAt: new Date(response.createdAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to submit compute job');
    }
  }

  /**
   * Get detailed information about a job
   */
  async getJob(jobId: string): Promise<JobInfo> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<JobInfo>(
        `/api/v1/compute/jobs/${jobId}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        createdAt: new Date(response.createdAt),
        startedAt: response.startedAt ? new Date(response.startedAt) : undefined,
        completedAt: response.completedAt ? new Date(response.completedAt) : undefined,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get job details');
    }
  }

  /**
   * Get the result of a completed job
   */
  async getResult(jobId: string): Promise<JobResult> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<JobResult>(
        `/api/v1/compute/jobs/${jobId}/result`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        completedAt: new Date(response.completedAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get job result');
    }
  }

  /**
   * Cancel a running job
   */
  async cancelJob(jobId: string): Promise<boolean> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{ cancelled: boolean }>(
        `/api/v1/compute/jobs/${jobId}/cancel`,
        { method: 'POST', headers }
      );
      return response.cancelled ?? false;
    } catch (error) {
      throw toPRSMError(error, 'Failed to cancel job');
    }
  }

  /**
   * List recent jobs
   */
  async listJobs(options: {
    status?: JobStatus;
    limit?: number;
    offset?: number;
  } = {}): Promise<JobListResponse> {
    try {
      const params = new URLSearchParams();
      if (options.status) params.append('status', options.status);
      if (options.limit) params.append('limit', options.limit.toString());
      if (options.offset) params.append('offset', options.offset.toString());

      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<JobListResponse>(
        `/api/v1/compute/jobs?${params}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        jobs: response.jobs.map(job => ({
          ...job,
          createdAt: new Date(job.createdAt),
          startedAt: job.startedAt ? new Date(job.startedAt) : undefined,
          completedAt: job.completedAt ? new Date(job.completedAt) : undefined,
        })),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to list jobs');
    }
  }

  /**
   * Wait for a job to complete
   */
  async waitForCompletion(
    jobId: string,
    options: {
      timeoutMs?: number;
      pollIntervalMs?: number;
      onProgress?: (job: JobInfo) => void;
    } = {}
  ): Promise<JobResult> {
    const timeoutMs = options.timeoutMs ?? 600000; // 10 minutes
    const pollIntervalMs = options.pollIntervalMs ?? this.config.pollInterval;
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const job = await this.getJob(jobId);

      if (options.onProgress) {
        options.onProgress(job);
      }

      if (job.status === JobStatus.COMPLETED) {
        return await this.getResult(jobId);
      }

      if ([JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT].includes(job.status)) {
        throw new PRSMError(
          `Job ${jobId} ended with status: ${job.status}`,
          'JOB_FAILED',
          jobId,
          { job }
        );
      }

      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }

    throw new PRSMError(
      `Timeout waiting for job ${jobId}`,
      'JOB_TIMEOUT',
      jobId
    );
  }

  /**
   * Estimate the FTNS cost for a job
   */
  async estimateCost(request: {
    prompt: string;
    model?: string;
    maxTokens?: number;
  }): Promise<number> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{ estimatedCost: number }>(
        '/api/v1/compute/estimate',
        {
          method: 'POST',
          headers,
          body: {
            prompt: request.prompt,
            model: request.model ?? 'nwtn',
            max_tokens: request.maxTokens ?? 1000,
          },
        }
      );
      return response.estimatedCost ?? 0;
    } catch (error) {
      throw toPRSMError(error, 'Failed to estimate cost');
    }
  }

  /**
   * Get current queue status
   */
  async getQueueStatus(): Promise<Record<string, unknown>> {
    try {
      const headers = await this.client.getAuthHeaders();
      return await this.client.makeRequest<Record<string, unknown>>(
        '/api/v1/compute/queue/status',
        { method: 'GET', headers }
      );
    } catch (error) {
      throw toPRSMError(error, 'Failed to get queue status');
    }
  }

  /**
   * Get available compute models
   */
  async getAvailableModels(): Promise<Array<Record<string, unknown>>> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{ models: Array<Record<string, unknown>> }>(
        '/api/v1/compute/models',
        { method: 'GET', headers }
      );
      return response.models ?? [];
    } catch (error) {
      throw toPRSMError(error, 'Failed to get available models');
    }
  }
}
