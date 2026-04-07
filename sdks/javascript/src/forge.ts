/**
 * PRSM Agent Forge Client
 * Ring 5 - The Brain: LLM decomposition, planning, and agent dispatch
 */

import type { PRSMClient } from './client';
import { PRSMError, toPRSMError } from './errors';

// ============================================================================
// TYPES
// ============================================================================

export enum ExecutionRoute {
  DIRECT_LLM = 'direct_llm',
  SINGLE_AGENT = 'single_agent',
  SWARM = 'swarm',
}

export enum ThermalRequirement {
  BURST = 'burst',
  SUSTAINED = 'sustained',
  THROTTLED = 'throttled',
}

export enum HardwareTier {
  T1 = 't1', // < 5 TFLOPS (mobile, IoT)
  T2 = 't2', // 5-30 TFLOPS (consoles, mid-range)
  T3 = 't3', // 30-80 TFLOPS (high-end desktop, M-series)
  T4 = 't4', // 80+ TFLOPS (datacenter, multi-GPU)
}

export interface TaskDecomposition {
  query: string;
  requiredDatasets: string[];
  operations: string[];
  parallelizable: boolean;
  minHardwareTier: HardwareTier;
  estimatedComplexity: number;
}

export interface TaskPlan {
  decomposition: TaskDecomposition;
  route: ExecutionRoute;
  targetShardCids: string[];
  estimatedPcu: number;
  thermalRequirement: ThermalRequirement;
}

export interface CostQuote {
  estimatedFtns: number;
  breakdown: {
    compute: number;
    data: number;
    network: number;
  };
  validUntil: Date;
}

export interface ForgeResult {
  query: string;
  decomposition: Record<string, unknown>;
  plan: Record<string, unknown>;
  costQuote?: Record<string, unknown>;
  result: {
    response?: string;
    agentId?: string;
    jobId?: string;
    route: string;
    status: string;
    shardsCompleted?: number;
    totalPcu?: number;
    aggregatedOutput?: string;
    error?: string;
  };
  instructions: string;
  traceId: number;
}

export interface ForgeRequest {
  query: string;
  budgetFtns?: number;
  shardCids?: string[];
}

export interface HardwareProfile {
  tflops: number;
  gpuModel?: string;
  vramGb: number;
  ramGb: number;
  tier: HardwareTier;
  thermal: ThermalRequirement;
  teeCapable: boolean;
}

export interface ForgeManagerConfig {
  defaultBudget?: number;
  defaultTimeout?: number;
}

// ============================================================================
// FORGE MANAGER
// ============================================================================

export class ForgeManager {
  private client: PRSMClient;
  private config: Required<ForgeManagerConfig>;

  constructor(client: PRSMClient, config?: ForgeManagerConfig) {
    this.client = client;
    this.config = {
      defaultBudget: config?.defaultBudget ?? 10.0,
      defaultTimeout: config?.defaultTimeout ?? 300,
    };
  }

  /**
   * Run the full Agent Forge pipeline: decompose -> plan -> execute
   */
  async run(request: ForgeRequest): Promise<ForgeResult> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<ForgeResult>(
        '/api/v1/compute/forge',
        {
          method: 'POST',
          headers,
          body: {
            query: request.query,
            budget_ftns: request.budgetFtns ?? this.config.defaultBudget,
            shard_cids: request.shardCids ?? [],
          },
        }
      );

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to run forge pipeline');
    }
  }

  /**
   * Decompose a query into structured task fields
   */
  async decompose(query: string): Promise<TaskDecomposition> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<TaskDecomposition>(
        '/api/v1/compute/forge/decompose',
        {
          method: 'POST',
          headers,
          body: { query },
        }
      );

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to decompose query');
    }
  }

  /**
   * Build a task plan from a decomposition
   */
  async plan(
    decomposition: TaskDecomposition,
    shardCids?: string[]
  ): Promise<{ plan: TaskPlan; costQuote?: CostQuote }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{
        plan: TaskPlan;
        costQuote?: CostQuote;
      }>('/api/v1/compute/forge/plan', {
        method: 'POST',
        headers,
        body: {
          decomposition,
          shard_cids: shardCids ?? [],
        },
      });

      return {
        plan: response.plan,
        costQuote: response.costQuote
          ? {
              ...response.costQuote,
              validUntil: new Date(response.costQuote.validUntil),
            }
          : undefined,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to create plan');
    }
  }

  /**
   * Execute a task plan
   */
  async execute(
    plan: TaskPlan,
    budgetFtns?: number
  ): Promise<Record<string, unknown>> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<Record<string, unknown>>(
        '/api/v1/compute/forge/execute',
        {
          method: 'POST',
          headers,
          body: {
            plan,
            budget_ftns: budgetFtns ?? this.config.defaultBudget,
          },
        }
      );

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to execute plan');
    }
  }

  /**
   * Get a cost quote for a query
   */
  async quote(request: {
    query: string;
    shardCids?: string[];
  }): Promise<CostQuote> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<CostQuote>(
        '/api/v1/compute/forge/quote',
        {
          method: 'POST',
          headers,
          body: {
            query: request.query,
            shard_cids: request.shardCids ?? [],
          },
        }
      );

      return {
        ...response,
        validUntil: new Date(response.validUntil),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get quote');
    }
  }

  /**
   * Get hardware profile for the connected node
   */
  async getHardwareProfile(): Promise<HardwareProfile> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<HardwareProfile>(
        '/api/v1/node/hardware-profile',
        { method: 'GET', headers }
      );

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to get hardware profile');
    }
  }

  /**
   * Get forge execution traces
   */
  async getTraces(options: { limit?: number } = {}): Promise<
    Array<Record<string, unknown>>
  > {
    try {
      const params = new URLSearchParams();
      if (options.limit) params.append('limit', options.limit.toString());

      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{
        traces: Array<Record<string, unknown>>;
      }>(`/api/v1/compute/forge/traces?${params}`, {
        method: 'GET',
        headers,
      });

      return response.traces ?? [];
    } catch (error) {
      throw toPRSMError(error, 'Failed to get traces');
    }
  }
}
