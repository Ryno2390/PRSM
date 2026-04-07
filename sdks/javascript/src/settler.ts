/**
 * PRSM Settler Client
 * Ring 6 - The Polish: L2-style staking, batch signatures, settlement
 */

import type { PRSMClient } from './client';
import { PRSMError, toPRSMError } from './errors';

// ============================================================================
// TYPES
// ============================================================================

export enum SettlerStatus {
  ACTIVE = 'active',
  UNBONDING = 'unbonding',
  EXITED = 'exited',
  SLASHED = 'slashed',
}

export interface SettlerInfo {
  settlerId: string;
  nodeId: string;
  status: SettlerStatus;
  stakedAmount: number;
  bondedAmount: number;
  unbondingAt?: Date;
  totalRewards: number;
  slashCount: number;
  createdAt: Date;
}

export interface SettlerRegistration {
  nodeId: string;
  stakeAmount: number;
  beneficiaryAddress: string;
  signature: string;
}

export interface BatchSignature {
  batchId: string;
  transactionHashes: string[];
  aggregateSignature: string;
  signerCount: number;
  threshold: number;
  createdAt: Date;
}

export interface BatchInfo {
  batchId: string;
  status: 'pending' | 'signed' | 'submitted' | 'confirmed' | 'failed';
  transactionCount: number;
  totalFtns: number;
  signatures: BatchSignature[];
  submittedAt?: Date;
  confirmedAt?: Date;
  transactionHash?: string;
}

export interface SlashProposal {
  proposalId: string;
  settlerId: string;
  reason: string;
  evidenceCid: string;
  proposedSlashAmount: number;
  votes: number;
  requiredVotes: number;
  status: 'pending' | 'approved' | 'rejected' | 'executed';
  createdAt: Date;
}

export interface SettlerManagerConfig {
  defaultStakeAmount?: number;
}

// ============================================================================
// SETTLER MANAGER
// ============================================================================

export class SettlerManager {
  private client: PRSMClient;
  private config: Required<SettlerManagerConfig>;

  constructor(client: PRSMClient, config?: SettlerManagerConfig) {
    this.client = client;
    this.config = {
      defaultStakeAmount: config?.defaultStakeAmount ?? 1000,
    };
  }

  /**
   * Register as a settler (stake FTNS to become a settlement validator)
   */
  async register(registration: SettlerRegistration): Promise<SettlerInfo> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<SettlerInfo>(
        '/api/v1/settler/register',
        {
          method: 'POST',
          headers,
          body: {
            node_id: registration.nodeId,
            stake_amount: registration.stakeAmount,
            beneficiary_address: registration.beneficiaryAddress,
            signature: registration.signature,
          },
        }
      );

      return {
        ...response,
        unbondingAt: response.unbondingAt
          ? new Date(response.unbondingAt)
          : undefined,
        createdAt: new Date(response.createdAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to register as settler');
    }
  }

  /**
   * Get settler information
   */
  async getSettler(settlerId: string): Promise<SettlerInfo> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<SettlerInfo>(
        `/api/v1/settler/${settlerId}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        unbondingAt: response.unbondingAt
          ? new Date(response.unbondingAt)
          : undefined,
        createdAt: new Date(response.createdAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get settler info');
    }
  }

  /**
   * Initiate unbonding (start exit process)
   */
  async unbond(settlerId: string): Promise<{ unbondingAt: Date }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{ unbonding_at: string }>(
        '/api/v1/settler/unbond',
        {
          method: 'POST',
          headers,
          body: { settler_id: settlerId },
        }
      );

      return {
        unbondingAt: new Date(response.unbonding_at),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to initiate unbonding');
    }
  }

  /**
   * Withdraw staked funds after unbonding period
   */
  async withdraw(settlerId: string): Promise<{ amount: number; txHash: string }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{
        amount: number;
        tx_hash: string;
      }>('/api/v1/settler/withdraw', {
        method: 'POST',
        headers,
        body: { settler_id: settlerId },
      });

      return {
        amount: response.amount,
        txHash: response.tx_hash,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to withdraw stake');
    }
  }

  /**
   * List active settlers
   */
  async listActive(): Promise<SettlerInfo[]> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<SettlerInfo[]>(
        '/api/v1/settler/list/active',
        { method: 'GET', headers }
      );

      return response.map(s => ({
        ...s,
        unbondingAt: s.unbondingAt ? new Date(s.unbondingAt) : undefined,
        createdAt: new Date(s.createdAt),
      }));
    } catch (error) {
      throw toPRSMError(error, 'Failed to list active settlers');
    }
  }

  /**
   * Sign a batch of transactions
   */
  async signBatch(batchId: string): Promise<BatchSignature> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<BatchSignature>(
        '/api/v1/settler/batch/sign',
        {
          method: 'POST',
          headers,
          body: { batch_id: batchId },
        }
      );

      return {
        ...response,
        createdAt: new Date(response.createdAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to sign batch');
    }
  }

  /**
   * Get batch information
   */
  async getBatch(batchId: string): Promise<BatchInfo> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<BatchInfo>(
        `/api/v1/settler/batch/${batchId}`,
        { method: 'GET', headers }
      );

      return {
        ...response,
        signatures: response.signatures?.map(s => ({
          ...s,
          createdAt: new Date(s.createdAt),
        })),
        submittedAt: response.submittedAt
          ? new Date(response.submittedAt)
          : undefined,
        confirmedAt: response.confirmedAt
          ? new Date(response.confirmedAt)
          : undefined,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get batch info');
    }
  }

  /**
   * List pending batches awaiting signatures
   */
  async listPendingBatches(): Promise<BatchInfo[]> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<BatchInfo[]>(
        '/api/v1/settler/batch/list/pending',
        { method: 'GET', headers }
      );

      return response.map(b => ({
        ...b,
        signatures: b.signatures?.map(s => ({
          ...s,
          createdAt: new Date(s.createdAt),
        })),
        submittedAt: b.submittedAt ? new Date(b.submittedAt) : undefined,
        confirmedAt: b.confirmedAt ? new Date(b.confirmedAt) : undefined,
      }));
    } catch (error) {
      throw toPRSMError(error, 'Failed to list pending batches');
    }
  }

  /**
   * Propose a slash for misbehavior
   */
  async proposeSlash(request: {
    settlerId: string;
    reason: string;
    evidenceCid: string;
    proposedSlashAmount: number;
  }): Promise<SlashProposal> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<SlashProposal>(
        '/api/v1/settler/slash/propose',
        {
          method: 'POST',
          headers,
          body: {
            settler_id: request.settlerId,
            reason: request.reason,
            evidence_cid: request.evidenceCid,
            proposed_slash_amount: request.proposedSlashAmount,
          },
        }
      );

      return {
        ...response,
        createdAt: new Date(response.createdAt),
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to propose slash');
    }
  }

  /**
   * Execute an approved slash proposal
   */
  async executeSlash(proposalId: string): Promise<{ slashedAmount: number }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{ slashed_amount: number }>(
        `/api/v1/settler/slash/${proposalId}/execute`,
        { method: 'POST', headers }
      );

      return {
        slashedAmount: response.slashed_amount,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to execute slash');
    }
  }

  /**
   * Export settlement ledger
   */
  async exportLedger(options: {
    format?: 'json' | 'csv';
    startDate?: Date;
    endDate?: Date;
  } = {}): Promise<Record<string, unknown>> {
    try {
      const params = new URLSearchParams();
      if (options.format) params.append('format', options.format);
      if (options.startDate)
        params.append('start_date', options.startDate.toISOString());
      if (options.endDate)
        params.append('end_date', options.endDate.toISOString());

      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<Record<string, unknown>>(
        `/api/v1/settler/ledger/export?${params}`,
        { method: 'GET', headers }
      );

      return response;
    } catch (error) {
      throw toPRSMError(error, 'Failed to export ledger');
    }
  }

  /**
   * Get settler statistics
   */
  async getStats(): Promise<{
    totalSettlers: number;
    activeSettlers: number;
    totalStaked: number;
    pendingBatches: number;
    totalBatchesProcessed: number;
    averageBatchSize: number;
  }> {
    try {
      const headers = await this.client.getAuthHeaders();
      const response = await this.client.makeRequest<{
        total_settlers: number;
        active_settlers: number;
        total_staked: number;
        pending_batches: number;
        total_batches_processed: number;
        average_batch_size: number;
      }>('/api/v1/settler/stats', { method: 'GET', headers });

      return {
        totalSettlers: response.total_settlers,
        activeSettlers: response.active_settlers,
        totalStaked: response.total_staked,
        pendingBatches: response.pending_batches,
        totalBatchesProcessed: response.total_batches_processed,
        averageBatchSize: response.average_batch_size,
      };
    } catch (error) {
      throw toPRSMError(error, 'Failed to get settler stats');
    }
  }
}
