/**
 * PRSM Governance Client
 * Participate in PRSM governance and voting
 */

import { PRSMClient } from './client';
import { PRSMError } from './errors';

// ============================================================================
// ENUMS
// ============================================================================

export enum ProposalStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  VOTING = 'voting',
  PASSED = 'passed',
  REJECTED = 'rejected',
  EXECUTED = 'executed',
  EXPIRED = 'expired',
  CANCELLED = 'cancelled'
}

export enum ProposalType {
  PARAMETER_CHANGE = 'parameter_change',
  PROTOCOL_UPGRADE = 'protocol_upgrade',
  TREASURY_SPEND = 'treasury_spend',
  MODEL_ADDITION = 'model_addition',
  MODEL_REMOVAL = 'model_removal',
  FEE_ADJUSTMENT = 'fee_adjustment',
  GOVERNANCE_CHANGE = 'governance_change',
  OTHER = 'other'
}

export enum VoteChoice {
  YES = 'yes',
  NO = 'no',
  ABSTAIN = 'abstain'
}

// ============================================================================
// TYPES
// ============================================================================

export interface Proposal {
  /** Unique proposal ID */
  proposalId: string;
  /** Proposal title */
  title: string;
  /** Proposal description */
  description: string;
  /** Type of proposal */
  proposalType: ProposalType;
  /** Current status */
  status: ProposalStatus;
  /** Proposer address */
  proposer: string;
  /** Creation timestamp */
  createdAt: string;
  /** Voting start time */
  votingStarts: string;
  /** Voting end time */
  votingEnds: string;
  /** Required quorum percentage */
  quorum: number;
  /** Required approval threshold */
  threshold: number;
  /** Yes votes (FTNS) */
  votesYes: number;
  /** No votes (FTNS) */
  votesNo: number;
  /** Abstain votes (FTNS) */
  votesAbstain: number;
  /** Total voters */
  totalVoters: number;
  /** Proposal parameters */
  parameters: Record<string, any>;
  /** Additional metadata */
  metadata: Record<string, any>;
}

export interface ProposalCreate {
  /** Proposal title */
  title: string;
  /** Proposal description */
  description: string;
  /** Type of proposal */
  proposalType: ProposalType;
  /** Proposal parameters */
  parameters?: Record<string, any>;
  /** Voting duration in days */
  durationDays?: number;
  /** Required quorum */
  quorum?: number;
  /** Approval threshold */
  threshold?: number;
}

export interface Vote {
  /** Vote ID */
  voteId: string;
  /** Proposal ID */
  proposalId: string;
  /** Voter address */
  voter: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Voting power (FTNS) */
  votingPower: number;
  /** Vote timestamp */
  timestamp: string;
  /** Vote reason */
  reason?: string;
}

export interface VoteRequest {
  /** Proposal to vote on */
  proposalId: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Vote reason */
  reason?: string;
}

export interface GovernanceStats {
  /** Total proposals */
  totalProposals: number;
  /** Active proposals */
  activeProposals: number;
  /** Total votes cast */
  totalVotes: number;
  /** Total voting power */
  totalVotingPower: number;
  /** Participation rate */
  participationRate: number;
  /** Rate of quorum met */
  quorumMetRate: number;
}

export interface DelegationInfo {
  /** Delegator address */
  delegator: string;
  /** Delegate address */
  delegate: string;
  /** Delegated voting power */
  votingPower: number;
  /** Delegation timestamp */
  createdAt: string;
}

export interface ProposalListOptions {
  /** Filter by status */
  status?: ProposalStatus;
  /** Filter by type */
  proposalType?: ProposalType;
  /** Maximum results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

// ============================================================================
// GOVERNANCE MANAGER
// ============================================================================

export interface GovernanceManagerConfig {
  /** Default voting duration in days */
  defaultDurationDays?: number;
  /** Default quorum requirement */
  defaultQuorum?: number;
  /** Default approval threshold */
  defaultThreshold?: number;
}

export class GovernanceManager {
  private client: PRSMClient;
  private config: GovernanceManagerConfig;

  constructor(client: PRSMClient, config?: GovernanceManagerConfig) {
    this.client = client;
    this.config = config || {
      defaultDurationDays: 7,
      defaultQuorum: 0.1,
      defaultThreshold: 0.5
    };
  }

  /**
   * Create a new governance proposal
   * 
   * @param proposal - Proposal details
   * @returns Created proposal
   * 
   * @example
   * const proposal = await client.governance.createProposal({
   *   title: 'Increase FTNS staking rewards',
   *   description: 'Proposal to increase staking APY from 5% to 7%',
   *   proposalType: ProposalType.PARAMETER_CHANGE,
   *   parameters: { stakingApy: 0.07 }
   * });
   */
  async createProposal(proposal: ProposalCreate): Promise<Proposal> {
    const requestData = {
      title: proposal.title,
      description: proposal.description,
      proposalType: proposal.proposalType,
      parameters: proposal.parameters || {},
      durationDays: proposal.durationDays ?? this.config.defaultDurationDays ?? 7,
      quorum: proposal.quorum ?? this.config.defaultQuorum ?? 0.1,
      threshold: proposal.threshold ?? this.config.defaultThreshold ?? 0.5
    };

    return await this.client.makeRequest<Proposal>(
      '/api/v1/governance/proposals',
      {
        method: 'POST',
        body: requestData
      }
    );
  }

  /**
   * Get proposal details
   * 
   * @param proposalId - Proposal identifier
   * @returns Proposal details
   * 
   * @example
   * const proposal = await client.governance.getProposal('prop_123');
   * console.log(`Status: ${proposal.status}`);
   */
  async getProposal(proposalId: string): Promise<Proposal> {
    return await this.client.makeRequest<Proposal>(
      `/api/v1/governance/proposals/${proposalId}`,
      { method: 'GET' }
    );
  }

  /**
   * List governance proposals
   * 
   * @param options - List options
   * @returns List of proposals
   */
  async listProposals(options?: ProposalListOptions): Promise<Proposal[]> {
    const params = new URLSearchParams();
    if (options?.status) params.append('status', options.status);
    if (options?.proposalType) params.append('type', options.proposalType);
    if (options?.limit) params.append('limit', options.limit.toString());
    if (options?.offset) params.append('offset', options.offset.toString());

    const queryString = params.toString();
    const url = queryString ? `/api/v1/governance/proposals?${queryString}` : '/api/v1/governance/proposals';

    const response = await this.client.makeRequest<{ proposals: Proposal[] }>(
      url,
      { method: 'GET' }
    );

    return response.proposals;
  }

  /**
   * Cast a vote on a proposal
   * 
   * @param proposalId - Proposal to vote on
   * @param choice - Vote choice
   * @param reason - Optional reason for vote
   * @returns Vote record
   * 
   * @example
   * const vote = await client.governance.vote(
   *   'prop_123',
   *   VoteChoice.YES,
   *   'This proposal benefits the network'
   * );
   */
  async vote(proposalId: string, choice: VoteChoice, reason?: string): Promise<Vote> {
    const requestData: any = {
      proposalId,
      choice
    };
    if (reason !== undefined) {
      requestData.reason = reason;
    }

    return await this.client.makeRequest<Vote>(
      `/api/v1/governance/proposals/${proposalId}/vote`,
      {
        method: 'POST',
        body: requestData
      }
    );
  }

  /**
   * Get vote details
   * 
   * @param proposalId - Proposal identifier
   * @param voter - Voter address (optional, defaults to current user)
   * @returns Vote record or null if not voted
   */
  async getVote(proposalId: string, voter?: string): Promise<Vote | null> {
    const queryString = voter ? `?voter=${voter}` : '';
    
    try {
      return await this.client.makeRequest<Vote>(
        `/api/v1/governance/proposals/${proposalId}/vote${queryString}`,
        { method: 'GET' }
      );
    } catch (error) {
      return null;
    }
  }

  /**
   * Get all votes for a proposal
   * 
   * @param proposalId - Proposal identifier
   * @param limit - Maximum results
   * @param offset - Offset for pagination
   * @returns List of votes
   */
  async getProposalVotes(proposalId: string, limit?: number, offset?: number): Promise<Vote[]> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());

    const queryString = params.toString();
    const url = queryString 
      ? `/api/v1/governance/proposals/${proposalId}/votes?${queryString}` 
      : `/api/v1/governance/proposals/${proposalId}/votes`;

    const response = await this.client.makeRequest<{ votes: Vote[] }>(
      url,
      { method: 'GET' }
    );

    return response.votes;
  }

  /**
   * Cancel a proposal (proposer only)
   * 
   * @param proposalId - Proposal to cancel
   * @returns True if cancelled successfully
   */
  async cancelProposal(proposalId: string): Promise<boolean> {
    const response = await this.client.makeRequest<{ cancelled: boolean }>(
      `/api/v1/governance/proposals/${proposalId}/cancel`,
      { method: 'POST' }
    );
    return response.cancelled;
  }

  /**
   * Execute a passed proposal
   * 
   * @param proposalId - Proposal to execute
   * @returns True if executed successfully
   */
  async executeProposal(proposalId: string): Promise<boolean> {
    const response = await this.client.makeRequest<{ executed: boolean }>(
      `/api/v1/governance/proposals/${proposalId}/execute`,
      { method: 'POST' }
    );
    return response.executed;
  }

  /**
   * Delegate voting power to another address
   * 
   * @param delegateAddress - Address to delegate to
   * @param amount - Amount to delegate (optional, defaults to all)
   * @returns Delegation info
   * 
   * @example
   * const delegation = await client.governance.delegate('0xabc...', 1000);
   * console.log(`Delegated ${delegation.votingPower} voting power`);
   */
  async delegate(delegateAddress: string, amount?: number): Promise<DelegationInfo> {
    const body: Record<string, any> = { delegate: delegateAddress };
    if (amount !== undefined) {
      body.amount = amount;
    }

    return await this.client.makeRequest<DelegationInfo>(
      '/api/v1/governance/delegate',
      {
        method: 'POST',
        body
      }
    );
  }

  /**
   * Remove delegation
   * 
   * @param delegateAddress - Address to undelegate from
   * @returns True if undelegated successfully
   */
  async undelegate(delegateAddress: string): Promise<boolean> {
    const response = await this.client.makeRequest<{ undelegated: boolean }>(
      '/api/v1/governance/undelegate',
      {
        method: 'POST',
        body: { delegate: delegateAddress }
      }
    );
    return response.undelegated;
  }

  /**
   * Get user's delegations
   * 
   * @returns List of delegations
   */
  async getDelegations(): Promise<DelegationInfo[]> {
    const response = await this.client.makeRequest<{ delegations: DelegationInfo[] }>(
      '/api/v1/governance/delegations',
      { method: 'GET' }
    );
    return response.delegations;
  }

  /**
   * Get user's voting power
   * 
   * @returns Voting power in FTNS
   */
  async getVotingPower(): Promise<number> {
    const response = await this.client.makeRequest<{ votingPower: number }>(
      '/api/v1/governance/voting-power',
      { method: 'GET' }
    );
    return response.votingPower;
  }

  /**
   * Get governance statistics
   * 
   * @returns Governance statistics
   */
  async getStats(): Promise<GovernanceStats> {
    return await this.client.makeRequest<GovernanceStats>(
      '/api/v1/governance/stats',
      { method: 'GET' }
    );
  }

  /**
   * Get all active proposals
   * 
   * @returns List of active proposals
   */
  async getActiveProposals(): Promise<Proposal[]> {
    return this.listProposals({ status: ProposalStatus.ACTIVE });
  }

  /**
   * Get comments on a proposal
   * 
   * @param proposalId - Proposal identifier
   * @param limit - Maximum results
   * @returns List of comments
   */
  async getProposalComments(proposalId: string, limit?: number): Promise<any[]> {
    const queryString = limit ? `?limit=${limit}` : '';
    const response = await this.client.makeRequest<{ comments: any[] }>(
      `/api/v1/governance/proposals/${proposalId}/comments${queryString}`,
      { method: 'GET' }
    );
    return response.comments;
  }

  /**
   * Add a comment to a proposal
   * 
   * @param proposalId - Proposal identifier
   * @param comment - Comment text
   * @returns Created comment
   */
  async addComment(proposalId: string, comment: string): Promise<any> {
    return await this.client.makeRequest<any>(
      `/api/v1/governance/proposals/${proposalId}/comments`,
      {
        method: 'POST',
        body: { comment }
      }
    );
  }
}