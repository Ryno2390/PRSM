"""
PRSM Teams Data Models

Defines the data structures for collaborative team functionality within PRSM.
Teams enable shared resource access, token pooling, and collaborative AI development.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from prsm.core.models import PRSMBaseModel, TimestampMixin


# === Team Enums ===

class TeamType(str, Enum):
    """Types of teams in PRSM ecosystem"""
    RESEARCH = "research"          # Academic research teams
    DEVELOPMENT = "development"    # Model development teams  
    ENTERPRISE = "enterprise"      # Corporate teams
    COMMUNITY = "community"        # Community-driven teams
    INSTITUTIONAL = "institutional"  # University/institution teams
    

class TeamRole(str, Enum):
    """Roles within a team"""
    OWNER = "owner"               # Full control, can dissolve team
    ADMIN = "admin"               # Administrative privileges
    OPERATOR = "operator"         # Can submit jobs and tasks
    TREASURER = "treasurer"       # FTNS distribution authority
    MEMBER = "member"            # Basic team member
    COLLABORATOR = "collaborator" # External collaborator
    

class TeamMembershipStatus(str, Enum):
    """Status of team membership"""
    PENDING = "pending"           # Invitation sent, not accepted
    ACTIVE = "active"             # Active team member
    INACTIVE = "inactive"         # Temporarily inactive
    SUSPENDED = "suspended"       # Suspended by team governance
    LEFT = "left"                # Member left the team
    REMOVED = "removed"          # Member removed by team


class GovernanceModel(str, Enum):
    """Team governance models from teams.md"""
    AUTOCRATIC = "autocratic"     # One founder has full control
    MERITOCRATIC = "meritocratic" # FTNS-weighted votes + code contribution
    DEMOCRATIC = "democratic"     # One user = one vote
    DAO_HYBRID = "dao_hybrid"     # Custom smart contract constitution


class RewardPolicy(str, Enum):
    """Token reward distribution policies"""
    PROPORTIONAL = "proportional"      # Based on contribution metrics
    EQUAL_SHARES = "equal_shares"      # Equal distribution
    STAKE_WEIGHTED = "stake_weighted"  # Based on FTNS contribution
    PERFORMANCE_WEIGHTED = "performance_weighted"  # Based on performance metrics
    CUSTOM = "custom"                  # Custom policy defined by team


# === Core Team Models ===

class Team(TimestampMixin):
    """
    Core team entity with shared resources and governance
    
    Based on teams.md specifications for collaborative research units.
    """
    team_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=1000)
    team_type: TeamType = TeamType.RESEARCH
    
    # Visual identity
    avatar_url: Optional[str] = None
    logo_url: Optional[str] = None
    
    # Team configuration
    governance_model: GovernanceModel = GovernanceModel.DEMOCRATIC
    reward_policy: RewardPolicy = RewardPolicy.PROPORTIONAL
    is_public: bool = True  # Discoverable in team directory
    max_members: Optional[int] = None
    
    # Financial settings
    entry_stake_required: float = Field(default=0.0, ge=0.0)  # FTNS required to join
    
    # Research focus
    research_domains: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Status
    is_active: bool = True
    founding_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Statistics (computed fields)
    member_count: int = Field(default=0, ge=0)
    total_ftns_earned: float = Field(default=0.0, ge=0.0)
    total_tasks_completed: int = Field(default=0, ge=0)
    impact_score: float = Field(default=0.0, ge=0.0)
    
    # Metadata
    external_links: Dict[str, str] = Field(default_factory=dict)
    contact_info: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Team name cannot be empty')
        return v.strip()


class TeamMember(TimestampMixin):
    """
    Team membership record with roles and contribution tracking
    """
    membership_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    user_id: str
    
    # Membership details
    role: TeamRole = TeamRole.MEMBER
    status: TeamMembershipStatus = TeamMembershipStatus.PENDING
    invited_by: Optional[str] = None
    invitation_message: Optional[str] = None
    
    # Dates
    invited_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    joined_at: Optional[datetime] = None
    left_at: Optional[datetime] = None
    
    # Contribution tracking
    ftns_contributed: float = Field(default=0.0, ge=0.0)
    tasks_completed: int = Field(default=0, ge=0)
    models_contributed: int = Field(default=0, ge=0)
    datasets_uploaded: int = Field(default=0, ge=0)
    
    # Performance metrics
    performance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reputation_score: float = Field(default=0.5, ge=0.0, le=1.0)
    collaboration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Permissions
    can_invite_members: bool = False
    can_manage_tasks: bool = False
    can_access_treasury: bool = False
    can_vote: bool = True
    
    # Metadata
    bio: Optional[str] = None
    expertise_areas: List[str] = Field(default_factory=list)
    public_profile: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamWallet(TimestampMixin):
    """
    Team's multisig wallet for shared FTNS resources
    """
    wallet_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    
    # Wallet configuration
    is_multisig: bool = True
    required_signatures: int = Field(default=1, ge=1)
    authorized_signers: List[str] = Field(default_factory=list)  # user_ids
    
    # Balances
    total_balance: float = Field(default=0.0, ge=0.0)
    available_balance: float = Field(default=0.0, ge=0.0)
    locked_balance: float = Field(default=0.0, ge=0.0)  # For ongoing tasks
    
    # Distribution policy
    reward_policy: RewardPolicy = RewardPolicy.PROPORTIONAL
    policy_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Metrics for distribution calculation
    distribution_metrics: List[str] = Field(
        default_factory=lambda: ["task_submissions", "model_contributions", "query_accuracy"]
    )
    metric_weights: List[float] = Field(
        default_factory=lambda: [0.4, 0.4, 0.2]
    )
    
    # Treasury management
    auto_distribution_enabled: bool = False
    distribution_frequency_days: int = 30
    last_distribution: Optional[datetime] = None
    
    # Security
    wallet_address: Optional[str] = None  # Blockchain address for cross-chain integration
    spending_limits: Dict[str, float] = Field(default_factory=dict)  # role -> limit
    emergency_freeze: bool = False
    
    @field_validator('metric_weights')
    @classmethod
    def validate_weights(cls, v, info):
        if v and len(v) > 0:
            if abs(sum(v) - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError('Metric weights must sum to 1.0')
        return v


class TeamTask(TimestampMixin):
    """
    Collaborative task executed by the team
    """
    task_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    
    # Task details
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    task_type: str = Field(default="research")  # research, development, training, etc.
    
    # Assignment
    assigned_to: List[str] = Field(default_factory=list)  # user_ids
    created_by: str
    priority: str = Field(default="medium")  # low, medium, high, critical
    
    # Execution
    status: str = Field(default="pending")  # pending, in_progress, completed, failed, cancelled
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # FTNS allocation
    ftns_budget: float = Field(default=0.0, ge=0.0)
    ftns_spent: float = Field(default=0.0, ge=0.0)
    
    # Deadlines
    due_date: Optional[datetime] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    
    # Results
    output_artifacts: List[str] = Field(default_factory=list)  # IPFS CIDs
    output_models: List[str] = Field(default_factory=list)     # Model IDs
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Collaboration
    requires_consensus: bool = False
    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    votes_for: int = Field(default=0, ge=0)
    votes_against: int = Field(default=0, ge=0)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    external_links: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamGovernance(TimestampMixin):
    """
    Team governance configuration and voting records
    """
    governance_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    
    # Governance configuration
    model: GovernanceModel = GovernanceModel.DEMOCRATIC
    constitution: Dict[str, Any] = Field(default_factory=dict)
    
    # Voting configuration
    voting_period_days: int = Field(default=7, ge=1, le=30)
    quorum_percentage: float = Field(default=0.5, ge=0.0, le=1.0)
    approval_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Role management
    role_assignments: Dict[str, List[str]] = Field(default_factory=dict)  # role -> user_ids
    role_term_limits: Dict[str, int] = Field(default_factory=dict)       # role -> days
    
    # Proposal types and thresholds
    proposal_types: List[str] = Field(
        default_factory=lambda: ["membership", "treasury", "governance", "technical"]
    )
    type_thresholds: Dict[str, float] = Field(default_factory=dict)
    
    # Emergency procedures
    emergency_roles: List[str] = Field(default_factory=list)  # Users with emergency powers
    emergency_procedures: Dict[str, Any] = Field(default_factory=dict)
    
    # Constitutional limits
    max_owner_power: float = Field(default=0.4, ge=0.0, le=1.0)  # Max power for any single owner
    member_protection_threshold: float = Field(default=0.25, ge=0.0, le=1.0)  # Min power for member protection
    
    # Active proposals
    active_proposals: List[UUID] = Field(default_factory=list)
    
    # Governance statistics
    total_proposals: int = Field(default=0, ge=0)
    proposals_passed: int = Field(default=0, ge=0)
    average_participation: float = Field(default=0.0, ge=0.0, le=1.0)
    last_vote: Optional[datetime] = None


class TeamProposal(TimestampMixin):
    """
    Team governance proposal
    """
    proposal_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    
    # Proposal details
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    proposal_type: str = Field(default="general")
    proposed_by: str
    
    # Voting details
    voting_starts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    voting_ends: datetime
    
    # Vote counts
    votes_for: int = Field(default=0, ge=0)
    votes_against: int = Field(default=0, ge=0)
    votes_abstain: int = Field(default=0, ge=0)
    
    # Voting power (for weighted voting)
    power_for: float = Field(default=0.0, ge=0.0)
    power_against: float = Field(default=0.0, ge=0.0)
    power_abstain: float = Field(default=0.0, ge=0.0)
    
    # Status
    status: str = Field(default="active")  # active, passed, failed, executed, cancelled
    execution_status: Optional[str] = None
    execution_date: Optional[datetime] = None
    
    # Requirements
    required_quorum: float = Field(default=0.5, ge=0.0, le=1.0)
    required_approval: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Metadata
    proposed_changes: Dict[str, Any] = Field(default_factory=dict)
    implementation_plan: Optional[str] = None
    estimated_cost: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamVote(TimestampMixin):
    """
    Individual vote on a team proposal
    """
    vote_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    team_id: UUID
    voter_id: str
    
    # Vote details
    vote: str = Field(default="for")  # for, against, abstain
    voting_power: float = Field(default=1.0, ge=0.0)
    rationale: Optional[str] = None
    
    # Delegation
    is_delegated: bool = False
    delegated_by: Optional[str] = None
    
    # Metadata
    vote_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamInvitation(TimestampMixin):
    """
    Team invitation management
    """
    invitation_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    invited_user: str
    invited_by: str
    
    # Invitation details
    role: TeamRole = TeamRole.MEMBER
    message: Optional[str] = None
    
    # Status
    status: str = Field(default="pending")  # pending, accepted, declined, expired, cancelled
    expires_at: datetime
    responded_at: Optional[datetime] = None
    
    # Requirements
    required_stake: float = Field(default=0.0, ge=0.0)
    required_skills: List[str] = Field(default_factory=list)
    
    # Metadata
    invitation_code: Optional[str] = None  # For public invitations
    metadata: Dict[str, Any] = Field(default_factory=dict)


# === Team Discovery and Directory Models ===

class TeamDirectory(PRSMBaseModel):
    """
    Team directory entry for discovery
    """
    team_id: UUID
    name: str
    description: str
    team_type: TeamType
    
    # Public metrics
    member_count: int
    impact_score: float
    total_ftns_earned: float
    research_domains: List[str]
    
    # Discovery metadata  
    is_recruiting: bool = False
    looking_for_skills: List[str] = Field(default_factory=list)
    contact_info: Dict[str, str] = Field(default_factory=dict)
    
    # Rankings
    popularity_rank: Optional[int] = None
    impact_rank: Optional[int] = None
    activity_rank: Optional[int] = None


class TeamBadge(PRSMBaseModel):
    """
    Achievement badges for teams
    """
    badge_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    
    # Badge details
    badge_type: str  # verified_team, high_impact, multidisciplinary, etc.
    title: str
    description: str
    icon_url: Optional[str] = None
    
    # Criteria
    criteria_met: Dict[str, Any] = Field(default_factory=dict)
    earned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Status
    is_active: bool = True
    expires_at: Optional[datetime] = None


# === Team Performance and Analytics ===

class TeamMetrics(TimestampMixin):
    """
    Team performance metrics and analytics
    """
    metrics_id: UUID = Field(default_factory=uuid4)
    team_id: UUID
    
    # Time period
    period_start: datetime
    period_end: datetime
    
    # Financial metrics
    ftns_earned: float = Field(default=0.0, ge=0.0)
    ftns_spent: float = Field(default=0.0, ge=0.0)
    ftns_distributed: float = Field(default=0.0, ge=0.0)
    
    # Activity metrics
    tasks_completed: int = Field(default=0, ge=0)
    models_created: int = Field(default=0, ge=0)
    datasets_contributed: int = Field(default=0, ge=0)
    research_papers: int = Field(default=0, ge=0)
    
    # Collaboration metrics
    active_members: int = Field(default=0, ge=0)
    new_members: int = Field(default=0, ge=0)
    member_retention_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    governance_participation: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Quality metrics
    average_task_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    peer_review_score: float = Field(default=0.0, ge=0.0, le=1.0)
    external_citations: int = Field(default=0, ge=0)
    
    # Impact metrics
    research_impact_score: float = Field(default=0.0, ge=0.0)
    technology_transfer_count: int = Field(default=0, ge=0)
    industry_partnerships: int = Field(default=0, ge=0)
    
    # Network metrics
    collaboration_network_size: int = Field(default=0, ge=0)
    cross_team_projects: int = Field(default=0, ge=0)
    external_partnerships: int = Field(default=0, ge=0)


# === Update forward references ===
Team.model_rebuild()
TeamMember.model_rebuild()
TeamWallet.model_rebuild()
TeamTask.model_rebuild()
TeamGovernance.model_rebuild()
TeamProposal.model_rebuild()
TeamVote.model_rebuild()
TeamInvitation.model_rebuild()