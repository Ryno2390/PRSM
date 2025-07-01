"""
PRSM Core Data Models
Enhanced from Co-Lab's models.py with PRSM-specific structures

This module defines the complete data model architecture for PRSM, including:

Core Session Management:
- PRSMSession: Enhanced sessions with FTNS context allocation
- ReasoningStep: Individual steps in the transparent reasoning trace
- SafetyFlag: Safety violations and concerns tracking

Task Hierarchy System:
- ArchitectTask: Hierarchical task decomposition structures
- TaskHierarchy: Complete decomposition trees with dependencies
- AgentResponse: Results from the 5-layer agent pipeline

Teacher Model Framework:
- TeacherModel: Distilled models for training specialists
- Curriculum: Adaptive learning paths for model improvement
- LearningSession: Training interactions between teachers and students

Safety & Governance:
- CircuitBreakerEvent: Safety system activations and responses
- GovernanceProposal: Community-driven system evolution proposals
- Vote: Democratic decision-making records

FTNS Token Economy:
- FTNSTransaction: Complete token transaction history
- FTNSBalance: User balances with locked/unlocked amounts
- ContextUsage: Detailed tracking of computational resource usage

P2P Federation:
- PeerNode: Distributed network peer information
- ModelShard: Distributed model storage and verification

Performance Monitoring:
- PerformanceMetric: System health and efficiency tracking
- ImprovementOpportunity: Automated enhancement suggestions
- ImprovementProposal: Structured system improvement workflow

Advanced Tokenomics:
- ImpactMetrics: Research impact and citation tracking
- MarketplaceListing: Model rental marketplace integration
- RoyaltyPayment: Content creator compensation system

Institutional Dynamics (NEW):
- InstitutionalParticipant: Enterprise-scale participants with tiers and quotas
- ProvenanceNode: Strategic provenance tracking for competitive incentives
- ConcentrationMetrics: Anti-monopoly monitoring and enforcement
- RevenueDistribution: Strategic revenue sharing to drive adoption

Integration Layer:
- IntegrationSource: External platform content with security assessments
- SecurityScanResult: Comprehensive security validation results
- CredentialData: Encrypted storage for API keys and OAuth tokens

All models follow the PRSMBaseModel pattern with:
- Consistent UUID-based identification
- Timestamp tracking for audit trails  
- Pydantic validation for data integrity
- JSON serialization for distributed communication
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Import FTNS models from tokenomics
try:
    from ..tokenomics.models import FTNSWallet
except ImportError:
    # Fallback if tokenomics models not available
    FTNSWallet = None

# Import SQLAlchemy Base and Session for database models
try:
    from .database import Base
    from sqlalchemy.orm import Session
except ImportError:
    # Fallback if database not available
    Base = None
    Session = None


# === Enums ===

class TaskStatus(str, Enum):
    """Status of tasks in the PRSM system
    
    Used throughout the task hierarchy to track progress:
    - PENDING: Task created but not yet started
    - IN_PROGRESS: Task currently being executed by an agent
    - COMPLETED: Task finished successfully with results
    - FAILED: Task execution failed due to error or timeout
    - CANCELLED: Task cancelled before completion (manual or system)
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Types of agents in the 5-layer PRSM architecture
    
    The PRSM agent framework consists of five specialized agent types:
    - ARCHITECT: Recursively decompose complex tasks into subtasks
    - PROMPTER: Optimize prompts for specific domains and models
    - ROUTER: Select optimal models for each task based on capability
    - EXECUTOR: Execute tasks using distributed specialist models
    - COMPILER: Synthesize results hierarchically into coherent responses
    """
    ARCHITECT = "architect"
    PROMPTER = "prompter"
    ROUTER = "router"
    EXECUTOR = "executor"
    COMPILER = "compiler"


class UserRole(str, Enum):
    """User roles for RBAC system"""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    DEVELOPER = "developer"
    MODERATOR = "moderator"


class SafetyLevel(str, Enum):
    """Safety levels for circuit breaker system
    
    Graduated safety response levels for the distributed circuit breaker:
    - NONE: No safety concerns detected
    - LOW: Minor issues that should be logged but don't require action
    - MEDIUM: Moderate concerns requiring monitoring and potential intervention
    - HIGH: Serious issues requiring immediate circuit breaker activation
    - CRITICAL: Emergency situations requiring network-wide halt
    """
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelType(str, Enum):
    """Types of models in PRSM ecosystem
    
    Classification system for models in the PRSM network:
    - TEACHER: Distilled models that train other models using RLVR
    - STUDENT: Models being trained by teacher models
    - SPECIALIST: Domain-specific models optimized for particular tasks
    - GENERAL: General-purpose models for broad capability tasks
    """
    TEACHER = "teacher"
    STUDENT = "student"
    SPECIALIST = "specialist"
    GENERAL = "general"


# === Base Models ===

class PRSMBaseModel(BaseModel):
    """Base model for all PRSM entities
    
    Provides standardized configuration for all PRSM data models:
    - Automatic attribute mapping from external sources
    - Enum value serialization for API compatibility
    - Consistent datetime and UUID JSON encoding
    - Field name validation and normalization
    
    All PRSM models inherit from this base to ensure consistency
    across the distributed system components.
    """
    
    model_config = {
        "from_attributes": True,
        "use_enum_values": True,
        "validate_by_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    }


class TimestampMixin(PRSMBaseModel):
    """Mixin for models that need timestamps
    
    Automatically tracks creation and modification times for audit trails.
    Essential for:
    - Distributed consensus and ordering
    - Performance analysis and debugging  
    - Governance and voting record keeping
    - FTNS transaction history verification
    """
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None


# === Core PRSM Models ===

class PRSMSession(TimestampMixin):
    """
    Enhanced session model for NWTN interactions
    Extends Co-Lab's session concept with context allocation
    
    Core session management for PRSM queries:
    - FTNS token allocation for computational resources
    - Complete reasoning trace for transparency and debugging
    - Safety flag tracking for circuit breaker integration
    - Context usage monitoring for economic optimization
    
    Each session represents a complete user interaction from
    initial query through final response delivery.
    """
    session_id: UUID = Field(default_factory=uuid4)
    user_id: str
    nwtn_context_allocation: int = Field(default=0, description="FTNS tokens allocated for context")
    context_used: int = Field(default=0, description="Context tokens consumed")
    reasoning_trace: List["ReasoningStep"] = Field(default_factory=list)
    safety_flags: List["SafetyFlag"] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningStep(PRSMBaseModel):
    """Individual step in the reasoning trace
    
    Provides complete transparency into PRSM's decision-making process.
    Each step records:
    - Which agent performed the operation
    - Input data and processing parameters
    - Output results and metadata
    - Execution time and confidence metrics
    
    Essential for:
    - User trust through transparent reasoning
    - System debugging and optimization
    - Academic reproducibility requirements
    - Governance auditing and validation
    """
    step_id: UUID = Field(default_factory=uuid4)
    agent_type: AgentType
    agent_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    confidence_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyFlag(PRSMBaseModel):
    """Safety violation or concern"""
    flag_id: UUID = Field(default_factory=uuid4)
    level: SafetyLevel
    category: str
    description: str
    triggered_by: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False


# === Task Hierarchy Models ===

class ArchitectTask(TimestampMixin):
    """
    Enhanced from Co-Lab's SubTask for hierarchical decomposition
    """
    task_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    parent_task_id: Optional[UUID] = None
    level: int = Field(default=0, description="Decomposition hierarchy level")
    instruction: str
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dependencies: List[UUID] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskHierarchy(PRSMBaseModel):
    """Complete task decomposition hierarchy"""
    root_task: ArchitectTask
    subtasks: Dict[str, ArchitectTask] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    max_depth: int = 5
    total_tasks: int = 0


# === Agent Models ===

class AgentResponse(TimestampMixin):
    """
    Enhanced from Co-Lab's SubAIResponse for PRSM agents
    """
    response_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    agent_type: AgentType
    input_data: str
    output_data: Optional[Any] = None
    success: bool = True
    error_message: Optional[str] = None
    processing_time: float = Field(default=0.0)
    safety_validated: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(TimestampMixin):
    """Performance metrics for agent operations"""
    metric_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    agent_type: AgentType
    operation_name: str
    duration_seconds: float
    success: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompilerResult(TimestampMixin):
    """Result from hierarchical compilation"""
    result_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    compilation_level: str  # "elemental", "mid", "final"
    input_count: int = Field(default=0)
    compiled_result: Optional[Any] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_trace: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# === Teacher Model System ===

class TeacherModel(TimestampMixin):
    """Distilled teacher model for training other models"""
    teacher_id: UUID = Field(default_factory=uuid4)
    name: str
    specialization: str
    model_type: ModelType = ModelType.TEACHER
    performance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    curriculum_ids: List[UUID] = Field(default_factory=list)
    student_models: List[UUID] = Field(default_factory=list)
    rlvr_score: Optional[float] = None
    ipfs_cid: Optional[str] = None
    version: str = "1.0.0"
    active: bool = True


class Curriculum(TimestampMixin):
    """Training curriculum generated by teacher models"""
    curriculum_id: UUID = Field(default_factory=uuid4)
    teacher_id: UUID
    domain: str
    difficulty_level: float = Field(ge=0.0, le=1.0)
    training_examples: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict)
    effectiveness_score: Optional[float] = None


class LearningSession(TimestampMixin):
    """Training session between teacher and student"""
    session_id: UUID = Field(default_factory=uuid4)
    teacher_id: UUID
    student_id: UUID
    curriculum_id: UUID
    performance_before: Dict[str, float] = Field(default_factory=dict)
    performance_after: Dict[str, float] = Field(default_factory=dict)
    learning_gain: Optional[float] = None
    completed: bool = False


# === Safety & Governance Models ===

class CircuitBreakerEvent(TimestampMixin):
    """Circuit breaker activation event"""
    event_id: UUID = Field(default_factory=uuid4)
    triggered_by: str
    safety_level: SafetyLevel
    reason: str
    affected_components: List[str] = Field(default_factory=list)
    resolution_action: Optional[str] = None
    resolved_at: Optional[datetime] = None


class GovernanceProposal(TimestampMixin):
    """Governance proposal for system changes"""
    proposal_id: UUID = Field(default_factory=uuid4)
    proposer_id: str
    title: str
    description: str
    proposal_type: str  # "safety", "economic", "technical", "governance"
    voting_starts: datetime
    voting_ends: datetime
    votes_for: int = 0
    votes_against: int = 0
    total_voting_power: float = 0.0
    status: str = "active"  # "active", "approved", "rejected", "executed"


class Vote(TimestampMixin):
    """Individual vote on a governance proposal"""
    vote_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    voter_id: str
    vote: bool  # True for yes, False for no
    voting_power: float
    rationale: Optional[str] = None


# === FTNS Token Models ===

class FTNSTransaction(TimestampMixin):
    """
    Enhanced from Co-Lab's transaction model for FTNS tokens
    """
    transaction_id: UUID = Field(default_factory=uuid4)
    from_user: Optional[str] = None  # None for system minting
    to_user: str
    amount: float
    transaction_type: str  # "reward", "charge", "transfer", "dividend"
    description: str
    context_units: Optional[int] = None  # For context-based charges
    ipfs_cid: Optional[str] = None  # For provenance-based rewards
    block_hash: Optional[str] = None  # For distributed ledger integration


class FTNSBalance(TimestampMixin):
    """User FTNS token balance"""
    user_id: str
    balance: float = Field(default=0.0, ge=0.0)
    locked_balance: float = Field(default=0.0, ge=0.0)  # For governance voting
    last_dividend: Optional[datetime] = None


class ContextUsage(TimestampMixin):
    """Context usage tracking for NWTN sessions"""
    session_id: UUID
    user_id: str
    context_allocated: int
    context_used: int = Field(default=0)
    ftns_cost: float = Field(default=0.0)
    ftns_charged: float = Field(default=0.0)
    allocation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completion_time: Optional[datetime] = None
    usage_stages: List[Dict[str, Any]] = Field(default_factory=list)


class ProvenanceRecord(TimestampMixin):
    """Track content usage for royalty payments"""
    record_id: UUID = Field(default_factory=uuid4)
    content_cid: str
    uploader_id: str
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None
    total_rewards_paid: float = Field(default=0.0)


# === P2P Federation Models ===

class PeerNode(TimestampMixin):
    """P2P network peer information"""
    node_id: str
    peer_id: str
    multiaddr: str
    capabilities: List[str] = Field(default_factory=list)
    reputation_score: float = Field(default=0.5, ge=0.0, le=1.0)
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True


class ModelShard(TimestampMixin):
    """Distributed model shard information"""
    shard_id: UUID = Field(default_factory=uuid4)
    model_cid: str
    shard_index: int
    total_shards: int
    hosted_by: List[str] = Field(default_factory=list)  # Node IDs
    verification_hash: str
    size_bytes: int


class User(PRSMBaseModel):
    """User model for authentication and authorization"""
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="User display name")
    email: Optional[str] = Field(None, description="User email address")
    role: UserRole = Field(default=UserRole.USER, description="User role for RBAC")
    is_active: bool = Field(default=True, description="Whether user account is active")
    is_premium: bool = Field(default=False, description="Whether user has premium access")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")


# === Request/Response Models ===

class UserInput(PRSMBaseModel):
    """
    Enhanced from Co-Lab's UserInput for NWTN queries
    """
    user_id: str
    prompt: str
    context_allocation: Optional[int] = None  # FTNS tokens to spend
    preferences: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[UUID] = None


class ClarifiedPrompt(PRSMBaseModel):
    """NWTN's clarification of user input"""
    original_prompt: str
    clarified_prompt: str
    intent_category: str
    complexity_estimate: float
    context_required: int
    suggested_agents: List[AgentType] = Field(default_factory=list)


class PRSMResponse(TimestampMixin):
    """
    Enhanced from Co-Lab's FinalResponse for PRSM
    """
    session_id: UUID
    user_id: str
    final_answer: str
    reasoning_trace: List[ReasoningStep] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    context_used: int
    ftns_charged: float
    sources: List[str] = Field(default_factory=list)
    safety_validated: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


# === Performance Monitoring Models ===

class MetricType(str, Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    LATENCY = "latency" 
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"
    COST_EFFICIENCY = "cost_efficiency"


class ImprovementType(str, Enum):
    """Types of system improvements"""
    ARCHITECTURE = "architecture"
    HYPERPARAMETER = "hyperparameter"
    TRAINING_DATA = "training_data"
    MODEL_SIZE = "model_size"
    OPTIMIZATION = "optimization"
    SAFETY_ENHANCEMENT = "safety_enhancement"


class PerformanceMetric(BaseModel):
    """Individual performance metric measurement"""
    metric_id: UUID = Field(default_factory=uuid4)
    model_id: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = Field(default_factory=dict)
    baseline_value: Optional[float] = None
    improvement_percentage: Optional[float] = None


class ComparisonReport(BaseModel):
    """Report comparing model performance against baselines"""
    report_id: UUID = Field(default_factory=uuid4) 
    model_id: str
    baseline_model_id: str
    comparison_metrics: List[PerformanceMetric]
    overall_improvement: float
    significant_improvements: List[str]
    regressions: List[str]
    recommendation: str
    confidence_score: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImprovementOpportunity(BaseModel):
    """Identified opportunity for system improvement"""
    opportunity_id: UUID = Field(default_factory=uuid4)
    improvement_type: ImprovementType
    target_component: str
    current_performance: float
    expected_improvement: float
    confidence: float
    implementation_cost: float
    priority_score: float
    description: str
    supporting_data: Dict[str, Any] = Field(default_factory=dict)
    identified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceAnalysis(BaseModel):
    """Comprehensive performance analysis results"""
    analysis_id: UUID = Field(default_factory=uuid4)
    model_id: str
    analysis_period_start: datetime
    analysis_period_end: datetime
    metrics_analyzed: List[PerformanceMetric]
    trends: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    improvement_opportunities: List[ImprovementOpportunity]
    overall_health_score: float
    recommendations: List[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProposalStatus(str, Enum):
    """Status of improvement proposals"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    SIMULATING = "simulating"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTING = "implementing"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationResult(BaseModel):
    """Results from simulating a proposed improvement"""
    simulation_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    simulation_duration: float  # seconds
    predicted_performance_change: Dict[str, float]
    resource_requirements: Dict[str, float]
    risk_assessment: Dict[str, float]
    confidence_score: float
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)
    simulation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyCheck(BaseModel):
    """Safety validation results for improvement proposals"""
    check_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    safety_score: float  # 0-1, higher is safer
    potential_risks: List[str]
    risk_mitigation_strategies: List[str]
    circuit_breaker_impact: Dict[str, Any] = Field(default_factory=dict)
    governance_requirements: List[str]
    approval_required: bool
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImprovementProposal(BaseModel):
    """Proposed improvement to the PRSM system"""
    proposal_id: UUID = Field(default_factory=uuid4)
    improvement_type: ImprovementType
    target_component: str
    title: str
    description: str
    technical_details: Dict[str, Any] = Field(default_factory=dict)
    expected_benefits: Dict[str, float]
    implementation_cost: float
    timeline_estimate: int  # days
    priority_score: float
    status: ProposalStatus = ProposalStatus.PENDING
    
    # Analysis results
    weakness_analysis: Dict[str, Any] = Field(default_factory=dict)
    simulation_result: Optional[SimulationResult] = None
    safety_check: Optional[SafetyCheck] = None
    
    # Metadata
    proposed_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approval_history: List[Dict[str, Any]] = Field(default_factory=list)


class UpdatePackage(BaseModel):
    """Package for distributing system updates across the network"""
    package_id: UUID = Field(default_factory=uuid4)
    proposals_included: List[UUID]
    update_type: str  # "hotfix", "feature", "optimization", "security"
    version: str
    update_files: List[str]
    rollback_plan: Dict[str, Any] = Field(default_factory=dict)
    target_nodes: List[str]
    deployment_strategy: str  # "rolling", "blue_green", "canary"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestResults(BaseModel):
    """Results from A/B testing of improvement proposals"""
    test_id: UUID = Field(default_factory=uuid4)
    proposals_tested: List[UUID]
    test_duration: float  # hours
    sample_size: int
    control_group_performance: Dict[str, float]
    treatment_group_performance: Dict[str, float]
    statistical_significance: Dict[str, float]
    winner: Optional[UUID] = None
    confidence_interval: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    test_completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# === Advanced Tokenomics Models ===

class ImpactMetrics(BaseModel):
    """Metrics for tracking research and content impact"""
    content_id: str
    total_citations: int = 0
    total_downloads: int = 0
    total_usage_hours: float = 0.0
    unique_users: int = 0
    impact_score: float = 0.0
    academic_citations: int = 0
    industry_applications: int = 0
    geographical_reach: int = 0  # Number of countries
    time_to_first_citation: Optional[float] = None  # Days
    collaboration_coefficient: float = 0.0
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PricingModel(BaseModel):
    """Pricing model for marketplace transactions"""
    pricing_id: UUID = Field(default_factory=uuid4)
    base_price: float
    currency: str = "FTNS"
    pricing_type: str = Field(..., description="hourly, usage, subscription, one_time")
    usage_tiers: Dict[str, float] = Field(default_factory=dict)
    volume_discounts: Dict[str, float] = Field(default_factory=dict)
    minimum_duration: Optional[float] = None  # Hours
    maximum_duration: Optional[float] = None  # Hours
    dynamic_pricing_enabled: bool = False
    demand_multiplier: float = 1.0
    peak_hour_multiplier: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketplaceListing(BaseModel):
    """Marketplace listing for model rentals"""
    listing_id: UUID = Field(default_factory=uuid4)
    model_id: str
    owner_id: str
    title: str
    description: str
    pricing_model: PricingModel
    availability_status: str = Field(default="available", description="available, rented, maintenance")
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    supported_features: List[str] = Field(default_factory=list)
    terms_of_service: str = ""
    maximum_concurrent_users: int = 1
    geographical_restrictions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketplaceTransaction(BaseModel):
    """Transaction record for marketplace activity"""
    transaction_id: UUID = Field(default_factory=uuid4)
    listing_id: UUID
    buyer_id: str
    seller_id: str
    transaction_type: str = Field(..., description="rental, purchase, subscription")
    amount: float
    currency: str = "FTNS"
    duration: Optional[float] = None  # Hours for rentals
    platform_fee: float = 0.0
    status: str = Field(default="pending", description="pending, completed, failed, refunded")
    payment_method: str = "ftns_balance"
    escrow_amount: float = 0.0
    usage_metrics: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DividendDistribution(BaseModel):
    """Quarterly dividend distribution record"""
    distribution_id: UUID = Field(default_factory=uuid4)
    quarter: str  # e.g., "2025-Q2"
    total_pool: float
    distribution_date: datetime
    eligible_holders: List[str]
    distribution_amounts: Dict[str, float] = Field(default_factory=dict)
    distribution_method: str = "proportional"
    minimum_holding_period: int = 30  # Days
    minimum_balance: float = 1.0
    bonus_multipliers: Dict[str, float] = Field(default_factory=dict)
    status: str = Field(default="pending", description="pending, processing, completed")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RoyaltyPayment(BaseModel):
    """Royalty payment for content usage"""
    payment_id: UUID = Field(default_factory=uuid4)
    content_id: str
    creator_id: str
    usage_period_start: datetime
    usage_period_end: datetime
    total_usage: float
    usage_type: str = Field(..., description="download, citation, computation, derived_work")
    royalty_rate: float
    base_amount: float
    bonus_amount: float = 0.0
    total_amount: float
    impact_multiplier: float = 1.0
    quality_score: float = 1.0
    status: str = Field(default="pending", description="pending, paid, disputed")
    payment_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# === Update forward references ===
PRSMSession.model_rebuild()
TaskHierarchy.model_rebuild()
CompilerResult.model_rebuild()
PRSMResponse.model_rebuild()