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
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Import FTNS models from tokenomics
try:
    from prsm.economy.tokenomics.models import FTNSWallet
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
    """Types of agents in the PRSM architecture
    
    The PRSM agent framework consists of core and specialized agent types:
    Core agents:
    - ARCHITECT: Recursively decompose complex tasks into subtasks
    - PROMPTER: Optimize prompts for specific domains and models
    - ROUTER: Select optimal models for each task based on capability
    - EXECUTOR: Execute tasks using distributed specialist models
    - COMPILER: Synthesize results hierarchically into coherent responses
    
    NWTN specialized agents:
    - CANDIDATE_GENERATOR: Generate diverse candidate answers (System 1)
    - CANDIDATE_EVALUATOR: Evaluate and validate candidates (System 2)
    """
    ARCHITECT = "architect"
    PROMPTER = "prompter"
    ROUTER = "router"
    EXECUTOR = "executor"
    COMPILER = "compiler"
    CANDIDATE_GENERATOR = "candidate_generator"
    CANDIDATE_EVALUATOR = "candidate_evaluator"


class UserRole(str, Enum):
    """User roles for RBAC system"""
    ADMIN = "admin"                    # Full system access
    RESEARCHER = "researcher"          # Full AI model access
    DEVELOPER = "developer"            # Development and testing access
    ANALYST = "analyst"                # Read-only analysis access
    ENTERPRISE = "enterprise"          # Enterprise-level access with enhanced features
    USER = "user"                      # Basic user access
    GUEST = "guest"                    # Limited guest access
    PREMIUM = "premium"                # Premium user access (legacy)
    MODERATOR = "moderator"            # Moderation access (legacy)


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
        "populate_by_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    }
    
    def dict(self, **kwargs):
        """Backwards compatibility for Pydantic v1 dict() method"""
        return self.model_dump(**kwargs)
    
    def json(self, **kwargs):
        """Backwards compatibility for Pydantic v1 json() method"""
        return self.model_dump_json(**kwargs)


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
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    nwtn_context_allocation: int = Field(default=0, description="FTNS tokens allocated for context")
    context_used: int = Field(default=0, description="Context tokens consumed")
    reasoning_trace: List["ReasoningStep"] = Field(default_factory=list)
    safety_flags: List["SafetyFlag"] = Field(default_factory=list)
    status: str = "pending"
    query_count: int = Field(default=0, description="Number of queries in this session")
    total_cost: Decimal = Field(default=Decimal("0.0"), description="Total FTNS cost for this session")
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('session_id', mode='before')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id is a valid UUID format"""
        if isinstance(v, UUID):
            return str(v)
        if isinstance(v, str):
            try:
                # Try to parse as UUID to validate format
                UUID(v)
                return v
            except ValueError:
                raise ValueError(f"Invalid UUID format: {v}")
        return str(v)
    
    @field_validator('status', mode='before')
    @classmethod
    def validate_status(cls, v):
        """Validate status field accepts only valid string values"""
        valid_statuses = ["pending", "in_progress", "completed", "failed", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v


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
    flag_id: Union[UUID, str] = Field(default_factory=uuid4)
    session_id: Optional[Union[UUID, str]] = None
    level: Optional[SafetyLevel] = None
    severity: Optional[str] = None  # Alias for level
    category: Optional[str] = None
    flag_type: Optional[str] = None  # Alias for category
    description: str
    triggered_by: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Handle level/severity aliases
        if 'severity' in data and 'level' not in data:
            data['level'] = data['severity']
        elif 'level' in data and 'severity' not in data:
            data['severity'] = str(data['level'])
        
        # Handle category/flag_type aliases
        if 'flag_type' in data and 'category' not in data:
            data['category'] = data['flag_type']
        elif 'category' in data and 'flag_type' not in data:
            data['flag_type'] = data['category']
        
        super().__init__(**data)


# === Task Hierarchy Models ===

class ArchitectTask(TimestampMixin):
    """
    Enhanced from Co-Lab's SubTask for hierarchical decomposition
    """
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: Optional[UUID] = None
    user_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    level: int = Field(default=0, description="Decomposition hierarchy level")
    instruction: Optional[str] = None
    description: Optional[str] = None
    complexity: Optional[int] = None  # Integer complexity 1-5
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_cost: Optional[Decimal] = None
    dependencies: List[UUID] = Field(default_factory=list)
    status: Union[TaskStatus, str] = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Ensure task_id and parent_task_id are strings
        if 'task_id' in data and isinstance(data['task_id'], UUID):
            data['task_id'] = str(data['task_id'])
        if 'parent_task_id' in data and isinstance(data['parent_task_id'], UUID):
            data['parent_task_id'] = str(data['parent_task_id'])
        
        # Handle description/instruction aliases
        if 'description' in data and 'instruction' not in data:
            data['instruction'] = data['description']
        elif 'instruction' in data and 'description' not in data:
            data['description'] = data['instruction']
        super().__init__(**data)


class TaskHierarchy(PRSMBaseModel):
    """Complete task decomposition hierarchy"""
    root_task: ArchitectTask
    subtasks: Dict[str, ArchitectTask] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    max_depth: int = 5
    total_tasks: int = 0


class AgentTask(TimestampMixin):
    """
    Task assigned to an AI agent in the PRSM network
    
    Represents a unit of work that can be executed by an agent,
    with support for hierarchical decomposition and result tracking.
    """
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_type: Optional[AgentType] = None
    parent_task_id: Optional[str] = None
    instruction: str
    context: Dict[str, Any] = Field(default_factory=dict)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=0, ge=0, le=10)
    estimated_tokens: int = Field(default=0, ge=0)
    actual_tokens: int = Field(default=0, ge=0)
    ftns_budget: Optional[Decimal] = None
    ftns_spent: Decimal = Field(default=Decimal("0.0"))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def mark_in_progress(self):
        """Mark task as in progress"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
    
    def mark_completed(self, output_data: Optional[Dict[str, Any]] = None):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if output_data:
            self.output_data = output_data
    
    def mark_failed(self, error_message: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message


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
    event_id: Union[UUID, str] = Field(default_factory=uuid4)
    component: Optional[str] = None
    event_type: Optional[str] = None
    triggered_by: Optional[str] = None
    safety_level: Optional[SafetyLevel] = None
    reason: Optional[str] = None
    threshold_value: Optional[int] = None
    current_value: Optional[int] = None
    action_taken: Optional[str] = None
    recovery_time_seconds: Optional[int] = None
    affected_components: List[str] = Field(default_factory=list)
    resolution_action: Optional[str] = None
    resolved_at: Optional[datetime] = None


class GovernanceProposal(TimestampMixin):
    """Governance proposal for system changes"""
    proposal_id: Union[UUID, str] = Field(default_factory=uuid4)
    proposer_id: str
    title: str
    description: str
    proposal_type: str  # "safety", "economic", "technical", "governance", "parameter_change"
    voting_starts: Optional[datetime] = None
    voting_ends: Optional[datetime] = None
    voting_end_date: Optional[datetime] = None  # Alias for voting_ends
    votes_for: int = 0
    votes_against: int = 0
    total_voting_power: Decimal = Decimal("0.0")
    required_quorum: Optional[Decimal] = None
    status: str = "active"  # "active", "approved", "rejected", "executed"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Handle voting_ends/voting_end_date aliases
        if 'voting_end_date' in data and 'voting_ends' not in data:
            data['voting_ends'] = data['voting_end_date']
        elif 'voting_ends' in data and 'voting_end_date' not in data:
            data['voting_end_date'] = data['voting_ends']
        super().__init__(**data)


class Vote(TimestampMixin):
    """Individual vote on a governance proposal"""
    vote_id: Union[UUID, str] = Field(default_factory=uuid4)
    proposal_id: Union[UUID, str]
    voter_id: str
    vote: Optional[bool] = None  # True for yes, False for no
    vote_choice: Optional[bool] = None  # Alias for vote
    voting_power: Decimal
    rationale: Optional[str] = None
    reason: Optional[str] = None  # Alias for rationale
    
    def __init__(self, **data):
        # Handle vote/vote_choice aliases
        if 'vote_choice' in data and 'vote' not in data:
            data['vote'] = data['vote_choice']
        elif 'vote' in data and 'vote_choice' not in data:
            data['vote_choice'] = data['vote']
        
        # Handle rationale/reason aliases
        if 'reason' in data and 'rationale' not in data:
            data['rationale'] = data['reason']
        elif 'rationale' in data and 'reason' not in data:
            data['reason'] = data['rationale']
        
        super().__init__(**data)


# === FTNS Token Models ===

class FTNSTransaction(TimestampMixin):
    """
    Enhanced from Co-Lab's transaction model for FTNS tokens
    """
    transaction_id: Union[UUID, str] = Field(default_factory=uuid4)
    from_user: Optional[str] = None  # None for system minting
    to_user: Optional[str] = None
    user_id: Optional[str] = None  # Alias for to_user for backwards compatibility
    amount: Decimal
    transaction_type: str  # "reward", "charge", "transfer", "dividend", "fee", "refund"
    description: Optional[str] = None
    status: str = Field(default="pending")  # "pending", "processing", "confirmed", "failed", "cancelled"
    fee: Optional[Decimal] = None
    context_units: Optional[int] = None  # For context-based charges
    ipfs_cid: Optional[str] = None  # For provenance-based rewards
    block_hash: Optional[str] = None  # For distributed ledger integration
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Handle user_id/to_user aliases
        if 'user_id' in data and 'to_user' not in data:
            data['to_user'] = data['user_id']
        elif 'to_user' in data and 'user_id' not in data:
            data['user_id'] = data['to_user']
        super().__init__(**data)


class FTNSBalance(TimestampMixin):
    """User FTNS token balance"""
    user_id: str
    balance: Decimal = Field(default=Decimal("0.0"), ge=0.0)
    total_balance: Optional[Decimal] = None  # Alias for balance
    locked_balance: Decimal = Field(default=Decimal("0.0"), ge=0.0)  # For governance voting
    reserved_balance: Optional[Decimal] = None  # Alias for locked_balance
    pending_balance: Decimal = Field(default=Decimal("0.0"), ge=0.0)  # Pending transactions
    last_dividend: Optional[datetime] = None
    
    def __init__(self, **data):
        # Handle balance/total_balance aliases
        if 'total_balance' in data and 'balance' not in data:
            data['balance'] = data['total_balance']
        elif 'balance' in data and 'total_balance' not in data:
            data['total_balance'] = data['balance']
        
        # Handle locked_balance/reserved_balance aliases
        if 'reserved_balance' in data and 'locked_balance' not in data:
            data['locked_balance'] = data['reserved_balance']
        elif 'locked_balance' in data and 'reserved_balance' not in data:
            data['reserved_balance'] = data['locked_balance']
        
        super().__init__(**data)


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
    input_id: Union[UUID, str] = Field(default_factory=uuid4)
    user_id: str
    prompt: Optional[str] = None
    content: Optional[str] = None  # Alias for prompt for backwards compatibility
    context_allocation: Optional[int] = None  # FTNS tokens to spend
    preferences: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[UUID] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('content', mode='before')
    @classmethod
    def validate_content(cls, v):
        """Validate content is not empty"""
        if v == "":
            raise ValueError("Content cannot be empty string")
        return v
    
    @field_validator('prompt', mode='before')
    @classmethod
    def validate_prompt(cls, v):
        """Validate prompt is not empty"""
        if v == "":
            raise ValueError("Prompt cannot be empty string")
        return v
    
    def __init__(self, **data):
        # If content is provided but not prompt, use content as prompt
        if 'content' in data and 'prompt' not in data:
            data['prompt'] = data['content']
        # If prompt is provided but not content, set content to prompt
        elif 'prompt' in data and 'content' not in data:
            data['content'] = data['prompt']
        super().__init__(**data)


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
    ECONOMIC_EFFICIENCY = "economic_efficiency"


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