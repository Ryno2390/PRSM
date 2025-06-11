"""
PRSM Distillation Data Models
Comprehensive data structures for the automated distillation system
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ..core.models import PRSMBaseModel, TimestampMixin


class TrainingMetrics(BaseModel):
    """Metrics collected during training"""
    step: int
    loss: float
    accuracy: float = 0.0
    distillation_loss: float = 0.0
    student_loss: float = 0.0
    learning_rate: float = 0.0
    temperature: float = 4.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DistillationStatus(str, Enum):
    """Status of distillation jobs"""
    QUEUED = "queued"
    ANALYZING_TEACHER = "analyzing_teacher"
    GENERATING_ARCHITECTURE = "generating_architecture"
    TRAINING = "training"
    EVALUATING = "evaluating"
    VALIDATING_SAFETY = "validating_safety"
    OPTIMIZING = "optimizing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationTarget(str, Enum):
    """Optimization targets for distilled models"""
    SPEED = "speed"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    SIZE = "size"
    BALANCED = "balanced"


class ModelSize(str, Enum):
    """Target model sizes"""
    TINY = "tiny"        # < 100M parameters
    SMALL = "small"      # 100M - 1B parameters
    MEDIUM = "medium"    # 1B - 10B parameters
    LARGE = "large"      # 10B+ parameters


class TrainingStrategy(str, Enum):
    """Training strategies for distillation"""
    BASIC = "basic"
    PROGRESSIVE = "progressive"
    ENSEMBLE = "ensemble"
    ADVERSARIAL = "adversarial"
    CURRICULUM = "curriculum"
    SELF_SUPERVISED = "self_supervised"


class DistillationRequest(PRSMBaseModel):
    """
    User request for automated distillation
    
    Comprehensive specification for creating a distilled model including
    teacher model selection, optimization targets, economic parameters,
    and deployment requirements.
    """
    request_id: UUID = Field(default_factory=uuid4)
    user_id: str
    
    # Teacher model specification
    teacher_model: str = Field(..., description="Teacher model identifier or API endpoint")
    teacher_models: Optional[List[Dict[str, Any]]] = Field(default=None, description="Multi-teacher ensemble")
    
    # Domain and specialization
    domain: str = Field(..., description="Target domain (e.g., 'medical_research', 'legal_analysis')")
    specialization: Optional[str] = Field(default=None, description="Specific area within domain")
    use_case: Optional[str] = Field(default=None, description="Intended use case")
    
    # Architecture requirements
    target_size: ModelSize = ModelSize.SMALL
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
    target_architecture: Optional[str] = Field(default="transformer", description="Model architecture type")
    
    # Custom architecture parameters
    layer_count: Optional[int] = Field(default=None, ge=1, le=50)
    attention_heads: Optional[int] = Field(default=None, ge=1, le=32)
    hidden_size: Optional[int] = Field(default=None, ge=128, le=4096)
    vocabulary_size: Optional[int] = Field(default=None, ge=1000, le=100000)
    
    # Training configuration
    training_strategy: TrainingStrategy = TrainingStrategy.PROGRESSIVE
    augmentation_techniques: List[str] = Field(default_factory=list)
    custom_training_data: Optional[str] = Field(default=None, description="IPFS CID for custom training data")
    
    # Performance requirements
    max_inference_latency: Optional[str] = Field(default=None, description="Maximum acceptable latency (e.g., '100ms')")
    max_memory_usage: Optional[str] = Field(default=None, description="Maximum memory usage (e.g., '1GB')")
    target_hardware: List[str] = Field(default_factory=lambda: ["cpu"], description="Target deployment hardware")
    
    # Quality requirements
    quality_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Minimum quality vs teacher")
    safety_level: str = Field(default="standard", description="Required safety validation level")
    
    # Economic parameters
    budget_ftns: int = Field(..., ge=100, description="Maximum FTNS tokens to spend")
    revenue_sharing: float = Field(default=0.0, ge=0.0, le=1.0, description="Revenue share with teacher owners")
    marketplace_listing: bool = Field(default=True, description="Automatically list in marketplace")
    pricing_model: Optional[str] = Field(default="usage_based", description="Pricing strategy for marketplace")
    
    # Advanced options
    preserve_existing_knowledge: bool = Field(default=True, description="For incremental updates")
    enable_continual_learning: bool = Field(default=False, description="Support for online learning")
    compression_techniques: List[str] = Field(default_factory=list, description="Additional compression methods")
    
    # Metadata
    name: Optional[str] = Field(default=None, description="Human-readable model name")
    description: Optional[str] = Field(default=None, description="Model description")
    tags: List[str] = Field(default_factory=list, description="Search tags")


class TeacherAnalysis(TimestampMixin):
    """
    Analysis results for teacher model capabilities
    
    Comprehensive assessment of teacher model to guide distillation process.
    """
    analysis_id: UUID = Field(default_factory=uuid4)
    teacher_model: str
    
    # Capability analysis
    identified_capabilities: List[str] = Field(default_factory=list)
    domain_expertise: Dict[str, float] = Field(default_factory=dict)
    knowledge_areas: List[str] = Field(default_factory=list)
    reasoning_patterns: List[str] = Field(default_factory=list)
    
    # Architecture insights
    estimated_parameters: Optional[int] = None
    attention_patterns: Dict[str, Any] = Field(default_factory=dict)
    layer_importance: List[float] = Field(default_factory=list)
    bottleneck_layers: List[int] = Field(default_factory=list)
    
    # Performance characteristics
    inference_speed: Optional[float] = None  # tokens/second
    memory_usage: Optional[int] = None       # MB
    computational_complexity: Optional[str] = None
    
    # Distillation feasibility
    distillation_difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    recommended_compression_ratio: float = Field(default=10.0, ge=1.0, le=1000.0)
    critical_knowledge_areas: List[str] = Field(default_factory=list)
    
    # Quality metrics
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_score: float = Field(default=0.0, ge=0.0, le=1.0)


class StudentArchitecture(TimestampMixin):
    """
    Generated architecture for student model
    
    Optimal architecture design based on requirements and teacher analysis.
    """
    architecture_id: UUID = Field(default_factory=uuid4)
    distillation_request_id: UUID
    
    # Core architecture
    model_type: str = Field(default="transformer")
    layer_count: int = Field(..., ge=1, le=50)
    attention_heads: int = Field(..., ge=1, le=32)
    hidden_size: int = Field(..., ge=128, le=4096)
    intermediate_size: int = Field(..., ge=256, le=8192)
    vocabulary_size: int = Field(..., ge=1000, le=100000)
    
    # Specialized components
    specialized_layers: List[Dict[str, Any]] = Field(default_factory=list)
    compression_techniques: List[str] = Field(default_factory=list)
    optimization_strategies: List[str] = Field(default_factory=list)
    
    # Performance predictions
    estimated_parameters: int = Field(..., ge=1000000)
    estimated_size_mb: float = Field(..., ge=1.0)
    estimated_inference_speed: float = Field(..., ge=1.0)  # tokens/second
    estimated_memory_usage: int = Field(..., ge=100)       # MB
    
    # Quality predictions
    predicted_accuracy: float = Field(ge=0.0, le=1.0)
    compression_ratio: float = Field(ge=1.0, le=1000.0)
    efficiency_score: float = Field(ge=0.0, le=1.0)
    
    # Architecture rationale
    design_decisions: Dict[str, str] = Field(default_factory=dict)
    trade_off_analysis: Dict[str, Any] = Field(default_factory=dict)
    alternative_architectures: List[Dict[str, Any]] = Field(default_factory=list)


class TrainingConfig(PRSMBaseModel):
    """
    Training configuration for distillation process
    
    Comprehensive training parameters optimized for the specific distillation task.
    """
    config_id: UUID = Field(default_factory=uuid4)
    
    # Training strategy
    strategy: TrainingStrategy = TrainingStrategy.PROGRESSIVE
    num_epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-1)
    
    # Distillation parameters
    distillation_temperature: float = Field(default=3.0, ge=1.0, le=10.0)
    alpha_knowledge_distillation: float = Field(default=0.7, ge=0.0, le=1.0)
    alpha_student_loss: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Multi-teacher parameters
    teacher_weights: Dict[str, float] = Field(default_factory=dict)
    knowledge_alignment: bool = Field(default=True)
    ensemble_method: Optional[str] = Field(default="weighted_average")
    
    # Optimization settings
    optimizer: str = Field(default="adamw")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    warmup_steps: int = Field(default=1000, ge=0, le=10000)
    gradient_clipping: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Regularization
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    attention_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    layer_dropout: float = Field(default=0.0, ge=0.0, le=0.3)
    
    # Data augmentation
    augmentation_techniques: List[str] = Field(default_factory=list)
    augmentation_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    
    # Validation and checkpointing
    validation_frequency: int = Field(default=500, ge=100, le=5000)
    checkpoint_frequency: int = Field(default=1000, ge=500, le=10000)
    early_stopping_patience: int = Field(default=3, ge=1, le=10)
    
    # Resource constraints
    max_training_time_hours: int = Field(default=24, ge=1, le=168)  # 1 week max
    memory_limit_gb: int = Field(default=16, ge=4, le=128)
    compute_budget_ftns: int = Field(default=500, ge=100, le=10000)


class QualityMetrics(TimestampMixin):
    """
    Comprehensive quality assessment for distilled models
    
    Multi-dimensional evaluation covering accuracy, efficiency, safety, and usability.
    """
    metrics_id: UUID = Field(default_factory=uuid4)
    model_id: str
    
    # Core performance metrics
    accuracy_score: float = Field(ge=0.0, le=1.0)
    precision_score: float = Field(ge=0.0, le=1.0)
    recall_score: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    
    # Efficiency metrics
    inference_latency_ms: float = Field(ge=0.0)
    throughput_tokens_per_sec: float = Field(ge=0.0)
    memory_usage_mb: int = Field(ge=0)
    energy_efficiency_score: float = Field(ge=0.0, le=1.0)
    
    # Quality metrics
    knowledge_retention: float = Field(ge=0.0, le=1.0, description="% of teacher knowledge retained")
    coherence_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    fluency_score: float = Field(ge=0.0, le=1.0)
    
    # Comparative metrics
    teacher_model_comparison: Dict[str, float] = Field(default_factory=dict)
    compression_ratio: float = Field(ge=1.0)
    speed_improvement: float = Field(ge=1.0)
    cost_efficiency: float = Field(ge=0.0, le=1.0)
    
    # Domain-specific metrics
    domain_accuracy: Dict[str, float] = Field(default_factory=dict)
    specialized_capability_scores: Dict[str, float] = Field(default_factory=dict)
    edge_case_handling: float = Field(ge=0.0, le=1.0)
    
    # Robustness metrics
    adversarial_robustness: float = Field(ge=0.0, le=1.0)
    noise_tolerance: float = Field(ge=0.0, le=1.0)
    out_of_distribution_detection: float = Field(ge=0.0, le=1.0)
    
    # User experience metrics
    usability_score: float = Field(ge=0.0, le=1.0)
    api_compatibility: float = Field(ge=0.0, le=1.0)
    documentation_quality: float = Field(ge=0.0, le=1.0)
    
    # Overall assessment
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    deployment_readiness: bool = Field(default=False)
    recommended_improvements: List[str] = Field(default_factory=list)


class SafetyAssessment(TimestampMixin):
    """
    Safety validation results for distilled models
    
    Comprehensive safety evaluation ensuring model compliance with PRSM safety standards.
    """
    assessment_id: UUID = Field(default_factory=uuid4)
    model_id: str
    
    # Circuit breaker integration
    circuit_breaker_compliance: bool = Field(default=False)
    safety_threshold_compliance: bool = Field(default=False)
    emergency_halt_capability: bool = Field(default=False)
    
    # Content safety
    harmful_content_detection: float = Field(ge=0.0, le=1.0)
    bias_detection_score: float = Field(ge=0.0, le=1.0)
    toxicity_score: float = Field(ge=0.0, le=1.0)
    privacy_protection_score: float = Field(ge=0.0, le=1.0)
    
    # Output safety
    unsafe_output_rate: float = Field(ge=0.0, le=1.0)
    safety_filter_effectiveness: float = Field(ge=0.0, le=1.0)
    prompt_injection_resistance: float = Field(ge=0.0, le=1.0)
    jailbreak_resistance: float = Field(ge=0.0, le=1.0)
    
    # Behavioral safety
    consistency_under_pressure: float = Field(ge=0.0, le=1.0)
    ethical_reasoning_capability: float = Field(ge=0.0, le=1.0)
    refusal_appropriateness: float = Field(ge=0.0, le=1.0)
    
    # Technical safety
    input_validation_robustness: float = Field(ge=0.0, le=1.0)
    output_format_compliance: float = Field(ge=0.0, le=1.0)
    error_handling_quality: float = Field(ge=0.0, le=1.0)
    
    # Audit and compliance
    decision_traceability: bool = Field(default=False)
    governance_compliance: bool = Field(default=False)
    audit_trail_completeness: float = Field(ge=0.0, le=1.0)
    
    # Safety recommendations
    identified_risks: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    monitoring_requirements: List[str] = Field(default_factory=list)
    
    # Overall safety
    overall_safety_score: float = Field(ge=0.0, le=1.0)
    deployment_safety_approved: bool = Field(default=False)
    safety_certification_level: str = Field(default="pending")


class DistillationJob(TimestampMixin):
    """
    Complete distillation job tracking
    
    Central record for the entire distillation process from request to deployment.
    """
    job_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    user_id: str
    
    # Job status
    status: DistillationStatus = DistillationStatus.QUEUED
    progress_percentage: int = Field(default=0, ge=0, le=100)
    current_stage: str = Field(default="queued")
    estimated_completion: Optional[datetime] = None
    
    # Process tracking
    teacher_analysis_id: Optional[UUID] = None
    architecture_id: Optional[UUID] = None
    training_config_id: Optional[UUID] = None
    
    # Results
    final_model_id: Optional[str] = None
    quality_metrics_id: Optional[UUID] = None
    safety_assessment_id: Optional[UUID] = None
    
    # Resource usage
    compute_time_hours: float = Field(default=0.0, ge=0.0)
    ftns_spent: int = Field(default=0, ge=0)
    storage_used_gb: float = Field(default=0.0, ge=0.0)
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = Field(default=0, ge=0)
    
    # Deployment info
    marketplace_listing_id: Optional[str] = None
    deployment_endpoints: List[str] = Field(default_factory=list)
    model_version: str = Field(default="1.0.0")
    
    # Notifications and communication
    notification_preferences: Dict[str, bool] = Field(default_factory=dict)
    progress_updates: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    priority: str = Field(default="normal")


class DistillationJobStatus(PRSMBaseModel):
    """
    Real-time status information for distillation jobs
    
    Provides current status, progress, and next steps for ongoing distillation.
    """
    job_id: UUID
    status: DistillationStatus
    progress: int = Field(ge=0, le=100)
    current_stage: str
    stage_progress: int = Field(ge=0, le=100)
    
    # Time estimates
    elapsed_time_minutes: int = Field(ge=0)
    estimated_remaining_minutes: Optional[int] = None
    estimated_completion: Optional[datetime] = None
    
    # Resource usage
    current_ftns_spent: int = Field(ge=0)
    estimated_total_cost: int = Field(ge=0)
    compute_resources_used: Dict[str, Any] = Field(default_factory=dict)
    
    # Current activity
    current_activity: str
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Results (when available)
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    quality_preview: Optional[Dict[str, float]] = None
    safety_preview: Optional[Dict[str, float]] = None
    
    # Logs and debugging
    recent_logs: List[str] = Field(default_factory=list)
    debug_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Next steps
    next_stage: Optional[str] = None
    user_action_required: bool = Field(default=False)
    action_required_message: Optional[str] = None