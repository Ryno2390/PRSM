"""
PRSM Expanded Marketplace Models
================================

Comprehensive data models for PRSM's expanded marketplace ecosystem covering:
1. AI Models (existing)
2. MCP Tools (existing) 
3. Curated Datasets
4. Agentic Functions/Workflows
5. Computational Infrastructure
6. Knowledge Graphs & Ontologies
7. Evaluation & Benchmarking Services
8. AI Training & Optimization Services
9. AI Safety & Governance Tools

This unified marketplace enables a complete AI development ecosystem with
FTNS token integration, quality assurance, and decentralized governance.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Integer, Decimal as SQLDecimal, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from prsm.core.models import BaseDBModel

Base = declarative_base()


# ============================================================================
# UNIFIED MARKETPLACE ENUMS
# ============================================================================

class ResourceType(str, Enum):
    """All marketplace resource types"""
    AI_MODEL = "ai_model"
    MCP_TOOL = "mcp_tool"
    DATASET = "dataset"
    AGENT_WORKFLOW = "agent_workflow"
    COMPUTE_RESOURCE = "compute_resource"
    KNOWLEDGE_RESOURCE = "knowledge_resource"
    EVALUATION_SERVICE = "evaluation_service"
    TRAINING_SERVICE = "training_service"
    SAFETY_TOOL = "safety_tool"


class ResourceStatus(str, Enum):
    """Universal resource status"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"
    MAINTENANCE = "maintenance"


class PricingModel(str, Enum):
    """Universal pricing models"""
    FREE = "free"
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    FREEMIUM = "freemium"
    ENTERPRISE = "enterprise"
    AUCTION = "auction"
    REVENUE_SHARE = "revenue_share"


class QualityGrade(str, Enum):
    """Universal quality grades"""
    EXPERIMENTAL = "experimental"
    COMMUNITY = "community"
    VERIFIED = "verified"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


# ============================================================================
# CURATED DATASETS MARKETPLACE
# ============================================================================

class DatasetCategory(str, Enum):
    """Dataset categories"""
    TRAINING_DATA = "training_data"
    FINE_TUNING = "fine_tuning"
    EVALUATION_BENCHMARKS = "evaluation_benchmarks"
    SCIENTIFIC_RESEARCH = "scientific_research"
    MULTIMODAL = "multimodal"
    MEDICAL_RESEARCH = "medical_research"
    LEGAL_DOCUMENTS = "legal_documents"
    FINANCIAL_DATA = "financial_data"
    SYNTHETIC_DATA = "synthetic_data"
    TIME_SERIES = "time_series"
    GRAPH_DATA = "graph_data"
    DOMAIN_SPECIFIC = "domain_specific"


class DataFormat(str, Enum):
    """Supported data formats"""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    ARROW = "arrow"
    PICKLE = "pickle"
    NUMPY = "numpy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class DatasetLicense(str, Enum):
    """Dataset licensing options"""
    MIT = "mit"
    APACHE_2 = "apache_2"
    GPL_3 = "gpl_3"
    BSD_3_CLAUSE = "bsd_3_clause"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    CC_BY_NC = "cc_by_nc"
    PROPRIETARY = "proprietary"
    CUSTOM = "custom"


class DatasetListing(BaseModel):
    """Dataset marketplace listing"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    category: DatasetCategory
    
    # Dataset specifications
    size_bytes: int = Field(..., ge=0)
    record_count: int = Field(..., ge=0)
    feature_count: Optional[int] = Field(None, ge=0)
    data_format: DataFormat
    schema_definition: Optional[Dict[str, Any]] = None
    
    # Data quality metrics
    completeness_score: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    accuracy_score: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    consistency_score: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    freshness_date: Optional[datetime] = None
    
    # Licensing and compliance
    license_type: DatasetLicense
    ethical_review_status: str = Field(default="pending")
    privacy_compliance: List[str] = Field(default_factory=list)  # GDPR, HIPAA, etc.
    data_lineage: Optional[Dict[str, Any]] = None
    
    # Pricing
    pricing_model: PricingModel = PricingModel.FREE
    base_price: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_record: Decimal = Field(default=Decimal('0'), ge=0)
    subscription_price: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Marketplace metadata
    owner_user_id: UUID
    provider_name: Optional[str] = None
    quality_grade: QualityGrade = QualityGrade.EXPERIMENTAL
    download_count: int = Field(default=0, ge=0)
    rating_average: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    
    # Access and distribution
    access_url: Optional[str] = None
    sample_data_url: Optional[str] = None
    documentation_url: Optional[str] = None
    preprocessing_scripts: Optional[List[str]] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# AGENTIC FUNCTIONS/WORKFLOWS MARKETPLACE
# ============================================================================

class AgentType(str, Enum):
    """Types of AI agents"""
    RESEARCH_AGENT = "research_agent"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    SCIENTIFIC_SIMULATION = "scientific_simulation"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    DECISION_SUPPORT = "decision_support"
    CONTENT_CREATION = "content_creation"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"


class AgentCapability(str, Enum):
    """Agent capabilities"""
    MULTI_STEP_REASONING = "multi_step_reasoning"
    TOOL_USAGE = "tool_usage"
    MEMORY_MANAGEMENT = "memory_management"
    LEARNING_ADAPTATION = "learning_adaptation"
    PARALLEL_PROCESSING = "parallel_processing"
    HUMAN_INTERACTION = "human_interaction"
    API_INTEGRATION = "api_integration"
    FILE_PROCESSING = "file_processing"
    WEB_SCRAPING = "web_scraping"
    DATABASE_OPERATIONS = "database_operations"


class AgentWorkflowListing(BaseModel):
    """Agent workflow marketplace listing"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    agent_type: AgentType
    
    # Capabilities and specifications
    capabilities: List[AgentCapability] = Field(default_factory=list)
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    max_execution_time: Optional[int] = Field(None, ge=1)  # seconds
    memory_requirements: Optional[int] = Field(None, ge=0)  # MB
    
    # Workflow definition
    workflow_config: Dict[str, Any] = Field(default_factory=dict)
    required_tools: List[str] = Field(default_factory=list)
    required_models: List[str] = Field(default_factory=list)
    environment_requirements: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    average_execution_time: Optional[Decimal] = Field(None, ge=0)
    success_rate: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    accuracy_score: Optional[Decimal] = Field(None, ge=0, le=1)
    
    # Pricing
    pricing_model: PricingModel = PricingModel.PAY_PER_USE
    base_price: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_execution: Decimal = Field(default=Decimal('0'), ge=0)
    subscription_price: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Marketplace metadata
    owner_user_id: UUID
    provider_name: Optional[str] = None
    quality_grade: QualityGrade = QualityGrade.EXPERIMENTAL
    execution_count: int = Field(default=0, ge=0)
    rating_average: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    
    # Distribution
    deployment_url: Optional[str] = None
    source_code_url: Optional[str] = None
    documentation_url: Optional[str] = None
    example_usage: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# COMPUTATIONAL INFRASTRUCTURE MARKETPLACE
# ============================================================================

class ComputeResourceType(str, Enum):
    """Types of compute resources"""
    GPU_CLUSTER = "gpu_cluster"
    CPU_CLUSTER = "cpu_cluster"
    TPU_ACCESS = "tpu_access"
    FPGA_ARRAY = "fpga_array"
    QUANTUM_SIMULATOR = "quantum_simulator"
    EDGE_COMPUTING = "edge_computing"
    DISTRIBUTED_STORAGE = "distributed_storage"
    SPECIALIZED_HARDWARE = "specialized_hardware"
    CLOUD_FUNCTIONS = "cloud_functions"
    CONTAINER_ORCHESTRATION = "container_orchestration"


class ComputeCapability(str, Enum):
    """Compute capabilities"""
    HIGH_MEMORY = "high_memory"
    HIGH_BANDWIDTH = "high_bandwidth"
    LOW_LATENCY = "low_latency"
    FAULT_TOLERANT = "fault_tolerant"
    AUTO_SCALING = "auto_scaling"
    GPU_ACCELERATION = "gpu_acceleration"
    PARALLEL_PROCESSING = "parallel_processing"
    REAL_TIME_PROCESSING = "real_time_processing"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"


class ComputeResourceListing(BaseModel):
    """Compute resource marketplace listing"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    resource_type: ComputeResourceType
    
    # Hardware specifications
    cpu_cores: Optional[int] = Field(None, ge=1)
    memory_gb: Optional[int] = Field(None, ge=1)
    storage_gb: Optional[int] = Field(None, ge=1)
    gpu_count: Optional[int] = Field(None, ge=0)
    gpu_model: Optional[str] = None
    network_bandwidth_gbps: Optional[Decimal] = Field(None, ge=0)
    
    # Capabilities and features
    capabilities: List[ComputeCapability] = Field(default_factory=list)
    supported_frameworks: List[str] = Field(default_factory=list)
    operating_systems: List[str] = Field(default_factory=list)
    geographic_regions: List[str] = Field(default_factory=list)
    
    # Performance metrics
    uptime_percentage: Decimal = Field(default=Decimal('99.0'), ge=0, le=100)
    average_latency_ms: Optional[Decimal] = Field(None, ge=0)
    throughput_ops_per_sec: Optional[int] = Field(None, ge=0)
    
    # Availability and scheduling
    availability_schedule: Optional[Dict[str, Any]] = None
    min_rental_duration: Optional[int] = Field(None, ge=1)  # minutes
    max_rental_duration: Optional[int] = Field(None, ge=1)  # minutes
    auto_scaling_enabled: bool = Field(default=False)
    
    # Pricing
    pricing_model: PricingModel = PricingModel.PAY_PER_USE
    price_per_hour: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_compute_unit: Decimal = Field(default=Decimal('0'), ge=0)
    setup_fee: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Marketplace metadata
    owner_user_id: UUID
    provider_name: Optional[str] = None
    quality_grade: QualityGrade = QualityGrade.COMMUNITY
    usage_hours: Decimal = Field(default=Decimal('0'), ge=0)
    rating_average: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    
    # Access and configuration
    access_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    configuration_template: Optional[Dict[str, Any]] = None
    security_features: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# KNOWLEDGE GRAPHS & ONTOLOGIES MARKETPLACE
# ============================================================================

class KnowledgeResourceType(str, Enum):
    """Types of knowledge resources"""
    DOMAIN_ONTOLOGY = "domain_ontology"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SEMANTIC_EMBEDDINGS = "semantic_embeddings"
    EXPERT_KNOWLEDGE_BASE = "expert_knowledge_base"
    CONCEPT_HIERARCHY = "concept_hierarchy"
    RELATION_SCHEMA = "relation_schema"
    REASONING_RULES = "reasoning_rules"
    FACT_DATABASE = "fact_database"


class KnowledgeDomain(str, Enum):
    """Knowledge domains"""
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    GENERAL = "general"
    ACADEMIC = "academic"
    INDUSTRIAL = "industrial"
    GOVERNMENT = "government"
    CULTURAL = "cultural"


class KnowledgeResourceListing(BaseModel):
    """Knowledge resource marketplace listing"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    resource_type: KnowledgeResourceType
    
    # Content specifications
    domain: KnowledgeDomain
    entity_count: Optional[int] = Field(None, ge=0)
    relation_count: Optional[int] = Field(None, ge=0)
    fact_count: Optional[int] = Field(None, ge=0)
    coverage_scope: Optional[str] = None
    
    # Quality metrics
    completeness_score: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    accuracy_score: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    consistency_score: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    expert_validation: bool = Field(default=False)
    
    # Technical specifications
    format_type: str = Field(default="rdf")  # RDF, OWL, JSON-LD, etc.
    query_languages: List[str] = Field(default_factory=list)  # SPARQL, Cypher, etc.
    reasoning_capabilities: List[str] = Field(default_factory=list)
    update_frequency: Optional[str] = None
    
    # Pricing
    pricing_model: PricingModel = PricingModel.SUBSCRIPTION
    base_price: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_query: Decimal = Field(default=Decimal('0'), ge=0)
    subscription_price: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Marketplace metadata
    owner_user_id: UUID
    provider_name: Optional[str] = None
    quality_grade: QualityGrade = QualityGrade.COMMUNITY
    query_count: int = Field(default=0, ge=0)
    rating_average: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    
    # Access and integration
    access_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    sparql_endpoint: Optional[str] = None
    documentation_url: Optional[str] = None
    integration_examples: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# EVALUATION & BENCHMARKING SERVICES MARKETPLACE
# ============================================================================

class EvaluationServiceType(str, Enum):
    """Types of evaluation services"""
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    SAFETY_TESTING = "safety_testing"
    BIAS_AUDITING = "bias_auditing"
    ROBUSTNESS_TESTING = "robustness_testing"
    DOMAIN_VALIDATION = "domain_validation"
    SECURITY_ASSESSMENT = "security_assessment"
    COMPLIANCE_CHECKING = "compliance_checking"
    INTERPRETABILITY_ANALYSIS = "interpretability_analysis"


class EvaluationMetric(str, Enum):
    """Evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    FAIRNESS = "fairness"
    ROBUSTNESS = "robustness"
    SAFETY = "safety"


class EvaluationServiceListing(BaseModel):
    """Evaluation service marketplace listing"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    service_type: EvaluationServiceType
    
    # Service specifications
    supported_models: List[str] = Field(default_factory=list)
    evaluation_metrics: List[EvaluationMetric] = Field(default_factory=list)
    test_datasets: List[str] = Field(default_factory=list)
    evaluation_protocols: List[str] = Field(default_factory=list)
    
    # Quality and reliability
    benchmark_validity: bool = Field(default=False)
    peer_reviewed: bool = Field(default=False)
    reproducibility_score: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    validation_count: int = Field(default=0, ge=0)
    
    # Performance characteristics
    average_evaluation_time: Optional[int] = Field(None, ge=1)  # minutes
    max_model_size: Optional[str] = None
    supported_frameworks: List[str] = Field(default_factory=list)
    
    # Pricing
    pricing_model: PricingModel = PricingModel.PAY_PER_USE
    base_price: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_evaluation: Decimal = Field(default=Decimal('0'), ge=0)
    subscription_price: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Marketplace metadata
    owner_user_id: UUID
    provider_name: Optional[str] = None
    quality_grade: QualityGrade = QualityGrade.VERIFIED
    evaluation_count: int = Field(default=0, ge=0)
    rating_average: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    
    # Service access
    service_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    documentation_url: Optional[str] = None
    example_reports: Optional[List[str]] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# AI TRAINING & OPTIMIZATION SERVICES MARKETPLACE
# ============================================================================

class TrainingServiceType(str, Enum):
    """Types of training services"""
    CUSTOM_FINE_TUNING = "custom_fine_tuning"
    DISTILLATION_SERVICE = "distillation_service"
    PROMPT_ENGINEERING = "prompt_engineering"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    MODEL_COMPRESSION = "model_compression"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"


class TrainingFramework(str, Enum):
    """Supported training frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    HUGGINGFACE = "huggingface"
    TRANSFORMERS = "transformers"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    CUSTOM = "custom"


class TrainingServiceListing(BaseModel):
    """Training service marketplace listing"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    service_type: TrainingServiceType
    
    # Service capabilities
    supported_frameworks: List[TrainingFramework] = Field(default_factory=list)
    supported_architectures: List[str] = Field(default_factory=list)
    max_model_parameters: Optional[int] = Field(None, ge=1)
    supported_data_types: List[str] = Field(default_factory=list)
    
    # Training specifications
    max_training_time: Optional[int] = Field(None, ge=1)  # hours
    available_compute: Optional[str] = None
    distributed_training: bool = Field(default=False)
    automated_tuning: bool = Field(default=False)
    
    # Quality metrics
    success_rate: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    average_improvement: Optional[Decimal] = Field(None, ge=0)
    client_satisfaction: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    
    # Pricing
    pricing_model: PricingModel = PricingModel.PAY_PER_USE
    base_price: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_hour: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_parameter: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Marketplace metadata
    owner_user_id: UUID
    provider_name: Optional[str] = None
    quality_grade: QualityGrade = QualityGrade.PREMIUM
    training_jobs_completed: int = Field(default=0, ge=0)
    rating_average: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    
    # Service access
    service_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    documentation_url: Optional[str] = None
    portfolio_examples: Optional[List[str]] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# AI SAFETY & GOVERNANCE TOOLS MARKETPLACE
# ============================================================================

class SafetyToolType(str, Enum):
    """Types of AI safety tools"""
    ALIGNMENT_VALIDATOR = "alignment_validator"
    INTERPRETABILITY_TOOL = "interpretability_tool"
    COMPLIANCE_CHECKER = "compliance_checker"
    ETHICAL_GUIDELINE_VALIDATOR = "ethical_guideline_validator"
    BIAS_DETECTOR = "bias_detector"
    FAIRNESS_AUDITOR = "fairness_auditor"
    TRANSPARENCY_TOOL = "transparency_tool"
    ACCOUNTABILITY_TRACKER = "accountability_tracker"


class ComplianceStandard(str, Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    EU_AI_ACT = "eu_ai_act"
    IEEE_STANDARDS = "ieee_standards"
    CUSTOM = "custom"


class SafetyToolListing(BaseModel):
    """AI safety tool marketplace listing"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    tool_type: SafetyToolType
    
    # Tool capabilities
    supported_models: List[str] = Field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = Field(default_factory=list)
    detection_capabilities: List[str] = Field(default_factory=list)
    reporting_formats: List[str] = Field(default_factory=list)
    
    # Validation and certification
    third_party_validated: bool = Field(default=False)
    certification_bodies: List[str] = Field(default_factory=list)
    audit_trail_support: bool = Field(default=False)
    
    # Performance metrics
    detection_accuracy: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    false_positive_rate: Decimal = Field(default=Decimal('0'), ge=0, le=1)
    processing_speed: Optional[str] = None
    
    # Pricing
    pricing_model: PricingModel = PricingModel.SUBSCRIPTION
    base_price: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_scan: Decimal = Field(default=Decimal('0'), ge=0)
    enterprise_price: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Marketplace metadata
    owner_user_id: UUID
    provider_name: Optional[str] = None
    quality_grade: QualityGrade = QualityGrade.ENTERPRISE
    scans_performed: int = Field(default=0, ge=0)
    rating_average: Decimal = Field(default=Decimal('0'), ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    
    # Tool access
    tool_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    documentation_url: Optional[str] = None
    compliance_reports: Optional[List[str]] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# UNIFIED MARKETPLACE DATABASE MODELS
# ============================================================================

class UnifiedResourceListingDB(BaseDBModel):
    """Unified database model for all marketplace resources"""
    __tablename__ = "marketplace_unified_listings"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Universal fields
    resource_type = Column(String(50), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    owner_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    
    # Status and quality
    status = Column(String(50), nullable=False, default="pending_review", index=True)
    quality_grade = Column(String(50), nullable=False, default="experimental", index=True)
    featured = Column(Boolean, default=False, index=True)
    verified = Column(Boolean, default=False, index=True)
    
    # Pricing
    pricing_model = Column(String(50), nullable=False, default="free")
    base_price = Column(SQLDecimal(20, 8), default=0)
    subscription_price = Column(SQLDecimal(20, 8), default=0)
    
    # Marketplace metrics
    usage_count = Column(Integer, default=0)
    rating_average = Column(SQLDecimal(3, 2), default=0)
    rating_count = Column(Integer, default=0)
    revenue_total = Column(SQLDecimal(20, 8), default=0)
    
    # Metadata and configuration
    resource_metadata = Column(JSON)  # Resource-specific data
    technical_specs = Column(JSON)    # Technical specifications
    access_config = Column(JSON)      # Access and integration config
    tags = Column(JSON)               # Searchable tags
    
    # Provider information
    provider_name = Column(String(255))
    provider_url = Column(String(512))
    documentation_url = Column(String(512))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    last_used_at = Column(DateTime(timezone=True))
    
    # Relationships
    reviews = relationship("ResourceReviewDB", back_populates="resource")
    orders = relationship("ResourceOrderDB", back_populates="resource")


class ResourceReviewDB(BaseDBModel):
    """Reviews for marketplace resources"""
    __tablename__ = "marketplace_resource_reviews"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_unified_listings.id"), nullable=False, index=True)
    reviewer_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    
    # Review content
    rating = Column(Integer, nullable=False)  # 1-5 stars
    title = Column(String(255))
    content = Column(Text)
    
    # Review metadata
    verified_purchase = Column(Boolean, default=False)
    helpful_votes = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    resource = relationship("UnifiedResourceListingDB", back_populates="reviews")


class ResourceOrderDB(BaseDBModel):
    """Orders for marketplace resources"""
    __tablename__ = "marketplace_resource_orders"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_number = Column(String(50), nullable=False, unique=True, index=True)
    
    # Order details
    buyer_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    resource_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_unified_listings.id"), nullable=False, index=True)
    
    # Pricing
    quantity = Column(Integer, default=1)
    unit_price = Column(SQLDecimal(20, 8), nullable=False)
    total_amount = Column(SQLDecimal(20, 8), nullable=False)
    platform_fee = Column(SQLDecimal(20, 8), default=0)
    
    # Status
    order_status = Column(String(50), nullable=False, default="pending", index=True)
    payment_status = Column(String(50), nullable=False, default="pending")
    
    # Metadata
    order_metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    resource = relationship("UnifiedResourceListingDB", back_populates="orders")


# ============================================================================
# MARKETPLACE SEARCH AND DISCOVERY MODELS
# ============================================================================

class UnifiedSearchFilters(BaseModel):
    """Unified search filters for all marketplace resources"""
    resource_types: Optional[List[ResourceType]] = None
    pricing_models: Optional[List[PricingModel]] = None
    quality_grades: Optional[List[QualityGrade]] = None
    
    # Price filtering
    min_price: Optional[Decimal] = Field(None, ge=0)
    max_price: Optional[Decimal] = Field(None, ge=0)
    
    # Quality filtering
    min_rating: Optional[Decimal] = Field(None, ge=0, le=5)
    verified_only: bool = False
    featured_only: bool = False
    
    # Content filtering
    tags: Optional[List[str]] = None
    search_query: Optional[str] = Field(None, max_length=255)
    provider_name: Optional[str] = None
    
    # Sorting
    sort_by: str = Field(default="popularity", pattern=r'^(popularity|price|created_at|rating|name|usage_count)$')
    sort_order: str = Field(default="desc", pattern=r'^(asc|desc)$')
    
    # Pagination
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class MarketplaceSearchResponse(BaseModel):
    """Response model for marketplace search"""
    resources: List[Dict[str, Any]]
    total_count: int
    has_more: bool
    facets: Dict[str, Any]
    search_metadata: Dict[str, Any]


class MarketplaceStatsResponse(BaseModel):
    """Comprehensive marketplace statistics"""
    total_resources: int
    resources_by_type: Dict[str, int]
    total_providers: int
    total_revenue: Decimal
    total_transactions: int
    active_users: int
    
    # Quality metrics
    average_rating: Decimal
    verification_rate: Decimal
    
    # Growth metrics
    new_resources_this_month: int
    revenue_growth_rate: Decimal
    
    # Top performers
    top_resources: List[Dict[str, Any]]
    top_providers: List[Dict[str, Any]]
    trending_categories: List[str]