"""
PRSM Marketplace Database Models
===============================

Complete SQLAlchemy database models for the comprehensive PRSM marketplace.
Supports 9 asset types: AI Models, Datasets, Agents, Tools, Infrastructure,
Knowledge Resources, Evaluation Services, Training Services, and Safety Tools.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Column, String, Integer, BigInteger, Text, Boolean, DateTime, 
    DECIMAL, JSON, ForeignKey, Table, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

Base = declarative_base()


# ============================================================================
# ASSOCIATION TABLES FOR MANY-TO-MANY RELATIONSHIPS
# ============================================================================

# Tags association table
marketplace_resource_tags = Table(
    'marketplace_resource_tags',
    Base.metadata,
    Column('resource_id', PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True),
    Column('tag_id', PGUUID(as_uuid=True), ForeignKey('marketplace_tags.id'), primary_key=True)
)

# Reviews association table
marketplace_resource_reviews = Table(
    'marketplace_resource_reviews',
    Base.metadata,
    Column('resource_id', PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True),
    Column('review_id', PGUUID(as_uuid=True), ForeignKey('marketplace_reviews.id'), primary_key=True)
)


# ============================================================================
# CORE MARKETPLACE TABLES
# ============================================================================

class MarketplaceResource(Base):
    """Base table for all marketplace resources (polymorphic)"""
    __tablename__ = 'marketplace_resources'
    
    # Primary key and type discrimination
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_type = Column(String(50), nullable=False, index=True)  # Discriminator
    
    # Basic information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    short_description = Column(String(500))
    
    # Ownership and provider
    owner_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    provider_name = Column(String(255))
    provider_verified = Column(Boolean, default=False)
    
    # Status and quality
    status = Column(String(50), nullable=False, default='draft', index=True)
    quality_grade = Column(String(50), nullable=False, default='experimental')
    
    # Pricing
    pricing_model = Column(String(50), nullable=False, default='free')
    base_price = Column(DECIMAL(10, 2), default=0)
    subscription_price = Column(DECIMAL(10, 2), default=0)
    enterprise_price = Column(DECIMAL(10, 2), default=0)
    
    # Usage and ratings
    download_count = Column(Integer, default=0)
    usage_count = Column(Integer, default=0)
    rating_average = Column(DECIMAL(3, 2), default=0)
    rating_count = Column(Integer, default=0)
    
    # Metadata
    version = Column(String(50), default='1.0.0')
    documentation_url = Column(String(1000))
    source_url = Column(String(1000))
    license_type = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    published_at = Column(DateTime(timezone=True))
    
    # Polymorphic configuration
    __mapper_args__ = {
        'polymorphic_identity': 'base',
        'polymorphic_on': resource_type
    }
    
    # Relationships
    tags = relationship("MarketplaceTag", secondary=marketplace_resource_tags, back_populates="resources")
    reviews = relationship("MarketplaceReview", secondary=marketplace_resource_reviews, back_populates="resources")
    orders = relationship("MarketplaceOrder", back_populates="resource")
    
    # Indexes
    __table_args__ = (
        Index('idx_marketplace_resources_type_status', 'resource_type', 'status'),
        Index('idx_marketplace_resources_owner_type', 'owner_user_id', 'resource_type'),
        Index('idx_marketplace_resources_rating', 'rating_average'),
        Index('idx_marketplace_resources_created', 'created_at'),
    )


class AIModelListing(MarketplaceResource):
    """AI Model marketplace listings"""
    __tablename__ = 'ai_model_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Model specifications
    model_category = Column(String(50), nullable=False, index=True)
    model_provider = Column(String(100))
    model_architecture = Column(String(100))
    parameter_count = Column(String(50))  # e.g., "7B", "70B", "175B"
    context_length = Column(Integer)
    
    # Capabilities
    capabilities = Column(JSON)  # List of capabilities
    languages_supported = Column(JSON)  # List of supported languages
    modalities = Column(JSON)  # List of supported modalities (text, image, audio, etc.)
    
    # Performance metrics
    benchmark_scores = Column(JSON)  # Dict of benchmark -> score
    latency_ms = Column(Integer)
    throughput_tokens_per_second = Column(Integer)
    memory_requirements_gb = Column(Integer)
    
    # Fine-tuning information (for fine-tuned models)
    is_fine_tuned = Column(Boolean, default=False)
    base_model_id = Column(PGUUID(as_uuid=True), ForeignKey('ai_model_listings.id'))
    fine_tuning_dataset = Column(String(500))
    fine_tuning_task = Column(String(200))
    
    # API access
    api_endpoint = Column(String(1000))
    api_key_required = Column(Boolean, default=True)
    rate_limits = Column(JSON)  # Rate limiting configuration
    
    __mapper_args__ = {
        'polymorphic_identity': 'ai_model'
    }
    
    # Self-referential relationship for base model
    fine_tuned_variants = relationship("AIModelListing", backref=backref('base_model', remote_side=[id]))


class DatasetListing(MarketplaceResource):
    """Dataset marketplace listings"""
    __tablename__ = 'dataset_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Dataset specifications
    dataset_category = Column(String(50), nullable=False, index=True)
    data_format = Column(String(50), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    record_count = Column(Integer, nullable=False)
    feature_count = Column(Integer)
    
    # Schema and structure
    schema_definition = Column(JSON)  # Data schema
    sample_data = Column(JSON)  # Sample records
    column_descriptions = Column(JSON)  # Column metadata
    
    # Data quality metrics
    completeness_score = Column(DECIMAL(3, 2), default=0)
    accuracy_score = Column(DECIMAL(3, 2), default=0)
    consistency_score = Column(DECIMAL(3, 2), default=0)
    freshness_date = Column(DateTime(timezone=True))
    
    # Compliance and ethics
    ethical_review_status = Column(String(50), default='pending')
    privacy_compliance = Column(JSON)  # List of compliance standards
    data_lineage = Column(JSON)  # Data provenance information
    bias_assessment = Column(JSON)  # Bias analysis results
    
    # Access and distribution
    access_url = Column(String(1000))
    sample_data_url = Column(String(1000))
    preprocessing_scripts = Column(JSON)  # List of script URLs
    data_loader_code = Column(Text)  # Code to load the dataset
    
    __mapper_args__ = {
        'polymorphic_identity': 'dataset'
    }


class AgentWorkflowListing(MarketplaceResource):
    """AI Agent and Workflow marketplace listings"""
    __tablename__ = 'agent_workflow_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Agent specifications
    agent_type = Column(String(50), nullable=False, index=True)
    agent_capabilities = Column(JSON)  # List of capabilities
    
    # Technical requirements
    required_models = Column(JSON)  # List of model requirements
    required_tools = Column(JSON)  # List of MCP tools required
    environment_requirements = Column(JSON)  # System requirements
    
    # Configuration
    default_configuration = Column(JSON)  # Default agent config
    customization_options = Column(JSON)  # Available customization
    workflow_definition = Column(JSON)  # Workflow steps/DAG
    
    # Performance metrics
    success_rate = Column(DECIMAL(3, 2), default=0)
    average_execution_time = Column(Integer)  # in seconds
    resource_usage = Column(JSON)  # CPU, memory, etc.
    
    # Integration
    api_endpoints = Column(JSON)  # Available API endpoints
    webhook_support = Column(Boolean, default=False)
    integration_examples = Column(JSON)  # Code examples
    
    __mapper_args__ = {
        'polymorphic_identity': 'agent_workflow'
    }


class MCPToolListing(MarketplaceResource):
    """MCP Tool marketplace listings"""
    __tablename__ = 'mcp_tool_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Tool specifications
    tool_category = Column(String(50), nullable=False, index=True)
    protocol_version = Column(String(20), default='1.0')
    
    # Functionality
    functions_provided = Column(JSON)  # List of functions
    input_schema = Column(JSON)  # JSON schema for inputs
    output_schema = Column(JSON)  # JSON schema for outputs
    
    # Integration
    installation_method = Column(String(50))  # pip, npm, docker, etc.
    package_name = Column(String(200))
    container_image = Column(String(500))
    configuration_schema = Column(JSON)
    
    # Security
    security_requirements = Column(JSON)  # Security considerations
    sandboxing_enabled = Column(Boolean, default=False)
    permission_requirements = Column(JSON)  # Required permissions
    
    # Compatibility
    compatible_models = Column(JSON)  # List of compatible models
    platform_support = Column(JSON)  # Supported platforms
    dependencies = Column(JSON)  # Required dependencies
    
    __mapper_args__ = {
        'polymorphic_identity': 'mcp_tool'
    }


class ComputeResourceListing(MarketplaceResource):
    """Computational infrastructure marketplace listings"""
    __tablename__ = 'compute_resource_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Hardware specifications
    resource_type = Column(String(50), nullable=False)  # gpu_cluster, cpu_cluster, tpu, etc.
    cpu_cores = Column(Integer)
    memory_gb = Column(Integer)
    storage_gb = Column(Integer)
    gpu_count = Column(Integer)
    gpu_type = Column(String(100))
    
    # Availability and location
    geographic_regions = Column(JSON)  # Available regions
    availability_schedule = Column(JSON)  # Time slots
    uptime_percentage = Column(DECIMAL(5, 2), default=99.9)
    
    # Performance metrics
    benchmark_scores = Column(JSON)  # Performance benchmarks
    network_bandwidth_gbps = Column(Integer)
    latency_ms = Column(Integer)
    
    # Pricing
    price_per_hour = Column(DECIMAL(10, 4))
    price_per_compute_unit = Column(DECIMAL(10, 4))
    minimum_rental_hours = Column(Integer, default=1)
    
    # Software environment
    available_frameworks = Column(JSON)  # ML frameworks available
    container_support = Column(Boolean, default=True)
    custom_images_allowed = Column(Boolean, default=False)
    
    __mapper_args__ = {
        'polymorphic_identity': 'compute_resource'
    }


class KnowledgeResourceListing(MarketplaceResource):
    """Knowledge resources marketplace listings"""
    __tablename__ = 'knowledge_resource_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Knowledge type
    knowledge_type = Column(String(50), nullable=False, index=True)  # ontology, knowledge_graph, etc.
    domain_specialization = Column(String(100))  # medical, legal, financial, etc.
    
    # Content specifications
    entity_count = Column(Integer)
    relationship_count = Column(Integer)
    concept_hierarchy_depth = Column(Integer)
    
    # Access methods
    query_languages = Column(JSON)  # SPARQL, GraphQL, etc.
    api_endpoints = Column(JSON)  # Available endpoints
    export_formats = Column(JSON)  # Available export formats
    
    # Quality metrics
    expert_validation_score = Column(DECIMAL(3, 2), default=0)
    coverage_completeness = Column(DECIMAL(3, 2), default=0)
    update_frequency = Column(String(50))  # daily, weekly, monthly
    
    # Integration
    embedding_models_supported = Column(JSON)  # Compatible embedding models
    rag_integration_examples = Column(JSON)  # RAG integration examples
    
    __mapper_args__ = {
        'polymorphic_identity': 'knowledge_resource'
    }


class EvaluationServiceListing(MarketplaceResource):
    """Evaluation and benchmarking service listings"""
    __tablename__ = 'evaluation_service_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Service type
    evaluation_type = Column(String(50), nullable=False, index=True)  # benchmark, safety_test, etc.
    
    # Evaluation capabilities
    supported_model_types = Column(JSON)  # Types of models that can be evaluated
    evaluation_metrics = Column(JSON)  # Available metrics
    benchmark_datasets = Column(JSON)  # Benchmark datasets used
    
    # Process
    evaluation_methodology = Column(Text)  # Description of methodology
    reproducibility_score = Column(DECIMAL(3, 2), default=0)
    peer_review_status = Column(String(50), default='pending')
    
    # Results
    typical_evaluation_time = Column(Integer)  # in minutes
    result_formats = Column(JSON)  # Available result formats
    comparison_baselines = Column(JSON)  # Available baselines
    
    # Compliance
    standard_compliance = Column(JSON)  # Industry standards compliance
    certification_authority = Column(String(200))
    
    __mapper_args__ = {
        'polymorphic_identity': 'evaluation_service'
    }


class TrainingServiceListing(MarketplaceResource):
    """AI training and optimization service listings"""
    __tablename__ = 'training_service_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Service type
    training_type = Column(String(50), nullable=False, index=True)  # fine_tuning, distillation, etc.
    
    # Capabilities
    supported_frameworks = Column(JSON)  # PyTorch, TensorFlow, JAX, etc.
    supported_model_architectures = Column(JSON)  # Transformer, CNN, etc.
    optimization_techniques = Column(JSON)  # LoRA, QLoRA, etc.
    
    # Resources
    available_compute = Column(JSON)  # Available compute resources
    maximum_model_size = Column(String(50))  # e.g., "70B parameters"
    distributed_training = Column(Boolean, default=False)
    
    # Process
    automated_hyperparameter_tuning = Column(Boolean, default=False)
    early_stopping = Column(Boolean, default=True)
    checkpointing_enabled = Column(Boolean, default=True)
    
    # Results
    typical_training_time = Column(Integer)  # in hours
    model_compression_ratio = Column(DECIMAL(3, 2))  # for distillation services
    performance_retention = Column(DECIMAL(3, 2))  # performance retained after optimization
    
    __mapper_args__ = {
        'polymorphic_identity': 'training_service'
    }


class SafetyToolListing(MarketplaceResource):
    """AI safety and governance tool listings"""
    __tablename__ = 'safety_tool_listings'
    
    id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), primary_key=True)
    
    # Tool type
    safety_category = Column(String(50), nullable=False, index=True)  # alignment, interpretability, etc.
    
    # Capabilities
    supported_model_types = Column(JSON)  # Model types supported
    detection_capabilities = Column(JSON)  # What the tool can detect
    prevention_mechanisms = Column(JSON)  # Prevention capabilities
    
    # Compliance
    regulatory_compliance = Column(JSON)  # GDPR, HIPAA, EU AI Act, etc.
    certification_status = Column(String(50))
    third_party_validated = Column(Boolean, default=False)
    
    # Integration
    real_time_monitoring = Column(Boolean, default=False)
    batch_processing = Column(Boolean, default=True)
    api_integration = Column(Boolean, default=True)
    
    # Metrics
    false_positive_rate = Column(DECIMAL(5, 4))
    false_negative_rate = Column(DECIMAL(5, 4))
    processing_latency_ms = Column(Integer)
    
    __mapper_args__ = {
        'polymorphic_identity': 'safety_tool'
    }


# ============================================================================
# SUPPORTING TABLES
# ============================================================================

class MarketplaceTag(Base):
    """Tags for marketplace resources"""
    __tablename__ = 'marketplace_tags'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True, index=True)
    category = Column(String(50))  # technical, domain, quality, etc.
    usage_count = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    resources = relationship("MarketplaceResource", secondary=marketplace_resource_tags, back_populates="tags")


class MarketplaceReview(Base):
    """Reviews and ratings for marketplace resources"""
    __tablename__ = 'marketplace_reviews'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    
    # Review content
    rating = Column(Integer, nullable=False)  # 1-5 stars
    title = Column(String(200))
    content = Column(Text)
    
    # Review metadata
    verified_purchase = Column(Boolean, default=False)
    helpful_count = Column(Integer, default=0)
    usage_duration_days = Column(Integer)  # How long they used it
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    resources = relationship("MarketplaceResource", secondary=marketplace_resource_reviews, back_populates="reviews")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'id', name='unique_user_review_per_resource'),
        Index('idx_marketplace_reviews_rating', 'rating'),
        Index('idx_marketplace_reviews_created', 'created_at'),
    )


class MarketplaceOrder(Base):
    """Orders and transactions in the marketplace"""
    __tablename__ = 'marketplace_orders'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), nullable=False, index=True)
    buyer_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    
    # Order details
    order_type = Column(String(50), nullable=False)  # purchase, subscription, rental
    quantity = Column(Integer, default=1)
    unit_price = Column(DECIMAL(10, 2), nullable=False)
    total_price = Column(DECIMAL(10, 2), nullable=False)
    currency = Column(String(10), default='FTNS')
    
    # Order status
    status = Column(String(50), nullable=False, default='pending', index=True)
    payment_status = Column(String(50), nullable=False, default='pending')
    fulfillment_status = Column(String(50), nullable=False, default='pending')
    
    # Subscription details (for subscription orders)
    subscription_start_date = Column(DateTime(timezone=True))
    subscription_end_date = Column(DateTime(timezone=True))
    auto_renewal = Column(Boolean, default=False)
    
    # Rental details (for compute resources)
    rental_start_time = Column(DateTime(timezone=True))
    rental_end_time = Column(DateTime(timezone=True))
    
    # Transaction metadata
    transaction_id = Column(String(255))  # External payment transaction ID
    access_granted_at = Column(DateTime(timezone=True))
    access_expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    resource = relationship("MarketplaceResource", back_populates="orders")
    
    __table_args__ = (
        Index('idx_marketplace_orders_buyer_status', 'buyer_user_id', 'status'),
        Index('idx_marketplace_orders_resource_status', 'resource_id', 'status'),
        Index('idx_marketplace_orders_created', 'created_at'),
    )


class MarketplaceAnalytics(Base):
    """Analytics and metrics for marketplace resources"""
    __tablename__ = 'marketplace_analytics'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(PGUUID(as_uuid=True), ForeignKey('marketplace_resources.id'), nullable=False, index=True)
    
    # Time period
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # Usage metrics
    views = Column(Integer, default=0)
    downloads = Column(Integer, default=0)
    purchases = Column(Integer, default=0)
    revenue = Column(DECIMAL(10, 2), default=0)
    
    # Performance metrics
    average_rating = Column(DECIMAL(3, 2))
    review_count = Column(Integer, default=0)
    conversion_rate = Column(DECIMAL(5, 4))  # views to purchases
    
    # User engagement
    unique_viewers = Column(Integer, default=0)
    return_users = Column(Integer, default=0)
    average_session_duration = Column(Integer)  # in seconds
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('resource_id', 'date', 'period_type', name='unique_analytics_per_period'),
        Index('idx_marketplace_analytics_date', 'date'),
        Index('idx_marketplace_analytics_resource_date', 'resource_id', 'date'),
    )


# ============================================================================
# ADDITIONAL CONSTRAINTS AND INDEXES
# ============================================================================

# BigInteger already imported above

# Additional indexes for performance
Index('idx_ai_models_category_provider', AIModelListing.model_category, AIModelListing.model_provider)
Index('idx_datasets_category_format', DatasetListing.dataset_category, DatasetListing.data_format)
Index('idx_agents_type_capabilities', AgentWorkflowListing.agent_type)
Index('idx_tools_category', MCPToolListing.tool_category)
Index('idx_compute_type_location', ComputeResourceListing.resource_type)
Index('idx_knowledge_type_domain', KnowledgeResourceListing.knowledge_type, KnowledgeResourceListing.domain_specialization)
Index('idx_evaluation_type', EvaluationServiceListing.evaluation_type)
Index('idx_training_type', TrainingServiceListing.training_type)
Index('idx_safety_category', SafetyToolListing.safety_category)