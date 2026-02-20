"""
DEPRECATED: This module is deprecated.

Use prsm.economy.marketplace.expanded_models instead, which provides
comprehensive coverage of all marketplace resource types.

This module is kept for backward compatibility only.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Integer, DECIMAL, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ModelCategory(str, Enum):
    """Categories for AI models in the marketplace"""
    LANGUAGE_MODEL = "language_model"
    IMAGE_GENERATION = "image_generation"
    CODE_GENERATION = "code_generation"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    FINE_TUNED = "fine_tuned"
    RESEARCH = "research"
    CUSTOM = "custom"


class PricingTier(str, Enum):
    """Pricing tiers for model access"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    CUSTOM = "custom"


class ModelProvider(str, Enum):
    """Model providers and platforms"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGING_FACE = "hugging_face"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    META = "meta"
    COHERE = "cohere"
    STABILITY = "stability"
    COMMUNITY = "community"
    CUSTOM = "custom"
    PRSM_NATIVE = "prsm_native"


class ModelStatus(str, Enum):
    """Status of model listings"""
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    SUSPENDED = "suspended"


# Database Models

class ModelListingDB(Base):
    """Database model for marketplace model listings"""
    __tablename__ = "marketplace_model_listings"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    model_id = Column(String(255), nullable=False, unique=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    
    # Ownership & Provider Info
    owner_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    provider_name = Column(String(255))
    provider_url = Column(String(512))
    model_version = Column(String(50))
    
    # Pricing Information
    pricing_tier = Column(String(50), nullable=False, default="free")
    base_price = Column(DECIMAL(20, 8), default=0)  # Price in FTNS tokens
    price_per_token = Column(DECIMAL(20, 8), default=0)
    price_per_request = Column(DECIMAL(20, 8), default=0)
    price_per_minute = Column(DECIMAL(20, 8), default=0)
    
    # Technical Specifications
    context_length = Column(Integer)
    max_tokens = Column(Integer)
    input_modalities = Column(JSON)  # ["text", "image", "audio"]
    output_modalities = Column(JSON)  # ["text", "image", "audio"]
    languages_supported = Column(JSON)  # ["en", "es", "fr", ...]
    
    # Performance Metrics
    average_response_time = Column(DECIMAL(10, 3))  # seconds
    tokens_per_second = Column(Integer)
    uptime_percentage = Column(DECIMAL(5, 2))
    quality_score = Column(DECIMAL(3, 2))  # 0-10 scale
    
    # Usage Statistics
    total_requests = Column(Integer, default=0)
    total_tokens_processed = Column(Integer, default=0)
    total_revenue = Column(DECIMAL(20, 8), default=0)
    active_rentals = Column(Integer, default=0)
    
    # Marketplace Status
    status = Column(String(50), nullable=False, default="pending_review", index=True)
    featured = Column(Boolean, default=False, index=True)
    verified = Column(Boolean, default=False, index=True)
    popularity_score = Column(DECIMAL(10, 2), default=0)
    
    # Metadata and Configuration
    additional_data = Column(JSON)  # Flexible metadata storage
    api_endpoint = Column(String(512))
    documentation_url = Column(String(512))
    license_type = Column(String(100))
    tags = Column(JSON)  # ["gpt", "chat", "completion", ...]
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    last_used_at = Column(DateTime(timezone=True))
    
    # Relationships
    rentals = relationship("RentalAgreementDB", back_populates="model_listing")
    orders = relationship("MarketplaceOrderDB", back_populates="model_listing")


class RentalAgreementDB(Base):
    """Database model for model rental agreements"""
    __tablename__ = "marketplace_rental_agreements"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Parties
    renter_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    model_listing_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_model_listings.id"), nullable=False, index=True)
    
    # Rental Terms
    rental_type = Column(String(50), nullable=False)  # "hourly", "daily", "monthly", "per_use"
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True))
    duration_hours = Column(Integer)
    
    # Pricing
    total_cost = Column(DECIMAL(20, 8), nullable=False)
    hourly_rate = Column(DECIMAL(20, 8))
    token_allowance = Column(Integer)
    tokens_used = Column(Integer, default=0)
    
    # Usage Limits
    max_requests_per_hour = Column(Integer)
    max_concurrent_requests = Column(Integer, default=1)
    priority_level = Column(Integer, default=1)  # 1=low, 5=high
    
    # Status
    status = Column(String(50), nullable=False, default="active", index=True)  # active, expired, cancelled, suspended
    payment_status = Column(String(50), nullable=False, default="pending")
    
    # Metadata
    usage_metadata = Column(JSON)
    terms_accepted_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    model_listing = relationship("ModelListingDB", back_populates="rentals")
    orders = relationship("MarketplaceOrderDB", back_populates="rental_agreement")


class MarketplaceOrderDB(Base):
    """Database model for marketplace orders and transactions"""
    __tablename__ = "marketplace_orders"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Order Information
    order_number = Column(String(50), nullable=False, unique=True, index=True)
    buyer_user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    model_listing_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_model_listings.id"), nullable=False, index=True)
    rental_agreement_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_rental_agreements.id"), index=True)
    
    # Pricing
    subtotal = Column(DECIMAL(20, 8), nullable=False)
    platform_fee = Column(DECIMAL(20, 8), default=0)
    total_amount = Column(DECIMAL(20, 8), nullable=False)
    currency = Column(String(10), default="FTNS")
    
    # Payment Information
    payment_method = Column(String(50))  # "ftns_balance", "crypto_wallet", "credit_card"
    payment_status = Column(String(50), nullable=False, default="pending", index=True)
    payment_transaction_id = Column(String(255))
    payment_completed_at = Column(DateTime(timezone=True))
    
    # Order Status
    order_status = Column(String(50), nullable=False, default="pending", index=True)
    fulfillment_status = Column(String(50), default="pending")
    
    # Metadata
    order_metadata = Column(JSON)
    billing_address = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    model_listing = relationship("ModelListingDB", back_populates="orders")
    rental_agreement = relationship("RentalAgreementDB", back_populates="orders")


# Pydantic Models

class ModelMetadata(BaseModel):
    """Metadata for AI models"""
    training_data_size: Optional[str] = None
    training_compute: Optional[str] = None
    model_architecture: Optional[str] = None
    parameter_count: Optional[str] = None
    fine_tuning_dataset: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, float]] = None
    limitations: Optional[List[str]] = None
    use_cases: Optional[List[str]] = None
    safety_measures: Optional[List[str]] = None
    ethical_considerations: Optional[str] = None


class ModelListing(BaseModel):
    """Pydantic model for marketplace model listings"""
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    model_id: str = Field(..., min_length=1, max_length=255)
    provider: ModelProvider
    category: ModelCategory
    
    # Ownership
    owner_user_id: UUID
    provider_name: Optional[str] = Field(None, max_length=255)
    provider_url: Optional[str] = Field(None, max_length=512)
    model_version: Optional[str] = Field(None, max_length=50)
    
    # Pricing
    pricing_tier: PricingTier = PricingTier.FREE
    base_price: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_token: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_request: Decimal = Field(default=Decimal('0'), ge=0)
    price_per_minute: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Technical Specs
    context_length: Optional[int] = Field(None, ge=1)
    max_tokens: Optional[int] = Field(None, ge=1)
    input_modalities: List[str] = Field(default_factory=lambda: ["text"])
    output_modalities: List[str] = Field(default_factory=lambda: ["text"])
    languages_supported: List[str] = Field(default_factory=lambda: ["en"])
    
    # Performance
    average_response_time: Optional[Decimal] = Field(None, ge=0)
    tokens_per_second: Optional[int] = Field(None, ge=0)
    uptime_percentage: Optional[Decimal] = Field(None, ge=0, le=100)
    quality_score: Optional[Decimal] = Field(None, ge=0, le=10)
    
    # Status
    status: ModelStatus = ModelStatus.PENDING_REVIEW
    featured: bool = False
    verified: bool = False
    popularity_score: Decimal = Field(default=Decimal('0'), ge=0)
    
    # Additional Info
    metadata: Optional[ModelMetadata] = None
    api_endpoint: Optional[str] = Field(None, max_length=512)
    documentation_url: Optional[str] = Field(None, max_length=512)
    license_type: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags list"""
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        return [tag.lower().strip() for tag in v if tag.strip()]
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description content"""
        if v and len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters")
        return v.strip() if v else v


class RentalAgreement(BaseModel):
    """Pydantic model for rental agreements"""
    id: Optional[UUID] = None
    renter_user_id: UUID
    model_listing_id: UUID
    
    # Terms
    rental_type: str = Field(..., pattern=r'^(hourly|daily|monthly|per_use|unlimited)$')
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_hours: Optional[int] = Field(None, ge=1)
    
    # Pricing
    total_cost: Decimal = Field(..., ge=0)
    hourly_rate: Optional[Decimal] = Field(None, ge=0)
    token_allowance: Optional[int] = Field(None, ge=0)
    tokens_used: int = Field(default=0, ge=0)
    
    # Limits
    max_requests_per_hour: Optional[int] = Field(None, ge=1)
    max_concurrent_requests: int = Field(default=1, ge=1, le=100)
    priority_level: int = Field(default=1, ge=1, le=5)
    
    # Status
    status: str = Field(default="active", pattern=r'^(active|expired|cancelled|suspended)$')
    payment_status: str = Field(default="pending", pattern=r'^(pending|paid|failed|refunded)$')
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    terms_accepted_at: Optional[datetime] = None


class MarketplaceOrder(BaseModel):
    """Pydantic model for marketplace orders"""
    id: Optional[UUID] = None
    order_number: Optional[str] = None
    buyer_user_id: UUID
    model_listing_id: UUID
    rental_agreement_id: Optional[UUID] = None
    
    # Pricing
    subtotal: Decimal = Field(..., ge=0)
    platform_fee: Decimal = Field(default=Decimal('0'), ge=0)
    total_amount: Decimal = Field(..., ge=0)
    currency: str = Field(default="FTNS", pattern=r'^[A-Z]{3,10}$')
    
    # Payment
    payment_method: Optional[str] = Field(None, pattern=r'^(ftns_balance|crypto_wallet|credit_card)$')
    payment_status: str = Field(default="pending", pattern=r'^(pending|completed|failed|cancelled|refunded)$')
    payment_transaction_id: Optional[str] = None
    payment_completed_at: Optional[datetime] = None
    
    # Status
    order_status: str = Field(default="pending", pattern=r'^(pending|confirmed|processing|completed|cancelled)$')
    fulfillment_status: str = Field(default="pending", pattern=r'^(pending|processing|fulfilled|failed)$')
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Request/Response Models

class CreateModelListingRequest(BaseModel):
    """Request model for creating a new model listing"""
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    model_id: str = Field(..., min_length=1, max_length=255)
    provider: ModelProvider
    category: ModelCategory
    
    # Optional fields
    provider_name: Optional[str] = Field(None, max_length=255)
    provider_url: Optional[str] = Field(None, max_length=512)
    model_version: Optional[str] = Field(None, max_length=50)
    
    # Pricing
    pricing_tier: PricingTier = PricingTier.FREE
    base_price: Optional[Decimal] = Field(None, ge=0)
    price_per_token: Optional[Decimal] = Field(None, ge=0)
    price_per_request: Optional[Decimal] = Field(None, ge=0)
    price_per_minute: Optional[Decimal] = Field(None, ge=0)
    
    # Technical specs
    context_length: Optional[int] = Field(None, ge=1, le=1000000)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    input_modalities: List[str] = Field(default_factory=lambda: ["text"])
    output_modalities: List[str] = Field(default_factory=lambda: ["text"])
    languages_supported: List[str] = Field(default_factory=lambda: ["en"])
    
    # Additional info
    api_endpoint: Optional[str] = Field(None, max_length=512)
    documentation_url: Optional[str] = Field(None, max_length=512)
    license_type: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class RentModelRequest(BaseModel):
    """Request model for renting a model"""
    model_listing_id: UUID
    rental_type: str = Field(..., pattern=r'^(hourly|daily|monthly|per_use|unlimited)$')
    duration_hours: Optional[int] = Field(None, ge=1, le=8760)  # Max 1 year
    token_allowance: Optional[int] = Field(None, ge=1)
    max_requests_per_hour: Optional[int] = Field(None, ge=1, le=10000)
    max_concurrent_requests: int = Field(default=1, ge=1, le=10)
    payment_method: str = Field(default="ftns_balance", pattern=r'^(ftns_balance|crypto_wallet)$')


class MarketplaceSearchFilters(BaseModel):
    """Search filters for marketplace"""
    category: Optional[ModelCategory] = None
    provider: Optional[ModelProvider] = None
    pricing_tier: Optional[PricingTier] = None
    min_price: Optional[Decimal] = Field(None, ge=0)
    max_price: Optional[Decimal] = Field(None, ge=0)
    verified_only: bool = False
    featured_only: bool = False
    tags: Optional[List[str]] = None
    search_query: Optional[str] = Field(None, max_length=255)
    sort_by: str = Field(default="popularity", pattern=r'^(popularity|price|created_at|rating|name)$')
    sort_order: str = Field(default="desc", pattern=r'^(asc|desc)$')
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class MarketplaceStatsResponse(BaseModel):
    """Response model for marketplace statistics"""
    total_models: int
    total_providers: int
    total_categories: int
    total_rentals: int
    total_revenue: Decimal
    active_rentals: int
    featured_models: int
    verified_models: int
    average_model_rating: Optional[Decimal]
    most_popular_category: Optional[str]
    top_providers: List[Dict[str, Any]]
    
    model_config = {
        "from_attributes": True
    }