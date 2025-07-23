"""
Pydantic Validation Schemas
============================

Comprehensive validation schemas for all PRSM API endpoints and data structures.
"""

from pydantic import BaseModel, validator, Field, root_validator
from typing import Optional, Dict, List, Any, Union
from decimal import Decimal
from datetime import datetime
from enum import Enum

from .sanitization import sanitize_text_input, sanitize_query_content, sanitize_user_id
from .exceptions import BusinessLogicValidationError


class ThinkingModeEnum(str, Enum):
    """Thinking mode validation enum"""
    QUICK = "quick"  
    INTERMEDIATE = "intermediate"
    DEEP = "deep"


class VerbosityLevelEnum(str, Enum):
    """Verbosity level validation enum"""
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    ACADEMIC = "academic"


class UserTierEnum(str, Enum):
    """User tier validation enum"""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    RESEARCH = "research"


# Base validation schema
class BaseValidationSchema(BaseModel):
    """Base schema with common validation rules"""
    
    class Config:
        # Validate on assignment
        validate_assignment = True
        # Use enum values
        use_enum_values = True
        # Allow population by field name or alias
        allow_population_by_field_name = True
        # Forbid extra fields by default
        extra = "forbid"
    
    @root_validator(pre=True)
    def sanitize_string_fields(cls, values):
        """Sanitize all string fields"""
        if isinstance(values, dict):
            sanitized = {}
            for key, value in values.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    try:
                        sanitized[key] = sanitize_text_input(value, field_name=key)
                    except Exception:
                        # If sanitization fails, keep original for validation error
                        sanitized[key] = value
                else:
                    sanitized[key] = value
            return sanitized
        return values


# User validation schemas
class UserValidationSchema(BaseValidationSchema):
    """User information validation"""
    user_id: str = Field(..., min_length=3, max_length=64, description="User identifier")
    user_tier: UserTierEnum = Field(UserTierEnum.STANDARD, description="User service tier")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        return sanitize_user_id(v)


# Query validation schemas
class QueryValidationSchema(BaseValidationSchema):
    """Core query validation schema"""
    query: str = Field(..., min_length=3, max_length=50000, description="User query text")
    user_id: str = Field(..., min_length=3, max_length=64, description="User identifier")
    query_id: Optional[str] = Field(None, max_length=128, description="Optional query identifier")
    
    @validator('query')
    def validate_query_content(cls, v):
        return sanitize_query_content(v)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        return sanitize_user_id(v)
    
    @validator('query_id', pre=True)
    def validate_query_id(cls, v):
        if v is not None:
            # Generate query ID if not provided
            import uuid
            return str(uuid.uuid4())
        return v


class NWTNRequestSchema(QueryValidationSchema):
    """NWTN processing request validation"""
    thinking_mode: ThinkingModeEnum = Field(
        ThinkingModeEnum.INTERMEDIATE,
        description="Reasoning complexity level"
    )
    verbosity_level: VerbosityLevelEnum = Field(
        VerbosityLevelEnum.STANDARD,
        description="Output detail level"
    )
    user_tier: UserTierEnum = Field(
        UserTierEnum.STANDARD,
        description="User service tier"
    )
    
    # Advanced configuration
    breakthrough_discovery_enabled: bool = Field(
        False,
        description="Enable breakthrough discovery mode"
    )
    analogical_chain_depth: int = Field(
        3,
        ge=1,
        le=6,
        description="Maximum analogical reasoning chain depth"
    )
    max_processing_time_seconds: int = Field(
        3600,
        ge=10,
        le=7200,
        description="Maximum processing time in seconds"
    )
    
    # Context and preferences
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional processing context"
    )
    show_reasoning_trace: bool = Field(
        False,
        description="Include detailed reasoning trace in response"
    )
    
    @validator('context')
    def validate_context(cls, v):
        if v is not None:
            # Limit context size
            if len(str(v)) > 10000:
                raise BusinessLogicValidationError(
                    "Context too large",
                    business_rule="max_context_size",
                    field="context",
                    value=len(str(v))
                )
        return v
    
    @root_validator
    def validate_processing_requirements(cls, values):
        """Validate processing requirement combinations"""
        thinking_mode = values.get('thinking_mode')
        max_time = values.get('max_processing_time_seconds')
        
        # DEEP mode requires more time
        if thinking_mode == ThinkingModeEnum.DEEP and max_time < 300:
            raise BusinessLogicValidationError(
                "DEEP thinking mode requires at least 300 seconds processing time",
                business_rule="deep_mode_time_requirement",
                field="max_processing_time_seconds"
            )
        
        return values


# Tokenomics validation schemas
class TokenomicsRequestSchema(BaseValidationSchema):
    """FTNS tokenomics request validation"""
    user_id: str = Field(..., min_length=3, max_length=64)
    operation_type: str = Field(..., min_length=1, max_length=50)
    amount: Optional[Decimal] = Field(None, ge=0, decimal_places=8)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        return sanitize_user_id(v)
    
    @validator('operation_type')
    def validate_operation_type(cls, v):
        allowed_operations = [
            'query_cost_calculation', 'balance_check', 'transaction',
            'market_rate_query', 'supply_adjustment'
        ]
        if v not in allowed_operations:
            raise BusinessLogicValidationError(
                f"Invalid operation type: {v}",
                business_rule="allowed_operations",
                field="operation_type",
                value=v
            )
        return v


class PricingCalculationRequestSchema(TokenomicsRequestSchema):
    """FTNS pricing calculation request"""
    query: str = Field(..., min_length=3, max_length=50000)
    thinking_mode: ThinkingModeEnum = Field(ThinkingModeEnum.INTERMEDIATE)
    verbosity_level: VerbosityLevelEnum = Field(VerbosityLevelEnum.STANDARD)
    user_tier: UserTierEnum = Field(UserTierEnum.STANDARD)
    
    @validator('query')
    def validate_query_content(cls, v):
        return sanitize_query_content(v)


# Marketplace validation schemas  
class MarketplaceRequestSchema(BaseValidationSchema):
    """Marketplace operation validation"""
    user_id: str = Field(..., min_length=3, max_length=64)
    operation: str = Field(..., min_length=1, max_length=50)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        return sanitize_user_id(v)
    
    @validator('operation')
    def validate_operation(cls, v):
        allowed_operations = [
            'search_assets', 'get_asset', 'create_asset', 'update_asset',
            'delete_asset', 'purchase_asset', 'rate_asset'
        ]
        if v not in allowed_operations:
            raise BusinessLogicValidationError(
                f"Invalid marketplace operation: {v}",
                business_rule="allowed_marketplace_operations",
                field="operation",
                value=v
            )
        return v


class AssetSearchSchema(MarketplaceRequestSchema):
    """Asset search validation"""
    search_term: str = Field(..., min_length=1, max_length=200)
    asset_type: Optional[str] = Field(None, max_length=50)
    max_results: int = Field(20, ge=1, le=100)
    min_quality_score: float = Field(0.0, ge=0.0, le=1.0)
    
    @validator('search_term')
    def validate_search_term(cls, v):
        return sanitize_text_input(v, field_name="search_term")
    
    @validator('asset_type')
    def validate_asset_type(cls, v):
        if v is not None:
            allowed_types = [
                'ai_model', 'dataset', 'tool', 'service', 'workflow',
                'knowledge_resource', 'evaluation_service'
            ]
            if v not in allowed_types:
                raise BusinessLogicValidationError(
                    f"Invalid asset type: {v}",
                    business_rule="allowed_asset_types",
                    field="asset_type",
                    value=v
                )
        return v


class AssetCreationSchema(MarketplaceRequestSchema):
    """Asset creation validation"""
    name: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    asset_type: str = Field(..., min_length=1, max_length=50)
    price_ftns: Decimal = Field(..., ge=0, decimal_places=8)
    
    # Optional fields
    tags: Optional[List[str]] = Field(None, max_items=20)
    metadata: Optional[Dict[str, Any]] = Field(None)
    
    @validator('name')
    def validate_name(cls, v):
        return sanitize_text_input(v, field_name="name")
    
    @validator('description')
    def validate_description(cls, v):
        return sanitize_text_input(v, field_name="description")
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None:
            # Sanitize each tag
            sanitized_tags = []
            for i, tag in enumerate(v):
                if isinstance(tag, str):
                    sanitized_tag = sanitize_text_input(tag, max_length=50, field_name=f"tags[{i}]")
                    if len(sanitized_tag.strip()) > 0:
                        sanitized_tags.append(sanitized_tag)
            return sanitized_tags
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        if v is not None:
            # Limit metadata size
            if len(str(v)) > 5000:
                raise BusinessLogicValidationError(
                    "Metadata too large",
                    business_rule="max_metadata_size",
                    field="metadata",
                    value=len(str(v))
                )
        return v


# API response validation schemas
class APIResponseSchema(BaseModel):
    """Standard API response format"""
    success: bool = Field(..., description="Request success status")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request tracking ID")


class ValidationErrorResponseSchema(APIResponseSchema):
    """Validation error response format"""
    success: bool = Field(False)
    error: Dict[str, Any] = Field(..., description="Validation error details")
    validation_errors: List[Dict[str, Any]] = Field(
        ..., description="Detailed validation errors"
    )


# Batch validation schema
class BatchRequestSchema(BaseValidationSchema):
    """Batch processing request validation"""
    requests: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100)
    batch_id: Optional[str] = Field(None, max_length=128)
    
    @validator('batch_id', pre=True)
    def generate_batch_id(cls, v):
        if v is None:
            import uuid
            return str(uuid.uuid4())
        return v
    
    @validator('requests')
    def validate_request_size(cls, v):
        # Limit total batch size
        total_size = sum(len(str(req)) for req in v)
        if total_size > 500000:  # 500KB limit
            raise BusinessLogicValidationError(
                "Batch request too large",
                business_rule="max_batch_size",
                field="requests",
                value=total_size
            )
        return v


# Health check schema
class HealthCheckSchema(BaseModel):
    """Health check validation (minimal validation needed)"""
    service: Optional[str] = Field(None, max_length=50)
    
    class Config:
        extra = "allow"  # Allow extra fields for health checks