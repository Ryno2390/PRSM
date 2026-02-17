"""
PRSM API Standards and Configuration
Centralized configuration for API consistency and standardization
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


# API Configuration Constants
class APIConfig:
    """Centralized API configuration and standards"""
    
    # API Versioning
    API_VERSION = "v1"
    API_PREFIX = f"/api/{API_VERSION}"
    
    # Standard HTTP Status Codes
    STATUS_CODES = {
        "SUCCESS": 200,
        "CREATED": 201,
        "NO_CONTENT": 204,
        "BAD_REQUEST": 400,
        "UNAUTHORIZED": 401,
        "FORBIDDEN": 403,
        "NOT_FOUND": 404,
        "CONFLICT": 409,
        "UNPROCESSABLE_ENTITY": 422,
        "TOO_MANY_REQUESTS": 429,
        "INTERNAL_SERVER_ERROR": 500,
        "SERVICE_UNAVAILABLE": 503
    }
    
    # Rate Limiting Configuration
    RATE_LIMITS = {
        "guest": {"requests": 50, "window": 60},
        "user": {"requests": 100, "window": 60},
        "premium": {"requests": 300, "window": 60},
        "developer": {"requests": 500, "window": 60},
        "researcher": {"requests": 500, "window": 60},
        "enterprise": {"requests": 1000, "window": 60},
        "admin": {"requests": 2000, "window": 60}
    }
    
    # OpenAPI Tags for Documentation
    OPENAPI_TAGS = [
        {
            "name": "Authentication",
            "description": "User authentication, authorization, and session management"
        },
        {
            "name": "Models",
            "description": "AI model operations, execution, and management"
        },
        {
            "name": "Marketplace",
            "description": "FTNS token marketplace and model trading"
        },
        {
            "name": "Budget",
            "description": "Budget management and cost tracking"
        },
        {
            "name": "Tasks",
            "description": "Task management and execution tracking"
        },
        {
            "name": "Teams",
            "description": "Team collaboration and project management"
        },
        {
            "name": "Analytics",
            "description": "Usage analytics and performance metrics"
        },
        {
            "name": "Safety",
            "description": "Safety monitoring and circuit breaker systems"
        },
        {
            "name": "Governance",
            "description": "Platform governance and voting mechanisms"
        },
        {
            "name": "Health",
            "description": "System health checks and monitoring"
        }
    ]


class ErrorType(str, Enum):
    """Standardized error types"""
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    CONFLICT_ERROR = "conflict_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    SAFETY_VIOLATION = "safety_violation"


class APIErrorResponse(BaseModel):
    """Standardized error response format"""
    success: bool = False
    error_type: ErrorType
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None
    trace_id: Optional[str] = None


class BaseAPIResponse(BaseModel):
    """Base response model for all API endpoints"""
    success: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None


class PaginationParams(BaseModel):
    """Standard pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: Optional[str] = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseAPIResponse):
    """Standard paginated response format"""
    data: List[Any]
    pagination: Dict[str, Any] = Field(
        description="Pagination metadata",
        example={
            "page": 1,
            "page_size": 20,
            "total_items": 100,
            "total_pages": 5,
            "has_next": True,
            "has_previous": False
        }
    )


class HealthCheckResponse(BaseAPIResponse):
    """Standard health check response"""
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checks: Dict[str, Any] = Field(
        description="Individual component health checks",
        example={
            "database": {"status": "healthy", "response_time_ms": 5},
            "redis": {"status": "healthy", "response_time_ms": 2},
            "external_apis": {"status": "healthy", "response_time_ms": 150}
        }
    )


# Standard HTTP Response Examples
RESPONSE_EXAMPLES = {
    "success": {
        "description": "Successful operation",
        "content": {
            "application/json": {
                "example": {
                    "success": True,
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_123456789"
                }
            }
        }
    },
    "validation_error": {
        "description": "Validation error",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "error_type": "validation_error",
                    "message": "Invalid request parameters",
                    "details": {
                        "field_errors": {
                            "email": ["Invalid email format"],
                            "password": ["Password must be at least 8 characters"]
                        }
                    },
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_123456789"
                }
            }
        }
    },
    "authentication_error": {
        "description": "Authentication failed",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "error_type": "authentication_error",
                    "message": "Invalid or expired authentication token",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_123456789"
                }
            }
        }
    },
    "rate_limit_error": {
        "description": "Rate limit exceeded",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "error_type": "rate_limit_error",
                    "message": "Rate limit exceeded. Please try again later.",
                    "details": {
                        "retry_after": 60,
                        "limit": 100,
                        "window": 60
                    },
                    "timestamp": "2024-01-15T10:30:00Z",
                    "request_id": "req_123456789"
                }
            }
        }
    }
}


# Standard Security Headers (Production-grade)
# These headers address OWASP security recommendations
SECURITY_HEADERS = {
    # Prevent MIME type sniffing attacks
    "X-Content-Type-Options": "nosniff",

    # Prevent clickjacking attacks
    "X-Frame-Options": "DENY",

    # XSS protection (legacy but still useful)
    "X-XSS-Protection": "1; mode=block",

    # HSTS - Force HTTPS for 1 year including subdomains
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",

    # Control referrer information
    "Referrer-Policy": "strict-origin-when-cross-origin",

    # Permissions policy (replaces Feature-Policy)
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",

    # Content Security Policy - Restrict content sources
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' wss: https:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),

    # Cross-Origin policies for enhanced isolation
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",

    # Prevent caching of sensitive API responses
    "Cache-Control": "no-store, max-age=0"
}


# Standard CORS Configuration
CORS_CONFIG = {
    "allow_origins": ["https://prsm.ai", "https://app.prsm.ai"],
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    "allow_headers": [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-Request-ID",
        "X-Trace-ID"
    ]
}


def get_router_config(name: str, description: str) -> Dict[str, Any]:
    """Generate standardized router configuration"""
    return {
        "prefix": f"{APIConfig.API_PREFIX}/{name.lower()}",
        "tags": [name.title()],
        "responses": {
            400: RESPONSE_EXAMPLES["validation_error"],
            401: RESPONSE_EXAMPLES["authentication_error"],
            429: RESPONSE_EXAMPLES["rate_limit_error"]
        }
    }


def get_standard_responses() -> Dict[int, Dict[str, Any]]:
    """Get standard response configurations for OpenAPI"""
    return {
        200: RESPONSE_EXAMPLES["success"],
        400: RESPONSE_EXAMPLES["validation_error"],
        401: RESPONSE_EXAMPLES["authentication_error"],
        429: RESPONSE_EXAMPLES["rate_limit_error"]
    }