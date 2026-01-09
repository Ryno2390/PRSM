"""
Standardized Exception Handling for PRSM API
Provides consistent error responses and exception handling patterns
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .standards import APIErrorResponse, ErrorType, APIConfig


logger = logging.getLogger(__name__)


class PRSMException(Exception):
    """Base exception for PRSM API errors"""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}
        self.trace_id = trace_id or str(uuid4())
        super().__init__(self.message)


class ValidationException(PRSMException):
    """Validation error exception"""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, Any]] = None):
        details = {"field_errors": field_errors} if field_errors else None
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            status_code=APIConfig.STATUS_CODES["BAD_REQUEST"],
            details=details
        )


class AuthenticationException(PRSMException):
    """Authentication error exception"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            error_type=ErrorType.AUTHENTICATION_ERROR,
            status_code=APIConfig.STATUS_CODES["UNAUTHORIZED"]
        )


class AuthorizationException(PRSMException):
    """Authorization error exception"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_type=ErrorType.AUTHORIZATION_ERROR,
            status_code=APIConfig.STATUS_CODES["FORBIDDEN"]
        )


class NotFoundException(PRSMException):
    """Resource not found exception"""
    
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} with identifier '{identifier}' not found"
        super().__init__(
            message=message,
            error_type=ErrorType.NOT_FOUND_ERROR,
            status_code=APIConfig.STATUS_CODES["NOT_FOUND"],
            details={"resource": resource, "identifier": identifier}
        )


class ConflictException(PRSMException):
    """Resource conflict exception"""
    
    def __init__(self, message: str, conflicting_resource: Optional[str] = None):
        details = {"conflicting_resource": conflicting_resource} if conflicting_resource else None
        super().__init__(
            message=message,
            error_type=ErrorType.CONFLICT_ERROR,
            status_code=APIConfig.STATUS_CODES["CONFLICT"],
            details=details
        )


class RateLimitException(PRSMException):
    """Rate limit exceeded exception"""
    
    def __init__(self, retry_after: int, limit: int, window: int):
        message = f"Rate limit exceeded. Try again in {retry_after} seconds."
        super().__init__(
            message=message,
            error_type=ErrorType.RATE_LIMIT_ERROR,
            status_code=APIConfig.STATUS_CODES["TOO_MANY_REQUESTS"],
            details={
                "retry_after": retry_after,
                "limit": limit,
                "window": window
            }
        )


class InsufficientFundsException(PRSMException):
    """Insufficient FTNS funds exception"""
    
    def __init__(self, required: float, available: float):
        message = f"Insufficient FTNS balance. Required: {required:.4f}, Available: {available:.4f}"
        super().__init__(
            message=message,
            error_type=ErrorType.INSUFFICIENT_FUNDS,
            status_code=APIConfig.STATUS_CODES["UNPROCESSABLE_ENTITY"],
            details={
                "required": required,
                "available": available,
                "deficit": required - available
            }
        )


class SafetyViolationException(PRSMException):
    """Safety policy violation exception"""
    
    def __init__(self, message: str, safety_level: str, violation_details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type=ErrorType.SAFETY_VIOLATION,
            status_code=APIConfig.STATUS_CODES["UNPROCESSABLE_ENTITY"],
            details={
                "safety_level": safety_level,
                "violation_details": violation_details or {}
            }
        )


class ServiceUnavailableException(PRSMException):
    """Service temporarily unavailable exception"""
    
    def __init__(self, service: str, estimated_recovery: Optional[int] = None):
        message = f"Service '{service}' is temporarily unavailable"
        if estimated_recovery:
            message += f". Estimated recovery in {estimated_recovery} seconds"
        
        details = {"service": service}
        if estimated_recovery:
            details["estimated_recovery"] = estimated_recovery
            
        super().__init__(
            message=message,
            error_type=ErrorType.SERVICE_UNAVAILABLE,
            status_code=APIConfig.STATUS_CODES["SERVICE_UNAVAILABLE"],
            details=details
        )


async def prsm_exception_handler(request: Request, exc: PRSMException) -> JSONResponse:
    """Handle PRSM custom exceptions"""
    
    # Log the exception
    logger.error(
        f"PRSM Exception: {exc.error_type} - {exc.message}",
        extra={
            "trace_id": exc.trace_id,
            "status_code": exc.status_code,
            "details": exc.details,
            "request_url": str(request.url),
            "request_method": request.method
        }
    )
    
    # Create standardized error response
    error_response = APIErrorResponse(
        error_type=exc.error_type,
        message=exc.message,
        details=exc.details,
        request_id=getattr(request.state, 'request_id', None),
        trace_id=exc.trace_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers={"X-Trace-ID": exc.trace_id}
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions"""
    
    # Map HTTP status codes to error types
    error_type_mapping = {
        400: ErrorType.VALIDATION_ERROR,
        401: ErrorType.AUTHENTICATION_ERROR,
        403: ErrorType.AUTHORIZATION_ERROR,
        404: ErrorType.NOT_FOUND_ERROR,
        409: ErrorType.CONFLICT_ERROR,
        422: ErrorType.VALIDATION_ERROR,
        429: ErrorType.RATE_LIMIT_ERROR,
        500: ErrorType.INTERNAL_ERROR,
        503: ErrorType.SERVICE_UNAVAILABLE
    }
    
    error_type = error_type_mapping.get(exc.status_code, ErrorType.INTERNAL_ERROR)
    trace_id = str(uuid4())
    
    # Log the exception
    logger.error(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "trace_id": trace_id,
            "status_code": exc.status_code,
            "request_url": str(request.url),
            "request_method": request.method
        }
    )
    
    # Create standardized error response
    error_response = APIErrorResponse(
        error_type=error_type,
        message=str(exc.detail),
        request_id=getattr(request.state, 'request_id', None),
        trace_id=trace_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers={"X-Trace-ID": trace_id}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation exceptions"""
    
    trace_id = str(uuid4())
    
    # Process validation errors
    field_errors = {}
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        field_errors[field] = message
    
    # Log the exception
    logger.warning(
        f"Validation Exception: {len(exc.errors())} field errors",
        extra={
            "trace_id": trace_id,
            "field_errors": field_errors,
            "request_url": str(request.url),
            "request_method": request.method
        }
    )
    
    # Create standardized error response
    error_response = APIErrorResponse(
        error_type=ErrorType.VALIDATION_ERROR,
        message="Request validation failed",
        details={"field_errors": field_errors},
        request_id=getattr(request.state, 'request_id', None),
        trace_id=trace_id
    )
    
    return JSONResponse(
        status_code=APIConfig.STATUS_CODES["BAD_REQUEST"],
        content=error_response.dict(),
        headers={"X-Trace-ID": trace_id}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    
    trace_id = str(uuid4())
    
    # Log the exception with full traceback
    logger.error(
        f"Unexpected Exception: {type(exc).__name__} - {str(exc)}",
        extra={
            "trace_id": trace_id,
            "traceback": traceback.format_exc(),
            "request_url": str(request.url),
            "request_method": request.method
        },
        exc_info=True
    )
    
    # Create standardized error response (don't expose internal details)
    error_response = APIErrorResponse(
        error_type=ErrorType.INTERNAL_ERROR,
        message="An unexpected error occurred. Please try again later.",
        request_id=getattr(request.state, 'request_id', None),
        trace_id=trace_id
    )
    
    return JSONResponse(
        status_code=APIConfig.STATUS_CODES["INTERNAL_SERVER_ERROR"],
        content=error_response.dict(),
        headers={"X-Trace-ID": trace_id}
    )


def register_exception_handlers(app):
    """Register all exception handlers with FastAPI app"""
    
    app.add_exception_handler(PRSMException, prsm_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)


# Convenience functions for common exceptions
def raise_validation_error(message: str, field_errors: Optional[Dict[str, Any]] = None):
    """Raise a standardized validation error"""
    raise ValidationException(message, field_errors)


def raise_not_found(resource: str, identifier: str):
    """Raise a standardized not found error"""
    raise NotFoundException(resource, identifier)


def raise_unauthorized(message: str = "Authentication required"):
    """Raise a standardized authentication error"""
    raise AuthenticationException(message)


def raise_forbidden(message: str = "Insufficient permissions"):
    """Raise a standardized authorization error"""
    raise AuthorizationException(message)


def raise_conflict(message: str, conflicting_resource: Optional[str] = None):
    """Raise a standardized conflict error"""
    raise ConflictException(message, conflicting_resource)


def raise_rate_limit(retry_after: int, limit: int, window: int):
    """Raise a standardized rate limit error"""
    raise RateLimitException(retry_after, limit, window)


def raise_insufficient_funds(required: float, available: float):
    """Raise a standardized insufficient funds error"""
    raise InsufficientFundsException(required, available)


def raise_safety_violation(message: str, safety_level: str, violation_details: Optional[Dict[str, Any]] = None):
    """Raise a standardized safety violation error"""
    raise SafetyViolationException(message, safety_level, violation_details)


def raise_service_unavailable(service: str, estimated_recovery: Optional[int] = None):
    """Raise a standardized service unavailable error"""
    raise ServiceUnavailableException(service, estimated_recovery)