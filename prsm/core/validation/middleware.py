"""
Validation Middleware
====================

FastAPI middleware for request validation and security checking.
"""

import logging
import time
from typing import Dict, Any, Optional, Type, Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError as PydanticValidationError

from .exceptions import (
    ValidationError, SecurityValidationError, SchemaValidationError,
    ValidationErrorCollection
)
from .schemas import BaseValidationSchema, APIResponseSchema, ValidationErrorResponseSchema
from .sanitization import prevent_injection_attacks

logger = logging.getLogger(__name__)


class ValidationMiddleware:
    """Middleware for request validation and sanitization"""
    
    def __init__(self, app, validation_config: Optional[Dict[str, Any]] = None):
        self.app = app
        self.config = validation_config or {}
        self.route_schemas: Dict[str, Type[BaseModel]] = {}
        self.validation_stats = {
            "total_requests": 0,
            "validation_errors": 0,
            "security_errors": 0,
            "sanitization_applied": 0
        }
    
    async def __call__(self, request: Request, call_next):
        """Process request with validation"""
        start_time = time.time()
        
        # Update stats
        self.validation_stats["total_requests"] += 1
        
        try:
            # Pre-process validation
            await self._validate_request(request)
            
            # Process request
            response = await call_next(request)
            
            # Post-process validation (if needed)
            response = await self._validate_response(response)
            
            # Add validation headers
            self._add_validation_headers(response, start_time)
            
            return response
            
        except ValidationError as e:
            self.validation_stats["validation_errors"] += 1
            return self._create_validation_error_response(e, request)
        
        except SecurityValidationError as e:
            self.validation_stats["security_errors"] += 1
            logger.warning(f"Security validation error: {e.message}", extra={
                "security_risk": e.security_risk,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent"),
                "path": request.url.path
            })
            return self._create_security_error_response(e, request)
        
        except Exception as e:
            logger.error(f"Unexpected validation middleware error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal validation error", "message": str(e)}
            )
    
    def register_route_schema(self, path: str, method: str, schema: Type[BaseModel]) -> None:
        """Register validation schema for a specific route"""
        route_key = f"{method.upper()}:{path}"
        self.route_schemas[route_key] = schema
        logger.debug(f"Registered validation schema for {route_key}")
    
    async def _validate_request(self, request: Request) -> None:
        """Validate incoming request"""
        # Get route-specific schema
        route_key = f"{request.method}:{request.url.path}"
        schema = self.route_schemas.get(route_key)
        
        if schema is None:
            # No specific validation schema - apply general security checks
            await self._apply_general_security_checks(request)
            return
        
        # Get request data
        request_data = await self._extract_request_data(request)
        
        # Apply input sanitization
        if request_data:
            try:
                sanitized_data = prevent_injection_attacks(request_data)
                if sanitized_data != request_data:
                    self.validation_stats["sanitization_applied"] += 1
                request_data = sanitized_data
            except Exception as e:
                raise SecurityValidationError(
                    f"Input sanitization failed: {str(e)}",
                    security_risk="sanitization_failure"
                )
        
        # Validate against schema
        try:
            validated_data = schema(**request_data) if request_data else schema()
            
            # Store validated data for use in route handlers
            request.state.validated_data = validated_data
            
        except PydanticValidationError as e:
            raise SchemaValidationError(
                pydantic_errors=e.errors(),
                schema_name=schema.__name__,
                input_data=request_data or {}
            )
    
    async def _extract_request_data(self, request: Request) -> Dict[str, Any]:
        """Extract data from request for validation"""
        request_data = {}
        
        # Query parameters
        if request.query_params:
            request_data.update(dict(request.query_params))
        
        # Path parameters
        if hasattr(request, 'path_params') and request.path_params:
            request_data.update(request.path_params)  
        
        # JSON body
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
                if isinstance(body, dict):
                    request_data.update(body)
                else:
                    request_data["body"] = body
            except Exception as e:
                logger.warning(f"Failed to parse JSON body: {e}")
        
        # Form data
        elif request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
            try:
                form = await request.form()
                request_data.update(dict(form))
            except Exception as e:
                logger.warning(f"Failed to parse form data: {e}")
        
        return request_data
    
    async def _apply_general_security_checks(self, request: Request) -> None:
        """Apply general security checks when no specific schema is available"""
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            raise SecurityValidationError(
                "Request too large",
                security_risk="oversized_request"
            )
        
        # Check for suspicious headers
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 1000:
            raise SecurityValidationError(
                "Suspicious user agent header",
                security_risk="oversized_header"
            )
        
        # Basic rate limiting check (could be enhanced)
        client_ip = request.client.host
        if self._is_rate_limited(client_ip):
            raise SecurityValidationError(
                "Rate limit exceeded",
                security_risk="rate_limit_exceeded"
            )
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Basic rate limiting check (placeholder implementation)"""
        # This would be implemented with Redis or similar for production
        # For now, just return False
        return False
    
    async def _validate_response(self, response: Response) -> Response:
        """Validate outgoing response (optional)"""
        # Could add response validation here if needed
        return response
    
    def _add_validation_headers(self, response: Response, start_time: float) -> None:
        """Add validation-related headers to response"""
        processing_time = time.time() - start_time
        response.headers["X-Validation-Time"] = f"{processing_time:.3f}"
        response.headers["X-Validation-Version"] = "1.0"
    
    def _create_validation_error_response(
        self, 
        error: ValidationError, 
        request: Request
    ) -> JSONResponse:
        """Create validation error response"""
        error_response = ValidationErrorResponseSchema(
            success=False,
            message="Validation failed",
            error=error.to_dict(),
            validation_errors=[error.to_dict()],
            request_id=getattr(request.state, 'request_id', None)
        )
        
        status_code = 400
        if isinstance(error, SecurityValidationError):
            status_code = 403
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )
    
    def _create_security_error_response(
        self,
        error: SecurityValidationError,
        request: Request
    ) -> JSONResponse:
        """Create security error response"""
        # Log security event
        logger.warning(f"Security validation failed: {error.message}", extra={
            "security_risk": error.security_risk,
            "client_ip": request.client.host,
            "path": request.url.path,
            "user_agent": request.headers.get("user-agent")
        })
        
        # Return generic error message for security
        error_response = APIResponseSchema(
            success=False,
            message="Request validation failed",
            error={"code": "SECURITY_VALIDATION_ERROR", "type": "validation_error"},
            request_id=getattr(request.state, 'request_id', None)
        )
        
        return JSONResponse(
            status_code=403,
            content=error_response.dict()
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            **self.validation_stats,
            "error_rate": self.validation_stats["validation_errors"] / max(self.validation_stats["total_requests"], 1),
            "security_error_rate": self.validation_stats["security_errors"] / max(self.validation_stats["total_requests"], 1)
        }


class SecurityValidationMiddleware(ValidationMiddleware):
    """Enhanced security-focused validation middleware"""  
    
    def __init__(self, app, validation_config: Optional[Dict[str, Any]] = None):
        super().__init__(app, validation_config)
        self.security_config = validation_config.get("security", {}) if validation_config else {}
        
        # Security settings
        self.max_request_size = self.security_config.get("max_request_size", 10 * 1024 * 1024)
        self.enable_ip_filtering = self.security_config.get("enable_ip_filtering", False)
        self.blocked_ips = set(self.security_config.get("blocked_ips", []))
        self.rate_limit_enabled = self.security_config.get("rate_limit_enabled", True)
    
    async def _apply_general_security_checks(self, request: Request) -> None:
        """Enhanced security checks"""
        await super()._apply_general_security_checks(request)
        
        # IP filtering
        if self.enable_ip_filtering:
            client_ip = request.client.host
            if client_ip in self.blocked_ips:
                raise SecurityValidationError(
                    "IP address blocked",
                    security_risk="blocked_ip",
                    context={"client_ip": client_ip}
                )
        
        # Enhanced header validation
        self._validate_security_headers(request)
        
        # Path validation
        self._validate_request_path(request)
    
    def _validate_security_headers(self, request: Request) -> None:
        """Validate security-related headers"""
        # Check for suspicious headers
        suspicious_headers = ["x-forwarded-for", "x-real-ip", "x-remote-addr"]
        for header_name in suspicious_headers:
            header_value = request.headers.get(header_name)
            if header_value and len(header_value) > 200:
                raise SecurityValidationError(
                    f"Suspicious {header_name} header",
                    security_risk="suspicious_header",
                    context={"header": header_name, "length": len(header_value)}
                )
        
        # Validate content-type if present
        content_type = request.headers.get("content-type")
        if content_type:
            allowed_types = [
                "application/json", "application/x-www-form-urlencoded",
                "multipart/form-data", "text/plain"
            ]
            if not any(content_type.startswith(allowed) for allowed in allowed_types):
                logger.warning(f"Unusual content-type: {content_type}")
    
    def _validate_request_path(self, request: Request) -> None:
        """Validate request path for security issues"""
        path = request.url.path
        
        # Check for path traversal
        if "../" in path or "..%2f" in path.lower():
            raise SecurityValidationError(
                "Path traversal attempt detected",
                security_risk="path_traversal",
                context={"path": path}
            )
        
        # Check for suspicious patterns
        suspicious_patterns = [
            "/etc/", "/proc/", "/sys/", "/var/",
            "cmd.exe", "powershell", "/bin/"
        ]
        
        path_lower = path.lower()
        for pattern in suspicious_patterns:
            if pattern in path_lower:
                logger.warning(f"Suspicious path pattern detected: {pattern} in {path}")


# Validation helper functions
def validate_request(request_data: Dict[str, Any], schema: Type[BaseModel]) -> BaseModel:
    """Validate request data against schema"""
    try:
        return schema(**request_data)
    except PydanticValidationError as e:
        raise SchemaValidationError(
            pydantic_errors=e.errors(),
            schema_name=schema.__name__,
            input_data=request_data
        )


def validate_with_schema(data: Any, schema: Type[BaseModel]) -> BaseModel:
    """Generic validation with schema"""
    return validate_request(data if isinstance(data, dict) else {"data": data}, schema)