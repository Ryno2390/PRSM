"""
Security Status API
==================

API endpoints for monitoring security systems, request limits,
and input sanitization statistics.
"""

import structlog
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from prsm.auth import get_current_user
from prsm.auth.models import UserRole
from prsm.auth.auth_manager import auth_manager
from prsm.security import (
    request_limits_config, websocket_limits_manager,
    input_sanitizer
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/security", tags=["security"])


class SecurityStatusResponse(BaseModel):
    """Response model for security status"""
    request_limits: Dict[str, Any]
    websocket_limits: Dict[str, Any]
    input_sanitization: Dict[str, Any]
    system_status: Dict[str, Any]
    timestamp: str


@router.get("/status")
async def get_security_status(
    current_user: str = Depends(get_current_user)
) -> SecurityStatusResponse:
    """
    Get comprehensive security system status
    
    🔐 SECURITY:
    - Requires authentication
    - Admin permissions for detailed statistics
    - Returns sanitized information for regular users
    """
    try:
        # Check if user is admin for detailed stats
        user = await auth_manager.get_user_by_id(current_user)
        is_admin = user and user.role in [UserRole.ADMIN, UserRole.MODERATOR]
        
        # Get request limits statistics
        request_stats = {
            "enabled": True,
            "default_max_body_size": request_limits_config.default_max_body_size,
            "websocket_max_message_size": request_limits_config.websocket_max_message_size,
            "rate_limit_requests_per_minute": request_limits_config.rate_limit_requests_per_minute,
            "expensive_endpoints_count": len(request_limits_config.expensive_endpoints)
        }
        
        # Add detailed stats for admins
        if is_admin:
            # Get middleware stats if available
            # Note: In a real implementation, we'd get this from the middleware instance
            request_stats.update({
                "endpoint_limits_count": len(request_limits_config.endpoint_body_limits),
                "expensive_endpoints": list(request_limits_config.expensive_endpoints)
            })
        
        # Get WebSocket statistics
        websocket_stats = websocket_limits_manager.get_connection_stats()
        websocket_stats.update({
            "max_message_size": request_limits_config.websocket_max_message_size,
            "max_messages_per_minute": request_limits_config.websocket_max_messages_per_minute,
            "connection_timeout_seconds": request_limits_config.websocket_connection_timeout_seconds
        })
        
        # Get input sanitization configuration
        sanitization_stats = {
            "enabled": True,
            "max_json_depth": input_sanitizer.config.max_json_depth,
            "max_json_keys": input_sanitizer.config.max_json_keys,
            "max_string_length": input_sanitizer.config.max_string_length,
            "allowed_html_tags_count": len(input_sanitizer.config.allowed_html_tags),
            "sql_injection_patterns_count": len(input_sanitizer.config.sql_injection_patterns)
        }
        
        if is_admin:
            sanitization_stats.update({
                "allowed_html_tags": list(input_sanitizer.config.allowed_html_tags),
                "allowed_url_schemes": list(input_sanitizer.config.allowed_url_schemes)
            })
        
        # System status
        system_status = {
            "security_middleware_active": True,
            "input_sanitization_active": True,
            "websocket_protection_active": True,
            "user_permissions": "admin" if is_admin else "user"
        }
        
        return SecurityStatusResponse(
            request_limits=request_stats,
            websocket_limits=websocket_stats,
            input_sanitization=sanitization_stats,
            system_status=system_status,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get security status",
                    user_id=current_user,
                    error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security status"
        )


@router.get("/limits/config")
async def get_request_limits_config(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get request limits configuration (admin only)
    
    🔐 SECURITY:
    - Requires authentication and admin permissions
    - Returns detailed configuration for system monitoring
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required"
            )
        
        return {
            "default_max_body_size": request_limits_config.default_max_body_size,
            "endpoint_body_limits": request_limits_config.endpoint_body_limits,
            "websocket_max_message_size": request_limits_config.websocket_max_message_size,
            "websocket_max_messages_per_minute": request_limits_config.websocket_max_messages_per_minute,
            "rate_limit_requests_per_minute": request_limits_config.rate_limit_requests_per_minute,
            "rate_limit_expensive_requests_per_minute": request_limits_config.rate_limit_expensive_requests_per_minute,
            "expensive_endpoints": list(request_limits_config.expensive_endpoints),
            "request_timeout_seconds": request_limits_config.request_timeout_seconds,
            "websocket_connection_timeout_seconds": request_limits_config.websocket_connection_timeout_seconds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get request limits config",
                    user_id=current_user,
                    error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve request limits configuration"
        )


@router.get("/sanitization/config")
async def get_sanitization_config(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get input sanitization configuration (admin only)
    
    🔐 SECURITY:
    - Requires authentication and admin permissions
    - Returns detailed sanitization configuration
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required"
            )
        
        return {
            "allowed_html_tags": list(input_sanitizer.config.allowed_html_tags),
            "allowed_html_attributes": input_sanitizer.config.allowed_html_attributes,
            "max_json_depth": input_sanitizer.config.max_json_depth,
            "max_json_keys": input_sanitizer.config.max_json_keys,
            "max_string_length": input_sanitizer.config.max_string_length,
            "allowed_url_schemes": list(input_sanitizer.config.allowed_url_schemes),
            "sql_injection_patterns_count": len(input_sanitizer.config.sql_injection_patterns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get sanitization config",
                    user_id=current_user,
                    error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sanitization configuration"
        )


@router.post("/test/sanitization")
async def test_input_sanitization(
    test_data: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Test input sanitization on provided data (admin only)
    
    🔐 SECURITY:
    - Requires authentication and admin permissions
    - Allows testing sanitization logic
    - Does not store or log sensitive test data
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required"
            )
        
        from prsm.security import sanitize_string, sanitize_json
        
        results = {}
        
        # Test string sanitization if provided
        if "test_string" in test_data:
            try:
                sanitized = await sanitize_string(
                    test_data["test_string"],
                    allow_html=test_data.get("allow_html", False),
                    max_length=test_data.get("max_length"),
                    field_name="test_string"
                )
                results["string_sanitization"] = {
                    "success": True,
                    "original": test_data["test_string"],
                    "sanitized": sanitized
                }
            except Exception as e:
                results["string_sanitization"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test JSON sanitization if provided
        if "test_json" in test_data:
            try:
                sanitized = await sanitize_json(
                    test_data["test_json"],
                    max_depth=test_data.get("max_depth"),
                    field_name="test_json"
                )
                results["json_sanitization"] = {
                    "success": True,
                    "original": test_data["test_json"],
                    "sanitized": sanitized
                }
            except Exception as e:
                results["json_sanitization"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to test input sanitization",
                    user_id=current_user,
                    error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test input sanitization"
        )