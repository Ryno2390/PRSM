"""
PRSM Security Module
===================

Comprehensive security framework for PRSM including input sanitization,
request size limits, TLS configuration, and protection against common
web application attacks.

Security Features:
- Input sanitization and validation
- Request size limits
- TLS/SSL configuration for production
- Secure models with validation
- Rate limiting
- Authentication and authorization
"""

from .input_sanitization import (
    InputSanitizer, SanitizationConfig, input_sanitizer,
    sanitize_string, sanitize_json, validate_url, validate_file_path
)

from .request_limits import (
    RequestLimitsMiddleware, RequestLimitsConfig, WebSocketLimitsManager,
    request_limits_config, websocket_limits_manager,
    validate_websocket_message, cleanup_websocket_connection
)

from .secure_models import (
    SecureBaseModel, SecureUserInput, SecureCredentialData,
    SecureWebSocketMessage, SecureTransferRequest, SecureConversationMessage,
    SecureFileUpload, create_secure_user_input, create_secure_websocket_message,
    create_secure_transfer_request
)

from .tls_config import (
    TLSConfig, TLSMode, TLSVersion,
    get_tls_config, get_database_ssl_config, get_redis_ssl_config,
    get_enhanced_security_headers, get_uvicorn_ssl_config,
    validate_production_tls, create_ssl_context
)

__all__ = [
    # Input sanitization
    "InputSanitizer",
    "SanitizationConfig", 
    "input_sanitizer",
    "sanitize_string",
    "sanitize_json",
    "validate_url",
    "validate_file_path",
    
    # Request limits
    "RequestLimitsMiddleware",
    "RequestLimitsConfig",
    "WebSocketLimitsManager",
    "request_limits_config",
    "websocket_limits_manager",
    "validate_websocket_message",
    "cleanup_websocket_connection",
    
    # Secure models
    "SecureBaseModel",
    "SecureUserInput",
    "SecureCredentialData",
    "SecureWebSocketMessage",
    "SecureTransferRequest",
    "SecureConversationMessage",
    "SecureFileUpload",
    "create_secure_user_input",
    "create_secure_websocket_message",
    "create_secure_transfer_request",

    # TLS configuration
    "TLSConfig",
    "TLSMode",
    "TLSVersion",
    "get_tls_config",
    "get_database_ssl_config",
    "get_redis_ssl_config",
    "get_enhanced_security_headers",
    "get_uvicorn_ssl_config",
    "validate_production_tls",
    "create_ssl_context"
]