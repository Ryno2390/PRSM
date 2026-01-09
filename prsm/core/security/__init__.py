"""
PRSM Security Module
===================

Comprehensive security framework for PRSM including input sanitization,
request size limits, and protection against common web application attacks.
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
    "create_secure_transfer_request"
]