"""
PRSM SDK Exceptions
Custom exception classes for PRSM SDK error handling
"""


class PRSMError(Exception):
    """Base exception for all PRSM SDK errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class AuthenticationError(PRSMError):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTH_ERROR")


class InsufficientFundsError(PRSMError):
    """Raised when FTNS balance is insufficient for operation"""
    
    def __init__(self, required: float, available: float):
        message = f"Insufficient FTNS balance. Required: {required}, Available: {available}"
        super().__init__(message, "INSUFFICIENT_FUNDS", {
            "required": required,
            "available": available
        })


class SafetyViolationError(PRSMError):
    """Raised when content violates safety policies"""
    
    def __init__(self, message: str, safety_level: str = None):
        super().__init__(message, "SAFETY_VIOLATION", {
            "safety_level": safety_level
        })


class NetworkError(PRSMError):
    """Raised when network requests fail"""
    
    def __init__(self, message: str):
        super().__init__(message, "NETWORK_ERROR")


class ModelNotFoundError(PRSMError):
    """Raised when requested model is not available"""
    
    def __init__(self, model_id: str):
        message = f"Model not found: {model_id}"
        super().__init__(message, "MODEL_NOT_FOUND", {
            "model_id": model_id
        })


class ToolExecutionError(PRSMError):
    """Raised when MCP tool execution fails"""
    
    def __init__(self, tool_name: str, error_message: str):
        message = f"Tool execution failed: {tool_name} - {error_message}"
        super().__init__(message, "TOOL_EXECUTION_ERROR", {
            "tool_name": tool_name,
            "error_message": error_message
        })


class RateLimitError(PRSMError):
    """Raised when API rate limits are exceeded"""
    
    def __init__(self, retry_after: int = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, "RATE_LIMIT_ERROR", {
            "retry_after": retry_after
        })


class ValidationError(PRSMError):
    """Raised when request validation fails"""
    
    def __init__(self, field: str, message: str):
        super().__init__(f"Validation error for field '{field}': {message}", "VALIDATION_ERROR", {
            "field": field,
            "validation_message": message
        })