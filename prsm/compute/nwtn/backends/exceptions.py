#!/usr/bin/env python3
"""
Backend Exceptions
===================

Exception hierarchy for LLM backend errors. These exceptions provide
structured error handling for the NWTN pipeline's backend interactions.

Exception Hierarchy:
    BackendError (base)
    ├── BackendUnavailableError - Backend not available for requests
    ├── BackendTimeoutError - Request timed out
    ├── BackendRateLimitError - Rate limit exceeded
    ├── BackendAuthenticationError - API key invalid or missing
    ├── BackendResponseError - Invalid response from backend
    ├── AllBackendsFailedError - All backends in fallback chain failed
    └── ModelNotFoundError - Requested model not found
"""

from typing import List, Optional, Tuple


class BackendError(Exception):
    """
    Base exception for all backend-related errors.
    
    All backend exceptions inherit from this class, making it easy
    to catch any backend-related error with a single except clause.
    
    Attributes:
        message: Human-readable error description
        backend_type: The backend type that raised the error (if applicable)
    """
    
    def __init__(self, message: str, backend_type: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.backend_type = backend_type
    
    def __str__(self) -> str:
        if self.backend_type:
            return f"[{self.backend_type}] {self.message}"
        return self.message


class BackendUnavailableError(BackendError):
    """
    Backend is not available for requests.
    
    This error is raised when:
    - The backend service is down or unreachable
    - Network connectivity issues prevent API calls
    - The backend failed to initialize properly
    - Required dependencies are not installed
    
    Attributes:
        message: Human-readable error description
        backend_type: The backend type that is unavailable
        original_error: The underlying exception (if any)
    """
    
    def __init__(
        self,
        message: str,
        backend_type: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message, backend_type)
        self.original_error = original_error
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.original_error:
            return f"{base} (caused by: {self.original_error})"
        return base


class BackendTimeoutError(BackendError):
    """
    Request to the backend timed out.
    
    This error is raised when:
    - The API request exceeds the configured timeout
    - The backend takes too long to respond
    - Network latency is too high
    
    Attributes:
        message: Human-readable error description
        backend_type: The backend type that timed out
        timeout_seconds: The timeout duration that was exceeded
    """
    
    def __init__(
        self,
        message: str,
        backend_type: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ):
        super().__init__(message, backend_type)
        self.timeout_seconds = timeout_seconds
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.timeout_seconds:
            return f"{base} (timeout: {self.timeout_seconds}s)"
        return base


class BackendRateLimitError(BackendError):
    """
    Rate limit exceeded for the backend.
    
    This error is raised when:
    - API rate limits are exceeded
    - Too many requests in a time window
    - Token limits are exceeded
    
    Attributes:
        message: Human-readable error description
        backend_type: The backend type that rate limited
        retry_after: Seconds to wait before retrying (if provided by API)
        limit_type: Type of limit exceeded (requests, tokens, etc.)
    """
    
    def __init__(
        self,
        message: str,
        backend_type: Optional[str] = None,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None
    ):
        super().__init__(message, backend_type)
        self.retry_after = retry_after
        self.limit_type = limit_type
    
    def __str__(self) -> str:
        base = super().__str__()
        parts = []
        if self.retry_after:
            parts.append(f"retry_after={self.retry_after}s")
        if self.limit_type:
            parts.append(f"limit={self.limit_type}")
        if parts:
            return f"{base} ({', '.join(parts)})"
        return base


class BackendAuthenticationError(BackendError):
    """
    Authentication failed for the backend.
    
    This error is raised when:
    - API key is missing
    - API key is invalid or expired
    - Authentication credentials are malformed
    - Access is denied due to permissions
    
    Attributes:
        message: Human-readable error description
        backend_type: The backend type that failed authentication
        auth_type: Type of authentication that failed (api_key, oauth, etc.)
    """
    
    def __init__(
        self,
        message: str,
        backend_type: Optional[str] = None,
        auth_type: Optional[str] = None
    ):
        super().__init__(message, backend_type)
        self.auth_type = auth_type or "api_key"
    
    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (auth_type={self.auth_type})"


class BackendResponseError(BackendError):
    """
    Invalid or unexpected response from the backend.
    
    This error is raised when:
    - Response JSON is malformed
    - Required fields are missing from response
    - Response status code indicates an error
    - Response content is invalid
    
    Attributes:
        message: Human-readable error description
        backend_type: The backend type that returned invalid response
        status_code: HTTP status code (if applicable)
        response_data: The problematic response data (if available)
    """
    
    def __init__(
        self,
        message: str,
        backend_type: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[dict] = None
    ):
        super().__init__(message, backend_type)
        self.status_code = status_code
        self.response_data = response_data
    
    def __str__(self) -> str:
        base = super().__str__()
        parts = []
        if self.status_code:
            parts.append(f"status={self.status_code}")
        if self.response_data:
            # Truncate response data in string representation
            data_str = str(self.response_data)[:100]
            parts.append(f"data={data_str}...")
        if parts:
            return f"{base} ({', '.join(parts)})"
        return base


class AllBackendsFailedError(BackendError):
    """
    All backends in the fallback chain failed.
    
    This error is raised when:
    - The primary backend fails
    - All fallback backends also fail
    - No available backends can process the request
    
    Attributes:
        message: Human-readable error description
        errors: List of (backend_type, error_message) tuples for each failure
    """
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[Tuple[str, str]]] = None
    ):
        super().__init__(message)
        self.errors = errors or []
    
    def __str__(self) -> str:
        if not self.errors:
            return self.message
        
        error_summary = "; ".join(
            f"{backend}: {error}" for backend, error in self.errors
        )
        return f"{self.message} - Failures: [{error_summary}]"


class ModelNotFoundError(BackendError):
    """
    Requested model not found on the backend.
    
    This error is raised when:
    - The specified model_id doesn't exist
    - The model is not available in the region
    - The model has been deprecated or removed
    
    Attributes:
        message: Human-readable error description
        backend_type: The backend type where model was not found
        model_id: The model identifier that was not found
        available_models: List of available models (if known)
    """
    
    def __init__(
        self,
        message: str,
        backend_type: Optional[str] = None,
        model_id: Optional[str] = None,
        available_models: Optional[List[str]] = None
    ):
        super().__init__(message, backend_type)
        self.model_id = model_id
        self.available_models = available_models or []
    
    def __str__(self) -> str:
        base = super().__str__()
        parts = []
        if self.model_id:
            parts.append(f"model={self.model_id}")
        if self.available_models:
            # Show first few available models
            models_str = ", ".join(self.available_models[:3])
            if len(self.available_models) > 3:
                models_str += f" (+{len(self.available_models) - 3} more)"
            parts.append(f"available=[{models_str}]")
        if parts:
            return f"{base} ({', '.join(parts)})"
        return base


# Convenience aliases for backward compatibility
APIKeyValidationError = BackendAuthenticationError
RateLimitError = BackendRateLimitError