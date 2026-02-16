"""
PRSM Core Validation Framework
==============================

Centralized input validation system using Pydantic for all PRSM components.
This prevents security vulnerabilities and ensures data integrity.
"""

from .schemas import *
from .middleware import *
from .sanitization import *
from .exceptions import *


def validate_session_data(data: dict) -> bool:
    """Validate session data structure
    
    Args:
        data: Session data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If data is invalid
    """
    # Basic validation - can be extended
    required_fields = ['session_id', 'user_id']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")
    return True


def validate_user_input(data: dict) -> bool:
    """Validate user input data structure
    
    Args:
        data: User input data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If data is invalid
    """
    # Basic validation - can be extended
    required_fields = ['user_id']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")
    
    # Check for content or prompt
    if 'content' not in data and 'prompt' not in data:
        raise ValidationError("Missing required field: content or prompt")
    
    return True


__all__ = [
    # Validation schemas
    'QueryValidationSchema',
    'UserValidationSchema', 
    'NWTNRequestSchema',
    'TokenomicsRequestSchema',
    'MarketplaceRequestSchema',
    
    # Middleware
    'ValidationMiddleware',
    'SecurityValidationMiddleware',
    
    # Sanitization
    'sanitize_text_input',
    'sanitize_query_content',
    'prevent_injection_attacks',
    
    # Exceptions
    'ValidationError',
    'SecurityValidationError',
    'InputSanitizationError',
    
    # Validators
    'validate_request',
    'validate_with_schema',
    'validate_session_data',
    'validate_user_input',
]