"""
Validation Exception Hierarchy
==============================

Custom exception classes for validation errors with detailed context.
"""

from typing import Dict, List, Any, Optional


class ValidationError(Exception):
    """Base validation error with detailed context"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        error_code: str = "VALIDATION_ERROR",
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.field = field
        self.value = value
        self.error_code = error_code
        self.context = context or {}
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "error": self.error_code,
            "message": self.message,
            "field": self.field,
            "context": self.context
        }


class SecurityValidationError(ValidationError):
    """Security-related validation errors"""
    
    def __init__(
        self,
        message: str,
        security_risk: str,
        field: Optional[str] = None,
        value: Any = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.security_risk = security_risk
        context = context or {}
        context["security_risk"] = security_risk
        
        super().__init__(
            message=message,
            field=field,
            value=value,
            error_code="SECURITY_VALIDATION_ERROR",
            context=context
        )


class InputSanitizationError(SecurityValidationError):
    """Input sanitization errors"""
    
    def __init__(
        self,
        message: str,
        original_input: str,
        sanitized_input: str,
        field: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.original_input = original_input
        self.sanitized_input = sanitized_input
        
        context = context or {}
        context.update({
            "original_length": len(original_input),
            "sanitized_length": len(sanitized_input),
            "modification_made": original_input != sanitized_input
        })
        
        super().__init__(
            message=message,
            security_risk="input_sanitization",
            field=field,
            value=original_input,
            context=context
        )


class SchemaValidationError(ValidationError):
    """Pydantic schema validation errors with enhanced context"""
    
    def __init__(
        self,
        pydantic_errors: List[Dict[str, Any]],
        schema_name: str,
        input_data: Dict[str, Any]
    ):
        self.pydantic_errors = pydantic_errors
        self.schema_name = schema_name
        self.input_data = input_data
        
        # Create user-friendly error message
        error_messages = []
        for error in pydantic_errors:
            field_path = " -> ".join(str(x) for x in error.get("loc", []))
            error_msg = error.get("msg", "Validation failed")
            error_messages.append(f"{field_path}: {error_msg}")
        
        message = f"Schema validation failed for {schema_name}: {'; '.join(error_messages)}"
        
        super().__init__(
            message=message,
            error_code="SCHEMA_VALIDATION_ERROR",
            context={
                "schema": schema_name,
                "errors": pydantic_errors,
                "field_count": len(input_data)
            }
        )


class BusinessLogicValidationError(ValidationError):
    """Business logic validation errors"""
    
    def __init__(
        self,
        message: str,
        business_rule: str,
        field: Optional[str] = None,
        value: Any = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.business_rule = business_rule
        
        context = context or {}
        context["business_rule"] = business_rule
        
        super().__init__(
            message=message,
            field=field,
            value=value,
            error_code="BUSINESS_LOGIC_ERROR",
            context=context
        )


class RateLimitValidationError(ValidationError):
    """Rate limiting validation errors"""
    
    def __init__(
        self,
        message: str,
        user_id: str,
        limit_type: str,
        current_count: int,
        limit: int,
        reset_time: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.limit_type = limit_type
        self.current_count = current_count
        self.limit = limit
        self.reset_time = reset_time
        
        context = context or {}
        context.update({
            "user_id": user_id,
            "limit_type": limit_type,
            "current_count": current_count,
            "limit": limit,
            "reset_time": reset_time
        })
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            context=context
        )


class ResourceValidationError(ValidationError):
    """Resource availability validation errors"""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        requested: Any,
        available: Any,
        context: Optional[Dict[str, Any]] = None
    ):
        self.resource_type = resource_type
        self.requested = requested
        self.available = available
        
        context = context or {}
        context.update({
            "resource_type": resource_type,
            "requested": requested,
            "available": available
        })
        
        super().__init__(
            message=message,
            error_code="RESOURCE_VALIDATION_ERROR",
            context=context
        )


# Validation error aggregator
class ValidationErrorCollection:
    """Collection of validation errors for batch validation"""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def add_error(self, error: ValidationError) -> None:
        """Add a validation error to the collection"""
        self.errors.append(error)
    
    def add_field_error(
        self,
        field: str,
        message: str,
        value: Any = None,
        error_code: str = "FIELD_VALIDATION_ERROR"
    ) -> None:
        """Add a field-specific validation error"""
        error = ValidationError(
            message=message,
            field=field,
            value=value,
            error_code=error_code
        )
        self.add_error(error)
    
    def has_errors(self) -> bool:
        """Check if there are any validation errors"""
        return len(self.errors) > 0
    
    def get_error_count(self) -> int:
        """Get the number of validation errors"""
        return len(self.errors)
    
    def get_errors_by_field(self, field: str = None) -> List[ValidationError]:
        """Get errors for a specific field"""
        if field is None:
            return self.errors
        return [error for error in self.errors if error.field == field]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all errors to dictionary format"""
        return {
            "error_count": len(self.errors),
            "errors": [error.to_dict() for error in self.errors]
        }
    
    def raise_if_errors(self) -> None:
        """Raise exception if there are any validation errors"""
        if self.has_errors():
            raise ValidationError(
                message=f"Validation failed with {len(self.errors)} error(s)",
                error_code="MULTIPLE_VALIDATION_ERRORS",
                context=self.to_dict()
            )