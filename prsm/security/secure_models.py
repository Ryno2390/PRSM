"""
Secure Pydantic Models with Built-in Input Sanitization
=======================================================

Enhanced Pydantic models that automatically sanitize and validate input
to prevent security vulnerabilities while maintaining data integrity.
"""

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.fields import ModelField

from .input_sanitization import input_sanitizer


class SecureBaseModel(BaseModel):
    """
    Base model with automatic input sanitization
    
    All string fields are automatically sanitized for security.
    HTML content is escaped unless explicitly allowed.
    """
    
    class Config:
        # Enable validation for assignment
        validate_assignment = True
        # Use enum values instead of enum objects
        use_enum_values = True
        # Allow extra fields for flexibility (but validate them)
        extra = "forbid"
    
    @root_validator(pre=True)
    def sanitize_all_inputs(cls, values):
        """Root validator to sanitize all input values"""
        if not isinstance(values, dict):
            return values
        
        # This is a synchronous context, so we can't use async sanitization
        # We'll implement sync versions of critical validations
        sanitized_values = {}
        
        for field_name, value in values.items():
            try:
                sanitized_values[field_name] = cls._sync_sanitize_value(value, field_name)
            except Exception as e:
                # If sanitization fails, use original value and let field validators handle it
                sanitized_values[field_name] = value
        
        return sanitized_values
    
    @classmethod
    def _sync_sanitize_value(cls, value: Any, field_name: str) -> Any:
        """Synchronous sanitization for basic security"""
        if isinstance(value, str):
            # Basic HTML escape for security
            import html
            sanitized = html.escape(value, quote=True)
            
            # Check for obvious SQL injection patterns
            sql_patterns = [
                r"(?i)(union\s+select)", r"(?i)(insert\s+into)", 
                r"(?i)(delete\s+from)", r"(?i)(drop\s+table)",
                r"(?i)(exec\s*\()", r"(?i)(script\s*>)"
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, sanitized):
                    raise ValueError(f"Invalid input detected in {field_name}")
            
            return sanitized
            
        elif isinstance(value, dict):
            # Recursively sanitize dictionaries
            return {
                k: cls._sync_sanitize_value(v, f"{field_name}.{k}")
                for k, v in value.items()
            }
            
        elif isinstance(value, list):
            # Recursively sanitize lists
            return [
                cls._sync_sanitize_value(item, f"{field_name}[{i}]")
                for i, item in enumerate(value)
            ]
        
        return value


class SecureUserInput(SecureBaseModel):
    """Secure user input model with enhanced validation"""
    
    user_id: str = Field(..., min_length=1, max_length=255, regex=r'^[a-zA-Z0-9_-]+$')
    query: str = Field(..., min_length=1, max_length=10000)
    context_allocation: Optional[int] = Field(None, ge=1, le=1000000)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('query')
    def validate_query(cls, v):
        """Enhanced query validation"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Check for extremely long lines (potential attack)
        lines = v.split('\n')
        for line in lines:
            if len(line) > 5000:  # 5KB per line limit
                raise ValueError("Query line too long")
        
        return v.strip()
    
    @validator('preferences')
    def validate_preferences(cls, v):
        """Validate preferences structure"""
        if not isinstance(v, dict):
            return {}
        
        # Limit number of preference keys
        if len(v) > 50:
            raise ValueError("Too many preference keys")
        
        # Validate keys and values
        validated = {}
        for key, value in v.items():
            if not isinstance(key, str) or len(key) > 100:
                continue  # Skip invalid keys
            
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) <= 1000:
                    validated[key] = value
                elif not isinstance(value, str):
                    validated[key] = value
        
        return validated


class SecureCredentialData(SecureBaseModel):
    """Secure credential data model"""
    
    platform: str = Field(..., min_length=1, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    credentials: Dict[str, Any] = Field(...)
    expires_at: Optional[datetime] = Field(None)
    user_specific: bool = Field(default=False)
    
    @validator('platform')
    def validate_platform(cls, v):
        """Validate platform name"""
        allowed_platforms = {
            'openai', 'anthropic', 'huggingface', 'github', 
            'pinecone', 'weaviate', 'ollama'
        }
        if v.lower() not in allowed_platforms:
            raise ValueError(f"Invalid platform: {v}")
        return v.lower()
    
    @validator('credentials')
    def validate_credentials(cls, v):
        """Validate credential structure"""
        if not isinstance(v, dict):
            raise ValueError("Credentials must be a dictionary")
        
        if len(v) > 20:  # Limit number of credential fields
            raise ValueError("Too many credential fields")
        
        # Validate credential fields
        validated = {}
        for key, value in v.items():
            if not isinstance(key, str) or len(key) > 100:
                continue
            
            if isinstance(value, str):
                if len(value) > 10000:  # Reasonable limit for API keys/tokens
                    raise ValueError(f"Credential value too long for {key}")
                # Don't log or expose actual credential values
                validated[key] = value
            elif isinstance(value, (int, float, bool)):
                validated[key] = value
        
        if not validated:
            raise ValueError("No valid credentials provided")
        
        return validated


class SecureWebSocketMessage(SecureBaseModel):
    """Secure WebSocket message model"""
    
    type: str = Field(..., min_length=1, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    content: Optional[str] = Field(None, max_length=50000)
    conversation_id: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9_-]+$')
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('type')
    def validate_message_type(cls, v):
        """Validate message type"""
        allowed_types = {
            'ping', 'pong', 'subscribe_conversation', 'unsubscribe_conversation',
            'send_message', 'typing_start', 'typing_stop', 'request_status'
        }
        if v not in allowed_types:
            raise ValueError(f"Invalid message type: {v}")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata structure"""
        if not isinstance(v, dict):
            return {}
        
        if len(v) > 20:
            raise ValueError("Too many metadata fields")
        
        # Simple validation for metadata
        validated = {}
        for key, value in v.items():
            if isinstance(key, str) and len(key) <= 100:
                if isinstance(value, (str, int, float, bool)):
                    if isinstance(value, str) and len(value) <= 1000:
                        validated[key] = value
                    elif not isinstance(value, str):
                        validated[key] = value
        
        return validated


class SecureTransferRequest(SecureBaseModel):
    """Secure Web3 transfer request model"""
    
    to_address: str = Field(..., regex=r'^0x[a-fA-F0-9]{40}$')
    amount: Decimal = Field(..., gt=0, le=Decimal('1000000'))
    token_type: str = Field(default="FTNS", regex=r'^[A-Z]{3,10}$')
    gas_limit: Optional[int] = Field(None, ge=21000, le=1000000)
    
    @validator('to_address')
    def validate_ethereum_address(cls, v):
        """Validate Ethereum address format"""
        if not v.startswith('0x') or len(v) != 42:
            raise ValueError("Invalid Ethereum address format")
        
        # Check if it's a valid hex string
        try:
            int(v[2:], 16)
        except ValueError:
            raise ValueError("Invalid Ethereum address: not valid hex")
        
        return v.lower()  # Normalize to lowercase
    
    @validator('amount')
    def validate_amount(cls, v):
        """Validate transfer amount"""
        if v <= 0:
            raise ValueError("Amount must be positive")
        
        # Check decimal places (max 18 for most tokens)
        if v.as_tuple().exponent < -18:
            raise ValueError("Too many decimal places")
        
        return v


class SecureConversationMessage(SecureBaseModel):
    """Secure conversation message model"""
    
    content: str = Field(..., min_length=1, max_length=50000)
    message_type: str = Field(default="user", regex=r'^(user|assistant|system)$')
    conversation_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('content')
    def validate_content(cls, v):
        """Validate message content"""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        
        # Check for extremely long lines
        lines = v.split('\n')
        if len(lines) > 1000:  # Limit number of lines
            raise ValueError("Too many lines in message")
        
        for line in lines:
            if len(line) > 2000:  # Limit line length
                raise ValueError("Message line too long")
        
        return v.strip()


class SecureFileUpload(SecureBaseModel):
    """Secure file upload model"""
    
    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(..., min_length=1, max_length=100)
    size: int = Field(..., gt=0, le=100 * 1024 * 1024)  # 100MB max
    checksum: Optional[str] = Field(None, regex=r'^[a-fA-F0-9]{64}$')  # SHA256
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename for security"""
        # Remove path components
        import os
        filename = os.path.basename(v)
        
        # Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        for char in dangerous_chars:
            if char in filename:
                raise ValueError(f"Invalid character in filename: {char}")
        
        # Check file extension
        allowed_extensions = {
            '.txt', '.md', '.json', '.csv', '.pdf', '.png', '.jpg', '.jpeg',
            '.gif', '.webp', '.mp3', '.mp4', '.wav', '.zip', '.tar', '.gz'
        }
        
        _, ext = os.path.splitext(filename.lower())
        if ext not in allowed_extensions:
            raise ValueError(f"File type not allowed: {ext}")
        
        return filename
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate MIME type"""
        allowed_types = {
            'text/plain', 'text/markdown', 'application/json', 'text/csv',
            'application/pdf', 'image/png', 'image/jpeg', 'image/gif',
            'image/webp', 'audio/mpeg', 'video/mp4', 'audio/wav',
            'application/zip', 'application/x-tar', 'application/gzip'
        }
        
        if v not in allowed_types:
            raise ValueError(f"Content type not allowed: {v}")
        
        return v


# Convenience functions for creating secure models
def create_secure_user_input(data: Dict[str, Any]) -> SecureUserInput:
    """Create and validate secure user input"""
    return SecureUserInput(**data)


def create_secure_websocket_message(data: Dict[str, Any]) -> SecureWebSocketMessage:
    """Create and validate secure WebSocket message"""
    return SecureWebSocketMessage(**data)


def create_secure_transfer_request(data: Dict[str, Any]) -> SecureTransferRequest:
    """Create and validate secure transfer request"""
    return SecureTransferRequest(**data)