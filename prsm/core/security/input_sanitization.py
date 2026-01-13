"""
Input Sanitization & Validation Security Module
===============================================

Comprehensive input sanitization and validation middleware for PRSM API security.
Protects against XSS, injection attacks, and malicious input while preserving
legitimate user content functionality.
"""

import html
import json
import re
import bleach
import structlog
from typing import Any, Dict, List, Optional, Union, Set
from urllib.parse import urlparse
from datetime import datetime, timezone

from fastapi import Request, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel, Field, validator

from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class SanitizationConfig(BaseModel):
    """Configuration for input sanitization"""
    
    # HTML sanitization settings
    allowed_html_tags: Set[str] = Field(default_factory=lambda: {
        'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'blockquote',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'code', 'pre'
    })
    
    allowed_html_attributes: Dict[str, List[str]] = Field(default_factory=lambda: {
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'width', 'height'],
        'code': ['class'],
        'pre': ['class']
    })
    
    # JSON validation settings
    max_json_depth: int = Field(default=10)
    max_json_keys: int = Field(default=1000)
    max_string_length: int = Field(default=100000)
    
    # URL validation settings
    allowed_url_schemes: Set[str] = Field(default_factory=lambda: {'http', 'https', 'mailto'})
    
    # SQL injection patterns to detect
    sql_injection_patterns: List[str] = Field(default_factory=lambda: [
        r"(?i)(union\s+select)",
        r"(?i)(insert\s+into)",
        r"(?i)(delete\s+from)",
        r"(?i)(drop\s+table)",
        r"(?i)(exec\s*\()",
        r"(?i)(script\s*>)",
        r"(?i)(javascript\s*:)",
        r"(?i)(vbscript\s*:)",
        r"(?i)(onload\s*=)",
        r"(?i)(onerror\s*=)"
    ])


class InputSanitizer:
    """
    Input sanitization and validation engine
    
    Features:
    - HTML sanitization with allowlist approach
    - SQL injection pattern detection
    - XSS prevention through content sanitization
    - JSON structure validation and depth limiting
    - URL validation and scheme filtering
    - Path traversal prevention
    - Special character normalization
    """
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        self.config = config or SanitizationConfig()
        
        # Pre-compile regex patterns for performance
        self.sql_injection_regex = [
            re.compile(pattern) for pattern in self.config.sql_injection_patterns
        ]
        
        # Configure bleach for HTML sanitization
        self.html_cleaner = bleach.Cleaner(
            tags=self.config.allowed_html_tags,
            attributes=self.config.allowed_html_attributes,
            strip=True,
            strip_comments=True
        )
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r'\.\.(/|\\)'),
            re.compile(r'(/|\\)\.\.'),
            re.compile(r'\0'),  # Null byte injection
            re.compile(r'%2e%2e'),  # URL encoded ..
            re.compile(r'%252e%252e'),  # Double URL encoded ..
        ]
    
    async def sanitize_string(
        self,
        value: str,
        allow_html: bool = False,
        max_length: Optional[int] = None,
        field_name: str = "input"
    ) -> str:
        """
        Sanitize a string input
        
        Args:
            value: Input string to sanitize
            allow_html: Whether to allow HTML content (sanitized)
            max_length: Maximum allowed length
            field_name: Field name for logging
            
        Returns:
            Sanitized string
            
        Raises:
            HTTPException: If input is malicious or invalid
        """
        try:
            if not isinstance(value, str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid input type for {field_name}: expected string"
                )
            
            # Check length limits
            if max_length and len(value) > max_length:
                await self._log_security_event(
                    "input_length_exceeded",
                    {"field": field_name, "length": len(value), "max_length": max_length}
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Input too long for {field_name}: {len(value)} > {max_length}"
                )
            
            # Check for SQL injection patterns
            await self._check_sql_injection(value, field_name)
            
            # Sanitize HTML if allowed, otherwise escape it
            if allow_html:
                # Use bleach to sanitize HTML with allowlist
                sanitized = self.html_cleaner.clean(value)
            else:
                # Escape HTML entities
                sanitized = html.escape(value, quote=True)
            
            # Normalize Unicode and remove dangerous characters
            sanitized = self._normalize_unicode(sanitized)
            
            return sanitized
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error sanitizing string", field=field_name, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Input sanitization error"
            )
    
    async def sanitize_json(
        self,
        data: Union[Dict, List, Any],
        max_depth: Optional[int] = None,
        current_depth: int = 0,
        field_name: str = "json_data"
    ) -> Union[Dict, List, Any]:
        """
        Sanitize JSON data recursively
        
        Args:
            data: JSON data to sanitize
            max_depth: Maximum nesting depth
            current_depth: Current recursion depth
            field_name: Field name for logging
            
        Returns:
            Sanitized JSON data
        """
        try:
            max_depth = max_depth or self.config.max_json_depth
            
            # Check depth limit
            if current_depth > max_depth:
                await self._log_security_event(
                    "json_depth_exceeded",
                    {"field": field_name, "depth": current_depth, "max_depth": max_depth}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"JSON nesting too deep for {field_name}: {current_depth} > {max_depth}"
                )
            
            if isinstance(data, dict):
                # Check key count
                if len(data) > self.config.max_json_keys:
                    await self._log_security_event(
                        "json_keys_exceeded",
                        {"field": field_name, "keys": len(data), "max_keys": self.config.max_json_keys}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Too many JSON keys for {field_name}: {len(data)} > {self.config.max_json_keys}"
                    )
                
                # Recursively sanitize dictionary
                sanitized = {}
                for key, value in data.items():
                    # Sanitize keys
                    clean_key = await self.sanitize_string(
                        str(key), 
                        allow_html=False, 
                        max_length=255,
                        field_name=f"{field_name}.key"
                    )
                    
                    # Recursively sanitize values
                    clean_value = await self.sanitize_json(
                        value, 
                        max_depth, 
                        current_depth + 1,
                        f"{field_name}.{clean_key}"
                    )
                    
                    sanitized[clean_key] = clean_value
                
                return sanitized
                
            elif isinstance(data, list):
                # Recursively sanitize list items
                sanitized = []
                for i, item in enumerate(data):
                    clean_item = await self.sanitize_json(
                        item, 
                        max_depth, 
                        current_depth + 1,
                        f"{field_name}[{i}]"
                    )
                    sanitized.append(clean_item)
                
                return sanitized
                
            elif isinstance(data, str):
                # Sanitize string values
                return await self.sanitize_string(
                    data, 
                    allow_html=False,
                    max_length=self.config.max_string_length,
                    field_name=field_name
                )
                
            else:
                # Return primitive types as-is (int, float, bool, None)
                return data
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error sanitizing JSON", field=field_name, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JSON sanitization error"
            )
    
    async def validate_url(self, url: str, field_name: str = "url") -> str:
        """
        Validate and sanitize URL input
        
        Args:
            url: URL to validate
            field_name: Field name for logging
            
        Returns:
            Validated URL
            
        Raises:
            HTTPException: If URL is malicious or invalid
        """
        try:
            if not url:
                return url
            
            # Parse URL
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme and parsed.scheme.lower() not in self.config.allowed_url_schemes:
                await self._log_security_event(
                    "invalid_url_scheme",
                    {"field": field_name, "url": url, "scheme": parsed.scheme}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid URL scheme for {field_name}: {parsed.scheme}"
                )
            
            # Check for suspicious patterns
            suspicious_patterns = [
                'javascript:', 'vbscript:', 'data:', 'file:',
                'ftp:', 'gopher:', 'ldap:', 'dict:'
            ]
            
            url_lower = url.lower()
            for pattern in suspicious_patterns:
                if pattern in url_lower:
                    await self._log_security_event(
                        "suspicious_url_pattern",
                        {"field": field_name, "url": url, "pattern": pattern}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Suspicious URL pattern detected in {field_name}"
                    )
            
            return url
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error validating URL", field=field_name, url=url, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid URL format for {field_name}"
            )
    
    async def validate_file_path(self, path: str, field_name: str = "path") -> str:
        """
        Validate file path to prevent path traversal attacks
        
        Args:
            path: File path to validate
            field_name: Field name for logging
            
        Returns:
            Validated path
            
        Raises:
            HTTPException: If path is malicious
        """
        try:
            if not path:
                return path
            
            # Check for path traversal patterns
            for pattern in self.path_traversal_patterns:
                if pattern.search(path):
                    await self._log_security_event(
                        "path_traversal_attempt",
                        {"field": field_name, "path": path}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid path format for {field_name}"
                    )
            
            # Check for absolute paths (depending on use case)
            if path.startswith('/') or (len(path) > 1 and path[1] == ':'):
                await self._log_security_event(
                    "absolute_path_attempt",
                    {"field": field_name, "path": path}
                )
                # Note: This might be legitimate in some cases
                logger.warning("Absolute path detected", field=field_name, path=path)
            
            return path
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error validating file path", field=field_name, path=path, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid path format for {field_name}"
            )
    
    async def _check_sql_injection(self, value: str, field_name: str):
        """Check for SQL injection patterns"""
        for pattern in self.sql_injection_regex:
            if pattern.search(value):
                await self._log_security_event(
                    "sql_injection_attempt",
                    {"field": field_name, "pattern": pattern.pattern}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid input detected for {field_name}"
                )
    
    def _normalize_unicode(self, value: str) -> str:
        """Normalize Unicode characters and remove dangerous ones"""
        # Normalize Unicode to NFC form
        import unicodedata
        normalized = unicodedata.normalize('NFC', value)
        
        # Remove control characters except common whitespace
        cleaned = ''.join(
            char for char in normalized
            if unicodedata.category(char) != 'Cc' or char in '\n\r\t '
        )
        
        return cleaned
    
    async def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring"""
        try:
            await audit_logger.log_security_event(
                event_type=f"input_sanitization_{event_type}",
                user_id="system",  # Input sanitization is system-level
                details=details,
                security_level="warning"
            )
        except Exception as e:
            logger.error("Failed to log security event", error=str(e))


# Global input sanitizer instance
input_sanitizer = InputSanitizer()


async def sanitize_string(
    value: str,
    allow_html: bool = False,
    max_length: Optional[int] = None,
    field_name: str = "input"
) -> str:
    """Convenience function for string sanitization"""
    return await input_sanitizer.sanitize_string(value, allow_html, max_length, field_name)


async def sanitize_json(
    data: Union[Dict, List, Any],
    max_depth: Optional[int] = None,
    field_name: str = "json_data"
) -> Union[Dict, List, Any]:
    """Convenience function for JSON sanitization"""
    return await input_sanitizer.sanitize_json(data, max_depth, 0, field_name)


async def validate_url(url: str, field_name: str = "url") -> str:
    """Convenience function for URL validation"""
    return await input_sanitizer.validate_url(url, field_name)


async def validate_file_path(path: str, field_name: str = "path") -> str:
    """Convenience function for file path validation"""
    return await input_sanitizer.validate_file_path(path, field_name)