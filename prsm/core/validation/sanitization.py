"""
Input Sanitization System
=========================

Comprehensive input sanitization to prevent security vulnerabilities.
"""

import re
import html
import urllib.parse
from typing import Optional, Dict, Any, List
import logging

from .exceptions import InputSanitizationError, SecurityValidationError

logger = logging.getLogger(__name__)


class SecurityPatterns:
    """Security patterns for detecting and preventing attacks"""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"('|(\\')|(;|\\x3b)|(--|\s|\\n|\\r|\\t|\\v|\\f|\\0|\\x0b|\x0c|\x0d|\x0a)|(union|select|insert|delete|update|drop|create|alter|exec|execute|script|javascript|vbscript|onload|onerror|onclick|onmouseover|onmouseout|onkeypress|onkeyup|onkeydown|onchange|onblur|onfocus|onsubmit|onreset|onselect|onabort|ondblclick|onmousedown|onmouseup|onmousemove|onmouseenter|onmouseleave|onwheel|ondrag|ondrop|ondragover|ondragenter|ondragleave|ondragstart|ondragend|oncopy|oncut|onpaste|onbeforeunload|onerror|onhashchange|onload|onpageshow|onpagehide|onpopstate|onresize|onstorage|onunload|onafterprint|onbeforeprint).*",
        r"(\\x(3c|3e|2f|22|27))",
        r"(<|>|&lt;|&gt;)",
        r"(script|javascript|vbscript|onload|onerror)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<style[^>]*>.*?</style>",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]\\]",
        r"\.\./",
        r"~",
        r"/etc/passwd",
        r"/bin/",
        r"/usr/bin/",
        r"cmd\.exe",
        r"powershell",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"/etc/",
        r"/proc/",
        r"/sys/",
        r"c:\\",
        r"\\windows\\",
    ]


class InputSanitizer:
    """Comprehensive input sanitization system"""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.security_patterns = SecurityPatterns()
    
    def sanitize_text_input(
        self,
        text: str,
        max_length: int = 50000,
        allow_html: bool = False,
        field_name: str = "input"
    ) -> str:
        """Sanitize general text input"""
        if not isinstance(text, str):
            raise InputSanitizationError(
                "Input must be a string",
                original_input=str(text),
                sanitized_input="",
                field=field_name
            )
        
        original_text = text
        
        # Length validation
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated from {len(original_text)} to {max_length} characters")
        
        # Remove null bytes and control characters
        text = self._remove_control_characters(text)
        
        # HTML sanitization
        if not allow_html:
            text = html.escape(text)
        else:
            text = self._sanitize_html(text)
        
        # Security pattern checking
        self._check_security_patterns(text, field_name)
        
        # Unicode normalization
        text = self._normalize_unicode(text)
        
        if text != original_text:
            logger.info(f"Input sanitized for field {field_name}")
        
        return text
    
    def sanitize_query_content(
        self,
        query: str,
        max_length: int = 50000,
        field_name: str = "query"
    ) -> str:
        """Sanitize user query content for NWTN processing"""
        if not query or not query.strip():
            raise InputSanitizationError(
                "Query cannot be empty",
                original_input=query,
                sanitized_input="",
                field=field_name
            )
        
        original_query = query
        
        # Basic text sanitization
        query = self.sanitize_text_input(
            query,
            max_length=max_length,
            allow_html=False,
            field_name=field_name
        )
        
        # Query-specific sanitization
        query = self._sanitize_query_specific_patterns(query)
        
        # Validate minimum length after sanitization
        if len(query.strip()) < 3:
            raise InputSanitizationError(
                "Query too short after sanitization",
                original_input=original_query,
                sanitized_input=query,
                field=field_name
            )
        
        return query
    
    def sanitize_user_id(
        self,
        user_id: str,
        field_name: str = "user_id"
    ) -> str:
        """Sanitize user ID input"""
        if not isinstance(user_id, str):
            raise InputSanitizationError(
                "User ID must be a string",
                original_input=str(user_id),
                sanitized_input="",
                field=field_name
            )
        
        original_user_id = user_id
        
        # Remove whitespace
        user_id = user_id.strip()
        
        # Validate format (alphanumeric, underscore, hyphen only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise SecurityValidationError(
                "User ID contains invalid characters",
                security_risk="invalid_user_id_format",
                field=field_name,
                value=original_user_id
            )
        
        # Length validation
        if len(user_id) < 3 or len(user_id) > 64:
            raise InputSanitizationError(
                "User ID must be between 3 and 64 characters",
                original_input=original_user_id,
                sanitized_input=user_id,
                field=field_name
            )
        
        return user_id
    
    def sanitize_file_path(
        self,
        file_path: str,
        field_name: str = "file_path"
    ) -> str:
        """Sanitize file path input"""
        if not isinstance(file_path, str):
            raise InputSanitizationError(
                "File path must be a string",
                original_input=str(file_path),
                sanitized_input="",
                field=field_name
            )
        
        original_path = file_path
        
        # Check for path traversal attacks
        for pattern in self.security_patterns.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                raise SecurityValidationError(
                    "Path traversal attack detected",
                    security_risk="path_traversal",
                    field=field_name,
                    value=original_path
                )
        
        # Normalize path
        file_path = urllib.parse.unquote(file_path)
        file_path = file_path.replace("\\", "/")
        
        # Remove multiple slashes
        file_path = re.sub(r'/+', '/', file_path)
        
        return file_path
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters except common whitespace"""
        # Keep tab, newline, carriage return
        allowed_chars = ['\t', '\n', '\r']
        
        cleaned = []
        for char in text:
            if ord(char) < 32 and char not in allowed_chars:
                continue  # Skip control character
            elif ord(char) == 127:  # DEL character
                continue
            else:
                cleaned.append(char)
        
        return ''.join(cleaned)
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML while preserving safe tags"""
        # For now, we'll escape all HTML for maximum security
        # In the future, could implement whitelist-based HTML sanitization
        return html.escape(text)
    
    def _check_security_patterns(self, text: str, field_name: str) -> None:
        """Check for security attack patterns"""
        text_lower = text.lower()
        
        # Check SQL injection patterns
        for pattern in self.security_patterns.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise SecurityValidationError(
                    "SQL injection pattern detected",
                    security_risk="sql_injection",
                    field=field_name,
                    value=text
                )
        
        # Check XSS patterns
        for pattern in self.security_patterns.XSS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise SecurityValidationError(
                    "XSS pattern detected",
                    security_risk="xss_attack",
                    field=field_name,
                    value=text
                )
        
        # Check command injection patterns (strict mode only)
        if self.strict_mode:
            for pattern in self.security_patterns.COMMAND_INJECTION_PATTERNS:
                if re.search(pattern, text):
                    raise SecurityValidationError(
                        "Command injection pattern detected",
                        security_risk="command_injection",
                        field=field_name,
                        value=text
                    )
    
    def _sanitize_query_specific_patterns(self, query: str) -> str:
        """Apply query-specific sanitization"""
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove common problematic patterns while preserving query meaning
        # This is conservative to avoid breaking legitimate queries
        
        return query
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text"""
        import unicodedata
        
        # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
        return unicodedata.normalize('NFC', text)


# Global sanitizer instance
_default_sanitizer = InputSanitizer(strict_mode=True)


def sanitize_text_input(
    text: str,
    max_length: int = 50000,
    allow_html: bool = False,
    field_name: str = "input"
) -> str:
    """Convenience function for text sanitization"""
    return _default_sanitizer.sanitize_text_input(
        text, max_length, allow_html, field_name
    )


def sanitize_query_content(
    query: str,
    max_length: int = 50000,
    field_name: str = "query"
) -> str:
    """Convenience function for query sanitization"""
    return _default_sanitizer.sanitize_query_content(
        query, max_length, field_name
    )


def sanitize_user_id(user_id: str, field_name: str = "user_id") -> str:
    """Convenience function for user ID sanitization"""
    return _default_sanitizer.sanitize_user_id(user_id, field_name)


def prevent_injection_attacks(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prevent injection attacks across all input fields"""
    sanitized_data = {}
    
    for key, value in input_data.items():
        if isinstance(value, str):
            sanitized_data[key] = sanitize_text_input(value, field_name=key)
        elif isinstance(value, dict):
            sanitized_data[key] = prevent_injection_attacks(value)
        elif isinstance(value, list):
            sanitized_data[key] = [
                sanitize_text_input(item, field_name=f"{key}[{i}]") 
                if isinstance(item, str) else item
                for i, item in enumerate(value)
            ]
        else:
            sanitized_data[key] = value
    
    return sanitized_data