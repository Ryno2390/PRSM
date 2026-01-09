"""
MCP Security Manager
===================

Security validation and sandboxing for MCP tool execution.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from .models import ToolCall, ToolDefinition, SecurityLevel

logger = logging.getLogger(__name__)


class SecurityRisk(str, Enum):
    """Security risk levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityValidationResult:
    """Result of security validation"""
    approved: bool
    risk_level: SecurityRisk
    reason: Optional[str] = None
    warnings: List[str] = None
    restrictions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.restrictions is None:
            self.restrictions = {}


class MCPSecurityManager:
    """
    Security manager for MCP tool execution
    
    Provides:
    - Tool call validation and risk assessment
    - Parameter sanitization and validation
    - Execution sandboxing and restrictions
    - Security policy enforcement
    """
    
    def __init__(self):
        # Security policies
        self.blocked_patterns = [
            r'.*\b(exec|eval|system|shell)\s*\(',  # Code execution patterns
            r'.*\b(subprocess|os\.system|commands)\.',  # System command patterns
            r'.*\b(__import__|globals|locals)\s*\(',  # Import/globals access
            r'.*\b(file|open)\s*\([\'"].*/etc/.*[\'"]',  # System file access
            r'.*\b(rm|del|delete|drop)\s+.*',  # Destructive operations
        ]
        
        self.suspicious_keywords = {
            'password', 'secret', 'key', 'token', 'auth', 'credential',
            'private', 'confidential', 'sensitive', 'admin', 'root',
            'sudo', 'su', 'chmod', 'chown', 'kill', 'terminate'
        }
        
        # Tool-specific restrictions
        self.tool_restrictions = {
            SecurityLevel.LOW: {
                'max_parameter_size': 1024,  # 1KB
                'allowed_file_operations': {'read'},
                'network_access': False,
                'system_access': False
            },
            SecurityLevel.MEDIUM: {
                'max_parameter_size': 10240,  # 10KB
                'allowed_file_operations': {'read', 'write'},
                'network_access': True,
                'system_access': False
            },
            SecurityLevel.HIGH: {
                'max_parameter_size': 102400,  # 100KB
                'allowed_file_operations': {'read', 'write', 'create'},
                'network_access': True,
                'system_access': True
            },
            SecurityLevel.CRITICAL: {
                'max_parameter_size': 1048576,  # 1MB
                'allowed_file_operations': {'read', 'write', 'create', 'delete'},
                'network_access': True,
                'system_access': True
            }
        }
        
        # User-specific security settings
        self.user_security_levels: Dict[str, SecurityLevel] = {}
        self.user_restrictions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized MCP security manager")
    
    async def validate_tool_call(self, tool_call: ToolCall, 
                                tool_def: ToolDefinition) -> SecurityValidationResult:
        """
        Validate a tool call for security compliance
        
        Args:
            tool_call: The tool call to validate
            tool_def: Tool definition with security metadata
            
        Returns:
            Security validation result
        """
        try:
            logger.debug(f"Validating tool call: {tool_call.tool_name} for user {tool_call.user_id}")
            
            # Check user security level
            user_max_level = self.get_user_security_level(tool_call.user_id)
            if self._security_level_value(tool_def.security_level) > self._security_level_value(user_max_level):
                return SecurityValidationResult(
                    approved=False,
                    risk_level=SecurityRisk.HIGH,
                    reason=f"Tool security level ({tool_def.security_level.value}) exceeds user permissions ({user_max_level.value})"
                )
            
            # Validate parameters
            param_result = await self._validate_parameters(tool_call, tool_def)
            if not param_result.approved:
                return param_result
            
            # Check for suspicious patterns
            pattern_result = await self._check_suspicious_patterns(tool_call)
            if not pattern_result.approved:
                return pattern_result
            
            # Validate against tool restrictions
            restriction_result = await self._validate_restrictions(tool_call, tool_def)
            if not restriction_result.approved:
                return restriction_result
            
            # Determine overall risk level
            risk_level = await self._assess_risk_level(tool_call, tool_def)
            
            # Build final result
            warnings = []
            warnings.extend(param_result.warnings)
            warnings.extend(pattern_result.warnings)
            warnings.extend(restriction_result.warnings)
            
            restrictions = {}
            restrictions.update(param_result.restrictions)
            restrictions.update(restriction_result.restrictions)
            
            logger.debug(f"Tool call validation passed: {tool_call.tool_name} (risk: {risk_level.value})")
            
            return SecurityValidationResult(
                approved=True,
                risk_level=risk_level,
                warnings=warnings,
                restrictions=restrictions
            )
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return SecurityValidationResult(
                approved=False,
                risk_level=SecurityRisk.CRITICAL,
                reason=f"Security validation failed: {str(e)}"
            )
    
    def set_user_security_level(self, user_id: str, level: SecurityLevel):
        """Set maximum security level for a user"""
        self.user_security_levels[user_id] = level
        logger.info(f"Set security level for user {user_id}: {level.value}")
    
    def get_user_security_level(self, user_id: str) -> SecurityLevel:
        """Get maximum security level for a user"""
        return self.user_security_levels.get(user_id, SecurityLevel.MEDIUM)
    
    def set_user_restrictions(self, user_id: str, restrictions: Dict[str, Any]):
        """Set additional restrictions for a user"""
        self.user_restrictions[user_id] = restrictions
        logger.info(f"Set custom restrictions for user {user_id}")
    
    def get_user_restrictions(self, user_id: str) -> Dict[str, Any]:
        """Get additional restrictions for a user"""
        return self.user_restrictions.get(user_id, {})
    
    # Private validation methods
    
    async def _validate_parameters(self, tool_call: ToolCall, 
                                 tool_def: ToolDefinition) -> SecurityValidationResult:
        """Validate tool call parameters for security"""
        warnings = []
        restrictions = {}
        
        # Check parameter size limits
        max_size = self.tool_restrictions[tool_def.security_level]['max_parameter_size']
        total_size = len(str(tool_call.parameters))
        
        if total_size > max_size:
            return SecurityValidationResult(
                approved=False,
                risk_level=SecurityRisk.HIGH,
                reason=f"Parameter size ({total_size} bytes) exceeds limit ({max_size} bytes)"
            )
        
        # Check for sensitive data in parameters
        for key, value in tool_call.parameters.items():
            value_str = str(value).lower()
            
            # Check for suspicious keywords
            for keyword in self.suspicious_keywords:
                if keyword in value_str:
                    warnings.append(f"Suspicious keyword '{keyword}' found in parameter '{key}'")
            
            # Check for potential file paths
            if isinstance(value, str) and ('/' in value or '\\' in value):
                if any(sensitive_path in value.lower() for sensitive_path in ['/etc/', '/root/', '/home/', 'c:\\windows\\']):
                    return SecurityValidationResult(
                        approved=False,
                        risk_level=SecurityRisk.HIGH,
                        reason=f"Potential access to sensitive system path in parameter '{key}': {value}"
                    )
            
            # Check for code injection patterns
            if isinstance(value, str):
                for pattern in self.blocked_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return SecurityValidationResult(
                            approved=False,
                            risk_level=SecurityRisk.CRITICAL,
                            reason=f"Potential code injection detected in parameter '{key}'"
                        )
        
        return SecurityValidationResult(
            approved=True,
            risk_level=SecurityRisk.LOW,
            warnings=warnings,
            restrictions=restrictions
        )
    
    async def _check_suspicious_patterns(self, tool_call: ToolCall) -> SecurityValidationResult:
        """Check for suspicious patterns in tool call"""
        warnings = []
        
        # Check tool name for suspicious patterns
        tool_name_lower = tool_call.tool_name.lower()
        
        suspicious_tool_patterns = [
            'exec', 'eval', 'system', 'shell', 'cmd', 'run', 'execute',
            'delete', 'remove', 'destroy', 'kill', 'terminate'
        ]
        
        for pattern in suspicious_tool_patterns:
            if pattern in tool_name_lower:
                warnings.append(f"Tool name contains potentially dangerous pattern: '{pattern}'")
        
        # Additional context-based checks
        if tool_call.security_context:
            context_str = str(tool_call.security_context).lower()
            if any(keyword in context_str for keyword in self.suspicious_keywords):
                warnings.append("Security context contains suspicious keywords")
        
        return SecurityValidationResult(
            approved=True,
            risk_level=SecurityRisk.LOW,
            warnings=warnings
        )
    
    async def _validate_restrictions(self, tool_call: ToolCall, 
                                   tool_def: ToolDefinition) -> SecurityValidationResult:
        """Validate against tool and user restrictions"""
        warnings = []
        restrictions = {}
        
        # Get tool-level restrictions
        tool_restrictions = self.tool_restrictions[tool_def.security_level]
        
        # Get user-specific restrictions
        user_restrictions = self.get_user_restrictions(tool_call.user_id)
        
        # Merge restrictions (user restrictions take precedence)
        merged_restrictions = {**tool_restrictions, **user_restrictions}
        
        # Apply network access restrictions
        if not merged_restrictions.get('network_access', True):
            # Check if tool might need network access
            network_indicators = ['http', 'url', 'download', 'upload', 'api', 'request']
            if any(indicator in tool_call.tool_name.lower() for indicator in network_indicators):
                warnings.append("Tool may require network access but it's restricted for this user")
                restrictions['network_access'] = False
        
        # Apply file operation restrictions
        allowed_file_ops = merged_restrictions.get('allowed_file_operations', set())
        file_indicators = {
            'read': ['read', 'get', 'fetch', 'load'],
            'write': ['write', 'save', 'store', 'update'],
            'create': ['create', 'new', 'make', 'generate'],
            'delete': ['delete', 'remove', 'del', 'destroy']
        }
        
        for operation, indicators in file_indicators.items():
            if any(indicator in tool_call.tool_name.lower() for indicator in indicators):
                if operation not in allowed_file_ops:
                    return SecurityValidationResult(
                        approved=False,
                        risk_level=SecurityRisk.MEDIUM,
                        reason=f"File operation '{operation}' not allowed for this user/tool combination"
                    )
        
        # Apply system access restrictions
        if not merged_restrictions.get('system_access', True):
            system_indicators = ['system', 'os', 'process', 'command', 'shell', 'terminal']
            if any(indicator in tool_call.tool_name.lower() for indicator in system_indicators):
                return SecurityValidationResult(
                    approved=False,
                    risk_level=SecurityRisk.HIGH,
                    reason="System access not allowed for this user"
                )
        
        return SecurityValidationResult(
            approved=True,
            risk_level=SecurityRisk.LOW,
            warnings=warnings,
            restrictions=restrictions
        )
    
    async def _assess_risk_level(self, tool_call: ToolCall, 
                               tool_def: ToolDefinition) -> SecurityRisk:
        """Assess overall risk level for the tool call"""
        risk_factors = []
        
        # Base risk from tool security level
        base_risk = {
            SecurityLevel.LOW: SecurityRisk.LOW,
            SecurityLevel.MEDIUM: SecurityRisk.MEDIUM,
            SecurityLevel.HIGH: SecurityRisk.HIGH,
            SecurityLevel.CRITICAL: SecurityRisk.CRITICAL
        }[tool_def.security_level]
        
        risk_factors.append(base_risk)
        
        # Additional risk factors
        param_size = len(str(tool_call.parameters))
        if param_size > 10000:  # Large parameters increase risk
            risk_factors.append(SecurityRisk.MEDIUM)
        
        # Tool name risk assessment
        tool_name_lower = tool_call.tool_name.lower()
        high_risk_names = ['execute', 'eval', 'system', 'shell', 'delete', 'remove']
        if any(name in tool_name_lower for name in high_risk_names):
            risk_factors.append(SecurityRisk.HIGH)
        
        # Return highest risk level
        risk_values = [self._risk_level_value(r) for r in risk_factors]
        max_risk_value = max(risk_values)
        
        return self._value_to_risk_level(max_risk_value)
    
    def _security_level_value(self, level: SecurityLevel) -> int:
        """Convert security level to numeric value"""
        return {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 2,
            SecurityLevel.HIGH: 3,
            SecurityLevel.CRITICAL: 4
        }[level]
    
    def _risk_level_value(self, risk: SecurityRisk) -> int:
        """Convert risk level to numeric value"""
        return {
            SecurityRisk.NONE: 0,
            SecurityRisk.LOW: 1,
            SecurityRisk.MEDIUM: 2,
            SecurityRisk.HIGH: 3,
            SecurityRisk.CRITICAL: 4
        }[risk]
    
    def _value_to_risk_level(self, value: int) -> SecurityRisk:
        """Convert numeric value to risk level"""
        risk_map = {
            0: SecurityRisk.NONE,
            1: SecurityRisk.LOW,
            2: SecurityRisk.MEDIUM,
            3: SecurityRisk.HIGH,
            4: SecurityRisk.CRITICAL
        }
        return risk_map.get(value, SecurityRisk.CRITICAL)
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security manager statistics"""
        return {
            "total_users_configured": len(self.user_security_levels),
            "users_with_restrictions": len(self.user_restrictions),
            "security_level_distribution": {
                level.value: sum(1 for l in self.user_security_levels.values() if l == level)
                for level in SecurityLevel
            },
            "blocked_patterns_count": len(self.blocked_patterns),
            "suspicious_keywords_count": len(self.suspicious_keywords)
        }