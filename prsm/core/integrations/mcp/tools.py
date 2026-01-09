"""
MCP Tool Registry and Management
===============================

Tool registry and management system for MCP-discovered tools.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict

from .models import ToolDefinition, ToolCall, ToolResult, ToolParameterType, SecurityLevel

logger = logging.getLogger(__name__)


class Tool:
    """Wrapper for MCP tool with additional metadata and validation"""
    
    def __init__(self, definition: ToolDefinition):
        self.definition = definition
        self.name = definition.name
        self.description = definition.description
        self.parameters = definition.parameters
        self.security_level = definition.security_level
        
        # Usage statistics
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.last_used = None
        
        # Validation cache
        self._parameter_cache = {}
        self._build_parameter_cache()
    
    def _build_parameter_cache(self):
        """Build parameter validation cache"""
        self._parameter_cache = {
            "required": {p.name for p in self.parameters if p.required},
            "optional": {p.name for p in self.parameters if not p.required},
            "types": {p.name: p.type for p in self.parameters},
            "enums": {p.name: p.enum for p in self.parameters if p.enum},
            "patterns": {p.name: p.pattern for p in self.parameters if p.pattern}
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validation result with errors and warnings
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "processed_params": {}
        }
        
        provided_params = set(parameters.keys())
        required_params = self._parameter_cache["required"]
        optional_params = self._parameter_cache["optional"]
        all_params = required_params | optional_params
        
        # Check for missing required parameters
        missing_required = required_params - provided_params
        if missing_required:
            result["valid"] = False
            result["errors"].append(f"Missing required parameters: {', '.join(missing_required)}")
        
        # Check for unknown parameters
        unknown_params = provided_params - all_params
        if unknown_params:
            result["warnings"].append(f"Unknown parameters will be ignored: {', '.join(unknown_params)}")
        
        # Validate parameter types and constraints
        for param_name, value in parameters.items():
            if param_name not in all_params:
                continue
            
            param_errors = self._validate_parameter_value(param_name, value)
            if param_errors:
                result["valid"] = False
                result["errors"].extend(param_errors)
            else:
                result["processed_params"][param_name] = value
        
        # Add defaults for missing optional parameters
        for param in self.parameters:
            if param.name not in result["processed_params"] and param.default is not None:
                result["processed_params"][param.name] = param.default
        
        return result
    
    def _validate_parameter_value(self, param_name: str, value: Any) -> List[str]:
        """Validate a single parameter value"""
        errors = []
        param_type = self._parameter_cache["types"].get(param_name)
        
        if param_type is None:
            return errors
        
        # Type validation
        if param_type == ToolParameterType.STRING and not isinstance(value, str):
            errors.append(f"Parameter '{param_name}' must be a string")
        elif param_type == ToolParameterType.INTEGER and not isinstance(value, int):
            errors.append(f"Parameter '{param_name}' must be an integer")
        elif param_type == ToolParameterType.NUMBER and not isinstance(value, (int, float)):
            errors.append(f"Parameter '{param_name}' must be a number")
        elif param_type == ToolParameterType.BOOLEAN and not isinstance(value, bool):
            errors.append(f"Parameter '{param_name}' must be a boolean")
        elif param_type == ToolParameterType.ARRAY and not isinstance(value, list):
            errors.append(f"Parameter '{param_name}' must be an array")
        elif param_type == ToolParameterType.OBJECT and not isinstance(value, dict):
            errors.append(f"Parameter '{param_name}' must be an object")
        
        # Enum validation
        enum_values = self._parameter_cache["enums"].get(param_name)
        if enum_values and value not in enum_values:
            errors.append(f"Parameter '{param_name}' must be one of: {', '.join(map(str, enum_values))}")
        
        # Pattern validation for strings
        if param_type == ToolParameterType.STRING and isinstance(value, str):
            pattern = self._parameter_cache["patterns"].get(param_name)
            if pattern:
                import re
                if not re.match(pattern, value):
                    errors.append(f"Parameter '{param_name}' does not match required pattern: {pattern}")
        
        # Range validation for numbers
        param_def = next((p for p in self.parameters if p.name == param_name), None)
        if param_def and isinstance(value, (int, float)):
            if param_def.minimum is not None and value < param_def.minimum:
                errors.append(f"Parameter '{param_name}' must be >= {param_def.minimum}")
            if param_def.maximum is not None and value > param_def.maximum:
                errors.append(f"Parameter '{param_name}' must be <= {param_def.maximum}")
        
        return errors
    
    def record_execution(self, result: ToolResult):
        """Record execution statistics"""
        self.call_count += 1
        self.last_used = datetime.utcnow()
        self.total_execution_time += result.execution_time
        
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.call_count == 0:
            return 0.0
        return (self.success_count / self.call_count) * 100
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time"""
        if self.call_count == 0:
            return 0.0
        return self.total_execution_time / self.call_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "security_level": self.security_level.value
        }


class ToolRegistry:
    """Registry for managing MCP tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, Set[str]] = defaultdict(set)
        self.security_levels: Dict[SecurityLevel, Set[str]] = defaultdict(set)
        self.tags: Dict[str, Set[str]] = defaultdict(set)
        
        # Registry statistics
        self.total_registrations = 0
        self.last_update = None
        
        logger.info("Initialized MCP tool registry")
    
    def register_tool(self, definition: ToolDefinition) -> bool:
        """
        Register a tool in the registry
        
        Args:
            definition: Tool definition to register
            
        Returns:
            True if registration successful
        """
        try:
            tool = Tool(definition)
            
            # Check for name conflicts
            if definition.name in self.tools:
                logger.warning(f"Tool '{definition.name}' already registered, updating...")
            
            # Register tool
            self.tools[definition.name] = tool
            
            # Update indices
            if definition.category:
                self.categories[definition.category].add(definition.name)
            
            self.security_levels[definition.security_level].add(definition.name)
            
            for tag in definition.tags:
                self.tags[tag].add(definition.name)
            
            self.total_registrations += 1
            self.last_update = datetime.utcnow()
            
            logger.info(f"Registered tool: {definition.name} (category: {definition.category}, security: {definition.security_level.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool '{definition.name}': {e}")
            return False
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[str] = None, 
                   security_level: Optional[SecurityLevel] = None,
                   tags: Optional[List[str]] = None) -> List[Tool]:
        """
        List tools with optional filtering
        
        Args:
            category: Filter by category
            security_level: Filter by security level
            tags: Filter by tags (AND operation)
            
        Returns:
            List of matching tools
        """
        tools = list(self.tools.values())
        
        # Filter by category
        if category:
            tools = [t for t in tools if t.definition.category == category]
        
        # Filter by security level
        if security_level:
            tools = [t for t in tools if t.definition.security_level == security_level]
        
        # Filter by tags
        if tags:
            tools = [t for t in tools if all(tag in t.definition.tags for tag in tags)]
        
        return tools
    
    def search_tools(self, query: str) -> List[Tool]:
        """
        Search tools by name, description, or tags
        
        Args:
            query: Search query string
            
        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self.tools.values():
            # Search in name
            if query_lower in tool.name.lower():
                matching_tools.append((tool, 3))  # High relevance
                continue
            
            # Search in description
            if query_lower in tool.description.lower():
                matching_tools.append((tool, 2))  # Medium relevance
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in tool.definition.tags):
                matching_tools.append((tool, 1))  # Low relevance
                continue
        
        # Sort by relevance and return tools
        matching_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in matching_tools]
    
    def get_tools_by_security_level(self, max_level: SecurityLevel) -> List[Tool]:
        """Get tools with security level at or below the specified level"""
        level_order = {
            SecurityLevel.LOW: 0,
            SecurityLevel.MEDIUM: 1,
            SecurityLevel.HIGH: 2,
            SecurityLevel.CRITICAL: 3
        }
        
        max_level_value = level_order[max_level]
        
        return [
            tool for tool in self.tools.values()
            if level_order[tool.security_level] <= max_level_value
        ]
    
    def get_categories(self) -> List[str]:
        """Get list of all tool categories"""
        return list(self.categories.keys())
    
    def get_tags(self) -> List[str]:
        """Get list of all tool tags"""
        return list(self.tags.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        now = datetime.utcnow()
        
        # Calculate usage statistics
        total_calls = sum(tool.call_count for tool in self.tools.values())
        total_successes = sum(tool.success_count for tool in self.tools.values())
        
        # Find most/least used tools
        most_used = max(self.tools.values(), key=lambda t: t.call_count, default=None)
        least_used = min(self.tools.values(), key=lambda t: t.call_count, default=None)
        
        # Recently used tools (last 24 hours)
        cutoff = now - timedelta(hours=24)
        recent_tools = [
            tool for tool in self.tools.values()
            if tool.last_used and tool.last_used > cutoff
        ]
        
        return {
            "total_tools": len(self.tools),
            "total_categories": len(self.categories),
            "total_tags": len(self.tags),
            "security_distribution": {
                level.value: len(tools) for level, tools in self.security_levels.items()
            },
            "total_calls": total_calls,
            "total_successes": total_successes,
            "overall_success_rate": (total_successes / max(total_calls, 1)) * 100,
            "most_used_tool": most_used.name if most_used else None,
            "least_used_tool": least_used.name if least_used else None,
            "recently_used_tools": len(recent_tools),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
    
    def validate_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a tool call before execution
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Returns:
            Validation result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "valid": False,
                "errors": [f"Tool '{tool_name}' not found"],
                "warnings": [],
                "processed_params": {}
            }
        
        return tool.validate_parameters(parameters)
    
    def record_tool_execution(self, tool_name: str, result: ToolResult):
        """Record tool execution result for statistics"""
        tool = self.get_tool(tool_name)
        if tool:
            tool.record_execution(result)
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the registry
        
        Args:
            name: Name of tool to unregister
            
        Returns:
            True if tool was found and unregistered
        """
        if name not in self.tools:
            return False
        
        tool = self.tools[name]
        
        # Remove from indices
        if tool.definition.category:
            self.categories[tool.definition.category].discard(name)
        
        self.security_levels[tool.security_level].discard(name)
        
        for tag in tool.definition.tags:
            self.tags[tag].discard(name)
        
        # Remove tool
        del self.tools[name]
        
        logger.info(f"Unregistered tool: {name}")
        return True
    
    def clear(self):
        """Clear all tools from registry"""
        self.tools.clear()
        self.categories.clear()
        self.security_levels.clear()
        self.tags.clear()
        
        logger.info("Cleared all tools from registry")