"""
Model Context Protocol (MCP) Integration
========================================

PRSM's implementation of the Model Context Protocol for tool integration.

The Model Context Protocol (MCP) is a standardized protocol for connecting
AI applications with external tools and data sources. This module provides
a comprehensive MCP client implementation that allows PRSM to:

- Connect to MCP servers hosting tools and resources
- Discover available tools and their capabilities
- Execute tool calls with proper parameter validation
- Manage tool sessions and state
- Handle errors and rate limiting
- Provide security sandboxing for tool execution

Key Components:
- MCPClient: Core client for connecting to MCP servers
- ToolRegistry: Manages discovered tools and their metadata
- SessionManager: Handles tool sessions and state
- SecurityManager: Sandboxing and validation for tool execution
- ResourceManager: Access to external data sources via MCP

Usage:
    from prsm.core.integrations.mcp import MCPClient, ToolRegistry
    
    # Connect to an MCP server
    client = MCPClient("http://localhost:3000")
    await client.connect()
    
    # Discover available tools
    tools = await client.list_tools()
    
    # Execute a tool
    result = await client.call_tool("weather", {"location": "San Francisco"})
"""

from .client import MCPClient, MCPError, MCPConnectionError, MCPToolError
from .tools import ToolRegistry, Tool, ToolCall, ToolResult
from .session import SessionManager, MCPSession
from .security import MCPSecurityManager
from .resources import ResourceManager, MCPResource
from .models import (
    MCPServerInfo, MCPCapabilities, MCPProtocolVersion,
    ToolDefinition, ToolParameter, ResourceDefinition,
    MCPMessage, MCPRequest, MCPResponse, MCPNotification
)

__all__ = [
    "MCPClient",
    "MCPError", 
    "MCPConnectionError",
    "MCPToolError",
    "ToolRegistry",
    "Tool",
    "ToolCall", 
    "ToolResult",
    "SessionManager",
    "MCPSession",
    "MCPSecurityManager",
    "ResourceManager",
    "MCPResource",
    "MCPServerInfo",
    "MCPCapabilities", 
    "MCPProtocolVersion",
    "ToolDefinition",
    "ToolParameter",
    "ResourceDefinition",
    "MCPMessage",
    "MCPRequest",
    "MCPResponse", 
    "MCPNotification"
]