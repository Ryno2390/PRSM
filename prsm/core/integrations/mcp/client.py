"""
MCP Client Implementation
========================

Core client for connecting to and interacting with MCP (Model Context Protocol) servers.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from uuid import UUID, uuid4
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .models import (
    MCPServerInfo, MCPCapabilities, ToolDefinition, ResourceDefinition,
    MCPRequest, MCPResponse, MCPNotification, ToolCall, ToolResult,
    MCPSession, MCPProtocolVersion, MCPMethod, MCPIntegrationConfig
)
from .tools import ToolRegistry
from .session import SessionManager
from .security import MCPSecurityManager
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base MCP error"""
    pass


class MCPConnectionError(MCPError):
    """MCP connection error"""
    pass


class MCPToolError(MCPError):
    """MCP tool execution error"""
    pass


class MCPProtocolError(MCPError):
    """MCP protocol error"""
    pass


class MCPClient:
    """
    MCP (Model Context Protocol) client for connecting to external tool servers.
    
    Provides a comprehensive interface for:
    - Connecting to MCP servers via HTTP or WebSocket
    - Discovering available tools and resources
    - Executing tool calls with proper validation
    - Managing sessions and state
    - Security sandboxing and validation
    """
    
    def __init__(self, server_uri: str, config: Optional[MCPIntegrationConfig] = None):
        """
        Initialize MCP client
        
        Args:
            server_uri: URI of the MCP server (http:// or ws://)
            config: Optional integration configuration
        """
        self.server_uri = server_uri
        self.config = config or MCPIntegrationConfig()
        self.settings = get_settings()
        
        # Connection state
        self.connected = False
        self.connection = None
        self.server_info: Optional[MCPServerInfo] = None
        
        # Components
        self.tool_registry = ToolRegistry()
        self.session_manager = SessionManager()
        self.security_manager = MCPSecurityManager()
        
        # Protocol state
        self.next_request_id = 1
        self.pending_requests: Dict[Union[str, int], asyncio.Future] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "last_activity": None
        }
        
        logger.info(f"Initialized MCP client for {server_uri}")
    
    async def connect(self, timeout: int = 30) -> bool:
        """
        Connect to the MCP server
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to MCP server: {self.server_uri}")
            
            if self.server_uri.startswith("ws://") or self.server_uri.startswith("wss://"):
                await self._connect_websocket(timeout)
            else:
                await self._connect_http(timeout)
            
            # Initialize connection
            await self._initialize_connection()
            
            # Discover available tools
            await self._discover_tools()
            
            self.connected = True
            logger.info(f"Successfully connected to MCP server: {self.server_info.name if self.server_info else 'Unknown'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_uri}: {e}")
            raise MCPConnectionError(f"Connection failed: {e}")
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        try:
            if self.connection:
                if hasattr(self.connection, 'close'):
                    await self.connection.close()
                elif hasattr(self.connection, 'close_session'):
                    await self.connection.close_session()
            
            self.connected = False
            self.connection = None
            logger.info(f"Disconnected from MCP server: {self.server_uri}")
            
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
    
    async def list_tools(self) -> List[ToolDefinition]:
        """
        List all available tools from the MCP server
        
        Returns:
            List of available tool definitions
        """
        if not self.connected:
            raise MCPConnectionError("Not connected to MCP server")
        
        try:
            request = MCPRequest(
                id=self._next_request_id(),
                method=MCPMethod.LIST_TOOLS
            )
            
            response = await self._send_request(request)
            
            tools = []
            for tool_data in response.get("result", {}).get("tools", []):
                tool = ToolDefinition(**tool_data)
                tools.append(tool)
                self.tool_registry.register_tool(tool)
            
            logger.info(f"Discovered {len(tools)} tools from MCP server")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise MCPToolError(f"Failed to list tools: {e}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], 
                       user_id: str, session_id: Optional[UUID] = None,
                       timeout: Optional[int] = None) -> ToolResult:
        """
        Execute a tool call
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            user_id: ID of the user making the call
            session_id: Optional session ID for context
            timeout: Optional timeout override
            
        Returns:
            Tool execution result
        """
        if not self.connected:
            raise MCPConnectionError("Not connected to MCP server")
        
        # Validate tool exists
        tool_def = self.tool_registry.get_tool(tool_name)
        if not tool_def:
            raise MCPToolError(f"Tool '{tool_name}' not found")
        
        # Create tool call
        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=timeout or self.config.default_timeout
        )
        
        start_time = time.time()
        
        try:
            logger.info(f"Executing tool call: {tool_name} for user {user_id}")
            
            # Security validation
            security_result = await self.security_manager.validate_tool_call(tool_call, tool_def)
            if not security_result.approved:
                raise MCPToolError(f"Security validation failed: {security_result.reason}")
            
            # Create MCP request
            request = MCPRequest(
                id=self._next_request_id(),
                method=MCPMethod.CALL_TOOL,
                params={
                    "name": tool_name,
                    "arguments": parameters
                }
            )
            
            # Send request with timeout
            response_data = await asyncio.wait_for(
                self._send_request(request),
                timeout=tool_call.timeout_seconds
            )
            
            # Process response
            execution_time = time.time() - start_time
            
            if "error" in response_data:
                error_info = response_data["error"]
                result = ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_name,
                    success=False,
                    error=f"{error_info.get('message', 'Unknown error')} (code: {error_info.get('code', -1)})",
                    execution_time=execution_time
                )
                self.stats["failed_requests"] += 1
            else:
                result_data = response_data.get("result", {})
                result = ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_name,
                    success=True,
                    result=result_data.get("content"),
                    execution_time=execution_time,
                    output_logs=result_data.get("logs", []),
                    metadata=result_data.get("metadata", {})
                )
                self.stats["successful_requests"] += 1
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["total_response_time"] += execution_time
            self.stats["last_activity"] = time.time()
            
            logger.info(f"Tool call completed: {tool_name} ({'success' if result.success else 'failed'}) in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Tool call timed out: {tool_name} after {execution_time:.2f}s")
            self.stats["failed_requests"] += 1
            self.stats["total_requests"] += 1
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_name,
                success=False,
                error=f"Tool call timed out after {tool_call.timeout_seconds} seconds",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool call failed: {tool_name} - {e}")
            self.stats["failed_requests"] += 1
            self.stats["total_requests"] += 1
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def list_resources(self) -> List[ResourceDefinition]:
        """
        List available resources from the MCP server
        
        Returns:
            List of available resource definitions
        """
        if not self.connected:
            raise MCPConnectionError("Not connected to MCP server")
        
        try:
            request = MCPRequest(
                id=self._next_request_id(),
                method=MCPMethod.LIST_RESOURCES
            )
            
            response = await self._send_request(request)
            
            resources = []
            for resource_data in response.get("result", {}).get("resources", []):
                resource = ResourceDefinition(**resource_data)
                resources.append(resource)
            
            logger.info(f"Discovered {len(resources)} resources from MCP server")
            return resources
            
        except Exception as e:
            logger.error(f"Failed to list resources: {e}")
            raise MCPError(f"Failed to list resources: {e}")
    
    async def ping(self) -> bool:
        """
        Ping the MCP server to check connectivity
        
        Returns:
            True if server responds
        """
        try:
            request = MCPRequest(
                id=self._next_request_id(),
                method=MCPMethod.PING
            )
            
            start_time = time.time()
            await self._send_request(request, timeout=5)
            response_time = time.time() - start_time
            
            logger.debug(f"MCP server ping successful: {response_time:.3f}s")
            return True
            
        except Exception as e:
            logger.warning(f"MCP server ping failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        total_requests = self.stats["total_requests"]
        avg_response_time = (
            self.stats["total_response_time"] / max(total_requests, 1)
        )
        
        return {
            "server_uri": self.server_uri,
            "connected": self.connected,
            "server_info": self.server_info.dict() if self.server_info else None,
            "total_requests": total_requests,
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": (self.stats["successful_requests"] / max(total_requests, 1)) * 100,
            "average_response_time": avg_response_time,
            "available_tools": len(self.tool_registry.list_tools()),
            "last_activity": self.stats["last_activity"]
        }
    
    # Private methods
    
    async def _connect_http(self, timeout: int):
        """Connect via HTTP"""
        self.connection = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
        
        # Test connection
        async with self.connection.get(f"{self.server_uri}/health") as response:
            if response.status != 200:
                raise MCPConnectionError(f"HTTP connection failed: {response.status}")
    
    async def _connect_websocket(self, timeout: int):
        """Connect via WebSocket"""
        try:
            self.connection = await asyncio.wait_for(
                websockets.connect(self.server_uri),
                timeout=timeout
            )
            
            # Start message handler
            asyncio.create_task(self._websocket_message_handler())
            
        except Exception as e:
            raise MCPConnectionError(f"WebSocket connection failed: {e}")
    
    async def _initialize_connection(self):
        """Initialize the MCP connection"""
        request = MCPRequest(
            id=self._next_request_id(),
            method=MCPMethod.INITIALIZE,
            params={
                "protocolVersion": MCPProtocolVersion.V1_0,
                "capabilities": {
                    "tools": True,
                    "resources": True
                },
                "clientInfo": {
                    "name": "PRSM-MCP-Client",
                    "version": "1.0.0"
                }
            }
        )
        
        response = await self._send_request(request)
        
        if "result" in response:
            server_data = response["result"]
            self.server_info = MCPServerInfo(
                name=server_data.get("serverInfo", {}).get("name", "Unknown"),
                version=server_data.get("serverInfo", {}).get("version", "Unknown"),
                capabilities=MCPCapabilities(**server_data.get("capabilities", {}))
            )
    
    async def _discover_tools(self):
        """Discover and register available tools"""
        try:
            tools = await self.list_tools()
            logger.info(f"Registered {len(tools)} tools from MCP server")
        except Exception as e:
            logger.warning(f"Failed to discover tools: {e}")
    
    def _next_request_id(self) -> int:
        """Generate next request ID"""
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id
    
    async def _send_request(self, request: MCPRequest, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Send request and wait for response"""
        if isinstance(self.connection, aiohttp.ClientSession):
            return await self._send_http_request(request, timeout)
        else:
            return await self._send_websocket_request(request, timeout)
    
    async def _send_http_request(self, request: MCPRequest, timeout: Optional[int]) -> Dict[str, Any]:
        """Send HTTP request"""
        headers = {"Content-Type": "application/json"}
        timeout_obj = aiohttp.ClientTimeout(total=timeout or self.config.default_timeout)
        
        async with self.connection.post(
            f"{self.server_uri}/mcp",
            json=request.dict(),
            headers=headers,
            timeout=timeout_obj
        ) as response:
            if response.status != 200:
                raise MCPProtocolError(f"HTTP error: {response.status}")
            
            return await response.json()
    
    async def _send_websocket_request(self, request: MCPRequest, timeout: Optional[int]) -> Dict[str, Any]:
        """Send WebSocket request"""
        if not self.connection:
            raise MCPConnectionError("WebSocket not connected")
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request.id] = future
        
        try:
            # Send request
            await self.connection.send(json.dumps(request.dict()))
            
            # Wait for response
            response = await asyncio.wait_for(
                future,
                timeout=timeout or self.config.default_timeout
            )
            
            return response
            
        finally:
            # Clean up
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
    
    async def _websocket_message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.connection:
                try:
                    data = json.loads(message)
                    
                    # Handle response
                    if "id" in data and data["id"] in self.pending_requests:
                        future = self.pending_requests[data["id"]]
                        if not future.done():
                            future.set_result(data)
                    
                    # Handle notification
                    elif "method" in data and "id" not in data:
                        await self._handle_notification(data)
                        
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    
        except ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"WebSocket message handler error: {e}")
            self.connected = False
    
    async def _handle_notification(self, data: Dict[str, Any]):
        """Handle MCP notifications"""
        method = data.get("method")
        params = data.get("params", {})
        
        logger.debug(f"Received notification: {method}")
        
        # Handle specific notification types
        if method == "tools/updated":
            await self._discover_tools()
        elif method == "resources/updated":
            # Could refresh resource list
            pass