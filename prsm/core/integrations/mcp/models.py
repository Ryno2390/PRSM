"""
MCP Protocol Models
==================

Data models and types for the Model Context Protocol implementation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class MCPProtocolVersion(str, Enum):
    """Supported MCP protocol versions"""
    V1_0 = "1.0"
    V2_0 = "2.0"


class MCPMessageType(str, Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response" 
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPMethod(str, Enum):
    """Standard MCP methods"""
    # Server management
    INITIALIZE = "initialize"
    PING = "ping"
    
    # Tool management
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    
    # Resource management
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    
    # Session management
    CREATE_SESSION = "session/create"
    UPDATE_SESSION = "session/update"
    CLOSE_SESSION = "session/close"


class ToolParameterType(str, Enum):
    """Tool parameter types"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class SecurityLevel(str, Enum):
    """Tool security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Core Protocol Models

class MCPMessage(BaseModel):
    """Base MCP message"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    
    class Config:
        extra = "allow"


class MCPRequest(MCPMessage):
    """MCP request message"""
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(MCPMessage):
    """MCP response message"""
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPNotification(MCPMessage):
    """MCP notification message (no response expected)"""
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPError(BaseModel):
    """MCP error information"""
    code: int
    message: str
    data: Optional[Any] = None


# Server and Capability Models

class MCPCapabilities(BaseModel):
    """MCP server capabilities"""
    tools: bool = False
    resources: bool = False
    prompts: bool = False
    sampling: bool = False
    experimental: Dict[str, bool] = Field(default_factory=dict)


class MCPServerInfo(BaseModel):
    """MCP server information"""
    name: str
    version: str
    protocol_version: MCPProtocolVersion = MCPProtocolVersion.V1_0
    capabilities: MCPCapabilities
    description: Optional[str] = None
    homepage: Optional[str] = None
    contact: Optional[Dict[str, str]] = None


# Tool Models

class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str
    type: ToolParameterType
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    
    class Config:
        extra = "allow"


class ToolDefinition(BaseModel):
    """Tool definition from MCP server"""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    category: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        extra = "allow"


class ToolCall(BaseModel):
    """Tool execution request"""
    call_id: UUID = Field(default_factory=uuid4)
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[UUID] = None
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    timeout_seconds: int = 30
    security_context: Optional[Dict[str, Any]] = None


class ToolResult(BaseModel):
    """Tool execution result"""
    call_id: UUID
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    output_logs: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.utcnow)


# Resource Models

class ResourceDefinition(BaseModel):
    """Resource definition from MCP server"""
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None
    size: Optional[int] = None
    modified: Optional[datetime] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)


class MCPResource(BaseModel):
    """MCP resource with content"""
    definition: ResourceDefinition
    content: Optional[Any] = None
    encoding: str = "utf-8"
    cached_at: Optional[datetime] = None
    cache_expires: Optional[datetime] = None


# Session Models

class MCPSession(BaseModel):
    """MCP session state"""
    session_id: UUID = Field(default_factory=uuid4)
    server_uri: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[UUID] = Field(default_factory=list)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()


# Integration Models

class MCPServerConnection(BaseModel):
    """MCP server connection configuration"""
    name: str
    uri: str
    auth_token: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)
    security_settings: Dict[str, Any] = Field(default_factory=dict)


class MCPIntegrationConfig(BaseModel):
    """MCP integration configuration"""
    enabled: bool = True
    max_concurrent_calls: int = 10
    default_timeout: int = 30
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    sandbox_enabled: bool = True
    rate_limit_per_minute: int = 60
    cache_ttl_seconds: int = 300
    servers: List[MCPServerConnection] = Field(default_factory=list)


# Statistics and Monitoring

class MCPStatistics(BaseModel):
    """MCP integration statistics"""
    total_servers: int = 0
    active_servers: int = 0
    total_tools: int = 0
    total_tool_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    last_reset: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_tool_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_tool_calls) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        return 100.0 - self.success_rate