"""
PRSM SDK Tool Executor
Execute MCP (Model Context Protocol) tools through PRSM
"""

import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from .exceptions import ToolExecutionError, PRSMError

logger = structlog.get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of MCP tools"""
    DATA = "data"
    COMPUTATION = "computation"
    STORAGE = "storage"
    NETWORK = "network"
    AI = "ai"
    UTILITY = "utility"


class ToolInfo(BaseModel):
    """Information about an MCP tool"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: ToolCategory = Field(..., description="Tool category")
    parameters: Dict[str, Any] = Field(..., description="Parameter schema")
    cost_per_execution: float = Field(..., description="FTNS cost per execution")
    safety_level: str = Field(..., description="Required safety level")
    provider: str = Field(..., description="Tool provider")
    version: str = Field(..., description="Tool version")
    is_available: bool = Field(True, description="Currently available")
    avg_execution_time: float = Field(..., description="Average execution time in seconds")
    success_rate: float = Field(..., description="Success rate (0-1)")


class ToolExecutionRequest(BaseModel):
    """Request to execute a tool"""
    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    context: Optional[Dict[str, Any]] = Field(None, description="Execution context")
    timeout: int = Field(60, description="Execution timeout in seconds")
    safety_level: str = Field("moderate", description="Safety level for execution")


class ToolExecutionResult(BaseModel):
    """Result of tool execution"""
    execution_id: str = Field(..., description="Unique execution ID")
    tool_name: str = Field(..., description="Tool that was executed")
    result: Any = Field(..., description="Execution result")
    execution_time: float = Field(..., description="Execution time in seconds")
    ftns_cost: float = Field(..., description="FTNS cost for execution")
    safety_status: str = Field(..., description="Safety assessment")
    success: bool = Field(..., description="Execution success status")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(..., description="Execution timestamp")


class ToolSearchRequest(BaseModel):
    """Search request for tools"""
    query: Optional[str] = Field(None, description="Search query")
    category: Optional[ToolCategory] = Field(None, description="Filter by category")
    max_cost: Optional[float] = Field(None, description="Maximum cost per execution")
    min_success_rate: Optional[float] = Field(None, description="Minimum success rate")
    provider: Optional[str] = Field(None, description="Filter by provider")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")


class ToolSearchResult(BaseModel):
    """Search result containing tools"""
    tools: List[ToolInfo] = Field(default_factory=list, description="Found tools")
    total: int = Field(..., description="Total matching tools")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current limit")


class ToolExecutor:
    """
    Execute MCP tools through PRSM
    
    Provides methods for:
    - Discovering available tools
    - Executing tools
    - Managing tool permissions
    - Tracking tool usage
    """
    
    def __init__(self, client):
        """
        Initialize tool executor
        
        Args:
            client: PRSMClient instance for making API requests
        """
        self._client = client
    
    async def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        limit: int = 20
    ) -> List[ToolInfo]:
        """
        List available MCP tools
        
        Args:
            category: Filter by category
            limit: Maximum results
            
        Returns:
            List of ToolInfo objects
            
        Example:
            tools = await client.tools.list_tools()
            for tool in tools:
                print(f"{tool.name}: {tool.description}")
        """
        params = {"limit": limit}
        if category:
            params["category"] = category.value
        
        response = await self._client._request(
            "GET",
            "/tools",
            params=params
        )
        
        return [ToolInfo(**t) for t in response.get("tools", [])]
    
    async def search_tools(
        self,
        query: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        max_cost: Optional[float] = None,
        min_success_rate: Optional[float] = None,
        provider: Optional[str] = None,
        limit: int = 20
    ) -> ToolSearchResult:
        """
        Search for MCP tools
        
        Args:
            query: Search query string
            category: Filter by category
            max_cost: Maximum cost per execution
            min_success_rate: Minimum success rate
            provider: Filter by provider
            limit: Maximum results
            
        Returns:
            ToolSearchResult with matching tools
        """
        request = ToolSearchRequest(
            query=query,
            category=category,
            max_cost=max_cost,
            min_success_rate=min_success_rate,
            provider=provider,
            limit=limit
        )
        
        response = await self._client._request(
            "POST",
            "/tools/search",
            json_data=request.model_dump(exclude_none=True)
        )
        
        return ToolSearchResult(**response)
    
    async def get_tool(self, tool_name: str) -> ToolInfo:
        """
        Get detailed information about a specific tool
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolInfo with tool details
            
        Raises:
            ToolExecutionError: If tool doesn't exist
        """
        try:
            response = await self._client._request(
                "GET",
                f"/tools/{tool_name}"
            )
            return ToolInfo(**response)
        except PRSMError as e:
            if "not found" in str(e).lower():
                raise ToolExecutionError(tool_name, "Tool not found")
            raise
    
    async def execute(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        safety_level: str = "moderate"
    ) -> ToolExecutionResult:
        """
        Execute an MCP tool
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            context: Execution context
            timeout: Execution timeout in seconds
            safety_level: Safety level for execution
            
        Returns:
            ToolExecutionResult with execution details
            
        Raises:
            ToolExecutionError: If execution fails
            
        Example:
            result = await client.tools.execute(
                "web_search",
                parameters={"query": "latest AI research"},
                timeout=30
            )
            print(f"Result: {result.result}")
        """
        request = ToolExecutionRequest(
            tool_name=tool_name,
            parameters=parameters or {},
            context=context,
            timeout=timeout,
            safety_level=safety_level
        )
        
        try:
            response = await self._client._request(
                "POST",
                "/tools/execute",
                json_data=request.model_dump()
            )
            
            result = ToolExecutionResult(**response)
            
            if not result.success:
                raise ToolExecutionError(tool_name, result.error or "Execution failed")
            
            return result
            
        except PRSMError as e:
            raise ToolExecutionError(tool_name, str(e))
    
    async def validate_parameters(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Validate tool parameters without executing
        
        Args:
            tool_name: Tool name
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid
            
        Raises:
            ValidationError: If parameters are invalid
        """
        response = await self._client._request(
            "POST",
            f"/tools/{tool_name}/validate",
            json_data={"parameters": parameters}
        )
        
        return response.get("valid", False)
    
    async def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 50
    ) -> List[ToolExecutionResult]:
        """
        Get tool execution history
        
        Args:
            tool_name: Filter by tool name
            limit: Maximum results
            
        Returns:
            List of ToolExecutionResult objects
        """
        params = {"limit": limit}
        if tool_name:
            params["tool_name"] = tool_name
        
        response = await self._client._request(
            "GET",
            "/tools/history",
            params=params
        )
        
        return [ToolExecutionResult(**r) for r in response.get("executions", [])]
    
    async def estimate_cost(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Estimate FTNS cost for tool execution
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            
        Returns:
            Estimated cost in FTNS
        """
        response = await self._client._request(
            "POST",
            f"/tools/{tool_name}/estimate-cost",
            json_data={"parameters": parameters or {}}
        )
        
        return response.get("estimated_cost", 0.0)
    
    async def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """
        Get usage statistics for a tool
        
        Args:
            tool_name: Tool name
            
        Returns:
            Dictionary with usage statistics
        """
        response = await self._client._request(
            "GET",
            f"/tools/{tool_name}/stats"
        )
        
        return response