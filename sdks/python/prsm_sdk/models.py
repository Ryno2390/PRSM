"""
PRSM SDK Data Models
Pydantic models for PRSM API requests and responses
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """AI model providers supported by PRSM"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    PRSM_DISTILLED = "prsm_distilled"
    PRSM = "prsm"


class SafetyLevel(str, Enum):
    """Safety monitoring levels"""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class QueryRequest(BaseModel):
    """Request for AI query execution"""
    prompt: str = Field(..., description="The query/prompt to process")
    model_id: Optional[str] = Field(None, description="Specific model to use")
    max_tokens: int = Field(1000, description="Maximum tokens in response")
    temperature: float = Field(0.7, description="Response randomness (0.0-1.0)")
    system_prompt: Optional[str] = Field(None, description="System instruction")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    tools: Optional[List[str]] = Field(None, description="MCP tools to enable")
    safety_level: SafetyLevel = Field(SafetyLevel.MODERATE, description="Safety monitoring level")


class PRSMResponse(BaseModel):
    """Response from PRSM AI query"""
    content: str = Field(..., description="AI-generated response content")
    model_id: str = Field(..., description="Model used for generation")
    provider: ModelProvider = Field(..., description="Provider of the model")
    execution_time: float = Field(..., description="Time taken in seconds")
    token_usage: Dict[str, int] = Field(..., description="Token consumption details")
    ftns_cost: float = Field(..., description="Cost in FTNS tokens")
    reasoning_trace: Optional[List[str]] = Field(None, description="Step-by-step reasoning")
    safety_status: SafetyLevel = Field(..., description="Safety assessment level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")


class FTNSBalance(BaseModel):
    """FTNS token balance information"""
    total_balance: float = Field(..., description="Total FTNS balance")
    available_balance: float = Field(..., description="Available for spending")
    reserved_balance: float = Field(..., description="Reserved for pending operations")
    earned_today: float = Field(0.0, description="FTNS earned today")
    spent_today: float = Field(0.0, description="FTNS spent today")
    last_updated: datetime = Field(..., description="Last balance update")


class ModelInfo(BaseModel):
    """Information about available AI models"""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable name")
    provider: ModelProvider = Field(..., description="Model provider")
    description: str = Field(..., description="Model description")
    capabilities: List[str] = Field(..., description="Model capabilities")
    cost_per_token: float = Field(..., description="FTNS cost per token")
    max_tokens: int = Field(..., description="Maximum token limit")
    context_window: int = Field(..., description="Context window size")
    is_available: bool = Field(..., description="Currently available")
    performance_rating: float = Field(..., description="Performance score (0-1)")
    safety_rating: float = Field(..., description="Safety score (0-1)")
    created_at: datetime = Field(..., description="Model creation date")


class ToolSpec(BaseModel):
    """MCP tool specification"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters schema")
    cost_per_execution: float = Field(..., description="FTNS cost per execution")
    safety_level: SafetyLevel = Field(..., description="Required safety level")
    provider: str = Field(..., description="Tool provider")
    version: str = Field(..., description="Tool version")


class SafetyStatus(BaseModel):
    """Current safety monitoring status"""
    overall_status: SafetyLevel = Field(..., description="Overall safety level")
    active_monitors: int = Field(..., description="Number of active monitors")
    threats_detected: int = Field(0, description="Threats detected today")
    circuit_breakers_triggered: int = Field(0, description="Circuit breakers triggered")
    last_assessment: datetime = Field(..., description="Last safety assessment")
    network_health: float = Field(..., description="Network health score (0-1)")


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    request_id: Optional[str] = Field(None, description="Associated request ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MarketplaceQuery(BaseModel):
    """Query for marketplace model search"""
    query: str = Field(..., description="Search query")
    provider: Optional[ModelProvider] = Field(None, description="Filter by provider")
    max_cost: Optional[float] = Field(None, description="Maximum cost per token")
    min_performance: Optional[float] = Field(None, description="Minimum performance rating")
    capabilities: Optional[List[str]] = Field(None, description="Required capabilities")
    limit: int = Field(20, description="Maximum results to return")


class ToolExecutionRequest(BaseModel):
    """Request for MCP tool execution"""
    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    context: Optional[Dict[str, Any]] = Field(None, description="Execution context")
    safety_level: SafetyLevel = Field(SafetyLevel.MODERATE, description="Safety level")


class ToolExecutionResponse(BaseModel):
    """Response from MCP tool execution"""
    result: Any = Field(..., description="Tool execution result")
    execution_time: float = Field(..., description="Execution time in seconds")
    ftns_cost: float = Field(..., description="FTNS cost for execution")
    safety_status: SafetyLevel = Field(..., description="Safety assessment")
    success: bool = Field(..., description="Execution success status")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")