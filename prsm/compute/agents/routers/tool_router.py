"""
MCP Tool Router Agent
Advanced routing for connecting models with appropriate MCP tools

This module implements the Tool Router layer, a crucial component for integrating
Model Context Protocol (MCP) tools into PRSM's agent framework. It serves as the
bridge between distilled models and external tools, enabling sophisticated
tool-augmented AI workflows.

Core Functionality:
- Tool discovery and matching based on task requirements
- MCP protocol integration for standardized tool communication
- Security validation and sandboxing for tool execution
- Performance tracking and optimization for tool usage
- Tool marketplace integration for dynamic tool discovery

Architecture Integration:
The Tool Router extends PRSM's existing 5-layer agent framework:
Prompter → Orchestrator → Router → [Tool Router] → Distilled Model + Tools → Compiler

This creates powerful recursive capabilities where:
- Models can request tools during execution
- Tools can invoke other models recursively  
- Complex multi-step workflows become possible
- Real-world data access enhances AI capabilities
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.compute.agents.base import BaseAgent
from prsm.core.config import get_settings
from prsm.core.models import AgentType, ArchitectTask, TimestampMixin

logger = structlog.get_logger(__name__)
settings = get_settings()


class ToolType(str, Enum):
    """Types of MCP tools available"""
    DATA_ACCESS = "data_access"           # Database queries, API calls, file reading
    COMPUTATION = "computation"           # Mathematical calculations, simulations
    COMMUNICATION = "communication"       # Email, messaging, notifications
    FILE_SYSTEM = "file_system"          # File operations, directory management
    WEB_INTERACTION = "web_interaction"   # Web scraping, browser automation
    SCIENTIFIC = "scientific"             # Lab instruments, analysis tools
    MULTIMEDIA = "multimedia"             # Image/video processing, generation
    BLOCKCHAIN = "blockchain"             # Web3 operations, smart contracts
    SYSTEM = "system"                     # OS operations, process management
    CUSTOM = "custom"                     # User-defined tools


class ToolCapability(str, Enum):
    """Specific capabilities that tools can provide"""
    READ = "read"                         # Read data or files
    WRITE = "write"                       # Write data or files
    EXECUTE = "execute"                   # Execute commands or scripts
    ANALYZE = "analyze"                   # Analyze data or content
    TRANSFORM = "transform"               # Transform data format
    VALIDATE = "validate"                 # Validate data or operations
    MONITOR = "monitor"                   # Monitor systems or processes
    NOTIFY = "notify"                     # Send notifications or alerts


class ToolSecurityLevel(str, Enum):
    """Security levels for tool execution"""
    SAFE = "safe"                         # Safe tools with no side effects
    RESTRICTED = "restricted"             # Limited tools with controlled access
    PRIVILEGED = "privileged"             # Powerful tools requiring permissions
    DANGEROUS = "dangerous"               # Potentially harmful tools


class MCPToolSpec(BaseModel):
    """Specification for an MCP tool"""
    tool_id: str
    name: str
    description: str
    tool_type: ToolType
    capabilities: List[ToolCapability]
    security_level: ToolSecurityLevel
    
    # MCP protocol details
    mcp_server_url: str
    tool_schema: Dict[str, Any]
    
    # Metadata
    version: str = "1.0.0"
    provider: str = "unknown"
    cost_per_use: Optional[float] = None
    rate_limit: Optional[int] = None
    requires_auth: bool = False
    
    # Performance data
    average_latency: float = 0.0
    success_rate: float = 1.0
    popularity_score: float = 0.0
    
    # Dependencies and requirements
    required_permissions: List[str] = Field(default_factory=list)
    compatible_models: List[str] = Field(default_factory=list)
    operating_systems: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolRequest(BaseModel):
    """Request for tool assistance from a model"""
    request_id: UUID = Field(default_factory=uuid4)
    model_id: str
    task_description: str
    
    # Tool requirements
    required_tool_types: List[ToolType] = Field(default_factory=list)
    required_capabilities: List[ToolCapability] = Field(default_factory=list)
    max_security_level: ToolSecurityLevel = ToolSecurityLevel.RESTRICTED
    
    # Constraints
    max_latency: Optional[float] = None
    max_cost: Optional[float] = None
    preferred_providers: List[str] = Field(default_factory=list)
    
    # Context
    task_context: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCandidate(BaseModel):
    """Candidate tool for a specific request"""
    tool_spec: MCPToolSpec
    compatibility_score: float = Field(ge=0.0, le=1.0)
    security_score: float = Field(ge=0.0, le=1.0)
    performance_score: float = Field(ge=0.0, le=1.0)
    cost_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Execution context
    estimated_latency: float = 0.0
    estimated_cost: float = 0.0
    security_constraints: List[str] = Field(default_factory=list)
    
    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall score based on component scores"""
        if weights is None:
            weights = {
                "compatibility": 0.4,
                "performance": 0.3,
                "security": 0.2,
                "cost": 0.1
            }
        
        score = (
            self.compatibility_score * weights.get("compatibility", 0.4) +
            self.performance_score * weights.get("performance", 0.3) +
            self.security_score * weights.get("security", 0.2) +
            self.cost_score * weights.get("cost", 0.1)
        )
        
        self.overall_score = min(score, 1.0)
        return self.overall_score


class ToolRoutingDecision(TimestampMixin):
    """Complete tool routing decision"""
    decision_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    
    primary_tool: ToolCandidate
    backup_tools: List[ToolCandidate] = Field(default_factory=list)
    
    confidence_score: float = Field(ge=0.0, le=1.0)
    routing_time: float
    reasoning: str
    
    # Execution plan
    execution_order: List[str] = Field(default_factory=list)
    parallel_execution: bool = False
    fallback_strategy: str = "sequential"


class ToolExecutionRequest(BaseModel):
    """Request to execute a specific tool"""
    execution_id: UUID = Field(default_factory=uuid4)
    tool_id: str
    tool_action: str
    parameters: Dict[str, Any]
    
    # Security context
    user_id: str
    permissions: List[str] = Field(default_factory=list)
    sandbox_level: ToolSecurityLevel = ToolSecurityLevel.RESTRICTED
    
    # Execution constraints
    timeout_seconds: float = 30.0
    max_memory_mb: Optional[int] = None
    max_output_size: Optional[int] = None
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolExecutionResult(BaseModel):
    """Result from tool execution"""
    execution_id: UUID
    tool_id: str
    success: bool
    
    # Result data
    result_data: Any = None
    output_format: str = "json"
    
    # Execution metadata
    execution_time: float
    memory_used: Optional[int] = None
    tokens_consumed: Optional[int] = None
    cost_incurred: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    
    # Security audit
    security_violations: List[str] = Field(default_factory=list)
    permissions_used: List[str] = Field(default_factory=list)
    
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MCPToolRegistry:
    """Registry of available MCP tools"""
    
    def __init__(self):
        self.tools: Dict[str, MCPToolSpec] = {}
        self.tool_providers: Dict[str, List[str]] = {}  # provider -> tool_ids
        self.tool_types_index: Dict[ToolType, List[str]] = {}  # type -> tool_ids
        self.capability_index: Dict[ToolCapability, List[str]] = {}  # capability -> tool_ids
        
        # Initialize with built-in tools
        self._register_builtin_tools()
    
    def register_tool(self, tool_spec: MCPToolSpec):
        """Register a new MCP tool"""
        self.tools[tool_spec.tool_id] = tool_spec
        
        # Update provider index
        if tool_spec.provider not in self.tool_providers:
            self.tool_providers[tool_spec.provider] = []
        self.tool_providers[tool_spec.provider].append(tool_spec.tool_id)
        
        # Update type index
        if tool_spec.tool_type not in self.tool_types_index:
            self.tool_types_index[tool_spec.tool_type] = []
        self.tool_types_index[tool_spec.tool_type].append(tool_spec.tool_id)
        
        # Update capability index
        for capability in tool_spec.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(tool_spec.tool_id)
        
        logger.info("MCP tool registered",
                   tool_id=tool_spec.tool_id,
                   tool_type=tool_spec.tool_type.value,
                   provider=tool_spec.provider)
    
    def discover_tools(self, tool_types: List[ToolType] = None,
                      capabilities: List[ToolCapability] = None,
                      security_level: ToolSecurityLevel = None,
                      provider: str = None) -> List[MCPToolSpec]:
        """Discover tools matching criteria"""
        candidate_tool_ids = set(self.tools.keys())
        
        # Filter by tool types
        if tool_types:
            type_matches = set()
            for tool_type in tool_types:
                type_matches.update(self.tool_types_index.get(tool_type, []))
            candidate_tool_ids &= type_matches
        
        # Filter by capabilities
        if capabilities:
            capability_matches = set()
            for capability in capabilities:
                capability_matches.update(self.capability_index.get(capability, []))
            candidate_tool_ids &= capability_matches
        
        # Filter by provider
        if provider:
            provider_tools = set(self.tool_providers.get(provider, []))
            candidate_tool_ids &= provider_tools
        
        # Filter by security level
        if security_level:
            security_filtered = set()
            for tool_id in candidate_tool_ids:
                tool = self.tools[tool_id]
                if self._security_level_allows(tool.security_level, security_level):
                    security_filtered.add(tool_id)
            candidate_tool_ids = security_filtered
        
        return [self.tools[tool_id] for tool_id in candidate_tool_ids]
    
    def get_tool(self, tool_id: str) -> Optional[MCPToolSpec]:
        """Get tool by ID"""
        return self.tools.get(tool_id)
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "providers": len(self.tool_providers),
            "tool_types": {
                tool_type.value: len(tool_ids) 
                for tool_type, tool_ids in self.tool_types_index.items()
            },
            "security_levels": {
                level.value: len([t for t in self.tools.values() if t.security_level == level])
                for level in ToolSecurityLevel
            }
        }
    
    def _security_level_allows(self, tool_level: ToolSecurityLevel, 
                              max_level: ToolSecurityLevel) -> bool:
        """Check if tool security level is allowed"""
        level_hierarchy = {
            ToolSecurityLevel.SAFE: 0,
            ToolSecurityLevel.RESTRICTED: 1,
            ToolSecurityLevel.PRIVILEGED: 2,
            ToolSecurityLevel.DANGEROUS: 3
        }
        return level_hierarchy[tool_level] <= level_hierarchy[max_level]
    
    def _register_builtin_tools(self):
        """Register built-in MCP tools"""
        builtin_tools = [
            MCPToolSpec(
                tool_id="web_search",
                name="Web Search",
                description="Search the web for information using various search engines",
                tool_type=ToolType.WEB_INTERACTION,
                capabilities=[ToolCapability.READ, ToolCapability.ANALYZE],
                security_level=ToolSecurityLevel.SAFE,
                mcp_server_url="http://localhost:3000/mcp/web-search",
                tool_schema={
                    "name": "web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                },
                provider="builtin",
                average_latency=2.5,
                success_rate=0.95
            ),
            MCPToolSpec(
                tool_id="file_reader",
                name="File Reader",
                description="Read files from the local filesystem",
                tool_type=ToolType.FILE_SYSTEM,
                capabilities=[ToolCapability.READ],
                security_level=ToolSecurityLevel.RESTRICTED,
                mcp_server_url="http://localhost:3000/mcp/file-system",
                tool_schema={
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to file"},
                            "encoding": {"type": "string", "default": "utf-8"}
                        },
                        "required": ["file_path"]
                    }
                },
                provider="builtin",
                required_permissions=["file_read"],
                average_latency=0.1,
                success_rate=0.98
            ),
            MCPToolSpec(
                tool_id="python_executor",
                name="Python Code Executor",
                description="Execute Python code in a secure sandbox",
                tool_type=ToolType.COMPUTATION,
                capabilities=[ToolCapability.EXECUTE, ToolCapability.ANALYZE],
                security_level=ToolSecurityLevel.PRIVILEGED,
                mcp_server_url="http://localhost:3000/mcp/python",
                tool_schema={
                    "name": "execute_python",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"},
                            "timeout": {"type": "number", "default": 30}
                        },
                        "required": ["code"]
                    }
                },
                provider="builtin",
                required_permissions=["code_execution"],
                average_latency=1.0,
                success_rate=0.92
            ),
            MCPToolSpec(
                tool_id="database_query",
                name="Database Query Tool",
                description="Execute SQL queries on connected databases",
                tool_type=ToolType.DATA_ACCESS,
                capabilities=[ToolCapability.READ, ToolCapability.ANALYZE],
                security_level=ToolSecurityLevel.RESTRICTED,
                mcp_server_url="http://localhost:3000/mcp/database",
                tool_schema={
                    "name": "sql_query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query"},
                            "database": {"type": "string", "description": "Database name"}
                        },
                        "required": ["query"]
                    }
                },
                provider="builtin",
                required_permissions=["database_read"],
                average_latency=0.5,
                success_rate=0.96
            ),
            MCPToolSpec(
                tool_id="api_client",
                name="HTTP API Client",
                description="Make HTTP requests to external APIs",
                tool_type=ToolType.COMMUNICATION,
                capabilities=[ToolCapability.READ, ToolCapability.WRITE],
                security_level=ToolSecurityLevel.RESTRICTED,
                mcp_server_url="http://localhost:3000/mcp/http",
                tool_schema={
                    "name": "http_request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Request URL"},
                            "method": {"type": "string", "default": "GET"},
                            "headers": {"type": "object", "default": {}},
                            "body": {"type": "string", "description": "Request body"}
                        },
                        "required": ["url"]
                    }
                },
                provider="builtin",
                required_permissions=["network_access"],
                average_latency=1.5,
                success_rate=0.94
            )
        ]
        
        for tool in builtin_tools:
            self.register_tool(tool)


class ToolRouter(BaseAgent):
    """
    MCP Tool Router Agent
    
    Core component for integrating Model Context Protocol (MCP) tools into 
    PRSM's agent framework. Provides intelligent routing between models and 
    tools for enhanced AI capabilities.
    
    Key Responsibilities:
    - Match model requests to appropriate MCP tools
    - Validate security requirements and permissions
    - Optimize tool selection for performance and cost
    - Coordinate tool execution with proper sandboxing
    - Track tool usage and performance metrics
    
    Integration with PRSM:
    - Extends existing router framework with tool capabilities
    - Integrates with FTNS token system for tool usage costs
    - Connects to safety monitoring for secure tool execution
    - Provides recursive tool chaining for complex workflows
    
    MCP Protocol Support:
    - Standard MCP tool discovery and invocation
    - Tool schema validation and parameter mapping
    - Bidirectional communication with MCP servers
    - Error handling and graceful degradation
    """
    
    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id=agent_id, agent_type=AgentType.ROUTER)
        
        self.tool_registry = MCPToolRegistry()
        self.routing_decisions: List[ToolRoutingDecision] = []
        self.execution_cache: Dict[str, ToolExecutionResult] = {}
        
        # Performance tracking
        self.tool_performance: Dict[str, List[float]] = {}  # tool_id -> latencies
        self.tool_success_rates: Dict[str, float] = {}
        
        logger.info("Tool Router initialized",
                   agent_id=self.agent_id,
                   available_tools=len(self.tool_registry.tools))
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> ToolRoutingDecision:
        """
        Process tool routing request
        
        Args:
            input_data: ToolRequest or request data
            context: Optional routing context
            
        Returns:
            ToolRoutingDecision: Complete routing decision with selected tools
        """
        start_time = time.time()
        
        # Parse tool request
        if isinstance(input_data, ToolRequest):
            tool_request = input_data
        elif isinstance(input_data, dict):
            tool_request = ToolRequest(**input_data)
        else:
            # Convert string/other to tool request
            tool_request = ToolRequest(
                model_id=context.get("model_id", "unknown") if context else "unknown",
                task_description=str(input_data),
                required_tool_types=[ToolType.CUSTOM],
                required_capabilities=[ToolCapability.ANALYZE]
            )
        
        logger.info("Processing tool routing request",
                   agent_id=self.agent_id,
                   model_id=tool_request.model_id,
                   task=tool_request.task_description[:100],
                   required_types=len(tool_request.required_tool_types))
        
        # Discover candidate tools
        candidates = await self._discover_tool_candidates(tool_request)
        
        if not candidates:
            logger.warning("No tool candidates found", 
                          request_id=str(tool_request.request_id))
            # Return empty decision
            return ToolRoutingDecision(
                request_id=tool_request.request_id,
                primary_tool=ToolCandidate(
                    tool_spec=MCPToolSpec(
                        tool_id="no_tool",
                        name="No Tool Available",
                        description="No suitable tools found",
                        tool_type=ToolType.CUSTOM,
                        capabilities=[],
                        security_level=ToolSecurityLevel.SAFE,
                        mcp_server_url="",
                        tool_schema={}
                    ),
                    compatibility_score=0.0,
                    security_score=0.0,
                    performance_score=0.0,
                    cost_score=0.0
                ),
                backup_tools=[],
                confidence_score=0.0,
                routing_time=time.time() - start_time,
                reasoning="No suitable tools found for the request"
            )
        
        # Score and rank candidates
        for candidate in candidates:
            candidate.calculate_overall_score()
        
        candidates.sort(key=lambda c: c.overall_score, reverse=True)
        
        # Create routing decision
        decision = ToolRoutingDecision(
            request_id=tool_request.request_id,
            primary_tool=candidates[0],
            backup_tools=candidates[1:min(4, len(candidates))],
            confidence_score=self._calculate_confidence(candidates),
            routing_time=time.time() - start_time,
            reasoning=self._generate_reasoning(tool_request, candidates[0]),
            execution_order=[c.tool_spec.tool_id for c in candidates[:3]],
            parallel_execution=await self._can_execute_parallel(candidates[:3])
        )
        
        # Store decision
        self.routing_decisions.append(decision)
        
        logger.info("Tool routing completed",
                   agent_id=self.agent_id,
                   primary_tool=decision.primary_tool.tool_spec.tool_id,
                   candidates_found=len(candidates),
                   confidence=decision.confidence_score,
                   routing_time=f"{decision.routing_time:.3f}s")
        
        return decision
    
    async def _discover_tool_candidates(self, request: ToolRequest) -> List[ToolCandidate]:
        """Discover and evaluate tool candidates for a request"""
        # Discover tools based on explicit requirements
        tools = self.tool_registry.discover_tools(
            tool_types=request.required_tool_types,
            capabilities=request.required_capabilities,
            security_level=request.max_security_level
        )
        
        # If no explicit requirements, infer from task description
        if not tools and not request.required_tool_types:
            inferred_types, inferred_capabilities = await self._infer_tool_requirements(
                request.task_description
            )
            tools = self.tool_registry.discover_tools(
                tool_types=inferred_types,
                capabilities=inferred_capabilities,
                security_level=request.max_security_level
            )
        
        candidates = []
        for tool in tools:
            candidate = ToolCandidate(
                tool_spec=tool,
                compatibility_score=await self._calculate_compatibility(request, tool),
                security_score=await self._calculate_security_score(request, tool),
                performance_score=await self._calculate_performance_score(tool),
                cost_score=await self._calculate_cost_score(request, tool)
            )
            candidates.append(candidate)
        
        return candidates
    
    async def _infer_tool_requirements(self, task_description: str) -> tuple[List[ToolType], List[ToolCapability]]:
        """Infer tool requirements from task description"""
        task_lower = task_description.lower()
        
        tool_types = []
        capabilities = []
        
        # Data access patterns
        if any(word in task_lower for word in ["search", "find", "lookup", "query", "get"]):
            tool_types.append(ToolType.DATA_ACCESS)
            capabilities.append(ToolCapability.READ)
        
        if any(word in task_lower for word in ["web", "internet", "online", "google"]):
            tool_types.append(ToolType.WEB_INTERACTION)
            capabilities.append(ToolCapability.READ)
        
        # File operations
        if any(word in task_lower for word in ["file", "document", "read", "write", "save"]):
            tool_types.append(ToolType.FILE_SYSTEM)
            if any(word in task_lower for word in ["write", "save", "create"]):
                capabilities.append(ToolCapability.WRITE)
            else:
                capabilities.append(ToolCapability.READ)
        
        # Computation
        if any(word in task_lower for word in ["calculate", "compute", "analyze", "process", "run"]):
            tool_types.append(ToolType.COMPUTATION)
            capabilities.append(ToolCapability.EXECUTE)
            capabilities.append(ToolCapability.ANALYZE)
        
        # Communication
        if any(word in task_lower for word in ["email", "send", "notify", "message", "api"]):
            tool_types.append(ToolType.COMMUNICATION)
            capabilities.append(ToolCapability.WRITE)
            capabilities.append(ToolCapability.NOTIFY)
        
        # Scientific tools
        if any(word in task_lower for word in ["experiment", "lab", "scientific", "research", "data"]):
            tool_types.append(ToolType.SCIENTIFIC)
            capabilities.append(ToolCapability.ANALYZE)
        
        # Default fallback
        if not tool_types:
            tool_types.append(ToolType.CUSTOM)
            capabilities.append(ToolCapability.ANALYZE)
        
        return tool_types, capabilities
    
    async def _calculate_compatibility(self, request: ToolRequest, tool: MCPToolSpec) -> float:
        """Calculate compatibility score between request and tool"""
        score = 0.4  # Base compatibility
        
        # Type matching
        if request.required_tool_types:
            if tool.tool_type in request.required_tool_types:
                score += 0.3
        else:
            score += 0.1  # No specific type required
        
        # Capability matching
        if request.required_capabilities:
            matching_capabilities = set(tool.capabilities) & set(request.required_capabilities)
            score += 0.2 * (len(matching_capabilities) / len(request.required_capabilities))
        
        # Model compatibility
        if tool.compatible_models:
            if request.model_id in tool.compatible_models:
                score += 0.1
        else:
            score += 0.05  # Assume compatible if not specified
        
        return min(score, 1.0)
    
    async def _calculate_security_score(self, request: ToolRequest, tool: MCPToolSpec) -> float:
        """Calculate security score for tool usage"""
        # Base security score based on tool security level
        level_scores = {
            ToolSecurityLevel.SAFE: 1.0,
            ToolSecurityLevel.RESTRICTED: 0.8,
            ToolSecurityLevel.PRIVILEGED: 0.6,
            ToolSecurityLevel.DANGEROUS: 0.3
        }
        
        score = level_scores.get(tool.security_level, 0.5)
        
        # Check if tool security level is acceptable
        if not self.tool_registry._security_level_allows(tool.security_level, request.max_security_level):
            score = 0.0
        
        # Provider trust score
        trusted_providers = ["builtin", "verified", "official"]
        if tool.provider in trusted_providers:
            score *= 1.1
        
        return min(score, 1.0)
    
    async def _calculate_performance_score(self, tool: MCPToolSpec) -> float:
        """Calculate performance score for tool"""
        score = 0.5  # Base score
        
        # Success rate contribution
        score += tool.success_rate * 0.3
        
        # Latency contribution (lower is better)
        if tool.average_latency > 0:
            latency_score = max(0.1, 1.0 - (tool.average_latency / 10.0))
            score += latency_score * 0.2
        
        # Historical performance (if available)
        if tool.tool_id in self.tool_performance:
            recent_latencies = self.tool_performance[tool.tool_id][-10:]  # Last 10
            if recent_latencies:
                avg_latency = sum(recent_latencies) / len(recent_latencies)
                recent_score = max(0.1, 1.0 - (avg_latency / 10.0))
                score = (score + recent_score) / 2  # Average with historical
        
        return min(score, 1.0)
    
    async def _calculate_cost_score(self, request: ToolRequest, tool: MCPToolSpec) -> float:
        """Calculate cost score for tool usage"""
        if tool.cost_per_use is None:
            return 0.8  # Free tools get good score
        
        if request.max_cost is None:
            # No budget limit, just prefer cheaper tools
            return max(0.1, 1.0 - (tool.cost_per_use / 10.0))  # Normalize to $10 max
        
        if tool.cost_per_use > request.max_cost:
            return 0.0  # Too expensive
        
        # Linear score based on cost efficiency
        cost_ratio = tool.cost_per_use / request.max_cost
        return max(0.1, 1.0 - cost_ratio)
    
    def _calculate_confidence(self, candidates: List[ToolCandidate]) -> float:
        """Calculate confidence in tool routing decision"""
        if not candidates:
            return 0.0
        
        top_score = candidates[0].overall_score
        
        if len(candidates) == 1:
            return top_score
        
        # Consider score gap between top candidates
        second_score = candidates[1].overall_score
        score_gap = top_score - second_score
        
        # Higher confidence when there's a clear winner
        confidence = top_score * (0.7 + 0.3 * min(score_gap * 2, 1.0))
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, request: ToolRequest, candidate: ToolCandidate) -> str:
        """Generate human-readable reasoning for tool selection"""
        tool = candidate.tool_spec
        reasons = []
        
        reasons.append(f"Selected '{tool.name}' ({tool.tool_type.value})")
        
        if candidate.compatibility_score > 0.8:
            reasons.append("high compatibility with task requirements")
        elif candidate.compatibility_score > 0.6:
            reasons.append("good compatibility with task requirements")
        else:
            reasons.append("moderate compatibility with task requirements")
        
        if tool.security_level == ToolSecurityLevel.SAFE:
            reasons.append("safe execution environment")
        elif tool.security_level == ToolSecurityLevel.RESTRICTED:
            reasons.append("controlled security environment")
        
        if candidate.performance_score > 0.8:
            reasons.append(f"excellent performance (avg latency: {tool.average_latency:.1f}s)")
        elif candidate.performance_score > 0.6:
            reasons.append(f"good performance (avg latency: {tool.average_latency:.1f}s)")
        
        if tool.cost_per_use is None:
            reasons.append("no usage cost")
        elif tool.cost_per_use < 0.01:
            reasons.append("low cost")
        
        reasons.append(f"overall suitability: {candidate.overall_score:.0%}")
        
        return "; ".join(reasons)
    
    async def _can_execute_parallel(self, candidates: List[ToolCandidate]) -> bool:
        """Check if tools can be executed in parallel"""
        if len(candidates) < 2:
            return False
        
        # Check for conflicting operations
        write_operations = sum(1 for c in candidates 
                             if ToolCapability.WRITE in c.tool_spec.capabilities)
        
        # Don't parallel execute if multiple tools write to same resource type
        if write_operations > 1:
            return False
        
        # Check if tools are independent
        tool_types = set(c.tool_spec.tool_type for c in candidates)
        
        # Some tool types shouldn't run in parallel
        conflicting_types = {
            (ToolType.FILE_SYSTEM, ToolType.FILE_SYSTEM),
            (ToolType.SYSTEM, ToolType.SYSTEM)
        }
        
        for t1 in tool_types:
            for t2 in tool_types:
                if t1 != t2 and (t1, t2) in conflicting_types:
                    return False
        
        return True
    
    async def execute_tool(self, execution_request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute a tool with proper security and monitoring"""
        start_time = time.time()
        
        tool = self.tool_registry.get_tool(execution_request.tool_id)
        if not tool:
            return ToolExecutionResult(
                execution_id=execution_request.execution_id,
                tool_id=execution_request.tool_id,
                success=False,
                execution_time=0.0,
                error_message=f"Tool not found: {execution_request.tool_id}",
                error_code="TOOL_NOT_FOUND"
            )
        
        try:
            # Validate security permissions
            if not await self._validate_tool_permissions(execution_request, tool):
                return ToolExecutionResult(
                    execution_id=execution_request.execution_id,
                    tool_id=execution_request.tool_id,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Insufficient permissions for tool execution",
                    error_code="PERMISSION_DENIED",
                    security_violations=["Insufficient permissions"]
                )
            
            # Execute tool through MCP protocol
            result = await self._execute_mcp_tool(tool, execution_request)
            
            execution_time = time.time() - start_time
            
            # Update performance tracking
            if tool.tool_id not in self.tool_performance:
                self.tool_performance[tool.tool_id] = []
            self.tool_performance[tool.tool_id].append(execution_time)
            
            # Keep only recent performance data
            if len(self.tool_performance[tool.tool_id]) > 100:
                self.tool_performance[tool.tool_id] = self.tool_performance[tool.tool_id][-50:]
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Tool execution failed",
                        tool_id=execution_request.tool_id,
                        error=str(e))
            
            return ToolExecutionResult(
                execution_id=execution_request.execution_id,
                tool_id=execution_request.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    async def _validate_tool_permissions(self, request: ToolExecutionRequest, 
                                        tool: MCPToolSpec) -> bool:
        """Validate that user has permissions to execute tool"""
        # Check required permissions
        for required_perm in tool.required_permissions:
            if required_perm not in request.permissions:
                logger.warning("Missing required permission",
                              tool_id=tool.tool_id,
                              required=required_perm,
                              user_permissions=request.permissions)
                return False
        
        # Check security level compatibility
        if not self.tool_registry._security_level_allows(tool.security_level, request.sandbox_level):
            logger.warning("Tool security level not allowed",
                          tool_security=tool.security_level.value,
                          max_allowed=request.sandbox_level.value)
            return False
        
        return True
    
    async def _execute_mcp_tool(self, tool: MCPToolSpec, 
                               request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute tool through MCP protocol"""
        # For now, simulate MCP execution
        # In production, this would make actual MCP calls
        
        await asyncio.sleep(tool.average_latency)  # Simulate network latency
        
        # Simulate tool execution based on type
        if tool.tool_type == ToolType.WEB_INTERACTION:
            result_data = {
                "search_results": [
                    {"title": "Sample Result 1", "url": "https://example.com/1"},
                    {"title": "Sample Result 2", "url": "https://example.com/2"}
                ]
            }
        elif tool.tool_type == ToolType.FILE_SYSTEM:
            result_data = {
                "file_content": "Sample file content...",
                "file_size": 1024
            }
        elif tool.tool_type == ToolType.COMPUTATION:
            result_data = {
                "calculation_result": 42,
                "computation_time": 0.1
            }
        else:
            result_data = {
                "message": f"Tool {tool.tool_id} executed successfully",
                "parameters": request.parameters
            }
        
        return ToolExecutionResult(
            execution_id=request.execution_id,
            tool_id=tool.tool_id,
            success=True,
            result_data=result_data,
            execution_time=tool.average_latency,
            tokens_consumed=100,  # Simulate token usage
            cost_incurred=tool.cost_per_use,
            permissions_used=tool.required_permissions
        )
    
    def get_tool_analytics(self) -> Dict[str, Any]:
        """Get tool usage analytics"""
        return {
            "total_routing_decisions": len(self.routing_decisions),
            "available_tools": len(self.tool_registry.tools),
            "tool_performance": {
                tool_id: {
                    "avg_latency": sum(latencies) / len(latencies),
                    "usage_count": len(latencies)
                }
                for tool_id, latencies in self.tool_performance.items()
            },
            "tool_registry_stats": self.tool_registry.get_tool_stats(),
            "recent_decisions": [
                {
                    "tool_selected": d.primary_tool.tool_spec.tool_id,
                    "confidence": d.confidence_score,
                    "routing_time": d.routing_time
                }
                for d in self.routing_decisions[-10:]  # Last 10
            ]
        }


# Factory function
def create_tool_router() -> ToolRouter:
    """Create a tool router agent"""
    return ToolRouter()


# Global tool registry instance
tool_registry = MCPToolRegistry()