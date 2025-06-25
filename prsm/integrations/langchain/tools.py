"""
PRSM LangChain Agent Tools
=========================

Collection of LangChain-compatible tools that leverage PRSM's capabilities.
These tools can be used with LangChain agents to provide AI-powered functionality.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field

try:
    from langchain.tools import BaseTool
    from langchain.tools.base import ToolException
    from langchain.callbacks.manager import (
        CallbackManagerForToolRun,
        AsyncCallbackManagerForToolRun,
    )
except ImportError:
    raise ImportError(
        "LangChain is required for PRSM LangChain integration. "
        "Install with: pip install langchain"
    )

from prsm_sdk import PRSMClient, PRSMError
from prsm.core.exceptions import PRSMException
from prsm.integrations.mcp import MCPClient

logger = logging.getLogger(__name__)


class PRSMQueryInput(BaseModel):
    """Input schema for PRSM query tool."""
    prompt: str = Field(description="The question or prompt to send to PRSM")
    context_allocation: Optional[int] = Field(
        default=50, description="FTNS tokens to allocate (10-500)"
    )
    user_id: Optional[str] = Field(
        default="langchain-agent", description="User ID for the query"
    )


class PRSMQueryTool(BaseTool):
    """
    LangChain tool for querying PRSM.
    
    This tool allows LangChain agents to use PRSM for complex reasoning,
    scientific modeling, and AI-powered analysis.
    
    Example:
        tool = PRSMQueryTool(
            prsm_client=PRSMClient("http://localhost:8000", "api-key")
        )
        
        result = tool.run({
            "prompt": "Analyze the implications of quantum computing on cryptography",
            "context_allocation": 100
        })
    """
    
    name: str = "prsm_query"
    description: str = (
        "Query PRSM for complex reasoning, scientific analysis, and AI-powered insights. "
        "Input should be a clear question or analysis request. "
        "Use this for: scientific questions, technical analysis, complex reasoning, "
        "research queries, and when you need advanced AI capabilities."
    )
    args_schema: Type[BaseModel] = PRSMQueryInput
    
    prsm_client: PRSMClient = Field(..., description="PRSM client instance")
    default_user_id: str = Field(default="langchain-agent", description="Default user ID")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
    
    def _run(
        self,
        prompt: str,
        context_allocation: Optional[int] = 50,
        user_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Synchronously run the PRSM query tool.
        
        Args:
            prompt: The question or prompt to send to PRSM
            context_allocation: FTNS tokens to allocate
            user_id: User ID for the query
            run_manager: Callback manager for handling callbacks
        
        Returns:
            The response from PRSM
        """
        try:
            # Use provided user_id or default
            actual_user_id = user_id or self.default_user_id
            
            # Run async query in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    self.prsm_client.query(
                        prompt=prompt,
                        user_id=actual_user_id,
                        context_allocation=context_allocation or 50
                    )
                )
                
                # Log usage information
                if run_manager:
                    run_manager.on_text(
                        f"\nPRSM Query Details:\n"
                        f"- Cost: {response.ftns_charged} FTNS\n"
                        f"- Processing Time: {response.processing_time:.2f}s\n"
                        f"- Quality Score: {response.quality_score}/100\n"
                    )
                
                return response.final_answer
                
            finally:
                loop.close()
        
        except PRSMError as e:
            logger.error(f"PRSM query tool failed: {e}")
            raise ToolException(f"PRSM query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM query tool: {e}")
            raise ToolException(f"PRSM query tool error: {e}")
    
    async def _arun(
        self,
        prompt: str,
        context_allocation: Optional[int] = 50,
        user_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Asynchronously run the PRSM query tool.
        
        Args:
            prompt: The question or prompt to send to PRSM
            context_allocation: FTNS tokens to allocate
            user_id: User ID for the query
            run_manager: Async callback manager for handling callbacks
        
        Returns:
            The response from PRSM
        """
        try:
            # Use provided user_id or default
            actual_user_id = user_id or self.default_user_id
            
            response = await self.prsm_client.query(
                prompt=prompt,
                user_id=actual_user_id,
                context_allocation=context_allocation or 50
            )
            
            # Log usage information
            if run_manager:
                await run_manager.on_text(
                    f"\nPRSM Query Details:\n"
                    f"- Cost: {response.ftns_charged} FTNS\n"
                    f"- Processing Time: {response.processing_time:.2f}s\n"
                    f"- Quality Score: {response.quality_score}/100\n"
                )
            
            return response.final_answer
            
        except PRSMError as e:
            logger.error(f"PRSM async query tool failed: {e}")
            raise ToolException(f"PRSM async query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM async query tool: {e}")
            raise ToolException(f"PRSM async query tool error: {e}")


class PRSMAnalysisInput(BaseModel):
    """Input schema for PRSM analysis tool."""
    data: str = Field(description="Data or information to analyze")
    analysis_type: str = Field(
        description="Type of analysis: summary, technical, scientific, statistical, or custom"
    )
    depth: Optional[str] = Field(
        default="moderate", description="Analysis depth: shallow, moderate, or deep"
    )


class PRSMAnalysisTool(BaseTool):
    """
    LangChain tool for specialized data analysis using PRSM.
    
    This tool is optimized for analyzing data, documents, and information
    with different levels of depth and focus areas.
    """
    
    name: str = "prsm_analysis"
    description: str = (
        "Analyze data, documents, or information using PRSM's specialized analysis capabilities. "
        "Supports different analysis types: summary, technical, scientific, statistical. "
        "Use this for: data analysis, document review, research synthesis, "
        "technical evaluation, and structured analysis tasks."
    )
    args_schema: Type[BaseModel] = PRSMAnalysisInput
    
    prsm_client: PRSMClient = Field(..., description="PRSM client instance")
    default_user_id: str = Field(default="langchain-analyst", description="Default user ID")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
    
    def _build_analysis_prompt(self, data: str, analysis_type: str, depth: str) -> str:
        """
        Build specialized prompt for analysis.
        
        Args:
            data: Data to analyze
            analysis_type: Type of analysis
            depth: Analysis depth
        
        Returns:
            Formatted analysis prompt
        """
        depth_instructions = {
            "shallow": "Provide a brief overview and key points.",
            "moderate": "Provide detailed analysis with insights and implications.",
            "deep": "Provide comprehensive analysis with detailed reasoning, implications, and recommendations."
        }
        
        type_instructions = {
            "summary": "Summarize the key points, themes, and important information.",
            "technical": "Focus on technical aspects, methodologies, and implementation details.",
            "scientific": "Analyze from a scientific perspective with evidence evaluation and hypothesis testing.",
            "statistical": "Focus on data patterns, statistical significance, and quantitative insights.",
            "custom": "Provide comprehensive analysis considering all relevant aspects."
        }
        
        prompt = f"""
Please perform a {analysis_type} analysis of the following data with {depth} depth.

Analysis Instructions:
{type_instructions.get(analysis_type, type_instructions['custom'])}
{depth_instructions.get(depth, depth_instructions['moderate'])}

Data to Analyze:
{data}

Please provide your analysis:
"""
        
        return prompt
    
    def _run(
        self,
        data: str,
        analysis_type: str,
        depth: Optional[str] = "moderate",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Synchronously run the PRSM analysis tool.
        
        Args:
            data: Data or information to analyze
            analysis_type: Type of analysis to perform
            depth: Analysis depth
            run_manager: Callback manager for handling callbacks
        
        Returns:
            Analysis results from PRSM
        """
        try:
            # Build analysis prompt
            prompt = self._build_analysis_prompt(data, analysis_type, depth or "moderate")
            
            # Determine context allocation based on depth
            context_allocation = {
                "shallow": 30,
                "moderate": 75,
                "deep": 150
            }.get(depth or "moderate", 75)
            
            # Run async query in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    self.prsm_client.query(
                        prompt=prompt,
                        user_id=self.default_user_id,
                        context_allocation=context_allocation
                    )
                )
                
                return response.final_answer
                
            finally:
                loop.close()
        
        except PRSMError as e:
            logger.error(f"PRSM analysis tool failed: {e}")
            raise ToolException(f"PRSM analysis failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM analysis tool: {e}")
            raise ToolException(f"PRSM analysis tool error: {e}")
    
    async def _arun(
        self,
        data: str,
        analysis_type: str,
        depth: Optional[str] = "moderate",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Asynchronously run the PRSM analysis tool.
        
        Args:
            data: Data or information to analyze
            analysis_type: Type of analysis to perform
            depth: Analysis depth
            run_manager: Async callback manager for handling callbacks
        
        Returns:
            Analysis results from PRSM
        """
        try:
            # Build analysis prompt
            prompt = self._build_analysis_prompt(data, analysis_type, depth or "moderate")
            
            # Determine context allocation based on depth
            context_allocation = {
                "shallow": 30,
                "moderate": 75,
                "deep": 150
            }.get(depth or "moderate", 75)
            
            response = await self.prsm_client.query(
                prompt=prompt,
                user_id=self.default_user_id,
                context_allocation=context_allocation
            )
            
            return response.final_answer
            
        except PRSMError as e:
            logger.error(f"PRSM async analysis tool failed: {e}")
            raise ToolException(f"PRSM async analysis failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM async analysis tool: {e}")
            raise ToolException(f"PRSM async analysis tool error: {e}")


class PRSMMCPToolInput(BaseModel):
    """Input schema for PRSM MCP tool."""
    tool_name: str = Field(description="Name of the MCP tool to execute")
    parameters: Dict[str, Any] = Field(description="Parameters for the MCP tool")
    user_id: Optional[str] = Field(
        default="langchain-mcp", description="User ID for the tool execution"
    )


class PRSMMCPTool(BaseTool):
    """
    LangChain tool for executing MCP tools through PRSM.
    
    This tool allows LangChain agents to use external tools and services
    through PRSM's Model Context Protocol integration.
    """
    
    name: str = "prsm_mcp_tool"
    description: str = (
        "Execute external tools and services through PRSM's MCP integration. "
        "Supports code execution, data processing, API calls, and specialized tools. "
        "Use this for: code formatting, data analysis, web requests, "
        "file operations, and external service integration."
    )
    args_schema: Type[BaseModel] = PRSMMCPToolInput
    
    mcp_client: MCPClient = Field(..., description="PRSM MCP client instance")
    default_user_id: str = Field(default="langchain-mcp", description="Default user ID")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
    
    def _run(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Synchronously run the PRSM MCP tool.
        
        Args:
            tool_name: Name of the MCP tool to execute
            parameters: Parameters for the tool
            user_id: User ID for the execution
            run_manager: Callback manager for handling callbacks
        
        Returns:
            Results from the MCP tool execution
        """
        try:
            # Use provided user_id or default
            actual_user_id = user_id or self.default_user_id
            
            # Run async tool execution in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.mcp_client.call_tool(
                        tool_name=tool_name,
                        parameters=parameters,
                        user_id=actual_user_id
                    )
                )
                
                if result.success:
                    # Log execution information
                    if run_manager:
                        run_manager.on_text(
                            f"\nMCP Tool Execution:\n"
                            f"- Tool: {tool_name}\n"
                            f"- Execution Time: {result.execution_time:.2f}s\n"
                            f"- Status: Success\n"
                        )
                    
                    return str(result.result)
                else:
                    raise ToolException(f"MCP tool execution failed: {result.error}")
                
            finally:
                loop.close()
        
        except Exception as e:
            logger.error(f"PRSM MCP tool failed: {e}")
            raise ToolException(f"MCP tool execution failed: {e}")
    
    async def _arun(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Asynchronously run the PRSM MCP tool.
        
        Args:
            tool_name: Name of the MCP tool to execute
            parameters: Parameters for the tool
            user_id: User ID for the execution
            run_manager: Async callback manager for handling callbacks
        
        Returns:
            Results from the MCP tool execution
        """
        try:
            # Use provided user_id or default
            actual_user_id = user_id or self.default_user_id
            
            result = await self.mcp_client.call_tool(
                tool_name=tool_name,
                parameters=parameters,
                user_id=actual_user_id
            )
            
            if result.success:
                # Log execution information
                if run_manager:
                    await run_manager.on_text(
                        f"\nMCP Tool Execution:\n"
                        f"- Tool: {tool_name}\n"
                        f"- Execution Time: {result.execution_time:.2f}s\n"
                        f"- Status: Success\n"
                    )
                
                return str(result.result)
            else:
                raise ToolException(f"MCP tool execution failed: {result.error}")
            
        except Exception as e:
            logger.error(f"PRSM async MCP tool failed: {e}")
            raise ToolException(f"Async MCP tool execution failed: {e}")


class PRSMAgentTools:
    """
    Collection of PRSM-powered tools for LangChain agents.
    
    This class provides a convenient way to create and manage multiple
    PRSM tools for use in LangChain agent workflows.
    
    Example:
        prsm_client = PRSMClient("http://localhost:8000", "api-key")
        mcp_client = MCPClient("http://localhost:3000")
        
        tools_manager = PRSMAgentTools(
            prsm_client=prsm_client,
            mcp_client=mcp_client
        )
        
        # Get all tools
        tools = tools_manager.get_tools()
        
        # Use with LangChain agent
        from langchain.agents import initialize_agent
        agent = initialize_agent(
            tools,
            llm,
            agent="zero-shot-react-description",
            verbose=True
        )
    """
    
    def __init__(
        self,
        prsm_client: PRSMClient,
        mcp_client: Optional[MCPClient] = None,
        default_user_id: str = "langchain-agent"
    ):
        """
        Initialize PRSM agent tools.
        
        Args:
            prsm_client: PRSM client instance
            mcp_client: Optional MCP client for external tools
            default_user_id: Default user ID for tool executions
        """
        self.prsm_client = prsm_client
        self.mcp_client = mcp_client
        self.default_user_id = default_user_id
        
        # Initialize tools
        self._tools = self._create_tools()
    
    def _create_tools(self) -> List[BaseTool]:
        """
        Create the collection of PRSM tools.
        
        Returns:
            List of initialized PRSM tools
        """
        tools = []
        
        # Always include basic PRSM query tool
        tools.append(
            PRSMQueryTool(
                prsm_client=self.prsm_client,
                default_user_id=self.default_user_id
            )
        )
        
        # Add analysis tool
        tools.append(
            PRSMAnalysisTool(
                prsm_client=self.prsm_client,
                default_user_id=self.default_user_id
            )
        )
        
        # Add MCP tool if client available
        if self.mcp_client:
            tools.append(
                PRSMMCPTool(
                    mcp_client=self.mcp_client,
                    default_user_id=self.default_user_id
                )
            )
        
        logger.info(f"Initialized {len(tools)} PRSM tools for LangChain")
        return tools
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all available PRSM tools.
        
        Returns:
            List of PRSM tools
        """
        return self._tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool if found, None otherwise
        """
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
    
    def list_tool_names(self) -> List[str]:
        """
        Get list of all tool names.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self._tools]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all tools.
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {tool.name: tool.description for tool in self._tools}
