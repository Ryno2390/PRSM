"""
PRSM LangChain Chains
====================

Custom LangChain chains that leverage PRSM's capabilities for specialized workflows.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import Field

try:
    from langchain.chains.base import Chain
    from langchain.chains.conversation.base import ConversationChain
    from langchain.schema import BaseMemory
    from langchain.callbacks.manager import (
        CallbackManagerForChainRun,
        AsyncCallbackManagerForChainRun,
    )
except ImportError:
    raise ImportError(
        "LangChain is required for PRSM LangChain integration. "
        "Install with: pip install langchain"
    )

from prsm_sdk import PRSMClient, PRSMError
from prsm.core.exceptions import PRSMException
from .llm import PRSMLangChainLLM
from .memory import PRSMChatMemory

logger = logging.getLogger(__name__)


class PRSMChain(Chain):
    """
    Custom LangChain chain that uses PRSM for complex reasoning tasks.
    
    This chain provides specialized workflows for PRSM, including
    multi-step reasoning, analysis pipelines, and context-aware processing.
    
    Example:
        chain = PRSMChain(
            prsm_client=PRSMClient("http://localhost:8000", "api-key"),
            workflow_type="analysis"
        )
        
        result = chain.run({
            "input": "Analyze the market trends for renewable energy",
            "depth": "comprehensive"
        })
    """
    
    prsm_client: PRSMClient = Field(..., description="PRSM client instance")
    workflow_type: str = Field(default="general", description="Type of workflow to execute")
    default_user_id: str = Field(default="langchain-chain", description="Default user ID")
    context_allocation: int = Field(default=100, description="Default FTNS allocation")
    
    # Chain configuration
    input_key: str = "input"
    output_key: str = "output"
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
    
    @property
    def input_keys(self) -> List[str]:
        """Input keys expected by the chain."""
        return [self.input_key, "user_id", "context_allocation", "workflow_params"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys produced by the chain."""
        return [self.output_key, "metadata"]
    
    def _get_workflow_prompt(self, input_text: str, workflow_params: Dict[str, Any]) -> str:
        """
        Generate workflow-specific prompt based on workflow type.
        
        Args:
            input_text: Input text to process
            workflow_params: Additional workflow parameters
        
        Returns:
            Formatted prompt for the workflow
        """
        if self.workflow_type == "analysis":
            depth = workflow_params.get("depth", "moderate")
            focus = workflow_params.get("focus", "general")
            
            return f"""
Perform a {depth} analysis with focus on {focus} aspects.

Input to analyze:
{input_text}

Please provide a structured analysis including:
1. Key insights and findings
2. Important patterns or trends
3. Implications and recommendations
4. Supporting evidence and reasoning

Analysis:
"""
        
        elif self.workflow_type == "research":
            scope = workflow_params.get("scope", "comprehensive")
            methodology = workflow_params.get("methodology", "systematic")
            
            return f"""
Conduct {scope} research using a {methodology} approach.

Research question/topic:
{input_text}

Please provide:
1. Background and context
2. Key findings and evidence
3. Multiple perspectives and viewpoints
4. Gaps and limitations
5. Conclusions and future directions

Research output:
"""
        
        elif self.workflow_type == "problem_solving":
            approach = workflow_params.get("approach", "systematic")
            constraints = workflow_params.get("constraints", [])
            
            constraints_text = "\n".join([f"- {c}" for c in constraints]) if constraints else "None specified"
            
            return f"""
Solve the following problem using a {approach} approach.

Problem:
{input_text}

Constraints:
{constraints_text}

Please provide:
1. Problem analysis and breakdown
2. Potential solutions and approaches
3. Evaluation of alternatives
4. Recommended solution with rationale
5. Implementation considerations

Solution:
"""
        
        elif self.workflow_type == "synthesis":
            sources = workflow_params.get("sources", [])
            perspective = workflow_params.get("perspective", "balanced")
            
            return f"""
Synthesize information from multiple sources with a {perspective} perspective.

Topic:
{input_text}

Sources to consider:
{chr(10).join([f"- {s}" for s in sources]) if sources else "Use available knowledge"}

Please provide:
1. Integrated overview of the topic
2. Synthesis of key themes and patterns
3. Reconciliation of conflicting information
4. Comprehensive understanding
5. Novel insights from synthesis

Synthesis:
"""
        
        else:  # general workflow
            return f"""
Process the following input with comprehensive reasoning:

{input_text}

Please provide a thorough and well-reasoned response.
"""
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the PRSM chain synchronously.
        
        Args:
            inputs: Input dictionary containing required keys
            run_manager: Callback manager for handling callbacks
        
        Returns:
            Dictionary containing output and metadata
        """
        # Extract inputs
        input_text = inputs[self.input_key]
        user_id = inputs.get("user_id", self.default_user_id)
        context_allocation = inputs.get("context_allocation", self.context_allocation)
        workflow_params = inputs.get("workflow_params", {})
        
        try:
            # Generate workflow-specific prompt
            prompt = self._get_workflow_prompt(input_text, workflow_params)
            
            # Log workflow information
            if run_manager:
                run_manager.on_text(
                    f"\nPRSM Chain Workflow: {self.workflow_type}\n"
                    f"Context Allocation: {context_allocation} FTNS\n"
                )
            
            # Run async query in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    self.prsm_client.query(
                        prompt=prompt,
                        user_id=user_id,
                        context_allocation=context_allocation
                    )
                )
                
                # Log results
                if run_manager:
                    run_manager.on_text(
                        f"\nPRSM Chain Results:\n"
                        f"- Cost: {response.ftns_charged} FTNS\n"
                        f"- Processing Time: {response.processing_time:.2f}s\n"
                        f"- Quality Score: {response.quality_score}/100\n"
                    )
                
                return {
                    self.output_key: response.final_answer,
                    "metadata": {
                        "workflow_type": self.workflow_type,
                        "cost": response.ftns_charged,
                        "processing_time": response.processing_time,
                        "quality_score": response.quality_score,
                        "query_id": response.query_id,
                    }
                }
                
            finally:
                loop.close()
        
        except PRSMError as e:
            logger.error(f"PRSM chain execution failed: {e}")
            raise PRSMException(f"PRSM chain failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM chain: {e}")
            raise PRSMException(f"PRSM chain error: {e}")
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the PRSM chain asynchronously.
        
        Args:
            inputs: Input dictionary containing required keys
            run_manager: Async callback manager for handling callbacks
        
        Returns:
            Dictionary containing output and metadata
        """
        # Extract inputs
        input_text = inputs[self.input_key]
        user_id = inputs.get("user_id", self.default_user_id)
        context_allocation = inputs.get("context_allocation", self.context_allocation)
        workflow_params = inputs.get("workflow_params", {})
        
        try:
            # Generate workflow-specific prompt
            prompt = self._get_workflow_prompt(input_text, workflow_params)
            
            # Log workflow information
            if run_manager:
                await run_manager.on_text(
                    f"\nPRSM Chain Workflow: {self.workflow_type}\n"
                    f"Context Allocation: {context_allocation} FTNS\n"
                )
            
            response = await self.prsm_client.query(
                prompt=prompt,
                user_id=user_id,
                context_allocation=context_allocation
            )
            
            # Log results
            if run_manager:
                await run_manager.on_text(
                    f"\nPRSM Chain Results:\n"
                    f"- Cost: {response.ftns_charged} FTNS\n"
                    f"- Processing Time: {response.processing_time:.2f}s\n"
                    f"- Quality Score: {response.quality_score}/100\n"
                )
            
            return {
                self.output_key: response.final_answer,
                "metadata": {
                    "workflow_type": self.workflow_type,
                    "cost": response.ftns_charged,
                    "processing_time": response.processing_time,
                    "quality_score": response.quality_score,
                    "query_id": response.query_id,
                }
            }
            
        except PRSMError as e:
            logger.error(f"PRSM async chain execution failed: {e}")
            raise PRSMException(f"PRSM async chain failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM async chain: {e}")
            raise PRSMException(f"PRSM async chain error: {e}")
    
    @property
    def _chain_type(self) -> str:
        """Return the chain type identifier."""
        return f"prsm_{self.workflow_type}"


class PRSMConversationChain(ConversationChain):
    """
    PRSM-powered conversation chain with enhanced memory and context management.
    
    This chain extends LangChain's ConversationChain to use PRSM's conversational
    capabilities with session management and context preservation.
    
    Example:
        prsm_llm = PRSMLangChainLLM(
            base_url="http://localhost:8000",
            api_key="api-key"
        )
        
        memory = PRSMChatMemory(
            prsm_client=prsm_client,
            user_id="conversation-user"
        )
        
        conversation = PRSMConversationChain(
            llm=prsm_llm,
            memory=memory,
            conversation_type="research"
        )
        
        response = conversation.predict(input="Let's discuss quantum computing")
    """
    
    conversation_type: str = Field(default="general", description="Type of conversation")
    context_enhancement: bool = Field(default=True, description="Whether to enhance context")
    prsm_client: Optional[PRSMClient] = Field(default=None, description="PRSM client for enhancements")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        # Extract PRSM-specific parameters
        conversation_type = kwargs.pop("conversation_type", "general")
        context_enhancement = kwargs.pop("context_enhancement", True)
        prsm_client = kwargs.pop("prsm_client", None)
        
        # Initialize parent
        super().__init__(**kwargs)
        
        # Set PRSM-specific attributes
        self.conversation_type = conversation_type
        self.context_enhancement = context_enhancement
        self.prsm_client = prsm_client
    
    def _enhance_prompt_with_context(self, prompt: str) -> str:
        """
        Enhance the conversation prompt with PRSM-specific context.
        
        Args:
            prompt: Original prompt
        
        Returns:
            Enhanced prompt with additional context
        """
        if not self.context_enhancement:
            return prompt
        
        context_instructions = {
            "research": (
                "This is a research-focused conversation. Please provide detailed, "
                "evidence-based responses with citations when appropriate. "
                "Consider multiple perspectives and acknowledge uncertainties."
            ),
            "technical": (
                "This is a technical discussion. Please provide precise, "
                "technically accurate responses with implementation details "
                "and practical considerations."
            ),
            "educational": (
                "This is an educational conversation. Please provide clear, "
                "well-structured explanations that build understanding progressively. "
                "Use examples and analogies when helpful."
            ),
            "analytical": (
                "This is an analytical conversation. Please provide thorough analysis "
                "with logical reasoning, consideration of alternatives, "
                "and evidence-based conclusions."
            ),
            "creative": (
                "This is a creative conversation. Please provide imaginative, "
                "innovative responses while maintaining accuracy "
                "and practical relevance when appropriate."
            ),
        }
        
        context_instruction = context_instructions.get(
            self.conversation_type,
            "Please provide helpful, accurate, and comprehensive responses."
        )
        
        enhanced_prompt = f"""
Conversation Context: {context_instruction}

{prompt}

Please respond appropriately for this type of conversation:
"""
        
        return enhanced_prompt
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the PRSM conversation chain.
        
        Args:
            inputs: Input dictionary
            run_manager: Callback manager for handling callbacks
        
        Returns:
            Dictionary containing response and metadata
        """
        # Enhance prompt if enabled
        if self.context_enhancement and "input" in inputs:
            inputs["input"] = self._enhance_prompt_with_context(inputs["input"])
        
        # Log conversation information
        if run_manager:
            run_manager.on_text(
                f"\nPRSM Conversation Type: {self.conversation_type}\n"
                f"Context Enhancement: {self.context_enhancement}\n"
            )
        
        # Call parent implementation
        result = super()._call(inputs, run_manager)
        
        # Add PRSM-specific metadata if available
        if hasattr(self.llm, '_last_cost'):
            result["metadata"] = {
                "conversation_type": self.conversation_type,
                "cost": getattr(self.llm, '_last_cost', 0),
                "processing_time": getattr(self.llm, '_last_time', 0),
                "quality_score": getattr(self.llm, '_last_quality', 0),
            }
        
        return result
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the PRSM conversation chain asynchronously.
        
        Args:
            inputs: Input dictionary
            run_manager: Async callback manager for handling callbacks
        
        Returns:
            Dictionary containing response and metadata
        """
        # Enhance prompt if enabled
        if self.context_enhancement and "input" in inputs:
            inputs["input"] = self._enhance_prompt_with_context(inputs["input"])
        
        # Log conversation information
        if run_manager:
            await run_manager.on_text(
                f"\nPRSM Conversation Type: {self.conversation_type}\n"
                f"Context Enhancement: {self.context_enhancement}\n"
            )
        
        # Call parent implementation
        result = await super()._acall(inputs, run_manager)
        
        # Add PRSM-specific metadata if available
        if hasattr(self.llm, '_last_cost'):
            result["metadata"] = {
                "conversation_type": self.conversation_type,
                "cost": getattr(self.llm, '_last_cost', 0),
                "processing_time": getattr(self.llm, '_last_time', 0),
                "quality_score": getattr(self.llm, '_last_quality', 0),
            }
        
        return result
    
    @property
    def _chain_type(self) -> str:
        """Return the chain type identifier."""
        return f"prsm_conversation_{self.conversation_type}"
