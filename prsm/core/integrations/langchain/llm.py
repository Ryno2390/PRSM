"""
PRSM LangChain LLM Implementation
================================

LangChain-compatible LLM wrapper for PRSM that enables using PRSM as a drop-in
replacement for other LLMs in LangChain applications.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from pydantic import Field

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
    from langchain.schema import Generation, LLMResult
except ImportError:
    raise ImportError(
        "LangChain is required for PRSM LangChain integration. "
        "Install with: pip install langchain"
    )

from prsm_sdk import PRSMClient, PRSMError
from prsm.core.exceptions import PRSMException

logger = logging.getLogger(__name__)


class PRSMLangChainLLM(LLM):
    """
    LangChain LLM implementation using PRSM as the backend.
    
    This class wraps PRSM's query functionality to provide a LangChain-compatible
    interface, enabling PRSM to be used in any LangChain application.
    
    Example:
        llm = PRSMLangChainLLM(
            base_url="http://localhost:8000",
            api_key="your-api-key",
            default_user_id="langchain-user"
        )
        
        # Use directly
        response = llm("What is quantum computing?")
        
        # Use in LangChain chains
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        prompt = PromptTemplate(template="Explain {topic} in simple terms", input_variables=["topic"])
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(topic="machine learning")
    """
    
    base_url: str = Field(..., description="PRSM server base URL")
    api_key: str = Field(..., description="PRSM API key")
    default_user_id: str = Field(default="langchain-user", description="Default user ID for queries")
    context_allocation: int = Field(default=50, description="Default FTNS allocation per query")
    quality_threshold: int = Field(default=75, description="Minimum quality score threshold")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    
    # Internal client - not included in serialization
    _client: Optional[PRSMClient] = None
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize PRSM client"""
        try:
            self._client = PRSMClient(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
            logger.info(f"PRSM LangChain LLM initialized with base_url: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize PRSM client: {e}")
            raise PRSMException(f"PRSM client initialization failed: {e}")
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "prsm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "base_url": self.base_url,
            "default_user_id": self.default_user_id,
            "context_allocation": self.context_allocation,
            "quality_threshold": self.quality_threshold,
            "timeout": self.timeout,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call PRSM with the given prompt and return the response.
        
        Args:
            prompt: The prompt to send to PRSM
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Returns:
            The text response from PRSM
        """
        if not self._client:
            self._initialize_client()
        
        # Extract parameters from kwargs
        user_id = kwargs.get("user_id", self.default_user_id)
        context_allocation = kwargs.get("context_allocation", self.context_allocation)
        session_id = kwargs.get("session_id")
        
        try:
            # Run async query in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    self._client.query(
                        prompt=prompt,
                        user_id=user_id,
                        context_allocation=context_allocation,
                        session_id=session_id
                    )
                )
                
                # Check quality threshold
                if response.quality_score < self.quality_threshold:
                    logger.warning(
                        f"Response quality {response.quality_score} below threshold {self.quality_threshold}"
                    )
                
                # Log callback information
                if run_manager:
                    run_manager.on_text(f"PRSM Query Cost: {response.ftns_charged} FTNS\n")
                    run_manager.on_text(f"Processing Time: {response.processing_time:.2f}s\n")
                    run_manager.on_text(f"Quality Score: {response.quality_score}/100\n")
                
                return response.final_answer
                
            finally:
                loop.close()
        
        except PRSMError as e:
            logger.error(f"PRSM query failed: {e}")
            raise PRSMException(f"PRSM query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM LLM call: {e}")
            raise PRSMException(f"PRSM LLM call failed: {e}")
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async call to PRSM with the given prompt.
        
        Args:
            prompt: The prompt to send to PRSM
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Async callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Returns:
            The text response from PRSM
        """
        if not self._client:
            self._initialize_client()
        
        # Extract parameters from kwargs
        user_id = kwargs.get("user_id", self.default_user_id)
        context_allocation = kwargs.get("context_allocation", self.context_allocation)
        session_id = kwargs.get("session_id")
        
        try:
            response = await self._client.query(
                prompt=prompt,
                user_id=user_id,
                context_allocation=context_allocation,
                session_id=session_id
            )
            
            # Check quality threshold
            if response.quality_score < self.quality_threshold:
                logger.warning(
                    f"Response quality {response.quality_score} below threshold {self.quality_threshold}"
                )
            
            # Log callback information
            if run_manager:
                await run_manager.on_text(f"PRSM Query Cost: {response.ftns_charged} FTNS\n")
                await run_manager.on_text(f"Processing Time: {response.processing_time:.2f}s\n")
                await run_manager.on_text(f"Quality Score: {response.quality_score}/100\n")
            
            return response.final_answer
            
        except PRSMError as e:
            logger.error(f"PRSM async query failed: {e}")
            raise PRSMException(f"PRSM async query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM async LLM call: {e}")
            raise PRSMException(f"PRSM async LLM call failed: {e}")
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts to process
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Returns:
            LLMResult containing all generations
        """
        generations = []
        total_cost = 0.0
        total_time = 0.0
        
        for prompt in prompts:
            try:
                # Call individual prompt
                response_text = self._call(prompt, stop, run_manager, **kwargs)
                
                # Create generation object
                generation = Generation(
                    text=response_text,
                    generation_info={
                        "prsm_cost": getattr(self, "_last_cost", 0),
                        "prsm_time": getattr(self, "_last_time", 0),
                        "prsm_quality": getattr(self, "_last_quality", 0),
                    }
                )
                generations.append([generation])
                
            except Exception as e:
                logger.error(f"Failed to generate response for prompt: {e}")
                # Create empty generation for failed prompt
                generation = Generation(
                    text="",
                    generation_info={"error": str(e)}
                )
                generations.append([generation])
        
        return LLMResult(
            generations=generations,
            llm_output={
                "total_cost": total_cost,
                "total_time": total_time,
                "model_name": "prsm",
            }
        )
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Async generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts to process
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Async callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Returns:
            LLMResult containing all generations
        """
        generations = []
        total_cost = 0.0
        total_time = 0.0
        
        # Process prompts concurrently
        tasks = []
        for prompt in prompts:
            task = self._acall(prompt, stop, run_manager, **kwargs)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to generate response for prompt {i}: {response}")
                generation = Generation(
                    text="",
                    generation_info={"error": str(response)}
                )
            else:
                generation = Generation(
                    text=response,
                    generation_info={
                        "prsm_cost": getattr(self, "_last_cost", 0),
                        "prsm_time": getattr(self, "_last_time", 0),
                        "prsm_quality": getattr(self, "_last_quality", 0),
                    }
                )
            
            generations.append([generation])
        
        return LLMResult(
            generations=generations,
            llm_output={
                "total_cost": total_cost,
                "total_time": total_time,
                "model_name": "prsm",
            }
        )
    
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream response from PRSM.
        
        Args:
            prompt: The prompt to send to PRSM
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Async callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Yields:
            Text chunks from the response
        """
        if not self._client:
            self._initialize_client()
        
        # Extract parameters from kwargs
        user_id = kwargs.get("user_id", self.default_user_id)
        
        try:
            async for chunk in self._client.stream(prompt, user_id=user_id):
                if chunk.content:
                    if run_manager:
                        await run_manager.on_text(chunk.content)
                    yield chunk.content
                    
        except PRSMError as e:
            logger.error(f"PRSM stream failed: {e}")
            raise PRSMException(f"PRSM stream failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM stream: {e}")
            raise PRSMException(f"PRSM stream failed: {e}")
    
    async def close(self) -> None:
        """Close the PRSM client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("PRSM LangChain LLM client closed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._client:
            # Note: We can't run async code in __del__, so we just log
            logger.warning("PRSM LangChain LLM client not properly closed. Use await llm.close()")
