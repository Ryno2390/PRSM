"""
PRSM LangChain Chat Model Implementation
======================================

Chat model implementation for LangChain that uses PRSM's conversational capabilities.
Supports both single messages and conversation history.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from pydantic import Field

try:
    from langchain.chat_models.base import BaseChatModel
    from langchain.schema import (
        BaseMessage,
        AIMessage,
        HumanMessage,
        SystemMessage,
        ChatMessage,
        ChatResult,
        ChatGeneration,
    )
    from langchain.callbacks.manager import (
        CallbackManagerForLLMRun,
        AsyncCallbackManagerForLLMRun,
    )
except ImportError:
    raise ImportError(
        "LangChain is required for PRSM LangChain integration. "
        "Install with: pip install langchain"
    )

from prsm_sdk import PRSMClient, PRSMError
from prsm.core.exceptions import PRSMException

logger = logging.getLogger(__name__)


class PRSMChatModel(BaseChatModel):
    """
    LangChain Chat Model implementation using PRSM.
    
    This class provides a chat-based interface to PRSM, enabling conversation
    history and context management through LangChain's chat model interface.
    
    Example:
        chat_model = PRSMChatModel(
            base_url="http://localhost:8000",
            api_key="your-api-key",
            default_user_id="chat-user"
        )
        
        # Single message
        messages = [HumanMessage(content="Hello, how are you?")]
        response = chat_model(messages)
        
        # Conversation with history
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="What is machine learning?"),
            AIMessage(content="Machine learning is..."),
            HumanMessage(content="Can you give me an example?")
        ]
        response = chat_model(messages)
    """
    
    base_url: str = Field(..., description="PRSM server base URL")
    api_key: str = Field(..., description="PRSM API key")
    default_user_id: str = Field(default="chat-user", description="Default user ID for queries")
    context_allocation: int = Field(default=50, description="Default FTNS allocation per query")
    quality_threshold: int = Field(default=75, description="Minimum quality score threshold")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    maintain_session: bool = Field(default=True, description="Whether to maintain conversation sessions")
    max_context_length: int = Field(default=4000, description="Maximum context length in characters")
    
    # Internal client and session tracking
    _client: Optional[PRSMClient] = None
    _sessions: Dict[str, str] = {}  # user_id -> session_id mapping
    
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
            logger.info(f"PRSM Chat Model initialized with base_url: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize PRSM client: {e}")
            raise PRSMException(f"PRSM client initialization failed: {e}")
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "prsm-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "base_url": self.base_url,
            "default_user_id": self.default_user_id,
            "context_allocation": self.context_allocation,
            "quality_threshold": self.quality_threshold,
            "maintain_session": self.maintain_session,
        }
    
    def _format_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Convert LangChain messages to a PRSM-compatible prompt.
        
        Args:
            messages: List of LangChain messages
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            elif isinstance(message, ChatMessage):
                prompt_parts.append(f"{message.role}: {message.content}")
            else:
                prompt_parts.append(f"Message: {message.content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Truncate if too long
        if len(full_prompt) > self.max_context_length:
            logger.warning(
                f"Prompt length {len(full_prompt)} exceeds max length {self.max_context_length}, truncating"
            )
            full_prompt = full_prompt[-self.max_context_length:]
        
        return full_prompt
    
    def _extract_user_id_from_kwargs(self, **kwargs) -> str:
        """Extract user ID from kwargs or use default."""
        return kwargs.get("user_id", self.default_user_id)
    
    async def _get_or_create_session(self, user_id: str) -> Optional[str]:
        """
        Get existing session or create new one for user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Session ID if maintaining sessions, None otherwise
        """
        if not self.maintain_session:
            return None
        
        if user_id not in self._sessions:
            try:
                # Create new session through PRSM
                session = await self._client.create_session(
                    user_id=user_id,
                    initial_context={"conversation_type": "langchain_chat"}
                )
                self._sessions[user_id] = session.session_id
                logger.info(f"Created new session {session.session_id} for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to create session for user {user_id}: {e}")
                return None
        
        return self._sessions.get(user_id)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat response synchronously.
        
        Args:
            messages: List of chat messages
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Returns:
            ChatResult with generated response
        """
        if not self._client:
            self._initialize_client()
        
        # Extract parameters
        user_id = self._extract_user_id_from_kwargs(**kwargs)
        context_allocation = kwargs.get("context_allocation", self.context_allocation)
        
        # Format messages to prompt
        prompt = self._format_messages_to_prompt(messages)
        
        try:
            # Run async code in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Get or create session
                session_id = loop.run_until_complete(
                    self._get_or_create_session(user_id)
                )
                
                # Query PRSM
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
                    run_manager.on_text(f"PRSM Chat Cost: {response.ftns_charged} FTNS\n")
                    run_manager.on_text(f"Processing Time: {response.processing_time:.2f}s\n")
                    run_manager.on_text(f"Quality Score: {response.quality_score}/100\n")
                
                # Create AI message
                ai_message = AIMessage(
                    content=response.final_answer,
                    additional_kwargs={
                        "prsm_cost": response.ftns_charged,
                        "prsm_time": response.processing_time,
                        "prsm_quality": response.quality_score,
                        "query_id": response.query_id,
                    }
                )
                
                generation = ChatGeneration(
                    message=ai_message,
                    generation_info={
                        "prsm_cost": response.ftns_charged,
                        "prsm_time": response.processing_time,
                        "prsm_quality": response.quality_score,
                    }
                )
                
                return ChatResult(
                    generations=[generation],
                    llm_output={
                        "total_cost": response.ftns_charged,
                        "total_time": response.processing_time,
                        "model_name": "prsm-chat",
                    }
                )
                
            finally:
                loop.close()
        
        except PRSMError as e:
            logger.error(f"PRSM chat query failed: {e}")
            raise PRSMException(f"PRSM chat query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM chat call: {e}")
            raise PRSMException(f"PRSM chat call failed: {e}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat response asynchronously.
        
        Args:
            messages: List of chat messages
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Async callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Returns:
            ChatResult with generated response
        """
        if not self._client:
            self._initialize_client()
        
        # Extract parameters
        user_id = self._extract_user_id_from_kwargs(**kwargs)
        context_allocation = kwargs.get("context_allocation", self.context_allocation)
        
        # Format messages to prompt
        prompt = self._format_messages_to_prompt(messages)
        
        try:
            # Get or create session
            session_id = await self._get_or_create_session(user_id)
            
            # Query PRSM
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
                await run_manager.on_text(f"PRSM Chat Cost: {response.ftns_charged} FTNS\n")
                await run_manager.on_text(f"Processing Time: {response.processing_time:.2f}s\n")
                await run_manager.on_text(f"Quality Score: {response.quality_score}/100\n")
            
            # Create AI message
            ai_message = AIMessage(
                content=response.final_answer,
                additional_kwargs={
                    "prsm_cost": response.ftns_charged,
                    "prsm_time": response.processing_time,
                    "prsm_quality": response.quality_score,
                    "query_id": response.query_id,
                }
            )
            
            generation = ChatGeneration(
                message=ai_message,
                generation_info={
                    "prsm_cost": response.ftns_charged,
                    "prsm_time": response.processing_time,
                    "prsm_quality": response.quality_score,
                }
            )
            
            return ChatResult(
                generations=[generation],
                llm_output={
                    "total_cost": response.ftns_charged,
                    "total_time": response.processing_time,
                    "model_name": "prsm-chat",
                }
            )
            
        except PRSMError as e:
            logger.error(f"PRSM async chat query failed: {e}")
            raise PRSMException(f"PRSM async chat query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM async chat call: {e}")
            raise PRSMException(f"PRSM async chat call failed: {e}")
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGeneration]:
        """
        Stream chat response from PRSM.
        
        Args:
            messages: List of chat messages
            stop: Stop sequences (not used by PRSM but kept for compatibility)
            run_manager: Async callback manager for handling callbacks
            **kwargs: Additional keyword arguments
        
        Yields:
            ChatGeneration objects with streaming content
        """
        if not self._client:
            self._initialize_client()
        
        # Extract parameters
        user_id = self._extract_user_id_from_kwargs(**kwargs)
        
        # Format messages to prompt
        prompt = self._format_messages_to_prompt(messages)
        
        try:
            # Get or create session
            session_id = await self._get_or_create_session(user_id)
            
            async for chunk in self._client.stream(prompt, user_id=user_id, session_id=session_id):
                if chunk.content:
                    if run_manager:
                        await run_manager.on_text(chunk.content)
                    
                    ai_message = AIMessage(
                        content=chunk.content,
                        additional_kwargs={
                            "chunk": True,
                            "streaming": True,
                        }
                    )
                    
                    yield ChatGeneration(
                        message=ai_message,
                        generation_info={"chunk": True}
                    )
                    
        except PRSMError as e:
            logger.error(f"PRSM chat stream failed: {e}")
            raise PRSMException(f"PRSM chat stream failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in PRSM chat stream: {e}")
            raise PRSMException(f"PRSM chat stream failed: {e}")
    
    def clear_session(self, user_id: Optional[str] = None) -> None:
        """
        Clear session for a specific user or all users.
        
        Args:
            user_id: User ID to clear session for. If None, clears all sessions.
        """
        if user_id:
            if user_id in self._sessions:
                del self._sessions[user_id]
                logger.info(f"Cleared session for user {user_id}")
        else:
            self._sessions.clear()
            logger.info("Cleared all sessions")
    
    async def close(self) -> None:
        """Close the PRSM client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._sessions.clear()
            logger.info("PRSM Chat Model client closed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._client:
            logger.warning("PRSM Chat Model client not properly closed. Use await chat_model.close()")
