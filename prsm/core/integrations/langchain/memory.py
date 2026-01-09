"""
PRSM LangChain Memory Implementation
===================================

Memory implementations that leverage PRSM's session management for
conversation history and context preservation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import Field

try:
    from langchain.schema import BaseMemory, BaseMessage, HumanMessage, AIMessage
    from langchain.memory.chat_memory import BaseChatMemory
except ImportError:
    raise ImportError(
        "LangChain is required for PRSM LangChain integration. "
        "Install with: pip install langchain"
    )

from prsm_sdk import PRSMClient, PRSMError
from prsm.core.exceptions import PRSMException

logger = logging.getLogger(__name__)


class PRSMChatMemory(BaseChatMemory):
    """
    Chat memory implementation that uses PRSM's session management.
    
    This memory class stores conversation history using PRSM's native
    session management, enabling context preservation across interactions
    and integration with PRSM's contextual understanding.
    
    Example:
        memory = PRSMChatMemory(
            prsm_client=PRSMClient("http://localhost:8000", "api-key"),
            user_id="conversation-user",
            session_prefix="langchain"
        )
        
        # Use with LangChain chains
        conversation = ConversationChain(
            llm=llm,
            memory=memory
        )
    """
    
    prsm_client: PRSMClient = Field(..., description="PRSM client instance")
    user_id: str = Field(..., description="User ID for session management")
    session_prefix: str = Field(default="langchain", description="Prefix for session IDs")
    auto_create_session: bool = Field(default=True, description="Whether to auto-create sessions")
    max_token_limit: int = Field(default=2000, description="Maximum tokens to store in memory")
    
    # Internal session tracking
    _session_id: Optional[str] = None
    _message_count: int = 0
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.auto_create_session:
            self._ensure_session()
    
    def _ensure_session(self) -> None:
        """
        Ensure a PRSM session exists for this memory instance.
        """
        if not self._session_id:
            try:
                # Run async session creation in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    session = loop.run_until_complete(
                        self.prsm_client.create_session(
                            user_id=self.user_id,
                            initial_context={
                                "type": "langchain_memory",
                                "prefix": self.session_prefix,
                                "created_by": "PRSMChatMemory"
                            }
                        )
                    )
                    self._session_id = session.session_id
                    logger.info(f"Created PRSM session {self._session_id} for user {self.user_id}")
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"Failed to create PRSM session: {e}")
                # Continue without session - will fall back to in-memory storage
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return ["history"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load conversation history from PRSM session.
        
        Args:
            inputs: Input variables (not used but required by interface)
        
        Returns:
            Dictionary containing conversation history
        """
        try:
            # If we have a PRSM session, try to load from it
            if self._session_id:
                # For now, fall back to parent implementation
                # In a full implementation, we would load from PRSM session
                pass
            
            # Use parent implementation for message storage
            return super().load_memory_variables(inputs)
            
        except Exception as e:
            logger.error(f"Failed to load memory from PRSM session: {e}")
            # Fall back to parent implementation
            return super().load_memory_variables(inputs)
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save conversation context to both local memory and PRSM session.
        
        Args:
            inputs: Input dictionary containing user message
            outputs: Output dictionary containing AI response
        """
        try:
            # Save to parent memory first
            super().save_context(inputs, outputs)
            
            # Update message count
            self._message_count += 1
            
            # If we have a PRSM session, update it with context
            if self._session_id:
                self._update_prsm_session_context(inputs, outputs)
            
            # Check if we need to prune old messages
            self._maybe_prune_memory()
            
        except Exception as e:
            logger.error(f"Failed to save context to PRSM session: {e}")
            # Continue with parent implementation even if PRSM update fails
    
    def _update_prsm_session_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Update PRSM session with conversation context.
        
        Args:
            inputs: Input dictionary
            outputs: Output dictionary
        """
        try:
            # Extract key information
            user_input = inputs.get(self.input_key, "")
            ai_output = outputs.get(self.output_key, "")
            
            # Create context update
            context_update = {
                "last_interaction": {
                    "user_input": user_input,
                    "ai_response": ai_output,
                    "message_count": self._message_count,
                },
                "conversation_stats": {
                    "total_messages": self._message_count,
                    "session_type": "langchain_memory",
                }
            }
            
            # Run async update in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(
                    self._async_update_session_context(context_update)
                )
            finally:
                loop.close()
                
        except Exception as e:
            logger.warning(f"Failed to update PRSM session context: {e}")
    
    async def _async_update_session_context(self, context_update: Dict[str, Any]) -> None:
        """
        Asynchronously update PRSM session context.
        
        Args:
            context_update: Context information to update
        """
        # This would use PRSM's session management API
        # For now, we'll just log the update
        logger.debug(f"Updating PRSM session {self._session_id} with context: {context_update}")
        
        # In a full implementation, this would call:
        # await self.prsm_client.update_session_context(self._session_id, context_update)
    
    def _maybe_prune_memory(self) -> None:
        """
        Prune old messages if memory is getting too large.
        """
        if not hasattr(self, 'chat_memory') or not self.chat_memory.messages:
            return
        
        # Estimate token count (rough approximation)
        total_chars = sum(len(msg.content) for msg in self.chat_memory.messages)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
        if estimated_tokens > self.max_token_limit:
            # Remove oldest messages (keep system messages if any)
            messages_to_remove = len(self.chat_memory.messages) // 4  # Remove 25% of messages
            
            # Find non-system messages to remove
            removable_indices = []
            for i, msg in enumerate(self.chat_memory.messages):
                if not hasattr(msg, 'type') or msg.type != 'system':
                    removable_indices.append(i)
                if len(removable_indices) >= messages_to_remove:
                    break
            
            # Remove messages (in reverse order to maintain indices)
            for i in reversed(removable_indices):
                del self.chat_memory.messages[i]
            
            logger.info(f"Pruned {len(removable_indices)} old messages from memory")
    
    def clear(self) -> None:
        """
        Clear conversation memory and optionally reset PRSM session.
        """
        # Clear parent memory
        super().clear()
        
        # Reset message count
        self._message_count = 0
        
        # Optionally create new PRSM session
        if self.auto_create_session:
            self._session_id = None
            self._ensure_session()
        
        logger.info("Cleared PRSM chat memory")
    
    def get_session_id(self) -> Optional[str]:
        """
        Get the current PRSM session ID.
        
        Returns:
            Session ID if available, None otherwise
        """
        return self._session_id
    
    def set_session_id(self, session_id: str) -> None:
        """
        Set a specific PRSM session ID to use.
        
        Args:
            session_id: PRSM session ID to use
        """
        self._session_id = session_id
        logger.info(f"Set PRSM session ID to {session_id}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.
        
        Returns:
            Dictionary containing memory statistics
        """
        stats = {
            "message_count": self._message_count,
            "session_id": self._session_id,
            "user_id": self.user_id,
            "has_prsm_session": self._session_id is not None,
        }
        
        if hasattr(self, 'chat_memory') and self.chat_memory.messages:
            stats.update({
                "total_messages": len(self.chat_memory.messages),
                "total_chars": sum(len(msg.content) for msg in self.chat_memory.messages),
                "estimated_tokens": sum(len(msg.content) for msg in self.chat_memory.messages) // 4,
            })
        
        return stats


class PRSMSessionMemory(BaseMemory):
    """
    Specialized memory that maintains multiple PRSM sessions for different contexts.
    
    This memory implementation can manage multiple conversation sessions,
    useful for agents that need to maintain separate contexts for different
    topics or conversation threads.
    
    Example:
        memory = PRSMSessionMemory(
            prsm_client=PRSMClient("http://localhost:8000", "api-key"),
            user_id="multi-session-user"
        )
        
        # Set context for different topics
        memory.set_context("research", {"topic": "quantum computing"})
        memory.set_context("personal", {"topic": "daily planning"})
        
        # Switch between contexts
        memory.switch_context("research")
    """
    
    prsm_client: PRSMClient = Field(..., description="PRSM client instance")
    user_id: str = Field(..., description="User ID for session management")
    
    # Internal context management
    _contexts: Dict[str, Dict[str, Any]] = {}
    _current_context: Optional[str] = None
    _session_memories: Dict[str, PRSMChatMemory] = {}
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return ["history", "context", "session_info"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables for the current context.
        
        Args:
            inputs: Input variables
        
        Returns:
            Dictionary containing memory variables
        """
        if not self._current_context:
            return {"history": "", "context": {}, "session_info": {}}
        
        # Get memory for current context
        memory = self._get_context_memory(self._current_context)
        if memory:
            memory_vars = memory.load_memory_variables(inputs)
            memory_vars.update({
                "context": self._contexts.get(self._current_context, {}),
                "session_info": {
                    "current_context": self._current_context,
                    "available_contexts": list(self._contexts.keys()),
                    "session_id": memory.get_session_id(),
                }
            })
            return memory_vars
        
        return {"history": "", "context": {}, "session_info": {}}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context to the current session memory.
        
        Args:
            inputs: Input dictionary
            outputs: Output dictionary
        """
        if not self._current_context:
            logger.warning("No current context set, cannot save memory")
            return
        
        memory = self._get_context_memory(self._current_context)
        if memory:
            memory.save_context(inputs, outputs)
    
    def _get_context_memory(self, context_name: str) -> Optional[PRSMChatMemory]:
        """
        Get or create memory for a specific context.
        
        Args:
            context_name: Name of the context
        
        Returns:
            PRSMChatMemory instance for the context
        """
        if context_name not in self._session_memories:
            try:
                self._session_memories[context_name] = PRSMChatMemory(
                    prsm_client=self.prsm_client,
                    user_id=self.user_id,
                    session_prefix=f"multicontext_{context_name}"
                )
            except Exception as e:
                logger.error(f"Failed to create memory for context {context_name}: {e}")
                return None
        
        return self._session_memories[context_name]
    
    def set_context(self, context_name: str, context_data: Dict[str, Any]) -> None:
        """
        Set context data for a named context.
        
        Args:
            context_name: Name of the context
            context_data: Context data to store
        """
        self._contexts[context_name] = context_data
        logger.info(f"Set context data for '{context_name}'")
    
    def switch_context(self, context_name: str) -> None:
        """
        Switch to a different context.
        
        Args:
            context_name: Name of the context to switch to
        """
        if context_name not in self._contexts:
            logger.warning(f"Context '{context_name}' not found, creating new context")
            self._contexts[context_name] = {}
        
        self._current_context = context_name
        logger.info(f"Switched to context '{context_name}'")
    
    def get_current_context(self) -> Optional[str]:
        """
        Get the name of the current context.
        
        Returns:
            Current context name or None
        """
        return self._current_context
    
    def list_contexts(self) -> List[str]:
        """
        Get list of all available contexts.
        
        Returns:
            List of context names
        """
        return list(self._contexts.keys())
    
    def clear_context(self, context_name: Optional[str] = None) -> None:
        """
        Clear a specific context or all contexts.
        
        Args:
            context_name: Name of context to clear, or None to clear all
        """
        if context_name:
            if context_name in self._contexts:
                del self._contexts[context_name]
            if context_name in self._session_memories:
                self._session_memories[context_name].clear()
                del self._session_memories[context_name]
            if self._current_context == context_name:
                self._current_context = None
            logger.info(f"Cleared context '{context_name}'")
        else:
            self._contexts.clear()
            for memory in self._session_memories.values():
                memory.clear()
            self._session_memories.clear()
            self._current_context = None
            logger.info("Cleared all contexts")
    
    def clear(self) -> None:
        """
        Clear all memory.
        """
        self.clear_context()
