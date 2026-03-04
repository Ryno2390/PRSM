#!/usr/bin/env python3
"""
Model Backend Base Classes
===========================

This module defines the abstract base class and data structures for LLM backends.
All backend implementations must inherit from ModelBackend and implement its
abstract methods.

Classes:
    BackendType: Enum of supported backend types
    TokenUsage: Token usage statistics
    GenerateResult: Result from text generation
    EmbedResult: Result from text embedding
    ModelBackend: Abstract base class for all backends
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class BackendType(str, Enum):
    """
    Supported backend types for LLM inference.
    
    Each backend type corresponds to a different LLM provider or mode:
    - ANTHROPIC: Claude API (claude-3-5-sonnet, claude-3-opus, etc.)
    - OPENAI: GPT API (gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.)
    - LOCAL: Local inference (Ollama, transformers, etc.)
    - MOCK: Testing backend with deterministic responses
    """
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"
    MOCK = "mock"


@dataclass
class TokenUsage:
    """
    Token usage statistics for a generation request.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        """Calculate total if not explicitly set"""
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class GenerateResult:
    """
    Result from a text generation request.
    
    This dataclass encapsulates all information returned from a generation
    request, including the generated content, metadata about the model used,
    token usage statistics, and execution timing.
    
    Attributes:
        content: The generated text content
        model_id: The model identifier used for generation
        provider: The backend provider type
        token_usage: Token usage statistics
        execution_time: Time taken to generate the response (seconds)
        finish_reason: Why generation stopped ("stop", "length", etc.)
        metadata: Additional provider-specific metadata
    """
    content: str
    model_id: str
    provider: BackendType
    token_usage: TokenUsage
    execution_time: float
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "content": self.content,
            "model_id": self.model_id,
            "provider": self.provider.value,
            "token_usage": self.token_usage.to_dict(),
            "execution_time": self.execution_time,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata
        }


@dataclass
class EmbedResult:
    """
    Result from a text embedding request.
    
    This dataclass encapsulates the embedding vector and associated metadata
    returned from an embedding request.
    
    Attributes:
        embedding: The embedding vector (list of floats)
        model_id: The embedding model identifier
        provider: The backend provider type
        token_count: Number of tokens in the input text
        metadata: Additional provider-specific metadata
    """
    embedding: List[float]
    model_id: str
    provider: BackendType
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "embedding": self.embedding,
            "model_id": self.model_id,
            "provider": self.provider.value,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


class ModelBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    All backends must implement this interface to be used in the NWTN pipeline.
    The interface supports both text generation and embedding operations.
    
    The backend lifecycle is:
    1. Create instance with configuration
    2. Call initialize() to set up connections/validate credentials
    3. Use generate()/embed() for inference
    4. Call close() to clean up resources
    
    Backends can also be used as async context managers:
    
        async with MyBackend(config) as backend:
            result = await backend.generate("Hello, world!")
    
    Attributes:
        config: Backend-specific configuration dictionary
        _initialized: Whether the backend has been initialized
    
    Example:
        class MyBackend(ModelBackend):
            @property
            def backend_type(self) -> BackendType:
                return BackendType.MOCK
            
            async def generate(self, prompt: str, **kwargs) -> GenerateResult:
                # Implementation here
                pass
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backend instance.
        
        Args:
            config: Optional configuration dictionary for backend-specific settings
        """
        self.config = config or {}
        self._initialized = False
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """
        Return the backend type identifier.
        
        Returns:
            BackendType: The type of this backend (ANTHROPIC, OPENAI, LOCAL, MOCK)
        """
        pass
    
    @property
    @abstractmethod
    def models_supported(self) -> List[str]:
        """
        Return list of supported model IDs.
        
        Returns:
            List[str]: List of model identifiers supported by this backend
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend.
        
        Called once before first use. Should validate API keys,
        load models, establish connections, etc.
        
        Raises:
            BackendAuthenticationError: If API key is invalid
            BackendUnavailableError: If backend cannot be initialized
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Clean up resources.
        
        Called when the backend is no longer needed. Should close
        connections, free memory, etc.
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerateResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt text
            model_id: Specific model to use (backend default if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            system_prompt: Optional system prompt for chat models
            **kwargs: Additional provider-specific parameters
            
        Returns:
            GenerateResult: The generated content and metadata
            
        Raises:
            BackendUnavailableError: If backend is not available
            BackendRateLimitError: If rate limit is exceeded
            BackendTimeoutError: If request times out
            BackendResponseError: If response is invalid
        """
        pass
    
    @abstractmethod
    async def embed(
        self,
        text: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> EmbedResult:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model_id: Specific embedding model to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            EmbedResult: The embedding vector and metadata
            
        Raises:
            NotImplementedError: If backend doesn't support embeddings
            BackendUnavailableError: If backend is not available
            BackendRateLimitError: If rate limit is exceeded
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the backend is available and healthy.
        
        Should verify:
        - API keys are valid (for cloud providers)
        - Model is loaded (for local backends)
        - Network connectivity exists
        - Rate limits are not exceeded
        
        Returns:
            bool: True if backend can process requests
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Args:
            model_id: Specific model to get info for (all if None)
            
        Returns:
            Dict with model information including:
            - model_id: Model identifier
            - provider: Backend provider
            - context_window: Maximum context length
            - supports_streaming: Whether streaming is supported
            - supports_functions: Whether function calling is supported
            - pricing: Cost per token (if applicable)
        """
        pass
    
    async def __aenter__(self) -> "ModelBackend":
        """
        Async context manager entry.
        
        Initializes the backend when entering the context.
        
        Returns:
            ModelBackend: The initialized backend instance
        """
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit.
        
        Cleans up resources when exiting the context.
        """
        await self.close()
    
    def __repr__(self) -> str:
        """String representation of the backend"""
        return f"{self.__class__.__name__}(type={self.backend_type.value}, initialized={self._initialized})"