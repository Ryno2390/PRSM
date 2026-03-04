#!/usr/bin/env python3
"""
OpenAI Backend
==============

OpenAI GPT API backend implementation. Supports GPT-4, GPT-4 Turbo,
GPT-3.5 Turbo, and text-embedding models.

Usage:
    backend = OpenAIBackend(api_key="sk-...")
    async with backend:
        result = await backend.generate("What is AI?")
        print(result.content)
        
        # Embeddings
        embed_result = await backend.embed("text to embed")
        print(len(embed_result.embedding))

Requirements:
    - aiohttp package
    - Valid OpenAI API key
"""

import time
from typing import Dict, List, Optional, Any

from .base import ModelBackend, BackendType, GenerateResult, EmbedResult, TokenUsage
from .exceptions import (
    BackendUnavailableError,
    BackendRateLimitError,
    BackendAuthenticationError,
    BackendResponseError,
    BackendTimeoutError,
)


class OpenAIBackend(ModelBackend):
    """
    OpenAI GPT API backend.
    
    Supported Models:
    - gpt-4o (default)
    - gpt-4-turbo
    - gpt-4
    - gpt-3.5-turbo
    - text-embedding-3-small (embeddings)
    - text-embedding-3-large (embeddings)
    
    Attributes:
        DEFAULT_MODEL: Default model for generation
        DEFAULT_EMBEDDING_MODEL: Default model for embeddings
        BASE_URL: API base URL
        api_key: OpenAI API key
        session: HTTP session for requests
    """
    
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    BASE_URL = "https://api.openai.com/v1"
    
    # Model information
    MODELS = {
        "gpt-4o": {
            "model_id": "gpt-4o",
            "provider": "openai",
            "context_window": 128000,
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": True,
            "pricing": {"input": 0.005, "output": 0.015}  # per 1K tokens
        },
        "gpt-4-turbo": {
            "model_id": "gpt-4-turbo",
            "provider": "openai",
            "context_window": 128000,
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": True,
            "pricing": {"input": 0.01, "output": 0.03}
        },
        "gpt-4": {
            "model_id": "gpt-4",
            "provider": "openai",
            "context_window": 8192,
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": False,
            "pricing": {"input": 0.03, "output": 0.06}
        },
        "gpt-3.5-turbo": {
            "model_id": "gpt-3.5-turbo",
            "provider": "openai",
            "context_window": 16385,
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": False,
            "pricing": {"input": 0.0005, "output": 0.0015}
        },
        "text-embedding-3-small": {
            "model_id": "text-embedding-3-small",
            "provider": "openai",
            "context_window": 8191,
            "dimensions": 1536,
            "supports_streaming": False,
            "supports_functions": False,
            "pricing": {"input": 0.00002, "output": 0}
        },
        "text-embedding-3-large": {
            "model_id": "text-embedding-3-large",
            "provider": "openai",
            "context_window": 8191,
            "dimensions": 3072,
            "supports_streaming": False,
            "supports_functions": False,
            "pricing": {"input": 0.00013, "output": 0}
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs
    ):
        """
        Initialize the OpenAI backend.
        
        Args:
            api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__(kwargs)
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = kwargs.get("base_url", self.BASE_URL)
        self._session: Optional[Any] = None
    
    @property
    def backend_type(self) -> BackendType:
        """Return the backend type identifier."""
        return BackendType.OPENAI
    
    @property
    def models_supported(self) -> List[str]:
        """Return list of supported model IDs."""
        return list(self.MODELS.keys())
    
    async def initialize(self) -> None:
        """
        Initialize the backend.
        
        Creates the HTTP session and validates the API key.
        
        Raises:
            BackendAuthenticationError: If API key is missing or invalid
            BackendUnavailableError: If initialization fails
        """
        if self._initialized:
            return
        
        # Import aiohttp here to allow module to load without it
        try:
            import aiohttp
        except ImportError:
            raise BackendUnavailableError(
                "aiohttp package not installed. Install with: pip install aiohttp",
                backend_type=self.backend_type.value
            )
        
        # Validate API key exists
        if not self.api_key:
            raise BackendAuthenticationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.",
                backend_type=self.backend_type.value
            )
        
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        # Validate API key with a minimal request
        try:
            is_available = await self.is_available()
            if not is_available:
                raise BackendAuthenticationError(
                    "Invalid OpenAI API key or API unavailable",
                    backend_type=self.backend_type.value
                )
            self._initialized = True
        except BackendAuthenticationError:
            await self.close()
            raise
        except Exception as e:
            await self.close()
            raise BackendUnavailableError(
                f"Failed to initialize OpenAI backend: {e}",
                backend_type=self.backend_type.value
            )
    
    async def close(self) -> None:
        """Clean up resources and close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False
    
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
        Generate text from a prompt using GPT.
        
        Args:
            prompt: The input prompt
            model_id: Specific model to use (default: gpt-4o)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (top_p, frequency_penalty, presence_penalty, stop)
            
        Returns:
            GenerateResult: The generated content and metadata
            
        Raises:
            BackendUnavailableError: If backend is not initialized
            BackendRateLimitError: If rate limit is exceeded
            BackendAuthenticationError: If API key is invalid
            BackendResponseError: If response is invalid
        """
        if not self._initialized:
            await self.initialize()
        
        import aiohttp
        
        model = model_id or self.DEFAULT_MODEL
        start_time = time.time()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request payload
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        
        try:
            async with self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                elif response.status == 429:
                    retry_after = response.headers.get("retry-after")
                    raise BackendRateLimitError(
                        "OpenAI rate limit exceeded",
                        backend_type=self.backend_type.value,
                        retry_after=int(retry_after) if retry_after else None
                    )
                elif response.status == 401:
                    raise BackendAuthenticationError(
                        "Invalid OpenAI API key",
                        backend_type=self.backend_type.value
                    )
                elif response.status == 408:
                    raise BackendTimeoutError(
                        "OpenAI API request timed out",
                        backend_type=self.backend_type.value,
                        timeout_seconds=self.timeout
                    )
                else:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", str(response.status))
                    except:
                        error_msg = f"HTTP {response.status}"
                    raise BackendResponseError(
                        f"OpenAI API error: {error_msg}",
                        backend_type=self.backend_type.value,
                        status_code=response.status
                    )
            
            execution_time = time.time() - start_time
            
            # Parse response
            choices = data.get("choices", [])
            if not choices:
                raise BackendResponseError(
                    "No choices in OpenAI response",
                    backend_type=self.backend_type.value,
                    response_data=data
                )
            
            choice = choices[0]
            content = choice.get("message", {}).get("content", "")
            
            usage = data.get("usage", {})
            
            return GenerateResult(
                content=content,
                model_id=model,
                provider=self.backend_type,
                token_usage=TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0)
                ),
                execution_time=execution_time,
                finish_reason=choice.get("finish_reason", "stop"),
                metadata={"raw_response": data}
            )
        
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(
                f"Network error calling OpenAI API: {e}",
                backend_type=self.backend_type.value
            )
    
    async def embed(
        self,
        text: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> EmbedResult:
        """
        Generate embedding for text using OpenAI embedding models.
        
        Args:
            text: Text to embed
            model_id: Specific embedding model to use (default: text-embedding-3-small)
            **kwargs: Additional parameters (dimensions)
            
        Returns:
            EmbedResult: The embedding vector and metadata
            
        Raises:
            BackendUnavailableError: If backend is not initialized
            BackendResponseError: If response is invalid
        """
        if not self._initialized:
            await self.initialize()
        
        import aiohttp
        
        model = model_id or self.DEFAULT_EMBEDDING_MODEL
        start_time = time.time()
        
        # Build request payload
        payload = {
            "model": model,
            "input": text
        }
        
        # Add optional parameters
        if "dimensions" in kwargs:
            payload["dimensions"] = kwargs["dimensions"]
        
        try:
            async with self._session.post(
                f"{self.base_url}/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                elif response.status == 429:
                    retry_after = response.headers.get("retry-after")
                    raise BackendRateLimitError(
                        "OpenAI rate limit exceeded",
                        backend_type=self.backend_type.value,
                        retry_after=int(retry_after) if retry_after else None
                    )
                elif response.status == 401:
                    raise BackendAuthenticationError(
                        "Invalid OpenAI API key",
                        backend_type=self.backend_type.value
                    )
                else:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", str(response.status))
                    except:
                        error_msg = f"HTTP {response.status}"
                    raise BackendResponseError(
                        f"OpenAI embedding error: {error_msg}",
                        backend_type=self.backend_type.value,
                        status_code=response.status
                    )
            
            # Parse response
            embedding_data = data.get("data", [])
            if not embedding_data:
                raise BackendResponseError(
                    "No embedding data in OpenAI response",
                    backend_type=self.backend_type.value,
                    response_data=data
                )
            
            embedding = embedding_data[0].get("embedding", [])
            usage = data.get("usage", {})
            
            return EmbedResult(
                embedding=embedding,
                model_id=model,
                provider=self.backend_type,
                token_count=usage.get("total_tokens", 0),
                metadata={
                    "raw_response": data,
                    "dimensions": len(embedding)
                }
            )
        
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(
                f"Network error calling OpenAI API: {e}",
                backend_type=self.backend_type.value
            )
    
    async def is_available(self) -> bool:
        """
        Check if the backend is available and healthy.
        
        Makes a minimal request to validate connectivity and API key.
        
        Returns:
            bool: True if backend can process requests
        """
        if not self.api_key or not self._session:
            return False
        
        try:
            import aiohttp
            async with self._session.get(
                f"{self.base_url}/models"
            ) as response:
                return response.status == 200
        except Exception:
            return False
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Args:
            model_id: Specific model to get info for (all if None)
            
        Returns:
            Dict with model information
        """
        if model_id:
            return self.MODELS.get(model_id, {})
        return self.MODELS.copy()
    
    def __repr__(self) -> str:
        """String representation of the backend."""
        return f"OpenAIBackend(model={self.DEFAULT_MODEL}, initialized={self._initialized})"