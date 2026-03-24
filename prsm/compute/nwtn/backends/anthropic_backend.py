#!/usr/bin/env python3
"""
Anthropic Backend
=================

Anthropic Claude API backend implementation. Supports Claude 3.5 Sonnet,
Claude 3 Opus, and other Claude models via the Anthropic API.

Usage:
    backend = AnthropicBackend(api_key="sk-ant-...")
    async with backend:
        result = await backend.generate("What is AI?")
        print(result.content)

Requirements:
    - aiohttp package
    - Valid Anthropic API key
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


class AnthropicBackend(ModelBackend):
    """
    Anthropic Claude API backend.
    
    Supported Models:
    - claude-3-5-sonnet-20241022 (default)
    - claude-3-opus-20240229
    - claude-3-haiku-20240307
    - claude-2.1
    - claude-2.0
    
    Note: Anthropic does not provide an embedding API. Use OpenAI or Local
    backend for embeddings.
    
    Attributes:
        DEFAULT_MODEL: Default model identifier
        API_VERSION: Anthropic API version
        base_url: API base URL
        api_key: Anthropic API key
        session: HTTP session for requests
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-5"
    API_VERSION = "2023-06-01"
    BASE_URL = "https://api.anthropic.com/v1"
    
    # Model information
    MODELS = {
        "claude-3-5-sonnet-20241022": {
            "model_id": "claude-3-5-sonnet-20241022",
            "provider": "anthropic",
            "context_window": 200000,
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": True,
            "pricing": {"input": 0.003, "output": 0.015}  # per 1K tokens
        },
        "claude-3-opus-20240229": {
            "model_id": "claude-3-opus-20240229",
            "provider": "anthropic",
            "context_window": 200000,
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": True,
            "pricing": {"input": 0.015, "output": 0.075}
        },
        "claude-3-haiku-20240307": {
            "model_id": "claude-3-haiku-20240307",
            "provider": "anthropic",
            "context_window": 200000,
            "supports_streaming": True,
            "supports_functions": True,
            "supports_vision": True,
            "pricing": {"input": 0.00025, "output": 0.00125}
        },
        "claude-2.1": {
            "model_id": "claude-2.1",
            "provider": "anthropic",
            "context_window": 200000,
            "supports_streaming": True,
            "supports_functions": False,
            "supports_vision": False,
            "pricing": {"input": 0.008, "output": 0.024}
        },
        "claude-2.0": {
            "model_id": "claude-2.0",
            "provider": "anthropic",
            "context_window": 100000,
            "supports_streaming": True,
            "supports_functions": False,
            "supports_vision": False,
            "pricing": {"input": 0.008, "output": 0.024}
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs
    ):
        """
        Initialize the Anthropic backend.
        
        Args:
            api_key: Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__(kwargs)
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = kwargs.get("base_url", self.BASE_URL)
        self._session: Optional[Any] = None
        self._models_cache: Dict[str, Any] = {}
    
    @property
    def backend_type(self) -> BackendType:
        """Return the backend type identifier."""
        return BackendType.ANTHROPIC
    
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
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.",
                backend_type=self.backend_type.value
            )
        
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": self.API_VERSION
            }
        )
        
        # Validate API key with a minimal request
        try:
            is_available = await self.is_available()
            if not is_available:
                raise BackendAuthenticationError(
                    "Invalid Anthropic API key or API unavailable",
                    backend_type=self.backend_type.value
                )
            self._initialized = True
        except BackendAuthenticationError:
            await self.close()
            raise
        except Exception as e:
            await self.close()
            raise BackendUnavailableError(
                f"Failed to initialize Anthropic backend: {e}",
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
        Generate text from a prompt using Claude.
        
        Args:
            prompt: The input prompt
            model_id: Specific model to use (default: claude-3-5-sonnet)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (top_p, top_k, stop_sequences)
            
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
        
        # Build request payload
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "stop_sequences" in kwargs:
            payload["stop_sequences"] = kwargs["stop_sequences"]
        
        try:
            async with self._session.post(
                f"{self.base_url}/messages",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                elif response.status == 429:
                    retry_after = response.headers.get("retry-after")
                    raise BackendRateLimitError(
                        "Anthropic rate limit exceeded",
                        backend_type=self.backend_type.value,
                        retry_after=int(retry_after) if retry_after else None
                    )
                elif response.status == 401:
                    raise BackendAuthenticationError(
                        "Invalid Anthropic API key",
                        backend_type=self.backend_type.value
                    )
                elif response.status == 408:
                    raise BackendTimeoutError(
                        "Anthropic API request timed out",
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
                        f"Anthropic API error: {error_msg}",
                        backend_type=self.backend_type.value,
                        status_code=response.status
                    )
            
            execution_time = time.time() - start_time
            
            # Parse response
            content = data.get("content", [{}])
            if isinstance(content, list) and len(content) > 0:
                text = content[0].get("text", "")
            else:
                text = ""
            
            usage = data.get("usage", {})
            
            return GenerateResult(
                content=text,
                model_id=model,
                provider=self.backend_type,
                token_usage=TokenUsage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                ),
                execution_time=execution_time,
                finish_reason=data.get("stop_reason", "stop"),
                metadata={"raw_response": data}
            )
        
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(
                f"Network error calling Anthropic API: {e}",
                backend_type=self.backend_type.value
            )
    
    async def embed(
        self,
        text: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> EmbedResult:
        """
        Generate embedding for text.
        
        Note: Anthropic does not provide an embedding API.
        
        Raises:
            NotImplementedError: Always, as Anthropic doesn't support embeddings
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Use OpenAI or Local backend for embeddings."
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
            async with self._session.post(
                f"{self.base_url}/messages",
                json={
                    "model": self.DEFAULT_MODEL,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "test"}]
                }
            ) as response:
                # 200 means success, 429 means valid key but rate limited
                return response.status in [200, 429]
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
        return f"AnthropicBackend(model={self.DEFAULT_MODEL}, initialized={self._initialized})"