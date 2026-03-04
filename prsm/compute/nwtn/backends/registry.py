#!/usr/bin/env python3
"""
Backend Registry
================

Central registry for LLM backends. Manages backend lifecycle, selection,
and fallback logic.

Classes:
    BackendHealth: Health status of a backend
    BackendRegistry: Central registry for backend management
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .base import ModelBackend, BackendType, GenerateResult, EmbedResult
from .config import BackendConfig
from .exceptions import (
    BackendUnavailableError,
    BackendRateLimitError,
    AllBackendsFailedError,
)

logger = logging.getLogger(__name__)


@dataclass
class BackendHealth:
    """
    Health status of a backend.
    
    Tracks availability, error counts, and last check time for each backend.
    
    Attributes:
        backend_type: The type of backend
        is_available: Whether the backend is currently available
        last_check: Timestamp of last health check
        error_count: Number of consecutive errors
        last_error: Last error message (if any)
        total_requests: Total number of requests made
        successful_requests: Number of successful requests
    """
    backend_type: BackendType
    is_available: bool = True
    last_check: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    total_requests: int = 0
    successful_requests: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "backend_type": self.backend_type.value,
            "is_available": self.is_available,
            "last_check": self.last_check,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.success_rate
        }


class BackendRegistry:
    """
    Central registry for LLM backends.
    
    Manages backend lifecycle, selection, and fallback logic. The registry
    maintains a collection of backends and handles:
    - Backend initialization and cleanup
    - Health checking and monitoring
    - Request routing with fallback chains
    - Retry logic with exponential backoff
    
    Usage:
        config = BackendConfig.from_environment()
        registry = BackendRegistry(config)
        await registry.initialize()
        
        # Execute with automatic fallback
        result = await registry.execute_with_fallback(
            prompt="What is AI?",
            system_prompt="You are a helpful assistant."
        )
        
        await registry.close()
    
    Attributes:
        config: Backend configuration
        _backends: Dictionary of backend type to backend instance
        _health: Dictionary of backend type to health status
        _initialized: Whether the registry has been initialized
    """
    
    def __init__(self, config: BackendConfig):
        """
        Initialize the registry with configuration.
        
        Args:
            config: Backend configuration specifying primary backend,
                    fallback chain, and other settings
        """
        self.config = config
        self._backends: Dict[BackendType, ModelBackend] = {}
        self._health: Dict[BackendType, BackendHealth] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize all configured backends.
        
        Creates and initializes backends based on the configuration.
        Backends that fail to initialize are logged but don't prevent
        other backends from being initialized.
        
        Raises:
            AllBackendsFailedError: If no backends can be initialized
        """
        if self._initialized:
            return
        
        # Collect backends to initialize (primary + fallback chain, deduplicated)
        backends_to_init = [self.config.primary_backend] + self.config.fallback_chain
        backends_to_init = list(dict.fromkeys(backends_to_init))  # Remove duplicates
        
        initialized_count = 0
        
        for backend_type in backends_to_init:
            try:
                backend = self._create_backend(backend_type)
                await backend.initialize()
                self._backends[backend_type] = backend
                self._health[backend_type] = BackendHealth(
                    backend_type=backend_type,
                    is_available=True,
                    last_check=time.time(),
                    error_count=0
                )
                initialized_count += 1
                logger.info(f"Initialized backend: {backend_type.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize backend {backend_type.value}: {e}")
                self._health[backend_type] = BackendHealth(
                    backend_type=backend_type,
                    is_available=False,
                    last_check=time.time(),
                    error_count=1,
                    last_error=str(e)
                )
        
        if initialized_count == 0:
            raise AllBackendsFailedError(
                "No backends could be initialized",
                errors=[(b.value, "Failed to initialize") for b in backends_to_init]
            )
        
        self._initialized = True
        logger.info(f"Backend registry initialized with {initialized_count} backend(s)")
    
    def _create_backend(self, backend_type: BackendType) -> ModelBackend:
        """
        Factory method to create backend instances.
        
        Args:
            backend_type: Type of backend to create
            
        Returns:
            ModelBackend: The created backend instance
            
        Raises:
            ValueError: If backend type is unknown
        """
        if backend_type == BackendType.ANTHROPIC:
            from .anthropic_backend import AnthropicBackend
            return AnthropicBackend(
                api_key=self.config.anthropic_api_key,
                timeout=self.config.timeout_seconds
            )
        
        elif backend_type == BackendType.OPENAI:
            from .openai_backend import OpenAIBackend
            return OpenAIBackend(
                api_key=self.config.openai_api_key,
                timeout=self.config.timeout_seconds
            )
        
        elif backend_type == BackendType.LOCAL:
            from .local_backend import LocalBackend
            return LocalBackend(
                model_path=self.config.local_model_path,
                ollama_host=self.config.ollama_host,
                use_ollama=self.config.use_ollama,
                timeout=self.config.timeout_seconds
            )
        
        elif backend_type == BackendType.MOCK:
            from .mock_backend import MockBackend
            return MockBackend(delay_seconds=self.config.mock_delay_seconds)
        
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def register(self, backend: ModelBackend) -> None:
        """
        Register a backend instance.
        
        This allows manual registration of custom backends or
        pre-configured backend instances.
        
        Args:
            backend: The backend instance to register
        """
        self._backends[backend.backend_type] = backend
        self._health[backend.backend_type] = BackendHealth(
            backend_type=backend.backend_type,
            is_available=True,
            last_check=time.time(),
            error_count=0
        )
        logger.info(f"Registered backend: {backend.backend_type.value}")
    
    def get_backend(self, backend_type: BackendType) -> Optional[ModelBackend]:
        """
        Get a specific backend by type.
        
        Args:
            backend_type: The type of backend to retrieve
            
        Returns:
            Optional[ModelBackend]: The backend instance, or None if not registered
        """
        return self._backends.get(backend_type)
    
    def get_health(self, backend_type: BackendType) -> Optional[BackendHealth]:
        """
        Get health status for a specific backend.
        
        Args:
            backend_type: The type of backend
            
        Returns:
            Optional[BackendHealth]: Health status, or None if not registered
        """
        return self._health.get(backend_type)
    
    async def get_available_backend(self) -> Tuple[ModelBackend, BackendType]:
        """
        Get the first available backend from the fallback chain.
        
        Iterates through the primary backend and fallback chain to find
        the first backend that is available and healthy.
        
        Returns:
            Tuple[ModelBackend, BackendType]: The available backend and its type
            
        Raises:
            AllBackendsFailedError: If no backends are available
        """
        chain = [self.config.primary_backend] + self.config.fallback_chain
        
        for backend_type in chain:
            if backend_type in self._backends:
                backend = self._backends[backend_type]
                try:
                    if await backend.is_available():
                        return backend, backend_type
                except Exception as e:
                    logger.debug(f"Backend {backend_type.value} availability check failed: {e}")
                    continue
        
        raise AllBackendsFailedError(
            "No backends available",
            errors=[(b.value, "Not available") for b in chain if b in self._backends]
        )
    
    async def execute_with_fallback(
        self,
        prompt: str,
        **kwargs
    ) -> GenerateResult:
        """
        Execute generation with automatic fallback.
        
        Tries backends in order of the fallback chain until one succeeds.
        Implements retry logic with exponential backoff for transient errors.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments for generation (model_id, max_tokens, etc.)
            
        Returns:
            GenerateResult: The generation result
            
        Raises:
            AllBackendsFailedError: If all backends fail
        """
        chain = [self.config.primary_backend] + self.config.fallback_chain
        errors: List[Tuple[str, str]] = []
        
        for backend_type in chain:
            if backend_type not in self._backends:
                continue
            
            backend = self._backends[backend_type]
            health = self._health[backend_type]
            
            try:
                # Check availability
                if not await backend.is_available():
                    logger.debug(f"Backend {backend_type.value} not available, skipping")
                    continue
                
                # Execute with retry logic
                result = await self._execute_with_retry(
                    backend, prompt, **kwargs
                )
                
                # Reset error count on success
                health.error_count = 0
                health.successful_requests += 1
                health.total_requests += 1
                
                return result
            
            except BackendRateLimitError as e:
                # Don't retry rate limits - let fallback handle it
                errors.append((backend_type.value, str(e)))
                health.error_count += 1
                health.last_error = str(e)
                health.total_requests += 1
                logger.warning(
                    f"Backend {backend_type.value} rate limited: {e}. Trying next backend."
                )
            
            except Exception as e:
                errors.append((backend_type.value, str(e)))
                health.error_count += 1
                health.last_error = str(e)
                health.total_requests += 1
                logger.warning(
                    f"Backend {backend_type.value} failed: {e}. Trying next backend."
                )
        
        raise AllBackendsFailedError(
            "All backends failed",
            errors=errors
        )
    
    async def _execute_with_retry(
        self,
        backend: ModelBackend,
        prompt: str,
        **kwargs
    ) -> GenerateResult:
        """
        Execute with exponential backoff retry.
        
        Args:
            backend: The backend to use
            prompt: The input prompt
            **kwargs: Additional arguments for generation
            
        Returns:
            GenerateResult: The generation result
            
        Raises:
            BackendUnavailableError: If all retries fail
            BackendRateLimitError: If rate limited (not retried)
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await backend.generate(prompt, **kwargs)
            
            except BackendRateLimitError:
                # Don't retry rate limits - let fallback handle it
                raise
            
            except BackendUnavailableError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_exponential_base ** attempt
                    )
                    logger.info(f"Retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
        
        raise last_error or BackendUnavailableError("Retry failed")
    
    async def embed_with_fallback(
        self,
        text: str,
        **kwargs
    ) -> EmbedResult:
        """
        Execute embedding with automatic fallback.
        
        Tries backends in order of the fallback chain until one succeeds.
        Implements retry logic with exponential backoff for transient errors.
        
        Args:
            text: The text to embed
            **kwargs: Additional arguments for embedding (model_id, dimensions, etc.)
            
        Returns:
            EmbedResult: The embedding result
            
        Raises:
            AllBackendsFailedError: If all backends fail
        """
        chain = [self.config.primary_backend] + self.config.fallback_chain
        errors: List[Tuple[str, str]] = []
        
        for backend_type in chain:
            if backend_type not in self._backends:
                continue
            
            backend = self._backends[backend_type]
            health = self._health[backend_type]
            
            try:
                # Check availability
                if not await backend.is_available():
                    logger.debug(f"Backend {backend_type.value} not available for embedding, skipping")
                    continue
                
                # Execute embedding
                result = await backend.embed(text, **kwargs)
                
                # Reset error count on success
                health.error_count = 0
                health.successful_requests += 1
                health.total_requests += 1
                
                return result
            
            except BackendRateLimitError as e:
                # Don't retry rate limits - let fallback handle it
                errors.append((backend_type.value, str(e)))
                health.error_count += 1
                health.last_error = str(e)
                health.total_requests += 1
                logger.warning(
                    f"Backend {backend_type.value} rate limited during embedding: {e}. Trying next backend."
                )
            
            except Exception as e:
                errors.append((backend_type.value, str(e)))
                health.error_count += 1
                health.last_error = str(e)
                health.total_requests += 1
                logger.warning(
                    f"Backend {backend_type.value} embedding failed: {e}. Trying next backend."
                )
        
        raise AllBackendsFailedError(
            "All backends failed for embedding",
            errors=errors
        )
    
    async def health_check_all(self) -> Dict[BackendType, BackendHealth]:
        """
        Check health of all registered backends.
        
        Returns:
            Dict[BackendType, BackendHealth]: Health status for each backend
        """
        for backend_type, backend in self._backends.items():
            try:
                is_available = await backend.is_available()
                health = self._health[backend_type]
                health.is_available = is_available
                health.last_check = time.time()
            except Exception as e:
                health = self._health[backend_type]
                health.is_available = False
                health.last_error = str(e)
                health.last_check = time.time()
        
        return self._health.copy()
    
    async def close(self) -> None:
        """
        Close all backends and clean up resources.
        """
        for backend_type, backend in self._backends.items():
            try:
                await backend.close()
                logger.info(f"Closed backend: {backend_type.value}")
            except Exception as e:
                logger.warning(f"Error closing backend {backend_type.value}: {e}")
        
        self._backends.clear()
        self._initialized = False
    
    async def __aenter__(self) -> "BackendRegistry":
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.close()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall status of the registry.
        
        Returns:
            Dict with registry status including backend health
        """
        return {
            "initialized": self._initialized,
            "primary_backend": self.config.primary_backend.value,
            "fallback_chain": [b.value for b in self.config.fallback_chain],
            "backends": {
                bt.value: health.to_dict()
                for bt, health in self._health.items()
            }
        }