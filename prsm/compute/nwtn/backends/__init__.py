#!/usr/bin/env python3
"""
NWTN LLM Backend Integration
============================

This module provides a unified interface for multiple LLM backends,
enabling the NWTN pipeline to use real LLM API calls instead of
hardcoded mock outputs.

Supported Backends:
- Anthropic (Claude API)
- OpenAI (GPT API)
- Local (Ollama/Transformers)
- Mock (Testing)

Usage:
    from prsm.compute.nwtn.backends import BackendRegistry, BackendConfig
    
    config = BackendConfig.from_environment()
    registry = BackendRegistry(config)
    await registry.initialize()
    
    result = await registry.execute_with_fallback(
        prompt="What is the meaning of life?",
        system_prompt="You are a philosophical advisor."
    )
    print(result.content)
"""

from .base import (
    ModelBackend,
    BackendType,
    GenerateResult,
    EmbedResult,
    TokenUsage,
)

from .config import BackendConfig

from .registry import BackendRegistry, BackendHealth

from .exceptions import (
    BackendError,
    BackendUnavailableError,
    BackendTimeoutError,
    BackendRateLimitError,
    BackendAuthenticationError,
    BackendResponseError,
    AllBackendsFailedError,
    ModelNotFoundError,
)

__all__ = [
    # Base classes and types
    "ModelBackend",
    "BackendType",
    "GenerateResult",
    "EmbedResult",
    "TokenUsage",
    
    # Configuration
    "BackendConfig",
    
    # Registry
    "BackendRegistry",
    "BackendHealth",
    
    # Exceptions
    "BackendError",
    "BackendUnavailableError",
    "BackendTimeoutError",
    "BackendRateLimitError",
    "BackendAuthenticationError",
    "BackendResponseError",
    "AllBackendsFailedError",
    "ModelNotFoundError",
]