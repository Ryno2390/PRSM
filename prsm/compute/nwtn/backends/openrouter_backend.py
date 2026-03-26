#!/usr/bin/env python3
"""
OpenRouter Backend
==================

OpenRouter is an API aggregator that gives unified access to 200+ models
from every major provider (Anthropic, Google, Meta, Mistral, DeepSeek,
Qwen, etc.) through a single OpenAI-compatible endpoint.

Why this matters for PRSM
--------------------------
OpenRouter lets NWTN route each Agent Team task to the model best suited
for that specific job — not one expensive frontier model for everything:

    Interview mode      → cheap conversational model (e.g. Llama 3.1 8B)
    MetaPlanner         → structured output model (e.g. Gemini Flash)
    Nightly Synthesis   → quality narrative model (e.g. Claude Haiku)
    CheckpointReviewer  → code-specialized model (e.g. DeepSeek Coder)
    BSC Predictor       → runs locally on MPS (no API cost at all)

This is the core PRSM micro-model philosophy demonstrated in practice.

API compatibility
-----------------
OpenRouter is OpenAI-API-compatible (/chat/completions, same JSON format).
This backend extends OpenAIBackend, overriding only:
  - BASE_URL          → https://openrouter.ai/api/v1
  - backend_type      → BackendType.OPENROUTER
  - Extra headers     → HTTP-Referer, X-Title (OpenRouter attribution)
  - MODELS catalog    → the OpenRouter model universe

Configuration
-------------
Set the OPENROUTER_API_KEY environment variable, or pass api_key= directly.
The API key must start with "sk-or-v1-...".
"""

import os
import time
from typing import Any, Dict, List, Optional

from .base import BackendType, GenerateResult, TokenUsage
from .exceptions import (
    BackendAuthenticationError,
    BackendRateLimitError,
    BackendResponseError,
    BackendUnavailableError,
)
from .openai_backend import OpenAIBackend


# ── Curated model catalog with pricing (USD per 1K tokens) ────────────────
# These are the models recommended for Phase 10 NWTN tasks.
# OpenRouter supports 200+ models; this is a curated subset.
# Pricing from openrouter.ai/models (March 2026 — verify for current rates).

OPENROUTER_MODELS: Dict[str, Dict[str, Any]] = {
    # ── Ultra-cheap / free models ────────────────────────────────────────
    "meta-llama/llama-3.1-8b-instruct": {
        "display_name": "Llama 3.1 8B Instruct",
        "provider": "Meta",
        "context_window": 131072,
        "pricing": {"input": 0.000055, "output": 0.000055},
        "tier": "budget",
        "strengths": ["conversational", "interview", "fast"],
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "display_name": "Llama 3.2 3B Instruct",
        "provider": "Meta",
        "context_window": 131072,
        "pricing": {"input": 0.000015, "output": 0.000025},
        "tier": "budget",
        "strengths": ["conversational", "fast", "ultra-cheap"],
    },
    "mistralai/mistral-7b-instruct": {
        "display_name": "Mistral 7B Instruct",
        "provider": "Mistral",
        "context_window": 32768,
        "pricing": {"input": 0.000055, "output": 0.000055},
        "tier": "budget",
        "strengths": ["instruction-following", "json-output"],
    },
    "qwen/qwen-2.5-7b-instruct": {
        "display_name": "Qwen 2.5 7B Instruct",
        "provider": "Alibaba",
        "context_window": 131072,
        "pricing": {"input": 0.000072, "output": 0.000072},
        "tier": "budget",
        "strengths": ["multilingual", "coding", "reasoning"],
    },

    # ── Mid-tier models ─────────────────────────────────────────────────
    "google/gemini-flash-1.5": {
        "display_name": "Gemini Flash 1.5",
        "provider": "Google",
        "context_window": 1000000,
        "pricing": {"input": 0.000075, "output": 0.0003},
        "tier": "mid",
        "strengths": ["structured-output", "planning", "long-context"],
    },
    "google/gemini-flash-1.5-8b": {
        "display_name": "Gemini Flash 1.5 8B",
        "provider": "Google",
        "context_window": 1000000,
        "pricing": {"input": 0.0000375, "output": 0.00015},
        "tier": "budget",
        "strengths": ["structured-output", "fast", "cheap"],
    },
    "deepseek/deepseek-coder-v2-lite-instruct": {
        "display_name": "DeepSeek Coder V2 Lite",
        "provider": "DeepSeek",
        "context_window": 128000,
        "pricing": {"input": 0.00014, "output": 0.00028},
        "tier": "mid",
        "strengths": ["code-analysis", "diff-review", "security-audit"],
    },
    "deepseek/deepseek-chat-v3": {
        "display_name": "DeepSeek Chat V3",
        "provider": "DeepSeek",
        "context_window": 65536,
        "pricing": {"input": 0.00027, "output": 0.00110},
        "tier": "mid",
        "strengths": ["reasoning", "analysis", "cost-effective"],
    },
    "mistralai/mistral-small": {
        "display_name": "Mistral Small",
        "provider": "Mistral",
        "context_window": 32768,
        "pricing": {"input": 0.00020, "output": 0.00060},
        "tier": "mid",
        "strengths": ["instruction-following", "json-output", "fast"],
    },

    # ── Quality models ───────────────────────────────────────────────────
    "anthropic/claude-3-haiku": {
        "display_name": "Claude 3 Haiku",
        "provider": "Anthropic",
        "context_window": 200000,
        "pricing": {"input": 0.00025, "output": 0.00125},
        "tier": "quality",
        "strengths": ["narrative-writing", "synthesis", "clarity"],
    },
    "anthropic/claude-3.5-haiku": {
        "display_name": "Claude 3.5 Haiku",
        "provider": "Anthropic",
        "context_window": 200000,
        "pricing": {"input": 0.00080, "output": 0.00400},
        "tier": "quality",
        "strengths": ["reasoning", "coding", "synthesis", "fast"],
    },
    "google/gemini-pro-1.5": {
        "display_name": "Gemini Pro 1.5",
        "provider": "Google",
        "context_window": 2000000,
        "pricing": {"input": 0.00125, "output": 0.005},
        "tier": "quality",
        "strengths": ["planning", "reasoning", "long-context"],
    },
    "mistralai/mistral-large": {
        "display_name": "Mistral Large",
        "provider": "Mistral",
        "context_window": 128000,
        "pricing": {"input": 0.002, "output": 0.006},
        "tier": "quality",
        "strengths": ["reasoning", "planning", "analysis"],
    },

    # ── Premium models ───────────────────────────────────────────────────
    "anthropic/claude-3.5-sonnet": {
        "display_name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "context_window": 200000,
        "pricing": {"input": 0.003, "output": 0.015},
        "tier": "premium",
        "strengths": ["reasoning", "coding", "analysis", "best-quality"],
    },
    "anthropic/claude-opus-4": {
        "display_name": "Claude Opus 4",
        "provider": "Anthropic",
        "context_window": 200000,
        "pricing": {"input": 0.015, "output": 0.075},
        "tier": "ultra",
        "strengths": ["architecture", "planning", "highest-quality"],
    },
    "openai/gpt-4o": {
        "display_name": "GPT-4o",
        "provider": "OpenAI",
        "context_window": 128000,
        "pricing": {"input": 0.005, "output": 0.015},
        "tier": "premium",
        "strengths": ["general", "coding", "vision"],
    },
}

# Recommended model for each Phase 10 NWTN task (cost-optimised defaults)
TASK_MODEL_ROUTING: Dict[str, str] = {
    "interview":    "meta-llama/llama-3.1-8b-instruct",       # $0.02/1M — cheap conversational
    "planning":     "mistralai/mistral-small-3.1-24b-instruct",# $0.03/1M — structured JSON
    "synthesis":    "anthropic/claude-3-haiku",                # $0.25/1M — quality narrative
    "checkpoint":   "qwen/qwen2.5-coder-7b-instruct",          # $0.03/1M — code-specialized
    "general":      "google/gemini-2.0-flash-lite-001",        # $0.075/1M — budget general
}


# ── Backend implementation ─────────────────────────────────────────────────

class OpenRouterBackend(OpenAIBackend):
    """
    OpenRouter backend — 200+ models via a single OpenAI-compatible API.

    Extends OpenAIBackend with:
    - OpenRouter base URL
    - HTTP-Referer + X-Title headers (required by OpenRouter)
    - Full OpenRouter model catalog
    - Task-based model routing helpers

    Parameters
    ----------
    api_key : str, optional
        OpenRouter API key.  Falls back to ``OPENROUTER_API_KEY`` env var.
    default_model : str, optional
        Model to use when no specific model is requested.
        Defaults to ``google/gemini-flash-1.5-8b`` (cheap, capable).
    site_url : str
        Value for the HTTP-Referer header (your project URL).
    site_name : str
        Value for the X-Title header (your project name).
    timeout : int
        Request timeout in seconds (default 120).
    """

    BASE_URL    = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "google/gemini-flash-1.5-8b"
    MODELS = OPENROUTER_MODELS  # override the OpenAI model catalog

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        site_url: str = "https://github.com/Ryno2390/PRSM",
        site_name: str = "PRSM — Protocol for Recursive Scientific Modeling",
        timeout: int = 120,
        **kwargs,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        super().__init__(api_key=resolved_key, timeout=timeout, **kwargs)
        self.default_model = default_model
        self.site_url  = site_url
        self.site_name = site_name
        # Override base URL
        self.base_url = self.BASE_URL

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OPENROUTER

    @property
    def models_supported(self) -> List[str]:
        return list(OPENROUTER_MODELS.keys())

    async def initialize(self) -> None:
        """Initialize with OpenRouter-specific headers."""
        if self._initialized:
            return

        import aiohttp

        if not self.api_key:
            raise BackendAuthenticationError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key=.",
                backend_type=self.backend_type.value,
            )
        if not self.api_key.startswith("sk-or-"):
            raise BackendAuthenticationError(
                f"API key does not look like an OpenRouter key (should start with 'sk-or-'). "
                f"Got: {self.api_key[:12]}...",
                backend_type=self.backend_type.value,
            )

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "Authorization":  f"Bearer {self.api_key}",
                "Content-Type":   "application/json",
                "HTTP-Referer":   self.site_url,
                "X-Title":        self.site_name,
            },
        )
        self._initialized = True

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> GenerateResult:
        """
        Generate a response via OpenRouter.

        Parameters
        ----------
        prompt : str
        model : str, optional
            Any model ID from the OpenRouter catalog (e.g.
            ``"google/gemini-flash-1.5"``).  Defaults to
            ``self.default_model``.
        system_prompt : str, optional
        max_tokens : int
        temperature : float
        """
        if not self._initialized:
            await self.initialize()

        resolved_model = model or self.default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":      resolved_model,
            "messages":   messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.time()
        try:
            async with self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                if response.status == 401:
                    raise BackendAuthenticationError(
                        "OpenRouter authentication failed. Check your API key.",
                        backend_type=self.backend_type.value,
                    )
                if response.status == 429:
                    raise BackendRateLimitError(
                        "OpenRouter rate limit hit.",
                        backend_type=self.backend_type.value,
                    )
                if response.status != 200:
                    text = await response.text()
                    raise BackendResponseError(
                        f"OpenRouter HTTP {response.status}: {text[:300]}",
                        backend_type=self.backend_type.value,
                    )
                data = await response.json()

        except (BackendAuthenticationError, BackendRateLimitError, BackendResponseError):
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"OpenRouter request failed: {exc}",
                backend_type=self.backend_type.value,
            ) from exc

        elapsed = time.time() - t0

        choice  = data["choices"][0]
        content = choice["message"]["content"]
        usage   = data.get("usage", {})

        return GenerateResult(
            content=content,
            model_id=resolved_model,
            provider=self.backend_type,
            token_usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            execution_time=elapsed,
            finish_reason=choice.get("finish_reason", "stop"),
            metadata={"raw_response": data, "model": resolved_model},
        )

    # ── Convenience helpers ────────────────────────────────────────────────

    @classmethod
    def for_task(
        cls,
        task: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "OpenRouterBackend":
        """
        Return an ``OpenRouterBackend`` pre-configured for a specific NWTN task.

        Parameters
        ----------
        task : str
            One of: ``'interview'``, ``'planning'``, ``'synthesis'``,
            ``'checkpoint'``, ``'general'``.

        Example
        -------
        .. code-block:: python

            backend = OpenRouterBackend.for_task("synthesis")
            result  = await backend.generate("Summarise today's work…")
        """
        model = TASK_MODEL_ROUTING.get(task, TASK_MODEL_ROUTING["general"])
        return cls(api_key=api_key, default_model=model, **kwargs)

    def model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return the catalog entry for *model_id*, or None if unknown."""
        return OPENROUTER_MODELS.get(model_id)

    def models_by_tier(self, tier: str) -> List[str]:
        """Return all model IDs in the given pricing tier."""
        return [
            m for m, info in OPENROUTER_MODELS.items()
            if info.get("tier") == tier
        ]

    def estimated_cost_usd(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD for a given number of tokens."""
        info = OPENROUTER_MODELS.get(model_id, {})
        pricing = info.get("pricing", {"input": 0, "output": 0})
        return (
            (input_tokens / 1000) * pricing["input"]
            + (output_tokens / 1000) * pricing["output"]
        )
