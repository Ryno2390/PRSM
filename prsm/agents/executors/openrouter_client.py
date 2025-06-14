#!/usr/bin/env python3
"""
OpenRouter Unified API Client for PRSM
=====================================

Unified access to multiple AI providers (OpenAI, Anthropic, Google, etc.) 
through OpenRouter's single API with automatic failover and cost optimization.

🎯 KEY BENEFITS:
- Single API key for all major AI models
- Automatic failover between providers
- Transparent cost comparison
- No markup over direct API pricing
- Access to free open-source models

🔧 PRSM INTEGRATION:
- Replaces individual provider clients
- Maintains ModelExecutionRequest/Response interface
- Supports cost tracking and budget management
- Enables model performance benchmarking
"""

import asyncio
import aiohttp
import json
import time
from decimal import Decimal
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import structlog

# Import PRSM base classes
from .api_clients import (
    BaseModelClient,
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelProvider
)

logger = structlog.get_logger(__name__)


@dataclass
class OpenRouterModel:
    """OpenRouter model metadata"""
    id: str
    name: str
    provider: str
    pricing: Dict[str, float]  # input/output pricing per 1M tokens
    context_length: int
    supports_system_prompt: bool = True
    supports_streaming: bool = True


class OpenRouterClient(BaseModelClient):
    """Unified client for multiple AI providers via OpenRouter"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.total_cost = Decimal('0')
        self.request_count = 0
        
        # Popular model configurations
        self.models = {
            # OpenAI models
            "gpt-4-turbo": OpenRouterModel(
                id="openai/gpt-4-turbo",
                name="GPT-4 Turbo",
                provider="OpenAI",
                pricing={"input": 10.0, "output": 30.0},  # per 1M tokens
                context_length=128000
            ),
            "gpt-4": OpenRouterModel(
                id="openai/gpt-4",
                name="GPT-4",
                provider="OpenAI",
                pricing={"input": 30.0, "output": 60.0},
                context_length=8192
            ),
            "gpt-3.5-turbo": OpenRouterModel(
                id="openai/gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider="OpenAI",
                pricing={"input": 0.5, "output": 1.5},
                context_length=16385
            ),
            
            # Anthropic models
            "claude-3-opus": OpenRouterModel(
                id="anthropic/claude-3-opus",
                name="Claude 3 Opus",
                provider="Anthropic",
                pricing={"input": 15.0, "output": 75.0},
                context_length=200000
            ),
            "claude-3-sonnet": OpenRouterModel(
                id="anthropic/claude-3-sonnet",
                name="Claude 3 Sonnet",
                provider="Anthropic",
                pricing={"input": 3.0, "output": 15.0},
                context_length=200000
            ),
            "claude-3-haiku": OpenRouterModel(
                id="anthropic/claude-3-haiku",
                name="Claude 3 Haiku",
                provider="Anthropic",
                pricing={"input": 0.25, "output": 1.25},
                context_length=200000
            ),
            
            # Google models
            "gemini-pro": OpenRouterModel(
                id="google/gemini-pro",
                name="Gemini Pro",
                provider="Google",
                pricing={"input": 0.5, "output": 1.5},
                context_length=32768
            ),
            
            # Free models for testing
            "llama-3-8b": OpenRouterModel(
                id="meta-llama/llama-3-8b-instruct:free",
                name="Llama 3 8B (Free)",
                provider="Meta",
                pricing={"input": 0.0, "output": 0.0},
                context_length=8192
            ),
            "mixtral-8x7b": OpenRouterModel(
                id="mistralai/mixtral-8x7b-instruct:free",
                name="Mixtral 8x7B (Free)",
                provider="Mistral",
                pricing={"input": 0.0, "output": 0.0},
                context_length=32768
            )
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenRouter API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/PRSM-AI/prsm",
            "X-Title": "PRSM Protocol"
        }
    
    async def _setup_client(self) -> None:
        """OpenRouter-specific setup"""
        logger.info("OpenRouter client initialized", 
                   models_available=len(self.models))
    
    async def initialize(self) -> None:
        """Initialize the OpenRouter client"""
        self.session = aiohttp.ClientSession(
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=60)
        )
        await self._setup_client()
    
    async def close(self) -> None:
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    def get_model_info(self, model_key: str) -> Optional[OpenRouterModel]:
        """Get model information by key"""
        return self.models.get(model_key)
    
    def list_available_models(self) -> List[str]:
        """List all available model keys"""
        return list(self.models.keys())
    
    def get_cost_estimate(self, model_key: str, prompt_tokens: int, max_tokens: int) -> Decimal:
        """Estimate cost for a request"""
        model = self.models.get(model_key)
        if not model:
            return Decimal('0')
        
        input_cost = Decimal(str(model.pricing['input'])) * Decimal(str(prompt_tokens)) / Decimal('1000000')
        output_cost = Decimal(str(model.pricing['output'])) * Decimal(str(max_tokens)) / Decimal('1000000')
        
        return input_cost + output_cost
    
    async def execute(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """Execute a model request through OpenRouter"""
        start_time = time.time()
        
        try:
            # Get model info
            model = self.models.get(request.model_id)
            if not model:
                return ModelExecutionResponse(
                    content="",
                    provider=ModelProvider.OPENAI,  # Default fallback
                    model_id=request.model_id,
                    execution_time=time.time() - start_time,
                    token_usage={},
                    success=False,
                    error=f"Model '{request.model_id}' not available. Use list_available_models() to see options."
                )
            
            # Build request payload
            messages = []
            if request.system_prompt and model.supports_system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": request.prompt
            })
            
            payload = {
                "model": model.id,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
            
            logger.info("Sending request to OpenRouter",
                       model=model.name,
                       provider=model.provider,
                       tokens_requested=request.max_tokens)
            
            # Make API request
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                
                response_data = await response.json()
                
                if response.status != 200:
                    error_msg = response_data.get('error', {}).get('message', f"HTTP {response.status}")
                    return ModelExecutionResponse(
                        content="",
                        provider=ModelProvider.OPENAI,
                        model_id=request.model_id,
                        execution_time=time.time() - start_time,
                        token_usage={},
                        success=False,
                        error=error_msg
                    )
                
                # Extract response
                content = response_data['choices'][0]['message']['content']
                usage = response_data.get('usage', {})
                
                # Calculate cost
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                
                cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
                self.total_cost += cost
                self.request_count += 1
                
                execution_time = time.time() - start_time
                
                logger.info("OpenRouter request completed",
                           model=model.name,
                           provider=model.provider,
                           tokens=total_tokens,
                           cost_usd=float(cost),
                           latency_ms=execution_time * 1000)
                
                return ModelExecutionResponse(
                    content=content,
                    provider=ModelProvider.OPENAI,  # Unified under OpenRouter
                    model_id=request.model_id,
                    execution_time=execution_time,
                    token_usage={
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens
                    },
                    success=True,
                    metadata={
                        'actual_provider': model.provider,
                        'actual_model_id': model.id,
                        'cost_usd': float(cost),
                        'openrouter_routing': True
                    }
                )
                
        except Exception as e:
            logger.error("OpenRouter request failed", error=str(e))
            return ModelExecutionResponse(
                content="",
                provider=ModelProvider.OPENAI,
                model_id=request.model_id,
                execution_time=time.time() - start_time,
                token_usage={},
                success=False,
                error=f"Request failed: {str(e)}"
            )
    
    def _calculate_cost(self, model: OpenRouterModel, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """Calculate precise cost for a request"""
        input_cost = Decimal(str(model.pricing['input'])) * Decimal(str(prompt_tokens)) / Decimal('1000000')
        output_cost = Decimal(str(model.pricing['output'])) * Decimal(str(completion_tokens)) / Decimal('1000000')
        return input_cost + output_cost
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'total_requests': self.request_count,
            'total_cost_usd': float(self.total_cost),
            'avg_cost_per_request': float(self.total_cost / max(self.request_count, 1)),
            'available_models': len(self.models)
        }
