#!/usr/bin/env python3
"""
OpenRouter Unified API Client for PRSM - Enterprise Edition
========================================================

Advanced multi-provider AI model access with intelligent routing, cost optimization,
and enterprise-grade reliability features for PRSM's Tier 2 development.

🎯 KEY FEATURES:
- Access to 100+ AI models from multiple providers
- Intelligent model selection based on cost/quality/latency
- Real-time provider health monitoring and failover
- Advanced cost optimization algorithms
- Dynamic pricing and availability data
- Integration with PRSM routing system

🔧 TIER 2 CAPABILITIES:
- Multi-provider load balancing
- Performance benchmarking and analytics
- Budget management and cost controls
- Automatic failover and retry logic
- Model performance profiling
- Enterprise reliability features
"""

import asyncio
import aiohttp
import json
import time
import statistics
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import structlog
import random

# Import PRSM base classes
from .api_clients import (
    BaseModelClient,
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelProvider
)
from ...config.model_config_manager import get_model_config_manager, ModelConfiguration

logger = structlog.get_logger(__name__)


class ModelTier(Enum):
    """Model quality/capability tiers"""
    FREE = "free"
    BASIC = "basic"
    ADVANCED = "advanced"
    PREMIUM = "premium"
    ULTRA = "ultra"


class ProviderStatus(Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ModelCapabilities:
    """Detailed model capabilities"""
    supports_system_prompt: bool = True
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_code: bool = True
    supports_json_mode: bool = False
    max_output_tokens: int = 4096


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model selection"""
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    quality_score: float = 0.8
    cost_efficiency: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    error_count: int = 0


@dataclass
class ProviderHealth:
    """Provider health status and metrics"""
    status: ProviderStatus = ProviderStatus.HEALTHY
    last_check: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0
    consecutive_failures: int = 0


@dataclass
class OpenRouterModel:
    """Enhanced OpenRouter model metadata"""
    id: str
    name: str
    provider: str
    tier: ModelTier
    pricing: Dict[str, float]  # input/output pricing per 1M tokens
    context_length: int
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    performance: ModelPerformanceMetrics = field(default_factory=ModelPerformanceMetrics)
    is_available: bool = True
    popularity_score: float = 0.5
    description: str = ""


class ModelSelector:
    """Intelligent model selection based on multiple criteria"""
    
    def __init__(self):
        self.selection_weights = {
            'cost': 0.3,
            'quality': 0.25,
            'latency': 0.2,
            'success_rate': 0.15,
            'availability': 0.1
        }
    
    def select_best_model(
        self,
        models: Dict[str, OpenRouterModel],
        criteria: Dict[str, Any],
        budget_limit: Optional[float] = None
    ) -> Optional[str]:
        """Select the best model based on criteria and constraints"""
        
        if not models:
            return None
        
        # Filter available models
        available_models = {k: v for k, v in models.items() if v.is_available}
        
        if not available_models:
            return None
        
        # Apply budget filter if specified
        if budget_limit:
            available_models = {
                k: v for k, v in available_models.items()
                if self._estimate_cost(v, criteria.get('expected_tokens', 1000)) <= budget_limit
            }
        
        if not available_models:
            logger.warning("No models available within budget", budget=budget_limit)
            return None
        
        # Score each model
        scored_models = []
        for model_key, model in available_models.items():
            score = self._calculate_model_score(model, criteria)
            scored_models.append((model_key, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Model selection completed",
                   selected=scored_models[0][0],
                   score=scored_models[0][1],
                   candidates=len(scored_models))
        
        return scored_models[0][0]
    
    def _calculate_model_score(self, model: OpenRouterModel, criteria: Dict[str, Any]) -> float:
        """Calculate composite score for model selection"""
        
        # Cost score (lower cost = higher score)
        expected_tokens = criteria.get('expected_tokens', 1000)
        cost = self._estimate_cost(model, expected_tokens)
        cost_score = 1.0 / (1.0 + float(cost) * 1000)  # Normalize cost impact
        
        # Quality score from performance metrics
        quality_score = model.performance.quality_score
        
        # Latency score (lower latency = higher score)
        latency_score = 1.0 / (1.0 + model.performance.avg_latency_ms / 1000)
        
        # Success rate score
        success_score = model.performance.success_rate
        
        # Availability score
        availability_score = 1.0 if model.is_available else 0.0
        
        # Weighted composite score
        composite_score = (
            self.selection_weights['cost'] * cost_score +
            self.selection_weights['quality'] * quality_score +
            self.selection_weights['latency'] * latency_score +
            self.selection_weights['success_rate'] * success_score +
            self.selection_weights['availability'] * availability_score
        )
        
        return composite_score
    
    def _estimate_cost(self, model: OpenRouterModel, tokens: int) -> Decimal:
        """Estimate cost for model with given token count"""
        input_cost = Decimal(str(model.pricing['input'])) * Decimal(str(tokens)) / Decimal('1000000')
        return input_cost


class CostOptimizer:
    """Advanced cost optimization algorithms"""
    
    def __init__(self):
        self.cost_history = []
        self.budget_alerts = []
    
    def optimize_request(
        self,
        request: ModelExecutionRequest,
        available_models: Dict[str, OpenRouterModel],
        budget_constraint: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Optimize request for cost efficiency"""
        
        # Estimate token usage
        estimated_tokens = self._estimate_token_usage(request.prompt)
        
        # Find cost-efficient models
        efficient_models = self._find_efficient_models(
            available_models, estimated_tokens, budget_constraint
        )
        
        if not efficient_models:
            # Fallback to cheapest available model
            cheapest = min(
                available_models.items(),
                key=lambda x: x[1].pricing['input'] + x[1].pricing['output']
            )
            return cheapest[0], {'cost_optimized': True, 'fallback_reason': 'budget_constraint'}
        
        # Select best efficient model
        best_model = max(efficient_models.items(), key=lambda x: x[1].performance.cost_efficiency)
        
        return best_model[0], {
            'cost_optimized': True,
            'estimated_cost': self._calculate_cost(best_model[1], estimated_tokens),
            'alternatives_considered': len(efficient_models)
        }
    
    def _estimate_token_usage(self, prompt: str) -> int:
        """Estimate token usage from prompt"""
        # Simple estimation: ~4 characters per token
        return max(len(prompt) // 4, 100)
    
    def _find_efficient_models(
        self,
        models: Dict[str, OpenRouterModel],
        tokens: int,
        budget: Optional[float]
    ) -> Dict[str, OpenRouterModel]:
        """Find models that offer good cost efficiency"""
        
        efficient = {}
        
        for key, model in models.items():
            if not model.is_available:
                continue
            
            cost = self._calculate_cost(model, tokens)
            
            if budget and float(cost) > budget:
                continue
            
            # Calculate efficiency score
            efficiency = model.performance.quality_score / max(float(cost), 0.001)
            
            if efficiency > 0.1:  # Minimum efficiency threshold
                efficient[key] = model
        
        return efficient
    
    def _calculate_cost(self, model: OpenRouterModel, tokens: int) -> Decimal:
        """Calculate cost for model and token count"""
        input_cost = Decimal(str(model.pricing['input'])) * Decimal(str(tokens)) / Decimal('1000000')
        output_cost = Decimal(str(model.pricing['output'])) * Decimal(str(tokens)) / Decimal('1000000')
        return input_cost + output_cost


class HealthMonitor:
    """Provider health monitoring and failover management"""
    
    def __init__(self):
        self.provider_health: Dict[str, ProviderHealth] = {}
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.now()
    
    async def check_provider_health(self, session: aiohttp.ClientSession, base_url: str) -> ProviderHealth:
        """Check health of a specific provider"""
        start_time = time.time()
        
        try:
            # Simple health check via models endpoint
            async with session.get(f"{base_url}/models", timeout=aiohttp.ClientTimeout(total=10)) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    return ProviderHealth(
                        status=ProviderStatus.HEALTHY,
                        response_time_ms=response_time,
                        last_check=datetime.now()
                    )
                else:
                    return ProviderHealth(
                        status=ProviderStatus.DEGRADED,
                        response_time_ms=response_time,
                        last_check=datetime.now(),
                        error_rate=0.1
                    )
        
        except asyncio.TimeoutError:
            return ProviderHealth(
                status=ProviderStatus.DEGRADED,
                response_time_ms=10000,
                last_check=datetime.now(),
                error_rate=0.2
            )
        
        except Exception as e:
            logger.warning("Provider health check failed", error=str(e))
            return ProviderHealth(
                status=ProviderStatus.UNHEALTHY,
                response_time_ms=0,
                last_check=datetime.now(),
                error_rate=1.0
            )
    
    def update_provider_health(self, provider: str, success: bool, response_time: float):
        """Update provider health based on request outcome"""
        if provider not in self.provider_health:
            self.provider_health[provider] = ProviderHealth()
        
        health = self.provider_health[provider]
        
        if success:
            health.consecutive_failures = 0
            health.error_rate = max(0, health.error_rate - 0.05)
        else:
            health.consecutive_failures += 1
            health.error_rate = min(1.0, health.error_rate + 0.1)
        
        health.response_time_ms = response_time * 1000
        health.last_check = datetime.now()
        
        # Update status based on metrics
        if health.consecutive_failures >= 5:
            health.status = ProviderStatus.OFFLINE
        elif health.error_rate > 0.3:
            health.status = ProviderStatus.UNHEALTHY
        elif health.error_rate > 0.1 or health.response_time_ms > 5000:
            health.status = ProviderStatus.DEGRADED
        else:
            health.status = ProviderStatus.HEALTHY
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers"""
        healthy = []
        for provider, health in self.provider_health.items():
            if health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                healthy.append(provider)
        return healthy


class OpenRouterClient(BaseModelClient):
    """Enterprise-grade OpenRouter client with advanced features"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.total_cost = Decimal('0')
        self.request_count = 0
        
        # Advanced components
        self.model_selector = ModelSelector()
        self.cost_optimizer = CostOptimizer()
        self.health_monitor = HealthMonitor()
        
        # Configuration
        self.auto_failover = True
        self.max_retries = 3
        self.retry_delay = 1.0
        self.budget_limit: Optional[float] = None
        
        # Initialize model catalog from ModelConfigManager
        self.config_manager = get_model_config_manager()
        self.models = self._load_models_from_config()
        
        # Performance tracking
        self.performance_history = []
        self.last_models_update = datetime.now()
    
    def _load_models_from_config(self) -> Dict[str, OpenRouterModel]:
        """Load model catalog from ModelConfigManager"""
        models = {}
        
        # Get all models from configuration
        all_models = self.config_manager.get_all_models()
        
        for model_id, model_config in all_models.items():
            # Convert ModelConfiguration to OpenRouterModel
            openrouter_model = self._convert_to_openrouter_model(model_config)
            if openrouter_model:
                models[model_id] = openrouter_model
        
        logger.info("Model catalog loaded from configuration", total_models=len(models))
        return models
    
    def _convert_to_openrouter_model(self, model_config: ModelConfiguration) -> Optional['OpenRouterModel']:
        """Convert ModelConfiguration to OpenRouterModel format"""
        try:
            # Map ModelTier from config to OpenRouter ModelTier
            tier_mapping = {
                "free": ModelTier.FREE,
                "basic": ModelTier.BASIC,
                "premium": ModelTier.PREMIUM,
                "enterprise": ModelTier.ULTRA
            }
            
            tier = tier_mapping.get(model_config.tier.value, ModelTier.BASIC)
            
            # Convert pricing (config uses per-1k tokens, OpenRouter expects different format)
            pricing = {
                "input": float(model_config.pricing.input_cost_per_1k),
                "output": float(model_config.pricing.output_cost_per_1k)
            }
            
            # Convert capabilities
            capabilities = ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=model_config.supports_streaming,
                supports_tools=model_config.supports_tools,
                supports_vision=model_config.supports_vision,
                supports_code="code_generation" in [cap.value for cap in model_config.capabilities],
                max_output_tokens=model_config.max_tokens
            )
            
            # Create OpenRouterModel
            return OpenRouterModel(
                id=f"{model_config.provider}/{model_config.id}",
                name=model_config.name,
                provider=model_config.provider.title(),
                tier=tier,
                pricing=pricing,
                context_length=model_config.context_length,
                capabilities=capabilities,
                description=f"{model_config.name} - {model_config.provider} model"
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert model {model_config.id}: {e}")
            return None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenRouter API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/PRSM-AI/prsm",
            "X-Title": "PRSM Protocol - Enterprise Edition"
        }
    
    async def _setup_client(self) -> None:
        """Enhanced setup with health monitoring"""
        logger.info("Initializing OpenRouter enterprise client",
                   models_available=len(self.models),
                   features=['intelligent_selection', 'cost_optimization', 'health_monitoring'])
    
    async def initialize(self) -> None:
                tier=ModelTier.PREMIUM,
                pricing={"input": 10.0, "output": 30.0},
                context_length=128000,
                capabilities=ModelCapabilities(supports_tools=True, supports_vision=True),
                description="Most capable GPT-4 model with 128k context"
            ),
            "gpt-4": OpenRouterModel(
                id="openai/gpt-4",
                name="GPT-4",
                provider="OpenAI",
                tier=ModelTier.PREMIUM,
                pricing={"input": 30.0, "output": 60.0},
                context_length=8192,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Original GPT-4 model with excellent reasoning"
            ),
            "gpt-4-vision": OpenRouterModel(
                id="openai/gpt-4-vision-preview",
                name="GPT-4 Vision",
                provider="OpenAI",
                tier=ModelTier.ULTRA,
                pricing={"input": 10.0, "output": 30.0},
                context_length=128000,
                capabilities=ModelCapabilities(supports_tools=True, supports_vision=True),
                description="GPT-4 with vision capabilities"
            ),
            "gpt-3.5-turbo": OpenRouterModel(
                id="openai/gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider="OpenAI",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.5, "output": 1.5},
                context_length=16385,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Fast and efficient general-purpose model"
            )
        }
        
        # Anthropic Models
        anthropic_models = {
            "claude-3-opus": OpenRouterModel(
                id="anthropic/claude-3-opus",
                name="Claude 3 Opus",
                provider="Anthropic",
                tier=ModelTier.ULTRA,
                pricing={"input": 15.0, "output": 75.0},
                context_length=200000,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Most capable Claude model for complex tasks"
            ),
            "claude-3-sonnet": OpenRouterModel(
                id="anthropic/claude-3-sonnet",
                name="Claude 3 Sonnet",
                provider="Anthropic",
                tier=ModelTier.PREMIUM,
                pricing={"input": 3.0, "output": 15.0},
                context_length=200000,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Balanced Claude model for most use cases"
            ),
            "claude-3-haiku": OpenRouterModel(
                id="anthropic/claude-3-haiku",
                name="Claude 3 Haiku",
                provider="Anthropic",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.25, "output": 1.25},
                context_length=200000,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Fastest Claude model for quick tasks"
            )
        }
        
        # Google Models
        google_models = {
            "gemini-pro": OpenRouterModel(
                id="google/gemini-pro",
                name="Gemini Pro",
                provider="Google",
                tier=ModelTier.PREMIUM,
                pricing={"input": 0.5, "output": 1.5},
                context_length=32768,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Google's most capable model"
            ),
            "gemini-pro-vision": OpenRouterModel(
                id="google/gemini-pro-vision",
                name="Gemini Pro Vision",
                provider="Google",
                tier=ModelTier.PREMIUM,
                pricing={"input": 0.5, "output": 1.5},
                context_length=32768,
                capabilities=ModelCapabilities(supports_tools=True, supports_vision=True),
                description="Gemini with vision capabilities"
            )
        }
        
        # Meta/Llama Models
        meta_models = {
            "llama-3-70b": OpenRouterModel(
                id="meta-llama/llama-3-70b-instruct",
                name="Llama 3 70B",
                provider="Meta",
                tier=ModelTier.PREMIUM,
                pricing={"input": 0.9, "output": 0.9},
                context_length=8192,
                description="Large open-source model with strong performance"
            ),
            "llama-3-8b": OpenRouterModel(
                id="meta-llama/llama-3-8b-instruct",
                name="Llama 3 8B",
                provider="Meta",
                tier=ModelTier.BASIC,
                pricing={"input": 0.2, "output": 0.2},
                context_length=8192,
                description="Efficient open-source model"
            ),
            "llama-3-8b-free": OpenRouterModel(
                id="meta-llama/llama-3-8b-instruct:free",
                name="Llama 3 8B (Free)",
                provider="Meta",
                tier=ModelTier.FREE,
                pricing={"input": 0.0, "output": 0.0},
                context_length=8192,
                description="Free version of Llama 3 8B"
            )
        }
        
        # Mistral Models
        mistral_models = {
            "mixtral-8x7b": OpenRouterModel(
                id="mistralai/mixtral-8x7b-instruct",
                name="Mixtral 8x7B",
                provider="Mistral",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.24, "output": 0.24},
                context_length=32768,
                description="Mixture-of-experts model with strong performance"
            ),
            "mixtral-8x7b-free": OpenRouterModel(
                id="mistralai/mixtral-8x7b-instruct:free",
                name="Mixtral 8x7B (Free)",
                provider="Mistral",
                tier=ModelTier.FREE,
                pricing={"input": 0.0, "output": 0.0},
                context_length=32768,
                description="Free version of Mixtral 8x7B"
            ),
            "mistral-7b": OpenRouterModel(
                id="mistralai/mistral-7b-instruct",
                name="Mistral 7B",
                provider="Mistral",
                tier=ModelTier.BASIC,
                pricing={"input": 0.13, "output": 0.13},
                context_length=8192,
                description="Efficient instruction-following model"
            )
        }
        
        # Cohere Models
        cohere_models = {
            "command-r-plus": OpenRouterModel(
                id="cohere/command-r-plus",
                name="Command R+",
                provider="Cohere",
                tier=ModelTier.PREMIUM,
                pricing={"input": 3.0, "output": 15.0},
                context_length=128000,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Cohere's most capable model"
            ),
            "command-r": OpenRouterModel(
                id="cohere/command-r",
                name="Command R",
                provider="Cohere",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.5, "output": 1.5},
                context_length=128000,
                capabilities=ModelCapabilities(supports_tools=True),
                description="Balanced model for general use"
            )
        }
        
        # Specialized Models
        specialized_models = {
            "dolphin-mixtral": OpenRouterModel(
                id="cognitivecomputations/dolphin-mixtral-8x7b",
                name="Dolphin Mixtral 8x7B",
                provider="Cognitive Computations",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.5, "output": 0.5},
                context_length=32768,
                description="Uncensored model based on Mixtral"
            ),
            "nous-hermes-mixtral": OpenRouterModel(
                id="nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
                name="Nous Hermes 2 Mixtral 8x7B",
                provider="Nous Research",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.27, "output": 0.27},
                context_length=32768,
                description="Fine-tuned Mixtral for better instruction following"
            ),
            "yi-34b": OpenRouterModel(
                id="01-ai/yi-34b-chat",
                name="Yi 34B Chat",
                provider="01.AI",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.72, "output": 0.72},
                context_length=4096,
                description="High-quality Chinese-English bilingual model"
            )
        }
        
        # Code-specialized models
        code_models = {
            "codellama-34b": OpenRouterModel(
                id="codellama/codellama-34b-instruct",
                name="CodeLlama 34B",
                provider="Meta",
                tier=ModelTier.ADVANCED,
                pricing={"input": 0.72, "output": 0.72},
                context_length=16384,
                capabilities=ModelCapabilities(supports_code=True),
                description="Specialized for code generation and analysis"
            ),
            "deepseek-coder": OpenRouterModel(
                id="deepseek/deepseek-coder-6.7b-instruct",
                name="DeepSeek Coder 6.7B",
                provider="DeepSeek",
                tier=ModelTier.BASIC,
                pricing={"input": 0.14, "output": 0.28},
                context_length=16384,
                capabilities=ModelCapabilities(supports_code=True),
                description="Efficient code-focused model"
            )
        }
        
        # Combine all models
        models.update(openai_models)
        models.update(anthropic_models)
        models.update(google_models)
        models.update(meta_models)
        models.update(mistral_models)
        models.update(cohere_models)
        models.update(specialized_models)
        models.update(code_models)
        
        logger.info("Model catalog initialized", total_models=len(models))
        return models
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenRouter API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/PRSM-AI/prsm",
            "X-Title": "PRSM Protocol - Enterprise Edition"
        }
    
    async def _setup_client(self) -> None:
        """Enhanced setup with health monitoring"""
        logger.info("Initializing OpenRouter enterprise client",
                   models_available=len(self.models),
                   features=['intelligent_selection', 'cost_optimization', 'health_monitoring'])
        
        # Initial health check
        await self._update_model_availability()
    
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
    
    async def _update_model_availability(self) -> None:
        """Update model availability from OpenRouter API"""
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    available_model_ids = {model['id'] for model in data.get('data', [])}
                    
                    # Update availability status
                    for model_key, model in self.models.items():
                        model.is_available = model.id in available_model_ids
                    
                    self.last_models_update = datetime.now()
                    logger.info("Model availability updated",
                               total_models=len(self.models),
                               available_models=sum(1 for m in self.models.values() if m.is_available))
        
        except Exception as e:
            logger.warning("Failed to update model availability", error=str(e))
    
    def get_model_info(self, model_key: str) -> Optional[OpenRouterModel]:
        """Get model information by key"""
        return self.models.get(model_key)
    
    def list_available_models(self, tier: Optional[ModelTier] = None, provider: Optional[str] = None) -> List[str]:
        """List available models with optional filtering"""
        models = []
        for key, model in self.models.items():
            if not model.is_available:
                continue
            if tier and model.tier != tier:
                continue
            if provider and model.provider.lower() != provider.lower():
                continue
            models.append(key)
        return sorted(models)
    
    def get_models_by_tier(self, tier: ModelTier) -> Dict[str, OpenRouterModel]:
        """Get all models in a specific tier"""
        return {k: v for k, v in self.models.items() if v.tier == tier and v.is_available}
    
    def get_cost_estimate(self, model_key: str, prompt_tokens: int, max_tokens: int) -> Decimal:
        """Estimate cost for a request"""
        model = self.models.get(model_key)
        if not model:
            return Decimal('0')
        
        input_cost = Decimal(str(model.pricing['input'])) * Decimal(str(prompt_tokens)) / Decimal('1000000')
        output_cost = Decimal(str(model.pricing['output'])) * Decimal(str(max_tokens)) / Decimal('1000000')
        
        return input_cost + output_cost
    
    def set_budget_limit(self, limit_usd: float) -> None:
        """Set budget limit for cost optimization"""
        self.budget_limit = limit_usd
        logger.info("Budget limit set", limit_usd=limit_usd)
    
    async def execute(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """Execute a model request with advanced features"""
        start_time = time.time()
        
        # Check if we need to update model availability
        if datetime.now() - self.last_models_update > timedelta(hours=1):
            await self._update_model_availability()
        
        # Intelligent model selection if not specified or if optimization is requested
        selected_model_key = request.model_id
        optimization_metadata = {}
        
        if request.model_id == "auto" or request.metadata.get('optimize', False):
            criteria = {
                'expected_tokens': len(request.prompt) // 4,
                'priority': request.metadata.get('priority', 'balanced'),
                'max_latency': request.metadata.get('max_latency_ms', 5000)
            }
            
            selected_model_key = self.model_selector.select_best_model(
                self.models, criteria, self.budget_limit
            )
            
            if not selected_model_key:
                return ModelExecutionResponse(
                    content="",
                    provider=ModelProvider.OPENAI,
                    model_id=request.model_id,
                    execution_time=time.time() - start_time,
                    token_usage={},
                    success=False,
                    error="No suitable models available within constraints"
                )
            
            optimization_metadata['auto_selected'] = True
            optimization_metadata['selection_criteria'] = criteria
        
        # Cost optimization
        if self.budget_limit or request.metadata.get('cost_optimize', False):
            optimized_model, opt_meta = self.cost_optimizer.optimize_request(
                request, self.models, self.budget_limit
            )
            if optimized_model != selected_model_key:
                selected_model_key = optimized_model
                optimization_metadata.update(opt_meta)
        
        # Execute with retry logic and failover
        for attempt in range(self.max_retries):
            try:
                response = await self._execute_single_request(request, selected_model_key, start_time)
                
                # Update performance metrics
                self._update_performance_metrics(selected_model_key, response, time.time() - start_time)
                
                # Add optimization metadata
                if optimization_metadata:
                    response.metadata.update(optimization_metadata)
                
                return response
            
            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed", 
                              model=selected_model_key, error=str(e))
                
                if attempt < self.max_retries - 1 and self.auto_failover:
                    # Try failover to alternative model
                    alternative = self._get_failover_model(selected_model_key)
                    if alternative:
                        selected_model_key = alternative
                        logger.info("Failing over to alternative model", new_model=alternative)
                        await asyncio.sleep(self.retry_delay)
                        continue
                
                # Final attempt failed
                return ModelExecutionResponse(
                    content="",
                    provider=ModelProvider.OPENAI,
                    model_id=request.model_id,
                    execution_time=time.time() - start_time,
                    token_usage={},
                    success=False,
                    error=f"Request failed after {self.max_retries} attempts: {str(e)}"
                )
    
    async def _execute_single_request(
        self, 
        request: ModelExecutionRequest, 
        model_key: str, 
        start_time: float
    ) -> ModelExecutionResponse:
        """Execute a single request to OpenRouter"""
        
        model = self.models.get(model_key)
        if not model:
            raise ValueError(f"Model '{model_key}' not found")
        
        # Build request payload
        messages = []
        if request.system_prompt and model.capabilities.supports_system_prompt:
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
        
        # Add optional parameters
        if model.capabilities.supports_json_mode and request.metadata.get('json_mode', False):
            payload["response_format"] = {"type": "json_object"}
        
        if model.capabilities.supports_tools and request.metadata.get('tools'):
            payload["tools"] = request.metadata['tools']
        
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
                raise Exception(error_msg)
            
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
            
            # Update provider health
            self.health_monitor.update_provider_health(model.provider, True, execution_time)
            
            logger.info("OpenRouter request completed",
                       model=model.name,
                       provider=model.provider,
                       tokens=total_tokens,
                       cost_usd=float(cost),
                       latency_ms=execution_time * 1000)
            
            return ModelExecutionResponse(
                content=content,
                provider=ModelProvider.OPENAI,  # Unified under OpenRouter
                model_id=model_key,
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
                    'openrouter_routing': True,
                    'model_tier': model.tier.value,
                    'provider_health': self.health_monitor.provider_health.get(model.provider, {})
                }
            )
    
    def _get_failover_model(self, failed_model_key: str) -> Optional[str]:
        """Get alternative model for failover"""
        failed_model = self.models.get(failed_model_key)
        if not failed_model:
            return None
        
        # Find models from different providers with similar capabilities
        alternatives = []
        for key, model in self.models.items():
            if (model.is_available and 
                model.provider != failed_model.provider and
                model.tier == failed_model.tier):
                alternatives.append(key)
        
        if alternatives:
            return random.choice(alternatives)
        
        # Fallback to any available model
        available = [k for k, v in self.models.items() if v.is_available]
        return random.choice(available) if available else None
    
    def _update_performance_metrics(self, model_key: str, response: ModelExecutionResponse, latency: float):
        """Update performance metrics for a model"""
        model = self.models.get(model_key)
        if not model:
            return
        
        perf = model.performance
        
        # Update running averages
        if perf.request_count == 0:
            perf.avg_latency_ms = latency * 1000
        else:
            # Exponential moving average
            alpha = 0.1
            perf.avg_latency_ms = alpha * (latency * 1000) + (1 - alpha) * perf.avg_latency_ms
        
        perf.request_count += 1
        
        if response.success:
            perf.success_rate = (perf.success_rate * (perf.request_count - 1) + 1.0) / perf.request_count
        else:
            perf.error_count += 1
            perf.success_rate = (perf.success_rate * (perf.request_count - 1)) / perf.request_count
        
        # Update cost efficiency (quality/cost ratio)
        if response.metadata.get('cost_usd', 0) > 0:
            efficiency = perf.quality_score / response.metadata['cost_usd']
            perf.cost_efficiency = (perf.cost_efficiency + efficiency) / 2
        
        perf.last_updated = datetime.now()
    
    def _calculate_cost(self, model: OpenRouterModel, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """Calculate precise cost for a request"""
        input_cost = Decimal(str(model.pricing['input'])) * Decimal(str(prompt_tokens)) / Decimal('1000000')
        output_cost = Decimal(str(model.pricing['output'])) * Decimal(str(completion_tokens)) / Decimal('1000000')
        return input_cost + output_cost
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        provider_stats = {}
        for provider, health in self.health_monitor.provider_health.items():
            provider_stats[provider] = {
                'status': health.status.value,
                'response_time_ms': health.response_time_ms,
                'error_rate': health.error_rate,
                'uptime_percentage': health.uptime_percentage
            }
        
        return {
            'total_requests': self.request_count,
            'total_cost_usd': float(self.total_cost),
            'avg_cost_per_request': float(self.total_cost / max(self.request_count, 1)),
            'total_models': len(self.models),
            'available_models': sum(1 for m in self.models.values() if m.is_available),
            'provider_health': provider_stats,
            'budget_limit': self.budget_limit,
            'budget_remaining': self.budget_limit - float(self.total_cost) if self.budget_limit else None,
            'models_by_tier': {
                tier.value: len(self.get_models_by_tier(tier))
                for tier in ModelTier
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        model_performance = {}
        
        for key, model in self.models.items():
            if model.performance.request_count > 0:
                model_performance[key] = {
                    'name': model.name,
                    'provider': model.provider,
                    'tier': model.tier.value,
                    'requests': model.performance.request_count,
                    'success_rate': model.performance.success_rate,
                    'avg_latency_ms': model.performance.avg_latency_ms,
                    'cost_efficiency': model.performance.cost_efficiency,
                    'quality_score': model.performance.quality_score
                }
        
        return {
            'session_summary': self.get_session_stats(),
            'model_performance': model_performance,
            'top_performers': self._get_top_performing_models(),
            'cost_analysis': self._get_cost_analysis()
        }
    
    def _get_top_performing_models(self) -> Dict[str, List[str]]:
        """Get top performing models by different metrics"""
        models_with_data = {k: v for k, v in self.models.items() if v.performance.request_count > 0}
        
        if not models_with_data:
            return {}
        
        return {
            'lowest_latency': sorted(
                models_with_data.keys(),
                key=lambda k: models_with_data[k].performance.avg_latency_ms
            )[:5],
            'highest_success_rate': sorted(
                models_with_data.keys(),
                key=lambda k: models_with_data[k].performance.success_rate,
                reverse=True
            )[:5],
            'best_cost_efficiency': sorted(
                models_with_data.keys(),
                key=lambda k: models_with_data[k].performance.cost_efficiency,
                reverse=True
            )[:5]
        }
    
    def _get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis breakdown"""
        provider_costs = {}
        tier_costs = {}
        
        for model in self.models.values():
            if model.performance.request_count > 0:
                # Estimate cost contribution (simplified)
                model_cost = float(self.total_cost) * (model.performance.request_count / self.request_count)
                
                if model.provider not in provider_costs:
                    provider_costs[model.provider] = 0
                provider_costs[model.provider] += model_cost
                
                if model.tier.value not in tier_costs:
                    tier_costs[model.tier.value] = 0
                tier_costs[model.tier.value] += model_cost
        
        return {
            'total_cost': float(self.total_cost),
            'cost_by_provider': provider_costs,
            'cost_by_tier': tier_costs,
            'avg_cost_per_request': float(self.total_cost / max(self.request_count, 1))
        }