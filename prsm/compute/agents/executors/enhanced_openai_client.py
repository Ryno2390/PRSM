"""
Enhanced OpenAI Client with Cost Management and Retry Logic
===========================================================

Extends the base OpenAI client with production-ready features:
- Retry logic with exponential backoff
- Cost tracking and budget management  
- Rate limiting and usage optimization
- Advanced error handling and monitoring
"""

import asyncio
import time
from decimal import Decimal
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import aiohttp
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .api_clients import BaseModelClient, ModelExecutionRequest, ModelExecutionResponse, ModelProvider
from prsm.core.config.model_config_manager import get_model_config_manager

logger = structlog.get_logger(__name__)


@dataclass
class CostTracker:
    """Track API costs and usage for budget management"""
    
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0
    cost_by_model: Dict[str, Decimal] = field(default_factory=dict)
    
    def calculate_request_cost(self, model_id: str, token_usage: Dict[str, int]) -> Decimal:
        """Calculate cost for a single request using ModelConfigManager"""
        config_manager = get_model_config_manager()
        model_pricing = config_manager.get_model_pricing(model_id, "openai")
        
        if not model_pricing:
            # Fallback to default GPT-4 pricing if model not found
            logger.warning(f"Pricing not found for model {model_id}, using GPT-4 fallback")
            input_cost_per_1k = Decimal("0.03")
            output_cost_per_1k = Decimal("0.06")
        else:
            input_cost_per_1k = model_pricing.input_cost_per_1k
            output_cost_per_1k = model_pricing.output_cost_per_1k
        
        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)
        
        input_cost = (Decimal(str(input_tokens)) / 1000) * input_cost_per_1k
        output_cost = (Decimal(str(output_tokens)) / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def track_request(self, model_id: str, token_usage: Dict[str, int]) -> Decimal:
        """Track usage and cost for a request"""
        cost = self.calculate_request_cost(model_id, token_usage)
        
        self.total_cost += cost
        self.total_input_tokens += token_usage.get("prompt_tokens", 0)
        self.total_output_tokens += token_usage.get("completion_tokens", 0)
        self.request_count += 1
        
        if model_id not in self.cost_by_model:
            self.cost_by_model[model_id] = Decimal("0")
        self.cost_by_model[model_id] += cost
        
        return cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return {
            "total_cost_usd": float(self.total_cost),
            "total_requests": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_cost_per_request": float(self.total_cost / max(self.request_count, 1)),
            "cost_by_model": {k: float(v) for k, v in self.cost_by_model.items()}
        }


@dataclass
class RateLimiter:
    """Rate limiting for API requests"""
    
    requests_per_minute: int = 3500  # OpenAI default for paid accounts
    tokens_per_minute: int = 90000   # OpenAI default for paid accounts
    
    request_timestamps: List[float] = field(default_factory=list)
    token_usage_timestamps: List[tuple] = field(default_factory=list)  # (timestamp, tokens)
    
    async def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if rate limits would be exceeded"""
        now = time.time()
        
        # Clean old timestamps (older than 1 minute)
        cutoff = now - 60
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        self.token_usage_timestamps = [(ts, tokens) for ts, tokens in self.token_usage_timestamps if ts > cutoff]
        
        # Check request rate limit
        if len(self.request_timestamps) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self.token_usage_timestamps)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            wait_time = 60 - (now - self.token_usage_timestamps[0][0])
            if wait_time > 0:
                logger.warning(f"Token rate limit hit, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(now)
        self.token_usage_timestamps.append((now, estimated_tokens))


class EnhancedOpenAIClient(BaseModelClient):
    """
    Enhanced OpenAI client with production features
    
    🚀 ENHANCEMENTS:
    - Automatic retry with exponential backoff
    - Cost tracking and budget management
    - Rate limiting and usage optimization
    - Advanced error handling and recovery
    - Performance monitoring and metrics
    """
    
    def __init__(self, api_key: str, budget_limit_usd: float = 100.0, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.openai.com/v1"
        self.budget_limit = Decimal(str(budget_limit_usd))
        self.cost_tracker = CostTracker()
        self.rate_limiter = RateLimiter()
        
        # Configuration
        self.max_retries = kwargs.get("max_retries", 3)
        self.enable_cost_tracking = kwargs.get("enable_cost_tracking", True)
        self.enable_rate_limiting = kwargs.get("enable_rate_limiting", True)
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PRSM/1.0 (Enhanced OpenAI Client)"
        }
    
    async def _setup_client(self):
        """Enhanced setup with API validation"""
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [model["id"] for model in data.get("data", [])]
                    logger.info("Enhanced OpenAI client initialized", models_available=len(available_models))
                elif response.status == 401:
                    logger.error("OpenAI API key is invalid")
                    raise ValueError("Invalid OpenAI API key")
                else:
                    logger.warning(f"OpenAI API returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced OpenAI client: {e}")
            raise
    
    def _estimate_tokens(self, request: ModelExecutionRequest) -> int:
        """Estimate token usage for rate limiting"""
        # Rough estimation: 1 token ≈ 0.75 words
        prompt_words = len(request.prompt.split())
        system_words = len(request.system_prompt.split()) if request.system_prompt else 0
        estimated_input = int((prompt_words + system_words) / 0.75)
        
        return estimated_input + request.max_tokens
    
    def _check_budget(self, estimated_cost: Decimal) -> bool:
        """Check if request would exceed budget"""
        if not self.enable_cost_tracking:
            return True
            
        projected_total = self.cost_tracker.total_cost + estimated_cost
        return projected_total <= self.budget_limit
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_api_request(self, payload: Dict[str, Any]) -> aiohttp.ClientResponse:
        """Make API request with retry logic"""
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 429:  # Rate limit
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                raise aiohttp.ClientError("Rate limited")
            
            return response
    
    async def execute(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """
        Enhanced execution with cost management and retry logic
        
        🔄 EXECUTION FLOW:
        1. Pre-flight checks (budget, rate limits)
        2. Format request with optimization
        3. Execute with retry logic
        4. Track costs and usage
        5. Return enhanced response
        """
        start_time = time.time()
        
        try:
            # 🚦 PRE-FLIGHT CHECKS
            estimated_tokens = self._estimate_tokens(request)
            
            if self.enable_rate_limiting:
                await self.rate_limiter.wait_if_needed(estimated_tokens)
            
            if self.enable_cost_tracking:
                estimated_cost = self.cost_tracker.calculate_request_cost(
                    request.model_id, 
                    {"prompt_tokens": int(estimated_tokens * 0.7), "completion_tokens": request.max_tokens}
                )
                
                if not self._check_budget(estimated_cost):
                    return ModelExecutionResponse(
                        content="",
                        provider=ModelProvider.OPENAI,
                        model_id=request.model_id,
                        execution_time=time.time() - start_time,
                        token_usage={},
                        success=False,
                        error=f"Budget limit exceeded. Current: ${self.cost_tracker.total_cost}, Limit: ${self.budget_limit}"
                    )
            
            # 📝 FORMAT REQUEST
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            payload = {
                "model": request.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": 1.0,  # Add for more consistent results
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            
            # 🚀 EXECUTE WITH RETRY
            response = await self._make_api_request(payload)
            execution_time = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                
                # 💰 TRACK COSTS
                actual_cost = Decimal("0")
                if self.enable_cost_tracking:
                    actual_cost = self.cost_tracker.track_request(
                        request.model_id, 
                        data.get("usage", {})
                    )
                
                logger.info(
                    "OpenAI request successful",
                    model=request.model_id,
                    execution_time=execution_time,
                    cost_usd=float(actual_cost),
                    tokens=data.get("usage", {})
                )
                
                return ModelExecutionResponse(
                    content=data["choices"][0]["message"]["content"],
                    provider=ModelProvider.OPENAI,
                    model_id=request.model_id,
                    execution_time=execution_time,
                    token_usage=data.get("usage", {}),
                    success=True,
                    metadata={
                        "finish_reason": data["choices"][0].get("finish_reason"),
                        "cost_usd": float(actual_cost),
                        "budget_remaining": float(self.budget_limit - self.cost_tracker.total_cost),
                        "request_id": data.get("id"),
                        "model": data.get("model")
                    }
                )
            else:
                error_data = await response.json()
                error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                
                logger.error(
                    "OpenAI request failed",
                    status=response.status,
                    error=error_msg,
                    model=request.model_id
                )
                
                return ModelExecutionResponse(
                    content="",
                    provider=ModelProvider.OPENAI,
                    model_id=request.model_id,
                    execution_time=time.time() - start_time,
                    token_usage={},
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Enhanced OpenAI execution failed: {e}")
            
            return ModelExecutionResponse(
                content="",
                provider=ModelProvider.OPENAI,
                model_id=request.model_id,
                execution_time=execution_time,
                token_usage={},
                success=False,
                error=str(e)
            )
    
    async def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage and cost summary"""
        summary = self.cost_tracker.get_summary()
        summary.update({
            "budget_limit_usd": float(self.budget_limit),
            "budget_used_percent": float(self.cost_tracker.total_cost / self.budget_limit * 100),
            "budget_remaining_usd": float(self.budget_limit - self.cost_tracker.total_cost),
            "rate_limiting_enabled": self.enable_rate_limiting,
            "cost_tracking_enabled": self.enable_cost_tracking
        })
        return summary
    
    async def reset_budget(self, new_limit_usd: float = None):
        """Reset budget and usage tracking"""
        if new_limit_usd:
            self.budget_limit = Decimal(str(new_limit_usd))
        
        self.cost_tracker = CostTracker()
        logger.info(f"Budget reset to ${self.budget_limit}")


# Factory function for easy client creation
async def create_enhanced_openai_client(api_key: str, **kwargs) -> EnhancedOpenAIClient:
    """Create and initialize an enhanced OpenAI client"""
    client = EnhancedOpenAIClient(api_key, **kwargs)
    await client.initialize()
    return client