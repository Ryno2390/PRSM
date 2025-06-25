"""
Intelligent Provider Selection and Routing System for PRSM
Multi-provider load balancing with cost, latency, and availability optimization

🎯 PURPOSE IN PRSM:
Central routing intelligence that selects optimal AI providers based on:
- Real-time cost analysis across providers
- Latency requirements and SLA compliance
- Provider availability and health monitoring
- Quality requirements and model capabilities
- Automatic failover and load balancing
- Budget management and cost optimization
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import structlog

# Import PRSM model clients
from .api_clients import ModelClientRegistry, ModelProvider, ModelExecutionRequest, ModelExecutionResponse
from .enhanced_anthropic_client import EnhancedAnthropicClient, ClaudeModel
from .enhanced_ollama_client import EnhancedOllamaClient, OllamaModel
from .openrouter_client import OpenRouterClient, OpenRouterModel

logger = structlog.get_logger(__name__)

class RoutingStrategy(Enum):
    """Available routing strategies"""
    COST_OPTIMIZED = "cost_optimized"      # Minimize cost
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize response time
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize output quality
    BALANCED = "balanced"                   # Balance all factors
    AVAILABILITY_FIRST = "availability_first"  # Prioritize uptime
    LOCAL_PREFERRED = "local_preferred"    # Prefer local models when possible

class TaskType(Enum):
    """Task type categorization for optimal routing"""
    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    Q_AND_A = "q_and_a"
    MULTIMODAL = "multimodal"

@dataclass
class ProviderMetrics:
    """Real-time metrics for each provider"""
    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost: float = 0.0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    success_rate: float = 0.0
    availability: float = 1.0
    last_successful_request: float = 0.0
    consecutive_failures: int = 0
    
    def update_metrics(self, success: bool, response_time: float, cost: float):
        """Update provider metrics with new request data"""
        self.total_requests += 1
        self.total_response_time += response_time
        self.total_cost += cost
        
        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
            self.last_successful_request = time.time()
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1
        
        # Calculate rolling averages
        self.success_rate = self.successful_requests / self.total_requests
        self.average_response_time = self.total_response_time / self.total_requests
        
        # Update availability based on recent performance
        if self.consecutive_failures > 5:
            self.availability = max(0.1, 1.0 - (self.consecutive_failures * 0.1))
        else:
            self.availability = min(1.0, self.success_rate * 1.1)

@dataclass
class RoutingDecision:
    """Result of routing decision with rationale"""
    selected_provider: str
    selected_model: str
    routing_strategy: RoutingStrategy
    decision_factors: Dict[str, float]
    estimated_cost: float
    estimated_latency: float
    confidence_score: float
    alternatives_considered: List[str]
    fallback_providers: List[str]

@dataclass
class RoutingConstraints:
    """Constraints for routing decisions"""
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    min_quality: Optional[float] = None
    preferred_providers: Optional[List[str]] = None
    excluded_providers: Optional[List[str]] = None
    require_local: bool = False
    require_cloud: bool = False
    budget_limit: Optional[float] = None

class IntelligentRouter:
    """
    Intelligent routing system for PRSM multi-provider ecosystem
    
    🚀 FEATURES:
    - Multi-provider load balancing across OpenAI, Anthropic, Ollama, OpenRouter
    - Real-time performance monitoring and health checking
    - Cost optimization with budget management
    - Latency-aware routing with SLA compliance
    - Automatic failover and circuit breaker patterns
    - Quality-based model selection
    - Task-specific routing optimization
    """
    
    def __init__(self,
                 openai_client: Optional[Any] = None,
                 anthropic_client: Optional[EnhancedAnthropicClient] = None,
                 ollama_client: Optional[EnhancedOllamaClient] = None,
                 openrouter_client: Optional[OpenRouterClient] = None,
                 default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
                 health_check_interval: int = 300):  # 5 minutes
        """
        Initialize intelligent router
        
        Args:
            openai_client: OpenAI client instance
            anthropic_client: Enhanced Anthropic client
            ollama_client: Enhanced Ollama client  
            openrouter_client: OpenRouter client
            default_strategy: Default routing strategy
            health_check_interval: Seconds between health checks
        """
        self.clients = {
            "openai": openai_client,
            "anthropic": anthropic_client,
            "ollama": ollama_client,
            "openrouter": openrouter_client
        }
        
        # Remove None clients
        self.clients = {k: v for k, v in self.clients.items() if v is not None}
        
        self.default_strategy = default_strategy
        self.health_check_interval = health_check_interval
        
        # Performance tracking
        self.provider_metrics: Dict[str, ProviderMetrics] = {}
        for provider_name in self.clients.keys():
            self.provider_metrics[provider_name] = ProviderMetrics(provider_name)
        
        # Cost and quality mappings (simplified)
        self.provider_cost_tiers = {
            "openai": {"gpt-4": 0.06, "gpt-3.5-turbo": 0.002},
            "anthropic": {"claude-3-opus": 0.075, "claude-3-sonnet": 0.015, "claude-3-haiku": 0.00125},
            "ollama": {"default": 0.0},  # Local models are "free"
            "openrouter": {"default": 0.01}  # Variable pricing
        }
        
        self.provider_quality_scores = {
            "openai": {"gpt-4": 0.95, "gpt-3.5-turbo": 0.85},
            "anthropic": {"claude-3-opus": 0.96, "claude-3-sonnet": 0.90, "claude-3-haiku": 0.80},
            "ollama": {"llama2-70b": 0.85, "llama2-13b": 0.75, "llama2-7b": 0.65},
            "openrouter": {"default": 0.80}
        }
        
        # Task-specific routing preferences
        self.task_preferences = {
            TaskType.CODE_GENERATION: ["openai", "anthropic", "openrouter"],
            TaskType.REASONING: ["anthropic", "openai", "openrouter"],
            TaskType.CREATIVE_WRITING: ["anthropic", "openai", "openrouter"],
            TaskType.GENERAL_CHAT: ["ollama", "openrouter", "openai"],
            TaskType.ANALYSIS: ["anthropic", "openai", "openrouter"],
            TaskType.SUMMARIZATION: ["openai", "anthropic", "openrouter"]
        }
        
        # Health monitoring
        self._last_health_check = 0.0
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize router and start health monitoring"""
        # Initialize all clients
        for provider_name, client in self.clients.items():
            try:
                if hasattr(client, 'initialize'):
                    await client.initialize()
                logger.info(f"Initialized {provider_name} client")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name}: {e}")
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Intelligent router initialized successfully")
    
    async def close(self):
        """Close router and all clients"""
        if self._health_check_task:
            self._health_check_task.cancel()
            
        for provider_name, client in self.clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                logger.error(f"Error closing {provider_name}: {e}")
    
    async def route_request(self,
                           request: ModelExecutionRequest,
                           strategy: Optional[RoutingStrategy] = None,
                           constraints: Optional[RoutingConstraints] = None,
                           task_type: Optional[TaskType] = None) -> ModelExecutionResponse:
        """
        Route request to optimal provider
        
        Args:
            request: Model execution request
            strategy: Routing strategy to use
            constraints: Routing constraints
            task_type: Type of task for optimization
        
        Returns:
            Model execution response with routing metadata
        """
        strategy = strategy or self.default_strategy
        constraints = constraints or RoutingConstraints()
        
        # Make routing decision
        decision = await self._make_routing_decision(request, strategy, constraints, task_type)
        
        # Execute request with selected provider
        response = await self._execute_with_provider(
            decision.selected_provider,
            decision.selected_model,
            request
        )
        
        # Update metrics
        self.provider_metrics[decision.selected_provider].update_metrics(
            response.success,
            response.execution_time,
            getattr(response, 'cost', 0.0)
        )
        
        # Add routing metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            "routing_decision": {
                "provider": decision.selected_provider,
                "model": decision.selected_model,
                "strategy": decision.routing_strategy.value,
                "confidence": decision.confidence_score,
                "alternatives": decision.alternatives_considered,
                "decision_factors": decision.decision_factors
            }
        })
        
        return response
    
    async def _make_routing_decision(self,
                                   request: ModelExecutionRequest,
                                   strategy: RoutingStrategy,
                                   constraints: RoutingConstraints,
                                   task_type: Optional[TaskType]) -> RoutingDecision:
        """Make intelligent routing decision"""
        
        # Get available providers
        available_providers = self._get_available_providers(constraints)
        
        if not available_providers:
            raise RuntimeError("No available providers match constraints")
        
        # Score each provider
        provider_scores = {}
        decision_factors = {}
        
        for provider_name in available_providers:
            score, factors = await self._score_provider(
                provider_name, request, strategy, constraints, task_type
            )
            provider_scores[provider_name] = score
            decision_factors[provider_name] = factors
        
        # Select best provider
        best_provider = max(provider_scores, key=provider_scores.get)
        best_score = provider_scores[best_provider]
        
        # Select model for provider
        selected_model = await self._select_model_for_provider(
            best_provider, request, strategy, constraints
        )
        
        # Calculate estimates
        estimated_cost = self._estimate_cost(best_provider, selected_model, request)
        estimated_latency = self._estimate_latency(best_provider, request)
        
        # Prepare alternatives and fallbacks
        alternatives = list(provider_scores.keys())
        alternatives.remove(best_provider)
        alternatives.sort(key=lambda p: provider_scores[p], reverse=True)
        
        fallback_providers = [p for p in alternatives if provider_scores[p] > 0.5][:3]
        
        return RoutingDecision(
            selected_provider=best_provider,
            selected_model=selected_model,
            routing_strategy=strategy,
            decision_factors=decision_factors[best_provider],
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            confidence_score=best_score,
            alternatives_considered=alternatives,
            fallback_providers=fallback_providers
        )
    
    def _get_available_providers(self, constraints: RoutingConstraints) -> List[str]:
        """Get providers that meet constraints"""
        available = []
        
        for provider_name, metrics in self.provider_metrics.items():
            # Check availability
            if metrics.availability < 0.8 or metrics.consecutive_failures > 3:
                continue
            
            # Check constraints
            if constraints.excluded_providers and provider_name in constraints.excluded_providers:
                continue
                
            if constraints.preferred_providers and provider_name not in constraints.preferred_providers:
                continue
                
            if constraints.require_local and provider_name not in ["ollama"]:
                continue
                
            if constraints.require_cloud and provider_name in ["ollama"]:
                continue
            
            available.append(provider_name)
        
        return available
    
    async def _score_provider(self,
                            provider_name: str,
                            request: ModelExecutionRequest,
                            strategy: RoutingStrategy,
                            constraints: RoutingConstraints,
                            task_type: Optional[TaskType]) -> Tuple[float, Dict[str, float]]:
        """Score provider based on strategy and constraints"""
        
        metrics = self.provider_metrics[provider_name]
        
        # Base scoring factors
        cost_score = self._calculate_cost_score(provider_name, request)
        latency_score = self._calculate_latency_score(provider_name, constraints)
        quality_score = self._calculate_quality_score(provider_name, request)
        availability_score = metrics.availability
        reliability_score = metrics.success_rate
        
        # Task-specific preference
        task_preference_score = 1.0
        if task_type and task_type in self.task_preferences:
            preferred_providers = self.task_preferences[task_type]
            if provider_name in preferred_providers:
                position = preferred_providers.index(provider_name)
                task_preference_score = 1.0 - (position * 0.1)
        
        factors = {
            "cost": cost_score,
            "latency": latency_score,
            "quality": quality_score,
            "availability": availability_score,
            "reliability": reliability_score,
            "task_preference": task_preference_score
        }
        
        # Strategy-based weighting
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            weights = {"cost": 0.5, "availability": 0.2, "reliability": 0.15, "quality": 0.1, "latency": 0.05}
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            weights = {"latency": 0.4, "availability": 0.25, "reliability": 0.2, "cost": 0.1, "quality": 0.05}
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            weights = {"quality": 0.4, "reliability": 0.25, "availability": 0.2, "cost": 0.1, "latency": 0.05}
        elif strategy == RoutingStrategy.AVAILABILITY_FIRST:
            weights = {"availability": 0.4, "reliability": 0.3, "latency": 0.15, "cost": 0.1, "quality": 0.05}
        elif strategy == RoutingStrategy.LOCAL_PREFERRED:
            if provider_name == "ollama":
                return 0.95, factors  # Strong preference for local
            weights = {"cost": 0.3, "availability": 0.25, "reliability": 0.2, "quality": 0.15, "latency": 0.1}
        else:  # BALANCED
            weights = {"cost": 0.2, "latency": 0.2, "quality": 0.2, "availability": 0.2, "reliability": 0.15, "task_preference": 0.05}
        
        # Calculate weighted score
        total_score = sum(factors[factor] * weights.get(factor, 0) for factor in factors)
        
        # Apply task preference bonus
        total_score *= task_preference_score
        
        return min(1.0, total_score), factors
    
    def _calculate_cost_score(self, provider_name: str, request: ModelExecutionRequest) -> float:
        """Calculate cost score (higher = cheaper)"""
        if provider_name == "ollama":
            return 1.0  # Local models are free
        
        # Simplified cost calculation
        cost_tier = self.provider_cost_tiers.get(provider_name, {}).get("default", 0.01)
        
        # Normalize to 0-1 range (assuming max cost of $0.10 per 1K tokens)
        normalized_cost = min(1.0, cost_tier / 0.10)
        
        return 1.0 - normalized_cost
    
    def _calculate_latency_score(self, provider_name: str, constraints: RoutingConstraints) -> float:
        """Calculate latency score (higher = faster)"""
        metrics = self.provider_metrics[provider_name]
        
        if metrics.total_requests == 0:
            # Default latency estimates
            default_latencies = {
                "ollama": 2.0,      # Local inference can be slow
                "openai": 1.0,      # Generally fast
                "anthropic": 1.2,   # Slightly slower
                "openrouter": 1.5   # Variable
            }
            avg_latency = default_latencies.get(provider_name, 2.0)
        else:
            avg_latency = metrics.average_response_time
        
        # Normalize against maximum acceptable latency
        max_latency = constraints.max_latency or 10.0
        latency_score = max(0.1, 1.0 - (avg_latency / max_latency))
        
        return latency_score
    
    def _calculate_quality_score(self, provider_name: str, request: ModelExecutionRequest) -> float:
        """Calculate quality score"""
        quality_scores = self.provider_quality_scores.get(provider_name, {"default": 0.75})
        
        # Try to match specific model, otherwise use default
        model_id = request.model_id.lower()
        for model_key, score in quality_scores.items():
            if model_key in model_id or model_key == "default":
                return score
        
        return quality_scores.get("default", 0.75)
    
    async def _select_model_for_provider(self,
                                       provider_name: str,
                                       request: ModelExecutionRequest,
                                       strategy: RoutingStrategy,
                                       constraints: RoutingConstraints) -> str:
        """Select optimal model for chosen provider"""
        
        if provider_name == "openai":
            # Select based on requirements
            if strategy == RoutingStrategy.COST_OPTIMIZED:
                return "gpt-3.5-turbo"
            elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
                return "gpt-4"
            else:
                return "gpt-3.5-turbo"
        
        elif provider_name == "anthropic":
            if strategy == RoutingStrategy.COST_OPTIMIZED:
                return ClaudeModel.CLAUDE_3_HAIKU.value
            elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
                return ClaudeModel.CLAUDE_3_OPUS.value
            else:
                return ClaudeModel.CLAUDE_3_SONNET.value
        
        elif provider_name == "ollama":
            # Select based on available models and requirements
            if hasattr(self.clients["ollama"], 'get_model_recommendations'):
                recommendations = self.clients["ollama"].get_model_recommendations(
                    "general", "balanced"
                )
                if recommendations:
                    return recommendations[0].value
            return OllamaModel.LLAMA2_7B_CHAT.value
        
        elif provider_name == "openrouter":
            # Use OpenRouter's auto-selection
            return "auto"
        
        return request.model_id or "default"
    
    def _estimate_cost(self, provider_name: str, model: str, request: ModelExecutionRequest) -> float:
        """Estimate request cost"""
        if provider_name == "ollama":
            return 0.0  # Local models
        
        # Simplified cost estimation
        estimated_tokens = len(request.prompt) * 0.75 + request.max_tokens
        cost_per_1k = self.provider_cost_tiers.get(provider_name, {}).get("default", 0.01)
        
        return (estimated_tokens / 1000) * cost_per_1k
    
    def _estimate_latency(self, provider_name: str, request: ModelExecutionRequest) -> float:
        """Estimate response latency"""
        metrics = self.provider_metrics[provider_name]
        
        if metrics.total_requests > 0:
            return metrics.average_response_time
        
        # Default estimates
        defaults = {
            "ollama": 3.0,
            "openai": 1.5,
            "anthropic": 2.0,
            "openrouter": 2.5
        }
        
        return defaults.get(provider_name, 2.0)
    
    async def _execute_with_provider(self,
                                   provider_name: str,
                                   model: str,
                                   request: ModelExecutionRequest) -> ModelExecutionResponse:
        """Execute request with specific provider"""
        client = self.clients[provider_name]
        
        try:
            if provider_name == "anthropic":
                messages = [{"role": "user", "content": request.prompt}]
                if request.system_prompt:
                    messages.insert(0, {"role": "system", "content": request.system_prompt})
                
                claude_model = ClaudeModel(model) if model in [m.value for m in ClaudeModel] else ClaudeModel.CLAUDE_3_SONNET
                response = await client.complete(messages, claude_model)
                
                return ModelExecutionResponse(
                    content=response.content,
                    provider=ModelProvider.ANTHROPIC,
                    model_id=response.model,
                    execution_time=response.response_time,
                    token_usage=response.usage,
                    success=response.success,
                    error=response.error,
                    metadata={"cost": response.cost}
                )
            
            elif provider_name == "ollama":
                ollama_model = OllamaModel(model) if model in [m.value for m in OllamaModel] else OllamaModel.LLAMA2_7B_CHAT
                response = await client.generate(ollama_model, request.prompt, system=request.system_prompt)
                
                return ModelExecutionResponse(
                    content=response["response"],
                    provider=ModelProvider.LOCAL,
                    model_id=response["model"],
                    execution_time=response["performance"]["response_time"],
                    token_usage={"local_inference": True},
                    success=response.get("done", True),
                    metadata=response["performance"]
                )
            
            elif provider_name == "openrouter":
                messages = [{"role": "user", "content": request.prompt}]
                if request.system_prompt:
                    messages.insert(0, {"role": "system", "content": request.system_prompt})
                
                response = await client.complete(messages, model)
                
                return ModelExecutionResponse(
                    content=response["content"],
                    provider=ModelProvider.OPENAI,  # OpenRouter uses OpenAI format
                    model_id=response["model"],
                    execution_time=response["response_time"],
                    token_usage=response["usage"],
                    success=response["success"],
                    error=response.get("error"),
                    metadata={"cost": response["cost"], "provider": response.get("provider")}
                )
            
            else:
                # Fallback for other providers
                return ModelExecutionResponse(
                    content="Provider not implemented",
                    provider=ModelProvider.LOCAL,
                    model_id=model,
                    execution_time=0.0,
                    token_usage={},
                    success=False,
                    error=f"Provider {provider_name} not implemented"
                )
                
        except Exception as e:
            logger.error(f"Error executing with {provider_name}: {e}")
            return ModelExecutionResponse(
                content="",
                provider=ModelProvider.LOCAL,
                model_id=model,
                execution_time=0.0,
                token_usage={},
                success=False,
                error=str(e)
            )
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers"""
        for provider_name, client in self.clients.items():
            try:
                # Simple health check request
                test_request = ModelExecutionRequest(
                    prompt="test",
                    model_id="default",
                    max_tokens=1
                )
                
                start_time = time.time()
                response = await self._execute_with_provider(provider_name, "default", test_request)
                response_time = time.time() - start_time
                
                # Update metrics
                metrics = self.provider_metrics[provider_name]
                metrics.update_metrics(response.success, response_time, 0.0)
                
                logger.debug(f"Health check {provider_name}: {'✅' if response.success else '❌'}")
                
            except Exception as e:
                logger.warning(f"Health check failed for {provider_name}: {e}")
                metrics = self.provider_metrics[provider_name]
                metrics.update_metrics(False, 10.0, 0.0)  # Assume 10s timeout
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all providers"""
        status = {}
        
        for provider_name, metrics in self.provider_metrics.items():
            status[provider_name] = {
                "availability": metrics.availability,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "total_requests": metrics.total_requests,
                "consecutive_failures": metrics.consecutive_failures,
                "total_cost": metrics.total_cost,
                "last_successful_request": metrics.last_successful_request
            }
        
        return status
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing analytics and insights"""
        total_requests = sum(m.total_requests for m in self.provider_metrics.values())
        total_cost = sum(m.total_cost for m in self.provider_metrics.values())
        
        if total_requests == 0:
            return {"total_requests": 0, "total_cost": 0, "providers": {}}
        
        provider_breakdown = {}
        for provider_name, metrics in self.provider_metrics.items():
            provider_breakdown[provider_name] = {
                "request_share": metrics.total_requests / total_requests,
                "cost_share": metrics.total_cost / total_cost if total_cost > 0 else 0,
                "average_cost_per_request": metrics.total_cost / metrics.total_requests if metrics.total_requests > 0 else 0,
                "performance_score": metrics.success_rate * metrics.availability
            }
        
        return {
            "total_requests": total_requests,
            "total_cost": total_cost,
            "average_cost_per_request": total_cost / total_requests,
            "overall_success_rate": sum(m.successful_requests for m in self.provider_metrics.values()) / total_requests,
            "providers": provider_breakdown
        }

# Example usage
async def example_usage():
    """Example of intelligent router usage"""
    
    # Initialize clients (would be real clients in practice)
    anthropic_client = EnhancedAnthropicClient(api_key="test")
    ollama_client = EnhancedOllamaClient()
    openrouter_client = OpenRouterClient(api_key="test")
    
    # Create router
    router = IntelligentRouter(
        anthropic_client=anthropic_client,
        ollama_client=ollama_client,
        openrouter_client=openrouter_client,
        default_strategy=RoutingStrategy.BALANCED
    )
    
    await router.initialize()
    
    # Route a request
    request = ModelExecutionRequest(
        prompt="Explain quantum computing",
        model_id="auto",
        max_tokens=500
    )
    
    constraints = RoutingConstraints(
        max_cost=0.01,
        max_latency=5.0
    )
    
    response = await router.route_request(
        request,
        strategy=RoutingStrategy.COST_OPTIMIZED,
        constraints=constraints,
        task_type=TaskType.GENERAL_CHAT
    )
    
    print(f"Response: {response.content}")
    print(f"Routed to: {response.metadata['routing_decision']['provider']}")
    
    # Get analytics
    analytics = router.get_routing_analytics()
    print(f"Total requests: {analytics['total_requests']}")
    
    await router.close()

if __name__ == "__main__":
    asyncio.run(example_usage())