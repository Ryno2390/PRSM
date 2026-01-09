#!/usr/bin/env python3
"""
Unified Model Router for PRSM
============================

Consolidated routing system that combines:
- Privacy-first routing with PII detection (from hybrid_router.py)
- Multi-provider load balancing and health monitoring (from intelligent_router.py)
- Advanced cost optimization and quality scoring
- Real-time performance analytics and circuit breaker patterns
- Intelligent fallback strategies with comprehensive monitoring

🎯 UNIFIED FEATURES:
- Privacy-aware routing with automatic PII detection
- Multi-provider health monitoring and failover
- Cost optimization with budget management
- Performance-based routing with SLA compliance
- Quality-aware model selection
- Real-time analytics and comprehensive logging
"""

import asyncio
import re
import time
import statistics
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID
import structlog

# Import PRSM components
from .api_clients import (
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelProvider
)
from .enhanced_anthropic_client import EnhancedAnthropicClient, ClaudeModel
from .enhanced_ollama_client import EnhancedOllamaClient, OllamaModel
from .openrouter_client import OpenRouterClient
from prsm.core.usage_tracker import GenericUsageTracker, ResourceType, CostCategory, OperationType
from prsm.core.config.model_config_manager import get_model_config_manager

logger = structlog.get_logger(__name__)


class RoutingStrategy(Enum):
    """Unified routing strategies combining both routers"""
    # From HybridModelRouter
    PRIVACY_FIRST = "privacy_first"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_BASED = "performance_based"
    CAPABILITY_AWARE = "capability_aware"
    
    # From IntelligentRouter
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    AVAILABILITY_FIRST = "availability_first"
    LOCAL_PREFERRED = "local_preferred"
    
    # New unified strategy
    HYBRID_INTELLIGENT = "hybrid_intelligent"
    BALANCED = "balanced"


class SensitivityLevel(Enum):
    """Data sensitivity classification for privacy routing"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


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
        self.success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
        self.average_response_time = self.total_response_time / self.total_requests if self.total_requests > 0 else 0.0
        
        # Update availability based on recent performance
        if self.consecutive_failures > 5:
            self.availability = max(0.1, 1.0 - (self.consecutive_failures * 0.1))
        else:
            self.availability = min(1.0, self.success_rate * 1.1)


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
    privacy_required: bool = False


@dataclass
class UnifiedRoutingDecision:
    """Comprehensive routing decision with all metadata"""
    selected_provider: str
    selected_model: str
    routing_strategy: RoutingStrategy
    use_local: bool
    client_type: str
    reasoning: str
    sensitivity_level: SensitivityLevel
    estimated_cost: float
    estimated_latency: float
    confidence_score: float
    decision_factors: Dict[str, float]
    priority_factors: List[str]
    alternatives_considered: List[str]
    fallback_providers: List[str]
    privacy_detected: bool = False
    quality_score: float = 0.0


class PrivacyDetector:
    """Enhanced privacy detection system"""
    
    def __init__(self):
        # Regex patterns for sensitive data
        self.patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s?\d{3}-\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'bank_account': re.compile(r'\b\d{8,17}\b'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            'api_key': re.compile(r'\b[a-zA-Z0-9]{20,}\b'),
            'password': re.compile(r'password[:\s]*[^\s]+', re.IGNORECASE)
        }
        
        # Enhanced sensitive keywords
        self.sensitive_keywords = {
            'financial': ['salary', 'income', 'bank', 'account', 'credit', 'loan', 'mortgage', 'investment', 'revenue', 'profit'],
            'medical': ['diagnosis', 'symptoms', 'medication', 'treatment', 'doctor', 'hospital', 'patient', 'health', 'medical'],
            'personal': ['password', 'confidential', 'private', 'secret', 'internal', 'proprietary', 'classified', 'sensitive'],
            'legal': ['lawsuit', 'contract', 'agreement', 'litigation', 'attorney', 'legal', 'court', 'settlement'],
            'corporate': ['acquisition', 'merger', 'layoffs', 'reorganization', 'strategic', 'competitive', 'insider']
        }
    
    def analyze_sensitivity(self, text: str, system_prompt: str = "") -> Tuple[SensitivityLevel, List[str]]:
        """Analyze text for sensitive content with detailed reporting"""
        combined_text = f"{text} {system_prompt}".lower()
        detected_patterns = []
        
        # Check for PII patterns
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(combined_text):
                detected_patterns.append(f"pii:{pattern_name}")
                logger.warning("PII detected", pattern_type=pattern_name)
                return SensitivityLevel.RESTRICTED, detected_patterns
        
        # Check for sensitive keywords
        high_risk_count = 0
        for category, keywords in self.sensitive_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            if matches >= 2:  # Multiple matches in same category
                detected_patterns.append(f"keywords:{category}:{matches}")
                logger.info("Sensitive content detected", category=category, matches=matches)
                return SensitivityLevel.CONFIDENTIAL, detected_patterns
            high_risk_count += matches
        
        if high_risk_count >= 3:  # Multiple sensitive keywords across categories
            detected_patterns.append(f"multi_category:{high_risk_count}")
            return SensitivityLevel.CONFIDENTIAL, detected_patterns
        elif high_risk_count >= 1:
            detected_patterns.append(f"low_risk:{high_risk_count}")
            return SensitivityLevel.INTERNAL, detected_patterns
        
        return SensitivityLevel.PUBLIC, detected_patterns


class UnifiedModelRouter:
    """
    Unified model router combining privacy-first and intelligent routing
    
    🔄 UNIFIED CAPABILITIES:
    - Privacy-first routing with automatic PII detection
    - Multi-provider health monitoring and load balancing
    - Real-time performance metrics and analytics
    - Cost optimization with budget management
    - Quality-aware model selection
    - Intelligent fallback strategies
    - Comprehensive usage tracking integration
    """
    
    def __init__(self,
                 openai_client: Optional[Any] = None,
                 anthropic_client: Optional[EnhancedAnthropicClient] = None,
                 ollama_client: Optional[EnhancedOllamaClient] = None,
                 openrouter_client: Optional[OpenRouterClient] = None,
                 usage_tracker: Optional[GenericUsageTracker] = None,
                 default_strategy: RoutingStrategy = RoutingStrategy.HYBRID_INTELLIGENT,
                 health_check_interval: int = 300):
        """
        Initialize unified router
        
        Args:
            openai_client: OpenAI client instance
            anthropic_client: Enhanced Anthropic client
            ollama_client: Enhanced Ollama client  
            openrouter_client: OpenRouter client
            usage_tracker: Usage tracking instance
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
        self.usage_tracker = usage_tracker
        
        # Privacy detection
        self.privacy_detector = PrivacyDetector()
        
        # Performance tracking
        self.provider_metrics: Dict[str, ProviderMetrics] = {}
        for provider_name in self.clients.keys():
            self.provider_metrics[provider_name] = ProviderMetrics(provider_name)
        
        # Model capability mapping (unified from both routers)
        self.capability_map = {
            TaskType.CODE_GENERATION: ["openai", "anthropic", "openrouter", "ollama"],
            TaskType.REASONING: ["anthropic", "openai", "openrouter", "ollama"],
            TaskType.CREATIVE_WRITING: ["anthropic", "openai", "openrouter", "ollama"],
            TaskType.GENERAL_CHAT: ["ollama", "openrouter", "openai", "anthropic"],
            TaskType.ANALYSIS: ["anthropic", "openai", "openrouter", "ollama"],
            TaskType.SUMMARIZATION: ["openai", "anthropic", "openrouter", "ollama"],
            TaskType.TRANSLATION: ["openai", "anthropic", "openrouter", "ollama"],
            TaskType.Q_AND_A: ["ollama", "openai", "anthropic", "openrouter"]
        }
        
        # Enhanced cost optimization (from both routers)
        self.cost_thresholds = {
            'development': 0.01,
            'testing': 0.05,
            'production': 0.20
        }
        
        # Initialize ModelConfigManager for dynamic configuration
        self.config_manager = get_model_config_manager()
        
        # Load provider cost and quality mappings from configuration
        self._load_provider_configurations()
        
        # Health monitoring
        self._last_health_check = 0.0
        self._health_check_task: Optional[asyncio.Task] = None
    
    def _load_provider_configurations(self):
        """Load provider cost and quality mappings from ModelConfigManager"""
        self.provider_cost_tiers = {}
        self.provider_quality_scores = {}
        
        # Get all providers and their models
        all_providers = self.config_manager.get_all_providers()
        all_models = self.config_manager.get_all_models()
        
        for provider_name, provider_config in all_providers.items():
            self.provider_cost_tiers[provider_name] = {}
            self.provider_quality_scores[provider_name] = {}
            
            # Get models for this provider
            provider_models = self.config_manager.get_models_by_provider(provider_name)
            
            for model_id, model_config in provider_models.items():
                # Set cost (using output cost as it's typically higher)
                self.provider_cost_tiers[provider_name][model_id] = float(model_config.pricing.output_cost_per_1k)
                
                # Set quality score
                self.provider_quality_scores[provider_name][model_id] = model_config.quality_score
            
            # Set default values for provider if no models found
            if not provider_models:
                self.provider_cost_tiers[provider_name]["default"] = 0.01
                self.provider_quality_scores[provider_name]["default"] = 0.8
        
        logger.info("Loaded provider configurations from ModelConfigManager", 
                   providers=len(all_providers), models=len(all_models))
    
    async def initialize(self):
        """Initialize router and all clients"""
        # Initialize all clients
        for provider_name, client in self.clients.items():
            try:
                if hasattr(client, 'initialize'):
                    await client.initialize()
                logger.info("Provider initialized", provider=provider_name)
            except Exception as e:
                logger.error("Failed to initialize provider", provider=provider_name, error=str(e))
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Unified router initialized successfully")
    
    async def close(self):
        """Close router and all clients"""
        if self._health_check_task:
            self._health_check_task.cancel()
            
        for provider_name, client in self.clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                logger.error("Error closing provider", provider=provider_name, error=str(e))
    
    def detect_task_type(self, prompt: str, system_prompt: str = "") -> TaskType:
        """Detect task type from prompt content"""
        combined = f"{prompt} {system_prompt}".lower()
        
        # Task detection patterns
        patterns = {
            TaskType.CODE_GENERATION: ['code', 'function', 'class', 'variable', 'debug', 'programming', 'algorithm', 'python', 'javascript'],
            TaskType.CREATIVE_WRITING: ['story', 'poem', 'creative', 'write', 'compose', 'imagine', 'fiction', 'narrative'],
            TaskType.ANALYSIS: ['analyze', 'compare', 'evaluate', 'research', 'study', 'data', 'examine', 'assess'],
            TaskType.TRANSLATION: ['translate', 'french', 'spanish', 'german', 'chinese', 'japanese', 'language'],
            TaskType.SUMMARIZATION: ['summarize', 'summary', 'brief', 'condense', 'outline', 'key points'],
            TaskType.REASONING: ['explain', 'reason', 'logic', 'solve', 'problem', 'think', 'deduce'],
            TaskType.Q_AND_A: ['what', 'how', 'why', 'when', 'where', 'question', 'answer']
        }
        
        # Score each task type
        scores = {}
        for task_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in combined)
            if score > 0:
                scores[task_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return TaskType.GENERAL_CHAT
    
    async def route_request(self,
                           request: ModelExecutionRequest,
                           strategy: Optional[RoutingStrategy] = None,
                           constraints: Optional[RoutingConstraints] = None,
                           task_type: Optional[TaskType] = None,
                           session_id: Optional[UUID] = None,
                           user_id: str = "system") -> ModelExecutionResponse:
        """
        Route request with unified intelligence
        
        Args:
            request: Model execution request
            strategy: Routing strategy to use
            constraints: Routing constraints
            task_type: Type of task for optimization
            session_id: Session ID for tracking
            user_id: User ID for tracking
        
        Returns:
            Model execution response with routing metadata
        """
        strategy = strategy or self.default_strategy
        constraints = constraints or RoutingConstraints()
        task_type = task_type or self.detect_task_type(request.prompt, request.system_prompt or "")
        
        # Analyze privacy requirements
        sensitivity, privacy_patterns = self.privacy_detector.analyze_sensitivity(
            request.prompt, request.system_prompt or ""
        )
        
        # Override constraints for privacy
        if sensitivity in [SensitivityLevel.RESTRICTED, SensitivityLevel.CONFIDENTIAL]:
            constraints.require_local = True
            constraints.privacy_required = True
            logger.info("Privacy override applied", sensitivity=sensitivity.value, patterns=privacy_patterns)
        
        # Make unified routing decision
        decision = await self._make_unified_routing_decision(
            request, strategy, constraints, task_type, sensitivity
        )
        
        logger.info("Routing decision made",
                   provider=decision.selected_provider,
                   model=decision.selected_model,
                   reasoning=decision.reasoning,
                   sensitivity=decision.sensitivity_level.value,
                   confidence=decision.confidence_score)
        
        # Execute request with selected provider
        start_time = time.time()
        response = await self._execute_with_provider(
            decision.selected_provider,
            decision.selected_model,
            request
        )
        execution_time = time.time() - start_time
        
        # Update metrics
        self.provider_metrics[decision.selected_provider].update_metrics(
            response.success,
            execution_time,
            getattr(response, 'cost', decision.estimated_cost)
        )
        
        # Track usage if tracker available
        if self.usage_tracker and session_id:
            try:
                await self._track_routing_usage(
                    request, response, decision, session_id, user_id, execution_time
                )
            except Exception as e:
                logger.error("Failed to track routing usage", error=str(e))
        
        # Add comprehensive routing metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            "unified_routing": {
                "provider": decision.selected_provider,
                "model": decision.selected_model,
                "strategy": decision.routing_strategy.value,
                "confidence": decision.confidence_score,
                "reasoning": decision.reasoning,
                "sensitivity_level": decision.sensitivity_level.value,
                "privacy_detected": decision.privacy_detected,
                "task_type": task_type.value,
                "estimated_cost": decision.estimated_cost,
                "actual_execution_time": execution_time,
                "decision_factors": decision.decision_factors,
                "alternatives": decision.alternatives_considered,
                "fallbacks": decision.fallback_providers
            }
        })
        
        return response
    
    async def _make_unified_routing_decision(self,
                                           request: ModelExecutionRequest,
                                           strategy: RoutingStrategy,
                                           constraints: RoutingConstraints,
                                           task_type: TaskType,
                                           sensitivity: SensitivityLevel) -> UnifiedRoutingDecision:
        """Make unified routing decision combining all factors"""
        
        # Get available providers
        available_providers = self._get_available_providers(constraints)
        
        if not available_providers:
            raise RuntimeError("No available providers match constraints")
        
        # Apply strategy-specific routing
        if strategy == RoutingStrategy.PRIVACY_FIRST or constraints.privacy_required:
            return await self._privacy_first_routing(sensitivity, task_type, available_providers, request, strategy)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            return await self._cost_optimized_routing(task_type, available_providers, request, strategy)
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return await self._latency_optimized_routing(task_type, available_providers, request, strategy, constraints)
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return await self._quality_optimized_routing(task_type, available_providers, request, strategy)
        elif strategy == RoutingStrategy.LOCAL_PREFERRED:
            return await self._local_preferred_routing(task_type, available_providers, request, strategy)
        else:  # HYBRID_INTELLIGENT, BALANCED, etc.
            return await self._hybrid_intelligent_routing(
                sensitivity, task_type, available_providers, request, strategy, constraints
            )
    
    async def _privacy_first_routing(self, sensitivity: SensitivityLevel, task_type: TaskType,
                                   available_providers: List[str], request: ModelExecutionRequest,
                                   strategy: RoutingStrategy) -> UnifiedRoutingDecision:
        """Privacy-first routing with mandatory local processing for sensitive data"""
        
        if sensitivity in [SensitivityLevel.RESTRICTED, SensitivityLevel.CONFIDENTIAL]:
            # Must use local
            local_providers = [p for p in available_providers if p == "ollama"]
            if not local_providers:
                raise ValueError("Sensitive data detected but no local models available")
            
            model = await self._select_best_model("ollama", request, strategy)
            
            return UnifiedRoutingDecision(
                selected_provider="ollama",
                selected_model=model,
                routing_strategy=strategy,
                use_local=True,
                client_type="ollama",
                reasoning=f"Privacy-first: {sensitivity.value} data requires local processing",
                sensitivity_level=sensitivity,
                estimated_cost=0.001,
                estimated_latency=3.0,
                confidence_score=0.95,
                decision_factors={"privacy": 1.0, "data_sovereignty": 1.0},
                priority_factors=["privacy_required", "data_protection"],
                alternatives_considered=[],
                fallback_providers=[],
                privacy_detected=True,
                quality_score=0.75
            )
        
        # For public/internal data, prefer cloud for quality but respect preferences
        return await self._quality_optimized_routing(task_type, available_providers, request, strategy)
    
    async def _cost_optimized_routing(self, task_type: TaskType, available_providers: List[str],
                                    request: ModelExecutionRequest, strategy: RoutingStrategy) -> UnifiedRoutingDecision:
        """Cost-optimized routing prioritizing lowest cost"""
        
        # Always prefer local for cost optimization
        if "ollama" in available_providers:
            model = await self._select_best_model("ollama", request, strategy)
            
            return UnifiedRoutingDecision(
                selected_provider="ollama",
                selected_model=model,
                routing_strategy=strategy,
                use_local=True,
                client_type="ollama",
                reasoning="Cost-optimized: Local execution eliminates API fees",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=0.0,
                estimated_latency=3.0,
                confidence_score=0.90,
                decision_factors={"cost": 1.0, "local_available": 0.8},
                priority_factors=["zero_api_cost", "cost_optimization"],
                alternatives_considered=[p for p in available_providers if p != "ollama"],
                fallback_providers=[p for p in available_providers if p != "ollama"][:2],
                privacy_detected=False,
                quality_score=0.75
            )
        
        # If no local, find cheapest cloud option
        cloud_providers = [p for p in available_providers if p != "ollama"]
        if cloud_providers:
            cheapest_provider = min(cloud_providers, key=lambda p: self._estimate_provider_cost(p, request))
            model = await self._select_best_model(cheapest_provider, request, strategy)
            cost = self._estimate_provider_cost(cheapest_provider, request)
            
            return UnifiedRoutingDecision(
                selected_provider=cheapest_provider,
                selected_model=model,
                routing_strategy=strategy,
                use_local=False,
                client_type=cheapest_provider,
                reasoning=f"Cost-optimized: {cheapest_provider} is cheapest available cloud option",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=cost,
                estimated_latency=2.0,
                confidence_score=0.85,
                decision_factors={"cost": 0.9, "availability": 0.7},
                priority_factors=["minimum_cloud_cost"],
                alternatives_considered=[p for p in cloud_providers if p != cheapest_provider],
                fallback_providers=[p for p in cloud_providers if p != cheapest_provider][:2],
                privacy_detected=False,
                quality_score=0.80
            )
        
        raise RuntimeError("No providers available for cost-optimized routing")
    
    async def _latency_optimized_routing(self, task_type: TaskType, available_providers: List[str],
                                       request: ModelExecutionRequest, strategy: RoutingStrategy,
                                       constraints: RoutingConstraints) -> UnifiedRoutingDecision:
        """Latency-optimized routing for fastest response"""
        
        # Score providers by latency
        provider_latencies = {}
        for provider in available_providers:
            metrics = self.provider_metrics[provider]
            if metrics.total_requests > 0:
                provider_latencies[provider] = metrics.average_response_time
            else:
                # Default latency estimates
                defaults = {"ollama": 3.0, "openai": 1.5, "anthropic": 2.0, "openrouter": 2.5}
                provider_latencies[provider] = defaults.get(provider, 2.0)
        
        # Select fastest provider within constraints
        fastest_provider = min(provider_latencies, key=provider_latencies.get)
        fastest_latency = provider_latencies[fastest_provider]
        
        # Check if it meets latency constraints
        if constraints.max_latency and fastest_latency > constraints.max_latency:
            raise RuntimeError(f"No provider can meet latency requirement of {constraints.max_latency}s")
        
        model = await self._select_best_model(fastest_provider, request, strategy)
        cost = self._estimate_provider_cost(fastest_provider, request)
        
        return UnifiedRoutingDecision(
            selected_provider=fastest_provider,
            selected_model=model,
            routing_strategy=strategy,
            use_local=(fastest_provider == "ollama"),
            client_type=fastest_provider,
            reasoning=f"Latency-optimized: {fastest_provider} has fastest response time ({fastest_latency:.1f}s)",
            sensitivity_level=SensitivityLevel.PUBLIC,
            estimated_cost=cost,
            estimated_latency=fastest_latency,
            confidence_score=0.88,
            decision_factors={"latency": 1.0, "availability": 0.8},
            priority_factors=["low_latency", "fast_response"],
            alternatives_considered=[p for p in available_providers if p != fastest_provider],
            fallback_providers=[p for p in available_providers if p != fastest_provider][:2],
            privacy_detected=False,
            quality_score=self.provider_quality_scores.get(fastest_provider, {}).get("default", 0.75)
        )
    
    async def _quality_optimized_routing(self, task_type: TaskType, available_providers: List[str],
                                       request: ModelExecutionRequest, strategy: RoutingStrategy) -> UnifiedRoutingDecision:
        """Quality-optimized routing for best output quality"""
        
        # Score providers by quality for this task
        quality_scores = {}
        for provider in available_providers:
            provider_quality = self.provider_quality_scores.get(provider, {"default": 0.75})
            # Use task-specific quality if available, otherwise default
            quality_scores[provider] = max(provider_quality.values())
        
        best_provider = max(quality_scores, key=quality_scores.get)
        best_quality = quality_scores[best_provider]
        
        model = await self._select_best_model(best_provider, request, strategy)
        cost = self._estimate_provider_cost(best_provider, request)
        
        return UnifiedRoutingDecision(
            selected_provider=best_provider,
            selected_model=model,
            routing_strategy=strategy,
            use_local=(best_provider == "ollama"),
            client_type=best_provider,
            reasoning=f"Quality-optimized: {best_provider} has highest quality score ({best_quality:.2f})",
            sensitivity_level=SensitivityLevel.PUBLIC,
            estimated_cost=cost,
            estimated_latency=self._estimate_provider_latency(best_provider),
            confidence_score=0.92,
            decision_factors={"quality": 1.0, "reliability": 0.8},
            priority_factors=["high_quality", "best_output"],
            alternatives_considered=[p for p in available_providers if p != best_provider],
            fallback_providers=[p for p in available_providers if p != best_provider][:2],
            privacy_detected=False,
            quality_score=best_quality
        )
    
    async def _local_preferred_routing(self, task_type: TaskType, available_providers: List[str],
                                     request: ModelExecutionRequest, strategy: RoutingStrategy) -> UnifiedRoutingDecision:
        """Local-preferred routing prioritizing local models"""
        
        if "ollama" in available_providers:
            model = await self._select_best_model("ollama", request, strategy)
            
            return UnifiedRoutingDecision(
                selected_provider="ollama",
                selected_model=model,
                routing_strategy=strategy,
                use_local=True,
                client_type="ollama",
                reasoning="Local-preferred: Using local model as preferred",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=0.0,
                estimated_latency=3.0,
                confidence_score=0.85,
                decision_factors={"local_preference": 1.0, "cost": 1.0},
                priority_factors=["local_preferred", "data_sovereignty"],
                alternatives_considered=[p for p in available_providers if p != "ollama"],
                fallback_providers=[p for p in available_providers if p != "ollama"][:2],
                privacy_detected=False,
                quality_score=0.75
            )
        
        # Fallback to other routing if no local available
        return await self._quality_optimized_routing(task_type, available_providers, request, strategy)
    
    async def _hybrid_intelligent_routing(self, sensitivity: SensitivityLevel, task_type: TaskType,
                                        available_providers: List[str], request: ModelExecutionRequest,
                                        strategy: RoutingStrategy, constraints: RoutingConstraints) -> UnifiedRoutingDecision:
        """Hybrid intelligent routing balancing all factors"""
        
        # Privacy override
        if sensitivity in [SensitivityLevel.RESTRICTED, SensitivityLevel.CONFIDENTIAL]:
            return await self._privacy_first_routing(sensitivity, task_type, available_providers, request, strategy)
        
        # Score all providers comprehensively
        provider_scores = {}
        all_factors = {}
        
        for provider in available_providers:
            factors = await self._calculate_comprehensive_score(provider, request, task_type, constraints)
            # Weighted scoring for hybrid approach
            score = (
                factors["cost"] * 0.25 +
                factors["quality"] * 0.25 +
                factors["latency"] * 0.20 +
                factors["availability"] * 0.15 +
                factors["reliability"] * 0.10 +
                factors["task_preference"] * 0.05
            )
            provider_scores[provider] = score
            all_factors[provider] = factors
        
        best_provider = max(provider_scores, key=provider_scores.get)
        best_score = provider_scores[best_provider]
        best_factors = all_factors[best_provider]
        
        model = await self._select_best_model(best_provider, request, strategy)
        cost = self._estimate_provider_cost(best_provider, request)
        latency = self._estimate_provider_latency(best_provider)
        
        # Determine reasoning
        top_factor = max(best_factors, key=best_factors.get)
        reasoning = f"Hybrid: {best_provider} selected (top factor: {top_factor}={best_factors[top_factor]:.2f})"
        
        return UnifiedRoutingDecision(
            selected_provider=best_provider,
            selected_model=model,
            routing_strategy=strategy,
            use_local=(best_provider == "ollama"),
            client_type=best_provider,
            reasoning=reasoning,
            sensitivity_level=sensitivity,
            estimated_cost=cost,
            estimated_latency=latency,
            confidence_score=best_score,
            decision_factors=best_factors,
            priority_factors=[f for f, v in best_factors.items() if v > 0.7],
            alternatives_considered=[p for p in available_providers if p != best_provider],
            fallback_providers=sorted([p for p in available_providers if p != best_provider], 
                                    key=lambda p: provider_scores[p], reverse=True)[:2],
            privacy_detected=False,
            quality_score=best_factors["quality"]
        )
    
    async def _calculate_comprehensive_score(self, provider: str, request: ModelExecutionRequest,
                                           task_type: TaskType, constraints: RoutingConstraints) -> Dict[str, float]:
        """Calculate comprehensive scoring factors for a provider"""
        
        metrics = self.provider_metrics[provider]
        
        # Cost score (higher = cheaper)
        if provider == "ollama":
            cost_score = 1.0
        else:
            cost_tier = self.provider_cost_tiers.get(provider, {}).get("default", 0.01)
            cost_score = max(0.1, 1.0 - (cost_tier / 0.10))  # Normalize against $0.10 max
        
        # Quality score
        quality_scores = self.provider_quality_scores.get(provider, {"default": 0.75})
        quality_score = max(quality_scores.values())
        
        # Latency score (higher = faster)
        if metrics.total_requests > 0:
            avg_latency = metrics.average_response_time
        else:
            defaults = {"ollama": 3.0, "openai": 1.5, "anthropic": 2.0, "openrouter": 2.5}
            avg_latency = defaults.get(provider, 2.0)
        
        max_latency = constraints.max_latency or 10.0
        latency_score = max(0.1, 1.0 - (avg_latency / max_latency))
        
        # Availability and reliability from metrics
        availability_score = metrics.availability
        reliability_score = metrics.success_rate
        
        # Task preference score
        task_preference_score = 1.0
        if task_type in self.capability_map:
            preferred_providers = self.capability_map[task_type]
            if provider in preferred_providers:
                position = preferred_providers.index(provider)
                task_preference_score = 1.0 - (position * 0.1)
        
        return {
            "cost": cost_score,
            "quality": quality_score,
            "latency": latency_score,
            "availability": availability_score,
            "reliability": reliability_score,
            "task_preference": task_preference_score
        }
    
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
    
    async def _select_best_model(self, provider: str, request: ModelExecutionRequest, strategy: RoutingStrategy) -> str:
        """Select optimal model for chosen provider"""
        
        if provider == "anthropic":
            if strategy == RoutingStrategy.COST_OPTIMIZED:
                return ClaudeModel.CLAUDE_3_HAIKU.value
            elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
                return ClaudeModel.CLAUDE_3_OPUS.value
            else:
                return ClaudeModel.CLAUDE_3_SONNET.value
        
        elif provider == "ollama":
            if hasattr(self.clients["ollama"], 'get_model_recommendations'):
                recommendations = self.clients["ollama"].get_model_recommendations("general", "balanced")
                if recommendations:
                    return recommendations[0].value
            return OllamaModel.LLAMA2_7B_CHAT.value
        
        elif provider == "openai":
            if strategy == RoutingStrategy.COST_OPTIMIZED:
                return "gpt-3.5-turbo"
            elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
                return "gpt-4"
            else:
                return "gpt-3.5-turbo"
        
        elif provider == "openrouter":
            return "auto"
        
        return request.model_id or "default"
    
    def _estimate_provider_cost(self, provider: str, request: ModelExecutionRequest) -> float:
        """Estimate request cost for provider"""
        if provider == "ollama":
            return 0.0
        
        estimated_tokens = len(request.prompt) * 0.75 + request.max_tokens
        cost_per_1k = self.provider_cost_tiers.get(provider, {}).get("default", 0.01)
        
        return (estimated_tokens / 1000) * cost_per_1k
    
    def _estimate_provider_latency(self, provider: str) -> float:
        """Estimate response latency for provider"""
        metrics = self.provider_metrics[provider]
        
        if metrics.total_requests > 0:
            return metrics.average_response_time
        
        defaults = {"ollama": 3.0, "openai": 1.5, "anthropic": 2.0, "openrouter": 2.5}
        return defaults.get(provider, 2.0)
    
    async def _execute_with_provider(self, provider: str, model: str, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """Execute request with specific provider (unified from both routers)"""
        client = self.clients[provider]
        
        try:
            if provider == "anthropic":
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
            
            elif provider == "ollama":
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
            
            elif provider == "openrouter":
                messages = [{"role": "user", "content": request.prompt}]
                if request.system_prompt:
                    messages.insert(0, {"role": "system", "content": request.system_prompt})
                
                response = await client.complete(messages, model)
                
                return ModelExecutionResponse(
                    content=response["content"],
                    provider=ModelProvider.OPENAI,
                    model_id=response["model"],
                    execution_time=response["response_time"],
                    token_usage=response["usage"],
                    success=response["success"],
                    error=response.get("error"),
                    metadata={"cost": response["cost"], "provider": response.get("provider")}
                )
            
            else:
                return ModelExecutionResponse(
                    content="Provider not implemented",
                    provider=ModelProvider.LOCAL,
                    model_id=model,
                    execution_time=0.0,
                    token_usage={},
                    success=False,
                    error=f"Provider {provider} not implemented"
                )
                
        except Exception as e:
            logger.error("Error executing with provider", provider=provider, error=str(e))
            return ModelExecutionResponse(
                content="",
                provider=ModelProvider.LOCAL,
                model_id=model,
                execution_time=0.0,
                token_usage={},
                success=False,
                error=str(e)
            )
    
    async def _track_routing_usage(self, request: ModelExecutionRequest, response: ModelExecutionResponse,
                                 decision: UnifiedRoutingDecision, session_id: UUID, user_id: str,
                                 execution_time: float):
        """Track routing decision and execution with usage tracker"""
        
        # Track the routing operation
        await self.usage_tracker.track_performance(
            OperationType.AGENT_COORDINATION,
            execution_time * 1000,  # Convert to milliseconds
            response.success,
            user_id,
            session_id,
            metadata={
                "routing_strategy": decision.routing_strategy.value,
                "selected_provider": decision.selected_provider,
                "selected_model": decision.selected_model,
                "confidence_score": decision.confidence_score,
                "privacy_detected": decision.privacy_detected,
                "sensitivity_level": decision.sensitivity_level.value
            }
        )
        
        # Track cost if estimated
        if decision.estimated_cost > 0:
            await self.usage_tracker.track_cost(
                CostCategory.MODEL_INFERENCE,
                decision.estimated_cost,
                user_id,
                session_id,
                description=f"Routing to {decision.selected_provider}:{decision.selected_model}"
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
                logger.error("Health monitor error", error=str(e))
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers"""
        for provider_name, client in self.clients.items():
            try:
                test_request = ModelExecutionRequest(
                    prompt="health check",
                    model_id="default",
                    max_tokens=1
                )
                
                start_time = time.time()
                response = await self._execute_with_provider(provider_name, "default", test_request)
                response_time = time.time() - start_time
                
                metrics = self.provider_metrics[provider_name]
                metrics.update_metrics(response.success, response_time, 0.0)
                
                logger.debug("Health check completed", provider=provider_name, success=response.success)
                
            except Exception as e:
                logger.warning("Health check failed", provider=provider_name, error=str(e))
                metrics = self.provider_metrics[provider_name]
                metrics.update_metrics(False, 10.0, 0.0)
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""
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
                "performance_score": metrics.success_rate * metrics.availability,
                "availability": metrics.availability,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "consecutive_failures": metrics.consecutive_failures
            }
        
        return {
            "total_requests": total_requests,
            "total_cost": total_cost,
            "average_cost_per_request": total_cost / total_requests,
            "overall_success_rate": sum(m.successful_requests for m in self.provider_metrics.values()) / total_requests,
            "providers": provider_breakdown,
            "health_status": {name: metrics.availability > 0.8 for name, metrics in self.provider_metrics.items()}
        }


# Convenience function to create unified router
def create_unified_router(**kwargs) -> UnifiedModelRouter:
    """Create a unified router with optional parameters"""
    return UnifiedModelRouter(**kwargs)