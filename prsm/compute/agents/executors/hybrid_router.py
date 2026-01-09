#!/usr/bin/env python3
"""
Hybrid Cloud/Local Model Router for PRSM
=======================================

Intelligent routing system that determines whether to process queries locally 
or via cloud APIs based on:
- Privacy sensitivity
- Cost optimization  
- Performance requirements
- Model availability
- Query complexity

🔀 ROUTING STRATEGIES:
1. **Privacy-First**: Sensitive data → Local models
2. **Cost-Optimized**: Development/testing → Local, Production → Cloud
3. **Performance-Based**: Low latency → Local, High quality → Cloud  
4. **Capability-Aware**: Code tasks → CodeLlama, General → GPT/Claude

🛡️ PRIVACY DETECTION:
- PII (SSN, emails, phone numbers)
- Financial data (credit cards, account numbers)
- Medical information (symptoms, conditions)
- Personal details (names, addresses)
- Proprietary/confidential content
"""

import re
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog

# Import PRSM clients
from .api_clients import (
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelProvider
)
from .openrouter_client import OpenRouterClient
from .ollama_client import OllamaClient

logger = structlog.get_logger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies"""
    PRIVACY_FIRST = "privacy_first"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_BASED = "performance_based"
    CAPABILITY_AWARE = "capability_aware"
    HYBRID_INTELLIGENT = "hybrid_intelligent"


class SensitivityLevel(Enum):
    """Data sensitivity classification"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class RoutingDecision:
    """Result of routing analysis"""
    use_local: bool
    client_type: str  # 'ollama' or 'openrouter'
    model_id: str
    reasoning: str
    sensitivity_level: SensitivityLevel
    estimated_cost: float
    priority_factors: List[str] = field(default_factory=list)


class PrivacyDetector:
    """Detects sensitive content that should stay local"""
    
    def __init__(self):
        # Regex patterns for sensitive data
        self.patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s?\d{3}-\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'bank_account': re.compile(r'\b\d{8,17}\b'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }
        
        # Keywords indicating sensitive content
        self.sensitive_keywords = {
            'financial': ['salary', 'income', 'bank', 'account', 'credit', 'loan', 'mortgage', 'investment'],
            'medical': ['diagnosis', 'symptoms', 'medication', 'treatment', 'doctor', 'hospital', 'patient'],
            'personal': ['password', 'confidential', 'private', 'secret', 'internal', 'proprietary'],
            'legal': ['lawsuit', 'contract', 'agreement', 'litigation', 'attorney', 'legal']
        }
    
    def analyze_sensitivity(self, text: str, system_prompt: str = "") -> SensitivityLevel:
        """Analyze text for sensitive content"""
        combined_text = f"{text} {system_prompt}".lower()
        
        # Check for PII patterns
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(combined_text):
                logger.info("PII detected", pattern_type=pattern_name)
                return SensitivityLevel.RESTRICTED
        
        # Check for sensitive keywords
        high_risk_count = 0
        for category, keywords in self.sensitive_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            if matches >= 2:  # Multiple matches in same category
                logger.info("Sensitive content detected", category=category, matches=matches)
                return SensitivityLevel.CONFIDENTIAL
            high_risk_count += matches
        
        if high_risk_count >= 3:  # Multiple sensitive keywords across categories
            return SensitivityLevel.CONFIDENTIAL
        elif high_risk_count >= 1:
            return SensitivityLevel.INTERNAL
        
        return SensitivityLevel.PUBLIC


class HybridModelRouter:
    """Intelligent router for cloud/local model selection"""
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 ollama_base_url: str = "http://localhost:11434",
                 default_strategy: RoutingStrategy = RoutingStrategy.HYBRID_INTELLIGENT):
        
        self.strategy = default_strategy
        self.privacy_detector = PrivacyDetector()
        
        # Initialize clients
        self.openrouter_client = OpenRouterClient(openrouter_api_key) if openrouter_api_key else None
        self.ollama_client = OllamaClient(ollama_base_url)
        
        # Model capability mapping
        self.capability_map = {
            'code': ['codellama:7b', 'gpt-4', 'claude-3-sonnet'],
            'creative': ['gpt-4', 'claude-3-opus', 'llama3.2:3b'],
            'analytical': ['claude-3-sonnet', 'gpt-4-turbo', 'mistral:7b'],
            'general': ['gpt-3.5-turbo', 'claude-3-haiku', 'llama3.2:1b'],
            'multilingual': ['gpt-4', 'claude-3-sonnet', 'llama3.2:1b']
        }
        
        # Cost optimization thresholds
        self.cost_thresholds = {
            'development': 0.01,  # $0.01 per request
            'testing': 0.05,      # $0.05 per request
            'production': 0.20    # $0.20 per request
        }
    
    async def initialize(self):
        """Initialize both clients"""
        await self.ollama_client.initialize()
        if self.openrouter_client:
            await self.openrouter_client.initialize()
    
    async def close(self):
        """Close client connections"""
        await self.ollama_client.close()
        if self.openrouter_client:
            await self.openrouter_client.close()
    
    def detect_query_type(self, prompt: str, system_prompt: str = "") -> str:
        """Detect the type of query for capability-aware routing"""
        combined = f"{prompt} {system_prompt}".lower()
        
        # Code-related keywords
        code_keywords = ['code', 'function', 'class', 'variable', 'debug', 'programming', 'algorithm']
        if any(keyword in combined for keyword in code_keywords):
            return 'code'
        
        # Creative keywords
        creative_keywords = ['story', 'poem', 'creative', 'write', 'compose', 'imagine']
        if any(keyword in combined for keyword in creative_keywords):
            return 'creative'
        
        # Analytical keywords
        analytical_keywords = ['analyze', 'compare', 'evaluate', 'research', 'study', 'data']
        if any(keyword in combined for keyword in analytical_keywords):
            return 'analytical'
        
        # Multilingual keywords
        multilingual_keywords = ['translate', 'french', 'spanish', 'german', 'chinese', 'japanese']
        if any(keyword in combined for keyword in multilingual_keywords):
            return 'multilingual'
        
        return 'general'
    
    async def make_routing_decision(self, request: ModelExecutionRequest) -> RoutingDecision:
        """Make intelligent routing decision"""
        
        # Analyze sensitivity
        sensitivity = self.privacy_detector.analyze_sensitivity(
            request.prompt, 
            request.system_prompt or ""
        )
        
        # Detect query capability requirements
        query_type = self.detect_query_type(request.prompt, request.system_prompt or "")
        
        # Get available models
        local_models = await self.ollama_client.list_available_models()
        cloud_models = self.openrouter_client.list_available_models() if self.openrouter_client else []
        
        # Apply routing strategy
        if self.strategy == RoutingStrategy.PRIVACY_FIRST:
            return self._privacy_first_routing(sensitivity, query_type, local_models, cloud_models, request)
        
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_routing(query_type, local_models, cloud_models, request)
        
        elif self.strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._performance_based_routing(query_type, local_models, cloud_models, request)
        
        elif self.strategy == RoutingStrategy.CAPABILITY_AWARE:
            return self._capability_aware_routing(query_type, local_models, cloud_models, request)
        
        else:  # HYBRID_INTELLIGENT
            return self._hybrid_intelligent_routing(sensitivity, query_type, local_models, cloud_models, request)
    
    def _capability_aware_routing(self, query_type: str, local_models: List[str], 
                                 cloud_models: List[str], request: ModelExecutionRequest) -> RoutingDecision:
        """Route based on model capabilities for query type"""
        
        # Get best models for this query type
        preferred_models = self.capability_map.get(query_type, self.capability_map['general'])
        
        # Check cloud first for quality
        for model in preferred_models:
            if model in cloud_models:
                cloud_cost = self._estimate_cloud_cost(model, request)
                return RoutingDecision(
                    use_local=False,
                    client_type='openrouter',
                    model_id=model,
                    reasoning=f"Capability-aware: {model} best for {query_type} tasks",
                    sensitivity_level=SensitivityLevel.PUBLIC,
                    estimated_cost=cloud_cost,
                    priority_factors=["capability_match", "quality_priority"]
                )
        
        # Check local models
        for model in preferred_models:
            if model in local_models:
                return RoutingDecision(
                    use_local=True,
                    client_type='ollama',
                    model_id=model,
                    reasoning=f"Capability-aware: {model} available locally for {query_type}",
                    sensitivity_level=SensitivityLevel.PUBLIC,
                    estimated_cost=0.001,
                    priority_factors=["capability_match", "local_available"]
                )
        
        # Fallback to best available model
        if cloud_models:
            best_cloud = self._select_best_model(query_type, cloud_models, 'cloud')
            cloud_cost = self._estimate_cloud_cost(best_cloud, request)
            return RoutingDecision(
                use_local=False,
                client_type='openrouter', 
                model_id=best_cloud,
                reasoning=f"Capability-aware: Using {best_cloud} as fallback for {query_type}",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=cloud_cost,
                priority_factors=["fallback_cloud"]
            )
        
        if local_models:
            best_local = self._select_best_model(query_type, local_models, 'local')
            return RoutingDecision(
                use_local=True,
                client_type='ollama',
                model_id=best_local,
                reasoning=f"Capability-aware: Using {best_local} as local fallback for {query_type}",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=0.001,
                priority_factors=["fallback_local"]
            )
        
        raise ValueError("No models available for capability-aware routing")
    
    def _performance_based_routing(self, query_type: str, local_models: List[str],
                                  cloud_models: List[str], request: ModelExecutionRequest) -> RoutingDecision:
        """Route based on performance requirements"""
        
        # For low-latency requirements, prefer local
        if request.max_tokens < 50:  # Short responses
            if local_models:
                best_local = self._select_best_model(query_type, local_models, 'local')
                return RoutingDecision(
                    use_local=True,
                    client_type='ollama',
                    model_id=best_local,
                    reasoning="Performance-based: Low latency requirement - using local",
                    sensitivity_level=SensitivityLevel.PUBLIC,
                    estimated_cost=0.001,
                    priority_factors=["low_latency", "short_response"]
                )
        
        # For high-quality requirements, prefer cloud
        if request.max_tokens > 200:  # Long, complex responses
            if cloud_models:
                best_cloud = self._select_best_model(query_type, cloud_models, 'cloud')
                cloud_cost = self._estimate_cloud_cost(best_cloud, request)
                return RoutingDecision(
                    use_local=False,
                    client_type='openrouter',
                    model_id=best_cloud,
                    reasoning="Performance-based: High quality requirement - using cloud",
                    sensitivity_level=SensitivityLevel.PUBLIC,
                    estimated_cost=cloud_cost,
                    priority_factors=["high_quality", "complex_response"]
                )
        
        # Balanced preference for moderate requests
        if local_models:
            best_local = self._select_best_model(query_type, local_models, 'local')
            return RoutingDecision(
                use_local=True,
                client_type='ollama',
                model_id=best_local,
                reasoning="Performance-based: Balanced requirement - using local",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=0.001,
                priority_factors=["balanced_performance"]
            )
        
        # Cloud fallback
        if cloud_models:
            best_cloud = self._select_best_model(query_type, cloud_models, 'cloud')
            cloud_cost = self._estimate_cloud_cost(best_cloud, request)
            return RoutingDecision(
                use_local=False,
                client_type='openrouter',
                model_id=best_cloud,
                reasoning="Performance-based: Cloud fallback",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=cloud_cost,
                priority_factors=["cloud_fallback"]
            )
        
        raise ValueError("No models available for performance-based routing")
    
    def _privacy_first_routing(self, sensitivity: SensitivityLevel, query_type: str, 
                              local_models: List[str], cloud_models: List[str],
                              request: ModelExecutionRequest) -> RoutingDecision:
        """Route based on privacy requirements"""
        
        if sensitivity in [SensitivityLevel.RESTRICTED, SensitivityLevel.CONFIDENTIAL]:
            # Must use local
            if local_models:
                best_local = self._select_best_model(query_type, local_models, 'local')
                return RoutingDecision(
                    use_local=True,
                    client_type='ollama',
                    model_id=best_local,
                    reasoning=f"Privacy-first: {sensitivity.value} data must stay local",
                    sensitivity_level=sensitivity,
                    estimated_cost=0.001,  # Local cost estimate
                    priority_factors=["privacy_required", "data_sovereignty"]
                )
            else:
                raise ValueError("Sensitive data detected but no local models available")
        
        # For public/internal data, prefer cloud for quality
        if cloud_models:
            best_cloud = self._select_best_model(query_type, cloud_models, 'cloud')
            cloud_cost = self._estimate_cloud_cost(best_cloud, request)
            return RoutingDecision(
                use_local=False,
                client_type='openrouter',
                model_id=best_cloud,
                reasoning=f"Privacy-first: {sensitivity.value} data safe for cloud processing",
                sensitivity_level=sensitivity,
                estimated_cost=cloud_cost,
                priority_factors=["quality_optimization", "cloud_capabilities"]
            )
        
        # Fallback to local
        best_local = self._select_best_model(query_type, local_models, 'local')
        return RoutingDecision(
            use_local=True,
            client_type='ollama',
            model_id=best_local,
            reasoning="Privacy-first: Fallback to local (cloud unavailable)",
            sensitivity_level=sensitivity,
            estimated_cost=0.001,
            priority_factors=["local_fallback"]
        )
    
    def _cost_optimized_routing(self, query_type: str, local_models: List[str], 
                               cloud_models: List[str], request: ModelExecutionRequest) -> RoutingDecision:
        """Route to minimize cost"""
        
        # Always prefer local for cost optimization
        if local_models:
            best_local = self._select_best_model(query_type, local_models, 'local')
            return RoutingDecision(
                use_local=True,
                client_type='ollama',
                model_id=best_local,
                reasoning="Cost-optimized: Local execution eliminates API fees",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=0.001,
                priority_factors=["zero_api_cost", "development_efficiency"]
            )
        
        # If no local models, use cheapest cloud option
        if cloud_models:
            cheapest_cloud = self._select_cheapest_cloud_model(cloud_models)
            cloud_cost = self._estimate_cloud_cost(cheapest_cloud, request)
            return RoutingDecision(
                use_local=False,
                client_type='openrouter',
                model_id=cheapest_cloud,
                reasoning="Cost-optimized: Using cheapest cloud model",
                sensitivity_level=SensitivityLevel.PUBLIC,
                estimated_cost=cloud_cost,
                priority_factors=["minimum_cloud_cost"]
            )
        
        raise ValueError("No models available for cost-optimized routing")
    
    def _hybrid_intelligent_routing(self, sensitivity: SensitivityLevel, query_type: str,
                                   local_models: List[str], cloud_models: List[str],
                                   request: ModelExecutionRequest) -> RoutingDecision:
        """Intelligent hybrid routing considering all factors"""
        
        # Privacy override
        if sensitivity in [SensitivityLevel.RESTRICTED, SensitivityLevel.CONFIDENTIAL]:
            if local_models:
                best_local = self._select_best_model(query_type, local_models, 'local')
                return RoutingDecision(
                    use_local=True,
                    client_type='ollama',
                    model_id=best_local,
                    reasoning=f"Hybrid: Privacy override - {sensitivity.value} data requires local processing",
                    sensitivity_level=sensitivity,
                    estimated_cost=0.001,
                    priority_factors=["privacy_override", "data_protection"]
                )
        
        # Quality vs Cost tradeoff for non-sensitive data
        token_estimate = len(request.prompt.split()) + request.max_tokens
        
        # For small requests, prefer quality (cloud)
        if token_estimate < 100 and cloud_models:
            best_cloud = self._select_best_model(query_type, cloud_models, 'cloud')
            cloud_cost = self._estimate_cloud_cost(best_cloud, request)
            
            if cloud_cost < self.cost_thresholds['production']:
                return RoutingDecision(
                    use_local=False,
                    client_type='openrouter',
                    model_id=best_cloud,
                    reasoning="Hybrid: Small request - prioritizing quality via cloud",
                    sensitivity_level=sensitivity,
                    estimated_cost=cloud_cost,
                    priority_factors=["quality_priority", "small_request"]
                )
        
        # For large requests or cost-sensitive scenarios, prefer local
        if local_models:
            best_local = self._select_best_model(query_type, local_models, 'local')
            return RoutingDecision(
                use_local=True,
                client_type='ollama',
                model_id=best_local,
                reasoning="Hybrid: Large request or cost optimization - using local",
                sensitivity_level=sensitivity,
                estimated_cost=0.001,
                priority_factors=["cost_efficiency", "local_capability"]
            )
        
        # Fallback to cloud
        if cloud_models:
            best_cloud = self._select_best_model(query_type, cloud_models, 'cloud')
            cloud_cost = self._estimate_cloud_cost(best_cloud, request)
            return RoutingDecision(
                use_local=False,
                client_type='openrouter',
                model_id=best_cloud,
                reasoning="Hybrid: Fallback to cloud (no local models available)",
                sensitivity_level=sensitivity,
                estimated_cost=cloud_cost,
                priority_factors=["cloud_fallback"]
            )
        
        raise ValueError("No models available for hybrid routing")
    
    def _select_best_model(self, query_type: str, available_models: List[str], client_type: str) -> str:
        """Select best model for query type from available options"""
        
        # Get preferred models for this query type
        preferred = self.capability_map.get(query_type, self.capability_map['general'])
        
        # Find intersection with available models
        for model in preferred:
            if model in available_models:
                return model
        
        # If no preferred model available, return first available
        return available_models[0] if available_models else None
    
    def _select_cheapest_cloud_model(self, cloud_models: List[str]) -> str:
        """Select cheapest cloud model"""
        # Preference order by cost (cheapest first)
        cost_order = ['claude-3-haiku', 'gpt-3.5-turbo', 'gemini-pro', 'claude-3-sonnet', 'gpt-4', 'claude-3-opus']
        
        for model in cost_order:
            if model in cloud_models:
                return model
        
        return cloud_models[0] if cloud_models else None
    
    def _estimate_cloud_cost(self, model_id: str, request: ModelExecutionRequest) -> float:
        """Estimate cloud API cost"""
        if not self.openrouter_client:
            return 0.0
        
        token_estimate = len(request.prompt.split()) + request.max_tokens
        return float(self.openrouter_client.get_cost_estimate(model_id, token_estimate, request.max_tokens))
    
    async def execute_with_routing(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """Execute request with intelligent routing"""
        
        # Make routing decision
        decision = await self.make_routing_decision(request)
        
        logger.info("Routing decision made",
                   use_local=decision.use_local,
                   model=decision.model_id,
                   reasoning=decision.reasoning,
                   sensitivity=decision.sensitivity_level.value)
        
        # Execute with chosen client
        if decision.use_local:
            # Update request model_id for local execution
            local_request = ModelExecutionRequest(
                prompt=request.prompt,
                model_id=decision.model_id,
                provider=ModelProvider.LOCAL,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                context=request.context
            )
            response = await self.ollama_client.execute(local_request)
        else:
            # Update request model_id for cloud execution
            cloud_request = ModelExecutionRequest(
                prompt=request.prompt,
                model_id=decision.model_id,
                provider=ModelProvider.OPENAI,  # OpenRouter uses this
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                context=request.context
            )
            response = await self.openrouter_client.execute(cloud_request)
        
        # Add routing metadata to response
        if response.metadata is None:
            response.metadata = {}
        
        response.metadata.update({
            'routing_decision': {
                'use_local': decision.use_local,
                'client_type': decision.client_type,
                'reasoning': decision.reasoning,
                'sensitivity_level': decision.sensitivity_level.value,
                'estimated_cost': decision.estimated_cost,
                'priority_factors': decision.priority_factors
            }
        })
        
        return response