#!/usr/bin/env python3
"""
NWTN Meta-Reasoning Engine Routing and Load Balancing
====================================================

Load balancing, engine selection, and request routing components
for the NWTN meta-reasoning system.
"""

from __future__ import annotations
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

from .types import (
    ReasoningEngine,
    LoadBalancingStrategy,
    LoadBalancingMode,
    EngineWorkload
)


@dataclass
class LoadBalancingMetrics:
    """Metrics for load balancing performance"""
    
    total_requests: int = 0
    balanced_requests: int = 0
    failed_balancing_attempts: int = 0
    engine_utilization: Dict[ReasoningEngine, float] = field(default_factory=dict)
    response_time_distribution: Dict[ReasoningEngine, List[float]] = field(default_factory=dict)
    load_balancing_overhead: float = 0.0
    strategy_switches: int = 0
    last_strategy_switch: Optional[datetime] = None
    
    def update_request_metrics(self, engine_type: ReasoningEngine, response_time: float, success: bool):
        """Update metrics for a completed request"""
        self.total_requests += 1
        
        if success:
            self.balanced_requests += 1
            
            # Update response time distribution
            if engine_type not in self.response_time_distribution:
                self.response_time_distribution[engine_type] = []
            self.response_time_distribution[engine_type].append(response_time)
            
            # Keep only last 100 response times
            if len(self.response_time_distribution[engine_type]) > 100:
                self.response_time_distribution[engine_type].pop(0)
        else:
            self.failed_balancing_attempts += 1
    
    def get_balancing_success_rate(self) -> float:
        """Get load balancing success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.balanced_requests / self.total_requests
    
    def get_average_response_time(self, engine_type: ReasoningEngine) -> float:
        """Get average response time for an engine"""
        if engine_type not in self.response_time_distribution:
            return 0.0
        
        times = self.response_time_distribution[engine_type]
        if not times:
            return 0.0
        
        return statistics.mean(times)


class AdaptiveSelectionStrategy(Enum):
    """Adaptive selection strategies"""
    CONTEXT_AWARE = "context_aware"           # Select based on query context
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Select based on historical performance
    PROBLEM_TYPE_MATCHING = "problem_type_matching"   # Select based on problem type analysis
    MULTI_CRITERIA = "multi_criteria"         # Multi-criteria decision making
    MACHINE_LEARNING = "machine_learning"     # ML-based selection (future)
    HYBRID_ADAPTIVE = "hybrid_adaptive"       # Hybrid approach combining multiple strategies


class ProblemType(Enum):
    """Types of problems for adaptive selection"""
    LOGICAL_REASONING = "logical_reasoning"
    PATTERN_RECOGNITION = "pattern_recognition"
    CAUSAL_ANALYSIS = "causal_analysis"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    SCENARIO_ANALYSIS = "scenario_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    OPTIMIZATION = "optimization"
    UNKNOWN = "unknown"


class ContextualFactor(Enum):
    """Contextual factors for adaptive selection"""
    DOMAIN = "domain"                         # Problem domain (medical, financial, etc.)
    URGENCY = "urgency"                       # Time sensitivity
    COMPLEXITY = "complexity"                 # Problem complexity level
    UNCERTAINTY = "uncertainty"               # Level of uncertainty in data
    EVIDENCE_STRENGTH = "evidence_strength"   # Strength of available evidence
    STAKEHOLDER_REQUIREMENTS = "stakeholder_requirements"  # Specific requirements
    REGULATORY_CONSTRAINTS = "regulatory_constraints"      # Compliance requirements
    RESOURCE_CONSTRAINTS = "resource_constraints"          # Available resources
    QUALITY_REQUIREMENTS = "quality_requirements"          # Quality expectations
    EXPLAINABILITY_NEEDS = "explainability_needs"         # Need for explanations


@dataclass
class AdaptiveSelectionContext:
    """Context for adaptive engine selection"""
    
    query: str
    problem_type: ProblemType = ProblemType.UNKNOWN
    contextual_factors: Dict[ContextualFactor, Any] = field(default_factory=dict)
    historical_performance: Dict[ReasoningEngine, float] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.contextual_factors:
            self.contextual_factors = {}
        if not self.historical_performance:
            self.historical_performance = {}
        if not self.user_preferences:
            self.user_preferences = {}
        if not self.constraints:
            self.constraints = {}


@dataclass
class EngineSelectionScore:
    """Score for engine selection"""
    
    engine: ReasoningEngine
    total_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def __post_init__(self):
        if not self.component_scores:
            self.component_scores = {}
        if not self.reasoning:
            self.reasoning = []


class AdaptiveEngineSelector:
    """Adaptive engine selector that chooses engines based on context and performance"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.strategy = AdaptiveSelectionStrategy.HYBRID_ADAPTIVE
        self.enabled = True
        
        # Engine suitability mappings
        self.engine_problem_type_mapping = {
            ProblemType.LOGICAL_REASONING: [ReasoningEngine.CLAUDE, ReasoningEngine.GPT4],
            ProblemType.PATTERN_RECOGNITION: [ReasoningEngine.GEMINI_PRO, ReasoningEngine.CLAUDE],
            ProblemType.CAUSAL_ANALYSIS: [ReasoningEngine.CLAUDE, ReasoningEngine.GPT4],
            ProblemType.UNCERTAINTY_QUANTIFICATION: [ReasoningEngine.GPT4, ReasoningEngine.GEMINI_PRO],
            ProblemType.HYPOTHESIS_GENERATION: [ReasoningEngine.CLAUDE, ReasoningEngine.COHERE],
            ProblemType.SCENARIO_ANALYSIS: [ReasoningEngine.GPT4, ReasoningEngine.CLAUDE],
            ProblemType.COMPARATIVE_ANALYSIS: [ReasoningEngine.GEMINI_PRO, ReasoningEngine.CLAUDE],
            ProblemType.PREDICTION: [ReasoningEngine.GPT4, ReasoningEngine.CLAUDE],
            ProblemType.CLASSIFICATION: [ReasoningEngine.GEMINI_PRO, ReasoningEngine.GPT4],
            ProblemType.OPTIMIZATION: [ReasoningEngine.CLAUDE, ReasoningEngine.GPT4]
        }
        
        # Context keywords for problem type detection
        self.problem_type_keywords = {
            ProblemType.LOGICAL_REASONING: ["logical", "proof", "theorem", "deduction", "inference"],
            ProblemType.PATTERN_RECOGNITION: ["pattern", "trend", "similarity", "correlation", "regularity"],
            ProblemType.CAUSAL_ANALYSIS: ["cause", "effect", "influence", "impact", "mechanism"],
            ProblemType.UNCERTAINTY_QUANTIFICATION: ["probability", "likelihood", "risk", "uncertainty", "confidence"],
            ProblemType.HYPOTHESIS_GENERATION: ["hypothesis", "theory", "explanation", "possible", "might"],
            ProblemType.SCENARIO_ANALYSIS: ["scenario", "alternative", "what if", "counterfactual", "simulation"],
            ProblemType.COMPARATIVE_ANALYSIS: ["compare", "contrast", "similarity", "difference", "analogy"],
            ProblemType.PREDICTION: ["predict", "forecast", "future", "estimate", "projection"],
            ProblemType.CLASSIFICATION: ["classify", "categorize", "group", "type", "kind"],
            ProblemType.OPTIMIZATION: ["optimize", "best", "maximum", "minimum", "improve"]
        }
        
        # Selection history for learning
        self.selection_history = []
        self.performance_history = {}
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.performance_window = 100  # Last 100 selections
        
        # Scoring weights
        self.scoring_weights = {
            "problem_type_match": 0.3,
            "historical_performance": 0.25,
            "current_health": 0.2,
            "load_balance": 0.15,
            "contextual_suitability": 0.1
        }
    
    def detect_problem_type(self, query: str, context: Dict[str, Any] = None) -> ProblemType:
        """Detect problem type from query and context"""
        query_lower = str(query).lower()
        
        # Check for explicit problem type in context
        if context and "problem_type" in context:
            try:
                return ProblemType(context["problem_type"])
            except ValueError:
                pass
        
        # Keyword-based detection
        type_scores = {}
        for problem_type, keywords in self.problem_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                type_scores[problem_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return ProblemType.UNKNOWN
    
    def extract_contextual_factors(self, context: Dict[str, Any]) -> Dict[ContextualFactor, Any]:
        """Extract contextual factors from context"""
        factors = {}
        
        # Map context keys to contextual factors
        context_mapping = {
            "domain": ContextualFactor.DOMAIN,
            "urgency": ContextualFactor.URGENCY,
            "complexity": ContextualFactor.COMPLEXITY,
            "uncertainty": ContextualFactor.UNCERTAINTY,
            "evidence_strength": ContextualFactor.EVIDENCE_STRENGTH,
            "stakeholders": ContextualFactor.STAKEHOLDER_REQUIREMENTS,
            "regulatory": ContextualFactor.REGULATORY_CONSTRAINTS,
            "resources": ContextualFactor.RESOURCE_CONSTRAINTS,
            "quality": ContextualFactor.QUALITY_REQUIREMENTS,
            "explainability": ContextualFactor.EXPLAINABILITY_NEEDS
        }
        
        for key, factor in context_mapping.items():
            if key in context:
                factors[factor] = context[key]
        
        return factors
    
    def select_engines_adaptively(self, query: str, context: Dict[str, Any] = None, 
                                 num_engines: int = 3) -> List[ReasoningEngine]:
        """Select engines adaptively based on query and context"""
        if not self.enabled:
            return list(ReasoningEngine)[:num_engines]
        
        # Create adaptive selection context
        problem_type = self.detect_problem_type(query, context)
        contextual_factors = self.extract_contextual_factors(context or {})
        
        selection_context = AdaptiveSelectionContext(
            query=query,
            problem_type=problem_type,
            contextual_factors=contextual_factors,
            historical_performance=self._get_historical_performance(),
            user_preferences=context.get("user_preferences", {}) if context else {},
            constraints=context.get("constraints", {}) if context else {}
        )
        
        # Score all engines
        engine_scores = []
        for engine_type in ReasoningEngine:
            score = self._calculate_engine_score(engine_type, selection_context)
            engine_scores.append(score)
        
        # Sort by total score (descending)
        engine_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Select top engines
        selected_engines = [score.engine for score in engine_scores[:num_engines]]
        
        # Record selection for learning
        self._record_selection(selection_context, selected_engines, engine_scores)
        
        return selected_engines
    
    def _calculate_engine_score(self, engine_type: ReasoningEngine, 
                               selection_context: AdaptiveSelectionContext) -> EngineSelectionScore:
        """Calculate comprehensive score for an engine"""
        score = EngineSelectionScore(engine=engine_type, total_score=0.0)
        
        # 1. Problem type matching score
        problem_type_score = self._calculate_problem_type_score(engine_type, selection_context)
        score.component_scores["problem_type_match"] = problem_type_score
        
        # 2. Historical performance score
        historical_score = self._calculate_historical_performance_score(engine_type, selection_context)
        score.component_scores["historical_performance"] = historical_score
        
        # 3. Current health score
        health_score = self._calculate_health_score(engine_type)
        score.component_scores["current_health"] = health_score
        
        # 4. Load balance score
        load_score = self._calculate_load_balance_score(engine_type)
        score.component_scores["load_balance"] = load_score
        
        # 5. Contextual suitability score
        contextual_score = self._calculate_contextual_suitability_score(engine_type, selection_context)
        score.component_scores["contextual_suitability"] = contextual_score
        
        # Calculate weighted total score
        score.total_score = sum(
            self.scoring_weights[component] * component_score
            for component, component_score in score.component_scores.items()
        )
        
        # Generate reasoning
        score.reasoning = self._generate_selection_reasoning(engine_type, score.component_scores)
        
        # Calculate confidence
        score.confidence = min(1.0, max(0.0, score.total_score))
        
        return score
    
    def _calculate_problem_type_score(self, engine_type: ReasoningEngine, 
                                     selection_context: AdaptiveSelectionContext) -> float:
        """Calculate problem type matching score"""
        problem_type = selection_context.problem_type
        
        if problem_type == ProblemType.UNKNOWN:
            return 0.5  # Neutral score for unknown problems
        
        suitable_engines = self.engine_problem_type_mapping.get(problem_type, [])
        
        if engine_type in suitable_engines:
            # Higher score for primary suitability
            return 1.0 if suitable_engines.index(engine_type) == 0 else 0.8
        
        return 0.2  # Low score for non-suitable engines
    
    def _calculate_historical_performance_score(self, engine_type: ReasoningEngine,
                                              selection_context: AdaptiveSelectionContext) -> float:
        """Calculate historical performance score"""
        if engine_type not in self.performance_history:
            return 0.5  # Neutral score for engines with no history
        
        engine_history = self.performance_history[engine_type]
        if not engine_history:
            return 0.5
        
        # Calculate recent performance (last 20 executions)
        recent_performance = engine_history[-20:]
        avg_performance = statistics.mean(recent_performance)
        
        # Normalize to 0-1 scale
        return min(1.0, max(0.0, avg_performance))
    
    def _calculate_health_score(self, engine_type: ReasoningEngine) -> float:
        """Calculate current health score"""
        try:
            health_report = self.meta_reasoning_engine.health_monitor.get_engine_health_report(engine_type)
            return health_report.get('health_score', 0.5) / 100.0  # Normalize to 0-1
        except:
            return 0.5
    
    def _calculate_load_balance_score(self, engine_type: ReasoningEngine) -> float:
        """Calculate load balance score (inverse of current load)"""
        try:
            if hasattr(self.meta_reasoning_engine, 'load_balancer') and engine_type in self.meta_reasoning_engine.load_balancer.engine_workloads:
                workload = self.meta_reasoning_engine.load_balancer.engine_workloads[engine_type]
                return 1.0 - workload.calculate_load_score()
        except:
            pass
        return 0.5
    
    def _calculate_contextual_suitability_score(self, engine_type: ReasoningEngine,
                                               selection_context: AdaptiveSelectionContext) -> float:
        """Calculate contextual suitability score"""
        score = 0.5  # Base score
        
        # Adjust based on contextual factors
        factors = selection_context.contextual_factors
        
        # Domain-specific adjustments
        if ContextualFactor.DOMAIN in factors:
            domain = factors[ContextualFactor.DOMAIN]
            if domain == "medical" and engine_type == ReasoningEngine.CLAUDE:
                score += 0.3
            elif domain == "financial" and engine_type == ReasoningEngine.GPT4:
                score += 0.3
            elif domain == "legal" and engine_type == ReasoningEngine.CLAUDE:
                score += 0.3
        
        # Urgency adjustments
        if ContextualFactor.URGENCY in factors:
            urgency = factors[ContextualFactor.URGENCY]
            if urgency == "high" and engine_type in [ReasoningEngine.CLAUDE, ReasoningEngine.GPT4]:
                score += 0.2  # These engines are typically faster
        
        # Quality requirements
        if ContextualFactor.QUALITY_REQUIREMENTS in factors:
            quality = factors[ContextualFactor.QUALITY_REQUIREMENTS]
            if quality == "high" and engine_type in [ReasoningEngine.CLAUDE, ReasoningEngine.GPT4]:
                score += 0.2  # These engines typically provide higher quality
        
        return min(1.0, max(0.0, score))
    
    def _generate_selection_reasoning(self, engine_type: ReasoningEngine, 
                                     component_scores: Dict[str, float]) -> List[str]:
        """Generate reasoning for engine selection"""
        reasoning = []
        
        for component, score in component_scores.items():
            if score > 0.7:
                reasoning.append(f"High {component.replace('_', ' ')}: {score:.2f}")
            elif score < 0.3:
                reasoning.append(f"Low {component.replace('_', ' ')}: {score:.2f}")
        
        return reasoning
    
    def _get_historical_performance(self) -> Dict[ReasoningEngine, float]:
        """Get historical performance for all engines"""
        performance = {}
        
        for engine_type in ReasoningEngine:
            try:
                profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
                if profile:
                    # Combine execution time and quality for overall performance
                    time_score = 1.0 / (1.0 + profile.avg_response_time)
                    quality_score = profile.avg_quality_score
                    performance[engine_type] = (time_score + quality_score) / 2.0
                else:
                    performance[engine_type] = 0.5
            except:
                performance[engine_type] = 0.5
        
        return performance
    
    def _record_selection(self, selection_context: AdaptiveSelectionContext, 
                         selected_engines: List[ReasoningEngine], 
                         engine_scores: List[EngineSelectionScore]):
        """Record selection for learning"""
        selection_record = {
            "timestamp": datetime.now(timezone.utc),
            "query": selection_context.query,
            "problem_type": selection_context.problem_type,
            "selected_engines": selected_engines,
            "engine_scores": engine_scores
        }
        
        self.selection_history.append(selection_record)
        
        # Keep only recent history
        if len(self.selection_history) > self.performance_window:
            self.selection_history.pop(0)
    
    def update_performance_feedback(self, engine_type: ReasoningEngine, 
                                   performance_score: float, query: str):
        """Update performance feedback for learning"""
        if engine_type not in self.performance_history:
            self.performance_history[engine_type] = []
        
        self.performance_history[engine_type].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[engine_type]) > self.performance_window:
            self.performance_history[engine_type].pop(0)


class LoadBalancer:
    """Advanced load balancer for reasoning engines"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.strategy = LoadBalancingStrategy.PERFORMANCE_BASED
        self.mode = LoadBalancingMode.ACTIVE_ACTIVE
        self.enabled = True
        
        # Engine workload tracking
        self.engine_workloads: Dict[ReasoningEngine, EngineWorkload] = {}
        self.initialize_workloads()
        
        # Load balancing state
        self.round_robin_index = 0
        self.engine_weights: Dict[ReasoningEngine, float] = {}
        self.initialize_weights()
        
        # Metrics
        self.metrics = LoadBalancingMetrics()
        
        # Adaptive thresholds
        self.high_load_threshold = 0.8
        self.low_load_threshold = 0.2
        self.response_time_threshold = 5.0
        
        # Strategy adaptation
        self.strategy_adaptation_enabled = True
        self.strategy_evaluation_interval = 300  # 5 minutes
        self.last_strategy_evaluation = datetime.now(timezone.utc)
    
    def initialize_workloads(self):
        """Initialize workload tracking for all engines"""
        for engine_type in ReasoningEngine:
            self.engine_workloads[engine_type] = EngineWorkload(
                engine_id=f"{engine_type.value}_engine",
                current_requests=0,
                queue_depth=0,
                avg_response_time=0.0,
                success_rate=100.0
            )
    
    def initialize_weights(self):
        """Initialize engine weights based on baseline performance"""
        # Default equal weights
        for engine_type in ReasoningEngine:
            self.engine_weights[engine_type] = 1.0
    
    def select_engine(self, context: Dict[str, Any] = None) -> ReasoningEngine:
        """Select the best engine based on current load balancing strategy"""
        if not self.enabled:
            return self._fallback_selection()
        
        start_time = time.time()
        
        # Get available engines (healthy and not isolated)
        available_engines = self._get_available_engines()
        
        if not available_engines:
            return self._fallback_selection()
        
        # Select engine based on strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_engine = self._round_robin_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_engine = self._least_connections_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_engine = self._weighted_round_robin_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            selected_engine = self._performance_based_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_engine = self._least_response_time_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            selected_engine = self._random_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.HASH_BASED:
            selected_engine = self._hash_based_selection(available_engines, context)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            selected_engine = self._adaptive_selection(available_engines, context)
        else:
            selected_engine = self._fallback_selection()
        
        # Update workload
        if selected_engine in self.engine_workloads:
            workload = self.engine_workloads[selected_engine]
            workload.current_requests += 1
            workload.update_utilization()
        
        # Update metrics
        self.metrics.load_balancing_overhead += time.time() - start_time
        
        # Adaptive strategy evaluation
        if self.strategy_adaptation_enabled:
            self._evaluate_strategy_adaptation()
        
        return selected_engine
    
    def _get_available_engines(self) -> List[ReasoningEngine]:
        """Get list of available engines (healthy and not isolated)"""
        available_engines = []
        
        for engine_type in ReasoningEngine:
            # Check if engine is healthy
            try:
                if self.meta_reasoning_engine.health_monitor.should_use_engine(engine_type):
                    available_engines.append(engine_type)
            except:
                # If health monitor is not available, assume engine is available
                available_engines.append(engine_type)
        
        return available_engines if available_engines else list(ReasoningEngine)
    
    def _round_robin_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Round robin engine selection"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        selected_engine = available_engines[self.round_robin_index % len(available_engines)]
        self.round_robin_index += 1
        return selected_engine
    
    def _least_connections_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine with least active connections"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        min_load = float('inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            if engine_type in self.engine_workloads:
                workload = self.engine_workloads[engine_type]
                current_load = workload.current_requests + workload.queue_depth
                
                if current_load < min_load:
                    min_load = current_load
                    selected_engine = engine_type
        
        return selected_engine
    
    def _weighted_round_robin_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Weighted round robin selection based on engine weights"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        # Calculate cumulative weights
        cumulative_weights = []
        total_weight = 0
        
        for engine_type in available_engines:
            weight = self.engine_weights.get(engine_type, 1.0)
            total_weight += weight
            cumulative_weights.append(total_weight)
        
        # Select based on weight
        rand_value = random.random() * total_weight
        
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_value <= cum_weight:
                return available_engines[i]
        
        return available_engines[0]
    
    def _performance_based_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine based on performance metrics"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        best_score = float('-inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            try:
                profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
                if profile:
                    # Calculate performance score (lower response time and higher quality is better)
                    time_score = 1.0 / (1.0 + profile.avg_response_time)
                    quality_score = profile.avg_quality_score
                    reliability_score = profile.reliability_score / 100.0
                    
                    overall_score = time_score * 0.4 + quality_score * 0.4 + reliability_score * 0.2
                    
                    if overall_score > best_score:
                        best_score = overall_score
                        selected_engine = engine_type
            except:
                pass
        
        return selected_engine
    
    def _least_response_time_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine with least average response time"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        min_response_time = float('inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            avg_response_time = self.metrics.get_average_response_time(engine_type)
            if avg_response_time == 0.0:
                avg_response_time = 1.0  # Default for engines with no history
            
            if avg_response_time < min_response_time:
                min_response_time = avg_response_time
                selected_engine = engine_type
        
        return selected_engine
    
    def _random_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Random engine selection"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        return random.choice(available_engines)
    
    def _hash_based_selection(self, available_engines: List[ReasoningEngine], context: Dict[str, Any] = None) -> ReasoningEngine:
        """Hash-based selection for session affinity"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        # Use session ID or user ID for hashing if available
        hash_key = ""
        if context:
            hash_key = context.get("session_id") or context.get("user_id") or context.get("query", "")
        
        if not hash_key:
            return self._round_robin_selection(available_engines)
        
        # Simple hash-based selection
        hash_value = hash(hash_key) % len(available_engines)
        return available_engines[hash_value]
    
    def _adaptive_selection(self, available_engines: List[ReasoningEngine], context: Dict[str, Any] = None) -> ReasoningEngine:
        """Adaptive selection combining multiple factors"""
        if not available_engines:
            return ReasoningEngine.CLAUDE
        
        best_score = float('-inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            # Get various metrics
            if engine_type in self.engine_workloads:
                workload = self.engine_workloads[engine_type]
                load_score = 1.0 - workload.calculate_load_score()  # Lower load is better
            else:
                load_score = 1.0
            
            try:
                profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
                if profile:
                    performance_score = 1.0 / (1.0 + profile.avg_response_time)
                    quality_score = profile.avg_quality_score
                else:
                    performance_score = 0.5
                    quality_score = 0.5
            except:
                performance_score = 0.5
                quality_score = 0.5
            
            try:
                health_report = self.meta_reasoning_engine.health_monitor.get_engine_health_report(engine_type)
                health_score = health_report.get('health_score', 50.0) / 100.0
            except:
                health_score = 0.5
            
            # Weighted combination
            adaptive_score = (
                load_score * 0.3 +
                performance_score * 0.3 +
                quality_score * 0.2 +
                health_score * 0.2
            )
            
            if adaptive_score > best_score:
                best_score = adaptive_score
                selected_engine = engine_type
        
        return selected_engine
    
    def _fallback_selection(self) -> ReasoningEngine:
        """Fallback selection when load balancing is disabled or fails"""
        return ReasoningEngine.CLAUDE
    
    def complete_request(self, engine_type: ReasoningEngine, response_time: float, success: bool):
        """Mark a request as completed and update metrics"""
        if engine_type in self.engine_workloads:
            workload = self.engine_workloads[engine_type]
            workload.current_requests = max(0, workload.current_requests - 1)
            workload.avg_response_time = (workload.avg_response_time + response_time) / 2
            if success:
                workload.success_rate = min(100.0, workload.success_rate + 0.1)
            else:
                workload.success_rate = max(0.0, workload.success_rate - 1.0)
            workload.update_utilization()
        
        self.metrics.update_request_metrics(engine_type, response_time, success)
        
        # Update engine utilization
        if engine_type in self.engine_workloads:
            self.metrics.engine_utilization[engine_type] = self.engine_workloads[engine_type].utilization_percent
    
    def _evaluate_strategy_adaptation(self):
        """Evaluate and potentially adapt the load balancing strategy"""
        now = datetime.now(timezone.utc)
        
        if (now - self.last_strategy_evaluation).total_seconds() < self.strategy_evaluation_interval:
            return
        
        self.last_strategy_evaluation = now
        
        # Simple adaptation logic
        current_success_rate = self.metrics.get_balancing_success_rate()
        
        if current_success_rate < 0.8:
            # Switch to a more conservative strategy
            if self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
                self.strategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME
                self.metrics.strategy_switches += 1
                self.metrics.last_strategy_switch = now
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                self.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
                self.metrics.strategy_switches += 1
                self.metrics.last_strategy_switch = now
    
    def update_engine_weights(self):
        """Update engine weights based on current performance"""
        for engine_type in ReasoningEngine:
            try:
                profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
                health_report = self.meta_reasoning_engine.health_monitor.get_engine_health_report(engine_type)
                
                if profile and health_report:
                    # Calculate weight based on performance and health
                    performance_weight = 1.0 / (1.0 + profile.avg_response_time)
                    health_weight = health_report.get('health_score', 50.0) / 100.0
                    
                    self.engine_weights[engine_type] = (performance_weight + health_weight) / 2.0
                else:
                    self.engine_weights[engine_type] = 0.5
            except:
                self.engine_weights[engine_type] = 0.5
    
    def get_load_balancing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics"""
        return {
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "enabled": self.enabled,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "balanced_requests": self.metrics.balanced_requests,
                "success_rate": self.metrics.get_balancing_success_rate(),
                "failed_attempts": self.metrics.failed_balancing_attempts,
                "load_balancing_overhead": self.metrics.load_balancing_overhead,
                "strategy_switches": self.metrics.strategy_switches
            },
            "engine_workloads": {
                engine_type.value: {
                    "current_requests": workload.current_requests,
                    "queue_depth": workload.queue_depth,
                    "avg_response_time": workload.avg_response_time,
                    "success_rate": workload.success_rate,
                    "utilization_percent": workload.utilization_percent
                }
                for engine_type, workload in self.engine_workloads.items()
            },
            "engine_weights": {
                engine_type.value: weight
                for engine_type, weight in self.engine_weights.items()
            }
        }
    
    def reset_metrics(self):
        """Reset load balancing metrics"""
        self.metrics = LoadBalancingMetrics()
        self.initialize_workloads()
    
    def enable_load_balancing(self):
        """Enable load balancing"""
        self.enabled = True
    
    def disable_load_balancing(self):
        """Disable load balancing"""
        self.enabled = False
    
    def set_strategy(self, strategy: LoadBalancingStrategy):
        """Set load balancing strategy"""
        self.strategy = strategy
        if strategy != LoadBalancingStrategy.ADAPTIVE:
            self.strategy_adaptation_enabled = False
    
    def enable_strategy_adaptation(self):
        """Enable adaptive strategy switching"""
        self.strategy_adaptation_enabled = True
    
    def disable_strategy_adaptation(self):
        """Disable adaptive strategy switching"""
        self.strategy_adaptation_enabled = False


# Export classes for use in other modules
__all__ = [
    'LoadBalancingMetrics',
    'AdaptiveSelectionStrategy',
    'ProblemType',
    'ContextualFactor',
    'AdaptiveSelectionContext',
    'EngineSelectionScore',
    'AdaptiveEngineSelector',
    'LoadBalancer'
]