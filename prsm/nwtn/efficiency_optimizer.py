#!/usr/bin/env python3
"""
NWTN Efficiency Optimizer
Addresses the "Scale vs. Efficiency" problem identified in Stochastic Parrots

This module implements efficiency optimization strategies to maximize 
reasoning quality per compute unit, avoiding the "bigger is better" trap
that leads to massive resource consumption with diminishing returns.

Key principles from Stochastic Parrots analysis:
1. Maximize insight per compute unit, not just scale
2. Use targeted reasoning instead of brute force
3. Leverage knowledge sharing to avoid redundant computation
4. Optimize System 1/System 2 balance based on uncertainty
5. Implement efficient knowledge caching and reuse

Usage:
    from prsm.nwtn.efficiency_optimizer import EfficiencyOptimizer
    
    optimizer = EfficiencyOptimizer()
    optimized_path = await optimizer.optimize_reasoning_path(query, context)
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, Field

from prsm.nwtn.hybrid_architecture import (
    HybridNWTNEngine, SOC, UserGoal, CommunicativeIntent, ConfidenceLevel
)

logger = structlog.get_logger(__name__)


class OptimizationStrategy(str, Enum):
    """Strategies for optimizing reasoning efficiency"""
    CACHED_KNOWLEDGE = "cached_knowledge"
    MINIMAL_SYSTEM1 = "minimal_system1"
    TARGETED_SYSTEM2 = "targeted_system2"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    UNCERTAINTY_DRIVEN = "uncertainty_driven"
    PROGRESSIVE_DEPTH = "progressive_depth"


class EfficiencyMetrics(BaseModel):
    """Metrics for tracking efficiency optimization"""
    
    query_id: str
    total_compute_time: float
    system1_time: float
    system2_time: float
    
    # Quality metrics
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning_quality: float = Field(ge=0.0, le=1.0)
    intent_alignment: float = Field(ge=0.0, le=1.0)
    
    # Efficiency metrics
    compute_per_insight: float  # Lower is better
    knowledge_reuse_rate: float = Field(ge=0.0, le=1.0)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    
    # Resource usage
    system1_calls: int = Field(default=0)
    system2_calls: int = Field(default=0)
    cache_lookups: int = Field(default=0)
    
    # Optimization strategy used
    strategy: OptimizationStrategy
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class KnowledgeCache:
    """Cached knowledge to avoid redundant computation"""
    
    query_pattern: str
    domain: str
    socs: List[SOC]
    confidence: float
    reasoning_trace: List[Dict[str, Any]]
    
    # Metadata
    usage_count: int = 0
    last_accessed: datetime = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def update_access(self):
        """Update access metadata"""
        self.usage_count += 1
        self.last_accessed = datetime.now(timezone.utc)


class EfficiencyOptimizer:
    """
    Optimizer for maximizing reasoning quality per compute unit
    
    Implements strategies to avoid the "stochastic parrot" problem of
    massive scale with diminishing returns by focusing on targeted,
    efficient reasoning.
    """
    
    def __init__(self):
        self.knowledge_cache: Dict[str, KnowledgeCache] = {}
        self.efficiency_history: List[EfficiencyMetrics] = []
        
        # Optimization thresholds
        self.cache_threshold = 0.8  # Confidence threshold for caching
        self.system2_threshold = 0.6  # Uncertainty threshold for System 2
        self.max_cache_size = 1000
        
        # Performance tracking
        self.total_queries = 0
        self.cache_hits = 0
        self.system2_avoided = 0
        
        logger.info("Initialized NWTN Efficiency Optimizer")
    
    async def optimize_reasoning_path(
        self, 
        query: str, 
        context: Dict[str, Any] = None,
        user_goal: UserGoal = None
    ) -> Dict[str, Any]:
        """
        Optimize reasoning path for maximum efficiency
        
        Flow:
        1. Check knowledge cache for similar queries
        2. If cache hit with high confidence, return cached result
        3. If cache miss, determine optimal System 1/System 2 balance
        4. Execute minimal necessary computation
        5. Cache high-quality results for future use
        """
        
        start_time = time.time()
        query_id = str(uuid4())
        
        logger.info("Optimizing reasoning path", query_id=query_id, query=query[:50])
        
        # Step 1: Check knowledge cache
        cache_result = await self._check_knowledge_cache(query, context)
        if cache_result:
            logger.info("Cache hit - returning cached knowledge", query_id=query_id)
            return await self._format_cached_result(cache_result, query_id, start_time)
        
        # Step 2: Determine optimal strategy
        strategy = await self._determine_optimization_strategy(query, context, user_goal)
        
        # Step 3: Execute optimized reasoning
        result = await self._execute_optimized_reasoning(query, context, user_goal, strategy)
        
        # Step 4: Cache high-quality results
        await self._cache_result_if_valuable(query, context, result)
        
        # Step 5: Track efficiency metrics
        await self._track_efficiency_metrics(query_id, start_time, result, strategy)
        
        return result
    
    async def _check_knowledge_cache(self, query: str, context: Dict[str, Any]) -> Optional[KnowledgeCache]:
        """Check if we have cached knowledge for similar queries"""
        
        # Simple similarity check (in full implementation, use semantic similarity)
        query_lower = query.lower()
        
        for cache_key, cached_knowledge in self.knowledge_cache.items():
            # Check for pattern match
            if cached_knowledge.query_pattern.lower() in query_lower:
                # Check if confidence is high enough
                if cached_knowledge.confidence >= self.cache_threshold:
                    cached_knowledge.update_access()
                    self.cache_hits += 1
                    return cached_knowledge
        
        return None
    
    async def _determine_optimization_strategy(
        self, 
        query: str, 
        context: Dict[str, Any], 
        user_goal: UserGoal
    ) -> OptimizationStrategy:
        """Determine the most efficient strategy for this query"""
        
        # Analyze query complexity
        complexity_indicators = [
            "analyze", "compare", "complex", "multiple", "relationship",
            "why", "how", "explain", "derive", "prove"
        ]
        
        query_lower = query.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Analyze domain
        domain_indicators = {
            "physics": ["physics", "force", "energy", "momentum", "wave"],
            "chemistry": ["reaction", "molecule", "chemical", "bond", "catalyst"],
            "math": ["equation", "derivative", "integral", "function", "theorem"],
            "general": []
        }
        
        domain = "general"
        for domain_name, indicators in domain_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                domain = domain_name
                break
        
        # Determine strategy based on complexity and domain
        if complexity_score == 0:
            return OptimizationStrategy.MINIMAL_SYSTEM1
        elif complexity_score <= 2:
            return OptimizationStrategy.UNCERTAINTY_DRIVEN
        elif domain != "general":
            return OptimizationStrategy.TARGETED_SYSTEM2
        else:
            return OptimizationStrategy.PROGRESSIVE_DEPTH
    
    async def _execute_optimized_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        user_goal: UserGoal,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Execute reasoning using the optimal strategy"""
        
        if strategy == OptimizationStrategy.MINIMAL_SYSTEM1:
            return await self._minimal_system1_reasoning(query, context, user_goal)
        elif strategy == OptimizationStrategy.UNCERTAINTY_DRIVEN:
            return await self._uncertainty_driven_reasoning(query, context, user_goal)
        elif strategy == OptimizationStrategy.TARGETED_SYSTEM2:
            return await self._targeted_system2_reasoning(query, context, user_goal)
        elif strategy == OptimizationStrategy.PROGRESSIVE_DEPTH:
            return await self._progressive_depth_reasoning(query, context, user_goal)
        else:
            # Default to full hybrid processing
            engine = HybridNWTNEngine()
            return await engine.process_query(query, context)
    
    async def _minimal_system1_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        user_goal: UserGoal
    ) -> Dict[str, Any]:
        """Use minimal System 1 processing for simple queries"""
        
        start_time = time.time()
        
        # Simple pattern matching for straightforward queries
        engine = HybridNWTNEngine()
        
        # Skip expensive System 2 validation for simple queries
        socs = await engine._system1_soc_recognition(query, context)
        
        # Generate response directly from System 1
        response = {
            "response": f"Based on pattern recognition: {query}",
            "reasoning_trace": [
                {
                    "step": "minimal_system1",
                    "description": "Used fast pattern recognition only",
                    "compute_time": time.time() - start_time
                }
            ],
            "socs_used": [{"name": soc.name, "confidence": soc.confidence} for soc in socs],
            "optimization_strategy": "minimal_system1",
            "system2_avoided": True
        }
        
        self.system2_avoided += 1
        return response
    
    async def _uncertainty_driven_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        user_goal: UserGoal
    ) -> Dict[str, Any]:
        """Use System 2 only for high-uncertainty areas"""
        
        start_time = time.time()
        
        engine = HybridNWTNEngine()
        
        # Step 1: System 1 recognition
        socs = await engine._system1_soc_recognition(query, context)
        
        # Step 2: Identify high-uncertainty SOCs
        high_uncertainty_socs = [soc for soc in socs if soc.confidence < self.system2_threshold]
        
        # Step 3: Apply System 2 only to uncertain areas
        if high_uncertainty_socs:
            validated_socs = await engine._system2_validation(high_uncertainty_socs)
            
            # Combine validated uncertain SOCs with confident System 1 SOCs
            final_socs = []
            validated_names = {soc.name for soc in validated_socs}
            
            for soc in socs:
                if soc.name in validated_names:
                    # Use validated version
                    final_socs.extend([s for s in validated_socs if s.name == soc.name])
                else:
                    # Use original System 1 version
                    final_socs.append(soc)
        else:
            final_socs = socs
        
        response = {
            "response": f"Uncertainty-driven analysis: {query}",
            "reasoning_trace": [
                {
                    "step": "uncertainty_driven",
                    "description": f"Applied System 2 to {len(high_uncertainty_socs)} uncertain SOCs",
                    "system1_socs": len(socs),
                    "system2_socs": len(high_uncertainty_socs),
                    "compute_time": time.time() - start_time
                }
            ],
            "socs_used": [{"name": soc.name, "confidence": soc.confidence} for soc in final_socs],
            "optimization_strategy": "uncertainty_driven",
            "system2_selective": True
        }
        
        return response
    
    async def _targeted_system2_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        user_goal: UserGoal
    ) -> Dict[str, Any]:
        """Use targeted System 2 reasoning for domain-specific queries"""
        
        # Full hybrid processing but with domain-specific optimization
        engine = HybridNWTNEngine()
        result = await engine.process_query(query, context)
        
        # Add optimization metadata
        result["optimization_strategy"] = "targeted_system2"
        result["domain_optimized"] = True
        
        return result
    
    async def _progressive_depth_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        user_goal: UserGoal
    ) -> Dict[str, Any]:
        """Use progressive depth - start shallow, go deeper if needed"""
        
        start_time = time.time()
        
        # Step 1: Shallow analysis
        shallow_result = await self._minimal_system1_reasoning(query, context, user_goal)
        
        # Step 2: Check if shallow analysis is sufficient
        if user_goal and user_goal.depth_required == "shallow":
            return shallow_result
        
        # Step 3: If deeper analysis needed, use full hybrid
        engine = HybridNWTNEngine()
        deep_result = await engine.process_query(query, context)
        
        # Combine results
        deep_result["optimization_strategy"] = "progressive_depth"
        deep_result["shallow_analysis"] = shallow_result
        deep_result["progressive_reasoning"] = True
        
        return deep_result
    
    async def _cache_result_if_valuable(self, query: str, context: Dict[str, Any], result: Dict[str, Any]):
        """Cache high-quality results for future use"""
        
        # Only cache if result has high confidence
        if "socs_used" in result:
            avg_confidence = sum(soc.get("confidence", 0) for soc in result["socs_used"]) / len(result["socs_used"])
            
            if avg_confidence >= self.cache_threshold:
                # Create cache entry
                cache_key = f"{query[:50]}_{hash(str(context))}"
                
                cached_knowledge = KnowledgeCache(
                    query_pattern=query,
                    domain=context.get("domain", "general") if context else "general",
                    socs=[],  # Simplified for now
                    confidence=avg_confidence,
                    reasoning_trace=result.get("reasoning_trace", [])
                )
                
                self.knowledge_cache[cache_key] = cached_knowledge
                
                # Maintain cache size
                if len(self.knowledge_cache) > self.max_cache_size:
                    # Remove least recently used
                    oldest_key = min(
                        self.knowledge_cache.keys(),
                        key=lambda k: self.knowledge_cache[k].last_accessed or datetime.min
                    )
                    del self.knowledge_cache[oldest_key]
                
                logger.info("Cached high-quality result", cache_key=cache_key, confidence=avg_confidence)
    
    async def _format_cached_result(self, cached_knowledge: KnowledgeCache, query_id: str, start_time: float) -> Dict[str, Any]:
        """Format cached knowledge as a result"""
        
        return {
            "response": f"From cached knowledge: {cached_knowledge.query_pattern}",
            "reasoning_trace": cached_knowledge.reasoning_trace + [
                {
                    "step": "cache_retrieval",
                    "description": "Retrieved from knowledge cache",
                    "cache_usage_count": cached_knowledge.usage_count,
                    "compute_time": time.time() - start_time
                }
            ],
            "socs_used": [{"name": soc.name, "confidence": soc.confidence} for soc in cached_knowledge.socs],
            "optimization_strategy": "cached_knowledge",
            "cache_hit": True,
            "query_id": query_id
        }
    
    async def _track_efficiency_metrics(
        self,
        query_id: str,
        start_time: float,
        result: Dict[str, Any],
        strategy: OptimizationStrategy
    ):
        """Track efficiency metrics for optimization"""
        
        total_time = time.time() - start_time
        
        # Calculate efficiency metrics
        metrics = EfficiencyMetrics(
            query_id=query_id,
            total_compute_time=total_time,
            system1_time=total_time * 0.3,  # Estimated
            system2_time=total_time * 0.7,  # Estimated
            confidence_score=0.8,  # Default
            reasoning_quality=0.8,  # Default
            intent_alignment=0.8,  # Default
            compute_per_insight=total_time,  # Simplified
            knowledge_reuse_rate=self.cache_hits / max(1, self.total_queries),
            cache_hit_rate=self.cache_hits / max(1, self.total_queries),
            system1_calls=1,
            system2_calls=1 if not result.get("system2_avoided", False) else 0,
            cache_lookups=1,
            strategy=strategy
        )
        
        self.efficiency_history.append(metrics)
        self.total_queries += 1
        
        logger.info(
            "Tracked efficiency metrics",
            query_id=query_id,
            compute_time=total_time,
            strategy=strategy.value,
            cache_hit_rate=metrics.cache_hit_rate
        )
    
    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get efficiency optimization statistics"""
        
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_queries),
            "system2_avoided": self.system2_avoided,
            "system2_avoidance_rate": self.system2_avoided / max(1, self.total_queries),
            "average_compute_time": sum(m.total_compute_time for m in self.efficiency_history) / max(1, len(self.efficiency_history)),
            "knowledge_cache_size": len(self.knowledge_cache),
            "strategies_used": list(set(m.strategy for m in self.efficiency_history))
        }
    
    async def optimize_system_balance(self, uncertainty_threshold: float = None):
        """Optimize the System 1/System 2 balance based on efficiency data"""
        
        if uncertainty_threshold:
            self.system2_threshold = uncertainty_threshold
        
        # Analyze efficiency history to optimize thresholds
        if len(self.efficiency_history) > 10:
            # Calculate optimal thresholds based on performance
            high_quality_queries = [m for m in self.efficiency_history if m.reasoning_quality > 0.8]
            
            if high_quality_queries:
                # Find the threshold that maximizes quality while minimizing compute
                optimal_threshold = sum(m.confidence_score for m in high_quality_queries) / len(high_quality_queries)
                self.system2_threshold = max(0.5, min(0.8, optimal_threshold))
        
        logger.info("Optimized system balance", system2_threshold=self.system2_threshold)