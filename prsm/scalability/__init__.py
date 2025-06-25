"""
PRSM Scalability Enhancement Package

Comprehensive scalability improvements addressing the 300-user breaking point
identified in Phase 3 testing. Includes intelligent routing, CPU optimization,
auto-scaling, and advanced caching.

Expected improvements:
- Breaking point: 300 → 500+ users (67% improvement)
- Throughput: 6,984 → 10,000+ ops/sec (44% improvement)
- CPU usage: Reduce 15-30% across bottleneck components
- Latency: 20-40% reduction through intelligent caching
"""

from .intelligent_router import IntelligentRouter, ComponentMetrics, RoutingRule
from .cpu_optimizer import CPUOptimizer, CPUOptimizationConfig
from .auto_scaler import AutoScaler, ScalingMetrics, ScalingPolicy, LoadBalancingPolicy
from .advanced_cache import AdvancedCache, CacheManager, CacheLevel, CacheItem
from .scalability_orchestrator import ScalabilityOrchestrator

__all__ = [
    "IntelligentRouter",
    "ComponentMetrics", 
    "RoutingRule",
    "CPUOptimizer",
    "CPUOptimizationConfig",
    "AutoScaler",
    "ScalingMetrics",
    "ScalingPolicy", 
    "LoadBalancingPolicy",
    "AdvancedCache",
    "CacheManager",
    "CacheLevel",
    "CacheItem",
    "ScalabilityOrchestrator"
]