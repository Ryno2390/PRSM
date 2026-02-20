"""
PRSM Performance Module

This module provides performance instrumentation, optimization, and benchmarking
capabilities for the PRSM system.

Components:
- instrumentation: Timing decorators, memory profiling, metric collection
- optimization: Caching strategies, batch processing optimization  
- benchmark_orchestrator: Test orchestration, result aggregation, reporting
- benchmark_collector: Global performance metric collection
"""

from prsm.performance.instrumentation import (
    timing_decorator,
    TimingContext,
    MemoryProfiler,
    PerformanceTimer,
    measure_performance,
    get_timer,
)
from prsm.performance.optimization import (
    CacheManager,
    CacheConfig,
    CacheLevel,
    PerformanceOptimizer,
    QueryOptimizer,
    APIOptimizer,
    BatchProcessor,
)
from prsm.performance.benchmark_collector import (
    BenchmarkCollector,
    MetricStats,
    get_global_collector,
    reset_global_collector,
)
from prsm.performance.benchmark_orchestrator import (
    BenchmarkOrchestrator,
    BenchmarkType,
    LoadProfile,
    BenchmarkScenario,
    BenchmarkResult,
)

__all__ = [
    # Instrumentation
    'timing_decorator',
    'TimingContext',
    'MemoryProfiler',
    'PerformanceTimer',
    'measure_performance',
    'get_timer',
    # Optimization
    'CacheManager',
    'CacheConfig',
    'CacheLevel',
    'PerformanceOptimizer',
    'QueryOptimizer',
    'APIOptimizer',
    'BatchProcessor',
    # Benchmark Collector
    'BenchmarkCollector',
    'MetricStats',
    'get_global_collector',
    'reset_global_collector',
    # Benchmark Orchestrator
    'BenchmarkOrchestrator',
    'BenchmarkType',
    'LoadProfile',
    'BenchmarkScenario',
    'BenchmarkResult',
]
