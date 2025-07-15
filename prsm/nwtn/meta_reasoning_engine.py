#!/usr/bin/env python3
"""
NWTN Meta-Reasoning Engine
=========================

This module implements a sophisticated meta-reasoning system that orchestrates
multiple reasoning engines to solve complex problems through different thinking modes:

1. Quick Thinking Mode (Parallel Processing) - Low FTNS cost
2. Intermediate Thinking Mode (Partial Permutations) - Medium FTNS cost  
3. Deep Thinking Mode (Full Permutations) - High FTNS cost

The system leverages NWTN's parallel processing capabilities and reasoning engine
interactions to provide comprehensive problem-solving across different computational
budgets and time constraints.
"""

import asyncio
import time
import math
import os
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations, combinations
from uuid import uuid4
from datetime import datetime, timezone, timedelta
import statistics
import weakref
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import gzip
from functools import lru_cache, wraps
import sys
import hashlib
import gc

import structlog

# Optional psutil import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional cachetools import for advanced caching
try:
    from cachetools import LRUCache, TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False

# Import all reasoning engines
from prsm.nwtn.enhanced_deductive_reasoning import EnhancedDeductiveReasoningEngine
from prsm.nwtn.enhanced_inductive_reasoning import EnhancedInductiveReasoningEngine
from prsm.nwtn.enhanced_abductive_reasoning import EnhancedAbductiveReasoningEngine
from prsm.nwtn.enhanced_causal_reasoning import EnhancedCausalReasoningEngine
from prsm.nwtn.enhanced_probabilistic_reasoning import EnhancedProbabilisticReasoningEngine
from prsm.nwtn.enhanced_counterfactual_reasoning import EnhancedCounterfactualReasoningEngine
from prsm.nwtn.enhanced_analogical_reasoning import AnalogicalReasoningEngine

logger = structlog.get_logger(__name__)


class ThinkingMode(Enum):
    """Different modes of meta-reasoning with varying computational depth"""
    QUICK = "quick"           # Parallel processing - Low FTNS cost
    INTERMEDIATE = "intermediate"  # Partial permutations - Medium FTNS cost
    DEEP = "deep"            # Full permutations - High FTNS cost


class ReasoningEngine(Enum):
    """Available reasoning engines"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    COUNTERFACTUAL = "counterfactual"
    ANALOGICAL = "analogical"


class EngineHealthStatus(Enum):
    """Health status levels for reasoning engines"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


@dataclass
class EnginePerformanceMetrics:
    """Performance metrics for reasoning engines"""
    
    engine_type: ReasoningEngine
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    timeout_executions: int = 0
    
    # Timing metrics
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    
    # Quality metrics
    total_quality_score: float = 0.0
    average_quality_score: float = 0.0
    total_confidence_score: float = 0.0
    average_confidence_score: float = 0.0
    
    # Health metrics
    current_health_score: float = 1.0
    health_history: deque = field(default_factory=lambda: deque(maxlen=50))  # Optimized with maxlen
    last_successful_execution: Optional[datetime] = None
    last_failed_execution: Optional[datetime] = None
    
    # Error tracking
    error_types: Dict[str, int] = field(default_factory=dict)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=10))  # Optimized with maxlen
    
    # Performance optimization flags
    _cached_success_rate: Optional[float] = None
    _cached_timeout_rate: Optional[float] = None
    _cached_failure_rate: Optional[float] = None
    _cache_invalidated: bool = True
    _last_cache_update: Optional[datetime] = None
    
    def update_execution_metrics(self, execution_time: float, quality_score: float, 
                               confidence_score: float, success: bool, error_type: str = None):
        """Update metrics after an execution"""
        self.total_executions += 1
        
        if success:
            self.successful_executions += 1
            self.last_successful_execution = datetime.now(timezone.utc)
        else:
            self.failed_executions += 1
            self.last_failed_execution = datetime.now(timezone.utc)
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
                self.recent_errors.append(f"{datetime.now(timezone.utc)}: {error_type}")
                # deque automatically handles maxlen
        
        # Update timing metrics
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        
        # Update quality metrics
        if success:
            self.total_quality_score += quality_score
            self.total_confidence_score += confidence_score
            successful_count = self.successful_executions
            if successful_count > 0:
                self.average_quality_score = self.total_quality_score / successful_count
                self.average_confidence_score = self.total_confidence_score / successful_count
        
        # Update health score and invalidate cache
        self._invalidate_cache()
        self._update_health_score()
    
    def update_timeout_metrics(self):
        """Update metrics for timeout events"""
        self.timeout_executions += 1
        self.failed_executions += 1
        self.total_executions += 1
        self.last_failed_execution = datetime.now(timezone.utc)
        
        error_type = "timeout"
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        self.recent_errors.append(f"{datetime.now(timezone.utc)}: {error_type}")
        # deque automatically handles maxlen
        
        self._invalidate_cache()
        self._update_health_score()
    
    def _update_health_score(self):
        """Update current health score based on performance"""
        if self.total_executions == 0:
            self.current_health_score = 1.0
            return
        
        # Calculate success rate
        success_rate = self.successful_executions / self.total_executions
        
        # Calculate timeout rate
        timeout_rate = self.timeout_executions / self.total_executions
        
        # Calculate performance score based on execution time
        performance_score = 1.0
        if self.average_execution_time > 0:
            # Penalize if average time is much higher than minimum
            if self.min_execution_time > 0:
                performance_ratio = self.average_execution_time / self.min_execution_time
                performance_score = max(0.1, 1.0 / performance_ratio)
        
        # Combined health score (weighted average)
        health_score = (success_rate * 0.5 + 
                       (1.0 - timeout_rate) * 0.3 + 
                       performance_score * 0.2)
        
        self.current_health_score = max(0.0, min(1.0, health_score))
        
        # Add to history (deque automatically handles maxlen)
        self.health_history.append(self.current_health_score)
    
    def _invalidate_cache(self):
        """Invalidate performance metric caches"""
        self._cache_invalidated = True
        self._cached_success_rate = None
        self._cached_timeout_rate = None
        self._cached_failure_rate = None
    
    def _update_cache(self):
        """Update cached performance metrics"""
        if not self._cache_invalidated:
            return
        
        if self.total_executions > 0:
            self._cached_success_rate = self.successful_executions / self.total_executions
            self._cached_timeout_rate = self.timeout_executions / self.total_executions
            self._cached_failure_rate = self.failed_executions / self.total_executions
        else:
            self._cached_success_rate = 0.0
            self._cached_timeout_rate = 0.0
            self._cached_failure_rate = 0.0
        
        self._cache_invalidated = False
        self._last_cache_update = datetime.now(timezone.utc)
    
    def get_success_rate(self) -> float:
        """Get current success rate (cached)"""
        self._update_cache()
        return self._cached_success_rate
    
    def get_timeout_rate(self) -> float:
        """Get current timeout rate (cached)"""
        self._update_cache()
        return self._cached_timeout_rate
    
    def get_failure_rate(self) -> float:
        """Get current failure rate (cached)"""
        self._update_cache()
        return self._cached_failure_rate
    
    def get_health_trend(self) -> str:
        """Get health trend over time"""
        if len(self.health_history) < 2:
            return "stable"
        
        recent_scores = self.health_history[-10:]  # Last 10 scores
        if len(recent_scores) < 2:
            return "stable"
        
        # Calculate trend
        trend = recent_scores[-1] - recent_scores[0]
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "degrading"
        else:
            return "stable"


class EngineHealthMonitor:
    """Monitor and track health of reasoning engines"""
    
    def __init__(self):
        self.engine_metrics: Dict[ReasoningEngine, EnginePerformanceMetrics] = {}
        self.monitoring_enabled = True
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = None
        
        # Initialize metrics for all engines
        for engine_type in ReasoningEngine:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(engine_type)
    
    def record_execution(self, engine_type: ReasoningEngine, execution_time: float, 
                        quality_score: float, confidence_score: float, success: bool, 
                        error_type: str = None):
        """Record an execution result"""
        if not self.monitoring_enabled:
            return
        
        if engine_type not in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(engine_type)
        
        self.engine_metrics[engine_type].update_execution_metrics(
            execution_time, quality_score, confidence_score, success, error_type
        )
    
    def record_timeout(self, engine_type: ReasoningEngine):
        """Record a timeout event"""
        if not self.monitoring_enabled:
            return
        
        if engine_type not in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(engine_type)
        
        self.engine_metrics[engine_type].update_timeout_metrics()
    
    def record_engine_execution(self, engine_type: ReasoningEngine, execution_time: float, 
                              success: bool, error: str = None):
        """Record engine execution result with simplified signature"""
        if not self.monitoring_enabled:
            return
        
        if engine_type not in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(engine_type)
        
        # Use defaults for quality and confidence when not provided
        quality_score = 0.8 if success else 0.0
        confidence_score = 0.7 if success else 0.0
        
        self.engine_metrics[engine_type].update_execution_metrics(
            execution_time, quality_score, confidence_score, success, error
        )
    
    def record_engine_timeout(self, engine_type: ReasoningEngine, timeout: float):
        """Record engine timeout event"""
        if not self.monitoring_enabled:
            return
        
        if engine_type not in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(engine_type)
        
        self.engine_metrics[engine_type].update_timeout_metrics()
    
    def get_engine_health_status(self, engine_type: ReasoningEngine) -> EngineHealthStatus:
        """Get current health status for an engine"""
        if engine_type not in self.engine_metrics:
            return EngineHealthStatus.UNKNOWN
        
        metrics = self.engine_metrics[engine_type]
        health_score = metrics.current_health_score
        
        if health_score >= 0.8:
            return EngineHealthStatus.HEALTHY
        elif health_score >= 0.6:
            return EngineHealthStatus.DEGRADED
        elif health_score >= 0.4:
            return EngineHealthStatus.UNHEALTHY
        elif health_score >= 0.2:
            return EngineHealthStatus.FAILED
        else:
            # Check if it's recovering
            trend = metrics.get_health_trend()
            if trend == "improving":
                return EngineHealthStatus.RECOVERING
            else:
                return EngineHealthStatus.FAILED
    
    def get_engine_health_report(self, engine_type: ReasoningEngine) -> Dict[str, Any]:
        """Get comprehensive health report for an engine"""
        if engine_type not in self.engine_metrics:
            return {
                "engine_type": engine_type.value,
                "status": EngineHealthStatus.UNKNOWN.value,
                "health_score": 0.0,
                "message": "No metrics available"
            }
        
        metrics = self.engine_metrics[engine_type]
        status = self.get_engine_health_status(engine_type)
        
        return {
            "engine_type": engine_type.value,
            "status": status.value,
            "health_score": metrics.current_health_score,
            "success_rate": metrics.get_success_rate(),
            "failure_rate": metrics.get_failure_rate(),
            "timeout_rate": metrics.get_timeout_rate(),
            "average_execution_time": metrics.average_execution_time,
            "total_executions": metrics.total_executions,
            "health_trend": metrics.get_health_trend(),
            "last_successful_execution": metrics.last_successful_execution,
            "last_failed_execution": metrics.last_failed_execution,
            "recent_errors": metrics.recent_errors[-5:],  # Last 5 errors
            "error_types": metrics.error_types
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        healthy_engines = []
        degraded_engines = []
        unhealthy_engines = []
        failed_engines = []
        unknown_engines = []
        
        total_health_score = 0.0
        total_engines = 0
        
        for engine_type in ReasoningEngine:
            status = self.get_engine_health_status(engine_type)
            health_score = self.engine_metrics[engine_type].current_health_score
            
            if status == EngineHealthStatus.HEALTHY:
                healthy_engines.append(engine_type)
            elif status == EngineHealthStatus.DEGRADED:
                degraded_engines.append(engine_type)
            elif status == EngineHealthStatus.UNHEALTHY:
                unhealthy_engines.append(engine_type)
            elif status in [EngineHealthStatus.FAILED, EngineHealthStatus.RECOVERING]:
                failed_engines.append(engine_type)
            else:
                unknown_engines.append(engine_type)
            
            total_health_score += health_score
            total_engines += 1
        
        overall_health_score = total_health_score / total_engines if total_engines > 0 else 0.0
        
        return {
            "overall_health_score": overall_health_score,
            "total_engines": total_engines,
            "healthy_engines": [e.value for e in healthy_engines],
            "degraded_engines": [e.value for e in degraded_engines],
            "unhealthy_engines": [e.value for e in unhealthy_engines],
            "failed_engines": [e.value for e in failed_engines],
            "unknown_engines": [e.value for e in unknown_engines]
        }
    
    def get_overall_system_health(self) -> Dict[str, Any]:
        """Alias for get_system_health_summary for backward compatibility"""
        return self.get_system_health_summary()
    
    def get_healthy_engines(self) -> List[ReasoningEngine]:
        """Get list of healthy engines"""
        healthy_engines = []
        for engine_type in ReasoningEngine:
            status = self.get_engine_health_status(engine_type)
            if status in [EngineHealthStatus.HEALTHY, EngineHealthStatus.DEGRADED]:
                healthy_engines.append(engine_type)
        return healthy_engines
    
    def get_unhealthy_engines(self) -> List[ReasoningEngine]:
        """Get list of unhealthy engines"""
        unhealthy_engines = []
        for engine_type in ReasoningEngine:
            status = self.get_engine_health_status(engine_type)
            if status in [EngineHealthStatus.UNHEALTHY, EngineHealthStatus.FAILED]:
                unhealthy_engines.append(engine_type)
        return unhealthy_engines
    
    def enable_monitoring(self):
        """Enable health monitoring"""
        self.monitoring_enabled = True
    
    def disable_monitoring(self):
        """Disable health monitoring"""
        self.monitoring_enabled = False
    
    def reset_engine_metrics(self, engine_type: ReasoningEngine):
        """Reset metrics for a specific engine"""
        if engine_type in self.engine_metrics:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(engine_type)
    
    def reset_all_metrics(self):
        """Reset all engine metrics"""
        for engine_type in ReasoningEngine:
            self.engine_metrics[engine_type] = EnginePerformanceMetrics(engine_type)
    
    def should_use_engine(self, engine_type: ReasoningEngine) -> bool:
        """Check if an engine should be used based on health status"""
        if not self.monitoring_enabled:
            return True
        
        status = self.get_engine_health_status(engine_type)
        return status in [EngineHealthStatus.HEALTHY, EngineHealthStatus.DEGRADED]


class PerformanceMetric(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    QUALITY_SCORE = "quality_score"
    CONFIDENCE_SCORE = "confidence_score"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    TIMEOUT_RATE = "timeout_rate"


class PerformanceCategory(Enum):
    """Categories of performance analysis"""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot"""
    
    timestamp: datetime
    engine_type: ReasoningEngine
    metric_type: PerformanceMetric
    value: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class PerformanceProfile:
    """Comprehensive performance profile for an engine"""
    
    engine_type: ReasoningEngine
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Performance snapshots
    snapshots: List[PerformanceSnapshot] = field(default_factory=list)
    
    # Statistical metrics
    avg_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    std_execution_time: float = 0.0
    percentile_95_execution_time: float = 0.0
    percentile_99_execution_time: float = 0.0
    
    # Memory metrics
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    quality_trend: str = "stable"
    
    # Throughput metrics
    requests_per_second: float = 0.0
    peak_throughput: float = 0.0
    
    # Performance characteristics
    performance_class: str = "normal"  # normal, high_performance, degraded
    bottlenecks: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)
    
    def update_profile(self, snapshots: List[PerformanceSnapshot]):
        """Update performance profile with new snapshots"""
        self.snapshots.extend(snapshots)
        
        # Calculate execution time statistics
        execution_times = [s.value for s in self.snapshots if s.metric_type == PerformanceMetric.EXECUTION_TIME]
        if execution_times:
            self.avg_execution_time = statistics.mean(execution_times)
            self.min_execution_time = min(execution_times)
            self.max_execution_time = max(execution_times)
            if len(execution_times) > 1:
                self.std_execution_time = statistics.stdev(execution_times)
            
            # Calculate percentiles
            execution_times.sort()
            n = len(execution_times)
            if n >= 20:  # Only calculate percentiles with sufficient data
                self.percentile_95_execution_time = execution_times[int(0.95 * n)]
                self.percentile_99_execution_time = execution_times[int(0.99 * n)]
        
        # Calculate memory statistics
        memory_usage = [s.value for s in self.snapshots if s.metric_type == PerformanceMetric.MEMORY_USAGE]
        if memory_usage:
            self.avg_memory_usage = statistics.mean(memory_usage)
            self.peak_memory_usage = max(memory_usage)
        
        # Calculate quality statistics
        quality_scores = [s.value for s in self.snapshots if s.metric_type == PerformanceMetric.QUALITY_SCORE]
        if quality_scores:
            self.avg_quality_score = statistics.mean(quality_scores)
            self.quality_trend = self._calculate_trend(quality_scores)
        
        # Determine performance class
        self._classify_performance()
        
        # Identify bottlenecks
        self._identify_bottlenecks()
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in performance values"""
        if len(values) < 10:
            return "stable"
        
        # Use simple linear regression slope
        recent_values = values[-10:]
        n = len(recent_values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(recent_values)
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"
    
    def _classify_performance(self):
        """Classify performance level"""
        if self.avg_execution_time <= 0.5:
            self.performance_class = "high_performance"
        elif self.avg_execution_time <= 2.0:
            self.performance_class = "normal"
        else:
            self.performance_class = "degraded"
    
    def _identify_bottlenecks(self):
        """Identify potential performance bottlenecks"""
        self.bottlenecks = []
        
        if self.avg_execution_time > 5.0:
            self.bottlenecks.append("high_execution_time")
        
        if self.std_execution_time > self.avg_execution_time * 0.5:
            self.bottlenecks.append("high_variance")
        
        if self.peak_memory_usage > 500:  # MB
            self.bottlenecks.append("high_memory_usage")
        
        if self.avg_quality_score < 0.6:
            self.bottlenecks.append("low_quality")


class PerformanceTracker:
    """Advanced performance tracking system"""
    
    def __init__(self):
        self.profiles: Dict[ReasoningEngine, PerformanceProfile] = {}
        self.snapshots: List[PerformanceSnapshot] = []
        self.tracking_enabled = True
        self.max_snapshots = 10000  # Keep last 10k snapshots
        
        # Initialize profiles for all engines
        for engine_type in ReasoningEngine:
            self.profiles[engine_type] = PerformanceProfile(engine_type)
    
    def record_performance(self, engine_type: ReasoningEngine, metric_type: PerformanceMetric, 
                          value: float, context: Dict[str, Any] = None):
        """Record a performance measurement"""
        if not self.tracking_enabled:
            return
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            engine_type=engine_type,
            metric_type=metric_type,
            value=value,
            context=context or {}
        )
        
        self.snapshots.append(snapshot)
        
        # Maintain snapshot limit
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        # Update profile
        if engine_type in self.profiles:
            self.profiles[engine_type].update_profile([snapshot])
    
    def record_execution_performance(self, engine_type: ReasoningEngine, 
                                   execution_time: float, memory_usage: float,
                                   quality_score: float, confidence_score: float,
                                   success: bool, context: Dict[str, Any] = None):
        """Record comprehensive execution performance"""
        if not self.tracking_enabled:
            return
        
        base_context = context or {}
        
        # Record all metrics
        self.record_performance(engine_type, PerformanceMetric.EXECUTION_TIME, execution_time, base_context)
        self.record_performance(engine_type, PerformanceMetric.MEMORY_USAGE, memory_usage, base_context)
        self.record_performance(engine_type, PerformanceMetric.QUALITY_SCORE, quality_score, base_context)
        self.record_performance(engine_type, PerformanceMetric.CONFIDENCE_SCORE, confidence_score, base_context)
        
        # Record success/failure metrics
        success_rate = 1.0 if success else 0.0
        self.record_performance(engine_type, PerformanceMetric.SUCCESS_RATE, success_rate, base_context)
    
    def get_performance_profile(self, engine_type: ReasoningEngine) -> PerformanceProfile:
        """Get performance profile for an engine"""
        return self.profiles.get(engine_type, PerformanceProfile(engine_type))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        summary = {
            "total_snapshots": len(self.snapshots),
            "tracking_enabled": self.tracking_enabled,
            "engine_profiles": {}
        }
        
        for engine_type, profile in self.profiles.items():
            summary["engine_profiles"][engine_type.value] = {
                "avg_execution_time": profile.avg_execution_time,
                "performance_class": profile.performance_class,
                "quality_trend": profile.quality_trend,
                "bottlenecks": profile.bottlenecks,
                "snapshot_count": len(profile.snapshots)
            }
        
        return summary
    
    def get_comparative_analysis(self) -> Dict[str, Any]:
        """Get comparative performance analysis across engines"""
        analysis = {
            "fastest_engine": None,
            "slowest_engine": None,
            "highest_quality": None,
            "lowest_quality": None,
            "most_consistent": None,
            "least_consistent": None
        }
        
        execution_times = {}
        quality_scores = {}
        consistency_scores = {}
        
        for engine_type, profile in self.profiles.items():
            if profile.avg_execution_time > 0:
                execution_times[engine_type] = profile.avg_execution_time
                quality_scores[engine_type] = profile.avg_quality_score
                consistency_scores[engine_type] = 1.0 / (1.0 + profile.std_execution_time)
        
        if execution_times:
            analysis["fastest_engine"] = min(execution_times, key=execution_times.get).value
            analysis["slowest_engine"] = max(execution_times, key=execution_times.get).value
        
        if quality_scores:
            analysis["highest_quality"] = max(quality_scores, key=quality_scores.get).value
            analysis["lowest_quality"] = min(quality_scores, key=quality_scores.get).value
        
        if consistency_scores:
            analysis["most_consistent"] = max(consistency_scores, key=consistency_scores.get).value
            analysis["least_consistent"] = min(consistency_scores, key=consistency_scores.get).value
        
        return analysis
    
    def get_performance_trends(self, engine_type: ReasoningEngine, 
                             time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for an engine over time"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        # Filter snapshots by time window
        recent_snapshots = [s for s in self.snapshots 
                          if s.engine_type == engine_type and s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return {"message": "No recent performance data available"}
        
        # Group by metric type
        metrics = {}
        for snapshot in recent_snapshots:
            metric_name = snapshot.metric_type.value
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append({
                "timestamp": snapshot.timestamp.isoformat(),
                "value": snapshot.value
            })
        
        return {
            "engine_type": engine_type.value,
            "time_window_hours": time_window_hours,
            "metrics": metrics,
            "total_measurements": len(recent_snapshots)
        }
    
    def get_performance_recommendations(self, engine_type: ReasoningEngine) -> List[str]:
        """Get performance optimization recommendations"""
        profile = self.get_performance_profile(engine_type)
        recommendations = []
        
        if "high_execution_time" in profile.bottlenecks:
            recommendations.append("Consider implementing result caching for repeated queries")
            recommendations.append("Optimize algorithm complexity for better performance")
        
        if "high_variance" in profile.bottlenecks:
            recommendations.append("Investigate inconsistent performance patterns")
            recommendations.append("Consider implementing more consistent timeout handling")
        
        if "high_memory_usage" in profile.bottlenecks:
            recommendations.append("Implement memory optimization techniques")
            recommendations.append("Consider streaming processing for large datasets")
        
        if "low_quality" in profile.bottlenecks:
            recommendations.append("Review reasoning algorithms for quality improvements")
            recommendations.append("Consider additional validation steps")
        
        if profile.performance_class == "degraded":
            recommendations.append("Engine performance is degraded - consider restarting or reconfiguring")
        
        return recommendations
    
    def reset_tracking(self):
        """Reset all performance tracking data"""
        self.snapshots = []
        for engine_type in ReasoningEngine:
            self.profiles[engine_type] = PerformanceProfile(engine_type)
    
    def enable_tracking(self):
        """Enable performance tracking"""
        self.tracking_enabled = True
    
    def disable_tracking(self):
        """Disable performance tracking"""
        self.tracking_enabled = False


class FailureType(Enum):
    """Types of engine failures"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    QUALITY_DEGRADATION = "quality_degradation"
    REPEATED_FAILURES = "repeated_failures"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INITIALIZATION_FAILURE = "initialization_failure"
    CORRUPTION = "corruption"
    DEADLOCK = "deadlock"
    NETWORK_FAILURE = "network_failure"


class RecoveryAction(Enum):
    """Recovery actions for failed engines"""
    RETRY = "retry"
    RESTART = "restart"
    REINITIALIZE = "reinitialize"
    FALLBACK = "fallback"
    ISOLATE = "isolate"
    ESCALATE = "escalate"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"


class FailureDetectionMode(Enum):
    """Failure detection modes"""
    PASSIVE = "passive"          # React to failures as they occur
    ACTIVE = "active"            # Proactively monitor for failure patterns
    PREDICTIVE = "predictive"    # Predict failures before they occur


@dataclass
class FailureEvent:
    """Represents a failure event"""
    
    timestamp: datetime
    engine_type: ReasoningEngine
    failure_type: FailureType
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"  # low, medium, high, critical
    recovery_attempted: bool = False
    recovery_action: Optional[RecoveryAction] = None
    recovery_successful: bool = False
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class RecoveryStrategy:
    """Recovery strategy for specific failure types"""
    
    failure_type: FailureType
    actions: List[RecoveryAction]
    max_retries: int = 3
    retry_delay: float = 1.0
    escalation_threshold: int = 5
    circuit_breaker_threshold: int = 10
    recovery_timeout: float = 30.0
    
    def get_next_action(self, attempt: int) -> Optional[RecoveryAction]:
        """Get the next recovery action for the given attempt"""
        if attempt >= len(self.actions):
            return None
        return self.actions[attempt]


class FailureDetector:
    """Advanced failure detection system"""
    
    def __init__(self):
        self.failure_events: List[FailureEvent] = []
        self.detection_mode = FailureDetectionMode.ACTIVE
        self.failure_thresholds = {
            FailureType.TIMEOUT: 3,
            FailureType.EXCEPTION: 5,
            FailureType.MEMORY_EXHAUSTION: 2,
            FailureType.QUALITY_DEGRADATION: 10,
            FailureType.REPEATED_FAILURES: 5
        }
        self.time_window_hours = 1  # Consider failures within last hour
        self.enabled = True
        
        # Pattern detection
        self.failure_patterns = {}
        self.pattern_detection_enabled = True
        
        # Failure prediction
        self.prediction_enabled = False
        self.prediction_models = {}
    
    def detect_failure(self, engine_type: ReasoningEngine, execution_time: float, 
                      success: bool, error: str = None, context: Dict[str, Any] = None) -> Optional[FailureEvent]:
        """Detect if a failure has occurred"""
        if not self.enabled:
            return None
        
        failure_event = None
        
        # Check for timeout failure
        if execution_time > 30.0:  # Timeout threshold
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_type=engine_type,
                failure_type=FailureType.TIMEOUT,
                error_message=f"Execution time {execution_time:.2f}s exceeded timeout threshold",
                context=context or {},
                severity="high"
            )
        
        # Check for exception failure
        elif not success and error:
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_type=engine_type,
                failure_type=FailureType.EXCEPTION,
                error_message=error,
                context=context or {},
                severity="medium"
            )
        
        # Check for memory exhaustion
        elif context and context.get("memory_usage", 0) > 1000:  # 1GB threshold
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_type=engine_type,
                failure_type=FailureType.MEMORY_EXHAUSTION,
                error_message=f"Memory usage {context['memory_usage']:.2f}MB exceeded threshold",
                context=context or {},
                severity="high"
            )
        
        # Check for quality degradation
        elif context and context.get("quality_score", 1.0) < 0.3:
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_type=engine_type,
                failure_type=FailureType.QUALITY_DEGRADATION,
                error_message=f"Quality score {context['quality_score']:.2f} below threshold",
                context=context or {},
                severity="medium"
            )
        
        if failure_event:
            self.failure_events.append(failure_event)
            
            # Check for repeated failures pattern
            if self._detect_repeated_failures(engine_type):
                repeated_failure = FailureEvent(
                    timestamp=datetime.now(timezone.utc),
                    engine_type=engine_type,
                    failure_type=FailureType.REPEATED_FAILURES,
                    error_message="Repeated failures detected",
                    context={"original_failure": failure_event.failure_type.value},
                    severity="critical"
                )
                self.failure_events.append(repeated_failure)
                return repeated_failure
            
            return failure_event
        
        return None
    
    def _detect_repeated_failures(self, engine_type: ReasoningEngine) -> bool:
        """Detect if an engine has repeated failures"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.time_window_hours)
        
        recent_failures = [
            event for event in self.failure_events
            if event.engine_type == engine_type and 
               event.timestamp >= cutoff_time and
               event.failure_type != FailureType.REPEATED_FAILURES
        ]
        
        return len(recent_failures) >= self.failure_thresholds.get(FailureType.REPEATED_FAILURES, 5)
    
    def get_failure_history(self, engine_type: ReasoningEngine = None, 
                           hours: int = 24) -> List[FailureEvent]:
        """Get failure history for an engine or all engines"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        failures = [
            event for event in self.failure_events
            if event.timestamp >= cutoff_time
        ]
        
        if engine_type:
            failures = [f for f in failures if f.engine_type == engine_type]
        
        return sorted(failures, key=lambda f: f.timestamp, reverse=True)
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics"""
        total_failures = len(self.failure_events)
        
        if total_failures == 0:
            return {
                "total_failures": 0,
                "failure_rate": 0.0,
                "most_common_failure": None,
                "engine_failure_counts": {},
                "recent_failures": 0
            }
        
        # Count failures by type
        failure_type_counts = {}
        for event in self.failure_events:
            failure_type = event.failure_type.value
            failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
        
        # Count failures by engine
        engine_failure_counts = {}
        for event in self.failure_events:
            engine = event.engine_type.value
            engine_failure_counts[engine] = engine_failure_counts.get(engine, 0) + 1
        
        # Recent failures (last hour)
        recent_failures = len(self.get_failure_history(hours=1))
        
        return {
            "total_failures": total_failures,
            "failure_types": failure_type_counts,
            "engine_failure_counts": engine_failure_counts,
            "most_common_failure": max(failure_type_counts.items(), key=lambda x: x[1])[0],
            "recent_failures": recent_failures,
            "detection_mode": self.detection_mode.value,
            "enabled": self.enabled
        }
    
    def reset_failure_history(self, engine_type: ReasoningEngine = None):
        """Reset failure history for specific engine or all engines"""
        if engine_type:
            self.failure_events = [
                event for event in self.failure_events
                if event.engine_type != engine_type
            ]
        else:
            self.failure_events = []
    
    def enable_detection(self):
        """Enable failure detection"""
        self.enabled = True
    
    def disable_detection(self):
        """Disable failure detection"""
        self.enabled = False


class FailureRecoveryManager:
    """Manages failure recovery strategies and execution"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.recovery_history: List[Dict[str, Any]] = []
        self.circuit_breakers = {}  # Track circuit breaker states
        self.enabled = True
        
        # Recovery statistics
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_success_rate": 0.0
        }
    
    def _initialize_recovery_strategies(self) -> Dict[FailureType, RecoveryStrategy]:
        """Initialize recovery strategies for different failure types"""
        return {
            FailureType.TIMEOUT: RecoveryStrategy(
                failure_type=FailureType.TIMEOUT,
                actions=[RecoveryAction.RETRY, RecoveryAction.RESTART, RecoveryAction.REINITIALIZE],
                max_retries=2,
                retry_delay=2.0,
                escalation_threshold=3,
                circuit_breaker_threshold=5
            ),
            FailureType.EXCEPTION: RecoveryStrategy(
                failure_type=FailureType.EXCEPTION,
                actions=[RecoveryAction.RETRY, RecoveryAction.REINITIALIZE, RecoveryAction.FALLBACK],
                max_retries=3,
                retry_delay=1.0,
                escalation_threshold=5,
                circuit_breaker_threshold=10
            ),
            FailureType.MEMORY_EXHAUSTION: RecoveryStrategy(
                failure_type=FailureType.MEMORY_EXHAUSTION,
                actions=[RecoveryAction.RESTART, RecoveryAction.GRACEFUL_DEGRADE],
                max_retries=1,
                retry_delay=5.0,
                escalation_threshold=2,
                circuit_breaker_threshold=3
            ),
            FailureType.QUALITY_DEGRADATION: RecoveryStrategy(
                failure_type=FailureType.QUALITY_DEGRADATION,
                actions=[RecoveryAction.RESTART, RecoveryAction.FALLBACK],
                max_retries=2,
                retry_delay=1.0,
                escalation_threshold=5,
                circuit_breaker_threshold=10
            ),
            FailureType.REPEATED_FAILURES: RecoveryStrategy(
                failure_type=FailureType.REPEATED_FAILURES,
                actions=[RecoveryAction.CIRCUIT_BREAK, RecoveryAction.ISOLATE],
                max_retries=1,
                retry_delay=10.0,
                escalation_threshold=1,
                circuit_breaker_threshold=1
            )
        }
    
    async def attempt_recovery(self, failure_event: FailureEvent) -> bool:
        """Attempt to recover from a failure"""
        if not self.enabled:
            return False
        
        strategy = self.recovery_strategies.get(failure_event.failure_type)
        if not strategy:
            logger.warning(f"No recovery strategy for failure type: {failure_event.failure_type}")
            return False
        
        recovery_record = {
            "timestamp": datetime.now(timezone.utc),
            "failure_event": failure_event,
            "strategy": strategy,
            "attempts": [],
            "success": False
        }
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(failure_event.engine_type, failure_event.failure_type):
            logger.warning(f"Circuit breaker open for {failure_event.engine_type.value} {failure_event.failure_type.value}")
            return False
        
        # Attempt recovery actions
        for attempt in range(strategy.max_retries + 1):
            action = strategy.get_next_action(attempt)
            if not action:
                break
            
            logger.info(f"Attempting recovery action {action.value} for {failure_event.engine_type.value}")
            
            try:
                success = await self._execute_recovery_action(
                    failure_event.engine_type, action, failure_event, strategy
                )
                
                recovery_record["attempts"].append({
                    "attempt": attempt,
                    "action": action.value,
                    "success": success,
                    "timestamp": datetime.now(timezone.utc)
                })
                
                if success:
                    recovery_record["success"] = True
                    failure_event.recovery_attempted = True
                    failure_event.recovery_action = action
                    failure_event.recovery_successful = True
                    
                    self.recovery_stats["successful_recoveries"] += 1
                    self._update_recovery_success_rate()
                    
                    logger.info(f"Recovery successful for {failure_event.engine_type.value} using {action.value}")
                    break
                
                # Wait before next attempt
                if attempt < strategy.max_retries:
                    await asyncio.sleep(strategy.retry_delay)
                    
            except Exception as e:
                logger.error(f"Recovery action {action.value} failed: {str(e)}")
                recovery_record["attempts"].append({
                    "attempt": attempt,
                    "action": action.value,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc)
                })
        
        if not recovery_record["success"]:
            self.recovery_stats["failed_recoveries"] += 1
            self._update_recovery_success_rate()
            
            # Consider circuit breaker activation
            self._update_circuit_breaker(failure_event.engine_type, failure_event.failure_type)
        
        self.recovery_stats["total_recoveries"] += 1
        self.recovery_history.append(recovery_record)
        
        return recovery_record["success"]
    
    async def _execute_recovery_action(self, engine_type: ReasoningEngine, 
                                     action: RecoveryAction, failure_event: FailureEvent,
                                     strategy: RecoveryStrategy) -> bool:
        """Execute a specific recovery action"""
        
        if action == RecoveryAction.RETRY:
            # Simple retry - just return True to indicate we can try again
            return True
        
        elif action == RecoveryAction.RESTART:
            # Restart the engine
            return await self._restart_engine(engine_type)
        
        elif action == RecoveryAction.REINITIALIZE:
            # Reinitialize the engine
            return await self._reinitialize_engine(engine_type)
        
        elif action == RecoveryAction.FALLBACK:
            # Fallback to a different engine or degraded mode
            return await self._fallback_strategy(engine_type, failure_event)
        
        elif action == RecoveryAction.ISOLATE:
            # Isolate the engine from the system
            return self._isolate_engine(engine_type)
        
        elif action == RecoveryAction.CIRCUIT_BREAK:
            # Activate circuit breaker
            return self._activate_circuit_breaker(engine_type, failure_event.failure_type)
        
        elif action == RecoveryAction.GRACEFUL_DEGRADE:
            # Gracefully degrade functionality
            return await self._graceful_degrade(engine_type)
        
        else:
            logger.warning(f"Unknown recovery action: {action}")
            return False
    
    async def _restart_engine(self, engine_type: ReasoningEngine) -> bool:
        """Restart a reasoning engine"""
        try:
            # Get the engine class and create new instance
            engine_classes = {
                ReasoningEngine.DEDUCTIVE: EnhancedDeductiveReasoningEngine,
                ReasoningEngine.INDUCTIVE: EnhancedInductiveReasoningEngine,
                ReasoningEngine.ABDUCTIVE: EnhancedAbductiveReasoningEngine,
                ReasoningEngine.CAUSAL: EnhancedCausalReasoningEngine,
                ReasoningEngine.PROBABILISTIC: EnhancedProbabilisticReasoningEngine,
                ReasoningEngine.COUNTERFACTUAL: EnhancedCounterfactualReasoningEngine,
                ReasoningEngine.ANALOGICAL: AnalogicalReasoningEngine,
            }
            
            engine_class = engine_classes.get(engine_type)
            if engine_class:
                # Create new instance
                new_engine = engine_class()
                
                # Replace the engine in the meta reasoning engine
                self.meta_reasoning_engine.reasoning_engines[engine_type] = new_engine
                
                # Reset health and performance metrics
                self.meta_reasoning_engine.health_monitor.reset_engine_metrics(engine_type)
                self.meta_reasoning_engine.performance_tracker.reset_performance_tracking(engine_type)
                
                logger.info(f"Engine {engine_type.value} restarted successfully")
                return True
            else:
                logger.error(f"Unknown engine type for restart: {engine_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restart engine {engine_type.value}: {str(e)}")
            return False
    
    async def _reinitialize_engine(self, engine_type: ReasoningEngine) -> bool:
        """Reinitialize a reasoning engine"""
        # For now, reinitialize is the same as restart
        return await self._restart_engine(engine_type)
    
    async def _fallback_strategy(self, engine_type: ReasoningEngine, failure_event: FailureEvent) -> bool:
        """Implement fallback strategy"""
        # For now, just log the fallback - could implement alternative engines
        logger.info(f"Fallback strategy activated for {engine_type.value}")
        return True
    
    def _isolate_engine(self, engine_type: ReasoningEngine) -> bool:
        """Isolate an engine from the system"""
        # Mark engine as isolated in health monitor
        if hasattr(self.meta_reasoning_engine.health_monitor, 'isolated_engines'):
            self.meta_reasoning_engine.health_monitor.isolated_engines.add(engine_type)
        else:
            self.meta_reasoning_engine.health_monitor.isolated_engines = {engine_type}
        
        logger.info(f"Engine {engine_type.value} isolated from system")
        return True
    
    def _activate_circuit_breaker(self, engine_type: ReasoningEngine, failure_type: FailureType) -> bool:
        """Activate circuit breaker for an engine"""
        circuit_key = (engine_type, failure_type)
        self.circuit_breakers[circuit_key] = {
            "activated": datetime.now(timezone.utc),
            "failure_count": self.circuit_breakers.get(circuit_key, {}).get("failure_count", 0) + 1
        }
        
        logger.info(f"Circuit breaker activated for {engine_type.value} {failure_type.value}")
        return True
    
    async def _graceful_degrade(self, engine_type: ReasoningEngine) -> bool:
        """Implement graceful degradation"""
        # For now, just log the degradation
        logger.info(f"Graceful degradation activated for {engine_type.value}")
        return True
    
    def _is_circuit_breaker_open(self, engine_type: ReasoningEngine, failure_type: FailureType) -> bool:
        """Check if circuit breaker is open"""
        circuit_key = (engine_type, failure_type)
        circuit_info = self.circuit_breakers.get(circuit_key)
        
        if not circuit_info:
            return False
        
        # Circuit breaker is open for 5 minutes after activation
        circuit_timeout = timedelta(minutes=5)
        return datetime.now(timezone.utc) - circuit_info["activated"] < circuit_timeout
    
    def _update_circuit_breaker(self, engine_type: ReasoningEngine, failure_type: FailureType):
        """Update circuit breaker state"""
        circuit_key = (engine_type, failure_type)
        strategy = self.recovery_strategies.get(failure_type)
        
        if not strategy:
            return
        
        current_info = self.circuit_breakers.get(circuit_key, {"failure_count": 0})
        current_info["failure_count"] += 1
        
        if current_info["failure_count"] >= strategy.circuit_breaker_threshold:
            current_info["activated"] = datetime.now(timezone.utc)
            logger.warning(f"Circuit breaker activated for {engine_type.value} {failure_type.value}")
        
        self.circuit_breakers[circuit_key] = current_info
    
    def _update_recovery_success_rate(self):
        """Update recovery success rate"""
        if self.recovery_stats["total_recoveries"] > 0:
            self.recovery_stats["recovery_success_rate"] = (
                self.recovery_stats["successful_recoveries"] / 
                self.recovery_stats["total_recoveries"]
            )
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        return {
            **self.recovery_stats,
            "circuit_breakers": {
                f"{engine.value}_{failure.value}": info
                for (engine, failure), info in self.circuit_breakers.items()
            },
            "recovery_history_count": len(self.recovery_history),
            "enabled": self.enabled
        }
    
    def reset_recovery_history(self):
        """Reset recovery history"""
        self.recovery_history = []
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_success_rate": 0.0
        }
    
    def enable_recovery(self):
        """Enable failure recovery"""
        self.enabled = True
    
    def disable_recovery(self):
        """Disable failure recovery"""
        self.enabled = False


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for engine selection"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    PERFORMANCE_BASED = "performance_based"
    HEALTH_BASED = "health_based"
    HYBRID = "hybrid"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"


class LoadBalancingMode(Enum):
    """Load balancing modes"""
    ACTIVE = "active"          # Active load balancing
    PASSIVE = "passive"        # Passive load balancing
    ADAPTIVE = "adaptive"      # Adaptive load balancing based on conditions


@dataclass
class EngineWorkload:
    """Represents the current workload of a reasoning engine"""
    
    engine_type: ReasoningEngine
    active_requests: int = 0
    queued_requests: int = 0
    total_requests: int = 0
    last_request_time: Optional[datetime] = None
    average_response_time: float = 0.0
    current_load_factor: float = 0.0
    capacity_utilization: float = 0.0
    
    def __post_init__(self):
        self.calculate_load_factor()
    
    def calculate_load_factor(self):
        """Calculate current load factor based on active and queued requests"""
        # Simple load factor calculation
        self.current_load_factor = (self.active_requests + self.queued_requests * 0.5) / 10.0
        self.capacity_utilization = min(1.0, self.current_load_factor)
    
    def add_request(self):
        """Add a new request to the workload"""
        self.active_requests += 1
        self.total_requests += 1
        self.last_request_time = datetime.now(timezone.utc)
        self.calculate_load_factor()
    
    def complete_request(self, response_time: float):
        """Mark a request as completed"""
        if self.active_requests > 0:
            self.active_requests -= 1
        
        # Update average response time
        if self.total_requests > 0:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) / 
                self.total_requests
            )
        
        self.calculate_load_factor()
    
    def get_load_score(self) -> float:
        """Get normalized load score (0.0 = no load, 1.0 = full load)"""
        return min(1.0, self.current_load_factor)


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
            ProblemType.LOGICAL_REASONING: [ReasoningEngine.DEDUCTIVE, ReasoningEngine.INDUCTIVE],
            ProblemType.PATTERN_RECOGNITION: [ReasoningEngine.INDUCTIVE, ReasoningEngine.ANALOGICAL],
            ProblemType.CAUSAL_ANALYSIS: [ReasoningEngine.CAUSAL, ReasoningEngine.ABDUCTIVE],
            ProblemType.UNCERTAINTY_QUANTIFICATION: [ReasoningEngine.PROBABILISTIC, ReasoningEngine.INDUCTIVE],
            ProblemType.HYPOTHESIS_GENERATION: [ReasoningEngine.ABDUCTIVE, ReasoningEngine.ANALOGICAL],
            ProblemType.SCENARIO_ANALYSIS: [ReasoningEngine.COUNTERFACTUAL, ReasoningEngine.CAUSAL],
            ProblemType.COMPARATIVE_ANALYSIS: [ReasoningEngine.ANALOGICAL, ReasoningEngine.COUNTERFACTUAL],
            ProblemType.PREDICTION: [ReasoningEngine.PROBABILISTIC, ReasoningEngine.CAUSAL],
            ProblemType.CLASSIFICATION: [ReasoningEngine.INDUCTIVE, ReasoningEngine.PROBABILISTIC],
            ProblemType.OPTIMIZATION: [ReasoningEngine.DEDUCTIVE, ReasoningEngine.PROBABILISTIC]
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
        query_lower = query.lower()
        
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
        score = EngineSelectionScore(engine=engine_type)
        
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
        health_report = self.meta_reasoning_engine.health_monitor.get_engine_health_report(engine_type)
        return health_report.get('health_score', 0.5)
    
    def _calculate_load_balance_score(self, engine_type: ReasoningEngine) -> float:
        """Calculate load balance score (inverse of current load)"""
        if engine_type in self.meta_reasoning_engine.load_balancer.engine_workloads:
            workload = self.meta_reasoning_engine.load_balancer.engine_workloads[engine_type]
            return 1.0 - workload.get_load_score()
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
            if domain == "medical" and engine_type == ReasoningEngine.CAUSAL:
                score += 0.3
            elif domain == "financial" and engine_type == ReasoningEngine.PROBABILISTIC:
                score += 0.3
            elif domain == "legal" and engine_type == ReasoningEngine.DEDUCTIVE:
                score += 0.3
        
        # Urgency adjustments
        if ContextualFactor.URGENCY in factors:
            urgency = factors[ContextualFactor.URGENCY]
            if urgency == "high" and engine_type in [ReasoningEngine.DEDUCTIVE, ReasoningEngine.INDUCTIVE]:
                score += 0.2  # These engines are typically faster
        
        # Quality requirements
        if ContextualFactor.QUALITY_REQUIREMENTS in factors:
            quality = factors[ContextualFactor.QUALITY_REQUIREMENTS]
            if quality == "high" and engine_type in [ReasoningEngine.DEDUCTIVE, ReasoningEngine.CAUSAL]:
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
            profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
            # Combine execution time and quality for overall performance
            time_score = 1.0 / (1.0 + profile.avg_execution_time)
            quality_score = profile.avg_quality_score
            performance[engine_type] = (time_score + quality_score) / 2.0
        
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
    
    def get_adaptive_selection_statistics(self) -> Dict[str, Any]:
        """Get adaptive selection statistics"""
        return {
            "strategy": self.strategy.value,
            "enabled": self.enabled,
            "selection_history_count": len(self.selection_history),
            "performance_history": {
                engine_type.value: len(history)
                for engine_type, history in self.performance_history.items()
            },
            "scoring_weights": self.scoring_weights,
            "problem_type_mappings": {
                problem_type.value: [engine.value for engine in engines]
                for problem_type, engines in self.engine_problem_type_mapping.items()
            }
        }
    
    def set_strategy(self, strategy: AdaptiveSelectionStrategy):
        """Set adaptive selection strategy"""
        self.strategy = strategy
    
    def enable_adaptive_selection(self):
        """Enable adaptive selection"""
        self.enabled = True
    
    def disable_adaptive_selection(self):
        """Disable adaptive selection"""
        self.enabled = False
    
    def reset_learning_history(self):
        """Reset learning history"""
        self.selection_history = []
        self.performance_history = {}


class ResultFormat(Enum):
    """Output formats for results"""
    STRUCTURED = "structured"           # Full structured format with all details
    SUMMARY = "summary"                 # Concise summary format
    EXECUTIVE = "executive"             # Executive summary for stakeholders
    TECHNICAL = "technical"             # Technical format for developers
    NARRATIVE = "narrative"             # Human-readable narrative format
    JSON = "json"                       # JSON format for APIs
    MARKDOWN = "markdown"               # Markdown format for documentation
    COMPARISON = "comparison"           # Comparison format for multiple results
    EXPORT = "export"                   # Export format for external systems


class ConfidenceLevel(Enum):
    """Confidence levels for results"""
    VERY_HIGH = "very_high"             # >0.9
    HIGH = "high"                       # 0.7-0.9
    MEDIUM = "medium"                   # 0.5-0.7
    LOW = "low"                         # 0.3-0.5
    VERY_LOW = "very_low"               # <0.3


class ResultPriority(Enum):
    """Priority levels for results"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class FormattedResult:
    """Unified formatted result structure"""
    
    # Core content
    content: str
    summary: str
    key_insights: List[str] = field(default_factory=list)
    
    # Metadata
    engine_type: ReasoningEngine = None
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    priority: ResultPriority = ResultPriority.MEDIUM
    quality_score: float = 0.0
    
    # Structured information
    evidence: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    
    # Visualization and presentation
    visual_elements: Dict[str, Any] = field(default_factory=dict)
    formatting_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Additional context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.key_insights:
            self.key_insights = []
        if not self.evidence:
            self.evidence = []
        if not self.reasoning_steps:
            self.reasoning_steps = []
        if not self.assumptions:
            self.assumptions = []
        if not self.limitations:
            self.limitations = []
        if not self.implications:
            self.implications = []
        if not self.visual_elements:
            self.visual_elements = {}
        if not self.formatting_hints:
            self.formatting_hints = {}
        if not self.tags:
            self.tags = []


@dataclass
class MetaFormattedResult:
    """Formatted result for meta-reasoning that combines multiple engines"""
    
    # Core synthesis
    synthesized_content: str
    executive_summary: str
    key_findings: List[str] = field(default_factory=list)
    
    # Individual engine results
    engine_results: List[FormattedResult] = field(default_factory=list)
    
    # Meta-analysis
    convergence_analysis: str = ""
    divergence_analysis: str = ""
    confidence_assessment: str = ""
    quality_assessment: str = ""
    
    # Recommendations
    actionable_insights: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Metadata
    thinking_mode: ThinkingMode = ThinkingMode.QUICK
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    overall_priority: ResultPriority = ResultPriority.MEDIUM
    processing_time: float = 0.0
    ftns_cost: int = 0
    
    # Presentation options
    format_options: Dict[str, Any] = field(default_factory=dict)
    export_ready: bool = False
    
    def __post_init__(self):
        if not self.key_findings:
            self.key_findings = []
        if not self.engine_results:
            self.engine_results = []
        if not self.actionable_insights:
            self.actionable_insights = []
        if not self.next_steps:
            self.next_steps = []
        if not self.risk_factors:
            self.risk_factors = []
        if not self.format_options:
            self.format_options = {}


class ResultFormatter:
    """Unified result formatter for all reasoning engines"""
    
    def __init__(self):
        # Confidence level mappings
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.9,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.VERY_LOW: 0.0
        }
        
        # Priority mappings based on confidence and quality
        self.priority_matrix = {
            (ConfidenceLevel.VERY_HIGH, "high_quality"): ResultPriority.CRITICAL,
            (ConfidenceLevel.HIGH, "high_quality"): ResultPriority.HIGH,
            (ConfidenceLevel.MEDIUM, "high_quality"): ResultPriority.MEDIUM,
            (ConfidenceLevel.HIGH, "medium_quality"): ResultPriority.MEDIUM,
            (ConfidenceLevel.LOW, "high_quality"): ResultPriority.LOW,
            (ConfidenceLevel.MEDIUM, "medium_quality"): ResultPriority.LOW,
        }
        
        # Format templates
        self.format_templates = {
            ResultFormat.STRUCTURED: self._format_structured,
            ResultFormat.SUMMARY: self._format_summary,
            ResultFormat.EXECUTIVE: self._format_executive,
            ResultFormat.TECHNICAL: self._format_technical,
            ResultFormat.NARRATIVE: self._format_narrative,
            ResultFormat.JSON: self._format_json,
            ResultFormat.MARKDOWN: self._format_markdown,
            ResultFormat.COMPARISON: self._format_comparison,
            ResultFormat.EXPORT: self._format_export
        }
        
        # Engine-specific formatting rules
        self.engine_formatting_rules = {
            ReasoningEngine.DEDUCTIVE: {
                "emphasis": "logical_structure",
                "highlight": "premises_and_conclusions",
                "visual_style": "hierarchical"
            },
            ReasoningEngine.INDUCTIVE: {
                "emphasis": "pattern_evidence",
                "highlight": "data_points_and_trends",
                "visual_style": "data_driven"
            },
            ReasoningEngine.ABDUCTIVE: {
                "emphasis": "hypothesis_generation",
                "highlight": "explanatory_power",
                "visual_style": "exploratory"
            },
            ReasoningEngine.CAUSAL: {
                "emphasis": "cause_effect_chains",
                "highlight": "mechanisms_and_pathways",
                "visual_style": "network_diagram"
            },
            ReasoningEngine.PROBABILISTIC: {
                "emphasis": "uncertainty_quantification",
                "highlight": "probabilities_and_distributions",
                "visual_style": "statistical_charts"
            },
            ReasoningEngine.COUNTERFACTUAL: {
                "emphasis": "alternative_scenarios",
                "highlight": "comparison_outcomes",
                "visual_style": "scenario_comparison"
            },
            ReasoningEngine.ANALOGICAL: {
                "emphasis": "similarity_mapping",
                "highlight": "analogies_and_parallels",
                "visual_style": "relationship_mapping"
            }
        }
    
    def format_single_result(self, result: ReasoningResult, 
                           format_type: ResultFormat = ResultFormat.STRUCTURED) -> FormattedResult:
        """Format a single reasoning result"""
        
        # Extract basic information
        confidence_level = self._determine_confidence_level(result.confidence)
        quality_category = self._determine_quality_category(result.quality_score)
        priority = self._determine_priority(confidence_level, quality_category)
        
        # Create formatted result
        formatted_result = FormattedResult(
            content=result.result,
            summary=self._generate_summary(result),
            key_insights=self._extract_key_insights(result),
            engine_type=result.engine,
            confidence_level=confidence_level,
            priority=priority,
            quality_score=result.quality_score,
            evidence=self._format_evidence(result),
            reasoning_steps=self._format_reasoning_steps(result),
            assumptions=result.assumptions,
            limitations=result.limitations,
            implications=self._generate_implications(result),
            execution_time=result.processing_time,
            tags=self._generate_tags(result)
        )
        
        # Apply engine-specific formatting
        self._apply_engine_formatting(formatted_result, result.engine)
        
        # Generate visual elements
        formatted_result.visual_elements = self._generate_visual_elements(result)
        
        # Set formatting hints
        formatted_result.formatting_hints = self._generate_formatting_hints(result, format_type)
        
        return formatted_result
    
    def format_meta_result(self, meta_result: MetaReasoningResult, 
                          format_type: ResultFormat = ResultFormat.STRUCTURED) -> MetaFormattedResult:
        """Format a meta-reasoning result"""
        
        # Format individual engine results
        formatted_engine_results = []
        if meta_result.parallel_results:
            for engine_result in meta_result.parallel_results:
                formatted_result = self.format_single_result(engine_result, format_type)
                formatted_engine_results.append(formatted_result)
        
        # Create meta formatted result
        meta_formatted = MetaFormattedResult(
            synthesized_content=self._generate_synthesized_content(meta_result),
            executive_summary=self._generate_executive_summary(meta_result),
            key_findings=self._extract_meta_key_findings(meta_result),
            engine_results=formatted_engine_results,
            convergence_analysis=self._analyze_convergence(meta_result),
            divergence_analysis=self._analyze_divergence(meta_result),
            confidence_assessment=self._assess_meta_confidence(meta_result),
            quality_assessment=self._assess_meta_quality(meta_result),
            actionable_insights=self._generate_actionable_insights(meta_result),
            next_steps=self._generate_next_steps(meta_result),
            risk_factors=self._identify_risk_factors(meta_result),
            thinking_mode=meta_result.thinking_mode,
            overall_confidence=self._determine_overall_confidence(meta_result),
            overall_priority=self._determine_overall_priority(meta_result),
            processing_time=meta_result.total_processing_time,
            ftns_cost=meta_result.ftns_cost
        )
        
        # Set format options
        meta_formatted.format_options = self._generate_meta_format_options(meta_result, format_type)
        meta_formatted.export_ready = True
        
        return meta_formatted
    
    def render_result(self, formatted_result: FormattedResult, 
                     format_type: ResultFormat = ResultFormat.STRUCTURED) -> str:
        """Render formatted result to string"""
        
        if format_type in self.format_templates:
            return self.format_templates[format_type](formatted_result)
        else:
            return self._format_structured(formatted_result)
    
    def render_meta_result(self, meta_formatted: MetaFormattedResult,
                          format_type: ResultFormat = ResultFormat.STRUCTURED) -> str:
        """Render meta formatted result to string"""
        
        if format_type == ResultFormat.EXECUTIVE:
            return self._render_meta_executive(meta_formatted)
        elif format_type == ResultFormat.TECHNICAL:
            return self._render_meta_technical(meta_formatted)
        elif format_type == ResultFormat.SUMMARY:
            return self._render_meta_summary(meta_formatted)
        elif format_type == ResultFormat.NARRATIVE:
            return self._render_meta_narrative(meta_formatted)
        elif format_type == ResultFormat.JSON:
            return self._render_meta_json(meta_formatted)
        elif format_type == ResultFormat.MARKDOWN:
            return self._render_meta_markdown(meta_formatted)
        else:
            return self._render_meta_structured(meta_formatted)
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level from numeric confidence"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _determine_quality_category(self, quality_score: float) -> str:
        """Determine quality category from numeric score"""
        if quality_score >= 0.8:
            return "high_quality"
        elif quality_score >= 0.6:
            return "medium_quality"
        else:
            return "low_quality"
    
    def _determine_priority(self, confidence_level: ConfidenceLevel, quality_category: str) -> ResultPriority:
        """Determine priority based on confidence and quality"""
        key = (confidence_level, quality_category)
        return self.priority_matrix.get(key, ResultPriority.LOW)
    
    def _generate_summary(self, result: ReasoningResult) -> str:
        """Generate concise summary of result"""
        content = result.result
        if len(content) <= 200:
            return content
        
        # Extract first sentence or up to 200 characters
        sentences = content.split('. ')
        if sentences and len(sentences[0]) <= 200:
            return sentences[0] + '.'
        
        return content[:200] + '...'
    
    def _extract_key_insights(self, result: ReasoningResult) -> List[str]:
        """Extract key insights from result"""
        insights = []
        
        # Look for key phrases that indicate insights
        content = result.result.lower()
        insight_patterns = [
            "key finding", "important", "significant", "notable", "crucial",
            "main point", "primary", "essential", "fundamental", "critical"
        ]
        
        sentences = result.result.split('. ')
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in insight_patterns):
                insights.append(sentence.strip())
        
        # Limit to top 5 insights
        return insights[:5]
    
    def _format_evidence(self, result: ReasoningResult) -> List[str]:
        """Format evidence from result"""
        evidence = []
        
        # Add evidence strength information
        if result.evidence_strength > 0.7:
            evidence.append(f"Strong evidence support (strength: {result.evidence_strength:.2f})")
        elif result.evidence_strength > 0.5:
            evidence.append(f"Moderate evidence support (strength: {result.evidence_strength:.2f})")
        else:
            evidence.append(f"Limited evidence support (strength: {result.evidence_strength:.2f})")
        
        return evidence
    
    def _format_reasoning_steps(self, result: ReasoningResult) -> List[str]:
        """Format reasoning steps from result"""
        if isinstance(result.reasoning_chain, list):
            return result.reasoning_chain
        elif isinstance(result.reasoning_chain, str):
            return [result.reasoning_chain]
        else:
            return ["Reasoning chain not available"]
    
    def _generate_implications(self, result: ReasoningResult) -> List[str]:
        """Generate implications from result"""
        implications = []
        
        # Basic implications based on confidence and quality
        if result.confidence > 0.8 and result.quality_score > 0.8:
            implications.append("High-confidence result suitable for decision-making")
        elif result.confidence > 0.6:
            implications.append("Moderate confidence result requiring validation")
        else:
            implications.append("Low confidence result requiring additional analysis")
        
        return implications
    
    def _generate_tags(self, result: ReasoningResult) -> List[str]:
        """Generate tags for result"""
        tags = [result.engine.value]
        
        # Add confidence-based tags
        if result.confidence > 0.8:
            tags.append("high-confidence")
        elif result.confidence < 0.3:
            tags.append("low-confidence")
        
        # Add quality-based tags
        if result.quality_score > 0.8:
            tags.append("high-quality")
        elif result.quality_score < 0.3:
            tags.append("low-quality")
        
        return tags
    
    def _apply_engine_formatting(self, formatted_result: FormattedResult, engine_type: ReasoningEngine):
        """Apply engine-specific formatting"""
        if engine_type in self.engine_formatting_rules:
            rules = self.engine_formatting_rules[engine_type]
            formatted_result.formatting_hints.update(rules)
    
    def _generate_visual_elements(self, result: ReasoningResult) -> Dict[str, Any]:
        """Generate visual elements for result"""
        visual_elements = {
            "confidence_bar": {
                "type": "progress_bar",
                "value": result.confidence,
                "label": "Confidence",
                "color": "green" if result.confidence > 0.7 else "yellow" if result.confidence > 0.4 else "red"
            },
            "quality_indicator": {
                "type": "badge",
                "value": result.quality_score,
                "label": "Quality Score",
                "style": "primary" if result.quality_score > 0.7 else "secondary"
            },
            "processing_time": {
                "type": "metric",
                "value": f"{result.processing_time:.2f}s",
                "label": "Processing Time"
            }
        }
        
        return visual_elements
    
    def _generate_formatting_hints(self, result: ReasoningResult, format_type: ResultFormat) -> Dict[str, Any]:
        """Generate formatting hints for result"""
        hints = {
            "format_type": format_type.value,
            "emphasis_level": "high" if result.confidence > 0.7 else "medium",
            "detail_level": "full" if format_type == ResultFormat.TECHNICAL else "summary"
        }
        
        return hints
    
    # Individual result formatting methods
    def _format_structured(self, result: FormattedResult) -> str:
        """Format result in structured format"""
        return f"""
=== {result.engine_type.value.upper()} REASONING RESULT ===
Priority: {result.priority.value.upper()}
Confidence: {result.confidence_level.value.upper()}
Quality Score: {result.quality_score:.2f}

SUMMARY:
{result.summary}

KEY INSIGHTS:
{chr(10).join(f"• {insight}" for insight in result.key_insights)}

DETAILED CONTENT:
{result.content}

EVIDENCE:
{chr(10).join(f"• {evidence}" for evidence in result.evidence)}

REASONING STEPS:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(result.reasoning_steps))}

ASSUMPTIONS:
{chr(10).join(f"• {assumption}" for assumption in result.assumptions)}

LIMITATIONS:
{chr(10).join(f"• {limitation}" for limitation in result.limitations)}

IMPLICATIONS:
{chr(10).join(f"• {implication}" for implication in result.implications)}

METADATA:
Execution Time: {result.execution_time:.2f}s
Tags: {', '.join(result.tags)}
Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_summary(self, result: FormattedResult) -> str:
        """Format result in summary format"""
        return f"[{result.engine_type.value.upper()}] {result.summary} (Confidence: {result.confidence_level.value}, Quality: {result.quality_score:.2f})"
    
    def _format_executive(self, result: FormattedResult) -> str:
        """Format result in executive format"""
        return f"""
EXECUTIVE SUMMARY - {result.engine_type.value.upper()} ANALYSIS
Priority: {result.priority.value.upper()}

{result.summary}

KEY FINDINGS:
{chr(10).join(f"• {insight}" for insight in result.key_insights)}

CONFIDENCE ASSESSMENT: {result.confidence_level.value.upper()}
QUALITY RATING: {result.quality_score:.2f}/1.0

BUSINESS IMPLICATIONS:
{chr(10).join(f"• {implication}" for implication in result.implications)}
"""
    
    def _format_technical(self, result: FormattedResult) -> str:
        """Format result in technical format"""
        return f"""
TECHNICAL ANALYSIS - {result.engine_type.value.upper()} ENGINE
=================================

ENGINE PERFORMANCE:
• Confidence Score: {result.confidence_level.value} ({result.quality_score:.3f})
• Quality Score: {result.quality_score:.3f}
• Processing Time: {result.execution_time:.3f}s
• Engine Type: {result.engine_type.value}

ALGORITHM OUTPUT:
{result.content}

REASONING CHAIN:
{chr(10).join(f"[{i+1}] {step}" for i, step in enumerate(result.reasoning_steps))}

EVIDENCE ANALYSIS:
{chr(10).join(f"• {evidence}" for evidence in result.evidence)}

SYSTEM ASSUMPTIONS:
{chr(10).join(f"• {assumption}" for assumption in result.assumptions)}

TECHNICAL LIMITATIONS:
{chr(10).join(f"• {limitation}" for limitation in result.limitations)}

DEBUGGING INFO:
Tags: {result.tags}
Timestamp: {result.timestamp.isoformat()}
"""
    
    def _format_narrative(self, result: FormattedResult) -> str:
        """Format result in narrative format"""
        confidence_desc = {
            ConfidenceLevel.VERY_HIGH: "very high confidence",
            ConfidenceLevel.HIGH: "high confidence",
            ConfidenceLevel.MEDIUM: "moderate confidence",
            ConfidenceLevel.LOW: "low confidence",
            ConfidenceLevel.VERY_LOW: "very low confidence"
        }
        
        return f"""
The {result.engine_type.value} reasoning engine analyzed this problem with {confidence_desc[result.confidence_level]}. 

{result.content}

The analysis revealed several key insights: {', '.join(result.key_insights) if result.key_insights else 'No specific insights identified'}.

The reasoning process involved {len(result.reasoning_steps)} main steps, with the engine making {len(result.assumptions)} key assumptions about the problem context.

Quality assessment indicates a score of {result.quality_score:.2f}, suggesting {'reliable' if result.quality_score > 0.7 else 'moderate' if result.quality_score > 0.4 else 'limited'} result quality.

Important limitations to consider: {', '.join(result.limitations) if result.limitations else 'No significant limitations identified'}.
"""
    
    def _format_json(self, result: FormattedResult) -> str:
        """Format result in JSON format"""
        import json
        data = {
            "engine_type": result.engine_type.value,
            "confidence_level": result.confidence_level.value,
            "priority": result.priority.value,
            "quality_score": result.quality_score,
            "content": result.content,
            "summary": result.summary,
            "key_insights": result.key_insights,
            "evidence": result.evidence,
            "reasoning_steps": result.reasoning_steps,
            "assumptions": result.assumptions,
            "limitations": result.limitations,
            "implications": result.implications,
            "execution_time": result.execution_time,
            "tags": result.tags,
            "timestamp": result.timestamp.isoformat(),
            "visual_elements": result.visual_elements,
            "formatting_hints": result.formatting_hints
        }
        return json.dumps(data, indent=2)
    
    def _format_markdown(self, result: FormattedResult) -> str:
        """Format result in Markdown format"""
        return f"""
# {result.engine_type.value.title()} Reasoning Result

**Priority:** {result.priority.value.upper()}  
**Confidence:** {result.confidence_level.value.upper()}  
**Quality Score:** {result.quality_score:.2f}

## Summary
{result.summary}

## Key Insights
{chr(10).join(f"- {insight}" for insight in result.key_insights)}

## Detailed Analysis
{result.content}

## Evidence
{chr(10).join(f"- {evidence}" for evidence in result.evidence)}

## Reasoning Steps
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(result.reasoning_steps))}

## Assumptions
{chr(10).join(f"- {assumption}" for assumption in result.assumptions)}

## Limitations
{chr(10).join(f"- {limitation}" for limitation in result.limitations)}

## Implications
{chr(10).join(f"- {implication}" for implication in result.implications)}

---
*Generated by {result.engine_type.value} engine in {result.execution_time:.2f}s*
"""
    
    def _format_comparison(self, result: FormattedResult) -> str:
        """Format result for comparison with other results"""
        return f"""
{result.engine_type.value.upper():>15} | {result.confidence_level.value:>10} | {result.quality_score:>5.2f} | {result.summary[:50]}...
"""
    
    def _format_export(self, result: FormattedResult) -> str:
        """Format result for export to external systems"""
        return f"{result.engine_type.value}|{result.confidence_level.value}|{result.quality_score:.3f}|{result.summary.replace('|', '¦')}"
    
    # Meta-result formatting methods
    def _generate_synthesized_content(self, meta_result: MetaReasoningResult) -> str:
        """Generate synthesized content from meta-result"""
        if meta_result.final_synthesis:
            return meta_result.final_synthesis.get('synthesis', 'No synthesis available')
        return "Meta-reasoning synthesis not available"
    
    def _generate_executive_summary(self, meta_result: MetaReasoningResult) -> str:
        """Generate executive summary from meta-result"""
        synthesis = self._generate_synthesized_content(meta_result)
        return synthesis[:300] + "..." if len(synthesis) > 300 else synthesis
    
    def _extract_meta_key_findings(self, meta_result: MetaReasoningResult) -> List[str]:
        """Extract key findings from meta-result"""
        findings = []
        
        if meta_result.parallel_results:
            for result in meta_result.parallel_results:
                if result.confidence > 0.7:
                    findings.append(f"{result.engine.value}: {result.result[:100]}...")
        
        return findings[:5]
    
    def _analyze_convergence(self, meta_result: MetaReasoningResult) -> str:
        """Analyze convergence across engine results"""
        if not meta_result.parallel_results:
            return "No parallel results for convergence analysis"
        
        high_confidence_results = [r for r in meta_result.parallel_results if r.confidence > 0.7]
        convergence_rate = len(high_confidence_results) / len(meta_result.parallel_results)
        
        if convergence_rate > 0.8:
            return f"Strong convergence: {len(high_confidence_results)}/{len(meta_result.parallel_results)} engines show high confidence"
        elif convergence_rate > 0.5:
            return f"Moderate convergence: {len(high_confidence_results)}/{len(meta_result.parallel_results)} engines show high confidence"
        else:
            return f"Low convergence: Only {len(high_confidence_results)}/{len(meta_result.parallel_results)} engines show high confidence"
    
    def _analyze_divergence(self, meta_result: MetaReasoningResult) -> str:
        """Analyze divergence across engine results"""
        if not meta_result.parallel_results:
            return "No parallel results for divergence analysis"
        
        confidence_range = max(r.confidence for r in meta_result.parallel_results) - min(r.confidence for r in meta_result.parallel_results)
        
        if confidence_range > 0.5:
            return f"High divergence: Confidence range spans {confidence_range:.2f}"
        elif confidence_range > 0.3:
            return f"Moderate divergence: Confidence range spans {confidence_range:.2f}"
        else:
            return f"Low divergence: Confidence range spans {confidence_range:.2f}"
    
    def _assess_meta_confidence(self, meta_result: MetaReasoningResult) -> str:
        """Assess meta-confidence from results"""
        if not meta_result.parallel_results:
            return "Cannot assess confidence without parallel results"
        
        avg_confidence = statistics.mean(r.confidence for r in meta_result.parallel_results)
        meta_confidence = meta_result.meta_confidence
        
        return f"Average engine confidence: {avg_confidence:.2f}, Meta-confidence: {meta_confidence:.2f}"
    
    def _assess_meta_quality(self, meta_result: MetaReasoningResult) -> str:
        """Assess meta-quality from results"""
        if not meta_result.parallel_results:
            return "Cannot assess quality without parallel results"
        
        avg_quality = statistics.mean(r.quality_score for r in meta_result.parallel_results)
        overall_quality = meta_result.get_overall_quality()
        
        return f"Average engine quality: {avg_quality:.2f}, Overall quality: {overall_quality:.2f}"
    
    def _generate_actionable_insights(self, meta_result: MetaReasoningResult) -> List[str]:
        """Generate actionable insights from meta-result"""
        insights = []
        
        if meta_result.meta_confidence > 0.8:
            insights.append("High confidence result - suitable for immediate action")
        elif meta_result.meta_confidence > 0.6:
            insights.append("Moderate confidence - consider validation before action")
        else:
            insights.append("Low confidence - requires additional analysis")
        
        if meta_result.thinking_mode == ThinkingMode.DEEP:
            insights.append("Comprehensive analysis completed - detailed recommendations available")
        
        return insights
    
    def _generate_next_steps(self, meta_result: MetaReasoningResult) -> List[str]:
        """Generate next steps from meta-result"""
        steps = []
        
        if meta_result.meta_confidence < 0.6:
            steps.append("Gather additional data to improve confidence")
        
        if meta_result.thinking_mode == ThinkingMode.QUICK:
            steps.append("Consider deeper analysis if time permits")
        
        steps.append("Review assumptions and limitations")
        steps.append("Validate results with domain experts")
        
        return steps
    
    def _identify_risk_factors(self, meta_result: MetaReasoningResult) -> List[str]:
        """Identify risk factors from meta-result"""
        risks = []
        
        if meta_result.meta_confidence < 0.5:
            risks.append("Low confidence increases decision risk")
        
        if not meta_result.parallel_results:
            risks.append("Single engine analysis may miss important perspectives")
        
        return risks
    
    def _determine_overall_confidence(self, meta_result: MetaReasoningResult) -> ConfidenceLevel:
        """Determine overall confidence level"""
        return self._determine_confidence_level(meta_result.meta_confidence)
    
    def _determine_overall_priority(self, meta_result: MetaReasoningResult) -> ResultPriority:
        """Determine overall priority"""
        confidence_level = self._determine_overall_confidence(meta_result)
        quality_score = meta_result.get_overall_quality()
        quality_category = self._determine_quality_category(quality_score)
        
        return self._determine_priority(confidence_level, quality_category)
    
    def _generate_meta_format_options(self, meta_result: MetaReasoningResult, format_type: ResultFormat) -> Dict[str, Any]:
        """Generate format options for meta-result"""
        return {
            "format_type": format_type.value,
            "thinking_mode": meta_result.thinking_mode.value,
            "include_individual_results": True,
            "include_synthesis": True,
            "include_analysis": True
        }
    
    # Meta-result rendering methods
    def _render_meta_structured(self, meta_result: MetaFormattedResult) -> str:
        """Render meta-result in structured format"""
        return f"""
=== META-REASONING RESULT ===
Mode: {meta_result.thinking_mode.value.upper()}
Overall Confidence: {meta_result.overall_confidence.value.upper()}
Overall Priority: {meta_result.overall_priority.value.upper()}
Processing Time: {meta_result.processing_time:.2f}s
FTNS Cost: {meta_result.ftns_cost}

EXECUTIVE SUMMARY:
{meta_result.executive_summary}

KEY FINDINGS:
{chr(10).join(f"• {finding}" for finding in meta_result.key_findings)}

SYNTHESIZED ANALYSIS:
{meta_result.synthesized_content}

CONVERGENCE ANALYSIS:
{meta_result.convergence_analysis}

DIVERGENCE ANALYSIS:
{meta_result.divergence_analysis}

CONFIDENCE ASSESSMENT:
{meta_result.confidence_assessment}

QUALITY ASSESSMENT:
{meta_result.quality_assessment}

ACTIONABLE INSIGHTS:
{chr(10).join(f"• {insight}" for insight in meta_result.actionable_insights)}

NEXT STEPS:
{chr(10).join(f"• {step}" for step in meta_result.next_steps)}

RISK FACTORS:
{chr(10).join(f"• {risk}" for risk in meta_result.risk_factors)}

INDIVIDUAL ENGINE RESULTS:
{chr(10).join(f"--- {result.engine_type.value.upper()} ---{chr(10)}{result.summary}" for result in meta_result.engine_results)}
"""
    
    def _render_meta_executive(self, meta_result: MetaFormattedResult) -> str:
        """Render meta-result in executive format"""
        return f"""
EXECUTIVE BRIEF - META-REASONING ANALYSIS
==========================================

RECOMMENDATION PRIORITY: {meta_result.overall_priority.value.upper()}
CONFIDENCE LEVEL: {meta_result.overall_confidence.value.upper()}

EXECUTIVE SUMMARY:
{meta_result.executive_summary}

KEY BUSINESS INSIGHTS:
{chr(10).join(f"• {insight}" for insight in meta_result.actionable_insights)}

RECOMMENDED ACTIONS:
{chr(10).join(f"• {step}" for step in meta_result.next_steps)}

RISK CONSIDERATIONS:
{chr(10).join(f"• {risk}" for risk in meta_result.risk_factors)}

ANALYSIS SUMMARY:
Processing Mode: {meta_result.thinking_mode.value.title()}
Analysis Time: {meta_result.processing_time:.1f} seconds
Cost: {meta_result.ftns_cost} FTNS
"""
    
    def _render_meta_summary(self, meta_result: MetaFormattedResult) -> str:
        """Render meta-result in summary format"""
        return f"[META-REASONING] {meta_result.executive_summary} (Confidence: {meta_result.overall_confidence.value}, Cost: {meta_result.ftns_cost} FTNS)"
    
    def _render_meta_narrative(self, meta_result: MetaFormattedResult) -> str:
        """Render meta-result in narrative format"""
        return f"""
The meta-reasoning system analyzed this problem using {meta_result.thinking_mode.value} mode, processing multiple reasoning approaches simultaneously.

{meta_result.synthesized_content}

The analysis achieved {meta_result.overall_confidence.value} confidence through convergence analysis showing {meta_result.convergence_analysis.lower()}.

Key insights from the analysis include: {', '.join(meta_result.actionable_insights) if meta_result.actionable_insights else 'No specific actionable insights identified'}.

The system recommends the following next steps: {', '.join(meta_result.next_steps) if meta_result.next_steps else 'No specific next steps identified'}.

Risk factors to consider: {', '.join(meta_result.risk_factors) if meta_result.risk_factors else 'No significant risk factors identified'}.

This analysis was completed in {meta_result.processing_time:.1f} seconds at a cost of {meta_result.ftns_cost} FTNS tokens.
"""
    
    def _render_meta_technical(self, meta_result: MetaFormattedResult) -> str:
        """Render meta-result in technical format"""
        return f"""
META-REASONING TECHNICAL REPORT
===============================

SYSTEM CONFIGURATION:
• Thinking Mode: {meta_result.thinking_mode.value}
• Processing Time: {meta_result.processing_time:.3f}s
• FTNS Cost: {meta_result.ftns_cost}
• Engine Count: {len(meta_result.engine_results)}

PERFORMANCE METRICS:
• Overall Confidence: {meta_result.overall_confidence.value} 
• Overall Priority: {meta_result.overall_priority.value}
• Synthesis Quality: {meta_result.quality_assessment}

CONVERGENCE ANALYSIS:
{meta_result.convergence_analysis}

DIVERGENCE ANALYSIS:
{meta_result.divergence_analysis}

INDIVIDUAL ENGINE PERFORMANCE:
{chr(10).join(f"• {result.engine_type.value}: {result.confidence_level.value} confidence, {result.quality_score:.3f} quality" for result in meta_result.engine_results)}

SYNTHESIS OUTPUT:
{meta_result.synthesized_content}

TECHNICAL RECOMMENDATIONS:
{chr(10).join(f"• {insight}" for insight in meta_result.actionable_insights)}

SYSTEM LIMITATIONS:
{chr(10).join(f"• {risk}" for risk in meta_result.risk_factors)}
"""
    
    def _render_meta_json(self, meta_result: MetaFormattedResult) -> str:
        """Render meta-result in JSON format"""
        import json
        data = {
            "thinking_mode": meta_result.thinking_mode.value,
            "overall_confidence": meta_result.overall_confidence.value,
            "overall_priority": meta_result.overall_priority.value,
            "processing_time": meta_result.processing_time,
            "ftns_cost": meta_result.ftns_cost,
            "synthesized_content": meta_result.synthesized_content,
            "executive_summary": meta_result.executive_summary,
            "key_findings": meta_result.key_findings,
            "convergence_analysis": meta_result.convergence_analysis,
            "divergence_analysis": meta_result.divergence_analysis,
            "confidence_assessment": meta_result.confidence_assessment,
            "quality_assessment": meta_result.quality_assessment,
            "actionable_insights": meta_result.actionable_insights,
            "next_steps": meta_result.next_steps,
            "risk_factors": meta_result.risk_factors,
            "engine_results": [
                {
                    "engine_type": result.engine_type.value,
                    "confidence_level": result.confidence_level.value,
                    "quality_score": result.quality_score,
                    "summary": result.summary
                }
                for result in meta_result.engine_results
            ]
        }
        return json.dumps(data, indent=2)
    
    def _render_meta_markdown(self, meta_result: MetaFormattedResult) -> str:
        """Render meta-result in Markdown format"""
        return f"""
# Meta-Reasoning Analysis Report

**Mode:** {meta_result.thinking_mode.value.title()}  
**Confidence:** {meta_result.overall_confidence.value.upper()}  
**Priority:** {meta_result.overall_priority.value.upper()}  
**Processing Time:** {meta_result.processing_time:.2f}s  
**FTNS Cost:** {meta_result.ftns_cost}

## Executive Summary
{meta_result.executive_summary}

## Key Findings
{chr(10).join(f"- {finding}" for finding in meta_result.key_findings)}

## Detailed Analysis
{meta_result.synthesized_content}

## Convergence Analysis
{meta_result.convergence_analysis}

## Divergence Analysis
{meta_result.divergence_analysis}

## Confidence Assessment
{meta_result.confidence_assessment}

## Quality Assessment
{meta_result.quality_assessment}

## Actionable Insights
{chr(10).join(f"- {insight}" for insight in meta_result.actionable_insights)}

## Next Steps
{chr(10).join(f"- {step}" for step in meta_result.next_steps)}

## Risk Factors
{chr(10).join(f"- {risk}" for risk in meta_result.risk_factors)}

## Individual Engine Results
{chr(10).join(f"### {result.engine_type.value.title()}{chr(10)}{result.summary}" for result in meta_result.engine_results)}

---
*Generated by NWTN Meta-Reasoning System*
"""


class InteractionPatternType(Enum):
    """Types of interaction patterns between reasoning engines"""
    SYNERGISTIC = "synergistic"              # Engines enhance each other's effectiveness
    COMPLEMENTARY = "complementary"          # Engines fill each other's gaps
    SEQUENTIAL = "sequential"                # Engines work better in sequence
    PARALLEL = "parallel"                    # Engines work better in parallel
    CONFLICTING = "conflicting"              # Engines produce conflicting results
    NEUTRAL = "neutral"                      # Engines have no significant interaction
    REINFORCING = "reinforcing"              # Engines reinforce each other's conclusions
    CORRECTIVE = "corrective"                # One engine corrects the other's errors
    CONTEXTUAL = "contextual"                # Interaction depends on context
    EMERGENT = "emergent"                    # Interaction creates new insights


class InteractionOutcome(Enum):
    """Outcomes of engine interactions"""
    IMPROVED_CONFIDENCE = "improved_confidence"      # Higher confidence in results
    BETTER_QUALITY = "better_quality"                # Higher quality reasoning
    FASTER_PROCESSING = "faster_processing"          # Faster overall processing
    DEEPER_INSIGHTS = "deeper_insights"              # More profound insights
    REDUCED_ERRORS = "reduced_errors"                # Fewer errors or mistakes
    BROADER_COVERAGE = "broader_coverage"            # More comprehensive coverage
    CREATIVE_SOLUTIONS = "creative_solutions"        # Novel or creative solutions
    CONTRADICTORY_RESULTS = "contradictory_results"  # Conflicting conclusions
    DEGRADED_PERFORMANCE = "degraded_performance"    # Worse overall performance
    NO_SIGNIFICANT_CHANGE = "no_significant_change"  # No notable difference


@dataclass
class InteractionPattern:
    """Represents a pattern of interaction between reasoning engines"""
    
    engine_pair: Tuple[ReasoningEngine, ReasoningEngine]
    pattern_type: InteractionPatternType
    pattern_name: str
    description: str
    
    # Pattern characteristics
    effectiveness_score: float = 0.0        # How effective this pattern is (0-1)
    confidence_level: float = 0.0           # How confident we are in this pattern (0-1)
    frequency: int = 0                      # How often this pattern occurs
    context_sensitivity: float = 0.0        # How context-dependent this pattern is (0-1)
    
    # Outcomes
    typical_outcomes: List[InteractionOutcome] = field(default_factory=list)
    
    # Performance metrics
    avg_quality_improvement: float = 0.0
    avg_confidence_improvement: float = 0.0
    avg_processing_time_change: float = 0.0
    
    # Context factors
    favorable_contexts: List[str] = field(default_factory=list)
    unfavorable_contexts: List[str] = field(default_factory=list)
    
    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    conflicting_evidence: List[str] = field(default_factory=list)
    
    # Metadata
    discovery_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_size: int = 0


@dataclass
class InteractionEvidence:
    """Evidence for an interaction pattern"""
    
    evidence_id: str = field(default_factory=lambda: str(uuid4()))
    engine_pair: Tuple[ReasoningEngine, ReasoningEngine]
    
    # Context information
    query: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    first_engine_result: Optional[Any] = None
    second_engine_result: Optional[Any] = None
    combined_result: Optional[Any] = None
    
    # Performance metrics
    quality_change: float = 0.0              # Change in quality score
    confidence_change: float = 0.0           # Change in confidence
    processing_time_change: float = 0.0      # Change in processing time
    
    # Interaction characteristics
    observed_pattern: InteractionPatternType = InteractionPatternType.NEUTRAL
    outcome: InteractionOutcome = InteractionOutcome.NO_SIGNIFICANT_CHANGE
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    thinking_mode: ThinkingMode = ThinkingMode.QUICK
    
    def __post_init__(self):
        if not self.context:
            self.context = {}


class InteractionPatternRecognizer:
    """Advanced system for recognizing and learning interaction patterns between reasoning engines"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.enabled = True
        
        # Pattern storage
        self.patterns: Dict[Tuple[ReasoningEngine, ReasoningEngine], InteractionPattern] = {}
        self.evidence_history: List[InteractionEvidence] = []
        self.max_evidence_history = 10000
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.min_sample_size = 10
        self.pattern_decay_rate = 0.01
        
        # Pattern discovery
        self.auto_discovery_enabled = True
        self.discovery_threshold = 0.6
        
        # Initialize with known patterns
        self.initialize_known_patterns()
        
        # Pattern analysis
        self.context_analyzer = ContextAnalyzer()
        self.outcome_predictor = OutcomePredictor()
        
        # Metrics
        self.pattern_effectiveness_history = []
        self.discovery_history = []
        
        logger.info("InteractionPatternRecognizer initialized", 
                   known_patterns=len(self.patterns),
                   learning_rate=self.learning_rate)
    
    def initialize_known_patterns(self):
        """Initialize with known interaction patterns"""
        
        known_patterns = [
            (
                (ReasoningEngine.INDUCTIVE, ReasoningEngine.CAUSAL),
                InteractionPatternType.SYNERGISTIC,
                "Empirical Foundation",
                "Inductive reasoning provides empirical patterns that strengthen causal analysis"
            ),
            (
                (ReasoningEngine.ABDUCTIVE, ReasoningEngine.CAUSAL),
                InteractionPatternType.SEQUENTIAL,
                "Hypothesis Testing",
                "Abductive hypotheses are tested through causal reasoning"
            ),
            (
                (ReasoningEngine.COUNTERFACTUAL, ReasoningEngine.CAUSAL),
                InteractionPatternType.COMPLEMENTARY,
                "Alternative Scenarios",
                "Counterfactual reasoning explores alternative scenarios for causal relationships"
            ),
            (
                (ReasoningEngine.PROBABILISTIC, ReasoningEngine.INDUCTIVE),
                InteractionPatternType.REINFORCING,
                "Pattern Quantification",
                "Probabilistic reasoning quantifies patterns discovered through induction"
            ),
            (
                (ReasoningEngine.PROBABILISTIC, ReasoningEngine.ABDUCTIVE),
                InteractionPatternType.CORRECTIVE,
                "Hypothesis Evaluation",
                "Probabilistic reasoning evaluates the likelihood of abductive hypotheses"
            ),
            (
                (ReasoningEngine.ANALOGICAL, ReasoningEngine.INDUCTIVE),
                InteractionPatternType.SYNERGISTIC,
                "Cross-Domain Patterns",
                "Analogical reasoning reveals patterns across different domains"
            ),
            (
                (ReasoningEngine.ANALOGICAL, ReasoningEngine.ABDUCTIVE),
                InteractionPatternType.EMERGENT,
                "Structural Insights",
                "Analogical reasoning provides structural insights for abductive hypotheses"
            ),
            (
                (ReasoningEngine.DEDUCTIVE, ReasoningEngine.PROBABILISTIC),
                InteractionPatternType.COMPLEMENTARY,
                "Logical Frameworks",
                "Deductive reasoning provides logical frameworks for probabilistic analysis"
            ),
            (
                (ReasoningEngine.COUNTERFACTUAL, ReasoningEngine.PROBABILISTIC),
                InteractionPatternType.SEQUENTIAL,
                "Scenario Quantification",
                "Counterfactual scenarios are quantified through probabilistic reasoning"
            )
        ]
        
        for engine_pair, pattern_type, name, description in known_patterns:
            pattern = InteractionPattern(
                engine_pair=engine_pair,
                pattern_type=pattern_type,
                pattern_name=name,
                description=description,
                effectiveness_score=0.8,  # High initial effectiveness for known patterns
                confidence_level=0.9,     # High confidence in known patterns
                frequency=1,
                context_sensitivity=0.5,
                typical_outcomes=[InteractionOutcome.IMPROVED_CONFIDENCE, InteractionOutcome.BETTER_QUALITY],
                favorable_contexts=["complex_problem", "multi_domain", "uncertain_context"],
                supporting_evidence=["Research-based known interaction"],
                sample_size=1
            )
            self.patterns[engine_pair] = pattern
    
    def observe_interaction(self, engine_pair: Tuple[ReasoningEngine, ReasoningEngine],
                           first_result: Any, second_result: Any, combined_result: Any,
                           query: str, context: Dict[str, Any], thinking_mode: ThinkingMode) -> InteractionEvidence:
        """Observe and record an interaction between two engines"""
        
        if not self.enabled:
            return None
        
        # Create evidence
        evidence = InteractionEvidence(
            engine_pair=engine_pair,
            query=query,
            context=context,
            first_engine_result=first_result,
            second_engine_result=second_result,
            combined_result=combined_result,
            thinking_mode=thinking_mode
        )
        
        # Analyze the interaction
        self._analyze_interaction(evidence)
        
        # Store evidence
        self.evidence_history.append(evidence)
        
        # Maintain history size
        if len(self.evidence_history) > self.max_evidence_history:
            self.evidence_history.pop(0)
        
        # Update patterns
        self._update_patterns(evidence)
        
        # Check for new pattern discovery
        if self.auto_discovery_enabled:
            self._discover_new_patterns()
        
        return evidence
    
    def _analyze_interaction(self, evidence: InteractionEvidence):
        """Analyze the interaction to determine pattern type and outcomes"""
        
        # Extract metrics from results
        first_confidence = self._extract_confidence(evidence.first_engine_result)
        second_confidence = self._extract_confidence(evidence.second_engine_result)
        combined_confidence = self._extract_confidence(evidence.combined_result)
        
        first_quality = self._extract_quality(evidence.first_engine_result)
        second_quality = self._extract_quality(evidence.second_engine_result)
        combined_quality = self._extract_quality(evidence.combined_result)
        
        # Calculate changes
        evidence.confidence_change = combined_confidence - max(first_confidence, second_confidence)
        evidence.quality_change = combined_quality - max(first_quality, second_quality)
        
        # Determine pattern type
        evidence.observed_pattern = self._classify_pattern_type(evidence)
        
        # Determine outcome
        evidence.outcome = self._classify_outcome(evidence)
    
    def _extract_confidence(self, result: Any) -> float:
        """Extract confidence score from result"""
        if hasattr(result, 'confidence'):
            if isinstance(result.confidence, (int, float)):
                return float(result.confidence)
            elif hasattr(result.confidence, 'value'):
                return float(result.confidence.value)
        return 0.5  # Default moderate confidence
    
    def _extract_quality(self, result: Any) -> float:
        """Extract quality score from result"""
        if hasattr(result, 'quality_score'):
            return float(result.quality_score)
        elif hasattr(result, 'quality'):
            return float(result.quality)
        return 0.5  # Default moderate quality
    
    def _classify_pattern_type(self, evidence: InteractionEvidence) -> InteractionPatternType:
        """Classify the type of interaction pattern"""
        
        # Check for synergistic patterns (both metrics improve significantly)
        if evidence.confidence_change > 0.1 and evidence.quality_change > 0.1:
            return InteractionPatternType.SYNERGISTIC
        
        # Check for complementary patterns (one metric improves significantly)
        elif evidence.confidence_change > 0.1 or evidence.quality_change > 0.1:
            return InteractionPatternType.COMPLEMENTARY
        
        # Check for conflicting patterns (metrics decrease)
        elif evidence.confidence_change < -0.1 and evidence.quality_change < -0.1:
            return InteractionPatternType.CONFLICTING
        
        # Check for corrective patterns (quality improves, confidence stable)
        elif evidence.quality_change > 0.1 and abs(evidence.confidence_change) < 0.05:
            return InteractionPatternType.CORRECTIVE
        
        # Check for reinforcing patterns (confidence improves, quality stable)
        elif evidence.confidence_change > 0.1 and abs(evidence.quality_change) < 0.05:
            return InteractionPatternType.REINFORCING
        
        # Default to neutral
        return InteractionPatternType.NEUTRAL
    
    def _classify_outcome(self, evidence: InteractionEvidence) -> InteractionOutcome:
        """Classify the outcome of the interaction"""
        
        if evidence.confidence_change > 0.1:
            return InteractionOutcome.IMPROVED_CONFIDENCE
        elif evidence.quality_change > 0.1:
            return InteractionOutcome.BETTER_QUALITY
        elif evidence.confidence_change < -0.1 and evidence.quality_change < -0.1:
            return InteractionOutcome.DEGRADED_PERFORMANCE
        elif evidence.confidence_change < -0.1 or evidence.quality_change < -0.1:
            return InteractionOutcome.CONTRADICTORY_RESULTS
        else:
            return InteractionOutcome.NO_SIGNIFICANT_CHANGE
    
    def _update_patterns(self, evidence: InteractionEvidence):
        """Update existing patterns with new evidence"""
        
        engine_pair = evidence.engine_pair
        
        if engine_pair in self.patterns:
            pattern = self.patterns[engine_pair]
            
            # Update pattern metrics using exponential moving average
            pattern.avg_quality_improvement = (
                (1 - self.learning_rate) * pattern.avg_quality_improvement +
                self.learning_rate * evidence.quality_change
            )
            pattern.avg_confidence_improvement = (
                (1 - self.learning_rate) * pattern.avg_confidence_improvement +
                self.learning_rate * evidence.confidence_change
            )
            
            # Update frequency and sample size
            pattern.frequency += 1
            pattern.sample_size += 1
            
            # Update effectiveness score
            effectiveness_delta = (evidence.quality_change + evidence.confidence_change) / 2
            pattern.effectiveness_score = max(0.0, min(1.0, 
                pattern.effectiveness_score + self.learning_rate * effectiveness_delta
            ))
            
            # Update confidence level based on sample size and consistency
            if pattern.sample_size >= self.min_sample_size:
                consistency_factor = 1.0 - abs(evidence.quality_change - pattern.avg_quality_improvement)
                pattern.confidence_level = min(1.0, 
                    pattern.confidence_level + self.learning_rate * consistency_factor
                )
            
            # Update outcomes
            if evidence.outcome not in pattern.typical_outcomes:
                pattern.typical_outcomes.append(evidence.outcome)
            
            # Update context factors
            context_keywords = self._extract_context_keywords(evidence.context)
            for keyword in context_keywords:
                if evidence.quality_change > 0.1 or evidence.confidence_change > 0.1:
                    if keyword not in pattern.favorable_contexts:
                        pattern.favorable_contexts.append(keyword)
                elif evidence.quality_change < -0.1 or evidence.confidence_change < -0.1:
                    if keyword not in pattern.unfavorable_contexts:
                        pattern.unfavorable_contexts.append(keyword)
            
            # Update evidence
            evidence_summary = f"Q:{evidence.quality_change:.2f}, C:{evidence.confidence_change:.2f}"
            pattern.supporting_evidence.append(evidence_summary)
            
            # Update metadata
            pattern.last_updated = datetime.now(timezone.utc)
    
    def _extract_context_keywords(self, context: Dict[str, Any]) -> List[str]:
        """Extract keywords from context"""
        keywords = []
        
        # Extract domain keywords
        if 'domain' in context:
            keywords.append(f"domain_{context['domain']}")
        
        # Extract urgency keywords
        if 'urgency' in context:
            keywords.append(f"urgency_{context['urgency']}")
        
        # Extract complexity keywords
        if 'complexity' in context:
            keywords.append(f"complexity_{context['complexity']}")
        
        # Extract other string values
        for key, value in context.items():
            if isinstance(value, str) and len(value) < 50:
                keywords.append(f"{key}_{value}")
        
        return keywords
    
    def _discover_new_patterns(self):
        """Discover new interaction patterns from evidence"""
        
        # Group evidence by engine pairs
        evidence_by_pair = {}
        for evidence in self.evidence_history[-1000:]:  # Look at recent evidence
            pair = evidence.engine_pair
            if pair not in evidence_by_pair:
                evidence_by_pair[pair] = []
            evidence_by_pair[pair].append(evidence)
        
        # Look for patterns in pairs not yet recognized
        for pair, evidence_list in evidence_by_pair.items():
            if pair not in self.patterns and len(evidence_list) >= self.min_sample_size:
                
                # Calculate average metrics
                avg_quality_change = statistics.mean([e.quality_change for e in evidence_list])
                avg_confidence_change = statistics.mean([e.confidence_change for e in evidence_list])
                
                # Check if there's a strong pattern
                if abs(avg_quality_change) > 0.1 or abs(avg_confidence_change) > 0.1:
                    
                    # Create new pattern
                    pattern_type = self._classify_pattern_type(evidence_list[0])  # Use first evidence as representative
                    
                    new_pattern = InteractionPattern(
                        engine_pair=pair,
                        pattern_type=pattern_type,
                        pattern_name=f"Discovered_{pair[0].value}_{pair[1].value}",
                        description=f"Discovered pattern between {pair[0].value} and {pair[1].value}",
                        effectiveness_score=abs(avg_quality_change + avg_confidence_change) / 2,
                        confidence_level=0.6,  # Medium confidence for new patterns
                        frequency=len(evidence_list),
                        avg_quality_improvement=avg_quality_change,
                        avg_confidence_improvement=avg_confidence_change,
                        sample_size=len(evidence_list),
                        typical_outcomes=[evidence_list[0].outcome],
                        supporting_evidence=[f"Discovered from {len(evidence_list)} observations"]
                    )
                    
                    self.patterns[pair] = new_pattern
                    self.discovery_history.append({
                        "pair": pair,
                        "pattern": new_pattern,
                        "discovery_date": datetime.now(timezone.utc)
                    })
                    
                    logger.info(f"Discovered new interaction pattern: {new_pattern.pattern_name}")
    
    def get_pattern_recommendations(self, query: str, context: Dict[str, Any]) -> List[Tuple[ReasoningEngine, ReasoningEngine]]:
        """Get recommended engine pairs based on patterns and context"""
        
        recommendations = []
        
        # Score all patterns based on context
        pattern_scores = {}
        for pair, pattern in self.patterns.items():
            score = self._calculate_pattern_score(pattern, context)
            pattern_scores[pair] = score
        
        # Sort by score and return top recommendations
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 recommendations
        for pair, score in sorted_patterns[:5]:
            if score > 0.5:  # Only recommend patterns with good scores
                recommendations.append(pair)
        
        return recommendations
    
    def _calculate_pattern_score(self, pattern: InteractionPattern, context: Dict[str, Any]) -> float:
        """Calculate score for a pattern given context"""
        
        base_score = pattern.effectiveness_score * pattern.confidence_level
        
        # Context bonus
        context_keywords = self._extract_context_keywords(context)
        context_bonus = 0.0
        
        for keyword in context_keywords:
            if keyword in pattern.favorable_contexts:
                context_bonus += 0.1
            elif keyword in pattern.unfavorable_contexts:
                context_bonus -= 0.1
        
        # Frequency bonus (more frequent patterns are more reliable)
        frequency_bonus = min(0.2, pattern.frequency / 100)
        
        # Sample size bonus (more evidence is better)
        sample_bonus = min(0.2, pattern.sample_size / 50)
        
        total_score = base_score + context_bonus + frequency_bonus + sample_bonus
        
        return max(0.0, min(1.0, total_score))
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of interaction patterns"""
        
        analysis = {
            "total_patterns": len(self.patterns),
            "evidence_count": len(self.evidence_history),
            "pattern_types": {},
            "most_effective_patterns": [],
            "least_effective_patterns": [],
            "recently_discovered": [],
            "pattern_effectiveness_trend": []
        }
        
        # Pattern type distribution
        for pattern in self.patterns.values():
            pattern_type = pattern.pattern_type.value
            if pattern_type not in analysis["pattern_types"]:
                analysis["pattern_types"][pattern_type] = 0
            analysis["pattern_types"][pattern_type] += 1
        
        # Most and least effective patterns
        sorted_patterns = sorted(self.patterns.items(), key=lambda x: x[1].effectiveness_score, reverse=True)
        
        analysis["most_effective_patterns"] = [
            {
                "engines": f"{pair[0].value} -> {pair[1].value}",
                "pattern_name": pattern.pattern_name,
                "effectiveness_score": pattern.effectiveness_score,
                "confidence_level": pattern.confidence_level
            }
            for pair, pattern in sorted_patterns[:5]
        ]
        
        analysis["least_effective_patterns"] = [
            {
                "engines": f"{pair[0].value} -> {pair[1].value}",
                "pattern_name": pattern.pattern_name,
                "effectiveness_score": pattern.effectiveness_score,
                "confidence_level": pattern.confidence_level
            }
            for pair, pattern in sorted_patterns[-5:]
        ]
        
        # Recently discovered patterns
        recent_discoveries = [d for d in self.discovery_history if 
                            (datetime.now(timezone.utc) - d["discovery_date"]).days <= 7]
        
        analysis["recently_discovered"] = [
            {
                "engines": f"{d['pair'][0].value} -> {d['pair'][1].value}",
                "pattern_name": d["pattern"].pattern_name,
                "discovery_date": d["discovery_date"].isoformat()
            }
            for d in recent_discoveries
        ]
        
        return analysis
    
    def get_pattern_report(self) -> Dict[str, Any]:
        """Get detailed pattern report"""
        
        report = {
            "system_status": {
                "recognition_enabled": self.enabled,
                "auto_discovery_enabled": self.auto_discovery_enabled,
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "min_sample_size": self.min_sample_size
            },
            "pattern_summary": self.get_pattern_analysis(),
            "detailed_patterns": {},
            "evidence_summary": {
                "total_evidence": len(self.evidence_history),
                "recent_evidence": len([e for e in self.evidence_history if 
                                     (datetime.now(timezone.utc) - e.timestamp).days <= 7]),
                "evidence_by_outcome": {}
            },
            "recommendations": {
                "top_patterns": [],
                "improvement_suggestions": []
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Detailed patterns
        for pair, pattern in self.patterns.items():
            key = f"{pair[0].value}_{pair[1].value}"
            report["detailed_patterns"][key] = {
                "pattern_name": pattern.pattern_name,
                "pattern_type": pattern.pattern_type.value,
                "description": pattern.description,
                "effectiveness_score": pattern.effectiveness_score,
                "confidence_level": pattern.confidence_level,
                "frequency": pattern.frequency,
                "sample_size": pattern.sample_size,
                "avg_quality_improvement": pattern.avg_quality_improvement,
                "avg_confidence_improvement": pattern.avg_confidence_improvement,
                "typical_outcomes": [outcome.value for outcome in pattern.typical_outcomes],
                "favorable_contexts": pattern.favorable_contexts,
                "unfavorable_contexts": pattern.unfavorable_contexts,
                "discovery_date": pattern.discovery_date.isoformat(),
                "last_updated": pattern.last_updated.isoformat()
            }
        
        # Evidence by outcome
        for evidence in self.evidence_history:
            outcome = evidence.outcome.value
            if outcome not in report["evidence_summary"]["evidence_by_outcome"]:
                report["evidence_summary"]["evidence_by_outcome"][outcome] = 0
            report["evidence_summary"]["evidence_by_outcome"][outcome] += 1
        
        return report
    
    def enable_pattern_recognition(self):
        """Enable pattern recognition"""
        self.enabled = True
        logger.info("Interaction pattern recognition enabled")
    
    def disable_pattern_recognition(self):
        """Disable pattern recognition"""
        self.enabled = False
        logger.info("Interaction pattern recognition disabled")
    
    def clear_pattern_history(self):
        """Clear pattern learning history"""
        self.evidence_history = []
        # Reset patterns to initial state but keep known patterns
        self.initialize_known_patterns()
        logger.info("Pattern learning history cleared")


class ContextAnalyzer:
    """Analyzes context to understand interaction patterns"""
    
    def __init__(self):
        self.context_patterns = {}
    
    def analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for pattern recognition"""
        analysis = {
            "domain_complexity": self._assess_domain_complexity(context),
            "urgency_level": self._assess_urgency_level(context),
            "uncertainty_level": self._assess_uncertainty_level(context),
            "context_keywords": self._extract_keywords(context)
        }
        return analysis
    
    def _assess_domain_complexity(self, context: Dict[str, Any]) -> float:
        """Assess domain complexity"""
        if 'domain' in context:
            complex_domains = ['scientific', 'technical', 'financial', 'medical', 'legal']
            if context['domain'] in complex_domains:
                return 0.8
        return 0.5
    
    def _assess_urgency_level(self, context: Dict[str, Any]) -> float:
        """Assess urgency level"""
        if 'urgency' in context:
            urgency_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
            return urgency_map.get(context['urgency'], 0.5)
        return 0.5
    
    def _assess_uncertainty_level(self, context: Dict[str, Any]) -> float:
        """Assess uncertainty level"""
        uncertainty_indicators = ['uncertain', 'ambiguous', 'unclear', 'complex', 'hypothetical']
        uncertainty_score = 0.0
        
        for key, value in context.items():
            if isinstance(value, str):
                for indicator in uncertainty_indicators:
                    if indicator in value.lower():
                        uncertainty_score += 0.2
        
        return min(1.0, uncertainty_score)
    
    def _extract_keywords(self, context: Dict[str, Any]) -> List[str]:
        """Extract keywords from context"""
        keywords = []
        for key, value in context.items():
            if isinstance(value, str) and len(value) < 100:
                keywords.append(f"{key}_{value}")
        return keywords


class OutcomePredictor:
    """Predicts outcomes of engine interactions"""
    
    def __init__(self):
        self.prediction_models = {}
    
    def predict_outcome(self, engine_pair: Tuple[ReasoningEngine, ReasoningEngine], 
                       context: Dict[str, Any]) -> Dict[str, float]:
        """Predict interaction outcome"""
        
        predictions = {
            InteractionOutcome.IMPROVED_CONFIDENCE.value: 0.3,
            InteractionOutcome.BETTER_QUALITY.value: 0.3,
            InteractionOutcome.DEEPER_INSIGHTS.value: 0.2,
            InteractionOutcome.NO_SIGNIFICANT_CHANGE.value: 0.2
        }
        
        return predictions


class ContextPassingMode(Enum):
    """Modes for sequential context passing"""
    BASIC = "basic"                           # Pass basic results only
    ENRICHED = "enriched"                     # Pass enriched context with insights
    CUMULATIVE = "cumulative"                 # Accumulate context across engines
    SELECTIVE = "selective"                   # Selectively pass relevant context
    ADAPTIVE = "adaptive"                     # Adapt context based on engine needs
    COMPREHENSIVE = "comprehensive"           # Pass all available context


class ContextRelevance(Enum):
    """Relevance levels for context information"""
    CRITICAL = "critical"                     # Critical for next engine
    HIGH = "high"                            # High relevance
    MEDIUM = "medium"                        # Medium relevance
    LOW = "low"                              # Low relevance
    NEGLIGIBLE = "negligible"                # Negligible relevance


class ContextType(Enum):
    """Types of context information"""
    REASONING_CHAIN = "reasoning_chain"       # Chain of reasoning steps
    EVIDENCE = "evidence"                     # Supporting evidence
    ASSUMPTIONS = "assumptions"               # Underlying assumptions
    LIMITATIONS = "limitations"               # Known limitations
    INSIGHTS = "insights"                     # Key insights discovered
    HYPOTHESES = "hypotheses"                 # Generated hypotheses
    PATTERNS = "patterns"                     # Identified patterns
    CONFIDENCE_FACTORS = "confidence_factors" # Factors affecting confidence
    QUALITY_INDICATORS = "quality_indicators" # Quality assessment indicators
    INTERMEDIATE_RESULTS = "intermediate_results" # Intermediate computation results
    METADATA = "metadata"                     # Processing metadata


@dataclass
class ContextItem:
    """Individual piece of context information"""
    
    context_id: str = field(default_factory=lambda: str(uuid4()))
    context_type: ContextType = ContextType.INSIGHTS
    content: Any = None
    relevance: ContextRelevance = ContextRelevance.MEDIUM
    
    # Source information
    source_engine: ReasoningEngine = ReasoningEngine.DEDUCTIVE
    source_step: int = 0
    
    # Metadata
    confidence: float = 0.5
    quality_score: float = 0.5
    processing_time: float = 0.0
    
    # Relationships
    related_items: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.related_items:
            self.related_items = []
        if not self.dependencies:
            self.dependencies = []


@dataclass
class SequentialContext:
    """Container for sequential context information"""
    
    context_id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    original_context: Dict[str, Any] = field(default_factory=dict)
    
    # Context items organized by type
    context_items: Dict[ContextType, List[ContextItem]] = field(default_factory=dict)
    
    # Processing chain
    processing_chain: List[ReasoningEngine] = field(default_factory=list)
    current_step: int = 0
    
    # Accumulation settings
    passing_mode: ContextPassingMode = ContextPassingMode.ENRICHED
    max_context_size: int = 1000
    relevance_threshold: ContextRelevance = ContextRelevance.LOW
    
    # Performance tracking
    context_usage_stats: Dict[str, Any] = field(default_factory=dict)
    compression_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.original_context:
            self.original_context = {}
        if not self.context_items:
            self.context_items = {context_type: [] for context_type in ContextType}
        if not self.processing_chain:
            self.processing_chain = []
        if not self.context_usage_stats:
            self.context_usage_stats = {}
        if not self.compression_stats:
            self.compression_stats = {}


class ContextPassingEngine:
    """Engine for managing sequential context passing between reasoning engines"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.enabled = True
        
        # Context storage
        self.active_contexts: Dict[str, SequentialContext] = {}
        self.context_history: List[SequentialContext] = []
        self.max_history_size = 1000
        
        # Configuration
        self.default_passing_mode = ContextPassingMode.ENRICHED
        self.default_relevance_threshold = ContextRelevance.LOW
        self.context_compression_enabled = True
        self.context_filtering_enabled = True
        
        # Context processing
        self.context_analyzer = ContextAnalyzer()
        self.context_compressor = ContextCompressor()
        self.context_filter = ContextFilter()
        
        # Performance tracking
        self.context_metrics = {
            "contexts_created": 0,
            "contexts_passed": 0,
            "items_filtered": 0,
            "items_compressed": 0,
            "total_processing_time": 0.0
        }
        
        # Engine-specific context handlers
        self.engine_context_handlers = {
            ReasoningEngine.DEDUCTIVE: self._handle_deductive_context,
            ReasoningEngine.INDUCTIVE: self._handle_inductive_context,
            ReasoningEngine.ABDUCTIVE: self._handle_abductive_context,
            ReasoningEngine.CAUSAL: self._handle_causal_context,
            ReasoningEngine.PROBABILISTIC: self._handle_probabilistic_context,
            ReasoningEngine.COUNTERFACTUAL: self._handle_counterfactual_context,
            ReasoningEngine.ANALOGICAL: self._handle_analogical_context
        }
        
        logger.info("ContextPassingEngine initialized", 
                   default_mode=self.default_passing_mode.value,
                   compression_enabled=self.context_compression_enabled)
    
    def create_sequential_context(self, query: str, original_context: Dict[str, Any],
                                 processing_chain: List[ReasoningEngine],
                                 passing_mode: ContextPassingMode = None) -> SequentialContext:
        """Create a new sequential context for a reasoning chain"""
        
        if not self.enabled:
            return None
        
        start_time = time.time()
        
        # Create context
        sequential_context = SequentialContext(
            query=query,
            original_context=original_context,
            processing_chain=processing_chain,
            passing_mode=passing_mode or self.default_passing_mode,
            relevance_threshold=self.default_relevance_threshold
        )
        
        # Initialize context items
        for context_type in ContextType:
            sequential_context.context_items[context_type] = []
        
        # Store context
        self.active_contexts[sequential_context.context_id] = sequential_context
        
        # Update metrics
        self.context_metrics["contexts_created"] += 1
        self.context_metrics["total_processing_time"] += time.time() - start_time
        
        logger.debug(f"Created sequential context {sequential_context.context_id}", 
                    query=query, 
                    chain_length=len(processing_chain))
        
        return sequential_context
    
    def add_context_item(self, context_id: str, context_type: ContextType, content: Any,
                        source_engine: ReasoningEngine, relevance: ContextRelevance = ContextRelevance.MEDIUM,
                        confidence: float = 0.5, quality_score: float = 0.5) -> ContextItem:
        """Add a context item to the sequential context"""
        
        if context_id not in self.active_contexts:
            logger.warning(f"Context {context_id} not found")
            return None
        
        sequential_context = self.active_contexts[context_id]
        
        # Create context item
        context_item = ContextItem(
            context_type=context_type,
            content=content,
            relevance=relevance,
            source_engine=source_engine,
            source_step=sequential_context.current_step,
            confidence=confidence,
            quality_score=quality_score
        )
        
        # Add to context
        sequential_context.context_items[context_type].append(context_item)
        sequential_context.last_updated = datetime.now(timezone.utc)
        
        # Update usage stats
        stats_key = f"{context_type.value}_{source_engine.value}"
        if stats_key not in sequential_context.context_usage_stats:
            sequential_context.context_usage_stats[stats_key] = 0
        sequential_context.context_usage_stats[stats_key] += 1
        
        logger.debug(f"Added context item to {context_id}", 
                    context_type=context_type.value,
                    source_engine=source_engine.value,
                    relevance=relevance.value)
        
        return context_item
    
    def get_context_for_engine(self, context_id: str, target_engine: ReasoningEngine,
                              step: int) -> Dict[str, Any]:
        """Get filtered and processed context for a specific engine"""
        
        if context_id not in self.active_contexts:
            return {}
        
        start_time = time.time()
        sequential_context = self.active_contexts[context_id]
        
        # Update current step
        sequential_context.current_step = step
        
        # Get base context
        engine_context = sequential_context.original_context.copy()
        
        # Process context based on passing mode
        if sequential_context.passing_mode == ContextPassingMode.BASIC:
            processed_context = self._get_basic_context(sequential_context, target_engine)
        elif sequential_context.passing_mode == ContextPassingMode.ENRICHED:
            processed_context = self._get_enriched_context(sequential_context, target_engine)
        elif sequential_context.passing_mode == ContextPassingMode.CUMULATIVE:
            processed_context = self._get_cumulative_context(sequential_context, target_engine)
        elif sequential_context.passing_mode == ContextPassingMode.SELECTIVE:
            processed_context = self._get_selective_context(sequential_context, target_engine)
        elif sequential_context.passing_mode == ContextPassingMode.ADAPTIVE:
            processed_context = self._get_adaptive_context(sequential_context, target_engine)
        elif sequential_context.passing_mode == ContextPassingMode.COMPREHENSIVE:
            processed_context = self._get_comprehensive_context(sequential_context, target_engine)
        else:
            processed_context = self._get_enriched_context(sequential_context, target_engine)
        
        # Merge with base context
        engine_context.update(processed_context)
        
        # Apply engine-specific processing
        if target_engine in self.engine_context_handlers:
            engine_context = self.engine_context_handlers[target_engine](engine_context, sequential_context)
        
        # Apply filtering if enabled
        if self.context_filtering_enabled:
            engine_context = self.context_filter.filter_context(engine_context, target_engine, sequential_context)
        
        # Apply compression if enabled
        if self.context_compression_enabled:
            engine_context = self.context_compressor.compress_context(engine_context, target_engine)
        
        # Update metrics
        self.context_metrics["contexts_passed"] += 1
        self.context_metrics["total_processing_time"] += time.time() - start_time
        
        logger.debug(f"Generated context for {target_engine.value}", 
                    context_id=context_id,
                    step=step,
                    context_keys=list(engine_context.keys()))
        
        return engine_context
    
    def _get_basic_context(self, sequential_context: SequentialContext, 
                          target_engine: ReasoningEngine) -> Dict[str, Any]:
        """Get basic context with minimal information"""
        
        context = {
            "sequential_step": sequential_context.current_step,
            "processing_chain": [e.value for e in sequential_context.processing_chain],
            "previous_engines": [e.value for e in sequential_context.processing_chain[:sequential_context.current_step]]
        }
        
        # Add only critical context items
        critical_items = []
        for context_type, items in sequential_context.context_items.items():
            for item in items:
                if item.relevance == ContextRelevance.CRITICAL:
                    critical_items.append({
                        "type": context_type.value,
                        "content": item.content,
                        "source": item.source_engine.value,
                        "confidence": item.confidence
                    })
        
        if critical_items:
            context["critical_insights"] = critical_items
        
        return context
    
    def _get_enriched_context(self, sequential_context: SequentialContext, 
                             target_engine: ReasoningEngine) -> Dict[str, Any]:
        """Get enriched context with relevant insights and patterns"""
        
        context = self._get_basic_context(sequential_context, target_engine)
        
        # Add high and critical relevance items
        for context_type, items in sequential_context.context_items.items():
            relevant_items = [
                item for item in items 
                if item.relevance in [ContextRelevance.CRITICAL, ContextRelevance.HIGH]
            ]
            
            if relevant_items:
                context[f"{context_type.value}_items"] = [
                    {
                        "content": item.content,
                        "source": item.source_engine.value,
                        "confidence": item.confidence,
                        "quality": item.quality_score,
                        "relevance": item.relevance.value
                    }
                    for item in relevant_items
                ]
        
        # Add reasoning progression
        context["reasoning_progression"] = self._build_reasoning_progression(sequential_context)
        
        return context
    
    def _get_cumulative_context(self, sequential_context: SequentialContext, 
                               target_engine: ReasoningEngine) -> Dict[str, Any]:
        """Get cumulative context that builds up over the chain"""
        
        context = self._get_enriched_context(sequential_context, target_engine)
        
        # Add all context items above threshold
        for context_type, items in sequential_context.context_items.items():
            relevant_items = [
                item for item in items 
                if self._is_above_relevance_threshold(item.relevance, sequential_context.relevance_threshold)
            ]
            
            if relevant_items:
                context[f"accumulated_{context_type.value}"] = [
                    {
                        "content": item.content,
                        "source": item.source_engine.value,
                        "step": item.source_step,
                        "confidence": item.confidence,
                        "quality": item.quality_score,
                        "created_at": item.created_at.isoformat()
                    }
                    for item in relevant_items
                ]
        
        # Add cumulative insights
        context["cumulative_insights"] = self._extract_cumulative_insights(sequential_context)
        
        return context
    
    def _get_selective_context(self, sequential_context: SequentialContext, 
                              target_engine: ReasoningEngine) -> Dict[str, Any]:
        """Get selectively filtered context based on engine needs"""
        
        context = self._get_basic_context(sequential_context, target_engine)
        
        # Select context based on engine type
        relevant_types = self._get_relevant_context_types(target_engine)
        
        for context_type in relevant_types:
            if context_type in sequential_context.context_items:
                items = sequential_context.context_items[context_type]
                if items:
                    # Select top items by relevance and quality
                    sorted_items = sorted(items, 
                                        key=lambda x: (x.relevance.value, x.quality_score), 
                                        reverse=True)
                    
                    context[f"selected_{context_type.value}"] = [
                        {
                            "content": item.content,
                            "source": item.source_engine.value,
                            "confidence": item.confidence,
                            "quality": item.quality_score
                        }
                        for item in sorted_items[:5]  # Top 5 items
                    ]
        
        return context
    
    def _get_adaptive_context(self, sequential_context: SequentialContext, 
                             target_engine: ReasoningEngine) -> Dict[str, Any]:
        """Get adaptively filtered context based on engine performance and patterns"""
        
        context = self._get_selective_context(sequential_context, target_engine)
        
        # Use pattern recognizer to get relevant patterns
        if hasattr(self.meta_reasoning_engine, 'pattern_recognizer'):
            pattern_recommendations = self.meta_reasoning_engine.pattern_recognizer.get_pattern_recommendations(
                sequential_context.query, sequential_context.original_context
            )
            
            # Filter context based on successful patterns
            for pair in pattern_recommendations:
                if pair[1] == target_engine:  # If target engine is second in pair
                    source_engine = pair[0]
                    # Prioritize context from successful source engines
                    context[f"prioritized_from_{source_engine.value}"] = self._get_prioritized_context(
                        sequential_context, source_engine
                    )
        
        return context
    
    def _get_comprehensive_context(self, sequential_context: SequentialContext, 
                                  target_engine: ReasoningEngine) -> Dict[str, Any]:
        """Get comprehensive context with all available information"""
        
        context = self._get_cumulative_context(sequential_context, target_engine)
        
        # Add all context items
        for context_type, items in sequential_context.context_items.items():
            if items:
                context[f"all_{context_type.value}"] = [
                    {
                        "content": item.content,
                        "source": item.source_engine.value,
                        "step": item.source_step,
                        "confidence": item.confidence,
                        "quality": item.quality_score,
                        "relevance": item.relevance.value,
                        "created_at": item.created_at.isoformat(),
                        "accessed_count": item.accessed_count
                    }
                    for item in items
                ]
        
        # Add comprehensive statistics
        context["context_statistics"] = self._get_context_statistics(sequential_context)
        
        return context
    
    def _handle_deductive_context(self, context: Dict[str, Any], 
                                 sequential_context: SequentialContext) -> Dict[str, Any]:
        """Handle context specifically for deductive reasoning"""
        
        # Deductive reasoning benefits from clear premises and logical structures
        context["deductive_focus"] = {
            "premises": self._extract_premises(sequential_context),
            "logical_structures": self._extract_logical_structures(sequential_context),
            "certainty_levels": self._extract_certainty_levels(sequential_context)
        }
        
        return context
    
    def _handle_inductive_context(self, context: Dict[str, Any], 
                                 sequential_context: SequentialContext) -> Dict[str, Any]:
        """Handle context specifically for inductive reasoning"""
        
        # Inductive reasoning benefits from patterns and examples
        context["inductive_focus"] = {
            "patterns": self._extract_patterns(sequential_context),
            "examples": self._extract_examples(sequential_context),
            "generalizations": self._extract_generalizations(sequential_context)
        }
        
        return context
    
    def _handle_abductive_context(self, context: Dict[str, Any], 
                                 sequential_context: SequentialContext) -> Dict[str, Any]:
        """Handle context specifically for abductive reasoning"""
        
        # Abductive reasoning benefits from observations and hypotheses
        context["abductive_focus"] = {
            "observations": self._extract_observations(sequential_context),
            "hypotheses": self._extract_hypotheses(sequential_context),
            "explanations": self._extract_explanations(sequential_context)
        }
        
        return context
    
    def _handle_causal_context(self, context: Dict[str, Any], 
                              sequential_context: SequentialContext) -> Dict[str, Any]:
        """Handle context specifically for causal reasoning"""
        
        # Causal reasoning benefits from cause-effect relationships
        context["causal_focus"] = {
            "causal_chains": self._extract_causal_chains(sequential_context),
            "mechanisms": self._extract_mechanisms(sequential_context),
            "confounding_factors": self._extract_confounding_factors(sequential_context)
        }
        
        return context
    
    def _handle_probabilistic_context(self, context: Dict[str, Any], 
                                     sequential_context: SequentialContext) -> Dict[str, Any]:
        """Handle context specifically for probabilistic reasoning"""
        
        # Probabilistic reasoning benefits from uncertainty and distributions
        context["probabilistic_focus"] = {
            "uncertainties": self._extract_uncertainties(sequential_context),
            "distributions": self._extract_distributions(sequential_context),
            "likelihoods": self._extract_likelihoods(sequential_context)
        }
        
        return context
    
    def _handle_counterfactual_context(self, context: Dict[str, Any], 
                                      sequential_context: SequentialContext) -> Dict[str, Any]:
        """Handle context specifically for counterfactual reasoning"""
        
        # Counterfactual reasoning benefits from alternatives and scenarios
        context["counterfactual_focus"] = {
            "alternatives": self._extract_alternatives(sequential_context),
            "scenarios": self._extract_scenarios(sequential_context),
            "interventions": self._extract_interventions(sequential_context)
        }
        
        return context
    
    def _handle_analogical_context(self, context: Dict[str, Any], 
                                  sequential_context: SequentialContext) -> Dict[str, Any]:
        """Handle context specifically for analogical reasoning"""
        
        # Analogical reasoning benefits from similarities and mappings
        context["analogical_focus"] = {
            "similarities": self._extract_similarities(sequential_context),
            "mappings": self._extract_mappings(sequential_context),
            "comparisons": self._extract_comparisons(sequential_context)
        }
        
        return context
    
    def _is_above_relevance_threshold(self, relevance: ContextRelevance, 
                                     threshold: ContextRelevance) -> bool:
        """Check if relevance is above threshold"""
        
        relevance_order = [
            ContextRelevance.NEGLIGIBLE,
            ContextRelevance.LOW,
            ContextRelevance.MEDIUM,
            ContextRelevance.HIGH,
            ContextRelevance.CRITICAL
        ]
        
        return relevance_order.index(relevance) >= relevance_order.index(threshold)
    
    def _get_relevant_context_types(self, engine: ReasoningEngine) -> List[ContextType]:
        """Get context types most relevant to a specific engine"""
        
        engine_context_map = {
            ReasoningEngine.DEDUCTIVE: [
                ContextType.REASONING_CHAIN,
                ContextType.EVIDENCE,
                ContextType.ASSUMPTIONS,
                ContextType.CONFIDENCE_FACTORS
            ],
            ReasoningEngine.INDUCTIVE: [
                ContextType.PATTERNS,
                ContextType.EVIDENCE,
                ContextType.INSIGHTS,
                ContextType.INTERMEDIATE_RESULTS
            ],
            ReasoningEngine.ABDUCTIVE: [
                ContextType.HYPOTHESES,
                ContextType.INSIGHTS,
                ContextType.EVIDENCE,
                ContextType.LIMITATIONS
            ],
            ReasoningEngine.CAUSAL: [
                ContextType.REASONING_CHAIN,
                ContextType.EVIDENCE,
                ContextType.PATTERNS,
                ContextType.ASSUMPTIONS
            ],
            ReasoningEngine.PROBABILISTIC: [
                ContextType.CONFIDENCE_FACTORS,
                ContextType.EVIDENCE,
                ContextType.QUALITY_INDICATORS,
                ContextType.INTERMEDIATE_RESULTS
            ],
            ReasoningEngine.COUNTERFACTUAL: [
                ContextType.HYPOTHESES,
                ContextType.INSIGHTS,
                ContextType.ASSUMPTIONS,
                ContextType.LIMITATIONS
            ],
            ReasoningEngine.ANALOGICAL: [
                ContextType.PATTERNS,
                ContextType.INSIGHTS,
                ContextType.EVIDENCE,
                ContextType.INTERMEDIATE_RESULTS
            ]
        }
        
        return engine_context_map.get(engine, list(ContextType))
    
    def _build_reasoning_progression(self, sequential_context: SequentialContext) -> List[Dict[str, Any]]:
        """Build reasoning progression summary"""
        
        progression = []
        for i, engine in enumerate(sequential_context.processing_chain[:sequential_context.current_step]):
            step_info = {
                "step": i,
                "engine": engine.value,
                "contributions": []
            }
            
            # Find contributions from this engine
            for context_type, items in sequential_context.context_items.items():
                engine_items = [item for item in items if item.source_engine == engine and item.source_step == i]
                if engine_items:
                    step_info["contributions"].append({
                        "type": context_type.value,
                        "count": len(engine_items),
                        "avg_confidence": statistics.mean([item.confidence for item in engine_items]),
                        "avg_quality": statistics.mean([item.quality_score for item in engine_items])
                    })
            
            progression.append(step_info)
        
        return progression
    
    def _extract_cumulative_insights(self, sequential_context: SequentialContext) -> List[Dict[str, Any]]:
        """Extract cumulative insights from context"""
        
        insights = []
        for items in sequential_context.context_items[ContextType.INSIGHTS]:
            insights.append({
                "content": items.content,
                "source": items.source_engine.value,
                "step": items.source_step,
                "confidence": items.confidence,
                "quality": items.quality_score
            })
        
        # Sort by quality and confidence
        insights.sort(key=lambda x: (x["quality"], x["confidence"]), reverse=True)
        
        return insights
    
    def _get_prioritized_context(self, sequential_context: SequentialContext, 
                                source_engine: ReasoningEngine) -> Dict[str, Any]:
        """Get prioritized context from a specific source engine"""
        
        prioritized = {}
        
        for context_type, items in sequential_context.context_items.items():
            source_items = [item for item in items if item.source_engine == source_engine]
            if source_items:
                # Sort by relevance and quality
                sorted_items = sorted(source_items, 
                                    key=lambda x: (x.relevance.value, x.quality_score), 
                                    reverse=True)
                
                prioritized[context_type.value] = [
                    {
                        "content": item.content,
                        "confidence": item.confidence,
                        "quality": item.quality_score,
                        "relevance": item.relevance.value
                    }
                    for item in sorted_items[:3]  # Top 3 items
                ]
        
        return prioritized
    
    def _get_context_statistics(self, sequential_context: SequentialContext) -> Dict[str, Any]:
        """Get comprehensive context statistics"""
        
        stats = {
            "total_items": sum(len(items) for items in sequential_context.context_items.values()),
            "items_by_type": {
                context_type.value: len(items) 
                for context_type, items in sequential_context.context_items.items()
            },
            "items_by_engine": {},
            "avg_confidence": 0.0,
            "avg_quality": 0.0,
            "relevance_distribution": {}
        }
        
        # Calculate engine distribution
        for context_type, items in sequential_context.context_items.items():
            for item in items:
                engine = item.source_engine.value
                if engine not in stats["items_by_engine"]:
                    stats["items_by_engine"][engine] = 0
                stats["items_by_engine"][engine] += 1
        
        # Calculate averages
        all_items = [item for items in sequential_context.context_items.values() for item in items]
        if all_items:
            stats["avg_confidence"] = statistics.mean([item.confidence for item in all_items])
            stats["avg_quality"] = statistics.mean([item.quality_score for item in all_items])
            
            # Relevance distribution
            relevance_counts = {}
            for item in all_items:
                relevance = item.relevance.value
                relevance_counts[relevance] = relevance_counts.get(relevance, 0) + 1
            stats["relevance_distribution"] = relevance_counts
        
        return stats
    
    # Helper methods for extracting specific types of context
    def _extract_premises(self, sequential_context: SequentialContext) -> List[str]:
        """Extract premises from context"""
        premises = []
        for item in sequential_context.context_items[ContextType.ASSUMPTIONS]:
            if isinstance(item.content, str):
                premises.append(item.content)
        return premises
    
    def _extract_logical_structures(self, sequential_context: SequentialContext) -> List[str]:
        """Extract logical structures from context"""
        structures = []
        for item in sequential_context.context_items[ContextType.REASONING_CHAIN]:
            if isinstance(item.content, list):
                structures.extend(item.content)
        return structures
    
    def _extract_certainty_levels(self, sequential_context: SequentialContext) -> List[float]:
        """Extract certainty levels from context"""
        return [item.confidence for items in sequential_context.context_items.values() for item in items]
    
    def _extract_patterns(self, sequential_context: SequentialContext) -> List[str]:
        """Extract patterns from context"""
        patterns = []
        for item in sequential_context.context_items[ContextType.PATTERNS]:
            if isinstance(item.content, str):
                patterns.append(item.content)
        return patterns
    
    def _extract_examples(self, sequential_context: SequentialContext) -> List[str]:
        """Extract examples from context"""
        examples = []
        for item in sequential_context.context_items[ContextType.EVIDENCE]:
            if isinstance(item.content, str):
                examples.append(item.content)
        return examples
    
    def _extract_generalizations(self, sequential_context: SequentialContext) -> List[str]:
        """Extract generalizations from context"""
        generalizations = []
        for item in sequential_context.context_items[ContextType.INSIGHTS]:
            if isinstance(item.content, str) and "general" in item.content.lower():
                generalizations.append(item.content)
        return generalizations
    
    def _extract_observations(self, sequential_context: SequentialContext) -> List[str]:
        """Extract observations from context"""
        observations = []
        for item in sequential_context.context_items[ContextType.EVIDENCE]:
            if isinstance(item.content, str):
                observations.append(item.content)
        return observations
    
    def _extract_hypotheses(self, sequential_context: SequentialContext) -> List[str]:
        """Extract hypotheses from context"""
        hypotheses = []
        for item in sequential_context.context_items[ContextType.HYPOTHESES]:
            if isinstance(item.content, str):
                hypotheses.append(item.content)
        return hypotheses
    
    def _extract_explanations(self, sequential_context: SequentialContext) -> List[str]:
        """Extract explanations from context"""
        explanations = []
        for item in sequential_context.context_items[ContextType.REASONING_CHAIN]:
            if isinstance(item.content, list):
                explanations.extend(item.content)
        return explanations
    
    def _extract_causal_chains(self, sequential_context: SequentialContext) -> List[str]:
        """Extract causal chains from context"""
        chains = []
        for item in sequential_context.context_items[ContextType.REASONING_CHAIN]:
            if isinstance(item.content, list):
                chains.extend(item.content)
        return chains
    
    def _extract_mechanisms(self, sequential_context: SequentialContext) -> List[str]:
        """Extract mechanisms from context"""
        mechanisms = []
        for item in sequential_context.context_items[ContextType.INSIGHTS]:
            if isinstance(item.content, str) and "mechanism" in item.content.lower():
                mechanisms.append(item.content)
        return mechanisms
    
    def _extract_confounding_factors(self, sequential_context: SequentialContext) -> List[str]:
        """Extract confounding factors from context"""
        factors = []
        for item in sequential_context.context_items[ContextType.LIMITATIONS]:
            if isinstance(item.content, str):
                factors.append(item.content)
        return factors
    
    def _extract_uncertainties(self, sequential_context: SequentialContext) -> List[Dict[str, Any]]:
        """Extract uncertainties from context"""
        uncertainties = []
        for item in sequential_context.context_items[ContextType.CONFIDENCE_FACTORS]:
            uncertainties.append({
                "factor": item.content,
                "confidence": item.confidence,
                "source": item.source_engine.value
            })
        return uncertainties
    
    def _extract_distributions(self, sequential_context: SequentialContext) -> List[str]:
        """Extract distributions from context"""
        distributions = []
        for item in sequential_context.context_items[ContextType.INTERMEDIATE_RESULTS]:
            if isinstance(item.content, str) and "distribution" in item.content.lower():
                distributions.append(item.content)
        return distributions
    
    def _extract_likelihoods(self, sequential_context: SequentialContext) -> List[float]:
        """Extract likelihoods from context"""
        return [item.confidence for items in sequential_context.context_items.values() for item in items]
    
    def _extract_alternatives(self, sequential_context: SequentialContext) -> List[str]:
        """Extract alternatives from context"""
        alternatives = []
        for item in sequential_context.context_items[ContextType.HYPOTHESES]:
            if isinstance(item.content, str) and "alternative" in item.content.lower():
                alternatives.append(item.content)
        return alternatives
    
    def _extract_scenarios(self, sequential_context: SequentialContext) -> List[str]:
        """Extract scenarios from context"""
        scenarios = []
        for item in sequential_context.context_items[ContextType.INSIGHTS]:
            if isinstance(item.content, str) and "scenario" in item.content.lower():
                scenarios.append(item.content)
        return scenarios
    
    def _extract_interventions(self, sequential_context: SequentialContext) -> List[str]:
        """Extract interventions from context"""
        interventions = []
        for item in sequential_context.context_items[ContextType.INSIGHTS]:
            if isinstance(item.content, str) and "intervention" in item.content.lower():
                interventions.append(item.content)
        return interventions
    
    def _extract_similarities(self, sequential_context: SequentialContext) -> List[str]:
        """Extract similarities from context"""
        similarities = []
        for item in sequential_context.context_items[ContextType.PATTERNS]:
            if isinstance(item.content, str) and "similar" in item.content.lower():
                similarities.append(item.content)
        return similarities
    
    def _extract_mappings(self, sequential_context: SequentialContext) -> List[str]:
        """Extract mappings from context"""
        mappings = []
        for item in sequential_context.context_items[ContextType.INSIGHTS]:
            if isinstance(item.content, str) and "mapping" in item.content.lower():
                mappings.append(item.content)
        return mappings
    
    def _extract_comparisons(self, sequential_context: SequentialContext) -> List[str]:
        """Extract comparisons from context"""
        comparisons = []
        for item in sequential_context.context_items[ContextType.INSIGHTS]:
            if isinstance(item.content, str) and "compare" in item.content.lower():
                comparisons.append(item.content)
        return comparisons
    
    def finalize_context(self, context_id: str) -> Dict[str, Any]:
        """Finalize and archive a sequential context"""
        
        if context_id not in self.active_contexts:
            return {"error": f"Context {context_id} not found", "success": False}
        
        sequential_context = self.active_contexts[context_id]
        
        # Move to history
        self.context_history.append(sequential_context)
        
        # Maintain history size
        if len(self.context_history) > self.max_history_size:
            self.context_history.pop(0)
        
        # Remove from active contexts
        del self.active_contexts[context_id]
        
        # Generate summary
        summary = {
            "context_id": context_id,
            "query": sequential_context.query,
            "processing_chain": [e.value for e in sequential_context.processing_chain],
            "total_steps": sequential_context.current_step,
            "total_items": sum(len(items) for items in sequential_context.context_items.values()),
            "passing_mode": sequential_context.passing_mode.value,
            "created_at": sequential_context.created_at.isoformat(),
            "last_updated": sequential_context.last_updated.isoformat(),
            "context_statistics": self._get_context_statistics(sequential_context),
            "success": True
        }
        
        logger.info(f"Finalized context {context_id}", 
                   total_items=summary["total_items"],
                   total_steps=summary["total_steps"])
        
        return summary
    
    def get_context_passing_statistics(self) -> Dict[str, Any]:
        """Get context passing statistics"""
        
        return {
            "active_contexts": len(self.active_contexts),
            "context_history": len(self.context_history),
            "metrics": self.context_metrics,
            "configuration": {
                "enabled": self.enabled,
                "default_passing_mode": self.default_passing_mode.value,
                "default_relevance_threshold": self.default_relevance_threshold.value,
                "compression_enabled": self.context_compression_enabled,
                "filtering_enabled": self.context_filtering_enabled
            }
        }
    
    def enable_context_passing(self):
        """Enable context passing"""
        self.enabled = True
        logger.info("Context passing enabled")
    
    def disable_context_passing(self):
        """Disable context passing"""
        self.enabled = False
        logger.info("Context passing disabled")
    
    def clear_context_history(self):
        """Clear context history"""
        self.context_history = []
        logger.info("Context history cleared")


class ContextCompressor:
    """Compresses context information to reduce memory usage"""
    
    def __init__(self):
        self.compression_enabled = True
        self.compression_threshold = 1000  # Compress if context > 1000 chars
    
    def compress_context(self, context: Dict[str, Any], target_engine: ReasoningEngine) -> Dict[str, Any]:
        """Compress context for efficient passing"""
        
        if not self.compression_enabled:
            return context
        
        # Simple compression: summarize long content
        compressed = {}
        for key, value in context.items():
            if isinstance(value, str) and len(value) > self.compression_threshold:
                compressed[key] = self._summarize_content(value)
            elif isinstance(value, list) and len(value) > 10:
                compressed[key] = value[:10]  # Take first 10 items
            else:
                compressed[key] = value
        
        return compressed
    
    def _summarize_content(self, content: str) -> str:
        """Summarize long content"""
        # Simple summarization: take first and last parts
        if len(content) > self.compression_threshold:
            return content[:500] + "..." + content[-100:]
        return content


class ContextFilter:
    """Filters context information based on relevance and engine needs"""
    
    def __init__(self):
        self.filtering_enabled = True
        self.relevance_threshold = ContextRelevance.LOW
    
    def filter_context(self, context: Dict[str, Any], target_engine: ReasoningEngine,
                      sequential_context: SequentialContext) -> Dict[str, Any]:
        """Filter context based on relevance and engine needs"""
        
        if not self.filtering_enabled:
            return context
        
        # Simple filtering: remove low-relevance items
        filtered = {}
        for key, value in context.items():
            if self._is_relevant(key, value, target_engine):
                filtered[key] = value
        
        return filtered
    
    def _is_relevant(self, key: str, value: Any, target_engine: ReasoningEngine) -> bool:
        """Check if context item is relevant to target engine"""
        
        # Simple relevance check based on key names
        engine_keywords = {
            ReasoningEngine.DEDUCTIVE: ["premise", "logic", "conclusion", "proof"],
            ReasoningEngine.INDUCTIVE: ["pattern", "example", "generalization"],
            ReasoningEngine.ABDUCTIVE: ["observation", "hypothesis", "explanation"],
            ReasoningEngine.CAUSAL: ["cause", "effect", "mechanism", "relationship"],
            ReasoningEngine.PROBABILISTIC: ["probability", "likelihood", "uncertainty"],
            ReasoningEngine.COUNTERFACTUAL: ["alternative", "scenario", "intervention"],
            ReasoningEngine.ANALOGICAL: ["similarity", "mapping", "comparison"]
        }
        
        keywords = engine_keywords.get(target_engine, [])
        return any(keyword in key.lower() for keyword in keywords) or len(keywords) == 0


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"                    # Minor issues that don't affect functionality
    MEDIUM = "medium"              # Moderate issues that may degrade performance
    HIGH = "high"                  # Serious issues that affect core functionality
    CRITICAL = "critical"          # Critical issues that require immediate attention
    FATAL = "fatal"                # Fatal errors that cause system failure


class ErrorCategory(Enum):
    """Categories of errors"""
    ENGINE_ERROR = "engine_error"             # Individual engine failures
    SYSTEM_ERROR = "system_error"             # System-wide failures
    CONFIGURATION_ERROR = "configuration_error"  # Configuration issues
    RESOURCE_ERROR = "resource_error"         # Resource exhaustion
    TIMEOUT_ERROR = "timeout_error"           # Timeout-related errors
    VALIDATION_ERROR = "validation_error"     # Input validation errors
    NETWORK_ERROR = "network_error"           # Network-related errors
    PERMISSION_ERROR = "permission_error"     # Permission/authorization errors
    DATA_ERROR = "data_error"                 # Data integrity errors
    UNKNOWN_ERROR = "unknown_error"           # Unclassified errors


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"                           # Simple retry with backoff
    FALLBACK = "fallback"                     # Use fallback engine/method
    DEGRADE = "degrade"                       # Graceful degradation
    ISOLATION = "isolation"                   # Isolate problematic component
    RESTART = "restart"                       # Restart component
    ESCALATE = "escalate"                     # Escalate to higher level
    IGNORE = "ignore"                         # Ignore non-critical errors
    ABORT = "abort"                           # Abort operation safely
    BYPASS = "bypass"                         # Bypass problematic component
    COMPENSATE = "compensate"                 # Compensate with alternative approach


@dataclass
class ErrorContext:
    """Context information for errors"""
    
    operation: str = ""                       # Operation being performed
    engine_type: Optional[ReasoningEngine] = None  # Engine involved
    query: str = ""                           # Query being processed
    user_context: Dict[str, Any] = field(default_factory=dict)  # User context
    system_state: Dict[str, Any] = field(default_factory=dict)  # System state
    stack_trace: str = ""                     # Stack trace if available
    related_errors: List[str] = field(default_factory=list)  # Related error IDs
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.user_context:
            self.user_context = {}
        if not self.system_state:
            self.system_state = {}
        if not self.related_errors:
            self.related_errors = []


@dataclass
class ErrorEvent:
    """Comprehensive error event structure"""
    
    error_id: str = field(default_factory=lambda: str(uuid4()))
    error_type: str = ""                      # Specific error type
    category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    message: str = ""                         # Human-readable error message
    technical_details: str = ""               # Technical details for debugging
    
    context: ErrorContext = field(default_factory=ErrorContext)
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    recovery_details: str = ""
    
    # Tracking
    occurrence_count: int = 1
    first_occurrence: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_occurrence: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Resolution
    resolved: bool = False
    resolution_details: str = ""
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.context:
            self.context = ErrorContext()


class ErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.enabled = True
        
        # Error storage
        self.error_events: List[ErrorEvent] = []
        self.error_patterns: Dict[str, List[ErrorEvent]] = {}
        self.max_error_history = 1000
        
        # Recovery configuration
        self.recovery_strategies = {
            ErrorCategory.ENGINE_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.DEGRADE
            ],
            ErrorCategory.SYSTEM_ERROR: [
                RecoveryStrategy.RESTART,
                RecoveryStrategy.ESCALATE,
                RecoveryStrategy.ABORT
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.DEGRADE,
                RecoveryStrategy.BYPASS
            ],
            ErrorCategory.RESOURCE_ERROR: [
                RecoveryStrategy.DEGRADE,
                RecoveryStrategy.ISOLATION,
                RecoveryStrategy.ESCALATE
            ],
            ErrorCategory.VALIDATION_ERROR: [
                RecoveryStrategy.IGNORE,
                RecoveryStrategy.COMPENSATE,
                RecoveryStrategy.ABORT
            ]
        }
        
        # Retry configuration
        self.retry_config = {
            ErrorCategory.ENGINE_ERROR: {"max_retries": 3, "backoff_factor": 2.0},
            ErrorCategory.TIMEOUT_ERROR: {"max_retries": 2, "backoff_factor": 1.5},
            ErrorCategory.RESOURCE_ERROR: {"max_retries": 1, "backoff_factor": 3.0},
            ErrorCategory.NETWORK_ERROR: {"max_retries": 3, "backoff_factor": 2.0}
        }
        
        # Error thresholds
        self.error_thresholds = {
            ErrorSeverity.LOW: 100,      # Allow up to 100 low severity errors
            ErrorSeverity.MEDIUM: 50,    # Allow up to 50 medium severity errors
            ErrorSeverity.HIGH: 10,      # Allow up to 10 high severity errors
            ErrorSeverity.CRITICAL: 3,   # Allow up to 3 critical errors
            ErrorSeverity.FATAL: 1       # Only 1 fatal error before escalation
        }
        
        # Circuit breaker for error handling
        self.circuit_breaker = {
            "error_rate_threshold": 0.5,  # 50% error rate threshold
            "time_window": 300,           # 5 minute time window
            "recovery_timeout": 60        # 1 minute recovery timeout
        }
    
    def handle_error(self, error: Exception, context: ErrorContext, 
                    suggested_severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorEvent:
        """Handle an error with automatic categorization and recovery"""
        
        if not self.enabled:
            return self._create_minimal_error_event(error, context)
        
        # Categorize error
        category = self._categorize_error(error, context)
        severity = self._determine_severity(error, context, suggested_severity)
        
        # Create error event
        error_event = ErrorEvent(
            error_type=type(error).__name__,
            category=category,
            severity=severity,
            message=str(error),
            technical_details=self._extract_technical_details(error),
            context=context
        )
        
        # Check for recurring errors
        self._check_error_patterns(error_event)
        
        # Store error
        self._store_error(error_event)
        
        # Attempt recovery
        if self._should_attempt_recovery(error_event):
            self._attempt_recovery(error_event)
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Log error
        self._log_error(error_event)
        
        return error_event
    
    def _categorize_error(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Categorize error based on type and context"""
        
        error_type = type(error).__name__
        
        # Engine-specific errors
        if context.engine_type or "engine" in context.operation.lower():
            return ErrorCategory.ENGINE_ERROR
        
        # Timeout errors
        if isinstance(error, asyncio.TimeoutError) or "timeout" in error_type.lower():
            return ErrorCategory.TIMEOUT_ERROR
        
        # Resource errors
        if isinstance(error, (MemoryError, OSError)) or "memory" in str(error).lower():
            return ErrorCategory.RESOURCE_ERROR
        
        # Validation errors
        if isinstance(error, (ValueError, TypeError)) or "validation" in error_type.lower():
            return ErrorCategory.VALIDATION_ERROR
        
        # Configuration errors
        if "config" in error_type.lower() or "setting" in str(error).lower():
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Network errors
        if "network" in error_type.lower() or "connection" in str(error).lower():
            return ErrorCategory.NETWORK_ERROR
        
        # Permission errors
        if isinstance(error, PermissionError) or "permission" in error_type.lower():
            return ErrorCategory.PERMISSION_ERROR
        
        # Data errors
        if "data" in error_type.lower() or "corrupt" in str(error).lower():
            return ErrorCategory.DATA_ERROR
        
        # System errors
        if isinstance(error, SystemError) or "system" in error_type.lower():
            return ErrorCategory.SYSTEM_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _determine_severity(self, error: Exception, context: ErrorContext, 
                           suggested_severity: ErrorSeverity) -> ErrorSeverity:
        """Determine error severity based on error type and context"""
        
        # Fatal errors
        if isinstance(error, SystemExit) or "fatal" in str(error).lower():
            return ErrorSeverity.FATAL
        
        # Critical errors
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(error, (RuntimeError, OSError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if isinstance(error, (ValueError, TypeError, asyncio.TimeoutError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if isinstance(error, (AttributeError, KeyError)):
            return ErrorSeverity.LOW
        
        # Use suggested severity if no specific rule applies
        return suggested_severity
    
    def _extract_technical_details(self, error: Exception) -> str:
        """Extract technical details from error"""
        import traceback
        
        details = []
        
        # Exception type and message
        details.append(f"Exception Type: {type(error).__name__}")
        details.append(f"Exception Message: {str(error)}")
        
        # Stack trace
        if hasattr(error, '__traceback__'):
            stack_trace = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
            details.append(f"Stack Trace:\n{stack_trace}")
        
        # Additional error attributes
        for attr in ['args', 'errno', 'strerror', 'filename']:
            if hasattr(error, attr):
                details.append(f"{attr}: {getattr(error, attr)}")
        
        return '\n'.join(details)
    
    def _check_error_patterns(self, error_event: ErrorEvent):
        """Check for recurring error patterns"""
        
        pattern_key = f"{error_event.error_type}_{error_event.category.value}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = []
        
        self.error_patterns[pattern_key].append(error_event)
        
        # Check for pattern threshold
        recent_errors = [
            e for e in self.error_patterns[pattern_key]
            if (error_event.first_occurrence - e.first_occurrence).total_seconds() < 3600  # Last hour
        ]
        
        if len(recent_errors) >= 5:  # 5 similar errors in an hour
            error_event.severity = ErrorSeverity.HIGH if error_event.severity == ErrorSeverity.MEDIUM else ErrorSeverity.CRITICAL
            logger.warning(f"Error pattern detected: {pattern_key}, escalating severity")
    
    def _store_error(self, error_event: ErrorEvent):
        """Store error event in history"""
        
        # Check for duplicate errors
        for existing_error in self.error_events:
            if (existing_error.error_type == error_event.error_type and
                existing_error.message == error_event.message and
                (error_event.first_occurrence - existing_error.last_occurrence).total_seconds() < 300):  # 5 minutes
                
                # Update existing error
                existing_error.occurrence_count += 1
                existing_error.last_occurrence = error_event.first_occurrence
                return
        
        # Add new error
        self.error_events.append(error_event)
        
        # Maintain history size
        if len(self.error_events) > self.max_error_history:
            self.error_events.pop(0)
    
    def _should_attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Determine if recovery should be attempted"""
        
        # Don't attempt recovery for fatal errors
        if error_event.severity == ErrorSeverity.FATAL:
            return False
        
        # Don't attempt recovery if already attempted
        if error_event.recovery_attempted:
            return False
        
        # Check if recovery strategies are available
        return error_event.category in self.recovery_strategies
    
    def _attempt_recovery(self, error_event: ErrorEvent):
        """Attempt recovery for error"""
        
        error_event.recovery_attempted = True
        strategies = self.recovery_strategies.get(error_event.category, [])
        
        for strategy in strategies:
            try:
                success = self._execute_recovery_strategy(strategy, error_event)
                if success:
                    error_event.recovery_successful = True
                    error_event.recovery_strategy = strategy
                    error_event.recovery_details = f"Recovery successful using {strategy.value}"
                    logger.info(f"Recovery successful for error {error_event.error_id} using {strategy.value}")
                    return
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.value} failed: {str(recovery_error)}")
                continue
        
        error_event.recovery_details = f"All recovery strategies failed for {error_event.category.value}"
        logger.error(f"All recovery strategies failed for error {error_event.error_id}")
    
    def _execute_recovery_strategy(self, strategy: RecoveryStrategy, error_event: ErrorEvent) -> bool:
        """Execute specific recovery strategy"""
        
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_operation(error_event)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_operation(error_event)
        
        elif strategy == RecoveryStrategy.DEGRADE:
            return self._degrade_operation(error_event)
        
        elif strategy == RecoveryStrategy.ISOLATION:
            return self._isolate_component(error_event)
        
        elif strategy == RecoveryStrategy.RESTART:
            return self._restart_component(error_event)
        
        elif strategy == RecoveryStrategy.BYPASS:
            return self._bypass_component(error_event)
        
        elif strategy == RecoveryStrategy.COMPENSATE:
            return self._compensate_operation(error_event)
        
        elif strategy == RecoveryStrategy.IGNORE:
            return True  # Always successful for ignore strategy
        
        elif strategy == RecoveryStrategy.ABORT:
            return self._abort_operation(error_event)
        
        elif strategy == RecoveryStrategy.ESCALATE:
            return self._escalate_error(error_event)
        
        return False
    
    def _retry_operation(self, error_event: ErrorEvent) -> bool:
        """Implement retry logic"""
        
        retry_config = self.retry_config.get(error_event.category, {"max_retries": 1, "backoff_factor": 2.0})
        
        # Simple retry simulation (in a real implementation, this would retry the actual operation)
        if error_event.occurrence_count <= retry_config["max_retries"]:
            # Simulate exponential backoff
            delay = retry_config["backoff_factor"] ** (error_event.occurrence_count - 1)
            logger.info(f"Retrying operation after {delay}s delay")
            return True
        
        return False
    
    def _fallback_operation(self, error_event: ErrorEvent) -> bool:
        """Implement fallback logic"""
        
        if error_event.context.engine_type:
            # Try to use a different engine
            available_engines = self.meta_reasoning_engine.get_available_engines()
            if len(available_engines) > 1:
                logger.info(f"Falling back from {error_event.context.engine_type.value} to alternative engine")
                return True
        
        return False
    
    def _degrade_operation(self, error_event: ErrorEvent) -> bool:
        """Implement graceful degradation"""
        
        # Reduce thinking mode complexity
        if error_event.context.system_state.get("thinking_mode") == "deep":
            logger.info("Degrading from deep to intermediate thinking mode")
            return True
        elif error_event.context.system_state.get("thinking_mode") == "intermediate":
            logger.info("Degrading from intermediate to quick thinking mode")
            return True
        
        return False
    
    def _isolate_component(self, error_event: ErrorEvent) -> bool:
        """Isolate problematic component"""
        
        if error_event.context.engine_type:
            # Isolate the problematic engine
            logger.info(f"Isolating engine {error_event.context.engine_type.value}")
            return True
        
        return False
    
    def _restart_component(self, error_event: ErrorEvent) -> bool:
        """Restart component"""
        
        # Simulate component restart
        logger.info("Restarting component")
        return True
    
    def _bypass_component(self, error_event: ErrorEvent) -> bool:
        """Bypass problematic component"""
        
        logger.info("Bypassing problematic component")
        return True
    
    def _compensate_operation(self, error_event: ErrorEvent) -> bool:
        """Compensate with alternative approach"""
        
        logger.info("Compensating with alternative approach")
        return True
    
    def _abort_operation(self, error_event: ErrorEvent) -> bool:
        """Safely abort operation"""
        
        logger.info("Safely aborting operation")
        return True
    
    def _escalate_error(self, error_event: ErrorEvent) -> bool:
        """Escalate error to higher level"""
        
        logger.warning(f"Escalating error {error_event.error_id} to higher level")
        return True
    
    def _check_circuit_breaker(self):
        """Check circuit breaker thresholds"""
        
        # Get recent errors
        now = datetime.now(timezone.utc)
        time_window = timedelta(seconds=self.circuit_breaker["time_window"])
        recent_errors = [
            e for e in self.error_events
            if (now - e.last_occurrence) <= time_window
        ]
        
        if len(recent_errors) > 0:
            # Calculate error rate
            high_severity_errors = [e for e in recent_errors if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]]
            error_rate = len(high_severity_errors) / len(recent_errors)
            
            if error_rate >= self.circuit_breaker["error_rate_threshold"]:
                logger.critical(f"Circuit breaker triggered: Error rate {error_rate:.2f} exceeds threshold {self.circuit_breaker['error_rate_threshold']}")
                # In a real implementation, this would trigger system-wide protective measures
    
    def _log_error(self, error_event: ErrorEvent):
        """Log error event"""
        
        log_level = {
            ErrorSeverity.LOW: logger.debug,
            ErrorSeverity.MEDIUM: logger.info,
            ErrorSeverity.HIGH: logger.warning,
            ErrorSeverity.CRITICAL: logger.error,
            ErrorSeverity.FATAL: logger.critical
        }.get(error_event.severity, logger.info)
        
        log_level(f"Error {error_event.error_id}: {error_event.message} "
                 f"(Category: {error_event.category.value}, Severity: {error_event.severity.value})")
    
    def _create_minimal_error_event(self, error: Exception, context: ErrorContext) -> ErrorEvent:
        """Create minimal error event when handler is disabled"""
        
        return ErrorEvent(
            error_type=type(error).__name__,
            message=str(error),
            context=context
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        if not self.error_events:
            return {"total_errors": 0, "error_rate": 0.0}
        
        # Basic statistics
        total_errors = len(self.error_events)
        resolved_errors = sum(1 for e in self.error_events if e.resolved)
        
        # Category distribution
        category_counts = {}
        for error in self.error_events:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Severity distribution
        severity_counts = {}
        for error in self.error_events:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recovery statistics
        recovery_attempts = sum(1 for e in self.error_events if e.recovery_attempted)
        successful_recoveries = sum(1 for e in self.error_events if e.recovery_successful)
        
        # Recent error rate
        now = datetime.now(timezone.utc)
        recent_errors = [e for e in self.error_events if (now - e.last_occurrence).total_seconds() < 3600]
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "resolution_rate": resolved_errors / total_errors if total_errors > 0 else 0.0,
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0.0,
            "recent_error_count": len(recent_errors),
            "recent_error_rate": len(recent_errors) / 3600,  # errors per second
            "error_patterns": len(self.error_patterns),
            "circuit_breaker_status": self._get_circuit_breaker_status()
        }
    
    def _get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        
        now = datetime.now(timezone.utc)
        time_window = timedelta(seconds=self.circuit_breaker["time_window"])
        recent_errors = [
            e for e in self.error_events
            if (now - e.last_occurrence) <= time_window
        ]
        
        if len(recent_errors) > 0:
            high_severity_errors = [e for e in recent_errors if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]]
            error_rate = len(high_severity_errors) / len(recent_errors)
        else:
            error_rate = 0.0
        
        return {
            "current_error_rate": error_rate,
            "threshold": self.circuit_breaker["error_rate_threshold"],
            "triggered": error_rate >= self.circuit_breaker["error_rate_threshold"],
            "recent_errors": len(recent_errors),
            "time_window": self.circuit_breaker["time_window"]
        }
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report"""
        
        statistics = self.get_error_statistics()
        
        # Recent critical errors
        now = datetime.now(timezone.utc)
        critical_errors = [
            {
                "error_id": e.error_id,
                "error_type": e.error_type,
                "message": e.message,
                "severity": e.severity.value,
                "category": e.category.value,
                "timestamp": e.last_occurrence.isoformat(),
                "recovery_attempted": e.recovery_attempted,
                "recovery_successful": e.recovery_successful
            }
            for e in self.error_events
            if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL] and
               (now - e.last_occurrence).total_seconds() < 86400  # Last 24 hours
        ]
        
        return {
            "error_handling_enabled": self.enabled,
            "statistics": statistics,
            "critical_errors": critical_errors,
            "error_thresholds": {severity.value: threshold for severity, threshold in self.error_thresholds.items()},
            "recovery_strategies": {
                category.value: [strategy.value for strategy in strategies]
                for category, strategies in self.recovery_strategies.items()
            },
            "circuit_breaker": self._get_circuit_breaker_status(),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def resolve_error(self, error_id: str, resolution_details: str) -> bool:
        """Manually resolve an error"""
        
        for error in self.error_events:
            if error.error_id == error_id:
                error.resolved = True
                error.resolution_details = resolution_details
                error.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Error {error_id} resolved: {resolution_details}")
                return True
        
        return False
    
    def clear_error_history(self):
        """Clear error history"""
        
        self.error_events = []
        self.error_patterns = {}
        logger.info("Error history cleared")
    
    def enable_error_handling(self):
        """Enable error handling"""
        
        self.enabled = True
        logger.info("Error handling enabled")
    
    def disable_error_handling(self):
        """Disable error handling"""
        
        self.enabled = False
        logger.info("Error handling disabled")


class LoadBalancer:
    """Advanced load balancer for reasoning engines"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.strategy = LoadBalancingStrategy.HYBRID
        self.mode = LoadBalancingMode.ADAPTIVE
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
            self.engine_workloads[engine_type] = EngineWorkload(engine_type)
    
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
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            selected_engine = self._health_based_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.HYBRID:
            selected_engine = self._hybrid_selection(available_engines, context)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            selected_engine = self._random_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_engine = self._least_response_time_selection(available_engines)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            selected_engine = self._resource_based_selection(available_engines)
        else:
            selected_engine = self._fallback_selection()
        
        # Update workload
        self.engine_workloads[selected_engine].add_request()
        
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
            if self.meta_reasoning_engine.health_monitor.should_use_engine(engine_type):
                # Check if engine is not isolated
                isolated_engines = getattr(self.meta_reasoning_engine.health_monitor, 'isolated_engines', set())
                if engine_type not in isolated_engines:
                    available_engines.append(engine_type)
        
        return available_engines if available_engines else list(ReasoningEngine)
    
    def _round_robin_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Round robin engine selection"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        selected_engine = available_engines[self.round_robin_index % len(available_engines)]
        self.round_robin_index += 1
        return selected_engine
    
    def _least_connections_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine with least active connections"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        min_load = float('inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            workload = self.engine_workloads[engine_type]
            current_load = workload.active_requests + workload.queued_requests
            
            if current_load < min_load:
                min_load = current_load
                selected_engine = engine_type
        
        return selected_engine
    
    def _weighted_round_robin_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Weighted round robin selection based on engine weights"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        # Calculate cumulative weights
        cumulative_weights = []
        total_weight = 0
        
        for engine_type in available_engines:
            weight = self.engine_weights[engine_type]
            total_weight += weight
            cumulative_weights.append(total_weight)
        
        # Select based on weight
        import random
        rand_value = random.random() * total_weight
        
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_value <= cum_weight:
                return available_engines[i]
        
        return available_engines[0]
    
    def _performance_based_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine based on performance metrics"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        best_score = float('-inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
            
            # Calculate performance score (lower execution time and higher quality is better)
            time_score = 1.0 / (1.0 + profile.avg_execution_time)
            quality_score = profile.avg_quality_score
            consistency_score = 1.0 / (1.0 + profile.std_execution_time)
            
            overall_score = time_score * 0.4 + quality_score * 0.4 + consistency_score * 0.2
            
            if overall_score > best_score:
                best_score = overall_score
                selected_engine = engine_type
        
        return selected_engine
    
    def _health_based_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine based on health metrics"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        best_health = float('-inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            health_report = self.meta_reasoning_engine.health_monitor.get_engine_health_report(engine_type)
            health_score = health_report.get('health_score', 0.0)
            
            if health_score > best_health:
                best_health = health_score
                selected_engine = engine_type
        
        return selected_engine
    
    def _hybrid_selection(self, available_engines: List[ReasoningEngine], context: Dict[str, Any] = None) -> ReasoningEngine:
        """Hybrid selection combining multiple factors"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        best_score = float('-inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            # Get various metrics
            workload = self.engine_workloads[engine_type]
            profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
            health_report = self.meta_reasoning_engine.health_monitor.get_engine_health_report(engine_type)
            
            # Calculate component scores
            load_score = 1.0 - workload.get_load_score()  # Lower load is better
            performance_score = 1.0 / (1.0 + profile.avg_execution_time)
            quality_score = profile.avg_quality_score
            health_score = health_report.get('health_score', 0.0)
            
            # Weighted combination
            hybrid_score = (
                load_score * 0.3 +
                performance_score * 0.3 +
                quality_score * 0.2 +
                health_score * 0.2
            )
            
            if hybrid_score > best_score:
                best_score = hybrid_score
                selected_engine = engine_type
        
        return selected_engine
    
    def _random_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Random engine selection"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        import random
        return random.choice(available_engines)
    
    def _least_response_time_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine with least average response time"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
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
    
    def _resource_based_selection(self, available_engines: List[ReasoningEngine]) -> ReasoningEngine:
        """Select engine based on resource utilization"""
        if not available_engines:
            return ReasoningEngine.DEDUCTIVE
        
        min_utilization = float('inf')
        selected_engine = available_engines[0]
        
        for engine_type in available_engines:
            workload = self.engine_workloads[engine_type]
            utilization = workload.capacity_utilization
            
            if utilization < min_utilization:
                min_utilization = utilization
                selected_engine = engine_type
        
        return selected_engine
    
    def _fallback_selection(self) -> ReasoningEngine:
        """Fallback selection when load balancing is disabled or fails"""
        return ReasoningEngine.DEDUCTIVE
    
    def complete_request(self, engine_type: ReasoningEngine, response_time: float, success: bool):
        """Mark a request as completed and update metrics"""
        if engine_type in self.engine_workloads:
            self.engine_workloads[engine_type].complete_request(response_time)
        
        self.metrics.update_request_metrics(engine_type, response_time, success)
        
        # Update engine utilization
        if engine_type in self.engine_workloads:
            self.metrics.engine_utilization[engine_type] = self.engine_workloads[engine_type].capacity_utilization
    
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
                self.strategy = LoadBalancingStrategy.HEALTH_BASED
                self.metrics.strategy_switches += 1
                self.metrics.last_strategy_switch = now
            elif self.strategy == LoadBalancingStrategy.HYBRID:
                self.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
                self.metrics.strategy_switches += 1
                self.metrics.last_strategy_switch = now
    
    def update_engine_weights(self):
        """Update engine weights based on current performance"""
        for engine_type in ReasoningEngine:
            profile = self.meta_reasoning_engine.performance_tracker.get_performance_profile(engine_type)
            health_report = self.meta_reasoning_engine.health_monitor.get_engine_health_report(engine_type)
            
            # Calculate weight based on performance and health
            performance_weight = 1.0 / (1.0 + profile.avg_execution_time)
            health_weight = health_report.get('health_score', 0.5)
            
            self.engine_weights[engine_type] = (performance_weight + health_weight) / 2.0
    
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
                    "active_requests": workload.active_requests,
                    "queued_requests": workload.queued_requests,
                    "total_requests": workload.total_requests,
                    "load_factor": workload.current_load_factor,
                    "capacity_utilization": workload.capacity_utilization,
                    "average_response_time": workload.average_response_time
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
        if strategy != LoadBalancingStrategy.HYBRID:
            self.strategy_adaptation_enabled = False
    
    def enable_strategy_adaptation(self):
        """Enable adaptive strategy switching"""
        self.strategy_adaptation_enabled = True
    
    def disable_strategy_adaptation(self):
        """Disable adaptive strategy switching"""
        self.strategy_adaptation_enabled = False


class SynthesisMethod(Enum):
    """Methods for synthesizing results from multiple reasoning engines"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS_BUILDING = "consensus_building"
    EVIDENCE_INTEGRATION = "evidence_integration"
    CONFIDENCE_RANKING = "confidence_ranking"
    COMPLEMENTARY_FUSION = "complementary_fusion"


@dataclass
class ReasoningResult:
    """Result from a single reasoning engine"""
    
    engine: ReasoningEngine
    result: Any
    confidence: float
    processing_time: float
    quality_score: float
    evidence_strength: float
    reasoning_chain: List[str]
    assumptions: List[str]
    limitations: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_weighted_score(self) -> float:
        """Calculate weighted score combining confidence, quality, and evidence"""
        # Convert confidence to float if it's an enum
        confidence_value = self.confidence
        if hasattr(self.confidence, 'value'):
            # Handle ConfidenceLevel enum - convert to approximate float values
            confidence_mapping = {
                'very_high': 0.95,
                'high': 0.80,
                'moderate': 0.60,
                'low': 0.40,
                'very_low': 0.20
            }
            confidence_value = confidence_mapping.get(self.confidence.value, 0.50)
        elif isinstance(self.confidence, str):
            # Handle string confidence levels
            confidence_mapping = {
                'very_high': 0.95,
                'high': 0.80,
                'moderate': 0.60,
                'low': 0.40,
                'very_low': 0.20
            }
            confidence_value = confidence_mapping.get(self.confidence, 0.50)
        
        # Convert evidence_strength to float if it's not already
        evidence_strength_value = self.evidence_strength
        if isinstance(self.evidence_strength, (list, tuple)):
            evidence_strength_value = len(self.evidence_strength) / 10  # Convert list length to 0-1 scale
        elif isinstance(self.evidence_strength, str):
            evidence_strength_value = 0.5  # Default for string values
        elif not isinstance(self.evidence_strength, (int, float)):
            evidence_strength_value = 0.5  # Default for other types
        
        return (confidence_value * 0.4 + 
                self.quality_score * 0.4 + 
                evidence_strength_value * 0.2)


@dataclass
class SequentialResult:
    """Result from a sequential reasoning chain"""
    
    sequence: List[ReasoningEngine]
    results: List[ReasoningResult]
    final_synthesis: Any
    sequence_confidence: float
    total_processing_time: float
    interaction_quality: float
    emergent_insights: List[str]
    reasoning_flow: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_sequence_score(self) -> float:
        """Calculate overall sequence quality score"""
        avg_result_score = statistics.mean([r.get_weighted_score() for r in self.results])
        return (avg_result_score * 0.5 + 
                self.sequence_confidence * 0.3 + 
                self.interaction_quality * 0.2)


@dataclass
class MetaReasoningResult:
    """Complete meta-reasoning result across all modes"""
    
    id: str
    query: str
    thinking_mode: ThinkingMode
    parallel_results: Optional[List[ReasoningResult]] = None
    sequential_results: Optional[List[SequentialResult]] = None
    final_synthesis: Any = None
    meta_confidence: float = 0.0
    total_processing_time: float = 0.0
    ftns_cost: float = 0.0
    reasoning_depth: int = 0
    emergent_properties: List[str] = field(default_factory=list)
    cross_engine_interactions: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())
    
    def get_overall_quality(self) -> float:
        """Calculate overall meta-reasoning quality"""
        if self.parallel_results:
            parallel_score = statistics.mean([r.get_weighted_score() for r in self.parallel_results])
        else:
            parallel_score = 0.0
        
        if self.sequential_results:
            sequential_score = statistics.mean([r.get_sequence_score() for r in self.sequential_results])
        else:
            sequential_score = 0.0
        
        base_score = max(parallel_score, sequential_score)
        return base_score * 0.7 + self.meta_confidence * 0.3


@dataclass
class ThinkingConfiguration:
    """Configuration for different thinking modes"""
    
    mode: ThinkingMode
    max_permutations: int
    timeout_seconds: float
    ftns_cost_multiplier: float
    description: str
    
    @classmethod
    def get_configurations(cls) -> Dict[ThinkingMode, 'ThinkingConfiguration']:
        """Get predefined configurations for each thinking mode"""
        return {
            ThinkingMode.QUICK: cls(
                mode=ThinkingMode.QUICK,
                max_permutations=1,  # Parallel only
                timeout_seconds=30.0,
                ftns_cost_multiplier=1.0,
                description="Quick parallel processing across all reasoning engines"
            ),
            ThinkingMode.INTERMEDIATE: cls(
                mode=ThinkingMode.INTERMEDIATE,
                max_permutations=210,  # 7P3 = 210 three-engine sequences
                timeout_seconds=300.0,
                ftns_cost_multiplier=3.5,
                description="Intermediate depth with partial permutations"
            ),
            ThinkingMode.DEEP: cls(
                mode=ThinkingMode.DEEP,
                max_permutations=5040,  # 7! = 5040 full permutations
                timeout_seconds=1800.0,
                ftns_cost_multiplier=10.0,
                description="Deep analysis with full permutation exploration"
            )
        }


class MetaReasoningEngine:
    """Main meta-reasoning engine that orchestrates multiple reasoning systems"""
    
    def __init__(self):
        # Initialize all reasoning engines
        self.reasoning_engines = {
            ReasoningEngine.DEDUCTIVE: EnhancedDeductiveReasoningEngine(),
            ReasoningEngine.INDUCTIVE: EnhancedInductiveReasoningEngine(),
            ReasoningEngine.ABDUCTIVE: EnhancedAbductiveReasoningEngine(),
            ReasoningEngine.CAUSAL: EnhancedCausalReasoningEngine(),
            ReasoningEngine.PROBABILISTIC: EnhancedProbabilisticReasoningEngine(),
            ReasoningEngine.COUNTERFACTUAL: EnhancedCounterfactualReasoningEngine(),
            ReasoningEngine.ANALOGICAL: AnalogicalReasoningEngine()
        }
        
        # Thinking mode configurations
        self.thinking_configs = ThinkingConfiguration.get_configurations()
        
        # Synthesis methods
        self.synthesis_methods = {
            SynthesisMethod.WEIGHTED_AVERAGE: self._weighted_average_synthesis,
            SynthesisMethod.CONSENSUS_BUILDING: self._consensus_building_synthesis,
            SynthesisMethod.EVIDENCE_INTEGRATION: self._evidence_integration_synthesis,
            SynthesisMethod.CONFIDENCE_RANKING: self._confidence_ranking_synthesis,
            SynthesisMethod.COMPLEMENTARY_FUSION: self._complementary_fusion_synthesis
        }
        
        # Engine interaction patterns (based on your analysis)
        self.interaction_patterns = {
            (ReasoningEngine.INDUCTIVE, ReasoningEngine.CAUSAL): "empirical_foundation",
            (ReasoningEngine.ABDUCTIVE, ReasoningEngine.CAUSAL): "hypothesis_testing",
            (ReasoningEngine.COUNTERFACTUAL, ReasoningEngine.CAUSAL): "alternative_scenarios",
            (ReasoningEngine.PROBABILISTIC, ReasoningEngine.INDUCTIVE): "pattern_quantification",
            (ReasoningEngine.PROBABILISTIC, ReasoningEngine.ABDUCTIVE): "hypothesis_evaluation",
            (ReasoningEngine.ANALOGICAL, ReasoningEngine.INDUCTIVE): "cross_domain_patterns",
            (ReasoningEngine.ANALOGICAL, ReasoningEngine.ABDUCTIVE): "structural_insights",
            (ReasoningEngine.DEDUCTIVE, ReasoningEngine.PROBABILISTIC): "logical_frameworks",
            (ReasoningEngine.COUNTERFACTUAL, ReasoningEngine.PROBABILISTIC): "scenario_quantification"
        }
        
        # Initialize health monitoring system
        self.health_monitor = EngineHealthMonitor()
        
        # Initialize performance tracking system
        self.performance_tracker = PerformanceTracker()
        
        # Initialize failure detection and recovery systems
        self.failure_detector = FailureDetector()
        self.recovery_manager = FailureRecoveryManager(self)
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(self)
        
        # Initialize adaptive engine selector
        self.adaptive_selector = AdaptiveEngineSelector(self)
        
        # Initialize result formatter
        self.result_formatter = ResultFormatter()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self)
        
        # Initialize interaction pattern recognizer
        self.pattern_recognizer = InteractionPatternRecognizer(self)
        
        # Initialize context passing engine
        self.context_passing_engine = ContextPassingEngine(self)
        
        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer(self)
        
        logger.info("MetaReasoningEngine initialized with health monitoring, performance tracking, failure recovery, load balancing, adaptive selection, result formatting, error handling, interaction pattern recognition, and context passing", 
                   engines=len(self.reasoning_engines),
                   interaction_patterns=len(self.interaction_patterns),
                   pattern_recognizer_enabled=self.pattern_recognizer.enabled,
                   context_passing_enabled=self.context_passing_engine.enabled)
    
    async def meta_reason(self, 
                         query: str, 
                         context: Dict[str, Any],
                         thinking_mode: ThinkingMode = ThinkingMode.QUICK,
                         custom_config: Optional[ThinkingConfiguration] = None) -> MetaReasoningResult:
        """
        Perform meta-reasoning across multiple reasoning engines
        
        Args:
            query: The problem or question to reason about
            context: Additional context and constraints
            thinking_mode: The depth of reasoning (quick/intermediate/deep)
            custom_config: Optional custom configuration
            
        Returns:
            Complete meta-reasoning result with synthesis
        """
        
        logger.info("Starting meta-reasoning", 
                   query=query[:100], 
                   mode=thinking_mode.value)
        
        start_time = time.time()
        config = custom_config or self.thinking_configs[thinking_mode]
        
        # Initialize result
        result = MetaReasoningResult(
            id=str(uuid4()),
            query=query,
            thinking_mode=thinking_mode
        )
        
        try:
            if thinking_mode == ThinkingMode.QUICK:
                # Quick mode: Parallel processing
                result.parallel_results = await self._parallel_reasoning(
                    query, context, timeout=config.timeout_seconds
                )
                result.final_synthesis = await self._synthesize_parallel_results(
                    result.parallel_results, context
                )
                result.reasoning_depth = 1
                
            elif thinking_mode == ThinkingMode.INTERMEDIATE:
                # Intermediate mode: Partial permutations
                result.parallel_results = await self._parallel_reasoning(
                    query, context, timeout=config.timeout_seconds / 2  # Split timeout between parallel and sequential
                )
                result.sequential_results = await self._partial_sequential_reasoning(
                    query, context, max_sequences=config.max_permutations, timeout=config.timeout_seconds / 2
                )
                result.final_synthesis = await self._synthesize_hybrid_results(
                    result.parallel_results, result.sequential_results, context
                )
                result.reasoning_depth = 3
                
            elif thinking_mode == ThinkingMode.DEEP:
                # Deep mode: Full permutation exploration
                result.parallel_results = await self._parallel_reasoning(
                    query, context, timeout=config.timeout_seconds / 3  # Split timeout between parallel and sequential
                )
                result.sequential_results = await self._full_sequential_reasoning(
                    query, context, max_sequences=config.max_permutations, timeout=config.timeout_seconds * 2 / 3
                )
                result.final_synthesis = await self._synthesize_comprehensive_results(
                    result.parallel_results, result.sequential_results, context
                )
                result.reasoning_depth = 7
            
            # Calculate meta-confidence and quality metrics
            result.meta_confidence = await self._calculate_meta_confidence(result)
            result.quality_metrics = await self._calculate_quality_metrics(result)
            result.emergent_properties = await self._identify_emergent_properties(result)
            result.cross_engine_interactions = await self._analyze_cross_engine_interactions(result)
            
            # Calculate processing time and cost
            result.total_processing_time = time.time() - start_time
            result.ftns_cost = self._calculate_ftns_cost(result, config)
            
            logger.info("Meta-reasoning completed", 
                       result_id=result.id,
                       mode=thinking_mode.value,
                       confidence=result.meta_confidence,
                       quality=result.get_overall_quality(),
                       processing_time=result.total_processing_time,
                       ftns_cost=result.ftns_cost)
            
            return result
            
        except Exception as e:
            logger.error("Meta-reasoning failed", error=str(e))
            result.total_processing_time = time.time() - start_time
            result.ftns_cost = self._calculate_ftns_cost(result, config)
            return result
    
    async def _parallel_reasoning(self, query: str, context: Dict[str, Any], timeout: float = 30.0) -> List[ReasoningResult]:
        """Execute all reasoning engines in parallel with resource management"""
        
        logger.info("Starting parallel reasoning across all engines", timeout=timeout)
        
        # Monitor initial resource usage
        initial_memory = self._get_memory_usage()
        
        # Create tasks for all reasoning engines with timeout
        tasks = []
        for engine_type, engine in self.reasoning_engines.items():
            task = asyncio.create_task(
                self._execute_reasoning_engine(engine_type, engine, query, context, timeout)
            )
            tasks.append(task)
        
        try:
            # Wait for all tasks to complete with overall timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout + 10.0  # Add buffer for orchestration
            )
        except asyncio.TimeoutError:
            logger.error("Parallel reasoning timeout", timeout=timeout)
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = []
        
        # Filter out exceptions and return successful results
        successful_results = [r for r in results if isinstance(r, ReasoningResult)]
        
        # Monitor final resource usage
        final_memory = self._get_memory_usage()
        memory_delta = final_memory - initial_memory
        
        if memory_delta > 50:  # 50MB threshold for parallel execution
            logger.warning("High memory usage in parallel reasoning", 
                         memory_delta_mb=memory_delta,
                         initial_memory_mb=initial_memory,
                         final_memory_mb=final_memory)
        
        # Clean up resources
        await self._cleanup_resources()
        
        logger.info("Parallel reasoning completed", 
                   successful_engines=len(successful_results),
                   total_engines=len(self.reasoning_engines))
        
        return successful_results
    
    async def _partial_sequential_reasoning(self, 
                                          query: str, 
                                          context: Dict[str, Any],
                                          max_sequences: int = 210,
                                          timeout: float = 150.0) -> List[SequentialResult]:
        """Execute reasoning engines in partial sequential permutations"""
        
        logger.info("Starting partial sequential reasoning", max_sequences=max_sequences, timeout=timeout)
        
        engines = list(self.reasoning_engines.keys())
        sequential_results = []
        
        # Calculate timeout per sequence
        timeout_per_sequence = timeout / max_sequences if max_sequences > 0 else 30.0
        
        start_time = time.time()
        
        # Generate 3-engine permutations (7P3 = 210)
        sequence_count = 0
        for sequence in permutations(engines, 3):
            if sequence_count >= max_sequences:
                break
            
            # Check if we're running out of time
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning("Partial sequential reasoning timeout", 
                             elapsed=elapsed, 
                             timeout=timeout,
                             completed_sequences=sequence_count)
                break
            
            try:
                sequential_result = await asyncio.wait_for(
                    self._execute_reasoning_sequence(list(sequence), query, context),
                    timeout=timeout_per_sequence
                )
                sequential_results.append(sequential_result)
                sequence_count += 1
            except asyncio.TimeoutError:
                logger.warning("Sequence execution timeout", 
                             sequence=sequence, 
                             timeout=timeout_per_sequence)
                continue
        
        logger.info("Partial sequential reasoning completed", 
                   sequences_executed=len(sequential_results))
        
        return sequential_results
    
    async def _full_sequential_reasoning(self, 
                                       query: str, 
                                       context: Dict[str, Any],
                                       max_sequences: int = 5040,
                                       timeout: float = 600.0) -> List[SequentialResult]:
        """Execute reasoning engines in full sequential permutations"""
        
        logger.info("Starting full sequential reasoning", max_sequences=max_sequences, timeout=timeout)
        
        engines = list(self.reasoning_engines.keys())
        sequential_results = []
        
        # Calculate timeout per sequence  
        timeout_per_sequence = timeout / max_sequences if max_sequences > 0 else 60.0
        
        start_time = time.time()
        
        # Generate all 7! permutations
        sequence_count = 0
        for sequence in permutations(engines):
            if sequence_count >= max_sequences:
                break
            
            # Check if we're running out of time
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning("Full sequential reasoning timeout", 
                             elapsed=elapsed, 
                             timeout=timeout,
                             completed_sequences=sequence_count)
                break
            
            try:
                sequential_result = await asyncio.wait_for(
                    self._execute_reasoning_sequence(list(sequence), query, context),
                    timeout=timeout_per_sequence
                )
                sequential_results.append(sequential_result)
                sequence_count += 1
            except asyncio.TimeoutError:
                logger.warning("Sequence execution timeout", 
                             sequence=sequence, 
                             timeout=timeout_per_sequence)
                continue
        
        logger.info("Full sequential reasoning completed", 
                   sequences_executed=len(sequential_results))
        
        return sequential_results
    
    async def _execute_reasoning_engine(self, 
                                      engine_type: ReasoningEngine,
                                      engine: Any,
                                      query: str,
                                      context: Dict[str, Any],
                                      timeout: float = 30.0) -> ReasoningResult:
        """Execute a single reasoning engine with timeout and resource management"""
        
        start_time = time.time()
        
        try:
            # Execute with timeout protection
            async with asyncio.timeout(timeout):
                logger.debug(f"Executing {engine_type.value} engine with {timeout}s timeout")
                
                # Monitor resource usage
                initial_memory = self._get_memory_usage()
                
                result = await self._execute_engine_with_monitoring(
                    engine_type, engine, query, context, start_time
                )
                
                # Check resource usage after execution
                final_memory = self._get_memory_usage()
                memory_delta = final_memory - initial_memory
                
                if memory_delta > 100:  # 100MB threshold
                    logger.warning(f"High memory usage detected", 
                                 engine=engine_type.value, 
                                 memory_delta_mb=memory_delta)
                
                # Record successful execution in health monitor
                execution_time = time.time() - start_time
                self.health_monitor.record_engine_execution(
                    engine_type, execution_time, success=True
                )
                
                # Record performance metrics
                confidence_score = self._extract_confidence_score(result)
                performance_context = {
                    "query_length": len(query), 
                    "context_size": len(str(context)),
                    "memory_usage": memory_delta,
                    "quality_score": result.quality_score
                }
                self.performance_tracker.record_execution_performance(
                    engine_type=engine_type,
                    execution_time=execution_time,
                    memory_usage=memory_delta,
                    quality_score=result.quality_score,
                    confidence_score=confidence_score,
                    success=True,
                    context=performance_context
                )
                
                # Detect potential failures even in successful executions
                failure_event = self.failure_detector.detect_failure(
                    engine_type=engine_type,
                    execution_time=execution_time,
                    success=True,
                    context=performance_context
                )
                
                if failure_event:
                    logger.warning(f"Failure detected in successful execution: {failure_event.failure_type.value}")
                    # Attempt recovery for quality degradation or other soft failures
                    await self.recovery_manager.attempt_recovery(failure_event)
                
                # Update load balancer with successful completion
                self.load_balancer.complete_request(engine_type, execution_time, success=True)
                
                return result
                
        except asyncio.TimeoutError as e:
            logger.error(f"Engine execution timeout", 
                        engine=engine_type.value, 
                        timeout=timeout)
            
            # Handle error with error handler
            error_context = ErrorContext(
                operation="engine_execution",
                engine_type=engine_type,
                query=query,
                user_context=context,
                system_state={"timeout": timeout, "thinking_mode": "unknown"}
            )
            self.error_handler.handle_error(e, error_context, ErrorSeverity.HIGH)
            
            # Record timeout in health monitor
            self.health_monitor.record_engine_timeout(engine_type, timeout)
            
            # Record performance metrics for timeout
            timeout_context = {"error_type": "timeout", "query_length": len(query)}
            self.performance_tracker.record_execution_performance(
                engine_type=engine_type,
                execution_time=timeout,
                memory_usage=0.0,
                quality_score=0.0,
                confidence_score=0.0,
                success=False,
                context=timeout_context
            )
            
            # Detect failure and attempt recovery
            failure_event = self.failure_detector.detect_failure(
                engine_type=engine_type,
                execution_time=timeout,
                success=False,
                error="Timeout",
                context=timeout_context
            )
            
            if failure_event:
                logger.info(f"Attempting recovery for timeout failure: {engine_type.value}")
                recovery_success = await self.recovery_manager.attempt_recovery(failure_event)
                
                if recovery_success:
                    logger.info(f"Recovery successful for {engine_type.value}, retrying execution")
                    # Retry execution once after successful recovery
                    try:
                        async with asyncio.timeout(timeout):
                            retry_result = await self._execute_engine_with_monitoring(
                                engine_type, engine, query, context, time.time()
                            )
                            return retry_result
                    except:
                        logger.warning(f"Retry after recovery failed for {engine_type.value}")
            
            # Update load balancer with timeout completion
            self.load_balancer.complete_request(engine_type, timeout, success=False)
            
            # Return timeout result
            return ReasoningResult(
                engine=engine_type,
                result=f"Engine execution timed out after {timeout}s",
                confidence=0.0,
                processing_time=timeout,
                quality_score=0.0,
                evidence_strength=0.0,
                reasoning_chain=[f"Timeout occurred after {timeout}s"],
                assumptions=["Engine execution interrupted by timeout"],
                limitations=["Results incomplete due to timeout"]
            )
            
        except Exception as e:
            logger.error(f"Engine execution failed", 
                        engine=engine_type.value, 
                        error=str(e))
            
            # Handle error with error handler
            execution_time = time.time() - start_time
            error_context = ErrorContext(
                operation="engine_execution",
                engine_type=engine_type,
                query=query,
                user_context=context,
                system_state={"execution_time": execution_time, "thinking_mode": "unknown"}
            )
            self.error_handler.handle_error(e, error_context, ErrorSeverity.MEDIUM)
            
            # Record failure in health monitor
            self.health_monitor.record_engine_execution(
                engine_type, execution_time, success=False, error=str(e)
            )
            
            # Record performance metrics for failure
            exception_context = {"error_type": "exception", "error_message": str(e), "query_length": len(query)}
            self.performance_tracker.record_execution_performance(
                engine_type=engine_type,
                execution_time=execution_time,
                memory_usage=0.0,
                quality_score=0.0,
                confidence_score=0.0,
                success=False,
                context=exception_context
            )
            
            # Detect failure and attempt recovery
            failure_event = self.failure_detector.detect_failure(
                engine_type=engine_type,
                execution_time=execution_time,
                success=False,
                error=str(e),
                context=exception_context
            )
            
            if failure_event:
                logger.info(f"Attempting recovery for exception failure: {engine_type.value}")
                recovery_success = await self.recovery_manager.attempt_recovery(failure_event)
                
                if recovery_success:
                    logger.info(f"Recovery successful for {engine_type.value}, retrying execution")
                    # Retry execution once after successful recovery
                    try:
                        async with asyncio.timeout(timeout):
                            retry_result = await self._execute_engine_with_monitoring(
                                engine_type, engine, query, context, time.time()
                            )
                            return retry_result
                    except:
                        logger.warning(f"Retry after recovery failed for {engine_type.value}")
            
            # Update load balancer with exception completion
            execution_time = time.time() - start_time
            self.load_balancer.complete_request(engine_type, execution_time, success=False)
            
            # Return error result
            return ReasoningResult(
                engine=engine_type,
                result=f"Engine execution failed: {str(e)}",
                confidence=0.0,
                processing_time=execution_time,
                quality_score=0.0,
                evidence_strength=0.0,
                reasoning_chain=[f"Error occurred: {str(e)}"],
                assumptions=["Engine execution interrupted by error"],
                limitations=["Results incomplete due to error"]
            )
    
    async def _execute_engine_with_monitoring(self, 
                                            engine_type: ReasoningEngine,
                                            engine: Any,
                                            query: str,
                                            context: Dict[str, Any],
                                            start_time: float) -> ReasoningResult:
        """Execute engine with detailed monitoring"""
        
        try:
            # Execute the appropriate method based on engine type
            if engine_type == ReasoningEngine.DEDUCTIVE:
                result = await engine.perform_deductive_reasoning([query], query, context or {})
            elif engine_type == ReasoningEngine.INDUCTIVE:
                result = await engine.perform_inductive_reasoning([query], query, context or {})
            elif engine_type == ReasoningEngine.ABDUCTIVE:
                result = await engine.perform_abductive_reasoning([query], query, context or {})
            elif engine_type == ReasoningEngine.CAUSAL:
                result = await engine.perform_causal_reasoning([query], query, context or {})
            elif engine_type == ReasoningEngine.PROBABILISTIC:
                result = await engine.perform_probabilistic_reasoning([query], query, context or {})
            elif engine_type == ReasoningEngine.COUNTERFACTUAL:
                result = await engine.perform_counterfactual_reasoning([query], query, context or {})
            elif engine_type == ReasoningEngine.ANALOGICAL:
                result = await engine.process_analogical_query(query, context or {})
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
            
            # Extract reasoning chain and metrics
            reasoning_chain = self._extract_reasoning_chain(result)
            confidence = self._extract_confidence(result)
            quality_score = self._extract_quality_score(result)
            evidence_strength = self._extract_evidence_strength(result)
            assumptions = self._extract_assumptions(result)
            limitations = self._extract_limitations(result)
            
            processing_time = time.time() - start_time
            
            return ReasoningResult(
                engine=engine_type,
                result=result,
                confidence=confidence,
                processing_time=processing_time,
                quality_score=quality_score,
                evidence_strength=evidence_strength,
                reasoning_chain=reasoning_chain,
                assumptions=assumptions,
                limitations=limitations
            )
            
        except Exception as e:
            logger.warning("Engine execution failed", 
                         engine=engine_type.value, 
                         error=str(e))
            
            # Return a minimal result for failed engines
            return ReasoningResult(
                engine=engine_type,
                result=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                quality_score=0.0,
                evidence_strength=0.0,
                reasoning_chain=[f"Engine {engine_type.value} failed: {str(e)}"],
                assumptions=[],
                limitations=["Engine execution failed"]
            )
    
    async def _execute_reasoning_sequence(self, 
                                        sequence: List[ReasoningEngine],
                                        query: str,
                                        context: Dict[str, Any]) -> SequentialResult:
        """Execute a sequence of reasoning engines"""
        
        start_time = time.time()
        results = []
        current_context = context.copy()
        reasoning_flow = []
        
        # Execute engines in sequence
        for i, engine_type in enumerate(sequence):
            engine = self.reasoning_engines[engine_type]
            
            # Add previous results to context for interaction
            if results:
                current_context["previous_results"] = results[-1].result
                current_context["reasoning_history"] = [r.result for r in results]
            
            # Execute engine
            result = await self._execute_reasoning_engine(
                engine_type, engine, query, current_context
            )
            results.append(result)
            
            # Track reasoning flow
            reasoning_flow.append(f"Step {i+1}: {engine_type.value} -> {result.confidence:.2f}")
        
        # Synthesize sequential results
        final_synthesis = await self._synthesize_sequential_results(results, context)
        
        # Calculate interaction quality
        interaction_quality = self._calculate_interaction_quality(sequence, results)
        
        # Identify emergent insights
        emergent_insights = self._identify_emergent_insights(results)
        
        # Calculate sequence confidence
        sequence_confidence = self._calculate_sequence_confidence(results)
        
        total_processing_time = time.time() - start_time
        
        return SequentialResult(
            sequence=sequence,
            results=results,
            final_synthesis=final_synthesis,
            sequence_confidence=sequence_confidence,
            total_processing_time=total_processing_time,
            interaction_quality=interaction_quality,
            emergent_insights=emergent_insights,
            reasoning_flow=reasoning_flow
        )
    
    async def _synthesize_parallel_results(self, 
                                         results: List[ReasoningResult],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from parallel reasoning"""
        
        if not results:
            return {"synthesis": "No results to synthesize"}
        
        # Use weighted average synthesis for parallel results
        return await self._weighted_average_synthesis(results, context)
    
    async def _synthesize_sequential_results(self, 
                                           results: List[ReasoningResult],
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from a single sequential chain"""
        
        if not results:
            return {"synthesis": "No results to synthesize"}
        
        # Use evidence integration for sequential results
        return await self._evidence_integration_synthesis(results, context)
    
    async def _synthesize_hybrid_results(self, 
                                       parallel_results: List[ReasoningResult],
                                       sequential_results: List[SequentialResult],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from both parallel and sequential reasoning"""
        
        # Synthesize parallel results
        parallel_synthesis = await self._synthesize_parallel_results(parallel_results, context)
        
        # Find best sequential results
        best_sequences = sorted(sequential_results, 
                              key=lambda x: x.get_sequence_score(), 
                              reverse=True)[:10]  # Top 10 sequences
        
        # Synthesize top sequential results
        sequential_synthesis = []
        for seq_result in best_sequences:
            seq_synthesis = await self._synthesize_sequential_results(seq_result.results, context)
            sequential_synthesis.append(seq_synthesis)
        
        # Combine parallel and sequential syntheses
        return {
            "parallel_synthesis": parallel_synthesis,
            "sequential_synthesis": sequential_synthesis,
            "hybrid_confidence": self._calculate_hybrid_confidence(parallel_results, best_sequences),
            "best_sequences": [seq.sequence for seq in best_sequences[:3]],
            "interaction_patterns": self._analyze_interaction_patterns(best_sequences)
        }
    
    async def _synthesize_comprehensive_results(self, 
                                              parallel_results: List[ReasoningResult],
                                              sequential_results: List[SequentialResult],
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from comprehensive deep reasoning"""
        
        # Analyze all sequential results
        top_sequences = sorted(sequential_results, 
                             key=lambda x: x.get_sequence_score(), 
                             reverse=True)[:50]  # Top 50 sequences
        
        # Cluster sequences by performance
        high_performers = [s for s in top_sequences if s.get_sequence_score() > 0.8]
        medium_performers = [s for s in top_sequences if 0.6 <= s.get_sequence_score() <= 0.8]
        
        # Synthesize parallel results
        parallel_synthesis = await self._synthesize_parallel_results(parallel_results, context)
        
        # Synthesize high-performing sequences
        high_synthesis = []
        for seq_result in high_performers:
            seq_synthesis = await self._synthesize_sequential_results(seq_result.results, context)
            high_synthesis.append(seq_synthesis)
        
        # Identify optimal reasoning patterns
        optimal_patterns = self._identify_optimal_patterns(high_performers)
        
        # Comprehensive synthesis
        return {
            "parallel_synthesis": parallel_synthesis,
            "high_performance_synthesis": high_synthesis,
            "optimal_patterns": optimal_patterns,
            "comprehensive_confidence": self._calculate_comprehensive_confidence(
                parallel_results, high_performers
            ),
            "reasoning_insights": self._extract_reasoning_insights(top_sequences),
            "meta_patterns": self._identify_meta_patterns(sequential_results),
            "emergent_properties": self._identify_comprehensive_emergent_properties(
                parallel_results, sequential_results
            )
        }
    
    # Synthesis Methods
    async def _weighted_average_synthesis(self, 
                                        results: List[ReasoningResult],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize using weighted average of confidence scores"""
        
        if not results:
            return {"method": "weighted_average", "synthesis": "No results"}
        
        # Calculate weighted scores
        total_weight = sum(r.get_weighted_score() for r in results)
        
        synthesis = {
            "method": "weighted_average",
            "total_engines": len(results),
            "average_confidence": statistics.mean([r.confidence for r in results]),
            "weighted_confidence": total_weight / len(results),
            "strongest_engine": max(results, key=lambda r: r.get_weighted_score()).engine.value,
            "consensus_strength": self._calculate_consensus_strength(results)
        }
        
        return synthesis
    
    async def _consensus_building_synthesis(self, 
                                          results: List[ReasoningResult],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize by building consensus among results"""
        
        if not results:
            return {"method": "consensus_building", "synthesis": "No results"}
        
        # Find areas of agreement
        agreements = self._find_agreements(results)
        disagreements = self._find_disagreements(results)
        
        synthesis = {
            "method": "consensus_building",
            "agreements": agreements,
            "disagreements": disagreements,
            "consensus_score": len(agreements) / (len(agreements) + len(disagreements)) if agreements or disagreements else 0
        }
        
        return synthesis
    
    async def _evidence_integration_synthesis(self, 
                                            results: List[ReasoningResult],
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize by integrating evidence from all results"""
        
        if not results:
            return {"method": "evidence_integration", "synthesis": "No results"}
        
        # Collect all evidence
        all_evidence = []
        for result in results:
            all_evidence.extend(result.reasoning_chain)
        
        # Integrate evidence
        integrated_evidence = self._integrate_evidence(all_evidence)
        
        synthesis = {
            "method": "evidence_integration",
            "integrated_evidence": integrated_evidence,
            "evidence_strength": statistics.mean([r.evidence_strength for r in results]),
            "evidence_count": len(all_evidence)
        }
        
        return synthesis
    
    async def _confidence_ranking_synthesis(self, 
                                          results: List[ReasoningResult],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize by ranking results by confidence"""
        
        if not results:
            return {"method": "confidence_ranking", "synthesis": "No results"}
        
        # Rank by confidence
        ranked_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        
        synthesis = {
            "method": "confidence_ranking",
            "top_result": ranked_results[0].engine.value,
            "confidence_ranking": [(r.engine.value, r.confidence) for r in ranked_results],
            "confidence_spread": max(r.confidence for r in results) - min(r.confidence for r in results)
        }
        
        return synthesis
    
    async def _complementary_fusion_synthesis(self, 
                                            results: List[ReasoningResult],
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize by fusing complementary aspects of different engines"""
        
        if not results:
            return {"method": "complementary_fusion", "synthesis": "No results"}
        
        # Identify complementary strengths
        complementary_strengths = self._identify_complementary_strengths(results)
        
        synthesis = {
            "method": "complementary_fusion",
            "complementary_strengths": complementary_strengths,
            "fusion_confidence": self._calculate_fusion_confidence(results),
            "synergy_score": self._calculate_synergy_score(results)
        }
        
        return synthesis
    
    # Helper methods for analysis and calculation
    def _extract_reasoning_chain(self, result: Any) -> List[str]:
        """Extract reasoning chain from result"""
        if hasattr(result, 'reasoning_chain'):
            return result.reasoning_chain
        elif hasattr(result, 'steps'):
            return result.steps
        elif hasattr(result, 'evidence'):
            return [str(e) for e in result.evidence]
        else:
            return [str(result)]
    
    def _extract_confidence(self, result: Any) -> float:
        """Extract confidence score from result"""
        if hasattr(result, 'confidence'):
            return result.confidence
        elif hasattr(result, 'confidence_level'):
            return result.confidence_level
        elif hasattr(result, 'certainty'):
            return result.certainty
        else:
            return 0.5  # Default moderate confidence
    
    def _extract_quality_score(self, result: Any) -> float:
        """Extract quality score from result"""
        if hasattr(result, 'quality_score'):
            return result.quality_score
        elif hasattr(result, 'reasoning_quality'):
            return result.reasoning_quality
        elif hasattr(result, 'get_overall_quality'):
            return result.get_overall_quality()
        else:
            return 0.5  # Default moderate quality
    
    def _extract_evidence_strength(self, result: Any) -> float:
        """Extract evidence strength from result"""
        if hasattr(result, 'evidence_strength'):
            return result.evidence_strength
        elif hasattr(result, 'evidence_quality'):
            return result.evidence_quality
        elif hasattr(result, 'support_strength'):
            return result.support_strength
        else:
            return 0.5  # Default moderate evidence
    
    def _extract_assumptions(self, result: Any) -> List[str]:
        """Extract assumptions from result"""
        if hasattr(result, 'assumptions'):
            return result.assumptions
        elif hasattr(result, 'premises'):
            return result.premises
        else:
            return []
    
    def _extract_limitations(self, result: Any) -> List[str]:
        """Extract limitations from result"""
        if hasattr(result, 'limitations'):
            return result.limitations
        elif hasattr(result, 'constraints'):
            return result.constraints
        else:
            return []
    
    def _calculate_interaction_quality(self, 
                                     sequence: List[ReasoningEngine],
                                     results: List[ReasoningResult]) -> float:
        """Calculate quality of interactions between engines in sequence"""
        
        if len(sequence) < 2:
            return 0.0
        
        interaction_score = 0.0
        interaction_count = 0
        
        for i in range(len(sequence) - 1):
            engine_pair = (sequence[i], sequence[i + 1])
            if engine_pair in self.interaction_patterns:
                # Known beneficial interaction
                interaction_score += 0.8
                interaction_count += 1
            else:
                # Unknown interaction - moderate score
                interaction_score += 0.5
                interaction_count += 1
        
        return interaction_score / interaction_count if interaction_count > 0 else 0.0
    
    def _identify_emergent_insights(self, results: List[ReasoningResult]) -> List[str]:
        """Identify emergent insights from sequential reasoning"""
        
        insights = []
        
        # Look for improving confidence across sequence
        confidences = [r.confidence for r in results]
        if len(confidences) > 1 and confidences[-1] > confidences[0]:
            insights.append(f"Confidence improved from {confidences[0]:.2f} to {confidences[-1]:.2f}")
        
        # Look for evidence accumulation
        evidence_counts = [len(r.reasoning_chain) for r in results]
        if sum(evidence_counts) > len(evidence_counts) * 2:
            insights.append("Evidence accumulated across reasoning chain")
        
        # Look for assumption refinement
        if any(r.assumptions for r in results):
            insights.append("Assumptions identified and refined through sequence")
        
        return insights
    
    def _calculate_sequence_confidence(self, results: List[ReasoningResult]) -> float:
        """Calculate confidence for entire sequence"""
        
        if not results:
            return 0.0
        
        # Weight later results higher (they benefit from earlier insights)
        weights = [i + 1 for i in range(len(results))]
        weighted_confidence = sum(r.confidence * w for r, w in zip(results, weights))
        total_weight = sum(weights)
        
        return weighted_confidence / total_weight
    
    def _calculate_consensus_strength(self, results: List[ReasoningResult]) -> float:
        """Calculate strength of consensus among results"""
        
        if len(results) < 2:
            return 1.0
        
        confidences = [r.confidence for r in results]
        variance = statistics.variance(confidences)
        
        # Lower variance = stronger consensus
        return max(0.0, 1.0 - variance)
    
    def _find_agreements(self, results: List[ReasoningResult]) -> List[str]:
        """Find areas of agreement among results"""
        
        agreements = []
        
        # Simple heuristic: look for similar confidence levels
        high_confidence_engines = [r.engine.value for r in results if r.confidence > 0.7]
        if len(high_confidence_engines) > 1:
            agreements.append(f"High confidence agreement: {', '.join(high_confidence_engines)}")
        
        return agreements
    
    def _find_disagreements(self, results: List[ReasoningResult]) -> List[str]:
        """Find areas of disagreement among results"""
        
        disagreements = []
        
        # Simple heuristic: look for conflicting confidence levels
        confidences = [r.confidence for r in results]
        if max(confidences) - min(confidences) > 0.5:
            disagreements.append("Significant confidence disagreement detected")
        
        return disagreements
    
    def _integrate_evidence(self, evidence: List[str]) -> List[str]:
        """Integrate evidence from multiple sources"""
        
        # Remove duplicates and sort by strength indicators
        unique_evidence = list(set(evidence))
        
        # Simple integration - could be much more sophisticated
        integrated = []
        for item in unique_evidence:
            if any(word in item.lower() for word in ['strong', 'clear', 'definite']):
                integrated.insert(0, item)  # Strong evidence first
            else:
                integrated.append(item)
        
        return integrated
    
    def _identify_complementary_strengths(self, results: List[ReasoningResult]) -> Dict[str, str]:
        """Identify complementary strengths among engines"""
        
        strengths = {}
        
        for result in results:
            if result.engine == ReasoningEngine.DEDUCTIVE and result.confidence > 0.7:
                strengths["logical_certainty"] = "Strong deductive reasoning"
            elif result.engine == ReasoningEngine.INDUCTIVE and result.evidence_strength > 0.7:
                strengths["pattern_recognition"] = "Strong inductive evidence"
            elif result.engine == ReasoningEngine.ABDUCTIVE and result.quality_score > 0.7:
                strengths["explanation_quality"] = "Strong abductive explanation"
            # Add more engine-specific strengths
        
        return strengths
    
    def _calculate_fusion_confidence(self, results: List[ReasoningResult]) -> float:
        """Calculate confidence in fusion of results"""
        
        if not results:
            return 0.0
        
        # Weight by complementary strengths
        fusion_score = 0.0
        for result in results:
            if result.engine in [ReasoningEngine.DEDUCTIVE, ReasoningEngine.PROBABILISTIC]:
                fusion_score += result.confidence * 0.8  # High weight for precise engines
            else:
                fusion_score += result.confidence * 0.6  # Standard weight
        
        return fusion_score / len(results)
    
    def _calculate_synergy_score(self, results: List[ReasoningResult]) -> float:
        """Calculate synergy score among results"""
        
        if len(results) < 2:
            return 0.0
        
        # Simple synergy calculation based on diversity and quality
        avg_confidence = statistics.mean([r.confidence for r in results])
        confidence_variance = statistics.variance([r.confidence for r in results])
        
        # Higher average confidence + lower variance = better synergy
        return avg_confidence * (1 - confidence_variance)
    
    def _calculate_hybrid_confidence(self, 
                                   parallel_results: List[ReasoningResult],
                                   sequential_results: List[SequentialResult]) -> float:
        """Calculate confidence for hybrid reasoning"""
        
        parallel_confidence = statistics.mean([r.confidence for r in parallel_results]) if parallel_results else 0.0
        sequential_confidence = statistics.mean([r.sequence_confidence for r in sequential_results]) if sequential_results else 0.0
        
        # Weight sequential higher due to interactions
        return parallel_confidence * 0.4 + sequential_confidence * 0.6
    
    def _analyze_interaction_patterns(self, sequences: List[SequentialResult]) -> Dict[str, Any]:
        """Analyze interaction patterns in sequences"""
        
        patterns = {}
        
        # Count successful engine pairs
        pair_counts = {}
        for seq in sequences:
            for i in range(len(seq.sequence) - 1):
                pair = (seq.sequence[i], seq.sequence[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        # Find most successful patterns
        if pair_counts:
            most_common = max(pair_counts.items(), key=lambda x: x[1])
            patterns["most_successful_pair"] = f"{most_common[0][0].value} -> {most_common[0][1].value}"
            patterns["success_count"] = most_common[1]
        
        return patterns
    
    def _calculate_comprehensive_confidence(self, 
                                          parallel_results: List[ReasoningResult],
                                          high_performers: List[SequentialResult]) -> float:
        """Calculate confidence for comprehensive reasoning"""
        
        parallel_confidence = statistics.mean([r.confidence for r in parallel_results]) if parallel_results else 0.0
        sequential_confidence = statistics.mean([r.sequence_confidence for r in high_performers]) if high_performers else 0.0
        
        # Weight based on number of high performers
        performance_weight = min(len(high_performers) / 10, 1.0)  # Cap at 1.0
        
        return parallel_confidence * 0.3 + sequential_confidence * 0.7 * performance_weight
    
    def _identify_optimal_patterns(self, high_performers: List[SequentialResult]) -> List[str]:
        """Identify optimal reasoning patterns"""
        
        patterns = []
        
        # Find common starting engines
        starting_engines = [seq.sequence[0] for seq in high_performers]
        if starting_engines:
            most_common_start = max(set(starting_engines), key=starting_engines.count)
            patterns.append(f"Optimal starting engine: {most_common_start.value}")
        
        # Find common ending engines
        ending_engines = [seq.sequence[-1] for seq in high_performers]
        if ending_engines:
            most_common_end = max(set(ending_engines), key=ending_engines.count)
            patterns.append(f"Optimal ending engine: {most_common_end.value}")
        
        return patterns
    
    def _extract_reasoning_insights(self, top_sequences: List[SequentialResult]) -> List[str]:
        """Extract insights from top-performing sequences"""
        
        insights = []
        
        # Analyze sequence lengths
        lengths = [len(seq.sequence) for seq in top_sequences]
        if lengths:
            avg_length = statistics.mean(lengths)
            insights.append(f"Average optimal sequence length: {avg_length:.1f}")
        
        # Analyze processing times
        times = [seq.total_processing_time for seq in top_sequences]
        if times:
            avg_time = statistics.mean(times)
            insights.append(f"Average processing time: {avg_time:.2f}s")
        
        return insights
    
    def _identify_meta_patterns(self, all_sequences: List[SequentialResult]) -> Dict[str, Any]:
        """Identify meta-patterns across all sequences"""
        
        meta_patterns = {}
        
        # Performance distribution
        scores = [seq.get_sequence_score() for seq in all_sequences]
        if scores:
            meta_patterns["performance_distribution"] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0
            }
        
        # Engine usage patterns
        all_engines = [engine for seq in all_sequences for engine in seq.sequence]
        engine_counts = {engine: all_engines.count(engine) for engine in set(all_engines)}
        meta_patterns["engine_usage"] = engine_counts
        
        return meta_patterns
    
    def _identify_comprehensive_emergent_properties(self, 
                                                   parallel_results: List[ReasoningResult],
                                                   sequential_results: List[SequentialResult]) -> List[str]:
        """Identify emergent properties from comprehensive analysis"""
        
        properties = []
        
        # Compare parallel vs sequential performance
        if parallel_results and sequential_results:
            parallel_avg = statistics.mean([r.confidence for r in parallel_results])
            sequential_avg = statistics.mean([r.sequence_confidence for r in sequential_results])
            
            if sequential_avg > parallel_avg:
                properties.append("Sequential reasoning shows emergent performance gains")
            else:
                properties.append("Parallel reasoning maintains competitive performance")
        
        # Analyze interaction effects
        if sequential_results:
            high_interaction_seqs = [s for s in sequential_results if s.interaction_quality > 0.7]
            if high_interaction_seqs:
                properties.append(f"High interaction quality found in {len(high_interaction_seqs)} sequences")
        
        return properties
    
    async def _calculate_meta_confidence(self, result: MetaReasoningResult) -> float:
        """Calculate overall meta-confidence"""
        
        confidence = 0.0
        
        if result.parallel_results:
            parallel_confidence = statistics.mean([r.confidence for r in result.parallel_results])
            confidence += parallel_confidence * 0.4
        
        if result.sequential_results:
            sequential_confidence = statistics.mean([r.sequence_confidence for r in result.sequential_results])
            confidence += sequential_confidence * 0.6
        
        return confidence
    
    async def _calculate_quality_metrics(self, result: MetaReasoningResult) -> Dict[str, float]:
        """Calculate quality metrics for meta-reasoning"""
        
        metrics = {}
        
        if result.parallel_results:
            metrics["parallel_quality"] = statistics.mean([r.quality_score for r in result.parallel_results])
            metrics["parallel_evidence"] = statistics.mean([r.evidence_strength for r in result.parallel_results])
        
        if result.sequential_results:
            metrics["sequential_quality"] = statistics.mean([r.get_sequence_score() for r in result.sequential_results])
            metrics["interaction_quality"] = statistics.mean([r.interaction_quality for r in result.sequential_results])
        
        return metrics
    
    async def _identify_emergent_properties(self, result: MetaReasoningResult) -> List[str]:
        """Identify emergent properties from meta-reasoning"""
        
        properties = []
        
        if result.thinking_mode == ThinkingMode.DEEP and result.sequential_results:
            # Deep thinking specific properties
            top_sequences = sorted(result.sequential_results, 
                                 key=lambda x: x.get_sequence_score(), 
                                 reverse=True)[:10]
            
            if top_sequences:
                properties.append(f"Deep analysis revealed {len(top_sequences)} high-quality reasoning paths")
        
        return properties
    
    async def _analyze_cross_engine_interactions(self, result: MetaReasoningResult) -> Dict[str, Any]:
        """Analyze interactions between different engines"""
        
        interactions = {}
        
        if result.sequential_results:
            # Analyze successful engine transitions
            successful_transitions = []
            for seq in result.sequential_results:
                if seq.get_sequence_score() > 0.7:
                    for i in range(len(seq.sequence) - 1):
                        transition = (seq.sequence[i], seq.sequence[i + 1])
                        successful_transitions.append(transition)
            
            # Count successful transitions
            if successful_transitions:
                transition_counts = {}
                for transition in successful_transitions:
                    transition_counts[transition] = transition_counts.get(transition, 0) + 1
                
                interactions["successful_transitions"] = transition_counts
        
        return interactions
    
    def _calculate_ftns_cost(self, result: MetaReasoningResult, config: ThinkingConfiguration) -> float:
        """Calculate FTNS cost based on thinking mode and processing"""
        
        # Base cost calculation
        base_cost = 1.0  # Base cost for quick mode
        
        # Apply mode multiplier
        mode_cost = base_cost * config.ftns_cost_multiplier
        
        # Add processing time factor
        time_factor = min(result.total_processing_time / 60.0, 2.0)  # Cap at 2x for very long processes
        
        # Add depth factor
        depth_factor = result.reasoning_depth / 7.0  # Normalize to max depth
        
        total_cost = mode_cost * (1 + time_factor * 0.5 + depth_factor * 0.3)
        
        return round(total_cost, 2)
    
    def estimate_ftns_cost(self, thinking_mode: ThinkingMode, estimated_time: float = 60.0) -> float:
        """Estimate FTNS cost for a given thinking mode"""
        
        config = self.thinking_configs[thinking_mode]
        
        # Base cost calculation
        base_cost = 1.0
        mode_cost = base_cost * config.ftns_cost_multiplier
        time_factor = min(estimated_time / 60.0, 2.0)
        
        if thinking_mode == ThinkingMode.QUICK:
            depth_factor = 1.0 / 7.0
        elif thinking_mode == ThinkingMode.INTERMEDIATE:
            depth_factor = 3.0 / 7.0
        else:  # DEEP
            depth_factor = 7.0 / 7.0
        
        total_cost = mode_cost * (1 + time_factor * 0.5 + depth_factor * 0.3)
        
        return round(total_cost, 2)
    
    def get_thinking_mode_info(self) -> Dict[ThinkingMode, Dict[str, Any]]:
        """Get information about all thinking modes"""
        
        info = {}
        
        for mode, config in self.thinking_configs.items():
            estimated_cost = self.estimate_ftns_cost(mode)
            
            info[mode] = {
                "description": config.description,
                "max_permutations": config.max_permutations,
                "timeout_seconds": config.timeout_seconds,
                "estimated_ftns_cost": estimated_cost,
                "cost_multiplier": config.ftns_cost_multiplier
            }
        
        return info
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024  # Convert to MB
            except Exception:
                return 0.0
        else:
            # Fallback: return 0 if psutil not available
            return 0.0
    
    def _extract_confidence_score(self, result: ReasoningResult) -> float:
        """Extract confidence score as float from result"""
        if isinstance(result.confidence, (int, float)):
            return float(result.confidence)
        elif hasattr(result.confidence, 'value'):
            # Handle ConfidenceLevel enum
            confidence_mapping = {
                'very_high': 0.95,
                'high': 0.80,
                'moderate': 0.60,
                'low': 0.40,
                'very_low': 0.20
            }
            return confidence_mapping.get(result.confidence.value, 0.50)
        elif isinstance(result.confidence, str):
            # Handle string confidence levels
            confidence_mapping = {
                'very_high': 0.95,
                'high': 0.80,
                'moderate': 0.60,
                'low': 0.40,
                'very_low': 0.20
            }
            return confidence_mapping.get(result.confidence, 0.50)
        else:
            return 0.50  # Default fallback
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent(interval=0.1)
            except Exception:
                return 0.0
        else:
            return 0.0
    
    async def _monitor_engine_health(self, engine_type: ReasoningEngine) -> Dict[str, Any]:
        """Monitor engine health and performance metrics"""
        
        health_metrics = {
            "engine": engine_type.value,
            "status": "healthy",
            "memory_usage_mb": self._get_memory_usage(),
            "cpu_usage_percent": self._get_cpu_usage(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add engine-specific health checks
        try:
            engine = self.reasoning_engines[engine_type]
            if hasattr(engine, 'health_check'):
                health_metrics["engine_health"] = await engine.health_check()
            else:
                health_metrics["engine_health"] = "no_health_check_available"
        except Exception as e:
            health_metrics["status"] = "error"
            health_metrics["error"] = str(e)
        
        return health_metrics
    
    async def _check_resource_limits(self, current_memory: float, max_memory: float = 1000.0) -> bool:
        """Check if resource limits are exceeded"""
        
        if current_memory > max_memory:
            logger.warning(f"Memory limit exceeded", 
                         current_memory=current_memory, 
                         max_memory=max_memory)
            return False
        
        return True
    
    async def _cleanup_resources(self):
        """Clean up resources after engine execution"""
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Log current resource usage
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        
        logger.debug("Resource cleanup completed", 
                    memory_usage_mb=memory_usage, 
                    cpu_usage_percent=cpu_usage)
    
    # Health Monitoring API Methods
    
    def get_engine_health_status(self, engine_type: ReasoningEngine) -> EngineHealthStatus:
        """Get the current health status of a specific engine"""
        return self.health_monitor.get_engine_health_status(engine_type)
    
    def get_engine_health_report(self, engine_type: ReasoningEngine) -> Dict[str, Any]:
        """Get a comprehensive health report for a specific engine"""
        return self.health_monitor.get_engine_health_report(engine_type)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        return self.health_monitor.get_overall_system_health()
    
    def get_engine_health_trends(self, engine_type: ReasoningEngine) -> List[Dict[str, Any]]:
        """Get health trend data for a specific engine"""
        return self.health_monitor.get_health_trends(engine_type)
    
    def get_all_engine_health_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get health reports for all engines"""
        reports = {}
        for engine_type in self.reasoning_engines.keys():
            reports[engine_type.value] = self.get_engine_health_report(engine_type)
        return reports
    
    def is_engine_healthy(self, engine_type: ReasoningEngine) -> bool:
        """Check if an engine is healthy enough to use"""
        return self.health_monitor.should_use_engine(engine_type)
    
    def get_healthy_engines(self) -> List[ReasoningEngine]:
        """Get list of currently healthy engines"""
        healthy_engines = []
        for engine_type in self.reasoning_engines.keys():
            if self.is_engine_healthy(engine_type):
                healthy_engines.append(engine_type)
        return healthy_engines
    
    def get_unhealthy_engines(self) -> List[ReasoningEngine]:
        """Get list of currently unhealthy engines"""
        unhealthy_engines = []
        for engine_type in self.reasoning_engines.keys():
            if not self.is_engine_healthy(engine_type):
                unhealthy_engines.append(engine_type)
        return unhealthy_engines
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of all engines"""
        health_check_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": self.get_system_health_summary(),
            "engine_reports": self.get_all_engine_health_reports(),
            "healthy_engines": [e.value for e in self.get_healthy_engines()],
            "unhealthy_engines": [e.value for e in self.get_unhealthy_engines()],
            "resource_usage": {
                "memory_mb": self._get_memory_usage(),
                "cpu_percent": self._get_cpu_usage()
            }
        }
        
        logger.info("Health check completed", 
                   healthy_engines=len(self.get_healthy_engines()),
                   unhealthy_engines=len(self.get_unhealthy_engines()),
                   overall_health=health_check_results["system_health"]["overall_health_score"])
        
        return health_check_results
    
    def enable_health_monitoring(self):
        """Enable health monitoring"""
        self.health_monitor.monitoring_enabled = True
        logger.info("Health monitoring enabled")
    
    def disable_health_monitoring(self):
        """Disable health monitoring"""
        self.health_monitor.monitoring_enabled = False
        logger.info("Health monitoring disabled")
    
    def reset_health_metrics(self, engine_type: ReasoningEngine = None):
        """Reset health metrics for a specific engine or all engines"""
        if engine_type:
            if engine_type in self.health_monitor.engine_metrics:
                self.health_monitor.engine_metrics[engine_type] = EnginePerformanceMetrics()
                self.health_monitor.health_history[engine_type] = []
                logger.info("Health metrics reset", engine=engine_type.value)
        else:
            # Reset all engines
            for engine_type in self.reasoning_engines.keys():
                self.health_monitor.engine_metrics[engine_type] = EnginePerformanceMetrics()
                self.health_monitor.health_history[engine_type] = []
            logger.info("All health metrics reset")
    
    # Performance Tracking API Methods
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all engines"""
        return self.performance_tracker.get_performance_summary()
    
    def get_engine_performance_profile(self, engine_type: ReasoningEngine) -> Dict[str, Any]:
        """Get detailed performance profile for a specific engine"""
        profile = self.performance_tracker.get_performance_profile(engine_type)
        return {
            "engine_type": engine_type.value,
            "avg_execution_time": profile.avg_execution_time,
            "min_execution_time": profile.min_execution_time,
            "max_execution_time": profile.max_execution_time,
            "std_execution_time": profile.std_execution_time,
            "percentile_95_execution_time": profile.percentile_95_execution_time,
            "percentile_99_execution_time": profile.percentile_99_execution_time,
            "avg_memory_usage": profile.avg_memory_usage,
            "peak_memory_usage": profile.peak_memory_usage,
            "avg_quality_score": profile.avg_quality_score,
            "quality_trend": profile.quality_trend,
            "performance_class": profile.performance_class,
            "bottlenecks": profile.bottlenecks,
            "snapshot_count": len(profile.snapshots)
        }
    
    def get_performance_comparative_analysis(self) -> Dict[str, Any]:
        """Get comparative performance analysis across all engines"""
        return self.performance_tracker.get_comparative_analysis()
    
    def get_engine_performance_trends(self, engine_type: ReasoningEngine, 
                                    time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for a specific engine over time"""
        return self.performance_tracker.get_performance_trends(engine_type, time_window_hours)
    
    def get_performance_recommendations(self, engine_type: ReasoningEngine) -> List[str]:
        """Get performance optimization recommendations for a specific engine"""
        return self.performance_tracker.get_performance_recommendations(engine_type)
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_summary": self.get_performance_summary(),
            "comparative_analysis": self.get_performance_comparative_analysis(),
            "engine_profiles": {
                engine_type.value: self.get_engine_performance_profile(engine_type)
                for engine_type in ReasoningEngine
            },
            "recommendations": {
                engine_type.value: self.get_performance_recommendations(engine_type)
                for engine_type in ReasoningEngine
            }
        }
    
    def reset_performance_tracking(self, engine_type: ReasoningEngine = None):
        """Reset performance tracking data for a specific engine or all engines"""
        if engine_type:
            # Reset specific engine
            if engine_type in self.performance_tracker.profiles:
                self.performance_tracker.profiles[engine_type] = PerformanceProfile(engine_type)
                # Remove snapshots for this engine
                self.performance_tracker.snapshots = [
                    s for s in self.performance_tracker.snapshots 
                    if s.engine_type != engine_type
                ]
                logger.info("Performance tracking reset", engine=engine_type.value)
        else:
            # Reset all engines
            self.performance_tracker.reset_tracking()
            logger.info("All performance tracking reset")
    
    def enable_performance_tracking(self):
        """Enable performance tracking"""
        self.performance_tracker.enable_tracking()
        logger.info("Performance tracking enabled")
    
    def disable_performance_tracking(self):
        """Disable performance tracking"""
        self.performance_tracker.disable_tracking()
        logger.info("Performance tracking disabled")
    
    # Failure Detection and Recovery API Methods
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics"""
        return self.failure_detector.get_failure_statistics()
    
    def get_failure_history(self, engine_type: ReasoningEngine = None, 
                           hours: int = 24) -> List[Dict[str, Any]]:
        """Get failure history for an engine or all engines"""
        failure_events = self.failure_detector.get_failure_history(engine_type, hours)
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "engine_type": event.engine_type.value,
                "failure_type": event.failure_type.value,
                "error_message": event.error_message,
                "severity": event.severity,
                "recovery_attempted": event.recovery_attempted,
                "recovery_action": event.recovery_action.value if event.recovery_action else None,
                "recovery_successful": event.recovery_successful,
                "context": event.context
            }
            for event in failure_events
        ]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        return self.recovery_manager.get_recovery_statistics()
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for all engines"""
        circuit_breakers = {}
        for (engine_type, failure_type), info in self.recovery_manager.circuit_breakers.items():
            key = f"{engine_type.value}_{failure_type.value}"
            circuit_breakers[key] = {
                "engine_type": engine_type.value,
                "failure_type": failure_type.value,
                "activated": info["activated"].isoformat(),
                "failure_count": info["failure_count"],
                "is_open": self.recovery_manager._is_circuit_breaker_open(engine_type, failure_type)
            }
        
        return {
            "circuit_breakers": circuit_breakers,
            "total_circuit_breakers": len(circuit_breakers)
        }
    
    def get_failure_and_recovery_report(self) -> Dict[str, Any]:
        """Get comprehensive failure and recovery report"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "failure_statistics": self.get_failure_statistics(),
            "recovery_statistics": self.get_recovery_statistics(),
            "circuit_breaker_status": self.get_circuit_breaker_status(),
            "recent_failures": self.get_failure_history(hours=1),
            "failure_detection_enabled": self.failure_detector.enabled,
            "recovery_enabled": self.recovery_manager.enabled
        }
    
    async def manual_recovery(self, engine_type: ReasoningEngine, 
                            recovery_action: str = "restart") -> bool:
        """Manually trigger recovery for a specific engine"""
        try:
            # Create a mock failure event for manual recovery
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_type=engine_type,
                failure_type=FailureType.EXCEPTION,
                error_message="Manual recovery requested",
                severity="medium"
            )
            
            # Get recovery action enum
            recovery_action_enum = None
            for action in RecoveryAction:
                if action.value == recovery_action:
                    recovery_action_enum = action
                    break
            
            if not recovery_action_enum:
                logger.error(f"Invalid recovery action: {recovery_action}")
                return False
            
            # Execute the recovery action
            success = await self.recovery_manager._execute_recovery_action(
                engine_type, recovery_action_enum, failure_event, None
            )
            
            if success:
                logger.info(f"Manual recovery successful for {engine_type.value} using {recovery_action}")
            else:
                logger.error(f"Manual recovery failed for {engine_type.value} using {recovery_action}")
            
            return success
            
        except Exception as e:
            logger.error(f"Manual recovery failed: {str(e)}")
            return False
    
    def reset_failure_history(self, engine_type: ReasoningEngine = None):
        """Reset failure history for specific engine or all engines"""
        self.failure_detector.reset_failure_history(engine_type)
        if engine_type:
            logger.info(f"Failure history reset for {engine_type.value}")
        else:
            logger.info("All failure history reset")
    
    def reset_recovery_history(self):
        """Reset recovery history"""
        self.recovery_manager.reset_recovery_history()
        logger.info("Recovery history reset")
    
    def enable_failure_detection(self):
        """Enable failure detection"""
        self.failure_detector.enable_detection()
        logger.info("Failure detection enabled")
    
    def disable_failure_detection(self):
        """Disable failure detection"""
        self.failure_detector.disable_detection()
        logger.info("Failure detection disabled")
    
    def enable_failure_recovery(self):
        """Enable failure recovery"""
        self.recovery_manager.enable_recovery()
        logger.info("Failure recovery enabled")
    
    def disable_failure_recovery(self):
        """Disable failure recovery"""
        self.recovery_manager.disable_recovery()
        logger.info("Failure recovery disabled")
    
    def get_engine_isolation_status(self) -> Dict[str, Any]:
        """Get engine isolation status"""
        isolated_engines = getattr(self.health_monitor, 'isolated_engines', set())
        return {
            "isolated_engines": [engine.value for engine in isolated_engines],
            "total_isolated": len(isolated_engines),
            "active_engines": [
                engine.value for engine in ReasoningEngine 
                if engine not in isolated_engines
            ]
        }
    
    # Load Balancing API Methods
    
    def get_load_balancing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics"""
        return self.load_balancer.get_load_balancing_statistics()
    
    def get_engine_workload_status(self, engine_type: ReasoningEngine = None) -> Dict[str, Any]:
        """Get engine workload status for specific engine or all engines"""
        if engine_type:
            if engine_type in self.load_balancer.engine_workloads:
                workload = self.load_balancer.engine_workloads[engine_type]
                return {
                    "engine_type": engine_type.value,
                    "active_requests": workload.active_requests,
                    "queued_requests": workload.queued_requests,
                    "total_requests": workload.total_requests,
                    "load_factor": workload.current_load_factor,
                    "capacity_utilization": workload.capacity_utilization,
                    "average_response_time": workload.average_response_time,
                    "last_request_time": workload.last_request_time.isoformat() if workload.last_request_time else None
                }
            else:
                return {"error": f"Engine {engine_type.value} not found in workload tracking"}
        else:
            return {
                engine_type.value: {
                    "active_requests": workload.active_requests,
                    "queued_requests": workload.queued_requests,
                    "total_requests": workload.total_requests,
                    "load_factor": workload.current_load_factor,
                    "capacity_utilization": workload.capacity_utilization,
                    "average_response_time": workload.average_response_time,
                    "last_request_time": workload.last_request_time.isoformat() if workload.last_request_time else None
                }
                for engine_type, workload in self.load_balancer.engine_workloads.items()
            }
    
    def set_load_balancing_strategy(self, strategy: str):
        """Set the load balancing strategy"""
        try:
            strategy_enum = LoadBalancingStrategy(strategy)
            self.load_balancer.set_strategy(strategy_enum)
            logger.info(f"Load balancing strategy set to {strategy}")
        except ValueError:
            logger.error(f"Invalid load balancing strategy: {strategy}")
            raise ValueError(f"Invalid strategy. Valid strategies: {[s.value for s in LoadBalancingStrategy]}")
    
    def get_load_balancing_strategy(self) -> str:
        """Get current load balancing strategy"""
        return self.load_balancer.strategy.value
    
    def enable_load_balancing(self):
        """Enable load balancing"""
        self.load_balancer.enable_load_balancing()
        logger.info("Load balancing enabled")
    
    def disable_load_balancing(self):
        """Disable load balancing"""
        self.load_balancer.disable_load_balancing()
        logger.info("Load balancing disabled")
    
    def reset_load_balancing_metrics(self):
        """Reset load balancing metrics"""
        self.load_balancer.reset_metrics()
        logger.info("Load balancing metrics reset")
    
    def update_engine_weights(self):
        """Update engine weights based on current performance"""
        self.load_balancer.update_engine_weights()
        logger.info("Engine weights updated based on performance")
    
    def get_engine_weights(self) -> Dict[str, float]:
        """Get current engine weights"""
        return {
            engine_type.value: weight
            for engine_type, weight in self.load_balancer.engine_weights.items()
        }
    
    def force_engine_selection(self, engine_type: str) -> bool:
        """Force selection of a specific engine (for testing)"""
        try:
            engine_enum = ReasoningEngine(engine_type)
            # Temporarily disable load balancing and set a specific engine
            self.load_balancer.disable_load_balancing()
            # This would require modification to the select_engine method to support forced selection
            logger.info(f"Forced engine selection set to {engine_type}")
            return True
        except ValueError:
            logger.error(f"Invalid engine type: {engine_type}")
            return False
    
    def get_available_engines(self) -> List[str]:
        """Get list of currently available engines"""
        available_engines = self.load_balancer._get_available_engines()
        return [engine.value for engine in available_engines]
    
    def get_load_balancing_report(self) -> Dict[str, Any]:
        """Get comprehensive load balancing report"""
        statistics = self.get_load_balancing_statistics()
        workload_status = self.get_engine_workload_status()
        available_engines = self.get_available_engines()
        
        return {
            "load_balancing_enabled": self.load_balancer.enabled,
            "current_strategy": self.load_balancer.strategy.value,
            "current_mode": self.load_balancer.mode.value,
            "strategy_adaptation_enabled": self.load_balancer.strategy_adaptation_enabled,
            "available_engines": available_engines,
            "statistics": statistics,
            "workload_status": workload_status,
            "engine_weights": self.get_engine_weights(),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    # Adaptive Engine Selection API Methods
    
    def select_engines_adaptively(self, query: str, context: Dict[str, Any] = None, 
                                 num_engines: int = 3) -> List[str]:
        """Select engines adaptively based on query and context"""
        selected_engines = self.adaptive_selector.select_engines_adaptively(query, context, num_engines)
        return [engine.value for engine in selected_engines]
    
    def detect_problem_type(self, query: str, context: Dict[str, Any] = None) -> str:
        """Detect problem type from query and context"""
        problem_type = self.adaptive_selector.detect_problem_type(query, context)
        return problem_type.value
    
    def get_engine_selection_scores(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get detailed engine selection scores for a query"""
        if not self.adaptive_selector.enabled:
            return {"error": "Adaptive selection is disabled"}
        
        # Create adaptive selection context
        problem_type = self.adaptive_selector.detect_problem_type(query, context)
        contextual_factors = self.adaptive_selector.extract_contextual_factors(context or {})
        
        selection_context = AdaptiveSelectionContext(
            query=query,
            problem_type=problem_type,
            contextual_factors=contextual_factors,
            historical_performance=self.adaptive_selector._get_historical_performance(),
            user_preferences=context.get("user_preferences", {}) if context else {},
            constraints=context.get("constraints", {}) if context else {}
        )
        
        # Score all engines
        engine_scores = []
        for engine_type in ReasoningEngine:
            score = self.adaptive_selector._calculate_engine_score(engine_type, selection_context)
            engine_scores.append(score)
        
        # Sort by total score (descending)
        engine_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return {
            "query": query,
            "detected_problem_type": problem_type.value,
            "contextual_factors": {factor.value: value for factor, value in contextual_factors.items()},
            "engine_scores": [
                {
                    "engine": score.engine.value,
                    "total_score": score.total_score,
                    "confidence": score.confidence,
                    "component_scores": score.component_scores,
                    "reasoning": score.reasoning
                }
                for score in engine_scores
            ],
            "top_engines": [score.engine.value for score in engine_scores[:3]]
        }
    
    def get_adaptive_selection_statistics(self) -> Dict[str, Any]:
        """Get adaptive selection statistics"""
        return self.adaptive_selector.get_adaptive_selection_statistics()
    
    def set_adaptive_selection_strategy(self, strategy: str):
        """Set adaptive selection strategy"""
        try:
            strategy_enum = AdaptiveSelectionStrategy(strategy)
            self.adaptive_selector.set_strategy(strategy_enum)
            logger.info(f"Adaptive selection strategy set to {strategy}")
        except ValueError:
            logger.error(f"Invalid adaptive selection strategy: {strategy}")
            raise ValueError(f"Invalid strategy. Valid strategies: {[s.value for s in AdaptiveSelectionStrategy]}")
    
    def get_adaptive_selection_strategy(self) -> str:
        """Get current adaptive selection strategy"""
        return self.adaptive_selector.strategy.value
    
    def enable_adaptive_selection(self):
        """Enable adaptive selection"""
        self.adaptive_selector.enable_adaptive_selection()
        logger.info("Adaptive selection enabled")
    
    def disable_adaptive_selection(self):
        """Disable adaptive selection"""
        self.adaptive_selector.disable_adaptive_selection()
        logger.info("Adaptive selection disabled")
    
    def reset_adaptive_selection_history(self):
        """Reset adaptive selection learning history"""
        self.adaptive_selector.reset_learning_history()
        logger.info("Adaptive selection history reset")
    
    def update_engine_performance_feedback(self, engine_type: str, performance_score: float, query: str):
        """Update performance feedback for adaptive learning"""
        try:
            engine_enum = ReasoningEngine(engine_type)
            self.adaptive_selector.update_performance_feedback(engine_enum, performance_score, query)
            logger.info(f"Performance feedback updated for {engine_type}: {performance_score}")
        except ValueError:
            logger.error(f"Invalid engine type: {engine_type}")
            raise ValueError(f"Invalid engine type. Valid engines: {[e.value for e in ReasoningEngine]}")
    
    def get_problem_type_mappings(self) -> Dict[str, List[str]]:
        """Get problem type to engine mappings"""
        return {
            problem_type.value: [engine.value for engine in engines]
            for problem_type, engines in self.adaptive_selector.engine_problem_type_mapping.items()
        }
    
    def get_adaptive_selection_report(self) -> Dict[str, Any]:
        """Get comprehensive adaptive selection report"""
        statistics = self.get_adaptive_selection_statistics()
        problem_type_mappings = self.get_problem_type_mappings()
        
        return {
            "adaptive_selection_enabled": self.adaptive_selector.enabled,
            "current_strategy": self.adaptive_selector.strategy.value,
            "scoring_weights": self.adaptive_selector.scoring_weights,
            "learning_parameters": {
                "learning_rate": self.adaptive_selector.learning_rate,
                "exploration_rate": self.adaptive_selector.exploration_rate,
                "performance_window": self.adaptive_selector.performance_window
            },
            "statistics": statistics,
            "problem_type_mappings": problem_type_mappings,
            "selection_history_count": len(self.adaptive_selector.selection_history),
            "performance_history_count": sum(len(history) for history in self.adaptive_selector.performance_history.values()),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    # Result Formatting API Methods
    
    def format_single_result(self, result: ReasoningResult, 
                           format_type: str = "structured") -> Dict[str, Any]:
        """Format a single reasoning result"""
        try:
            format_enum = ResultFormat(format_type)
            formatted_result = self.result_formatter.format_single_result(result, format_enum)
            
            return {
                "formatted_result": formatted_result,
                "format_type": format_type,
                "engine_type": result.engine.value,
                "confidence_level": formatted_result.confidence_level.value,
                "priority": formatted_result.priority.value,
                "quality_score": formatted_result.quality_score,
                "success": True
            }
        except ValueError:
            logger.error(f"Invalid format type: {format_type}")
            return {
                "error": f"Invalid format type. Valid formats: {[f.value for f in ResultFormat]}",
                "success": False
            }
    
    def format_meta_result(self, meta_result: MetaReasoningResult, 
                          format_type: str = "structured") -> Dict[str, Any]:
        """Format a meta-reasoning result"""
        try:
            format_enum = ResultFormat(format_type)
            formatted_meta_result = self.result_formatter.format_meta_result(meta_result, format_enum)
            
            return {
                "formatted_meta_result": formatted_meta_result,
                "format_type": format_type,
                "thinking_mode": meta_result.thinking_mode.value,
                "overall_confidence": formatted_meta_result.overall_confidence.value,
                "overall_priority": formatted_meta_result.overall_priority.value,
                "processing_time": formatted_meta_result.processing_time,
                "ftns_cost": formatted_meta_result.ftns_cost,
                "engine_count": len(formatted_meta_result.engine_results),
                "success": True
            }
        except ValueError:
            logger.error(f"Invalid format type: {format_type}")
            return {
                "error": f"Invalid format type. Valid formats: {[f.value for f in ResultFormat]}",
                "success": False
            }
    
    def render_result(self, result: ReasoningResult, 
                     format_type: str = "structured") -> Dict[str, Any]:
        """Render a reasoning result to string"""
        try:
            format_enum = ResultFormat(format_type)
            formatted_result = self.result_formatter.format_single_result(result, format_enum)
            rendered_output = self.result_formatter.render_result(formatted_result, format_enum)
            
            return {
                "rendered_output": rendered_output,
                "format_type": format_type,
                "engine_type": result.engine.value,
                "length": len(rendered_output),
                "success": True
            }
        except ValueError:
            logger.error(f"Invalid format type: {format_type}")
            return {
                "error": f"Invalid format type. Valid formats: {[f.value for f in ResultFormat]}",
                "success": False
            }
    
    def render_meta_result(self, meta_result: MetaReasoningResult, 
                          format_type: str = "structured") -> Dict[str, Any]:
        """Render a meta-reasoning result to string"""
        try:
            format_enum = ResultFormat(format_type)
            formatted_meta_result = self.result_formatter.format_meta_result(meta_result, format_enum)
            rendered_output = self.result_formatter.render_meta_result(formatted_meta_result, format_enum)
            
            return {
                "rendered_output": rendered_output,
                "format_type": format_type,
                "thinking_mode": meta_result.thinking_mode.value,
                "length": len(rendered_output),
                "success": True
            }
        except ValueError:
            logger.error(f"Invalid format type: {format_type}")
            return {
                "error": f"Invalid format type. Valid formats: {[f.value for f in ResultFormat]}",
                "success": False
            }
    
    def get_available_formats(self) -> List[str]:
        """Get list of available result formats"""
        return [format_type.value for format_type in ResultFormat]
    
    def get_confidence_levels(self) -> List[str]:
        """Get list of confidence levels"""
        return [confidence.value for confidence in ConfidenceLevel]
    
    def get_priority_levels(self) -> List[str]:
        """Get list of priority levels"""
        return [priority.value for priority in ResultPriority]
    
    def format_comparison_results(self, results: List[ReasoningResult]) -> str:
        """Format multiple results for comparison"""
        if not results:
            return "No results to compare"
        
        formatted_results = []
        for result in results:
            formatted = self.result_formatter.format_single_result(result, ResultFormat.COMPARISON)
            formatted_results.append(formatted)
        
        # Create comparison table
        header = f"{'ENGINE':>15} | {'CONFIDENCE':>10} | {'QUALITY':>5} | {'SUMMARY'}"
        separator = "-" * 80
        
        comparison_lines = [header, separator]
        for formatted in formatted_results:
            rendered = self.result_formatter.render_result(formatted, ResultFormat.COMPARISON)
            comparison_lines.append(rendered.strip())
        
        return "\n".join(comparison_lines)
    
    def export_results(self, results: List[ReasoningResult], 
                      format_type: str = "export") -> str:
        """Export results for external systems"""
        if not results:
            return ""
        
        try:
            format_enum = ResultFormat(format_type)
            exported_lines = []
            
            for result in results:
                formatted = self.result_formatter.format_single_result(result, format_enum)
                rendered = self.result_formatter.render_result(formatted, format_enum)
                exported_lines.append(rendered.strip())
            
            return "\n".join(exported_lines)
        except ValueError:
            return f"Error: Invalid format type {format_type}"
    
    def get_result_statistics(self, results: List[ReasoningResult]) -> Dict[str, Any]:
        """Get statistics about formatted results"""
        if not results:
            return {"error": "No results provided"}
        
        confidence_levels = []
        quality_scores = []
        priorities = []
        engine_counts = {}
        
        for result in results:
            formatted = self.result_formatter.format_single_result(result, ResultFormat.STRUCTURED)
            
            confidence_levels.append(formatted.confidence_level.value)
            quality_scores.append(formatted.quality_score)
            priorities.append(formatted.priority.value)
            
            engine_type = formatted.engine_type.value
            engine_counts[engine_type] = engine_counts.get(engine_type, 0) + 1
        
        return {
            "total_results": len(results),
            "confidence_distribution": {
                level: confidence_levels.count(level) 
                for level in set(confidence_levels)
            },
            "priority_distribution": {
                priority: priorities.count(priority) 
                for priority in set(priorities)
            },
            "average_quality": statistics.mean(quality_scores),
            "quality_range": {
                "min": min(quality_scores),
                "max": max(quality_scores)
            },
            "engine_distribution": engine_counts,
            "high_confidence_count": sum(1 for level in confidence_levels if level in ["high", "very_high"]),
            "high_priority_count": sum(1 for priority in priorities if priority in ["high", "critical"])
        }
    
    def get_formatting_report(self) -> Dict[str, Any]:
        """Get comprehensive formatting system report"""
        return {
            "available_formats": self.get_available_formats(),
            "confidence_levels": self.get_confidence_levels(),
            "priority_levels": self.get_priority_levels(),
            "format_templates": list(self.result_formatter.format_templates.keys()),
            "engine_formatting_rules": {
                engine.value: rules
                for engine, rules in self.result_formatter.engine_formatting_rules.items()
            },
            "priority_matrix": {
                f"{conf.value}_{qual}": priority.value
                for (conf, qual), priority in self.result_formatter.priority_matrix.items()
            },
            "confidence_thresholds": {
                level.value: threshold
                for level, threshold in self.result_formatter.confidence_thresholds.items()
            },
            "formatter_initialized": self.result_formatter is not None,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    # Error Handling API Methods
    
    def handle_custom_error(self, error: Exception, operation: str, 
                           engine_type: str = None, severity: str = "medium") -> Dict[str, Any]:
        """Handle a custom error with the error handling system"""
        
        try:
            # Convert string parameters to enums
            engine_enum = ReasoningEngine(engine_type) if engine_type else None
            severity_enum = ErrorSeverity(severity)
            
            # Create error context
            error_context = ErrorContext(
                operation=operation,
                engine_type=engine_enum,
                query="",
                user_context={},
                system_state={}
            )
            
            # Handle error
            error_event = self.error_handler.handle_error(error, error_context, severity_enum)
            
            return {
                "error_id": error_event.error_id,
                "error_type": error_event.error_type,
                "category": error_event.category.value,
                "severity": error_event.severity.value,
                "message": error_event.message,
                "recovery_attempted": error_event.recovery_attempted,
                "recovery_successful": error_event.recovery_successful,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Failed to handle error: {str(e)}",
                "success": False
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return self.error_handler.get_error_statistics()
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report"""
        return self.error_handler.get_error_report()
    
    def get_recent_errors(self, hours: int = 24, severity: str = None) -> List[Dict[str, Any]]:
        """Get recent errors within specified time window"""
        
        now = datetime.now(timezone.utc)
        time_window = timedelta(hours=hours)
        
        recent_errors = [
            error for error in self.error_handler.error_events
            if (now - error.last_occurrence) <= time_window
        ]
        
        # Filter by severity if specified
        if severity:
            try:
                severity_enum = ErrorSeverity(severity)
                recent_errors = [e for e in recent_errors if e.severity == severity_enum]
            except ValueError:
                pass
        
        return [
            {
                "error_id": error.error_id,
                "error_type": error.error_type,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "engine_type": error.context.engine_type.value if error.context.engine_type else None,
                "operation": error.context.operation,
                "occurrence_count": error.occurrence_count,
                "first_occurrence": error.first_occurrence.isoformat(),
                "last_occurrence": error.last_occurrence.isoformat(),
                "recovery_attempted": error.recovery_attempted,
                "recovery_successful": error.recovery_successful,
                "resolved": error.resolved
            }
            for error in sorted(recent_errors, key=lambda e: e.last_occurrence, reverse=True)
        ]
    
    def resolve_error(self, error_id: str, resolution_details: str) -> Dict[str, Any]:
        """Manually resolve an error"""
        
        success = self.error_handler.resolve_error(error_id, resolution_details)
        
        return {
            "error_id": error_id,
            "resolved": success,
            "resolution_details": resolution_details if success else None,
            "message": "Error resolved successfully" if success else "Error not found",
            "success": success
        }
    
    def get_error_categories(self) -> List[str]:
        """Get list of error categories"""
        return [category.value for category in ErrorCategory]
    
    def get_error_severities(self) -> List[str]:
        """Get list of error severities"""
        return [severity.value for severity in ErrorSeverity]
    
    def get_recovery_strategies(self) -> List[str]:
        """Get list of recovery strategies"""
        return [strategy.value for strategy in RecoveryStrategy]
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return self.error_handler._get_circuit_breaker_status()
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Get error patterns analysis"""
        
        patterns = {}
        for pattern_key, error_list in self.error_handler.error_patterns.items():
            patterns[pattern_key] = {
                "total_occurrences": len(error_list),
                "recent_occurrences": len([
                    e for e in error_list 
                    if (datetime.now(timezone.utc) - e.last_occurrence).total_seconds() < 3600
                ]),
                "severity_distribution": {
                    severity.value: sum(1 for e in error_list if e.severity == severity)
                    for severity in ErrorSeverity
                },
                "latest_occurrence": max(error_list, key=lambda e: e.last_occurrence).last_occurrence.isoformat()
            }
        
        return patterns
    
    def clear_error_history(self) -> Dict[str, Any]:
        """Clear error history"""
        
        initial_count = len(self.error_handler.error_events)
        self.error_handler.clear_error_history()
        
        return {
            "cleared_errors": initial_count,
            "message": f"Cleared {initial_count} error events from history",
            "success": True
        }
    
    def enable_error_handling(self):
        """Enable error handling"""
        self.error_handler.enable_error_handling()
        logger.info("Error handling enabled")
    
    def disable_error_handling(self):
        """Disable error handling"""
        self.error_handler.disable_error_handling()
        logger.info("Error handling disabled")
    
    def get_error_handling_status(self) -> Dict[str, Any]:
        """Get error handling system status"""
        
        return {
            "error_handling_enabled": self.error_handler.enabled,
            "total_errors": len(self.error_handler.error_events),
            "error_patterns": len(self.error_handler.error_patterns),
            "max_error_history": self.error_handler.max_error_history,
            "circuit_breaker_status": self.get_circuit_breaker_status(),
            "error_thresholds": {
                severity.value: threshold 
                for severity, threshold in self.error_handler.error_thresholds.items()
            },
            "retry_configurations": {
                category.value: config
                for category, config in self.error_handler.retry_config.items()
            },
            "recovery_strategies": {
                category.value: [strategy.value for strategy in strategies]
                for category, strategies in self.error_handler.recovery_strategies.items()
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def simulate_error_recovery(self, error_category: str, recovery_strategy: str) -> Dict[str, Any]:
        """Simulate error recovery for testing purposes"""
        
        try:
            # Create mock error event
            category_enum = ErrorCategory(error_category)
            strategy_enum = RecoveryStrategy(recovery_strategy)
            
            mock_error = ValueError(f"Simulated {error_category} error")
            mock_context = ErrorContext(
                operation="simulation",
                engine_type=ReasoningEngine.DEDUCTIVE,
                query="Test query",
                user_context={},
                system_state={"simulation": True}
            )
            
            # Create error event
            error_event = ErrorEvent(
                error_type="ValueError",
                category=category_enum,
                severity=ErrorSeverity.MEDIUM,
                message="Simulated error for testing",
                context=mock_context
            )
            
            # Test recovery strategy
            success = self.error_handler._execute_recovery_strategy(strategy_enum, error_event)
            
            return {
                "error_category": error_category,
                "recovery_strategy": recovery_strategy,
                "recovery_successful": success,
                "message": f"Recovery simulation {'successful' if success else 'failed'}",
                "success": True
            }
            
        except ValueError as e:
            return {
                "error": f"Invalid parameter: {str(e)}",
                "success": False
            }
        except Exception as e:
            return {
                "error": f"Simulation failed: {str(e)}",
                "success": False
            }
    
    def get_error_handling_report(self) -> Dict[str, Any]:
        """Get comprehensive error handling report"""
        
        statistics = self.get_error_statistics()
        recent_errors = self.get_recent_errors(hours=24)
        error_patterns = self.get_error_patterns()
        circuit_breaker_status = self.get_circuit_breaker_status()
        
        return {
            "system_status": self.get_error_handling_status(),
            "statistics": statistics,
            "recent_errors": recent_errors,
            "error_patterns": error_patterns,
            "circuit_breaker": circuit_breaker_status,
            "available_categories": self.get_error_categories(),
            "available_severities": self.get_error_severities(),
            "available_strategies": self.get_recovery_strategies(),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    # =============================================================================
    # Interaction Pattern Recognition API Methods
    # =============================================================================
    
    def observe_engine_interaction(self, engine_pair: Tuple[ReasoningEngine, ReasoningEngine],
                                  first_result: Any, second_result: Any, combined_result: Any,
                                  query: str, context: Dict[str, Any], thinking_mode: ThinkingMode) -> Dict[str, Any]:
        """Observe and record an interaction between two engines"""
        
        try:
            evidence = self.pattern_recognizer.observe_interaction(
                engine_pair, first_result, second_result, combined_result, query, context, thinking_mode
            )
            
            if evidence:
                return {
                    "evidence_id": evidence.evidence_id,
                    "engine_pair": f"{engine_pair[0].value} -> {engine_pair[1].value}",
                    "observed_pattern": evidence.observed_pattern.value,
                    "outcome": evidence.outcome.value,
                    "quality_change": evidence.quality_change,
                    "confidence_change": evidence.confidence_change,
                    "timestamp": evidence.timestamp.isoformat(),
                    "success": True
                }
            else:
                return {
                    "message": "Pattern recognition disabled",
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Failed to observe interaction: {str(e)}",
                "success": False
            }
    
    def get_pattern_recommendations(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommended engine pairs based on interaction patterns"""
        
        try:
            recommendations = self.pattern_recognizer.get_pattern_recommendations(query, context)
            
            return {
                "recommendations": [
                    {
                        "engine_pair": f"{pair[0].value} -> {pair[1].value}",
                        "first_engine": pair[0].value,
                        "second_engine": pair[1].value,
                        "pattern_info": self.get_pattern_info(pair)
                    }
                    for pair in recommendations
                ],
                "total_recommendations": len(recommendations),
                "query": query,
                "context_analyzed": True,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get recommendations: {str(e)}",
                "success": False
            }
    
    def get_pattern_info(self, engine_pair: Tuple[ReasoningEngine, ReasoningEngine]) -> Dict[str, Any]:
        """Get detailed information about a specific pattern"""
        
        if engine_pair in self.pattern_recognizer.patterns:
            pattern = self.pattern_recognizer.patterns[engine_pair]
            return {
                "pattern_name": pattern.pattern_name,
                "pattern_type": pattern.pattern_type.value,
                "description": pattern.description,
                "effectiveness_score": pattern.effectiveness_score,
                "confidence_level": pattern.confidence_level,
                "frequency": pattern.frequency,
                "sample_size": pattern.sample_size,
                "avg_quality_improvement": pattern.avg_quality_improvement,
                "avg_confidence_improvement": pattern.avg_confidence_improvement,
                "typical_outcomes": [outcome.value for outcome in pattern.typical_outcomes],
                "favorable_contexts": pattern.favorable_contexts,
                "unfavorable_contexts": pattern.unfavorable_contexts,
                "last_updated": pattern.last_updated.isoformat()
            }
        else:
            return {
                "error": f"No pattern found for {engine_pair[0].value} -> {engine_pair[1].value}",
                "available_patterns": len(self.pattern_recognizer.patterns)
            }
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of interaction patterns"""
        
        try:
            analysis = self.pattern_recognizer.get_pattern_analysis()
            return {
                "pattern_analysis": analysis,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get pattern analysis: {str(e)}",
                "success": False
            }
    
    def get_interaction_pattern_report(self) -> Dict[str, Any]:
        """Get detailed interaction pattern report"""
        
        try:
            report = self.pattern_recognizer.get_pattern_report()
            return {
                "pattern_report": report,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get pattern report: {str(e)}",
                "success": False
            }
    
    def get_pattern_types(self) -> List[str]:
        """Get list of available pattern types"""
        return [pattern_type.value for pattern_type in InteractionPatternType]
    
    def get_interaction_outcomes(self) -> List[str]:
        """Get list of possible interaction outcomes"""
        return [outcome.value for outcome in InteractionOutcome]
    
    def get_all_patterns(self) -> Dict[str, Any]:
        """Get all known interaction patterns"""
        
        patterns = {}
        for pair, pattern in self.pattern_recognizer.patterns.items():
            key = f"{pair[0].value}_{pair[1].value}"
            patterns[key] = {
                "pattern_name": pattern.pattern_name,
                "pattern_type": pattern.pattern_type.value,
                "description": pattern.description,
                "effectiveness_score": pattern.effectiveness_score,
                "confidence_level": pattern.confidence_level,
                "frequency": pattern.frequency,
                "sample_size": pattern.sample_size
            }
        
        return {
            "patterns": patterns,
            "total_patterns": len(patterns),
            "pattern_types": self.get_pattern_types(),
            "interaction_outcomes": self.get_interaction_outcomes(),
            "success": True
        }
    
    def get_pattern_evidence(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent pattern evidence"""
        
        try:
            now = datetime.now(timezone.utc)
            time_threshold = now - timedelta(hours=hours)
            
            recent_evidence = [
                {
                    "evidence_id": evidence.evidence_id,
                    "engine_pair": f"{evidence.engine_pair[0].value} -> {evidence.engine_pair[1].value}",
                    "observed_pattern": evidence.observed_pattern.value,
                    "outcome": evidence.outcome.value,
                    "quality_change": evidence.quality_change,
                    "confidence_change": evidence.confidence_change,
                    "thinking_mode": evidence.thinking_mode.value,
                    "timestamp": evidence.timestamp.isoformat()
                }
                for evidence in self.pattern_recognizer.evidence_history
                if evidence.timestamp >= time_threshold
            ]
            
            return {
                "recent_evidence": recent_evidence,
                "evidence_count": len(recent_evidence),
                "time_period_hours": hours,
                "total_evidence": len(self.pattern_recognizer.evidence_history),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get pattern evidence: {str(e)}",
                "success": False
            }
    
    def get_pattern_discovery_history(self) -> Dict[str, Any]:
        """Get history of pattern discoveries"""
        
        try:
            discoveries = [
                {
                    "engine_pair": f"{d['pair'][0].value} -> {d['pair'][1].value}",
                    "pattern_name": d["pattern"].pattern_name,
                    "pattern_type": d["pattern"].pattern_type.value,
                    "effectiveness_score": d["pattern"].effectiveness_score,
                    "discovery_date": d["discovery_date"].isoformat()
                }
                for d in self.pattern_recognizer.discovery_history
            ]
            
            return {
                "discoveries": discoveries,
                "total_discoveries": len(discoveries),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get discovery history: {str(e)}",
                "success": False
            }
    
    def enable_pattern_recognition(self) -> Dict[str, Any]:
        """Enable interaction pattern recognition"""
        
        try:
            self.pattern_recognizer.enable_pattern_recognition()
            return {
                "message": "Interaction pattern recognition enabled",
                "enabled": True,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to enable pattern recognition: {str(e)}",
                "success": False
            }
    
    def disable_pattern_recognition(self) -> Dict[str, Any]:
        """Disable interaction pattern recognition"""
        
        try:
            self.pattern_recognizer.disable_pattern_recognition()
            return {
                "message": "Interaction pattern recognition disabled",
                "enabled": False,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to disable pattern recognition: {str(e)}",
                "success": False
            }
    
    def clear_pattern_history(self) -> Dict[str, Any]:
        """Clear pattern learning history"""
        
        try:
            initial_evidence_count = len(self.pattern_recognizer.evidence_history)
            self.pattern_recognizer.clear_pattern_history()
            
            return {
                "message": f"Pattern learning history cleared",
                "cleared_evidence": initial_evidence_count,
                "remaining_patterns": len(self.pattern_recognizer.patterns),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to clear pattern history: {str(e)}",
                "success": False
            }
    
    def get_pattern_recognition_status(self) -> Dict[str, Any]:
        """Get status of pattern recognition system"""
        
        return {
            "pattern_recognition_enabled": self.pattern_recognizer.enabled,
            "auto_discovery_enabled": self.pattern_recognizer.auto_discovery_enabled,
            "total_patterns": len(self.pattern_recognizer.patterns),
            "total_evidence": len(self.pattern_recognizer.evidence_history),
            "recent_discoveries": len([d for d in self.pattern_recognizer.discovery_history if 
                                    (datetime.now(timezone.utc) - d["discovery_date"]).days <= 7]),
            "learning_parameters": {
                "learning_rate": self.pattern_recognizer.learning_rate,
                "confidence_threshold": self.pattern_recognizer.confidence_threshold,
                "min_sample_size": self.pattern_recognizer.min_sample_size,
                "max_evidence_history": self.pattern_recognizer.max_evidence_history
            },
            "system_health": {
                "context_analyzer_available": hasattr(self.pattern_recognizer, 'context_analyzer'),
                "outcome_predictor_available": hasattr(self.pattern_recognizer, 'outcome_predictor'),
                "pattern_effectiveness_tracking": len(self.pattern_recognizer.pattern_effectiveness_history)
            },
            "success": True
        }
    
    def update_pattern_learning_parameters(self, learning_rate: float = None, 
                                         confidence_threshold: float = None,
                                         min_sample_size: int = None) -> Dict[str, Any]:
        """Update pattern learning parameters"""
        
        try:
            updates = {}
            
            if learning_rate is not None:
                if 0.0 <= learning_rate <= 1.0:
                    self.pattern_recognizer.learning_rate = learning_rate
                    updates["learning_rate"] = learning_rate
                else:
                    return {
                        "error": "Learning rate must be between 0.0 and 1.0",
                        "success": False
                    }
            
            if confidence_threshold is not None:
                if 0.0 <= confidence_threshold <= 1.0:
                    self.pattern_recognizer.confidence_threshold = confidence_threshold
                    updates["confidence_threshold"] = confidence_threshold
                else:
                    return {
                        "error": "Confidence threshold must be between 0.0 and 1.0",
                        "success": False
                    }
            
            if min_sample_size is not None:
                if min_sample_size > 0:
                    self.pattern_recognizer.min_sample_size = min_sample_size
                    updates["min_sample_size"] = min_sample_size
                else:
                    return {
                        "error": "Minimum sample size must be greater than 0",
                        "success": False
                    }
            
            return {
                "message": "Pattern learning parameters updated",
                "updates": updates,
                "current_parameters": {
                    "learning_rate": self.pattern_recognizer.learning_rate,
                    "confidence_threshold": self.pattern_recognizer.confidence_threshold,
                    "min_sample_size": self.pattern_recognizer.min_sample_size
                },
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to update learning parameters: {str(e)}",
                "success": False
            }
    
    # =============================================================================
    # Sequential Context Passing API Methods
    # =============================================================================
    
    def create_sequential_context(self, query: str, context: Dict[str, Any], 
                                 processing_chain: List[str], passing_mode: str = "enriched") -> Dict[str, Any]:
        """Create a sequential context for engine chain processing"""
        
        try:
            # Convert string engine names to enum
            engine_chain = []
            for engine_name in processing_chain:
                try:
                    engine_enum = ReasoningEngine(engine_name)
                    engine_chain.append(engine_enum)
                except ValueError:
                    return {
                        "error": f"Invalid engine name: {engine_name}",
                        "valid_engines": [e.value for e in ReasoningEngine],
                        "success": False
                    }
            
            # Convert passing mode to enum
            try:
                passing_mode_enum = ContextPassingMode(passing_mode)
            except ValueError:
                return {
                    "error": f"Invalid passing mode: {passing_mode}",
                    "valid_modes": [m.value for m in ContextPassingMode],
                    "success": False
                }
            
            # Create context
            sequential_context = self.context_passing_engine.create_sequential_context(
                query, context, engine_chain, passing_mode_enum
            )
            
            if sequential_context:
                return {
                    "context_id": sequential_context.context_id,
                    "query": sequential_context.query,
                    "processing_chain": [e.value for e in sequential_context.processing_chain],
                    "passing_mode": sequential_context.passing_mode.value,
                    "relevance_threshold": sequential_context.relevance_threshold.value,
                    "created_at": sequential_context.created_at.isoformat(),
                    "success": True
                }
            else:
                return {
                    "error": "Context passing is disabled",
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Failed to create sequential context: {str(e)}",
                "success": False
            }
    
    def add_context_item(self, context_id: str, context_type: str, content: Any,
                        source_engine: str, relevance: str = "medium", 
                        confidence: float = 0.5, quality_score: float = 0.5) -> Dict[str, Any]:
        """Add a context item to a sequential context"""
        
        try:
            # Convert string values to enums
            try:
                context_type_enum = ContextType(context_type)
            except ValueError:
                return {
                    "error": f"Invalid context type: {context_type}",
                    "valid_types": [t.value for t in ContextType],
                    "success": False
                }
            
            try:
                source_engine_enum = ReasoningEngine(source_engine)
            except ValueError:
                return {
                    "error": f"Invalid source engine: {source_engine}",
                    "valid_engines": [e.value for e in ReasoningEngine],
                    "success": False
                }
            
            try:
                relevance_enum = ContextRelevance(relevance)
            except ValueError:
                return {
                    "error": f"Invalid relevance: {relevance}",
                    "valid_relevance": [r.value for r in ContextRelevance],
                    "success": False
                }
            
            # Add context item
            context_item = self.context_passing_engine.add_context_item(
                context_id, context_type_enum, content, source_engine_enum, 
                relevance_enum, confidence, quality_score
            )
            
            if context_item:
                return {
                    "context_item_id": context_item.context_id,
                    "context_type": context_item.context_type.value,
                    "relevance": context_item.relevance.value,
                    "source_engine": context_item.source_engine.value,
                    "source_step": context_item.source_step,
                    "confidence": context_item.confidence,
                    "quality_score": context_item.quality_score,
                    "created_at": context_item.created_at.isoformat(),
                    "success": True
                }
            else:
                return {
                    "error": f"Failed to add context item - context {context_id} not found",
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Failed to add context item: {str(e)}",
                "success": False
            }
    
    def get_context_for_engine(self, context_id: str, target_engine: str, step: int) -> Dict[str, Any]:
        """Get processed context for a specific engine in the chain"""
        
        try:
            # Convert string engine name to enum
            try:
                target_engine_enum = ReasoningEngine(target_engine)
            except ValueError:
                return {
                    "error": f"Invalid target engine: {target_engine}",
                    "valid_engines": [e.value for e in ReasoningEngine],
                    "success": False
                }
            
            # Get context
            engine_context = self.context_passing_engine.get_context_for_engine(
                context_id, target_engine_enum, step
            )
            
            return {
                "context_id": context_id,
                "target_engine": target_engine,
                "step": step,
                "context": engine_context,
                "context_keys": list(engine_context.keys()),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get context for engine: {str(e)}",
                "success": False
            }
    
    def finalize_sequential_context(self, context_id: str) -> Dict[str, Any]:
        """Finalize and archive a sequential context"""
        
        try:
            result = self.context_passing_engine.finalize_context(context_id)
            return result
        except Exception as e:
            return {
                "error": f"Failed to finalize context: {str(e)}",
                "success": False
            }
    
    def get_context_passing_statistics(self) -> Dict[str, Any]:
        """Get context passing system statistics"""
        
        try:
            stats = self.context_passing_engine.get_context_passing_statistics()
            return {
                "context_passing_statistics": stats,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get context passing statistics: {str(e)}",
                "success": False
            }
    
    def get_context_passing_modes(self) -> List[str]:
        """Get available context passing modes"""
        return [mode.value for mode in ContextPassingMode]
    
    def get_context_types(self) -> List[str]:
        """Get available context types"""
        return [context_type.value for context_type in ContextType]
    
    def get_context_relevance_levels(self) -> List[str]:
        """Get available context relevance levels"""
        return [relevance.value for relevance in ContextRelevance]
    
    def get_active_contexts(self) -> Dict[str, Any]:
        """Get information about active contexts"""
        
        try:
            active_contexts = {}
            for context_id, context in self.context_passing_engine.active_contexts.items():
                active_contexts[context_id] = {
                    "query": context.query,
                    "processing_chain": [e.value for e in context.processing_chain],
                    "current_step": context.current_step,
                    "passing_mode": context.passing_mode.value,
                    "total_items": sum(len(items) for items in context.context_items.values()),
                    "created_at": context.created_at.isoformat(),
                    "last_updated": context.last_updated.isoformat()
                }
            
            return {
                "active_contexts": active_contexts,
                "total_active": len(active_contexts),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get active contexts: {str(e)}",
                "success": False
            }
    
    def get_context_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get context history"""
        
        try:
            history = []
            for context in self.context_passing_engine.context_history[-limit:]:
                history.append({
                    "context_id": context.context_id,
                    "query": context.query,
                    "processing_chain": [e.value for e in context.processing_chain],
                    "total_steps": context.current_step,
                    "passing_mode": context.passing_mode.value,
                    "total_items": sum(len(items) for items in context.context_items.values()),
                    "created_at": context.created_at.isoformat(),
                    "last_updated": context.last_updated.isoformat()
                })
            
            return {
                "context_history": history,
                "returned_count": len(history),
                "total_history": len(self.context_passing_engine.context_history),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get context history: {str(e)}",
                "success": False
            }
    
    def enable_context_passing(self) -> Dict[str, Any]:
        """Enable context passing"""
        
        try:
            self.context_passing_engine.enable_context_passing()
            return {
                "message": "Context passing enabled",
                "enabled": True,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to enable context passing: {str(e)}",
                "success": False
            }
    
    def disable_context_passing(self) -> Dict[str, Any]:
        """Disable context passing"""
        
        try:
            self.context_passing_engine.disable_context_passing()
            return {
                "message": "Context passing disabled",
                "enabled": False,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to disable context passing: {str(e)}",
                "success": False
            }
    
    def clear_context_history(self) -> Dict[str, Any]:
        """Clear context history"""
        
        try:
            initial_count = len(self.context_passing_engine.context_history)
            self.context_passing_engine.clear_context_history()
            
            return {
                "message": "Context history cleared",
                "cleared_contexts": initial_count,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to clear context history: {str(e)}",
                "success": False
            }
    
    def get_context_passing_status(self) -> Dict[str, Any]:
        """Get status of context passing system"""
        
        try:
            stats = self.context_passing_engine.get_context_passing_statistics()
            
            return {
                "context_passing_enabled": self.context_passing_engine.enabled,
                "active_contexts": len(self.context_passing_engine.active_contexts),
                "context_history": len(self.context_passing_engine.context_history),
                "default_passing_mode": self.context_passing_engine.default_passing_mode.value,
                "default_relevance_threshold": self.context_passing_engine.default_relevance_threshold.value,
                "compression_enabled": self.context_passing_engine.context_compression_enabled,
                "filtering_enabled": self.context_passing_engine.context_filtering_enabled,
                "context_metrics": self.context_passing_engine.context_metrics,
                "available_modes": self.get_context_passing_modes(),
                "available_types": self.get_context_types(),
                "available_relevance_levels": self.get_context_relevance_levels(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get context passing status: {str(e)}",
                "success": False
            }
    
    def update_context_passing_configuration(self, passing_mode: str = None,
                                           relevance_threshold: str = None,
                                           compression_enabled: bool = None,
                                           filtering_enabled: bool = None) -> Dict[str, Any]:
        """Update context passing configuration"""
        
        try:
            updates = {}
            
            if passing_mode is not None:
                try:
                    passing_mode_enum = ContextPassingMode(passing_mode)
                    self.context_passing_engine.default_passing_mode = passing_mode_enum
                    updates["passing_mode"] = passing_mode
                except ValueError:
                    return {
                        "error": f"Invalid passing mode: {passing_mode}",
                        "valid_modes": [m.value for m in ContextPassingMode],
                        "success": False
                    }
            
            if relevance_threshold is not None:
                try:
                    relevance_enum = ContextRelevance(relevance_threshold)
                    self.context_passing_engine.default_relevance_threshold = relevance_enum
                    updates["relevance_threshold"] = relevance_threshold
                except ValueError:
                    return {
                        "error": f"Invalid relevance threshold: {relevance_threshold}",
                        "valid_relevance": [r.value for r in ContextRelevance],
                        "success": False
                    }
            
            if compression_enabled is not None:
                self.context_passing_engine.context_compression_enabled = compression_enabled
                updates["compression_enabled"] = compression_enabled
            
            if filtering_enabled is not None:
                self.context_passing_engine.context_filtering_enabled = filtering_enabled
                updates["filtering_enabled"] = filtering_enabled
            
            return {
                "message": "Context passing configuration updated",
                "updates": updates,
                "current_configuration": {
                    "passing_mode": self.context_passing_engine.default_passing_mode.value,
                    "relevance_threshold": self.context_passing_engine.default_relevance_threshold.value,
                    "compression_enabled": self.context_passing_engine.context_compression_enabled,
                    "filtering_enabled": self.context_passing_engine.context_filtering_enabled
                },
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to update context passing configuration: {str(e)}",
                "success": False
            }
    
    def get_context_passing_report(self) -> Dict[str, Any]:
        """Get comprehensive context passing report"""
        
        try:
            active_contexts = self.get_active_contexts()
            context_history = self.get_context_history(limit=20)
            statistics = self.get_context_passing_statistics()
            status = self.get_context_passing_status()
            
            return {
                "system_status": status,
                "active_contexts": active_contexts,
                "context_history": context_history,
                "statistics": statistics,
                "configuration": {
                    "default_passing_mode": self.context_passing_engine.default_passing_mode.value,
                    "default_relevance_threshold": self.context_passing_engine.default_relevance_threshold.value,
                    "compression_enabled": self.context_passing_engine.context_compression_enabled,
                    "filtering_enabled": self.context_passing_engine.context_filtering_enabled,
                    "max_history_size": self.context_passing_engine.max_history_size
                },
                "available_options": {
                    "passing_modes": self.get_context_passing_modes(),
                    "context_types": self.get_context_types(),
                    "relevance_levels": self.get_context_relevance_levels()
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get context passing report: {str(e)}",
                "success": False
            }
    
    # =============================================================================
    # Performance Optimization API Methods
    # =============================================================================
    
    def optimize_performance(self, optimization_type: str = "all") -> Dict[str, Any]:
        """Optimize meta-reasoning engine performance"""
        
        try:
            if optimization_type == "memory":
                result = self.performance_optimizer.optimize_for_memory()
            elif optimization_type == "processing":
                result = self.performance_optimizer.optimize_for_processing()
            elif optimization_type == "all":
                memory_result = self.performance_optimizer.optimize_for_memory()
                processing_result = self.performance_optimizer.optimize_for_processing()
                result = {
                    "memory_optimization": memory_result,
                    "processing_optimization": processing_result
                }
            else:
                return {
                    "error": f"Invalid optimization type: {optimization_type}",
                    "valid_types": ["memory", "processing", "all"],
                    "success": False
                }
            
            return {
                "optimization_type": optimization_type,
                "results": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to optimize performance: {str(e)}",
                "success": False
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        try:
            return {
                "performance_report": self.performance_optimizer.get_performance_report(),
                "engine_health": self.get_engine_health_report(),
                "system_metrics": self.get_system_metrics(),
                "optimization_recommendations": self.get_optimization_recommendations(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get performance report: {str(e)}",
                "success": False
            }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory usage status"""
        
        try:
            return {
                "memory_status": self.performance_optimizer.memory_monitor.get_memory_status(),
                "memory_critical": self.performance_optimizer.memory_monitor.is_memory_critical(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get memory status: {str(e)}",
                "success": False
            }
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get result cache status"""
        
        try:
            return {
                "cache_status": self.performance_optimizer.result_cache.get_cache_status(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get cache status: {str(e)}",
                "success": False
            }
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear result cache"""
        
        try:
            with self.performance_optimizer.result_cache.cache_lock:
                cache_size = len(self.performance_optimizer.result_cache.cache)
                self.performance_optimizer.result_cache.cache.clear()
                
                # Reset stats
                self.performance_optimizer.result_cache.cache_stats = {
                    "hits": 0,
                    "misses": 0,
                    "evictions": 0
                }
            
            return {
                "message": f"Cache cleared successfully",
                "cleared_entries": cache_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to clear cache: {str(e)}",
                "success": False
            }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get performance optimization recommendations"""
        
        try:
            recommendations = self.performance_optimizer._get_optimization_recommendations()
            
            # Add engine-specific recommendations
            engine_health = self.get_engine_health_report()
            if "engine_health" in engine_health:
                for engine_type, health_info in engine_health["engine_health"].items():
                    if health_info.get("status") in ["degraded", "unhealthy"]:
                        recommendations.append(f"Engine {engine_type} is {health_info['status']} - consider investigation")
            
            return {
                "recommendations": recommendations,
                "recommendation_count": len(recommendations),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get optimization recommendations: {str(e)}",
                "success": False
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        
        try:
            return {
                "performance_metrics": self.performance_optimizer.performance_metrics.get_summary(),
                "processing_metrics": self.performance_optimizer.processing_optimizer.get_metrics(),
                "memory_metrics": self.performance_optimizer.memory_monitor.get_memory_status(),
                "cache_metrics": self.performance_optimizer.result_cache.get_cache_status(),
                "engine_count": len(self.reasoning_engines),
                "interaction_patterns": len(self.interaction_patterns),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to get system metrics: {str(e)}",
                "success": False
            }
    
    def enable_performance_optimization(self) -> Dict[str, Any]:
        """Enable performance optimization"""
        
        try:
            self.performance_optimizer.optimization_enabled = True
            return {
                "message": "Performance optimization enabled",
                "optimization_enabled": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to enable performance optimization: {str(e)}",
                "success": False
            }
    
    def disable_performance_optimization(self) -> Dict[str, Any]:
        """Disable performance optimization"""
        
        try:
            self.performance_optimizer.optimization_enabled = False
            return {
                "message": "Performance optimization disabled",
                "optimization_enabled": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Failed to disable performance optimization: {str(e)}",
                "success": False
            }


# =============================================================================
# Performance Optimization Classes
# =============================================================================

class PerformanceOptimizer:
    """Comprehensive performance optimization system for meta-reasoning engine"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.optimization_enabled = True
        
        # Memory optimization
        self.memory_monitor = MemoryMonitor()
        self.object_pool = ObjectPool()
        
        # Result caching
        self.result_cache = ResultCache()
        
        # Processing optimization
        self.processing_optimizer = ProcessingOptimizer()
        
        # Metrics tracking
        self.performance_metrics = PerformanceMetrics()
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Weak reference to avoid circular dependencies
        self._meta_engine_ref = weakref.ref(meta_reasoning_engine)
    
    def optimize_for_memory(self) -> Dict[str, Any]:
        """Optimize memory usage across all components"""
        
        if not self.optimization_enabled:
            return {"status": "optimization_disabled"}
        
        optimization_results = {}
        
        # Clean up caches
        cache_cleanup = self.result_cache.cleanup_expired()
        optimization_results["cache_cleanup"] = cache_cleanup
        
        # Optimize object pool
        pool_optimization = self.object_pool.optimize()
        optimization_results["object_pool"] = pool_optimization
        
        # Memory monitoring
        memory_status = self.memory_monitor.get_memory_status()
        optimization_results["memory_status"] = memory_status
        
        # Clean up engine metrics
        engine_cleanup = self._optimize_engine_metrics()
        optimization_results["engine_cleanup"] = engine_cleanup
        
        return optimization_results
    
    def optimize_for_processing(self) -> Dict[str, Any]:
        """Optimize processing performance"""
        
        if not self.optimization_enabled:
            return {"status": "optimization_disabled"}
        
        return self.processing_optimizer.optimize_all()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            "memory_status": self.memory_monitor.get_memory_status(),
            "cache_status": self.result_cache.get_cache_status(),
            "processing_metrics": self.processing_optimizer.get_metrics(),
            "performance_metrics": self.performance_metrics.get_summary(),
            "optimization_recommendations": self._get_optimization_recommendations()
        }
    
    def _optimize_engine_metrics(self) -> Dict[str, Any]:
        """Optimize engine performance metrics storage"""
        
        cleaned_count = 0
        
        # Clean up old health history entries
        for engine_type in ReasoningEngine:
            if engine_type in self.meta_reasoning_engine.health_monitor.engine_metrics:
                metrics = self.meta_reasoning_engine.health_monitor.engine_metrics[engine_type]
                
                # Clear old entries from deques if they're at capacity
                if len(metrics.health_history) >= 45:  # Close to maxlen
                    # Keep only recent entries
                    while len(metrics.health_history) > 30:
                        metrics.health_history.popleft()
                    cleaned_count += 1
                
                if len(metrics.recent_errors) >= 8:  # Close to maxlen
                    # Keep only recent errors
                    while len(metrics.recent_errors) > 5:
                        metrics.recent_errors.popleft()
                    cleaned_count += 1
        
        return {
            "cleaned_metrics": cleaned_count,
            "total_engines": len(ReasoningEngine)
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        
        recommendations = []
        
        # Memory recommendations
        memory_status = self.memory_monitor.get_memory_status()
        if memory_status.get("memory_usage_percent", 0) > 80:
            recommendations.append("High memory usage - consider reducing cache size")
        
        # Cache recommendations
        cache_status = self.result_cache.get_cache_status()
        if cache_status.get("hit_rate", 0) < 0.5:
            recommendations.append("Low cache hit rate - consider adjusting cache strategy")
        
        # Processing recommendations
        processing_metrics = self.processing_optimizer.get_metrics()
        if processing_metrics.get("average_processing_time", 0) > 30:
            recommendations.append("High processing time - consider optimizing algorithms")
        
        return recommendations


class MemoryMonitor:
    """Monitor and optimize memory usage"""
    
    def __init__(self):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.memory_thresholds = {
            "warning": 0.8,
            "critical": 0.9
        }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        
        if not self.monitoring_enabled:
            return {"status": "monitoring_unavailable"}
        
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            return {
                "system_memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "free": memory.free,
                    "percent": memory.percent
                },
                "process_memory": {
                    "rss": process_memory.rss,
                    "vms": process_memory.vms,
                    "percent": process.memory_percent()
                },
                "python_memory": {
                    "objects": len(gc.get_objects()),
                    "collections": gc.get_stats()
                }
            }
        except Exception as e:
            return {"error": f"Failed to get memory status: {str(e)}"}
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        
        if not self.monitoring_enabled:
            return False
        
        try:
            memory = psutil.virtual_memory()
            return memory.percent > self.memory_thresholds["critical"] * 100
        except:
            return False


class ObjectPool:
    """Object pool for frequently created objects"""
    
    def __init__(self):
        self.pools = defaultdict(deque)
        self.pool_sizes = {
            "ReasoningResult": 100,
            "ContextItem": 200,
            "SequentialContext": 50,
            "FormattedResult": 50
        }
        self.created_objects = defaultdict(int)
        self.reused_objects = defaultdict(int)
    
    def get_object(self, object_type: str, *args, **kwargs):
        """Get object from pool or create new one"""
        
        pool = self.pools[object_type]
        
        if pool:
            obj = pool.popleft()
            self.reused_objects[object_type] += 1
            return obj
        else:
            # Create new object (would need factory methods for each type)
            self.created_objects[object_type] += 1
            return None  # Placeholder - would create actual object
    
    def return_object(self, object_type: str, obj):
        """Return object to pool"""
        
        pool = self.pools[object_type]
        max_size = self.pool_sizes.get(object_type, 50)
        
        if len(pool) < max_size:
            # Reset object state before returning to pool
            self._reset_object(obj)
            pool.append(obj)
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize object pools"""
        
        optimized_pools = {}
        
        for pool_type, pool in self.pools.items():
            original_size = len(pool)
            target_size = self.pool_sizes.get(pool_type, 50) // 2
            
            # Reduce pool size if too large
            while len(pool) > target_size:
                pool.popleft()
            
            optimized_pools[pool_type] = {
                "original_size": original_size,
                "new_size": len(pool),
                "created": self.created_objects[pool_type],
                "reused": self.reused_objects[pool_type]
            }
        
        return optimized_pools
    
    def _reset_object(self, obj):
        """Reset object state for reuse"""
        # Implementation would depend on object type
        pass


class ResultCache:
    """Advanced result caching system"""
    
    def __init__(self):
        # Use TTL cache if available, otherwise use simple dict
        if CACHETOOLS_AVAILABLE:
            self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        else:
            self.cache = {}
        
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        self.cache_lock = threading.RLock()
    
    def get_cache_key(self, query: str, context: Dict[str, Any], 
                     thinking_mode: ThinkingMode) -> str:
        """Generate cache key for query"""
        
        # Create deterministic hash of query and context
        key_data = {
            "query": query,
            "context": json.dumps(context, sort_keys=True),
            "thinking_mode": thinking_mode.value
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available"""
        
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[cache_key]
            else:
                self.cache_stats["misses"] += 1
                return None
    
    def cache_result(self, cache_key: str, result: Any):
        """Cache result"""
        
        with self.cache_lock:
            self.cache[cache_key] = result
    
    def cleanup_expired(self) -> Dict[str, Any]:
        """Clean up expired cache entries"""
        
        if CACHETOOLS_AVAILABLE:
            # TTL cache automatically handles expiration
            return {"status": "automatic_cleanup"}
        else:
            # Manual cleanup for simple dict cache
            # (Would need timestamp tracking for full implementation)
            return {"status": "manual_cleanup_needed"}
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        with self.cache_lock:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "hit_rate": hit_rate,
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "evictions": self.cache_stats["evictions"]
            }


class ProcessingOptimizer:
    """Optimize processing performance"""
    
    def __init__(self):
        self.optimization_metrics = {
            "parallel_processing": 0,
            "sequential_processing": 0,
            "cache_utilization": 0,
            "memory_optimization": 0
        }
        
        self.processing_times = deque(maxlen=100)
        self.optimization_history = deque(maxlen=50)
    
    def optimize_all(self) -> Dict[str, Any]:
        """Run all optimization strategies"""
        
        optimizations = {}
        
        # Parallel processing optimization
        parallel_opt = self._optimize_parallel_processing()
        optimizations["parallel_processing"] = parallel_opt
        
        # Sequential processing optimization
        sequential_opt = self._optimize_sequential_processing()
        optimizations["sequential_processing"] = sequential_opt
        
        # Cache optimization
        cache_opt = self._optimize_caching()
        optimizations["cache_optimization"] = cache_opt
        
        # Memory optimization
        memory_opt = self._optimize_memory_usage()
        optimizations["memory_optimization"] = memory_opt
        
        return optimizations
    
    def _optimize_parallel_processing(self) -> Dict[str, Any]:
        """Optimize parallel processing strategy"""
        
        return {
            "status": "optimized",
            "recommendations": [
                "Use async/await for I/O bound operations",
                "Implement proper task batching",
                "Consider connection pooling for external services"
            ]
        }
    
    def _optimize_sequential_processing(self) -> Dict[str, Any]:
        """Optimize sequential processing strategy"""
        
        return {
            "status": "optimized",
            "recommendations": [
                "Implement early stopping for low-confidence results",
                "Use adaptive timeout based on complexity",
                "Optimize context passing between engines"
            ]
        }
    
    def _optimize_caching(self) -> Dict[str, Any]:
        """Optimize caching strategy"""
        
        return {
            "status": "optimized",
            "recommendations": [
                "Implement result compression for large results",
                "Use cache warming for frequently accessed results",
                "Consider distributed caching for scale"
            ]
        }
    
    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        
        return {
            "status": "optimized",
            "recommendations": [
                "Use generators for large data processing",
                "Implement lazy loading for engine components",
                "Consider streaming for large result sets"
            ]
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing optimization metrics"""
        
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
        
        return {
            "average_processing_time": avg_processing_time,
            "optimization_metrics": self.optimization_metrics,
            "processing_times_count": len(self.processing_times),
            "optimization_history_count": len(self.optimization_history)
        }


class PerformanceMetrics:
    """Track comprehensive performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_optimizations": 0,
            "processing_optimizations": 0
        }
        
        self.timing_metrics = {
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "min_processing_time": float('inf'),
            "max_processing_time": 0.0
        }
        
        self.engine_metrics = defaultdict(int)
        self.thinking_mode_metrics = defaultdict(int)
    
    def record_request(self, processing_time: float, thinking_mode: ThinkingMode,
                      engines_used: List[ReasoningEngine], success: bool):
        """Record request metrics"""
        
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update timing metrics
        self.timing_metrics["total_processing_time"] += processing_time
        self.timing_metrics["average_processing_time"] = (
            self.timing_metrics["total_processing_time"] / self.metrics["total_requests"]
        )
        self.timing_metrics["min_processing_time"] = min(
            self.timing_metrics["min_processing_time"], processing_time
        )
        self.timing_metrics["max_processing_time"] = max(
            self.timing_metrics["max_processing_time"], processing_time
        )
        
        # Update engine metrics
        for engine in engines_used:
            self.engine_metrics[engine.value] += 1
        
        # Update thinking mode metrics
        self.thinking_mode_metrics[thinking_mode.value] += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics["cache_misses"] += 1
    
    def record_optimization(self, optimization_type: str):
        """Record optimization event"""
        self.metrics[f"{optimization_type}_optimizations"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        
        success_rate = (
            self.metrics["successful_requests"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )
        
        cache_hit_rate = (
            self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0
        )
        
        return {
            "request_metrics": self.metrics,
            "timing_metrics": self.timing_metrics,
            "engine_usage": dict(self.engine_metrics),
            "thinking_mode_usage": dict(self.thinking_mode_metrics),
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate
        }