"""
Performance Profiler
====================

Function and memory profiling decorators and utilities for performance analysis.
"""

import cProfile
import functools
import io  
import logging
import pstats
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import os

from .monitor import get_performance_monitor, MetricType

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

@dataclass
class ProfileResult:
    """Result of function profiling"""
    function_name: str
    execution_time_ms: float
    call_count: int
    memory_peak_mb: float
    memory_current_mb: float
    cpu_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    call_stack: Optional[List[str]] = None
    memory_trace: Optional[Dict[str, Any]] = None

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    current_mb: float
    peak_mb: float
    blocks_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)

class PerformanceProfiler:
    """Performance profiler with memory and CPU tracking"""
    
    def __init__(self):
        self._profiles: Dict[str, List[ProfileResult]] = {}
        self._memory_snapshots: List[MemorySnapshot] = []
        self._lock = threading.RLock()
        self._memory_tracking_enabled = False
        
    def enable_memory_tracking(self) -> None:
        """Enable memory tracking"""
        if not self._memory_tracking_enabled:
            tracemalloc.start()
            self._memory_tracking_enabled = True
            logger.info("Memory tracking enabled")
    
    def disable_memory_tracking(self) -> None:
        """Disable memory tracking"""
        if self._memory_tracking_enabled:
            tracemalloc.stop()
            self._memory_tracking_enabled = False
            logger.info("Memory tracking disabled")
    
    def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        if not self._memory_tracking_enabled:
            self.enable_memory_tracking()
        
        try:
            # Get current memory usage
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / (1024 * 1024)
            peak_mb = peak / (1024 * 1024)
            
            # Get top memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            top_allocations = []
            for index, stat in enumerate(top_stats[:10]):
                top_allocations.append({
                    'rank': index + 1,
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count,
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown'
                })
            
            memory_snapshot = MemorySnapshot(
                current_mb=current_mb,
                peak_mb=peak_mb,
                blocks_count=len(top_stats),
                top_allocations=top_allocations
            )
            
            with self._lock:
                self._memory_snapshots.append(memory_snapshot)
                # Keep only last 100 snapshots
                if len(self._memory_snapshots) > 100:
                    self._memory_snapshots = self._memory_snapshots[-100:]
            
            return memory_snapshot
            
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            return MemorySnapshot(current_mb=0, peak_mb=0, blocks_count=0)
    
    def profile_function(
        self, 
        func: F,
        track_memory: bool = True,
        track_calls: bool = False
    ) -> ProfileResult:
        """Profile a single function call"""
        function_name = f"{func.__module__}.{func.__name__}"
        
        # Memory tracking setup
        memory_start = None
        if track_memory and self._memory_tracking_enabled:
            memory_start = tracemalloc.get_traced_memory()
        
        # CPU profiling setup
        profiler = None
        if track_calls:
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Execute function
        start_time = time.perf_counter()
        try:
            result = func()
            success = True
        except Exception as e:
            logger.error(f"Function {function_name} failed during profiling: {e}")
            success = False
            result = None
        finally:
            end_time = time.perf_counter()
            
            # Stop CPU profiling
            if profiler:
                profiler.disable()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Memory analysis
        memory_peak_mb = 0
        memory_current_mb = 0
        memory_trace = None
        
        if track_memory and self._memory_tracking_enabled:
            try:
                current, peak = tracemalloc.get_traced_memory()
                if memory_start:
                    memory_current_mb = (current - memory_start[0]) / (1024 * 1024)
                    memory_peak_mb = (peak - memory_start[1]) / (1024 * 1024)
                else:
                    memory_current_mb = current / (1024 * 1024)
                    memory_peak_mb = peak / (1024 * 1024)
                
                # Get memory allocation details
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                memory_trace = {
                    'total_allocations': len(top_stats),
                    'top_allocations': [
                        {
                            'size_mb': stat.size / (1024 * 1024),
                            'count': stat.count,
                            'file': stat.traceback.format()[0] if stat.traceback else 'unknown'
                        }
                        for stat in top_stats[:5]
                    ]
                }
            except Exception as e:
                logger.warning(f"Memory analysis failed: {e}")
        
        # CPU analysis
        cpu_time_ms = execution_time_ms  # Default fallback
        call_stack = None
        
        if profiler and track_calls:
            try:
                # Analyze CPU profiling results
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats(10)  # Top 10 functions
                
                # Extract call stack information
                call_stack = s.getvalue().split('\n')[:15]  # First 15 lines
                
                # Calculate CPU time
                total_calls = ps.total_calls
                cpu_time_ms = execution_time_ms  # For now, use wall time
                
            except Exception as e:
                logger.warning(f"CPU profiling analysis failed: {e}")
        
        # Create profile result
        profile_result = ProfileResult(
            function_name=function_name,
            execution_time_ms=execution_time_ms,
            call_count=1,
            memory_peak_mb=memory_peak_mb,
            memory_current_mb=memory_current_mb,
            cpu_time_ms=cpu_time_ms,
            call_stack=call_stack,
            memory_trace=memory_trace
        )
        
        # Store result
        with self._lock:
            if function_name not in self._profiles:
                self._profiles[function_name] = []
            self._profiles[function_name].append(profile_result)
            
            # Keep only last 50 profiles per function
            if len(self._profiles[function_name]) > 50:
                self._profiles[function_name] = self._profiles[function_name][-50:]
        
        # Record metrics
        monitor = get_performance_monitor()
        monitor.record_timer(function_name, execution_time_ms)
        if track_memory:
            monitor.record_metric(
                f"{function_name}_memory_peak",
                memory_peak_mb,
                MetricType.GAUGE
            )
        
        return profile_result
    
    def get_function_profiles(self, function_name: Optional[str] = None) -> Dict[str, List[ProfileResult]]:
        """Get profiling results for functions"""
        with self._lock:
            if function_name:
                return {function_name: self._profiles.get(function_name, [])}
            return dict(self._profiles)
    
    def get_memory_snapshots(self, limit: int = 50) -> List[MemorySnapshot]:
        """Get recent memory snapshots"""
        with self._lock:
            return self._memory_snapshots[-limit:]
    
    def analyze_performance_trends(self, function_name: str) -> Dict[str, Any]:
        """Analyze performance trends for a function"""
        with self._lock:
            profiles = self._profiles.get(function_name, [])
        
        if not profiles:
            return {"error": f"No profiles found for {function_name}"}
        
        # Calculate statistics
        execution_times = [p.execution_time_ms for p in profiles]
        memory_peaks = [p.memory_peak_mb for p in profiles]
        
        recent_profiles = profiles[-10:]  # Last 10 calls
        recent_times = [p.execution_time_ms for p in recent_profiles]
        
        analysis = {
            "function_name": function_name,
            "total_calls": len(profiles),
            "execution_time": {
                "avg_ms": sum(execution_times) / len(execution_times),
                "min_ms": min(execution_times),
                "max_ms": max(execution_times),
                "recent_avg_ms": sum(recent_times) / len(recent_times) if recent_times else 0
            },
            "memory_usage": {
                "avg_peak_mb": sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
                "max_peak_mb": max(memory_peaks) if memory_peaks else 0,
                "min_peak_mb": min(memory_peaks) if memory_peaks else 0
            },
            "trend_analysis": self._calculate_trends(execution_times, memory_peaks),
            "last_profiled": profiles[-1].timestamp.isoformat() if profiles else None
        }
        
        return analysis
    
    def _calculate_trends(self, execution_times: List[float], memory_peaks: List[float]) -> Dict[str, str]:
        """Calculate performance trends"""
        trends = {}
        
        if len(execution_times) >= 5:
            # Simple trend analysis - compare first half vs second half
            mid_point = len(execution_times) // 2
            first_half_avg = sum(execution_times[:mid_point]) / mid_point
            second_half_avg = sum(execution_times[mid_point:]) / (len(execution_times) - mid_point)
            
            if second_half_avg > first_half_avg * 1.1:
                trends["execution_time"] = "degrading"
            elif second_half_avg < first_half_avg * 0.9:
                trends["execution_time"] = "improving"
            else:
                trends["execution_time"] = "stable"
        else:
            trends["execution_time"] = "insufficient_data"
        
        if memory_peaks and len(memory_peaks) >= 5:
            mid_point = len(memory_peaks) // 2
            first_half_avg = sum(memory_peaks[:mid_point]) / mid_point
            second_half_avg = sum(memory_peaks[mid_point:]) / (len(memory_peaks) - mid_point)
            
            if second_half_avg > first_half_avg * 1.1:
                trends["memory_usage"] = "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                trends["memory_usage"] = "decreasing"
            else:
                trends["memory_usage"] = "stable"
        else:
            trends["memory_usage"] = "insufficient_data"
        
        return trends
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        report_lines = ["PRSM Performance Report", "=" * 50, ""]
        
        # Memory overview
        if self._memory_snapshots:
            latest_snapshot = self._memory_snapshots[-1]
            report_lines.extend([
                f"Memory Usage (Current): {latest_snapshot.current_mb:.2f} MB",
                f"Memory Usage (Peak): {latest_snapshot.peak_mb:.2f} MB",
                f"Active Memory Blocks: {latest_snapshot.blocks_count}",
                ""
            ])
        
        # Top functions by execution time
        all_profiles = []
        for func_profiles in self._profiles.values():
            all_profiles.extend(func_profiles)
        
        if all_profiles:
            # Sort by execution time
            sorted_profiles = sorted(all_profiles, key=lambda p: p.execution_time_ms, reverse=True)
            
            report_lines.extend([
                "Top Functions by Execution Time:",
                "-" * 40
            ])
            
            for i, profile in enumerate(sorted_profiles[:10]):
                report_lines.append(
                    f"{i+1:2d}. {profile.function_name:<50} {profile.execution_time_ms:8.2f}ms"
                )
            
            report_lines.append("")
            
            # Top functions by memory usage
            sorted_by_memory = sorted(all_profiles, key=lambda p: p.memory_peak_mb, reverse=True)
            
            report_lines.extend([
                "Top Functions by Memory Usage:",
                "-" * 40
            ])
            
            for i, profile in enumerate(sorted_by_memory[:10]):
                if profile.memory_peak_mb > 0:
                    report_lines.append(
                        f"{i+1:2d}. {profile.function_name:<50} {profile.memory_peak_mb:8.2f}MB"
                    )
            
            report_lines.append("")
        
        # Performance trends
        report_lines.extend([
            "Performance Trends:",
            "-" * 20
        ])
        
        for func_name in list(self._profiles.keys())[:5]:  # Top 5 functions
            trends = self.analyze_performance_trends(func_name)
            if "trend_analysis" in trends:
                exec_trend = trends["trend_analysis"].get("execution_time", "unknown")
                mem_trend = trends["trend_analysis"].get("memory_usage", "unknown")
                report_lines.append(f"{func_name}: Time={exec_trend}, Memory={mem_trend}")
        
        report_lines.extend(["", f"Report generated: {datetime.now(timezone.utc).isoformat()}"])
        
        return "\n".join(report_lines)

# Global profiler instance
_profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    return _profiler

# Decorators for easy profiling

def profile_function(
    track_memory: bool = True,
    track_calls: bool = False,
    enable_monitoring: bool = True
) -> Callable[[F], F]:
    """
    Decorator to profile function performance.
    
    Args:
        track_memory: Whether to track memory usage
        track_calls: Whether to track function calls (CPU profiling)
        enable_monitoring: Whether to send metrics to performance monitor
        
    Returns:
        Decorated function with profiling
        
    Example:
        @profile_function(track_memory=True, track_calls=True)
        def expensive_function():
            # Expensive computation
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            
            # Create a closure for profiling
            def profiled_function():
                return func(*args, **kwargs)
            
            # Profile the function
            profile_result = profiler.profile_function(
                profiled_function,
                track_memory=track_memory,
                track_calls=track_calls
            )
            
            # Log significant performance issues
            if profile_result.execution_time_ms > 1000:  # > 1 second
                logger.warning(
                    f"Slow function detected: {func.__name__} took "
                    f"{profile_result.execution_time_ms:.2f}ms"
                )
            
            if profile_result.memory_peak_mb > 100:  # > 100MB
                logger.warning(
                    f"High memory usage: {func.__name__} used "
                    f"{profile_result.memory_peak_mb:.2f}MB"
                )
            
            return profile_result.function_name  # Return the actual function result
        
        return wrapper
    return decorator

def async_profile_function(
    track_memory: bool = True,
    track_calls: bool = False
) -> Callable[[F], F]:
    """
    Decorator to profile async function performance.
    
    Args:
        track_memory: Whether to track memory usage
        track_calls: Whether to track function calls
        
    Returns:
        Decorated async function with profiling
        
    Example:
        @async_profile_function(track_memory=True)
        async def expensive_async_function():
            # Expensive async computation
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = get_profiler()
            
            # For async functions, we need to handle differently
            async def profiled_function():
                return await func(*args, **kwargs)
            
            # Profile execution
            start_time = time.perf_counter()
            
            if track_memory:
                profiler.enable_memory_tracking()
                memory_start = tracemalloc.get_traced_memory() if profiler._memory_tracking_enabled else (0, 0)
            
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Async function {func.__name__} failed during profiling: {e}")
                raise
            finally:
                end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            # Record metrics
            monitor = get_performance_monitor()
            monitor.record_timer(f"{func.__module__}.{func.__name__}", execution_time_ms)
            
            if track_memory and profiler._memory_tracking_enabled:
                current, peak = tracemalloc.get_traced_memory()
                memory_used_mb = (current - memory_start[0]) / (1024 * 1024)
                monitor.record_metric(
                    f"{func.__name__}_memory_usage",
                    memory_used_mb,
                    MetricType.GAUGE
                )
            
            return result
        
        return wrapper
    return decorator

def memory_profile(func: F) -> F:
    """
    Simple decorator for memory profiling.
    
    Example:
        @memory_profile
        def memory_intensive_function():
            # Memory intensive operation
            return result
    """
    return profile_function(track_memory=True, track_calls=False)(func)

@contextmanager
def profile_context(name: str):
    """
    Context manager for profiling code blocks.
    
    Example:
        with profile_context("expensive_operation"):
            # Expensive code here
            result = complex_calculation()
    """
    profiler = get_profiler()
    monitor = get_performance_monitor()
    
    # Start timing and memory tracking
    start_time = time.perf_counter()
    profiler.enable_memory_tracking()
    memory_start = tracemalloc.get_traced_memory() if profiler._memory_tracking_enabled else (0, 0)
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Calculate memory usage
        memory_used_mb = 0
        if profiler._memory_tracking_enabled:
            current, peak = tracemalloc.get_traced_memory()
            memory_used_mb = (current - memory_start[0]) / (1024 * 1024)
        
        # Record metrics
        monitor.record_timer(name, execution_time_ms)
        if memory_used_mb > 0:
            monitor.record_metric(f"{name}_memory_usage", memory_used_mb, MetricType.GAUGE)
        
        # Log if significant
        if execution_time_ms > 1000:
            logger.info(f"Profile context '{name}' took {execution_time_ms:.2f}ms")
        if memory_used_mb > 10:
            logger.info(f"Profile context '{name}' used {memory_used_mb:.2f}MB")