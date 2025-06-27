"""
PRSM Performance Profiler
========================

Advanced performance profiling system for PRSM applications.
Provides detailed insights into system performance, resource usage,
and bottleneck identification.
"""

import asyncio
import time
import psutil
import functools
import tracemalloc
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import threading
import json
from collections import defaultdict


@dataclass
class ProfileResult:
    """Results from performance profiling"""
    function_name: str
    execution_time: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    call_count: int
    timestamp: float
    stack_trace: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for PRSM systems
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.profiles: Dict[str, List[ProfileResult]] = defaultdict(list)
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        if self.enable_memory_tracking:
            tracemalloc.start()
    
    def profile(self, func_name: Optional[str] = None, include_stack: bool = False):
        """
        Decorator for profiling function performance
        """
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._profile_async_execution(name, func, include_stack, *args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._profile_sync_execution(name, func, include_stack, *args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    async def _profile_async_execution(self, name: str, func: Callable, include_stack: bool, *args, **kwargs):
        """Profile async function execution"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            profile_result = ProfileResult(
                function_name=name,
                execution_time=end_time - start_time,
                memory_usage={
                    'start': start_memory,
                    'end': end_memory,
                    'delta': end_memory - start_memory
                },
                cpu_usage=end_cpu - start_cpu,
                call_count=1,
                timestamp=start_time,
                stack_trace=traceback.format_stack() if include_stack else None
            )
            
            with self.lock:
                self.profiles[name].append(profile_result)
    
    def _profile_sync_execution(self, name: str, func: Callable, include_stack: bool, *args, **kwargs):
        """Profile sync function execution"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            profile_result = ProfileResult(
                function_name=name,
                execution_time=end_time - start_time,
                memory_usage={
                    'start': start_memory,
                    'end': end_memory,
                    'delta': end_memory - start_memory
                },
                cpu_usage=end_cpu - start_cpu,
                call_count=1,
                timestamp=start_time,
                stack_trace=traceback.format_stack() if include_stack else None
            )
            
            with self.lock:
                self.profiles[name].append(profile_result)
    
    @asynccontextmanager
    async def profile_context(self, name: str, custom_metrics: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling code blocks
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            profile_result = ProfileResult(
                function_name=name,
                execution_time=end_time - start_time,
                memory_usage={
                    'start': start_memory,
                    'end': end_memory,
                    'delta': end_memory - start_memory
                },
                cpu_usage=end_cpu - start_cpu,
                call_count=1,
                timestamp=start_time,
                custom_metrics=custom_metrics or {}
            )
            
            with self.lock:
                self.profiles[name].append(profile_result)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.enable_memory_tracking:
            try:
                current, peak = tracemalloc.get_traced_memory()
                return current / 1024 / 1024  # Convert to MB
            except:
                pass
        
        # Fallback to process memory
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_profile_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for profiled functions
        """
        with self.lock:
            if function_name:
                profiles = self.profiles.get(function_name, [])
                if not profiles:
                    return {}
                
                return self._calculate_summary(function_name, profiles)
            else:
                summary = {}
                for name, profiles in self.profiles.items():
                    summary[name] = self._calculate_summary(name, profiles)
                return summary
    
    def _calculate_summary(self, name: str, profiles: List[ProfileResult]) -> Dict[str, Any]:
        """Calculate summary statistics for a function"""
        if not profiles:
            return {}
        
        execution_times = [p.execution_time for p in profiles]
        memory_deltas = [p.memory_usage['delta'] for p in profiles]
        cpu_usage = [p.cpu_usage for p in profiles]
        
        return {
            'function_name': name,
            'call_count': len(profiles),
            'execution_time': {
                'total': sum(execution_times),
                'average': sum(execution_times) / len(execution_times),
                'min': min(execution_times),
                'max': max(execution_times)
            },
            'memory_usage': {
                'average_delta': sum(memory_deltas) / len(memory_deltas),
                'max_delta': max(memory_deltas),
                'min_delta': min(memory_deltas)
            },
            'cpu_usage': {
                'average': sum(cpu_usage) / len(cpu_usage),
                'max': max(cpu_usage),
                'min': min(cpu_usage)
            },
            'last_execution': profiles[-1].timestamp
        }
    
    def get_top_functions(self, metric: str = 'execution_time', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top functions by specified metric
        """
        summary = self.get_profile_summary()
        
        if metric == 'execution_time':
            sorted_functions = sorted(
                summary.items(),
                key=lambda x: x[1]['execution_time']['total'],
                reverse=True
            )
        elif metric == 'memory_usage':
            sorted_functions = sorted(
                summary.items(),
                key=lambda x: x[1]['memory_usage']['average_delta'],
                reverse=True
            )
        elif metric == 'cpu_usage':
            sorted_functions = sorted(
                summary.items(),
                key=lambda x: x[1]['cpu_usage']['average'],
                reverse=True
            )
        elif metric == 'call_count':
            sorted_functions = sorted(
                summary.items(),
                key=lambda x: x[1]['call_count'],
                reverse=True
            )
        else:
            return []
        
        return [func[1] for func in sorted_functions[:limit]]
    
    def clear_profiles(self, function_name: Optional[str] = None):
        """Clear profiling data"""
        with self.lock:
            if function_name:
                self.profiles.pop(function_name, None)
            else:
                self.profiles.clear()
    
    def export_profiles(self, filename: str):
        """Export profiling data to JSON file"""
        with self.lock:
            data = {}
            for name, profiles in self.profiles.items():
                data[name] = [
                    {
                        'function_name': p.function_name,
                        'execution_time': p.execution_time,
                        'memory_usage': p.memory_usage,
                        'cpu_usage': p.cpu_usage,
                        'call_count': p.call_count,
                        'timestamp': p.timestamp,
                        'custom_metrics': p.custom_metrics
                    }
                    for p in profiles
                ]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'process_count': len(psutil.pids()),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }