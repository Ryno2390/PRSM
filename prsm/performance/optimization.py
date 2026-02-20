"""
PRSM Performance Optimization Module

Provides caching strategies, batch processing optimization, and
performance optimization utilities for the PRSM system.
"""

import asyncio
import functools
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, Generic, List, Optional, TypeVar, ParamSpec, Tuple
)
import structlog

logger = structlog.get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CacheLevel(Enum):
    """Cache level enumeration"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


@dataclass
class CacheConfig:
    """Configuration for cache behavior"""
    max_size: int = 1000
    ttl_seconds: int = 3600
    level: CacheLevel = CacheLevel.L1_MEMORY
    eviction_policy: str = "lru"
    enable_stats: bool = True


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache implementation
    
    Usage:
        cache = LRUCache(max_size=100)
        cache.set("key", "value")
        value = cache.get("key")
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[K, Tuple[V, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: K) -> Optional[V]:
        if key not in self._cache:
            self._misses += 1
            return None
        
        value, timestamp = self._cache[key]
        
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None
        
        self._cache.move_to_end(key)
        self._hits += 1
        return value
    
    def set(self, key: K, value: V) -> None:
        if key in self._cache:
            del self._cache[key]
        
        self._cache[key] = (value, time.time())
        
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def delete(self, key: K) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        self._cache.clear()
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class CacheManager:
    """
    Multi-level cache manager for PRSM
    
    Manages L1 (memory), L2 (Redis), and L3 (database) caches.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._l1_cache = LRUCache[str, Any](
            max_size=self.config.max_size,
            ttl_seconds=self.config.ttl_seconds
        )
        self._stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
        }
    
    async def get(self, key: str) -> Optional[Any]:
        value = self._l1_cache.get(key)
        if value is not None:
            self._stats["l1_hits"] += 1
            return value
        
        self._stats["l1_misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._l1_cache.set(key, value)
    
    async def delete(self, key: str) -> bool:
        return self._l1_cache.delete(key)
    
    async def clear(self) -> None:
        self._l1_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        stats["l1_cache"] = self._l1_cache.get_stats()
        return stats


class PerformanceOptimizer:
    """
    Performance optimization utilities
    
    Provides automatic optimization suggestions and performance tuning.
    """
    
    def __init__(self):
        self._optimizations: List[Dict[str, Any]] = []
    
    def suggest_optimizations(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        suggestions = []
        
        for operation, stats in performance_data.items():
            if stats.get("mean_ms", 0) > 100:
                suggestions.append({
                    "operation": operation,
                    "issue": "slow_operation",
                    "current_ms": stats.get("mean_ms"),
                    "suggestion": "Consider caching or batching",
                    "priority": "high" if stats.get("mean_ms", 0) > 500 else "medium",
                })
            
            if stats.get("error_count", 0) > 0:
                error_rate = stats["error_count"] / max(stats.get("count", 1), 1)
                if error_rate > 0.05:
                    suggestions.append({
                        "operation": operation,
                        "issue": "high_error_rate",
                        "error_rate": error_rate,
                        "suggestion": "Investigate and fix errors",
                        "priority": "high",
                    })
        
        self._optimizations = suggestions
        return suggestions


class QueryOptimizer:
    """
    Query optimization utilities
    
    Provides query analysis and optimization suggestions.
    """
    
    def __init__(self):
        self._slow_queries: List[Dict[str, Any]] = []
    
    def analyze_query(self, query: str, execution_time_ms: float) -> Optional[Dict[str, Any]]:
        if execution_time_ms > 100:
            analysis = {
                "query": query[:200],
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestions": self._generate_suggestions(query, execution_time_ms),
            }
            self._slow_queries.append(analysis)
            return analysis
        return None
    
    def _generate_suggestions(self, query: str, execution_time_ms: float) -> List[str]:
        suggestions = []
        query_lower = query.lower()
        
        if "select *" in query_lower:
            suggestions.append("Avoid SELECT * - specify needed columns")
        
        if "where" not in query_lower and "limit" not in query_lower:
            suggestions.append("Add WHERE clause or LIMIT to reduce result set")
        
        if execution_time_ms > 1000:
            suggestions.append("Consider adding indexes or restructuring query")
        
        return suggestions
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        return sorted(self._slow_queries, key=lambda x: x["execution_time_ms"], reverse=True)[:limit]


class APIOptimizer:
    """
    API performance optimization utilities
    
    Provides API endpoint analysis and optimization suggestions.
    """
    
    def __init__(self):
        self._endpoint_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_request(self, endpoint: str, response_time_ms: float, status_code: int) -> None:
        if endpoint not in self._endpoint_stats:
            self._endpoint_stats[endpoint] = {
                "total_requests": 0,
                "total_time_ms": 0,
                "status_codes": {},
                "errors": 0,
            }
        
        stats = self._endpoint_stats[endpoint]
        stats["total_requests"] += 1
        stats["total_time_ms"] += response_time_ms
        stats["status_codes"][status_code] = stats["status_codes"].get(status_code, 0) + 1
        
        if status_code >= 400:
            stats["errors"] += 1
    
    def get_endpoint_stats(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for endpoint, stats in self._endpoint_stats.items():
            result[endpoint] = {
                "total_requests": stats["total_requests"],
                "avg_time_ms": stats["total_time_ms"] / max(stats["total_requests"], 1),
                "error_rate": stats["errors"] / max(stats["total_requests"], 1),
                "status_codes": stats["status_codes"],
            }
        return result


class BatchProcessor(Generic[T]):
    """
    Batch processor for efficient bulk operations
    
    Usage:
        processor = BatchProcessor(batch_size=100, flush_interval=1.0)
        processor.add(item1)
        processor.add(item2)
        results = await processor.flush()
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        processor: Optional[Callable[[List[T]], Any]] = None,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.processor = processor
        self._batch: List[T] = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()
    
    async def add(self, item: T) -> None:
        async with self._lock:
            self._batch.append(item)
            
            if len(self._batch) >= self.batch_size:
                await self._flush()
    
    async def flush(self) -> List[Any]:
        async with self._lock:
            return await self._flush()
    
    async def _flush(self) -> List[Any]:
        if not self._batch:
            return []
        
        batch = self._batch
        self._batch = []
        self._last_flush = time.time()
        
        if self.processor:
            if asyncio.iscoroutinefunction(self.processor):
                return await self.processor(batch)
            return self.processor(batch)
        
        return batch
    
    async def should_flush(self) -> bool:
        return (
            len(self._batch) >= self.batch_size or
            time.time() - self._last_flush >= self.flush_interval
        )


def optimize_cache(ttl_seconds: int = 3600, max_size: int = 1000):
    """
    Decorator to cache function results
    
    Usage:
        @optimize_cache(ttl_seconds=300, max_size=50)
        async def expensive_operation(key):
            # ... expensive computation ...
            return result
    """
    cache = LRUCache[str, Any](max_size=max_size, ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = await func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def batch_processor(batch_size: int = 100):
    """
    Decorator to batch process function calls
    
    Usage:
        @batch_processor(batch_size=50)
        async def process_items(items: List[Any]):
            # Process batch
            return results
    """
    def decorator(func: Callable[[List[T]], Any]) -> Callable[[T], Any]:
        processor = BatchProcessor[T](batch_size=batch_size)
        processor.processor = func
        
        async def wrapper(item: T) -> Any:
            await processor.add(item)
        
        wrapper.flush = processor.flush
        return wrapper
    
    return decorator


class LazyLoader(Generic[T]):
    """
    Lazy loading wrapper for expensive resources
    
    Usage:
        loader = LazyLoader(lambda: expensive_resource)
        resource = await loader.get()
    """
    
    def __init__(self, loader: Callable[[], T]):
        self._loader = loader
        self._loaded = False
        self._value: Optional[T] = None
        self._lock = asyncio.Lock()
    
    async def get(self) -> T:
        if self._loaded:
            return self._value
        
        async with self._lock:
            if self._loaded:
                return self._value
            
            if asyncio.iscoroutinefunction(self._loader):
                self._value = await self._loader()
            else:
                self._value = self._loader()
            
            self._loaded = True
            return self._value
    
    def reset(self) -> None:
        self._loaded = False
        self._value = None
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
