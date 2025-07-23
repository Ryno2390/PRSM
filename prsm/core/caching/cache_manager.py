"""
Cache Manager
=============

Central cache management system with multi-tier support and performance monitoring.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Union, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import weakref

from ..config.manager import get_config
from ..errors.exceptions import ProcessingError, ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheLevel(Enum):
    """Cache levels in order of speed (fastest to slowest)"""
    MEMORY = "memory"
    REDIS = "redis" 
    FILE = "file"
    REMOTE = "remote"

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    key: str
    value: T
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: Optional[int] = None
    source_level: Optional[CacheLevel] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl_seconds is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry_time
    
    def update_access(self) -> None:
        """Update access metadata"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    cache_name: str
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    average_response_time_ms: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit(self, response_time_ms: float) -> None:
        """Update statistics for cache hit"""
        self.total_requests += 1
        self.cache_hits += 1
        self._update_response_time(response_time_ms)
        self._calculate_hit_rate()
    
    def update_miss(self, response_time_ms: float) -> None:
        """Update statistics for cache miss"""
        self.total_requests += 1
        self.cache_misses += 1
        self._update_response_time(response_time_ms)
        self._calculate_hit_rate()
    
    def update_error(self) -> None:
        """Update error count"""
        self.errors += 1
    
    def _update_response_time(self, response_time_ms: float) -> None:
        """Update rolling average response time"""
        if self.total_requests == 1:
            self.average_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.average_response_time_ms
            )
    
    def _calculate_hit_rate(self) -> None:
        """Calculate cache hit rate"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests

class CacheManager:
    """Multi-tier cache manager with performance monitoring"""
    
    _instance: Optional['CacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'CacheManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if hasattr(self, '_initialized'):
            return
        
        self._caches: Dict[str, Any] = {}
        self._cache_stats: Dict[str, CacheStats] = {}
        self._subscribers: weakref.WeakSet = weakref.WeakSet()
        self._config = get_config()
        self._lock = threading.RLock()
        self._background_tasks: List[asyncio.Task] = []
        self._initialized = True
        
        # Initialize default caches
        self._initialize_default_caches()
    
    def _initialize_default_caches(self) -> None:
        """Initialize default cache instances"""
        try:
            # Import here to avoid circular imports
            from .cache_strategies import MemoryCache, MultiTierCache
            
            # Create default cache configurations
            cache_configs = {
                'reasoning': {
                    'max_size_mb': 256,
                    'ttl_seconds': 3600,  # 1 hour
                    'eviction_policy': 'lru'
                },
                'embedding': {
                    'max_size_mb': 512,
                    'ttl_seconds': 604800,  # 7 days
                    'eviction_policy': 'lru'
                },
                'api_response': {
                    'max_size_mb': 128,
                    'ttl_seconds': 86400,  # 24 hours
                    'eviction_policy': 'lru'
                },
                'database': {
                    'max_size_mb': 64,
                    'ttl_seconds': 1800,  # 30 minutes
                    'eviction_policy': 'lru'
                },
                'session': {
                    'max_size_mb': 32,
                    'ttl_seconds': 7200,  # 2 hours
                    'eviction_policy': 'lru'
                }
            }
            
            for cache_name, config in cache_configs.items():
                cache_instance = MemoryCache(
                    name=cache_name,
                    max_size_bytes=config['max_size_mb'] * 1024 * 1024,
                    default_ttl=config['ttl_seconds']
                )
                self._caches[cache_name] = cache_instance
                self._cache_stats[cache_name] = CacheStats(cache_name=cache_name)
                
                logger.info(f"Initialized {cache_name} cache", 
                          max_size_mb=config['max_size_mb'],
                          ttl_seconds=config['ttl_seconds'])
                
        except Exception as e:
            logger.error(f"Failed to initialize default caches: {e}")
            raise ConfigurationError(
                "Cache initialization failed",
                config_key="cache_initialization",
                config_value=str(e)
            ) from e
    
    def get_cache(self, cache_name: str) -> Optional[Any]:
        """Get cache instance by name"""
        with self._lock:
            return self._caches.get(cache_name)
    
    def register_cache(self, cache_name: str, cache_instance: Any) -> None:
        """Register a new cache instance"""
        with self._lock:
            self._caches[cache_name] = cache_instance
            self._cache_stats[cache_name] = CacheStats(cache_name=cache_name)
            logger.info(f"Registered cache: {cache_name}")
    
    async def get(
        self, 
        cache_name: str, 
        key: str, 
        default: Optional[T] = None
    ) -> Optional[T]:
        """Get value from cache"""
        start_time = time.perf_counter()
        
        try:
            cache = self.get_cache(cache_name)
            if cache is None:
                logger.warning(f"Cache '{cache_name}' not found")
                self._cache_stats.get(cache_name, CacheStats(cache_name)).update_error()
                return default
            
            # Generate cache key
            cache_key = self._generate_key(key)
            
            # Attempt to get from cache
            if hasattr(cache, 'get_async'):
                result = await cache.get_async(cache_key)
            else:
                result = cache.get(cache_key)
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            if result is not None:
                self._cache_stats[cache_name].update_hit(response_time)
                logger.debug(f"Cache hit: {cache_name}:{cache_key[:16]}...")
                return result
            else:
                self._cache_stats[cache_name].update_miss(response_time)
                logger.debug(f"Cache miss: {cache_name}:{cache_key[:16]}...")
                return default
                
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            self._cache_stats.get(cache_name, CacheStats(cache_name)).update_error()
            logger.error(f"Cache get error for {cache_name}: {e}")
            return default
    
    async def set(
        self,
        cache_name: str,
        key: str,
        value: T,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            cache = self.get_cache(cache_name)
            if cache is None:
                logger.warning(f"Cache '{cache_name}' not found")
                return False
            
            cache_key = self._generate_key(key)
            
            if hasattr(cache, 'set_async'):
                success = await cache.set_async(cache_key, value, ttl_seconds)
            else:
                success = cache.set(cache_key, value, ttl_seconds)
            
            if success:
                logger.debug(f"Cache set: {cache_name}:{cache_key[:16]}...")
                # Notify subscribers of cache update
                self._notify_subscribers('cache_set', cache_name, cache_key)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set error for {cache_name}: {e}")
            return False
    
    async def invalidate(self, cache_name: str, key: Optional[str] = None) -> bool:
        """Invalidate cache entries"""
        try:
            cache = self.get_cache(cache_name)
            if cache is None:
                return False
            
            if key is None:
                # Clear entire cache
                if hasattr(cache, 'clear_async'):
                    await cache.clear_async()
                else:
                    cache.clear()
                logger.info(f"Cleared entire cache: {cache_name}")
            else:
                # Remove specific key
                cache_key = self._generate_key(key)
                if hasattr(cache, 'delete_async'):
                    await cache.delete_async(cache_key)
                else:
                    cache.delete(cache_key)
                logger.debug(f"Invalidated cache key: {cache_name}:{cache_key[:16]}...")
            
            self._notify_subscribers('cache_invalidate', cache_name, key)
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation error for {cache_name}: {e}")
            return False
    
    def _generate_key(self, key: Union[str, Dict, List, tuple]) -> str:
        """Generate normalized cache key"""
        if isinstance(key, str):
            return key
        
        # Convert complex objects to JSON and hash
        try:
            json_str = json.dumps(key, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.sha256(str(key).encode()).hexdigest()
    
    def get_stats(self, cache_name: Optional[str] = None) -> Dict[str, CacheStats]:
        """Get cache statistics"""
        if cache_name:
            stats = self._cache_stats.get(cache_name)
            return {cache_name: stats} if stats else {}
        
        return dict(self._cache_stats)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cache statistics"""
        total_requests = sum(stats.total_requests for stats in self._cache_stats.values())
        total_hits = sum(stats.cache_hits for stats in self._cache_stats.values())
        total_size = sum(stats.total_size_bytes for stats in self._cache_stats.values())
        
        return {
            'total_caches': len(self._caches),
            'total_requests': total_requests,
            'total_hits': total_hits,
            'global_hit_rate': total_hits / max(total_requests, 1),
            'total_size_bytes': total_size,
            'cache_names': list(self._caches.keys())
        }
    
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to cache events"""
        self._subscribers.add(callback)
    
    def _notify_subscribers(self, event_type: str, cache_name: str, key: Optional[str] = None) -> None:
        """Notify subscribers of cache events"""
        for callback in self._subscribers:
            try:
                callback(event_type, cache_name, key)
            except Exception as e:
                logger.error(f"Error notifying cache subscriber: {e}")
    
    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Cache cleanup task
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._background_tasks.append(cleanup_task)
        
        # Statistics update task
        stats_task = asyncio.create_task(self._periodic_stats_update())
        self._background_tasks.append(stats_task)
        
        logger.info("Started cache background tasks")
    
    async def stop_background_tasks(self) -> None:
        """Stop background maintenance tasks"""
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("Stopped cache background tasks")
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cache cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                for cache_name, cache in self._caches.items():
                    if hasattr(cache, 'cleanup'):
                        try:
                            if hasattr(cache, 'cleanup_async'):
                                await cache.cleanup_async()
                            else:
                                cache.cleanup()
                        except Exception as e:
                            logger.error(f"Cache cleanup error for {cache_name}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")
    
    async def _periodic_stats_update(self) -> None:
        """Periodic statistics update task"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update cache size statistics
                for cache_name, cache in self._caches.items():
                    if hasattr(cache, 'get_size'):
                        try:
                            size = cache.get_size()
                            self._cache_stats[cache_name].total_size_bytes = size
                        except Exception as e:
                            logger.error(f"Size update error for {cache_name}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic stats update error: {e}")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager