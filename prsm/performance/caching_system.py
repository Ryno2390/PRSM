"""
PRSM Comprehensive Caching System
Multi-layer caching with Redis clustering, intelligent invalidation, and performance monitoring
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import hashlib
import pickle
import time
import statistics
from collections import defaultdict, deque
import redis.asyncio as aioredis
from redis.asyncio.cluster import RedisCluster
import logging
from functools import wraps
import inspect

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layer types"""
    L1_MEMORY = "l1_memory"      # In-memory cache (fastest)
    L2_REDIS = "l2_redis"        # Redis cache (shared)
    L3_CDN = "l3_cdn"           # CDN cache (edge)


class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    TTL = "ttl"                 # Time-to-live based
    LRU = "lru"                 # Least recently used
    LFU = "lfu"                 # Least frequently used
    WRITE_THROUGH = "write_through"    # Write to cache and storage
    WRITE_BEHIND = "write_behind"      # Write to cache, async to storage
    REFRESH_AHEAD = "refresh_ahead"    # Proactive refresh before expiry


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    NEVER = "never"
    ON_MEMORY_PRESSURE = "on_memory_pressure"
    ON_EXPIRY = "on_expiry"
    ON_INVALIDATION = "on_invalidation"


@dataclass
class CacheKey:
    """Structured cache key"""
    namespace: str
    identifier: str
    version: str = "v1"
    params_hash: Optional[str] = None
    
    def __str__(self) -> str:
        key_parts = [self.namespace, self.identifier, self.version]
        if self.params_hash:
            key_parts.append(self.params_hash)
        return ":".join(key_parts)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    avg_response_time_ms: float = 0.0
    hit_ratio: float = 0.0
    key_count: int = 0
    
    def update_hit_ratio(self):
        total_requests = self.hits + self.misses
        self.hit_ratio = self.hits / total_requests if total_requests > 0 else 0.0


class MemoryCache:
    """L1 in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU tracking
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.expires_at and datetime.now(timezone.utc) > entry.expires_at:
                    await self._remove_key(key)
                    self.stats.misses += 1
                    return None
                
                # Update access tracking
                entry.access_count += 1
                entry.last_accessed = datetime.now(timezone.utc)
                
                # Move to end of access order (most recently used)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.stats.hits += 1
                self.stats.update_hit_ratio()
                return entry.value
            
            self.stats.misses += 1
            self.stats.update_hit_ratio()
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 tags: Optional[Set[str]] = None) -> bool:
        """Set value in memory cache"""
        async with self._lock:
            try:
                # Calculate size
                size_bytes = len(pickle.dumps(value)) if value is not None else 0
                
                # Check memory limits
                if size_bytes > self.max_memory_bytes:
                    logger.warning(f"Value too large for memory cache: {size_bytes} bytes")
                    return False
                
                # Create cache entry
                expires_at = None
                if ttl_seconds:
                    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(timezone.utc),
                    expires_at=expires_at,
                    size_bytes=size_bytes,
                    tags=tags or set()
                )
                
                # Evict if necessary
                await self._ensure_capacity(size_bytes)
                
                # Store entry
                self.cache[key] = entry
                self.access_order.append(key)
                
                # Update stats
                self.stats.memory_usage_bytes += size_bytes
                self.stats.key_count = len(self.cache)
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting memory cache: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        async with self._lock:
            return await self._remove_key(key)
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with matching tags"""
        async with self._lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                await self._remove_key(key)
            
            return len(keys_to_remove)
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = CacheStats()
    
    async def _remove_key(self, key: str) -> bool:
        """Remove key from cache"""
        if key not in self.cache:
            return False
        
        entry = self.cache.pop(key)
        self.stats.memory_usage_bytes -= entry.size_bytes
        self.stats.evictions += 1
        self.stats.key_count = len(self.cache)
        
        if key in self.access_order:
            self.access_order.remove(key)
        
        return True
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry"""
        # Check size limit
        while (len(self.cache) >= self.max_size or 
               self.stats.memory_usage_bytes + new_entry_size > self.max_memory_bytes):
            
            if not self.access_order:
                break
            
            # Remove least recently used
            lru_key = self.access_order.popleft()
            await self._remove_key(lru_key)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats


class RedisClusterCache:
    """L2 Redis cluster cache with high availability"""
    
    def __init__(self, cluster_nodes: List[Dict[str, Any]], 
                 password: Optional[str] = None,
                 max_connections_per_node: int = 20):
        self.cluster_nodes = cluster_nodes
        self.password = password
        self.max_connections_per_node = max_connections_per_node
        self.cluster: Optional[RedisCluster] = None
        self.stats = CacheStats()
        self.response_times = deque(maxlen=1000)  # Track response times
    
    async def initialize(self):
        """Initialize Redis cluster connection"""
        try:
            startup_nodes = [
                {"host": node["host"], "port": node["port"]} 
                for node in self.cluster_nodes
            ]
            
            self.cluster = RedisCluster(
                startup_nodes=startup_nodes,
                password=self.password,
                max_connections_per_node=self.max_connections_per_node,
                skip_full_coverage_check=True,
                decode_responses=False,  # Handle binary data
                health_check_interval=30
            )
            
            # Test connection
            await self.cluster.ping()
            logger.info(f"✅ Redis cluster initialized with {len(startup_nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cluster: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cluster"""
        if not self.cluster:
            return None
        
        start_time = time.time()
        try:
            data = await self.cluster.get(key)
            
            if data is not None:
                value = pickle.loads(data)
                self.stats.hits += 1
                
                # Update access count
                await self.cluster.incr(f"{key}:access_count")
                await self.cluster.expire(f"{key}:access_count", 86400)  # 24 hour TTL
                
                response_time_ms = (time.time() - start_time) * 1000
                self.response_times.append(response_time_ms)
                self._update_avg_response_time()
                
                self.stats.update_hit_ratio()
                return value
            
            self.stats.misses += 1
            self.stats.update_hit_ratio()
            return None
            
        except Exception as e:
            logger.error(f"Redis cluster get error: {e}")
            self.stats.misses += 1
            self.stats.update_hit_ratio()
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 tags: Optional[Set[str]] = None) -> bool:
        """Set value in Redis cluster"""
        if not self.cluster:
            return False
        
        try:
            # Serialize value
            data = pickle.dumps(value)
            
            # Set main key
            if ttl_seconds:
                success = await self.cluster.setex(key, ttl_seconds, data)
            else:
                success = await self.cluster.set(key, data)
            
            if success and tags:
                # Store tags for invalidation
                pipeline = self.cluster.pipeline()
                for tag in tags:
                    pipeline.sadd(f"tag:{tag}", key)
                    if ttl_seconds:
                        pipeline.expire(f"tag:{tag}", ttl_seconds)
                await pipeline.execute()
            
            if success:
                self.stats.key_count = await self.cluster.dbsize()
            
            return bool(success)
            
        except Exception as e:
            logger.error(f"Redis cluster set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cluster"""
        if not self.cluster:
            return False
        
        try:
            deleted = await self.cluster.delete(key)
            if deleted:
                # Clean up access count
                await self.cluster.delete(f"{key}:access_count")
                self.stats.evictions += 1
                self.stats.key_count = await self.cluster.dbsize()
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Redis cluster delete error: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all keys with matching tags"""
        if not self.cluster:
            return 0
        
        try:
            keys_to_delete = set()
            
            # Get all keys for each tag
            for tag in tags:
                tag_keys = await self.cluster.smembers(f"tag:{tag}")
                keys_to_delete.update(tag_keys)
            
            if keys_to_delete:
                # Delete keys and clean up tags
                pipeline = self.cluster.pipeline()
                for key in keys_to_delete:
                    pipeline.delete(key)
                    pipeline.delete(f"{key}:access_count")
                
                for tag in tags:
                    pipeline.delete(f"tag:{tag}")
                
                await pipeline.execute()
                
                self.stats.evictions += len(keys_to_delete)
                self.stats.key_count = await self.cluster.dbsize()
            
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Redis cluster tag invalidation error: {e}")
            return 0
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        if not self.cluster:
            return 0
        
        try:
            # Use SCAN to find keys with namespace prefix
            keys_to_delete = []
            cursor = 0
            
            while True:
                cursor, keys = await self.cluster.scan(
                    cursor=cursor, 
                    match=f"{namespace}:*", 
                    count=1000
                )
                keys_to_delete.extend(keys)
                
                if cursor == 0:
                    break
            
            if keys_to_delete:
                # Delete in batches
                batch_size = 1000
                for i in range(0, len(keys_to_delete), batch_size):
                    batch = keys_to_delete[i:i + batch_size]
                    await self.cluster.delete(*batch)
                
                self.stats.evictions += len(keys_to_delete)
                self.stats.key_count = await self.cluster.dbsize()
            
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Redis cluster namespace clear error: {e}")
            return 0
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get Redis cluster information"""
        if not self.cluster:
            return {}
        
        try:
            info = await self.cluster.cluster_info()
            nodes_info = await self.cluster.cluster_nodes()
            
            return {
                "cluster_state": info.get("cluster_state"),
                "cluster_slots_assigned": info.get("cluster_slots_assigned"),
                "cluster_slots_ok": info.get("cluster_slots_ok"),
                "cluster_known_nodes": info.get("cluster_known_nodes"),
                "nodes": len(nodes_info),
                "total_memory_usage": await self._get_total_memory_usage()
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            return {}
    
    async def _get_total_memory_usage(self) -> int:
        """Get total memory usage across cluster"""
        try:
            total_memory = 0
            nodes = await self.cluster.cluster_nodes()
            
            for node_id, node_info in nodes.items():
                if node_info.get("flags", []).count("master") > 0:
                    # Get memory info from master nodes
                    memory_info = await self.cluster.memory_usage("*")  # This is approximate
                    if memory_info:
                        total_memory += memory_info
            
            return total_memory
            
        except Exception:
            return 0
    
    def _update_avg_response_time(self):
        """Update average response time"""
        if self.response_times:
            self.stats.avg_response_time_ms = statistics.mean(self.response_times)
    
    async def close(self):
        """Close Redis cluster connection"""
        if self.cluster:
            await self.cluster.close()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats


class ComprehensiveCacheManager:
    """Multi-layer cache manager with intelligent routing"""
    
    def __init__(self, 
                 memory_cache_config: Optional[Dict[str, Any]] = None,
                 redis_cluster_config: Optional[Dict[str, Any]] = None):
        
        # Initialize L1 memory cache
        memory_config = memory_cache_config or {}
        self.memory_cache = MemoryCache(
            max_size=memory_config.get("max_size", 10000),
            max_memory_mb=memory_config.get("max_memory_mb", 512)
        )
        
        # Initialize L2 Redis cluster cache
        self.redis_cache: Optional[RedisClusterCache] = None
        if redis_cluster_config:
            self.redis_cache = RedisClusterCache(
                cluster_nodes=redis_cluster_config["nodes"],
                password=redis_cluster_config.get("password"),
                max_connections_per_node=redis_cluster_config.get("max_connections_per_node", 20)
            )
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.cache_strategies: Dict[str, CacheStrategy] = {}
        self.invalidation_patterns: Dict[str, List[str]] = {}
        
        # Performance monitoring
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.cache_warming_tasks: Set[asyncio.Task] = set()
    
    async def initialize(self):
        """Initialize cache manager"""
        try:
            if self.redis_cache:
                await self.redis_cache.initialize()
            
            logger.info("✅ Comprehensive cache manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def get(self, key: Union[str, CacheKey], 
                 use_layers: List[CacheLayer] = None) -> Optional[Any]:
        """Get value from cache with multi-layer fallback"""
        
        cache_key = str(key) if isinstance(key, CacheKey) else key
        layers = use_layers or [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
        
        start_time = time.time()
        
        # Try each layer in order
        for layer in layers:
            try:
                if layer == CacheLayer.L1_MEMORY:
                    value = await self.memory_cache.get(cache_key)
                    if value is not None:
                        self._record_hit(cache_key, layer, time.time() - start_time)
                        return value
                
                elif layer == CacheLayer.L2_REDIS and self.redis_cache:
                    value = await self.redis_cache.get(cache_key)
                    if value is not None:
                        # Populate higher layers
                        await self._populate_higher_layers(cache_key, value, layer, layers)
                        self._record_hit(cache_key, layer, time.time() - start_time)
                        return value
                
            except Exception as e:
                logger.error(f"Error getting from {layer.value}: {e}")
                continue
        
        self._record_miss(cache_key, time.time() - start_time)
        return None
    
    async def set(self, key: Union[str, CacheKey], value: Any,
                 ttl_seconds: Optional[int] = None,
                 tags: Optional[Set[str]] = None,
                 strategy: CacheStrategy = CacheStrategy.TTL,
                 use_layers: List[CacheLayer] = None) -> bool:
        """Set value in cache across specified layers"""
        
        cache_key = str(key) if isinstance(key, CacheKey) else key
        ttl = ttl_seconds or self.default_ttl
        layers = use_layers or [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
        
        success = True
        
        # Set in all specified layers
        for layer in layers:
            try:
                if layer == CacheLayer.L1_MEMORY:
                    layer_success = await self.memory_cache.set(cache_key, value, ttl, tags)
                    success = success and layer_success
                
                elif layer == CacheLayer.L2_REDIS and self.redis_cache:
                    layer_success = await self.redis_cache.set(cache_key, value, ttl, tags)
                    success = success and layer_success
                
            except Exception as e:
                logger.error(f"Error setting in {layer.value}: {e}")
                success = False
        
        # Store strategy for this key
        if success:
            self.cache_strategies[cache_key] = strategy
        
        return success
    
    async def delete(self, key: Union[str, CacheKey],
                    use_layers: List[CacheLayer] = None) -> bool:
        """Delete key from specified cache layers"""
        
        cache_key = str(key) if isinstance(key, CacheKey) else key
        layers = use_layers or [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
        
        success = True
        
        for layer in layers:
            try:
                if layer == CacheLayer.L1_MEMORY:
                    await self.memory_cache.delete(cache_key)
                
                elif layer == CacheLayer.L2_REDIS and self.redis_cache:
                    await self.redis_cache.delete(cache_key)
                
            except Exception as e:
                logger.error(f"Error deleting from {layer.value}: {e}")
                success = False
        
        # Clean up strategy
        self.cache_strategies.pop(cache_key, None)
        
        return success
    
    async def invalidate_by_tags(self, tags: Set[str],
                               use_layers: List[CacheLayer] = None) -> Dict[str, int]:
        """Invalidate cache entries by tags across layers"""
        
        layers = use_layers or [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
        results = {}
        
        for layer in layers:
            try:
                if layer == CacheLayer.L1_MEMORY:
                    count = await self.memory_cache.invalidate_by_tags(tags)
                    results[layer.value] = count
                
                elif layer == CacheLayer.L2_REDIS and self.redis_cache:
                    count = await self.redis_cache.invalidate_by_tags(tags)
                    results[layer.value] = count
                
            except Exception as e:
                logger.error(f"Error invalidating tags in {layer.value}: {e}")
                results[layer.value] = 0
        
        return results
    
    async def clear_namespace(self, namespace: str,
                            use_layers: List[CacheLayer] = None) -> Dict[str, int]:
        """Clear all keys in a namespace across layers"""
        
        layers = use_layers or [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS]
        results = {}
        
        for layer in layers:
            try:
                if layer == CacheLayer.L1_MEMORY:
                    # Memory cache doesn't have namespace clearing, so clear all
                    await self.memory_cache.clear()
                    results[layer.value] = 0  # Can't count specifically
                
                elif layer == CacheLayer.L2_REDIS and self.redis_cache:
                    count = await self.redis_cache.clear_namespace(namespace)
                    results[layer.value] = count
                
            except Exception as e:
                logger.error(f"Error clearing namespace in {layer.value}: {e}")
                results[layer.value] = 0
        
        return results
    
    async def warm_cache(self, warming_func: Callable, 
                        keys: List[Union[str, CacheKey]],
                        batch_size: int = 100) -> int:
        """Warm cache with data using provided function"""
        
        warmed_count = 0
        
        # Process keys in batches
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            
            # Create warming task for this batch
            task = asyncio.create_task(self._warm_batch(warming_func, batch))
            self.cache_warming_tasks.add(task)
            
            try:
                batch_count = await task
                warmed_count += batch_count
            except Exception as e:
                logger.error(f"Cache warming batch failed: {e}")
            finally:
                self.cache_warming_tasks.discard(task)
        
        logger.info(f"Cache warming completed: {warmed_count} keys warmed")
        return warmed_count
    
    async def _warm_batch(self, warming_func: Callable, 
                         keys: List[Union[str, CacheKey]]) -> int:
        """Warm a batch of cache keys"""
        
        warmed_count = 0
        
        for key in keys:
            try:
                cache_key = str(key) if isinstance(key, CacheKey) else key
                
                # Check if already cached
                if await self.get(key) is not None:
                    continue
                
                # Generate value using warming function
                if inspect.iscoroutinefunction(warming_func):
                    value = await warming_func(cache_key)
                else:
                    value = warming_func(cache_key)
                
                if value is not None:
                    await self.set(key, value)
                    warmed_count += 1
                
            except Exception as e:
                logger.error(f"Error warming key {key}: {e}")
        
        return warmed_count
    
    async def _populate_higher_layers(self, key: str, value: Any, 
                                    current_layer: CacheLayer,
                                    all_layers: List[CacheLayer]):
        """Populate higher-priority cache layers"""
        
        # Find layers with higher priority than current
        layer_priority = {
            CacheLayer.L1_MEMORY: 1,
            CacheLayer.L2_REDIS: 2,
            CacheLayer.L3_CDN: 3
        }
        
        current_priority = layer_priority[current_layer]
        
        for layer in all_layers:
            if layer_priority[layer] < current_priority:
                try:
                    if layer == CacheLayer.L1_MEMORY:
                        await self.memory_cache.set(key, value, self.default_ttl)
                except Exception as e:
                    logger.error(f"Error populating {layer.value}: {e}")
    
    def _record_hit(self, key: str, layer: CacheLayer, response_time: float):
        """Record cache hit metrics"""
        self.performance_metrics[f"{layer.value}_hits"].append({
            "key": key,
            "timestamp": time.time(),
            "response_time": response_time
        })
    
    def _record_miss(self, key: str, response_time: float):
        """Record cache miss metrics"""
        self.performance_metrics["misses"].append({
            "key": key,
            "timestamp": time.time(),
            "response_time": response_time
        })
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        stats = {
            "memory_cache": self.memory_cache.get_stats().__dict__,
            "redis_cache": self.redis_cache.get_stats().__dict__ if self.redis_cache else None,
            "performance_metrics": {
                metric: len(values) for metric, values in self.performance_metrics.items()
            },
            "cache_strategies": len(self.cache_strategies),
            "active_warming_tasks": len(self.cache_warming_tasks)
        }
        
        # Add Redis cluster info if available
        if self.redis_cache:
            stats["redis_cluster_info"] = await self.redis_cache.get_cluster_info()
        
        return stats
    
    async def close(self):
        """Close cache manager and cleanup resources"""
        
        # Cancel warming tasks
        for task in self.cache_warming_tasks:
            if not task.done():
                task.cancel()
        
        if self.cache_warming_tasks:
            await asyncio.gather(*self.cache_warming_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_cache:
            await self.redis_cache.close()
        
        # Clear memory cache
        await self.memory_cache.clear()
        
        logger.info("Cache manager closed")


def cache_result(key_func: Optional[Callable] = None,
                ttl_seconds: Optional[int] = None,
                tags: Optional[Set[str]] = None,
                strategy: CacheStrategy = CacheStrategy.TTL,
                use_layers: Optional[List[CacheLayer]] = None):
    """Decorator for caching function results"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Generate key from function name and arguments
                params_str = f"{args}:{sorted(kwargs.items())}"
                params_hash = hashlib.md5(params_str.encode()).hexdigest()
                cache_key = f"func:{func.__name__}:{params_hash}"
            
            # Try to get from cache
            cached_value = await cache_manager.get(cache_key, use_layers)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(
                cache_key, result, ttl_seconds, tags, strategy, use_layers
            )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.create_task(async_wrapper(*args, **kwargs))
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global cache manager instance
cache_manager: Optional[ComprehensiveCacheManager] = None


async def initialize_cache_system(memory_config: Optional[Dict[str, Any]] = None,
                                redis_cluster_config: Optional[Dict[str, Any]] = None):
    """Initialize the global cache system"""
    global cache_manager
    
    cache_manager = ComprehensiveCacheManager(
        memory_cache_config=memory_config,
        redis_cluster_config=redis_cluster_config
    )
    
    await cache_manager.initialize()
    logger.info("✅ Comprehensive cache system initialized")


def get_cache_manager() -> ComprehensiveCacheManager:
    """Get the global cache manager instance"""
    if cache_manager is None:
        raise RuntimeError("Cache system not initialized. Call initialize_cache_system() first.")
    return cache_manager


async def shutdown_cache_system():
    """Shutdown the cache system"""
    if cache_manager:
        await cache_manager.close()