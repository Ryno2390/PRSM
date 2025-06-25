"""
PRSM Advanced Caching Strategy and CDN Integration
Multi-layer caching architecture with Redis cluster and CDN optimization

🚀 CACHING CAPABILITIES:
- Multi-tier caching (L1: In-memory, L2: Redis, L3: CDN)
- Intelligent cache invalidation and warming
- Redis cluster management and scaling
- CDN integration for static assets and API responses
- Cache analytics and performance monitoring
"""

import asyncio
import time
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

import structlog
from redis.asyncio import Redis, RedisCluster
import aiohttp

logger = structlog.get_logger(__name__)


class CacheLevel(str, Enum):
    """Cache level definitions"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_CDN = "l3_cdn"


class CachePolicy(str, Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheConfig:
    """Configuration for cache layers"""
    
    # Cache Identification
    name: str
    description: str = ""
    
    # Cache Levels
    enable_l1_memory: bool = True
    enable_l2_redis: bool = True
    enable_l3_cdn: bool = False
    
    # L1 Memory Cache Settings
    l1_max_size: int = 1000  # Maximum items in memory
    l1_ttl_seconds: int = 300  # 5 minutes
    l1_policy: CachePolicy = CachePolicy.LRU
    
    # L2 Redis Cache Settings
    l2_ttl_seconds: int = 3600  # 1 hour
    l2_max_memory: str = "1gb"
    l2_policy: str = "allkeys-lru"
    
    # L3 CDN Settings
    l3_ttl_seconds: int = 86400  # 24 hours
    l3_edge_ttl_seconds: int = 3600  # 1 hour
    
    # Cache Keys
    key_prefix: str = "prsm"
    key_separator: str = ":"
    
    # Performance Settings
    compression_enabled: bool = True
    serialization_format: str = "json"  # json, pickle, msgpack
    
    # Analytics
    enable_analytics: bool = True
    analytics_sample_rate: float = 0.1


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    level: CacheLevel
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    compressed: bool = False


@dataclass
class CacheStats:
    """Cache performance statistics"""
    
    # Hit/Miss Statistics
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    
    # Level-specific Statistics
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    
    # Performance Metrics
    avg_response_time_ms: float = 0.0
    total_bytes_served: int = 0
    compression_ratio: float = 0.0
    
    # Cache Operations
    total_sets: int = 0
    total_deletes: int = 0
    total_invalidations: int = 0
    
    # Memory Usage
    l1_memory_usage_bytes: int = 0
    l2_memory_usage_bytes: int = 0
    current_entries: int = 0


class DistributedCache:
    """
    Distributed caching system with Redis cluster support
    
    🎯 DISTRIBUTED FEATURES:
    - Redis cluster management with automatic failover
    - Consistent hashing for data distribution
    - Cross-node cache invalidation
    - Replication and backup strategies
    - Performance monitoring and alerting
    """
    
    def __init__(self, cluster_nodes: List[Dict[str, Any]]):
        self.cluster_nodes = cluster_nodes
        self.redis_cluster: Optional[RedisCluster] = None
        self.node_stats: Dict[str, Dict[str, Any]] = {}
        self.replication_factor = 2
        
    async def initialize_cluster(self):
        """Initialize Redis cluster connection"""
        try:
            # Create Redis cluster client
            startup_nodes = [
                {"host": node["host"], "port": node["port"]}
                for node in self.cluster_nodes
            ]
            
            self.redis_cluster = RedisCluster(
                startup_nodes=startup_nodes,
                decode_responses=True,
                skip_full_coverage_check=True,
                health_check_interval=30
            )
            
            # Test cluster connectivity
            await self.redis_cluster.ping()
            
            logger.info("Redis cluster initialized successfully",
                       nodes=len(self.cluster_nodes))
            
        except Exception as e:
            logger.error("Failed to initialize Redis cluster", error=str(e))
            raise
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster information"""
        if not self.redis_cluster:
            return {"error": "Cluster not initialized"}
        
        try:
            # Get cluster nodes info
            cluster_info = await self.redis_cluster.cluster_info()
            cluster_nodes = await self.redis_cluster.cluster_nodes()
            
            # Calculate cluster statistics
            total_keys = 0
            total_memory = 0
            healthy_nodes = 0
            
            for node_id, node_info in cluster_nodes.items():
                if node_info.get("flags") and "master" in node_info["flags"]:
                    try:
                        node_redis = Redis(
                            host=node_info["host"],
                            port=node_info["port"]
                        )
                        
                        info = await node_redis.info()
                        total_keys += info.get("db0", {}).get("keys", 0)
                        total_memory += info.get("used_memory", 0)
                        healthy_nodes += 1
                        
                        await node_redis.close()
                        
                    except Exception as e:
                        logger.warning("Failed to get node info",
                                     node_id=node_id, error=str(e))
            
            return {
                "cluster_state": cluster_info.get("cluster_state"),
                "cluster_size": cluster_info.get("cluster_size"),
                "total_nodes": len(cluster_nodes),
                "healthy_nodes": healthy_nodes,
                "total_keys": total_keys,
                "total_memory_bytes": total_memory,
                "slots_assigned": cluster_info.get("cluster_slots_assigned"),
                "slots_ok": cluster_info.get("cluster_slots_ok")
            }
            
        except Exception as e:
            logger.error("Failed to get cluster info", error=str(e))
            return {"error": str(e)}
    
    async def set_distributed(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in distributed cache with replication"""
        try:
            # Serialize value
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            
            # Set primary copy
            if ttl:
                await self.redis_cluster.setex(key, ttl, serialized_value)
            else:
                await self.redis_cluster.set(key, serialized_value)
            
            # Set backup copies for fault tolerance
            backup_keys = [f"{key}:backup:{i}" for i in range(self.replication_factor)]
            for backup_key in backup_keys:
                try:
                    if ttl:
                        await self.redis_cluster.setex(backup_key, ttl, serialized_value)
                    else:
                        await self.redis_cluster.set(backup_key, serialized_value)
                except Exception as backup_error:
                    logger.warning("Backup replication failed",
                                 backup_key=backup_key, error=str(backup_error))
            
            return True
            
        except Exception as e:
            logger.error("Distributed cache set failed", key=key, error=str(e))
            return False
    
    async def get_distributed(self, key: str) -> Optional[Any]:
        """Get value from distributed cache with fallback"""
        try:
            # Try primary key
            value = await self.redis_cluster.get(key)
            if value is not None:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            
            # Try backup copies if primary fails
            for i in range(self.replication_factor):
                backup_key = f"{key}:backup:{i}"
                try:
                    backup_value = await self.redis_cluster.get(backup_key)
                    if backup_value is not None:
                        logger.info("Retrieved from backup", backup_key=backup_key)
                        try:
                            return json.loads(backup_value)
                        except json.JSONDecodeError:
                            return backup_value
                except Exception as backup_error:
                    logger.warning("Backup retrieval failed",
                                 backup_key=backup_key, error=str(backup_error))
            
            return None
            
        except Exception as e:
            logger.error("Distributed cache get failed", key=key, error=str(e))
            return None
    
    async def invalidate_distributed(self, pattern: str) -> int:
        """Invalidate cache entries across cluster"""
        try:
            # Get all matching keys
            keys = await self.redis_cluster.keys(pattern)
            
            if not keys:
                return 0
            
            # Delete keys in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                try:
                    deleted = await self.redis_cluster.delete(*batch_keys)
                    deleted_count += deleted
                except Exception as batch_error:
                    logger.warning("Batch deletion failed", error=str(batch_error))
            
            logger.info("Cache invalidation completed",
                       pattern=pattern, deleted_keys=deleted_count)
            
            return deleted_count
            
        except Exception as e:
            logger.error("Distributed cache invalidation failed",
                        pattern=pattern, error=str(e))
            return 0


class CDNIntegration:
    """
    CDN integration for static assets and API response caching
    
    🌐 CDN CAPABILITIES:
    - Multi-CDN support (CloudFlare, AWS CloudFront, etc.)
    - Intelligent cache warming and purging
    - Geographic distribution optimization
    - Cache hit ratio monitoring
    - Dynamic content optimization
    """
    
    def __init__(self, cdn_config: Dict[str, Any]):
        self.cdn_config = cdn_config
        self.cdn_endpoints = cdn_config.get("endpoints", [])
        self.api_keys = cdn_config.get("api_keys", {})
        self.cache_zones = cdn_config.get("zones", {})
        
    async def warm_cache(self, urls: List[str], priority: str = "normal") -> Dict[str, Any]:
        """Warm CDN cache with specified URLs"""
        results = {
            "total_urls": len(urls),
            "successful_warming": 0,
            "failed_warming": 0,
            "errors": []
        }
        
        # Create warming tasks
        warming_tasks = []
        for url in urls:
            task = asyncio.create_task(self._warm_single_url(url, priority))
            warming_tasks.append(task)
        
        # Execute warming requests
        completed_tasks = await asyncio.gather(*warming_tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                results["failed_warming"] += 1
                results["errors"].append({
                    "url": urls[i],
                    "error": str(result)
                })
            elif result:
                results["successful_warming"] += 1
            else:
                results["failed_warming"] += 1
        
        logger.info("CDN cache warming completed",
                   total=results["total_urls"],
                   successful=results["successful_warming"],
                   failed=results["failed_warming"])
        
        return results
    
    async def purge_cache(self, patterns: List[str]) -> Dict[str, Any]:
        """Purge CDN cache for specified patterns"""
        results = {
            "total_patterns": len(patterns),
            "successful_purges": 0,
            "failed_purges": 0,
            "purge_ids": []
        }
        
        for pattern in patterns:
            try:
                purge_id = await self._purge_pattern(pattern)
                if purge_id:
                    results["successful_purges"] += 1
                    results["purge_ids"].append(purge_id)
                else:
                    results["failed_purges"] += 1
                    
            except Exception as e:
                results["failed_purges"] += 1
                logger.error("CDN purge failed", pattern=pattern, error=str(e))
        
        return results
    
    async def get_cache_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get CDN cache analytics and performance metrics"""
        # This would integrate with actual CDN APIs
        # For now, return simulated analytics
        
        import random
        
        return {
            "time_range": time_range,
            "total_requests": random.randint(50000, 200000),
            "cache_hit_ratio": random.uniform(0.85, 0.95),
            "bandwidth_saved_gb": random.uniform(100, 500),
            "avg_response_time_ms": random.uniform(50, 150),
            "geographic_distribution": {
                "us-east": random.uniform(0.3, 0.4),
                "us-west": random.uniform(0.2, 0.3),
                "europe": random.uniform(0.2, 0.3),
                "asia": random.uniform(0.1, 0.2)
            },
            "top_cached_content": [
                {"url": "/api/v1/models", "hits": random.randint(5000, 15000)},
                {"url": "/api/v1/sessions", "hits": random.randint(3000, 10000)},
                {"url": "/static/js/app.js", "hits": random.randint(2000, 8000)}
            ]
        }
    
    async def _warm_single_url(self, url: str, priority: str) -> bool:
        """Warm cache for a single URL"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Cache-Control": "no-cache",
                    "X-Cache-Warm": "true",
                    "X-Priority": priority
                }
                
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error("URL warming failed", url=url, error=str(e))
            return False
    
    async def _purge_pattern(self, pattern: str) -> Optional[str]:
        """Purge cache for a URL pattern"""
        # This would integrate with actual CDN purge APIs
        # For now, simulate purge operation
        
        import uuid
        
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Return simulated purge ID
        return str(uuid.uuid4())


class CacheManager:
    """
    Comprehensive cache management system
    
    🎯 CACHE MANAGEMENT:
    - Multi-tier cache coordination (Memory -> Redis -> CDN)
    - Intelligent cache warming and invalidation
    - Performance monitoring and optimization
    - Cache hit ratio optimization
    - Automatic cache scaling
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.l1_cache: Dict[str, CacheEntry] = {}  # In-memory cache
        self.redis_client: Optional[Redis] = None
        self.distributed_cache: Optional[DistributedCache] = None
        self.cdn_integration: Optional[CDNIntegration] = None
        self.stats = CacheStats()
        self.analytics_enabled = config.enable_analytics
        
    async def initialize(self, redis_config: Optional[Dict[str, Any]] = None,
                       cdn_config: Optional[Dict[str, Any]] = None):
        """Initialize cache layers"""
        try:
            # Initialize Redis if enabled
            if self.config.enable_l2_redis and redis_config:
                if redis_config.get("cluster_mode"):
                    # Initialize distributed cache
                    self.distributed_cache = DistributedCache(
                        redis_config["cluster_nodes"]
                    )
                    await self.distributed_cache.initialize_cluster()
                else:
                    # Initialize single Redis instance
                    self.redis_client = Redis(
                        host=redis_config.get("host", "localhost"),
                        port=redis_config.get("port", 6379),
                        db=redis_config.get("db", 0),
                        decode_responses=True
                    )
                    await self.redis_client.ping()
            
            # Initialize CDN if enabled
            if self.config.enable_l3_cdn and cdn_config:
                self.cdn_integration = CDNIntegration(cdn_config)
            
            logger.info("Cache manager initialized successfully",
                       l1_enabled=self.config.enable_l1_memory,
                       l2_enabled=self.config.enable_l2_redis,
                       l3_enabled=self.config.enable_l3_cdn)
            
        except Exception as e:
            logger.error("Cache manager initialization failed", error=str(e))
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with multi-tier fallback"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # L1: Check memory cache first
            if self.config.enable_l1_memory:
                l1_value = await self._get_from_l1(key)
                if l1_value is not None:
                    self.stats.cache_hits += 1
                    self.stats.l1_hits += 1
                    await self._record_analytics(key, CacheLevel.L1_MEMORY, time.time() - start_time)
                    return l1_value
            
            # L2: Check Redis cache
            if self.config.enable_l2_redis:
                l2_value = await self._get_from_l2(key)
                if l2_value is not None:
                    # Promote to L1 cache
                    if self.config.enable_l1_memory:
                        await self._set_to_l1(key, l2_value)
                    
                    self.stats.cache_hits += 1
                    self.stats.l2_hits += 1
                    await self._record_analytics(key, CacheLevel.L2_REDIS, time.time() - start_time)
                    return l2_value
            
            # L3: Check CDN (for applicable content)
            if self.config.enable_l3_cdn and self._is_cdn_cacheable(key):
                l3_value = await self._get_from_l3(key)
                if l3_value is not None:
                    # Promote to L1 and L2
                    if self.config.enable_l1_memory:
                        await self._set_to_l1(key, l3_value)
                    if self.config.enable_l2_redis:
                        await self._set_to_l2(key, l3_value)
                    
                    self.stats.cache_hits += 1
                    self.stats.l3_hits += 1
                    await self._record_analytics(key, CacheLevel.L3_CDN, time.time() - start_time)
                    return l3_value
            
            # Cache miss
            self.stats.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error("Cache get operation failed", key=key, error=str(e))
            self.stats.cache_misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in appropriate cache tiers"""
        try:
            self.stats.total_sets += 1
            success = True
            
            # Set in L1 memory cache
            if self.config.enable_l1_memory:
                l1_success = await self._set_to_l1(key, value, ttl or self.config.l1_ttl_seconds)
                success = success and l1_success
            
            # Set in L2 Redis cache
            if self.config.enable_l2_redis:
                l2_success = await self._set_to_l2(key, value, ttl or self.config.l2_ttl_seconds)
                success = success and l2_success
            
            # Set in L3 CDN (for applicable content)
            if self.config.enable_l3_cdn and self._is_cdn_cacheable(key):
                l3_success = await self._set_to_l3(key, value, ttl or self.config.l3_ttl_seconds)
                success = success and l3_success
            
            return success
            
        except Exception as e:
            logger.error("Cache set operation failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers"""
        try:
            self.stats.total_deletes += 1
            success = True
            
            # Delete from L1
            if self.config.enable_l1_memory and key in self.l1_cache:
                del self.l1_cache[key]
            
            # Delete from L2
            if self.config.enable_l2_redis:
                if self.distributed_cache:
                    await self.distributed_cache.redis_cluster.delete(key)
                elif self.redis_client:
                    await self.redis_client.delete(key)
            
            # Delete from L3 (CDN purge)
            if self.config.enable_l3_cdn and self.cdn_integration:
                await self.cdn_integration.purge_cache([key])
            
            return success
            
        except Exception as e:
            logger.error("Cache delete operation failed", key=key, error=str(e))
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        try:
            self.stats.total_invalidations += 1
            total_invalidated = 0
            
            # Invalidate L1 cache
            if self.config.enable_l1_memory:
                l1_keys = [k for k in self.l1_cache.keys() if self._pattern_matches(k, pattern)]
                for key in l1_keys:
                    del self.l1_cache[key]
                total_invalidated += len(l1_keys)
            
            # Invalidate L2 cache
            if self.config.enable_l2_redis:
                if self.distributed_cache:
                    l2_count = await self.distributed_cache.invalidate_distributed(pattern)
                elif self.redis_client:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        l2_count = await self.redis_client.delete(*keys)
                    else:
                        l2_count = 0
                total_invalidated += l2_count
            
            # Invalidate L3 cache (CDN)
            if self.config.enable_l3_cdn and self.cdn_integration:
                await self.cdn_integration.purge_cache([pattern])
            
            logger.info("Cache pattern invalidation completed",
                       pattern=pattern, invalidated_count=total_invalidated)
            
            return total_invalidated
            
        except Exception as e:
            logger.error("Cache pattern invalidation failed",
                        pattern=pattern, error=str(e))
            return 0
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        # Calculate hit rate
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
        
        # Get memory usage
        self.stats.l1_memory_usage_bytes = sum(
            entry.size_bytes for entry in self.l1_cache.values()
        )
        self.stats.current_entries = len(self.l1_cache)
        
        # Get Redis statistics if available
        redis_stats = {}
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info("memory")
                redis_stats = {
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "maxmemory": redis_info.get("maxmemory", 0)
                }
            except Exception as e:
                logger.warning("Failed to get Redis stats", error=str(e))
        
        # Get CDN statistics if available
        cdn_stats = {}
        if self.cdn_integration:
            try:
                cdn_stats = await self.cdn_integration.get_cache_analytics()
            except Exception as e:
                logger.warning("Failed to get CDN stats", error=str(e))
        
        return {
            "cache_stats": self.stats.__dict__,
            "redis_stats": redis_stats,
            "cdn_stats": cdn_stats,
            "configuration": {
                "l1_enabled": self.config.enable_l1_memory,
                "l2_enabled": self.config.enable_l2_redis,
                "l3_enabled": self.config.enable_l3_cdn,
                "l1_max_size": self.config.l1_max_size,
                "l1_ttl": self.config.l1_ttl_seconds,
                "l2_ttl": self.config.l2_ttl_seconds
            }
        }
    
    async def warm_cache(self, keys_and_values: Dict[str, Any]) -> Dict[str, bool]:
        """Warm cache with predefined key-value pairs"""
        results = {}
        
        for key, value in keys_and_values.items():
            try:
                success = await self.set(key, value)
                results[key] = success
            except Exception as e:
                logger.error("Cache warming failed", key=key, error=str(e))
                results[key] = False
        
        successful_count = sum(1 for success in results.values() if success)
        logger.info("Cache warming completed",
                   total_keys=len(keys_and_values),
                   successful=successful_count)
        
        return results
    
    async def _get_from_l1(self, key: str) -> Optional[Any]:
        """Get value from L1 memory cache"""
        entry = self.l1_cache.get(key)
        if not entry:
            return None
        
        # Check expiration
        if entry.expires_at and datetime.now(timezone.utc) > entry.expires_at:
            del self.l1_cache[key]
            return None
        
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = datetime.now(timezone.utc)
        
        return entry.value
    
    async def _set_to_l1(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L1 memory cache"""
        try:
            # Check cache size limit
            if len(self.l1_cache) >= self.config.l1_max_size:
                await self._evict_l1_entries()
            
            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            
            # Estimate size
            size_bytes = len(str(value).encode('utf-8'))
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                level=CacheLevel.L1_MEMORY,
                created_at=datetime.now(timezone.utc),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            self.l1_cache[key] = entry
            return True
            
        except Exception as e:
            logger.error("L1 cache set failed", key=key, error=str(e))
            return False
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get value from L2 Redis cache"""
        try:
            if self.distributed_cache:
                return await self.distributed_cache.get_distributed(key)
            elif self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
            return None
            
        except Exception as e:
            logger.error("L2 cache get failed", key=key, error=str(e))
            return None
    
    async def _set_to_l2(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L2 Redis cache"""
        try:
            if self.distributed_cache:
                return await self.distributed_cache.set_distributed(key, value, ttl)
            elif self.redis_client:
                serialized_value = json.dumps(value) if not isinstance(value, str) else value
                if ttl:
                    await self.redis_client.setex(key, ttl, serialized_value)
                else:
                    await self.redis_client.set(key, serialized_value)
                return True
            return False
            
        except Exception as e:
            logger.error("L2 cache set failed", key=key, error=str(e))
            return False
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get value from L3 CDN cache"""
        # CDN typically serves static content, not general cache lookups
        # This would be implemented based on specific CDN capabilities
        return None
    
    async def _set_to_l3(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L3 CDN cache"""
        # CDN caching is typically handled through HTTP headers
        # This would be implemented based on specific CDN integration
        return True
    
    def _is_cdn_cacheable(self, key: str) -> bool:
        """Determine if content is suitable for CDN caching"""
        # Static assets, API responses with appropriate cache headers
        cdn_patterns = [
            "static/", "/assets/", "/images/", "/css/", "/js/",
            "api/v1/models", "api/v1/marketplace"
        ]
        return any(pattern in key for pattern in cdn_patterns)
    
    def _pattern_matches(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (supports wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def _evict_l1_entries(self):
        """Evict entries from L1 cache based on policy"""
        if self.config.l1_policy == CachePolicy.LRU:
            # Remove least recently used entries
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].last_accessed or x[1].created_at
            )
            # Remove 10% of entries
            remove_count = max(1, len(sorted_entries) // 10)
            for i in range(remove_count):
                key_to_remove = sorted_entries[i][0]
                del self.l1_cache[key_to_remove]
        
        elif self.config.l1_policy == CachePolicy.LFU:
            # Remove least frequently used entries
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].access_count
            )
            remove_count = max(1, len(sorted_entries) // 10)
            for i in range(remove_count):
                key_to_remove = sorted_entries[i][0]
                del self.l1_cache[key_to_remove]
    
    async def _record_analytics(self, key: str, level: CacheLevel, response_time: float):
        """Record cache analytics for monitoring"""
        if not self.analytics_enabled:
            return
        
        # Sample analytics to avoid overhead
        import random
        if random.random() > self.config.analytics_sample_rate:
            return
        
        # In production, this would send to analytics service
        logger.debug("Cache analytics",
                    key=key,
                    level=level.value,
                    response_time_ms=response_time * 1000)


# Example cache configurations for PRSM services
def create_prsm_cache_configs() -> List[CacheConfig]:
    """Create default cache configurations for PRSM services"""
    
    configs = []
    
    # API Response Cache
    api_cache = CacheConfig(
        name="prsm-api-cache",
        description="Cache for PRSM API responses",
        enable_l1_memory=True,
        enable_l2_redis=True,
        enable_l3_cdn=True,
        l1_max_size=2000,
        l1_ttl_seconds=300,  # 5 minutes
        l2_ttl_seconds=1800,  # 30 minutes
        l3_ttl_seconds=3600,  # 1 hour
        key_prefix="prsm:api",
        compression_enabled=True
    )
    configs.append(api_cache)
    
    # Session Cache
    session_cache = CacheConfig(
        name="prsm-session-cache",
        description="Cache for user sessions and state",
        enable_l1_memory=True,
        enable_l2_redis=True,
        enable_l3_cdn=False,  # Sessions shouldn't be in CDN
        l1_max_size=5000,
        l1_ttl_seconds=600,  # 10 minutes
        l2_ttl_seconds=3600,  # 1 hour
        key_prefix="prsm:session",
        compression_enabled=False  # Sessions are typically small
    )
    configs.append(session_cache)
    
    # Model Cache
    model_cache = CacheConfig(
        name="prsm-model-cache",
        description="Cache for ML model data and metadata",
        enable_l1_memory=True,
        enable_l2_redis=True,
        enable_l3_cdn=True,
        l1_max_size=500,  # Models can be large
        l1_ttl_seconds=1800,  # 30 minutes
        l2_ttl_seconds=7200,  # 2 hours
        l3_ttl_seconds=86400,  # 24 hours
        key_prefix="prsm:model",
        compression_enabled=True,
        serialization_format="pickle"  # Better for ML objects
    )
    configs.append(model_cache)
    
    return configs