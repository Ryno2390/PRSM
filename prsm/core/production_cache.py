"""
Production Caching Layer for PRSM
=================================

Multi-level caching implementation to improve response times
and reduce database load for high concurrent usage.
"""

import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
from functools import wraps
import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)

class ProductionCacheManager:
    """Production-grade caching manager with multiple cache levels"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0
        }
    
    async def initialize(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize cache connections"""
        try:
            self.redis_client = await redis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            await self.redis_client.ping()
            logger.info("✅ Redis cache connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed, using local cache only: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (tries Redis first, then local)"""
        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Try local cache
        if key in self.local_cache:
            entry = self.local_cache[key]
            if entry["expires"] > datetime.now():
                self.cache_stats["hits"] += 1
                return entry["value"]
            else:
                del self.local_cache[key]
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache (both Redis and local)"""
        self.cache_stats["sets"] += 1
        
        # Set in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Set in local cache (with size limit)
        if len(self.local_cache) < 1000:  # Limit local cache size
            self.local_cache[key] = {
                "value": value,
                "expires": datetime.now() + timedelta(seconds=ttl)
            }
    
    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        self.cache_stats["invalidations"] += 1
        
        # Invalidate Redis
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis invalidation failed: {e}")
        
        # Invalidate local cache
        keys_to_delete = [k for k in self.local_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self.local_cache[key]
    
    def cached(self, ttl: int = 300, prefix: str = "default"):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = self._generate_cache_key(prefix, func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / max(1, total_requests)) * 100
        
        return {
            **self.cache_stats,
            "hit_rate_percentage": round(hit_rate, 2),
            "local_cache_size": len(self.local_cache)
        }

# Global cache manager instance
_cache_manager: Optional[ProductionCacheManager] = None

async def get_cache_manager() -> ProductionCacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ProductionCacheManager()
        await _cache_manager.initialize()
    return _cache_manager

# Commonly used cache decorators
def cache_api_response(ttl: int = 300):
    """Cache API response for specified TTL"""
    async def get_cache():
        return await get_cache_manager()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache()
            return await cache_manager.cached(ttl=ttl, prefix="api")(func)(*args, **kwargs)
        return wrapper
    return decorator

def cache_database_query(ttl: int = 600):
    """Cache database query results"""
    async def get_cache():
        return await get_cache_manager()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache()
            return await cache_manager.cached(ttl=ttl, prefix="db")(func)(*args, **kwargs)
        return wrapper
    return decorator