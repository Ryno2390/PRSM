#!/usr/bin/env python3
"""
PRSM Advanced Caching System

Implements intelligent multi-level caching to reduce latency and improve
throughput for RLT components. Addresses latency bottlenecks identified
in Phase 3 testing.

IMPLEMENTATION STATUS:
- Multi-level Caching: ✅ Core algorithm implemented (Lines 78-167)
- Security Features: ✅ HMAC signatures implemented (Lines 96-98, 176-234)
- Performance Testing: ⚠️ Benchmark suite exists, production metrics pending
- Latency Measurement: ❌ Baseline measurements not yet established
- Cache Hit Ratio: ❌ Production metrics not yet available

DEVELOPMENT NOTES:
- Caching algorithms are functional and ready for testing
- Performance improvements will be measured in production deployment
- Benchmark suite available for performance validation
"""

import asyncio
import time
import hashlib
import json
import threading
import statistics
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from enum import Enum
import pickle
import zlib
import hmac
import os
import logging

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels with different characteristics"""
    L1_MEMORY = "l1_memory"        # Fastest, smallest
    L2_MEMORY = "l2_memory"        # Fast, medium size
    L3_PERSISTENT = "l3_persistent" # Slower, largest


@dataclass
class CacheItem:
    """Cache item with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if item has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of item in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class CacheConfig:
    """Configuration for a cache level"""
    max_size_mb: float
    default_ttl_seconds: int
    eviction_policy: str = "lru"  # lru, lfu, ttl, adaptive
    compression_enabled: bool = True
    serialization_format: str = "pickle"  # pickle, json
    background_cleanup_interval: int = 60
    hit_ratio_threshold: float = 0.7


class AdvancedCache:
    """
    Multi-level intelligent cache with adaptive policies
    """
    
    def __init__(self, component_id: str, config: Dict[str, Any] = None):
        self.component_id = component_id
        self.config = config or {}
        
        # Cache levels
        self.caches: Dict[CacheLevel, Dict[str, CacheItem]] = {
            CacheLevel.L1_MEMORY: OrderedDict(),
            CacheLevel.L2_MEMORY: OrderedDict(),
            CacheLevel.L3_PERSISTENT: OrderedDict()
        }
        
        # Cache configurations
        self.cache_configs = self._initialize_cache_configs()
        
        # Security: Generate or load signing key for secure pickle
        self._pickle_signing_key = os.getenv('PRSM_CACHE_SIGNING_KEY', os.urandom(32))
        
        # Performance metrics
        self.stats = {
            "total_requests": 0,
            "total_hits": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "total_misses": 0,
            "total_evictions": 0,
            "avg_response_time_ms": 0.0,
            "hit_ratio": 0.0
        }
        
        # Access patterns for adaptive caching
        self.access_patterns = defaultdict(list)
        self.popular_keys = defaultdict(int)
        
        # Thread safety
        self.locks = {
            level: threading.RLock() for level in CacheLevel
        }
        
        # Background cleanup
        self.cleanup_task = None
        self.running = False
        
        logger.info(f"Advanced cache initialized for {component_id}")
    
    def _secure_pickle_dumps(self, value: Any) -> bytes:
        """Securely serialize value with signature"""
        data = pickle.dumps(value)
        signature = hmac.new(self._pickle_signing_key, data, hashlib.sha256).digest()
        return signature + data
    
    def _secure_pickle_loads(self, signed_data: bytes) -> Any:
        """Securely deserialize value with signature verification"""
        if len(signed_data) < 32:  # 32 bytes for SHA256 signature
            raise ValueError("Invalid signed pickle data")
        
        signature = signed_data[:32]
        data = signed_data[32:]
        
        expected_signature = hmac.new(self._pickle_signing_key, data, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Pickle signature verification failed - potential tampering")
        
        return pickle.loads(data)  # nosec B301 - secure: signature verified above
    
    def _initialize_cache_configs(self) -> Dict[CacheLevel, CacheConfig]:
        """Initialize cache configurations based on component"""
        
        # Component-specific cache configurations
        component_configs = {
            "seal_service": {
                CacheLevel.L1_MEMORY: CacheConfig(max_size_mb=50, default_ttl_seconds=300),
                CacheLevel.L2_MEMORY: CacheConfig(max_size_mb=200, default_ttl_seconds=600),
                CacheLevel.L3_PERSISTENT: CacheConfig(max_size_mb=500, default_ttl_seconds=1800)
            },
            "distributed_rlt_network": {
                CacheLevel.L1_MEMORY: CacheConfig(max_size_mb=30, default_ttl_seconds=120),
                CacheLevel.L2_MEMORY: CacheConfig(max_size_mb=100, default_ttl_seconds=300),
                CacheLevel.L3_PERSISTENT: CacheConfig(max_size_mb=300, default_ttl_seconds=900)
            },
            "rlt_quality_monitor": {
                CacheLevel.L1_MEMORY: CacheConfig(max_size_mb=40, default_ttl_seconds=180),
                CacheLevel.L2_MEMORY: CacheConfig(max_size_mb=150, default_ttl_seconds=600),
                CacheLevel.L3_PERSISTENT: CacheConfig(max_size_mb=400, default_ttl_seconds=1200)
            },
            "rlt_enhanced_compiler": {
                CacheLevel.L1_MEMORY: CacheConfig(max_size_mb=60, default_ttl_seconds=600),
                CacheLevel.L2_MEMORY: CacheConfig(max_size_mb=250, default_ttl_seconds=1200),
                CacheLevel.L3_PERSISTENT: CacheConfig(max_size_mb=600, default_ttl_seconds=3600)
            },
            "rlt_claims_validator": {
                CacheLevel.L1_MEMORY: CacheConfig(max_size_mb=35, default_ttl_seconds=360),
                CacheLevel.L2_MEMORY: CacheConfig(max_size_mb=120, default_ttl_seconds=720),
                CacheLevel.L3_PERSISTENT: CacheConfig(max_size_mb=350, default_ttl_seconds=1440)
            }
        }
        
        # Default configuration
        default_config = {
            CacheLevel.L1_MEMORY: CacheConfig(max_size_mb=32, default_ttl_seconds=300),
            CacheLevel.L2_MEMORY: CacheConfig(max_size_mb=128, default_ttl_seconds=600),
            CacheLevel.L3_PERSISTENT: CacheConfig(max_size_mb=256, default_ttl_seconds=1200)
        }
        
        return component_configs.get(self.component_id, default_config)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (checks all levels)"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # Check L1 cache first (fastest)
        item = await self._get_from_level(key, CacheLevel.L1_MEMORY)
        if item is not None:
            self.stats["l1_hits"] += 1
            self.stats["total_hits"] += 1
            await self._record_access(key, CacheLevel.L1_MEMORY)
            return item
        
        # Check L2 cache
        item = await self._get_from_level(key, CacheLevel.L2_MEMORY)
        if item is not None:
            self.stats["l2_hits"] += 1
            self.stats["total_hits"] += 1
            await self._record_access(key, CacheLevel.L2_MEMORY)
            # Promote to L1 if frequently accessed
            await self._maybe_promote_to_l1(key, item)
            return item
        
        # Check L3 cache
        item = await self._get_from_level(key, CacheLevel.L3_PERSISTENT)
        if item is not None:
            self.stats["l3_hits"] += 1
            self.stats["total_hits"] += 1
            await self._record_access(key, CacheLevel.L3_PERSISTENT)
            # Promote to L2 if frequently accessed
            await self._maybe_promote_to_l2(key, item)
            return item
        
        # Cache miss
        self.stats["total_misses"] += 1
        
        # Update performance metrics
        response_time = (time.time() - start_time) * 1000
        self._update_response_time(response_time)
        
        return None
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put item in cache (starts at L1, may be demoted based on access patterns)"""
        
        # Determine cache level based on value characteristics
        cache_level = await self._determine_cache_level(key, value)
        
        # Serialize and compress if needed
        serialized_value, size_bytes = await self._serialize_value(value, cache_level)
        
        # Create cache item
        cache_item = CacheItem(
            key=key,
            value=serialized_value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl_seconds or self.cache_configs[cache_level].default_ttl_seconds,
            size_bytes=size_bytes
        )
        
        # Store in appropriate cache level
        success = await self._put_in_level(key, cache_item, cache_level)
        
        if success:
            # Track access pattern
            self.popular_keys[key] += 1
            
        return success
    
    async def _get_from_level(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Get item from specific cache level"""
        
        with self.locks[level]:
            cache = self.caches[level]
            
            if key not in cache:
                return None
            
            cache_item = cache[key]
            
            # Check if expired
            if cache_item.is_expired:
                del cache[key]
                return None
            
            # Update access info
            cache_item.last_accessed = datetime.now()
            cache_item.access_count += 1
            cache_item.hit_count += 1
            
            # Move to end for LRU
            cache.move_to_end(key)
            
            # Deserialize value
            return await self._deserialize_value(cache_item.value, level)
    
    async def _put_in_level(self, key: str, cache_item: CacheItem, level: CacheLevel) -> bool:
        """Put item in specific cache level"""
        
        config = self.cache_configs[level]
        
        with self.locks[level]:
            cache = self.caches[level]
            
            # Check if we need to evict items
            await self._ensure_capacity(level, cache_item.size_bytes)
            
            # Store item
            cache[key] = cache_item
            
            return True
    
    async def _determine_cache_level(self, key: str, value: Any) -> CacheLevel:
        """Determine appropriate cache level for new item"""
        
        # Size-based determination
        estimated_size = await self._estimate_size(value)
        
        # Small items (< 1KB) go to L1
        if estimated_size < 1024:
            return CacheLevel.L1_MEMORY
        
        # Medium items (< 10KB) go to L2
        elif estimated_size < 10240:
            return CacheLevel.L2_MEMORY
        
        # Large items go to L3
        else:
            return CacheLevel.L3_PERSISTENT
    
    async def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, dict)):
                return len(str(value))
            else:
                # Use pickle to estimate
                return len(self._secure_pickle_dumps(value))
        except:
            return 1024  # Default estimate
    
    async def _serialize_value(self, value: Any, level: CacheLevel) -> tuple[Any, int]:
        """Serialize and optionally compress value"""
        
        config = self.cache_configs[level]
        
        try:
            if config.serialization_format == "json":
                serialized = json.dumps(value, default=str)
            else:
                serialized = self._secure_pickle_dumps(value)
            
            size_bytes = len(serialized) if isinstance(serialized, bytes) else len(serialized.encode())
            
            # Compress if enabled and beneficial
            if config.compression_enabled and size_bytes > 512:
                if isinstance(serialized, str):
                    serialized = serialized.encode()
                compressed = zlib.compress(serialized)
                
                # Use compression if it saves at least 20%
                if len(compressed) < size_bytes * 0.8:
                    return compressed, len(compressed)
            
            return serialized, size_bytes
            
        except Exception as e:
            logger.error(f"Serialization error for {level}: {e}")
            return value, await self._estimate_size(value)
    
    async def _deserialize_value(self, serialized_value: Any, level: CacheLevel) -> Any:
        """Deserialize and decompress value"""
        
        config = self.cache_configs[level]
        
        try:
            data = serialized_value
            
            # Try decompression first
            if isinstance(data, bytes):
                try:
                    data = zlib.decompress(data)
                except:
                    pass  # Not compressed
            
            # Deserialize
            if config.serialization_format == "json":
                if isinstance(data, bytes):
                    data = data.decode()
                return json.loads(data)
            else:
                if isinstance(data, str):
                    data = data.encode()
                return self._secure_pickle_loads(data)
                
        except Exception as e:
            logger.error(f"Deserialization error for {level}: {e}")
            return serialized_value
    
    async def _ensure_capacity(self, level: CacheLevel, new_item_size: int):
        """Ensure cache level has capacity for new item"""
        
        config = self.cache_configs[level]
        cache = self.caches[level]
        max_size_bytes = config.max_size_mb * 1024 * 1024
        
        # Calculate current size
        current_size = sum(item.size_bytes for item in cache.values())
        
        # Evict items if necessary
        while current_size + new_item_size > max_size_bytes and cache:
            evicted_key = await self._evict_item(level)
            if evicted_key:
                current_size -= cache[evicted_key].size_bytes if evicted_key in cache else 0
                self.stats["total_evictions"] += 1
            else:
                break
    
    async def _evict_item(self, level: CacheLevel) -> Optional[str]:
        """Evict item based on eviction policy"""
        
        config = self.cache_configs[level]
        cache = self.caches[level]
        
        if not cache:
            return None
        
        if config.eviction_policy == "lru":
            # Remove least recently used (first item in OrderedDict)
            key = next(iter(cache))
            del cache[key]
            return key
        
        elif config.eviction_policy == "lfu":
            # Remove least frequently used
            lfu_key = min(cache.keys(), key=lambda k: cache[k].access_count)
            del cache[lfu_key]
            return lfu_key
        
        elif config.eviction_policy == "ttl":
            # Remove item closest to expiration
            ttl_key = min(
                cache.keys(),
                key=lambda k: cache[k].created_at + timedelta(seconds=cache[k].ttl_seconds or 0)
            )
            del cache[ttl_key]
            return ttl_key
        
        elif config.eviction_policy == "adaptive":
            # Adaptive eviction based on access patterns
            return await self._adaptive_eviction(level)
        
        else:
            # Default to LRU
            key = next(iter(cache))
            del cache[key]
            return key
    
    async def _adaptive_eviction(self, level: CacheLevel) -> Optional[str]:
        """Adaptive eviction based on multiple factors"""
        
        cache = self.caches[level]
        
        if not cache:
            return None
        
        # Score items based on multiple factors
        item_scores = {}
        current_time = datetime.now()
        
        for key, item in cache.items():
            age = (current_time - item.created_at).total_seconds()
            time_since_access = (current_time - item.last_accessed).total_seconds()
            
            # Lower score = more likely to be evicted
            score = (
                item.access_count * 0.3 +           # Frequency factor
                (1 / max(1, time_since_access)) * 0.3 +  # Recency factor
                (1 / max(1, age)) * 0.2 +           # Age factor
                item.hit_count * 0.2                # Hit ratio factor
            )
            
            item_scores[key] = score
        
        # Evict item with lowest score
        evict_key = min(item_scores.keys(), key=lambda k: item_scores[k])
        del cache[evict_key]
        return evict_key
    
    async def _maybe_promote_to_l1(self, key: str, value: Any):
        """Promote frequently accessed L2 item to L1"""
        
        # Promote if accessed more than threshold
        if self.popular_keys[key] > 5:
            estimated_size = await self._estimate_size(value)
            
            # Only promote if L1 has space or item is small
            if estimated_size < 10240:  # 10KB limit for L1 promotion
                await self.put(key, value)
    
    async def _maybe_promote_to_l2(self, key: str, value: Any):
        """Promote frequently accessed L3 item to L2"""
        
        # Promote if accessed more than threshold
        if self.popular_keys[key] > 3:
            estimated_size = await self._estimate_size(value)
            
            # Only promote if L2 has space or item is reasonably sized
            if estimated_size < 102400:  # 100KB limit for L2 promotion
                cache_item = CacheItem(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    size_bytes=estimated_size
                )
                await self._put_in_level(key, cache_item, CacheLevel.L2_MEMORY)
    
    async def _record_access(self, key: str, level: CacheLevel):
        """Record access pattern for adaptive caching"""
        
        self.access_patterns[key].append({
            "timestamp": datetime.now(),
            "level": level,
            "cache_hit": True
        })
        
        # Keep only recent access patterns (last 100)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time"""
        
        # Simple moving average
        alpha = 0.1
        self.stats["avg_response_time_ms"] = (
            alpha * response_time_ms + 
            (1 - alpha) * self.stats["avg_response_time_ms"]
        )
    
    async def start_background_cleanup(self):
        """Start background cleanup task"""
        
        if self.cleanup_task is not None:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._background_cleanup_loop())
    
    async def stop_background_cleanup(self):
        """Stop background cleanup task"""
        
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
    
    async def _background_cleanup_loop(self):
        """Background cleanup loop"""
        
        while self.running:
            try:
                await self._cleanup_expired_items()
                await self._update_cache_statistics()
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_items(self):
        """Remove expired items from all cache levels"""
        
        for level in CacheLevel:
            with self.locks[level]:
                cache = self.caches[level]
                expired_keys = [
                    key for key, item in cache.items()
                    if item.is_expired
                ]
                
                for key in expired_keys:
                    del cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired items from {level}")
    
    async def _update_cache_statistics(self):
        """Update cache statistics"""
        
        total_requests = self.stats["total_requests"]
        total_hits = self.stats["total_hits"]
        
        if total_requests > 0:
            self.stats["hit_ratio"] = total_hits / total_requests
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        # Calculate sizes for each level
        level_stats = {}
        total_items = 0
        total_size_mb = 0
        
        for level in CacheLevel:
            cache = self.caches[level]
            items_count = len(cache)
            size_bytes = sum(item.size_bytes for item in cache.values())
            size_mb = size_bytes / (1024 * 1024)
            
            level_stats[level.value] = {
                "items_count": items_count,
                "size_mb": round(size_mb, 2),
                "max_size_mb": self.cache_configs[level].max_size_mb,
                "utilization_percent": round((size_mb / self.cache_configs[level].max_size_mb) * 100, 1)
            }
            
            total_items += items_count
            total_size_mb += size_mb
        
        return {
            "component_id": self.component_id,
            "total_requests": self.stats["total_requests"],
            "total_hits": self.stats["total_hits"],
            "total_misses": self.stats["total_misses"],
            "hit_ratio": round(self.stats["hit_ratio"], 3),
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"], 
            "l3_hits": self.stats["l3_hits"],
            "total_evictions": self.stats["total_evictions"],
            "avg_response_time_ms": round(self.stats["avg_response_time_ms"], 2),
            "total_items": total_items,
            "total_size_mb": round(total_size_mb, 2),
            "level_statistics": level_stats,
            "popular_keys_count": len([k for k, count in self.popular_keys.items() if count > 2])
        }
    
    async def clear_cache(self, level: Optional[CacheLevel] = None):
        """Clear cache (all levels or specific level)"""
        
        if level:
            with self.locks[level]:
                self.caches[level].clear()
        else:
            for cache_level in CacheLevel:
                with self.locks[cache_level]:
                    self.caches[cache_level].clear()
        
        logger.info(f"Cache cleared for {self.component_id} - {level or 'all levels'}")


class CacheManager:
    """Manages caches for multiple components"""
    
    def __init__(self):
        self.component_caches: Dict[str, AdvancedCache] = {}
        self.running = False
        
    async def get_cache(self, component_id: str) -> AdvancedCache:
        """Get or create cache for component"""
        
        if component_id not in self.component_caches:
            cache = AdvancedCache(component_id)
            await cache.start_background_cleanup()
            self.component_caches[component_id] = cache
        
        return self.component_caches[component_id]
    
    async def cache_function_result(self, component_id: str, function_name: str, 
                                 args: tuple, kwargs: dict, result: Any, 
                                 ttl_seconds: Optional[int] = None) -> bool:
        """Cache function result with intelligent key generation"""
        
        cache = await self.get_cache(component_id)
        
        # Generate cache key
        cache_key = self._generate_cache_key(function_name, args, kwargs)
        
        return await cache.put(cache_key, result, ttl_seconds)
    
    async def get_cached_function_result(self, component_id: str, function_name: str,
                                       args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached function result"""
        
        cache = await self.get_cache(component_id)
        
        # Generate cache key
        cache_key = self._generate_cache_key(function_name, args, kwargs)
        
        return await cache.get(cache_key)
    
    def _generate_cache_key(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """Generate deterministic cache key from function call"""
        
        # Create deterministic string representation
        key_data = {
            "function": function_name,
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        
        # Hash for consistent key length
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all component caches"""
        
        all_stats = {}
        total_hit_ratio = 0
        total_items = 0
        total_size_mb = 0
        
        for component_id, cache in self.component_caches.items():
            stats = cache.get_cache_statistics()
            all_stats[component_id] = stats
            
            total_hit_ratio += stats["hit_ratio"]
            total_items += stats["total_items"]
            total_size_mb += stats["total_size_mb"]
        
        return {
            "component_count": len(self.component_caches),
            "total_items": total_items,
            "total_size_mb": round(total_size_mb, 2),
            "average_hit_ratio": round(total_hit_ratio / max(1, len(self.component_caches)), 3),
            "component_statistics": all_stats
        }
    
    async def shutdown(self):
        """Shutdown all caches"""
        
        for cache in self.component_caches.values():
            await cache.stop_background_cleanup()
        
        self.component_caches.clear()


# Demo and testing
async def demo_advanced_cache():
    """Demonstrate advanced caching capabilities"""
    
    print("💾 PRSM Advanced Caching System Demo")
    print("=" * 60)
    
    cache_manager = CacheManager()
    
    # Test caching for different components
    test_components = [
        "seal_service",
        "distributed_rlt_network", 
        "rlt_quality_monitor"
    ]
    
    print("📊 Testing cache performance...")
    
    for component_id in test_components:
        cache = await cache_manager.get_cache(component_id)
        
        print(f"\n🔧 Testing {component_id}:")
        
        # Test data of various sizes
        test_data = [
            ("small_result", "Simple string result"),
            ("medium_result", {"data": list(range(100)), "metadata": {"size": 100}}),
            ("large_result", {"data": list(range(1000)), "complex": {"nested": {"structure": list(range(50))}}})
        ]
        
        # Test cache operations
        for key, value in test_data:
            # Put operation
            success = await cache.put(key, value)
            print(f"  ✅ Cached {key}: {success}")
            
            # Get operation
            retrieved = await cache.get(key)
            hit = retrieved is not None
            print(f"  🎯 Retrieved {key}: {'HIT' if hit else 'MISS'}")
        
        # Test function result caching
        async def expensive_computation(n: int) -> dict:
            await asyncio.sleep(0.01)  # Simulate computation time
            return {"result": n * n, "computed_at": time.time()}
        
        # Cache function results
        for i in range(5):
            # Check cache first
            cached_result = await cache_manager.get_cached_function_result(
                component_id, "expensive_computation", (i,), {}
            )
            
            if cached_result is None:
                # Compute and cache
                result = await expensive_computation(i)
                await cache_manager.cache_function_result(
                    component_id, "expensive_computation", (i,), {}, result
                )
                print(f"  🔄 Computed and cached expensive_computation({i})")
            else:
                print(f"  ⚡ Used cached expensive_computation({i})")
    
    # Show cache statistics
    print("\n📈 Cache Statistics:")
    all_stats = cache_manager.get_all_statistics()
    
    print(f"Total components: {all_stats['component_count']}")
    print(f"Total cached items: {all_stats['total_items']}")
    print(f"Total cache size: {all_stats['total_size_mb']}MB")
    print(f"Average hit ratio: {all_stats['average_hit_ratio']*100:.1f}%")
    
    print("\n📊 Per-Component Statistics:")
    for component_id, stats in all_stats['component_statistics'].items():
        print(f"  {component_id}:")
        print(f"    Hit ratio: {stats['hit_ratio']*100:.1f}%")
        print(f"    Items: {stats['total_items']}")
        print(f"    Size: {stats['total_size_mb']}MB")
        print(f"    L1 hits: {stats['l1_hits']}")
        print(f"    L2 hits: {stats['l2_hits']}")
        print(f"    L3 hits: {stats['l3_hits']}")
    
    # Cleanup
    await cache_manager.shutdown()
    
    print("\n✅ Advanced caching demo completed!")
    return cache_manager


if __name__ == "__main__":
    asyncio.run(demo_advanced_cache())