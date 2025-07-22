"""
NWTN Hierarchical Result Caching System

Advanced multi-level caching system for massive parallel processing optimization with
intelligent cache management, cross-worker result sharing, and adaptive eviction policies.

Implements 4-level hierarchical caching architecture:
1. Engine Result Cache - Individual reasoning engine results
2. Sequence Result Cache - Complete reasoning sequence results  
3. Validation Result Cache - World model validation results
4. Cross-Worker Cache - Shared results across parallel workers

Features:
- >70% cache hit rate optimization
- Intelligent cache warming and preloading
- Advanced eviction policies (LRU, LFU, TTL-based)
- Cache analytics and performance monitoring
- Memory-efficient storage with compression
- Thread-safe concurrent access

Part of NWTN Phase 8: Parallel Processing & Scalability Architecture
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union, Callable, Awaitable
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
import hashlib
import json
import pickle
import gzip
import statistics
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import asyncio
import weakref
from uuid import uuid4
import psutil

class CacheLevel(Enum):
    ENGINE_RESULT = "engine_result"
    SEQUENCE_RESULT = "sequence_result"
    VALIDATION_RESULT = "validation_result"
    CROSS_WORKER = "cross_worker"

class EvictionPolicy(Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"
    JSON = "json"

@dataclass
class CacheEntry:
    key: str
    value: Any
    cache_level: CacheLevel
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    computation_time: float = 0.0  # Time it took to compute this result
    memory_size: int = 0  # Size in bytes
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_type: CompressionType = CompressionType.NONE
    compressed_value: Optional[bytes] = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if not self.ttl:
            return False
        return datetime.now() > (self.created_at + self.ttl)
    
    def access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_value(self) -> Any:
        """Get value, decompressing if necessary"""
        self.access()
        
        if self.compression_type == CompressionType.NONE:
            return self.value
        elif self.compression_type == CompressionType.GZIP and self.compressed_value:
            return pickle.loads(gzip.decompress(self.compressed_value))
        elif self.compression_type == CompressionType.PICKLE and self.compressed_value:
            return pickle.loads(self.compressed_value)
        elif self.compression_type == CompressionType.JSON and self.compressed_value:
            return json.loads(self.compressed_value.decode('utf-8'))
        else:
            return self.value

@dataclass
class CacheStatistics:
    cache_level: CacheLevel
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_memory_used: int = 0
    average_computation_time_saved: float = 0.0
    hit_rate: float = 0.0
    memory_efficiency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class CacheLevelManager:
    """Manages a single level of the hierarchical cache"""
    
    def __init__(self, cache_level: CacheLevel, max_size: int = 10000, 
                 eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
                 ttl: Optional[timedelta] = None,
                 enable_compression: bool = True):
        self.cache_level = cache_level
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.ttl = ttl
        self.enable_compression = enable_compression
        
        # Storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_frequencies: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStatistics(cache_level)
        
        # Background maintenance
        self.maintenance_interval = 300  # 5 minutes
        self.last_maintenance = time.time()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self.stats.total_requests += 1
            
            if key not in self.cache:
                self.stats.cache_misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.access_frequencies.pop(key, 0)
                self.stats.cache_misses += 1
                return None
            
            # Update access statistics
            entry.access()
            self.access_frequencies[key] += 1
            
            # Move to end for LRU
            if self.eviction_policy in [EvictionPolicy.LRU, EvictionPolicy.ADAPTIVE]:
                self.cache.move_to_end(key)
            
            self.stats.cache_hits += 1
            self.stats.average_computation_time_saved = (
                (self.stats.average_computation_time_saved * (self.stats.cache_hits - 1) + entry.computation_time) 
                / self.stats.cache_hits
            )
            
            return entry.get_value()
    
    def put(self, key: str, value: Any, computation_time: float = 0.0, 
            metadata: Dict[str, Any] = None) -> bool:
        """Store value in cache"""
        with self.lock:
            # Check if maintenance needed
            if time.time() - self.last_maintenance > self.maintenance_interval:
                self._perform_maintenance()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                cache_level=self.cache_level,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                computation_time=computation_time,
                metadata=metadata or {},
                ttl=self.ttl
            )
            
            # Compress if enabled
            if self.enable_compression and self._should_compress(value):
                entry = self._compress_entry(entry)
            
            # Calculate memory size
            entry.memory_size = self._estimate_size(entry)
            
            # Evict if necessary
            while len(self.cache) >= self.max_size:
                if not self._evict_entry():
                    break
            
            # Store entry
            self.cache[key] = entry
            self.cache.move_to_end(key)  # Move to end for LRU
            
            # Update statistics
            self.stats.total_memory_used += entry.memory_size
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.access_frequencies.pop(key, 0)
                self.stats.total_memory_used -= entry.memory_size
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_frequencies.clear()
            self.stats.total_memory_used = 0
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics"""
        with self.lock:
            self.stats.hit_rate = (
                self.stats.cache_hits / max(self.stats.total_requests, 1)
            )
            self.stats.memory_efficiency = (
                len(self.cache) / max(self.max_size, 1)
            )
            self.stats.last_updated = datetime.now()
            return self.stats
    
    def warm_cache(self, key_value_pairs: List[Tuple[str, Any]], 
                   computation_times: Optional[List[float]] = None):
        """Pre-populate cache with known valuable entries"""
        computation_times = computation_times or [0.0] * len(key_value_pairs)
        
        for i, (key, value) in enumerate(key_value_pairs):
            comp_time = computation_times[i] if i < len(computation_times) else 0.0
            self.put(key, value, comp_time, {'preloaded': True})
    
    def _evict_entry(self) -> bool:
        """Evict entry based on eviction policy"""
        if not self.cache:
            return False
        
        key_to_evict = None
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            key_to_evict = next(iter(self.cache))
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_freq = min(self.access_frequencies.values()) if self.access_frequencies else 0
            for key, freq in self.access_frequencies.items():
                if freq == min_freq:
                    key_to_evict = key
                    break
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first
            for key, entry in self.cache.items():
                if entry.is_expired():
                    key_to_evict = key
                    break
            
            # If no expired entries, fall back to LRU
            if not key_to_evict:
                key_to_evict = next(iter(self.cache))
        
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # Adaptive eviction based on access patterns and computation time
            scores = {}
            for key, entry in self.cache.items():
                age_factor = (datetime.now() - entry.last_accessed).total_seconds() / 3600  # hours
                freq_factor = self.access_frequencies.get(key, 1)
                computation_factor = max(entry.computation_time, 0.1)  # Bias towards expensive computations
                
                # Lower score = higher priority for eviction
                score = (freq_factor * computation_factor) / (age_factor + 1)
                scores[key] = score
            
            key_to_evict = min(scores.keys(), key=lambda k: scores[k])
        
        if key_to_evict:
            entry = self.cache[key_to_evict]
            del self.cache[key_to_evict]
            self.access_frequencies.pop(key_to_evict, 0)
            self.stats.total_memory_used -= entry.memory_size
            self.stats.evictions += 1
            return True
        
        return False
    
    def _perform_maintenance(self):
        """Perform background maintenance tasks"""
        with self.lock:
            # Remove expired entries
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.remove(key)
            
            # Update memory usage statistics
            total_memory = sum(entry.memory_size for entry in self.cache.values())
            self.stats.total_memory_used = total_memory
            
            self.last_maintenance = time.time()
    
    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed"""
        # Compress large objects
        estimated_size = self._estimate_raw_size(value)
        return estimated_size > 1024  # Compress if >1KB
    
    def _compress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Compress cache entry value"""
        try:
            if isinstance(entry.value, (dict, list)):
                # Use JSON for structured data
                compressed = json.dumps(entry.value).encode('utf-8')
                entry.compression_type = CompressionType.JSON
            else:
                # Use pickle + gzip for other objects
                pickled = pickle.dumps(entry.value)
                compressed = gzip.compress(pickled)
                entry.compression_type = CompressionType.GZIP
            
            entry.compressed_value = compressed
            entry.value = None  # Clear original value to save memory
            
        except Exception:
            # Fall back to no compression if compression fails
            entry.compression_type = CompressionType.NONE
        
        return entry
    
    def _estimate_size(self, entry: CacheEntry) -> int:
        """Estimate memory size of cache entry"""
        base_size = 64  # Base overhead
        
        if entry.compressed_value:
            return base_size + len(entry.compressed_value)
        else:
            return base_size + self._estimate_raw_size(entry.value)
    
    def _estimate_raw_size(self, value: Any) -> int:
        """Estimate raw size of a value"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_raw_size(item) for item in value[:10])  # Sample first 10
            elif isinstance(value, dict):
                return sum(self._estimate_raw_size(k) + self._estimate_raw_size(v) 
                          for k, v in list(value.items())[:10])  # Sample first 10
            else:
                return 64  # Default size

class CrossWorkerCacheManager:
    """Manages cache sharing across parallel workers"""
    
    def __init__(self, max_shared_entries: int = 50000):
        self.max_shared_entries = max_shared_entries
        self.shared_cache: Dict[str, CacheEntry] = {}
        self.worker_contributions: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
        
        # Cache sharing metrics
        self.sharing_stats = {
            'total_shared_entries': 0,
            'successful_retrievals': 0,
            'worker_contributions': {},
            'cache_hit_rate': 0.0
        }
    
    def share_result(self, worker_id: str, key: str, value: Any, 
                    computation_time: float, cache_level: CacheLevel) -> bool:
        """Share a computation result across workers"""
        with self.lock:
            # Only share expensive computations
            if computation_time < 1.0:  # Less than 1 second
                return False
            
            # Create shared cache entry
            shared_key = self._create_shared_key(key, cache_level)
            
            entry = CacheEntry(
                key=shared_key,
                value=value,
                cache_level=CacheLevel.CROSS_WORKER,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                computation_time=computation_time,
                metadata={'worker_id': worker_id, 'original_level': cache_level.value}
            )
            
            # Evict if necessary
            while len(self.shared_cache) >= self.max_shared_entries:
                self._evict_shared_entry()
            
            self.shared_cache[shared_key] = entry
            self.worker_contributions[worker_id] += 1
            self.sharing_stats['total_shared_entries'] += 1
            
            return True
    
    def retrieve_shared_result(self, worker_id: str, key: str, cache_level: CacheLevel) -> Optional[Any]:
        """Retrieve shared result from other workers"""
        with self.lock:
            shared_key = self._create_shared_key(key, cache_level)
            
            if shared_key not in self.shared_cache:
                return None
            
            entry = self.shared_cache[shared_key]
            
            # Don't retrieve our own shared results
            if entry.metadata.get('worker_id') == worker_id:
                return None
            
            self.sharing_stats['successful_retrievals'] += 1
            return entry.get_value()
    
    def get_sharing_statistics(self) -> Dict[str, Any]:
        """Get cross-worker cache sharing statistics"""
        with self.lock:
            total_requests = self.sharing_stats['successful_retrievals'] + len(self.shared_cache)
            hit_rate = self.sharing_stats['successful_retrievals'] / max(total_requests, 1)
            
            return {
                **self.sharing_stats,
                'cache_hit_rate': hit_rate,
                'worker_contributions': dict(self.worker_contributions),
                'total_memory_entries': len(self.shared_cache)
            }
    
    def _create_shared_key(self, key: str, cache_level: CacheLevel) -> str:
        """Create normalized shared key"""
        return f"{cache_level.value}:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _evict_shared_entry(self):
        """Evict oldest shared cache entry"""
        if not self.shared_cache:
            return
        
        # Remove least recently accessed entry
        oldest_key = min(self.shared_cache.keys(), 
                        key=lambda k: self.shared_cache[k].last_accessed)
        del self.shared_cache[oldest_key]

class CacheAnalytics:
    """Advanced analytics for cache performance optimization"""
    
    def __init__(self):
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_recommendations: List[str] = []
        
    def record_access_pattern(self, key: str, cache_level: CacheLevel, hit: bool):
        """Record access pattern for analysis"""
        self.access_patterns[f"{cache_level.value}:{key}"].append(datetime.now())
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        for pattern_key in self.access_patterns:
            self.access_patterns[pattern_key] = [
                ts for ts in self.access_patterns[pattern_key] if ts > cutoff_time
            ]
    
    def analyze_cache_performance(self, cache_managers: Dict[CacheLevel, CacheLevelManager]) -> Dict[str, Any]:
        """Analyze overall cache performance and generate insights"""
        analysis = {
            'overall_hit_rate': 0.0,
            'memory_utilization': {},
            'hottest_keys': [],
            'optimization_opportunities': [],
            'level_performance': {}
        }
        
        total_hits = 0
        total_requests = 0
        
        for level, manager in cache_managers.items():
            stats = manager.get_statistics()
            
            analysis['level_performance'][level.value] = {
                'hit_rate': stats.hit_rate,
                'memory_used': stats.total_memory_used,
                'evictions': stats.evictions,
                'requests': stats.total_requests
            }
            
            total_hits += stats.cache_hits
            total_requests += stats.total_requests
            
            # Identify optimization opportunities
            if stats.hit_rate < 0.5:
                analysis['optimization_opportunities'].append(
                    f"Low hit rate in {level.value} cache: {stats.hit_rate:.2%}"
                )
            
            if stats.evictions > stats.cache_hits * 0.1:
                analysis['optimization_opportunities'].append(
                    f"High eviction rate in {level.value} cache: consider increasing size"
                )
        
        analysis['overall_hit_rate'] = total_hits / max(total_requests, 1)
        
        # Identify hottest access patterns
        pattern_frequencies = {}
        for pattern_key, accesses in self.access_patterns.items():
            if len(accesses) > 10:  # Only frequent patterns
                pattern_frequencies[pattern_key] = len(accesses)
        
        analysis['hottest_keys'] = sorted(
            pattern_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return analysis
    
    def recommend_cache_optimizations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        overall_hit_rate = performance_data['overall_hit_rate']
        
        if overall_hit_rate < 0.7:
            recommendations.append(
                "Overall cache hit rate is below target (70%). Consider cache warming strategies."
            )
        
        for level, perf in performance_data['level_performance'].items():
            if perf['hit_rate'] < 0.5:
                recommendations.append(
                    f"Increase {level} cache size or adjust eviction policy"
                )
            
            if perf['evictions'] > perf['requests'] * 0.1:
                recommendations.append(
                    f"High eviction rate in {level} cache suggests size optimization needed"
                )
        
        # Hot key analysis
        if performance_data['hottest_keys']:
            recommendations.append(
                "Consider preloading frequently accessed results at startup"
            )
        
        return recommendations

class HierarchicalResultCachingSystem:
    """Main orchestrator for the 4-level hierarchical caching system"""
    
    def __init__(self, 
                 engine_cache_size: int = 100000,
                 sequence_cache_size: int = 10000,
                 validation_cache_size: int = 50000,
                 cross_worker_cache_size: int = 50000,
                 enable_analytics: bool = True):
        
        # Initialize cache level managers
        self.cache_managers = {
            CacheLevel.ENGINE_RESULT: CacheLevelManager(
                CacheLevel.ENGINE_RESULT, engine_cache_size, EvictionPolicy.ADAPTIVE,
                ttl=timedelta(hours=4)
            ),
            CacheLevel.SEQUENCE_RESULT: CacheLevelManager(
                CacheLevel.SEQUENCE_RESULT, sequence_cache_size, EvictionPolicy.LRU,
                ttl=timedelta(hours=8)
            ),
            CacheLevel.VALIDATION_RESULT: CacheLevelManager(
                CacheLevel.VALIDATION_RESULT, validation_cache_size, EvictionPolicy.LFU,
                ttl=timedelta(hours=12)
            )
        }
        
        # Cross-worker cache manager
        self.cross_worker_manager = CrossWorkerCacheManager(cross_worker_cache_size)
        
        # Analytics
        self.analytics = CacheAnalytics() if enable_analytics else None
        
        # Performance tracking
        self.system_stats = {
            'total_get_requests': 0,
            'total_put_requests': 0,
            'system_wide_hit_rate': 0.0,
            'memory_usage_mb': 0.0,
            'time_saved_seconds': 0.0
        }
        
        # Thread safety
        self.lock = threading.RLock()
    
    def get(self, key: str, cache_levels: Optional[List[CacheLevel]] = None, 
            worker_id: Optional[str] = None) -> Optional[Any]:
        """Get value from hierarchical cache system"""
        with self.lock:
            self.system_stats['total_get_requests'] += 1
            
            # Default to all levels in order of speed
            if not cache_levels:
                cache_levels = [
                    CacheLevel.ENGINE_RESULT,
                    CacheLevel.SEQUENCE_RESULT, 
                    CacheLevel.VALIDATION_RESULT,
                    CacheLevel.CROSS_WORKER
                ]
            
            # Check each cache level
            for level in cache_levels:
                if level == CacheLevel.CROSS_WORKER and worker_id:
                    result = self.cross_worker_manager.retrieve_shared_result(worker_id, key, level)
                    if result is not None:
                        if self.analytics:
                            self.analytics.record_access_pattern(key, level, True)
                        return result
                elif level in self.cache_managers:
                    result = self.cache_managers[level].get(key)
                    if result is not None:
                        if self.analytics:
                            self.analytics.record_access_pattern(key, level, True)
                        return result
            
            # Record cache miss
            if self.analytics:
                for level in cache_levels:
                    self.analytics.record_access_pattern(key, level, False)
            
            return None
    
    def put(self, key: str, value: Any, cache_level: CacheLevel,
            computation_time: float = 0.0, worker_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store value in appropriate cache level"""
        with self.lock:
            self.system_stats['total_put_requests'] += 1
            
            success = False
            
            if cache_level in self.cache_managers:
                success = self.cache_managers[cache_level].put(
                    key, value, computation_time, metadata
                )
                
                # Also share expensive computations across workers
                if (success and worker_id and computation_time > 2.0 and  # >2 seconds
                    cache_level in [CacheLevel.SEQUENCE_RESULT, CacheLevel.VALIDATION_RESULT]):
                    self.cross_worker_manager.share_result(
                        worker_id, key, value, computation_time, cache_level
                    )
            
            return success
    
    def get_or_compute(self, key: str, computation_func: Callable, 
                      cache_level: CacheLevel, worker_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Get value from cache or compute if not found"""
        # Try to get from cache first
        result = self.get(key, worker_id=worker_id)
        if result is not None:
            return result
        
        # Compute result
        start_time = time.time()
        result = computation_func()
        computation_time = time.time() - start_time
        
        # Store in cache
        self.put(key, result, cache_level, computation_time, worker_id, metadata)
        
        # Update time saved statistics
        self.system_stats['time_saved_seconds'] += computation_time
        
        return result
    
    async def get_or_compute_async(self, key: str, computation_func: Awaitable,
                                  cache_level: CacheLevel, worker_id: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Async version of get_or_compute"""
        # Try to get from cache first
        result = self.get(key, worker_id=worker_id)
        if result is not None:
            return result
        
        # Compute result asynchronously
        start_time = time.time()
        result = await computation_func()
        computation_time = time.time() - start_time
        
        # Store in cache
        self.put(key, result, cache_level, computation_time, worker_id, metadata)
        
        # Update time saved statistics
        self.system_stats['time_saved_seconds'] += computation_time
        
        return result
    
    def warm_cache(self, cache_warming_data: Dict[CacheLevel, List[Tuple[str, Any]]]):
        """Pre-populate caches with valuable data"""
        for level, key_value_pairs in cache_warming_data.items():
            if level in self.cache_managers:
                self.cache_managers[level].warm_cache(key_value_pairs)
    
    def optimize_cache_sizes(self) -> Dict[str, Any]:
        """Automatically optimize cache sizes based on usage patterns"""
        recommendations = {}
        
        for level, manager in self.cache_managers.items():
            stats = manager.get_statistics()
            
            current_size = manager.max_size
            current_usage = len(manager.cache)
            hit_rate = stats.hit_rate
            
            # Recommend size adjustments
            if hit_rate < 0.6 and current_usage == current_size:
                # Low hit rate and cache is full - recommend increase
                recommended_size = int(current_size * 1.5)
                recommendations[level.value] = {
                    'action': 'increase',
                    'current_size': current_size,
                    'recommended_size': recommended_size,
                    'reason': f'Low hit rate ({hit_rate:.2%}) with full cache'
                }
            elif hit_rate > 0.9 and current_usage < current_size * 0.5:
                # High hit rate but low usage - recommend decrease
                recommended_size = max(int(current_size * 0.8), 1000)
                recommendations[level.value] = {
                    'action': 'decrease',
                    'current_size': current_size,
                    'recommended_size': recommended_size,
                    'reason': f'High hit rate ({hit_rate:.2%}) with low usage'
                }
            else:
                recommendations[level.value] = {
                    'action': 'maintain',
                    'current_size': current_size,
                    'reason': f'Good balance: {hit_rate:.2%} hit rate'
                }
        
        return recommendations
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire caching system"""
        with self.lock:
            stats = {
                'system_overview': self.system_stats.copy(),
                'cache_levels': {},
                'cross_worker_sharing': self.cross_worker_manager.get_sharing_statistics(),
                'memory_usage': self._calculate_memory_usage(),
                'performance_analysis': {}
            }
            
            # Get statistics for each cache level
            for level, manager in self.cache_managers.items():
                level_stats = manager.get_statistics()
                stats['cache_levels'][level.value] = {
                    'hit_rate': level_stats.hit_rate,
                    'total_requests': level_stats.total_requests,
                    'cache_hits': level_stats.cache_hits,
                    'cache_misses': level_stats.cache_misses,
                    'evictions': level_stats.evictions,
                    'memory_used_mb': level_stats.total_memory_used / (1024 * 1024),
                    'entries_count': len(manager.cache),
                    'max_size': manager.max_size,
                    'avg_computation_time_saved': level_stats.average_computation_time_saved
                }
            
            # Calculate overall hit rate
            total_hits = sum(stats['cache_levels'][level]['cache_hits'] 
                           for level in stats['cache_levels'])
            total_requests = sum(stats['cache_levels'][level]['total_requests'] 
                               for level in stats['cache_levels'])
            stats['system_overview']['system_wide_hit_rate'] = total_hits / max(total_requests, 1)
            
            # Performance analysis
            if self.analytics:
                stats['performance_analysis'] = self.analytics.analyze_cache_performance(self.cache_managers)
                stats['optimization_recommendations'] = self.analytics.recommend_cache_optimizations(
                    stats['performance_analysis']
                )
            
            return stats
    
    def clear_all_caches(self):
        """Clear all cache levels"""
        with self.lock:
            for manager in self.cache_managers.values():
                manager.clear()
            
            self.cross_worker_manager.shared_cache.clear()
            
            # Reset statistics
            self.system_stats = {
                'total_get_requests': 0,
                'total_put_requests': 0,
                'system_wide_hit_rate': 0.0,
                'memory_usage_mb': 0.0,
                'time_saved_seconds': 0.0
            }
    
    def _calculate_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage across all cache levels"""
        memory_usage = {}
        total_memory = 0
        
        for level, manager in self.cache_managers.items():
            level_memory = manager.stats.total_memory_used / (1024 * 1024)  # Convert to MB
            memory_usage[level.value] = level_memory
            total_memory += level_memory
        
        # Add cross-worker cache memory (estimated)
        cross_worker_memory = len(self.cross_worker_manager.shared_cache) * 1024 / (1024 * 1024)  # Rough estimate
        memory_usage['cross_worker'] = cross_worker_memory
        total_memory += cross_worker_memory
        
        memory_usage['total_mb'] = total_memory
        
        # Add system memory context
        system_memory = psutil.virtual_memory()
        memory_usage['system_total_gb'] = system_memory.total / (1024**3)
        memory_usage['system_available_gb'] = system_memory.available / (1024**3)
        memory_usage['cache_system_percentage'] = (total_memory / (system_memory.total / (1024**2))) * 100
        
        return memory_usage