"""
Cache Strategies
================

Different cache implementation strategies for various use cases.
"""

import asyncio
import os
import pickle
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, TypeVar
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheStrategy(ABC):
    """Abstract base class for cache strategies"""
    
    def __init__(self, name: str, default_ttl: Optional[int] = None):
        self.name = name
        self.default_ttl = default_ttl
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'size': 0
        }
    
    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values from cache"""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get current cache size in bytes"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / max(total_requests, 1)
        
        return {
            'name': self.name,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'sets': self._stats['sets'],
            'deletes': self._stats['deletes'],
            'hit_rate': hit_rate,
            'size_bytes': self._stats['size']
        }


class MemoryCache(CacheStrategy):
    """In-memory LRU cache with TTL support"""
    
    def __init__(
        self, 
        name: str, 
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
        default_ttl: Optional[int] = None
    ):
        super().__init__(name, default_ttl)
        self.max_size_bytes = max_size_bytes
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get value from memory cache"""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL expiration
            if self._is_expired(entry):
                del self._cache[key]
                self._update_size()
                self._stats['misses'] += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry['last_accessed'] = time.time()
            entry['access_count'] += 1
            
            self._stats['hits'] += 1
            return entry['value']
    
    def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        with self._lock:
            try:
                # Calculate value size
                value_size = self._calculate_size(value)
                
                # Check if single value exceeds max size
                if value_size > self.max_size_bytes:
                    logger.warning(f"Value too large for cache: {value_size} > {self.max_size_bytes}")
                    return False
                
                # Remove existing entry if present
                if key in self._cache:
                    del self._cache[key]
                
                # Evict entries if necessary
                while (self._current_size + value_size > self.max_size_bytes and 
                       len(self._cache) > 0):
                    self._evict_lru()
                
                # Add new entry
                ttl = ttl_seconds or self.default_ttl
                expiry_time = time.time() + ttl if ttl else None
                
                entry = {
                    'value': value,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'access_count': 1,
                    'size': value_size,
                    'expires_at': expiry_time
                }
                
                self._cache[key] = entry
                self._update_size()
                self._stats['sets'] += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Memory cache set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._update_size()
                self._stats['deletes'] += 1
                return True
            return False
    
    def clear(self) -> None:
        """Clear all values from memory cache"""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._stats['size'] = 0
    
    def get_size(self) -> int:
        """Get current cache size in bytes"""
        return self._current_size
    
    def cleanup(self) -> None:
        """Remove expired entries"""
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if self._is_expired(entry, current_time):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self._update_size()
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries from {self.name}")
    
    def _is_expired(self, entry: Dict[str, Any], current_time: Optional[float] = None) -> bool:
        """Check if cache entry is expired"""
        if entry.get('expires_at') is None:
            return False
        
        current_time = current_time or time.time()
        return current_time > entry['expires_at']
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self._cache:
            key, _ = self._cache.popitem(last=False)  # Remove first (oldest) item
            logger.debug(f"Evicted LRU entry: {key[:16]}... from {self.name}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            # Use pickle to get accurate size
            return len(pickle.dumps(value))
        except Exception:
            # Fallback to string representation
            return len(str(value).encode('utf-8'))
    
    def _update_size(self) -> None:
        """Update current cache size"""
        self._current_size = sum(entry['size'] for entry in self._cache.values())
        self._stats['size'] = self._current_size


class FileCache(CacheStrategy):
    """File-based cache for persistent storage"""
    
    def __init__(
        self, 
        name: str,
        cache_dir: Optional[str] = None,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB default
        default_ttl: Optional[int] = None
    ):
        super().__init__(name, default_ttl)
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / f"prsm_cache_{name}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self._lock = threading.RLock()
        self._index_file = self.cache_dir / "cache_index.json"
        self._load_index()
    
    def get(self, key: str) -> Optional[T]:
        """Get value from file cache"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                self._stats['misses'] += 1
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check TTL expiration
                if self._is_expired(data):
                    file_path.unlink()
                    self._remove_from_index(key)
                    self._stats['misses'] += 1
                    return None
                
                # Update access time
                data['last_accessed'] = time.time()
                data['access_count'] += 1
                
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                self._update_index(key, data)
                self._stats['hits'] += 1
                
                return data['value']
                
            except Exception as e:
                logger.error(f"File cache read error: {e}")
                self._stats['misses'] += 1
                return None
    
    def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in file cache"""
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                
                ttl = ttl_seconds or self.default_ttl
                expiry_time = time.time() + ttl if ttl else None
                
                data = {
                    'value': value,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'access_count': 1,
                    'expires_at': expiry_time
                }
                
                # Write to temporary file first, then rename (atomic operation)
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
                
                temp_path.rename(file_path)
                
                # Update index
                self._update_index(key, data)
                
                # Evict old files if necessary
                self._enforce_size_limit()
                
                self._stats['sets'] += 1
                return True
                
            except Exception as e:
                logger.error(f"File cache write error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from file cache"""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                try:
                    file_path.unlink()
                    self._remove_from_index(key)
                    self._stats['deletes'] += 1
                    return True
                except Exception as e:
                    logger.error(f"File cache delete error: {e}")
            
            return False
    
    def clear(self) -> None:
        """Clear all files from cache"""
        with self._lock:
            try:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
                
                self._index = {}
                self._save_index()
                self._stats['size'] = 0
                
            except Exception as e:
                logger.error(f"File cache clear error: {e}")
    
    def get_size(self) -> int:
        """Get current cache size in bytes"""
        try:
            total_size = 0
            for file_path in self.cache_dir.glob("*.cache"):
                total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0
    
    def cleanup(self) -> None:
        """Remove expired files"""
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._index.items():
                if entry.get('expires_at') and current_time > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.delete(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired files from {self.name}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Use hash to avoid filesystem issues with special characters
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _load_index(self) -> None:
        """Load cache index from file"""
        try:
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
            else:
                self._index = {}
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            self._index = {}
    
    def _save_index(self) -> None:
        """Save cache index to file"""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _update_index(self, key: str, data: Dict[str, Any]) -> None:
        """Update cache index entry"""
        self._index[key] = {
            'created_at': data['created_at'],
            'last_accessed': data['last_accessed'],
            'access_count': data['access_count'],
            'expires_at': data.get('expires_at'),
            'size': self._get_file_path(key).stat().st_size if self._get_file_path(key).exists() else 0
        }
        self._save_index()
    
    def _remove_from_index(self, key: str) -> None:
        """Remove entry from cache index"""
        if key in self._index:
            del self._index[key]
            self._save_index()
    
    def _is_expired(self, data: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        if data.get('expires_at') is None:
            return False
        return time.time() > data['expires_at']
    
    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size by removing old files"""
        current_size = self.get_size()
        
        if current_size <= self.max_size_bytes:
            return
        
        # Sort files by last access time (LRU)
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        # Remove oldest files until under size limit
        for key, _ in sorted_entries:
            if current_size <= self.max_size_bytes:
                break
            
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_size = file_path.stat().st_size
                self.delete(key)
                current_size -= file_size


class RedisCache(CacheStrategy):
    """Redis-based cache for distributed caching"""
    
    def __init__(
        self, 
        name: str,
        redis_client=None,
        key_prefix: Optional[str] = None,
        default_ttl: Optional[int] = None
    ):
        super().__init__(name, default_ttl)
        self.redis_client = redis_client
        self.key_prefix = key_prefix or f"prsm:cache:{name}:"
        self._available = self._check_redis_availability()
    
    def _check_redis_availability(self) -> bool:
        """Check if Redis is available"""
        if self.redis_client is None:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis not available for cache {self.name}: {e}")
            return False
    
    def get(self, key: str) -> Optional[T]:
        """Get value from Redis cache"""
        if not self._available:
            self._stats['misses'] += 1
            return None
        
        try:
            redis_key = self.key_prefix + key
            data = self.redis_client.get(redis_key)
            
            if data is None:
                self._stats['misses'] += 1
                return None
            
            value = pickle.loads(data)
            self._stats['hits'] += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self._stats['misses'] += 1
            return None
    
    def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self._available:
            return False
        
        try:
            redis_key = self.key_prefix + key
            data = pickle.dumps(value)
            ttl = ttl_seconds or self.default_ttl
            
            if ttl:
                success = self.redis_client.setex(redis_key, ttl, data)
            else:
                success = self.redis_client.set(redis_key, data)
            
            if success:
                self._stats['sets'] += 1
            
            return bool(success)
            
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self._available:
            return False
        
        try:
            redis_key = self.key_prefix + key
            deleted = self.redis_client.delete(redis_key)
            
            if deleted:
                self._stats['deletes'] += 1
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all values with prefix from Redis cache"""
        if not self._available:
            return
        
        try:
            # Use scan to find all keys with prefix
            for key in self.redis_client.scan_iter(match=self.key_prefix + "*"):
                self.redis_client.delete(key)
            
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
    
    def get_size(self) -> int:
        """Get approximate cache size (not exact for Redis)"""
        if not self._available:
            return 0
        
        try:
            # Count keys with prefix (approximation)
            key_count = 0
            for _ in self.redis_client.scan_iter(match=self.key_prefix + "*"):
                key_count += 1
            
            # Rough estimate: average 1KB per key
            return key_count * 1024
            
        except Exception:
            return 0


class MultiTierCache(CacheStrategy):
    """Multi-tier cache combining memory, Redis, and file caches"""
    
    def __init__(
        self,
        name: str,
        memory_cache: Optional[MemoryCache] = None,
        redis_cache: Optional[RedisCache] = None,
        file_cache: Optional[FileCache] = None,
        default_ttl: Optional[int] = None
    ):
        super().__init__(name, default_ttl)
        
        self.memory_cache = memory_cache or MemoryCache(f"{name}_memory")
        self.redis_cache = redis_cache
        self.file_cache = file_cache
        
        # Order of cache tiers (fastest to slowest)
        self.tiers = [
            ('memory', self.memory_cache),
            ('redis', self.redis_cache),
            ('file', self.file_cache)
        ]
        
        # Filter out None caches
        self.tiers = [(tier_name, cache) for tier_name, cache in self.tiers if cache is not None]
    
    def get(self, key: str) -> Optional[T]:
        """Get value from multi-tier cache"""
        for tier_name, cache in self.tiers:
            try:
                value = cache.get(key)
                if value is not None:
                    # Populate higher tiers with the found value
                    self._populate_higher_tiers(key, value, tier_name)
                    self._stats['hits'] += 1
                    return value
            except Exception as e:
                logger.error(f"Error getting from {tier_name} cache: {e}")
                continue
        
        self._stats['misses'] += 1
        return None
    
    def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in all cache tiers"""
        success = False
        
        for tier_name, cache in self.tiers:
            try:
                if cache.set(key, value, ttl_seconds):
                    success = True
            except Exception as e:
                logger.error(f"Error setting in {tier_name} cache: {e}")
        
        if success:
            self._stats['sets'] += 1
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete value from all cache tiers"""
        deleted = False
        
        for tier_name, cache in self.tiers:
            try:
                if cache.delete(key):
                    deleted = True
            except Exception as e:
                logger.error(f"Error deleting from {tier_name} cache: {e}")
        
        if deleted:
            self._stats['deletes'] += 1
        
        return deleted
    
    def clear(self) -> None:
        """Clear all cache tiers"""
        for tier_name, cache in self.tiers:
            try:
                cache.clear()
            except Exception as e:
                logger.error(f"Error clearing {tier_name} cache: {e}")
    
    def get_size(self) -> int:
        """Get total size across all tiers"""
        total_size = 0
        for tier_name, cache in self.tiers:
            try:
                total_size += cache.get_size()
            except Exception as e:
                logger.error(f"Error getting size from {tier_name} cache: {e}")
        
        return total_size
    
    def _populate_higher_tiers(self, key: str, value: T, found_tier: str) -> None:
        """Populate higher tiers with found value"""
        for tier_name, cache in self.tiers:
            if tier_name == found_tier:
                break
            
            try:
                cache.set(key, value, self.default_ttl)
            except Exception as e:
                logger.error(f"Error populating {tier_name} cache: {e}")