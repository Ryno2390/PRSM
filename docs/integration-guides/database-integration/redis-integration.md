# Redis Integration Guide

Integrate PRSM with Redis for high-performance caching, session management, and real-time data storage.

## 🎯 Overview

This guide covers integrating PRSM with Redis, including setup, caching strategies, session management, and production best practices for optimal performance.

## 📋 Prerequisites

- Redis 6.0+ installed
- PRSM instance configured
- Basic knowledge of Redis and caching concepts
- Python development environment

## 🚀 Quick Start

### 1. Redis Setup

```bash
# Install Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server

# Install Redis (macOS with Homebrew)
brew install redis
brew services start redis

# Install Redis (CentOS/RHEL)
sudo yum install redis
sudo systemctl start redis
sudo systemctl enable redis

# Install Redis (Docker)
docker run --name redis -d -p 6379:6379 redis:7-alpine

# Test Redis connection
redis-cli ping
# Should return: PONG
```

### 2. Redis Configuration

```redis
# /etc/redis/redis.conf (Production settings)

# Network settings
bind 127.0.0.1 ::1
port 6379
timeout 300
tcp-keepalive 300

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence
save 900 1
save 300 10
save 60 10000
dir /var/lib/redis
dbfilename dump.rdb

# Security
requirepass your_secure_password_here
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Performance
tcp-backlog 511
databases 16
```

### 3. Basic Connection Test

```python
# test_connection.py
import redis
import asyncio
import aioredis

def test_sync_connection():
    """Test synchronous Redis connection."""
    try:
        r = redis.Redis(
            host='localhost',
            port=6379,
            password='your_secure_password_here',
            decode_responses=True
        )
        
        # Test connection
        r.ping()
        print("✅ Synchronous Redis connection successful!")
        
        # Test basic operations
        r.set('test_key', 'Hello Redis!')
        value = r.get('test_key')
        print(f"Test value: {value}")
        
        r.delete('test_key')
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def test_async_connection():
    """Test asynchronous Redis connection."""
    try:
        redis_client = aioredis.from_url(
            "redis://localhost:6379",
            password="your_secure_password_here",
            decode_responses=True
        )
        
        # Test connection
        await redis_client.ping()
        print("✅ Asynchronous Redis connection successful!")
        
        # Test basic operations
        await redis_client.set('test_async_key', 'Hello Async Redis!')
        value = await redis_client.get('test_async_key')
        print(f"Test async value: {value}")
        
        await redis_client.delete('test_async_key')
        await redis_client.close()
        return True
        
    except Exception as e:
        print(f"❌ Async connection failed: {e}")
        return False

if __name__ == "__main__":
    test_sync_connection()
    asyncio.run(test_async_connection())
```

## 🔧 PRSM Redis Integration

### Cache Configuration

```python
# prsm/core/cache.py
import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Union, Dict
from datetime import timedelta
import aioredis
import redis
from pydantic import BaseSettings

logger = logging.getLogger(__name__)

class CacheSettings(BaseSettings):
    """Redis cache configuration."""
    
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_max_connections: int = 20
    redis_retry_on_timeout: bool = True
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    query_cache_ttl: int = 1800  # 30 minutes
    session_ttl: int = 86400  # 24 hours
    rate_limit_ttl: int = 60  # 1 minute
    
    class Config:
        env_prefix = "PRSM_CACHE_"

class RedisCache:
    """Redis cache manager for PRSM."""
    
    def __init__(self, settings: CacheSettings):
        self.settings = settings
        self._redis_pool = None
        self._async_redis = None
        
    async def initialize(self):
        """Initialize Redis connections."""
        try:
            # Async Redis connection
            self._async_redis = aioredis.from_url(
                self.settings.redis_url,
                password=self.settings.redis_password,
                db=self.settings.redis_db,
                max_connections=self.settings.redis_max_connections,
                retry_on_timeout=self.settings.redis_retry_on_timeout,
                decode_responses=True
            )
            
            # Test connection
            await self._async_redis.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def close(self):
        """Close Redis connections."""
        if self._async_redis:
            await self._async_redis.close()
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = f"{prefix}:{':'.join(map(str, args))}"
        if kwargs:
            key_data += f":{json.dumps(kwargs, sort_keys=True)}"
        
        # Hash long keys
        if len(key_data) > 200:
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"
        
        return key_data
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self._async_redis.get(key)
            if value is None:
                return None
            
            # Try JSON first, then pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value.encode('latin1'))
                
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.settings.default_ttl
            
            # Serialize value
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                serialized = pickle.dumps(value).decode('latin1')
            
            await self._async_redis.setex(key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            result = await self._async_redis.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self._async_redis.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter."""
        try:
            return await self._async_redis.incrby(key, amount)
        except Exception as e:
            logger.warning(f"Cache increment failed for key {key}: {e}")
            return None
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        try:
            return await self._async_redis.expire(key, ttl)
        except Exception as e:
            logger.warning(f"Cache expire failed for key {key}: {e}")
            return False

# Global cache instance
cache_settings = CacheSettings()
redis_cache = RedisCache(cache_settings)
```

### Query Result Caching

```python
# prsm/core/query_cache.py
import hashlib
import json
from typing import Dict, Any, Optional
from prsm.core.cache import redis_cache
from prsm.models.query import QueryRequest, QueryResponse

class QueryCacheManager:
    """Manage caching of PRSM query results."""
    
    def __init__(self):
        self.cache = redis_cache
        self.cache_prefix = "prsm:query"
    
    def _generate_query_key(self, request: QueryRequest) -> str:
        """Generate cache key for query request."""
        # Create deterministic hash of query parameters
        cache_data = {
            "prompt": request.prompt,
            "user_id": request.user_id,
            "context": request.context,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "model": request.model
        }
        
        query_str = json.dumps(cache_data, sort_keys=True)
        query_hash = hashlib.sha256(query_str.encode()).hexdigest()
        
        return f"{self.cache_prefix}:{query_hash}"
    
    async def get_cached_response(
        self, 
        request: QueryRequest
    ) -> Optional[QueryResponse]:
        """Get cached query response."""
        if not request.use_cache:
            return None
        
        cache_key = self._generate_query_key(request)
        cached_data = await self.cache.get(cache_key)
        
        if cached_data:
            try:
                return QueryResponse(**cached_data)
            except Exception as e:
                # Invalid cached data, remove it
                await self.cache.delete(cache_key)
                return None
        
        return None
    
    async def cache_response(
        self, 
        request: QueryRequest, 
        response: QueryResponse
    ) -> bool:
        """Cache query response."""
        if not request.use_cache or not response.cache_response:
            return False
        
        cache_key = self._generate_query_key(request)
        cache_data = response.dict()
        
        # Add cache metadata
        cache_data["cached_at"] = response.timestamp
        cache_data["cache_key"] = cache_key
        
        return await self.cache.set(
            cache_key, 
            cache_data, 
            ttl=self.cache.settings.query_cache_ttl
        )
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cached queries for a user."""
        pattern = f"{self.cache_prefix}:*"
        invalidated = 0
        
        try:
            # Get all query cache keys
            keys = await self.cache._async_redis.keys(pattern)
            
            for key in keys:
                cached_data = await self.cache.get(key)
                if cached_data and cached_data.get("user_id") == user_id:
                    await self.cache.delete(key)
                    invalidated += 1
            
        except Exception as e:
            logger.warning(f"Failed to invalidate user cache: {e}")
        
        return invalidated
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.cache._async_redis.info()
            pattern = f"{self.cache_prefix}:*"
            query_keys = await self.cache._async_redis.keys(pattern)
            
            return {
                "total_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_queries_cached": len(query_keys),
                "hit_rate": await self._calculate_hit_rate(),
                "cache_size": info.get("used_memory", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    async def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        try:
            hits = await self.cache.get("cache:hits") or 0
            misses = await self.cache.get("cache:misses") or 0
            total = hits + misses
            
            return (hits / total * 100) if total > 0 else 0.0
        except:
            return 0.0

# Global query cache manager
query_cache = QueryCacheManager()
```

### Session Management

```python
# prsm/core/session.py
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from prsm.core.cache import redis_cache

class SessionManager:
    """Redis-based session management for PRSM."""
    
    def __init__(self):
        self.cache = redis_cache
        self.session_prefix = "prsm:session"
        self.user_sessions_prefix = "prsm:user_sessions"
    
    async def create_session(
        self, 
        user_id: str, 
        session_data: Dict[str, Any] = None
    ) -> str:
        """Create new user session."""
        session_id = str(uuid.uuid4())
        session_key = f"{self.session_prefix}:{session_id}"
        user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
        
        # Session data
        session_info = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "data": session_data or {}
        }
        
        # Store session
        await self.cache.set(
            session_key, 
            session_info, 
            ttl=self.cache.settings.session_ttl
        )
        
        # Add to user's session list
        await self.cache._async_redis.sadd(user_sessions_key, session_id)
        await self.cache.expire(user_sessions_key, self.cache.settings.session_ttl)
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        session_key = f"{self.session_prefix}:{session_id}"
        session_data = await self.cache.get(session_key)
        
        if session_data:
            # Update last activity
            session_data["last_activity"] = datetime.utcnow().isoformat()
            await self.cache.set(
                session_key, 
                session_data, 
                ttl=self.cache.settings.session_ttl
            )
        
        return session_data
    
    async def update_session(
        self, 
        session_id: str, 
        data: Dict[str, Any]
    ) -> bool:
        """Update session data."""
        session_key = f"{self.session_prefix}:{session_id}"
        session_data = await self.cache.get(session_key)
        
        if not session_data:
            return False
        
        # Update data
        session_data["data"].update(data)
        session_data["last_activity"] = datetime.utcnow().isoformat()
        
        return await self.cache.set(
            session_key, 
            session_data, 
            ttl=self.cache.settings.session_ttl
        )
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        session_key = f"{self.session_prefix}:{session_id}"
        session_data = await self.cache.get(session_key)
        
        if session_data:
            user_id = session_data.get("user_id")
            if user_id:
                user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
                await self.cache._async_redis.srem(user_sessions_key, session_id)
        
        return await self.cache.delete(session_key)
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
        session_ids = await self.cache._async_redis.smembers(user_sessions_key)
        
        sessions = []
        for session_id in session_ids:
            session_data = await self.get_session(session_id)
            if session_data:
                sessions.append(session_data)
        
        return sessions
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        pattern = f"{self.session_prefix}:*"
        expired_count = 0
        
        try:
            keys = await self.cache._async_redis.keys(pattern)
            
            for key in keys:
                if not await self.cache.exists(key):
                    expired_count += 1
                    
        except Exception as e:
            logger.warning(f"Session cleanup failed: {e}")
        
        return expired_count

# Global session manager
session_manager = SessionManager()
```

### Rate Limiting

```python
# prsm/core/rate_limiter.py
import time
from typing import Optional, Dict, Any
from prsm.core.cache import redis_cache

class RateLimiter:
    """Redis-based rate limiter for PRSM API."""
    
    def __init__(self):
        self.cache = redis_cache
        self.prefix = "prsm:rate_limit"
    
    async def is_allowed(
        self, 
        identifier: str, 
        limit: int, 
        window: int,
        burst: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (user_id, IP, etc.)
            limit: Number of requests allowed per window
            window: Time window in seconds
            burst: Optional burst limit
        
        Returns:
            Dict with allowed status and metadata
        """
        current_time = int(time.time())
        window_start = current_time - (current_time % window)
        
        # Keys for sliding window
        window_key = f"{self.prefix}:{identifier}:{window_start}"
        burst_key = f"{self.prefix}:burst:{identifier}"
        
        try:
            # Get current count
            current_count = await self.cache._async_redis.get(window_key) or 0
            current_count = int(current_count)
            
            # Check burst limit if specified
            if burst:
                burst_count = await self.cache._async_redis.get(burst_key) or 0
                burst_count = int(burst_count)
                
                if burst_count >= burst:
                    return {
                        "allowed": False,
                        "reason": "burst_limit_exceeded",
                        "current_count": current_count,
                        "limit": limit,
                        "burst_count": burst_count,
                        "burst_limit": burst,
                        "reset_time": window_start + window
                    }
            
            # Check window limit
            if current_count >= limit:
                return {
                    "allowed": False,
                    "reason": "rate_limit_exceeded",
                    "current_count": current_count,
                    "limit": limit,
                    "reset_time": window_start + window
                }
            
            # Increment counters
            pipe = self.cache._async_redis.pipeline()
            pipe.incr(window_key)
            pipe.expire(window_key, window)
            
            if burst:
                pipe.incr(burst_key)
                pipe.expire(burst_key, 60)  # Reset burst every minute
            
            await pipe.execute()
            
            return {
                "allowed": True,
                "current_count": current_count + 1,
                "limit": limit,
                "remaining": limit - current_count - 1,
                "reset_time": window_start + window
            }
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # Allow request on error to prevent blocking
            return {"allowed": True, "error": str(e)}
    
    async def reset_limit(self, identifier: str, window: int) -> bool:
        """Reset rate limit for identifier."""
        try:
            current_time = int(time.time())
            window_start = current_time - (current_time % window)
            window_key = f"{self.prefix}:{identifier}:{window_start}"
            burst_key = f"{self.prefix}:burst:{identifier}"
            
            await self.cache.delete(window_key)
            await self.cache.delete(burst_key)
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False
    
    async def get_limit_status(
        self, 
        identifier: str, 
        window: int
    ) -> Dict[str, Any]:
        """Get current rate limit status."""
        try:
            current_time = int(time.time())
            window_start = current_time - (current_time % window)
            window_key = f"{self.prefix}:{identifier}:{window_start}"
            
            current_count = await self.cache._async_redis.get(window_key) or 0
            current_count = int(current_count)
            
            return {
                "current_count": current_count,
                "window_start": window_start,
                "window_end": window_start + window,
                "time_remaining": (window_start + window) - current_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get limit status: {e}")
            return {}

# Global rate limiter
rate_limiter = RateLimiter()
```

## 🏗️ Production Configuration

### Redis Cluster Setup

```yaml
# docker-compose.redis-cluster.yml
version: '3.8'

services:
  redis-node-1:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    volumes:
      - ./redis-cluster/redis.conf:/usr/local/etc/redis/redis.conf
      - redis-node-1-data:/data
    ports:
      - "7001:6379"
    networks:
      - redis-cluster

  redis-node-2:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    volumes:
      - ./redis-cluster/redis.conf:/usr/local/etc/redis/redis.conf
      - redis-node-2-data:/data
    ports:
      - "7002:6379"
    networks:
      - redis-cluster

  redis-node-3:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    volumes:
      - ./redis-cluster/redis.conf:/usr/local/etc/redis/redis.conf
      - redis-node-3-data:/data
    ports:
      - "7003:6379"
    networks:
      - redis-cluster

  redis-node-4:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    volumes:
      - ./redis-cluster/redis.conf:/usr/local/etc/redis/redis.conf
      - redis-node-4-data:/data
    ports:
      - "7004:6379"
    networks:
      - redis-cluster

  redis-node-5:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    volumes:
      - ./redis-cluster/redis.conf:/usr/local/etc/redis/redis.conf
      - redis-node-5-data:/data
    ports:
      - "7005:6379"
    networks:
      - redis-cluster

  redis-node-6:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    volumes:
      - ./redis-cluster/redis.conf:/usr/local/etc/redis/redis.conf
      - redis-node-6-data:/data
    ports:
      - "7006:6379"
    networks:
      - redis-cluster

  redis-cluster-init:
    image: redis:7-alpine
    depends_on:
      - redis-node-1
      - redis-node-2
      - redis-node-3
      - redis-node-4
      - redis-node-5
      - redis-node-6
    command: >
      sh -c "
        sleep 10 &&
        redis-cli --cluster create 
        redis-node-1:6379 
        redis-node-2:6379 
        redis-node-3:6379 
        redis-node-4:6379 
        redis-node-5:6379 
        redis-node-6:6379 
        --cluster-replicas 1 --cluster-yes
      "
    networks:
      - redis-cluster

volumes:
  redis-node-1-data:
  redis-node-2-data:
  redis-node-3-data:
  redis-node-4-data:
  redis-node-5-data:
  redis-node-6-data:

networks:
  redis-cluster:
    driver: bridge
```

### High Availability Setup

```python
# prsm/core/redis_ha.py
import asyncio
import logging
from typing import List, Optional, Dict, Any
import aioredis
from aioredis.sentinel import Sentinel

logger = logging.getLogger(__name__)

class RedisHighAvailability:
    """Redis HA configuration with Sentinel."""
    
    def __init__(
        self,
        sentinel_hosts: List[tuple],
        service_name: str = "mymaster",
        password: Optional[str] = None
    ):
        self.sentinel_hosts = sentinel_hosts
        self.service_name = service_name
        self.password = password
        self.sentinel = None
        self.master = None
        self.slave = None
    
    async def initialize(self):
        """Initialize Sentinel connections."""
        try:
            self.sentinel = Sentinel(
                self.sentinel_hosts,
                password=self.password
            )
            
            # Get master and slave connections
            self.master = self.sentinel.master_for(
                self.service_name,
                decode_responses=True,
                password=self.password
            )
            
            self.slave = self.sentinel.slave_for(
                self.service_name,
                decode_responses=True,
                password=self.password
            )
            
            # Test connections
            await self.master.ping()
            await self.slave.ping()
            
            logger.info("Redis HA initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis HA: {e}")
            raise
    
    async def get_master(self):
        """Get master Redis connection."""
        return self.master
    
    async def get_slave(self):
        """Get slave Redis connection for read operations."""
        return self.slave
    
    async def is_master_available(self) -> bool:
        """Check if master is available."""
        try:
            await self.master.ping()
            return True
        except:
            return False
    
    async def failover(self):
        """Trigger manual failover."""
        try:
            for sentinel in self.sentinel.sentinels:
                await sentinel.execute_command(
                    "SENTINEL", "FAILOVER", self.service_name
                )
            logger.info("Failover triggered")
            return True
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return False

# HA Redis cache implementation
class HARedisCache(RedisCache):
    """High availability Redis cache."""
    
    def __init__(self, settings: CacheSettings, ha_config: RedisHighAvailability):
        super().__init__(settings)
        self.ha = ha_config
    
    async def initialize(self):
        """Initialize HA Redis connections."""
        await self.ha.initialize()
        self._redis_master = await self.ha.get_master()
        self._redis_slave = await self.ha.get_slave()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from slave (read replica)."""
        try:
            value = await self._redis_slave.get(key)
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value.encode('latin1'))
                
        except Exception as e:
            # Fallback to master
            logger.warning(f"Slave read failed, trying master: {e}")
            return await super().get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value on master."""
        try:
            ttl = ttl or self.settings.default_ttl
            
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                serialized = pickle.dumps(value).decode('latin1')
            
            await self._redis_master.setex(key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.warning(f"Master write failed: {e}")
            return False
```

## 📊 Monitoring and Performance

### Redis Monitoring

```python
# prsm/monitoring/redis_monitor.py
import asyncio
import time
import logging
from typing import Dict, Any, List
from prometheus_client import Gauge, Counter, Histogram
from prsm.core.cache import redis_cache

logger = logging.getLogger(__name__)

# Prometheus metrics
redis_memory_usage = Gauge('redis_memory_usage_bytes', 'Redis memory usage in bytes')
redis_connected_clients = Gauge('redis_connected_clients', 'Number of connected Redis clients')
redis_operations_total = Counter('redis_operations_total', 'Total Redis operations', ['operation', 'status'])
redis_operation_duration = Histogram('redis_operation_duration_seconds', 'Redis operation duration')

class RedisMonitor:
    """Redis performance and health monitoring."""
    
    def __init__(self):
        self.cache = redis_cache
        self.running = False
    
    async def start_monitoring(self, interval: int = 30):
        """Start monitoring Redis metrics."""
        self.running = True
        
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
    
    async def _collect_metrics(self):
        """Collect Redis metrics."""
        try:
            info = await self.cache._async_redis.info()
            
            # Memory metrics
            memory_used = info.get('used_memory', 0)
            redis_memory_usage.set(memory_used)
            
            # Client metrics
            connected_clients = info.get('connected_clients', 0)
            redis_connected_clients.set(connected_clients)
            
            # Performance metrics
            await self._test_performance()
            
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
    
    async def _test_performance(self):
        """Test Redis performance."""
        # Test SET operation
        start_time = time.time()
        try:
            await self.cache.set("perf_test", "test_value", ttl=60)
            duration = time.time() - start_time
            redis_operation_duration.observe(duration)
            redis_operations_total.labels(operation='set', status='success').inc()
        except Exception:
            redis_operations_total.labels(operation='set', status='error').inc()
        
        # Test GET operation
        start_time = time.time()
        try:
            await self.cache.get("perf_test")
            duration = time.time() - start_time
            redis_operation_duration.observe(duration)
            redis_operations_total.labels(operation='get', status='success').inc()
        except Exception:
            redis_operations_total.labels(operation='get', status='error').inc()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get Redis health status."""
        try:
            start_time = time.time()
            await self.cache._async_redis.ping()
            ping_time = (time.time() - start_time) * 1000  # ms
            
            info = await self.cache._async_redis.info()
            
            return {
                "status": "healthy",
                "ping_time_ms": round(ping_time, 2),
                "memory_usage": info.get('used_memory_human', 'N/A'),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "redis_version": info.get('redis_version', 'N/A'),
                "uptime_seconds": info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def get_slow_queries(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get slow Redis queries."""
        try:
            slow_log = await self.cache._async_redis.slowlog_get(count)
            
            queries = []
            for entry in slow_log:
                queries.append({
                    "id": entry.get('id'),
                    "timestamp": entry.get('start_time'),
                    "duration_microseconds": entry.get('duration'),
                    "command": " ".join(entry.get('command', [])),
                    "client_ip": entry.get('client_address', 'N/A')
                })
            
            return queries
            
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []

# Global Redis monitor
redis_monitor = RedisMonitor()
```

## 🔧 Testing and Debugging

### Redis Testing Suite

```python
# tests/test_redis_integration.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from prsm.core.cache import RedisCache, CacheSettings
from prsm.core.session import SessionManager
from prsm.core.rate_limiter import RateLimiter

@pytest.fixture
async def redis_cache():
    """Test Redis cache fixture."""
    settings = CacheSettings(
        redis_url="redis://localhost:6379/15",  # Test database
        default_ttl=60
    )
    cache = RedisCache(settings)
    await cache.initialize()
    yield cache
    await cache.close()

@pytest.fixture
async def session_manager(redis_cache):
    """Test session manager fixture."""
    manager = SessionManager()
    manager.cache = redis_cache
    return manager

@pytest.fixture
async def rate_limiter(redis_cache):
    """Test rate limiter fixture."""
    limiter = RateLimiter()
    limiter.cache = redis_cache
    return limiter

class TestRedisCache:
    """Test Redis cache functionality."""
    
    async def test_basic_operations(self, redis_cache):
        """Test basic cache operations."""
        # Test set and get
        await redis_cache.set("test_key", "test_value")
        value = await redis_cache.get("test_key")
        assert value == "test_value"
        
        # Test JSON serialization
        data = {"key": "value", "number": 42}
        await redis_cache.set("json_key", data)
        retrieved = await redis_cache.get("json_key")
        assert retrieved == data
        
        # Test expiration
        await redis_cache.set("temp_key", "temp_value", ttl=1)
        assert await redis_cache.exists("temp_key")
        await asyncio.sleep(2)
        assert not await redis_cache.exists("temp_key")
    
    async def test_increment_operations(self, redis_cache):
        """Test increment operations."""
        # Test counter increment
        count1 = await redis_cache.increment("counter")
        assert count1 == 1
        
        count2 = await redis_cache.increment("counter", 5)
        assert count2 == 6
        
        # Test existing key
        await redis_cache.set("existing_counter", "10")
        count3 = await redis_cache.increment("existing_counter")
        assert count3 == 11
    
    async def test_error_handling(self, redis_cache):
        """Test error handling."""
        # Test get non-existent key
        value = await redis_cache.get("non_existent")
        assert value is None
        
        # Test delete non-existent key
        result = await redis_cache.delete("non_existent")
        assert not result

class TestSessionManager:
    """Test session management."""
    
    async def test_session_lifecycle(self, session_manager):
        """Test complete session lifecycle."""
        user_id = "test_user_123"
        session_data = {"preferences": {"theme": "dark"}}
        
        # Create session
        session_id = await session_manager.create_session(user_id, session_data)
        assert session_id is not None
        
        # Get session
        session = await session_manager.get_session(session_id)
        assert session["user_id"] == user_id
        assert session["data"]["preferences"]["theme"] == "dark"
        
        # Update session
        update_data = {"last_page": "/dashboard"}
        await session_manager.update_session(session_id, update_data)
        
        updated_session = await session_manager.get_session(session_id)
        assert updated_session["data"]["last_page"] == "/dashboard"
        
        # Delete session
        await session_manager.delete_session(session_id)
        deleted_session = await session_manager.get_session(session_id)
        assert deleted_session is None
    
    async def test_user_sessions(self, session_manager):
        """Test user session management."""
        user_id = "test_user_456"
        
        # Create multiple sessions
        session1 = await session_manager.create_session(user_id, {"device": "mobile"})
        session2 = await session_manager.create_session(user_id, {"device": "desktop"})
        
        # Get user sessions
        sessions = await session_manager.get_user_sessions(user_id)
        assert len(sessions) == 2
        
        devices = [s["data"]["device"] for s in sessions]
        assert "mobile" in devices
        assert "desktop" in devices

class TestRateLimiter:
    """Test rate limiting functionality."""
    
    async def test_rate_limiting(self, rate_limiter):
        """Test basic rate limiting."""
        identifier = "test_user"
        limit = 5
        window = 60
        
        # Test allowed requests
        for i in range(limit):
            result = await rate_limiter.is_allowed(identifier, limit, window)
            assert result["allowed"]
            assert result["remaining"] == limit - i - 1
        
        # Test exceeded limit
        result = await rate_limiter.is_allowed(identifier, limit, window)
        assert not result["allowed"]
        assert result["reason"] == "rate_limit_exceeded"
    
    async def test_burst_limiting(self, rate_limiter):
        """Test burst rate limiting."""
        identifier = "burst_user"
        limit = 10
        window = 60
        burst = 3
        
        # Test burst limit
        for i in range(burst):
            result = await rate_limiter.is_allowed(identifier, limit, window, burst)
            assert result["allowed"]
        
        # Test burst exceeded
        result = await rate_limiter.is_allowed(identifier, limit, window, burst)
        assert not result["allowed"]
        assert result["reason"] == "burst_limit_exceeded"
    
    async def test_reset_limit(self, rate_limiter):
        """Test resetting rate limits."""
        identifier = "reset_user"
        limit = 2
        window = 60
        
        # Exceed limit
        await rate_limiter.is_allowed(identifier, limit, window)
        await rate_limiter.is_allowed(identifier, limit, window)
        result = await rate_limiter.is_allowed(identifier, limit, window)
        assert not result["allowed"]
        
        # Reset and try again
        await rate_limiter.reset_limit(identifier, window)
        result = await rate_limiter.is_allowed(identifier, limit, window)
        assert result["allowed"]

class TestRedisPerformance:
    """Test Redis performance characteristics."""
    
    async def test_concurrent_operations(self, redis_cache):
        """Test concurrent Redis operations."""
        async def set_operation(i):
            await redis_cache.set(f"concurrent_key_{i}", f"value_{i}")
            return await redis_cache.get(f"concurrent_key_{i}")
        
        # Run 100 concurrent operations
        tasks = [set_operation(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed successfully
        assert len(results) == 100
        for i, result in enumerate(results):
            assert result == f"value_{i}"
    
    async def test_large_data_operations(self, redis_cache):
        """Test operations with large data."""
        large_data = {"data": "x" * 10000, "items": list(range(1000))}
        
        # Test storing and retrieving large data
        await redis_cache.set("large_key", large_data)
        retrieved = await redis_cache.get("large_key")
        
        assert retrieved["data"] == large_data["data"]
        assert retrieved["items"] == large_data["items"]

# Performance benchmark
async def benchmark_redis_operations():
    """Benchmark Redis operations."""
    import time
    
    settings = CacheSettings(redis_url="redis://localhost:6379/15")
    cache = RedisCache(settings)
    await cache.initialize()
    
    # Benchmark SET operations
    start_time = time.time()
    for i in range(1000):
        await cache.set(f"bench_key_{i}", f"value_{i}")
    set_duration = time.time() - start_time
    
    # Benchmark GET operations
    start_time = time.time()
    for i in range(1000):
        await cache.get(f"bench_key_{i}")
    get_duration = time.time() - start_time
    
    print(f"SET operations: {1000/set_duration:.2f} ops/sec")
    print(f"GET operations: {1000/get_duration:.2f} ops/sec")
    
    await cache.close()

if __name__ == "__main__":
    asyncio.run(benchmark_redis_operations())
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Redis server installed and configured
- [ ] Security settings applied (password, renamed commands)
- [ ] Memory limits configured
- [ ] Persistence settings configured
- [ ] Network security configured (bind address, firewall)

### Production Setup
- [ ] High availability configured (Sentinel/Cluster)
- [ ] Monitoring and alerting set up
- [ ] Backup strategy implemented
- [ ] Performance tuning completed
- [ ] Load testing performed

### Integration Testing
- [ ] Cache operations tested
- [ ] Session management tested
- [ ] Rate limiting verified
- [ ] Error handling tested
- [ ] Performance benchmarks established

---

**Need help with Redis integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).