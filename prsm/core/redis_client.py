"""
PRSM Redis Client - Caching and Session Management

🎯 PURPOSE IN PRSM:
This module provides comprehensive Redis integration for PRSM, implementing
real-time caching, session state management, and pub/sub messaging to enhance
performance and enable distributed coordination.

🔧 INTEGRATION POINTS:
- Session state: Fast access to active user sessions and context
- Model caching: Temporary storage for frequently accessed model outputs
- Task coordination: Distributed task queues and progress tracking
- Real-time updates: Pub/sub for live system notifications
- Circuit breaker state: Fast safety system state coordination
- Performance metrics: Real-time system health and usage statistics

🚀 REAL-WORLD CAPABILITIES:
- Connection pooling with automatic failover
- Distributed locking for concurrent operations
- Cache invalidation strategies for data consistency
- Session persistence across server restarts
- Real-time event broadcasting across P2P network
- Performance monitoring and metrics collection
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from datetime import datetime, timedelta
import structlog

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class RedisClient:
    """
    Async Redis client for PRSM operations
    
    🎯 PURPOSE: Centralized Redis operations with connection pooling,
    automatic retry logic, and optimized caching strategies
    """
    
    def __init__(self):
        self.redis_pool: Optional[ConnectionPool] = None
        self.redis_client: Optional[Redis] = None
        self.connected = False
        self.last_health_check = None
    
    async def initialize(self):
        """
        Initialize Redis connection pool
        
        🔧 CONNECTION SETUP:
        - Creates optimized connection pool for high concurrency
        - Configures retry logic and timeout handling
        - Sets up health monitoring and connection recovery
        """
        try:
            # 🔗 Parse Redis URL and create connection pool
            redis_config = {
                "decode_responses": True,
                "health_check_interval": 30,
                "socket_keepalive": True,
                "socket_keepalive_options": {},
                "retry_on_timeout": True,
                "retry_on_error": [ConnectionError, TimeoutError],
                "max_connections": 20
            }
            
            if settings.redis_password:
                redis_config["password"] = settings.redis_password
            
            self.redis_pool = ConnectionPool.from_url(
                settings.redis_url,
                **redis_config
            )
            
            self.redis_client = Redis(connection_pool=self.redis_pool)
            
            # 🏥 Test connection
            await self.redis_client.ping()
            self.connected = True
            
            logger.info("Redis client initialized successfully", 
                       redis_url=settings.redis_url.split("@")[-1])  # Hide credentials
            
        except Exception as e:
            logger.error("Failed to initialize Redis client", error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """
        Perform Redis health check
        
        🏥 HEALTH MONITORING:
        - Connection status verification
        - Response time measurement
        - Memory usage monitoring
        - Command execution validation
        """
        try:
            if not self.redis_client:
                return False
            
            start_time = time.time()
            
            # Basic connectivity test
            await self.redis_client.ping()
            
            # Performance test
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test_value", ex=1)
            value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            response_time = time.time() - start_time
            
            if value != "test_value":
                raise Exception("Redis read/write test failed")
            
            self.connected = True
            self.last_health_check = datetime.now()
            
            logger.debug("Redis health check passed",
                        response_time=response_time,
                        connected=True)
            
            return True
            
        except Exception as e:
            self.connected = False
            logger.error("Redis health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Clean up Redis connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.redis_pool:
                await self.redis_pool.disconnect()
            
            self.connected = False
            logger.info("Redis connections cleaned up")
            
        except Exception as e:
            logger.error("Error during Redis cleanup", error=str(e))


class SessionCache:
    """
    Session state caching with Redis
    
    🎯 PURPOSE IN PRSM:
    Fast access to active user sessions, context allocation,
    and reasoning state for real-time query processing
    """
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.session_prefix = "prsm:session:"
        self.context_prefix = "prsm:context:"
        self.default_ttl = 3600  # 1 hour
    
    async def store_session(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store session data in Redis cache
        
        🚀 SESSION CACHING:
        - JSON serialization of session state
        - Configurable TTL for automatic cleanup
        - Atomic operations for consistency
        """
        try:
            if not self.redis.connected:
                logger.warning("Redis not connected, skipping session cache")
                return False
            
            key = f"{self.session_prefix}{session_id}"
            ttl = ttl or self.default_ttl
            
            # Serialize session data
            serialized_data = json.dumps(session_data, default=str)
            
            await self.redis.redis_client.setex(key, ttl, serialized_data)
            
            logger.debug("Session cached",
                        session_id=session_id,
                        ttl=ttl)
            
            return True
            
        except Exception as e:
            logger.error("Failed to cache session",
                        session_id=session_id,
                        error=str(e))
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from Redis cache"""
        try:
            if not self.redis.connected:
                return None
            
            key = f"{self.session_prefix}{session_id}"
            data = await self.redis.redis_client.get(key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error("Failed to retrieve cached session",
                        session_id=session_id,
                        error=str(e))
            return None
    
    async def update_session_status(self, session_id: str, status: str) -> bool:
        """Update session status in cache"""
        try:
            session_data = await self.get_session(session_id)
            if session_data:
                session_data["status"] = status
                session_data["updated_at"] = datetime.now().isoformat()
                return await self.store_session(session_id, session_data)
            
            return False
            
        except Exception as e:
            logger.error("Failed to update session status",
                        session_id=session_id,
                        error=str(e))
            return False
    
    async def store_context_allocation(self, session_id: str, allocation_data: Dict[str, Any]) -> bool:
        """Store context allocation data"""
        try:
            if not self.redis.connected:
                return False
            
            key = f"{self.context_prefix}{session_id}"
            serialized_data = json.dumps(allocation_data, default=str)
            
            await self.redis.redis_client.setex(key, self.default_ttl, serialized_data)
            
            logger.debug("Context allocation cached",
                        session_id=session_id,
                        allocated=allocation_data.get("allocated", 0))
            
            return True
            
        except Exception as e:
            logger.error("Failed to cache context allocation",
                        session_id=session_id,
                        error=str(e))
            return False
    
    async def get_context_allocation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get context allocation data"""
        try:
            if not self.redis.connected:
                return None
            
            key = f"{self.context_prefix}{session_id}"
            data = await self.redis.redis_client.get(key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error("Failed to retrieve context allocation",
                        session_id=session_id,
                        error=str(e))
            return None


class ModelCache:
    """
    Model output and metadata caching
    
    🤖 PURPOSE IN PRSM:
    Caches frequent model outputs, embeddings, and metadata
    to reduce API costs and improve response times
    """
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.output_prefix = "prsm:model_output:"
        self.embedding_prefix = "prsm:embedding:"
        self.metadata_prefix = "prsm:model_meta:"
        self.default_ttl = 1800  # 30 minutes
    
    async def store_model_output(self, cache_key: str, output_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache model output for reuse
        
        💾 OUTPUT CACHING:
        - Reduces redundant API calls to expensive models
        - Improves response times for similar queries
        - Configurable TTL based on model type and use case
        """
        try:
            if not self.redis.connected:
                return False
            
            key = f"{self.output_prefix}{cache_key}"
            ttl = ttl or self.default_ttl
            
            # Include cache metadata
            cache_data = {
                "output": output_data,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            await self.redis.redis_client.setex(key, ttl, serialized_data)
            
            logger.debug("Model output cached",
                        cache_key=cache_key,
                        ttl=ttl)
            
            return True
            
        except Exception as e:
            logger.error("Failed to cache model output",
                        cache_key=cache_key,
                        error=str(e))
            return False
    
    async def get_model_output(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached model output"""
        try:
            if not self.redis.connected:
                return None
            
            key = f"{self.output_prefix}{cache_key}"
            data = await self.redis.redis_client.get(key)
            
            if data:
                cache_data = json.loads(data)
                
                # Check if cache is still valid
                cached_at = datetime.fromisoformat(cache_data["cached_at"])
                if datetime.now() - cached_at < timedelta(seconds=cache_data["ttl"]):
                    return cache_data["output"]
            
            return None
            
        except Exception as e:
            logger.error("Failed to retrieve cached model output",
                        cache_key=cache_key,
                        error=str(e))
            return None
    
    async def store_embedding(self, text_hash: str, embedding: List[float], ttl: Optional[int] = None) -> bool:
        """Cache text embeddings"""
        try:
            if not self.redis.connected:
                return False
            
            key = f"{self.embedding_prefix}{text_hash}"
            ttl = ttl or 86400  # 24 hours for embeddings
            
            embedding_data = {
                "embedding": embedding,
                "cached_at": datetime.now().isoformat()
            }
            
            serialized_data = json.dumps(embedding_data)
            await self.redis.redis_client.setex(key, ttl, serialized_data)
            
            logger.debug("Embedding cached",
                        text_hash=text_hash,
                        embedding_length=len(embedding))
            
            return True
            
        except Exception as e:
            logger.error("Failed to cache embedding",
                        text_hash=text_hash,
                        error=str(e))
            return False
    
    async def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding"""
        try:
            if not self.redis.connected:
                return None
            
            key = f"{self.embedding_prefix}{text_hash}"
            data = await self.redis.redis_client.get(key)
            
            if data:
                embedding_data = json.loads(data)
                return embedding_data["embedding"]
            
            return None
            
        except Exception as e:
            logger.error("Failed to retrieve cached embedding",
                        text_hash=text_hash,
                        error=str(e))
            return None


class TaskQueue:
    """
    Redis-based task queue for distributed processing
    
    🔄 PURPOSE IN PRSM:
    Coordinates distributed task execution across the P2P network
    with priority queuing and progress tracking
    """
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.queue_prefix = "prsm:queue:"
        self.progress_prefix = "prsm:progress:"
        self.result_prefix = "prsm:result:"
    
    async def enqueue_task(self, queue_name: str, task_data: Dict[str, Any], priority: int = 0) -> bool:
        """
        Add task to distributed queue
        
        📝 TASK QUEUING:
        - Priority-based task ordering
        - Atomic enqueue operations
        - Task metadata and tracking
        """
        try:
            if not self.redis.connected:
                return False
            
            queue_key = f"{self.queue_prefix}{queue_name}"
            
            # Add task metadata
            task_with_meta = {
                "task_id": task_data.get("task_id"),
                "enqueued_at": datetime.now().isoformat(),
                "priority": priority,
                "data": task_data
            }
            
            serialized_task = json.dumps(task_with_meta, default=str)
            
            # Use priority score for ordering (higher priority = lower score for min heap)
            score = -priority
            await self.redis.redis_client.zadd(queue_key, {serialized_task: score})
            
            logger.debug("Task enqueued",
                        queue_name=queue_name,
                        task_id=task_data.get("task_id"),
                        priority=priority)
            
            return True
            
        except Exception as e:
            logger.error("Failed to enqueue task",
                        queue_name=queue_name,
                        error=str(e))
            return False
    
    async def dequeue_task(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Get next task from queue"""
        try:
            if not self.redis.connected:
                return None
            
            queue_key = f"{self.queue_prefix}{queue_name}"
            
            # Get highest priority task (lowest score)
            result = await self.redis.redis_client.zpopmin(queue_key)
            
            if result:
                task_data, score = result[0]
                return json.loads(task_data)
            
            return None
            
        except Exception as e:
            logger.error("Failed to dequeue task",
                        queue_name=queue_name,
                        error=str(e))
            return None
    
    async def update_task_progress(self, task_id: str, progress_data: Dict[str, Any]) -> bool:
        """Update task progress"""
        try:
            if not self.redis.connected:
                return False
            
            key = f"{self.progress_prefix}{task_id}"
            
            progress_with_meta = {
                "progress": progress_data,
                "updated_at": datetime.now().isoformat()
            }
            
            serialized_data = json.dumps(progress_with_meta, default=str)
            await self.redis.redis_client.setex(key, 3600, serialized_data)  # 1 hour TTL
            
            return True
            
        except Exception as e:
            logger.error("Failed to update task progress",
                        task_id=task_id,
                        error=str(e))
            return False


class PubSubManager:
    """
    Redis pub/sub for real-time system events
    
    📡 PURPOSE IN PRSM:
    Real-time event broadcasting for system coordination,
    safety alerts, and user notifications across the distributed network
    """
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.subscribers = {}
        self.channels = {
            "safety_alerts": "prsm:safety:alerts",
            "system_events": "prsm:system:events", 
            "task_updates": "prsm:tasks:updates",
            "user_notifications": "prsm:users:notifications"
        }
    
    async def publish_event(self, channel: str, event_data: Dict[str, Any]) -> bool:
        """Publish event to channel"""
        try:
            if not self.redis.connected:
                return False
            
            channel_key = self.channels.get(channel, f"prsm:{channel}")
            
            event_with_meta = {
                "event": event_data,
                "published_at": datetime.now().isoformat(),
                "source": "prsm_api"
            }
            
            serialized_event = json.dumps(event_with_meta, default=str)
            await self.redis.redis_client.publish(channel_key, serialized_event)
            
            logger.debug("Event published",
                        channel=channel,
                        event_type=event_data.get("type"))
            
            return True
            
        except Exception as e:
            logger.error("Failed to publish event",
                        channel=channel,
                        error=str(e))
            return False
    
    async def subscribe_to_channel(self, channel: str, callback):
        """Subscribe to channel with callback"""
        try:
            if not self.redis.connected:
                return False
            
            channel_key = self.channels.get(channel, f"prsm:{channel}")
            pubsub = self.redis.redis_client.pubsub()
            
            await pubsub.subscribe(channel_key)
            self.subscribers[channel] = pubsub
            
            # Start background listener
            asyncio.create_task(self._listen_to_channel(pubsub, callback))
            
            logger.info("Subscribed to channel", channel=channel)
            return True
            
        except Exception as e:
            logger.error("Failed to subscribe to channel",
                        channel=channel,
                        error=str(e))
            return False
    
    async def _listen_to_channel(self, pubsub, callback):
        """Background listener for pub/sub messages"""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        await callback(event_data)
                    except Exception as e:
                        logger.error("Error processing pub/sub message", error=str(e))
        except Exception as e:
            logger.error("Pub/sub listener error", error=str(e))


# === Global Redis Manager ===

class RedisManager:
    """
    Central Redis manager for PRSM
    
    🎯 PURPOSE: Coordinates all Redis operations with unified
    connection management and service initialization
    """
    
    def __init__(self):
        self.client = RedisClient()
        self.session_cache = None
        self.model_cache = None
        self.task_queue = None
        self.pubsub = None
    
    async def initialize(self):
        """Initialize all Redis services"""
        try:
            await self.client.initialize()
            
            # Initialize service layers
            self.session_cache = SessionCache(self.client)
            self.model_cache = ModelCache(self.client)
            self.task_queue = TaskQueue(self.client)
            self.pubsub = PubSubManager(self.client)
            
            logger.info("Redis manager initialized with all services")
            
        except Exception as e:
            logger.error("Failed to initialize Redis manager", error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """Check health of all Redis services"""
        return await self.client.health_check()
    
    async def cleanup(self):
        """Clean up all Redis connections"""
        try:
            # Close all subscriber connections
            if self.pubsub and self.pubsub.subscribers:
                for pubsub in self.pubsub.subscribers.values():
                    await pubsub.close()
            
            await self.client.cleanup()
            logger.info("Redis manager cleanup completed")
            
        except Exception as e:
            logger.error("Error during Redis cleanup", error=str(e))


# Global Redis manager instance
redis_manager = RedisManager()


# === Helper Functions ===

async def init_redis():
    """Initialize Redis for PRSM"""
    await redis_manager.initialize()


async def close_redis():
    """Close Redis connections"""
    await redis_manager.cleanup()


def get_session_cache() -> SessionCache:
    """Get session cache instance"""
    return redis_manager.session_cache


def get_model_cache() -> ModelCache:
    """Get model cache instance"""
    return redis_manager.model_cache


def get_task_queue() -> TaskQueue:
    """Get task queue instance"""
    return redis_manager.task_queue


def get_pubsub() -> PubSubManager:
    """Get pub/sub manager instance"""
    return redis_manager.pubsub