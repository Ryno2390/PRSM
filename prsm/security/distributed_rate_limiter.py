"""
Distributed Rate Limiting System
===============================

Production-grade rate limiting using Redis for distributed enforcement.
Addresses Gemini's finding about in-memory rate limiting being unsuitable
for distributed production environments.

Features:
- Redis-based distributed rate limiting
- Multiple rate limit windows (minute, hour, day)
- IP-based and user-based limiting
- Sliding window algorithms
- Automatic IP blocking for abuse
- Fallback to database when Redis unavailable
"""

import asyncio
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

import redis.asyncio as redis
from fastapi import Request, HTTPException
from pydantic import BaseModel

from prsm.core.database_service import get_database_service
from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class RateLimitType(Enum):
    """Types of rate limits"""
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    limit: int
    window_seconds: int
    burst_limit: Optional[int] = None
    block_duration: Optional[int] = None


class RateLimitResult(BaseModel):
    """Result of rate limit check"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None
    message: Optional[str] = None


class DistributedRateLimiter:
    """
    Production-grade distributed rate limiter using Redis.
    
    Implements sliding window rate limiting with multiple tiers:
    - Per-user limits
    - Per-IP limits
    - Per-endpoint limits
    - Global system limits
    
    Features automatic escalation and IP blocking for abuse.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.database_service = get_database_service()
        
        # Rate limit configurations by context
        self.rate_limits = {
            # Standard API usage
            "api_default": [
                RateLimitRule("per_minute", 60, 60),
                RateLimitRule("per_hour", 1000, 3600),
                RateLimitRule("per_day", 10000, 86400)
            ],
            
            # Authentication endpoints (stricter)
            "auth": [
                RateLimitRule("per_minute", 10, 60, block_duration=300),
                RateLimitRule("per_hour", 50, 3600, block_duration=3600),
                RateLimitRule("per_day", 200, 86400)
            ],
            
            # ML/AI processing endpoints (resource intensive)
            "ml_processing": [
                RateLimitRule("per_minute", 10, 60),
                RateLimitRule("per_hour", 100, 3600),
                RateLimitRule("per_day", 500, 86400)
            ],
            
            # File upload endpoints
            "file_upload": [
                RateLimitRule("per_minute", 5, 60),
                RateLimitRule("per_hour", 50, 3600),
                RateLimitRule("per_day", 200, 86400)
            ],
            
            # Search endpoints
            "search": [
                RateLimitRule("per_minute", 30, 60),
                RateLimitRule("per_hour", 500, 3600),
                RateLimitRule("per_day", 2000, 86400)
            ],
            
            # Admin endpoints (privileged but still limited)
            "admin": [
                RateLimitRule("per_minute", 100, 60),
                RateLimitRule("per_hour", 2000, 3600),
                RateLimitRule("per_day", 20000, 86400)
            ]
        }
        
        # Global system limits (protect against DDoS)
        self.global_limits = [
            RateLimitRule("global_per_second", 1000, 1),
            RateLimitRule("global_per_minute", 10000, 60)
        ]
        
        # IP-based limits (prevent single IP abuse)
        self.ip_limits = [
            RateLimitRule("ip_per_minute", 100, 60, block_duration=300),
            RateLimitRule("ip_per_hour", 2000, 3600, block_duration=3600)
        ]
        
        # Initialize Redis connection
        asyncio.create_task(self._init_redis())
    
    async def _init_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        try:
            redis_url = settings.REDIS_URL or "redis://localhost:6379/1"  # Use DB 1 for rate limiting
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established for distributed rate limiting")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            logger.warning("⚠️ Falling back to database-only rate limiting")
            self.redis_client = None

    async def check_rate_limit(
        self,
        user_id: Optional[str],
        ip_address: str,
        endpoint: str,
        request: Request
    ) -> RateLimitResult:
        """
        Comprehensive rate limit check with multiple tiers.
        
        Process:
        1. Check if IP is blocked
        2. Check global system limits
        3. Check IP-specific limits
        4. Check user-specific limits
        5. Check endpoint-specific limits
        6. Record request if allowed
        """
        try:
            # 1. Check if IP is blocked
            if await self._is_ip_blocked(ip_address):
                return RateLimitResult(
                    allowed=False,
                    limit=0,
                    remaining=0,
                    reset_time=int(time.time() + 3600),  # 1 hour default
                    retry_after=3600,
                    message="IP address is blocked"
                )
            
            # 2. Check global system limits
            global_result = await self._check_global_limits()
            if not global_result.allowed:
                return global_result
            
            # 3. Check IP-specific limits
            ip_result = await self._check_ip_limits(ip_address)
            if not ip_result.allowed:
                # Consider auto-blocking for severe IP abuse
                await self._consider_ip_blocking(ip_address, ip_result)
                return ip_result
            
            # 4. Check user-specific limits (if authenticated)
            if user_id:
                user_result = await self._check_user_limits(user_id, endpoint)
                if not user_result.allowed:
                    return user_result
            
            # 5. Check endpoint-specific limits
            endpoint_result = await self._check_endpoint_limits(endpoint, user_id or ip_address)
            if not endpoint_result.allowed:
                return endpoint_result
            
            # 6. Record successful request
            await self._record_request(user_id, ip_address, endpoint)
            
            # Return most restrictive remaining count
            min_remaining = min(
                global_result.remaining,
                ip_result.remaining,
                endpoint_result.remaining
            )
            
            return RateLimitResult(
                allowed=True,
                limit=endpoint_result.limit,
                remaining=min_remaining,
                reset_time=endpoint_result.reset_time
            )
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open for availability (log the failure)
            await self._log_rate_limit_error(user_id, ip_address, endpoint, str(e))
            return RateLimitResult(
                allowed=True,
                limit=1000,
                remaining=999,
                reset_time=int(time.time() + 3600),
                message="Rate limiting temporarily unavailable"
            )

    async def _check_global_limits(self) -> RateLimitResult:
        """Check global system rate limits"""
        for rule in self.global_limits:
            result = await self._check_sliding_window(
                key=f"global:{rule.name}",
                rule=rule
            )
            if not result.allowed:
                return result
        
        return RateLimitResult(allowed=True, limit=10000, remaining=9999, reset_time=int(time.time() + 60))

    async def _check_ip_limits(self, ip_address: str) -> RateLimitResult:
        """Check IP-specific rate limits"""
        for rule in self.ip_limits:
            result = await self._check_sliding_window(
                key=f"ip:{ip_address}:{rule.name}",
                rule=rule
            )
            if not result.allowed:
                return result
        
        return RateLimitResult(allowed=True, limit=2000, remaining=1999, reset_time=int(time.time() + 3600))

    async def _check_user_limits(self, user_id: str, endpoint: str) -> RateLimitResult:
        """Check user-specific rate limits based on endpoint category"""
        endpoint_category = self._categorize_endpoint(endpoint)
        rules = self.rate_limits.get(endpoint_category, self.rate_limits["api_default"])
        
        for rule in rules:
            result = await self._check_sliding_window(
                key=f"user:{user_id}:{endpoint_category}:{rule.name}",
                rule=rule
            )
            if not result.allowed:
                return result
        
        return RateLimitResult(allowed=True, limit=1000, remaining=999, reset_time=int(time.time() + 3600))

    async def _check_endpoint_limits(self, endpoint: str, identifier: str) -> RateLimitResult:
        """Check endpoint-specific rate limits"""
        endpoint_category = self._categorize_endpoint(endpoint)
        rules = self.rate_limits.get(endpoint_category, self.rate_limits["api_default"])
        
        # Use most restrictive rule for response
        most_restrictive = None
        
        for rule in rules:
            result = await self._check_sliding_window(
                key=f"endpoint:{identifier}:{endpoint_category}:{rule.name}",
                rule=rule
            )
            if not result.allowed:
                return result
            
            if most_restrictive is None or result.remaining < most_restrictive.remaining:
                most_restrictive = result
        
        return most_restrictive or RateLimitResult(
            allowed=True, limit=60, remaining=59, reset_time=int(time.time() + 60)
        )

    async def _check_sliding_window(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """
        Implement sliding window rate limiting using Redis.
        Uses sorted sets for precise sliding window implementation.
        """
        if not self.redis_client:
            # Fallback to database implementation
            return await self._database_rate_limit_check(key, rule)
        
        try:
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Execute pipeline
            results = await pipe.execute()
            current_count = results[1]
            
            # Check if limit exceeded
            if current_count >= rule.limit:
                # Calculate reset time
                oldest_entry = await self.redis_client.zrange(key, 0, 0, withscores=True)
                reset_time = int(oldest_entry[0][1] + rule.window_seconds) if oldest_entry else int(current_time + rule.window_seconds)
                
                return RateLimitResult(
                    allowed=False,
                    limit=rule.limit,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=reset_time - int(current_time),
                    message=f"Rate limit exceeded for {rule.name}"
                )
            
            # Add current request to window
            await self.redis_client.zadd(key, {str(current_time): current_time})
            
            # Set expiry for cleanup
            await self.redis_client.expire(key, rule.window_seconds + 60)
            
            return RateLimitResult(
                allowed=True,
                limit=rule.limit,
                remaining=rule.limit - current_count - 1,
                reset_time=int(current_time + rule.window_seconds)
            )
            
        except Exception as e:
            logger.error(f"Redis sliding window check failed for {key}: {e}")
            # Fallback to database
            return await self._database_rate_limit_check(key, rule)

    async def _database_rate_limit_check(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Fallback database-based rate limiting"""
        try:
            current_time = datetime.now(timezone.utc)
            window_start = current_time - timedelta(seconds=rule.window_seconds)
            
            async with self.database_service.get_session() as session:
                # Count requests in current window
                count_query = """
                    SELECT COUNT(*) FROM rate_limit_log 
                    WHERE rate_key = :key 
                    AND timestamp > :window_start
                """
                
                result = await session.execute(count_query, {
                    "key": key,
                    "window_start": window_start
                })
                
                current_count = result.scalar() or 0
                
                if current_count >= rule.limit:
                    return RateLimitResult(
                        allowed=False,
                        limit=rule.limit,
                        remaining=0,
                        reset_time=int(time.time() + rule.window_seconds),
                        message=f"Rate limit exceeded (database fallback)"
                    )
                
                # Record this request
                insert_query = """
                    INSERT INTO rate_limit_log (rate_key, timestamp, rule_name)
                    VALUES (:key, :timestamp, :rule_name)
                """
                
                await session.execute(insert_query, {
                    "key": key,
                    "timestamp": current_time,
                    "rule_name": rule.name
                })
                
                await session.commit()
                
                return RateLimitResult(
                    allowed=True,
                    limit=rule.limit,
                    remaining=rule.limit - current_count - 1,
                    reset_time=int(time.time() + rule.window_seconds)
                )
                
        except Exception as e:
            logger.error(f"Database rate limit check failed: {e}")
            # Ultimate fallback - allow request but log
            return RateLimitResult(
                allowed=True,
                limit=rule.limit,
                remaining=rule.limit - 1,
                reset_time=int(time.time() + rule.window_seconds),
                message="Rate limiting unavailable"
            )

    def _categorize_endpoint(self, endpoint: str) -> str:
        """Categorize endpoint for appropriate rate limiting"""
        endpoint = endpoint.lower()
        
        if "/auth/" in endpoint or "/login" in endpoint or "/register" in endpoint:
            return "auth"
        elif "/ml/" in endpoint or "/ai/" in endpoint or "/process" in endpoint:
            return "ml_processing"
        elif "/upload" in endpoint or "/file" in endpoint:
            return "file_upload"
        elif "/search" in endpoint or "/query" in endpoint:
            return "search"
        elif "/admin/" in endpoint:
            return "admin"
        else:
            return "api_default"

    async def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        try:
            if self.redis_client:
                # Check Redis blocked IPs set
                is_blocked = await self.redis_client.sismember("blocked_ips", ip_address)
                if is_blocked:
                    return True
            
            # Check database for persistent blocks
            async with self.database_service.get_session() as session:
                query = """
                    SELECT COUNT(*) FROM blocked_ips 
                    WHERE ip_address = :ip_address 
                    AND is_active = true
                    AND (expires_at IS NULL OR expires_at > NOW())
                """
                
                result = await session.execute(query, {"ip_address": ip_address})
                count = result.scalar() or 0
                
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to check blocked IP {ip_address}: {e}")
            return False

    async def _consider_ip_blocking(self, ip_address: str, rate_limit_result: RateLimitResult):
        """Consider blocking IP for severe rate limit violations"""
        try:
            # Check if this IP has been rate limited frequently
            if self.redis_client:
                violations_key = f"violations:{ip_address}"
                violations = await self.redis_client.incr(violations_key)
                await self.redis_client.expire(violations_key, 3600)  # 1 hour window
                
                # Block if more than 10 violations in an hour
                if violations >= 10:
                    await self.block_ip(ip_address, "Automatic block due to repeated rate limit violations", duration=3600)
                    
        except Exception as e:
            logger.error(f"Failed to consider IP blocking for {ip_address}: {e}")

    async def block_ip(self, ip_address: str, reason: str, duration: Optional[int] = None):
        """Block IP address for abuse"""
        try:
            expires_at = None
            if duration:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=duration)
            
            # Add to Redis for immediate effect
            if self.redis_client:
                await self.redis_client.sadd("blocked_ips", ip_address)
                if duration:
                    await self.redis_client.expire(f"blocked_ip:{ip_address}", duration)
            
            # Add to database for persistence
            async with self.database_service.get_session() as session:
                query = """
                    INSERT INTO blocked_ips (ip_address, reason, expires_at)
                    VALUES (:ip_address, :reason, :expires_at)
                    ON CONFLICT (ip_address) DO UPDATE SET
                        reason = :reason,
                        expires_at = :expires_at,
                        is_active = true,
                        blocked_at = NOW()
                """
                
                await session.execute(query, {
                    "ip_address": ip_address,
                    "reason": reason,
                    "expires_at": expires_at
                })
                
                await session.commit()
                
            logger.warning(f"IP blocked: {ip_address} - {reason}")
            
        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {e}")

    async def _record_request(self, user_id: Optional[str], ip_address: str, endpoint: str):
        """Record successful request for analytics"""
        try:
            if self.redis_client:
                # Record in Redis for real-time analytics
                analytics_key = f"analytics:{datetime.now().strftime('%Y%m%d%H')}"  # Hourly buckets
                await self.redis_client.hincrby(analytics_key, endpoint, 1)
                await self.redis_client.expire(analytics_key, 86400)  # Keep for 24 hours
                
        except Exception as e:
            logger.debug(f"Failed to record request analytics: {e}")

    async def _log_rate_limit_error(self, user_id: Optional[str], ip_address: str, endpoint: str, error: str):
        """Log rate limiting system errors"""
        try:
            async with self.database_service.get_session() as session:
                query = """
                    INSERT INTO security_events (event_type, user_id, ip_address, severity, details)
                    VALUES ('rate_limit_error', :user_id, :ip_address, 'medium', :details)
                """
                
                await session.execute(query, {
                    "user_id": user_id,
                    "ip_address": ip_address,
                    "details": json.dumps({
                        "endpoint": endpoint,
                        "error": error,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log rate limit error: {e}")

    async def get_rate_limit_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiting statistics for monitoring"""
        try:
            stats = {
                "total_requests_last_hour": 0,
                "blocked_ips_count": 0,
                "rate_limit_violations_last_hour": 0,
                "top_endpoints": []
            }
            
            if self.redis_client:
                # Get hourly analytics
                current_hour = datetime.now().strftime('%Y%m%d%H')
                analytics_key = f"analytics:{current_hour}"
                hourly_stats = await self.redis_client.hgetall(analytics_key)
                
                if hourly_stats:
                    stats["total_requests_last_hour"] = sum(int(v) for v in hourly_stats.values())
                    stats["top_endpoints"] = sorted(
                        hourly_stats.items(), 
                        key=lambda x: int(x[1]), 
                        reverse=True
                    )[:10]
                
                # Get blocked IPs count
                stats["blocked_ips_count"] = await self.redis_client.scard("blocked_ips")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get rate limit stats: {e}")
            return {"error": str(e)}

    async def cleanup_expired_blocks(self) -> int:
        """Clean up expired IP blocks (run periodically)"""
        try:
            cleaned_count = 0
            
            # Clean up database
            async with self.database_service.get_session() as session:
                query = """
                    UPDATE blocked_ips 
                    SET is_active = false 
                    WHERE expires_at < NOW() AND is_active = true
                """
                
                result = await session.execute(query)
                await session.commit()
                cleaned_count = result.rowcount
            
            # Clean up Redis
            if self.redis_client:
                # Redis TTL handles automatic cleanup, but we can manually sync
                pass
            
            logger.info(f"Cleaned up {cleaned_count} expired IP blocks")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired blocks: {e}")
            return 0


# Global instance
_rate_limiter = None

async def get_rate_limiter() -> DistributedRateLimiter:
    """Get the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = DistributedRateLimiter()
    return _rate_limiter


# FastAPI middleware integration
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware for FastAPI"""
    rate_limiter = await get_rate_limiter()
    
    # Extract user info (implement based on your auth system)
    user_id = getattr(request.state, 'user_id', None)
    ip_address = request.client.host
    endpoint = request.url.path
    
    # Check rate limits
    result = await rate_limiter.check_rate_limit(user_id, ip_address, endpoint, request)
    
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail=result.message or "Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": str(result.remaining),
                "X-RateLimit-Reset": str(result.reset_time),
                "Retry-After": str(result.retry_after or 60)
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(result.limit)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-RateLimit-Reset"] = str(result.reset_time)
    
    return response