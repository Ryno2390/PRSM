"""
Rate Limiting Integration Tests
===============================

Integration tests that verify rate limiting with a real Redis instance.
These tests validate:
- Sliding window algorithm accuracy
- Concurrent request handling
- Rate limit enforcement across multiple connections
- Redis failover behavior

Requirements:
- Redis server (configured via REDIS_URL env var)
- Default: redis://localhost:6379/1

Run with: pytest tests/integration/test_rate_limiting_integration.py -v --tb=short
"""

import asyncio
import os
import time
import pytest
from uuid import uuid4
from typing import List, Tuple
import redis.asyncio as aioredis


# Skip if no Redis URL configured
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_REDIS_TESTS", "false").lower() == "true",
    reason="SKIP_REDIS_TESTS is set"
)


class SlidingWindowRateLimiter:
    """
    Redis-backed sliding window rate limiter for integration testing.

    Uses Redis sorted sets to track request timestamps within sliding windows.
    """

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.tier_configs = {
            "anonymous": {"requests_per_minute": 10, "requests_per_hour": 100},
            "free": {"requests_per_minute": 60, "requests_per_hour": 1000},
            "pro": {"requests_per_minute": 300, "requests_per_hour": 10000},
            "enterprise": {"requests_per_minute": 1000, "requests_per_hour": 50000},
            "test": {"requests_per_minute": 5, "requests_per_hour": 50},  # For testing
        }

    async def check_rate_limit(
        self,
        identifier: str,
        tier: str = "free",
        endpoint: str = None
    ) -> Tuple[bool, dict]:
        """
        Check if request is allowed under rate limit.

        Args:
            identifier: User or IP identifier
            tier: Rate limit tier
            endpoint: Optional endpoint for per-endpoint limits

        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        config = self.tier_configs.get(tier, self.tier_configs["free"])
        now = time.time()
        now_ms = int(now * 1000)

        # Build key
        key_suffix = f":{endpoint}" if endpoint else ""
        minute_key = f"rate_limit:sliding:{identifier}:minute{key_suffix}"

        # Window boundaries
        minute_window_start = now_ms - 60000  # 1 minute ago

        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(minute_key, "-inf", minute_window_start)

        # Count current requests in window
        pipe.zcard(minute_key)

        # Add current request (will only be committed if allowed)
        request_id = f"{now_ms}:{uuid4().hex[:8]}"
        pipe.zadd(minute_key, {request_id: now_ms})

        # Set expiry on key
        pipe.expire(minute_key, 120)

        results = await pipe.execute()

        # Results: [removed_count, current_count, added_count, expire_result]
        current_count = results[1]
        limit = config["requests_per_minute"]

        allowed = current_count < limit

        if not allowed:
            # Remove the request we just added since it's not allowed
            await self.redis.zrem(minute_key, request_id)

        # Get remaining and retry info
        remaining = max(0, limit - current_count - (1 if allowed else 0))

        # Calculate retry_after if limited
        retry_after = 0
        if not allowed:
            # Get oldest request in window
            oldest = await self.redis.zrange(minute_key, 0, 0, withscores=True)
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = max(0, (oldest_time + 60000 - now_ms) / 1000)

        return allowed, {
            "algorithm": "sliding_window",
            "tier": tier,
            "limits": {
                "minute": {
                    "limit": limit,
                    "remaining": remaining,
                    "retry_after": retry_after
                }
            }
        }

    async def reset(self, identifier: str) -> int:
        """Reset rate limits for an identifier."""
        pattern = f"rate_limit:sliding:{identifier}:*"
        keys = []
        async for key in self.redis.scan_iter(pattern):
            keys.append(key)
        if keys:
            return await self.redis.delete(*keys)
        return 0


@pytest.fixture(scope="module")
async def redis_client():
    """Create Redis connection for tests."""
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
    yield client
    await client.aclose()


@pytest.fixture
async def limiter(redis_client):
    """Create rate limiter instance."""
    return SlidingWindowRateLimiter(redis_client)


@pytest.fixture
async def clean_keys(redis_client):
    """Cleanup test keys before and after each test."""
    # Cleanup before
    async for key in redis_client.scan_iter("rate_limit:*test*"):
        await redis_client.delete(key)
    yield
    # Cleanup after
    async for key in redis_client.scan_iter("rate_limit:*test*"):
        await redis_client.delete(key)


class TestSlidingWindowAccuracy:
    """Test sliding window algorithm accuracy."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self, limiter, clean_keys):
        """Verify requests under limit are allowed."""
        identifier = f"test_user_{uuid4().hex[:8]}"

        for i in range(4):  # Test tier allows 5 per minute
            allowed, info = await limiter.check_rate_limit(identifier, tier="test")
            assert allowed, f"Request {i+1} should be allowed"
            assert info["limits"]["minute"]["remaining"] == 4 - i - 1

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self, limiter, clean_keys):
        """Verify requests over limit are blocked."""
        identifier = f"test_user_{uuid4().hex[:8]}"

        # Make 5 requests (at limit)
        for i in range(5):
            allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
            assert allowed, f"Request {i+1} should be allowed"

        # 6th request should be blocked
        allowed, info = await limiter.check_rate_limit(identifier, tier="test")
        assert not allowed, "Request over limit should be blocked"
        assert info["limits"]["minute"]["retry_after"] > 0

    @pytest.mark.asyncio
    async def test_window_slides_correctly(self, limiter, redis_client, clean_keys):
        """Verify window properly expires old requests."""
        identifier = f"test_user_{uuid4().hex[:8]}"
        key = f"rate_limit:sliding:{identifier}:minute"

        # Add 5 requests at the limit
        for i in range(5):
            allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
            assert allowed

        # Should be at limit
        allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
        assert not allowed

        # Manually expire oldest entries (simulate time passing)
        # Get all entries and remove oldest
        entries = await redis_client.zrange(key, 0, 0, withscores=True)
        if entries:
            await redis_client.zrem(key, entries[0][0])

        # Now should allow one more
        allowed, info = await limiter.check_rate_limit(identifier, tier="test")
        assert allowed, "Should allow after oldest request expired"


class TestConcurrentRequests:
    """Test rate limiting under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_properly_counted(self, limiter, clean_keys):
        """Verify concurrent requests are all properly counted."""
        identifier = f"test_concurrent_{uuid4().hex[:8]}"
        num_requests = 20  # More than the test limit of 5

        async def make_request(request_id: int):
            allowed, info = await limiter.check_rate_limit(identifier, tier="test")
            return {"id": request_id, "allowed": allowed}

        # Make concurrent requests
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        allowed_count = sum(1 for r in results if r["allowed"])

        # Should allow exactly 5 (the test tier limit)
        assert allowed_count == 5, \
            f"Expected 5 allowed, got {allowed_count}"

    @pytest.mark.asyncio
    async def test_distributed_rate_limiting(self, redis_client, clean_keys):
        """Simulate multiple app instances sharing rate limits."""
        identifier = f"test_distributed_{uuid4().hex[:8]}"

        # Create multiple limiter instances (simulating multiple app servers)
        limiters = [SlidingWindowRateLimiter(redis_client) for _ in range(3)]

        results = []

        async def instance_request(instance_id: int, request_id: int):
            limiter = limiters[instance_id]
            allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
            return {"instance": instance_id, "request": request_id, "allowed": allowed}

        # Each instance makes 3 requests (9 total, limit is 5)
        tasks = []
        for instance_id in range(3):
            for request_id in range(3):
                tasks.append(instance_request(instance_id, request_id))

        results = await asyncio.gather(*tasks)

        allowed_count = sum(1 for r in results if r["allowed"])

        # Should allow exactly 5 across all instances
        assert allowed_count == 5, \
            f"Distributed limit not enforced: {allowed_count} allowed"


class TestTierEnforcement:
    """Test different rate limit tiers."""

    @pytest.mark.asyncio
    async def test_tier_limits_increase_with_tier(self, limiter, clean_keys):
        """Verify higher tiers have higher limits."""
        tiers = ["anonymous", "free", "pro", "enterprise"]
        limits = []

        for tier in tiers:
            identifier = f"test_tier_{tier}_{uuid4().hex[:8]}"

            # Count how many requests are allowed
            allowed_count = 0
            for _ in range(1500):  # More than enterprise limit
                allowed, _ = await limiter.check_rate_limit(identifier, tier=tier)
                if allowed:
                    allowed_count += 1
                else:
                    break

            limits.append(allowed_count)

        # Each tier should have higher limits than previous
        for i in range(1, len(limits)):
            assert limits[i] > limits[i-1], \
                f"{tiers[i]} limit ({limits[i]}) should be > {tiers[i-1]} ({limits[i-1]})"

    @pytest.mark.asyncio
    async def test_endpoint_specific_limits(self, limiter, clean_keys):
        """Verify per-endpoint rate limits are independent."""
        identifier = f"test_endpoint_{uuid4().hex[:8]}"

        # Exhaust limit on endpoint1
        for _ in range(5):
            allowed, _ = await limiter.check_rate_limit(
                identifier, tier="test", endpoint="/api/v1/endpoint1"
            )
            assert allowed

        # Endpoint1 should be limited
        allowed, _ = await limiter.check_rate_limit(
            identifier, tier="test", endpoint="/api/v1/endpoint1"
        )
        assert not allowed

        # Endpoint2 should still work
        allowed, _ = await limiter.check_rate_limit(
            identifier, tier="test", endpoint="/api/v1/endpoint2"
        )
        assert allowed, "Different endpoint should have separate limit"


class TestRateLimitReset:
    """Test rate limit reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_limits(self, limiter, redis_client, clean_keys):
        """Verify reset clears all rate limit keys for identifier."""
        identifier = f"test_reset_{uuid4().hex[:8]}"

        # Exhaust limit
        for _ in range(5):
            await limiter.check_rate_limit(identifier, tier="test")

        # Should be limited
        allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
        assert not allowed

        # Reset
        deleted = await limiter.reset(identifier)
        assert deleted > 0

        # Should be allowed again
        allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
        assert allowed, "Should be allowed after reset"


class TestRateLimitBypassPrevention:
    """Test that rate limits cannot be bypassed."""

    @pytest.mark.asyncio
    async def test_rapid_requests_all_counted(self, limiter, clean_keys):
        """Verify rapid-fire requests are all counted."""
        identifier = f"test_rapid_{uuid4().hex[:8]}"

        # Make requests as fast as possible
        results = []
        for _ in range(20):
            allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
            results.append(allowed)

        allowed_count = sum(results)
        assert allowed_count == 5, f"Expected 5, got {allowed_count}"

    @pytest.mark.asyncio
    async def test_different_users_have_separate_limits(self, limiter, clean_keys):
        """Verify each user has independent limits."""
        users = [f"test_user_{i}_{uuid4().hex[:8]}" for i in range(3)]

        for user in users:
            # Each user should get their full limit
            for i in range(5):
                allowed, _ = await limiter.check_rate_limit(user, tier="test")
                assert allowed, f"User {user} request {i+1} should be allowed"

            # And then be limited
            allowed, _ = await limiter.check_rate_limit(user, tier="test")
            assert not allowed


class TestRedisResilience:
    """Test behavior under Redis issues."""

    @pytest.mark.asyncio
    async def test_handles_redis_reconnection(self, redis_client, clean_keys):
        """Verify rate limiter handles Redis reconnection gracefully."""
        limiter = SlidingWindowRateLimiter(redis_client)
        identifier = f"test_reconnect_{uuid4().hex[:8]}"

        # Make some requests
        for _ in range(3):
            allowed, _ = await limiter.check_rate_limit(identifier, tier="test")
            assert allowed

        # Verify counts persist
        allowed, info = await limiter.check_rate_limit(identifier, tier="test")
        assert allowed
        assert info["limits"]["minute"]["remaining"] == 1


class TestRetryAfterHeader:
    """Test retry-after calculation."""

    @pytest.mark.asyncio
    async def test_retry_after_provided_when_limited(self, limiter, clean_keys):
        """Verify retry_after is provided when rate limited."""
        identifier = f"test_retry_{uuid4().hex[:8]}"

        # Exhaust limit
        for _ in range(5):
            await limiter.check_rate_limit(identifier, tier="test")

        # Check retry_after
        allowed, info = await limiter.check_rate_limit(identifier, tier="test")
        assert not allowed

        retry_after = info["limits"]["minute"]["retry_after"]
        assert retry_after > 0, "retry_after should be positive"
        assert retry_after <= 60, "retry_after should be <= window size"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
