"""Unit tests for Redis-backed rate limiting in check_rate_limit() (Phase 3 Item 3b)."""
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.interface.api.exceptions import RateLimitException


def _make_user(user_id="user1", role="user"):
    user = MagicMock()
    user.id = user_id
    user.role = MagicMock()
    user.role.value = role
    return user


def _make_request(path="/api/v1/query"):
    req = MagicMock()
    req.url.path = path
    return req


def _make_redis(connected=True, pipeline_counts=(0,)):
    """Build a mock RedisClient with pipeline returning given ZCARD values."""
    redis = MagicMock()
    redis.connected = connected

    if connected:
        # pipeline() is synchronous; execute() is a coroutine
        pipe = MagicMock()
        pipe.zremrangebyscore = MagicMock()
        pipe.zcard = MagicMock()
        pipe.execute = AsyncMock(side_effect=[
            [None, count] for count in pipeline_counts
        ])
        redis.redis_client = MagicMock()
        redis.redis_client.pipeline = MagicMock(return_value=pipe)
        redis.redis_client.zadd = AsyncMock(return_value=1)
        redis.redis_client.expire = AsyncMock(return_value=True)
        redis.redis_client.zrange = AsyncMock(return_value=[])
    else:
        redis.redis_client = None

    return redis


class TestRateLimitRedisPath:
    @pytest.mark.asyncio
    async def test_allowed_request_calls_redis_pipeline(self):
        """When Redis is connected and under limit, pipeline is called."""
        from prsm.interface.api.dependencies import check_rate_limit

        redis = _make_redis(connected=True, pipeline_counts=[0])  # 0 prior requests
        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            result = await check_rate_limit(_make_request(), _make_user())

        assert result is not None
        redis.redis_client.pipeline().zremrangebyscore.assert_called()
        redis.redis_client.pipeline().zcard.assert_called()
        redis.redis_client.zadd.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocks_at_limit(self):
        """When Redis returns count >= limit, raises RateLimitException."""
        from prsm.interface.api.dependencies import check_rate_limit

        # user tier "user" has limit=100; simulate 100 existing requests
        redis = _make_redis(connected=True, pipeline_counts=[100])
        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            with pytest.raises(RateLimitException):
                await check_rate_limit(_make_request(), _make_user())

        # zadd must NOT be called when blocked
        redis.redis_client.zadd.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allows_just_under_limit(self):
        """Request at count = limit - 1 is allowed."""
        from prsm.interface.api.dependencies import check_rate_limit

        redis = _make_redis(connected=True, pipeline_counts=[99])  # limit is 100
        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            result = await check_rate_limit(_make_request(), _make_user())

        assert result is not None
        redis.redis_client.zadd.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_key_is_namespaced(self):
        """Redis key follows prsm:rate_limit: prefix."""
        from prsm.interface.api.dependencies import check_rate_limit

        redis = _make_redis(connected=True, pipeline_counts=[0])
        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            await check_rate_limit(_make_request("/api/v1/query"), _make_user("user42"))

        zadd_call = redis.redis_client.zadd.call_args
        key_used = zadd_call[0][0]  # first positional arg
        assert key_used.startswith("prsm:rate_limit:")
        assert "user42" in key_used

    @pytest.mark.asyncio
    async def test_rate_limit_exception_not_swallowed_by_redis_except(self):
        """RateLimitException from the Redis path is re-raised, not treated as Redis error."""
        from prsm.interface.api.dependencies import check_rate_limit

        redis = _make_redis(connected=True, pipeline_counts=[100])  # at limit → raises
        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            with pytest.raises(RateLimitException):
                await check_rate_limit(_make_request(), _make_user())


class TestRateLimitFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_memory_when_redis_disconnected(self):
        """When Redis is not connected, in-memory fallback is used without error."""
        from prsm.interface.api.dependencies import check_rate_limit, _rate_limit_storage

        redis = _make_redis(connected=False)
        _rate_limit_storage.clear()

        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            result = await check_rate_limit(_make_request(), _make_user("user_fallback"))

        assert result is not None
        # In-memory dict should have been populated
        assert any("user_fallback" in k for k in _rate_limit_storage)

    @pytest.mark.asyncio
    async def test_falls_back_when_redis_pipeline_raises(self):
        """When Redis pipeline raises (non-rate-limit error), falls back to in-memory."""
        from prsm.interface.api.dependencies import check_rate_limit, _rate_limit_storage

        redis = MagicMock()
        redis.connected = True
        pipe = MagicMock()
        pipe.zremrangebyscore = MagicMock()
        pipe.zcard = MagicMock()
        pipe.execute = AsyncMock(side_effect=ConnectionError("Redis timeout"))
        redis.redis_client = MagicMock()
        redis.redis_client.pipeline = MagicMock(return_value=pipe)

        _rate_limit_storage.clear()

        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            # Should not raise — falls back to in-memory
            result = await check_rate_limit(_make_request(), _make_user("user_redis_err"))

        assert result is not None

    @pytest.mark.asyncio
    async def test_memory_fallback_blocks_at_limit(self):
        """In-memory fallback correctly enforces limits when Redis is unavailable."""
        from prsm.interface.api.dependencies import check_rate_limit, _rate_limit_storage
        from prsm.interface.api.standards import APIConfig

        redis = _make_redis(connected=False)
        _rate_limit_storage.clear()

        user = _make_user("user_mem_limit")
        req = _make_request()
        tier = "user"
        limit = APIConfig.RATE_LIMITS[tier]["requests"]

        # Pre-fill in-memory storage to simulate being at the limit
        rate_key = f"{user.id}:{req.url.path}:{tier}"
        _rate_limit_storage[rate_key] = {
            "requests": [time.time()] * limit,
            "last_reset": None,
        }

        with patch("prsm.interface.api.dependencies.get_redis_client", return_value=redis):
            with pytest.raises(RateLimitException):
                await check_rate_limit(req, user)
