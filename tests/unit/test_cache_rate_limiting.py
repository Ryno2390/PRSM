"""
Unit tests for cache rate limiting functionality.

Tests the CacheRateLimiter class and pattern-based cache invalidation
without requiring Redis or external dependencies.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.core.caching.cache_manager import (
    CacheRateLimiter,
    get_cache_rate_limiter,
    CacheManager,
)
from prsm.core.caching.cache_strategies import MemoryCache
from prsm.core.caching.decorators import api_cache


class TestCacheRateLimiter:
    """Tests for the CacheRateLimiter class."""

    def test_rate_limiter_allows_calls_within_limit(self):
        """Test that rate limiter allows calls within the limit."""
        limiter = CacheRateLimiter(max_calls=10, window_seconds=60)
        
        # Call is_allowed 5 times - all should return True
        results = [limiter.is_allowed('test_key') for _ in range(5)]
        
        assert all(results), "All 5 calls within limit should return True"

    def test_rate_limiter_blocks_when_limit_exceeded(self):
        """Test that rate limiter blocks calls when limit is exceeded."""
        limiter = CacheRateLimiter(max_calls=10, window_seconds=60)
        
        # Call is_allowed 11 times
        results = [limiter.is_allowed('test_key') for _ in range(11)]
        
        # First 10 should return True
        assert all(results[:10]), "First 10 calls should return True"
        # 11th should return False
        assert results[10] is False, "11th call should return False (limit exceeded)"

    def test_rate_limiter_resets_after_window(self):
        """Test that rate limiter resets after the window expires."""
        # Use a mock to simulate time passing for reliability
        with patch('prsm.core.caching.cache_manager.time.monotonic') as mock_time:
            # Start at time 0
            mock_time.return_value = 0.0
            
            limiter = CacheRateLimiter(max_calls=3, window_seconds=1)
            
            # Exhaust the limit at time 0
            results = [limiter.is_allowed('test_key') for _ in range(3)]
            assert all(results), "First 3 calls should return True"
            
            # Next call should be blocked (still at time 0)
            assert limiter.is_allowed('test_key') is False, "4th call should be blocked"
            
            # Advance time past the window (1 second window + margin)
            mock_time.return_value = 1.5
            
            # Should be allowed again after window expires
            assert limiter.is_allowed('test_key') is True, "Call after window expiry should return True"

    def test_rate_limiter_get_remaining_decrements(self):
        """Test that get_remaining returns correct remaining calls."""
        limiter = CacheRateLimiter(max_calls=10, window_seconds=60)
        
        # Make 3 calls
        for _ in range(3):
            limiter.is_allowed('test_key')
        
        # Should have 7 remaining
        remaining = limiter.get_remaining('test_key')
        assert remaining == 7, f"Expected 7 remaining, got {remaining}"

    def test_rate_limiter_reset_clears_state(self):
        """Test that reset clears the rate limit state for a key."""
        limiter = CacheRateLimiter(max_calls=3, window_seconds=60)
        
        # Exhaust the limit
        for _ in range(3):
            limiter.is_allowed('test_key')
        
        # Should be blocked now
        assert limiter.is_allowed('test_key') is False, "Should be blocked after exhausting limit"
        
        # Reset the key
        limiter.reset('test_key')
        
        # Should be allowed again
        assert limiter.is_allowed('test_key') is True, "Should be allowed after reset"

    def test_rate_limiter_separate_keys_are_independent(self):
        """Test that rate limits for different keys are independent."""
        limiter = CacheRateLimiter(max_calls=2, window_seconds=60)
        
        # Exhaust key_a
        assert limiter.is_allowed('key_a') is True
        assert limiter.is_allowed('key_a') is True
        
        # key_a should be blocked
        assert limiter.is_allowed('key_a') is False, "key_a should be blocked"
        
        # key_b should still be allowed (independent)
        assert limiter.is_allowed('key_b') is True, "key_b should be allowed (independent key)"


class TestPatternInvalidation:
    """Tests for pattern-based cache invalidation."""

    @pytest.mark.asyncio
    async def test_pattern_invalidation_matches_glob(self):
        """Test that pattern invalidation matches glob patterns correctly."""
        # Create a MemoryCache instance
        memory_cache = MemoryCache(name="test_cache", max_size_bytes=1024 * 1024)
        
        # Add test keys
        memory_cache.set("user_1", {"name": "Alice"})
        memory_cache.set("user_2", {"name": "Bob"})
        memory_cache.set("session_3", {"token": "abc123"})
        
        # Create a CacheManager and register the memory cache
        cache_manager = CacheManager()
        cache_manager._caches["test_cache"] = memory_cache
        
        # Invalidate keys matching "user_*"
        invalidated = await cache_manager.invalidate_by_pattern("test_cache", "user_*")
        
        assert invalidated == 2, f"Expected 2 keys invalidated, got {invalidated}"
        assert memory_cache.get("user_1") is None, "user_1 should be deleted"
        assert memory_cache.get("user_2") is None, "user_2 should be deleted"
        assert memory_cache.get("session_3") is not None, "session_3 should remain"

    @pytest.mark.asyncio
    async def test_pattern_invalidation_question_mark_wildcard(self):
        """Test that pattern invalidation handles ? wildcard correctly."""
        # Create a MemoryCache instance
        memory_cache = MemoryCache(name="test_cache", max_size_bytes=1024 * 1024)
        
        # Add test keys
        memory_cache.set("abc", "value_abc")
        memory_cache.set("aXc", "value_aXc")
        memory_cache.set("aXXc", "value_aXXc")
        
        # Create a CacheManager and register the memory cache
        cache_manager = CacheManager()
        cache_manager._caches["test_cache"] = memory_cache
        
        # Invalidate keys matching "a?c" (? matches exactly one character)
        invalidated = await cache_manager.invalidate_by_pattern("test_cache", "a?c")
        
        assert invalidated == 2, f"Expected 2 keys invalidated, got {invalidated}"
        assert memory_cache.get("abc") is None, "abc should be deleted"
        assert memory_cache.get("aXc") is None, "aXc should be deleted"
        assert memory_cache.get("aXXc") is not None, "aXXc should remain (doesn't match a?c)"

    @pytest.mark.asyncio
    async def test_invalidate_routes_glob_to_pattern(self):
        """Test that invalidate() auto-detects glob patterns and routes to invalidate_by_pattern."""
        # Create a MemoryCache instance
        memory_cache = MemoryCache(name="test_cache", max_size_bytes=1024 * 1024)
        
        # Add test keys
        memory_cache.set("key_1", "value_1")
        memory_cache.set("key_2", "value_2")
        memory_cache.set("other", "value_other")
        
        # Create a CacheManager and register the memory cache
        cache_manager = CacheManager()
        cache_manager._caches["test_cache"] = memory_cache
        
        # Call invalidate with a glob pattern (not invalidate_by_pattern)
        result = await cache_manager.invalidate("test_cache", "key_*")
        
        assert result is True, "invalidate should return True"
        assert memory_cache.get("key_1") is None, "key_1 should be deleted"
        assert memory_cache.get("key_2") is None, "key_2 should be deleted"
        assert memory_cache.get("other") is not None, "other should remain"


class TestAPICacheRateLimiting:
    """Tests for the api_cache decorator rate limiting."""

    @pytest.mark.asyncio
    async def test_api_cache_rate_limit_blocks_and_serves_stale(self):
        """Test that rate-limited api_cache serves stale cache when rate limited."""
        # Create a fresh CacheManager and MemoryCache for isolation
        memory_cache = MemoryCache(name="api_response", max_size_bytes=1024 * 1024)
        
        # We need to mock the get_cache_manager to return our test cache manager
        # and also mock the rate limiter
        test_cache_manager = CacheManager()
        test_cache_manager._caches["api_response"] = memory_cache
        
        # Create a mock rate limiter that will return False (rate limited)
        mock_limiter = MagicMock()
        mock_limiter.is_allowed = MagicMock(return_value=False)
        
        # Track whether the underlying function was called
        function_called = False
        
        # Define a test function decorated with api_cache
        # Use include_auth=True to avoid exclude_args bug (strings vs integers)
        @api_cache(
            ttl_seconds=3600,
            include_auth=True,
            rate_limit_key='test_rate_limit',
            rate_limit_max_calls=1
        )
        async def test_api_function(query: str) -> str:
            nonlocal function_called
            function_called = True
            return f"fresh_result_for_{query}"
        
        # Pre-populate the cache with a known value
        # We need to compute the cache key the same way the decorator does
        from prsm.core.caching.decorators import _generate_cache_key
        cache_key = _generate_cache_key(
            test_api_function, ('test_query',), {}, "api_call", True, True, []
        )
        await test_cache_manager.set("api_response", cache_key, "cached_result")
        
        # Patch both get_cache_manager and get_cache_rate_limiter
        with patch('prsm.core.caching.decorators.get_cache_manager', return_value=test_cache_manager), \
             patch('prsm.core.caching.cache_manager.get_cache_rate_limiter', return_value=mock_limiter):
            
            # Call the decorated function
            result = await test_api_function("test_query")
            
            # Should return cached value without calling the underlying function
            assert result == "cached_result", f"Expected cached_result, got {result}"
            assert not function_called, "Underlying function should not be called when rate limited and cache exists"


class TestCacheRateLimiterSingleton:
    """Tests for the get_cache_rate_limiter singleton function."""

    def test_get_cache_rate_limiter_returns_singleton(self):
        """Test that get_cache_rate_limiter returns the same instance."""
        # Reset the global singleton for this test
        import prsm.core.caching.cache_manager as cm
        cm._cache_rate_limiter = None
        
        limiter1 = get_cache_rate_limiter()
        limiter2 = get_cache_rate_limiter()
        
        assert limiter1 is limiter2, "Should return the same singleton instance"

    def test_get_cache_rate_limiter_uses_default_params(self):
        """Test that get_cache_rate_limiter uses default parameters."""
        # Reset the global singleton for this test
        import prsm.core.caching.cache_manager as cm
        cm._cache_rate_limiter = None
        
        limiter = get_cache_rate_limiter()
        
        assert limiter.max_calls == 60, f"Expected max_calls=60, got {limiter.max_calls}"
        assert limiter.window_seconds == 60, f"Expected window_seconds=60, got {limiter.window_seconds}"

    def test_get_cache_rate_limiter_custom_params_on_first_call(self):
        """Test that custom params are used when creating new singleton."""
        # Reset the global singleton for this test
        import prsm.core.caching.cache_manager as cm
        cm._cache_rate_limiter = None
        
        limiter = get_cache_rate_limiter(max_calls=100, window_seconds=120)
        
        assert limiter.max_calls == 100, f"Expected max_calls=100, got {limiter.max_calls}"
        assert limiter.window_seconds == 120, f"Expected window_seconds=120, got {limiter.window_seconds}"
