"""Unit tests for ContextManager Redis + DB persistence (Phase 3 Item 3d)."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from prsm.compute.nwtn.context_manager import ContextManager, ContextUsage

_HISTORY_LIMIT = ContextManager._HISTORY_LIMIT


@pytest.fixture
def cm():
    return ContextManager()


class TestRecordUsagePersistence:
    def test_record_usage_schedules_persist_task(self, cm):
        """record_usage() schedules _persist_usage via create_task."""
        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            cm.record_usage("sess-1", 75, 100)
        mock_loop.create_task.assert_called_once()

    def test_record_usage_returns_usage_synchronously(self, cm):
        """Return value is synchronous — not a coroutine."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            usage = cm.record_usage("sess-2", 50, 100)
        assert isinstance(usage, ContextUsage)
        assert usage.context_used == 50

    def test_record_usage_no_crash_without_event_loop(self, cm):
        """RuntimeError from get_running_loop is silently swallowed."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            usage = cm.record_usage("sess-3", 50, 100)  # must not raise
        assert usage is not None

    def test_history_cap_respected(self, cm):
        """usage_history is capped at _HISTORY_LIMIT entries."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            for i in range(_HISTORY_LIMIT + 10):
                cm.record_usage(f"sess-{i}", 50, 100)
        assert len(cm.usage_history) == _HISTORY_LIMIT


class TestPersistUsageHelper:
    @pytest.mark.asyncio
    async def test_persist_usage_writes_to_redis(self, cm):
        """_persist_usage() calls Redis setex with correct key and TTL."""
        usage = ContextUsage("sess-x", 60, 100)
        mock_redis = MagicMock()
        mock_redis.connected = True
        mock_redis.redis_client = AsyncMock()
        mock_redis.redis_client.setex = AsyncMock(return_value=True)

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis), \
             patch("prsm.core.database.get_async_session") as mock_ctx:
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock()
            mock_db.commit = AsyncMock()
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await cm._persist_usage("sess-x", usage)

        mock_redis.redis_client.setex.assert_awaited_once()
        key_arg = mock_redis.redis_client.setex.call_args[0][0]
        assert "prsm:context:usage:sess-x" == key_arg
        ttl_arg = mock_redis.redis_client.setex.call_args[0][1]
        assert ttl_arg == 86400

    @pytest.mark.asyncio
    async def test_persist_usage_updates_db(self, cm):
        """_persist_usage() issues UPDATE on PRSMSessionModel."""
        from uuid import uuid4
        usage = ContextUsage(str(uuid4()), 60, 100)
        mock_redis = MagicMock()
        mock_redis.connected = False

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis), \
             patch("prsm.core.database.get_async_session") as mock_ctx:
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock()
            mock_db.commit = AsyncMock()
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await cm._persist_usage(usage.session_id, usage)

        mock_db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persist_usage_redis_failure_does_not_raise(self, cm):
        """Redis errors in _persist_usage are swallowed (non-critical)."""
        usage = ContextUsage("sess-fail", 50, 100)
        mock_redis = MagicMock()
        mock_redis.connected = True
        mock_redis.redis_client = AsyncMock()
        mock_redis.redis_client.setex = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis), \
             patch("prsm.core.database.get_async_session") as mock_ctx:
            mock_db = AsyncMock()
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await cm._persist_usage("sess-fail", usage)  # must not raise


class TestGetSessionUsage:
    @pytest.mark.asyncio
    async def test_returns_from_memory_first(self, cm):
        """In-memory hit skips Redis entirely."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            cm.record_usage("sess-mem", 80, 100)
        result = await cm.get_session_usage("sess-mem")
        assert result is not None
        assert result.context_used == 80

    @pytest.mark.asyncio
    async def test_falls_back_to_redis(self, cm):
        """When not in memory, reads from Redis and hydrates in-memory."""
        payload = json.dumps({
            "session_id": "sess-redis",
            "context_used": 70,
            "context_allocated": 100,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        mock_redis = MagicMock()
        mock_redis.connected = True
        mock_redis.redis_client = AsyncMock()
        mock_redis.redis_client.get = AsyncMock(return_value=payload)

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis):
            result = await cm.get_session_usage("sess-redis")

        assert result is not None
        assert result.context_used == 70
        assert "sess-redis" in cm.session_usage  # hydrated

    @pytest.mark.asyncio
    async def test_returns_none_when_nowhere(self, cm):
        """Returns None when neither memory nor Redis has the session."""
        mock_redis = MagicMock()
        mock_redis.connected = True
        mock_redis.redis_client = AsyncMock()
        mock_redis.redis_client.get = AsyncMock(return_value=None)

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis):
            result = await cm.get_session_usage("sess-missing")

        assert result is None


class TestOptimizeContextAllocation:
    @pytest.mark.asyncio
    async def test_hydrates_from_db_when_history_empty(self, cm):
        """optimize_context_allocation calls _load_history_from_db when history is empty."""
        db_history = [
            {"session_id": "s1", "used": 80, "allocated": 100,
             "efficiency": 0.8, "timestamp": ""},
            {"session_id": "s2", "used": 50, "allocated": 100,
             "efficiency": 0.5, "timestamp": ""},
        ]
        with patch.object(cm, "_load_history_from_db", AsyncMock(return_value=db_history)):
            result = await cm.optimize_context_allocation()

        assert result["sample_size"] == 2
        assert 0 < result["avg_efficiency"] < 1

    @pytest.mark.asyncio
    async def test_uses_in_memory_history_when_available(self, cm):
        """Does not call _load_history_from_db when history is populated."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            cm.record_usage("s1", 80, 100)

        with patch.object(cm, "_load_history_from_db",
                          AsyncMock(return_value=[])) as mock_db_load:
            await cm.optimize_context_allocation()

        mock_db_load.assert_not_awaited()


class TestClearSession:
    def test_clear_session_schedules_redis_delete(self, cm):
        """clear_session() schedules Redis key deletion."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            cm.record_usage("sess-clear", 50, 100)

        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            cm.clear_session("sess-clear")

        mock_loop.create_task.assert_called_once()
