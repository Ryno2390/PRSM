"""Unit tests for PostQuantumAuthManager DB+Redis persistence (Phase 3 Item 3f)."""
import pytest
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from sqlalchemy import JSON
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.dialects.postgresql import JSONB
from prsm.core.database import Base


@pytest.fixture
async def db_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    # Replace JSONB columns with JSON for SQLite compatibility
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, JSONB):
                column.type = JSON()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _session():
        async with factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    with patch("prsm.core.database.get_async_session", _session):
        yield

    await engine.dispose()


@pytest.fixture
def auth_manager(db_session_factory):
    from prsm.core.auth.post_quantum_auth import PostQuantumAuthManager
    from prsm.core.cryptography.post_quantum import SecurityLevel
    return PostQuantumAuthManager(SecurityLevel.LEVEL_1)


class TestIdentityPersistence:
    @pytest.mark.asyncio
    async def test_create_persists_to_db(self, auth_manager, db_session_factory):
        """create_post_quantum_identity() writes to pq_identities table."""
        from prsm.core.database import get_async_session, PQIdentityModel
        user_id = uuid4()
        identity = await auth_manager.create_post_quantum_identity(user_id)
        assert identity is not None

        async with get_async_session() as db:
            row = await db.get(PQIdentityModel, user_id)
        assert row is not None
        assert row.security_level == identity.security_level.value

    @pytest.mark.asyncio
    async def test_get_identity_loads_from_db(self, auth_manager, db_session_factory):
        """get_identity() hydrates from DB when not in memory."""
        user_id = uuid4()
        await auth_manager.create_post_quantum_identity(user_id)

        # Evict from in-memory cache
        del auth_manager.identities[user_id]

        # Should reload from DB
        recovered = await auth_manager.get_identity(user_id)
        assert recovered is not None
        assert recovered.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_identity_returns_none_for_unknown(self, auth_manager, db_session_factory):
        """get_identity() returns None for a user with no identity."""
        result = await auth_manager.get_identity(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_upgrade_updates_db_row(self, auth_manager, db_session_factory):
        """upgrade_security_level() overwrites DB keypair_json."""
        from prsm.core.database import get_async_session, PQIdentityModel
        from prsm.core.cryptography.post_quantum import SecurityLevel
        user_id = uuid4()
        await auth_manager.create_post_quantum_identity(user_id, SecurityLevel.LEVEL_1)

        result = await auth_manager.upgrade_security_level(user_id, SecurityLevel.LEVEL_3)
        assert result is True

        async with get_async_session() as db:
            row = await db.get(PQIdentityModel, user_id)
        assert row.security_level == SecurityLevel.LEVEL_3.value


class TestChallengePersistence:
    @pytest.mark.asyncio
    async def test_create_challenge_writes_to_redis(self, auth_manager, db_session_factory):
        """create_auth_challenge() calls Redis setex with correct TTL."""
        user_id = uuid4()
        await auth_manager.create_post_quantum_identity(user_id)

        mock_redis = MagicMock()
        mock_redis.connected = True
        mock_redis.redis_client = AsyncMock()
        mock_redis.redis_client.setex = AsyncMock(return_value=True)

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis):
            challenge = await auth_manager.create_auth_challenge(user_id, challenge_lifetime_minutes=5)

        assert challenge is not None
        mock_redis.redis_client.setex.assert_awaited_once()
        call_args = mock_redis.redis_client.setex.call_args[0]
        assert "prsm:auth:challenge:" in call_args[0]
        assert call_args[1] == 300  # 5 * 60

    @pytest.mark.asyncio
    async def test_verify_falls_back_to_redis(self, auth_manager, db_session_factory):
        """verify_auth_signature() reads challenge from Redis when not in memory."""
        user_id = uuid4()
        await auth_manager.create_post_quantum_identity(user_id)
        challenge = await auth_manager.create_auth_challenge(user_id)
        # Evict from in-memory
        del auth_manager.challenges[challenge.challenge_id]

        # Sign the challenge
        signature = await auth_manager.sign_authentication_challenge(
            user_id, challenge.challenge_data
        )

        # Mock Redis returning the challenge
        mock_redis = MagicMock()
        mock_redis.connected = True
        mock_redis.redis_client = AsyncMock()
        mock_redis.redis_client.get = AsyncMock(return_value=json.dumps(challenge.to_dict()))
        mock_redis.redis_client.delete = AsyncMock()

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis):
            is_valid, msg = await auth_manager.verify_auth_signature(
                challenge.challenge_id, signature.to_dict()
            )

        assert is_valid is True, msg

    @pytest.mark.asyncio
    async def test_verify_deletes_redis_challenge_on_success(self, auth_manager, db_session_factory):
        """Successful verification deletes the Redis key."""
        user_id = uuid4()
        await auth_manager.create_post_quantum_identity(user_id)
        challenge = await auth_manager.create_auth_challenge(user_id)
        signature = await auth_manager.sign_authentication_challenge(
            user_id, challenge.challenge_data
        )

        mock_redis = MagicMock()
        mock_redis.connected = True
        mock_redis.redis_client = AsyncMock()
        mock_redis.redis_client.get = AsyncMock(return_value=None)  # not in Redis
        mock_redis.redis_client.delete = AsyncMock()

        with patch("prsm.core.redis_client.get_redis_client", return_value=mock_redis):
            is_valid, _ = await auth_manager.verify_auth_signature(
                challenge.challenge_id, signature.to_dict()
            )

        assert is_valid is True
        mock_redis.redis_client.delete.assert_awaited()
