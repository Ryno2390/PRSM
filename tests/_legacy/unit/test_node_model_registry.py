"""Unit tests for _NodeModelRegistryAdapter DB-backed discovery (Phase 3 Item 3c)."""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch
from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text as sa_text
from sqlalchemy.pool import StaticPool


@pytest.fixture
async def db_session_factory():
    """SQLite in-memory DB with teacher_models table matching TeacherModelModel schema."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    # Create teacher_models table with all columns from TeacherModelModel
    async with engine.begin() as conn:
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS teacher_models (
                teacher_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                specialization TEXT NOT NULL DEFAULT 'general',
                model_type TEXT DEFAULT 'teacher',
                performance_score REAL DEFAULT 0.0,
                curriculum_ids TEXT DEFAULT '[]',
                student_models TEXT DEFAULT '[]',
                rlvr_score REAL,
                ipfs_cid TEXT,
                version TEXT DEFAULT '1.0.0',
                active INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT
            )
        """))
        await conn.commit()
    
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _session():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    with patch("prsm.core.database.get_async_session", _session):
        yield

    await engine.dispose()


@pytest.fixture
async def adapter(db_session_factory):
    from prsm.node.node import _NodeModelRegistryAdapter
    return _NodeModelRegistryAdapter()


@pytest.fixture
async def seeded_db(db_session_factory):
    """Seed the DB with known teacher models for discovery tests."""
    from prsm.core.database import get_async_session

    models = [
        {"teacher_id": str(uuid4()), "name": "ML Expert",
         "specialization": "machine_learning", "model_type": "teacher", 
         "performance_score": 0.92, "active": 1,
         "created_at": datetime.now(timezone.utc).isoformat()},
        {"teacher_id": str(uuid4()), "name": "General Helper",
         "specialization": "general", "model_type": "teacher",
         "performance_score": 0.75, "active": 1,
         "created_at": datetime.now(timezone.utc).isoformat()},
        {"teacher_id": str(uuid4()), "name": "Code Assistant",
         "specialization": "code_generation", "model_type": "teacher",
         "performance_score": 0.88, "active": 1,
         "created_at": datetime.now(timezone.utc).isoformat()},
        {"teacher_id": str(uuid4()), "name": "Retired Model",
         "specialization": "machine_learning", "model_type": "teacher",
         "performance_score": 0.60, "active": 0,
         "created_at": datetime.now(timezone.utc).isoformat()},
    ]
    async with get_async_session() as db:
        for m in models:
            await db.execute(
                sa_text("""
                    INSERT INTO teacher_models (teacher_id, name, specialization, model_type, performance_score, active, created_at)
                    VALUES (:teacher_id, :name, :specialization, :model_type, :performance_score, :active, :created_at)
                """),
                m
            )
        await db.commit()


class TestDiscoverSpecialists:
    @pytest.mark.asyncio
    async def test_returns_domain_matches(self, adapter, seeded_db):
        """Returns models whose specialization contains the domain string."""
        results = await adapter.discover_specialists("machine_learning")
        names = [r.name for r in results]
        assert "ML Expert" in names

    @pytest.mark.asyncio
    async def test_always_includes_general(self, adapter, seeded_db):
        """General-purpose models appear in all domain queries."""
        results = await adapter.discover_specialists("legal")
        names = [r.name for r in results]
        assert "General Helper" in names

    @pytest.mark.asyncio
    async def test_excludes_inactive_models(self, adapter, seeded_db):
        """Active=False models are never returned."""
        results = await adapter.discover_specialists("machine_learning")
        names = [r.name for r in results]
        assert "Retired Model" not in names

    @pytest.mark.asyncio
    async def test_ordered_by_performance_desc(self, adapter, seeded_db):
        """Results are ordered by performance_score descending."""
        results = await adapter.discover_specialists("machine_learning")
        scores = [r.performance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_returns_teacher_model_objects(self, adapter, seeded_db):
        """Return type is List[TeacherModel], not dicts."""
        from prsm.core.models import TeacherModel
        results = await adapter.discover_specialists("general")
        assert all(isinstance(r, TeacherModel) for r in results)

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_list(self, adapter, db_session_factory):
        """Empty DB returns empty list, not the old hardcoded fallback."""
        results = await adapter.discover_specialists("general")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_hardcoded_general_helper(self, adapter, db_session_factory):
        """The old hardcoded TeacherModel must never appear from an empty DB."""
        results = await adapter.discover_specialists("general")
        names = [r.name for r in results]
        assert "General Helper" not in names


class TestRegisterTeacherModel:
    @pytest.mark.asyncio
    async def test_register_persists_to_db(self, adapter, db_session_factory):
        """register_teacher_model writes a row to teacher_models."""
        from prsm.core.models import TeacherModel
        from prsm.core.database import get_async_session

        model = TeacherModel(name="New Specialist", specialization="physics",
                             performance_score=0.95)
        success = await adapter.register_teacher_model(model, "QmTestCID")
        assert success is True

        # Verify it's now discoverable
        async with get_async_session() as db:
            result = await db.execute(
                sa_text("SELECT * FROM teacher_models WHERE name = 'New Specialist'")
            )
            row = result.fetchone()
        assert row is not None
        assert row.ipfs_cid == "QmTestCID"
        assert row.specialization == "physics"

    @pytest.mark.asyncio
    async def test_register_then_discover(self, adapter, db_session_factory):
        """A registered model is immediately discoverable."""
        from prsm.core.models import TeacherModel

        model = TeacherModel(name="Registered Model", specialization="quantum_physics",
                             performance_score=0.90)
        await adapter.register_teacher_model(model, "QmQuantumCID")

        results = await adapter.discover_specialists("quantum")
        names = [r.name for r in results]
        assert "Registered Model" in names

    @pytest.mark.asyncio
    async def test_register_duplicate_is_idempotent(self, adapter, db_session_factory):
        """Registering the same name+version twice returns True (no crash)."""
        from prsm.core.models import TeacherModel

        model = TeacherModel(name="Dup Model", specialization="math",
                             performance_score=0.80)
        first = await adapter.register_teacher_model(model, "QmFirst")
        second = await adapter.register_teacher_model(model, "QmSecond")
        assert first is True
        assert second is True
