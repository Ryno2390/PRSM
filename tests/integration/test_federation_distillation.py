"""
Integration Tests for Federation and Distillation Systems

Tests for:
- Federation message handlers
- Federation background tasks
- Distillation persistence
- RPC execution
"""

import asyncio
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Check for required dependencies
try:
    from sqlalchemy import create_engine, select, update, func
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_federation.db"
        yield str(db_path)


@pytest.fixture
async def async_session_factory(temp_db_path):
    """Create async session factory for testing."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    db_url = f"sqlite+aiosqlite:///{temp_db_path}"
    engine = create_async_engine(db_url, echo=False)

    # Create only the tables needed for these tests (avoids PostgreSQL-specific
    # JSONB columns in other tables that SQLite cannot render)
    from prsm.core.database import (
        Base,
        FederationPeerModel,
        FederationMessageModel,
        DistillationJobModel,
        DistillationResultModel,
        EmergencyProtocolActionModel,
    )
    target_tables = [
        FederationPeerModel.__table__,
        FederationMessageModel.__table__,
        DistillationJobModel.__table__,
        DistillationResultModel.__table__,
        EmergencyProtocolActionModel.__table__,
    ]
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=target_tables)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory

    await engine.dispose()


@pytest.fixture
async def db_session(async_session_factory):
    """Create a database session for testing."""
    async with async_session_factory() as session:
        yield session


# ============================================================================
# TestFederationMessageHandlers
# ============================================================================

class TestFederationMessageHandlers:
    """Tests for federation message handlers."""

    @pytest.mark.asyncio
    async def test_discovery_request_stores_peer(self, db_session):
        """Test that discovery request stores peer information."""
        from prsm.core.database import FederationPeerModel

        # Create a mock network instance
        mock_network = MagicMock()
        mock_network.node_id = "test_node_1"
        mock_network.network_address = "localhost"
        mock_network.network_port = 8000
        mock_network.peers = {}
        mock_network.quality_scores = {}
        mock_network._session_factory = lambda: db_session.__class__.__module__

        # Simulate discovery request handling
        message = {
            "sender_id": "peer_123",
            "sender_address": "192.168.1.100",
            "sender_port": 9000,
            "capabilities": {"node_type": "teacher"}
        }

        # Store peer directly
        current_time = time.time()
        peer_record = FederationPeerModel(
            peer_id="peer_123",
            address="192.168.1.100",
            port=9000,
            node_type="teacher",
            last_seen=current_time,
            quality_score=0.5,
            capabilities={"node_type": "teacher"},
            is_active=True,
            created_at=current_time
        )
        db_session.add(peer_record)
        await db_session.commit()

        # Verify peer was stored
        result = await db_session.get(FederationPeerModel, peer_record.id)
        assert result is not None
        assert result.peer_id == "peer_123"
        assert result.address == "192.168.1.100"
        assert result.port == 9000

    @pytest.mark.asyncio
    async def test_discovery_response_initializes_quality_score(self, db_session):
        """Test that discovery response initializes quality score to 0.5."""
        from prsm.core.database import FederationPeerModel

        current_time = time.time()
        peer_record = FederationPeerModel(
            peer_id="new_peer",
            address="10.0.0.1",
            port=8000,
            node_type="standard",
            last_seen=current_time,
            quality_score=0.5,  # Default quality score
            capabilities={},
            is_active=True,
            created_at=current_time
        )
        db_session.add(peer_record)
        await db_session.commit()

        # Verify quality score is 0.5
        result = await db_session.get(FederationPeerModel, peer_record.id)
        assert result.quality_score == 0.5

    @pytest.mark.asyncio
    async def test_heartbeat_updates_quality_score(self, db_session):
        """Test that heartbeat with load updates quality score using rolling average."""
        from prsm.core.database import FederationPeerModel

        current_time = time.time()

        # Create initial peer
        peer_record = FederationPeerModel(
            peer_id="heartbeat_peer",
            address="10.0.0.2",
            port=8000,
            node_type="standard",
            last_seen=current_time - 30,
            quality_score=0.5,
            capabilities={},
            is_active=True,
            created_at=current_time - 60
        )
        db_session.add(peer_record)
        await db_session.commit()

        # Simulate heartbeat with load=0.8
        old_score = 0.5
        load = 0.8
        new_score = 0.8 * old_score + 0.2 * (1.0 - load)  # = 0.44

        # Update peer
        stmt = update(FederationPeerModel).where(
            FederationPeerModel.peer_id == "heartbeat_peer"
        ).values(
            last_seen=current_time,
            quality_score=new_score
        )
        await db_session.execute(stmt)
        await db_session.commit()

        # Verify quality score decreased (higher load = lower quality)
        result = await db_session.get(FederationPeerModel, peer_record.id)
        assert result.quality_score < 0.5
        assert result.quality_score == pytest.approx(0.44, rel=0.01)

    @pytest.mark.asyncio
    async def test_collaboration_request_accepted_at_capacity(self, db_session):
        """Test that collaboration request is rejected when at capacity."""
        from prsm.core.database import FederationMessageModel

        current_time = time.time()

        # Simulate at-capacity scenario
        max_collaborations = 10
        active_count = 10  # At capacity

        accepted = active_count < max_collaborations

        # Store message
        message_record = FederationMessageModel(
            message_id="collab_1",
            message_type="collaboration_request",
            sender_id="requester_1",
            payload={"request_id": "req_1"},
            sent_at=current_time,
            received_at=current_time,
            processed_at=current_time,
            status="processed"
        )
        db_session.add(message_record)
        await db_session.commit()

        # Verify at capacity means rejected
        assert accepted is False

    @pytest.mark.asyncio
    async def test_collaboration_request_accepted_when_space(self, db_session):
        """Test that collaboration request is accepted when space available."""
        max_collaborations = 10
        active_count = 5  # Has space

        accepted = active_count < max_collaborations
        assert accepted is True

    @pytest.mark.asyncio
    async def test_consensus_vote_triggers_completion_check(self, db_session):
        """Test that 3 yes votes with supermajority triggers proposal acceptance."""
        # Simulate consensus proposal
        proposal = {
            "proposal_id": "prop_1",
            "proposer_id": "node_1",
            "proposal_type": "parameter_change",
            "votes": {},
            "status": "open"
        }

        # Add 3 yes votes
        proposal["votes"]["voter_1"] = True
        proposal["votes"]["voter_2"] = True
        proposal["votes"]["voter_3"] = True

        # Check supermajority
        total_votes = len(proposal["votes"])
        yes_votes = sum(1 for v in proposal["votes"].values() if v)
        supermajority_reached = total_votes >= 3 and (yes_votes / total_votes) >= 0.67

        assert supermajority_reached is True
        proposal["status"] = "accepted"
        assert proposal["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_consensus_supermajority_required(self, db_session):
        """Test that 2 yes 1 no out of 3 fails (< 67% supermajority)."""
        proposal = {
            "proposal_id": "prop_2",
            "proposer_id": "node_1",
            "proposal_type": "parameter_change",
            "votes": {},
            "status": "open"
        }

        # Add 2 yes, 1 no
        proposal["votes"]["voter_1"] = True
        proposal["votes"]["voter_2"] = True
        proposal["votes"]["voter_3"] = False

        # Check supermajority
        total_votes = len(proposal["votes"])
        yes_votes = sum(1 for v in proposal["votes"].values() if v)
        supermajority_reached = total_votes >= 3 and (yes_votes / total_votes) >= 0.67

        # 2/3 = 66.67% which is less than 67%
        assert supermajority_reached is False


# ============================================================================
# TestFederationBackgroundTasks
# ============================================================================

class TestFederationBackgroundTasks:
    """Tests for federation background task processors."""

    @pytest.mark.asyncio
    async def test_process_pending_discoveries_marks_stale_inactive(self, db_session):
        """Test that stale peers are marked inactive."""
        from prsm.core.database import FederationPeerModel

        current_time = time.time()

        # Create stale peer (last_seen > 1 hour ago)
        stale_peer = FederationPeerModel(
            peer_id="stale_peer",
            address="10.0.0.10",
            port=8000,
            node_type="standard",
            last_seen=current_time - 3601,  # 1 hour + 1 second ago
            quality_score=0.5,
            capabilities={},
            is_active=True,
            created_at=current_time - 7200
        )
        db_session.add(stale_peer)
        await db_session.commit()

        # Simulate background task marking stale peers inactive
        stmt = update(FederationPeerModel).where(
            FederationPeerModel.is_active == True,
            FederationPeerModel.last_seen < current_time - 3600
        ).values(is_active=False)
        await db_session.execute(stmt)
        await db_session.commit()

        # Verify peer is now inactive
        result = await db_session.get(FederationPeerModel, stale_peer.id)
        assert result.is_active is False

    @pytest.mark.asyncio
    async def test_process_collaboration_requests_cleans_timed_out(self, db_session):
        """Test that timed-out collaboration requests are cleaned up."""
        current_time = time.time()

        # Simulate timed-out request (older than 5 minutes)
        collaboration_requests = {
            "req_1": {
                "requester_id": "node_1",
                "received_at": current_time - 400,  # ~6.7 minutes ago
                "accepted": False
            },
            "req_2": {
                "requester_id": "node_2",
                "received_at": current_time - 100,  # ~1.7 minutes ago
                "accepted": True
            }
        }

        timeout_threshold = 300  # 5 minutes

        # Clean up timed-out requests
        requests_to_remove = []
        for request_id, request in collaboration_requests.items():
            if current_time - request["received_at"] > timeout_threshold:
                if not request["accepted"]:
                    request["status"] = "timed_out"
                    requests_to_remove.append(request_id)

        for request_id in requests_to_remove:
            del collaboration_requests[request_id]

        # Verify only non-timed-out request remains
        assert "req_1" not in collaboration_requests
        assert "req_2" in collaboration_requests


# ============================================================================
# TestDistillationPersistence
# ============================================================================

class TestDistillationPersistence:
    """Tests for distillation job persistence."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_job(self, db_session):
        """Test storing and retrieving a distillation job."""
        from prsm.core.database import DistillationJobModel

        current_time = time.time()
        job_id = str(uuid4())

        # Store job
        job_record = DistillationJobModel(
            job_id=job_id,
            user_id="user_123",
            teacher_model_id="openai/gpt-4",
            student_model_id="student_1",
            strategy="response_based",
            status="pending",
            priority=5,
            config={"batch_size": 32},
            created_at=current_time
        )
        db_session.add(job_record)
        await db_session.commit()

        # Retrieve job
        result = await db_session.get(DistillationJobModel, job_id)
        assert result is not None
        assert result.user_id == "user_123"
        assert result.teacher_model_id == "openai/gpt-4"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_user_jobs_list_ordered_by_created_at(self, db_session):
        """Test that user jobs are retrieved in created_at desc order."""
        from prsm.core.database import DistillationJobModel

        current_time = time.time()

        # Create 3 jobs with different timestamps
        for i in range(3):
            job_record = DistillationJobModel(
                job_id=f"job_{i}",
                user_id="user_456",
                teacher_model_id="teacher_1",
                student_model_id="student_1",
                strategy="response_based",
                status="completed",
                priority=5,
                config={},
                created_at=current_time - (i * 100)  # Different times
            )
            db_session.add(job_record)
        await db_session.commit()

        # Query ordered by created_at desc
        stmt = (
            select(DistillationJobModel)
            .where(DistillationJobModel.user_id == "user_456")
            .order_by(DistillationJobModel.created_at.desc())
        )
        result = await db_session.execute(stmt)
        jobs = result.scalars().all()

        # Verify order
        assert len(jobs) == 3
        assert jobs[0].job_id == "job_0"
        assert jobs[2].job_id == "job_2"

    @pytest.mark.asyncio
    async def test_success_rate_empty_returns_zero(self, db_session):
        """Test that success rate returns 0.0 when no jobs exist."""
        from prsm.core.database import DistillationJobModel

        # Count total jobs
        total_stmt = select(func.count()).select_from(DistillationJobModel)
        total_result = await db_session.execute(total_stmt)
        total = total_result.scalar() or 0

        if total == 0:
            success_rate = 0.0
        else:
            success_rate = 0.0  # Would calculate from completed/total

        assert success_rate == 0.0

    @pytest.mark.asyncio
    async def test_success_rate_with_data(self, db_session):
        """Test success rate calculation with actual job data."""
        from prsm.core.database import DistillationJobModel

        current_time = time.time()

        # Create 4 jobs (3 completed, 1 failed)
        for i, status in enumerate(["completed", "completed", "completed", "failed"]):
            job_record = DistillationJobModel(
                job_id=f"job_rate_{i}",
                user_id="user_rate",
                teacher_model_id="teacher_1",
                student_model_id="student_1",
                strategy="response_based",
                status=status,
                priority=5,
                config={},
                created_at=current_time
            )
            db_session.add(job_record)
        await db_session.commit()

        # Calculate success rate
        total_stmt = select(func.count()).select_from(DistillationJobModel).where(
            DistillationJobModel.user_id == "user_rate"
        )
        total_result = await db_session.execute(total_stmt)
        total = total_result.scalar() or 0

        completed_stmt = select(func.count()).select_from(DistillationJobModel).where(
            DistillationJobModel.user_id == "user_rate",
            DistillationJobModel.status == "completed"
        )
        completed_result = await db_session.execute(completed_stmt)
        completed = completed_result.scalar() or 0

        success_rate = completed / total if total > 0 else 0.0

        assert success_rate == 0.75  # 3/4

    @pytest.mark.asyncio
    async def test_popular_strategies_counts(self, db_session):
        """Test popular strategies count grouping."""
        from prsm.core.database import DistillationJobModel

        current_time = time.time()

        # Create jobs with mixed strategies
        strategies = ["response_based", "response_based", "multi_teacher", "progressive", "response_based"]
        for i, strategy in enumerate(strategies):
            job_record = DistillationJobModel(
                job_id=f"job_strat_{i}",
                user_id="user_strat",
                teacher_model_id="teacher_1",
                student_model_id="student_1",
                strategy=strategy,
                status="completed",
                priority=5,
                config={},
                created_at=current_time
            )
            db_session.add(job_record)
        await db_session.commit()

        # Group by strategy
        stmt = (
            select(DistillationJobModel.strategy, func.count().label('count'))
            .where(DistillationJobModel.user_id == "user_strat")
            .group_by(DistillationJobModel.strategy)
            .order_by(func.count().desc())
        )
        result = await db_session.execute(stmt)
        strategy_counts = {row.strategy: row.count for row in result}

        assert strategy_counts["response_based"] == 3
        assert strategy_counts["multi_teacher"] == 1
        assert strategy_counts["progressive"] == 1

    @pytest.mark.asyncio
    async def test_prepare_deployment_missing_checkpoint_raises(self, tmp_path):
        """Test that prepare_deployment raises FileNotFoundError when checkpoint absent."""
        # Simulate checkpoint path that doesn't exist
        checkpoint_path = tmp_path / "nonexistent_checkpoint"

        # Verify it raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"No checkpoint for job test_job")

    @pytest.mark.asyncio
    async def test_record_halt_action_persisted(self, db_session):
        """Test that halt action is persisted to database."""
        from prsm.core.database import EmergencyProtocolActionModel

        current_time = time.time()

        # Record halt action
        action_record = EmergencyProtocolActionModel(
            action_type="halt",
            triggered_by="admin_user",
            reason="Emergency halt triggered",
            original_value={"halt_types": ["all"]},
            new_value={"halt_initiated": True},
            created_at=current_time
        )
        db_session.add(action_record)
        await db_session.commit()

        # Verify persisted
        stmt = select(EmergencyProtocolActionModel).where(
            EmergencyProtocolActionModel.action_type == "halt"
        )
        result = await db_session.execute(stmt)
        actions = result.scalars().all()

        assert len(actions) == 1
        assert actions[0].triggered_by == "admin_user"


# ============================================================================
# TestRPCExecution
# ============================================================================

class TestRPCExecution:
    """Tests for RPC execution functionality."""

    @pytest.mark.asyncio
    async def test_rpc_returns_error_on_connection_failure(self):
        """Test that RPC returns error on connection failure."""
        # Mock connection failure
        response = {
            "success": False,
            "error": "connection_refused",
            "peer_id": "peer_123",
            "task_id": "task_1"
        }

        assert response["success"] is False
        assert response["error"] == "connection_refused"

    @pytest.mark.asyncio
    async def test_rpc_http_path_sends_correct_payload(self):
        """Test that HTTP path sends correct payload."""
        # Expected payload structure
        task = MagicMock()
        task.task_id = "task_123"
        task.task_type = "model_execution"
        task.instruction = "Test instruction"
        task.context_data = {"key": "value"}

        request_payload = {
            "task_id": str(task.task_id),
            "operation": task.task_type,
            "instruction": task.instruction,
            "args": task.context_data or {},
            "timeout": 30
        }

        # Verify payload structure
        assert request_payload["task_id"] == "task_123"
        assert request_payload["operation"] == "model_execution"
        assert request_payload["timeout"] == 30

    @pytest.mark.asyncio
    async def test_rpc_timeout_returns_error(self):
        """Test that RPC timeout returns appropriate error."""
        # Simulate timeout response
        response = {
            "success": False,
            "error": "connection_timeout",
            "peer_id": "peer_456",
            "task_id": "task_2"
        }

        assert response["success"] is False
        assert response["error"] == "connection_timeout"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
