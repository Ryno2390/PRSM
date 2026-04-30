"""
Sprint 4 Phase 5: Collaboration System Bridge Tests
=====================================================

Tests for the bridge between CollaborationManager (high-level sessions)
and AgentCollaboration (P2P protocols). Verifies dispatch, bidirectional
mapping, and protocol completion callbacks.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from prsm.collaboration import (
    CollaborationManager,
    CollaborationType,
    CollaborationStatus,
    CollaborationSession,
    CollaborationResult,
)
from prsm.node.agent_collaboration import (
    AgentCollaboration,
    TaskOffer,
    TaskStatus,
    ReviewRequest,
    ReviewStatus,
    KnowledgeQuery,
)
from prsm.node.local_ledger import LocalLedger, TransactionType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest_asyncio.fixture
async def ledger():
    """Create a real in-memory ledger."""
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet("node-A", "Test Node A")
    await ledger.create_wallet("system", "System")
    await ledger.credit(
        wallet_id="node-A",
        amount=1000.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="Test grant",
    )
    yield ledger
    await ledger.close()


@pytest.fixture
def mock_gossip():
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


@pytest.fixture
def agent_collab(mock_gossip, ledger):
    """Create a real AgentCollaboration with a real ledger."""
    return AgentCollaboration(
        gossip=mock_gossip,
        node_id="node-A",
        ledger=ledger,
    )


@pytest.fixture
def manager(agent_collab):
    """Create a CollaborationManager wired to AgentCollaboration."""
    cm = CollaborationManager()
    cm.set_agent_collaboration(agent_collab)
    return cm


# =============================================================================
# TEST: dispatch_session — Task Delegation
# =============================================================================

class TestDispatchTask:
    """Test dispatching TASK_DELEGATION sessions to the P2P layer."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_task_and_starts_session(self, manager, agent_collab):
        """Dispatching a task session should create a P2P task and start the session."""
        session = manager.create_session(
            collaboration_type=CollaborationType.TASK_DELEGATION,
            initiator_agent_id="agent-1",
            participant_agent_ids=["agent-2"],
            metadata={
                "title": "Analyze dataset",
                "description": "Run statistical analysis on dataset X",
                "ftns_budget": 25.0,
                "required_capabilities": ["statistics"],
                "deadline_seconds": 1800.0,
            },
        )

        protocol_id = await manager.dispatch_session(session.session_id)

        assert protocol_id is not None
        assert session.status == CollaborationStatus.ACTIVE
        assert session.started_at is not None

        # Verify the task was created on the P2P layer
        task = agent_collab.tasks.get(protocol_id)
        assert task is not None
        assert task.title == "Analyze dataset"
        assert task.ftns_budget == 25.0
        assert task.required_capabilities == ["statistics"]

        # Verify bidirectional mapping
        assert manager.get_protocol_id(session.session_id) == protocol_id
        assert manager.get_session_for_protocol(protocol_id) == session

    @pytest.mark.asyncio
    async def test_dispatch_insufficient_balance_fails_session(self, manager):
        """Dispatch with insufficient FTNS should fail the session."""
        session = manager.create_session(
            collaboration_type=CollaborationType.TASK_DELEGATION,
            initiator_agent_id="agent-1",
            participant_agent_ids=[],
            metadata={
                "description": "Too expensive",
                "ftns_budget": 50000.0,  # Way more than available
            },
        )

        protocol_id = await manager.dispatch_session(session.session_id)

        assert protocol_id is None
        assert session.status == CollaborationStatus.FAILED
        assert "error" in session.metadata


# =============================================================================
# TEST: dispatch_session — Peer Review
# =============================================================================

class TestDispatchReview:
    """Test dispatching PEER_REVIEW sessions to the P2P layer."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_review_request(self, manager, agent_collab):
        """Dispatching a review session should create a P2P review request."""
        session = manager.create_session(
            collaboration_type=CollaborationType.PEER_REVIEW,
            initiator_agent_id="agent-1",
            participant_agent_ids=["agent-2", "agent-3"],
            metadata={
                "content_cid": "QmReviewContent123",
                "description": "Review my research paper",
                "ftns_per_review": 0.5,
                "max_reviewers": 2,
                "required_capabilities": ["research"],
            },
        )

        protocol_id = await manager.dispatch_session(session.session_id)

        assert protocol_id is not None
        assert session.status == CollaborationStatus.ACTIVE

        review = agent_collab.reviews.get(protocol_id)
        assert review is not None
        assert review.content_cid == "QmReviewContent123"
        assert review.ftns_per_review == 0.5
        assert review.max_reviewers == 2


# =============================================================================
# TEST: dispatch_session — Knowledge Exchange
# =============================================================================

class TestDispatchQuery:
    """Test dispatching KNOWLEDGE_EXCHANGE sessions to the P2P layer."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_knowledge_query(self, manager, agent_collab):
        """Dispatching a query session should create a P2P knowledge query."""
        session = manager.create_session(
            collaboration_type=CollaborationType.KNOWLEDGE_EXCHANGE,
            initiator_agent_id="agent-1",
            participant_agent_ids=[],
            metadata={
                "topic": "transformer architectures",
                "description": "What are the latest advances in attention mechanisms?",
                "ftns_per_response": 0.2,
                "max_responses": 3,
            },
        )

        protocol_id = await manager.dispatch_session(session.session_id)

        assert protocol_id is not None
        assert session.status == CollaborationStatus.ACTIVE

        query = agent_collab.queries.get(protocol_id)
        assert query is not None
        assert query.topic == "transformer architectures"
        assert query.ftns_per_response == 0.2
        assert query.max_responses == 3


# =============================================================================
# TEST: dispatch_session — Edge cases
# =============================================================================

class TestDispatchEdgeCases:
    """Test edge cases in session dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_unknown_session_returns_none(self, manager):
        """Dispatching a non-existent session should return None."""
        result = await manager.dispatch_session("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_without_agent_collab_returns_none(self):
        """Dispatching without a wired AgentCollaboration should return None."""
        cm = CollaborationManager()  # No agent_collab set
        session = cm.create_session(
            collaboration_type=CollaborationType.TASK_DELEGATION,
            initiator_agent_id="agent-1",
            participant_agent_ids=[],
            metadata={"description": "test", "ftns_budget": 1.0},
        )

        result = await cm.dispatch_session(session.session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_already_active_session_returns_none(self, manager):
        """Cannot dispatch a session that is already active."""
        session = manager.create_session(
            collaboration_type=CollaborationType.TASK_DELEGATION,
            initiator_agent_id="agent-1",
            participant_agent_ids=[],
            metadata={"description": "test", "ftns_budget": 1.0},
        )
        session.start()  # Already active

        result = await manager.dispatch_session(session.session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_joint_reasoning_starts_locally(self, manager):
        """JOINT_REASONING has no P2P protocol; session starts locally."""
        session = manager.create_session(
            collaboration_type=CollaborationType.JOINT_REASONING,
            initiator_agent_id="agent-1",
            participant_agent_ids=["agent-2"],
        )

        result = await manager.dispatch_session(session.session_id)

        assert result is None  # No protocol ID
        assert session.status == CollaborationStatus.ACTIVE  # But session started


# =============================================================================
# TEST: on_protocol_complete — completion callback
# =============================================================================

class TestProtocolCompletion:
    """Test that P2P protocol completions update session state."""

    @pytest.mark.asyncio
    async def test_task_completion_updates_session(self, manager, agent_collab):
        """When a task completes, the linked session should complete too."""
        session = manager.create_session(
            collaboration_type=CollaborationType.TASK_DELEGATION,
            initiator_agent_id="agent-1",
            participant_agent_ids=[],
            metadata={"description": "Test task", "ftns_budget": 10.0},
        )

        protocol_id = await manager.dispatch_session(session.session_id)
        assert session.status == CollaborationStatus.ACTIVE

        # Simulate protocol completion
        result = manager.on_protocol_complete(
            protocol_id=protocol_id,
            success=True,
            outputs={"answer": "42"},
            ftns_spent=10.0,
        )

        assert result is not None
        assert result.success is True
        assert result.outputs == {"answer": "42"}
        assert result.ftns_spent == 10.0
        assert session.status == CollaborationStatus.COMPLETED

        # Mappings should be cleaned up
        assert manager.get_protocol_id(session.session_id) is None
        assert manager.get_session_for_protocol(protocol_id) is None

    @pytest.mark.asyncio
    async def test_failed_protocol_fails_session(self, manager, agent_collab):
        """When a protocol fails, the linked session should fail too."""
        session = manager.create_session(
            collaboration_type=CollaborationType.TASK_DELEGATION,
            initiator_agent_id="agent-1",
            participant_agent_ids=[],
            metadata={"description": "Will fail", "ftns_budget": 5.0},
        )

        protocol_id = await manager.dispatch_session(session.session_id)

        result = manager.on_protocol_complete(
            protocol_id=protocol_id,
            success=False,
            outputs={"error": "Provider crashed"},
        )

        assert result is not None
        assert result.success is False
        assert session.status == CollaborationStatus.FAILED

    def test_unknown_protocol_id_returns_none(self, manager):
        """Completing an unknown protocol ID should return None."""
        result = manager.on_protocol_complete(
            protocol_id="unknown-protocol",
            success=True,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_result_after_completion(self, manager, agent_collab):
        """get_result() should return the CollaborationResult after completion."""
        session = manager.create_session(
            collaboration_type=CollaborationType.KNOWLEDGE_EXCHANGE,
            initiator_agent_id="agent-1",
            participant_agent_ids=[],
            metadata={
                "topic": "test",
                "description": "test query",
                "ftns_per_response": 0.1,
                "max_responses": 2,
            },
        )

        protocol_id = await manager.dispatch_session(session.session_id)
        manager.on_protocol_complete(protocol_id, success=True, outputs={"data": "result"})

        result = manager.get_result(session.session_id)
        assert result is not None
        assert result.session_id == session.session_id
        assert result.outputs == {"data": "result"}
