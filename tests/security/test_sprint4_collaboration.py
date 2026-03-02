"""
Sprint 4: Core Collaboration Robustness Tests
===============================================

Tests for the FTNS payment integration, state change broadcasts,
expiry enforcement, and bounded memory in agent collaboration protocols.
"""

import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.node.agent_collaboration import (
    AgentCollaboration,
    BidStrategy,
    TaskOffer,
    TaskStatus,
    ReviewRequest,
    ReviewStatus,
    KnowledgeQuery,
    MAX_COMPLETED_RECORDS,
)
from prsm.node.agent_registry import AgentRecord
from prsm.node.local_ledger import LocalLedger, TransactionType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest_asyncio.fixture
async def ledger():
    """Create a real in-memory ledger for testing."""
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet("node-A", "Test Node A")
    await ledger.create_wallet("node-B", "Test Node B")
    await ledger.create_wallet("system", "System")
    # Give node-A some tokens
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
    """Create a mock gossip protocol."""
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


@pytest.fixture
def mock_ledger_sync():
    """Create a mock ledger sync."""
    sync = MagicMock()
    sync.signed_transfer = AsyncMock()
    sync.broadcast_transaction = AsyncMock()
    return sync


@pytest.fixture
def collab(mock_gossip, ledger, mock_ledger_sync):
    """Create an AgentCollaboration instance with real ledger."""
    ac = AgentCollaboration(
        gossip=mock_gossip,
        node_id="node-A",
        ledger=ledger,
        ledger_sync=mock_ledger_sync,
    )
    return ac


# =============================================================================
# PHASE 1: FTNS PAYMENT INTEGRATION
# =============================================================================

class TestTaskEscrowAndPayment:
    """Test FTNS escrow on task posting and payment on completion."""

    @pytest.mark.asyncio
    async def test_post_task_escrows_budget(self, collab, ledger):
        """Posting a task should debit the budget from the requester's wallet."""
        balance_before = await ledger.get_balance("node-A")

        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Test Task",
            description="Do something",
            ftns_budget=50.0,
        )

        balance_after = await ledger.get_balance("node-A")
        assert balance_after == balance_before - 50.0
        assert task.escrow_tx_id is not None
        assert task.status == TaskStatus.OPEN

    @pytest.mark.asyncio
    async def test_post_task_insufficient_balance(self, collab, ledger):
        """Posting a task with insufficient balance should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot post task"):
            await collab.post_task(
                requester_agent_id="agent-1",
                title="Expensive Task",
                description="Too expensive",
                ftns_budget=5000.0,  # More than the 1000 FTNS grant
            )

    @pytest.mark.asyncio
    async def test_complete_task_pays_agent(self, collab, ledger, mock_ledger_sync):
        """Completing a task should pay the assigned agent via ledger_sync."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Test Task",
            description="Do something",
            ftns_budget=25.0,
        )

        # Simulate a bid from a remote agent
        task.bids.append({
            "bidder_agent_id": "agent-2",
            "bidder_node_id": "node-B",
            "estimated_cost": 20.0,
            "estimated_seconds": 60,
        })

        await collab.assign_task(task.task_id, "agent-2")
        assert task.status == TaskStatus.ASSIGNED

        result = await collab.complete_task(task.task_id, {"output": "done"})
        assert result is True
        assert task.status == TaskStatus.COMPLETED

        # Verify payment was sent via ledger_sync
        mock_ledger_sync.signed_transfer.assert_called_once_with(
            to_wallet="node-B",
            amount=25.0,
            description="Task payment: Test Task",
        )

    @pytest.mark.asyncio
    async def test_cancel_task_refunds_escrow(self, collab, ledger):
        """Cancelling a task should refund the escrowed FTNS."""
        balance_before = await ledger.get_balance("node-A")

        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Cancelled Task",
            description="Will be cancelled",
            ftns_budget=30.0,
        )

        balance_after_post = await ledger.get_balance("node-A")
        assert balance_after_post == balance_before - 30.0

        await collab.cancel_task(task.task_id)

        balance_after_cancel = await ledger.get_balance("node-A")
        assert balance_after_cancel == balance_before  # Full refund
        assert task.status == TaskStatus.CANCELLED


class TestReviewEscrowAndPayment:
    """Test FTNS escrow on review requests and payment on review submission."""

    @pytest.mark.asyncio
    async def test_request_review_escrows_budget(self, collab, ledger):
        """Requesting a review should escrow (ftns_per_review * max_reviewers)."""
        balance_before = await ledger.get_balance("node-A")

        review = await collab.request_review(
            submitter_agent_id="agent-1",
            content_cid="QmTest123",
            description="Review this",
            ftns_per_review=0.5,
            max_reviewers=3,
        )

        balance_after = await ledger.get_balance("node-A")
        assert balance_after == balance_before - 1.5  # 0.5 * 3
        assert review.escrow_tx_id is not None

    @pytest.mark.asyncio
    async def test_submit_review_pays_reviewer(self, collab, ledger, mock_ledger_sync):
        """Submitting a review should pay the reviewer from escrow."""
        review = await collab.request_review(
            submitter_agent_id="agent-1",
            content_cid="QmTest123",
            description="Review this",
            ftns_per_review=0.5,
            max_reviewers=2,
        )

        result = await collab.submit_review(
            review_id=review.review_id,
            reviewer_agent_id="agent-2",
            reviewer_node_id="node-B",
            verdict="accept",
            comments="Looks good",
        )

        assert result is True
        mock_ledger_sync.signed_transfer.assert_called_once_with(
            to_wallet="node-B",
            amount=0.5,
            description=f"Review payment: {review.review_id[:8]}",
        )
        assert "agent-2" in review.paid_reviewers


class TestQueryEscrowAndPayment:
    """Test FTNS escrow on knowledge queries and payment on response."""

    @pytest.mark.asyncio
    async def test_post_query_escrows_budget(self, collab, ledger):
        """Posting a query should escrow (ftns_per_response * max_responses)."""
        balance_before = await ledger.get_balance("node-A")

        query = await collab.post_query(
            requester_agent_id="agent-1",
            topic="AI Safety",
            question="What is alignment?",
            ftns_per_response=0.1,
            max_responses=5,
        )

        balance_after = await ledger.get_balance("node-A")
        assert balance_after == balance_before - 0.5  # 0.1 * 5
        assert query.escrow_tx_id is not None

    @pytest.mark.asyncio
    async def test_submit_response_pays_responder(self, collab, ledger, mock_ledger_sync):
        """Submitting a knowledge response should pay the responder."""
        query = await collab.post_query(
            requester_agent_id="agent-1",
            topic="AI",
            question="What is ML?",
            ftns_per_response=0.2,
            max_responses=3,
        )

        result = await collab.submit_response(
            query_id=query.query_id,
            responder_agent_id="agent-2",
            responder_node_id="node-B",
            answer="Machine Learning is...",
        )

        assert result is True
        mock_ledger_sync.signed_transfer.assert_called_once_with(
            to_wallet="node-B",
            amount=0.2,
            description=f"Query response payment: {query.query_id[:8]}",
        )
        assert "agent-2" in query.paid_responders


# =============================================================================
# PHASE 2: STATE CHANGE BROADCASTS
# =============================================================================

class TestStateBroadcasts:
    """Test that all state changes are broadcast via gossip."""

    @pytest.mark.asyncio
    async def test_assign_task_broadcasts(self, collab, mock_gossip):
        """assign_task should broadcast GOSSIP_TASK_ASSIGN."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Test",
            description="Test",
            ftns_budget=10.0,
        )
        mock_gossip.publish.reset_mock()

        await collab.assign_task(task.task_id, "agent-2")

        mock_gossip.publish.assert_called_once()
        call_args = mock_gossip.publish.call_args
        assert call_args[0][0] == "agent_task_assign"
        assert call_args[0][1]["task_id"] == task.task_id
        assert call_args[0][1]["assigned_agent_id"] == "agent-2"

    @pytest.mark.asyncio
    async def test_complete_task_broadcasts(self, collab, mock_gossip):
        """complete_task should broadcast GOSSIP_TASK_COMPLETE."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Test",
            description="Test",
            ftns_budget=10.0,
        )
        await collab.assign_task(task.task_id, "agent-2")
        mock_gossip.publish.reset_mock()

        await collab.complete_task(task.task_id, {"result": "ok"})

        # Should have been called (complete broadcast)
        calls = mock_gossip.publish.call_args_list
        complete_calls = [c for c in calls if c[0][0] == "agent_task_complete"]
        assert len(complete_calls) == 1
        assert complete_calls[0][0][1]["task_id"] == task.task_id

    @pytest.mark.asyncio
    async def test_cancel_task_broadcasts(self, collab, mock_gossip):
        """cancel_task should broadcast GOSSIP_TASK_CANCEL."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Test",
            description="Test",
            ftns_budget=10.0,
        )
        mock_gossip.publish.reset_mock()

        await collab.cancel_task(task.task_id)

        calls = mock_gossip.publish.call_args_list
        cancel_calls = [c for c in calls if c[0][0] == "agent_task_cancel"]
        assert len(cancel_calls) == 1

    @pytest.mark.asyncio
    async def test_submit_review_broadcasts(self, collab, mock_gossip):
        """submit_review should broadcast GOSSIP_REVIEW_SUBMIT."""
        review = await collab.request_review(
            submitter_agent_id="agent-1",
            content_cid="QmTest",
            description="Review",
            ftns_per_review=0.1,
        )
        mock_gossip.publish.reset_mock()

        await collab.submit_review(
            review_id=review.review_id,
            reviewer_agent_id="agent-2",
            reviewer_node_id="node-B",
            verdict="accept",
        )

        calls = mock_gossip.publish.call_args_list
        review_calls = [c for c in calls if c[0][0] == "agent_review_submit"]
        assert len(review_calls) == 1
        assert review_calls[0][0][1]["verdict"] == "accept"

    @pytest.mark.asyncio
    async def test_submit_response_broadcasts(self, collab, mock_gossip):
        """submit_response should broadcast GOSSIP_KNOWLEDGE_RESPONSE."""
        query = await collab.post_query(
            requester_agent_id="agent-1",
            topic="Test",
            question="What?",
            ftns_per_response=0.1,
        )
        mock_gossip.publish.reset_mock()

        await collab.submit_response(
            query_id=query.query_id,
            responder_agent_id="agent-2",
            responder_node_id="node-B",
            answer="Answer here",
        )

        calls = mock_gossip.publish.call_args_list
        response_calls = [c for c in calls if c[0][0] == "agent_knowledge_response"]
        assert len(response_calls) == 1


# =============================================================================
# PHASE 2: INCOMING STATE CHANGE HANDLERS
# =============================================================================

class TestIncomingStateChanges:
    """Test gossip handlers for incoming state change notifications."""

    @pytest.mark.asyncio
    async def test_on_task_assign_updates_local_state(self, collab):
        """Receiving a task assignment should update local task state."""
        # Simulate receiving a task offer from remote node
        await collab._on_task_offer("", {
            "task_id": "task-123",
            "requester_agent_id": "remote-agent",
            "requester_node_id": "node-B",
            "title": "Remote Task",
            "description": "From network",
            "ftns_budget": 10.0,
        }, "node-B")

        assert "task-123" in collab.tasks
        assert collab.tasks["task-123"].status == TaskStatus.OPEN

        # Now receive assignment notification
        await collab._on_task_assign("", {
            "task_id": "task-123",
            "assigned_agent_id": "agent-2",
        }, "node-B")

        assert collab.tasks["task-123"].status == TaskStatus.ASSIGNED
        assert collab.tasks["task-123"].assigned_agent_id == "agent-2"

    @pytest.mark.asyncio
    async def test_on_task_complete_updates_and_archives(self, collab):
        """Receiving a task completion should update status and archive."""
        await collab._on_task_offer("", {
            "task_id": "task-456",
            "requester_agent_id": "remote-agent",
            "requester_node_id": "node-B",
            "title": "Remote Task",
            "ftns_budget": 5.0,
        }, "node-B")

        await collab._on_task_complete("", {
            "task_id": "task-456",
        }, "node-B")

        # Should be archived (removed from active, added to completed)
        assert "task-456" not in collab.tasks
        assert "task-456" in collab._completed_tasks

    @pytest.mark.asyncio
    async def test_on_review_submit_updates_state(self, collab):
        """Receiving a review submission should add the review record."""
        await collab._on_review_request("", {
            "review_id": "rev-789",
            "submitter_agent_id": "remote-agent",
            "submitter_node_id": "node-B",
            "content_cid": "QmTest",
            "description": "Review this",
            "ftns_per_review": 0.1,
            "max_reviewers": 3,
        }, "node-B")

        await collab._on_review_submit("", {
            "review_id": "rev-789",
            "reviewer_agent_id": "reviewer-1",
            "reviewer_node_id": "node-C",
            "verdict": "accept",
            "review_status": "pending",
        }, "node-C")

        review = collab.reviews["rev-789"]
        assert len(review.reviews) == 1
        assert review.reviews[0]["verdict"] == "accept"


# =============================================================================
# PHASE 3: EXPIRY ENFORCEMENT & BOUNDED MEMORY
# =============================================================================

class TestExpiryEnforcement:
    """Test that expired collaborations are cleaned up with escrow refunds."""

    @pytest.mark.asyncio
    async def test_expired_task_cancelled_and_refunded(self, collab, ledger):
        """Tasks past their deadline should be cancelled with escrow refund."""
        balance_before = await ledger.get_balance("node-A")

        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Will Expire",
            description="Short deadline",
            ftns_budget=20.0,
            deadline_seconds=1.0,  # 1 second deadline
        )

        # Artificially age the task
        task.created_at = time.time() - 100

        await collab._expire_stale_records()

        assert task.status == TaskStatus.CANCELLED
        assert task.task_id not in collab.tasks
        assert task.task_id in collab._completed_tasks

        balance_after = await ledger.get_balance("node-A")
        assert balance_after == balance_before  # Full refund

    @pytest.mark.asyncio
    async def test_expired_review_refunded(self, collab, ledger):
        """Reviews past timeout should be closed with unused escrow refunded."""
        balance_before = await ledger.get_balance("node-A")

        review = await collab.request_review(
            submitter_agent_id="agent-1",
            content_cid="QmTest",
            description="Will expire",
            ftns_per_review=0.5,
            max_reviewers=3,
        )

        # Artificially age the review
        review.created_at = time.time() - 5000

        await collab._expire_stale_records()

        assert review.review_id not in collab.reviews
        assert review.review_id in collab._completed_reviews

        balance_after = await ledger.get_balance("node-A")
        # 1.5 was escrowed, all 3 unused, so 1.5 refunded
        assert balance_after == balance_before

    @pytest.mark.asyncio
    async def test_expired_query_refunded(self, collab, ledger):
        """Queries past timeout should refund unused response escrow."""
        balance_before = await ledger.get_balance("node-A")

        query = await collab.post_query(
            requester_agent_id="agent-1",
            topic="Test",
            question="Will expire?",
            ftns_per_response=0.1,
            max_responses=5,
        )

        # Artificially age the query
        query.created_at = time.time() - 5000

        await collab._expire_stale_records()

        assert query.query_id not in collab.queries
        assert query.query_id in collab._completed_queries

        balance_after = await ledger.get_balance("node-A")
        # 0.5 was escrowed, all 5 unused, so 0.5 refunded
        assert balance_after == balance_before


class TestBoundedMemory:
    """Test that completed record archives respect size bounds."""

    def test_archive_enforces_max_size(self, collab):
        """Completed archives should evict oldest records when full."""
        # Fill the archive beyond the limit
        for i in range(MAX_COMPLETED_RECORDS + 50):
            task = TaskOffer(
                task_id=f"task-{i}",
                status=TaskStatus.COMPLETED,
            )
            collab._completed_tasks[task.task_id] = task

        collab._enforce_archive_bounds()

        assert len(collab._completed_tasks) == MAX_COMPLETED_RECORDS
        # The oldest 50 should have been evicted
        assert "task-0" not in collab._completed_tasks
        assert "task-49" not in collab._completed_tasks
        # The newest should still be there
        assert f"task-{MAX_COMPLETED_RECORDS + 49}" in collab._completed_tasks

    def test_archive_task_moves_from_active(self, collab):
        """Archiving should remove from active dict and add to archive."""
        task = TaskOffer(task_id="test-archive", status=TaskStatus.COMPLETED)
        collab.tasks["test-archive"] = task

        collab._archive_task(task)

        assert "test-archive" not in collab.tasks
        assert "test-archive" in collab._completed_tasks


# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

class TestGracefulShutdown:
    """Test that stopping the collaboration system refunds active escrows."""

    @pytest.mark.asyncio
    async def test_stop_refunds_open_tasks(self, collab, ledger):
        """Stopping should cancel open tasks and refund escrow."""
        balance_before = await ledger.get_balance("node-A")

        await collab.post_task(
            requester_agent_id="agent-1",
            title="Will be refunded on stop",
            description="Test",
            ftns_budget=15.0,
        )

        balance_after_post = await ledger.get_balance("node-A")
        assert balance_after_post == balance_before - 15.0

        await collab.stop()

        balance_after_stop = await ledger.get_balance("node-A")
        assert balance_after_stop == balance_before  # Refunded


# =============================================================================
# STATS
# =============================================================================

class TestStats:
    """Test collaboration statistics reporting."""

    @pytest.mark.asyncio
    async def test_stats_reflect_state(self, collab):
        """Stats should accurately reflect current collaboration state."""
        await collab.post_task(
            requester_agent_id="agent-1",
            title="Task 1",
            description="Test",
            ftns_budget=5.0,
        )
        await collab.post_task(
            requester_agent_id="agent-1",
            title="Task 2",
            description="Test",
            ftns_budget=5.0,
        )

        stats = collab.get_stats()
        assert stats["active_tasks"] == 2
        assert stats["open_tasks"] == 2
        assert stats["assigned_tasks"] == 0
        assert stats["archived_tasks"] == 0
        assert stats["total_active_bids"] == 0
        assert stats["bid_strategy"] == "best_score"


# =============================================================================
# BID SELECTION (Sprint 5)
# =============================================================================

@pytest.fixture
def mock_registry():
    """Create a mock agent registry with some agents."""
    registry = MagicMock()

    def _lookup(agent_id):
        agents = {
            "agent-fast": AgentRecord(
                agent_id="agent-fast",
                agent_name="Fast Agent",
                agent_type="compute",
                principal_id="p1",
                principal_public_key="pk1",
                public_key_b64="pub1",
                delegation_cert="cert1",
                capabilities=["nlp", "vision", "code"],
                last_seen=time.time(),
            ),
            "agent-cheap": AgentRecord(
                agent_id="agent-cheap",
                agent_name="Cheap Agent",
                agent_type="compute",
                principal_id="p2",
                principal_public_key="pk2",
                public_key_b64="pub2",
                delegation_cert="cert2",
                capabilities=["nlp"],
                last_seen=time.time(),
            ),
            "agent-stale": AgentRecord(
                agent_id="agent-stale",
                agent_name="Stale Agent",
                agent_type="compute",
                principal_id="p3",
                principal_public_key="pk3",
                public_key_b64="pub3",
                delegation_cert="cert3",
                capabilities=["nlp", "vision", "code"],
                last_seen=time.time() - 7200,  # 2 hours old
            ),
        }
        return agents.get(agent_id)

    registry.lookup = MagicMock(side_effect=_lookup)
    return registry


@pytest.fixture
def collab_with_registry(mock_gossip, ledger, mock_ledger_sync, mock_registry):
    """AgentCollaboration instance with agent registry wired in."""
    ac = AgentCollaboration(
        gossip=mock_gossip,
        node_id="node-A",
        ledger=ledger,
        ledger_sync=mock_ledger_sync,
        agent_registry=mock_registry,
    )
    return ac


class TestBidValidation:
    """Test bid validation in submit_bid()."""

    @pytest.mark.asyncio
    async def test_reject_over_budget_bid(self, collab):
        """Bids exceeding the task budget should be rejected."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Budget Test",
            description="Testing bid validation",
            ftns_budget=10.0,
        )

        result = await collab.submit_bid(
            task_id=task.task_id,
            bidder_agent_id="agent-2",
            estimated_cost=15.0,  # Over budget
            estimated_seconds=60,
        )

        assert result is False
        assert len(task.bids) == 0

    @pytest.mark.asyncio
    async def test_reject_bid_on_non_open_task(self, collab):
        """Bids on non-OPEN tasks should be rejected."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Assigned Task",
            description="Already assigned",
            ftns_budget=20.0,
        )

        # Add a valid bid and assign
        task.bids.append({
            "bidder_agent_id": "agent-2",
            "bidder_node_id": "node-B",
            "estimated_cost": 10.0,
            "estimated_seconds": 60,
        })
        await collab.assign_task(task.task_id, "agent-2")

        # Now try to bid on the assigned task
        result = await collab.submit_bid(
            task_id=task.task_id,
            bidder_agent_id="agent-3",
            estimated_cost=8.0,
            estimated_seconds=30,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_accept_valid_bid(self, collab):
        """Valid bids within budget should be accepted."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Valid Bid Test",
            description="Test",
            ftns_budget=50.0,
        )

        result = await collab.submit_bid(
            task_id=task.task_id,
            bidder_agent_id="agent-2",
            estimated_cost=25.0,
            estimated_seconds=120,
        )

        assert result is True
        assert len(task.bids) == 1


class TestScoreBid:
    """Test bid scoring logic."""

    @pytest.mark.asyncio
    async def test_cost_efficiency_ranking(self, collab_with_registry):
        """Cheaper bids should score higher on cost component."""
        task = await collab_with_registry.post_task(
            requester_agent_id="agent-1",
            title="Score Test",
            description="Testing scoring",
            ftns_budget=100.0,
            deadline_seconds=3600.0,
        )

        cheap_bid = {
            "bidder_agent_id": "agent-cheap",
            "estimated_cost": 20.0,
            "estimated_seconds": 1800.0,
        }
        expensive_bid = {
            "bidder_agent_id": "agent-cheap",
            "estimated_cost": 80.0,
            "estimated_seconds": 1800.0,
        }

        cheap_score = collab_with_registry.score_bid(cheap_bid, task)
        expensive_score = collab_with_registry.score_bid(expensive_bid, task)

        assert cheap_score > expensive_score

    @pytest.mark.asyncio
    async def test_capability_match_scoring(self, collab_with_registry):
        """Agents with more matching capabilities should score higher."""
        task = await collab_with_registry.post_task(
            requester_agent_id="agent-1",
            title="Cap Test",
            description="Testing capabilities",
            ftns_budget=100.0,
            required_capabilities=["nlp", "vision", "code"],
        )

        full_match_bid = {
            "bidder_agent_id": "agent-fast",  # has nlp, vision, code
            "estimated_cost": 50.0,
            "estimated_seconds": 1800.0,
        }
        partial_match_bid = {
            "bidder_agent_id": "agent-cheap",  # has only nlp
            "estimated_cost": 50.0,
            "estimated_seconds": 1800.0,
        }

        full_score = collab_with_registry.score_bid(full_match_bid, task)
        partial_score = collab_with_registry.score_bid(partial_match_bid, task)

        assert full_score > partial_score

    @pytest.mark.asyncio
    async def test_over_budget_scores_zero(self, collab_with_registry):
        """Over-budget bids should score exactly 0.0."""
        task = await collab_with_registry.post_task(
            requester_agent_id="agent-1",
            title="Over Budget",
            description="Test",
            ftns_budget=10.0,
        )

        over_bid = {
            "bidder_agent_id": "agent-fast",
            "estimated_cost": 15.0,
            "estimated_seconds": 60,
        }

        score = collab_with_registry.score_bid(over_bid, task)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_freshness_rewards_recent_agents(self, collab_with_registry):
        """Recently seen agents should score higher on freshness than stale ones."""
        task = await collab_with_registry.post_task(
            requester_agent_id="agent-1",
            title="Freshness Test",
            description="Test",
            ftns_budget=100.0,
        )

        fresh_bid = {
            "bidder_agent_id": "agent-fast",  # last_seen = now
            "estimated_cost": 50.0,
            "estimated_seconds": 1800.0,
        }
        stale_bid = {
            "bidder_agent_id": "agent-stale",  # last_seen = 2 hours ago
            "estimated_cost": 50.0,
            "estimated_seconds": 1800.0,
        }

        fresh_score = collab_with_registry.score_bid(fresh_bid, task)
        stale_score = collab_with_registry.score_bid(stale_bid, task)

        assert fresh_score > stale_score


class TestSelectBestBid:
    """Test bid selection strategies."""

    @pytest.mark.asyncio
    async def test_lowest_cost_strategy(self, mock_gossip, ledger, mock_ledger_sync):
        """LOWEST_COST strategy should pick the cheapest valid bid."""
        ac = AgentCollaboration(
            gossip=mock_gossip,
            node_id="node-A",
            ledger=ledger,
            bid_strategy=BidStrategy.LOWEST_COST,
        )

        task = await ac.post_task(
            requester_agent_id="agent-1",
            title="Cost Test",
            description="Test",
            ftns_budget=100.0,
        )

        task.bids = [
            {"bidder_agent_id": "a1", "estimated_cost": 50.0, "estimated_seconds": 100},
            {"bidder_agent_id": "a2", "estimated_cost": 20.0, "estimated_seconds": 200},
            {"bidder_agent_id": "a3", "estimated_cost": 80.0, "estimated_seconds": 50},
        ]

        winner = ac.select_best_bid(task)
        assert winner is not None
        assert winner["bidder_agent_id"] == "a2"

    @pytest.mark.asyncio
    async def test_fastest_strategy(self, mock_gossip, ledger, mock_ledger_sync):
        """FASTEST strategy should pick the bid with shortest estimated time."""
        ac = AgentCollaboration(
            gossip=mock_gossip,
            node_id="node-A",
            ledger=ledger,
            bid_strategy=BidStrategy.FASTEST,
        )

        task = await ac.post_task(
            requester_agent_id="agent-1",
            title="Speed Test",
            description="Test",
            ftns_budget=100.0,
        )

        task.bids = [
            {"bidder_agent_id": "a1", "estimated_cost": 50.0, "estimated_seconds": 100},
            {"bidder_agent_id": "a2", "estimated_cost": 20.0, "estimated_seconds": 200},
            {"bidder_agent_id": "a3", "estimated_cost": 80.0, "estimated_seconds": 30},
        ]

        winner = ac.select_best_bid(task)
        assert winner is not None
        assert winner["bidder_agent_id"] == "a3"

    @pytest.mark.asyncio
    async def test_best_score_strategy(self, collab_with_registry):
        """BEST_SCORE strategy should pick the highest composite-scored bid."""
        task = await collab_with_registry.post_task(
            requester_agent_id="agent-1",
            title="Score Strategy",
            description="Test",
            ftns_budget=100.0,
            required_capabilities=["nlp", "vision", "code"],
        )

        task.bids = [
            {"bidder_agent_id": "agent-fast", "estimated_cost": 40.0, "estimated_seconds": 600},
            {"bidder_agent_id": "agent-cheap", "estimated_cost": 10.0, "estimated_seconds": 3000},
        ]

        winner = collab_with_registry.select_best_bid(task)
        assert winner is not None
        # agent-fast should win: better capability match + better time, even if costlier
        assert winner["bidder_agent_id"] == "agent-fast"

    @pytest.mark.asyncio
    async def test_no_bids_returns_none(self, collab):
        """Selecting from an empty bid list should return None."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Empty Bids",
            description="Test",
            ftns_budget=50.0,
        )

        winner = collab.select_best_bid(task)
        assert winner is None

    @pytest.mark.asyncio
    async def test_all_over_budget_returns_none(self, collab):
        """When all bids exceed budget, selection should return None."""
        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="All Over Budget",
            description="Test",
            ftns_budget=10.0,
        )

        task.bids = [
            {"bidder_agent_id": "a1", "estimated_cost": 15.0, "estimated_seconds": 60},
            {"bidder_agent_id": "a2", "estimated_cost": 20.0, "estimated_seconds": 30},
        ]

        winner = collab.select_best_bid(task)
        assert winner is None


class TestAutoAssignTask:
    """Test the auto_assign_task pipeline."""

    @pytest.mark.asyncio
    async def test_auto_assign_basic(self, collab):
        """auto_assign_task should select and assign the best bid."""
        collab.bid_window_seconds = 2.0
        collab.min_bids = 1

        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Auto-assign Test",
            description="Test",
            ftns_budget=50.0,
        )

        # Pre-populate a bid (simulating network arrival)
        task.bids.append({
            "bidder_agent_id": "agent-2",
            "bidder_node_id": "node-B",
            "estimated_cost": 30.0,
            "estimated_seconds": 120,
            "timestamp": time.time(),
        })

        agent_id = await collab.auto_assign_task(task.task_id)

        assert agent_id == "agent-2"
        assert task.status == TaskStatus.ASSIGNED
        assert task.assigned_agent_id == "agent-2"

    @pytest.mark.asyncio
    async def test_auto_assign_no_bids_returns_none(self, collab):
        """auto_assign_task with no bids should return None."""
        collab.bid_window_seconds = 2.0
        collab.min_bids = 1

        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="No Bids",
            description="Test",
            ftns_budget=50.0,
        )

        agent_id = await collab.auto_assign_task(task.task_id)

        assert agent_id is None
        assert task.status == TaskStatus.OPEN  # Still open, not assigned

    @pytest.mark.asyncio
    async def test_auto_assign_early_exit_on_min_bids(self, collab):
        """auto_assign_task should exit early once min_bids met past half window."""
        collab.bid_window_seconds = 10.0
        collab.min_bids = 1

        task = await collab.post_task(
            requester_agent_id="agent-1",
            title="Early Exit",
            description="Test",
            ftns_budget=50.0,
        )

        # Pre-populate enough bids
        task.bids.append({
            "bidder_agent_id": "agent-2",
            "bidder_node_id": "node-B",
            "estimated_cost": 25.0,
            "estimated_seconds": 60,
            "timestamp": time.time(),
        })

        start = time.time()
        agent_id = await collab.auto_assign_task(task.task_id)
        elapsed = time.time() - start

        assert agent_id == "agent-2"
        # Should exit around half-window (5s) + 1s polling, not wait the full 10s
        assert elapsed < 8.0
