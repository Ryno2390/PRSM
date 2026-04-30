"""
Tests for Whiteboard Priority Router.

Tests the priority-based push/pull routing system:
    ROUTINE   → inbox only (agents pull at their own breakpoint)
    IMPORTANT → inbox + flag (agents notified at next check-in)
    URGENT    → inbox + urgent flag + EventBus URGENT_UPDATE event (immediate interrupt)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.nwtn.bsc.event_bus import BSCEvent, EventBus, EventType
from prsm.compute.nwtn.team.live_scribe import (
    AgentInbox,
    PrioritizedUpdate,
    UpdatePriority,
)
from prsm.compute.nwtn.team.whiteboard_router import RoutingResult, WhiteboardRouter


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_whiteboard_entry():
    """Create a mock WhiteboardEntry for testing."""
    entry = MagicMock()
    entry.id = 123
    entry.chunk = "Test whiteboard entry content"
    entry.source_agent = "agent/coder-20260327"
    entry.surprise_score = 0.5
    entry.promoted_at = datetime.now(timezone.utc)
    return entry


@pytest.fixture
def mock_event_bus():
    """Create a mock EventBus for testing."""
    bus = AsyncMock(spec=EventBus)
    return bus


@pytest.fixture
def agent_inbox():
    """Create a real AgentInbox for testing."""
    return AgentInbox(session_id="test-session-123")


# ======================================================================
# Test 1: ROUTINE update inbox only
# ======================================================================

async def test_routine_update_inbox_only(mock_whiteboard_entry, agent_inbox, mock_event_bus):
    """ROUTINE: inbox.push called, no EventBus publish."""
    # Register an agent
    await agent_inbox.register_agent("agent/test")

    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=mock_event_bus)

    update = PrioritizedUpdate(
        entry=mock_whiteboard_entry,
        priority=UpdatePriority.ROUTINE,
        relevant_agents=["agent/test"],
        reason="normal progress update",
    )

    result = await router.route(update, session_id="sess-1")

    # Verify result
    assert result.priority == UpdatePriority.ROUTINE
    assert result.agents_notified == ["agent/test"]
    assert result.urgent_event_published is False
    assert result.entry_id == 123
    assert result.reason == "normal progress update"

    # Verify EventBus was NOT called
    mock_event_bus.publish.assert_not_called()

    # Verify inbox has the update
    pending = await agent_inbox.pending_count("agent/test")
    assert pending["routine"] == 1


# ======================================================================
# Test 2: IMPORTANT update inbox only
# ======================================================================

async def test_important_update_inbox_only(mock_whiteboard_entry, agent_inbox, mock_event_bus):
    """IMPORTANT: inbox.push called, no EventBus publish."""
    # Register an agent
    await agent_inbox.register_agent("agent/test")

    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=mock_event_bus)

    update = PrioritizedUpdate(
        entry=mock_whiteboard_entry,
        priority=UpdatePriority.IMPORTANT,
        relevant_agents=["agent/test"],
        reason="high surprise score",
    )

    result = await router.route(update, session_id="sess-1")

    # Verify result
    assert result.priority == UpdatePriority.IMPORTANT
    assert result.agents_notified == ["agent/test"]
    assert result.urgent_event_published is False

    # Verify EventBus was NOT called
    mock_event_bus.publish.assert_not_called()

    # Verify inbox has the update
    pending = await agent_inbox.pending_count("agent/test")
    assert pending["important"] == 1


# ======================================================================
# Test 3: URGENT update publishes event
# ======================================================================

async def test_urgent_update_publishes_event(mock_whiteboard_entry, agent_inbox, mock_event_bus):
    """URGENT: inbox.push AND EventBus.publish called."""
    # Register an agent
    await agent_inbox.register_agent("agent/test")

    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=mock_event_bus)

    update = PrioritizedUpdate(
        entry=mock_whiteboard_entry,
        priority=UpdatePriority.URGENT,
        relevant_agents=["agent/test"],
        reason="conflict detected",
    )

    result = await router.route(update, session_id="sess-1")

    # Verify result
    assert result.priority == UpdatePriority.URGENT
    assert result.agents_notified == ["agent/test"]
    assert result.urgent_event_published is True

    # Verify EventBus WAS called
    mock_event_bus.publish.assert_called_once()
    call_args = mock_event_bus.publish.call_args
    event = call_args[0][0]

    assert isinstance(event, BSCEvent)
    assert event.event_type == EventType.URGENT_UPDATE
    assert event.session_id == "sess-1"

    # Verify inbox has the update
    pending = await agent_inbox.pending_count("agent/test")
    assert pending["urgent"] == 1


# ======================================================================
# Test 4: URGENT event data format
# ======================================================================

async def test_urgent_event_data_format(mock_whiteboard_entry, agent_inbox, mock_event_bus):
    """URGENT event has correct agent_ids, entry_id, reason."""
    # Register agents
    await agent_inbox.register_agent("agent/test1")
    await agent_inbox.register_agent("agent/test2")

    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=mock_event_bus)

    update = PrioritizedUpdate(
        entry=mock_whiteboard_entry,
        priority=UpdatePriority.URGENT,
        relevant_agents=["agent/test1", "agent/test2"],
        reason="critical conflict detected",
    )

    await router.route(update, session_id="sess-42")

    # Verify event data
    call_args = mock_event_bus.publish.call_args
    event = call_args[0][0]

    assert event.data["agent_ids"] == ["agent/test1", "agent/test2"]
    assert event.data["entry_id"] == "123"
    assert event.data["reason"] == "critical conflict detected"
    assert event.data["session_id"] == "sess-42"


# ======================================================================
# Test 5: URGENT without EventBus graceful degradation
# ======================================================================

async def test_urgent_without_event_bus_graceful(mock_whiteboard_entry, agent_inbox):
    """No event_bus configured, URGENT still pushes to inbox."""
    # Register an agent
    await agent_inbox.register_agent("agent/test")

    # Create router WITHOUT event bus
    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=None)

    update = PrioritizedUpdate(
        entry=mock_whiteboard_entry,
        priority=UpdatePriority.URGENT,
        relevant_agents=["agent/test"],
        reason="urgent but no bus",
    )

    result = await router.route(update, session_id="sess-1")

    # Verify result shows no event published
    assert result.priority == UpdatePriority.URGENT
    assert result.urgent_event_published is False

    # Verify inbox still got the update
    pending = await agent_inbox.pending_count("agent/test")
    assert pending["urgent"] == 1

    # Verify urgent flag is set
    has_urgent = await agent_inbox.has_urgent("agent/test")
    assert has_urgent is True


# ======================================================================
# Test 6: Routing stats track correctly
# ======================================================================

async def test_routing_stats_track_correctly(mock_whiteboard_entry, agent_inbox, mock_event_bus):
    """Route 2 routine, 1 important, 1 urgent → stats correct."""
    # Register agents
    await agent_inbox.register_agent("agent/test")

    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=mock_event_bus)

    # Route 2 ROUTINE
    for i in range(2):
        entry = MagicMock()
        entry.id = 100 + i
        entry.chunk = f"Routine update {i}"
        entry.source_agent = "agent/coder"
        entry.surprise_score = 0.3
        entry.promoted_at = datetime.now(timezone.utc)

        update = PrioritizedUpdate(
            entry=entry,
            priority=UpdatePriority.ROUTINE,
            relevant_agents=["agent/test"],
            reason="routine",
        )
        await router.route(update, session_id="sess-1")

    # Route 1 IMPORTANT
    entry = MagicMock()
    entry.id = 200
    entry.chunk = "Important update"
    entry.source_agent = "agent/coder"
    entry.surprise_score = 0.88
    entry.promoted_at = datetime.now(timezone.utc)

    update = PrioritizedUpdate(
        entry=entry,
        priority=UpdatePriority.IMPORTANT,
        relevant_agents=["agent/test"],
        reason="high surprise",
    )
    await router.route(update, session_id="sess-1")

    # Route 1 URGENT
    entry = MagicMock()
    entry.id = 300
    entry.chunk = "Urgent update"
    entry.source_agent = "agent/coder"
    entry.surprise_score = 0.98
    entry.promoted_at = datetime.now(timezone.utc)

    update = PrioritizedUpdate(
        entry=entry,
        priority=UpdatePriority.URGENT,
        relevant_agents=["agent/test"],
        reason="conflict",
    )
    await router.route(update, session_id="sess-1")

    # Verify stats
    stats = router.get_stats()
    assert stats["routine_count"] == 2
    assert stats["important_count"] == 1
    assert stats["urgent_count"] == 1
    assert stats["total_routed"] == 4


# ======================================================================
# Test 7: ScribeAgent exposes router stats
# ======================================================================

async def test_scribe_agent_exposes_router_stats(mock_whiteboard_entry, agent_inbox, mock_event_bus):
    """ScribeAgent.status() includes router_stats when configured."""
    from prsm.compute.nwtn.team.scribe_agent import ScribeAgent

    # Create a mock LiveScribe
    mock_live_scribe = AsyncMock()
    mock_live_scribe.status = AsyncMock(return_value={
        "session_id": "sess-1",
        "inbox": {},
        "checkpoint": {"all_ready": False},
    })

    # Create router
    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=mock_event_bus)

    # Create ScribeAgent with router
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe, router=router)

    # Get status
    status = await scribe_agent.status()

    # Verify router_stats is present
    assert "router_stats" in status
    assert status["router_stats"]["routine_count"] == 0
    assert status["router_stats"]["important_count"] == 0
    assert status["router_stats"]["urgent_count"] == 0
    assert status["router_stats"]["total_routed"] == 0


# ======================================================================
# Test 8: ScribeAgent route_update delegates
# ======================================================================

async def test_scribe_agent_route_update_delegates(mock_whiteboard_entry, agent_inbox, mock_event_bus):
    """ScribeAgent.route_update() delegates to router."""
    from prsm.compute.nwtn.team.scribe_agent import ScribeAgent

    # Create a mock LiveScribe
    mock_live_scribe = AsyncMock()
    mock_live_scribe.status = AsyncMock(return_value={"session_id": "sess-1"})

    # Register agent
    await agent_inbox.register_agent("agent/test")

    # Create router
    router = WhiteboardRouter(agent_inbox=agent_inbox, event_bus=mock_event_bus)

    # Create ScribeAgent with router
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe, router=router)

    # Create update
    update = PrioritizedUpdate(
        entry=mock_whiteboard_entry,
        priority=UpdatePriority.URGENT,
        relevant_agents=["agent/test"],
        reason="test delegation",
    )

    # Route via ScribeAgent
    result = await scribe_agent.route_update(update, session_id="sess-1")

    # Verify delegation worked
    assert result is not None
    assert result.priority == UpdatePriority.URGENT
    assert result.urgent_event_published is True

    # Verify router stats updated
    status = await scribe_agent.status()
    assert status["router_stats"]["urgent_count"] == 1


# ======================================================================
# Test 9: ScribeAgent no router no-op
# ======================================================================

async def test_scribe_agent_no_router_noop():
    """ScribeAgent.route_update() without router → no error."""
    from prsm.compute.nwtn.team.scribe_agent import ScribeAgent

    # Create a mock LiveScribe
    mock_live_scribe = AsyncMock()
    mock_live_scribe.status = AsyncMock(return_value={"session_id": "sess-1"})

    # Create ScribeAgent WITHOUT router
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe, router=None)

    # Create a mock update
    mock_entry = MagicMock()
    mock_entry.id = 999
    update = PrioritizedUpdate(
        entry=mock_entry,
        priority=UpdatePriority.URGENT,
        relevant_agents=["agent/test"],
        reason="test no-op",
    )

    # Should not raise
    result = await scribe_agent.route_update(update, session_id="sess-1")

    # Should return None
    assert result is None

    # Status should not have router_stats
    status = await scribe_agent.status()
    assert "router_stats" not in status


# ======================================================================
# Test 10: URGENT_UPDATE event type in bus
# ======================================================================

async def test_urgent_update_event_type_in_bus():
    """URGENT_UPDATE is a valid EventType that can be subscribed to."""
    bus = EventBus()

    # Verify URGENT_UPDATE exists in EventType
    assert EventType.URGENT_UPDATE in EventType
    assert EventType.URGENT_UPDATE.value == "urgent_update"

    # Verify EventBus has empty subscriber list for URGENT_UPDATE
    assert bus.subscriber_count(EventType.URGENT_UPDATE) == 0

    # Subscribe to URGENT_UPDATE
    callback = AsyncMock()
    await bus.subscribe(EventType.URGENT_UPDATE, callback)

    assert bus.subscriber_count(EventType.URGENT_UPDATE) == 1

    # Publish an URGENT_UPDATE event
    event = BSCEvent(
        event_type=EventType.URGENT_UPDATE,
        session_id="sess-1",
        data={
            "agent_ids": ["agent/test"],
            "entry_id": "123",
            "reason": "test event",
            "session_id": "sess-1",
        },
    )
    await bus.publish(event)

    # Verify callback was called
    callback.assert_called_once_with(event)
