"""
Tests for ScribeAgent and WhiteboardStore.query_for_agent
=========================================================

Tests the event-driven ScribeAgent and the whiteboard pull API.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the modules under test
from prsm.compute.nwtn.bsc import BSCEvent, EventBus, EventType
from prsm.compute.nwtn.team.scribe_agent import ScribeAgent
from prsm.compute.nwtn.whiteboard.schema import WhiteboardEntry


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_live_scribe():
    """Create a mock LiveScribe instance."""
    scribe = MagicMock()
    scribe.check_and_run_checkpoint = AsyncMock(return_value=None)
    scribe.get_agent_context = AsyncMock(return_value="context for agent")
    scribe.status = AsyncMock(return_value={"active": True, "entries": 5})
    return scribe


@pytest.fixture
def event_bus():
    """Create a real EventBus for testing."""
    return EventBus()


@pytest.fixture
def sample_entries():
    """Create sample whiteboard entries for testing."""
    now = datetime.now(timezone.utc)
    return [
        WhiteboardEntry(
            id=1,
            session_id="test-session",
            source_agent="agent/coder-1",
            chunk="Entry 1 from coder-1",
            surprise_score=0.5,
            raw_perplexity=10.0,
            similarity_score=0.3,
            timestamp=now,
        ),
        WhiteboardEntry(
            id=2,
            session_id="test-session",
            source_agent="agent/coder-2",
            chunk="Entry 2 from coder-2",
            surprise_score=0.9,
            raw_perplexity=20.0,
            similarity_score=0.2,
            timestamp=now,
        ),
        WhiteboardEntry(
            id=3,
            session_id="test-session",
            source_agent="agent/coder-1",
            chunk="Entry 3 from coder-1",
            surprise_score=0.7,
            raw_perplexity=15.0,
            similarity_score=0.4,
            timestamp=now,
        ),
        WhiteboardEntry(
            id=4,
            session_id="test-session",
            source_agent="agent/coder-3",
            chunk="Entry 4 from coder-3",
            surprise_score=0.3,
            raw_perplexity=8.0,
            similarity_score=0.1,
            timestamp=now,
        ),
    ]


# ======================================================================
# ScribeAgent Tests
# ======================================================================

@pytest.mark.asyncio
async def test_scribe_agent_starts_and_subscribes(mock_live_scribe, event_bus):
    """Test that start() subscribes to ROUND_ADVANCED on EventBus."""
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)

    # Initially not running
    assert not scribe_agent._running

    # Start the agent
    await scribe_agent.start(event_bus)

    # Verify it's running and subscribed
    assert scribe_agent._running
    assert event_bus.subscriber_count(EventType.ROUND_ADVANCED) == 1

    # Clean up
    await scribe_agent.stop()


@pytest.mark.asyncio
async def test_scribe_agent_triggers_checkpoint_on_round_advanced(mock_live_scribe, event_bus):
    """Test that publishing ROUND_ADVANCED event triggers check_and_run_checkpoint()."""
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)
    await scribe_agent.start(event_bus)

    # Publish a ROUND_ADVANCED event
    event = BSCEvent(
        event_type=EventType.ROUND_ADVANCED,
        data={"round": 1},
        session_id="test-session",
    )
    await event_bus.publish(event)

    # Give the background task time to run
    await asyncio.sleep(0.1)

    # Verify check_and_run_checkpoint was called
    mock_live_scribe.check_and_run_checkpoint.assert_called_once()

    # Clean up
    await scribe_agent.stop()


@pytest.mark.asyncio
async def test_scribe_agent_checkpoint_error_does_not_crash(mock_live_scribe, event_bus):
    """Test that LiveScribe errors don't crash the ScribeAgent."""
    # Make check_and_run_checkpoint raise an exception
    mock_live_scribe.check_and_run_checkpoint.side_effect = RuntimeError("Checkpoint failed!")

    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)
    await scribe_agent.start(event_bus)

    # Publish a ROUND_ADVANCED event
    event = BSCEvent(
        event_type=EventType.ROUND_ADVANCED,
        data={"round": 1},
        session_id="test-session",
    )
    await event_bus.publish(event)

    # Give the background task time to run
    await asyncio.sleep(0.1)

    # Agent should still be running (no crash)
    assert scribe_agent._running

    # checkpoint_errors should be incremented
    status = await scribe_agent.status()
    assert status["checkpoint_errors"] == 1

    # Clean up
    await scribe_agent.stop()


@pytest.mark.asyncio
async def test_scribe_agent_stop_unsubscribes(mock_live_scribe, event_bus):
    """Test that stop() unsubscribes from EventBus."""
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)

    await scribe_agent.start(event_bus)
    assert event_bus.subscriber_count(EventType.ROUND_ADVANCED) == 1

    await scribe_agent.stop()

    # Should be unsubscribed
    assert event_bus.subscriber_count(EventType.ROUND_ADVANCED) == 0
    assert not scribe_agent._running


@pytest.mark.asyncio
async def test_get_context_delegates_to_live_scribe(mock_live_scribe, event_bus):
    """Test that get_context(agent_id) calls live_scribe.get_agent_context(agent_id)."""
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)

    result = await scribe_agent.get_context("agent/coder-1")

    mock_live_scribe.get_agent_context.assert_called_once_with("agent/coder-1")
    assert result == "context for agent"


@pytest.mark.asyncio
async def test_status_includes_running_flag(mock_live_scribe, event_bus):
    """Test that status() includes running: True when started, False after stopped."""
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)

    # Before start
    status = await scribe_agent.status()
    assert status["running"] is False

    # After start
    await scribe_agent.start(event_bus)
    status = await scribe_agent.status()
    assert status["running"] is True

    # After stop
    await scribe_agent.stop()
    status = await scribe_agent.status()
    assert status["running"] is False


# ======================================================================
# WhiteboardStore.query_for_agent Tests
# ======================================================================

@pytest.mark.asyncio
async def test_whiteboard_query_for_agent_excludes_own(sample_entries):
    """Test that query_for_agent returns only entries from other agents."""
    # Create a mock store
    store = MagicMock()
    store.get_all = AsyncMock(return_value=sample_entries)

    # Import the actual method implementation
    from prsm.compute.nwtn.whiteboard.store import WhiteboardStore

    # Bind the method to our mock
    store.query_for_agent = WhiteboardStore.query_for_agent.__get__(store, type(store))

    # Query for agent/coder-1 (should exclude entries from coder-1)
    results = await store.query_for_agent(
        session_id="test-session",
        agent_id="agent/coder-1",
    )

    # Should only have entries from coder-2 and coder-3
    assert len(results) == 2
    agent_ids = {e.source_agent for e in results}
    assert "agent/coder-1" not in agent_ids
    assert "agent/coder-2" in agent_ids
    assert "agent/coder-3" in agent_ids


@pytest.mark.asyncio
async def test_whiteboard_query_for_agent_include_own(sample_entries):
    """Test that query_for_agent(include_own=True) returns all entries."""
    # Create a mock store
    store = MagicMock()
    store.get_all = AsyncMock(return_value=sample_entries)

    # Import the actual method implementation
    from prsm.compute.nwtn.whiteboard.store import WhiteboardStore

    # Bind the method to our mock
    store.query_for_agent = WhiteboardStore.query_for_agent.__get__(store, type(store))

    # Query with include_own=True
    results = await store.query_for_agent(
        session_id="test-session",
        agent_id="agent/coder-1",
        include_own=True,
    )

    # Should have all 4 entries
    assert len(results) == 4


@pytest.mark.asyncio
async def test_whiteboard_query_for_agent_sorted_by_surprise(sample_entries):
    """Test that results are sorted by surprise_score DESC."""
    # Create a mock store
    store = MagicMock()
    store.get_all = AsyncMock(return_value=sample_entries)

    # Import the actual method implementation
    from prsm.compute.nwtn.whiteboard.store import WhiteboardStore

    # Bind the method to our mock
    store.query_for_agent = WhiteboardStore.query_for_agent.__get__(store, type(store))

    # Query for agent/coder-1 (excludes coder-1's entries)
    results = await store.query_for_agent(
        session_id="test-session",
        agent_id="agent/coder-1",
    )

    # Results should be sorted by surprise_score DESC
    assert len(results) == 2
    assert results[0].surprise_score >= results[1].surprise_score

    # Entry from coder-2 has highest surprise (0.9), should be first
    assert results[0].source_agent == "agent/coder-2"
    assert results[0].surprise_score == 0.9


@pytest.mark.asyncio
async def test_whiteboard_query_for_agent_respects_max_entries(sample_entries):
    """Test that max_entries parameter limits results."""
    # Create a mock store
    store = MagicMock()
    store.get_all = AsyncMock(return_value=sample_entries)

    # Import the actual method implementation
    from prsm.compute.nwtn.whiteboard.store import WhiteboardStore

    # Bind the method to our mock
    store.query_for_agent = WhiteboardStore.query_for_agent.__get__(store, type(store))

    # Query with max_entries=1
    results = await store.query_for_agent(
        session_id="test-session",
        agent_id="agent/coder-1",
        max_entries=1,
    )

    # Should only have 1 entry
    assert len(results) == 1
    # Should be the highest surprise entry
    assert results[0].surprise_score == 0.9


# ======================================================================
# Integration Tests
# ======================================================================

@pytest.mark.asyncio
async def test_scribe_agent_counters_in_status(mock_live_scribe, event_bus):
    """Test that checkpoints_run and checkpoint_errors are tracked in status."""
    # Set up successful checkpoint result
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.synthesis = MagicMock()
    mock_result.ledger_entry = MagicMock()
    mock_live_scribe.check_and_run_checkpoint.return_value = mock_result

    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)
    await scribe_agent.start(event_bus)

    # Publish multiple ROUND_ADVANCED events
    for i in range(3):
        event = BSCEvent(
            event_type=EventType.ROUND_ADVANCED,
            data={"round": i},
            session_id="test-session",
        )
        await event_bus.publish(event)

    # Give background tasks time to complete
    await asyncio.sleep(0.2)

    status = await scribe_agent.status()
    assert status["checkpoints_run"] == 3
    assert status["checkpoint_errors"] == 0

    await scribe_agent.stop()


@pytest.mark.asyncio
async def test_scribe_agent_double_start_raises(mock_live_scribe, event_bus):
    """Test that calling start() twice raises RuntimeError."""
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)

    await scribe_agent.start(event_bus)

    with pytest.raises(RuntimeError, match="already running"):
        await scribe_agent.start(event_bus)

    await scribe_agent.stop()


@pytest.mark.asyncio
async def test_scribe_agent_stop_is_idempotent(mock_live_scribe, event_bus):
    """Test that calling stop() multiple times is safe."""
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)

    await scribe_agent.start(event_bus)
    await scribe_agent.stop()
    await scribe_agent.stop()  # Should not raise

    assert not scribe_agent._running
