"""
Tests for ContextPressureMonitor and NWTNSession context reset functionality
==============================================================================

Tests cover:
- Token recording and threshold detection
- Handoff artifact structure (required keys present)
- agents_needing_reset() returns correct agents
- reset_agent_context() clears state
- NWTNSession.run() backward compat (no context_monitor arg)
- NWTNSession.run() with context_monitor that triggers a reset
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.compute.nwtn.team.context_monitor import (
    ContextPressureMonitor,
    ContextPressureLevel,
    AgentTokenState,
)


# ======================================================================
# ContextPressureLevel Enum Tests
# ======================================================================

def test_pressure_level_enum_values():
    """Test that all expected pressure levels exist."""
    assert ContextPressureLevel.OK.value == "ok"
    assert ContextPressureLevel.WARNING.value == "warning"
    assert ContextPressureLevel.CRITICAL.value == "critical"
    assert ContextPressureLevel.HARD_LIMIT.value == "hard_limit"


def test_pressure_level_enum_count():
    """Test that we have exactly 4 pressure levels."""
    assert len(ContextPressureLevel) == 4


# ======================================================================
# ContextPressureMonitor Initialization Tests
# ======================================================================

def test_monitor_default_context_window():
    """Test that monitor initializes with default 200k context window."""
    monitor = ContextPressureMonitor()
    assert monitor.context_window_size == 200_000


def test_monitor_custom_context_window():
    """Test that monitor accepts custom context window size."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    assert monitor.context_window_size == 100_000


def test_monitor_starts_empty():
    """Test that monitor starts with no tracked agents."""
    monitor = ContextPressureMonitor()
    assert monitor.get_all_agents() == {}


# ======================================================================
# Token Recording and Threshold Detection Tests
# ======================================================================

def test_record_tokens_returns_ok():
    """Test that low token usage returns OK level."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    level = monitor.record_tokens("agent-1", 1, 10_000)  # 10%
    assert level == ContextPressureLevel.OK


def test_record_tokens_returns_warning():
    """Test that 70%+ usage returns WARNING level."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    level = monitor.record_tokens("agent-1", 1, 70_000)  # 70%
    assert level == ContextPressureLevel.WARNING


def test_record_tokens_returns_critical():
    """Test that 85%+ usage returns CRITICAL level."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    level = monitor.record_tokens("agent-1", 1, 85_000)  # 85%
    assert level == ContextPressureLevel.CRITICAL


def test_record_tokens_returns_hard_limit():
    """Test that 95%+ usage returns HARD_LIMIT level."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    level = monitor.record_tokens("agent-1", 1, 95_000)  # 95%
    assert level == ContextPressureLevel.HARD_LIMIT


def test_record_tokens_accumulates():
    """Test that tokens accumulate across multiple calls."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 30_000)
    monitor.record_tokens("agent-1", 2, 40_000)
    assert monitor.get_token_count("agent-1") == 70_000


def test_record_tokens_multiple_agents():
    """Test that multiple agents are tracked independently."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 50_000)
    monitor.record_tokens("agent-2", 1, 80_000)

    assert monitor.get_token_count("agent-1") == 50_000
    assert monitor.get_token_count("agent-2") == 80_000
    assert monitor.get_pressure_level("agent-1") == ContextPressureLevel.OK
    assert monitor.get_pressure_level("agent-2") == ContextPressureLevel.WARNING


def test_get_pressure_level_unknown_agent():
    """Test that unknown agent returns OK level."""
    monitor = ContextPressureMonitor()
    assert monitor.get_pressure_level("unknown") == ContextPressureLevel.OK


def test_get_token_count_unknown_agent():
    """Test that unknown agent returns 0 tokens."""
    monitor = ContextPressureMonitor()
    assert monitor.get_token_count("unknown") == 0


# ======================================================================
# agents_needing_reset() Tests
# ======================================================================

def test_agents_needing_reset_empty():
    """Test that no agents need reset when all are OK."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 50_000)
    assert monitor.agents_needing_reset() == []


def test_agents_needing_reset_critical():
    """Test that agents at CRITICAL level are returned."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 85_000)  # CRITICAL
    monitor.record_tokens("agent-2", 1, 50_000)  # OK

    needing_reset = monitor.agents_needing_reset()
    assert "agent-1" in needing_reset
    assert "agent-2" not in needing_reset


def test_agents_needing_reset_hard_limit():
    """Test that agents at HARD_LIMIT level are returned."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 95_000)  # HARD_LIMIT

    needing_reset = monitor.agents_needing_reset()
    assert "agent-1" in needing_reset


def test_agents_needing_reset_excludes_warning():
    """Test that WARNING level agents are NOT returned."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 70_000)  # WARNING

    assert monitor.agents_needing_reset() == []


def test_agents_at_warning():
    """Test that agents_at_warning returns only WARNING level agents."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 70_000)  # WARNING
    monitor.record_tokens("agent-2", 1, 85_000)  # CRITICAL

    warning_agents = monitor.agents_at_warning()
    assert "agent-1" in warning_agents
    assert "agent-2" not in warning_agents


# ======================================================================
# reset_agent_context() Tests
# ======================================================================

def test_reset_agent_context_clears_tokens():
    """Test that reset clears token count."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 90_000)
    assert monitor.get_token_count("agent-1") == 90_000

    monitor.reset_agent_context("agent-1")
    assert monitor.get_token_count("agent-1") == 0


def test_reset_agent_context_resets_pressure_level():
    """Test that reset resets pressure level to OK."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 90_000)
    assert monitor.get_pressure_level("agent-1") == ContextPressureLevel.CRITICAL

    monitor.reset_agent_context("agent-1")
    assert monitor.get_pressure_level("agent-1") == ContextPressureLevel.OK


def test_reset_agent_context_increments_reset_count():
    """Test that reset increments the reset count."""
    monitor = ContextPressureMonitor(context_window_size=100_000)

    monitor.record_tokens("agent-1", 1, 90_000)
    assert monitor.get_reset_count("agent-1") == 0

    monitor.reset_agent_context("agent-1")
    assert monitor.get_reset_count("agent-1") == 1

    monitor.record_tokens("agent-1", 2, 90_000)
    monitor.reset_agent_context("agent-1")
    assert monitor.get_reset_count("agent-1") == 2


def test_reset_agent_context_unknown_agent_creates_state():
    """Test that resetting unknown agent creates state for it."""
    monitor = ContextPressureMonitor()
    monitor.reset_agent_context("new-agent")
    assert monitor.get_token_count("new-agent") == 0
    assert monitor.get_reset_count("new-agent") == 1


def test_get_reset_count_unknown_agent():
    """Test that unknown agent has reset count of 0."""
    monitor = ContextPressureMonitor()
    assert monitor.get_reset_count("unknown") == 0


# ======================================================================
# Handoff Artifact Tests
# ======================================================================

def test_build_handoff_artifact_required_keys():
    """Test that handoff artifact contains all required keys."""
    monitor = ContextPressureMonitor()
    monitor.record_tokens("agent-1", 5, 90_000)

    artifact = monitor.build_handoff_artifact(
        agent_id="agent-1",
        whiteboard_entries=[],
        meta_plan_summary="Test plan summary",
        goal="Test goal",
    )

    required_keys = [
        "goal",
        "completed_work",
        "pending_criteria",
        "key_decisions",
        "next_steps",
        "round_number",
        "reset_count",
        "agent_id",
        "tokens_before_reset",
        "meta_plan_summary",
    ]

    for key in required_keys:
        assert key in artifact, f"Missing required key: {key}"


def test_build_handoff_artifact_goal():
    """Test that handoff artifact contains the goal."""
    monitor = ContextPressureMonitor()
    monitor.record_tokens("agent-1", 1, 90_000)

    artifact = monitor.build_handoff_artifact(
        agent_id="agent-1",
        whiteboard_entries=[],
        meta_plan_summary="Summary",
        goal="Build feature X",
    )

    assert artifact["goal"] == "Build feature X"


def test_build_handoff_artifact_default_goal():
    """Test that handoff artifact has default goal when not provided."""
    monitor = ContextPressureMonitor()
    monitor.record_tokens("agent-1", 1, 90_000)

    artifact = monitor.build_handoff_artifact(
        agent_id="agent-1",
        whiteboard_entries=[],
        meta_plan_summary="Summary",
    )

    assert "not available" in artifact["goal"].lower()


def test_build_handoff_artifact_round_number():
    """Test that handoff artifact contains correct round number."""
    monitor = ContextPressureMonitor()
    monitor.record_tokens("agent-1", 7, 90_000)

    artifact = monitor.build_handoff_artifact(
        agent_id="agent-1",
        whiteboard_entries=[],
        meta_plan_summary="Summary",
    )

    assert artifact["round_number"] == 7


def test_build_handoff_artifact_extract_decisions():
    """Test that handoff artifact extracts decisions from whiteboard."""
    monitor = ContextPressureMonitor()
    monitor.record_tokens("agent-1", 1, 90_000)

    whiteboard_entries = [
        {"content": "DECISION: Use PostgreSQL for database"},
        {"content": "Random notes here"},
        {"chunk": "DECIDED: Implement REST API first"},
    ]

    artifact = monitor.build_handoff_artifact(
        agent_id="agent-1",
        whiteboard_entries=whiteboard_entries,
        meta_plan_summary="Summary",
    )

    decisions = artifact["key_decisions"]
    assert len(decisions) >= 2
    assert any("PostgreSQL" in d for d in decisions)
    assert any("REST API" in d for d in decisions)


def test_build_handoff_artifact_tokens_before_reset():
    """Test that handoff artifact records tokens before reset."""
    monitor = ContextPressureMonitor(context_window_size=100_000)
    monitor.record_tokens("agent-1", 1, 50_000)
    monitor.record_tokens("agent-1", 2, 40_000)  # Total: 90k

    artifact = monitor.build_handoff_artifact(
        agent_id="agent-1",
        whiteboard_entries=[],
        meta_plan_summary="Summary",
    )

    assert artifact["tokens_before_reset"] == 90_000


def test_build_handoff_artifact_next_steps_default():
    """Test that handoff artifact provides default next steps when none found."""
    monitor = ContextPressureMonitor()
    monitor.record_tokens("agent-1", 1, 90_000)

    artifact = monitor.build_handoff_artifact(
        agent_id="agent-1",
        whiteboard_entries=[],
        meta_plan_summary="Summary",
    )

    next_steps = artifact["next_steps"]
    assert len(next_steps) >= 1
    assert any("whiteboard" in step.lower() for step in next_steps)


# ======================================================================
# Clear All Tests
# ======================================================================

def test_clear_all_removes_agents():
    """Test that clear_all removes all tracked agents."""
    monitor = ContextPressureMonitor()
    monitor.record_tokens("agent-1", 1, 50_000)
    monitor.record_tokens("agent-2", 1, 60_000)

    assert len(monitor.get_all_agents()) == 2

    monitor.clear_all()

    assert monitor.get_all_agents() == {}
    assert monitor.get_token_count("agent-1") == 0
    assert monitor.get_token_count("agent-2") == 0


# ======================================================================
# NWTNSession.run() Backward Compatibility Tests
# ======================================================================

@pytest.mark.asyncio
async def test_run_backward_compat_no_context_monitor():
    """Test that NWTNSession.run() works without context_monitor (backward compat)."""
    from prsm.compute.nwtn.session import NWTNSession, RunResult

    # Create mock adapter
    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={"pending_agents": []})

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Run WITHOUT context_monitor
    result = await session.run(max_rounds=2, round_poll_interval=0.01)

    # Verify result
    assert isinstance(result, RunResult)
    assert result.rounds_completed == 2
    assert result.converged is False
    assert result.context_resets_triggered == 0


@pytest.mark.asyncio
async def test_run_with_context_monitor_no_reset():
    """Test that context_monitor tracks tokens when provided but no reset triggered."""
    from prsm.compute.nwtn.session import NWTNSession

    monitor = ContextPressureMonitor(context_window_size=100_000)

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={"pending_agents": []})

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Agent output function - returns short text (low token count)
    async def short_output_fn(agent_id, round_num):
        return "Small output"

    # Run WITH context_monitor
    result = await session.run(
        max_rounds=2,
        round_poll_interval=0.01,
        agent_output_fn=short_output_fn,
        context_monitor=monitor,
    )

    # Should complete without reset (token count too low)
    assert result.context_resets_triggered == 0
    # Verify tokens were recorded
    assert monitor.get_token_count("agent-1") > 0


@pytest.mark.asyncio
async def test_run_with_context_monitor_triggers_reset():
    """Test that context_monitor triggers reset when agent exceeds threshold."""
    from prsm.compute.nwtn.session import NWTNSession

    monitor = ContextPressureMonitor(context_window_size=100)  # Very small for testing

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={
        "pending_agents": [],
        "whiteboard_entries": [],
        "meta_plan_summary": "Test plan",
    })

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Agent output function - returns large text to trigger threshold
    # The session estimates tokens as: len(output.split()) * 4
    # To exceed 85% of 100 tokens, we need >85 words
    async def large_output_fn(agent_id, round_num):
        # Return 100 words -> 400 tokens -> exceeds threshold
        words = "word " * 100
        return words.strip()

    # Run WITH context_monitor
    result = await session.run(
        max_rounds=2,
        round_poll_interval=0.01,
        agent_output_fn=large_output_fn,
        context_monitor=monitor,
    )

    # Should have triggered at least one reset
    assert result.context_resets_triggered >= 1


@pytest.mark.asyncio
async def test_run_result_has_context_resets_field():
    """Test that RunResult has context_resets_triggered field."""
    from prsm.compute.nwtn.session import RunResult

    result = RunResult(
        session_id="test",
        rounds_completed=1,
        converged=False,
        convergence_summary={},
        final_status="active",
        elapsed_seconds=1.0,
    )

    # Default value should be 0
    assert hasattr(result, "context_resets_triggered")
    assert result.context_resets_triggered == 0


@pytest.mark.asyncio
async def test_run_context_monitor_multiple_agents():
    """Test that context_monitor handles multiple agents correctly."""
    from prsm.compute.nwtn.session import NWTNSession

    monitor = ContextPressureMonitor(context_window_size=100)

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={
        "pending_agents": [],
        "whiteboard_entries": [],
        "meta_plan_summary": "Test plan",
    })

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1", "agent-2"]
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # One agent triggers reset, other doesn't
    async def mixed_output_fn(agent_id, round_num):
        if agent_id == "agent-1":
            # 100 words -> 400 tokens -> exceeds threshold
            return ("word " * 100).strip()
        else:
            return "small output"  # 2 words -> 8 tokens

    result = await session.run(
        max_rounds=2,
        round_poll_interval=0.01,
        agent_output_fn=mixed_output_fn,
        context_monitor=monitor,
    )

    # agent-1 should have triggered reset, agent-2 should not
    assert monitor.get_reset_count("agent-1") >= 1  # Was reset
    assert monitor.get_token_count("agent-1") == 0  # Cleared after reset
    assert monitor.get_token_count("agent-2") > 0  # Not reset (but tracked)


# ======================================================================
# Integration Tests
# ======================================================================

def test_full_pressure_level_progression():
    """Test agent going through all pressure levels."""
    monitor = ContextPressureMonitor(context_window_size=100_000)

    # Start OK
    level1 = monitor.record_tokens("agent-1", 1, 50_000)
    assert level1 == ContextPressureLevel.OK

    # Progress to WARNING
    level2 = monitor.record_tokens("agent-1", 2, 25_000)  # Total: 75k (75%)
    assert level2 == ContextPressureLevel.WARNING

    # Progress to CRITICAL
    level3 = monitor.record_tokens("agent-1", 3, 15_000)  # Total: 90k (90%)
    assert level3 == ContextPressureLevel.CRITICAL

    # Reset and back to OK
    monitor.reset_agent_context("agent-1")
    level4 = monitor.get_pressure_level("agent-1")
    assert level4 == ContextPressureLevel.OK


def test_boundary_thresholds():
    """Test exact threshold boundary values."""
    monitor = ContextPressureMonitor(context_window_size=100_000)

    # Just under WARNING (70%)
    level = monitor.record_tokens("agent-1", 1, 69_999)
    assert level == ContextPressureLevel.OK

    monitor.reset_agent_context("agent-1")

    # Exactly WARNING threshold
    level = monitor.record_tokens("agent-1", 1, 70_000)
    assert level == ContextPressureLevel.WARNING

    monitor.reset_agent_context("agent-1")

    # Just under CRITICAL (85%)
    level = monitor.record_tokens("agent-1", 1, 84_999)
    assert level == ContextPressureLevel.WARNING

    monitor.reset_agent_context("agent-1")

    # Exactly CRITICAL threshold
    level = monitor.record_tokens("agent-1", 1, 85_000)
    assert level == ContextPressureLevel.CRITICAL

    monitor.reset_agent_context("agent-1")

    # Just under HARD_LIMIT (95%)
    level = monitor.record_tokens("agent-1", 1, 94_999)
    assert level == ContextPressureLevel.CRITICAL

    monitor.reset_agent_context("agent-1")

    # Exactly HARD_LIMIT threshold
    level = monitor.record_tokens("agent-1", 1, 95_000)
    assert level == ContextPressureLevel.HARD_LIMIT