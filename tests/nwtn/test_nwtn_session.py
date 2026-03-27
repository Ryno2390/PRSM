"""
Tests for NWTNSession and ScribeAgent wiring in NWTNOpenClawAdapter
===================================================================

Tests the convenience factory and ScribeAgent lifecycle management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the modules under test
from prsm.compute.nwtn.bsc.event_bus import EventBus, EventType


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_store():
    """Create a mock WhiteboardStore."""
    store = MagicMock()
    store.create_session = AsyncMock(return_value=None)
    store.archive_session = AsyncMock(return_value=None)
    store.write = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_promoter():
    """Create a mock BSCPromoter."""
    promoter = MagicMock()
    promoter.advance_round = MagicMock(return_value={"round": 0, "epsilon": None, "dedup_evicted": 0})
    return promoter


@pytest.fixture
def event_bus():
    """Create a real EventBus for testing."""
    return EventBus()


@pytest.fixture
def mock_meta_plan():
    """Create a mock MetaPlan."""
    from prsm.compute.nwtn.team.planner import Milestone
    plan = MagicMock()
    plan.title = "Test Plan"
    plan.session_id = "test-session"
    plan.milestones = [Milestone(title="Phase 1", description="Initial implementation")]
    plan.to_whiteboard_entry = MagicMock(return_value="## Test Plan\n\nMilestone 1: Test")
    plan.roles_by_priority = MagicMock(return_value=[])
    return plan


@pytest.fixture
def mock_team():
    """Create a mock AgentTeam."""
    team = MagicMock()
    member1 = MagicMock()
    member1.role = "coder"
    member1.branch_name = "coder-main-20260327"
    member1.agent_id = "agent/coder-main-20260327"
    member2 = MagicMock()
    member2.role = "architect"
    member2.branch_name = "architect-main-20260327"
    member2.agent_id = "agent/architect-main-20260327"
    team.members = [member1, member2]
    return team


@pytest.fixture
def mock_interview_brief():
    """Create a mock interview brief."""
    brief = MagicMock()
    brief.goal = "Test goal"
    brief.context = "Test context"
    return brief


# ======================================================================
# NWTNSession Factory Tests - Mock the adapter at the import location
# ======================================================================

@pytest.mark.asyncio
async def test_nwtn_session_create_builds_adapter():
    """Test that NWTNSession.create() returns a session with session_id set."""
    # Mock the imports inside create() method
    with patch("prsm.compute.nwtn.whiteboard.store.WhiteboardStore") as mock_store_cls, \
         patch("prsm.compute.nwtn.bsc.promoter.BSCPromoter") as mock_promoter_cls, \
         patch("prsm.compute.nwtn.bsc.deployment.BSCDeploymentConfig") as mock_config_cls, \
         patch("prsm.compute.nwtn.openclaw.adapter.NWTNOpenClawAdapter") as mock_adapter_cls:
        
        # Set up mocks
        mock_store = MagicMock()
        mock_store.create_session = AsyncMock(return_value=None)
        mock_store_cls.return_value = mock_store
        
        mock_promoter = MagicMock()
        mock_promoter_cls.from_config.return_value = mock_promoter
        
        mock_config_cls.auto.return_value = MagicMock()
        
        # Mock adapter and its start_session method
        mock_adapter = MagicMock()
        mock_state = MagicMock()
        mock_state.session_id = "test-session-123"
        mock_state.team_members = ["coder-main"]
        mock_state.scribe_running = True
        mock_adapter.start_session = AsyncMock(return_value=mock_state)
        mock_adapter_cls.return_value = mock_adapter
        
        from prsm.compute.nwtn.session import NWTNSession
        
        # Create session
        session = await NWTNSession.create(goal="Test goal")
        
        # Verify session was created correctly
        assert session.session_id == "test-session-123"
        assert session.team_members == ["coder-main"]


@pytest.mark.asyncio
async def test_nwtn_session_scribe_starts():
    """Test that after create(), session.scribe_running == True."""
    with patch("prsm.compute.nwtn.whiteboard.store.WhiteboardStore") as mock_store_cls, \
         patch("prsm.compute.nwtn.bsc.promoter.BSCPromoter") as mock_promoter_cls, \
         patch("prsm.compute.nwtn.bsc.deployment.BSCDeploymentConfig") as mock_config_cls, \
         patch("prsm.compute.nwtn.openclaw.adapter.NWTNOpenClawAdapter") as mock_adapter_cls:
        
        # Set up mocks
        mock_store = MagicMock()
        mock_store.create_session = AsyncMock(return_value=None)
        mock_store_cls.return_value = mock_store
        
        mock_promoter = MagicMock()
        mock_promoter_cls.from_config.return_value = mock_promoter
        
        mock_config_cls.auto.return_value = MagicMock()
        
        # Mock adapter with scribe_running=True
        mock_adapter = MagicMock()
        mock_state = MagicMock()
        mock_state.session_id = "test-session"
        mock_state.team_members = []
        mock_state.scribe_running = True
        mock_adapter.start_session = AsyncMock(return_value=mock_state)
        mock_adapter_cls.return_value = mock_adapter
        
        from prsm.compute.nwtn.session import NWTNSession
        
        # Create session
        session = await NWTNSession.create(goal="Test goal")
        
        # Verify scribe is running
        assert session.scribe_running is True


@pytest.mark.asyncio
async def test_nwtn_session_team_members_populated():
    """Test that session.team_members is non-empty list."""
    with patch("prsm.compute.nwtn.whiteboard.store.WhiteboardStore") as mock_store_cls, \
         patch("prsm.compute.nwtn.bsc.promoter.BSCPromoter") as mock_promoter_cls, \
         patch("prsm.compute.nwtn.bsc.deployment.BSCDeploymentConfig") as mock_config_cls, \
         patch("prsm.compute.nwtn.openclaw.adapter.NWTNOpenClawAdapter") as mock_adapter_cls:
        
        # Set up mocks
        mock_store = MagicMock()
        mock_store.create_session = AsyncMock(return_value=None)
        mock_store_cls.return_value = mock_store
        
        mock_promoter = MagicMock()
        mock_promoter_cls.from_config.return_value = mock_promoter
        
        mock_config_cls.auto.return_value = MagicMock()
        
        # Mock adapter with team members
        mock_adapter = MagicMock()
        mock_state = MagicMock()
        mock_state.session_id = "test-session"
        mock_state.team_members = ["coder-auth-20260327", "architect-main-20260327", "tester-api-20260327"]
        mock_state.scribe_running = True
        mock_adapter.start_session = AsyncMock(return_value=mock_state)
        mock_adapter_cls.return_value = mock_adapter
        
        from prsm.compute.nwtn.session import NWTNSession
        
        # Create session
        session = await NWTNSession.create(goal="Build auth system")
        
        # Verify team members are populated
        assert isinstance(session.team_members, list)
        assert len(session.team_members) == 3


@pytest.mark.asyncio
async def test_nwtn_session_status_dict():
    """Test that session.status() returns dict with all required keys."""
    with patch("prsm.compute.nwtn.whiteboard.store.WhiteboardStore") as mock_store_cls, \
         patch("prsm.compute.nwtn.bsc.promoter.BSCPromoter") as mock_promoter_cls, \
         patch("prsm.compute.nwtn.bsc.deployment.BSCDeploymentConfig") as mock_config_cls, \
         patch("prsm.compute.nwtn.openclaw.adapter.NWTNOpenClawAdapter") as mock_adapter_cls:
        
        # Set up mocks
        mock_store = MagicMock()
        mock_store.create_session = AsyncMock(return_value=None)
        mock_store_cls.return_value = mock_store
        
        mock_promoter = MagicMock()
        mock_promoter_cls.from_config.return_value = mock_promoter
        
        mock_config_cls.auto.return_value = MagicMock()
        
        # Mock adapter
        mock_adapter = MagicMock()
        mock_state = MagicMock()
        mock_state.session_id = "test-session"
        mock_state.goal = "Test goal"
        mock_state.status = "active"
        mock_state.team_members = ["coder-1"]
        mock_state.scribe_running = True
        mock_adapter.start_session = AsyncMock(return_value=mock_state)
        mock_adapter_cls.return_value = mock_adapter
        
        from prsm.compute.nwtn.session import NWTNSession
        
        # Create session and get status
        session = await NWTNSession.create(goal="Test goal")
        status = session.status()
        
        # Verify all required keys are present
        assert "session_id" in status
        assert "goal" in status
        assert "status" in status
        assert "team_members" in status
        assert "scribe_running" in status


@pytest.mark.asyncio
async def test_nwtn_session_close_stops_scribe():
    """Test that close() calls adapter.end_session() and ScribeAgent.stop()."""
    with patch("prsm.compute.nwtn.whiteboard.store.WhiteboardStore") as mock_store_cls, \
         patch("prsm.compute.nwtn.bsc.promoter.BSCPromoter") as mock_promoter_cls, \
         patch("prsm.compute.nwtn.bsc.deployment.BSCDeploymentConfig") as mock_config_cls, \
         patch("prsm.compute.nwtn.openclaw.adapter.NWTNOpenClawAdapter") as mock_adapter_cls:
        
        # Set up mocks
        mock_store = MagicMock()
        mock_store.create_session = AsyncMock(return_value=None)
        mock_store_cls.return_value = mock_store
        
        mock_promoter = MagicMock()
        mock_promoter_cls.from_config.return_value = mock_promoter
        
        mock_config_cls.auto.return_value = MagicMock()
        
        # Mock adapter
        mock_adapter = MagicMock()
        mock_state = MagicMock()
        mock_state.session_id = "test-session"
        mock_state.team_members = []
        mock_state.scribe_running = True
        mock_adapter.start_session = AsyncMock(return_value=mock_state)
        mock_adapter.end_session = AsyncMock(return_value=None)
        mock_adapter_cls.return_value = mock_adapter
        
        from prsm.compute.nwtn.session import NWTNSession
        
        # Create and close session
        session = await NWTNSession.create(goal="Test goal")
        await session.close()
        
        # Verify end_session was called
        mock_adapter.end_session.assert_called_once_with("test-session")


# ======================================================================
# Adapter ScribeAgent Wiring Tests
# ======================================================================

@pytest.mark.asyncio
async def test_adapter_scribe_wired_on_start(mock_store, mock_promoter, mock_meta_plan, mock_team, mock_interview_brief):
    """Test that adapter.start_session() creates ScribeAgent and starts it."""
    from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter
    
    # Patch the imports at their source modules (where they're imported FROM)
    with patch("prsm.compute.nwtn.team.InterviewSession") as mock_interview_cls, \
         patch("prsm.compute.nwtn.team.MetaPlanner") as mock_planner_cls, \
         patch("prsm.compute.nwtn.team.TeamAssembler") as mock_assembler_cls, \
         patch("prsm.compute.nwtn.team.BranchManager") as mock_branch_mgr_cls, \
         patch("prsm.compute.nwtn.whiteboard.WhiteboardMonitor") as mock_monitor_cls, \
         patch("prsm.compute.nwtn.openclaw.adapter._build_heartbeat_hook") as mock_heartbeat_fn:
        
        # Set up all mocks
        mock_interview = MagicMock()
        mock_interview.run = AsyncMock(return_value=mock_interview_brief)
        mock_interview_cls.return_value = mock_interview
        
        mock_planner = MagicMock()
        mock_planner.generate = AsyncMock(return_value=mock_meta_plan)
        mock_planner_cls.return_value = mock_planner
        
        mock_assembler = MagicMock()
        mock_assembler.assemble = AsyncMock(return_value=mock_team)
        mock_assembler_cls.return_value = mock_assembler
        
        # Mock BranchManager instance
        mock_branch_mgr_instance = MagicMock()
        mock_branch_mgr_instance.create_team_branches = AsyncMock(return_value=None)
        mock_branch_mgr_cls.return_value = mock_branch_mgr_instance
        
        mock_monitor = MagicMock()
        mock_monitor.watch_agent = MagicMock(return_value=None)
        mock_monitor.start = AsyncMock(return_value=None)
        mock_monitor.stop = AsyncMock(return_value=None)
        mock_monitor_cls.return_value = mock_monitor
        
        mock_heartbeat = MagicMock()
        mock_heartbeat.start = AsyncMock(return_value=None)
        mock_heartbeat_fn.return_value = mock_heartbeat
        
        # Create adapter
        adapter = NWTNOpenClawAdapter(
            whiteboard_store=mock_store,
            promoter=mock_promoter,
            ledger=None,
            signer=None,
            synthesizer=None,
        )
        
        # Start session
        state = await adapter.start_session(goal="Test goal")
        
        # Verify ScribeAgent was created and started
        assert adapter._scribe_agent is not None
        assert state.scribe_running is True


@pytest.mark.asyncio
async def test_adapter_end_session_stops_scribe(mock_store, mock_promoter):
    """Test that adapter.end_session() stops ScribeAgent."""
    from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter, SessionState
    
    with patch("prsm.compute.nwtn.openclaw.adapter._build_heartbeat_hook") as mock_heartbeat_fn:
        
        mock_heartbeat = MagicMock()
        mock_heartbeat.stop = AsyncMock(return_value=None)
        mock_heartbeat.synthesis_count = 0
        mock_heartbeat_fn.return_value = mock_heartbeat
        
        # Create adapter
        adapter = NWTNOpenClawAdapter(
            whiteboard_store=mock_store,
            promoter=mock_promoter,
            ledger=None,
            signer=None,
            synthesizer=None,
        )
        
        # Manually set up a mock ScribeAgent
        mock_scribe = MagicMock()
        mock_scribe.stop = AsyncMock(return_value=None)
        adapter._scribe_agent = mock_scribe
        
        # Create a session state
        state = SessionState(session_id="test-session", goal="Test")
        adapter._sessions["test-session"] = state
        adapter._monitor = MagicMock()
        adapter._monitor.stop = AsyncMock(return_value=None)
        
        # End session
        await adapter.end_session("test-session")
        
        # Verify ScribeAgent.stop() was called
        mock_scribe.stop.assert_called_once()
        
        # Verify _scribe_agent and _router are cleared
        assert adapter._scribe_agent is None
        assert adapter._router is None


@pytest.mark.asyncio
async def test_adapter_event_bus_created():
    """Test that NWTNOpenClawAdapter._event_bus is an EventBus instance."""
    from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter
    
    # Create adapter with minimal args
    adapter = NWTNOpenClawAdapter(
        whiteboard_store=MagicMock(),
        promoter=MagicMock(),
        ledger=None,
        signer=None,
        synthesizer=None,
    )
    
    # Verify _event_bus is an EventBus
    assert hasattr(adapter, "_event_bus")
    assert isinstance(adapter._event_bus, EventBus)
    
    # Verify initial state
    assert adapter._scribe_agent is None
    assert adapter._router is None


# ======================================================================
# Integration Tests
# ======================================================================

@pytest.mark.asyncio
async def test_full_session_lifecycle(mock_store, mock_promoter, mock_meta_plan, mock_team, mock_interview_brief):
    """Test full session lifecycle: create → active → close."""
    from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter
    
    # Patch the imports at their source modules
    with patch("prsm.compute.nwtn.team.InterviewSession") as mock_interview_cls, \
         patch("prsm.compute.nwtn.team.MetaPlanner") as mock_planner_cls, \
         patch("prsm.compute.nwtn.team.TeamAssembler") as mock_assembler_cls, \
         patch("prsm.compute.nwtn.team.branch_manager.BranchManager") as mock_branch_mgr_cls, \
         patch("prsm.compute.nwtn.whiteboard.WhiteboardMonitor") as mock_monitor_cls, \
         patch("prsm.compute.nwtn.openclaw.adapter._build_heartbeat_hook") as mock_heartbeat_fn:
        
        # Set up all mocks
        mock_interview = MagicMock()
        mock_interview.run = AsyncMock(return_value=mock_interview_brief)
        mock_interview_cls.return_value = mock_interview
        
        mock_planner = MagicMock()
        mock_planner.generate = AsyncMock(return_value=mock_meta_plan)
        mock_planner_cls.return_value = mock_planner
        
        mock_assembler = MagicMock()
        mock_assembler.assemble = AsyncMock(return_value=mock_team)
        mock_assembler_cls.return_value = mock_assembler
        
        # Mock BranchManager instance
        mock_branch_mgr_instance = MagicMock()
        mock_branch_mgr_instance.create_team_branches = AsyncMock(return_value=None)
        mock_branch_mgr_cls.return_value = mock_branch_mgr_instance
        
        mock_monitor = MagicMock()
        mock_monitor.watch_agent = MagicMock(return_value=None)
        mock_monitor.start = AsyncMock(return_value=None)
        mock_monitor.stop = AsyncMock(return_value=None)
        mock_monitor_cls.return_value = mock_monitor
        
        mock_heartbeat = MagicMock()
        mock_heartbeat.start = AsyncMock(return_value=None)
        mock_heartbeat.stop = AsyncMock(return_value=None)
        mock_heartbeat.synthesis_count = 0
        mock_heartbeat_fn.return_value = mock_heartbeat
        
        # Create adapter
        adapter = NWTNOpenClawAdapter(
            whiteboard_store=mock_store,
            promoter=mock_promoter,
            ledger=None,
            signer=None,
            synthesizer=None,
        )
        
        # Start session
        state = await adapter.start_session(goal="Build feature X")
        
        # Verify session is active
        assert state.status == "active"
        assert state.scribe_running is True
        assert len(state.team_members) == 2
        
        # End session
        await adapter.end_session(state.session_id)
        
        # Verify session is closed
        assert state.status == "closed"
        assert adapter._scribe_agent is None


# ======================================================================
# NWTNSession.run() Tests
# ======================================================================

@pytest.mark.asyncio
async def test_run_returns_run_result():
    """Test that run() returns RunResult with rounds_completed and converged=False when no agent_output_fn."""
    from prsm.compute.nwtn.session import NWTNSession, RunResult

    # Create mock adapter and state
    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={"pending_agents": []})

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1", "agent-2"]
    mock_state.scribe_running = True

    # Create session directly (bypass factory)
    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Run with no agent_output_fn
    result = await session.run(max_rounds=3, round_poll_interval=0.01)

    # Verify result
    assert isinstance(result, RunResult)
    assert result.rounds_completed == 3
    assert result.converged is False
    assert result.session_id == "test-session"


@pytest.mark.asyncio
async def test_run_converges_early():
    """Test that run() converges early when agent_output_fn returns TASK COMPLETE."""
    from prsm.compute.nwtn.session import NWTNSession

    # Create mock adapter
    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.convergence_summary = MagicMock(return_value={"pending_agents": []})

    # Track convergence calls - converge after round 2
    convergence_state = {"round": 0}

    def mock_is_converged(session_id):
        return convergence_state["round"] >= 2

    def mock_scan_convergence(session_id, agent_id, output, round_num):
        # Simulate convergence signal detection
        if "TASK COMPLETE" in output:
            convergence_state["round"] = round_num

    mock_adapter.is_session_converged = MagicMock(side_effect=mock_is_converged)
    mock_adapter.scan_agent_convergence = MagicMock(side_effect=mock_scan_convergence)

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1", "agent-2"]
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Agent output function that returns TASK COMPLETE
    async def agent_output_fn(agent_id, round_num):
        return "TASK COMPLETE - All work finished"

    # Run with max_rounds=5, should converge by round 2
    result = await session.run(
        max_rounds=5,
        round_poll_interval=0.01,
        agent_output_fn=agent_output_fn,
    )

    assert result.converged is True
    assert result.rounds_completed < 5


@pytest.mark.asyncio
async def test_run_respects_max_rounds():
    """Test that run() respects max_rounds when convergence never happens."""
    from prsm.compute.nwtn.session import NWTNSession

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={"pending_agents": ["agent-1"]})
    mock_adapter.scan_agent_convergence = MagicMock(return_value=False)

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    async def never_converge_fn(agent_id, round_num):
        return "Still working..."

    result = await session.run(
        max_rounds=2,
        round_poll_interval=0.01,
        agent_output_fn=never_converge_fn,
    )

    assert result.rounds_completed == 2
    assert result.converged is False


@pytest.mark.asyncio
async def test_run_result_has_elapsed_seconds():
    """Test that RunResult.elapsed_seconds is positive."""
    from prsm.compute.nwtn.session import NWTNSession

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={})

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = []
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    result = await session.run(max_rounds=2, round_poll_interval=0.01)

    assert result.elapsed_seconds > 0


@pytest.mark.asyncio
async def test_run_status_tracks_rounds():
    """Test that status() returns rounds_completed after run()."""
    from prsm.compute.nwtn.session import NWTNSession

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={})

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = []
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    await session.run(max_rounds=5, round_poll_interval=0.01)

    status = session.status()
    assert status["rounds_completed"] == 5
    assert status["converged"] is False


@pytest.mark.asyncio
async def test_run_agent_output_error_does_not_crash():
    """Test that run() continues when agent_output_fn raises an exception."""
    from prsm.compute.nwtn.session import NWTNSession

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={})

    mock_state = MagicMock()
    mock_state.session_id = "test-session"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    call_count = {"count": 0}

    async def failing_agent_fn(agent_id, round_num):
        call_count["count"] += 1
        raise RuntimeError("Agent crashed!")

    result = await session.run(
        max_rounds=3,
        round_poll_interval=0.01,
        agent_output_fn=failing_agent_fn,
    )

    # Should complete all rounds despite errors
    assert result.rounds_completed == 3
    assert result.converged is False
    # Verify agent_output_fn was called each round
    assert call_count["count"] == 3


@pytest.mark.asyncio
async def test_run_result_session_id_matches():
    """Test that RunResult.session_id matches session.session_id."""
    from prsm.compute.nwtn.session import NWTNSession

    mock_adapter = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock(return_value=None)
    mock_adapter.is_session_converged = MagicMock(return_value=False)
    mock_adapter.convergence_summary = MagicMock(return_value={})

    mock_state = MagicMock()
    mock_state.session_id = "unique-session-123"
    mock_state.goal = "Test goal"
    mock_state.status = "active"
    mock_state.team_members = []
    mock_state.scribe_running = True

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    result = await session.run(max_rounds=1, round_poll_interval=0.01)

    assert result.session_id == session.session_id
    assert result.session_id == "unique-session-123"