"""
Comprehensive tests for NWTNTraceLogger — FEAT-20260331-001.

Tests cover:
- Directory structure creation
- Config logging
- Round start/end with file creation
- Agent output recording (with 500 char cap)
- Quality report recording and chunk counting
- Context pressure recording with reset tracking
- Feedback recording
- Convergence recording
- Session finalization
- Backward compatibility (trace_logger=None)
- JSON validity and roundtrip
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.nwtn.trace_logger import (
    HarnessConfig,
    NWTNTraceLogger,
    RoundTrace,
    SessionMeta,
    create_trace_logger,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_traces_dir():
    """Create a temporary directory for trace outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def trace_logger(temp_traces_dir):
    """Create a trace logger with a temporary traces directory."""
    return NWTNTraceLogger(
        session_id="test-session-001",
        goal="Test goal for trace logger",
        traces_dir=temp_traces_dir,
    )


# ============================================================================
# Test: Directory Structure
# ============================================================================


def test_creates_directory_structure(temp_traces_dir):
    """NWTNTraceLogger creates .nwtn_traces/{session_id}/rounds/ structure."""
    logger = NWTNTraceLogger(
        session_id="abc123",
        goal="Test",
        traces_dir=temp_traces_dir,
    )

    # Check directory structure was created
    session_dir = temp_traces_dir / "abc123"
    rounds_dir = session_dir / "rounds"

    assert session_dir.exists()
    assert session_dir.is_dir()
    assert rounds_dir.exists()
    assert rounds_dir.is_dir()


def test_default_traces_dir():
    """NWTNTraceLogger defaults to .nwtn_traces in cwd."""
    logger = NWTNTraceLogger(
        session_id="test",
        goal="Test",
    )
    assert logger.traces_dir == Path(".nwtn_traces")


# ============================================================================
# Test: Config Logging
# ============================================================================


def test_log_config_writes_valid_json(trace_logger, temp_traces_dir):
    """log_config() writes harness_config.json with valid JSON."""
    config = HarnessConfig(
        quality_threshold=0.42,
        max_rounds=15,
        round_poll_interval=3.0,
        context_pressure_warning_pct=0.65,
        context_pressure_critical_pct=0.80,
        context_pressure_hard_limit_pct=0.92,
        extra={"custom_param": "value"},
    )

    trace_logger.log_config(config)

    config_path = temp_traces_dir / "test-session-001" / "harness_config.json"
    assert config_path.exists()

    # Verify it's valid JSON
    with open(config_path) as f:
        loaded = json.load(f)

    assert loaded["quality_threshold"] == 0.42
    assert loaded["max_rounds"] == 15
    assert loaded["round_poll_interval"] == 3.0
    assert loaded["context_pressure_warning_pct"] == 0.65
    assert loaded["context_pressure_critical_pct"] == 0.80
    assert loaded["context_pressure_hard_limit_pct"] == 0.92
    assert loaded["extra"]["custom_param"] == "value"


def test_log_config_default_values(trace_logger, temp_traces_dir):
    """log_config() uses default HarnessConfig values correctly."""
    config = HarnessConfig()
    trace_logger.log_config(config)

    config_path = temp_traces_dir / "test-session-001" / "harness_config.json"
    with open(config_path) as f:
        loaded = json.load(f)

    assert loaded["quality_threshold"] == 0.35
    assert loaded["max_rounds"] == 20
    assert loaded["round_poll_interval"] == 5.0


# ============================================================================
# Test: Round Start/End
# ============================================================================


def test_start_round_creates_round_trace(trace_logger):
    """start_round() initializes a RoundTrace in memory."""
    trace_logger.start_round(1)

    assert 1 in trace_logger._current_rounds
    rt = trace_logger._current_rounds[1]
    assert isinstance(rt, RoundTrace)
    assert rt.round_number == 1
    assert rt.timestamp is not None


def test_end_round_writes_json_file(trace_logger, temp_traces_dir):
    """end_round() writes round_{N:03d}.json to disk."""
    trace_logger.start_round(1)
    trace_logger.end_round(1)

    round_path = temp_traces_dir / "test-session-001" / "rounds" / "round_001.json"
    assert round_path.exists()

    # Verify it's valid JSON
    with open(round_path) as f:
        loaded = json.load(f)

    assert loaded["round_number"] == 1
    assert "timestamp" in loaded


def test_end_round_with_zero_padding(trace_logger, temp_traces_dir):
    """Round files are zero-padded to 3 digits."""
    trace_logger.start_round(5)
    trace_logger.end_round(5)

    round_path = temp_traces_dir / "test-session-001" / "rounds" / "round_005.json"
    assert round_path.exists()

    trace_logger.start_round(42)
    trace_logger.end_round(42)

    round_path = temp_traces_dir / "test-session-001" / "rounds" / "round_042.json"
    assert round_path.exists()


def test_end_round_no_trace_exists(trace_logger, temp_traces_dir):
    """end_round() is a no-op if no trace exists for the round."""
    # Should not raise
    trace_logger.end_round(999)

    # No file should be created
    round_path = temp_traces_dir / "test-session-001" / "rounds" / "round_999.json"
    assert not round_path.exists()


# ============================================================================
# Test: Agent Output Recording
# ============================================================================


def test_record_agent_output_stores_output(trace_logger):
    """record_agent_output() stores agent output in round trace."""
    trace_logger.start_round(1)
    trace_logger.record_agent_output("coder-1", 1, "This is the agent output")

    rt = trace_logger._current_rounds[1]
    assert "coder-1" in rt.agent_outputs
    assert rt.agent_outputs["coder-1"] == "This is the agent output"


def test_record_agent_output_caps_at_500_chars(trace_logger):
    """record_agent_output() caps output at 500 characters."""
    trace_logger.start_round(1)

    long_output = "x" * 1000
    trace_logger.record_agent_output("coder-1", 1, long_output)

    rt = trace_logger._current_rounds[1]
    assert len(rt.agent_outputs["coder-1"]) == 500


def test_record_agent_output_no_round(trace_logger):
    """record_agent_output() is a no-op if round not started."""
    # Should not raise
    trace_logger.record_agent_output("coder-1", 1, "output")


# ============================================================================
# Test: Quality Report Recording
# ============================================================================


def test_record_quality_report_increments_counts(trace_logger):
    """record_quality_report() increments chunks_evaluated and chunks_promoted."""
    trace_logger.start_round(1)

    report1 = {"passed": True, "score": 0.85}
    report2 = {"passed": False, "score": 0.25}
    report3 = {"passed": True, "score": 0.90}

    trace_logger.record_quality_report(1, report1)
    trace_logger.record_quality_report(1, report2)
    trace_logger.record_quality_report(1, report3)

    rt = trace_logger._current_rounds[1]
    assert rt.chunks_evaluated == 3
    assert rt.chunks_promoted == 2  # only report1 and report3 passed


def test_record_quality_report_stores_reports(trace_logger):
    """record_quality_report() stores full report dicts."""
    trace_logger.start_round(1)

    report = {"passed": True, "score": 0.85, "agent_id": "tester-1"}
    trace_logger.record_quality_report(1, report)

    rt = trace_logger._current_rounds[1]
    assert len(rt.quality_reports) == 1
    assert rt.quality_reports[0] == report


def test_record_quality_report_no_round(trace_logger):
    """record_quality_report() is a no-op if round not started."""
    # Should not raise
    trace_logger.record_quality_report(1, {"passed": True})


# ============================================================================
# Test: Context Pressure Recording
# ============================================================================


def test_record_context_pressure_stores_tokens(trace_logger):
    """record_context_pressure() stores token counts and pressure levels."""
    trace_logger.start_round(1)

    trace_logger.record_context_pressure(1, "coder-1", 5000, level="WARNING")

    rt = trace_logger._current_rounds[1]
    assert rt.token_counts["coder-1"] == 5000
    assert rt.pressure_levels["coder-1"] == "WARNING"


def test_record_context_pressure_reset_triggered(trace_logger):
    """record_context_pressure() adds agent to context_resets when reset_triggered=True."""
    trace_logger.start_round(1)

    trace_logger.record_context_pressure(1, "coder-1", 10000, level="CRITICAL", reset_triggered=True)
    trace_logger.record_context_pressure(1, "coder-2", 8000, level="WARNING", reset_triggered=False)

    rt = trace_logger._current_rounds[1]
    assert "coder-1" in rt.context_resets
    assert "coder-2" not in rt.context_resets


def test_record_context_pressure_default_level(trace_logger):
    """record_context_pressure() defaults to level='OK'."""
    trace_logger.start_round(1)

    trace_logger.record_context_pressure(1, "coder-1", 1000)

    rt = trace_logger._current_rounds[1]
    assert rt.pressure_levels["coder-1"] == "OK"


def test_record_context_pressure_no_round(trace_logger):
    """record_context_pressure() is a no-op if round not started."""
    # Should not raise
    trace_logger.record_context_pressure(1, "coder-1", 5000)


# ============================================================================
# Test: Feedback Recording
# ============================================================================


def test_record_feedback_stores_count_and_targets(trace_logger):
    """record_feedback() stores count and target list."""
    trace_logger.start_round(1)

    trace_logger.record_feedback(1, 3, ["coder-1", "coder-2", "architect-main"])

    rt = trace_logger._current_rounds[1]
    assert rt.feedback_published == 3
    assert "coder-1" in rt.feedback_targets
    assert "coder-2" in rt.feedback_targets
    assert "architect-main" in rt.feedback_targets


def test_record_feedback_no_round(trace_logger):
    """record_feedback() is a no-op if round not started."""
    # Should not raise
    trace_logger.record_feedback(1, 5, ["agent-1"])


# ============================================================================
# Test: Convergence Recording
# ============================================================================


def test_record_convergence_stores_pending_agents(trace_logger):
    """record_convergence() stores pending agents list."""
    trace_logger.start_round(1)

    trace_logger.record_convergence(1, ["coder-1", "tester-1"], converged=False)

    rt = trace_logger._current_rounds[1]
    assert rt.pending_agents == ["coder-1", "tester-1"]
    assert rt.converged is False


def test_record_convergence_marks_converged(trace_logger):
    """record_convergence() sets converged=True when provided."""
    trace_logger.start_round(1)

    trace_logger.record_convergence(1, [], converged=True)

    rt = trace_logger._current_rounds[1]
    assert rt.converged is True


def test_record_convergence_no_round(trace_logger):
    """record_convergence() is a no-op if round not started."""
    # Should not raise
    trace_logger.record_convergence(1, [])


# ============================================================================
# Test: Session Finalization
# ============================================================================


def test_finalize_writes_session_meta(trace_logger, temp_traces_dir):
    """finalize() writes session_meta.json with all fields."""
    trace_logger.set_team(["coder-1", "architect-main"])

    trace_logger.finalize(
        converged=True,
        rounds_completed=5,
        context_resets_triggered=2,
        feedback_rounds_completed=3,
        elapsed_seconds=42.5,
        final_status="completed",
    )

    meta_path = temp_traces_dir / "test-session-001" / "session_meta.json"
    assert meta_path.exists()

    with open(meta_path) as f:
        loaded = json.load(f)

    assert loaded["session_id"] == "test-session-001"
    assert loaded["goal"] == "Test goal for trace logger"
    assert loaded["converged"] is True
    assert loaded["rounds_completed"] == 5
    assert loaded["context_resets_triggered"] == 2
    assert loaded["feedback_rounds_completed"] == 3
    assert loaded["elapsed_seconds"] == 42.5
    assert loaded["final_status"] == "completed"
    assert "started_at" in loaded
    assert "finished_at" in loaded
    assert "coder-1" in loaded["team_members"]
    assert "architect-main" in loaded["team_members"]


def test_finalize_finished_at_timestamp(trace_logger, temp_traces_dir):
    """finalize() sets finished_at timestamp."""
    import time

    trace_logger.finalize(
        converged=False,
        rounds_completed=1,
        context_resets_triggered=0,
        feedback_rounds_completed=0,
        elapsed_seconds=1.0,
        final_status="timeout",
    )

    meta_path = temp_traces_dir / "test-session-001" / "session_meta.json"
    with open(meta_path) as f:
        loaded = json.load(f)

    assert loaded["finished_at"] is not None


# ============================================================================
# Test: Set Team
# ============================================================================


def test_set_team(trace_logger):
    """set_team() stores team members in meta."""
    trace_logger.set_team(["agent-1", "agent-2", "agent-3"])

    assert trace_logger._meta.team_members == ["agent-1", "agent-2", "agent-3"]


def test_set_team_copies_list(trace_logger):
    """set_team() copies the list to avoid mutation issues."""
    team = ["agent-1", "agent-2"]
    trace_logger.set_team(team)

    team.append("agent-3")

    # _meta should not have agent-3
    assert "agent-3" not in trace_logger._meta.team_members


# ============================================================================
# Test: Round Files Valid JSON Roundtrip
# ============================================================================


def test_round_files_are_valid_json_roundtrip(trace_logger, temp_traces_dir):
    """Round files can be loaded back as RoundTrace data."""
    trace_logger.start_round(1)
    trace_logger.record_agent_output("coder-1", 1, "output")
    trace_logger.record_quality_report(1, {"passed": True, "score": 0.8})
    trace_logger.record_context_pressure(1, "coder-1", 5000, level="WARNING")
    trace_logger.record_feedback(1, 2, ["coder-2"])
    trace_logger.record_convergence(1, ["tester-1"], converged=False)
    trace_logger.end_round(1)

    round_path = temp_traces_dir / "test-session-001" / "rounds" / "round_001.json"
    with open(round_path) as f:
        loaded = json.load(f)

    # Verify all fields survived the roundtrip
    assert loaded["round_number"] == 1
    assert loaded["agent_outputs"]["coder-1"] == "output"
    assert loaded["chunks_evaluated"] == 1
    assert loaded["chunks_promoted"] == 1
    assert loaded["token_counts"]["coder-1"] == 5000
    assert loaded["pressure_levels"]["coder-1"] == "WARNING"
    assert loaded["feedback_published"] == 2
    assert "coder-2" in loaded["feedback_targets"]
    assert loaded["pending_agents"] == ["tester-1"]
    assert loaded["converged"] is False


# ============================================================================
# Test: Full Session Trace Flow
# ============================================================================


def test_full_session_trace_flow(temp_traces_dir):
    """Integration test: full session trace from start to finish."""
    logger = NWTNTraceLogger(
        session_id="integration-test",
        goal="Full session trace test",
        traces_dir=temp_traces_dir,
    )

    # Log config
    config = HarnessConfig(quality_threshold=0.5, max_rounds=10)
    logger.log_config(config)

    # Set team
    logger.set_team(["coder-1", "architect-main", "tester-1"])

    # Round 1
    logger.start_round(1)
    logger.record_agent_output("coder-1", 1, "Code output here")
    logger.record_agent_output("architect-main", 1, "Architecture decisions")
    logger.record_quality_report(1, {"passed": True, "score": 0.75})
    logger.record_context_pressure(1, "coder-1", 3000, level="OK")
    logger.record_convergence(1, ["tester-1"], converged=False)
    logger.end_round(1)

    # Round 2
    logger.start_round(2)
    logger.record_agent_output("tester-1", 2, "Test results")
    logger.record_quality_report(2, {"passed": True, "score": 0.90})
    logger.record_context_pressure(2, "coder-1", 8000, level="WARNING", reset_triggered=True)
    logger.record_feedback(2, 1, ["coder-1"])
    logger.record_convergence(2, [], converged=True)
    logger.end_round(2)

    # Finalize
    logger.finalize(
        converged=True,
        rounds_completed=2,
        context_resets_triggered=1,
        feedback_rounds_completed=1,
        elapsed_seconds=12.5,
        final_status="completed",
    )

    # Verify all files exist
    session_dir = temp_traces_dir / "integration-test"
    assert (session_dir / "harness_config.json").exists()
    assert (session_dir / "session_meta.json").exists()
    assert (session_dir / "rounds" / "round_001.json").exists()
    assert (session_dir / "rounds" / "round_002.json").exists()

    # Verify session_meta
    with open(session_dir / "session_meta.json") as f:
        meta = json.load(f)
    assert meta["converged"] is True
    assert meta["rounds_completed"] == 2
    assert meta["context_resets_triggered"] == 1

    # Verify round 2 has context reset
    with open(session_dir / "rounds" / "round_002.json") as f:
        round2 = json.load(f)
    assert "coder-1" in round2["context_resets"]


# ============================================================================
# Test: Backward Compatibility (trace_logger=None)
# ============================================================================


@pytest.mark.asyncio
async def test_trace_logger_none_backward_compat():
    """
    NWTNSession.run() with trace_logger=None should not break anything.
    
    This test uses the existing e2e smoke test pattern to verify
    that trace_logger=None (the default) works correctly.
    """
    from prsm.compute.nwtn.session import NWTNSession

    # Mock the adapter and state
    mock_adapter = MagicMock()
    mock_adapter.is_session_converged.return_value = True  # Immediate convergence
    mock_adapter.convergence_summary.return_value = {"pending_agents": []}
    mock_adapter.advance_bsc_round = AsyncMock()

    mock_state = MagicMock()
    mock_state.session_id = "test-backward-compat"
    mock_state.goal = "Test backward compatibility"
    mock_state.status = "completed"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = False

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Run with trace_logger=None (default)
    result = await session.run(
        max_rounds=5,
        round_poll_interval=0.01,
        trace_logger=None,  # Explicit None
    )

    assert result.session_id == "test-backward-compat"
    assert result.converged is True
    assert result.trace_path is None  # Should be None when trace_logger is None


@pytest.mark.asyncio
async def test_trace_logger_with_session_integration(temp_traces_dir):
    """
    NWTNSession.run() with trace_logger creates trace files.
    """
    from prsm.compute.nwtn.session import NWTNSession
    from prsm.compute.nwtn.trace_logger import HarnessConfig

    # Mock the adapter and state
    mock_adapter = MagicMock()
    mock_adapter.is_session_converged.return_value = False
    mock_adapter.convergence_summary.return_value = {"pending_agents": ["agent-1"]}
    mock_adapter.scan_agent_convergence = MagicMock()
    mock_adapter.advance_bsc_round = AsyncMock()

    mock_state = MagicMock()
    mock_state.session_id = "test-with-logger"
    mock_state.goal = "Test with trace logger"
    mock_state.status = "completed"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = False

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Create trace logger
    trace_logger = NWTNTraceLogger(
        session_id="test-with-logger",
        goal="Test with trace logger",
        traces_dir=temp_traces_dir,
    )
    trace_logger.log_config(HarnessConfig())

    # Mock agent output function
    async def mock_output_fn(agent_id, round_num):
        return f"Output from {agent_id} in round {round_num}"

    # Run for 2 rounds (won't converge)
    result = await session.run(
        max_rounds=2,
        round_poll_interval=0.01,
        agent_output_fn=mock_output_fn,
        trace_logger=trace_logger,
    )

    assert result.session_id == "test-with-logger"
    assert result.rounds_completed == 2
    assert result.trace_path is not None
    assert str(temp_traces_dir) in result.trace_path

    # Verify trace files exist
    session_dir = temp_traces_dir / "test-with-logger"
    assert (session_dir / "session_meta.json").exists()
    assert (session_dir / "rounds" / "round_001.json").exists()
    assert (session_dir / "rounds" / "round_002.json").exists()


# ============================================================================
# Test: HarnessConfig Expansion (FEAT-20260331-002)
# ============================================================================


def test_harness_config_defaults():
    """Verify default field values for expanded HarnessConfig."""
    config = HarnessConfig()
    assert config.quality_threshold == 0.35
    assert config.kl_epsilon == 0.1
    assert config.similarity_threshold == 0.85
    assert config.max_rounds == 20
    assert config.round_poll_interval == 5.0
    assert config.context_pressure_warning_pct == 0.70
    assert config.context_pressure_critical_pct == 0.85
    assert config.context_pressure_hard_limit_pct == 0.95
    assert config.feedback_quality_threshold == 0.80
    assert config.config_version == "1.0"
    assert config.extra == {}


def test_harness_config_roundtrip():
    """HarnessConfig.from_dict(config.to_dict()) produces identical values."""
    config = HarnessConfig(
        quality_threshold=0.5,
        kl_epsilon=0.05,
        similarity_threshold=0.9,
        max_rounds=30,
        round_poll_interval=10.0,
        context_pressure_warning_pct=0.6,
        context_pressure_critical_pct=0.75,
        context_pressure_hard_limit_pct=0.90,
        feedback_quality_threshold=0.85,
        config_version="2.0",
        extra={"custom": "value"},
    )
    restored = HarnessConfig.from_dict(config.to_dict())
    assert restored.quality_threshold == 0.5
    assert restored.kl_epsilon == 0.05
    assert restored.similarity_threshold == 0.9
    assert restored.max_rounds == 30
    assert restored.round_poll_interval == 10.0
    assert restored.context_pressure_warning_pct == 0.6
    assert restored.context_pressure_critical_pct == 0.75
    assert restored.context_pressure_hard_limit_pct == 0.90
    assert restored.feedback_quality_threshold == 0.85
    assert restored.config_version == "2.0"
    assert restored.extra == {"custom": "value"}


def test_harness_config_from_components_no_components():
    """from_components() works with all None args."""
    config = HarnessConfig.from_components(
        pipeline=None,
        context_monitor=None,
        max_rounds=15,
        round_poll_interval=2.5,
    )
    assert config.max_rounds == 15
    assert config.round_poll_interval == 2.5
    # All other fields should be defaults
    assert config.quality_threshold == 0.35
    assert config.kl_epsilon == 0.1
    assert config.similarity_threshold == 0.85


def test_harness_config_from_dict_ignores_unknown_keys():
    """Extra keys in dict don't raise errors."""
    config = HarnessConfig.from_dict({
        "quality_threshold": 0.42,
        "unknown_key": "should_be_ignored",
        "another_unknown": 123,
    })
    assert config.quality_threshold == 0.42
    # Should not raise, and unknown keys should be ignored


def test_create_trace_logger_writes_config(temp_traces_dir):
    """create_trace_logger(..., harness_config=HarnessConfig(...)) writes harness_config.json."""
    config = HarnessConfig(quality_threshold=0.5, max_rounds=10)
    tl = create_trace_logger(
        session_id="test-create",
        goal="Test create_trace_logger",
        traces_dir=temp_traces_dir,
        harness_config=config,
    )

    config_path = temp_traces_dir / "test-create" / "harness_config.json"
    assert config_path.exists()

    with open(config_path) as f:
        loaded = json.load(f)

    assert loaded["quality_threshold"] == 0.5
    assert loaded["max_rounds"] == 10


@pytest.mark.asyncio
async def test_session_run_writes_default_config_when_trace_logger_provided(temp_traces_dir):
    """When trace_logger is provided but harness_config is None, harness_config.json is still written."""
    from prsm.compute.nwtn.session import NWTNSession
    from prsm.compute.nwtn.trace_logger import NWTNTraceLogger

    # Mock the adapter and state
    mock_adapter = MagicMock()
    mock_adapter.is_session_converged.return_value = True  # Immediate convergence
    mock_adapter.convergence_summary.return_value = {"pending_agents": []}
    mock_adapter.advance_bsc_round = AsyncMock()

    mock_state = MagicMock()
    mock_state.session_id = "test-default-config"
    mock_state.goal = "Test default config"
    mock_state.status = "completed"
    mock_state.team_members = ["agent-1"]
    mock_state.scribe_running = False

    session = NWTNSession(adapter=mock_adapter, session_state=mock_state)

    # Create trace logger without providing harness_config
    trace_logger = NWTNTraceLogger(
        session_id="test-default-config",
        goal="Test default config",
        traces_dir=temp_traces_dir,
    )

    # Run with trace_logger but no harness_config
    result = await session.run(
        max_rounds=5,
        round_poll_interval=0.01,
        trace_logger=trace_logger,
    )

    assert result.session_id == "test-default-config"

    # Verify harness_config.json was written with default values
    config_path = temp_traces_dir / "test-default-config" / "harness_config.json"
    assert config_path.exists()

    with open(config_path) as f:
        loaded = json.load(f)

    assert loaded["max_rounds"] == 5
    assert loaded["round_poll_interval"] == 0.01
    assert loaded["quality_threshold"] == 0.35  # default
