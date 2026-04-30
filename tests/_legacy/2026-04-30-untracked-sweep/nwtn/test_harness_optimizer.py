"""
Tests for HarnessOptimizer — Meta-Harness outer loop for NWTN harness optimization.
"""
import json
import pytest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

from prsm.compute.nwtn.trace_logger import HarnessConfig, SessionMeta
from prsm.compute.nwtn.harness_optimizer import (
    HarnessOptimizer,
    OptimizationHistory,
    SessionOutcome,
    ParetoPoint,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_traces_dir(tmp_path):
    """Create a temporary traces directory."""
    return tmp_path / ".nwtn_traces"


@pytest.fixture
def sample_harness_config():
    """Sample harness config for testing."""
    return HarnessConfig(
        quality_threshold=0.35,
        kl_epsilon=0.1,
        similarity_threshold=0.85,
        max_rounds=20,
        round_poll_interval=5.0,
        context_pressure_warning_pct=0.70,
        context_pressure_critical_pct=0.85,
        context_pressure_hard_limit_pct=0.95,
        feedback_quality_threshold=0.80,
    )


@pytest.fixture
def sample_session_meta():
    """Sample session metadata for testing."""
    return SessionMeta(
        session_id="test-session-12345678",
        goal="Test optimization",
        started_at="2026-03-31T10:00:00Z",
        finished_at="2026-03-31T10:30:00Z",
        converged=True,
        rounds_completed=5,
        context_resets_triggered=1,
        feedback_rounds_completed=2,
        elapsed_seconds=1800.0,
        final_status="completed",
        team_members=["agent-1", "agent-2"],
    )


# =============================================================================
# SessionOutcome Tests
# =============================================================================

def test_efficiency_score_zero_rounds():
    """efficiency_score returns 0.0 when rounds_completed is 0."""
    config = HarnessConfig()
    meta = SessionMeta(
        session_id="test",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        rounds_completed=0,
    )
    outcome = SessionOutcome(session_id="test", config=config, meta=meta)
    assert outcome.efficiency_score == 0.0


def test_efficiency_score_converged_no_resets():
    """Converged session with no resets has higher score than non-converged."""
    config = HarnessConfig()
    
    # Converged, no resets
    meta_converged = SessionMeta(
        session_id="conv",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
        context_resets_triggered=0,
    )
    outcome_converged = SessionOutcome(
        session_id="conv",
        config=config,
        meta=meta_converged,
        avg_chunks_promoted_per_round=2.0,
        context_reset_rate=0.0,
    )
    
    # Non-converged, some resets
    meta_not_converged = SessionMeta(
        session_id="notconv",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=False,
        rounds_completed=5,
        context_resets_triggered=2,
    )
    outcome_not_converged = SessionOutcome(
        session_id="notconv",
        config=config,
        meta=meta_not_converged,
        avg_chunks_promoted_per_round=2.0,
        context_reset_rate=0.4,
    )
    
    assert outcome_converged.efficiency_score > outcome_not_converged.efficiency_score


def test_efficiency_score_ordering():
    """Verify better outcomes score higher."""
    config = HarnessConfig()
    
    # Best: converged, no resets, high promotion
    meta_best = SessionMeta(
        session_id="best",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
        context_resets_triggered=0,
    )
    outcome_best = SessionOutcome(
        session_id="best",
        config=config,
        meta=meta_best,
        avg_chunks_promoted_per_round=3.0,
        context_reset_rate=0.0,
    )
    
    # Medium: converged, some resets
    meta_medium = SessionMeta(
        session_id="medium",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
        context_resets_triggered=1,
    )
    outcome_medium = SessionOutcome(
        session_id="medium",
        config=config,
        meta=meta_medium,
        avg_chunks_promoted_per_round=3.0,
        context_reset_rate=0.2,
    )
    
    # Worst: not converged, high resets
    meta_worst = SessionMeta(
        session_id="worst",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=False,
        rounds_completed=5,
        context_resets_triggered=3,
    )
    outcome_worst = SessionOutcome(
        session_id="worst",
        config=config,
        meta=meta_worst,
        avg_chunks_promoted_per_round=1.0,
        context_reset_rate=0.6,
    )
    
    assert outcome_best.efficiency_score > outcome_medium.efficiency_score
    assert outcome_medium.efficiency_score > outcome_worst.efficiency_score


# =============================================================================
# OptimizationHistory Tests
# =============================================================================

def test_optimization_history_best_outcome_empty():
    """best_outcome returns None when no outcomes exist."""
    history = OptimizationHistory()
    assert history.best_outcome is None


def test_optimization_history_best_outcome_returns_max():
    """best_outcome returns the outcome with highest efficiency_score."""
    config = HarnessConfig()
    
    meta1 = SessionMeta(
        session_id="low",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
        context_resets_triggered=0,
    )
    outcome1 = SessionOutcome(
        session_id="low",
        config=config,
        meta=meta1,
        avg_chunks_promoted_per_round=1.0,
        context_reset_rate=0.0,
    )
    
    meta2 = SessionMeta(
        session_id="high",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
        context_resets_triggered=0,
    )
    outcome2 = SessionOutcome(
        session_id="high",
        config=config,
        meta=meta2,
        avg_chunks_promoted_per_round=5.0,
        context_reset_rate=0.0,
    )
    
    history = OptimizationHistory(outcomes=[outcome1, outcome2])
    assert history.best_outcome.session_id == "high"


def test_optimization_history_to_prompt_context_empty():
    """to_prompt_context returns default message when no outcomes."""
    history = OptimizationHistory()
    result = history.to_prompt_context()
    assert result == "No prior sessions. Use default HarnessConfig values."


def test_optimization_history_to_prompt_context_with_data():
    """to_prompt_context contains session ids and config values."""
    config = HarnessConfig(quality_threshold=0.40, kl_epsilon=0.15)
    meta = SessionMeta(
        session_id="abc12345def",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=3,
        context_resets_triggered=1,
        feedback_rounds_completed=1,
    )
    outcome = SessionOutcome(
        session_id="abc12345def",
        config=config,
        meta=meta,
        avg_chunks_promoted_per_round=2.5,
        context_reset_rate=0.33,
    )
    
    history = OptimizationHistory(outcomes=[outcome])
    result = history.to_prompt_context()
    
    assert "abc12345" in result
    assert "quality_threshold=0.4" in result
    assert "kl_epsilon=0.15" in result
    assert "efficiency_score:" in result
    assert "converged: True" in result


# =============================================================================
# HarnessOptimizer.load_history Tests
# =============================================================================

def test_load_history_empty_dir(temp_traces_dir):
    """load_history() on nonexistent dir returns empty history."""
    optimizer = HarnessOptimizer(traces_dir=temp_traces_dir)
    history = optimizer.load_history()
    
    assert len(history.outcomes) == 0
    assert len(history.pareto_frontier) == 0


def test_load_history_single_session(temp_traces_dir, sample_harness_config, sample_session_meta):
    """Create fake trace files, verify session loads correctly."""
    session_dir = temp_traces_dir / sample_session_meta.session_id
    session_dir.mkdir(parents=True)
    
    # Write harness_config.json
    config_path = session_dir / "harness_config.json"
    config_path.write_text(json.dumps(asdict(sample_harness_config)))
    
    # Write session_meta.json
    meta_path = session_dir / "session_meta.json"
    meta_path.write_text(json.dumps(asdict(sample_session_meta)))
    
    optimizer = HarnessOptimizer(traces_dir=temp_traces_dir)
    history = optimizer.load_history()
    
    assert len(history.outcomes) == 1
    assert history.outcomes[0].session_id == sample_session_meta.session_id
    assert history.outcomes[0].config.quality_threshold == sample_harness_config.quality_threshold
    assert history.outcomes[0].meta.converged == sample_session_meta.converged


def test_load_history_skips_incomplete_sessions(temp_traces_dir, sample_harness_config, sample_session_meta):
    """Sessions missing files are skipped."""
    session_dir = temp_traces_dir / sample_session_meta.session_id
    session_dir.mkdir(parents=True)
    
    # Only write session_meta.json, not harness_config.json
    meta_path = session_dir / "session_meta.json"
    meta_path.write_text(json.dumps(asdict(sample_session_meta)))
    
    optimizer = HarnessOptimizer(traces_dir=temp_traces_dir)
    history = optimizer.load_history()
    
    assert len(history.outcomes) == 0


def test_build_outcome_with_rounds(temp_traces_dir, sample_harness_config, sample_session_meta):
    """Create round files with chunks_promoted, verify avg_chunks_promoted_per_round."""
    session_dir = temp_traces_dir / sample_session_meta.session_id
    session_dir.mkdir(parents=True)
    
    # Write harness_config.json
    config_path = session_dir / "harness_config.json"
    config_path.write_text(json.dumps(asdict(sample_harness_config)))
    
    # Write session_meta.json
    meta_path = session_dir / "session_meta.json"
    meta_path.write_text(json.dumps(asdict(sample_session_meta)))
    
    # Write round files
    rounds_dir = session_dir / "rounds"
    rounds_dir.mkdir()
    
    round1 = {"round_number": 1, "chunks_promoted": 3}
    round2 = {"round_number": 2, "chunks_promoted": 5}
    round3 = {"round_number": 3, "chunks_promoted": 4}
    
    (rounds_dir / "round_001.json").write_text(json.dumps(round1))
    (rounds_dir / "round_002.json").write_text(json.dumps(round2))
    (rounds_dir / "round_003.json").write_text(json.dumps(round3))
    
    optimizer = HarnessOptimizer(traces_dir=temp_traces_dir)
    history = optimizer.load_history()
    
    # (3 + 5 + 4) / 3 = 4.0
    assert len(history.outcomes) == 1
    assert history.outcomes[0].avg_chunks_promoted_per_round == 4.0


# =============================================================================
# HarnessOptimizer._compute_pareto Tests
# =============================================================================

def test_compute_pareto_single_point():
    """Single outcome → single pareto point."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig()
    meta = SessionMeta(
        session_id="single",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome = SessionOutcome(
        session_id="single",
        config=config,
        meta=meta,
        avg_chunks_promoted_per_round=2.0,
        context_reset_rate=0.1,
    )
    
    pareto = optimizer._compute_pareto([outcome])
    assert len(pareto) == 1
    assert pareto[0].session_id == "single"


def test_compute_pareto_dominated_removed():
    """Dominated point is excluded from frontier."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig()
    
    # Dominated: lower efficiency, higher reset rate
    meta_dominated = SessionMeta(
        session_id="dominated",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=False,
        rounds_completed=5,
    )
    outcome_dominated = SessionOutcome(
        session_id="dominated",
        config=config,
        meta=meta_dominated,
        avg_chunks_promoted_per_round=1.0,
        context_reset_rate=0.5,
    )
    
    # Dominating: higher efficiency, lower reset rate
    meta_dominating = SessionMeta(
        session_id="dominating",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome_dominating = SessionOutcome(
        session_id="dominating",
        config=config,
        meta=meta_dominating,
        avg_chunks_promoted_per_round=3.0,
        context_reset_rate=0.1,
    )
    
    pareto = optimizer._compute_pareto([outcome_dominated, outcome_dominating])
    
    assert len(pareto) == 1
    assert pareto[0].session_id == "dominating"


def test_compute_pareto_multiple_nondominated():
    """Multiple non-dominated points all appear on frontier."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig()
    
    # High efficiency, high reset rate
    meta1 = SessionMeta(
        session_id="high_eff_high_reset",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome1 = SessionOutcome(
        session_id="high_eff_high_reset",
        config=config,
        meta=meta1,
        avg_chunks_promoted_per_round=5.0,
        context_reset_rate=0.4,
    )
    
    # Low efficiency, low reset rate
    meta2 = SessionMeta(
        session_id="low_eff_low_reset",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=False,
        rounds_completed=5,
    )
    outcome2 = SessionOutcome(
        session_id="low_eff_low_reset",
        config=config,
        meta=meta2,
        avg_chunks_promoted_per_round=1.0,
        context_reset_rate=0.1,
    )
    
    pareto = optimizer._compute_pareto([outcome1, outcome2])
    
    # Neither dominates the other, both should be on frontier
    assert len(pareto) == 2
    session_ids = {p.session_id for p in pareto}
    assert "high_eff_high_reset" in session_ids
    assert "low_eff_low_reset" in session_ids


# =============================================================================
# HarnessOptimizer.propose_next_config Tests
# =============================================================================

def test_propose_next_config_no_history():
    """Returns default config when no history exists."""
    optimizer = HarnessOptimizer()
    history = OptimizationHistory()
    
    result = optimizer.propose_next_config(history)
    
    assert result.quality_threshold == HarnessConfig().quality_threshold
    assert result.kl_epsilon == HarnessConfig().kl_epsilon


def test_propose_next_config_no_history_with_current():
    """Returns current_config when provided and no history."""
    optimizer = HarnessOptimizer()
    history = OptimizationHistory()
    current = HarnessConfig(quality_threshold=0.50)
    
    result = optimizer.propose_next_config(history, current_config=current)
    
    assert result.quality_threshold == 0.50


def test_propose_next_config_low_convergence():
    """Low convergence rate lowers quality_threshold."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig(quality_threshold=0.40)
    
    # Create outcomes with low convergence (0/2 = 0%)
    outcomes = []
    for i in range(2):
        meta = SessionMeta(
            session_id=f"session_{i}",
            goal="test",
            started_at="2026-03-31T10:00:00Z",
            converged=False,  # All not converged
            rounds_completed=5,
        )
        outcomes.append(SessionOutcome(
            session_id=f"session_{i}",
            config=config,
            meta=meta,
            avg_chunks_promoted_per_round=1.0,
            context_reset_rate=0.0,
        ))
    
    history = OptimizationHistory(outcomes=outcomes)
    result = optimizer.propose_next_config(history)
    
    # Should lower from 0.40 to 0.35
    assert abs(result.quality_threshold - 0.35) < 1e-9


def test_propose_next_config_high_convergence():
    """High convergence rate raises quality_threshold."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig(quality_threshold=0.40)
    
    # Create outcomes with high convergence and good efficiency
    outcomes = []
    for i in range(5):
        meta = SessionMeta(
            session_id=f"session_{i}",
            goal="test",
            started_at="2026-03-31T10:00:00Z",
            converged=True,  # All converged
            rounds_completed=5,
        )
        outcomes.append(SessionOutcome(
            session_id=f"session_{i}",
            config=config,
            meta=meta,
            avg_chunks_promoted_per_round=2.0,  # Good promotion
            context_reset_rate=0.0,
        ))
    
    history = OptimizationHistory(outcomes=outcomes)
    result = optimizer.propose_next_config(history)
    
    # Should raise from 0.40 to 0.45
    assert result.quality_threshold == 0.45


def test_propose_next_config_high_reset_rate():
    """High reset rate relaxes context pressure thresholds."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig(
        context_pressure_warning_pct=0.70,
        context_pressure_critical_pct=0.85,
    )
    
    # Create outcomes with high reset rate
    outcomes = []
    for i in range(3):
        meta = SessionMeta(
            session_id=f"session_{i}",
            goal="test",
            started_at="2026-03-31T10:00:00Z",
            converged=True,
            rounds_completed=5,
            context_resets_triggered=2,  # High resets
        )
        outcomes.append(SessionOutcome(
            session_id=f"session_{i}",
            config=config,
            meta=meta,
            avg_chunks_promoted_per_round=2.0,
            context_reset_rate=0.4,  # > 0.3 threshold
        ))
    
    history = OptimizationHistory(outcomes=outcomes)
    result = optimizer.propose_next_config(history)
    
    # Should increase thresholds
    assert result.context_pressure_warning_pct == 0.75
    assert result.context_pressure_critical_pct == 0.90


def test_propose_next_config_low_promotion():
    """Low chunk promotion lowers kl_epsilon."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig(kl_epsilon=0.15)
    
    # Create outcomes with low promotion
    outcomes = []
    for i in range(3):
        meta = SessionMeta(
            session_id=f"session_{i}",
            goal="test",
            started_at="2026-03-31T10:00:00Z",
            converged=True,
            rounds_completed=5,
        )
        outcomes.append(SessionOutcome(
            session_id=f"session_{i}",
            config=config,
            meta=meta,
            avg_chunks_promoted_per_round=0.5,  # < 1.0 threshold
            context_reset_rate=0.1,
        ))
    
    history = OptimizationHistory(outcomes=outcomes)
    result = optimizer.propose_next_config(history)
    
    # Should lower from 0.15 to 0.13
    assert result.kl_epsilon == 0.13


def test_propose_next_config_quality_threshold_floor():
    """quality_threshold doesn't go below floor (0.15)."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig(quality_threshold=0.18)
    
    outcomes = []
    for i in range(3):
        meta = SessionMeta(
            session_id=f"session_{i}",
            goal="test",
            started_at="2026-03-31T10:00:00Z",
            converged=False,
            rounds_completed=5,
        )
        outcomes.append(SessionOutcome(
            session_id=f"session_{i}",
            config=config,
            meta=meta,
            avg_chunks_promoted_per_round=1.0,
            context_reset_rate=0.0,
        ))
    
    history = OptimizationHistory(outcomes=outcomes)
    result = optimizer.propose_next_config(history)
    
    # 0.18 - 0.05 = 0.13, but floor is 0.15
    assert result.quality_threshold == 0.15


def test_propose_next_config_kl_epsilon_floor():
    """kl_epsilon doesn't go below floor (0.05)."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig(kl_epsilon=0.06)
    
    outcomes = []
    for i in range(3):
        meta = SessionMeta(
            session_id=f"session_{i}",
            goal="test",
            started_at="2026-03-31T10:00:00Z",
            converged=True,
            rounds_completed=5,
        )
        outcomes.append(SessionOutcome(
            session_id=f"session_{i}",
            config=config,
            meta=meta,
            avg_chunks_promoted_per_round=0.5,  # Triggers kl_epsilon reduction
            context_reset_rate=0.1,
        ))
    
    history = OptimizationHistory(outcomes=outcomes)
    result = optimizer.propose_next_config(history)
    
    # 0.06 - 0.02 = 0.04, but floor is 0.05
    assert result.kl_epsilon == 0.05


# =============================================================================
# HarnessOptimizer.summarize Tests
# =============================================================================

def test_summarize_empty():
    """Returns 'No optimization history yet.' when no outcomes."""
    optimizer = HarnessOptimizer()
    history = OptimizationHistory()
    
    result = optimizer.summarize(history)
    
    assert result == "No optimization history yet."


def test_summarize_with_outcomes():
    """Includes session count and best config."""
    optimizer = HarnessOptimizer()
    config = HarnessConfig(quality_threshold=0.42, kl_epsilon=0.12, max_rounds=15)
    
    meta = SessionMeta(
        session_id="best_session_abc",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=10,
        context_resets_triggered=1,
    )
    outcome = SessionOutcome(
        session_id="best_session_abc",
        config=config,
        meta=meta,
        avg_chunks_promoted_per_round=3.0,
        context_reset_rate=0.1,
    )
    
    history = OptimizationHistory(outcomes=[outcome])
    result = optimizer.summarize(history)
    
    assert "Sessions evaluated: 1" in result
    assert "Converged: 1/1" in result
    assert "best_ses" in result  # session_id[:8] (first 8 chars of "best_session_abc")
    assert "quality_threshold=0.42" in result
    assert "kl_epsilon=0.12" in result
    assert "max_rounds=15" in result


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_optimization_cycle(temp_traces_dir, sample_harness_config):
    """Test full cycle: load history → propose → verify sensible changes."""
    # Create multiple sessions with varying outcomes
    for i, (converged, resets, promoted) in enumerate([
        (False, 3, 0.5),  # Bad session
        (True, 1, 2.0),   # Medium session
        (True, 0, 4.0),   # Good session
    ]):
        session_id = f"session_{i:04d}"
        session_dir = temp_traces_dir / session_id
        session_dir.mkdir(parents=True)
        
        # Write config
        config = HarnessConfig(quality_threshold=0.35 + i * 0.05)
        (session_dir / "harness_config.json").write_text(json.dumps(asdict(config)))
        
        # Write meta
        meta = SessionMeta(
            session_id=session_id,
            goal="test",
            started_at="2026-03-31T10:00:00Z",
            converged=converged,
            rounds_completed=5,
            context_resets_triggered=resets,
        )
        (session_dir / "session_meta.json").write_text(json.dumps(asdict(meta)))
        
        # Write rounds
        rounds_dir = session_dir / "rounds"
        rounds_dir.mkdir()
        for r in range(5):
            (rounds_dir / f"round_{r+1:03d}.json").write_text(
                json.dumps({"round_number": r+1, "chunks_promoted": int(promoted)})
            )
    
    optimizer = HarnessOptimizer(traces_dir=temp_traces_dir)
    history = optimizer.load_history()
    
    assert len(history.outcomes) == 3
    assert len(history.pareto_frontier) >= 1
    
    # Best outcome should be the third one (converged, no resets, highest promotion)
    assert history.best_outcome.session_id == "session_0002"
    
    # Propose should use best outcome's config as base
    proposal = optimizer.propose_next_config(history)
    assert proposal.quality_threshold >= 0.35


# =============================================================================
# LLMProposer Tests
# =============================================================================

def test_llm_proposer_returns_none_without_api_key():
    """LLMProposer returns None when ANTHROPIC_API_KEY is not set."""
    from prsm.compute.nwtn.harness_optimizer import LLMProposer
    
    proposer = LLMProposer()
    config = HarnessConfig()
    meta = SessionMeta(
        session_id="test",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome = SessionOutcome(
        session_id="test",
        config=config,
        meta=meta,
    )
    history = OptimizationHistory(outcomes=[outcome])
    
    # Ensure ANTHROPIC_API_KEY is not set
    import os
    original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    
    try:
        result = proposer.propose(history)
        assert result is None
    finally:
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key


def test_llm_proposer_returns_none_on_invalid_json():
    """LLMProposer returns None when LLM returns invalid JSON."""
    from prsm.compute.nwtn.harness_optimizer import LLMProposer
    from unittest.mock import MagicMock, patch
    
    proposer = LLMProposer()
    config = HarnessConfig()
    meta = SessionMeta(
        session_id="test",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome = SessionOutcome(
        session_id="test",
        config=config,
        meta=meta,
    )
    history = OptimizationHistory(outcomes=[outcome])
    
    # Mock client with invalid JSON response
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "this is not valid json {{{"
    mock_message.content = [mock_content]
    mock_client.messages.create.return_value = mock_message
    
    with patch.object(proposer, "_get_client", return_value=mock_client):
        result = proposer.propose(history)
    
    assert result is None


def test_llm_proposer_parses_valid_json_response():
    """LLMProposer parses valid JSON and returns HarnessConfig."""
    from prsm.compute.nwtn.harness_optimizer import LLMProposer
    from unittest.mock import MagicMock, patch
    
    proposer = LLMProposer()
    config = HarnessConfig()
    meta = SessionMeta(
        session_id="test",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome = SessionOutcome(
        session_id="test",
        config=config,
        meta=meta,
    )
    history = OptimizationHistory(outcomes=[outcome])
    
    # Mock client with valid JSON response
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '{"quality_threshold": 0.28, "kl_epsilon": 0.08, "similarity_threshold": 0.85, "max_rounds": 20, "round_poll_interval": 5.0, "context_pressure_warning_pct": 0.75, "context_pressure_critical_pct": 0.88, "context_pressure_hard_limit_pct": 0.95, "feedback_quality_threshold": 0.80, "config_version": "1.0", "extra": {}}'
    mock_message.content = [mock_content]
    mock_client.messages.create.return_value = mock_message
    
    with patch.object(proposer, "_get_client", return_value=mock_client):
        result = proposer.propose(history)
    
    assert result is not None
    assert result.quality_threshold == 0.28
    assert result.kl_epsilon == 0.08
    assert result.similarity_threshold == 0.85
    assert result.max_rounds == 20


def test_llm_proposer_strips_markdown_fences():
    """LLMProposer strips ```json ... ``` fences from response."""
    from prsm.compute.nwtn.harness_optimizer import LLMProposer
    from unittest.mock import MagicMock, patch
    
    proposer = LLMProposer()
    config = HarnessConfig()
    meta = SessionMeta(
        session_id="test",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome = SessionOutcome(
        session_id="test",
        config=config,
        meta=meta,
    )
    history = OptimizationHistory(outcomes=[outcome])
    
    # Mock client with markdown-fenced JSON response
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '''```json
{"quality_threshold": 0.30, "kl_epsilon": 0.10, "similarity_threshold": 0.80, "max_rounds": 15, "round_poll_interval": 5.0, "context_pressure_warning_pct": 0.70, "context_pressure_critical_pct": 0.85, "context_pressure_hard_limit_pct": 0.95, "feedback_quality_threshold": 0.75, "config_version": "1.0", "extra": {}}
```'''
    mock_message.content = [mock_content]
    mock_client.messages.create.return_value = mock_message
    
    with patch.object(proposer, "_get_client", return_value=mock_client):
        result = proposer.propose(history)
    
    assert result is not None
    assert result.quality_threshold == 0.30
    assert result.kl_epsilon == 0.10


def test_propose_next_config_uses_llm_when_available():
    """propose_next_config uses LLM proposal when use_llm=True and available."""
    from prsm.compute.nwtn.harness_optimizer import LLMProposer
    from unittest.mock import MagicMock, patch
    
    optimizer = HarnessOptimizer()
    config = HarnessConfig(quality_threshold=0.40)
    meta = SessionMeta(
        session_id="test",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=True,
        rounds_completed=5,
    )
    outcome = SessionOutcome(
        session_id="test",
        config=config,
        meta=meta,
    )
    history = OptimizationHistory(outcomes=[outcome])
    
    # Mock LLMProposer to return a specific config
    llm_proposer = MagicMock()
    expected_config = HarnessConfig(quality_threshold=0.25, kl_epsilon=0.05)
    llm_proposer.propose.return_value = expected_config
    
    result = optimizer.propose_next_config(history, use_llm=True, llm_proposer=llm_proposer)
    
    assert result.quality_threshold == 0.25
    assert result.kl_epsilon == 0.05
    llm_proposer.propose.assert_called_once()


def test_propose_next_config_falls_back_to_rules_when_llm_none():
    """propose_next_config falls back to rule-based when LLM returns None."""
    from prsm.compute.nwtn.harness_optimizer import LLMProposer
    from unittest.mock import MagicMock
    
    optimizer = HarnessOptimizer()
    config = HarnessConfig(quality_threshold=0.40)
    meta = SessionMeta(
        session_id="test",
        goal="test",
        started_at="2026-03-31T10:00:00Z",
        converged=False,
        rounds_completed=5,
    )
    outcome = SessionOutcome(
        session_id="test",
        config=config,
        meta=meta,
        avg_chunks_promoted_per_round=1.0,
        context_reset_rate=0.0,
    )
    history = OptimizationHistory(outcomes=[outcome])
    
    # Mock LLMProposer to return None
    llm_proposer = MagicMock()
    llm_proposer.propose.return_value = None
    
    result = optimizer.propose_next_config(history, use_llm=True, llm_proposer=llm_proposer)
    
    # Should fall back to rule-based: low convergence → lower quality_threshold
    assert abs(result.quality_threshold - 0.35) < 1e-9  # 0.40 - 0.05
