"""
Tests for synthetic trace generation and optimization workflow.

Tests that:
- Generator creates valid trace files
- Optimizer loads and processes synthetic traces
- Optimizer produces valid proposals
- Optimizer identifies efficiency differences between sessions
- Pareto frontier is computed correctly
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prsm.compute.nwtn.trace_logger import HarnessConfig, NWTNTraceLogger
from prsm.compute.nwtn.harness_optimizer import HarnessOptimizer


class TestSyntheticTraceGeneration:
    """Tests for synthetic trace generation."""

    def test_generate_synthetic_traces_creates_files(self, tmp_path: Path):
        """Run the generator and verify 10 session dirs are created."""
        from scripts.generate_synthetic_traces import generate_synthetic_traces

        count = generate_synthetic_traces(tmp_path)

        assert count == 10, f"Expected 10 sessions, got {count}"

        # Verify each session has required files
        session_dirs = list(tmp_path.iterdir())
        assert len(session_dirs) == 10, f"Expected 10 session dirs, found {len(session_dirs)}"

        for session_dir in session_dirs:
            assert session_dir.is_dir()
            assert (session_dir / "harness_config.json").exists()
            assert (session_dir / "session_meta.json").exists()
            assert (session_dir / "rounds").is_dir()

            # Verify rounds directory has round files
            round_files = list((session_dir / "rounds").glob("round_*.json"))
            assert len(round_files) > 0, f"No round files in {session_dir.name}"

    def test_generated_traces_have_valid_structure(self, tmp_path: Path):
        """Verify generated traces have valid JSON structure."""
        from scripts.generate_synthetic_traces import generate_synthetic_traces

        generate_synthetic_traces(tmp_path)

        for session_dir in tmp_path.iterdir():
            if not session_dir.is_dir():
                continue

            # Load and validate harness_config.json
            config_path = session_dir / "harness_config.json"
            config = json.loads(config_path.read_text())
            assert "quality_threshold" in config
            assert "kl_epsilon" in config
            assert 0 <= config["quality_threshold"] <= 1
            assert 0 <= config["kl_epsilon"] <= 1

            # Load and validate session_meta.json
            meta_path = session_dir / "session_meta.json"
            meta = json.loads(meta_path.read_text())
            assert "session_id" in meta
            assert "converged" in meta
            assert "rounds_completed" in meta
            assert meta["rounds_completed"] > 0

            # Verify round files
            rounds_dir = session_dir / "rounds"
            for round_file in sorted(rounds_dir.glob("round_*.json")):
                round_data = json.loads(round_file.read_text())
                assert "round_number" in round_data
                assert "chunks_evaluated" in round_data
                assert "chunks_promoted" in round_data


class TestOptimizerIntegration:
    """Tests for optimizer integration with synthetic traces."""

    def test_optimizer_loads_synthetic_traces(self, tmp_path: Path):
        """Generate traces and verify optimizer loads all 10 sessions."""
        from scripts.generate_synthetic_traces import generate_synthetic_traces

        generate_synthetic_traces(tmp_path)

        optimizer = HarnessOptimizer(traces_dir=tmp_path)
        history = optimizer.load_history()

        assert len(history.outcomes) == 10, f"Expected 10 outcomes, got {len(history.outcomes)}"

        for outcome in history.outcomes:
            assert outcome.session_id.startswith("synth-")
            assert outcome.config is not None
            assert outcome.meta is not None
            assert isinstance(outcome.config, HarnessConfig)
            assert 0 <= outcome.config.quality_threshold <= 1.0
            assert outcome.config.max_rounds > 0
            assert outcome.meta.rounds_completed > 0

    def test_optimizer_proposes_valid_config(self, tmp_path: Path):
        """Verify proposed config has valid field values."""
        from scripts.generate_synthetic_traces import generate_synthetic_traces

        generate_synthetic_traces(tmp_path)

        optimizer = HarnessOptimizer(traces_dir=tmp_path)
        history = optimizer.load_history()
        proposal = optimizer.propose_next_config(history)

        assert 0.0 <= proposal.quality_threshold <= 1.0, f"quality_threshold out of bounds: {proposal.quality_threshold}"
        assert 0.0 <= proposal.kl_epsilon <= 1.0, f"kl_epsilon out of bounds: {proposal.kl_epsilon}"
        assert proposal.max_rounds > 0, f"max_rounds must be positive: {proposal.max_rounds}"
        assert 0.0 <= proposal.context_pressure_warning_pct <= 1.0
        assert 0.0 <= proposal.context_pressure_critical_pct <= 1.0

    def test_optimizer_identifies_best_session(self, tmp_path: Path):
        """Verify synth-008 (sweet spot) has higher efficiency than synth-002 (poor config)."""
        from scripts.generate_synthetic_traces import generate_synthetic_traces

        generate_synthetic_traces(tmp_path)

        optimizer = HarnessOptimizer(traces_dir=tmp_path)
        history = optimizer.load_history()

        # Find synth-002 and synth-008 outcomes
        synth_002 = next((o for o in history.outcomes if o.session_id == "synth-002"), None)
        synth_008 = next((o for o in history.outcomes if o.session_id == "synth-008"), None)

        assert synth_002 is not None, "synth-002 not found"
        assert synth_008 is not None, "synth-008 not found"

        # synth-008 should have higher efficiency
        # synth-002: high threshold, no convergence, low promotion
        # synth-008: sweet spot, converged, high promotion
        assert synth_008.efficiency_score > synth_002.efficiency_score, (
            f"synth-008 ({synth_008.efficiency_score:.4f}) should beat "
            f"synth-002 ({synth_002.efficiency_score:.4f})"
        )

    def test_optimizer_pareto_frontier_nonempty(self, tmp_path: Path):
        """Verify Pareto frontier has at least 1 point after loading synthetic traces."""
        from scripts.generate_synthetic_traces import generate_synthetic_traces

        generate_synthetic_traces(tmp_path)

        optimizer = HarnessOptimizer(traces_dir=tmp_path)
        history = optimizer.load_history()

        assert len(history.pareto_frontier) >= 1, "Pareto frontier should have at least 1 point"

        # Verify Pareto points have valid config
        for point in history.pareto_frontier:
            assert point.config is not None
            assert isinstance(point.config, HarnessConfig)

    def test_full_optimization_loop(self, tmp_path: Path):
        """Generate traces, load history, propose, verify proposal is valid."""
        from scripts.generate_synthetic_traces import generate_synthetic_traces

        generate_synthetic_traces(tmp_path)

        optimizer = HarnessOptimizer(traces_dir=tmp_path)
        history = optimizer.load_history()

        assert len(history.outcomes) == 10

        proposal = optimizer.propose_next_config(history)

        # Proposal should be valid HarnessConfig
        assert isinstance(proposal, HarnessConfig)
        # Check valid bounds (quality_threshold should be in [0, 1])
        assert 0.0 <= proposal.quality_threshold <= 1.0
        assert 0.0 <= proposal.kl_epsilon <= 1.0
        assert proposal.max_rounds > 0

        # Proposal should differ from worst config (high threshold, no convergence)
        # The worst config would be something like synth-002 (high threshold, failed)
        worst_session = min(history.outcomes, key=lambda o: o.efficiency_score)

        # Proposal should not be identical to worst session config
        # (optimizer should propose something better)
        if worst_session.session_id == "synth-002":
            # synth-002 has quality_threshold=0.60, proposal should be lower
            assert proposal.quality_threshold < worst_session.config.quality_threshold, (
                f"Proposal quality_threshold ({proposal.quality_threshold}) should be "
                f"better than worst session ({worst_session.session_id})'s {worst_session.config.quality_threshold}"
            )

        # Verify proposal can be serialized and deserialized
        proposal_dict = proposal.to_dict()
        restored = HarnessConfig.from_dict(proposal_dict)
        assert restored.quality_threshold == proposal.quality_threshold


class TestTraceLoggerDirect:
    """Direct tests for NWTNTraceLogger."""

    def test_trace_logger_creates_valid_trace(self, tmp_path: Path):
        """Test basic trace creation with NWTNTraceLogger."""
        traces_dir = tmp_path / "traces"

        # Create a simple trace
        logger = NWTNTraceLogger(
            session_id="test-direct-001",
            goal="Test goal for direct trace logger test",
            traces_dir=traces_dir,
        )

        config = HarnessConfig(quality_threshold=0.35, kl_epsilon=0.1)
        logger.log_config(config)
        logger.set_team(["agent-a", "agent-b", "agent-c"])

        # Create a single round
        logger.start_round(1)
        logger.record_quality_report(1, {"chunk_id": "c1", "passed": True, "quality_score": 0.5})
        logger.record_convergence(1, pending_agents=[], converged=True)
        logger.end_round(1)

        # Finalize
        logger.finalize(
            converged=True,
            rounds_completed=1,
            context_resets_triggered=0,
            feedback_rounds_completed=0,
            elapsed_seconds=10.0,
            final_status="CONVERGED",
        )

        # Verify files exist
        session_dir = traces_dir / "test-direct-001"
        assert (session_dir / "harness_config.json").exists()
        assert (session_dir / "session_meta.json").exists()
        assert (session_dir / "rounds" / "round_001.json").exists()

        # Verify content
        meta = json.loads((session_dir / "session_meta.json").read_text())
        assert meta["converged"] is True
        assert meta["rounds_completed"] == 1
