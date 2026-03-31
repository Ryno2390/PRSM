"""
NWTNTraceLogger — Per-session execution trace logging for Meta-Harness optimization.

Writes structured JSON traces to `.nwtn_traces/{session_id}/` for each session.
These traces form the filesystem that a Meta-Harness optimizer reads to propose
improved harness configurations.

Trace structure per session:
    .nwtn_traces/
    └── {session_id}/
        ├── harness_config.json     — hyperparameters used this run
        ├── session_meta.json       — goal, timestamps, outcome
        └── rounds/
            └── round_{N:03d}.json  — per-round trace data

Design principles (from Meta-Harness paper):
- Full history, not compressed summaries
- Selectively queryable via grep/cat (structured JSON)
- Include execution traces, scores, and config together
- Zero overhead when disabled (trace_logger=None pattern)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoundTrace:
    """Trace data for a single checkpoint round."""
    round_number: int
    timestamp: str
    # Convergence
    agent_outputs: Dict[str, str] = field(default_factory=dict)       # agent_id -> output snippet (first 500 chars)
    convergence_signals: Dict[str, bool] = field(default_factory=dict) # agent_id -> converged?
    pending_agents: List[str] = field(default_factory=list)
    converged: bool = False
    # Quality gate
    quality_reports: List[Dict[str, Any]] = field(default_factory=list)  # QualityReport dicts
    chunks_evaluated: int = 0
    chunks_promoted: int = 0
    # Context pressure
    token_counts: Dict[str, int] = field(default_factory=dict)  # agent_id -> token count
    context_resets: List[str] = field(default_factory=list)     # agent_ids reset this round
    pressure_levels: Dict[str, str] = field(default_factory=dict)  # agent_id -> level name
    # Feedback
    feedback_published: int = 0
    feedback_targets: List[str] = field(default_factory=list)
    # BSC event counts
    events_advanced: int = 0


@dataclass
class HarnessConfig:
    """Snapshot of the harness hyperparameters used for this session."""
    quality_threshold: float = 0.35
    max_rounds: int = 20
    round_poll_interval: float = 5.0
    context_pressure_warning_pct: float = 0.70
    context_pressure_critical_pct: float = 0.85
    context_pressure_hard_limit_pct: float = 0.95
    extra: Dict[str, Any] = field(default_factory=dict)  # any additional params


@dataclass
class SessionMeta:
    """Top-level session metadata."""
    session_id: str
    goal: str
    started_at: str
    finished_at: Optional[str] = None
    converged: bool = False
    rounds_completed: int = 0
    context_resets_triggered: int = 0
    feedback_rounds_completed: int = 0
    elapsed_seconds: float = 0.0
    final_status: str = ""
    team_members: List[str] = field(default_factory=list)


class NWTNTraceLogger:
    """
    Writes structured execution traces to disk for Meta-Harness optimization.

    Usage
    -----
    logger = NWTNTraceLogger(session_id="abc123", goal="...", traces_dir=Path(".nwtn_traces"))
    logger.log_config(HarnessConfig(quality_threshold=0.35))
    logger.start_round(1)
    logger.record_agent_output("coder-1", 1, "output text")
    logger.record_quality_report(1, quality_report_dict)
    logger.record_context_pressure(1, "coder-1", token_count=5000, level="WARNING")
    logger.record_convergence(1, {"coder-1": True})
    logger.end_round(1)
    logger.finalize(run_result)
    """

    def __init__(
        self,
        session_id: str,
        goal: str,
        traces_dir: Path = None,
    ):
        self.session_id = session_id
        self.goal = goal
        self.traces_dir = traces_dir or Path(".nwtn_traces")
        self._session_dir = self.traces_dir / session_id
        self._rounds_dir = self._session_dir / "rounds"
        self._rounds_dir.mkdir(parents=True, exist_ok=True)
        self._current_rounds: Dict[int, RoundTrace] = {}
        self._meta = SessionMeta(
            session_id=session_id,
            goal=goal,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def log_config(self, config: HarnessConfig) -> None:
        """Write harness config to disk immediately."""
        path = self._session_dir / "harness_config.json"
        path.write_text(json.dumps(asdict(config), indent=2))

    def set_team(self, team_members: List[str]) -> None:
        self._meta.team_members = list(team_members)

    def start_round(self, round_number: int) -> None:
        """Initialize trace for a new round."""
        self._current_rounds[round_number] = RoundTrace(
            round_number=round_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def record_agent_output(self, agent_id: str, round_number: int, output: str) -> None:
        rt = self._current_rounds.get(round_number)
        if rt is None:
            return
        rt.agent_outputs[agent_id] = output[:500]  # cap at 500 chars

    def record_convergence_signal(self, agent_id: str, round_number: int, converged: bool) -> None:
        rt = self._current_rounds.get(round_number)
        if rt is None:
            return
        rt.convergence_signals[agent_id] = converged

    def record_quality_report(self, round_number: int, report: Dict[str, Any]) -> None:
        """Record a QualityReport dict (call after each QualityGate evaluation)."""
        rt = self._current_rounds.get(round_number)
        if rt is None:
            return
        rt.quality_reports.append(report)
        rt.chunks_evaluated += 1
        if report.get("passed"):
            rt.chunks_promoted += 1

    def record_context_pressure(
        self,
        round_number: int,
        agent_id: str,
        token_count: int,
        level: str = "OK",
        reset_triggered: bool = False,
    ) -> None:
        rt = self._current_rounds.get(round_number)
        if rt is None:
            return
        rt.token_counts[agent_id] = token_count
        rt.pressure_levels[agent_id] = level
        if reset_triggered:
            rt.context_resets.append(agent_id)

    def record_feedback(self, round_number: int, count: int, targets: List[str]) -> None:
        rt = self._current_rounds.get(round_number)
        if rt is None:
            return
        rt.feedback_published = count
        rt.feedback_targets = list(targets)

    def record_convergence(self, round_number: int, pending_agents: List[str], converged: bool = False) -> None:
        rt = self._current_rounds.get(round_number)
        if rt is None:
            return
        rt.pending_agents = list(pending_agents)
        rt.converged = converged

    def end_round(self, round_number: int) -> None:
        """Flush round trace to disk."""
        rt = self._current_rounds.get(round_number)
        if rt is None:
            return
        path = self._rounds_dir / f"round_{round_number:03d}.json"
        path.write_text(json.dumps(asdict(rt), indent=2))

    def finalize(
        self,
        converged: bool,
        rounds_completed: int,
        context_resets_triggered: int,
        feedback_rounds_completed: int,
        elapsed_seconds: float,
        final_status: str,
    ) -> None:
        """Write session_meta.json with final outcome."""
        self._meta.finished_at = datetime.now(timezone.utc).isoformat()
        self._meta.converged = converged
        self._meta.rounds_completed = rounds_completed
        self._meta.context_resets_triggered = context_resets_triggered
        self._meta.feedback_rounds_completed = feedback_rounds_completed
        self._meta.elapsed_seconds = elapsed_seconds
        self._meta.final_status = final_status
        path = self._session_dir / "session_meta.json"
        path.write_text(json.dumps(asdict(self._meta), indent=2))
        logger.info(
            "NWTNTraceLogger: session %s trace written to %s (%d rounds)",
            self.session_id,
            self._session_dir,
            rounds_completed,
        )
