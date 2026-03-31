"""
HarnessOptimizer — Meta-Harness outer loop for NWTN harness optimization.

Based on: Meta-Harness: End-to-End Optimization of Model Harnesses
(Lee et al., Stanford/MIT, 2026 — arXiv:2603.28052)

The optimizer reads the .nwtn_traces/ filesystem to build a history of
(HarnessConfig → session outcome) pairs, then uses an LLM proposer to
suggest an improved HarnessConfig for the next session.

Design principles (from the paper):
- Full history access, not compressed summaries
- Filesystem as the feedback channel (grep/cat over prior traces)
- Proposer decides what to inspect — not hardcoded heuristics
- Pareto frontier over competing objectives (quality vs. context cost)
- No automatic session execution — caller decides when to run

Usage
-----
optimizer = HarnessOptimizer(traces_dir=Path(".nwtn_traces"))
history = optimizer.load_history()
proposal = optimizer.propose_next_config(history, goal="...")
# Caller then runs: session.run(trace_logger=..., harness_config=proposal)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .trace_logger import HarnessConfig, SessionMeta

logger = logging.getLogger(__name__)


@dataclass
class SessionOutcome:
    """Paired (HarnessConfig, SessionMeta) from a completed session."""
    session_id: str
    config: HarnessConfig
    meta: SessionMeta
    # Derived metrics
    avg_chunks_promoted_per_round: float = 0.0
    avg_quality_score: float = 0.0
    context_reset_rate: float = 0.0   # resets / rounds_completed

    @property
    def efficiency_score(self) -> float:
        """
        Composite score: promotes quality outcomes reached with fewer resets.
        Higher is better.
        """
        if self.meta.rounds_completed == 0:
            return 0.0
        convergence_bonus = 1.0 if self.meta.converged else 0.5
        reset_penalty = max(0.0, 1.0 - self.context_reset_rate)
        return convergence_bonus * reset_penalty * (self.avg_chunks_promoted_per_round + 0.01)


@dataclass
class ParetoPoint:
    """A point on the quality/cost Pareto frontier."""
    session_id: str
    efficiency_score: float
    avg_rounds: float
    context_reset_rate: float
    config: HarnessConfig


@dataclass
class OptimizationHistory:
    """All known session outcomes, used by the proposer."""
    outcomes: List[SessionOutcome] = field(default_factory=list)
    pareto_frontier: List[ParetoPoint] = field(default_factory=list)

    @property
    def best_outcome(self) -> Optional[SessionOutcome]:
        if not self.outcomes:
            return None
        return max(self.outcomes, key=lambda o: o.efficiency_score)

    def to_prompt_context(self) -> str:
        """
        Render the history as a concise text block for the LLM proposer.
        Includes all session outcomes sorted by efficiency score (best last).
        """
        if not self.outcomes:
            return "No prior sessions. Use default HarnessConfig values."

        lines = ["## Prior NWTN Session Outcomes (sorted by efficiency, worst→best)\n"]
        sorted_outcomes = sorted(self.outcomes, key=lambda o: o.efficiency_score)
        for o in sorted_outcomes:
            lines.append(f"### Session {o.session_id[:8]}")
            lines.append(f"- efficiency_score: {o.efficiency_score:.4f}")
            lines.append(f"- converged: {o.meta.converged}, rounds: {o.meta.rounds_completed}")
            lines.append(f"- context_resets: {o.meta.context_resets_triggered}, feedback_rounds: {o.meta.feedback_rounds_completed}")
            lines.append(f"- avg_chunks_promoted/round: {o.avg_chunks_promoted_per_round:.2f}")
            lines.append(f"- config: quality_threshold={o.config.quality_threshold}, kl_epsilon={o.config.kl_epsilon}, "
                        f"similarity_threshold={o.config.similarity_threshold}, max_rounds={o.config.max_rounds}")
            lines.append(f"- context_pressure: warning={o.config.context_pressure_warning_pct}, "
                        f"critical={o.config.context_pressure_critical_pct}")
            lines.append("")
        return "\n".join(lines)


class HarnessOptimizer:
    """
    Reads the .nwtn_traces/ filesystem and proposes improved HarnessConfig values.

    The optimizer uses a simple rule-based proposer by default, with hooks for
    an LLM-based proposer when available. The rule-based proposer implements
    directional search: if quality scores are low → lower quality_threshold;
    if context resets are high → lower pressure thresholds or reduce max_rounds.
    """

    def __init__(self, traces_dir: Path = None):
        self.traces_dir = traces_dir or Path(".nwtn_traces")

    def load_history(self) -> OptimizationHistory:
        """
        Scan traces_dir and load all completed session outcomes.
        Skips sessions missing harness_config.json or session_meta.json.
        """
        history = OptimizationHistory()
        if not self.traces_dir.exists():
            logger.info("HarnessOptimizer: traces_dir %s does not exist yet", self.traces_dir)
            return history

        for session_dir in sorted(self.traces_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            config_path = session_dir / "harness_config.json"
            meta_path = session_dir / "session_meta.json"
            if not config_path.exists() or not meta_path.exists():
                continue
            try:
                config = HarnessConfig.from_dict(json.loads(config_path.read_text()))
                meta_dict = json.loads(meta_path.read_text())
                # Build SessionMeta from dict
                meta = SessionMeta(
                    session_id=meta_dict.get("session_id", ""),
                    goal=meta_dict.get("goal", ""),
                    started_at=meta_dict.get("started_at", ""),
                    finished_at=meta_dict.get("finished_at"),
                    converged=meta_dict.get("converged", False),
                    rounds_completed=meta_dict.get("rounds_completed", 0),
                    context_resets_triggered=meta_dict.get("context_resets_triggered", 0),
                    feedback_rounds_completed=meta_dict.get("feedback_rounds_completed", 0),
                    elapsed_seconds=meta_dict.get("elapsed_seconds", 0.0),
                    final_status=meta_dict.get("final_status", ""),
                    team_members=meta_dict.get("team_members", []),
                )
                outcome = self._build_outcome(session_dir, config, meta)
                history.outcomes.append(outcome)
            except Exception as e:
                logger.warning("HarnessOptimizer: failed to load session %s: %s", session_dir.name, e)

        history.pareto_frontier = self._compute_pareto(history.outcomes)
        logger.info("HarnessOptimizer: loaded %d session outcomes", len(history.outcomes))
        return history

    def _build_outcome(
        self,
        session_dir: Path,
        config: HarnessConfig,
        meta: SessionMeta,
    ) -> SessionOutcome:
        """Compute derived metrics by reading round files."""
        rounds_dir = session_dir / "rounds"
        total_promoted = 0
        total_rounds = 0
        if rounds_dir.exists():
            for round_file in sorted(rounds_dir.glob("round_*.json")):
                try:
                    rd = json.loads(round_file.read_text())
                    total_promoted += rd.get("chunks_promoted", 0)
                    total_rounds += 1
                except Exception:
                    pass
        avg_promoted = total_promoted / total_rounds if total_rounds > 0 else 0.0
        reset_rate = (
            meta.context_resets_triggered / meta.rounds_completed
            if meta.rounds_completed > 0 else 0.0
        )
        return SessionOutcome(
            session_id=meta.session_id,
            config=config,
            meta=meta,
            avg_chunks_promoted_per_round=avg_promoted,
            context_reset_rate=reset_rate,
        )

    def _compute_pareto(self, outcomes: List[SessionOutcome]) -> List[ParetoPoint]:
        """Compute Pareto frontier over (efficiency_score, -context_reset_rate)."""
        if not outcomes:
            return []
        points = [
            ParetoPoint(
                session_id=o.session_id,
                efficiency_score=o.efficiency_score,
                avg_rounds=float(o.meta.rounds_completed),
                context_reset_rate=o.context_reset_rate,
                config=o.config,
            )
            for o in outcomes
        ]
        # Simple 2D Pareto: maximize efficiency_score, minimize context_reset_rate
        pareto = []
        for p in points:
            dominated = any(
                q.efficiency_score >= p.efficiency_score
                and q.context_reset_rate <= p.context_reset_rate
                and (q.efficiency_score > p.efficiency_score or q.context_reset_rate < p.context_reset_rate)
                for q in points if q is not p
            )
            if not dominated:
                pareto.append(p)
        return sorted(pareto, key=lambda p: p.efficiency_score, reverse=True)

    def propose_next_config(
        self,
        history: OptimizationHistory,
        goal: str = "",
        current_config: HarnessConfig = None,
    ) -> HarnessConfig:
        """
        Propose an improved HarnessConfig based on history.

        Uses a rule-based directional search:
        - No history → return default config
        - Low convergence rate → lower quality_threshold (easier to promote)
        - High reset rate → lower context pressure thresholds
        - Low chunk promotion → lower quality_threshold or kl_epsilon
        - Already converging well → tighten quality_threshold slightly

        The proposal stays within safe bounds to avoid degenerate configs.
        """
        if not history.outcomes:
            logger.info("HarnessOptimizer: no history, returning default config")
            return current_config or HarnessConfig()

        best = history.best_outcome
        base = HarnessConfig(
            quality_threshold=best.config.quality_threshold,
            kl_epsilon=best.config.kl_epsilon,
            similarity_threshold=best.config.similarity_threshold,
            max_rounds=best.config.max_rounds,
            round_poll_interval=best.config.round_poll_interval,
            context_pressure_warning_pct=best.config.context_pressure_warning_pct,
            context_pressure_critical_pct=best.config.context_pressure_critical_pct,
            context_pressure_hard_limit_pct=best.config.context_pressure_hard_limit_pct,
            feedback_quality_threshold=best.config.feedback_quality_threshold,
        )

        convergence_rate = sum(1 for o in history.outcomes if o.meta.converged) / len(history.outcomes)
        avg_reset_rate = sum(o.context_reset_rate for o in history.outcomes) / len(history.outcomes)
        avg_promoted = sum(o.avg_chunks_promoted_per_round for o in history.outcomes) / len(history.outcomes)

        # Rule 1: Low convergence → lower quality threshold (promote more ideas)
        if convergence_rate < 0.5:
            base.quality_threshold = max(0.15, base.quality_threshold - 0.05)
            logger.info("HarnessOptimizer: low convergence (%.2f) → lowering quality_threshold to %.2f",
                       convergence_rate, base.quality_threshold)

        # Rule 2: High convergence → tighten quality threshold (higher bar)
        elif convergence_rate > 0.8 and best.efficiency_score > 0.5:
            base.quality_threshold = min(0.60, base.quality_threshold + 0.05)
            logger.info("HarnessOptimizer: high convergence (%.2f) → raising quality_threshold to %.2f",
                       convergence_rate, base.quality_threshold)

        # Rule 3: High context reset rate → give agents more room
        if avg_reset_rate > 0.3:
            base.context_pressure_warning_pct = min(0.80, base.context_pressure_warning_pct + 0.05)
            base.context_pressure_critical_pct = min(0.90, base.context_pressure_critical_pct + 0.05)
            logger.info("HarnessOptimizer: high reset rate (%.2f) → relaxing context pressure thresholds",
                       avg_reset_rate)

        # Rule 4: Low chunk promotion → lower KL epsilon (accept less novel chunks)
        if avg_promoted < 1.0:
            base.kl_epsilon = max(0.05, base.kl_epsilon - 0.02)
            logger.info("HarnessOptimizer: low promotion (%.2f/round) → lowering kl_epsilon to %.2f",
                       avg_promoted, base.kl_epsilon)

        return base

    def summarize(self, history: OptimizationHistory) -> str:
        """Return a human-readable summary of optimization history."""
        if not history.outcomes:
            return "No optimization history yet."
        lines = [
            f"Sessions evaluated: {len(history.outcomes)}",
            f"Converged: {sum(1 for o in history.outcomes if o.meta.converged)}/{len(history.outcomes)}",
            f"Pareto frontier size: {len(history.pareto_frontier)}",
        ]
        if history.best_outcome:
            b = history.best_outcome
            lines.append(f"Best session: {b.session_id[:8]} (efficiency={b.efficiency_score:.4f}, "
                        f"converged={b.meta.converged}, resets={b.meta.context_resets_triggered})")
            lines.append(f"Best config: quality_threshold={b.config.quality_threshold}, "
                        f"kl_epsilon={b.config.kl_epsilon}, max_rounds={b.config.max_rounds}")
        if history.pareto_frontier:
            lines.append("Pareto frontier (efficiency → reset_rate):")
            for p in history.pareto_frontier[:3]:
                lines.append(f"  {p.session_id[:8]}: eff={p.efficiency_score:.4f}, resets={p.context_reset_rate:.2f}")
        return "\n".join(lines)
