#!/usr/bin/env python
"""
Synthetic NWTN Trace Generator for Meta-Harness Phase 4 seeding.

Generates 10 synthetic NWTN trace sessions covering diverse regions of the
config space. Each session writes the full trace structure that
HarnessOptimizer.load_history() reads.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.compute.nwtn.trace_logger import (
    HarnessConfig,
    NWTNTraceLogger,
    RoundTrace,
)


SYNTHETIC_SESSIONS = [
    # (session_id, config_kwargs, outcome_kwargs)
    # outcome_kwargs: converged, rounds_completed, context_resets, chunks_promoted_per_round

    # Low quality threshold — easy to promote, converges quickly
    ("synth-001", {"quality_threshold": 0.20, "kl_epsilon": 0.10},
     {"converged": True, "rounds": 6, "resets": 0, "promoted_per_round": 3.5}),

    # High quality threshold — hard to promote, fails to converge
    ("synth-002", {"quality_threshold": 0.60, "kl_epsilon": 0.10},
     {"converged": False, "rounds": 20, "resets": 2, "promoted_per_round": 0.3}),

    # Default config — moderate convergence
    ("synth-003", {"quality_threshold": 0.35, "kl_epsilon": 0.10},
     {"converged": True, "rounds": 12, "resets": 1, "promoted_per_round": 1.8}),

    # Low KL epsilon — accepts less novel chunks, moderate convergence
    ("synth-004", {"quality_threshold": 0.35, "kl_epsilon": 0.05},
     {"converged": True, "rounds": 10, "resets": 0, "promoted_per_round": 2.1}),

    # High KL epsilon — very strict novelty filter, low promotion
    ("synth-005", {"quality_threshold": 0.35, "kl_epsilon": 0.25},
     {"converged": False, "rounds": 20, "resets": 3, "promoted_per_round": 0.5}),

    # Tight context pressure — many resets, poor convergence
    ("synth-006", {"quality_threshold": 0.35, "kl_epsilon": 0.10,
                   "context_pressure_warning_pct": 0.50, "context_pressure_critical_pct": 0.65},
     {"converged": False, "rounds": 15, "resets": 8, "promoted_per_round": 1.0}),

    # Relaxed context pressure — few resets, good convergence
    ("synth-007", {"quality_threshold": 0.35, "kl_epsilon": 0.10,
                   "context_pressure_warning_pct": 0.80, "context_pressure_critical_pct": 0.92},
     {"converged": True, "rounds": 11, "resets": 0, "promoted_per_round": 2.3}),

    # Sweet spot — low threshold + low KL + relaxed pressure
    ("synth-008", {"quality_threshold": 0.25, "kl_epsilon": 0.07,
                   "context_pressure_warning_pct": 0.75, "context_pressure_critical_pct": 0.88},
     {"converged": True, "rounds": 8, "resets": 0, "promoted_per_round": 3.0}),

    # Overly permissive — too low threshold, noisy promotions, slow convergence
    ("synth-009", {"quality_threshold": 0.10, "kl_epsilon": 0.03},
     {"converged": True, "rounds": 18, "resets": 0, "promoted_per_round": 6.0}),

    # High max_rounds + moderate config — eventually converges
    ("synth-010", {"quality_threshold": 0.40, "kl_epsilon": 0.12, "max_rounds": 30},
     {"converged": True, "rounds": 24, "resets": 2, "promoted_per_round": 1.4}),
]


def generate_synthetic_traces(output_dir: Path) -> int:
    """Generate all synthetic trace sessions. Returns count of sessions created."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)  # Reproducibility

    for session_id, config_kwargs, outcome_kwargs in SYNTHETIC_SESSIONS:
        _generate_session(
            output_dir=output_dir,
            session_id=session_id,
            config_kwargs=config_kwargs,
            outcome_kwargs=outcome_kwargs,
        )
        print(f"  Generated session: {session_id}")

    return len(SYNTHETIC_SESSIONS)


def _generate_session(
    output_dir: Path,
    session_id: str,
    config_kwargs: Dict[str, Any],
    outcome_kwargs: Dict[str, Any],
) -> None:
    """Generate a single synthetic trace session."""
    converged = outcome_kwargs["converged"]
    total_rounds = outcome_kwargs["rounds"]
    total_resets = outcome_kwargs["resets"]
    promoted_per_round = outcome_kwargs["promoted_per_round"]

    # 1. Create HarnessConfig
    config = HarnessConfig(**config_kwargs)

    # 2. Create NWTNTraceLogger
    goal = f"Synthetic goal for session {session_id}: optimize distributed consensus algorithm"
    trace_logger = NWTNTraceLogger(
        session_id=session_id,
        goal=goal,
        traces_dir=output_dir,
    )

    # 3. Log config
    trace_logger.log_config(config)

    # 4. Set team
    trace_logger.set_team(["agent-alpha", "agent-beta", "agent-gamma"])

    # 5. Distribute reset rounds evenly
    reset_rounds = set()
    if total_resets > 0:
        # Distribute resets across the session, avoiding first and last rounds
        available_rounds = list(range(2, total_rounds)) if total_rounds > 2 else [1]
        step = max(1, len(available_rounds) // max(1, total_resets))
        for i in range(total_resets):
            if i * step < len(available_rounds):
                reset_rounds.add(available_rounds[i * step])

    # 6. Generate rounds
    elapsed_seconds = 0.0
    for n in range(1, total_rounds + 1):
        trace_logger.start_round(n)

        # Generate synthetic quality reports
        # promoted_per_round chunks pass, a few more fail
        num_promoted = int(promoted_per_round + random.uniform(-0.5, 0.5))
        num_promoted = max(0, num_promoted)
        num_failed = random.randint(1, 3)

        for i in range(num_failed):
            report = {
                "chunk_id": f"chunk-fail-{n}-{i}",
                "quality_score": random.uniform(0.1, config.quality_threshold - 0.05),
                "passed": False,
                "reason": "below_threshold",
            }
            trace_logger.record_quality_report(n, report)

        for i in range(num_promoted):
            report = {
                "chunk_id": f"chunk-pass-{n}-{i}",
                "quality_score": random.uniform(config.quality_threshold + 0.05, 0.95),
                "passed": True,
                "reason": "above_threshold",
            }
            trace_logger.record_quality_report(n, report)

        # Context pressure if this is a reset round
        if n in reset_rounds:
            # Pick an agent to reset
            agent = random.choice(["agent-alpha", "agent-beta", "agent-gamma"])
            trace_logger.record_context_pressure(
                round_number=n,
                agent_id=agent,
                token_count=int(config.context_pressure_critical_pct * 100000),
                level="CRITICAL",
                reset_triggered=True,
            )

        # Record convergence
        is_converged = (n == total_rounds and converged)
        pending = [] if is_converged else (["agent-alpha"] if n == total_rounds else ["agent-alpha", "agent-beta"])
        trace_logger.record_convergence(n, pending_agents=pending, converged=is_converged)

        # End round
        trace_logger.end_round(n)

        elapsed_seconds += random.uniform(8.0, 25.0)

    # 7. Finalize
    trace_logger.finalize(
        converged=converged,
        rounds_completed=total_rounds,
        context_resets_triggered=total_resets,
        feedback_rounds_completed=random.randint(0, total_rounds // 3),
        elapsed_seconds=elapsed_seconds,
        final_status="CONVERGED" if converged else "MAX_ROUNDS_REACHED",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NWTN traces")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".nwtn_traces"),
        help="Output directory for traces (default: .nwtn_traces)",
    )
    args = parser.parse_args()

    print(f"Generating synthetic traces to: {args.output_dir}")
    count = generate_synthetic_traces(args.output_dir)
    print(f"Generated {count} sessions in {args.output_dir}")


if __name__ == "__main__":
    main()
