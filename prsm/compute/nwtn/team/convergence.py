"""
Convergence Tracker
===================

Detects when an Agent Team has finished its work and the session can end
early, avoiding redundant rounds that waste tokens and time.

Evidence from 9-round live test
--------------------------------
The security reviewer issued "SECURITY CLEARED" in rounds 7, 8, AND 9.
The data scientist confirmed "implementation is correct" in round 7.
The coder confirmed "no new bugs, implementation is correct" in round 9.
Rounds 8-9 were entirely redundant: no new findings, no whiteboard promotions
from these agents, just confirmation of already-known conclusions.

Detecting convergence at round 7 would have saved 2 rounds × 4 agents = 8
LLM calls (~$0.01 and ~3-4 minutes per 9-round session).

How it works
------------
The tracker scans each agent's output for convergence phrases.  When an
agent's signal has appeared in at least ``min_consecutive_rounds``
consecutive rounds, that agent is marked "converged."  When ALL registered
agents are converged, the session can end early.

The tracker does NOT end the session — it only provides the signal.  The
``NWTNOpenClawAdapter`` (or the test script) decides whether to act on it.

Convergence is conservative: a single "no changes needed" in one round is
not enough.  The agent must repeat the convergence signal for at least
``min_consecutive_rounds`` (default 2) before it counts.  This prevents
false positives from "nothing to do this round" mid-session statements.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Convergence signal patterns ──────────────────────────────────────────────
# Conservative patterns — each requires a clear, unambiguous completion signal.
# Casual "no issues" during active work does NOT trigger convergence.
# The agent must use language that clearly marks the task as DONE.

_CONVERGENCE_PATTERNS = re.compile(
    r'('
    # Security clearance
    r'SECURITY\s+CLEARED'
    r'|security\s+confidence\s+level\s*[:\s]+[4-5]'
    r'|all\s+(previously\s+identified\s+)?(issues?|concerns?)\s+(are\s+)?(resolved|addressed|fixed)'
    # Implementation complete
    r'|implementation\s+is\s+(correct|complete|production.ready)'
    r'|no\s+(new|further|additional|remaining)\s+(issue|change|update|bug|concern|gap)s?'
    r'|no\s+changes?\s+(are\s+)?(needed|required|necessary)'
    r'|all\s+(tests?|milestones?|criteria)\s+(pass|met|complete|satisfied)'
    # Formula/schema done
    r'|formula\s+is\s+(correct|validated|verified|finalized)'
    r'|schema\s+is\s+(stable|finalized|complete)'
    r'|TASK\s+COMPLETE'
    r')',
    re.IGNORECASE,
)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ConvergenceSignal:
    """A single convergence observation for one agent in one round."""
    agent_id: str
    round_number: int
    signal_text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentConvergenceState:
    """Tracks convergence history for a single agent."""
    agent_id: str
    signals: List[ConvergenceSignal] = field(default_factory=list)
    converged_since_round: Optional[int] = None
    """Round number at which stable convergence was first confirmed."""


# ── ConvergenceTracker ────────────────────────────────────────────────────────

class ConvergenceTracker:
    """
    Monitors agent outputs for completion signals and reports when the
    Agent Team has converged.

    Parameters
    ----------
    min_consecutive_rounds : int
        An agent must emit convergence signals in at least this many
        consecutive rounds before being marked as converged.  Default 2.
        Higher values reduce false positives at the cost of later detection.
    """

    def __init__(self, min_consecutive_rounds: int = 2) -> None:
        self._min_rounds = min_consecutive_rounds
        self._states: Dict[str, AgentConvergenceState] = {}
        self._registered_agents: Set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str) -> None:
        """Register an agent that must converge before the session can end."""
        self._registered_agents.add(agent_id)
        if agent_id not in self._states:
            self._states[agent_id] = AgentConvergenceState(agent_id=agent_id)

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def scan_output(self, agent_id: str, output: str, round_number: int) -> bool:
        """
        Scan an agent's round output for convergence signals.

        Parameters
        ----------
        agent_id : str
        output : str
            The full text of the agent's output for this round.
        round_number : int
            The round index (0-based).

        Returns
        -------
        bool
            True if a convergence signal was detected this round.
        """
        matches = _CONVERGENCE_PATTERNS.findall(output)
        if not matches:
            # No signal — reset if agent was previously signalling
            state = self._states.setdefault(agent_id, AgentConvergenceState(agent_id))
            # Only reset if they hadn't reached stable convergence
            if state.converged_since_round is None and state.signals:
                gap = round_number - state.signals[-1].round_number
                if gap > 1:
                    logger.debug(
                        "ConvergenceTracker: %s convergence streak broken at R%d",
                        agent_id, round_number,
                    )
                    state.signals.clear()
            return False

        # Signal found
        signal_text = matches[0] if isinstance(matches[0], str) else matches[0][0]
        state = self._states.setdefault(agent_id, AgentConvergenceState(agent_id))
        state.signals.append(ConvergenceSignal(
            agent_id=agent_id,
            round_number=round_number,
            signal_text=signal_text,
        ))

        # Check for stable convergence (consecutive rounds)
        if state.converged_since_round is None:
            consecutive = self._count_consecutive(state.signals, round_number)
            if consecutive >= self._min_rounds:
                state.converged_since_round = round_number - self._min_rounds + 1
                logger.info(
                    "ConvergenceTracker: %s CONVERGED at R%d (signal: '%s')",
                    agent_id, state.converged_since_round, signal_text,
                )

        return True

    # ------------------------------------------------------------------
    # Convergence queries
    # ------------------------------------------------------------------

    def is_agent_converged(self, agent_id: str) -> bool:
        """Return True if the agent has reached stable convergence."""
        state = self._states.get(agent_id)
        return state is not None and state.converged_since_round is not None

    def all_converged(self, agent_ids: Optional[List[str]] = None) -> bool:
        """
        Return True if ALL specified agents (or all registered agents) are
        converged.

        Parameters
        ----------
        agent_ids : list of str, optional
            Subset of agents to check.  Defaults to all registered agents.
        """
        agents = agent_ids if agent_ids is not None else list(self._registered_agents)
        if not agents:
            return False
        return all(self.is_agent_converged(a) for a in agents)

    def converged_agents(self) -> List[str]:
        """List of agents that have reached stable convergence."""
        return [
            a for a in self._registered_agents
            if self.is_agent_converged(a)
        ]

    def pending_agents(self) -> List[str]:
        """List of registered agents that have NOT yet converged."""
        return [
            a for a in self._registered_agents
            if not self.is_agent_converged(a)
        ]

    def convergence_summary(self) -> dict:
        """Return a summary dict suitable for logging or session metadata."""
        return {
            "all_converged": self.all_converged(),
            "converged": {
                a: self._states[a].converged_since_round
                for a in self.converged_agents()
            },
            "pending": self.pending_agents(),
            "signals_by_agent": {
                a: [
                    {"round": s.round_number, "signal": s.signal_text}
                    for s in self._states.get(a, AgentConvergenceState(a)).signals
                ]
                for a in self._registered_agents
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _count_consecutive(signals: List[ConvergenceSignal], current_round: int) -> int:
        """Count how many consecutive rounds ending at current_round have signals."""
        if not signals:
            return 0
        count = 0
        expected = current_round
        for sig in reversed(signals):
            if sig.round_number == expected:
                count += 1
                expected -= 1
            else:
                break
        return count
