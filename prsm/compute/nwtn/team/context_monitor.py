"""
Context Pressure Monitor
========================

Monitors token usage per agent and detects context pressure thresholds,
enabling context resets with structured handoff artifacts for long-running
agentic sessions.

Based on Anthropic's harness design article on "context anxiety" — as the
context window fills, agents prematurely wrap up work. The solution is a
context reset: clearing the context window and starting a fresh agent with
a structured handoff artifact that carries the previous agent's state.

Thresholds:
- WARNING (70%): log warning, flag agent for handoff prep
- CRITICAL (85%): trigger context reset for agent
- HARD_LIMIT (95%): force immediate reset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextPressureLevel(Enum):
    """Pressure level thresholds for context window usage."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    HARD_LIMIT = "hard_limit"


@dataclass
class AgentTokenState:
    """Tracks token usage state for a single agent."""
    total_tokens: int = 0
    current_round: int = 0
    highest_round_seen: int = 0
    pressure_level: ContextPressureLevel = ContextPressureLevel.OK
    reset_count: int = 0


class ContextPressureMonitor:
    """
    Monitors token usage per agent and detects context pressure thresholds.

    This class tracks how many tokens each agent has accumulated and determines
    when a context reset should be triggered. When an agent exceeds the CRITICAL
    threshold, it builds a structured handoff artifact that can be used to
    continue work with a fresh agent.

    Example
    -------
    >>> monitor = ContextPressureMonitor(context_window_size=200_000)
    >>> level = monitor.record_tokens("agent-1", 1, 150_000)
    >>> level
    <ContextPressureLevel.WARNING: 'warning'>
    >>> if level == ContextPressureLevel.CRITICAL:
    ...     artifact = monitor.build_handoff_artifact("agent-1", [...], "Plan summary")
    ...     monitor.reset_agent_context("agent-1")
    """

    # Threshold percentages
    WARNING_THRESHOLD = 0.70   # 70%
    CRITICAL_THRESHOLD = 0.85  # 85%
    HARD_LIMIT_THRESHOLD = 0.95  # 95%

    def __init__(self, context_window_size: int = 200_000):
        """
        Initialize the context pressure monitor.

        Parameters
        ----------
        context_window_size : int
            The maximum context window size in tokens. Defaults to 200k tokens
            (typical for modern large context models).
        """
        self._context_window_size = context_window_size
        self._agent_states: Dict[str, AgentTokenState] = {}

    @property
    def context_window_size(self) -> int:
        """Return the configured context window size."""
        return self._context_window_size

    def _get_or_create_state(self, agent_id: str) -> AgentTokenState:
        """Get or create the token state for an agent."""
        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = AgentTokenState()
        return self._agent_states[agent_id]

    def _calculate_pressure_level(self, token_count: int) -> ContextPressureLevel:
        """Determine the pressure level for a given token count."""
        ratio = token_count / self._context_window_size

        if ratio >= self.HARD_LIMIT_THRESHOLD:
            return ContextPressureLevel.HARD_LIMIT
        elif ratio >= self.CRITICAL_THRESHOLD:
            return ContextPressureLevel.CRITICAL
        elif ratio >= self.WARNING_THRESHOLD:
            return ContextPressureLevel.WARNING
        else:
            return ContextPressureLevel.OK

    def record_tokens(
        self,
        agent_id: str,
        round_num: int,
        token_count: int,
    ) -> ContextPressureLevel:
        """
        Record token usage for an agent and return current pressure level.

        This method should be called after each agent interaction to track
        cumulative token usage. The pressure level is determined by comparing
        the total tokens against the configured thresholds.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.
        round_num : int
            The current round number (0-indexed or 1-indexed).
        token_count : int
            The number of tokens used in this interaction.

        Returns
        -------
        ContextPressureLevel
            The current pressure level for this agent.
        """
        state = self._get_or_create_state(agent_id)

        # Accumulate tokens
        state.total_tokens += token_count
        state.current_round = round_num
        state.highest_round_seen = max(state.highest_round_seen, round_num)

        # Calculate new pressure level
        new_level = self._calculate_pressure_level(state.total_tokens)
        old_level = state.pressure_level
        state.pressure_level = new_level

        # Log warnings for pressure changes
        if new_level != old_level:
            ratio = state.total_tokens / self._context_window_size
            if new_level == ContextPressureLevel.WARNING:
                logger.warning(
                    "ContextPressureMonitor: agent=%s reached WARNING threshold "
                    "(%.1f%%, %d tokens)",
                    agent_id, ratio * 100, state.total_tokens
                )
            elif new_level == ContextPressureLevel.CRITICAL:
                logger.warning(
                    "ContextPressureMonitor: agent=%s reached CRITICAL threshold "
                    "(%.1f%%, %d tokens) - reset recommended",
                    agent_id, ratio * 100, state.total_tokens
                )
            elif new_level == ContextPressureLevel.HARD_LIMIT:
                logger.warning(
                    "ContextPressureMonitor: agent=%s reached HARD_LIMIT threshold "
                    "(%.1f%%, %d tokens) - immediate reset required",
                    agent_id, ratio * 100, state.total_tokens
                )

        return new_level

    def get_pressure_level(self, agent_id: str) -> ContextPressureLevel:
        """
        Get the current pressure level for an agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.

        Returns
        -------
        ContextPressureLevel
            The current pressure level, or OK if agent not found.
        """
        state = self._agent_states.get(agent_id)
        return state.pressure_level if state else ContextPressureLevel.OK

    def get_token_count(self, agent_id: str) -> int:
        """
        Get the current token count for an agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.

        Returns
        -------
        int
            The total token count, or 0 if agent not found.
        """
        state = self._agent_states.get(agent_id)
        return state.total_tokens if state else 0

    def agents_needing_reset(self) -> List[str]:
        """
        Return agent_ids that have exceeded the CRITICAL threshold.

        This method returns all agents that should have their context reset.
        Agents at HARD_LIMIT are also included as they definitely need reset.

        Returns
        -------
        List[str]
            List of agent IDs needing context reset.
        """
        return [
            agent_id
            for agent_id, state in self._agent_states.items()
            if state.pressure_level in (
                ContextPressureLevel.CRITICAL,
                ContextPressureLevel.HARD_LIMIT,
            )
        ]

    def agents_at_warning(self) -> List[str]:
        """
        Return agent_ids at WARNING level (but not yet critical).

        Returns
        -------
        List[str]
            List of agent IDs at warning level.
        """
        return [
            agent_id
            for agent_id, state in self._agent_states.items()
            if state.pressure_level == ContextPressureLevel.WARNING
        ]

    def build_handoff_artifact(
        self,
        agent_id: str,
        whiteboard_entries: List[dict],
        meta_plan_summary: str,
        goal: Optional[str] = None,
    ) -> dict:
        """
        Build a structured handoff artifact for context reset.

        The handoff artifact carries the essential state from the previous
        agent to the fresh agent, enabling seamless continuation of work.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent being reset.
        whiteboard_entries : List[dict]
            Recent whiteboard entries to extract key decisions from.
            Each entry should have 'content' or 'chunk' field.
        meta_plan_summary : str
            Summary of the current meta-plan state.
        goal : str, optional
            The original session goal. If not provided, will be marked
            as "not available".

        Returns
        -------
        dict
            Structured handoff artifact with keys:
            - goal: original session goal
            - completed_work: summary of what's been done
            - pending_criteria: unmet acceptance criteria
            - key_decisions: important decisions from whiteboard
            - next_steps: concrete next actions for the fresh agent
            - round_number: current round
            - reset_count: how many times this agent has been reset
        """
        state = self._get_or_create_state(agent_id)

        # Extract key decisions from whiteboard entries
        key_decisions = []
        for entry in whiteboard_entries[:10]:  # Take up to 10 most recent
            content = entry.get("content") or entry.get("chunk") or ""
            if content:
                # Extract decision-like content (lines starting with decision markers)
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith(("DECISION:", "DECIDED:", "✓", "✔", "- DECISION")):
                        key_decisions.append(line)
                    elif "decision" in line.lower() and len(line) > 20:
                        key_decisions.append(line)

        # Build the handoff artifact
        artifact = {
            "goal": goal or "Goal not available in handoff context",
            "completed_work": self._summarize_completed_work(whiteboard_entries),
            "pending_criteria": self._extract_pending_criteria(whiteboard_entries),
            "key_decisions": key_decisions[:5],  # Top 5 decisions
            "next_steps": self._suggest_next_steps(whiteboard_entries, agent_id),
            "round_number": state.current_round,
            "reset_count": state.reset_count,
            "agent_id": agent_id,
            "tokens_before_reset": state.total_tokens,
            "meta_plan_summary": meta_plan_summary,
        }

        logger.info(
            "ContextPressureMonitor: built handoff artifact for agent=%s "
            "(round=%d, tokens=%d, decisions=%d)",
            agent_id, state.current_round, state.total_tokens, len(key_decisions)
        )

        return artifact

    def _summarize_completed_work(self, whiteboard_entries: List[dict]) -> str:
        """Summarize completed work from whiteboard entries."""
        if not whiteboard_entries:
            return "No completed work recorded."

        # Look for completion markers
        completed = []
        for entry in whiteboard_entries:
            content = entry.get("content") or entry.get("chunk") or ""
            for line in content.split("\n"):
                line = line.strip()
                if any(marker in line for marker in
                       ["COMPLETE", "DONE", "✓", "✔", "FINISHED", "IMPLEMENTED"]):
                    if len(line) > 10:
                        completed.append(line)

        if completed:
            return "\n".join(f"• {item}" for item in completed[:5])
        else:
            return f"Session has {len(whiteboard_entries)} whiteboard entries. Review recent context."

    def _extract_pending_criteria(self, whiteboard_entries: List[dict]) -> List[str]:
        """Extract pending/unmet acceptance criteria from whiteboard."""
        pending = []

        for entry in whiteboard_entries:
            content = entry.get("content") or entry.get("chunk") or ""
            for line in content.split("\n"):
                line = line.strip()
                # Look for TODO, FIXME, or incomplete markers
                if any(marker in line.upper() for marker in
                       ["TODO", "FIXME", "PENDING", "INCOMPLETE", "NOT DONE", "REMAINING"]):
                    if len(line) > 10:
                        pending.append(line)
                # Also look for unchecked boxes
                elif line.startswith("[ ]") or line.startswith("- [ ]"):
                    pending.append(line)

        return pending[:10]  # Top 10 pending items

    def _suggest_next_steps(
        self,
        whiteboard_entries: List[dict],
        agent_id: str,
    ) -> List[str]:
        """Suggest concrete next steps for the fresh agent."""
        # Extract actionable items from recent entries
        next_steps = []

        for entry in whiteboard_entries[-5:]:  # Last 5 entries
            content = entry.get("content") or entry.get("chunk") or ""
            for line in content.split("\n"):
                line = line.strip()
                # Look for action items
                if line.startswith(("ACTION:", "NEXT:", "STEP:")):
                    next_steps.append(line)
                elif any(marker in line.upper() for marker in
                         ["SHOULD", "MUST", "NEED TO", "REQUIRED"]):
                    if len(line) > 15 and len(line) < 200:
                        next_steps.append(line)

        if next_steps:
            return next_steps[:5]
        else:
            return [
                "Review recent whiteboard entries for context",
                "Continue work on the session goal",
                "Check for any pending acceptance criteria",
            ]

    def reset_agent_context(self, agent_id: str) -> None:
        """
        Mark an agent as reset and clear its token accumulator.

        This method should be called after a context reset has been performed
        and a fresh agent has been started with the handoff artifact.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent being reset.
        """
        state = self._get_or_create_state(agent_id)

        # Increment reset count before clearing
        reset_count = state.reset_count + 1
        old_tokens = state.total_tokens

        # Clear token state but keep reset count and round info
        state.total_tokens = 0
        state.pressure_level = ContextPressureLevel.OK
        state.reset_count = reset_count

        logger.info(
            "ContextPressureMonitor: reset context for agent=%s "
            "(previous_tokens=%d, reset_count=%d)",
            agent_id, old_tokens, reset_count
        )

    def get_all_agents(self) -> Dict[str, AgentTokenState]:
        """
        Get the state of all tracked agents.

        Returns
        -------
        Dict[str, AgentTokenState]
            Dictionary mapping agent IDs to their token state.
        """
        return dict(self._agent_states)

    def get_reset_count(self, agent_id: str) -> int:
        """
        Get the number of times an agent has been reset.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.

        Returns
        -------
        int
            Number of resets, or 0 if agent not found.
        """
        state = self._agent_states.get(agent_id)
        return state.reset_count if state else 0

    def clear_all(self) -> None:
        """Clear all tracked agent states. Useful for testing."""
        self._agent_states.clear()
        logger.debug("ContextPressureMonitor: cleared all agent states")
