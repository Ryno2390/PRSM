"""
Whiteboard Query Interface
==========================

Structured read interface for agents to consume the Active Whiteboard.

Agents use this module to:
  1. Load shared situational awareness at the start of a work session.
  2. Check what teammates have already discovered before starting a task.
  3. Find the most relevant whiteboard entries for a specific topic.
  4. Get the full ``WhiteboardSnapshot`` for handoff to the Nightly
     Synthesis agent (Sub-phase 10.4).

All methods are read-only.  Only the ``WhiteboardStore`` writes entries.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .schema import WhiteboardEntry, WhiteboardSnapshot
from .store import WhiteboardStore

logger = logging.getLogger(__name__)


class WhiteboardQuery:
    """
    Read-only query facade over a ``WhiteboardStore``.

    Parameters
    ----------
    store : WhiteboardStore
        An open store instance.
    """

    def __init__(self, store: WhiteboardStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Primary agent-facing methods
    # ------------------------------------------------------------------

    async def compressed_state(
        self,
        session_id: str,
        max_chars: int = 4000,
        strategy: str = "hybrid",
    ) -> str:
        """
        Return the whiteboard as a compact, agent-readable string.

        This is the primary method agents call to understand shared context.
        The returned string is also what the BSC predictor receives as its
        *context* parameter for evaluating new chunks.

        Parameters
        ----------
        session_id : str
        max_chars : int
            Approximate character budget.  Longer whiteboards are truncated
            using *strategy*.
        strategy : str
            ``"recent"`` | ``"top"`` (highest surprise) | ``"hybrid"`` (default).

        Returns
        -------
        str
        """
        return await self._store.compressed_state(
            session_id=session_id,
            max_chars=max_chars,
            strategy=strategy,
        )

    async def snapshot(self, session_id: str) -> WhiteboardSnapshot:
        """
        Return the complete ``WhiteboardSnapshot`` for *session_id*.

        Used by the Nightly Synthesis agent to consume the full session
        history at end-of-day.
        """
        return await self._store.snapshot(session_id)

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    async def recent(
        self, session_id: str, n: int = 10
    ) -> List[WhiteboardEntry]:
        """Return the *n* most recent whiteboard entries."""
        return await self._store.get_recent(session_id, n)

    async def by_agent(
        self, session_id: str, source_agent: str
    ) -> List[WhiteboardEntry]:
        """Return all entries contributed by a specific agent."""
        return await self._store.get_by_agent(session_id, source_agent)

    async def top_surprise(
        self, session_id: str, n: int = 10
    ) -> List[WhiteboardEntry]:
        """
        Return the *n* most surprising entries.

        Useful for a new agent joining the team mid-session who wants to
        quickly understand the most important discoveries.
        """
        return await self._store.get_top_surprise(session_id, n)

    async def search(
        self, session_id: str, keyword: str
    ) -> List[WhiteboardEntry]:
        """
        Return all entries whose *chunk* contains *keyword* (case-insensitive).

        Simple substring search — sufficient for agent-driven lookups.
        For semantic search, use the BSC's ``SemanticDeduplicator`` directly
        against the entry embeddings.
        """
        all_entries = await self._store.get_all(session_id)
        keyword_lower = keyword.lower()
        return [e for e in all_entries if keyword_lower in e.chunk.lower()]

    async def entry_count(self, session_id: str) -> int:
        """Return the total number of entries in the session."""
        return await self._store.entry_count(session_id)

    # ------------------------------------------------------------------
    # Onboarding helper
    # ------------------------------------------------------------------

    async def onboarding_brief(
        self,
        session_id: str,
        agent_name: str,
        max_chars: int = 3000,
    ) -> str:
        """
        Generate an onboarding brief for a new agent joining mid-session.

        Returns a structured string that gives the agent:
          - The current whiteboard state (compressed)
          - The agents currently on the team
          - The top-surprise entries it should be aware of

        Parameters
        ----------
        session_id : str
        agent_name : str
            The new agent's identifier (used for personalisation).
        max_chars : int
            Character budget for the onboarding brief.
        """
        snap = await self.snapshot(session_id)
        if snap.entry_count == 0:
            return (
                f"Welcome to session {session_id}, {agent_name}. "
                "The whiteboard is empty — you are among the first to join."
            )

        top = snap.top_surprise(5)
        agents_list = ", ".join(snap.agents) if snap.agents else "none yet"

        brief_parts = [
            f"=== Onboarding Brief for {agent_name} ===",
            f"Session: {session_id}",
            f"Total whiteboard entries: {snap.entry_count}",
            f"Active team members: {agents_list}",
            "",
            "--- Top Discoveries (highest surprise) ---",
        ]
        for e in top:
            brief_parts.append(f"  [{e.agent_short}] {e.chunk[:200]}")

        brief_parts += [
            "",
            "--- Current Whiteboard State ---",
        ]

        # Reserve space for the state section
        header_text = "\n".join(brief_parts) + "\n"
        remaining = max_chars - len(header_text)
        state = await self.compressed_state(session_id, max_chars=max(500, remaining))
        brief_parts.append(state)

        return "\n".join(brief_parts)
