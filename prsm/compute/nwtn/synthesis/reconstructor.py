"""
Narrative Re-constructor (Nightly Synthesis Agent)
===================================================

At the end of each working session, the Re-constructor reads the Active
Whiteboard's ``WhiteboardSnapshot`` and synthesises it into a coherent,
human-readable narrative suitable for the Project Ledger.

This solves the "disconnected bullet points" problem: raw whiteboard entries
are high-surprise fragments written by different agents at different times.
The reconstructor weaves them into a document that the *next* session's
agents can read and immediately understand.

Output format
-------------
The synthesis is a structured Markdown narrative:

    ## Session Summary — YYYY-MM-DD (sess-…)

    **Objective (from MetaPlan):** …

    ### What was accomplished
    …

    ### Key pivots and discoveries
    (high-surprise entries, i.e. entries with surprise_score ≥ pivot_threshold)

    ### What is still pending
    …

    ### Meta-plan milestone status
    - Milestone 1 (BSC Core) — COMPLETE ✓
    - Milestone 2 (Whiteboard) — IN PROGRESS

    ### Agent contributions
    | Agent | Entries | Avg. Surprise |
    |-------|---------|---------------|
    | coder | 12 | 0.72 |

Fallback
--------
If no LLM backend is available, a ``TemplateSynthesizer`` builds the
narrative directly from the snapshot data (no prose — structured headings
and bullet points only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PIVOT_THRESHOLD = 0.75
"""Surprise score above which a whiteboard entry is highlighted as a 'pivot'."""


# ======================================================================
# Data models
# ======================================================================

@dataclass
class SynthesisResult:
    """
    Output of the Re-constructor Agent for one session.

    This is the content that gets appended to the Project Ledger.
    """
    session_id: str
    narrative: str
    """Full Markdown narrative — the Project Ledger entry body."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    whiteboard_entry_count: int = 0
    agents_involved: List[str] = field(default_factory=list)
    pivot_count: int = 0
    """Number of high-surprise entries (surprise_score ≥ PIVOT_THRESHOLD)."""
    llm_assisted: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# NarrativeSynthesizer
# ======================================================================

class NarrativeSynthesizer:
    """
    Synthesises a ``WhiteboardSnapshot`` into a ``SynthesisResult``.

    Parameters
    ----------
    backend_registry : optional
        A ``BackendRegistry`` for LLM-powered narrative generation.
        Falls back to the template synthesiser if None or on error.
    pivot_threshold : float
        Surprise score above which an entry is treated as a key pivot.
    """

    def __init__(
        self,
        backend_registry=None,
        pivot_threshold: float = PIVOT_THRESHOLD,
    ) -> None:
        self._backend = backend_registry
        self._pivot_threshold = pivot_threshold

    async def synthesise(
        self,
        snapshot,            # WhiteboardSnapshot
        meta_plan=None,      # MetaPlan (optional — provides milestone context)
        previous_summary: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Generate a ``SynthesisResult`` for the session.

        Parameters
        ----------
        snapshot : WhiteboardSnapshot
        meta_plan : MetaPlan, optional
        previous_summary : str, optional
            The previous ledger entry's narrative — gives the LLM temporal
            context for what was done before this session.

        Returns
        -------
        SynthesisResult
        """
        if snapshot.entry_count == 0:
            return self._empty_synthesis(snapshot.session_id)

        if self._backend is not None:
            try:
                return await self._llm_synthesise(snapshot, meta_plan, previous_summary)
            except Exception as exc:
                logger.warning(
                    "LLM synthesis failed (%s); using template synthesiser", exc
                )

        return self._template_synthesise(snapshot, meta_plan)

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    async def _llm_synthesise(
        self,
        snapshot,
        meta_plan,
        previous_summary: Optional[str],
    ) -> SynthesisResult:
        pivots = [e for e in snapshot.entries if e.surprise_score >= self._pivot_threshold]
        regular = [e for e in snapshot.entries if e.surprise_score < self._pivot_threshold]

        pivot_block = "\n".join(
            f"- [{e.agent_short}] {e.chunk}" for e in pivots[:10]
        )
        regular_block = "\n".join(
            f"- [{e.agent_short}] {e.chunk}" for e in regular[:20]
        )

        meta_block = ""
        if meta_plan:
            meta_block = (
                f"\nMetaPlan objective: {meta_plan.objective}\n"
                f"Milestones: {', '.join(m.title for m in meta_plan.milestones)}"
            )

        prev_block = (
            f"\nPrevious session summary:\n{previous_summary[:800]}"
            if previous_summary
            else ""
        )

        prompt = (
            "You are NWTN's Nightly Synthesis Agent. Write a concise, coherent "
            "session summary in Markdown. The summary will be appended to the "
            "Project Ledger and read by agents starting fresh tomorrow.\n\n"
            f"Session ID: {snapshot.session_id}\n"
            f"Date: {snapshot.last_updated or snapshot.created_at}\n"
            f"Agents: {', '.join(snapshot.agents)}\n"
            f"Total whiteboard entries: {snapshot.entry_count}\n"
            f"{meta_block}{prev_block}\n\n"
            f"Key pivots/discoveries (high surprise):\n{pivot_block or '(none)'}\n\n"
            f"Routine progress entries:\n{regular_block or '(none)'}\n\n"
            "Write a 3-5 paragraph narrative covering: (1) what was accomplished, "
            "(2) any pivots or unexpected findings, (3) what is still pending, "
            "(4) milestone status. Be specific, not generic. Use Markdown headings."
        )

        result = await self._backend.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.4,
        )

        narrative = _format_narrative(
            content=result.text.strip(),
            snapshot=snapshot,
            meta_plan=meta_plan,
        )

        return SynthesisResult(
            session_id=snapshot.session_id,
            narrative=narrative,
            whiteboard_entry_count=snapshot.entry_count,
            agents_involved=snapshot.agents,
            pivot_count=len(pivots),
            llm_assisted=True,
        )

    # ------------------------------------------------------------------
    # Template synthesis (no LLM)
    # ------------------------------------------------------------------

    def _template_synthesise(self, snapshot, meta_plan) -> SynthesisResult:
        pivots = [e for e in snapshot.entries if e.surprise_score >= self._pivot_threshold]
        regular = [e for e in snapshot.entries if e.surprise_score < self._pivot_threshold]

        date_str = (
            snapshot.last_updated.strftime("%Y-%m-%d")
            if snapshot.last_updated
            else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )

        lines: List[str] = []
        lines += [
            f"## Session Summary — {date_str} ({snapshot.session_id})",
            "",
        ]

        if meta_plan:
            lines += [
                f"**Objective:** {meta_plan.objective}",
                "",
            ]

        # Agents table
        lines += ["### Agent Contributions", ""]
        lines += ["| Agent | Entries | Avg. Surprise |", "|-------|---------|---------------|"]
        for agent in snapshot.agents:
            agent_entries = snapshot.entries_by_agent(agent)
            avg_surp = (
                sum(e.surprise_score for e in agent_entries) / len(agent_entries)
                if agent_entries else 0.0
            )
            short = agent.removeprefix("agent/")
            lines.append(f"| {short} | {len(agent_entries)} | {avg_surp:.2f} |")
        lines.append("")

        # Pivots
        if pivots:
            lines += [
                f"### Key Pivots and Discoveries ({len(pivots)} high-surprise events)",
                "",
            ]
            for e in pivots[:15]:
                lines.append(f"- **[{e.agent_short}]** {e.chunk}")
            lines.append("")
        else:
            lines += ["### Pivots", "", "No high-surprise events in this session.", ""]

        # Regular progress
        if regular:
            lines += [
                f"### Progress Entries ({len(regular)} entries)",
                "",
            ]
            for e in regular[:20]:
                lines.append(f"- [{e.agent_short}] {e.chunk}")
            lines.append("")

        # Milestone status
        if meta_plan and meta_plan.milestones:
            lines += ["### Milestone Status", ""]
            for i, ms in enumerate(meta_plan.milestones):
                lines.append(f"- Milestone {i+1}: **{ms.title}** — {ms.description}")
            lines.append("")

        narrative = "\n".join(lines)

        return SynthesisResult(
            session_id=snapshot.session_id,
            narrative=narrative,
            whiteboard_entry_count=snapshot.entry_count,
            agents_involved=snapshot.agents,
            pivot_count=len(pivots),
            llm_assisted=False,
        )

    def _empty_synthesis(self, session_id: str) -> SynthesisResult:
        narrative = (
            f"## Session Summary — {datetime.now(timezone.utc).strftime('%Y-%m-%d')} "
            f"({session_id})\n\n"
            "No whiteboard entries recorded in this session."
        )
        return SynthesisResult(
            session_id=session_id,
            narrative=narrative,
            whiteboard_entry_count=0,
            llm_assisted=False,
        )


# ======================================================================
# Helpers
# ======================================================================

def _format_narrative(content: str, snapshot, meta_plan) -> str:
    """
    Wrap the LLM-generated prose with a standardised header block.
    """
    date_str = (
        snapshot.last_updated.strftime("%Y-%m-%d")
        if snapshot.last_updated
        else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    agents_str = ", ".join(
        a.removeprefix("agent/") for a in snapshot.agents
    ) or "unknown"

    header = (
        f"## Session Summary — {date_str} ({snapshot.session_id})\n\n"
        f"**Session:** `{snapshot.session_id}`  \n"
        f"**Date:** {date_str}  \n"
        f"**Agents:** {agents_str}  \n"
        f"**Whiteboard entries:** {snapshot.entry_count}  \n"
    )
    if meta_plan:
        header += f"**Objective:** {meta_plan.objective}  \n"
    header += "\n---\n\n"

    return header + content
