"""
Live Scribe Agent
=================

Real-time coordinator that runs during NWTN Agent Team working sessions.
Curates the shared whiteboard, manages cross-agent information flow,
detects conflicts, and coordinates the checkpoint lifecycle.

The Live Scribe is the "scribe" of the NWTN architecture — it receives
promoted chunks from the BSC, triages them by priority, distributes them
to agent inboxes, watches for conflicts, and orchestrates checkpoint cycles.

Position in the NWTN pipeline
------------------------------

    BSC Promoter → on_chunk_promoted() → Priority Tagger → Agent Inboxes
                                               ↓
                                        Conflict Detector
                                               ↓
                                        Checkpoint Lifecycle Manager
                                               ↓
                                    NarrativeSynthesizer → ProjectLedger

The Live Scribe complements the Nightly Synthesis Agent:
  - Nightly Synthesis: runs at END of session → produces cohesive narrative
  - Live Scribe: runs DURING session → real-time triage and distribution

Key responsibilities
--------------------
1. **Priority Tagger**: Assigns ROUTINE / IMPORTANT / URGENT priority to
   whiteboard updates based on surprise score, milestone relevance, and
   conflict detection.

2. **Agent Notification System**: Manages per-agent inboxes with priority-aware
   retrieval. ROUTINE items wait for breakpoints; IMPORTANT flags agents;
   URGENT interrupts immediately.

3. **Conflict Detector**: Identifies semantic conflicts between new entries
   and existing whiteboard state (contradictions, superseded information,
   agent disagreements).

4. **Checkpoint Lifecycle Manager**: Monitors convergence signals, triggers
   synthesis when all agents reach checkpoints, appends to Project Ledger,
   and assigns next milestones.

Design principles
-----------------
- All classes work WITHOUT an LLM backend (backend_registry is optional).
- Priority tagging falls back to score thresholds and rule-based heuristics.
- Conflict resolution uses cosine similarity + negation pattern detection.
- Checkpoint lifecycle uses ConvergenceTracker signals + explicit markers.

Quick start
-----------
.. code-block:: python

    from prsm.compute.nwtn.team import LiveScribe
    from prsm.compute.nwtn.bsc import PromotionDecision

    scribe = LiveScribe(
        whiteboard_store=store,
        meta_plan=meta_plan,
        team=team,
        convergence_tracker=tracker,
        narrative_synthesizer=synthesizer,
        project_ledger=ledger,
    )

    # Called by BSCPromoter after promotion
    update = await scribe.on_chunk_promoted(decision)

    # Get context for an agent starting work
    context = await scribe.get_agent_context("agent/coder-20260326")

    # Check and run checkpoint cycle
    result = await scribe.check_and_run_checkpoint()
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from prsm.compute.nwtn.bsc import PromotionDecision
    from prsm.compute.nwtn.whiteboard import WhiteboardStore, WhiteboardEntry
    from prsm.compute.nwtn.team.planner import MetaPlan, Milestone
    from prsm.compute.nwtn.team.assembler import AgentTeam, TeamMember
    from prsm.compute.nwtn.team.convergence import ConvergenceTracker
    from prsm.compute.nwtn.synthesis.reconstructor import NarrativeSynthesizer, SynthesisResult
    from prsm.compute.nwtn.synthesis.ledger import ProjectLedger, LedgerEntry

logger = logging.getLogger(__name__)


# ======================================================================
# Priority System
# ======================================================================

class UpdatePriority(str, Enum):
    """
    Priority level assigned to whiteboard updates.

    Determines how quickly agents receive the update:
      - ROUTINE: Added to inbox, read at next breakpoint.
      - IMPORTANT: Added to inbox AND flagged for immediate attention.
      - URGENT: Injected into agent context immediately (interrupts work).
    """
    ROUTINE = "routine"
    IMPORTANT = "important"
    URGENT = "urgent"


@dataclass
class PrioritizedUpdate:
    """
    A whiteboard entry with priority metadata and distribution info.

    Produced by the Priority Tagger and consumed by the Agent Notification
    System.
    """
    entry: "WhiteboardEntry"
    """The whiteboard entry being prioritized."""

    priority: UpdatePriority
    """Assigned priority level."""

    relevant_agents: List[str] = field(default_factory=list)
    """Agent IDs that should see this update (based on cross-agent relevance)."""

    conflict_detected: bool = False
    """True if this update conflicts with existing whiteboard state."""

    conflict_details: Optional[str] = None
    """Human-readable description of the conflict, if any."""

    milestone_refs: List[int] = field(default_factory=list)
    """Indices into MetaPlan.milestones that this entry references."""

    reason: str = ""
    """Human-readable explanation for the priority assignment."""


# ======================================================================
# Conflict Detection
# ======================================================================

@dataclass
class ConflictReport:
    """
    Report generated when a conflict is detected between whiteboard entries.

    Conflicts can be contradictions, superseded information, or agent
    disagreements.
    """
    new_entry_id: int
    """ID of the newly promoted entry."""

    conflicting_entry_ids: List[int] = field(default_factory=list)
    """IDs of existing entries that conflict with the new entry."""

    conflict_type: str = "contradiction"
    """Type: 'contradiction', 'superseded', 'disagreement'."""

    description: str = ""
    """Human-readable description of the conflict."""

    affected_agents: List[str] = field(default_factory=list)
    """Agents whose work is affected by this conflict."""

    suggested_resolution: Optional[str] = None
    """LLM-suggested resolution (populated only if backend available)."""


@dataclass
class ConflictLog:
    """
    Persistent record of all detected conflicts during a session.

    Stored in memory and optionally serialized for session metadata.
    """
    session_id: str
    conflicts: List[ConflictReport] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add(self, report: ConflictReport) -> None:
        """Add a conflict report to the log."""
        self.conflicts.append(report)

    @property
    def count(self) -> int:
        return len(self.conflicts)

    def conflicts_for_agent(self, agent_id: str) -> List[ConflictReport]:
        """Return all conflicts involving a specific agent."""
        return [c for c in self.conflicts if agent_id in c.affected_agents]


class ConflictDetector:
    """
    Detects semantic conflicts between whiteboard entries.

    Uses a combination of:
      - Cosine similarity (via SemanticDeduplicator's embedding model)
      - Negation pattern detection
      - Contradiction keyword matching

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity to consider two entries as semantically
        related (before checking for negation). Default 0.75.
    backend_registry : optional
        LLM backend for enhanced conflict resolution. Falls back to
        rule-based heuristics if None.
    """

    # Negation patterns that indicate contradiction when found near
    # semantically similar content.
    NEGATION_PATTERNS = [
        r'\bnot\b',
        r'\bno\s+(?:longer|more|longer)\b',
        r'\bnever\b',
        r'\bincorrect(?:ly)?\b',
        r'\bwrong(?:ly)?\b',
        r'\binvalid(?:ated)?\b',
        r'\bdeprecated\b',
        r'\babandoned\b',
        r'\breplaced\b',
        r'\binstead\b',
        r'\bhowever\b',
        r'\bactually\b',
        r'\bcontrary\b',
        r'\bcontra-indicated\b',
        r'\bdisabled\b',
        r'\bremoved\b',
        r'\bdeleted\b',
        r'\bobsolete\b',
        r'\boutdated\b',
        r'\bsuperseded\b',
        r'\bvoid\b',
        r'\bcancelled?\b',
        r'\breverted?\b',
        r'\brolled?\s+back\b',
        r'\bdoesn\'t\b',
        r'\bdoes\s+not\b',
        r'\bisn\'t\b',
        r'\bis\s+not\b',
        r'\baren\'t\b',
        r'\bare\s+not\b',
        r'\bwasn\'t\b',
        r'\bwas\s+not\b',
        r'\bweren\'t\b',
        r'\bwere\s+not\b',
        r'\bshouldn\'t\b',
        r'\bshould\s+not\b',
        r'\bcannot\b',
        r'\bcan\s+not\b',
        r'\bwon\'t\b',
        r'\bwill\s+not\b',
        r'\bdon\'t\b',
        r'\bdo\s+not\b',
    ]

    # Compiled regex for performance
    _NEGATION_RE = re.compile('|'.join(NEGATION_PATTERNS), re.IGNORECASE)

    # Contradiction indicators that suggest direct opposition
    CONTRADICTION_PATTERNS = [
        (r'\b(?:is|are|was|were)\s+(?:now|currently)\s+(\w+)', r'\b(?:is|are|was|were)\s+not\s+\1'),
        (r'\b(?:should|must|need\s+to)\s+(\w+)', r'\b(?:should\s+not|must\s+not|don\'t)\s+\1'),
    ]

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        backend_registry=None,
        embedding_model: Optional[Any] = None,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._backend = backend_registry
        self._embedding_model = embedding_model

    async def check_conflicts(
        self,
        new_entry: "WhiteboardEntry",
        existing_entries: List["WhiteboardEntry"],
    ) -> Optional[ConflictReport]:
        """
        Check if a new whiteboard entry conflicts with existing entries.

        Parameters
        ----------
        new_entry : WhiteboardEntry
            The newly promoted entry to check.
        existing_entries : List[WhiteboardEntry]
            Existing whiteboard entries to compare against.

        Returns
        -------
        ConflictReport | None
            A conflict report if a conflict is detected, else None.
        """
        if not existing_entries:
            return None

        new_text = new_entry.chunk.lower()
        new_has_negation = bool(self._NEGATION_RE.search(new_entry.chunk))

        conflicting_ids: List[int] = []
        conflict_type = "contradiction"
        best_description = ""
        affected_agents: List[str] = []

        for existing in existing_entries:
            if existing.id == new_entry.id:
                continue

            existing_text = existing.chunk.lower()
            existing_has_negation = bool(self._NEGATION_RE.search(existing.chunk))

            # Compute semantic similarity
            similarity = await self._compute_similarity(
                new_entry.chunk, existing.chunk
            )

            if similarity >= self._similarity_threshold:
                # High similarity + asymmetric negation = conflict
                if new_has_negation != existing_has_negation:
                    conflicting_ids.append(existing.id)
                    affected_agents.append(existing.source_agent)
                    conflict_type = "contradiction"
                    best_description = (
                        f"New entry contradicts existing entry #{existing.id}: "
                        f"similar content ({similarity:.2f}) but negation detected."
                    )
                    logger.debug(
                        "ConflictDetector: contradiction found between #%d and #%d "
                        "(similarity=%.2f, negation asymmetry)",
                        new_entry.id, existing.id, similarity,
                    )

                # Both have negation but different stance = disagreement
                elif new_has_negation and existing_has_negation:
                    # Check if they're negating different things
                    if not self._same_negation_target(new_entry.chunk, existing.chunk):
                        conflicting_ids.append(existing.id)
                        affected_agents.append(existing.source_agent)
                        conflict_type = "disagreement"
                        best_description = (
                            f"Agents disagree: entry #{existing.id} and new entry "
                            f"have conflicting positions (similarity={similarity:.2f})."
                        )

        if not conflicting_ids:
            return None

        # Deduplicate affected agents
        affected_agents = list(set(affected_agents))

        # Try LLM for resolution suggestion
        suggested_resolution = None
        if self._backend is not None:
            suggested_resolution = await self._llm_resolution(
                new_entry, 
                [e for e in existing_entries if e.id in conflicting_ids],
                best_description,
            )

        return ConflictReport(
            new_entry_id=new_entry.id,
            conflicting_entry_ids=conflicting_ids,
            conflict_type=conflict_type,
            description=best_description,
            affected_agents=affected_agents,
            suggested_resolution=suggested_resolution,
        )

    async def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two text strings.

        Uses the embedding model if available; falls back to simple
        token overlap heuristic.
        """
        if self._embedding_model is not None:
            try:
                # Assume embedding model has an encode method
                emb1 = await self._embedding_model.encode(text1)
                emb2 = await self._embedding_model.encode(text2)
                # Cosine similarity
                dot = sum(a * b for a, b in zip(emb1, emb2))
                norm1 = sum(a * a for a in emb1) ** 0.5
                norm2 = sum(b * b for b in emb2) ** 0.5
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot / (norm1 * norm2)
            except Exception as exc:
                logger.warning("Embedding similarity failed: %s", exc)

        # Fallback: token overlap (Jaccard similarity)
        tokens1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
        tokens2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def _same_negation_target(self, text1: str, text2: str) -> bool:
        """
        Check if two negated texts are negating the same thing.

        This is a heuristic to distinguish "disagreement" from "same negation".
        """
        # Extract the word immediately following negation patterns
        def get_negation_targets(text: str) -> Set[str]:
            targets = set()
            for match in self._NEGATION_RE.finditer(text):
                # Get next few words after the negation
                start = match.end()
                remaining = text[start:start+30].lower()
                words = re.findall(r'\b\w{3,}\b', remaining)[:3]
                targets.update(words)
            return targets

        targets1 = get_negation_targets(text1)
        targets2 = get_negation_targets(text2)
        return bool(targets1 & targets2)

    async def _llm_resolution(
        self,
        new_entry: "WhiteboardEntry",
        conflicting_entries: List["WhiteboardEntry"],
        description: str,
    ) -> Optional[str]:
        """Ask LLM for a suggested conflict resolution."""
        if not conflicting_entries:
            return None

        try:
            conflict_block = "\n".join(
                f"- [{e.agent_short}] #{e.id}: {e.chunk[:200]}"
                for e in conflicting_entries[:3]
            )
            prompt = (
                f"A conflict has been detected between agent outputs:\n\n"
                f"New entry [{new_entry.agent_short}]: {new_entry.chunk[:300]}\n\n"
                f"Conflicting with:\n{conflict_block}\n\n"
                f"Description: {description}\n\n"
                f"Suggest a one-sentence resolution or clarification. "
                f"If this is a genuine disagreement, suggest how to proceed."
            )
            result = await self._backend.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.3,
            )
            return result.text.strip()
        except Exception as exc:
            logger.warning("LLM conflict resolution failed: %s", exc)
            return None


# ======================================================================
# Agent Notification System
# ======================================================================

@dataclass
class InboxEntry:
    """A single entry in an agent's inbox."""
    update: PrioritizedUpdate
    read: bool = False
    delivered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentInboxState:
    """Internal state for a single agent's inbox."""
    agent_id: str
    entries: List[InboxEntry] = field(default_factory=list)
    last_read_at: Optional[datetime] = None

    def pending_counts(self) -> Dict[str, int]:
        """Return counts of unread entries by priority."""
        counts = {"routine": 0, "important": 0, "urgent": 0}
        for ie in self.entries:
            if not ie.read:
                key = ie.update.priority.value
                counts[key] = counts.get(key, 0) + 1
        return counts


class AgentInbox:
    """
    Per-agent update queue with priority-aware retrieval.

    Manages how agents receive whiteboard updates based on the hybrid
    sharing model:
      - ROUTINE: queued for next breakpoint
      - IMPORTANT: queued + flagged
      - URGENT: immediate delivery marker

    Parameters
    ----------
    session_id : str
        Identifier of the working session.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._inboxes: Dict[str, AgentInboxState] = {}
        self._urgent_flags: Dict[str, bool] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(self, update: PrioritizedUpdate) -> None:
        """
        Push a prioritized update to the relevant agent inboxes.

        ROUTINE and IMPORTANT: added to each agent's inbox.
        URGENT: added to inbox AND sets urgent flag.
        """
        async with self._lock:
            # Determine which agents to deliver to
            target_agents = update.relevant_agents if update.relevant_agents else []
            if not target_agents:
                # If no specific agents, all agents should see it
                target_agents = list(self._inboxes.keys())

            for agent_id in target_agents:
                if agent_id not in self._inboxes:
                    self._inboxes[agent_id] = AgentInboxState(agent_id=agent_id)

                inbox = self._inboxes[agent_id]
                inbox.entries.append(InboxEntry(update=update))

                # Set urgent flag if applicable
                if update.priority == UpdatePriority.URGENT:
                    self._urgent_flags[agent_id] = True
                    logger.info(
                        "AgentInbox: URGENT update for %s (entry #%d)",
                        agent_id, update.entry.id,
                    )
                elif update.priority == UpdatePriority.IMPORTANT:
                    logger.debug(
                        "AgentInbox: IMPORTANT update for %s (entry #%d)",
                        agent_id, update.entry.id,
                    )

    async def read(
        self,
        agent_id: str,
        min_priority: UpdatePriority = UpdatePriority.ROUTINE,
    ) -> List[PrioritizedUpdate]:
        """
        Read unread updates for an agent at or above min_priority.

        Returns entries in priority order (URGENT > IMPORTANT > ROUTINE),
        then by time received.

        Does NOT mark entries as read — call mark_read() separately.
        """
        async with self._lock:
            inbox = self._inboxes.get(agent_id)
            if not inbox:
                return []

            priority_order = {
                UpdatePriority.URGENT: 0,
                UpdatePriority.IMPORTANT: 1,
                UpdatePriority.ROUTINE: 2,
            }
            min_order = priority_order.get(min_priority, 2)

            unread = [
                ie for ie in inbox.entries
                if not ie.read and priority_order.get(ie.update.priority, 2) <= min_order
            ]

            # Sort by priority then time
            unread.sort(key=lambda ie: (
                priority_order.get(ie.update.priority, 2),
                ie.delivered_at,
            ))

            return [ie.update for ie in unread]

    async def has_urgent(self, agent_id: str) -> bool:
        """Return True if the agent has urgent unread updates."""
        async with self._lock:
            return self._urgent_flags.get(agent_id, False)

    async def clear_urgent_flag(self, agent_id: str) -> None:
        """Clear the urgent flag for an agent after processing."""
        async with self._lock:
            self._urgent_flags.pop(agent_id, None)

    async def mark_read(self, agent_id: str, entry_ids: List[int]) -> None:
        """Mark specific entries as read for an agent."""
        async with self._lock:
            inbox = self._inboxes.get(agent_id)
            if not inbox:
                return

            id_set = set(entry_ids)
            for ie in inbox.entries:
                if ie.update.entry.id in id_set:
                    ie.read = True

            inbox.last_read_at = datetime.now(timezone.utc)

            # Clear urgent flag if all urgent items are read
            urgent_unread = any(
                ie.update.priority == UpdatePriority.URGENT and not ie.read
                for ie in inbox.entries
            )
            if not urgent_unread:
                self._urgent_flags.pop(agent_id, None)

    async def pending_count(self, agent_id: str) -> Dict[str, int]:
        """Return unread entry counts by priority for an agent."""
        async with self._lock:
            inbox = self._inboxes.get(agent_id)
            if not inbox:
                return {"routine": 0, "important": 0, "urgent": 0}
            return inbox.pending_counts()

    async def total_pending(self, agent_id: str) -> int:
        """Return total unread count for an agent."""
        async with self._lock:
            inbox = self._inboxes.get(agent_id)
            if not inbox:
                return 0
            return sum(1 for ie in inbox.entries if not ie.read)

    # ------------------------------------------------------------------
    # Agent Registration
    # ------------------------------------------------------------------

    async def register_agent(self, agent_id: str) -> None:
        """Register an agent to receive updates."""
        async with self._lock:
            if agent_id not in self._inboxes:
                self._inboxes[agent_id] = AgentInboxState(agent_id=agent_id)

    async def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent's inbox."""
        async with self._lock:
            self._inboxes.pop(agent_id, None)
            self._urgent_flags.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def status(self) -> Dict[str, Any]:
        """Return inbox status for all agents."""
        async with self._lock:
            return {
                agent_id: {
                    "pending": inbox.pending_counts(),
                    "urgent_flag": self._urgent_flags.get(agent_id, False),
                    "last_read": inbox.last_read_at.isoformat() if inbox.last_read_at else None,
                }
                for agent_id, inbox in self._inboxes.items()
            }


# ======================================================================
# Checkpoint Lifecycle Manager
# ======================================================================

@dataclass
class CheckpointReadiness:
    """
    Status report on whether the team is ready for a checkpoint cycle.
    """
    all_ready: bool
    """True if all agents have reached their checkpoint."""

    ready_agents: List[str] = field(default_factory=list)
    """Agents that have signaled checkpoint readiness."""

    pending_agents: List[str] = field(default_factory=list)
    """Agents still working toward their checkpoint."""

    blocking_reasons: Dict[str, str] = field(default_factory=dict)
    """agent_id -> reason why they're not ready."""


@dataclass
class CheckpointCycleResult:
    """
    Result of a completed checkpoint cycle.

    Returned by CheckpointLifecycleManager.initiate_checkpoint().
    """
    success: bool
    """True if the cycle completed successfully."""

    synthesis: Optional["SynthesisResult"] = None
    """The narrative synthesis produced by NarrativeSynthesizer."""

    ledger_entry: Optional["LedgerEntry"] = None
    """The ledger entry appended to ProjectLedger."""

    next_assignments: Dict[str, int] = field(default_factory=dict)
    """agent_id -> milestone_index for next checkpoint."""

    narrative_summary: str = ""
    """What agents will read when they 'return to work'."""

    error: Optional[str] = None
    """Error message if success=False."""


class CheckpointLifecycleManager:
    """
    Manages the checkpoint cycle for the agent team.

    Coordinates:
      1. Monitor agent progress toward checkpoints
      2. When all ready: signal "go home", synthesize, append to ledger
      3. Assign next checkpoints from MetaPlan
      4. Signal "return to work" with new narrative

    Parameters
    ----------
    meta_plan : MetaPlan
        The session's plan with milestones.
    team : AgentTeam
        The assembled team.
    convergence_tracker : ConvergenceTracker
        Tracks agent convergence signals.
    narrative_synthesizer : NarrativeSynthesizer
        Produces session syntheses.
    project_ledger : ProjectLedger
        Append-only session history.
    backend_registry : optional
        LLM backend for enhanced checkpoint decisions.
    """

    def __init__(
        self,
        meta_plan: "MetaPlan",
        team: "AgentTeam",
        convergence_tracker: "ConvergenceTracker",
        narrative_synthesizer: "NarrativeSynthesizer",
        project_ledger: "ProjectLedger",
        backend_registry=None,
    ) -> None:
        self._plan = meta_plan
        self._team = team
        self._tracker = convergence_tracker
        self._synthesizer = narrative_synthesizer
        self._ledger = project_ledger
        self._backend = backend_registry
        self._whiteboard_store: Optional["WhiteboardStore"] = None

        # Current checkpoint assignments: agent_id -> milestone_index
        self._current_assignments: Dict[str, int] = {}
        # Explicit "checkpoint reached" signals from agents
        self._checkpoint_signals: Set[str] = set()
        # Agents currently "at home" (waiting for next assignment)
        self._agents_home: Set[str] = set()

        # Evaluator — adversarial criterion-based assessment at each checkpoint
        # Imported lazily to avoid circular imports
        from prsm.compute.nwtn.team.evaluator import CheckpointEvaluator
        self._evaluator = CheckpointEvaluator(
            meta_plan=meta_plan,
            backend_registry=backend_registry,
        )

        # Agent inbox reference (injected by LiveScribe after construction)
        self._inbox: Optional["AgentInbox"] = None

    def set_whiteboard_store(self, store: "WhiteboardStore") -> None:
        """Inject the whiteboard store (called by LiveScribe during init)."""
        self._whiteboard_store = store

    def set_inbox(self, inbox: "AgentInbox") -> None:
        """Inject the AgentInbox so evaluator results can notify failing agents."""
        self._inbox = inbox

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_checkpoint(self, agent_id: str, milestone_index: int) -> None:
        """Assign a specific milestone as an agent's checkpoint target."""
        if milestone_index < 0 or milestone_index >= len(self._plan.milestones):
            raise ValueError(f"Invalid milestone index: {milestone_index}")
        self._current_assignments[agent_id] = milestone_index
        self._agents_home.discard(agent_id)
        self._checkpoint_signals.discard(agent_id)
        logger.info(
            "CheckpointLifecycleManager: assigned %s to milestone %d",
            agent_id, milestone_index,
        )

    def signal_checkpoint_reached(self, agent_id: str) -> None:
        """Called when an agent explicitly signals it has reached its checkpoint."""
        self._checkpoint_signals.add(agent_id)
        logger.info(
            "CheckpointLifecycleManager: %s signaled checkpoint reached",
            agent_id,
        )

    async def check_checkpoint_readiness(self) -> CheckpointReadiness:
        """
        Check if all agents are ready for a checkpoint cycle.

        An agent is ready if:
          - They have signaled "checkpoint reached", OR
          - The ConvergenceTracker marks them as converged, AND
          - They are not currently "at home"
        """
        ready_agents: List[str] = []
        pending_agents: List[str] = []
        blocking_reasons: Dict[str, str] = {}

        for member in self._team.members:
            agent_id = member.agent_id

            # Skip agents at home
            if agent_id in self._agents_home:
                continue

            # Check explicit signal
            if agent_id in self._checkpoint_signals:
                ready_agents.append(agent_id)
                continue

            # Check convergence
            if self._tracker.is_agent_converged(agent_id):
                ready_agents.append(agent_id)
                continue

            # Not ready
            pending_agents.append(agent_id)
            milestone_idx = self._current_assignments.get(agent_id, 0)
            if milestone_idx < len(self._plan.milestones):
                ms = self._plan.milestones[milestone_idx]
                blocking_reasons[agent_id] = (
                    f"Working toward milestone {milestone_idx}: {ms.title}"
                )
            else:
                blocking_reasons[agent_id] = "No checkpoint assigned"

        all_ready = len(pending_agents) == 0 and len(ready_agents) > 0

        return CheckpointReadiness(
            all_ready=all_ready,
            ready_agents=ready_agents,
            pending_agents=pending_agents,
            blocking_reasons=blocking_reasons,
        )

    async def initiate_checkpoint(self) -> CheckpointCycleResult:
        """
        Execute the full checkpoint cycle.

        Steps:
          1. Verify all agents are ready
          2. Signal agents to "go home"
          3. Run NarrativeSynthesizer
          4. Append to ProjectLedger
          5. Assign next milestones
          6. Signal agents to "return to work"

        Returns
        -------
        CheckpointCycleResult
        """
        readiness = await self.check_checkpoint_readiness()
        if not readiness.all_ready:
            return CheckpointCycleResult(
                success=False,
                error=f"Not all agents ready: {readiness.pending_agents}",
            )

        try:
            # 1. Signal "go home"
            self._agents_home.update(readiness.ready_agents)
            logger.info(
                "CheckpointLifecycleManager: signaling %d agents to go home",
                len(readiness.ready_agents),
            )

            # 2. Get whiteboard snapshot
            # Note: whiteboard access is handled by LiveScribe
            snapshot = await self._get_snapshot()
            if snapshot is None:
                return CheckpointCycleResult(
                    success=False,
                    error="Could not retrieve whiteboard snapshot",
                )

            # 3. Adversarial evaluation — assess each agent's work before synthesis.
            #    The evaluator is skeptical by default: criteria are NOT met until
            #    proven otherwise.  Failing agents get an IMPORTANT inbox flag.
            evaluation_batch = await self._run_checkpoint_evaluation(
                snapshot=snapshot,
                ready_agents=readiness.ready_agents,
            )

            # 4. Run synthesis (evaluation results are embedded in the narrative)
            previous_summary = self._ledger.to_onboarding_context(max_entries=1, max_chars=1000)
            synthesis = await self._synthesizer.synthesise(
                snapshot=snapshot,
                meta_plan=self._plan,
                previous_summary=previous_summary,
            )

            # Inject evaluation block into the synthesis narrative
            if evaluation_batch is not None:
                synthesis = self._embed_evaluation_in_synthesis(synthesis, evaluation_batch)

            # 5. Append to ledger (signing is handled internally)
            from prsm.compute.nwtn.synthesis.signer import LedgerSigner
            signer = LedgerSigner()  # Generates new key if needed
            ledger_entry = self._ledger.append(synthesis, signer)

            # 6. Assign next milestones
            next_assignments = await self.assign_next_checkpoints()

            # 7. Prepare narrative summary for agents
            narrative_summary = self._build_narrative_summary(synthesis, next_assignments)

            # 8. Notify failing agents via inbox so they know to address gaps
            if evaluation_batch is not None:
                await self._notify_failing_agents(evaluation_batch)

            # Clear signals and send agents back to work
            self._checkpoint_signals.clear()
            self._agents_home.clear()

            logger.info(
                "CheckpointLifecycleManager: checkpoint cycle complete, "
                "%d next assignments made (eval passed: %s/%s)",
                len(next_assignments),
                len(evaluation_batch.passed_agents) if evaluation_batch else "n/a",
                len(readiness.ready_agents),
            )

            return CheckpointCycleResult(
                success=True,
                synthesis=synthesis,
                ledger_entry=ledger_entry,
                next_assignments=next_assignments,
                narrative_summary=narrative_summary,
            )

        except Exception as exc:
            logger.error("Checkpoint cycle failed: %s", exc)
            return CheckpointCycleResult(
                success=False,
                error=str(exc),
            )

    async def assign_next_checkpoints(self) -> Dict[str, int]:
        """
        Assign the next milestone checkpoint to each agent.

        Uses a round-robin assignment based on current progress through
        the milestone list.

        Returns
        -------
        Dict[str, int]
            agent_id -> milestone_index
        """
        assignments: Dict[str, int] = {}

        # Find the highest milestone index currently assigned
        max_current = max(self._current_assignments.values()) if self._current_assignments else -1

        # Next milestone (or wrap to 0 if all complete)
        next_idx = (max_current + 1) % max(1, len(self._plan.milestones))

        for member in self._team.members:
            # Assign next milestone to each agent
            # Could be smarter about role-based assignment here
            assignments[member.agent_id] = next_idx
            self._current_assignments[member.agent_id] = next_idx

        return assignments

    def get_current_checkpoint(self, agent_id: str) -> Optional["Milestone"]:
        """Get the current checkpoint target for an agent."""
        idx = self._current_assignments.get(agent_id)
        if idx is not None and idx < len(self._plan.milestones):
            return self._plan.milestones[idx]
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_snapshot(self):
        """Get the current whiteboard snapshot from the whiteboard store."""
        if self._whiteboard_store is None:
            logger.error("CheckpointLifecycleManager: no whiteboard store — call set_whiteboard_store()")
            return None
        return await self._whiteboard_store.snapshot(self._plan.session_id)

    async def _run_checkpoint_evaluation(
        self,
        snapshot,
        ready_agents: List[str],
    ) -> Optional["EvaluationBatch"]:
        """
        Run adversarial evaluation on each agent's checkpoint work.

        Groups whiteboard entries by source agent and evaluates against
        the agent's current milestone criteria.

        Parameters
        ----------
        snapshot : WhiteboardSnapshot
            Current whiteboard snapshot.
        ready_agents : List[str]
            Agent IDs that reached the checkpoint.

        Returns
        -------
        EvaluationBatch or None if evaluation fails or no entries exist.
        """
        if not ready_agents:
            return None

        try:
            # Group entries by source agent
            agent_entries: Dict[str, List] = {agent_id: [] for agent_id in ready_agents}

            for entry in snapshot.entries:
                if entry.source_agent in agent_entries:
                    agent_entries[entry.source_agent].append(entry)

            # Determine which milestone to evaluate against (use most common assignment)
            milestone_idx = 0
            if self._current_assignments:
                milestone_idx = max(
                    self._current_assignments.values(),
                    key=lambda idx: list(self._current_assignments.values()).count(idx),
                )
            milestone_idx = min(milestone_idx, len(self._plan.milestones) - 1)
            milestone = self._plan.milestones[milestone_idx]

            logger.info(
                "CheckpointLifecycleManager: running adversarial evaluation "
                "for %d agents at milestone %d (%s)",
                len(ready_agents), milestone_idx, milestone.title,
            )

            batch = await self._evaluator.evaluate_team(
                agent_entries=agent_entries,
                milestone=milestone,
            )

            # Log summary for operators
            for result in batch.results:
                level = logging.WARNING if not result.passed else logging.INFO
                logger.log(level, "  Evaluator: %s", result.summary())

            return batch

        except Exception as exc:
            logger.warning(
                "CheckpointLifecycleManager: evaluation failed (%s) — "
                "proceeding with synthesis without evaluation",
                exc,
            )
            return None

    def _embed_evaluation_in_synthesis(
        self,
        synthesis: "SynthesisResult",
        evaluation_batch: "EvaluationBatch",
    ) -> "SynthesisResult":
        """
        Embed the evaluation block into the synthesis narrative.

        Returns a new SynthesisResult with the evaluation summary appended.
        """
        try:
            eval_block = evaluation_batch.to_narrative_block()
            updated_narrative = synthesis.narrative + "\n\n" + eval_block

            # Return a copy with updated narrative (SynthesisResult is a dataclass)
            import dataclasses
            return dataclasses.replace(synthesis, narrative=updated_narrative)
        except Exception as exc:
            logger.warning(
                "CheckpointLifecycleManager: could not embed evaluation in narrative (%s)", exc
            )
            return synthesis

    async def _notify_failing_agents(self, evaluation_batch: "EvaluationBatch") -> None:
        """
        Send IMPORTANT inbox notifications to agents whose work failed evaluation.

        Creates a synthetic PrioritizedUpdate for each failing agent so they
        know to address the gaps before the next checkpoint.
        """
        if self._inbox is None:
            logger.debug(
                "CheckpointLifecycleManager: no inbox reference — cannot notify failing agents"
            )
            return

        for result in evaluation_batch.results:
            if result.passed:
                continue

            # Build a human-readable message about what needs fixing
            issues_text = "\n".join(f"  • {issue}" for issue in result.issues_found[:5])
            message = (
                f"⚠️ Checkpoint evaluation: your work did NOT fully meet milestone criteria "
                f"(score={result.quality_score:.2f}).\n\n"
                f"Issues identified:\n{issues_text}\n\n"
                f"Please address these before the next checkpoint.\n"
                f"(Evaluator confidence: {result.confidence:.2f})"
            )

            # We create a synthetic "evaluator" whiteboard entry for the notification
            try:
                from prsm.compute.nwtn.whiteboard.schema import WhiteboardEntry
                fake_entry = WhiteboardEntry(
                    id=-1,
                    session_id=self._plan.session_id,
                    source_agent="nwtn/evaluator",
                    chunk=message,
                    surprise_score=0.8,
                    promoted_at=datetime.now(timezone.utc),
                )
                update = PrioritizedUpdate(
                    entry=fake_entry,
                    priority=UpdatePriority.IMPORTANT,
                    relevant_agents=[result.agent_id],
                    reason="checkpoint evaluation failed — criteria not met",
                )
                await self._inbox.push(update, target_agents=[result.agent_id])
                logger.info(
                    "CheckpointLifecycleManager: notified %s of evaluation failure",
                    result.agent_id,
                )
            except Exception as exc:
                logger.warning(
                    "CheckpointLifecycleManager: could not send inbox notification to %s: %s",
                    result.agent_id, exc,
                )

    def _build_narrative_summary(
        self,
        synthesis: "SynthesisResult",
        next_assignments: Dict[str, int],
    ) -> str:
        """Build the summary agents read when returning to work."""
        lines = [
            "## Checkpoint Complete — Returning to Work\n",
            f"**Session:** {synthesis.session_id}",
            f"**Synthesis:** {synthesis.narrative[:500]}{'...' if len(synthesis.narrative) > 500 else ''}\n",
            "**Next Checkpoints:**",
        ]

        for agent_id, ms_idx in next_assignments.items():
            if ms_idx < len(self._plan.milestones):
                ms = self._plan.milestones[ms_idx]
                agent_short = agent_id.removeprefix("agent/")
                lines.append(f"  - {agent_short}: Milestone {ms_idx + 1} — {ms.title}")

        return "\n".join(lines)


# ======================================================================
# Priority Tagger
# ======================================================================

class PriorityTagger:
    """
    Assigns priority levels to whiteboard updates.

    Priority is determined by:
      - Surprise score (very high > 0.85 → at least IMPORTANT)
      - Semantic conflict detection
      - Milestone keyword matching
      - Cross-agent relevance scoring

    Parameters
    ----------
    meta_plan : MetaPlan
        The session's plan for milestone keyword matching.
    conflict_detector : ConflictDetector
        Detector for semantic conflicts.
    backend_registry : optional
        LLM backend for enhanced priority decisions.
    """

    # Thresholds for score-based priority
    SURPRISE_IMPORTANT_THRESHOLD = 0.85
    SURPRISE_URGENT_THRESHOLD = 0.95

    def __init__(
        self,
        meta_plan: "MetaPlan",
        conflict_detector: ConflictDetector,
        backend_registry=None,
    ) -> None:
        self._plan = meta_plan
        self._detector = conflict_detector
        self._backend = backend_registry

    async def tag(
        self,
        entry: "WhiteboardEntry",
        existing_entries: List["WhiteboardEntry"],
        team_members: List["TeamMember"],
    ) -> PrioritizedUpdate:
        """
        Assign priority and metadata to a whiteboard entry.

        Parameters
        ----------
        entry : WhiteboardEntry
            The newly promoted entry.
        existing_entries : List[WhiteboardEntry]
            Existing entries for conflict detection.
        team_members : List[TeamMember]
            Team members for cross-agent relevance.

        Returns
        -------
        PrioritizedUpdate
        """
        # Start with ROUTINE
        priority = UpdatePriority.ROUTINE
        reasons: List[str] = []
        milestone_refs: List[int] = []
        relevant_agents: List[str] = []

        # 1. Check surprise score
        if entry.surprise_score >= self.SURPRISE_URGENT_THRESHOLD:
            priority = UpdatePriority.URGENT
            reasons.append(f"very high surprise ({entry.surprise_score:.3f})")
        elif entry.surprise_score >= self.SURPRISE_IMPORTANT_THRESHOLD:
            priority = UpdatePriority.IMPORTANT
            reasons.append(f"high surprise ({entry.surprise_score:.3f})")

        # 2. Check milestone references
        ms_refs = self._find_milestone_references(entry.chunk)
        if ms_refs:
            milestone_refs = ms_refs
            reasons.append(f"references milestone(s) {ms_refs}")
            if priority == UpdatePriority.ROUTINE:
                priority = UpdatePriority.IMPORTANT

        # 3. Check for conflicts
        conflict_report = await self._detector.check_conflicts(entry, existing_entries)
        conflict_detected = conflict_report is not None
        conflict_details = conflict_report.description if conflict_report else None

        if conflict_detected:
            priority = UpdatePriority.URGENT
            reasons.append("conflict detected with existing whiteboard entries")
            relevant_agents = conflict_report.affected_agents if conflict_report else []

        # 4. Cross-agent relevance
        cross_agent_refs = self._find_cross_agent_relevance(entry, team_members)
        if cross_agent_refs:
            # Add to relevant agents (avoiding duplicates)
            for agent_id in cross_agent_refs:
                if agent_id not in relevant_agents and agent_id != entry.source_agent:
                    relevant_agents.append(agent_id)

        # Build reason string
        reason = "; ".join(reasons) if reasons else "normal progress update"

        return PrioritizedUpdate(
            entry=entry,
            priority=priority,
            relevant_agents=relevant_agents,
            conflict_detected=conflict_detected,
            conflict_details=conflict_details,
            milestone_refs=milestone_refs,
            reason=reason,
        )

    def _find_milestone_references(self, text: str) -> List[int]:
        """
        Find which milestone keywords appear in the text.

        Returns list of milestone indices.
        """
        refs: List[int] = []
        text_lower = text.lower()

        for idx, milestone in enumerate(self._plan.milestones):
            # Extract keywords from milestone title and description
            keywords = re.findall(r'\b\w{4,}\b', (milestone.title + " " + milestone.description).lower())
            # Check if multiple keywords appear
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:  # Require at least 2 keyword matches
                refs.append(idx)

        return refs

    def _find_cross_agent_relevance(
        self,
        entry: "WhiteboardEntry",
        team_members: List["TeamMember"],
    ) -> List[str]:
        """
        Find agents whose work might be relevant to this entry.

        Uses role capability matching and keyword overlap.
        """
        relevant: List[str] = []
        text_lower = entry.chunk.lower()

        for member in team_members:
            # Skip the source agent
            if member.agent_id == entry.source_agent:
                continue

            # Check if entry mentions keywords related to the member's role
            role_keywords = set(member.role.replace("-", " ").split())
            role_keywords.update(member.capabilities)

            matches = sum(1 for kw in role_keywords if kw in text_lower)
            if matches >= 2:
                relevant.append(member.agent_id)

        return relevant


# ======================================================================
# Live Scribe (Main Orchestrator)
# ======================================================================

class LiveScribe:
    """
    Real-time scribe agent for NWTN Agent Teams.

    Runs during working sessions. Curates the whiteboard, manages
    cross-agent information flow, detects conflicts, and coordinates
    the checkpoint lifecycle.

    The Live Scribe is the central coordinator that:
      - Receives promoted chunks from the BSC
      - Tags them with priority
      - Distributes to agent inboxes
      - Detects conflicts
      - Manages checkpoint cycles

    Parameters
    ----------
    whiteboard_store : WhiteboardStore
        Persistent whiteboard storage.
    meta_plan : MetaPlan
        The session's plan.
    team : AgentTeam
        The assembled agent team.
    convergence_tracker : ConvergenceTracker
        Tracks agent convergence.
    narrative_synthesizer : NarrativeSynthesizer
        Produces session syntheses.
    project_ledger : ProjectLedger
        Append-only session history.
    backend_registry : optional
        LLM backend for enhanced decisions. Falls back to heuristics.
    """

    def __init__(
        self,
        whiteboard_store: "WhiteboardStore",
        meta_plan: "MetaPlan",
        team: "AgentTeam",
        convergence_tracker: "ConvergenceTracker",
        narrative_synthesizer: "NarrativeSynthesizer",
        project_ledger: "ProjectLedger",
        backend_registry=None,
    ) -> None:
        self._store = whiteboard_store
        self._plan = meta_plan
        self._team = team
        self._tracker = convergence_tracker
        self._synthesizer = narrative_synthesizer
        self._ledger = project_ledger
        self._backend = backend_registry

        # Initialize components
        self._conflict_detector = ConflictDetector(backend_registry=backend_registry)
        self._priority_tagger = PriorityTagger(
            meta_plan=meta_plan,
            conflict_detector=self._conflict_detector,
            backend_registry=backend_registry,
        )
        self._inbox = AgentInbox(session_id=meta_plan.session_id)
        self._conflict_log = ConflictLog(session_id=meta_plan.session_id)

        # Checkpoint lifecycle manager
        self._checkpoint_mgr = CheckpointLifecycleManager(
            meta_plan=meta_plan,
            team=team,
            convergence_tracker=convergence_tracker,
            narrative_synthesizer=narrative_synthesizer,
            project_ledger=project_ledger,
            backend_registry=backend_registry,
        )
        self._checkpoint_mgr.set_whiteboard_store(whiteboard_store)
        # Wire inbox so the evaluator can send failure notifications to agents
        self._checkpoint_mgr.set_inbox(self._inbox)

        # Register team members with convergence tracker (inbox registration
        # is deferred to setup() to avoid requiring a running event loop at
        # construction time).
        self._members_registered = False
        for member in team.members:
            self._tracker.register_agent(member.agent_id)

        # Initial checkpoint assignments
        self._assign_initial_checkpoints()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """
        Async initialisation — call once before using the Live Scribe.

        Registers all team members with the agent inbox.  Separated from
        ``__init__`` because inbox registration is async and we do not want
        to require a running event loop at construction time.
        """
        if not self._members_registered:
            for member in self._team.members:
                await self._inbox.register_agent(member.agent_id)
            self._members_registered = True
            logger.info(
                "LiveScribe: setup complete — %d agents registered",
                len(self._team.members),
            )

    async def on_chunk_promoted(self, decision: "PromotionDecision") -> PrioritizedUpdate:
        """
        Called by BSCPromoter after a chunk is promoted.

        This is the main entry point for the Live Scribe. It:
          1. Writes the entry to the whiteboard
          2. Tags with priority
          3. Checks for conflicts
          4. Distributes to agent inboxes

        Parameters
        ----------
        decision : PromotionDecision
            The BSC's decision to promote this chunk.

        Returns
        -------
        PrioritizedUpdate
            The prioritized update ready for distribution.
        """
        logger.info(
            "LiveScribe: processing promoted chunk from %s (score=%.3f)",
            decision.metadata.source_agent, decision.surprise_score,
        )

        # 1. Write to whiteboard
        entry = await self._store.write(decision)

        # 2. Get existing entries for conflict detection
        existing = await self._store.get_all(decision.metadata.session_id)

        # 3. Tag with priority
        update = await self._priority_tagger.tag(
            entry=entry,
            existing_entries=existing[:-1],  # Exclude the new entry
            team_members=self._team.members,
        )

        # 4. Log conflicts
        if update.conflict_detected:
            self._conflict_log.add(ConflictReport(
                new_entry_id=entry.id,
                conflicting_entry_ids=[],  # Populated by detector
                conflict_type="contradiction",
                description=update.conflict_details or "Conflict detected",
                affected_agents=update.relevant_agents,
            ))
            logger.warning(
                "LiveScribe: CONFLICT detected in entry #%d from %s",
                entry.id, entry.source_agent,
            )

        # 5. Distribute to inboxes
        await self._inbox.push(update)

        return update

    async def get_agent_context(self, agent_id: str) -> str:
        """
        Build the context string an agent should read before starting work.

        Includes:
          - Compressed whiteboard state
          - Unread inbox items
          - Current checkpoint target

        Parameters
        ----------
        agent_id : str
            The agent requesting context.

        Returns
        -------
        str
            Formatted context string.
        """
        lines: List[str] = []

        # 1. Whiteboard state
        whiteboard_state = await self._store.compressed_state(
            self._plan.session_id,
            max_chars=2000,
        )
        lines.append("## Current Whiteboard State\n")
        lines.append(whiteboard_state)
        lines.append("")

        # 2. Unread inbox items
        pending = await self._inbox.pending_count(agent_id)
        total_pending = sum(pending.values())

        if total_pending > 0:
            lines.append("## Your Inbox\n")
            urgent_count = pending.get("urgent", 0)
            important_count = pending.get("important", 0)

            if urgent_count > 0:
                lines.append(f"**⚠️ URGENT: {urgent_count} item(s) require immediate attention**\n")

            if important_count > 0:
                lines.append(f"**📌 IMPORTANT: {important_count} item(s) flagged**\n")

            # Get actual items
            updates = await self._inbox.read(agent_id, min_priority=UpdatePriority.IMPORTANT)
            if updates:
                lines.append("Important/urgent items:")
                for u in updates[:5]:
                    priority_marker = "🔴" if u.priority == UpdatePriority.URGENT else "🟡"
                    lines.append(f"  {priority_marker} [{u.entry.agent_short}] {u.entry.chunk[:100]}...")

            lines.append("")

        # 3. Checkpoint target
        checkpoint = self._checkpoint_mgr.get_current_checkpoint(agent_id)
        if checkpoint:
            lines.append("## Your Checkpoint Target\n")
            lines.append(f"**Milestone:** {checkpoint.title}")
            lines.append(f"**Description:** {checkpoint.description}")
            if checkpoint.merge_criteria:
                lines.append("**Merge criteria:**")
                for c in checkpoint.merge_criteria:
                    lines.append(f"  - {c}")
            lines.append("")

        # 4. Urgent flag
        if await self._inbox.has_urgent(agent_id):
            lines.insert(0, "\n⚠️ **URGENT UPDATES AWAITING YOUR ATTENTION** ⚠️\n")

        return "\n".join(lines)

    async def check_and_run_checkpoint(self) -> Optional[CheckpointCycleResult]:
        """
        Check if checkpoint conditions are met; if so, run the full cycle.

        Returns
        -------
        CheckpointCycleResult | None
            Result if a checkpoint cycle was run, None if not ready.
        """
        readiness = await self._checkpoint_mgr.check_checkpoint_readiness()

        if readiness.all_ready:
            logger.info(
                "LiveScribe: all agents ready for checkpoint (%s)",
                readiness.ready_agents,
            )
            return await self._checkpoint_mgr.initiate_checkpoint()

        return None

    async def signal_checkpoint_reached(self, agent_id: str) -> None:
        """Signal that an agent has reached its checkpoint."""
        self._checkpoint_mgr.signal_checkpoint_reached(agent_id)

    async def status(self) -> Dict[str, Any]:
        """
        Return live scribe status: inbox counts, conflicts, checkpoint progress.

        Returns
        -------
        dict
            Comprehensive status report.
        """
        inbox_status = await self._inbox.status()
        readiness = await self._checkpoint_mgr.check_checkpoint_readiness()

        return {
            "session_id": self._plan.session_id,
            "inbox": inbox_status,
            "conflicts": {
                "total": self._conflict_log.count,
                "recent": [
                    {
                        "new_entry_id": c.new_entry_id,
                        "conflict_type": c.conflict_type,
                        "affected_agents": c.affected_agents,
                    }
                    for c in self._conflict_log.conflicts[-5:]
                ],
            },
            "checkpoint": {
                "all_ready": readiness.all_ready,
                "ready_agents": readiness.ready_agents,
                "pending_agents": readiness.pending_agents,
                "blocking_reasons": readiness.blocking_reasons,
            },
            "convergence": self._tracker.convergence_summary(),
        }

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    async def get_urgent_updates(self, agent_id: str) -> List[PrioritizedUpdate]:
        """Get all urgent updates for an agent."""
        return await self._inbox.read(agent_id, min_priority=UpdatePriority.URGENT)

    async def mark_updates_read(self, agent_id: str, entry_ids: List[int]) -> None:
        """Mark updates as read for an agent."""
        await self._inbox.mark_read(agent_id, entry_ids)

    async def has_urgent(self, agent_id: str) -> bool:
        """Check if an agent has urgent updates."""
        return await self._inbox.has_urgent(agent_id)

    # ------------------------------------------------------------------
    # Evaluator tuning interface (P3 loop)
    # ------------------------------------------------------------------

    def review_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Return the full evaluation history with divergence notes.

        Delegates to ``CheckpointEvaluator.review_evaluation_history()``.
        Use this in the P3 tuning loop to identify systematic errors and
        criterion prompts that need refinement.

        Returns
        -------
        List[Dict]
            Evaluation log entries with agent_id, scores, issues, divergence_notes.
        """
        return self._checkpoint_mgr._evaluator.review_evaluation_history()

    def update_evaluation_criterion_prompt(
        self,
        agent_id: str,
        criterion: str,
        new_prompt: str,
    ) -> None:
        """
        Override an evaluator's prompt for a specific (agent, criterion) pair.

        Allows human operators to tune the evaluator when divergence notes
        reveal systematic judgment errors.

        Parameters
        ----------
        agent_id : str
            Target agent ID, or ``"*"`` to apply globally.
        criterion : str
            Exact criterion text as it appears in the MetaPlan milestone.
        new_prompt : str
            Replacement prompt fragment for the LLM's assessment of this criterion.
        """
        self._checkpoint_mgr._evaluator.update_criteria_prompt(agent_id, criterion, new_prompt)

    @property
    def evaluator(self):
        """
        Direct access to the underlying CheckpointEvaluator.

        Provided for advanced use-cases like pre-evaluation, custom hooks,
        or integration tests.
        """
        return self._checkpoint_mgr._evaluator

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assign_initial_checkpoints(self) -> None:
        """Assign initial checkpoint targets to all team members."""
        for i, member in enumerate(self._team.members):
            # Distribute milestones round-robin
            milestone_idx = i % len(self._plan.milestones) if self._plan.milestones else 0
            self._checkpoint_mgr.assign_checkpoint(member.agent_id, milestone_idx)


# ======================================================================
# Exports
# ======================================================================

from prsm.compute.nwtn.team.evaluator import (  # noqa: E402 — import after dataclasses defined above
    CheckpointEvaluator,
    EvaluationResult,
    EvaluationBatch,
)

__all__ = [
    # Priority system
    "UpdatePriority",
    "PrioritizedUpdate",
    "PriorityTagger",
    # Conflict detection
    "ConflictDetector",
    "ConflictReport",
    "ConflictLog",
    # Agent notification
    "AgentInbox",
    "InboxEntry",
    "AgentInboxState",
    # Checkpoint lifecycle
    "CheckpointLifecycleManager",
    "CheckpointReadiness",
    "CheckpointCycleResult",
    # Main orchestrator
    "LiveScribe",
    # Evaluator (re-exported for convenience)
    "CheckpointEvaluator",
    "EvaluationResult",
    "EvaluationBatch",
]
