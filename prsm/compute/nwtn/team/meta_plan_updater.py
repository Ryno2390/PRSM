"""
Meta-Plan Updater
=================

Dynamic update capabilities for NWTN Agent Team MetaPlans.

Currently, the MetaPlan is generated once at session start and never updated.
This means milestone estimates go stale as actual work progresses and diverges
from initial predictions. This module adds the ability to dynamically update
the MetaPlan based on actual work completed, checkpoint results, and session
progress.

Position in the checkpoint cycle
---------------------------------

The MetaPlanUpdater integrates with the existing checkpoint lifecycle:

    CheckpointLifecycleManager.initiate_checkpoint()
              ↓
    For each CheckpointDecision → updater.record_checkpoint_result()
              ↓
    Periodically call updater.suggest_replan()
              ↓
    If confidence > threshold → apply_replan()
              ↓
    CheckpointLifecycleManager.assign_next_checkpoints() uses updated plan

The MetaPlanUpdater does NOT modify the MetaPlan's milestones list directly
through external references. It mutates self._plan.milestones in-place,
which is the same object held by CheckpointLifecycleManager and
CheckpointReviewer (passed by reference at construction time).

Key capabilities
----------------
1. **Milestone Progress Tracking**: Runtime tracking data for each milestone
   including status, assigned agents, effort spent, and criteria satisfaction.

2. **Plan Mutation**: Add, defer, cancel milestones; update effort estimates;
   reorder priorities.

3. **Smart Replanning**: LLM-powered suggestions with heuristic fallback for
   effort drift detection and blocking analysis.

Quick start
-----------
.. code-block:: python

    from prsm.compute.nwtn.team import MetaPlanUpdater, MetaPlan

    updater = MetaPlanUpdater(meta_plan=meta_plan, backend_registry=backend)

    # Record checkpoint decisions as they occur
    progress = await updater.record_checkpoint_result(decision)

    # Periodically check if replan is needed
    suggestion = await updater.suggest_replan(whiteboard_snapshot)
    if suggestion.confidence > 0.8:
        result = await updater.apply_replan(suggestion)

    # Get current progress
    all_progress = updater.get_all_progress()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from prsm.compute.nwtn.team.planner import MetaPlan, Milestone
    from prsm.compute.nwtn.team.checkpoint import CheckpointDecision

logger = logging.getLogger(__name__)


# ======================================================================
# Milestone Status Tracking
# ======================================================================

class MilestoneStatus(str, Enum):
    """
    Runtime status of a milestone during session execution.

    Milestones transition through states as agents work on them:

        PENDING → IN_PROGRESS → COMPLETED
                    ↓
                BLOCKED → IN_PROGRESS (after unblock)
                    ↓
                DEFERRED (optional)
                    ↓
                CANCELLED (optional)
    """
    PENDING = "pending"
    """Not yet started — no agents assigned or work begun."""

    IN_PROGRESS = "in_progress"
    """At least one agent actively working on this milestone."""

    BLOCKED = "blocked"
    """Explicitly blocked by a dependency, conflict, or external factor."""

    COMPLETED = "completed"
    """All merge criteria met; milestone is finished."""

    DEFERRED = "deferred"
    """Pushed to a later checkpoint cycle; deprioritized but not cancelled."""

    CANCELLED = "cancelled"
    """No longer relevant; removed from active consideration."""


@dataclass
class MilestoneProgress:
    """
    Runtime tracking data for a single milestone.

    Tracks the lifecycle of a milestone from assignment through completion,
    including agent assignments, checkpoint decisions, and criteria
    satisfaction status.
    """
    milestone_index: int
    """Index into MetaPlan.milestones list."""

    status: MilestoneStatus = MilestoneStatus.PENDING
    """Current runtime status of this milestone."""

    assigned_agents: List[str] = field(default_factory=list)
    """Agent IDs currently or previously working on this milestone."""

    started_at: Optional[datetime] = None
    """Timestamp when first agent began work on this milestone."""

    completed_at: Optional[datetime] = None
    """Timestamp when status transitioned to COMPLETED."""

    estimated_effort: str = "medium"
    """Original effort estimate from Milestone ('small', 'medium', 'large')."""

    actual_rounds_spent: int = 0
    """Number of checkpoint cycles this milestone has been active."""

    checkpoint_decisions: List[Any] = field(default_factory=list)
    """CheckpointDecisions that referenced this milestone."""

    merge_criteria_met: List[bool] = field(default_factory=list)
    """Tracks which merge_criteria are satisfied (parallel to Milestone.merge_criteria)."""

    blocking_reason: Optional[str] = None
    """Human-readable reason for BLOCKED status, if applicable."""

    notes: List[str] = field(default_factory=list)
    """Accumulated observations about this milestone's progress."""

    @property
    def is_terminal(self) -> bool:
        """Return True if this milestone is in a terminal state."""
        return self.status in (
            MilestoneStatus.COMPLETED,
            MilestoneStatus.CANCELLED,
        )

    @property
    def is_active(self) -> bool:
        """Return True if this milestone is actively being worked on."""
        return self.status == MilestoneStatus.IN_PROGRESS

    @property
    def criteria_satisfied_count(self) -> int:
        """Return the number of satisfied merge criteria."""
        return sum(1 for met in self.merge_criteria_met if met)

    @property
    def criteria_total_count(self) -> int:
        """Return the total number of merge criteria."""
        return len(self.merge_criteria_met)

    @property
    def completion_ratio(self) -> float:
        """Return the ratio of satisfied to total merge criteria (0-1)."""
        if not self.merge_criteria_met:
            return 0.0
        return self.criteria_satisfied_count / self.criteria_total_count


# ======================================================================
# Data Models for Replanning
# ======================================================================

@dataclass
class MilestoneAdjustment:
    """
    A single proposed change to a milestone.

    Part of a ReplanSuggestion, these adjustments describe specific
    modifications to make to the MetaPlan.
    """
    milestone_index: int
    """Index of the milestone to modify."""

    action: str
    """Type of adjustment: 'update_effort', 'defer', 'cancel', 'reorder',
    'add_criteria', 'remove_criteria', 'unblock', 'mark_blocked'."""

    details: str = ""
    """Specific details of the change (e.g., new effort value, reason)."""

    reason: str = ""
    """Human-readable explanation for why this adjustment is proposed."""


@dataclass
class ReplanSuggestion:
    """
    A proposed set of changes to the MetaPlan.

    Generated by MetaPlanUpdater.suggest_replan(), this represents a
    thoughtful analysis of current progress and recommended adjustments.
    The caller decides whether to apply the suggestion.
    """
    session_id: str
    """Session identifier from the MetaPlan."""

    adjustments: List[MilestoneAdjustment] = field(default_factory=list)
    """Specific milestone modifications proposed."""

    new_milestones: List["Milestone"] = field(default_factory=list)
    """New milestones to add to the MetaPlan."""

    suggested_order: Optional[List[int]] = None
    """New milestone ordering (indices), if reorder needed."""

    rationale: str = ""
    """Overall explanation of why these changes are recommended."""

    confidence: float = 0.0
    """Confidence level of the suggestion (0-1). Higher = more certain."""

    llm_assisted: bool = False
    """True if an LLM contributed to this suggestion; False if heuristic only."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this suggestion was generated."""

    @property
    def has_adjustments(self) -> bool:
        """Return True if there are any adjustments or new milestones."""
        return bool(self.adjustments) or bool(self.new_milestones)


@dataclass
class ReplanResult:
    """
    Result of applying a ReplanSuggestion to the MetaPlan.

    Returned by MetaPlanUpdater.apply_replan(), summarizing what changes
    were actually made.
    """
    success: bool
    """True if all changes were applied successfully."""

    adjustments_applied: int = 0
    """Number of milestone adjustments that were applied."""

    milestones_added: int = 0
    """Number of new milestones added."""

    milestones_deferred: int = 0
    """Number of milestones moved to DEFERRED status."""

    milestones_cancelled: int = 0
    """Number of milestones moved to CANCELLED status."""

    new_milestone_count: int = 0
    """Total number of milestones after replan."""

    summary: str = ""
    """Human-readable summary of the replan outcome."""

    errors: List[str] = field(default_factory=list)
    """Any errors encountered during replan application."""


# ======================================================================
# Effort Drift Detection
# ======================================================================

class EffortDriftDetector:
    """
    Detects when milestones are taking significantly longer or shorter
    than their estimated effort.

    Uses a simple mapping from effort levels to expected checkpoint cycles:

        small  → expected 1-2 cycles
        medium → expected 2-4 cycles
        large  → expected 4-7 cycles

    If actual_rounds_spent exceeds the upper bound, flags as "drifting slow".
    If completed before the lower bound, flags as "drifting fast".

    This provides the heuristic fallback for suggest_replan() when no
    LLM backend is available.
    """

    # Effort level → (min_expected, max_expected) checkpoint cycles
    EFFORT_BOUNDS: Dict[str, tuple[int, int]] = {
        "small": (1, 2),
        "medium": (2, 4),
        "large": (4, 7),
    }

    # Default for unknown effort levels
    DEFAULT_BOUNDS = (2, 5)

    def detect_drift(
        self,
        progress: MilestoneProgress,
    ) -> Optional[tuple[str, str]]:
        """
        Check if a milestone is showing effort drift.

        Parameters
        ----------
        progress : MilestoneProgress
            The milestone progress data to analyze.

        Returns
        -------
        tuple[str, str] | None
            (drift_type, explanation) if drift detected, else None.
            drift_type is one of: 'drifting_slow', 'drifting_fast'.
        """
        if progress.is_terminal:
            return None

        bounds = self.EFFORT_BOUNDS.get(
            progress.estimated_effort.lower(),
            self.DEFAULT_BOUNDS,
        )
        min_expected, max_expected = bounds
        actual = progress.actual_rounds_spent

        # Check for slow drift (exceeding upper bound)
        if actual > max_expected:
            drift_type = "drifting_slow"
            explanation = (
                f"Milestone {progress.milestone_index} has taken {actual} rounds "
                f"but was estimated as '{progress.estimated_effort}' "
                f"(expected {min_expected}-{max_expected} rounds). "
                f"Consider updating effort estimate or investigating blockers."
            )
            return (drift_type, explanation)

        # Check for fast completion (only meaningful if in progress)
        # We don't flag "fast" for pending milestones that haven't started
        if progress.status == MilestoneStatus.IN_PROGRESS and actual < min_expected:
            # Only flag fast drift if there's meaningful progress
            # (some criteria met) but it's going unusually quickly
            if progress.criteria_satisfied_count > 0 and actual < min_expected:
                drift_type = "drifting_fast"
                explanation = (
                    f"Milestone {progress.milestone_index} is progressing faster "
                    f"than expected ({actual} rounds vs. minimum {min_expected} for "
                    f"'{progress.estimated_effort}'). May be easier than anticipated."
                )
                return (drift_type, explanation)

        return None

    def detect_all_drift(
        self,
        all_progress: List[MilestoneProgress],
    ) -> List[tuple[int, str, str]]:
        """
        Check all milestones for effort drift.

        Parameters
        ----------
        all_progress : List[MilestoneProgress]
            Progress data for all milestones.

        Returns
        -------
        List[tuple[int, str, str]]
            List of (milestone_index, drift_type, explanation) for each
            milestone showing drift.
        """
        drifts: List[tuple[int, str, str]] = []

        for progress in all_progress:
            result = self.detect_drift(progress)
            if result:
                drift_type, explanation = result
                drifts.append((progress.milestone_index, drift_type, explanation))

        return drifts


# ======================================================================
# Meta-Plan Updater
# ======================================================================

class MetaPlanUpdater:
    """
    Manages dynamic updates to the MetaPlan based on session progress.

    The updater tracks milestone progress, adjusts effort estimates,
    can add/defer/cancel milestones, and reorder priorities based on
    what actually happens during the session.

    Key responsibilities:
      - Track which agents are working on which milestones
      - Record checkpoint decisions and update milestone progress
      - Detect effort drift and blocking situations
      - Generate replan suggestions (LLM-powered or heuristic)
      - Apply approved replans to the MetaPlan

    Parameters
    ----------
    meta_plan : MetaPlan
        The session's MetaPlan. This object is mutated in-place as
        changes are applied — the same reference is shared with
        CheckpointLifecycleManager and CheckpointReviewer.

    backend_registry : optional
        LLM backend for smart replanning. If None, all replan suggestions
        use heuristic rules from EffortDriftDetector.
    """

    def __init__(
        self,
        meta_plan: "MetaPlan",
        backend_registry=None,
    ) -> None:
        self._plan = meta_plan
        self._backend = backend_registry

        # Initialize progress tracking for all existing milestones
        self._progress: Dict[int, MilestoneProgress] = {}
        self._initialize_progress()

        # Track agent -> milestone assignments
        self._agent_assignments: Dict[str, int] = {}

        # Drift detector for heuristic fallback
        self._drift_detector = EffortDriftDetector()

        # History of replan suggestions
        self._suggestion_history: List[ReplanSuggestion] = []
        self._applied_replans: List[ReplanResult] = []

    def _initialize_progress(self) -> None:
        """Initialize MilestoneProgress for all existing milestones."""
        for idx, milestone in enumerate(self._plan.milestones):
            if idx not in self._progress:
                self._progress[idx] = MilestoneProgress(
                    milestone_index=idx,
                    estimated_effort=milestone.estimated_effort,
                    merge_criteria_met=[False] * len(milestone.merge_criteria),
                )

    # ------------------------------------------------------------------
    # Progress Tracking
    # ------------------------------------------------------------------

    async def record_checkpoint_result(
        self,
        decision: "CheckpointDecision",
    ) -> MilestoneProgress:
        """
        Record a checkpoint decision and update milestone progress.

        This is the main integration point with CheckpointLifecycleManager.
        After each checkpoint cycle, call this method for each decision.

        Updates:
          - actual_rounds_spent counter
          - merge_criteria_met tracking (if criteria can be inferred)
          - milestone status (may transition to COMPLETED)
          - assigned_agents list

        Parameters
        ----------
        decision : CheckpointDecision
            The checkpoint review decision to record.

        Returns
        -------
        MilestoneProgress
            Updated progress for the affected milestone, or None if
            no milestone was involved.
        """
        milestone_idx = decision.milestone_index
        if milestone_idx is None:
            logger.debug(
                "MetaPlanUpdater: checkpoint decision for %s has no milestone index",
                decision.branch_name,
            )
            # Return a dummy progress for API consistency
            return MilestoneProgress(milestone_index=-1)

        if milestone_idx < 0 or milestone_idx >= len(self._plan.milestones):
            logger.warning(
                "MetaPlanUpdater: invalid milestone index %d in decision for %s",
                milestone_idx, decision.branch_name,
            )
            return MilestoneProgress(milestone_index=-1)

        progress = self._progress.get(milestone_idx)
        if progress is None:
            # Shouldn't happen, but be defensive
            progress = MilestoneProgress(
                milestone_index=milestone_idx,
                estimated_effort=self._plan.milestones[milestone_idx].estimated_effort,
                merge_criteria_met=[False] * len(
                    self._plan.milestones[milestone_idx].merge_criteria
                ),
            )
            self._progress[milestone_idx] = progress

        # Record the decision
        progress.checkpoint_decisions.append(decision)

        # Increment rounds spent
        progress.actual_rounds_spent += 1

        # Track agent assignment
        agent_id = self._extract_agent_id(decision.branch_name)
        if agent_id and agent_id not in progress.assigned_agents:
            progress.assigned_agents.append(agent_id)
            self._agent_assignments[agent_id] = milestone_idx

        # Update status to IN_PROGRESS if still PENDING
        if progress.status == MilestoneStatus.PENDING:
            progress.status = MilestoneStatus.IN_PROGRESS
            progress.started_at = datetime.now(timezone.utc)
            logger.info(
                "MetaPlanUpdater: milestone %d transitioned to IN_PROGRESS",
                milestone_idx,
            )

        # If approved, check if all merge criteria are met
        if decision.approved:
            # Use alignment score as a proxy for criteria satisfaction
            # Higher alignment suggests more criteria met
            alignment = decision.meta_plan_alignment

            # Heuristic: if alignment is very high, mark all criteria met
            # In a more sophisticated system, we'd parse the diff for specific criteria
            if alignment >= 0.9:
                progress.merge_criteria_met = [True] * len(progress.merge_criteria_met)
            elif alignment >= 0.7:
                # Mark some criteria as met based on alignment
                num_to_mark = int(alignment * len(progress.merge_criteria_met))
                for i in range(num_to_mark):
                    if i < len(progress.merge_criteria_met):
                        progress.merge_criteria_met[i] = True

            # Check if all criteria are now satisfied
            if all(progress.merge_criteria_met):
                progress.status = MilestoneStatus.COMPLETED
                progress.completed_at = datetime.now(timezone.utc)
                logger.info(
                    "MetaPlanUpdater: milestone %d COMPLETED (all criteria met)",
                    milestone_idx,
                )
                progress.notes.append(
                    f"Completed at round {progress.actual_rounds_spent} "
                    f"(alignment={alignment:.2f})"
                )

        # Add reviewer notes if present
        if decision.reviewer_notes:
            progress.notes.append(f"Reviewer: {decision.reviewer_notes}")

        logger.debug(
            "MetaPlanUpdater: recorded checkpoint for milestone %d "
            "(status=%s, rounds=%d, criteria=%d/%d)",
            milestone_idx, progress.status.value, progress.actual_rounds_spent,
            progress.criteria_satisfied_count, progress.criteria_total_count,
        )

        return progress

    async def mark_milestone_status(
        self,
        milestone_index: int,
        status: MilestoneStatus,
        reason: str = "",
    ) -> None:
        """
        Manually update a milestone's status.

        Use this for explicit status changes like BLOCKED or DEFERRED
        that are determined by external logic.

        Parameters
        ----------
        milestone_index : int
            Index of the milestone to update.

        status : MilestoneStatus
            New status to set.

        reason : str
            Human-readable reason for the status change.
        """
        if milestone_index < 0 or milestone_index >= len(self._plan.milestones):
            raise ValueError(f"Invalid milestone index: {milestone_index}")

        progress = self._progress.get(milestone_index)
        if progress is None:
            raise ValueError(f"No progress tracking for milestone {milestone_index}")

        old_status = progress.status
        progress.status = status

        if status == MilestoneStatus.BLOCKED:
            progress.blocking_reason = reason
        elif status == MilestoneStatus.COMPLETED:
            progress.completed_at = datetime.now(timezone.utc)
        elif status == MilestoneStatus.IN_PROGRESS:
            if progress.started_at is None:
                progress.started_at = datetime.now(timezone.utc)

        if reason:
            progress.notes.append(f"Status change: {old_status.value} → {status.value}: {reason}")

        logger.info(
            "MetaPlanUpdater: milestone %d status changed %s → %s (%s)",
            milestone_index, old_status.value, status.value, reason or "no reason given",
        )

    def get_progress(self, milestone_index: int) -> Optional[MilestoneProgress]:
        """
        Get current progress for a milestone.

        Parameters
        ----------
        milestone_index : int
            Index of the milestone.

        Returns
        -------
        MilestoneProgress | None
            Progress data, or None if index is invalid.
        """
        if milestone_index < 0 or milestone_index >= len(self._plan.milestones):
            return None
        return self._progress.get(milestone_index)

    def get_all_progress(self) -> List[MilestoneProgress]:
        """
        Get progress for all milestones.

        Returns
        -------
        List[MilestoneProgress]
            Progress data for all milestones, in index order.
        """
        result = []
        for idx in range(len(self._plan.milestones)):
            if idx in self._progress:
                result.append(self._progress[idx])
            else:
                # Initialize missing progress
                progress = MilestoneProgress(
                    milestone_index=idx,
                    estimated_effort=self._plan.milestones[idx].estimated_effort,
                    merge_criteria_met=[False] * len(
                        self._plan.milestones[idx].merge_criteria
                    ),
                )
                self._progress[idx] = progress
                result.append(progress)
        return result

    def get_active_milestones(self) -> List[int]:
        """Return indices of milestones that are not terminal (completed/cancelled)."""
        return [
            idx for idx, prog in self._progress.items()
            if not prog.is_terminal
        ]

    def get_milestone_for_agent(self, agent_id: str) -> Optional[int]:
        """Get the milestone index an agent is currently assigned to."""
        return self._agent_assignments.get(agent_id)

    # ------------------------------------------------------------------
    # Plan Mutation
    # ------------------------------------------------------------------

    async def add_milestone(
        self,
        milestone: "Milestone",
        insert_after: Optional[int] = None,
    ) -> int:
        """
        Add a new milestone to the MetaPlan.

        The milestone is inserted after the specified index (or appended
        if insert_after is None). All subsequent milestone indices are
        updated in the progress tracking.

        Parameters
        ----------
        milestone : Milestone
            The milestone to add.

        insert_after : int | None
            Index after which to insert. If None, appends to the end.

        Returns
        -------
        int
            Index of the newly added milestone.
        """
        # Determine insertion point
        if insert_after is None:
            insert_idx = len(self._plan.milestones)
            self._plan.milestones.append(milestone)
        else:
            if insert_after < 0 or insert_after >= len(self._plan.milestones):
                raise ValueError(f"Invalid insert_after index: {insert_after}")
            insert_idx = insert_after + 1
            self._plan.milestones.insert(insert_idx, milestone)

        # Initialize progress for the new milestone
        new_progress = MilestoneProgress(
            milestone_index=insert_idx,
            estimated_effort=milestone.estimated_effort,
            merge_criteria_met=[False] * len(milestone.merge_criteria),
        )

        # Shift progress entries for subsequent milestones
        shifted_progress: Dict[int, MilestoneProgress] = {}
        for idx in sorted(self._progress.keys(), reverse=True):
            if idx >= insert_idx:
                # Shift this progress entry
                prog = self._progress[idx]
                prog.milestone_index = idx + 1
                shifted_progress[idx + 1] = prog
            else:
                shifted_progress[idx] = self._progress[idx]

        # Add the new progress entry
        shifted_progress[insert_idx] = new_progress
        self._progress = shifted_progress

        # Update agent assignments
        for agent_id, ms_idx in list(self._agent_assignments.items()):
            if ms_idx >= insert_idx:
                self._agent_assignments[agent_id] = ms_idx + 1

        logger.info(
            "MetaPlanUpdater: added milestone at index %d: %s",
            insert_idx, milestone.title,
        )

        return insert_idx

    async def defer_milestone(
        self,
        milestone_index: int,
        reason: str,
    ) -> None:
        """
        Move a milestone to DEFERRED status and push it to the end of the list.

        Deferred milestones are deprioritized but not cancelled. They can
        be reactivated later if needed.

        Parameters
        ----------
        milestone_index : int
            Index of the milestone to defer.

        reason : str
            Human-readable reason for deferring.
        """
        if milestone_index < 0 or milestone_index >= len(self._plan.milestones):
            raise ValueError(f"Invalid milestone index: {milestone_index}")

        progress = self._progress.get(milestone_index)
        if progress is None:
            raise ValueError(f"No progress tracking for milestone {milestone_index}")

        if progress.is_terminal:
            raise ValueError(
                f"Cannot defer milestone {milestone_index}: "
                f"already in terminal state {progress.status.value}"
            )

        # Mark as deferred
        progress.status = MilestoneStatus.DEFERRED
        progress.notes.append(f"Deferred: {reason}")

        # Move to end of milestones list
        milestone = self._plan.milestones.pop(milestone_index)
        self._plan.milestones.append(milestone)

        # Update progress indices
        old_progress = self._progress.pop(milestone_index)
        old_progress.milestone_index = len(self._plan.milestones) - 1

        # Shift progress entries
        shifted_progress: Dict[int, MilestoneProgress] = {}
        for idx, prog in self._progress.items():
            if idx > milestone_index:
                prog.milestone_index = idx - 1
                shifted_progress[idx - 1] = prog
            else:
                shifted_progress[idx] = prog

        # Add the deferred progress at the end
        shifted_progress[len(self._plan.milestones) - 1] = old_progress
        self._progress = shifted_progress

        # Update agent assignments
        for agent_id, ms_idx in list(self._agent_assignments.items()):
            if ms_idx == milestone_index:
                self._agent_assignments[agent_id] = len(self._plan.milestones) - 1
            elif ms_idx > milestone_index:
                self._agent_assignments[agent_id] = ms_idx - 1

        logger.info(
            "MetaPlanUpdater: deferred milestone %d → %d (%s)",
            milestone_index, len(self._plan.milestones) - 1, reason,
        )

    async def cancel_milestone(
        self,
        milestone_index: int,
        reason: str,
    ) -> None:
        """
        Cancel a milestone that's no longer relevant.

        Cancelled milestones remain in the list (for historical record)
        but are marked as CANCELLED and excluded from active assignment.

        Parameters
        ----------
        milestone_index : int
            Index of the milestone to cancel.

        reason : str
            Human-readable reason for cancellation.
        """
        if milestone_index < 0 or milestone_index >= len(self._plan.milestones):
            raise ValueError(f"Invalid milestone index: {milestone_index}")

        progress = self._progress.get(milestone_index)
        if progress is None:
            raise ValueError(f"No progress tracking for milestone {milestone_index}")

        # Mark as cancelled
        progress.status = MilestoneStatus.CANCELLED
        progress.notes.append(f"Cancelled: {reason}")

        # Clear agent assignments for this milestone
        agents_to_clear = [
            agent_id for agent_id, ms_idx in self._agent_assignments.items()
            if ms_idx == milestone_index
        ]
        for agent_id in agents_to_clear:
            del self._agent_assignments[agent_id]

        logger.info(
            "MetaPlanUpdater: cancelled milestone %d (%s)",
            milestone_index, reason,
        )

    async def update_effort_estimate(
        self,
        milestone_index: int,
        new_estimate: str,
    ) -> None:
        """
        Update a milestone's effort estimate based on actual progress.

        This updates both the Milestone.estimated_effort field and the
        MilestoneProgress.estimated_effort field.

        Parameters
        ----------
        milestone_index : int
            Index of the milestone to update.

        new_estimate : str
            New effort estimate ('small', 'medium', or 'large').
        """
        if milestone_index < 0 or milestone_index >= len(self._plan.milestones):
            raise ValueError(f"Invalid milestone index: {milestone_index}")

        if new_estimate.lower() not in ("small", "medium", "large"):
            raise ValueError(f"Invalid effort estimate: {new_estimate}")

        # Update the milestone
        old_estimate = self._plan.milestones[milestone_index].estimated_effort
        self._plan.milestones[milestone_index].estimated_effort = new_estimate.lower()

        # Update the progress
        progress = self._progress.get(milestone_index)
        if progress:
            progress.estimated_effort = new_estimate.lower()
            progress.notes.append(
                f"Effort estimate updated: {old_estimate} → {new_estimate}"
            )

        logger.info(
            "MetaPlanUpdater: milestone %d effort updated %s → %s",
            milestone_index, old_estimate, new_estimate,
        )

    async def reorder_milestones(
        self,
        new_order: List[int],
    ) -> None:
        """
        Reorder milestones by index.

        The new_order must include all non-completed, non-cancelled milestones.
        Completed/cancelled milestones stay in place.

        Parameters
        ----------
        new_order : List[int]
            New ordering of milestone indices. Must be a valid permutation
            of the active milestone indices.

        Raises
        ------
        ValueError
            If new_order is invalid (missing milestones, duplicates,
            or invalid indices).
        """
        # Identify active (non-terminal) milestones
        active_indices = set(self.get_active_milestones())
        terminal_indices = set(
            idx for idx in range(len(self._plan.milestones))
            if idx not in active_indices
        )

        # Validate new_order
        new_order_set = set(new_order)
        if new_order_set != active_indices:
            missing = active_indices - new_order_set
            extra = new_order_set - active_indices
            if missing:
                raise ValueError(f"new_order missing active milestones: {missing}")
            if extra:
                raise ValueError(f"new_order contains non-active milestones: {extra}")

        if len(new_order) != len(set(new_order)):
            raise ValueError("new_order contains duplicate indices")

        # Build the new milestones list
        # Terminal milestones stay in their original positions
        old_milestones = list(self._plan.milestones)
        new_milestones: List["Milestone"] = [None] * len(old_milestones)  # type: ignore

        # First, place terminal milestones
        for idx in terminal_indices:
            new_milestones[idx] = old_milestones[idx]

        # Then, place active milestones in new order
        # We need to map old positions to new positions
        # Strategy: find empty slots (non-terminal positions) and fill them in order
        non_terminal_positions = [
            idx for idx in range(len(old_milestones))
            if idx not in terminal_indices
        ]

        for new_pos, old_idx in enumerate(new_order):
            if new_pos < len(non_terminal_positions):
                target_pos = non_terminal_positions[new_pos]
                new_milestones[target_pos] = old_milestones[old_idx]

        # Remove any None placeholders (shouldn't happen if validation passed)
        new_milestones = [m for m in new_milestones if m is not None]

        # Handle the case where terminal milestones were interspersed
        # In this case, we need a simpler approach: just reorder everything
        # and let terminal ones move too (they're marked anyway)
        # Actually, let's simplify: just apply the new order to the entire list

        # Simpler approach: build a completely new ordered list
        ordered_milestones: List["Milestone"] = []
        old_to_new: Dict[int, int] = {}

        # Add milestones in the new order
        for new_idx, old_idx in enumerate(new_order):
            old_to_new[old_idx] = new_idx
            ordered_milestones.append(old_milestones[old_idx])

        # Add terminal milestones at the end (or we could preserve their positions)
        for idx in sorted(terminal_indices):
            if idx not in old_to_new:
                old_to_new[idx] = len(ordered_milestones)
                ordered_milestones.append(old_milestones[idx])

        # Apply the new ordering
        self._plan.milestones = ordered_milestones

        # Update progress indices
        new_progress: Dict[int, MilestoneProgress] = {}
        for old_idx, new_idx in old_to_new.items():
            if old_idx in self._progress:
                prog = self._progress[old_idx]
                prog.milestone_index = new_idx
                new_progress[new_idx] = prog
        self._progress = new_progress

        # Update agent assignments
        for agent_id, old_idx in list(self._agent_assignments.items()):
            if old_idx in old_to_new:
                self._agent_assignments[agent_id] = old_to_new[old_idx]

        logger.info(
            "MetaPlanUpdater: reordered milestones, new order: %s",
            new_order,
        )

    # ------------------------------------------------------------------
    # Smart Replanning
    # ------------------------------------------------------------------

    async def suggest_replan(
        self,
        whiteboard_snapshot: Optional[Any] = None,
        conflict_log: Optional[Any] = None,
    ) -> ReplanSuggestion:
        """
        Analyze current progress and suggest plan adjustments.

        Looks at:
          - Which milestones are taking longer than estimated
          - Which milestones were completed faster than expected
          - Any BLOCKED milestones
          - Whiteboard content for new discoveries
          - Conflict log for unresolved disagreements

        With LLM backend: generates a thoughtful replan suggestion.
        Without LLM: uses heuristic rules (effort drift detection, blocking analysis).

        The returned suggestion does NOT auto-apply. The caller decides
        whether to proceed with apply_replan().

        Parameters
        ----------
        whiteboard_snapshot : optional
            Current whiteboard state for context. Used by LLM backend
            for informed suggestions.

        conflict_log : optional
            Log of detected conflicts during the session. Used to identify
            disagreements that might warrant replanning.

        Returns
        -------
        ReplanSuggestion
            Proposed changes to the MetaPlan.
        """
        all_progress = self.get_all_progress()

        # Try LLM-powered replanning first
        if self._backend is not None:
            try:
                suggestion = await self._llm_suggest_replan(
                    all_progress, whiteboard_snapshot, conflict_log
                )
                self._suggestion_history.append(suggestion)
                return suggestion
            except Exception as exc:
                logger.warning(
                    "LLM replan suggestion failed (%s); falling back to heuristics",
                    exc,
                )

        # Heuristic fallback
        suggestion = self._heuristic_suggest_replan(
            all_progress, whiteboard_snapshot, conflict_log
        )
        self._suggestion_history.append(suggestion)
        return suggestion

    async def _llm_suggest_replan(
        self,
        all_progress: List[MilestoneProgress],
        whiteboard_snapshot: Optional[Any],
        conflict_log: Optional[Any],
    ) -> ReplanSuggestion:
        """Generate replan suggestion using LLM backend."""
        # Build progress summary
        progress_summary = self._build_progress_summary(all_progress)

        # Build whiteboard summary if available
        whiteboard_summary = ""
        if whiteboard_snapshot is not None:
            # Assume whiteboard has a way to get compressed state
            if hasattr(whiteboard_snapshot, 'compressed_state'):
                try:
                    whiteboard_summary = await whiteboard_snapshot.compressed_state(
                        self._plan.session_id, max_chars=1000
                    )
                except Exception:
                    pass
            elif hasattr(whiteboard_snapshot, 'to_summary'):
                whiteboard_summary = str(whiteboard_snapshot.to_summary())[:1000]

        # Build conflict summary if available
        conflict_summary = ""
        if conflict_log is not None and hasattr(conflict_log, 'conflicts'):
            conflicts = conflict_log.conflicts[:5]  # Limit to first 5
            conflict_summary = "\n".join(
                f"- {c.description}" for c in conflicts
            )

        prompt = (
            f"You are NWTN, a project planning AI. Analyze the current session "
            f"progress and suggest adjustments to the MetaPlan.\n\n"
            f"Session: {self._plan.session_id}\n"
            f"Objective: {self._plan.objective}\n\n"
            f"Progress:\n{progress_summary}\n\n"
            f"Whiteboard summary:\n{whiteboard_summary or 'Not available'}\n\n"
            f"Recent conflicts:\n{conflict_summary or 'None'}\n\n"
            f"Suggest adjustments to the MetaPlan. Return ONLY a JSON object with:\n"
            f"  adjustments (list): each has milestone_index (int), action (str), "
            f"details (str), reason (str)\n"
            f"  rationale (str): overall explanation\n"
            f"  confidence (float 0-1): how confident in this suggestion\n"
            f"\n"
            f"Actions can be: 'update_effort', 'defer', 'cancel', 'reorder', "
            f"'mark_blocked', 'unblock', 'add_criteria', 'remove_criteria'.\n"
            f"Be conservative — only suggest changes that clearly improve the plan."
        )

        result = await self._backend.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.2,
        )

        # Parse LLM response
        from .interview import _extract_json_object
        data = _extract_json_object(result.text.strip())

        if not data:
            # Fallback to heuristics if parsing fails
            logger.warning("LLM returned unparseable replan JSON; using heuristics")
            return self._heuristic_suggest_replan(
                all_progress, whiteboard_snapshot, conflict_log
            )

        # Build adjustments
        adjustments: List[MilestoneAdjustment] = []
        for adj_data in data.get("adjustments", []):
            adjustments.append(MilestoneAdjustment(
                milestone_index=int(adj_data.get("milestone_index", 0)),
                action=str(adj_data.get("action", "")),
                details=str(adj_data.get("details", "")),
                reason=str(adj_data.get("reason", "")),
            ))

        return ReplanSuggestion(
            session_id=self._plan.session_id,
            adjustments=adjustments,
            new_milestones=[],
            suggested_order=None,
            rationale=str(data.get("rationale", "LLM analysis")),
            confidence=float(data.get("confidence", 0.7)),
            llm_assisted=True,
        )

    def _heuristic_suggest_replan(
        self,
        all_progress: List[MilestoneProgress],
        whiteboard_snapshot: Optional[Any],
        conflict_log: Optional[Any],
    ) -> ReplanSuggestion:
        """Generate replan suggestion using heuristic rules."""
        adjustments: List[MilestoneAdjustment] = []
        rationale_parts: List[str] = []

        # 1. Check for effort drift
        drifts = self._drift_detector.detect_all_drift(all_progress)
        for ms_idx, drift_type, explanation in drifts:
            if drift_type == "drifting_slow":
                # Suggest updating effort estimate or investigating blockers
                progress = self._progress.get(ms_idx)
                if progress and progress.status != MilestoneStatus.BLOCKED:
                    # Check if it's been slow for multiple rounds
                    if progress.actual_rounds_spent > 5:
                        adjustments.append(MilestoneAdjustment(
                            milestone_index=ms_idx,
                            action="update_effort",
                            details="large",
                            reason=explanation,
                        ))
                        rationale_parts.append(
                            f"Milestone {ms_idx} is running slow; suggest upgrading effort estimate."
                        )
            elif drift_type == "drifting_fast":
                # Could suggest downgrading effort, but be conservative
                pass

        # 2. Check for blocked milestones
        blocked_milestones = [
            p for p in all_progress
            if p.status == MilestoneStatus.BLOCKED
        ]
        for blocked in blocked_milestones:
            rationale_parts.append(
                f"Milestone {blocked.milestone_index} is blocked: "
                f"{blocked.blocking_reason or 'unknown reason'}"
            )
            # Could suggest unblock actions, but this requires more context

        # 3. Check for deferred milestones that might need attention
        deferred_milestones = [
            p for p in all_progress
            if p.status == MilestoneStatus.DEFERRED
        ]
        if deferred_milestones:
            rationale_parts.append(
                f"{len(deferred_milestones)} milestone(s) are deferred."
            )

        # 4. Check completion progress
        completed_count = sum(
            1 for p in all_progress
            if p.status == MilestoneStatus.COMPLETED
        )
        total_count = len(all_progress)

        # 5. Check for conflicts that might warrant replanning
        conflict_count = 0
        if conflict_log is not None and hasattr(conflict_log, 'count'):
            conflict_count = conflict_log.count

        if conflict_count > 3:
            rationale_parts.append(
                f"{conflict_count} conflicts detected; consider reviewing plan coherence."
            )

        # Build confidence based on how much we know
        confidence = 0.5  # Base confidence for heuristics
        if drifts:
            confidence += 0.2
        if blocked_milestones:
            confidence += 0.1
        confidence = min(confidence, 0.85)  # Cap at 0.85 for heuristic

        # Build rationale
        if not rationale_parts:
            rationale = "No significant issues detected. Plan appears on track."
            confidence = 0.6
        else:
            rationale = " | ".join(rationale_parts)

        return ReplanSuggestion(
            session_id=self._plan.session_id,
            adjustments=adjustments,
            new_milestones=[],
            suggested_order=None,
            rationale=rationale,
            confidence=confidence,
            llm_assisted=False,
        )

    async def apply_replan(
        self,
        suggestion: ReplanSuggestion,
    ) -> ReplanResult:
        """
        Apply an approved ReplanSuggestion to the MetaPlan.

        Processes each adjustment in order and applies the changes.
        If any adjustment fails, records the error but continues
        processing remaining adjustments.

        Parameters
        ----------
        suggestion : ReplanSuggestion
            The approved suggestion to apply.

        Returns
        -------
        ReplanResult
            Summary of what was applied.
        """
        if not suggestion.has_adjustments:
            return ReplanResult(
                success=True,
                summary="No adjustments to apply.",
            )

        errors: List[str] = []
        adjustments_applied = 0
        milestones_added = 0
        milestones_deferred = 0
        milestones_cancelled = 0

        # Apply each adjustment
        for adj in suggestion.adjustments:
            try:
                result = await self._apply_adjustment(adj)
                adjustments_applied += 1
                if result == "added":
                    milestones_added += 1
                elif result == "deferred":
                    milestones_deferred += 1
                elif result == "cancelled":
                    milestones_cancelled += 1
            except Exception as exc:
                error_msg = (
                    f"Failed to apply adjustment {adj.action} "
                    f"to milestone {adj.milestone_index}: {exc}"
                )
                errors.append(error_msg)
                logger.warning("MetaPlanUpdater: %s", error_msg)

        # Add new milestones
        for new_ms in suggestion.new_milestones:
            try:
                await self.add_milestone(new_ms)
                milestones_added += 1
            except Exception as exc:
                errors.append(f"Failed to add milestone: {exc}")

        # Apply reordering if specified
        if suggestion.suggested_order:
            try:
                await self.reorder_milestones(suggestion.suggested_order)
            except Exception as exc:
                errors.append(f"Failed to reorder milestones: {exc}")

        success = len(errors) == 0

        # Build summary
        summary_parts: List[str] = []
        if adjustments_applied > 0:
            summary_parts.append(f"{adjustments_applied} adjustment(s) applied")
        if milestones_added > 0:
            summary_parts.append(f"{milestones_added} milestone(s) added")
        if milestones_deferred > 0:
            summary_parts.append(f"{milestones_deferred} milestone(s) deferred")
        if milestones_cancelled > 0:
            summary_parts.append(f"{milestones_cancelled} milestone(s) cancelled")

        summary = "; ".join(summary_parts) if summary_parts else "No changes made"
        if errors:
            summary += f" ({len(errors)} error(s))"

        result = ReplanResult(
            success=success,
            adjustments_applied=adjustments_applied,
            milestones_added=milestones_added,
            milestones_deferred=milestones_deferred,
            milestones_cancelled=milestones_cancelled,
            new_milestone_count=len(self._plan.milestones),
            summary=summary,
            errors=errors,
        )

        self._applied_replans.append(result)

        logger.info(
            "MetaPlanUpdater: applied replan (success=%s, adjustments=%d, errors=%d)",
            success, adjustments_applied, len(errors),
        )

        return result

    async def _apply_adjustment(
        self,
        adj: MilestoneAdjustment,
    ) -> str:
        """
        Apply a single milestone adjustment.

        Returns the type of change made: 'updated', 'deferred', 'cancelled',
        'added', or 'none'.
        """
        action = adj.action.lower()
        ms_idx = adj.milestone_index

        if action == "update_effort":
            await self.update_effort_estimate(ms_idx, adj.details or "medium")
            return "updated"

        elif action == "defer":
            await self.defer_milestone(ms_idx, adj.reason or adj.details or "Replan suggestion")
            return "deferred"

        elif action == "cancel":
            await self.cancel_milestone(ms_idx, adj.reason or adj.details or "Replan suggestion")
            return "cancelled"

        elif action == "mark_blocked":
            await self.mark_milestone_status(
                ms_idx, MilestoneStatus.BLOCKED, adj.reason or adj.details or ""
            )
            return "updated"

        elif action == "unblock":
            progress = self._progress.get(ms_idx)
            if progress and progress.status == MilestoneStatus.BLOCKED:
                await self.mark_milestone_status(
                    ms_idx, MilestoneStatus.IN_PROGRESS,
                    adj.reason or "Unblocked"
                )
            return "updated"

        elif action == "add_criteria":
            # Add a new merge criterion to the milestone
            if ms_idx < 0 or ms_idx >= len(self._plan.milestones):
                raise ValueError(f"Invalid milestone index: {ms_idx}")
            new_criterion = adj.details
            if new_criterion:
                self._plan.milestones[ms_idx].merge_criteria.append(new_criterion)
                # Update progress tracking
                progress = self._progress.get(ms_idx)
                if progress:
                    progress.merge_criteria_met.append(False)
            return "updated"

        elif action == "remove_criteria":
            # Remove a merge criterion (by index or text match)
            # For now, we don't support this as it could break tracking
            logger.warning(
                "MetaPlanUpdater: 'remove_criteria' action not fully supported"
            )
            return "none"

        else:
            logger.warning(
                "MetaPlanUpdater: unknown adjustment action: %s", action
            )
            return "none"

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _extract_agent_id(self, branch_name: str) -> Optional[str]:
        """Extract agent ID from a branch name."""
        # Branch names follow: agent/<role>-<date>
        # We want to return the full agent/<role>-<date> string
        if branch_name.startswith("agent/"):
            return branch_name
        return None

    def _build_progress_summary(
        self,
        all_progress: List[MilestoneProgress],
    ) -> str:
        """Build a human-readable summary of milestone progress."""
        lines: List[str] = []

        for prog in all_progress:
            ms = self._plan.milestones[prog.milestone_index]
            status_icon = {
                MilestoneStatus.PENDING: "⏳",
                MilestoneStatus.IN_PROGRESS: "🔄",
                MilestoneStatus.BLOCKED: "🚫",
                MilestoneStatus.COMPLETED: "✅",
                MilestoneStatus.DEFERRED: "⏸️",
                MilestoneStatus.CANCELLED: "❌",
            }.get(prog.status, "❓")

            criteria_str = f"{prog.criteria_satisfied_count}/{prog.criteria_total_count}"
            lines.append(
                f"  {status_icon} [{prog.milestone_index}] {ms.title} "
                f"({prog.status.value}, effort={prog.estimated_effort}, "
                f"rounds={prog.actual_rounds_spent}, criteria={criteria_str})"
            )

            if prog.blocking_reason:
                lines.append(f"      Blocked: {prog.blocking_reason}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def suggestion_history(self) -> List[ReplanSuggestion]:
        """Return the history of replan suggestions made."""
        return list(self._suggestion_history)

    @property
    def applied_replans(self) -> List[ReplanResult]:
        """Return the history of applied replans."""
        return list(self._applied_replans)

    def status_summary(self) -> Dict[str, Any]:
        """Return a summary of the updater's current state."""
        all_progress = self.get_all_progress()

        status_counts: Dict[str, int] = {}
        for prog in all_progress:
            status_val = prog.status.value
            status_counts[status_val] = status_counts.get(status_val, 0) + 1

        return {
            "session_id": self._plan.session_id,
            "total_milestones": len(self._plan.milestones),
            "status_counts": status_counts,
            "suggestions_made": len(self._suggestion_history),
            "replans_applied": len(self._applied_replans),
            "has_llm_backend": self._backend is not None,
            "agent_assignments": dict(self._agent_assignments),
        }


# ======================================================================
# Integration Note
# ======================================================================

# Integration with CheckpointLifecycleManager:
#
# After each checkpoint cycle in live_scribe.py:
#   1. CheckpointLifecycleManager.initiate_checkpoint() completes
#   2. For each CheckpointDecision → updater.record_checkpoint_result(decision)
#   3. Periodically call updater.suggest_replan()
#   4. If suggestion confidence > threshold → present to user or auto-apply
#   5. updater.apply_replan() mutates the MetaPlan in-place
#   6. CheckpointLifecycleManager.assign_next_checkpoints() now uses updated plan
#
# The MetaPlanUpdater does NOT modify the MetaPlan's milestones list directly
# through external references. It mutates self._plan.milestones in-place,
# which is the same object held by CheckpointLifecycleManager and
# CheckpointReviewer (passed by reference at construction time).
#
# Example integration in live_scribe.py:
#
#     class LiveScribe:
#         def __init__(self, ...):
#             ...
#             self._meta_plan_updater = MetaPlanUpdater(
#                 meta_plan=meta_plan,
#                 backend_registry=backend_registry,
#             )
#
#         async def on_checkpoint_complete(self, decisions: List[CheckpointDecision]):
#             for decision in decisions:
#                 await self._meta_plan_updater.record_checkpoint_result(decision)
#
#             # Periodically check for replan
#             if self._should_check_replan():
#                 suggestion = await self._meta_plan_updater.suggest_replan(
#                     whiteboard_snapshot=await self._store.snapshot(self._plan.session_id),
#                     conflict_log=self._conflict_log,
#                 )
#                 if suggestion.confidence > 0.8 and suggestion.has_adjustments:
#                     result = await self._meta_plan_updater.apply_replan(suggestion)
#                     logger.info("Auto-applied replan: %s", result.summary)


# ======================================================================
# Import additions for team/__init__.py
# ======================================================================
#
# Add the following to prsm/compute/nwtn/team/__init__.py:
#
# from .meta_plan_updater import (
#     MilestoneStatus,
#     MilestoneProgress,
#     MilestoneAdjustment,
#     ReplanSuggestion,
#     ReplanResult,
#     EffortDriftDetector,
#     MetaPlanUpdater,
# )
