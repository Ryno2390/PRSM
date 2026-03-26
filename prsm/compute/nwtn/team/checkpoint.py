"""
Checkpoint Reviewer
===================

NWTN's role as merge manager.  When an agent signals that its work is ready
for integration, the ``CheckpointReviewer`` evaluates the branch diff against
the ``MetaPlan`` and returns an approve/reject decision before any merge is
allowed to proceed.

Review algorithm
----------------
1. Fetch the unified diff of the agent's branch vs. ``main``.
2. Check structural guards (empty diff, diff size limits).
3. Optionally query an LLM to score the diff's alignment with the MetaPlan.
4. If approved, invoke ``BranchManager.merge_branch()`` with a signed commit
   message that records the decision rationale.
5. Return a ``CheckpointDecision`` (always — callers act on ``approved``).

Heuristic fallback
------------------
If no LLM backend is available, the reviewer applies rule-based heuristics:
  - Rejects diffs that touch files in the ``constraints`` list.
  - Accepts diffs that reference ≥1 milestone keyword from the MetaPlan.
  - Falls back to accepting any non-empty diff if no keyword evidence exists
    (trusts the agent, but flags it as ``heuristic=True`` in the decision).

Tamper-evident merge commits
-----------------------------
The commit message written by ``approve_and_merge()`` includes:
  - Session ID and branch name
  - Milestone aligned to
  - Meta-plan alignment score
  - Reviewer rationale
  - The SHA of the last Project Ledger entry (Sub-phase 10.4 back-links)

This embeds the checkpoint decision in the git history so the Project
Ledger's hash chain and the commit log both record the same facts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .branch_manager import BranchManager, GitDiff
from .planner import MetaPlan

logger = logging.getLogger(__name__)

# Diff size limits
MAX_DIFF_LINES = 5000
MIN_DIFF_LINES = 1


# ======================================================================
# Data models
# ======================================================================

@dataclass
class CheckpointDecision:
    """Result of a single checkpoint review."""
    branch_name: str
    session_id: str
    approved: bool
    reason: str
    diff_summary: str
    """Brief summary of what the diff contains."""
    meta_plan_alignment: float
    """0–1 score: how well the diff aligns with the MetaPlan objective."""
    milestone_index: Optional[int]
    """Index into MetaPlan.milestones, or None if no milestone identified."""
    heuristic: bool = False
    """True if the decision was made without an LLM (rule-based only)."""
    reviewer_notes: str = ""
    merge_commit_sha: Optional[str] = None
    """Populated after a successful ``approve_and_merge()`` call."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def verdict(self) -> str:
        return "APPROVED" if self.approved else "REJECTED"


# ======================================================================
# CheckpointReviewer
# ======================================================================

class CheckpointReviewer:
    """
    Reviews agent branch diffs and manages the merge lifecycle.

    Parameters
    ----------
    branch_manager : BranchManager
        Used to fetch diffs and execute merges.
    meta_plan : MetaPlan
        The session's north-star plan against which diffs are evaluated.
    backend_registry : optional
        LLM backend for alignment scoring.  Falls back to heuristics if None.
    ledger_sha : str
        SHA of the latest Project Ledger commit (written into merge commit
        messages for tamper-evident cross-referencing).
    """

    def __init__(
        self,
        branch_manager: BranchManager,
        meta_plan: MetaPlan,
        backend_registry=None,
        ledger_sha: str = "",
    ) -> None:
        self._bm = branch_manager
        self._plan = meta_plan
        self._backend = backend_registry
        self._ledger_sha = ledger_sha
        self._decisions: List[CheckpointDecision] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def review(self, branch_name: str) -> CheckpointDecision:
        """
        Review a branch and return a ``CheckpointDecision``.

        Does NOT automatically merge — call ``approve_and_merge()`` after
        inspecting the decision if you want the merge to proceed.

        Parameters
        ----------
        branch_name : str
            The agent branch to review.

        Returns
        -------
        CheckpointDecision
        """
        logger.info("CheckpointReviewer: reviewing %s", branch_name)

        # Fetch diff
        try:
            diff = await self._bm.get_diff(branch_name)
        except RuntimeError as exc:
            return self._reject(
                branch_name, f"Could not fetch diff: {exc}", diff_summary=""
            )

        # Structural guards
        guard_result = self._structural_guard(diff)
        if guard_result is not None:
            decision = guard_result
            self._decisions.append(decision)
            return decision

        diff_summary = self._summarise_diff(diff)

        # LLM alignment scoring
        if self._backend is not None:
            try:
                decision = await self._llm_review(branch_name, diff, diff_summary)
                self._decisions.append(decision)
                return decision
            except Exception as exc:
                logger.warning(
                    "LLM review failed (%s); falling back to heuristics", exc
                )

        # Heuristic fallback
        decision = self._heuristic_review(branch_name, diff, diff_summary)
        self._decisions.append(decision)
        return decision

    async def approve_and_merge(self, decision: CheckpointDecision) -> CheckpointDecision:
        """
        Execute the merge for an already-approved ``CheckpointDecision``.

        Builds a tamper-evident commit message embedding the session ID,
        branch, milestone, alignment score, and the latest Project Ledger SHA.

        Returns
        -------
        CheckpointDecision
            Same object with ``merge_commit_sha`` populated.
        """
        if not decision.approved:
            raise ValueError(
                f"Cannot merge a rejected decision for {decision.branch_name}"
            )

        milestone_info = ""
        if decision.milestone_index is not None and decision.milestone_index < len(
            self._plan.milestones
        ):
            ms = self._plan.milestones[decision.milestone_index]
            milestone_info = f"\nMilestone: {ms.title}"

        commit_msg = (
            f"checkpoint({decision.branch_name}): merge approved by NWTN\n\n"
            f"Session: {decision.session_id}\n"
            f"Branch: {decision.branch_name}{milestone_info}\n"
            f"Meta-plan alignment: {decision.meta_plan_alignment:.3f}\n"
            f"Reviewer: {'LLM' if not decision.heuristic else 'heuristic'}\n"
            f"Reason: {decision.reason}\n"
            f"Project Ledger SHA: {self._ledger_sha or 'pending'}\n"
        )

        try:
            sha = await self._bm.merge_branch(decision.branch_name, commit_msg)
            decision.merge_commit_sha = sha
            logger.info(
                "CheckpointReviewer: merged %s (sha=%s)", decision.branch_name, sha[:8]
            )
        except RuntimeError as exc:
            logger.error("Merge failed for %s: %s", decision.branch_name, exc)
            raise

        return decision

    async def review_and_merge(self, branch_name: str) -> CheckpointDecision:
        """Convenience: review then merge if approved."""
        decision = await self.review(branch_name)
        if decision.approved:
            await self.approve_and_merge(decision)
        return decision

    def update_ledger_sha(self, sha: str) -> None:
        """Update the Project Ledger SHA used in future merge commit messages."""
        self._ledger_sha = sha

    @property
    def decision_history(self) -> List[CheckpointDecision]:
        return list(self._decisions)

    # ------------------------------------------------------------------
    # Internal: structural guards
    # ------------------------------------------------------------------

    def _structural_guard(self, diff: GitDiff) -> Optional[CheckpointDecision]:
        """Return a rejection decision for obviously bad diffs, or None."""
        if diff.is_empty:
            return self._reject(
                diff.branch_name,
                "Empty diff — no changes to merge",
                diff_summary="(empty)",
            )
        line_count = diff.diff_text.count("\n")
        if line_count > MAX_DIFF_LINES:
            return self._reject(
                diff.branch_name,
                f"Diff too large ({line_count} lines > limit {MAX_DIFF_LINES}). "
                "Break into smaller checkpoints.",
                diff_summary=f"{line_count} lines, {len(diff.files_changed)} files",
            )
        return None

    # ------------------------------------------------------------------
    # Internal: LLM review
    # ------------------------------------------------------------------

    async def _llm_review(
        self, branch_name: str, diff: GitDiff, diff_summary: str
    ) -> CheckpointDecision:
        """Score the diff's alignment with the MetaPlan using an LLM."""
        plan_block = self._plan.to_whiteboard_entry()
        diff_excerpt = diff.diff_text[:3000]  # keep prompt manageable

        prompt = (
            "You are NWTN, a project checkpoint reviewer. Evaluate whether the "
            "following git diff is aligned with the project's MetaPlan.\n\n"
            f"MetaPlan:\n{plan_block}\n\n"
            f"Diff summary: {diff_summary}\n\n"
            f"Diff excerpt:\n```diff\n{diff_excerpt}\n```\n\n"
            "Return ONLY a JSON object with:\n"
            "  approved (bool): should this be merged?\n"
            "  alignment_score (float 0-1): how well does this align with the MetaPlan?\n"
            "  milestone_index (int or null): which milestone (0-based) does this complete?\n"
            "  reason (str): one-sentence rationale\n"
            "  notes (str): any additional reviewer comments\n"
        )
        result = await self._backend.generate(
            prompt=prompt,
            max_tokens=400,
            temperature=0.1,
        )

        from .interview import _extract_json_object
        data = _extract_json_object(result.text.strip())

        approved = bool(data.get("approved", True))
        alignment = float(data.get("alignment_score", 0.7))
        milestone_idx = data.get("milestone_index")
        if isinstance(milestone_idx, str):
            milestone_idx = int(milestone_idx) if milestone_idx.isdigit() else None

        return CheckpointDecision(
            branch_name=branch_name,
            session_id=self._plan.session_id,
            approved=approved,
            reason=str(data.get("reason", "LLM review complete")),
            diff_summary=diff_summary,
            meta_plan_alignment=alignment,
            milestone_index=milestone_idx,
            heuristic=False,
            reviewer_notes=str(data.get("notes", "")),
        )

    # ------------------------------------------------------------------
    # Internal: heuristic review
    # ------------------------------------------------------------------

    def _heuristic_review(
        self, branch_name: str, diff: GitDiff, diff_summary: str
    ) -> CheckpointDecision:
        """
        Rule-based alignment check without an LLM.

        Rules (applied in order):
        1. Reject if diff touches any path listed in constraints.
        2. Score milestone alignment by keyword overlap.
        3. Accept if score > 0 or diff is non-trivial.
        """
        diff_lower = diff.diff_text.lower()

        # Rule 1: constraint violations
        for constraint in self._plan.constraints:
            keywords = re.findall(r"\b\w{4,}\b", constraint.lower())
            if any(kw in diff_lower for kw in keywords[:3]):
                return self._reject(
                    branch_name,
                    f"Diff appears to violate constraint: '{constraint}'",
                    diff_summary=diff_summary,
                    heuristic=True,
                )

        # Rule 2: milestone keyword scoring
        best_score = 0.0
        best_idx: Optional[int] = None

        for idx, milestone in enumerate(self._plan.milestones):
            keywords = re.findall(
                r"\b\w{5,}\b",
                (milestone.title + " " + milestone.description).lower(),
            )
            if not keywords:
                continue
            hits = sum(1 for kw in keywords if kw in diff_lower)
            score = hits / len(keywords)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Rule 3: accept if non-trivial or any keywords matched
        alignment = max(0.5, best_score)  # floor at 0.5 for non-empty diffs
        approved = diff.insertions > 0 or diff.deletions > 0

        reason = (
            f"Heuristic review: {diff.insertions}+ / {diff.deletions}- lines; "
            f"milestone alignment={alignment:.2f}"
        )
        return CheckpointDecision(
            branch_name=branch_name,
            session_id=self._plan.session_id,
            approved=approved,
            reason=reason,
            diff_summary=diff_summary,
            meta_plan_alignment=alignment,
            milestone_index=best_idx,
            heuristic=True,
        )

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _reject(
        self,
        branch_name: str,
        reason: str,
        diff_summary: str,
        heuristic: bool = False,
    ) -> CheckpointDecision:
        return CheckpointDecision(
            branch_name=branch_name,
            session_id=self._plan.session_id,
            approved=False,
            reason=reason,
            diff_summary=diff_summary,
            meta_plan_alignment=0.0,
            milestone_index=None,
            heuristic=heuristic,
        )

    def _summarise_diff(self, diff: GitDiff) -> str:
        files = ", ".join(diff.files_changed[:5])
        if len(diff.files_changed) > 5:
            files += f" (+{len(diff.files_changed) - 5} more)"
        return (
            f"{diff.insertions} insertions, {diff.deletions} deletions "
            f"across {len(diff.files_changed)} file(s): {files}"
        )
