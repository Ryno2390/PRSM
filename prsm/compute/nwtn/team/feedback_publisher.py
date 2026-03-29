"""
EvaluatorFeedbackPublisher
==========================
Publishes CheckpointEvaluator results back to agent inboxes via EventBus,
closing the generator/evaluator feedback loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from prsm.compute.nwtn.bsc.event_bus import EventBus
    from prsm.compute.nwtn.team.evaluator import EvaluationBatch, EvaluationResult

logger = logging.getLogger(__name__)


class EvaluatorFeedbackPublisher:
    """
    Publishes evaluation feedback to agents via the EventBus.

    After a CheckpointEvaluator completes team evaluation, this publisher
    sends structured feedback messages to agents who need improvement:
    - Quality score < 0.8
    - Issues found (non-empty)
    - Unmet criteria (criteria_met < total_criteria)

    Parameters
    ----------
    event_bus : EventBus
        The event bus to publish feedback events to.
    """

    def __init__(self, event_bus: "EventBus"):
        self._bus = event_bus

    async def publish_feedback(
        self,
        batch: "EvaluationBatch",
        round_number: int,
        session_id: str,
    ) -> int:
        """
        Publish feedback for agents that need it.

        Only targets agents with quality_score < 0.8 OR issues_found non-empty
        OR criteria_met < total_criteria.

        Parameters
        ----------
        batch : EvaluationBatch
            The batch of evaluation results from checkpoint evaluation.
        round_number : int
            The current round number.
        session_id : str
            The session ID for event routing.

        Returns
        -------
        int
            Count of feedback messages published.
        """
        from prsm.compute.nwtn.bsc.event_bus import BSCEvent, EventType

        published_count = 0

        for result in batch.results:
            # Calculate criteria counts
            criteria_met_count = sum(1 for v in result.criteria_met.values() if v)
            total_criteria = len(result.criteria_met)

            # Determine if agent needs feedback
            needs_feedback = (
                result.quality_score < 0.8
                or bool(result.issues_found)
                or criteria_met_count < total_criteria
            )

            if not needs_feedback:
                logger.debug(
                    "EvaluatorFeedbackPublisher: skipping %s (score=%.2f, issues=%d, criteria=%d/%d)",
                    result.agent_id,
                    result.quality_score,
                    len(result.issues_found),
                    criteria_met_count,
                    total_criteria,
                )
                continue

            # Build the feedback event
            feedback_message = self.format_feedback_message(result, round_number)

            event = BSCEvent(
                event_type=EventType.EVALUATION_FEEDBACK,
                data={
                    "agent_id": result.agent_id,
                    "round_number": round_number,
                    "quality_score": result.quality_score,
                    "feedback_message": feedback_message,
                    "criteria_met": criteria_met_count,
                    "total_criteria": total_criteria,
                    "issues_found": result.issues_found,
                    "divergence_notes": result.divergence_notes,
                },
                session_id=session_id,
            )

            await self._bus.publish(event)
            published_count += 1

            logger.info(
                "EvaluatorFeedbackPublisher: published feedback for %s "
                "(round=%d, score=%.2f, issues=%d, criteria=%d/%d)",
                result.agent_id,
                round_number,
                result.quality_score,
                len(result.issues_found),
                criteria_met_count,
                total_criteria,
            )

        return published_count

    def format_feedback_message(
        self, result: "EvaluationResult", round_number: int
    ) -> str:
        """
        Format evaluation result as structured feedback.

        Must include sections: quality score, criteria met, issues found,
        what to address.

        Parameters
        ----------
        result : EvaluationResult
            The evaluation result to format.
        round_number : int
            The current round number.

        Returns
        -------
        str
            Formatted feedback message for agent consumption.
        """
        criteria_met_count = sum(1 for v in result.criteria_met.values() if v)
        total_criteria = len(result.criteria_met)

        lines: List[str] = [
            f"## Evaluation Feedback (Round {round_number})",
            "",
            f"**Agent:** {result.agent_id}",
            f"**Quality Score:** {result.quality_score:.2f} {'✅ PASS' if result.passed else '❌ NEEDS WORK'}",
            "",
            "### Criteria Met",
            f"Met **{criteria_met_count}/{total_criteria}** acceptance criteria.",
            "",
        ]

        # List specific criteria status
        if result.criteria_met:
            lines.append("| Criterion | Status |")
            lines.append("|-----------|--------|")
            for criterion, met in result.criteria_met.items():
                status = "✅ Met" if met else "❌ Not met"
                # Truncate long criteria for readability
                display_criterion = criterion if len(criterion) <= 60 else criterion[:57] + "..."
                lines.append(f"| {display_criterion} | {status} |")
            lines.append("")

        # Issues found section
        if result.issues_found:
            lines.append("### Issues Found")
            lines.append("")
            for i, issue in enumerate(result.issues_found, 1):
                lines.append(f"{i}. {issue}")
            lines.append("")

        # What to address section
        lines.append("### What to Address")
        lines.append("")

        if result.quality_score < 0.8:
            lines.append(
                f"- **Improve quality:** Current score ({result.quality_score:.2f}) is below target (0.8). "
                "Focus on completeness and correctness."
            )

        if criteria_met_count < total_criteria:
            unmet = [
                c for c, met in result.criteria_met.items() if not met
            ]
            lines.append(
                f"- **Address unmet criteria:** {len(unmet)} criteria remain unmet. "
                "Review your work against each acceptance criterion."
            )

        if result.issues_found:
            lines.append(
                f"- **Fix identified issues:** {len(result.issues_found)} issue(s) flagged. "
                "Address each issue systematically."
            )

        if result.divergence_notes:
            lines.append("")
            lines.append("### Evaluator Divergence Notes")
            lines.append("")
            lines.append(result.divergence_notes)

        return "\n".join(lines)


__all__ = ["EvaluatorFeedbackPublisher"]
