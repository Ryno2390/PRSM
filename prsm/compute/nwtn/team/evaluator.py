"""
Checkpoint Evaluator
====================

Adversarial, criterion-based evaluation of agent work at checkpoint synthesis.

Inspired by Anthropic's guidance that "out of the box, Claude is a poor QA
agent" — evaluators must be explicitly tuned for skepticism.  The
``CheckpointEvaluator`` assumes work is **incomplete until proven otherwise**
and checks every acceptance criterion from the MetaPlan milestone before
the NarrativeSynthesizer produces the session narrative.

Design goals
------------
- **Skeptical by default**: no criterion is considered met unless the agent's
  whiteboard entries contain explicit, verifiable evidence.
- **Criterion-driven**: evaluation rubric comes directly from the MetaPlan
  milestone's ``merge_criteria`` (per-milestone) and plan-level
  ``success_criteria``.
- **Divergence logging**: where the evaluator's judgment differs from what
  an agent claims in their entries, the difference is recorded for the P3
  tuning loop.
- **Tunable**: prompt overrides for individual (agent, criterion) pairs allow
  human-in-the-loop refinement without rewriting the evaluator class.

Quick start
-----------
.. code-block:: python

    from prsm.compute.nwtn.team.evaluator import CheckpointEvaluator
    from prsm.compute.nwtn.team.planner import MetaPlan

    evaluator = CheckpointEvaluator(meta_plan=meta_plan, backend_registry=backend)
    result = await evaluator.evaluate_agent_work(
        agent_id="agent/coder-20260326",
        checkpoint_entries=entries,
        milestone=meta_plan.milestones[0],
    )
    print(result.quality_score, result.issues_found)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from prsm.compute.nwtn.whiteboard.schema import WhiteboardEntry
    from prsm.compute.nwtn.team.planner import MetaPlan, Milestone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence levels used when the backend is unavailable
# ---------------------------------------------------------------------------
_HEURISTIC_CONFIDENCE = 0.55
"""Confidence assigned to heuristic evaluations (no LLM backend)."""

_KEYWORD_WEIGHT = 0.15
"""Per-matching-keyword quality bump when doing heuristic scoring."""


# ======================================================================
# Data Models
# ======================================================================

@dataclass
class EvaluationResult:
    """
    Result of evaluating a single agent's checkpoint work.

    Returned by ``CheckpointEvaluator.evaluate_agent_work()``.

    Attributes
    ----------
    agent_id : str
        The evaluated agent's identifier.
    milestone_index : int
        Zero-based index into ``MetaPlan.milestones`` for this checkpoint.
    criteria_met : Dict[str, bool]
        Maps each acceptance criterion string to whether the agent met it.
    quality_score : float
        Holistic quality score in ``[0.0, 1.0]``.  Derived from the fraction
        of criteria met, weighted by confidence.
    issues_found : List[str]
        Concrete problems identified by the evaluator (incomplete work,
        missed edge cases, contradicted assumptions, etc.).
    confidence : float
        How confident the evaluator is in its own judgment ``[0.0, 1.0]``.
        Lower values signal that the criterion evidence was ambiguous.
    divergence_notes : str
        Free-text record of where the evaluator's assessment *differs* from
        what the agent claims in its own entries.  Feeds the P3 tuning loop.
    llm_assisted : bool
        True when the evaluation was produced with an LLM backend.
    evaluated_at : datetime
        UTC timestamp of when this result was produced.
    """
    agent_id: str
    milestone_index: int
    criteria_met: Dict[str, bool] = field(default_factory=dict)
    quality_score: float = 0.0
    issues_found: List[str] = field(default_factory=list)
    confidence: float = 0.0
    divergence_notes: str = ""
    llm_assisted: bool = False
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def passed(self) -> bool:
        """True when the quality score clears the 0.6 pass threshold."""
        return self.quality_score >= 0.6

    @property
    def criteria_pass_rate(self) -> float:
        """Fraction of criteria that were met (0.0 if no criteria)."""
        if not self.criteria_met:
            return 0.0
        met = sum(1 for v in self.criteria_met.values() if v)
        return met / len(self.criteria_met)

    def summary(self) -> str:
        """One-line human-readable summary for log output."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.agent_id} @ milestone {self.milestone_index} — "
            f"score={self.quality_score:.2f}, "
            f"criteria {sum(1 for v in self.criteria_met.values() if v)}/{len(self.criteria_met)} met, "
            f"confidence={self.confidence:.2f}"
        )


@dataclass
class EvaluationBatch:
    """
    Aggregated evaluation results for all agents at a single checkpoint.

    Produced by ``CheckpointEvaluator.evaluate_team()``.
    """
    milestone_index: int
    results: List[EvaluationResult] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed_agents(self) -> List[str]:
        return [r.agent_id for r in self.results if r.passed]

    @property
    def failed_agents(self) -> List[str]:
        return [r.agent_id for r in self.results if not r.passed]

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    def team_quality_score(self) -> float:
        """Mean quality score across all agents."""
        if not self.results:
            return 0.0
        return sum(r.quality_score for r in self.results) / len(self.results)

    def to_narrative_block(self) -> str:
        """Render an evaluation summary suitable for embedding in the checkpoint narrative."""
        lines: List[str] = [
            "### Checkpoint Evaluation Results",
            "",
            f"**Milestone index:** {self.milestone_index}  ",
            f"**Team quality score:** {self.team_quality_score():.2f}  ",
            f"**All passed:** {'✅ Yes' if self.all_passed else '❌ No'}",
            "",
            "| Agent | Score | Criteria Met | Pass |",
            "|-------|-------|--------------|------|",
        ]
        for r in self.results:
            short = r.agent_id.removeprefix("agent/")
            criteria_str = f"{sum(1 for v in r.criteria_met.values() if v)}/{len(r.criteria_met)}"
            pass_icon = "✅" if r.passed else "❌"
            lines.append(f"| {short} | {r.quality_score:.2f} | {criteria_str} | {pass_icon} |")

        if any(r.issues_found for r in self.results):
            lines += ["", "#### Issues Identified", ""]
            for r in self.results:
                if r.issues_found:
                    short = r.agent_id.removeprefix("agent/")
                    lines.append(f"**{short}:**")
                    for issue in r.issues_found:
                        lines.append(f"  - {issue}")
            lines.append("")

        if any(r.divergence_notes for r in self.results):
            lines += ["", "#### Evaluator Divergence Notes (for tuning)", ""]
            for r in self.results:
                if r.divergence_notes:
                    short = r.agent_id.removeprefix("agent/")
                    lines.append(f"**{short}:** {r.divergence_notes}")
            lines.append("")

        return "\n".join(lines)


# ======================================================================
# CheckpointEvaluator
# ======================================================================

class CheckpointEvaluator:
    """
    Adversarial evaluator that checks each agent's work against the MetaPlan
    milestone's acceptance criteria before checkpoint synthesis.

    The evaluator is deliberately skeptical: it assumes criteria are **not**
    met unless the evidence in the agent's whiteboard entries clearly
    demonstrates completion.

    Parameters
    ----------
    meta_plan : MetaPlan
        The session's plan.  Milestone ``merge_criteria`` and plan-level
        ``success_criteria`` are used as the evaluation rubric.
    backend_registry : optional
        LLM backend for criterion-level reasoning.  Falls back to
        heuristic keyword analysis when unavailable.

    Attributes
    ----------
    _history : List[Dict]
        Running log of evaluation results with divergence notes.
        Consumed by ``review_evaluation_history()`` for the P3 tuning loop.
    _criteria_prompt_overrides : Dict[str, str]
        Maps ``"<agent_id>:<criterion>"`` to a custom evaluation prompt
        fragment.  Injected by ``update_criteria_prompt()``.
    """

    def __init__(
        self,
        meta_plan: "MetaPlan",
        backend_registry=None,
    ) -> None:
        self._plan = meta_plan
        self._backend = backend_registry
        self._history: List[Dict[str, Any]] = []
        self._criteria_prompt_overrides: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_agent_work(
        self,
        agent_id: str,
        checkpoint_entries: List["WhiteboardEntry"],
        milestone: "Milestone",
    ) -> EvaluationResult:
        """
        Evaluate one agent's checkpoint work against a milestone's criteria.

        The evaluation checks:
          - Each ``milestone.merge_criteria`` item
          - Each ``meta_plan.success_criteria`` item (global)
          - Heuristic quality signals (entry count, surprise scores)

        The evaluator is **skeptical**: it defaults to "not met" and only
        flips to "met" when evidence is found.

        Parameters
        ----------
        agent_id : str
            The agent being evaluated.
        checkpoint_entries : List[WhiteboardEntry]
            All whiteboard entries attributed to this agent at this checkpoint.
        milestone : Milestone
            The milestone the agent was targeting.

        Returns
        -------
        EvaluationResult
        """
        milestone_index = self._milestone_index(milestone)
        logger.info(
            "CheckpointEvaluator: evaluating %s at milestone %d (%s)",
            agent_id,
            milestone_index,
            milestone.title,
        )

        # Collect all criteria: milestone-specific + global plan-level
        criteria = list(milestone.merge_criteria or [])
        for sc in (self._plan.success_criteria or []):
            if sc not in criteria:
                criteria.append(sc)

        if not criteria:
            # No explicit criteria: do a quality-only assessment
            return self._quality_only_result(agent_id, milestone_index, checkpoint_entries)

        # Attempt LLM evaluation first
        if self._backend is not None:
            try:
                result = await self._llm_evaluate(
                    agent_id, checkpoint_entries, milestone, milestone_index, criteria
                )
                self._record_history(result)
                return result
            except Exception as exc:
                logger.warning(
                    "CheckpointEvaluator: LLM evaluation failed (%s); "
                    "falling back to heuristics",
                    exc,
                )

        # Heuristic fallback
        result = self._heuristic_evaluate(
            agent_id, checkpoint_entries, milestone, milestone_index, criteria
        )
        self._record_history(result)
        return result

    async def evaluate_team(
        self,
        agent_entries: Dict[str, List["WhiteboardEntry"]],
        milestone: "Milestone",
    ) -> EvaluationBatch:
        """
        Evaluate all agents in the team against the same milestone.

        Parameters
        ----------
        agent_entries : Dict[str, List[WhiteboardEntry]]
            Maps agent_id → their checkpoint entries.
        milestone : Milestone
            The shared milestone checkpoint.

        Returns
        -------
        EvaluationBatch
            Aggregated results across all agents.
        """
        milestone_index = self._milestone_index(milestone)
        results: List[EvaluationResult] = []

        for agent_id, entries in agent_entries.items():
            result = await self.evaluate_agent_work(agent_id, entries, milestone)
            results.append(result)

        batch = EvaluationBatch(milestone_index=milestone_index, results=results)
        logger.info(
            "CheckpointEvaluator: team eval complete — "
            "%d/%d agents passed (team score=%.2f)",
            len(batch.passed_agents),
            len(results),
            batch.team_quality_score(),
        )
        return batch

    # ------------------------------------------------------------------
    # Tuning hooks (P3 loop)
    # ------------------------------------------------------------------

    def review_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Return the full evaluation history with divergence notes.

        Used by the P3 tuning loop to identify systematic evaluator errors
        and criterion prompts that need refinement.

        Returns
        -------
        List[Dict]
            Each entry contains: agent_id, milestone_index, criteria_met,
            quality_score, issues_found, confidence, divergence_notes,
            llm_assisted, evaluated_at.
        """
        return list(self._history)

    def update_criteria_prompt(
        self,
        agent_id: str,
        criterion: str,
        new_prompt: str,
    ) -> None:
        """
        Override the evaluation prompt fragment for a specific (agent, criterion) pair.

        Used by human reviewers to tune evaluator behaviour when divergence
        notes reveal systematic errors.

        Parameters
        ----------
        agent_id : str
            The agent whose criterion prompt is being adjusted.
            Use ``"*"`` to apply globally to all agents.
        criterion : str
            The criterion string (must match exactly what's in the milestone).
        new_prompt : str
            Replacement prompt fragment that guides the LLM's assessment.
        """
        key = f"{agent_id}:{criterion}"
        self._criteria_prompt_overrides[key] = new_prompt
        logger.info(
            "CheckpointEvaluator: updated criterion prompt for %s | %r",
            agent_id,
            criterion[:60],
        )

    def clear_criteria_overrides(self, agent_id: Optional[str] = None) -> None:
        """
        Clear prompt overrides for a specific agent (or all agents if None).

        Parameters
        ----------
        agent_id : str, optional
            If given, only clears overrides for this agent.  If None, clears all.
        """
        if agent_id is None:
            self._criteria_prompt_overrides.clear()
        else:
            keys_to_remove = [
                k for k in self._criteria_prompt_overrides if k.startswith(f"{agent_id}:")
            ]
            for k in keys_to_remove:
                del self._criteria_prompt_overrides[k]

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    async def _llm_evaluate(
        self,
        agent_id: str,
        entries: List["WhiteboardEntry"],
        milestone: "Milestone",
        milestone_index: int,
        criteria: List[str],
    ) -> EvaluationResult:
        """Score all criteria using the LLM backend."""
        entries_block = self._format_entries_block(entries)
        criteria_block = self._format_criteria_block(agent_id, criteria)
        plan_block = (
            f"Project objective: {self._plan.objective}\n"
            f"Milestone {milestone_index}: {milestone.title} — {milestone.description}"
        )

        prompt = (
            "You are NWTN's Checkpoint Evaluator, an ADVERSARIAL quality reviewer.\n"
            "Your default assumption is that work is INCOMPLETE until proven otherwise.\n"
            "You are NOT looking to give the benefit of the doubt — you are looking for gaps.\n\n"
            f"{plan_block}\n\n"
            f"Agent under review: {agent_id}\n\n"
            "Agent's whiteboard entries:\n"
            f"{entries_block}\n\n"
            "Acceptance criteria to evaluate:\n"
            f"{criteria_block}\n\n"
            "For each criterion:\n"
            "  1. Search the agent's entries for EXPLICIT evidence of completion.\n"
            "  2. Mark as MET only if the evidence is unambiguous.\n"
            "  3. If the agent CLAIMS completion but evidence is weak or absent, "
            "mark as NOT MET and note the divergence.\n\n"
            "Also assess overall quality:\n"
            "  - Is the work complete? Are edge cases handled?\n"
            "  - Are there contradictions in the entries?\n"
            "  - What is missing?\n\n"
            "Return ONLY a JSON object with:\n"
            "  criteria_met (object): {criterion_text: true/false}\n"
            "  quality_score (float 0-1): overall quality assessment\n"
            "  issues_found (list of strings): specific gaps or problems\n"
            "  confidence (float 0-1): how confident you are in this evaluation\n"
            "  divergence_notes (string): where agent self-assessment differs from "
            "your evaluation (or empty string)\n"
        )

        result = await self._backend.generate(
            prompt=prompt,
            max_tokens=600,
            temperature=0.1,  # Low temperature: we want consistent adversarial judgment
        )

        from prsm.compute.nwtn.team.interview import _extract_json_object
        data = _extract_json_object(result.text.strip()) or {}

        # Parse criteria_met — map raw keys back to canonical criterion strings
        raw_criteria_met: Dict[str, Any] = data.get("criteria_met", {})
        criteria_met = self._map_criteria_results(criteria, raw_criteria_met)

        quality_score = float(data.get("quality_score", self._score_from_criteria(criteria_met)))
        issues_found: List[str] = [str(i) for i in data.get("issues_found", [])]
        confidence = float(data.get("confidence", 0.7))
        divergence_notes = str(data.get("divergence_notes", ""))

        # If quality is suspiciously high but criteria aren't all met, flag divergence
        if quality_score > 0.8 and not all(criteria_met.values()):
            unmet = [c for c, met in criteria_met.items() if not met]
            divergence_notes = (
                divergence_notes
                + f" [Evaluator note: quality_score={quality_score:.2f} "
                f"but {len(unmet)} criteria unmet: {unmet[:2]}]"
            ).strip()

        return EvaluationResult(
            agent_id=agent_id,
            milestone_index=milestone_index,
            criteria_met=criteria_met,
            quality_score=quality_score,
            issues_found=issues_found,
            confidence=confidence,
            divergence_notes=divergence_notes,
            llm_assisted=True,
        )

    # ------------------------------------------------------------------
    # Heuristic evaluation (no LLM)
    # ------------------------------------------------------------------

    def _heuristic_evaluate(
        self,
        agent_id: str,
        entries: List["WhiteboardEntry"],
        milestone: "Milestone",
        milestone_index: int,
        criteria: List[str],
    ) -> EvaluationResult:
        """
        Rule-based evaluation fallback.

        For each criterion, extract keywords and check whether they appear
        in the agent's entries.  Skeptical default: requires at least two
        keyword matches to mark a criterion as met.
        """
        issues: List[str] = []
        criteria_met: Dict[str, bool] = {}
        divergences: List[str] = []

        # Combine all entry text for scanning
        entry_text_lower = " ".join(e.chunk for e in entries).lower()
        entry_claim_words = self._extract_claim_words(entries)

        for criterion in criteria:
            keywords = self._extract_criterion_keywords(criterion)
            match_count = sum(1 for kw in keywords if kw in entry_text_lower)

            # Skeptical threshold: need ≥ 2 keyword hits (or 1 if criterion is short)
            required_hits = 1 if len(keywords) <= 2 else 2
            met = match_count >= required_hits

            criteria_met[criterion] = met

            if not met:
                issues.append(
                    f"Criterion not clearly demonstrated: '{criterion[:80]}' "
                    f"(matched {match_count}/{len(keywords)} keywords)"
                )

            # Divergence: agent claims completion but we don't see evidence
            claim_overlap = sum(1 for kw in keywords if kw in entry_claim_words)
            if claim_overlap >= 1 and not met:
                divergences.append(
                    f"Agent may claim criterion '{criterion[:60]}' but evidence "
                    f"in entries is insufficient (keyword coverage: {match_count}/{len(keywords)})"
                )

        # Entry-count quality signal
        if len(entries) == 0:
            issues.append("Agent has no whiteboard entries at checkpoint — no work evidenced")
            quality_score = 0.0
        elif len(entries) < 2:
            issues.append("Very few entries at checkpoint — work may be incomplete")
            quality_score = self._score_from_criteria(criteria_met) * 0.7
        else:
            # Base score from criteria + bonus for high-surprise entries
            base = self._score_from_criteria(criteria_met)
            avg_surprise = sum(e.surprise_score for e in entries) / len(entries)
            quality_score = min(1.0, base + avg_surprise * 0.1)

        # Check for contradictions (simplistic: look for negation patterns)
        contradiction = self._detect_contradiction_signals(entries)
        if contradiction:
            issues.append(f"Potential contradiction detected in entries: {contradiction}")

        return EvaluationResult(
            agent_id=agent_id,
            milestone_index=milestone_index,
            criteria_met=criteria_met,
            quality_score=quality_score,
            issues_found=issues,
            confidence=_HEURISTIC_CONFIDENCE,
            divergence_notes="; ".join(divergences),
            llm_assisted=False,
        )

    def _quality_only_result(
        self,
        agent_id: str,
        milestone_index: int,
        entries: List["WhiteboardEntry"],
    ) -> EvaluationResult:
        """Produce a quality score when no explicit criteria exist."""
        issues: List[str] = []

        if not entries:
            return EvaluationResult(
                agent_id=agent_id,
                milestone_index=milestone_index,
                criteria_met={},
                quality_score=0.0,
                issues_found=["No whiteboard entries — work not evidenced"],
                confidence=_HEURISTIC_CONFIDENCE,
                divergence_notes="",
                llm_assisted=False,
            )

        avg_surprise = sum(e.surprise_score for e in entries) / len(entries)
        quality_score = min(1.0, 0.5 + avg_surprise * 0.5)

        if len(entries) < 3:
            issues.append("Few entries — work completeness is unclear")
            quality_score *= 0.8

        return EvaluationResult(
            agent_id=agent_id,
            milestone_index=milestone_index,
            criteria_met={},
            quality_score=quality_score,
            issues_found=issues,
            confidence=_HEURISTIC_CONFIDENCE,
            divergence_notes="",
            llm_assisted=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _milestone_index(self, milestone: "Milestone") -> int:
        """Look up a milestone's index in the plan."""
        for i, m in enumerate(self._plan.milestones):
            if m.title == milestone.title:
                return i
        return 0

    def _format_entries_block(self, entries: List["WhiteboardEntry"]) -> str:
        """Render entries as a numbered list for the LLM prompt."""
        if not entries:
            return "(no entries)"
        lines = []
        for i, e in enumerate(entries[:30], 1):  # Cap at 30 to keep prompt manageable
            lines.append(
                f"{i}. [{e.agent_short}] (surprise={e.surprise_score:.2f}) {e.chunk}"
            )
        return "\n".join(lines)

    def _format_criteria_block(self, agent_id: str, criteria: List[str]) -> str:
        """Render criteria with any custom prompt overrides injected."""
        lines = []
        for i, criterion in enumerate(criteria, 1):
            override = (
                self._criteria_prompt_overrides.get(f"{agent_id}:{criterion}")
                or self._criteria_prompt_overrides.get(f"*:{criterion}")
            )
            if override:
                lines.append(f"{i}. {criterion}  [Evaluator note: {override}]")
            else:
                lines.append(f"{i}. {criterion}")
        return "\n".join(lines)

    def _extract_criterion_keywords(self, criterion: str) -> List[str]:
        """Extract significant words (≥4 chars, not common stop words) from a criterion."""
        stop_words = {
            "that", "this", "with", "from", "have", "been", "must", "should",
            "will", "each", "when", "then", "than", "also", "into", "only",
            "both", "some", "such", "more", "other", "above", "below",
        }
        words = re.findall(r"\b[a-z]{4,}\b", criterion.lower())
        return [w for w in words if w not in stop_words]

    def _extract_claim_words(self, entries: List["WhiteboardEntry"]) -> set:
        """
        Extract words associated with completion claims in entry text.

        Looks for phrases like "completed", "done", "finished", "implemented",
        "tested", "verified", "deployed".
        """
        claim_patterns = [
            r"\b(complet|finish|done|implement|test|verif|deploy|resolv|fix|add|creat)\w*\b"
        ]
        words: set = set()
        for entry in entries:
            for pattern in claim_patterns:
                matches = re.findall(pattern, entry.chunk.lower())
                words.update(matches)
        return words

    def _detect_contradiction_signals(self, entries: List["WhiteboardEntry"]) -> Optional[str]:
        """
        Simple heuristic: look for negation words near key claims.

        Returns a brief description if a potential contradiction is found.
        """
        negation_re = re.compile(
            r"\b(not|never|cannot|won'?t|doesn'?t|didn'?t|no longer|broken|fails?|incorrect)\b",
            re.IGNORECASE,
        )
        for entry in entries:
            if negation_re.search(entry.chunk):
                return f"Entry #{entry.id} contains negation pattern"
        return None

    def _score_from_criteria(self, criteria_met: Dict[str, bool]) -> float:
        """Compute a quality score from the fraction of criteria met."""
        if not criteria_met:
            return 0.5  # No criteria: neutral
        met = sum(1 for v in criteria_met.values() if v)
        return met / len(criteria_met)

    def _map_criteria_results(
        self,
        canonical_criteria: List[str],
        raw: Dict[str, Any],
    ) -> Dict[str, bool]:
        """
        Map LLM-returned criterion results back to canonical criterion strings.

        The LLM may truncate or paraphrase keys; we do a best-effort fuzzy match.
        """
        result: Dict[str, bool] = {c: False for c in canonical_criteria}

        for raw_key, raw_value in raw.items():
            raw_lower = raw_key.lower().strip()
            # Try exact match first
            for criterion in canonical_criteria:
                if criterion.lower() == raw_lower:
                    result[criterion] = bool(raw_value)
                    break
            else:
                # Fuzzy: find the canonical criterion with the most word overlap
                best_match: Optional[str] = None
                best_overlap = 0
                raw_words = set(re.findall(r"\b\w{4,}\b", raw_lower))
                for criterion in canonical_criteria:
                    crit_words = set(re.findall(r"\b\w{4,}\b", criterion.lower()))
                    overlap = len(raw_words & crit_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = criterion
                if best_match and best_overlap >= 1:
                    result[best_match] = bool(raw_value)

        return result

    def _record_history(self, result: EvaluationResult) -> None:
        """Append an evaluation result to the history log."""
        self._history.append({
            "agent_id": result.agent_id,
            "milestone_index": result.milestone_index,
            "criteria_met": dict(result.criteria_met),
            "quality_score": result.quality_score,
            "issues_found": list(result.issues_found),
            "confidence": result.confidence,
            "divergence_notes": result.divergence_notes,
            "llm_assisted": result.llm_assisted,
            "evaluated_at": result.evaluated_at.isoformat(),
        })


# ======================================================================
# Exports
# ======================================================================

__all__ = [
    "EvaluationResult",
    "EvaluationBatch",
    "CheckpointEvaluator",
]
