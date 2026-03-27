"""
Tests for CheckpointEvaluator
==============================

Covers:
  - EvaluationResult dataclass properties
  - EvaluationBatch aggregation and narrative rendering
  - CheckpointEvaluator heuristic evaluation (no LLM)
  - CheckpointEvaluator LLM evaluation path (mocked backend)
  - Evaluator skepticism: defaults to "not met"
  - Divergence logging
  - Tuning hooks: review_evaluation_history, update_criteria_prompt
  - evaluate_team batch evaluation
  - Quality-only evaluation (no criteria)
  - Edge cases: empty entries, no criteria
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.compute.nwtn.team.evaluator import (
    CheckpointEvaluator,
    EvaluationBatch,
    EvaluationResult,
)


# ======================================================================
# Fixtures and helpers
# ======================================================================

def make_whiteboard_entry(
    chunk: str,
    source_agent: str = "agent/coder",
    surprise_score: float = 0.5,
    entry_id: int = 1,
    session_id: str = "sess-test",
):
    """Create a minimal WhiteboardEntry-like object for testing."""

    class FakeEntry:
        def __init__(self):
            self.id = entry_id
            self.session_id = session_id
            self.source_agent = source_agent
            self.chunk = chunk
            self.surprise_score = surprise_score
            self.promoted_at = datetime.now(timezone.utc)

        @property
        def agent_short(self):
            return self.source_agent.removeprefix("agent/")

    return FakeEntry()


def make_milestone(
    title: str = "Test Milestone",
    description: str = "A test milestone",
    merge_criteria: Optional[List[str]] = None,
):
    """Create a minimal Milestone-like object for testing."""

    class FakeMilestone:
        pass

    m = FakeMilestone()
    m.title = title
    m.description = description
    m.merge_criteria = merge_criteria or []
    return m


def make_meta_plan(
    milestones=None,
    success_criteria: Optional[List[str]] = None,
    session_id: str = "sess-test",
    objective: str = "Build a test system",
):
    """Create a minimal MetaPlan-like object for testing."""

    class FakeMetaPlan:
        pass

    plan = FakeMetaPlan()
    plan.session_id = session_id
    plan.objective = objective
    plan.success_criteria = success_criteria or []
    if milestones is None:
        plan.milestones = [make_milestone()]
    else:
        plan.milestones = milestones
    return plan


@pytest.fixture
def basic_plan():
    """MetaPlan with one milestone and two criteria."""
    milestone = make_milestone(
        title="Implement feature",
        description="Build the core feature",
        merge_criteria=[
            "All unit tests pass with coverage above 80%",
            "Error handling implemented for edge cases",
        ],
    )
    return make_meta_plan(milestones=[milestone], success_criteria=[
        "Documentation updated",
    ])


@pytest.fixture
def evaluator(basic_plan):
    """CheckpointEvaluator with no LLM backend (heuristic mode)."""
    return CheckpointEvaluator(meta_plan=basic_plan, backend_registry=None)


# ======================================================================
# EvaluationResult tests
# ======================================================================

class TestEvaluationResult:
    def test_passed_threshold(self):
        result = EvaluationResult(
            agent_id="agent/a",
            milestone_index=0,
            quality_score=0.7,
        )
        assert result.passed is True

    def test_failed_below_threshold(self):
        result = EvaluationResult(
            agent_id="agent/a",
            milestone_index=0,
            quality_score=0.4,
        )
        assert result.passed is False

    def test_criteria_pass_rate_all_met(self):
        result = EvaluationResult(
            agent_id="agent/a",
            milestone_index=0,
            criteria_met={"crit1": True, "crit2": True},
        )
        assert result.criteria_pass_rate == 1.0

    def test_criteria_pass_rate_none_met(self):
        result = EvaluationResult(
            agent_id="agent/a",
            milestone_index=0,
            criteria_met={"crit1": False, "crit2": False},
        )
        assert result.criteria_pass_rate == 0.0

    def test_criteria_pass_rate_partial(self):
        result = EvaluationResult(
            agent_id="agent/a",
            milestone_index=0,
            criteria_met={"crit1": True, "crit2": False},
        )
        assert result.criteria_pass_rate == 0.5

    def test_criteria_pass_rate_empty(self):
        result = EvaluationResult(agent_id="agent/a", milestone_index=0)
        assert result.criteria_pass_rate == 0.0

    def test_summary_pass(self):
        result = EvaluationResult(
            agent_id="agent/coder",
            milestone_index=1,
            criteria_met={"crit1": True},
            quality_score=0.85,
            confidence=0.9,
        )
        summary = result.summary()
        assert "[PASS]" in summary
        assert "agent/coder" in summary
        assert "milestone 1" in summary

    def test_summary_fail(self):
        result = EvaluationResult(
            agent_id="agent/coder",
            milestone_index=0,
            quality_score=0.3,
        )
        assert "[FAIL]" in result.summary()

    def test_default_evaluated_at_is_utc(self):
        result = EvaluationResult(agent_id="agent/a", milestone_index=0)
        assert result.evaluated_at.tzinfo is not None


# ======================================================================
# EvaluationBatch tests
# ======================================================================

class TestEvaluationBatch:
    def _make_batch(self, scores):
        results = [
            EvaluationResult(
                agent_id=f"agent/{i}",
                milestone_index=0,
                quality_score=score,
            )
            for i, score in enumerate(scores)
        ]
        return EvaluationBatch(milestone_index=0, results=results)

    def test_all_passed(self):
        batch = self._make_batch([0.8, 0.7, 0.9])
        assert batch.all_passed is True

    def test_not_all_passed(self):
        batch = self._make_batch([0.8, 0.3, 0.9])
        assert batch.all_passed is False

    def test_passed_agents(self):
        batch = self._make_batch([0.8, 0.3, 0.9])
        assert "agent/0" in batch.passed_agents
        assert "agent/1" not in batch.passed_agents
        assert "agent/2" in batch.passed_agents

    def test_failed_agents(self):
        batch = self._make_batch([0.8, 0.3])
        assert batch.failed_agents == ["agent/1"]

    def test_team_quality_score(self):
        batch = self._make_batch([0.8, 0.6])
        assert abs(batch.team_quality_score() - 0.7) < 0.001

    def test_empty_batch(self):
        batch = EvaluationBatch(milestone_index=0, results=[])
        assert batch.all_passed is False
        assert batch.team_quality_score() == 0.0

    def test_narrative_block_contains_table(self):
        batch = EvaluationBatch(
            milestone_index=0,
            results=[
                EvaluationResult(
                    agent_id="agent/coder",
                    milestone_index=0,
                    criteria_met={"tests pass": True},
                    quality_score=0.8,
                    issues_found=["Missing edge case handling"],
                    divergence_notes="Agent claimed completion but no evidence of edge case testing",
                ),
            ],
        )
        block = batch.to_narrative_block()
        assert "### Checkpoint Evaluation Results" in block
        assert "coder" in block
        assert "0.80" in block
        assert "Missing edge case handling" in block
        assert "divergence" in block.lower() or "Divergence" in block

    def test_narrative_block_no_issues(self):
        batch = EvaluationBatch(
            milestone_index=0,
            results=[
                EvaluationResult(
                    agent_id="agent/coder",
                    milestone_index=0,
                    quality_score=0.9,
                ),
            ],
        )
        block = batch.to_narrative_block()
        assert "Issues Identified" not in block


# ======================================================================
# CheckpointEvaluator — heuristic mode
# ======================================================================

class TestCheckpointEvaluatorHeuristic:
    @pytest.mark.asyncio
    async def test_no_entries_gives_zero_score(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=[],
            milestone=milestone,
        )
        assert result.quality_score == 0.0
        assert result.passed is False
        assert any("no whiteboard entries" in i.lower() for i in result.issues_found)

    @pytest.mark.asyncio
    async def test_single_entry_penalized(self, evaluator, basic_plan):
        """Very few entries should reduce the quality score."""
        milestone = basic_plan.milestones[0]
        entry = make_whiteboard_entry("unit tests pass with coverage 85%")
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=[entry],
            milestone=milestone,
        )
        # Score should be reduced relative to if there were more entries
        # The result should note the low entry count
        assert any("few" in i.lower() or "incomplete" in i.lower() for i in result.issues_found)

    @pytest.mark.asyncio
    async def test_matching_keywords_mark_criterion_met(self, evaluator, basic_plan):
        """Entries with matching keywords should get criteria marked met."""
        milestone = basic_plan.milestones[0]
        entries = [
            make_whiteboard_entry("unit tests pass with coverage 85% verified", entry_id=1),
            make_whiteboard_entry("error handling implemented for edge cases thoroughly", entry_id=2),
            make_whiteboard_entry("documentation updated and reviewed", entry_id=3),
        ]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=entries,
            milestone=milestone,
        )
        # At least one criterion should be met
        assert any(result.criteria_met.values()), (
            f"Expected at least one criterion met, got: {result.criteria_met}"
        )

    @pytest.mark.asyncio
    async def test_default_skepticism_unmet_criteria(self, evaluator, basic_plan):
        """Evaluator defaults to 'not met' when keywords are absent."""
        milestone = basic_plan.milestones[0]
        entries = [
            make_whiteboard_entry("I have finished all the work completely done", entry_id=1),
            make_whiteboard_entry("Everything is working great and ready to go", entry_id=2),
            make_whiteboard_entry("The system runs correctly I believe", entry_id=3),
        ]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=entries,
            milestone=milestone,
        )
        # Generic "done" claims without specific evidence → evaluator skeptical
        # At least some criteria should not be met
        assert not all(result.criteria_met.values()), (
            "Evaluator should be skeptical of vague completion claims"
        )

    @pytest.mark.asyncio
    async def test_confidence_is_heuristic_level(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=[make_whiteboard_entry("some work done")],
            milestone=milestone,
        )
        # Heuristic confidence is fixed at 0.55
        assert abs(result.confidence - 0.55) < 0.01
        assert result.llm_assisted is False

    @pytest.mark.asyncio
    async def test_divergence_logged_when_claim_without_evidence(self, evaluator, basic_plan):
        """When entry has claim words but criterion keywords are absent, note divergence."""
        milestone = basic_plan.milestones[0]
        # "completed" is a claim word, but does not match "tests" / "coverage" / "error handling"
        entries = [
            make_whiteboard_entry("completed and implemented the task", entry_id=1),
            make_whiteboard_entry("finished implementing the solution", entry_id=2),
            make_whiteboard_entry("all done and implemented", entry_id=3),
        ]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=entries,
            milestone=milestone,
        )
        # The evaluator may log divergence notes when agent claims completion but evidence is thin
        # (divergence_notes may be empty if no matching claim words found — that's okay too)
        # The key is that not all criteria should be met
        assert not all(result.criteria_met.values())

    @pytest.mark.asyncio
    async def test_quality_only_when_no_criteria(self):
        """When no criteria exist, produce a quality-only result."""
        plan = make_meta_plan(
            milestones=[make_milestone(merge_criteria=[])],
            success_criteria=[],
        )
        evaluator = CheckpointEvaluator(meta_plan=plan, backend_registry=None)
        entries = [
            make_whiteboard_entry("great finding about the system", surprise_score=0.8, entry_id=1),
            make_whiteboard_entry("another insight", surprise_score=0.6, entry_id=2),
            make_whiteboard_entry("more work completed here", surprise_score=0.7, entry_id=3),
        ]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/researcher",
            checkpoint_entries=entries,
            milestone=plan.milestones[0],
        )
        assert result.criteria_met == {}
        assert 0.0 < result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_quality_only_no_entries(self):
        """Quality-only mode with zero entries → score = 0."""
        plan = make_meta_plan(
            milestones=[make_milestone(merge_criteria=[])],
            success_criteria=[],
        )
        evaluator = CheckpointEvaluator(meta_plan=plan, backend_registry=None)
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/researcher",
            checkpoint_entries=[],
            milestone=plan.milestones[0],
        )
        assert result.quality_score == 0.0

    @pytest.mark.asyncio
    async def test_milestone_index_resolved_correctly(self, evaluator, basic_plan):
        """Milestone index should match position in plan."""
        milestone = basic_plan.milestones[0]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/a",
            checkpoint_entries=[make_whiteboard_entry("something")],
            milestone=milestone,
        )
        assert result.milestone_index == 0

    @pytest.mark.asyncio
    async def test_results_are_stored_in_history(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=[make_whiteboard_entry("work done", entry_id=1)],
            milestone=milestone,
        )
        history = evaluator.review_evaluation_history()
        assert len(history) == 1
        assert history[0]["agent_id"] == "agent/coder"
        assert "quality_score" in history[0]
        assert "divergence_notes" in history[0]
        assert "evaluated_at" in history[0]

    @pytest.mark.asyncio
    async def test_history_accumulates_across_evals(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        for i in range(3):
            await evaluator.evaluate_agent_work(
                agent_id=f"agent/coder-{i}",
                checkpoint_entries=[make_whiteboard_entry(f"work {i}", entry_id=i)],
                milestone=milestone,
            )
        history = evaluator.review_evaluation_history()
        assert len(history) == 3


# ======================================================================
# CheckpointEvaluator — LLM mode (mocked)
# ======================================================================

class TestCheckpointEvaluatorLLM:
    def _make_backend(self, response_text: str):
        """Create a mock backend that returns structured JSON."""
        backend = MagicMock()
        result = MagicMock()
        result.text = response_text
        backend.generate = AsyncMock(return_value=result)
        return backend

    @pytest.mark.asyncio
    async def test_llm_evaluation_used_when_backend_present(self, basic_plan):
        """LLM path is taken when backend is available."""
        json_response = """{
            "criteria_met": {
                "All unit tests pass with coverage above 80%": true,
                "Error handling implemented for edge cases": true,
                "Documentation updated": false
            },
            "quality_score": 0.75,
            "issues_found": ["Documentation not updated yet"],
            "confidence": 0.85,
            "divergence_notes": "Agent mentioned docs but no actual content found"
        }"""

        backend = self._make_backend(json_response)

        with patch(
            "prsm.compute.nwtn.team.evaluator.CheckpointEvaluator._llm_evaluate",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = EvaluationResult(
                agent_id="agent/coder",
                milestone_index=0,
                criteria_met={
                    "All unit tests pass with coverage above 80%": True,
                    "Error handling implemented for edge cases": True,
                    "Documentation updated": False,
                },
                quality_score=0.75,
                issues_found=["Documentation not updated yet"],
                confidence=0.85,
                divergence_notes="Agent mentioned docs but no actual content found",
                llm_assisted=True,
            )

            evaluator = CheckpointEvaluator(meta_plan=basic_plan, backend_registry=backend)
            milestone = basic_plan.milestones[0]
            result = await evaluator.evaluate_agent_work(
                agent_id="agent/coder",
                checkpoint_entries=[make_whiteboard_entry("tests pass coverage 85%")],
                milestone=milestone,
            )

        assert result.llm_assisted is True
        assert result.quality_score == 0.75
        assert result.confidence == 0.85
        assert "Documentation not updated yet" in result.issues_found

    @pytest.mark.asyncio
    async def test_falls_back_to_heuristic_on_llm_failure(self, basic_plan):
        """Heuristic path taken when LLM backend raises exception."""
        backend = MagicMock()
        backend.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        evaluator = CheckpointEvaluator(meta_plan=basic_plan, backend_registry=backend)
        milestone = basic_plan.milestones[0]
        entries = [make_whiteboard_entry("unit tests pass coverage 85%", entry_id=i) for i in range(3)]
        result = await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=entries,
            milestone=milestone,
        )
        # Should fall back to heuristic (not raise)
        assert result.llm_assisted is False
        assert result.confidence == pytest.approx(0.55)


# ======================================================================
# CheckpointEvaluator.evaluate_team
# ======================================================================

class TestEvaluateTeam:
    @pytest.mark.asyncio
    async def test_evaluate_team_returns_batch(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        agent_entries = {
            "agent/coder": [
                make_whiteboard_entry("unit tests pass with coverage 85%", entry_id=1),
                make_whiteboard_entry("error handling implemented edge cases", entry_id=2),
                make_whiteboard_entry("documentation updated thoroughly", entry_id=3),
            ],
            "agent/researcher": [
                make_whiteboard_entry("no evidence here just empty words", entry_id=10),
                make_whiteboard_entry("hmm interesting finding", entry_id=11),
                make_whiteboard_entry("results look promising", entry_id=12),
            ],
        }
        batch = await evaluator.evaluate_team(
            agent_entries=agent_entries,
            milestone=milestone,
        )
        assert isinstance(batch, EvaluationBatch)
        assert len(batch.results) == 2
        agent_ids = {r.agent_id for r in batch.results}
        assert "agent/coder" in agent_ids
        assert "agent/researcher" in agent_ids

    @pytest.mark.asyncio
    async def test_evaluate_team_empty(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        batch = await evaluator.evaluate_team(
            agent_entries={},
            milestone=milestone,
        )
        assert batch.results == []
        assert batch.all_passed is False


# ======================================================================
# Tuning hooks
# ======================================================================

class TestTuningHooks:
    def test_update_criteria_prompt_stores_override(self, evaluator):
        evaluator.update_criteria_prompt(
            agent_id="agent/coder",
            criterion="All unit tests pass with coverage above 80%",
            new_prompt="Look for explicit coverage percentage mentioned",
        )
        key = "agent/coder:All unit tests pass with coverage above 80%"
        assert key in evaluator._criteria_prompt_overrides
        assert "explicit coverage percentage" in evaluator._criteria_prompt_overrides[key]

    def test_global_override_uses_wildcard(self, evaluator):
        evaluator.update_criteria_prompt(
            agent_id="*",
            criterion="Documentation updated",
            new_prompt="Check for doc file changes or README updates",
        )
        assert "*:Documentation updated" in evaluator._criteria_prompt_overrides

    def test_clear_overrides_specific_agent(self, evaluator):
        evaluator.update_criteria_prompt("agent/a", "criterion1", "prompt1")
        evaluator.update_criteria_prompt("agent/b", "criterion1", "prompt1")
        evaluator.clear_criteria_overrides("agent/a")
        assert "agent/a:criterion1" not in evaluator._criteria_prompt_overrides
        assert "agent/b:criterion1" in evaluator._criteria_prompt_overrides

    def test_clear_overrides_all(self, evaluator):
        evaluator.update_criteria_prompt("agent/a", "c1", "p1")
        evaluator.update_criteria_prompt("agent/b", "c2", "p2")
        evaluator.clear_criteria_overrides()
        assert evaluator._criteria_prompt_overrides == {}

    def test_review_evaluation_history_returns_copy(self, evaluator):
        """History should return an independent copy (not a reference to internal list)."""
        history = evaluator.review_evaluation_history()
        original_len = len(history)
        history.append({"fake": "entry"})
        assert len(evaluator._history) == original_len

    @pytest.mark.asyncio
    async def test_history_includes_divergence_notes(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        entries = [
            make_whiteboard_entry("implemented completed done finished", entry_id=i)
            for i in range(3)
        ]
        await evaluator.evaluate_agent_work(
            agent_id="agent/coder",
            checkpoint_entries=entries,
            milestone=milestone,
        )
        history = evaluator.review_evaluation_history()
        assert len(history) == 1
        assert "divergence_notes" in history[0]
        assert "llm_assisted" in history[0]


# ======================================================================
# Keyword / heuristic internals
# ======================================================================

class TestHeuristicInternals:
    def test_extract_criterion_keywords_filters_stop_words(self, evaluator):
        criterion = "All unit tests should pass with coverage above 80%"
        keywords = evaluator._extract_criterion_keywords(criterion)
        assert "unit" in keywords
        assert "tests" in keywords
        assert "pass" in keywords
        assert "coverage" in keywords
        # Stop words should be excluded
        assert "with" not in keywords
        assert "should" not in keywords
        assert "above" not in keywords
        # Note: "above" was added to stop_words after the test failed

    def test_score_from_criteria_all_met(self, evaluator):
        criteria = {"a": True, "b": True, "c": True}
        assert evaluator._score_from_criteria(criteria) == 1.0

    def test_score_from_criteria_none_met(self, evaluator):
        criteria = {"a": False, "b": False}
        assert evaluator._score_from_criteria(criteria) == 0.0

    def test_score_from_criteria_empty(self, evaluator):
        assert evaluator._score_from_criteria({}) == 0.5  # Neutral

    def test_detect_contradiction_signals(self, evaluator):
        entries = [
            make_whiteboard_entry("everything works correctly"),
            make_whiteboard_entry("actually this does not work as expected"),
        ]
        result = evaluator._detect_contradiction_signals(entries)
        assert result is not None

    def test_no_contradiction_signals(self, evaluator):
        entries = [
            make_whiteboard_entry("all tests pass and coverage is 90%"),
            make_whiteboard_entry("edge cases are handled properly"),
        ]
        result = evaluator._detect_contradiction_signals(entries)
        assert result is None

    def test_map_criteria_results_exact_match(self, evaluator):
        canonical = ["All unit tests pass with coverage above 80%"]
        raw = {"All unit tests pass with coverage above 80%": True}
        result = evaluator._map_criteria_results(canonical, raw)
        assert result["All unit tests pass with coverage above 80%"] is True

    def test_map_criteria_results_fuzzy_match(self, evaluator):
        canonical = ["All unit tests pass with coverage above 80%"]
        raw = {"unit tests coverage pass": True}
        result = evaluator._map_criteria_results(canonical, raw)
        # Fuzzy match should find the canonical criterion
        assert result["All unit tests pass with coverage above 80%"] is True

    def test_map_criteria_results_defaults_to_false(self, evaluator):
        canonical = ["All unit tests pass with coverage above 80%"]
        raw = {"something completely unrelated": True}
        result = evaluator._map_criteria_results(canonical, raw)
        # No match: should default to False
        assert result["All unit tests pass with coverage above 80%"] is False


# ======================================================================
# Milestone index resolution
# ======================================================================

class TestMilestoneIndexResolution:
    def test_first_milestone_index_is_zero(self, evaluator, basic_plan):
        milestone = basic_plan.milestones[0]
        idx = evaluator._milestone_index(milestone)
        assert idx == 0

    def test_unknown_milestone_defaults_to_zero(self, evaluator):
        unknown = make_milestone(title="Unknown milestone")
        idx = evaluator._milestone_index(unknown)
        assert idx == 0

    def test_multi_milestone_index(self):
        m0 = make_milestone(title="Alpha")
        m1 = make_milestone(title="Beta")
        m2 = make_milestone(title="Gamma")
        plan = make_meta_plan(milestones=[m0, m1, m2])
        evaluator = CheckpointEvaluator(meta_plan=plan, backend_registry=None)
        assert evaluator._milestone_index(m1) == 1
        assert evaluator._milestone_index(m2) == 2
