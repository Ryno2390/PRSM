"""
Tests for EvaluatorFeedbackPublisher
====================================

Covers:
  - format_feedback_message() includes required sections
  - publish_feedback() skips agents that don't need feedback
  - publish_feedback() publishes to agents with issues
  - publish_feedback() publishes to agents with low quality scores
  - publish_feedback() publishes to agents with unmet criteria
  - publish_feedback() returns correct count
  - NWTNSession.run() backward compatibility (no feedback_publisher)
  - feedback_rounds_completed increments correctly
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.compute.nwtn.team.evaluator import EvaluationResult, EvaluationBatch
from prsm.compute.nwtn.team.feedback_publisher import EvaluatorFeedbackPublisher
from prsm.compute.nwtn.bsc.event_bus import EventBus, EventType, BSCEvent


# ======================================================================
# Fixtures and helpers
# ======================================================================

def make_evaluation_result(
    agent_id: str = "agent/coder",
    milestone_index: int = 0,
    quality_score: float = 0.5,
    criteria_met: Dict[str, bool] = None,
    issues_found: List[str] = None,
    confidence: float = 0.8,
    divergence_notes: str = "",
    llm_assisted: bool = True,
) -> EvaluationResult:
    """Create an EvaluationResult for testing."""
    if criteria_met is None:
        criteria_met = {"Complete the feature": False, "Add tests": False}
    if issues_found is None:
        issues_found = []

    return EvaluationResult(
        agent_id=agent_id,
        milestone_index=milestone_index,
        criteria_met=criteria_met,
        quality_score=quality_score,
        issues_found=issues_found,
        confidence=confidence,
        divergence_notes=divergence_notes,
        llm_assisted=llm_assisted,
    )


def make_evaluation_batch(
    milestone_index: int = 0,
    results: List[EvaluationResult] = None,
) -> EvaluationBatch:
    """Create an EvaluationBatch for testing."""
    if results is None:
        results = [make_evaluation_result()]
    return EvaluationBatch(milestone_index=milestone_index, results=results)


# ======================================================================
# Tests: format_feedback_message
# ======================================================================

class TestFormatFeedbackMessage:
    """Tests for format_feedback_message method."""

    def test_includes_quality_score_section(self):
        """Feedback message must include quality score."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(quality_score=0.65)
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "Quality Score" in message
        assert "0.65" in message

    def test_includes_criteria_met_section(self):
        """Feedback message must include criteria met information."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(
            criteria_met={"Feature A": True, "Feature B": False}
        )
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "Criteria Met" in message
        assert "1/2" in message  # 1 met out of 2 total

    def test_includes_issues_found_section(self):
        """Feedback message must include issues found when present."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(
            issues_found=["Missing error handling", "No tests"]
        )
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "Issues Found" in message
        assert "Missing error handling" in message
        assert "No tests" in message

    def test_includes_what_to_address_section(self):
        """Feedback message must include 'What to Address' section."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(quality_score=0.5)
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "What to Address" in message

    def test_includes_round_number(self):
        """Feedback message must include round number."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result()
        
        message = publisher.format_feedback_message(result, round_number=5)
        
        assert "Round 5" in message

    def test_includes_agent_id(self):
        """Feedback message must include agent ID."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(agent_id="agent/architect")
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "agent/architect" in message

    def test_shows_pass_status_for_passing_score(self):
        """Passing quality score shows PASS status."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(quality_score=0.8)
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "PASS" in message


# ======================================================================
# Tests: publish_feedback - skipping logic
# ======================================================================

class TestPublishFeedbackSkipLogic:
    """Tests for publish_feedback skipping agents that don't need feedback."""

    @pytest.mark.asyncio
    async def test_skips_agent_with_high_score_no_issues_all_criteria_met(self):
        """Should skip agents with score >= 0.8, no issues, and all criteria met."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        result = make_evaluation_result(
            agent_id="agent/good",
            quality_score=0.9,
            criteria_met={"Task A": True, "Task B": True},
            issues_found=[],
        )
        batch = make_evaluation_batch(results=[result])
        
        count = await publisher.publish_feedback(batch, round_number=1, session_id="sess-1")
        
        assert count == 0
        event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_multiple_passing_agents(self):
        """Should skip all agents meeting pass criteria."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        results = [
            make_evaluation_result(
                agent_id=f"agent/good-{i}",
                quality_score=0.85,
                criteria_met={"Task": True},
                issues_found=[],
            )
            for i in range(3)
        ]
        batch = make_evaluation_batch(results=results)
        
        count = await publisher.publish_feedback(batch, round_number=1, session_id="sess-1")
        
        assert count == 0
        event_bus.publish.assert_not_called()


# ======================================================================
# Tests: publish_feedback - publishing logic
# ======================================================================

class TestPublishFeedbackPublishing:
    """Tests for publish_feedback publishing to agents that need it."""

    @pytest.mark.asyncio
    async def test_publishes_to_agent_with_issues(self):
        """Should publish feedback to agents with issues found."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        result = make_evaluation_result(
            agent_id="agent/buggy",
            quality_score=0.9,  # High score but has issues
            criteria_met={"Task A": True, "Task B": True},
            issues_found=["Critical bug in error handling"],
        )
        batch = make_evaluation_batch(results=[result])
        
        count = await publisher.publish_feedback(batch, round_number=1, session_id="sess-1")
        
        assert count == 1
        event_bus.publish.assert_called_once()
        
        # Verify the event structure
        call_args = event_bus.publish.call_args
        event = call_args[0][0]
        assert event.event_type == EventType.EVALUATION_FEEDBACK
        assert event.data["agent_id"] == "agent/buggy"
        assert event.data["round_number"] == 1
        assert event.data["quality_score"] == 0.9
        assert "feedback_message" in event.data

    @pytest.mark.asyncio
    async def test_publishes_to_agent_with_low_quality_score(self):
        """Should publish feedback to agents with quality_score < 0.8."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        result = make_evaluation_result(
            agent_id="agent/struggling",
            quality_score=0.45,
            criteria_met={"Task A": True, "Task B": True},
            issues_found=[],
        )
        batch = make_evaluation_batch(results=[result])
        
        count = await publisher.publish_feedback(batch, round_number=2, session_id="sess-2")
        
        assert count == 1
        event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publishes_to_agent_with_unmet_criteria(self):
        """Should publish feedback to agents with unmet criteria."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        result = make_evaluation_result(
            agent_id="agent/incomplete",
            quality_score=0.8,
            criteria_met={"Task A": True, "Task B": False},  # One unmet
            issues_found=[],
        )
        batch = make_evaluation_batch(results=[result])
        
        count = await publisher.publish_feedback(batch, round_number=1, session_id="sess-1")
        
        assert count == 1
        event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_correct_count_mixed_results(self):
        """Should return correct count for mix of passing and failing agents."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        results = [
            # Agent 1: passes all checks
            make_evaluation_result(
                agent_id="agent/perfect",
                quality_score=0.9,
                criteria_met={"Task A": True},
                issues_found=[],
            ),
            # Agent 2: has issues
            make_evaluation_result(
                agent_id="agent/buggy",
                quality_score=0.85,
                criteria_met={"Task A": True},
                issues_found=["Missing tests"],
            ),
            # Agent 3: low score
            make_evaluation_result(
                agent_id="agent/struggling",
                quality_score=0.5,
                criteria_met={"Task A": True},
                issues_found=[],
            ),
        ]
        batch = make_evaluation_batch(results=results)
        
        count = await publisher.publish_feedback(batch, round_number=1, session_id="sess-1")
        
        assert count == 2  # Only agent/buggy and agent/struggling
        assert event_bus.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_payload_includes_criteria_counts(self):
        """Published event should include criteria_met and total_criteria counts."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        result = make_evaluation_result(
            criteria_met={"Task A": True, "Task B": False, "Task C": True},
        )
        batch = make_evaluation_batch(results=[result])
        
        await publisher.publish_feedback(batch, round_number=1, session_id="sess-1")
        
        call_args = event_bus.publish.call_args
        event = call_args[0][0]
        
        assert event.data["criteria_met"] == 2  # 2 met
        assert event.data["total_criteria"] == 3  # 3 total


# ======================================================================
# Tests: NWTNSession integration
# ======================================================================

class TestNWTNSessionIntegration:
    """Tests for NWTNSession.run() with feedback_publisher."""

    @pytest.mark.asyncio
    async def test_run_backward_compatible_without_feedback_publisher(self):
        """NWTNSession.run() should work without feedback_publisher parameter."""
        from prsm.compute.nwtn.session import NWTNSession, RunResult
        from unittest.mock import MagicMock, AsyncMock
        
        # Create a mock adapter and state
        mock_adapter = MagicMock()
        mock_adapter.convergence_summary = MagicMock(return_value={
            "pending_agents": [],
            "whiteboard_entries": [],
            "meta_plan_summary": "",
        })
        mock_adapter.is_session_converged = MagicMock(return_value=True)
        mock_adapter.advance_bsc_round = AsyncMock()
        
        mock_state = MagicMock()
        mock_state.session_id = "test-session"
        mock_state.goal = "Test goal"
        mock_state.status = "completed"
        mock_state.team_members = ["agent/coder"]
        mock_state.scribe_running = True
        
        session = NWTNSession(adapter=mock_adapter, session_state=mock_state)
        
        # Run without feedback_publisher - should not raise
        result = await session.run(max_rounds=1, round_poll_interval=0.01)
        
        assert isinstance(result, RunResult)
        assert result.feedback_rounds_completed == 0

    @pytest.mark.asyncio
    async def test_feedback_rounds_completed_increments_with_publisher(self):
        """feedback_rounds_completed should increment when publisher provided and batch available."""
        from prsm.compute.nwtn.session import NWTNSession
        from unittest.mock import MagicMock, AsyncMock
        
        # Create mock event bus
        mock_event_bus = MagicMock(spec=EventBus)
        mock_event_bus.publish = AsyncMock()
        
        # Create mock feedback publisher
        feedback_publisher = EvaluatorFeedbackPublisher(event_bus=mock_event_bus)
        
        # Create evaluation batch for the convergence summary
        eval_result = make_evaluation_result(
            agent_id="agent/coder",
            quality_score=0.5,
            criteria_met={"Task": False},
            issues_found=["Not done"],
        )
        eval_batch = make_evaluation_batch(results=[eval_result])
        
        # Create mock adapter and state
        mock_adapter = MagicMock()
        mock_adapter.convergence_summary = MagicMock(return_value={
            "pending_agents": [],
            "whiteboard_entries": [],
            "meta_plan_summary": "",
            "evaluation_batch": eval_batch,
        })
        mock_adapter.is_session_converged = MagicMock(return_value=True)
        mock_adapter.advance_bsc_round = AsyncMock()
        
        mock_state = MagicMock()
        mock_state.session_id = "test-session"
        mock_state.goal = "Test goal"
        mock_state.status = "completed"
        mock_state.team_members = ["agent/coder"]
        mock_state.scribe_running = True
        
        session = NWTNSession(adapter=mock_adapter, session_state=mock_state)
        
        result = await session.run(
            max_rounds=1,
            round_poll_interval=0.01,
            feedback_publisher=feedback_publisher,
        )
        
        # Should have published feedback and incremented counter
        assert result.feedback_rounds_completed == 1
        mock_event_bus.publish.assert_called()


# ======================================================================
# Tests: Edge cases
# ======================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_batch_returns_zero(self):
        """Empty batch should return 0 with no publishes."""
        event_bus = MagicMock(spec=EventBus)
        event_bus.publish = AsyncMock()
        publisher = EvaluatorFeedbackPublisher(event_bus=event_bus)
        
        batch = make_evaluation_batch(results=[])
        
        count = await publisher.publish_feedback(batch, round_number=1, session_id="sess-1")
        
        assert count == 0
        event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_divergence_notes(self):
        """Feedback should include divergence notes when present."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(
            divergence_notes="Agent claimed completion but evidence missing"
        )
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "Evaluator Divergence Notes" in message
        assert "Agent claimed completion but evidence missing" in message

    def test_format_feedback_handles_empty_criteria(self):
        """Format should handle empty criteria dict gracefully."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        result = make_evaluation_result(criteria_met={})
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        assert "0/0" in message  # Should show 0 met out of 0 total

    def test_format_feedback_truncates_long_criteria(self):
        """Format should truncate very long criteria strings."""
        publisher = EvaluatorFeedbackPublisher(event_bus=MagicMock())
        long_criterion = "A" * 100  # Very long criterion
        result = make_evaluation_result(criteria_met={long_criterion: False})
        
        message = publisher.format_feedback_message(result, round_number=1)
        
        # Should truncate to ~60 chars with ellipsis
        assert "..." in message


# ======================================================================
# Tests: Event type registration
# ======================================================================

class TestEventTypeRegistration:
    """Tests that EVALUATION_FEEDBACK is properly registered."""

    def test_evaluation_feedback_event_type_exists(self):
        """EventType enum should include EVALUATION_FEEDBACK."""
        assert hasattr(EventType, "EVALUATION_FEEDBACK")
        assert EventType.EVALUATION_FEEDBACK.value == "evaluation_feedback"

    def test_event_bus_can_handle_evaluation_feedback(self):
        """EventBus should be able to handle EVALUATION_FEEDBACK events."""
        bus = EventBus()
        
        # Should not raise when creating event
        event = BSCEvent(
            event_type=EventType.EVALUATION_FEEDBACK,
            data={"agent_id": "test", "round_number": 1},
            session_id="test-session",
        )
        
        assert event.event_type == EventType.EVALUATION_FEEDBACK
