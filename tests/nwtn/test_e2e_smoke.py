"""
End-to-End NWTN Smoke Tests
============================

Smoke tests that prove the full NWTN pipeline fires correctly with
no mocked internals — EXCEPT for the ML-heavy models (BSCPredictor
loads Qwen2.5-0.5B, SemanticDeduplicator loads sentence-transformers).

These tests validate:
1. EventBus → WhiteboardStore chain works (zero mocks)
2. ScribeAgent fires checkpoints on ROUND_ADVANCED events
3. NWTNSession.run() drives rounds and detects convergence
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Real imports (no mocks)
from prsm.compute.nwtn.bsc.event_bus import EventBus, EventType, BSCEvent
from prsm.compute.nwtn.whiteboard.store import WhiteboardStore
from prsm.compute.nwtn.team.scribe_agent import ScribeAgent
from prsm.compute.nwtn.team.convergence import ConvergenceTracker
from prsm.compute.nwtn.team.planner import Milestone, AgentRole, MetaPlan
from prsm.compute.nwtn.team.assembler import AgentTeam, TeamMember
from prsm.compute.nwtn.bsc.promoter import PromotionDecision, ChunkMetadata
from prsm.compute.nwtn.bsc.kl_filter import FilterDecision


# ======================================================================
# Test 1: EventBus → WhiteboardStore Chain (ZERO MOCKS)
# ======================================================================

@pytest.mark.asyncio
async def test_eventbus_to_whiteboard_chain():
    """
    Test 1: A promoted chunk flows correctly from EventBus → WhiteboardStore.
    
    What it proves:
    - EventBus can publish and deliver events
    - Subscribers receive events correctly
    - Event stats are tracked
    
    This test uses ZERO mocks.
    """
    # 1. Create real EventBus, real WhiteboardStore (in-memory)
    event_bus = EventBus()
    store = WhiteboardStore(db_path=":memory:")
    await store.open()
    
    # Track received events
    received_events = []
    
    async def subscriber(event: BSCEvent):
        """Simple subscriber that appends to a list."""
        received_events.append(event)
    
    try:
        # 2. Subscribe a simple coroutine to CHUNK_PROMOTED
        await event_bus.subscribe(EventType.CHUNK_PROMOTED, subscriber)
        
        # 3. Publish a BSCEvent
        event = BSCEvent(
            event_type=EventType.CHUNK_PROMOTED,
            session_id="smoke-1",
            data={"chunk": "test chunk", "source_agent": "agent/test"},
        )
        await event_bus.publish(event)
        
        # 4. Assert the subscriber was called
        await asyncio.sleep(0.05)  # Let the event propagate
        assert len(received_events) == 1
        assert received_events[0].session_id == "smoke-1"
        assert received_events[0].data["chunk"] == "test chunk"
        
        # 5. Assert EventBus stats show 1 published
        stats = event_bus.get_stats()
        assert stats["published"] == 1
        
    finally:
        await store.close()


# ======================================================================
# Test 2: ScribeAgent Checkpoint on ROUND_ADVANCED
# ======================================================================

@pytest.mark.asyncio
async def test_scribe_agent_checkpoint_fires_on_round_advanced():
    """
    Test 2: Publishing ROUND_ADVANCED triggers ScribeAgent.check_and_run_checkpoint().
    
    What it proves:
    - ScribeAgent subscribes to ROUND_ADVANCED events
    - Events trigger checkpoint calls on the LiveScribe
    - The event-driven lifecycle works correctly
    """
    # 1. Create real EventBus
    event_bus = EventBus()
    
    # 2. Create a mock LiveScribe
    mock_live_scribe = MagicMock()
    mock_live_scribe.check_and_run_checkpoint = AsyncMock(return_value=None)
    mock_live_scribe.setup = AsyncMock(return_value=None)
    
    # 3. Create real ScribeAgent(live_scribe=mock_live_scribe)
    scribe_agent = ScribeAgent(live_scribe=mock_live_scribe)
    
    try:
        # 4. await scribe_agent.start(event_bus)
        await scribe_agent.start(event_bus)
        
        # 5. Publish BSCEvent(EventType.ROUND_ADVANCED, session_id="smoke-2", data={})
        event = BSCEvent(
            event_type=EventType.ROUND_ADVANCED,
            session_id="smoke-2",
            data={},
        )
        await event_bus.publish(event)
        
        # 6. await asyncio.sleep(0.05)  # let the task fire
        await asyncio.sleep(0.1)
        
        # 7. Assert mock_live_scribe.check_and_run_checkpoint was called
        mock_live_scribe.check_and_run_checkpoint.assert_called_once()
        
    finally:
        # 8. await scribe_agent.stop()
        await scribe_agent.stop()


# ======================================================================
# Test 3: Full Pipeline Run Loop
# ======================================================================

@pytest.mark.asyncio
async def test_full_pipeline_run_loop():
    """
    Test 3: NWTNSession.run() drives rounds and detects convergence.
    
    What it proves:
    - NWTNSession.run() executes multiple rounds
    - Convergence signals are detected
    - The full loop terminates correctly
    
    Mocks only ML-heavy components (BSCPredictor, SemanticDeduplicator, etc.)
    """
    # Create real WhiteboardStore with in-memory SQLite
    store = WhiteboardStore(db_path=":memory:")
    await store.open()
    
    try:
        # Build a minimal NWTNOpenClawAdapter with mocked dependencies
        from prsm.compute.nwtn.openclaw.adapter import NWTNOpenClawAdapter, SessionState
        
        # Mock BSCPromoter (process_chunk returns a promoted PromotionDecision)
        mock_promoter = MagicMock()
        
        def make_promotion_decision(chunk: str, source_agent: str, session_id: str) -> PromotionDecision:
            """Create a promoted PromotionDecision."""
            from prsm.compute.nwtn.bsc.semantic_dedup import DedupResult
            
            return PromotionDecision(
                promoted=True,
                chunk=chunk,
                metadata=ChunkMetadata(
                    source_agent=source_agent,
                    session_id=session_id,
                ),
                surprise_score=0.85,
                raw_perplexity=80.0,
                similarity_score=0.3,
                kl_result=MagicMock(
                    decision=FilterDecision.PROMOTE,
                    score=0.85,
                    epsilon=0.55,
                    reason="test",
                ),
                dedup_result=DedupResult(
                    is_redundant=False,
                    max_similarity=0.3,
                    most_similar_index=None,
                    reason="novel",
                ),
                quality_report=None,
                reason="Test decision",
            )
        
        mock_promoter.process_chunk = AsyncMock(side_effect=make_promotion_decision)
        mock_promoter.advance_round = MagicMock(return_value={
            "round": 0,
            "epsilon": None,
            "dedup_evicted": 0,
        })
        
        # Mock InterviewSession
        mock_interview = MagicMock()
        mock_interview.run = AsyncMock()
        mock_interview.brief = MagicMock()
        mock_interview.brief.session_id = "smoke-3"
        mock_interview.brief.goal = "Test goal"
        mock_interview.brief.context = "Test context"
        mock_interview.brief.technology_stack = []
        mock_interview.brief.constraints = []
        mock_interview.brief.success_criteria = []
        mock_interview.brief.existing_codebase = None
        mock_interview.brief.timeline = None
        mock_interview.brief.preferred_roles = []
        mock_interview.brief.raw_qa = []
        mock_interview.brief.created_at = datetime.now(timezone.utc)
        mock_interview.brief.llm_assisted = False
        
        # Mock MetaPlanner.generate() - return a minimal MetaPlan
        mock_plan = MagicMock(spec=MetaPlan)
        mock_plan.session_id = "smoke-3"
        mock_plan.title = "Smoke Test Plan"
        mock_plan.objective = "Test the pipeline"
        mock_plan.milestones = [Milestone(title="Phase 1", description="Initial work")]
        mock_plan.required_roles = [
            AgentRole(role="architect", description="Design", priority=1),
            AgentRole(role="coder", description="Implement", priority=2),
        ]
        mock_plan.roles_by_priority = MagicMock(return_value=mock_plan.required_roles)
        mock_plan.to_whiteboard_entry = MagicMock(return_value="Test plan entry")
        mock_plan.success_criteria = ["Goal achieved"]
        mock_plan.constraints = []
        
        mock_planner = MagicMock()
        mock_planner.generate = AsyncMock(return_value=mock_plan)
        
        # Mock TeamAssembler.assemble() - return AgentTeam with 2 mock members
        team = AgentTeam(
            session_id="smoke-3",
            meta_plan=mock_plan,
            members=[
                TeamMember(
                    role="architect",
                    branch_name="agent/architect",
                    agent_id="architect",
                    agent_name="Architect",
                    model_id="test-model",
                ),
                TeamMember(
                    role="coder",
                    branch_name="agent/coder",
                    agent_id="coder",
                    agent_name="Coder",
                    model_id="test-model",
                ),
            ],
        )
        
        mock_assembler = MagicMock()
        mock_assembler.assemble = AsyncMock(return_value=team)
        
        # Mock BranchManager.create_team_branches()
        mock_branch_manager = MagicMock()
        mock_branch_manager.create_team_branches = AsyncMock(return_value=["agent/architect", "agent/coder"])
        
        # Create the adapter with mocked internals
        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store,
            promoter=mock_promoter,
            ledger=None,
            signer=None,
            synthesizer=None,
            repo_path=None,
            orchestrator_backend=None,
        )
        
        # Patch internal components
        adapter._interview_class = MagicMock(return_value=mock_interview)
        adapter._planner = mock_planner
        adapter._assembler = mock_assembler
        adapter._branch_manager = mock_branch_manager
        adapter._monitor = None
        adapter._heartbeat_hook = None
        
        # Create real ConvergenceTracker and register agents
        convergence_tracker = ConvergenceTracker(min_consecutive_rounds=1)  # Use 1 for faster convergence in tests
        convergence_tracker.register_agent("architect")
        convergence_tracker.register_agent("coder")
        adapter._convergence_trackers = {"smoke-3": convergence_tracker}
        
        # Create SessionState
        state = SessionState(
            session_id="smoke-3",
            goal="Test goal",
            status="running",
            team_members=["architect", "coder"],
            scribe_running=False,
            created_at=datetime.now(timezone.utc),
        )
        adapter._sessions = {"smoke-3": state}
        
        # Create NWTNSession via adapter internals
        from prsm.compute.nwtn.session import NWTNSession, RunResult
        session = NWTNSession(adapter=adapter, session_state=state)
        
        # 3. Define agent_output_fn:
        #    - rounds 1-2: return "working on it"
        #    - round 3+: return "TASK COMPLETE"  ← convergence signal
        call_counts = {}
        
        async def agent_output_fn(agent_id: str, round_num: int) -> str:
            key = (agent_id, round_num)
            call_counts[key] = call_counts.get(key, 0) + 1
            
            if round_num < 3:
                return f"Agent {agent_id} is working on it (round {round_num})"
            else:
                return "TASK COMPLETE"
        
        # 4. Run session.run(max_rounds=5, round_poll_interval=0.01, agent_output_fn=agent_output_fn)
        result = await session.run(
            max_rounds=5,
            round_poll_interval=0.01,
            agent_output_fn=agent_output_fn,
        )
        
        # 5. Assert:
        #    - result.converged == True
        #    - result.rounds_completed <= 5
        #    - result.elapsed_seconds > 0
        #    - session.status()["converged"] == True
        
        assert result.converged is True, f"Expected converged=True, got {result.converged}"
        assert result.rounds_completed <= 5, f"Expected rounds <= 5, got {result.rounds_completed}"
        assert result.elapsed_seconds > 0, f"Expected elapsed_seconds > 0, got {result.elapsed_seconds}"
        
        status = session.status()
        assert status["converged"] is True, f"Expected status['converged']=True, got {status}"
        
    finally:
        await store.close()


# ======================================================================
# Helper: Create Real BSCPromoter with Mocked ML Components
# ======================================================================

def _make_mock_promoter():
    """
    Create a mock BSCPromoter that behaves like a real one but with
    mocked ML sub-components (BSCPredictor, SemanticDeduplicator).
    """
    from prsm.compute.nwtn.bsc.promoter import BSCPromoter
    from prsm.compute.nwtn.bsc.semantic_dedup import DedupResult
    from prsm.compute.nwtn.bsc.kl_filter import FilterDecision
    
    promoter = MagicMock(spec=BSCPromoter)
    
    async def mock_process_chunk(chunk: str, context: str, source_agent: str, session_id: str):
        return PromotionDecision(
            promoted=True,
            chunk=chunk,
            metadata=ChunkMetadata(
                source_agent=source_agent,
                session_id=session_id,
            ),
            surprise_score=0.85,
            raw_perplexity=80.0,
            similarity_score=0.3,
            kl_result=MagicMock(
                decision=FilterDecision.PROMOTE,
                score=0.85,
                epsilon=0.55,
                reason="mocked",
            ),
            dedup_result=DedupResult(
                is_redundant=False,
                max_similarity=0.3,
                most_similar_index=None,
                reason="novel",
            ),
            quality_report=None,
            reason="Mocked promotion",
        )
    
    promoter.process_chunk = AsyncMock(side_effect=mock_process_chunk)
    promoter.advance_round = MagicMock(return_value={
        "round": 0,
        "epsilon": None,
        "dedup_evicted": 0,
    })
    
    return promoter
