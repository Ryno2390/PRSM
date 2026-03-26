"""
NWTN ↔ OpenClaw Adapter
========================

The central integration layer that wires NWTN's Agent Team pipeline to
OpenClaw's runtime.

Responsibilities
----------------
1. **Session bootstrap**: receives a user goal from the Gateway, runs the
   NWTN interview, generates the MetaPlan, assembles the Agent Team.
2. **Branch setup**: creates per-agent git branches via BranchManager.
3. **Whiteboard monitoring**: registers each agent's MEMORY.md file with
   the WhiteboardMonitor (BSC pipeline).
4. **Skills registration**: publishes each team member as an OpenClaw skill
   via the SkillsBridge.
5. **Synthesis scheduling**: attaches a HeartbeatHook to the Gateway
   heartbeat events so that Nightly Synthesis fires automatically.
6. **Event routing**: dispatches Gateway messages (user messages, heartbeat,
   agent file events) to the right handlers.

Standalone mode
---------------
If no ``gateway`` is provided (or OpenClaw is not running), the adapter
works in standalone mode — you provide the QuestionCallback directly and
control the session lifecycle manually.  All components still function
identically.

OpenClaw agent file layout (default)
-------------------------------------
``~/.openclaw/agents/<role>/MEMORY.md``

Override via ``agent_base_dir`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default OpenClaw agent memory file layout
DEFAULT_OPENCLAW_AGENT_DIR = Path.home() / ".openclaw" / "agents"


# ======================================================================
# Session state snapshot
# ======================================================================

@dataclass
class SessionState:
    """Runtime state maintained by the adapter for a single NWTN session."""
    session_id: str
    goal: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "initialising"
    """'initialising' | 'interviewing' | 'planning' | 'active' | 'synthesising' | 'closed'"""
    meta_plan_title: str = ""
    team_members: List[str] = field(default_factory=list)
    """Branch names of assembled team members."""
    whiteboard_entries: int = 0
    synthesis_count: int = 0
    last_synthesis: Optional[datetime] = None


# ======================================================================
# NWTNOpenClawAdapter
# ======================================================================

class NWTNOpenClawAdapter:
    """
    Main NWTN ↔ OpenClaw integration coordinator.

    Parameters
    ----------
    whiteboard_store : WhiteboardStore
    promoter : BSCPromoter
    ledger : ProjectLedger
    signer : LedgerSigner
    synthesizer : NarrativeSynthesizer
    repo_path : Path, optional
        Git repository for agent branches (defaults to cwd).
    gateway : OpenClawGateway, optional
        Live Gateway connection.  If None, standalone mode is used.
    model_registry : ModelRegistry, optional
        For registry-based agent selection.
    agent_base_dir : Path, optional
        Root directory where OpenClaw stores agent MEMORY.md files.
    dag_anchor : DAGAnchor, optional
        For milestone DAG anchoring.
    backend_registry : optional
        LLM backends for interview/planning/synthesis.
    """

    def __init__(
        self,
        whiteboard_store,
        promoter,
        ledger,
        signer,
        synthesizer,
        repo_path: Optional[Path] = None,
        gateway=None,
        model_registry=None,
        agent_base_dir: Optional[Path] = None,
        dag_anchor=None,
        backend_registry=None,
    ) -> None:
        self._store = whiteboard_store
        self._promoter = promoter
        self._ledger = ledger
        self._signer = signer
        self._synthesizer = synthesizer
        self._repo_path = repo_path or Path.cwd()
        self._gateway = gateway
        self._model_registry = model_registry
        self._agent_dir = agent_base_dir or DEFAULT_OPENCLAW_AGENT_DIR
        self._dag_anchor = dag_anchor
        self._backend = backend_registry

        # Lazy-initialised components
        self._monitor = None
        self._heartbeat_hook = None
        self._branch_manager = None
        self._skills_bridge = None

        # Active sessions (session_id → SessionState)
        self._sessions: Dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start_session(
        self,
        goal: str,
        session_id: Optional[str] = None,
        ask: Optional[Callable] = None,
    ) -> SessionState:
        """
        Bootstrap a new NWTN session from an initial goal statement.

        This is the top-level entry point: it runs the full pipeline from
        interview to team assembly and monitoring setup.

        Parameters
        ----------
        goal : str
            The user's initial goal statement.
        session_id : str, optional
            Explicit session ID; auto-generated if omitted.
        ask : QuestionCallback, optional
            Async function to present questions to the user.  If None and
            a Gateway is connected, ``gateway.ask_user`` is used.  In
            standalone mode without a gateway, questions are skipped.

        Returns
        -------
        SessionState
        """
        import uuid as _uuid
        from prsm.compute.nwtn.team import (
            InterviewSession, MetaPlanner, TeamAssembler, BranchManager,
        )
        from prsm.compute.nwtn.whiteboard import WhiteboardMonitor

        session_id = session_id or str(_uuid.uuid4())[:8]
        state = SessionState(session_id=session_id, goal=goal, status="initialising")
        self._sessions[session_id] = state

        await self._store.create_session(session_id)

        # Determine question callback
        question_cb = ask
        if question_cb is None and self._gateway and self._gateway.is_connected:
            question_cb = self._gateway.ask_user
        if question_cb is None:
            async def _silent(_q):
                return "n/a"
            question_cb = _silent

        # 1. Interview
        state.status = "interviewing"
        logger.info("Adapter: starting interview for session=%s", session_id)
        interview = InterviewSession(
            ask=question_cb,
            backend_registry=self._backend,
            session_id=session_id,
        )
        brief = await interview.run(goal)

        # 2. Plan
        state.status = "planning"
        planner = MetaPlanner(backend_registry=self._backend)
        meta_plan = await planner.generate(brief)
        state.meta_plan_title = meta_plan.title
        logger.info(
            "Adapter: meta-plan '%s' generated (%d milestones)",
            meta_plan.title, len(meta_plan.milestones),
        )

        # 3. Assemble team
        assembler = TeamAssembler(
            model_registry=self._model_registry,
            date_suffix=datetime.now(timezone.utc).strftime("%Y%m%d"),
        )
        team = await assembler.assemble(meta_plan)
        state.team_members = [m.branch_name for m in team.members]

        # 4. Create git branches
        self._branch_manager = BranchManager(repo_path=self._repo_path)
        try:
            await self._branch_manager.create_team_branches(team.members)
            logger.info(
                "Adapter: created %d git branches", len(team.members)
            )
        except RuntimeError as exc:
            logger.warning("Adapter: branch creation failed (%s) — continuing", exc)

        # 5. Register skills with OpenClaw (if Gateway connected)
        self._skills_bridge = _import_skills_bridge()
        if self._gateway and self._gateway.is_connected:
            await self._register_skills(team, session_id)

        # 6. Start whiteboard monitor
        self._monitor = WhiteboardMonitor(
            promoter=self._promoter,
            store=self._store,
            min_chunk_chars=80,
        )
        for member in team.members:
            memory_path = self._agent_memory_path(member.role, member.branch_name)
            self._monitor.watch_agent(
                file_path=str(memory_path),
                source_agent=member.branch_name,
                session_id=session_id,
            )
        await self._monitor.start()

        # 7. Start heartbeat hook
        self._heartbeat_hook = _build_heartbeat_hook(
            store=self._store,
            synthesizer=self._synthesizer,
            ledger=self._ledger,
            signer=self._signer,
            session_id=session_id,
            meta_plan=meta_plan,
            dag_anchor=self._dag_anchor,
        )
        await self._heartbeat_hook.start()

        # 8. Write meta-plan to whiteboard as first entry
        await self._write_meta_plan_to_whiteboard(meta_plan, session_id)

        state.status = "active"
        logger.info("Adapter: session %s active — team: %s", session_id, state.team_members)
        return state

    async def end_session(self, session_id: str) -> None:
        """
        Gracefully close a session: force synthesis, close monitor, archive whiteboard.
        """
        state = self._sessions.get(session_id)
        if not state:
            return
        state.status = "synthesising"

        # Force final synthesis
        if self._heartbeat_hook:
            await self._heartbeat_hook.stop(force_synthesis=True)
            state.synthesis_count = self._heartbeat_hook.synthesis_count
            state.last_synthesis = datetime.now(timezone.utc)

        # Stop monitor
        if self._monitor:
            await self._monitor.stop()

        # Archive whiteboard
        await self._store.archive_session(session_id)

        state.status = "closed"
        logger.info("Adapter: session %s closed", session_id)

    def get_state(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------
    # Gateway event routing
    # ------------------------------------------------------------------

    async def handle_gateway_message(self, msg) -> None:
        """
        Route an inbound ``OpenClawMessage`` to the appropriate handler.

        Called by the main Gateway event loop.
        """
        if msg.type == "heartbeat":
            if self._heartbeat_hook:
                await self._heartbeat_hook.on_gateway_heartbeat()

        elif msg.type == "agent_event" and msg.event == "file_updated":
            # Supplement watchfiles: manually trigger a read on the updated file
            if self._monitor and msg.path:
                resolved = str(Path(msg.path).resolve())
                agent = self._monitor._agents.get(resolved)
                if agent:
                    await self._monitor._read_delta(agent)

        elif msg.type == "message":
            logger.debug(
                "Adapter: received user message from %s: %s",
                msg.from_user, (msg.text or "")[:80],
            )
            # User messages during an active session go to the whiteboard
            # as external context (not BSC-filtered — direct inject)
            for session_id in self._sessions:
                if self._sessions[session_id].status == "active":
                    await self._inject_user_context(msg.text or "", session_id)
                    break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _write_meta_plan_to_whiteboard(self, meta_plan, session_id: str) -> None:
        """Write the MetaPlan as the first (always-promoted) whiteboard entry."""
        from prsm.compute.nwtn.bsc import (
            PromotionDecision, ChunkMetadata, FilterDecision, KLFilterResult,
        )
        from datetime import datetime, timezone

        decision = PromotionDecision(
            promoted=True,
            chunk=meta_plan.to_whiteboard_entry(),
            metadata=ChunkMetadata(
                source_agent="nwtn/planner",
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
            ),
            surprise_score=1.0,
            raw_perplexity=0.0,
            similarity_score=0.0,
            kl_result=KLFilterResult(
                decision=FilterDecision.PROMOTE,
                score=1.0,
                epsilon=0.0,
                reason="MetaPlan forced-promoted at session start",
            ),
            dedup_result=None,
            reason="MetaPlan written to whiteboard at session start",
        )
        await self._store.write(decision)
        logger.debug("Adapter: MetaPlan written to whiteboard for session=%s", session_id)

    async def _inject_user_context(self, text: str, session_id: str) -> None:
        """Direct-inject a user message into the whiteboard (bypasses BSC)."""
        from prsm.compute.nwtn.bsc import (
            PromotionDecision, ChunkMetadata, FilterDecision, KLFilterResult,
        )

        decision = PromotionDecision(
            promoted=True,
            chunk=f"[USER] {text}",
            metadata=ChunkMetadata(
                source_agent="user/gateway",
                session_id=session_id,
            ),
            surprise_score=0.9,
            raw_perplexity=0.0,
            similarity_score=0.0,
            kl_result=KLFilterResult(
                decision=FilterDecision.PROMOTE,
                score=0.9,
                epsilon=0.0,
                reason="Direct user input",
            ),
            dedup_result=None,
            reason="User message injected via Gateway",
        )
        await self._store.write(decision)

    async def _register_skills(self, team, session_id: str) -> None:
        """Register team members as OpenClaw skills via the Gateway."""
        if not self._gateway or not self._skills_bridge:
            return
        try:
            skills = [
                self._skills_bridge.team_member_to_skill(m).to_dict()
                for m in team.members
            ]
            await self._gateway._conn.send_json({
                "type": "register_skills",
                "session_id": session_id,
                "skills": skills,
            })
            logger.info(
                "Adapter: registered %d skills with OpenClaw Gateway", len(skills)
            )
        except Exception as exc:
            logger.warning("Adapter: skill registration failed: %s", exc)

    def _agent_memory_path(self, role: str, branch_name: str) -> Path:
        """
        Return the expected MEMORY.md path for an agent.

        Priority:
        1. ``<agent_base_dir>/<role>/MEMORY.md``
        2. Fallback: cwd agent output directory
        """
        primary = self._agent_dir / role / "MEMORY.md"
        if primary.parent.exists():
            return primary
        # Fallback: write to a local directory the tests can control
        return Path(f".nwtn_agents/{role}/MEMORY.md")


# ======================================================================
# Module-level helpers (avoid circular imports)
# ======================================================================

def _import_skills_bridge():
    from .skills_bridge import SkillsBridge
    return SkillsBridge()


def _build_heartbeat_hook(
    store, synthesizer, ledger, signer,
    session_id, meta_plan, dag_anchor,
):
    from .heartbeat_hook import HeartbeatHook
    return HeartbeatHook(
        whiteboard_store=store,
        synthesizer=synthesizer,
        ledger=ledger,
        signer=signer,
        session_id=session_id,
        meta_plan=meta_plan,
        dag_anchor=dag_anchor,
        min_whiteboard_entries=1,  # lower threshold for testing
        min_session_minutes=0.0,
    )
