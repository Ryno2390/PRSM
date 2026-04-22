"""
Tests for Sub-phase 10.5: OpenClaw Integration

Coverage:
  1. Gateway — connect/disconnect, send/receive, ask_user, message routing
  2. SkillsBridge — capability mapping, role→tools, skill spec generation
  3. HeartbeatHook — timer logic, gateway heartbeat, forced synthesis
  4. NWTNOpenClawAdapter — session lifecycle, event routing, standalone mode
  5. End-to-end — full standalone session: goal → plan → monitor → synthesise
"""

from __future__ import annotations

import asyncio
import subprocess as _subprocess_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Capture real Popen at import time (same pattern as team coordination tests)
_REAL_POPEN = _subprocess_module.Popen

from prsm.compute.nwtn.openclaw import (
    DEFAULT_GATEWAY_URL,
    HeartbeatHook,
    NWTNOpenClawAdapter,
    OpenClawGateway,
    OpenClawMessage,
    OpenClawSkillSpec,
    SessionState,
    SkillsBridge,
)
from prsm.compute.nwtn.openclaw.gateway import AbstractGatewayConnection


# ======================================================================
# Mock Gateway Connection
# ======================================================================

class MockGatewayConnection:
    """
    In-process mock of an OpenClaw Gateway WebSocket connection.

    Uses two asyncio Queues: outbound (what NWTN sends) and inbound
    (what we inject as 'Gateway messages').
    """

    def __init__(self) -> None:
        self._outbound: asyncio.Queue = asyncio.Queue()
        self._inbound: asyncio.Queue = asyncio.Queue()
        self._closed = False

    async def send_json(self, data: Dict[str, Any]) -> None:
        await self._outbound.put(data)

    async def receive_json(self) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.wait_for(self._inbound.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    # --- Test helpers ---
    async def inject(self, data: Dict[str, Any]) -> None:
        """Simulate a message arriving from the Gateway."""
        await self._inbound.put(data)

    async def next_sent(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Pop the next message NWTN sent to the Gateway."""
        try:
            return await asyncio.wait_for(self._outbound.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def sent_messages(self) -> List[Dict[str, Any]]:
        """Drain all pending outbound messages synchronously."""
        msgs = []
        while not self._outbound.empty():
            msgs.append(self._outbound.get_nowait())
        return msgs


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_conn():
    return MockGatewayConnection()


@pytest.fixture
def gateway(mock_conn):
    gw = OpenClawGateway(url=DEFAULT_GATEWAY_URL, connection=mock_conn)
    return gw


@pytest.fixture
async def connected_gateway(mock_conn):
    gw = OpenClawGateway(url=DEFAULT_GATEWAY_URL, connection=mock_conn)
    await gw.connect()
    yield gw
    await gw.disconnect()


def _make_components(tmp_path):
    """Build a minimal set of NWTN components for integration tests."""
    from prsm.compute.nwtn.whiteboard import WhiteboardStore
    from prsm.compute.nwtn.synthesis import (
        ProjectLedger, LedgerSigner, NarrativeSynthesizer,
    )
    from prsm.compute.nwtn.bsc import BSCPromoter
    from unittest.mock import MagicMock, AsyncMock

    # Whiteboard store
    store = WhiteboardStore(":memory:")

    # Mocked BSC promoter (always promotes)
    promoter = MagicMock(spec=BSCPromoter)
    promoter.warmup = AsyncMock()
    promoter.stats = {}

    async def _promote(chunk, context, source_agent, session_id, extra_metadata=None):
        from prsm.compute.nwtn.bsc import (
            PromotionDecision, ChunkMetadata, FilterDecision,
            KLFilterResult, DedupResult,
        )
        return PromotionDecision(
            promoted=True, chunk=chunk,
            metadata=ChunkMetadata(source_agent=source_agent, session_id=session_id),
            surprise_score=0.8, raw_perplexity=80.0, similarity_score=0.1,
            kl_result=KLFilterResult(decision=FilterDecision.PROMOTE, score=0.8,
                                     epsilon=0.55, reason="mock"),
            dedup_result=DedupResult(is_redundant=False, max_similarity=0.1,
                                     most_similar_index=None, reason="mock"),
            reason="mock promote",
        )

    promoter.process_chunk = _promote

    ledger_dir = tmp_path / "ledger"
    ledger = ProjectLedger(ledger_dir=ledger_dir, project_title="Test")
    signer = LedgerSigner()
    signer.load_or_generate()
    synthesizer = NarrativeSynthesizer(backend_registry=None)

    return store, promoter, ledger, signer, synthesizer


# ======================================================================
# 1. OpenClawGateway
# ======================================================================

class TestOpenClawGateway:
    @pytest.mark.asyncio
    async def test_connect_sends_register(self, mock_conn):
        gw = OpenClawGateway(connection=mock_conn)
        await gw.connect()

        reg = await mock_conn.next_sent(timeout=1.0)
        assert reg is not None
        assert reg["type"] == "register"
        assert reg["agent_id"] == "nwtn"
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_is_connected_after_connect(self, mock_conn):
        gw = OpenClawGateway(connection=mock_conn)
        assert not gw.is_connected
        await gw.connect()
        assert gw.is_connected
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_sets_not_connected(self, connected_gateway):
        await connected_gateway.disconnect()
        assert not connected_gateway.is_connected

    @pytest.mark.asyncio
    async def test_send_text_emits_reply_frame(self, mock_conn, connected_gateway):
        await mock_conn.next_sent()  # consume register
        await connected_gateway.send_text("Hello, user!", reply_to="msg-1", to_user="user_x")

        sent = await mock_conn.next_sent(timeout=1.0)
        assert sent["type"] == "reply"
        assert sent["text"] == "Hello, user!"
        assert sent["reply_to"] == "msg-1"

    @pytest.mark.asyncio
    async def test_messages_yields_inbound(self, mock_conn, connected_gateway):
        await mock_conn.next_sent()  # consume register

        # Inject a user message
        await mock_conn.inject({
            "type": "message", "id": "m1", "from": "user_abc",
            "channel": "telegram", "text": "Build PRSM",
        })

        received = []
        async for msg in connected_gateway.messages():
            received.append(msg)
            break  # take just one

        assert len(received) == 1
        assert received[0].type == "message"
        assert received[0].text == "Build PRSM"

    @pytest.mark.asyncio
    async def test_message_from_dict(self):
        d = {
            "type": "heartbeat",
            "timestamp": "2026-03-26T18:00:00+00:00",
        }
        msg = OpenClawMessage.from_dict(d)
        assert msg.type == "heartbeat"
        assert msg.timestamp is not None

    @pytest.mark.asyncio
    async def test_ask_user_resolved_by_answer(self, mock_conn, connected_gateway):
        await mock_conn.next_sent()  # consume register

        # Simulate the ask being sent and an answer arriving
        async def _inject_answer():
            await asyncio.sleep(0.05)
            # Find the ask_id from the sent message
            ask_msg = await mock_conn.next_sent(timeout=1.0)
            assert ask_msg["type"] == "ask"
            ask_id = ask_msg["id"]
            await mock_conn.inject({
                "type": "answer", "id": ask_id,
                "text": "Python and FastAPI",
            })

        asyncio.create_task(_inject_answer())
        answer = await connected_gateway.ask_user("What's your tech stack?")
        assert answer == "Python and FastAPI"

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_conn):
        async with OpenClawGateway(connection=mock_conn) as gw:
            assert gw.is_connected
        assert not gw.is_connected


# ======================================================================
# 2. SkillsBridge
# ======================================================================

class TestSkillsBridge:
    def test_capabilities_to_tools_code_generation(self):
        bridge = SkillsBridge()
        tools = bridge.capabilities_to_tools(["code_generation"])
        assert "code_executor" in tools

    def test_capabilities_to_tools_analysis(self):
        bridge = SkillsBridge()
        tools = bridge.capabilities_to_tools(["analysis"])
        assert "web_search" in tools or "file_reader" in tools

    def test_capabilities_to_tools_union(self):
        bridge = SkillsBridge()
        tools_combined = bridge.capabilities_to_tools(["code_generation", "analysis"])
        tools_code = bridge.capabilities_to_tools(["code_generation"])
        tools_analysis = bridge.capabilities_to_tools(["analysis"])
        assert set(tools_combined) == set(tools_code) | set(tools_analysis)

    def test_tools_to_capabilities_reverse(self):
        bridge = SkillsBridge()
        caps = bridge.tools_to_capabilities(["code_executor"])
        assert "code_generation" in caps or "testing" in caps

    def test_role_to_skill_spec_backend_coder(self):
        bridge = SkillsBridge()
        spec = bridge.role_to_skill_spec(
            role="backend-coder",
            model_id="anthropic/claude-sonnet-4-6",
            agent_name="PRSM Coder",
        )
        assert isinstance(spec, OpenClawSkillSpec)
        assert spec.model == "anthropic/claude-sonnet-4-6"
        assert "code_executor" in spec.tools

    def test_role_to_skill_spec_unknown_role(self):
        bridge = SkillsBridge()
        spec = bridge.role_to_skill_spec("exotic-specialist", "model-x")
        assert isinstance(spec, OpenClawSkillSpec)
        assert spec.tools  # should have some fallback tools

    def test_skill_spec_to_dict(self):
        spec = OpenClawSkillSpec(
            skill_id="s1", name="Test", description="desc",
            model="model-x", tools=["code_executor"],
        )
        d = spec.to_dict()
        assert d["skill_id"] == "s1"
        assert "code_executor" in d["tools"]

    def test_team_member_to_skill(self):
        from prsm.compute.nwtn.team.assembler import TeamMember

        member = TeamMember(
            role="backend-coder",
            agent_id="agent-001",
            agent_name="PRSM Coder",
            model_id="anthropic/claude-sonnet-4-6",
            branch_name="agent/backend-coder-20260326",
            capabilities=["code_generation"],
        )
        bridge = SkillsBridge()
        spec = bridge.team_member_to_skill(member)
        assert spec.metadata["prsm_role"] == "backend-coder"
        assert spec.metadata["prsm_branch"] == "agent/backend-coder-20260326"

    def test_skill_to_capabilities_round_trip(self):
        bridge = SkillsBridge()
        spec = bridge.role_to_skill_spec("backend-coder", "model-x")
        caps = bridge.skill_to_capabilities(spec)
        assert len(caps) > 0

    def test_extra_mappings_override(self):
        bridge = SkillsBridge(extra_mappings={"my_capability": ["my_tool"]})
        tools = bridge.capabilities_to_tools(["my_capability"])
        assert "my_tool" in tools


# ======================================================================
# 3. HeartbeatHook
# ======================================================================

class TestHeartbeatHook:
    def _make_hook(self, tmp_path, session_id="sess-hb"):
        store, promoter, ledger, signer, synthesizer = _make_components(tmp_path)
        # Pre-open store and create session synchronously is awkward in sync code;
        # we'll do it in the test
        return store, promoter, ledger, signer, synthesizer, session_id

    @pytest.mark.asyncio
    async def test_start_and_stop(self, tmp_path):
        store, _, ledger, signer, synth, sid = self._make_hook(tmp_path)
        await store.open()
        await store.create_session(sid)
        ledger.load()

        hook = HeartbeatHook(
            whiteboard_store=store,
            synthesizer=synth,
            ledger=ledger,
            signer=signer,
            session_id=sid,
            min_whiteboard_entries=0,
            min_session_minutes=0.0,
        )
        await hook.start()
        assert hook._running
        await hook.stop(force_synthesis=False)
        assert not hook._running
        await store.close()

    @pytest.mark.asyncio
    async def test_forced_synthesis_on_stop(self, tmp_path):
        store, promoter, ledger, signer, synth, sid = self._make_hook(tmp_path)
        await store.open()
        await store.create_session(sid)
        # Add one whiteboard entry
        from prsm.compute.nwtn.bsc import (
            PromotionDecision, ChunkMetadata, FilterDecision, KLFilterResult, DedupResult,
        )
        await store.write(PromotionDecision(
            promoted=True, chunk="Important finding.",
            metadata=ChunkMetadata(source_agent="agent/coder", session_id=sid),
            surprise_score=0.8, raw_perplexity=80.0, similarity_score=0.1,
            kl_result=KLFilterResult(decision=FilterDecision.PROMOTE, score=0.8,
                                     epsilon=0.55, reason="ok"),
            dedup_result=DedupResult(is_redundant=False, max_similarity=0.1,
                                     most_similar_index=None, reason="ok"),
            reason="ok",
        ))
        ledger.load()

        hook = HeartbeatHook(
            whiteboard_store=store,
            synthesizer=synth,
            ledger=ledger,
            signer=signer,
            session_id=sid,
            min_whiteboard_entries=0,
            min_session_minutes=0.0,
        )
        await hook.start()
        await hook.stop(force_synthesis=True)

        assert hook.synthesis_count == 1
        assert ledger.entry_count == 1
        result = ledger.verify()
        assert result.valid
        await store.close()

    @pytest.mark.asyncio
    async def test_manual_trigger(self, tmp_path):
        store, _, ledger, signer, synth, sid = self._make_hook(tmp_path)
        await store.open()
        await store.create_session(sid)
        ledger.load()

        hook = HeartbeatHook(
            whiteboard_store=store,
            synthesizer=synth,
            ledger=ledger,
            signer=signer,
            session_id=sid,
            min_whiteboard_entries=0,
            min_session_minutes=0.0,
        )
        await hook.start()
        entry = await hook.trigger()
        assert entry is not None
        assert ledger.entry_count == 1
        await hook.stop(force_synthesis=False)
        await store.close()

    @pytest.mark.asyncio
    async def test_on_complete_callback_called(self, tmp_path):
        store, _, ledger, signer, synth, sid = self._make_hook(tmp_path)
        await store.open()
        await store.create_session(sid)
        ledger.load()

        completed = []

        async def on_done(entry):
            completed.append(entry)

        hook = HeartbeatHook(
            whiteboard_store=store,
            synthesizer=synth,
            ledger=ledger,
            signer=signer,
            session_id=sid,
            on_complete=on_done,
            min_whiteboard_entries=0,
            min_session_minutes=0.0,
        )
        await hook.start()
        await hook.trigger()
        await hook.stop(force_synthesis=False)

        assert len(completed) == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_gateway_heartbeat_updates_timestamp(self, tmp_path):
        store, _, ledger, signer, synth, sid = self._make_hook(tmp_path)
        await store.open()
        await store.create_session(sid)
        ledger.load()

        hook = HeartbeatHook(
            whiteboard_store=store,
            synthesizer=synth,
            ledger=ledger,
            signer=signer,
            session_id=sid,
            min_whiteboard_entries=100,  # threshold too high to trigger synthesis
            min_session_minutes=0.0,
        )
        await hook.start()
        prev = hook._last_gateway_heartbeat
        await hook.on_gateway_heartbeat()
        assert hook._last_gateway_heartbeat > prev
        await hook.stop(force_synthesis=False)
        await store.close()

    @pytest.mark.asyncio
    async def test_skips_synthesis_below_entry_threshold(self, tmp_path):
        store, _, ledger, signer, synth, sid = self._make_hook(tmp_path)
        await store.open()
        await store.create_session(sid)
        ledger.load()

        hook = HeartbeatHook(
            whiteboard_store=store,
            synthesizer=synth,
            ledger=ledger,
            signer=signer,
            session_id=sid,
            min_whiteboard_entries=50,  # whiteboard has 0 entries
            min_session_minutes=0.0,
        )
        await hook.start()
        await hook.on_gateway_heartbeat()
        # No synthesis because entry count < threshold
        assert hook.synthesis_count == 0
        await hook.stop(force_synthesis=False)
        await store.close()


# ======================================================================
# 4. NWTNOpenClawAdapter
# ======================================================================

class TestNWTNOpenClawAdapter:
    @pytest.mark.asyncio
    async def test_standalone_session_start(self, tmp_path):
        """Session starts correctly without an OpenClaw Gateway."""
        store, promoter, ledger, signer, synth = _make_components(tmp_path)
        await store.open()
        ledger.load()

        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store,
            promoter=promoter,
            ledger=ledger,
            signer=signer,
            synthesizer=synth,
            repo_path=None,  # no git operations
        )

        answers = iter(["Python, aiosqlite", "No breaking changes", "Tests pass", "2 weeks", ""])
        async def mock_ask(q):
            return next(answers, "n/a")

        # Patch BranchManager to avoid real git ops
        with patch("prsm.compute.nwtn.openclaw.adapter.NWTNOpenClawAdapter"
                   "._agent_memory_path", return_value=tmp_path / "MEMORY.md"), \
             patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                   ".create_team_branches", new=AsyncMock(return_value={})):

            state = await adapter.start_session(
                goal="Build the NWTN BSC system",
                ask=mock_ask,
                session_id="standalone-001",
            )

        assert state.session_id == "standalone-001"
        assert state.status == "active"
        assert len(state.team_members) > 0
        assert state.meta_plan_title

        await adapter.end_session("standalone-001")
        assert adapter.get_state("standalone-001").status == "closed"
        assert ledger.entry_count >= 1
        assert ledger.verify().valid
        await store.close()

    @pytest.mark.asyncio
    async def test_end_session_writes_ledger(self, tmp_path):
        store, promoter, ledger, signer, synth = _make_components(tmp_path)
        await store.open()
        ledger.load()

        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store,
            promoter=promoter,
            ledger=ledger,
            signer=signer,
            synthesizer=synth,
        )
        async def ask(q):
            return "n/a"

        with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                   ".create_team_branches", new=AsyncMock(return_value={})):
            await adapter.start_session("Build X", ask=ask, session_id="s1")

        await adapter.end_session("s1")
        assert ledger.entry_count == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_get_state_returns_session(self, tmp_path):
        store, promoter, ledger, signer, synth = _make_components(tmp_path)
        await store.open()
        ledger.load()

        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store, promoter=promoter,
            ledger=ledger, signer=signer, synthesizer=synth,
        )
        async def ask(q):
            return "n/a"

        with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                   ".create_team_branches", new=AsyncMock(return_value={})):
            state = await adapter.start_session("Goal", ask=ask, session_id="s-get")

        assert adapter.get_state("s-get") is state
        assert adapter.get_state("nonexistent") is None
        await store.close()

    @pytest.mark.asyncio
    async def test_handle_gateway_heartbeat(self, tmp_path):
        store, promoter, ledger, signer, synth = _make_components(tmp_path)
        await store.open()
        ledger.load()

        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store, promoter=promoter,
            ledger=ledger, signer=signer, synthesizer=synth,
        )
        async def ask(q):
            return "n/a"

        with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                   ".create_team_branches", new=AsyncMock(return_value={})):
            await adapter.start_session("Goal", ask=ask, session_id="hb-sess")

        # Inject a heartbeat message
        hb_msg = OpenClawMessage(type="heartbeat")
        await adapter.handle_gateway_message(hb_msg)

        # Heartbeat hook should have received it
        assert adapter._heartbeat_hook._last_gateway_heartbeat > 0
        await store.close()

    @pytest.mark.asyncio
    async def test_meta_plan_written_to_whiteboard(self, tmp_path):
        store, promoter, ledger, signer, synth = _make_components(tmp_path)
        await store.open()
        ledger.load()

        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store, promoter=promoter,
            ledger=ledger, signer=signer, synthesizer=synth,
        )
        async def ask(q):
            return "n/a"

        with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                   ".create_team_branches", new=AsyncMock(return_value={})):
            await adapter.start_session("Test Goal", ask=ask, session_id="mp-sess")

        count = await store.entry_count("mp-sess")
        assert count >= 1  # MetaPlan was written
        entries = await store.get_all("mp-sess")
        assert any("META-PLAN" in e.chunk for e in entries)
        await store.close()


# ======================================================================
# 5. End-to-end: full standalone session
# ======================================================================

class TestOpenClawIntegrationEndToEnd:
    @pytest.mark.asyncio
    async def test_complete_standalone_session(self, tmp_path):
        """
        Full standalone session: goal → interview → plan → assemble →
        monitor (simulated agent output) → heartbeat → synthesis → ledger.
        """
        store, promoter, ledger, signer, synth = _make_components(tmp_path)
        await store.open()
        ledger.load()

        # Set up an agent output file
        agent_file = tmp_path / "coder_MEMORY.md"
        agent_file.write_text("")

        adapter = NWTNOpenClawAdapter(
            whiteboard_store=store,
            promoter=promoter,
            ledger=ledger,
            signer=signer,
            synthesizer=synth,
        )

        answers = iter(["Python", "No constraints", "Tests pass", "1 week"])
        async def ask(q):
            return next(answers, "n/a")

        with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                   ".create_team_branches", new=AsyncMock(return_value={})), \
             patch("prsm.compute.nwtn.openclaw.adapter.NWTNOpenClawAdapter"
                   "._agent_memory_path", return_value=agent_file):

            state = await adapter.start_session(
                goal="Build a BSC-powered agent system",
                ask=ask,
                session_id="e2e-oc-001",
            )

        assert state.status == "active"

        # Simulate an agent writing to its MEMORY.md
        agent_file.write_text(
            "DISCOVERY: The BSC predictor needs a 3B model minimum for "
            "accurate perplexity evaluation on code+English mixed input. "
            "Smaller models (0.5B) show high false-positive rates on technical text."
        )

        # Manually trigger a read (replaces real watchfiles event)
        if adapter._monitor:
            agent_states = list(adapter._monitor._agents.values())
            if agent_states:
                await adapter._monitor._read_delta(agent_states[0])
                await adapter._monitor._submit_chunk(agent_states[0], force=True)

        # Force synthesis
        await adapter.end_session("e2e-oc-001")

        assert state.status == "closed"
        assert ledger.entry_count >= 1
        assert ledger.verify().valid

        # Onboarding context should mention the discovery
        ctx = ledger.to_onboarding_context()
        assert len(ctx) > 50

        await store.close()
