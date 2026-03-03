"""P2 Tranche 1 observability baseline tests for canonical collaboration path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.collaboration import CollaborationManager, CollaborationType
from prsm.node.agent_collaboration import AgentCollaboration
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.transport import P2PMessage, PeerConnection, WebSocketTransport


@pytest.mark.asyncio
async def test_transport_handshake_failure_taxonomy_emits_deterministically() -> None:
    """Handshake failure telemetry should classify bounded reason labels."""
    local = generate_node_identity("local")
    remote = generate_node_identity("remote")
    transport = WebSocketTransport(local, host="127.0.0.1", port=19930)

    # Missing public key
    missing_pk = P2PMessage(
        msg_type="handshake",
        sender_id=remote.node_id,
        payload={},
    )
    missing_pk.sign(remote)
    ok, reason = await transport._validate_handshake_message(missing_pk, require_ack_for=False)
    assert ok is False
    assert reason == "Missing public key"

    # Replay nonce
    replay = P2PMessage(
        msg_type="handshake",
        sender_id=remote.node_id,
        payload={"public_key": remote.public_key_b64},
    )
    replay.sign(remote)
    ok1, _ = await transport._validate_handshake_message(replay, require_ack_for=False)
    ok2, reason2 = await transport._validate_handshake_message(replay, require_ack_for=False)
    assert ok1 is True
    assert ok2 is False
    assert reason2 == "Replay nonce"

    telemetry = transport.get_telemetry_snapshot()
    assert telemetry["handshake_failure_total"] == 2
    assert telemetry["handshake_failure_reasons"]["missing_public_key"] == 1
    assert telemetry["handshake_failure_reasons"]["replay_nonce"] == 1


@pytest.mark.asyncio
async def test_transport_dispatch_success_and_failure_reasons() -> None:
    """Dispatch telemetry should track successful and failed handler execution."""
    local = generate_node_identity("local")
    transport = WebSocketTransport(local, host="127.0.0.1", port=19931)

    async def _ok_handler(msg, peer):
        return None

    async def _bad_handler(msg, peer):
        raise RuntimeError("boom")

    transport.on_message("direct", _ok_handler)
    transport.on_message("direct", _bad_handler)

    await transport._dispatch(
        P2PMessage(msg_type="direct", sender_id="peer-x", payload={"k": "v"}),
        PeerConnection(peer_id="peer-x", address="x", websocket=None),
    )

    telemetry = transport.get_telemetry_snapshot()
    assert telemetry["dispatch_success_total"] == 1
    assert telemetry["dispatch_failure_total"] == 1
    assert telemetry["dispatch_failure_reasons"]["handler_exception"] == 1


@pytest.mark.asyncio
async def test_gossip_publish_forward_drop_counts_with_bounded_labels() -> None:
    """Gossip telemetry should classify publish/forward/drop using bounded subtype labels."""
    identity = generate_node_identity("gossip-node")
    transport = SimpleNamespace(
        identity=identity,
        on_message=MagicMock(),
        gossip=AsyncMock(return_value=1),
    )
    gossip = GossipProtocol(transport=transport, fanout=2, default_ttl=3)

    await gossip.publish("agent_task_offer", {"task_id": "t-1"})

    await gossip._handle_gossip(
        P2PMessage(
            msg_type="gossip",
            sender_id="peer-a",
            payload={"subtype": "agent_task_offer", "data": {}, "origin": "peer-a"},
            ttl=2,
        ),
        PeerConnection(peer_id="peer-a", address="x", websocket=None),
    )
    await gossip._handle_gossip(
        P2PMessage(
            msg_type="gossip",
            sender_id="peer-b",
            payload={"subtype": "non_canonical_subtype", "data": {}, "origin": "peer-b"},
            ttl=1,
        ),
        PeerConnection(peer_id="peer-b", address="y", websocket=None),
    )

    telemetry = gossip.get_telemetry_snapshot()
    assert telemetry["publish_total"] == 1
    assert telemetry["publish_by_subtype"]["agent_task_offer"] == 1
    assert telemetry["forward_total"] == 1
    assert telemetry["forward_by_subtype"]["agent_task_offer"] == 1
    assert telemetry["drop_total"] == 1
    assert telemetry["drop_by_subtype"]["other"] == 1
    assert telemetry["drop_by_reason"]["ttl_exhausted"] == 1


@pytest.mark.asyncio
async def test_collaboration_protocol_transition_and_terminal_reason_counters() -> None:
    """AgentCollaboration telemetry should emit deterministic transition and terminal reasons."""
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)

    collab = AgentCollaboration(gossip=gossip, node_id="node-A")

    task = await collab.post_task(
        requester_agent_id="agent-1",
        title="Telemetry Task",
        description="desc",
        ftns_budget=0.0,
    )
    await collab.assign_task(task.task_id, "agent-2")
    await collab.complete_task(task.task_id, {"ok": True})

    review = await collab.request_review(
        submitter_agent_id="agent-1",
        content_cid="Qm123",
        description="review",
        ftns_per_review=0.0,
        max_reviewers=1,
    )
    await collab.submit_review(
        review_id=review.review_id,
        reviewer_agent_id="agent-2",
        reviewer_node_id="node-B",
        verdict="accept",
    )

    telemetry = collab.get_telemetry_snapshot()
    assert telemetry["protocol_transition_by_type"]["task_posted"] == 1
    assert telemetry["protocol_transition_by_type"]["task_assigned"] == 1
    assert telemetry["protocol_transition_by_type"]["task_completed"] == 1
    assert telemetry["protocol_transition_by_type"]["review_requested"] == 1
    assert telemetry["protocol_transition_by_type"]["review_submitted"] == 1
    assert telemetry["protocol_transition_by_type"]["review_finalized"] == 1
    assert telemetry["terminal_outcome_by_reason"]["task_completed"] == 1
    assert telemetry["terminal_outcome_by_reason"]["review_accepted"] == 1


@pytest.mark.asyncio
async def test_collaboration_manager_dispatch_reason_counters() -> None:
    """CollaborationManager bridge telemetry should classify dispatch outcomes by reason."""
    manager = CollaborationManager()

    missing = await manager.dispatch_session("missing-session")
    assert missing is None

    session_unwired = manager.create_session(
        collaboration_type=CollaborationType.TASK_DELEGATION,
        initiator_agent_id="agent-1",
        participant_agent_ids=[],
    )
    unwired = await manager.dispatch_session(session_unwired.session_id)
    assert unwired is None

    class _FailingAgentCollab:
        async def post_task(self, **kwargs):
            raise ValueError("insufficient")

    manager.set_agent_collaboration(_FailingAgentCollab())
    session_error = manager.create_session(
        collaboration_type=CollaborationType.TASK_DELEGATION,
        initiator_agent_id="agent-1",
        participant_agent_ids=[],
        metadata={"description": "x", "ftns_budget": 1.0},
    )
    failed = await manager.dispatch_session(session_error.session_id)
    assert failed is None

    class _SuccessTask:
        task_id = "task-success-1"

    class _SuccessAgentCollab:
        async def post_task(self, **kwargs):
            return _SuccessTask()

    manager.set_agent_collaboration(_SuccessAgentCollab())
    session_ok = manager.create_session(
        collaboration_type=CollaborationType.TASK_DELEGATION,
        initiator_agent_id="agent-1",
        participant_agent_ids=[],
        metadata={"description": "ok", "ftns_budget": 0.0},
    )
    ok_protocol_id = await manager.dispatch_session(session_ok.session_id)
    assert ok_protocol_id == "task-success-1"

    telemetry = manager.get_telemetry_snapshot()
    assert telemetry["dispatch_failure_total"] == 3
    assert telemetry["dispatch_success_total"] == 1
    assert telemetry["dispatch_reasons"]["session_not_found"] == 1
    assert telemetry["dispatch_reasons"]["agent_collab_not_wired"] == 1
    assert telemetry["dispatch_reasons"]["dispatch_value_error"] == 1
    assert telemetry["dispatch_reasons"]["dispatch_success"] == 1
