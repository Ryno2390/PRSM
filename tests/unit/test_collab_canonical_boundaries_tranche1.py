"""P2 Tranche 1 canonical-boundary guard tests for collaboration dispatch."""

import pytest

from prsm.collaboration import CollaborationManager, CollaborationStatus, CollaborationType
from prsm.compute.federation import enhanced_p2p_network, multi_region_p2p_network, p2p_network, scalable_p2p_network


class _CanonicalAgentCollabStub:
    """Minimal canonical bridge stub for CollaborationManager dispatch tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post_task(self, **kwargs):
        self.calls.append(("post_task", kwargs))

        class _Task:
            task_id = "canon-task-1"

        return _Task()

    async def request_review(self, **kwargs):
        self.calls.append(("request_review", kwargs))

        class _Review:
            review_id = "canon-review-1"

        return _Review()

    async def post_query(self, **kwargs):
        self.calls.append(("post_query", kwargs))

        class _Query:
            query_id = "canon-query-1"

        return _Query()


def _guard_no_federation_p2p(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail hard if any known federation P2P stack constructor is touched."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("federation_p2p_stack_must_not_be_used_for_canonical_collab_dispatch")

    monkeypatch.setattr(p2p_network.P2PModelNetwork, "__init__", _boom)
    monkeypatch.setattr(enhanced_p2p_network.ProductionP2PNetwork, "__init__", _boom)
    monkeypatch.setattr(scalable_p2p_network.ScalableP2PNetwork, "__init__", _boom)
    monkeypatch.setattr(multi_region_p2p_network.MultiRegionP2PNetwork, "__init__", _boom)


@pytest.mark.asyncio
async def test_task_dispatch_remains_on_canonical_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK_DELEGATION dispatch must stay on CollaborationManager -> AgentCollaboration bridge."""
    _guard_no_federation_p2p(monkeypatch)
    manager = CollaborationManager()
    canonical = _CanonicalAgentCollabStub()
    manager.set_agent_collaboration(canonical)

    session = manager.create_session(
        collaboration_type=CollaborationType.TASK_DELEGATION,
        initiator_agent_id="agent-1",
        participant_agent_ids=["agent-2"],
        metadata={"description": "canonical task", "ftns_budget": 0.0},
    )
    protocol_id = await manager.dispatch_session(session.session_id)

    assert protocol_id == "canon-task-1"
    assert session.status == CollaborationStatus.ACTIVE
    assert canonical.calls and canonical.calls[0][0] == "post_task"


@pytest.mark.asyncio
async def test_review_dispatch_remains_on_canonical_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """PEER_REVIEW dispatch must stay on CollaborationManager -> AgentCollaboration bridge."""
    _guard_no_federation_p2p(monkeypatch)
    manager = CollaborationManager()
    canonical = _CanonicalAgentCollabStub()
    manager.set_agent_collaboration(canonical)

    session = manager.create_session(
        collaboration_type=CollaborationType.PEER_REVIEW,
        initiator_agent_id="agent-1",
        participant_agent_ids=["agent-2"],
        metadata={"description": "canonical review", "content_cid": "QmCID"},
    )
    protocol_id = await manager.dispatch_session(session.session_id)

    assert protocol_id == "canon-review-1"
    assert session.status == CollaborationStatus.ACTIVE
    assert canonical.calls and canonical.calls[0][0] == "request_review"


@pytest.mark.asyncio
async def test_query_dispatch_remains_on_canonical_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """KNOWLEDGE_EXCHANGE dispatch must stay on CollaborationManager -> AgentCollaboration bridge."""
    _guard_no_federation_p2p(monkeypatch)
    manager = CollaborationManager()
    canonical = _CanonicalAgentCollabStub()
    manager.set_agent_collaboration(canonical)

    session = manager.create_session(
        collaboration_type=CollaborationType.KNOWLEDGE_EXCHANGE,
        initiator_agent_id="agent-1",
        participant_agent_ids=[],
        metadata={"description": "canonical query", "topic": "infra"},
    )
    protocol_id = await manager.dispatch_session(session.session_id)

    assert protocol_id == "canon-query-1"
    assert session.status == CollaborationStatus.ACTIVE
    assert canonical.calls and canonical.calls[0][0] == "post_query"

