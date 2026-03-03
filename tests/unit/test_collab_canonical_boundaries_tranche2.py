"""P2 Tranche 2 compatibility-fence and canonical-boundary tests for collaboration dispatch."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prsm.collaboration import CollaborationManager, CollaborationStatus, CollaborationType
from prsm.compute.federation import enhanced_p2p_network, p2p_network


_ROOT = Path(__file__).resolve().parents[2]


class _CanonicalAgentCollabStub:
    """Minimal canonical bridge stub for CollaborationManager dispatch tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post_task(self, **kwargs):
        self.calls.append(("post_task", kwargs))
        return SimpleNamespace(task_id="canon-task-2")

    async def request_review(self, **kwargs):
        self.calls.append(("request_review", kwargs))
        return SimpleNamespace(review_id="canon-review-2")

    async def post_query(self, **kwargs):
        self.calls.append(("post_query", kwargs))
        return SimpleNamespace(query_id="canon-query-2")


class _TaskStub:
    """Minimal task-like object compatible with federation execution APIs for fence tests."""

    task_id = "task-tranche2"

    def dict(self):
        return {"task_id": self.task_id, "instruction": "x"}


def _guard_no_federation_coordination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if canonical collaboration dispatch touches federation coordination entrypoints."""

    async def _boom(*_args, **_kwargs):
        raise AssertionError("canonical_collaboration_dispatch_must_not_call_federation_coordinate_distributed_execution")

    monkeypatch.setattr(p2p_network.P2PModelNetwork, "coordinate_distributed_execution", _boom)
    monkeypatch.setattr(enhanced_p2p_network.ProductionP2PNetwork, "coordinate_distributed_execution", _boom)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "collab_type,metadata,expected_protocol_id,expected_call",
    [
        (
            CollaborationType.TASK_DELEGATION,
            {"description": "canonical task", "ftns_budget": 0.0},
            "canon-task-2",
            "post_task",
        ),
        (
            CollaborationType.PEER_REVIEW,
            {"description": "canonical review", "content_cid": "QmCID"},
            "canon-review-2",
            "request_review",
        ),
        (
            CollaborationType.KNOWLEDGE_EXCHANGE,
            {"description": "canonical query", "topic": "infra"},
            "canon-query-2",
            "post_query",
        ),
    ],
)
async def test_dispatch_remains_on_canonical_manager_bridge_not_federation(
    monkeypatch: pytest.MonkeyPatch,
    collab_type: CollaborationType,
    metadata: dict,
    expected_protocol_id: str,
    expected_call: str,
) -> None:
    """Canonical collaboration dispatch must stay on manager bridge, never federation execution APIs."""
    _guard_no_federation_coordination(monkeypatch)
    manager = CollaborationManager()
    canonical = _CanonicalAgentCollabStub()
    manager.set_agent_collaboration(canonical)

    session = manager.create_session(
        collaboration_type=collab_type,
        initiator_agent_id="agent-1",
        participant_agent_ids=["agent-2"],
        metadata=metadata,
    )
    protocol_id = await manager.dispatch_session(session.session_id)

    assert protocol_id == expected_protocol_id
    assert session.status == CollaborationStatus.ACTIVE
    assert canonical.calls and canonical.calls[0][0] == expected_call


@pytest.mark.asyncio
async def test_federation_p2p_execution_entrypoint_emits_compatibility_fence_warning() -> None:
    """Federation P2P collaboration-like entrypoints stay compatibility-only with explicit redirect warning."""
    network = p2p_network.P2PModelNetwork()
    network.safety_monitor.validate_model_output = AsyncMock(return_value=False)

    with pytest.warns(
        RuntimeWarning,
        match=(
            "Compatibility-only collaboration entrypoint used"
            ".*CollaborationManager.dispatch_session"
        ),
    ):
        with pytest.raises(ValueError, match="failed safety validation"):
            await network.coordinate_distributed_execution(_TaskStub())


@pytest.mark.asyncio
async def test_enhanced_federation_execution_entrypoint_emits_compatibility_fence_warning() -> None:
    """Enhanced federation collaboration-like entrypoints stay compatibility-only with explicit redirect warning."""
    network = object.__new__(enhanced_p2p_network.ProductionP2PNetwork)
    network.safety_monitor = SimpleNamespace(validate_model_output=AsyncMock(return_value=False))

    with pytest.warns(
        RuntimeWarning,
        match=(
            "Compatibility-only collaboration entrypoint used"
            ".*CollaborationManager.dispatch_session"
        ),
    ):
        with pytest.raises(ValueError, match="failed safety validation"):
            await enhanced_p2p_network.ProductionP2PNetwork.coordinate_distributed_execution(
                network,
                _TaskStub(),
            )


def test_collab_bridge_security_suite_uses_canonical_manager_dispatch_not_federation_entrypoints() -> None:
    """Collaboration-focused security suite should use CollaborationManager dispatch as primary path."""
    collab_bridge_test = _ROOT / "tests/security/test_sprint4_collab_bridge.py"
    source = collab_bridge_test.read_text(encoding="utf-8")

    assert "manager.dispatch_session(" in source
    assert "coordinate_distributed_execution(" not in source
