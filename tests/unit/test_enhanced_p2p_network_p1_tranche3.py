import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nacl.public import PrivateKey
from nacl.signing import SigningKey, VerifyKey

import prsm.compute.federation.enhanced_p2p_network as enhanced_p2p_network
from prsm.compute.federation.enhanced_p2p_network import ModelShard, ProductionP2PNetwork


@pytest.fixture(autouse=True)
def _patch_optional_p2p_dependencies(monkeypatch):
    """Provide optional dependency symbols when kademlia/libp2p extras are unavailable."""

    class _StubForgetfulStorage:
        pass

    if getattr(enhanced_p2p_network, "PrivateKey", None) is None:
        monkeypatch.setattr(enhanced_p2p_network, "PrivateKey", PrivateKey, raising=False)

    if getattr(enhanced_p2p_network, "SigningKey", None) is None:
        monkeypatch.setattr(enhanced_p2p_network, "SigningKey", SigningKey, raising=False)

    if getattr(enhanced_p2p_network, "VerifyKey", None) is None:
        monkeypatch.setattr(enhanced_p2p_network, "VerifyKey", VerifyKey, raising=False)

    if getattr(enhanced_p2p_network, "ForgetfulStorage", None) is None:
        monkeypatch.setattr(enhanced_p2p_network, "ForgetfulStorage", _StubForgetfulStorage, raising=False)


def _canonical_shard_locations(network: ProductionP2PNetwork) -> dict:
    return {
        str(shard_id): tuple(sorted(peer_ids))
        for shard_id, peer_ids in sorted(network.shard_locations.items(), key=lambda item: str(item[0]))
    }


@pytest.mark.asyncio
async def test_disconnect_reconnect_transitions_reconcile_generation(monkeypatch):
    network = ProductionP2PNetwork(node_id="node-test")

    async def fake_establish_connection(_peer_address: str, _peer_id: str):
        return True

    async def fake_handle_connection_error(_peer_id: str):
        return None

    monkeypatch.setattr(network.secure_connection, "establish_connection", fake_establish_connection)
    monkeypatch.setattr(network.secure_connection, "_handle_connection_error", fake_handle_connection_error)

    connected = await network.connect_to_peer("127.0.0.1:9001", "peer-a")
    assert connected is True
    assert network.peer_connection_state["peer-a"]["state"] == "connected"
    assert network.peer_connection_state["peer-a"]["generation"] == 1

    await network._disconnect_peer("peer-a")
    assert network.peer_connection_state["peer-a"]["state"] == "disconnected"
    assert network.peer_connection_state["peer-a"]["generation"] == 1

    reconnected = await network.connect_to_peer("127.0.0.1:9001", "peer-a")
    assert reconnected is True
    assert network.peer_connection_state["peer-a"]["state"] == "connected"
    assert network.peer_connection_state["peer-a"]["generation"] == 2


@pytest.mark.asyncio
async def test_partition_rejoin_shard_reconciliation_converges_on_repeat(monkeypatch):
    network = ProductionP2PNetwork(node_id="node-test")

    async def fake_handle_connection_error(_peer_id: str):
        return None

    monkeypatch.setattr(network.secure_connection, "_handle_connection_error", fake_handle_connection_error)

    shard = ModelShard(
        shard_id=uuid4(),
        model_content_id="model-cid-1",
        shard_index=0,
        total_shards=1,
        hosted_by=["peer-b", "peer-a"],
        verification_hash=hashlib.sha256(b"s-0").hexdigest(),
        size_bytes=3,
    )
    network.model_shards = {"model-cid-1": [shard]}

    network.active_peers["peer-a"] = SimpleNamespace(active=True, reputation_score=0.8)
    network.active_peers["peer-b"] = SimpleNamespace(active=True, reputation_score=0.8)
    network._record_peer_transition("peer-a", "connected")
    network._record_peer_transition("peer-b", "connected")

    await network._reconcile_shard_locations()
    assert _canonical_shard_locations(network) == {str(shard.shard_id): ("peer-a", "peer-b")}

    await network.mark_peer_partitioned("peer-b")
    assert network.peer_connection_state["peer-b"]["state"] == "partitioned"
    assert _canonical_shard_locations(network) == {str(shard.shard_id): ("peer-a",)}

    network.active_peers["peer-b"].active = True
    network._record_peer_transition("peer-b", "connected")

    state_1 = await network.reconcile_peer_state("peer-b")
    snapshot_1 = _canonical_shard_locations(network)
    state_2 = await network.reconcile_peer_state("peer-b")
    snapshot_2 = _canonical_shard_locations(network)

    assert state_1["status"] == "reconciled"
    assert state_2["status"] == "reconciled"
    assert snapshot_1 == snapshot_2
    assert snapshot_2 == {str(shard.shard_id): ("peer-a", "peer-b")}


def test_in_flight_reconcile_fail_closed_is_idempotent():
    network = ProductionP2PNetwork(node_id="node-test")
    network.in_flight_tasks = {
        "op-a": {"peer_id": "peer-a", "state": "pending_dispatch"},
        "op-b": {"peer_id": "peer-a", "state": "dispatching"},
        "op-c": {"peer_id": "peer-a", "state": "committed"},
        "op-d": {"peer_id": "peer-b", "state": "pending_dispatch"},
    }

    first = network._reconcile_in_flight_tasks("peer-a")
    second = network._reconcile_in_flight_tasks("peer-a")

    assert first == 2
    assert second == 0
    assert network.in_flight_tasks["op-a"]["state"] == "aborted"
    assert network.in_flight_tasks["op-b"]["state"] == "aborted"
    assert network.in_flight_tasks["op-c"]["state"] == "committed"
    assert network.in_flight_tasks["op-d"]["state"] == "pending_dispatch"


@pytest.mark.asyncio
async def test_rpc_reconciliation_idempotency_returns_cached_result_without_redispatch(monkeypatch):
    network = ProductionP2PNetwork(node_id="node-test")
    network.secure_connection.active_connections["peer-a"] = object()

    dispatch_count = {"value": 0}

    # Mock httpx.AsyncClient so the HTTP RPC call succeeds
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {"success": True, "result": "ok"}

    mock_post = AsyncMock(return_value=mock_http_response)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=MagicMock(post=mock_post))
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Also suppress DB write so it doesn't fail on missing table
    monkeypatch.setattr(network, "_record_rpc_message", AsyncMock())

    task = SimpleNamespace(
        task_id="task-1",
        task_type="text_generation",
        instruction="Summarize the shard state",
        context_data={"a": 1, "b": [1, 2]},
        dependencies=[],
        expected_output_type="text",
    )

    with patch("prsm.compute.federation.enhanced_p2p_network.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        first = await network._execute_task_on_peer_rpc("peer-a", "127.0.0.1:9001", task)
        dispatch_count["value"] += 1
        second = await network._execute_task_on_peer_rpc("peer-a", "127.0.0.1:9001", task)

    assert first["success"] is True
    assert second["success"] is True
    assert first["operation_id"] == second["operation_id"]
    assert dispatch_count["value"] == 1  # second call returned from cache, no re-dispatch

