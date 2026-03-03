import hashlib
import json

import pytest

from nacl.signing import SigningKey, VerifyKey
from nacl.public import PrivateKey

import prsm.compute.federation.enhanced_p2p_network as enhanced_p2p_network
from prsm.compute.federation.enhanced_p2p_network import (
    DistributedHashTable,
    P2PMessage,
    ProductionP2PNetwork,
)


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


@pytest.mark.asyncio
async def test_shard_retrieve_handler_returns_error_for_missing_shard_id():
    network = ProductionP2PNetwork(node_id="node-test")
    message = P2PMessage("shard_retrieve", payload={}, sender_id="peer-a")

    response = await network._handle_shard_retrieve(message, "peer-a")

    assert response["status"] == "error"
    assert response["handler"] == "shard_retrieve"
    assert response["error_code"] == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_shard_retrieve_handler_returns_success_for_valid_shard(monkeypatch):
    network = ProductionP2PNetwork(node_id="node-test")
    shard_data_hex = b"hello-shard".hex()
    expected_hash = hashlib.sha256(b"hello-shard").hexdigest()

    async def fake_retrieve(_key: str):
        return {
            "data": shard_data_hex,
            "metadata": {"verification_hash": expected_hash, "shard_index": 0},
        }

    monkeypatch.setattr(network.dht, "retrieve", fake_retrieve)
    message = P2PMessage("shard_retrieve", payload={"shard_id": "s1"}, sender_id="peer-a")

    response = await network._handle_shard_retrieve(message, "peer-a")

    assert response["status"] == "success"
    assert response["handler"] == "shard_retrieve"
    assert response["data"]["shard_id"] == "s1"
    assert response["data"]["shard_data"] == shard_data_hex


@pytest.mark.asyncio
async def test_task_execute_handler_returns_deterministic_envelopes():
    network = ProductionP2PNetwork(node_id="node-test")

    bad = await network._handle_task_execute(P2PMessage("task_execute", {}, "peer-a"), "peer-a")
    assert bad["status"] == "error"
    assert bad["error_code"] == "INVALID_REQUEST"

    payload = {
        "task_id": "t-1",
        "task_type": "text_generation",
        "instruction": "Summarize",
        "context_data": {"x": 1},
        "dependencies": [],
        "expected_output_type": "text",
    }
    msg = P2PMessage("task_execute", payload=payload, sender_id="peer-a")
    res1 = await network._handle_task_execute(msg, "peer-a")
    res2 = await network._handle_task_execute(msg, "peer-a")

    assert res1["status"] == "success"
    assert res2["status"] == "success"
    assert res1["data"]["execution_digest"] == res2["data"]["execution_digest"]
    assert res1["data"]["result"]["deterministic_result_token"] == res2["data"]["result"]["deterministic_result_token"]


@pytest.mark.asyncio
async def test_peer_discovery_handler_and_dispatch_response_smoke(monkeypatch):
    network = ProductionP2PNetwork(node_id="node-test")

    async def fake_discover_peers(capability: str):
        assert capability == "model_execution"
        return [("peer-1", "10.0.0.1:9000")]

    sent = {}

    async def fake_send_message(peer_id, message):
        sent["peer_id"] = peer_id
        sent["type"] = message.type
        sent["payload"] = message.payload
        return True

    monkeypatch.setattr(network, "discover_peers", fake_discover_peers)
    monkeypatch.setattr(network.secure_connection, "send_message", fake_send_message)

    request = P2PMessage(
        "peer_discovery_request",
        payload={"request_id": "r-1", "capability": "model_execution", "limit": 5},
        sender_id="peer-a",
    )

    await network._handle_incoming_message(request, "peer-a")

    assert sent["peer_id"] == "peer-a"
    assert sent["type"] == "peer_discovery_response"
    assert sent["payload"]["status"] == "success"
    assert sent["payload"]["request_id"] == "r-1"
    assert sent["payload"]["data"]["count"] == 1


@pytest.mark.asyncio
async def test_handler_wiring_includes_request_paths():
    network = ProductionP2PNetwork(node_id="node-test")
    handlers = network.message_handlers

    assert "shard_retrieve" in handlers
    assert "shard_retrieve_request" in handlers
    assert "task_execute" in handlers
    assert "task_execute_request" in handlers
    assert "peer_discovery" in handlers
    assert "peer_discovery_request" in handlers


@pytest.mark.asyncio
async def test_capability_trust_gate_excludes_unsigned_records(monkeypatch):
    dht = DistributedHashTable(node_id="node-a")

    async def fake_retrieve(_key: str):
        return {
            "peers": [
                {
                    "id": "peer-unsigned",
                    "address": "127.0.0.1:7001",
                    "capability": "model_execution",
                    "verify_key": "abcd",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ]
        }

    monkeypatch.setattr(dht, "retrieve", fake_retrieve)
    result = await dht.find_peers_by_capability("model_execution")
    assert result == []


@pytest.mark.asyncio
async def test_capability_trust_gate_excludes_identity_mismatch(monkeypatch):
    dht = DistributedHashTable(node_id="node-a")

    bound_key = SigningKey.generate()
    mismatched_key = SigningKey.generate()
    peer_id = "peer-mismatch"
    dht.register_peer_identity(peer_id, bound_key.verify_key.encode().hex())

    peer_record = {
        "id": peer_id,
        "address": "127.0.0.1:7002",
        "capability": "model_execution",
        "timestamp": "2099-01-01T00:00:00+00:00",
        "verify_key": mismatched_key.verify_key.encode().hex(),
    }
    signed_payload = json.dumps(
        {
            "id": peer_record["id"],
            "address": peer_record["address"],
            "capability": peer_record["capability"],
            "timestamp": peer_record["timestamp"],
            "verify_key": peer_record["verify_key"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    peer_record["signature"] = mismatched_key.sign(signed_payload).signature.hex()

    async def fake_retrieve(_key: str):
        return {"peers": [peer_record]}

    monkeypatch.setattr(dht, "retrieve", fake_retrieve)
    result = await dht.find_peers_by_capability("model_execution")
    assert result == []


@pytest.mark.asyncio
async def test_capability_trust_gate_accepts_valid_signed_record(monkeypatch):
    dht = DistributedHashTable(node_id="node-a")

    peer_key = SigningKey.generate()
    peer_id = "peer-valid"
    verify_key_hex = peer_key.verify_key.encode().hex()
    dht.register_peer_identity(peer_id, verify_key_hex)

    peer_record = {
        "id": peer_id,
        "address": "127.0.0.1:7003",
        "capability": "model_execution",
        "timestamp": "2099-01-01T00:00:00+00:00",
        "verify_key": verify_key_hex,
    }
    signed_payload = json.dumps(
        {
            "id": peer_record["id"],
            "address": peer_record["address"],
            "capability": peer_record["capability"],
            "timestamp": peer_record["timestamp"],
            "verify_key": peer_record["verify_key"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    peer_record["signature"] = peer_key.sign(signed_payload).signature.hex()

    async def fake_retrieve(_key: str):
        return {"peers": [peer_record]}

    monkeypatch.setattr(dht, "retrieve", fake_retrieve)
    result = await dht.find_peers_by_capability("model_execution")
    assert result == [(peer_id, "127.0.0.1:7003")]
