import json

import pytest

from nacl.public import PrivateKey
from nacl.signing import SigningKey, VerifyKey

import prsm.compute.federation.enhanced_p2p_network as enhanced_p2p_network
from prsm.compute.federation.enhanced_p2p_network import DistributedHashTable, ProductionP2PNetwork


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


def _signed_record(peer_id: str, address: str, capability: str, timestamp: str, key: SigningKey) -> dict:
    verify_key_hex = key.verify_key.encode().hex()
    record = {
        "id": peer_id,
        "address": address,
        "capability": capability,
        "timestamp": timestamp,
        "verify_key": verify_key_hex,
    }
    payload = json.dumps(
        {
            "id": record["id"],
            "address": record["address"],
            "capability": record["capability"],
            "timestamp": record["timestamp"],
            "verify_key": record["verify_key"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record["signature"] = key.sign(payload).signature.hex()
    return record


@pytest.mark.asyncio
async def test_rotation_accepted_via_explicit_update_flow(monkeypatch):
    dht = DistributedHashTable(node_id="node-a")
    peer_id = "peer-rotate"
    old_key = SigningKey.generate()
    new_key = SigningKey.generate()

    assert dht.register_peer_identity(peer_id, old_key.verify_key.encode().hex()) is True
    assert (
        dht.rotate_peer_identity_key(
            peer_id,
            previous_verify_key_hex=old_key.verify_key.encode().hex(),
            new_verify_key_hex=new_key.verify_key.encode().hex(),
        )
        is True
    )

    rotated_record = _signed_record(
        peer_id=peer_id,
        address="127.0.0.1:8001",
        capability="model_execution",
        timestamp="2099-01-01T00:00:00+00:00",
        key=new_key,
    )

    async def fake_retrieve(_key: str):
        return {"peers": [rotated_record]}

    monkeypatch.setattr(dht, "retrieve", fake_retrieve)
    peers = await dht.find_peers_by_capability("model_execution")
    assert peers == [(peer_id, "127.0.0.1:8001")]


@pytest.mark.asyncio
async def test_revoked_identity_excluded_from_routing(monkeypatch):
    dht = DistributedHashTable(node_id="node-a")
    peer_id = "peer-revoked"
    peer_key = SigningKey.generate()

    assert dht.register_peer_identity(peer_id, peer_key.verify_key.encode().hex()) is True
    assert dht.revoke_peer_identity(peer_id) is True

    peer_record = _signed_record(
        peer_id=peer_id,
        address="127.0.0.1:8002",
        capability="model_execution",
        timestamp="2099-01-01T00:00:00+00:00",
        key=peer_key,
    )

    async def fake_retrieve(_key: str):
        return {"peers": [peer_record]}

    monkeypatch.setattr(dht, "retrieve", fake_retrieve)
    peers = await dht.find_peers_by_capability("model_execution")
    assert peers == []


@pytest.mark.asyncio
async def test_unknown_identity_excluded(monkeypatch):
    dht = DistributedHashTable(node_id="node-a")
    unregistered_key = SigningKey.generate()
    peer_record = _signed_record(
        peer_id="peer-unknown",
        address="127.0.0.1:8003",
        capability="model_execution",
        timestamp="2099-01-01T00:00:00+00:00",
        key=unregistered_key,
    )

    async def fake_retrieve(_key: str):
        return {"peers": [peer_record]}

    monkeypatch.setattr(dht, "retrieve", fake_retrieve)
    peers = await dht.find_peers_by_capability("model_execution")
    assert peers == []


@pytest.mark.asyncio
async def test_stale_capability_record_excluded(monkeypatch):
    dht = DistributedHashTable(node_id="node-a")
    peer_id = "peer-stale"
    stale_key = SigningKey.generate()
    assert dht.register_peer_identity(peer_id, stale_key.verify_key.encode().hex()) is True

    stale_record = _signed_record(
        peer_id=peer_id,
        address="127.0.0.1:8004",
        capability="model_execution",
        timestamp="2020-01-01T00:00:00+00:00",
        key=stale_key,
    )

    async def fake_retrieve(_key: str):
        return {"peers": [stale_record]}

    monkeypatch.setattr(dht, "retrieve", fake_retrieve)
    peers = await dht.find_peers_by_capability("model_execution")
    assert peers == []


@pytest.mark.asyncio
async def test_trusted_signed_records_still_routable(monkeypatch):
    network = ProductionP2PNetwork(node_id="node-test")

    trusted_id = "peer-trusted"
    trusted_key = SigningKey.generate()
    network.dht.register_peer_identity(trusted_id, trusted_key.verify_key.encode().hex())

    trusted_record = _signed_record(
        peer_id=trusted_id,
        address="10.1.0.5:7777",
        capability="model_execution",
        timestamp="2099-01-01T00:00:00+00:00",
        key=trusted_key,
    )

    async def fake_retrieve(_key: str):
        return {"peers": [trusted_record]}

    monkeypatch.setattr(network.dht, "retrieve", fake_retrieve)
    peers = await network.dht.find_peers_by_capability("model_execution")
    assert peers == [(trusted_id, "10.1.0.5:7777")]


@pytest.mark.asyncio
async def test_routing_candidate_selection_smoke_after_lifecycle_ops(monkeypatch):
    network = ProductionP2PNetwork(node_id="node-test")

    trusted_id = "peer-ok"
    revoked_id = "peer-revoked"
    unknown_id = "peer-unknown"

    trusted_key = SigningKey.generate()
    revoked_key = SigningKey.generate()
    unknown_key = SigningKey.generate()

    network.dht.register_peer_identity(trusted_id, trusted_key.verify_key.encode().hex())
    network.dht.register_peer_identity(revoked_id, revoked_key.verify_key.encode().hex())
    network.dht.revoke_peer_identity(revoked_id)

    trusted_record = _signed_record(
        peer_id=trusted_id,
        address="10.0.0.11:9000",
        capability="model_execution",
        timestamp="2099-01-01T00:00:00+00:00",
        key=trusted_key,
    )
    revoked_record = _signed_record(
        peer_id=revoked_id,
        address="10.0.0.12:9000",
        capability="model_execution",
        timestamp="2099-01-01T00:00:00+00:00",
        key=revoked_key,
    )
    unknown_record = _signed_record(
        peer_id=unknown_id,
        address="10.0.0.13:9000",
        capability="model_execution",
        timestamp="2099-01-01T00:00:00+00:00",
        key=unknown_key,
    )

    async def fake_retrieve(_key: str):
        return {"peers": [trusted_record, revoked_record, unknown_record]}

    monkeypatch.setattr(network.dht, "retrieve", fake_retrieve)
    selected = await network._find_execution_peers_via_dht(task=None)

    assert selected == [(trusted_id, "10.0.0.11:9000")]

