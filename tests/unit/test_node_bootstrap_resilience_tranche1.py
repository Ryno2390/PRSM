"""Tranche 1 bootstrap resilience tests for node discovery startup behavior."""

from types import SimpleNamespace

import pytest

from prsm.node.discovery import DISCOVERY_PEER_REQUEST, PeerDiscovery
from prsm.node.transport import MSG_GOSSIP, PeerConnection


class _MockTransport:
    def __init__(self, connect_results):
        self.identity = SimpleNamespace(node_id="node-local")
        self.host = "127.0.0.1"
        self.port = 19001
        self.peers = {}
        self.peer_count = 0
        self._connect_results = list(connect_results)
        self.connect_calls = []
        self.sent = []

    def on_message(self, _msg_type, _handler):
        return None

    async def connect_to_peer(self, address):
        self.connect_calls.append(address)
        if not self._connect_results:
            return None
        return self._connect_results.pop(0)

    async def send_to_peer(self, peer_id, msg):
        self.sent.append((peer_id, msg))
        return True


def _peer(peer_id: str, address: str) -> PeerConnection:
    return PeerConnection(peer_id=peer_id, address=address, websocket=object(), outbound=True)


@pytest.mark.asyncio
async def test_bootstrap_falls_back_to_secondary_candidate_when_primary_fails():
    transport = _MockTransport(connect_results=[None, _peer("peer-b", "b:9001")])
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["a:9001", "b:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    connected = await discovery.bootstrap()

    assert connected == 1
    assert transport.connect_calls == ["a:9001", "b:9001"]
    status = discovery.get_bootstrap_status()
    assert status["success_node"] == "b:9001"
    assert status["degraded_mode"] is False
    assert status["failed_nodes"] == ["a:9001"]
    assert len(transport.sent) == 1
    peer_id, message = transport.sent[0]
    assert peer_id == "peer-b"
    assert message.msg_type == MSG_GOSSIP
    assert message.payload.get("subtype") == DISCOVERY_PEER_REQUEST


@pytest.mark.asyncio
async def test_bootstrap_continues_startup_in_degraded_mode_when_all_targets_fail():
    transport = _MockTransport(connect_results=[None, None, None, None])
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["a:9001", "b:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=2,
    )

    await discovery.start()

    assert discovery._running is True
    assert len(discovery._tasks) == 2
    assert transport.connect_calls == ["a:9001", "a:9001", "b:9001", "b:9001"]
    status = discovery.get_bootstrap_status()
    assert status["connected_count"] == 0
    assert status["degraded_mode"] is True
    assert status["failed_nodes"] == ["a:9001", "b:9001"]
    assert transport.sent == []

    await discovery.stop()


@pytest.mark.asyncio
async def test_bootstrap_success_preserves_normal_peer_discovery_flow():
    transport = _MockTransport(connect_results=[_peer("peer-a", "a:9001")])
    discovery = PeerDiscovery(
        transport=transport,
        bootstrap_nodes=["a:9001"],
        bootstrap_connect_timeout=1.0,
        bootstrap_retry_attempts=1,
    )

    connected = await discovery.bootstrap()

    assert connected == 1
    status = discovery.get_bootstrap_status()
    assert status["success_node"] == "a:9001"
    assert status["degraded_mode"] is False
    assert len(transport.sent) == 1
    _, message = transport.sent[0]
    assert message.payload["subtype"] == DISCOVERY_PEER_REQUEST
    assert message.payload["max_peers"] == 20

