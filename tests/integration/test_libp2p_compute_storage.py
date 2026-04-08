"""
Integration tests for compute/storage lifecycle over libp2p transport.

Uses a MockLibp2pTransport that routes messages between in-process nodes
with JSON serialization fidelity (mimics the FFI boundary).
"""
import asyncio
import json
import time
import uuid
import pytest
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from prsm.node.transport import MSG_GOSSIP, MSG_DIRECT, P2PMessage, PeerConnection
from prsm.node.discovery import PeerInfo


class MockLibp2pTransport:
    """In-process transport that routes messages with JSON serialization fidelity."""

    def __init__(self, node_id: str, network: "MockNetwork"):
        self.identity = MagicMock()
        self.identity.node_id = node_id
        self._handlers: Dict[str, List[Callable]] = {}
        self._network = network
        self._peers: Dict[str, PeerConnection] = {}

    def on_message(self, msg_type: str, handler: Callable) -> None:
        self._handlers.setdefault(msg_type, []).append(handler)

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        """Send via network with JSON round-trip for serialization fidelity."""
        target = self._network.get_transport(peer_id)
        if target is None:
            return False
        wire = json.dumps({
            "msg_type": msg.msg_type,
            "sender_id": msg.sender_id,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
            "signature": msg.signature,
            "ttl": msg.ttl,
            "nonce": msg.nonce,
        })
        raw = json.loads(wire)
        reconstructed = P2PMessage(
            msg_type=raw["msg_type"],
            sender_id=raw["sender_id"],
            payload=raw["payload"],
            timestamp=raw["timestamp"],
            signature=raw["signature"],
            ttl=raw["ttl"],
            nonce=raw["nonce"],
        )
        peer = PeerConnection(peer_id=msg.sender_id, address="mock", websocket=None)
        for handler in target._handlers.get(msg.msg_type, []):
            await handler(reconstructed, peer)
        return True

    def sign(self, msg):
        pass


class MockNetwork:
    """Routes messages between MockLibp2pTransport instances."""

    def __init__(self):
        self._transports: Dict[str, MockLibp2pTransport] = {}

    def add_node(self, node_id: str) -> MockLibp2pTransport:
        t = MockLibp2pTransport(node_id, self)
        self._transports[node_id] = t
        return t

    def get_transport(self, node_id: str) -> Optional[MockLibp2pTransport]:
        return self._transports.get(node_id)


class MockGossip:
    """In-process gossip that broadcasts to all nodes with JSON fidelity."""

    def __init__(self, node_id: str, network: MockNetwork):
        self._node_id = node_id
        self._network = network
        self._callbacks: Dict[str, List[Callable]] = {}
        self.published: List[Dict[str, Any]] = []
        self.ledger = None

    def subscribe(self, subtype: str, callback: Callable) -> None:
        self._callbacks.setdefault(subtype, []).append(callback)

    async def publish(self, subtype: str, data: Dict[str, Any], ttl=None) -> int:
        wire = json.dumps({"subtype": subtype, "data": data, "sender_id": self._node_id})
        raw = json.loads(wire)

        self.published.append(raw)

        for node_id, transport in self._network._transports.items():
            gossip = _gossip_registry.get(node_id)
            if gossip:
                for cb in gossip._callbacks.get(subtype, []):
                    await cb(subtype, raw["data"], raw["sender_id"])
        return 1


_gossip_registry: Dict[str, MockGossip] = {}


def make_test_node(node_id: str, network: MockNetwork):
    """Create a test node with mock transport and gossip."""
    transport = network.add_node(node_id)
    gossip = MockGossip(node_id, network)
    _gossip_registry[node_id] = gossip
    return transport, gossip
