"""Helpers for Phase 2 integration tests: 3-node loopback cluster.

Spins up three lightweight nodes sharing a single in-process transport
hub and a single LocalLedger. Each node carries the minimal surface
needed for the remote-shard flow: identity, loopback transport, shared
ledger, PaymentEscrow, ComputeProvider, RemoteShardDispatcher.

This skips the full PRSMNode bootstrap (hundreds of optional deps)
and keeps the test stripped to the remote-shard acceptance criterion.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

from prsm.compute.remote_dispatcher import RemoteShardDispatcher
from prsm.compute.shard_receipt import ReceiptOnlyVerification
from prsm.node.compute_provider import ComputeProvider
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.payment_escrow import PaymentEscrow
from prsm.node.transport import MSG_DIRECT, P2PMessage


class _LoopbackPeer:
    """Stub peer object matching PeerConnection's duck-typed contract —
    only `.peer_id` is read by handlers in the Phase 2 flow."""
    def __init__(self, node_id: str):
        self.peer_id = node_id


class LoopbackTransport:
    """In-process transport: routes P2PMessages between nodes in the hub.

    Implements the minimum interface RemoteShardDispatcher + ComputeProvider
    need: on_message(msg_type, handler), send_to_peer(peer_id, msg),
    get_peer(node_id).
    """

    def __init__(self, node_id: str, hub: "LoopbackHub"):
        self.node_id = node_id
        self.hub = hub
        self._handlers: Dict[str, List[Callable]] = {}

    def on_message(self, msg_type: str, handler: Callable) -> None:
        self._handlers.setdefault(msg_type, []).append(handler)

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        target = self.hub.get_transport(peer_id)
        if target is None:
            return False
        # Deliver in the next event-loop tick to match real-network
        # async semantics (sender shouldn't block on receiver handlers).
        asyncio.get_event_loop().call_soon(
            lambda: asyncio.create_task(target._deliver(msg, self.node_id))
        )
        return True

    def get_peer(self, node_id: str) -> Optional[_LoopbackPeer]:
        if self.hub.get_transport(node_id) is None:
            return None
        return _LoopbackPeer(node_id)

    async def _deliver(self, msg: P2PMessage, sender_node_id: str) -> None:
        peer = _LoopbackPeer(sender_node_id)
        for handler in self._handlers.get(msg.msg_type, []):
            try:
                await handler(msg, peer)
            except Exception:
                pass


class LoopbackHub:
    """Registry of node_id → LoopbackTransport for in-process delivery."""

    def __init__(self) -> None:
        self._transports: Dict[str, LoopbackTransport] = {}

    def register(self, node_id: str) -> LoopbackTransport:
        tx = LoopbackTransport(node_id=node_id, hub=self)
        self._transports[node_id] = tx
        return tx

    def get_transport(self, node_id: str) -> Optional[LoopbackTransport]:
        return self._transports.get(node_id)


@dataclass
class Phase2TestNode:
    """Minimal node surface for Phase 2 integration tests."""
    identity: NodeIdentity
    transport: LoopbackTransport
    ledger: LocalLedger
    payment_escrow: PaymentEscrow
    compute_provider: Optional[ComputeProvider] = None
    remote_shard_dispatcher: Optional[RemoteShardDispatcher] = None


async def spin_up_three_node_cluster() -> List[Phase2TestNode]:
    """Build three Phase2TestNode instances on a shared ledger + hub.

    Node 0 = requester, Node 1 = provider B, Node 2 = provider C. All
    three have ComputeProvider handlers wired so any of them can serve
    shard_execute_request (the test only exercises B and C as providers,
    but the wiring is symmetric).
    """
    ledger = LocalLedger(":memory:")
    await ledger.initialize()

    hub = LoopbackHub()
    nodes: List[Phase2TestNode] = []
    gossip_stub = MagicMock()
    gossip_stub.subscribe = MagicMock()
    gossip_stub.publish = MagicMock()

    for i in range(3):
        identity = generate_node_identity(display_name=f"test-node-{i}")
        transport = hub.register(identity.node_id)

        await ledger.create_wallet(identity.node_id, f"node-{i}")

        escrow = PaymentEscrow(ledger=ledger, node_id=identity.node_id)

        provider = ComputeProvider(
            identity=identity,
            transport=transport,
            gossip=gossip_stub,
            ledger=ledger,
            max_concurrent_jobs=10,
        )
        # Register the MSG_DIRECT handler without calling full start()
        # (which touches gossip subscriptions we don't exercise here).
        transport.on_message(MSG_DIRECT, provider._on_direct_message)

        dispatcher = RemoteShardDispatcher(
            identity=identity,
            transport=transport,
            payment_escrow=escrow,
            verification_strategy=ReceiptOnlyVerification(),
            default_timeout=5.0,
            max_retries=0,
            max_shard_bytes=10 * 1024 * 1024,
            local_fallback=None,
        )

        nodes.append(Phase2TestNode(
            identity=identity,
            transport=transport,
            ledger=ledger,
            payment_escrow=escrow,
            compute_provider=provider,
            remote_shard_dispatcher=dispatcher,
        ))

    return nodes
