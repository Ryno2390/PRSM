"""Phase 3 integration-test fixture: 3-node marketplace cluster.

Extends the Phase 2 Phase2TestNode / LoopbackHub fixture with the
full Phase 3 marketplace wiring:
  - MarketplaceAdvertiser on each node (so every node broadcasts).
  - MarketplaceDirectory on each node (so every node discovers).
  - EligibilityFilter + ReputationTracker per node.
  - PriceNegotiator (client) per node.
  - MarketplaceOrchestrator composing them all.

Gossip is wired through a minimal in-process `LoopbackGossipHub` that
fans out GOSSIP_MARKETPLACE_LISTING messages from any node to every
other node's subscribers. Matches the Phase 2 LoopbackHub pattern but
for gossip-topic dispatch.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Let the Phase 3 test reuse the Phase 2 loopback transport hub.
sys.path.insert(0, os.path.dirname(__file__))
from conftest_phase2 import LoopbackHub, LoopbackTransport, _LoopbackPeer  # noqa: E402

from prsm.compute.remote_dispatcher import RemoteShardDispatcher
from prsm.compute.shard_receipt import ReceiptOnlyVerification
from prsm.marketplace.advertiser import MarketplaceAdvertiser
from prsm.marketplace.directory import MarketplaceDirectory
from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.orchestrator import MarketplaceOrchestrator
from prsm.marketplace.price_handshake import PriceNegotiator
from prsm.marketplace.reputation import ReputationTracker
from prsm.node.compute_provider import ComputeProvider
from prsm.node.gossip import GOSSIP_MARKETPLACE_LISTING
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.payment_escrow import PaymentEscrow
from prsm.node.transport import MSG_DIRECT


class LoopbackGossipHub:
    """In-process gossip fan-out. Each registered node subscribes via
    `subscribe(topic, handler)`; any node's `publish(topic, data)` fans
    out to every subscriber (including the publisher itself, matching
    typical gossip semantics where the publisher sees its own message)."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, handler: Callable) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    async def publish(
        self, topic: str, data: Dict[str, Any], origin: str = "loopback",
    ) -> int:
        handlers = list(self._subscribers.get(topic, []))
        for h in handlers:
            asyncio.get_event_loop().call_soon(
                lambda h=h: asyncio.create_task(h(topic, data, origin))
            )
        return len(handlers)


class _NodeGossip:
    """Per-node gossip wrapper. Routes subscribe() into the shared hub
    tagged with this node's identity, and publish() into the hub so
    every node (including this one) sees the message."""

    def __init__(self, hub: LoopbackGossipHub, origin: str):
        self._hub = hub
        self._origin = origin

    def subscribe(self, topic: str, handler: Callable) -> None:
        self._hub.subscribe(topic, handler)

    async def publish(
        self, topic: str, data: Dict[str, Any], ttl: Optional[int] = None,
    ) -> int:
        return await self._hub.publish(topic, data, origin=self._origin)


@dataclass
class Phase3TestNode:
    """Phase 3 test node — has both the Phase 2 remote-dispatch surface
    AND the full marketplace stack (advertiser + directory + filter +
    reputation + price negotiator + orchestrator)."""
    identity: NodeIdentity
    transport: LoopbackTransport
    gossip: _NodeGossip
    ledger: LocalLedger
    payment_escrow: PaymentEscrow
    compute_provider: ComputeProvider
    remote_shard_dispatcher: RemoteShardDispatcher
    advertiser: MarketplaceAdvertiser
    directory: MarketplaceDirectory
    eligibility: EligibilityFilter
    reputation: ReputationTracker
    price_negotiator: PriceNegotiator
    orchestrator: MarketplaceOrchestrator


async def spin_up_three_node_marketplace_cluster(
    provider_prices: Optional[List[float]] = None,
    provider_tee_capable: Optional[List[bool]] = None,
) -> List[Phase3TestNode]:
    """Build 3 Phase3TestNode instances on a shared ledger + transport hub
    + gossip hub.

    Node 0 = requester (runs an advertiser too so the symmetry is
    preserved, but the test focuses on 0's orchestrator dispatching to
    1 and 2).
    Node 1 = provider B.
    Node 2 = provider C.

    provider_prices: per-node advertised price_per_shard_ftns (defaults
                     to [0.03, 0.03, 0.03] — all equal).
    provider_tee_capable: per-node tee_capable flag (defaults to all False).
    """
    prices = provider_prices or [0.03, 0.03, 0.03]
    tee_flags = provider_tee_capable or [False, False, False]
    assert len(prices) == 3 and len(tee_flags) == 3

    ledger = LocalLedger(":memory:")
    await ledger.initialize()

    transport_hub = LoopbackHub()
    gossip_hub = LoopbackGossipHub()

    nodes: List[Phase3TestNode] = []

    for i in range(3):
        identity = generate_node_identity(display_name=f"phase3-node-{i}")
        transport = transport_hub.register(identity.node_id)
        gossip = _NodeGossip(gossip_hub, origin=identity.node_id)

        await ledger.create_wallet(identity.node_id, f"node-{i}")
        escrow = PaymentEscrow(ledger=ledger, node_id=identity.node_id)

        compute_provider = ComputeProvider(
            identity=identity, transport=transport,
            gossip=gossip, ledger=ledger, max_concurrent_jobs=10,
        )
        transport.on_message(MSG_DIRECT, compute_provider._on_direct_message)

        advertiser = MarketplaceAdvertiser(
            identity=identity, gossip=gossip,
            compute_provider=compute_provider,
            capacity_shards_per_sec=5.0,
            max_shard_bytes=10 * 1024 * 1024,
            supported_dtypes=["float64"],
            price_per_shard_ftns=prices[i],
            tee_capable=tee_flags[i],
            stake_tier="standard",
            rebroadcast_interval_sec=3600.0,  # effectively disable in test
            ttl_seconds=3600,
        )
        compute_provider._marketplace_advertiser = advertiser

        directory = MarketplaceDirectory(gossip=gossip)
        reputation = ReputationTracker()
        eligibility = EligibilityFilter(reputation_tracker=reputation)

        price_negotiator = PriceNegotiator(
            identity=identity, transport=transport, default_timeout=2.0,
        )

        remote_dispatcher = RemoteShardDispatcher(
            identity=identity, transport=transport,
            payment_escrow=escrow,
            verification_strategy=ReceiptOnlyVerification(),
            default_timeout=5.0,
            max_retries=0,
            max_shard_bytes=10 * 1024 * 1024,
            local_fallback=None,
        )

        orchestrator = MarketplaceOrchestrator(
            identity=identity,
            directory=directory,
            eligibility_filter=eligibility,
            reputation=reputation,
            price_negotiator=price_negotiator,
            remote_dispatcher=remote_dispatcher,
        )

        nodes.append(Phase3TestNode(
            identity=identity, transport=transport, gossip=gossip,
            ledger=ledger, payment_escrow=escrow,
            compute_provider=compute_provider,
            remote_shard_dispatcher=remote_dispatcher,
            advertiser=advertiser,
            directory=directory,
            eligibility=eligibility,
            reputation=reputation,
            price_negotiator=price_negotiator,
            orchestrator=orchestrator,
        ))

    return nodes
