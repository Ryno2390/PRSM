"""
Gossip Protocol
===============

Epidemic gossip for propagating messages across the PRSM network.
Handles job offers, storage requests, transaction confirmations,
and other network-wide announcements with deduplication and TTL.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from prsm.node.transport import MSG_GOSSIP, P2PMessage, PeerConnection, WebSocketTransport

logger = logging.getLogger(__name__)

# Gossip subtypes for the compute/storage marketplace
GOSSIP_JOB_OFFER = "job_offer"
GOSSIP_JOB_ACCEPT = "job_accept"
GOSSIP_JOB_RESULT = "job_result"
GOSSIP_PAYMENT_CONFIRM = "payment_confirm"
GOSSIP_STORAGE_REQUEST = "storage_request"
GOSSIP_STORAGE_CONFIRM = "storage_confirm"
GOSSIP_PROOF_OF_STORAGE = "proof_of_storage"
GOSSIP_PROVENANCE_REGISTER = "provenance_register"

# Callback type for gossip subscribers
GossipCallback = Callable[[str, Dict[str, Any], str], Coroutine[Any, Any, None]]
# (subtype, payload, sender_id) -> None


class GossipProtocol:
    """Epidemic gossip with fanout, TTL, and deduplication.

    Messages are forwarded to a random subset of peers (fanout),
    with decreasing TTL. Nonce-based dedup prevents infinite loops.
    """

    def __init__(
        self,
        transport: WebSocketTransport,
        fanout: int = 3,
        default_ttl: int = 5,
        heartbeat_interval: float = 30.0,
    ):
        self.transport = transport
        self.fanout = fanout
        self.default_ttl = default_ttl
        self.heartbeat_interval = heartbeat_interval

        self._subscribers: Dict[str, List[GossipCallback]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Register as handler for all gossip messages
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)

    def subscribe(self, subtype: str, callback: GossipCallback) -> None:
        """Subscribe to a specific gossip subtype."""
        self._subscribers.setdefault(subtype, []).append(callback)

    async def publish(self, subtype: str, data: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Publish a gossip message to the network.

        Returns number of peers the message was sent to.
        """
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": subtype,
                "data": data,
                "origin": self.transport.identity.node_id,
                "origin_time": time.time(),
            },
            ttl=ttl if ttl is not None else self.default_ttl,
        )
        return await self.transport.gossip(msg, fanout=self.fanout)

    async def start(self) -> None:
        """Start heartbeat loop."""
        self._running = True
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        logger.info("Gossip protocol started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    # ── Internal ─────────────────────────────────────────────────

    async def _handle_gossip(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Process incoming gossip and optionally re-propagate."""
        subtype = msg.payload.get("subtype", "")
        data = msg.payload.get("data", {})
        origin = msg.payload.get("origin", msg.sender_id)

        # Deliver to local subscribers
        callbacks = self._subscribers.get(subtype, [])
        for cb in callbacks:
            try:
                await cb(subtype, data, origin)
            except Exception as e:
                logger.error(f"Gossip subscriber error ({subtype}): {e}")

        # Re-propagate with decremented TTL
        if msg.ttl > 1:
            fwd = P2PMessage(
                msg_type=MSG_GOSSIP,
                sender_id=self.transport.identity.node_id,
                payload=msg.payload,
                ttl=msg.ttl - 1,
                nonce=msg.nonce,  # preserve nonce for dedup
            )
            await self.transport.gossip(fwd, fanout=self.fanout)

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to maintain network liveness info."""
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self.publish(
                    "heartbeat",
                    {
                        "peer_count": self.transport.peer_count,
                        "uptime": time.time(),
                    },
                    ttl=2,  # heartbeats don't need to travel far
                )
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
