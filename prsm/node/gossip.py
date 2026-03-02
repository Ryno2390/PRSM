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
GOSSIP_CONTENT_ADVERTISE = "content_advertise"
GOSSIP_CONTENT_ACCESS = "content_access"
GOSSIP_FTNS_TRANSACTION = "ftns_transaction"
GOSSIP_AGENT_ADVERTISE = "agent_advertise"
GOSSIP_AGENT_DEREGISTER = "agent_deregister"

# Gossip subtypes for agent collaboration protocols
GOSSIP_TASK_ASSIGN = "agent_task_assign"
GOSSIP_TASK_COMPLETE = "agent_task_complete"
GOSSIP_TASK_CANCEL = "agent_task_cancel"
GOSSIP_REVIEW_SUBMIT = "agent_review_submit"
GOSSIP_KNOWLEDGE_RESPONSE = "agent_knowledge_response"

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
        gossip_log_retention: float = 3600.0,
    ):
        self.transport = transport
        self.fanout = fanout
        self.default_ttl = default_ttl
        self.heartbeat_interval = heartbeat_interval
        self.gossip_log_retention = gossip_log_retention

        self._subscribers: Dict[str, List[GossipCallback]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Ledger for gossip persistence (set post-construction by node.py)
        self.ledger: Optional[Any] = None

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

        # Persist to gossip log (skip heartbeats — too frequent, low value)
        if self.ledger and subtype != "heartbeat":
            try:
                await self.ledger.log_gossip(
                    nonce=msg.nonce,
                    subtype=subtype,
                    origin=origin,
                    payload=data,
                    ttl=msg.ttl,
                )
            except Exception:
                pass  # Fire-and-forget; don't break gossip on log failure

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

    async def get_catchup_messages(
        self,
        since: float,
        subtypes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return persisted gossip messages received after *since*.

        Used by newly-connected peers to catch up on missed state changes.
        """
        if not self.ledger:
            return []
        return await self.ledger.get_recent_gossip(since, subtypes)

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to maintain network liveness info."""
        prune_counter = 0
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

            # Prune old gossip log entries every ~10 heartbeats
            prune_counter += 1
            if prune_counter >= 10 and self.ledger:
                prune_counter = 0
                try:
                    pruned = await self.ledger.prune_gossip_log(self.gossip_log_retention)
                    if pruned:
                        logger.debug(f"Pruned {pruned} old gossip log entries")
                except Exception:
                    pass
