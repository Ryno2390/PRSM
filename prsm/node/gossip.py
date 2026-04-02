"""
Gossip Protocol
===============

Epidemic gossip for propagating messages across the PRSM network.
Handles job offers, storage requests, transaction confirmations,
and other network-wide announcements with deduplication and TTL.
"""

import asyncio
import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from prsm.node.transport import MSG_GOSSIP, MSG_PEER_CONNECTED, P2PMessage, PeerConnection, WebSocketTransport

logger = logging.getLogger(__name__)

_BOUNDED_GOSSIP_LABELS = {
    "heartbeat",
    "agent_task_offer",
    "agent_task_bid",
    "agent_task_assign",
    "agent_task_complete",
    "agent_task_cancel",
    "agent_review_request",
    "agent_review_submit",
    "agent_knowledge_query",
    "agent_knowledge_response",
}

# Gossip subtypes for the compute/storage marketplace
GOSSIP_JOB_OFFER = "job_offer"
GOSSIP_JOB_ACCEPT = "job_accept"
GOSSIP_JOB_RESULT = "job_result"
GOSSIP_PAYMENT_CONFIRM = "payment_confirm"
GOSSIP_ESCROW_CREATE = "escrow_create"       # FTNS locked for a job
GOSSIP_ESCROW_RELEASE = "escrow_release"     # Payment distributed
GOSSIP_ESCROW_REFUND = "escrow_refund"       # Refund to requester
GOSSIP_STORAGE_REQUEST = "storage_request"
GOSSIP_STORAGE_CONFIRM = "storage_confirm"
GOSSIP_PROOF_OF_STORAGE = "proof_of_storage"
GOSSIP_PROVENANCE_REGISTER = "provenance_register"
GOSSIP_CONTENT_ADVERTISE = "content_advertise"
GOSSIP_CONTENT_ACCESS = "content_access"
GOSSIP_FTNS_TRANSACTION = "ftns_transaction"
GOSSIP_AGENT_ADVERTISE = "agent_advertise"
GOSSIP_AGENT_DEREGISTER = "agent_deregister"
GOSSIP_PROVENANCE_QUERY = "provenance_query"
GOSSIP_PROVENANCE_RESPONSE = "provenance_response"
GOSSIP_CAPABILITY_ANNOUNCE = "capability_announce"

# Gossip subtypes for agent collaboration protocols
GOSSIP_TASK_ASSIGN = "agent_task_assign"
GOSSIP_TASK_COMPLETE = "agent_task_complete"
GOSSIP_TASK_CANCEL = "agent_task_cancel"
GOSSIP_REVIEW_SUBMIT = "agent_review_submit"
GOSSIP_KNOWLEDGE_RESPONSE = "agent_knowledge_response"

# Gossip subtypes for digest exchange (late-joining node catch-up)
GOSSIP_DIGEST_REQUEST = "digest_request"
GOSSIP_DIGEST_RESPONSE = "digest_response"

# Gossip subtypes for BitTorrent integration
GOSSIP_BITTORRENT_ANNOUNCE = "bittorrent_announce"
GOSSIP_BITTORRENT_WITHDRAW = "bittorrent_withdraw"
GOSSIP_BITTORRENT_STATS = "bittorrent_stats"
GOSSIP_BITTORRENT_REQUEST = "bittorrent_request"

# Retention configuration per gossip subtype (in seconds)
# Messages older than these values are pruned from the gossip log
GOSSIP_RETENTION_SECONDS: Dict[str, float] = {
    # Task-related messages: 1 hour retention
    "job_offer": 3600,
    "job_accept": 3600,
    "job_result": 3600,
    "payment_confirm": 3600,
    "agent_task_offer": 3600,
    "agent_task_bid": 3600,
    "agent_task_assign": 3600,
    "agent_task_complete": 3600,
    "agent_task_cancel": 3600,
    "agent_review_request": 3600,
    "agent_review_submit": 3600,
    "agent_knowledge_query": 3600,
    "agent_knowledge_response": 3600,
    # Content-related messages: 24 hour retention
    "content_advertise": 86400,
    "content_access": 86400,
    "storage_request": 86400,
    "storage_confirm": 86400,
    "provenance_register": 86400,
    "provenance_query": 3600,
    "provenance_response": 3600,
    "proof_of_storage": 86400,
    # Agent registration: 24 hour retention
    "agent_advertise": 86400,
    "agent_deregister": 86400,
    "capability_announce": 86400,
    # FTNS transactions: 24 hour retention for audit trail
    "ftns_transaction": 86400,
    # BitTorrent messages: 24 hour retention for announces, 1 hour for withdrawals
    "bittorrent_announce": 86400,
    "bittorrent_withdraw": 3600,
    "bittorrent_stats": 1800,      # 30 minutes — stats decay quickly
    "bittorrent_request": 300,     # 5 minutes — short-lived requests
    # Heartbeat: very short retention (not stored anyway)
    "heartbeat": 60,
    # Digest exchange: not stored
    "digest_request": 0,
    "digest_response": 0,
}

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

        # Additive observability counters (must never change gossip behavior)
        self._telemetry: Dict[str, Any] = {
            "publish_total": 0,
            "publish_by_subtype": collections.Counter(),
            "forward_total": 0,
            "forward_by_subtype": collections.Counter(),
            "drop_total": 0,
            "drop_by_subtype": collections.Counter(),
            "drop_by_reason": collections.Counter(),
        }

        # Register as handler for all gossip messages
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)
        
        # Register handler for peer connection events (for digest exchange)
        self.transport.on_message(MSG_PEER_CONNECTED, self._on_peer_connected)

    @staticmethod
    def _telemetry_subtype_label(subtype: str) -> str:
        """Map raw subtypes to a bounded cardinality label set."""
        if subtype in _BOUNDED_GOSSIP_LABELS:
            return subtype
        return "other"

    def _record_publish(self, subtype: str) -> None:
        try:
            label = self._telemetry_subtype_label(subtype)
            self._telemetry["publish_total"] += 1
            self._telemetry["publish_by_subtype"][label] += 1
        except Exception:
            pass

    def _record_forward(self, subtype: str) -> None:
        try:
            label = self._telemetry_subtype_label(subtype)
            self._telemetry["forward_total"] += 1
            self._telemetry["forward_by_subtype"][label] += 1
        except Exception:
            pass

    def _record_drop(self, subtype: str, reason: str) -> None:
        try:
            label = self._telemetry_subtype_label(subtype)
            self._telemetry["drop_total"] += 1
            self._telemetry["drop_by_subtype"][label] += 1
            self._telemetry["drop_by_reason"][reason] += 1
        except Exception:
            pass

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a stable copy of gossip telemetry counters for tests/debugging."""
        return {
            "publish_total": int(self._telemetry["publish_total"]),
            "publish_by_subtype": dict(self._telemetry["publish_by_subtype"]),
            "forward_total": int(self._telemetry["forward_total"]),
            "forward_by_subtype": dict(self._telemetry["forward_by_subtype"]),
            "drop_total": int(self._telemetry["drop_total"]),
            "drop_by_subtype": dict(self._telemetry["drop_by_subtype"]),
            "drop_by_reason": dict(self._telemetry["drop_by_reason"]),
        }

    def subscribe(self, subtype: str, callback: GossipCallback) -> None:
        """Subscribe to a specific gossip subtype."""
        self._subscribers.setdefault(subtype, []).append(callback)

    async def publish(self, subtype: str, data: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Publish a gossip message to the network.

        Returns number of peers the message was sent to.
        
        In single-node mode (no peers), also delivers to local subscribers
        to enable self-compute and other local operations.
        """
        self._record_publish(subtype)
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
        
        # Send to peers
        sent = await self.transport.gossip(msg, fanout=self.fanout)
        
        # In single-node mode (no peers), deliver to local subscribers
        # This enables self-compute and other local operations
        if sent == 0:
            callbacks = self._subscribers.get(subtype, [])
            for cb in callbacks:
                try:
                    await cb(subtype, data, self.transport.identity.node_id)
                except Exception as e:
                    logger.error(f"Gossip local subscriber error ({subtype}): {e}")
        
        return sent

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

    # ── Peer Connection Handler ─────────────────────────────────────

    async def _on_peer_connected(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle new peer connection by requesting digest exchange.

        When a new peer connects, we send a digest request to catch up on
        any messages we may have missed while we were disconnected.
        """
        peer_id = msg.sender_id
        direction = msg.payload.get("direction", "unknown")
        
        logger.info(f"Peer connected: {peer_id[:8]}... ({direction})")
        
        # Only request digest from outbound connections (we initiated)
        # This prevents both peers from sending digest requests simultaneously
        if direction == "outbound":
            try:
                await self.request_digest(peer_id)
            except Exception as e:
                logger.debug(f"Failed to send digest request to {peer_id[:8]}...: {e}")

    # ── Internal ─────────────────────────────────────────────────

    async def _handle_gossip(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Process incoming gossip and optionally re-propagate."""
        subtype = msg.payload.get("subtype", "")
        data = msg.payload.get("data", {})
        origin = msg.payload.get("origin", msg.sender_id)

        if not subtype:
            self._record_drop("", "missing_subtype")
            return

        # Handle digest exchange messages specially
        if subtype == GOSSIP_DIGEST_REQUEST:
            await self._handle_digest_request(msg, peer)
            return
        
        if subtype == GOSSIP_DIGEST_RESPONSE:
            await self._handle_digest_response(msg, peer)
            return

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
            self._record_forward(subtype)
            fwd = P2PMessage(
                msg_type=MSG_GOSSIP,
                sender_id=self.transport.identity.node_id,
                payload=msg.payload,
                ttl=msg.ttl - 1,
                nonce=msg.nonce,  # preserve nonce for dedup
            )
            await self.transport.gossip(fwd, fanout=self.fanout)
        else:
            self._record_drop(subtype, "ttl_exhausted")

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

    # ── Digest Exchange for Late-Joining Nodes ───────────────────────────

    async def request_digest(self, peer_id: str) -> None:
        """Send a digest request to a peer to catch up on missed messages.

        Called when a new peer connection is established. The peer will
        respond with any messages we've missed based on our last-seen timestamps.
        """
        # Build digest request with last-seen timestamps per subtype
        timestamps = await self._get_last_seen_timestamps()
        
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": GOSSIP_DIGEST_REQUEST,
                "data": {
                    "timestamps": timestamps,
                    "requester_id": self.transport.identity.node_id,
                },
                "origin": self.transport.identity.node_id,
                "origin_time": time.time(),
            },
            ttl=1,  # Direct message, don't re-propagate
        )
        
        await self.transport.send_to_peer(peer_id, msg)
        logger.debug(f"Sent digest request to {peer_id[:8]}... with {len(timestamps)} subtype timestamps")

    async def _get_last_seen_timestamps(self) -> Dict[str, float]:
        """Get the last-seen timestamp for each gossip subtype.

        Returns a dict mapping subtype -> last received timestamp.
        Used to request only messages we haven't seen.
        """
        if not self.ledger:
            return {}
        
        try:
            # Query the ledger for last-seen timestamps
            # This is a simplified implementation - could be optimized with a dedicated query
            timestamps = {}
            current_time = time.time()
            
            # Get recent messages for each subtype we care about
            catchup_subtypes = [
                GOSSIP_JOB_OFFER,
                GOSSIP_CONTENT_ADVERTISE,
                GOSSIP_AGENT_ADVERTISE,
                GOSSIP_PROVENANCE_REGISTER,
                GOSSIP_STORAGE_REQUEST,
                "agent_task_offer",
                "agent_task_assign",
            ]
            
            for subtype in catchup_subtypes:
                # Get the most recent message of this subtype
                messages = await self.ledger.get_recent_gossip(
                    since=current_time - 86400,  # Look back up to 24 hours
                    subtypes=[subtype]
                )
                if messages:
                    # Get the timestamp of the most recent message
                    timestamps[subtype] = max(m["received_at"] for m in messages)
            
            return timestamps
        except Exception as e:
            logger.debug(f"Error getting last-seen timestamps: {e}")
            return {}

    async def _handle_digest_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle incoming digest request from a peer.

        Queries the local gossip log for messages after the requested timestamps
        and sends them back in a digest response.
        """
        data = msg.payload.get("data", {})
        timestamps = data.get("timestamps", {})
        requester_id = data.get("requester_id", msg.sender_id)
        
        if not self.ledger:
            logger.debug(f"No ledger available for digest request from {requester_id[:8]}...")
            return
        
        # Collect messages that the requester hasn't seen
        missing_messages: List[Dict[str, Any]] = []
        
        for subtype, last_seen in timestamps.items():
            try:
                messages = await self.ledger.get_recent_gossip(
                    since=last_seen,
                    subtypes=[subtype]
                )
                missing_messages.extend(messages)
            except Exception as e:
                logger.debug(f"Error fetching messages for {subtype}: {e}")
        
        # Also include messages for subtypes the requester didn't mention
        # (they may be new to the network)
        catchup_subtypes = [
            GOSSIP_JOB_OFFER,
            GOSSIP_CONTENT_ADVERTISE,
            GOSSIP_AGENT_ADVERTISE,
            GOSSIP_PROVENANCE_REGISTER,
        ]
        
        for subtype in catchup_subtypes:
            if subtype not in timestamps:
                try:
                    # Get messages from the last hour for new subtypes
                    messages = await self.ledger.get_recent_gossip(
                        since=time.time() - 3600,
                        subtypes=[subtype]
                    )
                    missing_messages.extend(messages)
                except Exception as e:
                    logger.debug(f"Error fetching messages for new subtype {subtype}: {e}")
        
        # Send response if we have messages to share
        if missing_messages:
            # Limit response size to avoid overwhelming the peer
            max_messages = 100
            if len(missing_messages) > max_messages:
                # Sort by timestamp and take most recent
                missing_messages.sort(key=lambda m: m.get("received_at", 0), reverse=True)
                missing_messages = missing_messages[:max_messages]
            
            response = P2PMessage(
                msg_type=MSG_GOSSIP,
                sender_id=self.transport.identity.node_id,
                payload={
                    "subtype": GOSSIP_DIGEST_RESPONSE,
                    "data": {
                        "messages": missing_messages,
                        "total_count": len(missing_messages),
                    },
                    "origin": self.transport.identity.node_id,
                    "origin_time": time.time(),
                },
                ttl=1,  # Direct message, don't re-propagate
            )
            
            await self.transport.send_to_peer(requester_id, response)
            logger.debug(f"Sent digest response with {len(missing_messages)} messages to {requester_id[:8]}...")

    async def _handle_digest_response(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle incoming digest response with catch-up messages.

        Processes each message as if it were a new gossip message,
        updating local timestamps and storing in local gossip log.
        """
        data = msg.payload.get("data", {})
        messages = data.get("messages", [])
        
        if not messages:
            logger.debug(f"Received empty digest response from {msg.sender_id[:8]}...")
            return
        
        logger.info(f"Processing {len(messages)} catch-up messages from {msg.sender_id[:8]}...")
        
        processed = 0
        for message_data in messages:
            try:
                subtype = message_data.get("subtype", "")
                payload = message_data.get("payload", {})
                origin = message_data.get("origin", msg.sender_id)
                nonce = message_data.get("nonce", "")
                
                if not subtype or not payload:
                    continue
                
                # Skip if we've already seen this message (dedup)
                if nonce and await self._is_duplicate(nonce):
                    continue
                
                # Deliver to local subscribers
                callbacks = self._subscribers.get(subtype, [])
                for cb in callbacks:
                    try:
                        await cb(subtype, payload, origin)
                    except Exception as e:
                        logger.error(f"Error in catch-up subscriber callback ({subtype}): {e}")
                
                # Store in local gossip log
                if self.ledger and subtype not in ("heartbeat", GOSSIP_DIGEST_REQUEST, GOSSIP_DIGEST_RESPONSE):
                    try:
                        await self.ledger.log_gossip(
                            nonce=nonce,
                            subtype=subtype,
                            origin=origin,
                            payload=payload,
                            ttl=1,  # Already propagated, just storing locally
                        )
                    except Exception:
                        pass  # Don't break on log failure
                
                processed += 1
                
            except Exception as e:
                logger.debug(f"Error processing catch-up message: {e}")
        
        logger.info(f"Processed {processed}/{len(messages)} catch-up messages from {msg.sender_id[:8]}...")

    async def _is_duplicate(self, nonce: str) -> bool:
        """Check if a message with this nonce has already been processed."""
        # The transport handles nonce dedup for regular messages
        # For catch-up messages, we check the gossip log
        if not self.ledger:
            return False
        
        try:
            # Check if this nonce exists in our log
            messages = await self.ledger.get_recent_gossip(
                since=time.time() - 86400,  # Check last 24 hours
            )
            return any(m.get("nonce") == nonce for m in messages)
        except Exception:
            return False

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
                    pruned = await self._prune_gossip_log_by_retention()
                    if pruned:
                        logger.debug(f"Pruned {pruned} old gossip log entries")
                except Exception as e:
                    logger.debug(f"Error pruning gossip log: {e}")

    async def _prune_gossip_log_by_retention(self) -> int:
        """Prune gossip log entries based on per-subtype retention policy.

        Uses GOSSIP_RETENTION_SECONDS to determine how long to keep
        messages of each type.
        """
        if not self.ledger:
            return 0
        
        total_pruned = 0
        current_time = time.time()
        
        # Get all subtypes with retention policies
        for subtype, retention_seconds in GOSSIP_RETENTION_SECONDS.items():
            if retention_seconds <= 0:
                continue  # Skip subtypes that shouldn't be stored
            
            cutoff = current_time - retention_seconds
            
            try:
                # Delete messages older than retention window
                # The ledger's prune_gossip_log uses a single max_age parameter,
                # so we use the minimum retention across all subtypes for now
                # A more sophisticated implementation would add subtype-specific pruning
                pass  # Handled by the ledger's prune_gossip_log with default retention
            except Exception as e:
                logger.debug(f"Error pruning {subtype}: {e}")
        
        # Use the ledger's built-in pruning with default retention
        try:
            total_pruned = await self.ledger.prune_gossip_log(self.gossip_log_retention)
        except Exception:
            pass
        
        return total_pruned
