"""
libp2p GossipSub Wrapper
========================

Thin wrapper around ``Libp2pTransport`` that exposes the same public API
as ``GossipProtocol`` (prsm/node/gossip.py).

The key difference from the WebSocket-based GossipProtocol is that
GossipSub topic subscription is *lazy*: ``PrsmSubscribe`` is only called
once per unique subtype, the first time a callback is registered for it.
Subsequent callbacks for the same subtype reuse the existing subscription.

Message envelope format (JSON, published to GossipSub topic):
    {
        "subtype":   str,   # e.g. "job_offer"
        "data":      dict,
        "sender_id": str,
        "timestamp": float
    }
"""

import json
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from prsm.node.transport import MSG_GOSSIP, P2PMessage

logger = logging.getLogger(__name__)

# Re-export the same callback type as gossip.py
GossipCallback = Callable[[str, Dict[str, Any], str], Coroutine[Any, Any, None]]
# (subtype, payload, sender_id) -> None


class Libp2pGossip:
    """GossipSub-backed gossip layer for Libp2pTransport.

    Provides the same public interface as ``GossipProtocol`` so higher-level
    code (node.py, tests) can swap implementations without changes.
    """

    def __init__(self, transport: Any, **kwargs: Any) -> None:
        """
        Args:
            transport: A ``Libp2pTransport`` instance.
            **kwargs:  Ignored (for drop-in compatibility with GossipProtocol).
        """
        self.transport = transport

        # Set post-construction by node.py (same pattern as GossipProtocol)
        self.ledger: Optional[Any] = None

        # subtype -> list of callbacks
        self._callbacks: Dict[str, List[GossipCallback]] = {}

        # Subtypes for which PrsmSubscribe has already been called
        self._subscribed_topics: set = set()

        # Additive telemetry counters (never change gossip behaviour)
        self._telemetry: Dict[str, int] = {
            "publish_total": 0,
            "deliver_total": 0,
            "error_total": 0,
        }

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Register handler for inbound gossip messages from the transport."""
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)
        logger.info("Libp2pGossip started")

    async def stop(self) -> None:
        """No-op — GossipSub shuts down with the libp2p host."""

    # ── Public API (mirrors GossipProtocol) ──────────────────────

    def subscribe(self, subtype: str, callback: GossipCallback) -> None:
        """Register a callback for *subtype*.

        Lazily calls ``PrsmSubscribe`` the first time a subtype is registered.
        """
        topic = self._topic_name(subtype)

        # Lazy subscription: only call PrsmSubscribe once per topic
        if topic not in self._subscribed_topics:
            try:
                self.transport._lib.PrsmSubscribe(
                    self.transport._handle,
                    topic.encode("utf-8"),
                )
            except Exception as exc:
                logger.warning("PrsmSubscribe failed for topic %s: %s", topic, exc)
            self._subscribed_topics.add(topic)

        self._callbacks.setdefault(subtype, []).append(callback)

    async def publish(
        self,
        subtype: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """Publish a gossip message over GossipSub.

        Args:
            subtype: Message subtype label (e.g. ``"job_offer"``).
            data:    Arbitrary payload dict.
            ttl:     Ignored in GossipSub mode (kept for API compatibility).

        Returns:
            1 on success, 0 on failure.
        """
        topic = self._topic_name(subtype)
        sender_id = self.transport.identity.node_id

        envelope = {
            "subtype": subtype,
            "data": data,
            "sender_id": sender_id,
            "timestamp": time.time(),
        }

        payload_bytes = json.dumps(envelope).encode("utf-8")

        rc = 0
        try:
            rc = self.transport._lib.PrsmPublish(
                self.transport._handle,
                topic.encode("utf-8"),
                payload_bytes,
                len(payload_bytes),
            )
        except Exception as exc:
            logger.error("PrsmPublish error (%s): %s", subtype, exc)
            self._telemetry["error_total"] += 1
            return 0

        if rc == 0:
            self._telemetry["publish_total"] += 1

            # Optional ledger persistence (fire-and-forget)
            if self.ledger and subtype != "heartbeat":
                try:
                    await self.ledger.log_gossip(
                        nonce="",
                        subtype=subtype,
                        origin=sender_id,
                        payload=data,
                        ttl=1,
                    )
                except Exception:
                    pass

            return 1

        self._telemetry["error_total"] += 1
        return 0

    async def get_catchup_messages(
        self,
        since: float,
        subtypes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Delegate to the ledger when available (same as GossipProtocol)."""
        if not self.ledger:
            return []
        return await self.ledger.get_recent_gossip(since, subtypes)

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a stable copy of gossip telemetry counters."""
        return {
            "publish_total": int(self._telemetry["publish_total"]),
            "deliver_total": int(self._telemetry["deliver_total"]),
            "error_total": int(self._telemetry["error_total"]),
            "subscribed_topics": list(self._subscribed_topics),
        }

    # ── Internal ─────────────────────────────────────────────────

    async def _handle_gossip(
        self, msg: P2PMessage, peer_info: Any
    ) -> None:
        """Dispatch an inbound gossip message to registered callbacks."""
        try:
            payload = msg.payload
            if isinstance(payload, (str, bytes)):
                payload = json.loads(payload)

            subtype = payload.get("subtype", "")
            data = payload.get("data", {})
            sender_id = payload.get("sender_id", msg.sender_id)
        except Exception as exc:
            logger.debug("Failed to parse gossip envelope: %s", exc)
            return

        if not subtype:
            return

        callbacks = self._callbacks.get(subtype, [])
        for cb in callbacks:
            try:
                await cb(subtype, data, sender_id)
                self._telemetry["deliver_total"] += 1
            except Exception as exc:
                logger.error("Gossip callback error (%s): %s", subtype, exc)
                self._telemetry["error_total"] += 1

        # Optional ledger persistence
        if self.ledger and subtype != "heartbeat":
            try:
                await self.ledger.log_gossip(
                    nonce=msg.nonce,
                    subtype=subtype,
                    origin=sender_id,
                    payload=data,
                    ttl=msg.ttl,
                )
            except Exception:
                pass

    @staticmethod
    def _topic_name(subtype: str) -> str:
        """Map a gossip subtype to a GossipSub topic string."""
        return f"prsm/{subtype}"
