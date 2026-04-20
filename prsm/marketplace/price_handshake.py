"""Phase 3 Task 7: requester-side price-quote client.

Sends a shard_price_quote_request to a provider, awaits the ack/reject
response. Parallels the Phase 2 dispatcher's future-based MSG_DIRECT
round-trip but for the lighter-weight price-handshake protocol.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from prsm.node.transport import MSG_DIRECT, P2PMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceQuote:
    """Accepted price quote from a provider, ready to consume as
    escrow_amount_ftns for the subsequent shard dispatch."""
    request_id: str
    listing_id: str
    shard_index: int
    quoted_price_ftns: float
    quote_expires_unix: int
    provider_id: str
    provider_pubkey_b64: str
    signature: str

    def is_expired(self, at_unix: Optional[int] = None) -> bool:
        now = at_unix if at_unix is not None else int(time.time())
        return self.quote_expires_unix < now


@dataclass(frozen=True)
class PriceQuoteRejected:
    """Provider-sent rejection. Reason strings per
    ComputeProvider._on_shard_price_quote_request: no_active_listing,
    overloaded, shard_too_large, above_ceiling."""
    request_id: str
    listing_id: str
    reason: str


class PriceNegotiator:
    """Requester-side price-quote round-trip.

    Owns a _pending dict of asyncio.Future keyed by request_id,
    populated on response by _on_direct_message. Parallel to the
    Phase 2 dispatcher's future-based response routing.
    """

    def __init__(self, identity, transport, default_timeout: float = 5.0):
        self.identity = identity
        self.transport = transport
        self.default_timeout = default_timeout
        self._pending: Dict[str, asyncio.Future] = {}
        transport.on_message(MSG_DIRECT, self._on_direct_message)

    async def request_quote(
        self,
        listing,
        shard_index: int,
        shard_size_bytes: int,
        max_acceptable_price_ftns: float,
    ):
        """Send a shard_price_quote_request to the listing's provider
        and return a PriceQuote on accept or PriceQuoteRejected on
        reject. Returns None on timeout.

        Separately validates that the provider's quoted price does NOT
        exceed the listing's advertised price_per_shard_ftns (if it
        does, the listing is stale or the provider is lying — caller
        treats as rejected)."""
        request_id = f"quote-{uuid.uuid4().hex[:16]}"
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        payload = {
            "subtype": "shard_price_quote_request",
            "request_id": request_id,
            "listing_id": listing.listing_id,
            "shard_index": shard_index,
            "shard_size_bytes": shard_size_bytes,
            "max_acceptable_price_ftns": max_acceptable_price_ftns,
            "deadline_unix": int(time.time()) + int(self.default_timeout) + 1,
        }
        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload=payload,
        )

        try:
            await self.transport.send_to_peer(listing.provider_id, msg)
            response = await asyncio.wait_for(
                future, timeout=self.default_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"price quote request {request_id} to "
                f"{listing.provider_id[:12]}… timed out"
            )
            return None
        finally:
            self._pending.pop(request_id, None)

        subtype = response.get("subtype")
        if subtype == "shard_price_quote_reject":
            return PriceQuoteRejected(
                request_id=request_id,
                listing_id=listing.listing_id,
                reason=response.get("reason", "unknown"),
            )

        if subtype != "shard_price_quote_ack":
            logger.warning(
                f"price quote request {request_id}: unexpected response "
                f"subtype {subtype!r}"
            )
            return None

        quoted_price = float(response.get("quoted_price_ftns", 0))
        if quoted_price > listing.price_per_shard_ftns:
            # Provider lied above the listing ceiling — treat as reject.
            logger.warning(
                f"provider {listing.provider_id[:12]}… quoted "
                f"{quoted_price} but listing caps at "
                f"{listing.price_per_shard_ftns}; rejecting"
            )
            return PriceQuoteRejected(
                request_id=request_id,
                listing_id=listing.listing_id,
                reason="quote_exceeds_listing",
            )

        return PriceQuote(
            request_id=request_id,
            listing_id=listing.listing_id,
            shard_index=shard_index,
            quoted_price_ftns=quoted_price,
            quote_expires_unix=int(response.get("quote_expires_unix", 0)),
            provider_id=response.get("provider_id", ""),
            provider_pubkey_b64=response.get("provider_pubkey_b64", ""),
            signature=response.get("signature", ""),
        )

    async def _on_direct_message(self, msg: P2PMessage, peer) -> None:
        subtype = msg.payload.get("subtype", "")
        if subtype not in ("shard_price_quote_ack", "shard_price_quote_reject"):
            return
        request_id = msg.payload.get("request_id")
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(msg.payload)
