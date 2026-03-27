"""
Lightweight async pub/sub event bus for BSC pipeline events.

Connects BSC pipeline components (promoter, filter, dedup) to downstream
consumers (LiveScribe, monitoring, logging) without tight coupling.

The event bus is fire-and-forget: subscriber errors are logged but never
propagate to the publisher.  Subscribers are called concurrently via
``asyncio.gather`` (with ``return_exceptions=True``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Event Types
# ======================================================================

class EventType(Enum):
    """Catalog of BSC pipeline events."""
    CHUNK_PROMOTED = "chunk_promoted"
    CHUNK_REJECTED = "chunk_rejected"
    ROUND_ADVANCED = "round_advanced"


@dataclass(frozen=True)
class BSCEvent:
    """
    An immutable event emitted by the BSC pipeline.

    Parameters
    ----------
    event_type : EventType
        The kind of event.
    data : Dict[str, Any]
        Event payload.  For CHUNK_PROMOTED this contains ``"decision"``
        (a ``PromotionDecision``) and ``"metadata"`` (``ChunkMetadata``).
    session_id : str
        Session that produced the event.
    timestamp : datetime
        UTC timestamp of event creation (auto-populated).
    """
    event_type: EventType
    data: Dict[str, Any]
    session_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ======================================================================
# Event Bus
# ======================================================================

class EventBus:
    """
    Async publish/subscribe event bus for BSC pipeline events.

    Subscribers are async callables registered per event type.  Publishing
    is fire-and-forget: each subscriber is wrapped in ``try/except`` so a
    failing handler never breaks the pipeline.

    Example
    -------
    >>> bus = EventBus()
    >>> async def on_promoted(event: BSCEvent) -> None:
    ...     print(event.data)
    >>> await bus.subscribe(EventType.CHUNK_PROMOTED, on_promoted)
    >>> await bus.publish(BSCEvent(
    ...     event_type=EventType.CHUNK_PROMOTED,
    ...     data={"decision": decision},
    ...     session_id="sess-1",
    ... ))
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[BSCEvent], Awaitable[None]]]] = {
            et: [] for et in EventType
        }
        self._lock = asyncio.Lock()
        # Stats
        self._published: int = 0
        self._delivery_errors: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[BSCEvent], Awaitable[None]],
    ) -> None:
        """
        Register a callback for a specific event type.

        The same callback can be registered multiple times (it will be
        called once per registration).
        """
        async with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(
                    "EventBus: subscribed %s to %s",
                    getattr(callback, "__name__", repr(callback)),
                    event_type.value,
                )

    async def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[BSCEvent], Awaitable[None]],
    ) -> None:
        """Remove a previously registered callback."""
        async with self._lock:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(
                    "EventBus: unsubscribed %s from %s",
                    getattr(callback, "__name__", repr(callback)),
                    event_type.value,
                )
            except ValueError:
                logger.warning(
                    "EventBus: tried to unsubscribe %s from %s but it was not registered",
                    getattr(callback, "__name__", repr(callback)),
                    event_type.value,
                )

    async def publish(self, event: BSCEvent) -> None:
        """
        Publish an event to all registered subscribers.

        Fire-and-forget: each subscriber is called concurrently.  Errors
        are logged and counted but never raised.
        """
        async with self._lock:
            subscribers = list(self._subscribers[event.event_type])

        if not subscribers:
            logger.debug("EventBus: no subscribers for %s", event.event_type.value)
            self._published += 1
            return

        # Call all subscribers concurrently
        results = await asyncio.gather(
            *(self._safe_deliver(cb, event) for cb in subscribers),
            return_exceptions=True,
        )

        # Count delivery errors
        errors = sum(1 for r in results if r is not True)
        self._delivery_errors += errors
        self._published += 1

        if errors:
            logger.warning(
                "EventBus: %d/%d subscriber errors for %s (session=%s)",
                errors, len(subscribers), event.event_type.value, event.session_id,
            )

    async def clear(self, event_type: Optional[EventType] = None) -> None:
        """Remove all subscribers (or all for a specific event type)."""
        async with self._lock:
            if event_type is not None:
                self._subscribers[event_type] = []
            else:
                for et in EventType:
                    self._subscribers[et] = []

    def subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """Return number of registered subscribers."""
        if event_type is not None:
            return len(self._subscribers[event_type])
        return sum(len(subs) for subs in self._subscribers.values())

    def get_stats(self) -> Dict[str, Any]:
        """Return bus statistics."""
        return {
            "published": self._published,
            "delivery_errors": self._delivery_errors,
            "subscribers": {
                et.value: len(self._subscribers[et]) for et in EventType
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe_deliver(
        callback: Callable[[BSCEvent], Awaitable[None]],
        event: BSCEvent,
    ) -> bool:
        """Deliver an event to a single callback, catching and logging errors."""
        try:
            await callback(event)
            return True
        except Exception:
            logger.exception(
                "EventBus: subscriber %s raised for %s",
                getattr(callback, "__name__", repr(callback)),
                event.event_type.value,
            )
            return False
