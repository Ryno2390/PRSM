"""
Whiteboard Push Handler — BSC → LiveScribe Bridge
===================================================

Subscribes to BSC EventBus events and pushes promoted chunks to the
LiveScribe for distribution to agent inboxes.

This is the event-driven bridge that completes the pipeline:

    Agent → BSC → EventBus → **WhiteboardPushHandler** → LiveScribe → Agent Inboxes

Without this handler, the BSC pipeline correctly filters for novelty and
quality but never notifies agents about what was promoted.  The handler
is the "last mile" that turns a static filter into a real-time
notification system.

Typical usage
-------------
.. code-block:: python

    bus = EventBus()
    scribe = LiveScribe(whiteboard_store=store, ...)
    handler = WhiteboardPushHandler(event_bus=bus, live_scribe=scribe)

    await handler.start()
    # ... BSC processes chunks, publishes CHUNK_PROMOTED events ...
    # ... handler pushes them to scribe automatically ...
    await handler.stop()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from .event_bus import BSCEvent, EventBus, EventType

if TYPE_CHECKING:
    from ..team.live_scribe import LiveScribe

logger = logging.getLogger(__name__)


@dataclass
class PushStats:
    """Cumulative statistics for the push handler."""

    pushed: int = 0
    """Chunks successfully pushed to LiveScribe."""

    failed: int = 0
    """Pushes that raised an exception."""

    conflicts: int = 0
    """Pushes that detected a conflict with existing whiteboard entries."""

    skipped: int = 0
    """Events received but skipped (missing decision data)."""

    def to_dict(self) -> Dict[str, int]:
        return {
            "pushed": self.pushed,
            "failed": self.failed,
            "conflicts": self.conflicts,
            "skipped": self.skipped,
        }


class WhiteboardPushHandler:
    """
    Subscribes to BSC :class:`EventBus` and pushes promoted chunks to a
    :class:`~prsm.compute.nwtn.team.live_scribe.LiveScribe`.

    The handler is intentionally stateless beyond its statistics counter.
    It does not buffer or reorder events — each ``CHUNK_PROMOTED`` event
    is forwarded to the scribe immediately.

    Parameters
    ----------
    event_bus : EventBus
        The bus to subscribe to.
    live_scribe : LiveScribe
        The scribe that distributes updates to agent inboxes.
    """

    def __init__(self, event_bus: EventBus, live_scribe: "LiveScribe") -> None:
        self._bus = event_bus
        self._scribe = live_scribe
        self._stats = PushStats()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to ``CHUNK_PROMOTED`` events."""
        if self._running:
            logger.warning("WhiteboardPushHandler: already running")
            return
        await self._bus.subscribe(EventType.CHUNK_PROMOTED, self._on_promoted)
        self._running = True
        logger.info("WhiteboardPushHandler: started — listening for CHUNK_PROMOTED events")

    async def stop(self) -> None:
        """Unsubscribe from events."""
        if not self._running:
            return
        await self._bus.unsubscribe(EventType.CHUNK_PROMOTED, self._on_promoted)
        self._running = False
        logger.info("WhiteboardPushHandler: stopped")

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return cumulative push statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset all counters to zero."""
        self._stats = PushStats()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _on_promoted(self, event: BSCEvent) -> None:
        """
        Handle a ``CHUNK_PROMOTED`` event by pushing to LiveScribe.

        Errors are caught and counted — this handler must never raise,
        because the EventBus delivers events fire-and-forget.
        """
        decision = event.data.get("decision")
        if decision is None:
            self._stats.skipped += 1
            logger.warning(
                "WhiteboardPushHandler: received CHUNK_PROMOTED without decision data "
                "(session=%s)", event.session_id,
            )
            return

        try:
            update = await self._scribe.on_chunk_promoted(decision)
            self._stats.pushed += 1
            if getattr(update, "conflict_detected", False):
                self._stats.conflicts += 1
                logger.info(
                    "WhiteboardPushHandler: conflict detected for chunk from %s "
                    "(session=%s)",
                    getattr(decision.metadata, "source_agent", "unknown"),
                    event.session_id,
                )
            logger.debug(
                "WhiteboardPushHandler: pushed chunk from %s (session=%s, "
                "surprise=%.3f)",
                getattr(decision.metadata, "source_agent", "unknown"),
                event.session_id,
                getattr(decision, "surprise_score", 0.0),
            )
        except Exception:
            self._stats.failed += 1
            logger.exception(
                "WhiteboardPushHandler: failed to push chunk from %s (session=%s)",
                getattr(decision.metadata, "source_agent", "unknown"),
                event.session_id,
            )
