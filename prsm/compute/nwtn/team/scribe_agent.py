"""
ScribeAgent — Event-Driven Checkpoint Coordinator
===================================================

Thin async wrapper around LiveScribe that listens for ROUND_ADVANCED
events on the EventBus and triggers checkpoint cycles autonomously.

The ScribeAgent runs as a background task, responding to events from
the BSC pipeline without blocking the main execution path.

Design Principles
-----------------
- Non-blocking checkpoint runs (fire via asyncio.create_task)
- Graceful error handling (never crash on LiveScribe errors)
- Simple start/stop lifecycle
- Delegates context retrieval and status to LiveScribe

Quick Start
-----------
>>> from prsm.compute.nwtn.team import ScribeAgent, LiveScribe
>>> from prsm.compute.nwtn.bsc import EventBus
>>>
>>> scribe_agent = ScribeAgent(live_scribe=live_scribe)
>>> await scribe_agent.start(event_bus)
>>> # ... agent team works ...
>>> status = await scribe_agent.status()
>>> await scribe_agent.stop()
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from prsm.compute.nwtn.bsc import BSCEvent, EventBus, EventType
    from prsm.compute.nwtn.team.live_scribe import LiveScribe, PrioritizedUpdate
    from prsm.compute.nwtn.team.whiteboard_router import WhiteboardRouter

logger = logging.getLogger(__name__)


class ScribeAgent:
    """
    Event-driven checkpoint coordinator for NWTN Agent Teams.

    Subscribes to ROUND_ADVANCED events on the EventBus and triggers
    checkpoint cycles via LiveScribe.check_and_run_checkpoint().

    Checkpoint runs are non-blocking — they're fired as background tasks
    so the event handler returns immediately without blocking the publisher.

    Parameters
    ----------
    live_scribe : LiveScribe
        Pre-constructed LiveScribe instance. Must have setup() called.
    router : WhiteboardRouter, optional
        Optional router for priority-based update routing. If configured,
        route_update() will delegate to this router.
    """

    def __init__(
        self,
        live_scribe: "LiveScribe",
        router: Optional["WhiteboardRouter"] = None,
    ) -> None:
        self._live_scribe = live_scribe
        self._router = router
        self._event_bus: Optional["EventBus"] = None
        self._subscription_callback: Optional[Callable[["BSCEvent"], Any]] = None
        self._running: bool = False
        self._checkpoints_run: int = 0
        self._checkpoint_errors: int = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, event_bus: "EventBus") -> None:
        """
        Subscribe to ROUND_ADVANCED events and begin listening.

        Parameters
        ----------
        event_bus : EventBus
            The event bus to subscribe to.

        Raises
        ------
        RuntimeError
            If already running.
        """
        async with self._lock:
            if self._running:
                raise RuntimeError("ScribeAgent is already running")

            from prsm.compute.nwtn.bsc import EventType

            self._event_bus = event_bus
            self._subscription_callback = self._on_round_advanced
            await event_bus.subscribe(EventType.ROUND_ADVANCED, self._subscription_callback)
            self._running = True
            logger.info("ScribeAgent: started and subscribed to ROUND_ADVANCED events")

    async def stop(self) -> None:
        """
        Unsubscribe from the event bus and stop listening.

        Safe to call multiple times — idempotent.
        """
        async with self._lock:
            if not self._running:
                return

            if self._event_bus and self._subscription_callback:
                from prsm.compute.nwtn.bsc import EventType
                await self._event_bus.unsubscribe(
                    EventType.ROUND_ADVANCED, self._subscription_callback
                )

            self._event_bus = None
            self._subscription_callback = None
            self._running = False
            logger.info("ScribeAgent: stopped and unsubscribed from ROUND_ADVANCED events")

    # ------------------------------------------------------------------
    # Event Handling
    # ------------------------------------------------------------------

    async def _on_round_advanced(self, event: "BSCEvent") -> None:
        """
        Handle ROUND_ADVANCED event by triggering a checkpoint cycle.

        The checkpoint runs in a background task (non-blocking) so the
        event handler returns immediately without blocking the publisher.
        Errors are caught and logged — never crash the agent.
        """
        # Fire checkpoint in background (non-blocking)
        asyncio.create_task(self._run_checkpoint_safe(event.session_id))

    async def _run_checkpoint_safe(self, session_id: str) -> None:
        """
        Run a checkpoint cycle with error handling.

        All exceptions are caught, logged, and counted — never propagate.
        """
        try:
            result = await self._live_scribe.check_and_run_checkpoint()

            async with self._lock:
                self._checkpoints_run += 1

            if result is not None:
                if result.success:
                    logger.info(
                        "ScribeAgent: checkpoint cycle completed (session=%s) — "
                        "synthesis=%s, ledger_entry=%s",
                        session_id,
                        result.synthesis is not None,
                        result.ledger_entry is not None,
                    )
                else:
                    logger.warning(
                        "ScribeAgent: checkpoint cycle failed (session=%s): %s",
                        session_id,
                        result.error,
                    )
            else:
                logger.debug(
                    "ScribeAgent: checkpoint not ready (session=%s)",
                    session_id,
                )

        except Exception as exc:
            async with self._lock:
                self._checkpoint_errors += 1
            logger.error(
                "ScribeAgent: checkpoint run error (session=%s): %s",
                session_id,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Context Retrieval
    # ------------------------------------------------------------------

    async def get_context(self, agent_id: str) -> str:
        """
        Get context for a specific agent.

        Delegates to LiveScribe.get_agent_context().

        Parameters
        ----------
        agent_id : str
            Agent identifier (e.g., "agent/coder-20260326").

        Returns
        -------
        str
            Context string for the agent to read.
        """
        return await self._live_scribe.get_agent_context(agent_id)

    # ------------------------------------------------------------------
    # Update Routing
    # ------------------------------------------------------------------

    async def route_update(
        self,
        update: "PrioritizedUpdate",
        session_id: str,
    ):
        """
        Route a prioritized update through the WhiteboardRouter.

        If no router is configured, this is a no-op.

        Parameters
        ----------
        update : PrioritizedUpdate
            The prioritized update to route.
        session_id : str
            Session ID for EventBus events.

        Returns
        -------
        RoutingResult | None
            Result from the router, or None if no router configured.
        """
        if self._router is None:
            return None

        return await self._router.route(update, session_id)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status(self) -> dict:
        """
        Return current status of the ScribeAgent.

        Includes:
        - `running`: bool — whether the agent is currently active
        - `checkpoints_run`: int — number of checkpoint cycles triggered
        - `checkpoint_errors`: int — number of checkpoint errors
        - `router_stats`: dict — routing statistics (if router configured)
        - Plus all fields from LiveScribe.status()

        Returns
        -------
        dict
            Status dictionary.
        """
        # Get live scribe status
        scribe_status = await self._live_scribe.status()

        async with self._lock:
            result = {
                **scribe_status,
                "running": self._running,
                "checkpoints_run": self._checkpoints_run,
                "checkpoint_errors": self._checkpoint_errors,
            }

            # Include router stats if configured
            if self._router is not None:
                result["router_stats"] = self._router.get_stats()

            return result
