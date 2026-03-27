"""
Whiteboard Priority Router
===========================

Routes prioritized whiteboard updates to agents based on priority level.

Priority-based push/pull model:
    ROUTINE   → inbox only (agents pull at their own breakpoint)
    IMPORTANT → inbox + flag (agents notified at next check-in)
    URGENT    → inbox + urgent flag + EventBus URGENT_UPDATE event (immediate interrupt)

This router bridges the LiveScribe's priority system with the EventBus,
enabling real-time interrupt-driven notifications for urgent updates
while maintaining the pull-based model for routine work.

Quick Start
-----------
>>> from prsm.compute.nwtn.team import WhiteboardRouter, AgentInbox
>>> from prsm.compute.nwtn.bsc import EventBus
>>>
>>> router = WhiteboardRouter(agent_inbox=inbox, event_bus=bus)
>>> result = await router.route(update, session_id="sess-123")
>>> print(result.urgent_event_published)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from prsm.compute.nwtn.bsc import BSCEvent, EventBus
    from prsm.compute.nwtn.team.live_scribe import AgentInbox, PrioritizedUpdate, UpdatePriority

logger = logging.getLogger(__name__)


# ======================================================================
# Routing Result
# ======================================================================

@dataclass
class RoutingResult:
    """
    Result of routing a prioritized update.

    Returned by WhiteboardRouter.route() to indicate what actions were taken.
    """
    priority: "UpdatePriority"
    """Priority level of the routed update."""

    agents_notified: List[str] = field(default_factory=list)
    """Agent IDs that received the update in their inbox."""

    urgent_event_published: bool = False
    """True if a URGENT_UPDATE event was published to EventBus."""

    entry_id: Optional[int] = None
    """ID of the whiteboard entry."""

    reason: str = ""
    """Reason for priority assignment (copied from update)."""


# ======================================================================
# Whiteboard Router
# ======================================================================

class WhiteboardRouter:
    """
    Routes prioritized whiteboard updates based on priority level.

    ROUTINE  → inbox only (agents pull at their own breakpoint)
    IMPORTANT → inbox + flag (agents notified at next check-in)
    URGENT   → inbox + urgent flag + EventBus URGENT_UPDATE event (immediate interrupt)

    The router always pushes updates to agent inboxes regardless of priority.
    For URGENT updates only, it also publishes an event to the EventBus so
    agents can receive real-time interrupt notifications.

    If no EventBus is configured, URGENT updates still set the inbox flag
    (graceful degradation).

    Parameters
    ----------
    agent_inbox : AgentInbox
        The agent inbox for distributing updates.
    event_bus : EventBus, optional
        Event bus for publishing URGENT_UPDATE events. If None, urgent
        updates still work but only via the inbox (no real-time interrupt).
    """

    def __init__(
        self,
        agent_inbox: "AgentInbox",
        event_bus: Optional["EventBus"] = None,
    ) -> None:
        self._inbox = agent_inbox
        self._event_bus = event_bus

        # Stats tracking
        self._stats: Dict[str, int] = {
            "routine_count": 0,
            "important_count": 0,
            "urgent_count": 0,
            "total_routed": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        update: "PrioritizedUpdate",
        session_id: str,
    ) -> RoutingResult:
        """
        Route a prioritized update to the right agents via the right mechanism.

        Always pushes to inbox. For URGENT only, also publishes to EventBus.

        Parameters
        ----------
        update : PrioritizedUpdate
            The prioritized update to route.
        session_id : str
            Session ID for the EventBus event.

        Returns
        -------
        RoutingResult
            Result indicating what actions were taken.
        """
        from prsm.compute.nwtn.bsc import BSCEvent, EventType
        from prsm.compute.nwtn.team.live_scribe import UpdatePriority

        # Always push to inbox
        await self._inbox.push(update)

        # Track stats
        priority_key = f"{update.priority.value}_count"
        self._stats[priority_key] = self._stats.get(priority_key, 0) + 1
        self._stats["total_routed"] += 1

        result = RoutingResult(
            priority=update.priority,
            agents_notified=update.relevant_agents.copy() if update.relevant_agents else [],
            entry_id=update.entry.id if update.entry else None,
            reason=update.reason,
            urgent_event_published=False,
        )

        # For URGENT only, publish to EventBus
        if update.priority == UpdatePriority.URGENT and self._event_bus is not None:
            event = BSCEvent(
                event_type=EventType.URGENT_UPDATE,
                session_id=session_id,
                data={
                    "agent_ids": update.relevant_agents,
                    "entry_id": str(update.entry.id) if update.entry else None,
                    "reason": update.reason,
                    "session_id": session_id,
                },
            )
            await self._event_bus.publish(event)
            result.urgent_event_published = True
            logger.info(
                "WhiteboardRouter: URGENT update published to EventBus "
                "(entry=%s, agents=%s)",
                result.entry_id,
                result.agents_notified,
            )
        elif update.priority == UpdatePriority.URGENT:
            logger.info(
                "WhiteboardRouter: URGENT update routed (no EventBus) "
                "(entry=%s, agents=%s)",
                result.entry_id,
                result.agents_notified,
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Return routing statistics.

        Returns
        -------
        dict
            Contains: routine_count, important_count, urgent_count, total_routed
        """
        return self._stats.copy()


# ======================================================================
# Exports
# ======================================================================

__all__ = [
    "RoutingResult",
    "WhiteboardRouter",
]
