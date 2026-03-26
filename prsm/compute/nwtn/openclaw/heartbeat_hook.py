"""
Heartbeat Hook
==============

Triggers the Nightly Synthesis pipeline at the end of each working session.

Two trigger sources
-------------------
1. **OpenClaw Gateway heartbeat** — the Gateway emits ``{"type": "heartbeat"}``
   events on a configurable schedule (default: every 30 minutes).  When
   HeartbeatHook receives one, it checks whether a synthesis is due.

2. **Periodic timer fallback** — if no Gateway heartbeat has arrived within
   ``max_gap_seconds``, the hook fires its own timer.  This ensures synthesis
   happens even when the Gateway is not running (standalone mode).

Session-end detection
---------------------
The hook considers a session "ready for synthesis" when:
  - At least ``min_whiteboard_entries`` entries are on the whiteboard, AND
  - At least ``min_session_minutes`` have elapsed since the session started, OR
  - A forced synthesis is requested via ``trigger()``.

After synthesis
---------------
The hook:
  1. Calls ``NarrativeSynthesizer.synthesise()`` with the current snapshot.
  2. Appends the result to the ``ProjectLedger`` with a ``LedgerSigner``.
  3. Calls the optional ``on_complete`` callback with the new ``LedgerEntry``.
  4. Resets the session state (clears the Active Whiteboard for the next day).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)

# Type alias for the completion callback
OnCompleteCallback = Callable[
    ["Any"],  # LedgerEntry — avoid circular import
    Coroutine[Any, Any, None],
]

DEFAULT_HEARTBEAT_INTERVAL = 1800   # 30 min (matches OpenClaw default)
DEFAULT_MAX_GAP_S           = 1800   # fire self-timer if no Gateway heartbeat in 30 min
DEFAULT_MIN_ENTRIES         = 3      # skip synthesis for near-empty sessions
DEFAULT_MIN_SESSION_MIN     = 5      # skip synthesis for very short sessions


# ======================================================================
# HeartbeatHook
# ======================================================================

class HeartbeatHook:
    """
    Manages end-of-session synthesis triggers.

    Parameters
    ----------
    whiteboard_store : WhiteboardStore
        The Active Whiteboard to snapshot.
    synthesizer : NarrativeSynthesizer
        Re-constructor Agent.
    ledger : ProjectLedger
        Append-only Project Ledger.
    signer : LedgerSigner
        Key for signing new entries.
    session_id : str
        Current session identifier.
    meta_plan : MetaPlan, optional
        Provides milestone context for the synthesis narrative.
    on_complete : OnCompleteCallback, optional
        Async callback invoked with the new LedgerEntry after synthesis.
    dag_anchor : DAGAnchor, optional
        If provided, major entries are anchored to the PRSM DAG.
    heartbeat_interval : int
        Seconds between self-timer fires (when Gateway heartbeat is absent).
    max_gap_seconds : int
        After this many seconds without a Gateway heartbeat, the self-timer
        activates.
    min_whiteboard_entries : int
        Minimum entries required before synthesis is considered worth running.
    min_session_minutes : float
        Minimum minutes elapsed before synthesis can run.
    """

    def __init__(
        self,
        whiteboard_store,
        synthesizer,
        ledger,
        signer,
        session_id: str,
        meta_plan=None,
        on_complete: Optional[OnCompleteCallback] = None,
        dag_anchor=None,
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        max_gap_seconds: int = DEFAULT_MAX_GAP_S,
        min_whiteboard_entries: int = DEFAULT_MIN_ENTRIES,
        min_session_minutes: float = DEFAULT_MIN_SESSION_MIN,
    ) -> None:
        self._store = whiteboard_store
        self._synthesizer = synthesizer
        self._ledger = ledger
        self._signer = signer
        self._session_id = session_id
        self._meta_plan = meta_plan
        self._on_complete = on_complete
        self._dag_anchor = dag_anchor

        self._heartbeat_interval = heartbeat_interval
        self._max_gap_s = max_gap_seconds
        self._min_entries = min_whiteboard_entries
        self._min_session_min = min_session_minutes

        self._session_start: float = time.monotonic()
        self._last_gateway_heartbeat: float = 0.0
        self._synthesis_count: int = 0

        self._timer_task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the self-timer loop."""
        if self._running:
            return
        self._running = True
        self._session_start = time.monotonic()
        self._timer_task = asyncio.create_task(
            self._timer_loop(), name="nwtn-heartbeat-timer"
        )
        logger.info(
            "HeartbeatHook started: session=%s interval=%ds",
            self._session_id, self._heartbeat_interval,
        )

    async def stop(self, force_synthesis: bool = True) -> None:
        """
        Stop the hook, optionally forcing a final synthesis run.

        Parameters
        ----------
        force_synthesis : bool
            If True, run synthesis even if minimum thresholds are not met.
            Useful for end-of-session cleanup.
        """
        self._running = False
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        if force_synthesis:
            await self._run_synthesis(forced=True)
        logger.info("HeartbeatHook stopped: session=%s", self._session_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def on_gateway_heartbeat(self) -> None:
        """
        Called by the Gateway adapter when a heartbeat event is received.
        Updates the last-heartbeat timestamp and runs synthesis if due.
        """
        self._last_gateway_heartbeat = time.monotonic()
        logger.debug("HeartbeatHook: Gateway heartbeat received")
        await self._maybe_synthesise()

    async def trigger(self) -> Optional[Any]:
        """
        Force an immediate synthesis run, bypassing thresholds.

        Returns
        -------
        LedgerEntry or None
        """
        return await self._run_synthesis(forced=True)

    @property
    def synthesis_count(self) -> int:
        """Number of successful synthesis runs in this session."""
        return self._synthesis_count

    # ------------------------------------------------------------------
    # Internal: timer loop
    # ------------------------------------------------------------------

    async def _timer_loop(self) -> None:
        """
        Periodically fire synthesis if no Gateway heartbeat has arrived
        recently enough.
        """
        while self._running:
            await asyncio.sleep(min(self._heartbeat_interval, 60))
            if not self._running:
                break

            now = time.monotonic()
            gap_since_gateway = now - self._last_gateway_heartbeat
            if gap_since_gateway >= self._max_gap_s:
                logger.debug(
                    "HeartbeatHook: self-timer firing (%.0fs since last Gateway heartbeat)",
                    gap_since_gateway,
                )
                await self._maybe_synthesise()

    # ------------------------------------------------------------------
    # Internal: synthesis gate and execution
    # ------------------------------------------------------------------

    async def _maybe_synthesise(self) -> None:
        """Run synthesis if minimum thresholds are met."""
        session_min = (time.monotonic() - self._session_start) / 60.0
        if session_min < self._min_session_min:
            logger.debug(
                "HeartbeatHook: session too young (%.1f min < %.1f min), skipping",
                session_min, self._min_session_min,
            )
            return

        entry_count = await self._store.entry_count(self._session_id)
        if entry_count < self._min_entries:
            logger.debug(
                "HeartbeatHook: too few whiteboard entries (%d < %d), skipping",
                entry_count, self._min_entries,
            )
            return

        await self._run_synthesis(forced=False)

    async def _run_synthesis(self, *, forced: bool) -> Optional[Any]:
        """Execute the full synthesis pipeline and return the new LedgerEntry."""
        try:
            from prsm.compute.nwtn.whiteboard.query import WhiteboardQuery

            query = WhiteboardQuery(self._store)
            snapshot = await query.snapshot(self._session_id)

            if snapshot.entry_count == 0 and not forced:
                return None

            # Get previous summary for temporal context
            previous_summary: Optional[str] = None
            latest = self._ledger.latest_entry()
            if latest:
                previous_summary = latest.content[:800]

            synthesis = await self._synthesizer.synthesise(
                snapshot,
                meta_plan=self._meta_plan,
                previous_summary=previous_summary,
            )

            entry = self._ledger.append(synthesis, self._signer)
            self._synthesis_count += 1

            logger.info(
                "HeartbeatHook: synthesis complete — entry #%d "
                "(session=%s, entries=%d, forced=%s)",
                entry.entry_index, self._session_id,
                snapshot.entry_count, forced,
            )

            # Optional DAG anchor for forced (milestone) syntheses
            if self._dag_anchor and forced:
                receipt = await self._dag_anchor.anchor(
                    entry.entry_index,
                    entry.chain_hash,
                    self._session_id,
                )
                if receipt.success:
                    self._ledger.update_dag_anchor(
                        entry.entry_index, receipt.dag_tx_id
                    )

            # Fire callback
            if self._on_complete:
                await self._on_complete(entry)

            return entry

        except Exception as exc:
            logger.error(
                "HeartbeatHook: synthesis failed for session %s: %s",
                self._session_id, exc,
            )
            return None
