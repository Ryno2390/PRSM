"""
Whiteboard Monitor
==================

Watches agent output files for new content and feeds deltas through the BSC
Promoter into the Active Whiteboard store.

Architecture
------------

Each agent on the team has one or more *watched files* — typically the
OpenClaw ``MEMORY.md`` file and/or a structured output log.  The monitor
tracks the last byte offset read from each file.  When ``watchfiles`` fires
a change event, the monitor reads from the last offset to EOF, debounces
small partial writes, and submits the accumulated delta to the BSC pipeline.

    OpenClaw MEMORY.md  ──►  watchfiles event
                                    │
                              read delta (byte offset tracking)
                                    │
                              debounce buffer (min chunk size / max wait)
                                    │
                           BSCPromoter.process_chunk()
                                    │
                         promoted? ─┴─ yes ──► WhiteboardStore.write()
                                    │
                                    no ──► discard

Debouncing
----------
OpenClaw may flush its MEMORY.md in multiple small writes within a short
window.  A raw delta of "Updating " is not a useful BSC chunk.  The monitor
waits up to *debounce_secs* seconds for the buffer to grow to at least
*min_chunk_chars* characters before submitting to the BSC.  A force-flush
is triggered if *max_buffer_secs* elapses, so long sessions are never
silently dropped.

File tracking
-------------
The monitor stores per-file state in ``WatchedAgent`` objects keyed by the
real (resolved) file path.  If a watched file is replaced or rotated, the
monitor resets the offset to 0 and processes the new content from the start.

Graceful shutdown
-----------------
``await monitor.stop()`` sets the cancellation flag, which causes the
``watchfiles.awatch()`` generator to stop on its next iteration.  Any
buffered content is force-flushed before the monitor exits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiofiles

from prsm.compute.nwtn.bsc import BSCPromoter
from .store import WhiteboardStore

logger = logging.getLogger(__name__)

# Minimum characters in the debounce buffer before submitting to the BSC.
# Below this the chunk is too short for meaningful perplexity evaluation.
DEFAULT_MIN_CHUNK_CHARS: int = 80

# Maximum seconds to hold a partial buffer before force-flushing.
DEFAULT_MAX_BUFFER_SECS: float = 5.0

# Seconds of quiet after the last write before the buffer is submitted.
DEFAULT_DEBOUNCE_SECS: float = 1.0


@dataclass
class WatchedAgent:
    """Per-agent file-watch state."""

    file_path: Path
    source_agent: str
    session_id: str

    # Byte offset: how far into the file we have already read
    last_offset: int = 0

    # Accumulated partial content not yet submitted to BSC
    buffer: str = ""

    # Wallclock time of the last append to the buffer
    last_append_ts: float = field(default_factory=time.monotonic)

    # Wallclock time of the first append in the current buffer epoch
    buffer_start_ts: float = field(default_factory=time.monotonic)

    # inode/size at last read — used to detect file rotation
    last_inode: Optional[int] = None
    last_size: int = 0


class WhiteboardMonitor:
    """
    Watches agent output files and feeds BSC-filtered deltas to the whiteboard.

    Parameters
    ----------
    promoter : BSCPromoter
        The BSC pipeline.  ``process_chunk()`` is called for every delta.
    store : WhiteboardStore
        The Active Whiteboard store.  ``write()`` is called for every
        promoted decision.
    min_chunk_chars : int
        Minimum buffer size before submitting a chunk to the BSC.
    debounce_secs : float
        Seconds of quiet after the last file write before submitting.
    max_buffer_secs : float
        Seconds before a non-empty buffer is force-flushed regardless of
        quiet time.
    pre_filters : list of callable, optional
        Functions applied to each chunk **before** BSC evaluation, in order.
        Signature: ``(chunk: str) -> str``.  Return the empty string to drop
        the chunk entirely (it will not be submitted to the BSC).

        Use cases (from live 9-round test findings):
          - Strip prompt-header echoes: LLMs sometimes repeat the round
            header (``ROUND N — TASK DESCRIPTION``) at the start of their
            response.  The BSC scored these at 1.00 (maximum surprise because
            they are very different from prior whiteboard content) and promoted
            them as if they were agent discoveries.  8% of promotions in the
            live test were prompt echoes.  A pre-filter strips them cleanly.
          - Remove boilerplate preamble (``Sure, I'll help with...``).
          - Sanitise PII before the chunk reaches the whiteboard.

        Example — register a round-header stripper::

            import re
            def strip_round_header(chunk: str) -> str:
                return re.sub(r'^ROUND \\d+\\s*[—-][^\\n]*\\n', '', chunk).strip()

            monitor = WhiteboardMonitor(
                promoter=promoter, store=store,
                pre_filters=[strip_round_header],
            )
    """

    def __init__(
        self,
        promoter: BSCPromoter,
        store: WhiteboardStore,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
        debounce_secs: float = DEFAULT_DEBOUNCE_SECS,
        max_buffer_secs: float = DEFAULT_MAX_BUFFER_SECS,
        pre_filters: Optional[List] = None,
    ) -> None:
        self._promoter = promoter
        self._store = store
        self._min_chunk_chars = min_chunk_chars
        self._debounce_secs = debounce_secs
        self._max_buffer_secs = max_buffer_secs
        self._pre_filters: List = list(pre_filters or [])

        # file_path (str) → WatchedAgent
        self._agents: Dict[str, WatchedAgent] = {}

        self._running = False
        self._stop_event = asyncio.Event()
        self._watch_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def watch_agent(
        self,
        file_path: str,
        source_agent: str,
        session_id: str,
    ) -> None:
        """
        Register an agent's output file for monitoring.

        Safe to call before ``start()``.  If the file does not yet exist
        the monitor will begin watching as soon as it is created.

        Parameters
        ----------
        file_path : str
            Absolute path to the agent's output / MEMORY.md file.
        source_agent : str
            Agent identifier (e.g. ``"agent/coder-20260326"``).
        session_id : str
            Session this agent belongs to.
        """
        resolved = str(Path(file_path).resolve())
        self._agents[resolved] = WatchedAgent(
            file_path=Path(resolved),
            source_agent=source_agent,
            session_id=session_id,
        )
        logger.info(
            "WhiteboardMonitor: registered %s → %s", source_agent, resolved
        )

    def unwatch_agent(self, file_path: str) -> None:
        """Deregister a file.  Any buffered content is discarded."""
        resolved = str(Path(file_path).resolve())
        self._agents.pop(resolved, None)

    def add_pre_filter(self, fn) -> None:
        """
        Append a pre-filter function applied to chunks before BSC evaluation.

        The function receives a chunk string and returns a (possibly modified)
        string.  Return the empty string to drop the chunk entirely.

        Pre-filters are applied in registration order.

        Parameters
        ----------
        fn : callable
            ``(chunk: str) -> str``
        """
        self._pre_filters.append(fn)

    def clear_pre_filters(self) -> None:
        """Remove all registered pre-filter functions."""
        self._pre_filters.clear()

    @property
    def watched_paths(self) -> Set[str]:
        """Set of currently watched file paths."""
        return set(self._agents.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background file-watching and debounce-flush tasks."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        self._watch_task = asyncio.create_task(
            self._watch_loop(), name="whiteboard-monitor-watch"
        )
        self._flush_task = asyncio.create_task(
            self._flush_loop(), name="whiteboard-monitor-flush"
        )
        logger.info(
            "WhiteboardMonitor started — watching %d file(s)", len(self._agents)
        )

    async def stop(self) -> None:
        """
        Gracefully stop monitoring.

        Force-flushes all pending buffers before exiting so no content is lost.
        """
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        # Force-flush all pending buffers
        for agent in list(self._agents.values()):
            if agent.buffer.strip():
                await self._submit_chunk(agent, force=True)

        for task in (self._watch_task, self._flush_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("WhiteboardMonitor stopped")

    # ------------------------------------------------------------------
    # Internal: watch loop (watchfiles)
    # ------------------------------------------------------------------

    async def _watch_loop(self) -> None:
        """
        Main loop: uses ``watchfiles.awatch`` for efficient filesystem events.

        Falls back gracefully if watchfiles is not available (shouldn't
        happen since it's installed, but defensive coding).
        """
        try:
            from watchfiles import awatch, Change  # type: ignore[import]
        except ImportError:
            logger.warning(
                "watchfiles not available — falling back to polling monitor"
            )
            await self._polling_loop()
            return

        watched_paths = list(self._agents.keys())
        if not watched_paths:
            logger.warning("WhiteboardMonitor: no files to watch")
            return

        try:
            async for changes in awatch(
                *watched_paths,
                stop_event=self._stop_event,
                poll_delay_ms=500,
            ):
                for change_type, path in changes:
                    path_str = str(Path(path).resolve())
                    agent = self._agents.get(path_str)
                    if agent is None:
                        continue
                    await self._read_delta(agent)
        except asyncio.CancelledError:
            pass

    async def _polling_loop(self) -> None:
        """Fallback polling loop for environments without watchfiles."""
        while self._running:
            for agent in list(self._agents.values()):
                await self._read_delta(agent)
            await asyncio.sleep(2.0)

    # ------------------------------------------------------------------
    # Internal: flush loop (debounce timer)
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """
        Periodically checks all agent buffers and force-flushes any that
        have exceeded the max buffer age or have been quiet long enough.
        """
        while self._running:
            now = time.monotonic()
            for agent in list(self._agents.values()):
                if not agent.buffer.strip():
                    continue
                quiet_for = now - agent.last_append_ts
                age = now - agent.buffer_start_ts

                if quiet_for >= self._debounce_secs or age >= self._max_buffer_secs:
                    await self._submit_chunk(agent, force=False)

            await asyncio.sleep(self._debounce_secs / 2)

    # ------------------------------------------------------------------
    # Internal: file reading and delta extraction
    # ------------------------------------------------------------------

    async def _read_delta(self, agent: WatchedAgent) -> None:
        """
        Read any new bytes from *agent.file_path* since the last read.

        Handles file rotation (file replaced or truncated) by resetting the
        offset when the inode changes or the file shrinks.
        """
        path = agent.file_path
        if not path.exists():
            return

        try:
            stat = path.stat()
        except OSError:
            return

        current_inode = stat.st_ino
        current_size = stat.st_size

        # Detect file rotation
        if (
            agent.last_inode is not None and current_inode != agent.last_inode
        ) or current_size < agent.last_offset:
            logger.info(
                "WhiteboardMonitor: file rotated for %s — resetting offset",
                agent.source_agent,
            )
            agent.last_offset = 0
            agent.buffer = ""

        agent.last_inode = current_inode
        agent.last_size = current_size

        if current_size <= agent.last_offset:
            return  # no new bytes

        try:
            async with aiofiles.open(path, mode="r", encoding="utf-8", errors="replace") as f:
                await f.seek(agent.last_offset)
                delta = await f.read()
        except OSError as exc:
            logger.warning("WhiteboardMonitor: read error for %s: %s", path, exc)
            return

        if not delta:
            return

        agent.last_offset += len(delta.encode("utf-8", errors="replace"))

        # Append to buffer
        now = time.monotonic()
        if not agent.buffer:
            agent.buffer_start_ts = now
        agent.buffer += delta
        agent.last_append_ts = now

        # Submit immediately if buffer is large enough
        if len(agent.buffer) >= self._min_chunk_chars:
            await self._submit_chunk(agent, force=False)

    # ------------------------------------------------------------------
    # Internal: BSC submission
    # ------------------------------------------------------------------

    async def _submit_chunk(self, agent: WatchedAgent, *, force: bool) -> None:
        """
        Submit the current buffer for *agent* through the BSC pipeline.

        Clears the buffer regardless of the BSC outcome (discard or promote).
        If promoted, writes the decision to the whiteboard store.
        """
        chunk = agent.buffer.strip()
        agent.buffer = ""

        if not chunk:
            return

        # Apply pre-filters before any other processing
        for fn in self._pre_filters:
            chunk = fn(chunk)
            if not chunk:
                logger.debug(
                    "WhiteboardMonitor: chunk dropped by pre-filter for %s",
                    agent.source_agent,
                )
                return

        if not force and len(chunk) < self._min_chunk_chars:
            # Too short and not forced — put it back and wait
            agent.buffer = chunk
            return

        # Get the current whiteboard state as BSC context
        try:
            context = await self._store.compressed_state(agent.session_id)
        except Exception:
            context = ""

        try:
            decision = await self._promoter.process_chunk(
                chunk=chunk,
                context=context,
                source_agent=agent.source_agent,
                session_id=agent.session_id,
            )
        except Exception as exc:
            logger.error(
                "WhiteboardMonitor: BSC error for %s: %s", agent.source_agent, exc
            )
            return

        if decision.promoted:
            try:
                await self._store.write(decision)
            except Exception as exc:
                logger.error(
                    "WhiteboardMonitor: store write error: %s", exc
                )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def pending_buffers(self) -> Dict[str, int]:
        """Map of source_agent → pending buffer length for each watched agent."""
        return {
            a.source_agent: len(a.buffer)
            for a in self._agents.values()
            if a.buffer
        }
