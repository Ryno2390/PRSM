"""
prsm.compute.nwtn.whiteboard — Active Whiteboard
=================================================

The Active Whiteboard is the shared short-term memory of an Agent Team
working session.  It receives only BSC-promoted chunks — information that
is both surprising and non-redundant — keeping shared context lean regardless
of team size or session length.

Components
----------
WhiteboardStore
    aiosqlite-backed persistence layer.  The single writer: only the BSC
    Promoter (via the Monitor) calls ``store.write()``.

WhiteboardMonitor
    watchfiles-powered file watcher.  Monitors each agent's OpenClaw
    MEMORY.md (and other output files), feeds deltas through the BSC
    pipeline, and writes promoted chunks to the store.

WhiteboardQuery
    Read-only facade for agents to consume the whiteboard.  Provides
    ``compressed_state()`` (the context string fed back to the BSC
    predictor) and ``onboarding_brief()`` for agents joining mid-session.

WhiteboardEntry / WhiteboardSnapshot / SessionRecord
    Pydantic schema types shared across all layers.

Typical session lifecycle
-------------------------
.. code-block:: python

    from prsm.compute.nwtn.whiteboard import (
        WhiteboardStore, WhiteboardMonitor, WhiteboardQuery
    )
    from prsm.compute.nwtn.bsc import BSCPromoter, BSCDeploymentConfig

    # 1. Open the store
    store = WhiteboardStore(db_path="~/.prsm/sessions/sess-001/whiteboard.db")
    async with store:
        await store.create_session("sess-001")

        # 2. Wire up the BSC
        promoter = BSCPromoter.from_config(BSCDeploymentConfig.auto())
        await promoter.warmup()

        # 3. Start the monitor
        monitor = WhiteboardMonitor(promoter=promoter, store=store)
        monitor.watch_agent(
            file_path="/home/user/.openclaw/agents/coder/MEMORY.md",
            source_agent="agent/coder-20260326",
            session_id="sess-001",
        )
        await monitor.start()

        # 4. Agents work … monitor feeds BSC … whiteboard fills

        # 5. Query
        query = WhiteboardQuery(store)
        state = await query.compressed_state("sess-001")
        print(state)

        # 6. End of session
        await monitor.stop()
        snapshot = await query.snapshot("sess-001")
        # → hand to Nightly Synthesis agent (Sub-phase 10.4)
"""

from .monitor import WhiteboardMonitor, WatchedAgent
from .query import WhiteboardQuery
from .schema import SessionRecord, WhiteboardEntry, WhiteboardSnapshot
from .store import WhiteboardStore

__all__ = [
    # Store
    "WhiteboardStore",
    # Monitor
    "WhiteboardMonitor",
    "WatchedAgent",
    # Query
    "WhiteboardQuery",
    # Schema
    "WhiteboardEntry",
    "WhiteboardSnapshot",
    "SessionRecord",
]
