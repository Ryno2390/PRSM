"""
Whiteboard Store
================

aiosqlite-backed persistent store for the Active Whiteboard.

The store is the single writer of whiteboard entries.  Only the BSC
Promoter (via the Monitor) should call ``write()``; all other callers
are readers.

Lifecycle
---------
1. ``await store.open()``     — creates / opens the SQLite file.
2. ``await store.write(decision)`` — BSC-promoted chunks arrive here.
3. ``store.compressed_state(session_id)`` — agents and the BSC read this.
4. ``await store.archive_session(session_id)`` — marks session as closed.
5. ``await store.close()``    — closes the aiosqlite connection.

The database is a single SQLite file stored at *db_path*.  Passing
``":memory:"`` gives an in-process, zero-persistence store suitable for
unit tests.

Schema
------
sessions(session_id, created_at, last_updated, meta)
entries(id, session_id, source_agent, chunk, surprise_score,
        raw_perplexity, similarity_score, timestamp,
        kl_reason, dedup_reason, extra)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiosqlite

from prsm.compute.nwtn.bsc import PromotionDecision
from .schema import SessionRecord, WhiteboardEntry, WhiteboardSnapshot

logger = logging.getLogger(__name__)

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    last_updated TEXT,
    meta         TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_ENTRIES = """
CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    source_agent    TEXT    NOT NULL,
    chunk           TEXT    NOT NULL,
    surprise_score  REAL    NOT NULL,
    raw_perplexity  REAL    NOT NULL,
    similarity_score REAL   NOT NULL,
    timestamp       TEXT    NOT NULL,
    kl_reason       TEXT    NOT NULL DEFAULT '',
    dedup_reason    TEXT    NOT NULL DEFAULT '',
    extra           TEXT    NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

_CREATE_IDX_SESSION = (
    "CREATE INDEX IF NOT EXISTS idx_entries_session "
    "ON entries(session_id, timestamp);"
)


class WhiteboardStore:
    """
    Async SQLite-backed store for the Active Whiteboard.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite file, or ``":memory:"`` for an in-process store.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        self._db_path = str(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open the database and create tables if they do not exist."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute(_CREATE_SESSIONS)
        await self._db.execute(_CREATE_ENTRIES)
        await self._db.execute(_CREATE_IDX_SESSION)
        await self._db.commit()
        logger.debug("WhiteboardStore opened: %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> "WhiteboardStore":
        await self.open()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def create_session(
        self, session_id: str, meta: Optional[Dict] = None
    ) -> SessionRecord:
        """Register a new session.  Idempotent: safe to call multiple times."""
        self._require_open()
        now = _now()
        await self._db.execute(
            "INSERT OR IGNORE INTO sessions(session_id, created_at, meta) "
            "VALUES (?, ?, ?)",
            (session_id, now, json.dumps(meta or {})),
        )
        await self._db.commit()
        return SessionRecord(session_id=session_id, created_at=_parse(now), meta=meta or {})

    async def archive_session(self, session_id: str) -> None:
        """Mark a session as closed (sets last_updated)."""
        self._require_open()
        await self._db.execute(
            "UPDATE sessions SET last_updated = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        await self._db.commit()

    async def clear_session(self, session_id: str) -> int:
        """Delete all entries for *session_id*.  Returns the number deleted."""
        self._require_open()
        async with self._db.execute(
            "DELETE FROM entries WHERE session_id = ?", (session_id,)
        ) as cur:
            count = cur.rowcount
        await self._db.commit()
        logger.info("Whiteboard cleared: session=%s (%d entries removed)", session_id, count)
        return count

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    async def write(self, decision: PromotionDecision) -> WhiteboardEntry:
        """
        Persist a promoted ``PromotionDecision`` to the whiteboard.

        Mutates ``decision.whiteboard_index`` in-place with the assigned row ID
        so callers can track which whiteboard entry corresponds to which decision.

        Parameters
        ----------
        decision : PromotionDecision
            Must have ``decision.promoted == True``.

        Returns
        -------
        WhiteboardEntry
            The persisted entry with its assigned ``id``.
        """
        self._require_open()
        if not decision.promoted:
            raise ValueError("Only promoted decisions should be written to the whiteboard")

        m = decision.metadata
        ts = m.timestamp.isoformat()
        kl_reason = decision.kl_result.reason if decision.kl_result else ""
        dedup_reason = (
            decision.dedup_result.reason if decision.dedup_result else ""
        )

        async with self._db.execute(
            """
            INSERT INTO entries
                (session_id, source_agent, chunk, surprise_score, raw_perplexity,
                 similarity_score, timestamp, kl_reason, dedup_reason, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.session_id,
                m.source_agent,
                decision.chunk,
                decision.surprise_score,
                decision.raw_perplexity,
                decision.similarity_score,
                ts,
                kl_reason,
                dedup_reason,
                json.dumps(m.extra),
            ),
        ) as cur:
            row_id = cur.lastrowid

        # Touch session last_updated
        await self._db.execute(
            "UPDATE sessions SET last_updated = ? WHERE session_id = ?",
            (_now(), m.session_id),
        )
        await self._db.commit()

        # Back-fill the whiteboard index on the decision object (dataclass, mutable)
        decision.whiteboard_index = row_id

        entry = WhiteboardEntry(
            id=row_id,
            session_id=m.session_id,
            source_agent=m.source_agent,
            chunk=decision.chunk,
            surprise_score=decision.surprise_score,
            raw_perplexity=decision.raw_perplexity,
            similarity_score=decision.similarity_score,
            timestamp=m.timestamp,
            kl_reason=kl_reason,
            dedup_reason=dedup_reason,
            extra=m.extra,
        )
        logger.debug(
            "Whiteboard entry #%d written: agent=%s score=%.3f",
            row_id, m.source_agent, decision.surprise_score,
        )
        return entry

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def get_all(self, session_id: str) -> List[WhiteboardEntry]:
        """Return all entries for *session_id* ordered by timestamp ascending."""
        self._require_open()
        async with self._db.execute(
            "SELECT * FROM entries WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_entry(r) for r in rows]

    async def get_recent(self, session_id: str, n: int = 10) -> List[WhiteboardEntry]:
        """Return the *n* most recent entries for *session_id*."""
        self._require_open()
        async with self._db.execute(
            "SELECT * FROM entries WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, n),
        ) as cur:
            rows = await cur.fetchall()
        return list(reversed([_row_to_entry(r) for r in rows]))

    async def get_by_agent(
        self, session_id: str, source_agent: str
    ) -> List[WhiteboardEntry]:
        """Return all entries from a specific agent."""
        self._require_open()
        async with self._db.execute(
            "SELECT * FROM entries WHERE session_id = ? AND source_agent = ? "
            "ORDER BY timestamp ASC",
            (session_id, source_agent),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_entry(r) for r in rows]

    async def get_top_surprise(
        self, session_id: str, n: int = 10
    ) -> List[WhiteboardEntry]:
        """Return the *n* most surprising entries."""
        self._require_open()
        async with self._db.execute(
            "SELECT * FROM entries WHERE session_id = ? "
            "ORDER BY surprise_score DESC LIMIT ?",
            (session_id, n),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_entry(r) for r in rows]

    async def entry_count(self, session_id: str) -> int:
        """Return the number of entries for *session_id*."""
        self._require_open()
        async with self._db.execute(
            "SELECT COUNT(*) FROM entries WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def snapshot(self, session_id: str) -> WhiteboardSnapshot:
        """Return a complete ``WhiteboardSnapshot`` for *session_id*."""
        entries = await self.get_all(session_id)

        # Fetch session metadata
        async with self._db.execute(
            "SELECT created_at, last_updated FROM sessions WHERE session_id = ?",
            (session_id,),
        ) as cur:
            srow = await cur.fetchone()

        created_at = _parse(srow["created_at"]) if srow else None
        last_updated = (
            _parse(srow["last_updated"])
            if srow and srow["last_updated"]
            else None
        )

        return WhiteboardSnapshot(
            session_id=session_id,
            entries=entries,
            created_at=created_at,
            last_updated=last_updated,
        )

    # ------------------------------------------------------------------
    # Compressed state (context string for BSC / agent consumption)
    # ------------------------------------------------------------------

    async def compressed_state(
        self,
        session_id: str,
        max_chars: int = 4000,
        strategy: str = "hybrid",
    ) -> str:
        """
        Return a compact, human-readable representation of the current whiteboard
        suitable for feeding to the BSC predictor as *context* and for agents to
        read as their shared situational awareness.

        Parameters
        ----------
        session_id : str
        max_chars : int
            Approximate maximum character count of the returned string.
        strategy : str
            ``"recent"``  — most recent entries only.
            ``"top"``     — highest surprise score entries only.
            ``"hybrid"``  — (default) half recent + half top, deduplicated,
                            sorted by timestamp.

        Returns
        -------
        str
            Formatted whiteboard string.
        """
        self._require_open()
        count = await self.entry_count(session_id)

        if count == 0:
            return f"[Whiteboard — Session: {session_id} — empty]"

        half = max(1, count // 2)

        if strategy == "recent":
            entries = await self.get_recent(session_id, count)
        elif strategy == "top":
            entries = await self.get_top_surprise(session_id, count)
        else:  # hybrid
            recent = await self.get_recent(session_id, half)
            top = await self.get_top_surprise(session_id, half)
            # deduplicate by id, preserve timestamp order
            seen: set = set()
            merged = []
            for e in recent + top:
                if e.id not in seen:
                    seen.add(e.id)
                    merged.append(e)
            entries = sorted(merged, key=lambda e: e.timestamp)

        header = (
            f"[Whiteboard — Session: {session_id} — "
            f"{count} entries from {len({e.source_agent for e in entries})} agent(s)]\n\n"
        )

        lines: list[str] = []
        budget = max_chars - len(header)

        for e in entries:
            line = f"[{e.agent_short} @ {e.timestamp_short}] {e.chunk}\n"
            if budget - len(line) < 0:
                # Truncate this entry to fit
                available = budget - 20
                if available > 0:
                    lines.append(f"[{e.agent_short} @ {e.timestamp_short}] {e.chunk[:available]}…\n")
                break
            lines.append(line)
            budget -= len(line)

        return header + "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._db is None:
            raise RuntimeError(
                "WhiteboardStore is not open. Call await store.open() first."
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _row_to_entry(row: aiosqlite.Row) -> WhiteboardEntry:
    return WhiteboardEntry(
        id=row["id"],
        session_id=row["session_id"],
        source_agent=row["source_agent"],
        chunk=row["chunk"],
        surprise_score=row["surprise_score"],
        raw_perplexity=row["raw_perplexity"],
        similarity_score=row["similarity_score"],
        timestamp=_parse(row["timestamp"]),
        kl_reason=row["kl_reason"] or "",
        dedup_reason=row["dedup_reason"] or "",
        extra=json.loads(row["extra"] or "{}"),
    )
