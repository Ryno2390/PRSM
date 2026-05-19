"""Persistent storage for watcher `last_processed_block` baselines.

Closes the afternoon-arc deferred item from
`project_phase78_afternoon_arc_2026_05_08.md`: today's first-tick
watcher semantics reset baseline at every restart, losing
startup-window events. With persistence, restarts pick up where
they left off.

Module surface:

  - LastProcessedBlockStore (Protocol) — load / save / delete
    keyed by per-watcher identifier
  - InMemoryLastProcessedBlockStore — for tests + ephemeral cases
  - FilesystemLastProcessedBlockStore — JSON file per watcher key
    under ~/.prsm/watchers/ by default

The watchers (KeyDistributionWatcher / StorageSlashingWatcher /
CompensationDistributorWatcher) accept an optional `state_store`
kwarg. When provided:

  - First tick: load persisted block; if found, use it (subsequent
    polling picks up from there); if missing/corrupt, fall back to
    chain-tip baseline + persist.
  - Each successful baseline advance: persist the new value.
  - Persistence write failures: log + continue (don't crash);
    next successful tick re-persists.

Watcher keys:
  "key_distribution"             — KeyDistributionWatcher
  "storage_slashing"             — StorageSlashingWatcher
  "compensation_distributor"     — CompensationDistributorWatcher
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Protocol


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────────


class LastProcessedBlockStore(Protocol):
    """Persist + retrieve `last_processed_block` baselines, keyed
    by per-watcher identifier."""

    def load(self, watcher_key: str) -> Optional[int]:
        """Return persisted block height for `watcher_key`, or None
        if missing / corrupt / unreachable."""
        ...

    def save(self, watcher_key: str, block: int) -> None:
        """Persist `block` for `watcher_key`. Raises on programmer
        error (negative block, etc.); fails-soft via logging on
        IO error."""
        ...

    def delete(self, watcher_key: str) -> None:
        """Remove the persisted entry for `watcher_key`. No-op if
        already absent."""
        ...


# ──────────────────────────────────────────────────────────────────────
# In-memory impl (tests + ephemeral runs)
# ──────────────────────────────────────────────────────────────────────


class InMemoryLastProcessedBlockStore:
    """Dict-backed store; baseline is lost on process exit."""

    def __init__(self) -> None:
        self._records: dict[str, int] = {}

    def load(self, watcher_key: str) -> Optional[int]:
        return self._records.get(watcher_key)

    def save(self, watcher_key: str, block: int) -> None:
        if not isinstance(block, int) or block < 0:
            raise ValueError(
                f"block must be non-negative int, got {block!r}"
            )
        self._records[watcher_key] = block

    def delete(self, watcher_key: str) -> None:
        self._records.pop(watcher_key, None)


# ──────────────────────────────────────────────────────────────────────
# Filesystem impl (production)
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_BASE_DIR = Path.home() / ".prsm" / "watchers"


class FilesystemLastProcessedBlockStore:
    """JSON-file-per-watcher store; one `<watcher_key>.json` file
    per persisted entry. Per-key files avoid concurrent-write
    contention between watchers (each watcher writes its own file).

    File contents:
        {
            "watcher_key": "<key>",
            "last_processed_block": <int>
        }

    base_dir defaults to `~/.prsm/watchers/` and is auto-created
    on first save. Read failures (missing file / corrupt JSON /
    unexpected shape) return None — the calling watcher falls
    back to chain-tip baseline. Write failures log at WARNING
    and don't propagate — the next successful tick re-persists.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_BASE_DIR

    def _path(self, watcher_key: str) -> Path:
        return self.base_dir / f"{watcher_key}.json"

    def load(self, watcher_key: str) -> Optional[int]:
        path = self._path(watcher_key)
        if not path.exists():
            return None
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "FilesystemLastProcessedBlockStore: corrupt or invalid "
                "JSON at %s (%s); treating as missing — watcher will "
                "fall back to chain-tip baseline",
                path, exc,
            )
            return None
        if not isinstance(body, dict):
            return None
        value = body.get("last_processed_block")
        if not isinstance(value, int) or value < 0:
            return None
        return value

    def save(self, watcher_key: str, block: int) -> None:
        if not isinstance(block, int) or block < 0:
            raise ValueError(
                f"block must be non-negative int, got {block!r}"
            )
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "watcher_key": watcher_key,
                "last_processed_block": block,
            }
            self._path(watcher_key).write_text(json.dumps(payload))
        except OSError as exc:
            # Log + continue. The next successful tick re-persists.
            logger.warning(
                "FilesystemLastProcessedBlockStore: failed to persist "
                "watcher_key=%s block=%d to %s: %s — next tick will "
                "retry",
                watcher_key, block, self.base_dir, exc,
            )

    def delete(self, watcher_key: str) -> None:
        path = self._path(watcher_key)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "FilesystemLastProcessedBlockStore: failed to delete "
                "%s: %s",
                path, exc,
            )


# ──────────────────────────────────────────────────────────────────────
# Sprint 549 — persistent event-level dedup
# ──────────────────────────────────────────────────────────────────────
#
# Sibling concern to LastProcessedBlockStore. The block store handles
# "where did I last scan to"; this handles "did I already dispatch
# THIS event". Without it, restart catch-up re-emits any events the
# previous run handled between callback dispatch and the post-loop
# baseline persist — duplicate distribution-log rows, duplicate
# webhook fires, etc. Sprint-544's InboundCheckpointStore.credited_
# deposits pattern carried over to the sibling watchers.
#
# Schema is intentionally minimal:
#   (watcher_key TEXT, tx_hash TEXT, log_index INT) PRIMARY KEY
# One row per (watcher, on-chain-event-identity) tuple.
# ──────────────────────────────────────────────────────────────────────


class EventDedupStore:
    """SQLite-backed (watcher_key, tx_hash, log_index) dedup.

    Lazily creates the parent dir + schema on first construction.
    Used by ``CompensationDistributorWatcher`` (sprint 549) and —
    deferred to sprints 550/551 — ``KeyDistributionWatcher`` +
    ``StorageSlashingWatcher``.
    """

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS processed_events ("
        "  watcher_key TEXT NOT NULL,"
        "  tx_hash TEXT NOT NULL,"
        "  log_index INTEGER NOT NULL,"
        "  PRIMARY KEY (watcher_key, tx_hash, log_index)"
        ")"
    )

    def __init__(self, db_path: str) -> None:
        import sqlite3
        self._db_path = db_path
        if db_path != ":memory:":
            parent = Path(db_path).parent
            if str(parent) and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False,
        )
        self._conn.execute(self._SCHEMA)
        self._conn.commit()

    def has_processed_event(
        self, watcher_key: str, tx_hash: str, log_index: int,
    ) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed_events "
            "WHERE watcher_key = ? AND tx_hash = ? "
            "AND log_index = ? LIMIT 1",
            (watcher_key, tx_hash, int(log_index)),
        ).fetchone()
        return row is not None

    def mark_processed_event(
        self, watcher_key: str, tx_hash: str, log_index: int,
    ) -> None:
        """Idempotent insert — calling twice on the same tuple is
        a no-op."""
        self._conn.execute(
            "INSERT OR IGNORE INTO processed_events "
            "(watcher_key, tx_hash, log_index) VALUES (?, ?, ?)",
            (watcher_key, tx_hash, int(log_index)),
        )
        self._conn.commit()

    def summary(self) -> dict:
        """Sprint 552: per-watcher rollup for the
        ``/admin/watcher-event-dedup`` operator visibility endpoint.

        Returns a dict keyed by ``watcher_key`` with three fields:
          - ``rows_processed`` — total dedup rows for this watcher
          - ``latest_tx_hash`` — most-recently-marked tx_hash
            (rowid ordering — INSERT OR IGNORE on existing tuples
            does NOT bump rowid, so "most recent" means the last
            distinct event marked)
          - ``latest_log_index`` — log_index from that same row

        Empty dict if no events processed yet.
        """
        # Aggregate count per watcher_key. Latest row per watcher_key
        # is the max(rowid) — SQLite assigns rowids monotonically on
        # INSERT, and INSERT OR IGNORE doesn't bump on duplicate, so
        # this is "the last distinct event added".
        rows = self._conn.execute(
            "SELECT watcher_key, COUNT(*) AS n, "
            "       MAX(rowid) AS max_rid "
            "FROM processed_events GROUP BY watcher_key"
        ).fetchall()
        out: dict = {}
        for watcher_key, count, max_rid in rows:
            latest = self._conn.execute(
                "SELECT tx_hash, log_index FROM processed_events "
                "WHERE rowid = ?",
                (max_rid,),
            ).fetchone()
            tx_hash, log_idx = (latest or (None, None))
            out[watcher_key] = {
                "rows_processed": int(count),
                "latest_tx_hash": tx_hash,
                "latest_log_index": (
                    int(log_idx) if log_idx is not None else None
                ),
            }
        return out
