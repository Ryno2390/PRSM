"""Sprint 249 — in-memory ring of on-chain content-royalty
dispatch results.

The sprint-248 forge settlement block fires one ``distribute_
royalty`` tx per contributing shard and logs sent/skipped/
failed counts. Operators need finer-grained inspection (which
shard for which job at what tx_hash, when), so this ring buffer
captures the per-shard ``DispatchResult`` records and is
surfaced via GET /admin/royalty-dispatch-history.

Symmetric in shape to HeartbeatRecordedRing + SlashEventRing:
in-memory deque + optional filesystem persistence under
``PRSM_ROYALTY_DISPATCH_LOG_DIR``.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoyaltyDispatchEntry:
    timestamp: float
    job_id: str
    cid: str
    status: str  # sent|skipped_no_record|skipped_bad_hash|failed
    tx_hash: Optional[str]
    gross_wei: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "cid": self.cid,
            "status": self.status,
            "tx_hash": self.tx_hash,
            "gross_wei": self.gross_wei,
            "error": self.error,
        }


_DEFAULT_MAX_ENTRIES = 1024


class RoyaltyDispatchRing:
    """Bounded in-memory ring of RoyaltyDispatchEntry records."""

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[RoyaltyDispatchEntry] = deque(
            maxlen=max_entries,
        )
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        loaded: list = []
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                loaded.append(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RoyaltyDispatchRing: skipping corrupt %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            try:
                self._entries.append(RoyaltyDispatchEntry(
                    timestamp=d["timestamp"],
                    job_id=d["job_id"],
                    cid=d["cid"],
                    status=d["status"],
                    tx_hash=d.get("tx_hash"),
                    gross_wei=int(d.get("gross_wei", 0)),
                    error=d.get("error"),
                ))
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "RoyaltyDispatchRing: malformed entry %r: %s",
                    d, exc,
                )

    def _write_to_disk(self, entry: RoyaltyDispatchEntry) -> None:
        if self._persist_dir is None:
            return
        ts_int = int(entry.timestamp * 1000)
        path = (
            self._persist_dir /
            f"{ts_int}_{entry.job_id[:10]}_{entry.cid[:10]}.json"
        )
        try:
            path.write_text(json.dumps(entry.to_dict()))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RoyaltyDispatchRing: disk write failed: %s", exc,
            )

    def append(
        self,
        *,
        job_id: str,
        cid: str,
        status: str,
        tx_hash: Optional[str],
        gross_wei: int,
        error: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        entry = RoyaltyDispatchEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            job_id=job_id,
            cid=cid,
            status=status,
            tx_hash=tx_hash,
            gross_wei=int(gross_wei),
            error=error,
        )
        self._entries.append(entry)
        self._write_to_disk(entry)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> List[RoyaltyDispatchEntry]:
        if not isinstance(limit, int) or limit <= 0 or limit > 1000:
            raise ValueError(
                f"limit must be in [1, 1000], got {limit}"
            )
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(
                f"offset must be >= 0, got {offset}"
            )
        snap = list(self._entries)
        snap.reverse()
        if status:
            snap = [e for e in snap if e.status == status]
        if job_id:
            snap = [e for e in snap if e.job_id == job_id]
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)
