"""In-memory ring buffer for on-chain Distributed events.

Each Distributed event observed by the
CompensationDistributorWatcher appends here. Operators verify
emission rounds are landing on chain via
GET /admin/distribution-history.

Symmetric to SlashEventRing + HeartbeatRecordedRing: same
opt-in filesystem persistence (sprint 92) via
PRSM_DISTRIBUTION_LOG_DIR env.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedEntry:
    timestamp: float
    to_creator: int
    to_operator: int
    to_grant: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "to_creator": self.to_creator,
            "to_operator": self.to_operator,
            "to_grant": self.to_grant,
            "total_distributed": (
                self.to_creator + self.to_operator + self.to_grant
            ),
        }


_DEFAULT_MAX_ENTRIES = 256


class DistributedEventRing:
    """Bounded in-memory ring of DistributedEntry records."""

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
        self._entries: Deque[DistributedEntry] = deque(maxlen=max_entries)
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
            except Exception as exc:
                logger.warning(
                    "DistributedEventRing: skipping corrupt %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            self._entries.append(DistributedEntry(
                timestamp=d["timestamp"],
                to_creator=d["to_creator"],
                to_operator=d["to_operator"],
                to_grant=d["to_grant"],
            ))

    def _write_to_disk(self, entry: DistributedEntry) -> None:
        if self._persist_dir is None:
            return
        ts_int = int(entry.timestamp)
        # Distribution events occur ~hourly to ~daily; ts_int alone
        # is collision-free in practice. Add suffix for safety.
        path = (
            self._persist_dir /
            f"{ts_int}_{entry.to_creator}.json"
        )
        try:
            path.write_text(json.dumps(entry.to_dict()))
        except Exception as exc:
            logger.warning(
                "DistributedEventRing: disk write failed: %s", exc,
            )

    def append(
        self,
        *,
        to_creator: int,
        to_operator: int,
        to_grant: int,
        timestamp: Optional[float] = None,
    ) -> None:
        entry = DistributedEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            to_creator=int(to_creator),
            to_operator=int(to_operator),
            to_grant=int(to_grant),
        )
        self._entries.append(entry)
        self._write_to_disk(entry)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DistributedEntry]:
        if limit <= 0 or limit > 1000:
            raise ValueError(f"limit must be in [1, 1000], got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        snap = list(self._entries)
        snap.reverse()
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)
