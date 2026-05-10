"""In-memory ring buffer for on-chain Distributed events.

Each Distributed event observed by the
CompensationDistributorWatcher appends here. Operators verify
emission rounds are landing on chain via
GET /admin/distribution-history.

Symmetric to SlashEventRing + HeartbeatRecordedRing: same
shape, separate concern. Each Distributed event records the
(to_creator, to_operator, to_grant) split.

v1 in-process; restart loses buffer (events still on-chain
authoritative).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


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

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[DistributedEntry] = deque(maxlen=max_entries)

    def append(
        self,
        *,
        to_creator: int,
        to_operator: int,
        to_grant: int,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(DistributedEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            to_creator=int(to_creator),
            to_operator=int(to_operator),
            to_grant=int(to_grant),
        ))

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
