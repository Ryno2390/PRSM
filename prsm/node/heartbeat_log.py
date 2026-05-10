"""In-memory ring buffer for on-chain HeartbeatRecorded events.

Each HeartbeatRecorded event observed by the
StorageSlashingWatcher appends here. Operators verify scheduler
liveness via GET /admin/heartbeat-history.

Symmetric to SlashEventRing: same shape, separate concern. A
slash watcher reports both heartbeat-success (this ring) and
slash events (slash_event_log) — operators answer "are my
heartbeats landing on-chain?" without grepping JSON logs.

v1 in-process; restart loses buffer (events also persisted
on-chain).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass(frozen=True)
class HeartbeatRecordedEntry:
    timestamp: float
    provider: str
    onchain_timestamp: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "onchain_timestamp": self.onchain_timestamp,
        }


_DEFAULT_MAX_ENTRIES = 256


class HeartbeatRecordedRing:
    """Bounded in-memory ring of HeartbeatRecordedEntry records."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[HeartbeatRecordedEntry] = deque(maxlen=max_entries)

    def append(
        self,
        *,
        provider: str,
        onchain_timestamp: int,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(HeartbeatRecordedEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            provider=provider,
            onchain_timestamp=int(onchain_timestamp),
        ))

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
    ) -> List[HeartbeatRecordedEntry]:
        if limit <= 0 or limit > 1000:
            raise ValueError(f"limit must be in [1, 1000], got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        snap = list(self._entries)
        snap.reverse()
        if provider:
            p_lower = provider.lower()
            snap = [e for e in snap if e.provider.lower() == p_lower]
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)
