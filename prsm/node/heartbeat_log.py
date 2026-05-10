"""In-memory ring buffer for on-chain HeartbeatRecorded events.

Each HeartbeatRecorded event observed by the
StorageSlashingWatcher appends here. Operators verify scheduler
liveness via GET /admin/heartbeat-history.

Symmetric to SlashEventRing: same shape + same opt-in filesystem
persistence (sprint 92) via PRSM_HEARTBEAT_LOG_DIR env.
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
        self._entries: Deque[HeartbeatRecordedEntry] = deque(maxlen=max_entries)
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
                    "HeartbeatRecordedRing: skipping corrupt %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            self._entries.append(HeartbeatRecordedEntry(
                timestamp=d["timestamp"],
                provider=d["provider"],
                onchain_timestamp=d["onchain_timestamp"],
            ))

    def _write_to_disk(self, entry: HeartbeatRecordedEntry) -> None:
        if self._persist_dir is None:
            return
        ts_int = int(entry.timestamp)
        # Use onchain_timestamp + provider hash for uniqueness when
        # multiple heartbeats land in same wall-clock second.
        provider_short = entry.provider[:10] if entry.provider else "x"
        path = (
            self._persist_dir /
            f"{ts_int}_{entry.onchain_timestamp}_{provider_short}.json"
        )
        try:
            path.write_text(json.dumps(entry.to_dict()))
        except Exception as exc:
            logger.warning(
                "HeartbeatRecordedRing: disk write failed: %s", exc,
            )

    def append(
        self,
        *,
        provider: str,
        onchain_timestamp: int,
        timestamp: Optional[float] = None,
    ) -> None:
        entry = HeartbeatRecordedEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            provider=provider,
            onchain_timestamp=int(onchain_timestamp),
        )
        self._entries.append(entry)
        self._write_to_disk(entry)

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
