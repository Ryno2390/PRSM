"""In-memory ring buffer for state-changing API requests.

Production-debugging feature: every non-GET API request appends
to a bounded ring buffer with (timestamp, method, path, requester,
status_code, request_id). Operators investigating a complaint or
unexpected state-change query the buffer via /audit/recent for a
quick view of recent writes.

Bounded by deque(maxlen=...) so memory is capped.

v2 (2026-05-09): optional filesystem persistence via
``persist_dir`` — node restart no longer wipes the audit log;
operators have forensic continuity across restarts. v1
in-memory-only behavior preserved when ``persist_dir`` is None.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, List, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditEntry:
    """Single audit-log record. Frozen so consumers can't mutate
    in-place. The wire shape mirrors the dataclass field set."""

    timestamp: float          # unix epoch seconds
    method: str               # HTTP method (POST / PUT / PATCH / DELETE)
    path: str                 # request path
    requester: Optional[str]  # node id / identity that made the request
    status_code: int          # HTTP response code
    request_id: str           # X-Request-ID header value

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "method": self.method,
            "path": self.path,
            "requester": self.requester,
            "status_code": self.status_code,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuditEntry":
        return cls(
            timestamp=float(data["timestamp"]),
            method=str(data["method"]),
            path=str(data["path"]),
            requester=data.get("requester"),
            status_code=int(data["status_code"]),
            request_id=str(data["request_id"]),
        )


_DEFAULT_MAX_ENTRIES = 1024


class AuditLogRing:
    """Bounded in-memory ring buffer of AuditEntry records.

    Thread-safety: deque.append is atomic in CPython under the GIL,
    sufficient for v1's single-process use. List snapshots taken
    from the deque are independent copies so iterating the result
    of recent() doesn't race with concurrent appends.
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
        retention_days: Optional[float] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[AuditEntry] = deque(maxlen=max_entries)
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        self._retention_seconds: Optional[float] = (
            retention_days * 86400.0
            if retention_days is not None and retention_days > 0
            else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            if self._retention_seconds is not None:
                self._prune_old_disk_entries()
            self._load_from_disk()

    def _prune_old_disk_entries(self) -> None:
        """Delete disk files older than the retention window.
        Called once at startup when retention_days is set; the
        operator can re-trigger by restarting the node."""
        assert self._persist_dir is not None
        assert self._retention_seconds is not None
        cutoff = time.time() - self._retention_seconds
        deleted = 0
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                ts = float(data.get("timestamp", 0))
                if ts < cutoff:
                    path.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(
                    "AuditLogRing: retention prune skip %s: %s",
                    path, e,
                )
        if deleted > 0:
            logger.info(
                "AuditLogRing: pruned %d disk entries older "
                "than %s seconds",
                deleted, self._retention_seconds,
            )

    @staticmethod
    def _entry_filename(entry: AuditEntry) -> str:
        """Filename for an entry: timestamp + short hash of
        request_id for uniqueness. Sortable by name (timestamp
        prefix). Hash defends against same-timestamp collisions
        + filesystem-unsafe request_id chars."""
        h = hashlib.sha256(
            entry.request_id.encode("utf-8"),
        ).hexdigest()[:8]
        return f"{entry.timestamp:020.6f}-{h}.json"

    def _write_to_disk(self, entry: AuditEntry) -> None:
        if self._persist_dir is None:
            return
        try:
            path = self._persist_dir / self._entry_filename(entry)
            path.write_text(json.dumps(entry.to_dict()))
        except Exception as e:
            logger.warning(
                "AuditLogRing: disk write failed: %s", e,
            )

    def _load_from_disk(self) -> None:
        """Scan persist_dir on init + populate the ring. Records
        sorted by timestamp. Corrupt files logged + skipped.
        Oldest dropped if disk count exceeds max_entries."""
        assert self._persist_dir is not None
        loaded: list = []
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                entry = AuditEntry.from_dict(data)
                loaded.append(entry)
            except Exception as e:
                logger.warning(
                    "AuditLogRing: skipping corrupt file %s: %s",
                    path, e,
                )
        # Sort by timestamp ascending; deque maxlen will evict
        # oldest if list is longer than max_entries.
        loaded.sort(key=lambda e: e.timestamp)
        for entry in loaded:
            self._entries.append(entry)

    def append(
        self,
        *,
        method: str,
        path: str,
        requester: Optional[str],
        status_code: int,
        request_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        entry = AuditEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            method=method,
            path=path,
            requester=requester,
            status_code=status_code,
            request_id=request_id,
        )
        self._entries.append(entry)
        self._write_to_disk(entry)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Return the most-recent entries first, paginated."""
        if limit <= 0 or limit > 1000:
            raise ValueError(
                f"limit must be in [1, 1000], got {limit}"
            )
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        # Snapshot deque to a list for stable indexing under
        # concurrent appends.
        snap = list(self._entries)
        # Most-recent first.
        snap.reverse()
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)

    def max_entries(self) -> int:
        return self._max_entries
