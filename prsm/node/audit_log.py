"""In-memory ring buffer for state-changing API requests.

Production-debugging feature: every non-GET API request appends
to a bounded ring buffer with (timestamp, method, path, requester,
status_code, request_id). Operators investigating a complaint or
unexpected state-change query the buffer via /audit/recent for a
quick view of recent writes.

Bounded by deque(maxlen=...) so memory is capped. v1 is in-process
only; future filesystem persistence (similar to JobHistoryStore's
v2 disk-backed mode) is the natural extension when in-process
restarts become a recovery concern.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


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


_DEFAULT_MAX_ENTRIES = 1024


class AuditLogRing:
    """Bounded in-memory ring buffer of AuditEntry records.

    Thread-safety: deque.append is atomic in CPython under the GIL,
    sufficient for v1's single-process use. List snapshots taken
    from the deque are independent copies so iterating the result
    of recent() doesn't race with concurrent appends.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[AuditEntry] = deque(maxlen=max_entries)

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
        self._entries.append(AuditEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            method=method,
            path=path,
            requester=requester,
            status_code=status_code,
            request_id=request_id,
        ))

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
