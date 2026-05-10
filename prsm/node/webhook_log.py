"""In-memory ring buffer for webhook delivery attempts.

Production-debugging feature: every webhook dispatch (whether
success or failure) appends to a bounded ring buffer with
(timestamp, event, url, success, attempts, status_code, error).
Operators verifying their webhook integration use
GET /admin/webhook-history to see what fired.

Bounded by deque(maxlen=...) so memory is capped. v1 is
in-process only; restart loses the buffer.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass(frozen=True)
class WebhookLogEntry:
    timestamp: float
    event: str
    url: str
    success: bool
    attempts: int
    status_code: Optional[int]
    error: Optional[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "url": self.url,
            "success": self.success,
            "attempts": self.attempts,
            "status_code": self.status_code,
            "error": self.error,
        }


_DEFAULT_MAX_ENTRIES = 256


class WebhookLogRing:
    """Bounded in-memory ring buffer of WebhookLogEntry records."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[WebhookLogEntry] = deque(maxlen=max_entries)

    def append(
        self,
        *,
        event: str,
        url: str,
        success: bool,
        attempts: int,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self._entries.append(WebhookLogEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            event=event,
            url=url,
            success=success,
            attempts=attempts,
            status_code=status_code,
            error=error,
        ))

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WebhookLogEntry]:
        if limit <= 0 or limit > 1000:
            raise ValueError(
                f"limit must be in [1, 1000], got {limit}"
            )
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        snap = list(self._entries)
        snap.reverse()  # most-recent first
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)

    def max_entries(self) -> int:
        return self._max_entries
