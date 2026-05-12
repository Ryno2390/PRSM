"""Sprint 284 — webhook replay protection.

Sprint 283 ships HMAC signature verification: only payloads
signed by the vendor's secret pass. But a signed payload stays
signed forever — an attacker who captures one (via a leak, a
MITM that doesn't yield the secret, server log scrape, etc.)
can replay it indefinitely.

Two defenses ship together:

  1. Timestamp window. Persona embeds the signing timestamp
     in its header (``t=<unix>``). If the timestamp is more
     than ``tolerance_sec`` away from "now" (in either
     direction), reject. Default 300s matches Stripe /
     Coinbase Commerce convention.

  2. Signature-hash dedup ring. Vendor-agnostic. The full
     signature value is a perfect replay token — it's
     cryptographically unique per (body, timestamp, secret).
     Bounded in-memory ring tracks recently-seen signatures;
     second occurrence → reject.

Both work for Persona; the dedup ring is the primary defense
for Onfido (whose signature carries no timestamp). Apply both
when available for belt-and-suspenders.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, Set, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_TOLERANCE_SEC = 300  # 5 minutes
_DEFAULT_RING_SIZE = 10_000


def is_timestamp_fresh(
    ts_str: str,
    current_time: float,
    tolerance_sec: int = _DEFAULT_TOLERANCE_SEC,
) -> Tuple[bool, str]:
    """Check whether a vendor-supplied timestamp string is
    within the freshness window centered on ``current_time``.

    Returns (ok, reason). reason is "" on success; on failure
    it's a short human-readable explanation for logging
    (NOT for return to the vendor — info leak protection).

    Both stale (too old) and future-skewed timestamps are
    rejected. Symmetric tolerance prevents an attacker from
    exploiting clock skew in either direction.
    """
    if not ts_str:
        return (False, "empty timestamp")
    try:
        ts = float(ts_str)
    except (ValueError, TypeError):
        return (
            False,
            f"could not parse timestamp {ts_str!r}",
        )
    delta = abs(current_time - ts)
    if delta > tolerance_sec:
        return (
            False,
            f"timestamp out of window: "
            f"|now - t| = {delta:.1f}s > "
            f"tolerance {tolerance_sec}s",
        )
    return (True, "")


class WebhookReplayRing:
    """Bounded set of recently-seen webhook replay tokens.

    Eviction is FIFO (oldest seen first). The ring size
    governs how long a captured signature remains
    re-rejectable — for typical webhook volumes (≤ thousands
    per hour) 10K entries spans hours of history, well
    beyond the typical 300s timestamp window.

    Thread-safe via a single mutex.
    """

    def __init__(
        self, max_entries: int = _DEFAULT_RING_SIZE,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._order: Deque[str] = deque(maxlen=max_entries)
        self._set: Set[str] = set()
        self._lock = threading.Lock()

    def seen(self, token: str) -> bool:
        """Returns True if the token has been recorded
        within the retention window. Does NOT mutate."""
        with self._lock:
            return token in self._set

    def record(self, token: str) -> bool:
        """Record a token. Returns True on first occurrence,
        False on duplicate. Caller treats False as "replay
        detected — reject this webhook."""
        if not isinstance(token, str) or not token:
            raise ValueError(
                "token must be a non-empty string"
            )
        with self._lock:
            if token in self._set:
                return False
            # If the deque is at max capacity, the oldest
            # token will be evicted by the append — also
            # purge it from the lookup set.
            if len(self._order) >= self._max_entries:
                evicted = self._order[0]
                self._set.discard(evicted)
            self._order.append(token)
            self._set.add(token)
            return True

    def count(self) -> int:
        with self._lock:
            return len(self._set)
