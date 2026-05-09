"""Per-requester token-bucket rate limiter for /compute/forge.

DoS-protection: cap requests/sec/requester. Default unset means
no limiting (v1 behavior preserved).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional


class SimpleTokenBucket:
    """In-memory token-bucket per requester.

    rate: tokens added per second (steady-state requests/sec).
    burst: max tokens (instantaneous burst capacity).

    Each requester has its own bucket initialized to ``burst``.
    Calls to ``try_consume`` either consume 1 token + return True,
    or reject + return False if the bucket is empty.
    """

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if burst <= 0:
            raise ValueError(f"burst must be positive, got {burst}")
        self.rate = float(rate)
        self.burst = int(burst)
        # {requester_id → (tokens, last_refill_ts)}
        self._buckets: Dict[str, list] = {}
        # Override-able for tests; defaults to time.time.
        self._now: Callable[[], float] = time.time

    def _refill(self, requester_id: str) -> None:
        now = self._now()
        if requester_id not in self._buckets:
            # Fresh bucket starts full.
            self._buckets[requester_id] = [float(self.burst), now]
            return
        tokens, last = self._buckets[requester_id]
        elapsed = now - last
        if elapsed > 0:
            tokens = min(
                float(self.burst), tokens + elapsed * self.rate,
            )
        self._buckets[requester_id] = [tokens, now]

    def try_consume(self, requester_id: str) -> bool:
        """Consume 1 token if available; return True/False."""
        self._refill(requester_id)
        bucket = self._buckets[requester_id]
        if bucket[0] >= 1.0:
            bucket[0] -= 1.0
            return True
        return False

    def retry_after(self, requester_id: str) -> float:
        """Seconds until the requester gets one more token. Zero
        when at least one token already available."""
        self._refill(requester_id)
        tokens = self._buckets[requester_id][0]
        if tokens >= 1.0:
            return 0.0
        deficit = 1.0 - tokens
        return deficit / self.rate


_GLOBAL_BUCKET: Optional[SimpleTokenBucket] = None
_GLOBAL_RATE: Optional[float] = None


def get_or_build_bucket(rate: Optional[float]) -> Optional[SimpleTokenBucket]:
    """Process-global cached bucket. Returns None when rate is
    falsy (no limiting). Rebuilds on rate change."""
    global _GLOBAL_BUCKET, _GLOBAL_RATE
    if not rate or rate <= 0:
        _GLOBAL_BUCKET = None
        _GLOBAL_RATE = None
        return None
    if _GLOBAL_BUCKET is None or _GLOBAL_RATE != rate:
        # Burst = rate (steady-state cap, no extra burst capacity).
        # Operators that want burst > rate can override by editing
        # this builder; the env var maps cleanly to "N/sec".
        burst = max(1, int(rate))
        _GLOBAL_BUCKET = SimpleTokenBucket(rate=rate, burst=burst)
        _GLOBAL_RATE = rate
    return _GLOBAL_BUCKET


def reset_global_bucket() -> None:
    """Test helper — wipe the process-global bucket between cases."""
    global _GLOBAL_BUCKET, _GLOBAL_RATE
    _GLOBAL_BUCKET = None
    _GLOBAL_RATE = None
