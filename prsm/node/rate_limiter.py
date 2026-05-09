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


_GLOBAL_BUCKETS: Dict[str, SimpleTokenBucket] = {}
_GLOBAL_RATES: Dict[str, float] = {}

# Backwards-compat aliases — older callers expecting the legacy
# single-bucket API (pre-2026-05-09) keep working.
_GLOBAL_BUCKET: Optional[SimpleTokenBucket] = None
_GLOBAL_RATE: Optional[float] = None


def get_or_build_bucket(
    rate: Optional[float],
    *,
    name: str = "_default",
) -> Optional[SimpleTokenBucket]:
    """Process-global cached bucket, keyed by ``name``. Returns
    None when rate is falsy (no limiting). Rebuilds on rate
    change.

    The ``name`` argument lets distinct endpoints maintain
    independent rate-limit windows. Default name preserves the
    pre-2026-05-09 single-bucket behavior for callers passing
    no name.
    """
    global _GLOBAL_BUCKET, _GLOBAL_RATE  # legacy aliases
    if not rate or rate <= 0:
        _GLOBAL_BUCKETS.pop(name, None)
        _GLOBAL_RATES.pop(name, None)
        if name == "_default":
            _GLOBAL_BUCKET = None
            _GLOBAL_RATE = None
        return None
    cached_rate = _GLOBAL_RATES.get(name)
    if name not in _GLOBAL_BUCKETS or cached_rate != rate:
        # Burst = rate (steady-state cap, no extra burst capacity).
        burst = max(1, int(rate))
        bucket = SimpleTokenBucket(rate=rate, burst=burst)
        _GLOBAL_BUCKETS[name] = bucket
        _GLOBAL_RATES[name] = rate
        if name == "_default":
            _GLOBAL_BUCKET = bucket
            _GLOBAL_RATE = rate
    return _GLOBAL_BUCKETS[name]


def reset_global_bucket(name: Optional[str] = None) -> None:
    """Test helper — wipe a named bucket (or all buckets when
    name is None) between cases."""
    global _GLOBAL_BUCKET, _GLOBAL_RATE
    if name is None:
        _GLOBAL_BUCKETS.clear()
        _GLOBAL_RATES.clear()
        _GLOBAL_BUCKET = None
        _GLOBAL_RATE = None
        return
    _GLOBAL_BUCKETS.pop(name, None)
    _GLOBAL_RATES.pop(name, None)
    if name == "_default":
        _GLOBAL_BUCKET = None
        _GLOBAL_RATE = None
