"""Per-peer, per-category rate limiting with throttle + ban escalation.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §3.6 + §6 Task 5.

Plan-specified default limits:

  * DHT queries:    100 / minute / peer
  * Direct messages: 500 / minute / peer
  * Shard dispatch:  50 / hour / peer (marketplace-layer; included here
    for completeness if the transport wants the same plumbing)

Escalation ladder:

  * Normal: request counts toward the sliding window.
  * Throttled (soft, 60s default): requests rejected with
    `RateLimitResult.THROTTLED`. Window keeps accumulating; a throttled
    peer cannot "wait out" their violation by spamming.
  * Banned (hard, 1h default): same as throttled but longer. Triggered
    when a peer accumulates enough throttle events to exceed
    `violations_for_ban` within `violation_memory_sec`.

Usage:

    limiter = RateLimiter(limits={"dht": RateLimit(100, 60.0)})
    result = limiter.check_and_consume(peer_id, "dht")
    if result is RateLimitResult.ALLOWED:
        handle_request()
    else:
        reject(result)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict


__all__ = [
    "RateLimit",
    "RateLimitResult",
    "RateLimiter",
    "DEFAULT_LIMITS",
]


class RateLimitResult(Enum):
    ALLOWED = "allowed"
    OVER_LIMIT = "over_limit"  # would exceed; triggers throttle
    THROTTLED = "throttled"    # soft throttle in effect
    BANNED = "banned"          # hard ban in effect


@dataclass(frozen=True)
class RateLimit:
    max_per_window: int
    window_sec: float


# Plan §3.6 defaults — exposed so callers can plug them in by name.
DEFAULT_LIMITS: Dict[str, RateLimit] = {
    "dht": RateLimit(max_per_window=100, window_sec=60.0),
    "direct_message": RateLimit(max_per_window=500, window_sec=60.0),
    "shard_dispatch": RateLimit(max_per_window=50, window_sec=3600.0),
}


@dataclass
class _PeerLimits:
    requests: Dict[str, Deque[float]] = field(default_factory=dict)
    violations: Deque[float] = field(default_factory=deque)
    throttled_until: float = 0.0
    banned_until: float = 0.0


class RateLimiter:
    def __init__(
        self,
        *,
        limits: Dict[str, RateLimit] | None = None,
        throttle_duration_sec: float = 60.0,
        ban_duration_sec: float = 3600.0,
        violations_for_ban: int = 3,
        violation_memory_sec: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if violations_for_ban < 1:
            raise ValueError("violations_for_ban must be >= 1")
        self._limits: Dict[str, RateLimit] = dict(limits or DEFAULT_LIMITS)
        self._throttle_duration = throttle_duration_sec
        self._ban_duration = ban_duration_sec
        self._violations_for_ban = violations_for_ban
        self._violation_memory = violation_memory_sec
        self._clock = clock
        self._peers: Dict[str, _PeerLimits] = {}

    # ---- primary API ------------------------------------------------------

    def check_and_consume(
        self, peer_id: str, category: str
    ) -> RateLimitResult:
        """Check the rate limit + (if allowed) record the request.

        Returns ALLOWED / OVER_LIMIT / THROTTLED / BANNED. OVER_LIMIT means
        "this specific request tipped the peer into throttled state" —
        subsequent requests during the throttle window return THROTTLED.
        """
        if category not in self._limits:
            raise ValueError(f"unknown category: {category}")
        limit = self._limits[category]
        now = self._clock()
        p = self._peers.setdefault(peer_id, _PeerLimits())

        # Banned / throttled states short-circuit before any window math.
        if now < p.banned_until:
            return RateLimitResult.BANNED
        if now < p.throttled_until:
            return RateLimitResult.THROTTLED

        # Window trim.
        ts_deque = p.requests.setdefault(category, deque())
        cutoff = now - limit.window_sec
        while ts_deque and ts_deque[0] < cutoff:
            ts_deque.popleft()

        # Over-limit → register a violation + engage throttle / maybe ban.
        if len(ts_deque) >= limit.max_per_window:
            self._register_violation(p, now)
            if self._should_ban(p, now):
                p.banned_until = now + self._ban_duration
                return RateLimitResult.BANNED
            p.throttled_until = now + self._throttle_duration
            return RateLimitResult.OVER_LIMIT

        # Allowed.
        ts_deque.append(now)
        return RateLimitResult.ALLOWED

    def state_of(self, peer_id: str) -> RateLimitResult:
        """Inspect current state without consuming a request."""
        p = self._peers.get(peer_id)
        if p is None:
            return RateLimitResult.ALLOWED
        now = self._clock()
        if now < p.banned_until:
            return RateLimitResult.BANNED
        if now < p.throttled_until:
            return RateLimitResult.THROTTLED
        return RateLimitResult.ALLOWED

    def unban(self, peer_id: str) -> None:
        """Manual override — clears throttle + ban + violation history."""
        p = self._peers.get(peer_id)
        if p is None:
            return
        p.throttled_until = 0.0
        p.banned_until = 0.0
        p.violations.clear()

    # ---- violation bookkeeping -------------------------------------------

    def _register_violation(self, p: _PeerLimits, now: float) -> None:
        cutoff = now - self._violation_memory
        while p.violations and p.violations[0] < cutoff:
            p.violations.popleft()
        p.violations.append(now)

    def _should_ban(self, p: _PeerLimits, now: float) -> bool:
        cutoff = now - self._violation_memory
        # Count fresh violations only (already trimmed in _register_violation
        # but keep defensive).
        while p.violations and p.violations[0] < cutoff:
            p.violations.popleft()
        return len(p.violations) >= self._violations_for_ban
