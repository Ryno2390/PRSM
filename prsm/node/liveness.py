"""Ping-based peer liveness tracking with miss-threshold eviction.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §3.4 + §6 Task 5.

Usage pattern (one tick per `ping_interval_sec`, driven by the transport
loop):

    monitor = LivenessMonitor(ping_interval_sec=30, dead_threshold=3)
    monitor.register(peer_id)
    ...
    while True:
        result = monitor.tick()
        for pid in result.due_for_ping:
            transport.send_ping(pid)
            monitor.record_ping_sent(pid)
        for pid in result.evicted:
            transport.close(pid)
        # on pong arrival (any time between ticks):
        #   monitor.record_pong_received(peer_id)

Design decisions:

  * Explicit `tick()` rather than an internal timer — the transport owns
    the scheduling context. Testability + determinism follow.
  * `ping_outstanding` flag prevents double-counting a single unanswered
    ping as multiple consecutive misses — each ping contributes at most
    one miss before a new ping is sent or the peer is evicted.
  * Miss counter resets on any successful pong, not only the pong that
    matches the outstanding ping. Late pongs count as recovery.
  * Evicted peers are not re-pinged. Call `register()` again to reinstate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List


__all__ = [
    "LivenessMonitor",
    "TickResult",
]


@dataclass
class _PeerState:
    last_ping_sent: float = 0.0
    last_pong_received: float = 0.0
    consecutive_missed: int = 0
    ping_outstanding: bool = False
    alive: bool = True


@dataclass(frozen=True)
class TickResult:
    due_for_ping: List[str]
    evicted: List[str]


class LivenessMonitor:
    def __init__(
        self,
        *,
        ping_interval_sec: float = 30.0,
        dead_threshold: int = 3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if dead_threshold < 1:
            raise ValueError("dead_threshold must be >= 1")
        if ping_interval_sec <= 0:
            raise ValueError("ping_interval_sec must be > 0")
        self._interval = ping_interval_sec
        self._threshold = dead_threshold
        self._clock = clock
        self._peers: Dict[str, _PeerState] = {}

    # ---- registration -----------------------------------------------------

    def register(self, peer_id: str) -> None:
        """Start tracking a peer. Peer is marked alive with no outstanding
        ping. If the peer is already registered, this is a no-op."""
        if peer_id not in self._peers:
            self._peers[peer_id] = _PeerState(
                last_pong_received=self._clock(),
            )

    def unregister(self, peer_id: str) -> None:
        self._peers.pop(peer_id, None)

    def is_alive(self, peer_id: str) -> bool:
        s = self._peers.get(peer_id)
        return s is not None and s.alive

    def tracked_peers(self) -> List[str]:
        return list(self._peers.keys())

    # ---- event recording --------------------------------------------------

    def record_ping_sent(self, peer_id: str) -> None:
        s = self._peers.get(peer_id)
        if s is None or not s.alive:
            return
        s.last_ping_sent = self._clock()
        s.ping_outstanding = True

    def record_pong_received(self, peer_id: str) -> None:
        s = self._peers.get(peer_id)
        if s is None or not s.alive:
            return
        s.last_pong_received = self._clock()
        s.ping_outstanding = False
        s.consecutive_missed = 0

    # ---- per-interval evaluation -----------------------------------------

    def tick(self) -> TickResult:
        """Evaluate outstanding pings + compute the ping-due list.

        Intended to be called once per `ping_interval_sec`. Can be called
        more frequently — the interval check inside guards double-counting.
        """
        now = self._clock()
        due: List[str] = []
        evicted: List[str] = []

        for pid, s in list(self._peers.items()):
            if not s.alive:
                continue

            # Outstanding ping that has aged past interval counts as one miss.
            if s.ping_outstanding and (now - s.last_ping_sent) >= self._interval:
                s.consecutive_missed += 1
                s.ping_outstanding = False
                if s.consecutive_missed >= self._threshold:
                    s.alive = False
                    evicted.append(pid)
                    continue

            # Peer is due for a (re-)ping if no ping outstanding and either
            # never pinged or last ping was at least interval ago.
            never_pinged = s.last_ping_sent == 0.0
            last_ping_aged = (now - s.last_ping_sent) >= self._interval
            if not s.ping_outstanding and (never_pinged or last_ping_aged):
                due.append(pid)

        return TickResult(due_for_ping=due, evicted=evicted)
