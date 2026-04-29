"""Phase 3.x.11 — server-side KV-cache lifecycle manager.

The sharded autoregressive decode path keeps each stage's
KV-cache **local** between tokens. The cache survives across
multiple ``INCREMENTAL`` dispatches for the same ``request_id``
so each per-token forward only computes one new attention
position rather than recomputing the full prompt.

This module owns the *handles* — the lifecycle metadata + LRU
bookkeeping. The actual tensor payload is opaque to the
manager: it's whatever the runner produces during its forward
pass (per-layer K + V tensors of shape
``[batch=1, n_heads, seq_len, head_dim]``). The runner attaches
the payload via ``handle.payload``; the manager doesn't
introspect it.

Lifecycle:
  - **Allocate** on PREFILL — fresh handle keyed on
    ``request_id``. Raises ``CacheAlreadyAllocatedError`` if a
    cache for this id already exists (caller bug — re-issuing
    PREFILL without an explicit ``evict`` first).
  - **Get** on INCREMENTAL — returns the existing handle and
    touches its LRU position + ``last_touch_time``. Returns
    ``None`` if no cache exists for this id (caller bug
    or cache was evicted out-from-under the request).
  - **Evict** on terminal (EOS / max_tokens) or on executor
    cancellation. Idempotent — evicting an unknown id is a
    no-op (matches the GeneratorExit cleanup path which may
    race with TTL eviction).
  - **Evict_idle** — periodic sweeper that drops handles past
    ``ttl_seconds``. Returns the list of evicted ids so the
    caller can log / metric.

Eviction policies:
  - **TTL** — bounds long-lived zombie-request memory if the
    executor crashed without explicit eviction.
  - **LRU cap** — when ``allocate`` is called and the pool is
    full, the LRU handle is evicted to make room. Operator
    config: ``max_cached_requests`` (default 64).

Thread safety: all public methods take a lock. Multiple
concurrent dispatches across different request_ids land on the
same manager; allocate / get / evict are atomic w.r.t. each
other. The lock is short-held — long operations on the handle
(building the cache payload during forward) happen OUTSIDE
the lock with the handle as a local reference.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


# Phase 3.x.11.y — speculative-decoding cache truncation hook.
# Called by ``KVCacheManager.rollback`` with the handle's payload +
# the requested drop count; returns the truncated payload (typically
# the same reference, mutated in place — matches the
# forward_incremental contract). Caller (the runner) injects this
# at rollback time; the manager is payload-opaque.
TruncateCacheFn = Callable[[Any, int], Any]


__all__ = [
    "KVCacheHandle",
    "KVCacheManager",
    "CacheAlreadyAllocatedError",
    "CacheNotFoundError",
    "TruncateCacheFn",
]


class CacheAlreadyAllocatedError(RuntimeError):
    """Raised by ``KVCacheManager.allocate`` when a cache for
    the requested ``request_id`` already exists. This is a
    caller bug — the caller should ``evict`` before re-allocating
    or use ``get`` to access an existing handle."""


class CacheNotFoundError(RuntimeError):
    """Raised by ``KVCacheManager.evict`` when configured to
    fail on unknown ids. The default behavior is idempotent
    (no-op on unknown id), but tests + diagnostic paths can
    opt into strict mode."""


@dataclass
class KVCacheHandle:
    """Lifecycle metadata + opaque payload for one
    request-scoped KV-cache.

    The ``payload`` field is set by the runner during PREFILL
    (initial allocation) and read+mutated during INCREMENTAL
    dispatches. The manager doesn't introspect it — the payload
    shape is runner-defined (typically
    ``List[Tuple[k_tensor, v_tensor]]`` indexed by layer).

    ``last_touch_time`` is updated by ``KVCacheManager.get``
    and used by ``evict_idle`` to drop stale handles.

    ``tokens_generated`` is bumped by the tail variant of
    ``ShardedAutoregressiveRunner`` (Phase 3.x.11 Task 4) on
    each successful sample. The tail uses it to detect
    ``max_tokens`` cap (sets ``is_terminal=True`` when the
    counter reaches the request's ceiling). Non-tail stages
    never read or write this field — it stays 0 on non-tail
    runners.

    ``cached_positions`` (Phase 3.x.11.y Task 9 round-1 HIGH-1
    remediation) tracks the count of positions currently held
    in the KV-cache payload. PREFILL bumps by ``seq_len``;
    INCREMENTAL by 1; VERIFY by K+1. ALL stages (tail + non-
    tail) bump this counter — it's load-bearing for
    speculative-decoding rollback to clamp correctly on non-
    tail stages whose ``tokens_generated`` stays 0. Without
    this counter, ``rollback`` would silently no-op on non-
    tail stages (tail-only ``tokens_generated`` clamp →
    ``min(n_positions, 0) == 0`` → no truncate_fn call → cache
    grows unbounded with rejected speculative suffixes).

    Mutable on purpose: the runner's incremental forward pass
    needs to mutate ``payload`` in place + the tail bumps
    ``tokens_generated`` per sample + every forward bumps
    ``cached_positions``. The dataclass is NOT frozen.
    """

    request_id: str
    n_layers: int
    last_touch_time: float
    payload: Any = None
    tokens_generated: int = 0
    cached_positions: int = 0


class KVCacheManager:
    """Per-server cache lifecycle. Operator-config-driven LRU
    + TTL eviction; thread-safe.

    Constructor args:
      max_cached_requests   Default 64. Concurrent-request
                            ceiling. When ``allocate`` is
                            called with the pool full, the LRU
                            handle is evicted to make room.
      ttl_seconds           Default 300.0 (5 minutes). Idle
                            cache lifetime. ``evict_idle``
                            drops handles whose
                            ``last_touch_time`` is older than
                            ``now - ttl_seconds``.
      clock                 Defaults to ``time.monotonic``.
                            Tests inject a controllable clock
                            to drive TTL deterministically.
    """

    def __init__(
        self,
        *,
        max_cached_requests: int = 64,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(max_cached_requests, bool)
            or not isinstance(max_cached_requests, int)
        ):
            raise RuntimeError(
                f"KVCacheManager: max_cached_requests must be int, "
                f"got {type(max_cached_requests).__name__}"
            )
        if max_cached_requests <= 0:
            raise RuntimeError(
                f"KVCacheManager: max_cached_requests must be "
                f"positive, got {max_cached_requests}"
            )
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
        ):
            raise RuntimeError(
                f"KVCacheManager: ttl_seconds must be number, got "
                f"{type(ttl_seconds).__name__}"
            )
        if ttl_seconds <= 0:
            raise RuntimeError(
                f"KVCacheManager: ttl_seconds must be positive, "
                f"got {ttl_seconds}"
            )
        self._max = int(max_cached_requests)
        self._ttl = float(ttl_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        # OrderedDict gives us O(1) LRU bookkeeping —
        # ``move_to_end`` after each ``get`` puts the touched
        # handle at the right side; the LRU is at the left.
        self._handles: "OrderedDict[str, KVCacheHandle]" = OrderedDict()

    # ── public API ────────────────────────────────────────────────────

    def allocate(
        self, request_id: str, n_layers: int,
    ) -> KVCacheHandle:
        """Create a fresh handle for ``request_id``. Called by
        the runner on PREFILL. Raises
        ``CacheAlreadyAllocatedError`` if a cache already
        exists for this id. Evicts the LRU handle if the pool
        is at ``max_cached_requests``."""
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(
                "KVCacheManager.allocate: request_id must be "
                "non-empty string"
            )
        if (
            isinstance(n_layers, bool)
            or not isinstance(n_layers, int)
            or n_layers <= 0
        ):
            raise RuntimeError(
                f"KVCacheManager.allocate: n_layers must be "
                f"positive int, got {n_layers!r}"
            )
        with self._lock:
            if request_id in self._handles:
                raise CacheAlreadyAllocatedError(
                    f"KVCacheManager: cache already allocated for "
                    f"request_id={request_id!r}; caller must evict "
                    f"first or use get() to access existing handle"
                )
            # LRU eviction if we're at cap. Running BEFORE we
            # insert the new handle so the cap is respected
            # post-insert.
            while len(self._handles) >= self._max:
                _evicted_id, _evicted_handle = self._handles.popitem(
                    last=False,
                )
                # The evicted handle's payload may hold GPU
                # memory; releasing the local reference here
                # invokes the runner-side cleanup at the next
                # GC cycle. (Operators with strict-real-time
                # cleanup needs can wrap KVCacheHandle.payload
                # with explicit ``.close()`` — out of v1 scope.)
            handle = KVCacheHandle(
                request_id=request_id,
                n_layers=n_layers,
                last_touch_time=self._clock(),
            )
            self._handles[request_id] = handle
            return handle

    def get(self, request_id: str) -> Optional[KVCacheHandle]:
        """Return the handle for ``request_id`` and touch its
        LRU position + ``last_touch_time``. Returns ``None`` if
        no cache exists (caller — typically the runner on
        INCREMENTAL — should treat None as a request-state
        error and surface MALFORMED_REQUEST)."""
        with self._lock:
            handle = self._handles.get(request_id)
            if handle is None:
                return None
            self._handles.move_to_end(request_id, last=True)
            handle.last_touch_time = self._clock()
            return handle

    def evict(self, request_id: str) -> bool:
        """Drop the handle for ``request_id``. Idempotent:
        returns ``True`` if a handle was evicted, ``False`` if
        none existed. The False path is the common case for
        TTL/LRU + explicit-evict races."""
        with self._lock:
            handle = self._handles.pop(request_id, None)
            return handle is not None

    def rollback(
        self,
        request_id: str,
        n_positions: int,
        truncate_fn: TruncateCacheFn,
    ) -> Tuple[bool, int]:
        """Phase 3.x.11.y — truncate the LAST ``n_positions`` from
        the cache. Used by the speculative-decoding executor
        loop on rejected speculative suffix.

        ``truncate_fn(payload, actual_n) -> updated_payload`` is
        invoked under the manager's lock with the actual drop
        count; the returned payload is stored on the handle.
        Lock-held truncation prevents races where another thread
        reads the handle between truncation and the
        ``cached_positions`` decrement (catches a read-stale-cache
        bug pattern).

        **Phase 3.x.11.y Task 9 round-1 HIGH-1 remediation.**
        The clamp uses ``cached_positions`` (bumped on EVERY
        forward — PREFILL by ``seq_len``, INCREMENTAL by 1,
        VERIFY by K+1, on every stage tail or non-tail), NOT
        ``tokens_generated`` (tail-only counter). Without the
        cached_positions counter, rollback against a non-tail
        stage's handle silently no-ops (``min(N, 0) == 0``)
        and the model's KV-cache grows unbounded with
        speculatively-cached-but-rejected positions, breaking
        non-tail forward semantics on the next iteration.

        Idempotent paths:
          - unknown request_id → ``(False, 0)``
          - n_positions <= 0 → ``(False, 0)``
          - handle has zero cached positions → ``(False, 0)``
          - n_positions > cached_positions → drops everything
            cached; returns ``(True, prior_cached_positions)``

        Returns ``(rolled_back, actual_dropped)``. ``rolled_back``
        True iff at least one position was dropped;
        ``actual_dropped`` is the count actually removed (may be
        less than ``n_positions`` if cache had fewer cached
        positions).

        ``truncate_fn`` MUST NOT raise — exceptions propagate to
        the caller and the handle is left in a torn state. The
        runner-side ``model.truncate_cache`` is the typical
        impl; tests inject a mock.

        Side-effect on ``tokens_generated``: this counter is
        decremented BY THE CALLER (the runner) only when the
        runner is tail-capable AND the runner judges that
        emitted-but-now-rolled-back tokens should retroactively
        free up max_tokens budget. The manager doesn't touch
        ``tokens_generated`` here — the tail's max_tokens
        accounting is intentionally one-directional (emitted
        tokens count permanently against the cap, even if a
        downstream consumer cancellation rolls back the cache).
        """
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(
                "KVCacheManager.rollback: request_id must be "
                "non-empty string"
            )
        if (
            isinstance(n_positions, bool)
            or not isinstance(n_positions, int)
        ):
            raise RuntimeError(
                f"KVCacheManager.rollback: n_positions must be "
                f"int, got {type(n_positions).__name__}"
            )
        if not callable(truncate_fn):
            raise RuntimeError(
                "KVCacheManager.rollback: truncate_fn must be "
                "callable (typically model.truncate_cache)"
            )
        with self._lock:
            handle = self._handles.get(request_id)
            if handle is None:
                return (False, 0)
            if n_positions <= 0:
                return (False, 0)
            actual = min(n_positions, handle.cached_positions)
            if actual == 0:
                return (False, 0)
            updated_payload = truncate_fn(handle.payload, actual)
            handle.payload = updated_payload
            handle.cached_positions -= actual
            return (True, actual)

    def evict_idle(self) -> List[str]:
        """Drop handles past ``ttl_seconds`` since their last
        touch. Returns the list of evicted request_ids so the
        caller can log / metric. Intended for periodic sweeper
        threads — operators wire this to a background task at
        node startup.
        """
        cutoff = self._clock() - self._ttl
        evicted: List[str] = []
        with self._lock:
            # Iterate over a snapshot since we mutate the dict.
            for request_id, handle in list(self._handles.items()):
                if handle.last_touch_time < cutoff:
                    self._handles.pop(request_id, None)
                    evicted.append(request_id)
        return evicted

    def evict_all(self) -> List[str]:
        """Drop ALL handles. Operator-shutdown helper; returns
        the list of evicted ids."""
        with self._lock:
            evicted = list(self._handles.keys())
            self._handles.clear()
        return evicted

    def __len__(self) -> int:
        with self._lock:
            return len(self._handles)

    def __contains__(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._handles
