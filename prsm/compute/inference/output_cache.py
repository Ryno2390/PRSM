"""Sprint 810 — inference output cache primitive.

For deterministic inference (privacy_tier=none), repeat identical
prompts can be served from cache instead of re-running the model.
Saves real operator compute time for hot queries (e.g., the same
"capital of France" sample inference an operator runs on
self-test).

Sprint 810 ships the PRIMITIVE only. Sprint 811 will wire it
into the executor's request path so cache hits skip the model
forward-pass entirely.

Privacy invariant:
    Only privacy_tier == "none" is cacheable. Other tiers
    (standard / high / maximum) require DP per-request — caching
    a DP-noised output and serving it back violates the privacy
    accounting. `make_cache_key` returns "" for non-none tiers
    so the cache is a no-op (`OutputCache.get("")` returns None,
    `OutputCache.put("", ...)` is dropped).

Receipt protocol caveat (for sprint 811):
    A cache hit MUST NOT replay an old receipt verbatim — the
    receipt's job_id should be unique per request and the
    settler_signature is bound to the signing-payload (which
    includes job_id). Sprint 811 will derive a fresh receipt
    at retrieval time, reusing only the deterministic output
    (text + output_hash + epsilon_spent which is 0 for "none").

Env:
    PRSM_INFERENCE_OUTPUT_CACHE_ENABLED=1     (unset = disabled)
    PRSM_INFERENCE_OUTPUT_CACHE_TTL_S=3600    (default 1h)
    PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES=1024
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple


_DEFAULT_MAX_ENTRIES = 1024
_DEFAULT_TTL_SECONDS = 3600.0


def make_cache_key(
    prompt: str,
    model_id: str,
    max_tokens: int,
    privacy_tier: str,
) -> str:
    """Return SHA-256 hex digest of cache key components.

    Returns "" when privacy_tier != "none" — privacy invariant.
    Callers wrap `make_cache_key(...)` results in `if key:`
    before calling `.get`/`.put`.
    """
    if privacy_tier != "none":
        return ""
    h = hashlib.sha256()
    h.update(b"prsm-output-cache-v1\n")
    h.update(model_id.encode("utf-8") + b"\n")
    h.update(str(int(max_tokens)).encode("utf-8") + b"\n")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def should_cache(privacy_tier: str) -> bool:
    """True only for privacy_tier == 'none'. Other tiers need
    DP per-request + MUST NOT be cached."""
    return privacy_tier == "none"


class OutputCache:
    """Bounded LRU cache with per-entry TTL.

    Stored values are arbitrary (the inference layer caches a
    serializable dict {output, output_hash, duration_seconds,
    cost_ftns, ...} — the cache itself is value-agnostic).
    """

    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()
        # Sprint 814 — observability counters. Operators tuning
        # TTL + MAX_ENTRIES need hit rate + eviction visibility.
        self.hits: int = 0
        self.misses: int = 0
        self.puts: int = 0
        self.evictions: int = 0
        self.ttl_evictions: int = 0

    def get(self, key: str) -> Optional[Tuple[Any, float]]:
        """Return (value, age_seconds) when fresh, else None.

        Empty-key sentinel returns None always (privacy invariant
        defense-in-depth) and does NOT count against miss
        counter — empty-key isn't a real lookup."""
        if not key:
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, ts = entry
        age = self._clock() - ts
        if age > self.ttl_seconds:
            # Lazy eviction on read; counts as both miss + ttl_eviction
            self._entries.pop(key, None)
            self.misses += 1
            self.ttl_evictions += 1
            return None
        # LRU bump
        self._entries.move_to_end(key)
        self.hits += 1
        return value, age

    def put(self, key: str, value: Any) -> None:
        if not key:
            # Defense-in-depth: empty-key puts are dropped + don't
            # count.
            return
        self._entries[key] = (value, self._clock())
        self._entries.move_to_end(key)
        self.puts += 1
        # Eviction
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def count(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        """Sprint 814 — snapshot of all counters + capacity.

        Returns a plain dict suitable for JSON serialization
        (/admin endpoint exposition + Prometheus scrape)."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "puts": self.puts,
            "evictions": self.evictions,
            "ttl_evictions": self.ttl_evictions,
            "size": len(self._entries),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> None:
        self._entries.clear()
        # Sprint 814 — zero counters on clear so operator-
        # driven cache invalidation gives a fresh observability
        # window.
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0
        self.ttl_evictions = 0


def resolve_output_cache_from_env() -> Optional[OutputCache]:
    """Build an OutputCache from PRSM_INFERENCE_OUTPUT_CACHE_*
    env vars. Returns None when _ENABLED unset (safe default —
    no cache, no behavior change)."""
    enabled = (os.environ.get(
        "PRSM_INFERENCE_OUTPUT_CACHE_ENABLED",
    ) or "").strip().lower()
    if enabled not in ("1", "true", "yes"):
        return None

    raw_ttl = (os.environ.get(
        "PRSM_INFERENCE_OUTPUT_CACHE_TTL_S",
    ) or "").strip()
    try:
        ttl = float(raw_ttl) if raw_ttl else _DEFAULT_TTL_SECONDS
        if ttl <= 0:
            ttl = _DEFAULT_TTL_SECONDS
    except (ValueError, TypeError):
        ttl = _DEFAULT_TTL_SECONDS

    raw_max = (os.environ.get(
        "PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES",
    ) or "").strip()
    try:
        max_entries = int(raw_max) if raw_max else _DEFAULT_MAX_ENTRIES
        if max_entries < 1:
            max_entries = _DEFAULT_MAX_ENTRIES
    except (ValueError, TypeError):
        max_entries = _DEFAULT_MAX_ENTRIES

    return OutputCache(
        max_entries=max_entries, ttl_seconds=ttl,
    )
