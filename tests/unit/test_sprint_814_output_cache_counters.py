"""Sprint 814 — OutputCache hit/miss/evict counters.

Sprints 810-813 shipped the cache + integration. Operators
enabling cache need observability: how often are we hitting?
how often are we evicting? Without counters they can't tune
PRSM_INFERENCE_OUTPUT_CACHE_TTL_S + _MAX_ENTRIES against
their workload.

Sprint 814 adds counter attributes to OutputCache:
  .hits      — get() returned a fresh value
  .misses    — get() returned None (absent or expired)
  .puts      — put() recorded a value (excludes "" sentinel)
  .evictions — put() pushed an entry out via LRU bound
  .ttl_evictions — get() evicted a stale entry on read

Plus a `stats() -> Dict[str, Any]` snapshot returning all
counters + current size + capacity + ttl. Pure data — no I/O,
no side effects beyond zeroing counters on `clear()`.

Pin tests:
- Counters start at 0.
- get() on miss increments misses.
- get() on hit increments hits.
- put() increments puts.
- put() that triggers LRU bound increments evictions.
- get() on expired entry: counts as miss + ttl_eviction.
- put("", ...) does NOT increment puts (empty-key sentinel).
- stats() returns dict with all counters + size + max_entries
  + ttl_seconds.
- clear() zeros all counters + drops entries.
"""
from __future__ import annotations


# ---- Initial state --------------------------------------------


def test_counters_start_at_zero():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    assert c.hits == 0
    assert c.misses == 0
    assert c.puts == 0
    assert c.evictions == 0
    assert c.ttl_evictions == 0


# ---- get() ----------------------------------------------------


def test_get_miss_increments_misses():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.get("nokey")
    assert c.misses == 1
    assert c.hits == 0


def test_get_hit_increments_hits():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.put("k", "v")
    c.get("k")
    assert c.hits == 1
    assert c.misses == 0


def test_get_empty_key_no_counter_change():
    """Empty-key sentinel hits the early-return path BEFORE
    counter increment — not a real miss."""
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.get("")
    assert c.misses == 0
    assert c.hits == 0


# ---- put() ---------------------------------------------------


def test_put_increments_puts():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.put("k", "v")
    assert c.puts == 1


def test_put_empty_key_no_counter_change():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.put("", "ignored")
    assert c.puts == 0


def test_lru_eviction_increments_evictions():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=2, ttl_seconds=3600)
    c.put("a", 1)
    c.put("b", 2)
    assert c.evictions == 0
    c.put("c", 3)  # evicts 'a'
    assert c.evictions == 1
    c.put("d", 4)  # evicts 'b'
    assert c.evictions == 2


# ---- TTL eviction on read ------------------------------------


def test_ttl_expired_counts_as_miss_and_ttl_eviction():
    from prsm.compute.inference.output_cache import OutputCache
    fake_now = [1000.0]

    def clock():
        return fake_now[0]

    c = OutputCache(
        max_entries=10, ttl_seconds=60, clock=clock,
    )
    c.put("k", "v")
    fake_now[0] = 1061.0  # past TTL
    result = c.get("k")
    assert result is None
    assert c.misses == 1
    assert c.ttl_evictions == 1


# ---- stats() snapshot ----------------------------------------


def test_stats_returns_full_snapshot():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=4, ttl_seconds=900)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")          # hit
    c.get("nokey")      # miss

    stats = c.stats()
    assert isinstance(stats, dict)
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["puts"] == 2
    assert stats["evictions"] == 0
    assert stats["ttl_evictions"] == 0
    assert stats["size"] == 2
    assert stats["max_entries"] == 4
    assert stats["ttl_seconds"] == 900.0


# ---- clear() zeros counters ----------------------------------


def test_clear_zeros_counters():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.put("k", "v")
    c.get("k")
    c.get("nokey")
    assert c.count() > 0
    c.clear()
    assert c.count() == 0
    assert c.hits == 0
    assert c.misses == 0
    assert c.puts == 0
    assert c.evictions == 0
    assert c.ttl_evictions == 0
