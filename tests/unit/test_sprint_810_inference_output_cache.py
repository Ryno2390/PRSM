"""Sprint 810 — inference output cache primitive.

For deterministic inference requests (privacy_tier=none), repeat
identical prompts can be served from cache instead of re-running
the model. Saves real operator compute time for hot queries.

Sprint 810 ships the PRIMITIVE: a pure data structure + ENV
config + cache-key derivation. Sprint 811 will wire it into the
executor's request path.

  prsm/compute/inference/output_cache.py:
    make_cache_key(prompt, model_id, max_tokens, privacy_tier) -> str
        Canonical SHA-256 hex digest of the cache key components.
        Returns "" for non-cacheable tiers (privacy_tier != "none").

    should_cache(privacy_tier) -> bool
        True only for privacy_tier == "none". Other tiers need
        DP per-request and MUST NOT be cached.

    OutputCache(max_entries, ttl_seconds, clock=time.monotonic)
        LRU bounded cache. .get(key) returns (value, age_seconds)
        when fresh; None when missing or expired. .put(key, value)
        records timestamp. .count() / .clear() helpers.

    resolve_output_cache_from_env() -> Optional[OutputCache]
        Reads PRSM_INFERENCE_OUTPUT_CACHE_ENABLED + _TTL_S +
        _MAX_ENTRIES. Disabled by default; returns None.

Privacy invariant: cache key SHA-256 includes privacy_tier as
a defense-in-depth bit. Even if a caller accidentally tries to
get/put for a non-"none" tier, make_cache_key returns "" so the
cache becomes a no-op (.get with "" always returns None).

Pin tests for make_cache_key:
- Returns SHA-256 hex digest (64 chars) for tier="none".
- Different prompts → different keys.
- Different model_ids → different keys.
- Different max_tokens → different keys.
- Same (prompt, model, max_tokens, tier=none) → same key.
- privacy_tier != "none" → returns "" (no-cache sentinel).

Pin tests for should_cache:
- "none" → True; "standard"/"high"/"maximum" → False.

Pin tests for OutputCache:
- Empty cache: .get(any) → None.
- put + get round-trips value.
- LRU bound: max_entries=3, 4 puts → oldest dropped.
- TTL: entry older than ttl_seconds → get returns None.
- TTL: entry within ttl returns value + age.
- .get("") → None (no-cache sentinel never hits).

Pin tests for resolve_output_cache_from_env:
- Env unset → None.
- _ENABLED=1 → returns OutputCache instance.
- _TTL_S + _MAX_ENTRIES env values respected.
- Malformed values fall back to defaults.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock


def setup_function():
    for k in (
        "PRSM_INFERENCE_OUTPUT_CACHE_ENABLED",
        "PRSM_INFERENCE_OUTPUT_CACHE_TTL_S",
        "PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES",
    ):
        os.environ.pop(k, None)


def teardown_function():
    for k in (
        "PRSM_INFERENCE_OUTPUT_CACHE_ENABLED",
        "PRSM_INFERENCE_OUTPUT_CACHE_TTL_S",
        "PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES",
    ):
        os.environ.pop(k, None)


# ---- make_cache_key -------------------------------------------


def test_make_cache_key_returns_sha256_hex_for_none_tier():
    from prsm.compute.inference.output_cache import make_cache_key
    key = make_cache_key("hello", "gpt2", 8, "none")
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex digest
    int(key, 16)  # parseable as hex


def test_make_cache_key_different_prompts_yield_different_keys():
    from prsm.compute.inference.output_cache import make_cache_key
    k1 = make_cache_key("hello", "gpt2", 8, "none")
    k2 = make_cache_key("goodbye", "gpt2", 8, "none")
    assert k1 != k2


def test_make_cache_key_different_models_yield_different_keys():
    from prsm.compute.inference.output_cache import make_cache_key
    k1 = make_cache_key("hi", "gpt2", 8, "none")
    k2 = make_cache_key("hi", "llama-7b", 8, "none")
    assert k1 != k2


def test_make_cache_key_different_max_tokens_yield_different_keys():
    from prsm.compute.inference.output_cache import make_cache_key
    k1 = make_cache_key("hi", "gpt2", 8, "none")
    k2 = make_cache_key("hi", "gpt2", 16, "none")
    assert k1 != k2


def test_make_cache_key_deterministic():
    from prsm.compute.inference.output_cache import make_cache_key
    k1 = make_cache_key("hi", "gpt2", 8, "none")
    k2 = make_cache_key("hi", "gpt2", 8, "none")
    assert k1 == k2


def test_make_cache_key_returns_empty_for_non_none_tiers():
    """Privacy invariant: only 'none' is cacheable. Other tiers
    require DP per-request and MUST NOT be cached."""
    from prsm.compute.inference.output_cache import make_cache_key
    for tier in ("standard", "high", "maximum"):
        assert make_cache_key("hi", "gpt2", 8, tier) == ""


# ---- should_cache ---------------------------------------------


def test_should_cache_none_true():
    from prsm.compute.inference.output_cache import should_cache
    assert should_cache("none") is True


def test_should_cache_other_tiers_false():
    from prsm.compute.inference.output_cache import should_cache
    for tier in ("standard", "high", "maximum", "weird", ""):
        assert should_cache(tier) is False


# ---- OutputCache class ----------------------------------------


def test_empty_cache_get_returns_none():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    assert c.get("anykey") is None


def test_put_get_round_trip():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.put("k1", {"output": "x"})
    result = c.get("k1")
    assert result is not None
    value, age = result
    assert value == {"output": "x"}
    assert age >= 0


def test_lru_bound_drops_oldest():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=3, ttl_seconds=3600)
    c.put("k1", "v1")
    c.put("k2", "v2")
    c.put("k3", "v3")
    c.put("k4", "v4")
    # k1 was oldest by insertion order; should be evicted
    assert c.get("k1") is None
    # k4 (newest) still present
    assert c.get("k4") is not None


def test_ttl_expiration():
    """Entry beyond ttl_seconds → get returns None even though
    it was put + never evicted."""
    from prsm.compute.inference.output_cache import OutputCache

    fake_now = [1000.0]

    def clock():
        return fake_now[0]

    c = OutputCache(
        max_entries=10, ttl_seconds=60, clock=clock,
    )
    c.put("k1", "v1")
    # Just before TTL
    fake_now[0] = 1059.0
    assert c.get("k1") is not None
    # Past TTL
    fake_now[0] = 1061.0
    assert c.get("k1") is None


def test_empty_key_never_hits():
    """No-cache sentinel: make_cache_key returns "" for
    non-none tiers. OutputCache.get("") MUST return None even
    if "" was somehow put + that should be impossible too."""
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    assert c.get("") is None
    # put("") should be a no-op (defensive)
    c.put("", "should-be-ignored")
    assert c.get("") is None


def test_count_reflects_entries():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    assert c.count() == 0
    c.put("a", 1)
    c.put("b", 2)
    assert c.count() == 2


def test_clear_drops_all():
    from prsm.compute.inference.output_cache import OutputCache
    c = OutputCache(max_entries=10, ttl_seconds=3600)
    c.put("a", 1)
    c.clear()
    assert c.count() == 0
    assert c.get("a") is None


# ---- resolve_output_cache_from_env ----------------------------


def test_resolve_disabled_by_default():
    from prsm.compute.inference.output_cache import (
        resolve_output_cache_from_env,
    )
    assert resolve_output_cache_from_env() is None


def test_resolve_enabled_returns_cache():
    from prsm.compute.inference.output_cache import (
        OutputCache, resolve_output_cache_from_env,
    )
    os.environ["PRSM_INFERENCE_OUTPUT_CACHE_ENABLED"] = "1"
    c = resolve_output_cache_from_env()
    assert isinstance(c, OutputCache)


def test_resolve_ttl_env_respected():
    from prsm.compute.inference.output_cache import (
        resolve_output_cache_from_env,
    )
    os.environ["PRSM_INFERENCE_OUTPUT_CACHE_ENABLED"] = "1"
    os.environ["PRSM_INFERENCE_OUTPUT_CACHE_TTL_S"] = "7200"
    c = resolve_output_cache_from_env()
    assert c is not None
    assert c.ttl_seconds == 7200.0


def test_resolve_max_entries_env_respected():
    from prsm.compute.inference.output_cache import (
        resolve_output_cache_from_env,
    )
    os.environ["PRSM_INFERENCE_OUTPUT_CACHE_ENABLED"] = "1"
    os.environ[
        "PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES"
    ] = "500"
    c = resolve_output_cache_from_env()
    assert c is not None
    assert c.max_entries == 500


def test_resolve_malformed_values_fall_back_to_defaults():
    from prsm.compute.inference.output_cache import (
        resolve_output_cache_from_env,
    )
    os.environ["PRSM_INFERENCE_OUTPUT_CACHE_ENABLED"] = "1"
    os.environ["PRSM_INFERENCE_OUTPUT_CACHE_TTL_S"] = "not-a-number"
    os.environ[
        "PRSM_INFERENCE_OUTPUT_CACHE_MAX_ENTRIES"
    ] = "0"  # invalid
    c = resolve_output_cache_from_env()
    # Falls back to default ttl + default max_entries (>0)
    assert c is not None
    assert c.ttl_seconds > 0
    assert c.max_entries > 0
