"""Sprint 811 — wire sprint-810 OutputCache into the executor.

Sprint 810 shipped the cache primitive. Sprint 811 ships the
two integration helpers + tests for the lookup/store path:

  ParallaxScheduledExecutor._try_output_cache(request, cost)
      → Optional[InferenceResult]
    - Returns None when cache absent, key="" (non-cacheable
      tier), or miss.
    - On hit: builds a FRESH signed receipt (new job_id, fresh
      signature) from cached output bytes + 0 epsilon + 0
      duration_seconds. Returns InferenceResult(success=True,
      output=cached, receipt=fresh-signed).

  ParallaxScheduledExecutor._put_output_cache(request, outcome)
      → None
    - Stores {output, tee_attestation} into the cache keyed on
      sprint-810 make_cache_key. No-op when cache absent or
      key="" (non-cacheable tier).

These two helpers are the testable units. Source-shape pin
tests confirm execute() calls them at the right place.

Pin tests:
- ParallaxScheduledExecutor has `_output_cache` attr; None
  default.
- _try_output_cache returns None when _output_cache is None.
- _try_output_cache returns None on cache miss.
- _try_output_cache returns InferenceResult on hit with
  cached output + fresh signed receipt (verify signature).
- _try_output_cache returns None for non-cacheable tier
  (privacy_tier != "none").
- _put_output_cache stores output + tee_attestation.
- _put_output_cache no-op when cache absent.
- _put_output_cache no-op for non-cacheable tier.
- Source-shape: execute() calls _try_output_cache BEFORE
  invoking chain_executor.
- Source-shape: execute() calls _put_output_cache AFTER
  chain executor returns (and BEFORE _build_signed_receipt
  for the non-cached path).
"""
from __future__ import annotations

import inspect
from decimal import Decimal


def _make_executor_with_identity():
    """Build a barebones ParallaxScheduledExecutor + Ed25519
    identity for signing. Same pattern as sprint 778."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    from prsm.node.identity import generate_node_identity

    identity = generate_node_identity("output-cache-test")
    exec_ = ParallaxScheduledExecutor.__new__(ParallaxScheduledExecutor)
    exec_._identity = identity
    exec_._output_cache = None
    return exec_


def _make_request(prompt="hello", privacy_tier="none", max_tokens=8):
    from prsm.compute.inference.models import (
        InferenceRequest, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel
    return InferenceRequest(
        request_id="r1",
        prompt=prompt,
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel(privacy_tier),
        budget_ftns=Decimal("1.0"),
        max_tokens=max_tokens,
    )


# ---- Attribute exists ------------------------------------------


def test_output_cache_attr_default_none():
    """ParallaxScheduledExecutor instances have _output_cache
    attribute, default None."""
    exec_ = _make_executor_with_identity()
    assert getattr(exec_, "_output_cache", "MISSING") is None


# ---- _try_output_cache ------------------------------------------


def test_try_cache_returns_none_when_cache_none():
    exec_ = _make_executor_with_identity()
    result = exec_._try_output_cache(
        request=_make_request(), cost=Decimal("0.5"),
    )
    assert result is None


def test_try_cache_returns_none_on_miss():
    from prsm.compute.inference.output_cache import OutputCache
    exec_ = _make_executor_with_identity()
    exec_._output_cache = OutputCache(
        max_entries=10, ttl_seconds=3600,
    )
    # No put — cache is empty
    result = exec_._try_output_cache(
        request=_make_request(), cost=Decimal("0.5"),
    )
    assert result is None


def test_try_cache_hit_returns_fresh_signed_receipt():
    """Cache hit → InferenceResult with cached output text + a
    FRESH signed receipt (new job_id, fresh signature that
    verifies against the same identity)."""
    from prsm.compute.inference.output_cache import (
        OutputCache, make_cache_key,
    )
    from prsm.compute.inference.receipt import verify_receipt

    exec_ = _make_executor_with_identity()
    exec_._output_cache = OutputCache(
        max_entries=10, ttl_seconds=3600,
    )
    req = _make_request(prompt="hi", max_tokens=8)
    key = make_cache_key("hi", "gpt2", 8, "none")
    assert key  # non-empty
    exec_._output_cache.put(key, {
        "output": "cached-output-text",
        "tee_attestation": b"cached-att",
    })

    result = exec_._try_output_cache(
        request=req, cost=Decimal("0.5"),
    )
    assert result is not None
    assert result.success is True
    assert result.output == "cached-output-text"
    # Fresh signed receipt — verifies under the same identity
    assert result.receipt is not None
    assert verify_receipt(
        result.receipt, identity=exec_._identity,
    )
    # job_id should be a parallax-job-* prefix (NOT the cached
    # entry's job_id from when it was originally inserted —
    # there's no original here, but the prefix is what matters)
    assert "parallax-job" in result.receipt.job_id


def test_try_cache_non_none_tier_returns_none():
    """Privacy invariant: non-none tiers MUST NOT hit cache."""
    from prsm.compute.inference.output_cache import OutputCache
    exec_ = _make_executor_with_identity()
    exec_._output_cache = OutputCache(
        max_entries=10, ttl_seconds=3600,
    )
    req = _make_request(privacy_tier="standard")
    # Even if cache has SOMETHING under any key, the
    # tier-rejection path returns None before lookup.
    result = exec_._try_output_cache(
        request=req, cost=Decimal("0.5"),
    )
    assert result is None


# ---- _put_output_cache -----------------------------------------


def _make_outcome(output_text="hello", tee_attestation=b"att"):
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    return ChainExecutionResult(
        output=output_text,
        duration_seconds=1.0,
        tee_attestation=tee_attestation,
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )


def test_put_cache_stores_output_and_attestation():
    from prsm.compute.inference.output_cache import (
        OutputCache, make_cache_key,
    )
    cache = OutputCache(max_entries=10, ttl_seconds=3600)
    exec_ = _make_executor_with_identity()
    exec_._output_cache = cache

    req = _make_request(prompt="hi", max_tokens=8)
    outcome = _make_outcome(
        output_text="hello world", tee_attestation=b"my-att",
    )
    exec_._put_output_cache(request=req, outcome=outcome)

    key = make_cache_key("hi", "gpt2", 8, "none")
    hit = cache.get(key)
    assert hit is not None
    value, _ = hit
    assert value["output"] == "hello world"
    assert value["tee_attestation"] == b"my-att"


def test_put_cache_no_op_when_cache_absent():
    """_output_cache=None → put silently does nothing."""
    exec_ = _make_executor_with_identity()
    # cache is None; just verify no exception
    exec_._put_output_cache(
        request=_make_request(), outcome=_make_outcome(),
    )


def test_put_cache_no_op_for_non_none_tier():
    """Privacy invariant: standard/high/maximum NOT stored
    even if cache is configured."""
    from prsm.compute.inference.output_cache import (
        OutputCache, make_cache_key,
    )
    cache = OutputCache(max_entries=10, ttl_seconds=3600)
    exec_ = _make_executor_with_identity()
    exec_._output_cache = cache

    req = _make_request(privacy_tier="standard")
    exec_._put_output_cache(
        request=req, outcome=_make_outcome(),
    )
    # Cache should be empty — key was "" so put was a no-op
    assert cache.count() == 0


# ---- Source-shape pins on execute() ----------------------------


def test_execute_calls_try_cache_before_chain_executor():
    """execute() must check the cache BEFORE invoking the
    expensive chain_executor.execute_chain path."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    src = inspect.getsource(ParallaxScheduledExecutor.execute)
    try_idx = src.find("_try_output_cache")
    chain_idx = src.find("_chain_executor.execute_chain")
    assert try_idx > 0, (
        "Sprint 811: execute() must call _try_output_cache"
    )
    assert chain_idx > 0
    assert try_idx < chain_idx, (
        "_try_output_cache must run BEFORE the chain executor "
        "so a cache hit skips the model forward pass"
    )


def test_execute_calls_put_cache_after_success():
    """execute() must call _put_output_cache after the chain
    executor returns successfully so subsequent identical
    prompts hit cache."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    src = inspect.getsource(ParallaxScheduledExecutor.execute)
    put_idx = src.find("_put_output_cache")
    chain_idx = src.find("_chain_executor.execute_chain")
    assert put_idx > 0
    assert chain_idx > 0
    assert chain_idx < put_idx, (
        "_put_output_cache must follow chain executor success"
    )
